#include <stdint.h>
#include <stdbool.h>
#include <dirent.h>

#include "miner.h"
#include "pptable_v1_0.h"
#include "sysfs-gpu-controls.h"


bool initialized = false;
bool has_sysfs_hwcontrols = false;
bool opt_reorder = false;
int opt_hysteresis = 3;
int opt_targettemp = 80;
int opt_overheattemp = 85;


float sysfs_gpu_vddc(int gpu) { return -1; }
int sysfs_gpu_activity(int gpu) { return -1; }

int sysfs_set_vddc(int gpu, float fVddc) { return 1; }

#ifndef __linux__

static void sysfs_init(gpu_sysfs_info *info, int gpu_idx)
{
  memset(info, 0, sizeof(gpu_sysfs_info));
  info->fd_pptable = info->fd_fan = info->fd_temp = info->fd_pwm = info->fd_sclk = info->fd_mclk = -1;
}

#else

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


static void sysfs_init(gpu_sysfs_info *info, int gpu_idx)
{
  char path[256];
  struct dirent *inner_hwmon;
  
  info->fd_pptable = info->fd_fan = info->fd_temp = info->fd_pwm = info->fd_sclk = info->fd_mclk = -1;

  snprintf(path, sizeof(path), "/sys/bus/pci/devices/0000:%.2x:%.2x.%.1x/", 
    info->pcie_index[0], info->pcie_index[1], info->pcie_index[2]);
  size_t len = strlen(path);

  snprintf(path + len, sizeof(path) - len, "hwmon");
  DIR *hwmon = opendir(path); 
  if (hwmon == NULL) {
    applog(LOG_DEBUG, "Failed to open hwmon directory %s for GPU%d", path, gpu_idx);
    snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/hwmon", gpu_idx);
    len = strlen(path) - 5;
    hwmon = opendir(path);
    if (hwmon == NULL) {
      applog(LOG_DEBUG, "Failed to open hwmon directory %s for GPU%d", path, gpu_idx);
      return;
    }
  }
  
  snprintf(path + len, sizeof(path) - len, "pp_table");
  int fd = open(path, O_RDONLY | O_RSYNC);
  bool success = false;
  if (fd != -1) {
    int size = lseek(fd, 0, SEEK_END);
    if (size > 0) {
      lseek(fd, 0, SEEK_SET);
      info->pptable = (uint8_t*) malloc(size);
      info->default_pptable = (uint8_t*) malloc(size);
      info->pptable_size = read(fd, info->pptable, size);
      memcpy(info->default_pptable, info->pptable, info->pptable_size);
      info->fd_pptable = open(path, O_WRONLY | O_DSYNC);
      success = (info->fd_pptable != -1 && info->pptable_size > 0);
    }
    close(fd);
  }
  if (!success) {
    if (info->fd_pptable != -1) {
      close(info->fd_pptable);
      info->fd_pptable = -1;
    }
    applog(LOG_DEBUG, "Failed to open/read %s", path);
  }
  else
    pthread_mutex_init(&info->rw_lock, NULL);

  snprintf(path + len, sizeof(path) - len, "pp_dpm_sclk");
  info->fd_sclk = open(path, O_RDONLY | O_RSYNC);
  if (info->fd_sclk == -1)
    applog(LOG_DEBUG, "Failed to open %s", path);

  snprintf(path + len, sizeof(path) - len, "pp_dpm_mclk");
  info->fd_mclk = open(path, O_RDONLY | O_RSYNC);
  if (info->fd_mclk == -1)
    applog(LOG_DEBUG, "Failed to open %s", path);
 

  while (true) {
    inner_hwmon = readdir(hwmon);
    if (inner_hwmon == NULL) {
      closedir(hwmon);
      applog(LOG_DEBUG, "Failed to parse hwmon directory for GPU%d", gpu_idx);
      return;
    }
    if (inner_hwmon->d_type != DT_DIR)
      continue;
    if (!memcmp(inner_hwmon->d_name, "hwmon", 5))
      break;
  }
  
  snprintf(path + len, sizeof(path) - len, "hwmon/%s/", inner_hwmon->d_name);
  len = strlen(path);
  closedir(hwmon);

  snprintf(path + len, sizeof(path) - len, "pwm1");
  info->fd_fan = open(path, O_RDWR | O_RSYNC | O_DSYNC);
  if (info->fd_fan == -1)
    applog(LOG_DEBUG, "Failed to open %s", path);
  else
    pthread_mutex_init(&info->rw_lock, NULL);

  char buf[17] = {0};
  snprintf(path + len, sizeof(path) - len, "pwm1_min");
  fd = open(path, O_RDONLY | O_RSYNC);
  info->min_fanspeed = 0;
  if (fd != -1) {
    int bytes_read = read(fd, &buf, sizeof(buf) - 1);
    if (bytes_read > 0)
      info->min_fanspeed = strtoul(buf, NULL, 10);
    else
      applog(LOG_DEBUG, "Failed to read %s", path);
    close(fd);
  }
  else
    applog(LOG_DEBUG, "Failed to open %s", path);

  snprintf(path + len, sizeof(path) - len, "pwm1_max");
  fd = open(path, O_RDONLY | O_RSYNC);
  info->max_fanspeed = 255;
  if (fd != -1) {
    int bytes_read = read(fd, &buf, sizeof(buf) - 1);
    if (bytes_read > 0) {
      buf[bytes_read] = '\0';
      info->max_fanspeed = strtoul(buf, NULL, 10);
    }
    else
      applog(LOG_DEBUG, "Failed to read %s", path);
    close(fd);
  }
  else
    applog(LOG_DEBUG, "Failed to open %s", path);

  snprintf(path + len, sizeof(path) - len, "temp1_input");
  info->fd_temp = open(path, O_RDONLY | O_RSYNC);
  if (info->fd_temp == -1)
    applog(LOG_DEBUG, "Failed to open %s", path);

  snprintf(path + len, sizeof(path) - len, "fan1_input");
  info->fd_pwm = open(path, O_RDONLY | O_RSYNC);
  if (info->fd_pwm == -1)
    applog(LOG_DEBUG, "Failed to open %s", path);
}

#endif


static int parse_dpm_clk_table(int fd)
{
  char buf[1024];
  int ret = -1;
#ifdef __linux__
  if (has_sysfs_hwcontrols && fd != -1) {
    lseek(fd, 0, SEEK_SET);
    size_t len = read(fd, buf, sizeof(buf) - 1);
    if (len > 0) {
      buf[len] = '\0';
      char *ptr = strtok(buf, "\n");
      while (ptr != NULL) {
        int index;
        int freq;
        char active_str[3] = {0};
        sscanf(ptr, "%d: %dMhz%s", &index, &freq, active_str);
        if (active_str[0] == '*' || active_str[1] == '*') {
          ret = freq;
          break;
        }
        ptr = strtok(NULL, "\n");
      }
    }
  }
#endif
  return ret;
}

int sysfs_gpu_engineclock(int gpu)
{
  return parse_dpm_clk_table(gpus[gpu].sysfs_info.fd_sclk);
}

int sysfs_gpu_memclock(int gpu)
{
  return parse_dpm_clk_table(gpus[gpu].sysfs_info.fd_mclk);
}

float sysfs_gpu_temp(int gpu)
{
  gpu_sysfs_info *info = &gpus[gpu].sysfs_info;
  char temp_str[17];
  float ret = -1.f;
#ifdef __linux__
  if (has_sysfs_hwcontrols && info->fd_temp != -1) {
    lseek(info->fd_temp, 0, SEEK_SET);
    int bytes_read = read(info->fd_temp, &temp_str, sizeof(temp_str) - 1);
    if (bytes_read > 0) {
      temp_str[bytes_read] = '\0';
      ret = strtol(temp_str, NULL, 10) / 1000.f;
    }
  }
#endif
  return ret;
}

static int fanpercent_to_speed(gpu_sysfs_info *info, float fanpercent)
{
  int min = info->min_fanspeed;
  int max = info->max_fanspeed;

  float range = max - min;
  float speed = fanpercent / 100.f * range + min + 0.5f;
  return speed;
}

static float fanspeed_to_percent(gpu_sysfs_info *info, int speed)
{
  int min = info->min_fanspeed;
  int max = info->max_fanspeed;

  float range = max - min;
  float fanpercent = (speed - min) / range * 100.f;
  return fanpercent;
}

int sysfs_gpu_fanspeed(int gpu)
{
  gpu_sysfs_info *info = &gpus[gpu].sysfs_info;
  char speed_str[17];
  int ret = -1;
#ifdef __linux__
  if (has_sysfs_hwcontrols && info->fd_pwm != -1) {
    lseek(info->fd_pwm, 0, SEEK_SET);
    int bytes_read = read(info->fd_pwm, speed_str, sizeof(speed_str) - 1);
    if (bytes_read > 0) {
      speed_str[bytes_read] = '\0';
      ret = strtoul(speed_str, NULL, 10);
    }
  }
#endif
  return ret;
}

float sysfs_gpu_fanpercent(int gpu)
{
  gpu_sysfs_info *info = &gpus[gpu].sysfs_info;
  char speed_str[17];
  float ret = -1.f;
#ifdef __linux__
  if (has_sysfs_hwcontrols && info->fd_fan != -1) {
    pthread_mutex_lock(&info->rw_lock);
    lseek(info->fd_fan, 0, SEEK_SET);
    int bytes_read = read(info->fd_fan, speed_str, sizeof(speed_str) - 1);
    pthread_mutex_unlock(&info->rw_lock);
    if (bytes_read > 0) {
      speed_str[bytes_read] = '\0';
      unsigned long speed = strtoul(speed_str, NULL, 10);
      ret = fanspeed_to_percent(info, speed);
    }
  }
#endif
  return ret;
}


static int __set_fanspeed(gpu_sysfs_info *info, float fanpercent)
{
  char speed_str[17];
  int speed = fanpercent_to_speed(info, fanpercent);
  int ret = 1;
  snprintf(speed_str, sizeof(speed_str), "%d", speed);
#ifdef __linux__
  if (info->fd_fan != -1 && fanpercent >= 0.f) {
    lseek(info->fd_fan, 0, SEEK_SET);
    ret = (write(info->fd_fan, speed_str, strlen(speed_str)) <= 0);
    if (!ret)
      info->target_fanpercent = fanpercent;
  }
#endif
  return ret;
}

int sysfs_set_fanspeed(int gpu, float fanpercent)
{
  int ret = 1;
  if (has_sysfs_hwcontrols && gpus[gpu].has_sysfs_hwcontrols) {
    gpu_sysfs_info *info = &gpus[gpu].sysfs_info;
    applog(LOG_DEBUG, "GPU%d: set fanpercent to %.3g%%", gpu, fanpercent);
    pthread_mutex_lock(&info->rw_lock);
    ret = __set_fanspeed(info, fanpercent);
    pthread_mutex_unlock(&info->rw_lock);
  }
  return ret;
}


static Tonga_State_Array *get_state_array(uint8_t *pptable)
{
  Tonga_POWERPLAYTABLE *header = (Tonga_POWERPLAYTABLE*) pptable;
  return (Tonga_State_Array*) (pptable + header->usStateArrayOffset);
}

static Tonga_MCLK_Dependency_Table *get_mclk_table(uint8_t *pptable)
{
  Tonga_POWERPLAYTABLE *header = (Tonga_POWERPLAYTABLE*) pptable;
  return (Tonga_MCLK_Dependency_Table*) (pptable + header->usMclkDependencyTableOffset);
}

static Tonga_SCLK_Dependency_Table *get_sclk_table(uint8_t *pptable)
{
  Tonga_POWERPLAYTABLE *header = (Tonga_POWERPLAYTABLE*) pptable;
  return (Tonga_SCLK_Dependency_Table*) (pptable + header->usSclkDependencyTableOffset);
}

static Tonga_SCLK_Dependency_Record *get_sclk_record(Tonga_SCLK_Dependency_Table *sclk_tbl, int idx)
{
  Tonga_SCLK_Dependency_Record *record = NULL; 
  if (/*idx < sclk_tbl->ucNumEntries &&*/ sclk_tbl->ucRevId <= 1) {
    int entry_size = (sclk_tbl->ucRevId == 0 ? sizeof(Tonga_SCLK_Dependency_Record) : sizeof(Polaris_SCLK_Dependency_Record));
    record = (Tonga_SCLK_Dependency_Record*) ((uint8_t*) sclk_tbl->entries + idx * entry_size);
  }
  return record;
}

static int __apply_pptable(gpu_sysfs_info *info)
{
#ifdef __linux__
  lseek(info->fd_pptable, 0, SEEK_SET);
  int bytes_written = write(info->fd_pptable, info->pptable, info->pptable_size);
  if (opt_autofan && info->target_fanpercent >= 0)
    __set_fanspeed(info, info->target_fanpercent); // restore fan settings
  return (bytes_written != info->pptable_size);
#else
  return 0;
#endif
}


int sysfs_set_powertune(int gpu, int power_limit) 
{
  gpu_sysfs_info *info = &gpus[gpu].sysfs_info;
  int ret = 1;
  if (has_sysfs_hwcontrols && power_limit > 0 && info->fd_pptable != -1) {
    applog(LOG_DEBUG, "GPU%d: set power limit to %dW", gpu, power_limit);
    pthread_mutex_lock(&info->rw_lock);
    Tonga_POWERPLAYTABLE *header = (Tonga_POWERPLAYTABLE*) info->pptable;
    Tonga_PowerTune_Table *power_tbl = (Tonga_PowerTune_Table*) (info->pptable + header->usPowerTuneTableOffset);
    Tonga_PowerTune_Table *default_power_tbl = (Tonga_PowerTune_Table*) (info->default_pptable + header->usPowerTuneTableOffset);
    power_tbl->usTDC = (float) default_power_tbl->usTDC * power_limit / default_power_tbl->usMaximumPowerDeliveryLimit;
    power_tbl->usMaximumPowerDeliveryLimit = power_limit;
    ret = __apply_pptable(info);
    pthread_mutex_unlock(&info->rw_lock);
  }
  return ret;
}

static int gpu_powertune(int gpu)
{
  int ret = 0;
  gpu_sysfs_info *info = &gpus[gpu].sysfs_info;
  if (has_sysfs_hwcontrols && info->fd_pptable != -1) {
    Tonga_POWERPLAYTABLE *header = (Tonga_POWERPLAYTABLE*) info->pptable;
    Tonga_PowerTune_Table *power_tbl = (Tonga_PowerTune_Table*) (info->pptable + header->usPowerTuneTableOffset);
    ret = power_tbl->usMaximumPowerDeliveryLimit;
  }
  return ret;
}


static bool __set_engineclock(struct cgpu_info *cgpu, int iEngineClock)
{
  gpu_sysfs_info *info = &cgpu->sysfs_info;
  int engineclock = info->engineclock;

  if (iEngineClock <= 0)
    return false;
  if (cgpu->min_engine > 0)
    iEngineClock = MAX(iEngineClock, cgpu->min_engine);
  if (cgpu->gpu_engine > 0)
    iEngineClock = MIN(iEngineClock, cgpu->gpu_engine);
  applog(LOG_DEBUG, "GPU%d: set engineclock to %dMHz", cgpu->device_id, iEngineClock);
  iEngineClock *= 100;

  Tonga_SCLK_Dependency_Table *sclk_tbl = get_sclk_table(info->pptable);
  //replace current sclk table with default table
  size_t sclk_tbl_size = (uintptr_t) get_sclk_record(sclk_tbl, sclk_tbl->ucNumEntries) - (uintptr_t) sclk_tbl;
  memcpy(sclk_tbl, get_sclk_table(info->default_pptable), sclk_tbl_size);

  //parse pp_table
  Tonga_SCLK_Dependency_Record *prev_record = NULL, *record;
  int idx = sclk_tbl->ucNumEntries - 1;
  for (; idx >= 0; idx--) {
    record = get_sclk_record(sclk_tbl, idx);
    if (iEngineClock > record->ulSclk)
      break;
    prev_record = record;
  }
  if (prev_record != NULL)
    record = prev_record;

  //set new values
  if (idx >= 0) {
    record->ulSclk = iEngineClock;
    info->engineclock = iEngineClock / 100;
  }
  else
    info->engineclock = record->ulSclk / 100;

  Tonga_State_Array *state_array = get_state_array(info->pptable);
  idx = MIN(idx + 1, sclk_tbl->ucNumEntries - 1);
  state_array->entries[state_array->ucNumEntries - 1].ucEngineClockIndexHigh = idx;
  info->sclk_ind = idx;

  return (engineclock != info->engineclock);
}

int sysfs_set_engineclock(int gpu, int iEngineClock)
{
  struct cgpu_info *cgpu = &gpus[gpu];
  gpu_sysfs_info *info = &cgpu->sysfs_info;
  int ret = !has_sysfs_hwcontrols || info->fd_pptable == -1;
  if (!ret) {
    pthread_mutex_lock(&info->rw_lock);
    bool updated = __set_engineclock(cgpu, iEngineClock);
    if (updated)
      ret = __apply_pptable(info);
    pthread_mutex_unlock(&info->rw_lock);
  }
  return ret;
}

static bool __change_engineclock(struct cgpu_info *cgpu, bool increase)
{
  gpu_sysfs_info *info = &cgpu->sysfs_info;
  
  Tonga_SCLK_Dependency_Table *sclk_tbl = get_sclk_table(info->default_pptable);
  Tonga_SCLK_Dependency_Record *record = get_sclk_record(sclk_tbl, info->sclk_ind);
  bool updated = false; 
  if (increase) {
    //check whether the power state can be increased
    if (info->sclk_ind + 1 < sclk_tbl->ucNumEntries) {
      record = get_sclk_record(sclk_tbl, ++info->sclk_ind);
      updated = __set_engineclock(cgpu, record->ulSclk / 100);
    }
    else if (cgpu->gpu_engine > 0)
      updated = __set_engineclock(cgpu, cgpu->gpu_engine);
  }
  else {
    //check whether the power state can be decreased
    if (info->sclk_ind - 1 >= 0) {
      record = get_sclk_record(sclk_tbl, --info->sclk_ind);
      updated = __set_engineclock(cgpu, record->ulSclk / 100);
    }
    else if (cgpu->min_engine > 0)
      updated = __set_engineclock(cgpu, cgpu->min_engine);
  }
  return updated;
}


static bool __set_memoryclock(struct cgpu_info *cgpu, int iMemoryClock)
{
  if (iMemoryClock <= 0)
    return false;
  gpu_sysfs_info *info = &cgpu->sysfs_info;
  applog(LOG_DEBUG, "GPU%d: set memoryclock %dMHz", cgpu->device_id, iMemoryClock);
  int memclock = info->memclock;

  Tonga_MCLK_Dependency_Table* mclk_tbl = get_mclk_table(info->pptable);
  Tonga_MCLK_Dependency_Record *record = &mclk_tbl->entries[mclk_tbl->ucNumEntries - 1];
  record->ulMclk = iMemoryClock * 100;
  info->memclock = iMemoryClock;

  return (memclock != info->memclock);
}

int sysfs_set_memoryclock(int gpu, int iMemoryClock)
{
  struct cgpu_info *cgpu = &gpus[gpu];
  gpu_sysfs_info *info = &cgpu->sysfs_info;
  int ret = !has_sysfs_hwcontrols || info->fd_pptable == -1;
  if (!ret) {
    pthread_mutex_lock(&info->rw_lock);
    bool updated = __set_memoryclock(cgpu, iMemoryClock);
    if (updated)
      ret = __apply_pptable(info);
    pthread_mutex_unlock(&info->rw_lock);
  }
  return ret;
}

/* Returns whether the fanspeed is optimal already or not. The fan_window bool
 * tells us whether the current fanspeed is in the target range for fanspeeds.
 */
static bool sysfs_fan_autotune(int gpu, int temp, float fanpercent, int lasttemp, bool *fan_window)
{
  struct cgpu_info *cgpu = &gpus[gpu];
  gpu_sysfs_info *info = &gpus[gpu].sysfs_info;
  int tdiff = temp - lasttemp;
  int top = gpus[gpu].gpu_fan;
  int bot = gpus[gpu].min_fan;
  float newpercent = info->target_fanpercent;//fanpercent;
  float iMin = 0, iMax = 100;
    
  if (!opt_autoengine && temp > info->overheat_temp && fanpercent < iMax) {
    applog(LOG_WARNING, "Overheat detected on GPU %d, increasing fan to 100%% (temp was %d, overtemp is %d)\n", gpu, temp, info->overheat_temp);
    newpercent = iMax;

    dev_error(cgpu, REASON_DEV_OVER_HEAT);
  }
  else if (temp > info->target_temp && fanpercent < top && tdiff >= 0) {
    applog(LOG_DEBUG, "Temperature over target, increasing fanspeed");
    if (temp > info->target_temp + opt_hysteresis)
      newpercent = info->target_fanpercent + 10;
    else
      newpercent = info->target_fanpercent + 5;
    
    if (newpercent > top)
      newpercent = top;
  }
  else if (fanpercent > bot && temp < info->target_temp - opt_hysteresis) {
    /* Detect large swings of 5 degrees or more and change fan by
     * a proportion more */
    if (tdiff <= 0) {
      applog(LOG_DEBUG, "Temperature %d degrees below target, decreasing fanspeed", opt_hysteresis);
      newpercent = info->target_fanpercent - 1 + tdiff / 5;
    }
    else if (tdiff >= 5) {
      applog(LOG_DEBUG, "Temperature climbed %d while below target, increasing fanspeed", tdiff);
      newpercent = info->target_fanpercent + tdiff / 5;
    }
  }
  else {
    /* We're in the optimal range, make minor adjustments if the
     * temp is still drifting */
    if (fanpercent > bot && tdiff < 0 && lasttemp < info->target_temp) {
      applog(LOG_DEBUG, "Temperature dropping while in target range, decreasing fanspeed");
      newpercent = info->target_fanpercent + tdiff;
    }
    else if (fanpercent < top && tdiff > 0 && temp > info->target_temp - opt_hysteresis) {
      applog(LOG_DEBUG, "Temperature rising while in target range, increasing fanspeed");
      newpercent = info->target_fanpercent + tdiff;
    }
  }

  if (newpercent > iMax)
    newpercent = iMax;
  else if (newpercent < iMin)
    newpercent = iMin;

  if (newpercent < top)
    *fan_window = true;
  else
    *fan_window = false;

  if (newpercent != fanpercent) {
    applog(LOG_INFO, "Setting GPU %d fan percentage to %g", gpu, newpercent);
    
    set_fanspeed(gpu, newpercent);
    
    /* If the fanspeed is going down and we're below the top speed,
     * consider the fan optimal to prevent minute changes in
     * fanspeed delaying GPU engine speed changes */
    if (newpercent < fanpercent && *fan_window)
      return true;
    
    return false;
  }
  return true;
}

void sysfs_gpu_autotune(int gpu, enum dev_enable *denable)
{
  struct cgpu_info *cgpu = &gpus[gpu];
  if (!has_sysfs_hwcontrols || !cgpu->has_sysfs_hwcontrols)
    return;

  gpu_sysfs_info *info = &cgpu->sysfs_info;
  bool fan_window = true;
  int temp = sysfs_gpu_temp(gpu);
  int fanpercent = sysfs_gpu_fanpercent(gpu) + 0.5f;
  
  if (temp && fanpercent >= 0 && opt_autofan)
    sysfs_fan_autotune(gpu, temp, info->target_fanpercent, info->last_temp, &fan_window);
  info->last_temp = temp;

  uint32_t ctr_diff = info->ctr++ - info->last_ctr;
  if (opt_autoengine && info->fd_pptable != 1) {
    bool updated = false;
    pthread_mutex_lock(&info->rw_lock);
    if (temp > cgpu->cutofftemp && *denable == DEV_ENABLED) {
      applog(LOG_WARNING, "Hit thermal cutoff limit on GPU %d, disabling!", gpu);
      updated = __set_engineclock(cgpu, cgpu->min_engine);
      *denable = DEV_RECOVER;
      dev_error(cgpu, REASON_DEV_THERMAL_CUTOFF);
    }
    else if (temp > info->overheat_temp && *denable == DEV_ENABLED) {
      applog(LOG_WARNING, "Overheat detected, decreasing GPU %d clock speed", gpu);
      updated = __change_engineclock(cgpu, false);
      dev_error(cgpu, REASON_DEV_OVER_HEAT);
      /* Only try to tune engine speed up if this GPU is not disabled */
    }
    else if (temp < info->overheat_temp - opt_hysteresis && fan_window && ctr_diff >= 6 && *denable == DEV_ENABLED) {
      applog(LOG_DEBUG, "Temperature below overheat, increasing clock speed");
      updated = __change_engineclock(cgpu, true);
      info->last_ctr = info->ctr;
    }
    else if (temp < info->target_temp && *denable == DEV_RECOVER && opt_restart) {
      applog(LOG_NOTICE, "Device recovered to temperature below target, re-enabling");
      *denable = DEV_ENABLED;
      for (int i = 0; i < cgpu->threads; i++)
        cgsem_post(&cgpu->thr[i]->sem);
    }
    if (updated)
      __apply_pptable(info);
    pthread_mutex_unlock(&info->rw_lock);
  }
}

bool sysfs_gpu_stats(int gpu, float *temp, int *engineclock, int *memclock, float *vddc,
         int *activity, int *fanspeed, int *fanpercent, int *powertune)
{
  if (!has_sysfs_hwcontrols || !gpus[gpu].has_sysfs_hwcontrols)
    return false;

  *temp = gpu_temp(gpu);
  *fanspeed = gpu_fanspeed(gpu);
  *fanpercent = gpu_fanpercent(gpu) + 0.5f;
  *engineclock = gpu_engineclock(gpu);
  *memclock = gpu_memclock(gpu);
  *vddc = 0;
  *activity = 0;
  *powertune = gpu_powertune(gpu);

  return true;
}

void sysfs_cleanup(int nDevs)
{
#ifdef __linux__
  if (!has_sysfs_hwcontrols)
    return;

  has_sysfs_hwcontrols = false;
  for (int i = 0; i < nDevs; i++) {
    gpus[i].has_sysfs_hwcontrols = false;
    gpu_sysfs_info *info = &gpus[i].sysfs_info;
    // only overwrite pptable to default -> fd and memory are not released
    if (info->pptable != NULL && info->fd_pptable != -1) {
      pthread_mutex_lock(&info->rw_lock);
      memcpy(info->pptable, info->default_pptable, info->pptable_size);
      lseek(info->fd_pptable, 0, SEEK_SET);
      write(info->fd_pptable, info->pptable, info->pptable_size);
      pthread_mutex_unlock(&info->rw_lock);
      sync();
    }
  }
#endif
}

bool init_sysfs_hwcontrols(int nDevs)
{
  gpu_temp = &sysfs_gpu_temp;
  gpu_engineclock = &sysfs_gpu_engineclock;
  gpu_memclock = &sysfs_gpu_memclock;
  gpu_vddc = &sysfs_gpu_vddc;
  gpu_activity = &sysfs_gpu_activity;
  gpu_fanspeed = &sysfs_gpu_fanspeed;
  gpu_fanpercent = &sysfs_gpu_fanpercent;
  set_powertune = &sysfs_set_powertune;
  set_vddc = &sysfs_set_vddc;
  set_fanspeed = &sysfs_set_fanspeed;
  set_engineclock = &sysfs_set_engineclock;
  set_memoryclock = &sysfs_set_memoryclock;
  gpu_stats = &sysfs_gpu_stats;
  gpu_autotune = &sysfs_gpu_autotune;

  extern bool opt_noadl;
  if (opt_noadl) {
    extern bool adl_active;
    adl_active = false;
    has_sysfs_hwcontrols = false;
    for (int i = 0; i < nDevs; i++) {
      gpus[i].has_adl = false;
      gpus[i].has_sysfs_hwcontrols = false;
    }
    return false;
  }

  for (int i = 0; i < nDevs; ++i) {
    gpu_sysfs_info *info = &gpus[i].sysfs_info;
    if (!initialized)
      sysfs_init(info, i);
   
    info->target_fanpercent = sysfs_gpu_fanpercent(i);
    info->last_temp = sysfs_gpu_temp(i);
    if (!info->overheat_temp)
      info->overheat_temp = opt_overheattemp;
    if (!info->target_temp)
      info->target_temp = opt_targettemp;
    
    gpus[i].has_sysfs_hwcontrols = (info->fd_temp != -1) & (info->fd_fan != -1);
    Tonga_POWERPLAYTABLE *header = (Tonga_POWERPLAYTABLE*) info->pptable;
    COMMON_TABLE_HEADER *common_hdr = (COMMON_TABLE_HEADER*) &header->sHeader;
    //enable pptable support only for Tonga, Polaris and Fiji (?)
    if (info->fd_pptable != -1 &&
        (common_hdr->ucTableFormatRevision != 7 || common_hdr->ucTableContentRevision != 1 || header->ucTableRevision != 0)) {
      close(info->fd_pptable);
      info->fd_pptable = -1;
      applog(LOG_WARNING, "No sysfs pptable support for GPU%d (%s)", i, gpus[i].name);
    }
     
    if (gpus[i].has_sysfs_hwcontrols)
      has_sysfs_hwcontrols = true;
  }
  initialized = true;
  
  return has_sysfs_hwcontrols;  
}

