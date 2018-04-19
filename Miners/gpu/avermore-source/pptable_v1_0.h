/*
 * Copyright 2015 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef TONGA_PPTABLE_H
#define TONGA_PPTABLE_H

/** \file
 * This is a PowerPlay table header file
 */
#pragma pack(push, 1)

#define TONGA_PP_FANPARAMETERS_TACHOMETER_PULSES_PER_REVOLUTION_MASK 0x0f
#define TONGA_PP_FANPARAMETERS_NOFAN                                 0x80    /* No fan is connected to this controller. */

#define TONGA_PP_THERMALCONTROLLER_NONE      0
#define TONGA_PP_THERMALCONTROLLER_LM96163   17
#define TONGA_PP_THERMALCONTROLLER_TONGA     21
#define TONGA_PP_THERMALCONTROLLER_FIJI      22

/*
 * Thermal controller 'combo type' to use an external controller for Fan control and an internal controller for thermal.
 * We probably should reserve the bit 0x80 for this use.
 * To keep the number of these types low we should also use the same code for all ASICs (i.e. do not distinguish RV6xx and RV7xx Internal here).
 * The driver can pick the correct internal controller based on the ASIC.
 */

#define TONGA_PP_THERMALCONTROLLER_ADT7473_WITH_INTERNAL   0x89    /* ADT7473 Fan Control + Internal Thermal Controller */
#define TONGA_PP_THERMALCONTROLLER_EMC2103_WITH_INTERNAL   0x8D    /* EMC2103 Fan Control + Internal Thermal Controller */

/* TONGA_POWERPLAYTABLE::ulPlatformCaps */
#define TONGA_PP_PLATFORM_CAP_VDDGFX_CONTROL              0x1            /* This cap indicates whether vddgfx will be a separated power rail. */
#define TONGA_PP_PLATFORM_CAP_POWERPLAY                   0x2            /* This cap indicates whether this is a mobile part and CCC need to show Powerplay page. */
#define TONGA_PP_PLATFORM_CAP_SBIOSPOWERSOURCE            0x4            /* This cap indicates whether power source notificaiton is done by SBIOS directly. */
#define TONGA_PP_PLATFORM_CAP_DISABLE_VOLTAGE_ISLAND      0x8            /* Enable the option to overwrite voltage island feature to be disabled, regardless of VddGfx power rail support. */
#define ____RETIRE16____                                0x10
#define TONGA_PP_PLATFORM_CAP_HARDWAREDC                 0x20            /* This cap indicates whether power source notificaiton is done by GPIO directly. */
#define ____RETIRE64____                                0x40
#define ____RETIRE128____                               0x80
#define ____RETIRE256____                              0x100
#define ____RETIRE512____                              0x200
#define ____RETIRE1024____                             0x400
#define ____RETIRE2048____                             0x800
#define TONGA_PP_PLATFORM_CAP_MVDD_CONTROL             0x1000            /* This cap indicates dynamic MVDD is required. Uncheck to disable it. */
#define ____RETIRE2000____                            0x2000
#define ____RETIRE4000____                            0x4000
#define TONGA_PP_PLATFORM_CAP_VDDCI_CONTROL            0x8000            /* This cap indicates dynamic VDDCI is required. Uncheck to disable it. */
#define ____RETIRE10000____                          0x10000
#define TONGA_PP_PLATFORM_CAP_BACO                    0x20000            /* Enable to indicate the driver supports BACO state. */

#define TONGA_PP_PLATFORM_CAP_OUTPUT_THERMAL2GPIO17         0x100000     /* Enable to indicate the driver supports thermal2GPIO17. */
#define TONGA_PP_PLATFORM_COMBINE_PCC_WITH_THERMAL_SIGNAL  0x1000000     /* Enable to indicate if thermal and PCC are sharing the same GPIO */
#define TONGA_PLATFORM_LOAD_POST_PRODUCTION_FIRMWARE       0x2000000

/* PPLIB_NONCLOCK_INFO::usClassification */
#define PPLIB_CLASSIFICATION_UI_MASK               0x0007
#define PPLIB_CLASSIFICATION_UI_SHIFT              0
#define PPLIB_CLASSIFICATION_UI_NONE               0
#define PPLIB_CLASSIFICATION_UI_BATTERY            1
#define PPLIB_CLASSIFICATION_UI_BALANCED           3
#define PPLIB_CLASSIFICATION_UI_PERFORMANCE        5
/* 2, 4, 6, 7 are reserved */

#define PPLIB_CLASSIFICATION_BOOT                  0x0008
#define PPLIB_CLASSIFICATION_THERMAL               0x0010
#define PPLIB_CLASSIFICATION_LIMITEDPOWERSOURCE    0x0020
#define PPLIB_CLASSIFICATION_REST                  0x0040
#define PPLIB_CLASSIFICATION_FORCED                0x0080
#define PPLIB_CLASSIFICATION_ACPI                  0x1000

/* PPLIB_NONCLOCK_INFO::usClassification2 */
#define PPLIB_CLASSIFICATION2_LIMITEDPOWERSOURCE_2 0x0001

#define Tonga_DISALLOW_ON_DC                       0x00004000
#define Tonga_ENABLE_VARIBRIGHT                    0x00008000

#define Tonga_TABLE_REVISION_TONGA                 7

//vdd = vddc + vdd_offset - (vdd_offset >> 15 ? 0xffff : 0)


typedef struct _COMMON_TABLE_HEADER
{
  uint16_t usStructureSize;
  uint8_t  ucTableFormatRevision;   //Change it when the Parser is not backward compatible
  uint8_t  ucTableContentRevision;  //Change it only when the table needs to change but the firmware
                                  //Image can't be updated, while Driver needs to carry the new table!
} COMMON_TABLE_HEADER;

typedef struct _Tonga_POWERPLAYTABLE {
  COMMON_TABLE_HEADER sHeader;

  uint8_t  ucTableRevision;
  uint16_t usTableSize;            /*the size of header structure */

  uint32_t  ulGoldenPPID;
  uint32_t  ulGoldenRevision;
  uint16_t  usFormatID;

  uint16_t  usVoltageTime;           /*in microseconds */
  uint32_t  ulPlatformCaps;            /*See Tonga_CAPS_* */

  uint32_t  ulMaxODEngineClock;          /*For Overdrive.  */
  uint32_t  ulMaxODMemoryClock;          /*For Overdrive. */

  uint16_t  usPowerControlLimit;
  uint16_t  usUlvVoltageOffset;          /*in mv units */

  uint16_t  usStateArrayOffset;          /*points to Tonga_State_Array */
  uint16_t  usFanTableOffset;          /*points to Tonga_Fan_Table */
  uint16_t  usThermalControllerOffset;       /*points to Tonga_Thermal_Controller */
  uint16_t  usReserv;               /*CustomThermalPolicy removed for Tonga. Keep this filed as reserved. */

  uint16_t  usMclkDependencyTableOffset;     /*points to Tonga_MCLK_Dependency_Table */
  uint16_t  usSclkDependencyTableOffset;     /*points to Tonga_SCLK_Dependency_Table */
  uint16_t  usVddcLookupTableOffset;       /*points to Tonga_Voltage_Lookup_Table */
  uint16_t  usVddgfxLookupTableOffset;     /*points to Tonga_Voltage_Lookup_Table */

  uint16_t  usMMDependencyTableOffset;      /*points to Tonga_MM_Dependency_Table */

  uint16_t  usVCEStateTableOffset;         /*points to Tonga_VCE_State_Table; */

  uint16_t  usPPMTableOffset;          /*points to Tonga_PPM_Table */
  uint16_t  usPowerTuneTableOffset;        /*points to PowerTune_Table */

  uint16_t  usHardLimitTableOffset;        /*points to Tonga_Hard_Limit_Table */

  uint16_t  usPCIETableOffset;          /*points to Tonga_PCIE_Table */

  uint16_t  usGPIOTableOffset;          /*points to Tonga_GPIO_Table */

  uint16_t  usReserved[6];             /*TODO: modify reserved size to fit structure aligning */
} Tonga_POWERPLAYTABLE;

typedef struct _Tonga_State {
  uint8_t  ucEngineClockIndexHigh;
  uint8_t  ucEngineClockIndexLow;

  uint8_t  ucMemoryClockIndexHigh;
  uint8_t  ucMemoryClockIndexLow;

  uint8_t  ucPCIEGenLow;
  uint8_t  ucPCIEGenHigh;

  uint8_t  ucPCIELaneLow;
  uint8_t  ucPCIELaneHigh;

  uint16_t usClassification;
  uint32_t ulCapsAndSettings;
  uint16_t usClassification2;
  uint8_t  ucUnused[4];
} Tonga_State;

typedef struct _Tonga_State_Array {
  uint8_t ucRevId;
  uint8_t ucNumEntries;    /* Number of entries. */
  Tonga_State entries[1];  /* Dynamically allocate entries. */
} Tonga_State_Array;

typedef struct _Tonga_MCLK_Dependency_Record {
  uint8_t  ucVddcInd;  /* Vddc voltage */
  uint16_t usVddci;
  uint16_t usVddgfxOffset;  /* Offset relative to Vddc voltage */
  uint16_t usMvdd;
  uint32_t ulMclk;
  uint16_t usReserved;
} Tonga_MCLK_Dependency_Record;

typedef struct _Tonga_MCLK_Dependency_Table {
  uint8_t ucRevId;
  uint8_t ucNumEntries;                     /* Number of entries. */
  Tonga_MCLK_Dependency_Record entries[1];        /* Dynamically allocate entries. */
} Tonga_MCLK_Dependency_Table;

typedef struct _Tonga_SCLK_Dependency_Record {
  uint8_t  ucVddInd;                      /* Base voltage */
  uint16_t usVddcOffset;                    /* Offset relative to base voltage */
  uint32_t ulSclk;
  uint16_t usEdcCurrent;
  uint8_t  ucReliabilityTemperature;
  uint8_t  ucCKSVOffsetandDisable;                /* Bits 0~6: Voltage offset for CKS, Bit 7: Disable/enable for the SCLK level. */
} Tonga_SCLK_Dependency_Record;

typedef struct _Tonga_SCLK_Dependency_Table {
  uint8_t ucRevId;
  uint8_t ucNumEntries;                     /* Number of entries. */
  Tonga_SCLK_Dependency_Record entries[1];         /* Dynamically allocate entries. */
} Tonga_SCLK_Dependency_Table;

typedef struct _Polaris_SCLK_Dependency_Record {
  uint8_t  ucVddInd;                      /* Base voltage */
  uint16_t usVddcOffset;                    /* Offset relative to base voltage */
  uint32_t ulSclk;
  uint16_t usEdcCurrent;
  uint8_t  ucReliabilityTemperature;
  uint8_t  ucCKSVOffsetandDisable;      /* Bits 0~6: Voltage offset for CKS, Bit 7: Disable/enable for the SCLK level. */
  uint32_t  ulSclkOffset;
} Polaris_SCLK_Dependency_Record;

typedef struct _Polaris_SCLK_Dependency_Table {
  uint8_t ucRevId;
  uint8_t ucNumEntries;              /* Number of entries. */
  Polaris_SCLK_Dependency_Record entries[1];         /* Dynamically allocate entries. */
} Polaris_SCLK_Dependency_Table;

typedef struct _Tonga_PCIE_Record {
  uint8_t ucPCIEGenSpeed;
  uint8_t usPCIELaneWidth;
  uint8_t ucReserved[2];
} Tonga_PCIE_Record;

typedef struct _Tonga_PCIE_Table {
  uint8_t ucRevId;
  uint8_t ucNumEntries;                     /* Number of entries. */
  Tonga_PCIE_Record entries[1];              /* Dynamically allocate entries. */
} Tonga_PCIE_Table;

typedef struct _Polaris10_PCIE_Record {
  uint8_t ucPCIEGenSpeed;
  uint8_t usPCIELaneWidth;
  uint8_t ucReserved[2];
  uint32_t ulPCIE_Sclk;
} Polaris10_PCIE_Record;

typedef struct _Polaris10_PCIE_Table {
  uint8_t ucRevId;
  uint8_t ucNumEntries;                                         /* Number of entries. */
  Polaris10_PCIE_Record entries[1];                      /* Dynamically allocate entries. */
} Polaris10_PCIE_Table;


typedef struct _Tonga_MM_Dependency_Record {
  uint8_t   ucVddcInd;                       /* VDDC voltage */
  uint16_t  usVddgfxOffset;                    /* Offset relative to VDDC voltage */
  uint32_t  ulDClk;                        /* UVD D-clock */
  uint32_t  ulVClk;                        /* UVD V-clock */
  uint32_t  ulEClk;                        /* VCE clock */
  uint32_t  ulAClk;                        /* ACP clock */
  uint32_t  ulSAMUClk;                      /* SAMU clock */
} Tonga_MM_Dependency_Record;

typedef struct _Tonga_MM_Dependency_Table {
  uint8_t ucRevId;
  uint8_t ucNumEntries;                     /* Number of entries. */
  Tonga_MM_Dependency_Record entries[1];          /* Dynamically allocate entries. */
} Tonga_MM_Dependency_Table;

typedef struct _Tonga_Voltage_Lookup_Record {
  uint16_t usVdd;                         /* Base voltage */
  uint16_t usCACLow;
  uint16_t usCACMid;
  uint16_t usCACHigh;
} Tonga_Voltage_Lookup_Record;

typedef struct _Tonga_Voltage_Lookup_Table {
  uint8_t ucRevId;
  uint8_t ucNumEntries;                     /* Number of entries. */
  Tonga_Voltage_Lookup_Record entries[1];        /* Dynamically allocate entries. */
} Tonga_Voltage_Lookup_Table;

typedef struct _Tonga_Fan_Table {
  uint8_t   ucRevId;             /* Change this if the table format changes or version changes so that the other fields are not the same. */
  uint8_t   ucTHyst;             /* Temperature hysteresis. Integer. */
  uint16_t  usTMin;              /* The temperature, in 0.01 centigrades, below which we just run at a minimal PWM. */
  uint16_t  usTMed;              /* The middle temperature where we change slopes. */
  uint16_t  usTHigh;             /* The high point above TMed for adjusting the second slope. */
  uint16_t  usPWMMin;             /* The minimum PWM value in percent (0.01% increments). */
  uint16_t  usPWMMed;             /* The PWM value (in percent) at TMed. */
  uint16_t  usPWMHigh;             /* The PWM value at THigh. */
  uint16_t  usTMax;              /* The max temperature */
  uint8_t   ucFanControlMode;          /* Legacy or Fuzzy Fan mode */
  uint16_t  usFanPWMMax;            /* Maximum allowed fan power in percent */
  uint16_t  usFanOutputSensitivity;      /* Sensitivity of fan reaction to temepature changes */
  uint16_t  usFanRPMMax;            /* The default value in RPM */
  uint32_t  ulMinFanSCLKAcousticLimit;     /* Minimum Fan Controller SCLK Frequency Acoustic Limit. */
  uint8_t   ucTargetTemperature;       /* Advanced fan controller target temperature. */
  uint8_t   ucMinimumPWMLimit;         /* The minimum PWM that the advanced fan controller can set.  This should be set to the highest PWM that will run the fan at its lowest RPM. */
  uint16_t  usReserved;
} Tonga_Fan_Table;

typedef struct _Fiji_Fan_Table {
  uint8_t   ucRevId;             /* Change this if the table format changes or version changes so that the other fields are not the same. */
  uint8_t   ucTHyst;             /* Temperature hysteresis. Integer. */
  uint16_t  usTMin;              /* The temperature, in 0.01 centigrades, below which we just run at a minimal PWM. */
  uint16_t  usTMed;              /* The middle temperature where we change slopes. */
  uint16_t  usTHigh;             /* The high point above TMed for adjusting the second slope. */
  uint16_t  usPWMMin;             /* The minimum PWM value in percent (0.01% increments). */
  uint16_t  usPWMMed;             /* The PWM value (in percent) at TMed. */
  uint16_t  usPWMHigh;             /* The PWM value at THigh. */
  uint16_t  usTMax;              /* The max temperature */
  uint8_t   ucFanControlMode;          /* Legacy or Fuzzy Fan mode */
  uint16_t  usFanPWMMax;            /* Maximum allowed fan power in percent */
  uint16_t  usFanOutputSensitivity;      /* Sensitivity of fan reaction to temepature changes */
  uint16_t  usFanRPMMax;            /* The default value in RPM */
  uint32_t  ulMinFanSCLKAcousticLimit;    /* Minimum Fan Controller SCLK Frequency Acoustic Limit. */
  uint8_t   ucTargetTemperature;       /* Advanced fan controller target temperature. */
  uint8_t   ucMinimumPWMLimit;         /* The minimum PWM that the advanced fan controller can set.  This should be set to the highest PWM that will run the fan at its lowest RPM. */
  uint16_t  usFanGainEdge;
  uint16_t  usFanGainHotspot;
  uint16_t  usFanGainLiquid;
  uint16_t  usFanGainVrVddc;
  uint16_t  usFanGainVrMvdd;
  uint16_t  usFanGainPlx;
  uint16_t  usFanGainHbm;
  uint16_t  usReserved;
} Fiji_Fan_Table;

typedef struct _Tonga_Thermal_Controller {
  uint8_t ucRevId;
  uint8_t ucType;       /* one of TONGA_PP_THERMALCONTROLLER_* */
  uint8_t ucI2cLine;    /* as interpreted by DAL I2C */
  uint8_t ucI2cAddress;
  uint8_t ucFanParameters;  /* Fan Control Parameters. */
  uint8_t ucFanMinRPM;    /* Fan Minimum RPM (hundreds) -- for display purposes only. */
  uint8_t ucFanMaxRPM;    /* Fan Maximum RPM (hundreds) -- for display purposes only. */
  uint8_t ucReserved;
  uint8_t ucFlags;       /* to be defined */
} Tonga_Thermal_Controller;

typedef struct _Tonga_VCE_State_Record {
  uint8_t  ucVCEClockIndex;  /*index into usVCEDependencyTableOffset of 'Tonga_MM_Dependency_Table' type */
  uint8_t  ucFlag;    /* 2 bits indicates memory p-states */
  uint8_t  ucSCLKIndex;    /*index into Tonga_SCLK_Dependency_Table */
  uint8_t  ucMCLKIndex;    /*index into Tonga_MCLK_Dependency_Table */
} Tonga_VCE_State_Record;

typedef struct _Tonga_VCE_State_Table {
  uint8_t ucRevId;
  uint8_t ucNumEntries;
  Tonga_VCE_State_Record entries[1];
} Tonga_VCE_State_Table;

typedef struct _Tonga_PowerTune_Table {
  uint8_t  ucRevId;
  uint16_t usTDP;
  uint16_t usConfigurableTDP;
  uint16_t usTDC;
  uint16_t usBatteryPowerLimit;
  uint16_t usSmallPowerLimit;
  uint16_t usLowCACLeakage;
  uint16_t usHighCACLeakage;
  uint16_t usMaximumPowerDeliveryLimit;
  uint16_t usTjMax;
  uint16_t usPowerTuneDataSetID;
  uint16_t usEDCLimit;
  uint16_t usSoftwareShutdownTemp;
  uint16_t usClockStretchAmount;
  uint16_t usReserve[2];
} Tonga_PowerTune_Table;

typedef struct _Fiji_PowerTune_Table {
  uint8_t  ucRevId;
  uint16_t usTDP;
  uint16_t usConfigurableTDP;
  uint16_t usTDC;
  uint16_t usBatteryPowerLimit;
  uint16_t usSmallPowerLimit;
  uint16_t usLowCACLeakage;
  uint16_t usHighCACLeakage;
  uint16_t usMaximumPowerDeliveryLimit;
  uint16_t usTjMax;  /* For Fiji, this is also usTemperatureLimitEdge; */
  uint16_t usPowerTuneDataSetID;
  uint16_t usEDCLimit;
  uint16_t usSoftwareShutdownTemp;
  uint16_t usClockStretchAmount;
  uint16_t usTemperatureLimitHotspot;  /*The following are added for Fiji */
  uint16_t usTemperatureLimitLiquid1;
  uint16_t usTemperatureLimitLiquid2;
  uint16_t usTemperatureLimitVrVddc;
  uint16_t usTemperatureLimitVrMvdd;
  uint16_t usTemperatureLimitPlx;
  uint8_t  ucLiquid1_I2C_address;  /*Liquid */
  uint8_t  ucLiquid2_I2C_address;
  uint8_t  ucLiquid_I2C_Line;
  uint8_t  ucVr_I2C_address;  /*VR */
  uint8_t  ucVr_I2C_Line;
  uint8_t  ucPlx_I2C_address;  /*PLX */
  uint8_t  ucPlx_I2C_Line;
  uint16_t usReserved;
} Fiji_PowerTune_Table;

#define PPM_A_A    1
#define PPM_A_I    2
typedef struct _Tonga_PPM_Table {
  uint8_t   ucRevId;
  uint8_t   ucPpmDesign;      /*A+I or A+A */
  uint16_t  usCpuCoreNumber;
  uint32_t  ulPlatformTDP;
  uint32_t  ulSmallACPlatformTDP;
  uint32_t  ulPlatformTDC;
  uint32_t  ulSmallACPlatformTDC;
  uint32_t  ulApuTDP;
  uint32_t  ulDGpuTDP;
  uint32_t  ulDGpuUlvPower;
  uint32_t  ulTjmax;
} Tonga_PPM_Table;

typedef struct _Tonga_Hard_Limit_Record {
  uint32_t  ulSCLKLimit;
  uint32_t  ulMCLKLimit;
  uint16_t  usVddcLimit;
  uint16_t  usVddciLimit;
  uint16_t  usVddgfxLimit;
} Tonga_Hard_Limit_Record;

typedef struct _Tonga_Hard_Limit_Table {
  uint8_t ucRevId;
  uint8_t ucNumEntries;
  Tonga_Hard_Limit_Record entries[1];
} Tonga_Hard_Limit_Table;

typedef struct _Tonga_GPIO_Table {
  uint8_t  ucRevId;
  uint8_t  ucVRHotTriggeredSclkDpmIndex;    /* If VRHot signal is triggered SCLK will be limited to this DPM level */
  uint8_t  ucReserve[5];
} Tonga_GPIO_Table;

typedef struct _PPTable_Generic_SubTable_Header {
  uint8_t  ucRevId;
} PPTable_Generic_SubTable_Header;


#pragma pack(pop)


#endif

