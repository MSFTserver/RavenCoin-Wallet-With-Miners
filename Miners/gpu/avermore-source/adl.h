#ifndef ADL_H
#define ADL_H

extern bool adl_active;
extern bool opt_reorder;
extern int opt_hysteresis;
extern int opt_targettemp;
extern int opt_overheattemp;

#ifdef HAVE_ADL

void init_adl(int nDevs);
void change_gpusettings(int gpu);
void clear_adl(int nDevs);

#else /* HAVE_ADL */

#define adl_active (0)
static inline void init_adl(__maybe_unused int nDevs) {}
static inline void change_gpusettings(__maybe_unused int gpu) { }
static inline void clear_adl(__maybe_unused int nDevs) {}

#endif /* HAVE_ADL */

#endif /* ADL_H */
