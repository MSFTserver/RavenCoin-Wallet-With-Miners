#ifndef CREDITS_H
#define CREDITS_H

#include "miner.h"


extern int credits_test(unsigned char *pdata, const unsigned char *ptarget, uint32_t nonce);
extern void credits_regenhash(struct work *work);

#endif /* CREDITS_H */
