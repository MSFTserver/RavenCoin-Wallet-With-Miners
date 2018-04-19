#ifndef XEVAN_H
#define XEVAN_H

#include "miner.h"

extern int xevan_test(unsigned char *pdata, const unsigned char *ptarget,
			uint32_t nonce);
extern void xevan_regenhash(struct work *work);

#endif /* XEVAN_H */