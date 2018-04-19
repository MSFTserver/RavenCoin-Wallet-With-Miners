#ifndef YESCRYPT_H
#define YESCRYPT_H

#include "miner.h"
#define YESCRYPT_SCRATCHBUF_SIZE (128 * 2048 * 8 ) //uchar
#define YESCRYP_SECBUF_SIZE (128*64*8)
extern int yescrypt_test(unsigned char *pdata, const unsigned char *ptarget, uint32_t nonce);
extern void yescrypt_regenhash(struct work *work);

#endif /* YESCRYPT_H */
