#ifndef X16R_H
#define X16R_H

#include "miner.h"

#include <stdint.h>

extern int x16r_test(unsigned char *pdata, const unsigned char *ptarget,
			uint32_t nonce);
extern void x16r_regenhash(struct work *work);

enum {
  X16R_BLAKE = 0,
  X16R_BMW,
  X16R_GROESTL,
  X16R_JH,
  X16R_KECCAK,
  X16R_SKEIN,
  X16R_LUFFA,
  X16R_CUBEHASH,
  X16R_SHAVITE,
  X16R_SIMD,
  X16R_ECHO,
  X16R_HAMSI,
  X16R_FUGUE,
  X16R_SHABAL,
  X16R_WHIRLPOOL,
  X16R_SHA512,
  X16R_HASH_FUNC_COUNT
};

extern
const char* X16R_ALGO_NAMES[X16R_HASH_FUNC_COUNT];

static inline
void x16r_getalgolist(const uint8_t* data, char *output)
{
  uint8_t *orig = output;

  for (int j = 0; j < X16R_HASH_FUNC_COUNT; j++) {
    int b = (15 - j) >> 1;
    *output++ = (j & 1) ? data[b] & 0xF : data[b] >> 4;
  }
}


#endif /* X16R_H */
