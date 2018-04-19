#ifndef X16S_H
#define X16S_H

#include "miner.h"

#include <stdint.h>

extern int x16s_test(unsigned char *pdata, const unsigned char *ptarget,
  uint32_t nonce);
extern void x16s_regenhash(struct work *work);

enum {
  X16S_BLAKE = 0,
  X16S_BMW,
  X16S_GROESTL,
  X16S_JH,
  X16S_KECCAK,
  X16S_SKEIN,
  X16S_LUFFA,
  X16S_CUBEHASH,
  X16S_SHAVITE,
  X16S_SIMD,
  X16S_ECHO,
  X16S_HAMSI,
  X16S_FUGUE,
  X16S_SHABAL,
  X16S_WHIRLPOOL,
  X16S_SHA512,
  X16S_HASH_FUNC_COUNT
};

extern
const char* X16S_ALGO_NAMES[X16S_HASH_FUNC_COUNT];

static inline
void x16s_getalgolist(const uint8_t* data, uint8_t *output)
{
  for (uint8_t i = 0; i < X16S_HASH_FUNC_COUNT; i++)
    output[i] = i;

  for(int i = 0; i < X16S_HASH_FUNC_COUNT; i++){
    uint8_t b = (15 - i) >> 1; // 16 ascii hex chars, reversed
    uint8_t algoDigit = (i & 1) ? data[b] & 0xF : data[b] >> 4;
    // insert the nth character at the front
    char oldVal = output[algoDigit];
    for(int j=algoDigit; j-->0;)
      output[j+1] = output[j];
    output[0] = oldVal;
  }
}


#endif /* X16S_H */
