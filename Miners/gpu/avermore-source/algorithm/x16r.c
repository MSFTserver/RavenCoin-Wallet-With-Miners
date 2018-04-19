/*-
 * Copyright 2009 Colin Percival, 2011 ArtForz
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file was originally written by Colin Percival as part of the Tarsnap
 * online backup system.
 */

#include "config.h"
#include "miner.h"

#include "x16r.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>


#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_skein.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"

#ifdef _MSC_VER
# define _ALIGN(x) __declspec(align(x))
#else
# define _ALIGN(x) __attribute__ ((aligned(x)))
#endif


const char* X16R_ALGO_NAMES[X16R_HASH_FUNC_COUNT] = {
  "blake",
  "bmw512",
  "groestl",
  "jh512",
  "keccak",
  "skein",
  "luffa",
  "cube",
  "shavite",
  "simd",
  "echo",
  "hamsi",
  "fugue",
  "shabal",
  "whirlpool",
  "sha512"
};

/*
 * Encode a length len/4 vector of (uint32_t) into a length len vector of
 * (unsigned char) in big-endian form.  Assumes len is a multiple of 4.
 */
static inline void be32enc_vect(uint32_t *dst, const uint32_t *src, uint32_t len)
{
  uint32_t i;

  for (i = 0; i < len; i++)
    dst[i] = htobe32(src[i]);
}

void x16r_hash(void *state, const void *input)
{
  unsigned char _ALIGN(64) hash[128];

  sph_blake512_context ctx_blake;
  sph_bmw512_context ctx_bmw;
  sph_groestl512_context ctx_groestl;
  sph_jh512_context ctx_jh;
  sph_keccak512_context ctx_keccak;
  sph_skein512_context ctx_skein;
  sph_luffa512_context ctx_luffa;
  sph_cubehash512_context ctx_cubehash;
  sph_shavite512_context ctx_shavite;
  sph_simd512_context ctx_simd;
  sph_echo512_context ctx_echo;
  sph_hamsi512_context ctx_hamsi;
  sph_fugue512_context ctx_fugue;
  sph_shabal512_context ctx_shabal;
  sph_whirlpool_context ctx_whirlpool;
  sph_sha512_context ctx_sha512;

  const void *in = input;
  int size = 80;
  uint8_t hashOrder[X16R_HASH_FUNC_COUNT];

  if (opt_benchmark) {
    for (uint8_t i = 0; i < X16R_HASH_FUNC_COUNT; i++)
      hashOrder[i] = opt_benchmark_seq[i];
  }
  else x16r_getalgolist((uint8_t*)input + 4, hashOrder);

  for (int i = 0; i < X16R_HASH_FUNC_COUNT; i++)
  {
    const uint8_t algo = hashOrder[i];

    switch (algo) {
    case X16R_BLAKE:
      sph_blake512_init(&ctx_blake);
      sph_blake512(&ctx_blake, in, size);
      sph_blake512_close(&ctx_blake, hash);
      break;
    case X16R_BMW:
      sph_bmw512_init(&ctx_bmw);
      sph_bmw512(&ctx_bmw, in, size);
      sph_bmw512_close(&ctx_bmw, hash);
      break;
    case X16R_GROESTL:
      sph_groestl512_init(&ctx_groestl);
      sph_groestl512(&ctx_groestl, in, size);
      sph_groestl512_close(&ctx_groestl, hash);
      break;
    case X16R_SKEIN:
      sph_skein512_init(&ctx_skein);
      sph_skein512(&ctx_skein, in, size);
      sph_skein512_close(&ctx_skein, hash);
      break;
    case X16R_JH:
      sph_jh512_init(&ctx_jh);
      sph_jh512(&ctx_jh, in, size);
      sph_jh512_close(&ctx_jh, hash);
      break;
    case X16R_KECCAK:
      sph_keccak512_init(&ctx_keccak);
      sph_keccak512(&ctx_keccak, in, size);
      sph_keccak512_close(&ctx_keccak, hash);
      break;
    case X16R_LUFFA:
      sph_luffa512_init(&ctx_luffa);
      sph_luffa512(&ctx_luffa, in, size);
      sph_luffa512_close(&ctx_luffa, hash);
      break;
    case X16R_CUBEHASH:
      sph_cubehash512_init(&ctx_cubehash);
      sph_cubehash512(&ctx_cubehash, in, size);
      sph_cubehash512_close(&ctx_cubehash, hash);
      break;
    case X16R_SHAVITE:
      sph_shavite512_init(&ctx_shavite);
      sph_shavite512(&ctx_shavite, in, size);
      sph_shavite512_close(&ctx_shavite, hash);
      break;
    case X16R_SIMD:
      sph_simd512_init(&ctx_simd);
      sph_simd512(&ctx_simd, in, size);
      sph_simd512_close(&ctx_simd, hash);
      break;
    case X16R_ECHO:
      sph_echo512_init(&ctx_echo);
      sph_echo512(&ctx_echo, in, size);
      sph_echo512_close(&ctx_echo, hash);
      break;
    case X16R_HAMSI:
      sph_hamsi512_init(&ctx_hamsi);
      sph_hamsi512(&ctx_hamsi, in, size);
      sph_hamsi512_close(&ctx_hamsi, hash);
      break;
    case X16R_FUGUE:
      sph_fugue512_init(&ctx_fugue);
      sph_fugue512(&ctx_fugue, in, size);
      sph_fugue512_close(&ctx_fugue, hash);
      break;
    case X16R_SHABAL:
      sph_shabal512_init(&ctx_shabal);
      sph_shabal512(&ctx_shabal, in, size);
      sph_shabal512_close(&ctx_shabal, hash);
      break;
    case X16R_WHIRLPOOL:
      sph_whirlpool_init(&ctx_whirlpool);
      sph_whirlpool(&ctx_whirlpool, in, size);
      sph_whirlpool_close(&ctx_whirlpool, hash);
      break;
    case X16R_SHA512:
      sph_sha512_init(&ctx_sha512);
      sph_sha512(&ctx_sha512,(const void*) in, size);
      sph_sha512_close(&ctx_sha512,(void*) hash);
      break;
    }
    in = (const void*) hash;
    size = 64;
  }
  memcpy(state, hash, 32);
}

static const uint32_t diff1targ = 0x0000ffff;

/* Used externally as confirmation of correct OCL code */
int x16r_test(unsigned char *pdata, const unsigned char *ptarget, uint32_t nonce)
{
  uint32_t tmp_hash7, Htarg = le32toh(((const uint32_t *)ptarget)[7]);
  uint32_t data[20], ohash[8];

  be32enc_vect(data, (const uint32_t *)pdata, 19);
  data[19] = htobe32(nonce);
  x16r_hash(ohash, data);
  tmp_hash7 = be32toh(ohash[7]);

  applog(LOG_DEBUG, "htarget %08lx diff1 %08lx hash %08lx",
        (long unsigned int)Htarg,
        (long unsigned int)diff1targ,
        (long unsigned int)tmp_hash7);

  if (tmp_hash7 > diff1targ)
    return -1;

  if (tmp_hash7 > Htarg)
    return 0;

  return 1;
}

void x16r_regenhash(struct work *work)
{
  uint32_t data[20];
  uint32_t *nonce = (uint32_t *)(work->data + 76);
  uint32_t *ohash = (uint32_t *)(work->hash);

  be32enc_vect(data, (const uint32_t *)work->data, 19);
  data[19] = htobe32(*nonce);
  x16r_hash(ohash, data);
}

bool scanhash_x16r(struct thr_info *thr, const unsigned char __maybe_unused *pmidstate,
         unsigned char *pdata, unsigned char __maybe_unused *phash1,
         unsigned char __maybe_unused *phash, const unsigned char *ptarget,
         uint32_t max_nonce, uint32_t *last_nonce, uint32_t n)
{
  uint32_t *nonce = (uint32_t *)(pdata + 76);
  uint32_t data[20];
  uint32_t tmp_hash7;
  uint32_t Htarg = le32toh(((const uint32_t *)ptarget)[7]);
  bool ret = false;

  be32enc_vect(data, (const uint32_t *)pdata, 19);

  while(1)
  {
    uint32_t ostate[8];
    *nonce = ++n;
    data[19] = (n);
    x16r_hash(ostate, data);
    tmp_hash7 = (ostate[7]);

    applog(LOG_INFO, "data7 %08lx", (long unsigned int)data[7]);

    if(unlikely(tmp_hash7 <= Htarg))
    {
      ((uint32_t *)pdata)[19] = htobe32(n);
      *last_nonce = n;
      ret = true;
      break;
    }

    if (unlikely((n >= max_nonce) || thr->work_restart))
    {
      *last_nonce = n;
      break;
    }
  }

  return ret;
}
