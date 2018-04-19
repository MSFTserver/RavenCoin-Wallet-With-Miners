/*
 * Quick and dirty addition of Fugue-512 for X13
 *
 * Built on cbuchner1's implementation, actual hashing code
 * heavily based on phm's sgminer
 *
 *
 */
#include "cuda_helper_alexis.h"
#include "miner.h"
#include "cuda_vectors_alexis.h"
/*
 * X13 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014-2016  phm, Provos Alexis
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software", to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   phm <phm@inbox.com>
 * @author   Provos Alexis (Applied partial shared Mem utilization under CUDA 7.5 for compute5.0/5.2 / 2016)
 */

static __constant__ const uint32_t c_S[16] = {
		0x8807a57e, 0xe616af75, 0xc5d3e4db, 0xac9ab027,
		0xd915f117, 0xb6eecc54, 0x06e8020b, 0x4a92efd1,
		0xaac6e2c9, 0xddb21398, 0xcae65838, 0x437f203f,
		0x25ea78e7, 0x951fddd6, 0xda6ed11d, 0xe13e3567
};

static __device__ uint32_t mixtab0[256] = {
	0x63633297, 0x7c7c6feb, 0x77775ec7, 0x7b7b7af7, 0xf2f2e8e5, 0x6b6b0ab7,	0x6f6f16a7, 0xc5c56d39, 0x303090c0, 0x01010704, 0x67672e87, 0x2b2bd1ac, 0xfefeccd5, 0xd7d71371, 0xabab7c9a,
	0x767659c3, 0xcaca4005, 0x8282a33e, 0xc9c94909, 0x7d7d68ef, 0xfafad0c5,	0x5959947f, 0x4747ce07, 0xf0f0e6ed, 0xadad6e82, 0xd4d41a7d, 0xa2a243be, 0xafaf608a, 0x9c9cf946, 0xa4a451a6,
	0x727245d3, 0xc0c0762d, 0xb7b728ea, 0xfdfdc5d9, 0x9393d47a, 0x2626f298, 0x363682d8, 0x3f3fbdfc, 0xf7f7f3f1, 0xcccc521d, 0x34348cd0, 0xa5a556a2, 0xe5e58db9, 0xf1f1e1e9, 0x71714cdf,
	0xd8d83e4d, 0x313197c4, 0x15156b54, 0x04041c10, 0xc7c76331, 0x2323e98c, 0xc3c37f21, 0x18184860, 0x9696cf6e, 0x05051b14, 0x9a9aeb5e, 0x0707151c, 0x12127e48, 0x8080ad36, 0xe2e298a5,
	0xebeba781, 0x2727f59c, 0xb2b233fe, 0x757550cf, 0x09093f24, 0x8383a43a, 0x2c2cc4b0, 0x1a1a4668, 0x1b1b416c, 0x6e6e11a3, 0x5a5a9d73, 0xa0a04db6, 0x5252a553, 0x3b3ba1ec, 0xd6d61475,
	0xb3b334fa, 0x2929dfa4, 0xe3e39fa1, 0x2f2fcdbc, 0x8484b126, 0x5353a257, 0xd1d10169, 0x00000000, 0xededb599, 0x2020e080, 0xfcfcc2dd, 0xb1b13af2, 0x5b5b9a77, 0x6a6a0db3, 0xcbcb4701,
	0xbebe17ce, 0x3939afe4, 0x4a4aed33, 0x4c4cff2b, 0x5858937b, 0xcfcf5b11, 0xd0d0066d, 0xefefbb91, 0xaaaa7b9e, 0xfbfbd7c1, 0x4343d217, 0x4d4df82f, 0x333399cc, 0x8585b622, 0x4545c00f,
	0xf9f9d9c9, 0x02020e08, 0x7f7f66e7, 0x5050ab5b, 0x3c3cb4f0, 0x9f9ff04a, 0xa8a87596, 0x5151ac5f, 0xa3a344ba, 0x4040db1b, 0x8f8f800a, 0x9292d37e, 0x9d9dfe42, 0x3838a8e0, 0xf5f5fdf9,
	0xbcbc19c6, 0xb6b62fee, 0xdada3045, 0x2121e784, 0x10107040, 0xffffcbd1, 0xf3f3efe1, 0xd2d20865, 0xcdcd5519, 0x0c0c2430, 0x1313794c, 0xececb29d, 0x5f5f8667, 0x9797c86a, 0x4444c70b,
	0x1717655c, 0xc4c46a3d, 0xa7a758aa, 0x7e7e61e3, 0x3d3db3f4, 0x6464278b, 0x5d5d886f, 0x19194f64, 0x737342d7, 0x60603b9b, 0x8181aa32, 0x4f4ff627, 0xdcdc225d, 0x2222ee88, 0x2a2ad6a8,
	0x9090dd76, 0x88889516, 0x4646c903, 0xeeeebc95, 0xb8b805d6, 0x14146c50, 0xdede2c55, 0x5e5e8163, 0x0b0b312c, 0xdbdb3741, 0xe0e096ad, 0x32329ec8, 0x3a3aa6e8, 0x0a0a3628, 0x4949e43f,
	0x06061218, 0x2424fc90, 0x5c5c8f6b, 0xc2c27825, 0xd3d30f61, 0xacac6986, 0x62623593, 0x9191da72, 0x9595c662, 0xe4e48abd, 0x797974ff, 0xe7e783b1, 0xc8c84e0d, 0x373785dc, 0x6d6d18af,
	0x8d8d8e02, 0xd5d51d79, 0x4e4ef123, 0xa9a97292, 0x6c6c1fab, 0x5656b943, 0xf4f4fafd, 0xeaeaa085, 0x6565208f, 0x7a7a7df3, 0xaeae678e, 0x08083820, 0xbaba0bde, 0x787873fb, 0x2525fb94,
	0x2e2ecab8, 0x1c1c5470, 0xa6a65fae, 0xb4b421e6, 0xc6c66435, 0xe8e8ae8d, 0xdddd2559, 0x747457cb, 0x1f1f5d7c, 0x4b4bea37, 0xbdbd1ec2, 0x8b8b9c1a, 0x8a8a9b1e, 0x70704bdb, 0x3e3ebaf8,
	0xb5b526e2, 0x66662983, 0x4848e33b, 0x0303090c, 0xf6f6f4f5, 0x0e0e2a38, 0x61613c9f, 0x35358bd4, 0x5757be47, 0xb9b902d2, 0x8686bf2e, 0xc1c17129, 0x1d1d5374, 0x9e9ef74e, 0xe1e191a9,
	0xf8f8decd, 0x9898e556, 0x11117744, 0x696904bf, 0xd9d93949, 0x8e8e870e, 0x9494c166, 0x9b9bec5a, 0x1e1e5a78, 0x8787b82a, 0xe9e9a989, 0xcece5c15, 0x5555b04f, 0x2828d8a0, 0xdfdf2b51,
	0x8c8c8906, 0xa1a14ab2, 0x89899212, 0x0d0d2334, 0xbfbf10ca, 0xe6e684b5, 0x4242d513, 0x686803bb, 0x4141dc1f, 0x9999e252, 0x2d2dc3b4, 0x0f0f2d3c, 0xb0b03df6, 0x5454b74b, 0xbbbb0cda,
	0x16166258
};

#define mixtab0(x) shared[0][x]
#define mixtab1(x) shared[1][x]
#define mixtab2(x) shared[2][x]
#define mixtab3(x) shared[3][x]

#define TIX4(q, x00, x01, x04, x07, x08, x22, x24, x27, x30) { \
		x22 ^= x00; \
		x00 = (q); \
		x08 ^= (q); \
		x01 ^= x24; \
		x04 ^= x27; \
		x07 ^= x30; \
	}

#define CMIX36(x00, x01, x02, x04, x05, x06, x18, x19, x20) { \
		x00 ^= x04; \
		x01 ^= x05; \
		x02 ^= x06; \
		x18 ^= x04; \
		x19 ^= x05; \
		x20 ^= x06; \
	}

__device__ __forceinline__
static void SMIX(const uint32_t shared[4][256], uint32_t &x0,uint32_t &x1,uint32_t &x2,uint32_t &x3){
	uint32_t c0 = mixtab0(__byte_perm(x0,0,0x4443));
	uint32_t r1 = mixtab1(__byte_perm(x0,0,0x4442));
	uint32_t r2 = mixtab2(__byte_perm(x0,0,0x4441));
	uint32_t r3 = mixtab3(__byte_perm(x0,0,0x4440));
	c0 = c0 ^ r1 ^ r2 ^ r3;
	uint32_t r0 = mixtab0(__byte_perm(x1,0,0x4443));
	uint32_t c1 = r0 ^ mixtab1(__byte_perm(x1,0,0x4442));
	uint32_t tmp = mixtab2(__byte_perm(x1,0,0x4441));
	c1 ^= tmp;
	r2 ^= tmp;
	tmp = mixtab3(__byte_perm(x1,0,0x4440));
	c1 ^= tmp;
	r3 ^= tmp;
	uint32_t c2 = mixtab0(__byte_perm(x2,0,0x4443));
	r0 ^= c2;
	tmp = mixtab1(__byte_perm(x2,0,0x4442));
	c2 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x2,0,0x4441));
	c2 ^= tmp;
	tmp = mixtab3(__byte_perm(x2,0,0x4440));
	c2 ^= tmp;
	r3 ^= tmp;
	uint32_t c3 = mixtab0(__byte_perm(x3,0,0x4443));
	r0 ^= c3;
	tmp = mixtab1(__byte_perm(x3,0,0x4442));
	c3 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x3,0,0x4441));
	c3 ^= tmp;
	r2 ^= tmp;
	tmp = mixtab3(__byte_perm(x3,0,0x4440));
	c3 ^= tmp;
	x0 = ((c0 ^ (r0 << 0)) & 0xFF000000) | ((c1 ^ (r1 << 0)) & 0x00FF0000) | ((c2 ^ (r2 << 0)) & 0x0000FF00) | ((c3 ^ (r3 << 0)) & 0x000000FF);
	x1 = ((c1 ^ (r0 << 8)) & 0xFF000000) | ((c2 ^ (r1 << 8)) & 0x00FF0000) | ((c3 ^ (r2 << 8)) & 0x0000FF00) | ((c0 ^ (r3 >>24)) & 0x000000FF);
	x2 = ((c2 ^ (r0 <<16)) & 0xFF000000) | ((c3 ^ (r1 <<16)) & 0x00FF0000) | ((c0 ^ (r2 >>16)) & 0x0000FF00) | ((c1 ^ (r3 >>16)) & 0x000000FF);
	x3 = ((c3 ^ (r0 <<24)) & 0xFF000000) | ((c0 ^ (r1 >> 8)) & 0x00FF0000) | ((c1 ^ (r2 >> 8)) & 0x0000FF00) | ((c2 ^ (r3 >> 8)) & 0x000000FF);
}

__device__
static void SMIX_LDG(const uint32_t shared[4][256], uint32_t &x0,uint32_t &x1,uint32_t &x2,uint32_t &x3){
	uint32_t c0 = __ldg(&mixtab0[__byte_perm(x0,0,0x4443)]);
	uint32_t r1 = mixtab1(__byte_perm(x0,0,0x4442));
	uint32_t r2 = mixtab2(__byte_perm(x0,0,0x4441));
	uint32_t r3 = mixtab3(__byte_perm(x0,0,0x4440));
	c0 = c0 ^ r1 ^ r2 ^ r3;
	uint32_t r0 = __ldg(&mixtab0[__byte_perm(x1,0,0x4443)]);
	uint32_t c1 = r0 ^ mixtab1(__byte_perm(x1,0,0x4442));
	uint32_t tmp = mixtab2(__byte_perm(x1,0,0x4441));
	c1 ^= tmp;
	r2 ^= tmp;
	tmp = mixtab3(__byte_perm(x1,0,0x4440));
	c1 ^= tmp;
	r3 ^= tmp;
	uint32_t c2 = __ldg(&mixtab0[__byte_perm(x2,0,0x4443)]);
	r0 ^= c2;
	tmp = mixtab1(__byte_perm(x2,0,0x4442));
	c2 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x2,0,0x4441));
	c2 ^= tmp;
	tmp = mixtab3(__byte_perm(x2,0,0x4440));
	c2 ^= tmp;
	r3 ^= tmp;
	uint32_t c3 = __ldg(&mixtab0[__byte_perm(x3,0,0x4443)]);
	r0 ^= c3;
	tmp = mixtab1(__byte_perm(x3,0,0x4442));
	c3 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x3,0,0x4441));
	c3 ^= tmp;
	r2 ^= tmp;
	tmp = ROL8(__ldg(&mixtab0[__byte_perm(x3,0,0x4440)]));
	c3 ^= tmp;
	x0 = ((c0 ^ (r0 << 0)) & 0xFF000000) | ((c1 ^ (r1 << 0)) & 0x00FF0000) | ((c2 ^ (r2 << 0)) & 0x0000FF00) | ((c3 ^ (r3 << 0)) & 0x000000FF);
	x1 = ((c1 ^ (r0 << 8)) & 0xFF000000) | ((c2 ^ (r1 << 8)) & 0x00FF0000) | ((c3 ^ (r2 << 8)) & 0x0000FF00) | ((c0 ^ (r3 >>24)) & 0x000000FF);
	x2 = ((c2 ^ (r0 <<16)) & 0xFF000000) | ((c3 ^ (r1 <<16)) & 0x00FF0000) | ((c0 ^ (r2 >>16)) & 0x0000FF00) | ((c1 ^ (r3 >>16)) & 0x000000FF);
	x3 = ((c3 ^ (r0 <<24)) & 0xFF000000) | ((c0 ^ (r1 >> 8)) & 0x00FF0000) | ((c1 ^ (r2 >> 8)) & 0x0000FF00) | ((c2 ^ (r3 >> 8)) & 0x000000FF);
}
#define mROR3 { \
	B[ 6] = S[33], B[ 7] = S[34], B[ 8] = S[35]; \
	S[35] = S[32]; S[34] = S[31]; S[33] = S[30]; S[32] = S[29]; S[31] = S[28]; S[30] = S[27]; S[29] = S[26]; S[28] = S[25]; S[27] = S[24]; \
	S[26] = S[23]; S[25] = S[22]; S[24] = S[21]; S[23] = S[20]; S[22] = S[19]; S[21] = S[18]; S[20] = S[17]; S[19] = S[16]; S[18] = S[15]; \
	S[17] = S[14]; S[16] = S[13]; S[15] = S[12]; S[14] = S[11]; S[13] = S[10]; S[12] = S[ 9]; S[11] = S[ 8]; S[10] = S[ 7]; S[ 9] = S[ 6]; \
	S[ 8] = S[ 5]; S[ 7] = S[ 4]; S[ 6] = S[ 3]; S[ 5] = S[ 2]; S[ 4] = S[ 1]; S[ 3] = S[ 0]; S[ 2] = B[ 8]; S[ 1] = B[ 7]; S[ 0] = B[ 6]; \
	}

#define mROR8 { \
	B[ 1] = S[28], B[ 2] = S[29], B[ 3] = S[30], B[ 4] = S[31], B[ 5] = S[32], B[ 6] = S[33], B[ 7] = S[34], B[ 8] = S[35]; \
	S[35] = S[27]; S[34] = S[26]; S[33] = S[25]; S[32] = S[24]; S[31] = S[23]; S[30] = S[22]; S[29] = S[21]; S[28] = S[20]; S[27] = S[19]; \
	S[26] = S[18]; S[25] = S[17]; S[24] = S[16]; S[23] = S[15]; S[22] = S[14]; S[21] = S[13]; S[20] = S[12]; S[19] = S[11]; S[18] = S[10]; \
	S[17] = S[ 9]; S[16] = S[ 8]; S[15] = S[ 7]; S[14] = S[ 6]; S[13] = S[ 5]; S[12] = S[ 4]; S[11] = S[ 3]; S[10] = S[ 2]; S[ 9] = S[ 1]; \
	S[ 8] = S[ 0]; S[ 7] = B[ 8]; S[ 6] = B[ 7]; S[ 5] = B[ 6]; S[ 4] = B[ 5]; S[ 3] = B[ 4]; S[ 2] = B[ 3]; S[ 1] = B[ 2]; S[ 0] = B[ 1]; \
	}

#define mROR9 { \
	B[ 0] = S[27], B[ 1] = S[28], B[ 2] = S[29], B[ 3] = S[30], B[ 4] = S[31], B[ 5] = S[32], B[ 6] = S[33], B[ 7] = S[34], B[ 8] = S[35]; \
	S[35] = S[26]; S[34] = S[25]; S[33] = S[24]; S[32] = S[23]; S[31] = S[22]; S[30] = S[21]; S[29] = S[20]; S[28] = S[19]; S[27] = S[18]; \
	S[26] = S[17]; S[25] = S[16]; S[24] = S[15]; S[23] = S[14]; S[22] = S[13]; S[21] = S[12]; S[20] = S[11]; S[19] = S[10]; S[18] = S[ 9]; \
	S[17] = S[ 8]; S[16] = S[ 7]; S[15] = S[ 6]; S[14] = S[ 5]; S[13] = S[ 4]; S[12] = S[ 3]; S[11] = S[ 2]; S[10] = S[ 1]; S[ 9] = S[ 0]; \
	S[ 8] = B[ 8]; S[ 7] = B[ 7]; S[ 6] = B[ 6]; S[ 5] = B[ 5]; S[ 4] = B[ 4]; S[ 3] = B[ 3]; S[ 2] = B[ 2]; S[ 1] = B[ 1]; S[ 0] = B[ 0]; \
	}

#define FUGUE512_3(x, y, z) {  \
        TIX4(x, S[ 0], S[ 1], S[ 4], S[ 7], S[ 8], S[22], S[24], S[27], S[30]); \
        CMIX36(S[33], S[34], S[35], S[ 1], S[ 2], S[ 3], S[15], S[16], S[17]); \
        SMIX_LDG(shared, S[33], S[34], S[35], S[ 0]); \
        CMIX36(S[30], S[31], S[32], S[34], S[35], S[ 0], S[12], S[13], S[14]); \
        SMIX_LDG(shared, S[30], S[31], S[32], S[33]); \
        CMIX36(S[27], S[28], S[29], S[31], S[32], S[33], S[ 9], S[10], S[11]); \
        SMIX(shared, S[27], S[28], S[29], S[30]); \
        CMIX36(S[24], S[25], S[26], S[28], S[29], S[30], S[ 6], S[ 7], S[ 8]); \
        SMIX_LDG(shared, S[24], S[25], S[26], S[27]); \
        \
        TIX4(y, S[24], S[25], S[28], S[31], S[32], S[10], S[12], S[15], S[18]); \
        CMIX36(S[21], S[22], S[23], S[25], S[26], S[27], S[ 3], S[ 4], S[ 5]); \
        SMIX(shared, S[21], S[22], S[23], S[24]); \
        CMIX36(S[18], S[19], S[20], S[22], S[23], S[24], S[ 0], S[ 1], S[ 2]); \
        SMIX_LDG(shared, S[18], S[19], S[20], S[21]); \
        CMIX36(S[15], S[16], S[17], S[19], S[20], S[21], S[33], S[34], S[35]); \
        SMIX_LDG(shared, S[15], S[16], S[17], S[18]); \
        CMIX36(S[12], S[13], S[14], S[16], S[17], S[18], S[30], S[31], S[32]); \
        SMIX(shared, S[12], S[13], S[14], S[15]); \
        \
        TIX4(z, S[12], S[13], S[16], S[19], S[20], S[34], S[ 0], S[ 3], S[ 6]); \
        CMIX36(S[ 9], S[10], S[11], S[13], S[14], S[15], S[27], S[28], S[29]); \
        SMIX_LDG(shared, S[ 9], S[10], S[11], S[12]); \
        CMIX36(S[ 6], S[ 7], S[ 8], S[10], S[11], S[12], S[24], S[25], S[26]); \
        SMIX_LDG(shared, S[ 6], S[ 7], S[ 8], S[ 9]); \
        CMIX36(S[ 3], S[ 4], S[ 5], S[ 7], S[ 8], S[ 9], S[21], S[22], S[23]); \
        SMIX_LDG(shared, S[ 3], S[ 4], S[ 5], S[ 6]); \
        CMIX36(S[ 0], S[ 1], S[ 2], S[ 4], S[ 5], S[ 6], S[18], S[19], S[20]); \
        SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]); \
	}

/***************************************************/
// Die Hash-Funktion
__global__ __launch_bounds__(256,3)
void x13_fugue512_gpu_hash_64_alexis(uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint32_t shared[4][256];

//	if(threadIdx.x<256){
		const uint32_t tmp = mixtab0[threadIdx.x];
		shared[0][threadIdx.x] = tmp;
		shared[1][threadIdx.x] = ROR8(tmp);
		shared[2][threadIdx.x] = ROL16(tmp);
		shared[3][threadIdx.x] = ROL8(tmp);
//	}
	__syncthreads();
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *hash = (uint32_t*)&g_hash[thread<<3];

		uint32_t S[36];
		uint32_t B[ 9];

		uint32_t Hash[16];

		*(uint2x4*)&Hash[0] = __ldg4((uint2x4*)&hash[0]);
		*(uint2x4*)&Hash[8] = __ldg4((uint2x4*)&hash[8]);

		#pragma unroll 16
		for(int i = 0; i < 16; i++)
			Hash[i] = cuda_swab32(Hash[i]);

		__syncthreads();

		S[ 0] = S[ 1] = S[ 2] = S[ 3] = S[ 4] = S[ 5] = S[ 6] = S[ 7] = S[ 8] = S[ 9] = S[10] = S[11] = S[12] = S[13] = S[14] = S[15] = S[16] = S[17] = S[18] = S[19] = 0;
		*(uint2x4*)&S[20] = *(uint2x4*)&c_S[ 0];
		*(uint2x4*)&S[28] = *(uint2x4*)&c_S[ 8];

		FUGUE512_3(Hash[0x0], Hash[0x1], Hash[0x2]);
		FUGUE512_3(Hash[0x3], Hash[0x4], Hash[0x5]);
		FUGUE512_3(Hash[0x6], Hash[0x7], Hash[0x8]);
		FUGUE512_3(Hash[0x9], Hash[0xA], Hash[0xB]);
		FUGUE512_3(Hash[0xC], Hash[0xD], Hash[0xE]);
		FUGUE512_3(Hash[0xF], 0U, 512U);

		for (uint32_t i = 0; i < 32; i+=2){
			mROR3;
			CMIX36(S[ 0], S[ 1], S[ 2], S[ 4], S[ 5], S[ 6], S[18], S[19], S[20]);
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			mROR3;
			CMIX36(S[ 0], S[ 1], S[ 2], S[ 4], S[ 5], S[ 6], S[18], S[19], S[20]);
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
		}
		#pragma unroll
		for (uint32_t i = 0; i < 13; i ++) {
			S[ 4] ^= S[ 0];	S[ 9] ^= S[ 0];	S[18] ^= S[ 0];	S[27] ^= S[ 0];
			mROR9;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[18] ^= S[ 0];	S[27] ^= S[ 0];
			mROR9;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[19] ^= S[ 0];	S[27] ^= S[ 0];
			mROR9;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[19] ^= S[ 0];	S[28] ^= S[ 0];
			mROR8;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
		}
		S[ 4] ^= S[ 0];	S[ 9] ^= S[ 0];	S[18] ^= S[ 0];	S[27] ^= S[ 0];

		S[ 0] = cuda_swab32(S[ 1]);	S[ 1] = cuda_swab32(S[ 2]);	S[ 2] = cuda_swab32(S[ 3]);	S[ 3] = cuda_swab32(S[ 4]);
		S[ 4] = cuda_swab32(S[ 9]);	S[ 5] = cuda_swab32(S[10]);	S[ 6] = cuda_swab32(S[11]);	S[ 7] = cuda_swab32(S[12]);
		S[ 8] = cuda_swab32(S[18]);	S[ 9] = cuda_swab32(S[19]);	S[10] = cuda_swab32(S[20]);	S[11] = cuda_swab32(S[21]);
		S[12] = cuda_swab32(S[27]);	S[13] = cuda_swab32(S[28]);	S[14] = cuda_swab32(S[29]);	S[15] = cuda_swab32(S[30]);

		*(uint2x4*)&hash[ 0] = *(uint2x4*)&S[ 0];
		*(uint2x4*)&hash[ 8] = *(uint2x4*)&S[ 8];
	}
}

/***************************************************/
// The final hash function
__global__ __launch_bounds__(512,2) /* force 56 registers */
void x13_fugue512_gpu_hash_64_final_alexis(uint32_t threads,const uint32_t* __restrict__ g_hash,uint32_t* resNonce, const uint64_t target){

	__shared__ uint32_t shared[4][256];

	if(threadIdx.x<256){
		const uint32_t tmp = mixtab0[threadIdx.x];
		shared[0][threadIdx.x] = tmp;
		shared[1][threadIdx.x] = ROR8(tmp);
		shared[2][threadIdx.x] = ROL16(tmp);
		shared[3][threadIdx.x] = ROL8(tmp);
	}

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t* __restrict__ hash = &g_hash[thread<<4];

		uint32_t S[36];
		uint32_t B[ 9];
		uint32_t Hash[16];

		*(uint2x4*)&Hash[0] = __ldg4((uint2x4*)&hash[0]);
		*(uint2x4*)&Hash[8] = __ldg4((uint2x4*)&hash[8]);
		__syncthreads();
		S[ 0] = S[ 1] = S[ 2] = S[ 3] = S[ 4] = S[ 5] = S[ 6] = S[ 7] = S[ 8] = S[ 9] = S[10] = S[11] = S[12] = S[13] = S[14] = S[15] = S[16] = S[17] = S[18] = S[19] = 0;
		*(uint2x4*)&S[20] = *(uint2x4*)&c_S[ 0];
		*(uint2x4*)&S[28] = *(uint2x4*)&c_S[ 8];

		FUGUE512_3(Hash[0x0], Hash[0x1], Hash[0x2]);
		FUGUE512_3(Hash[0x3], Hash[0x4], Hash[0x5]);
		FUGUE512_3(Hash[0x6], Hash[0x7], Hash[0x8]);
		FUGUE512_3(Hash[0x9], Hash[0xA], Hash[0xB]);
		FUGUE512_3(Hash[0xC], Hash[0xD], Hash[0xE]);
		FUGUE512_3(Hash[0xF], 0, 512);

		for (int i = 0; i < 32; i++){
			mROR3;
			CMIX36(S[ 0], S[ 1], S[ 2], S[ 4], S[ 5], S[ 6], S[18], S[19], S[20]);
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
		}
		#pragma unroll
		for (int i = 0; i < 12; i++) {
			S[ 4] ^= S[ 0];	S[ 9] ^= S[ 0];	S[18] ^= S[ 0];	S[27] ^= S[ 0];
			mROR9;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[18] ^= S[ 0];	S[27] ^= S[ 0];
			mROR9;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[19] ^= S[ 0];	S[27] ^= S[ 0];
			mROR9;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[19] ^= S[ 0];	S[28] ^= S[ 0];
			mROR8;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
		}
		S[ 4] ^= S[ 0];	S[ 9] ^= S[ 0];	S[18] ^= S[ 0];	S[27] ^= S[ 0];
		mROR9;
		SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
		S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[18] ^= S[ 0];	S[27] ^= S[ 0];
		mROR9;
		SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
		S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[19] ^= S[ 0];	S[27] ^= S[ 0];
		mROR9;
		SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);

		S[ 3] = cuda_swab32(S[3]);	S[ 4] = cuda_swab32(S[4]^S[ 0]);

		const uint64_t check = *(uint64_t*)&S[ 3];
		if(check <= target){
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}

__host__
void x13_fugue512_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash){

	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x13_fugue512_gpu_hash_64_alexis<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

__host__
void x13_fugue512_cpu_hash_64_final_alexis(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target){

	const uint32_t threadsperblock = 512;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x13_fugue512_gpu_hash_64_final_alexis<<<grid, block>>>(threads, d_hash,d_resNonce,target);
}
