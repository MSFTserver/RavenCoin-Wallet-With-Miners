/*
 * Shabal-512 for X14/X15
 * Provos Alexis - 2016
 */
#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

/* $Id: shabal.c 175 2010-05-07 16:03:20Z tp $ */
/*
 * Shabal implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2007-2010 Projet RNRT SAPHIR
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS B[14] LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author Thomas Pornin <thomas.pornin@cryptolog.com>
 */

__device__ __forceinline__ void PERM_ELT(uint32_t &xa0,const uint32_t xa1,uint32_t &xb0,const uint32_t xb1,const uint32_t xb2,const uint32_t xb3,const uint32_t xc,const uint32_t xm){

		uint32_t tmp;
		#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
			asm ("lop3.b32 %0, %1, %2, %3, 0x9A;" : "=r"(tmp)	: "r"(xb2),"r"(xb3),"r"(xm));		// 0x9A = (F0 &(~CC)) ^ (AA)
		#else
			tmp = (xb2 & ~xb3) ^ xm;
		#endif

		xa0 = ((xa0 ^ xc ^ (ROTL32(xa1, 15) * 5U)) * 3U) ^ xb1 ^ tmp;
		xb0 = xor3x(0xFFFFFFFF, xa0, ROTL32(xb0, 1));
}

__device__ __forceinline__
void PERM_STEP_0(uint32_t *A,uint32_t *B,const uint32_t *C,const uint32_t* M){
		PERM_ELT(A[ 0], A[11], B[ 0], B[13], B[ 9], B[ 6], C[ 8], M[ 0]); PERM_ELT(A[ 1], A[ 0], B[ 1], B[14], B[10], B[ 7], C[ 7], M[ 1]);
		PERM_ELT(A[ 2], A[ 1], B[ 2], B[15], B[11], B[ 8], C[ 6], M[ 2]); PERM_ELT(A[ 3], A[ 2], B[ 3], B[ 0], B[12], B[ 9], C[ 5], M[ 3]);
		PERM_ELT(A[ 4], A[ 3], B[ 4], B[ 1], B[13], B[10], C[ 4], M[ 4]); PERM_ELT(A[ 5], A[ 4], B[ 5], B[ 2], B[14], B[11], C[ 3], M[ 5]);
		PERM_ELT(A[ 6], A[ 5], B[ 6], B[ 3], B[15], B[12], C[ 2], M[ 6]); PERM_ELT(A[ 7], A[ 6], B[ 7], B[ 4], B[ 0], B[13], C[ 1], M[ 7]);
		PERM_ELT(A[ 8], A[ 7], B[ 8], B[ 5], B[ 1], B[14], C[ 0], M[ 8]); PERM_ELT(A[ 9], A[ 8], B[ 9], B[ 6], B[ 2], B[15], C[15], M[ 9]);
		PERM_ELT(A[10], A[ 9], B[10], B[ 7], B[ 3], B[ 0], C[14], M[10]); PERM_ELT(A[11], A[10], B[11], B[ 8], B[ 4], B[ 1], C[13], M[11]);
		PERM_ELT(A[ 0], A[11], B[12], B[ 9], B[ 5], B[ 2], C[12], M[12]); PERM_ELT(A[ 1], A[ 0], B[13], B[10], B[ 6], B[ 3], C[11], M[13]);
		PERM_ELT(A[ 2], A[ 1], B[14], B[11], B[ 7], B[ 4], C[10], M[14]); PERM_ELT(A[ 3], A[ 2], B[15], B[12], B[ 8], B[ 5], C[ 9], M[15]);
}

__device__ __forceinline__
void PERM_STEP_1(uint32_t *A,uint32_t *B,const uint32_t *C,const uint32_t* M){
		PERM_ELT(A[ 4], A[ 3], B[ 0], B[13], B[ 9], B[ 6], C[ 8], M[ 0]); PERM_ELT(A[ 5], A[ 4], B[ 1], B[14], B[10], B[ 7], C[ 7], M[ 1]);
		PERM_ELT(A[ 6], A[ 5], B[ 2], B[15], B[11], B[ 8], C[ 6], M[ 2]); PERM_ELT(A[ 7], A[ 6], B[ 3], B[ 0], B[12], B[ 9], C[ 5], M[ 3]);
		PERM_ELT(A[ 8], A[ 7], B[ 4], B[ 1], B[13], B[10], C[ 4], M[ 4]); PERM_ELT(A[ 9], A[ 8], B[ 5], B[ 2], B[14], B[11], C[ 3], M[ 5]);
		PERM_ELT(A[10], A[ 9], B[ 6], B[ 3], B[15], B[12], C[ 2], M[ 6]); PERM_ELT(A[11], A[10], B[ 7], B[ 4], B[ 0], B[13], C[ 1], M[ 7]);
		PERM_ELT(A[ 0], A[11], B[ 8], B[ 5], B[ 1], B[14], C[ 0], M[ 8]); PERM_ELT(A[ 1], A[ 0], B[ 9], B[ 6], B[ 2], B[15], C[15], M[ 9]);
		PERM_ELT(A[ 2], A[ 1], B[10], B[ 7], B[ 3], B[ 0], C[14], M[10]); PERM_ELT(A[ 3], A[ 2], B[11], B[ 8], B[ 4], B[ 1], C[13], M[11]);
		PERM_ELT(A[ 4], A[ 3], B[12], B[ 9], B[ 5], B[ 2], C[12], M[12]); PERM_ELT(A[ 5], A[ 4], B[13], B[10], B[ 6], B[ 3], C[11], M[13]);
		PERM_ELT(A[ 6], A[ 5], B[14], B[11], B[ 7], B[ 4], C[10], M[14]); PERM_ELT(A[ 7], A[ 6], B[15], B[12], B[ 8], B[ 5], C[ 9], M[15]);
}

__device__ __forceinline__
void PERM_STEP_2(uint32_t *A,uint32_t *B,const uint32_t *C,const uint32_t* M){
		PERM_ELT(A[ 8], A[ 7], B[ 0], B[13], B[ 9], B[ 6], C[ 8], M[ 0]); PERM_ELT(A[ 9], A[ 8], B[ 1], B[14], B[10], B[ 7], C[ 7], M[ 1]);
		PERM_ELT(A[10], A[ 9], B[ 2], B[15], B[11], B[ 8], C[ 6], M[ 2]); PERM_ELT(A[11], A[10], B[ 3], B[ 0], B[12], B[ 9], C[ 5], M[ 3]);
		PERM_ELT(A[ 0], A[11], B[ 4], B[ 1], B[13], B[10], C[ 4], M[ 4]); PERM_ELT(A[ 1], A[ 0], B[ 5], B[ 2], B[14], B[11], C[ 3], M[ 5]);
		PERM_ELT(A[ 2], A[ 1], B[ 6], B[ 3], B[15], B[12], C[ 2], M[ 6]); PERM_ELT(A[ 3], A[ 2], B[ 7], B[ 4], B[ 0], B[13], C[ 1], M[ 7]);
		PERM_ELT(A[ 4], A[ 3], B[ 8], B[ 5], B[ 1], B[14], C[ 0], M[ 8]); PERM_ELT(A[ 5], A[ 4], B[ 9], B[ 6], B[ 2], B[15], C[15], M[ 9]);
		PERM_ELT(A[ 6], A[ 5], B[10], B[ 7], B[ 3], B[ 0], C[14], M[10]); PERM_ELT(A[ 7], A[ 6], B[11], B[ 8], B[ 4], B[ 1], C[13], M[11]);
		PERM_ELT(A[ 8], A[ 7], B[12], B[ 9], B[ 5], B[ 2], C[12], M[12]); PERM_ELT(A[ 9], A[ 8], B[13], B[10], B[ 6], B[ 3], C[11], M[13]);
		PERM_ELT(A[10], A[ 9], B[14], B[11], B[ 7], B[ 4], C[10], M[14]); PERM_ELT(A[11], A[10], B[15], B[12], B[ 8], B[ 5], C[ 9], M[15]);
}

__device__ __forceinline__
void ADD_BLOCK(uint32_t* A, const uint32_t *B){
	A[11]+= B[ 6]; A[10]+= B[ 5]; A[ 9]+= B[ 4]; A[ 8]+= B[ 3]; A[ 7]+= B[ 2]; A[ 6]+= B[ 1]; A[ 5]+= B[ 0]; A[ 4]+= B[15]; A[ 3]+= B[14]; A[ 2]+= B[13]; A[ 1]+= B[12]; A[ 0]+= B[11];
	A[11]+= B[10]; A[10]+= B[ 9]; A[ 9]+= B[ 8]; A[ 8]+= B[ 7]; A[ 7]+= B[ 6]; A[ 6]+= B[ 5]; A[ 5]+= B[ 4]; A[ 4]+= B[ 3]; A[ 3]+= B[ 2]; A[ 2]+= B[ 1]; A[ 1]+= B[ 0]; A[ 0]+= B[15];
	A[11]+= B[14]; A[10]+= B[13]; A[ 9]+= B[12]; A[ 8]+= B[11]; A[ 7]+= B[10]; A[ 6]+= B[ 9]; A[ 5]+= B[ 8]; A[ 4]+= B[ 7]; A[ 3]+= B[ 6]; A[ 2]+= B[ 5]; A[ 1]+= B[ 4]; A[ 0]+= B[ 3];
}
__device__ __forceinline__
void ROTATE(uint32_t* A){
	#pragma unroll 16
	for(int i=0;i<16;i++){
		A[ i] = ROTL32(A[ i],17);
	}
}
/***************************************************/
// GPU Hash Function
__global__ __launch_bounds__(384,3)
void x14_shabal512_gpu_hash_64_alexis(uint32_t threads, uint32_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint32_t A[]={
			0x20728DFD, 0x46C0BD53, 0xE782B699, 0x55304632, 0x71B4EF90, 0x0EA9E82C, 0xDBB930F1, 0xFAD06B8B,
			0xBE0CAE40, 0x8BD14410, 0x76D2ADAC, 0x28ACAB7F};
	uint32_t B[]={
			0xC1099CB7, 0x07B385F3, 0xE7442C26, 0xCC8AD640, 0xEB6F56C7, 0x1EA81AA9, 0x73B9D314, 0x1DE85D08,
			0x48910A5A, 0x893B22DB, 0xC5A0DF44, 0xBBC4324E, 0x72D2F240, 0x75941D99, 0x6D8BDE82, 0xA1A7502B};
	uint32_t C[]={
			0xD9BF68D1, 0x58BAD750, 0x56028CB2, 0x8134F359, 0xB5D469D8, 0x941A8CC2, 0x418B2A6E, 0x04052780,
			0x7F07D787, 0x5194358F, 0x3C60D665, 0xBE97D79A, 0x950C3434, 0xAED9A06D, 0x2537DC8D, 0x7CDB5969};
	uint32_t M[16];

	if (thread < threads){

		uint32_t *Hash = &g_hash[thread<<4];

		*(uint2x4*)&M[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&M[ 8] = __ldg4((uint2x4*)&Hash[ 8]);

		*(uint16*)&B[ 0]+= *(uint16*)&M[ 0];
		A[ 0] ^= 1;
		ROTATE(B);
		PERM_STEP_0(A,B,C,M);
		PERM_STEP_1(A,B,C,M);
		PERM_STEP_2(A,B,C,M);
		ADD_BLOCK(A,C);
		*(uint16*)&C[ 0]-= *(uint16*)&M[ 0];
//		SWAP_BC;

		M[ 0] = 0x80;
		M[ 1] = M[ 2] = M[ 3] = M[ 4] = M[ 5] = M[ 6] = M[ 7] = M[ 8] = M[ 9] = M[10] = M[11] = M[12] = M[13] = M[14] = M[15] = 0;
		C[ 0]+= M[ 0];
		A[ 0]^= 0x02;
		ROTATE(C);
		PERM_STEP_0(A,C,B,M);
		PERM_STEP_1(A,C,B,M);
		PERM_STEP_2(A,C,B,M);
		ADD_BLOCK(A,B);
		A[ 0] ^= 0x02;
		ROTATE(B);
		PERM_STEP_0(A,B,C,M);
		PERM_STEP_1(A,B,C,M);
		PERM_STEP_2(A,B,C,M);
		ADD_BLOCK(A,C);
		A[ 0] ^= 0x02;
		ROTATE(C);
		PERM_STEP_0(A,C,B,M);
		PERM_STEP_1(A,C,B,M);
		PERM_STEP_2(A,C,B,M);
		ADD_BLOCK(A,B);
		A[ 0] ^= 0x02;
		ROTATE(B);
		PERM_STEP_0(A,B,C,M);
		PERM_STEP_1(A,B,C,M);
		PERM_STEP_2(A,B,C,M);

		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&B[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&B[ 8];
	}
}

__host__ void x14_shabal512_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 384;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x14_shabal512_gpu_hash_64_alexis<<<grid, block>>>(threads, d_hash);
}

__global__ __launch_bounds__(512,2)
void x14_shabal512_gpu_hash_64_final_alexis(uint32_t threads,const uint32_t* __restrict__ g_hash,uint32_t* resNonce, const uint64_t target){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint32_t A[]={
			0x20728DFD, 0x46C0BD53, 0xE782B699, 0x55304632, 0x71B4EF90, 0x0EA9E82C, 0xDBB930F1, 0xFAD06B8B,
			0xBE0CAE40, 0x8BD14410, 0x76D2ADAC, 0x28ACAB7F};
	uint32_t B[]={
			0xC1099CB7, 0x07B385F3, 0xE7442C26, 0xCC8AD640, 0xEB6F56C7, 0x1EA81AA9, 0x73B9D314, 0x1DE85D08,
			0x48910A5A, 0x893B22DB, 0xC5A0DF44, 0xBBC4324E, 0x72D2F240, 0x75941D99, 0x6D8BDE82, 0xA1A7502B};
	uint32_t C[]={
			0xD9BF68D1, 0x58BAD750, 0x56028CB2, 0x8134F359, 0xB5D469D8, 0x941A8CC2, 0x418B2A6E, 0x04052780,
			0x7F07D787, 0x5194358F, 0x3C60D665, 0xBE97D79A, 0x950C3434, 0xAED9A06D, 0x2537DC8D, 0x7CDB5969};
	uint32_t M[16];

	if (thread < threads){

		const uint32_t *Hash = &g_hash[thread<<4];

		*(uint2x4*)&M[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&M[ 8] = __ldg4((uint2x4*)&Hash[ 8]);

		*(uint16*)&B[ 0]+= *(uint16*)&M[ 0];
		A[ 0] ^= 1;
		ROTATE(B);
		PERM_STEP_0(A,B,C,M);
		PERM_STEP_1(A,B,C,M);
		PERM_STEP_2(A,B,C,M);
		ADD_BLOCK(A,C);
		*(uint16*)&C[ 0]-= *(uint16*)&M[ 0];
//		SWAP_BC;

		M[ 0] = 0x80;
		M[ 1] = M[ 2] = M[ 3] = M[ 4] = M[ 5] = M[ 6] = M[ 7] = M[ 8] = M[ 9] = M[10] = M[11] = M[12] = M[13] = M[14] = M[15] = 0;
		C[ 0]+= M[ 0];
		A[ 0]^= 0x02;
		ROTATE(C);
		PERM_STEP_0(A,C,B,M);
		PERM_STEP_1(A,C,B,M);
		PERM_STEP_2(A,C,B,M);
		ADD_BLOCK(A,B);
		A[ 0] ^= 0x02;
		ROTATE(B);
		PERM_STEP_0(A,B,C,M);
		PERM_STEP_1(A,B,C,M);
		PERM_STEP_2(A,B,C,M);
		ADD_BLOCK(A,C);
		A[ 0] ^= 0x02;
		ROTATE(C);
		PERM_STEP_0(A,C,B,M);
		PERM_STEP_1(A,C,B,M);
		PERM_STEP_2(A,C,B,M);
		ADD_BLOCK(A,B);
		A[ 0] ^= 0x02;
		ROTATE(B);
		PERM_STEP_0(A,B,C,M);
		PERM_STEP_1(A,B,C,M);
//		PERM_STEP_2(A,B,C,M);
		PERM_ELT(A[ 8], A[ 7], B[ 0], B[13], B[ 9], B[ 6], C[ 8], M[ 0]); PERM_ELT(A[ 9], A[ 8], B[ 1], B[14], B[10], B[ 7], C[ 7], M[ 1]);
		PERM_ELT(A[10], A[ 9], B[ 2], B[15], B[11], B[ 8], C[ 6], M[ 2]); PERM_ELT(A[11], A[10], B[ 3], B[ 0], B[12], B[ 9], C[ 5], M[ 3]);
		PERM_ELT(A[ 0], A[11], B[ 4], B[ 1], B[13], B[10], C[ 4], M[ 4]); PERM_ELT(A[ 1], A[ 0], B[ 5], B[ 2], B[14], B[11], C[ 3], M[ 5]);
		PERM_ELT(A[ 2], A[ 1], B[ 6], B[ 3], B[15], B[12], C[ 2], M[ 6]); PERM_ELT(A[ 3], A[ 2], B[ 7], B[ 4], B[ 0], B[13], C[ 1], M[ 7]);

		if(*(uint64_t*)&B[ 6] <= target){
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}

__host__ void x14_shabal512_cpu_hash_64_final_alexis(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target){

	const uint32_t threadsperblock = 512;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x14_shabal512_gpu_hash_64_final_alexis<<<grid, block>>>(threads, d_hash, d_resNonce, target);
}
