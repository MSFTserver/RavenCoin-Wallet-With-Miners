/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
*/

#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }

//#define SHUFFLE

#if defined(SHUFFLE)

#define TPB 1024

__device__ __forceinline__
void rrounds(uint32_t *x){
	#pragma unroll 16
	for (int r = 0; r < 16; ++r) {
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[ 8] += x[ 0]; x[ 9] += x[ 1];
		x[10] += x[ 2]; x[11] += x[ 3];
		x[12] += x[ 4]; x[13] += x[ 5];
		x[14] += x[ 6]; x[15] += x[ 7];
		x[ 0] = ROTL32(x[ 0], 7);
		x[ 1] = ROTL32(x[ 1], 7);
		x[ 2] = ROTL32(x[ 2], 7);
		x[ 3] = ROTL32(x[ 3], 7);
		x[ 4] = ROTL32(x[ 4], 7);
		x[ 5] = ROTL32(x[ 5], 7);
		x[ 6] = ROTL32(x[ 6], 7);
		x[ 7] = ROTL32(x[ 7], 7);
		/* "swap x_00klm with x_01klm" */
		SWAP(x[ 0], x[ 4]); SWAP(x[ 1], x[ 5]);
		SWAP(x[ 2], x[ 6]); SWAP(x[ 3], x[ 7]);
		
		x[ 0] ^= x[ 8]; x[ 4] ^= x[12];
		x[ 1] ^= x[ 9]; x[ 5] ^= x[13];
		x[ 2] ^= x[10]; x[ 6] ^= x[14];
		x[ 3] ^= x[11]; x[ 7] ^= x[15];
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[ 8],x[10]); SWAP(x[ 9],x[11]);
		SWAP(x[12],x[14]); SWAP(x[13],x[15]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[ 8] += x[ 0]; x[ 9] += x[ 1];
		x[10] += x[ 2]; x[11] += x[ 3];
		x[12] += x[ 4]; x[13] += x[ 5];
		x[14] += x[ 6]; x[15] += x[ 7];
		x[ 0] = ROTL32(x[ 0],11);
		x[ 1] = ROTL32(x[ 1],11);
		x[ 2] = ROTL32(x[ 2],11);
		x[ 3] = ROTL32(x[ 3],11);
		x[ 4] = ROTL32(x[ 4],11);
		x[ 5] = ROTL32(x[ 5],11);
		x[ 6] = ROTL32(x[ 6],11);
		x[ 7] = ROTL32(x[ 7],11);
		/* "swap x_0j0lm with x_0j1lm" */
		x[ 0] = __shfl(x[ 0],threadIdx.x^1);
		x[ 1] = __shfl(x[ 1],threadIdx.x^1);
		x[ 2] = __shfl(x[ 2],threadIdx.x^1);
		x[ 3] = __shfl(x[ 3],threadIdx.x^1);
		x[ 4] = __shfl(x[ 4],threadIdx.x^1);
		x[ 5] = __shfl(x[ 5],threadIdx.x^1);
		x[ 6] = __shfl(x[ 6],threadIdx.x^1);
		x[ 7] = __shfl(x[ 7],threadIdx.x^1);
	
		x[ 0] ^= x[ 8]; x[ 1] ^= x[ 9];
		x[ 2] ^= x[10]; x[ 3] ^= x[11];
		x[ 4] ^= x[12]; x[ 5] ^= x[13];
		x[ 6] ^= x[14]; x[ 7] ^= x[15];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[ 8],x[ 9]); SWAP(x[10],x[11]);
		SWAP(x[12],x[13]); SWAP(x[14],x[15]);
	}
}
__global__ __launch_bounds__(TPB, 1)
void x11_cubehash512_gpu_hash_64(uint32_t threads, uint64_t *g_hash){
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x)>>1;

	const uint32_t even = (threadIdx.x & 1);

	if (thread < threads){
		uint32_t *Hash = (uint32_t*)&g_hash[8 * thread + 2*even];

		uint32_t x[16];

		if(even==0){
			x[ 0] = 0x2AEA2A61;	x[ 1] = 0x50F494D4;	x[ 2] = 0x2D538B8B;	x[ 3] = 0x4167D83E;
			x[ 4] = 0x4D42C787;	x[ 5] = 0xA647A8B3;	x[ 6] = 0x97CF0BEF;	x[ 7] = 0x825B4537;
			x[ 8] = 0xFCD398D9;	x[ 9] = 0x148FE485;	x[10] = 0x1B017BEF;	x[11] = 0xB6444532;
			x[12] = 0xD65C8A2B;	x[13] = 0xA5A70E75;	x[14] = 0xB1C62456;	x[15] = 0xBC796576;
		}else{
			x[ 0] = 0x3FEE2313;	x[ 1] = 0xC701CF8C;	x[ 2] = 0xCC39968E;	x[ 3] = 0x50AC5695;
			x[ 4] = 0xEEF864D2;	x[ 5] = 0xF22090C4;	x[ 6] = 0xD0E5CD33;	x[ 7] = 0xA23911AE;
			x[ 8] = 0x6A536159;	x[ 9] = 0x2FF5781C;	x[10] = 0x91FA7934;	x[11] = 0x0DBADEA9;
			x[12] = 0x1921C8F7;	x[13] = 0xE7989AF1; 	x[14] = 0x7795D246;	x[15] = 0xD43E3B44;
		}
		*(uint4*)&x[ 0]^= __ldg((uint4*)&Hash[ 0]);
		rrounds(x);

		*(uint4*)&x[ 0]^= __ldg((uint4*)&Hash[ 8]);

		rrounds(x);
		
		if(!even)
			x[ 0] ^= 0x80;

		rrounds(x);
		/* "the integer 1 is xored into the last state word x_11111" */
		if(even)
			x[15] ^= 1;
	
		#pragma unroll 10
		for (int i = 0; i < 10; ++i)
			rrounds(x);
	
		*(uint4*)&Hash[ 0] = *(uint4*)&x[ 0];
		*(uint4*)&Hash[ 8] = *(uint4*)&x[ 4];
//		g_hash[thread + (2*even+0) * threads]	= *(uint2*)&x[ 0];
//		g_hash[thread + (2*even+1) * threads]	= *(uint2*)&x[ 2];
	}
}
__host__
void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((2*threads + TPB-1)/TPB);
    dim3 block(TPB);

    x11_cubehash512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);

}

#else

#define TPB 768

__device__ __forceinline__
static void rrounds(uint32_t *x){
	#pragma unroll 2
	for (int r = 0; r < 16; r++) {
		/* "add x_0jklm into x_1jklmn modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0], 7);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1], 7);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2], 7);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3], 7);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4], 7);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5], 7);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6], 7);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7], 7);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8], 7);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9], 7);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10], 7);x[27] = x[27] + x[11];x[11] = ROTL32(x[11], 7);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12], 7);x[29] = x[29] + x[13];x[13] = ROTL32(x[13], 7);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14], 7);x[31] = x[31] + x[15];x[15] = ROTL32(x[15], 7);
		/* "swap x_00klm with x_01klm" */
		SWAP(x[ 0], x[ 8]);x[ 0] ^= x[16];x[ 8] ^= x[24];SWAP(x[ 1], x[ 9]);x[ 1] ^= x[17];x[ 9] ^= x[25];
		SWAP(x[ 2], x[10]);x[ 2] ^= x[18];x[10] ^= x[26];SWAP(x[ 3], x[11]);x[ 3] ^= x[19];x[11] ^= x[27];
		SWAP(x[ 4], x[12]);x[ 4] ^= x[20];x[12] ^= x[28];SWAP(x[ 5], x[13]);x[ 5] ^= x[21];x[13] ^= x[29];
		SWAP(x[ 6], x[14]);x[ 6] ^= x[22];x[14] ^= x[30];SWAP(x[ 7], x[15]);x[ 7] ^= x[23];x[15] ^= x[31];
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[16], x[18]); SWAP(x[17], x[19]); SWAP(x[20], x[22]); SWAP(x[21], x[23]);SWAP(x[24], x[26]); SWAP(x[25], x[27]); SWAP(x[28], x[30]); SWAP(x[29], x[31]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0],11);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1],11);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2],11);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3],11);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4],11);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5],11);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6],11);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7],11);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8],11);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9],11);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10],11);x[27] = x[27] + x[11];x[11] = ROTL32(x[11],11);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12],11);x[29] = x[29] + x[13];x[13] = ROTL32(x[13],11);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14],11);x[31] = x[31] + x[15];x[15] = ROTL32(x[15],11);
		/* "swap x_0j0lm with x_0j1lm" */
		SWAP(x[ 0], x[ 4]); x[ 0] ^= x[16]; x[ 4] ^= x[20]; SWAP(x[ 1], x[ 5]); x[ 1] ^= x[17]; x[ 5] ^= x[21];
		SWAP(x[ 2], x[ 6]); x[ 2] ^= x[18]; x[ 6] ^= x[22]; SWAP(x[ 3], x[ 7]); x[ 3] ^= x[19]; x[ 7] ^= x[23];
		SWAP(x[ 8], x[12]); x[ 8] ^= x[24]; x[12] ^= x[28]; SWAP(x[ 9], x[13]); x[ 9] ^= x[25]; x[13] ^= x[29];
		SWAP(x[10], x[14]); x[10] ^= x[26]; x[14] ^= x[30]; SWAP(x[11], x[15]); x[11] ^= x[27]; x[15] ^= x[31];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[16], x[17]); SWAP(x[18], x[19]); SWAP(x[20], x[21]); SWAP(x[22], x[23]);SWAP(x[24], x[25]); SWAP(x[26], x[27]); SWAP(x[28], x[29]); SWAP(x[30], x[31]);
	}
}

/***************************************************/
// GPU Hash Function
__global__ __launch_bounds__(TPB)
void x11_cubehash512_gpu_hash_64(uint32_t threads, uint64_t *g_hash){

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		uint32_t *Hash = (uint32_t*)&g_hash[8 * thread];

		uint32_t x[32] = {
			0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E,
			0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
			0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537,
			0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
			0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532,
			0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
			0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576,
			0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
		};
	
		// erste Hälfte des Hashes (32 bytes)
		//Update32(x, (const BitSequence*)Hash);
		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[0]);

		rrounds(x);

		// zweite Hälfte des Hashes (32 bytes)
	//        Update32(x, (const BitSequence*)(Hash+8));
		*(uint2x4*)&x[ 0] ^= __ldg4((uint2x4*)&Hash[8]);
		
		rrounds(x);

		// Padding Block
		x[ 0] ^= 0x80;
		rrounds(x);
	
	//	Final(x, (BitSequence*)Hash);
		x[31] ^= 1;

		/* "the state is then transformed invertibly through 10r identical rounds" */
		#pragma unroll 10
		for (int i = 0;i < 10;++i)
			rrounds(x);

		/* "output the first h/8 bytes of the state" */
		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&x[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&x[ 8];
	}
}


__host__
void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + TPB-1)/TPB);
    dim3 block(TPB);

    x11_cubehash512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);

}
#endif
