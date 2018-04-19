#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
/* reduce vstudio warnings (__byteperm, blockIdx...) */
#include <device_functions.h>
#include <device_launch_parameters.h>
#define __launch_bounds__(max_tpb, min_blocks)
#endif

#include <stdbool.h>
#include <stdint.h>

#ifndef UINT32_MAX
/* slackware need that */
#define UINT32_MAX UINT_MAX
#endif

#ifndef MAX_GPUS
#define MAX_GPUS 16
#endif

extern "C" short device_map[MAX_GPUS];
extern "C"  long device_sm[MAX_GPUS];

extern int cuda_arch[MAX_GPUS];

// common functions
extern int cuda_get_arch(int thr_id);
extern void cuda_check_cpu_init(int thr_id, uint32_t threads);
extern void cuda_check_cpu_free(int thr_id);
extern void cuda_check_cpu_setTarget(const void *ptarget);
extern uint32_t cuda_check_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash);
extern uint32_t cuda_check_hash_suppl(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash, uint8_t numNonce);
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);
extern void cudaReportHardwareFailure(int thr_id, cudaError_t error, const char* func);
extern __device__ __device_builtin__ void __syncthreads(void);
extern __device__ __device_builtin__ void __threadfence(void);

#ifndef __CUDA_ARCH__
// define blockDim and threadIdx for host
extern const dim3 blockDim;
extern const uint3 threadIdx;
#endif

#ifndef SPH_C32
#define SPH_C32(x) (x)
// #define SPH_C32(x) ((uint32_t)(x ## U))
#endif

#ifndef SPH_C64
#define SPH_C64(x) (x)
// #define SPH_C64(x) ((uint64_t)(x ## ULL))
#endif

#ifndef SPH_T32
#define SPH_T32(x) (x)
// #define SPH_T32(x) ((x) & SPH_C32(0xFFFFFFFF))
#endif

#ifndef SPH_T64
#define SPH_T64(x) (x)
// #define SPH_T64(x) ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))
#endif

#if __CUDA_ARCH__ < 320
// Host and Compute 3.0
#define ROTL32(x, n) SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define __ldg(x) (*(x))
#else
// Compute 3.2+
#define ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
#define ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
#endif

#define AS_U32(addr)   *((uint32_t*)(addr))
#define AS_U64(addr)   *((uint64_t*)(addr))
#define AS_UINT2(addr) *((uint2*)(addr))
#define AS_UINT4(addr) *((uint4*)(addr))
#define AS_UL2(addr)   *((ulonglong2*)(addr))

/*********************************************************************/
// Macros to catch CUDA errors in CUDA runtime calls

#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		fprintf(stderr, "Cuda error in func '%s' at line %i : %s.\n", \
		         __FUNCTION__, __LINE__, cudaGetErrorString(err) );   \
		exit(EXIT_FAILURE);                                           \
	}                                                                 \
} while (0)

#define CUDA_CALL_OR_RET(call) do {                                   \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		cudaReportHardwareFailure(thr_id, err, __FUNCTION__);         \
		return;                                                       \
	}                                                                 \
} while (0)

#define CUDA_CALL_OR_RET_X(call, ret) do {                            \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		cudaReportHardwareFailure(thr_id, err, __FUNCTION__);         \
		return ret;                                                   \
	}                                                                 \
} while (0)

/*********************************************************************/

__device__ __forceinline__ uint64_t MAKE_ULONGLONG(uint32_t LO, uint32_t HI){
	return __double_as_longlong(__hiloint2double(HI, LO));
//	return (uint64_t)LO | (((uint64_t)HI) << 32);
}

// das Hi Word in einem 64 Bit Typen ersetzen
__device__ __forceinline__ uint64_t REPLACE_HIDWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFFULL) | (((uint64_t)y) << 32U);
}

// das Lo Word in einem 64 Bit Typen ersetzen
__device__ __forceinline__ uint64_t REPLACE_LODWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFF00000000ULL) | ((uint64_t)y);
}

// Endian Drehung für 32 Bit Typen
#if defined(__CUDA_ARCH__)
__device__ __forceinline__ uint32_t cuda_swab32(uint32_t x)
{
	/* device */
	return __byte_perm(x, x, 0x0123);
}
#else
	/* host */
	#define cuda_swab32(x) \
	((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | \
		(((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#endif

// das Lo Word aus einem 64 Bit Typen extrahieren
__device__ __forceinline__ uint32_t _LODWORD(const uint64_t &x) {
	return (uint32_t)__double2loint(__longlong_as_double(x));
//	return (uint32_t)(x & 0xFFFFFFFFULL);
}

// das Hi Word aus einem 64 Bit Typen extrahieren
__device__ __forceinline__ uint32_t _HIDWORD(const uint64_t &x) {
	return (uint32_t)__double2hiint(__longlong_as_double(x));
//	return (uint32_t)(x >> 32);
}


__device__ __forceinline__ uint2 cuda_swab64_U2(uint2 a)
{
	// Input:       77665544 33221100
	// Output:      00112233 44556677
	uint2 result;
	result.y = __byte_perm(a.x, 0, 0x0123);
	result.x = __byte_perm(a.y, 0, 0x0123);
	return result;
}

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ uint64_t cuda_swab64(uint64_t x)
{
	// Input:       77665544 33221100
	// Output:      00112233 44556677
	uint64_t result = __byte_perm((uint32_t) x, 0, 0x0123);
	return (result << 32) | __byte_perm(_HIDWORD(x), 0, 0x0123);
}
#else
/* host */
#define cuda_swab64(x) \
		((uint64_t)((((uint64_t)(x) & 0xff00000000000000ULL) >> 56) | \
			(((uint64_t)(x) & 0x00ff000000000000ULL) >> 40) | \
			(((uint64_t)(x) & 0x0000ff0000000000ULL) >> 24) | \
			(((uint64_t)(x) & 0x000000ff00000000ULL) >>  8) | \
			(((uint64_t)(x) & 0x00000000ff000000ULL) <<  8) | \
			(((uint64_t)(x) & 0x0000000000ff0000ULL) << 24) | \
			(((uint64_t)(x) & 0x000000000000ff00ULL) << 40) | \
			(((uint64_t)(x) & 0x00000000000000ffULL) << 56)))
#endif

// swap two uint32_t without extra registers
__device__ __host__ __forceinline__ void xchg(uint32_t &x, uint32_t &y) {
	x ^= y; y = x ^ y; x ^= y;
}
// for other types...
#define XCHG(x, y) { x ^= y; y = x ^ y; x ^= y; }

static __host__ __device__ __forceinline__ uint2 vectorize(uint64_t v) {
	uint2 result;
#if defined(__CUDA_ARCH__)
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(result.x), "=r"(result.y) : "l"(v));
#else
	result.x = (uint32_t)(v);
	result.y = (uint32_t)(v >> 32);
#endif
	return result;
}

static __host__ __device__ __forceinline__ uint64_t devectorize(uint2 v) {
#if defined(__CUDA_ARCH__)
	return MAKE_ULONGLONG(v.x, v.y);
#else
	return (((uint64_t)v.y) << 32) + v.x;
#endif
}

#if defined(__CUDA_ARCH__)
	// Compute 3.2+
	#define ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
	#define ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
#else
	// Host and Compute 3.0
	#define ROTL32(x, n) SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
	#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
	#define __ldg(x) (*(x))
#endif

__device__ __forceinline__
uint32_t ROL16(const uint32_t a){
	return __byte_perm(a, 0, 0x1032);
}
__device__ __forceinline__ 
uint32_t ROL8(const uint32_t a){
	return __byte_perm(a, 0, 0x2103);
}
__device__ __forceinline__ 
uint32_t ROR8(const uint32_t a){
	return __byte_perm(a, 0, 0x0321);
}

// device asm for whirpool
__device__ __forceinline__
uint64_t xor1(uint64_t a, uint64_t b)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
	return result;
}

// device asm for whirpool
__device__ __forceinline__
uint64_t xor3(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("xor.b64 %0, %2, %3;\n\t"
	    "xor.b64 %0, %0, %1;\n\t"
		/* output : input registers */
		: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
}

// device asm for whirpool
__device__ __forceinline__
uint64_t xor5(uint64_t a, uint64_t b, uint64_t c, uint64_t d,uint64_t e)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(d) ,"l"(e));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(c));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(b));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(a));
	return result;
}

__device__ __forceinline__
uint64_t xor9(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d, const uint64_t e, const uint64_t f, const uint64_t g, const  uint64_t h,const  uint64_t i)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(h) ,"l"(i));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(g));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(f));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(e));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(d));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(c));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(b));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(a));	
	return result;
}

__device__ __forceinline__
uint64_t xor8(uint64_t a, uint64_t b, uint64_t c, uint64_t d,uint64_t e,uint64_t f,uint64_t g, uint64_t h)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(g) ,"l"(h));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(f));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(e));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(d));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(c));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(b));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(a));
	return result;
}

static __device__ __forceinline__ uint2 xorswap32(uint2 u, uint2 v)
{
	uint2 result;
	result.y = u.x ^ v.x;
	result.x = u.y ^ v.y;
	return result;
}

// device asm for x17
__device__ __forceinline__
uint64_t andor(const uint64_t a,const uint64_t b,const uint64_t c)
{
	uint64_t result;
	asm("{\n\t"
		".reg .u64 m,n;\n\t"
		"and.b64 m,  %1, %2;\n\t"
		" or.b64 n,  %1, %2;\n\t"
		"and.b64 %0, n,  %3;\n\t"
		" or.b64 %0, %0, m ;\n\t"
	"}\n"
	: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
//	return ((a | b) & c) | (a & b);
}

// device asm for x17
__device__ __forceinline__
uint64_t shr_u64(const uint64_t x, uint32_t n){
	uint64_t result;
	asm ("shr.b64 %0,%1,%2;\n\t" : "=l"(result) : "l"(x), "r"(n));
	return result;
//	return x >> n;
}

__device__ __forceinline__
uint64_t shl_u64(const uint64_t x, uint32_t n){
	uint64_t result;
	asm("shl.b64 %0,%1,%2;\n\t" : "=l"(result) : "l"(x), "r"(n));
	return result;
//	return x << n;
}

__device__ __forceinline__
uint32_t shr_u32(const uint32_t x,uint32_t n) {
	uint32_t result;
	asm("shr.b32 %0,%1,%2;"	: "=r"(result) : "r"(x), "r"(n));
	return result;
//	return x >> n;
}

__device__ __forceinline__
uint32_t shl_u32(const uint32_t x,uint32_t n) {
	uint32_t result;
	asm("shl.b32 %0,%1,%2;" : "=r"(result) : "r"(x), "r"(n));
	return result;
//	return x << n;
}

// 64-bit ROTATE RIGHT
#if defined(__CUDA_ARCH__)
/* complicated sm >= 3.5 one (with Funnel Shifter beschleunigt), to bench */
__device__ __forceinline__
uint64_t ROTR64(const uint64_t value, const int offset) {
	uint2 result;
	const uint2 tmp = vectorize(value);
		
	if(offset == 8) {
		result.x = __byte_perm(tmp.x, tmp.y, 0x4321);
		result.y = __byte_perm(tmp.y, tmp.x, 0x4321);
	}
	else if(offset == 16) {
		result.x = __byte_perm(tmp.x, tmp.y, 0x5432);
		result.y = __byte_perm(tmp.y, tmp.x, 0x5432);	
	}	
	else if(offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(tmp.x), "r"(tmp.y), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(tmp.y), "r"(tmp.x), "r"(offset));
	} else {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(tmp.y), "r"(tmp.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(tmp.x), "r"(tmp.y), "r"(offset));
	}
	return devectorize(result);
}
#else
/* host */
#define ROTR64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#endif

// 64-bit ROTATE LEFT
#if defined(__CUDA_ARCH__)
__device__ __forceinline__
uint64_t ROTL64(const uint64_t value, const int offset) {
	uint2 result;
	const uint2 tmp = vectorize(value);
	if(offset == 8){
		result.x = __byte_perm(tmp.x, tmp.y, 0x2107);
		result.y = __byte_perm(tmp.y, tmp.x, 0x2107);
	}
	else if(offset == 16) {
		result.x = __byte_perm(tmp.x, tmp.y, 0x1076);
		result.y = __byte_perm(tmp.y, tmp.x, 0x1076);
	}
	else if(offset == 24) {
		result.x = __byte_perm(tmp.x, tmp.y, 0x0765);
		result.y = __byte_perm(tmp.y, tmp.x, 0x0765);
	}
	else if(offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(tmp.x), "r"(tmp.y), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(tmp.y), "r"(tmp.x), "r"(offset));
	} else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(tmp.y), "r"(tmp.x), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(tmp.x), "r"(tmp.y), "r"(offset));
	}
	return devectorize(result);
}
#else
/* host */
#define ROTL64(x, n)  (((x) << (n)) | ((x) >> (64 - (n))))
#endif

__device__ __forceinline__
uint64_t SWAPDWORDS(uint64_t value){
	uint2 temp;
	asm("mov.b64 {%0, %1}, %2; ": "=r"(temp.x), "=r"(temp.y) : "l"(value));
	asm("mov.b64 %0, {%1, %2}; ": "=l"(value) : "r"(temp.y), "r"(temp.x));
	return value;
}

__device__ __forceinline__
uint2 SWAPDWORDS2(uint2 value){
	return make_uint2(value.y, value.x);
}

/* lyra2/bmw - uint2 vector's operators */

__device__ __forceinline__ 
uint2 SHL8(const uint2 a){
	uint2 result;
	result.y = __byte_perm(a.y, a.x, 0x2107);
	result.x = __byte_perm(a.x, 0, 0x2107);

	return result;
}

__device__ __forceinline__
void LOHI(uint32_t &lo, uint32_t &hi, uint64_t x) {
#if defined(__CUDA_ARCH__)
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(lo), "=r"(hi) : "l"(x));
#else
	lo = (uint32_t)(x);
	hi = (uint32_t)(x >> 32);
#endif
}

/**
 * uint2 direct ops by c++ operator definitions
 */
static __device__ __forceinline__ uint2 operator^ (const uint2 a,const uint32_t b) { return make_uint2(a.x^ b, a.y); }
static __device__ __forceinline__ uint2 operator^ (const uint2 a,const uint2 b) { return make_uint2(a.x ^ b.x, a.y ^ b.y); }
static __device__ __forceinline__ uint2 operator& (const uint2 a,const uint2 b) { return make_uint2(a.x & b.x, a.y & b.y); }
static __device__ __forceinline__ uint2 operator| (const uint2 a,const uint2 b) { return make_uint2(a.x | b.x, a.y | b.y); }
static __device__ __forceinline__ uint2 operator~ (const uint2 a) { return make_uint2(~a.x, ~a.y); }
static __device__ __forceinline__ void operator^= (uint2 &a,const uint2 b) { a = a ^ b; }

static __device__ __forceinline__ uint2 operator+ (const uint2 a,const uint2 b) {
#if defined(__CUDA_ARCH__) && CUDA_VERSION < 7000
	uint2 result;
	asm("{\n\t"
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
	"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
#else
	return vectorize(devectorize(a) + devectorize(b));
#endif
}

static __device__ __forceinline__ uint2 operator+ (const uint2 a,const uint64_t b) {
	return vectorize(devectorize(a) + b);
}

static __device__ __forceinline__ void operator+= (uint2 &a,const uint2 b) { a = a + b; }

static __device__ __forceinline__ uint2 operator- (const uint2 a,const uint2 b) {
#if defined(__CUDA_ARCH__) && CUDA_VERSION < 7000
	uint2 result;
	asm("{\n\t"
		"sub.cc.u32 %0,%2,%4; \n\t"
		"subc.u32 %1,%3,%5;   \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
#else
	return vectorize(devectorize(a) - devectorize(b));
#endif
}
static __device__ __forceinline__ void operator-= (uint2 &a,const uint2 b) { a = a - b; }

static __device__ __forceinline__ uint2 operator+ (const uint2 a,const uint32_t b)
{
#if defined(__CUDA_ARCH__) && CUDA_VERSION < 7000
	uint2 result;
	asm("add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
	return result;
#else
	return vectorize(devectorize(a) + b);
#endif
}

static __device__ __forceinline__ uint2 operator- (const uint2 a,const uint64_t b) {
	return vectorize(devectorize(a) - b);
}
static __device__ __forceinline__ uint2 operator- (const uint2 a,const uint32_t b)
{
#if defined(__CUDA_ARCH__) && CUDA_VERSION < 7000
	uint2 result;
	asm("sub.cc.u32 %0,%2,%4; \n\t"
		"subc.u32 %1,%3,%5;   \n\t"		
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
	return result;
#else
	return vectorize(devectorize(a) - b);
#endif
}

/**
 * basic multiplication between 64bit no carry outside that range (ie mul.lo.b64(a*b))
 * (what does uint64 "*" operator)
 */
static __device__ __forceinline__ uint2 operator* (const uint2 a,const uint2 b){
	uint2 result;
	asm("{\n\t"
		"mul.lo.u32        %0,%2,%4;  \n\t"
		"mul.hi.u32        %1,%2,%4;  \n\t"
		"mad.lo.cc.u32    %1,%3,%4,%1; \n\t"
		"madc.lo.u32      %1,%3,%5,%1; \n\t"
	"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}

// uint2 ROR/ROL methods
__device__ __forceinline__
uint2 ROR2(const uint2 a, const uint32_t offset){
	uint2 result;
#if __CUDA_ARCH__ > 300
	if (offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	} else /* if (offset < 64) */ {
		/* offset SHOULD BE < 64 ! */
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
#else
	if (!offset)
		result = a;
	else if (offset < 32) {
		result.y = ((a.y >> offset) | (a.x << (32 - offset)));
		result.x = ((a.x >> offset) | (a.y << (32 - offset)));
	} else if (offset == 32) {
		result.y = a.x;
		result.x = a.y;
	} else {
		result.y = ((a.x >> (offset - 32)) | (a.y << (64 - offset)));
		result.x = ((a.y >> (offset - 32)) | (a.x << (64 - offset)));
	}
#endif
	return result;
}

__device__ __forceinline__
uint2 ROL2(const uint2 a, const uint32_t offset)
{
	uint2 result;
#if __CUDA_ARCH__ > 300
	if (offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
#else
	if (!offset)
		result = a;
	else
		result = ROR2(a, 64 - offset);
#endif
	return result;
}

__device__ __forceinline__
uint2 SWAPUINT2(uint2 value)
{
	return make_uint2(value.y, value.x);
}

/* Byte aligned Rotations (lyra2) */
__device__ __forceinline__
uint2 ROL8(const uint2 a){
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x2107);
	result.y = __byte_perm(a.y, a.x, 0x2107);
	return result;
}
__device__ __forceinline__
uint2 ROR8(const uint2 a){
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x4321);
	result.y = __byte_perm(a.y, a.x, 0x4321);
	return result;
}
__device__ __forceinline__
uint2 ROR16(const uint2 a){
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x5432);
	result.y = __byte_perm(a.y, a.x, 0x5432);
	return result;
}
__device__ __forceinline__
uint2 ROL16(const uint2 a){
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x1076);
	result.y = __byte_perm(a.y, a.x, 0x1076);

	return result;
}

__device__ __forceinline__
uint2 ROR24(const uint2 a){
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x6543);
	result.y = __byte_perm(a.y, a.x, 0x6543);
	return result;
}
__device__ __forceinline__
uint2 ROL24(const uint2 a){
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x0765);
	result.y = __byte_perm(a.y, a.x, 0x0765);
	return result;
}
/* uint2 for bmw512 - to double check later */

__device__ __forceinline__
static uint2 SHL2(const uint2 a,const uint32_t n) {
	uint64_t result;
	const uint64_t x = devectorize(a);
	asm ("shl.b64 %0,%1,%2;\n\t" : "=l"(result) : "l"(x), "r"(n));
	return vectorize(result);
}

__device__ __forceinline__
static uint2 SHR2(const uint2 a,const uint32_t n){

	uint64_t result;
	const uint64_t x = devectorize(a);
	asm ("shr.b64 %0,%1,%2;\n\t" : "=l"(result) : "l"(x), "r"(n));
	return vectorize(result);
}

__device__ __forceinline__
uint32_t xor3x(uint32_t a,uint32_t b,uint32_t c){
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result) : "r"(a), "r"(b),"r"(c)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
	#else
		result = a^b^c;
	#endif
	return result;
}

__device__ __forceinline__
uint2 xor3x(const uint2 a,const uint2 b,const uint2 c){
	uint2 result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result.x) : "r"(a.x), "r"(b.x),"r"(c.x)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
		asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result.y) : "r"(a.y), "r"(b.y),"r"(c.y)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
	#else
		result = a^b^c;
	#endif
	return result;
}

__device__ __forceinline__
uint2 chi(const uint2 a,const uint2 b,const uint2 c){ //keccak - chi
	uint2 result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(result.x) : "r"(a.x), "r"(b.x),"r"(c.x)); //0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
		asm ("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(result.y) : "r"(a.y), "r"(b.y),"r"(c.y)); //0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
	#else
		result = a ^ (~b) & c;
	#endif
	return result;
}
__device__ __forceinline__
uint32_t chi(const uint32_t a,const uint32_t b,const uint32_t c){ //keccak - chi
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(result) : "r"(a), "r"(b),"r"(c)); //0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
	#else
		result = a ^ (~b) & c;
	#endif
	return result;
}
__device__ __forceinline__
uint32_t bfe(uint32_t x, uint32_t bit, uint32_t numBits) {
	uint32_t ret;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
	return ret;

}

__device__ __forceinline__
uint32_t bfi(uint32_t x, uint32_t a, uint32_t bit, uint32_t numBits) {
	uint32_t ret;
	asm("bfi.b32 %0, %1, %2, %3,%4;" : "=r"(ret) : "r"(x), "r"(a), "r"(bit), "r"(numBits));
	return ret;
}
#endif // #ifndef CUDA_HELPER_H

