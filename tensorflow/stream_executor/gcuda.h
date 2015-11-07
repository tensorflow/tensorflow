// Common declarations and includes for mixed-mode GPU usage at Google.
//
// This header serves to define a "common baseline" for GPU usage,
// either with gcudacc or nvcc, and on the host or device. The rule of thumb is,
// "if you're working with mixed-mode GPU code at Google, include this header."
#ifndef TENSORFLOW_STREAM_EXECUTOR_GCUDA_H_
#define TENSORFLOW_STREAM_EXECUTOR_GCUDA_H_

// Symbol glossary:
//   __CUDACC__: CUDA capable compiler, compiling host or device
//   __CUDA_ARCH__: Compiling device code
//   __GCUDACC__: Using gcudacc
//   __NVCC__: Using nvcc

// For device code compiled with gcudacc, CUDA_ASSUME(X) tells the compiler
// that it may assume that X is true. This can enable further optimization.
// It is undefined behavior if X is not true. X should not have side-effects
// and gcudacc will try to warn you if it does.
#if defined(__CUDA_ARCH__) && defined(__GCUDACC__)
#define CUDA_ASSUME(X) __builtin_assume(X)
#else
#define CUDA_ASSUME(X) do {} while (false)
#endif

namespace perftools {
namespace gputools {
namespace cache_config {
// A version of the KernelCacheConfig enum class, exposed for pre-C++11
// compilers.
enum CacheConfig {
  // Indicates no preference for device L1/shared memory configuration.
  kNoPreference,

  // Indicates a preference for more shared memory than L1 cache.
  kPreferShared,

  // Indicates a preference for more L1 cache than shared memory.
  kPreferL1,

  // Indicates a preference for equal amounts of L1 cache and shared memory.
  kPreferEqual,
};
}  // namespace cache_config

namespace shared_mem_config {
// A compatability-layer declaration of CUsharedconfig, needed to support
// cuFuncSetSharedMemConfig/cudaDeviceSetSharedMemConfig. Declared here for
// compatability with pre-C++11 compilers.
enum SharedMemConfig {
  // Indicates that the context's shared memory config should be used.
  kDefaultBankSize,

  // Specifies a four-byte bank size for shared memory.
  kFourByteBankSize,

  // Specifies an eight-byte bank size for shared memory.
  kEightByteBankSize,
};
}  // namespace shared_mem_config
}  // namespace gputools
}  // namespace perftools

#if !defined(__NVCC__) && !defined(GCUDACC_STANDALONE_MODE)
// Using gcudacc, either device-only or mixed-mode code. No special declarations
// are needed for host-only code being compiled under gcudacc.

// These includes are required by the code introduced during gcudacc operation.
// Since the user code may not directly include these headers, they may not be
// present in the build environment without inclusion here.
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/launch_dim.h"
#include "tensorflow/stream_executor/machine_manager.h"
#include "tensorflow/stream_executor/shared_memory_config.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"

// cudaConfigureCall is a symbol used by Clang when it sees a CUDA triple-angle-
// bracket launch, so we declare it here so the symbol resolves. It is not used
// by gcudacc-generated code, however, so it is not defined anywhere.
// In other words, this is a dummy declaration needed for parsing.

#ifdef __GCUDACC__
// These symbols only need to be defined during compilation with gcudacc.
namespace perftools {
namespace gputools {

// This class defines all the implicit conversions necessary to match launch
// dimensions against the cudaConfigureCall() signature, and sits where a dim3
// usually would in triple angle launches. This supports the kernel launch
// dimension styles:
//  kernel<<<1, 1>>>() and
//  kernel<<<BlockDim(...), ThreadDim(...)>>> and
//  kernel<<<dim3(1), dim3(1)>>>
// All of these are predicated upon implicit conversions, which are frowned upon
// by the style guide. Rather then add this CUDA-specific bad behavior to
// StreamExecutor headers, we isolate it here.
class LaunchDimConverter {
 public:
  LaunchDimConverter(unsigned long long int i) : _dim(i, 1, 1) {}  // NOLINT
  LaunchDimConverter(::perftools::gputools::BlockDim dim)
      :  // NOLINT
        _dim(dim.x, dim.y, dim.z) {}
  LaunchDimConverter(::perftools::gputools::ThreadDim dim)
      :  // NOLINT
        _dim(dim.x, dim.y, dim.z) {}
  LaunchDimConverter(dim3 dim) :  // NOLINT
    _dim(dim.x, dim.y, dim.z) {}

  ::perftools::gputools::BlockDim AsBlockDim() {
    return ::perftools::gputools::BlockDim(_dim.x, _dim.y, _dim.z);
  }

  ::perftools::gputools::ThreadDim AsThreadDim() {
    return ::perftools::gputools::ThreadDim(_dim.x, _dim.y, _dim.z);
  }

 private:
  ::perftools::gputools::Dim3D _dim;
};
}  // namespace gputools
}  // namespace perftools

int cudaConfigureCall(::perftools::gputools::LaunchDimConverter grid_size,
                      ::perftools::gputools::LaunchDimConverter block_size,
                      unsigned shared_size = 0,
                      ::perftools::gputools::Stream *stream = 0);
#endif

// The rest of the symbols in this block are needed during both StreamExecutor
// and user library compilation.
namespace perftools {
namespace gputools {

// Gets the preferred shared memory configuration for the device to which
// the specified executor is bound.
shared_mem_config::SharedMemConfig DeviceGetSharedMemConfig(
    StreamExecutor *stream_exec);

// Sets the preferred shared memory configuration for the device to which
// the specified executor is bound.
// Does not return an error if the current device is invalid.
void DeviceSetSharedMemConfig(StreamExecutor *stream_exec,
                              shared_mem_config::SharedMemConfig config);

// Sets the preferred cache configuration for the given kernel.
template <typename KernelT>
void FuncSetCacheConfig(Stream *stream, KernelT kernel,
                        cache_config::CacheConfig cache_config) {
  FuncSetCacheConfig(stream, reinterpret_cast<void *>(kernel), cache_config);
}

// Internal specialization of the above.
template <>
void FuncSetCacheConfig<void *>(Stream *stream, void *kernel,
                                cache_config::CacheConfig cache_config);

// Gets the preferred cache configuration for the given kernel.
template <typename KernelT>
KernelCacheConfig FuncGetCacheConfig(KernelT kernel) {
  return FuncGetCacheConfig(reinterpret_cast<void *>(kernel));
}

// Internal specialization of the above.
template <>
KernelCacheConfig FuncGetCacheConfig<void *>(void *kernel);

}  // namespace gputools
}  // namespace perftools

#elif defined(__NVCC__)
// NVCC code compilation, device-only or mixed mode. As above, no special
// declarations are needed for host-only code.
namespace perftools {
namespace gputools {
class Stream;
}  // namespace gputools
}  // namespace perftools

// --- BEGIN EXTERNALLY-DEFINED FUNCTIONS

// The following functions must be defined in some external library linked in to
// the final binary - they are _not_ defined in the StreamExecutor
// (in nvcc mode).

// Sets the preferred cache configuration for the specified kernel.
template <typename KernelT>
void SetCudaCacheConfig(perftools::gputools::Stream* stream, KernelT kernel,
    ::perftools::gputools::cache_config::CacheConfig preference);

// Gets the current device for use in CUDA runtime-emulating routines.
// "device" is the device ordinal as returned by
// StreamExecutor::device_ordinal().
int GetDevice();

// Sets the current device for use in CUDA runtime-emulating routines.
// "device" is the device ordinal as returned by
// StreamExecutor::device_ordinal().
void SetDevice(int device);

// --- END EXTERNALLY-DEFINED FUNCTIONS

namespace perftools {
namespace gputools {
template <typename KernelT>
void FuncSetCacheConfig(Stream *stream, KernelT kernel,
                        cache_config::CacheConfig cache_config) {
  SetCudaCacheConfig(stream, reinterpret_cast<void*>(kernel), cache_config);
}
}  // namespace gputools
}  // namespace perftools

// The following functions are declared extern "C" in CUDA's device_functions.h,
// so we have to wrap them for compatability with the cuda_builtin namespace.
// Thin wrappers to break these functions out of cuda_builtin are defined below.
__forceinline__ __device__ clock_t __gcuda_nvcc_clock() { return clock(); }
__forceinline__ __device__ int __gcuda_nvcc__clz(int x) {
  return __clz(x);
}
__forceinline__ __device__ int __gcuda_nvcc__clzll(long long int x) {
  return __clzll(x);
}
__forceinline__ __device__ float __gcuda_nvcc__fdividef(float a, float b) {
  return __fdividef(a, b);
}
__forceinline__ __device__ int __gcuda_nvcc__ffsll(long long int x) { // NOLINT
  return __ffsll(x);
}
__forceinline__ __device__ int __gcuda_nvcc__popc(unsigned int x) {
  return __popc(x);
}
__forceinline__ __device__ float __gcuda_nvcc__powf(float a, float b) {
  return __powf(a, b);
}
__forceinline__ __device__ void __gcuda_nvcc__sincosf(
    float x, float *sptr, float *cptr) {
  __sincosf(x, sptr, cptr);
}
__forceinline__ __device__ unsigned int __gcuda_nvcc__umulhi(
    unsigned int x, unsigned int y) {
  return __umulhi(x, y);
}

#if __CUDA_ARCH__ >= 200 || !defined(__CUDA_ARCH__)
__forceinline__ __device__ unsigned int __gcuda_nvcc__ballot(int x) {
  return __ballot(x);
}
#endif  // __CUDA_ARCH__ >= 200 || !defined(__CUDA_ARCH__)

// Forward-declare printf as nvcc does not declare it by itself and we
// need this file to compile even if it is included before including
// stdio.h or cstdio.
int printf(const char* format, ...);

namespace cuda_builtin {
using ::abs;
using ::atomicAdd;
using ::atomicCAS;
using ::ceil;
using ::ceilf;
using ::cos;
using ::cosf;
using ::erfcinv;
using ::erfcinvf;
using ::exp;
using ::expf;
using ::fabs;
using ::fabsf;
using ::floor;
using ::floorf;
using ::fabs;
using ::fabsf;
using ::fma;
using ::fmaf;
using ::fmax;
using ::fmaxf;
using ::fmin;
using ::fminf;
using ::log;
using ::log1p;
using ::log1pf;
using ::logf;
using ::max;
using ::min;
using ::powf;
using ::printf;
using ::sin;
using ::sinf;
using ::sincos;
using ::sincosf;
using ::sincospi;
using ::sincospif;
using ::sqrt;
using ::sqrtf;
using ::tanh;
using ::trunc;
using ::truncf;
using ::trunc;

// rsqrt and rsqrtf are functions defined by nvcc in both host and device mode.
// Add these functions to gcuda.h such that it is also host device. In device
// side they correspond to intrinsics while explicit definitions are provided
// below for host side.
#ifdef __CUDA_ARCH__
using ::rsqrt;
using ::rsqrtf;
#else
__forceinline__ __host__ __device__ float rsqrtf(float x) {
  return 1 / std::sqrt(x);
}
__forceinline__ __host__ __device__ double rsqrt(double x) {
  return 1 / std::sqrt(x);
}
#endif

__forceinline__ __device__ int clock() { return __gcuda_nvcc_clock(); }

__forceinline__ __device__ int __clz(int x) {
  return __gcuda_nvcc__clz(x);
}

__forceinline__ __device__ int __clz(long long int x) {
  return __gcuda_nvcc__clzll(x);
}

__forceinline__ __device__ float __fdividef(float a, float b) {
  return __gcuda_nvcc__fdividef(a, b);
}

__forceinline__ __device__ int __ffsll(long long int x) { // NOLINT
  return __gcuda_nvcc__ffsll(x);
}

__forceinline__ __device__ int __popc(unsigned int x) {
  return __gcuda_nvcc__popc(x);
}

__forceinline__ __device__ float __powf(float a, float b) {
  return __gcuda_nvcc__powf(a, b);
}

__forceinline__ __device__ void __sincosf(float x, float *sptr, float *cptr) {
  __gcuda_nvcc__sincosf(x, sptr, cptr);
}

__forceinline__ __device__ unsigned int __umulhi(unsigned int x,
                                                 unsigned int y) {
  return __gcuda_nvcc__umulhi(x, y);
}

#ifdef __CUDA_ARCH__
// These symbols are only visible when parsing device code.
using ::__double_as_longlong;
using ::__int_as_float;
using ::__float_as_int;
using ::__longlong_as_double;
#endif  // __CUDA_ARCH__

#if __CUDA_ARCH__ >= 200 || !defined(__CUDA_ARCH__)
__forceinline__ __device__ unsigned int __ballot(int x) {
  return __gcuda_nvcc__ballot(x);
}
#endif  // __CUDA_ARCH__ >= 200 || !defined(__CUDA_ARCH__)

#if __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)
using ::__shfl;
using ::__shfl_down;
using ::__shfl_up;
using ::__shfl_xor;
#endif  // __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)

#if __CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__)
using ::__ldg;
#endif  // __CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__)

#if __CUDA_API_VERSION < 6050
// CUDA < 6.5 defines isfinite as a macro, while CUDA >= 6.5 and gcudacc
// define isfinite as a function. Work around this for the CUDA 5.5 case,
// duplicating that macro definition.
#undef isfinite
#define __gcuda_nvcc_isfinite(x)                                  \
    (sizeof(x) == sizeof(float) ? __finitef(x) :                  \
        sizeof(x) == sizeof(double) ? __finite(x) : __finitel(x))
inline __device__ int isfinite(float x) {
  return __gcuda_nvcc_isfinite(x);
}
inline __device__ int isfinite(double x) {
  return __gcuda_nvcc_isfinite(x);
}
inline __device__ int isfinite(long double x) {
  return __gcuda_nvcc_isfinite(x);
}
#else
// CUDA API >= v6.5
using ::isfinite;
#endif  // __CUDA_API_VERSION >= 6050
}  // namespace cuda_builtin

#if __CUDA_API_VERSION >= 6050
// The second part of the isfinite workaround.
inline __device__ int isfinite(float x) {
  return __gcuda_nvcc_isfinite(x);
}
inline __device__ int isfinite(double x) {
  return __gcuda_nvcc_isfinite(x);
}
inline __device__ int isfinite(long double x) {
  return __gcuda_nvcc_isfinite(x);
}
#endif  // __CUDA_API_VERSION >= 6050

#endif  // defined(__NVCC__)

#endif  // TENSORFLOW_STREAM_EXECUTOR_GCUDA_H_
