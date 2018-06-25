/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_
#define TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_

/**
 * Wrappers and helpers for CUDA device code.
 *
 * Wraps the warp-cooperative intrinsics introduced in CUDA 9 to provide
 * backwards compatibility, see go/volta-porting for details.
 * Provides atomic operations on types that aren't natively supported.
 */

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <algorithm>
#include <complex>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#if GOOGLE_CUDA
#include "cuda/include/cuda.h"
#endif
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace detail {

// Helper for range-based for loop using 'delta' increments.
// Usage: see GpuGridRange?() functions below.
template <typename T>
class GpuGridRange {
  struct Iterator {
    __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
    __device__ T operator*() const { return index_; }
    __device__ Iterator& operator++() {
      index_ += delta_;
      return *this;
    }
    __device__ bool operator!=(const Iterator& other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __device__ GpuGridRange(T begin, T delta, T end)
      : begin_(begin), delta_(delta), end_(end) {}

  __device__ Iterator begin() const { return Iterator{begin_, delta_}; }
  __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};

}  // namespace detail

// Helper to visit indices in the range 0 <= i < count, using the x-coordinate
// of the global thread index. That is, each index i is visited by all threads
// with the same x-coordinate.
// Usage: for(int i : GpuGridRangeX(count)) { visit(i); }
template <typename T>
__device__ detail::GpuGridRange<T> GpuGridRangeX(T count) {
  return detail::GpuGridRange<T>(blockIdx.x * blockDim.x + threadIdx.x,
                                  gridDim.x * blockDim.x, count);
}

// Helper to visit indices in the range 0 <= i < count using the y-coordinate.
// Usage: for(int i : GpuGridRangeY(count)) { visit(i); }
template <typename T>
__device__ detail::GpuGridRange<T> GpuGridRangeY(T count) {
  return detail::GpuGridRange<T>(blockIdx.y * blockDim.y + threadIdx.y,
                                  gridDim.y * blockDim.y, count);
}

// Helper to visit indices in the range 0 <= i < count using the z-coordinate.
// Usage: for(int i : GpuGridRangeZ(count)) { visit(i); }
template <typename T>
__device__ detail::GpuGridRange<T> GpuGridRangeZ(T count) {
  return detail::GpuGridRange<T>(blockIdx.z * blockDim.z + threadIdx.z,
                                  gridDim.z * blockDim.z, count);
}

#if GOOGLE_CUDA
// Mask for all 32 threads in a warp.
const unsigned kCudaWarpAll = 0xffffffff;
#elif TENSORFLOW_USE_ROCM
// ROCM TODO add ROCM implementation
// Mask for all 64 threads in a wavefront.
const unsigned kCudaWarpAll = 0xffffffff;
#endif

// Returns the warp lane ID of the calling thread
__device__ inline unsigned GpuLaneId() {
  unsigned int lane_id;
#if GOOGLE_CUDA
  asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
#elif TENSORFLOW_USE_ROCM
  // ROCM TODO add ROCM implementation
  lane_id = hc::__lane_id();
#endif
  return lane_id;
}

namespace detail {
// Returns true if mask is a valid parameter for __shfl*sync to return a well
// defined value, assuming the calling lane will read from src_lane as part of
// the shuffle operation.
//
// Specifically, returns true iff mask has the calling lane bit and the src_lane
// bit set, and the src_lane calls this function with the same mask value
// (required for the two threads to wait for each other).
//
// On Volta, for some invalid masks, this function hangs or returns false
// positives, because the implementation shuffles with the same mask that
// we are validating. Run on Pascal if you suspect that the mask is incorrect.
__device__ inline bool GpuValidateShuffleSyncMask(unsigned mask,
                                                   unsigned src_lane) {
  unsigned src_dst_mask = 1u << GpuLaneId() | 1u << src_lane;
#if CUDA_VERSION >= 9000
  unsigned src_lane_mask = __shfl_sync(mask, mask, src_lane);
#else
#if GOOGLE_CUDA
  unsigned src_lane_mask = __shfl(mask, src_lane);
#elif TENSORFLOW_USE_ROCM
  unsigned src_lane_mask = __shfl(static_cast<int>(mask), static_cast<int>(src_lane));
#endif
#endif
  return (src_dst_mask & ~mask) == 0 && src_lane_mask == mask;
}

// Returns the actual source lane for shuffle.
__device__ inline unsigned GpuShuffleGetSrcLane(int src_lane, int width) {
  int lane_id = GpuLaneId();
  int lane_base = lane_id & ~width + 1;
  int lane_offset = src_lane & width - 1;
  return lane_base + lane_offset;
}

// Returns the source lane for shuffle up.
__device__ inline unsigned GpuShuffleUpGetSrcLane(unsigned delta, int width) {
  unsigned lane_id = GpuLaneId();
  if ((lane_id & width - 1) < delta) {
    return lane_id;
  }
  return lane_id - delta;
}

// Returns the source lane for shuffle down.
__device__ inline unsigned GpuShuffleDownGetSrcLane(unsigned delta,
                                                     int width) {
  unsigned lane_id = GpuLaneId();
  if ((lane_id & width - 1) + delta >= width) {
    return lane_id;
  }
  return lane_id + delta;
}

// Returns the source lane for shuffle xor.
__device__ inline unsigned GpuShuffleXorGetSrcLane(int lane_mask, int width) {
  int lane_id = GpuLaneId();
  int src_lane = lane_id ^ lane_mask;
  if (src_lane > (lane_id | width - 1)) {
    return lane_id;
  }
  return src_lane;
}
}  // namespace detail

// For all *_sync wrappers below, it is illegal to synchronize threads from
// different program locations, because that is not supported before sm_70.
// In other words, all threads in 'mask' must call the functions in convergence.
// Code that requires sm_70 (and CUDA 9) may use the intrinsic directly.
//
// It is also illegal to shuffle with a mask that produces an undefined result
// for any of the threads. Specifically, all source threads of the shuffle
// must have their corresponding bit in 'mask' set.

// Wrapper for __syncwarp. No-op for CUDA 8 and earlier.
__device__ inline void GpuSyncWarp(unsigned mask = kCudaWarpAll) {
  assert(mask & 1u << GpuLaneId());
#if CUDA_VERSION >= 9000
  __syncwarp(mask);
#endif
}

// Wrapper for __ballot_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
__device__ inline unsigned GpuBallotSync(unsigned mask, int pred) {
  assert(mask & 1u << GpuLaneId());
#if CUDA_VERSION >= 9000
  return __ballot_sync(mask, pred);
#else
  return __ballot(pred) & mask;  // Apply mask to match __ballot_sync's spec.
#endif
}

// Wrapper for __any_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
__device__ inline int GpuAnySync(unsigned mask, int pred) {
  assert(mask & 1u << GpuLaneId());
#if CUDA_VERSION >= 9000
  return __any_sync(mask, pred);
#else
  return __any(pred);
#endif
}

// Wrapper for __all_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
__device__ inline int GpuAllSync(unsigned mask, int pred) {
  assert(mask & 1u << GpuLaneId());
#if CUDA_VERSION >= 9000
  return __all_sync(mask, pred);
#else
  return __all(pred);
#endif
}

// Wrapper for __shfl_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
template <typename T>
__device__ T GpuShuffleSync(unsigned mask, T value, int src_lane,
                             int width = warpSize) {
  assert(!(width & width - 1));
  assert(detail::GpuValidateShuffleSyncMask(
      mask, detail::GpuShuffleGetSrcLane(src_lane, width)));
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, src_lane, width);
#else
  return __shfl(value, src_lane, width);
#endif
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double GpuShuffleSync(unsigned mask, double value,
                                         int src_lane, int width = warpSize) {
  unsigned lo, hi;
#if GOOGLE_CUDA
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = GpuShuffleSync(mask, hi, src_lane, width);
  lo = GpuShuffleSync(mask, lo, src_lane, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
#elif TENSORFLOW_USE_ROCM
  uint64_t tmp = static_cast<uint64_t>(value);
  lo = static_cast<unsigned>(tmp);
  hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl(static_cast<int>(hi), src_lane, width);
  lo = __shfl(static_cast<int>(lo), src_lane, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 | static_cast<uint64_t>(lo));
#endif
}

// Wrapper for __shfl_up_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
template <typename T>
__device__ inline T GpuShuffleUpSync(unsigned mask, T value, unsigned delta,
                                      int width = warpSize) {
  assert(!(width & width - 1));
  assert(detail::GpuValidateShuffleSyncMask(
      mask, detail::GpuShuffleUpGetSrcLane(delta, width)));
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(mask, value, delta, width);
#else
  return __shfl_up(value, delta, width);
#endif
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double GpuShuffleUpSync(unsigned mask, double value,
                                           unsigned delta,
                                           int width = warpSize) {
  unsigned lo, hi;
#if GOOGLE_CUDA
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = GpuShuffleUpSync(mask, hi, delta, width);
  lo = GpuShuffleUpSync(mask, lo, delta, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
#elif TENSORFLOW_USE_ROCM
  uint64_t tmp = static_cast<uint64_t>(value);
  lo = static_cast<unsigned>(tmp);
  hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl_up(static_cast<int>(hi), delta, width);
  lo = __shfl_up(static_cast<int>(lo), delta, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 | static_cast<uint64_t>(lo));
#endif
}

// Wrapper for __shfl_down_sync. All threads in 'mask' must call this function
// in convergence, see comment above for details.
template <typename T>
__device__ inline T GpuShuffleDownSync(unsigned mask, T value, unsigned delta,
                                        int width = warpSize) {
  assert(!(width & width - 1));
  assert(detail::GpuValidateShuffleSyncMask(
      mask, detail::GpuShuffleDownGetSrcLane(delta, width)));
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double GpuShuffleDownSync(unsigned mask, double value,
                                             unsigned delta,
                                             int width = warpSize) {
  unsigned lo, hi;
#if GOOGLE_CUDA
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = GpuShuffleDownSync(mask, hi, delta, width);
  lo = GpuShuffleDownSync(mask, lo, delta, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
#elif TENSORFLOW_USE_ROCM
  uint64_t tmp = static_cast<uint64_t>(value);
  lo = static_cast<unsigned>(tmp);
  hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl_down(static_cast<int>(hi), delta, width);
  lo = __shfl_down(static_cast<int>(lo), delta, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 | static_cast<uint64_t>(lo));
#endif
}

// Wrapper for __shfl_xor_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
template <typename T>
__device__ T GpuShuffleXorSync(unsigned mask, T value, int lane_mask,
                                int width = warpSize) {
  assert(!(width & width - 1));
  assert(detail::GpuValidateShuffleSyncMask(
      mask, detail::GpuShuffleXorGetSrcLane(lane_mask, width)));
#if GOOGLE_CUDA
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, lane_mask, width);
#else
  return __shfl_xor(value, lane_mask, width);
#endif
#elif TENSORFLOW_USE_ROCM
  // ROCM TODO: check if HIP should be changed to cope with more types
  return __shfl_xor(static_cast<int>(value), lane_mask, width);
#endif
}


#if TENSORFLOW_USE_ROCM
template<>
__device__ inline Eigen::half GpuShuffleXorSync(unsigned mask, Eigen::half value, int lane_mask,
                                int width) {
  assert(!(width & width - 1));
  assert(detail::GpuValidateShuffleSyncMask(
      mask, detail::GpuShuffleXorGetSrcLane(lane_mask, width)));
  return static_cast<Eigen::half>(__shfl_xor(static_cast<float>(value), lane_mask, width));
}
#endif
   
// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double GpuShuffleXorSync(unsigned mask, double value,
                                            int lane_mask,
                                            int width = warpSize) {
  unsigned lo, hi;
#if GOOGLE_CUDA
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = GpuShuffleXorSync(mask, hi, lane_mask, width);
  lo = GpuShuffleXorSync(mask, lo, lane_mask, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
#elif TENSORFLOW_USE_ROCM
  uint64_t tmp = static_cast<uint64_t>(value);
  lo = static_cast<unsigned>(tmp);
  hi = static_cast<unsigned>(tmp >> 32);
  hi = __shfl_xor(static_cast<int>(hi), lane_mask, width);
  lo = __shfl_xor(static_cast<int>(lo), lane_mask, width);
  return static_cast<double>(static_cast<uint64_t>(hi) << 32 | static_cast<uint64_t>(lo));
#endif
}

// Wrapper for __ldg.
template <typename T>
__host__ __device__ T GpuLdg(const T* address) {
#if __CUDA_ARCH__ >= 350
  return __ldg(address);
#else
  return *address;
#endif
}

__host__ __device__ inline bool GpuLdg(const bool* address) {
  return GpuLdg(reinterpret_cast<const char*>(address)) != 0;
}

__host__ __device__ inline std::complex<float> GpuLdg(
    const std::complex<float>* address) {
#if __CUDA_ARCH__ >= 350
  float2 mem = __ldg(reinterpret_cast<const float2*>(address));
  return std::complex<float>(mem.x, mem.y);
#else
  return *address;
#endif
}

__host__ __device__ inline std::complex<double> GpuLdg(
    const std::complex<double>* address) {
#if __CUDA_ARCH__ >= 350
  double2 mem = __ldg(reinterpret_cast<const double2*>(address));
  return std::complex<double>(mem.x, mem.y);
#else
  return *address;
#endif
}

// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T>
__global__ void SetZero(const int count, T* ptr) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1 && blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (int i : GpuGridRangeX(count)) {
    ptr[i] = T(0);
  }
}

// Helper to set all tensor entries to a specific value.
template <typename T>
__global__ void SetToValue(const int count, T* ptr, T value) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1 && blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (int i : GpuGridRangeX(count)) {
    ptr[i] = value;
  }
}

namespace detail {
// Helper function for atomic accumulation implemented as CAS.
template <typename T, typename F>
__device__ T GpuAtomicCasHelper(T* ptr, F accumulate) {
  T old = *ptr;
  T assumed;
  do {
    assumed = old;
    old = atomicCAS(ptr, assumed, accumulate(assumed));
  } while (assumed != old);
  return old;
}

// Overload for floating point (using integer comparison to handle NaN
// correctly).
template <typename F>
__device__ float GpuAtomicCasHelper(float* ptr, F accumulate) {
  return __float_as_int(
      GpuAtomicCasHelper(reinterpret_cast<int32*>(ptr), [accumulate](int32 a) {
        return __float_as_int(accumulate(__int_as_float(a)));
      }));
}
template <typename F>
__device__ double GpuAtomicCasHelper(double* ptr, F accumulate) {
#if TENSORFLOW_USE_ROCM
  // FIXME : remove the workaround below once bug is fixed
  // HIP has a bug in the implementation of __longlong_as_double
  // So workaround it by using reinterpret_cast<double*>
  uint64_t result = GpuAtomicCasHelper(
      reinterpret_cast<tensorflow::uint64*>(ptr),
      [accumulate](tensorflow::uint64 a) {
	return __double_as_longlong(accumulate(*(reinterpret_cast<double*>(&a))));
      });
 return *(reinterpret_cast<double*>(&result));
#else
  return __longlong_as_double(GpuAtomicCasHelper(
      reinterpret_cast<tensorflow::uint64*>(ptr),
      [accumulate](tensorflow::uint64 a) {
        return __double_as_longlong(accumulate(__longlong_as_double(a)));
      }));
#endif
}

// Overload of above function for half. Note that we don't have
// atomicCAS() for anything less than 32 bits, so we need to include the
// other 16 bits in the operation.
//
// This version is going to be very slow
// under high concurrency, since most threads will be spinning on failing
// their compare-and-swap tests. (The fact that we get false sharing on the
// neighboring fp16 makes this even worse.) If you are doing a large reduction,
// you are much better off with doing the intermediate steps in fp32 and then
// switching to fp16 as late as you can in the calculations.
//
// Note: Assumes little endian.
template <typename F>
__device__ Eigen::half GpuAtomicCasHelper(Eigen::half* ptr, F accumulate) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Not little endian");
#endif
  namespace half_impl = Eigen::half_impl;
  intptr_t intptr = reinterpret_cast<intptr_t>(ptr);
  assert(!(intptr & 0x1));  // should be 2-aligned.
  if (intptr & 0x2) {
    // The half is in the second part of the uint32 (upper 16 bits).
    uint32* address = reinterpret_cast<uint32*>(intptr - 2);
    uint32 result = GpuAtomicCasHelper(address, [accumulate](uint32 arg) {
      unsigned short high = static_cast<unsigned short>(arg >> 16);
      Eigen::half acc = accumulate(half_impl::raw_uint16_to_half(high));
      return (static_cast<uint32>(acc.x) << 16) | (arg & 0xffff);
    });
    return half_impl::raw_uint16_to_half(static_cast<uint16>(result >> 16));
  } else {
    // The half is in the first part of the uint32 (lower 16 bits).
    uint32* address = reinterpret_cast<uint32*>(intptr);
    uint32 result = GpuAtomicCasHelper(address, [accumulate](uint32 arg) {
      unsigned short low = static_cast<unsigned short>(arg & 0xffff);
      Eigen::half acc = accumulate(half_impl::raw_uint16_to_half(low));
      return (arg & 0xffff0000) | static_cast<uint32>(acc.x);
    });
    return half_impl::raw_uint16_to_half(static_cast<uint16>(result & 0xffff));
  }
}

template <typename From, typename To>
using ToTypeIfConvertible =
    typename std::enable_if<std::is_convertible<From, To>::value, To>::type;

}  // namespace detail

// CUDA provides atomic ops, but not for all types.  We provide wrappers
// for some ops and provide implementation for all reasonable types.

template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicAdd(T* ptr, U value) {
  return atomicAdd(ptr, value);
}

__device__ inline Eigen::half GpuAtomicAdd(Eigen::half* ptr,
                                            Eigen::half value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](Eigen::half a) { return a + value; });
}


#if (__CUDA_ARCH__ < 600) || TENSORFLOW_USE_ROCM
__device__ inline double GpuAtomicAdd(double* ptr, double value) {
  return detail::GpuAtomicCasHelper(ptr,
                                     [value](double a) { return a + value; });
}
#elif __clang__
// Clang cannot compile __nvvm_atom_add_gen_d builtin yet, use inline PTX.
// see https://reviews.llvm.org/D39638
__device__ inline double GpuAtomicAdd(double* ptr, double value) {
  double result;
  asm volatile("atom.add.f64 %0, [%1], %2;"
               : "=d"(result)
               : "l"(ptr), "d"(value)
               : "memory");
  return result;
}
#endif
// GpuAtomicAdd
// Specializations of GpuAtomicAdd for complex types, which GpuAtomicAdd does
// not support. We treat a std::complex<T>* as a T* (the C++ standard section
// 26.4.4 allows this explicitly) and atomic add the real and imaginary
// components individually. The operation as a whole is not atomic, but we can
// safely treat the components independently for the purpose of accumulating.

// ROCM TODO support GpuAtomicAdd for std::complex<>
#if GOOGLE_CUDA
__device__ inline std::complex<float> GpuAtomicAdd(std::complex<float>* ptr,
                                                    std::complex<float> value) {
  auto ptr_scalar = reinterpret_cast<float*>(ptr);
  return std::complex<float>(GpuAtomicAdd(ptr_scalar, value.real()),
                             GpuAtomicAdd(ptr_scalar + 1, value.imag()));
}

__device__ inline std::complex<double> GpuAtomicAdd(
    std::complex<double>* ptr, std::complex<double> value) {
  auto ptr_scalar = reinterpret_cast<double*>(ptr);
  return std::complex<double>(GpuAtomicAdd(ptr_scalar, value.real()),
                              GpuAtomicAdd(ptr_scalar + 1, value.imag()));
}
#endif

// GpuAtomicSub
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicSub(T* ptr, U value) {
  return atomicSub(ptr, value);
}

// Specializations of substraction which add the negative value.
__device__ inline float GpuAtomicSub(float* ptr, float value) {
  return GpuAtomicAdd(ptr, -value);
}

__device__ inline double GpuAtomicSub(double* ptr, double value) {
  return GpuAtomicAdd(ptr, -value);
}

__device__ inline tensorflow::uint64 GpuAtomicSub(tensorflow::uint64* ptr,
                                                   tensorflow::uint64 value) {
  return GpuAtomicAdd(ptr, -value);
}

__device__ inline Eigen::half GpuAtomicSub(Eigen::half* ptr,
                                            Eigen::half value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](Eigen::half a) { return a - value; });
}

// GpuAtomicMax
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicMax(T* ptr, U value) {
  return atomicMax(ptr, value);
}

__device__ inline float GpuAtomicMax(float* ptr, float value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](float a) { return max(a, value); });
}

__device__ inline double GpuAtomicMax(double* ptr, double value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](double a) { return max(a, value); });
}

__device__ inline Eigen::half GpuAtomicMax(Eigen::half* ptr,
                                            Eigen::half value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](Eigen::half a) { return max(a, value); });
}

#if __CUDA_ARCH__ < 320
__device__ inline tensorflow::uint64 GpuAtomicMax(tensorflow::uint64* ptr,
                                                   tensorflow::uint64 value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](tensorflow::uint64 a) { return max(a, value); });
}
#endif

// GpuAtomicMin
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicMin(T* ptr, U value) {
  return atomicMin(ptr, value);
}

__device__ inline float GpuAtomicMin(float* ptr, float value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](float a) { return min(a, value); });
}

__device__ inline double GpuAtomicMin(double* ptr, double value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](double a) { return min(a, value); });
}

__device__ inline Eigen::half GpuAtomicMin(Eigen::half* ptr,
                                            Eigen::half value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](Eigen::half a) { return min(a, value); });
}

#if __CUDA_ARCH__ < 320
__device__ inline tensorflow::uint64 GpuAtomicMin(tensorflow::uint64* ptr,
                                                   tensorflow::uint64 value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](tensorflow::uint64 a) { return min(a, value); });
}
#endif

// GpuAtomicMul
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicMul(T* ptr, U value) {
  return detail::GpuAtomicCasHelper(ptr, [value](T a) { return a * value; });
}

// GpuAtomicDiv
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicDiv(T* ptr, U value) {
  return detail::GpuAtomicCasHelper(ptr, [value](T a) { return a / value; });
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_
