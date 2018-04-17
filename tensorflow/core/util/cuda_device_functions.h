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

#ifndef TENSORFLOW_CORE_UTIL_CUDA_DEVICE_FUNCTIONS_H_
#define TENSORFLOW_CORE_UTIL_CUDA_DEVICE_FUNCTIONS_H_

/**
 * Wrappers and helpers for CUDA device code.
 *
 * Wraps the warp-cooperative intrinsics introduced in CUDA 9 to provide
 * backwards compatibility, see go/volta-porting for details.
 * Provides atomic operations on types that aren't natively supported.
 */

#if GOOGLE_CUDA

#include <algorithm>
#include <complex>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda/include/cuda.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace detail {

// Helper for range-based for loop using 'delta' increments.
// Usage: see CudaGridRange?() functions below.
template <typename T>
class CudaGridRange {
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
  __device__ CudaGridRange(T begin, T delta, T end)
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
// Usage: for(int i : CudaGridRangeX(count)) { visit(i); }
template <typename T>
__device__ detail::CudaGridRange<T> CudaGridRangeX(T count) {
  return detail::CudaGridRange<T>(blockIdx.x * blockDim.x + threadIdx.x,
                                  gridDim.x * blockDim.x, count);
}

// Helper to visit indices in the range 0 <= i < count using the y-coordinate.
// Usage: for(int i : CudaGridRangeY(count)) { visit(i); }
template <typename T>
__device__ detail::CudaGridRange<T> CudaGridRangeY(T count) {
  return detail::CudaGridRange<T>(blockIdx.y * blockDim.y + threadIdx.y,
                                  gridDim.y * blockDim.y, count);
}

// Helper to visit indices in the range 0 <= i < count using the z-coordinate.
// Usage: for(int i : CudaGridRangeZ(count)) { visit(i); }
template <typename T>
__device__ detail::CudaGridRange<T> CudaGridRangeZ(T count) {
  return detail::CudaGridRange<T>(blockIdx.z * blockDim.z + threadIdx.z,
                                  gridDim.z * blockDim.z, count);
}

// Mask for all 32 threads in a warp.
const unsigned kCudaWarpAll = 0xffffffff;

// Returns the warp lane ID of the calling thread
__device__ inline unsigned CudaLaneId() {
  unsigned int lane_id;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
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
__device__ inline bool CudaValidateShuffleSyncMask(unsigned mask,
                                                   unsigned src_lane) {
  unsigned src_dst_mask = 1u << CudaLaneId() | 1u << src_lane;
#if CUDA_VERSION >= 9000
  unsigned src_lane_mask = __shfl_sync(mask, mask, src_lane);
#else
  unsigned src_lane_mask = __shfl(mask, src_lane);
#endif
  return (src_dst_mask & ~mask) == 0 && src_lane_mask == mask;
}

// Returns the actual source lane for shuffle.
__device__ inline unsigned CudaShuffleGetSrcLane(int src_lane, int width) {
  int lane_id = CudaLaneId();
  int lane_base = lane_id & ~width + 1;
  int lane_offset = src_lane & width - 1;
  return lane_base + lane_offset;
}

// Returns the source lane for shuffle up.
__device__ inline unsigned CudaShuffleUpGetSrcLane(unsigned delta, int width) {
  unsigned lane_id = CudaLaneId();
  if ((lane_id & width - 1) < delta) {
    return lane_id;
  }
  return lane_id - delta;
}

// Returns the source lane for shuffle down.
__device__ inline unsigned CudaShuffleDownGetSrcLane(unsigned delta,
                                                     int width) {
  unsigned lane_id = CudaLaneId();
  if ((lane_id & width - 1) + delta >= width) {
    return lane_id;
  }
  return lane_id + delta;
}

// Returns the source lane for shuffle xor.
__device__ inline unsigned CudaShuffleXorGetSrcLane(int lane_mask, int width) {
  int lane_id = CudaLaneId();
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
__device__ inline void CudaSyncWarp(unsigned mask = kCudaWarpAll) {
  assert(mask & 1u << CudaLaneId());
#if CUDA_VERSION >= 9000
  __syncwarp(mask);
#endif
}

// Wrapper for __ballot_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
__device__ inline unsigned CudaBallotSync(unsigned mask, int pred) {
  assert(mask & 1u << CudaLaneId());
#if CUDA_VERSION >= 9000
  return __ballot_sync(mask, pred);
#else
  return __ballot(pred) & mask;  // Apply mask to match __ballot_sync's spec.
#endif
}

// Wrapper for __any_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
__device__ inline int CudaAnySync(unsigned mask, int pred) {
  assert(mask & 1u << CudaLaneId());
#if CUDA_VERSION >= 9000
  return __any_sync(mask, pred);
#else
  return __any(pred);
#endif
}

// Wrapper for __all_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
__device__ inline int CudaAllSync(unsigned mask, int pred) {
  assert(mask & 1u << CudaLaneId());
#if CUDA_VERSION >= 9000
  return __all_sync(mask, pred);
#else
  return __all(pred);
#endif
}

// Wrapper for __shfl_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
template <typename T>
__device__ T CudaShuffleSync(unsigned mask, T value, int src_lane,
                             int width = warpSize) {
  assert(!(width & width - 1));
  assert(detail::CudaValidateShuffleSyncMask(
      mask, detail::CudaShuffleGetSrcLane(src_lane, width)));
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, src_lane, width);
#else
  return __shfl(value, src_lane, width);
#endif
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double CudaShuffleSync(unsigned mask, double value,
                                         int src_lane, int width = warpSize) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = CudaShuffleSync(mask, hi, src_lane, width);
  lo = CudaShuffleSync(mask, lo, src_lane, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
}

// Wrapper for __shfl_up_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
template <typename T>
__device__ inline T CudaShuffleUpSync(unsigned mask, T value, unsigned delta,
                                      int width = warpSize) {
  assert(!(width & width - 1));
  assert(detail::CudaValidateShuffleSyncMask(
      mask, detail::CudaShuffleUpGetSrcLane(delta, width)));
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(mask, value, delta, width);
#else
  return __shfl_up(value, delta, width);
#endif
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double CudaShuffleUpSync(unsigned mask, double value,
                                           unsigned delta,
                                           int width = warpSize) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = CudaShuffleUpSync(mask, hi, delta, width);
  lo = CudaShuffleUpSync(mask, lo, delta, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
}

// Wrapper for __shfl_down_sync. All threads in 'mask' must call this function
// in convergence, see comment above for details.
template <typename T>
__device__ inline T CudaShuffleDownSync(unsigned mask, T value, unsigned delta,
                                        int width = warpSize) {
  assert(!(width & width - 1));
  assert(detail::CudaValidateShuffleSyncMask(
      mask, detail::CudaShuffleDownGetSrcLane(delta, width)));
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double CudaShuffleDownSync(unsigned mask, double value,
                                             unsigned delta,
                                             int width = warpSize) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = CudaShuffleDownSync(mask, hi, delta, width);
  lo = CudaShuffleDownSync(mask, lo, delta, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
}

// Wrapper for __shfl_xor_sync. All threads in 'mask' must call this function in
// convergence, see comment above for details.
template <typename T>
__device__ T CudaShuffleXorSync(unsigned mask, T value, int lane_mask,
                                int width = warpSize) {
  assert(!(width & width - 1));
  assert(detail::CudaValidateShuffleSyncMask(
      mask, detail::CudaShuffleXorGetSrcLane(lane_mask, width)));
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, lane_mask, width);
#else
  return __shfl_xor(value, lane_mask, width);
#endif
}

// Variant of the (undocumented) version from the CUDA SDK, but using unsigned
// instead of float for lo and hi (which is incorrect with ftz, for example).
// See b/69446944.
__device__ inline double CudaShuffleXorSync(unsigned mask, double value,
                                            int lane_mask,
                                            int width = warpSize) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));
  hi = CudaShuffleXorSync(mask, hi, lane_mask, width);
  lo = CudaShuffleXorSync(mask, lo, lane_mask, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(value) : "r"(lo), "r"(hi));
  return value;
}

// Wrapper for __ldg.
template <typename T>
__host__ __device__ T CudaLdg(const T* address) {
#if __CUDA_ARCH__ >= 350
  return __ldg(address);
#else
  return *address;
#endif
}

__host__ __device__ inline bool CudaLdg(const bool* address) {
  return CudaLdg(reinterpret_cast<const char*>(address)) != 0;
}

__host__ __device__ inline std::complex<float> CudaLdg(
    const std::complex<float>* address) {
#if __CUDA_ARCH__ >= 350
  float2 mem = __ldg(reinterpret_cast<const float2*>(address));
  return std::complex<float>(mem.x, mem.y);
#else
  return *address;
#endif
}

__host__ __device__ inline std::complex<double> CudaLdg(
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
  for (int i : CudaGridRangeX(count)) {
    ptr[i] = T(0);
  }
}

// Helper to set all tensor entries to a specific value.
template <typename T>
__global__ void SetToValue(const int count, T* ptr, T value) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1 && blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (int i : CudaGridRangeX(count)) {
    ptr[i] = value;
  }
}

namespace detail {
// Helper function for atomic accumulation implemented as CAS.
template <typename T, typename F>
__device__ T CudaAtomicCasHelper(T* ptr, F accumulate) {
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
__device__ float CudaAtomicCasHelper(float* ptr, F accumulate) {
  return __float_as_int(
      CudaAtomicCasHelper(reinterpret_cast<int32*>(ptr), [accumulate](int32 a) {
        return __float_as_int(accumulate(__int_as_float(a)));
      }));
}
template <typename F>
__device__ double CudaAtomicCasHelper(double* ptr, F accumulate) {
  return __longlong_as_double(CudaAtomicCasHelper(
      reinterpret_cast<tensorflow::uint64*>(ptr),
      [accumulate](tensorflow::uint64 a) {
        return __double_as_longlong(accumulate(__longlong_as_double(a)));
      }));
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
__device__ Eigen::half CudaAtomicCasHelper(Eigen::half* ptr, F accumulate) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Not little endian");
#endif
  namespace half_impl = Eigen::half_impl;
  intptr_t intptr = reinterpret_cast<intptr_t>(ptr);
  assert(!(intptr & 0x1));  // should be 2-aligned.
  if (intptr & 0x2) {
    // The half is in the second part of the uint32 (upper 16 bits).
    uint32* address = reinterpret_cast<uint32*>(intptr - 2);
    uint32 result = CudaAtomicCasHelper(address, [accumulate](uint32 arg) {
      unsigned short high = static_cast<unsigned short>(arg >> 16);
      Eigen::half acc = accumulate(half_impl::raw_uint16_to_half(high));
      return (static_cast<uint32>(acc.x) << 16) | (arg & 0xffff);
    });
    return half_impl::raw_uint16_to_half(static_cast<uint16>(result >> 16));
  } else {
    // The half is in the first part of the uint32 (lower 16 bits).
    uint32* address = reinterpret_cast<uint32*>(intptr);
    uint32 result = CudaAtomicCasHelper(address, [accumulate](uint32 arg) {
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
__device__ detail::ToTypeIfConvertible<U, T> CudaAtomicAdd(T* ptr, U value) {
  return atomicAdd(ptr, value);
}

__device__ inline Eigen::half CudaAtomicAdd(Eigen::half* ptr,
                                            Eigen::half value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](Eigen::half a) { return a + value; });
}


#if __CUDA_ARCH__ < 600
__device__ inline double CudaAtomicAdd(double* ptr, double value) {
  return detail::CudaAtomicCasHelper(ptr,
                                     [value](double a) { return a + value; });
}
#elif __clang__
// Clang cannot compile __nvvm_atom_add_gen_d builtin yet, use inline PTX.
// see https://reviews.llvm.org/D39638
__device__ inline double CudaAtomicAdd(double* ptr, double value) {
  double result;
  asm volatile("atom.add.f64 %0, [%1], %2;"
               : "=d"(result)
               : "l"(ptr), "d"(value)
               : "memory");
  return result;
}
#endif
// CudaAtomicAdd
// Specializations of CudaAtomicAdd for complex types, which CudaAtomicAdd does
// not support. We treat a std::complex<T>* as a T* (the C++ standard section
// 26.4.4 allows this explicitly) and atomic add the real and imaginary
// components individually. The operation as a whole is not atomic, but we can
// safely treat the components independently for the purpose of accumulating.
__device__ inline std::complex<float> CudaAtomicAdd(std::complex<float>* ptr,
                                                    std::complex<float> value) {
  auto ptr_scalar = reinterpret_cast<float*>(ptr);
  return std::complex<float>(CudaAtomicAdd(ptr_scalar, value.real()),
                             CudaAtomicAdd(ptr_scalar + 1, value.imag()));
}

__device__ inline std::complex<double> CudaAtomicAdd(
    std::complex<double>* ptr, std::complex<double> value) {
  auto ptr_scalar = reinterpret_cast<double*>(ptr);
  return std::complex<double>(CudaAtomicAdd(ptr_scalar, value.real()),
                              CudaAtomicAdd(ptr_scalar + 1, value.imag()));
}

// CudaAtomicSub
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> CudaAtomicSub(T* ptr, U value) {
  return atomicSub(ptr, value);
}

// Specializations of substraction which add the negative value.
__device__ inline float CudaAtomicSub(float* ptr, float value) {
  return CudaAtomicAdd(ptr, -value);
}

__device__ inline double CudaAtomicSub(double* ptr, double value) {
  return CudaAtomicAdd(ptr, -value);
}

__device__ inline tensorflow::uint64 CudaAtomicSub(tensorflow::uint64* ptr,
                                                   tensorflow::uint64 value) {
  return CudaAtomicAdd(ptr, -value);
}

__device__ inline Eigen::half CudaAtomicSub(Eigen::half* ptr,
                                            Eigen::half value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](Eigen::half a) { return a - value; });
}

// CudaAtomicMax
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> CudaAtomicMax(T* ptr, U value) {
  return atomicMax(ptr, value);
}

__device__ inline float CudaAtomicMax(float* ptr, float value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](float a) { return max(a, value); });
}

__device__ inline double CudaAtomicMax(double* ptr, double value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](double a) { return max(a, value); });
}

__device__ inline Eigen::half CudaAtomicMax(Eigen::half* ptr,
                                            Eigen::half value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](Eigen::half a) { return max(a, value); });
}

#if __CUDA_ARCH__ < 320
__device__ inline tensorflow::uint64 CudaAtomicMax(tensorflow::uint64* ptr,
                                                   tensorflow::uint64 value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](tensorflow::uint64 a) { return max(a, value); });
}
#endif

// CudaAtomicMin
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> CudaAtomicMin(T* ptr, U value) {
  return atomicMin(ptr, value);
}

__device__ inline float CudaAtomicMin(float* ptr, float value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](float a) { return min(a, value); });
}

__device__ inline double CudaAtomicMin(double* ptr, double value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](double a) { return min(a, value); });
}

__device__ inline Eigen::half CudaAtomicMin(Eigen::half* ptr,
                                            Eigen::half value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](Eigen::half a) { return min(a, value); });
}

#if __CUDA_ARCH__ < 320
__device__ inline tensorflow::uint64 CudaAtomicMin(tensorflow::uint64* ptr,
                                                   tensorflow::uint64 value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](tensorflow::uint64 a) { return min(a, value); });
}
#endif

// CudaAtomicMul
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> CudaAtomicMul(T* ptr, U value) {
  return detail::CudaAtomicCasHelper(ptr, [value](T a) { return a * value; });
}

// CudaAtomicDiv
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> CudaAtomicDiv(T* ptr, U value) {
  return detail::CudaAtomicCasHelper(ptr, [value](T a) { return a / value; });
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_CORE_UTIL_CUDA_KERNEL_HELPER_H_
