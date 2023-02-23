/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
#define TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <type_traits>

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#endif
#include "tensorflow/core/util/gpu_cuda_alias.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_launch_config.h"

#if GOOGLE_CUDA
#define TF_RED_WARPSIZE 32
#elif TENSORFLOW_USE_ROCM
// We don't define TF_RED_WARPSIZE here, because it can be either 32 or 64
// and the value is not known at compile time.
#endif

// Deprecated, use 'for(int i : GpuGridRangeX(n))' instead.
#define GPU_1D_KERNEL_LOOP(i, n) \
  for (int i : ::tensorflow::GpuGridRangeX<int>(n))
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i : ::tensorflow::GpuGridRangeX<int>(n))

// Deprecated, use 'for(int i : GpuGridRange?(n))' instead.
#define GPU_AXIS_KERNEL_LOOP(i, n, axis) \
  for (int i : ::tensorflow::GpuGridRange##axis<int>(n))
#define CUDA_AXIS_KERNEL_LOOP(i, n, axis) \
  for (int i : ::tensorflow::GpuGridRange##axis<int>(n))

#if GOOGLE_CUDA
#define gpuSuccess cudaSuccess
using gpuStream_t = cudaStream_t;
using gpuError_t = cudaError_t;
#elif TENSORFLOW_USE_ROCM
#define gpuSuccess hipSuccess
using gpuStream_t = hipStream_t;
using gpuError_t = hipError_t;
#endif

// macro wrapper to declare dynamic shared memory
#if GOOGLE_CUDA

#define GPU_DYNAMIC_SHARED_MEM_DECL(ALIGN, TYPE, NAME) \
  extern __shared__ __align__(ALIGN) TYPE NAME[]

#elif TENSORFLOW_USE_ROCM

#define GPU_DYNAMIC_SHARED_MEM_DECL(ALIGN, TYPE, NAME) \
  HIP_DYNAMIC_SHARED(TYPE, NAME)

#endif

namespace tensorflow {

#if GOOGLE_CUDA
// cudaGetErrorString is available to both host and device
__host__ __device__ inline const char* GpuGetErrorString(cudaError_t error) {
  return cudaGetErrorString(error);
}
#elif TENSORFLOW_USE_ROCM
// hipGetErrorString is available on host side only
inline const char* GpuGetErrorString(hipError_t error) {
  return hipGetErrorString(error);
}
#endif

// Returns a raw reference to the current cuda stream. Required by a
// number of kernel calls (for which StreamInterface* does not work),
// i.e. CUB and certain cublas primitives.
inline const gpuStream_t& GetGpuStream(OpKernelContext* context) {
  const gpuStream_t* ptr = CHECK_NOTNULL(
      reinterpret_cast<const gpuStream_t*>(context->op_device_context()
                                               ->stream()
                                               ->implementation()
                                               ->GpuStreamMemberHack()));
  return *ptr;
}

// Launches a GPU kernel through cudaLaunchKernel in CUDA environment, or
// hipLaunchKernel in ROCm environment with the given arguments.
//
// The kernel parameters 'Ts' must be constructible from the arguments 'Args'.
template <typename... Ts, typename... Args>
Status GpuLaunchKernel(void (*function)(Ts...), dim3 grid_dim, dim3 block_dim,
                       size_t shared_memory_size_bytes, gpuStream_t stream,
                       Args... arguments) {
  static_assert(detail::NoneIsReference<Ts...>(),
                "Kernels with reference arguments have undefined behaviour.");
  if (grid_dim.x * grid_dim.y * grid_dim.z > 0 &&
      block_dim.x * block_dim.y * block_dim.z > 0) {
#if GOOGLE_CUDA
    auto func_ptr = absl::bit_cast<const void*>(function);
    // Cast arguments and forward them as an array of pointers.
    auto args_tuple = std::tuple<Ts...>(arguments...);
    auto arg_ptrs = detail::GetArrayOfElementPointers(&args_tuple);
    auto result =
        cudaLaunchKernel(func_ptr, grid_dim, block_dim, arg_ptrs.data(),
                         shared_memory_size_bytes, stream);
    if (result != cudaSuccess) {
      return errors::Internal(cudaGetErrorString(result));
    }
#elif TENSORFLOW_USE_ROCM
    hipLaunchKernelGGL(function, grid_dim, block_dim, shared_memory_size_bytes,
                       stream, std::forward<Args>(arguments)...);
    TF_RETURN_IF_CUDA_ERROR(hipGetLastError());
#endif
  }
  return OkStatus();
}

// Perfect forwarding to make CudaLaunchKernel available to both ROCm and CUDA
// builds
template <typename... Args>
auto CudaLaunchKernel(Args&&... args)
    -> decltype(GpuLaunchKernel(std::forward<Args>(args)...)) {
  return GpuLaunchKernel(std::forward<Args>(args)...);
}

__host__ __device__ inline tensorflow::bfloat16 GpuLdg(
    const tensorflow::bfloat16* address) {
  return Eigen::numext::bit_cast<tensorflow::bfloat16>(
      GpuLdg(reinterpret_cast<const uint16_t*>(address)));
}
// Already aliased in gpu_device_functions.h

template <typename T>
__host__ __device__ inline T ldg(const T* ptr) {
  return GpuLdg(ptr);
}

template <typename T>
__host__ __device__ inline const T& tf_min(const T& x, const T& y) {
  return x < y ? x : y;
}

template <typename T>
__host__ __device__ inline const T& tf_max(const T& x, const T& y) {
  return x < y ? y : x;
}

// Overloads of the above functions for float and double.
__host__ __device__ inline float tf_min(float x, float y) {
  return fminf(x, y);
}
__host__ __device__ inline double tf_min(double x, double y) {
  return fmin(x, y);
}
__host__ __device__ inline float tf_max(float x, float y) {
  return fmaxf(x, y);
}
__host__ __device__ inline double tf_max(double x, double y) {
  return fmax(x, y);
}

// ROCM TODO re-enable them after adding fp16 support logic
#if GOOGLE_CUDA
__device__ inline Eigen::half GpuShuffleSync(unsigned mask, Eigen::half value,
                                             int src_lane,
                                             int width = warpSize) {
  return Eigen::half(
      GpuShuffleSync(mask, static_cast<uint16>(value), src_lane, width));
}
// Aliased in gpu_device_functions.h

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleUpSync(
    unsigned mask, Eigen::half value, int delta, int width = warpSize) {
  return Eigen::half(
      GpuShuffleUpSync(mask, static_cast<uint16>(value), delta, width));
}
// Aliased in gpu_device_functions.h

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleDownSync(
    unsigned mask, Eigen::half value, int delta, int width = warpSize) {
  return Eigen::half(
      GpuShuffleDownSync(mask, static_cast<uint16>(value), delta, width));
}
// Aliased in gpu_device_functions.h

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleXorSync(
    unsigned mask, Eigen::half value, int lane_mask, int width = warpSize) {
  return Eigen::half(
      GpuShuffleXorSync(mask, static_cast<uint16>(value), lane_mask, width));
}
// Aliased in gpu_device_functions.h
#endif

#ifdef __CUDA_ARCH__
#define UNROLL_ON_DEVICE _Pragma("unroll")
#else
#define UNROLL_ON_DEVICE
#endif

// Represents an aligned array of N elements of T. Data pointers can be
// reinterpreted as this type to generate vectorized loads/stores in a kernel.
template <typename T, int N>
class alignas(alignof(T) * N) AlignedVector {
 public:
  typedef T value_type;
  static constexpr const int kSize = N;

  AlignedVector() = default;

  // Uniform initialization.
  __host__ __device__ explicit AlignedVector(value_type uniform) {
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) { values_[i] = uniform; }
  }
  // Uniform initialization with explicit conversion.
  // Note: This is required for T=Eigen::half because it only supports explicit
  // conversions from other types and its template constructor is too relaxed
  // to be able to use std::is_constructible.
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  __host__ __device__ explicit AlignedVector(U uniform_u) {
    value_type uniform(uniform_u);
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) { values_[i] = uniform; }
  }
  // Implicit conversion.
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value, int>::type = 0>
  __host__ __device__ AlignedVector(const AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) { values_[i] = other[i]; }
  }
  // Explicit conversion.
  template <typename U,
            typename std::enable_if<!std::is_convertible<U, T>::value &&
                                        std::is_constructible<T, U>::value,
                                    int>::type = 0>
  __host__ __device__ explicit AlignedVector(const AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) {
      values_[i] = T(other[i]);
    }
  }

  __host__ __device__ value_type& operator[](int i) { return values_[i]; }
  __host__ __device__ const value_type& operator[](int i) const {
    return values_[i];
  }

#define DEFINE_BINARY_UPDATE_OPERATOR(op)                                      \
  __host__ __device__ AlignedVector& operator op(const AlignedVector& rhs) {   \
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) { values_[i] op rhs[i]; } \
    return *this;                                                              \
  }
  DEFINE_BINARY_UPDATE_OPERATOR(+=)
  DEFINE_BINARY_UPDATE_OPERATOR(-=)
  DEFINE_BINARY_UPDATE_OPERATOR(*=)
  DEFINE_BINARY_UPDATE_OPERATOR(/=)
#undef DEFINE_BINARY_UPDATE_OPERATOR

#define DEFINE_BINARY_OPERATOR(op)                          \
  friend __host__ __device__ AlignedVector operator op(     \
      const AlignedVector& lhs, const AlignedVector& rhs) { \
    AlignedVector ret;                                      \
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) {      \
      ret[i] = lhs[i] op rhs[i];                            \
    }                                                       \
    return ret;                                             \
  }
  DEFINE_BINARY_OPERATOR(+)
  DEFINE_BINARY_OPERATOR(-)
  DEFINE_BINARY_OPERATOR(*)
  DEFINE_BINARY_OPERATOR(/)
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_FUNCTION(func)                                        \
  friend __host__ __device__ AlignedVector func(const AlignedVector& lhs,   \
                                                const AlignedVector& rhs) { \
    AlignedVector ret;                                                      \
    UNROLL_ON_DEVICE for (int i = 0; i < kSize; ++i) {                      \
      ret[i] = func(lhs[i], rhs[i]);                                        \
    }                                                                       \
    return ret;                                                             \
  }
  DEFINE_BINARY_FUNCTION(min)
  DEFINE_BINARY_FUNCTION(max)
#undef DEFINE_BINARY_FUNCTION

 private:
  value_type values_[N];
};

#undef UNROLL_ON_DEVICE

// Returns the maximum power-of-two alignment (in units of elements, not bytes)
// of a stride or pointer value.
inline int64_t alignment_of(int64_t element_stride) {
  // A zero/nullptr value means that the stride/pointer is not used, so it
  // effectively has infinite alignment.
  constexpr int64_t kMaxAlignment = 512;
  if (element_stride == 0) return kMaxAlignment;
  return element_stride & -element_stride;
}

template <typename T>
inline int64_t alignment_of(T* ptr) {
  const intptr_t ptr_val = reinterpret_cast<std::uintptr_t>(ptr);
  // Pointers should always be aligned to sizeof(T) bytes.
  DCHECK_EQ(ptr_val % sizeof(T), 0);
  // Note that we want the alignment in elements, not bytes.
  return alignment_of(ptr_val / sizeof(T));
}

template <typename... Args>
int64_t MinAlignmentOf(Args... args) {
  return std::min({alignment_of(args)...});
}

namespace detail {

template <int64_t VecSize, template <int vec_size> class Functor>
struct DispatchToVectorizedHelper {
  template <typename... Args>
  Status operator()(int64_t max_vec_size, Args&&... args) const {
    if (max_vec_size >= VecSize) {
      return Functor<VecSize>()(std::forward<Args>(args)...);
    }
    return DispatchToVectorizedHelper<VecSize / 2, Functor>()(
        max_vec_size, std::forward<Args>(args)...);
  }
};
template <template <int vec_size> class Functor>
struct DispatchToVectorizedHelper<1, Functor> {
  template <typename... Args>
  Status operator()(int64_t max_vec_size, Args&&... args) const {
    return Functor<1>()(std::forward<Args>(args)...);
  }
};

}  // namespace detail

// Calls Functor<vec_size>()(args...) with vec_size set to the optimal GPU
// vector instruction size for type T that is <= max_vec_size. The max_vec_size
// argument should be set to the minimum alignment of all relevant parameters.
// Requires sizeof(T) to be a power of 2.
template <typename T, template <int vec_size> class Functor, typename... Args>
Status DispatchToVectorized(int64_t max_vec_size, Args&&... args) {
  static_assert((sizeof(T) & (sizeof(T) - 1)) == 0,
                "sizeof(T) must be a power of 2");
  if (max_vec_size <= 0) {
    return errors::InvalidArgument("DispatchToVectorized: max_vec_size (",
                                   max_vec_size,
                                   ") must be greater than zero.");
  }
  constexpr const int kOptimalVecSizeBytes = 16;
  // The optimal number of (aligned) elements of T to load/store in a
  // single instruction inside a kernel.
  constexpr const int optimal_vec_size =
      (kOptimalVecSizeBytes - 1) / sizeof(T) + 1;
  return detail::DispatchToVectorizedHelper<optimal_vec_size, Functor>()(
      max_vec_size, std::forward<Args>(args)...);
}

// Similar to std::upper_bound, this returns the index of the first element in
// [first, first + count) that is greater than `val`, or `count` if no such
// element is found. Assumes [first, first + count) is sorted.
namespace gpu_helper {
template <typename T, typename OutType = int32, typename Iterator = const T*>
__device__ OutType upper_bound(Iterator first, OutType count, T val) {
  Iterator orig = first;
  OutType step = 0;
  while (count > 0) {
    Iterator it = first;
    step = count / 2;
    it += step;
    if (!(val < *it)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }

  return first - orig;
}

// Similar to std::lower_bound, this returns the index of the first element in
// [first, first + count) that is not less than `val`, or `count` if no such
// element is found. Assumes [first, first + count) is sorted.
template <typename T, typename OutType = int32, typename Iterator = const T*>
__device__ OutType lower_bound(Iterator first, OutType count, T val) {
  Iterator orig = first;
  OutType step = 0;
  while (count > 0) {
    Iterator it = first;
    step = count / 2;
    it += step;
    if (*it < val) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }

  return first - orig;
}

}  // namespace gpu_helper

#ifndef TENSORFLOW_USE_ROCM
namespace cuda_helper = gpu_helper;
#endif

// For int division, we can substitute the fast multiplication for slow
// division. For detailed information see:
//   https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
//
// Warning: This implementation only works when the divisor is [1, INT32_MAX]
//          and the numerator has to be [0, INT32_MAX]. This is enough for our
//          purpose for computing integer indices.
// Basics: the typical int division can be written as:
//   n / d = (m * n) / 2^(32 + s)
// where 'n' is the numerator and 'd' is the divisor. For a given 'd', we
// need to find a magic number 'm' and a shift 's'. See update_magic().
struct FastDividerUint32 {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC FastDividerUint32(uint32_t d)
      : divisor(d) {
    assert(divisor >= 1 && divisor <= INT32_MAX);
    update_magic();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void update_magic() {
    // (1). The shift 's' is calculated by log2ceil(d).
#if defined(__CUDA_ARCH__)
    shift = 32 - __clz(divisor - 1);
#else
    for (shift = 0; shift < 32; shift++) {
      if ((1U << shift) >= divisor) break;
    }
#endif

    // (2). The magic number 'm' is calculated by:
    //   m = 2^(32 + s) / d + 1
    // Note, the digit '1' is to round up 'm * n', which will be rounded down
    // later by dividing two. In practice, 'm' is a 33-bit value. To fit the
    // 32-bit range, we introduce:
    //   magic = m - 2^32
    //         = 2^(32 + s) / d - 2^32 + 1
    //         = 2^32 * 2^s / d - 2^32 * d / d + 1
    //         = (2^32 * (2^s - d)) / d + 1, where 'magic' will be in 32-bit.
    uint64_t m = (0x100000000ull * ((0x1ull << shift) - divisor)) / divisor + 1;
    magic = static_cast<uint32_t>(m);
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC FastDividerUint32& operator=(
      uint32_t d) {
    assert(divisor >= 1 && divisor <= INT32_MAX);
    this->divisor = d;
    update_magic();
    return *this;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC operator uint32_t() const {
    return divisor;
  }

  uint32_t divisor;
  uint32_t magic;
  uint32_t shift;
};

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint32_t
operator/(const uint32_t n, const FastDividerUint32& fdiv) {
  // (3). We use the 32-bit 'magic' instead of 'm' in the formula:
  //   n / d = (m * n) / 2^(32 + s)
  //         = (magic + 2^32) * n / 2^(32 + s)
  //         = (magic * n) / 2^(32 + s) + n / 2^s
  //         = (magic * n) / 2^32 / 2^s + n / 2^s
  //         = (magic * n / 2^32 + n) / 2^s
#if defined(__CUDA_ARCH__)
  uint32_t q = __umulhi(n, fdiv.magic);
#else
  uint32_t q =
      static_cast<uint32_t>((static_cast<uint64_t>(n) * fdiv.magic) >> 32);
#endif
  return (n + q) >> fdiv.shift;
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint32_t
operator%(const uint32_t n, const FastDividerUint32& fdiv) {
  return n - (n / fdiv) * fdiv.divisor;
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint32_t
operator/(const int n, const FastDividerUint32& fdiv) {
  return static_cast<uint32_t>(n) / fdiv;
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint32_t
operator%(const int n, const FastDividerUint32& fdiv) {
  return static_cast<uint32_t>(n) % fdiv;
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
