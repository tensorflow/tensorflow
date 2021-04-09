/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/tensor_to_hash_bucket_op.h"
#include "third_party/farmhash_gpu/src/farmhash_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace {

// We set the buffer size to 47 as it is sufficient to cover the number of
// digits in any integer type or float type with 6 fractional digits.
constexpr int kSharedMemBufferSizePerThread = 47;
__device__ const char kNan[] = "nan";
__device__ const char kInf[] = "inf";
constexpr int kPrecision = 6;

template<typename T>
__device__ __forceinline__ void FillDigits(T val, int num_digits, int* i,
                                           char *buf) {
  eigen_assert(num_digits <= kSharedMemBufferSizePerThread - (*i));

  int factor = (val < 0 ? -1: 1);

  int num_digits_a = num_digits;
  do {
    int digit = static_cast<int>((val % 10) * factor);
    buf[(*i) + num_digits - 1] = digit + '0';
    val /= 10;
    num_digits--;
  } while(val != 0);

  (*i) += num_digits_a;
}

template<typename T>
__device__ __forceinline__ int IntegerToString(T val, char *buf) {
  int num_digits = 0;
  T val_a = val;
  do {
    val_a = val_a / 10;
    num_digits++;
  } while(val_a != 0);

  int i = 0;
  if (val < 0) {
    buf[i++] = '-';
  }

  FillDigits(val, num_digits, &i, buf);
  
  return i;
}

__device__ __forceinline__ int FloatToString(float val, char *buf) {
  int i = 0;
  if (isnan(val)) {
    for (auto c: kNan) {
      buf[i++] = c;
    }
    return i;
  }
  if (isinf(val)) {
    for (auto c: kInf) {
      buf[i++] = c;
    }
    return i;
  }

  bool is_neg = val < 0.0;
  if (is_neg) {
    val = -val;
    buf[i++] = '-';
  }

  // This position is reserved for the potential carry digit at the end.
  i++;
  int start_i = i;

  // Get the magnitude of the input number. And there should be at least one
  // digit before the decimal point.
  int m = static_cast<int>(log10(val));
  if (m < 1) {
    m = 0;
  }

  int digit;
  // Fill the digits of the integral and fractional parts.
  int fractional_digits = 0;
  while (fractional_digits < kPrecision) {
    double factor = pow(10.0, m);
    if (!isinf(factor) && factor > 0) {
      digit = floor(val / factor);
      val -= digit * factor;
      buf[i++] = digit + '0';
      if (m < 0) {
        fractional_digits++;
      }
    }
    if (m == 0) {
      buf[i++] = '.';
    }
    m--;
  }

  // Deal with the carry.
  double factor = pow(10.0, m);
  if (!isinf(factor) && factor > 0) {
    digit = floor(val / factor);
  }
  bool carry_detected = false;
  if (digit >= 5) {
    int sum = 0;
    int carry = 1;
    for (int j = i - 1; j >= start_i; j--) {
      if (buf[j] != '.') {
        sum = (carry + buf[j] - '0') % 10;
        carry = (carry + buf[j] - '0') / 10;
        buf[j] = sum + '0';
      }
    }
    if (carry != 0) {
      buf[start_i - 1] = carry + '0';
      carry_detected = true;
    }
  }

  // If the position reserved for carry is not used, we need to shift the
  // digits.
  if (!carry_detected) {
    for (int j = start_i - 1; j < i - 1; j++) {
      buf[j] = buf[j + 1];
    }
    i--;
  }

  return i;
}

template<typename T>
__device__ __forceinline__ int NumberToString(T val, char *buf) {
  return IntegerToString(val, buf);
}

template<>
__device__ __forceinline__ int NumberToString<float>(float val, char *buf) {
  return FloatToString(val, buf);
}

template<typename T>
__global__ void ComputeHashes(const T* __restrict__ vals, int vals_size,
                              int64 num_buckets, int64* __restrict__ hashes) {
  extern __shared__ char s[];

  GPU_1D_KERNEL_LOOP(tid, vals_size) {
    int size = NumberToString(
        vals[tid], s + threadIdx.x * kSharedMemBufferSizePerThread);
    uint64_t a_hash = ::util_gpu::Fingerprint64(
        s + threadIdx.x * kSharedMemBufferSizePerThread, size);
    int64 a_bucket = static_cast<int64>(a_hash % num_buckets);
    hashes[tid] = a_bucket;
  }
}

} // end namespace

namespace functor {

template <typename T>
void LaunchTensorToHashBucket<Eigen::GpuDevice, T>::operator()(
         OpKernelContext* c, const int64 num_buckets, const T* input,
         const int num_elems, int64* output) {
  auto* stream = c->op_device_context()->stream();
  const Eigen::GpuDevice& d = c->eigen_gpu_device();
  if (num_elems > 0) {
    constexpr size_t kThreadsLimitInBlock = 1024;

    size_t smem_bytes_allowed =
        stream->parent()->GetDeviceDescription().shared_memory_per_block();
    auto smem_bytes_per_thread = kSharedMemBufferSizePerThread * sizeof(char);
    size_t thread_per_block =
        min(kThreadsLimitInBlock, smem_bytes_allowed / smem_bytes_per_thread);

    auto smem_bytes_per_block = thread_per_block * smem_bytes_per_thread ;
    GpuLaunchConfig config = GetGpuLaunchConfigFixedBlockSize(
        num_elems, d, ComputeHashes<T>, smem_bytes_per_block, thread_per_block);
    OP_REQUIRES_OK(c, GpuLaunchKernel(
        ComputeHashes<T>, config.block_count, config.thread_per_block,
        smem_bytes_per_block, d.stream(), input, num_elems, num_buckets,
        output));
  }
}

} // namespace functor

#define REGISTER_FUNCTORS(type)                           \
    template struct functor::LaunchTensorToHashBucket<    \
        Eigen::GpuDevice, type>;

TF_CALL_INTEGRAL_TYPES(REGISTER_FUNCTORS);
REGISTER_FUNCTORS(float);

#undef REGISTER_FUNCTORS

} // namespace tensorflow
#endif // GOOGLE_CUDA

