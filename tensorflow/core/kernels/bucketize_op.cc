/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

#include "tensorflow/core/kernels/bucketize_op.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
struct BucketizeFunctor<CPUDevice, T> {
  // PRECONDITION: boundaries_vector must be sorted.
  static absl::Status Compute(OpKernelContext* context,
                              const typename TTypes<T, 1>::ConstTensor& input,
                              const std::vector<float>& boundaries_vector,
                              typename TTypes<int32_t, 1>::Tensor& output) {
    const int64_t N = input.size();
    const float* const boundaries_begin = boundaries_vector.data();
    const float* const boundaries_end =
        boundaries_begin + boundaries_vector.size();
    const T* const input_data = input.data();
    int32_t* const output_data = output.data();
    auto work = [&](int64_t start, int64_t limit) {
      for (int64_t i = start; i < limit; ++i) {
        output_data[i] = static_cast<int32_t>(
            std::upper_bound(boundaries_begin, boundaries_end, input_data[i]) -
            boundaries_begin);
      }
    };

    // Each element is an independent binary search, so shard the input across
    // the CPU worker threads. The cost estimate grows with the number of
    // comparisons per search (same shape as searchsorted_op's
    // Log2Ceiling(num_inputs), with a small multiplier so ParallelFor does not
    // overshard tiny-boundary cases). Small inputs stay on the caller thread,
    // where dispatch overhead would outweigh the work. NaN inputs keep the
    // historical IEEE upper_bound placement (past the last boundary); this is
    // intentional and matches prior Bucketize behavior.
    constexpr int64_t kMinParallelElements = 4096;
    if (N < kMinParallelElements) {
      work(0, N);
      return absl::OkStatus();
    }
    const int64_t cost_per_unit = std::max<int64_t>(
        1, 4 * Log2Ceiling64(boundaries_vector.size() + 1));
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        N, cost_per_unit, work);

    return absl::OkStatus();
  }
};

}  // namespace functor

template <typename Device, typename T>
class BucketizeOp : public OpKernel {
 public:
  explicit BucketizeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("boundaries", &boundaries_));
    OP_REQUIRES(context, std::is_sorted(boundaries_.begin(), boundaries_.end()),
                absl::InvalidArgumentError("Expected sorted boundaries"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const auto input = input_tensor.flat<T>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32_t>();
    if (input.size() > 0) {
      OP_REQUIRES_OK(context, functor::BucketizeFunctor<Device, T>::Compute(
                                  context, input, boundaries_, output));
    }
  }

 private:
  std::vector<float> boundaries_;
};

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Bucketize").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BucketizeOp<CPUDevice, T>);

REGISTER_KERNEL(int32_t);
REGISTER_KERNEL(int64_t);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Bucketize").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BucketizeOp<GPUDevice, T>);

REGISTER_KERNEL(int32_t);
REGISTER_KERNEL(int64_t);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
