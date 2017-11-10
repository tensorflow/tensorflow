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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using thread::ThreadPool;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct BucketizeFunctor<CPUDevice, T> {
  // PRECONDITION: boundaries_vector must be sorted.
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& input,
                        const std::vector<float>& boundaries_vector,
                        typename TTypes<int32, 1>::Tensor& output) {
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      auto first_bigger_it = std::upper_bound(
          boundaries_vector.begin(), boundaries_vector.end(), input(i));
      output(i) = first_bigger_it - boundaries_vector.begin();
    }

    return Status::OK();
  }
};
}  // namespace functor

template <typename Device, typename T>
class BucketizeOp : public OpKernel {
 public:
  explicit BucketizeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("boundaries", &boundaries_));
    OP_REQUIRES(context, std::is_sorted(boundaries_.begin(), boundaries_.end()),
                errors::InvalidArgument("Expected sorted boundaries"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const auto input = input_tensor.flat<T>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();
    OP_REQUIRES_OK(context, functor::BucketizeFunctor<Device, T>::Compute(
                                context, input, boundaries_, output));
  }

 private:
  std::vector<float> boundaries_;
};

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Bucketize").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BucketizeOp<CPUDevice, T>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Bucketize").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BucketizeOp<GPUDevice, T>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
