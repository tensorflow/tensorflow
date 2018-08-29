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

#include "tensorflow/core/kernels/bucketize_v2_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
struct BucketizeV2Functor<CPUDevice, T> {
  // PRECONDITION: boundaries must be sorted.
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& input,
                        const typename TTypes<float, 1>::ConstTensor& boundaries,
                        typename TTypes<int32, 1>::Tensor& output) {
    const int N = input.size();
    std::vector<float> boundaries_vector(boundaries.size());
    for(int i = 0; i < boundaries.size(); ++i){
      boundaries_vector[i] = boundaries(i);
    }
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
class BucketizeV2Op : public OpKernel {
 public:
  explicit BucketizeV2Op(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const auto input = input_tensor.flat<T>();

    const Tensor& boundaries_tensor = context->input(1);
    const auto boundaries = boundaries_tensor.flat<float>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();
    OP_REQUIRES_OK(context, functor::BucketizeV2Functor<Device, T>::Compute(
                                context, input, boundaries, output));
  }
};

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("BucketizeV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BucketizeV2Op<CPUDevice, T>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("BucketizeV2").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BucketizeV2Op<GPUDevice, T>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
