/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "sparsemax_functor.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SparsemaxOp : public OpKernel {
 public:
  explicit SparsemaxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& logits_in = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));

    // Create an output tensor
    Tensor* probability_out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, logits_in.shape(),
                                                     &probability_out));

    // Create temporary tensor used for storing sorted values. The tensor is
    // only used in the GPU op.
    Tensor temp_mat_tensor;
    Tensor temp_vec_tensor;
    if (std::is_same<Device, GPUDevice>::value) {
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                     TensorShape({logits_in.dim_size(0)}),
                                                     &temp_vec_tensor));
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                    logits_in.shape(),
                                                    &temp_mat_tensor));
    } else {
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                     TensorShape({0}),
                                                     &temp_vec_tensor));
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                    TensorShape({0, 0}),
                                                    &temp_mat_tensor));
    }

    // Setup data view
    auto input = logits_in.matrix<T>();
    auto temp_vec = temp_vec_tensor.flat<T>();
    auto temp_mat = temp_mat_tensor.matrix<T>();
    auto output = probability_out->matrix<T>();

    const Device& eigen_device = context->eigen_device<Device>();
    functor::Sparsemax<Device, T>()(eigen_device, input,
                                    temp_vec, temp_mat, output);
  }
};

#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(                 \
    Name("Sparsemax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    SparsemaxOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU

#if GOOGLE_CUDA

#define REGISTER_GPU(T) REGISTER_KERNEL_BUILDER(                 \
    Name("Sparsemax").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
    SparsemaxOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
