/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.
#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/colorspace_op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
class RGBToHSVOp : public OpKernel {
 public:
  explicit RGBToHSVOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, channels == 3,
                errors::FailedPrecondition(
                    "input must have 3 channels but input only has ", channels,
                    " channels."));

    // Create the output Tensor with the same dimensions as the input Tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    // Make a canonical image, maintaining the last (channel) dimension, while
    // flattening all others do give the functor easy to work with data.
    TTypes<float, 2>::ConstTensor input_data = input.flat_inner_dims<float>();
    TTypes<float, 2>::Tensor output_data = output->flat_inner_dims<float>();

    Tensor trange;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<float>::value,
                                        TensorShape({input_data.dimension(0)}),
                                        &trange));

    TTypes<float, 1>::Tensor range = trange.tensor<float, 1>();

    functor::RGBToHSV<Device>()(context->eigen_device<Device>(), input_data,
                                range, output_data);
  }
};

template <typename Device>
class HSVToRGBOp : public OpKernel {
 public:
  explicit HSVToRGBOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, channels == 3,
                errors::FailedPrecondition(
                    "input must have 3 channels but input only has ", channels,
                    " channels."));

    // Create the output Tensor with the same dimensions as the input Tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    TTypes<float, 2>::ConstTensor input_data = input.flat_inner_dims<float>();
    TTypes<float, 2>::Tensor output_data = output->flat_inner_dims<float>();

    functor::HSVToRGB<Device>()(context->eigen_device<Device>(), input_data,
                                output_data);
  }
};

REGISTER_KERNEL_BUILDER(Name("RGBToHSV").Device(DEVICE_CPU),
                        RGBToHSVOp<CPUDevice>);
template class RGBToHSVOp<CPUDevice>;
REGISTER_KERNEL_BUILDER(Name("HSVToRGB").Device(DEVICE_CPU),
                        HSVToRGBOp<CPUDevice>);
template class HSVToRGBOp<CPUDevice>;

#if GOOGLE_CUDA
// Forward declarations of the function specializations for GPU (to prevent
// building the GPU versions here, they will be built compiling _gpu.cu.cc).
namespace functor {
template <>
void RGBToHSV<GPUDevice>::operator()(const GPUDevice& d,
                                     TTypes<float, 2>::ConstTensor input_data,
                                     TTypes<float, 1>::Tensor range,
                                     TTypes<float, 2>::Tensor output_data);
extern template struct RGBToHSV<GPUDevice>;
template <>
void HSVToRGB<GPUDevice>::operator()(const GPUDevice& d,
                                     TTypes<float, 2>::ConstTensor input_data,
                                     TTypes<float, 2>::Tensor output_data);
extern template struct HSVToRGB<GPUDevice>;
}  // namespace functor
REGISTER_KERNEL_BUILDER(Name("RGBToHSV").Device(DEVICE_GPU),
                        RGBToHSVOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HSVToRGB").Device(DEVICE_GPU),
                        HSVToRGBOp<GPUDevice>);
#endif

}  // namespace tensorflow
