/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_CONV_OPS_3D_H_
#define TENSORFLOW_CORE_KERNELS_CONV_OPS_3D_H_

#include <vector>

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op_base.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/conv_3d.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#if GOOGLE_CUDA
#include "tensorflow/core/util/use_cudnn.h"
#endif

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, class OpKernelContextT>
struct LaunchConvOp;

template <typename T, class OpKernelContextT>
struct LaunchConvOp<CPUDevice, T, OpKernelContextT> {
  static void launch(OpKernelContextT* context, bool cudnn_use_autotune,
                     const Tensor& input, const Tensor& filter,
                     const std::array<int64, 3>& dilations,
                     const std::array<int64, 3>& strides, const Padding padding,
                     TensorFormat data_format, Tensor* output) {
    OP_REQUIRES(context, data_format == FORMAT_NHWC,
                errors::InvalidArgument("CPU implementation of Conv3D "
                                        "currently only supports the NHWC "
                                        "tensor format."));
    OP_REQUIRES(context,
                dilations[0] == 1 && dilations[1] == 1 && dilations[2] == 1,
                errors::InvalidArgument("CPU implementation of Conv3D "
                                        "currently only supports dilated rates "
                                        "of 1."));
    functor::CuboidConvolution<CPUDevice, T>()(
        context->template eigen_device<CPUDevice>(), output->tensor<T, 5>(),
        input.tensor<T, 5>(), filter.tensor<T, 5>(), strides[2], strides[1],
        strides[0], BrainPadding2EigenPadding(padding));
  }
};

template <typename Device, typename T, class OpKernelT,
          class OpKernelConstructionT, class OpKernelContextT>
class Conv3DOp : public BinaryOpBase<T, OpKernelT, OpKernelConstructionT> {
 public:
  explicit Conv3DOp(OpKernelConstructionT* context) :
      BinaryOpBase<T, OpKernelT, OpKernelConstructionT>(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'N') == 1 &&
         GetTensorDim(stride_, data_format_, 'C') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, '0') > 0 &&
         GetTensorDim(stride_, data_format_, '1') > 0 &&
         GetTensorDim(stride_, data_format_, '2') > 0),
        errors::InvalidArgument("Spatial strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'N') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'C') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));
    OP_REQUIRES(
        context,
        (GetTensorDim(dilation_, data_format_, '0') > 0 &&
         GetTensorDim(dilation_, data_format_, '1') > 0 &&
         GetTensorDim(dilation_, data_format_, '2') > 0),
        errors::InvalidArgument("Dilated rates should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
#if GOOGLE_CUDA
    cudnn_use_autotune_ = CudnnUseAutotune();
#else
    cudnn_use_autotune_ = false;
#endif
  }

  void Compute(OpKernelContextT* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_z, in_y, in_x, in_channels ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_z, filter_y, filter_x, in_channels, out_channels]
    const Tensor& filter = context->input(1);

    // NOTE: The ordering of the spatial dimensions is arbitrary, but has to be
    // kept consistent between input/filter/output.
    OP_REQUIRES(context, input.dims() == 5,
                errors::InvalidArgument("input must be 5-dimensional"));
    OP_REQUIRES(context, filter.dims() == 5,
                errors::InvalidArgument("filter must be 5-dimensional"));

    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    const int64 in_batch = GetTensorDim(input, data_format_, 'N');

    const int64 filter_depth = filter.dim_size(3);
    const int64 out_depth = filter.dim_size(4);

    OP_REQUIRES(context, in_depth % filter_depth == 0,
                errors::InvalidArgument(
                    "Input depth must be evenly divisible by filter depth: ",
                    in_depth, " vs ", filter_depth));

    // Dimension order for these arrays is: z, y, x.
    std::array<int64, 3> input_size = {
        {GetTensorDim(input, data_format_, '0'),
         GetTensorDim(input, data_format_, '1'),
         GetTensorDim(input, data_format_, '2')}};
    std::array<int64, 3> filter_size = {
        {filter.dim_size(0), filter.dim_size(1), filter.dim_size(2)}};
    std::array<int64, 3> dilations = {
        {GetTensorDim(dilation_, data_format_, '0'),
         GetTensorDim(dilation_, data_format_, '1'),
         GetTensorDim(dilation_, data_format_, '2')}};
    std::array<int64, 3> strides = {{GetTensorDim(stride_, data_format_, '0'),
                                     GetTensorDim(stride_, data_format_, '1'),
                                     GetTensorDim(stride_, data_format_, '2')}};
    std::array<int64, 3> out, padding;

    OP_REQUIRES_OK(
        context, Get3dOutputSizeV2(input_size, filter_size, dilations, strides,
                                   padding_, &out, &padding));
    TensorShape out_shape = ShapeFromFormat(
        data_format_, in_batch, {{out[0], out[1], out[2]}}, out_depth);
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Return early if nothing to do.
    if (out_shape.num_elements() == 0) return;

    LaunchConvOp<Device, T, OpKernelContextT>::launch(
        context, cudnn_use_autotune_, input, filter,
        dilations, strides, padding_, data_format_,
        output);
  }

 private:
  std::vector<int32> dilation_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool cudnn_use_autotune_;
};

}  // namespace tensorflow


#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_3D_H_
