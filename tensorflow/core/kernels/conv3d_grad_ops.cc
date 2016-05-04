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

// See docs in ../ops/nn_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/conv_3d.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

// #if GOOGLE_CUDA
// GPU operations not yet implemented.
// #include "tensorflow/core/kernels/conv3d_ops_gpu.h"
// #include "tensorflow/core/platform/stream_executor.h"
// #endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// The operation to compute Conv3D gradients.
//
// Largly based off the logic and operations for 2d convolutions in
// conv_grad_ops.cc

#define EXTRACT_AND_VERIFY_DIMENSIONS_3D(label)                                \
  const Tensor& out_backprop = context->input(2);                              \
  OP_REQUIRES(                                                                 \
      context, input_shape.dims() == 5,                                        \
      errors::InvalidArgument(label, ": input must be 5-dimensional"));        \
  OP_REQUIRES(                                                                 \
      context, filter_shape.dims() == 5,                                       \
      errors::InvalidArgument(label, ": filter must be 5-dimensional"));       \
  OP_REQUIRES(                                                                 \
      context, out_backprop.dims() == 5,                                       \
      errors::InvalidArgument(label, ": out_backprop must be 5-dimensional")); \
  const int64 batch = GetTensorDim(input_shape, data_format_, 'N');            \
  OP_REQUIRES(                                                                 \
      context, batch == GetTensorDim(out_backprop, data_format_, 'N'),         \
      errors::InvalidArgument(                                                 \
          label, ": input and out_backprop must have the same batch size"));   \
  const int64 input_depth = GetTensorDim(input_shape, data_format_, 'D');      \
  const int64 input_rows = GetTensorDim(input_shape, data_format_, 'H');       \
  const int64 input_cols = GetTensorDim(input_shape, data_format_, 'W');       \
  const int64 filter_depth = filter_shape.dim_size(0);                         \
  const int64 filter_rows = filter_shape.dim_size(1);                          \
  const int64 filter_cols = filter_shape.dim_size(2);                          \
  const int64 output_depth = GetTensorDim(out_backprop, data_format_, 'D');    \
  const int64 output_rows = GetTensorDim(out_backprop, data_format_, 'H');     \
  const int64 output_cols = GetTensorDim(out_backprop, data_format_, 'W');     \
  const int64 in_chan = GetTensorDim(input_shape, data_format_, 'C');          \
  OP_REQUIRES(context, in_chan == filter_shape.dim_size(3),                    \
              errors::InvalidArgument(                                         \
                  label, ": input and filter must have the same depth"));      \
  const int64 out_chan = filter_shape.dim_size(4);                             \
  OP_REQUIRES(                                                                 \
      context, out_chan == GetTensorDim(out_backprop, data_format_, 'C'),      \
      errors::InvalidArgument(                                                 \
          label, ": filter and out_backprop must have the same out_chan"));    \
  const auto stride_depth = GetTensorDim(strides_, data_format_, 'D');         \
  const auto stride_rows = GetTensorDim(strides_, data_format_, 'H');          \
  const auto stride_cols = GetTensorDim(strides_, data_format_, 'W');          \
  int out_depth = 0, out_rows = 0, out_cols = 0;                               \
  int pad_depth = 0, pad_rows = 0, pad_cols = 0;                               \
                                                                               \
  OP_REQUIRES_OK(                                                              \
        context,                                                               \
        Get3dOutputSize(input_rows, input_cols, input_depth,                   \
                        filter_rows, filter_cols, filter_depth,                \
                        stride_rows, stride_cols, stride_depth,                \
                        padding_, &out_rows, &out_cols, &out_depth,            \
                        &pad_rows, &pad_cols, &pad_depth) );                   \
                                                                               \
  OP_REQUIRES(                                                                 \
      context, output_rows == out_rows,                                        \
      errors::InvalidArgument(                                                 \
          label, ": Number of rows of out_backprop doesn't match computed: ",  \
          "actual = ", output_rows, ", computed = ", out_rows));               \
  OP_REQUIRES(                                                                 \
      context, output_cols == out_cols,                                        \
      errors::InvalidArgument(                                                 \
          label, ": Number of cols of out_backprop doesn't match computed: ",  \
          "actual = ", output_cols, ", computed = ", out_cols));               \
  OP_REQUIRES(                                                                 \
      context, output_depth == out_depth,                                      \
      errors::InvalidArgument(                                                 \
          label, ": Number of depth of out_backprop doesn't match computed: ", \
          "actual = ", output_depth, ", computed = ", out_depth));             \
  const auto expanded_out_depth = (output_depth - 1) * stride_depth + 1;       \
  const auto expanded_out_rows = (output_rows - 1) * stride_rows + 1;          \
  const auto expanded_out_cols = (output_cols - 1) * stride_cols + 1;          \
  const auto padded_out_depth = input_depth + filter_depth - 1;                \
  const auto padded_out_rows = input_rows + filter_rows - 1;                   \
  const auto padded_out_cols = input_cols + filter_cols - 1;                   \
  const int front_pad_depth = filter_depth - 1 - pad_depth;                    \
  const int top_pad_rows = filter_rows - 1 - pad_rows;                         \
  const int left_pad_cols = filter_cols - 1 - pad_cols;                        \
  const int back_pad_depth =                                                   \
      padded_out_depth - expanded_out_depth - front_pad_depth;                 \
  const int bottom_pad_rows =                                                  \
      padded_out_rows - expanded_out_rows - top_pad_rows;                      \
  const int right_pad_cols =                                                   \
      padded_out_cols - expanded_out_cols - left_pad_cols;                     \
  Eigen::DSizes<int, 5> strides{1, stride_depth, stride_rows, stride_cols, 1}; \
  VLOG(2) << "Conv3d: " << label                                               \
          << ": expanded_out_depth = " << expanded_out_depth                   \
          << ": expanded_out_rows = " << expanded_out_rows                     \
          << ", expanded_out_cols = " << expanded_out_cols                     \
          << ", filter_depth = " << filter_depth                               \
          << ", filter_rows = " << filter_rows                                 \
          << ", filter_cols = " << filter_cols                                 \
          << ", padded_out_depth = " << padded_out_depth                       \
          << ", padded_out_rows = " << padded_out_rows                         \
          << ", padded_out_cols = " << padded_out_cols                         \

namespace {

Status VectorToShape(const TTypes<int32>::ConstVec& sizes, TensorShape* out) {
  using Index = TTypes<int32>::ConstVec::Index;
  const Index dims = sizes.size();
  for (Index i = 0; i < dims; ++i) {
    if (sizes(i) >= 0) {
      out->AddDim(sizes(i));
    } else {
      return errors::InvalidArgument("Dimension ", sizes(i), " must be >= 0");
    }
  }

  return Status::OK();
}
}  // namespace

// The fast versions using eigen computations directly. They are only enabled
// for CPU for now since nvcc times out when trying to compile them.
// TODO(yangke): enable them for GPUs when we have a faster compiler.

template <typename Device, class T>
class Conv3DFastBackpropInputOp : public OpKernel {
 public:
  explicit Conv3DFastBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NDHWC,
                errors::InvalidArgument(
                    "Eigen Conv3DFastBackpropInputOp only supports NDHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[4] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv3DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context,
                   VectorToShape(input_sizes.vec<int32>(), &input_shape));
    const TensorShape& filter_shape = filter.shape();

    EXTRACT_AND_VERIFY_DIMENSIONS_3D("Conv3DBackpropInput");
    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));
    functor::CuboidConvolutionBackwardInput<Device, T>()(
        context->eigen_device<Device>(), in_backprop->tensor<T, 5>(),
        filter.tensor<T, 5>(), out_backprop.tensor<T, 5>(),
        input_depth, input_rows, input_cols,
        stride_depth, stride_rows, stride_cols);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv3DFastBackpropInputOp);
};

REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInput")
                            .Device(DEVICE_CPU)
                            // .Label("eigen_tensor")
                            .TypeConstraint<float>("T"),
                        Conv3DFastBackpropInputOp<CPUDevice, float>);

template <typename Device, class T>
class Conv3DFastBackpropFilterOp : public OpKernel {
 public:
  explicit Conv3DFastBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NDHWC,
                errors::InvalidArgument(
                    "Conv3DFastBackpropFilterOp only supports NHDWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[4] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));
    const TensorShape& input_shape = input.shape();
    TensorShape filter_shape;
    OP_REQUIRES_OK(context,
                   VectorToShape(filter_sizes.vec<int32>(), &filter_shape));

    EXTRACT_AND_VERIFY_DIMENSIONS_3D("Conv3DBackpropFilter");
    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    functor::CuboidConvolutionBackwardKernel<Device, T>()(
        context->eigen_device<Device>(), filter_backprop->tensor<T, 5>(),
        input.tensor<T, 5>(), out_backprop.tensor<T, 5>(),
        filter_depth, filter_rows, filter_cols,
        stride_depth, stride_rows, stride_cols);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv3DFastBackpropFilterOp);
};

REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilter")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        Conv3DFastBackpropFilterOp<CPUDevice, float>);

// // GPU definitions of both ops are not implemented yet.
#if GOOGLE_CUDA
// // TODO: Implement a version that compiles for GPU.
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
