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

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conv_3d.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
using perftools::gputools::dnn::DimIndex;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// TODO(mjanusz): Get rid of the macro and return shapes directly.
#define EXTRACT_AND_VERIFY_DIMENSIONS(label)                                   \
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
  const int64 batch = input_shape.dim_size(0);                                 \
  OP_REQUIRES(                                                                 \
      context, batch == out_backprop.dim_size(0),                              \
      errors::InvalidArgument(                                                 \
          label, ": input and out_backprop must have the same batch size"));   \
  const std::array<int64, 3> input_size = {                                    \
      {GetTensorDim(input_shape, data_format_, '0'),                           \
       GetTensorDim(input_shape, data_format_, '1'),                           \
       GetTensorDim(input_shape, data_format_, '2')}};                         \
  const int64 in_depth = GetTensorDim(input_shape, data_format_, 'C');         \
  const std::array<int64, 3> filter_size = {{filter_shape.dim_size(0),         \
                                             filter_shape.dim_size(1),         \
                                             filter_shape.dim_size(2)}};       \
  const int64 output_cols = GetTensorDim(out_backprop, data_format_, '2');     \
  const int64 output_rows = GetTensorDim(out_backprop, data_format_, '1');     \
  const int64 output_planes = GetTensorDim(out_backprop, data_format_, '0');   \
  OP_REQUIRES(context, in_depth == filter_shape.dim_size(3),                   \
              errors::InvalidArgument(                                         \
                  label, ": input and filter must have the same depth"));      \
  const int64 out_depth = filter_shape.dim_size(4);                            \
  OP_REQUIRES(                                                                 \
      context, out_depth == GetTensorDim(out_backprop, data_format_, 'C'),     \
      errors::InvalidArgument(                                                 \
          label, ": filter and out_backprop must have the same out_depth"));   \
  const std::array<int64, 3> strides = {                                       \
      {GetTensorDim(stride_, data_format_, '0'),                               \
       GetTensorDim(stride_, data_format_, '1'),                               \
       GetTensorDim(stride_, data_format_, '2')}};                             \
  std::array<int64, 3> out, padding;                                           \
  OP_REQUIRES_OK(context, Get3dOutputSize(input_size, filter_size, strides,    \
                                          padding_, &out, &padding));          \
  OP_REQUIRES(context, output_planes == out[0],                                \
              errors::InvalidArgument(                                         \
                  label,                                                       \
                  ": Number of planes of out_backprop doesn't match "          \
                  "computed:  actual = ",                                      \
                  output_planes, ", computed = ", out[0]));                    \
  OP_REQUIRES(                                                                 \
      context, output_rows == out[1],                                          \
      errors::InvalidArgument(                                                 \
          label, ": Number of rows of out_backprop doesn't match computed: ",  \
          "actual = ", output_rows, ", computed = ", out[1]));                 \
  OP_REQUIRES(                                                                 \
      context, output_cols == out[2],                                          \
      errors::InvalidArgument(                                                 \
          label, ": Number of cols of out_backprop doesn't match computed: ",  \
          "actual = ", output_cols, ", computed = ", out[2]));                 \
  const auto expanded_out_planes = (output_planes - 1) * strides[0] + 1;       \
  const auto expanded_out_rows = (output_rows - 1) * strides[1] + 1;           \
  const auto expanded_out_cols = (output_cols - 1) * strides[2] + 1;           \
  const auto padded_out_planes = input_size[0] + filter_size[0] - 1;           \
  const auto padded_out_rows = input_size[1] + filter_size[1] - 1;             \
  const auto padded_out_cols = input_size[2] + filter_size[2] - 1;             \
  const auto top_pad_planes = filter_size[0] - 1 - padding[0];                 \
  const auto top_pad_rows = filter_size[1] - 1 - padding[1];                   \
  const auto left_pad_cols = filter_size[2] - 1 - padding[2];                  \
  const auto bottom_pad_planes =                                               \
      padded_out_planes - expanded_out_planes - top_pad_planes;                \
  const auto bottom_pad_rows =                                                 \
      padded_out_rows - expanded_out_rows - top_pad_rows;                      \
  const auto right_pad_cols =                                                  \
      padded_out_cols - expanded_out_cols - left_pad_cols;                     \
  VLOG(2) << "Conv3d: " << label                                               \
          << ": expanded_out_planes = " << expanded_out_planes                 \
          << ": expanded_out_rows = " << expanded_out_rows                     \
          << ", expanded_out_cols = " << expanded_out_cols                     \
          << ", padded_out_planes = " << padded_out_planes                     \
          << ", padded_out_rows = " << padded_out_rows                         \
          << ", padded_out_cols = " << padded_out_cols                         \
          << ", top_pad_planes = " << top_pad_planes                           \
          << ", top_pad_rows = " << top_pad_rows                               \
          << ", left_pad_cols = " << left_pad_cols                             \
          << ", bottom_pad_planes = " << bottom_pad_planes                     \
          << ", bottom_pad_rows = " << bottom_pad_rows                         \
          << ", right_pad_cols = " << right_pad_cols

// Backprop for input.
template <typename Device, class T>
class Conv3DBackpropInputOp : public OpKernel {
 public:
  explicit Conv3DBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context),
        data_format_(FORMAT_NHWC),
        takes_shape_(type_string().find("V2") != std::string::npos) {
    // data_format is only available in V2.
    if (takes_shape_) {
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Conv3DBackpropInputOpV2 only supports NDHWC on the CPU."));
    }

    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'C') == 1 &&
         GetTensorDim(stride_, data_format_, 'N') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filter = context->input(1);
    const TensorShape& filter_shape = filter.shape();
    TensorShape input_shape;
    if (takes_shape_) {
      const Tensor& input_sizes = context->input(0);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                  input_sizes.vec<int32>(), &input_shape));
    } else {
      input_shape = context->input(0).shape();
    }
    EXTRACT_AND_VERIFY_DIMENSIONS("Conv3DBackpropInput");
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 5> pad_dims{
        {0, 0},
        {top_pad_planes, bottom_pad_planes},
        {top_pad_rows, bottom_pad_rows},
        {left_pad_cols, right_pad_cols},
        {0, 0}};
    Tensor* in_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    // Fill out a padded out_backprop.
    TensorShape padded_out_shape({batch, padded_out_planes, padded_out_rows,
                                  padded_out_cols, out_depth});
    Tensor padded_output;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::v(),
                                          padded_out_shape, &padded_output));
    Eigen::DSizes<Eigen::DenseIndex, 5> no_op_shuffle{0, 1, 2, 3, 4};
    Eigen::DSizes<Eigen::DenseIndex, 5> eigen_strides{1, strides[0], strides[1],
                                                      strides[2], 1};
    functor::InflatePadAndShuffle<Device, T, 5, Eigen::DenseIndex>()(
        context->eigen_device<Device>(), out_backprop.tensor<T, 5>(),
        eigen_strides, pad_dims, no_op_shuffle, padded_output.tensor<T, 5>());
    const Tensor& padded_output_cref = padded_output;

    // Fill a new "reverted" filter. We need to transpose the in_depth and
    // out_depth for the filter and reverse the planes, rows and cols.
    TensorShape r_filter_shape(
        {filter_size[0], filter_size[1], filter_size[2], out_depth, in_depth});
    Tensor r_filter;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                   r_filter_shape, &r_filter));
    Eigen::DSizes<Eigen::DenseIndex, 5> filter_order{0, 1, 2, 4, 3};
    Eigen::array<bool, 5> filter_rev_dims{true, true, true, false, false};
    functor::ShuffleAndReverse<Device, T, 5, Eigen::DenseIndex>()(
        context->eigen_device<Device>(), filter.tensor<T, 5>(), filter_order,
        filter_rev_dims, r_filter.tensor<T, 5>());
    const Tensor& r_filter_cref = r_filter;

    // Now we can call conv_3d directly.
    functor::CuboidConvolution<Device, T>()(
        context->eigen_device<Device>(), in_backprop->tensor<T, 5>(),
        padded_output_cref.tensor<T, 5>(), r_filter_cref.tensor<T, 5>(), 1, 1,
        1, BrainPadding2EigenPadding(VALID));
  }

 private:
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool takes_shape_;
};

#define REGISTER_CPU_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Conv3DBackpropInput").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      Conv3DBackpropInputOp<CPUDevice, T>);                                    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Conv3DBackpropInputV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv3DBackpropInputOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

// Backprop for filter.
template <typename Device, class T>
class Conv3DBackpropFilterOp : public OpKernel {
 public:
  explicit Conv3DBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context),
        data_format_(FORMAT_NHWC),
        takes_shape_(type_string().find("V2") != std::string::npos) {
    // data_format is only available in V2.
    if (takes_shape_) {
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Conv3DBackpropFilterOpV2 only supports NDHWC on the CPU."));
    }

    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'C') == 1 &&
         GetTensorDim(stride_, data_format_, 'N') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    TensorShape filter_shape;

    if (takes_shape_) {
      const Tensor& filter_sizes = context->input(1);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                  filter_sizes.vec<int32>(), &filter_shape));
    } else {
      filter_shape = context->input(1).shape();
    }

    EXTRACT_AND_VERIFY_DIMENSIONS("Conv3DBackpropFilter");
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 5> pad_dims{
        {0, 0},
        {top_pad_planes, bottom_pad_planes},
        {top_pad_rows, bottom_pad_rows},
        {left_pad_cols, right_pad_cols},
        {0, 0}};
    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    if (input_shape.num_elements() == 0) {
      filter_backprop->template flat<T>().setZero();
      return;
    }

    // For the backprop of the filter, we need to also transpose the
    // out_backprop.
    // The shape of backprop is
    //   [batch, out_z, out_y, out_x, out_depth]
    // And we need to change it to
    //   [out_depth, out_x, out_y, out_z, batch]
    Eigen::DSizes<Eigen::DenseIndex, 5> out_order{4, 1, 2, 3, 0};
    TensorShape padded_out_shape({out_depth, padded_out_planes, padded_out_rows,
                                  padded_out_cols, batch});
    Tensor padded_output;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::v(),
                                          padded_out_shape, &padded_output));
    Eigen::DSizes<Eigen::DenseIndex, 5> eigen_strides{1, strides[0], strides[1],
                                                      strides[2], 1};
    functor::InflatePadAndShuffle<Device, T, 5, Eigen::DenseIndex>()(
        context->eigen_device<Device>(), out_backprop.tensor<T, 5>(),
        eigen_strides, pad_dims, out_order, padded_output.tensor<T, 5>());
    const Tensor& padded_output_cref = padded_output;

    // For the backprop of the filter, we need to transpose the input.
    // The shape of input is
    //   [batch, in_z, in_y, in_x, in_depth]
    // And we need to change it to
    //   [in_z, in_y, in_x, batch, in_depth]
    Eigen::DSizes<Eigen::DenseIndex, 5> in_order{1, 2, 3, 0, 4};
    TensorShape in_shuffle_shape(
        {input_size[0], input_size[1], input_size[2], batch, in_depth});
    Tensor in_shuffle;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::v(),
                                          in_shuffle_shape, &in_shuffle));
    // No need for reversing this time.
    Eigen::array<bool, 5> no_reverse{false, false, false, false, false};
    functor::ShuffleAndReverse<Device, T, 5, Eigen::DenseIndex>()(
        context->eigen_device<Device>(), input.tensor<T, 5>(), in_order,
        no_reverse, in_shuffle.tensor<T, 5>());
    const Tensor& in_shuffle_cref = in_shuffle;

    // The output of the conv_3d would be
    //   [out_depth, filter_size[2], filter_size[1], filter_size[0], in_depth]
    // and we need to shuffle it back to
    //   [filter_size[2], filter_size[1], filter_size[0], in_depth, out_depth];
    // And we need to reverse the filter backprops.
    // So we need to allocate (sigh) yet another piece of memory to hold the
    // output.
    TensorShape filter_shuffle_shape(
        {out_depth, filter_size[0], filter_size[1], filter_size[2], in_depth});
    Tensor filter_shuffle;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                        filter_shuffle_shape, &filter_shuffle));
    functor::CuboidConvolution<Device, T>()(
        context->eigen_device<Device>(), filter_shuffle.tensor<T, 5>(),
        padded_output_cref.tensor<T, 5>(), in_shuffle_cref.tensor<T, 5>(), 1, 1,
        1, BrainPadding2EigenPadding(VALID));

    // Now copy the filter_backprop back to the destination.
    Eigen::DSizes<Eigen::DenseIndex, 5> filter_order{1, 2, 3, 4, 0};
    Eigen::array<bool, 5> filter_rev_dims{true, true, true, false, false};
    const Tensor& filter_shuffle_cref = filter_shuffle;
    functor::ShuffleAndReverse<Device, T, 5, Eigen::DenseIndex>()(
        context->eigen_device<Device>(), filter_shuffle_cref.tensor<T, 5>(),
        filter_order, filter_rev_dims, filter_backprop->tensor<T, 5>());
  }

 private:
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool takes_shape_;
};

#define REGISTER_CPU_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv3DBackpropFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv3DBackpropFilterOp<CPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilterV2")                      \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T"),                        \
                          Conv3DBackpropFilterOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

// GPU definitions of both ops.
#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
// This ensures that the custom implementation is used instead of the default
// Eigen one (which is used for CPU).
namespace functor {
#define DECLARE_GPU_SPEC(T)                                           \
  template <>                                                         \
  void TransformFilter<GPUDevice, T, int, 5>::operator()(             \
      const GPUDevice& d, typename TTypes<T, 5, int>::ConstTensor in, \
      typename TTypes<T, 5, int>::Tensor out);                        \
  template <>                                                         \
  void ReverseTransformFilter<GPUDevice, T, 5>::operator()(           \
      const GPUDevice& d, typename TTypes<T, 5>::ConstTensor in,      \
      typename TTypes<T, 5>::Tensor out);                             \
  template <>                                                         \
  void PadInput<GPUDevice, T, int, 5>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 5, int>::ConstTensor in, \
      const std::array<int, 3>& padding_left,                         \
      const std::array<int, 3>& padding_right,                        \
      typename TTypes<T, 5, int>::Tensor out, TensorFormat format);

DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// A dummy type to group backward data autotune results together.
struct Conv3dBackwardDataAutoTuneGroup {
  static string name() { return "Conv3dBwdData"; }
};
typedef AutoTuneSingleton<Conv3dBackwardDataAutoTuneGroup, ConvParameters,
                          perftools::gputools::dnn::AlgorithmConfig>

    AutoTuneConv3dBwdData;
template <typename T>
class Conv3DBackpropInputOp<GPUDevice, T> : public OpKernel {
 public:
  explicit Conv3DBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context),
        data_format_(FORMAT_NHWC),
        takes_shape_(type_string().find("V2") != std::string::npos) {
    // data_format is only available in V2.
    if (takes_shape_) {
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    }
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'C') == 1 &&
         GetTensorDim(stride_, data_format_, 'N') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }
  void Compute(OpKernelContext* context) override {
    const Tensor& filter = context->input(1);
    const TensorShape& filter_shape = filter.shape();
    TensorShape input_shape;
    if (takes_shape_) {
      const Tensor& input_sizes = context->input(0);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                  input_sizes.vec<int32>(), &input_shape));
    } else {
      input_shape = context->input(0).shape();
    }
    EXTRACT_AND_VERIFY_DIMENSIONS("Conv3DBackpropInput");
    Tensor* in_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    if (filter_size[0] == 1 && filter_size[1] == 1 && filter_size[2] == 1 &&
        stride_[0] == 1 && stride_[1] == 1 && stride_[2] == 1 &&
        data_format_ == FORMAT_NHWC) {
      const uint64 m = batch * input_size[0] * input_size[1] * input_size[2];
      const uint64 k = out_depth;
      const uint64 n = in_depth;

      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                  in_backprop->template flat<T>().size());

      auto transpose = perftools::gputools::blas::Transpose::kTranspose;
      auto no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;

      bool blas_launch_status =
          stream
              ->ThenBlasGemm(transpose, no_transpose, n, m, k, 1.0f, b_ptr, k,
                             a_ptr, k, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    } else if (filter_size[0] == input_size[0] &&
               filter_size[1] == input_size[1] &&
               filter_size[2] == input_size[2] && padding_ == Padding::VALID &&
               data_format_ == FORMAT_NHWC) {
      const uint64 m = batch;
      const uint64 k = out_depth;
      const uint64 n = input_size[0] * input_size[1] * input_size[2] * in_depth;

      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                  in_backprop->template flat<T>().size());

      auto transpose = perftools::gputools::blas::Transpose::kTranspose;
      auto no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;

      bool blas_launch_status =
          stream
              ->ThenBlasGemm(transpose, no_transpose, n, m, k, 1.0f, b_ptr, k,
                             a_ptr, k, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    }

    int padding_rows = 0, padding_cols = 0, padding_planes = 0;

    if (padding_ == Padding::SAME) {
      padding_planes = std::max<int>(
          0, (output_planes - 1) * strides[0] + filter_size[0] - input_size[0]);
      padding_cols = std::max<int>(
          0, (output_cols - 1) * strides[2] + filter_size[2] - input_size[2]);
      padding_rows = std::max<int>(
          0, (output_rows - 1) * strides[1] + filter_size[1] - input_size[1]);
    }
    const bool rows_odd = (padding_rows % 2 != 0);
    const bool cols_odd = (padding_cols % 2 != 0);
    const bool planes_odd = (padding_planes % 2 != 0);

    TensorShape compatible_input_shape;
    if (rows_odd || cols_odd || planes_odd) {
      // cuDNN only supports the same amount of padding on both sides.
      compatible_input_shape = {
          batch,
          in_depth,
          input_size[0] + planes_odd,
          input_size[1] + rows_odd,
          input_size[2] + cols_odd,
      };
    } else {
      compatible_input_shape = {batch, in_depth, input_size[0], input_size[1],
                                input_size[2]};
    }

    CHECK(padding_rows >= 0 && padding_cols >= 0 && padding_planes >= 0)
        << "Negative paddings: (" << padding_rows << ", " << padding_cols
        << ", " << padding_planes << ")";
    perftools::gputools::dnn::BatchDescriptor input_desc(3);
    input_desc.set_count(batch)
        .set_spatial_dim(DimIndex::X, compatible_input_shape.dim_size(4))
        .set_spatial_dim(DimIndex::Y, compatible_input_shape.dim_size(3))
        .set_spatial_dim(DimIndex::Z, compatible_input_shape.dim_size(2))
        .set_feature_map_count(in_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::BatchDescriptor output_desc(3);
    output_desc.set_count(batch)
        .set_spatial_dim(DimIndex::X, output_cols)
        .set_spatial_dim(DimIndex::Y, output_rows)
        .set_spatial_dim(DimIndex::Z, output_planes)
        .set_feature_map_count(out_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::FilterDescriptor filter_desc(3);
    filter_desc.set_spatial_dim(DimIndex::X, filter_size[2])
        .set_spatial_dim(DimIndex::Y, filter_size[1])
        .set_spatial_dim(DimIndex::Z, filter_size[0])
        .set_input_feature_map_count(in_depth)
        .set_output_feature_map_count(out_depth);
    perftools::gputools::dnn::ConvolutionDescriptor conv_desc(3);
    conv_desc.set_filter_stride(DimIndex::X, strides[2])
        .set_filter_stride(DimIndex::Y, strides[1])
        .set_filter_stride(DimIndex::Z, strides[0])
        .set_zero_padding(DimIndex::X, padding_cols / 2)
        .set_zero_padding(DimIndex::Y, padding_rows / 2)
        .set_zero_padding(DimIndex::Z, padding_planes / 2);

    // Shape: out, in, z, y, x.
    Tensor transformed_filter;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(DataTypeToEnum<T>::value,
                               TensorShape({out_depth, in_depth, filter_size[0],
                                            filter_size[1], filter_size[2]}),
                               &transformed_filter));
    functor::TransformFilter<GPUDevice, T, int, 5>()(
        context->eigen_device<GPUDevice>(), To32Bit(filter.tensor<T, 5>()),
        To32Bit(transformed_filter.tensor<T, 5>()));

    // Shape: batch, filters, z, y, x.
    Tensor transformed_out_backprop;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = {batch, out_depth, output_planes, output_rows,
                                output_cols};
      if (out_depth > 1) {
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::value, nchw_shape,
                                    &transformed_out_backprop));
        functor::NHWCToNCHW<GPUDevice, T, 5>()(
            context->eigen_device<GPUDevice>(), out_backprop.tensor<T, 5>(),
            transformed_out_backprop.tensor<T, 5>());
      } else {
        CHECK(transformed_out_backprop.CopyFrom(out_backprop, nchw_shape));
      }
    } else {
      transformed_out_backprop = out_backprop;
    }
    // Shape: batch, filters, z, y, x.
    Tensor pre_transformed_in_backprop;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(DataTypeToEnum<T>::value, compatible_input_shape,
                               &pre_transformed_in_backprop));

    auto out_backprop_ptr =
        AsDeviceMemory(transformed_out_backprop.template flat<T>().data(),
                       transformed_out_backprop.template flat<T>().size());
    auto filter_ptr =
        AsDeviceMemory(transformed_filter.template flat<T>().data(),
                       transformed_filter.template flat<T>().size());
    auto in_backprop_ptr =
        AsDeviceMemory(pre_transformed_in_backprop.template flat<T>().data(),
                       pre_transformed_in_backprop.template flat<T>().size());

    static int64 ConvolveBackwardDataScratchSize = GetCudnnWorkspaceLimit(
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default

    const int device_id = stream->parent()->device_ordinal();
    DataType dtype = context->input(0).dtype();
    const ConvParameters conv_parameters = {
        batch,
        in_depth,
        {{input_size[0], input_size[1], input_size[2]}},
        out_depth,
        {{filter_size[0], filter_size[1], filter_size[2]}},
        {{strides[0], strides[1], strides[2]}},
        {{padding_planes, padding_rows, padding_cols}},
        dtype,
        device_id,
    };

    using perftools::gputools::dnn::AlgorithmConfig;
    using perftools::gputools::dnn::AlgorithmDesc;
    using perftools::gputools::dnn::ProfileResult;
    AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune_ && !AutoTuneConv3dBwdData::GetInstance()->Find(
                                   conv_parameters, &algorithm_config)) {
      std::vector<AlgorithmDesc> algorithms;
      CHECK(stream->parent()->GetConvolveBackwardDataAlgorithms(
          conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      for (auto profile_algorithm : algorithms) {
        // TODO(zhengxq): profile each algorithm multiple times to better
        // accuracy.
        CudnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize,
                                                context);
        ProfileResult profile_result;
        bool cudnn_launch_status =
            stream
                ->ThenConvolveBackwardDataWithAlgorithm(
                    filter_desc, filter_ptr, output_desc, out_backprop_ptr,
                    conv_desc, input_desc, &in_backprop_ptr, &scratch_allocator,
                    AlgorithmConfig(profile_algorithm), &profile_result)
                .ok();
        if (cudnn_launch_status) {
          if (profile_result.is_valid()) {
            if (profile_result.elapsed_time_in_ms() <
                best_result.elapsed_time_in_ms()) {
              best_result = profile_result;
            }
            if (scratch_allocator.TotalByteSize() == 0 &&
                profile_result.elapsed_time_in_ms() <
                    best_result_no_scratch.elapsed_time_in_ms()) {
              best_result_no_scratch = profile_result;
            }
          }
        }
      }
      OP_REQUIRES(context,
                  best_result.is_valid() || best_result_no_scratch.is_valid(),
                  errors::NotFound("No algorithm worked!"));
      if (best_result.is_valid()) {
        algorithm_config.set_algorithm(best_result.algorithm());
      }
      if (best_result_no_scratch.is_valid()) {
        algorithm_config.set_algorithm_no_scratch(
            best_result_no_scratch.algorithm());
      }
      AutoTuneConv3dBwdData::GetInstance()->Insert(conv_parameters,
                                                   algorithm_config);
    }
    CudnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize,
                                            context);
    bool cudnn_launch_status =
        stream
            ->ThenConvolveBackwardDataWithAlgorithm(
                filter_desc, filter_ptr, output_desc, out_backprop_ptr,
                conv_desc, input_desc, &in_backprop_ptr, &scratch_allocator,
                algorithm_config, nullptr)
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(errors::Internal(
          "cuDNN Backward Data function launch failure : input shape(",
          input_shape.DebugString(), ") filter shape(",
          filter_shape.DebugString(), ")"));
    }

    if (rows_odd || cols_odd || planes_odd) {
      Tensor in_backprop_remove_padding;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            {batch, in_depth, input_size[0],
                                             input_size[1], input_size[2]},
                                            &in_backprop_remove_padding));

      // Remove the padding for odd spatial dimensions.
      functor::PadInput<GPUDevice, T, int, 5>()(
          context->eigen_device<GPUDevice>(),
          To32Bit(const_cast<const Tensor&>(pre_transformed_in_backprop)
                      .tensor<T, 5>()),
          {{0, 0, 0}}, {{-planes_odd, -rows_odd, -cols_odd}},
          To32Bit(in_backprop_remove_padding.tensor<T, 5>()), FORMAT_NCHW);

      pre_transformed_in_backprop = in_backprop_remove_padding;
    }

    if (data_format_ == FORMAT_NHWC) {
      auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
      functor::NCHWToNHWC<GPUDevice, T, 5>()(
          context->eigen_device<GPUDevice>(),
          toConstTensor(pre_transformed_in_backprop).template tensor<T, 5>(),
          in_backprop->tensor<T, 5>());
    } else {
      *in_backprop = pre_transformed_in_backprop;
    }
  }

 private:
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool takes_shape_;
  bool cudnn_use_autotune_;
};

// A dummy type to group backward filter autotune results together.
struct Conv3dBackwardFilterAutoTuneGroup {
  static string name() { return "Conv3dBwdFilter"; }
};
typedef AutoTuneSingleton<Conv3dBackwardFilterAutoTuneGroup, ConvParameters,
                          perftools::gputools::dnn::AlgorithmConfig>
    AutoTuneConv3dBwdFilter;

template <typename T>
class Conv3DBackpropFilterOp<GPUDevice, T> : public OpKernel {
 public:
  explicit Conv3DBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context),
        data_format_(FORMAT_NHWC),
        takes_shape_(type_string().find("V2") != std::string::npos) {
    // data_format is only available in V2.
    if (takes_shape_) {
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    }
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'C') == 1 &&
         GetTensorDim(stride_, data_format_, 'N') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    TensorShape filter_shape;
    if (takes_shape_) {
      const Tensor& filter_sizes = context->input(1);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                  filter_sizes.vec<int32>(), &filter_shape));
    } else {
      filter_shape = context->input(1).shape();
    }

    EXTRACT_AND_VERIFY_DIMENSIONS("Conv3DBackpropFilter");

    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    if (filter_size[1] == 1 && filter_size[2] == 1 && filter_size[0] == 1 &&
        strides[2] == 1 && strides[1] == 1 && strides[0] == 1 &&
        data_format_ == FORMAT_NHWC) {
      const uint64 m = in_depth;
      const uint64 k = batch * input_size[1] * input_size[2] * input_size[0];
      const uint64 n = out_depth;

      // The shape of output backprop is
      //   [batch, out_z, out_y, out_x, out_depth]
      // From cublas's perspective, it is: n x k
      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());

      // The shape of input is:
      //   [batch, in_z, in_y, in_x, in_depth],
      // From cublas's perspective, it is: m x k
      auto b_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                  input.template flat<T>().size());

      // The shape of the filter backprop is:
      //   [1, 1, 1, in_depth, out_depth]
      // From cublas's perspective, it is: n x m
      auto c_ptr = AsDeviceMemory(filter_backprop->template flat<T>().data(),
                                  filter_backprop->template flat<T>().size());

      bool blas_launch_status =
          stream
              ->ThenBlasGemm(perftools::gputools::blas::Transpose::kNoTranspose,
                             perftools::gputools::blas::Transpose::kTranspose,
                             n, m, k, 1.0f, a_ptr, n, b_ptr, m, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    } else if (filter_size[0] == input_size[0] &&
               filter_size[1] == input_size[1] &&
               filter_size[2] == input_size[2] && padding_ == Padding::VALID &&
               data_format_ == FORMAT_NHWC) {
      const uint64 m = input_size[0] * input_size[1] * input_size[2] * in_depth;
      const uint64 k = batch;
      const uint64 n = out_depth;

      auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                  input.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(filter_backprop->template flat<T>().data(),
                                  filter_backprop->template flat<T>().size());

      bool blas_launch_status =
          stream
              ->ThenBlasGemm(perftools::gputools::blas::Transpose::kNoTranspose,
                             perftools::gputools::blas::Transpose::kTranspose,
                             n, m, k, 1.0f, b_ptr, n, a_ptr, m, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    }

    int padding_rows = 0, padding_cols = 0, padding_planes = 0;

    if (padding_ == Padding::SAME) {
      padding_planes = std::max<int>(
          0, (output_planes - 1) * strides[0] + filter_size[0] - input_size[0]);
      padding_cols = std::max<int>(
          0, (output_cols - 1) * strides[2] + filter_size[2] - input_size[2]);
      padding_rows = std::max<int>(
          0, (output_rows - 1) * strides[1] + filter_size[1] - input_size[1]);
    }
    bool rows_odd = (padding_rows % 2 != 0);
    bool cols_odd = (padding_cols % 2 != 0);
    bool planes_odd = (padding_planes % 2 != 0);

    Tensor compatible_input;
    if (rows_odd || cols_odd || planes_odd) {
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(data_format_, batch,
                                                  {{input_size[0] + planes_odd,
                                                    input_size[1] + rows_odd,
                                                    input_size[2] + cols_odd}},
                                                  in_depth),
                                  &compatible_input));
      functor::PadInput<GPUDevice, T, int, 5>()(
          context->template eigen_device<GPUDevice>(),
          To32Bit(input.tensor<T, 5>()), {{0, 0, 0}},
          {{planes_odd, rows_odd, cols_odd}},
          To32Bit(compatible_input.tensor<T, 5>()), data_format_);
    } else {
      compatible_input = input;
    }

    CHECK(padding_rows >= 0 && padding_cols >= 0 && padding_planes >= 0)
        << "Negative paddings: (" << padding_rows << ", " << padding_cols
        << ", " << padding_planes << ")";
    perftools::gputools::dnn::BatchDescriptor input_desc(3);
    input_desc.set_count(batch)
        .set_spatial_dim(DimIndex::X,
                         GetTensorDim(compatible_input, data_format_, '2'))
        .set_spatial_dim(DimIndex::Y,
                         GetTensorDim(compatible_input, data_format_, '1'))
        .set_spatial_dim(DimIndex::Z,
                         GetTensorDim(compatible_input, data_format_, '0'))
        .set_feature_map_count(in_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::BatchDescriptor output_desc(3);
    output_desc.set_count(batch)
        .set_spatial_dim(DimIndex::X, output_cols)
        .set_spatial_dim(DimIndex::Y, output_rows)
        .set_spatial_dim(DimIndex::Z, output_planes)
        .set_feature_map_count(out_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::FilterDescriptor filter_desc(3);
    filter_desc.set_spatial_dim(DimIndex::X, filter_size[2])
        .set_spatial_dim(DimIndex::Y, filter_size[1])
        .set_spatial_dim(DimIndex::Z, filter_size[0])
        .set_input_feature_map_count(in_depth)
        .set_output_feature_map_count(out_depth);
    perftools::gputools::dnn::ConvolutionDescriptor conv_desc(3);
    conv_desc.set_filter_stride(DimIndex::X, strides[2])
        .set_filter_stride(DimIndex::Y, strides[1])
        .set_filter_stride(DimIndex::Z, strides[0])
        .set_zero_padding(DimIndex::X, padding_cols / 2)
        .set_zero_padding(DimIndex::Y, padding_rows / 2)
        .set_zero_padding(DimIndex::Z, padding_planes / 2);

    Tensor pre_transformed_filter_backprop;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(DataTypeToEnum<T>::value,
                               TensorShape({out_depth, in_depth, filter_size[0],
                                            filter_size[1], filter_size[2]}),
                               &pre_transformed_filter_backprop));

    Tensor transformed_out_backprop;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = {batch, out_depth, output_planes, output_rows,
                                output_cols};
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<T>::value, nchw_shape,
                                          &transformed_out_backprop));
      if (out_depth > 1) {
        functor::NHWCToNCHW<GPUDevice, T, 5>()(
            context->eigen_device<GPUDevice>(), out_backprop.tensor<T, 5>(),
            transformed_out_backprop.tensor<T, 5>());
      } else {
        CHECK(transformed_out_backprop.CopyFrom(out_backprop, nchw_shape));
      }
    } else {
      transformed_out_backprop = out_backprop;
    }
    Tensor transformed_input;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = {batch, in_depth, compatible_input.dim_size(1),
                                compatible_input.dim_size(2),
                                compatible_input.dim_size(3)};
      if (in_depth > 1) {
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::value,
                                              nchw_shape, &transformed_input));
        functor::NHWCToNCHW<GPUDevice, T, 5>()(
            context->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(compatible_input).tensor<T, 5>(),
            transformed_input.tensor<T, 5>());
      } else {
        CHECK(transformed_input.CopyFrom(compatible_input, nchw_shape));
      }
    } else {
      transformed_input = compatible_input;
    }

    auto out_backprop_ptr =
        AsDeviceMemory(transformed_out_backprop.template flat<T>().data(),
                       transformed_out_backprop.template flat<T>().size());
    auto filter_backprop_ptr = AsDeviceMemory(
        pre_transformed_filter_backprop.template flat<T>().data(),
        pre_transformed_filter_backprop.template flat<T>().size());
    auto input_ptr =
        AsDeviceMemory(transformed_input.template flat<T>().data(),
                       transformed_input.template flat<T>().size());

    static int64 ConvolveBackwardFilterScratchSize = GetCudnnWorkspaceLimit(
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default

    const int device_id = stream->parent()->device_ordinal();
    DataType dtype = input.dtype();
    const ConvParameters conv_parameters = {
        batch,
        in_depth,
        {{input_size[0], input_size[1], input_size[2]}},
        out_depth,
        {{filter_size[0], filter_size[1], filter_size[2]}},
        {{strides[0], strides[1], strides[2]}},
        {{padding_planes, padding_rows, padding_cols}},
        dtype,
        device_id,
    };

    using perftools::gputools::dnn::AlgorithmConfig;
    using perftools::gputools::dnn::AlgorithmDesc;
    using perftools::gputools::dnn::ProfileResult;
    AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune_ && !AutoTuneConv3dBwdFilter::GetInstance()->Find(
                                   conv_parameters, &algorithm_config)) {
      std::vector<AlgorithmDesc> algorithms;
      CHECK(stream->parent()->GetConvolveBackwardFilterAlgorithms(
          conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      for (auto profile_algorithm : algorithms) {
        // TODO(zhengxq): profile each algorithm multiple times to better
        // accuracy.
        CudnnScratchAllocator scratch_allocator(
            ConvolveBackwardFilterScratchSize, context);
        ProfileResult profile_result;
        bool cudnn_launch_status =
            stream
                ->ThenConvolveBackwardFilterWithAlgorithm(
                    input_desc, input_ptr, output_desc, out_backprop_ptr,
                    conv_desc, filter_desc, &filter_backprop_ptr,
                    &scratch_allocator, AlgorithmConfig(profile_algorithm),
                    &profile_result)
                .ok();
        if (cudnn_launch_status) {
          if (profile_result.is_valid()) {
            if (profile_result.elapsed_time_in_ms() <
                best_result.elapsed_time_in_ms()) {
              best_result = profile_result;
            }
            if (scratch_allocator.TotalByteSize() == 0 &&
                profile_result.elapsed_time_in_ms() <
                    best_result_no_scratch.elapsed_time_in_ms()) {
              best_result_no_scratch = profile_result;
            }
          }
        }
      }
      OP_REQUIRES(context,
                  best_result.is_valid() || best_result_no_scratch.is_valid(),
                  errors::NotFound("No algorithm worked!"));
      if (best_result.is_valid()) {
        algorithm_config.set_algorithm(best_result.algorithm());
      }
      if (best_result_no_scratch.is_valid()) {
        algorithm_config.set_algorithm_no_scratch(
            best_result_no_scratch.algorithm());
      }
      AutoTuneConv3dBwdFilter::GetInstance()->Insert(conv_parameters,
                                                     algorithm_config);
    }
    CudnnScratchAllocator scratch_allocator(ConvolveBackwardFilterScratchSize,
                                            context);
    bool cudnn_launch_status =
        stream
            ->ThenConvolveBackwardFilterWithAlgorithm(
                input_desc, input_ptr, output_desc, out_backprop_ptr, conv_desc,
                filter_desc, &filter_backprop_ptr, &scratch_allocator,
                algorithm_config, nullptr)
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(errors::Internal(
          "cuDNN Backward Filter function launch failure : input shape(",
          input_shape.DebugString(), ") filter shape(",
          filter_shape.DebugString(), ")"));
    }

    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::ReverseTransformFilter<GPUDevice, T, 5>()(
        context->eigen_device<GPUDevice>(),
        toConstTensor(pre_transformed_filter_backprop).template tensor<T, 5>(),
        filter_backprop->tensor<T, 5>());
  }

 private:
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool takes_shape_;
  bool cudnn_use_autotune_;
};

#define REGISTER_GPU_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv3DBackpropInput").Device(DEVICE_GPU).TypeConstraint<T>("T"),  \
      Conv3DBackpropInputOp<GPUDevice, T>);                                   \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInputV2")                       \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("input_sizes"),                     \
                          Conv3DBackpropInputOp<GPUDevice, T>);               \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv3DBackpropFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      Conv3DBackpropFilterOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilterV2")                      \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("filter_sizes"),                    \
                          Conv3DBackpropFilterOp<GPUDevice, T>);
TF_CALL_half(REGISTER_GPU_KERNEL);
TF_CALL_float(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
