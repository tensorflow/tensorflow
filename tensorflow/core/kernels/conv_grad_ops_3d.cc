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
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
using stream_executor::dnn::DimIndex;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

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

    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'C') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'N') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));

    // TODO(yangzihao): Add CPU version of dilated conv 3D.
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, '0') == 1 &&
                 GetTensorDim(dilation_, data_format_, '1') == 1 &&
                 GetTensorDim(dilation_, data_format_, '2') == 1),
                errors::InvalidArgument(
                    "Current CPU implementation does not yet support "
                    "dilation rates larger than 1."));

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

    const Tensor& out_backprop = context->input(2);
    const TensorShape& out_backprop_shape = out_backprop.shape();

    TensorShape input_shape;
    if (takes_shape_) {
      const Tensor& input_sizes = context->input(0);
      // MakeShape is able to handle both DT_INT32 and DT_INT64 for input_sizes.
      OP_REQUIRES_OK(context, MakeShape(input_sizes, &input_shape));
    } else {
      input_shape = context->input(0).shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context, ConvBackpropComputeDimensions(
                                "Conv3DBackpropInputOp", /*num_spatial_dims=*/3,
                                input_shape, filter_shape, out_backprop_shape,
                                stride_, padding_, data_format_, &dims));

    Tensor* in_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    functor::CuboidConvolutionBackwardInput<Device, T>()(
        context->eigen_device<Device>(),
        in_backprop->tensor<T, 5>(),                     // input_backward
        filter.tensor<T, 5>(),                           // filter
        out_backprop.tensor<T, 5>(),                     // output_backward
        static_cast<int>(dims.spatial_dims[0].stride),   // stride_planes
        static_cast<int>(dims.spatial_dims[1].stride),   // stride_rows
        static_cast<int>(dims.spatial_dims[2].stride));  // stride_cols
  }

 private:
  std::vector<int32> dilation_;
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

    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'C') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'N') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));

    // TODO(yangzihao): Add CPU version of dilated conv 3D.
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, '0') == 1 &&
                 GetTensorDim(dilation_, data_format_, '1') == 1 &&
                 GetTensorDim(dilation_, data_format_, '2') == 1),
                errors::InvalidArgument(
                    "Current CPU implementation does not yet support "
                    "dilation rates larger than 1."));

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

    const Tensor& out_backprop = context->input(2);
    const TensorShape& out_backprop_shape = out_backprop.shape();

    TensorShape filter_shape;
    if (takes_shape_) {
      const Tensor& filter_sizes = context->input(1);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                  filter_sizes.vec<int32>(), &filter_shape));
    } else {
      filter_shape = context->input(1).shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensions(
                       "Conv3DBackpropFilterOp", /*num_spatial_dims=*/3,
                       input_shape, filter_shape, out_backprop_shape, stride_,
                       padding_, data_format_, &dims));

    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    if (input_shape.num_elements() == 0) {
      filter_backprop->template flat<T>().setZero();
      return;
    }

    functor::CuboidConvolutionBackwardFilter<Device, T>()(
        context->eigen_device<Device>(),
        filter_backprop->tensor<T, 5>(),                 // filter_backward
        input.tensor<T, 5>(),                            // input
        out_backprop.tensor<T, 5>(),                     // output_backward
        static_cast<int>(dims.spatial_dims[0].stride),   // stride_planes
        static_cast<int>(dims.spatial_dims[1].stride),   // stride_rows
        static_cast<int>(dims.spatial_dims[2].stride));  // stride_cols
  }

 private:
  std::vector<int32> dilation_;
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
                          se::dnn::AlgorithmConfig>

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
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'C') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'N') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));
    OP_REQUIRES(
        context,
        (GetTensorDim(dilation_, data_format_, '0') > 0 &&
         GetTensorDim(dilation_, data_format_, '1') > 0 &&
         GetTensorDim(dilation_, data_format_, '2') > 0),
        errors::InvalidArgument("Dilated rates should be larger than 0."));
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
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, '0') > 0 &&
         GetTensorDim(stride_, data_format_, '1') > 0 &&
         GetTensorDim(stride_, data_format_, '2') > 0),
        errors::InvalidArgument("Spatial strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }
  void Compute(OpKernelContext* context) override {
    const Tensor& filter = context->input(1);
    const TensorShape& filter_shape = filter.shape();

    const Tensor& out_backprop = context->input(2);
    const TensorShape& out_backprop_shape = out_backprop.shape();

    TensorShape input_shape;
    if (takes_shape_) {
      const Tensor& input_sizes = context->input(0);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                  input_sizes.vec<int32>(), &input_shape));
    } else {
      input_shape = context->input(0).shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensionsV2(
                       "Conv3DBackpropInputOp", /*num_spatial_dims=*/3,
                       input_shape, filter_shape, out_backprop_shape, dilation_,
                       stride_, padding_, data_format_, &dims));

    Tensor* in_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    if (dims.filter_size(0) == 1 && dims.filter_size(1) == 1 &&
        dims.filter_size(2) == 1 && dims.dilation(0) == 1 &&
        dims.dilation(1) == 1 && dims.dilation(2) == 1 && dims.stride(0) == 1 &&
        dims.stride(1) == 1 && dims.stride(2) == 1 &&
        data_format_ == FORMAT_NHWC) {
      const uint64 m = dims.batch_size * dims.input_size(0) *
                       dims.input_size(1) * dims.input_size(2);
      const uint64 k = dims.out_depth;
      const uint64 n = dims.in_depth;

      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                  in_backprop->template flat<T>().size());

      auto transpose = se::blas::Transpose::kTranspose;
      auto no_transpose = se::blas::Transpose::kNoTranspose;

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
    } else if (dims.filter_size(0) == dims.input_size(0) &&
               dims.filter_size(1) == dims.input_size(1) &&
               dims.filter_size(2) == dims.input_size(2) &&
               padding_ == Padding::VALID && data_format_ == FORMAT_NHWC) {
      const uint64 m = dims.batch_size;
      const uint64 k = dims.out_depth;
      const uint64 n = dims.input_size(0) * dims.input_size(1) *
                       dims.input_size(2) * dims.in_depth;

      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                  in_backprop->template flat<T>().size());

      auto transpose = se::blas::Transpose::kTranspose;
      auto no_transpose = se::blas::Transpose::kNoTranspose;

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

    int padding_planes = dims.SpatialPadding(padding_, 0);
    int padding_rows = dims.SpatialPadding(padding_, 1);
    int padding_cols = dims.SpatialPadding(padding_, 2);
    const bool planes_odd = (padding_planes % 2 != 0);
    const bool rows_odd = (padding_rows % 2 != 0);
    const bool cols_odd = (padding_cols % 2 != 0);

    TensorShape compatible_input_shape;
    if (rows_odd || cols_odd || planes_odd) {
      // cuDNN only supports the same amount of padding on both sides.
      compatible_input_shape = {
          dims.batch_size,
          dims.in_depth,
          dims.input_size(0) + planes_odd,
          dims.input_size(1) + rows_odd,
          dims.input_size(2) + cols_odd,
      };
    } else {
      compatible_input_shape = {dims.batch_size, dims.in_depth,
                                dims.input_size(0), dims.input_size(1),
                                dims.input_size(2)};
    }

    CHECK(padding_rows >= 0 && padding_cols >= 0 && padding_planes >= 0)
        << "Negative paddings: (" << padding_rows << ", " << padding_cols
        << ", " << padding_planes << ")";
    se::dnn::BatchDescriptor input_desc(3);
    input_desc.set_count(dims.batch_size)
        .set_spatial_dim(DimIndex::X, compatible_input_shape.dim_size(4))
        .set_spatial_dim(DimIndex::Y, compatible_input_shape.dim_size(3))
        .set_spatial_dim(DimIndex::Z, compatible_input_shape.dim_size(2))
        .set_feature_map_count(dims.in_depth)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::BatchDescriptor output_desc(3);
    output_desc.set_count(dims.batch_size)
        .set_spatial_dim(DimIndex::X, dims.output_size(2))
        .set_spatial_dim(DimIndex::Y, dims.output_size(1))
        .set_spatial_dim(DimIndex::Z, dims.output_size(0))
        .set_feature_map_count(dims.out_depth)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::FilterDescriptor filter_desc(3);
    filter_desc.set_spatial_dim(DimIndex::X, dims.filter_size(2))
        .set_spatial_dim(DimIndex::Y, dims.filter_size(1))
        .set_spatial_dim(DimIndex::Z, dims.filter_size(0))
        .set_input_feature_map_count(dims.in_depth)
        .set_output_feature_map_count(dims.out_depth);
    se::dnn::ConvolutionDescriptor conv_desc(3);
    conv_desc.set_dilation_rate(DimIndex::X, dims.dilation(2))
        .set_dilation_rate(DimIndex::Y, dims.dilation(1))
        .set_dilation_rate(DimIndex::Z, dims.dilation(0))
        .set_filter_stride(DimIndex::X, dims.stride(2))
        .set_filter_stride(DimIndex::Y, dims.stride(1))
        .set_filter_stride(DimIndex::Z, dims.stride(0))
        .set_zero_padding(DimIndex::X, padding_cols / 2)
        .set_zero_padding(DimIndex::Y, padding_rows / 2)
        .set_zero_padding(DimIndex::Z, padding_planes / 2);

    // Shape: out, in, z, y, x.
    Tensor transformed_filter;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(
            DataTypeToEnum<T>::value,
            TensorShape({dims.out_depth, dims.in_depth, dims.filter_size(0),
                         dims.filter_size(1), dims.filter_size(2)}),
            &transformed_filter));
    functor::TransformFilter<GPUDevice, T, int, 5>()(
        context->eigen_device<GPUDevice>(), To32Bit(filter.tensor<T, 5>()),
        To32Bit(transformed_filter.tensor<T, 5>()));

    // Shape: batch, filters, z, y, x.
    Tensor transformed_out_backprop;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = {dims.batch_size, dims.out_depth,
                                dims.output_size(0), dims.output_size(1),
                                dims.output_size(2)};
      if (dims.out_depth > 1) {
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
        dims.batch_size,
        dims.in_depth,
        {{dims.input_size(0), dims.input_size(1), dims.input_size(2)}},
        FORMAT_NCHW,
        dims.out_depth,
        {{dims.filter_size(0), dims.filter_size(1), dims.filter_size(2)}},
        {{dims.dilation(0), dims.dilation(1), dims.dilation(2)}},
        {{dims.stride(0), dims.stride(1), dims.stride(2)}},
        {{padding_planes, padding_rows, padding_cols}},
        dtype,
        device_id,
    };

    using se::dnn::AlgorithmConfig;
    using se::dnn::AlgorithmDesc;
    using se::dnn::ProfileResult;
    AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune_ && !AutoTuneConv3dBwdData::GetInstance()->Find(
                                   conv_parameters, &algorithm_config)) {
      std::vector<AlgorithmDesc> algorithms;
      CHECK(stream->parent()->GetConvolveBackwardDataAlgorithms(
          conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(
              stream->parent()),
          &algorithms));
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
                     context->allocate_temp(
                         DataTypeToEnum<T>::value,
                         {dims.batch_size, dims.in_depth, dims.input_size(0),
                          dims.input_size(1), dims.input_size(2)},
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
  std::vector<int32> dilation_;
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
                          se::dnn::AlgorithmConfig>
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
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'C') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'N') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));
    OP_REQUIRES(
        context,
        (GetTensorDim(dilation_, data_format_, '0') > 0 &&
         GetTensorDim(dilation_, data_format_, '1') > 0 &&
         GetTensorDim(dilation_, data_format_, '2') > 0),
        errors::InvalidArgument("Dilated rates should be larger than 0."));
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
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, '0') > 0 &&
         GetTensorDim(stride_, data_format_, '1') > 0 &&
         GetTensorDim(stride_, data_format_, '2') > 0),
        errors::InvalidArgument("Spatial strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();

    const Tensor& out_backprop = context->input(2);
    const TensorShape& out_backprop_shape = out_backprop.shape();

    TensorShape filter_shape;
    if (takes_shape_) {
      const Tensor& filter_sizes = context->input(1);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                  filter_sizes.vec<int32>(), &filter_shape));
    } else {
      filter_shape = context->input(1).shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensionsV2(
                       "Conv3DBackpropFilterOp", /*num_spatial_dims=*/3,
                       input_shape, filter_shape, out_backprop_shape, dilation_,
                       stride_, padding_, data_format_, &dims));

    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    if (dims.filter_size(1) == 1 && dims.filter_size(2) == 1 &&
        dims.filter_size(0) == 1 && dims.dilation(2) == 1 &&
        dims.dilation(1) == 1 && dims.dilation(0) == 1 && dims.stride(2) == 1 &&
        dims.stride(1) == 1 && dims.stride(0) == 1 &&
        data_format_ == FORMAT_NHWC) {
      const uint64 m = dims.in_depth;
      const uint64 k = dims.batch_size * dims.input_size(1) *
                       dims.input_size(2) * dims.input_size(0);
      const uint64 n = dims.out_depth;

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
              ->ThenBlasGemm(se::blas::Transpose::kNoTranspose,
                             se::blas::Transpose::kTranspose, n, m, k, 1.0f,
                             a_ptr, n, b_ptr, m, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    } else if (dims.filter_size(0) == dims.input_size(0) &&
               dims.filter_size(1) == dims.input_size(1) &&
               dims.filter_size(2) == dims.input_size(2) &&
               padding_ == Padding::VALID && data_format_ == FORMAT_NHWC) {
      const uint64 m = dims.input_size(0) * dims.input_size(1) *
                       dims.input_size(2) * dims.in_depth;
      const uint64 k = dims.batch_size;
      const uint64 n = dims.out_depth;

      auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                  input.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(filter_backprop->template flat<T>().data(),
                                  filter_backprop->template flat<T>().size());

      bool blas_launch_status =
          stream
              ->ThenBlasGemm(se::blas::Transpose::kNoTranspose,
                             se::blas::Transpose::kTranspose, n, m, k, 1.0f,
                             b_ptr, n, a_ptr, m, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    }

    int padding_planes = dims.SpatialPadding(padding_, 0);
    int padding_rows = dims.SpatialPadding(padding_, 1);
    int padding_cols = dims.SpatialPadding(padding_, 2);
    const bool planes_odd = (padding_planes % 2 != 0);
    const bool rows_odd = (padding_rows % 2 != 0);
    const bool cols_odd = (padding_cols % 2 != 0);

    Tensor compatible_input;
    if (rows_odd || cols_odd || planes_odd) {
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataTypeToEnum<T>::value,
                         ShapeFromFormat(data_format_, dims.batch_size,
                                         {{dims.input_size(0) + planes_odd,
                                           dims.input_size(1) + rows_odd,
                                           dims.input_size(2) + cols_odd}},
                                         dims.in_depth),
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
    se::dnn::BatchDescriptor input_desc(3);
    input_desc.set_count(dims.batch_size)
        .set_spatial_dim(DimIndex::X,
                         GetTensorDim(compatible_input, data_format_, '2'))
        .set_spatial_dim(DimIndex::Y,
                         GetTensorDim(compatible_input, data_format_, '1'))
        .set_spatial_dim(DimIndex::Z,
                         GetTensorDim(compatible_input, data_format_, '0'))
        .set_feature_map_count(dims.in_depth)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::BatchDescriptor output_desc(3);
    output_desc.set_count(dims.batch_size)
        .set_spatial_dim(DimIndex::X, dims.output_size(2))
        .set_spatial_dim(DimIndex::Y, dims.output_size(1))
        .set_spatial_dim(DimIndex::Z, dims.output_size(0))
        .set_feature_map_count(dims.out_depth)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::FilterDescriptor filter_desc(3);
    filter_desc.set_spatial_dim(DimIndex::X, dims.filter_size(2))
        .set_spatial_dim(DimIndex::Y, dims.filter_size(1))
        .set_spatial_dim(DimIndex::Z, dims.filter_size(0))
        .set_input_feature_map_count(dims.in_depth)
        .set_output_feature_map_count(dims.out_depth);
    se::dnn::ConvolutionDescriptor conv_desc(3);
    conv_desc.set_dilation_rate(DimIndex::X, dims.dilation(2))
        .set_dilation_rate(DimIndex::Y, dims.dilation(1))
        .set_dilation_rate(DimIndex::Z, dims.dilation(0))
        .set_filter_stride(DimIndex::X, dims.stride(2))
        .set_filter_stride(DimIndex::Y, dims.stride(1))
        .set_filter_stride(DimIndex::Z, dims.stride(0))
        .set_zero_padding(DimIndex::X, padding_cols / 2)
        .set_zero_padding(DimIndex::Y, padding_rows / 2)
        .set_zero_padding(DimIndex::Z, padding_planes / 2);

    Tensor pre_transformed_filter_backprop;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(
            DataTypeToEnum<T>::value,
            TensorShape({dims.out_depth, dims.in_depth, dims.filter_size(0),
                         dims.filter_size(1), dims.filter_size(2)}),
            &pre_transformed_filter_backprop));

    Tensor transformed_out_backprop;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = {dims.batch_size, dims.out_depth,
                                dims.output_size(0), dims.output_size(1),
                                dims.output_size(2)};
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<T>::value, nchw_shape,
                                          &transformed_out_backprop));
      if (dims.out_depth > 1) {
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
      TensorShape nchw_shape = {
          dims.batch_size, dims.in_depth, compatible_input.dim_size(1),
          compatible_input.dim_size(2), compatible_input.dim_size(3)};
      if (dims.in_depth > 1) {
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
        dims.batch_size,
        dims.in_depth,
        {{dims.input_size(0), dims.input_size(1), dims.input_size(2)}},
        FORMAT_NCHW,
        dims.out_depth,
        {{dims.filter_size(0), dims.filter_size(1), dims.filter_size(2)}},
        {{dims.dilation(0), dims.dilation(1), dims.dilation(2)}},
        {{dims.stride(0), dims.stride(1), dims.stride(2)}},
        {{padding_planes, padding_rows, padding_cols}},
        dtype,
        device_id,
    };

    using se::dnn::AlgorithmConfig;
    using se::dnn::AlgorithmDesc;
    using se::dnn::ProfileResult;
    AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune_ && !AutoTuneConv3dBwdFilter::GetInstance()->Find(
                                   conv_parameters, &algorithm_config)) {
      std::vector<AlgorithmDesc> algorithms;
      CHECK(stream->parent()->GetConvolveBackwardFilterAlgorithms(
          conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(
              stream->parent()),
          &algorithms));
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
  std::vector<int32> dilation_;
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
