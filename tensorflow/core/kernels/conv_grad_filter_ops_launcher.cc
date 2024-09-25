/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "xla/tsl/framework/contraction/eigen_contraction_kernel.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/numeric_options_utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if GOOGLE_CUDA
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct LaunchConv2DBackpropFilterOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& out_backprop, const Tensor& input,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  const std::vector<int64_t>& explicit_paddings,
                  Tensor* filter_backprop, TensorFormat data_format) {
    std::vector<int32> dilations(4, 1);
    dilations[GetTensorDimIndex(data_format, 'H')] = row_dilation;
    dilations[GetTensorDimIndex(data_format, 'W')] = col_dilation;

    std::vector<int32> strides(4, 1);
    strides[GetTensorDimIndex(data_format, 'H')] = row_stride;
    strides[GetTensorDimIndex(data_format, 'W')] = col_stride;
    TensorShape filter_shape = filter_backprop->shape();

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        ctx, ConvBackpropComputeDimensionsV2(
                 "Conv2DBackpropFilter", /*num_spatial_dims=*/2, input.shape(),
                 filter_shape, out_backprop.shape(), dilations, strides,
                 padding, explicit_paddings, data_format, &dims));

    int64_t padding_top = -1, padding_bottom = -1;
    int64_t padding_left = -1, padding_right = -1;
    if (padding == EXPLICIT) {
      GetExplicitPaddingForDim(explicit_paddings, data_format, 'H',
                               &padding_top, &padding_bottom);
      GetExplicitPaddingForDim(explicit_paddings, data_format, 'W',
                               &padding_left, &padding_right);
    }
    int64_t expected_out_rows, expected_out_cols;
    // The function is guaranteed to succeed because we checked the output and
    // padding was valid earlier.
    TF_CHECK_OK(GetWindowedOutputSizeVerbose(
        dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
        row_dilation, row_stride, padding, &expected_out_rows, &padding_top,
        &padding_bottom));
    DCHECK_EQ(dims.spatial_dims[0].output_size, expected_out_rows);
    TF_CHECK_OK(GetWindowedOutputSizeVerbose(
        dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
        col_dilation, col_stride, padding, &expected_out_cols, &padding_left,
        &padding_right));
    DCHECK_EQ(dims.spatial_dims[1].output_size, expected_out_cols);

    const CPUDevice& d = ctx->eigen_device<CPUDevice>();

    // WARNING: Need to swap row/col, padding_top/padding_left, and
    // padding_bottom/padding_right when calling Eigen. Eigen expects tensors
    // in NWHC format, but Tensorflow uses NHWC.

    auto filter_backprop_t = filter_backprop->tensor<T, 4>();
    auto input_t = input.tensor<T, 4>();
    auto out_backprop_t = out_backprop.tensor<T, 4>();

    if (padding != EXPLICIT) {
      // If padding was not explicitly defined, Eigen spatial convolution
      // backward filter will infer correct forward paddings from input tensors.
      filter_backprop_t.device(d) = Eigen::SpatialConvolutionBackwardKernel(
          input_t, out_backprop_t, filter_backprop_t.dimension(1),
          filter_backprop_t.dimension(0), col_stride, row_stride, col_dilation,
          row_dilation);

    } else {
      // Otherwise we have to explicitly pad the input, before passing it to
      // spatial convolution backward filter.
      Eigen::array<std::pair<int, int>, 4> paddings;
      paddings[0] = {0, 0};
      paddings[1] = {padding_top, padding_bottom};
      paddings[2] = {padding_left, padding_right};
      paddings[3] = {0, 0};

      auto padded_t = input_t.pad(paddings, T(0));

      // TODO(ezhulenev): Pass explicit paddings to Eigen spatial backward
      // convolution and do not rely on tensor padding expression.
      filter_backprop_t.device(d) = Eigen::SpatialConvolutionBackwardKernel(
          padded_t, out_backprop_t, filter_backprop_t.dimension(1),
          filter_backprop_t.dimension(0), col_stride, row_stride, col_dilation,
          row_dilation);
    }
  }
};

template struct LaunchConv2DBackpropFilterOp<CPUDevice, Eigen::bfloat16>;
template struct LaunchConv2DBackpropFilterOp<CPUDevice, Eigen::half>;
template struct LaunchConv2DBackpropFilterOp<CPUDevice, float>;
template struct LaunchConv2DBackpropFilterOp<CPUDevice, double>;

// GPU definitions.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// The slow version (but compiles for GPU)

// A dummy type to group forward backward filter autotune results together.
struct ConvBackwardFilterAutotuneGroup {
  static string name() { return "ConvBwdFilter"; }
};

typedef AutotuneSingleton<ConvBackwardFilterAutotuneGroup, ConvParameters,
                          AutotuneEntry<se::dnn::ConvOp>>
    AutotuneConvBwdFilter;

template <typename T>
void LaunchConv2DBackpropFilterOpImpl(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& out_backprop, const Tensor& input, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64_t>& explicit_paddings, Tensor* filter_backprop,
    TensorFormat data_format) {
  using se::dnn::AlgorithmConfig;
  using se::dnn::AlgorithmDesc;
  using se::dnn::ProfileResult;

  std::vector<int32> dilations(4, 1);
  dilations[GetTensorDimIndex(data_format, 'H')] = row_dilation;
  dilations[GetTensorDimIndex(data_format, 'W')] = col_dilation;

  std::vector<int32> strides(4, 1);
  strides[GetTensorDimIndex(data_format, 'H')] = row_stride;
  strides[GetTensorDimIndex(data_format, 'W')] = col_stride;
  TensorShape filter_shape = filter_backprop->shape();

  ConvBackpropDimensions dims;
  OP_REQUIRES_OK(
      ctx, ConvBackpropComputeDimensionsV2(
               "Conv2DBackpropFilter", /*num_spatial_dims=*/2, input.shape(),
               filter_shape, out_backprop.shape(), dilations, strides, padding,
               explicit_paddings, data_format, &dims));

  int64_t padding_top = -1, padding_bottom = -1;
  int64_t padding_left = -1, padding_right = -1;
  if (padding == EXPLICIT) {
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'H', &padding_top,
                             &padding_bottom);
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'W', &padding_left,
                             &padding_right);
  }
  int64_t expected_out_rows, expected_out_cols;
  // The function is guaranteed to succeed because we checked the output and
  // padding was valid earlier.
  TF_CHECK_OK(GetWindowedOutputSizeVerbose(
      dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
      row_dilation, row_stride, padding, &expected_out_rows, &padding_top,
      &padding_bottom));
  DCHECK_EQ(dims.spatial_dims[0].output_size, expected_out_rows);
  TF_CHECK_OK(GetWindowedOutputSizeVerbose(
      dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
      col_dilation, col_stride, padding, &expected_out_cols, &padding_left,
      &padding_right));
  DCHECK_EQ(dims.spatial_dims[1].output_size, expected_out_cols);

  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  if (!use_cudnn) {
    ctx->SetStatus(errors::Unimplemented(
        "Conv2DBackprop for GPU is not currently supported "
        "without cudnn"));
    return;
  }
  auto* blas = stream->parent()->AsBlas();
  OP_REQUIRES(ctx, blas != nullptr, absl::InternalError("No BLAS for stream."));

  // If the filter in-depth (filter_shape.dim_size(2)) is 1 and smaller than the
  // input depth, it's a depthwise convolution. More generally, if the filter
  // in-depth divides but is smaller than the input depth, it is a grouped
  // convolution.
  bool is_grouped_convolution = filter_shape.dim_size(2) != dims.in_depth;
  bool cudnn_disable_conv_1x1_optimization_ = CudnnDisableConv1x1Optimization();
  if (!cudnn_disable_conv_1x1_optimization_ &&
      dims.spatial_dims[0].filter_size == 1 &&
      dims.spatial_dims[1].filter_size == 1 && !is_grouped_convolution &&
      dims.spatial_dims[0].stride == 1 && dims.spatial_dims[1].stride == 1 &&
      data_format == FORMAT_NHWC && (padding == VALID || padding == SAME)) {
    const uint64 m = dims.in_depth;
    const uint64 k = dims.batch_size * dims.spatial_dims[0].input_size *
                     dims.spatial_dims[1].input_size;
    const uint64 n = dims.out_depth;

    // The shape of output backprop is
    //   [batch, out_rows, out_cols, out_depth]
    //   From cublas's perspective, it is: n x k
    auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                out_backprop.template flat<T>().size());

    // The shape of input is
    //   [batch, in_rows, in_cols, in_depth],
    //   From cublas's perspective, it is: m x k
    auto b_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                input.template flat<T>().size());

    // the shape of the filter backprop from the conv_2d should be
    //   [1, 1, in_depth, out_depth]
    //   From cublas's perspective, it is: n x m
    auto c_ptr = AsDeviceMemory(filter_backprop->template flat<T>().data(),
                                filter_backprop->template flat<T>().size());

    OP_REQUIRES_OK(
        ctx, blas->BlasGemm(stream, se::blas::Transpose::kNoTranspose,
                            se::blas::Transpose::kTranspose, n, m, k, a_ptr, n,
                            b_ptr, m, &c_ptr, n, GetNumericOptions(),
                            se::blas::CallContext::kNone));
    return;
  } else if (dims.spatial_dims[0].filter_size ==
                 dims.spatial_dims[0].input_size &&
             dims.spatial_dims[1].filter_size ==
                 dims.spatial_dims[1].input_size &&
             !is_grouped_convolution && padding == VALID &&
             data_format == FORMAT_NHWC) {
    // The input data and filter have the same height/width, and we are not
    // using grouped convolution, so call cublas directly.
    const uint64 m = dims.spatial_dims[0].input_size *
                     dims.spatial_dims[1].input_size * dims.in_depth;
    const uint64 k = dims.batch_size;
    const uint64 n = dims.out_depth;

    auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                input.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                out_backprop.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(filter_backprop->template flat<T>().data(),
                                filter_backprop->template flat<T>().size());

    OP_REQUIRES_OK(
        ctx, blas->BlasGemm(stream, se::blas::Transpose::kNoTranspose,
                            se::blas::Transpose::kTranspose, n, m, k, b_ptr, n,
                            a_ptr, m, &c_ptr, n, GetNumericOptions(),
                            se::blas::CallContext::kNone));
    return;
  }

  const int64_t common_padding_rows = std::min(padding_top, padding_bottom);
  const int64_t common_padding_cols = std::min(padding_left, padding_right);
  Tensor compatible_input;
  if (padding_top != padding_bottom || padding_left != padding_right) {
    // Pad the input in the same way we did during the forward pass, so that
    // cuDNN or MIOpen receives the same input during the backward pass function
    // as it did during the forward pass function.
    const int64_t padding_rows_diff = std::abs(padding_bottom - padding_top);
    const int64_t padding_cols_diff = std::abs(padding_right - padding_left);
    const int64_t new_in_rows =
        dims.spatial_dims[0].input_size + padding_rows_diff;
    const int64_t new_in_cols =
        dims.spatial_dims[1].input_size + padding_cols_diff;
    const int64_t input_pad_top = padding_top - common_padding_rows;
    const int64_t input_pad_bottom = padding_bottom - common_padding_rows;
    const int64_t input_pad_left = padding_left - common_padding_cols;
    const int64_t input_pad_right = padding_right - common_padding_cols;
    TensorShape compatible_input_shape;
    OP_REQUIRES_OK(
        ctx, ShapeFromFormatWithStatus(data_format, dims.batch_size,
                                       new_in_rows, new_in_cols, dims.in_depth,
                                       &compatible_input_shape));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                compatible_input_shape, &compatible_input));

    functor::PadInput<GPUDevice, T, int, 4>()(
        ctx->template eigen_device<GPUDevice>(), To32Bit(input.tensor<T, 4>()),
        {{static_cast<int>(input_pad_top), static_cast<int>(input_pad_left)}},
        {{static_cast<int>(input_pad_bottom),
          static_cast<int>(input_pad_right)}},
        To32Bit(compatible_input.tensor<T, 4>()), data_format, T{});
  } else {
    compatible_input = input;
  }

  CHECK(common_padding_rows >= 0 && common_padding_cols >= 0)  // Crash OK
      << "Negative row or col paddings: (" << common_padding_rows << ", "
      << common_padding_cols << ")";

  const bool compute_in_nhwc =
      ComputeInNhwcEnabled(DataTypeToEnum<T>::value, stream);

  // We only do one directional conversion: NHWC->NCHW. We never convert in the
  // other direction. Grappler layout optimizer selects the preferred layout and
  // adds necessary annotations to the graph.
  const TensorFormat compute_data_format =
      (compute_in_nhwc && data_format == FORMAT_NHWC) ? FORMAT_NHWC
                                                      : FORMAT_NCHW;

  VLOG(3) << "Compute Conv2DBackpropFilter with cuDNN:"
          << " data_format=" << ToString(data_format)
          << " compute_data_format=" << ToString(compute_data_format);

  constexpr auto kComputeInNHWC =
      std::make_tuple(se::dnn::DataLayout::kBatchYXDepth,
                      se::dnn::FilterLayout::kOutputYXInput);
  constexpr auto kComputeInNCHW =
      std::make_tuple(se::dnn::DataLayout::kBatchDepthYX,
                      se::dnn::FilterLayout::kOutputInputYX);

  se::dnn::DataLayout compute_data_layout;
  se::dnn::FilterLayout filter_layout;

  std::tie(compute_data_layout, filter_layout) =
      compute_data_format == FORMAT_NHWC ? kComputeInNHWC : kComputeInNCHW;

  se::dnn::BatchDescriptor input_desc;
  input_desc.set_count(dims.batch_size)
      .set_height(GetTensorDim(compatible_input, data_format, 'H'))
      .set_width(GetTensorDim(compatible_input, data_format, 'W'))
      .set_feature_map_count(dims.in_depth)
      .set_layout(compute_data_layout);
  se::dnn::BatchDescriptor output_desc;
  output_desc.set_count(dims.batch_size)
      .set_height(dims.spatial_dims[0].output_size)
      .set_width(dims.spatial_dims[1].output_size)
      .set_feature_map_count(dims.out_depth)
      .set_layout(compute_data_layout);
  se::dnn::FilterDescriptor filter_desc;
  filter_desc.set_input_filter_height(dims.spatial_dims[0].filter_size)
      .set_input_filter_width(dims.spatial_dims[1].filter_size)
      .set_input_feature_map_count(filter_shape.dim_size(2))
      .set_output_feature_map_count(filter_shape.dim_size(3))
      .set_layout(filter_layout);
  se::dnn::ConvolutionDescriptor conv_desc;
  conv_desc.set_vertical_dilation_rate(dims.spatial_dims[0].dilation)
      .set_horizontal_dilation_rate(dims.spatial_dims[1].dilation)
      .set_vertical_filter_stride(dims.spatial_dims[0].stride)
      .set_horizontal_filter_stride(dims.spatial_dims[1].stride)
      .set_zero_padding_height(common_padding_rows)
      .set_zero_padding_width(common_padding_cols)
      .set_group_count(dims.in_depth / filter_shape.dim_size(2));

  // Tensorflow filter format: HWIO
  // cuDNN filter formats: (data format) -> (filter format)
  //   (1) NCHW -> OIHW
  //   (2) NHWC -> OHWI
  //
  // We compute filter backprop into temporary tensor, and then convert it to
  // the HWIO data format at the end.

  Tensor pre_transformed_filter_backprop;
  OP_REQUIRES_OK(
      ctx,
      ctx->allocate_temp(
          DataTypeToEnum<T>::value,
          TensorShape({filter_shape.dim_size(3), filter_shape.dim_size(2),
                       filter_shape.dim_size(0), filter_shape.dim_size(1)}),
          &pre_transformed_filter_backprop));

  Tensor transformed_out_backprop;
  if (data_format == FORMAT_NHWC && compute_data_format == FORMAT_NCHW) {
    VLOG(4) << "Convert the `out_backprop` tensor from NHWC to NCHW.";
    TensorShape compute_shape;
    OP_REQUIRES_OK(
        ctx, ShapeFromFormatWithStatus(compute_data_format, dims.batch_size,
                                       dims.spatial_dims[0].output_size,
                                       dims.spatial_dims[1].output_size,
                                       dims.out_depth, &compute_shape));
    if (dims.out_depth > 1) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<T>::value, compute_shape,
                                        &transformed_out_backprop));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(), out_backprop.tensor<T, 4>(),
          transformed_out_backprop.tensor<T, 4>());
    } else {
      // If depth <= 1, just reshape.
      CHECK(transformed_out_backprop.CopyFrom(out_backprop, compute_shape));
    }
  } else {
    transformed_out_backprop = out_backprop;
  }

  Tensor transformed_input;
  if (data_format == FORMAT_NHWC && compute_data_format == FORMAT_NCHW) {
    VLOG(4) << "Convert the `input` tensor from NHWC to NCHW.";
    TensorShape compute_shape;
    OP_REQUIRES_OK(ctx, ShapeFromFormatWithStatus(
                            compute_data_format,
                            GetTensorDim(compatible_input, data_format, 'N'),
                            GetTensorDim(compatible_input, data_format, 'H'),
                            GetTensorDim(compatible_input, data_format, 'W'),
                            GetTensorDim(compatible_input, data_format, 'C'),
                            &compute_shape));
    if (compute_shape.dim_size(1) > 1) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<T>::value, compute_shape,
                                        &transformed_input));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(compatible_input).tensor<T, 4>(),
          transformed_input.tensor<T, 4>());
    } else {
      // If depth <= 1, just reshape.
      CHECK(transformed_input.CopyFrom(compatible_input, compute_shape));
    }
  } else {
    transformed_input = compatible_input;
  }

  se::DeviceMemory<T> out_backprop_ptr =
      AsDeviceMemory(transformed_out_backprop.template flat<T>().data(),
                     transformed_out_backprop.template flat<T>().size());
  se::DeviceMemory<T> filter_backprop_ptr =
      AsDeviceMemory(pre_transformed_filter_backprop.template flat<T>().data(),
                     pre_transformed_filter_backprop.template flat<T>().size());
  auto input_ptr = AsDeviceMemory(transformed_input.template flat<T>().data(),
                                  transformed_input.template flat<T>().size());

  static int64_t ConvolveBackwardFilterScratchSize =
      GetDnnWorkspaceLimitOrDefault();
  ConvParameters conv_parameters = {
      stream->parent(),
      dims.batch_size,                     // batch
      dims.in_depth,                       // in_depths
      {{input_desc.height(),               // in_rows
        input_desc.width()}},              // in_cols
      compute_data_format,                 // compute_data_format
      dims.out_depth,                      // out_depths
      {{dims.spatial_dims[0].filter_size,  // filter_rows
        dims.spatial_dims[1].filter_size,  // filter_cols
        filter_shape.dim_size(2)}},        // filter_depth
      {{dims.spatial_dims[0].dilation,     // dilation_rows
        dims.spatial_dims[1].dilation}},   // dilation_cols
      {{dims.spatial_dims[0].stride,       // stride_rows
        dims.spatial_dims[1].stride}},     // stride_cols
      {{common_padding_rows,               // padding_rows
        common_padding_cols}},             // padding_cols
      input.dtype(),                       // tensor datatype
      conv_desc.group_count(),             // group_count
  };

  auto entry_or = AutotuneUnfusedConv(
      cudnn_use_autotune, AutotuneConvBwdFilter::GetInstance(), conv_parameters,
      ctx, se::dnn::ConvolutionKind::BACKWARD_FILTER, input_desc, input_ptr,
      filter_desc, filter_backprop_ptr, conv_desc, output_desc,
      out_backprop_ptr, ConvolveBackwardFilterScratchSize);
  OP_REQUIRES_OK(ctx, entry_or.status());
  auto autotune_entry = std::move(entry_or).value();

  DnnScratchAllocator scratch_allocator(ConvolveBackwardFilterScratchSize, ctx);
  Status cudnn_launch_status = LaunchAutotunedConv(
      autotune_entry, &scratch_allocator,
      se::dnn::ConvolutionKind::BACKWARD_FILTER, stream, input_desc, input_ptr,
      filter_desc, filter_backprop_ptr, conv_desc, output_desc,
      out_backprop_ptr);
  if (!cudnn_launch_status.ok()) {
    ctx->SetStatus(cudnn_launch_status);
    return;
  }

  FilterTensorFormat src_filter_format =
      compute_data_format == FORMAT_NCHW ? FORMAT_OIHW : FORMAT_OHWI;

  auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
  functor::ReverseTransformFilter<GPUDevice, T, 4>()(
      ctx->eigen_device<GPUDevice>(), src_filter_format,
      toConstTensor(pre_transformed_filter_backprop).template tensor<T, 4>(),
      filter_backprop->tensor<T, 4>());
}

template <typename T>
void LaunchConv2DBackpropFilterOp<Eigen::GpuDevice, T>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& out_backprop, const Tensor& input, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64_t>& explicit_paddings, Tensor* filter_backprop,
    TensorFormat data_format) {
  LaunchConv2DBackpropFilterOpImpl<T>(
      ctx, use_cudnn, cudnn_use_autotune, out_backprop, input, row_dilation,
      col_dilation, row_stride, col_stride, padding, explicit_paddings,
      filter_backprop, data_format);
}

template <>
void LaunchConv2DBackpropFilterOp<Eigen::GpuDevice, Eigen::bfloat16>::
operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
           const Tensor& out_backprop, const Tensor& input, int row_dilation,
           int col_dilation, int row_stride, int col_stride,
           const Padding& padding,
           const std::vector<int64_t>& explicit_paddings,
           Tensor* filter_backprop, TensorFormat data_format) {
  auto* stream = ctx->op_device_context()->stream();
  const bool cast_to_float = !IsBF16SupportedInOps(stream);

  if (cast_to_float) {
    Tensor casted_input = input;
    Tensor casted_out_backprop = out_backprop;
    Tensor casted_filter_backprop = *filter_backprop;

    const GPUDevice& device = ctx->eigen_device<GPUDevice>();
    functor::CastFunctor<GPUDevice, float, Eigen::bfloat16> cast;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_FLOAT, input.shape(), &casted_input));
    cast(device, casted_input.template flat<float>(),
         input.template flat<Eigen::bfloat16>());

    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, out_backprop.shape(),
                                           &casted_out_backprop));
    cast(device, casted_out_backprop.template flat<float>(),
         out_backprop.template flat<Eigen::bfloat16>());

    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, filter_backprop->shape(),
                                           &casted_filter_backprop));

    LaunchConv2DBackpropFilterOpImpl<float>(
        ctx, use_cudnn, cudnn_use_autotune, casted_out_backprop, casted_input,
        row_dilation, col_dilation, row_stride, col_stride, padding,
        explicit_paddings, &casted_filter_backprop, data_format);

    functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast_back;
    const Tensor& casted_filter_backprop_const = casted_filter_backprop;
    cast_back(device, filter_backprop->template flat<Eigen::bfloat16>(),
              casted_filter_backprop_const.template flat<float>());
    return;
  }

  LaunchConv2DBackpropFilterOpImpl<Eigen::bfloat16>(
      ctx, use_cudnn, cudnn_use_autotune, out_backprop, input, row_dilation,
      col_dilation, row_stride, col_stride, padding, explicit_paddings,
      filter_backprop, data_format);
}

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                             \
  template <>                                                           \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(               \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,         \
      typename TTypes<T, 4, int>::ConstTensor in,                       \
      typename TTypes<T, 4, int>::Tensor out);                          \
  extern template struct TransformFilter<GPUDevice, T, int, 4>;         \
  template <>                                                           \
  void PadInput<GPUDevice, T, int, 4>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,   \
      const std::array<int, 2>& padding_left,                           \
      const std::array<int, 2>& padding_right,                          \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format, \
      const T& padding_value);                                          \
  extern template struct PadInput<GPUDevice, T, int, 4>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(Eigen::bfloat16);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

template struct LaunchConv2DBackpropFilterOp<GPUDevice, float>;
template struct LaunchConv2DBackpropFilterOp<GPUDevice, Eigen::half>;
template struct LaunchConv2DBackpropFilterOp<GPUDevice, double>;

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
