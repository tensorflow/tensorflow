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

// See docs in ../ops/nn_ops.cc.

#include "tensorflow/core/kernels/conv_grad_input_ops.h"

#include <utility>

#include "tensorflow/core/profiler/lib/scoped_annotation.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/kernels/numeric_options_utils.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// To be used inside depthwise_conv_grad_op.cc.
template struct LaunchConv2DBackpropInputOp<CPUDevice, bfloat16>;
template struct LaunchConv2DBackpropInputOp<CPUDevice, Eigen::half>;
template struct LaunchConv2DBackpropInputOp<CPUDevice, float>;
template struct LaunchConv2DBackpropInputOp<CPUDevice, double>;

// GPU definitions.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// The slow version (but compiles for GPU)

// A dummy type to group forward backward data autotune results together.
struct ConvBackwardDataAutotuneGroup {
  static string name() { return "ConvBwdData"; }
};

typedef AutotuneSingleton<ConvBackwardDataAutotuneGroup, ConvParameters,
                          AutotuneEntry<se::dnn::ConvOp>>
    AutotuneConvBwdData;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Computes backprop input using Eigen::SpatialConvolutionBackwardInput on GPU
// for int32 inputs.
template <>
struct LaunchConv2DBackpropInputOp<GPUDevice, int32> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& out_backprop, const Tensor& filter,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  const std::vector<int64_t>& explicit_paddings,
                  Tensor* in_backprop, TensorFormat data_format) {
    LaunchConv2DBackpropInputOpImpl<GPUDevice, int32> launcher;
    launcher(ctx, use_cudnn, cudnn_use_autotune, out_backprop, filter,
             row_dilation, col_dilation, row_stride, col_stride, padding,
             explicit_paddings, in_backprop, data_format);
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
void LaunchConv2DBackpropInputOpGpuImpl(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& out_backprop, const Tensor& filter, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64_t>& explicit_paddings, Tensor* in_backprop,
    TensorFormat data_format) {
  using se::dnn::AlgorithmConfig;
  using se::dnn::AlgorithmDesc;
  using se::dnn::ProfileResult;

  std::vector<int32> strides(4, 1);
  std::vector<int32> dilations(4, 1);
  auto input_h = GetTensorDimIndex(data_format, 'H');
  auto input_w = GetTensorDimIndex(data_format, 'W');
  strides[input_h] = row_stride;
  strides[input_w] = col_stride;
  dilations[input_h] = row_dilation;
  dilations[input_w] = col_dilation;
  TensorShape input_shape = in_backprop->shape();

  const TensorShape& filter_shape = filter.shape();
  ConvBackpropDimensions dims;
  OP_REQUIRES_OK(
      ctx, ConvBackpropComputeDimensionsV2(
               "Conv2DSlowBackpropInput", /*num_spatial_dims=*/2, input_shape,
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
        "Conv2DBackpropInput for GPU is not currently supported "
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
  if (dims.spatial_dims[0].filter_size == 1 &&
      dims.spatial_dims[1].filter_size == 1 && !is_grouped_convolution &&
      dims.spatial_dims[0].stride == 1 && dims.spatial_dims[1].stride == 1 &&
      data_format == FORMAT_NHWC && (padding == VALID || padding == SAME)) {
    // 1x1 filter, so call cublas directly.
    const uint64 m = dims.batch_size * dims.spatial_dims[0].input_size *
                     dims.spatial_dims[1].input_size;
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

    OP_REQUIRES_OK(
        ctx, blas->BlasGemm(stream, transpose, no_transpose, n, m, k, b_ptr, k,
                            a_ptr, k, &c_ptr, n, GetNumericOptions(),
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
    const uint64 m = dims.batch_size;
    const uint64 k = dims.out_depth;
    const uint64 n = dims.spatial_dims[0].input_size *
                     dims.spatial_dims[1].input_size * dims.in_depth;

    auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                out_backprop.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                filter.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                in_backprop->template flat<T>().size());

    auto transpose = se::blas::Transpose::kTranspose;
    auto no_transpose = se::blas::Transpose::kNoTranspose;

    OP_REQUIRES_OK(
        ctx, blas->BlasGemm(stream, transpose, no_transpose, n, m, k, b_ptr, k,
                            a_ptr, k, &c_ptr, n, GetNumericOptions(),
                            se::blas::CallContext::kNone));
    return;
  }

  const int64_t common_padding_rows = std::min(padding_top, padding_bottom);
  const int64_t common_padding_cols = std::min(padding_left, padding_right);
  TensorShape compatible_input_shape;
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
    OP_REQUIRES_OK(
        ctx, ShapeFromFormatWithStatus(data_format, dims.batch_size,
                                       new_in_rows, new_in_cols, dims.in_depth,
                                       &compatible_input_shape));
  } else {
    compatible_input_shape = input_shape;
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

  VLOG(3) << "Compute Conv2DBackpropInput with cuDNN:"
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
      .set_height(GetTensorDim(compatible_input_shape, data_format, 'H'))
      .set_width(GetTensorDim(compatible_input_shape, data_format, 'W'))
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

  Tensor transformed_filter;
  const auto transform_filter = [&](FilterTensorFormat dst_format) -> Status {
    VLOG(4) << "Transform filter tensor from " << ToString(FORMAT_HWIO)
            << " to " << ToString(dst_format);

    TensorShape dst_shape =
        dst_format == FORMAT_OIHW
            ? TensorShape({filter.dim_size(3), filter.dim_size(2),
                           filter.dim_size(0), filter.dim_size(1)})
            : TensorShape({filter.dim_size(3), filter.dim_size(0),
                           filter.dim_size(1), filter.dim_size(2)});

    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<T>::value, dst_shape,
                                          &transformed_filter));
    functor::TransformFilter<GPUDevice, T, int, 4>()(
        ctx->eigen_device<GPUDevice>(), dst_format,
        To32Bit(filter.tensor<T, 4>()),
        To32Bit(transformed_filter.tensor<T, 4>()));

    return OkStatus();
  };

  if (compute_data_format == FORMAT_NCHW) {
    OP_REQUIRES_OK(ctx, transform_filter(FORMAT_OIHW));
  } else if (compute_data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(ctx, transform_filter(FORMAT_OHWI));
  } else {
    ctx->SetStatus(errors::InvalidArgument("Invalid compute data format: ",
                                           ToString(compute_data_format)));
    return;
  }

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
      // If depth <= 1, then just reshape.
      CHECK(transformed_out_backprop.CopyFrom(out_backprop, compute_shape));
    }
  } else {
    transformed_out_backprop = out_backprop;
  }

  Tensor pre_transformed_in_backprop;
  TensorShape pre_transformed_in_backprop_shape;
  OP_REQUIRES_OK(ctx,
                 ShapeFromFormatWithStatus(
                     compute_data_format,
                     GetTensorDim(compatible_input_shape, data_format, 'N'),
                     GetTensorDim(compatible_input_shape, data_format, 'H'),
                     GetTensorDim(compatible_input_shape, data_format, 'W'),
                     GetTensorDim(compatible_input_shape, data_format, 'C'),
                     &pre_transformed_in_backprop_shape));
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                         pre_transformed_in_backprop_shape,
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

  static int64_t ConvolveBackwardDataScratchSize =
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
        filter_shape.dim_size(2)}},        // filter_depths
      {{dims.spatial_dims[0].dilation,     // dilation_rows
        dims.spatial_dims[1].dilation}},   // dilation_cols
      {{dims.spatial_dims[0].stride,       // stride_rows
        dims.spatial_dims[1].stride}},     // stride_cols
      {{common_padding_rows,               // padding_rows
        common_padding_cols}},             // padding_cols
      out_backprop.dtype(),                // tensor data type
      conv_desc.group_count(),             // group_count
  };

  auto entry_or = AutotuneUnfusedConv(
      cudnn_use_autotune, AutotuneConvBwdData::GetInstance(), conv_parameters,
      ctx, se::dnn::ConvolutionKind::BACKWARD_DATA, input_desc, in_backprop_ptr,
      filter_desc, filter_ptr, conv_desc, output_desc, out_backprop_ptr,
      ConvolveBackwardDataScratchSize);
  OP_REQUIRES_OK(ctx, entry_or.status());
  auto autotune_entry = std::move(entry_or).value();

  DnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize, ctx);
  Status cudnn_launch_status =
      LaunchAutotunedConv(autotune_entry, &scratch_allocator,
                          se::dnn::ConvolutionKind::BACKWARD_DATA, stream,
                          input_desc, in_backprop_ptr, filter_desc, filter_ptr,
                          conv_desc, output_desc, out_backprop_ptr);
  if (!cudnn_launch_status.ok()) {
    ctx->SetStatus(cudnn_launch_status);
    return;
  }

  if (padding_top != padding_bottom || padding_left != padding_right) {
    Tensor in_backprop_remove_padding;
    TensorShape in_backprop_remove_padding_shape;
    OP_REQUIRES_OK(ctx, ShapeFromFormatWithStatus(
                            compute_data_format,
                            GetTensorDim(input_shape, data_format, 'N'),
                            GetTensorDim(input_shape, data_format, 'H'),
                            GetTensorDim(input_shape, data_format, 'W'),
                            GetTensorDim(input_shape, data_format, 'C'),
                            &in_backprop_remove_padding_shape));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           in_backprop_remove_padding_shape,
                                           &in_backprop_remove_padding));

    // Remove the padding that was added to the input shape above.
    const int64_t input_pad_top = padding_top - common_padding_rows;
    const int64_t input_pad_bottom = padding_bottom - common_padding_rows;
    const int64_t input_pad_left = padding_left - common_padding_cols;
    const int64_t input_pad_right = padding_right - common_padding_cols;
    functor::PadInput<GPUDevice, T, int, 4>()(
        ctx->template eigen_device<GPUDevice>(),
        To32Bit(const_cast<const Tensor&>(pre_transformed_in_backprop)
                    .tensor<T, 4>()),
        {{static_cast<int>(-input_pad_top), static_cast<int>(-input_pad_left)}},
        {{static_cast<int>(-input_pad_bottom),
          static_cast<int>(-input_pad_right)}},
        To32Bit(in_backprop_remove_padding.tensor<T, 4>()), compute_data_format,
        T{});

    pre_transformed_in_backprop = in_backprop_remove_padding;
  }

  if (data_format == FORMAT_NHWC && compute_data_format == FORMAT_NCHW) {
    VLOG(4) << "Convert the output tensor back from NCHW to NHWC.";
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::NCHWToNHWC<GPUDevice, T, 4>()(
        ctx->eigen_device<GPUDevice>(),
        toConstTensor(pre_transformed_in_backprop).template tensor<T, 4>(),
        in_backprop->tensor<T, 4>());
  } else {
    *in_backprop = pre_transformed_in_backprop;
  }
}

template <typename T>
void LaunchConv2DBackpropInputOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& out_backprop, const Tensor& filter, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64_t>& explicit_paddings, Tensor* in_backprop,
    TensorFormat data_format) {
  LaunchConv2DBackpropInputOpGpuImpl<T>(
      ctx, use_cudnn, cudnn_use_autotune, out_backprop, filter, row_dilation,
      col_dilation, row_stride, col_stride, padding, explicit_paddings,
      in_backprop, data_format);
}

template <>
void LaunchConv2DBackpropInputOp<GPUDevice, Eigen::bfloat16>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& out_backprop, const Tensor& filter, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64_t>& explicit_paddings, Tensor* in_backprop,
    TensorFormat data_format) {
  auto* stream = ctx->op_device_context()->stream();
  const bool cast_to_float = !IsBF16SupportedInOps(stream);

  if (cast_to_float) {
    Tensor casted_out_backprop = out_backprop;
    Tensor casted_filter = filter;
    Tensor casted_in_backprop = *in_backprop;

    const GPUDevice& device = ctx->eigen_device<GPUDevice>();
    functor::CastFunctor<GPUDevice, float, Eigen::bfloat16> cast;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, out_backprop.shape(),
                                           &casted_out_backprop));
    cast(device, casted_out_backprop.template flat<float>(),
         out_backprop.template flat<Eigen::bfloat16>());

    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_FLOAT, filter.shape(), &casted_filter));
    cast(device, casted_filter.template flat<float>(),
         filter.template flat<Eigen::bfloat16>());

    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, in_backprop->shape(),
                                           &casted_in_backprop));

    LaunchConv2DBackpropInputOpGpuImpl<float>(
        ctx, use_cudnn, cudnn_use_autotune, casted_out_backprop, casted_filter,
        row_dilation, col_dilation, row_stride, col_stride, padding,
        explicit_paddings, &casted_in_backprop, data_format);

    functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast_back;
    const Tensor& casted_in_backprop_const = casted_in_backprop;
    cast_back(device, in_backprop->template flat<Eigen::bfloat16>(),
              casted_in_backprop_const.template flat<float>());
    return;
  }

  LaunchConv2DBackpropInputOpGpuImpl<Eigen::bfloat16>(
      ctx, use_cudnn, cudnn_use_autotune, out_backprop, filter, row_dilation,
      col_dilation, row_stride, col_stride, padding, explicit_paddings,
      in_backprop, data_format);
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

template <>
void SpatialConvolutionBackwardInputFunc<GPUDevice, int32>::operator()(
    const GPUDevice&, typename TTypes<int32, 4>::Tensor,
    typename TTypes<int32, 4>::ConstTensor,
    typename TTypes<int32, 4>::ConstTensor, Eigen::DenseIndex,
    Eigen::DenseIndex, Eigen::DenseIndex, Eigen::DenseIndex);
extern template struct SpatialConvolutionBackwardInputFunc<GPUDevice, int32>;

template <>
void SpatialConvolutionBackwardInputWithExplicitPaddingFunc<
    GPUDevice, int32>::operator()(const GPUDevice&,
                                  typename TTypes<int32, 4>::Tensor,
                                  typename TTypes<int32, 4>::ConstTensor,
                                  typename TTypes<int32, 4>::ConstTensor,
                                  Eigen::DenseIndex, Eigen::DenseIndex,
                                  Eigen::DenseIndex, Eigen::DenseIndex,
                                  Eigen::DenseIndex, Eigen::DenseIndex,
                                  Eigen::DenseIndex, Eigen::DenseIndex);
extern template struct SpatialConvolutionBackwardInputWithExplicitPaddingFunc<
    GPUDevice, int32>;

}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("input_sizes"),
                        Conv2DBackpropInputOp<GPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("input_sizes"),
                        Conv2DBackpropInputOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .HostMemory("input_sizes"),
                        Conv2DBackpropInputOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::bfloat16>("T")
                            .HostMemory("input_sizes"),
                        Conv2DBackpropInputOp<GPUDevice, Eigen::bfloat16>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input_sizes"),
                        Conv2DBackpropInputOp<GPUDevice, int32>);

// To be used inside depthwise_conv_grad_op.cc.
// TODO(reedwm): Move this and the definition to depthwise_conv_grad_op.cc.
template struct LaunchConv2DBackpropInputOp<GPUDevice, float>;
template struct LaunchConv2DBackpropInputOp<GPUDevice, Eigen::half>;
template struct LaunchConv2DBackpropInputOp<GPUDevice, double>;

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
