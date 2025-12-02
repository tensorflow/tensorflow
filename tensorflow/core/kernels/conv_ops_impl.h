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

// See docs in ../ops/nn_ops.cc.

#ifndef TENSORFLOW_CORE_KERNELS_CONV_OPS_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_CONV_OPS_IMPL_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/op_requires.h"

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <string.h>

#include <array>
#include <atomic>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/synchronization/blocking_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_3d.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/deep_conv2d.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/numeric_options_utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/autotune_maps/conv_autotune_maps.h"
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

  /* cuDNN batch-splitting fallback removed (unrelated to XLA dict-key fix). */
  bool is_grouped_convolution = filter_depth != in_depth;
  // check if filter is 1x1 and stride/dilation are all ones
  bool one_filter = true;
  bool one_dilations = true;
  bool one_stride = true;
  for (int i = 0; i < spatial_dims; ++i) {
    one_filter = one_filter && (filter_dims[i] == 1);
    one_dilations = one_dilations && (dilations[i] == 1);
    one_stride = one_stride && (strides[i] == 1);
  }
  // check if filter is same spatial shape as input
  bool filter_same_dims = true;
  for (int i = 0; i < spatial_dims; ++i) {
    if (filter_dims[i] != in_dims[i]) filter_same_dims = false;
  }

  auto* blas = stream->parent()->AsBlas();
  OP_REQUIRES(context, blas != nullptr,
              absl::InternalError("No BLAS for stream."));
  if (!is_grouped_convolution && one_filter && one_dilations && one_stride &&
      data_format == FORMAT_NHWC && (padding == VALID || padding == SAME)) {
    // 1x1 filter, so call cublas directly.
    const uint64 m = in_batch * std::accumulate(in_dims.begin(), in_dims.end(),
                                                1, std::multiplies<>{});
    const uint64 k = in_depth;
    const uint64 n = out_depth;

    auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                input.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                filter.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                output->template flat<T>().size());

    auto no_transpose = se::blas::Transpose::kNoTranspose;
    OP_REQUIRES_OK(context, blas->BlasGemm(stream, no_transpose, no_transpose,
                                           n, m, k, b_ptr, n, a_ptr, k, &c_ptr,
                                           n, GetNumericOptions(),
                                           se::blas::CallContext::kNone));
    return;
  } else if (!is_grouped_convolution && filter_same_dims && padding == VALID &&
             data_format == FORMAT_NHWC) {
    // The input data and filter have the same spatial dimensions, so call
    // cublas directly.
    const uint64 m = in_batch;
    const uint64 k = in_depth * std::accumulate(in_dims.begin(), in_dims.end(),
                                                1, std::multiplies<>{});
    const uint64 n = out_depth;

    auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                input.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                filter.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                output->template flat<T>().size());

    auto no_transpose = se::blas::Transpose::kNoTranspose;
    OP_REQUIRES_OK(context, blas->BlasGemm(stream, no_transpose, no_transpose,
                                           n, m, k, b_ptr, n, a_ptr, k, &c_ptr,
                                           n, GetNumericOptions(),
                                           se::blas::CallContext::kNone));
    return;
  }

  const bool compute_in_nhwc = ComputeInNhwcEnabled(
      DataTypeToEnum<T>::value, stream, /*use_4d_tensor=*/(spatial_dims == 2));
  const TensorFormat compute_data_format =
      (compute_in_nhwc && data_format == FORMAT_NHWC) ? FORMAT_NHWC
                                                      : FORMAT_NCHW;

  VLOG(3) << "Compute Conv with cuDNN:"
          << " data_format=" << ToString(data_format)
          << " compute_data_format=" << ToString(compute_data_format);

  std::vector<int64_t> out_dims(output->dims());
  for (int i = 0; i < output->dims(); ++i) {
    out_dims[i] = output->dim_size(i);
  }
  std::vector<std::pair<int64_t, int64_t>> paddings(spatial_dims, {-1, -1});
  // Explicit only on 2D case.
  if (padding == EXPLICIT) {
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'H',
                             &paddings[0].first, &paddings[0].second);
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'W',
                             &paddings[1].first, &paddings[1].second);
  }

  // Get padding values, output should be valid, since it was checked before.
  std::vector<int64_t> out_dims_check(spatial_dims);
  for (int i = 0; i < spatial_dims; ++i) {
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                in_dims[i], filter_dims[i], dilations[i],
                                strides[i], padding, &out_dims_check[i],
                                &paddings[i].first, &paddings[i].second));
    OP_REQUIRES(context,
                (out_dims_check[i] == GetTensorDim(*output, data_format,
                                                   static_cast<char>('0' + i))),
                absl::InternalError("Output dimension doesn't match yo"));
  }

  bool assymmetric_padding = false;
  std::vector<int64_t> common_padding(spatial_dims);
  for (int i = 0; i < spatial_dims; ++i) {
    common_padding[i] = std::min(paddings[i].first, paddings[i].second);
    assymmetric_padding =
        assymmetric_padding || (paddings[i].first != paddings[i].second);
  }

  if (assymmetric_padding) {
    // cuDNN only supports padding the same amount on either side. So we
    // manually create a new padded input tensor.
    Tensor transformed_input;
    std::vector<int64_t> new_in_dims(input.dims());
    new_in_dims[0] = in_batch;
    for (int i = 0; i < spatial_dims; ++i) {
      int index = GetTensorSpatialDimIndex(input.dims(), data_format, i);
      new_in_dims[index] =
          in_dims[i] + std::abs(paddings[i].first - paddings[i].second);
    }
    new_in_dims[GetTensorDimIndex(data_format, 'C', input.dims())] = in_depth;
    TensorShape transformed_input_shape(new_in_dims);
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_input_shape,
                                                   &transformed_input));

    // Padding to add on transformed input.
    std::vector<std::pair<int64_t, int64_t>> transformed_input_padding(
        paddings);
    for (int i = 0; i < spatial_dims; ++i) {
      transformed_input_padding[i].first -= common_padding[i];
      transformed_input_padding[i].second -= common_padding[i];
    }

    // Check padding size.
    bool padding_bounds_valid = true;
    for (int i = 0; i < spatial_dims; ++i) {
      padding_bounds_valid =
          padding_bounds_valid &&
          FastBoundsCheck(transformed_input_padding[i].first,
                          std::numeric_limits<int>::max()) &&
          FastBoundsCheck(transformed_input_padding[i].second,
                          std::numeric_limits<int>::max());
    }
    OP_REQUIRES(context, padding_bounds_valid,
                absl::InvalidArgumentError("Padding is too large."));

    // Pad new input.
    if (input.dims() == 4) {
      std::array<int, 2> pad_left{
          static_cast<int>(transformed_input_padding[0].first),
          static_cast<int>(transformed_input_padding[1].first)};
      std::array<int, 2> pad_right{
          static_cast<int>(transformed_input_padding[0].second),
          static_cast<int>(transformed_input_padding[1].second)};
      functor::PadInput<GPUDevice, T, int, 4>()(
          context->eigen_device<GPUDevice>(),
          To32Bit(static_cast<const Tensor&>(input).tensor<T, 4>()), pad_left,
          pad_right, To32Bit(transformed_input.tensor<T, 4>()), data_format,
          T{});
    } else if (input.dims() == 5) {
      std::array<int, 3> pad_left{
          static_cast<int>(transformed_input_padding[0].first),
          static_cast<int>(transformed_input_padding[1].first),
          static_cast<int>(transformed_input_padding[2].first)};
      std::array<int, 3> pad_right{
          static_cast<int>(transformed_input_padding[0].second),
          static_cast<int>(transformed_input_padding[1].second),
          static_cast<int>(transformed_input_padding[2].second)};
      functor::PadInput<GPUDevice, T, int, 5>()(
          context->eigen_device<GPUDevice>(),
          To32Bit(static_cast<const Tensor&>(input).tensor<T, 5>()), pad_left,
          pad_right, To32Bit(transformed_input.tensor<T, 5>()), data_format,
          T{});
    } else {
      context->SetStatus(
          absl::InternalError("Failed to pad input, invalid dimensions."));
    }

    input = transformed_input;
    for (int i = 0; i < spatial_dims; ++i) {
      in_dims[i] = new_in_dims[GetTensorDimIndex(
          data_format, static_cast<char>('0' + i), input.dims())];
    }
  }

  if (data_format == FORMAT_NHWC && compute_data_format == FORMAT_NCHW) {
    VLOG(4) << "Convert the input tensor from NHWC to NCHW.";

    TensorShape channels_first_shape;
    OP_REQUIRES_OK(context,
                   ShapeFromFormatWithStatus(FORMAT_NCHW, in_batch, in_dims,
                                             in_depth, &channels_first_shape));

    if (in_depth > 1) {
      Tensor transformed_input;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                     channels_first_shape,
                                                     &transformed_input));
      if (input.dims() == 4) {
        functor::NHWCToNCHW<GPUDevice, T, 4>()(
            context->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(input).tensor<T, 4>(),
            transformed_input.tensor<T, 4>());
      } else if (input.dims() == 5) {
        functor::NHWCToNCHW<GPUDevice, T, 5>()(
            context->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(input).tensor<T, 5>(),
            transformed_input.tensor<T, 5>());
      } else {
        context->SetStatus(
            absl::InternalError("Failed to reshape input to channels first "
                                "format, invalid dimensions."));
      }
      input = transformed_input;
    } else {
      // Depth = 1, reshape.
      if (!input.CopyFrom(input, channels_first_shape)) {
        context->SetStatus(absl::InternalError(
            "Failed to reshape input to channels first format."));
      }
    }
  } else {
    DCHECK(data_format == compute_data_format)  // Crash OK.
        << "Illegal data and compute format pair:"
        << " data_format=" << ToString(data_format)
        << " compute_data_format=" << ToString(compute_data_format);
  }

  // Check paddings are not negative.
  bool non_negative_paddings = true;
  for (int i = 0; i < spatial_dims; ++i) {
    non_negative_paddings = non_negative_paddings && common_padding[i] >= 0;
  }
  OP_REQUIRES(context, non_negative_paddings,
              absl::InvalidArgumentError("Padding is negative."));

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

  se::dnn::BatchDescriptor input_desc(spatial_dims);
  input_desc.set_count(in_batch).set_feature_map_count(in_depth).set_layout(
      compute_data_layout);
  if (spatial_dims == 2) {
    input_desc.set_spatial_dim(stream_executor::dnn::DimIndex::X, in_dims[1])
        .set_spatial_dim(stream_executor::dnn::DimIndex::Y, in_dims[0]);
  } else if (spatial_dims == 3) {
    input_desc.set_spatial_dim(stream_executor::dnn::DimIndex::X, in_dims[2])
        .set_spatial_dim(stream_executor::dnn::DimIndex::Y, in_dims[1])
        .set_spatial_dim(stream_executor::dnn::DimIndex::Z, in_dims[0]);
  } else {
    context->SetStatus(
        absl::InternalError("Failed to set Input Descripitor:"
                            " invalid number of spatial dimensions"));
  }

  se::dnn::BatchDescriptor output_desc(spatial_dims);
  output_desc.set_count(GetTensorDim(*output, data_format, 'N'))
      .set_feature_map_count(GetTensorDim(*output, data_format, 'C'))
      .set_layout(compute_data_layout);
  if (spatial_dims == 2) {
    output_desc
        .set_spatial_dim(
            stream_executor::dnn::DimIndex::X,
            GetTensorDim(*output, data_format, static_cast<int>('1')))
        .set_spatial_dim(
            stream_executor::dnn::DimIndex::Y,
            GetTensorDim(*output, data_format, static_cast<int>('0')));
  } else if (spatial_dims == 3) {
    output_desc
        .set_spatial_dim(
            stream_executor::dnn::DimIndex::X,
            GetTensorDim(*output, data_format, static_cast<int>('2')))
        .set_spatial_dim(
            stream_executor::dnn::DimIndex::Y,
            GetTensorDim(*output, data_format, static_cast<int>('1')))
        .set_spatial_dim(
            stream_executor::dnn::DimIndex::Z,
            GetTensorDim(*output, data_format, static_cast<int>('0')));
  } else {
    context->SetStatus(
        absl::InternalError("Failed to set Output Descripitor: invalid "
                            "number of spatial dimensions"));
  }

  se::dnn::FilterDescriptor filter_desc(spatial_dims);
  filter_desc.set_input_feature_map_count(filter_depth)
      .set_output_feature_map_count(out_depth)
      .set_layout(filter_layout);
  if (spatial_dims == 2) {
    filter_desc
        .set_spatial_dim(stream_executor::dnn::DimIndex::X, filter_dims[1])
        .set_spatial_dim(stream_executor::dnn::DimIndex::Y, filter_dims[0]);
  } else if (spatial_dims == 3) {
    filter_desc
        .set_spatial_dim(stream_executor::dnn::DimIndex::X, filter_dims[2])
        .set_spatial_dim(stream_executor::dnn::DimIndex::Y, filter_dims[1])
        .set_spatial_dim(stream_executor::dnn::DimIndex::Z, filter_dims[0]);
  } else {
    context->SetStatus(
        absl::InternalError("Failed to set Filter Descripitor: invalid "
                            "number of spatial dimensions"));
  }

  se::dnn::ConvolutionDescriptor conv_desc(spatial_dims);
  if (spatial_dims == 2) {
    conv_desc.set_dilation_rate(stream_executor::dnn::DimIndex::X, dilations[1])
        .set_dilation_rate(stream_executor::dnn::DimIndex::Y, dilations[0])
        .set_filter_stride(stream_executor::dnn::DimIndex::X, strides[1])
        .set_filter_stride(stream_executor::dnn::DimIndex::Y, strides[0])
        .set_zero_padding(stream_executor::dnn::DimIndex::X, common_padding[1])
        .set_zero_padding(stream_executor::dnn::DimIndex::Y, common_padding[0]);
  } else if (spatial_dims == 3) {
    conv_desc.set_dilation_rate(stream_executor::dnn::DimIndex::X, dilations[2])
        .set_dilation_rate(stream_executor::dnn::DimIndex::Y, dilations[1])
        .set_dilation_rate(stream_executor::dnn::DimIndex::Z, dilations[0])
        .set_filter_stride(stream_executor::dnn::DimIndex::X, strides[2])
        .set_filter_stride(stream_executor::dnn::DimIndex::Y, strides[1])
        .set_filter_stride(stream_executor::dnn::DimIndex::Z, strides[0])
        .set_zero_padding(stream_executor::dnn::DimIndex::X, common_padding[2])
        .set_zero_padding(stream_executor::dnn::DimIndex::Y, common_padding[1])
        .set_zero_padding(stream_executor::dnn::DimIndex::Z, common_padding[0]);
  } else {
    context->SetStatus(
        absl::InternalError("Failed to set Convolution Descripitor: invalid "
                            "number of spatial dimensions"));
  }
  conv_desc.set_group_count(1);
  // TODO(b/290223810) Change group count when implementing group/depthwise.
  Tensor transformed_filter;
  auto dst_format =
      compute_data_format == FORMAT_NCHW ? FORMAT_OIHW : FORMAT_OHWI;
  VLOG(4) << "Transform filter tensor from " << ToString(FORMAT_HWIO) << " to "
          << ToString(dst_format);
  std::vector<int64_t> dst_shape_vec(spatial_dims + 2);
  dst_shape_vec[0] = out_depth;
  if (dst_format == FORMAT_OIHW) {
    dst_shape_vec[1] = filter_depth;
    for (int i = 2; i < filter.dims(); ++i) {
      dst_shape_vec[i] = filter_dims[i - 2];
    }
  } else {
    // Format OHWI
    dst_shape_vec[filter.dims() - 1] = filter_depth;
    for (int i = 1; i < filter.dims() - 1; ++i) {
      dst_shape_vec[i] = filter_dims[i - 1];
    }
  }
  TensorShape dst_shape(dst_shape_vec);
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DataTypeToEnum<T>::value, dst_shape,
                                        &transformed_filter));

  // Filter: [(spatial_dims), in, out] (HWIO)
  // T_filter: [out, in, (spatial_dims)] (OIHW) or
  // T_filter: [out, (spatial_dims), in] (OHWI)
  if (spatial_dims == 2) {
    functor::TransformFilter<GPUDevice, T, int, 4>()(
        context->eigen_device<GPUDevice>(), dst_format,
        To32Bit(filter.tensor<T, 4>()),
        To32Bit(transformed_filter.tensor<T, 4>()));
  } else if (spatial_dims == 3) {
    functor::TransformFilter<GPUDevice, T, int, 5>()(
        context->eigen_device<GPUDevice>(), dst_format,
        To32Bit(filter.tensor<T, 5>()),
        To32Bit(transformed_filter.tensor<T, 5>()));
  } else {
    context->SetStatus(absl::InternalError(
        "Failed to reshape filter, invalid spatial dimensions."));
  }

  Tensor transformed_output;
  if (data_format != compute_data_format) {
    VLOG(4) << "Allocate temporary memory for output in compute data format";
    TensorShape transformed_output_shape;
    OP_REQUIRES_OK(context, ShapeFromFormatWithStatus(
                                FORMAT_NCHW, in_batch, out_dims_check,
                                out_depth, &transformed_output_shape));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_output_shape,
                                                   &transformed_output));
  } else {
    transformed_output = *output;
  }

  auto input_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                  input.template flat<T>().size());
  auto filter_ptr =
      AsDeviceMemory(transformed_filter.template flat<T>().data(),
                     transformed_filter.template flat<T>().size());
  auto output_ptr =
      AsDeviceMemory(transformed_output.template flat<T>().data(),
                     transformed_output.template flat<T>().size());

  static int64_t ConvolveScratchSize = GetDnnWorkspaceLimitOrDefault();

  if (spatial_dims == 2) {
    filter_dims.push_back(filter_depth);
  }
  ConvParameters conv_parameters = {
      stream->parent(),
      in_batch,             // batch
      in_depth,             // in_depths
      in_dims,              // input spatial dims
      compute_data_format,  // compute_data_format
      out_depth,            // out_depths
      filter_dims,          // filter spatial dims
      dilations,            // dilations
      strides,              // strides
      common_padding,       // paddings (symmetrical)
      input.dtype(),        // tensor datatype
      conv_desc.group_count(),
  };

  auto entry_or = AutotuneUnfusedConv(
      cudnn_use_autotune, ConvAutotuneMap::GetInstance(), conv_parameters,
      context, se::dnn::ConvolutionKind::FORWARD, input_desc, input_ptr,
      filter_desc, filter_ptr, conv_desc, output_desc, output_ptr,
      ConvolveScratchSize);
  OP_REQUIRES_OK(context, entry_or.status());
  auto autotune_entry = std::move(entry_or).value();

  DnnScratchAllocator scratch_allocator(ConvolveScratchSize, context);
  Status cudnn_launch_status = LaunchAutotunedConv(
      autotune_entry, &scratch_allocator, se::dnn::ConvolutionKind::FORWARD,
      stream, input_desc, input_ptr, filter_desc, filter_ptr, conv_desc,
      output_desc, output_ptr);
  if (!cudnn_launch_status.ok()) {
    context->SetStatus(cudnn_launch_status);
    return;
  }

  if (data_format == FORMAT_NHWC && compute_data_format == FORMAT_NCHW) {
    VLOG(4) << "Convert the output tensor back from NCHW to NHWC.";
    if (spatial_dims == 2) {
      functor::NCHWToNHWC<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(transformed_output).tensor<T, 4>(),
          output->tensor<T, 4>());
    } else if (spatial_dims == 3) {
      functor::NCHWToNHWC<GPUDevice, T, 5>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(transformed_output).tensor<T, 5>(),
          output->tensor<T, 5>());
    } else {
      context->SetStatus(absl::InternalError(
          "Failed to convert output data foramt, invalid spatial dimensions."));
    }
  }
}

template <typename T>
void LaunchConvOp<GPUDevice, T>::operator()(
    OpKernelContext* context, bool cudnn_use_autotune, const Tensor& input,
    const Tensor& filter, const std::vector<int64>& dilations,
    const std::vector<int64>& strides, const Padding padding,
    const std::vector<int64_t>& explicit_paddings, TensorFormat data_format,
    Tensor* output) {
  // Get spatial dims for dilations and strides.
  int spatial_dims = input.dims() - 2;
  gtl::InlinedVector<int64_t, 3> strides_spatial(spatial_dims);
  gtl::InlinedVector<int64_t, 3> dilations_spatial(spatial_dims);
  for (int i = 0; i < spatial_dims; ++i) {
    strides_spatial[i] =
        GetTensorDim(strides, data_format, static_cast<char>(i + '0'));
    dilations_spatial[i] =
        GetTensorDim(dilations, data_format, static_cast<char>(i + '0'));
  }
  LaunchConvOpImpl<T>(context, cudnn_use_autotune, input, filter,
                      dilations_spatial, strides_spatial, padding,
                      explicit_paddings, data_format, output);
}

template <typename T>
void LaunchConv2DOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& input_param, const Tensor& filter, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64_t>& explicit_paddings, Tensor* output,
    TensorFormat data_format) {
  // Cast strides and dilations.
  gtl::InlinedVector<int64_t, 3> casted_strides = {row_stride, col_stride};
  gtl::InlinedVector<int64_t, 3> casted_dilations = {row_dilation,
                                                     col_dilation};
  LaunchConvOpImpl<T>(ctx, cudnn_use_autotune, input_param, filter,
                      casted_dilations, casted_strides, padding,
                      explicit_paddings, data_format, output);
}

// To be used inside depthwise_conv_op.cc.
extern template struct LaunchConv2DOp<GPUDevice, float>;
// extern template struct LaunchConv2DOp<GPUDevice, Eigen::bfloat16>;
extern template struct LaunchConv2DOp<GPUDevice, Eigen::half>;
extern template struct LaunchConv2DOp<GPUDevice, double>;

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_IMPL_H_
