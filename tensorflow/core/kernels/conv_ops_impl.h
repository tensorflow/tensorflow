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
#include <limits>
#include <map>
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
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/tf_allocator_adapter.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchGeneric {
  void operator()(OpKernelContext* ctx, const Tensor& input,
                  const Tensor& filter, int row_stride, int col_stride,
                  int row_dilation, int col_dilation, const Padding& padding,
                  const std::vector<int64_t>& explicit_paddings, Tensor* output,
                  TensorFormat data_format) {
    DCHECK(data_format == FORMAT_NHWC)
        << "Generic conv implementation only "
           "supports NHWC tensor format for now.";
    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 && row_stride == 1 &&
        col_stride == 1 && (padding == SAME || padding == VALID)) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      //
      // TODO(vrv): We should be able to call SpatialConvolution
      // and it will produce the same result, but doing so
      // led to NaNs during training.  Using matmul instead for now.
      int conv_width = 1;  // Width for the convolution step.
      for (int i = 0; i < 3; ++i) {
        conv_width *= output->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<Device, T>()(
          ctx->eigen_device<Device>(),
          output->shaped<T, 2>({conv_width, filter.dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter.dim_size(2)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair);
    } else if (filter.dim_size(0) == input.dim_size(1) &&
               filter.dim_size(1) == input.dim_size(2) && row_dilation == 1 &&
               col_dilation == 1 && padding == VALID) {
      // If the input data and filter have the same height/width,
      // the 2D convolution is reduced to matrix multiplication.
      const int k =  // Length of reduction dimension.
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<Device, T>()(
          ctx->eigen_device<Device>(),
          output->shaped<T, 2>({input.dim_size(0), filter.dim_size(3)}),
          input.shaped<T, 2>({input.dim_size(0), k}),
          filter.shaped<T, 2>({k, filter.dim_size(3)}), dim_pair);
    } else {
      if (padding == EXPLICIT) {
        functor::SpatialConvolution<Device, T>()(
            ctx->eigen_device<Device>(), output->tensor<T, 4>(),
            input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride, col_stride,
            row_dilation, col_dilation, static_cast<int>(explicit_paddings[2]),
            static_cast<int>(explicit_paddings[3]),
            static_cast<int>(explicit_paddings[4]),
            static_cast<int>(explicit_paddings[5]));
      } else {
        functor::SpatialConvolution<Device, T>()(
            ctx->eigen_device<Device>(), output->tensor<T, 4>(),
            input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride, col_stride,
            row_dilation, col_dilation, BrainPadding2EigenPadding(padding));
      }
    }
  }
};

// Compute grouped 2D convolutions on CPU. Unlike grouped convolution
// implementation in cuDNN this is faaaaaar from optimal and needs more work
// to deliver competitive performance. Currently it exists to close the feature
// parity gap between convolution operations on different devices.
template <typename T>
struct LaunchGrouped {
  void operator()(OpKernelContext* ctx, const Tensor& input,
                  const Tensor& filter, int row_stride, int col_stride,
                  int row_dilation, int col_dilation, const Padding& padding,
                  const std::vector<int64_t>& explicit_paddings, Tensor* output,
                  TensorFormat data_format) {
    DCHECK(data_format == FORMAT_NHWC)
        << "Grouped conv implementation only "
           "supports NHWC tensor format for now.";

    const int64_t in_depth = input.dim_size(3);
    const int64_t patch_depth = filter.dim_size(2);
    const int64_t num_groups = in_depth / patch_depth;

    // Shuffle input/filter tensors to have group as a leading dimension.
    std::array<int64_t, 5> shuffle({3, 0, 1, 2, 4});

    // Compute pre shuffle dimemnsions.
    auto pre_shuffle = [&](const Tensor& tensor) -> std::array<int64, 5> {
      return {tensor.dim_size(0), tensor.dim_size(1), tensor.dim_size(2),
              num_groups, tensor.dim_size(3) / num_groups};
    };

    // Compute post shuffle dimemnsions.
    auto post_shuffle = [&](const Tensor& tensor) -> std::array<int64, 5> {
      return {num_groups, tensor.dim_size(0), tensor.dim_size(1),
              tensor.dim_size(2), tensor.dim_size(3) / num_groups};
    };

    auto& device = ctx->eigen_device<CPUDevice>();

    absl::BlockingCounter shuffles_completed(2);
    auto on_shuffled = [&]() { shuffles_completed.DecrementCount(); };

    // Shuffle input into temporary tensor.
    Tensor input_shuffled;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(input.dtype(), TensorShape(post_shuffle(input)),
                                &input_shuffled));
    input_shuffled.tensor<T, 5>().device(device, on_shuffled) =
        input.shaped<T, 5>(pre_shuffle(input)).shuffle(shuffle);

    // Shuffle filter into temporary tensor.
    Tensor filter_shuffled;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(filter.dtype(),
                                           TensorShape(post_shuffle(filter)),
                                           &filter_shuffled));
    filter_shuffled.tensor<T, 5>().device(device, on_shuffled) =
        filter.shaped<T, 5>(pre_shuffle(filter)).shuffle(shuffle);

    // Wait for the completion of input/filter shuffles.
    shuffles_completed.Wait();

    // Write group convolution results into temporary output tensor.
    Tensor output_shuffled;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(output->dtype(),
                                           TensorShape(post_shuffle(*output)),
                                           &output_shuffled));

    for (int64_t i = 0; i < num_groups; ++i) {
      // TODO(ezhulenev): Run this loop using `parallelFor` (regular parallelFor
      // will lead to deadlock, SpatialConvolution has to use async Eigen
      // assignment). This requires small changes to Eigen to support async
      // exeuction for tensor chipping operation.

      // TODO(ezhulenev): Grouped convolution should also support 1x1 filter
      // optimization.

      auto input_slice = input_shuffled.tensor<T, 5>().template chip<0>(i);
      auto filter_slice = filter_shuffled.tensor<T, 5>().template chip<0>(i);
      auto output_slice = output_shuffled.tensor<T, 5>().template chip<0>(i);

      if (padding == EXPLICIT) {
        functor::SpatialConvolution<CPUDevice, T>()(
            ctx->eigen_device<CPUDevice>(), output_slice, input_slice,
            filter_slice, row_stride, col_stride, row_dilation, col_dilation,
            static_cast<int>(explicit_paddings[2]),
            static_cast<int>(explicit_paddings[3]),
            static_cast<int>(explicit_paddings[4]),
            static_cast<int>(explicit_paddings[5]));
      } else {
        functor::SpatialConvolution<CPUDevice, T>()(
            ctx->eigen_device<CPUDevice>(), output_slice, input_slice,
            filter_slice, row_stride, col_stride, row_dilation, col_dilation,
            BrainPadding2EigenPadding(padding));
      }
    }

    // Shuffle temporary output back into pre-shuffled shape.
    std::array<int64_t, 5> rev_shuffle({1, 2, 3, 0, 4});
    output->shaped<T, 5>(pre_shuffle(*output)).device(device) =
        output_shuffled.tensor<T, 5>().shuffle(rev_shuffle);
  }
};

template <typename Device, typename T>
struct LaunchConvOp;

template <typename T>
struct LaunchConvOp<CPUDevice, T> {
  void operator()(OpKernelContext* context, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter,
                  const std::vector<int64>& dilations,
                  const std::vector<int64>& strides, const Padding padding,
                  const std::vector<int64_t>& explicit_paddings,
                  TensorFormat data_format, Tensor* output) {
    // For now just calling existing launchers based on spatial dimensions.
    int spatial_dims = input.dims() - 2;

    if (spatial_dims == 2) {
      LaunchConv2DOp<CPUDevice, T>()(context, true, cudnn_use_autotune, input,
                                     filter, dilations[1], dilations[2],
                                     strides[1], strides[2], padding,
                                     explicit_paddings, output, data_format);
    } else {
      LaunchConv3DOp<CPUDevice, T>().launch(
          context, cudnn_use_autotune, input, filter,
          {dilations[1], dilations[2], dilations[3]},
          {strides[1], strides[2], strides[3]}, padding, data_format, output);
    }
  }
};

template <typename Device, typename T>
class ConvOp : public BinaryOp<T> {
 public:
  explicit ConvOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    // TODO(b/290223810) Add support for grouped and depthwise convolutions.
    OP_REQUIRES_OK(context, context->GetAttr("groups", &groups_));
    OP_REQUIRES(context, groups_ == 1,
                absl::UnimplementedError(
                    "Grouped/Depthwise Convolutions are not supported yet."));
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context,
                data_format_str == "CHANNELS_LAST" ||
                    data_format_str == "CHANNELS_FIRST",
                absl::InvalidArgumentError(
                    absl::StrCat("Unknown data format: ", data_format_str)));
    data_format_ =
        data_format_str == "CHANNELS_LAST" ? FORMAT_NHWC : FORMAT_NCHW;

    // Always assume filter_format is HWIO / DHWIO.
    filter_format_ = FilterTensorFormat::FORMAT_HWIO;

    // These parameters are checked against spatial dimensions on compute.
    OP_REQUIRES_OK(context, context->GetAttr("batch_dims", &batch_dims_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    if (context->HasAttr("explicit_paddings")) {
      OP_REQUIRES_OK(
          context, context->GetAttr("explicit_paddings", &explicit_paddings_));
    }
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, [spatial_dims], in_depth ].
    const Tensor& input = context->input(0);
    size_t original_input_dims = context->input(0).dims();
    const TensorShape original_input_shape = context->input(0).shape();
    int spatial_dims = original_input_dims - 1 - batch_dims_;

    // Input filter is of the following dimensions:
    // [ batch, [spatial dims], in_depth ].
    const Tensor& filter = context->input(1);

    OP_REQUIRES(context, (spatial_dims == 2 || spatial_dims == 3),
                absl::InvalidArgumentError(absl::StrCat(
                    "The input must have 2 or 3 spatial dimensions but got ",
                    spatial_dims)));

    OP_REQUIRES(
        context, filter.NumElements() > 0,
        absl::InvalidArgumentError("filter must not have zero elements "
                                   "(i.e. all dimensions must be non-zero)"));

    // Flatten tensor for computation.
    Tensor input_flat;
    if (batch_dims_ == 1) {
      input_flat = input;
    } else {
      std::vector<int64_t> in_flat_shape_vec(1, 1);
      for (int i = 0; i < batch_dims_; ++i) {
        in_flat_shape_vec[0] *= original_input_shape.dim_size(i);
      }
      for (int i = batch_dims_; i < original_input_shape.dims(); ++i) {
        in_flat_shape_vec.push_back(original_input_shape.dim_size(i));
      }
      TensorShape in_flat_shape(in_flat_shape_vec);
      if (!input_flat.CopyFrom(input, in_flat_shape)) {
        // This should never happen, since the output sizes should always be the
        // same after expanding batches.
        context->SetStatus(absl::InternalError(absl::StrCat(
            "Could not flatten input shape ",
            original_input_shape.DebugString(), " and flat input shape ",
            in_flat_shape.DebugString())));
      }
    }

    OP_REQUIRES(context, filter.dims() == 4 || filter.dims() == 5,
                absl::InvalidArgumentError(absl::StrCat(
                    "The filter must be rank 4 or 5 but got ", filter.dims())));
    for (int i = 0; i < spatial_dims; i++) {
      OP_REQUIRES(
          context,
          FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
          absl::InvalidArgumentError("filter too large"));
    }

    // Validate operation parameters based on inferred spatial dims.
    OP_REQUIRES(context, strides_.size() == spatial_dims + 2,
                absl::InvalidArgumentError(
                    absl::StrCat("Sliding window strides field must specify ",
                                 spatial_dims + 2, " dimensions")));

    OP_REQUIRES(context,
                (GetTensorDim(strides_, data_format_, 'C') == 1 &&
                 GetTensorDim(strides_, data_format_, 'N') == 1),
                absl::InvalidArgumentError(
                    "Current implementation does not support "
                    "strides in the batch and depth dimensions."));
    bool stride_valid = true;
    for (int i = 0; i < spatial_dims; ++i) {
      stride_valid =
          stride_valid && (GetTensorDim(strides_, data_format_,
                                        static_cast<char>(i + '0')) > 0);
    }
    OP_REQUIRES(
        context, stride_valid,
        absl::InvalidArgumentError("Spatial strides should be larger than 0."));
    if (dilations_.empty()) {
      dilations_ = std::vector<int64_t>(spatial_dims + 2, 1);
    } else {
      OP_REQUIRES(context, dilations_.size() == spatial_dims + 2,
                  absl::InvalidArgumentError(
                      absl::StrCat("Dilation rates field must specify",
                                   spatial_dims + 2, "dimensions")));
      OP_REQUIRES(context,
                  (GetTensorDim(dilations_, data_format_, 'N') == 1 &&
                   GetTensorDim(dilations_, data_format_, 'C') == 1),
                  absl::InvalidArgumentError(
                      "Current implementation does not support "
                      "dilation rates in the batch and depth dimensions."));
      bool dilation_valid = true;
      for (int i = 0; i < spatial_dims; ++i) {
        dilation_valid =
            dilation_valid && (GetTensorDim(dilations_, data_format_,
                                            static_cast<char>(i + '0')) > 0);
      }
      OP_REQUIRES(
          context, dilation_valid,
          absl::InvalidArgumentError("Dilated rates should be larger than 0."));
    }
    OP_REQUIRES_OK(context, CheckValidPadding(padding_, explicit_paddings_,
                                              spatial_dims + 2, data_format_));

    const int64_t in_depth_raw = GetTensorDim(input_flat, data_format_, 'C');
    const int64_t patch_depth_raw = GetFilterDim(filter, filter_format_, 'I');
    OP_REQUIRES(context,
                FastBoundsCheck(in_depth_raw, std::numeric_limits<int>::max()),
                absl::InvalidArgumentError("Input depth too large"));
    OP_REQUIRES(
        context,
        FastBoundsCheck(patch_depth_raw, std::numeric_limits<int>::max()),
        absl::InvalidArgumentError("Patch depth too large"));
    const int in_depth = static_cast<int>(in_depth_raw);
    const int patch_depth = static_cast<int>(patch_depth_raw);
    OP_REQUIRES(
        context, patch_depth > 0,
        absl::InvalidArgumentError(absl::StrCat(
            "filter depth must be stricly positive, got ", patch_depth)));
    OP_REQUIRES(context, in_depth == patch_depth,
                absl::InvalidArgumentError(absl::StrCat(
                    "Input depth must be equal to filter depth: ", in_depth,
                    " vs ", patch_depth)));

    const int out_depth =
        static_cast<int>(GetFilterDim(filter, filter_format_, 'O'));

    std::vector<int64_t> input_dims_raw(spatial_dims);
    std::vector<int> input_dims(spatial_dims);
    std::vector<int> filter_dims(spatial_dims);
    for (int i = 0; i < spatial_dims; ++i) {
      input_dims_raw[i] =
          GetTensorDim(input_flat, data_format_, static_cast<char>(i + '0'));
      OP_REQUIRES(
          context,
          FastBoundsCheck(input_dims_raw[i], std::numeric_limits<int>::max()),
          absl::InvalidArgumentError(
              absl::StrCat("Input spatial dimension ", i, " too large")));
      input_dims[i] = static_cast<int>(input_dims_raw[i]);
      filter_dims[i] = static_cast<int>(
          GetFilterDim(filter, filter_format_, static_cast<char>(i + '0')));
    }
    // The first dimension for input is batch.
    const int64_t batch_raw = GetTensorDim(input_flat, data_format_, 'N');
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                absl::InvalidArgumentError("Batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // Take the stride and dilation from the spatial dimensions only (we
    // do not support striding or dilation on the batch or depth dimension).
    std::vector<int64_t> stride_dims(spatial_dims);
    std::vector<int64_t> dilation_dims(spatial_dims);
    for (int i = 0; i < spatial_dims; ++i) {
      stride_dims[i] =
          GetTensorDim(strides_, data_format_, static_cast<char>(i + '0'));
      dilation_dims[i] =
          GetTensorDim(dilations_, data_format_, static_cast<char>(i + '0'));
    }
    std::vector<int64_t> pad_before(spatial_dims, -1);
    std::vector<int64_t> pad_after(spatial_dims, -1);
    if (padding_ == Padding::EXPLICIT) {
      GetExplicitPaddingForDim(explicit_paddings_, data_format_, 'H',
                               &pad_before[0], &pad_after[0]);
      GetExplicitPaddingForDim(explicit_paddings_, data_format_, 'W',
                               &pad_before[1], &pad_after[1]);
    }

    // Compute windowed output sizes for spatial dimensions.
    std::vector<int64_t> out_dims(spatial_dims);
    for (int i = 0; i < spatial_dims; ++i) {
      OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                  input_dims[i], filter_dims[i],
                                  dilation_dims[i], stride_dims[i], padding_,
                                  &out_dims[i], &pad_before[i], &pad_after[i]));
    }
    TensorShape out_shape;
    OP_REQUIRES_OK(context,
                   ShapeFromFormatWithStatus(data_format_, batch, out_dims,
                                             out_depth, &out_shape));

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    // If the input is empty, result can only be due to padding.
    if (input_flat.NumElements() == 0) {
      // Zero-out output and return.
      functor::SetZeroFunctor<Device, T>()(context->eigen_device<Device>(),
                                           output->template flat<T>());

      return;
    }

    launcher_(context, cudnn_use_autotune_, input_flat, filter, dilations_,
              strides_, padding_, explicit_paddings_, data_format_, output);

    // Reshape the output to preserve original batch dimensions.
    if (batch_dims_ != 1) {
      std::vector<int64_t> reshape_vect(batch_dims_);
      for (int i = 0; i < batch_dims_; ++i) {
        reshape_vect[i] = original_input_shape.dim_size(i);
      }
      for (int i = 1; i < out_shape.dims(); ++i) {
        reshape_vect.push_back(out_shape.dim_size(i));
      }
      TensorShape expanded_out_shape(reshape_vect);
      if (!output->CopyFrom(*output, expanded_out_shape)) {
        // This should never happen, since the output sizes should always be the
        // same after expanding batches.
        context->SetStatus(absl::InternalError(
            absl::StrCat("Could not expand dimension with flat output shape ",
                         out_shape.DebugString(), " and expanded output shape ",
                         expanded_out_shape.DebugString())));
      }
    }
  }

 private:
  std::vector<int64_t> strides_;
  Padding padding_;
  std::vector<int64_t> explicit_paddings_;
  TensorFormat data_format_;
  FilterTensorFormat filter_format_;
  std::vector<int64_t> dilations_;
  int batch_dims_;
  int groups_;
  bool cudnn_use_autotune_;

  LaunchConvOp<Device, T> launcher_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConvOp);
};

template <typename T>
struct LaunchConv2DOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_dilation,
                  int col_dilation, int row_stride, int col_stride,
                  const Padding& padding,
                  const std::vector<int64_t>& explicit_paddings, Tensor* output,
                  TensorFormat data_format) {
    if (data_format != FORMAT_NHWC) {
      ctx->SetStatus(errors::Unimplemented(
          "The Conv2D op currently only supports the NHWC tensor format on the "
          "CPU. The op was given the format: ",
          ToString(data_format)));
      return;
    }

    for (int64_t explicit_padding : explicit_paddings) {
      if (!FastBoundsCheck(explicit_padding, std::numeric_limits<int>::max())) {
        ctx->SetStatus(errors::InvalidArgument("filter too large"));
        return;
      }
    }

    const int64_t in_depth = input.dim_size(3);
    const int64_t out_depth = output->dim_size(3);
    const int64_t patch_depth = filter.dim_size(2);

    if (patch_depth <= 0) {
      ctx->SetStatus(errors::InvalidArgument(
          "filter depth must be stricly positive, got ", patch_depth));
      return;
    }
    if (in_depth % patch_depth != 0) {
      ctx->SetStatus(errors::InvalidArgument(
          "input depth must be evenly divisible by filter depth: ", in_depth,
          " vs ", patch_depth));
      return;
    }
    if (filter.NumElements() <= 0) {
      ctx->SetStatus(
          errors::InvalidArgument("filter must not have zero elements "
                                  "(i.e. all dimensions must be non-zero)"));
      return;
    }

    const int64_t num_groups = in_depth / patch_depth;
    if (num_groups <= 0) {
      ctx->SetStatus(errors::InvalidArgument(
          "number of groups must be stricly positive, got ", num_groups));
      return;
    }
    if (out_depth % num_groups != 0 || out_depth < num_groups) {
      ctx->SetStatus(errors::InvalidArgument(
          "output depth must be evenly divisible by number of groups: ",
          out_depth, " vs ", num_groups));
      return;
    }

    if (in_depth != patch_depth) {
      LaunchGrouped<T>()(ctx, input, filter, row_stride, col_stride,
                         row_dilation, col_dilation, padding, explicit_paddings,
                         output, data_format);
    } else {
      LaunchGeneric<CPUDevice, T>()(ctx, input, filter, row_stride, col_stride,
                                    row_dilation, col_dilation, padding,
                                    explicit_paddings, output, data_format);
    }
  }
};
extern template struct LaunchConv2DOp<CPUDevice, Eigen::bfloat16>;
extern template struct LaunchConv2DOp<CPUDevice, Eigen::half>;
extern template struct LaunchConv2DOp<CPUDevice, float>;
extern template struct LaunchConv2DOp<CPUDevice, double>;
extern template struct LaunchConv2DOp<CPUDevice, int32>;

template <typename Device, typename T>
class LaunchDeepConvOp {
 public:
  static bool Run(OpKernelContext* ctx, const Tensor& input,
                  const Tensor& filter, int batch, int input_rows,
                  int input_cols, int in_depth, int filter_rows,
                  int filter_cols, int pad_rows, int pad_cols, int out_rows,
                  int /*out_cols*/, int /*out_depth*/, int /*dilation_rows*/,
                  int /*dilation_cols*/, int /*stride_rows*/,
                  int /*stride_cols*/, Tensor* /*output*/,
                  TensorFormat /*data_format*/) {
    return false;
  }
};

template <typename Device, typename T>
class Conv2DOp : public BinaryOp<T> {
 public:
  explicit Conv2DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));

    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));

    TensorShape out_shape;
    OP_REQUIRES_OK(
        context, ShapeFromFormatWithStatus(
                     params_.data_format, dimensions.batch, dimensions.out_rows,
                     dimensions.out_cols, dimensions.out_depth, &out_shape));

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "Conv2D: in_depth = " << dimensions.in_depth
            << ", patch_depth = " << dimensions.patch_depth
            << ", input_cols = " << dimensions.input_cols
            << ", filter_cols = " << dimensions.filter_cols
            << ", input_rows = " << dimensions.input_rows
            << ", filter_rows = " << dimensions.filter_rows
            << ", stride_rows = " << dimensions.stride_rows
            << ", stride_cols = " << dimensions.stride_cols
            << ", dilation_rows = " << dimensions.dilation_rows
            << ", dilation_cols = " << dimensions.dilation_cols
            << ", out_depth = " << dimensions.out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    // If the input is empty, result can only be due to padding.
    if (input.NumElements() == 0) {
      // Zero-out output and return.
      functor::SetZeroFunctor<Device, T>()(context->eigen_device<Device>(),
                                           output->template flat<T>());

      return;
    }

    if (params_.padding != EXPLICIT &&
        LaunchDeepConvOp<Device, T>::Run(
            context, input, filter, dimensions.batch, dimensions.input_rows,
            dimensions.input_cols, dimensions.in_depth, dimensions.filter_rows,
            dimensions.filter_cols, dimensions.pad_rows_before,
            dimensions.pad_cols_before, dimensions.out_rows,
            dimensions.out_cols, dimensions.out_depth, dimensions.dilation_rows,
            dimensions.dilation_cols, dimensions.stride_rows,
            dimensions.stride_cols, output, params_.data_format)) {
      return;
    }

    launcher_(context, use_cudnn_, cudnn_use_autotune_, input, filter,
              dimensions.dilation_rows, dimensions.dilation_cols,
              dimensions.stride_rows, dimensions.stride_cols, params_.padding,
              params_.explicit_paddings, output, params_.data_format);
  }

 private:
  Conv2DParameters params_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  LaunchConv2DOp<Device, T> launcher_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DOp);
};
extern template struct Conv2DOp<CPUDevice, Eigen::bfloat16>;
extern template struct Conv2DOp<CPUDevice, Eigen::half>;
extern template struct Conv2DOp<CPUDevice, float>;
extern template struct Conv2DOp<CPUDevice, double>;
extern template struct Conv2DOp<CPUDevice, int32>;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
void LaunchConv2DOpImpl(OpKernelContext* ctx, bool use_cudnn,
                        bool cudnn_use_autotune, const Tensor& input_param,
                        const Tensor& filter, int row_dilation,
                        int col_dilation, int row_stride, int col_stride,
                        const Padding& padding,
                        const std::vector<int64_t>& explicit_paddings,
                        Tensor* output, TensorFormat data_format) {
  using se::dnn::AlgorithmConfig;
  using se::dnn::AlgorithmDesc;
  using se::dnn::ProfileResult;
  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  if (!use_cudnn) {
    ctx->SetStatus(
        errors::Unimplemented("Conv2D for GPU is not currently supported "
                              "without cudnn"));
    return;
  }

  Tensor input = input_param;
  const int64_t in_batch = GetTensorDim(input, data_format, 'N');
  int64_t in_rows = GetTensorDim(input, data_format, 'H');
  int64_t in_cols = GetTensorDim(input, data_format, 'W');
  const int64_t in_depths = GetTensorDim(input, data_format, 'C');
  const int64_t patch_rows = filter.dim_size(0);
  const int64_t patch_cols = filter.dim_size(1);
  const int64_t patch_depths = filter.dim_size(2);

  OP_REQUIRES(
      ctx, filter.NumElements() > 0,
      errors::InvalidArgument("filter must not have zero elements "
                              "(i.e. all dimensions must be non-zero)"));

  // If the filter in-depth (patch_depths) is 1 and smaller than the input
  // depth, it's a depthwise convolution. More generally, if the filter in-depth
  // divides but is smaller than the input depth, it is a grouped convolution.
  bool is_grouped_convolution = patch_depths != in_depths;
  if (patch_rows == 1 && patch_cols == 1 && !is_grouped_convolution &&
      row_dilation == 1 && col_dilation == 1 && row_stride == 1 &&
      col_stride == 1 && data_format == FORMAT_NHWC &&
      (padding == VALID || padding == SAME)) {
    // 1x1 filter, so call cublas directly.
    const uint64 m = in_batch * in_rows * in_cols;
    const uint64 k = patch_depths;
    const uint64 n = filter.dim_size(3);

    auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                input.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                filter.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                output->template flat<T>().size());

    auto no_transpose = se::blas::Transpose::kNoTranspose;
    OP_REQUIRES_OK(
        ctx, stream->ThenBlasGemm(no_transpose, no_transpose, n, m, k, b_ptr, n,
                                  a_ptr, k, &c_ptr, n, GetNumericOptions()));
    return;
  } else if (patch_rows == in_rows && patch_cols == in_cols &&
             !is_grouped_convolution && row_dilation == 1 &&
             col_dilation == 1 && padding == VALID &&
             data_format == FORMAT_NHWC) {
    // The input data and filter have the same height/width, so call cublas
    // directly.
    const uint64 m = in_batch;
    const uint64 k = patch_rows * patch_cols * patch_depths;
    const uint64 n = filter.dim_size(3);

    auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                input.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                filter.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                output->template flat<T>().size());

    auto no_transpose = se::blas::Transpose::kNoTranspose;
    OP_REQUIRES_OK(
        ctx, stream->ThenBlasGemm(no_transpose, no_transpose, n, m, k, b_ptr, n,
                                  a_ptr, k, &c_ptr, n, GetNumericOptions()));
    return;
  }

  const bool compute_in_nhwc =
      ComputeInNhwcEnabled(DataTypeToEnum<T>::value, stream);
  // fast NHWC implementation is a CUDA only feature

  // We only do one directional conversion: NHWC->NCHW. We never convert in the
  // other direction. Grappler layout optimizer selects preferred layout and
  // adds necessary annotations to the graph.
  // TODO(ezhulenev): Convert in other direction for fp16?
  const TensorFormat compute_data_format =
      (compute_in_nhwc && data_format == FORMAT_NHWC) ? FORMAT_NHWC
                                                      : FORMAT_NCHW;

  VLOG(3) << "Compute Conv2D with cuDNN:"
          << " data_format=" << ToString(data_format)
          << " compute_data_format=" << ToString(compute_data_format);

  const int64_t out_batch = GetTensorDim(*output, data_format, 'N');
  const int64_t out_rows = GetTensorDim(*output, data_format, 'H');
  const int64_t out_cols = GetTensorDim(*output, data_format, 'W');
  const int64_t out_depths = GetTensorDim(*output, data_format, 'C');
  int64_t padding_top = -1, padding_bottom = -1;
  int64_t padding_left = -1, padding_right = -1;
  if (padding == EXPLICIT) {
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'H', &padding_top,
                             &padding_bottom);
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'W', &padding_left,
                             &padding_right);
  }
  int64_t out_rows_check, out_cols_check;
  Status status = GetWindowedOutputSizeVerbose(
      in_rows, patch_rows, row_dilation, row_stride, padding, &out_rows_check,
      &padding_top, &padding_bottom);
  // The status is guaranteed to be OK because we checked the output and padding
  // was valid earlier.
  TF_CHECK_OK(status);
  DCHECK_EQ(out_rows, out_rows_check);
  status = GetWindowedOutputSizeVerbose(in_cols, patch_cols, col_dilation,
                                        col_stride, padding, &out_cols_check,
                                        &padding_left, &padding_right);
  TF_CHECK_OK(status);
  DCHECK_EQ(out_cols, out_cols_check);

  const int64_t common_padding_rows = std::min(padding_top, padding_bottom);
  const int64_t common_padding_cols = std::min(padding_left, padding_right);
  if (padding_top != padding_bottom || padding_left != padding_right) {
    // cuDNN only supports padding the same amount on the left and right sides,
    // and on the top and bottom sides. So we manually create a new padded
    // input tensor such that we can pass it to cuDNN.
    VLOG(4) << "Pad input tensor:"
            << " padding_top=" << padding_top
            << " padding_bottom=" << padding_bottom
            << " padding_left=" << padding_left
            << " padding_right=" << padding_right;

    // TODO(reedwm): In some cases, we can avoid an allocation even if the two
    // padding sides are different. For example, if the input is 2x2, the filter
    // is 1x1, the stride is 2, and the padding is (1, 0, 1, 0), the result is
    // equivalent to as if the padding is (1, 1, 1, 1). Changing the padding in
    // such a way would allow us to avoid the allocation.
    Tensor transformed_input;
    const int64_t padding_rows_diff = std::abs(padding_bottom - padding_top);
    const int64_t padding_cols_diff = std::abs(padding_right - padding_left);
    const int64_t new_in_rows = in_rows + padding_rows_diff;
    const int64_t new_in_cols = in_cols + padding_cols_diff;
    TensorShape transformed_input_shape;
    OP_REQUIRES_OK(ctx, ShapeFromFormatWithStatus(
                            data_format, in_batch, new_in_rows, new_in_cols,
                            in_depths, &transformed_input_shape));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                transformed_input_shape, &transformed_input));

    const int64_t input_pad_top = padding_top - common_padding_rows;
    const int64_t input_pad_bottom = padding_bottom - common_padding_rows;
    const int64_t input_pad_left = padding_left - common_padding_cols;
    const int64_t input_pad_right = padding_right - common_padding_cols;
    bool in_bounds =
        FastBoundsCheck(input_pad_top, std::numeric_limits<int>::max()) &&
        FastBoundsCheck(input_pad_bottom, std::numeric_limits<int>::max()) &&
        FastBoundsCheck(input_pad_left, std::numeric_limits<int>::max()) &&
        FastBoundsCheck(input_pad_right, std::numeric_limits<int>::max());
    if (!in_bounds) {
      ctx->SetStatus(errors::InvalidArgument("Padding is too large."));
      return;
    }
    functor::PadInput<GPUDevice, T, int, 4>()(
        ctx->eigen_device<GPUDevice>(),
        To32Bit(static_cast<const Tensor&>(input).tensor<T, 4>()),
        {{static_cast<int>(input_pad_top), static_cast<int>(input_pad_left)}},
        {{static_cast<int>(input_pad_bottom),
          static_cast<int>(input_pad_right)}},
        To32Bit(transformed_input.tensor<T, 4>()), data_format, T{});

    input = transformed_input;
    in_rows = new_in_rows;
    in_cols = new_in_cols;
  }

  if (data_format == FORMAT_NHWC && compute_data_format == FORMAT_NCHW) {
    VLOG(4) << "Convert the input tensor from NHWC to NCHW.";

    TensorShape nchw_shape;
    OP_REQUIRES_OK(
        ctx, ShapeFromFormatWithStatus(FORMAT_NCHW, in_batch, in_rows, in_cols,
                                       in_depths, &nchw_shape));
    if (in_depths > 1) {
      Tensor transformed_input;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             nchw_shape, &transformed_input));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(input).tensor<T, 4>(),
          transformed_input.tensor<T, 4>());
      input = transformed_input;
    } else {
      // If depth <= 1, then just reshape.
      DCHECK(input.CopyFrom(input, nchw_shape));
    }
  } else {
    DCHECK(data_format == compute_data_format)  // Crash OK
        << "Illegal data and compute format pair:"
        << " data_format=" << ToString(data_format)
        << " compute_data_format=" << ToString(compute_data_format);
  }

  DCHECK(common_padding_rows >= 0 && common_padding_cols >= 0)  // Crash OK
      << "Negative row or col paddings: (" << common_padding_rows << ", "
      << common_padding_cols << ")";

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
  input_desc.set_count(in_batch)
      .set_feature_map_count(in_depths)
      .set_height(in_rows)
      .set_width(in_cols)
      .set_layout(compute_data_layout);
  se::dnn::BatchDescriptor output_desc;
  output_desc.set_count(out_batch)
      .set_height(out_rows)
      .set_width(out_cols)
      .set_feature_map_count(out_depths)
      .set_layout(compute_data_layout);
  se::dnn::FilterDescriptor filter_desc;
  filter_desc.set_input_filter_height(patch_rows)
      .set_input_filter_width(patch_cols)
      .set_input_feature_map_count(patch_depths)
      .set_output_feature_map_count(filter.dim_size(3))
      .set_layout(filter_layout);
  se::dnn::ConvolutionDescriptor conv_desc;
  conv_desc.set_vertical_dilation_rate(row_dilation)
      .set_horizontal_dilation_rate(col_dilation)
      .set_vertical_filter_stride(row_stride)
      .set_horizontal_filter_stride(col_stride)
      .set_zero_padding_height(common_padding_rows)
      .set_zero_padding_width(common_padding_cols)
      .set_group_count(in_depths / patch_depths);

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

  Tensor transformed_output;
  if (data_format != compute_data_format) {
    VLOG(4) << "Allocate temporary memory for output in compute data format";
    TensorShape transformed_output_shape;
    OP_REQUIRES_OK(ctx, ShapeFromFormatWithStatus(
                            compute_data_format, out_batch, out_rows, out_cols,
                            out_depths, &transformed_output_shape));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                transformed_output_shape, &transformed_output));
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

  ConvParameters conv_parameters = {
      stream->parent(),
      in_batch,                 // batch
      in_depths,                // in_depths
      {{in_rows,                // in_rows
        in_cols}},              // in_cols
      compute_data_format,      // compute_data_format
      out_depths,               // out_depths
      {{patch_rows,             // filter_rows
        patch_cols,             // filter_cols
        patch_depths}},         // filter_depths
      {{row_dilation,           // dilation_rows
        col_dilation}},         // dilation_cols
      {{row_stride,             // stride_rows
        col_stride}},           // stride_cols
      {{common_padding_rows,    // padding_rows
        common_padding_cols}},  // padding_cols
      input.dtype(),            // tensor datatype
      conv_desc.group_count(),
  };

  auto entry_or = AutotuneUnfusedConv(
      cudnn_use_autotune, ConvAutotuneMap::GetInstance(), conv_parameters, ctx,
      se::dnn::ConvolutionKind::FORWARD, input_desc, input_ptr, filter_desc,
      filter_ptr, conv_desc, output_desc, output_ptr, ConvolveScratchSize);
  OP_REQUIRES_OK(ctx, entry_or.status());
  auto autotune_entry = std::move(entry_or).value();

  DnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
  Status cudnn_launch_status = LaunchAutotunedConv(
      autotune_entry, &scratch_allocator, se::dnn::ConvolutionKind::FORWARD,
      stream, input_desc, input_ptr, filter_desc, filter_ptr, conv_desc,
      output_desc, output_ptr);
  if (!cudnn_launch_status.ok()) {
    ctx->SetStatus(cudnn_launch_status);
    return;
  }

  if (data_format == FORMAT_NHWC && compute_data_format == FORMAT_NCHW) {
    VLOG(4) << "Convert the output tensor back from NCHW to NHWC.";
    functor::NCHWToNHWC<GPUDevice, T, 4>()(
        ctx->eigen_device<GPUDevice>(),
        const_cast<const Tensor&>(transformed_output).tensor<T, 4>(),
        output->tensor<T, 4>());
  }
}

template <typename T>
void LaunchConv2DOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& input_param, const Tensor& filter, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64_t>& explicit_paddings, Tensor* output,
    TensorFormat data_format) {
  LaunchConv2DOpImpl<T>(ctx, use_cudnn, cudnn_use_autotune, input_param, filter,
                        row_dilation, col_dilation, row_stride, col_stride,
                        padding, explicit_paddings, output, data_format);
}

// To be used inside depthwise_conv_op.cc.
extern template struct LaunchConv2DOp<GPUDevice, float>;
// extern template struct LaunchConv2DOp<GPUDevice, Eigen::bfloat16>;
extern template struct LaunchConv2DOp<GPUDevice, Eigen::half>;
extern template struct LaunchConv2DOp<GPUDevice, double>;

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_IMPL_H_
