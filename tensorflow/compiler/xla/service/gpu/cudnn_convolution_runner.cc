/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_convolution_runner.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {
namespace {

using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::Stream;
using se::dnn::AlgorithmConfig;
using se::dnn::BatchDescriptor;
using se::dnn::ConvolutionDescriptor;
using se::dnn::DataLayout;
using se::dnn::DimIndex;
using se::dnn::FilterDescriptor;
using se::dnn::FilterLayout;
using se::dnn::ProfileResult;

struct CudnnConvParams {
  // Here are the fields related to cuDNN's fused convolution. The result thus
  // is defined as:
  //   activation(conv_result_scale * conv(x, w) +
  //       side_input_scale * side_input + broadcast(bias))
  //
  // The most common fused conv is conv forward + relu/identity, for example.
  //
  // bias_buf is a single-dimensional array, with the length equal to the number
  // of output features. It'll be broadcasted to the output shape in order to be
  // added to the final results.
  //
  // side_input_buf, if valid, must have the same shape as the output buffer.
  struct FusionParams {
    se::dnn::ActivationMode mode;
    double side_input_scale;
    se::DeviceMemoryBase bias_buf;
    se::DeviceMemoryBase side_input_buf;  // nullable
  };

  CudnnConvKind kind;
  const Shape* input_shape;
  const Shape* filter_shape;
  const Shape* output_shape;
  se::DeviceMemoryBase input_buf;
  se::DeviceMemoryBase filter_buf;
  se::DeviceMemoryBase output_buf;
  const Window* window;
  const ConvolutionDimensionNumbers* dnums;
  int64 feature_group_count;
  se::dnn::AlgorithmConfig algorithm;
  double conv_result_scale;

  absl::optional<FusionParams> fusion;
};

// A StreamExecutor ScratchAllocator that wraps a single XLA allocation,
// returning it (in its entirety) the first time Allocate() is called.
class ScratchBufAllocator : public se::ScratchAllocator {
 public:
  explicit ScratchBufAllocator(se::DeviceMemoryBase scratch)
      : scratch_(scratch) {}

  ~ScratchBufAllocator() override = default;

  int64 GetMemoryLimitInBytes(se::Stream* /*stream*/) override {
    return scratch_.size();
  }

  se::port::StatusOr<DeviceMemory<uint8>> AllocateBytes(
      se::Stream* stream, int64 byte_size) override {
    if (allocated_) {
      return se::port::InternalError(
          "Can't allocate twice from a ScratchBufAllocator.");
    }
    if (byte_size > scratch_.size()) {
      return se::port::InternalError(absl::StrCat(
          "Can't allocate ", byte_size,
          " bytes from a ScratchBufAllocator of size ", scratch_.size()));
    }

    allocated_ = true;
    return se::DeviceMemory<uint8>(scratch_);
  }

 private:
  se::DeviceMemoryBase scratch_;
  bool allocated_ = false;
};

template <typename T>
Status RunCudnnConvolutionImpl(CudnnConvParams params,
                               se::ScratchAllocator* scratch_allocator,
                               se::Stream* stream,
                               se::dnn::ProfileResult* profile_result) {
  CudnnConvKind kind = params.kind;
  const Shape& input_shape = *params.input_shape;
  const Shape& filter_shape = *params.filter_shape;
  const Shape& output_shape = *params.output_shape;
  DeviceMemory<T> input_buf(params.input_buf);
  DeviceMemory<T> filter_buf(params.filter_buf);
  DeviceMemory<T> output_buf(params.output_buf);
  const Window& window = *params.window;
  const ConvolutionDimensionNumbers& dnums = *params.dnums;
  int64 feature_group_count = params.feature_group_count;
  AlgorithmConfig algorithm = params.algorithm;

  VLOG(3) << "Convolution Algorithm: " << algorithm.algorithm().algo_id();
  VLOG(3) << "tensor_ops_enabled: "
          << algorithm.algorithm().tensor_ops_enabled();
  VLOG(3) << "Convolution kind: " << CudnnConvKindToString(kind);
  VLOG(3) << "input shape: " << ShapeUtil::HumanStringWithLayout(input_shape);
  VLOG(3) << "filter shape: " << ShapeUtil::HumanStringWithLayout(filter_shape);
  VLOG(3) << "Output shape: " << ShapeUtil::HumanStringWithLayout(output_shape);
  VLOG(3) << "Window: { " << window.ShortDebugString() << " }";
  VLOG(3) << "Dim nums: { " << dnums.ShortDebugString() << " }";

  const int num_dimensions = window.dimensions_size();
  CHECK_LE(num_dimensions, 3);
  // cuDNN does not support 1D convolutions. We therefore express 1D
  // convolutions as 2D convolutions where the first spatial dimension is 1.
  // This matches the behavior of TF (see definition of conv1d in
  // tensorflow/python/ops/nn_ops.py).
  const int effective_num_dimensions = std::max(2, num_dimensions);

  CHECK_EQ(primitive_util::NativeToPrimitiveType<T>(),
           output_shape.element_type())
      << ShapeUtil::HumanString(output_shape);

  CHECK_EQ(num_dimensions, dnums.input_spatial_dimensions_size());
  CHECK_EQ(num_dimensions, dnums.kernel_spatial_dimensions_size());
  CHECK_EQ(num_dimensions, dnums.output_spatial_dimensions_size());
  for (const WindowDimension& dim : window.dimensions()) {
    CHECK_EQ(dim.padding_low(), dim.padding_high());
  }

  // cuDNN's convolution APIs support the BDYX layout for activations/output and
  // the OIYX layout for weights.
  DataLayout input_dl;
  FilterLayout filter_dl;
  DataLayout output_dl;

  TF_ASSIGN_OR_RETURN(std::tie(input_dl, filter_dl, output_dl),
                      XlaConvLayoutsToStreamExecutorLayouts(
                          dnums, input_shape.layout(), filter_shape.layout(),
                          output_shape.layout()));

  BatchDescriptor input_descriptor(effective_num_dimensions);
  input_descriptor.set_layout(input_dl)
      .set_feature_map_count(
          input_shape.dimensions(dnums.input_feature_dimension()))
      .set_count(input_shape.dimensions(dnums.input_batch_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    // Note that the dimensions are reversed. The same holds below.
    input_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        input_shape.dimensions(dnums.input_spatial_dimensions(dim)));
  }

  FilterDescriptor filter_descriptor(effective_num_dimensions);
  filter_descriptor.set_layout(filter_dl)
      .set_input_feature_map_count(
          filter_shape.dimensions(dnums.kernel_input_feature_dimension()))
      .set_output_feature_map_count(
          filter_shape.dimensions(dnums.kernel_output_feature_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    filter_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        filter_shape.dimensions(dnums.kernel_spatial_dimensions(dim)));
  }

  ConvolutionDescriptor convolution_descriptor(effective_num_dimensions);
  convolution_descriptor.set_group_count(feature_group_count);
  for (int dim = 0; dim < num_dimensions; ++dim) {
    convolution_descriptor
        .set_zero_padding(
            static_cast<DimIndex>(effective_num_dimensions - dim - 1),
            window.dimensions(dim).padding_low())
        .set_filter_stride(
            static_cast<DimIndex>(effective_num_dimensions - dim - 1),
            window.dimensions(dim).stride());
  }

  BatchDescriptor output_descriptor(effective_num_dimensions);
  output_descriptor.set_layout(output_dl)
      .set_feature_map_count(
          output_shape.dimensions(dnums.output_feature_dimension()))
      .set_count(output_shape.dimensions(dnums.output_batch_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    output_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        output_shape.dimensions(dnums.output_spatial_dimensions(dim)));
  }

  // Add a singleton dimension in the 1D convolution case.
  if (num_dimensions == 1) {
    input_descriptor.set_spatial_dim(static_cast<DimIndex>(0), 1);
    output_descriptor.set_spatial_dim(static_cast<DimIndex>(0), 1);
    filter_descriptor.set_spatial_dim(static_cast<DimIndex>(0), 1);
    convolution_descriptor.set_zero_padding(static_cast<DimIndex>(0), 0)
        .set_filter_stride(static_cast<DimIndex>(0), 1);
  }

  switch (kind) {
    case CudnnConvKind::kForward:
      if (params.conv_result_scale != 1) {
        return InternalError(
            "StreamExecutor doesn't support scaled convolution: %lf.",
            params.conv_result_scale);
      }
      stream->ThenConvolveWithAlgorithm(
          input_descriptor, input_buf, filter_descriptor, filter_buf,
          convolution_descriptor, output_descriptor, &output_buf,
          scratch_allocator, algorithm, profile_result);
      break;
    case CudnnConvKind::kBackwardInput:
      if (params.conv_result_scale != 1) {
        return InternalError(
            "StreamExecutor doesn't support scaled convolution: %lf.",
            params.conv_result_scale);
      }
      stream->ThenConvolveBackwardDataWithAlgorithm(
          filter_descriptor, filter_buf, output_descriptor, output_buf,
          convolution_descriptor, input_descriptor, &input_buf,
          scratch_allocator, algorithm, profile_result);
      break;
    case CudnnConvKind::kBackwardFilter:
      if (params.conv_result_scale != 1) {
        return InternalError(
            "StreamExecutor doesn't support scaled convolution: %lf.",
            params.conv_result_scale);
      }
      stream->ThenConvolveBackwardFilterWithAlgorithm(
          input_descriptor, input_buf, output_descriptor, output_buf,
          convolution_descriptor, filter_descriptor, &filter_buf,
          scratch_allocator, algorithm, profile_result);
      break;
    case CudnnConvKind::kForwardActivation: {
      BatchDescriptor bias_desc;
      bias_desc.set_count(1)
          .set_height(1)
          .set_width(1)
          .set_feature_map_count(
              output_shape.dimensions(dnums.output_feature_dimension()))
          .set_layout(output_dl);

      se::DeviceMemory<T> side_input(params.fusion->side_input_buf);
      // If there is no side input, use output as the side input.
      if (side_input.is_null()) {
        if (params.fusion->side_input_scale != 0) {
          return InternalError(
              "Side input scale is not 0, yet no side input buffer is "
              "provided");
        }
        // Since side-input scale is 0, the values in the side input don't
        // matter.  The simplest thing to do would be to pass in a null buffer
        // for the side input, but cudnn doesn't allow this.  cudnn does promise
        // that if side-input-scale is 0 the side input won't be read, so we
        // just pass in the output buffer, since it's handy and has the correct
        // size.
        side_input = output_buf;
      }

      stream->ThenFusedConvolveWithAlgorithm(
          input_descriptor, input_buf, params.conv_result_scale,
          filter_descriptor, filter_buf, convolution_descriptor, side_input,
          params.fusion->side_input_scale, bias_desc,
          DeviceMemory<T>(params.fusion->bias_buf), params.fusion->mode,
          output_descriptor, &output_buf, scratch_allocator, algorithm,
          profile_result);
      break;
    }
  }

  if (!stream->ok()) {
    return InternalError(
        "Unable to launch convolution with type %s and algorithm (%d, %d)",
        CudnnConvKindToString(kind), algorithm.algorithm().algo_id(),
        algorithm.algorithm_no_scratch().algo_id());
  }
  return Status::OK();
}

// Returns the cudnn convolution parameters generated from conv, which must be a
// custom-call to a cudnn convolution.
StatusOr<CudnnConvParams> GetCudnnConvParams(
    const HloCustomCallInstruction* conv,
    absl::Span<se::DeviceMemoryBase> operand_buffers,
    se::DeviceMemoryBase result_buffer) {
  CudnnConvParams params;

  TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig backend_config,
                      conv->backend_config<CudnnConvBackendConfig>());
  const auto& target = conv->custom_call_target();
  const auto& lhs_shape = conv->operand(0)->shape();
  const auto& rhs_shape = conv->operand(1)->shape();
  const auto& conv_result_shape = conv->shape().tuple_shapes(0);

  params.window = &conv->window();
  params.dnums = &conv->convolution_dimension_numbers();
  params.feature_group_count = conv->feature_group_count();
  params.algorithm = se::dnn::AlgorithmConfig(se::dnn::AlgorithmDesc(
      backend_config.algorithm(), backend_config.tensor_ops_enabled(),
      backend_config.scratch_size()));
  params.conv_result_scale = backend_config.conv_result_scale();

  if (target == kCudnnConvForwardCallTarget) {
    params.kind = CudnnConvKind::kForward;
    params.input_shape = &lhs_shape;
    params.filter_shape = &rhs_shape;
    params.output_shape = &conv_result_shape;
    params.input_buf = operand_buffers[0];
    params.filter_buf = operand_buffers[1];
    params.output_buf = result_buffer;
  } else if (target == kCudnnConvBackwardInputCallTarget) {
    params.kind = CudnnConvKind::kBackwardInput;
    params.input_shape = &conv_result_shape;
    params.filter_shape = &rhs_shape;
    params.output_shape = &lhs_shape;
    params.input_buf = result_buffer;
    params.filter_buf = operand_buffers[1];
    params.output_buf = operand_buffers[0];
  } else if (target == kCudnnConvBackwardFilterCallTarget) {
    params.kind = CudnnConvKind::kBackwardFilter;
    params.input_shape = &lhs_shape;
    params.filter_shape = &conv_result_shape;
    params.output_shape = &rhs_shape;
    params.input_buf = operand_buffers[0];
    params.filter_buf = result_buffer;
    params.output_buf = operand_buffers[1];
  } else if (target == kCudnnConvBiasActivationForwardCallTarget) {
    params.kind = CudnnConvKind::kForwardActivation;
    params.input_shape = &lhs_shape;
    params.filter_shape = &rhs_shape;
    params.output_shape = &conv_result_shape;
    params.fusion.emplace();
    auto& fusion = *params.fusion;
    if (backend_config.activation_mode() <
        static_cast<int64>(se::dnn::ActivationMode::kNumActivationModes)) {
      fusion.mode = static_cast<se::dnn::ActivationMode>(
          backend_config.activation_mode());
    } else {
      return InternalError("Bad activation mode: %s",
                           backend_config.ShortDebugString());
    }
    fusion.side_input_scale = backend_config.side_input_scale();
    params.input_buf = operand_buffers[0];
    params.filter_buf = operand_buffers[1];
    params.output_buf = result_buffer;
    params.fusion->bias_buf = operand_buffers[2];
    if (operand_buffers.size() >= 4) {
      params.fusion->side_input_buf = operand_buffers[3];
    }
  } else {
    return InternalError("Unexpected custom call target: %s", target);
  }
  return params;
}

}  // anonymous namespace

Status RunCudnnConvolution(const HloCustomCallInstruction* conv,
                           absl::Span<se::DeviceMemoryBase> operand_buffers,
                           se::DeviceMemoryBase result_buffer,
                           se::DeviceMemoryBase scratch_buf, se::Stream* stream,
                           se::dnn::ProfileResult* profile_result) {
  ScratchBufAllocator scratch_allocator(scratch_buf);
  return RunCudnnConvolution(conv, operand_buffers, result_buffer,
                             &scratch_allocator, stream, profile_result);
}

Status RunCudnnConvolution(const HloCustomCallInstruction* conv,
                           absl::Span<se::DeviceMemoryBase> operand_buffers,
                           se::DeviceMemoryBase result_buffer,
                           se::ScratchAllocator* scratch_allocator,
                           se::Stream* stream,
                           se::dnn::ProfileResult* profile_result) {
  TF_ASSIGN_OR_RETURN(CudnnConvParams params,
                      GetCudnnConvParams(conv, operand_buffers, result_buffer));

  PrimitiveType output_primitive_type =
      conv->shape().tuple_shapes(0).element_type();
  switch (output_primitive_type) {
    case F16:
      return RunCudnnConvolutionImpl<Eigen::half>(params, scratch_allocator,
                                                  stream, profile_result);
    case F32:
      return RunCudnnConvolutionImpl<float>(params, scratch_allocator, stream,
                                            profile_result);
    case F64:
      return RunCudnnConvolutionImpl<double>(params, scratch_allocator, stream,
                                             profile_result);
    default:
      LOG(FATAL) << ShapeUtil::HumanString(*params.output_shape);
  }
}

}  // namespace gpu
}  // namespace xla
