/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_conv_runner.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "xla/util.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::dnn::BatchDescriptor;
using se::dnn::ConvolutionDescriptor;
using se::dnn::DataLayout;
using se::dnn::DimIndex;
using se::dnn::FilterDescriptor;
using se::dnn::FilterLayout;

template <typename ElementType, typename OutputType>
absl::Status RunGpuConvUnfused(const GpuConvParams& params, se::Stream* stream,
                               RunConvOptions options,
                               DeviceMemory<ElementType> input_buf,
                               DeviceMemory<ElementType> filter_buf,
                               DeviceMemory<OutputType> output_buf,
                               DeviceMemoryBase scratch_memory) {
  if (params.config->conv_result_scale != 1) {
    return Internal("StreamExecutor doesn't support scaled convolution: %lf.",
                    params.config->conv_result_scale);
  }

  TF_ASSIGN_OR_RETURN(se::dnn::ConvolutionKind kind,
                      GetDNNConvKindFromCudnnConvKind(params.config->kind));

  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType input_type,
      GetDNNDataTypeFromPrimitiveType(params.config->input_type));

  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(params.config->output_type));

  se::dnn::LazyOpRunner<se::dnn::ConvOp>* lazy_runner =
      options.runner_cache->AsConvRunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::ConvOp>> local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }

  se::dnn::ConvOp::Config config{kind,
                                 input_type,
                                 output_type,
                                 params.config->input_descriptor,
                                 params.config->filter_descriptor,
                                 params.config->output_descriptor,
                                 params.config->conv_desc};
  TF_ASSIGN_OR_RETURN(auto* runner,
                      lazy_runner->GetOrCreateRunner(config, stream));

  return (*runner)(stream, options.profile_result, scratch_memory, input_buf,
                   filter_buf, output_buf);
}

template <typename ElementType, typename OutputType>
absl::Status RunGpuConvGraph(const GpuConvParams& params, se::Stream* stream,
                             RunConvOptions options,
                             DeviceMemory<ElementType> input_buf,
                             DeviceMemory<ElementType> filter_buf,
                             DeviceMemory<OutputType> output_buf,
                             DeviceMemoryBase scratch_memory) {
  if (params.config->conv_result_scale != 1) {
    return Internal("StreamExecutor doesn't support scaled convolution: %lf.",
                    params.config->conv_result_scale);
  }

  TF_ASSIGN_OR_RETURN(se::dnn::ConvolutionKind kind,
                      GetDNNConvKindFromCudnnConvKind(params.config->kind));

  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType input_type,
      GetDNNDataTypeFromPrimitiveType(params.config->input_type));

  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(params.config->output_type));

  se::dnn::LazyOpRunner<se::dnn::GraphConvOp>* lazy_runner =
      options.runner_cache->AsGraphConvRunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::GraphConvOp>> local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }

  se::dnn::GraphConvOp::Config config{kind,
                                      input_type,
                                      output_type,
                                      params.config->input_descriptor,
                                      params.config->filter_descriptor,
                                      params.config->output_descriptor,
                                      params.config->conv_desc,
                                      params.config->serialized_graph};
  TF_ASSIGN_OR_RETURN(auto* runner,
                      lazy_runner->GetOrCreateRunner(config, stream));

  std::vector<DeviceMemoryBase> operands = {input_buf, filter_buf, output_buf};
  // Insert the optional operands ahead of the output.
  operands.insert(operands.end() - 1, params.operand_bufs.begin(),
                  params.operand_bufs.end());

  // Insert any additional outputs at the end.
  operands.insert(operands.end(), params.aux_bufs.begin(),
                  params.aux_bufs.end());

  return (*runner)(stream, options.profile_result, scratch_memory, operands);
}

template <typename ElementType, typename BiasType, typename OutputType>
absl::Status RunGpuConvForwardActivation(
    const GpuConvParams& params, se::Stream* stream, RunConvOptions options,
    DeviceMemory<ElementType> input_buf, DeviceMemory<ElementType> filter_buf,
    DeviceMemory<OutputType> output_buf, DeviceMemoryBase scratch_memory) {
  se::DeviceMemory<OutputType> side_input(params.fusion->side_input_buf);
  // If there is no side input, use output as the side input.
  if (side_input.is_null()) {
    if (params.config->fusion->side_input_scale != 0) {
      return Internal(
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

  se::dnn::LazyOpRunner<se::dnn::FusedConvOp>* lazy_runner =
      options.runner_cache->AsFusedConvRunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::FusedConvOp>> local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }

  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType input_type,
      GetDNNDataTypeFromPrimitiveType(params.config->input_type));

  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(params.config->output_type));

  se::dnn::FusedConvOp::Config config{se::dnn::ConvolutionKind::FORWARD,
                                      input_type,
                                      BiasTypeForInputType(input_type),
                                      output_type,
                                      params.config->conv_result_scale,
                                      params.config->fusion->side_input_scale,
                                      params.config->fusion->leakyrelu_alpha,
                                      params.config->input_descriptor,
                                      params.config->filter_descriptor,
                                      params.config->bias_descriptor,
                                      params.config->output_descriptor,
                                      params.config->conv_desc,
                                      params.config->fusion->mode};
  TF_ASSIGN_OR_RETURN(auto* runner,
                      lazy_runner->GetOrCreateRunner(config, stream));

  return (*runner)(stream, options.profile_result, scratch_memory, input_buf,
                   filter_buf, side_input, params.fusion->bias_buf, output_buf);
}

// StreamExecutor supports various data types via overloading, and the support
// is maintained on-demand. To avoid calling into non-exist overloads, we have
// to carefully not call into them by using enable_if.
// TODO(timshen): Ideally, to avoid such complication in the runner, we can turn
// StreamExecutor overloadings to template functions, and for unsupported data
// types return runtime errors.
// This is the specialization for double, float, and half types.  All kinds of
// convolutions are supported here.
template <typename ElementType, typename BiasType, typename OutputType,
          typename std::enable_if<
              !std::is_integral<ElementType>::value>::type* = nullptr>
absl::Status RunGpuConvInternalImpl(const GpuConvParams& params,
                                    se::Stream* stream, RunConvOptions options,
                                    DeviceMemory<ElementType> input_buf,
                                    DeviceMemory<ElementType> filter_buf,
                                    DeviceMemory<OutputType> output_buf,
                                    DeviceMemoryBase scratch_memory) {
  switch (params.config->kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kBackwardInput:
    case CudnnConvKind::kBackwardFilter:
      return RunGpuConvUnfused(params, stream, options, input_buf, filter_buf,
                               output_buf, scratch_memory);
    case CudnnConvKind::kForwardActivation: {
      return RunGpuConvForwardActivation<ElementType, BiasType, OutputType>(
          params, stream, options, input_buf, filter_buf, output_buf,
          scratch_memory);
      case CudnnConvKind::kForwardGraph:
        return RunGpuConvGraph(params, stream, options, input_buf, filter_buf,
                               output_buf, scratch_memory);
    }
  }
  return absl::OkStatus();
}

// Specialization for integer types.  Only two forward convolutions are allowed.
template <typename ElementType, typename BiasType, typename OutputType,
          typename std::enable_if<std::is_integral<ElementType>::value>::type* =
              nullptr>
absl::Status RunGpuConvInternalImpl(const GpuConvParams& params,
                                    se::Stream* stream, RunConvOptions options,
                                    DeviceMemory<ElementType> input_buf,
                                    DeviceMemory<ElementType> filter_buf,
                                    DeviceMemory<OutputType> output_buf,
                                    DeviceMemoryBase scratch_memory) {
  switch (params.config->kind) {
    case CudnnConvKind::kForward:
      return RunGpuConvUnfused(params, stream, options, input_buf, filter_buf,
                               output_buf, scratch_memory);
    case CudnnConvKind::kForwardActivation:
      return RunGpuConvForwardActivation<ElementType, BiasType, OutputType>(
          params, stream, options, input_buf, filter_buf, output_buf,
          scratch_memory);
    default:
      return Internal(
          "Only convolution kinds kForward and kForwardActivation are "
          "supported for integer types");
  }
  return absl::OkStatus();
}

template <typename ElementType, typename BiasType, typename OutputType>
absl::Status RunGpuConvImpl(const GpuConvParams& params, se::Stream* stream,
                            se::DeviceMemoryBase scratch_memory,
                            RunConvOptions options) {
  auto input_buf = se::DeviceMemory<ElementType>(params.input_buf);
  auto filter_buf = se::DeviceMemory<ElementType>(params.filter_buf);
  auto output_buf = se::DeviceMemory<OutputType>(params.output_buf);

  absl::Status run_status =
      RunGpuConvInternalImpl<ElementType, BiasType, OutputType>(
          params, stream, options, input_buf, filter_buf, output_buf,
          scratch_memory);

  if (!run_status.ok()) {
    return run_status;
  }

  if (!stream->ok()) {
    se::dnn::AlgorithmDesc algorithm = params.config->algorithm;
    if (options.runner_cache) {
      algorithm = options.runner_cache->ToAlgorithmDesc();
    }
    return Internal(
        "Unable to launch convolution with type %s and algorithm %s",
        CudnnConvKindToString(params.config->kind), algorithm.ToString());
  }
  return absl::OkStatus();
}

int64_t GetVectCSize(DataLayout layout) {
  switch (layout) {
    case DataLayout::kBatchDepthYX4:
      return 4;
    case DataLayout::kBatchDepthYX32:
      return 32;
    default:
      return 1;
  }
}

int64_t GetVectCSize(FilterLayout layout) {
  switch (layout) {
    case FilterLayout::kOutputInputYX4:
      return 4;
    case FilterLayout::kOutputInputYX32:
    case FilterLayout::kOutputInputYX32_CudnnReordered:
      return 32;
    default:
      return 1;
  }
}

}  // anonymous namespace

absl::StatusOr<GpuConvConfig> GetGpuConvConfig(
    const GpuConvDescriptor& desc, const absl::string_view inst_as_string) {
  GpuConvConfig config;

  const Shape& operand0_shape = desc.operand0_shape;
  const Shape& operand1_shape = desc.operand1_shape;
  const Shape& result_shape = desc.result_shape;
  const CudnnConvBackendConfig& backend_config = desc.backend_config;

  config.input_type = operand0_shape.element_type();
  config.output_type = result_shape.element_type();
  config.kind = desc.kind;
  config.algorithm = se::dnn::AlgorithmDesc(backend_config.algorithm());
  config.conv_result_scale = backend_config.conv_result_scale();
  config.serialized_graph = backend_config.serialized_graph();

  switch (config.kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
    case CudnnConvKind::kForwardGraph:
      config.input_shape = operand0_shape;
      config.filter_shape = operand1_shape;
      config.output_shape = result_shape;
      break;
    case CudnnConvKind::kBackwardInput:
      config.input_shape = result_shape;
      config.filter_shape = operand1_shape;
      config.output_shape = operand0_shape;
      break;
    case CudnnConvKind::kBackwardFilter:
      config.input_shape = operand0_shape;
      config.filter_shape = result_shape;
      config.output_shape = operand1_shape;
      break;
    default:
      return Internal("Unknown convolution kind");
  }

  if (config.kind == CudnnConvKind::kForwardActivation) {
    if (!se::dnn::ActivationMode_IsValid(backend_config.activation_mode())) {
      return Internal("Bad activation mode: %s",
                      backend_config.ShortDebugString());
    }

    GpuConvConfig::FusionConfig fusion;
    fusion.mode =
        static_cast<se::dnn::ActivationMode>(backend_config.activation_mode());
    fusion.side_input_scale = backend_config.side_input_scale();
    fusion.leakyrelu_alpha = backend_config.leakyrelu_alpha();

    config.fusion = fusion;
  }

  const Window& window = desc.window;
  const ConvolutionDimensionNumbers& dnums = desc.dnums;

  VLOG(3) << "Convolution Algorithm: " << config.algorithm.ToString();
  VLOG(3) << "Convolution kind: " << CudnnConvKindToString(config.kind);
  VLOG(3) << "input shape: "
          << ShapeUtil::HumanStringWithLayout(config.input_shape);
  VLOG(3) << "filter shape: "
          << ShapeUtil::HumanStringWithLayout(config.filter_shape);
  VLOG(3) << "Output shape: "
          << ShapeUtil::HumanStringWithLayout(config.output_shape);
  VLOG(3) << "Window: { " << window.ShortDebugString() << " }";
  VLOG(3) << "Dim nums: { " << dnums.ShortDebugString() << " }";
  if (backend_config.reordered_int8_nchw_vect()) {
    VLOG(3) << "Filter and bias (if present) must be reordered with "
            << "cudnnReorderFilterAndBias";
  }

  const int num_dimensions = window.dimensions_size();
  CHECK_LE(num_dimensions, 3) << inst_as_string;

  // cuDNN does not support 1D convolutions. We therefore express 1D
  // convolutions as 2D convolutions where the first spatial dimension is 1.
  // This matches the behavior of TF (see definition of conv1d in
  // tensorflow/python/ops/nn_ops.py).
  const int effective_num_dimensions = std::max(2, num_dimensions);

  // If one dimension is reversed, we need to have all dimensions reversed (so
  // we're doing convolution not cross correlation).
  const bool dims_reversed =
      window.dimensions_size() > 0 && window.dimensions()[0].window_reversal();

  CHECK_EQ(num_dimensions, dnums.input_spatial_dimensions_size())
      << inst_as_string;
  CHECK_EQ(num_dimensions, dnums.kernel_spatial_dimensions_size())
      << inst_as_string;
  CHECK_EQ(num_dimensions, dnums.output_spatial_dimensions_size())
      << inst_as_string;
  for (const WindowDimension& dim : window.dimensions()) {
    CHECK_EQ(dims_reversed, dim.window_reversal()) << inst_as_string;
    CHECK_EQ(dim.padding_low(), dim.padding_high()) << inst_as_string;
    CHECK_EQ(dim.base_dilation(), 1)
        << "cudnn does not support base dilation; it "
           "must be made explicit with a kPad: "
        << inst_as_string;
  }

  // cuDNN's convolution APIs support the BDYX layout for activations/output and
  // the OIYX layout for weights.
  DataLayout input_dl;
  FilterLayout filter_dl;
  DataLayout output_dl;

  const Shape& input_shape = config.input_shape;
  const Shape& filter_shape = config.filter_shape;
  const Shape& output_shape = config.output_shape;

  TF_ASSIGN_OR_RETURN(std::tie(input_dl, filter_dl, output_dl),
                      XlaConvShapesToStreamExecutorLayouts(
                          dnums, input_shape, filter_shape, output_shape));
  if (backend_config.reordered_int8_nchw_vect()) {
    CHECK_EQ(filter_dl, FilterLayout::kOutputInputYX32);
    filter_dl = FilterLayout::kOutputInputYX32_CudnnReordered;
  }

  BatchDescriptor& input_descriptor = config.input_descriptor;
  input_descriptor = BatchDescriptor(effective_num_dimensions);
  input_descriptor.set_layout(input_dl)
      .set_feature_map_count(
          GetVectCSize(input_dl) *
          input_shape.dimensions(dnums.input_feature_dimension()))
      .set_count(input_shape.dimensions(dnums.input_batch_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    // Note that the dimensions are reversed. The same holds below.
    input_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        input_shape.dimensions(dnums.input_spatial_dimensions(dim)));
  }

  FilterDescriptor& filter_descriptor = config.filter_descriptor;
  filter_descriptor = FilterDescriptor(effective_num_dimensions);
  filter_descriptor.set_layout(filter_dl)
      .set_input_feature_map_count(
          GetVectCSize(filter_dl) *
          filter_shape.dimensions(dnums.kernel_input_feature_dimension()))
      .set_output_feature_map_count(
          filter_shape.dimensions(dnums.kernel_output_feature_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    filter_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        filter_shape.dimensions(dnums.kernel_spatial_dimensions(dim)));
  }

  config.conv_desc = ConvolutionDescriptor(effective_num_dimensions);
  config.conv_desc.set_group_count(desc.feature_group_count);
  config.conv_desc.set_convolution_not_crosscorr(dims_reversed);
  for (int dim = 0; dim < num_dimensions; ++dim) {
    config.conv_desc
        .set_zero_padding(
            static_cast<DimIndex>(effective_num_dimensions - dim - 1),
            window.dimensions(dim).padding_low())
        .set_filter_stride(
            static_cast<DimIndex>(effective_num_dimensions - dim - 1),
            window.dimensions(dim).stride())
        .set_dilation_rate(
            static_cast<DimIndex>(effective_num_dimensions - dim - 1),
            window.dimensions(dim).window_dilation());
  }

  BatchDescriptor& output_descriptor = config.output_descriptor;
  output_descriptor = BatchDescriptor(effective_num_dimensions);
  output_descriptor.set_layout(output_dl)
      .set_feature_map_count(
          GetVectCSize(output_dl) *
          output_shape.dimensions(dnums.output_feature_dimension()))
      .set_count(output_shape.dimensions(dnums.output_batch_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    output_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        output_shape.dimensions(dnums.output_spatial_dimensions(dim)));
  }

  // Add a singleton dimension in the 1D convolution case.
  for (int dim = 0; dim < effective_num_dimensions - num_dimensions; dim++) {
    input_descriptor.set_spatial_dim(static_cast<DimIndex>(dim), 1);
    output_descriptor.set_spatial_dim(static_cast<DimIndex>(dim), 1);
    filter_descriptor.set_spatial_dim(static_cast<DimIndex>(dim), 1);
    config.conv_desc.set_zero_padding(static_cast<DimIndex>(dim), 0)
        .set_filter_stride(static_cast<DimIndex>(dim), 1);
  }

  // Initialize bias descriptor for fused convolutions.
  BatchDescriptor& bias_descriptor = config.bias_descriptor;
  bias_descriptor = BatchDescriptor(config.output_descriptor.ndims());
  bias_descriptor.set_count(1)
      .set_height(1)
      .set_width(1)
      .set_feature_map_count(config.output_descriptor.feature_map_count())
      .set_layout([&] {
        // Normalize NCHW_VECT_C to NCHW for layout of `bias`, even though it's
        // actually the same (because `bias` only has one dimension):  cudnn
        // does not accept NCHW_VECT_C for `bias`.
        se::dnn::DataLayout layout = config.output_descriptor.layout();
        switch (layout) {
          case se::dnn::DataLayout::kBatchDepthYX4:
          case se::dnn::DataLayout::kBatchDepthYX32:
            return se::dnn::DataLayout::kBatchDepthYX;
          default:
            return layout;
        }
      }());
  if (bias_descriptor.ndims() == 3) {
    bias_descriptor.set_spatial_dim(se::dnn::DimIndex::Z, 1);
  }

  return config;
}

absl::StatusOr<GpuConvConfig> GetGpuConvConfig(
    const HloCustomCallInstruction* cudnn_call) {
  GpuConvDescriptor descriptor;

  TF_ASSIGN_OR_RETURN(descriptor.kind, GetCudnnConvKind(cudnn_call));
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      cudnn_call->backend_config<GpuBackendConfig>());
  descriptor.backend_config = gpu_backend_config.cudnn_conv_backend_config();
  descriptor.operand0_shape = cudnn_call->operand(0)->shape();
  descriptor.operand1_shape = cudnn_call->operand(1)->shape();
  descriptor.result_shape = cudnn_call->shape().tuple_shapes(0);
  descriptor.scratch_size =
      cudnn_call->shape().tuple_shapes().back().dimensions(0);

  descriptor.window = cudnn_call->window();
  descriptor.dnums = cudnn_call->convolution_dimension_numbers();
  descriptor.feature_group_count = cudnn_call->feature_group_count();
  return GetGpuConvConfig(descriptor, cudnn_call->ToString());
}

absl::StatusOr<GpuConvParams> GetGpuConvParams(
    const GpuConvConfig& config,
    absl::Span<const se::DeviceMemoryBase> operand_buffers,
    absl::Span<const se::DeviceMemoryBase> result_buffers) {
  GpuConvParams params;
  params.config = &config;

  switch (config.kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
    case CudnnConvKind::kForwardGraph:
      params.input_buf = operand_buffers[0];
      params.filter_buf = operand_buffers[1];
      params.output_buf = result_buffers[0];
      break;
    case CudnnConvKind::kBackwardInput:
      params.input_buf = result_buffers[0];
      params.filter_buf = operand_buffers[1];
      params.output_buf = operand_buffers[0];
      break;
    case CudnnConvKind::kBackwardFilter:
      params.input_buf = operand_buffers[0];
      params.filter_buf = result_buffers[0];
      params.output_buf = operand_buffers[1];
      break;
  }

  if (config.kind == CudnnConvKind::kForwardGraph) {
    params.operand_bufs = {operand_buffers.begin() + 2, operand_buffers.end()};
    params.aux_bufs = {result_buffers.begin() + 1, result_buffers.end()};
  }

  if (config.kind == CudnnConvKind::kForwardActivation) {
    params.fusion.emplace();
    GpuConvParams::FusionParams& fusion = *params.fusion;
    fusion.bias_buf = operand_buffers[2];
    if (operand_buffers.size() >= 4) {
      fusion.side_input_buf = operand_buffers[3];
    }
  }

  return params;
}

absl::Status RunGpuConv(const gpu::GpuConvConfig& config,
                        absl::Span<const se::DeviceMemoryBase> operand_buffers,
                        absl::Span<const se::DeviceMemoryBase> result_buffers,
                        se::DeviceMemoryBase scratch_memory, se::Stream* stream,
                        RunConvOptions options) {
  TF_ASSIGN_OR_RETURN(
      GpuConvParams params,
      GetGpuConvParams(config, operand_buffers, result_buffers));

  PrimitiveType input_primitive_type = config.input_type;
  switch (input_primitive_type) {
    case F8E4M3FN:
      if (config.kind != CudnnConvKind::kForwardGraph) {
        return Internal("FP8 convolution requires graph mode.");
      }
      return RunGpuConvImpl<tsl::float8_e4m3fn, tsl::float8_e4m3fn,
                            tsl::float8_e4m3fn>(params, stream, scratch_memory,
                                                options);
    case F8E5M2:
      if (config.kind != CudnnConvKind::kForwardGraph) {
        return Internal("FP8 convolution requires graph mode.");
      }
      return RunGpuConvImpl<tsl::float8_e5m2, tsl::float8_e5m2,
                            tsl::float8_e5m2>(params, stream, scratch_memory,
                                              options);
    case F16:
      return RunGpuConvImpl<Eigen::half, Eigen::half, Eigen::half>(
          params, stream, scratch_memory, options);
    case BF16:
      return RunGpuConvImpl<Eigen::bfloat16, Eigen::bfloat16, Eigen::bfloat16>(
          params, stream, scratch_memory, options);
    case F32:
      return RunGpuConvImpl<float, float, float>(params, stream, scratch_memory,
                                                 options);
    case F64:
      return RunGpuConvImpl<double, double, double>(params, stream,
                                                    scratch_memory, options);
    case S8: {
      PrimitiveType output_primitive_type = config.output_type;
      switch (output_primitive_type) {
        case F32:
          return RunGpuConvImpl<int8_t, float, float>(params, stream,
                                                      scratch_memory, options);
        case S8:
          return RunGpuConvImpl<int8_t, float, int8_t>(params, stream,
                                                       scratch_memory, options);
        default:
          return Unimplemented("Unimplemented convolution");
      }
    }
    default:
      return Unimplemented("Unimplemented convolution");
  }
}

}  // namespace gpu
}  // namespace xla
