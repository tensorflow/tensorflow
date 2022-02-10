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

#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

se::dnn::BatchDescriptor GetBiasDescriptor(const GpuConvConfig& config) {
  se::dnn::BatchDescriptor result(config.output_descriptor.ndims());
  result.set_count(1)
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
  if (result.ndims() == 3) {
    result.set_spatial_dim(se::dnn::DimIndex::Z, 1);
  }
  return result;
}

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

// A StreamExecutor ScratchAllocator that wraps a single XLA allocation,
// returning it (in its entirety) the first time Allocate() is called.
class ScratchBufAllocator : public se::ScratchAllocator {
 public:
  explicit ScratchBufAllocator(se::DeviceMemoryBase scratch)
      : scratch_(scratch) {}

  ~ScratchBufAllocator() override = default;

  int64_t GetMemoryLimitInBytes() override { return scratch_.size(); }

  se::port::StatusOr<DeviceMemory<uint8_t>> AllocateBytes(
      int64_t byte_size) override {
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
    return se::DeviceMemory<uint8_t>(scratch_);
  }

 private:
  se::DeviceMemoryBase scratch_;
  bool allocated_ = false;
};

template <typename ElementType, typename OutputType>
Status RunGpuConvUnfused(GpuConvParams params, se::Stream* stream,
                         RunConvOptions options,
                         DeviceMemory<ElementType> input_buf,
                         DeviceMemory<ElementType> filter_buf,
                         DeviceMemory<OutputType> output_buf,
                         DeviceMemoryBase scratch_memory) {
  if (params.config.conv_result_scale != 1) {
    return InternalError(
        "StreamExecutor doesn't support scaled convolution: %lf.",
        params.config.conv_result_scale);
  }

  TF_ASSIGN_OR_RETURN(se::dnn::ConvolutionKind kind,
                      GetDNNConvKindFromCudnnConvKind(params.config.kind));

  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType input_type,
      GetDNNDataTypeFromPrimitiveType(params.config.input_type));

  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(params.config.output_type));

  se::dnn::LazyOpRunner<se::dnn::ConvOp>* lazy_runner =
      options.runner_cache->AsConvRunner();
  absl::optional<se::dnn::LazyOpRunner<se::dnn::ConvOp>> local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config.algorithm);
    lazy_runner = &*local_runner;
  }

  se::dnn::ConvOp::Config config{kind,
                                 input_type,
                                 output_type,
                                 params.config.input_descriptor,
                                 params.config.filter_descriptor,
                                 params.config.output_descriptor,
                                 params.config.conv_desc};
  TF_ASSIGN_OR_RETURN(auto* runner,
                      lazy_runner->GetOrCreateRunner(config, stream));

  return (*runner)(stream, input_buf, filter_buf, output_buf, scratch_memory,
                   options.profile_result);
}

template <typename ElementType, typename BiasType, typename OutputType>
Status RunGpuConvForwardActivation(GpuConvParams params, se::Stream* stream,
                                   RunConvOptions options,
                                   DeviceMemory<ElementType> input_buf,
                                   DeviceMemory<ElementType> filter_buf,
                                   DeviceMemory<OutputType> output_buf,
                                   DeviceMemoryBase scratch_memory) {
  BatchDescriptor bias_desc = GetBiasDescriptor(params.config);

  se::DeviceMemory<OutputType> side_input(params.fusion->side_input_buf);
  // If there is no side input, use output as the side input.
  if (side_input.is_null()) {
    if (params.config.fusion->side_input_scale != 0) {
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

  se::dnn::LazyOpRunner<se::dnn::FusedConvOp>* lazy_runner =
      options.runner_cache->AsFusedConvRunner();
  absl::optional<se::dnn::LazyOpRunner<se::dnn::FusedConvOp>> local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config.algorithm);
    lazy_runner = &*local_runner;
  }

  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType input_type,
      GetDNNDataTypeFromPrimitiveType(params.config.input_type));

  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(params.config.output_type));

  se::dnn::FusedConvOp::Config config{se::dnn::ConvolutionKind::FORWARD,
                                      input_type,
                                      BiasTypeForInputType(input_type),
                                      output_type,
                                      params.config.conv_result_scale,
                                      params.config.fusion->side_input_scale,
                                      params.config.input_descriptor,
                                      params.config.filter_descriptor,
                                      bias_desc,
                                      params.config.output_descriptor,
                                      params.config.conv_desc,
                                      params.config.fusion->mode};
  TF_ASSIGN_OR_RETURN(auto* runner,
                      lazy_runner->GetOrCreateRunner(config, stream));

  return (*runner)(stream, input_buf, filter_buf, side_input,
                   params.fusion->bias_buf, output_buf, scratch_memory,
                   options.profile_result);
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
Status RunGpuConvInternalImpl(GpuConvParams params, se::Stream* stream,
                              RunConvOptions options,
                              DeviceMemory<ElementType> input_buf,
                              DeviceMemory<ElementType> filter_buf,
                              DeviceMemory<OutputType> output_buf,
                              DeviceMemoryBase scratch_memory) {
  switch (params.config.kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kBackwardInput:
    case CudnnConvKind::kBackwardFilter:
      return RunGpuConvUnfused(params, stream, options, input_buf, filter_buf,
                               output_buf, scratch_memory);
    case CudnnConvKind::kForwardActivation: {
      return RunGpuConvForwardActivation<ElementType, BiasType, OutputType>(
          params, stream, options, input_buf, filter_buf, output_buf,
          scratch_memory);
    }
  }
  return Status::OK();
}

// Specialization for integer types.  Only two forward convolutions are allowed.
template <typename ElementType, typename BiasType, typename OutputType,
          typename std::enable_if<std::is_integral<ElementType>::value>::type* =
              nullptr>
Status RunGpuConvInternalImpl(GpuConvParams params, se::Stream* stream,
                              RunConvOptions options,
                              DeviceMemory<ElementType> input_buf,
                              DeviceMemory<ElementType> filter_buf,
                              DeviceMemory<OutputType> output_buf,
                              DeviceMemoryBase scratch_memory) {
  switch (params.config.kind) {
    case CudnnConvKind::kForward:
      return RunGpuConvUnfused(params, stream, options, input_buf, filter_buf,
                               output_buf, scratch_memory);
    case CudnnConvKind::kForwardActivation:
      return RunGpuConvForwardActivation<ElementType, BiasType, OutputType>(
          params, stream, options, input_buf, filter_buf, output_buf,
          scratch_memory);
    default:
      return InternalError(
          "Only convolution kinds kForward and kForwardActivation are "
          "supported for integer types");
  }
  return Status::OK();
}

template <typename ElementType, typename BiasType, typename OutputType>
Status RunGpuConvImpl(const GpuConvParams& params, se::Stream* stream,
                      se::DeviceMemoryBase scratch_memory,
                      RunConvOptions options) {
  auto input_buf = se::DeviceMemory<ElementType>(params.input_buf);
  auto filter_buf = se::DeviceMemory<ElementType>(params.filter_buf);
  auto output_buf = se::DeviceMemory<OutputType>(params.output_buf);

  se::dnn::AlgorithmDesc algorithm = params.config.algorithm;
  if (options.runner_cache) {
    algorithm = options.runner_cache->ToAlgorithmDesc();
  }

  Status run_status = RunGpuConvInternalImpl<ElementType, BiasType, OutputType>(
      params, stream, options, input_buf, filter_buf, output_buf,
      scratch_memory);

  if (run_status != Status::OK()) {
    return run_status;
  }

  if (!stream->ok()) {
    return InternalError(
        "Unable to launch convolution with type %s and algorithm %s",
        CudnnConvKindToString(params.config.kind), algorithm.ToString());
  }
  return Status::OK();
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
      return 32;
    default:
      return 1;
  }
}

}  // anonymous namespace

StatusOr<GpuConvConfig> GetGpuConvConfig(
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

  switch (config.kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
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
      return InternalError("Unknown convolution kind");
  }

  if (config.kind == CudnnConvKind::kForwardActivation) {
    config.fusion.emplace();
    GpuConvConfig::FusionConfig& fusion = *config.fusion;
    if (!se::dnn::ActivationMode_IsValid(backend_config.activation_mode())) {
      return InternalError("Bad activation mode: %s",
                           backend_config.ShortDebugString());
    }
    fusion.mode =
        static_cast<se::dnn::ActivationMode>(backend_config.activation_mode());
    fusion.side_input_scale = backend_config.side_input_scale();
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

  return config;
}

StatusOr<GpuConvConfig> GetGpuConvConfig(
    const HloCustomCallInstruction* cudnn_call) {
  GpuConvDescriptor descriptor;

  TF_ASSIGN_OR_RETURN(descriptor.kind, GetCudnnConvKind(cudnn_call));
  TF_ASSIGN_OR_RETURN(descriptor.backend_config,
                      cudnn_call->backend_config<CudnnConvBackendConfig>());
  descriptor.operand0_shape = cudnn_call->operand(0)->shape();
  descriptor.operand1_shape = cudnn_call->operand(1)->shape();
  descriptor.result_shape = cudnn_call->shape().tuple_shapes(0);
  descriptor.scratch_size = cudnn_call->shape().tuple_shapes(1).dimensions(0);
  descriptor.window = cudnn_call->window();
  descriptor.dnums = cudnn_call->convolution_dimension_numbers();
  descriptor.feature_group_count = cudnn_call->feature_group_count();
  return GetGpuConvConfig(descriptor, cudnn_call->ToString());
}

StatusOr<GpuConvParams> GetGpuConvParams(
    const GpuConvConfig& config,
    absl::Span<const se::DeviceMemoryBase> operand_buffers,
    se::DeviceMemoryBase result_buffer) {
  GpuConvParams params;
  params.config = config;

  switch (config.kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
      params.input_buf = operand_buffers[0];
      params.filter_buf = operand_buffers[1];
      params.output_buf = result_buffer;
      break;
    case CudnnConvKind::kBackwardInput:
      params.input_buf = result_buffer;
      params.filter_buf = operand_buffers[1];
      params.output_buf = operand_buffers[0];
      break;
    case CudnnConvKind::kBackwardFilter:
      params.input_buf = operand_buffers[0];
      params.filter_buf = result_buffer;
      params.output_buf = operand_buffers[1];
      break;
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

Status RunGpuConv(const gpu::GpuConvConfig& config,
                  absl::Span<const se::DeviceMemoryBase> operand_buffers,
                  se::DeviceMemoryBase result_buffer,
                  se::DeviceMemoryBase scratch_memory, se::Stream* stream,
                  RunConvOptions options) {
  TF_ASSIGN_OR_RETURN(GpuConvParams params,
                      GetGpuConvParams(config, operand_buffers, result_buffer));

  PrimitiveType input_primitive_type = config.input_type;
  switch (input_primitive_type) {
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
