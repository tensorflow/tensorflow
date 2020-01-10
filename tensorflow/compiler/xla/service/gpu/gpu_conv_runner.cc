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
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
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

// A StreamExecutor ScratchAllocator that wraps a single XLA allocation,
// returning it (in its entirety) the first time Allocate() is called.
class ScratchBufAllocator : public se::ScratchAllocator {
 public:
  explicit ScratchBufAllocator(se::DeviceMemoryBase scratch)
      : scratch_(scratch) {}

  ~ScratchBufAllocator() override = default;

  int64 GetMemoryLimitInBytes() override { return scratch_.size(); }

  se::port::StatusOr<DeviceMemory<uint8>> AllocateBytes(
      int64 byte_size) override {
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

template <typename ElementType, typename OutputType>
Status RunGpuConvForward(GpuConvParams params,
                         se::ScratchAllocator* scratch_allocator,
                         se::Stream* stream, RunConvOptions options,
                         DeviceMemory<ElementType> input_buf,
                         DeviceMemory<ElementType> filter_buf,
                         DeviceMemory<OutputType> output_buf,
                         AlgorithmConfig algorithm) {
  if (params.conv_result_scale != 1) {
    return InternalError(
        "StreamExecutor doesn't support scaled convolution: %lf.",
        params.conv_result_scale);
  }
  stream->ThenConvolveWithAlgorithm(
      params.input_descriptor, input_buf, params.filter_descriptor, filter_buf,
      params.conv_desc, params.output_descriptor, &output_buf,
      scratch_allocator, algorithm, options.profile_result);
  return Status::OK();
}

template <typename ElementType, typename BiasType, typename OutputType>
Status RunGpuConvForwardActivation(GpuConvParams params,
                                   se::ScratchAllocator* scratch_allocator,
                                   se::Stream* stream, RunConvOptions options,
                                   DeviceMemory<ElementType> input_buf,
                                   DeviceMemory<ElementType> filter_buf,
                                   DeviceMemory<OutputType> output_buf,
                                   AlgorithmConfig algorithm) {
  BatchDescriptor bias_desc;
  bias_desc.set_count(1)
      .set_height(1)
      .set_width(1)
      .set_feature_map_count(params.output_descriptor.feature_map_count())
      .set_layout(params.output_descriptor.layout());

  se::DeviceMemory<OutputType> side_input(params.fusion->side_input_buf);
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
      params.input_descriptor, input_buf, params.conv_result_scale,
      params.filter_descriptor, filter_buf, params.conv_desc, side_input,
      params.fusion->side_input_scale, bias_desc,
      DeviceMemory<BiasType>(params.fusion->bias_buf), params.fusion->mode,
      params.output_descriptor, &output_buf, scratch_allocator, algorithm,
      options.profile_result);

  return Status::OK();
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
Status RunGpuConvInternalImpl(GpuConvParams params,
                              se::ScratchAllocator* scratch_allocator,
                              se::Stream* stream, RunConvOptions options,
                              DeviceMemory<ElementType> input_buf,
                              DeviceMemory<ElementType> filter_buf,
                              DeviceMemory<OutputType> output_buf,
                              AlgorithmConfig algorithm) {
  switch (params.kind) {
    case CudnnConvKind::kForward:
      return RunGpuConvForward(params, scratch_allocator, stream, options,
                               input_buf, filter_buf, output_buf, algorithm);
    case CudnnConvKind::kBackwardInput:
      if (params.conv_result_scale != 1) {
        return InternalError(
            "StreamExecutor doesn't support scaled convolution: %lf.",
            params.conv_result_scale);
      }
      stream->ThenConvolveBackwardDataWithAlgorithm(
          params.filter_descriptor, filter_buf, params.output_descriptor,
          output_buf, params.conv_desc, params.input_descriptor, &input_buf,
          scratch_allocator, algorithm, options.profile_result);
      break;
    case CudnnConvKind::kBackwardFilter:
      if (params.conv_result_scale != 1) {
        return InternalError(
            "StreamExecutor doesn't support scaled convolution: %lf.",
            params.conv_result_scale);
      }
      stream->ThenConvolveBackwardFilterWithAlgorithm(
          params.input_descriptor, input_buf, params.output_descriptor,
          output_buf, params.conv_desc, params.filter_descriptor, &filter_buf,
          scratch_allocator, algorithm, options.profile_result);
      break;
    case CudnnConvKind::kForwardActivation: {
      return RunGpuConvForwardActivation<ElementType, BiasType, OutputType>(
          params, scratch_allocator, stream, options, input_buf, filter_buf,
          output_buf, algorithm);
    }
  }
  return Status::OK();
}

// Specialization for integer types.  Only two forward convolutions are allowed.
template <typename ElementType, typename BiasType, typename OutputType,
          typename std::enable_if<std::is_integral<ElementType>::value>::type* =
              nullptr>
Status RunGpuConvInternalImpl(GpuConvParams params,
                              se::ScratchAllocator* scratch_allocator,
                              se::Stream* stream, RunConvOptions options,
                              DeviceMemory<ElementType> input_buf,
                              DeviceMemory<ElementType> filter_buf,
                              DeviceMemory<OutputType> output_buf,
                              AlgorithmConfig algorithm) {
  switch (params.kind) {
    case CudnnConvKind::kForward:
      return RunGpuConvForward(params, scratch_allocator, stream, options,
                               input_buf, filter_buf, output_buf, algorithm);
    case CudnnConvKind::kForwardActivation:
      return RunGpuConvForwardActivation<ElementType, BiasType, OutputType>(
          params, scratch_allocator, stream, options, input_buf, filter_buf,
          output_buf, algorithm);
    default:
      return InternalError(
          "Only convolution kinds kForward and kForwardActivation are "
          "supported for integer types");
  }
  return Status::OK();
}

template <typename ElementType, typename BiasType, typename OutputType>
Status RunGpuConvImpl(const GpuConvParams& params,
                      se::ScratchAllocator* scratch_allocator,
                      se::Stream* stream, RunConvOptions options) {
  auto input_buf = se::DeviceMemory<ElementType>(params.input_buf);
  auto filter_buf = se::DeviceMemory<ElementType>(params.filter_buf);
  auto output_buf = se::DeviceMemory<OutputType>(params.output_buf);
  AlgorithmConfig algorithm = params.algorithm;

  if (options.algo_override.has_value()) {
    algorithm = AlgorithmConfig(*options.algo_override);
  }

  Status run_status = RunGpuConvInternalImpl<ElementType, BiasType, OutputType>(
      params, scratch_allocator, stream, options, input_buf, filter_buf,
      output_buf, algorithm);

  if (run_status != Status::OK()) {
    return run_status;
  }

  if (!stream->ok()) {
    return InternalError(
        "Unable to launch convolution with type %s and algorithm (%d, %s)",
        CudnnConvKindToString(params.kind), algorithm.algorithm()->algo_id(),
        algorithm.algorithm_no_scratch().has_value()
            ? absl::StrCat(algorithm.algorithm_no_scratch()->algo_id())
            : "none");
  }
  return Status::OK();
}

}  // anonymous namespace

StatusOr<GpuConvParams> GetGpuConvParams(
    const HloCustomCallInstruction* conv,
    absl::Span<se::DeviceMemoryBase> operand_buffers,
    se::DeviceMemoryBase result_buffer) {
  GpuConvParams params;

  TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig backend_config,
                      conv->backend_config<CudnnConvBackendConfig>());
  TF_ASSIGN_OR_RETURN(params.kind, GetCudnnConvKind(conv));
  const Shape* input_shape;
  const Shape* filter_shape;
  const Shape* output_shape;

  // The third field is scratch size stored from conv_algorithm_picker
  // The operand is added to the shape field of the conv instruction
  // in GpuConvAlgorithmPicker::RunOnInstruction() call.
  params.algorithm = se::dnn::AlgorithmConfig(
      se::dnn::AlgorithmDesc(backend_config.algorithm(),
                             backend_config.tensor_ops_enabled()),
      conv->shape().tuple_shapes(1).dimensions(0));
  params.conv_result_scale = backend_config.conv_result_scale();

  switch (params.kind) {
    case CudnnConvKind::kForward:
      input_shape = &conv->operand(0)->shape();
      filter_shape = &conv->operand(1)->shape();
      output_shape = &conv->shape().tuple_shapes(0);
      params.input_buf = operand_buffers[0];
      params.filter_buf = operand_buffers[1];
      params.output_buf = result_buffer;
      break;
    case CudnnConvKind::kBackwardInput:
      input_shape = &conv->shape().tuple_shapes(0);
      filter_shape = &conv->operand(1)->shape();
      output_shape = &conv->operand(0)->shape();
      params.input_buf = result_buffer;
      params.filter_buf = operand_buffers[1];
      params.output_buf = operand_buffers[0];
      break;
    case CudnnConvKind::kBackwardFilter:
      input_shape = &conv->operand(0)->shape();
      filter_shape = &conv->shape().tuple_shapes(0);
      output_shape = &conv->operand(1)->shape();
      params.input_buf = operand_buffers[0];
      params.filter_buf = result_buffer;
      params.output_buf = operand_buffers[1];
      break;
    case CudnnConvKind::kForwardActivation: {
      input_shape = &conv->operand(0)->shape();
      filter_shape = &conv->operand(1)->shape();
      output_shape = &conv->shape().tuple_shapes(0);
      params.fusion.emplace();
      GpuConvParams::FusionParams& fusion = *params.fusion;
      if (!se::dnn::ActivationMode_IsValid(backend_config.activation_mode())) {
        return InternalError("Bad activation mode: %s",
                             backend_config.ShortDebugString());
      }
      fusion.mode = static_cast<se::dnn::ActivationMode>(
          backend_config.activation_mode());
      fusion.side_input_scale = backend_config.side_input_scale();
      params.input_buf = operand_buffers[0];
      params.filter_buf = operand_buffers[1];
      params.output_buf = result_buffer;
      params.fusion->bias_buf = operand_buffers[2];
      if (operand_buffers.size() >= 4) {
        params.fusion->side_input_buf = operand_buffers[3];
      }
    }
  }

  const Window& window = conv->window();
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();

  VLOG(3) << "Convolution Algorithm: "
          << params.algorithm.algorithm()->algo_id();
  VLOG(3) << "tensor_ops_enabled: "
          << params.algorithm.algorithm()->tensor_ops_enabled();
  VLOG(3) << "Convolution kind: " << CudnnConvKindToString(params.kind);
  VLOG(3) << "input shape: " << ShapeUtil::HumanStringWithLayout(*input_shape);
  VLOG(3) << "filter shape: "
          << ShapeUtil::HumanStringWithLayout(*filter_shape);
  VLOG(3) << "Output shape: "
          << ShapeUtil::HumanStringWithLayout(*output_shape);
  VLOG(3) << "Window: { " << window.ShortDebugString() << " }";
  VLOG(3) << "Dim nums: { " << dnums.ShortDebugString() << " }";

  const int num_dimensions = window.dimensions_size();
  CHECK_LE(num_dimensions, 3) << conv->ToString();
  CHECK_GE(num_dimensions, 1) << conv->ToString();
  // cuDNN does not support 1D convolutions. We therefore express 1D
  // convolutions as 2D convolutions where the first spatial dimension is 1.
  // This matches the behavior of TF (see definition of conv1d in
  // tensorflow/python/ops/nn_ops.py).
  const int effective_num_dimensions = std::max(2, num_dimensions);

  // If one dimension is reversed, we need to have all dimensions reversed (so
  // we're doing convolution not cross correlation).
  const bool dims_reversed = window.dimensions()[0].window_reversal();

  CHECK_EQ(num_dimensions, dnums.input_spatial_dimensions_size())
      << conv->ToString();
  CHECK_EQ(num_dimensions, dnums.kernel_spatial_dimensions_size())
      << conv->ToString();
  CHECK_EQ(num_dimensions, dnums.output_spatial_dimensions_size())
      << conv->ToString();
  for (const WindowDimension& dim : window.dimensions()) {
    CHECK_EQ(dims_reversed, dim.window_reversal()) << conv->ToString();
    CHECK_EQ(dim.padding_low(), dim.padding_high()) << conv->ToString();
    CHECK_EQ(dim.base_dilation(), 1)
        << "cudnn does not support base dilation; it "
           "must be made explicit with a kPad: "
        << conv->ToString();
  }

  // cuDNN's convolution APIs support the BDYX layout for activations/output and
  // the OIYX layout for weights.
  DataLayout input_dl;
  FilterLayout filter_dl;
  DataLayout output_dl;

  TF_ASSIGN_OR_RETURN(std::tie(input_dl, filter_dl, output_dl),
                      XlaConvLayoutsToStreamExecutorLayouts(
                          dnums, input_shape->layout(), filter_shape->layout(),
                          output_shape->layout()));

  BatchDescriptor& input_descriptor = params.input_descriptor;
  input_descriptor = BatchDescriptor(effective_num_dimensions);
  input_descriptor.set_layout(input_dl)
      .set_feature_map_count(
          input_shape->dimensions(dnums.input_feature_dimension()))
      .set_count(input_shape->dimensions(dnums.input_batch_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    // Note that the dimensions are reversed. The same holds below.
    input_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        input_shape->dimensions(dnums.input_spatial_dimensions(dim)));
  }

  FilterDescriptor& filter_descriptor = params.filter_descriptor;
  filter_descriptor = FilterDescriptor(effective_num_dimensions);
  filter_descriptor.set_layout(filter_dl)
      .set_input_feature_map_count(
          filter_shape->dimensions(dnums.kernel_input_feature_dimension()))
      .set_output_feature_map_count(
          filter_shape->dimensions(dnums.kernel_output_feature_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    filter_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        filter_shape->dimensions(dnums.kernel_spatial_dimensions(dim)));
  }

  params.conv_desc = ConvolutionDescriptor(effective_num_dimensions);
  params.conv_desc.set_group_count(conv->feature_group_count());
  params.conv_desc.set_convolution_not_crosscorr(dims_reversed);
  for (int dim = 0; dim < num_dimensions; ++dim) {
    params.conv_desc
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

  BatchDescriptor& output_descriptor = params.output_descriptor;
  output_descriptor = BatchDescriptor(effective_num_dimensions);
  output_descriptor.set_layout(output_dl)
      .set_feature_map_count(
          output_shape->dimensions(dnums.output_feature_dimension()))
      .set_count(output_shape->dimensions(dnums.output_batch_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    output_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        output_shape->dimensions(dnums.output_spatial_dimensions(dim)));
  }

  // Add a singleton dimension in the 1D convolution case.
  if (num_dimensions == 1) {
    input_descriptor.set_spatial_dim(static_cast<DimIndex>(0), 1);
    output_descriptor.set_spatial_dim(static_cast<DimIndex>(0), 1);
    filter_descriptor.set_spatial_dim(static_cast<DimIndex>(0), 1);
    params.conv_desc.set_zero_padding(static_cast<DimIndex>(0), 0)
        .set_filter_stride(static_cast<DimIndex>(0), 1);
  }

  return params;
}

Status RunGpuConv(const HloCustomCallInstruction* conv,
                  absl::Span<se::DeviceMemoryBase> operand_buffers,
                  se::DeviceMemoryBase result_buffer,
                  se::DeviceMemoryBase scratch_buf, se::Stream* stream,
                  RunConvOptions options) {
  ScratchBufAllocator scratch_allocator(scratch_buf);
  return RunGpuConv(conv, operand_buffers, result_buffer, &scratch_allocator,
                    stream, options);
}

Status RunGpuConv(const HloCustomCallInstruction* conv,
                  absl::Span<se::DeviceMemoryBase> operand_buffers,
                  se::DeviceMemoryBase result_buffer,
                  se::ScratchAllocator* scratch_allocator, se::Stream* stream,
                  RunConvOptions options) {
  TF_ASSIGN_OR_RETURN(GpuConvParams params,
                      GetGpuConvParams(conv, operand_buffers, result_buffer));

  PrimitiveType input_primitive_type = conv->operand(0)->shape().element_type();
  switch (input_primitive_type) {
    case F16:
      return RunGpuConvImpl<Eigen::half, Eigen::half, Eigen::half>(
          params, scratch_allocator, stream, options);
    case F32:
      return RunGpuConvImpl<float, float, float>(params, scratch_allocator,
                                                 stream, options);
    case F64:
      return RunGpuConvImpl<double, double, double>(params, scratch_allocator,
                                                    stream, options);
    case S8: {
      PrimitiveType output_primitive_type =
          conv->shape().tuple_shapes(0).element_type();
      switch (output_primitive_type) {
        case F32:
          return RunGpuConvImpl<int8, float, float>(params, scratch_allocator,
                                                    stream, options);
        case S8:
          return RunGpuConvImpl<int8, float, int8>(params, scratch_allocator,
                                                   stream, options);
        default:
          LOG(FATAL) << conv->ToString();
      }
    }
    default:
      LOG(FATAL) << conv->ToString();
  }
}

}  // namespace gpu
}  // namespace xla
