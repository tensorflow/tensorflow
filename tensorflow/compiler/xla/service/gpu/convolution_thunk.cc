/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"

#include <string>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {

using se::dnn::BatchDescriptor;
using se::dnn::ConvolutionDescriptor;
using se::dnn::DataLayout;
using se::dnn::FilterDescriptor;
using se::dnn::FilterLayout;

ConvolveScratchAllocator::ConvolveScratchAllocator(
    int device_ordinal, DeviceMemoryAllocator* memory_allocator)
    : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {}

ConvolveScratchAllocator::~ConvolveScratchAllocator() {
  for (auto& allocated_buffer : allocated_buffers_) {
    if (!memory_allocator_->Deallocate(device_ordinal_, &allocated_buffer)
             .ok()) {
      // The program can still continue with failed deallocation.
      LOG(ERROR) << "Failed to deallocate the allocated buffer: "
                 << allocated_buffer.opaque();
    }
  }
}

int64 ConvolveScratchAllocator::GetMemoryLimitInBytes(se::Stream* stream) {
  constexpr int64 kConvolveScratchSize = 1LL << 32;  // 4GB by default.
  return kConvolveScratchSize;
}

se::port::StatusOr<se::DeviceMemory<uint8>>
ConvolveScratchAllocator::AllocateBytes(se::Stream* stream, int64 byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes(stream)) {
    return se::port::Status(
        se::port::error::RESOURCE_EXHAUSTED,
        tensorflow::strings::Printf(
            "Allocating %lld bytes exceeds the memory limit of %lld bytes.",
            byte_size, GetMemoryLimitInBytes(stream)));
  }

  auto status_or_memory =
      memory_allocator_->Allocate(device_ordinal_, byte_size,
                                  /*retry_on_failure=*/false);
  if (!status_or_memory.ok()) {
    return se::port::Status(se::port::error::RESOURCE_EXHAUSTED,
                            tensorflow::strings::Printf(
                                "Failed to allocate %lld bytes on device %d.",
                                byte_size, device_ordinal_));
  }
  se::DeviceMemoryBase allocated_buffer = status_or_memory.ValueOrDie();
  allocated_buffers_.push_back(allocated_buffer);
  total_allocated_bytes_ += byte_size;
  return se::DeviceMemory<uint8>(allocated_buffer);
}

string ConvolutionKindToString(
    ConvolutionThunk::ConvolutionKind convolution_kind) {
  switch (convolution_kind) {
    case ConvolutionThunk::ConvolutionKind::kForward:
      return "forward";
    case ConvolutionThunk::ConvolutionKind::kBackwardFilter:
      return "backward_filter";
    case ConvolutionThunk::ConvolutionKind::kBackwardInput:
      return "backward_input";
  }
  return "unknown convolution kind";
}

ConvolutionThunk::ConvolutionThunk(
    ConvolutionKind convolution_kind,
    const BufferAllocation::Slice& input_buffer,
    const BufferAllocation::Slice& filter_buffer,
    const BufferAllocation::Slice& output_buffer, const Shape& input_shape,
    const Shape& filter_shape, const Shape& output_shape, const Window& window,
    const ConvolutionDimensionNumbers& dim_nums, const HloInstruction* hlo)
    : Thunk(Kind::kConvolution, hlo),
      convolution_kind_(convolution_kind),
      input_buffer_(input_buffer),
      filter_buffer_(filter_buffer),
      output_buffer_(output_buffer),
      input_shape_(input_shape),
      filter_shape_(filter_shape),
      output_shape_(output_shape),
      window_(window),
      dim_nums_(dim_nums) {}

tensorflow::Status ConvolutionThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream) {
  VLOG(3) << "Convolution kind: " << ConvolutionKindToString(convolution_kind_);
  VLOG(3) << "input shape: { " << input_shape_.ShortDebugString() << " }";
  VLOG(3) << "filter shape: { " << filter_shape_.ShortDebugString() << " }";
  VLOG(3) << "Output shape: { " << output_shape_.ShortDebugString() << " }";
  VLOG(3) << "Dim nums: { " << dim_nums_.ShortDebugString() << " }";
  VLOG(3) << "Window: { " << window_.ShortDebugString() << " }";

  const int num_dimensions = window_.dimensions_size();
  CHECK_LE(num_dimensions, 3);
  // cuDNN does not support 1D convolutions. We therefore express 1D
  // convolutions as 2D convolutions where the first spatial dimension is 1.
  // This matches the behavior of TF (see definition of conv1d in
  // tensorflow/python/ops/nn_ops.py).
  const int effective_num_dimensions = std::max(2, num_dimensions);

  CHECK_EQ(F32, output_shape_.element_type());
  CHECK_EQ(num_dimensions, dim_nums_.spatial_dimensions_size());
  CHECK_EQ(num_dimensions, dim_nums_.kernel_spatial_dimensions_size());
  for (const WindowDimension& dim : window_.dimensions()) {
    CHECK_EQ(dim.padding_low(), dim.padding_high());
  }

  // cuDNN's convolution APIs support the BDYX layout for activations/output and
  // the OIYX layout for weights.
  BatchDescriptor input_descriptor(effective_num_dimensions);
  input_descriptor.set_layout(DataLayout::kBatchDepthYX)
      .set_feature_map_count(
          input_shape_.dimensions(dim_nums_.feature_dimension()))
      .set_count(input_shape_.dimensions(dim_nums_.batch_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    // Note that the dimensions are reversed. The same holds below.
    input_descriptor.set_spatial_dim(
        static_cast<se::dnn::DimIndex>(effective_num_dimensions - dim - 1),
        input_shape_.dimensions(dim_nums_.spatial_dimensions(dim)));
  }

  FilterDescriptor filter_descriptor(effective_num_dimensions);
  filter_descriptor.set_layout(FilterLayout::kOutputInputYX)
      .set_input_feature_map_count(
          filter_shape_.dimensions(dim_nums_.kernel_input_feature_dimension()))
      .set_output_feature_map_count(filter_shape_.dimensions(
          dim_nums_.kernel_output_feature_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    filter_descriptor.set_spatial_dim(
        static_cast<se::dnn::DimIndex>(effective_num_dimensions - dim - 1),
        filter_shape_.dimensions(dim_nums_.kernel_spatial_dimensions(dim)));
  }

  ConvolutionDescriptor convolution_descriptor(effective_num_dimensions);
  for (int dim = 0; dim < num_dimensions; ++dim) {
    convolution_descriptor
        .set_zero_padding(
            static_cast<se::dnn::DimIndex>(effective_num_dimensions - dim - 1),
            window_.dimensions(dim).padding_low())
        .set_filter_stride(
            static_cast<se::dnn::DimIndex>(effective_num_dimensions - dim - 1),
            window_.dimensions(dim).stride());
  }

  BatchDescriptor output_descriptor(effective_num_dimensions);
  output_descriptor.set_layout(DataLayout::kBatchDepthYX)
      .set_feature_map_count(
          output_shape_.dimensions(dim_nums_.feature_dimension()))
      .set_count(output_shape_.dimensions(dim_nums_.batch_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    output_descriptor.set_spatial_dim(
        static_cast<se::dnn::DimIndex>(effective_num_dimensions - dim - 1),
        output_shape_.dimensions(dim_nums_.spatial_dimensions(dim)));
  }

  // Add a singleton dimension in the 1D convolution case.
  if (num_dimensions == 1) {
    input_descriptor.set_spatial_dim(static_cast<se::dnn::DimIndex>(0), 1);
    output_descriptor.set_spatial_dim(static_cast<se::dnn::DimIndex>(0), 1);
    filter_descriptor.set_spatial_dim(static_cast<se::dnn::DimIndex>(0), 1);
    convolution_descriptor
        .set_zero_padding(static_cast<se::dnn::DimIndex>(0), 0)
        .set_filter_stride(static_cast<se::dnn::DimIndex>(0), 1);
  }

  se::DeviceMemory<float> input_data(
      buffer_allocations.GetDeviceAddress(input_buffer_));
  se::DeviceMemory<float> filter_data(
      buffer_allocations.GetDeviceAddress(filter_buffer_));
  se::DeviceMemory<float> output_data(
      buffer_allocations.GetDeviceAddress(output_buffer_));
  return ConvolveWithTune(input_descriptor, input_data, filter_descriptor,
                          filter_data, output_descriptor, output_data,
                          convolution_descriptor, buffer_allocations, stream);
}

tensorflow::Status ConvolutionThunk::Convolve(
    const BatchDescriptor& input_descriptor, se::DeviceMemory<float> input_data,
    const FilterDescriptor& filter_descriptor,
    se::DeviceMemory<float> filter_data,
    const BatchDescriptor& output_descriptor,
    se::DeviceMemory<float> output_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const se::dnn::AlgorithmConfig& algorithm_config, se::Stream* stream,
    ConvolveScratchAllocator* scratch_allocator,
    se::dnn::ProfileResult* profile_result) {
  bool launch_ok;
  switch (convolution_kind_) {
    case ConvolutionKind::kBackwardFilter:
      launch_ok =
          stream
              ->ThenConvolveBackwardFilterWithAlgorithm(
                  input_descriptor, input_data, output_descriptor, output_data,
                  convolution_descriptor, filter_descriptor, &filter_data,
                  scratch_allocator, algorithm_config, profile_result)
              .ok();
      break;
    case ConvolutionKind::kBackwardInput:
      launch_ok = stream
                      ->ThenConvolveBackwardDataWithAlgorithm(
                          filter_descriptor, filter_data, output_descriptor,
                          output_data, convolution_descriptor, input_descriptor,
                          &input_data, scratch_allocator, algorithm_config,
                          profile_result)
                      .ok();
      break;
    case ConvolutionKind::kForward:
      launch_ok =
          stream
              ->ThenConvolveWithAlgorithm(
                  input_descriptor, input_data, filter_descriptor, filter_data,
                  convolution_descriptor, output_descriptor, &output_data,
                  scratch_allocator, algorithm_config, profile_result)
              .ok();
      break;
  }
  if (launch_ok) {
    return tensorflow::Status::OK();
  }
  return InternalError(
      "Unable to launch convolution for thunk %p with type %s and algorithm "
      "(%lld, %lld)",
      this, ConvolutionKindToString(convolution_kind_).c_str(),
      algorithm_config.algorithm(), algorithm_config.algorithm_no_scratch());
}

std::vector<se::dnn::AlgorithmType> ConvolutionThunk::GetAlgorithms(
    se::StreamExecutor* stream_exec) const {
  std::vector<se::dnn::AlgorithmType> algorithms;
  // TODO(yangzihao): Currently disable the use of winograd nonfused in XLA
  // by default. Should send in conv parameters and enable it when
  // ShouldIncludeWinogradNonfusedAlgo() returns true.
  switch (convolution_kind_) {
    case ConvolutionKind::kBackwardFilter:
      CHECK(stream_exec->GetConvolveBackwardFilterAlgorithms(
          /*with_winograd_nonfused=*/false, &algorithms));
      break;
    case ConvolutionKind::kBackwardInput:
      CHECK(stream_exec->GetConvolveBackwardDataAlgorithms(
          /*with_winograd_nonfused=*/false, &algorithms));
      break;
    case ConvolutionKind::kForward:
      CHECK(stream_exec->GetConvolveAlgorithms(/*with_winograd_nonfused=*/false,
                                               &algorithms));
      break;
  }
  return algorithms;
}

tensorflow::Status ConvolutionThunk::ConvolveWithTune(
    const BatchDescriptor& input_descriptor, se::DeviceMemory<float> input_data,
    const FilterDescriptor& filter_descriptor,
    se::DeviceMemory<float> filter_data,
    const BatchDescriptor& output_descriptor,
    se::DeviceMemory<float> output_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BufferAllocations& buffer_allocations, se::Stream* stream) {
  // TODO(b/29126320): Try cudnn v5's new auto-tuner when it's rolled out.
  if (best_algorithm_.algorithm() == se::dnn::kDefaultAlgorithm) {
    // Auto-tuning either is disabled or only happens in the first run of this
    // function.
    VLOG(2) << "Profiling for best convolution algorithm used for "
               "ConvolutionThunk: "
            << this;

    se::dnn::ProfileResult best_result;
    se::dnn::ProfileResult best_result_without_scratch;
    for (se::dnn::AlgorithmType algorithm : GetAlgorithms(stream->parent())) {
      ConvolveScratchAllocator scratch_allocator(
          buffer_allocations.device_ordinal(),
          buffer_allocations.memory_allocator());
      se::dnn::ProfileResult profile_result;
      bool launch_ok =
          Convolve(input_descriptor, input_data, filter_descriptor, filter_data,
                   output_descriptor, output_data, convolution_descriptor,
                   se::dnn::AlgorithmConfig(algorithm, algorithm), stream,
                   &scratch_allocator, &profile_result)
              .ok();
      if (launch_ok && profile_result.is_valid()) {
        if (profile_result.elapsed_time_in_ms() <
            best_result.elapsed_time_in_ms()) {
          best_result = profile_result;
        }
        if (scratch_allocator.TotalAllocatedBytes() == 0 &&
            profile_result.elapsed_time_in_ms() <
                best_result_without_scratch.elapsed_time_in_ms()) {
          best_result_without_scratch = profile_result;
        }
      }
    }

    if (best_result.is_valid()) {
      best_algorithm_.set_algorithm(best_result.algorithm());
    } else {
      LOG(ERROR) << "No convolution algorithm works with profiling. Fall back "
                    "to the default algorithm.";
      best_algorithm_.set_algorithm(se::dnn::kDefaultAlgorithm);
    }

    if (best_result_without_scratch.is_valid()) {
      best_algorithm_.set_algorithm_no_scratch(
          best_result_without_scratch.algorithm());
    } else {
      LOG(ERROR) << "No convolution algorithm without scratch works with "
                    "profiling. Fall back "
                    "to the default algorithm.";
      best_algorithm_.set_algorithm_no_scratch(se::dnn::kDefaultAlgorithm);
    }
  }

  {
    VLOG(2) << "Using convolution algorithm (" << best_algorithm_.algorithm()
            << ", " << best_algorithm_.algorithm_no_scratch()
            << ") for ConvolutionThunk: " << this;
    ConvolveScratchAllocator scratch_allocator(
        buffer_allocations.device_ordinal(),
        buffer_allocations.memory_allocator());
    return Convolve(input_descriptor, input_data, filter_descriptor,
                    filter_data, output_descriptor, output_data,
                    convolution_descriptor, best_algorithm_, stream,
                    &scratch_allocator, nullptr);
  }
}

}  // namespace gpu
}  // namespace xla
