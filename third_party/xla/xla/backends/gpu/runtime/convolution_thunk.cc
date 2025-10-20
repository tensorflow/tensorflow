/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/convolution_thunk.h"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
using buffer_assignment::BufferAllocationSliceProto;

absl::StatusOr<std::unique_ptr<ConvolutionThunk>> ConvolutionThunk::Create(
    ThunkInfo thunk_info, GpuConvDescriptor descriptor,
    std::vector<BufferAllocation::Slice> operand_slices,
    std::vector<BufferAllocation::Slice> result_slices,
    BufferAllocation::Slice scratch_slice) {
  TF_ASSIGN_OR_RETURN(GpuConvConfig config,
                      GetGpuConvConfig(descriptor, /*inst_as_string=*/""));

  // Can't use std::make_unique because the constructor is private.
  return absl::WrapUnique(new ConvolutionThunk(
      thunk_info, std::move(descriptor), std::move(config),
      std::move(operand_slices), std::move(result_slices), scratch_slice));
}

ConvolutionThunk::ConvolutionThunk(
    ThunkInfo thunk_info, GpuConvDescriptor descriptor, GpuConvConfig config,
    std::vector<BufferAllocation::Slice> operand_slices,
    std::vector<BufferAllocation::Slice> result_slices,
    BufferAllocation::Slice scratch_slice)
    : Thunk(Kind::kConvolution, thunk_info),
      operand_buffers_(std::move(operand_slices)),
      result_buffers_(std::move(result_slices)),
      scratch_buffer_(scratch_slice),
      descriptor_(std::move(descriptor)),
      config_(std::move(config)) {}

GenericConvRunner& ConvolutionThunk::GetOrCreateRunner(
    const stream_executor::Stream* stream, bool* runner_created) {
  absl::MutexLock lock(mu_);
  auto it = runner_cache_.find(stream);
  *runner_created = (it == runner_cache_.end());
  if (*runner_created) {
    it = runner_cache_
             .insert({stream, std::make_unique<GenericConvRunner>(config_)})
             .first;
  }
  return *it->second;
}

absl::Status ConvolutionThunk::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  std::vector<se::DeviceMemoryBase> operand_se_buffers, result_se_buffers;
  operand_se_buffers.reserve(operand_buffers_.size());
  for (BufferAllocation::Slice buffer : operand_buffers_) {
    operand_se_buffers.push_back(buffer_allocations.GetDeviceAddress(buffer));
  }

  result_se_buffers.reserve(result_buffers_.size());
  for (BufferAllocation::Slice buffer : result_buffers_) {
    result_se_buffers.push_back(buffer_allocations.GetDeviceAddress(buffer));
  }

  se::DeviceMemoryBase scratch =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  bool runner_created = false;
  RunConvOptions opts;
  opts.runner_cache = &GetOrCreateRunner(params.stream, &runner_created);

  if (runner_created && params.stream->parent()
                            ->GetDeviceDescription()
                            .gpu_compute_capability()
                            .IsRocm()) {
    TF_ASSIGN_OR_RETURN(
        GpuConvParams conv_params,
        GetGpuConvParams(config_, operand_se_buffers, result_se_buffers));

    TF_ASSIGN_OR_RETURN(se::dnn::DataType input_type,
                        GetDNNDataTypeFromPrimitiveType(config_.input_type));

    TF_ASSIGN_OR_RETURN(se::dnn::DataType output_type,
                        GetDNNDataTypeFromPrimitiveType(config_.output_type));

    TF_ASSIGN_OR_RETURN(auto dnn,
                        se::dnn::internal::GetDnnFromStream(params.stream));
    se::OwningScratchAllocator<> scratch_allocator(
        buffer_allocations.device_ordinal(),
        buffer_allocations.memory_allocator());

    std::vector<se::dnn::ProfileResult> profile_results;
    dnn->GetMIOpenConvolveAlgorithms(
        CudnnConvKindToProto(config_.kind), input_type, output_type,
        params.stream, config_.input_descriptor, conv_params.input_buf,
        config_.filter_descriptor, conv_params.filter_buf,
        config_.output_descriptor, conv_params.output_buf, config_.conv_desc,
        &scratch_allocator, &profile_results);
  }

  TF_RETURN_IF_ERROR(RunGpuConv(config_, absl::MakeSpan(operand_se_buffers),
                                absl::MakeSpan(result_se_buffers), scratch,
                                params.stream, opts));

  // Note: Convolution has a tuple buffer as an output, but we don't need to
  // populate it as no one should be reading from the tuple directly.
  if (!params.stream->ok()) {
    return Internal("ConvolutionThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ConvolutionThunk>> ConvolutionThunk::FromProto(
    ThunkInfo thunk_info, const ConvolutionThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(GpuConvDescriptor descriptor,
                      GpuConvDescriptor::FromProto(proto.conv_descriptor()));

  std::vector<BufferAllocation::Slice> operand_slices;
  operand_slices.reserve(proto.operand_buffers_size());
  for (const BufferAllocationSliceProto& slice_proto :
       proto.operand_buffers()) {
    TF_ASSIGN_OR_RETURN(
        operand_slices.emplace_back(),
        BufferAllocation::Slice::FromProto(slice_proto, buffer_allocations));
  }

  std::vector<BufferAllocation::Slice> result_slices;
  result_slices.reserve(proto.result_buffers_size());
  for (const BufferAllocationSliceProto& slice_proto : proto.result_buffers()) {
    TF_ASSIGN_OR_RETURN(
        result_slices.emplace_back(),
        BufferAllocation::Slice::FromProto(slice_proto, buffer_allocations));
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scratch_slice,
                      BufferAllocation::Slice::FromProto(proto.scratch_buffer(),
                                                         buffer_allocations));

  return Create(std::move(thunk_info), std::move(descriptor),
                std::move(operand_slices), std::move(result_slices),
                scratch_slice);
}

absl::StatusOr<ThunkProto> ConvolutionThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  ConvolutionThunkProto* conv_proto = proto.mutable_convolution_thunk();
  *conv_proto->mutable_conv_descriptor() = descriptor_.ToProto();

  for (const BufferAllocation::Slice& slice : operand_buffers_) {
    TF_ASSIGN_OR_RETURN(*conv_proto->add_operand_buffers(), slice.ToProto());
  }
  for (const BufferAllocation::Slice& slice : result_buffers_) {
    TF_ASSIGN_OR_RETURN(*conv_proto->add_result_buffers(), slice.ToProto());
  }
  TF_ASSIGN_OR_RETURN(*conv_proto->mutable_scratch_buffer(),
                      scratch_buffer_.ToProto());

  return proto;
}

}  // namespace gpu
}  // namespace xla
