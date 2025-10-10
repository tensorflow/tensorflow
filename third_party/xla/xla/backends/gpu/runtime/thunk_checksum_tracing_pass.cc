/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/thunk_checksum_tracing_pass.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/sdc_buffer_id.h"
#include "xla/backends/gpu/runtime/sdc_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/sdc_log.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace se = stream_executor;

constexpr size_t kLogSizeBytes = 64 * 1024;

namespace {

void InitSdcLog(se::Stream* stream, void** data, const char*, size_t,
                XlaCustomCallStatus* call_status) {
  se::DeviceMemory<uint8_t> log_ptr(
      se::DeviceMemoryBase(data[0], kLogSizeBytes));
  absl::StatusOr<se::cuda::SdcLog> result =
      se::cuda::SdcLog::CreateOnDevice(*stream, log_ptr);
  if (!result.ok()) {
    absl::Status status = result.status();
    LOG(ERROR) << "[SDC LOG] Failed to initialize SDC log: " << status;
    XlaCustomCallStatusSetFailure(call_status, status.message().data(),
                                  status.message().size());
    return;
  }
  VLOG(1) << "[SDC LOG] SDC log initialized";
}

// Returns true if given slice is an output of given thunk.
bool IsThunkOutput(const Thunk* thunk, const BufferAllocation::Slice& slice) {
  return absl::c_any_of(thunk->buffer_uses(), [&](const BufferUse& use) {
    return use.slice() == slice && use.HasDefinedContentsOnOutput();
  });
}

// Returns true if given slice is an output of any predecessor of given thunk.
bool IsDataDependency(const Thunk* thunk,
                      const BufferAllocation::Slice& slice) {
  return absl::c_any_of(thunk->control_predecessors(),
                        [&](const Thunk* predecessor) {
                          return IsThunkOutput(predecessor, slice);
                        });
}

// If the thunk has any interesting buffers to check, turns it into a sequence
// of:
// - SdcThunk checking the buffers before execution
// - The original thunk
// - SdcThunk checking the buffers after execution
absl::StatusOr<std::unique_ptr<Thunk>> WrapThunk(
    std::unique_ptr<Thunk> thunk, BufferAllocation::Slice log_slice) {
  const auto& thunk_buffers = thunk->buffer_uses();
  if (thunk_buffers.empty()) {
    return thunk;
  }

  absl::flat_hash_map<SdcBufferId, BufferAllocation::Slice>
      buffers_to_check_before;
  absl::flat_hash_map<SdcBufferId, BufferAllocation::Slice>
      buffers_to_check_after;

  for (size_t buffer_idx = 0; buffer_idx < thunk_buffers.size(); ++buffer_idx) {
    absl::StatusOr<SdcBufferId> buffer_id =
        SdcBufferId::Create(thunk->thunk_info().thunk_id, buffer_idx);
    if (!buffer_id.ok()) {
      LOG(WARNING) << "Skipping buffer " << buffer_idx << " in thunk "
                   << thunk->thunk_info().thunk_id << ": "
                   << buffer_id.status();
      continue;
    }

    const BufferUse& use = thunk_buffers[buffer_idx];
    // Only checksum if the thunk isn't an output of another thunk.
    if (use.HasDefinedContentsOnInput() &&
        !IsDataDependency(thunk.get(), use.slice())) {
      buffers_to_check_before.emplace(buffer_id.value(), use.slice());
    }
    if (use.HasDefinedContentsOnOutput()) {
      buffers_to_check_after.emplace(buffer_id.value(), use.slice());
    }
  }

  if (buffers_to_check_before.empty() && buffers_to_check_after.empty()) {
    return thunk;
  }

  std::vector<std::unique_ptr<Thunk>> sdc_sequence;
  if (!buffers_to_check_before.empty()) {
    auto sdc_before_thunk = std::make_unique<SdcThunk>(
        Thunk::ThunkInfo(), log_slice, std::move(buffers_to_check_before));
    thunk->add_control_predecessor(sdc_before_thunk.get());
    sdc_sequence.push_back(std::move(sdc_before_thunk));
  }

  Thunk* thunk_ptr = thunk.get();
  sdc_sequence.push_back(std::move(thunk));

  if (!buffers_to_check_after.empty()) {
    auto sdc_after_thunk = std::make_unique<SdcThunk>(
        Thunk::ThunkInfo(), log_slice, std::move(buffers_to_check_after));
    sdc_after_thunk->add_control_predecessor(thunk_ptr);
    sdc_sequence.push_back(std::move(sdc_after_thunk));
  }

  return std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                           std::move(sdc_sequence));
}

}  // namespace

absl::StatusOr<bool> ThunkChecksumTracingPass::Run(
    SequentialThunk* root_thunk, const DebugOptions& debug_options,
    const HloModule* hlo_module_ptr, const se::DeviceDescription& device_info,
    ThunkPassBufferAllocator& allocator) {
  VLOG(1) << "[SDC LOG] ThunkChecksumTracingPass running";
  if (hlo_module_ptr == nullptr) {
    VLOG(1) << "[SDC LOG] HLO module is null, skipping";
    return false;
  }
  const HloModule& hlo_module = *hlo_module_ptr;

  TF_ASSIGN_OR_RETURN(BufferAllocation * log_alloc,
                      allocator.NewEmptyAllocation(kLogSizeBytes));
  BufferAllocation::Slice log_slice(log_alloc, 0, log_alloc->size());
  CustomCallThunk::Slice shaped_log_slice(log_slice,
                                          Shape(PrimitiveType::OPAQUE_TYPE));

  ThunkSequence& thunks = root_thunk->thunks();
  TF_ASSIGN_OR_RETURN(
      auto sdc_init_thunk,
      CustomCallThunk::Create(Thunk::ThunkInfo(), "xla_gpu_sdc_log_init",
                              &InitSdcLog, {shaped_log_slice}, {}, ""));
  auto sdc_dump_thunk = std::make_unique<SdcDumpLogThunk>(
      Thunk::ThunkInfo(), log_slice, hlo_module, debug_options);

  for (auto& thunk : thunks) {
    TF_ASSIGN_OR_RETURN(thunk, WrapThunk(std::move(thunk), log_slice));
    thunk->add_control_predecessor(sdc_init_thunk.get());
    sdc_dump_thunk->add_control_predecessor(thunk.get());
  }

  thunks.reserve(thunks.size() + 2);
  thunks.insert(thunks.begin(), std::move(sdc_init_thunk));
  thunks.push_back(std::move(sdc_dump_thunk));

  return true;
}

}  // namespace xla::gpu
