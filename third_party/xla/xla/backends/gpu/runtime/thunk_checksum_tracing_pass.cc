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
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
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
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/dump.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/sdc_log.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace se = stream_executor;

// With SdcLogEntry size of 8 bytes, this is enough to hold ~8K entries.
constexpr size_t kLogSizeBytes = 64 * 1024;

namespace {

// If the thunk has any interesting buffers to check, turns it into a sequence
// of:
// - SdcThunk checking the buffers before execution
// - The original thunk
// - SdcThunk checking the buffers after execution
//
// If the thunk got wrapped, the data dependencies between the thunks will be
// configured to ensure `predecessor_thunk` executes before the wrapped thunk
// and `successor_thunk` executes after.
absl::StatusOr<std::unique_ptr<Thunk>> WrapThunk(
    std::unique_ptr<Thunk> thunk, BufferAllocation::Slice log_slice,
    const Thunk& predecessor_thunk, Thunk& successor_thunk) {
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
    if (use.HasDefinedContentsOnInput()) {
      buffers_to_check_before.emplace(buffer_id.value(), use.slice());
    }
    if (use.HasDefinedContentsOnOutput()) {
      buffers_to_check_after.emplace(buffer_id.value(), use.slice());
    }
  }

  if (buffers_to_check_before.empty() && buffers_to_check_after.empty()) {
    return thunk;
  }

  std::vector<std::unique_ptr<Thunk>> thunk_and_checks;
  if (!buffers_to_check_before.empty()) {
    auto sdc_before_thunk = std::make_unique<SdcThunk>(
        Thunk::ThunkInfo(), log_slice, std::move(buffers_to_check_before));
    thunk->add_control_predecessor(sdc_before_thunk.get());
    thunk_and_checks.push_back(std::move(sdc_before_thunk));
  }

  Thunk* thunk_ptr = thunk.get();
  thunk_and_checks.push_back(std::move(thunk));

  if (!buffers_to_check_after.empty()) {
    auto sdc_after_thunk = std::make_unique<SdcThunk>(
        Thunk::ThunkInfo(), log_slice, std::move(buffers_to_check_after));
    sdc_after_thunk->add_control_predecessor(thunk_ptr);
    thunk_and_checks.push_back(std::move(sdc_after_thunk));
  }

  auto wrapped_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(thunk_and_checks));
  wrapped_thunk->add_control_predecessor(&predecessor_thunk);
  successor_thunk.add_control_predecessor(wrapped_thunk.get());
  return wrapped_thunk;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kDebugLogInitHandler,
    [](se::Stream* absl_nonnull stream, xla::ffi::Buffer<U8> log_buffer) {
      return se::cuda::SdcLog::CreateOnDevice(*stream,
                                              log_buffer.device_memory())
          .status();
    },
    xla::ffi::Ffi::Bind().Ctx<xla::ffi::Stream>().Arg<xla::ffi::Buffer<U8>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kDebugLogDumpHandler,
    [](se::Stream* stream, const HloComputation* absl_nonnull hlo_computation,
       xla::ffi::Buffer<U8> log_buffer) {
      VLOG(1) << "[SDC LOG] HLO computation ptr: " << hlo_computation;
      const HloModule* hlo_module = hlo_computation->parent();
      VLOG(1) << "[SDC LOG] HLO module ptr: " << hlo_module;
      VLOG(1) << "[SDC LOG] HLO module name: " << hlo_module->name();
      CHECK(hlo_module != nullptr);
      const DebugOptions& debug_options = hlo_module->config().debug_options();

      se::cuda::SdcLog sdc_log = se::cuda::SdcLog::FromDeviceMemoryUnchecked(
          log_buffer.device_memory());
      TF_ASSIGN_OR_RETURN(xla::gpu::SdcLogProto sdc_log_proto,
                          sdc_log.ReadProto(*stream));
      VLOG(1) << "[SDC LOG] read " << sdc_log_proto.entries_size()
              << " entries";
      DumpPerExecutionProtobufToFile(*hlo_module, sdc_log_proto, debug_options,
                                     "sdc_log", nullptr);
      return absl::OkStatus();
    },
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::Stream>()
        .Ctx<xla::ffi::CalledComputation>()
        .Arg<xla::ffi::Buffer<U8>>());

}  // namespace

absl::StatusOr<bool> ThunkChecksumTracingPass::Run(
    SequentialThunk* root_thunk, const DebugOptions& debug_options,
    const HloModule* absl_nullable hlo_module,
    const se::DeviceDescription& device_info,
    ThunkPassBufferAllocator& allocator) {
  VLOG(1) << "[SDC LOG] ThunkChecksumTracingPass running";
  if (hlo_module == nullptr) {
    // We need the HLO module to dump the SDC log proto to a file. If it's not
    // available, there's no point in doing extra work.
    VLOG(1) << "[SDC LOG] HLO module is null, skipping";
    return false;
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation * log_alloc,
                      allocator.NewEmptyAllocation(kLogSizeBytes));
  BufferAllocation::Slice log_slice(log_alloc, 0, log_alloc->size());
  ShapedSlice shaped_log_slice{
      /*slice=*/log_slice,
      /*shape=*/Shape(PrimitiveType::U8, /*dimensions=*/{log_alloc->size()}),
  };

  XLA_FFI_Handler_Bundle sdc_init_bundle{};
  sdc_init_bundle.execute = kDebugLogInitHandler;
  TF_ASSIGN_OR_RETURN(
      auto sdc_init_thunk,
      CustomCallThunk::Create(Thunk::ThunkInfo(), "xla_gpu_sdc_log_init",
                              sdc_init_bundle, /*operands=*/{shaped_log_slice},
                              /*results=*/{}, /*attributes=*/{},
                              hlo_module->entry_computation()));

  XLA_FFI_Handler_Bundle sdc_dump_bundle{};
  sdc_dump_bundle.execute = kDebugLogDumpHandler;
  TF_ASSIGN_OR_RETURN(
      auto sdc_dump_thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(), "xla_gpu_sdc_log_dump", sdc_dump_bundle,
          /*operands=*/{shaped_log_slice},
          /*results=*/{}, /*attributes=*/{}, hlo_module->entry_computation()));

  ThunkSequence& thunks = root_thunk->thunks();
  for (auto& thunk : thunks) {
    TF_ASSIGN_OR_RETURN(thunk,
                        WrapThunk(std::move(thunk), log_slice,
                                  /*predecessor_thunk=*/*sdc_init_thunk.get(),
                                  /*successor_thunk=*/*sdc_dump_thunk.get()));
  }

  thunks.reserve(thunks.size() + 2);
  thunks.insert(thunks.begin(), std::move(sdc_init_thunk));
  thunks.push_back(std::move(sdc_dump_thunk));

  return true;
}

}  // namespace xla::gpu
