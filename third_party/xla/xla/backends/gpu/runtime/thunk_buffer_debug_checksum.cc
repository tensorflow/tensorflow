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

#include "xla/backends/gpu/runtime/thunk_buffer_debug_checksum.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/buffer_debug_log.pb.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/buffers_checksum_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_filter.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/dump.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace se = stream_executor;

// With BufferDebugLogEntry size of 8 bytes, this is enough to hold ~8K entries.
constexpr size_t kLogSizeBytes = 64 * 1024;

namespace {

// If the thunk has any interesting buffers to check, turns it into a sequence
// of:
// - BuffersDebugChecksumThunk checking the buffers before execution
// - The original thunk
// - BuffersDebugChecksumThunk checking the buffers after execution
//
// If the thunk got wrapped, the data dependencies between the thunks will be
// configured to ensure `predecessor_thunk` executes before the wrapped thunk
// and `successor_thunk` executes after.
//
// If the thunk has no interesting buffers to check, it is returned as is. It
// can never return nullptr.
std::unique_ptr<Thunk> WrapWithChecksumThunk(
    std::unique_ptr<Thunk> thunk, BufferAllocation::Slice log_slice,
    const Thunk& predecessor_thunk, Thunk& successor_thunk,
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store) {
  const auto& thunk_buffers = thunk->buffer_uses();
  if (thunk_buffers.empty()) {
    return thunk;
  }

  absl::flat_hash_map<size_t, BufferAllocation::Slice> buffers_to_check_before;
  absl::flat_hash_map<size_t, BufferAllocation::Slice> buffers_to_check_after;

  for (size_t buffer_idx = 0; buffer_idx < thunk_buffers.size(); ++buffer_idx) {
    const BufferUse& use = thunk_buffers[buffer_idx];
    if (use.HasDefinedContentsOnInput()) {
      buffers_to_check_before.emplace(buffer_idx, use.slice());
    }
    if (use.HasDefinedContentsOnOutput()) {
      buffers_to_check_after.emplace(buffer_idx, use.slice());
    }
  }

  if (buffers_to_check_before.empty() && buffers_to_check_after.empty()) {
    return thunk;
  }

  std::vector<std::unique_ptr<Thunk>> thunk_and_checks;
  if (!buffers_to_check_before.empty()) {
    auto buffer_debug_before_thunk =
        std::make_unique<BuffersDebugChecksumThunk>(
            Thunk::ThunkInfo(), log_slice, thunk->thunk_info().thunk_id,
            std::move(buffers_to_check_before),
            /*runs_before_checked_thunk=*/true, metadata_store);
    thunk->add_control_predecessor(buffer_debug_before_thunk.get());
    thunk_and_checks.push_back(std::move(buffer_debug_before_thunk));
  }

  Thunk* thunk_ptr = thunk.get();
  thunk_and_checks.push_back(std::move(thunk));

  if (!buffers_to_check_after.empty()) {
    auto buffer_debug_after_thunk = std::make_unique<BuffersDebugChecksumThunk>(
        Thunk::ThunkInfo(), log_slice, thunk_ptr->thunk_info().thunk_id,
        std::move(buffers_to_check_after),
        /*runs_before_checked_thunk=*/false, metadata_store);
    buffer_debug_after_thunk->add_control_predecessor(thunk_ptr);
    thunk_and_checks.push_back(std::move(buffer_debug_after_thunk));
  }

  auto wrapped_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(thunk_and_checks));
  wrapped_thunk->add_control_predecessor(&predecessor_thunk);
  successor_thunk.add_control_predecessor(wrapped_thunk.get());
  return wrapped_thunk;
}

// Saves the contents of the BufferDebugLog stored in `log_buffer` to a file..
//
// `metadata_store` is used to retrieve the metadata for the log entries.
// The filename is derived from the HLO module name and the log dump path
// configured in `debug_options`.
absl::Status DumpBufferDebugChecksumLog(
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store,
    se::Stream* stream, const HloComputation* absl_nonnull hlo_computation,
    xla::ffi::Buffer<U8> log_buffer) {
  VLOG(1) << "HLO computation ptr: " << hlo_computation;
  const HloModule* hlo_module = hlo_computation->parent();
  VLOG(1) << "HLO module ptr: " << hlo_module;
  VLOG(1) << "HLO module name: " << hlo_module->name();
  CHECK(hlo_module != nullptr);
  const DebugOptions& debug_options = hlo_module->config().debug_options();

  auto buffer_debug_log =
      se::gpu::BufferDebugLog<BufferDebugLogEntry>::FromDeviceAddressUnchecked(
          log_buffer.device_memory());
  TF_ASSIGN_OR_RETURN(std::vector<BufferDebugLogEntry> log_entries,
                      buffer_debug_log.ReadFromDevice(*stream));
  BufferDebugLogProto buffer_debug_log_proto =
      metadata_store->EntriesToProto(log_entries);

  VLOG(1) << "read " << buffer_debug_log_proto.entries_size() << " entries";
  DumpPerExecutionProtobufToFile(*hlo_module, buffer_debug_log_proto,
                                 debug_options, "buffer_debug_log", nullptr);
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kBufferDebugChecksumLogInitHandler,
    [](se::Stream* absl_nonnull stream, xla::ffi::Buffer<U8> log_buffer) {
      return se::gpu::BufferDebugLog<BufferDebugLogEntry>::CreateOnDevice(
                 *stream, log_buffer.device_memory())
          .status();
    },
    xla::ffi::Ffi::Bind().Ctx<xla::ffi::Stream>().Arg<xla::ffi::Buffer<U8>>());

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CreateDebugInitThunk(
    BufferAllocation::Slice log_slice,
    const HloModule* absl_nonnull hlo_module) {
  ShapedSlice shaped_log_slice{
      /*slice=*/log_slice,
      /*shape=*/Shape(PrimitiveType::U8, /*dimensions=*/{log_slice.size()}),
  };

  XLA_FFI_Handler_Bundle buffer_debug_init_bundle{};
  buffer_debug_init_bundle.execute = kBufferDebugChecksumLogInitHandler;
  return CustomCallThunk::Create(
      Thunk::ThunkInfo(), "xla_gpu_buffer_debug_log_init",
      buffer_debug_init_bundle, /*operands=*/{shaped_log_slice},
      /*results=*/{}, /*attributes=*/{}, hlo_module->entry_computation(),
      se::GpuComputeCapability());
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CreateBufferDebugDumpThunk(
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store,
    BufferAllocation::Slice log_slice,
    const HloModule* absl_nonnull hlo_module) {
  ShapedSlice shaped_log_slice{
      /*slice=*/log_slice,
      /*shape=*/Shape(PrimitiveType::U8, /*dimensions=*/{log_slice.size()}),
  };

  CustomCallThunk::OwnedHandlerBundle dump_bundle{};
  dump_bundle.execute =
      xla::ffi::Ffi::Bind()
          .Ctx<xla::ffi::Stream>()
          .Ctx<xla::ffi::CalledComputation>()
          .Arg<xla::ffi::Buffer<U8>>()
          .To(absl::bind_front(DumpBufferDebugChecksumLog, metadata_store));
  return CustomCallThunk::Create(
      Thunk::ThunkInfo(), "xla_gpu_buffer_debug_log_dump",
      std::move(dump_bundle),
      /*operands=*/{shaped_log_slice},
      /*results=*/{}, /*attributes=*/{}, hlo_module->entry_computation(),
      se::GpuComputeCapability());
}

}  // namespace
absl::Status RunChecksumPassInternal(SequentialThunk* root_thunk,
                                     const DebugOptions& debug_options,
                                     const HloModule* absl_nonnull hlo_module,
                                     ThunkPassBufferAllocator& allocator) {
  std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store =
      std::make_shared<BufferDebugLogEntryMetadataStore>();

  TF_ASSIGN_OR_RETURN(BufferAllocation * log_alloc,
                      allocator.NewEmptyAllocation(kLogSizeBytes));
  BufferAllocation::Slice log_slice(log_alloc, 0, log_alloc->size());

  TF_ASSIGN_OR_RETURN(auto buffer_debug_init_thunk,
                      CreateDebugInitThunk(log_slice, hlo_module));

  TF_ASSIGN_OR_RETURN(
      auto buffer_debug_dump_thunk,
      CreateBufferDebugDumpThunk(metadata_store, log_slice, hlo_module));

  ThunkFilter thunk_filter = CreateThunkFilter(debug_options);
  TF_RETURN_IF_ERROR(
      root_thunk->TransformNested([&](std::unique_ptr<Thunk> thunk) {
        if (thunk_filter(*thunk) == InstrumentAction::kSkip) {
          return thunk;
        }
        VLOG(1) << "Wrapping with checksum thunk";
        return WrapWithChecksumThunk(
            std::move(thunk), log_slice,
            /*predecessor_thunk=*/*buffer_debug_init_thunk,
            /*successor_thunk=*/*buffer_debug_dump_thunk, metadata_store);
      }));

  ThunkSequence& thunks = root_thunk->thunks();
  thunks.reserve(thunks.size() + 2);
  thunks.insert(thunks.begin(), std::move(buffer_debug_init_thunk));
  thunks.push_back(std::move(buffer_debug_dump_thunk));
  return absl::OkStatus();
}

}  // namespace xla::gpu
