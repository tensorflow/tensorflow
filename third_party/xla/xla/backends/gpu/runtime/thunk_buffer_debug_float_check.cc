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

#include "xla/backends/gpu/runtime/thunk_buffer_debug_float_check.h"

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
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/buffers_float_check_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
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
#include "xla/shape.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace se = stream_executor;

// With BufferDebugLogEntry size of 8 bytes, this is enough to hold ~8K entries.
constexpr size_t kLogSizeBytes = 64 * 1024;

namespace {

std::unique_ptr<Thunk> WrapWithFloatCheckThunk(
    std::unique_ptr<Thunk> thunk, BufferAllocation::Slice log_slice,
    const Thunk& predecessor_thunk, Thunk& successor_thunk,
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store) {
  const auto& thunk_buffers = thunk->buffer_uses();
  if (thunk_buffers.empty()) {
    VLOG(1) << "No buffers in thunk " << thunk->thunk_info().thunk_id
            << ", skipping";
    return thunk;
  }

  absl::flat_hash_map<size_t, BufferAllocation::Slice> buffers_to_check;
  for (size_t buffer_idx = 0; buffer_idx < thunk_buffers.size(); ++buffer_idx) {
    VLOG(1) << "Buffer " << buffer_idx << " in thunk "
            << thunk->thunk_info().thunk_id;
    const BufferUse& use = thunk_buffers[buffer_idx];
    const BufferAllocation::Slice& slice = use.slice();
    if (slice.allocation() == nullptr) {
      VLOG(1) << "Buffer " << buffer_idx << " in thunk "
              << thunk->thunk_info().thunk_id
              << " has null allocation, skipping";
      continue;
    }
    if (slice.element_type() != PrimitiveType::F32 &&
        slice.element_type() != PrimitiveType::BF16) {
      VLOG(1) << "Buffer " << buffer_idx << " in thunk "
              << thunk->thunk_info().thunk_id
              << " has unsupported element type "
              << PrimitiveType_Name(slice.element_type()) << ", skipping";
      continue;
    }
    if (!use.HasDefinedContentsOnOutput()) {
      VLOG(1) << "Buffer " << buffer_idx << " in thunk "
              << thunk->thunk_info().thunk_id
              << " has no defined contents, skipping";
      continue;
    }
    buffers_to_check.emplace(buffer_idx, use.slice());
    VLOG(1) << "Found buffer " << buffer_idx << " in thunk "
            << thunk->thunk_info().thunk_id << " with element type "
            << PrimitiveType_Name(slice.element_type()) << " and size "
            << slice.size();
  }

  if (buffers_to_check.empty()) {
    return thunk;
  }

  VLOG(1) << "Wrapping thunk " << thunk->thunk_info().thunk_id
          << " with float check thunk due to presence of buffers: "
          << buffers_to_check.size();
  std::vector<std::unique_ptr<Thunk>> thunk_and_checks;
  Thunk* thunk_ptr = thunk.get();
  thunk_and_checks.push_back(std::move(thunk));
  auto buffer_debug_float_check_thunk =
      std::make_unique<BuffersDebugFloatCheckThunk>(
          Thunk::ThunkInfo(), log_slice, thunk_ptr->thunk_info().thunk_id,
          std::move(buffers_to_check),
          /*runs_before_checked_thunk=*/false, std::move(metadata_store));
  buffer_debug_float_check_thunk->add_control_predecessor(thunk_ptr);
  thunk_and_checks.push_back(std::move(buffer_debug_float_check_thunk));
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
absl::Status BufferDebugFloatCheck(
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store,
    se::Stream* stream, const HloComputation* absl_nonnull hlo_computation,
    xla::ffi::Buffer<U8> log_buffer) {
  VLOG(1) << "HLO computation ptr: " << hlo_computation;
  const HloModule* hlo_module = hlo_computation->parent();
  VLOG(1) << "HLO module ptr: " << hlo_module;
  VLOG(1) << "HLO module name: " << hlo_module->name();
  CHECK(hlo_module != nullptr);
  const DebugOptions& debug_options = hlo_module->config().debug_options();

  se::gpu::BufferDebugLog buffer_debug_log =
      se::gpu::BufferDebugLog::FromDeviceMemoryUnchecked(
          log_buffer.device_memory());
  TF_ASSIGN_OR_RETURN(
      std::vector<BufferDebugLogEntry> log_entries,
      buffer_debug_log.ReadFromDevice<BufferDebugLogEntry>(*stream));
  BufferDebugLogProto buffer_debug_log_proto =
      metadata_store->EntriesToProto(log_entries);

  VLOG(1) << "read " << buffer_debug_log_proto.entries_size() << " entries";
  DumpPerExecutionProtobufToFile(*hlo_module, buffer_debug_log_proto,
                                 debug_options, "buffer_debug_log", nullptr);
  int non_zero_float_check_modules_count = 0;
  for (const auto& entry : buffer_debug_log_proto.entries()) {
    if (entry.check_type() ==
            BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS &&
        entry.checksum() > 0) {
      LOG(ERROR) << "Found entry with non zero float check count "
                 << entry.checksum() << " for thunk " << entry.thunk_id()
                 << " and execution " << entry.execution_id()
                 << " for module: \n"
                 << hlo_module->ToString();
      non_zero_float_check_modules_count++;
    }
  }
  if (non_zero_float_check_modules_count > 0 &&
      hlo_module->config().debug_options().xla_gpu_detect_nan() ==
          DebugOptions::NAN_CHECK_DETECTION_MODE_FAIL) {
    LOG(FATAL) << "Found " << non_zero_float_check_modules_count
               << " modules with non zero float check count";
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kBufferDebugFloatCheckLogInitHandler,
    [](se::Stream* absl_nonnull stream, xla::ffi::Buffer<U8> log_buffer) {
      return se::gpu::BufferDebugLog::CreateOnDevice<BufferDebugLogEntry>(
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
  buffer_debug_init_bundle.execute = kBufferDebugFloatCheckLogInitHandler;
  return CustomCallThunk::Create(
      Thunk::ThunkInfo(), "xla_gpu_buffer_debug_log_init",
      buffer_debug_init_bundle, /*operands=*/{shaped_log_slice},
      /*results=*/{}, /*attributes=*/{}, hlo_module->entry_computation());
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CreateBufferDebugDumpThunk(
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store,
    BufferAllocation::Slice log_slice,
    const HloModule* absl_nonnull hlo_module) {
  ShapedSlice shaped_log_slice{
      /*slice=*/log_slice,
      /*shape=*/Shape(PrimitiveType::U8, /*dimensions=*/{log_slice.size()}),
  };

  CustomCallThunk::OwnedHandlerBundle float_check_bundle{};
  float_check_bundle.execute =
      xla::ffi::Ffi::Bind()
          .Ctx<xla::ffi::Stream>()
          .Ctx<xla::ffi::CalledComputation>()
          .Arg<xla::ffi::Buffer<U8>>()
          .To(absl::bind_front(BufferDebugFloatCheck, metadata_store));
  return CustomCallThunk::Create(
      Thunk::ThunkInfo(), "xla_gpu_buffer_debug_log_dump",
      std::move(float_check_bundle),
      /*operands=*/{shaped_log_slice},
      /*results=*/{}, /*attributes=*/{}, hlo_module->entry_computation());
}

}  // namespace

absl::Status RunFloatCheckPassInternal(SequentialThunk* root_thunk,
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
  root_thunk->TransformAllNestedThunks([&](std::unique_ptr<Thunk> thunk) {
    if (thunk_filter(*thunk) == InstrumentAction::kSkip) {
      return thunk;
    }
    VLOG(1) << "Wrapping with float check thunk";
    return WrapWithFloatCheckThunk(
        std::move(thunk), log_slice,
        /*predecessor_thunk=*/*buffer_debug_init_thunk,
        /*successor_thunk=*/*buffer_debug_dump_thunk, metadata_store);
  });

  ThunkSequence& thunks = root_thunk->thunks();
  thunks.reserve(thunks.size() + 2);
  thunks.insert(thunks.begin(), std::move(buffer_debug_init_thunk));
  thunks.push_back(std::move(buffer_debug_dump_thunk));
  return absl::OkStatus();
}

}  // namespace xla::gpu
