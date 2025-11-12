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
#include <optional>
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
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace se = stream_executor;

// With BufferDebugFloatCheckEntry size of 16 bytes, this is enough to hold ~4K
// entries.
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
          Thunk::ThunkInfo(), thunk_ptr->thunk_info(), log_slice,
          std::move(buffers_to_check), std::move(metadata_store));
  buffer_debug_float_check_thunk->add_control_predecessor(thunk_ptr);
  thunk_and_checks.push_back(std::move(buffer_debug_float_check_thunk));
  auto wrapped_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(thunk_and_checks));
  wrapped_thunk->add_control_predecessor(&predecessor_thunk);
  successor_thunk.add_control_predecessor(wrapped_thunk.get());
  return wrapped_thunk;
}

void LogHloInstructionWithId(const HloModule* hlo_module,
                             const std::string& id) {
  for (const HloComputation* computation : hlo_module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->name() == id) {
        LOG(ERROR) << "HLO instruction with id " << id << ":\n\n"
                   << instruction->ToString() << "\n\n";
        if (instruction->opcode() == HloOpcode::kFusion) {
          auto fusion = xla::Cast<HloFusionInstruction>(instruction);
          LOG(ERROR) << "HLO fusion instruction computation:\n\n"
                     << fusion->fused_instructions_computation()->ToString()
                     << "\n\n";
        }
        return;
      }
    }
  }
  LOG(ERROR) << "HLO instruction with id " << id << " was not found";
}

absl::Status BufferDebugFloatCheck(
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store,
    se::Stream* stream, const HloComputation* absl_nonnull hlo_computation,
    xla::ffi::Buffer<U8> log_buffer) {
  VLOG(1) << "HLO computation ptr: " << hlo_computation;
  const HloModule* hlo_module = hlo_computation->parent();
  VLOG(1) << "HLO module ptr: " << hlo_module;
  VLOG(1) << "HLO module name: " << hlo_module->name();
  CHECK(hlo_module != nullptr);
  bool nan_check_enabled =
      hlo_module->config().debug_options().xla_gpu_detect_nan() !=
      DebugOptions::DETECTION_MODE_NONE;
  bool inf_check_enabled =
      hlo_module->config().debug_options().xla_gpu_detect_inf() !=
      DebugOptions::DETECTION_MODE_NONE;

  auto buffer_debug_log = se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::
      FromDeviceMemoryUnchecked(log_buffer.device_memory());
  TF_ASSIGN_OR_RETURN(std::vector<BufferDebugFloatCheckEntry> entries,
                      buffer_debug_log.ReadFromDevice(*stream));

  std::vector<BufferDebugLogEntryId> entry_ids;
  entry_ids.reserve(entries.size());
  for (const auto& entry : entries) {
    entry_ids.push_back(entry.entry_id);
  }

  VLOG(1) << "read " << entries.size() << " entries";
  auto entries_metadata = metadata_store->GetEntryMetadataBatch(entry_ids);
  int non_zero_nan_check_modules_count = 0;
  int non_zero_inf_check_modules_count = 0;
  CHECK_EQ(entries.size(), entries_metadata.size());

  for (int i = 0; i < entries.size(); ++i) {
    const auto& entry = entries[i];
    const auto& metadata = entries_metadata[i];
    if (!metadata.has_value()) {
      LOG(WARNING) << "Entry ID " << entry.entry_id
                   << " for float check not found in metadata";
      continue;
    }
    if (metadata->check_type !=
        BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS) {
      continue;
    }
    if (nan_check_enabled && entry.nan_count > 0) {
      LOG(ERROR) << "Found entry with non zero nan count " << entry.nan_count
                 << " for thunk " << entry.entry_id << " and execution "
                 << "with metadata: " << metadata->profile_annotation;
      non_zero_nan_check_modules_count++;
      LogHloInstructionWithId(hlo_module, metadata->profile_annotation);
    }
    if (inf_check_enabled && entry.inf_count > 0) {
      LOG(ERROR) << "Found entry with non zero inf count " << entry.inf_count
                 << " for thunk " << entry.entry_id << " and execution "
                 << metadata->execution_id
                 << "with metadata: " << metadata->profile_annotation;
      non_zero_inf_check_modules_count++;
      LogHloInstructionWithId(hlo_module, metadata->profile_annotation);
    }
  }
  if (non_zero_nan_check_modules_count > 0 &&
      hlo_module->config().debug_options().xla_gpu_detect_nan() ==
          DebugOptions::DETECTION_MODE_FAIL) {
    LOG(FATAL) << "Crash execution as requested by the xla_gpu_detect_nan flag "
                  "because "
               << non_zero_nan_check_modules_count
               << " NaN values were found in buffers.";
  }
  if (non_zero_inf_check_modules_count > 0 &&
      hlo_module->config().debug_options().xla_gpu_detect_inf() ==
          DebugOptions::DETECTION_MODE_FAIL) {
    LOG(FATAL) << "Crash execution as requested by the xla_gpu_detect_inf flag "
                  "because "
               << non_zero_inf_check_modules_count
               << " infinite values were found in buffers.";
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kBufferDebugFloatCheckLogInitHandler,
    [](se::Stream* absl_nonnull stream, xla::ffi::Buffer<U8> log_buffer) {
      return se::gpu::BufferDebugLog<xla::gpu::BufferDebugFloatCheckEntry>::
          CreateOnDevice(*stream, log_buffer.device_memory())
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
      Thunk::ThunkInfo(), "xla_gpu_buffer_debug_float_check_init",
      buffer_debug_init_bundle, /*operands=*/{shaped_log_slice},
      /*results=*/{}, /*attributes=*/{}, hlo_module->entry_computation());
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>>
CreateBufferDebugFloatCheckThunk(
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
      Thunk::ThunkInfo(), "xla_gpu_buffer_debug_float_check",
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
      CreateBufferDebugFloatCheckThunk(metadata_store, log_slice, hlo_module));

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
