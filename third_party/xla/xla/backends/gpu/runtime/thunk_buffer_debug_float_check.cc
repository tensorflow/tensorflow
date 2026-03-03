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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/buffer_debug_log.pb.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/buffers_float_check_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
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
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace se = stream_executor;

// With BufferDebugFloatCheckEntry size of 16 bytes, this is enough to hold ~4K
// entries.
constexpr size_t kLogSizeBytes = 64 * 1024;

namespace {

size_t CalculateTempBufferSize(const Thunk& thunk) {
  size_t max_buffer_size_bytes = 0;
  for (const BufferUse& use : thunk.buffer_uses()) {
    if (use.HasDefinedContentsOnOutput()) {
      max_buffer_size_bytes =
          std::max<size_t>(max_buffer_size_bytes, use.slice().size());
    }
  }

  // We're doing the float checks in 2 steps:
  // - parallel aggregation: one thread block writes partial result into the
  //   temp buffer. The number of thread blocks used will be limtied by the size
  //   calculated here.
  // - reduction of the temp buffer on a single thread block
  // To optimize for time, we want to do as much computation in parallel as we
  // can, but also consider the overhead of single-block reduction step.

  // Avoid making the reduction step use less than a block's worth of data. We
  // can't go any faster than that anyway.
  static constexpr size_t kMinElements = 1024;
  // Arbitrary limit of 1Mi elements. This should be enough to accomodate the
  // max number of thread blocks available on any supported GPU.
  static constexpr size_t kMaxElements = 1024 * 1024;
  const size_t size_elems =
      xla::CeilOfRatio(max_buffer_size_bytes, sizeof(uint32_t));
  const size_t sqrt_size_elems = std::sqrt(size_elems);
  return std::clamp(xla::CeilOfRatio(size_elems, sqrt_size_elems), kMinElements,
                    kMaxElements);
}

absl::StatusOr<std::unique_ptr<Thunk>> WrapWithFloatCheckThunk(
    std::unique_ptr<Thunk> thunk, BufferAllocation::Slice log_slice,
    const Thunk& predecessor_thunk, Thunk& successor_thunk,
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store,
    ThunkPassBufferAllocator& allocator) {
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
        slice.element_type() != PrimitiveType::BF16 &&
        slice.element_type() != PrimitiveType::F64) {
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

  const size_t temp_buffer_size_bytes =
      CalculateTempBufferSize(*thunk) * sizeof(xla::gpu::FloatCheckResult);
  TF_ASSIGN_OR_RETURN(BufferAllocation * tmp_alloc,
                      allocator.NewEmptyAllocation(temp_buffer_size_bytes));
  BufferAllocation::Slice tmp_slice(tmp_alloc, 0, tmp_alloc->size());

  VLOG(1) << "Wrapping thunk " << thunk->thunk_info().thunk_id
          << " with float check thunk due to presence of buffers: "
          << buffers_to_check.size();
  std::vector<std::unique_ptr<Thunk>> thunk_and_checks;
  Thunk* thunk_ptr = thunk.get();
  thunk_and_checks.push_back(std::move(thunk));
  auto buffer_debug_float_check_thunk =
      std::make_unique<BuffersDebugFloatCheckThunk>(
          Thunk::ThunkInfo(), thunk_ptr->thunk_info(), log_slice, tmp_slice,
          std::move(buffers_to_check), std::move(metadata_store));
  buffer_debug_float_check_thunk->add_control_predecessor(thunk_ptr);
  thunk_and_checks.push_back(std::move(buffer_debug_float_check_thunk));
  auto wrapped_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(thunk_and_checks));
  wrapped_thunk->add_control_predecessor(&predecessor_thunk);
  successor_thunk.add_control_predecessor(wrapped_thunk.get());
  return wrapped_thunk;
}

const HloInstruction* FindHloInstructionWithId(const HloModule* hlo_module,
                                               const std::string& id) {
  for (const HloComputation* computation : hlo_module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->name() == id) {
        return instruction;
      }
    }
  }
  return nullptr;
}

struct EnabledChecks {
  // Should log found NaNs?
  bool check_nans = false;
  // Should crash on found NaNs?
  bool check_nans_fatal = false;
  // Should log found Infs?
  bool check_infs = false;
  // Should crash on found Infs?
  bool check_infs_fatal = false;
  // Should log min/max values from buffers?
  bool log_minmax = false;
};

struct FloatCheckReportResult {
  // Did we report finding a NaN?
  bool reported_nan = false;
  // Did we report finding an Inf?
  bool reported_inf = false;
};

// Print a report for the given `entry` if `enabled_checks` requires it.
// If a fatal check fails, set `*out_should_crash` to true, otherwise leave it
// unchanged.
//
// `reported_nans` and `reported_infs` are caches used to avoid printing the
// same instruction multiple times.
// `enabled_checks` is a bitmask of `EnabledChecks` enum values.
FloatCheckReportResult ReportFloatCheckResult(
    const HloModule* hlo_module, const BufferDebugFloatCheckEntry& entry,
    const BufferDebugLogEntryMetadataStore::Metadata& metadata,
    const EnabledChecks& enabled_checks,
    const std::unordered_set<std::string>& reported_nans,
    const std::unordered_set<std::string>& reported_infs) {
  if (metadata.check_type !=
      BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS) {
    VLOG(1) << "Entry ID " << entry.entry_id
            << " for float check has unsupported check type "
            << BufferDebugLogEntryProto::CheckType_Name(metadata.check_type);
    return {};
  }

  const bool has_nans = entry.result.nan_count > 0;
  const bool has_infs = entry.result.inf_count > 0;

  if (!(enabled_checks.check_nans && has_nans &&
        !absl::c_contains(reported_nans, metadata.profile_annotation)) &&
      !(enabled_checks.check_infs && has_infs &&
        !absl::c_contains(reported_infs, metadata.profile_annotation)) &&
      !enabled_checks.log_minmax) {
    VLOG(2) << "No findings for enabled checks for entry ID " << entry.entry_id;
    return {};
  }

  // Short summary
  LOG(ERROR) << "Float check: "
             << (enabled_checks.check_nans && has_nans ? "found NaN, " : "")
             << (enabled_checks.check_infs && has_infs ? "found Inf, " : "")
             << "entry ID " << entry.entry_id << ", module "
             << hlo_module->name() << " (ID: " << hlo_module->unique_id()
             << "), execution with metadata: " << metadata.profile_annotation
             << ", result: " << entry.result;

  // Instruction/module details
  const HloInstruction* instruction =
      FindHloInstructionWithId(hlo_module, metadata.profile_annotation);
  if (!instruction) {
    LOG(ERROR) << "HLO instruction with id " << metadata.profile_annotation
               << " was not found";
  } else {
    LOG(ERROR) << "In HLO instruction with id " << metadata.profile_annotation
               << " of HLO module " << hlo_module->name() << ":\nStack trace:\n"
               << instruction->GetStackTraceStringFromMetadata(4) << "\n\n"
               << instruction->ToString() << "\n\n";
    if (instruction->opcode() == HloOpcode::kFusion) {
      auto fusion = xla::Cast<HloFusionInstruction>(instruction);
      LOG(ERROR) << "HLO fusion instruction computation:\n\n"
                 << fusion->fused_instructions_computation()->ToString()
                 << "\n\n";
    }
  }

  return FloatCheckReportResult{
      /*reported_nan=*/enabled_checks.check_nans && has_nans,
      /*reported_inf=*/enabled_checks.check_infs && has_infs,
  };
}

EnabledChecks GetEnabledChecks(const HloModule* absl_nonnull hlo_module) {
  DebugOptions::DetectionMode nan_detection_mode =
      hlo_module->config().debug_options().xla_gpu_detect_nan();
  DebugOptions::DetectionMode inf_detection_mode =
      hlo_module->config().debug_options().xla_gpu_detect_inf();

  return EnabledChecks{
      /*check_nans=*/nan_detection_mode != DebugOptions::DETECTION_MODE_NONE,
      /*check_nans_fatal=*/nan_detection_mode ==
          DebugOptions::DETECTION_MODE_FAIL,
      /*check_infs=*/inf_detection_mode != DebugOptions::DETECTION_MODE_NONE,
      /*check_infs_fatal=*/inf_detection_mode ==
          DebugOptions::DETECTION_MODE_FAIL,
      /*log_minmax=*/
      hlo_module->config().debug_options().xla_gpu_log_minmax(),
  };
}

absl::Status BufferDebugFloatCheck(
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store,
    se::Stream* stream, const HloComputation* absl_nonnull hlo_computation,
    xla::ffi::Buffer<PrimitiveType::U8> log_buffer) {
  VLOG(1) << "HLO computation ptr: " << hlo_computation;
  const HloModule* hlo_module = hlo_computation->parent();
  VLOG(1) << "HLO module ptr: " << hlo_module;
  VLOG(1) << "HLO module name: " << hlo_module->name();
  CHECK(hlo_module != nullptr);

  const EnabledChecks enabled_checks = GetEnabledChecks(hlo_module);

  auto buffer_debug_log = se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::
      FromDeviceAddressUnchecked(log_buffer.device_memory());
  TF_ASSIGN_OR_RETURN(std::vector<BufferDebugFloatCheckEntry> entries,
                      buffer_debug_log.ReadFromDevice(*stream));

  std::vector<BufferDebugLogEntryId> entry_ids;
  entry_ids.reserve(entries.size());
  for (const auto& entry : entries) {
    entry_ids.push_back(entry.entry_id);
  }

  VLOG(1) << "read " << entries.size() << " entries";
  auto entries_metadata = metadata_store->GetEntryMetadataBatch(entry_ids);
  CHECK_EQ(entries.size(), entries_metadata.size());

  std::unordered_set<std::string> reported_nans;
  std::unordered_set<std::string> reported_infs;

  bool should_crash = false;
  for (int i = 0; i < entries.size(); ++i) {
    const auto& entry = entries[i];
    const auto& metadata = entries_metadata[i];
    if (!metadata.has_value()) {
      VLOG(1) << "Entry ID " << entry.entry_id
              << " for float check not found in metadata";
      continue;
    }
    const FloatCheckReportResult result =
        ReportFloatCheckResult(hlo_module, entry, *metadata, enabled_checks,
                               reported_nans, reported_infs);
    if (result.reported_nan) {
      reported_nans.insert(metadata->profile_annotation);
    }
    if (result.reported_inf) {
      reported_infs.insert(metadata->profile_annotation);
    }
    if ((result.reported_nan && enabled_checks.check_nans_fatal) ||
        (result.reported_inf && enabled_checks.check_infs_fatal)) {
      should_crash = true;
    }
  }

  if (should_crash) {
    LOG(FATAL) << "Float check failed, aborting.";
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kBufferDebugFloatCheckLogInitHandler,
    [](se::Stream* absl_nonnull stream,
       xla::ffi::Buffer<PrimitiveType::U8> log_buffer) {
      return se::gpu::BufferDebugLog<xla::gpu::BufferDebugFloatCheckEntry>::
          CreateOnDevice(*stream, log_buffer.device_memory())
              .status();
    },
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::Stream>()
        .Arg<xla::ffi::Buffer<PrimitiveType::U8>>());

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
      /*results=*/{}, /*attributes=*/{}, hlo_module->entry_computation(),
      se::GpuComputeCapability());
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
          .Arg<xla::ffi::Buffer<PrimitiveType::U8>>()
          .To(absl::bind_front(BufferDebugFloatCheck, metadata_store));
  return CustomCallThunk::Create(
      Thunk::ThunkInfo(), "xla_gpu_buffer_debug_float_check",
      std::move(float_check_bundle),
      /*operands=*/{shaped_log_slice},
      /*results=*/{}, /*attributes=*/{}, hlo_module->entry_computation(),
      se::GpuComputeCapability());
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
  TF_RETURN_IF_ERROR(root_thunk->TransformNested(
      [&](std::unique_ptr<Thunk> thunk)
          -> absl::StatusOr<std::unique_ptr<Thunk>> {
        if (thunk_filter(*thunk) == InstrumentAction::kSkip) {
          return thunk;
        }
        VLOG(1) << "Wrapping with float check thunk";
        return WrapWithFloatCheckThunk(
            std::move(thunk), log_slice,
            /*predecessor_thunk=*/*buffer_debug_init_thunk,
            /*successor_thunk=*/*buffer_debug_dump_thunk, metadata_store,
            allocator);
      }));

  ThunkSequence& thunks = root_thunk->thunks();
  thunks.reserve(thunks.size() + 2);
  thunks.insert(thunks.begin(), std::move(buffer_debug_init_thunk));
  thunks.push_back(std::move(buffer_debug_dump_thunk));
  return absl::OkStatus();
}

}  // namespace xla::gpu
