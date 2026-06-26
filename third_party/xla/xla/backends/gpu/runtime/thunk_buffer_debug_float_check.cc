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
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/buffer_debug_log.pb.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/buffers_float_check_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_filter.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_pass.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/env.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla::gpu {

namespace se = stream_executor;

// With BufferDebugFloatCheckEntry size of 16 bytes, this is enough to hold ~4K
// entries.
constexpr size_t kLogSizeBytes = 64 * 1024;

namespace {

bool IsFloatTypeSupportedByChecker(PrimitiveType type) {
  return type == PrimitiveType::F32 || type == PrimitiveType::BF16 ||
         type == PrimitiveType::F16 || type == PrimitiveType::F64;
}

size_t TempBufferSizeFromMaxBufferSize(size_t max_buffer_size_bytes) {
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

size_t CalculateTempBufferSize(const Thunk& thunk) {
  size_t max_buffer_size_bytes = 0;
  for (const BufferUse& use : thunk.buffer_uses()) {
    if (use.HasDefinedContentsOnOutput()) {
      max_buffer_size_bytes =
          std::max<size_t>(max_buffer_size_bytes, use.slice().size());
    }
  }
  return TempBufferSizeFromMaxBufferSize(max_buffer_size_bytes);
}

// Finds an HLO instruction by its name. This is an O(N) search and should only
// be used in cold paths (e.g. anomaly reporting).
const HloInstruction* FindHloInstructionWithId(const HloModule* hlo_module,
                                               absl::string_view name) {
  if (hlo_module == nullptr) return nullptr;
  for (const HloComputation* computation : hlo_module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->name() == name) {
        return instruction;
      }
    }
  }
  return nullptr;
}

// Copies device memory for all read-write (inout) buffers to host memory.
//
// The order of elements in `inout_buffers` must match the order of buffers
// passed through `remaining_args`.
absl::Status BackupBuffers(
    std::vector<BufferAllocation::Slice> inout_buffers,
    std::shared_ptr<absl::flat_hash_map<BufferAllocation::Slice,
                                        std::vector<uint8_t>>> absl_nonnull
    backup_data,
    se::Stream* stream, xla::ffi::RemainingArgs remaining_args) {
  backup_data->clear();
  backup_data->reserve(remaining_args.size());
  for (size_t i = 0; i < remaining_args.size(); ++i) {
    if (i >= inout_buffers.size()) {
      return absl::InternalError("Mismatch in backup buffer count");
    }
    ASSIGN_OR_RETURN(auto any_buf, remaining_args.get<xla::ffi::AnyBuffer>(i));
    if (any_buf.size_bytes() == 0) continue;
    const BufferAllocation::Slice& slice = inout_buffers[i];
    std::vector<uint8_t>& storage = (*backup_data)[slice];
    storage.resize(any_buf.size_bytes());
    se::DeviceMemoryBase active_ptr(const_cast<void*>(any_buf.untyped_data()),
                                    any_buf.size_bytes());
    // TODO(b/485867926): Investigate backing up the buffers in device memory
    // for efficiency.
    RETURN_IF_ERROR(
        stream->Memcpy(storage.data(), active_ptr, any_buf.size_bytes()));
  }
  return absl::OkStatus();
}

absl::Status DumpHloSnapshot(
    se::Stream* stream, const HloInstruction* instr,
    const std::vector<BufferAllocation::Slice>& operand_slices,
    const absl::flat_hash_map<BufferAllocation::Slice, std::vector<uint8_t>>&
        backup_map,
    xla::ffi::RemainingArgs remaining_args, const std::string& crash_dump_dir,
    absl::string_view profile_annotation) {
  HloSnapshot snapshot;
  snapshot.set_execution_platform("cuda");

  std::unique_ptr<HloModule> dump_module = std::make_unique<HloModule>(
      "crash_dump_module", HloModuleConfig(instr->GetModule()->config()));

  HloCloneContext context(dump_module.get());
  HloComputation::Builder builder(instr->name());
  std::vector<HloInstruction*> params_list;
  params_list.reserve(instr->operand_count());
  for (int64_t i = 0; i < instr->operand_count(); ++i) {
    params_list.push_back(
        builder.AddInstruction(HloInstruction::CreateParameter(
            i, instr->operand(i)->shape(), absl::StrCat("param_", i))));
  }
  builder.AddInstruction(
      instr->CloneWithNewOperands(instr->shape(), params_list, &context));
  dump_module->AddEntryComputation(builder.Build());
  dump_module->mutable_config().SetDefaultComputationLayout(
      dump_module->entry_computation()->ComputeProgramShape());

  *snapshot.mutable_hlo()->mutable_hlo_module() = dump_module->ToProto();

  std::vector<Literal> literals;
  literals.reserve(instr->operand_count());

  size_t arg_index = 0;
  for (int64_t i = 0; i < instr->operand_count(); ++i) {
    const HloInstruction* op = instr->operand(i);
    literals.push_back(Literal(op->shape()));
    Literal& literal = literals.back();
    absl::Status status = ShapeUtil::ForEachSubshapeWithStatus(
        op->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
          if (subshape.IsArray()) {
            if (arg_index >= operand_slices.size()) {
              return absl::InternalError("Not enough operand slices");
            }
            BufferAllocation::Slice slice = operand_slices[arg_index];
            ASSIGN_OR_RETURN(
                auto any_buf,
                remaining_args.get<xla::ffi::AnyBuffer>(arg_index));
            arg_index++;
            if (any_buf.size_bytes() == 0) return absl::OkStatus();

            if (auto it = backup_map.find(slice);
                it != backup_map.end() &&
                it->second.size() >= any_buf.size_bytes()) {
              std::memcpy(literal.untyped_data(index), it->second.data(),
                          any_buf.size_bytes());
            } else {
              se::DeviceMemoryBase active_ptr(
                  const_cast<void*>(any_buf.untyped_data()),
                  any_buf.size_bytes());
              RETURN_IF_ERROR(stream->Memcpy(literal.untyped_data(index),
                                             active_ptr, any_buf.size_bytes()));
            }
          }
          return absl::OkStatus();
        });
    RETURN_IF_ERROR(status);
  }

  RETURN_IF_ERROR(stream->BlockHostUntilDone());
  for (const Literal& literal : literals) {
    *snapshot.add_arguments() = literal.ToProto();
  }

  tsl::Env* env = tsl::Env::Default();
  RETURN_IF_ERROR(env->RecursivelyCreateDir(crash_dump_dir));
  std::string filename = tsl::io::JoinPath(
      crash_dump_dir,
      absl::StrCat("dump_event_",
                   SanitizeFileName(std::string(profile_annotation)),
                   ".snapshot.pb"));
  RETURN_IF_ERROR(
      tsl::WriteStringToFile(env, filename, snapshot.SerializeAsString()));

  return absl::OkStatus();
}

bool HasAnomaly(const std::vector<BufferDebugFloatCheckEntry>& entries,
                const DebugOptions& debug_options) {
  const bool check_nan =
      debug_options.xla_gpu_detect_nan() != DebugOptions::DETECTION_MODE_NONE;
  const bool check_inf =
      debug_options.xla_gpu_detect_inf() != DebugOptions::DETECTION_MODE_NONE;
  if (!check_nan && !check_inf) return false;

  for (const auto& entry : entries) {
    if (check_nan && entry.result.nan_count > 0) return true;
    if (check_inf && entry.result.inf_count > 0) return true;
  }
  return false;
}

// Captures an HLO snapshot and writes it to a crash dump directory if a
// numerical anomaly (NaN or Inf) is detected.
//
// The `backup_data` shared pointer connects this function with the preceding
// `BackupBuffers` custom call thunk. `BackupBuffers` runs before the checked
// thunk executes, saving copies of all read-write (inout) buffers into
// `backup_data`. If an anomaly is detected, this function uses those backed-up
// contents for snapshot reconstruction; otherwise, it clears `backup_data` to
// release host memory before the next thunk execution.
absl::Status FloatCheckCrashDump(
    absl::string_view profile_annotation,
    std::shared_ptr<absl::flat_hash_map<BufferAllocation::Slice,
                                        std::vector<uint8_t>>> absl_nonnull
    backup_data,
    const HloInstruction* absl_nonnull instr,
    std::vector<BufferAllocation::Slice> operand_slices, se::Stream* stream,
    xla::ffi::Buffer<PrimitiveType::U8> log_buffer,
    xla::ffi::RemainingArgs remaining_args) {
  auto buffer_debug_log = se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::
      FromDeviceAddressUnchecked(log_buffer.device_memory());
  ASSIGN_OR_RETURN(std::vector<BufferDebugFloatCheckEntry> entries,
                   buffer_debug_log.ReadFromDevice(*stream));
  RETURN_IF_ERROR(buffer_debug_log.Clear(*stream));

  if (HasAnomaly(entries, instr->GetModule()->config().debug_options())) {
    const std::string& dump_to =
        instr->GetModule()->config().debug_options().xla_dump_to();
    LOG(ERROR) << "Numerical anomaly detected in thunk " << profile_annotation
               << "; capturing HLO snapshot to " << dump_to;

    std::string crash_dump_dir = tsl::io::JoinPath(dump_to, "crash_dump");

    absl::Status status =
        DumpHloSnapshot(stream, instr, operand_slices, *backup_data,
                        remaining_args, crash_dump_dir, profile_annotation);
    if (status.ok()) {
      LOG(FATAL) << "Float check crash dump generated to directory: "
                 << crash_dump_dir << " for thunk " << profile_annotation;
    } else {
      LOG(FATAL) << "Float check crash dump failed to generate for thunk "
                 << profile_annotation << ": " << status;
    }
  }

  backup_data->clear();

  return absl::OkStatus();
}

struct ThunkBuffersToCheckAndBackup {
  absl::flat_hash_map<size_t, BufferAllocation::Slice> buffers_to_check;
  std::vector<BufferAllocation::Slice> buffers_to_backup;
};

ThunkBuffersToCheckAndBackup GetBuffersToCheckOrBackup(const Thunk& thunk) {
  ThunkBuffersToCheckAndBackup result;
  const auto& thunk_buffers = thunk.buffer_uses();
  for (size_t buffer_idx = 0; buffer_idx < thunk_buffers.size(); ++buffer_idx) {
    VLOG(1) << "Buffer " << buffer_idx << " in thunk "
            << thunk.thunk_info().thunk_id;
    const BufferUse& use = thunk_buffers[buffer_idx];
    const BufferAllocation::Slice& slice = use.slice();
    if (slice.allocation() == nullptr) {
      VLOG(1) << "Buffer " << buffer_idx << " in thunk "
              << thunk.thunk_info().thunk_id
              << " has null allocation, skipping";
      continue;
    }
    if (use.HasDefinedContentsOnOutput()) {
      if (IsFloatTypeSupportedByChecker(slice.element_type())) {
        result.buffers_to_check.emplace(buffer_idx, slice);
        VLOG(1) << "Found buffer " << buffer_idx << " in thunk "
                << thunk.thunk_info().thunk_id << " with element type "
                << PrimitiveType_Name(slice.element_type()) << " and size "
                << slice.size();
      } else {
        VLOG(1) << "Buffer " << buffer_idx << " in thunk "
                << thunk.thunk_info().thunk_id
                << " has unsupported element type "
                << PrimitiveType_Name(slice.element_type()) << ", skipping";
      }
    } else {
      VLOG(1) << "Buffer " << buffer_idx << " in thunk "
              << thunk.thunk_info().thunk_id
              << " has no defined contents, skipping";
    }
    if (use.access() == BufferUse::MemoryAccess::kWrite &&
        use.HasDefinedContentsOnInput()) {
      result.buffers_to_backup.push_back(slice);
    }
  }
  return result;
}

absl::StatusOr<std::vector<NullableShapedSlice>> GetInstructionOperands(
    const HloInstruction* instr, const BufferAssignment* buffer_assignment) {
  std::vector<NullableShapedSlice> operands;
  operands.reserve(instr->operand_count());
  for (int64_t i = 0; i < instr->operand_count(); ++i) {
    const HloInstruction* op = instr->operand(i);
    absl::Status status = ShapeUtil::ForEachSubshapeWithStatus(
        op->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
          if (subshape.IsArray()) {
            ASSIGN_OR_RETURN(auto slice,
                             buffer_assignment->GetUniqueSlice(op, index));
            operands.push_back(ShapedSlice{slice, subshape});
          }
          return absl::OkStatus();
        });
    RETURN_IF_ERROR(status);
  }
  return operands;
}

// Wraps a thunk with synchronous float checking, snapshot dumping on anomaly,
// and read-write buffer backup/restoration.
absl::StatusOr<std::unique_ptr<Thunk>> WrapWithSyncDumpThunk(
    std::unique_ptr<Thunk> thunk,
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store,
    BufferAllocation::Slice log_slice, const HloModule* absl_nonnull hlo_module,
    const BufferAssignment* absl_nonnull buffer_assignment,
    ThunkPassBufferAllocator& allocator,
    const absl::flat_hash_map<absl::string_view, const HloInstruction*>&
        hlo_instruction_map) {
  if (thunk->buffer_uses().empty()) {
    VLOG(3) << "Skipping sync dump wrapping for thunk "
            << thunk->thunk_info().thunk_id << ": no buffers used";
    return thunk;
  }

  auto [buffers_to_check, inout_buffers] = GetBuffersToCheckOrBackup(*thunk);

  if (buffers_to_check.empty()) {
    VLOG(3) << "Skipping sync dump wrapping for thunk "
            << thunk->thunk_info().thunk_id
            << ": no float-typed output buffers to check";
    return thunk;
  }

  std::string profile_annotation = thunk->thunk_info().profile_annotation;

  const HloInstruction* instr = nullptr;
  if (auto it = hlo_instruction_map.find(profile_annotation);
      it != hlo_instruction_map.end()) {
    instr = it->second;
  }
  if (instr == nullptr) {
    LOG(WARNING) << "Skipping sync dump wrapping for thunk "
                 << thunk->thunk_info().thunk_id
                 << ": HLO instruction not found for annotation "
                 << profile_annotation;
    return thunk;
  }

  const size_t temp_buffer_size_bytes =
      CalculateTempBufferSize(*thunk) * sizeof(xla::gpu::FloatCheckResult);
  ASSIGN_OR_RETURN(BufferAllocation * tmp_alloc,
                   allocator.NewEmptyAllocation(temp_buffer_size_bytes));
  BufferAllocation::Slice tmp_slice(tmp_alloc, 0, tmp_alloc->size());

  ASSIGN_OR_RETURN(std::vector<NullableShapedSlice> operands,
                   GetInstructionOperands(instr, buffer_assignment));

  std::vector<BufferAllocation::Slice> operand_slices;
  operand_slices.reserve(operands.size());
  absl::c_transform(operands, std::back_inserter(operand_slices),
                    [](const NullableShapedSlice& op) { return op->slice; });

  operands.insert(
      operands.begin(),
      ShapedSlice{log_slice, Shape(PrimitiveType::U8, {log_slice.size()})});

  std::shared_ptr<
      absl::flat_hash_map<BufferAllocation::Slice, std::vector<uint8_t>>>
      backup_data = std::make_shared<
          absl::flat_hash_map<BufferAllocation::Slice, std::vector<uint8_t>>>();

  CustomCallThunk::OwnedHandlerBundle dump_bundle{};
  dump_bundle.execute =
      xla::ffi::Ffi::Bind()
          .Ctx<xla::ffi::Stream>()
          .Arg<xla::ffi::Buffer<PrimitiveType::U8>>()
          .RemainingArgs()
          .To(absl::bind_front(FloatCheckCrashDump, profile_annotation,
                               backup_data, instr, std::move(operand_slices)));

  ASSIGN_OR_RETURN(
      auto dump_thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(), "xla_gpu_float_check_crash_dump",
          std::move(dump_bundle), operands, /*results=*/{}, /*attributes=*/{},
          hlo_module->entry_computation(), se::GpuComputeCapability()));

  ThunkSequence sequence;
  // The sequence can have up to 4 thunks: backup, original, check, and dump.
  sequence.reserve(4);
  if (!inout_buffers.empty()) {
    CustomCallThunk::OwnedHandlerBundle backup_bundle{};
    backup_bundle.execute =
        xla::ffi::Ffi::Bind().Ctx<xla::ffi::Stream>().RemainingArgs().To(
            absl::bind_front(BackupBuffers, inout_buffers, backup_data));

    std::vector<NullableShapedSlice> backup_operands;
    backup_operands.reserve(inout_buffers.size());
    for (const auto& slice : inout_buffers) {
      backup_operands.push_back(
          ShapedSlice{slice, Shape(PrimitiveType::U8, {slice.size()})});
    }

    ASSIGN_OR_RETURN(
        auto backup_thunk,
        CustomCallThunk::Create(
            Thunk::ThunkInfo(), "xla_gpu_inout_buffers_backup",
            std::move(backup_bundle), backup_operands, /*results=*/{},
            /*attributes=*/{}, hlo_module->entry_computation(),
            se::GpuComputeCapability()));
    sequence.push_back(std::move(backup_thunk));
  }
  Thunk::ThunkInfo checked_thunk_info = thunk->thunk_info();
  sequence.push_back(std::move(thunk));

  auto check_thunk = std::make_unique<BuffersDebugFloatCheckThunk>(
      Thunk::ThunkInfo(), checked_thunk_info, log_slice, tmp_slice,
      std::move(buffers_to_check), metadata_store);
  sequence.push_back(std::move(check_thunk));
  sequence.push_back(std::move(dump_thunk));

  return std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                           std::move(sequence));
}

absl::StatusOr<std::unique_ptr<Thunk>> WrapWithFloatCheckThunk(
    std::unique_ptr<Thunk> thunk, BufferAllocation::Slice log_slice,
    const Thunk& predecessor_thunk, Thunk& successor_thunk,
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store,
    ThunkPassBufferAllocator& allocator) {
  if (thunk->buffer_uses().empty()) {
    VLOG(1) << "No buffers in thunk " << thunk->thunk_info().thunk_id
            << ", skipping";
    return thunk;
  }

  absl::flat_hash_map<size_t, BufferAllocation::Slice> buffers_to_check =
      GetBuffersToCheckOrBackup(*thunk).buffers_to_check;

  if (buffers_to_check.empty()) {
    return thunk;
  }

  const size_t temp_buffer_size_bytes =
      CalculateTempBufferSize(*thunk) * sizeof(xla::gpu::FloatCheckResult);
  ASSIGN_OR_RETURN(BufferAllocation * tmp_alloc,
                   allocator.NewEmptyAllocation(temp_buffer_size_bytes));
  BufferAllocation::Slice tmp_slice(tmp_alloc, 0, tmp_alloc->size());

  VLOG(1) << "Wrapping thunk " << thunk->thunk_info().thunk_id
          << " with float check thunk due to presence of buffers: "
          << buffers_to_check.size();
  ThunkSequence thunk_and_checks;
  Thunk* thunk_ptr = thunk.get();
  thunk_and_checks.push_back(std::move(thunk));
  auto buffer_debug_float_check_thunk =
      std::make_unique<BuffersDebugFloatCheckThunk>(
          Thunk::ThunkInfo(), thunk_ptr->thunk_info(), log_slice, tmp_slice,
          std::move(buffers_to_check), std::move(metadata_store));
  thunk_and_checks.push_back(std::move(buffer_debug_float_check_thunk));
  auto wrapped_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(thunk_and_checks));
  return wrapped_thunk;
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
  ASSIGN_OR_RETURN(std::vector<BufferDebugFloatCheckEntry> entries,
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

absl::StatusOr<std::unique_ptr<BuffersDebugFloatCheckThunk>>
CreateOutputBuffersCheckThunk(
    const DebugOptions& debug_options, const HloModule* absl_nonnull hlo_module,
    const BufferAssignment* absl_nonnull buffer_assignment,
    BufferAllocation::Slice log_slice,
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store,
    ThunkPassBufferAllocator& allocator) {
  if (!debug_options.xla_gpu_experimental_thunk_buffer_debug_module_outputs()) {
    return nullptr;
  }
  if (buffer_assignment == nullptr) {
    LOG(ERROR)
        << "Buffer assignment is null, cannot determine module output buffers";
    return nullptr;
  }

  absl::flat_hash_map<size_t, ShapedSlice> buffers_to_check_shaped;
  ASSIGN_OR_RETURN(buffers_to_check_shaped,
                   GetOutputShapedBuffers(hlo_module, buffer_assignment));

  absl::flat_hash_map<size_t, BufferAllocation::Slice> buffers_to_check;
  buffers_to_check.reserve(buffers_to_check_shaped.size());
  size_t max_buffer_size_bytes = 0;

  for (const auto& [idx, shaped_slice] : buffers_to_check_shaped) {
    const BufferAllocation::Slice& slice = shaped_slice.slice;
    if (IsFloatTypeSupportedByChecker(slice.element_type())) {
      buffers_to_check.emplace(idx, slice);
      max_buffer_size_bytes =
          std::max<size_t>(max_buffer_size_bytes, slice.size());
    }
  }

  if (buffers_to_check.empty()) {
    VLOG(1) << "No output buffers with float types, skipping output check";
    return nullptr;
  }

  const size_t temp_buffer_size_bytes =
      TempBufferSizeFromMaxBufferSize(max_buffer_size_bytes);

  ASSIGN_OR_RETURN(BufferAllocation * tmp_alloc,
                   allocator.NewEmptyAllocation(temp_buffer_size_bytes));
  BufferAllocation::Slice tmp_slice(tmp_alloc, 0, tmp_alloc->size());

  Thunk::ThunkInfo checked_thunk_info;
  checked_thunk_info.profile_annotation =
      absl::StrCat("Module ", hlo_module->name(), " Output Check");

  return std::make_unique<BuffersDebugFloatCheckThunk>(
      Thunk::ThunkInfo(), checked_thunk_info, log_slice, tmp_slice,
      std::move(buffers_to_check), metadata_store);
}

}  // namespace

absl::Status RunFloatCheckPassInternal(
    ThunkSequence* thunk_sequence, const DebugOptions& debug_options,
    const HloModule* absl_nonnull hlo_module,
    const BufferAssignment* buffer_assignment,
    ThunkPassBufferAllocator& allocator) {
  const bool dump_mode =
      debug_options.xla_gpu_detect_nan() == DebugOptions::DETECTION_MODE_DUMP ||
      debug_options.xla_gpu_detect_inf() == DebugOptions::DETECTION_MODE_DUMP;

  if (dump_mode && debug_options.xla_dump_to().empty()) {
    return absl::InvalidArgumentError(
        "xla_dump_to MUST be defined when crash dump mode is requested.");
  }

  std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store =
      std::make_shared<BufferDebugLogEntryMetadataStore>();
  ASSIGN_OR_RETURN(BufferAllocation * log_alloc,
                   allocator.NewEmptyAllocation(kLogSizeBytes));
  BufferAllocation::Slice log_slice(log_alloc, 0, log_alloc->size());

  ASSIGN_OR_RETURN(auto buffer_debug_init_thunk,
                   CreateDebugInitThunk(log_slice, hlo_module));

  ThunkFilter thunk_filter = CreateThunkFilter(debug_options);

  if (dump_mode) {
    absl::flat_hash_map<absl::string_view, const HloInstruction*>
        hlo_instruction_map;
    for (const HloComputation* computation : hlo_module->computations()) {
      for (const HloInstruction* instruction : computation->instructions()) {
        hlo_instruction_map[instruction->name()] = instruction;
      }
    }

    auto transform_callback = [&](std::unique_ptr<Thunk> thunk)
        -> absl::StatusOr<std::unique_ptr<Thunk>> {
      if (thunk_filter(*thunk) == InstrumentAction::kSkip) {
        return thunk;
      }
      return WrapWithSyncDumpThunk(std::move(thunk), metadata_store, log_slice,
                                   hlo_module, buffer_assignment, allocator,
                                   hlo_instruction_map);
    };
    RETURN_IF_ERROR(thunk_sequence->TransformNested(transform_callback));

    thunk_sequence->reserve(thunk_sequence->size() + 1);
    thunk_sequence->insert(thunk_sequence->begin(),
                           std::move(buffer_debug_init_thunk));
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(
      auto buffer_debug_dump_thunk,
      CreateBufferDebugFloatCheckThunk(metadata_store, log_slice, hlo_module));

  auto transform_callback = [&](std::unique_ptr<Thunk> thunk)
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
  };

  RETURN_IF_ERROR(thunk_sequence->TransformNested(transform_callback));
  ASSIGN_OR_RETURN(
      std::unique_ptr<BuffersDebugFloatCheckThunk> output_buffers_check_thunk,
      CreateOutputBuffersCheckThunk(debug_options, hlo_module,
                                    buffer_assignment, log_slice,
                                    metadata_store, allocator));

  thunk_sequence->reserve(thunk_sequence->size() + 3);
  thunk_sequence->insert(thunk_sequence->begin(),
                         std::move(buffer_debug_init_thunk));
  if (output_buffers_check_thunk != nullptr) {
    thunk_sequence->push_back(std::move(output_buffers_check_thunk));
  }
  thunk_sequence->push_back(std::move(buffer_debug_dump_thunk));
  return absl::OkStatus();
}

}  // namespace xla::gpu
