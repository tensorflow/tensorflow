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

#include "xla/backends/gpu/runtime/thunk_buffer_debug_pass.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "re2/re2.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/buffers_checksum_thunk.h"
#include "xla/backends/gpu/runtime/buffers_nan_count_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
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
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
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

std::unique_ptr<Thunk> WrapWithNanCounterThunk(
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
              << " has no defined contents on output, skipping";
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
          << " with nan counter thunk due to presence of buffers: "
          << buffers_to_check.size();
  std::vector<std::unique_ptr<Thunk>> thunk_and_checks;
  Thunk* thunk_ptr = thunk.get();
  thunk_and_checks.push_back(std::move(thunk));
  auto buffer_debug_nan_counter_thunk =
      std::make_unique<BuffersDebugNanCountThunk>(
          Thunk::ThunkInfo(), log_slice, thunk_ptr->thunk_info().thunk_id,
          std::move(buffers_to_check),
          /*runs_before_checked_thunk=*/false, std::move(metadata_store));
  buffer_debug_nan_counter_thunk->add_control_predecessor(thunk_ptr);
  thunk_and_checks.push_back(std::move(buffer_debug_nan_counter_thunk));
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
absl::Status DumpBufferDebugLog(
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
  TF_ASSIGN_OR_RETURN(std::vector<BufferDebugLogEntry> log_entries,
                      buffer_debug_log.ReadFromDevice(*stream));
  BufferDebugLogProto buffer_debug_log_proto =
      metadata_store->EntriesToProto(log_entries);

  VLOG(1) << "read " << buffer_debug_log_proto.entries_size() << " entries";
  DumpPerExecutionProtobufToFile(*hlo_module, buffer_debug_log_proto,
                                 debug_options, "buffer_debug_log", nullptr);
  int non_zero_nan_count_modules_count = 0;
  for (const auto& entry : buffer_debug_log_proto.entries()) {
    if (entry.check_type() == BufferDebugLogEntryProto::CHECK_TYPE_NAN_COUNT &&
        entry.checksum() > 0) {
      LOG(ERROR) << "Found entry with non zero nan count " << entry.checksum()
                 << " for thunk " << entry.thunk_id() << " and execution "
                 << entry.execution_id() << " for module: \n"
                 << hlo_module->ToString();
      non_zero_nan_count_modules_count++;
    }
  }
  if (non_zero_nan_count_modules_count > 0 &&
      hlo_module->config().debug_options().xla_gpu_detect_nan() ==
          DebugOptions::NAN_CHECK_DETECTION_MODE_FAIL) {
    LOG(FATAL) << "Found " << non_zero_nan_count_modules_count
               << " modules with non zero nan count";
  }
  return absl::OkStatus();
}

// A boolean-like value returned from thunk filters to indicate whether the
// thunk should be instrumented or left as is.
enum class InstrumentAction : bool {
  // Don't instrument the thunk, leave it as is.
  kSkip,
  // Instrument the thunk.
  kInstrument,
};

// A function that decides whether the thunk should be instrumented
// (kInstrument) or not (kSkip).
using ThunkFilter = absl::AnyInvocable<InstrumentAction(const Thunk&) const>;

// Creates a thunk filter that filters thunks by their IDs, based the allowed
// ranges passed in debug options.
ThunkFilter CreateThunkIdFilter(const DebugOptions& debug_options) {
  std::vector<std::pair<int64_t, int64_t>> thunk_id_ranges;
  for (const auto& range :
       debug_options.xla_gpu_experimental_thunk_buffer_debug_filter()
           .thunk_id_ranges()) {
    VLOG(1) << "Thunk filter: id range [" << range.first() << ", "
            << range.last() << "]";
    thunk_id_ranges.emplace_back(range.first(), range.last());
  }

  return [id_ranges = std::move(thunk_id_ranges)](const Thunk& thunk) {
    if (id_ranges.empty()) {
      return InstrumentAction::kInstrument;
    }

    const ThunkId thunk_id = thunk.thunk_info().thunk_id;
    if (absl::c_any_of(id_ranges, [&](const auto& range) {
          VLOG(2) << "Thunk filter: check ID range: " << range.first
                  << " <= " << thunk_id.value() << " <= " << range.second;
          return range.first <= thunk_id.value() &&
                 thunk_id.value() <= range.second;
        })) {
      VLOG(2) << "Thunk filter: ID matches";
      return InstrumentAction::kInstrument;
    }

    VLOG(2) << "Thunk filter: ID does not match";
    return InstrumentAction::kSkip;
  };
}

// Creates a thunk filter that filters thunks by matching their profile
// annotations against regexes configured in debug options.
ThunkFilter CreateProfileAnnotationRegexFilter(
    const DebugOptions& debug_options) {
  std::vector<std::unique_ptr<RE2>> profile_annotation_regexes;
  for (const auto& regex :
       debug_options.xla_gpu_experimental_thunk_buffer_debug_filter()
           .profile_annotation_regexes()) {
    VLOG(1) << "Thunk filter: profile annotation regex: " << regex;
    profile_annotation_regexes.push_back(std::make_unique<RE2>(regex));
  }
  return [regexes = std::move(profile_annotation_regexes)](const Thunk& thunk) {
    if (regexes.empty()) {
      return InstrumentAction::kInstrument;
    }

    const std::string& profile_annotation =
        thunk.thunk_info().profile_annotation;
    if (absl::c_any_of(regexes, [&](const auto& regex) {
          VLOG(2) << "Thunk filter: check profile annotation regex: "
                  << regex->pattern();
          return RE2::PartialMatch(profile_annotation, *regex);
        })) {
      VLOG(2) << "Thunk filter: profile annotation matches";
      return InstrumentAction::kInstrument;
    }

    VLOG(2) << "Thunk filter: profile annotation does not match";
    return InstrumentAction::kSkip;
  };
}

// Creates a thunk filter that filters thunks by all the conditions configured
// in debug options.
ThunkFilter CreateThunkFilter(const DebugOptions& debug_options) {
  std::vector<ThunkFilter> filters;
  filters.push_back(CreateThunkIdFilter(debug_options));
  filters.push_back(CreateProfileAnnotationRegexFilter(debug_options));

  return [filters = std::move(filters)](const Thunk& thunk) {
    VLOG(2) << "Thunk filter: check ID " << thunk.thunk_info().thunk_id
            << ", profile annotation " << thunk.thunk_info().profile_annotation;
    if (absl::c_all_of(filters, [&](const auto& filter) {
          return filter(thunk) == InstrumentAction::kInstrument;
        })) {
      return InstrumentAction::kInstrument;
    }
    return InstrumentAction::kSkip;
  };
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kDebugLogInitHandler,
    [](se::Stream* absl_nonnull stream, xla::ffi::Buffer<U8> log_buffer) {
      return se::gpu::BufferDebugLog::CreateOnDevice(*stream,
                                                     log_buffer.device_memory())
          .status();
    },
    xla::ffi::Ffi::Bind().Ctx<xla::ffi::Stream>().Arg<xla::ffi::Buffer<U8>>());

}  // namespace

absl::StatusOr<bool> ThunkBufferDebugPass::Run(
    SequentialThunk* root_thunk, const DebugOptions& debug_options,
    const HloModule* absl_nullable hlo_module,
    const se::DeviceDescription& device_info,
    ThunkPassBufferAllocator& allocator) {
  VLOG(1) << "ThunkBufferDebugPass running";
  if (hlo_module == nullptr) {
    // We need the HLO module to dump the buffer debug log proto to a file. If
    // it's not available, there's no point in doing extra work.
    VLOG(1) << "HLO module is null, skip buffer checksumming";
    return false;
  }

  std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store =
      std::make_shared<BufferDebugLogEntryMetadataStore>();

  TF_ASSIGN_OR_RETURN(BufferAllocation * log_alloc,
                      allocator.NewEmptyAllocation(kLogSizeBytes));
  BufferAllocation::Slice log_slice(log_alloc, 0, log_alloc->size());
  ShapedSlice shaped_log_slice{
      /*slice=*/log_slice,
      /*shape=*/Shape(PrimitiveType::U8, /*dimensions=*/{log_alloc->size()}),
  };

  XLA_FFI_Handler_Bundle buffer_debug_init_bundle{};
  buffer_debug_init_bundle.execute = kDebugLogInitHandler;
  TF_ASSIGN_OR_RETURN(
      auto buffer_debug_init_thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(), "xla_gpu_buffer_debug_log_init",
          buffer_debug_init_bundle, /*operands=*/{shaped_log_slice},
          /*results=*/{}, /*attributes=*/{}, hlo_module->entry_computation()));

  CustomCallThunk::OwnedHandlerBundle dump_bundle{};
  dump_bundle.execute =
      xla::ffi::Ffi::Bind()
          .Ctx<xla::ffi::Stream>()
          .Ctx<xla::ffi::CalledComputation>()
          .Arg<xla::ffi::Buffer<U8>>()
          .To(absl::bind_front(DumpBufferDebugLog, metadata_store));
  TF_ASSIGN_OR_RETURN(auto buffer_debug_dump_thunk,
                      CustomCallThunk::Create(Thunk::ThunkInfo(),
                                              "xla_gpu_buffer_debug_log_dump",
                                              std::move(dump_bundle),
                                              /*operands=*/{shaped_log_slice},
                                              /*results=*/{}, /*attributes=*/{},
                                              hlo_module->entry_computation()));

  ThunkFilter thunk_filter = CreateThunkFilter(debug_options);
  root_thunk->TransformAllNestedThunks([&](std::unique_ptr<Thunk> thunk) {
    if (thunk_filter(*thunk) == InstrumentAction::kSkip) {
      return thunk;
    }
    switch (mode_) {
      case Mode::kChecksum:
        VLOG(1) << "Wrapping with checksum thunk";
        return WrapWithChecksumThunk(
            std::move(thunk), log_slice,
            /*predecessor_thunk=*/*buffer_debug_init_thunk,
            /*successor_thunk=*/*buffer_debug_dump_thunk, metadata_store);
      case Mode::kNanCounter:
        VLOG(1) << "Wrapping with nan counter thunk";
        return WrapWithNanCounterThunk(
            std::move(thunk), log_slice,
            /*predecessor_thunk=*/*buffer_debug_init_thunk,
            /*successor_thunk=*/*buffer_debug_dump_thunk, metadata_store);
    }
    return thunk;
  });

  ThunkSequence& thunks = root_thunk->thunks();
  thunks.reserve(thunks.size() + 2);
  thunks.insert(thunks.begin(), std::move(buffer_debug_init_thunk));
  thunks.push_back(std::move(buffer_debug_dump_thunk));

  return true;
}

}  // namespace xla::gpu
