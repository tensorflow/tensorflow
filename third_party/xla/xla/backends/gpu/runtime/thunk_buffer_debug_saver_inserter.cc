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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/runtime_intrinsics.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_filter.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::StatusOr<std::unique_ptr<Thunk>> InsertBufferSaverCustomCall(
    const HloModule& hlo_module, std::unique_ptr<Thunk> thunk,
    const std::string& path) {
  std::vector<std::unique_ptr<Thunk>> sequence;
  sequence.emplace_back(std::move(thunk));

  absl::flat_hash_set<BufferAllocation::Slice> processed;

  Thunk::BufferUses uses = sequence[0]->buffer_uses();
  // Results are last in the list. Process in reverse order in case of InOut
  // argument, which appears in the list twice.
  for (int i = uses.size() - 1; i >= 0; i--) {
    const BufferUse& buffer = uses[i];
    if (buffer.access() != BufferUse::MemoryAccess::kWrite) {
      continue;
    }

    const BufferAllocation::Slice& slice = buffer.slice();
    if (!processed.insert(slice).second) {
      continue;
    }

    ShapedSlice output{slice, buffer.shape()};
    ffi::AttributesMap attributes{
        {"dir", ffi::Attribute{path}},
        {"metadata", {sequence[0]->thunk_info().profile_annotation}}};

    Thunk::ThunkInfo info;
    info.profile_annotation =
        absl::StrCat("Buffer saver ", sequence[0]->profile_annotation());
    info.execution_stream_id = sequence[0]->execution_stream_id();

    TF_ASSIGN_OR_RETURN(
        auto log_thunk,
        CustomCallThunk::Create(
            info, std::string{kXlaGpuAppendToFileCustomCallTag}, {output},
            {std::nullopt}, attributes, hlo_module.entry_computation(), "GPU",
            stream_executor::GpuComputeCapability()));
    log_thunk->add_control_predecessor(sequence[0].get());
    sequence.emplace_back(std::move(log_thunk));
  }

  auto wrapped_thunk = std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                                         std::move(sequence));
  return std::unique_ptr<Thunk>(std::move(wrapped_thunk));
}

}  // namespace

absl::Status RunDebugSaverInserter(SequentialThunk& root_thunk,
                                   const DebugOptions& debug_options,
                                   const HloModule& hlo_module) {
  if (debug_options.xla_dump_to().empty()) {
    LOG(WARNING)
        << "Buffer saver enabled but target directory is not provided.";
    return absl::OkStatus();
  }
  ThunkFilter thunk_filter = CreateThunkFilter(debug_options);
  return root_thunk.TransformNested(
      [&](std::unique_ptr<Thunk> thunk)
          -> absl::StatusOr<std::unique_ptr<Thunk>> {
        if (thunk_filter(*thunk) == InstrumentAction::kSkip) {
          return thunk;
        }
        return InsertBufferSaverCustomCall(hlo_module, std::move(thunk),
                                           debug_options.xla_dump_to());
      });
}

}  // namespace xla::gpu
