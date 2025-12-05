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

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_checksum.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_float_check.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_saver_inserter.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

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

  switch (mode_) {
    case Mode::kChecksum:
      TF_RETURN_IF_ERROR(RunChecksumPassInternal(root_thunk, debug_options,
                                                 hlo_module, allocator));
      break;
    case Mode::kFloatChecker:
      TF_RETURN_IF_ERROR(RunFloatCheckPassInternal(root_thunk, debug_options,
                                                   hlo_module, allocator));
      break;
    case Mode::kBufferSaver:
      TF_RETURN_IF_ERROR(
          RunDebugSaverInserter(*root_thunk, debug_options, *hlo_module));
      break;
  }

  return true;
}

}  // namespace xla::gpu
