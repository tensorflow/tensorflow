/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/nvptx_executable.h"

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

NVPTXExecutable::NVPTXExecutable(
    const string& text, const std::vector<uint8>& cubin,
    std::pair<int, int> compute_capability,
    std::unique_ptr<const ThunkSchedule> thunk_schedule,
    std::unique_ptr<const HloModule> hlo_module,
    std::unique_ptr<const BufferAssignment> assignment,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
    : GpuExecutable(std::move(text), std::move(thunk_schedule),
                    std::move(hlo_module), std::move(assignment),
                    std::move(hlo_profile_printer_data),
                    std::move(hlo_profile_index_map)),
      cubin_(cubin),
      compute_capability_(compute_capability) {}

Status NVPTXExecutable::CheckCompatibilityWithServiceExecutableRunOptions(
    const ServiceExecutableRunOptions* run_options) {
  se::Stream* main_stream = run_options->stream();

  std::pair<int, int> stream_compute_compatibility;
  main_stream->parent()->GetDeviceDescription().cuda_compute_capability(
      &stream_compute_compatibility.first,
      &stream_compute_compatibility.second);
  TF_RET_CHECK(stream_compute_compatibility == compute_capability_)
      << "Compute capability mismatch; expected {" << compute_capability_.first
      << ", " << compute_capability_.second << "}, but was {"
      << stream_compute_compatibility.first << ", "
      << stream_compute_compatibility.second << "}";

  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
