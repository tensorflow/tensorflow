/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_TOOLS_COMPARISON_OFFLINE_UTILS_H_
#define XLA_HLO_TOOLS_COMPARISON_OFFLINE_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"

namespace xla::numerics::comparison {

ScopeInstructionProto ProtoFromScopeInstruction(const ScopeInstruction& si);

// File utilities
absl::StatusOr<std::vector<std::string>> FindFiles(
    absl::Span<const std::string> dirs, absl::string_view pattern);

absl::StatusOr<std::string> FindOneFile(absl::Span<const std::string> dirs,
                                        absl::string_view pattern);

struct LaunchInfoResult {
  std::string path;
  int64_t launch_barrier_id;
};

absl::StatusOr<LaunchInfoResult> FindLaunchInfo(
    absl::Span<const std::string> dirs, absl::string_view module_name,
    std::optional<int64_t> launch_barrier_id);

absl::StatusOr<std::unique_ptr<HloModule>> ReadHloModuleFromFile(
    absl::string_view path,
    xla::StackFrameIndexProto* stack_frame_index = nullptr);

// Holds data needed for one run (baseline or target).
struct RunData {
  std::string module_name;
  int64_t launch_barrier_id;
  std::unique_ptr<const DeviceAssignment> device_assignment;
  std::unique_ptr<HloModule> original_module;
  std::unique_ptr<HloModule> optimized_module;
  xla::StackFrameIndexProto stack_frame_index;
  std::vector<std::string> log_files;
  // device_ids[i] is the device_id for log_files[i].
  std::vector<int64_t> device_ids_for_log_files;
};

absl::StatusOr<RunData> LoadRunData(absl::Span<const std::string> dirs,
                                    absl::string_view module_name,
                                    std::optional<int64_t> launch_barrier_id);

}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_OFFLINE_UTILS_H_
