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

#include "xla/pjrt/tpu/tpu_topology_utils_external.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/tpu/tpu_topology_args.h"

namespace xla {

absl::StatusOr<std::unique_ptr<xla::PjRtTopologyDescription>>
CreateTpuTopologyDescription(const TpuTopologyArgs& args) {
  return absl::UnimplementedError("Not implemented");
}

absl::StatusOr<TpuTopologyArgs> TpuTopologyArgsFromPjRtTopologyDescription(
    const xla::PjRtTopologyDescription& topology_description) {
  return absl::UnimplementedError("Not implemented");
}

absl::string_view TpuVersionToString(TpuVersion version) {
  switch (version) {
    case TpuVersion::TPU_VERSION_V2:
      return "TPU v2";
    case TpuVersion::TPU_VERSION_V3:
      return "TPU v3";
    case TpuVersion::TPU_VERSION_V4:
      return "TPU v4";
    case TpuVersion::TPU_VERSION_V5:
      return "TPU v5";
    default:
      return "Unknown TPU version";
  }
}

}  // namespace xla
