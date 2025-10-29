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

#ifndef XLA_PJRT_TPU_TPU_TOPOLOGY_UTILS_EXTERNAL_H_
#define XLA_PJRT_TPU_TPU_TOPOLOGY_UTILS_EXTERNAL_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/tpu/tpu_topology_args.h"

namespace xla {
namespace external {

// Creates a new PjRtTopologyDescription from the given TpuTopologyArgs.
absl::StatusOr<std::unique_ptr<xla::PjRtTopologyDescription>>
CreateTpuTopologyDescription(const TpuTopologyArgs& args);

// Creates a new TpuTopologyArgs from the given PjRtTopologyDescription.
absl::StatusOr<TpuTopologyArgs> TpuTopologyArgsFromPjRtTopologyDescription(
    const xla::PjRtTopologyDescription& topology_description);

absl::string_view TpuVersionToString(TpuVersion version);

}  // namespace external
}  // namespace xla

#endif  // XLA_PJRT_TPU_TPU_TOPOLOGY_UTILS_EXTERNAL_H_
