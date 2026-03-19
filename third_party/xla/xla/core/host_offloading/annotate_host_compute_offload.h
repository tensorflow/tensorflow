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

#ifndef XLA_CORE_HOST_OFFLOADING_ANNOTATE_HOST_COMPUTE_OFFLOAD_H_
#define XLA_CORE_HOST_OFFLOADING_ANNOTATE_HOST_COMPUTE_OFFLOAD_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Annotates host compute offloaded instructions with the frontend attribute
// `_xla_compute_type=host`.
class AnnotateHostComputeOffload : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "annotate-host-compute-offload";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_CORE_HOST_OFFLOADING_ANNOTATE_HOST_COMPUTE_OFFLOAD_H_
