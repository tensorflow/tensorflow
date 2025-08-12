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

#ifndef XLA_SERVICE_HLO_CYCLE_DETECTION_H_
#define XLA_SERVICE_HLO_CYCLE_DETECTION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {

class CycleDetectionVisitor : public DfsHloVisitorWithDefault {
 public:
  absl::Status VerifyNoCycle(HloModule* module) {
    for (auto* comp : module->computations()) {
      TF_RETURN_IF_ERROR(comp->Accept(this));
    }
    return absl::OkStatus();
  }
  // Relies on HloInstruction::Accept() to run PostOrderDFS which does cycle
  // detection by default.
  absl::Status VerifyNoCycle(HloModuleGroup* module_group) {
    for (auto* comp : module_group->module(0).computations()) {
      TF_RETURN_IF_ERROR(comp->Accept(this));
    }
    return absl::OkStatus();
  }
  absl::Status DefaultAction(HloInstruction* instruction) override {
    return absl::OkStatus();
  }
};

// HLO pass that detects cycles in the HLO graph.
class HloCycleDetection : public HloModulePass {
 public:
  absl::string_view name() const override { return "hlo-cycle-detection"; }

  // Never returns true; no instructions are ever modified by this pass.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    TF_RETURN_IF_ERROR(visitor_.VerifyNoCycle(module));
    return false;
  }

  using HloPassInterface::RunOnModuleGroup;
  absl::StatusOr<bool> RunOnModuleGroup(
      HloModuleGroup* module_group,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override {
    TF_RETURN_IF_ERROR(visitor_.VerifyNoCycle(module_group));
    return false;
  }

 private:
  CycleDetectionVisitor visitor_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_CYCLE_DETECTION_H_
