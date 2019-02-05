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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_FIX_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_FIX_H_

#include <algorithm>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_group.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Do an HLO pass to a fix point.
template <typename Pass>
class HloPassFix : public Pass {
 public:
  template <typename... Args>
  explicit HloPassFix(Args&&... args) : Pass(args...) {}

  StatusOr<bool> Run(HloModule* module) override {
    bool changed = false;
    bool changed_this_iteration = true;
    int64 iteration_count = 0;
    int64 limit =
        std::max(static_cast<int64>(1000), module->instruction_count());
    VLOG(3) << "Running HloPassFix.";
    while (changed_this_iteration) {
      TF_ASSIGN_OR_RETURN(changed_this_iteration, Pass::Run(module));
      changed |= changed_this_iteration;
      VLOG(3) << "changed_this_iteration: " << changed_this_iteration;
      ++iteration_count;
      if (iteration_count == limit) {
        LOG(ERROR)
            << "Unexpectedly high number of iterations in HLO passes ("
            << iteration_count
            << ")\nIf compilation hangs here, please file a bug with XLA.";
      }
    }
    return changed;
  }

  StatusOr<bool> RunOnModuleGroup(HloModuleGroup* module_group) override {
    bool changed = false;
    bool changed_this_iteration = true;
    int64 iteration_count = 0;
    int64 limit = 1000;
    for (const HloModule* module : module_group->modules()) {
      limit = std::max<int64>(limit, module->instruction_count());
    }
    VLOG(3) << "Running HloPassFix.";
    while (changed_this_iteration) {
      TF_ASSIGN_OR_RETURN(changed_this_iteration,
                          Pass::RunOnModuleGroup(module_group));
      changed |= changed_this_iteration;
      VLOG(3) << "changed_this_iteration: " << changed_this_iteration;
      ++iteration_count;
      if (iteration_count == limit) {
        LOG(ERROR)
            << "Unexpectedly high number of iterations in HLO passes ("
            << iteration_count
            << ")\nIf compilation hangs here, please file a bug with XLA.";
      }
    }
    return changed;
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_FIX_H_
