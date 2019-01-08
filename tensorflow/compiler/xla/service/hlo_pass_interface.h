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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_INTERFACE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_INTERFACE_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_group.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Base class for HLO passes. These are used with the HloPassPipeline to
// organize a sequence of passes. An HLO pass should not extend this class
// directly; it should extend HloModulePass or HloModuleGroupPass.
class HloPassInterface {
 public:
  virtual ~HloPassInterface() = default;
  virtual absl::string_view name() const = 0;

  // Run the pass on the given HLO module.  Returns whether it modified the
  // module.
  virtual StatusOr<bool> Run(HloModule* module) = 0;

  // Run the pass on the given HLO module group. Returns whether it modified the
  // module group. Ideally, the module group variant would be named "Run" as
  // well, but C++ does not handle overloaded virtual methods well.
  virtual StatusOr<bool> RunOnModuleGroup(HloModuleGroup* module_group) = 0;
};

// Base class for passes which are module-scoped.
class HloModulePass : public HloPassInterface {
 public:
  // Runs the pass on a module group by iterating through each module in the
  // group.
  StatusOr<bool> RunOnModuleGroup(HloModuleGroup* module_group) override {
    bool changed = false;
    for (HloModule* module : module_group->modules()) {
      TF_ASSIGN_OR_RETURN(bool module_changed, Run(module));
      changed |= module_changed;
    }
    return changed;
  };
};

// Base class for passes which are module-group scoped. These passes cannot run
// on an HLO module.
class HloModuleGroupPass : public HloPassInterface {
 public:
  StatusOr<bool> Run(HloModule* module) override {
    return InternalError("Module group pass cannot be run on a module");
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_INTERFACE_H_
