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

#ifndef XLA_SERVICE_HLO_MODULE_EXECUTABLE_H_
#define XLA_SERVICE_HLO_MODULE_EXECUTABLE_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/computation_layout.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/stream_executor/kernel_stats.h"

namespace xla {

// Base class for executables that keep the HloModule they were compiled from
// and answer metadata queries (name, module config, entry computation layout,
// input/output alias config, ...) from it.
class HloModuleExecutable : public Executable {
 public:
  // The hlo_module parameter may be nullptr, if the given executable type
  // doesn't need it for execution. The metadata accessors below CHECK-fail
  // then, except for `name()`.
  explicit HloModuleExecutable(std::shared_ptr<HloModule> hlo_module);

  std::shared_ptr<HloModule> shared_module() const override {
    return hlo_module_;
  }
  const HloModuleConfig& module_config() const override;
  Shape result_shape() const override;
  ComputationLayout compute_computation_layout() const override;
  absl::string_view name() const override;
  const HloInputOutputAliasConfig& input_output_alias_config() const override;
  const HloBufferDonorConfig& buffer_donor_config() const override;

  // Doesn't expose a buffer assignment. Subclasses that keep one override this.
  const BufferAssignmentProto* buffer_assignment_proto() const override {
    return nullptr;
  }

  const ModuleStats& module_stats() const override { return module_stats_; }

  // Sets a module_stats map of kernel name to relevant kernel stats.
  void set_module_stats(ModuleStats module_stats);

 protected:
  HloModule* module_ptr() const override { return hlo_module_.get(); }

 private:
  // HloModule this was compiled from. BufferAssignment keeps pointers to
  // HloInstructions owned by the HloModule so we need to keep the HloModule
  // around if we keep the BufferAssignment around.
  //
  // This member may be nullptr, if the given executable type doesn't need it
  // for execution.
  std::shared_ptr<HloModule> hlo_module_;

  // A map from kernel name to relevant kernel stats.
  ModuleStats module_stats_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_MODULE_EXECUTABLE_H_
