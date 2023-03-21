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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_LAYOUT_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_LAYOUT_ASSIGNMENT_H_

#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {
namespace cpu {

// CPU-specific layout assignment pass which preassigns layouts to satisfy
// layout constraints for operands and results of library calls.
class CpuLayoutAssignment : public LayoutAssignment {
 public:
  explicit CpuLayoutAssignment(
      ComputationLayout* entry_computation_layout,
      const TargetMachineFeatures* target_machine_features,
      ChannelLayoutConstraints* channel_constraints = nullptr)
      : LayoutAssignment(entry_computation_layout, channel_constraints),
        target_machine_features_(*target_machine_features) {}
  ~CpuLayoutAssignment() override {}

 protected:
  Status AddBackendConstraints(LayoutConstraints* constraints) override;
  // The CPU backend does not use memory spaces, so there is no need to
  // propagate them.
  Status PropagateMemorySpace(HloModule* module) override { return OkStatus(); }

  const TargetMachineFeatures& target_machine_features_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_LAYOUT_ASSIGNMENT_H_
