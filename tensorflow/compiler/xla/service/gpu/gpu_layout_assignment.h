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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_LAYOUT_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_LAYOUT_ASSIGNMENT_H_

#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace gpu {

// GPU-specific layout assignment pass which preassigns layouts to satisfy
// layout constraints for operands and results of library calls.
class GpuLayoutAssignment : public LayoutAssignment {
 public:
  explicit GpuLayoutAssignment(
      const ComputationLayout& entry_computation_layout)
      : LayoutAssignment(entry_computation_layout) {}
  ~GpuLayoutAssignment() override {}

 protected:
  Status AddBackendConstraints(LayoutConstraints* constraints) override;
  Status PropagateOperandConstraint(
      const OperandLayoutConstraint& layout_constraint,
      LayoutConstraints* constraints) override;
  Status PropagateBufferConstraint(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints) override;
  bool CustomCallRequiresMajorFirstLayout(
      const HloInstruction* instruction) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_LAYOUT_ASSIGNMENT_H_
