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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace gpu {

// Determines the schedule of HLO instructions, represented by the total order
// of thunk launches, and the partial order of HLO instructions. The HLO
// instructions are only partially ordered, despite the total ordering of thunk
// launches, because thunks may be scheduled onto concurrent streams. This
// schedule is used by BufferAssigner to determine buffer liveness (i.e. to
// minimize allocations), and also by ThunkSchedule to determine the thunk
// launch order. This class differs from xla::HloSchedule in that HloSchedule
// represents a total order of all instructions in the module for backends which
// execute HLO instructions strictly sequentially.
class GpuHloSchedule {
 public:
  // Constructs an GpuHloSchedule for the given module, based on the given
  // stream assignment.
  static StatusOr<std::unique_ptr<GpuHloSchedule>> Build(
      const HloModule& module, const StreamAssignment& stream_assignment,
      int64 pointer_size);

  // Returns the total order of thunk launches, represented in terms of HLO
  // instructions.
  const std::vector<HloInstruction*>& ThunkLaunchOrder() const {
    return thunk_launch_order_;
  }

  // Returns the partial order of HLO instructions. This method may only be
  // called once. The order is based on the total order of thunk lanches, the
  // stream assignment, and the data dependencies in the HLO DAG.
  std::unique_ptr<HloOrdering> ConsumeHloOrdering() {
    return std::move(hlo_ordering_);
  }

 private:
  GpuHloSchedule();

  std::vector<HloInstruction*> thunk_launch_order_;
  std::unique_ptr<HloOrdering> hlo_ordering_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_
