/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_CUSTOM_CALL_HANDLER_H_
#define XLA_SERVICE_SPMD_CUSTOM_CALL_HANDLER_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/shape_inference.h"
#include "xla/service/spmd/dot_handler.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {

// Creators of custom ops defined by the partitioner itself.

// Creates a custom op that rotates data along `dim` with the given amount.
std::unique_ptr<HloInstruction> CreateCustomCallSPMDInternal_RotateRight(
    HloInstruction* input, int64_t dim, int64_t amount);

// Functor class for creating sharded block-scaled dots with operands of type
// PartitionedHloMX.
class CreateShardedScaledDotFunctor final
    : public CreateShardedFunctorBase<PartitionedHloMX> {
 public:
  CreateShardedScaledDotFunctor(HloCustomCallInstruction* block_scaled_dot,
                                const DotDimensionNumbers& dimension_numbers)
      : block_scaled_dot_(block_scaled_dot),
        dimension_numbers_(dimension_numbers) {}

  // Implements the creation of sharded block-scaled dots.
  absl::StatusOr<HloInstruction*> CreateSharded(
      const PartitionedHloMX& ll, const PartitionedHloMX& rr, SpmdBuilder* b,
      const Window& conv_window) const override {
    HloInstruction* l = ll.operand().hlo();
    HloInstruction* r = rr.operand().hlo();
    HloInstruction* l_scale = ll.scale().hlo();
    HloInstruction* r_scale = rr.scale().hlo();
    TF_ASSIGN_OR_RETURN(Shape sharded_scaled_dot_shape,
                        ShapeInference::InferDotOpShape(
                            l->shape(), r->shape(), dimension_numbers_,
                            /*preferred_element_type=*/
                            block_scaled_dot_->shape().element_type()));

    return b->AddInstruction(HloInstruction::CreateCustomCall(
        sharded_scaled_dot_shape, {l, r, l_scale, r_scale},
        "__op$block_scaled_dot", ""));
  }

 private:
  HloCustomCallInstruction* block_scaled_dot_;
  const DotDimensionNumbers& dimension_numbers_;
};

}  // namespace spmd
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_CUSTOM_CALL_HANDLER_H_
