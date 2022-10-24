/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/horizontal_input_fusion.h"

#include <algorithm>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla {
namespace gpu {

namespace {

// Gets the representative input shape of the multi-output fusion.
Shape GetInputShapeForMultiOutputFusion(const HloInstruction& instr) {
  // Get the HLO that determines the emitter used for lowering.
  const HloInstruction* real_hero = GetRealHeroForMultiOutputFusion(instr);
  if (real_hero->operands().empty()) {
    // Simply return an empty shape if the representative node has no input
    // operands.
    return Shape();
  } else {
    return real_hero->operand(0)->shape();
  }
}

class HorizontalInputFusionImpl {
 public:
  explicit HorizontalInputFusionImpl(HloComputation* computation)
      : computation_(computation) {}

  ~HorizontalInputFusionImpl() {}

  StatusOr<bool> Run();

 private:
  HloComputation* computation_;
};  // HorizontalInputFusionImpl

// Compares one-by-one the dimensions of `shape_a` and `shape_b` from left to
// right.
bool CompareShapeDimsFromLeftToRight(const Shape& shape_a,
                                     const Shape& shape_b) {
  if (shape_a.rank() != shape_b.rank()) {
    return shape_a.rank() < shape_b.rank();
  }
  auto dims_a = shape_a.dimensions();
  auto dims_b = shape_b.dimensions();
  for (size_t i = 0; i < dims_a.size(); ++i) {
    if (dims_a[i] != dims_b[i]) {
      return dims_a[i] < dims_b[i];
    }
  }
  return true;
}

std::vector<HloInstruction*> FindAndSortFusionCandidates(
    HloInstruction* consumer) {
  absl::flat_hash_set<HloInstruction*> fusion_instr_set;
  std::vector<HloInstruction*> fusion_instrs;
  for (HloInstruction* opnd : consumer->operands()) {
    HloInstruction* predecessor = opnd->LatestNonGteAncestor();
    // Find out the input fusion instructions whose only consumer is `consumer`.
    // This guarantees that fusing these candidates will never create cycles, as
    // there is no back edge.
    if (IsInputFusibleReduction(*predecessor) &&
        IsConsumerTheOnlyNonRootUser(*predecessor, *consumer)) {
      if (fusion_instr_set.insert(predecessor).second) {
        fusion_instrs.push_back(predecessor);
      }
    }
  }

  std::sort(fusion_instrs.begin(), fusion_instrs.end(),
            [&](const HloInstruction* a, const HloInstruction* b) {
              Shape shape_a = GetInputShapeForMultiOutputFusion(*a);
              Shape shape_b = GetInputShapeForMultiOutputFusion(*b);
              if (!ShapeUtil::EqualIgnoringElementType(shape_a, shape_b)) {
                // Sort shapes according to dimensions, so that the same input
                // shapes will be placed adjacent each other.
                return CompareShapeDimsFromLeftToRight(shape_a, shape_b);
              }
              // Sort `fusion_instrs` according to instruction counts, because
              // we'd like to fuse together computations of similar sizes.
              return GetInstrCountOfFusible(*a) < GetInstrCountOfFusible(*b);
            });

  return fusion_instrs;
}

StatusOr<bool> HorizontalInputFusionImpl::Run() {
  bool changed = false;
  XLA_VLOG_LINES(3, computation_->ToString());

  // Using def-to-use order is sound since we do not modify users.
  std::vector<HloInstruction*> def_to_use_order =
      computation_->MakeInstructionPostOrder();
  for (HloInstruction* consumer : def_to_use_order) {
    auto candidates = FindAndSortFusionCandidates(consumer);
    if (candidates.size() <= 1) {
      continue;
    }

    // Convert candidates into fusions if needed.
    for (size_t j = 0; j < candidates.size(); ++j) {
      if (candidates[j]->opcode() != HloOpcode::kFusion) {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * fusion_instr,
            MakeFusionInstruction(candidates[j],
                                  HloInstruction::FusionKind::kInput));
        candidates[j] = fusion_instr;
        changed = true;
      }
    }

    size_t fusion_anchor_id = 0;
    for (size_t j = 1; j < candidates.size(); ++j) {
      HloInstruction* fusion_anchor = candidates[fusion_anchor_id];
      HloInstruction* fused = candidates[j];
      if (ShapesCompatibleForMultiOutputFusion(*fusion_anchor, *fused) &&
          FusionFitsInBudget(*fusion_anchor, *fused)) {
        VLOG(3) << "Fuse " << fused->ToString() << " into "
                << fusion_anchor->ToString();
        fusion_anchor->MergeFusionInstructionIntoMultiOutput(fused);
        changed = true;
      } else {
        // Update the `fusion_anchor_id` since `fused` is either not
        // compatible or not beneficial to be fused with current fusion anchor.
        VLOG(3) << j - fusion_anchor_id - 1 << " instructions are fused.";
        fusion_anchor_id = j;
      }
    }
  }

  return changed;
}

}  // namespace

StatusOr<bool> GpuHorizontalInputFusion::RunOnComputation(
    HloComputation* computation) {
  HorizontalInputFusionImpl horizontal_fusion_impl(computation);
  return horizontal_fusion_impl.Run();
}

StatusOr<bool> GpuHorizontalInputFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  VLOG(2) << "Run horizontal input fusion.";
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(changed, RunOnComputation(comp));
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
