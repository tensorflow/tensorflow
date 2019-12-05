/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/horizontal_fusion.h"

#include <algorithm>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace xla {
namespace gpu {

namespace {

absl::InlinedVector<HloInstruction*, 2> GetOutputsOfFusion(
    const HloInstruction& insn) {
  CHECK(insn.opcode() == HloOpcode::kFusion);
  auto root = insn.fused_expression_root();
  if (root->opcode() != HloOpcode::kTuple) {
    return {root};
  } else {
    return root->operands();
  }
}

// Return the number of outputs of the fused computation.
size_t GetOutputSizeOfFusion(const HloInstruction& insn) {
  CHECK(insn.opcode() == HloOpcode::kFusion);
  auto root = insn.fused_expression_root();
  if (root->opcode() != HloOpcode::kTuple) {
    return 1;
  } else {
    return ShapeUtil::TupleElementCount(root->shape());
  }
}

class HorizontalFusionImpl {
 public:
  HorizontalFusionImpl(HloComputation* computation)
      : computation_(computation) {}

  ~HorizontalFusionImpl() {}

  StatusOr<bool> Run();

 private:
  Status Fuse(absl::Span<HloInstruction*> fused_fusion_insns);

  Status CreateFusedComputation(
      absl::Span<HloInstruction*> fused_fusion_insns,
      std::unique_ptr<HloComputation>* uniq_computation,
      std::vector<HloInstruction*>* bound_operands);

  // FusionCandidates collects profitable candidates for a given consumer
  // instruction. GetNextSpanOfFusions() can then be iteratively invoked to
  // acquire the next set of fusion candidates based on some heuristics.
  class FusionCandidates {
   public:
    FusionCandidates(HloInstruction* consumer) : fusion_insns_(), pos_(0) {
      Initialize(consumer);
    }

    // Get a span of fusions to be fused.
    absl::Span<HloInstruction*> GetNextSpanOfFusions();

   private:
    void Initialize(HloInstruction*);

    std::vector<HloInstruction*> fusion_insns_;
    // pos_ points to the start position of next span.
    size_t pos_;
  };

  HloComputation* computation_;
};  // HorizontalFusionImpl

void HorizontalFusionImpl::FusionCandidates::Initialize(
    HloInstruction* consumer) {
  // First, find out all fusion instructions. We will filter out
  // unsupported/non-profitable cases below.
  absl::flat_hash_set<HloInstruction*> fusion_insns;
  for (auto opnd : consumer->operands()) {
    auto predecessor = opnd->LatestNonGteAncestor();
    if (predecessor->opcode() == HloOpcode::kFusion) {
      fusion_insns.insert(predecessor);
    }
  }


  auto is_supported_fusion = [&](const HloInstruction& insn) -> bool {
    // Support only kLoop fusion.
    if (!insn.IsLoopFusion()) {
      return false;
    }

    return true;
  };

  std::function<bool(const HloInstruction&)> is_consumer_the_only_user =
      [&](const HloInstruction& insn) {
        return absl::c_all_of(insn.users(), [&](const HloInstruction* user) {
          if (user->opcode() == HloOpcode::kGetTupleElement) {
            // Skip GTE.
            return is_consumer_the_only_user(*user);
          } else if (user == consumer) {
            // user is consumer.
            return true;
          } else if (user == user->parent()->root_instruction()) {
            // Consumed by ROOT is always fine, since it is impossible to create
            // cycles through ROOT.
            return true;
          } else {
            return false;
          }
        });
      };

  auto is_profitable = [](const HloInstruction& insn) -> bool {
    // Do not fuse computation with too large shapes or too many instructions,
    // because it may not be easily profitable.
    constexpr int64 kShapeThreshold = 128 * 2048;
    constexpr int64 kInsnCountThreshold = 30;
    auto root = insn.fused_expression_root();

    // Too large shapes are not easily profitable.
    if (root->opcode() == HloOpcode::kTuple) {
      // Since all output shapes are the same, use the first shape as the
      // representative.
      auto shape = root->operand(0)->shape();
      if (ShapeUtil::ElementsIn(shape) > kShapeThreshold) {
        return false;
      }
    } else {
      auto shape = root->shape();
      if (ShapeUtil::ElementsIn(shape) > kShapeThreshold) {
        return false;
      }
    }

    // Having too many instructions is not easily profitable.
    if (insn.fused_instruction_count() > kInsnCountThreshold) {
      return false;
    }

    return true;
  };

  // Do not fuse computations with non-row-major layouts because it may
  // increase bandwidth pressure.
  auto all_row_major = [](const HloInstruction& fusion_insn) -> bool {
    auto insns = fusion_insn.fused_instructions_computation()->instructions();
    for (auto insn : insns) {
      if (insn->shape().layout().format() != DENSE) {
        continue;
      }
      if (!LayoutUtil::IsMonotonicWithDim0Major(insn->shape().layout())) {
        return false;
      }
    }
    return true;
  };

  for (auto insn : fusion_insns) {
    if (!is_supported_fusion(*insn)) {
      VLOG(2) << "Reject unsupported fusion insn " << insn->ToString();
      continue;
    } else if (!is_consumer_the_only_user(*insn)) {
      VLOG(2) << "Reject maybe illegal insn " << insn->ToString();
      continue;
    } else if (!is_profitable(*insn)) {
      VLOG(2) << "Reject may-not-be profitable fusion insn "
              << insn->ToString();
      continue;
    } else if (!all_row_major(*insn)) {
      VLOG(2) << "Reject non-row-major fusion insn " << insn->ToString();
      continue;
    } else {
      fusion_insns_.push_back(insn);
    }
  }

  // Sort according to the number of outputs and instruciton counts.
  std::sort(fusion_insns_.begin(), fusion_insns_.end(),
            [&](const HloInstruction* a, const HloInstruction* b) {
              if (GetOutputSizeOfFusion(*a) == GetOutputSizeOfFusion(*b)) {
                return a->fused_instruction_count() <
                       b->fused_instruction_count();
              }
              return GetOutputSizeOfFusion(*a) < GetOutputSizeOfFusion(*b);
            });
}

// Get a next span of fusion instructions to be fused.
absl::Span<HloInstruction*>
HorizontalFusionImpl::FusionCandidates::GetNextSpanOfFusions() {
  if (pos_ >= fusion_insns_.size()) {
    return absl::Span<HloInstruction*>();
  }

  // Fusing too many computations at a time may not be easily profitable and
  // may increase compile time due to large kernels. Set a limit to it.
  constexpr int64 kMaxFusionBatchSize = 32;
  // CUDA has a parameter size limit of ~4k bytes.
  constexpr int64 kMaxCudaParamSize = 4000;
  size_t accum_io_size = 0;
  auto reach_max_fusion_batch_size = [&](size_t left, size_t right) -> bool {
    if (right - left >= kMaxFusionBatchSize) {
      return true;
    }

    accum_io_size += fusion_insns_.at(right)->fused_parameters().size() +
                     GetOutputSizeOfFusion(*fusion_insns_.at(right));

    if (accum_io_size * 8 >= kMaxCudaParamSize) {
      return true;
    }

    return false;
  };

  size_t left = pos_;
  size_t right = pos_ + 1;
  size_t first_output_size = GetOutputSizeOfFusion(*fusion_insns_[left]);
  for (; right < fusion_insns_.size(); ++right) {
    if (first_output_size != GetOutputSizeOfFusion(*fusion_insns_[right])) {
      // Cannot fuse computations who have different numbers of outputs.
      break;
    } else if (fusion_insns_[left]->fused_instruction_count() !=
               fusion_insns_[right]->fused_instruction_count()) {
      // Do not fuse computations of different instruction counts as it may
      // introduce control divergence.
      break;
    } else if (reach_max_fusion_batch_size(left, right)) {
      // Hit max fusion batch size.
      break;
    }
  }

  pos_ = right;
  return absl::MakeSpan(fusion_insns_).subspan(left, right - left);
}

Status HorizontalFusionImpl::CreateFusedComputation(
    absl::Span<HloInstruction*> fused_fusion_insns,
    std::unique_ptr<HloComputation>* uniq_computation,
    std::vector<HloInstruction*>* bound_operands) {
  // First, build a computation with only params.
  HloComputation::Builder b("horizontally_fused_computation");
  size_t fused_comp_param_id = 0;
  for (size_t i = 0; i < fused_fusion_insns.size(); ++i) {
    auto old_params = fused_fusion_insns[i]->fused_parameters();
    for (size_t j = 0; j < old_params.size(); ++j) {
      auto bound_opnd = fused_fusion_insns[i]->mutable_operand(j);
      // in a form of param_i_j
      auto new_param = b.AddInstruction(HloInstruction::CreateParameter(
          fused_comp_param_id++, bound_opnd->shape(),
          absl::StrCat("param_", i, ".", j)));
      bound_operands->push_back(bound_opnd);
    }
  }
  *uniq_computation = b.Build();
  auto* comp = uniq_computation->get();

  // Preparing clone_map, which maps old operand to new operand.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> clone_map;
  size_t new_param_id = 0;
  for (size_t i = 0; i < fused_fusion_insns.size(); ++i) {
    auto old_params = fused_fusion_insns[i]->fused_parameters();
    for (size_t j = 0; j < old_params.size(); ++j) {
      auto old_param = old_params[j];
      auto new_param = comp->parameter_instruction(new_param_id++);
      clone_map.insert({old_param, new_param});
    }
  }

  // Clone every fused computation.
  for (size_t i = 0; i < fused_fusion_insns.size(); ++i) {
    auto def_to_use_order = fused_fusion_insns[i]
                                ->fused_instructions_computation()
                                ->MakeInstructionPostOrder();
    for (auto old_insn : def_to_use_order) {
      if (old_insn->opcode() == HloOpcode::kParameter) {
        // Parameters have been created.
        continue;
      }
      std::vector<HloInstruction*> new_opnds;
      for (auto old_opnd : old_insn->operands()) {
        CHECK(clone_map.find(old_opnd) != clone_map.end());
        new_opnds.push_back(clone_map[old_opnd]);
      }
      auto new_insn = comp->AddInstruction(
          old_insn->CloneWithNewOperands(old_insn->shape(), new_opnds));
      clone_map.insert({old_insn, new_insn});
    }
  }

  std::vector<HloInstruction*> concated_outputs;
  // Since we require each fusion to have the same number of outputs, we can
  // simply use the first fusion as the representative.
  size_t fused_insn_output_size = GetOutputSizeOfFusion(*fused_fusion_insns[0]);
  for (size_t i = 0; i < fused_insn_output_size; ++i) {
    std::vector<HloInstruction*> reshapes(fused_fusion_insns.size());
    for (size_t j = 0; j < fused_fusion_insns.size(); ++j) {
      auto old_output = GetOutputsOfFusion(*fused_fusion_insns[j])[i];
      auto new_output = clone_map[old_output];
      TF_ASSIGN_OR_RETURN(
          reshapes[j],
          MakeReshapeHlo(ShapeUtil::MakeShapeWithLayout(
                             new_output->shape().element_type(),
                             {ShapeUtil::ElementsIn(new_output->shape())},
                             /*minor_to_major=*/std::vector<int64>(1, 0)),
                         new_output));
    }
    TF_ASSIGN_OR_RETURN(auto concated_output, MakeConcatHlo(reshapes, 0));
    concated_outputs.push_back(concated_output);
  }

  // Make slices of outputs.
  std::vector<HloInstruction*> output_slices(concated_outputs.size() *
                                             fused_fusion_insns.size());
  for (size_t i = 0; i < concated_outputs.size(); ++i) {
    auto concated_output = concated_outputs[i];
    int64 slice_start = 0;
    // Create a slice per fused computation.
    for (size_t j = 0; j < fused_fusion_insns.size(); ++j) {
      auto old_output = GetOutputsOfFusion(*fused_fusion_insns[j])[i];
      auto shape = old_output->shape();
      int64 slice_limit = slice_start + ShapeUtil::ElementsIn(shape);
      TF_ASSIGN_OR_RETURN(
          output_slices[concated_outputs.size() * j + i],
          MakeSliceHlo(concated_output, {slice_start}, {slice_limit},
                       /*strides=*/{1}));
      slice_start = slice_limit;
    }
  }

  // Make a tuple of output_slices.
  auto tuple = comp->AddInstruction(HloInstruction::CreateTuple(output_slices));
  comp->set_root_instruction(tuple, /*accept_different_shape=*/true);

  return Status::OK();
}

Status HorizontalFusionImpl::Fuse(
    absl::Span<HloInstruction*> fused_fusion_insns) {
  // Fuse fused_fusion_insns and replace them with the new fused computation.
  std::unique_ptr<HloComputation> uniq_computation;
  std::vector<HloInstruction*> bound_operands;
  TF_RETURN_IF_ERROR(CreateFusedComputation(
      fused_fusion_insns, &uniq_computation, &bound_operands));
  auto fused_comp = computation_->parent()->AddEmbeddedComputation(
      std::move(uniq_computation));
  auto hori_fusion_insn =
      computation_->AddInstruction(HloInstruction::CreateFusion(
          fused_comp->root_instruction()->shape(),
          HloInstruction::FusionKind::kInput, bound_operands, fused_comp));
  fused_comp->SetFusionInstruction(hori_fusion_insn);

  // Insert bitcasts and replace corresponding users. Note that we do not insert
  // the bitcasts in the fused computation as it does not fit into the slice
  // input fusion pattern. However, inserting bitcasts outside the fused
  // computation creates no performance cost.
  size_t total_output_id = 0;
  for (size_t i = 0; i < fused_fusion_insns.size(); ++i) {
    std::vector<HloInstruction*> bitcasts;
    auto fused_insn = fused_fusion_insns[i];
    auto num_outputs = GetOutputSizeOfFusion(*fused_insn);
    for (size_t j = 0; j < num_outputs; ++j) {
      auto output = GetOutputsOfFusion(*fused_insn)[j];
      TF_ASSIGN_OR_RETURN(auto gep, MakeGetTupleElementHlo(hori_fusion_insn,
                                                           total_output_id++));
      bitcasts.push_back(computation_->AddInstruction(
          HloInstruction::CreateBitcast(output->shape(), gep)));
    }
    auto bitcast_or_tuple = (bitcasts.size() == 1)
                                ? bitcasts.at(0)
                                : computation_->AddInstruction(
                                      HloInstruction::CreateTuple(bitcasts));
    computation_->ReplaceInstruction(fused_insn, bitcast_or_tuple);
  }

  return Status::OK();
}

StatusOr<bool> HorizontalFusionImpl::Run() {
  bool changed = false;
  XLA_VLOG_LINES(3, computation_->ToString());

  // Using def-to-use order is sound since we do not modify users.
  std::vector<HloInstruction*> def_to_use_order =
      computation_->MakeInstructionPostOrder();
  for (size_t i = 0; i < def_to_use_order.size(); ++i) {
    auto consumer = def_to_use_order[i];
    HorizontalFusionImpl::FusionCandidates fusion_candidates(consumer);
    while (true) {
      auto fusions = fusion_candidates.GetNextSpanOfFusions();
      if (fusions.size() == 0) {
        break;
      } else if (fusions.size() == 1) {
        // Skip; there is just one fused_insn.
        continue;
      }

      changed = true;
      TF_RETURN_IF_ERROR(Fuse(fusions));
    }
  }

  return changed;
}

}  // anonymous

StatusOr<bool> GpuHorizontalFusion::RunOnComputation(
    HloComputation* computation) {
  HorizontalFusionImpl horizontal_fusion_impl(computation);
  return horizontal_fusion_impl.Run();
}

StatusOr<bool> GpuHorizontalFusion::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Run horizontal fusion.";
  for (auto* comp : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(changed, RunOnComputation(comp));
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
