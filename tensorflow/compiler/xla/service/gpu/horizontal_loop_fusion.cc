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

#include "tensorflow/compiler/xla/service/gpu/horizontal_loop_fusion.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla {
namespace gpu {

namespace {

PrimitiveType GetUniqueOutputTypeOfFusible(const HloInstruction& fusible) {
  auto outputs = GetOutputsOfFusible(fusible);
  CHECK(!outputs.empty());
  PrimitiveType first_output_type = outputs[0]->shape().element_type();
  for (size_t i = 1; i < outputs.size(); ++i) {
    PrimitiveType cur_output_type = outputs[i]->shape().element_type();
    CHECK(first_output_type == cur_output_type)
        << "Output types are expected to be unique, but see "
        << PrimitiveType_Name(first_output_type) << " and "
        << PrimitiveType_Name(cur_output_type);
  }

  return first_output_type;
}

class HorizontalLoopFusionImpl {
 public:
  explicit HorizontalLoopFusionImpl(HloComputation* computation,
                                    absl::string_view prefix)
      : computation_(computation), prefix_(prefix) {}

  ~HorizontalLoopFusionImpl() {}

  StatusOr<bool> Run();

 private:
  Status Fuse(absl::Span<HloInstruction*> fused_fusion_instrs);

  // Horizontally fuses `fused_fusion_instrs`. It is required that each of
  // `fused_fusion_instrs` is a kLoop fusion. Also, we require their numbers of
  // outputs to be the same, so that each output will be fused/concatenated with
  // the same number of outputs from other fused fusion instrs. Then, all the
  // fused outputs still have the same shapes for kernel generation.
  //
  // Returns the fused computation in `uniq_computation` and the operands that
  // are used by `uniq_computation`.
  Status CreateFusedComputation(
      absl::Span<HloInstruction*> fused_fusion_instrs,
      std::unique_ptr<HloComputation>* uniq_computation,
      std::vector<HloInstruction*>* bound_operands);

  // FusionCandidates collects profitable candidates for a given consumer
  // instruction. GetNextSpanOfFusions() can then be iteratively invoked to
  // acquire the next set of fusion candidates based on some heuristics.
  class FusionCandidates {
   public:
    explicit FusionCandidates(HloInstruction* consumer)
        : fusible_instrs_(), pos_(0) {
      Initialize(consumer);
    }

    // Gets a span of fusions to be fused.
    absl::Span<HloInstruction*> GetNextSpanOfFusions();

   private:
    void Initialize(HloInstruction*);

    std::vector<HloInstruction*> fusible_instrs_;
    // `pos_` points to the start position of the next span.
    size_t pos_;
  };

  HloComputation* computation_;
  std::string prefix_;
};  // HorizontalLoopFusionImpl

bool IsFusibleCandidate(const HloInstruction& instr) {
  // For now, we do not support fusing instruction with control flow.
  if (!instr.control_successors().empty() ||
      !instr.control_predecessors().empty()) {
    return false;
  }

  // Require no further check for element-wise instructions.
  if (instr.IsElementwise() && instr.operand_count() > 0) {
    return true;
  }

  // Exclude fusions other than kLoop.
  if (!instr.IsLoopFusion()) {
    return false;
  }

  // Cannot support fusion who has multiple output types, because the
  // concatenate (inserted for horizontal fusion) requires the same type
  // for all of its operands.
  auto outputs = GetOutputsOfFusible(instr);
  CHECK(!outputs.empty());
  const HloInstruction* first_output = outputs[0];
  for (size_t i = 1; i < outputs.size(); ++i) {
    if (first_output->shape().element_type() !=
        outputs[i]->shape().element_type()) {
      return false;
    }
  }

  return true;
}

// Returns whether `instr` is a profitable candidate to be horizontally fused.
// Since the primary benefit of horizontal fusion comes from reducing the
// kernel launch overhead, we want to exclude the instructions with
// insignificant kernel launch overhead. In other words, we exclude instructions
// if their computation latencies are longer than launch latencies. We estimate
// the computation latency of a given instruction by its shapes and the
// instruction count in its fused computation. We roughly observe that if a
// fusion instruction has shapes smaller than `kShapeThreshold` and has fewer
// instructions than `kInstrCountThreshold`, it is launch-latency-bound and
// profitable by horizontal fusion.
bool IsProfitableFusionCandidate(const HloInstruction& instr) {
  constexpr int64_t kShapeThreshold = 128 * 2048;
  constexpr int64_t kInstrCountThreshold = 30;
  const HloInstruction* root = (instr.opcode() == HloOpcode::kFusion)
                                   ? instr.fused_expression_root()
                                   : &instr;

  // Too large shapes are not easily profitable.
  if (root->opcode() == HloOpcode::kTuple) {
    // Since all output shapes are the same, use the first shape as the
    // representative.
    Shape shape = root->operand(0)->shape();
    if (ShapeUtil::ElementsIn(shape) > kShapeThreshold) {
      return false;
    }
  } else {
    Shape shape = root->shape();
    if (ShapeUtil::ElementsIn(shape) > kShapeThreshold) {
      return false;
    }
  }

  // Having too many instructions is not easily profitable.
  if (instr.opcode() == HloOpcode::kFusion &&
      instr.fused_instruction_count() > kInstrCountThreshold) {
    return false;
  }

  return true;
}

// Returns whether `fusion_instr` has only row-major layouts.
// The horizontal fusion excludes computations with non-row-major layouts,
// because fusing computations with different layouts can result in uncoalesced
// memory accesses and cause great performance overhead.
bool HasOnlyRowMajorLayout(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return LayoutUtil::IsMonotonicWithDim0Major(instr.shape().layout());
  }

  auto fused_instrs = instr.fused_instructions_computation()->instructions();
  for (HloInstruction* i : fused_instrs) {
    if (!LayoutUtil::IsDenseArray(i->shape())) {
      continue;
    }
    if (!LayoutUtil::IsMonotonicWithDim0Major(i->shape().layout())) {
      return false;
    }
  }
  return true;
}

// Returns whether any operand of `instr` is a parameter instruction that
// is shared with `fusion_instrs`.
bool AnyOpndIsParamSharedAmongFusions(
    const HloInstruction* instr,
    const absl::flat_hash_set<HloInstruction*>& fusion_instrs) {
  return absl::c_any_of(instr->operands(), [&](const HloInstruction* opnd) {
    return opnd->opcode() == HloOpcode::kParameter &&
           absl::c_any_of(opnd->users(), [&](const HloInstruction* user) {
             return user != instr && fusion_instrs.contains(user);
           });
  });
}

void HorizontalLoopFusionImpl::FusionCandidates::Initialize(
    HloInstruction* consumer) {
  // First, find out all potential target candidates. We will filter out
  // unsupported/non-profitable cases below.
  absl::flat_hash_set<HloInstruction*> fusible_candidates;
  std::vector<HloInstruction*> ordered_fusible_candidates;
  for (HloInstruction* opnd : consumer->operands()) {
    HloInstruction* predecessor = opnd->LatestNonGteAncestor();
    // We support kLoop fusion and element-wise HLOs now. We may extend the
    // support list if needs arise.
    if (IsFusibleCandidate(*predecessor)) {
      if (fusible_candidates.insert(predecessor).second) {
        // Add unseen fusion to ordered list.
        ordered_fusible_candidates.push_back(predecessor);
      }
    }
  }

  for (HloInstruction* instr : ordered_fusible_candidates) {
    if (!IsConsumerTheOnlyNonRootUser(*instr, *consumer)) {
      VLOG(2) << "Reject maybe illegal instr " << instr->ToString()
              << "; including it may create cycles in HLO.";
      continue;
    } else if (!IsProfitableFusionCandidate(*instr)) {
      VLOG(2) << "Reject may-not-be profitable fusion instr "
              << instr->ToString();
      continue;
    } else if (!HasOnlyRowMajorLayout(*instr)) {
      VLOG(2) << "Reject non-row-major fusion instr " << instr->ToString();
      continue;
    } else if (AnyOpndIsParamSharedAmongFusions(instr, fusible_candidates)) {
      // Don't fuse fusions whose operands are parameter instructions that are
      // shared among fusions because we cannot i/o alias the produced
      // horizontal fusion due to the concat insertion.
      VLOG(2) << "Reject the fusion instr because it shares parameter with"
              << " other fusion candidates, instr: " << instr->ToString();
      continue;
    } else {
      VLOG(2) << "Find a fusion candidate " << instr->ToString();
      // Encapsulate it into a fusion computation for unified representation
      // for later processing.
      fusible_instrs_.push_back(instr);
    }
  }

  // Sort `fusible_instrs_` according to output types, the number of outputs,
  // and instruction counts, because we only fuse instructions with the same
  // number/type of outputs and whose computations have the same instruction
  // count.
  std::sort(
      fusible_instrs_.begin(), fusible_instrs_.end(),
      [&](const HloInstruction* a, const HloInstruction* b) {
        if (GetUniqueOutputTypeOfFusible(*a) !=
            GetUniqueOutputTypeOfFusible(*b)) {
          return GetUniqueOutputTypeOfFusible(*a) <
                 GetUniqueOutputTypeOfFusible(*b);
        } else if (GetOutputSizeOfFusible(*a) != GetOutputSizeOfFusible(*b)) {
          return GetOutputSizeOfFusible(*a) < GetOutputSizeOfFusible(*b);
        } else {
          return GetInstrCountOfFusible(*a) < GetInstrCountOfFusible(*b);
        }
      });
}

// Gets a next span of fusion instructions to be fused.
absl::Span<HloInstruction*>
HorizontalLoopFusionImpl::FusionCandidates::GetNextSpanOfFusions() {
  if (pos_ >= fusible_instrs_.size()) {
    return absl::Span<HloInstruction*>();
  }

  // Fusing too many computations at a time may not be easily profitable and
  // may increase compile time due to large kernels. Set a limit to it.
  constexpr int64_t kMaxFusionBatchSize = 32;
  // CUDA has a parameter size limit of ~4k bytes.
  constexpr int64_t kMaxCudaParamSize = 4000;
  size_t accum_io_size = 0;
  auto reach_max_fusion_batch_size = [&](size_t left, size_t right) -> bool {
    if (right - left >= kMaxFusionBatchSize) {
      return true;
    }

    accum_io_size += fusible_instrs_.at(right)->operand_count() +
                     GetOutputSizeOfFusible(*fusible_instrs_.at(right));

    if (accum_io_size * 8 >= kMaxCudaParamSize) {
      return true;
    }

    return false;
  };

  size_t left = pos_;
  size_t right = pos_ + 1;
  size_t first_output_size = GetOutputSizeOfFusible(*fusible_instrs_[left]);
  PrimitiveType first_output_type =
      GetUniqueOutputTypeOfFusible(*fusible_instrs_[left]);
  for (; right < fusible_instrs_.size(); ++right) {
    PrimitiveType cur_output_type =
        GetUniqueOutputTypeOfFusible(*fusible_instrs_[right]);
    if (first_output_type != cur_output_type) {
      // Cannot fuse computations who have multiple output types.
      break;
    } else if (first_output_size !=
               GetOutputSizeOfFusible(*fusible_instrs_[right])) {
      // Cannot fuse computations who have different numbers of outputs.
      break;
    } else if (GetInstrCountOfFusible(*fusible_instrs_[left]) !=
               GetInstrCountOfFusible(*fusible_instrs_[right])) {
      // Do not fuse computations of different instruction counts as it may
      // introduce control divergence. This is a very simple heuristic to avoid
      // fusing computations with too much discrepancy and we may improve it
      // when the needs arise.
      break;
    } else if (reach_max_fusion_batch_size(left, right)) {
      // Hit max fusion batch size.
      break;
    }
  }

  pos_ = right;
  return absl::MakeSpan(fusible_instrs_).subspan(left, right - left);
}

Status HorizontalLoopFusionImpl::CreateFusedComputation(
    absl::Span<HloInstruction*> fused_fusion_instrs,
    std::unique_ptr<HloComputation>* uniq_computation,
    std::vector<HloInstruction*>* bound_operands) {
  // First, build a computation with only params.
  HloComputation::Builder b(prefix_ + "horizontally_fused_computation");
  size_t fused_comp_param_id = 0;
  for (size_t i = 0; i < fused_fusion_instrs.size(); ++i) {
    auto old_params = fused_fusion_instrs[i]->fused_parameters();
    for (size_t j = 0; j < old_params.size(); ++j) {
      HloInstruction* bound_opnd = fused_fusion_instrs[i]->mutable_operand(j);
      // in a form of param_i_j
      b.AddInstruction(HloInstruction::CreateParameter(
          fused_comp_param_id++, bound_opnd->shape(),
          absl::StrCat("param_", i, "_", j)));
      bound_operands->push_back(bound_opnd);
    }
  }
  // Always create a dummy tuple instruction to serve as the root of the
  // computation, as the existence of a root instruction is required by the
  // HloComputation. The real root instruction will replace it below.
  HloInstruction* dummy_root = b.AddInstruction(
      HloInstruction::CreateTuple(std::vector<HloInstruction*>{}));
  *uniq_computation = b.Build(dummy_root);
  HloComputation* comp = uniq_computation->get();

  // Preparing clone_map, which maps old operand to new operand.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> clone_map;
  size_t new_param_id = 0;
  for (size_t i = 0; i < fused_fusion_instrs.size(); ++i) {
    auto old_params = fused_fusion_instrs[i]->fused_parameters();
    for (size_t j = 0; j < old_params.size(); ++j) {
      HloInstruction* old_param = old_params[j];
      HloInstruction* new_param = comp->parameter_instruction(new_param_id++);
      clone_map.insert({old_param, new_param});
    }
  }

  // Clone every fused computation.
  const OpMetadata* metadata = nullptr;
  for (size_t i = 0; i < fused_fusion_instrs.size(); ++i) {
    auto def_to_use_order = fused_fusion_instrs[i]
                                ->fused_instructions_computation()
                                ->MakeInstructionPostOrder();
    for (HloInstruction* old_instr : def_to_use_order) {
      if (old_instr->opcode() == HloOpcode::kParameter ||
          (old_instr->opcode() == HloOpcode::kTuple &&
           old_instr == fused_fusion_instrs[i]->fused_expression_root())) {
        // Parameters have been created, and we don't need tuples from
        // multi-output fusions, as we will directly reference the tuple
        // operands instead by using GetOutputsOfFusible().
        continue;
      }
      std::vector<HloInstruction*> new_opnds;
      const auto& old_opnds = old_instr->operands();
      new_opnds.reserve(old_opnds.size());
      for (HloInstruction* old_opnd : old_opnds) {
        CHECK(clone_map.find(old_opnd) != clone_map.end());
        new_opnds.push_back(clone_map[old_opnd]);
      }
      HloInstruction* new_instr = comp->AddInstruction(
          old_instr->CloneWithNewOperands(old_instr->shape(), new_opnds));
      clone_map.insert({old_instr, new_instr});
      // Get the metadata from the last fused instruction.
      metadata = &old_instr->metadata();
    }
  }

  std::vector<HloInstruction*> concated_outputs;
  // Since we require each fusion to have the same number of outputs, we can
  // simply use the first fusion as the representative for output size.
  size_t fused_instr_output_size =
      GetOutputSizeOfFusible(*fused_fusion_instrs[0]);
  for (size_t i = 0; i < fused_instr_output_size; ++i) {
    std::vector<HloInstruction*> instr_outputs(fused_fusion_instrs.size());
    for (size_t j = 0; j < fused_fusion_instrs.size(); ++j) {
      const HloInstruction* old_output =
          GetOutputsOfFusible(*fused_fusion_instrs[j])[i];
      HloInstruction* new_output = clone_map[old_output];
      if (new_output->shape().dimensions_size() == 1) {
        instr_outputs[j] = new_output;
      } else {
        Shape new_shape = ShapeUtil::MakeShapeWithDenseLayout(
            new_output->shape().element_type(),
            {ShapeUtil::ElementsIn(new_output->shape())},
            /*minor_to_major=*/std::vector<int64_t>(1, 0));
        TF_ASSIGN_OR_RETURN(instr_outputs[j],
                            MakeReshapeHlo(new_shape, new_output));
      }
    }
    TF_ASSIGN_OR_RETURN(HloInstruction * concated_output,
                        MakeConcatHlo(instr_outputs, 0));
    concated_outputs.push_back(concated_output);
  }

  // Make slices of outputs.
  std::vector<HloInstruction*> output_slices(concated_outputs.size() *
                                             fused_fusion_instrs.size());
  for (size_t i = 0; i < concated_outputs.size(); ++i) {
    HloInstruction* concated_output = concated_outputs[i];
    int64_t slice_start = 0;
    // Create a slice per fused computation.
    for (size_t j = 0; j < fused_fusion_instrs.size(); ++j) {
      const HloInstruction* old_output =
          GetOutputsOfFusible(*fused_fusion_instrs[j])[i];
      Shape shape = old_output->shape();
      int64_t slice_limit = slice_start + ShapeUtil::ElementsIn(shape);
      TF_ASSIGN_OR_RETURN(
          output_slices[concated_outputs.size() * j + i],
          MakeSliceHlo(concated_output, {slice_start}, {slice_limit},
                       /*strides=*/{1}));
      slice_start = slice_limit;
    }
  }

  // Make a tuple of output_slices.
  HloInstruction* tuple = comp->AddInstruction(
      HloInstruction::CreateTuple(output_slices), metadata);
  comp->set_root_instruction(tuple, /*accept_different_shape=*/true);
  TF_RETURN_IF_ERROR(comp->RemoveInstruction(dummy_root));

  return OkStatus();
}

Status HorizontalLoopFusionImpl::Fuse(
    absl::Span<HloInstruction*> fused_fusion_instrs) {
  // Fuse fused_fusion_instrs and replace them with the new fused computation.
  std::unique_ptr<HloComputation> uniq_computation;
  std::vector<HloInstruction*> bound_operands;
  TF_RETURN_IF_ERROR(CreateFusedComputation(
      fused_fusion_instrs, &uniq_computation, &bound_operands));
  HloComputation* fused_comp = computation_->parent()->AddEmbeddedComputation(
      std::move(uniq_computation));
  HloInstruction* hori_fusion_instr = computation_->AddInstruction(
      HloInstruction::CreateFusion(fused_comp->root_instruction()->shape(),
                                   HloInstruction::FusionKind::kInput,
                                   bound_operands, fused_comp, prefix_),
      &fused_comp->root_instruction()->metadata());
  fused_comp->SetFusionInstruction(hori_fusion_instr);

  // Insert bitcasts and replace corresponding users. Note that we do not insert
  // the bitcasts in the fused computation as it does not fit into the slice
  // input fusion pattern. However, inserting bitcasts outside the fused
  // computation creates no performance cost.
  size_t total_output_id = 0;
  for (size_t i = 0; i < fused_fusion_instrs.size(); ++i) {
    std::vector<HloInstruction*> bitcasts_or_gte;
    HloInstruction* fused_instr = fused_fusion_instrs[i];
    size_t num_outputs = GetOutputSizeOfFusible(*fused_instr);
    for (size_t j = 0; j < num_outputs; ++j) {
      const HloInstruction* output = GetOutputsOfFusible(*fused_instr)[j];
      TF_ASSIGN_OR_RETURN(
          HloInstruction * gep,
          MakeGetTupleElementHlo(hori_fusion_instr, total_output_id++));
      // This pass runs late, so useless bitcast won't be cleaned up.
      if (output->shape().dimensions_size() == 1) {
        bitcasts_or_gte.push_back(gep);
      } else {
        bitcasts_or_gte.push_back(computation_->AddInstruction(
            HloInstruction::CreateBitcast(output->shape(), gep)));
      }
    }
    HloInstruction* bitcast_or_tuple =
        (bitcasts_or_gte.size() == 1)
            ? bitcasts_or_gte.at(0)
            : computation_->AddInstruction(
                  HloInstruction::CreateTuple(bitcasts_or_gte));
    HloComputation* old_computation =
        fused_instr->fused_instructions_computation();
    HloModule* module = old_computation->parent();
    TF_RETURN_IF_ERROR(
        computation_->ReplaceInstruction(fused_instr, bitcast_or_tuple));
    TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(old_computation));
  }

  TF_RETURN_IF_ERROR(Cast<HloFusionInstruction>(hori_fusion_instr)
                         ->DeduplicateFusionOperands());

  VLOG(1) << "Fused " << fused_fusion_instrs.size()
          << " instructions into: " << hori_fusion_instr->ToString();
  return OkStatus();
}

StatusOr<bool> HorizontalLoopFusionImpl::Run() {
  bool changed = false;
  XLA_VLOG_LINES(3, computation_->ToString());

  // Traverse from use to def. Bitcasts are placed after h-fusions to resolve
  // shape mismatch but bitcasts could prevent future h-fusion from happening.
  // So, a bottom-up, use-to-def order should be more favorable. It also helps
  // to save compiler iterations to reach the fixed point.
  std::vector<HloInstruction*> use_to_def_order =
      computation_->MakeInstructionPostOrder();
  absl::c_reverse(use_to_def_order);
  for (size_t i = 0; i < use_to_def_order.size(); ++i) {
    HloInstruction* consumer = use_to_def_order[i];
    HorizontalLoopFusionImpl::FusionCandidates fusion_candidates(consumer);
    while (true) {
      auto fusibles = fusion_candidates.GetNextSpanOfFusions();
      if (fusibles.empty()) {
        break;
      } else if (fusibles.size() == 1) {
        // Skip; there is just one fused_instr.
        continue;
      }

      changed = true;
      // Convert fusible into fusion_instrs to simplify the implementation of
      // `Fuse()`.
      std::vector<HloInstruction*> fusion_instrs;
      for (HloInstruction* instr : fusibles) {
        if (instr->opcode() == HloOpcode::kFusion) {
          fusion_instrs.push_back(instr);
        } else {
          TF_ASSIGN_OR_RETURN(
              HloInstruction * fusion_instr,
              MakeFusionInstruction(instr, HloInstruction::FusionKind::kLoop));
          fusion_instrs.push_back(fusion_instr);
        }
      }
      TF_RETURN_IF_ERROR(Fuse(absl::MakeSpan(fusion_instrs)));
    }
  }

  return changed;
}

}  // namespace

StatusOr<bool> GpuHorizontalLoopFusion::RunOnComputation(
    HloComputation* computation) {
  HorizontalLoopFusionImpl horizontal_fusion_impl(computation, prefix_);
  return horizontal_fusion_impl.Run();
}

StatusOr<bool> GpuHorizontalLoopFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  VLOG(2) << "Run horizontal fusion.";

  // Run on the entry computation is actually enough.
  TF_ASSIGN_OR_RETURN(changed, RunOnComputation(module->entry_computation()));

  return changed;
}

}  // namespace gpu
}  // namespace xla
