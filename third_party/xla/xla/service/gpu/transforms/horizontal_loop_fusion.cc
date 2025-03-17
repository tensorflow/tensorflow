/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/horizontal_loop_fusion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/sub_byte_normalization.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

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

bool IsShapeDefaultMemorySpace(const Shape& shape) {
  bool are_all_subshapes_default_space = true;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        are_all_subshapes_default_space &=
            LayoutUtil::MemorySpace(subshape) == Layout::kDefaultMemorySpace;
      });
  return are_all_subshapes_default_space;
}

class HorizontalLoopFusionImpl {
 public:
  explicit HorizontalLoopFusionImpl(
      HloComputation* computation,
      const se::DeviceDescription& device_description, absl::string_view prefix)
      : computation_(computation),
        device_description_(device_description),
        prefix_(prefix) {}

  ~HorizontalLoopFusionImpl() = default;

  absl::StatusOr<bool> Run();

 private:
  absl::Status Fuse(absl::Span<HloInstruction*> fused_fusion_instrs,
                    bool sliced_input_fusion,
                    std::vector<HloInstruction*>& to_fuse_candidates);

  // If `sliced_input_fusion` is true, Horizontally fuses `fused_fusion_instrs`
  // into kInput computation, else fuses `fused_fusion_instrs` into kLoop
  // computation.
  //
  // It is required that each of `fused_fusion_instrs` is a kLoop fusion. Also,
  // we require their numbers of outputs to be the same, so that each output
  // will be fused/concatenated with the same number of outputs from other fused
  // fusion instrs. Then, all the fused outputs still have the same shapes for
  // kernel generation.
  //
  // Returns the fused computation in `uniq_computation` and the operands that
  // are used by `uniq_computation`.
  absl::Status CreateFusedComputation(
      absl::Span<HloInstruction*> fused_fusion_instrs,
      std::unique_ptr<HloComputation>* uniq_computation,
      std::vector<HloInstruction*>* bound_operands, bool sliced_input_fusion);

  // Horizontally fuses the operands of consumer instruction,
  // `sliced_input_fusion` controls whether kInput or kLoop type fused
  // instruction want to be created. `to_fuse_candidates` is the instruction
  // stack that we want to try horizontally fuse its operands, when we create a
  // new fusion instruction, we push it to the stack in hope to further fuse its
  // operands.
  absl::StatusOr<bool> FuseConsumerOperands(
      HloInstruction* consumer, bool sliced_input_fusion,
      std::vector<HloInstruction*>& to_fuse_candidates);

  // FusionCandidates collects profitable candidates for a given consumer
  // instruction. GetNextSpanOfFusions() can then be iteratively invoked to
  // acquire the next set of fusion candidates based on some heuristics.
  class FusionCandidates {
   public:
    explicit FusionCandidates(HloInstruction* consumer,
                              bool sliced_input_fusion,
                              const se::DeviceDescription& device_description)
        : fusible_instrs_(),
          pos_(0),
          sliced_input_fusion_(sliced_input_fusion) {
      Initialize(consumer, device_description);
    }

    // Gets a span of fusions to be fused.
    absl::Span<HloInstruction*> GetNextSpanOfFusions();

   private:
    void Initialize(HloInstruction* consumer,
                    const se::DeviceDescription& device_description);

    std::vector<HloInstruction*> fusible_instrs_;
    // `pos_` points to the start position of the next span.
    size_t pos_;
    // `sliced_input_fusion_` flag controls whether we want to fuse
    // into kLoop (false) or kInput (True) type kernel
    bool sliced_input_fusion_;
  };

  HloComputation* computation_;
  const se::DeviceDescription& device_description_;
  std::string prefix_;
};  // HorizontalLoopFusionImpl

bool IsConcatenationInputFusion(const HloInstruction& instr) {
  return instr.IsInputFusion() &&
         instr.fused_expression_root()->opcode() == HloOpcode::kConcatenate;
}

bool IsDynamicUpdateSliceFusion(const HloInstruction* instr) {
  if (HloPredicateIsNotOp<HloOpcode::kFusion>(instr)) {
    return false;
  }
  auto root = instr->fused_expression_root();
  if (HloPredicateIsOp<HloOpcode::kTuple>(root)) {
    return absl::c_any_of(root->operands(),
                          HloPredicateIsOp<HloOpcode::kDynamicUpdateSlice>);
  }
  return HloPredicateIsOp<HloOpcode::kDynamicUpdateSlice>(root);
}

bool IsFusibleCandidate(const HloInstruction& instr,
                        const se::DeviceDescription& device_description) {
  // For now, we do not support fusing instruction with control flow.
  if (!instr.control_successors().empty() ||
      !instr.control_predecessors().empty()) {
    return false;
  }

  if (IsNestableVariadicReduction(instr, device_description) ||
      IsNestableVariadicReduceWindow(instr)) {
    return false;
  }

  // Only consider instructions with default memory space operands and outputs
  // to be fusable.
  if (!IsShapeDefaultMemorySpace(instr.shape())) return false;
  for (auto operand : instr.operands()) {
    if (!IsShapeDefaultMemorySpace(operand->shape())) return false;
  }

  // Require no further check for element-wise instructions.
  if (instr.IsElementwise() && instr.operand_count() > 0) {
    return true;
  }

  if (!(instr.IsLoopFusion() || IsConcatenationInputFusion(instr))) {
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
bool IsProfitableFusionCandidate(const HloInstruction& instr,
                                 bool sliced_input_fusion) {
  // For kLoop fused kernel, each GPU thread will process 1 or more elements
  // from each horizontal fused operands, while for kInput fused kernel, each
  // GPU thread can only process 1 element. From experience, we enable larger
  // tensor size threshold for kLoop fusion.
  const int64_t kShapeThreshold =
      (sliced_input_fusion || IsConcatenationInputFusion(instr)) ? 128 * 2048
                                                                 : 8192 * 8192;
  const int64_t kInstrCountThreshold = sliced_input_fusion ? 30 : 128;
  const HloInstruction* root = (instr.opcode() == HloOpcode::kFusion)
                                   ? instr.fused_expression_root()
                                   : &instr;

  // Too large shapes are not easily profitable.
  if (HloPredicateIsOp<HloOpcode::kTuple>(root)) {
    // Since all output shapes are the same, use the first shape as the
    // representative.
    root = root->operand(0);
  }
  if (ShapeUtil::ElementsIn(root->shape()) > kShapeThreshold) {
    VLOG(2) << "Profitable check failed due to element count with "
               "sliced_input_fusion="
            << sliced_input_fusion;
    return false;
  }

  // Having too many instructions is not easily profitable.
  if (instr.opcode() == HloOpcode::kFusion &&
      instr.fused_instruction_count() > kInstrCountThreshold) {
    return false;
  }

  return true;
}

// Returns whether any operand of `instr` is an instruction that is shared with
// `fusion_instrs`.
bool AnyOperandIsSharedAmongFusions(
    const HloInstruction* instr,
    const absl::flat_hash_set<HloInstruction*>& fusion_instrs) {
  return absl::c_any_of(instr->operands(), [&](const HloInstruction* opnd) {
    return absl::c_any_of(opnd->users(), [&](const HloInstruction* user) {
      return user != instr && fusion_instrs.contains(user);
    });
  });
}

HloInstruction* LatestNonTrivialAncestor(HloInstruction* hlo) {
  if (HloPredicateIsOp<HloOpcode::kGetTupleElement, HloOpcode::kBitcast>(hlo)) {
    return LatestNonTrivialAncestor(hlo->mutable_operand(0));
  }
  return hlo;
}

void HorizontalLoopFusionImpl::FusionCandidates::Initialize(
    HloInstruction* consumer, const se::DeviceDescription& device_description) {
  VLOG(4) << "Considering fusing inputs of " << consumer->ToShortString();

  // First, find out all potential target candidates. We will filter out
  // unsupported/non-profitable cases below.
  absl::flat_hash_set<HloInstruction*> fusible_candidates;
  std::vector<HloInstruction*> ordered_fusible_candidates;
  for (HloInstruction* opnd : consumer->operands()) {
    HloInstruction* predecessor = LatestNonTrivialAncestor(opnd);
    if (IsFusibleCandidate(*predecessor, device_description)) {
      if (fusible_candidates.insert(predecessor).second) {
        // Add unseen fusion to ordered list.
        VLOG(4) << "Considering " << predecessor->ToShortString();
        ordered_fusible_candidates.push_back(predecessor);
      }
    }
  }

  for (HloInstruction* instr : ordered_fusible_candidates) {
    if (!IsConsumerTheOnlyNonRootUser(*instr, *consumer)) {
      VLOG(2) << "sliced_input_fusion=" << sliced_input_fusion_
              << " rejects maybe illegal instr " << instr->ToString()
              << "; including it may create cycles in HLO.";
      continue;
    } else if (!IsProfitableFusionCandidate(*instr, sliced_input_fusion_)) {
      VLOG(2) << "sliced_input_fusion=" << sliced_input_fusion_
              << " rejects may-not-be profitable fusion instr"
              << instr->ToString();
      continue;
    } else if (IsDynamicUpdateSliceFusion(instr) &&
               AnyOperandIsSharedAmongFusions(instr, fusible_candidates)) {
      // Don't fuse DUS fusions with shared operands because we cannot
      // i/o alias the produced horizontal fusion due to the concat insertion.
      VLOG(2) << "sliced_input_fusion=" << sliced_input_fusion_
              << " rejects the DUS fusion because it shares an operand with"
              << " other fusion candidates: " << instr->ToString();
      continue;
    } else {
      // Encapsulate it into a fusion computation for unified representation
      // for later processing.
      fusible_instrs_.push_back(instr);
    }
  }

  // Sort `fusible_instrs_` according to output types, the number of outputs,
  // instruction counts, output tensor element count. For sliced input fusion,
  // we only fuse instructions with the same number/type of outputs.
  // For kLoop fusion, we in addition require the same output shape.
  // We did a sort here so the fusion candidates is
  // populating a continuous span.
  std::sort(fusible_instrs_.begin(), fusible_instrs_.end(),
            [&](const HloInstruction* a, const HloInstruction* b) {
              auto make_tuple_for_op = [](const HloInstruction* op) {
                return std::tuple{
                    GetUniqueOutputTypeOfFusible(*op),
                    GetOutputSizeOfFusible(*op), GetInstrCountOfFusible(*op),
                    ShapeUtil::ElementsIn(GetOutputsOfFusible(*op)[0]->shape()),
                    op->unique_id()};
              };
              return make_tuple_for_op(a) < make_tuple_for_op(b);
            });
}

// Gets a next span of fusion instructions to be fused.
absl::Span<HloInstruction*>
HorizontalLoopFusionImpl::FusionCandidates::GetNextSpanOfFusions() {
  if (pos_ >= fusible_instrs_.size()) {
    return absl::Span<HloInstruction*>();
  }

  // CUDA has a parameter size limit of ~4k bytes.
  constexpr int64_t kMaxCudaParamSize = 4000;

  // Fusing too many computations at a time may not be easily profitable and
  // may increase compile time due to large kernels. Set a limit to it.
  // From profiling results, we found an issue that large fused horizontal
  // kernel could have lower E2E perf, though the pure GPU kernel time is
  // shorter. TODO task for understanding why E2E perf regression for large
  // horiizontal fused kernel. Use the experience max fusion batch size based on
  // the fused instruction count of the operand
  const auto kMaxFusionBatchSize = [&]() -> int64_t {
    if (sliced_input_fusion_) {
      return 32;
    } else {
      if (fusible_instrs_[pos_]->opcode() == HloOpcode::kFusion) {
        return 32;
      } else {
        return 64;
      }
    }
  }();

  size_t left = pos_;
  size_t right = pos_;
  size_t accum_num_outputs = 0;
  size_t accum_io_size = 0;

  for (; right < fusible_instrs_.size(); ++right) {
    if (GetUniqueOutputTypeOfFusible(*fusible_instrs_[left]) !=
        GetUniqueOutputTypeOfFusible(*fusible_instrs_[right])) {
      // Cannot fuse computations who have multiple output types.
      VLOG(2) << "different multiple output types";
      break;
    }
    if (GetOutputSizeOfFusible(*fusible_instrs_[left]) !=
        GetOutputSizeOfFusible(*fusible_instrs_[right])) {
      // Cannot fuse computations who have different numbers of outputs.
      VLOG(2) << "different number of outputs";
      break;
    }
    if (!sliced_input_fusion_ &&
        !ShapeUtil::EqualIgnoringElementType(
            GetOutputsOfFusible(*fusible_instrs_[left])[0]->shape(),
            GetOutputsOfFusible(*fusible_instrs_[right])[0]->shape())) {
      // This is for fusing into kLoop type kernel, so we requires that each
      // fusion operand have the same shape
      VLOG(2) << "different output shape";
      break;
    }
    size_t num_outputs = GetOutputSizeOfFusible(*fusible_instrs_[right]);
    accum_num_outputs += num_outputs;
    if (accum_num_outputs >= kMaxFusionBatchSize) {
      // Hit max fusion batch size.
      VLOG(2) << "hit max fusion batch size: " << accum_num_outputs;
      break;
    }
    accum_io_size += fusible_instrs_.at(right)->operand_count() + num_outputs;
    if (accum_io_size * 8 >= kMaxCudaParamSize) {
      VLOG(2) << "hit max cuda param size: " << accum_io_size;
      break;
    }
  }

  // If right was not incremented, it means that the `left` instruction already
  // exceeds one of the limits. We can't do anything about that fusion here, so
  // we return a span of one instruction.
  if (left == right) {
    ++right;
  }

  VLOG(2) << "horizontal fuse get instruction span with " << (right - left)
          << " instructions for sliced_input_fusion=" << sliced_input_fusion_
          << " fusion";
  pos_ = right;
  return absl::MakeSpan(fusible_instrs_).subspan(left, right - left);
}

absl::StatusOr<bool> HorizontalLoopFusionImpl::FuseConsumerOperands(
    HloInstruction* consumer, bool sliced_input_fusion,
    std::vector<HloInstruction*>& to_fuse_candidates) {
  bool changed = false;
  FusionCandidates loop_fusion_candidates(consumer, sliced_input_fusion,
                                          device_description_);
  while (true) {
    auto fusibles = loop_fusion_candidates.GetNextSpanOfFusions();
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
      VLOG(2) << "next candidate: " << instr->ToString();
      if (HloPredicateIsOp<HloOpcode::kFusion>(instr)) {
        fusion_instrs.push_back(instr);
      } else {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * fusion_instr,
            MakeFusionInstruction(instr, HloInstruction::FusionKind::kLoop));
        fusion_instrs.push_back(fusion_instr);
      }
    }

    TF_RETURN_IF_ERROR(Fuse(absl::MakeSpan(fusion_instrs), sliced_input_fusion,
                            to_fuse_candidates));
  }
  return changed;
}

absl::Status HorizontalLoopFusionImpl::CreateFusedComputation(
    absl::Span<HloInstruction*> fused_fusion_instrs,
    std::unique_ptr<HloComputation>* uniq_computation,
    std::vector<HloInstruction*>* bound_operands, bool sliced_input_fusion) {
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
      if (HloPredicateIsOp<HloOpcode::kParameter>(old_instr) ||
          (sliced_input_fusion &&
           HloPredicateIsOp<HloOpcode::kTuple>(old_instr) &&
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

  // Since we require each fusion to have the same number of outputs, we can
  // simply use the first fusion as the representative for output size.
  size_t fused_instr_output_size =
      GetOutputSizeOfFusible(*fused_fusion_instrs[0]);

  if (sliced_input_fusion) {
    // Fusing into kInput fusion
    std::vector<HloInstruction*> concated_outputs;
    for (size_t i = 0; i < fused_instr_output_size; ++i) {
      std::vector<HloInstruction*> instr_outputs(fused_fusion_instrs.size());
      for (size_t j = 0; j < fused_fusion_instrs.size(); ++j) {
        const HloInstruction* old_output =
            GetOutputsOfFusible(*fused_fusion_instrs[j])[i];
        HloInstruction* new_output = clone_map[old_output];
        if (new_output->shape().rank() == 1) {
          instr_outputs[j] = new_output;
        } else {
          if (!LayoutUtil::IsMonotonicWithDim0Major(
                  new_output->shape().layout())) {
            new_output = comp->AddInstruction(HloInstruction::CreateBitcast(
                ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
                    new_output->shape()),
                new_output));
          }
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

  } else {
    // Fusing into kLoop fusion
    std::vector<HloInstruction*> tuple_operands(fused_instr_output_size *
                                                fused_fusion_instrs.size());
    // If fusing into kLoop fusion, the new fusion root is tuple of fused
    // fusion computaton's root.
    for (size_t i = 0; i < fused_instr_output_size; ++i) {
      for (size_t j = 0; j < fused_fusion_instrs.size(); ++j) {
        const HloInstruction* old_output =
            GetOutputsOfFusible(*fused_fusion_instrs[j])[i];
        HloInstruction* new_output = clone_map[old_output];
        tuple_operands[fused_instr_output_size * j + i] = new_output;
      }
    }
    // Make a tuple instruction of fused instruction outputs as
    // the root of fused computation.
    HloInstruction* tuple =
        comp->AddInstruction(HloInstruction::CreateTuple(tuple_operands));
    comp->set_root_instruction(tuple, /*accept_different_shape=*/true);
    TF_RETURN_IF_ERROR(comp->RemoveInstruction(dummy_root));
  }

  return absl::OkStatus();
}

absl::Status HorizontalLoopFusionImpl::Fuse(
    absl::Span<HloInstruction*> fused_fusion_instrs, bool sliced_input_fusion,
    std::vector<HloInstruction*>& to_fuse_candidates) {
  // Fuse fused_fusion_instrs and replace them with the new fused computation.
  std::unique_ptr<HloComputation> uniq_computation;
  std::vector<HloInstruction*> bound_operands;

  TF_RETURN_IF_ERROR(CreateFusedComputation(fused_fusion_instrs,
                                            &uniq_computation, &bound_operands,
                                            sliced_input_fusion));

  HloComputation* fused_comp = computation_->parent()->AddEmbeddedComputation(
      std::move(uniq_computation));
  HloInstruction* hori_fusion_instr = computation_->AddInstruction(
      HloInstruction::CreateFusion(fused_comp->root_instruction()->shape(),
                                   sliced_input_fusion
                                       ? HloInstruction::FusionKind::kInput
                                       : HloInstruction::FusionKind::kLoop,
                                   bound_operands, fused_comp, prefix_),
      &fused_comp->root_instruction()->metadata());
  fused_comp->SetFusionInstruction(hori_fusion_instr);

  // we push the newly fused instruction into fusion candidate stack, because
  // the operands of the newly fused instruction could now be possible to be
  // horizontally fused.
  to_fuse_candidates.push_back(hori_fusion_instr);

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
      if (output->shape().rank() == 1) {
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
  return absl::OkStatus();
}

absl::StatusOr<bool> HorizontalLoopFusionImpl::Run() {
  bool changed = false;
  XLA_VLOG_LINES(10, computation_->ToString());

  // Traverse from use to def. Bitcasts are placed after h-fusions to resolve
  // shape mismatch but bitcasts could prevent future h-fusion from happening.
  // So, a bottom-up, use-to-def order should be more favorable. It also helps
  // to save compiler iterations to reach the fixed point.
  std::vector<HloInstruction*> to_fuse_candidates =
      computation_->MakeInstructionPostOrder();

  while (!to_fuse_candidates.empty()) {
    HloInstruction* consumer = to_fuse_candidates.back();
    to_fuse_candidates.pop_back();

    // The consumer may be an operand of a previously fused instruction, so it
    // will no longer be valid - skip it. If the consumer has less than two
    // operands, there is nothing to fuse.
    if (consumer->IsDead() || consumer->operand_count() < 2) {
      continue;
    }

    // we first try to fuse into kLoop fusion instruction for those operands
    // that have the same shape.
    TF_ASSIGN_OR_RETURN(
        bool loop_fusion_changed,
        FuseConsumerOperands(consumer, false, to_fuse_candidates));

    // for the remaining operands with different shape, we further try fuse them
    // into kInput fusion instruction.
    TF_ASSIGN_OR_RETURN(
        bool sliced_input_fusion_changed,
        FuseConsumerOperands(consumer, true, to_fuse_candidates));

    changed = changed || loop_fusion_changed || sliced_input_fusion_changed;
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> HorizontalLoopFusion::RunOnComputation(
    HloComputation* computation) {
  HorizontalLoopFusionImpl horizontal_fusion_impl(computation,
                                                  device_description_, prefix_);
  return horizontal_fusion_impl.Run();
}

absl::StatusOr<bool> HorizontalLoopFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "Run horizontal fusion.";

  bool any_changed = false;
  if (only_entry_computation_) {
    TF_ASSIGN_OR_RETURN(any_changed,
                        RunOnComputation(module->entry_computation()));
  } else {
    for (HloComputation* computation :
         GetFusibleComputations(*module, execution_threads)) {
      TF_ASSIGN_OR_RETURN(bool changed, RunOnComputation(computation));
      any_changed |= changed;
    }
  }

  if (any_changed) {
    // Correctly set element_size_in_bits for any sub-byte added slice and
    // concatenate instructions
    TF_ASSIGN_OR_RETURN(
        [[maybe_unused]] bool unused,
        SubByteNormalization{SubByteNormalization::SET_ELEMENT_SIZE}.Run(
            module));
  }

  return any_changed;
}

}  // namespace gpu
}  // namespace xla
