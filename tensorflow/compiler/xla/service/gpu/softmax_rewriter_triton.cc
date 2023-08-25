/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/compiler/xla/service/gpu/softmax_rewriter_triton.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla::gpu {
namespace {

bool HasDefaultLayout(const Shape& shape) {
  return shape.has_layout() &&
         LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}

bool IsTritonSupportedInstruction(const HloInstruction* instr,
                                  const GpuVersion& gpu_version) {
  if (!instr->shape().IsArray()) {
    return false;
  }

  if (!IsTritonSupportedDataType(instr->shape().element_type(), gpu_version)) {
    return false;
  }

  for (const HloInstruction* operand : instr->operands()) {
    if (!IsTritonSupportedDataType(operand->shape().element_type(),
                                   gpu_version)) {
      return false;
    }
  }

  // TODO(bchetioui): expand with non-trivial instructions.
  if (instr->IsElementwise()) {
    return IsTritonSupportedElementwise(instr->opcode(),
                                        instr->shape().element_type());
  }

  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kParameter:
      return true;
    default:
      return false;
  }
}

// Returns true if a trivially connected producer of 'consumer' with opcode
// 'opcode' exists. If such an instruction is found, the value of 'producer' is
// set to it. The definition of "trivial" operations is as given in
// 'IsTriviallyFusible'.
bool TrivialEdge(HloInstruction** producer, HloInstruction* consumer,
                 HloOpcode opcode, const GpuVersion& gpu_version);

bool BitcastIsTilingNoop(HloInstruction* bitcast,
                         const GpuVersion& gpu_version) {
  CHECK_EQ(bitcast->opcode(), HloOpcode::kBitcast);

  if (ShapeUtil::IsEffectiveScalar(bitcast->shape())) {
    return true;
  }

  // In the Softmax rewriter for now, tiling is derived from a hero reduction
  // operation, which should be reducing its input on the last axis. Therefore,
  // a bitcast is always a no-op with regards to a tile if
  //   (1) it does not change the size of the reduction dimension of its input
  //       (the last one); if its input is already reduced, then (1) is true
  //       by default
  //   (2) the layout of its output is ordered in the same way as the layout of
  //       its input. This is a fuzzy definition, but since we assume fusible
  //       ops to always have a default layout, we can just check if both the
  //       bitcast and its input have a default layout
  auto last_dimension = [](const HloInstruction* instr) {
    return instr->shape().dimensions().back();
  };

  HloInstruction* reduce = nullptr;
  TrivialEdge(&reduce, bitcast->mutable_operand(0), HloOpcode::kReduce,
              gpu_version);

  return (HasDefaultLayout(bitcast->shape()) &&
          HasDefaultLayout(bitcast->operand(0)->shape()) &&
          (reduce != nullptr ||
           last_dimension(bitcast->operand(0)) == last_dimension(bitcast)));
}

bool IsTriviallyFusible(HloInstruction* instr, const GpuVersion& gpu_version,
                        int num_allowed_users = 1) {
  // Checks whether an op is trivially fusible. An op is said to be trivially
  // fusible if it does not increase the amount of memory read/written by the
  // resulting fusion, is compatible with any chosen tiling, and can be
  // codegen'd using Triton. The op is allowed to have up to num_allowed_users
  // users.
  if (instr->user_count() > num_allowed_users ||
      !HasDefaultLayout(instr->shape())) {
    return false;
  }

  if (instr->opcode() == HloOpcode::kBitcast &&
      BitcastIsTilingNoop(instr, gpu_version)) {
    return true;
  }

  if (instr->IsElementwise() && instr->operand_count() == 1) {
    return IsTritonSupportedInstruction(instr, gpu_version);
  }

  if (instr->IsElementwiseBinary() && instr->operand(0) == instr->operand(1)) {
    return IsTritonSupportedInstruction(instr, gpu_version);
  }

  return false;
}

bool TrivialEdge(HloInstruction** producer, HloInstruction* consumer,
                 HloOpcode opcode, const GpuVersion& gpu_version) {
  while (consumer->opcode() != opcode) {
    if (IsTriviallyFusible(consumer, gpu_version)) {
      consumer = consumer->mutable_operand(0);
    } else {
      return false;
    }
  }

  *producer = consumer;
  return true;
}

bool IsTriviallyConnectedProducerOf(HloInstruction* producer,
                                    HloInstruction* consumer,
                                    const GpuVersion& gpu_version) {
  if (producer == consumer) {
    return true;
  }

  HloInstruction* found_producer = consumer;
  while (
      TrivialEdge(&found_producer, consumer, producer->opcode(), gpu_version)) {
    if (found_producer == producer) {
      return true;
    }

    if (!IsTriviallyFusible(found_producer, gpu_version)) {
      return false;
    }

    consumer = found_producer->mutable_operand(0);
  }

  return false;
}

inline bool HasOneUse(const HloInstruction* instr) {
  return instr->user_count() == 1;
}

bool IsTritonSupportedComputation(const HloComputation* computation,
                                  const GpuVersion& gpu_version) {
  for (const HloInstruction* instr : computation->instructions()) {
    if (!IsTritonSupportedInstruction(instr, gpu_version)) {
      return false;
    }
  }
  return true;
}

std::optional<HloInstruction*> MatchesTritonCompatibleClosedReductionDiamond(
    HloInstruction* instr, const GpuVersion& gpu_version) {
  // Return the producer of the following pattern:
  //
  // producer
  // |    \
  // |  reduce_{max,sum,...}
  // |     |
  // |  broadcast
  // |   /
  // binop (elementwise)
  //
  // where each edge is allowed to contain also trivial operations that can be
  // generated by Triton. We mean by "trivial" here those operations that do not
  // increase the amount of memory read/written by the fusion, and that are
  // compatible with any chosen tiling.
  //
  // We also assume that the reduction is done on the last axis of the producer
  // array.
  std::optional<HloInstruction*> match_failure = std::nullopt;

  if (!instr->IsElementwiseBinary() ||
      !IsTritonSupportedInstruction(instr, gpu_version)) {
    return match_failure;
  }

  HloInstruction* producer;
  HloInstruction* broadcast;
  HloInstruction* reduce;

  if (!(TrivialEdge(&broadcast, instr->mutable_operand(1),
                    HloOpcode::kBroadcast, gpu_version) &&
        TrivialEdge(&reduce, broadcast->mutable_operand(0), HloOpcode::kReduce,
                    gpu_version) &&
        HasDefaultLayout(broadcast->shape()) &&
        HasDefaultLayout(reduce->shape()) && reduce->operand_count() == 2 &&
        reduce->operand(1)->opcode() == HloOpcode::kConstant &&
        IsTritonSupportedComputation(reduce->to_apply(), gpu_version))) {
    return match_failure;
  }

  if (!HasOneUse(broadcast) || !HasOneUse(reduce)) {
    return match_failure;
  }

  producer = reduce->mutable_operand(0);

  if (!(reduce->dimensions().size() == 1 &&
        reduce->dimensions(0) == producer->shape().rank() - 1 &&
        !absl::c_linear_search(broadcast->dimensions(),
                               broadcast->shape().rank() - 1))) {
    return match_failure;
  }

  // TODO(b/291204753): remove this filter. This heuristic enables flipping the
  // default flag while filtering out cases that could result in regressions.
  if (reduce->operand(0)->shape().dimensions().back() < 64) {
    return match_failure;
  }

  while (IsTriviallyFusible(producer, gpu_version)) {
    producer = producer->mutable_operand(0);
  }

  if (!HasDefaultLayout(producer->shape()) ||
      !IsTriviallyConnectedProducerOf(producer, instr->mutable_operand(0),
                                      gpu_version) ||
      !(producer == instr->operand(0) ||
        instr->operand(0)->user_count() == 1)) {
    return match_failure;
  }

  return producer;
}

// Finds the first non-fusible producer of a diamond. This instruction is either
//   1. the direct producer of the diamond, if that producer is used more than
//      twice and/or is not otherwise trivially fusible
//   2. the first parent instruction of the producer of the diamond such that
//      that instruction is used more than once, and/or is not trivially
//      fusible.
HloInstruction* FindFirstNonFusibleDiamondProducer(
    HloInstruction* diamond_producer, const GpuVersion& gpu_version) {
  if (IsTriviallyFusible(diamond_producer, gpu_version,
                         /*num_allowed_users=*/2)) {
    diamond_producer = diamond_producer->mutable_operand(0);
    while (IsTriviallyFusible(diamond_producer, gpu_version)) {
      diamond_producer = diamond_producer->mutable_operand(0);
    }
  }

  return diamond_producer;
}

Status FuseSoftmax(HloInstruction* root, HloInstruction* producer) {
  std::string suggested_name = "triton_softmax";
  HloComputation::Builder builder(absl::StrCat(suggested_name, "_computation"));
  // Original instruction -> fused one.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;

  old_to_new_mapping[producer] = builder.AddInstruction(
      HloInstruction::CreateParameter(0, producer->shape(), "parameter_0"));

  std::function<void(const HloInstruction*)> create_computation =
      [&](const HloInstruction* instr) -> void {
    if (old_to_new_mapping.contains(instr)) {
      return;
    }
    std::vector<HloInstruction*> new_operands;
    for (const HloInstruction* operand : instr->operands()) {
      create_computation(operand);
      new_operands.push_back(old_to_new_mapping[operand]);
    }
    old_to_new_mapping[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), new_operands));
  };
  create_computation(root);

  HloComputation* computation =
      root->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                           /*is_entry=*/false);

  HloInstruction* softmax_fusion =
      root->parent()->AddInstruction(HloInstruction::CreateFusion(
          root->shape(), HloInstruction::FusionKind::kCustom,
          std::vector<HloInstruction*>({producer}), computation));

  softmax_fusion->GetModule()->SetAndUniquifyInstrName(softmax_fusion,
                                                       suggested_name);

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      softmax_fusion->backend_config<FusionBackendConfig>());
  backend_config.set_kind(std::string(kTritonSoftmaxFusionKind));
  TF_RETURN_IF_ERROR(softmax_fusion->set_backend_config(backend_config));

  if (root->IsRoot()) {
    root->parent()->set_root_instruction(softmax_fusion);
    TF_RETURN_IF_ERROR(
        root->parent()->RemoveInstructionAndUnusedOperands(root));
  } else {
    TF_RETURN_IF_ERROR(
        root->parent()->ReplaceInstruction(root, softmax_fusion));
  }

  VLOG(5) << softmax_fusion->ToString();
  return OkStatus();
}

struct DiamondDescriptor {
  HloInstruction* root;
  HloInstruction* producer;
};

using DiamondChainDescriptor = DiamondDescriptor;
}  // anonymous namespace

StatusOr<bool> SoftmaxRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<DiamondDescriptor> matched_diamonds;

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    if (comp->IsCustomCallComputation()) {
      continue;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      PrimitiveType element_ty = instr->shape().element_type();
      // TODO(b/281980675): ensure that code generation also works well for FP8
      // and BF16. This fails for the moment due to these data types requiring
      // float normalization.
      if (element_ty != F16 && element_ty != F32) {
        continue;
      }

      if (auto producer = MatchesTritonCompatibleClosedReductionDiamond(
              instr, gpu_version_)) {
        matched_diamonds.push_back(DiamondDescriptor{instr, producer.value()});
      }
    }
  }

  if (matched_diamonds.empty()) {
    return false;
  }

  auto reduction_dimension_size_from_diamond_root =
      [](HloInstruction* diamond_root) {
        HloInstruction* instr = diamond_root->mutable_operand(1);
        while (instr->opcode() != HloOpcode::kReduce) {
          instr = instr->mutable_operand(0);
        }

        int operand_rank = instr->operand(0)->shape().rank();
        CHECK_EQ(instr->dimensions().size(), 1);
        CHECK_EQ(instr->dimensions(0), operand_rank - 1);
        return instr->operand(0)->shape().dimensions(operand_rank - 1);
      };

  auto last_trivially_fusible_user = [&](HloInstruction* instr) {
    while (HasOneUse(instr) && !instr->IsRoot() &&
           IsTriviallyFusible(instr->users().front(), gpu_version_)) {
      instr = instr->users().front();
    }

    // We do not care about the number of users for the last instruction of the
    // fusion, so attempt to fuse one more instruction with this relaxed
    // restriction.
    if (HasOneUse(instr) && !instr->IsRoot() &&
        IsTriviallyFusible(
            instr->users().front(), gpu_version_,
            /*num_allowed_users=*/instr->users().front()->user_count())) {
      instr = instr->users().front();
    }
    return instr;
  };

  // If we matched several diamonds, it may be possible for some of them to be
  // fused together. This is the case if the following conditions hold:
  //   1. The path between the root of diamond n towards the producer of
  //      diamond n+1 is composed only of trivially fusible operations. In that
  //      case, the first non-trivially fusible producer of diamond n+1 must be
  //      exactly the root of diamond n.
  //   2. The root of diamond n/first non-fusible producer of diamond n+1 must
  //      have
  //        a. exactly one user if it is not exactly the producer of diamond
  //           n+1;
  //        b/ exactly two users otherwise.
  //   3. The axis being reduced must have the same length in all the diamonds
  //      being fused together.
  //
  // Crucially, this approach relies on a diamond root never being considered a
  // trivially fusible operation.
  std::vector<DiamondChainDescriptor> diamond_chains;
  HloInstruction* current_fusion_producer = FindFirstNonFusibleDiamondProducer(
      matched_diamonds.front().producer, gpu_version_);
  int current_reduce_dimension_size =
      reduction_dimension_size_from_diamond_root(matched_diamonds.front().root);

  for (int diamond_idx = 1; diamond_idx < matched_diamonds.size();
       ++diamond_idx) {
    auto [diamond_root, diamond_producer] = matched_diamonds[diamond_idx];
    HloInstruction* previous_diamond_root =
        matched_diamonds[diamond_idx - 1].root;

    HloInstruction* first_non_fusible_diamond_producer =
        FindFirstNonFusibleDiamondProducer(diamond_producer, gpu_version_);

    int diamond_reduce_dimension_size =
        reduction_dimension_size_from_diamond_root(diamond_root);

    if (first_non_fusible_diamond_producer == previous_diamond_root &&  // 1
        ((first_non_fusible_diamond_producer != diamond_producer &&
          HasOneUse(first_non_fusible_diamond_producer)) ||  // 2.a
         (first_non_fusible_diamond_producer == diamond_producer &&
          first_non_fusible_diamond_producer->user_count() == 2)) &&  // 2.b
        diamond_reduce_dimension_size == current_reduce_dimension_size) {  // 3
      continue;
    }

    // The "last trivially fusible user" chain of diamond chain n should never
    // intersect with the "first non fusible diamond producer" chain of diamond
    // chain n+1: if these chains intersected, then all the intermediate ops
    // between the diamond chains could be trivially fused, and both diamond
    // chains could be fused into a single diamond chain. Note that this only
    // holds insofar as we do not allow fusing in bitcasts that modify the last
    // dimension of the input array. It is however possible for the last
    // trivially fusible user of diamond chain n to be the first non fusible
    // diamond producer of diamond chain n+1.
    diamond_chains.push_back(DiamondChainDescriptor{
        last_trivially_fusible_user(previous_diamond_root),
        current_fusion_producer});

    current_fusion_producer = first_non_fusible_diamond_producer;
    current_reduce_dimension_size = diamond_reduce_dimension_size;
  }

  // The last diamond chain is still open; close it.
  diamond_chains.push_back(DiamondChainDescriptor{
      last_trivially_fusible_user(matched_diamonds.back().root),
      current_fusion_producer});

  // The diamond chains must be emitted in reverse order, to make sure that
  // producer instructions are emitted correctly when the root of
  // diamond chain n is exactly the producer of diamond chain n+1.
  for (auto diamond_chain = diamond_chains.rbegin();
       diamond_chain != diamond_chains.rend(); ++diamond_chain) {
    auto [root, producer] = *diamond_chain;
    TF_RET_CHECK(FuseSoftmax(root, producer).ok());
  }
  return true;
}
}  // namespace xla::gpu
