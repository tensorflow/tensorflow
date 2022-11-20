/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/softmax_fusion.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <queue>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla::gpu {

namespace {
namespace m = ::xla::match;

bool MatchesSoftmaxPattern(HloInstruction* instr) {
  // Match the following pattern:
  //
  // producer
  // |   \
  // |  reduce
  // |     |
  // |  broadcast
  // |   /
  // root
  //
  // There should not be other users of these ops than indicated by the edges.
  // Between the root and the producer, there can be some optional unary
  // elementwise ops. Also, initially we only support major-to-minor layouts.

  HloInstruction* root;
  HloInstruction* broadcast;
  HloInstruction* reduce_or_reshape;
  HloInstruction* reduce;
  if (!Match(instr,
             m::Op(&root)
                 .WithOperand(
                     1, m::Broadcast(&broadcast,
                                     m::Op(&reduce_or_reshape).WithOneUse())
                            .WithOneUse())
                 // The root operation should be an elementwise binary op of
                 // rank 2.
                 .WithPredicate([](const HloInstruction* instr) {
                   int64_t rank = instr->shape().rank();
                   return instr->IsElementwiseBinary() &&
                          // If the product of the first dimensions is 1, it
                          // currently crashes the pipeline. Also, we expect
                          // that the performance is not so good if the
                          // reduction dimension is big compared to the other
                          // dimensions.
                          Product(absl::Span<const int64_t>(
                                      instr->shape().dimensions())
                                      .first(rank - 1)) >
                              instr->shape().dimensions(rank - 1);
                 }))) {
    return false;
  }
  reduce = reduce_or_reshape;
  if (reduce_or_reshape->opcode() == HloOpcode::kReshape) {
    // Check that the reshape only removes 1-sized dimensions.
    auto descr =
        reduce_or_reshape->ReshapeMerelyInsertsOrDeletes1SizedDimensions();
    if (!descr.has_value() || !descr->inserted_dimensions.empty()) {
      return false;
    }
    reduce = reduce_or_reshape->mutable_operand(0);
    if (reduce->user_count() != 1) {
      return false;
    }
  }
  // The reduction should reduce the last dimension of the operand shape.
  if (reduce->opcode() != HloOpcode::kReduce ||
      reduce->dimensions().size() != 1 ||
      reduce->dimensions()[0] != reduce->shape().rank() ||
      // Currently we only support F32, because the lowering uses gpu.shuffle op
      // which has this restriction.
      reduce->shape().element_type() != F32) {
    return false;
  }

  // The broadcast dimensions should be sorted.
  if (!std::is_sorted(broadcast->dimensions().begin(),
                      broadcast->dimensions().end())) {
    return false;
  }
  // The broadcast should "undo" the reduction. Therefore, the non-broadcasted
  // dimensions should be the last dimension and 1-sized dimensions.
  int64_t rank = broadcast->shape().rank();
  if (rank < 1) {
    return false;
  }
  int64_t pos = 0;
  for (int64_t i = 0; i < rank; ++i) {
    if (pos < broadcast->dimensions().size() &&
        broadcast->dimensions()[pos] == i) {
      // The last dimension should not be broadcasted from the operand.
      if (i == rank - 1) {
        return false;
      }
      ++pos;
    } else if (i < rank - 1 && broadcast->shape().dimensions(i) != 1) {
      return false;
    }
  }

  HloInstruction* producer = reduce->mutable_operand(0);

  bool has_major_to_minor_layout =
      LayoutUtil::IsMonotonicWithDim0Major(root->shape().layout()) &&
      LayoutUtil::IsMonotonicWithDim0Major(
          reduce_or_reshape->shape().layout()) &&
      LayoutUtil::IsMonotonicWithDim0Major(reduce->shape().layout()) &&
      LayoutUtil::IsMonotonicWithDim0Major(broadcast->shape().layout()) &&
      LayoutUtil::IsMonotonicWithDim0Major(reduce->shape().layout()) &&
      LayoutUtil::IsMonotonicWithDim0Major(producer->shape().layout());

  // Check whether the operand of the reduce is a direct or indirect operand of
  // 'root'.
  const HloInstruction* maybe_common_operand = reduce->operand(0);
  const HloInstruction* current_operand = root->operand(0);
  while (current_operand != maybe_common_operand) {
    // Any intermediate operand between 'root' and 'maybe_common_operand' needs
    // to be an unary elementwise op with a single user.
    if (current_operand->operand_count() != 1 ||
        !current_operand->IsElementwise() ||
        current_operand->user_count() > 1) {
      return false;
    }
    if (!LayoutUtil::IsMonotonicWithDim0Major(
            current_operand->shape().layout())) {
      has_major_to_minor_layout = false;
    }
    current_operand = current_operand->operand(0);
  }

  if (!has_major_to_minor_layout) {
    LOG(INFO) << "Not matching softmax pattern due to non-standard layout";
    return false;
  }

  return true;
}

HloInstruction* SoftmaxProducer(HloInstruction* softmax_root) {
  // The softmax producer is found by going up the chain
  // -> broadcast -> (reshape) -> reduce -> producer
  auto reduce_or_reshape = softmax_root->mutable_operand(1)->mutable_operand(0);
  if (reduce_or_reshape->opcode() == HloOpcode::kReduce) {
    return reduce_or_reshape->mutable_operand(0);
  }
  return reduce_or_reshape->mutable_operand(0)->mutable_operand(0);
}

bool IsSupportedBroadcast(HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kBroadcast) {
    return false;
  }
  int64_t rank = hlo->shape().rank();
  if (rank <= 2) {
    return true;
  }
  // TODO(akuegel): Remove this logic once we do not rely on collapsing shapes
  // to 2D.
  // For rank > 2, we need to collapse the shape to 2D. This only works if the
  // dimensions that are to be collapsed have the same state regarding whether
  // they are broadcasted or not.
  if (!hlo->dimensions().empty()) {
    // Make sure that the broadcast dimensions are sorted.
    if (!std::is_sorted(hlo->dimensions().begin(), hlo->dimensions().end())) {
      return false;
    }
    // If there is a broadcast dimension in the part of dimensions that are
    // collapsed into 1 dimension, then all those rank - 1 dimensions need to be
    // broadcast dimensions.
    if (hlo->dimensions(0) < rank - 1 &&
        (hlo->dimensions().size() < rank - 1 ||
         hlo->dimensions()[rank - 1] != rank - 1)) {
      return false;
    }
  }
  return true;
}

Status ReplaceSoftmaxWithCustomCall(HloInstruction* root,
                                    HloInstruction* producer) {
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;
  auto builder = HloComputation::Builder("softmax_computation");
  std::vector<HloInstruction*> custom_call_operands;
  absl::flat_hash_set<HloInstruction*> visited;
  std::queue<HloInstruction*> worklist;
  worklist.push(producer);
  visited.insert(producer);
  int64_t operand_idx = 0;
  // Fuse all elementwise and broadcast ops into the softmax fusion computation,
  // provided each of them (except the softmax root) has exactly one user. We do
  // this by searching for unfusable ops which become the parameters of the
  // computation. Everything that was fused will be reconstructed in the new
  // computation by remapping the ops to their new operands.
  while (!worklist.empty()) {
    HloInstruction* current = worklist.front();
    worklist.pop();
    // TODO(akuegel): Currently our MLIR lowering doesn't work if we fuse
    // constants in. This results in an error like:
    // 'memref.get_global' op '__constant_150xf32' does not reference a valid
    // global memref
    if ((current->user_count() == 1 ||
         (current == producer && current->user_count() == 2)) &&
        ((current->IsElementwise() &&
          current->opcode() != HloOpcode::kConstant) ||
         IsSupportedBroadcast(current))) {
      for (HloInstruction* operand : current->operands()) {
        if (!visited.contains(operand)) {
          visited.insert(operand);
          worklist.push(operand);
        }
      }
    } else {
      // The op is unfusable. Create a parameter for the softmax computation.
      custom_call_operands.push_back(current);
      old_to_new_mapping[current] =
          builder.AddInstruction(HloInstruction::CreateParameter(
              operand_idx, current->shape(),
              absl::StrCat("parameter_", operand_idx)));
      ++operand_idx;
    }
  }
  std::function<void(const HloInstruction*)> create_computation =
      [&](const HloInstruction* instr) {
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
  auto softmax_computation =
      root->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                           /*is_entry=*/false);
  auto softmax_custom_call =
      root->parent()->AddInstruction(HloInstruction::CreateCustomCall(
          root->shape(), custom_call_operands, softmax_computation,
          kSoftmaxCallTarget));
  if (root->IsRoot()) {
    root->parent()->set_root_instruction(softmax_custom_call);
    TF_RETURN_IF_ERROR(
        root->parent()->RemoveInstructionAndUnusedOperands(root));
  } else {
    TF_RETURN_IF_ERROR(
        root->parent()->ReplaceInstruction(root, softmax_custom_call));
  }
  return OkStatus();
}

}  // anonymous namespace

StatusOr<bool> SoftmaxFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> softmax_roots;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      softmax_producer_to_root_mapping;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    if (comp->IsCustomCallComputation()) {
      continue;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (MatchesSoftmaxPattern(instr)) {
        softmax_roots.push_back(instr);
        softmax_producer_to_root_mapping[SoftmaxProducer(instr)] = instr;
      }
    }
  }
  if (softmax_roots.empty()) {
    return false;
  }

  absl::flat_hash_set<HloInstruction*> processed_softmax_roots;
  for (HloInstruction* root : softmax_roots) {
    if (processed_softmax_roots.contains(root)) {
      continue;
    }
    processed_softmax_roots.insert(root);
    // Try to merge softmax patterns. Since we have the softmax roots in
    // post-order, we need to extend from the root towards the producer of the
    // next softmax pattern.
    HloInstruction* merged_root = root;
    while (merged_root->user_count() > 0) {
      HloInstruction* current = merged_root;
      bool valid = true;
      while (!softmax_producer_to_root_mapping.contains(current)) {
        if (current->user_count() != 1 || !current->IsElementwise()) {
          valid = false;
          break;
        }
        current = current->users()[0];
        // Only allow unary ops on the path from 'merged_root' to another
        // producer of a softmax pattern. Note that 'merged_root' itself does
        // not have to be an unary op, even if it is the producer of another
        // softmax pattern.
        if (current->operand_count() != 1) {
          valid = false;
          break;
        }
        // Again, we only allow the default layout for any unary ops on the
        // path.
        if (!LayoutUtil::IsMonotonicWithDim0Major(current->shape().layout())) {
          valid = false;
          break;
        }
      }
      // Now 'current' should point to the producer of a softmax pattern. We can
      // merge if this producer is an elementwise op with exactly two users (the
      // users from the softmax pattern).
      if (!valid || !current->IsElementwise() || current->user_count() != 2 ||
          !softmax_producer_to_root_mapping.contains(current)) {
        break;
      }
      // We have found the producer of another softmax pattern. Go to the root
      // of that pattern.
      merged_root = softmax_producer_to_root_mapping[current];
      processed_softmax_roots.insert(merged_root);
    }
    HloInstruction* merged_producer = SoftmaxProducer(root);
    TF_RETURN_IF_ERROR(
        ReplaceSoftmaxWithCustomCall(merged_root, merged_producer));
  }
  return true;
}

}  // namespace xla::gpu
