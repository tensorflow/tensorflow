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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
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
  HloInstruction* reduce;
  HloInstruction* producer;
  if (!Match(instr,
             m::Op(&root)
                 .WithOperand(
                     1, m::Broadcast(
                            &broadcast,
                            m::Reduce(&reduce, m::Op(&producer), m::Constant())
                                .WithOneUse()
                                // The reduction should reduce the last
                                // dimension of the operand shape.
                                .WithPredicate([](const HloInstruction* instr) {
                                  return instr->dimensions().size() == 1 &&
                                         instr->dimensions()[0] ==
                                             instr->shape().rank();
                                }))
                            .WithOneUse()
                            // The broadcast should "undo" the reduction.
                            .WithPredicate([](const HloInstruction* instr) {
                              int64_t rank = instr->shape().rank();
                              if (rank < 1) {
                                return false;
                              }
                              std::vector<int64_t> expected_dims(rank - 1);
                              std::iota(expected_dims.begin(),
                                        expected_dims.end(), 0);
                              return instr->dimensions() == expected_dims;
                            }))
                 // The root operation should be an elementwise binary op of
                 // rank 2.
                 // TODO(frgossen): Relax the rank 2 constraint when the
                 // pipeline can handle it.
                 .WithPredicate([](const HloInstruction* instr) {
                   return instr->IsElementwiseBinary() &&
                          instr->shape().rank() == 2;
                 }))) {
    return false;
  }
  bool has_major_to_minor_layout =
      LayoutUtil::IsMonotonicWithDim0Major(root->shape().layout()) &&
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
  // -> broadcast -> reduce -> producer
  return softmax_root->mutable_operand(1)->mutable_operand(0)->mutable_operand(
      0);
}

Status ReplaceSoftmaxWithCustomCall(HloInstruction* root,
                                    HloInstruction* producer) {
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;
  auto builder = HloComputation::Builder("softmax_computation");
  old_to_new_mapping[producer] = builder.AddInstruction(
      HloInstruction::CreateParameter(0, producer->shape(), "parameter_0"));
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
          root->shape(), {producer}, softmax_computation, kSoftmaxCallTarget));
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
