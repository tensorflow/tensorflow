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
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)
#include "tensorflow/compiler/xla/service/cpu/onednn_ops_rewriter.h"

#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/cpu/onednn_memory_util.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace cpu {

namespace {
namespace m = match;

std::optional<HloInstruction*> MatchSoftmax(HloInstruction* instr) {
  //
  // producer
  // |   \
  // |  reduce_max
  // |     |
  // |  reshape
  // |     |
  // |  broadcast
  // |     |
  // |  reshape
  // |     |
  // |  broadcast
  // |   /
  // subtract
  // |
  // exponential
  // |   \
  // |  reduce_sum
  // |     |
  // |  reshape
  // |     |
  // |  broadcast
  // |     |
  // |  reshape
  // |     |
  // |  broadcast
  // |   /
  // divide  // (instr parameter)
  //
  // where both reductions occur only on the last axis.
  HloInstruction* left_exponential;
  HloInstruction* right_exponential;
  HloInstruction* left_producer;
  HloInstruction* right_producer;

  // Lower diamond
  if (!Match(
          instr,
          m::Divide(m::Exp(&left_exponential, m::Op()),
                    m::Broadcast(m::Reshape(m::Broadcast(m::Reshape(
                        m::Reduce(m::Exp(&right_exponential, m::Op()), m::Op())
                            .WithPredicate([](const HloInstruction* reduce) {
                              HloComputation* reducer = reduce->to_apply();
                              return (reducer->root_instruction()->opcode() ==
                                          HloOpcode::kAdd &&
                                      reduce->dimensions().size() == 1 &&
                                      reduce->dimensions()[0] !=
                                          reduce->shape().rank() - 1);
                            })
                            .WithOneUse()))))))) {
    return std::nullopt;
  }

  if (left_exponential != right_exponential ||
      left_exponential->user_count() != 2)
    return std::nullopt;

  // Upper diamond
  if (!Match(left_exponential->mutable_operand(0),
             m::Subtract(
                 m::Op(&left_producer),
                 m::Broadcast(
                     m::Reshape(m::Broadcast(m::Reshape(
                         m::Reduce(m::Op(&right_producer), m::Op())
                             .WithPredicate([](const HloInstruction* reduce) {
                               HloComputation* reducer = reduce->to_apply();
                               return (reducer->root_instruction()->opcode() ==
                                           HloOpcode::kMaximum &&
                                       reduce->dimensions().size() == 1 &&
                                       reduce->dimensions()[0] !=
                                           reduce->shape().rank() - 1);
                             })
                             .WithOneUse()))))
                     .WithOneUse())
                 .WithOneUse())) {
    return std::nullopt;
  }

  if (left_producer != right_producer || left_producer->user_count() != 2)
    return std::nullopt;

  return left_producer;
}

}  // namespace

class OneDnnOpsRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleDivide(HloInstruction* divide_instr) override {
    if (divide_instr->HasControlDependencies()) return OkStatus();
    if (auto dtype = divide_instr->shape().element_type();
        !(dtype == F32 || dtype == BF16))
      return OkStatus();
    std::optional<HloInstruction*> producer;
    bool found_pattern = false;
    if (producer = MatchSoftmax(divide_instr)) {
      found_pattern = true;
    }

    if (!found_pattern) return OkStatus();

    const Shape& output_shape = divide_instr->shape();
    HloInstruction* softmax_call =
        divide_instr->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape, {producer.value()}, "__onednn$softmax"));
    TF_RETURN_IF_ERROR(ReplaceInstruction(divide_instr, softmax_call));

    return OkStatus();
  }
};

StatusOr<bool> OneDnnOpsRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  OneDnnOpsRewriterVisitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
