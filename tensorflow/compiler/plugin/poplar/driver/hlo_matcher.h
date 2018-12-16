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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_HLO_MATCHER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_HLO_MATCHER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/inplace_util.h"

namespace xla {

class HloModule;

namespace poplarplugin {

using NodeId = int64;
using NodeOperands = std::vector<NodeId>;
using NodeCondition = std::function<bool(HloInstruction*)>;

struct HloMatcherNode {
  // The opcode of the instruction to match
  HloOpcode opcode;

  // A list of operands of this instruction. A positive number refers to one of
  // the other entries in the match pattern. A negative number indicates that
  // this operand will be a parameter to the fused subgraph.  If multiple match
  // nodes have the same negative number, then the same instruction must be
  // the operand to each match node. The parameter number is given by the value
  // in the matching position in the parameter_indices list.
  NodeOperands operands;

  // If provided, this function will be called with the instruction. Only if
  // it returns true does the matching proceed.
  absl::optional<NodeCondition> node_condition;

  HloMatcherNode(HloOpcode opcode, NodeOperands operands)
      : opcode(opcode), operands(operands), node_condition(absl::nullopt){};

  HloMatcherNode(HloOpcode opcode, NodeOperands operands,
                 NodeCondition node_condition)
      : opcode(opcode), operands(operands), node_condition(node_condition){};
};

struct InstructionIndex {
  HloInstruction* inst;
  int64 op_idx;
};

using Trace = std::vector<InstructionIndex>;

struct HloMatcherMatched {
  HloComputation* computation;
  bool ok;
  std::vector<HloInstruction*> instructions;
  std::map<const HloInstruction*, std::vector<int64>> inst_parameters;
  std::vector<Trace> replacement_traces;
};

using PatternType = std::string;
using PatternMetaTarget = NodeId;
using PatternInputs = std::vector<NodeId>;
using PatternOutputs = std::vector<NodeId>;
using Pattern = std::vector<HloMatcherNode>;

struct HloMatcherPattern {
  // The name to give the extracted fused graph.
  PatternType type;

  // The index of the op within the fusion which should have its op_metadata
  // copied to the kCall instruction.
  PatternMetaTarget meta_target;

  // If op is an input then don't include this instruction in the fusion. The
  // fused subgraph will have a parameter where this instruction would be, and
  // the index of that parameter is given by the relative index in the inputs
  // vector.
  // Example:
  // inputs = {2, 1}
  // Then the instruction with label 2 will be a parameter instruction with
  // index 0 and the instruction with label 1 will be a parameter instruction
  // with index 1.
  PatternInputs inputs;

  // If an op is an output then replace all the uses of this node in the
  // computation with the output tensor from this fusion. If there is more than
  // one output then the fusion returns a tuple and the output tensor tuple
  // index is determined by the relative index in the outputs.
  // Example:
  // outputs = {2, 0}
  // will insert two GTE instructions into the graph, where GTE with tuple_index
  // == 0 will correspond to output tensor with label 2 and GTE with tuple_index
  // == 1 will correspond to output tensor with label 0.
  PatternOutputs outputs;

  // A vector of HloMatcherNode, describing the pattern to match.
  Pattern pattern;

  HloMatcherPattern(PatternType type, PatternMetaTarget meta_target,
                    PatternInputs inputs, PatternOutputs outputs,
                    Pattern pattern)
      : type(type),
        meta_target(meta_target),
        inputs(inputs),
        outputs(outputs),
        pattern(pattern) {
    Verify();
  }

  // This function verifies that the pattern is correct. We define a pattern
  // correct if the following conditions are all met:
  // * It has at least one output
  // * If we perform traversal from any output node, we can reach any input
  //   node.
  // * If we perform traversals from all the output nodes and combine the
  //   visited nodes, then every node in the pattern has to be visited at least
  //   once.
  void Verify();
};

using ReplacedInstructions = std::vector<HloInstruction*>;

struct OutlinedInfo {
  HloInstruction* call_to_outlined_computation;
  ReplacedInstructions removed_or_modified_instructions;
};

class HloMatcher : public HloModulePass {
 public:
  // By default never look through associative ops
  HloMatcher(const std::vector<HloMatcherPattern>& patterns,
             struct CompilerAnnotations& annotations, bool root_only,
             unsigned look_through_max_level = 0);

  ~HloMatcher() override = default;

  absl::string_view name() const override { return "matcher"; }

  StatusOr<bool> Run(HloModule* module) override;

 protected:
  OutlinedInfo OutlineExpressionFromComputation(
      const HloMatcherMatched& matched,
      const std::string& outlined_computation_name, const char metadata_index) {
    return OutlineExpressionFromComputation(matched, outlined_computation_name,
                                            metadata_index, {});
  }

  OutlinedInfo OutlineExpressionFromComputation(
      const HloMatcherMatched& matched,
      const std::string& outlined_computation_name, const char metadata_index,
      std::vector<HloInstruction*> forced_parameters);

  unsigned MarkReplacedInstructions(const OutlinedInfo& outlined_info);

  // A vector of lists of matches found. One vector entry per pattern, one list
  // entry per match in the computation
  std::vector<std::list<HloMatcherMatched>> matches_;

  // The list of patterns to try to find in the computations
  std::vector<HloMatcherPattern> patterns_;

  // The instruction annotations from the compiler
  struct CompilerAnnotations& annotations_;

 private:
  virtual unsigned ReplaceNodes() = 0;

  void MatchPatternStart(HloComputation*, HloInstruction* inst);
  bool MatchPattern(HloInstruction* inst, const HloMatcherPattern& pattern,
                    HloMatcherMatched& match);
  void AddMatch(unsigned pattern, const HloMatcherMatched& match);
  StatusOr<Trace> FindNextMatchingOp(HloInstruction* user, HloInstruction* inst,
                                     const HloOpcode desiredOpcode);
  std::set<HloInstruction*> ReorderGraph(const HloMatcherMatched& matched);

  bool root_computation_only_;
  unsigned look_through_max_depth_;

  // A map of instructions in the computation to matches. When replacing
  // instructions due to one match, other matches which contain the instruction
  // cannot also be applied
  std::multimap<const HloInstruction*, HloMatcherMatched*> match_map_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
