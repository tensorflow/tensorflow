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

#include "tensorflow/compiler/plugin/poplar/driver/meta_graph.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"

namespace xla {

class HloModule;

namespace poplarplugin {

using NodeId = int64;
using NodeOperands = std::vector<NodeId>;
using NodeCondition = std::function<bool(const HloInstruction*)>;

enum class HloMatcherOpcode {
  kAnyOpcode,
};

// A class which allows us to extend the HloOpcode enum for special cases for
// the HloMatcher.
class HloMatcherOpcodeTarget {
 public:
  HloMatcherOpcodeTarget(const HloOpcode& opcode);
  HloMatcherOpcodeTarget(const HloMatcherOpcode& opcode);

  const bool IsHloOpcode() const;
  const HloOpcode GetHloOpcode() const;

  const bool IsHloMatcherOpcode() const;
  const HloMatcherOpcode GetHloMatcherOpcode() const;

 private:
  absl::variant<HloOpcode, HloMatcherOpcode> opcode_;
};

class HloMatcherNode {
 public:
  HloMatcherNode(HloMatcherOpcodeTarget opcode_target, NodeOperands operands);

  HloMatcherNode(HloMatcherOpcodeTarget opcode_target, NodeOperands operands,
                 NodeCondition node_condition);

  const HloMatcherOpcodeTarget& GetOpcodeTarget() const;
  const NodeOperands& GetOperands() const;
  const absl::optional<NodeCondition>& GetNodeCondition() const;

  // Checks whether the instruction matches this node.
  const bool Matches(const HloInstruction* inst) const;

 private:
  // The opcode target of the instruction to match
  HloMatcherOpcodeTarget opcode_target_;

  // A list of operands of this instruction. A positive number refers to one of
  // the other entries in the match pattern. A negative number indicates that
  // this operand will be a parameter to the fused subgraph.  If multiple match
  // nodes have the same negative number, then the same instruction must be
  // the operand to each match node. The parameter number is given by the value
  // in the matching position in the parameter_indices list.
  NodeOperands operands_;

  // If provided, this function will be called with the instruction. Only if
  // it returns true does the matching proceed.
  absl::optional<NodeCondition> node_condition_;
};

using PatternType = std::string;
using PatternMetaTarget = NodeId;
using PatternInputs = std::vector<NodeId>;
using PatternOutputs = std::vector<NodeId>;
using Pattern = std::vector<HloMatcherNode>;

class HloMatcherPattern {
 public:
  HloMatcherPattern() = delete;

  HloMatcherPattern(PatternType type, PatternMetaTarget meta_target,
                    PatternInputs inputs, PatternOutputs outputs,
                    Pattern pattern);

  const PatternType& GetType() const;

  const PatternMetaTarget& GetMetaTarget() const;

  const PatternInputs& GetInputs() const;

  const PatternOutputs& GetOutputs() const;

  const Pattern& GetPatternNodes() const;

  const MetaGraph<NodeId>& GetNodesToOperandsMetaGraph() const;

  const MetaGraph<NodeId>& GetOperandsToNodesMetaGraph() const;

 private:
  // The name to give the extracted fused graph.
  PatternType type;

  // The index of the op within the fusion which should have its op_metadata
  // copied to the kFusion instruction.
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
  Pattern pattern_nodes;

  // Structures used to represent this pattern - the first graph represents the
  // connections between nodes and their operands, the second graph represents
  // the connections between operands and their usage nodes.
  std::pair<MetaGraph<NodeId>, MetaGraph<NodeId>> pattern_graphs;

  // This function verifies that the pattern is correct. We define a pattern
  // correct if the following conditions are all met:
  // * It has at least one output.
  // * The graph is connected.
  std::pair<MetaGraph<NodeId>, MetaGraph<NodeId>> VerifyAndGetGraphs();
};

struct InstructionIndex {
  HloInstruction* inst;
  int64 op_idx;
};

using Trace = std::vector<InstructionIndex>;

struct HloMatcherMatched {
  HloComputation* computation;
  unsigned pattern_idx;
  absl::flat_hash_map<NodeId, HloInstruction*> instruction_mapping;
  std::vector<Trace> replacement_traces;

  HloMatcherMatched(HloComputation* computation, const unsigned pattern_idx)
      : computation(computation), pattern_idx(pattern_idx) {}
};

using ReplacedInstructions = std::vector<HloInstruction*>;

class HloMatcher : public HloModulePass {
 public:
  // By default never look through associative ops
  HloMatcher(const std::vector<HloMatcherPattern>& patterns,
             struct CompilerAnnotations& annotations, bool root_only,
             bool requires_unique_sharding = false,
             unsigned look_through_max_level = 0);

  ~HloMatcher() override = default;

  absl::string_view name() const override { return "matcher"; }

  StatusOr<bool> Run(HloModule* module) override;

 protected:
  // Outlines the given match and return the instruction which calls the
  // outlined computation.
  HloInstruction* OutlineExpressionFromComputation(
      const HloMatcherMatched& matched,
      const std::string& outlined_computation_name,
      const absl::optional<int64> sharding_device,
      std::vector<HloInstruction*> forced_parameters = {});

  // The list of patterns to try to find in the computations
  std::vector<HloMatcherPattern> patterns_;

  // The instruction annotations from the compiler
  struct CompilerAnnotations& annotations_;

 private:
  virtual bool HandleMatch(HloMatcherMatched& match,
                           const absl::optional<int64> sharding_device) = 0;

  bool MatchPatternStart(HloComputation*);
  bool MatchPattern(HloInstruction* inst, const unsigned pattern_idx);

  absl::optional<Trace> FindNextMatchingOp(HloInstruction* user,
                                           HloInstruction* inst,
                                           const HloOpcode desiredOpcode);
  bool MatchPatternSingleOutput(HloInstruction* root,
                                const HloMatcherPattern& pattern,
                                HloMatcherMatched& match);

  std::set<HloInstruction*> ReorderGraph(const HloMatcherMatched& matched);

  bool root_computation_only_;
  bool requires_unique_sharding_;
  unsigned look_through_max_depth_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
