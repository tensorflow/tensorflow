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

namespace xla {

class HloModule;

namespace poplarplugin {

struct HloMatcherNode {
  // The opcode of the instruction to match
  HloOpcode opcode;

  // If true then don't include this instruction in the fusion. The fused
  // subgraph will have a parameter where this instruction would be, and the
  // index of that parameter is given by the one entry in the parameter_indices
  // member
  bool include_in_replacement;

  // If this instruction is a parameter to the fusion, this indicates the
  // parameter number which should be assigned in the fused subgraph
  int parameter_index;

  // If not null, this function will be called with the instruction. Only if
  // it returns true does the matching proceed.
  std::function<bool(HloInstruction*)> verification_fn;

  // A list of operands of this instruction. A positive number refers to one of
  // the other entries in the match pattern. A negative number indicates that
  // this operand will be a parameter to the fused subgraph.  If multiple match
  // nodes have the same negative number, then the same instruction must be
  // the operand to each match node. The parameter number is given by the value
  // in the matching position in the parameter_indices list.
  std::vector<unsigned int> operands;
};

struct HloMatcherMatched {
  HloComputation* computation;
  bool ok;
  std::vector<HloInstruction*> instructions;
  std::vector<std::pair<HloInstruction*, int64>> parameters;
};

struct FusedGraphInfo {
  // The names to give the extracted fused graphs
  const char* name;

  // The index of the op within each fusion which should have its op_metadata
  // copied to the kCall instruction.
  const char op_index;
};

using HloMatcherPattern = std::vector<HloMatcherNode>;
using ReplacedInstructions = std::vector<HloInstruction*>;

class HloMatcher : public HloPassInterface {
 public:
  HloMatcher(const std::vector<HloMatcherPattern>& patterns,
             struct CompilerAnnotations& annotations, bool root_only);

  ~HloMatcher() override = default;

  tensorflow::StringPiece name() const override { return "matcher"; }

  StatusOr<bool> Run(HloModule* module) override;

 protected:
  ReplacedInstructions OutlineExpressionFromComputation(
      const HloMatcherMatched& matched,
      const std::string& outlined_computation_name, const char metadata_index);

 private:
  virtual ReplacedInstructions ReplaceNodes(int pattern,
                                            const HloMatcherMatched&) = 0;

  void MatchPatternStart(HloComputation*, HloInstruction* inst);
  bool MatchPattern(HloInstruction* inst, const HloMatcherPattern& pattern,
                    HloMatcherMatched& match);
  void AddMatch(unsigned pattern, const HloMatcherMatched& match);

  bool root_computation_only_;

  // The list of patterns to try to find in the computations
  std::vector<HloMatcherPattern> patterns_;

  // A vector of lists of matches found. One vector entry per pattern, one list
  // entry per match in the computation
  std::vector<std::list<HloMatcherMatched>> matches_;

  // A map of instructions in the computation to matches. When replacing
  // instructions due to one match, other matches which contain the instruction
  // cannot also be applied
  std::multimap<const HloInstruction*, HloMatcherMatched*> match_map_;

  // The instruction annotations from the compiler
  struct CompilerAnnotations& annotations_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
