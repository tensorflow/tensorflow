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
  HloOpcode opcode;
  bool include_in_replacement;
  std::function<bool(HloInstruction*)> verification_fn;
  std::vector<int> operands;
};

struct HloMatcherMatched {
  HloComputation* computation;
  bool ok;
  std::vector<HloInstruction*> instructions;
};

using HloMatcherPattern = std::vector<HloMatcherNode>;
using ReplacedInstructions = std::vector<HloInstruction*>;

class HloMatcher : public HloPassInterface {
public:
  HloMatcher(const std::vector<HloMatcherPattern>& patterns, bool root_only);

  ~HloMatcher() override = default;

  tensorflow::StringPiece name() const override { return "matcher"; }

  StatusOr<bool> Run(HloModule *module) override;

private:
  virtual ReplacedInstructions ReplaceNodes(unsigned int pattern,
                                            const HloMatcherMatched&) = 0;

  void MatchPatternStart(HloComputation*, HloInstruction* inst);
  bool MatchPattern(HloInstruction* inst, const HloMatcherPattern& pattern,
                    HloMatcherMatched& match);
  void AddMatch(unsigned pattern, const HloMatcherMatched& match);


  bool root_computation_only_;
  std::vector<HloMatcherPattern> patterns_;
  std::set<HloInstruction*> visited_;
  std::vector<std::list<HloMatcherMatched>> matches_;
  std::multimap<const HloInstruction*, HloMatcherMatched*> match_map_;
};

}
}

#endif
