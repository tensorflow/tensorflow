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

#include "tensorflow/compiler/plugin/poplar/driver/outliner.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

static const char* names[] = {
  "pop_convolution",
};

static const std::vector<HloMatcherPattern> patterns = {

  // Stand-alone convolution
  {{HloOpcode::kConvolution, true, nullptr, {-1, -1}}},

};

Outliner::Outliner() : HloMatcher(patterns, true) {}

ReplacedInstructions Outliner::ReplaceNodes(unsigned int pattern,
                                            const HloMatcherMatched& match) {

  HloModule* module = match.computation->parent();

  std::vector<HloInstruction*> reversed(match.instructions.begin(),
                                       match.instructions.end());

  std::reverse(reversed.begin(), reversed.end());

  module->OutlineExpressionFromComputation(reversed,
                                           names[pattern],
                                           module->entry_computation());

  return match.instructions;
}

}
}
