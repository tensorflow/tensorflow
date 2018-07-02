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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXPRESSION_OUTLINER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXPRESSION_OUTLINER_H_

#include "tensorflow/compiler/plugin/poplar/driver/hlo_matcher.h"

#include <set>

namespace xla {

class HloModule;

namespace poplarplugin {

// Extract elementwise ops into a called sub-graph
// (must come after InplaceFinder)

class ExpressionOutliner : public HloMatcher {
 public:
  ExpressionOutliner(struct CompilerAnnotations& annotations);

  ~ExpressionOutliner() override = default;

  tensorflow::StringPiece name() const override { return "expression-outline"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  ReplacedInstructions ReplaceNodes(int, const HloMatcherMatched&) override;

  const std::set<const HloInstruction*>& inplace_instructions;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
