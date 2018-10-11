/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_IF_OP_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_IF_OP_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Rewrite If ops to use switch and merge nodes instead.
class LowerIfOpPass : public GraphOptimizationPass {
 public:
  static const char* const kLowerUsingSwitchMergeAttr;

  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  // Rewrite the given If node `n` in graph `g` to use the switch-merge
  // form. `flib` should contain the branch functions referenced by `n`.
  Status RewriteNode(Node* n, const FunctionLibraryDefinition& flib, Graph* g);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_IF_OP_H_
