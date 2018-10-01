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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_WHILE_OP_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_WHILE_OP_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Rewrite While ops to use lower level control flow primitives instead.
class LowerWhileOpPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  // Rewrite the given While node `n` in graph `g` to use the lower level
  // primitives Enter, Exit, Switch, Merge and NextIteration.
  Status RewriteNode(Node* n, Graph* g);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_WHILE_OP_H_
