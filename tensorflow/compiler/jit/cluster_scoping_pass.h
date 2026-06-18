/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_CLUSTER_SCOPING_PASS_H_
#define TENSORFLOW_COMPILER_JIT_CLUSTER_SCOPING_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// This pass adds scopes to nodes in the _XlaInternalScope attribute to guide
// the later clustering passes.  A major reason to do this is to prevent the
// clustering from losing critical parallelism in the Tensorflow graph, which
// can incur great performance degradation.
//
// This pass must be run before MarkForCompilationPass, as it stores the
// scoping information that MarkForCompilationPass will need to respect for
// clustering decision.
class ClusterScopingPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_CLUSTER_SCOPING_PASS_H_
