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

// An optimization passes that marks nodes that are to be compiled with
// attribute kXlaClusterAttr. Nodes with the same cluster ID will be compiled
// together.

#ifndef TENSORFLOW_COMPILER_JIT_MARK_FOR_COMPILATION_PASS_H_
#define TENSORFLOW_COMPILER_JIT_MARK_FOR_COMPILATION_PASS_H_

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// The attribute that marks nodes to be grouped into functions by the
// encapsulate subgraphs pass.
extern const char* const kXlaClusterAttr;

// Marks a subset of nodes in the graph which are to be clustered
// with an attribute _XlaCluster=<cluster id> so they are picked up by the
// EncapsulateSubgraphsPass.
class MarkForCompilationPass : public GraphOptimizationPass {
 public:
  MarkForCompilationPass() = default;

  absl::Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  absl::Status RunForTest(const GraphOptimizationPassOptions& options,
                          bool disable_deadness_analysis,
                          bool deterministic_cluster_names);

  friend class MarkForCompilationPassTestHelper;
};

absl::flat_hash_map<string, std::vector<string>>* GetAllowlistTable();

namespace testing {
// DO NOT USE IN PRODUCTION.
//
// Resets some internal state to let us write reliable unit tests.
void ResetClusterSequenceNumber();

// Return a list of operation that we choose not to put into the allowlist.
absl::flat_hash_set<string> GetKnownXLAAllowlistOp();
}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_MARK_FOR_COMPILATION_PASS_H_
