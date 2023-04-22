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

#ifndef TENSORFLOW_COMPILER_JIT_CLONE_CONSTANTS_FOR_BETTER_CLUSTERING_H_
#define TENSORFLOW_COMPILER_JIT_CLONE_CONSTANTS_FOR_BETTER_CLUSTERING_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
// Clones small host constants in the graph to make it easier to form larger
// clusters.
//
// This helps us in two ways:
//
//  - It reduces dependencies between clusters.  Let's say a constant C is used
//    by nodes X and Y.  If X and Y are put in different clusters (for whatever
//    reason) Y's cluster now has to wait for all the operations in X's cluster
//    to finish before it starts running.
//
//  - It lets us create bigger clusters in multi-GPU benchmarks.  Consider the
//    following graph:
//
//    digraph {
//      Const -> GPU_1
//      Const -> GPU_0_Y
//      GPU_0_X -> GPU_0_Y
//    }
//
//    We'd cluster Const and GPU_1 together (and place it on GPU_1), and this
//    will block us from clustering GPU_0_X and GPU_0_Y together since that
//    would increase the amount of work on GPU 0 waiting on work on GPU 1.
//    However, cloning Const into two copies, one for GPU_0_Y and one for GPU_1
//    will let us create one cluster containing {Const/copy_0, GPU_1} and
//    another containing {Const/copy_1, GPU_0_X, GPU_0_Y}.
//
// We only clone small host constants now to avoid increasing memory consumption
// too much.  Moreover, in practice the constants we have to duplicate are
// things like the `perm` input to `Transpose` and the `size` input to `Slice`
// which tend to be small anyway.

class CloneConstantsForBetterClusteringPass : public GraphOptimizationPass {
 public:
  CloneConstantsForBetterClusteringPass() = default;

  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  Status CloneSmallHostConstantInputs(
      Graph* g, const absl::flat_hash_set<string>& name_set, Node* n);
  string GenerateUniqueName(const absl::flat_hash_set<string>& name_set,
                            absl::string_view prefix);
  se::port::StatusOr<Node*> CloneNode(
      Graph* g, const absl::flat_hash_set<string>& name_set, Node* n);

  int unique_name_counter_ = 0;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_CLONE_CONSTANTS_FOR_BETTER_CLUSTERING_H_
