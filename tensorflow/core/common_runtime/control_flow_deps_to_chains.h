/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_CONTROL_FLOW_DEPS_TO_CHAINS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_CONTROL_FLOW_DEPS_TO_CHAINS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// Move control flow dependencies in functional control flow to chains.
// Chains are extra loop variables that serve as tokens for wiring control
// dependencies across loop iterations at a finer granularity, compared to just
// a single barrier at the end of each iteration. This enables the
// parallel_iterations feature for tf.while_loop.
//
// One separate chain is added for each of the body function's `control_ret`.
//
// For example:
//
//   while i > 0:
//     r = v.read_value()
//     s += expensive_operation(r)
//     assign = v.assign_add(1)  # control: r
//     i += 1
//
// The loop above can safely compute `r` and `assign` ahead of `s`, by the
// as-if rule. The separate switch/merge nodes that the loop lowers into support
// that.
// This transformation enables that to happen by rewriting the loop as follows:
//
//   chain = 0.0
//   while i > 0:
//     r = v.read_value()  # control: chain
//     s += expensive_operation(r)
//     assign = v.assign_add(1)  # control: r
//     i += 1
//     chain = identity(chain)  # control: assign
//
// This only rewires dependencies which need to cross scope boundaries, as the
// switch/merge lowering process has no other way of dealing correctly with
// those.
//
// This pass is best-effort and conservative, requiring attributes set by
// tf.while_loop and automatic_control_dependencies. When the required
// attributes are missing for a particular While node, no change is made to
// that node. Other While nodes are still processed if they do have the needed
// annotations.
// The pass can also be toggled by omitting the `_stateful_parallelism=True`
// attribute on the While node.
// When the pass returns with error, the graph is left in an invalid state.
// If successful, this pass also clears the body function's control_ret,
// which in effect removes the hard barrier that gates each loop iteration.
//
//
// TODO(mdan): Can we define that more formally?
class ControlFlowDepsToChainsPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_CONTROL_FLOW_DEPS_TO_CHAINS_H_
