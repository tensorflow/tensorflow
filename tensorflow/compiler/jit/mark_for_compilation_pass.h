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

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// The attribute that marks nodes to be grouped into functions by the
// encapsulate subgraphs pass.
extern const char* const kXlaClusterAttr;

// Pass that marks a subset of operators in the graph with attribute
// _XlaCluster so they are compiled by the EncapsulateSubgraphsPass.
class MarkForCompilationPass : public GraphOptimizationPass {
 public:
  MarkForCompilationPass() = default;

  Status Run(const GraphOptimizationPassOptions& options) override;

  // Run() just calls RunImpl() if --tf_xla_auto_jit is enabled. To run the pass
  // unconditionally, call RunImpl() directly.
  // is_compilable_fn, if set, is a predicate that must be true for a node to
  // be compiled.
  Status RunImpl(const GraphOptimizationPassOptions& options,
                 const std::function<bool(const Node*, const DeviceType&)>&
                     is_compilable_fn = {});
};

// Returns true iff 'ndef' is a call to a function that is compilable.  A
// function is compilable iff every operator in the function body is
// compilable.
bool IsCompilable(FunctionLibraryRuntime* flr, const NodeDef& ndef);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_MARK_FOR_COMPILATION_PASS_H_
