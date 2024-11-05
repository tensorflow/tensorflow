/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Optimization pass that merges VarHandleOps and ReadVariableOps into their
// fused forms.
//
// The goal of this pass is to fix a latency problem sometimes observed in
// inference benchmarks. Often a inference step starts by reading the value of
// many weights. Reading a resource variable requires a VarHandleOp and a
// ReadVariableOp per variable. Running hundreds of trivial ops can add hundreds
// of microseconds of latency to the critical path of an inference step. The
// inter-op latency of the executor can be easily hundreds of nanoseconds, which
// rapidly adds up over many inexpensive ops.
//
// This pass merges VarHandleOps that have only the graph source node as a
// predecessor into a single VarHandlesOp that reads all at once.
// It then merges ReadVariablesOp that have no control inputs and originate from
// the same handle op into a single large ReadVariablesOp.

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_VARIABLE_MERGER_PASS_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_VARIABLE_MERGER_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

class VariableMergerPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_VARIABLE_MERGER_PASS_H_
