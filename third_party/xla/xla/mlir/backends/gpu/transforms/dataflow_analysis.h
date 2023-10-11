/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_MLIR_BACKENDS_GPU_TRANSFORMS_DATAFLOW_ANALYSIS_H_
#define XLA_MLIR_BACKENDS_GPU_TRANSFORMS_DATAFLOW_ANALYSIS_H_

#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project

namespace xla {
namespace gpu {

class DataflowAnalysis {
 public:
  explicit DataflowAnalysis(mlir::Operation* op) {}

  struct Node {
    mlir::Operation* operation;
    size_t index;
    std::vector<size_t> children;
  };

  using DataflowGraph = std::vector<Node>;

  // This function creates a dataflow graph that represent data dependencies in
  // the graph capture function. The analysis relies on some properties of the
  // IR in XLA:
  //   (1) Buffer arguments do not alias. It is guaranteed that two buffer
  //       arguments to the graph capture function do not overlap.
  //   (2) XLA operations do not have any side effects beyond writing to its
  //       buffer arguments. So it is safe to reorder operations if they do not
  //       have write-conflicts.
  //   (3) We have information about read-only and read-write buffer arguments.
  DataflowGraph GetDataflowGraph(mlir::func::FuncOp graph_capture_function);

  std::string ToDot(const DataflowGraph& graph);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_MLIR_BACKENDS_GPU_TRANSFORMS_DATAFLOW_ANALYSIS_H_
