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

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/dataflow_analysis.h"

namespace xla {
namespace gpu {

namespace {

#define GEN_PASS_DEF_STREAMASSIGNMENTPASS
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT
using mlir::func::FuncOp;
using DataflowGraph = DataflowAnalysis::DataflowGraph;
using Node = DataflowAnalysis::Node;

class StreamAssignmentPass
    : public impl::StreamAssignmentPassBase<StreamAssignmentPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//

bool IsParallelizableOp(Operation* op) {
  return isa<mlir::gpu::LaunchFuncOp, mlir::gpu::MemcpyOp>(op);
}

//
// A simple algorithm to assign streams using the dependency information
// provided by the dataflow graph.
// Pseudocode:
//   stream = 0
//   while there exists op such that it is unassigned:
//     assign op to stream
//     while op has a child:
//       op = the last child in the order of execution in the capture function
//       assign op to stream
//     stream++
//
// When assigning a stream to a dependency chain, we find the next op in the
// chain by finding the last child of the current op. For example, in the
// following dependency graph, A and C are assigned to stream 0, while B is
// assigned to 1.
//
// A-->B  C
// |      ^
// +------|
//
std::vector<size_t> AssignStreams(const DataflowGraph& graph, int num_streams) {
  std::vector<size_t> stream_assignment(graph.size(), -1);
  size_t current_stream = 0;

  auto get_current_stream = [&]() -> size_t {
    size_t assigned_stream = current_stream;
    current_stream++;
    if (current_stream == num_streams) {
      current_stream = 0;
    }
    return assigned_stream;
  };

  auto get_first_unassigned_node = [&stream_assignment =
                                        std::as_const(stream_assignment),
                                    &graph]() -> std::optional<size_t> {
    for (auto [index, stream] : llvm::enumerate(stream_assignment)) {
      if (stream == -1 && IsParallelizableOp(graph[index].operation)) {
        return index;
      }
    }
    return std::nullopt;
  };

  auto get_last_unassigned_child = [&stream_assignment =
                                        std::as_const(stream_assignment),
                                    &graph](Node node) -> std::optional<Node> {
    for (int i = node.children.size() - 1; i >= 0; i--) {
      Node child = graph[node.children[i]];
      if (!IsParallelizableOp(child.operation)) continue;
      if (stream_assignment[child.index] == -1) {
        return child;
      }
    }
    return std::nullopt;
  };

  std::function<void(Node, size_t)> assign_stream_to_dependency_chain =
      [&](Node node, size_t stream) {
        stream_assignment[node.index] = stream;

        if (auto child = get_last_unassigned_child(node)) {
          assign_stream_to_dependency_chain(child.value(), stream);
        }
      };

  while (std::optional<size_t> unassigned_index = get_first_unassigned_node()) {
    Node unassigned_node = graph[unassigned_index.value()];
    size_t assigned_stream = get_current_stream();
    assign_stream_to_dependency_chain(unassigned_node, assigned_stream);
  }

  return stream_assignment;
}

//===----------------------------------------------------------------------===//

void StreamAssignmentPass::runOnOperation() {
  FuncOp func_op = getOperation();

  if (!absl::StrContains(func_op.getSymNameAttr().str(),
                         "xla.gpu.cuda.graph.capture")) {
    return;
  }

  SymbolTable sym_table(func_op->getParentOfType<mlir::ModuleOp>());
  ImplicitLocOpBuilder b(func_op->getLoc(), sym_table.getOp());

  DataflowAnalysis dataflow_analysis(func_op);
  DataflowGraph graph = dataflow_analysis.GetDataflowGraph(func_op);
  std::vector<size_t> stream_assignment = AssignStreams(graph, 10);

  for (auto [index, stream] : llvm::enumerate(stream_assignment)) {
    Node node = graph[index];
    Operation* op = node.operation;
    if (stream != -1) {
      op->setAttr(b.getStringAttr("stream"), b.getI64IntegerAttr(stream));
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createStreamAssignmentPass() {
  return std::make_unique<StreamAssignmentPass>();
}

}  // namespace gpu
}  // namespace xla
