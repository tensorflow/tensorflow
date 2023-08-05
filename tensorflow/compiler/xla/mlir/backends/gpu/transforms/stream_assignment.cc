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

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/dataflow_analysis.h"
#include "tensorflow/compiler/xla/mlir/runtime/utils/custom_calls.h"

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

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect>();
  }
};

static constexpr int kNumStreams = 10;

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

  // next: Assign all non parallelizable ops to stream 0.

  return stream_assignment;
}

std::optional<int> GetAssignedStream(Operation* op) {
  if (op->hasAttr("stream")) {
    return op->getAttrOfType<mlir::IntegerAttr>("stream").getInt();
  }
  return std::nullopt;
}

//
// Add synchronizations between assigned streams. The added custom call
// xla.streams.await() {from = A, to = [B, C, ...]} makes future work submitted
// to A wait for work that are already submitted to streams B, C, ...
//
// Pseudo code:
// For each node in the dependency graph
//   If the node has a stream A assigned
//     parents = A's parents
//     to_streams = the assigned streams of its parents
//     add xla.streams.await() {from = A, to = to_streams} before node
//
// TODO(anlunx): Handle the case where the cuda graph contains non
// parallelizable ops (cuBLAS, cuDNN).
//
void AddSynchronization(FuncOp await_op,
                        runtime::CustomCallDeclarations custom_calls,
                        const DataflowGraph& graph) {
  for (const Node& node : graph) {
    Operation* op = node.operation;
    std::optional<int> op_stream = GetAssignedStream(op);
    if (!op_stream.has_value()) {
      continue;
    }
    int from_stream = op_stream.value();

    std::array<bool, kNumStreams> dependent_streams;
    dependent_streams.fill(false);
    for (int i = 0; i < node.index; i++) {
      if (std::find(graph[i].children.begin(), graph[i].children.end(),
                    node.index) != graph[i].children.end()) {
        if (std::optional<int> to_stream =
                GetAssignedStream(graph[i].operation)) {
          if (to_stream.value() != from_stream) {
            dependent_streams[to_stream.value()] = true;
          }
        }
      }
    }

    ImplicitLocOpBuilder b(op->getLoc(), custom_calls.sym_table().getOp());
    llvm::SmallVector<Attribute> to_streams;
    for (int i = 0; i < kNumStreams; i++) {
      if (dependent_streams[i]) {
        to_streams.push_back(b.getI64IntegerAttr(i));
      }
    }

    if (to_streams.empty()) {
      continue;
    }

    b.setInsertionPoint(op);
    auto call = b.create<func::CallOp>(await_op.getName(), TypeRange());
    call->setAttr(b.getStringAttr("from"), b.getI64IntegerAttr(from_stream));
    call->setAttr(b.getStringAttr("to"), b.getArrayAttr(to_streams));
  }
}

//===----------------------------------------------------------------------===//

void StreamAssignmentPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable sym_table(module);
  runtime::CustomCallDeclarations custom_calls(std::move(sym_table));

  auto func_ops = llvm::to_vector(module.getOps<FuncOp>());
  ImplicitLocOpBuilder b(module->getLoc(), custom_calls.sym_table().getOp());
  func::FuncOp begin_marker = custom_calls.GetOrCreate(
      b, "xla.gpu.concurrent_region.begin", TypeRange(), TypeRange());
  func::FuncOp end_marker = custom_calls.GetOrCreate(
      b, "xla.gpu.concurrent_region.end", TypeRange(), TypeRange());
  func::FuncOp await_op = custom_calls.GetOrCreate(b, "xla.streams.await",
                                                   TypeRange(), TypeRange());

  for (auto func_op : func_ops) {
    if (!absl::StrContains(func_op.getSymNameAttr().str(),
                           "xla.gpu.graph.capture")) {
      continue;
    }

    DataflowAnalysis dataflow_analysis(func_op);
    DataflowGraph graph = dataflow_analysis.GetDataflowGraph(func_op);
    std::vector<size_t> stream_assignment = AssignStreams(graph, kNumStreams);

    size_t stream_count = 0;
    for (auto [index, stream] : llvm::enumerate(stream_assignment)) {
      stream_count = std::max(stream_count, stream + 1);
      Node node = graph[index];
      Operation* op = node.operation;
      ImplicitLocOpBuilder b(op->getLoc(), custom_calls.sym_table().getOp());
      if (stream != -1) {
        op->setAttr(b.getStringAttr("stream"), b.getI64IntegerAttr(stream));
      }
    }

    AddSynchronization(await_op, custom_calls, graph);

    ImplicitLocOpBuilder b(func_op->getLoc(), custom_calls.sym_table().getOp());
    auto first_op = &(*func_op.getOps().begin());
    b.setInsertionPoint(first_op);
    auto call = b.create<func::CallOp>(begin_marker.getName(), TypeRange());
    call->setAttr(b.getStringAttr("size"), b.getI64IntegerAttr(stream_count));

    auto op_it = func_op.getOps().begin();
    while (!isa<func::ReturnOp>(*op_it)) {
      op_it++;
    }
    Operation* return_op = &(*op_it);
    b.setInsertionPoint(return_op);
    b.create<func::CallOp>(end_marker.getName(), TypeRange());
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStreamAssignmentPass() {
  return std::make_unique<StreamAssignmentPass>();
}

}  // namespace gpu
}  // namespace xla
