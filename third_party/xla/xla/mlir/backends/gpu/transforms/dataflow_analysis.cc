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

#include "xla/mlir/backends/gpu/transforms/dataflow_analysis.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"

namespace xla {
namespace gpu {

namespace {

using namespace mlir;  // NOLINT
using mlir::BlockArgument;
using mlir::Operation;
using mlir::func::FuncOp;

// Represents a slice of the buffer argument to the graph capture function.
struct BufferUse {
  BlockArgument arg;
  size_t offset;
  size_t byte_len;

  // The buffer is only read by the operation.
  bool read_only;
};

BufferUse GetBufferUse(Value operand, bool read_only = false) {
  Operation* defining_op = operand.getDefiningOp();
  if (!defining_op) {
    auto block_argument = cast<mlir::BlockArgument>(operand);
    auto memref_type = cast<MemRefType>(block_argument.getType());
    size_t byte_len =
        (memref_type.getNumElements() * memref_type.getElementTypeBitWidth() +
         7) /
        8;
    return {block_argument, 0, byte_len, read_only};
  }

  if (isa<memref::ViewOp>(defining_op)) {
    auto view_op = cast<memref::ViewOp>(defining_op);
    auto buffer_use = GetBufferUse(view_op.getSource());

    IntegerAttr offset_attr;
    bool is_constant =
        matchPattern(view_op.getByteShift(), m_Constant(&offset_attr));
    if (!is_constant) {
      // Failed to refine the BufferUse.
      return buffer_use;
    }
    size_t offset = offset_attr.getInt();

    // Get len.
    auto memref_type = cast<MemRefType>(view_op.getType());
    // TODO(b/274157088): Handle the case where elements are complex numbers.
    if (!memref_type.getElementType().isIntOrFloat()) {
      return buffer_use;
    }

    size_t byte_len =
        (memref_type.getNumElements() * memref_type.getElementTypeBitWidth() +
         7) /
        8;

    return {buffer_use.arg, buffer_use.offset + offset, byte_len, read_only};
  }

  if (auto cast = dyn_cast<mlir::memref::ReinterpretCastOp>(defining_op)) {
    return GetBufferUse(cast.getSource(), read_only);
  }

  return {};
}

llvm::SmallVector<BufferUse> GetBufferUses(Operation& operation) {
  llvm::SmallVector<BufferUse> operand_buffer_uses;
  if (auto launch_func = dyn_cast<mlir::gpu::LaunchFuncOp>(operation)) {
    auto kernel_func =
        SymbolTable::lookupNearestSymbolFrom<mlir::gpu::GPUFuncOp>(
            &operation, launch_func.getKernel());
    auto kernel_operands = launch_func.getKernelOperands();
    for (auto it : llvm::enumerate(kernel_operands)) {
      BufferUse buffer_use = GetBufferUse(
          it.value(),
          /*read_only=*/!kernel_func.getArgAttrOfType<mlir::UnitAttr>(
              it.index(), "lmhlo.written"));
      operand_buffer_uses.push_back(buffer_use);
    }
  } else if (auto gemm = dyn_cast<lmhlo_gpu::GEMMOp>(operation)) {
    BufferUse buffer_use_0 = GetBufferUse(gemm.getA(), /*read_only=*/true);
    BufferUse buffer_use_1 = GetBufferUse(gemm.getB(), /*read_only=*/true);
    BufferUse buffer_use_2 = GetBufferUse(gemm.getC(), /*read_only=*/false);
    operand_buffer_uses.push_back(buffer_use_0);
    operand_buffer_uses.push_back(buffer_use_1);
    operand_buffer_uses.push_back(buffer_use_2);
  } else if (auto memcpy = dyn_cast<mlir::gpu::MemcpyOp>(operation)) {
    BufferUse src_buffer = GetBufferUse(memcpy.getSrc(), /*read_only=*/true);
    BufferUse dst_buffer = GetBufferUse(memcpy.getDst(), /*read_only=*/false);
    operand_buffer_uses.push_back(src_buffer);
    operand_buffer_uses.push_back(dst_buffer);
  }

  return operand_buffer_uses;
}

// Arguments to the graph capture function may have the "lmhlo.constant_name"
// attribute, which indicates that the passed-in buffer is constant.
bool IsConstant(BlockArgument block_argument) {
  // Check if the input buffer is marked as constant.
  Region* parent_region = block_argument.getParentRegion();
  auto parent_func = parent_region->getParentOfType<FuncOp>();
  unsigned parent_func_arg_index = block_argument.getArgNumber();
  auto cst = parent_func.getArgAttrOfType<StringAttr>(parent_func_arg_index,
                                                      "lmhlo.constant_name");
  return cst != nullptr;
}

// Check if two buffer_uses overlap.
bool HasDependency(BufferUse buffer_use_a, BufferUse buffer_use_b) {
  if (buffer_use_a.arg.getArgNumber() != buffer_use_b.arg.getArgNumber())
    return false;
  if (IsConstant(buffer_use_a.arg) || IsConstant(buffer_use_b.arg))
    return false;
  if (buffer_use_a.read_only && buffer_use_b.read_only) return false;

  // Check if two buffer slices overlap.
  size_t start1 = buffer_use_a.offset;
  size_t end1 = buffer_use_a.offset + buffer_use_a.byte_len;
  size_t start2 = buffer_use_b.offset;
  size_t end2 = buffer_use_b.offset + buffer_use_b.byte_len;
  if (std::max(start1, start2) < std::min(end1, end2)) {
    return true;
  }
  return false;
}

bool HasDependency(llvm::ArrayRef<BufferUse> buffer_uses_a,
                   llvm::ArrayRef<BufferUse> buffer_uses_b) {
  for (auto buffer_use_a : buffer_uses_a) {
    for (auto buffer_use_b : buffer_uses_b) {
      if (HasDependency(buffer_use_a, buffer_use_b)) return true;
    }
  }
  return false;
}

// Remove edges that are redundant for determining the execution order of
// kernels. We use the following algorithm to compute the transitive reduction:
//
// For source node in graph:
//   For each edge (source -> target)
//     longest_distance = the length of the longest path from source to target
//     if (longest_distance > 1):
//       remove (source -> target)
//
void TransitiveReduction(DataflowAnalysis::DataflowGraph& graph) {
  std::vector<std::vector<size_t>> parents(graph.size(), std::vector<size_t>());
  for (const DataflowAnalysis::Node& node : graph) {
    for (size_t child_index : node.children) {
      parents[child_index].push_back(node.index);
    }
  }

  std::vector<int> longest_distance(graph.size());
  for (DataflowAnalysis::Node& source : graph) {
    if (source.children.empty()) {
      continue;
    }

    std::fill(longest_distance.begin(), longest_distance.end(), 0);
    size_t farthest_child = source.children.back();
    for (size_t target = source.index + 1; target <= farthest_child; target++) {
      for (size_t mid : parents[target]) {
        // If the mid node is before source in the topological order, no path
        // source -> mid -> target can exits and we can skip it.
        if (mid >= source.index) {
          // If source -> mid -> target is longer than the longest path so far
          // from source -> target, update the longest distance.
          int candidate_longest_distance = longest_distance[mid] + 1;
          if (candidate_longest_distance > longest_distance[target]) {
            longest_distance[target] = candidate_longest_distance;
          }
        }
      }
    }

    source.children.erase(
        std::remove_if(
            source.children.begin(), source.children.end(),
            [&](size_t target) { return longest_distance[target] > 1; }),
        source.children.end());
  }
}

}  // namespace

DataflowAnalysis::DataflowGraph DataflowAnalysis::GetDataflowGraph(
    FuncOp graph_capture_function) {
  std::vector<Node> graph;
  for (auto [index, op] : llvm::enumerate(graph_capture_function.getOps())) {
    graph.push_back(Node{&op, index, {}});
  }

  // A vector that stores the buffer used by each operation in the graph. The
  // i-th operation's buffer uses are stored as the vector buffer_uses[i];
  std::vector<llvm::SmallVector<BufferUse>> buffer_uses;
  for (Operation& operation : graph_capture_function.getOps()) {
    buffer_uses.push_back(GetBufferUses(operation));
  }

  for (int i = 0; i < graph.size(); ++i) {
    Node& node_i = graph[i];
    llvm::ArrayRef<BufferUse> buffer_uses_i = buffer_uses[i];
    for (int j = i + 1; j < graph.size(); ++j) {
      llvm::ArrayRef<BufferUse> buffer_uses_j = buffer_uses[j];
      if (HasDependency(buffer_uses_i, buffer_uses_j)) {
        node_i.children.push_back(j);
      }
    }
  }

  TransitiveReduction(graph);
  return graph;
}

std::string DataflowAnalysis::ToDot(const DataflowGraph& graph) {
  std::string pad;
  std::string res;
  auto indent = [&] { pad.append(2, ' '); };
  auto outdent = [&] { pad.resize(pad.size() - 2); };
  auto addline = [&](auto&&... args) {
    absl::StrAppend(&res, pad, args..., "\n");
  };
  auto get_name = [](const Node& node) -> std::string {
    return absl::StrCat("\"", node.operation->getName().getStringRef().str(),
                        "_", node.index, "\"");
  };

  addline("digraph {");
  indent();
  for (const Node& node : graph) {
    for (size_t child_index : node.children) {
      Node child = graph[child_index];
      addline(get_name(node), " -> ", get_name(child));
    }
  }
  outdent();
  addline("}");
  return res;
}

}  // namespace gpu
}  // namespace xla
