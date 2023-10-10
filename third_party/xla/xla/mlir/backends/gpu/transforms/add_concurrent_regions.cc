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
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/backends/gpu/transforms/dataflow_analysis.h"
#include "xla/mlir/runtime/utils/custom_calls.h"
#include "tsl/platform/env.h"

namespace xla {
namespace gpu {

namespace {

#define GEN_PASS_DEF_ADDCONCURRENTREGIONSPASS
#include "xla/mlir/backends/gpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT
using mlir::func::FuncOp;
using xla::runtime::CustomCallDeclarations;

class AddConcurrentRegionsPass
    : public impl::AddConcurrentRegionsPassBase<AddConcurrentRegionsPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//

struct RegionInfo {
  Operation* start;
  Operation* end;
  int size;
};

bool IsNoOp(Operation* op) {
  return isa<memref::ViewOp, memref::ReinterpretCastOp, arith::ConstantOp>(op);
}

int GetKernelCount(llvm::ArrayRef<DataflowAnalysis::Node> region) {
  int kernel_count = 0;
  for (const DataflowAnalysis::Node& node : region) {
    Operation* op = node.operation;
    if (!IsNoOp(op)) {
      kernel_count++;
    }
  }
  return kernel_count;
}

// We use the size of the inputs to the kernel as a heuristic to avoid
// adding memory bound kernels to the concurrent region.
// The memory bandwidth on A100 is 2MB/us, so a data movement less than 10MB
// is hidden by the kernel launch overhead, which is 5us.
static constexpr int64_t kInputSizeThreshold = 10'000'000;

bool IsKernelMemoryBound(Operation* op) {
  if (auto launch_func = dyn_cast<mlir::gpu::LaunchFuncOp>(op)) {
    size_t size = 0;

    for (Value operand : launch_func.getOperands()) {
      if (auto memref_type = dyn_cast<MemRefType>(operand.getType())) {
        size += (memref_type.getNumElements() *
                     memref_type.getElementTypeBitWidth() +
                 7) /
                8;
      }
    }

    if (size > kInputSizeThreshold) {
      return true;
    }
  }

  return false;
}

//
// Return a list of pairs of operations, in which the first element is the
// first operation in the region, and the second is the last operation in the
// region.
//
// We currently use a greedy algorithm to determine region starting point:
//   regions = []
//   region = {first operation}
//   for operation in the capture function
//     if HasDependency(region, operation)
//       regions.add(region)
//       region = new region
//     else
//       region.add(operation)
//
llvm::SmallVector<RegionInfo> GetRegionInfos(
    FuncOp capture_func, DataflowAnalysis& dataflow_analysis) {
  llvm::SmallVector<RegionInfo> region_infos;
  DataflowAnalysis::DataflowGraph dataflow_graph =
      dataflow_analysis.GetDataflowGraph(capture_func);

  // If verbose logging is enabled print the dataflow graph as a DOT graph.
  if (VLOG_IS_ON(100)) {
    std::cout << "Dependency graph for graph capture function "
              << capture_func.getName().str() << ":\n"
              << dataflow_analysis.ToDot(dataflow_graph);
  }

  llvm::SmallVector<DataflowAnalysis::Node> region;

  auto store_region_and_start_new_region = [&]() {
    int kernel_count = GetKernelCount(region);
    if (kernel_count >= 2) {
      RegionInfo region_info = {region.front().operation,
                                region.back().operation, kernel_count};
      region_infos.push_back(region_info);
    }
    region.clear();
  };

  auto append_node_to_region = [&](const DataflowAnalysis::Node& node) {
    if (region.empty()) {
      if (!IsNoOp(node.operation)) {
        region.push_back(node);
      }
    } else {
      region.push_back(node);
    }
  };

  for (const DataflowAnalysis::Node& node : dataflow_graph) {
    if (isa<func::ReturnOp>(node.operation)) {
      break;
    }

    bool has_dependency = false;
    for (const DataflowAnalysis::Node& node_in_region : region) {
      std::vector<size_t> children = node_in_region.children;
      if (std::find(children.begin(), children.end(), node.index) !=
          children.end()) {
        has_dependency = true;
        break;
      }
    }

    if (IsKernelMemoryBound(node.operation)) {
      store_region_and_start_new_region();
    } else if (has_dependency) {
      store_region_and_start_new_region();
      append_node_to_region(node);
    } else {
      append_node_to_region(node);
    }
  }

  store_region_and_start_new_region();
  return region_infos;
}

void InsertConcurrentRegions(FuncOp capture_func,
                             CustomCallDeclarations& custom_calls,
                             DataflowAnalysis& dataflow_analysis) {
  llvm::SmallVector<RegionInfo> region_infos =
      GetRegionInfos(capture_func, dataflow_analysis);
  auto sym_table = custom_calls.sym_table();

  for (RegionInfo region_info : region_infos) {
    Operation* start = region_info.start;
    Operation* end = region_info.end;

    ImplicitLocOpBuilder b(start->getLoc(), sym_table.getOp());
    func::FuncOp begin_marker = custom_calls.GetOrCreate(
        b, "xla.gpu.concurrent_region.begin", TypeRange(), TypeRange());
    b.setInsertionPoint(start);
    auto call = b.create<func::CallOp>(begin_marker.getName(), TypeRange());
    call->setAttr(b.getStringAttr("size"),
                  IntegerAttr::get(b.getIntegerType(64), region_info.size));

    func::FuncOp end_marker = custom_calls.GetOrCreate(
        b, "xla.gpu.concurrent_region.end", TypeRange(), TypeRange());
    b.setInsertionPointAfter(end);
    b.create<func::CallOp>(end_marker.getName(), TypeRange());
  }
}

//===----------------------------------------------------------------------===//

void AddConcurrentRegionsPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable sym_table(module);
  CustomCallDeclarations custom_calls(std::move(sym_table));

  auto func_ops = llvm::to_vector(module.getOps<FuncOp>());

  for (auto func_op : func_ops) {
    // Find the gpu graph capture function.
    if (absl::StrContains(func_op.getSymNameAttr().str(),
                          "xla.gpu.graph.capture")) {
      InsertConcurrentRegions(func_op, custom_calls,
                              getAnalysis<DataflowAnalysis>());
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAddConcurrentRegionsPass() {
  return std::make_unique<AddConcurrentRegionsPass>();
}

}  // namespace gpu
}  // namespace xla
