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
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/utils/custom_calls.h"

namespace xla {
namespace gpu {

namespace {

#define GEN_PASS_DEF_ADDCONCURRENTREGIONSPASS
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT
using mlir::func::FuncOp;
using mlir::gpu::LaunchFuncOp;
using xla::runtime::CustomCallDeclarations;

class AddConcurrentRegionsPass
    : public impl::AddConcurrentRegionsPassBase<AddConcurrentRegionsPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//

// Given a kernel operand, find the block argument used to derive that operand.
BlockArgument FindBlockArgument(Value operand) {
  Operation* defining_op = operand.getDefiningOp();
  if (!defining_op) {
    return cast<mlir::BlockArgument>(operand);
  }

  // In a cuda graph capture region we can use either memref.view or
  // memref.reinterpret_cast to create a kernel operand.
  if (isa<mlir::memref::ViewOp>(defining_op)) {
    auto view = cast<mlir::memref::ViewOp>(defining_op);
    auto source = view.getSource();
    return FindBlockArgument(source);
  } else if (isa<mlir::memref::ReinterpretCastOp>(defining_op)) {
    auto reinterp_cast = cast<mlir::memref::ReinterpretCastOp>(defining_op);
    auto source = reinterp_cast.getSource();
    return FindBlockArgument(source);
  }

  return nullptr;
}

using RegionStartAndEnd = std::pair<Operation*, Operation*>;

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
// We use very conservative way of determining dependency between operations:
// Two kernel launches have dependency if they use same argument to the graph
// capture function.
// TODO(anlunx): Take into account offsets and sizes.
llvm::SmallVector<RegionStartAndEnd> GetRegionStartAndEnd(FuncOp capture_func) {
  llvm::SmallVector<RegionStartAndEnd> region_start_and_end;

  llvm::SmallVector<LaunchFuncOp> region;
  // Store the arguments to the graph capture function that are used by
  // operations in the region. The number stored is the index to the argument.
  absl::flat_hash_set<unsigned> region_args;

  auto store_region_and_start_new_region = [&]() {
    if (region.size() >= 2) {
      region_start_and_end.push_back(
          {region.front().getOperation(), region.back().getOperation()});
    }
    region.clear();
    region_args.clear();
  };

  auto launch_func_ops = llvm::to_vector(capture_func.getOps<LaunchFuncOp>());
  auto operations = capture_func.getOps();
  for (auto& operation : operations) {
    // TODO(anlunx): Support other ops.
    if (!isa<LaunchFuncOp>(operation)) {
      store_region_and_start_new_region();
      continue;
    }

    LaunchFuncOp launch_func = cast<LaunchFuncOp>(operation);

    bool has_dependency = false;
    llvm::SmallVector<unsigned> operand_args;
    for (auto operand : launch_func.getKernelOperands()) {
      BlockArgument block_argument = FindBlockArgument(operand);
      unsigned arg_index = block_argument.getArgNumber();
      operand_args.push_back(arg_index);
      if (region_args.find(arg_index) != region_args.end()) {
        has_dependency = true;
      }
    }

    if (has_dependency) {
      store_region_and_start_new_region();
    }

    region.push_back(launch_func);
    region_args.insert(operand_args.begin(), operand_args.end());
  }

  if (region.size() >= 2) {
    store_region_and_start_new_region();
  }

  return region_start_and_end;
}

void InsertConcurrentRegions(FuncOp capture_func,
                             CustomCallDeclarations& custom_calls) {
  llvm::SmallVector<RegionStartAndEnd> region_start_and_end =
      GetRegionStartAndEnd(capture_func);
  auto sym_table = custom_calls.sym_table();

  for (auto pair : region_start_and_end) {
    Operation* start = pair.first;
    Operation* end = pair.second;

    ImplicitLocOpBuilder b(start->getLoc(), sym_table.getOp());
    // See how graph launch is added.
    func::FuncOp begin_marker = custom_calls.GetOrCreate(
        b, "xla.gpu.concurrent_region.begin", TypeRange(), TypeRange());
    b.setInsertionPoint(start);
    b.create<func::CallOp>(begin_marker.getName(), TypeRange());

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
    // Find the cuda graph capture function.
    if (absl::StrContains(func_op.getSymNameAttr().str(),
                          "xla.gpu.cuda.graph.capture")) {
      auto region_start_and_end = GetRegionStartAndEnd(func_op);
      InsertConcurrentRegions(func_op, custom_calls);
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAddConcurrentRegionsPass() {
  return std::make_unique<AddConcurrentRegionsPass>();
}

}  // namespace gpu
}  // namespace xla
