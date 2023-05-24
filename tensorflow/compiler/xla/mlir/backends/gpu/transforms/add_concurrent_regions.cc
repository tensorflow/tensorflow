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
#include <memory>
#include <utility>

#include "absl/strings/match.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
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

// Represents a slice of the buffer argument to the graph capture function.
struct BufferUse {
  BlockArgument arg;
  size_t offset;
  size_t len;
  bool is_constant;
};

BufferUse GetBufferUse(Value operand) {
  Operation* defining_op = operand.getDefiningOp();
  if (!defining_op) {
    auto block_argument = cast<mlir::BlockArgument>(operand);

    // Check if the input buffer is marked as constant.
    Region* parent_region = block_argument.getParentRegion();
    auto parent_func = parent_region->getParentOfType<FuncOp>();
    unsigned parent_func_arg_index = block_argument.getArgNumber();
    auto cst = parent_func.getArgAttrOfType<StringAttr>(parent_func_arg_index,
                                                        "lmhlo.constant_name");

    auto memref_type = cast<MemRefType>(block_argument.getType());
    size_t len = memref_type.getNumElements() *
                 (memref_type.getElementTypeBitWidth() / 8);
    return {block_argument, 0, len, cst ? true : false};
  }

  if (isa<memref::ViewOp>(defining_op)) {
    auto view_op = cast<mlir::memref::ViewOp>(defining_op);
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
    size_t len = memref_type.getNumElements() *
                 (memref_type.getElementTypeBitWidth() / 8);

    return {buffer_use.arg, buffer_use.offset + offset, len,
            buffer_use.is_constant};
  }

  if (auto cast = dyn_cast<mlir::memref::ReinterpretCastOp>(defining_op)) {
    return GetBufferUse(cast.getSource());
  }

  return {};
}

// Check if buffer_use has any overlap with buffers in the region.
bool HasDependency(llvm::ArrayRef<BufferUse> region_buffer_uses,
                   BufferUse buffer_use) {
  if (buffer_use.is_constant) return false;

  for (auto buffer_use_in_region : region_buffer_uses) {
    if (buffer_use_in_region.is_constant ||
        buffer_use_in_region.arg.getArgNumber() !=
            buffer_use.arg.getArgNumber()) {
      continue;
    }

    // Check if two buffer slices overlap.
    size_t start1 = buffer_use_in_region.offset;
    size_t end1 = buffer_use_in_region.offset + buffer_use_in_region.len;
    size_t start2 = buffer_use.offset;
    size_t end2 = buffer_use.offset + buffer_use.len;
    if (std::max(start1, start2) < std::min(end1, end2)) {
      return true;
    }
  }

  return false;
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
llvm::SmallVector<RegionStartAndEnd> GetRegionStartAndEnd(FuncOp capture_func) {
  llvm::SmallVector<RegionStartAndEnd> region_start_and_end;

  // These two arrays stores the information about the current region that is
  // being processed. region contains the kernels, while buffer_uses stores the
  // buffer usage by the kernels in the region.
  llvm::SmallVector<LaunchFuncOp> region;
  llvm::SmallVector<BufferUse> buffer_uses;

  auto store_region_and_start_new_region = [&]() {
    if (region.size() >= 2) {
      region_start_and_end.push_back(
          {region.front().getOperation(), region.back().getOperation()});
    }
    region.clear();
    buffer_uses.clear();
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
    llvm::SmallVector<BufferUse> operand_buffer_uses;
    for (auto operand : launch_func.getKernelOperands()) {
      BufferUse buffer_use = GetBufferUse(operand);
      operand_buffer_uses.push_back(buffer_use);
      if (HasDependency(buffer_uses, buffer_use)) {
        has_dependency = true;
      }
    }

    if (has_dependency) {
      store_region_and_start_new_region();
    }

    region.push_back(launch_func);
    for (auto buffer_use : operand_buffer_uses) {
      buffer_uses.push_back(buffer_use);
    }
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
