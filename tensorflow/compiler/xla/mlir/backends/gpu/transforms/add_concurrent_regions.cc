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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/utils/custom_calls.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"

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

  // The buffer is only read by the operation.
  bool read_only;
};

BufferUse GetBufferUse(Value operand, bool read_only = false) {
  Operation* defining_op = operand.getDefiningOp();
  if (!defining_op) {
    auto block_argument = cast<mlir::BlockArgument>(operand);
    auto memref_type = cast<MemRefType>(block_argument.getType());
    size_t len =
        (memref_type.getNumElements() * memref_type.getElementTypeBitWidth() +
         7) /
        8;
    return {block_argument, 0, len, read_only};
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
    // TODO(b/274157088): Handle the case where elements are complex numbers.
    if (!memref_type.getElementType().isIntOrFloat()) {
      return buffer_use;
    }

    size_t len =
        (memref_type.getNumElements() * memref_type.getElementTypeBitWidth() +
         7) /
        8;

    return {buffer_use.arg, buffer_use.offset + offset, len, read_only};
  }

  if (auto cast = dyn_cast<mlir::memref::ReinterpretCastOp>(defining_op)) {
    return GetBufferUse(cast.getSource(), read_only);
  }

  return {};
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

// Check if buffer_use has any overlap with buffers in the region.
bool HasDependency(llvm::ArrayRef<BufferUse> region_buffer_uses,
                   BufferUse buffer_use) {
  if (IsConstant(buffer_use.arg)) return false;

  for (auto buffer_use_in_region : region_buffer_uses) {
    if (IsConstant(buffer_use_in_region.arg) ||
        buffer_use_in_region.arg.getArgNumber() !=
            buffer_use.arg.getArgNumber()) {
      continue;
    }

    // Two read-only accesses to the same buffer does not create dependency.
    if (buffer_use.read_only && buffer_use_in_region.read_only) continue;

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

struct RegionInfo {
  Operation* start;
  Operation* end;
  int size;
};

int GetKernelCount(llvm::ArrayRef<Operation*> region) {
  int kernel_count = 0;
  for (Operation* op : region) {
    if (!isa<memref::ViewOp, memref::ReinterpretCastOp, arith::ConstantOp>(
            op)) {
      kernel_count++;
    }
  }
  return kernel_count;
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
llvm::SmallVector<RegionInfo> GetRegionInfos(FuncOp capture_func) {
  llvm::SmallVector<RegionInfo> region_infos;

  // These two arrays stores the information about the current region that is
  // being processed. region contains the kernels, while buffer_uses stores the
  // buffer usage by the kernels in the region.
  llvm::SmallVector<Operation*> region;
  llvm::SmallVector<BufferUse> buffer_uses;

  auto store_region_and_start_new_region = [&]() {
    int kernel_count = GetKernelCount(region);
    if (kernel_count >= 2) {
      RegionInfo region_info = {region.front(), region.back(), kernel_count};
      region_infos.push_back(region_info);
    }
    region.clear();
    buffer_uses.clear();
  };

  auto operations = capture_func.getOps();
  for (auto& operation : operations) {
    // TODO(anlunx): Support other ops.
    llvm::SmallVector<BufferUse> operand_buffer_uses;
    if (auto launch_func = dyn_cast<LaunchFuncOp>(operation)) {
      auto kernel_func =
          SymbolTable::lookupNearestSymbolFrom<mlir::gpu::GPUFuncOp>(
              &operation, launch_func.getKernel());
      auto kernel_operands = launch_func.getKernelOperands();
      for (auto it : llvm::enumerate(kernel_operands)) {
        BufferUse buffer_use =
            GetBufferUse(it.value(),
                         /*read_only=*/!kernel_func.getArgAttrOfType<UnitAttr>(
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
    } else if ((!isa<memref::ViewOp>(operation) &&
                !isa<memref::ReinterpretCastOp>(operation) &&
                !isa<arith::ConstantOp>(operation)) ||
               region.empty()) {
      // Operation is unsupported by multi-streaming.
      store_region_and_start_new_region();
      continue;
    }

    bool has_dependency = false;
    for (BufferUse buffer_use : operand_buffer_uses) {
      if (HasDependency(buffer_uses, buffer_use)) {
        has_dependency = true;
      }
    }

    if (has_dependency) {
      store_region_and_start_new_region();
    }

    region.push_back(&operation);
    for (auto buffer_use : operand_buffer_uses) {
      buffer_uses.push_back(buffer_use);
    }
  }

  if (region.size() >= 2) {
    store_region_and_start_new_region();
  }

  return region_infos;
}

void InsertConcurrentRegions(FuncOp capture_func,
                             CustomCallDeclarations& custom_calls) {
  llvm::SmallVector<RegionInfo> region_infos = GetRegionInfos(capture_func);
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
    // Find the cuda graph capture function.
    if (absl::StrContains(func_op.getSymNameAttr().str(),
                          "xla.gpu.cuda.graph.capture")) {
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
