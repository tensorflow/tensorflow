/* Copyright 2026 The OpenXLA Authors.

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
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"  // IWYU pragma: keep
#include "xla/util.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_VIRTUALIZESUBKERNELPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

constexpr llvm::StringRef kGlobalSmemName = "global_smem";
constexpr int64_t kSharedMemoryAlignment = 16;
constexpr unsigned kSharedMemoryAddrSpace = 3;

void EnsureExternalGlobalSmem(mlir::ModuleOp module, int64_t alignment) {
  if (auto existing =
          module.lookupSymbol<mlir::LLVM::GlobalOp>(kGlobalSmemName)) {
    int64_t current_align =
        existing.getAlignment().value_or(kSharedMemoryAlignment);
    if (alignment > current_align) {
      existing.setAlignment(alignment);
    }
    return;
  }
  mlir::ImplicitLocOpBuilder b(module.getLoc(), module.getContext());
  b.setInsertionPointToStart(module.getBody());
  auto array_ty = mlir::LLVM::LLVMArrayType::get(b.getI8Type(), 0);
  mlir::LLVM::GlobalOp::create(b, array_ty, /*isConstant=*/false,
                               mlir::LLVM::Linkage::External,
                               std::string(kGlobalSmemName), mlir::Attribute{},
                               /*alignment=*/alignment,
                               /*addrSpace=*/kSharedMemoryAddrSpace);
}

int64_t PackSubKernelSharedMemory(mlir::ModuleOp sub_module) {
  llvm::SmallVector<mlir::LLVM::GlobalOp> shmem_globals;
  for (auto global : sub_module.getOps<mlir::LLVM::GlobalOp>()) {
    if (global.getAddrSpace() == kSharedMemoryAddrSpace &&
        global.getLinkage() != mlir::LLVM::Linkage::External &&
        global.getSymName() != kGlobalSmemName) {
      shmem_globals.push_back(global);
    }
  }
  if (shmem_globals.empty()) {
    return 0;
  }

  mlir::DataLayout data_layout(sub_module);

  absl::flat_hash_map<std::string, int64_t> global_offsets;
  int64_t total_bytes = 0;
  int64_t max_alignment = kSharedMemoryAlignment;
  for (mlir::LLVM::GlobalOp global : shmem_globals) {
    int64_t alignment = std::max<int64_t>(
        kSharedMemoryAlignment,
        data_layout.getTypeABIAlignment(global.getGlobalType()));
    if (auto align_attr = global.getAlignment()) {
      alignment = std::max<int64_t>(alignment, *align_attr);
    }
    max_alignment = std::max(max_alignment, alignment);
    total_bytes = RoundUpTo<int64_t>(total_bytes, alignment);
    global_offsets[global.getSymName().str()] = total_bytes;
    int64_t size_bytes = static_cast<int64_t>(
        data_layout.getTypeSize(global.getGlobalType()).getFixedValue());
    total_bytes += size_bytes;
  }
  total_bytes = RoundUpTo<int64_t>(total_bytes, max_alignment);
  EnsureExternalGlobalSmem(sub_module, max_alignment);

  llvm::SmallVector<mlir::LLVM::AddressOfOp> addr_ops;
  sub_module.walk([&](mlir::LLVM::AddressOfOp addr_op) {
    if (global_offsets.contains(addr_op.getGlobalName().str())) {
      addr_ops.push_back(addr_op);
    }
  });

  for (mlir::LLVM::AddressOfOp addr_op : addr_ops) {
    int64_t byte_offset = global_offsets.at(addr_op.getGlobalName().str());
    mlir::ImplicitLocOpBuilder b(addr_op.getLoc(), addr_op);
    mlir::Type ptr_3_ty = addr_op.getRes().getType();
    auto base = mlir::LLVM::AddressOfOp::create(b, ptr_3_ty, kGlobalSmemName);
    mlir::Value replacement = base.getResult();
    if (byte_offset > 0) {
      auto gep =
          mlir::LLVM::GEPOp::create(b, ptr_3_ty, b.getI8Type(), base,
                                    llvm::SmallVector<mlir::LLVM::GEPArg>{
                                        static_cast<int32_t>(byte_offset)});
      replacement = gep.getResult();
    }
    addr_op.replaceAllUsesWith(replacement);
    addr_op.erase();
  }

  for (mlir::LLVM::GlobalOp global : shmem_globals) {
    global.erase();
  }

  return total_bytes;
}

void ReplaceAllAndErase(mlir::ImplicitLocOpBuilder& b,
                        llvm::ArrayRef<mlir::Operation*> ops, mlir::Value val) {
  for (mlir::Operation* op : ops) {
    if (b.getInsertionPoint() == op->getIterator()) {
      b.setInsertionPointAfter(op);
    }
    op->replaceAllUsesWith(mlir::ValueRange{val});
    op->erase();
  }
}

mlir::Value CreateI32Const(mlir::ImplicitLocOpBuilder& b, int64_t val) {
  return mlir::LLVM::ConstantOp::create(
      b, b.getI32Type(), b.getI32IntegerAttr(static_cast<int32_t>(val)));
}

template <typename IdXOp, typename IdYOp, typename IdZOp, typename DimXOp,
          typename DimYOp, typename DimZOp>
mlir::LogicalResult VirtualizeIds(mlir::LLVM::LLVMFuncOp func,
                                  mlir::ImplicitLocOpBuilder& b, int64_t dim_x,
                                  int64_t dim_y, int64_t dim_z,
                                  mlir::Value linear_id) {
  llvm::SmallVector<mlir::Operation*> x_ops;
  llvm::SmallVector<mlir::Operation*> y_ops;
  llvm::SmallVector<mlir::Operation*> z_ops;
  llvm::SmallVector<mlir::Operation*> dim_x_ops;
  llvm::SmallVector<mlir::Operation*> dim_y_ops;
  llvm::SmallVector<mlir::Operation*> dim_z_ops;
  auto module = func->getParentOfType<mlir::ModuleOp>();
  mlir::WalkResult walk_res = module.walk([&](mlir::Operation* op) {
    if (op == linear_id.getDefiningOp()) {
      return mlir::WalkResult::advance();
    }
    if (!mlir::isa<IdXOp, IdYOp, IdZOp, DimXOp, DimYOp, DimZOp>(op)) {
      return mlir::WalkResult::advance();
    }
    if (op->getParentOfType<mlir::LLVM::LLVMFuncOp>() != func) {
      op->emitError(
          "Hardware ID and dimension ops in non-entry functions are not "
          "supported in horizontal fusion.");
      return mlir::WalkResult::interrupt();
    }
    if (mlir::isa<IdXOp>(op)) {
      x_ops.push_back(op);
    } else if (mlir::isa<IdYOp>(op)) {
      y_ops.push_back(op);
    } else if (mlir::isa<IdZOp>(op)) {
      z_ops.push_back(op);
    } else if (mlir::isa<DimXOp>(op)) {
      dim_x_ops.push_back(op);
    } else if (mlir::isa<DimYOp>(op)) {
      dim_y_ops.push_back(op);
    } else if (mlir::isa<DimZOp>(op)) {
      dim_z_ops.push_back(op);
    }
    return mlir::WalkResult::advance();
  });
  if (walk_res.wasInterrupted()) {
    return mlir::failure();
  }

  mlir::Value dim_x_const = CreateI32Const(b, dim_x);
  mlir::Value dim_y_const = CreateI32Const(b, dim_y);
  mlir::Value dim_z_const = CreateI32Const(b, dim_z);
  ReplaceAllAndErase(b, dim_x_ops, dim_x_const);
  ReplaceAllAndErase(b, dim_y_ops, dim_y_const);
  ReplaceAllAndErase(b, dim_z_ops, dim_z_const);

  if (x_ops.empty() && y_ops.empty() && z_ops.empty()) {
    return mlir::success();
  }

  mlir::Value zero = CreateI32Const(b, 0);
  mlir::Value id_x = linear_id;
  mlir::Value id_y = zero;
  mlir::Value id_z = zero;
  if (dim_y > 1 || dim_z > 1) {
    id_x = mlir::LLVM::URemOp::create(b, linear_id, dim_x_const);
    mlir::Value id_yz = mlir::LLVM::UDivOp::create(b, linear_id, dim_x_const);
    id_y = id_yz;
    if (dim_z > 1) {
      id_y = mlir::LLVM::URemOp::create(b, id_yz, dim_y_const);
      id_z = mlir::LLVM::UDivOp::create(b, id_yz, dim_y_const);
    }
  }
  ReplaceAllAndErase(b, x_ops, id_x);
  ReplaceAllAndErase(b, y_ops, id_y);
  ReplaceAllAndErase(b, z_ops, id_z);
  return mlir::success();
}

mlir::LogicalResult RewritePartialCtaBarriers(mlir::ModuleOp module,
                                              mlir::ImplicitLocOpBuilder& b,
                                              int64_t sub_threads,
                                              int64_t max_threads,
                                              int64_t warp_size) {
  if (sub_threads >= max_threads) {
    return mlir::success();
  }

  llvm::SmallVector<mlir::NVVM::BarrierOp> barrier_ops;
  module.walk([&](mlir::NVVM::BarrierOp bar) {
    if (!bar.getNumberOfThreads()) {
      barrier_ops.push_back(bar);
    }
  });
  if (barrier_ops.empty()) {
    return mlir::success();
  }

  if (warp_size > 0 && sub_threads % warp_size != 0) {
    return module.emitError(absl::StrCat(
        "Sub-kernel uses CTA barriers with ", sub_threads,
        " threads (< max_threads=", max_threads,
        "), which is not a multiple of warp size (", warp_size, ")."));
  }

  mlir::OpBuilder::InsertionGuard guard(b);
  for (mlir::NVVM::BarrierOp bar : barrier_ops) {
    b.setInsertionPoint(bar);
    mlir::Value bar_id = bar.getBarrierId();
    if (!bar_id) {
      bar_id = CreateI32Const(b, 0);
    }
    mlir::Value num_threads_val = CreateI32Const(b, sub_threads);
    mlir::NVVM::BarrierOp::create(b, bar.getLoc(), bar_id, num_threads_val,
                                  b.getBoolAttr(false));
    bar.erase();
  }
  return mlir::success();
}

mlir::LogicalResult StripTrailingScratchArgs(mlir::LLVM::LLVMFuncOp func,
                                             mlir::ImplicitLocOpBuilder& b,
                                             int64_t expected_num_tensor_args) {
  if (expected_num_tensor_args < 0) {
    return mlir::success();
  }
  if (static_cast<int64_t>(func.getNumArguments()) < expected_num_tensor_args) {
    return func.emitError(
        absl::StrCat("Expected at least ", expected_num_tensor_args,
                     " arguments on sub-kernel ", func.getSymName().str(),
                     ", got ", func.getNumArguments()));
  }
  auto num_tensor_args = static_cast<unsigned>(expected_num_tensor_args);
  for (unsigned i = 0; i < num_tensor_args; ++i) {
    if (func.getArgAttr(i, "tt.nv_tma_desc") != nullptr) {
      return func.emitError(
          "Triton sub-kernels with TMA descriptor arguments are not "
          "supported in horizontal fusion.");
    }
  }
  if (func.getNumArguments() <= num_tensor_args) {
    return mlir::success();
  }
  llvm::BitVector args_to_erase(func.getNumArguments(), false);
  for (unsigned idx = num_tensor_args; idx < func.getNumArguments(); ++idx) {
    mlir::BlockArgument scratch_arg = func.getArgument(idx);
    if (!scratch_arg.use_empty()) {
      auto null_ptr = mlir::LLVM::ZeroOp::create(b, scratch_arg.getType());
      scratch_arg.replaceAllUsesWith(null_ptr.getResult());
    }
    args_to_erase.set(idx);
  }
  if (mlir::failed(func.eraseArguments(args_to_erase))) {
    return func.emitError("Failed to erase trailing scratch arguments.");
  }
  return mlir::success();
}

class VirtualizeSubKernelPass
    : public impl::VirtualizeSubKernelPassBase<VirtualizeSubKernelPass> {
 public:
  using impl::VirtualizeSubKernelPassBase<
      VirtualizeSubKernelPass>::VirtualizeSubKernelPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    if (auto scratch_attr = module->getAttrOfType<mlir::IntegerAttr>(
            "ttg.global_scratch_memory_size");
        scratch_attr && scratch_attr.getInt() > 0) {
      module.emitError(
          "Triton sub-kernels requiring global scratch memory are not "
          "supported in horizontal fusion.");
      signalPassFailure();
      return;
    }

    auto func =
        module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(entry_function_name_);
    if (func == nullptr || func.isExternal()) {
      module.emitError(absl::StrCat("Sub-kernel entry function '",
                                    entry_function_name_, "' not found."));
      signalPassFailure();
      return;
    }

    mlir::Block& entry_block = func.getBody().front();
    mlir::ImplicitLocOpBuilder b(func.getLoc(), &getContext());
    b.setInsertionPointToStart(&entry_block);

    // 1. Strip trailing scratch pointer arguments when requested.
    if (mlir::failed(
            StripTrailingScratchArgs(func, b, expected_num_tensor_args_))) {
      signalPassFailure();
      return;
    }

    // 2. Append `%rel_bid_i32 : i32` parameter to `func`.
    mlir::Value rel_bid_i32 =
        entry_block.addArgument(b.getI32Type(), b.getLoc());
    llvm::SmallVector<mlir::Type> new_param_types(
        func.getArgumentTypes().begin(), func.getArgumentTypes().end());
    new_param_types.push_back(b.getI32Type());
    func.setFunctionType(mlir::LLVM::LLVMFunctionType::get(
        func.getFunctionType().getReturnType(), new_param_types,
        /*isVarArg=*/false));
    if (mlir::ArrayAttr arg_attrs = func.getArgAttrsAttr()) {
      llvm::SmallVector<mlir::Attribute> new_arg_attrs(arg_attrs.begin(),
                                                       arg_attrs.end());
      new_arg_attrs.push_back(b.getDictionaryAttr({}));
      func.setArgAttrsAttr(b.getArrayAttr(new_arg_attrs));
    }

    // 3. Virtualize NVVM block IDs and grid dims.
    if (mlir::failed(
            VirtualizeIds<mlir::NVVM::BlockIdXOp, mlir::NVVM::BlockIdYOp,
                          mlir::NVVM::BlockIdZOp, mlir::NVVM::GridDimXOp,
                          mlir::NVVM::GridDimYOp, mlir::NVVM::GridDimZOp>(
                func, b, block_dim_x_, block_dim_y_, block_dim_z_,
                rel_bid_i32))) {
      signalPassFailure();
      return;
    }

    // 4. Virtualize NVVM thread IDs and block dims.
    mlir::Value raw_tid = mlir::NVVM::ThreadIdXOp::create(b, b.getI32Type());
    if (mlir::failed(
            VirtualizeIds<mlir::NVVM::ThreadIdXOp, mlir::NVVM::ThreadIdYOp,
                          mlir::NVVM::ThreadIdZOp, mlir::NVVM::BlockDimXOp,
                          mlir::NVVM::BlockDimYOp, mlir::NVVM::BlockDimZOp>(
                func, b, thread_dim_x_, thread_dim_y_, thread_dim_z_,
                raw_tid))) {
      signalPassFailure();
      return;
    }

    // 5. Rewrite partial-CTA barriers when `sub_threads < max_threads_`.
    int64_t sub_threads = thread_dim_x_ * thread_dim_y_ * thread_dim_z_;
    if (mlir::failed(RewritePartialCtaBarriers(module, b, sub_threads,
                                               max_threads_, warp_size_))) {
      signalPassFailure();
      return;
    }

    // 6. Pack shared memory globals and record `"xla.shared_memory_bytes"`.
    int64_t packed_bytes = PackSubKernelSharedMemory(module);
    int64_t ttg_shared_bytes = 0;
    if (auto ttg_shared =
            module->getAttrOfType<mlir::IntegerAttr>("ttg.shared")) {
      ttg_shared_bytes = ttg_shared.getInt();
    }
    int64_t shmem_bytes = std::max(packed_bytes, ttg_shared_bytes);
    module->setAttr("xla.shared_memory_bytes",
                    b.getI64IntegerAttr(shmem_bytes));

    func->removeAttr("xla.entry");
    func->removeAttr("nvvm.kernel");
    func->removeAttr("nvvm.maxntid");
    func->removeAttr("nvvm.reqntid");
    func->removeAttr("nvvm.cluster_dim");
    func->removeAttr("rocdl.kernel");
    func->removeAttr("rocdl.flat_work_group_size");
  }
};

}  // namespace
}  // namespace gpu
}  // namespace xla
