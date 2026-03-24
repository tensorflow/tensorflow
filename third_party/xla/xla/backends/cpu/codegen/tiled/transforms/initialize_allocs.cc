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

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DEF_INITIALIZEALLOCSPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

mlir::func::FuncOp GetUnpoisonFuncDecl(mlir::ImplicitLocOpBuilder& builder,
                                       mlir::ModuleOp module) {
  mlir::Type ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::FunctionType func_type = mlir::FunctionType::get(
      builder.getContext(), {ptr_type, builder.getIndexType()}, {});

  auto existing_func =
      mlir::func::lookupFnDecl(module, "__msan_unpoison", func_type);
  // Crash OK - this would only fail when an existing declaration has the same
  // name with different signature.
  CHECK(mlir::succeeded(existing_func));
  if (*existing_func) {
    return *existing_func;
  }

  return mlir::func::createFnDecl(builder, module, "__msan_unpoison", func_type,
                                  /*setPrivate=*/true);
}

// Get the size of the buffer in bytes.
mlir::Value GetBufferSize(mlir::ImplicitLocOpBuilder& builder,
                          mlir::TypedValue<mlir::MemRefType> buffer,
                          mlir::DataLayout data_layout) {
  mlir::Value num_elements = mlir::arith::ConstantIndexOp::create(builder, 1);
  for (int64_t dim_idx = 0; dim_idx < buffer.getType().getRank(); ++dim_idx) {
    auto dim_size = mlir::memref::DimOp::create(builder, buffer, dim_idx);
    num_elements = mlir::arith::MulIOp::create(builder, num_elements, dim_size);
  }

  int64_t element_count =
      data_layout.getTypeSize(buffer.getType().getElementType());
  mlir::Value element_count_value =
      mlir::arith::ConstantIndexOp::create(builder, element_count);

  mlir::Value byte_size =
      mlir::arith::MulIOp::create(builder, num_elements, element_count_value);
  return byte_size;
}

void MarkUnPoison(mlir::ImplicitLocOpBuilder& builder,
                  mlir::TypedValue<mlir::MemRefType> buffer,
                  mlir::ModuleOp module) {
  auto func_decl = GetUnpoisonFuncDecl(builder, module);

  mlir::DataLayout data_layout(module);
  auto index_int_type = builder.getIntegerType(
      data_layout.getTypeSizeInBits(builder.getIndexType()));

  mlir::Value index_ptr =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, buffer);
  mlir::Value int_ptr =
      mlir::arith::IndexCastOp::create(builder, index_int_type, index_ptr);
  mlir::Value ptr = mlir::LLVM::IntToPtrOp::create(
      builder, mlir::LLVM::LLVMPointerType::get(builder.getContext()), int_ptr);

  mlir::Value buffer_size = GetBufferSize(builder, buffer, data_layout);

  mlir::func::CallOp::create(builder, func_decl, {ptr, buffer_size});
}

// Values chosen in the hope that they will result in failed tests if what
// otherwise would be uninitialized memory is incorrectly read.
std::optional<mlir::Value> GetConstantValue(mlir::ImplicitLocOpBuilder& builder,
                                            mlir::Type element_type) {
  if (element_type.isFloat()) {
    mlir::FloatType float_type = mlir::dyn_cast<mlir::FloatType>(element_type);
    llvm::APFloat nan_value =
        llvm::APFloat::getNaN(float_type.getFloatSemantics());
    return mlir::arith::ConstantOp::create(
        builder, mlir::FloatAttr::get(float_type, nan_value));
  }

  if (element_type.isInteger()) {
    mlir::IntegerType int_type =
        mlir::dyn_cast<mlir::IntegerType>(element_type);
    mlir::APInt value = mlir::APInt::getAllOnes(int_type.getWidth());
    return mlir::arith::ConstantOp::create(
        builder, mlir::IntegerAttr::get(int_type, value));
  }

  return std::nullopt;
}

void InitializeAlloc(mlir::memref::AllocOp op) {
  mlir::Location loc = op.getLoc();
  mlir::ImplicitLocOpBuilder builder(loc, op->getContext());
  builder.setInsertionPointAfter(op);

  mlir::TypedValue<mlir::MemRefType> memref = op.getResult();
  mlir::MemRefType memref_type = op.getType();

  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    op->emitWarning() << "Can't initialize alloc with no parent module";
    return;
  }

  std::optional<mlir::Value> value =
      GetConstantValue(builder, memref_type.getElementType());
  if (!value.has_value()) {
    op->emitWarning() << "Can't initialize alloc with unsupported element type";
    return;
  }

  MarkUnPoison(builder, memref, module);

  int64_t rank = memref_type.getRank();
  llvm::SmallVector<mlir::Value> lbs(
      rank, mlir::arith::ConstantIndexOp::create(builder, 0));
  llvm::SmallVector<mlir::Value> step(
      rank, mlir::arith::ConstantIndexOp::create(builder, 1));
  llvm::SmallVector<mlir::Value> ubs;
  ubs.reserve(rank);
  for (int64_t idx = 0; idx != rank; ++idx) {
    mlir::Value dim_size = mlir::memref::DimOp::create(builder, op, idx);
    ubs.push_back(dim_size);
  }

  mlir::scf::buildLoopNest(
      builder, loc, lbs, ubs, step,
      [value, memref](mlir::OpBuilder& loop_builder, mlir::Location loop_loc,
                      mlir::ValueRange ivs) {
        mlir::memref::StoreOp::create(loop_builder, loop_loc, *value, memref,
                                      ivs);
      });
}

class InitializeAllocsPass
    : public impl::InitializeAllocsPassBase<InitializeAllocsPass> {
 public:
  using InitializeAllocsPassBase::InitializeAllocsPassBase;

  void runOnOperation() override {
    llvm::SmallVector<mlir::memref::AllocOp> allocs;
    getOperation()->walk(
        [&](mlir::memref::AllocOp alloc) { allocs.push_back(alloc); });

    for (auto alloc : allocs) {
      InitializeAlloc(alloc);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateInitializeAllocsPass() {
  return std::make_unique<InitializeAllocsPass>();
}

}  // namespace xla::cpu
