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

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DEF_INITIALIZEALLOCSPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

// Values chosen in the hope that they will result in failed tests if what
// otherwise would be uninitialized memory is incorrectly read.
std::optional<mlir::Value> GetConstantValue(mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            mlir::Type element_type) {
  if (element_type.isFloat()) {
    mlir::FloatType float_type = mlir::dyn_cast<mlir::FloatType>(element_type);
    llvm::APFloat nan_value =
        llvm::APFloat::getNaN(float_type.getFloatSemantics());
    return mlir::arith::ConstantOp::create(
        builder, loc, mlir::FloatAttr::get(float_type, nan_value));
  }

  if (element_type.isInteger()) {
    mlir::IntegerType int_type =
        mlir::dyn_cast<mlir::IntegerType>(element_type);
    mlir::APInt value = mlir::APInt::getAllOnes(int_type.getWidth());
    return mlir::arith::ConstantOp::create(
        builder, loc, mlir::IntegerAttr::get(int_type, value));
  }

  return std::nullopt;
}

void InitializeAlloc(mlir::memref::AllocOp op) {
  mlir::Location loc = op.getLoc();
  mlir::OpBuilder builder(op->getContext());
  builder.setInsertionPointAfter(op);

  mlir::TypedValue<mlir::MemRefType> memref = op.getResult();
  mlir::MemRefType memref_type = op.getType();

  std::optional<mlir::Value> value =
      GetConstantValue(builder, loc, memref_type.getElementType());
  if (!value.has_value()) {
    op->emitWarning() << "Can't initialize alloc with unsupported element type";
    return;
  }

  int64_t rank = memref_type.getRank();
  llvm::SmallVector<mlir::Value> lbs(
      rank, mlir::arith::ConstantIndexOp::create(builder, loc, 0));
  llvm::SmallVector<mlir::Value> step(
      rank, mlir::arith::ConstantIndexOp::create(builder, loc, 1));
  llvm::SmallVector<mlir::Value> ubs;
  ubs.reserve(rank);
  for (int64_t idx = 0; idx != rank; ++idx) {
    mlir::Value dim_size = mlir::memref::DimOp::create(builder, loc, op, idx);
    ubs.push_back(dim_size);
  }

  mlir::scf::buildLoopNest(
      builder, loc, lbs, ubs, step,
      [value, memref](mlir::OpBuilder& builder, mlir::Location loc,
                      mlir::ValueRange ivs) {
        mlir::memref::StoreOp::create(builder, loc, *value, memref, ivs);
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
