/* Copyright 2023 The JAX Authors.

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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_DEBUGASSERTINSERTIONPASS
#define GEN_PASS_DEF_DEBUGASSERTINSERTIONPASS
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

using rule_type = std::function<void(Operation *)>;

template <typename Op>
rule_type as_generic_rule(void (*rule)(Op)) {
  return [rule](const Operation *op) { return rule(cast<Op>(op)); };
}

void assertIsValidSubwindow(Operation *op, mlir::ValueRange base_indices,
                            ArrayRef<int64_t> window_shape,
                            ArrayRef<int64_t> full_shape,
                            ArrayRef<int32_t> strides = {}) {
  if (base_indices.size() != window_shape.size() ||
      base_indices.size() != full_shape.size() ||
      (!strides.empty() && base_indices.size() != strides.size())) {
    return;  // Malformed op.
  }
  if (base_indices.empty()) {
    return;
  }
  Type idx_type = base_indices.front().getType();
  ImplicitLocOpBuilder builder(op->getLoc(), op);
  for (auto [dim, access] :
       llvm::enumerate(llvm::zip(base_indices, window_shape, full_shape))) {
    auto [idx, size, bound] = access;
    int64_t stride = strides.empty() ? 1 : strides[dim];
    Value positive = builder.create<arith::CmpIOp>(
        arith::CmpIPredicate::sge, idx,
        builder.create<arith::ConstantOp>(builder.getIntegerAttr(idx_type, 0)));
    Value in_bounds = builder.create<arith::CmpIOp>(
        arith::CmpIPredicate::slt,
        builder.create<arith::AddIOp>(
            idx, builder.create<arith::ConstantOp>(
                     builder.getIntegerAttr(idx_type, (size - 1) * stride))),
        builder.create<arith::ConstantOp>(
            builder.getIntegerAttr(idx_type, bound)));
    std::string msg;
    llvm::raw_string_ostream msg_builder(msg);
    msg_builder << "Operation " << op->getName().getStringRef().str()
                << " references out-of-bounds elements in dimension "
                << std::to_string(dim) << " (source location: " << op->getLoc()
                << ")";
    builder.create<cf::AssertOp>(
        builder.create<arith::AndIOp>(positive, in_bounds), msg);
  }
}

void vector_load_rule(vector::LoadOp op) {
  assertIsValidSubwindow(op, op.getIndices(),
                         /*window_shape=*/op.getVectorType().getShape(),
                         /*full_shape=*/op.getBase().getType().getShape());
}

void vector_store_rule(vector::StoreOp op) {
  assertIsValidSubwindow(op, op.getIndices(),
                         /*window_shape=*/op.getVectorType().getShape(),
                         /*full_shape=*/op.getBase().getType().getShape());
}

void tpu_memref_slice_rule(tpu::MemRefSliceOp op) {
  assertIsValidSubwindow(op, op.getBaseIdx(),
                         /*window_shape=*/op.getResult().getType().getShape(),
                         /*full_shape=*/op.getMemRef().getType().getShape());
}

void tpu_strided_load_rule(tpu::StridedLoadOp op) {
  assertIsValidSubwindow(op, op.getIndices(),
                         /*window_shape=*/op.getResult().getType().getShape(),
                         /*full_shape=*/op.getBase().getType().getShape(),
                         /*strides=*/op.getStrides());
}

void tpu_strided_store_rule(tpu::StridedStoreOp op) {
  assertIsValidSubwindow(
      op, op.getIndices(),
      /*window_shape=*/op.getValueToStore().getType().getShape(),
      /*full_shape=*/op.getBase().getType().getShape(),
      /*strides=*/op.getStrides());
}

void tpu_vector_store_rule(tpu::VectorStoreOp op) {
  // TODO(b/379925823): Take strides into account.
  assertIsValidSubwindow(
      op, op.getIndices(),
      /*window_shape=*/op.getValueToStore().getType().getShape(),
      /*full_shape=*/op.getBase().getType().getShape());
}

const llvm::StringMap<rule_type> &rules() {
  static auto rules = new llvm::StringMap<rule_type>{
      // TODO: tpu::LoadOp, tpu::StoreOp
      {vector::LoadOp::getOperationName(), as_generic_rule(vector_load_rule)},
      {vector::StoreOp::getOperationName(), as_generic_rule(vector_store_rule)},
      {tpu::MemRefSliceOp::getOperationName(),
       as_generic_rule(tpu_memref_slice_rule)},
      {tpu::StridedLoadOp::getOperationName(),
       as_generic_rule(tpu_strided_load_rule)},
      {tpu::StridedStoreOp::getOperationName(),
       as_generic_rule(tpu_strided_store_rule)},
      {tpu::VectorStoreOp::getOperationName(),
       as_generic_rule(tpu_vector_store_rule)},
  };
  return *rules;
}

struct DebugAssertInsertionPass
    : public impl::DebugAssertInsertionPassBase<DebugAssertInsertionPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([](Operation *op) {
      if (auto rule_it = rules().find(op->getName().getStringRef());
          rule_it != rules().end()) {
        const rule_type &rule = rule_it->getValue();
        rule(op);
      }
      return WalkResult::advance();
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDebugAssertInsertionPass() {
  return std::make_unique<DebugAssertInsertionPass>();
}

}  // namespace mlir::tpu
