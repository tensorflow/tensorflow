/* Copyright 2024 The JAX Authors.

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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_RELAYOUTINSERTIONPASS
#define GEN_PASS_DEF_RELAYOUTINSERTIONPASS
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

FailureOr<TypedValue<VectorType>> relayout(
    OpBuilder &builder, TypedValue<VectorType> v, VectorLayout src,
    VectorLayout dst, int hardware_generation,
    const std::array<int64_t, 2> target_shape) {
  // change bitwidth
  if (v.getType().getElementType() == builder.getI1Type() &&
      // TODO(jevinjiang): for other relayout changes (tiling, offsets, implicit
      // dim), we currently rely on apply-vector-layout pass to do the relayout.
      src.bitwidth() != dst.bitwidth()) {
    auto vreg_slice = src.vregSlice(target_shape, dst.bitwidth(), src.tiling());
    auto dst_bitwidth_layout = VectorLayout(
        dst.bitwidth(),
        {
            src.offsets()[0].has_value() ? *src.offsets()[0] % vreg_slice[0]
                                         : LayoutOffset(),
            src.offsets()[1].has_value() ? *src.offsets()[1] % vreg_slice[1]
                                         : LayoutOffset(),
        },
        src.tiling(), src.implicit_dim());
    if (!dst_bitwidth_layout.isValid(target_shape)) {
      return emitError(v.getLoc(),
                       "Not implemented: failed to infer valid layout during "
                       "relayout, got ")
             << dst_bitwidth_layout;
    }
    // We might be able to pack mask directly.
    // TODO(jevinjiang): Add support for 16bit -> 8bit mask packing.
    if (src.bitwidth() == 32 && dst.bitwidth() == 16 &&
        // TODO(jevinjiang): support mask packing for non-native source tiling.
        src.tiling()[0] == src.packing() * target_shape[0] &&
        src.tiling()[1] == target_shape[1]) {
      auto relayout_op =
          builder.create<tpu::RelayoutOp>(v.getLoc(), v.getType(), v);
      setLayout(relayout_op, src, dst_bitwidth_layout);
      return cast<TypedValue<VectorType>>(relayout_op.getResult());
    }
    CHECK(llvm::isPowerOf2_32(src.bitwidth()));
    CHECK(llvm::isPowerOf2_32(dst.bitwidth()));
    auto make_vty = [&](int bitwidth) {
      return VectorType::get(v.getType().getShape(),
                             builder.getIntegerType(bitwidth));
    };
    auto make_constant = [&](int val, VectorLayout layout) {
      auto vty = make_vty(layout.bitwidth());
      auto constant_op = builder.create<arith::ConstantOp>(
          v.getLoc(),
          DenseElementsAttr::get(
              vty, builder.getIntegerAttr(vty.getElementType(), val)));
      setOutLayout(constant_op,
                   VectorLayout(layout.bitwidth(), {std::nullopt, std::nullopt},
                                layout.tiling(), layout.implicit_dim()));
      return constant_op;
    };
    auto src_int_vty = make_vty(src.bitwidth());
    auto dst_int_vty = make_vty(dst.bitwidth());
    // TODO(jevinjiang): Since dst_bitwidth_layout will be firstly used in the
    // extSI or truncI below, we can reuse the inferExt and inferTrunc from
    // infer-vector-layout pass.
    auto ext_op = builder.create<arith::ExtUIOp>(v.getLoc(), src_int_vty, v);
    setLayout(ext_op, src, src);

    // TODO(jevinjiang): some conversion might not be supported in HW.
    Operation *cast_op =
        dst.bitwidth() > src.bitwidth()
            ? builder.create<arith::ExtSIOp>(v.getLoc(), dst_int_vty, ext_op)
            // TODO(jevinjiang): HW may support pack vmask directly.
            : builder.create<arith::TruncIOp>(v.getLoc(), dst_int_vty, ext_op);
    setLayout(cast_op, src, dst_bitwidth_layout);

    auto cmp_op = builder.create<arith::CmpIOp>(
        v.getLoc(), v.getType(), arith::CmpIPredicate::ne,
        cast_op->getResult(0), make_constant(0, dst_bitwidth_layout));
    setLayout(cmp_op, {dst_bitwidth_layout, dst_bitwidth_layout},
              dst_bitwidth_layout);
    return cast<TypedValue<VectorType>>(cmp_op.getResult());
  }
  // Fall through to generic relayout.
  auto relayout_op =
      builder.create<tpu::RelayoutOp>(v.getLoc(), v.getType(), v);
  setLayout(relayout_op, src, dst);

  return cast<TypedValue<VectorType>>(relayout_op.getResult());
}

LogicalResult insertRelayout(Operation &op, int hardware_generation,
                             std::array<int64_t, 2> target_shape);

LogicalResult insertRelayoutBlock(Block &block, int hardware_generation,
                                  const std::array<int64_t, 2> target_shape) {
  // We'll be modifying the block, so use early increment.
  for (Operation &op : make_early_inc_range(block)) {
    if (failed(insertRelayout(op, hardware_generation, target_shape))) {
      return failure();
    }
  }
  return success();
}

// TODO(jevinjiang): make relayout to an op so we don't need decide when to
// relayout in apply-vector-layout pass.
LogicalResult insertRelayout(Operation &op, int hardware_generation,
                             const std::array<int64_t, 2> target_shape) {
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> in_layouts,
                             getInLayouts(op, target_shape));
  if (in_layouts.size() != op.getNumOperands()) {
    return op.emitError("Expected the same number of operands as in_layouts");
  }
  if (isa<tpu::AssumeLayoutOp>(op)) {
    return success();
  }
  // Relayout the operands, if their requested input layouts don't match the
  // layouts in which they were produced.
  for (auto [idx, tup] :
       llvm::enumerate(llvm::zip(op.getOperands(), in_layouts))) {
    auto [operand, li] = tup;
    auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand);
    TPU_ASSERT_EQ_OP(vector_operand != nullptr, li.has_value());
    if (vector_operand == nullptr) {
      continue;
    }
    // The operand should always be an Operation (and not a BlockArgument)
    // since we expect the FuncOp to have only memrefs and semaphores as
    // arguments.
    auto op_result = dyn_cast<OpResult>(vector_operand);
    if (op_result == nullptr) {
      return op.emitError("Expected vector operand to be an operation result");
    }
    Operation *const def_op = op_result.getOwner();
    DCHECK(def_op);
    const unsigned res_idx = op_result.getResultNumber();
    FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                               getOutLayouts(*def_op, target_shape));
    const Layout lo = def_layouts[res_idx];
    TPU_ASSERT_OP(lo.has_value());
    if (*lo == *li) {
      continue;
    }
    OpBuilder builder(&op);
    FAILUREOR_ASSIGN_OR_RETURN(
        Value new_v, relayout(builder, vector_operand, /*src=*/*lo,
                              /*dst=*/*li, hardware_generation, target_shape));
    op.setOperand(idx, new_v);
  }

  for (auto &region : op.getRegions()) {
    for (auto &block : region.getBlocks()) {
      if (failed(
              insertRelayoutBlock(block, hardware_generation, target_shape))) {
        return failure();
      }
    }
  }
  return success();
}

struct RelayoutInsertionPass
    : public impl::RelayoutInsertionPassBase<RelayoutInsertionPass> {
  RelayoutInsertionPass(int generation, std::array<int64_t, 2> target_shape) {
    this->hardware_generation = generation;
    this->sublane_count = target_shape[0];
    this->lane_count = target_shape[1];
  }
  void runOnOperation() override {
    // Fail if hardware_generation has not been set from the default value.
    if (hardware_generation < 0) {
      getOperation().emitError("hardware_generation must be set");
      signalPassFailure();
      return;
    }
    func::FuncOp func = getOperation();
    auto result = func.walk([&](Operation *op) {
      if (insertRelayout(*op, hardware_generation, {sublane_count, lane_count})
              .failed()) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRelayoutInsertionPass(
    int hardware_generation, std::array<int64_t, 2> target_shape) {
  return std::make_unique<RelayoutInsertionPass>(hardware_generation,
                                                 target_shape);
}

}  // namespace mlir::tpu