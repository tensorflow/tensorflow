/* Copyright 2024 The OpenXLA Authors.

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
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/atomic_rmw_utils.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace emitters {
namespace {

#define GEN_PASS_DEF_VECTORIZELOADSANDSTORESPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

using mlir::Value;

namespace arith = ::mlir::arith;
namespace ml = ::mlir::LLVM;
namespace scf = mlir::scf;

// Tries to find the stride of a symbol or dimension in an affine expression.
// Returns std::nullopt if the stride could not be determined.
//
// Note: this function only attempts to handle the cases where the stride is
// known to be 0 or 1.
//
// Example: the stride of `d0` in `(d0 + d1)` is 1.
// Example: the stride of `d0` in `d0 * 2` is unknown (nullopt).
std::optional<int> GetStride(mlir::AffineExpr expr,
                             mlir::AffineExpr dim_or_sym) {
  if (auto binop = mlir::dyn_cast_or_null<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs_stride = GetStride(binop.getLHS(), dim_or_sym);
    auto rhs_stride = GetStride(binop.getRHS(), dim_or_sym);

    if (binop.getKind() == mlir::AffineExprKind::Add) {
      if (lhs_stride && rhs_stride) {
        return *lhs_stride + *rhs_stride;
      }
      return std::nullopt;
    }
    // Just return 0 if the expression doesn't occur on either side.
    if (lhs_stride == 0 && rhs_stride == 0) {
      return 0;
    }
    // Otherwise, we don't know the stride.
    return std::nullopt;
  }
  return expr == dim_or_sym ? 1 : 0;
}

int64_t GetAlignmentOfRemainder(mlir::AffineExpr expr,
                                mlir::AffineExpr dim_or_sym) {
  if (auto binop = mlir::dyn_cast_or_null<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs_align = GetAlignmentOfRemainder(binop.getLHS(), dim_or_sym);
    auto rhs_align = GetAlignmentOfRemainder(binop.getRHS(), dim_or_sym);

    std::optional<int64_t> rhs_cst = std::nullopt;
    if (binop.getRHS().getKind() == mlir::AffineExprKind::Constant) {
      rhs_cst = binop.getRHS().cast<mlir::AffineConstantExpr>().getValue();
    }

    switch (binop.getKind()) {
      case mlir::AffineExprKind::Add:
        if (binop.getLHS() == dim_or_sym) return rhs_align;
        if (binop.getRHS() == dim_or_sym) return lhs_align;
        return std::gcd(lhs_align, rhs_align);
      case mlir::AffineExprKind::Mul:
        return lhs_align * rhs_align;
      case mlir::AffineExprKind::FloorDiv:
      case mlir::AffineExprKind::CeilDiv:
        return 1;
      case mlir::AffineExprKind::Mod:
        // (a * c) % (b * c) = (a % b) * c.
        return std::gcd(lhs_align, rhs_align);
      default:
        llvm_unreachable("expr is none of the binary expressions");
    }
  }
  if (auto cst = mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    return cst.getValue();
  }
  return 1;
}

// Attempts to extract the vector type for the given loop. This means:
// - checks that the lower bound is 0
// - checks that the step is 1
// - checks that the upper bound is 2, 4, or 8.
// Returns a vector type with the given upper bound and the tensor's element
// type.
// All tensors are 1D after flatten-tensors pass.
mlir::VectorType GetVectorType(mlir::RankedTensorType tensor_type,
                               scf::ForOp loop) {
  if (tensor_type.getEncoding()) {
    return nullptr;
  }
  if (tensor_type.getRank() != 1) {
    return nullptr;
  }
  if (!mlir::VectorType::isValidElementType(tensor_type.getElementType())) {
    return nullptr;
  }
  if (mlir::getConstantIntValue(loop.getStep()) != 1 ||
      mlir::getConstantIntValue(loop.getLowerBound()) != 0) {
    return nullptr;
  }
  std::optional<int> vector_size =
      mlir::getConstantIntValue(loop.getUpperBound());
  if (vector_size != 2 && vector_size != 4 && vector_size != 8) {
    return nullptr;  // Unsupported vector size.
  }
  if (tensor_type.getShape().back() % *vector_size) {
    return nullptr;  // Misaligned start indices.
  }
  return mlir::VectorType::get({*vector_size}, tensor_type.getElementType());
}

// Support 2xf32 and 4xf32 vectors as described in
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom
// Support for more vector types and architectures can be added here if needed.
mlir::VectorType GetVectorTypeForAtomicRMW(mlir::RankedTensorType tensor_type,
                                           scf::ForOp loop) {
  if (tensor_type.getEncoding()) {
    return nullptr;
  }
  if (tensor_type.getRank() != 1) {
    return nullptr;
  }

  if (tensor_type.getElementType() != mlir::Float32Type::get(loop.getContext()))
    return nullptr;

  if (mlir::getConstantIntValue(loop.getStep()) != 1 ||
      mlir::getConstantIntValue(loop.getLowerBound()) != 0) {
    return nullptr;
  }

  std::optional<int> vector_size =
      mlir::getConstantIntValue(loop.getUpperBound());
  if (vector_size != 2 && vector_size != 4) {
    return nullptr;
  }

  // Misaligned start indices.
  if (tensor_type.getShape().back() % *vector_size) {
    return nullptr;
  }
  return mlir::VectorType::get({*vector_size}, tensor_type.getElementType());
}

std::optional<Value> GetVectorBaseIndices(Value index, scf::ForOp loop,
                                          mlir::VectorType vector_type,
                                          mlir::ImplicitLocOpBuilder& b) {
  Value induction_var = loop.getInductionVar();
  if (index == induction_var) {
    return b.create<arith::ConstantIndexOp>(0);
  }

  auto apply_indexing =
      mlir::dyn_cast_or_null<ApplyIndexingOp>(index.getDefiningOp());
  if (!apply_indexing) {
    return std::nullopt;
  }

  // We don't generate these, but they are allowed in theory.
  if (apply_indexing->getNumResults() != 1) {
    return std::nullopt;
  }
  mlir::AffineMap map = apply_indexing.getAffineMap();

  int induction_var_operand_index;
  mlir::AffineExpr induction_var_expr = nullptr;
  for (auto [index, operand] : llvm::enumerate(apply_indexing.getOperands())) {
    if (operand == induction_var) {
      if (induction_var_expr) {
        // The induction variable should be used only once.
        return std::nullopt;
      }
      induction_var_operand_index = index;
      induction_var_expr = index < map.getNumDims()
                               ? mlir::getAffineDimExpr(index, b.getContext())
                               : mlir::getAffineSymbolExpr(
                                     index - map.getNumDims(), b.getContext());
    } else if (!operand.getParentRegion()->isProperAncestor(
                   &loop.getBodyRegion())) {
      // If the operand is defined inside the loop, we can't hoist the
      // apply_indexing outside the loop.
      return std::nullopt;
    }
  }
  if (!induction_var_expr) {
    return std::nullopt;
  }

  if (GetStride(map.getResult(0), induction_var_expr) != 1) {
    // The indexing map is not contiguous in the vectorized dimension.
    return std::nullopt;
  }

  if (GetAlignmentOfRemainder(map.getResult(0), induction_var_expr) %
      vector_type.getNumElements()) {
    return std::nullopt;
  }

  auto operands = llvm::to_vector(apply_indexing.getOperands());
  operands[induction_var_operand_index] = b.create<arith::ConstantIndexOp>(0);

  return b.create<ApplyIndexingOp>(operands, apply_indexing.getIndexingMap())
      ->getResult(0);
}

bool IsConflictFree(mlir::tensor::ExtractOp op) {
  return op.getTensor().getParentRegion()->isProperAncestor(
      op->getParentRegion());
}

struct VectorizeLoad : mlir::OpRewritePattern<mlir::tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto loop = mlir::dyn_cast_or_null<scf::ForOp>(op->getParentOp());
    if (!loop) {
      return rewriter.notifyMatchFailure(op, "no loop found");
    }
    if (!IsConflictFree(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "source may be written in the loop");
    }

    auto vector_type = GetVectorType(op.getTensor().getType(), loop);
    if (!vector_type) {
      return rewriter.notifyMatchFailure(op, "not a vectorizable loop");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(loop);
    auto vector_index =
        GetVectorBaseIndices(op.getIndices().front(), loop, vector_type, b);
    if (!vector_index) {
      return rewriter.notifyMatchFailure(
          op, "the instruction does not access contiguous elements");
    }
    auto loaded_vector = b.create<mlir::vector::TransferReadOp>(
        vector_type, op.getTensor(), *vector_index, llvm::ArrayRef<bool>{true});
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(
        op, loaded_vector, loop.getInductionVar());
    return mlir::success();
  }
};

// Verifies that the insertions happening in the loop can all safely be batched
// in the end.
bool IsConflictFree(mlir::tensor::InsertOp op) {
  // The insertion's only use must be the yield.
  if (!op->hasOneUse() || !mlir::isa<scf::YieldOp>(*op->user_begin())) {
    return false;
  }
  // The destination must be one of the loop's block arguments, and the
  // destination must be the argument's only use.
  auto bbarg = mlir::dyn_cast<mlir::BlockArgument>(op.getDest());
  return bbarg && bbarg.hasOneUse() &&
         bbarg.getOwner()->getParentOp() == op->getParentOp();
}

bool IsConflictFree(AtomicRMWOp op) {
  // The insertion's only use must be the yield.
  if (!op->hasOneUse() || !mlir::isa<scf::YieldOp>(*op->user_begin())) {
    return false;
  }
  // The destination must be one of the loop's block arguments, and the
  // destination must be the argument's only use.
  auto bbarg = mlir::dyn_cast<mlir::BlockArgument>(op->getOpOperand(0).get());
  return bbarg && bbarg.hasOneUse() &&
         bbarg.getOwner()->getParentOp() == op->getParentOp();
}

class VectorizeAtomicRMW : public mlir::OpRewritePattern<AtomicRMWOp> {
 public:
  VectorizeAtomicRMW(mlir::MLIRContext* context, const DeviceSpec& device_spec)
      : OpRewritePattern<AtomicRMWOp>(context), device_spec_(device_spec) {}
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      AtomicRMWOp op, mlir::PatternRewriter& rewriter) const override {
    if (!device_spec_.IsNvidiaGpu() ||
        !device_spec_.gpu().cuda_compute_capability().IsAtLeastHopper()) {
      return rewriter.notifyMatchFailure(
          op,
          "atomic vectorization currently only supported on Hopper or later");
    }
    auto loop = mlir::dyn_cast_or_null<scf::ForOp>(op->getParentOp());
    if (!loop) {
      return rewriter.notifyMatchFailure(op, "no loop found");
    }

    if (!IsConflictFree(op)) {
      return rewriter.notifyMatchFailure(op, "write may be read back by loop");
    }

    auto vector_type = GetVectorTypeForAtomicRMW(op.getType(), loop);
    if (!vector_type) {
      return rewriter.notifyMatchFailure(op, "loop is not vectorizable");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(loop);
    auto atomic_modifier_parameters = emitters::GetAtomicModifierParameters(op);
    if (!atomic_modifier_parameters.has_value() ||
        atomic_modifier_parameters->second != ml::AtomicBinOp::fadd) {
      return rewriter.notifyMatchFailure(op,
                                         "atomic modifier is not vectorizable. "
                                         "Currently only supporting fadd.");
    }

    auto vector_index =
        GetVectorBaseIndices(op.getIndices().front(), loop, vector_type, b);
    if (!vector_index) {
      return rewriter.notifyMatchFailure(
          op, "the instruction does not access contiguous elements");
    }

    auto init =
        b.create<arith::ConstantOp>(b.getZeroAttr(vector_type)).getResult();

    auto yield_fn = [&](mlir::OpBuilder& yield_b, mlir::Location yield_loc,
                        llvm::ArrayRef<mlir::BlockArgument> bbarg) {
      auto induction_var =
          mlir::cast<scf::ForOp>(bbarg.front().getOwner()->getParentOp())
              .getInductionVar();
      auto insert_op = yield_b.create<mlir::vector::InsertOp>(
          yield_loc, atomic_modifier_parameters->first, bbarg.front(),
          induction_var);
      return llvm::SmallVector<Value>{insert_op.getResult()};
    };
    int result_index = op->use_begin()->getOperandNumber();
    auto new_for = *loop.replaceWithAdditionalYields(
        rewriter, init,
        /*replaceInitOperandUsesInLoop=*/false, yield_fn);

    b.setInsertionPointAfter(new_for);
    rewriter.replaceOp(op, op->getOpOperand(0).get());

    auto filled_vector = new_for->getResults().back();
    auto new_atomic_rmw = b.create<AtomicRMWOp>(
        new_for.getInits()[result_index], *vector_index, vector_type);
    mlir::ImplicitLocOpBuilder body_builder(new_atomic_rmw.getLoc(),
                                            new_atomic_rmw.getBodyBuilder());
    auto addf_op = body_builder.create<arith::AddFOp>(
        body_builder.getLoc(), vector_type, new_atomic_rmw.getCurrentValue(),
        filled_vector);
    body_builder.create<xla::YieldOp>(addf_op.getResult());
    new_for->getResult(result_index)
        .replaceAllUsesWith(new_atomic_rmw.getResult());

    return mlir::success();
  }

  const DeviceSpec& device_spec_;
};

struct VectorizeStore : mlir::OpRewritePattern<mlir::tensor::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto loop = mlir::dyn_cast_or_null<scf::ForOp>(op->getParentOp());
    if (!loop) {
      return rewriter.notifyMatchFailure(op, "no loop found");
    }
    if (!IsConflictFree(op)) {
      return rewriter.notifyMatchFailure(op, "write may be read back by loop");
    }
    auto vector_type = GetVectorType(op.getDest().getType(), loop);
    if (!vector_type) {
      return rewriter.notifyMatchFailure(op, "loop is not vectorizable");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(loop);
    auto vector_index =
        GetVectorBaseIndices(op.getIndices().front(), loop, vector_type, b);
    if (!vector_index) {
      return rewriter.notifyMatchFailure(
          op, "the instruction does not access contiguous elements");
    }

    auto init =
        b.create<arith::ConstantOp>(b.getZeroAttr(vector_type)).getResult();

    auto yield_fn = [&](mlir::OpBuilder& yield_b, mlir::Location yield_loc,
                        llvm::ArrayRef<mlir::BlockArgument> bbarg) {
      auto induction_var =
          mlir::cast<scf::ForOp>(bbarg.front().getOwner()->getParentOp())
              .getInductionVar();
      auto insert_op = yield_b.create<mlir::vector::InsertOp>(
          yield_loc, op.getScalar(), bbarg.front(), induction_var);
      return llvm::SmallVector<Value>{insert_op.getResult()};
    };
    int result_index = op->use_begin()->getOperandNumber();
    auto new_for = *loop.replaceWithAdditionalYields(
        rewriter, init,
        /*replaceInitOperandUsesInLoop=*/false, yield_fn);

    b.setInsertionPointAfter(new_for);
    rewriter.replaceOp(op, op.getDest());

    auto filled_vector = new_for->getResults().back();
    auto written = b.create<mlir::vector::TransferWriteOp>(
        filled_vector, new_for.getInits()[result_index], *vector_index,
        llvm::ArrayRef<bool>{true});
    new_for->getResult(result_index).replaceAllUsesWith(written.getResult());

    return mlir::success();
  }
};

class VectorizeLoadsAndStoresPass
    : public impl::VectorizeLoadsAndStoresPassBase<
          VectorizeLoadsAndStoresPass> {
 public:
  explicit VectorizeLoadsAndStoresPass(
      const VectorizeLoadsAndStoresPassOptions& options)
      : VectorizeLoadsAndStoresPassBase(options) {}

  explicit VectorizeLoadsAndStoresPass(
      const se::DeviceDescription& device_description)
      : device_spec_(device_description) {}

  void runOnOperation() override {
    if (target_type_ == "gpu" && !gpu_device_info_.empty()) {
      se::GpuDeviceInfoProto device_info;
      CHECK(tsl::protobuf::TextFormat::ParseFromString(gpu_device_info_,
                                                       &device_info));
      *device_spec_.mutable_type() = se::DeviceDescription(device_info);
    } else if (target_type_ == "cpu") {
      CHECK(gpu_device_info_.empty());
      *device_spec_.mutable_type() = CpuDeviceSpec{};
    }
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<VectorizeLoad, VectorizeStore>(mlir_context);
    patterns.add<VectorizeAtomicRMW>(mlir_context, device_spec_);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

  DeviceSpec device_spec_;
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateVectorizeLoadsAndStoresPass(
    const std::string& target_type, const std::string& gpu_device_info) {
  VectorizeLoadsAndStoresPassOptions options;
  options.gpu_device_info_ = gpu_device_info;
  options.target_type_ = target_type;
  return std::make_unique<VectorizeLoadsAndStoresPass>(options);
}

std::unique_ptr<mlir::Pass> CreateVectorizeLoadsAndStoresPass(
    const se::DeviceDescription& device_description) {
  return std::make_unique<VectorizeLoadsAndStoresPass>(device_description);
}

}  // namespace emitters
}  // namespace xla
