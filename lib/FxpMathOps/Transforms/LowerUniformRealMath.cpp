//===- LowerUniformRealMath.cpp  ------------------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/FxpMathOps/FxpMathOps.h"
#include "mlir/FxpMathOps/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Quantization/UniformSupport.h"

#include <functional>

using namespace mlir;
using namespace mlir::fxpmath;
using namespace mlir::quant;

namespace {

struct LowerUniformRealMathPass
    : public FunctionPass<LowerUniformRealMathPass> {
  void runOnFunction() override;
};

UniformQuantizedType getUniformElementType(Type t) {
  return QuantizedType::getQuantizedElementType(t)
      .dyn_cast_or_null<UniformQuantizedType>();
}

/// Computes the log2(x), rounded to an integral value. Returns whether 'x' can
/// be considered an exact integral value.
template <typename F> bool integralLog2(F x, int &log2Result) {
  const F xLog2 = std::log(x) * (1.0 / std::log(2.0));
  const F xLog2Rounded = std::round(xLog2);
  const F xLog2Frac = xLog2 - xLog2Rounded;
  log2Result = static_cast<int>(xLog2Rounded);
  // Allow small comparison slop below the level that would make a difference
  // for 2^16 levels.
  return std::abs(xLog2Frac) < 1e-6;
}

/// Helper class for operating on binary operations where all operands
/// and the result are a UniformQuantizedType.
struct RealBinaryOpInfo {
  RealBinaryOpInfo(Operation *op, Value *lhs, Value *rhs,
                   Optional<APFloat> clampMin, Optional<APFloat> clampMax)
      : op(op), lhs(lhs), rhs(rhs), clampMin(clampMin), clampMax(clampMax),
        lhsType(getUniformElementType(lhs->getType())),
        rhsType(getUniformElementType(rhs->getType())),
        resultType(getUniformElementType(*op->result_type_begin())),
        lhsStorageType(QuantizedType::castToStorageType(lhs->getType())),
        rhsStorageType(QuantizedType::castToStorageType(rhs->getType())),
        resultStorageType(
            QuantizedType::castToStorageType(*op->result_type_begin())) {}

  /// Returns whether this info is valid (all types defined, etc).
  bool isValid() const {
    return lhsType && rhsType && resultType && lhsStorageType &&
           rhsStorageType && resultStorageType;
  }

  /// Returns whether the storage type of all operands is identical.
  bool isSameStorageType() const {
    return lhsType.getStorageType() == rhsType.getStorageType() &&
           lhsType.getStorageType() == resultType.getStorageType();
  }

  /// Returns whether all operands and result are considered fixedpoint power
  /// of two, setting the lhs, rhs, and result log2 scale references.
  bool isFixedPointPOT(int &lhsLog2Scale, int &rhsLog2Scale,
                       int &resultLog2Scale) const {
    if (!lhsType.isFixedPoint() || !rhsType.isFixedPoint() ||
        !resultType.isFixedPoint()) {
      return false;
    }

    if (!integralLog2(lhsType.getScale(), lhsLog2Scale) ||
        !integralLog2(rhsType.getScale(), rhsLog2Scale) ||
        !integralLog2(resultType.getScale(), resultLog2Scale)) {
      return false;
    }

    return true;
  }

  /// Gets the result integer clamp range given the result quantized type
  // and any explicit clamp provided as attributes.
  std::pair<IntegerAttr, IntegerAttr> getClampMinMax() const {
    int64_t typeMin = resultType.getStorageTypeMin();
    int64_t typeMax = resultType.getStorageTypeMax();

    if (clampMin || clampMax) {
      UniformQuantizedValueConverter conv(resultType);
      if (clampMin) {
        typeMin = std::max(typeMin, conv.quantizeFloatToInt64(*clampMin));
      }
      if (clampMax) {
        typeMax = std::min(typeMax, conv.quantizeFloatToInt64(*clampMax));
      }
    }

    // The quantized, integral ops expect clamps as 32bit ints.
    return {
        IntegerAttr::get(IntegerType::get(32, resultType.getContext()),
                         typeMin),
        IntegerAttr::get(IntegerType::get(32, resultType.getContext()),
                         typeMax),
    };
  }

  Operation *op;
  Value *lhs;
  Value *rhs;
  Optional<APFloat> clampMin;
  Optional<APFloat> clampMax;

  // Element UniformQuantizedType for operands/result.
  UniformQuantizedType lhsType;
  UniformQuantizedType rhsType;
  UniformQuantizedType resultType;

  // Full storage-based types.
  Type lhsStorageType;
  Type rhsStorageType;
  Type resultStorageType;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Elementwise add
//===----------------------------------------------------------------------===//
/// Attempts to rewrite a fixed point power-of-two addition of two integers.
/// This supports a limited number of cases, but when supported, represents
/// the simplest computation.
static LogicalResult tryRewriteFixedPOTAddEw(const RealBinaryOpInfo &constInfo,
                                             PatternRewriter &rewriter) {
  if (!constInfo.isSameStorageType()) {
    return failure();
  }

  int lhsLog2Scale;
  int rhsLog2Scale;
  int resultLog2Scale;
  if (!constInfo.isFixedPointPOT(lhsLog2Scale, rhsLog2Scale, resultLog2Scale)) {
    return failure();
  }

  // Adjust shifts to be relative to the output.
  // Left shift of one input scale is supported. The other must match the result
  // scale.
  int lhsScaleShift = lhsLog2Scale - resultLog2Scale;
  int rhsScaleShift = rhsLog2Scale - resultLog2Scale;
  if (lhsScaleShift != 0 && rhsScaleShift != 0) {
    return failure();
  }
  if (lhsScaleShift > 0 || rhsScaleShift > 0) {
    return failure();
  }

  // State accessed by the closure.
  Operation *mathOp = constInfo.op;
  const auto clampMinMax = constInfo.getClampMinMax();
  Value *lhs = constInfo.lhs;
  Value *rhs = constInfo.rhs;
  Type lhsStorageType = constInfo.lhsStorageType;
  Type rhsStorageType = constInfo.rhsStorageType;

  // If the lhs operand is the one requiring a shift, swap it so that the shift
  // happens the rhs operand.
  if (lhsScaleShift != 0) {
    std::swap(lhs, rhs);
    std::swap(lhsStorageType, rhsStorageType);
    std::swap(lhsScaleShift, rhsScaleShift);
  }
  int rhsRightShift = -rhsScaleShift;

  // Cast operands to storage type.
  Value *lhsStorageValue =
      rewriter.create<StorageCastOp>(mathOp->getLoc(), lhsStorageType, lhs)
          .getResult();
  Value *rhsStorageValue =
      rewriter.create<StorageCastOp>(mathOp->getLoc(), rhsStorageType, rhs)
          .getResult();

  // Rescale the rhs operand if needed.
  if (rhsRightShift != 0) {
    rhsStorageValue =
        rewriter
            .create<RoundingDivideByPotFxpOp>(
                mathOp->getLoc(), rhsStorageValue,
                IntegerAttr::get(IntegerType::get(32, rewriter.getContext()),
                                 rhsRightShift))
            .getResult();
  }

  // Add.
  Value *sumValue = rewriter.create<SaturatingAddFxpOp>(
      mathOp->getLoc(), lhsStorageValue, rhsStorageValue, clampMinMax.first,
      clampMinMax.second);

  // Cast back for new result.
  rewriter.replaceOpWithNewOp<StorageCastOp>(
      mathOp, *mathOp->result_type_begin(), sumValue);
  return success();
}

namespace {

struct UniformRealAddEwPattern : public RewritePattern {
  UniformRealAddEwPattern(MLIRContext *context)
      : RewritePattern(RealAddEwOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const {
    auto addOp = op->cast<RealAddEwOp>();
    const RealBinaryOpInfo info(op, addOp.x(), addOp.y(), addOp.clamp_min(),
                                addOp.clamp_max());
    if (!info.isValid()) {
      return matchFailure();
    }

    // Try all of the permutations we support.
    if (succeeded(tryRewriteFixedPOTAddEw(info, rewriter))) {
      return matchSuccess();
    }

    return matchFailure();
  }
};

} // end anonymous namespace

void LowerUniformRealMathPass::runOnFunction() {
  auto &fn = getFunction();
  OwningRewritePatternList patterns;
  auto *context = &getContext();
  patterns.push_back(llvm::make_unique<UniformRealAddEwPattern>(context));
  applyPatternsGreedily(fn, std::move(patterns));
}

FunctionPassBase *createLowerUniformRealMathPass() {
  return new LowerUniformRealMathPass();
}

static PassRegistration<LowerUniformRealMathPass>
    pass("fxpmath-lower-uniform-real-math",
         "Lowers uniform-quantized real math ops to integer arithmetic.");
