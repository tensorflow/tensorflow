//===- FxpMathConfig.cpp - Reference fixed point config -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a TargetConfiguration for reference fixed-point math
// quantization scheme based on the FxpMathOps (plus a small category of
// extension ops that can be added from other dialects).
//
//===----------------------------------------------------------------------===//

#include "mlir/Quantizer/Configurations/FxpMathConfig.h"

#include "mlir/Dialect/FxpMathOps/FxpMathOps.h"
#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Quantizer/Support/ConstraintAnalysisGraph.h"
#include "mlir/Quantizer/Support/Metadata.h"
#include "mlir/Quantizer/Support/Statistics.h"
#include "mlir/Quantizer/Support/UniformConstraints.h"

using namespace mlir;
using namespace mlir::quantizer;
using namespace mlir::fxpmath;
using namespace mlir::quant;
using namespace std::placeholders;

namespace {

struct FxpMathTargetConfigImpl : public FxpMathTargetConfig {
  FxpMathTargetConfigImpl(SolverContext &context)
      : FxpMathTargetConfig(context) {
    Builder b(&context.getMlirContext());
    IntegerType i8Type = b.getIntegerType(8);
    IntegerType i16Type = b.getIntegerType(16);
    IntegerType i32Type = b.getIntegerType(32);

    q8 = addCandidateType(
        AnyQuantizedType::get(QuantizationFlags::Signed, i8Type, nullptr,
                              std::numeric_limits<int8_t>::min(),
                              std::numeric_limits<int8_t>::max()),
        CandidateQuantizedType::Scheme::UniformPerLayer);
    q16 = addCandidateType(
        AnyQuantizedType::get(QuantizationFlags::Signed, i16Type, nullptr,
                              std::numeric_limits<int16_t>::min(),
                              std::numeric_limits<int16_t>::max()),
        CandidateQuantizedType::Scheme::UniformPerLayer);
    q32ExplicitFixedPoint = addCandidateType(
        AnyQuantizedType::get(QuantizationFlags::Signed, i32Type, nullptr,
                              std::numeric_limits<int32_t>::min(),
                              std::numeric_limits<int32_t>::max()),
        CandidateQuantizedType::Scheme::UniformExplicitFixedPointScale);

    // Op handlers.
    addOpHandler<ConstantOp>(
        std::bind(&FxpMathTargetConfigImpl::handleConstant, this, _1, _2));
    addOpHandler<ReturnOp>(
        std::bind(&FxpMathTargetConfigImpl::handleTerminal, this, _1, _2));
    addOpHandler<quant::StatisticsOp>(
        std::bind(&FxpMathTargetConfigImpl::handleStats, this, _1, _2));

    // FxpMathOps.
    addOpHandler<RealAddEwOp>(
        std::bind(&FxpMathTargetConfigImpl::handleAdd, this, _1, _2));
    addOpHandler<RealMulEwOp>(
        std::bind(&FxpMathTargetConfigImpl::handleMul, this, _1, _2));
    addOpHandler<RealMatMulOp>(
        std::bind(&FxpMathTargetConfigImpl::handleMatMul, this, _1, _2));
    addOpHandler<RealMatMulBiasOp>(
        std::bind(&FxpMathTargetConfigImpl::handleMatMulBias, this, _1, _2));

    // Require stats ops.
    addRequireStatsOp<RealAddEwOp>();
    addRequireStatsOp<RealSubEwOp>();
    addRequireStatsOp<RealDivEwOp>();
    addRequireStatsOp<RealMulEwOp>();
    addRequireStatsOp<RealMatMulOp>();
    addRequireStatsOp<RealMatMulBiasOp>();
  }

  bool isHandledType(Type t) const final {
    if (t.isa<FloatType>())
      return true;
    return (t.isa<VectorType>() || t.isa<TensorType>()) &&
           t.cast<ShapedType>().getElementType().isa<FloatType>();
  }

  void finalizeAnchors(CAGSlice &cag) const override {
    cag.enumerateImpliedConnections(
        [&](CAGAnchorNode *from, CAGAnchorNode *to) {
          UniformConstraintsBuilder(cag).coupleAnchors(from, to);
        });
  }

  void addValueIdentityOpByName(StringRef opName) override {
    addOpHandlerByName(
        opName,
        std::bind(&FxpMathTargetConfigImpl::handleValueIdentity, this, _1, _2));
  }

  void handleValueIdentity(Operation *op, CAGSlice &cag) const {
    assert(op->getNumResults() == 1);
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto resultNode = cag.getResultAnchor(op, 0);
    resultNode->setTypeTransformRule(
        CAGAnchorNode::TypeTransformRule::DirectStorage);

    for (unsigned opIdx = 0, e = op->getNumOperands(); opIdx < e; ++opIdx) {
      if (!isHandledType(op->getOperand(opIdx)->getType()))
        continue;
      auto operandNode = cag.getOperandAnchor(op, opIdx);
      operandNode->setTypeTransformRule(
          CAGAnchorNode::TypeTransformRule::DirectStorage);
      UniformConstraintsBuilder(cag).coupleAnchors(operandNode, resultNode);
    }
  }

  void handleConstant(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto resultNode = cag.getResultAnchor(op, 0);
    resultNode->setTypeTransformRule(
        CAGAnchorNode::TypeTransformRule::ExpressedOnly);
    Attribute valueAttr;
    if (!matchPattern(op, m_Constant(&valueAttr))) {
      return;
    }

    AttributeTensorStatistics stats(valueAttr);
    TensorAxisStatistics layerStats;
    if (!stats.get(layerStats)) {
      op->emitOpError("could not compute statistics");
      return;
    }

    UniformConstraintsBuilder(cag).applyStats(resultNode, layerStats);
  }

  void handleTerminal(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getOperand(0)->getType()))
      return;
    auto operandNode = cag.getOperandAnchor(op, 0);
    operandNode->setTypeTransformRule(
        CAGAnchorNode::TypeTransformRule::ExpressedOnly);
  }

  void handleStats(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto argNode = cag.getOperandAnchor(op, 0);
    auto resultNode = cag.getResultAnchor(op, 0);
    UniformConstraintsBuilder(cag).coupleAnchors(argNode, resultNode);

    TensorAxisStatistics layerStats;
    auto statsOp = cast<quant::StatisticsOp>(op);
    auto layerStatsAttr = statsOp.layerStats();
    layerStats.minValue =
        layerStatsAttr.getValue<FloatAttr>(0).getValueAsDouble();
    layerStats.maxValue =
        layerStatsAttr.getValue<FloatAttr>(1).getValueAsDouble();
    UniformConstraintsBuilder(cag).applyStats(resultNode, layerStats);
  }

  void handleAdd(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Add supports 8/16 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({q8, q16});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    // NOTE: We couple the add such that the scale/zeroPoint match between
    // both args and the result. This is overly constrained in that it is
    // possible to write efficient add kernels with a bit more freedom (i.e.
    // zeroPoints can vary, scales can differ by a power of two, etc).
    // However, fully coupled yields the simples solutions on the fast path.
    // Further efficiency can be had by constraining the zeroPoint to 0, but
    // there isn't a constraint for this yet (and there are tradeoffs).
    UniformConstraintsBuilder(cag).coupleAnchors(lhs, resultNode);
    UniformConstraintsBuilder(cag).coupleAnchors(rhs, resultNode);
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleMul(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Mul supports 8/16 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({q8, q16});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleMatMul(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Mul supports 8/16 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({q8, q16});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleMatMulBias(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto bias = cag.getOperandAnchor(op, 2);
    bias->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});

    auto resultNode = cag.getResultAnchor(op, 0);
    UniformConstraintsBuilder(cag).propagateExplicitScale(resultNode, bias);

    // Mul supports 8/16 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({q8, q16});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void addRealMathOptionalConstraints(Operation *op, CAGAnchorNode *anchor,
                                      CAGSlice &cag) const {
    // TODO: It would be nice if these all extended some base trait instead
    // of requiring name lookup.
    auto clampMinAttr = op->getAttrOfType<FloatAttr>("clamp_min");
    auto clampMaxAttr = op->getAttrOfType<FloatAttr>("clamp_max");

    if (clampMinAttr || clampMaxAttr) {
      auto nan = APFloat::getQNaN(APFloat::IEEEdouble());
      auto clampMin = clampMinAttr ? clampMinAttr.getValue() : nan;
      auto clampMax = clampMaxAttr ? clampMaxAttr.getValue() : nan;
      UniformConstraintsBuilder(cag).clamp(anchor, clampMin, clampMax);
    }
  }

  unsigned q8;
  unsigned q16;
  unsigned q32ExplicitFixedPoint;
};

} // anonymous namespace

std::unique_ptr<FxpMathTargetConfig>
FxpMathTargetConfig::create(SolverContext &context) {
  return std::make_unique<FxpMathTargetConfigImpl>(context);
}
