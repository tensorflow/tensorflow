//===- InferQuantizedTypesPass.cpp - Infers quantized types ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the primary pass for instantiating a CAG, running it to
// convergence on a module to determine eligible quantized type transforms, and
// applying those transforms to the IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Quantizer/Configurations/FxpMathConfig.h"
#include "mlir/Quantizer/Support/Configuration.h"
#include "mlir/Quantizer/Support/ConstraintAnalysisGraph.h"
#include "mlir/Quantizer/Support/ConstraintAnalysisGraphTraits.h"
#include "mlir/Quantizer/Transforms/Passes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::quantizer;
using namespace mlir::quant;

namespace llvm {

template <>
struct DOTGraphTraits<const CAGSlice *>
    : public DOTGraphTraits<const CAGNode *> {
  DOTGraphTraits(bool isSimple = false)
      : DOTGraphTraits<const CAGNode *>(isSimple) {}

  std::string getNodeLabel(const CAGNode *node, const CAGSlice *graph) {
    std::string s;
    llvm::raw_string_ostream out(s);
    node->printLabel(out);
    return out.str();
  }

  static std::string getGraphProperties(const CAGSlice *) {
    return "rankdir=LR;";
  }

  static bool isNodeHidden(const CAGNode *node) {
    // Filter constraint nodes with no incoming or outgoing connections.
    // These orphans are often created as part of graph merging operations.
    return llvm::isa<CAGConstraintNode>(node) && node->isOrphan();
  }

  std::string getNodeAttributes(const CAGNode *node, const CAGSlice *graph) {
    switch (node->getKind()) {
    default:
      return std::string();
    case CAGNode::Kind::OperandAnchor:
      return "shape=record,color=yellow,style=filled";
    case CAGNode::Kind::ResultAnchor:
      return "shape=record,color=lightblue,style=filled";
    case CAGNode::Kind::Constraint:
      return "shape=record,style=dotted";
    }
  }
};

} // end namespace llvm

namespace {

class InferQuantizedTypesPass : public ModulePass<InferQuantizedTypesPass> {
public:
  InferQuantizedTypesPass() = default;
  InferQuantizedTypesPass(SolverContext &solverContext,
                          const TargetConfiguration &config)
      : explicitSolverContext(&solverContext), explicitConfig(&config) {}
  void runOnModule() override;
  void runWithConfig(SolverContext &solverContext,
                     const TargetConfiguration &config);

  void transformOperandType(CAGOperandAnchor *anchor, Type newType);
  void transformResultType(CAGResultAnchor *anchor, Type newType);

private:
  SolverContext *explicitSolverContext = nullptr;
  const TargetConfiguration *explicitConfig = nullptr;
};

} // end anonymous namespace

/// Maximum number of propagation rounds to run to converge the CAG before
/// signalling an error.
static const int kMaximumPropagationRounds = 1000;

static LogicalResult validateTypeConversion(Type newType, Type origType,
                                            Operation *op) {
  if (!newType) {
    return op->emitOpError() << "unsupported type conversion from " << newType;
  }
  return success();
}

void InferQuantizedTypesPass::runOnModule() {
  if (explicitSolverContext && explicitConfig) {
    // If explicitly constructed with a config and context.
    runWithConfig(*explicitSolverContext, *explicitConfig);
    return;
  }

  // For global pass registration, use defaults.
  SolverContext solverContext(*getModule().getContext());
  auto config = FxpMathTargetConfig::create(solverContext);
  runWithConfig(solverContext, *config);
}

void InferQuantizedTypesPass::runWithConfig(SolverContext &solverContext,
                                            const TargetConfiguration &config) {
  CAGSlice cag(solverContext);
  for (auto f : getModule().getOps<FuncOp>()) {
    f.walk([&cag, &config](Operation *op) { config.handleOp(op, cag); });
  }
  config.finalizeAnchors(cag);

  // Propagate.
  int propRound;
  for (propRound = kMaximumPropagationRounds; propRound > 0; --propRound) {
    auto propCount = cag.propagate(config);
    if (propCount == 0)
      break;
  }
  if (propRound == 0) {
    emitError(UnknownLoc::get(&getContext()),
              "exceeded maximum number of solver iterations (infinite loop?)");
    return;
  }

  // TODO: Only dump the GraphViz if a flag is set and move to a utility.
  // GraphViz.
  if (!solverContext.getDebugCAGDotPath().empty()) {
    auto actFileName =
        llvm::WriteGraph(const_cast<const CAGSlice *>(&cag), "CAG",
                         /*ShortNames=*/false,
                         /*Title=*/"CAG",
                         /*Filename=*/solverContext.getDebugCAGDotPath());
    llvm::errs() << "Wrote graphviz file: " << actFileName << "\n";
  }

  // Start transforming the types in order of anchor type (results, then
  // operands).
  // Apply result types.
  for (auto *node : cag) {
    auto anchorNode = dyn_cast<CAGResultAnchor>(node);
    if (!anchorNode)
      continue;
    if (Type newType = anchorNode->getTransformedType())
      transformResultType(anchorNode, newType);
  }

  // Apply operand types.
  for (auto *node : cag) {
    auto anchorNode = dyn_cast<CAGOperandAnchor>(node);
    if (!anchorNode)
      continue;
    if (Type newType = anchorNode->getTransformedType())
      transformOperandType(anchorNode, newType);
  }
}

void InferQuantizedTypesPass::transformOperandType(CAGOperandAnchor *anchor,
                                                   Type newType) {
  Value inputValue = anchor->getValue();
  Operation *op = anchor->getOp();
  OpBuilder b(op->getBlock(), Block::iterator(op));

  SmallVector<Value, 1> removeValuesIfDead;

  // Because we've already run the result transforms at this phase, it is
  // very likely that inputValue points to a dcast op whose input matches
  // our type. We detect that situation and route around just to save some
  // bulk in the IR.
  Value newTypedInputValue = inputValue;
  auto inputDcastOp =
      dyn_cast_or_null<DequantizeCastOp>(inputValue->getDefiningOp());
  if (inputDcastOp && inputDcastOp.arg()->getType() == newType) {
    // Can just use the dcast's input value.
    newTypedInputValue = inputDcastOp.arg();
    removeValuesIfDead.push_back(inputDcastOp);
  } else {
    // Need to synthesize a qcast.
    newTypedInputValue =
        b.create<QuantizeCastOp>(op->getLoc(), newType, inputValue);
  }

  switch (anchor->getTypeTransformRule()) {
  case CAGAnchorNode::TypeTransformRule::Direct:
    anchor->getOp()->setOperand(anchor->getOperandIdx(), newTypedInputValue);
    break;

  case CAGAnchorNode::TypeTransformRule::DirectStorage: {
    Type storageType = QuantizedType::castToStorageType(newType);
    if (failed(validateTypeConversion(storageType, newType, op)))
      return;
    anchor->getOp()->setOperand(
        anchor->getOperandIdx(),
        b.create<StorageCastOp>(op->getLoc(), storageType, newTypedInputValue));
    break;
  }

  case CAGAnchorNode::TypeTransformRule::ExpressedOnly:
    // Leave the anchor as-is and just cast in/out after it.
    anchor->getOp()->setOperand(
        anchor->getOperandIdx(),
        b.create<DequantizeCastOp>(op->getLoc(), anchor->getOriginalType(),
                                   newTypedInputValue));
    break;
  }

  for (Value removeValueIfDead : removeValuesIfDead) {
    if (removeValueIfDead->use_empty()) {
      removeValueIfDead->getDefiningOp()->erase();
    }
  }
}

void InferQuantizedTypesPass::transformResultType(CAGResultAnchor *anchor,
                                                  Type newType) {
  Value origResultValue = anchor->getValue();
  Operation *op = origResultValue->getDefiningOp();
  OpBuilder b(op->getBlock(), ++Block::iterator(op));

  Value replacedResultValue = nullptr;
  Value newResultValue = nullptr;
  switch (anchor->getTypeTransformRule()) {
  case CAGAnchorNode::TypeTransformRule::Direct:
    origResultValue->setType(newType);
    replacedResultValue = newResultValue = b.create<DequantizeCastOp>(
        op->getLoc(), anchor->getOriginalType(), origResultValue);
    break;

  case CAGAnchorNode::TypeTransformRule::DirectStorage: {
    Type storageType = QuantizedType::castToStorageType(newType);
    if (failed(validateTypeConversion(storageType, newType, op)))
      return;
    origResultValue->setType(storageType);
    replacedResultValue =
        b.create<StorageCastOp>(op->getLoc(), newType, origResultValue);
    newResultValue = b.create<DequantizeCastOp>(
        op->getLoc(), anchor->getOriginalType(), replacedResultValue);
    break;
  }

  case CAGAnchorNode::TypeTransformRule::ExpressedOnly:
    // Leave the anchor as-is and just cast in/out after it.
    replacedResultValue =
        b.create<QuantizeCastOp>(op->getLoc(), newType, origResultValue);
    newResultValue = b.create<DequantizeCastOp>(
        op->getLoc(), anchor->getOriginalType(), replacedResultValue);
    break;
  }

  if (replacedResultValue) {
    // Transform:
    //   origResultValue -->  replaceResultValue -> newResultValue
    //                   \->  [original uses]
    // To:
    //   origResultValue -> replaceResultValue ->
    //                      newResultValue -> [original uses]
    // Note that replaceResultValue may equal newResultValue or there may
    // be operands between the two.
    origResultValue->replaceAllUsesWith(newResultValue);
    replacedResultValue->getDefiningOp()->replaceUsesOfWith(newResultValue,
                                                            origResultValue);
  }
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::quantizer::createInferQuantizedTypesPass(
    SolverContext &solverContext, const TargetConfiguration &config) {
  return std::make_unique<InferQuantizedTypesPass>(solverContext, config);
}

static PassRegistration<InferQuantizedTypesPass>
    pass("quantizer-infer-quantized-types",
         "Infers quantized types for a module");
