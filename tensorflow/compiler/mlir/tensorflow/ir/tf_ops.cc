/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#include <algorithm>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Parser.h"  // TF:local_config_mlir
#include "mlir/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/STLExtras.h"  // TF:local_config_mlir
#include "mlir/Support/TypeUtilities.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace TF {

namespace {
#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_canonicalize.inc"
}  // namespace

//===----------------------------------------------------------------------===//
// TF op helper functions
//===----------------------------------------------------------------------===//

// Returns true if the given `value` is of ranked float tensor type with the
// given `rank`.
static inline bool isOfRankedFloatTensorType(Value *value, int rank) {
  auto type = value->getType().dyn_cast<RankedTensorType>();
  return type && type.getRank() == rank &&
         type.getElementType().isa<FloatType>();
}

// Returns true if the given `value` has the specified rank or has unranked
// type.
static inline bool IsOfRankOrUnranked(Value *value, int64_t rank) {
  if (auto type = value->getType().dyn_cast<RankedTensorType>()) {
    return type.getRank() == rank;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  RewriteListBuilder<AddToAddV2>::build(results, context);
}

//===----------------------------------------------------------------------===//
// AddV2Op
//===----------------------------------------------------------------------===//

void AddV2Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  RewriteListBuilder<AddV2OfNegLeft, AddV2OfNegRight>::build(results, context);
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

void BitcastOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  RewriteListBuilder<BitcastSameType, BitcastNested>::build(results, context);
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(BroadcastToOp op) {
  // TODO(antiagainst): check that
  // * The 'shape' input is an 1-D int tensor.
  // * Each dimension pair of the source and target shapes are either equal
  //   or one of them is one.
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

void CastOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  RewriteListBuilder<CastSameType>::build(results, context);
}

//===----------------------------------------------------------------------===//
// ConjOp
//===----------------------------------------------------------------------===//

void ConjOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  RewriteListBuilder<ConjNested>::build(results, context);
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

// Builds a constant op with the specified attribute `value`. The result
// op's type is deduced from `value`; if `value` is of scalar type,
// wraps it up with a tensor type of empty shape.
void ConstOp::build(Builder *builder, OperationState *result, Attribute value) {
  ShapedType type;
  if (auto elemAttr = value.dyn_cast<ElementsAttr>()) {
    type = elemAttr.getType();
  } else if (value.isa<BoolAttr>() || value.isa<FloatAttr>() ||
             value.isa<IntegerAttr>()) {
    // All TensorFlow types must be tensor types. In the build() method,
    // we want to provide more flexiblity by allowing attributes of scalar
    // types. But we need to wrap it up with ElementsAttr to construct
    // valid TensorFlow constants.
    type = RankedTensorType::get(/*shape=*/{}, value.getType());
    value = DenseElementsAttr::get(type, value);
  }
  // TODO: support other TensorFlow specific types.
  assert(type && "unsupported attribute type for building tf.Const");
  result->types.push_back(type);
  result->addAttribute("value", value);
}

void ConstOp::build(Builder *builder, OperationState *result, Type type,
                    Attribute value) {
  // Handle the case where the type and value are already tensors.
  if (type.isa<TensorType>() && value.isa<ElementsAttr>()) {
    result->addTypes(type);
    result->addAttribute("value", value);
    return;
  }

  // Otherwise, default to the attribute builder.
  ConstOp::build(builder, result, value);
  assert(type == result->types[0] && "type mismatch in construction");
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

void DivOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  RewriteListBuilder<DivWithSqrtDivisor>::build(results, context);
}

//===----------------------------------------------------------------------===//
// FakeQuantWithMinMaxArgsOp
//===----------------------------------------------------------------------===//
static LogicalResult Verify(FakeQuantWithMinMaxArgsOp op) {
  // TODO(fengliuai): moving the following to an utility method.
  const llvm::fltSemantics &semantics = op.min().getSemantics();
  float rmin, rmax;
  if (&semantics == &APFloat::IEEEsingle()) {
    rmin = op.min().convertToFloat();
    rmax = op.max().convertToFloat();
  } else {
    rmin = op.min().convertToDouble();
    rmax = op.max().convertToDouble();
  }
  // Range boundaries must be valid.
  if (rmin >= rmax) {
    return op.emitOpError("range is invalid: [" + Twine(std::to_string(rmin)) +
                          "," + Twine(std::to_string(rmax)) + "]");
  }
  // Range must straddle zero.
  if (rmin > 0.0 || rmax < 0.0) {
    return op.emitOpError("range failed to straddle zero: [" +
                          Twine(std::to_string(rmin)) + "," +
                          Twine(std::to_string(rmax)) + "]");
  }
  int64_t num_bits = op.num_bits().getSExtValue();
  if (num_bits < 2 || num_bits > 16) {
    return op.emitOpError(
        "requires num_bits to be between 2 and 16, inclusive");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FakeQuantWithMinMaxVarsOp
//===----------------------------------------------------------------------===//
static LogicalResult Verify(FakeQuantWithMinMaxVarsOp op) {
  if (!isOfRankedFloatTensorType(op.min(), 0))
    return op.emitOpError("requires min to be a 0d float tensor");

  if (!isOfRankedFloatTensorType(op.max(), 0))
    return op.emitOpError("requires max to be a 0d float tensor");

  int64_t num_bits = op.num_bits().getSExtValue();
  if (num_bits < 2 || num_bits > 16) {
    return op.emitOpError(
        "requires num_bits to be between 2 and 16, inclusive");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FusedBatchNormOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(FusedBatchNormOp op) {
  if (!isOfRankedFloatTensorType(op.x(), 4))
    return op.emitOpError("requires x to be a 4D float tensor");

  if (!isOfRankedFloatTensorType(op.scale(), 1))
    return op.emitOpError("requires scale to be a 1D float tensor");

  if (!isOfRankedFloatTensorType(op.offset(), 1))
    return op.emitOpError("requires offset to be a 1D float tensor");

  if (!isOfRankedFloatTensorType(op.mean(), 1))
    return op.emitOpError("requires mean to be a 1D float tensor");

  if (!isOfRankedFloatTensorType(op.variance(), 1))
    return op.emitOpError("requires variance to be a 1D float tensor");

  // TODO(antiagainst): check attributes

  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verify() {
  auto thenAttr = getAttrOfType<FunctionAttr>("then_branch");
  if (!thenAttr) return emitOpError("requires then_branch attribute");

  auto elseAttr = getAttrOfType<FunctionAttr>("else_branch");
  if (!elseAttr) return emitOpError("requires else_branch attribute");

  auto module = getParentOfType<Module>();
  auto thenFn = module.getNamedFunction(thenAttr.getValue());
  if (!thenFn)
    return emitOpError("then_branch refers to an undefined function : ")
           << thenAttr;
  auto elseFn = module.getNamedFunction(elseAttr.getValue());
  if (!elseFn)
    return emitOpError("else_branch refers to an undefined function : ")
           << elseAttr;
  auto thenFuncType = thenFn.getType();
  auto elseFuncType = elseFn.getType();

  // Non-conditional operands starting with the second operand are passed to
  // branches and should be pair-wise compatible with branches' inputs.
  unsigned expectedNumInputs = getNumOperands() - 1;
  if (thenFuncType.getNumInputs() != expectedNumInputs ||
      elseFuncType.getNumInputs() != expectedNumInputs)
    return emitError("branches should have " + Twine(expectedNumInputs) +
                     " inputs");

  for (unsigned i = 0; i < expectedNumInputs; ++i) {
    auto operandType = getOperand(i + 1)->getType().cast<TensorType>();
    auto thenInputType = thenFuncType.getInput(i).cast<TensorType>();
    if (!TensorCastOp::areCastCompatible(operandType, thenInputType))
      return emitError(
          llvm::formatv("then branch input type {0} is incompatible with "
                        "operand type {1} at index {2}",
                        thenInputType, operandType, i));

    auto elseInputType = elseFuncType.getInput(i).cast<TensorType>();
    if (!TensorCastOp::areCastCompatible(operandType, elseInputType))
      return emitError(
          llvm::formatv("else branch input type {0} is incompatible with "
                        "operand type {1} at index {2}",
                        elseInputType, operandType, i));

    // If branches have incompatible input types that means that no tensor can
    // serve as input to both the functions. Hence, the op is invalid.
    if (!TensorCastOp::areCastCompatible(thenInputType, elseInputType))
      return emitError(llvm::formatv(
          "branches inputs have incompatible types {0} and {1} at index {2}",
          thenInputType, elseInputType, i));
  }

  // Branches' results should be pair-wise compatible with the op results.
  unsigned expectedNumResults = getNumResults();
  if (thenFuncType.getNumResults() != expectedNumResults ||
      elseFuncType.getNumResults() != expectedNumResults)
    return emitError("branches should have " + Twine(expectedNumResults) +
                     " results");

  for (unsigned i = 0; i < expectedNumResults; ++i) {
    auto resultType = getResult(i)->getType().cast<TensorType>();
    auto thenResultType = thenFuncType.getResult(i).cast<TensorType>();
    if (!TensorCastOp::areCastCompatible(thenResultType, resultType))
      return emitError(
          llvm::formatv("then branch result type {0} is incompatible with op "
                        "result type {1} at index {2}",
                        thenResultType, resultType, i));

    auto elseResultType = elseFuncType.getResult(i).cast<TensorType>();
    if (!TensorCastOp::areCastCompatible(elseResultType, resultType))
      return emitError(
          llvm::formatv("else branch result type {0} is incompatible with op "
                        "result type {1} at index {2}",
                        elseResultType, resultType, i));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// InvertOp
//===----------------------------------------------------------------------===//

void InvertOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  RewriteListBuilder<InvertNested>::build(results, context);
}

//===----------------------------------------------------------------------===//
// LeakyReluOp
//===----------------------------------------------------------------------===//

OpFoldResult LeakyReluOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "leaky relu has one operand");

  // leaky_relu(x, alpha: 1) -> x
  if (alpha().convertToFloat() == 1.0f) return getOperand();

  auto calculate = [&](FloatAttr arg) {
    APFloat val = arg.getValue();
    if (val.isNegative()) val = alpha() * val;
    return FloatAttr::get(arg.getType(), val);
  };

  if (auto arg = operands[0].dyn_cast_or_null<FloatAttr>()) {
    return calculate(arg);
  } else if (auto arg = operands[0].dyn_cast_or_null<SplatElementsAttr>()) {
    if (auto elementAttr = arg.getSplatValue().dyn_cast<FloatAttr>())
      return DenseElementsAttr::get(arg.getType(), calculate(elementAttr));
  }
  return {};
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

void LogOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  RewriteListBuilder<LogOfSoftmax>::build(results, context);
}

//===----------------------------------------------------------------------===//
// LogicalNotOp
//===----------------------------------------------------------------------===//

void LogicalNotOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  RewriteListBuilder<LogicalNotNested, LogicalNotOfEqual, LogicalNotOfNotEqual,
                     LogicalNotOfGreater, LogicalNotOfGreaterEqual,
                     LogicalNotOfLess, LogicalNotOfLessEqual>::build(results,
                                                                     context);
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

void NegOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  RewriteListBuilder<NegNested>::build(results, context);
}

//===----------------------------------------------------------------------===//
// ReciprocalOp
//===----------------------------------------------------------------------===//

void ReciprocalOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  RewriteListBuilder<ReciprocalNested>::build(results, context);
}

//===----------------------------------------------------------------------===//
// RandomUniformOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(RandomUniformOp op) {
  if (!IsOfRankOrUnranked(op.shape(), 1))
    return op.emitOpError("shape must be 1D tensor");
  return success();
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

void RangeOp::build(Builder *builder, OperationState *result, Value *start,
                    Value *limit, Value *delta) {
  assert(start->getType() == limit->getType());
  assert(start->getType() == delta->getType());
  DenseIntElementsAttr start_val;
  DenseIntElementsAttr limit_val;
  DenseIntElementsAttr delta_val;
  if (matchPattern(start, m_Constant(&start_val)) &&
      matchPattern(limit, m_Constant(&limit_val)) &&
      matchPattern(delta, m_Constant(&delta_val))) {
    auto size = llvm::APIntOps::RoundingSDiv(
        *limit_val.begin() - *start_val.begin(), *delta_val.begin(),
        llvm::APInt::Rounding::DOWN);
    return RangeOp::build(
        builder, result,
        builder->getTensorType(
            size.getSExtValue(),
            start->getType().cast<TensorType>().getElementType()),
        start, limit, delta);
  }
  return RangeOp::build(
      builder, result,
      builder->getTensorType(
          {-1}, start->getType().cast<TensorType>().getElementType()),
      start, limit, delta);
}
//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

void RankOp::build(Builder *builder, OperationState *result, Value *input) {
  return RankOp::build(builder, result,
                       builder->getTensorType({}, builder->getIntegerType(32)),
                       input);
}

//===----------------------------------------------------------------------===//
// RealDivOp
//===----------------------------------------------------------------------===//

void RealDivOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  RewriteListBuilder<RealDivWithSqrtDivisor>::build(results, context);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// TODO(b/128020684): Verify the rank of the output and change to use
// m_Constant.
static LogicalResult Verify(ReshapeOp op) {
  auto shapeType = op.shape()->getType().cast<TensorType>();
  if (shapeType.getRank() != 1)
    return op.emitOpError("shape must be 1D tensor");
  auto rankByShape = shapeType.getShape()[0];
  auto typeOfTensor = op.tensor()->getType().cast<TensorType>();
  // No compile time verification for unknown sized shape.
  if (rankByShape == -1 || !typeOfTensor.hasRank()) return success();
  // Check values if constant shape. No compiling time verification for
  // non-constant shape.
  auto *shapeOp = op.shape()->getDefiningOp();
  if (!shapeOp) return success();
  Attribute shapeCst;
  if (auto shapeStdOp = dyn_cast<ConstantOp>(shapeOp)) {
    shapeCst = shapeStdOp.getValue();
  } else if (auto shapeTFOp = dyn_cast<ConstOp>(shapeOp)) {
    shapeCst = shapeTFOp.value();
  } else {
    return success();
  }
  auto shapeCstAttr = shapeCst.dyn_cast<ElementsAttr>();
  if (!shapeCstAttr) return op.emitOpError("shape must be a valid tensor");

  if (auto opaqueAttr = shapeCstAttr.dyn_cast<OpaqueElementsAttr>()) {
    opaqueAttr.decode(shapeCstAttr);
  }

  // We know the shape is a 1-D Tensor, then let us get the number of
  // elements it implies.
  unsigned numByShape = 1;
  unsigned unknownDimCount = 0;
  for (int i = 0, e = rankByShape; i != e; ++i) {
    auto num = shapeCstAttr.getValue(i).cast<IntegerAttr>().getInt();
    // The dimension size value can be -1, and that the real size needs to
    // be computed so that the total size remains constant. At most one
    // component of shape can be -1.
    if (num == -1) {
      if (++unknownDimCount > 1) {
        return op.emitOpError("more than one component of shape are -1");
      }
    } else {
      numByShape *= num;
    }
  }
  auto numByTensor = typeOfTensor.getNumElements();
  // If there is one component of shape is -1, the dimension should be
  // computed so that the total size remains constant.
  if (unknownDimCount == 1) {
    if (numByTensor % numByShape != 0)
      return op.emitOpError(
          "one component of shape is -1 but couldn't infer the dimension");
    return success();
  }
  // If the elements by the tensor and implies by the shape don't match,
  // fail this static check.
  if (numByTensor != numByShape) {
    return op.emitOpError(
        "mismatch in tensor elements and shape implied elements");
  }
  return success();
}

void ReshapeOp::build(Builder *builder, OperationState *result, Value *tensor,
                      Value *shape) {
  auto etype = tensor->getType().cast<ShapedType>().getElementType();
  DenseIntElementsAttr attr_shape;
  if (matchPattern(shape, m_Constant(&attr_shape))) {
    llvm::SmallVector<int64_t, 4> const_shape;
    if (attr_shape.isSplat()) {
      const_shape.assign(attr_shape.getType().getNumElements(),
                         (*attr_shape.begin()).getSExtValue());
    } else {
      const_shape.reserve(attr_shape.getType().getNumElements());
      for (auto dim : attr_shape) const_shape.push_back(dim.getSExtValue());
    }
    return ReshapeOp::build(builder, result,
                            builder->getTensorType(const_shape, etype), tensor,
                            shape);
  }
  return ReshapeOp::build(builder, result, builder->getTensorType(etype),
                          tensor, shape);
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ShapeOp op) {
  auto inputType = op.input()->getType();
  auto resultType = op.getType().dyn_cast<RankedTensorType>();
  if (!resultType || resultType.getShape().size() != 1)
    return op.emitOpError("requires 1D result type");

  auto rankedTensorType = inputType.dyn_cast<RankedTensorType>();
  if (rankedTensorType) {
    // The operand is a ranked tensor.
    if (resultType.hasStaticShape()) {
      if ((!rankedTensorType.getShape().empty() &&
           resultType.getDimSize(0) != rankedTensorType.getShape().size()))
        return op.emitOpError(
            "requires dimension size of result to match rank of operand");
    }
  } else {
    // The operand is an unranked tensor, verify that the result is dynamic.
    if (resultType.hasStaticShape())
      return op.emitOpError("requires dynamic shape result for unranked input");
  }

  Type elt = op.getType().cast<ShapedType>().getElementType();
  if (elt.isInteger(32) || elt.isInteger(64)) return success();
  return op.emitOpError("requires int32 or int64 return type");
}

OpFoldResult ShapeOp::fold(ArrayRef<Attribute> operands) {
  auto inputType = getOperand()->getType();
  auto rankedTensorType = inputType.dyn_cast<RankedTensorType>();
  if (!rankedTensorType || !rankedTensorType.hasStaticShape()) return {};

  auto shape = rankedTensorType.getShape();
  int rank = shape.size();

  Builder b(getContext());
  auto elementType = getType().cast<ShapedType>().getElementType();

  SmallVector<Attribute, 4> dimensions;
  dimensions.reserve(rank);
  for (int i = 0; i < rank; ++i)
    dimensions.push_back(b.getIntegerAttr(elementType, shape[i]));

  auto resultType = b.getTensorType({rank}, elementType);
  return b.getDenseElementsAttr(resultType, dimensions);
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(SoftmaxOp op) {
  if (!IsOfRankOrUnranked(op.logits(), 2))
    return op.emitOpError("requires operand to be 2D tensor");

  return success();
}

//===----------------------------------------------------------------------===//
// SquareOp
//===----------------------------------------------------------------------===//

void SquareOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  RewriteListBuilder<SquareOfSub>::build(results, context);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

void SubOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  RewriteListBuilder<SubOfNeg>::build(results, context);
}

//===----------------------------------------------------------------------===//
// TensorListReserveOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(TensorListReserveOp op) {
  if (!IsOfRankOrUnranked(op.element_shape(), 0) &&
      !IsOfRankOrUnranked(op.element_shape(), 1)) {
    return op.emitOpError("requires element_shape operand to be 0D/1D tensor");
  }

  if (!IsOfRankOrUnranked(op.num_elements(), 0)) {
    return op.emitOpError("requires num_elements operand to be 0D tensor");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(TransposeOp op) {
  // TODO(hinsu): Verify using a custom verifier that,
  // * Transpose permutation is 1-D of size equal to the rank of the first
  //   input, if the shapes are partially known. Requires use of a more
  //   restrictive type than TF_Tensor.
  // * Result shape dimensions are possible based on the input shape.
  return success();
}

// TODO(jpienaar): perm could be optional too.
void TransposeOp::build(Builder *builder, OperationState *result, Value *x,
                        Value *perm) {
  auto x_type = x->getType().cast<TensorType>();
  // If value is unranked, then so is results.
  if (!x_type.hasRank())
    return TransposeOp::build(builder, result,
                              builder->getTensorType(x_type.getElementType()),
                              x, perm);

  // TODO(jpienaar): Handle unknown perm case.

  // TODO(jpienaar): Extract utility function.
  auto etype = x_type.cast<ShapedType>().getElementType();
  DenseIntElementsAttr attr_shape;
  if (matchPattern(perm, m_Constant(&attr_shape))) {
    llvm::SmallVector<int64_t, 4> const_shape;
    if (attr_shape.isSplat()) {
      const_shape.assign(
          attr_shape.getType().getNumElements(),
          x_type.getDimSize((*attr_shape.begin()).getSExtValue()));
    } else {
      const_shape.reserve(attr_shape.getType().getNumElements());
      for (auto dim : attr_shape)
        const_shape.push_back(x_type.getDimSize(dim.getSExtValue()));
    }
    return TransposeOp::build(
        builder, result, builder->getTensorType(const_shape, etype), x, perm);
  }
  return TransposeOp::build(builder, result, builder->getTensorType(etype), x,
                            perm);
}

//===----------------------------------------------------------------------===//
// TruncateDivOp
//===----------------------------------------------------------------------===//

void TruncateDivOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  RewriteListBuilder<TruncateDivWithSqrtDivisor>::build(results, context);
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::verify() {
  auto condAttr = getAttrOfType<FunctionAttr>("cond");
  if (!condAttr) return emitOpError("requires cond attribute");

  auto module = getParentOfType<Module>();
  auto condFn = module.getNamedFunction(condAttr.getValue());
  auto condFuncType = condFn.getType();

  // Verify that the cond function has exactly one result.
  if (condFuncType.getNumResults() != 1)
    return emitOpError("requires cond function to have exactly one result");

  auto bodyAttr = getAttrOfType<FunctionAttr>("body");
  if (!bodyAttr) return emitOpError("requires body attribute");
  auto bodyFn = module.getNamedFunction(bodyAttr.getValue());
  auto bodyFuncType = bodyFn.getType();

  SmallVector<Type, 4> operands(getOperandTypes());
  SmallVector<Type, 4> results(getResultTypes());

  // Collect all the type lists for the op so that different pairs of type lists
  // can be compared for the compatibility.
  int numTypeLists = 5;
  std::pair<std::string, ArrayRef<Type>> typeLists[] = {
      {"operand", operands},
      {"body function result", bodyFuncType.getResults()},
      {"result", results},
      {"cond function input", condFuncType.getInputs()},
      {"body function input", bodyFuncType.getInputs()},
  };

  // A pair of type lists should be cast compatible with each other if one is
  // converted to the another for a function call or assignment or there is a
  // common source of inputs for both.  Therefore, the While op requires the
  // following pairs of type lists to be cast compatible for the tensor_cast
  // operation:
  //
  // * Operands and cond inputs to call the cond function before the
  //   first iteration.
  // * Operands and body inputs to call the body function for the first
  //   iteration if the cond functions returns True or equivalent result.
  // * Operands and results to assign cond function arguments to op results if
  //   the cond function returns False or equivalent result.
  // * All three pairs using cond inputs, body inputs and results as operand is
  //   a common source for all three.
  // * Body result and cond inputs to call the cond function for the subsequent
  //   iterations. Similarly, Body result should be compatible with body inputs
  //   and op results.
  //
  // Note that the operands and body results need not be compatible as they are
  // never converted from one to the another nor there is a common source
  // tensors.  Compatibility requirement is not transitive.

  for (int i = 0; i < numTypeLists; ++i) {
    // Skip the first pair as the While op operands and body function results
    // does not need to be compatible with each other.
    for (int j = std::max(2, i + 1); j < numTypeLists; ++j) {
      auto &a = typeLists[i];
      auto &b = typeLists[j];

      int aSize = a.second.size();
      if (aSize != b.second.size())
        return emitOpError(
            llvm::formatv("requires the number of {0}s to be equal to the "
                          "number of {1}s. Found {2} and {3}, respectively",
                          a.first, b.first, aSize, b.second.size()));

      for (int idx = 0; idx < aSize; ++idx) {
        auto aType = a.second[idx];
        auto bType = b.second[idx];

        if (!TensorCastOp::areCastCompatible(aType, bType))
          return emitError(llvm::formatv(
              "{0} type {1} is incompatible with {2} type {3} at index {4}",
              a.first, aType, b.first, bType, idx));
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XdivyOp
//===----------------------------------------------------------------------===//

void XdivyOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  RewriteListBuilder<XdivyWithSqrtDivisor>::build(results, context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc.inc"

//===----------------------------------------------------------------------===//
// TF Dialect
//===----------------------------------------------------------------------===//

TensorFlowDialect::TensorFlowDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf", context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc.inc"
      , IfOp, WhileOp>();
  addTypes<
#define HANDLE_TF_TYPE(tftype, enumerant, name) tftype##Type,
#define HANDLE_LAST_TF_TYPE(tftype, enumerant, name) tftype##Type
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
      >();

  // Support unknown operations because not all TensorFlow operations are
  // registered.
  allowUnknownOperations();
}

// Parses a type registered to this dialect.
Type TensorFlowDialect::parseType(StringRef data, Location loc) const {
  auto typeKind = llvm::StringSwitch<unsigned>(data)
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  .Case(name, TensorFlowTypes::enumerant)
// Custom TensorFlow types are handled separately at the end as they do partial
// match.
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
                      .StartsWith("variant", TensorFlowTypes::VARIANT)
                      .Default(0);
  switch (typeKind) {
    default:
      return (emitError(loc, "unknown TensorFlow type: " + data), nullptr);

#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  case TensorFlowTypes::enumerant:              \
    return tftype##Type::get(getContext());
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
    case TensorFlowTypes::VARIANT:
      return ParseVariantType(data, loc);
  }
}

// Prints a type registered to this dialect.
void TensorFlowDialect::printType(Type ty, raw_ostream &os) const {
  assert(ty.isa<TensorFlowType>());
  switch (ty.getKind()) {
    default:
      llvm_unreachable("unexpected tensorflow type kind");
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  case TensorFlowTypes::enumerant:              \
    os << name;                                 \
    break;
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name) \
  case TensorFlowTypes::enumerant:                     \
    Print##tftype##Type(ty.cast<tftype##Type>(), os);  \
    break;
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
  }
}

Type TensorFlowDialect::ParseVariantType(StringRef spec, Location loc) const {
  bool success = spec.consume_front("variant");
  DCHECK(success) << spec.str();

  // Default variant type without inferred subtypes.
  MLIRContext *context = getContext();
  if (spec.empty()) return VariantType::get(context);

  if (!spec.consume_front("<") || !spec.consume_back(">"))
    return emitError(loc) << "tf.variant delimiter <...> mismatch", nullptr;

  // Most variant types with subtypes have only one subtype.
  SmallVector<StringRef, 1> subtype_specs;
  llvm::SplitString(spec, subtype_specs, ",");
  if (subtype_specs.empty())
    return emitError(loc) << "invalid type: tf.variant<>", nullptr;

  SmallVector<TensorType, 1> subtypes;
  subtypes.reserve(subtype_specs.size());
  for (StringRef subtype_spec : subtype_specs) {
    subtype_spec = subtype_spec.trim();
    Type type = mlir::parseType(subtype_spec, context);
    if (!type) {
      return emitError(loc) << "invalid type: " << subtype_spec, nullptr;
    }

    if (TensorType tensor_ty = type.dyn_cast<TensorType>()) {
      subtypes.push_back(tensor_ty);
    } else {
      return emitError(loc) << "expected TensorType. Found: " << type, nullptr;
    }
  }
  return VariantType::getChecked(subtypes, context, loc);
}

void TensorFlowDialect::PrintVariantType(VariantType ty,
                                         raw_ostream &os) const {
  os << "variant";
  ArrayRef<TensorType> subtypes = ty.getSubtypes();
  if (subtypes.empty()) return;

  os << "<";
  interleaveComma(subtypes, os);
  os << ">";
}

Operation *TensorFlowDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  // If this is an opaque elements attribute or the result type doesn't match
  // the attribute type, then generate a tf.Const.
  if (value.isa<OpaqueElementsAttr>() || value.getType() != type)
    return builder.create<ConstOp>(loc, type, value);
  return nullptr;
}

// Verifies that the Op is a well-formed TensorFlow op, checking that all inputs
// and results are Tensor or other TensorFlow types, etc.
LogicalResult verifyTensorFlowOp(Operation *op) {
  if (op->getName().getDialect() != "tf")
    return op->emitError("TensorFlow op ")
           << op->getName() << " should start with 'tf.'";

  for (Type type : op->getOperandTypes()) {
    if (!IsValidTFTensorType(type))
      return op->emitOpError(
          "requires operands to have a valid TensorFlow tensor type");
  }

  for (Type type : op->getResultTypes()) {
    if (!IsValidTFTensorType(type))
      return op->emitOpError(
          "requires results to have a valid TensorFlow tensor type");
  }

  return success();
}

}  // namespace TF
}  // namespace mlir
