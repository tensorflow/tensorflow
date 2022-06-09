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

// This file defines the operations used in the MHLO dialect.

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h.inc"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_common.h"
#include "mlir-hlo/utils/convert_op_folder.h"
#include "mlir-hlo/utils/hlo_utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
#include "hlo_patterns.cc.inc"
}  // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.cc.inc"

namespace mlir {
namespace mhlo {
namespace {
void createArgs(ArrayRef<OpAsmParser::UnresolvedOperand> operands,
                ArrayRef<Type> types,
                SmallVector<OpAsmParser::Argument>& args) {
  for (auto argAndType : llvm::zip(operands, types)) {
    auto& arg = args.emplace_back();
    arg.ssaName = std::get<0>(argAndType);
    arg.type = std::get<1>(argAndType);
  }
}

//===----------------------------------------------------------------------===//
// Utilities for the canonicalize patterns
//===----------------------------------------------------------------------===//

// This is an arbitrary limit into how many elements can a splat attribute
// covers before we prevent folding from happening. Without such limit we can
// expand a single element splat to a multi-GB large tensor.
// The limit is arbitrary set low to allow expanding small computations, like
// shape manipulations for example.
// TODO(b/210478841): Define a constant folding policy that generalizes this.
constexpr int64_t kFoldExpandSplatEltLimit = 16;

// Similarly to the constant above, this is an arbitrary limit into how many
// elements can be folded by a binary operation folder.
// This limit doesn't apply to the following special cases:
//   1) Adding a zero.
//   2) Multiplying by one.
//   3) When both operands are splats.
// TODO(b/210478841): Define a constant folding policy that generalizes this.
constexpr int64_t kFoldBinaryOpEltLimit = 65536;

// Clamps value to the range [lower, upper].  Requires lower <= upper.
template <typename T>
static T clamp(const T& value, const T& lower, const T& upper) {
  assert(lower <= upper);
  return std::max(lower, std::min(value, upper));
}

// Verifies that dimension attribute for the op correctly indexes in operand or
// result shape.
template <typename OpT>
static LogicalResult verifyDimAttr(OpT op) {
  int64_t rank = -1;
  if (auto ty = op.operand().getType().template dyn_cast<RankedTensorType>()) {
    rank = ty.getRank();
  } else if (auto ty = op.getType().template dyn_cast<RankedTensorType>()) {
    rank = ty.getRank();
  } else {
    return success();
  }

  int64_t dim = op.dimension();
  if (dim < 0 || dim >= rank)
    return op.emitOpError() << "requires dimension attribute in range [0, "
                            << rank << "); found (" << dim << ")";
  return success();
}

// Given the start indices and slice sizes for a dynamic-slice that can be
// converted to a static slice, returns the limits for the static slice.
DenseIntElementsAttr buildSliceLimits(DenseIntElementsAttr startIndices,
                                      DenseIntElementsAttr sliceSizes,
                                      Builder* builder) {
  SmallVector<int64_t, 4> sliceLimits;
  for (int64_t i = 0; i < sliceSizes.getNumElements(); ++i) {
    int64_t startIndex = startIndices.getValues<IntegerAttr>()[i].getInt();
    int64_t sliceSize = sliceSizes.getValues<IntegerAttr>()[i].getInt();
    sliceLimits.push_back(startIndex + sliceSize);
  }
  return builder->getI64TensorAttr(sliceLimits);
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter& rewriter, Operation* op,
                                Region& region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-block region");
  Block* block = &region.front();
  Operation* terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.mergeBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

#include "mhlo_canonicalize.inc"

// Check if the dimension size is dynamic.
inline static bool isDynamicDimSize(int64_t val) {
  return val == ShapedType::kDynamicSize;
}

// Common shape function helper for RngNormal and RngUniform.
static LogicalResult rngInferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (operands.size() != 3)
    return emitOptionalError(location, "expected 3 operands");

  SmallVector<int64_t> shapeVector;
  Value shapeOperand = operands[2];
  auto shapeOperandType = shapeOperand.getType().cast<ShapedType>();
  Type elementType = getElementTypeOrSelf(operands[1]);

  // Match constant shape arguments.
  DenseIntElementsAttr shape;
  if (!matchPattern(shapeOperand, m_Constant(&shape))) {
    if (!shapeOperandType.hasRank()) {
      inferredReturnShapes.emplace_back(elementType);
      return success();
    }
    if (shapeOperandType.getRank() != 1)
      return emitOptionalError(location, "shape operand required to be 1D");
    int size = shapeOperandType.getDimSize(0);
    if (isDynamicDimSize(size)) {
      inferredReturnShapes.emplace_back(elementType);
      return success();
    }
    shapeVector.resize(size, ShapedType::kDynamicSize);
    inferredReturnShapes.emplace_back(shapeVector, elementType);
    return success();
  }

  shapeVector.reserve(shape.size());
  for (const APInt& fp : shape.getValues<APInt>())
    shapeVector.push_back(fp.getSExtValue());
  inferredReturnShapes.emplace_back(shapeVector, elementType);
  return success();
}

// Returns a new scalar integer value having type `type`. Here `type` must be
// an integer or index type.
Value maybeCastTo(OpBuilder& b, Location loc, Value value, Type type) {
  if (type == value.getType()) return value;
  assert(type.isIndex() || value.getType().isIndex());
  return b.create<arith::IndexCastOp>(loc, type, value);
}

DenseElementsAttr reshape(DenseElementsAttr attr, ShapedType newType) {
  // TODO(b/232866626): DenseElementsAttr::reshape is broken for bool splats.
  // Once that ticket is fixed, we can remove this conditional.
  if (attr.isSplat() && newType.getElementType().isInteger(/*width=*/1)) {
    auto splatValue = attr.getValues<bool>()[0];
    return DenseElementsAttr::get(newType, {splatValue});
  }
  return attr.reshape(newType);
}

//===----------------------------------------------------------------------===//
// Utilities for verifiers
//===----------------------------------------------------------------------===//

// Convert a 1D dense int64 attribute to a list of values.
SmallVector<int64_t> convertDenseIntAttr(
    llvm::Optional<mlir::DenseIntElementsAttr> optionalAttr) {
  if (!optionalAttr.hasValue()) return SmallVector<int64_t>{};

  mlir::DenseIntElementsAttr attr = *optionalAttr;
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

// Convert a 1D or Nx2 dense int64 attribute to a list of tuples.
FailureOr<SmallVector<std::pair<int64_t, int64_t>>> convertNx2Attribute(
    llvm::Optional<mlir::DenseIntElementsAttr> optionalAttr, Location loc) {
  if (!optionalAttr.hasValue())
    return SmallVector<std::pair<int64_t, int64_t>>{};
  mlir::DenseIntElementsAttr attr = *optionalAttr;

  auto attrType = attr.getType().cast<RankedTensorType>();  // ensured by ODS.
  if (attrType.getRank() > 1) {
    if (attrType.getRank() != 2 || attrType.getShape()[1] != 2)
      return (mlir::emitError(loc) << "expects the shape of padding-attribute "
                                      "to be {N, 2}, but got {"
                                   << attrType.getShape() << "}.",
              failure());
  } else {
    // Padding values can be provided as a 1D vector as well.
    if (attr.getValues<int64_t>().size() % 2 != 0)
      return (mlir::emitError(loc)
                  << "expects the padding-entries to have even number of "
                     "elements, but got "
                  << attr.getValues<int64_t>().size() << " elements.",
              failure());
  }

  auto it = attr.getValues<int64_t>().begin();
  SmallVector<std::pair<int64_t, int64_t>> out(attr.getNumElements() / 2);
  for (auto& item : out) {
    int64_t first = *it;
    ++it;
    int64_t second = *it;
    ++it;
    item = {first, second};
  }
  return out;
}

// If a window with the given bound in some dimension is dilated with the given
// dilation factor in that dimension, then the value returned is the bound for
// the array in that dimension after dilation.
//
// For a 1D array with 3 entries 1, 2, 3, a dilation factor of 2 yields a new
// window with values 1, x, 2, x, 3, where x indicates holes left by the
// dilation. So DilatedBound(3, 2) == 5.
int64_t dilatedBound(int64_t bound, int64_t dilation) {
  assert(bound >= 0 && "The dimension to dialate must be >= 0");
  if (bound == 0) return 0;

  // Suppose the array has three entries 123 and the dilation factor is 4. Then
  // the dilated array has 9 entries 1xxx2xxx3. Here, each original entry except
  // the last expands into 4 entries, so that is (bound - 1) * dilation. Then we
  // add 1 to account for the final input element.
  return (bound - 1) * dilation + 1;
}

// Returns the number of valid positions of a window with the given size and
// stride within an array with the given bound. This is the bound of an output
// array with one element per valid position of the window.
//
// For example, for arguments of (bound=5, window_size=2, stride=2), the
// returned value is 2. There are valid positions at offset 0 and offset 2,
// while offset 4 is not valid since the window's last entry would be at 5,
// which is beyond the bound of 5.
int64_t stridedBound(int64_t bound, int64_t windowSize, int64_t stride) {
  assert(windowSize >= 0 && "Expected window size to be >= 0");
  assert(bound >= 0 && "Expected bound to be >= 0");

  if (bound == 0 || windowSize > bound) return 0;

  // Without considering stride, the maximum valid offset is bound -
  // window_size. Taking stride into account, the valid offsets then have the
  // form q * stride for q = 0, ..., Q such that q * stride <= bound -
  // window_size. This implies that Q equals floor(bound - window_size /
  // stride). There are Q + 1 valid values of q, yielding the formula below.
  return (bound - windowSize) / stride + 1;
}

// WindowDimension described how the kernel window moves across the base area
// in a particular dimension.
// Describes the windowing in an operation such as convolution.
// The window is moved across a base area and for each position of the
// window a computation is performed. The field below describes the
// window and the movement of the window across a base area.
struct WindowDimension {
  int64_t size = 0;
  int64_t stride = 1;
  int64_t paddingLow = 0;
  int64_t paddingHigh = 0;
  int64_t windowDilation = 1;
  int64_t baseDilation = 1;
  bool windowReversal = false;
};

// Verifies various properties of window-attributes (viz., stride, padding,
// lhs_dilation and rhs_dilation) and collects all the window-attributes for
// each kernel spatial dimensions.
FailureOr<SmallVector<WindowDimension>>
verifyWindowAttributesAndInferWindowDimensions(
    ArrayRef<int64_t> windowDimensions, ArrayRef<int64_t> windowStrides,
    ArrayRef<std::pair<int64_t, int64_t>> padding,
    ArrayRef<int64_t> lhsDilation, ArrayRef<int64_t> rhsDilation,
    Location loc) {
  const auto verifySize = [&](const size_t attrSize,
                              StringRef attrName) -> LogicalResult {
    if (attrSize == 0 || attrSize == windowDimensions.size()) return success();
    return mlir::emitError(loc)
           << "expects " << attrName
           << " to have same dimension-size as size of "
              "window dimensions "
              "("
           << windowDimensions.size() << "), but got: " << attrSize << ".";
  };

  if (failed(verifySize(windowStrides.size(), "window-strides")))
    return failure();
  if (failed(verifySize(lhsDilation.size(), "base-dilation factors")))
    return failure();
  if (failed(verifySize(rhsDilation.size(), "window-dilation factors")))
    return failure();
  if (failed(verifySize(padding.size(), "padding-entries"))) return failure();

  SmallVector<WindowDimension> window(windowDimensions.size());
  for (size_t i = 0; i < windowDimensions.size(); i++) {
    WindowDimension& dim = window[i];

    dim.size = windowDimensions[i];
    if (!isDynamicDimSize(dim.size) && dim.size <= 0)
      return (mlir::emitError(loc)
                  << "expects window to have positive value for " << i
                  << "-th window dimension, but got " << dim.size << ".",
              failure());

    if (!windowStrides.empty()) dim.stride = windowStrides[i];
    if (dim.stride <= 0)
      return (mlir::emitError(loc)
                  << "expects window to have positive stride for " << i
                  << "-th window dimension, but got " << dim.stride << ".",
              failure());

    if (!lhsDilation.empty()) dim.baseDilation = lhsDilation[i];
    if (dim.baseDilation <= 0)
      return (mlir::emitError(loc) << "expects window to have positive base "
                                      "dilation factor for "
                                   << i << "-th window dimension, but got "
                                   << dim.baseDilation << ".",
              failure());

    if (!rhsDilation.empty()) dim.windowDilation = rhsDilation[i];
    if (dim.windowDilation <= 0)
      return (mlir::emitError(loc) << "expects window to have positive window "
                                      "dilation factor for "
                                   << i << "-th window dimension, but got "
                                   << dim.windowDilation << ".",
              failure());

    if (!padding.empty()) {
      dim.paddingLow = padding[i].first;
      dim.paddingHigh = padding[i].second;
    }
  }

  return window;
}

// Infer the shape of the output window.
//  Foreach dimension d,
//    output-window-shape[d] =
//            stridedBound(padding_low + dilatedBound(base_shape[d]) +
//            padding_high,
//                         dilatedBound(window_shape[d]))
//      where (padding_low, padding_high) is the padding-pair for d.
SmallVector<int64_t> inferWindowOutputShape(
    const ArrayRef<int64_t> baseShape, const ArrayRef<WindowDimension> window) {
  assert(baseShape.size() == window.size() &&
         "Size of window dimensions must match the size of base shape.");

  SmallVector<int64_t> outputDimensions(window.size());
  for (int64_t i = 0; i < window.size(); ++i) {
    if (isDynamicDimSize(baseShape[i]) || isDynamicDimSize(window[i].size)) {
      outputDimensions[i] = ShapedType::kDynamicSize;
    } else {
      const auto& dim = window[i];

      const int64_t dilatedBase = dilatedBound(baseShape[i], dim.baseDilation);
      const int64_t paddedDilatedBase =
          dim.paddingLow + dilatedBase + dim.paddingHigh;
      const int64_t dilatedWindow = dilatedBound(dim.size, dim.windowDilation);

      outputDimensions[i] =
          stridedBound(paddedDilatedBase, dilatedWindow, dim.stride);
    }
  }

  return outputDimensions;
}

// Return true if type1 and type2 are tensors and have the same
// element-type, else return false. With float element-types, ignore comparing
// floating-point precision if ignoreFpPrecision is True.
bool tensorsHaveSameElType(Type type1, Type type2, bool ignoreFpPrecision) {
  auto tensorTy1 = type1.dyn_cast<TensorType>();
  auto tensorTy2 = type2.dyn_cast<TensorType>();

  if (!tensorTy1 || !tensorTy2) return false;

  if (ignoreFpPrecision && tensorTy1.getElementType().isa<FloatType>() &&
      tensorTy2.getElementType().isa<FloatType>())
    return true;

  return tensorTy1.getElementType() == tensorTy2.getElementType();
}

// Return true if type1 and type2 are shape-compatible and have same element
// type. If 'ignoreFpPrecision' is True, then allow floats with different
// precisions while checking element-types.
bool compatibleShapeAndElementType(Type type1, Type type2,
                                   bool ignoreFpPrecision = false) {
  if (failed(verifyCompatibleShape(type1, type2))) return false;
  return tensorsHaveSameElType(type1.cast<ShapedType>(),
                               type2.cast<ShapedType>(), ignoreFpPrecision);
}

LogicalResult verifyReducerShape(
    Location loc, Block& block, ArrayRef<TensorType> inputArgTypes,
    ArrayRef<TensorType> initValueTypes, int64_t numInputs,
    ArrayRef<int64_t> allowedDimensions, bool allInputsUnranked,
    SmallVectorImpl<TensorType>& accumulatorSubShapes) {
  // Check that the number of reduction-region arguments matches with that of
  // reduce-op's arguments.
  if (block.getArguments().size() != numInputs * 2)
    return mlir::emitError(loc)
           << "Reduction-region must take " << numInputs * 2
           << " parameters, but takes " << block.getArguments().size()
           << " parameter(s)";

  // Check if the reduction-region produces non-zero outputs.
  if (block.getTerminator()->getOperands().empty())
    return mlir::emitError(loc)
           << "The reduction-region expected to return some value(s)";

  // Check that the reduction-region returns list- of tensors.
  // The number of result-tensors must match the `numInputs`.
  if (block.getTerminator()->getOperands().size() != numInputs)
    return mlir::emitError(loc)
           << "Reduction-region here must produce " << numInputs
           << " tensors, but produces "
           << block.getTerminator()->getOperands().size() << " instead";

  for (Value retOperand : block.getTerminator()->getOperands()) {
    auto tensorTy = retOperand.getType().dyn_cast<TensorType>();
    if (!tensorTy)
      return mlir::emitError(loc) << "Reduction-region here must produce "
                                     "tensor-typed result(s), but "
                                     "produces "
                                  << retOperand.getType() << " instead";

    accumulatorSubShapes.push_back(tensorTy);
  }

  // Consider typical reduce-* op syntax:
  //
  //      op(I(i), V(j)):
  //       block(BI(i), BV(j)):
  //         ... some computation ...
  //         return(R(i))
  //
  // where
  //  I(i)  : i-th input of op
  //  V(j)  : j-th init-value of op
  //  BI(i) : i-th input of reducer-function
  //  BV(j) : j-th init-value of reducer-function
  //  R(i)  : i-th return-type
  //
  //  Note that: |I(i)| == V(j)| == |BI(i)| == |BV(j)| == |R(i)|
  //
  //  Here are the type-constraints among V(j), BI(i), BV(j), and R(i).
  //    C1 : Check that BI(i) and R(i) have same shape and element-type.
  //    C2 : Check that BV(j) and R(i) have same shape and element-type.
  //    C3 : Check that V(j) and R(i) have same shape and element-type.
  //
  //  From C1, C2, and C3, we can infer that V(j), BI(i), BV(j), and R(i) all
  //  have compatible shapes and element-types.
  //  The next check, C4, adds constraints on how the type if I(i) is related
  //  to any_of(V(j), BI(i), BV(j), and R(i)), say BV(j);
  //
  //  C4.1 : Check that I(i) and BV(j) have same element-type.
  //  C4.2 : Check that shape of BV(j) is a 'sub-sequence' of
  //         'allowedDimensions'. 'allowedDimensions' is a list of dimensions
  //         which any of BI(i), BV(j), and R(i) is allowed to have.
  for (int64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    // Check C1.
    if (!compatibleShapeAndElementType(accumulatorSubShapes[inputIdx],
                                       block.getArgument(inputIdx).getType()))
      return mlir::emitError(loc)
             << "The type of reduction-region's parameter at index " << inputIdx
             << " is different than the corresponding result type: "
             << block.getArgument(inputIdx).getType() << " vs "
             << accumulatorSubShapes[inputIdx];

    // Check C2.
    if (!compatibleShapeAndElementType(
            accumulatorSubShapes[inputIdx],
            block.getArgument(numInputs + inputIdx).getType(),
            /*ignoreFpPrecision=*/true))
      return mlir::emitError(loc)
             << "The type of reduction-region's parameter at index "
             << numInputs + inputIdx
             << " is different than the corresponding result type: "
             << block.getArgument(numInputs + inputIdx).getType() << " vs "
             << accumulatorSubShapes[inputIdx];

    // Check C3.
    if (!compatibleShapeAndElementType(accumulatorSubShapes[inputIdx],
                                       initValueTypes[inputIdx],
                                       /*ignoreFpPrecision=*/true))
      return mlir::emitError(loc)
             << "The type of reduction-region's result type at index "
             << inputIdx
             << " differs from the op's corresponding init-value type: "
             << accumulatorSubShapes[inputIdx] << " vs "
             << initValueTypes[inputIdx];

    // Check C4.1.
    if (!tensorsHaveSameElType(
            inputArgTypes[inputIdx],
            block.getArgument(numInputs + inputIdx).getType(), true))
      return mlir::emitError(loc)
             << "The element-type of reduction-region's argument at index "
             << numInputs + inputIdx << " is expected to be "
             << inputArgTypes[inputIdx].getElementType() << ", but got "
             << block.getArgument(numInputs + inputIdx).getType()
             << " as its type.";

    // Check C4.2.
    Type blockArgType = block.getArgument(numInputs + inputIdx).getType();
    auto blockArgTensorTy = blockArgType.cast<TensorType>();

    if (allInputsUnranked || !blockArgTensorTy.hasRank()) return success();

    auto argShape = blockArgTensorTy.getShape();
    if (argShape.size() > allowedDimensions.size())
      return mlir::emitError(loc)
             << "The rank of reduction-region's argument at index "
             << numInputs + inputIdx
             << " is expected to be <= " << allowedDimensions.size() << ", got "
             << argShape.size();

    int64_t argShapeIdx = 0;
    for (int64_t outputShapeIdx = 0;
         outputShapeIdx < allowedDimensions.size() &&
         argShapeIdx < argShape.size();
         outputShapeIdx++)
      if (allowedDimensions[outputShapeIdx] == argShape[argShapeIdx])
        argShapeIdx++;

    if (argShapeIdx != argShape.size())
      return mlir::emitError(loc)
             << "The shape of reduction-region's argument at index "
             << numInputs + inputIdx
             << " is not compatible with that of reduce-op's input-parameter "
                "at index "
             << inputIdx;
  }

  return success();
}

unsigned potentiallyComplexBitwidth(Type type) {
  auto complexTy = type.dyn_cast<ComplexType>();
  return complexTy ? 2 * complexTy.getElementType().getIntOrFloatBitWidth()
                   : type.getIntOrFloatBitWidth();
}
}  // namespace

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceScatterOp::verify() {
  if (failed(mlir::hlo::VerifyReplicaGroups(*this, /*is_uniform_sized=*/true)))
    return failure();
  auto operandType = operand().getType().cast<TensorType>();
  bool operandTypeRanked = operandType.isa<RankedTensorType>();
  Block& block = computation().front();
  SmallVector<TensorType> accumulatorSubshapes;
  if (failed(verifyReducerShape(
          this->getLoc(), block, {operandType},
          {RankedTensorType::get({}, operandType.getElementType())},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/!operandTypeRanked, accumulatorSubshapes)))
    return failure();

  return mlir::hlo::VerifyReduceScatter(
      *this,
      /*operand_types=*/{operand().getType()},
      /*result_types=*/{getType()},
      /*scatter_dimension=*/scatter_dimension());
}

//===----------------------------------------------------------------------===//
// CompatibleOperandsAndResultType
//===----------------------------------------------------------------------===//

// TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
// support quantization or sparsity.
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                        \
  LogicalResult Op::inferReturnTypeComponents(                                \
      MLIRContext* context, Optional<Location> location,                      \
      ValueShapeRange operands, DictionaryAttr attributes,                    \
      RegionRange regions,                                                    \
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {          \
    return inferReturnTypeComponentsFromOperands(context, location, operands, \
                                                 attributes, regions,         \
                                                 inferredReturnShapes);       \
  }

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AddOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AllReduceOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AndOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Atan2Op)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CbrtOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CeilOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ClzOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CollectivePermuteOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CopyOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CosOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CrossReplicaSumOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DivOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ExpOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Expm1Op)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(FloorOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(LogOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Log1pOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(LogisticOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MaxOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MinOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MulOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NegOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NotOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(OrOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PopulationCountOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PowOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReducePrecisionOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RemOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReverseOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RoundOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RsqrtOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftLeftOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightArithmeticOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightLogicalOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SignOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SinOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SqrtOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SubOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanhOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(XorOp)

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

// Builds a constant op with the specified attribute `value`.
void ConstOp::build(OpBuilder& builder, OperationState& result,
                    Attribute value) {
  Type type;
  if (auto elemAttr = value.dyn_cast<ElementsAttr>()) {
    type = elemAttr.getType();
  } else if (value.isa<BoolAttr>() || value.isa<FloatAttr>() ||
             value.isa<IntegerAttr>()) {
    // All XLA types must be tensor types. In the build() method, we want to
    // provide more flexibility by allowing attributes of scalar types. But we
    // need to wrap it up with ElementsAttr to construct valid XLA constants.
    type = RankedTensorType::get(/*shape=*/{}, value.getType());
    value = DenseElementsAttr::get(type.cast<TensorType>(), value);
  }

  // TODO: support other XLA specific types.
  assert(type && "unsupported attribute type for building mhlo.constant");
  result.types.push_back(type);
  result.addAttribute("value", value);
}

LogicalResult ConstOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange, DictionaryAttr attributes,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  Type type = attributes.get("value").getType();
  inferredReturnTypes.push_back(type);
  return success();
}

bool ConstOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1) return false;
  auto lhsTy = l.front().cast<TensorType>();
  auto rhsTy = r.front().cast<TensorType>();
  // For comparisons of the uniform quantized element based tensor type, use the
  // storage type since the constant value will be stored through the underlying
  // storage type.
  if (auto rhsElemTy =
          rhsTy.getElementType().dyn_cast<quant::QuantizedType>()) {
    rhsTy = getSameShapeTensorType(rhsTy, rhsElemTy.getStorageType());
  }
  return lhsTy == rhsTy;
}

ParseResult ConstOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse the generic form.
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseRParen()) return failure();
    if (parser.parseOptionalAttrDict(result.attributes)) return failure();
    if (parser.parseColon() || parser.parseLParen() || parser.parseRParen() ||
        parser.parseArrow())
      return failure();
    Type resultTy;
    if (parser.parseType(resultTy)) {
      return failure();
    }
    result.addTypes(resultTy);
    return success();
  }

  ElementsAttr valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  if (parser.parseCustomAttributeWithFallback(valueAttr, Type{}, "value",
                                              result.attributes)) {
    return failure();
  }
  result.addTypes(valueAttr.getType());
  return success();
}

/// Print a `constant` op.
///
/// op ::= attr-dict $value
///
/// When the `value` and `output` have different type, it just uses the default
/// operator assembly format as a fallback.
void ConstOp::print(::mlir::OpAsmPrinter& p) {
  // If not all types are the same, use generic form.
  if (value().getType() != getType()) {
    p.printGenericOp(getOperation(), /*printOpName=*/false);
    return;
  }

  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  p << ' ';
  p.printStrippedAttrOrType(valueAttr());
}

//===----------------------------------------------------------------------===//
// CustomCallOp
//===----------------------------------------------------------------------===//

LogicalResult CustomCallOp::verify() {
  // If both operand and result layout attributes are not specified then nothing
  // to verify.
  if (!operand_layouts().hasValue() && !result_layouts().hasValue())
    return success();

  // Layout constraints for either both operands & results or none should be
  // specified.
  if (operand_layouts().hasValue() != result_layouts().hasValue())
    return emitOpError() << "Layout attributes should be specified for "
                            "either both operands and results or none.";

  // Helper function to verify types and the corresponding layouts.
  auto verifyTypesAndLayouts =
      [this](TypeRange types, mlir::ArrayAttr layouts,
             const std::string& valueName) -> LogicalResult {
    if (types.size() != layouts.size())
      return emitOpError() << "Number of " << valueName
                           << "s must match the number of " << valueName
                           << " layouts, " << types.size()
                           << " != " << layouts.size();

    for (const auto& indexedTypeAndLayout :
         llvm::enumerate(llvm::zip(types, layouts))) {
      // Get index for more descriptive error message.
      auto index = indexedTypeAndLayout.index();

      auto type = std::get<0>(indexedTypeAndLayout.value());
      auto layout = std::get<1>(indexedTypeAndLayout.value())
                        .cast<DenseIntElementsAttr>();

      if (type.isa<TupleType>())
        return emitOpError() << "Tuple types are not fully supported with "
                                "layout constraints yet";
      auto tensorType = type.dyn_cast<TensorType>();

      // For non-tensor types such as !mhlo.token, the layout should be empty.
      if (!tensorType) {
        if (layout.empty()) continue;
        return emitOpError()
               << "Only tensor types can have non-empty layout: " << valueName
               << " #" << index << " of type " << type << " has layout "
               << layout;
      }

      // For unranked tensors, we cannot verify the compatibility with layout
      // any further.
      if (!tensorType.hasRank()) continue;

      // Layout must be a permutation of [0, N) where N is the rank of the
      // tensor type.
      std::vector<int64_t> range(tensorType.getRank());
      std::iota(range.begin(), range.end(), 0);
      if (tensorType.getRank() != layout.size() ||
          !std::is_permutation(range.begin(), range.end(), layout.begin()))
        return emitOpError() << "incorrect layout " << layout << " for type "
                             << type << ", layout must be a permutation of [0, "
                             << tensorType.getRank() << ")";
    }
    return success();
  };

  // At this point both `operand_layouts` and `result_layouts` are defined.
  ArrayAttr operandLayouts = this->operand_layouts().getValue();
  ArrayAttr resultLayouts = this->result_layouts().getValue();

  // Full support for layouts for arbitrary nesting of tuples is not
  // supported yet.
  //
  // If result does not have any tuples, then i-th element of `result_layouts`
  // specifies the layout constraints on i-th result.
  //
  // For the common case of a single tuple result packing non-tuple values, the
  // i-th element of `result_layouts` specifies layout for i-th element of the
  // result tuple.
  TypeRange resultTypes;
  if (getNumResults() == 1 && getResult(0).getType().isa<TupleType>())
    resultTypes = getResult(0).getType().cast<TupleType>().getTypes();
  else
    resultTypes = getResultTypes();

  // Verify that operands and operand layouts match.
  if (failed(
          verifyTypesAndLayouts(getOperandTypes(), operandLayouts, "operand")))
    return failure();

  // Verify that results and result layouts match.
  return verifyTypesAndLayouts(resultTypes, resultLayouts, "result");
}

void CustomCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // CustomCall has "all possible effects" unless the has_side_effect is present
  // and set to false.
  auto hasSideEffect = (*this)->getAttrOfType<BoolAttr>("has_side_effect");
  if (hasSideEffect && !hasSideEffect.getValue()) return;
  effects.emplace_back(MemoryEffects::Allocate::get());
  effects.emplace_back(MemoryEffects::Free::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Read::get());
}

//===----------------------------------------------------------------------===//
// CholeskyOp
//===----------------------------------------------------------------------===//

// The following properties are already enforced by the ODS:
//   P0. a.element_type is floating or complex
// We intend to verify the following properties
//   P1. The 'a' argument to Cholesky must have rank >= 2, got shape %s
//   P2. The two minor dimensions of 'a' must have equal size, got %s.
LogicalResult CholeskyOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  CholeskyOp::Adaptor adaptor(operands, attributes, regions);
  Type aType = adaptor.a().getType();
  RankedTensorType aRankedType = aType.dyn_cast<RankedTensorType>();
  if (!aRankedType) {
    inferredReturnShapes.emplace_back(
        aType.cast<TensorType>().getElementType());
    return success();
  }

  ArrayRef<int64_t> aShape = aRankedType.getShape();
  if (aShape.size() < 2) {
    return emitOptionalError(
        location, "argument 'a' must have rank >= 2, got shape ", aShape, ".");
  }

  int64_t lastDim = aShape[aShape.size() - 1];
  int64_t penultimateDim = aShape[aShape.size() - 2];
  if (!isDynamicDimSize(lastDim) && !isDynamicDimSize(penultimateDim) &&
      lastDim != penultimateDim) {
    return emitOptionalError(
        location, "minor dimensions of 'a' must have equal size, got shape ",
        aShape, ".");
  }
  inferredReturnShapes.emplace_back(aRankedType.getShape(),
                                    aRankedType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// DotOp
//===----------------------------------------------------------------------===//
namespace {
bool dimCompatible(int64_t a, int64_t b) {
  return isDynamicDimSize(a) || isDynamicDimSize(b) || a == b;
}

ShapedType inferDotReturnType(ShapedType lhs, ShapedType rhs) {
  auto elementType = lhs.getElementType();
  if (!lhs.hasRank() || !rhs.hasRank()) {
    return UnrankedTensorType::get(elementType);
  }

  // vector dot vector
  if (1 == lhs.getRank() && 1 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(0), rhs.getDimSize(0))) {
    return RankedTensorType::get({}, elementType);
  }
  // matrix dot vector
  if (2 == lhs.getRank() && 1 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(1), rhs.getDimSize(0))) {
    return RankedTensorType::get({lhs.getDimSize(0)}, elementType);
  }
  // vector dot matrix
  if (1 == lhs.getRank() && 2 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(0), rhs.getDimSize(0))) {
    return RankedTensorType::get({rhs.getDimSize(1)}, elementType);
  }
  // matrix dot matrix
  if (2 == lhs.getRank() && 2 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(1), rhs.getDimSize(0))) {
    int64_t shape[2] = {lhs.getDimSize(0), rhs.getDimSize(1)};
    return RankedTensorType::get(shape, elementType);
  }
  return {};
}
}  // namespace

LogicalResult DotOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  DotOp::Adaptor op(operands);
  auto lhsType = op.lhs().getType().cast<ShapedType>();
  auto rhsType = op.rhs().getType().cast<ShapedType>();
  inferredReturnTypes.push_back(inferDotReturnType(lhsType, rhsType));
  return success();
}

LogicalResult DotOp::verify() {
  auto lhsType = lhs().getType().cast<ShapedType>();
  auto rhsType = rhs().getType().cast<ShapedType>();
  auto resultType = getType().cast<ShapedType>();
  auto expectReturnType = inferDotReturnType(lhsType, rhsType);
  if (!expectReturnType) {
    return emitError() << "Unexpected operands type: " << lhsType << " and "
                       << rhsType;
  }
  if (resultType.hasRank() && expectReturnType.hasRank()) {
    if (resultType.getShape() != expectReturnType.getShape()) {
      return emitError() << "Unexpected result type: has " << resultType
                         << " but inferred " << expectReturnType
                         << " from operands " << lhsType << " and " << rhsType;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DotGeneralOp
//===----------------------------------------------------------------------===//

LogicalResult DotGeneralOp::verify() {
  auto dotDimensionNumbers = this->dot_dimension_numbers();
  int64_t lhsBatchingDimensionsSize =
      dotDimensionNumbers.getLhsBatchingDimensions().size();
  int64_t rhsBatchingDimensionsSize =
      dotDimensionNumbers.getRhsBatchingDimensions().size();
  if (lhsBatchingDimensionsSize != rhsBatchingDimensionsSize) {
    return emitError()
           << "lhs and rhs should have the same number of batching dimensions";
  }
  int64_t lhsContractingDimensionsSize =
      dotDimensionNumbers.getLhsContractingDimensions().size();
  int64_t rhsContractingDimensionsSize =
      dotDimensionNumbers.getRhsContractingDimensions().size();
  if (lhsContractingDimensionsSize != rhsContractingDimensionsSize) {
    return emitError() << "lhs and rhs should have the same number of "
                          "contracting dimensions";
  }
  return success();
}

LogicalResult DotGeneralOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto lhsType = lhs().getType().dyn_cast<ShapedType>();
  auto rhsType = rhs().getType().dyn_cast<ShapedType>();
  if (!lhsType || !rhsType) {
    return failure();
  }

  Adaptor adaptor(operands);
  auto dimNumbers = dot_dimension_numbers();
  SmallVector<Value> dimensions;
  for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions()) {
    dimensions.push_back(
        builder.create<tensor::DimOp>(getLoc(), adaptor.lhs(), lhsDim));
  }

  for (int64_t i = 0; i < lhsType.getRank(); i++) {
    if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.lhs(), i));
    }
  }
  for (int64_t i = 0; i < rhsType.getRank(); i++) {
    if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.rhs(), i));
    }
  }

  reifiedReturnShapes.push_back(
      builder.create<tensor::FromElementsOp>(getLoc(), dimensions));
  return success();
}

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

// TODO(atondwal): add shape ineference for FFT that generates a return type

// We intend to verify the following properties
// P1. 1 <= rank <= 3
// P2. operand shape dimensions agree with fft_length for the given fft_type
// P3. Element types agree with fft_type
LogicalResult FftOp::verify() {
  // P1.
  auto fftRank = fft_length().size();
  if (!(fftRank <= 3 && fftRank >= 1)) {
    return emitOpError() << "rank must be between 1 and 3, but got " << fftRank
                         << ".";
  }

  // P2.
  auto operandType = operand().getType().dyn_cast<RankedTensorType>();
  if (!operandType) return success();
  auto operandShape = operandType.getShape();
  if (operandShape.size() < fftRank) {
    return emitOpError() << "operand rank must be greater than fft rank of "
                         << fftRank << " for operand of type " << operandType
                         << ".";
  }

  if (fft_type() == FftType::RFFT) {
    auto shapeBack = operandShape.take_back(fftRank);
    for (auto it : llvm::zip(shapeBack, fft_length().getValues<int64_t>())) {
      if (std::get<0>(it) != std::get<1>(it)) {
        return emitError()
               << "RFFT requires innermost dimensions match fft_length. Got: "
               << operandShape << " but wanted " << fft_length() << ".";
      }
    }
  }
  if (fft_type() == FftType::IRFFT) {
    auto shapeBack = operandShape.take_back(fftRank).drop_back();
    for (auto it : llvm::zip(shapeBack, fft_length().getValues<int64_t>())) {
      if (std::get<0>(it) != std::get<1>(it)) {
        return emitError() << "IRFFT requires non-final dimensions "
                              "match fft_length. Got: "
                           << operandShape << " but wanted " << fft_length()
                           << ", and " << std::get<0>(it)
                           << " != " << std::get<1>(it) << ".";
      }
    }
    if (operandShape[operandShape.size() - 1] !=
        fft_length().getValues<int64_t>()[fftRank - 1] / 2 + 1)
      return emitError() << "IRFFT requires innermost dimension match "
                            "fft_length[-1]/2+1. Got: "
                         << operandShape << " but fft_length is "
                         << fft_length() << ".";
  }

  // P3. Element type agreement
  // FFT : C -> C
  // IFF : C -> C
  // RFFT : R -> C
  // IRFFT : C -> R
  if (fft_type() == FftType::RFFT) {
    if (operandType.getElementType().isa<ComplexType>()) {
      return emitError() << "RFFT takes a real tensor as input, but is given "
                         << operandType << ".";
    }
  } else if (!operandType.getElementType().isa<ComplexType>()) {
    return emitError() << stringifyFftType(fft_type())
                       << " takes a complex tensor as input, but is given "
                       << operandType << ".";
  }

  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();
  if (!resultType) return success();
  if (fft_type() == FftType::IRFFT) {
    if (resultType.getElementType().isa<ComplexType>()) {
      return emitError()
             << "IRFFT produces a real tensor as output, but is given "
             << resultType << ".";
    }
  } else if (!resultType.getElementType().isa<ComplexType>()) {
    return emitError() << stringifyFftType(fft_type())
                       << " produces a complex tensor as output, but is given "
                       << resultType << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

// Converts gather ops to slice ops in case we have a single set of constant
// indices.
struct GatherSlice : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter& rewriter) const override {
    DenseIntElementsAttr index;
    if (!matchPattern(gather.start_indices(), m_Constant(&index)))
      return failure();

    const auto& dnums = gather.dimension_numbers();
    if (dnums.getIndexVectorDim() != 0 || index.getType().getRank() > 1)
      return failure();

    // TODO(tberghammer): Remove when the verifier catches this case what is
    // invalid if all previous condition holds.
    if (index.getNumElements() != dnums.getStartIndexMap().size())
      return failure();

    RankedTensorType operandType =
        gather->getOperand(0).getType().dyn_cast<RankedTensorType>();
    if (!operandType || !operandType.hasStaticShape()) return failure();

    auto sliceEnd =
        llvm::to_vector<8>(gather.slice_sizes().getValues<int64_t>());
    llvm::SmallVector<int64_t, 8> sliceStart(sliceEnd.size(), 0);
    for (auto it :
         llvm::zip(dnums.getStartIndexMap(), index.getValues<APInt>())) {
      int64_t mapIndex = std::get<0>(it);
      // Clamp the indices within bounds to faithfully mirror gather semantics.
      int64_t offset =
          clamp(std::get<1>(it).getSExtValue(), static_cast<int64_t>(0),
                operandType.getDimSize(mapIndex) - sliceEnd[mapIndex]);
      sliceStart[mapIndex] += offset;
      sliceEnd[mapIndex] += offset;
    }

    llvm::SmallVector<int64_t, 8> sliceStride(sliceEnd.size(), 1);
    llvm::SmallVector<int64_t, 8> sliceShape(sliceEnd.size());
    for (size_t i = 0; i < sliceEnd.size(); ++i) {
      sliceShape[i] = sliceEnd[i] - sliceStart[i];
    }
    Type elementType = gather.getType().cast<TensorType>().getElementType();
    auto sliceType = RankedTensorType::get(sliceShape, elementType);
    Value result = rewriter.create<SliceOp>(
        gather.getLoc(), sliceType, gather.getOperand(0),
        rewriter.getI64TensorAttr(sliceStart),
        rewriter.getI64TensorAttr(sliceEnd),
        rewriter.getI64TensorAttr(sliceStride));

    auto collapsedSliceDims = dnums.getCollapsedSliceDims();
    if (!collapsedSliceDims.empty()) {
      llvm::SmallVector<int64_t, 8> reshapeShape;
      for (size_t i = 0; i < sliceShape.size(); ++i) {
        if (llvm::count(collapsedSliceDims, i) == 0) {
          reshapeShape.push_back(sliceShape[i]);
        }
      }
      auto reshapeType = RankedTensorType::get(reshapeShape, elementType);
      result = rewriter.create<ReshapeOp>(gather.getLoc(), reshapeType, result);
    }

    result.setType(gather.getType());
    rewriter.replaceOp(gather, result);
    return success();
  }
};

void GatherOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<GatherSlice>(context);
}

namespace {

// following https://www.tensorflow.org/xla/operation_semantics#gather
// The bounds for the output array along dimension i is computed as follows:
// (1) If i is present in batch_dims (i.e. is equal to batch_dims[k] for some k)
// then we pick
// the corresponding dimension bounds out of start_indices.shape, skipping
// index_vector_dim
// (i.e. pick start_indices.shape.dims[k] if k < index_vector_dim and
// start_indices.shape.dims[k+1] otherwise).
// (2) If i is present in offset_dims (i.e. equal to offset_dims[k] for some k)
// then we pick
// the corresponding bound out of slice_sizes after accounting for
// collapsed_slice_dims
// (i.e. we pick adjusted_slice_sizes[k] where adjusted_slice_sizes is
// slice_sizes with the bounds at indices collapsed_slice_dims removed).

void getSliceSizeValues(GatherOp* gather, OpBuilder& builder, Location loc,
                        ValueRange operands,
                        SmallVectorImpl<Value>& sliceSizes) {
  for (int64_t val : gather->slice_sizes().getValues<int64_t>()) {
    sliceSizes.push_back(builder.create<arith::ConstantIndexOp>(loc, val));
  }
}

void getSliceSizeValues(DynamicGatherOp* d_gather, OpBuilder& builder,
                        Location loc, ValueRange operands,
                        SmallVectorImpl<Value>& sliceSizeValues) {
  DynamicGatherOp::Adaptor adaptor(operands);
  Value sliceSizes = adaptor.slice_sizes();
  auto sliceSizesTy = sliceSizes.getType().cast<ShapedType>();
  for (int64_t i = 0; i < sliceSizesTy.getDimSize(0); ++i) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    sliceSizeValues.push_back(
        builder.create<tensor::ExtractOp>(loc, sliceSizes, idx));
  }
}

static LogicalResult verifyGather(
    ShapeAdaptor operandShape, ShapeAdaptor startIndicesShape,
    ShapeAdaptor sliceSizesShape, GatherDimensionNumbersAttr dimensionNumbers,
    llvm::function_ref<InFlightDiagnostic()> errorEmitter) {
  // This should be fully expressible with type constraints, but it isn't
  // obvious how to do that with the current infrastructure.
  if (sliceSizesShape.hasRank() && sliceSizesShape.getRank() != 1)
    return errorEmitter() << "slice_sizes.rank != 1";

  int64_t indexVectorDim = dimensionNumbers.getIndexVectorDim();
  if (startIndicesShape.hasRank()) {
    // index_vector_dim == start_indices.rank implies a trailing 1 on the shape
    // of start_indices.
    if (indexVectorDim > startIndicesShape.getRank())
      return errorEmitter() << "index_vector_dim " << indexVectorDim
                            << " is out of bounds for start indices with rank "
                            << startIndicesShape.getRank();

    bool impliedTrailingDim = indexVectorDim == startIndicesShape.getRank();
    if (impliedTrailingDim || !startIndicesShape.isDynamicDim(indexVectorDim)) {
      int64_t effectiveDimSize;
      if (impliedTrailingDim)
        effectiveDimSize = 1;
      else
        effectiveDimSize = startIndicesShape.getDimSize(indexVectorDim);
      if (effectiveDimSize != dimensionNumbers.getStartIndexMap().size())
        return errorEmitter() << "start_index_map size ("
                              << dimensionNumbers.getStartIndexMap().size()
                              << ") is not equal to size of index dimension ("
                              << indexVectorDim << ") of start_indices ("
                              << effectiveDimSize << ")";
    }
  }

  int64_t impliedOperandRank = dimensionNumbers.getOffsetDims().size() +
                               dimensionNumbers.getCollapsedSliceDims().size();
  if (operandShape.hasRank() && operandShape.getRank() != impliedOperandRank)
    return errorEmitter() << "offset_dims size ("
                          << dimensionNumbers.getOffsetDims().size()
                          << ") plus collapse_slice_dims size ("
                          << dimensionNumbers.getCollapsedSliceDims().size()
                          << ") is not equal to operand rank ("
                          << operandShape.getRank() << ")";

  if (sliceSizesShape.hasStaticShape()) {
    int64_t sliceRank = sliceSizesShape.getNumElements();

    if (sliceRank != impliedOperandRank)
      return errorEmitter() << "slice_sizes size (" << sliceRank
                            << ") not equal to (implied) operand rank ("
                            << impliedOperandRank << ")";

    for (auto dim : dimensionNumbers.getCollapsedSliceDims())
      if (dim >= sliceRank)
        return errorEmitter()
               << "collapsed dimension " << dim
               << " is greater than slice_sizes.size (" << sliceRank << ")";
  }

  return success();
}

static LogicalResult verifyStaticGather(
    ShapeAdaptor operandShape, ShapeAdaptor startIndicesShape,
    DenseIntElementsAttr sliceSizes,
    GatherDimensionNumbersAttr dimensionNumbers,
    llvm::function_ref<InFlightDiagnostic()> errorEmitter) {
  // For some reason the getType call is necessary here
  if (failed(verifyGather(
          /*operandShape=*/operandShape,
          /*startIndicesShape=*/startIndicesShape,
          /*sliceSizesShape=*/sliceSizes.getType(), dimensionNumbers,
          errorEmitter)))
    return failure();

  for (auto dim : dimensionNumbers.getCollapsedSliceDims()) {
    int64_t sliceDimSize = sliceSizes.getValues<int64_t>()[dim];
    if (sliceDimSize != 1) {
      return errorEmitter() << "slice_sizes collapsed dimension " << dim
                            << " != 1 (" << sliceDimSize << ")";
    }
  }

  if (operandShape.hasRank()) {
    for (const auto& it : llvm::enumerate(sliceSizes.getValues<int64_t>())) {
      if (operandShape.isDynamicDim(it.index())) continue;
      auto operandDimSize = operandShape.getDimSize(it.index());
      auto sliceDimSize = it.value();
      if (sliceDimSize > operandDimSize)
        return errorEmitter() << "slice size (" << sliceDimSize
                              << ") is larger than operand dimension ("
                              << operandDimSize << ") at index " << it.index();
    }
  }
  return success();
}

template <typename dimTy>
static void inferGatherShape(
    int64_t resultRank, llvm::function_ref<dimTy(int64_t)> getStartIndicesDim,
    llvm::function_ref<dimTy(int64_t)> getSliceDim,
    GatherDimensionNumbersAttr dimensionNumbers,
    SmallVectorImpl<dimTy>& shape) {
  ArrayRef<int64_t> collapsedSliceDims =
      dimensionNumbers.getCollapsedSliceDims();
  int64_t indexVectorDim = dimensionNumbers.getIndexVectorDim();

  // We don't necessarily know the rank of sliceSizes, but we do know that it
  // can't be larger than the highest collapsed dimension. So go through those
  // and populate the leading dimensions of adjustedSliceSizes. The trailing
  // dimensions can just be adjusted by an offset.
  const auto* maxCollapsedDimIt =
      std::max_element(collapsedSliceDims.begin(), collapsedSliceDims.end());
  int64_t maxCollapsedDim = -1;
  if (maxCollapsedDimIt != collapsedSliceDims.end())
    maxCollapsedDim = *maxCollapsedDimIt;

  SmallVector<dimTy> adjustedSliceSizePrefix;
  for (int dimIndex = 0; dimIndex <= maxCollapsedDim; ++dimIndex) {
    if (llvm::is_contained(collapsedSliceDims, dimIndex)) continue;
    adjustedSliceSizePrefix.push_back(getSliceDim(dimIndex));
  }
  auto getAdjustedSliceDim = [&](int64_t index) -> dimTy {
    if (index < adjustedSliceSizePrefix.size())
      return adjustedSliceSizePrefix[index];
    return getSliceDim(index + collapsedSliceDims.size());
  };

  ArrayRef<int64_t> offsetDims = dimensionNumbers.getOffsetDims();

  // Dimensions in the output that aren't offset dimensions are called batch
  // dimensions.
  SmallVector<int64_t> batchDims;
  for (int dim = 0; dim < resultRank; ++dim)
    if (!llvm::is_contained(offsetDims, dim)) batchDims.push_back(dim);

  for (int i = 0; i < resultRank; ++i) {
    const auto* offsetDimsIt =
        std::find(offsetDims.begin(), offsetDims.end(), i);
    if (offsetDimsIt != offsetDims.end()) {
      auto index = std::distance(offsetDims.begin(), offsetDimsIt);
      shape.push_back(getAdjustedSliceDim(index));
      continue;
    }
    auto* batchDimsIt = std::find(batchDims.begin(), batchDims.end(), i);
    assert(batchDimsIt != batchDims.end());
    auto index = std::distance(batchDims.begin(), batchDimsIt);
    // This can never run into the special case where start_indices gets
    // implicitly expanded with a trailing 1 if
    // index_vector_dim = start_indices.rank because then index would equal
    // index_vector_dim, which means we'd be looking at index+1, which would be
    // out of bounds anyway.
    if (index >= indexVectorDim) ++index;
    shape.push_back(getStartIndicesDim(index));
  }
}

static LogicalResult inferGatherReturnTypeComponents(
    ShapeAdaptor operandShape, ShapeAdaptor startIndicesShape,
    llvm::function_ref<int64_t(int64_t)> getSliceDim,
    GatherDimensionNumbersAttr dimensionNumbers,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  Type elementType = operandShape.getElementType();

  // We need this to determine the result rank. We could still place bounds on
  // the result rank if that was something ShapedTypeComponents could express.
  if (!startIndicesShape.hasRank()) {
    inferredReturnShapes.push_back(elementType);
    return success();
  }

  ArrayRef<int64_t> offsetDims = dimensionNumbers.getOffsetDims();
  int64_t startIndicesRank = startIndicesShape.getRank();
  // If index_vector_dim == start_indices.rank, then an implicit trailing 1 is
  // appended to start_indices shape.
  if (dimensionNumbers.getIndexVectorDim() == startIndicesRank)
    ++startIndicesRank;
  int64_t resultRank = offsetDims.size() + startIndicesRank - 1;

  auto getStartIndicesDim = [&](int64_t index) {
    return startIndicesShape.getDimSize(index);
  };

  SmallVector<int64_t> shape;
  inferGatherShape<int64_t>(resultRank, getStartIndicesDim, getSliceDim,
                            dimensionNumbers, shape);

  inferredReturnShapes.emplace_back(shape, elementType);
  return success();
}

template <typename Op>
LogicalResult reifyGatherShape(Op* op, OpBuilder& builder, ValueRange operands,
                               SmallVectorImpl<Value>& reifiedReturnShapes) {
  // No support for unranked gather output shape a.t.m.
  auto resultTy =
      op->getResult().getType().template dyn_cast<RankedTensorType>();
  if (!resultTy) return failure();

  typename Op::Adaptor adaptor(operands);
  Value startIndices = adaptor.start_indices();

  Location loc = op->getLoc();
  int resultRank = resultTy.getRank();
  Type shapeElTy = startIndices.getType().cast<ShapedType>().getElementType();
  auto toShapeElType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeElTy);
  };

  SmallVector<Value, 4> sliceSizes;
  getSliceSizeValues(op, builder, loc, operands, sliceSizes);
  llvm::transform(sliceSizes, sliceSizes.begin(),
                  [&](Value v) { return toShapeElType(v); });

  auto getStartIndicesDim = [&](int64_t index) {
    return toShapeElType(
        builder.create<tensor::DimOp>(loc, startIndices, index));
  };
  SmallVector<Value, 4> shapeValues;
  auto getSliceDim = [&sliceSizes](int64_t index) -> Value {
    return sliceSizes[index];
  };
  inferGatherShape<Value>(resultRank, getStartIndicesDim, getSliceDim,
                          op->dimension_numbers(), shapeValues);

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc, RankedTensorType::get({resultRank}, shapeElTy), shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

}  // namespace

LogicalResult GatherOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return reifyGatherShape(this, builder, operands, reifiedReturnShapes);
}

LogicalResult GatherOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  // This can get called before other op verify methods, so we have to do a
  // bunch of verification up front. With a better story for ordering and/or
  // multi-phase op verification, this should hopefully all go away.
  Location loc = location.getValueOr(UnknownLoc::get(context));
  auto errorEmitter = [&loc]() {
    return mlir::emitError(loc)
           << "'" << GatherOp::getOperationName() << "' op ";
  };
  GatherOp::Adaptor adaptor(operands, attributes, regions);
  if (failed(adaptor.verify(loc))) return failure();

  // We want the ShapeAdaptors, so can't route via the adaptor :-/
  ShapeAdaptor operandShape = operands.getShape(0);
  ShapeAdaptor startIndicesShape = operands.getShape(1);
  GatherDimensionNumbersAttr dimensionNumbers = adaptor.dimension_numbers();
  DenseIntElementsAttr sliceSizesAttr = adaptor.slice_sizes();

  if (failed(verifyStaticGather(/*operandShape=*/operandShape,
                                /*startIndicesShape=*/startIndicesShape,
                                /*sliceSizes=*/sliceSizesAttr, dimensionNumbers,
                                errorEmitter)))
    return failure();

  auto getSliceDim = [&sliceSizesAttr](int64_t index) -> int64_t {
    return sliceSizesAttr.getValues<int64_t>()[index];
  };

  return inferGatherReturnTypeComponents(operandShape, startIndicesShape,
                                         getSliceDim, dimensionNumbers,
                                         inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// DynamicGatherOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicGatherOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return reifyGatherShape(this, builder, operands, reifiedReturnShapes);
}

LogicalResult DynamicGatherOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  // This can get called before other op verify methods, so we have to do a
  // bunch of verification up front. With a better story for ordering and/or
  // multi-phase op verification, this should hopefully all go away.
  Location loc = location.getValueOr(UnknownLoc::get(context));
  auto errorEmitter = [&loc]() {
    return mlir::emitError(loc)
           << "'" << DynamicGatherOp::getOperationName() << "' op ";
  };
  DynamicGatherOp::Adaptor adaptor(operands, attributes, regions);
  if (failed(adaptor.verify(loc))) return failure();

  // We want the ShapeAdaptors, so can't route via the adaptor :-/
  ShapeAdaptor operandShape = operands.getShape(0);
  ShapeAdaptor startIndicesShape = operands.getShape(1);
  ShapeAdaptor sliceSizesShape = operands.getShape(2);
  GatherDimensionNumbersAttr dimensionNumbers = adaptor.dimension_numbers();

  if (failed(verifyGather(/*operandShape=*/operandShape,
                          /*startIndicesShape=*/startIndicesShape,
                          /*sliceSizesShape=*/sliceSizesShape, dimensionNumbers,
                          errorEmitter)))
    return failure();

  auto getSliceDim = [](int64_t index) { return ShapedType::kDynamicSize; };
  return inferGatherReturnTypeComponents(operandShape, startIndicesShape,
                                         getSliceDim, dimensionNumbers,
                                         inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//
//
LogicalResult GetDimensionSizeOp::verify() { return verifyDimAttr(*this); }

/// Fold get_dimension_size when the said shape dimension is a constant.
OpFoldResult GetDimensionSizeOp::fold(ArrayRef<Attribute> attrs) {
  RankedTensorType type = operand().getType().dyn_cast<RankedTensorType>();
  if (!type) return {};

  int32_t dim = dimension();
  if (type.isDynamicDim(dim)) return {};
  // The result type is always is a 0-d i32 tensor.
  return DenseIntElementsAttr::get<int32_t>(
      getResult().getType().cast<RankedTensorType>(), type.getDimSize(dim));
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

LogicalResult IotaOp::verify() {
  auto shape = getType().cast<ShapedType>();
  if (!shape.hasRank()) return success();

  if (shape.getRank() == 0) return emitOpError() << "does not support scalars.";

  auto iotaDimension = this->iota_dimension();
  if (iotaDimension >= shape.getRank() || iotaDimension < 0)
    return emitOpError()
           << "iota dimension cannot go beyond the output rank or be negative.";
  return success();
}

// Iota operations across multiple dimensions can be reduced to an iota and a
// ranked broadcast.
struct IotaBroadcast : public OpRewritePattern<IotaOp> {
  using OpRewritePattern<IotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IotaOp iota,
                                PatternRewriter& rewriter) const override {
    auto resultTy = iota.getType().cast<ShapedType>();
    if (!resultTy.hasRank() || resultTy.getRank() < 2) {
      return failure();
    }

    auto iotaDimension = iota.iota_dimension();

    auto iotaType = RankedTensorType::get({resultTy.getDimSize(iotaDimension)},
                                          resultTy.getElementType());

    auto newIota = rewriter.create<IotaOp>(iota.getLoc(), iotaType,
                                           rewriter.getI64IntegerAttr(0));

    auto broadcastAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iotaDimension});
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(iota, resultTy, newIota,
                                                  broadcastAttr);
    return success();
  }
};

void IotaOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<IotaBroadcast>(context);
}

OpFoldResult IotaOp::fold(ArrayRef<Attribute> operands) {
  auto dimension = iota_dimension();
  auto resultTy = getResult().getType().cast<ShapedType>();
  if (resultTy.hasRank() && resultTy.getDimSize(dimension) == 1) {
    Builder builder(getContext());
    return builder.getZeroAttr(resultTy);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// DynamicIotaOp
//===----------------------------------------------------------------------===//

namespace {

struct DynamicIotaIsStatic : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter& rewriter) const override {
    auto resultTy = iota.getType().cast<ShapedType>();
    if (!resultTy.hasStaticShape()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IotaOp>(iota, resultTy, iota.iota_dimension());
    return success();
  }
};

// Dynamic Iota operations across multiple dimensions can be reduced to an iota
// and a ranked broadcast.
struct DynamicIotaBroadcast : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter& rewriter) const override {
    auto resultTy = iota.getType().cast<ShapedType>();
    if (!resultTy.hasRank() || resultTy.getRank() < 2) {
      return failure();
    }

    auto iotaDimension = iota.iota_dimension();
    auto iotaDimensionInt = iotaDimension;

    auto convertedShape = rewriter.create<arith::IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            iota.output_shape().getType().cast<ShapedType>().getShape(),
            rewriter.getI64Type()),
        iota.output_shape());

    auto slicedShape = rewriter.create<SliceOp>(
        iota.getLoc(), convertedShape,
        rewriter.getI64TensorAttr(iotaDimensionInt),
        rewriter.getI64TensorAttr(iotaDimensionInt + 1),
        rewriter.getI64TensorAttr(1));

    auto convertedSlicedShape = rewriter.create<arith::IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            {1},
            iota.output_shape().getType().cast<ShapedType>().getElementType()),
        slicedShape);

    auto iotaType = RankedTensorType::get(
        {resultTy.getDimSize(iotaDimensionInt)}, resultTy.getElementType());

    auto newIota = rewriter.create<DynamicIotaOp>(
        iota.getLoc(), iotaType, convertedSlicedShape,
        rewriter.getI64IntegerAttr(0));

    auto broadcastAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iotaDimension});
    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        iota, resultTy, newIota, iota.output_shape(), broadcastAttr);
    return success();
  }
};

}  // namespace

void DynamicIotaOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<DynamicIotaIsStatic>(context);
  results.add<DynamicIotaBroadcast>(context);
}

static Value castToIndexTensor(OpBuilder& builder, Location loc,
                               Value shapeOp) {
  ShapedType resultTy = shape::getExtentTensorType(
      builder.getContext(), shapeOp.getType().cast<ShapedType>().getDimSize(0));
  if (shapeOp.getType() == resultTy) return shapeOp;  // Nothing to do.
  return builder.create<arith::IndexCastOp>(loc, resultTy, shapeOp);
}

LogicalResult DynamicIotaOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicIotaOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.output_shape()));
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicUpdateSliceOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicUpdateSliceOp::verify() {
  OperandRange indices = start_indices();
  if (indices.size() <= 1) return success();

  // Note: start_indices is constrained to Variadic<HLO_ScalarIntTensor>, so it
  // is OK to cast indices to ShapedType here.
  auto idxTensor = indices.take_front().front().getType().cast<ShapedType>();
  Type firstElemTy = idxTensor.getElementType();
  Type elemTy;

  for (auto idx : llvm::drop_begin(indices, 1)) {
    idxTensor = idx.getType().cast<ShapedType>();
    elemTy = idxTensor.getElementType();

    if (firstElemTy != elemTy) {
      return emitOpError() << "start indices must have same element type "
                              "(encountered mismatch: "
                           << firstElemTy << " vs " << elemTy << ")";
    }
  }
  return success();
}

OpFoldResult DynamicUpdateSliceOp::fold(ArrayRef<Attribute> operands) {
  auto operandShape = this->operand().getType().cast<RankedTensorType>();
  auto updateShape = this->update().getType().cast<RankedTensorType>();

  if (operandShape != updateShape || !operandShape.hasStaticShape()) {
    return {};
  }

  // Ensure that indices are 0 constants. The 0 check mostly ensures
  // correctness. For non-constants, the pattern does not fold to avoid hiding
  // the behavior of incorrect user input.
  for (Value index : this->start_indices()) {
    DenseIntElementsAttr deAttr;
    if (!matchPattern(index, m_Constant(&deAttr))) return {};
    if (!deAttr.getSplatValue<IntegerAttr>().getValue().isZero()) return {};
  }
  return this->update();
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto operandTy = (*operands.begin()).getType().cast<ShapedType>();
  Type elementTy = operandTy.getElementType();
  if (auto complexTy = elementTy.dyn_cast<ComplexType>()) {
    elementTy = complexTy.getElementType();
  }

  Type resultTy;
  if (auto rankedOperandTy = operandTy.dyn_cast<RankedTensorType>()) {
    resultTy = RankedTensorType::get(operandTy.getShape(), elementTy,
                                     rankedOperandTy.getEncoding());
  } else if (operandTy.hasRank()) {
    resultTy = RankedTensorType::get(operandTy.getShape(), elementTy);
  } else {
    resultTy = UnrankedTensorType::get(elementTy);
  }
  inferredReturnTypes.push_back(resultTy);
  return success();
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

LogicalResult CollectivePermuteOp::verify() {
  return mlir::hlo::VerifyCollectivePermuteSourceTargetPairs(
      *this, source_target_pairs());
}

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

namespace {
// Checks:
//  P1. Same sizes for input, kernel and output spatial_dims.
//  P2. Spatial and non-spatial dimentions (for input,kernel, &output) should
//      be unique and in range [0, num_dims), where num_dims = rank of input
//      (lhs/rhs) tensors.
//
//  Note that the spatial + non-spatial dimensions may not cover all the
//  dimensions in the range [0,num) because of the presence of 'unknown'
//  dimensions (ref. cl/415132294).
LogicalResult isSpatialDimensionsValid(ConvOp op) {
  auto inputSpatialDimensions =
      op.dimension_numbers().getInputSpatialDimensions();
  auto kernelSpatialDimensions =
      op.dimension_numbers().getKernelSpatialDimensions();
  auto outputSpatialDimensions =
      op.dimension_numbers().getOutputSpatialDimensions();

  // P1.
  if ((inputSpatialDimensions.size() != kernelSpatialDimensions.size()) ||
      (inputSpatialDimensions.size() != outputSpatialDimensions.size()))
    return op.emitOpError() << "expects the same size for input, kernel and "
                               "output spatial-dimensions, but got "
                            << inputSpatialDimensions.size() << ", "
                            << kernelSpatialDimensions.size() << ", and "
                            << outputSpatialDimensions.size() << " resp.";

  // P2.
  SmallVector<int64_t> inputDnums(inputSpatialDimensions.size() + 2);
  inputDnums[0] = op.dimension_numbers().getInputBatchDimension();
  inputDnums[1] = op.dimension_numbers().getInputFeatureDimension();
  std::copy(inputSpatialDimensions.begin(), inputSpatialDimensions.end(),
            inputDnums.begin() + 2);

  SmallVector<int64_t> windowDnums(kernelSpatialDimensions.size() + 2);
  windowDnums[0] = op.dimension_numbers().getKernelInputFeatureDimension();
  windowDnums[1] = op.dimension_numbers().getKernelOutputFeatureDimension();
  std::copy(kernelSpatialDimensions.begin(), kernelSpatialDimensions.end(),
            windowDnums.begin() + 2);

  SmallVector<int64_t> outputDnums(outputSpatialDimensions.size() + 2);
  outputDnums[0] = op.dimension_numbers().getOutputBatchDimension();
  outputDnums[1] = op.dimension_numbers().getOutputFeatureDimension();
  std::copy(outputSpatialDimensions.begin(), outputSpatialDimensions.end(),
            outputDnums.begin() + 2);

  auto numDims = op.lhs().getType().cast<RankedTensorType>().getRank();
  const auto inRange = [numDims](int64_t i) { return 0 <= i && i < numDims; };

  if (!llvm::all_of(inputDnums, inRange) ||
      !llvm::all_of(windowDnums, inRange) ||
      !llvm::all_of(outputDnums, inRange))
    return op.emitOpError() << "expects input, kernel, and output "
                               "dimension-numbers to be in-range [0, "
                            << numDims << ").";

  const auto hasDuplicates = [](SmallVector<int64_t>& dnums) {
    std::sort(dnums.begin(), dnums.end());
    auto last = std::unique(dnums.begin(), dnums.end());
    return last != dnums.end();
  };

  if (hasDuplicates(inputDnums))
    return op.emitOpError()
           << "expects input dimension-numbers to be unique, got {"
           << inputDnums << "}.";

  if (hasDuplicates(windowDnums))
    return op.emitOpError()
           << "expects kernel dimension-numbers to be unique, got {"
           << windowDnums << "}.";

  if (hasDuplicates(outputDnums))
    return op.emitOpError()
           << "expects output dimension-numbers to be unique, got {"
           << outputDnums << "}.";

  return success();
}

// Verifies the following properties:
//  P1. The input, kernel, and output spatial-dimentions are valid.
//  P2. Given,
//          input-dimensions: b * input-spatial-dims * f
//          kernel-dimensions: kernel-spatial-dims * i * o
//          output-dimensions: b' * out-spatial-dims * f'
//            where b = input-batch-dims
//            where f = input-feature-dims
//            where i = kernel-input-feature-dims
//            where o = kernel-output-feature-dims
//            where b' = output-batch-dims
//            where f' = output-feature-dims
//      Check the following properties w.r.t feature_group_count (fgc) and
//      batch_group_count (bgc).
//        fgc > 0, bgc > 1 and !(fgc > 1 && bgc > 1)
//        b % bgc == 0
//        f % fgc == 0 and i = f / fgc
//        o (or f') % bgc == 0 and o (or f') % fgc == 0
LogicalResult verifyConvolutionAttributes(ConvOp op) {
  // P1.
  if (failed(isSpatialDimensionsValid(op))) return failure();

  // P2.
  const int64_t featureGroupCount = op.feature_group_count();
  const int64_t batchGroupCount = op.batch_group_count();

  if (featureGroupCount <= 0)
    return op.emitOpError()
           << "expects feature_group_count to be a positive number, got "
           << featureGroupCount << ".";

  if (batchGroupCount <= 0)
    return op.emitOpError()
           << "expects batch_group_count to be a positive number, got "
           << batchGroupCount << ".";

  if (batchGroupCount > 1 && featureGroupCount > 1)
    return op.emitOpError()
           << "expects batch_group_count and feature_group_count not to be "
              "both greater than 1. Got "
           << batchGroupCount << " and " << featureGroupCount << " resp.";

  auto lhsType = op.lhs().getType().cast<RankedTensorType>();
  const int64_t inputFeatures =
      lhsType.getShape()[op.dimension_numbers().getInputFeatureDimension()];
  const int64_t inputBatch =
      lhsType.getShape()[op.dimension_numbers().getInputBatchDimension()];

  auto rhsType = op.rhs().getType().cast<RankedTensorType>();
  const int64_t kernelInputFeatures =
      rhsType
          .getShape()[op.dimension_numbers().getKernelInputFeatureDimension()];
  const int64_t kernelOutputFeatures =
      rhsType
          .getShape()[op.dimension_numbers().getKernelOutputFeatureDimension()];

  if (!isDynamicDimSize(kernelOutputFeatures)) {
    if (kernelOutputFeatures % batchGroupCount != 0)
      return op.emitOpError() << "expects output feature dimension size ("
                              << kernelOutputFeatures
                              << ") to be a multiple of "
                                 "batch_group_count. Got batch_group_count = "
                              << batchGroupCount << ".";

    if (kernelOutputFeatures % featureGroupCount != 0)
      return op.emitOpError()
             << "expects kernel output feature dimension ("
             << kernelOutputFeatures
             << ") to be divisible by "
                "feature_group_count. For feature_group_count = "
             << featureGroupCount << ".";
  }

  if (!isDynamicDimSize(inputFeatures)) {
    if (inputFeatures % featureGroupCount != 0)
      return op.emitOpError()
             << "expects input feature dimension (" << inputFeatures
             << ") to be a multiple of "
                "feature_group_count. Got feature_group_count = "
             << featureGroupCount << ".";

    if (!isDynamicDimSize(kernelInputFeatures) &&
        inputFeatures / featureGroupCount != kernelInputFeatures)
      return op.emitOpError()
             << "expects input feature dimension (" << inputFeatures
             << ") / "
                "feature_group_count = kernel input feature dimension ("
             << kernelInputFeatures
             << "). Got feature_group_count = " << featureGroupCount << ".";
  }

  if (!isDynamicDimSize(inputBatch) && inputBatch % batchGroupCount != 0)
    return op.emitOpError() << "expects input batch dimension (" << inputBatch
                            << ") to be divisible by "
                               "batch_group_count. Got batch_group_count = "
                            << batchGroupCount << ".";

  return success();
}

// Infer the return-shape of ConvOp.
// Precondition:
//  1. Input args to ConvOp 'op' are RankedTypes.
//  2. rank-of(input-type) == rank-of(output-type)
SmallVector<int64_t> inferConvOpReturnShape(
    ConvOp op, const ArrayRef<WindowDimension> window) {
  // We keep the 'unknown' dimensions (cl/415132294) as it is in the
  // output-shape. To do that we initilize the output dimensions with the shape
  // of the return-type and updates only the spatial + non-spatial dimensions.
  // Precondition 2 ensures that size of output-shape == size of input-shape.
  SmallVector<int64_t> outputDimensions =
      to_vector(op.getResult().getType().cast<ShapedType>().getShape());

  // Infer the output spatial dimensions.
  auto lhsType = op.lhs().getType().cast<RankedTensorType>();
  auto inputSpatialDims = op.dimension_numbers().getInputSpatialDimensions();
  auto numSpatialDims = inputSpatialDims.size();
  SmallVector<int64_t> inputSpatialDimVals(numSpatialDims);
  for (int i = 0; i < numSpatialDims; ++i)
    inputSpatialDimVals[i] = lhsType.getShape()[inputSpatialDims[i]];

  auto windowOutputShape = inferWindowOutputShape(inputSpatialDimVals, window);

  for (int i = 0; i < window.size(); ++i)
    outputDimensions[op.dimension_numbers().getOutputSpatialDimensions()[i]] =
        windowOutputShape[i];

  // Infer the output-batch-dimension and output-feature-dimension.
  auto rhsType = op.rhs().getType().cast<RankedTensorType>();
  const int64_t inputBatch =
      lhsType.getShape()[op.dimension_numbers().getInputBatchDimension()];
  const int64_t kernelOutputFeatures =
      rhsType
          .getShape()[op.dimension_numbers().getKernelOutputFeatureDimension()];

  outputDimensions[op.dimension_numbers().getOutputBatchDimension()] =
      isDynamicDimSize(inputBatch) ? ShapedType::kDynamicSize
                                   : inputBatch / op.batch_group_count();
  outputDimensions[op.dimension_numbers().getOutputFeatureDimension()] =
      kernelOutputFeatures;

  return outputDimensions;
}
}  // namespace

/*
 * We intend to verify the following properties
 *  P1. Verify the input, kernel types.
 *  P2. Verify the convolution atributes.
 *  P3. Verify and collect the window atributes.
 *  P4. Verify the return shape.
 *      TODO(b/232574102): Verify the element-type of return-value.
 */
LogicalResult ConvOp::verify() {
  auto lhsType = lhs().getType().dyn_cast<RankedTensorType>();
  auto rhsType = rhs().getType().dyn_cast<RankedTensorType>();

  if (!lhsType || !rhsType) return success();

  // P1.
  int numDims = lhsType.getRank();
  if (numDims != rhsType.getRank())
    return emitOpError()
           << "expects convolution arguments to have same number of "
              "dimensions. Got: "
           << lhsType << " and " << rhsType << ".";

  if (numDims < 2)
    return emitOpError()
           << "expects convolution arguments to have >= 2 dimensions. "
              "Got: "
           << lhsType << " and " << rhsType << ".";

  // P2.
  if (failed(verifyConvolutionAttributes(*this))) return failure();

  // P3.
  auto kernelSpatialDimensions =
      dimension_numbers().getKernelSpatialDimensions();
  SmallVector<int64_t> windowDimensions(kernelSpatialDimensions.size());
  for (size_t i = 0; i < windowDimensions.size(); i++)
    windowDimensions[i] = rhsType.getShape()[kernelSpatialDimensions[i]];

  auto paddingOrErr = convertNx2Attribute(this->padding(), getLoc());
  if (failed(paddingOrErr)) return failure();
  SmallVector<std::pair<int64_t, int64_t>> padding = *paddingOrErr;

  auto windowOrErr = verifyWindowAttributesAndInferWindowDimensions(
      windowDimensions, convertDenseIntAttr(window_strides()), padding,
      convertDenseIntAttr(lhs_dilation()), convertDenseIntAttr(rhs_dilation()),
      getLoc());
  if (failed(windowOrErr)) return failure();

  // P4.
  auto actualReturnType = getResult().getType().cast<TensorType>();
  auto actualReturnElementType = actualReturnType.getElementType();
  if (!actualReturnType.hasRank()) return success();

  auto actualReturnRankedType = actualReturnType.cast<RankedTensorType>();
  if (numDims != actualReturnRankedType.getRank())
    return emitOpError() << "expects rank of convolution return-type to be "
                            "equal to input-ranks ("
                         << numDims << "), but got "
                         << actualReturnRankedType.getRank() << ".";

  auto expectedReturnShape = inferConvOpReturnShape(*this, *windowOrErr);
  auto expectedReturnType =
      RankedTensorType::get(expectedReturnShape, actualReturnElementType);
  if (failed(verifyCompatibleShape(expectedReturnType, actualReturnRankedType)))
    return emitOpError()
           << "has shape mismatch between the expected return-type ("
           << expectedReturnType << ") and actual return-type ("
           << actualReturnRankedType << ").";

  return success();
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void ConvertOp::build(OpBuilder& builder, OperationState& result, Value operand,
                      Type resultElementTy) {
  Type resultTy;
  Type operandTy = operand.getType();
  if (auto rankedTy = operandTy.dyn_cast<RankedTensorType>()) {
    resultTy = RankedTensorType::get(rankedTy.getShape(), resultElementTy);
  } else {
    resultTy = UnrankedTensorType::get(resultElementTy);
  }
  build(builder, result, resultTy, operand);
}

OpFoldResult ConvertOp::fold(ArrayRef<Attribute> operands) {
  auto operandTy = getOperand().getType().cast<TensorType>();
  auto resultTy = getResult().getType().cast<TensorType>();
  if (operandTy == resultTy) return getOperand();

  // If the result has non-static shape, a convert op is necessary to go from
  // static shape to non-static shape.
  if (!resultTy.hasStaticShape()) return {};

  // If the operand is constant, we can do the conversion now.
  if (auto elementsAttr = operands.front().dyn_cast_or_null<ElementsAttr>()) {
    return hlo::ConvertElementsAttr(elementsAttr,
                                    getElementTypeOrSelf(getResult()));
  }

  return {};
}

namespace {

struct EliminateRedundantConvert : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern<ConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter& rewriter) const override {
    auto convertOp = op.operand().getDefiningOp<ConvertOp>();
    if (!convertOp) {
      return failure();
    }
    auto firstType =
        convertOp.operand().getType().cast<TensorType>().getElementType();
    auto secondType =
        op.operand().getType().cast<TensorType>().getElementType();
    auto thirdType =
        op.getResult().getType().cast<TensorType>().getElementType();
    auto loc = rewriter.getFusedLoc({convertOp->getLoc(), op->getLoc()});
    if (firstType.isa<FloatType>() && secondType.isa<FloatType>() &&
        thirdType.isa<FloatType>()) {
      // fold when the second float type's width is longer than first,
      // like fp16 -> fp32 -> fp64, bf16 -> fp32 -> fp16
      if (secondType.cast<FloatType>().getWidth() >
          firstType.cast<FloatType>().getWidth()) {
        Value result = rewriter.create<ConvertOp>(loc, op.getResult().getType(),
                                                  convertOp.operand());
        rewriter.replaceOp(op, result);
        return success();
      }
    } else if (firstType.isa<IntegerType>() && secondType.isa<IntegerType>() &&
               thirdType.isa<IntegerType>()) {
      // fold when the second integer type's width is longer than first,
      // like i16 -> i32 -> i64, u16 -> i32 -> u32
      if (secondType.cast<IntegerType>().getWidth() >
          firstType.cast<IntegerType>().getWidth()) {
        Value result = rewriter.create<ConvertOp>(loc, op.getResult().getType(),
                                                  convertOp.operand());
        rewriter.replaceOp(op, result);
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

void ConvertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<EliminateIdentityConvert>(context);
  results.add<EliminateRedundantConvert>(context);
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleElementOp::verify() {
  auto indexVal = index();
  auto operandType = getOperand().getType().cast<TupleType>();
  if (indexVal >= operandType.size()) {
    return emitOpError(
        llvm::formatv("index {0} is out of bounds of operand with size {1}",
                      indexVal, operandType.size()));
  }

  auto expectedType = operandType.getType(indexVal);
  if (getType() != expectedType) {
    return emitOpError(llvm::formatv("has return type {0}, but expected {1}",
                                     getType(), expectedType));
  }
  return success();
}

OpFoldResult GetTupleElementOp::fold(ArrayRef<Attribute> operands) {
  if (auto tupleOp = getOperand().getDefiningOp<mhlo::TupleOp>()) {
    return tupleOp.getOperand(index());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

LogicalResult TupleOp::verify() {
  auto opType = getType().dyn_cast<TupleType>();
  if (!opType) return emitOpError("tuple op with non-tuple result");
  if (getNumOperands() != opType.size())
    return emitOpError(
        "number of operands to tuple expected to match number of types in "
        "resultant tuple type");
  for (const auto& it :
       llvm::enumerate(llvm::zip_first(getOperandTypes(), opType.getTypes()))) {
    if (std::get<0>(it.value()) != std::get<1>(it.value()))
      return emitOpError("has return type mismatch at ")
             << it.index() << "th value (" << std::get<0>(it.value())
             << " != " << std::get<1>(it.value()) << ")";
  }
  return success();
}

namespace {

// Pattern for unpacking and repacking the same tuple.
struct UnpackRepackSameTuple : public OpRewritePattern<TupleOp> {
  using OpRewritePattern<TupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleOp op,
                                PatternRewriter& rewriter) const override {
    if (op.val().empty()) return failure();

    Value firstElement = op.val().front();
    auto firstElementOp = firstElement.getDefiningOp<GetTupleElementOp>();
    if (!firstElementOp || firstElementOp.indexAttr().getInt() != 0)
      return failure();

    Value tuplePredecessor = firstElementOp.getOperand();
    if (tuplePredecessor.getType() != op.getType()) return failure();

    for (const auto& elementAndIdx : llvm::enumerate(op.val().drop_front(1))) {
      auto elementOp = elementAndIdx.value().getDefiningOp<GetTupleElementOp>();
      if (!elementOp ||
          elementOp.indexAttr().getInt() != elementAndIdx.index() + 1 ||
          elementOp.getOperand() != tuplePredecessor)
        return failure();
    }

    rewriter.replaceOp(op, tuplePredecessor);
    return success();
  }
};

}  // namespace

void TupleOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<UnpackRepackSameTuple>(context);
}

//===----------------------------------------------------------------------===//
// AllToAllOp
//===----------------------------------------------------------------------===//

LogicalResult AllToAllOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  AllToAllOp::Adaptor adaptor(operands, attributes, regions);
  Type operandType = adaptor.operand().getType();
  RankedTensorType operandRankedType = operandType.dyn_cast<RankedTensorType>();
  if (!operandRankedType) {
    inferredReturnShapes.emplace_back(
        operandType.cast<TensorType>().getElementType());
    return success();
  }

  int64_t inputRank = operandRankedType.getRank();
  int64_t splitDimension = static_cast<int64_t>(adaptor.split_dimension());
  int64_t concatDimension = static_cast<int64_t>(adaptor.concat_dimension());
  if (splitDimension >= inputRank || splitDimension < 0) {
    return emitOptionalError(location, "AllToAll split_dimension ",
                             splitDimension,
                             " is out-of-bounds for input rank ", inputRank);
  }
  if (concatDimension >= inputRank || concatDimension < 0) {
    return emitOptionalError(location, "AllToAll concat_dimension ",
                             concatDimension,
                             " is out-of-bounds for input rank ", inputRank);
  }

  // If operand is ranked, size of split dimension should be a multiple of split
  // count.
  int64_t splitCount = adaptor.split_count();
  auto splitDimSize = operandRankedType.getDimSize(splitDimension);
  if (splitDimSize % splitCount != 0) {
    return emitOptionalError(
        location, "split dimension has size ", splitDimSize,
        ", expected to be a multiple of split_count ", splitCount);
  }
  SmallVector<int64_t> resultShape(operandRankedType.getShape().begin(),
                                   operandRankedType.getShape().end());
  resultShape[splitDimension] /= splitCount;
  resultShape[concatDimension] *= splitCount;
  inferredReturnShapes.emplace_back(resultShape,
                                    operandRankedType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

LogicalResult AllGatherOp::verify() {
  // If operand and result are both ranked, then the size of the gather
  // dimension in the result should be a multiple of the size of the gather
  // dimension in the operand.
  auto operandType = operand().getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();
  uint64_t allGatherDimIndex = all_gather_dim();
  if (!operandType || !resultType ||
      operandType.isDynamicDim(allGatherDimIndex) ||
      resultType.isDynamicDim(allGatherDimIndex))
    return success();
  if (operandType.getDimSize(allGatherDimIndex) == 0)
    return emitOpError() << "operand gather dimension cannot be zero.";
  if ((resultType.getDimSize(allGatherDimIndex) %
       operandType.getDimSize(allGatherDimIndex)) != 0)
    return emitOpError()
           << "result gather dimension has size "
           << resultType.getDimSize(allGatherDimIndex)
           << ", expected to be a multiple of operand gather dimension size "
           << operandType.getDimSize(allGatherDimIndex);

  return success();
}

//===----------------------------------------------------------------------===//
// BatchNormGradOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormGradOp::verify() {
  // The following properties are already enforced by the ODS:
  //  1. Inputs 'operand' & 'grad_output' and outputs 'grad_operand',
  //     are ranked-tensors with floating-point (fp) type.
  //  2. The shapes of inputs 'operand' & 'grad_output' match.
  //  3. Inputs 'scale', 'mean', 'variance' and Outputs 'grad_scale',
  //     'grad_offset'  are all 1D fp tensors with same shape.
  //  4. The element-types of input 'operand' and outputs 'grad_scale',
  //     'grad_offset' match.
  //  5. The type of input 'operand' and output 'grad_operand' match.
  //
  // We intend to verify the following properties
  //  P1. Inputs 'operand' & 'grad_output' has the same shape with fp
  //      element-types, ignoring fp-precision : Inferred from (1) & (2).
  //  P2. The feature dimension 'feature_index' is a valid index in 'operand':
  //      Inferred from check C2 below.
  //  P3. Inputs 'scale', 'mean', 'variance' must be 1D tensors with same shape
  //      and fp element-type (ignoring precision) and the number of elements
  //      in its sole-dimension == number of features in the 'operand's
  //      feature-dimension 'feature_index': Inferred from (3) and check C3
  //      below.
  //  P4. Outputs 'grad_scale' & 'grad_offset' are 1D tensors with
  //      element-type == element-type of(operand) and same shape as any of
  //      the inputs 'scale', 'mean', or 'variance': Inferred from (3), (4) and
  //      check C3 below.
  //  P5. The type (shape + element-type) of input 'operand' and
  //      output 'grad_operand' must match: Inferred from (5).

  // C2.
  auto operandType = operand().getType().cast<RankedTensorType>();
  if (static_cast<int64_t>(feature_index()) >= operandType.getRank())
    return emitOpError() << "expects feature_index to be smaller "
                            "than the rank of operand type; got feature_index "
                         << feature_index() << ", and rank "
                         << operandType.getRank() << ".";

  if (static_cast<int64_t>(feature_index()) < 0)
    return emitOpError() << "expects feature_index to be a "
                         << "non-negative number, got "
                         << static_cast<int64_t>(feature_index()) << ".";

  auto gradOutputType = grad_output().getType().cast<RankedTensorType>();
  if (operandType.getRank() != gradOutputType.getRank())
    return emitOpError() << "expects 'operand' and 'grad_output' to have the "
                            "same rank. but got rank(oprand) "
                         << operandType.getRank() << " and rank(grad_output) "
                         << gradOutputType.getRank() << ".";

  // C3.
  const int64_t featureCount = operandType.getShape()[feature_index()];
  const int64_t scaleShape =
      scale().getType().cast<RankedTensorType>().getShape()[0];
  if (scaleShape != featureCount)
    return emitOpError() << "expects the size of scale factor to be "
                            "same as the feature count,"
                            " but the size of scale factor is "
                         << scaleShape << " and the feature count is "
                         << featureCount << ".";

  return success();
}

//===----------------------------------------------------------------------===//
// BatchNormTrainingOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormTrainingOp::verify() {
  // The following properties are already enforced by the ODS:
  //  1. 'operand' and 'output' are ranked tensors.
  //  2. 'scale', 'offset', 'batch_mean', 'batch_var' are 1D tensors.
  //  3. Types of 'operand' and 'output' matches.
  //  4. Same element-types for 'operand', 'batch_mean', & 'batch_var'.
  //  5. Same shapes for 'scale', 'offset', 'batch_mean', & 'batch_var'.

  auto operandType = operand().getType().cast<RankedTensorType>();
  if (static_cast<int64_t>(feature_index()) >= operandType.getRank())
    return emitOpError() << "expects feature_index to be smaller "
                            "than the rank of operand type; got feature_index "
                         << feature_index() << ", and rank "
                         << operandType.getRank() << ".";

  if (static_cast<int64_t>(feature_index()) < 0)
    return emitOpError() << "expects feature_index to be a "
                         << "non-negative number, got "
                         << static_cast<int64_t>(feature_index()) << ".";

  // Note:A valid value of feature-index implies 'operand_type.getRank() >=1'.

  const int64_t featureCount = operandType.getShape()[feature_index()];
  const int64_t scaleShape =
      scale().getType().cast<RankedTensorType>().getShape()[0];
  // Check number of elements in input 'scale' equals feature_count.
  // Together with (5) implies that 'scale', 'offset', 'batch_mean', &
  // 'batch_var' all have the same shape.
  if (scaleShape != featureCount)
    return emitOpError() << "expects the size of scale factor to be "
                            "same as the feature count,"
                            " but the size of scale factor is "
                         << scaleShape << " and the feature count is "
                         << featureCount << ".";

  return success();
}

//===----------------------------------------------------------------------===//
// BatchNormInferenceOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormInferenceOp::verify() {
  // The following properties are already enforced by the ODS:
  //  1. 'operand' and 'result' are ranked tensors.
  //  2. 'scale', 'offset', 'mean', 'variance' are 1D tensors.
  //  3. Types of 'operand' and 'result' matches.
  //  4. Same shapes for 'scale', 'offset', 'mean', & 'variance'.

  auto operandType = operand().getType().cast<RankedTensorType>();
  if (static_cast<int64_t>(feature_index()) >= operandType.getRank())
    return emitOpError() << "expects feature_index to be smaller "
                            "than the rank of operand type; got feature_index "
                         << feature_index() << ", and rank "
                         << operandType.getRank() << ".";

  if (static_cast<int64_t>(feature_index()) < 0)
    return emitOpError() << "expects feature_index to be a "
                         << "non-negative number, got "
                         << static_cast<int64_t>(feature_index()) << ".";

  // Note:A valid value of feature-index implies 'operand_type.getRank() >=1'.

  const int64_t featureCount = operandType.getShape()[feature_index()];
  const int64_t scaleSize =
      scale().getType().cast<RankedTensorType>().getShape()[0];
  // Check number of elements in input 'scale' equals feature_count.
  // Together with (4) implies that 'scale', 'offset', 'mean', &
  // 'variance' all have the same shape.
  if (scaleSize != featureCount)
    return emitOpError() << "expects the size of scale factor to be "
                            "same as the feature count,"
                            " but the size of scale factor is "
                         << scaleSize << " and the feature count is "
                         << featureCount << ".";

  return success();
}

//===----------------------------------------------------------------------===//
// BitcastConvertOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastConvertOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto operandType = operands[0].getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();

  // Only ranked tensors are supported.
  if (!operandType || !resultType) return failure();

  // Shape-changing bitcast convert is not implemented.
  // TODO(kramerb): This could be done by adjusting the last dimension.
  DataLayout dataLayout = DataLayout::closest(*this);
  unsigned operandElementSize =
      dataLayout.getTypeSizeInBits(operandType.getElementType());
  unsigned resultElementSize =
      dataLayout.getTypeSizeInBits(resultType.getElementType());
  if (operandElementSize != resultElementSize) return failure();

  return ::mlir::mhlo::deriveShapeFromOperand(
      &builder, getOperation(), operands.front(), &reifiedReturnShapes);
}

/*
 * We intend to verify the following properties
 * P1. We cannot convert between complex and real types (cf xla)
 * P3. The dimensions of the operand and the target
 * shape must match, except that the shape with the smaller element bitwidth has
 * an appropriately-sized additional innermost dimension, e.g.
 * ... x f32 => [bitcast_convert] => ... x 4 x i8
 * ... x 4 x i8 => [bitcast_convert] => ... x f32
 */
LogicalResult BitcastConvertOp::verify() {
  auto operandTensorType = operand().getType().cast<TensorType>();
  auto targetTensorType = getResult().getType().cast<TensorType>();

  // P1.
  auto targetElt = targetTensorType.getElementType();
  auto operandElt = operandTensorType.getElementType();
  if (targetElt.isa<ComplexType>() != operandElt.isa<ComplexType>()) {
    return emitOpError()
           << "cannot convert between real and complex types, but got: "
           << operandTensorType << " and " << targetTensorType;
  }

  auto targetEltBitwidth = potentiallyComplexBitwidth(targetElt);
  auto operandEltBitwidth = potentiallyComplexBitwidth(operandElt);

  // P2.
  auto operandType = operandTensorType.dyn_cast<RankedTensorType>();
  auto targetType = targetTensorType.dyn_cast<RankedTensorType>();
  if (!operandType || !targetType) return success();

  auto targetShape = targetType.getShape();
  auto operandShape = operandType.getShape();
  ArrayRef<int64_t> smallerEltShape, biggerEltShape;
  Type smallerElt, biggerElt;
  if (operandEltBitwidth < targetEltBitwidth) {
    smallerEltShape = operandShape;
    smallerElt = operandElt;
    biggerEltShape = targetShape;
    biggerElt = targetElt;
  } else {
    smallerEltShape = targetShape;
    smallerElt = targetElt;
    biggerEltShape = operandShape;
    biggerElt = operandElt;
  }

  ArrayRef<int64_t> smallerEltPrefix;
  auto smallerEltBitwidth = std::min(targetEltBitwidth, operandEltBitwidth);
  auto biggerEltBitwidth = std::max(targetEltBitwidth, operandEltBitwidth);
  if (operandEltBitwidth != targetEltBitwidth) {
    if (smallerEltShape.empty()) {
      return emitOpError() << "does not allow the smaller element type to be "
                              "part of a 0d tensor, but got: "
                           << operandType << " and " << targetType << ".";
    }
    smallerEltPrefix = smallerEltShape.drop_back();
    if (!isDynamicDimSize(smallerEltShape.back()) &&
        smallerEltShape.back() * smallerEltBitwidth != biggerEltBitwidth) {
      return emitOpError() << "requires compatible bitwidths. "
                           << "Got: " << operandType << " and " << targetType
                           << ", but " << smallerEltBitwidth << " * "
                           << smallerEltShape.back()
                           << " != " << biggerEltBitwidth << ".";
    }
  } else {
    smallerEltPrefix = smallerEltShape;
  }

  for (auto it : llvm::zip(smallerEltPrefix, biggerEltShape)) {
    auto targetDim = std::get<0>(it);
    auto operandDim = std::get<1>(it);
    if (!isDynamicDimSize(targetDim) && !isDynamicDimSize(operandDim)) {
      if (targetDim != operandDim) {
        return emitOpError() << "operand and result shapes must match except "
                                "for the innermost dimension of the shape with "
                                "the smaller element type. Got: "
                             << operandType << " and " << targetType << ".";
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// TODO(b/129012527) These should be expressed as type constraints.
LogicalResult BroadcastOp::verify() {
  auto sizes = broadcast_sizes();
  auto sizesType = sizes.getType();
  auto sizesRank = sizesType.getRank();
  if (sizesRank != 1) {
    return emitOpError(llvm::formatv(
        "broadcast_sizes has rank {0} instead of rank 1", sizesRank));
  }

  return success();
}

OpFoldResult BroadcastOp::fold(ArrayRef<Attribute> attrs) {
  auto type = getType().cast<RankedTensorType>();
  auto sizesType = broadcast_sizes().getType();
  if (sizesType.getNumElements() == 0) {
    return getOperand();
  }

  // Constant fold when an operand is a splat tensor attribute.
  if (!attrs[0] || !type.hasStaticShape()) return {};
  auto splatOperandAttr = attrs[0].dyn_cast<SplatElementsAttr>();
  if (!splatOperandAttr) return {};

  // Handle complex type
  if (type.getElementType().isa<ComplexType>()) {
    ComplexType complex = type.getElementType().cast<ComplexType>();
    if (complex.getElementType().isa<FloatType>()) {
      return DenseElementsAttr::get(
          type, {splatOperandAttr.getSplatValue<std::complex<APFloat>>()});
    }
    if (complex.getElementType().isa<IntegerType>()) {
      return DenseElementsAttr::get(
          type, {splatOperandAttr.getSplatValue<std::complex<APInt>>()});
    }
    return {};
  }

  return SplatElementsAttr::get(
      type, splatOperandAttr.getSplatValue<mlir::Attribute>());
}

LogicalResult BroadcastOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands, attributes, regions);
  Value operand = adaptor.operand();
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  Type elementTy = operandType.getElementType();
  auto dimensionAttr = adaptor.broadcast_sizes();
  for (int64_t size : dimensionAttr.getValues<int64_t>()) {
    if (size < 0)
      return emitOptionalError(location,
                               "Broadcast with negative dimension size ", size);
  }
  SmallVector<int64_t> shapeValues(dimensionAttr.getValues<int64_t>());
  llvm::append_range(shapeValues, operandType.getShape());

  inferredReturnShapes.emplace_back(shapeValues, elementTy);
  return success();
}

LogicalResult BroadcastOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands);
  Value operand = adaptor.operand();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Unranked tensors are not supported.
  if (!operandType) return failure();

  Location loc = getLoc();
  SmallVector<Value, 4> shapeValues;

  // Collect the broadcast sizes.
  for (const auto& size : broadcast_sizes()) {
    shapeValues.push_back(
        builder.create<arith::ConstantIndexOp>(loc, size.getZExtValue()));
  }

  // Collect the operand sizes.
  for (auto index : llvm::seq<int64_t>(0, operandType.getRank())) {
    shapeValues.push_back(
        builder.createOrFold<tensor::DimOp>(loc, operand, index));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            builder.getIndexType()),
      shapeValues));

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastInDimOp::verify() {
  auto operandType = operand().getType().dyn_cast<RankedTensorType>();
  if (!operandType) {
    // The following verification checks all depend on knowing the rank of
    // the operand. Bail out now if we don't know the rank of the operand.
    return success();
  }

  auto operandRank = operandType.getRank();
  if (!broadcast_dimensions()) {
    if (operandRank == 0) {
      return success();
    }
    return emitOpError(
        llvm::formatv("broadcast_dimensions is absent, but required because "
                      "operand has non-zero rank ({0})",
                      operandRank));
  }

  auto dimensions = broadcast_dimensions();
  auto dimensionsType = broadcast_dimensions().getType();
  auto dimensionsRank = dimensionsType.getRank();
  if (dimensionsRank != 1) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions has rank {0} instead of rank 1", dimensionsRank));
  }

  auto dimensionsSize = dimensionsType.getNumElements();
  if (dimensionsSize != operandRank) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions size ({0}) does not match operand rank ({1})",
        dimensionsSize, operandRank));
  }

  auto resultType = getResult().getType().cast<RankedTensorType>();
  auto resultRank = resultType.getRank();
  if (resultRank < operandRank) {
    return emitOpError(
        llvm::formatv("result rank ({0}) is less than operand rank ({1})",
                      resultRank, operandRank));
  }

  for (int i = 0; i != dimensionsSize; ++i) {
    auto dimIndex = dimensions.getValues<int64_t>()[i];
    if (dimIndex >= resultRank) {
      return emitOpError(
          llvm::formatv("broadcast_dimensions contains invalid value {0} for "
                        "result with rank {1}",
                        dimIndex, resultRank));
    }

    if (!operandType.isDynamicDim(i)) {
      auto dimSize = operandType.getDimSize(i);
      auto resultDimSize = resultType.getDimSize(dimIndex);
      if (dimSize != 1 && dimSize != resultDimSize) {
        return emitOpError(
            llvm::formatv("size of operand dimension {0} ({1}) is not equal to "
                          "1 or size of result dimension {2} ({3})",
                          i, dimSize, dimIndex, resultDimSize));
      }
    }
  }

  return success();
}

OpFoldResult BroadcastInDimOp::fold(ArrayRef<Attribute> attrs) {
  auto type = getType().cast<RankedTensorType>();
  if (type == getOperand().getType()) {
    auto broadcastValues = broadcast_dimensions().getValues<int64_t>();
    if (!std::equal(broadcastValues.begin(), broadcastValues.end(),
                    llvm::seq<int64_t>(0, type.getRank()).begin())) {
      return {};
    }
    return getOperand();
  }

  // Constant fold when an operand is a splat tensor attribute.
  if (!attrs[0] || !type.hasStaticShape()) return {};
  auto splatOperandAttr = attrs[0].dyn_cast<SplatElementsAttr>();
  if (!splatOperandAttr) return {};

  // Handle complex type
  if (type.getElementType().isa<ComplexType>()) {
    ComplexType complex = type.getElementType().cast<ComplexType>();
    if (complex.getElementType().isa<FloatType>()) {
      return DenseElementsAttr::get(
          type, {splatOperandAttr.getSplatValue<std::complex<APFloat>>()});
    }
    if (complex.getElementType().isa<IntegerType>()) {
      return DenseElementsAttr::get(
          type, {splatOperandAttr.getSplatValue<std::complex<APInt>>()});
    }
    return {};
  }

  return SplatElementsAttr::get(
      type, splatOperandAttr.getSplatValue<mlir::Attribute>());
}

// Simplify BroadcastInDim has the following behaviors: replace BroadcastInDim
// with Reshape or Transpose if they are equivalent or replace
// BroadcastInDim(BroadcastInDim(X)) with BroadcastInDim(X)
class BroadcastInDimSimplifier : public OpRewritePattern<BroadcastInDimOp> {
 public:
  using OpRewritePattern<BroadcastInDimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.operand().getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!operandType || !resultType) {
      return failure();
    }
    auto bsDimIndices = op.broadcast_dimensions().getValues<int64_t>();
    if (operandType.hasStaticShape() && resultType.hasStaticShape()) {
      bool sameTotalElements =
          operandType.getNumElements() == resultType.getNumElements();
      // BroadcastInDim equivalent to reshape
      if (llvm::is_sorted(bsDimIndices) && sameTotalElements) {
        rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.operand());
        return success();
      }
      // BroadcastInDim equivalent to transpose
      if (operandType.getRank() == resultType.getRank() && sameTotalElements) {
        rewriter.replaceOpWithNewOp<TransposeOp>(op, op.getType(), op.operand(),
                                                 op.broadcast_dimensions());
        return success();
      }
    }
    // eliminate redundant BroadcastInDim
    if (auto broadcastInDimOp = llvm::dyn_cast_or_null<BroadcastInDimOp>(
            op.operand().getDefiningOp())) {
      auto newIndices =
          broadcastInDimOp.broadcast_dimensions()
              .mapValues(op.broadcast_dimensions().getElementType(),
                         [&bsDimIndices](const APInt& dim) -> APInt {
                           return APInt(dim.getBitWidth(),
                                        bsDimIndices[dim.getSExtValue()], true);
                         })
              .cast<DenseIntElementsAttr>();
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
          op, op.getType(), broadcastInDimOp.operand(), newIndices);
      return success();
    }
    return failure();
  }
};

void BroadcastInDimOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                   MLIRContext* context) {
  results.add<BroadcastInDimSimplifier>(context);
}

//===----------------------------------------------------------------------===//
// DynamicBroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicBroadcastInDimOp::verify() {
  auto operandType = operand().getType().dyn_cast<RankedTensorType>();
  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();

  // If either the operand or result are unranked, there is very little
  // to verify statically.
  if (!operandType || !resultType) {
    return success();
  }

  auto outputDimensionsType =
      output_dimensions().getType().cast<RankedTensorType>();
  auto outputDimensionsSize = outputDimensionsType.getDimSize(0);
  auto operandRank = operandType.getRank();
  auto resultRank = resultType.getRank();

  // Verify broadcast_dimensions.
  auto bcastDimensions = broadcast_dimensions();
  auto bcastDimensionsType = broadcast_dimensions().getType();
  auto bcastDimensionsRank = bcastDimensionsType.getRank();
  // TODO(laurenzo): Update the BroadcastDimAttr to constrain its rank to 1.
  if (bcastDimensionsRank != 1) {
    return emitOpError(
        llvm::formatv("broadcast_dimensions has rank {0} instead of rank 1",
                      bcastDimensionsRank));
  }

  auto bcastDimensionsSize = bcastDimensionsType.getNumElements();
  if (bcastDimensionsSize != operandRank) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions size ({0}) does not match operand rank ({1})",
        bcastDimensionsSize, operandRank));
  }

  if (resultRank < operandRank) {
    return emitOpError(
        llvm::formatv("result rank ({0}) is less than operand rank ({1})",
                      resultRank, operandRank));
  }

  for (int i = 0; i != bcastDimensionsSize; ++i) {
    auto dimIndex = bcastDimensions.getValues<int64_t>()[i];
    if (dimIndex >= resultRank) {
      return emitOpError(
          llvm::formatv("broadcast_dimensions contains invalid value {0} for "
                        "result with rank {1}",
                        dimIndex, resultRank));
    }

    auto dimSize = operandType.getDimSize(i);
    auto resultDimSize = resultType.getDimSize(dimIndex);
    // Note: verifyCompatibleShapes doesn't consider size-1 broadcasting, so we
    // add a manual check for this.
    if (dimSize != 1 && failed(verifyCompatibleShape(dimSize, resultDimSize))) {
      return emitOpError(
          llvm::formatv("size of operand dimension {0} ({1}) is not compatible "
                        "with size of result dimension {2} ({3})",
                        i, dimSize, dimIndex, resultDimSize));
    }
  }

  if (outputDimensionsSize != resultRank) {
    return emitOpError(
        llvm::formatv("result rank ({0}) is not equal to number of output "
                      "dimensions ({1})",
                      resultRank, outputDimensionsSize));
  }

  return success();
}

namespace {
// If a DynamicBroadCastInDimOp is not actually dynamic, use an ordinary
// BroadcastInDimOp.
class DynamicBroadcastInDimOpNotActuallyDynamic
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto type = op.getType().dyn_cast<RankedTensorType>();
    auto operandType = op.operand().getType().dyn_cast<RankedTensorType>();
    auto outputDimOp = op.output_dimensions().getDefiningOp();
    if (!type || !operandType || !operandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires operand static shape");
    }
    // output has static shape, replace with broadcast_in_dim
    if (type.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(op, type, op.operand(),
                                                    op.broadcast_dimensions());
      return success();
    }
    // output_dimensions are constant, set output shape with output_dimensions,
    // then replace with broadcast_in_dim
    if (outputDimOp && outputDimOp->hasTrait<mlir::OpTrait::ConstantLike>()) {
      DenseIntElementsAttr shapeAttr;
      if (matchPattern(outputDimOp, m_Constant(&shapeAttr))) {
        SmallVector<int64_t> outputShape;
        for (APInt shape : shapeAttr.getValues<APInt>()) {
          outputShape.push_back(shape.getZExtValue());
        }
        Value result = rewriter.create<BroadcastInDimOp>(
            op.getLoc(),
            RankedTensorType::get(outputShape, type.getElementType()),
            op.operand(), op.broadcast_dimensions());
        // We are refining the type here. Not all operations can tolerate their
        // operands changing type. Operations from mhlo dialect can. So insert
        // a cast otherwise.
        if (llvm::any_of(op->getUsers(), [&](Operation* user) {
              return user->getDialect() != op->getDialect();
            })) {
          result = rewriter.create<tensor::CastOp>(
              op.getLoc(), op.getResult().getType(), result);
        }
        rewriter.replaceOp(op, result);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(
        op, "requires output static shape or constant broadcast dimensions");
  }
};

class ChainedDynamicBroadcastInDimCanonicalization
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp bcast,
                                PatternRewriter& rewriter) const override {
    auto precedingBcast =
        bcast.operand().getDefiningOp<DynamicBroadcastInDimOp>();
    if (!precedingBcast) return failure();

    // Compose broadcast dimensions.
    DenseIntElementsAttr precedingBcastDims =
        precedingBcast.broadcast_dimensions();
    DenseIntElementsAttr bcastDims = bcast.broadcast_dimensions();
    SmallVector<APInt, 4> composition;
    for (APInt precedingDim : precedingBcastDims) {
      composition.push_back(
          bcastDims.getValues<APInt>()[precedingDim.getZExtValue()]);
    }
    auto composedBcastDims =
        DenseIntElementsAttr::get(precedingBcastDims.getType(), composition);

    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        bcast, bcast.getType(), precedingBcast.operand(),
        bcast.output_dimensions(), composedBcastDims);
    return success();
  }
};
}  // namespace

void DynamicBroadcastInDimOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
  results.add<ChainedDynamicBroadcastInDimCanonicalization,
              DynamicBroadcastInDimOpNotActuallyDynamic,
              DynamicBroadcastToOwnShape_1, DynamicBroadcastToOwnShape_2,
              DynamicBroadcastToOwnShape_3, DynamicBroadcastToOwnShape_4>(
      context);
}

LogicalResult DynamicBroadcastInDimOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicBroadcastInDimOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.output_dimensions()));
  return success();
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

LogicalResult ClampOp::verify() {
  auto operandType = operand().getType().cast<RankedTensorType>();
  auto operandShape = operandType.getShape();
  auto minType = min().getType().cast<RankedTensorType>();

  auto minShape = minType.getShape();
  if (minShape != operandShape && minType.getRank() != 0) {
    return emitOpError(llvm::formatv(
        "min shape [{0}] is not scalar and does not match operand shape [{1}]",
        llvm::make_range(minShape.begin(), minShape.end()),
        llvm::make_range(operandShape.begin(), operandShape.end())));
  }

  auto maxType = max().getType().cast<RankedTensorType>();
  auto maxShape = maxType.getShape();
  if (maxShape != operandShape && maxType.getRank() != 0) {
    return emitOpError(llvm::formatv(
        "max shape [{0}] is not scalar and does not match operand shape [{1}]",
        llvm::make_range(maxShape.begin(), maxShape.end()),
        llvm::make_range(operandShape.begin(), operandShape.end())));
  }

  return success();
}

LogicalResult ClampOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  // For `mhlo.clamp`, the first operand may be a scalar.
  return deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// ComplexOp
//===----------------------------------------------------------------------===//

LogicalResult ComplexOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto type = operands[0].getType();
  auto elementTy = ComplexType::get(getElementTypeOrSelf(type));
  Type resultTy;
  if (auto rankedType = type.dyn_cast<RankedTensorType>()) {
    resultTy = RankedTensorType::get(rankedType.getShape(), elementTy,
                                     rankedType.getEncoding());
  } else if (type.isa<UnrankedTensorType>()) {
    resultTy = UnrankedTensorType::get(elementTy);
  } else {
    resultTy = elementTy;
  }
  inferredReturnTypes.push_back(resultTy);
  return success();
}

OpFoldResult ComplexOp::fold(ArrayRef<Attribute> operands) {
  auto realOp = getOperand(0).getDefiningOp<mhlo::RealOp>();
  auto imagOp = getOperand(1).getDefiningOp<mhlo::ImagOp>();
  if (realOp && imagOp && realOp.getOperand() == imagOp.getOperand()) {
    return realOp.getOperand();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ImagOp
//===----------------------------------------------------------------------===//

namespace {
Type createRealType(Type type) {
  auto elementTy = getElementTypeOrSelf(type);
  if (auto complexTy = elementTy.dyn_cast<ComplexType>()) {
    elementTy = complexTy.getElementType();
  }

  if (auto rankedType = type.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(rankedType.getShape(), elementTy,
                                 rankedType.getEncoding());
  }
  if (type.dyn_cast<UnrankedTensorType>()) {
    return UnrankedTensorType::get(elementTy);
  }

  return elementTy;
}
}  // namespace

LogicalResult ImagOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(createRealType(operands[0].getType()));
  return success();
}

OpFoldResult ImagOp::fold(ArrayRef<Attribute> operands) {
  if (auto complexOp = getOperand().getDefiningOp<mhlo::ComplexOp>()) {
    return complexOp.getOperand(1);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

TensorType getSameShapeTensorType(TensorType tensorType, Type elementType) {
  if (auto rankedTensorTy = tensorType.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(rankedTensorTy.getShape(), elementType);
  }
  if (auto unrankedTensorTy = tensorType.dyn_cast<UnrankedTensorType>()) {
    return UnrankedTensorType::get(elementType);
  }
  llvm_unreachable("unhandled type");
}

LogicalResult IsFiniteOp::inferReturnTypes(
    MLIRContext* ctx, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto argTy = operands.front().getType().cast<TensorType>();
  Builder b(ctx);
  inferredReturnTypes.push_back(getSameShapeTensorType(argTy, b.getI1Type()));
  return success();
}

//===----------------------------------------------------------------------===//
// RealOp
//===----------------------------------------------------------------------===//

LogicalResult RealOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(createRealType(operands[0].getType()));
  return success();
}

OpFoldResult RealOp::fold(ArrayRef<Attribute> operands) {
  if (auto complexOp = getOperand().getDefiningOp<mhlo::ComplexOp>()) {
    return complexOp.getOperand(0);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

namespace {
class SingleOperandConcatenateToCast : public OpRewritePattern<ConcatenateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    if (op.val().size() != 1) return failure();

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                                op.val().front());
    return success();
  }
};

class ConcatenateOperandRemoval : public OpRewritePattern<ConcatenateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    auto axis = op.dimension();
    llvm::SmallVector<Value, 6> newOperands;
    for (auto operand : op.getOperands()) {
      auto ty = operand.getType().cast<ShapedType>();
      if (!ty.hasRank() || ty.getDimSize(axis) != 0) {
        newOperands.push_back(operand);
      }
    }

    if (!newOperands.empty() && newOperands.size() < op.getNumOperands()) {
      rewriter.replaceOpWithNewOp<ConcatenateOp>(op, op.getResult().getType(),
                                                 newOperands, op.dimension());
      return success();
    }

    return failure();
  }
};

class ConcatenateForwarding : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    auto getFlattenedOperands = [&](const Value& val) -> ValueRange {
      auto definingOp = dyn_cast_or_null<ConcatenateOp>(val.getDefiningOp());
      // To avoid inflate the memory footprint, only flatten the ConcatenateOp
      // when it has only one use.
      if (definingOp && definingOp->hasOneUse() &&
          definingOp.dimension() == op.dimension())
        return definingOp.val();
      return val;
    };

    bool needToFlatten = false;
    int operandCount = 0;
    llvm::for_each(op.val(), [&](Value val) {
      auto result = getFlattenedOperands(val);
      if (result.size() != 1 || result[0] != val) needToFlatten = true;
      operandCount += result.size();
    });

    if (!needToFlatten) return failure();

    llvm::SmallVector<Value, 6> newOperands;
    newOperands.reserve(operandCount);

    for (auto operand : op.val()) {
      auto flattenedOperands = getFlattenedOperands(operand);
      newOperands.append(flattenedOperands.begin(), flattenedOperands.end());
    }

    rewriter.replaceOpWithNewOp<ConcatenateOp>(op, op.getResult().getType(),
                                               newOperands, op.dimension());
    return success();
  }
};

}  // namespace

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext*, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  if (operands.empty()) {
    return failure();
  }

  auto dimensionAttr = attributes.get("dimension").cast<IntegerAttr>();
  auto dimension = dimensionAttr.getInt();

  auto firstType = (*operands.begin()).getType().cast<ShapedType>();
  auto outElement = firstType.getElementType();

  for (auto operand : operands.getTypes()) {
    auto elementType = getElementTypeOrSelf(operand);
    if (elementType != outElement) {
      return failure();
    }
  }

  // Find the first ranked input to determine the output rank.
  for (auto type : operands.getTypes()) {
    auto shapedType = type.cast<ShapedType>();
    if (shapedType.hasRank()) {
      firstType = shapedType;
      break;
    }
  }

  // If all inputs are unranked, the result must be unranked.
  if (!firstType.hasRank()) {
    inferredReturnTypes.push_back(UnrankedTensorType::get(outElement));
    return success();
  }

  if (firstType.getRank() == 0)
    return emitOptionalError(location, "rank-0 values cannot be concatenated");

  auto outShape = llvm::to_vector<6>(firstType.getShape());

  // Determine what the non-concatenate dimensions should be.
  for (auto type : operands.getTypes()) {
    auto shapedTy = type.cast<ShapedType>();
    if (!shapedTy.hasRank()) {
      continue;
    }

    for (const auto& it : llvm::enumerate(shapedTy.getShape())) {
      // If a dimension is not dynamic, the output shape should match.
      if (ShapedType::isDynamic(outShape[it.index()])) {
        outShape[it.index()] = it.value();
      }
    }
  }

  outShape[dimension] = 0;

  for (auto operand : operands.getTypes()) {
    auto type = operand.cast<ShapedType>();
    if (!type.hasRank()) {
      inferredReturnTypes.push_back(UnrankedTensorType::get(outElement));
      return success();
    }

    // If the dimension is dynamic we know the output dimension is dynamic.
    auto dim = type.getShape()[dimension];
    if (dim == -1) {
      outShape[dimension] = -1;
      break;
    }

    outShape[dimension] += dim;
  }

  inferredReturnTypes.push_back(RankedTensorType::get(outShape, outElement));

  return success();
}

void ConcatenateOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<ConcatenateOperandRemoval, ConcatenateForwarding,
              SingleOperandConcatenateToCast>(context);
}

template <typename T>
static Attribute foldConcatenateHelper(ConcatenateOp* op,
                                       ArrayRef<Attribute> operands) {
  auto axis = op->dimension();
  auto type = op->getType().cast<ShapedType>();
  auto shape = type.getShape();

  size_t topSize = 1;
  for (int i = 0, e = axis; i < e; i++) {
    topSize = topSize * shape[i];
  }

  // TODO(b/210478841): Define a constant folding policy that generalizes this.
  if (type.getNumElements() * op->getNumOperands() > UINT32_MAX) {
    return {};
  }

  SmallVector<T, 6> values;
  for (size_t i = 0; i < topSize; i++) {
    for (auto operand : operands) {
      DenseElementsAttr attr = operand.cast<DenseElementsAttr>();
      size_t bottomSize = attr.getNumElements() / topSize;
      auto iter = attr.getValues<T>().begin() + i * bottomSize;
      values.append(iter, iter + bottomSize);
    }
  }

  return DenseElementsAttr::get(type, values);
}

static Attribute foldConcatenate(ConcatenateOp* op,
                                 ArrayRef<Attribute> operands) {
  for (auto operand : operands) {
    if (!operand) return {};
  }

  auto type = op->getResult().getType().cast<ShapedType>();
  auto etype = type.getElementType();
  if (etype.isa<IntegerType>()) {
    return foldConcatenateHelper<APInt>(op, operands);
  }

  if (etype.isa<FloatType>()) {
    return foldConcatenateHelper<APFloat>(op, operands);
  }

  return {};
}

OpFoldResult ConcatenateOp::fold(ArrayRef<Attribute> operands) {
  if (getNumOperands() == 1) return getOperand(0);

  ShapedType type = getResult().getType().cast<ShapedType>();
  if (!type.hasStaticShape()) return {};

  auto axis = dimension();
  if (auto attr = foldConcatenate(this, operands)) {
    return attr;
  }

  for (auto operand : getOperands()) {
    auto ty = operand.getType().cast<ShapedType>();
    if (ty.getDimSize(axis) != 0) {
      return {};
    }
  }

  return DenseElementsAttr::get(type, ArrayRef<Attribute>());
}

LogicalResult ConcatenateOp::verify() {
  Type elementType = getElementTypeOrSelf(getOperand(0).getType());
  RankedTensorType firstRankedType;
  int numOperands = getNumOperands();
  for (int i = 0; i < numOperands; i++) {
    auto secondType = getOperand(i).getType().dyn_cast<ShapedType>();
    if (secondType.getElementType() != elementType) {
      return emitOpError(
          llvm::formatv("operands (0) and ({0}) do not match element type", i));
    }

    if (!secondType.hasRank()) {
      continue;
    }

    if (!firstRankedType) {
      firstRankedType = secondType.cast<RankedTensorType>();
      continue;
    }

    if (firstRankedType.getRank() != secondType.getRank()) {
      return emitOpError(
          llvm::formatv("operands (0) and ({0}) do not match rank", i));
    }

    auto firstShape = secondType.getShape();
    auto secondShape = secondType.getShape();
    for (int d = 0; d < firstRankedType.getRank(); ++d) {
      if (firstShape[d] != secondShape[d] && d != dimension()) {
        return emitOpError(llvm::formatv(
            "operands (0) and ({0}) non-concat dimensions do not match "
            "({1}) != ({2})",
            i, llvm::make_range(firstShape.begin(), firstShape.end()),
            llvm::make_range(secondShape.begin(), secondShape.end())));
      }
    }
  }
  return success();
}

LogicalResult ConcatenateOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  ConcatenateOp::Adaptor adaptor(operands);
  auto inputs = adaptor.val();

  auto operandType = inputs[0].getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  SmallVector<SmallVector<Value, 4>, 4> allShapeValues;
  for (size_t inputId = 0; inputId < inputs.size(); ++inputId) {
    Value operand = inputs[inputId];
    auto operandType = operand.getType().dyn_cast<RankedTensorType>();
    if (!operandType) return failure();

    SmallVector<Value, 4> shapeVals;
    for (const auto& element : llvm::enumerate(operandType.getShape())) {
      Value valueDim = toShapeScalarType(
          builder.create<tensor::DimOp>(loc, operand, element.index()));
      shapeVals.push_back(valueDim);
    }
    allShapeValues.emplace_back(std::move(shapeVals));
  }

  int axis = this->dimension();
  auto& shapeValues = allShapeValues[0];
  for (size_t vecId = 1; vecId < allShapeValues.size(); ++vecId) {
    auto& otherShapeValues = allShapeValues[vecId];
    if (otherShapeValues.size() != shapeValues.size()) {
      this->emitOpError()
          << "Concatenate expects all operands must be of the same rank";
      return failure();
    }
    shapeValues[axis] = builder.create<arith::AddIOp>(loc, shapeValues[axis],
                                                      otherShapeValues[axis]);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

//===----------------------------------------------------------------------===//
// DynamicReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicReshapeOp::verify() {
  auto resultType = result().getType().dyn_cast<RankedTensorType>();
  auto outputShapeType = output_shape().getType().dyn_cast<RankedTensorType>();
  if (resultType && outputShapeType && outputShapeType.hasStaticShape() &&
      outputShapeType.getDimSize(0) != resultType.getRank()) {
    return emitError() << "output should have a rank equal to the number of "
                          "elements in output_shape";
  }
  return success();
}

LogicalResult DynamicReshapeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicReshapeOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.output_shape()));
  return success();
}

namespace {
class DynamicReshapeOpNotActuallyDynamic
    : public OpRewritePattern<DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto type = op.result().getType().dyn_cast<RankedTensorType>();
    if (!type || !type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires static shape tensor");
    }
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.operand());
    return success();
  }
};

// Canonicalizes
// %0 = some_op(%tensor)
// %1 = "mhlo.dynamic_reshape"(%0, %shape)
//      (tensor<?xT>, tensor<1xindex>) -> tensor<?xT>
// ... uses of %1.
//
// into
//
// ... uses of %0.
// This canonicalization is only correct if the input is correct!
// TODO(b/178779691): Use a more sophisticated canonicalization that preserves
// errors in input, and still allows us to get rid of redundant reshapes.
class RemoveRedundantRank1DynamicReshape
    : public OpRewritePattern<DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto type = op.result().getType().dyn_cast<RankedTensorType>();
    if (!type || type.getRank() != 1 || type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "requires rank 1 shape tensor with dynamic dimension");
    }
    auto operandType = op.operand().getType().dyn_cast<RankedTensorType>();
    if (!operandType || operandType.getRank() != 1 ||
        operandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "requires rank 1 shape tensor with dynamic dimension");
    }
    rewriter.replaceOp(op, {op.operand()});
    return success();
  }
};

// Canonicalizes
// %0 = "mhlo.dynamic_reshape"(%tensor, %shape)
// %1 = same_operands_and_result_shape_op(%tensor)
// %2 = "mhlo.dynamic_reshape"(%1, %shape)
// ... uses of %2.
//
// into
//
// %0 = "mhlo.dynamic_reshape"(%tensor, %shape)
// %1 = same_operands_and_result_shape_op(%tensor)
// ... uses of %1.
class DynamicReshapeOpSameShapeOpResult
    : public OpRewritePattern<DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* defOp = op.operand().getDefiningOp();
    if (!defOp ||
        !defOp->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) {
      return failure();
    }
    Operation* inputDefOp = defOp->getOperand(0).getDefiningOp();
    if (!inputDefOp) {
      return failure();
    }
    auto reshape = dyn_cast<DynamicReshapeOp>(*inputDefOp);
    if (reshape && reshape.output_shape() == op.output_shape()) {
      rewriter.replaceOp(op, {defOp->getResult(0)});
      return success();
    }
    return failure();
  }
};
}  // namespace

void DynamicReshapeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                   MLIRContext* context) {
  // clang-format off
  results.add<
      DynamicReshapeOpNotActuallyDynamic,
      DynamicReshapeOpSameShapeOpResult,
      RemoveRedundantDynamicBroadcast,
      RemoveRedundantDynamicReshape,
      RemoveRedundantRank1DynamicReshape,
      ShapeOfDynamicReshape
    >(context);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// DynamicSliceOp
//===----------------------------------------------------------------------===//

namespace {
// Canonicalizes DynamicSlice ops that can be replaced instead with Slice ops.
// This canonicalization is applied the case when the `begin` input values are
// compile time constants and thus can be made into a tensor.
struct DynamicSliceToSlice : public OpRewritePattern<DynamicSliceOp> {
  using OpRewritePattern<DynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicSliceOp dynamicSlice,
                                PatternRewriter& rewriter) const override {
    Value input = dynamicSlice.operand();
    auto inputTensor = input.getType().dyn_cast<RankedTensorType>();
    if (!inputTensor || !inputTensor.hasStaticShape()) return failure();

    auto sliceSizes = dynamicSlice.slice_sizes().getValues<int64_t>();
    SmallVector<int64_t, 4> tempStartIndices;
    for (const auto& indexAndSliceStart :
         llvm::enumerate(dynamicSlice.start_indices())) {
      APInt val;
      Value start = indexAndSliceStart.value();
      int64_t index = indexAndSliceStart.index();
      if (!matchPattern(start, m_ConstantInt(&val))) {
        return failure();
      }
      // Clamp the indices within bounds to faithfully mirror dynamic slice
      // semantics.
      int64_t clampedStart =
          clamp(val.getSExtValue(), static_cast<int64_t>(0),
                inputTensor.getDimSize(index) - sliceSizes[index]);
      tempStartIndices.push_back(clampedStart);
    }

    // At this point we've determined that the start indices are all constants;
    // pack them into a single tensor.
    auto loc = dynamicSlice.getLoc();
    int64_t inputRank = inputTensor.getRank();
    auto sliceStartIndices = rewriter.getI64TensorAttr(tempStartIndices);
    DenseIntElementsAttr sliceLimits = buildSliceLimits(
        sliceStartIndices, dynamicSlice.slice_sizes(), &rewriter);
    DenseIntElementsAttr sliceStrides =
        rewriter.getI64TensorAttr(SmallVector<int64_t, 4>(inputRank, 1));
    auto result = rewriter.create<SliceOp>(loc, input, sliceStartIndices,
                                           sliceLimits, sliceStrides);
    rewriter.replaceOp(dynamicSlice, {result});
    return success();
  }
};

}  // namespace

void DynamicSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                 MLIRContext* context) {
  results.add<DynamicSliceToSlice>(context);
}

// Verifies that the number of slice sizes and the number of start indices match
LogicalResult DynamicSliceOp::verify() {
  int numSliceSizes = slice_sizes().getNumElements();
  int numStartIndices = start_indices().size();
  if (numStartIndices != numSliceSizes) {
    return emitOpError() << "has mismatched number of slice sizes ("
                         << numSliceSizes << ") and number of start indices ("
                         << numStartIndices << ")";
  }
  auto operandType = operand().getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  if (operandType.getRank() != numStartIndices) {
    return emitOpError() << "has mismatched number of start indices ("
                         << numStartIndices << ") and the rank of operand ("
                         << operandType.getRank() << ")";
  }

  for (int i = 0; i < numSliceSizes; ++i) {
    int64_t slice_size = slice_sizes().getValues<int64_t>()[i];
    if (slice_size < 0) {
      return emitOpError() << "has negative size index to dynamic slice: "
                           << slice_size;
    } else if (!operandType.isDynamicDim(i)) {
      int64_t dim_size = operandType.getDimSize(i);
      if (slice_size > dim_size) {
        return emitOpError() << "has slice size " << slice_size
                             << " greater than dimension size " << dim_size
                             << " in dimension " << i << " of operand";
      }
    }
  }
  return success();
}

LogicalResult DynamicSliceOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, regions);
  Value operand = adaptor.operand();
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  auto sliceSizes = adaptor.slice_sizes();
  Type elementTy = operandType.getElementType();
  inferredReturnShapes.emplace_back(sliceSizes.getValues<int64_t>(), elementTy);
  return success();
}

//===----------------------------------------------------------------------===//
// RealDynamicSliceOp
//===----------------------------------------------------------------------===//
// Verifies that operand rank matches start_indices/limit_indices/strides size
LogicalResult RealDynamicSliceOp::verify() {
  auto inputType = operand().getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!inputType) return success();
  int inputRank = inputType.getRank();

  auto startType = start_indices().getType().cast<RankedTensorType>();
  auto limitType = limit_indices().getType().cast<RankedTensorType>();
  auto stridesType = strides().getType().cast<RankedTensorType>();

  if (inputRank != startType.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << inputRank << ") and start_indices size ("
                         << startType.getNumElements() << ")";
  }

  if (inputRank != limitType.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << inputRank << ") and limit_indices size ("
                         << limitType.getNumElements() << ")";
  }

  if (inputRank != stridesType.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << inputRank << ") and strides size ("
                         << stridesType.getNumElements() << ")";
  }

  return success();
}

namespace {
// Canonicalizes RealDynamicSlice ops that can be replaced instead with Slice
// ops. This canonicalization is applied the case when the `begin` input values
// are compile time constants and thus can be made into a tensor.
struct RealDynamicSliceIsStatic : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern<RealDynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RealDynamicSliceOp realDynamicSlice,
                                PatternRewriter& rewriter) const override {
    Location loc = realDynamicSlice.getLoc();
    Value input = realDynamicSlice.operand();
    Value output = realDynamicSlice.result();
    auto inputTy = input.getType().dyn_cast<RankedTensorType>();
    auto outputTy = output.getType().dyn_cast<RankedTensorType>();

    if (!inputTy || !outputTy || !inputTy.hasStaticShape() ||
        !outputTy.hasStaticShape()) {
      return failure();
    }

    int64_t inputRank = inputTy.getRank();

    auto startVal = realDynamicSlice.start_indices();
    auto limitVal = realDynamicSlice.limit_indices();
    auto strideVal = realDynamicSlice.strides();
    auto startOp = startVal.getDefiningOp<mlir::arith::ConstantOp>();
    auto limitOp = limitVal.getDefiningOp<mlir::arith::ConstantOp>();
    auto strideOp = strideVal.getDefiningOp<mlir::arith::ConstantOp>();
    if (!startOp || !limitOp || !strideOp) return failure();

    auto startAttr =
        startOp.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    auto limitAttr =
        limitOp.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    auto strideAttr =
        strideOp.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    if (!startAttr || !limitAttr || !strideAttr) return failure();

    SmallVector<int64_t, 4> tempStartIndices;
    SmallVector<int64_t, 4> tempLimitIndices;
    SmallVector<int64_t, 4> tempStride;
    for (int64_t dimIdx = 0; dimIdx < inputRank; dimIdx++) {
      int64_t start = startAttr.getValues<IntegerAttr>()[dimIdx].getInt();
      tempStartIndices.push_back(start);
      int64_t limit = limitAttr.getValues<IntegerAttr>()[dimIdx].getInt();
      tempLimitIndices.push_back(limit);
      int64_t end = strideAttr.getValues<IntegerAttr>()[dimIdx].getInt();
      tempStride.push_back(end);
    }

    DenseIntElementsAttr sliceStartIndices =
        rewriter.getI64TensorAttr(tempStartIndices);
    DenseIntElementsAttr sliceLimitIndices =
        rewriter.getI64TensorAttr(tempLimitIndices);
    DenseIntElementsAttr sliceStrides = rewriter.getI64TensorAttr(tempStride);
    auto result = rewriter.create<SliceOp>(loc, input, sliceStartIndices,
                                           sliceLimitIndices, sliceStrides);
    rewriter.replaceOp(realDynamicSlice, {result});
    return success();
  }
};
}  // namespace

void RealDynamicSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                     MLIRContext* context) {
  results.add<RealDynamicSliceIsStatic, RealDSliceToSlice>(context);
}

LogicalResult RealDynamicSliceOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  RealDynamicSliceOp::Adaptor adaptor(operands);
  Value operand = adaptor.operand();
  Value startIndices = adaptor.start_indices();
  Value limitIndices = adaptor.limit_indices();
  Value strides = adaptor.strides();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      startIndices.getType().cast<ShapedType>().getElementType();
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  one = maybeCastTo(builder, loc, one, shapeScalarType);
  for (const auto& element : llvm::enumerate(operandType.getShape())) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, element.index());
    Value valueStart =
        builder.create<tensor::ExtractOp>(loc, startIndices, offset);
    Value valueLimit =
        builder.create<tensor::ExtractOp>(loc, limitIndices, offset);
    Value valueStride = builder.create<tensor::ExtractOp>(loc, strides, offset);
    // size = (limit - start + stride - 1) / stride
    shapeValues.push_back(builder.create<arith::DivSIOp>(
        loc,
        builder.create<arith::SubIOp>(
            loc,
            builder.create<arith::AddIOp>(
                loc, valueStride,
                builder.create<arith::SubIOp>(loc, valueLimit, valueStart)),
            one),
        valueStride));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));
  return success();
}

//===----------------------------------------------------------------------===//
// InfeedOp
//===----------------------------------------------------------------------===//

// Checks that the result type is of the form `zero_or_more_type(s),
// mhlo::token`
LogicalResult InfeedOp::verify() {
  auto resultTypes = getResultTypes();
  if (resultTypes.empty())
    return emitOpError()
           << "result is expected to be at least of size 1, but got "
           << resultTypes.size();

  if (!resultTypes[resultTypes.size() - 1].isa<TokenType>())
    return emitOpError() << "last element of result types is expected to "
                            "be of token type, but got "
                         << resultTypes[resultTypes.size() - 1];

  // Verify layout attribute
  constexpr char kLayoutAttr[] = "layout";
  if (!getOperation()->hasAttr(kLayoutAttr)) return success();

  mlir::ArrayAttr layout =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(kLayoutAttr);
  if (!layout)
    return emitOpError() << "layout-attribute expected to be of array-type.";

  if (layout.size() != resultTypes.size() - 1) {
    return emitOpError() << "layout-attribute size must be "
                         << resultTypes.size() - 1
                         << " (which is the number of "
                            "op-results - 1 (for token result)), but got "
                         << layout.size();
  }

  for (auto childLayout : layout) {
    mlir::ArrayAttr childLayoutArr = childLayout.dyn_cast<mlir::ArrayAttr>();
    if (!childLayoutArr) {
      return emitOpError() << "layout-attribute expected to have "
                              "elements of type array, but got "
                           << childLayout;
    }

    for (auto i : childLayoutArr) {
      mlir::IntegerAttr attr = i.dyn_cast<mlir::IntegerAttr>();
      if (!attr) {
        return emitOpError() << "layout-attribute's leaf elements are "
                                "expected to be of type integer, but got "
                             << i;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Logical Ops
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) return lhs();

  auto rType = getType().cast<ShapedType>();
  auto lhsVal = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsVal = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnesValue()) {
      return rhs();
    }

    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return lhsVal;
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnesValue()) {
      return lhs();
    }

    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return rhsVal;
    }
  }

  if (!rhsVal || !lhsVal) return {};

  llvm::SmallVector<APInt, 4> values;
  values.reserve(rhsVal.getNumElements());
  for (auto it :
       llvm::zip(rhsVal.getValues<APInt>(), lhsVal.getValues<APInt>())) {
    values.push_back(std::get<0>(it) & std::get<1>(it));
  }

  return DenseIntElementsAttr::get(rType, values);
}

OpFoldResult OrOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) return lhs();

  auto rType = getType().cast<ShapedType>();
  auto lhsVal = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsVal = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnesValue()) {
      return lhsVal;
    }

    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return rhs();
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isAllOnesValue()) {
      return rhsVal;
    }

    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return lhs();
    }
  }

  if (!rhsVal || !lhsVal) return {};

  llvm::SmallVector<APInt, 4> values;
  values.reserve(rhsVal.getNumElements());
  for (auto it :
       llvm::zip(rhsVal.getValues<APInt>(), lhsVal.getValues<APInt>())) {
    values.push_back(std::get<0>(it) | std::get<1>(it));
  }

  return DenseIntElementsAttr::get(rType, values);
}

OpFoldResult XorOp::fold(ArrayRef<Attribute> operands) {
  // Fold x^x to 0. Attributes only support static shapes.
  auto rType = getType().cast<ShapedType>();
  if (lhs() == rhs() && rType.hasStaticShape()) {
    Builder builder(getContext());
    return builder.getZeroAttr(rType);
  }

  auto lhsVal = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsVal = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return rhs();
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue<IntegerAttr>().getValue().isNullValue()) {
      return lhs();
    }
  }

  if (!rhsVal || !lhsVal) return {};

  llvm::SmallVector<APInt, 4> values;
  values.reserve(rhsVal.getNumElements());
  for (auto it :
       llvm::zip(rhsVal.getValues<APInt>(), lhsVal.getValues<APInt>())) {
    values.push_back(std::get<0>(it) ^ std::get<1>(it));
  }

  return DenseIntElementsAttr::get(rType, values);
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

LogicalResult MapOp::verify() {
  // Checks if the number of `operands` match the arity of the map `computation`
  // region.
  auto& computationBlock = computation().front();
  auto computationArgs = computationBlock.getArguments();
  if (operands().size() != computationArgs.size())
    return emitOpError() << "expects number of operands to match the arity "
                            "of map computation, but got: "
                         << operands().size() << " and "
                         << computationArgs.size();

  // The parameters of computation should all be scalars and match the element
  // type of operands.
  for (const auto& indexedArg : llvm::enumerate(computationArgs)) {
    auto argType = indexedArg.value().getType().dyn_cast<TensorType>();
    if (!argType || argType.getRank() != 0)
      return emitOpError()
             << "computation arguments must be 0-rank tensor, but got: arg #"
             << indexedArg.index() << " of type "
             << indexedArg.value().getType();
    auto operandElemTy = operands()[indexedArg.index()]
                             .getType()
                             .cast<TensorType>()
                             .getElementType();
    if (argType.getElementType() != operandElemTy) {
      return emitOpError()
             << "element type of operands and computation arguments must "
                "match, but got: "
             << operandElemTy << " and " << argType.getElementType();
    }
  }

  // Mapped computation must return single output
  auto computationOutputs = computationBlock.getTerminator()->getOperands();
  if (computationOutputs.size() != 1)
    return emitOpError() << "computation must return single output, but got: "
                         << computationOutputs.size();

  // The output of computation must be scalar and have the same element type
  // as op result.
  auto computationOutputType =
      computationOutputs[0].getType().dyn_cast<TensorType>();
  if (!computationOutputType || computationOutputType.getRank() != 0)
    return emitOpError() << "computation must return 0-rank tensor, but got: "
                         << computationOutputs[0].getType();

  auto resultType = getType().cast<TensorType>();
  if (computationOutputType.getElementType() != resultType.getElementType())
    return emitOpError() << "element type of result and computation output "
                            "must match, but got: "
                         << resultType.getElementType() << " and "
                         << computationOutputType.getElementType();

  // Checks that the requested map dimension numbers are monotonically
  // increasing.
  DenseIntElementsAttr dimensions = this->dimensions();
  for (const auto& indexedValue :
       llvm::enumerate(dimensions.getValues<int64_t>())) {
    if (indexedValue.value() != indexedValue.index())
      return emitOpError() << "requires monotonically increasing dimension "
                              "numbers, but got: "
                           << dimensions;
  }

  // Checks that number of dimensions of operands matches the size of
  // `dimensions` since we currently only support mapping across all
  // dimensions: i.e., scalar map functions.
  auto operandType = operands()[0].getType().cast<TensorType>();
  if (operandType.hasRank()) {
    if (dimensions.size() != operandType.getShape().size())
      return emitOpError()
             << "applied to a subset of dimensions currently not supported: "
                "operand dimensions = "
             << operandType.getShape().size()
             << ", requested map dimensions size = " << dimensions.size();
  }

  return success();
}

OpFoldResult MapOp::fold(ArrayRef<Attribute> operands) {
  mlir::Block& bb = computation().front();
  mlir::Operation& frontOp = bb.front();

  auto retOp = mlir::dyn_cast<ReturnOp>(frontOp);
  if (!retOp) return nullptr;
  if (retOp.results().size() != 1) return nullptr;

  for (mlir::BlockArgument barg : bb.getArguments()) {
    if (barg == retOp.results()[0]) return getOperands()[barg.getArgNumber()];
  }
  return nullptr;
}

LogicalResult MapOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

// Checks that the result type is of the form `zero_or_more_type(s),
// mhlo::token`
LogicalResult RecvOp::verify() {
  auto resultTypes = getResultTypes();
  if (resultTypes.empty())
    return emitOpError()
           << "result is expected to be at least of size 1, but got "
           << resultTypes.size();
  if (!resultTypes[resultTypes.size() - 1].isa<TokenType>())
    return emitOpError() << "last element of result types is expected to "
                            "be of token type, but got "
                         << resultTypes[resultTypes.size() - 1];
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

OpFoldResult CopyOp::fold(ArrayRef<Attribute> operands) { return getOperand(); }

//===----------------------------------------------------------------------===//
// ReduceWindowOp
//===----------------------------------------------------------------------===//

namespace {
// Infer the return-type of ReduceWindowOp.
SmallVector<TensorType> inferReduceWindowOpReturnType(
    ArrayRef<TensorType> inputTypes, ArrayRef<TensorType> initTypes,
    const ArrayRef<WindowDimension> window) {
  SmallVector<TensorType> outputTypes;
  for (size_t i = 0; i < inputTypes.size(); ++i) {
    if (!inputTypes[i].hasRank()) {
      outputTypes.push_back(
          UnrankedTensorType::get(initTypes[i].getElementType()));
      continue;
    }

    outputTypes.push_back(RankedTensorType::get(
        inferWindowOutputShape(inputTypes[i].getShape(), window),
        initTypes[i].getElementType()));
  }

  return outputTypes;
}
}  // namespace

// We intend to verify the following properties
//  P1. The sizes of 'inputs' and 'init_values' must be at least 1.
//  P2. All `inputs` need to have compatible shapes.
//  P3. size-of(window_dimension) == rank-of(input),
//        where input is an element of 'inputs'.
//  P4. Verify and collect the window atributes.
//  P5. Verify the inner block defining the reducer function.
//  P6. Verify the return type.
LogicalResult ReduceWindowOp::verify() {
  // P1.
  // Note that the ODS ensures that there are even number of operands; Check if
  // that number is not zero.
  if (getOperands().empty())
    return emitOpError() << "expects the size of operands to be >= 2.";

  // Collect the input and init-value operands. Note that the operand-type is
  // enforced as "TensorType" by ODS.
  int64_t numInputs = getNumOperands() / 2;
  auto operandTensorTypes = llvm::to_vector<4>(llvm::map_range(
      getOperandTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); }));
  ArrayRef<TensorType> inputTypes(operandTensorTypes.begin(),
                                  operandTensorTypes.begin() + numInputs);
  ArrayRef<TensorType> initTypes(operandTensorTypes.begin() + numInputs,
                                 operandTensorTypes.end());

  // P2.
  if (failed(verifyCompatibleShapes(operands().getTypes())))
    return emitOpError() << "requires same shape for all inputs";

  // P3.
  SmallVector<int64_t> windowDims =
      convertDenseIntAttr(this->window_dimensions());
  for (const auto inputType : inputTypes) {
    if (!inputType.hasRank()) continue;
    if (inputType.getRank() != windowDims.size())
      return emitOpError()
             << "expects window-dimensions size == input rank, but got "
                "window-dimensions size: "
             << windowDims.size() << " and input: " << inputType
             << " with rank = " << inputType.getRank() << ".";
  }

  // P4.
  auto paddingOrErr = convertNx2Attribute(this->padding(), getLoc());
  if (failed(paddingOrErr)) return failure();
  SmallVector<std::pair<int64_t, int64_t>> padding = *paddingOrErr;

  auto windowOrErr = verifyWindowAttributesAndInferWindowDimensions(
      windowDims, convertDenseIntAttr(window_strides()), padding,
      /*lhs_dilation=*/convertDenseIntAttr(base_dilations()),
      /*rhs_dilation=*/convertDenseIntAttr(this->window_dilations()), getLoc());
  if (failed(windowOrErr)) return failure();

  // P5.
  bool allInputsUnranked =
      llvm::all_of(inputTypes, [](TensorType t) { return !t.hasRank(); });

  Block& block = body().front();
  SmallVector<TensorType> accumulatorSubshapes;
  if (failed(verifyReducerShape(this->getLoc(), block, inputTypes, initTypes,
                                numInputs, windowDims, allInputsUnranked,
                                accumulatorSubshapes)))
    return failure();

  // P6.
  if (numInputs != getNumResults())
    return emitOpError() << "expects " << numInputs
                         << " result values, but got " << getNumResults()
                         << ".";

  // The result-type is enforced as "TensorType" by ODS.
  auto resultTensorTypes = llvm::to_vector<4>(llvm::map_range(
      getResultTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); }));

  // Check if the element-type of results match with the ones derived from
  // the reducer-block. Already ensured that  |accumulator_subshapes| ==
  // num_inputs == num_of_results.
  for (int64_t shapeIdx = 0; shapeIdx < accumulatorSubshapes.size();
       shapeIdx++) {
    if (accumulatorSubshapes[shapeIdx].getElementType() !=
        resultTensorTypes[shapeIdx].getElementType()) {
      return emitError()
             << "expects the element-type of reduce-op's return-value at index "
             << shapeIdx
             << " to match the element-type of reducer-block's "
                "corresponding return-value, but got "
             << resultTensorTypes[shapeIdx].getElementType() << " and "
             << accumulatorSubshapes[shapeIdx].getElementType() << " resp.";
    }
  }

  // Check if the shape of results match with the ones derived from
  // the input-types and wndow-attributes.
  auto inferredReturnTypes = inferReduceWindowOpReturnType(
      inputTypes, accumulatorSubshapes, *windowOrErr);

  for (size_t i = 0; i < getNumResults(); i++) {
    if (failed(verifyCompatibleShape(resultTensorTypes[i],
                                     inferredReturnTypes[i]))) {
      return emitOpError()
             << "expects result at index " << i
             << " to have compatible shape with the corresponding "
                "inferred type, but got "
             << resultTensorTypes[i] << " and " << inferredReturnTypes[i]
             << " resp.";
    }
  }

  return success();
}

// Get the operation used for reduction applied to `result_index`th result. Its
// expected to be a binary operation that consumes `result_index`th and
// `result_index + operands().size`th arguments of the body.
Operation* ReduceWindowOp::getReductionOp(int resultIndex) {
  auto returnOp = cast<ReturnOp>(body().front().getTerminator());
  Operation* computeOp = returnOp.results()[resultIndex].getDefiningOp();
  if (computeOp->getNumOperands() != 2) return nullptr;
  auto arg0 = computeOp->getOperand(0).dyn_cast<BlockArgument>();
  auto arg1 = computeOp->getOperand(1).dyn_cast<BlockArgument>();
  if (!arg0 || !arg1) return nullptr;
  int arg0Num = arg0.getArgNumber();
  int arg1Num = arg1.getArgNumber();
  size_t otherArgIndex = resultIndex + operands().size();
  if (arg0Num == resultIndex && arg1Num == otherArgIndex) return computeOp;
  if (arg0Num == otherArgIndex && arg1Num == resultIndex &&
      computeOp->hasTrait<mlir::OpTrait::IsCommutative>())
    return computeOp;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ReducePrecisionOp
//===----------------------------------------------------------------------===//

// The following property is already enforced by the ODS:
//  P0. operand element type is float
//  P1. mantissa_bits >= 0
// We intend to verify the following properties
//  P2. exponent_bits >= 1
LogicalResult ReducePrecisionOp::verify() {
  if (exponent_bits() < 1) {
    return emitOpError() << "exponent_bits must be at least 1.";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

OpFoldResult ReverseOp::fold(ArrayRef<Attribute> operands) {
  auto input = operand();

  // No dimensions to reverse.
  if (dimensions().getNumElements() == 0) return input;

  llvm::SmallVector<APInt, 5> newDims;
  newDims.reserve(dimensions().getNumElements());

  auto shapedType = input.getType().cast<ShapedType>();
  for (auto dim : dimensions().getValues<APInt>()) {
    if (shapedType.getDimSize(dim.getLimitedValue()) != 1) {
      return nullptr;
    }
  }

  return input;
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

// Returns the result type after reducing operand of the given type across the
// specified dimensions.
static TensorType getReduceResultType(Type operandTy,
                                      DenseIntElementsAttr dimensions,
                                      Builder* builder) {
  Type elementTy = getElementTypeOrSelf(operandTy);

  auto rankedTy = operandTy.dyn_cast<RankedTensorType>();
  if (!rankedTy) return UnrankedTensorType::get(elementTy);

  int64_t rank = rankedTy.getRank();
  llvm::SmallVector<bool, 4> dimsMask(rank, false);
  for (int64_t dim : dimensions.getValues<int64_t>()) dimsMask[dim] = true;

  SmallVector<int64_t, 4> shape;
  for (int64_t i = 0; i < rank; ++i) {
    if (!dimsMask[i]) shape.push_back(rankedTy.getDimSize(i));
  }

  return RankedTensorType::get(shape, elementTy);
}

void ReduceOp::build(OpBuilder& builder, OperationState& state,
                     ValueRange operands, ValueRange initValues,
                     DenseIntElementsAttr dimensions) {
  SmallVector<Type, 1> resultTy;
  resultTy.reserve(operands.size());

  for (Value operand : operands) {
    resultTy.push_back(
        getReduceResultType(operand.getType(), dimensions, &builder));
  }
  build(builder, state, resultTy, operands, initValues, dimensions);
}

LogicalResult ReduceOp::fold(ArrayRef<Attribute> operands,
                             SmallVectorImpl<OpFoldResult>& results) {
  // No dimensions to reduce.
  if (dimensions().getNumElements() == 0) {
    for (Value operand : this->operands()) {
      results.push_back(operand);
    }
    return success();
  }

  // If all returned values in the ReduceOp region exists outside
  // the region replace the ReduceOp with those values.
  mlir::Block& bb = this->body().front();
  SmallVector<Value> replacedResults;
  if (auto retOp = mlir::dyn_cast<ReturnOp>(bb.back())) {
    for (Value result : retOp.results()) {
      if (result.getParentRegion() == retOp->getParentRegion())
        return failure();
      replacedResults.push_back(result);
    }

    results.insert(results.end(), replacedResults.begin(),
                   replacedResults.end());
    return success();
  }

  return failure();
}

bool hasSameOperandAndResultTypes(Operation& op) {
  Type expected;
  if (op.getNumResults() != 0) expected = op.getResult(0).getType();
  if (op.getNumOperands() != 0) expected = op.getOperand(0).getType();
  if (!expected) return false;

  auto typeMatch = [&](Type actual) { return actual == expected; };
  return llvm::all_of(op.getOperandTypes(), typeMatch) &&
         llvm::all_of(op.getResultTypes(), typeMatch);
}

// Checks the following eligibility criteria for compact printing of
// mhlo.reduce:
// E1. The reduce-op wraps a single inner-op in the associated region.
// E2. The single operation is a commutative binary-op from mhlo dialect, zero
//     region, producing single result such that the operands and result all
//     have the same type.
// E3. The reduce-op consist of at least one input-operand; The operand-types of
//     inner-op should be derived trivially from the element-type of reduce-op's
//     first input-operand.
// E4. The  arguments of the region's only basic block are forwarded perfectly
//     to inner-op's operands.
// E5. The reduce-op, inner-op, blocks arguments, and the return-op all have the
//     same location.
// E6. The single operation result is perfectly forwarded to the reduce op
//     return.
static bool isEligibleForCompactPrint(ReduceOp op) {
  // Check E1.
  auto& block = op.body().front();
  if (!hasSingleElement(block.without_terminator())) return false;

  Operation& innerOp = *block.begin();

  // Check E2.
  if (innerOp.getDialect() != op->getDialect()) return false;

  if (innerOp.getNumOperands() != 2 ||
      !innerOp.hasTrait<mlir::OpTrait::OneResult>() ||
      !hasSameOperandAndResultTypes(innerOp) ||
      !innerOp.hasTrait<mlir::OpTrait::IsCommutative>() ||
      !innerOp.hasTrait<mlir::OpTrait::ZeroRegions>())
    return false;

  // Check E3.
  if (op.operands().empty()) return false;

  auto elemType =
      op.operands()[0].getType().cast<TensorType>().getElementType();
  auto expectedInnerOpType = RankedTensorType::get(/*shape=*/{}, elemType);
  if (innerOp.getOperands()[0].getType() != expectedInnerOpType) return false;

  // Check E4.
  if (!llvm::equal(block.getArguments(), innerOp.getOperands())) return false;

  // Check E5.
  auto retOp = dyn_cast<ReturnOp>(block.getTerminator());
  if (!retOp) return false;

  auto blockArgLoc = block.getArgument(0).getLoc();
  if (blockArgLoc != block.getArgument(1).getLoc()) return false;

  if (innerOp.getLoc() != op.getLoc() || retOp.getLoc() != op.getLoc() ||
      blockArgLoc != op.getLoc())
    return false;

  // Check E6.
  return llvm::equal(innerOp.getResults(), retOp.getOperands());
}

void ReduceOp::print(OpAsmPrinter& p) {
  {
    // Print the pairs of operands under the form:
    //   (%arg0 init: %arg3), (%arg1 init: %arg4), (%arg2 init: %arg5)
    StringRef comma = "";
    int numOperandPairs = getNumOperands() / 2;
    for (int opId : llvm::seq<int>(0, numOperandPairs)) {
      p << comma << "(" << getOperand(opId)
        << " init: " << getOperand(opId + numOperandPairs) << ")";
      comma = ", ";
    }
  }

  // If the reduce-op is eligible for compact printing, we emit the one-liner:
  //  mhlo.reduce applies <inner-op> across dimensions = [...] : <func-type>
  // Note: We are not printing the function type of reduction operation. We
  // have some simplifying assumptions (refer to IsEligibleForCompactPrint::E3)
  // to derive the type from that of reduce-op.
  if (isEligibleForCompactPrint(*this)) {
    Operation& innerOp = body().front().front();
    p << " applies ";
    printEscapedString(innerOp.getName().getStringRef(), p.getStream());

    p << " across dimensions = [";
    llvm::interleaveComma(dimensions().getValues<int64_t>(), p);
    p << "]";
    p << " : ";
    p.printFunctionalType(*this);
  } else {
    p << " across dimensions = [";
    llvm::interleaveComma(dimensions().getValues<int64_t>(), p);
    p << "]";
    p.printOptionalAttrDict(getOperation()->getAttrs(), {"dimensions"});
    p << " : ";
    p.printFunctionalType(*this);
    p.printNewline();
    p << " reducer";
    {
      // Print the pairs of block operands under the form:
      //   (%arg0_elt, %arg0_acc) (%arg1_elt, %arg1_acc):
      Block& reducer = body().front();
      int numOperandPairs = getNumOperands() / 2;
      for (int opId : llvm::seq<int>(0, numOperandPairs)) {
        p << "(";
        p.printRegionArgument(reducer.getArgument(opId));
        p << ", ";
        p.printRegionArgument(reducer.getArgument(opId + numOperandPairs));
        p << ") ";
      }
    }
    p << ' ';
    p.printRegion(body(), /*printEntryBlockArgs=*/false);
  }
}

ParseResult ReduceOp::parse(OpAsmParser& parser, OperationState& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  Location currLocation = parser.getEncodedSourceLoc(loc);

  // Parse the operands of reduce-op, this is a list of pair under the form:
  //   (%arg0 init: %arg3), (%arg1 init: %arg4), (%arg2 init: %arg5)
  // Each input to reduce is paired with its init value, even though in memory
  // they are stored with the input first and the init values after.
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> initOperands;
  do {
    (void)parser.parseOptionalComma();
    if (parser.parseOptionalLParen()) break;
    OpAsmParser::UnresolvedOperand operand, initOperand;
    if (parser.parseOperand(operand) || parser.parseKeyword("init") ||
        parser.parseColon() || parser.parseOperand(initOperand) ||
        parser.parseRParen())
      return failure();
    operands.push_back(operand);
    initOperands.push_back(initOperand);
  } while (true);
  operands.append(initOperands);

  // Check if we are parsing the compact version of reduce-op:
  //  mhlo.reduce applies <inner-op> across dimensions = [...] : <func-type>
  // else parse the "region-based" variant.
  if (failed(parser.parseOptionalKeyword("applies"))) {
    // Parse the inner-op dimensions, reduce-op's function-type and
    // optional location.
    SmallVector<int64_t> dimensions;
    auto parseDim = [&]() -> ParseResult {
      if (parser.parseInteger(dimensions.emplace_back())) return failure();
      return success();
    };

    FunctionType reduceOpFntype;
    if (parser.parseKeyword("across") || parser.parseKeyword("dimensions") ||
        parser.parseEqual() ||
        parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                       parseDim) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseType(reduceOpFntype) ||
        parser.parseKeyword("reducer"))
      return failure();
    OpBuilder builder(parser.getBuilder().getContext());
    result.addAttribute("dimensions", builder.getI64TensorAttr(dimensions));

    // Parse the "reducer" region now.
    SmallVector<OpAsmParser::UnresolvedOperand, 2> reducerOperands;
    SmallVector<OpAsmParser::UnresolvedOperand, 2> reducerInitOperands;
    SmallVector<Type, 2> reducerTypes;
    SmallVector<Type, 2> reducerInitTypes;
    SmallVector<Optional<Location>, 2> reducerLocs;
    SmallVector<Optional<Location>, 2> reducerInitLocs;
    auto parseBlockOperand =
        [&](SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
            SmallVectorImpl<Type>& types,
            SmallVectorImpl<Optional<Location>>& locs) -> ParseResult {
      OpAsmParser::UnresolvedOperand operand;
      Type type;
      Optional<Location> loc;
      if (parser.parseOperand(operand, /*allowResultNumber=*/false) ||
          parser.parseColon() || parser.parseType(type) ||
          parser.parseOptionalLocationSpecifier(loc))
        return failure();
      operands.push_back(operand);
      types.push_back(type);
      locs.push_back(loc);
      return success();
    };
    do {
      if (failed(parser.parseOptionalLParen())) break;
      if (parseBlockOperand(reducerOperands, reducerTypes, reducerLocs) ||
          parser.parseComma() ||
          parseBlockOperand(reducerInitOperands, reducerInitTypes,
                            reducerInitLocs) ||
          parser.parseRParen())
        return failure();
    } while (true);
    reducerOperands.append(reducerInitOperands);
    reducerTypes.append(reducerInitTypes);
    reducerLocs.append(reducerInitLocs);
    result.addTypes(reduceOpFntype.getResults());
    SmallVector<OpAsmParser::Argument> reducerArgs;
    createArgs(reducerOperands, reducerTypes, reducerArgs);

    // Derive the SSA-values for reduce-op's operands and parse the region, and
    // the optional trailing location.
    Optional<Location> trailingLoc;
    if (parser.resolveOperands(operands, reduceOpFntype.getInputs(), loc,
                               result.operands) ||
        parser.parseRegion(*result.addRegion(), reducerArgs))
      return failure();
    // Set the individual block arguments.
    for (auto argAndLoc :
         llvm::zip(result.regions.front()->front().getArguments(), reducerLocs))
      if (std::get<1>(argAndLoc))
        std::get<0>(argAndLoc).setLoc(std::get<1>(argAndLoc).getValue());
    result.location = trailingLoc.getValueOr(currLocation);
    return success();
  }

  // Parse the inner-op name and check if the contract on inner-op
  // mentioned in "isEligibleForCompactPrint::E2" for pretty-priting is met.
  FailureOr<OperationName> innerOpNameInfo = parser.parseCustomOperationName();
  if (failed(innerOpNameInfo)) return failure();

  StringRef innerOpName = innerOpNameInfo->getStringRef();
  Dialect* innerOpDialect = innerOpNameInfo->getDialect();
  if (!innerOpDialect || !innerOpDialect->getNamespace().equals("mhlo") ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::NOperands<2>::Impl>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::OneResult>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::IsCommutative>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::ZeroRegions>()) {
    parser.emitError(loc,
                     "expected the inner-op to be a commutative binary-op from "
                     "mhlo dialect, zero region, producing single result");
    return failure();
  }

  // Parse the inner-op dimensions, reduce-op's function-type and
  // optional location.
  SmallVector<int64_t> dimensions;
  auto parseDim = [&]() -> ParseResult {
    if (parser.parseInteger(dimensions.emplace_back())) return failure();
    return success();
  };

  Optional<Location> explicitLoc;
  FunctionType reduceOpFntype;
  if (parser.parseKeyword("across") || parser.parseKeyword("dimensions") ||
      parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseDim) ||
      parser.parseColon() || parser.parseType(reduceOpFntype) ||
      parser.parseOptionalLocationSpecifier(explicitLoc))
    return failure();

  if (!reduceOpFntype || reduceOpFntype.getInputs().empty()) {
    if (!reduceOpFntype) return parser.emitError(loc, "expected function type");
    return parser.emitError(loc,
                            "input types missing in reduce-op function type");
  }

  // If location of reduce-op is explicitly provided, then use it; Else use
  // the parser's current location.
  Location reduceOpLoc = explicitLoc.getValueOr(currLocation);

  // Derive the SSA-values for reduce-op's operands.
  if (parser.resolveOperands(operands, reduceOpFntype.getInputs(), loc,
                             result.operands))
    return failure();

  // Derive the type of inner-op from that of reduce-op's input operand.
  auto innerOpType = RankedTensorType::get(
      /*shape=*/{}, getElementTypeOrSelf(reduceOpFntype.getInput(0)));

  // Add a region for reduce-op.
  Region& region = *result.addRegion();

  // Create a basic-block inside reduce-op's region.
  Block& block = region.emplaceBlock();
  auto lhs = block.addArgument(innerOpType, reduceOpLoc);
  auto rhs = block.addArgument(innerOpType, reduceOpLoc);

  // Create and insert an "inner-op" operation in the block.
  OpBuilder builder(parser.getBuilder().getContext());
  builder.setInsertionPointToStart(&block);

  OperationState innerOpState(reduceOpLoc, innerOpName);
  innerOpState.operands.push_back(lhs);
  innerOpState.operands.push_back(rhs);
  innerOpState.addTypes(innerOpType);

  Operation* innerOp = builder.create(innerOpState);

  // Insert a return statement in the block returning the inner-op's result.
  builder.create<ReturnOp>(innerOp->getLoc(), innerOp->getResults());

  // Populate the reduce-op operation-state with result-type, location, and
  // dimension attribute.
  result.addTypes(reduceOpFntype.getResults());
  result.location = innerOp->getLoc();
  result.addAttribute("dimensions", builder.getI64TensorAttr(dimensions));

  return success();
}

LogicalResult ReduceOp::verify() {
  // Check that there are even number of operands and >= 2.
  if (getNumOperands() % 2 != 0 || getOperands().empty())
    return emitOpError() << "expects the size of operands to be even and >= 2";

  // Collect the input and init-value operands. Note that the operand-type is
  // enforced as "TensorType" by ODS.
  int64_t numInputs = getNumOperands() / 2;
  auto operandTensorTypes = llvm::to_vector<4>(llvm::map_range(
      getOperandTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); }));
  ArrayRef<TensorType> inputArgTypes(operandTensorTypes.begin(),
                                     operandTensorTypes.begin() + numInputs);
  ArrayRef<TensorType> initValueTypes(operandTensorTypes.begin() + numInputs,
                                      operandTensorTypes.end());

  // Check for unranked tensors in input operands.
  int64_t rankedInputIdx = -1;
  for (int64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    if (inputArgTypes[inputIdx].hasRank()) {
      rankedInputIdx = inputIdx;
      break;
    }
  }

  bool allInputsUnranked = (rankedInputIdx == -1);

  // Check that all input operands have compatible shapes. The element types may
  // be different.
  if (!allInputsUnranked) {
    for (int64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
      if (failed(mlir::verifyCompatibleShape(inputArgTypes[rankedInputIdx],
                                             inputArgTypes[inputIdx]))) {
        return emitOpError()
               << "expects all inputs to have compatible shapes. Shape at"
               << " input-index " << inputIdx
               << " is not compatible with shape at input-index "
               << rankedInputIdx;
      }
    }
  }

  // Check that
  //   1. the dimensions of reduce-op are in-bounds for the given shape.
  //   2. the dimension-attribute have no duplicate entries.
  DenseSet<int64_t> dimensionsToReduceSet;
  for (int64_t dimension : dimensions().getValues<int64_t>()) {
    if ((!allInputsUnranked &&
         dimension >= inputArgTypes[rankedInputIdx].getRank()) ||
        dimension < 0) {
      return emitError() << "Out-of-bounds dimension " << dimension
                         << " for input-tensor rank: "
                         << inputArgTypes[rankedInputIdx].getRank();
    }

    if (!dimensionsToReduceSet.insert(dimension).second) {
      return emitError() << "Duplicate reduction dimension: " << dimension;
    }
  }

  // Verify the inner block defining the reducer function.
  SmallVector<int64_t> newDimensions;
  if (!allInputsUnranked) {
    for (int inputIdx = 0; inputIdx < inputArgTypes[rankedInputIdx].getRank();
         ++inputIdx) {
      if (!dimensionsToReduceSet.count(inputIdx)) {
        newDimensions.push_back(
            inputArgTypes[rankedInputIdx].getDimSize(inputIdx));
      }
    }
  }

  Block& block = body().front();
  SmallVector<TensorType> accumulatorSubShapes;
  if (failed(verifyReducerShape(this->getLoc(), block, inputArgTypes,
                                initValueTypes, numInputs, newDimensions,
                                allInputsUnranked, accumulatorSubShapes)))
    return failure();

  // Check if the reduce-op's result-type matches with the one derived from
  // the reducer-block and dimensions attribute.
  if (getResults().size() != accumulatorSubShapes.size())
    return emitError() << "Unexpected number of reduce-op's returned values: "
                       << getResults().size() << " vs "
                       << accumulatorSubShapes.size() << " (expected)";

  for (int64_t shapeIdx = 0; shapeIdx < accumulatorSubShapes.size();
       shapeIdx++) {
    // The result-type is enforced as "TensorType" by ODS.
    auto opResultType = getResult(shapeIdx).getType().cast<TensorType>();

    // Check element-type.
    if (accumulatorSubShapes[shapeIdx].getElementType() !=
        opResultType.getElementType()) {
      return emitError()
             << "Unexpected element-type for reduce-op's return value at index "
             << shapeIdx << ": " << opResultType.getElementType() << " vs "
             << accumulatorSubShapes[shapeIdx].getElementType()
             << " (expected)";
    }

    // Check shape.
    if (!allInputsUnranked && opResultType.hasRank() &&
        (newDimensions != opResultType.getShape())) {
      Type expectedResultType = RankedTensorType::get(
          newDimensions, accumulatorSubShapes[shapeIdx].getElementType());
      return emitError()
             << "Unexpected type for reduce-op's return value at index "
             << shapeIdx << ": " << opResultType << " vs " << expectedResultType
             << " (expected)";
    }
  }

  return success();
}

// Enable constant folding to occur within the region of the ReduceOp
// by replacing block argument uses with constants if:
//  1. All the ReduceOp operands are splat constants.
//  2. The ReduceOp region consists of a single logical AND or logical OR.
// The pattern leverages the idempotent property of the AND and OR operators
// to determine the value of a reduction on splat constants. Other boolean
// operators do not have this property, and need separate patterns to resolve
// reductions of their splat constants.
struct LowerBoolSplatConstantsIntoRegion : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    mlir::Block& bb = op.body().front();

    // Ensure only a compute op and return op exist and the
    // compute op is an AND or OR op.
    if (bb.getOperations().size() != 2) return failure();
    if (!mlir::isa<mhlo::AndOp, mhlo::OrOp>(bb.front())) return failure();

    // Ensure all operands are splat constants.
    SmallVector<DenseElementsAttr, 4> bargCstAttrs;
    for (auto inpAndBarg : llvm::zip(op.getOperands(), bb.getArguments())) {
      Value inp = std::get<0>(inpAndBarg);
      BlockArgument barg = std::get<1>(inpAndBarg);
      ConstOp cst = inp.getDefiningOp<ConstOp>();
      if (!cst) return failure();

      auto cstAttr = cst.value().dyn_cast_or_null<DenseElementsAttr>();
      if (!cstAttr.isSplat()) {
        return rewriter.notifyMatchFailure(op, "Must be splat constant.");
      }

      auto bargShapedType = barg.getType().dyn_cast<ShapedType>();
      if (!bargShapedType) return failure();

      auto bargCstAttr = DenseElementsAttr::get(
          bargShapedType, cstAttr.getSplatValue<mlir::Attribute>());
      bargCstAttrs.push_back(bargCstAttr);
    }

    // Create new splat constants to replace block arguments.
    for (BlockArgument barg : bb.getArguments()) {
      int argIdx = barg.getArgNumber();
      mhlo::ConstOp newCst = rewriter.create<mhlo::ConstOp>(
          bb.front().getLoc(), barg.getType(), bargCstAttrs[argIdx]);
      barg.replaceAllUsesWith(newCst);
    }
    return success();
  }
};

void ReduceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<LowerBoolSplatConstantsIntoRegion>(context);
}

LogicalResult ReduceOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  ReduceOp::Adaptor adaptor(operands);
  auto inputs = adaptor.operands();

  auto operandType = inputs[0].getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  SmallVector<int64_t, 4> dimensions(this->dimensions().getValues<int64_t>());
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto& element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto* it = std::find(dimensions.begin(), dimensions.end(), idx);
    if (it != dimensions.end()) {
      continue;
    }
    Value valueDim = toShapeScalarType(
        builder.create<tensor::DimOp>(loc, inputs[0], element.index()));
    shapeValues.push_back(valueDim);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  for (size_t i = 0; i < inputs.size(); ++i) {
    reifiedReturnShapes.push_back(outputShape);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RngBitGeneratorOp
//===----------------------------------------------------------------------===//

// Verify that input state has the same shape as output shape
LogicalResult RngBitGeneratorOp::verify() {
  auto initialShape = initial_state().getType().dyn_cast<RankedTensorType>();
  auto outputShape = output_state().getType().dyn_cast<RankedTensorType>();
  if (initialShape.getShape() != outputShape.getShape())
    return emitOpError()
           << "output state shape must match initial state shape. Got: "
           << initialShape << " and " << outputShape;
  return success();
}

//===----------------------------------------------------------------------===//
// RngNormalOp
//===----------------------------------------------------------------------===//

LogicalResult RngNormalOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  return rngInferReturnTypeComponents(context, location, operands, attributes,
                                      regions, inferredReturnShapes);
}

LogicalResult RngNormalOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  RngNormalOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.shape()));
  return success();
}

//===----------------------------------------------------------------------===//
// RngUniformOp
//===----------------------------------------------------------------------===//

LogicalResult RngUniformOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  return rngInferReturnTypeComponents(context, location, operands, attributes,
                                      regions, inferredReturnShapes);
}

LogicalResult RngUniformOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  RngUniformOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.shape()));
  return success();
}

//===----------------------------------------------------------------------===//
// XlaRngGetAndUpdateStateOp
//===----------------------------------------------------------------------===//

LogicalResult XlaRngGetAndUpdateStateOp::verify() {
  auto resultTy = getType().cast<RankedTensorType>();
  if (!resultTy) return emitOpError() << "Output is not ranked.";
  if (!resultTy.hasStaticShape())
    return emitOpError() << "Output is not statically shaped.";
  auto rank = resultTy.getRank();
  if (rank != 1)
    return emitOpError() << "Output is of rank " << rank << " instead of 1";
  auto extent = resultTy.getDimSize(0);
  if (extent != 2)
    return emitOpError() << "Output size is " << extent << " instead of 2";

  return success();
}

LogicalResult XlaRngGetAndUpdateStateOp::inferReturnTypes(
    MLIRContext* ctx, Optional<Location>, ValueRange, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(mlir::RankedTensorType::get(
      {2}, mlir::IntegerType::get(ctx, 64, IntegerType::Unsigned)));
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  // Either, all operands could be the same shape ...
  if (succeeded(verifyCompatibleShapes(getOperandTypes()))) return success();

  // ... or the predicate could be a scalar and the remaining two operands could
  // be of the same shape.
  auto predTy = pred().getType().dyn_cast<RankedTensorType>();
  bool predMayBeScalar = !predTy || predTy.getRank() == 0;
  if (!predMayBeScalar || failed(verifyCompatibleShapes(
                              {on_true().getType(), on_false().getType()}))) {
    return emitOpError()
           << "requires the same type for all operands and results";
  }
  return success();
}

OpFoldResult SelectOp::fold(ArrayRef<Attribute> operands) {
  if (on_true() == on_false()) {
    return on_true();
  }

  auto predicate = operands[0].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!predicate) {
    return {};
  }

  auto predicateTy = predicate.getType().cast<ShapedType>();
  if (!predicateTy.getElementType().isInteger(1)) {
    return {};
  }

  if (predicate.isSplat()) {
    return predicate.getSplatValue<APInt>().getBoolValue() ? on_true()
                                                           : on_false();
  }

  return {};
}

// simplify select(not(%pred), true_value, false_value) => select(%pred,
// false_value, true_value)
static LogicalResult selectCanonicalization(SelectOp selectOp,
                                            PatternRewriter& rewriter) {
  auto notOp = selectOp.pred().getDefiningOp<NotOp>();
  if (!notOp) {
    return failure();
  }
  std::array<Value, 3> newOperands = {notOp.operand(), selectOp.on_false(),
                                      selectOp.on_true()};
  rewriter.updateRootInPlace(
      selectOp, [&]() { selectOp.getOperation()->setOperands(newOperands); });
  return success();
}

void SelectOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* /*context*/) {
  results.add(&selectCanonicalization);
}

// Makes it such that a SelectOp that is a non-root operation in a DRR infers
// the return type based on operand type.
LogicalResult SelectOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SelectOp::Adaptor op(operands, attributes);
  auto trueType = op.on_true().getType().cast<TensorType>();
  auto falseType = op.on_true().getType().cast<TensorType>();

  // Check for type compatibility in the select op. This requires that the two
  // non-predicate operands:
  //   (a) have the same element type
  //   (b) have compatible shapes (i.e. the same shape and/or at least one
  //       dynamic shape)
  if (trueType.getElementType() != falseType.getElementType() ||
      failed(mlir::verifyCompatibleShape(trueType, falseType))) {
    return emitOptionalError(location, "incompatible operand types: ", trueType,
                             " and ", falseType);
  }

  // The output shape should be the most general of the operand shapes at each
  // dimension.
  ShapedTypeComponents& outputType = inferredReturnShapes.emplace_back();
  if (trueType == falseType || !trueType.hasRank()) {
    outputType = ShapedTypeComponents(trueType.cast<ShapedType>());
  } else if (!falseType.hasRank()) {
    outputType = ShapedTypeComponents(falseType.cast<ShapedType>());
  } else {
    assert(trueType.getRank() == falseType.getRank());
    llvm::SmallVector<int64_t, 4> dims;
    dims.reserve(trueType.getRank());
    for (auto dim : llvm::zip(trueType.getShape(), falseType.getShape())) {
      dims.push_back(std::get<0>(dim) == std::get<1>(dim)
                         ? std::get<0>(dim)
                         : ShapedType::kDynamicSize);
    }
    outputType = ShapedTypeComponents(dims, trueType.getElementType());
  }
  return success();
}

LogicalResult SelectOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  // For `hlo.select`, the first operand may be a scalar.
  return deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// SetDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult SetDimensionSizeOp::verify() {
  if (auto size = this->size().getType().dyn_cast<RankedTensorType>()) {
    if (size.getRank() != 0)
      return emitOpError() << "size operand should be of rank-0";
  }

  return verifyDimAttr(*this);
}

OpFoldResult SetDimensionSizeOp::fold(ArrayRef<Attribute> operands) {
  DenseElementsAttr input = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  if (input) return input;

  DenseElementsAttr size = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  if (!size || !size.isSplat()) return {};

  auto ty = getType().dyn_cast<RankedTensorType>();
  if (!ty) return {};

  int64_t dimSize = ty.getDimSize(dimension());
  if (dimSize == size.getSplatValue<IntegerAttr>().getInt()) return operand();
  return {};
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult PadOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  PadOp::Adaptor adaptor(operands, attributes, regions);
  auto inputType = adaptor.operand().getType().cast<RankedTensorType>();
  auto padType = adaptor.padding_value().getType().cast<RankedTensorType>();

  if (padType.getRank() != 0) {
    return emitOptionalError(
        location, llvm::formatv("padding value type should be a rank-0 "
                                "tensor, is rank {0}",
                                padType.getRank()));
  }

  const auto& paddingLow = adaptor.edge_padding_low();
  if (paddingLow.getType().getNumElements() != inputType.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "edge_padding_low length ({0}) must match operand rank ({1})",
            paddingLow.getType().getNumElements(), inputType.getRank()));
  }

  const auto& paddingHigh = adaptor.edge_padding_high();
  if (paddingHigh.getType().getNumElements() != inputType.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "edge_padding_high length ({0}) must match operand rank ({1})",
            paddingHigh.getType().getNumElements(), inputType.getRank()));
  }

  const auto& paddingInterior = adaptor.interior_padding();
  if (paddingInterior.getType().getNumElements() != inputType.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "interior_padding length ({0}) must match operand rank ({1})",
            paddingInterior.getType().getNumElements(), inputType.getRank()));
  }

  auto inputShape = inputType.getShape();
  SmallVector<int64_t> resultShape;
  for (int i = 0, e = inputShape.size(); i < e; i++) {
    if (isDynamicDimSize(inputShape[i])) {
      resultShape.push_back(ShapedType::kDynamicSize);
      continue;
    }

    int64_t paddingLowVal = paddingLow.getValues<APInt>()[i].getSExtValue();
    int64_t paddingHighVal = paddingHigh.getValues<APInt>()[i].getSExtValue();
    int64_t paddingInteriorVal =
        paddingInterior.getValues<APInt>()[i].getSExtValue();
    if (paddingInteriorVal < 0) {
      return emitOptionalError(
          location, llvm::formatv("Interior padding cannot be negative: {0}",
                                  paddingInteriorVal));
    }
    int64_t expectedOutput =
        inputShape[i] + paddingLowVal + paddingHighVal +
        std::max<int64_t>(inputShape[i] - 1, 0LL) * paddingInteriorVal;
    if (expectedOutput < 0) {
      return emitOptionalError(
          location,
          llvm::formatv("Padding result in negative size for dimension {0}",
                        i));
    }
    resultShape.push_back(expectedOutput);
  }
  inferredReturnShapes.emplace_back(resultShape, inputType.getElementType());

  return success();
}

template <typename T>
OpFoldResult padOpFoldHelper(DenseElementsAttr input, DenseElementsAttr padding,
                             RankedTensorType returnType,
                             DenseIntElementsAttr edgePaddingLow,
                             DenseIntElementsAttr edge_padding_high,
                             DenseIntElementsAttr interiorPadding) {
  // Fill the full result tensor with the padding value.
  llvm::SmallVector<T, 4> result(returnType.getNumElements(),
                                 padding.getValues<T>()[0]);

  auto nextIndex = [](llvm::SmallVector<uint64_t, 8>& index,
                      llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (index[i] < shape[i]) return;
      index[i] = 0;
    }
  };

  // Iterate over all elements of the input tensor and copy it to the correct
  // location in the output tensor.
  llvm::SmallVector<uint64_t, 8> index(input.getType().getRank(), 0);
  uint64_t numElements = input.getNumElements();
  for (uint64_t operandIdx = 0; operandIdx < numElements; operandIdx++) {
    uint64_t resultIdx = 0;
    uint64_t idxMultiplyer = 1;
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      resultIdx += (edgePaddingLow.getValues<int64_t>()[i] +
                    index[i] * (interiorPadding.getValues<int64_t>()[i] + 1)) *
                   idxMultiplyer;
      idxMultiplyer *= returnType.getDimSize(i);
    }
    result[resultIdx] = input.getValues<T>()[index];
    nextIndex(index, input.getType().getShape());
  }
  return DenseElementsAttr::get(returnType, result);
}

OpFoldResult PadOp::fold(ArrayRef<Attribute> operands) {
  // If all padding is zero then it is an identity pad.
  auto isZero = [](const APInt& i) { return i == 0; };
  if (llvm::all_of(edge_padding_low().getValues<APInt>(), isZero) &&
      llvm::all_of(edge_padding_high().getValues<APInt>(), isZero) &&
      llvm::all_of(interior_padding().getValues<APInt>(), isZero))
    return operand();

  // If any padding is negative then it isn't supported by the folder (yet).
  auto isNegative = [](const APInt& i) { return i.slt(0); };
  if (llvm::any_of(edge_padding_low().getValues<APInt>(), isNegative) ||
      llvm::any_of(edge_padding_high().getValues<APInt>(), isNegative) ||
      llvm::any_of(interior_padding().getValues<APInt>(), isNegative))
    return {};

  DenseElementsAttr input = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  DenseElementsAttr padding = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  RankedTensorType returnType = getType().dyn_cast_or_null<RankedTensorType>();
  if (!input || !input.getType().hasRank() || !padding || !returnType ||
      !returnType.hasStaticShape())
    return {};

  if (returnType.getElementType().isa<IntegerType>())
    return padOpFoldHelper<APInt>(input, padding, returnType,
                                  edge_padding_low(), edge_padding_high(),
                                  interior_padding());
  if (returnType.getElementType().isa<FloatType>())
    return padOpFoldHelper<APFloat>(input, padding, returnType,
                                    edge_padding_low(), edge_padding_high(),
                                    interior_padding());
  if (ComplexType complex =
          returnType.getElementType().dyn_cast_or_null<ComplexType>()) {
    // TODO(atondwal): Allow int types in HLO_complex
    if (complex.getElementType().isa<FloatType>())
      return padOpFoldHelper<std::complex<APFloat>>(
          input, padding, returnType, edge_padding_low(), edge_padding_high(),
          interior_padding());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// DynamicPadOp
//===----------------------------------------------------------------------===//

void DynamicPadOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                               MLIRContext* context) {
  results.add<DPadToPad>(context);
}

LogicalResult DynamicPadOp::verify() {
  auto inputType = operand().getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!inputType) return success();
  int inputRank = inputType.getRank();

  auto padType = padding_value().getType().cast<RankedTensorType>();
  if (padType.getRank() != 0) {
    return emitOpError() << "padding value type should be a rank-0";
  }

  auto paddingLowType = edge_padding_low().getType().cast<RankedTensorType>();
  if (paddingLowType.getNumElements() != inputRank) {
    return emitOpError() << "edge_padding_low length("
                         << paddingLowType.getNumElements()
                         << ") must match operand rank(" << inputRank << ").";
  }

  auto paddingHighType = edge_padding_high().getType().cast<RankedTensorType>();
  if (paddingHighType.getNumElements() != inputRank) {
    return emitOpError() << "edge_padding_high length("
                         << paddingHighType.getNumElements()
                         << ") must match operand rank(" << inputRank << ").";
  }

  auto interiorPaddingType =
      interior_padding().getType().cast<RankedTensorType>();
  if (interiorPaddingType.getNumElements() != inputRank) {
    return emitOpError() << "edge_padding_interior length("
                         << interiorPaddingType.getNumElements()
                         << ") must match operand rank(" << inputRank << ").";
  }

  auto outputType = getResult().getType().dyn_cast<RankedTensorType>();
  // If result is unranked, there is very little to verify statically.
  if (!outputType) return success();
  int outputRank = outputType.getRank();
  if (inputRank != outputRank) {
    return emitOpError() << "operand rank(" << inputRank
                         << ") must match result(" << outputRank << ").";
  }

  return success();
}

LogicalResult DynamicPadOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicPadOp::Adaptor adaptor(operands);
  Value operand = adaptor.operand();
  Value edgePaddingLow = adaptor.edge_padding_low();
  Value edgePaddingHigh = adaptor.edge_padding_high();
  Value interiorPadding = adaptor.interior_padding();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked pad a.t.m.
  if (!operandType) return failure();

  auto loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      edgePaddingLow.getType().cast<ShapedType>().getElementType();

  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  Value zero =
      toShapeScalarType(builder.create<arith::ConstantIndexOp>(loc, 0));
  Value one = toShapeScalarType(builder.create<arith::ConstantIndexOp>(loc, 1));

  for (int idx : llvm::seq<int>(0, operandType.getShape().size())) {
    Value valueDim =
        toShapeScalarType(builder.create<tensor::DimOp>(loc, operand, idx));
    Value offset = builder.create<arith::ConstantIndexOp>(loc, idx);
    Value valueLow =
        builder.create<tensor::ExtractOp>(loc, edgePaddingLow, offset);
    Value valueHigh =
        builder.create<tensor::ExtractOp>(loc, edgePaddingHigh, offset);
    Value valueInterior =
        builder.create<tensor::ExtractOp>(loc, interiorPadding, offset);
    // output_size = input_size + padding_low + padding_high + interior *
    // max(input_size - 1, 0)
    Value valueDimLessThanOne = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, valueDim, one);
    Value interiorSize = builder.create<arith::MulIOp>(
        loc, valueInterior,
        builder.create<mlir::arith::SelectOp>(
            loc, valueDimLessThanOne, zero,
            builder.create<arith::SubIOp>(loc, valueDim, one)));
    shapeValues.push_back(builder.create<arith::AddIOp>(
        loc,
        builder.create<arith::AddIOp>(
            loc, builder.create<arith::AddIOp>(loc, interiorSize, valueDim),
            valueLow),
        valueHigh));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  // If the operand type is dynamically shaped there is nothing to verify.
  auto operandTy = operand().getType().dyn_cast<RankedTensorType>();
  if (!operandTy || !operandTy.hasStaticShape()) return success();

  // If the operand type is statically shaped (not required) the number of
  // elements must match that of the result type.
  auto resultTy = getType().cast<RankedTensorType>();
  assert(resultTy && resultTy.hasStaticShape() &&
         "result type must be statically shaped");
  int64_t numResultElements = resultTy.getNumElements();
  int64_t numOperandElements = operandTy.getNumElements();
  if (numResultElements != numOperandElements)
    return emitOpError() << "number of output elements (" << numResultElements
                         << ") doesn't match expected number of elements ("
                         << numOperandElements << ")";

  return success();
}

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  if (getOperand().getType() == getType()) {
    return getOperand();
  }

  if (auto prevOp = getOperand().getDefiningOp<ReshapeOp>()) {
    setOperand(prevOp.getOperand());
    return getResult();
  }

  if (auto elements = operands.front().dyn_cast_or_null<DenseElementsAttr>()) {
    return reshape(elements, getResult().getType().cast<ShapedType>());
  }

  return {};
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<IdentityBroadcastReshape, IdentityBroadcastInDimReshape,
              EliminateRedundantReshape, EliminateIdentityReshape>(context);
}

//===----------------------------------------------------------------------===//
// ReplicaId Op
//===----------------------------------------------------------------------===//

LogicalResult ReplicaIdOp::inferReturnTypes(
    MLIRContext* context, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(RankedTensorType::get(
      /*shape=*/{}, IntegerType::get(context, 32, IntegerType::Unsigned)));
  return success();
}

//===----------------------------------------------------------------------===//
// AddDependency Op
//===----------------------------------------------------------------------===//

LogicalResult AddDependencyOp::inferReturnTypes(
    MLIRContext* context, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(operands.getTypes()[0]);
  return success();
}

//===----------------------------------------------------------------------===//
// If Op
//===----------------------------------------------------------------------===//

static LogicalResult verifyConditionalBranch(Operation* op, Region& region,
                                             llvm::Twine branchName) {
  if (region.getNumArguments() != 0)
    return op->emitOpError()
           << branchName << " must have 0 arguments, but found "
           << region.getNumArguments();

  TypeRange branchReturnTypes =
      region.front().getTerminator()->getOperandTypes();
  if (branchReturnTypes != op->getResultTypes())
    return op->emitOpError()
           << branchName << " returned types (" << branchReturnTypes
           << ") do not match op result types (" << op->getResultTypes() << ")";

  return success();
}

LogicalResult IfOp::verify() {
  if (failed(verifyConditionalBranch(*this, true_branch(),
                                     /*branchName=*/"true_branch"))) {
    return failure();
  }

  if (failed(verifyConditionalBranch(*this, false_branch(),
                                     /*branchName=*/"false_branch"))) {
    return failure();
  }
  return success();
}

static LogicalResult inlineIfConstantCondition(IfOp ifOp,
                                               PatternRewriter& rewriter) {
  DenseIntElementsAttr predAttr;
  if (!matchPattern(ifOp.pred(), m_Constant(&predAttr))) return failure();

  if (predAttr.getSplatValue<BoolAttr>().getValue()) {
    replaceOpWithRegion(rewriter, ifOp, ifOp.true_branch());
  } else {
    replaceOpWithRegion(rewriter, ifOp, ifOp.false_branch());
  }
  return success();
}

void IfOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add(&inlineIfConstantCondition);
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

LogicalResult CaseOp::verify() {
  auto numBranches = branches().size();

  for (unsigned i = 0; i < numBranches; ++i)
    if (failed(verifyConditionalBranch(*this, branches()[i],
                                       /*branchName=*/"branch " + Twine(i))))
      return failure();

  return success();
}

static LogicalResult inlineCaseConstantCondition(CaseOp caseOp,
                                                 PatternRewriter& rewriter) {
  DenseIntElementsAttr indexAttr;
  if (!matchPattern(caseOp.index(), m_Constant(&indexAttr))) {
    return failure();
  }
  int64_t index =
      indexAttr.getSplatValue<IntegerAttr>().getValue().getSExtValue();
  // For an OOB index, the last branch is executed as the default branch:
  // https://www.tensorflow.org/xla/operation_semantics#conditional
  if (index < 0 || index >= caseOp.getNumRegions())
    index = caseOp.getNumRegions() - 1;

  Region& region = caseOp.getRegion(index);
  if (!llvm::hasSingleElement(region)) return failure();
  replaceOpWithRegion(rewriter, caseOp, region);
  return success();
}

void CaseOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add(&inlineCaseConstantCondition);
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult SqrtOp::fold(ArrayRef<Attribute> operands) {
  auto val = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  if (!val) return {};

  auto type = getElementTypeOrSelf(getType());
  if (!type.isF32() && !type.isF64()) return {};

  auto shapedType = getType().cast<ShapedType>();
  if (!shapedType.hasStaticShape()) return {};

  int bitWidth = type.getIntOrFloatBitWidth();
  llvm::SmallVector<APFloat, 4> values;
  values.reserve(val.getNumElements());
  for (auto it : val.getValues<APFloat>()) {
    double value = bitWidth == 32 ? it.convertToFloat() : it.convertToDouble();
    if (value < 0) return {};
    value = std::sqrt(value);
    if (bitWidth == 32)
      values.emplace_back(static_cast<float>(value));
    else
      values.emplace_back(value);
  }
  return DenseFPElementsAttr::get(shapedType, values);
}

//===----------------------------------------------------------------------===//
// UnaryOps
//===----------------------------------------------------------------------===//

ParseResult parseUnaryOp(OpAsmParser& parser, OperationState& result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  Type type;
  // If the operand is in-between parentheses, use generic form.
  SMLoc loc = parser.getCurrentLocation();
  if (!parser.parseOptionalLParen()) {
    if (parser.parseOperandList(operands) || parser.parseRParen() ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseType(type))
      return failure();
    auto fnType = type.dyn_cast<FunctionType>();
    if (!fnType) {
      parser.emitError(loc, "expected function type");
      return failure();
    }
    if (parser.resolveOperands(operands, fnType.getInputs(), loc,
                               result.operands))
      return failure();
    result.addTypes(fnType.getResults());
    return success();
  }
  // Otherwise, use shorthand syntax.
  return failure(parser.parseOperandList(operands) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperands(operands, type, result.operands) ||
                 parser.addTypeToList(type, result.types));
}

void printUnaryOp(Operation* op, OpAsmPrinter& p) {
  assert(op->getNumResults() == 1 && "op should have one result");
  assert(op->getNumOperands() == 1 && "op should have one input");
  // If not all types are the same, use generic form.
  auto resultType = op->getResult(0).getType();
  if (resultType != op->getOperandTypes()[0]) {
    p.printGenericOp(op, /*printOpName=*/false);
    return;
  }
  // Otherwise, use the shorthand syntax.
  p << ' ';
  p.printOperands(op->getOperands());
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << resultType;
}

template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
static Attribute UnaryFolder(Op* op, ArrayRef<Attribute> attrs) {
  if (!attrs[0]) return {};

  DenseElementsAttr val = attrs[0].dyn_cast<DenseElementsAttr>();
  if (!val) return {};

  ShapedType type = op->getType().template cast<ShapedType>();
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!etype.isa<ElementType>()) {
    return {};
  }

  SmallVector<ValType, 6> values;
  values.reserve(val.getNumElements());
  for (const auto v : val.getValues<ValType>()) {
    values.push_back(Convert()(v));
  }

  return DenseElementsAttr::get(type, values);
}

struct Round {
  APFloat operator()(const APFloat& f) {
    APFloat r = f;
    r.roundToIntegral(llvm::RoundingMode::NearestTiesToAway);
    return r;
  }
};

struct LogicalNot {
  APInt operator()(const APInt& i) {
    return APInt(i.getBitWidth(), static_cast<uint64_t>(!i));
  }
};

template <typename FloatOrInt>
struct Sign {
  APFloat compute(const APFloat& f) {
    if (f.isZero() || f.isNaN()) return f;
    double value = f.isNegative() ? -1.0 : 1.0;
    APFloat val(value);
    bool unused;
    val.convert(f.getSemantics(), APFloat::rmNearestTiesToEven, &unused);
    return val;
  }

  APInt compute(const APInt& i) {
    APInt r = i;
    if (r == 0) return r;
    if (r.isNegative()) {
      return APInt(r.getBitWidth(), -1, /*isSigned=*/true);
    }
    return APInt(r.getBitWidth(), 1, /*isSigned=*/true);
  }

  FloatOrInt operator()(const FloatOrInt& fi) { return compute(fi); }
};

#define UNARY_FOLDER(Op, Func)                                                \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                          \
    if (getElementTypeOrSelf(getType()).isa<FloatType>())                     \
      return UnaryFolder<Op, FloatType, APFloat, Func<APFloat>>(this, attrs); \
    if (getElementTypeOrSelf(getType()).isa<IntegerType>())                   \
      return UnaryFolder<Op, IntegerType, APInt, Func<APInt>>(this, attrs);   \
    return {};                                                                \
  }

#define UNARY_FOLDER_INT(Op, Func)                                   \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                 \
    if (getElementTypeOrSelf(getType()).isa<IntegerType>())          \
      return UnaryFolder<Op, IntegerType, APInt, Func>(this, attrs); \
    return {};                                                       \
  }

#define UNARY_FOLDER_FLOAT(Op, Func)                                 \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                 \
    if (getElementTypeOrSelf(getType()).isa<FloatType>())            \
      return UnaryFolder<Op, FloatType, APFloat, Func>(this, attrs); \
    return {};                                                       \
  }

UNARY_FOLDER(NegOp, std::negate);
UNARY_FOLDER(SignOp, Sign);
UNARY_FOLDER_INT(NotOp, LogicalNot);
UNARY_FOLDER_FLOAT(RoundOp, Round);

#undef UNARY_FOLDER
#undef UNARY_FOLDER_INT
#undef UNARY_FOLDER_FLOAT

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {

// Updates the element type of a (presumed) tensor type 'x', returning either
// a permuted UnrankedTensorType or RankedTensorType.
static Type updateResultElementType(Builder* builder, Type x,
                                    Type elementType) {
  auto xRanked = x.dyn_cast<RankedTensorType>();
  if (!xRanked) {
    return UnrankedTensorType::get(elementType);
  }

  auto shapeX = xRanked.getShape();
  return RankedTensorType::get(shapeX, elementType);
}
}  // namespace

ParseResult parseBinaryOp(OpAsmParser& parser, OperationState& result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  Type type;
  // If the operand list is in-between parentheses, use generic form.
  SMLoc loc = parser.getCurrentLocation();
  if (!parser.parseOptionalLParen()) {
    if (parser.parseOperandList(operands) || parser.parseRParen() ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseType(type))
      return failure();
    auto fnType = type.dyn_cast<FunctionType>();
    if (!fnType) {
      parser.emitError(loc, "expected function type");
      return failure();
    }
    if (parser.resolveOperands(operands, fnType.getInputs(), loc,
                               result.operands))
      return failure();
    result.addTypes(fnType.getResults());
    return success();
  }
  // Otherwise, use shorthand syntax.
  return failure(parser.parseOperandList(operands) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperands(operands, type, result.operands) ||
                 parser.addTypeToList(type, result.types));
}

void printBinaryOp(Operation* op, OpAsmPrinter& p) {
  assert(op->getNumResults() == 1 && "op should have one result");
  // If not all types are the same, use generic form.
  auto resultType = op->getResult(0).getType();
  if (llvm::any_of(op->getOperandTypes(),
                   [&](Type type) { return type != resultType; })) {
    p.printGenericOp(op, /*printOpName=*/false);
    return;
  }
  // Otherwise, use the shorthand syntax.
  p << ' ';
  p.printOperands(op->getOperands());
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << resultType;
}

static const APFloat& addSign(const APFloat& v, Type) { return v; }
static APSInt addSign(const APInt& v, Type t) {
  // Add signedness information to the value, treating signless as signed.
  return APSInt(v, t.isUnsignedInteger());
}

template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
static Attribute BinaryFolder(Op* op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1]) return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs) return {};

  ShapedType type = op->getType().template cast<ShapedType>();
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!etype.isa<ElementType>()) {
    return {};
  }

  // Special case for folding splats no matter how large.
  // Only covers the case of both attrs being splats; operation-specific cases
  // like adding a zero or multiplying by one are handled elsewhere.
  SplatElementsAttr splatLhs = lhs.dyn_cast<SplatElementsAttr>();
  SplatElementsAttr splatRhs = rhs.dyn_cast<SplatElementsAttr>();
  if (splatLhs && splatRhs) {
    auto signedLhs = addSign(splatLhs.getSplatValue<ValType>(), etype);
    auto signedRhs = addSign(splatRhs.getSplatValue<ValType>(), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    return succeeded(result) ? SplatElementsAttr::get(type, *result)
                             : Attribute();
  }

  // Prevent folding if lhs/rhs are too large.
  if (lhs.getNumElements() > kFoldBinaryOpEltLimit) {
    return {};
  }

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<ValType>(), rhs.getValues<ValType>())) {
    auto signedLhs = addSign(std::get<0>(zip), etype);
    auto signedRhs = addSign(std::get<1>(zip), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    if (failed(result)) {
      return {};
    }
    values.push_back(std::move(*result));
  }

  return DenseElementsAttr::get(type, values);
}

template <typename T>
struct Divide : std::divides<T> {};

template <>
struct Divide<APSInt> {
  FailureOr<APSInt> operator()(const APSInt& a, const APSInt& b) const {
    if (b.isZero()) return failure();
    return a / b;
  }
};

template <typename T>
struct Remainder : std::modulus<T> {};

template <>
struct Remainder<APSInt> {
  FailureOr<APSInt> operator()(const APSInt& a, const APSInt& b) const {
    if (b.isZero()) return failure();
    return a % b;
  }
};

template <>
struct Remainder<APFloat> {
  APFloat operator()(const APFloat& a, const APFloat& b) const {
    APFloat result(a);
    result.remainder(b);
    return result;
  }
};

template <typename T>
struct Max {
  T operator()(const T& a, const T& b) const { return std::max<T>(a, b); }
};

template <typename T>
struct Min {
  T operator()(const T& a, const T& b) const { return std::min<T>(a, b); }
};

#define BINARY_FOLDER_INTERNAL(Op, Func)                                     \
  if (getElementTypeOrSelf(getType()).isa<FloatType>())                      \
    return BinaryFolder<Op, FloatType, APFloat, Func<APFloat>>(this, attrs); \
  if (getElementTypeOrSelf(getType()).isa<IntegerType>())                    \
    return BinaryFolder<Op, IntegerType, APInt, Func<APSInt>>(this, attrs);  \
  return {};

#define BINARY_FOLDER(Op, Func)                      \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) { \
    BINARY_FOLDER_INTERNAL(Op, Func)                 \
  }

// Addition, subtraction and multiplication use the std:: versions of the ops.
// Due to the other ops behaving differently in signed vs unsigned integers,
// APInts need a special implementation. Currently, it replicates signed int
// op behavior.
BINARY_FOLDER(SubOp, std::minus);
BINARY_FOLDER(DivOp, Divide);
BINARY_FOLDER(RemOp, Remainder);
BINARY_FOLDER(MaxOp, Max);
BINARY_FOLDER(MinOp, Min);

bool isSplatZero(SplatElementsAttr attr) {
  if (!attr) return false;
  if (attr.getElementType().isa<FloatType>()) {
    return attr.getSplatValue<APFloat>().isZero();
  } else if (attr.getElementType().isa<IntegerType>()) {
    return attr.getSplatValue<APInt>().isZero();
  } else {
    return false;
  }
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> attrs) {
  // Handle special case where one operand is 0:  x + 0 => x
  if (attrs[0] || attrs[1]) {
    SplatElementsAttr splatLhs = attrs[0].dyn_cast_or_null<SplatElementsAttr>();
    SplatElementsAttr splatRhs = attrs[1].dyn_cast_or_null<SplatElementsAttr>();
    if (isSplatZero(splatLhs)) return splatRhs ? (OpFoldResult)splatRhs : rhs();
    if (isSplatZero(splatRhs)) return splatLhs ? (OpFoldResult)splatLhs : lhs();
  }
  if (attrs[0] && attrs[1]) {
    BINARY_FOLDER_INTERNAL(AddOp, std::plus)
  }
  return {};
}

bool isSplatOne(SplatElementsAttr attr) {
  if (!attr) return false;
  if (attr.getElementType().isa<FloatType>()) {
    return attr.getSplatValue<APFloat>().convertToDouble() == 1.0;
  } else if (attr.getElementType().isa<IntegerType>()) {
    return attr.getSplatValue<APInt>().getSExtValue() == 1;
  } else {
    return false;
  }
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> attrs) {
  // Handle special case where one operand is 1: x * 1 => x
  if (attrs[0] || attrs[1]) {
    SplatElementsAttr splatLhs = attrs[0].dyn_cast_or_null<SplatElementsAttr>();
    SplatElementsAttr splatRhs = attrs[1].dyn_cast_or_null<SplatElementsAttr>();
    if (isSplatOne(splatLhs)) return splatRhs ? (OpFoldResult)splatRhs : rhs();
    if (isSplatOne(splatRhs)) return splatLhs ? (OpFoldResult)splatLhs : lhs();
  }
  if (attrs[0] && attrs[1]) {
    BINARY_FOLDER_INTERNAL(MulOp, std::multiplies);
  }
  return {};
}

#undef BINARY_FOLDER_INTERNAL
#undef BINARY_FOLDER

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// Returns output dimension size for slice result for the given arguments.
// Returns -1 if arguments are illegal.
static int64_t inferSliceDim(int64_t inputDim, int64_t start, int64_t end,
                             int64_t stride) {
  if (inputDim == -1 || start < 0 || start > end || end > inputDim ||
      stride == 0)
    return -1;

  return llvm::divideCeil(end - start, stride);
}

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  SliceOpAdaptor slice(operands, attributes);
  // TODO(jpienaar): Update this code after refactoring verify.
  if (failed(slice.verify(location.getValueOr(UnknownLoc::get(context))))) {
    return failure();
  }

  Type ty = slice.operand().getType();
  RankedTensorType rankedTy = ty.dyn_cast<RankedTensorType>();
  if (!rankedTy) {
    // The operand type is unranked, so the best we can infer for the result
    // type is an unranked tensor with the same element type as the operand
    // type.
    inferredReturnTypes.assign({ty});
    return success();
  }

  ShapedType attrTy = slice.start_indices().getType();
  if (attrTy.getRank() != 1) {
    return emitOptionalError(location, "start_indices has rank ",
                             attrTy.getRank(), " instead of required rank 1");
  }

  int64_t rank = rankedTy.getRank();
  if (attrTy.getNumElements() != rank) {
    return emitOptionalError(
        location, "the number of elements in start_indices (",
        attrTy.getNumElements(), ") does not match the rank of the operand (",
        rank, ")");
  }

  SmallVector<int64_t, 4> start(slice.start_indices().getValues<int64_t>());
  SmallVector<int64_t, 4> limit(slice.limit_indices().getValues<int64_t>());
  SmallVector<int64_t, 4> strideVals(slice.strides().getValues<int64_t>());

  SmallVector<int64_t, 4> shape;
  shape.reserve(rank);
  for (int64_t i = 0, e = rank; i != e; i++) {
    shape.push_back(inferSliceDim(rankedTy.getDimSize(i), start[i], limit[i],
                                  strideVals[i]));
  }
  inferredReturnTypes.assign(
      {RankedTensorType::get(shape, rankedTy.getElementType())});
  return success();
}

template <typename I, typename E>
static void sliceElements(I values, ArrayRef<int64_t> sizes,
                          ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
                          ArrayRef<int64_t> strides,
                          llvm::SmallVectorImpl<E>* outValues) {
  assert(starts.size() == limits.size());
  assert(starts.size() == strides.size());
  if (starts.empty()) return;

  int64_t start = starts.front();
  int64_t limit = limits.front();
  int64_t stride = strides.front();
  if (starts.size() == 1) {
    for (int i = start; i < limit; i += stride) {
      outValues->push_back(*(values + i));
    }
    return;
  }

  for (; start < limit; start += stride) {
    auto begin = values + start * sizes.front();
    sliceElements<I, E>(begin, sizes.drop_front(), starts.drop_front(),
                        limits.drop_front(), strides.drop_front(), outValues);
  }
}

template <typename I, typename E>
static Attribute foldSlice(SliceOp* op, I values) {
  auto start = llvm::to_vector<6>(op->start_indices().getValues<int64_t>());
  auto limit = llvm::to_vector<6>(op->limit_indices().getValues<int64_t>());
  auto stride = llvm::to_vector<6>(op->strides().getValues<int64_t>());

  auto resultType = op->operand().getType().cast<ShapedType>();
  if (!resultType.hasStaticShape()) return {};

  auto shape = resultType.getShape();
  int64_t count = resultType.getNumElements();
  if (count == 0) {
    return DenseElementsAttr::get<E>(
        op->getResult().getType().cast<ShapedType>(),
        /*list=*/{});
  }

  // Compute the striding for each dimension.
  llvm::SmallVector<int64_t, 6> sizes;
  sizes.reserve(shape.size());
  for (auto v : shape) {
    count = count / v;
    sizes.push_back(count);
  }

  llvm::SmallVector<E, 6> outValues;
  outValues.reserve(resultType.getNumElements());
  sliceElements<I, E>(values, sizes, start, limit, stride, &outValues);

  return DenseElementsAttr::get(op->getResult().getType().cast<ShapedType>(),
                                outValues);
}

OpFoldResult SliceOp::fold(ArrayRef<Attribute> operands) {
  // Check if the SliceOp is a NoOp operation.
  auto operandType = getOperand().getType().cast<ShapedType>();
  auto resultType = getResult().getType().cast<ShapedType>();

  if (operandType.hasStaticShape() && resultType.hasStaticShape() &&
      (operandType.getShape() == resultType.getShape())) {
    return getOperand();
  }

  if (operands.empty() || !operands.front()) return {};

  // Evaluate for statically valued inputs.
  DenseElementsAttr elements = operands.front().dyn_cast<DenseElementsAttr>();
  if (!elements) return {};

  auto etype = elements.getType().getElementType();
  if (etype.isa<IntegerType>()) {
    return foldSlice<DenseElementsAttr::IntElementIterator, APInt>(
        this, elements.value_begin<APInt>());
  }
  if (etype.isa<FloatType>()) {
    return foldSlice<DenseElementsAttr::FloatElementIterator, APFloat>(
        this, elements.value_begin<APFloat>());
  }

  return {};
}

namespace {
// In cases where a concat is fed into a slice, it is possible the concat
// can be simplified or bypassed. This checks which inputs to the concat are
// used by the slice, either reducing the number of concatenated values or
// entirely removes the concat.
struct SimplifyConcatSlice : public OpRewritePattern<SliceOp> {
  using OpRewritePattern<SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp slice,
                                PatternRewriter& rewriter) const override {
    auto resultTy = slice.getType().cast<ShapedType>();
    if (!resultTy.hasStaticShape()) {
      return failure();
    }

    auto sliceInput = slice.operand();
    auto sliceInputTy = sliceInput.getType().cast<ShapedType>();
    auto concat = sliceInput.getDefiningOp<ConcatenateOp>();
    if (!concat) {
      return failure();
    }

    auto dimension = concat.dimension();

    auto start = slice.start_indices().getValues<APInt>();
    auto limit = slice.limit_indices().getValues<APInt>();

    auto sliceStart = (*(start.begin() + dimension)).getSExtValue();
    auto sliceLimit = (*(limit.begin() + dimension)).getSExtValue();

    // We need to determine what inputs from the concat affect the slice, and
    // how the bounds of the slice need to be updated for the minimally required
    // inputs.
    int64_t runningSize = 0;
    int64_t frontOffset = sliceInputTy.getShape()[dimension];

    auto subsetStart = concat.operand_end();
    auto subsetEnd = concat.operand_end();
    for (auto it = concat.operand_begin(); it < concat.operand_end(); ++it) {
      auto input = *it;
      ShapedType inputTy = input.getType().cast<ShapedType>();
      if (inputTy.isDynamicDim(dimension)) {
        return failure();
      }
      auto dimSize = inputTy.getShape()[dimension];

      // If this position is in the slice its the start of the subset and we
      // need to update the start and limit values.
      if (runningSize + dimSize > sliceStart &&
          subsetStart == concat.operand_end()) {
        subsetStart = it;
        frontOffset = runningSize;
      }

      // Determine the last required offset.
      if (runningSize < sliceLimit) {
        subsetEnd = it + 1;
      }

      runningSize += dimSize;
    }

    auto subsetSize = subsetEnd - subsetStart;
    // We need all inputs so no optimization.
    if (subsetSize == concat.getNumOperands()) {
      return failure();
    }

    // If there's nothing to slice that means the output is an empty tensor and
    // there is dead code. We do nothing here and rely on other passes to clean
    // this up.
    if (subsetSize == 0) {
      return failure();
    }

    if (subsetSize > 1 && !concat.getResult().hasOneUse()) {
      return failure();
    }

    auto concatRange = OperandRange(subsetStart, subsetEnd);
    auto newConcat = rewriter.create<ConcatenateOp>(
        concat.getLoc(), concatRange, concat.dimension());

    llvm::SmallVector<APInt, 6> newStart(start);
    llvm::SmallVector<APInt, 6> newLimit(limit);
    newStart[dimension] -= frontOffset;
    newLimit[dimension] -= frontOffset;

    auto attrType = slice.start_indices().getType().cast<ShapedType>();
    auto create = rewriter.create<SliceOp>(
        slice.getLoc(), newConcat,
        DenseIntElementsAttr::get(attrType, newStart),
        DenseIntElementsAttr::get(attrType, newLimit), slice.strides());
    rewriter.replaceOp(slice, create.getResult());
    return success();
  }
};
}  // namespace

void SliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<SimplifyConcatSlice>(context);
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

void SortOp::build(OpBuilder& builder, OperationState& state,
                   ValueRange operands, int64_t dimension, bool isStable) {
  state.addOperands(operands);
  state.addAttribute("dimension", builder.getI64IntegerAttr(dimension));
  state.addAttribute("is_stable", builder.getBoolAttr(isStable));

  for (Value operand : operands) state.addTypes(operand.getType());

  state.addRegion();
}

LogicalResult SortOp::verify() {
  Operation::operand_range operands = this->operands();
  if (operands.empty()) return emitOpError("requires at least one input");

  // TODO(antiagainst): verify partionally dynamic shapes
  if (llvm::all_of(operands, [](Value operand) {
        return operand.getType().cast<ShapedType>().hasRank();
      })) {
    ArrayRef<int64_t> inputShape =
        (*operands.begin()).getType().cast<ShapedType>().getShape();

    if (llvm::any_of(llvm::drop_begin(operands, 1), [&](Value operand) {
          return operand.getType().cast<ShapedType>().getShape() != inputShape;
        }))
      return emitOpError("requires all inputs to have the same dimensions");

    int64_t rank = inputShape.size();
    int64_t cmpDim = dimension();
    if (cmpDim < -rank || cmpDim >= rank)
      return emitOpError("dimension attribute value must be in range [-")
             << rank << ", " << rank << "), but found " << cmpDim;
  }

  Block& block = comparator().front();
  size_t numOperands = getOperation()->getNumOperands();
  if (block.getNumArguments() != 2 * numOperands)
    return emitOpError("comparator block should have ")
           << 2 * numOperands << " arguments";

  for (const auto& indexedOperand : llvm::enumerate(operands)) {
    int index = indexedOperand.index();
    Type elementType =
        indexedOperand.value().getType().cast<ShapedType>().getElementType();
    Type tensorType = RankedTensorType::get({}, elementType);
    for (int i : {2 * index, 2 * index + 1}) {
      Type argType = block.getArgument(i).getType();
      if (argType != tensorType)
        return emitOpError("comparator block argument #")
               << i << " should be of type " << tensorType << " but got "
               << argType;
    }
  }

  // Mapped computation must return single output.
  auto comparatorResult = block.getTerminator()->getOperands();
  if (comparatorResult.size() != 1)
    return emitOpError() << "comparator must return single output, but got: "
                         << comparatorResult.size();

  // The output of computation must be 0-ranked tensor with element-type i1.
  auto comparatorResultType =
      comparatorResult[0].getType().dyn_cast<RankedTensorType>();
  if (!comparatorResultType || comparatorResultType.getRank() != 0 ||
      !comparatorResultType.getElementType().isInteger(1))
    return emitOpError() << "comparator must return tensor<i1>, but got: "
                         << comparatorResult[0].getType();

  // check number of return-values and their element-types.
  auto resultTypes = getResultTypes();
  if (resultTypes.size() != numOperands)
    return emitOpError() << "expects the number of results to be same as "
                            "number of operands. Got number of results = "
                         << resultTypes.size()
                         << " and number of operands = " << numOperands;

  for (auto it : llvm::zip(operands, getResultTypes()))
    if (std::get<0>(it).getType().cast<TensorType>().getElementType() !=
        std::get<1>(it).cast<TensorType>().getElementType())
      return emitOpError()
             << "expects the operands and results to have pairwize equal "
                "element-types, but got "
             << std::get<0>(it).getType().cast<TensorType>().getElementType()
             << " vs " << std::get<1>(it).cast<TensorType>().getElementType();

  return success();
}

/// Drops the operands if the results are not used and they are not used in
/// op.comparator().
static LogicalResult sortDropEmptyUseArgs(SortOp op,
                                          PatternRewriter& rewriter) {
  DenseSet<unsigned> erasedArgs;
  unsigned numOperands = op.getNumOperands();
  for (unsigned i = 0; i < numOperands; ++i) {
    if (!op.getResult(i).use_empty()) continue;
    Block& block = op.comparator().front();
    if (!block.getArgument(i * 2).use_empty()) continue;
    if (!block.getArgument(i * 2 + 1).use_empty()) continue;
    erasedArgs.insert(i);
  }
  if (erasedArgs.empty()) return failure();

  SmallVector<Value> newOperands;
  SmallVector<unsigned> erasedBlockArgs;
  for (const auto& en : llvm::enumerate(op.operands())) {
    if (erasedArgs.contains(en.index())) {
      erasedBlockArgs.push_back(en.index() * 2);
      erasedBlockArgs.push_back(en.index() * 2 + 1);
    } else {
      newOperands.push_back(en.value());
    }
  }

  auto newOp = rewriter.create<SortOp>(op.getLoc(), newOperands, op.dimension(),
                                       op.is_stable());
  Region& region = newOp.comparator();
  rewriter.inlineRegionBefore(op.comparator(), region, region.end());
  region.front().eraseArguments(erasedBlockArgs);

  SmallVector<Value> results;
  for (unsigned i = 0, j = 0; i < numOperands; ++i) {
    if (erasedArgs.contains(i)) {
      results.push_back({});
    } else {
      results.push_back(newOp.getResult(j++));
    }
  }
  rewriter.replaceOp(op, results);

  return success();
}

/// Set the sorting dimension to the last dimension if it's not set and the rank
/// is known.
static LogicalResult sortOpInferDefaultDimension(SortOp op,
                                                 PatternRewriter& rewriter) {
  auto ty = op.getResultTypes()[0].dyn_cast<ShapedType>();
  if (!ty) {
    return failure();
  }
  if (op.dimension() != -1) {
    return failure();
  }

  IntegerAttr dim = rewriter.getI64IntegerAttr(ty.getRank() - 1);
  auto newOp = rewriter.create<SortOp>(op.getLoc(), op.getResultTypes(),
                                       op.operands(), dim, op.is_stableAttr());
  Region& region = newOp.comparator();
  rewriter.inlineRegionBefore(op.comparator(), region, region.end());
  rewriter.replaceOp(op, newOp.getResults());

  return success();
}

void SortOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* /*context*/) {
  results.add(sortDropEmptyUseArgs);
  results.add(sortOpInferDefaultDimension);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  if (auto elements = operands.front().dyn_cast_or_null<SplatElementsAttr>()) {
    return elements.reshape(getResult().getType().cast<ShapedType>());
  }
  for (const auto& it : llvm::enumerate(permutation().getValues<APInt>())) {
    if (it.index() != it.value()) {
      return {};
    }
  }
  return getOperand();
}

// transpose(transpose(X)) => transpose(X)
static LogicalResult eliminateRedundantTranspse(TransposeOp op,
                                                PatternRewriter& rewriter) {
  auto tranposeOperand = op.operand().getDefiningOp<TransposeOp>();
  if (!tranposeOperand) {
    return failure();
  }
  auto operandPermutation = tranposeOperand.permutation().getValues<APInt>();
  auto newPermutation =
      op.permutation()
          .mapValues(op.permutation().getElementType(),
                     [&operandPermutation](const APInt& index) -> APInt {
                       return operandPermutation[index.getSExtValue()];
                     })
          .cast<DenseIntElementsAttr>();
  rewriter.replaceOpWithNewOp<TransposeOp>(
      op, op.getResult().getType(), tranposeOperand.operand(), newPermutation);
  return success();
}

// transpose(broadcast_in_dim(X)) => broadcast_in_dim(X)
static LogicalResult eliminateBroadcastInDimTranspose(
    TransposeOp op, PatternRewriter& rewriter) {
  auto broadcastInDimOp = op.operand().getDefiningOp<BroadcastInDimOp>();
  if (!broadcastInDimOp) {
    return failure();
  }
  DenseIntElementsAttr broadcastDimensions =
      broadcastInDimOp.broadcast_dimensions();
  DenseIntElementsAttr permutation = op.permutation();
  SmallVector<int64_t> newBroadcastDimensions;
  for (auto dimension : broadcastDimensions.getValues<int64_t>()) {
    int64_t index = 0;
    for (auto p : permutation.getValues<int64_t>()) {
      if (p == dimension) {
        newBroadcastDimensions.push_back(index);
        break;
      }
      index++;
    }
  }
  rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
      op, op->getResultTypes(), broadcastInDimOp.operand(),
      rewriter.getI64TensorAttr(newBroadcastDimensions));
  return success();
}

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* /*context*/) {
  results.add(eliminateRedundantTranspse);
  results.add(eliminateBroadcastInDimTranspose);
}

LogicalResult TransposeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  TransposeOp::Adaptor adaptor(operands);
  Value operand = adaptor.operand();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  SmallVector<int64_t, 4> permutation(this->permutation().getValues<int64_t>());
  SmallVector<Value, 4> shapeValues(permutation.size());

  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto& element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto* it = std::find(permutation.begin(), permutation.end(), idx);
    Value valueDim = toShapeScalarType(
        builder.createOrFold<tensor::DimOp>(loc, operand, element.index()));
    shapeValues[std::distance(permutation.begin(), it)] = valueDim;
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

// Method for InferTypeOpInterface: infer the return type from the operand type
// and the permutation.
LogicalResult TransposeOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnTypes) {
  auto type = operands[0].getType();
  auto rankedTy = type.dyn_cast<RankedTensorType>();
  if (!rankedTy) {
    auto shapedTy = type.dyn_cast<ShapedType>();
    if (!shapedTy)
      return emitOptionalError(loc,
                               "expected shaped type operand, got: ", type);
    inferredReturnTypes.emplace_back(shapedTy);
    return success();
  }
  auto permutation = attributes.getAs<DenseIntElementsAttr>("permutation");
  int64_t rank = rankedTy.getRank();
  if (!permutation)
    return emitOptionalError(loc,
                             "missing permutation attribute on TransposeOp");

  if (permutation.getType().getRank() != 1)
    return emitOptionalError(loc, "TransposeOp permutation has rank ",
                             permutation.getType().getRank(),
                             " instead of rank 1");

  if (permutation.size() != rank)
    return emitOptionalError(loc, "TransposeOp operand rank ", rank,
                             " does not match permutation size ",
                             permutation.size());

  SmallVector<int64_t> resultShape;
  ArrayRef<int64_t> inputShape = rankedTy.getShape();
  for (int64_t dim : permutation.getValues<int64_t>()) {
    if (dim >= rank) return failure();
    resultShape.push_back(inputShape[dim]);
  }
  inferredReturnTypes.emplace_back(resultShape, rankedTy.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// TriangularSolveOp
//===----------------------------------------------------------------------===//

LogicalResult TriangularSolveOp::verify() {
  auto aType = a().getType().dyn_cast<RankedTensorType>();

  // Skip verifier if a is unranked tensor.
  if (!aType) return success();

  // Check that a should have rank >= 2
  auto aRank = aType.getRank();
  if (aRank < 2)
    return emitOpError() << "operand 'a' must have rank >= 2, but got "
                         << aType;

  // The two minor dimensions of a must have same size.
  if (aType.getDimSize(aRank - 2) != aType.getDimSize(aRank - 1))
    return emitOpError() << "two minor dimensions of operand 'a' must have "
                            "equal size, but got "
                         << aType;

  auto bType = b().getType().dyn_cast<RankedTensorType>();
  // If b is unranked skip remaining checks.
  if (!bType) return success();

  // Check that a and b have same rank.
  auto bRank = bType.getRank();
  if (aRank != bRank)
    return emitOpError() << "operands must have equal rank, but got " << aType
                         << " and " << bType;

  // The shared dimension of a and b should match.
  if (aType.getDimSize(aRank - 1) !=
      bType.getDimSize(bRank - (left_side() ? 2 : 1)))
    return emitOpError() << "shared dimension of operands 'a' and 'b' does "
                            "not match, but got "
                         << aType << " and " << bType;

  // The leading batch dimensions of a and b must be equal.
  auto aBatchDims = aType.getShape().drop_back(2);
  auto bBatchDims = bType.getShape().drop_back(2);
  if (aBatchDims != bBatchDims)
    return emitOpError()
           << "leading batch dimensions of the operands must be same, but got "
           << aType << " and " << bType;

  // Result and argument b must have same shape.
  auto resultType = getType().dyn_cast<RankedTensorType>();
  if (!resultType) return success();
  if (resultType != bType)
    return emitOpError()
           << "result and operand 'b' must have same shape, but got "
           << resultType << " and " << bType;
  return success();
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleElementOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto tupleType = operands[0].getType().dyn_cast<TupleType>();
  if (!tupleType) return failure();

  auto indexAttr = attributes.get("index").cast<IntegerAttr>();
  auto index = indexAttr.getInt();
  if (index < 0 || index >= tupleType.size()) return failure();

  inferredReturnTypes.push_back(tupleType.getType(index));
  return success();
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext* context, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(TupleType::get(context, TypeRange(operands)));
  return success();
}

//===----------------------------------------------------------------------===//
// UnaryEinsumOp
//===----------------------------------------------------------------------===//

void UnaryEinsumOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<UnaryEinsumToEinsum>(context);
}

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

void CompareOp::build(OpBuilder& builder, OperationState& result, Value lhs,
                      Value rhs, ComparisonDirection comparisonDirection,
                      ComparisonType compareType) {
  build(builder, result, lhs, rhs,
        ComparisonDirectionAttr::get(builder.getContext(), comparisonDirection),
        ComparisonTypeAttr::get(builder.getContext(), compareType));
}

LogicalResult CompareOp::inferReturnTypeComponents(
    mlir::MLIRContext* ctx, llvm::Optional<mlir::Location>,
    ValueShapeRange operands, mlir::DictionaryAttr, mlir::RegionRange,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnTypes) {
  ShapedTypeComponents& components =
      inferredReturnTypes.emplace_back(IntegerType::get(ctx, /*width=*/1));
  auto argTy = operands.front().getType().cast<TensorType>();
  if (argTy.hasRank()) {
    components =
        ShapedTypeComponents(argTy.getShape(), components.getElementType());
  }
  return success();
}

LogicalResult CompareOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                &reifiedReturnShapes);
}

template <typename T>
struct Less : std::less<T> {};

template <>
struct Less<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.slt(b); }
};

template <typename T>
struct LessEqual : std::less_equal<T> {};

template <>
struct LessEqual<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.sle(b); }
};

template <typename T>
struct Greater : std::greater<T> {};

template <>
struct Greater<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.sgt(b); }
};

template <typename T>
struct GreaterEqual : std::greater_equal<T> {};

template <>
struct GreaterEqual<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.sge(b); }
};

template <typename Op, typename ElementType, typename SrcType, typename Convert>
static Attribute CompareFolder(CompareOp op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1]) return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs) return {};

  ShapedType operandType =
      op.getOperand(0).getType().template cast<ShapedType>();
  if (!operandType.hasStaticShape()) {
    return {};
  }

  if (!operandType.getElementType().isa<ElementType>()) {
    return {};
  }

  SmallVector<bool, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<SrcType>(), rhs.getValues<SrcType>())) {
    values.push_back(Convert()(std::get<0>(zip), std::get<1>(zip)));
  }

  auto resultTy = op.getType().cast<ShapedType>();
  return DenseElementsAttr::get(resultTy, values);
}

OpFoldResult CompareOp::fold(ArrayRef<Attribute> operands) {
  auto resultTy = getType().cast<ShapedType>();
  if (!resultTy.hasStaticShape()) return {};

  auto direction = comparison_direction();
  auto lhsTy = getElementTypeOrSelf(lhs());
  if (lhs() == rhs() && !lhsTy.isa<FloatType>() &&
      (!lhsTy.isa<ComplexType>() ||
       !lhsTy.cast<ComplexType>().getElementType().isa<FloatType>())) {
    if (direction == ComparisonDirection::LE ||
        direction == ComparisonDirection::EQ ||
        direction == ComparisonDirection::GE) {
      return DenseIntElementsAttr::get(resultTy, {true});
    }
    return DenseIntElementsAttr::get(resultTy, {false});
  }

  auto opElType = lhs().getType().cast<ShapedType>().getElementType();
  // Fold tensor<*xi1> != false to just return tensor<*xi1>
  if (direction == ComparisonDirection::NE && opElType.isInteger(1)) {
    DenseIntElementsAttr cstAttr;
    if (matchPattern(lhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && !cstAttr.getSplatValue<bool>()) {
        return rhs();
      }
    }

    if (matchPattern(rhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && !cstAttr.getSplatValue<bool>()) {
        return lhs();
      }
    }
  }

  // Fold tensor<*xi1> == True to just return tensor<*xi1>
  if (direction == ComparisonDirection::EQ && opElType.isInteger(1)) {
    DenseIntElementsAttr cstAttr;
    if (matchPattern(lhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && cstAttr.getSplatValue<bool>()) {
        return rhs();
      }
    }

    if (matchPattern(rhs(), m_Constant(&cstAttr))) {
      if (cstAttr.isSplat() && cstAttr.getSplatValue<bool>()) {
        return lhs();
      }
    }
  }

  if (!operands[0] || !operands[1]) {
    return {};
  }

#define COMPARE_FOLDER(Op, comparison, Func)                                \
  if (direction == comparison) {                                            \
    if (auto folded = CompareFolder<Op, FloatType, APFloat, Func<APFloat>>( \
            *this, operands))                                               \
      return folded;                                                        \
    if (auto folded = CompareFolder<Op, IntegerType, APInt, Func<APInt>>(   \
            *this, operands))                                               \
      return folded;                                                        \
  }

  COMPARE_FOLDER(CompareOp, ComparisonDirection::EQ, std::equal_to);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::NE, std::not_equal_to);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::LT, Less);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::LE, LessEqual);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::GT, Greater);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::GE, GreaterEqual);
#undef COMPARE_FOLDER

  return {};
}

//===----------------------------------------------------------------------===//
// SelectAndScatterOp
//===----------------------------------------------------------------------===//

namespace {
// Infer the return-type of SelectAndScatterOp.
TensorType inferSelectAndScatterOpReturnType(
    TensorType operandType, const ArrayRef<WindowDimension> window) {
  if (!operandType.hasRank())
    return UnrankedTensorType::get(operandType.getElementType());

  return RankedTensorType::get(
      inferWindowOutputShape(operandType.getShape(), window),
      operandType.getElementType());
}
}  // namespace

//  We intend to verify the following properties:
//   P1. Check if the select function has a proper shape of (T,T) -> PRED, where
//        T is a 0-D tensor with element-type same as 'operand' element-type.
//   P2. Verify scatter-computation type.
//   P3. size-of(window_dimension) == rank-of(input),
//         where input is an element of 'inputs'.
//   P4. Verify and collect the window attributes.
//   P5. Verify the return type matches the operand-type.
//   P6. Check if the result type of window operation matches the source type.
LogicalResult SelectAndScatterOp::verify() {
  auto operandType = operand().getType().cast<TensorType>();
  auto initValueType = init_value().getType().cast<TensorType>();
  auto sourceType = source().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();

  // P1.
  Block& selectBlock = select().front();

  if (selectBlock.getArguments().size() != 2)
    return emitOpError()
           << "expects the select-region to take 2 parameters, but takes "
           << selectBlock.getArguments().size();

  Type expectedSelectArgType =
      RankedTensorType::get({}, operandType.getElementType());
  for (const auto& selectArgIt : llvm::enumerate(selectBlock.getArguments()))
    if (!compatibleShapeAndElementType(expectedSelectArgType,
                                       selectArgIt.value().getType(),
                                       /*ignoreFpPrecision=*/true))
      return emitOpError()
             << "expects the type of select-region's parameter at index "
             << selectArgIt.index() << " to be " << expectedSelectArgType
             << ", but got " << selectArgIt.value().getType();

  auto selectResult = selectBlock.getTerminator()->getOperands();
  if (selectResult.size() != 1)
    return emitOpError()
           << "expects select-region to return single value, but got: "
           << selectResult.size();

  auto selectResultType = selectResult[0].getType().dyn_cast<TensorType>();
  if (!selectResultType || !selectResultType.getElementType().isInteger(1) ||
      (selectResultType.hasRank() &&
       selectResultType.cast<RankedTensorType>().getRank() != 0))
    return emitOpError() << "expects the return-type of select-region to be "
                            "tensor<i1>, but got: "
                         << selectResult[0].getType();

  // P2.
  Block& scatterBlock = scatter().front();
  SmallVector<TensorType> accumulatorSubshapes;
  if (failed(verifyReducerShape(
          this->getLoc(), scatterBlock,
          {RankedTensorType::get({}, sourceType.getElementType())},
          {initValueType},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/false, accumulatorSubshapes)))
    return failure();

  // P3.
  SmallVector<int64_t> windowDims =
      convertDenseIntAttr(this->window_dimensions());
  if (operandType.hasRank()) {
    if (operandType.getRank() != windowDims.size())
      return emitOpError()
             << "expects window-dimensions size == operand rank, but got "
                "window-dimensions size: "
             << windowDims.size() << " and operand-type: " << operandType
             << " with rank = " << operandType.getRank() << ".";
  }

  // P4.
  auto paddingOrErr = convertNx2Attribute(this->padding(), getLoc());
  if (failed(paddingOrErr)) return failure();
  SmallVector<std::pair<int64_t, int64_t>> padding = *paddingOrErr;

  auto windowOrErr = verifyWindowAttributesAndInferWindowDimensions(
      windowDims, convertDenseIntAttr(window_strides()), padding,
      /*lhs_dilation=*/{}, /*rhs_dilation=*/{}, getLoc());
  if (failed(windowOrErr)) return failure();

  // P5.
  if (!compatibleShapeAndElementType(operandType, resultType))
    return emitOpError()
           << "expects the return-type to match the operand-type, but got "
           << resultType << " and " << operandType << " resp.";

  // P6.
  auto windowResultType =
      inferSelectAndScatterOpReturnType(operandType, *windowOrErr);

  if (!compatibleShapeAndElementType(windowResultType, sourceType,
                                     /*ignoreFpPrecision=*/true))
    return emitOpError() << "expects source-type to be " << windowResultType
                         << ", but got" << sourceType;

  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

/*
 * We intend to verify the following properties:
 * P1. The 'update_window_dims' must be valid indices of 'updates' tensor.
 * P2. The 'inserted_window_dims' must be valid indices of 'operand' tensor.
 * P3. Check if the rank-of('operand') == size-of('update_window_dims') +
 *     size-of('inserted_window_dims')
 * P4. size-of('scatter_dims_to_operand_dims') =
 *         'scatter_indices'['index_vector_dim'] &
 *     'scatter_dims_to_operand_dims' must be valid indices of 'operand' tensor.
 */
LogicalResult ValidateScatterDimensionNumbers(
    ShapedType operandType, ArrayRef<int64_t> scatterIndicesShape,
    ShapedType updateType, bool operandTypeRanked,
    bool scatterIndicesTypeRanked, bool updatesTypeRanked,
    ScatterDimensionNumbersAttr dimNumbers, Location loc) {
  const auto hasDuplicates = [](SmallVector<int64_t>& nums) {
    if (!llvm::is_sorted(nums)) std::sort(nums.begin(), nums.end());
    auto last = std::unique(nums.begin(), nums.end());
    return last != nums.end();
  };

  // P1.
  auto updateWindowDims = to_vector(dimNumbers.getUpdateWindowDims());
  if (!llvm::is_sorted(updateWindowDims))
    return mlir::emitError(loc)
           << "Expects update_window_dims to be sorted; got: ["
           << updateWindowDims << "].";

  if (hasDuplicates(updateWindowDims))
    return mlir::emitError(loc)
           << "Expects update_window_dims to not repeat; got: ["
           << updateWindowDims << "].";

  if (updatesTypeRanked) {
    for (int64_t windowDim : updateWindowDims) {
      if (windowDim < 0 || windowDim >= updateType.getRank()) {
        return mlir::emitError(loc)
               << "Expects each element of update_window_dims to be in range "
                  "[0, "
                  "rank-of('updates') i.e. [0, "
               << updateType.getRank() << "). got: " << windowDim << ".";
      }
    }
  }

  // P2.
  auto insertedWindowDims = to_vector(dimNumbers.getInsertedWindowDims());
  if (!llvm::is_sorted(insertedWindowDims))
    return mlir::emitError(loc)
           << "Expects inserted_window_dims to be sorted; got: ["
           << insertedWindowDims << "].";

  if (hasDuplicates(insertedWindowDims))
    return mlir::emitError(loc)
           << "Expects inserted_window_dims to not repeat; got: ["
           << insertedWindowDims << "].";

  if (operandTypeRanked) {
    for (int64_t insertedDim : insertedWindowDims) {
      if (insertedDim < 0 || insertedDim >= operandType.getRank()) {
        return mlir::emitError(loc)
               << "Expects each element of inserted_window_dims to be in range "
                  "[0, rank-of('operand') i.e. [0, "
               << operandType.getRank() << "). got: " << insertedDim << ".";
      }
    }
  }

  // P3.
  if (operandTypeRanked) {
    auto windowSize = updateWindowDims.size() + insertedWindowDims.size();
    if (operandType.getRank() != windowSize)
      return mlir::emitError(loc)
             << "Expects rank-of operand to match "
                "size-of('update_window_dims')  + "
                "size-of('inserted_window_dims') i.e. "
             << windowSize << " but got " << operandType.getRank() << ".";
  }

  // P4.
  auto scatterDimsToOperandDims =
      to_vector(dimNumbers.getScatterDimsToOperandDims());
  auto indexVectorDim = dimNumbers.getIndexVectorDim();
  if (scatterIndicesTypeRanked) {
    if (!isDynamicDimSize(scatterIndicesShape[indexVectorDim]) &&
        scatterDimsToOperandDims.size() !=
            scatterIndicesShape[dimNumbers.getIndexVectorDim()])
      return mlir::emitError(loc)
             << "Scatter op has " << scatterDimsToOperandDims.size()
             << " elements in scatter_dims_to_operand_dims and the bound of "
                "dimension index_vector_dim="
             << dimNumbers.getIndexVectorDim() << " of scatter_indices is "
             << scatterIndicesShape[dimNumbers.getIndexVectorDim()]
             << ". These two numbers must be equal.";
  }

  if (operandTypeRanked) {
    for (int i = 0; i < scatterDimsToOperandDims.size(); ++i) {
      int64_t scatterDimToOperandDim = scatterDimsToOperandDims[i];
      if (scatterDimToOperandDim < 0 ||
          scatterDimToOperandDim >= operandType.getRank())
        return mlir::emitError(loc)
               << "Invalid scatter_dims_to_operand_dims mapping; domain is [0, "
               << operandType.getRank() << "), got: " << i << "->"
               << scatterDimToOperandDim << ".";
    }
  }

  if (hasDuplicates(scatterDimsToOperandDims))
    return mlir::emitError(loc)
           << "Expects scatter_dims_to_operand_dims to not repeat; got: ["
           << scatterDimsToOperandDims << "].";

  return success();
}
/*
 * We intend to verify the following properties:
 *  P0. scatter_indices argument must be an integral tensor. Enforced by ODS.
 *  P1. Scatter index leaf dimension must be within [0, rank(scatter_indices)"
 *      " + 1).
 *  P2. Verify reducer shape.
 *  P3. rank-of('updates[i]') == size-of('update_window_dims') +
 *      rank-of('scatter_indices') - 1, where 'scatter_indices' is expanded by a
 *      trailing 1 dimension if 'index_vector_dim' == rank-of('scatter_indices')
 *      for all values of `i`.
 *  P4. Validate the scatter-dimensions-numbers.
 *  P5. Valide the bounds of each of the 'updates' w.r.t the operands.
 *  P6. Validate the bounds of each of the 'updates' w.r.t the
 * 'scatter_indices'.
 *  P7. Check return types.
 */
LogicalResult ScatterOp::verify() {
  // Get the first operand and update, since variadic Scatter is not yet
  // implemented
  auto num_operands = operands().size();
  auto scatterIndicesType = scatter_indices().getType().dyn_cast<TensorType>();

  SmallVector<TensorType, 1> operandTypes =
      llvm::to_vector(llvm::map_range(operands().getTypes(), [](Type type) {
        return type.cast<TensorType>();
      }));
  SmallVector<TensorType, 1> updatesTypes = llvm::to_vector(llvm::map_range(
      updates().getTypes(), [](Type type) { return type.cast<TensorType>(); }));
  bool allOperandTypesRanked =
      llvm::all_of(operands().getTypes(),
                   [](Type type) { return type.isa<RankedTensorType>(); });
  bool scatterIndicesTypeRanked = scatterIndicesType.isa<RankedTensorType>();

  // P1.
  int64_t indexVectorDim = scatter_dimension_numbers().getIndexVectorDim();
  if (scatterIndicesTypeRanked) {
    if (indexVectorDim > scatterIndicesType.getRank() || indexVectorDim < 0)
      return emitOpError()
             << "expects scatter index leaf dimension to be within [0, "
                "rank(scatter_indices) + 1."
                " rank(scatter_indices) is "
             << scatterIndicesType.getRank()
             << " and scatter index leaf dimension is " << indexVectorDim
             << ".";
  }

  // P2.
  Block& block = update_computation().front();
  SmallVector<TensorType> accumulator_subshapes;
  SmallVector<TensorType> inputTypes, initValueTypes;
  for (int i = 0; i < num_operands; i++) {
    inputTypes.push_back(operandTypes[i]);
    initValueTypes.push_back(
        RankedTensorType::get({}, updatesTypes[i].getElementType()));
  }
  if (failed(verifyReducerShape(
          this->getLoc(), block, inputTypes, initValueTypes, num_operands,
          /*allowedDimensions=*/{},
          /*allInputsUnranked=*/!allOperandTypesRanked, accumulator_subshapes)))
    return failure();

  // P3.
  auto updateWindowDims = scatter_dimension_numbers().getUpdateWindowDims();
  SmallVector<int64_t> expandedScatterIndicesShape;
  if (scatterIndicesTypeRanked) {
    expandedScatterIndicesShape =
        llvm::to_vector(scatterIndicesType.getShape());
    if (expandedScatterIndicesShape.size() == indexVectorDim)
      expandedScatterIndicesShape.push_back(1);
  }

  for (int i = 0; i < num_operands; i++) {
    if (scatterIndicesTypeRanked && updatesTypes[i].isa<RankedTensorType>()) {
      int64_t expected_updates_rank =
          expandedScatterIndicesShape.size() - 1 + updateWindowDims.size();
      if (updatesTypes[i].getRank() != expected_updates_rank)
        return emitOpError()
               << "expects updates tensor must be of rank "
               << expected_updates_rank
               << " ( == rank-of('scatter_indices') - 1 + "
                  "size-of('update_window_dims'), where 'scatter_indices' is "
                  "expanded by a trailing 1 dimension if 'index_vector_dim' == "
                  "rank-of('scatter_indices')), but got "
               << updatesTypes[i].getRank() << ".";
    }
  }

  // P4.
  for (int i = 0; i < num_operands; i++) {
    if (failed(ValidateScatterDimensionNumbers(
            operandTypes[i], expandedScatterIndicesShape, updatesTypes[i],
            operandTypes[i].isa<RankedTensorType>(), scatterIndicesTypeRanked,
            updatesTypes[i].isa<RankedTensorType>(),
            scatter_dimension_numbers(), getLoc())))
      return failure();
  }

  // P5.
  for (int i = 0; i < num_operands; i++) {
    if (updatesTypes[i].isa<RankedTensorType>()) {
      auto updatesShape = updatesTypes[i].getShape();
      if (operandTypes[i].isa<RankedTensorType>()) {
        auto operandShape = operandTypes[i].getShape();
        auto inserted_window_dims =
            scatter_dimension_numbers().getInsertedWindowDims();

        int64_t inserted_dims_seen = 0;
        SmallVector<int64_t> max_update_slice_sizes;
        const auto dimensionsSize = operandTypes[i].getRank();
        max_update_slice_sizes.reserve(dimensionsSize);
        for (int i = 0; i < dimensionsSize; ++i) {
          if (inserted_dims_seen < inserted_window_dims.size() &&
              inserted_window_dims[inserted_dims_seen] == i) {
            ++inserted_dims_seen;
          } else {
            max_update_slice_sizes.push_back(operandShape[i]);
          }
        }

        for (int i = 0; i < updateWindowDims.size(); ++i) {
          auto update_window_dim = updateWindowDims[i];

          if (isDynamicDimSize(updatesShape[update_window_dim]) ||
              isDynamicDimSize(max_update_slice_sizes[i]))
            continue;

          if (updatesShape[update_window_dim] > max_update_slice_sizes[i]) {
            return emitOpError()
                   << "expects bounds of the window dimensions of "
                      "updates to not exceed the "
                      "bounds of the corresponding dimensions of "
                      "operand. For dimension "
                   << update_window_dim << ", updates bound is "
                   << updatesShape[update_window_dim] << ", operand bound is "
                   << max_update_slice_sizes[i] << ".";
          }
        }
      }

      // P6.
      if (scatterIndicesTypeRanked) {
        int64_t scatter_dims_seen = 0;
        for (int64_t i = 0; i < updatesShape.size(); ++i) {
          bool is_update_window_dim = std::binary_search(
              updateWindowDims.begin(), updateWindowDims.end(), i);

          if (is_update_window_dim) continue;
          if (scatter_dims_seen == indexVectorDim) ++scatter_dims_seen;

          if (!isDynamicDimSize(updatesShape[i]) &&
              !isDynamicDimSize(
                  expandedScatterIndicesShape[scatter_dims_seen]) &&
              (updatesShape[i] !=
               expandedScatterIndicesShape[scatter_dims_seen])) {
            return emitOpError()
                   << "expects bounds of the scatter dimensions of "
                      "updates to be same as the "
                      "bounds of the corresponding dimensions of "
                      "scatter indices. For "
                      "scatter dimension "
                   << i << ", updates bound is " << updatesShape[i]
                   << " , scatter_indices "
                      "bound is "
                   << expandedScatterIndicesShape[scatter_dims_seen] << ".";
          }
          ++scatter_dims_seen;
        }
      }
    }
  }

  // P7.
  for (int i = 0; i < num_operands; i++) {
    if (!compatibleShapeAndElementType(operandTypes[i], getResult(i).getType()))
      return emitOpError()
             << "expects the return type to be same as the operand type: "
             << operandTypes[i] << ", but got " << getResult(i).getType()
             << ".";
  }

  return success();
}

llvm::SmallVector<Attribute, 4> evaluateMhloRegion(Region& region,
                                                   ArrayRef<Attribute> inputs) {
  if (region.getNumArguments() != inputs.size()) return {};

  llvm::DenseMap<Value, Attribute> values;
  values.reserve(region.getNumArguments());
  for (auto it : llvm::zip(region.getArguments(), inputs)) {
    values.try_emplace(std::get<0>(it), std::get<1>(it));
  }

  for (auto& op : region.getOps()) {
    llvm::SmallVector<Attribute, 4> inputs;
    for (auto& operand : op.getOpOperands()) {
      inputs.push_back(values.lookup(operand.get()));
    }
    if (isa<ReturnOp>(op)) return inputs;

    llvm::SmallVector<OpFoldResult, 4> results;
    if (failed(op.fold(inputs, results))) return {};
    for (auto it : llvm::zip(op.getResults(), results)) {
      if (!std::get<1>(it).is<Attribute>()) return {};
      values.insert({std::get<0>(it), std::get<1>(it).get<Attribute>()});
    }
  }
  return {};
}

LogicalResult ScatterOp::fold(
    ArrayRef<Attribute> args,
    llvm::SmallVectorImpl<OpFoldResult>& foldResults) {
  // Variadic Scatter not yet implemented
  if (operands().size() != 1 || updates().size() != 1) return failure();
  auto index = args[1].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!index) return failure();

  auto baseType = operands().getTypes()[0].dyn_cast<RankedTensorType>();
  auto updateType = updates().getTypes()[0].dyn_cast<RankedTensorType>();
  auto indexType = index.getType().cast<RankedTensorType>();
  if (!baseType || !indexType || !updateType) return failure();

  // TODO(b/228310289): Work around canonicalization crash for complex types.
  // Remove after upstream MLIR has been fixed.
  if (baseType.getElementType().isa<ComplexType>()) return failure();

  // Catch a trivial full replacement of base with update, this does not require
  // these to be constant: just that we know the type.
  if (updateType == baseType && updateType.hasStaticShape() &&
      baseType.hasStaticShape() && index.isSplat() &&
      index.getSplatValue<uint32_t>() == 0 &&
      llvm::hasSingleElement(update_computation().front())) {
    foldResults.push_back(updates()[0]);
    return success();
  }
  auto base = args[0].dyn_cast_or_null<DenseElementsAttr>();
  auto update = args[2].dyn_cast_or_null<DenseElementsAttr>();
  if (!base || !update) return failure();

  // Prevent splat to be expanded if too large.
  if (base.isSplat() && base.getNumElements() > kFoldExpandSplatEltLimit)
    return failure();

  // Add the virtual trailing dimension of size 1 if indexVectorDim equals to
  // indexType.rank.
  const int64_t indexVectorDim =
      scatter_dimension_numbers().getIndexVectorDim();
  if (indexVectorDim == indexType.getRank()) {
    auto indexShape = indexType.getShape().vec();
    indexShape.push_back(1);
    indexType = RankedTensorType::get(indexShape, indexType.getElementType());
    index = reshape(index, indexType).cast<DenseIntElementsAttr>();
  }

  // Increment the multi-dimensional index vector based on the limits for each
  // dimension specified by shape and returns false if the index rolled around
  // with true otherwise.
  auto nextIndex = [](llvm::SmallVector<uint64_t, 8>& index,
                      llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (index[i] < shape[i]) return true;
      index[i] = 0;
    }
    return false;
  };

  // Iterate over all elements of the update tensor, then find the corresponding
  // value in the indices tensor to determine which location we have to update
  // in the base/result tensor.
  llvm::SmallVector<Attribute, 8> results(base.getValues<Attribute>());
  llvm::SmallVector<uint64_t, 8> updateIndex(updateType.getRank(), 0);
  llvm::SmallVector<uint64_t, 8> indexIndex;
  indexIndex.reserve(indexType.getRank());
  llvm::SmallVector<uint64_t, 8> baseIndex;
  baseIndex.reserve(baseType.getRank());
  do {
    // Compute the index for the slice of the indices tensor for this update
    // value.
    indexIndex.clear();
    if (indexVectorDim == 0) indexIndex.push_back(0);
    for (int64_t i = 0; i < updateIndex.size(); ++i) {
      if (llvm::count(scatter_dimension_numbers().getUpdateWindowDims(), i) ==
          0)
        indexIndex.push_back(updateIndex[i]);
      if (indexIndex.size() == indexVectorDim) indexIndex.push_back(0);
    }

    // Compute the index for the given update value in the base tensor.
    baseIndex.assign(baseType.getRank(), 0);
    uint64_t indexCount = indexType.getShape()[indexVectorDim];
    for (uint64_t i = 0; i < indexCount; ++i) {
      uint64_t operandDim =
          scatter_dimension_numbers().getScatterDimsToOperandDims()[i];
      indexIndex[indexVectorDim] = i;
      baseIndex[operandDim] +=
          index.getValues<APInt>()[indexIndex].getSExtValue();
    }
    uint64_t updateWindowDimIndex = 0;
    auto insertedWindowDims =
        scatter_dimension_numbers().getInsertedWindowDims();
    auto updateWindowDims = scatter_dimension_numbers().getUpdateWindowDims();
    for (uint64_t i = 0; i < baseIndex.size(); ++i) {
      if (llvm::count(insertedWindowDims, i)) continue;
      baseIndex[i] += updateIndex[updateWindowDims[updateWindowDimIndex]];
      updateWindowDimIndex++;
    }

    // Compute the linear index for the index into the base tensor.
    int64_t linearBaseIndex = 0;
    int64_t linearBaseIndexMultiplyer = 1;
    for (int64_t i = baseIndex.size() - 1; i >= 0; --i) {
      // Out of bound index have backend specific behaviour so avoid folding it.
      if (baseIndex[i] < 0 || baseIndex[i] >= baseType.getShape()[i])
        return failure();
      linearBaseIndex += baseIndex[i] * linearBaseIndexMultiplyer;
      linearBaseIndexMultiplyer *= baseType.getShape()[i];
    }

    // Evaluate update computation and update the value with the newly computed
    // attribute in the base tensor.
    auto lhs = DenseElementsAttr::get(
        RankedTensorType::get({}, baseType.getElementType()),
        results[linearBaseIndex]);
    auto rhs = DenseElementsAttr::get(
        RankedTensorType::get({}, baseType.getElementType()),
        update.getValues<Attribute>()[updateIndex]);
    auto newValue = evaluateMhloRegion(update_computation(), {lhs, rhs});
    if (newValue.size() != 1 || !newValue[0]) return failure();
    results[linearBaseIndex] =
        newValue[0].cast<DenseElementsAttr>().getValues<Attribute>()[0];
  } while (nextIndex(updateIndex, updateType.getShape()));

  foldResults.push_back(DenseElementsAttr::get(baseType, results));
  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::verify() {
  if (getNumOperands() != cond().front().getNumArguments())
    return emitOpError() << "mismatch in operand count (" << getNumOperands()
                         << ") vs the condition block argument count ("
                         << cond().front().getNumArguments() << ")";
  if (getNumOperands() != body().front().getNumArguments())
    return emitOpError() << "mismatch in operand count (" << getNumOperands()
                         << ") vs the body block argument count ("
                         << body().front().getNumArguments() << ")";
  for (const auto& enumeratedOperands : llvm::enumerate(
           llvm::zip(getOperandTypes(), cond().front().getArgumentTypes(),
                     body().front().getArgumentTypes()))) {
    int argCount = enumeratedOperands.index();
    const auto& operands = enumeratedOperands.value();
    Type operandType = std::get<0>(operands);
    Type condType = std::get<1>(operands);
    Type bodyType = std::get<2>(operands);
    if (operandType != condType)
      return emitOpError() << "type mismatch between operand #" << argCount
                           << " and the matching condition block argument: "
                           << operandType << " vs " << condType;
    if (operandType != bodyType)
      return emitOpError() << "type mismatch between operand #" << argCount
                           << " and the matching body block argument: "
                           << operandType << " vs " << bodyType;
  }
  // Check the return type for the condition block.
  {
    auto condReturnOp = cast<ReturnOp>(cond().front().back());
    if (condReturnOp->getNumOperands() != 1)
      return condReturnOp.emitOpError()
             << "expects a single operand for while condition body return, got "
             << condReturnOp->getNumOperands();
    auto operandType =
        condReturnOp->getOperand(0).getType().dyn_cast<RankedTensorType>();
    if (!operandType || operandType.getRank() != 0 ||
        !operandType.getElementType().isInteger(1))
      return condReturnOp.emitOpError()
             << "expects a zero-ranked tensor of i1, got "
             << condReturnOp->getOperand(0).getType();
  }
  // Check the return type for the body block.
  {
    auto bodyReturnOp = cast<ReturnOp>(body().front().back());
    if (bodyReturnOp->getNumOperands() != getNumOperands())
      return bodyReturnOp.emitOpError()
             << "expects body to return a many value as the operands ("
             << getNumOperands() << "), got " << bodyReturnOp->getNumOperands();
    for (const auto& enumeratedOperandTypes : llvm::enumerate(
             llvm::zip(bodyReturnOp->getOperandTypes(), getOperandTypes()))) {
      Type operandType = std::get<0>(enumeratedOperandTypes.value());
      Type returnType = std::get<1>(enumeratedOperandTypes.value());
      if (operandType != returnType)
        return bodyReturnOp.emitOpError()
               << "type mismatch between operand #"
               << enumeratedOperandTypes.index()
               << " and the enclosing WhileOp returned value: " << operandType
               << " vs " << returnType;
    }
  }
  return success();
}

/// Print a `while` op.
///
/// op ::= `mhlo.while` `(` assignment-list `)` `:` types attribute-dict
///         `cond` region
///         `do` region
/// assignment-list ::= assignment | assignment `,` assignment-list
/// assignment ::= ssa-value `=` ssa-value
void WhileOp::print(OpAsmPrinter& p) {
  p << '(';
  llvm::interleaveComma(llvm::zip(getBody()->getArguments(), getOperands()), p,
                        [&](auto zip) {
                          p.printOperand(std::get<0>(zip));
                          p << " = ";
                          p.printOperand(std::get<1>(zip));
                        });
  p << ")";
  if (getNumOperands()) {
    p << " : ";
    llvm::interleaveComma(getOperandTypes(), p);
  }
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p.printNewline();
  p << " cond ";
  p.printRegion(getRegion(0), /*printEntryBlockArgs=*/false);
  p << " do ";
  p.printRegion(getRegion(1), /*printEntryBlockArgs=*/false);
}

ParseResult WhileOp::parse(OpAsmParser& parser, OperationState& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  // Parse the operands of the while: these are of the form:
  //   %iter_arg = %init_val
  // where %iter_arg is the name of the block argument in the cond/body blocks
  // and %init_val is the actual operand.
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<OpAsmParser::UnresolvedOperand> iterArgs;
  if (parser.parseLParen()) return failure();
  do {
    if (succeeded(parser.parseOptionalRParen())) break;
    OpAsmParser::UnresolvedOperand operand, iterArg;
    if (parser.parseOperand(iterArg) || parser.parseEqual() ||
        parser.parseOperand(operand))
      return failure();
    iterArgs.push_back(iterArg);
    operands.push_back(operand);
    if (succeeded(parser.parseOptionalRParen())) break;
    if (failed(parser.parseComma())) return failure();
  } while (true);
  if (!operands.empty()) {
    if (parser.parseColon() || parser.parseTypeList(result.types))
      return failure();
  }

  SmallVector<OpAsmParser::Argument> args;
  createArgs(iterArgs, result.types, args);
  if (parser.resolveOperands(operands, result.types, loc, result.operands) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseKeyword("cond") ||
      parser.parseRegion(*result.addRegion(), args) ||
      parser.parseKeyword("do") ||
      parser.parseRegion(*result.addRegion(), args))
    return failure();
  return success();
}

static LogicalResult whileCanonicalization(WhileOp whileOp,
                                           PatternRewriter& rewriter) {
  // Turn loop invariant values into implicit capture.
  // Check if there is at least one value is forwarded from one iteration to the
  // next, or one of the yielded value is an implicit capture already. Otherwise
  // there is nothing to do here.
  Block* cond = whileOp.getBody(0);
  Block* body = whileOp.getBody(1);
  auto bodyReturnOp = cast<ReturnOp>(body->getTerminator());
  if (!llvm::any_of(llvm::zip(whileOp->getOperands(), body->getArguments(),
                              bodyReturnOp->getOperands()),
                    [&](auto zip) {
                      return (std::get<0>(zip) == std::get<2>(zip) ||
                              std::get<1>(zip) == std::get<2>(zip));
                    }))
    return rewriter.notifyMatchFailure(whileOp, "no loop invariant found");

  SmallVector<Value> newOperands, resultsToReplace;
  SmallVector<unsigned> invariantArgIdxs;
  for (const auto& enumeratedOperands : llvm::enumerate(llvm::zip(
           whileOp.getOperands(), cond->getArguments(), body->getArguments(),
           bodyReturnOp->getOperands(), whileOp->getResults()))) {
    const auto& operands = enumeratedOperands.value();
    Value whileOperand = std::get<0>(operands);
    BlockArgument condBlockArg = std::get<1>(operands);
    BlockArgument bodyBlockArg = std::get<2>(operands);
    Value bodyReturnOperand = std::get<3>(operands);
    Value whileResult = std::get<4>(operands);

    bool forwarded = (whileOperand == bodyReturnOperand ||
                      bodyBlockArg == bodyReturnOperand);
    if (forwarded) {
      invariantArgIdxs.push_back(enumeratedOperands.index());
      condBlockArg.replaceAllUsesWith(whileOperand);
      bodyBlockArg.replaceAllUsesWith(whileOperand);
      whileResult.replaceAllUsesWith(whileOperand);
      continue;
    }
    newOperands.push_back(whileOperand);
    resultsToReplace.push_back(whileResult);
  }
  cond->eraseArguments(invariantArgIdxs);
  body->eraseArguments(invariantArgIdxs);
  for (int idx : llvm::reverse(invariantArgIdxs))
    bodyReturnOp->eraseOperand(idx);

  WhileOp newWhileOp = rewriter.create<WhileOp>(
      whileOp.getLoc(), bodyReturnOp->getOperandTypes(), newOperands);
  newWhileOp.getBodyRegion(0).takeBody(whileOp.getBodyRegion(0));
  newWhileOp.getBodyRegion(1).takeBody(whileOp.getBodyRegion(1));
  for (auto results : llvm::zip(resultsToReplace, newWhileOp->getResults()))
    std::get<0>(results).replaceAllUsesWith(std::get<1>(results));
  rewriter.eraseOp(whileOp);
  return success();
}

void WhileOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add(&whileCanonicalization);
}

using mlir::hlo::parseWindowAttributes;
using mlir::hlo::printWindowAttributes;

}  // namespace mhlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.cc.inc"

namespace mlir {
namespace mhlo {

//===----------------------------------------------------------------------===//
// mhlo Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct HLOInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       BlockAndValueMapping& valueMapping) const final {
    return true;
  }
  // Operations in mhlo dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation*, Region*, bool,
                       BlockAndValueMapping&) const final {
    return true;
  }
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// mhlo Dialect Constructor
//===----------------------------------------------------------------------===//

MhloDialect::MhloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MhloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.cc.inc"
      >();
  addInterfaces<HLOInlinerInterface>();
  addTypes<TokenType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.cc.inc"
      >();
  context->loadDialect<tensor::TensorDialect>();
}

Type MhloDialect::parseType(DialectAsmParser& parser) const {
  StringRef dataType;
  if (parser.parseKeyword(&dataType)) return Type();

  if (dataType == "token") return TokenType::get(getContext());
  parser.emitError(parser.getNameLoc()) << "unknown mhlo type: " << dataType;
  return nullptr;
}

void MhloDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<TokenType>()) {
    os << "token";
    return;
  }
  os << "<unknown mhlo type>";
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute MhloDialect::parseAttribute(DialectAsmParser& parser,
                                      Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag))) return Attribute();
  {
    Attribute attr;
    auto parseResult = generatedAttributeParser(parser, attrTag, type, attr);
    if (parseResult.hasValue()) return attr;
  }
  parser.emitError(parser.getNameLoc(), "unknown mhlo attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void MhloDialect::printAttribute(Attribute attr, DialectAsmPrinter& os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  (void)result;
  assert(succeeded(result));
}

/// Helpers for attributes parsing.

static ParseResult parseDims(AsmParser& parser, SmallVector<int64_t>& dims) {
  dims.clear();
  if (parser.parseLSquare()) return failure();
  while (failed(parser.parseOptionalRSquare())) {
    dims.emplace_back();
    if (parser.parseInteger(dims.back())) return failure();
    (void)parser.parseOptionalComma();
  }
  return success();
}

static ParseResult parseDimsWithMinimumElements(AsmParser& parser,
                                                SmallVector<int64_t>& dims,
                                                int minElements) {
  if (failed(parseDims(parser, dims))) return failure();
  if (dims.size() < minElements)
    return parser.emitError(parser.getCurrentLocation())
           << "expected at least " << minElements << " element(s), found "
           << dims.size();
  return success();
}

/// Parse a custom attribute that resembles a struct of the form
/// <
///   foo = something_parsed_by_custom_parser,
///   bar = something_parsed_by_different_custom_parser,
///   baz something_parsed_by_another_custom_parser
/// >
/// The optional argument `parse_equal` array can be used to denote if
/// '=' follows the keyword (see baz in the example above) for a field. If
/// not provided, all fields must be followed by a '='.
static ParseResult parseStruct(
    AsmParser& parser, ArrayRef<StringRef> keywords,
    ArrayRef<llvm::function_ref<ParseResult()>> parseFuncs,
    ArrayRef<bool> parseEqual = {}) {
  assert(keywords.size() == parseFuncs.size());
  assert(parseEqual.empty() || parseEqual.size() == keywords.size());
  SmallVector<bool> seen(keywords.size(), false);
  while (failed(parser.parseOptionalGreater())) {
    bool foundOne = false;
    for (const auto& it : llvm::enumerate(keywords)) {
      size_t index = it.index();
      StringRef keyword = it.value();
      if (succeeded(parser.parseOptionalKeyword(keyword))) {
        if (seen[index]) {
          return parser.emitError(parser.getCurrentLocation())
                 << "duplicated `" << keyword << "` entry";
        }
        if (parseEqual.empty() || parseEqual[index]) {
          if (failed(parser.parseEqual())) return failure();
        }
        if (failed(parseFuncs[index]())) return failure();
        if (failed(parser.parseOptionalComma())) return parser.parseGreater();
        seen[index] = true;
        foundOne = true;
      }
    }
    if (!foundOne) {
      auto parseError = parser.emitError(parser.getCurrentLocation())
                        << "expected one of: ";
      llvm::interleaveComma(keywords, parseError, [&](StringRef kw) {
        parseError << '`' << kw << '`';
      });
      return parseError;
    }
  }
  return success();
}

// Helpers to print an optional array or integer field, to simplify writing
// attribute printers.
template <typename T>
static void printField(AsmPrinter& printer, StringRef name, T field,
                       StringRef& separator) {
  if (field != 0) {
    printer << separator << name << " = " << field;
    separator = ", ";
  }
}
template <typename T>
static void printField(AsmPrinter& printer, StringRef name, ArrayRef<T> field,
                       StringRef& separator) {
  if (!field.empty()) {
    printer << separator << name << " = [";
    llvm::interleaveComma(field, printer);
    printer << "]";
    separator = ", ";
  }
}
template <typename... Ts>
static void printStruct(AsmPrinter& printer, StringRef name,
                        Ts... printFields) {
  printer << "<";
  StringRef separator = "";
  // Fold expression to print each entry in the parameter pack.
  // TODO(mhlo-team): this can be simplified when TF moves to C++17.
  using unused = int[];
  (void)unused{0, (printField(printer, std::get<0>(printFields),
                              std::get<1>(printFields), separator),
                   0)...};
  printer << ">";
}

// Custom printer and parser for ScatterDimensionNumbersAttr.
void ScatterDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(printer, "scatter",
              std::make_pair("update_window_dims", getUpdateWindowDims()),
              std::make_pair("inserted_window_dims", getInsertedWindowDims()),
              std::make_pair("scatter_dims_to_operand_dims",
                             getScatterDimsToOperandDims()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}
Attribute ScatterDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  SmallVector<int64_t> updateWindowDims;
  SmallVector<int64_t> insertedWindowDims;
  SmallVector<int64_t> scatterDimsToOperandDims;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"update_window_dims", "inserted_window_dims",
           "scatter_dims_to_operand_dims", "index_vector_dim"},
          {[&]() { return parseDims(parser, updateWindowDims); },
           [&]() { return parseDims(parser, insertedWindowDims); },
           [&]() { return parseDims(parser, scatterDimsToOperandDims); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing scatter dimension numbers attribute";
    return {};
  }

  return ScatterDimensionNumbersAttr::get(
      parser.getContext(), updateWindowDims, insertedWindowDims,
      scatterDimsToOperandDims, indexVectorDim);
}

// Custom printer and parser for GatherDimensionNumbersAttr.
void GatherDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(printer, "gather", std::make_pair("offset_dims", getOffsetDims()),
              std::make_pair("collapsed_slice_dims", getCollapsedSliceDims()),
              std::make_pair("start_index_map", getStartIndexMap()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}

Attribute GatherDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};

  SmallVector<int64_t> offsetDims;
  SmallVector<int64_t> collapsedSliceDims;
  SmallVector<int64_t> startIndexMap;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"offset_dims", "collapsed_slice_dims", "start_index_map",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, offsetDims); },
           [&]() { return parseDims(parser, collapsedSliceDims); },
           [&]() { return parseDims(parser, startIndexMap); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing gather dimension numbers attribute";
    return {};
  }

  return GatherDimensionNumbersAttr::get(parser.getContext(), offsetDims,
                                         collapsedSliceDims, startIndexMap,
                                         indexVectorDim);
}

// Custom printer and parser for DotDimensionNumbersAttr.
void DotDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(
      printer, "dot",
      std::make_pair("lhs_batching_dimensions", getLhsBatchingDimensions()),
      std::make_pair("rhs_batching_dimensions", getRhsBatchingDimensions()),
      std::make_pair("lhs_contracting_dimensions",
                     getLhsContractingDimensions()),
      std::make_pair("rhs_contracting_dimensions",
                     getRhsContractingDimensions()));
}

Attribute DotDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};

  SmallVector<int64_t> lhsBatchingDimensions;
  SmallVector<int64_t> rhsBatchingDimensions;
  SmallVector<int64_t> lhsContractingDimensions;
  SmallVector<int64_t> rhsContractingDimensions;

  if (failed(parseStruct(
          parser,
          {"lhs_batching_dimensions", "rhs_batching_dimensions",
           "lhs_contracting_dimensions", "rhs_contracting_dimensions"},
          {[&]() { return parseDims(parser, lhsBatchingDimensions); },
           [&]() { return parseDims(parser, rhsBatchingDimensions); },
           [&]() { return parseDims(parser, lhsContractingDimensions); },
           [&]() { return parseDims(parser, rhsContractingDimensions); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return {};
  }
  return DotDimensionNumbersAttr::get(
      parser.getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions);
}

namespace {
enum NonSpatialDim : int64_t {
  IOBatch = -1,    // Input or output batch dimension
  IOFeature = -2,  // Input or output feature dimension
  KIFeature = -3,  // Kernel input feature dimension
  KOFeature = -4,  // Kernel output feature dimensions.
};

struct DenseMapInfoNonSpatialDim {
  static inline NonSpatialDim getEmptyKey() {
    return NonSpatialDim(DenseMapInfo<int64_t>::getEmptyKey());
  }

  static inline NonSpatialDim getTombstoneKey() {
    return NonSpatialDim(DenseMapInfo<int64_t>::getTombstoneKey());
  }

  static unsigned getHashValue(const NonSpatialDim& key) {
    return DenseMapInfo<int64_t>::getHashValue(key);
  }

  static bool isEqual(const NonSpatialDim& lhs, const NonSpatialDim& rhs) {
    return lhs == rhs;
  }
};

char nonSpatialDimToString(NonSpatialDim dim) {
  switch (dim) {
    case IOBatch:
      return 'b';
    case IOFeature:
      return 'f';
    case KIFeature:
      return 'i';
    case KOFeature:
      return 'o';
  }
  llvm_unreachable("Unknown NonSpatialDim");
}
}  // namespace

// Custom printer and parser for convolution attribute.
void printConvolutionDimensions(AsmPrinter& p, ConvDimensionNumbersAttr dnums) {
  // TODO(b/202040055): we should check the attribute invariant and print the
  // "raw" form if they are violated, otherwise we'll crash here.
  constexpr int64_t kUnknownDim = std::numeric_limits<int64_t>::min();
  auto printDim =
      [&](ArrayRef<int64_t> spatialDims,
          ArrayRef<std::pair<int64_t, NonSpatialDim>> nonSpatialDims) {
        int64_t numDims = 0;
        if (!spatialDims.empty()) {
          numDims =
              *std::max_element(spatialDims.begin(), spatialDims.end()) + 1;
        }
        for (const auto& dim : nonSpatialDims) {
          numDims = std::max(numDims, dim.first + 1);
        }

        llvm::SmallVector<int64_t> dims(numDims, kUnknownDim);
        // Fill each element of dims with a (< 0) NonSpatialDim enum or a (>=0)
        // spatial dimension index.
        for (const std::pair<int64_t, NonSpatialDim>& nonSpatialDim :
             nonSpatialDims) {
          dims[nonSpatialDim.first] = nonSpatialDim.second;
        }
        for (const auto& spatialDim : llvm::enumerate(spatialDims)) {
          dims[spatialDim.value()] = static_cast<int64_t>(spatialDim.index());
        }

        // Each dimension numbers will be printed as a comma separated list
        // surrounded by square brackets, e.g., [b, 0, 1, 2, f]
        p << '[';
        llvm::interleaveComma(dims, p, [&](int64_t dim) {
          if (dim == kUnknownDim) {
            p << "?";
          } else if (dim >= 0) {
            p << dim;
          } else {
            p << nonSpatialDimToString(static_cast<NonSpatialDim>(dim));
          }
        });
        p << ']';
      };

  printDim(dnums.getInputSpatialDimensions(),
           {{dnums.getInputBatchDimension(), IOBatch},
            {dnums.getInputFeatureDimension(), IOFeature}});
  p << "x";
  printDim(dnums.getKernelSpatialDimensions(),
           {{dnums.getKernelInputFeatureDimension(), KIFeature},
            {dnums.getKernelOutputFeatureDimension(), KOFeature}});
  p << "->";
  printDim(dnums.getOutputSpatialDimensions(),
           {{dnums.getOutputBatchDimension(), IOBatch},
            {dnums.getOutputFeatureDimension(), IOFeature}});
}

// Custom printer and parser for ConvDimensionNumbersAttr.
void ConvDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printer << "<";
  printConvolutionDimensions(printer, *this);
  printer << ">";
}

// If the attribute is written with `#mhlo.conv raw<`, we parse it as a struct
// instead of the compressed format. This enables writing tests covering
// impossible/invalid internal representation for the attribute.
static ParseResult parseConvolutionDimensionsRaw(
    AsmParser& parser, ConvDimensionNumbersAttr& dnums) {
  int64_t inputBatchDimension = 0;
  int64_t inputFeatureDimension = 0;
  SmallVector<int64_t> inputSpatialDimensions;
  int64_t kernelInputFeatureDimension = 0;
  int64_t kernelOutputFeatureDimension = 0;
  SmallVector<int64_t> kernelSpatialDimensions;
  int64_t outputBatchDimension = 0;
  int64_t outputFeatureDimension = 0;
  SmallVector<int64_t> outputSpatialDimensions;
  if (failed(parseStruct(
          parser,
          {"input_batch_dimension", "input_feature_dimension",
           "input_spatial_dimensions", "kernel_input_feature_dimension",
           "kernel_output_feature_dimension", "kernel_spatial_dimensions",
           "output_batch_dimension", "output_feature_dimension",
           "output_spatial_dimensions"},
          {
              [&]() { return parser.parseInteger(inputBatchDimension); },
              [&]() { return parser.parseInteger(inputFeatureDimension); },
              [&]() { return parseDims(parser, inputSpatialDimensions); },
              [&]() {
                return parser.parseInteger(kernelInputFeatureDimension);
              },
              [&]() {
                return parser.parseInteger(kernelOutputFeatureDimension);
              },
              [&]() { return parseDims(parser, kernelSpatialDimensions); },
              [&]() { return parser.parseInteger(outputBatchDimension); },
              [&]() { return parser.parseInteger(outputFeatureDimension); },
              [&]() { return parseDims(parser, outputSpatialDimensions); },
          }))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return failure();
  }
  dnums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), inputBatchDimension,
      inputFeatureDimension, inputSpatialDimensions,
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      kernelSpatialDimensions, outputBatchDimension, outputFeatureDimension,
      outputSpatialDimensions);
  return success();
}

ParseResult parseConvolutionDimensions(AsmParser& parser,
                                       ConvDimensionNumbersAttr& dnums) {
  // Parsing a single set of dim numbers gives the spatial dimensions as a
  // single ArrayRef<int64_t> and a list of non-spatial dimensions as
  // IntegerAttrs (indexed by the NonSpatialDim enum).
  using parse_dim_result_t =
      std::pair<llvm::SmallVector<int64_t>,
                llvm::SmallDenseMap<NonSpatialDim, int64_t, 4,
                                    DenseMapInfoNonSpatialDim>>;

  // Note that the allowed_non_spatial_dims is a set (as opposed to unordered
  // set) because its used to print a list of allowed non spatial dims in the
  // error messages, so making it a set keeps the error messages deterministic.
  auto parseDims =
      [&](std::set<NonSpatialDim, std::greater<>> allowedNonSpatialDims,
          parse_dim_result_t& parsedDims) -> ParseResult {
    auto& spatialDims = std::get<0>(parsedDims);
    auto& nonSpatialDims = std::get<1>(parsedDims);
    spatialDims.clear();
    nonSpatialDims.clear();

    // Parse the starting [
    if (parser.parseLSquare()) {
      return failure();
    }

    llvm::SmallDenseMap<int64_t, int64_t> spatialDimsMap;
    constexpr int64_t kInvalidDimension = -1;
    // Keep track of the maximum spatial dimension parsed as we expect to see
    // all the dimensions from 0 to maximum dimension parsed.
    int64_t maxParsedSpatialDim = kInvalidDimension;

    int64_t index = 0;
    do {
      int64_t spatialDim;
      auto dimLocation = parser.getCurrentLocation();
      OptionalParseResult parseResult = parser.parseOptionalInteger(spatialDim);
      if (parseResult.hasValue()) {
        if (parseResult.getValue().failed()) {
          return failure();
        }
        // We were successful in parsing an integer. Check if it is a valid
        // dimension (non-negative and no duplicate) and add its index to the
        // spatial dims map.
        if (spatialDim < 0)
          return parser.emitError(dimLocation)
                 << "Unexpected dimension " << spatialDim;
        if (!spatialDimsMap
                 .insert(std::pair<int64_t, int64_t>(spatialDim, index))
                 .second)
          return parser.emitError(dimLocation)
                 << "Duplicate entries for spatial dimension " << spatialDim;
        maxParsedSpatialDim = std::max(spatialDim, maxParsedSpatialDim);
      } else if (!parser.parseOptionalQuestion()) {
        // Do nothing other than increment `index` at the bottom of the loop;
        // '?' means "unknown dimension", and it's not represented in the
        // return value of this function.
      } else {
        // We did not parse an integer or question mark. We expect a keyword
        // token.
        StringRef keyword;
        if (parser.parseKeyword(&keyword)) {
          return failure();
        }
        if (keyword.size() != 1 || allowedNonSpatialDims.empty()) {
          return parser.emitError(dimLocation, "Unexpected keyword ")
                 << keyword;
        }
        // Check if the keyword matches one of the allowed non-spatial dims.
        // If so, add it to the non_spatial dims and remove it from the
        // allowed set so that it won't be allowed again.
        bool isAllowed = false;
        for (NonSpatialDim allowed : allowedNonSpatialDims) {
          if (keyword[0] == nonSpatialDimToString(allowed)) {
            nonSpatialDims.insert({allowed, index});
            allowedNonSpatialDims.erase(allowed);
            isAllowed = true;
            break;
          }
        }

        if (!isAllowed) {
          mlir::InFlightDiagnostic diag =
              parser.emitError(dimLocation, "Unexpected dimension ");
          diag << keyword << ", expecting ";
          llvm::interleaveComma(
              allowedNonSpatialDims, diag,
              [&](NonSpatialDim dim) { diag << nonSpatialDimToString(dim); });
          return diag;
        }
      }
      index++;
    } while (parser.parseOptionalComma().succeeded());

    // Make sure all expected non-spatial dimensions are parsed.
    if (!allowedNonSpatialDims.empty()) {
      mlir::InFlightDiagnostic diag =
          parser.emitError(parser.getCurrentLocation(), "Expected dimensions ");
      llvm::interleaveComma(
          allowedNonSpatialDims, diag,
          [&](NonSpatialDim dim) { diag << nonSpatialDimToString(dim); });
      diag << " not specified";
      return diag;
    }

    // parse ending ]
    if (parser.parseRSquare()) {
      return failure();
    }

    // Number of expected spatial dimensions is one more than the maximum parsed
    // spatial dimension. For example, if we parse [0, 3, 2, b, i, 1], then the
    // maximum parsed spatial dimension is 3 and the number of expected spatial
    // dimensions is 4.
    int64_t numSpatialDimensions = maxParsedSpatialDim + 1;
    spatialDims.resize(numSpatialDimensions);
    // Store spatial dimensions in a vector which maps spatial dim (vector
    // index) -> index in the tensor dimensions. For example, for parsed
    // dimension numbers [0, 3, 2, b, i, 1] the spatial dimension vector would
    // be [0, 5, 2, 1].
    //
    // Get all the unspecified spatial dimensions to throw a more descriptive
    // error later.
    llvm::SmallVector<int64_t> unspecifiedSpatialDims;
    constexpr int kPrintUnspecifiedDimsMax = 10;
    for (int dim = 0; dim < numSpatialDimensions; ++dim) {
      auto it = spatialDimsMap.find(dim);
      if (it == spatialDimsMap.end()) {
        // Have an upper bound on the number of unspecified dimensions to print
        // in the error message.
        if (unspecifiedSpatialDims.size() < kPrintUnspecifiedDimsMax)
          unspecifiedSpatialDims.push_back(dim);
        continue;
      }
      spatialDims[dim] = it->second;
    }

    // Verify that we got all spatial dimensions between 0 and maximum parsed
    // spatial dimension.
    if (!unspecifiedSpatialDims.empty()) {
      mlir::InFlightDiagnostic diag = parser.emitError(
          parser.getCurrentLocation(), "Expected spatial dimensions ");
      llvm::interleaveComma(unspecifiedSpatialDims, diag);
      diag << " not specified";
      return diag;
    }

    return success();
  };

  parse_dim_result_t parsedDims;
  if (parseDims({IOBatch, IOFeature}, parsedDims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> inputSpatialDimensions = parsedDims.first;
  int64_t inputBatchDimension = parsedDims.second[IOBatch];
  int64_t inputFeatureDimension = parsedDims.second[IOFeature];
  if (parser.parseKeyword("x")) return failure();
  if (parseDims({KIFeature, KOFeature}, parsedDims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> kernelSpatialDimensions = parsedDims.first;
  int64_t kernelInputFeatureDimension = parsedDims.second[KIFeature];
  int64_t kernelOutputFeatureDimension = parsedDims.second[KOFeature];
  if (parser.parseArrow()) {
    return failure();
  }
  if (parseDims({IOBatch, IOFeature}, parsedDims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> outputSpatialDimensions = parsedDims.first;
  int64_t outputBatchDimension = parsedDims.second[IOBatch];
  int64_t outputFeatureDimension = parsedDims.second[IOFeature];
  dnums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), inputBatchDimension,
      inputFeatureDimension, inputSpatialDimensions,
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      kernelSpatialDimensions, outputBatchDimension, outputFeatureDimension,
      outputSpatialDimensions);

  return success();
}

Attribute ConvDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  ConvDimensionNumbersAttr dnums;
  if (succeeded(parser.parseOptionalKeyword("raw"))) {
    if (failed(parseConvolutionDimensionsRaw(parser, dnums))) return {};
    return dnums;
  }
  if (failed(parseConvolutionDimensions(parser, dnums))) return {};
  if (failed(parser.parseGreater())) return {};
  return dnums;
}

// Custom printer and parser for ArgResultAliasAttr.
constexpr char kMustAlias[] = "must_alias";
constexpr char kResult[] = "result_index";
constexpr char kArgTupleIndices[] = "tuple_indices";

void ArgResultAliasAttr::print(AsmPrinter& printer) const {
  printer << "<";

  // The attribute can have empty tuple indices. Only print argument tuple
  // indices if they are non-empty.
  if (!getArgTupleIndices().empty())
    printer << kArgTupleIndices << " = [" << getArgTupleIndices() << "], ";

  // Print the result index followed by any result tuple indices if present.
  printer << kResult << " = [";
  printer << getResultIndex();
  if (!getResultTupleIndices().empty()) {
    printer << ", " << getResultTupleIndices();
  }
  printer << "]";

  // Print the "must_alias" keyword if this is a must alias, otherwise skip.
  if (getIsMustAlias()) printer << ", " << kMustAlias;

  printer << ">";
}

Attribute ArgResultAliasAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  llvm::SmallVector<int64_t> argTupleIndices;
  // The first element of result indices holds the aliased result index and the
  // remaining elements are the result tuple indices.
  llvm::SmallVector<int64_t> resultIndices;
  bool isMustAlias = false;

  // This conveys to parseStruct that keyword "must_alias" (3rd field) is not
  // followed by a "=", but other fields are.
  llvm::SmallVector<bool, 3> parseEqual = {true, true, false};

  if (failed(parseStruct(parser, {kArgTupleIndices, kResult, kMustAlias},
                         {[&]() { return parseDims(parser, argTupleIndices); },
                          [&]() {
                            // Since the first element is the index of result,
                            // at least one element is expected.
                            return parseDimsWithMinimumElements(
                                parser, resultIndices, /*minElements=*/1);
                          },
                          [&]() {
                            // always succeeds if the keyword "must_alias" was
                            // parsed
                            isMustAlias = true;
                            return success();
                          }},
                         parseEqual))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing argument-result alias attribute";
    return {};
  }

  int64_t resultIndex = resultIndices[0];
  auto resultTupleIndices =
      ArrayRef<int64_t>{resultIndices.begin() + 1, resultIndices.end()};

  return ArgResultAliasAttr::get(parser.getContext(), argTupleIndices,
                                 resultIndex, resultTupleIndices, isMustAlias);
}

// Returns the element type pointed to by `indices` in type `t`. If the indices
// are invalid, returns nullptr.
static Type getTypeFromTupleIndices(Type type, ArrayRef<int64_t> indices) {
  Type current = type;
  for (auto index : indices) {
    TupleType tupleType = current.dyn_cast<TupleType>();
    if (!tupleType || index >= tupleType.size()) return {};
    current = tupleType.getType(index);
  }
  return current;
}

static LogicalResult verifyArgResultAliasAttr(StringAttr attrName,
                                              ArgResultAliasAttr aliasAttr,
                                              unsigned argIndex,
                                              Operation* op) {
  // The attribute can only be applied to function-like operations.
  if (!isa<mlir::FunctionOpInterface>(op))
    return op->emitOpError() << "attribute " << attrName
                             << " can only be used on function-like operations";

  // Verify there are no negative indices.
  auto tupleIndices = llvm::concat<const int64_t>(
      aliasAttr.getArgTupleIndices(), aliasAttr.getResultTupleIndices());
  if (llvm::any_of(tupleIndices, [](const int64_t val) { return val < 0; }) ||
      aliasAttr.getResultIndex() < 0)
    return op->emitOpError()
           << "attribute " << attrName
           << " expects all argument and result indices to be >= 0";

  // Verify that the result index is not out of range. Since the attribute is a
  // function argument attribute, the argument index is always correct when this
  // verifier is called.
  FunctionOpInterface funcOp = cast<FunctionOpInterface>(op);
  ArrayRef<Type> argTypes = funcOp.getArgumentTypes();
  ArrayRef<Type> resultTypes = funcOp.getResultTypes();
  if (aliasAttr.getResultIndex() >= resultTypes.size())
    return op->emitOpError()
           << "attribute " << attrName
           << " result index is out of range, must be <" << resultTypes.size();

  // Verify that argument and result types pointed to by the indices are valid
  // and compatible.
  Type argType = getTypeFromTupleIndices(argTypes[argIndex],
                                         aliasAttr.getArgTupleIndices());
  if (!argType)
    return op->emitOpError()
           << "attribute " << attrName << " argument tuple indices are invalid";
  Type resultType =
      getTypeFromTupleIndices(resultTypes[aliasAttr.getResultIndex()],
                              aliasAttr.getResultTupleIndices());
  if (!resultType)
    return op->emitOpError()
           << "attribute " << attrName << " result tuple indices are invalid";

  if (failed(mlir::verifyCompatibleShape(argType, resultType)) ||
      getElementTypeOrSelf(argType) != getElementTypeOrSelf(resultType))
    return op->emitOpError() << "attribute " << attrName
                             << " aliases do not have compatible types, "
                             << argType << " vs. " << resultType;
  return success();
}

//===----------------------------------------------------------------------===//
// Type utilities
//===----------------------------------------------------------------------===//

Type getExpressedTypeOrSelf(Type type) {
  auto quantType = type.dyn_cast<quant::QuantizedType>();
  return quantType ? quantType.getExpressedType() : type;
}

bool isCompatibleForMhloTypeInference(Type tp1, Type tp2) {
  // Dynamism: We don't require shapes to be the same, we only require them
  // to be compatible, which means that:
  //   1) At least one of the shapes is unranked.
  //   2) Or both shapes have the same rank and their dimensions are compatible,
  //     i.e. for each pair of corresponding dimensions:
  //       2.1) At least one of the dimensions is dynamic,
  //       2.2) Or both dimensions are equal.
  // These relaxed rules simplify the implementation of type inference, allowing
  // ops with partially inferred types to pass verification.
  auto stp1 = tp1.dyn_cast<ShapedType>();
  auto stp2 = tp2.dyn_cast<ShapedType>();
  if (stp1 && stp2) {
    return succeeded(verifyCompatibleShape(stp1, stp2)) &&
           isCompatibleForMhloTypeInference(stp1.getElementType(),
                                            stp2.getElementType());
  }

  // Quantization: In the most general case, we allow any combination of
  // quantized/non-quantized across any combination of operands/results,
  // and some differences in quantization parameters across operands/results.
  // Individual ops may introduce additional constraints.
  auto qtp1 = tp1.dyn_cast<quant::QuantizedType>();
  auto qtp2 = tp2.dyn_cast<quant::QuantizedType>();
  if (qtp1 && qtp2) {
    if (qtp1.getStorageType() != qtp2.getStorageType() ||
        qtp1.getStorageTypeMin() != qtp2.getStorageTypeMin() ||
        qtp1.getStorageTypeMax() != qtp2.getStorageTypeMax())
      return false;
  }
  auto etp1 = getExpressedTypeOrSelf(tp1);
  auto etp2 = getExpressedTypeOrSelf(tp2);

  // Sparsity: In the most general case, we allow any combination of
  // sparsity/denseness across any combination of operands/results, as well as
  // differences in sparsity encodings for operands and results.
  // Individual ops may introduce additional constraints.
  // No additional code is needed to check this because of how sparsity is
  // currently implemented.

  // Default case: Unless dynamism, quantization and/or sparsity are involved,
  // the types are required to be exactly equal.
  return etp1 == etp2;
}

//===----------------------------------------------------------------------===//
// Builder utilities
//===----------------------------------------------------------------------===//

// Builds the region `body` for mhlo.sort's comparator: for each type in
// `element_types`, create two block arguments, one for lhs and one for rhs, and
// generates mhlo.compare op to compare them with the given `direction`.
//
// Note that this right now only does comparision on the first pair of block
// arguments.
static void buildSortComparisonBody(llvm::ArrayRef<Type> elementTypes,
                                    ComparisonDirection direction,
                                    llvm::Optional<StringRef> compareType,
                                    Region* body, OpBuilder* builder) {
  OpBuilder::InsertionGuard insertionPointGurad(*builder);

  Location loc = body->getLoc();
  Block* block = builder->createBlock(body);
  // Add two arguments for each element type.
  for (Type elementType : elementTypes) {
    TensorType tensorType = RankedTensorType::get({}, elementType);
    block->addArguments({tensorType, tensorType},
                        SmallVector<Location, 2>(2, loc));
  }

  ComparisonType typeAttr;
  if (compareType)
    typeAttr = symbolizeComparisonType(*compareType).getValue();
  else
    typeAttr = ComparisonType::NOTYPE;
  Value compare = builder->create<mhlo::CompareOp>(
      loc, block->getArgument(0), block->getArgument(1), direction, typeAttr);

  builder->create<mhlo::ReturnOp>(loc, compare);
}

SortOp CreateSortOp(PatternRewriter* rewriter, const Location& loc,
                    const llvm::ArrayRef<Value>& operands,
                    const llvm::ArrayRef<Type>& elementTypes, int64_t dimension,
                    bool isStable, ComparisonDirection direction) {
  assert(!operands.empty() && "No operands to sort");
  // Create the sort op.
  auto sortOp =
      rewriter->create<mhlo::SortOp>(loc, operands, dimension, isStable);

  // Use TOTALORDER comparison type instead of the default comparison if the
  // element type is of type float.
  llvm::Optional<StringRef> compareType = llvm::None;
  for (auto const& elementType : elementTypes)
    if (elementType.isa<FloatType>()) {
      compareType.emplace("TOTALORDER");
      break;
    }
  buildSortComparisonBody(elementTypes, direction, compareType,
                          &sortOp.comparator(), rewriter);
  return sortOp;
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult deriveShapeFromOperand(
    OpBuilder* builder, Operation* op, Value operand,
    SmallVectorImpl<Value>* reifiedReturnShapes) {
  auto shapedTy = operand.getType().dyn_cast<ShapedType>();
  if (!shapedTy) {
    op->emitOpError() << "operand is not a shaped type";
    return failure();
  }
  reifiedReturnShapes->assign(
      {builder->create<shape::ShapeOfOp>(op->getLoc(), operand)});
  return success();
}

//===----------------------------------------------------------------------===//
// MHLO Dialect Hooks
//===----------------------------------------------------------------------===//

Operation* MhloDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc) {
  // HLO dialect constants require the type of value and result to match.
  if (type != value.getType()) return nullptr;
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (auto elementsAttr = value.dyn_cast<ElementsAttr>())
    return builder.create<mhlo::ConstOp>(loc, type, elementsAttr);
  return nullptr;
}

LogicalResult MhloDialect::verifyRegionArgAttribute(Operation* op,
                                                    unsigned region_index,
                                                    unsigned argIndex,
                                                    NamedAttribute attr) {
  if (auto aliasAttr = attr.getValue().dyn_cast<ArgResultAliasAttr>()) {
    if (failed(
            verifyArgResultAliasAttr(attr.getName(), aliasAttr, argIndex, op)))
      return failure();
  }
  return success();
}

LogicalResult MhloDialect::verifyOperationAttribute(Operation* op,
                                                    NamedAttribute attr) {
  if (auto aliasAttr = attr.getValue().dyn_cast<ArgResultAliasAttr>()) {
    if (!isa<mlir::FunctionOpInterface>(op))
      return op->emitOpError()
             << "attribute " << attr.getName()
             << " can only be used on function-like operations";
  }
  return success();
}

}  // namespace mhlo
}  // namespace mlir
