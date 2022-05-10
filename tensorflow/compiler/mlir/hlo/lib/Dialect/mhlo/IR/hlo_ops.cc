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
static T Clamp(const T& value, const T& lower, const T& upper) {
  assert(lower <= upper);
  return std::max(lower, std::min(value, upper));
}

// Verifies that dimension attribute for the op correctly indexes in operand or
// result shape.
template <typename OpT>
static LogicalResult VerifyDimAttr(OpT op) {
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
DenseIntElementsAttr BuildSliceLimits(DenseIntElementsAttr start_indices,
                                      DenseIntElementsAttr slice_sizes,
                                      Builder* builder) {
  SmallVector<int64_t, 4> slice_limits;
  for (int64_t i = 0; i < slice_sizes.getNumElements(); ++i) {
    int64_t start_index = start_indices.getValues<IntegerAttr>()[i].getInt();
    int64_t slice_size = slice_sizes.getValues<IntegerAttr>()[i].getInt();
    slice_limits.push_back(start_index + slice_size);
  }
  return builder->getI64TensorAttr(slice_limits);
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void ReplaceOpWithRegion(PatternRewriter& rewriter, Operation* op,
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
Value MaybeCastTo(OpBuilder& b, Location loc, Value value, Type type) {
  if (type == value.getType()) return value;
  assert(type.isIndex() || value.getType().isIndex());
  return b.create<arith::IndexCastOp>(loc, type, value);
}

//===----------------------------------------------------------------------===//
// Utilities for verifiers
//===----------------------------------------------------------------------===//

// Convert a 1D dense int64 attribute to a list of values.
SmallVector<int64_t> convertDenseIntAttr(
    llvm::Optional<mlir::DenseIntElementsAttr> optional_attr) {
  if (!optional_attr.hasValue()) return SmallVector<int64_t>{};

  mlir::DenseIntElementsAttr attr = *optional_attr;
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

// Convert a 1D or Nx2 dense int64 attribute to a list of tuples.
FailureOr<SmallVector<std::pair<int64_t, int64_t>>> convertNx2Attribute(
    llvm::Optional<mlir::DenseIntElementsAttr> optional_attr, Location loc) {
  if (!optional_attr.hasValue())
    return SmallVector<std::pair<int64_t, int64_t>>{};
  mlir::DenseIntElementsAttr attr = *optional_attr;

  auto attr_type = attr.getType().cast<RankedTensorType>();  // ensured by ODS.
  if (attr_type.getRank() > 1) {
    if (attr_type.getRank() != 2 || attr_type.getShape()[1] != 2)
      return (mlir::emitError(loc) << "expects the shape of padding-attribute "
                                      "to be {N, 2}, but got {"
                                   << attr_type.getShape() << "}.",
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
int64_t stridedBound(int64_t bound, int64_t window_size, int64_t stride) {
  assert(window_size >= 0 && "Expected window size to be >= 0");
  assert(bound >= 0 && "Expected bound to be >= 0");

  if (bound == 0 || window_size > bound) return 0;

  // Without considering stride, the maximum valid offset is bound -
  // window_size. Taking stride into account, the valid offsets then have the
  // form q * stride for q = 0, ..., Q such that q * stride <= bound -
  // window_size. This implies that Q equals floor(bound - window_size /
  // stride). There are Q + 1 valid values of q, yielding the formula below.
  return (bound - window_size) / stride + 1;
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
  int64_t padding_low = 0;
  int64_t padding_high = 0;
  int64_t window_dilation = 1;
  int64_t base_dilation = 1;
  bool window_reversal = false;
};

// Verifies various properties of window-attributes (viz., stride, padding,
// lhs_dilation and rhs_dilation) and collects all the window-attributes for
// each kernel spatial dimensions.
FailureOr<SmallVector<WindowDimension>>
verifyWindowAttributesAndInferWindowDimensions(
    ArrayRef<int64_t> window_dimensions, ArrayRef<int64_t> window_strides,
    ArrayRef<std::pair<int64_t, int64_t>> padding,
    ArrayRef<int64_t> lhs_dilation, ArrayRef<int64_t> rhs_dilation,
    Location loc) {
  const auto verifySize = [&](const size_t attrSize,
                              StringRef attrName) -> LogicalResult {
    if (attrSize == 0 || attrSize == window_dimensions.size()) return success();
    return mlir::emitError(loc)
           << "expects " << attrName
           << " to have same dimension-size as size of "
              "window dimensions "
              "("
           << window_dimensions.size() << "), but got: " << attrSize << ".";
  };

  if (failed(verifySize(window_strides.size(), "window-strides")))
    return failure();
  if (failed(verifySize(lhs_dilation.size(), "base-dilation factors")))
    return failure();
  if (failed(verifySize(rhs_dilation.size(), "window-dilation factors")))
    return failure();
  if (failed(verifySize(padding.size(), "padding-entries"))) return failure();

  SmallVector<WindowDimension> window(window_dimensions.size());
  for (size_t i = 0; i < window_dimensions.size(); i++) {
    WindowDimension& dim = window[i];

    dim.size = window_dimensions[i];
    if (!isDynamicDimSize(dim.size) && dim.size <= 0)
      return (mlir::emitError(loc)
                  << "expects window to have positive value for " << i
                  << "-th window dimension, but got " << dim.size << ".",
              failure());

    if (!window_strides.empty()) dim.stride = window_strides[i];
    if (dim.stride <= 0)
      return (mlir::emitError(loc)
                  << "expects window to have positive stride for " << i
                  << "-th window dimension, but got " << dim.stride << ".",
              failure());

    if (!lhs_dilation.empty()) dim.base_dilation = lhs_dilation[i];
    if (dim.base_dilation <= 0)
      return (mlir::emitError(loc) << "expects window to have positive base "
                                      "dilation factor for "
                                   << i << "-th window dimension, but got "
                                   << dim.base_dilation << ".",
              failure());

    if (!rhs_dilation.empty()) dim.window_dilation = rhs_dilation[i];
    if (dim.window_dilation <= 0)
      return (mlir::emitError(loc) << "expects window to have positive window "
                                      "dilation factor for "
                                   << i << "-th window dimension, but got "
                                   << dim.window_dilation << ".",
              failure());

    if (!padding.empty()) {
      dim.padding_low = padding[i].first;
      dim.padding_high = padding[i].second;
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
    const ArrayRef<int64_t> base_shape,
    const ArrayRef<WindowDimension> window) {
  assert(base_shape.size() == window.size() &&
         "Size of window dimensions must match the size of base shape.");

  SmallVector<int64_t> output_dimensions(window.size());
  for (int64_t i = 0; i < window.size(); ++i) {
    if (isDynamicDimSize(base_shape[i]) || isDynamicDimSize(window[i].size)) {
      output_dimensions[i] = ShapedType::kDynamicSize;
    } else {
      const auto& dim = window[i];

      const int64_t dilated_base =
          dilatedBound(base_shape[i], dim.base_dilation);
      const int64_t padded_dilated_base =
          dim.padding_low + dilated_base + dim.padding_high;
      const int64_t dilated_window =
          dilatedBound(dim.size, dim.window_dilation);

      output_dimensions[i] =
          stridedBound(padded_dilated_base, dilated_window, dim.stride);
    }
  }

  return output_dimensions;
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

}  // namespace

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceScatterOp::verify() {
  if (failed(mlir::hlo::VerifyReplicaGroups(*this, /*is_uniform_sized=*/true)))
    return failure();
  auto operand_type = operand().getType().cast<TensorType>();
  bool operand_type_ranked = operand_type.isa<RankedTensorType>();
  Block& block = computation().front();
  SmallVector<TensorType> accumulator_subshapes;
  if (failed(verifyReducerShape(
          this->getLoc(), block, {operand_type},
          {RankedTensorType::get({}, operand_type.getElementType())},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/!operand_type_ranked, accumulator_subshapes)))
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
  auto verify_types_and_layouts =
      [this](TypeRange types, mlir::ArrayAttr layouts,
             const std::string& value_name) -> LogicalResult {
    if (types.size() != layouts.size())
      return emitOpError() << "Number of " << value_name
                           << "s must match the number of " << value_name
                           << " layouts, " << types.size()
                           << " != " << layouts.size();

    for (const auto& indexed_type_and_layout :
         llvm::enumerate(llvm::zip(types, layouts))) {
      // Get index for more descriptive error message.
      auto index = indexed_type_and_layout.index();

      auto type = std::get<0>(indexed_type_and_layout.value());
      auto layout = std::get<1>(indexed_type_and_layout.value())
                        .cast<DenseIntElementsAttr>();

      if (type.isa<TupleType>())
        return emitOpError() << "Tuple types are not fully supported with "
                                "layout constraints yet";
      auto tensor_type = type.dyn_cast<TensorType>();

      // For non-tensor types such as !mhlo.token, the layout should be empty.
      if (!tensor_type) {
        if (layout.empty()) continue;
        return emitOpError()
               << "Only tensor types can have non-empty layout: " << value_name
               << " #" << index << " of type " << type << " has layout "
               << layout;
      }

      // For unranked tensors, we cannot verify the compatibility with layout
      // any further.
      if (!tensor_type.hasRank()) continue;

      // Layout must be a permutation of [0, N) where N is the rank of the
      // tensor type.
      std::vector<int64_t> range(tensor_type.getRank());
      std::iota(range.begin(), range.end(), 0);
      if (tensor_type.getRank() != layout.size() ||
          !std::is_permutation(range.begin(), range.end(), layout.begin()))
        return emitOpError() << "incorrect layout " << layout << " for type "
                             << type << ", layout must be a permutation of [0, "
                             << tensor_type.getRank() << ")";
    }
    return success();
  };

  // At this point both `operand_layouts` and `result_layouts` are defined.
  ArrayAttr operand_layouts = this->operand_layouts().getValue();
  ArrayAttr result_layouts = this->result_layouts().getValue();

  // Full support for layouts for arbitrary nesting of tuples is not
  // supported yet.
  //
  // If result does not have any tuples, then i-th element of `result_layouts`
  // specifies the layout constraints on i-th result.
  //
  // For the common case of a single tuple result packing non-tuple values, the
  // i-th element of `result_layouts` specifies layout for i-th element of the
  // result tuple.
  TypeRange result_types;
  if (getNumResults() == 1 && getResult(0).getType().isa<TupleType>())
    result_types = getResult(0).getType().cast<TupleType>().getTypes();
  else
    result_types = getResultTypes();

  // Verify that operands and operand layouts match.
  if (failed(verify_types_and_layouts(getOperandTypes(), operand_layouts,
                                      "operand")))
    return failure();

  // Verify that results and result layouts match.
  return verify_types_and_layouts(result_types, result_layouts, "result");
}

void CustomCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // CustomCall has "all possible effects" unless the has_side_effect is present
  // and set to false.
  auto has_side_effect = (*this)->getAttrOfType<BoolAttr>("has_side_effect");
  if (has_side_effect && !has_side_effect.getValue()) return;
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
LogicalResult CholeskyOp::verify() {
  auto a_type = a().getType().dyn_cast<RankedTensorType>();
  if (!a_type) return success();  // Nothing to check for unranked tensors

  auto a_shape = a_type.getShape();
  if (a_shape.size() < 2) {
    return emitOpError() << "argument 'a' must have rank >= 2, got shape "
                         << a_shape << ".";
  }

  auto last_dim = a_shape[a_shape.size() - 1];
  auto penultimate_dim = a_shape[a_shape.size() - 2];
  if (isDynamicDimSize(last_dim) || isDynamicDimSize(penultimate_dim)) {
    return success();
  }
  if (last_dim != penultimate_dim) {
    return emitOpError()
           << "minor dimensions of 'a' must have equal size, got shape "
           << a_shape << ".";
  }

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
  auto element_type = lhs.getElementType();
  if (!lhs.hasRank() || !rhs.hasRank()) {
    return UnrankedTensorType::get(element_type);
  }

  // vector dot vector
  if (1 == lhs.getRank() && 1 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(0), rhs.getDimSize(0))) {
    return RankedTensorType::get({}, element_type);
  }
  // matrix dot vector
  if (2 == lhs.getRank() && 1 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(1), rhs.getDimSize(0))) {
    return RankedTensorType::get({lhs.getDimSize(0)}, element_type);
  }
  // vector dot matrix
  if (1 == lhs.getRank() && 2 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(0), rhs.getDimSize(0))) {
    return RankedTensorType::get({rhs.getDimSize(1)}, element_type);
  }
  // matrix dot matrix
  if (2 == lhs.getRank() && 2 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(1), rhs.getDimSize(0))) {
    int64_t shape[2] = {lhs.getDimSize(0), rhs.getDimSize(1)};
    return RankedTensorType::get(shape, element_type);
  }
  return {};
}
}  // namespace

LogicalResult DotOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  DotOp::Adaptor op(operands);
  auto lhs_type = op.lhs().getType().cast<ShapedType>();
  auto rhs_type = op.rhs().getType().cast<ShapedType>();
  inferredReturnTypes.push_back(inferDotReturnType(lhs_type, rhs_type));
  return success();
}

LogicalResult DotOp::verify() {
  auto lhs_type = lhs().getType().cast<ShapedType>();
  auto rhs_type = rhs().getType().cast<ShapedType>();
  auto result_type = getType().cast<ShapedType>();
  auto expect_return_type = inferDotReturnType(lhs_type, rhs_type);
  if (!expect_return_type) {
    return emitError() << "Unexpected operands type: " << lhs_type << " and "
                       << rhs_type;
  }
  if (result_type.hasRank() && expect_return_type.hasRank()) {
    if (result_type.getShape() != expect_return_type.getShape()) {
      return emitError() << "Unexpected result type: has " << result_type
                         << " but inferred " << expect_return_type
                         << " from operands " << lhs_type << " and "
                         << rhs_type;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DotGeneralOp
//===----------------------------------------------------------------------===//

LogicalResult DotGeneralOp::verify() {
  auto dot_dimension_numbers = this->dot_dimension_numbers();
  int64_t lhs_batching_dimensions_size =
      dot_dimension_numbers.getLhsBatchingDimensions().size();
  int64_t rhs_batching_dimensions_size =
      dot_dimension_numbers.getRhsBatchingDimensions().size();
  if (lhs_batching_dimensions_size != rhs_batching_dimensions_size) {
    return emitError()
           << "lhs and rhs should have the same number of batching dimensions";
  }
  int64_t lhs_contracting_dimensions_size =
      dot_dimension_numbers.getLhsContractingDimensions().size();
  int64_t rhs_contracting_dimensions_size =
      dot_dimension_numbers.getRhsContractingDimensions().size();
  if (lhs_contracting_dimensions_size != rhs_contracting_dimensions_size) {
    return emitError() << "lhs and rhs should have the same number of "
                          "contracting dimensions";
  }
  return success();
}

LogicalResult DotGeneralOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto lhs_type = lhs().getType().dyn_cast<ShapedType>();
  auto rhs_type = rhs().getType().dyn_cast<ShapedType>();
  if (!lhs_type || !rhs_type) {
    return failure();
  }

  Adaptor adaptor(operands);
  auto dim_numbers = dot_dimension_numbers();
  SmallVector<Value> dimensions;
  for (const int64_t lhs_dim : dim_numbers.getLhsBatchingDimensions()) {
    dimensions.push_back(
        builder.create<tensor::DimOp>(getLoc(), adaptor.lhs(), lhs_dim));
  }

  for (int64_t i = 0; i < lhs_type.getRank(); i++) {
    if (!llvm::is_contained(dim_numbers.getLhsContractingDimensions(), i) &&
        !llvm::is_contained(dim_numbers.getLhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.lhs(), i));
    }
  }
  for (int64_t i = 0; i < rhs_type.getRank(); i++) {
    if (!llvm::is_contained(dim_numbers.getRhsContractingDimensions(), i) &&
        !llvm::is_contained(dim_numbers.getRhsBatchingDimensions(), i)) {
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
  auto fft_rank = fft_length().size();
  if (!(fft_rank <= 3 && fft_rank >= 1)) {
    return emitOpError() << "rank must be between 1 and 3, but got " << fft_rank
                         << ".";
  }

  // P2.
  auto operand_type = operand().getType().dyn_cast<RankedTensorType>();
  if (!operand_type) return success();
  auto operand_shape = operand_type.getShape();
  if (operand_shape.size() < fft_rank) {
    return emitOpError() << "operand rank must be greater than fft rank of "
                         << fft_rank << " for operand of type " << operand_type
                         << ".";
  }

  if (fft_type() == FftType::RFFT) {
    auto shape_back = operand_shape.take_back(fft_rank);
    for (auto it : llvm::zip(shape_back, fft_length().getValues<int64_t>())) {
      if (std::get<0>(it) != std::get<1>(it)) {
        return emitError()
               << "RFFT requires innermost dimensions match fft_length. Got: "
               << operand_shape << " but wanted " << fft_length() << ".";
      }
    }
  }
  if (fft_type() == FftType::IRFFT) {
    auto shape_back = operand_shape.take_back(fft_rank).drop_back();
    for (auto it : llvm::zip(shape_back, fft_length().getValues<int64_t>())) {
      if (std::get<0>(it) != std::get<1>(it)) {
        return emitError() << "IRFFT requires non-final dimensions "
                              "match fft_length. Got: "
                           << operand_shape << " but wanted " << fft_length()
                           << ", and " << std::get<0>(it)
                           << " != " << std::get<1>(it) << ".";
      }
    }
    if (operand_shape[operand_shape.size() - 1] !=
        fft_length().getValues<int64_t>()[fft_rank - 1] / 2 + 1)
      return emitError() << "IRFFT requires innermost dimension match "
                            "fft_length[-1]/2+1. Got: "
                         << operand_shape << " but fft_length is "
                         << fft_length() << ".";
  }

  // P3. Element type agreement
  // FFT : C -> C
  // IFF : C -> C
  // RFFT : R -> C
  // IRFFT : C -> R
  if (fft_type() == FftType::RFFT) {
    if (operand_type.getElementType().isa<ComplexType>()) {
      return emitError() << "RFFT takes a real tensor as input, but is given "
                         << operand_type << ".";
    }
  } else if (!operand_type.getElementType().isa<ComplexType>()) {
    return emitError() << stringifyFftType(fft_type())
                       << " takes a complex tensor as input, but is given "
                       << operand_type << ".";
  }

  auto result_type = getResult().getType().dyn_cast<RankedTensorType>();
  if (!result_type) return success();
  if (fft_type() == FftType::IRFFT) {
    if (result_type.getElementType().isa<ComplexType>()) {
      return emitError()
             << "IRFFT produces a real tensor as output, but is given "
             << result_type << ".";
    }
  } else if (!result_type.getElementType().isa<ComplexType>()) {
    return emitError() << stringifyFftType(fft_type())
                       << " produces a complex tensor as output, but is given "
                       << result_type << ".";
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

    RankedTensorType operand_type =
        gather->getOperand(0).getType().dyn_cast<RankedTensorType>();
    if (!operand_type || !operand_type.hasStaticShape()) return failure();

    auto slice_end =
        llvm::to_vector<8>(gather.slice_sizes().getValues<int64_t>());
    llvm::SmallVector<int64_t, 8> slice_start(slice_end.size(), 0);
    for (auto it :
         llvm::zip(dnums.getStartIndexMap(), index.getValues<APInt>())) {
      int64_t map_index = std::get<0>(it);
      // Clamp the indices within bounds to faithfully mirror gather semantics.
      int64_t offset =
          Clamp(std::get<1>(it).getSExtValue(), static_cast<int64_t>(0),
                operand_type.getDimSize(map_index) - slice_end[map_index]);
      slice_start[map_index] += offset;
      slice_end[map_index] += offset;
    }

    llvm::SmallVector<int64_t, 8> slice_stride(slice_end.size(), 1);
    llvm::SmallVector<int64_t, 8> slice_shape(slice_end.size());
    for (size_t i = 0; i < slice_end.size(); ++i) {
      slice_shape[i] = slice_end[i] - slice_start[i];
    }
    Type element_type = gather.getType().cast<TensorType>().getElementType();
    auto slice_type = RankedTensorType::get(slice_shape, element_type);
    Value result = rewriter.create<SliceOp>(
        gather.getLoc(), slice_type, gather.getOperand(0),
        rewriter.getI64TensorAttr(slice_start),
        rewriter.getI64TensorAttr(slice_end),
        rewriter.getI64TensorAttr(slice_stride));

    auto collapsed_slice_dims = dnums.getCollapsedSliceDims();
    if (!collapsed_slice_dims.empty()) {
      llvm::SmallVector<int64_t, 8> reshape_shape;
      for (size_t i = 0; i < slice_shape.size(); ++i) {
        if (llvm::count(collapsed_slice_dims, i) == 0) {
          reshape_shape.push_back(slice_shape[i]);
        }
      }
      auto reshape_type = RankedTensorType::get(reshape_shape, element_type);
      result =
          rewriter.create<ReshapeOp>(gather.getLoc(), reshape_type, result);
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
                        SmallVectorImpl<Value>& slice_sizes) {
  for (int64_t val : gather->slice_sizes().getValues<int64_t>()) {
    slice_sizes.push_back(builder.create<arith::ConstantIndexOp>(loc, val));
  }
}

void getSliceSizeValues(DynamicGatherOp* d_gather, OpBuilder& builder,
                        Location loc, ValueRange operands,
                        SmallVectorImpl<Value>& slice_size_values) {
  DynamicGatherOp::Adaptor adaptor(operands);
  Value slice_sizes = adaptor.slice_sizes();
  auto slice_sizes_ty = slice_sizes.getType().cast<ShapedType>();
  for (int64_t i = 0; i < slice_sizes_ty.getDimSize(0); ++i) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    slice_size_values.push_back(
        builder.create<tensor::ExtractOp>(loc, slice_sizes, idx));
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
    return MaybeCastTo(builder, loc, v, shapeElTy);
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
LogicalResult GetDimensionSizeOp::verify() { return VerifyDimAttr(*this); }

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

  auto iota_dimension = this->iota_dimension();
  if (iota_dimension >= shape.getRank() || iota_dimension < 0)
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
    auto result_ty = iota.getType().cast<ShapedType>();
    if (!result_ty.hasRank() || result_ty.getRank() < 2) {
      return failure();
    }

    auto iota_dimension = iota.iota_dimension();

    auto iota_type = RankedTensorType::get(
        {result_ty.getDimSize(iota_dimension)}, result_ty.getElementType());

    auto new_iota = rewriter.create<IotaOp>(iota.getLoc(), iota_type,
                                            rewriter.getI64IntegerAttr(0));

    auto broadcast_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iota_dimension});
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(iota, result_ty, new_iota,
                                                  broadcast_attr);
    return success();
  }
};

void IotaOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<IotaBroadcast>(context);
}

OpFoldResult IotaOp::fold(ArrayRef<Attribute> operands) {
  auto dimension = iota_dimension();
  auto result_ty = getResult().getType().cast<ShapedType>();
  if (result_ty.hasRank() && result_ty.getDimSize(dimension) == 1) {
    Builder builder(getContext());
    return builder.getZeroAttr(result_ty);
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
    auto result_ty = iota.getType().cast<ShapedType>();
    if (!result_ty.hasStaticShape()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IotaOp>(iota, result_ty, iota.iota_dimension());
    return success();
  }
};

// Dynamic Iota operations across multiple dimensions can be reduced to an iota
// and a ranked broadcast.
struct DynamicIotaBroadcast : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter& rewriter) const override {
    auto result_ty = iota.getType().cast<ShapedType>();
    if (!result_ty.hasRank() || result_ty.getRank() < 2) {
      return failure();
    }

    auto iota_dimension = iota.iota_dimension();
    auto iota_dimension_int = iota_dimension;

    auto converted_shape = rewriter.create<arith::IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            iota.output_shape().getType().cast<ShapedType>().getShape(),
            rewriter.getI64Type()),
        iota.output_shape());

    auto sliced_shape = rewriter.create<SliceOp>(
        iota.getLoc(), converted_shape,
        rewriter.getI64TensorAttr(iota_dimension_int),
        rewriter.getI64TensorAttr(iota_dimension_int + 1),
        rewriter.getI64TensorAttr(1));

    auto converted_sliced_shape = rewriter.create<arith::IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            {1},
            iota.output_shape().getType().cast<ShapedType>().getElementType()),
        sliced_shape);

    auto iota_type = RankedTensorType::get(
        {result_ty.getDimSize(iota_dimension_int)}, result_ty.getElementType());

    auto new_iota = rewriter.create<DynamicIotaOp>(
        iota.getLoc(), iota_type, converted_sliced_shape,
        rewriter.getI64IntegerAttr(0));

    auto broadcast_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iota_dimension});
    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        iota, result_ty, new_iota, iota.output_shape(), broadcast_attr);
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
                               Value shape_op) {
  ShapedType result_ty = shape::getExtentTensorType(
      builder.getContext(),
      shape_op.getType().cast<ShapedType>().getDimSize(0));
  if (shape_op.getType() == result_ty) return shape_op;  // Nothing to do.
  return builder.create<arith::IndexCastOp>(loc, result_ty, shape_op);
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
  auto idx_tensor = indices.take_front().front().getType().cast<ShapedType>();
  Type first_elem_ty = idx_tensor.getElementType();
  Type elem_ty;

  for (auto idx : llvm::drop_begin(indices, 1)) {
    idx_tensor = idx.getType().cast<ShapedType>();
    elem_ty = idx_tensor.getElementType();

    if (first_elem_ty != elem_ty) {
      return emitOpError() << "start indices must have same element type "
                              "(encountered mismatch: "
                           << first_elem_ty << " vs " << elem_ty << ")";
    }
  }
  return success();
}

OpFoldResult DynamicUpdateSliceOp::fold(ArrayRef<Attribute> operands) {
  auto operand_shape = this->operand().getType().cast<RankedTensorType>();
  auto update_shape = this->update().getType().cast<RankedTensorType>();

  if (operand_shape != update_shape || !operand_shape.hasStaticShape()) {
    return {};
  }

  // Ensure that indices are 0 constants. The 0 check mostly ensures
  // correctness. For non-constants, the pattern does not fold to avoid hiding
  // the behavior of incorrect user input.
  for (Value index : this->start_indices()) {
    DenseIntElementsAttr de_attr;
    if (!matchPattern(index, m_Constant(&de_attr))) return {};
    if (!de_attr.getSplatValue<IntegerAttr>().getValue().isZero()) return {};
  }
  return this->update();
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto operand_ty = (*operands.begin()).getType().cast<ShapedType>();
  Type element_ty = operand_ty.getElementType();
  if (auto complex_ty = element_ty.dyn_cast<ComplexType>()) {
    element_ty = complex_ty.getElementType();
  }

  Type result_ty;
  if (operand_ty.hasRank()) {
    result_ty = RankedTensorType::get(operand_ty.getShape(), element_ty);
  } else {
    result_ty = UnrankedTensorType::get(element_ty);
  }
  inferredReturnTypes.push_back(result_ty);
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
// ConvertOp
//===----------------------------------------------------------------------===//

void ConvertOp::build(OpBuilder& builder, OperationState& result, Value operand,
                      Type result_element_ty) {
  Type result_ty;
  Type operand_ty = operand.getType();
  if (auto ranked_ty = operand_ty.dyn_cast<RankedTensorType>()) {
    result_ty = RankedTensorType::get(ranked_ty.getShape(), result_element_ty);
  } else {
    result_ty = UnrankedTensorType::get(result_element_ty);
  }
  build(builder, result, result_ty, operand);
}

OpFoldResult ConvertOp::fold(ArrayRef<Attribute> operands) {
  auto operand_ty = getOperand().getType().cast<TensorType>();
  auto result_ty = getResult().getType().cast<TensorType>();
  if (operand_ty == result_ty) return getOperand();

  // If the result has non-static shape, a convert op is necessary to go from
  // static shape to non-static shape.
  if (!result_ty.hasStaticShape()) return {};

  // TODO(hinsu): Handle unsigned types.
  if (operand_ty.getElementType().isUnsignedInteger() ||
      result_ty.getElementType().isUnsignedInteger()) {
    return {};
  }

  // If the operand is constant, we can do the conversion now.
  if (auto elementsAttr = operands.front().dyn_cast_or_null<ElementsAttr>()) {
    return hlo::ConvertElementsAttr(elementsAttr,
                                    getElementTypeOrSelf(getResult()));
  }

  return {};
}

void ConvertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<EliminateIdentityConvert>(context);
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
  if (auto tuple_op = getOperand().getDefiningOp<mhlo::TupleOp>()) {
    return tuple_op.getOperand(index());
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

    Value first_element = op.val().front();
    auto first_element_op = first_element.getDefiningOp<GetTupleElementOp>();
    if (!first_element_op || first_element_op.indexAttr().getInt() != 0)
      return failure();

    Value tuple_predecessor = first_element_op.getOperand();
    if (tuple_predecessor.getType() != op.getType()) return failure();

    for (const auto& element_and_idx :
         llvm::enumerate(op.val().drop_front(1))) {
      auto element_op =
          element_and_idx.value().getDefiningOp<GetTupleElementOp>();
      if (!element_op ||
          element_op.indexAttr().getInt() != element_and_idx.index() + 1 ||
          element_op.getOperand() != tuple_predecessor)
        return failure();
    }

    rewriter.replaceOp(op, tuple_predecessor);
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

LogicalResult AllToAllOp::verify() {
  // If operand is ranked, size of split dimension should be a multiple of split
  // count.
  auto type = getOperand().getType().dyn_cast<RankedTensorType>();
  if (!type) return success();
  auto split_dim_size = type.getDimSize(split_dimension());
  auto split_count = this->split_count();
  if (split_dim_size % split_count != 0) {
    return emitError() << "split dimension has size " << split_dim_size
                       << ", expected to be a multiple of split_count "
                       << split_count;
  }
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
  auto operand_type = operand().getType().cast<RankedTensorType>();
  if (static_cast<int64_t>(feature_index()) >= operand_type.getRank())
    return emitOpError() << "expects feature_index to be smaller "
                            "than the rank of operand type; got feature_index "
                         << feature_index() << ", and rank "
                         << operand_type.getRank() << ".";

  if (static_cast<int64_t>(feature_index()) < 0)
    return emitOpError() << "expects feature_index to be a "
                         << "non-negative number, got "
                         << static_cast<int64_t>(feature_index()) << ".";

  auto grad_output_type = grad_output().getType().cast<RankedTensorType>();
  if (operand_type.getRank() != grad_output_type.getRank())
    return emitOpError() << "expects 'operand' and 'grad_output' to have the "
                            "same rank. but got rank(oprand) "
                         << operand_type.getRank() << " and rank(grad_output) "
                         << grad_output_type.getRank() << ".";

  // C3.
  const int64_t feature_count = operand_type.getShape()[feature_index()];
  const int64_t scale_shape =
      scale().getType().cast<RankedTensorType>().getShape()[0];
  if (scale_shape != feature_count)
    return emitOpError() << "expects the size of scale factor to be "
                            "same as the feature count,"
                            " but the size of scale factor is "
                         << scale_shape << " and the feature count is "
                         << feature_count << ".";

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

  auto operand_type = operand().getType().cast<RankedTensorType>();
  if (static_cast<int64_t>(feature_index()) >= operand_type.getRank())
    return emitOpError() << "expects feature_index to be smaller "
                            "than the rank of operand type; got feature_index "
                         << feature_index() << ", and rank "
                         << operand_type.getRank() << ".";

  if (static_cast<int64_t>(feature_index()) < 0)
    return emitOpError() << "expects feature_index to be a "
                         << "non-negative number, got "
                         << static_cast<int64_t>(feature_index()) << ".";

  // Note:A valid value of feature-index implies 'operand_type.getRank() >=1'.

  const int64_t feature_count = operand_type.getShape()[feature_index()];
  const int64_t scale_shape =
      scale().getType().cast<RankedTensorType>().getShape()[0];
  // Check number of elements in input 'scale' equals feature_count.
  // Together with (5) implies that 'scale', 'offset', 'batch_mean', &
  // 'batch_var' all have the same shape.
  if (scale_shape != feature_count)
    return emitOpError() << "expects the size of scale factor to be "
                            "same as the feature count,"
                            " but the size of scale factor is "
                         << scale_shape << " and the feature count is "
                         << feature_count << ".";

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

  auto operand_type = operand().getType().cast<RankedTensorType>();
  if (static_cast<int64_t>(feature_index()) >= operand_type.getRank())
    return emitOpError() << "expects feature_index to be smaller "
                            "than the rank of operand type; got feature_index "
                         << feature_index() << ", and rank "
                         << operand_type.getRank() << ".";

  if (static_cast<int64_t>(feature_index()) < 0)
    return emitOpError() << "expects feature_index to be a "
                         << "non-negative number, got "
                         << static_cast<int64_t>(feature_index()) << ".";

  // Note:A valid value of feature-index implies 'operand_type.getRank() >=1'.

  const int64_t feature_count = operand_type.getShape()[feature_index()];
  const int64_t scale_size =
      scale().getType().cast<RankedTensorType>().getShape()[0];
  // Check number of elements in input 'scale' equals feature_count.
  // Together with (4) implies that 'scale', 'offset', 'mean', &
  // 'variance' all have the same shape.
  if (scale_size != feature_count)
    return emitOpError() << "expects the size of scale factor to be "
                            "same as the feature count,"
                            " but the size of scale factor is "
                         << scale_size << " and the feature count is "
                         << feature_count << ".";

  return success();
}

//===----------------------------------------------------------------------===//
// BitcastConvertOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastConvertOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto operand_type = operands[0].getType().dyn_cast<RankedTensorType>();
  auto result_type = getType().dyn_cast<RankedTensorType>();

  // Only ranked tensors are supported.
  if (!operand_type || !result_type) return failure();

  // Shape-changing bitcast convert is not implemented.
  // TODO(kramerb): This could be done by adjusting the last dimension.
  DataLayout data_layout = DataLayout::closest(*this);
  unsigned operand_element_size =
      data_layout.getTypeSizeInBits(operand_type.getElementType());
  unsigned result_element_size =
      data_layout.getTypeSizeInBits(result_type.getElementType());
  if (operand_element_size != result_element_size) return failure();

  return ::mlir::mhlo::deriveShapeFromOperand(
      &builder, getOperation(), operands.front(), &reifiedReturnShapes);
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
  auto operand_type = operand.getType().dyn_cast<RankedTensorType>();
  if (!operand_type) return failure();

  Type element_ty = operand_type.getElementType();
  auto dimension_attr = adaptor.broadcast_sizes();
  for (int64_t size : dimension_attr.getValues<int64_t>()) {
    if (size < 0)
      return emitOptionalError(location,
                               "Broadcast with negative dimension size ", size);
  }
  SmallVector<int64_t> shape_values(dimension_attr.getValues<int64_t>());
  llvm::append_range(shape_values, operand_type.getShape());

  inferredReturnShapes.emplace_back(shape_values, element_ty);
  return success();
}

LogicalResult BroadcastOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands);
  Value operand = adaptor.operand();

  auto operand_type = operand.getType().dyn_cast<RankedTensorType>();
  // Unranked tensors are not supported.
  if (!operand_type) return failure();

  Location loc = getLoc();
  SmallVector<Value, 4> shape_values;

  // Collect the broadcast sizes.
  for (const auto& size : broadcast_sizes()) {
    shape_values.push_back(
        builder.create<arith::ConstantIndexOp>(loc, size.getZExtValue()));
  }

  // Collect the operand sizes.
  for (auto index : llvm::seq<int64_t>(0, operand_type.getRank())) {
    shape_values.push_back(
        builder.createOrFold<tensor::DimOp>(loc, operand, index));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shape_values.size())},
                            builder.getIndexType()),
      shape_values));

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
    auto broadcast_values = broadcast_dimensions().getValues<int64_t>();
    if (!std::equal(broadcast_values.begin(), broadcast_values.end(),
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
    auto operand_type = op.operand().getType().dyn_cast<RankedTensorType>();
    auto result_type = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!operand_type || !result_type) {
      return failure();
    }
    auto bs_dim_indices = op.broadcast_dimensions().getValues<int64_t>();
    if (operand_type.hasStaticShape() && result_type.hasStaticShape()) {
      bool same_total_elements =
          operand_type.getNumElements() == result_type.getNumElements();
      // BroadcastInDim equivalent to reshape
      if (llvm::is_sorted(bs_dim_indices) && same_total_elements) {
        rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.operand());
        return success();
      }
      // BroadcastInDim equivalent to transpose
      if (operand_type.getRank() == result_type.getRank() &&
          same_total_elements) {
        rewriter.replaceOpWithNewOp<TransposeOp>(op, op.getType(), op.operand(),
                                                 op.broadcast_dimensions());
        return success();
      }
    }
    // eliminate redundant BroadcastInDim
    if (auto broadcast_in_dim_op = llvm::dyn_cast_or_null<BroadcastInDimOp>(
            op.operand().getDefiningOp())) {
      auto new_indices =
          broadcast_in_dim_op.broadcast_dimensions()
              .mapValues(op.broadcast_dimensions().getElementType(),
                         [&bs_dim_indices](const APInt& dim) -> APInt {
                           return APInt(dim.getBitWidth(),
                                        bs_dim_indices[dim.getSExtValue()],
                                        true);
                         })
              .cast<DenseIntElementsAttr>();
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
          op, op.getType(), broadcast_in_dim_op.operand(), new_indices);
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
    if (!type || !type.hasStaticShape() || !operandType ||
        !operandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires static shape");
    }
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
        op, op.getType(), op.operand(), op.broadcast_dimensions());
    return success();
  }
};

class ChainedDynamicBroadcastInDimCanonicalization
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp bcast,
                                PatternRewriter& rewriter) const override {
    auto preceding_bcast =
        bcast.operand().getDefiningOp<DynamicBroadcastInDimOp>();
    if (!preceding_bcast) return failure();

    // Compose broadcast dimensions.
    DenseIntElementsAttr preceding_bcast_dims =
        preceding_bcast.broadcast_dimensions();
    DenseIntElementsAttr bcast_dims = bcast.broadcast_dimensions();
    SmallVector<APInt, 4> composition;
    for (APInt preceding_dim : preceding_bcast_dims) {
      composition.push_back(
          bcast_dims.getValues<APInt>()[preceding_dim.getZExtValue()]);
    }
    auto composed_bcast_dims =
        DenseIntElementsAttr::get(preceding_bcast_dims.getType(), composition);

    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        bcast, bcast.getType(), preceding_bcast.operand(),
        bcast.output_dimensions(), composed_bcast_dims);
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
  auto element_ty = ComplexType::get(getElementTypeOrSelf(type));
  Type result_ty;
  if (auto ranked_type = type.dyn_cast<RankedTensorType>()) {
    result_ty = RankedTensorType::get(ranked_type.getShape(), element_ty);
  } else if (type.isa<UnrankedTensorType>()) {
    result_ty = UnrankedTensorType::get(element_ty);
  } else {
    result_ty = element_ty;
  }
  inferredReturnTypes.push_back(result_ty);
  return success();
}

OpFoldResult ComplexOp::fold(ArrayRef<Attribute> operands) {
  auto real_op = getOperand(0).getDefiningOp<mhlo::RealOp>();
  auto imag_op = getOperand(1).getDefiningOp<mhlo::ImagOp>();
  if (real_op && imag_op && real_op.getOperand() == imag_op.getOperand()) {
    return real_op.getOperand();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ImagOp
//===----------------------------------------------------------------------===//

namespace {
Type CreateRealType(Type type) {
  auto element_ty = getElementTypeOrSelf(type);
  if (auto complex_ty = element_ty.dyn_cast<ComplexType>()) {
    element_ty = complex_ty.getElementType();
  }

  if (auto ranked_type = type.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(ranked_type.getShape(), element_ty);
  }
  if (type.dyn_cast<UnrankedTensorType>()) {
    return UnrankedTensorType::get(element_ty);
  }

  return element_ty;
}
}  // namespace

LogicalResult ImagOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(CreateRealType(operands[0].getType()));
  return success();
}

OpFoldResult ImagOp::fold(ArrayRef<Attribute> operands) {
  if (auto complex_op = getOperand().getDefiningOp<mhlo::ComplexOp>()) {
    return complex_op.getOperand(1);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

TensorType getSameShapeTensorType(TensorType tensor_type, Type element_type) {
  if (auto ranked_tensor_ty = tensor_type.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(ranked_tensor_ty.getShape(), element_type);
  }
  if (auto unranked_tensor_ty = tensor_type.dyn_cast<UnrankedTensorType>()) {
    return UnrankedTensorType::get(element_type);
  }
  llvm_unreachable("unhandled type");
}

LogicalResult IsFiniteOp::inferReturnTypes(
    MLIRContext* ctx, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto arg_ty = operands.front().getType().cast<TensorType>();
  Builder b(ctx);
  inferredReturnTypes.push_back(getSameShapeTensorType(arg_ty, b.getI1Type()));
  return success();
}

//===----------------------------------------------------------------------===//
// RealOp
//===----------------------------------------------------------------------===//

LogicalResult RealOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(CreateRealType(operands[0].getType()));
  return success();
}

OpFoldResult RealOp::fold(ArrayRef<Attribute> operands) {
  if (auto complex_op = getOperand().getDefiningOp<mhlo::ComplexOp>()) {
    return complex_op.getOperand(0);
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
    llvm::SmallVector<Value, 6> new_operands;
    for (auto operand : op.getOperands()) {
      auto ty = operand.getType().cast<ShapedType>();
      if (!ty.hasRank() || ty.getDimSize(axis) != 0) {
        new_operands.push_back(operand);
      }
    }

    if (!new_operands.empty() && new_operands.size() < op.getNumOperands()) {
      rewriter.replaceOpWithNewOp<ConcatenateOp>(op, op.getResult().getType(),
                                                 new_operands, op.dimension());
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

  auto dimension_attr = attributes.get("dimension").cast<IntegerAttr>();
  auto dimension = dimension_attr.getInt();

  auto first_type = (*operands.begin()).getType().cast<ShapedType>();
  auto out_element = first_type.getElementType();

  for (auto operand : operands.getTypes()) {
    auto element_type = getElementTypeOrSelf(operand);
    if (element_type != out_element) {
      return failure();
    }
  }

  // Find the first ranked input to determine the output rank.
  for (auto type : operands.getTypes()) {
    auto shaped_type = type.cast<ShapedType>();
    if (shaped_type.hasRank()) {
      first_type = shaped_type;
      break;
    }
  }

  // If all inputs are unranked, the result must be unranked.
  if (!first_type.hasRank()) {
    inferredReturnTypes.push_back(UnrankedTensorType::get(out_element));
    return success();
  }

  if (first_type.getRank() == 0)
    return emitOptionalError(location, "rank-0 values cannot be concatenated");

  auto out_shape = llvm::to_vector<6>(first_type.getShape());

  // Determine what the non-concatenate dimensions should be.
  for (auto type : operands.getTypes()) {
    auto shaped_ty = type.cast<ShapedType>();
    if (!shaped_ty.hasRank()) {
      continue;
    }

    for (const auto& it : llvm::enumerate(shaped_ty.getShape())) {
      // If a dimension is not dynamic, the output shape should match.
      if (ShapedType::isDynamic(out_shape[it.index()])) {
        out_shape[it.index()] = it.value();
      }
    }
  }

  out_shape[dimension] = 0;

  for (auto operand : operands.getTypes()) {
    auto type = operand.cast<ShapedType>();
    if (!type.hasRank()) {
      inferredReturnTypes.push_back(UnrankedTensorType::get(out_element));
      return success();
    }

    // If the dimension is dynamic we know the output dimension is dynamic.
    auto dim = type.getShape()[dimension];
    if (dim == -1) {
      out_shape[dimension] = -1;
      break;
    }

    out_shape[dimension] += dim;
  }

  inferredReturnTypes.push_back(RankedTensorType::get(out_shape, out_element));

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

  size_t top_size = 1;
  for (int i = 0, e = axis; i < e; i++) {
    top_size = top_size * shape[i];
  }

  // TODO(b/210478841): Define a constant folding policy that generalizes this.
  if (type.getNumElements() * op->getNumOperands() > UINT32_MAX) {
    return {};
  }

  SmallVector<T, 6> values;
  for (size_t i = 0; i < top_size; i++) {
    for (auto operand : operands) {
      DenseElementsAttr attr = operand.cast<DenseElementsAttr>();
      size_t bottom_size = attr.getNumElements() / top_size;
      auto iter = attr.getValues<T>().begin() + i * bottom_size;
      values.append(iter, iter + bottom_size);
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
  Type element_type = getElementTypeOrSelf(getOperand(0).getType());
  RankedTensorType first_ranked_type;
  int num_operands = getNumOperands();
  for (int i = 0; i < num_operands; i++) {
    auto second_type = getOperand(i).getType().dyn_cast<ShapedType>();
    if (second_type.getElementType() != element_type) {
      return emitOpError(
          llvm::formatv("operands (0) and ({0}) do not match element type", i));
    }

    if (!second_type.hasRank()) {
      continue;
    }

    if (!first_ranked_type) {
      first_ranked_type = second_type.cast<RankedTensorType>();
      continue;
    }

    if (first_ranked_type.getRank() != second_type.getRank()) {
      return emitOpError(
          llvm::formatv("operands (0) and ({0}) do not match rank", i));
    }

    auto first_shape = second_type.getShape();
    auto second_shape = second_type.getShape();
    for (int d = 0; d < first_ranked_type.getRank(); ++d) {
      if (first_shape[d] != second_shape[d] && d != dimension()) {
        return emitOpError(llvm::formatv(
            "operands (0) and ({0}) non-concat dimensions do not match "
            "({1}) != ({2})",
            i, llvm::make_range(first_shape.begin(), first_shape.end()),
            llvm::make_range(second_shape.begin(), second_shape.end())));
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

  auto operand_type = inputs[0].getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operand_type) return failure();

  Location loc = this->getLoc();
  Type shape_scalar_type = builder.getIndexType();
  auto to_shape_scalar_type = [&](Value v) {
    return MaybeCastTo(builder, loc, v, shape_scalar_type);
  };

  SmallVector<SmallVector<Value, 4>, 4> all_shape_values;
  for (size_t input_id = 0; input_id < inputs.size(); ++input_id) {
    Value operand = inputs[input_id];
    auto operand_type = operand.getType().dyn_cast<RankedTensorType>();
    if (!operand_type) return failure();

    SmallVector<Value, 4> shape_vals;
    for (const auto& element : llvm::enumerate(operand_type.getShape())) {
      Value value_dim = to_shape_scalar_type(
          builder.create<tensor::DimOp>(loc, operand, element.index()));
      shape_vals.push_back(value_dim);
    }
    all_shape_values.emplace_back(std::move(shape_vals));
  }

  int axis = this->dimension();
  auto& shape_values = all_shape_values[0];
  for (size_t vec_id = 1; vec_id < all_shape_values.size(); ++vec_id) {
    auto& other_shape_values = all_shape_values[vec_id];
    if (other_shape_values.size() != shape_values.size()) {
      this->emitOpError()
          << "Concatenate expects all operands must be of the same rank";
      return failure();
    }
    shape_values[axis] = builder.create<arith::AddIOp>(
        loc, shape_values[axis], other_shape_values[axis]);
  }

  Value output_shape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shape_values.size())},
                            shape_scalar_type),
      shape_values);
  reifiedReturnShapes.push_back(output_shape);

  return success();
}

//===----------------------------------------------------------------------===//
// DynamicReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicReshapeOp::verify() {
  auto result_type = result().getType().dyn_cast<RankedTensorType>();
  auto output_shape_type =
      output_shape().getType().dyn_cast<RankedTensorType>();
  if (result_type && output_shape_type && output_shape_type.hasStaticShape() &&
      output_shape_type.getDimSize(0) != result_type.getRank()) {
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
    auto operand_type = op.operand().getType().dyn_cast<RankedTensorType>();
    if (!operand_type || operand_type.getRank() != 1 ||
        operand_type.hasStaticShape()) {
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
    Operation* def_op = op.operand().getDefiningOp();
    if (!def_op ||
        !def_op->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) {
      return failure();
    }
    Operation* input_def_op = def_op->getOperand(0).getDefiningOp();
    if (!input_def_op) {
      return failure();
    }
    auto reshape = dyn_cast<DynamicReshapeOp>(*input_def_op);
    if (reshape && reshape.output_shape() == op.output_shape()) {
      rewriter.replaceOp(op, {def_op->getResult(0)});
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

  LogicalResult matchAndRewrite(DynamicSliceOp dynamic_slice,
                                PatternRewriter& rewriter) const override {
    Value input = dynamic_slice.operand();
    auto input_tensor = input.getType().dyn_cast<RankedTensorType>();
    if (!input_tensor || !input_tensor.hasStaticShape()) return failure();

    auto slice_sizes = dynamic_slice.slice_sizes().getValues<int64_t>();
    SmallVector<int64_t, 4> temp_start_indices;
    for (const auto& index_and_slice_start :
         llvm::enumerate(dynamic_slice.start_indices())) {
      APInt val;
      Value start = index_and_slice_start.value();
      int64_t index = index_and_slice_start.index();
      if (!matchPattern(start, m_ConstantInt(&val))) {
        return failure();
      }
      // Clamp the indices within bounds to faithfully mirror dynamic slice
      // semantics.
      int64_t clamped_start =
          Clamp(val.getSExtValue(), static_cast<int64_t>(0),
                input_tensor.getDimSize(index) - slice_sizes[index]);
      temp_start_indices.push_back(clamped_start);
    }

    // At this point we've determined that the start indices are all constants;
    // pack them into a single tensor.
    auto loc = dynamic_slice.getLoc();
    int64_t input_rank = input_tensor.getRank();
    auto slice_start_indices = rewriter.getI64TensorAttr(temp_start_indices);
    DenseIntElementsAttr slice_limits = BuildSliceLimits(
        slice_start_indices, dynamic_slice.slice_sizes(), &rewriter);
    DenseIntElementsAttr slice_strides =
        rewriter.getI64TensorAttr(SmallVector<int64_t, 4>(input_rank, 1));
    auto result = rewriter.create<SliceOp>(loc, input, slice_start_indices,
                                           slice_limits, slice_strides);
    rewriter.replaceOp(dynamic_slice, {result});
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
  int num_slice_sizes = slice_sizes().getNumElements();
  int num_start_indices = start_indices().size();
  if (num_start_indices != num_slice_sizes) {
    return emitOpError() << "has mismatched number of slice sizes ("
                         << num_slice_sizes << ") and number of start indices ("
                         << num_start_indices << ")";
  }
  return success();
}

LogicalResult DynamicSliceOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, regions);
  Value operand = adaptor.operand();
  auto operand_type = operand.getType().dyn_cast<RankedTensorType>();
  if (!operand_type) return failure();

  auto slice_sizes = adaptor.slice_sizes();
  Type element_ty = operand_type.getElementType();
  inferredReturnShapes.emplace_back(slice_sizes.getValues<int64_t>(),
                                    element_ty);
  return success();
}

//===----------------------------------------------------------------------===//
// RealDynamicSliceOp
//===----------------------------------------------------------------------===//
// Verifies that operand rank matches start_indices/limit_indices/strides size
LogicalResult RealDynamicSliceOp::verify() {
  auto input_type = operand().getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!input_type) return success();
  int input_rank = input_type.getRank();

  auto start_type = start_indices().getType().cast<RankedTensorType>();
  auto limit_type = limit_indices().getType().cast<RankedTensorType>();
  auto strides_type = strides().getType().cast<RankedTensorType>();

  if (input_rank != start_type.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << input_rank << ") and start_indices size ("
                         << start_type.getNumElements() << ")";
  }

  if (input_rank != limit_type.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << input_rank << ") and limit_indices size ("
                         << limit_type.getNumElements() << ")";
  }

  if (input_rank != strides_type.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << input_rank << ") and strides size ("
                         << strides_type.getNumElements() << ")";
  }

  return success();
}

namespace {
// Canonicalizes RealDynamicSlice ops that can be replaced instead with Slice
// ops. This canonicalization is applied the case when the `begin` input values
// are compile time constants and thus can be made into a tensor.
struct RealDynamicSliceIsStatic : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern<RealDynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RealDynamicSliceOp real_dynamic_slice,
                                PatternRewriter& rewriter) const override {
    Location loc = real_dynamic_slice.getLoc();
    Value input = real_dynamic_slice.operand();
    Value output = real_dynamic_slice.result();
    auto input_ty = input.getType().dyn_cast<RankedTensorType>();
    auto output_ty = output.getType().dyn_cast<RankedTensorType>();

    if (!input_ty || !output_ty || !input_ty.hasStaticShape() ||
        !output_ty.hasStaticShape()) {
      return failure();
    }

    int64_t input_rank = input_ty.getRank();

    auto start_val = real_dynamic_slice.start_indices();
    auto limit_val = real_dynamic_slice.limit_indices();
    auto stride_val = real_dynamic_slice.strides();
    auto start_op = start_val.getDefiningOp<mlir::arith::ConstantOp>();
    auto limit_op = limit_val.getDefiningOp<mlir::arith::ConstantOp>();
    auto stride_op = stride_val.getDefiningOp<mlir::arith::ConstantOp>();
    if (!start_op || !limit_op || !stride_op) return failure();

    auto start_attr =
        start_op.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    auto limit_attr =
        limit_op.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    auto stride_attr =
        stride_op.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    if (!start_attr || !limit_attr || !stride_attr) return failure();

    SmallVector<int64_t, 4> temp_start_indices;
    SmallVector<int64_t, 4> temp_limit_indices;
    SmallVector<int64_t, 4> temp_stride;
    for (int64_t dim_idx = 0; dim_idx < input_rank; dim_idx++) {
      int64_t start = start_attr.getValues<IntegerAttr>()[dim_idx].getInt();
      temp_start_indices.push_back(start);
      int64_t limit = limit_attr.getValues<IntegerAttr>()[dim_idx].getInt();
      temp_limit_indices.push_back(limit);
      int64_t end = stride_attr.getValues<IntegerAttr>()[dim_idx].getInt();
      temp_stride.push_back(end);
    }

    DenseIntElementsAttr slice_start_indices =
        rewriter.getI64TensorAttr(temp_start_indices);
    DenseIntElementsAttr slice_limit_indices =
        rewriter.getI64TensorAttr(temp_limit_indices);
    DenseIntElementsAttr slice_strides = rewriter.getI64TensorAttr(temp_stride);
    auto result = rewriter.create<SliceOp>(loc, input, slice_start_indices,
                                           slice_limit_indices, slice_strides);
    rewriter.replaceOp(real_dynamic_slice, {result});
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
  Value start_indices = adaptor.start_indices();
  Value limit_indices = adaptor.limit_indices();
  Value strides = adaptor.strides();

  auto operand_type = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operand_type) return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shape_values;
  shape_values.reserve(operand_type.getRank());
  Type shape_scalar_type =
      start_indices.getType().cast<ShapedType>().getElementType();
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  one = MaybeCastTo(builder, loc, one, shape_scalar_type);
  for (const auto& element : llvm::enumerate(operand_type.getShape())) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, element.index());
    Value value_start =
        builder.create<tensor::ExtractOp>(loc, start_indices, offset);
    Value value_limit =
        builder.create<tensor::ExtractOp>(loc, limit_indices, offset);
    Value value_stride =
        builder.create<tensor::ExtractOp>(loc, strides, offset);
    // size = (limit - start + stride - 1) / stride
    shape_values.push_back(builder.create<arith::DivSIOp>(
        loc,
        builder.create<arith::SubIOp>(
            loc,
            builder.create<arith::AddIOp>(
                loc, value_stride,
                builder.create<arith::SubIOp>(loc, value_limit, value_start)),
            one),
        value_stride));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shape_values.size())},
                            shape_scalar_type),
      shape_values));
  return success();
}

//===----------------------------------------------------------------------===//
// InfeedOp
//===----------------------------------------------------------------------===//

// Checks that the result type is of the form `zero_or_more_type(s),
// mhlo::token`
LogicalResult InfeedOp::verify() {
  auto result_types = getResultTypes();
  if (result_types.empty())
    return emitOpError()
           << "result is expected to be at least of size 1, but got "
           << result_types.size();

  if (!result_types[result_types.size() - 1].isa<TokenType>())
    return emitOpError() << "last element of result types is expected to "
                            "be of token type, but got "
                         << result_types[result_types.size() - 1];

  // Verify layout attribute
  constexpr char kLayoutAttr[] = "layout";
  if (!getOperation()->hasAttr(kLayoutAttr)) return success();

  mlir::ArrayAttr layout =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(kLayoutAttr);
  if (!layout)
    return emitOpError() << "layout-attribute expected to be of array-type.";

  if (layout.size() != result_types.size() - 1) {
    return emitOpError() << "layout-attribute size must be "
                         << result_types.size() - 1
                         << " (which is the number of "
                            "op-results - 1 (for token result)), but got "
                         << layout.size();
  }

  for (auto child_layout : layout) {
    mlir::ArrayAttr child_layout_arr = child_layout.dyn_cast<mlir::ArrayAttr>();
    if (!child_layout_arr) {
      return emitOpError() << "layout-attribute expected to have "
                              "elements of type array, but got "
                           << child_layout;
    }

    for (auto i : child_layout_arr) {
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
  auto& computation_block = computation().front();
  auto computation_args = computation_block.getArguments();
  if (operands().size() != computation_args.size())
    return emitOpError() << "expects number of operands to match the arity "
                            "of map computation, but got: "
                         << operands().size() << " and "
                         << computation_args.size();

  // The parameters of computation should all be scalars and match the element
  // type of operands.
  auto operand_type = operands()[0].getType().cast<TensorType>();
  auto operand_elem_ty = operand_type.getElementType();

  for (const auto& indexed_arg : llvm::enumerate(computation_args)) {
    auto arg_type = indexed_arg.value().getType().dyn_cast<TensorType>();
    if (!arg_type || arg_type.getRank() != 0)
      return emitOpError()
             << "computation arguments must be 0-rank tensor, but got: arg #"
             << indexed_arg.index() << " of type "
             << indexed_arg.value().getType();
    if (arg_type.getElementType() != operand_elem_ty) {
      return emitOpError()
             << "element type of operands and computation arguments must "
                "match, but got: "
             << operand_elem_ty << " and " << arg_type.getElementType();
    }
  }

  // Mapped computation must return single output
  auto computation_outputs = computation_block.getTerminator()->getOperands();
  if (computation_outputs.size() != 1)
    return emitOpError() << "computation must return single output, but got: "
                         << computation_outputs.size();

  // The output of computation must be scalar and have the same element type
  // as op result.
  auto computation_output_type =
      computation_outputs[0].getType().dyn_cast<TensorType>();
  if (!computation_output_type || computation_output_type.getRank() != 0)
    return emitOpError() << "computation must return 0-rank tensor, but got: "
                         << computation_outputs[0].getType();

  auto result_type = getType().cast<TensorType>();
  if (computation_output_type.getElementType() != result_type.getElementType())
    return emitOpError() << "element type of result and computation output "
                            "must match, but got: "
                         << result_type.getElementType() << " and "
                         << computation_output_type.getElementType();

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
  if (operand_type.hasRank()) {
    if (dimensions.size() != operand_type.getShape().size())
      return emitOpError()
             << "applied to a subset of dimensions currently not supported: "
                "operand dimensions = "
             << operand_type.getShape().size()
             << ", requested map dimensions size = " << dimensions.size();
  }

  return success();
}

OpFoldResult MapOp::fold(ArrayRef<Attribute> operands) {
  mlir::Block& bb = computation().front();
  mlir::Operation& front_op = bb.front();

  auto ret_op = mlir::dyn_cast<ReturnOp>(front_op);
  if (!ret_op) return nullptr;
  if (ret_op.results().size() != 1) return nullptr;

  for (mlir::BlockArgument barg : bb.getArguments()) {
    if (barg == ret_op.results()[0]) return getOperands()[barg.getArgNumber()];
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

// Checks that the result type is of the form `zero_or_more_type(s),
// mhlo::token`
LogicalResult RecvOp::verify() {
  auto result_types = getResultTypes();
  if (result_types.empty())
    return emitOpError()
           << "result is expected to be at least of size 1, but got "
           << result_types.size();
  if (!result_types[result_types.size() - 1].isa<TokenType>())
    return emitOpError() << "last element of result types is expected to "
                            "be of token type, but got "
                         << result_types[result_types.size() - 1];
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
    ArrayRef<TensorType> input_types, ArrayRef<TensorType> init_types,
    const ArrayRef<WindowDimension> window) {
  SmallVector<TensorType> output_types;
  for (size_t i = 0; i < input_types.size(); ++i) {
    if (!input_types[i].hasRank()) {
      output_types.push_back(
          UnrankedTensorType::get(init_types[i].getElementType()));
      continue;
    }

    output_types.push_back(RankedTensorType::get(
        inferWindowOutputShape(input_types[i].getShape(), window),
        init_types[i].getElementType()));
  }

  return output_types;
}
}  // namespace

// We intend to verify the following properties
//  P1. The sizes of 'inputs' and 'init_values' must be atleast 1.
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
  int64_t num_inputs = getNumOperands() / 2;
  auto operand_tensor_types = llvm::to_vector<4>(llvm::map_range(
      getOperandTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); }));
  ArrayRef<TensorType> input_types(operand_tensor_types.begin(),
                                   operand_tensor_types.begin() + num_inputs);
  ArrayRef<TensorType> init_types(operand_tensor_types.begin() + num_inputs,
                                  operand_tensor_types.end());

  // P2.
  if (failed(verifyCompatibleShapes(operands().getTypes())))
    return emitOpError() << "requires same shape for all inputs";

  // P3.
  SmallVector<int64_t> window_dims =
      convertDenseIntAttr(this->window_dimensions());
  for (const auto input_type : input_types) {
    if (!input_type.hasRank()) continue;
    if (input_type.getRank() != window_dims.size())
      return emitOpError()
             << "expects window-dimensions size == input rank, but got "
                "window-dimensions size: "
             << window_dims.size() << " and input: " << input_type
             << " with rank = " << input_type.getRank() << ".";
  }

  // P4.
  auto padding_or_err = convertNx2Attribute(this->padding(), getLoc());
  if (failed(padding_or_err)) return failure();
  SmallVector<std::pair<int64_t, int64_t>> padding = *padding_or_err;

  auto window_or_err = verifyWindowAttributesAndInferWindowDimensions(
      window_dims, convertDenseIntAttr(window_strides()), padding,
      /*lhs_dilation=*/convertDenseIntAttr(base_dilations()),
      /*rhs_dilation=*/convertDenseIntAttr(this->window_dilations()), getLoc());
  if (failed(window_or_err)) return failure();

  // P5.
  bool all_inputs_unranked =
      llvm::all_of(input_types, [](TensorType t) { return !t.hasRank(); });

  Block& block = body().front();
  SmallVector<TensorType> accumulator_subshapes;
  if (failed(verifyReducerShape(this->getLoc(), block, input_types, init_types,
                                num_inputs, window_dims, all_inputs_unranked,
                                accumulator_subshapes)))
    return failure();

  // P6.
  if (num_inputs != getNumResults())
    return emitOpError() << "expects " << num_inputs
                         << " result values, but got " << getNumResults()
                         << ".";

  // The result-type is enforced as "TensorType" by ODS.
  auto result_tensor_types = llvm::to_vector<4>(llvm::map_range(
      getResultTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); }));

  // Check if the element-type of results match with the ones derived from
  // the reducer-block. Already ensured that  |accumulator_subshapes| ==
  // num_inputs == num_of_results.
  for (int64_t shape_idx = 0; shape_idx < accumulator_subshapes.size();
       shape_idx++) {
    if (accumulator_subshapes[shape_idx].getElementType() !=
        result_tensor_types[shape_idx].getElementType()) {
      return emitError()
             << "expects the element-type of reduce-op's return-value at index "
             << shape_idx
             << " to match the element-type of reducer-block's "
                "corresponding return-value, but got "
             << result_tensor_types[shape_idx].getElementType() << " and "
             << accumulator_subshapes[shape_idx].getElementType() << " resp.";
    }
  }

  // Check if the shape of results match with the ones derived from
  // the input-types and wndow-attributes.
  auto inferred_return_types = inferReduceWindowOpReturnType(
      input_types, accumulator_subshapes, *window_or_err);

  for (size_t i = 0; i < getNumResults(); i++) {
    if (failed(verifyCompatibleShape(result_tensor_types[i],
                                     inferred_return_types[i]))) {
      return emitOpError()
             << "expects result at index " << i
             << " to have compatible shape with the corresponding "
                "inferred type, but got "
             << result_tensor_types[i] << " and " << inferred_return_types[i]
             << " resp.";
    }
  }

  return success();
}

// Get the operation used for reduction applied to `result_index`th result. Its
// expected to be a binary operation that consumes `result_index`th and
// `result_index + operands().size`th arguments of the body.
Operation* ReduceWindowOp::getReductionOp(int result_index) {
  auto return_op = cast<ReturnOp>(body().front().getTerminator());
  Operation* compute_op = return_op.results()[result_index].getDefiningOp();
  if (compute_op->getNumOperands() != 2) return nullptr;
  auto arg0 = compute_op->getOperand(0).dyn_cast<BlockArgument>();
  auto arg1 = compute_op->getOperand(1).dyn_cast<BlockArgument>();
  if (!arg0 || !arg1) return nullptr;
  int arg0_num = arg0.getArgNumber();
  int arg1_num = arg1.getArgNumber();
  size_t other_arg_index = result_index + operands().size();
  if (arg0_num == result_index && arg1_num == other_arg_index)
    return compute_op;
  if (arg0_num == other_arg_index && arg1_num == result_index &&
      compute_op->hasTrait<mlir::OpTrait::IsCommutative>())
    return compute_op;
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

  llvm::SmallVector<APInt, 5> new_dims;
  new_dims.reserve(dimensions().getNumElements());

  auto shaped_type = input.getType().cast<ShapedType>();
  for (auto dim : dimensions().getValues<APInt>()) {
    if (shaped_type.getDimSize(dim.getLimitedValue()) != 1) {
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
static TensorType GetReduceResultType(Type operand_ty,
                                      DenseIntElementsAttr dimensions,
                                      Builder* builder) {
  Type element_ty = getElementTypeOrSelf(operand_ty);

  auto ranked_ty = operand_ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return UnrankedTensorType::get(element_ty);

  int64_t rank = ranked_ty.getRank();
  llvm::SmallVector<bool, 4> dims_mask(rank, false);
  for (int64_t dim : dimensions.getValues<int64_t>()) dims_mask[dim] = true;

  SmallVector<int64_t, 4> shape;
  for (int64_t i = 0; i < rank; ++i) {
    if (!dims_mask[i]) shape.push_back(ranked_ty.getDimSize(i));
  }

  return RankedTensorType::get(shape, element_ty);
}

void ReduceOp::build(OpBuilder& builder, OperationState& state,
                     ValueRange operands, ValueRange init_values,
                     DenseIntElementsAttr dimensions) {
  SmallVector<Type, 1> result_ty;
  result_ty.reserve(operands.size());

  for (Value operand : operands) {
    result_ty.push_back(
        GetReduceResultType(operand.getType(), dimensions, &builder));
  }
  build(builder, state, result_ty, operands, init_values, dimensions);
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
  SmallVector<Value> replaced_results;
  if (auto ret_op = mlir::dyn_cast<ReturnOp>(bb.back())) {
    for (Value result : ret_op.results()) {
      if (result.getParentRegion() == ret_op->getParentRegion())
        return failure();
      replaced_results.push_back(result);
    }

    results.insert(results.end(), replaced_results.begin(),
                   replaced_results.end());
    return success();
  }

  return failure();
}

bool HasSameOperandAndResultTypes(Operation& op) {
  Type expected;
  if (op.getNumResults() != 0) expected = op.getResult(0).getType();
  if (op.getNumOperands() != 0) expected = op.getOperand(0).getType();
  if (!expected) return false;

  auto type_match = [&](Type actual) { return actual == expected; };
  return llvm::all_of(op.getOperandTypes(), type_match) &&
         llvm::all_of(op.getResultTypes(), type_match);
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
      !HasSameOperandAndResultTypes(innerOp) ||
      !innerOp.hasTrait<mlir::OpTrait::IsCommutative>() ||
      !innerOp.hasTrait<mlir::OpTrait::ZeroRegion>())
    return false;

  // Check E3.
  if (op.operands().empty()) return false;

  auto elem_type =
      op.operands()[0].getType().cast<TensorType>().getElementType();
  auto expected_inner_op_type = RankedTensorType::get(/*shape=*/{}, elem_type);
  if (innerOp.getOperands()[0].getType() != expected_inner_op_type)
    return false;

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
    parser.parseOptionalComma();
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
      !innerOpNameInfo->hasTrait<mlir::OpTrait::ZeroRegion>()) {
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
    SmallVector<DenseElementsAttr, 4> barg_cst_attrs;
    for (auto inp_and_barg : llvm::zip(op.getOperands(), bb.getArguments())) {
      Value inp = std::get<0>(inp_and_barg);
      BlockArgument barg = std::get<1>(inp_and_barg);
      ConstOp cst = inp.getDefiningOp<ConstOp>();
      if (!cst) return failure();

      auto cst_attr = cst.value().dyn_cast_or_null<DenseElementsAttr>();
      if (!cst_attr.isSplat()) {
        return rewriter.notifyMatchFailure(op, "Must be splat constant.");
      }

      auto barg_shaped_type = barg.getType().dyn_cast<ShapedType>();
      if (!barg_shaped_type) return failure();

      auto barg_cst_attr = DenseElementsAttr::get(
          barg_shaped_type, cst_attr.getSplatValue<mlir::Attribute>());
      barg_cst_attrs.push_back(barg_cst_attr);
    }

    // Create new splat constants to replace block arguments.
    for (BlockArgument barg : bb.getArguments()) {
      int arg_idx = barg.getArgNumber();
      mhlo::ConstOp new_cst = rewriter.create<mhlo::ConstOp>(
          bb.front().getLoc(), barg.getType(), barg_cst_attrs[arg_idx]);
      barg.replaceAllUsesWith(new_cst);
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

  auto operand_type = inputs[0].getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operand_type) return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shape_values;
  SmallVector<int64_t, 4> dimensions(this->dimensions().getValues<int64_t>());
  shape_values.reserve(operand_type.getRank());
  Type shape_scalar_type = builder.getIndexType();
  auto to_shape_scalar_type = [&](Value v) {
    return MaybeCastTo(builder, loc, v, shape_scalar_type);
  };

  for (const auto& element : llvm::enumerate(operand_type.getShape())) {
    int64_t idx = element.index();
    auto* it = std::find(dimensions.begin(), dimensions.end(), idx);
    if (it != dimensions.end()) {
      continue;
    }
    Value value_dim = to_shape_scalar_type(
        builder.create<tensor::DimOp>(loc, inputs[0], element.index()));
    shape_values.push_back(value_dim);
  }

  Value output_shape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shape_values.size())},
                            shape_scalar_type),
      shape_values);
  for (size_t i = 0; i < inputs.size(); ++i) {
    reifiedReturnShapes.push_back(output_shape);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RngBitGeneratorOp
//===----------------------------------------------------------------------===//

// Verify that input state has the same shape as output shape
LogicalResult RngBitGeneratorOp::verify() {
  auto initial_shape = initial_state().getType().dyn_cast<RankedTensorType>();
  auto output_shape = output_state().getType().dyn_cast<RankedTensorType>();
  if (initial_shape.getShape() != output_shape.getShape())
    return emitOpError()
           << "output state shape must match initial state shape. Got: "
           << initial_shape << " and " << output_shape;
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
  auto result_ty = getType().cast<RankedTensorType>();
  if (!result_ty) return emitOpError() << "Output is not ranked.";
  if (!result_ty.hasStaticShape())
    return emitOpError() << "Output is not statically shaped.";
  auto rank = result_ty.getRank();
  if (rank != 1)
    return emitOpError() << "Output is of rank " << rank << " instead of 1";
  auto extent = result_ty.getDimSize(0);
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
  auto true_type = op.on_true().getType().cast<TensorType>();
  auto false_type = op.on_true().getType().cast<TensorType>();

  // Check for type compatibility in the select op. This requires that the two
  // non-predicate operands:
  //   (a) have the same element type
  //   (b) have compatible shapes (i.e. the same shape and/or at least one
  //       dynamic shape)
  if (true_type.getElementType() != false_type.getElementType() ||
      failed(mlir::verifyCompatibleShape(true_type, false_type))) {
    return emitOptionalError(location,
                             "incompatible operand types: ", true_type, " and ",
                             false_type);
  }

  // The output shape should be the most general of the operand shapes at each
  // dimension.
  ShapedTypeComponents& output_type = inferredReturnShapes.emplace_back();
  if (true_type == false_type || !true_type.hasRank()) {
    output_type = ShapedTypeComponents(true_type.cast<ShapedType>());
  } else if (!false_type.hasRank()) {
    output_type = ShapedTypeComponents(false_type.cast<ShapedType>());
  } else {
    assert(true_type.getRank() == false_type.getRank());
    llvm::SmallVector<int64_t, 4> dims;
    dims.reserve(true_type.getRank());
    for (auto dim : llvm::zip(true_type.getShape(), false_type.getShape())) {
      dims.push_back(std::get<0>(dim) == std::get<1>(dim)
                         ? std::get<0>(dim)
                         : ShapedType::kDynamicSize);
    }
    output_type = ShapedTypeComponents(dims, true_type.getElementType());
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

  return VerifyDimAttr(*this);
}

OpFoldResult SetDimensionSizeOp::fold(ArrayRef<Attribute> operands) {
  DenseElementsAttr input = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  if (input) return input;

  DenseElementsAttr size = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  if (!size || !size.isSplat()) return {};

  auto ty = getType().dyn_cast<RankedTensorType>();
  if (!ty) return {};

  int64_t dim_size = ty.getDimSize(dimension());
  if (dim_size == size.getSplatValue<IntegerAttr>().getInt()) return operand();
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
  auto input_type = adaptor.operand().getType().cast<RankedTensorType>();
  auto pad_type = adaptor.padding_value().getType().cast<RankedTensorType>();

  if (pad_type.getRank() != 0) {
    return emitOptionalError(
        location, llvm::formatv("padding value type should be a rank-0 "
                                "tensor, is rank {0}",
                                pad_type.getRank()));
  }

  const auto& padding_low = adaptor.edge_padding_low();
  if (padding_low.getType().getNumElements() != input_type.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "edge_padding_low length ({0}) must match operand rank ({1})",
            padding_low.getType().getNumElements(), input_type.getRank()));
  }

  const auto& padding_high = adaptor.edge_padding_high();
  if (padding_high.getType().getNumElements() != input_type.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "edge_padding_high length ({0}) must match operand rank ({1})",
            padding_high.getType().getNumElements(), input_type.getRank()));
  }

  const auto& padding_interior = adaptor.interior_padding();
  if (padding_interior.getType().getNumElements() != input_type.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "interior_padding length ({0}) must match operand rank ({1})",
            padding_interior.getType().getNumElements(), input_type.getRank()));
  }

  auto input_shape = input_type.getShape();
  SmallVector<int64_t> result_shape;
  for (int i = 0, e = input_shape.size(); i < e; i++) {
    int64_t padding_low_val = padding_low.getValues<APInt>()[i].getSExtValue();
    int64_t padding_high_val =
        padding_high.getValues<APInt>()[i].getSExtValue();
    int64_t padding_interior_val =
        padding_interior.getValues<APInt>()[i].getSExtValue();
    if (padding_interior_val < 0) {
      return emitOptionalError(
          location, llvm::formatv("Interior padding cannot be negative: {0}",
                                  padding_interior_val));
    }
    int64_t expected_output =
        input_shape[i] + padding_low_val + padding_high_val +
        std::max<int64_t>(input_shape[i] - 1, 0LL) * padding_interior_val;
    if (expected_output < 0) {
      return emitOptionalError(
          location,
          llvm::formatv("Padding result in negative size for dimension {0}",
                        i));
    }
    result_shape.push_back(expected_output);
  }
  inferredReturnShapes.emplace_back(result_shape, input_type.getElementType());

  return success();
}

OpFoldResult PadOp::fold(ArrayRef<Attribute> operands) {
  // If all padding is zero then it is an identity pad.
  auto is_zero = [](const APInt& i) { return i == 0; };
  if (llvm::all_of(edge_padding_low().getValues<APInt>(), is_zero) &&
      llvm::all_of(edge_padding_high().getValues<APInt>(), is_zero) &&
      llvm::all_of(interior_padding().getValues<APInt>(), is_zero))
    return operand();

  // If any padding is negative then it isn't supported by the folder (yet).
  auto is_negative = [](const APInt& i) { return i.slt(0); };
  if (llvm::any_of(edge_padding_low().getValues<APInt>(), is_negative) ||
      llvm::any_of(edge_padding_high().getValues<APInt>(), is_negative) ||
      llvm::any_of(interior_padding().getValues<APInt>(), is_negative))
    return {};

  DenseElementsAttr input = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  DenseElementsAttr padding = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  RankedTensorType return_type = getType().dyn_cast_or_null<RankedTensorType>();
  if (!input || !input.getType().hasRank() || !padding || !return_type ||
      !return_type.hasStaticShape())
    return {};

  // Fill the full result tensor with the padding value.
  llvm::SmallVector<Attribute, 4> result(return_type.getNumElements(),
                                         padding.getValues<Attribute>()[0]);

  auto next_index = [](llvm::SmallVector<uint64_t, 8>& index,
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
  uint64_t num_elements = input.getNumElements();
  for (uint64_t operand_idx = 0; operand_idx < num_elements; operand_idx++) {
    uint64_t result_idx = 0;
    uint64_t idx_multiplyer = 1;
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      result_idx +=
          (edge_padding_low().getValues<int64_t>()[i] +
           index[i] * (interior_padding().getValues<int64_t>()[i] + 1)) *
          idx_multiplyer;
      idx_multiplyer *= return_type.getDimSize(i);
    }
    result[result_idx] = input.getValues<Attribute>()[index];
    next_index(index, input.getType().getShape());
  }
  return DenseElementsAttr::get(return_type, result);
}

//===----------------------------------------------------------------------===//
// DynamicPadOp
//===----------------------------------------------------------------------===//

void DynamicPadOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                               MLIRContext* context) {
  results.add<DPadToPad>(context);
}

LogicalResult DynamicPadOp::verify() {
  auto input_type = operand().getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!input_type) return success();
  int input_rank = input_type.getRank();

  auto pad_type = padding_value().getType().cast<RankedTensorType>();
  if (pad_type.getRank() != 0) {
    return emitOpError() << "padding value type should be a rank-0";
  }

  auto padding_low_type = edge_padding_low().getType().cast<RankedTensorType>();
  if (padding_low_type.getNumElements() != input_rank) {
    return emitOpError() << "edge_padding_low length("
                         << padding_low_type.getNumElements()
                         << ") must match operand rank(" << input_rank << ").";
  }

  auto padding_high_type =
      edge_padding_high().getType().cast<RankedTensorType>();
  if (padding_high_type.getNumElements() != input_rank) {
    return emitOpError() << "edge_padding_high length("
                         << padding_high_type.getNumElements()
                         << ") must match operand rank(" << input_rank << ").";
  }

  auto interior_padding_type =
      interior_padding().getType().cast<RankedTensorType>();
  if (interior_padding_type.getNumElements() != input_rank) {
    return emitOpError() << "edge_padding_interior length("
                         << interior_padding_type.getNumElements()
                         << ") must match operand rank(" << input_rank << ").";
  }

  auto output_type = getResult().getType().dyn_cast<RankedTensorType>();
  // If result is unranked, there is very little to verify statically.
  if (!output_type) return success();
  int output_rank = output_type.getRank();
  if (input_rank != output_rank) {
    return emitOpError() << "operand rank(" << input_rank
                         << ") must match result(" << output_rank << ").";
  }

  return success();
}

LogicalResult DynamicPadOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicPadOp::Adaptor adaptor(operands);
  Value operand = adaptor.operand();
  Value edge_padding_low = adaptor.edge_padding_low();
  Value edge_padding_high = adaptor.edge_padding_high();
  Value interior_padding = adaptor.interior_padding();

  auto operand_type = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked pad a.t.m.
  if (!operand_type) return failure();

  auto loc = this->getLoc();
  SmallVector<Value, 4> shape_values;
  shape_values.reserve(operand_type.getRank());
  Type shape_scalar_type =
      edge_padding_low.getType().cast<ShapedType>().getElementType();

  auto to_shape_scalar_type = [&](Value v) {
    return MaybeCastTo(builder, loc, v, shape_scalar_type);
  };

  Value zero =
      to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(loc, 0));
  Value one =
      to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(loc, 1));

  for (int idx : llvm::seq<int>(0, operand_type.getShape().size())) {
    Value value_dim =
        to_shape_scalar_type(builder.create<tensor::DimOp>(loc, operand, idx));
    Value offset = builder.create<arith::ConstantIndexOp>(loc, idx);
    Value value_low =
        builder.create<tensor::ExtractOp>(loc, edge_padding_low, offset);
    Value value_high =
        builder.create<tensor::ExtractOp>(loc, edge_padding_high, offset);
    Value value_interior =
        builder.create<tensor::ExtractOp>(loc, interior_padding, offset);
    // output_size = input_size + padding_low + padding_high + interior *
    // max(input_size - 1, 0)
    Value value_dim_less_than_one = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, value_dim, one);
    Value interior_size = builder.create<arith::MulIOp>(
        loc, value_interior,
        builder.create<mlir::arith::SelectOp>(
            loc, value_dim_less_than_one, zero,
            builder.create<arith::SubIOp>(loc, value_dim, one)));
    shape_values.push_back(builder.create<arith::AddIOp>(
        loc,
        builder.create<arith::AddIOp>(
            loc, builder.create<arith::AddIOp>(loc, interior_size, value_dim),
            value_low),
        value_high));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shape_values.size())},
                            shape_scalar_type),
      shape_values));

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  // If the operand type is dynamically shaped there is nothing to verify.
  auto operand_ty = operand().getType().dyn_cast<RankedTensorType>();
  if (!operand_ty || !operand_ty.hasStaticShape()) return success();

  // If the operand type is statically shaped (not required) the number of
  // elements must match that of the result type.
  auto result_ty = getType().cast<RankedTensorType>();
  assert(result_ty && result_ty.hasStaticShape() &&
         "result type must be statically shaped");
  int64_t num_result_elements = result_ty.getNumElements();
  int64_t num_operand_elements = operand_ty.getNumElements();
  if (num_result_elements != num_operand_elements)
    return emitOpError() << "number of output elements (" << num_result_elements
                         << ") doesn't match expected number of elements ("
                         << num_operand_elements << ")";

  return success();
}

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  if (getOperand().getType() == getType()) {
    return getOperand();
  }

  if (auto prev_op = getOperand().getDefiningOp<ReshapeOp>()) {
    setOperand(prev_op.getOperand());
    return getResult();
  }

  if (auto elements = operands.front().dyn_cast_or_null<DenseElementsAttr>()) {
    return elements.reshape(getResult().getType().cast<ShapedType>());
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

static LogicalResult VerifyConditionalBranch(Operation* op, Region& region,
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
  if (failed(VerifyConditionalBranch(*this, true_branch(),
                                     /*branchName=*/"true_branch"))) {
    return failure();
  }

  if (failed(VerifyConditionalBranch(*this, false_branch(),
                                     /*branchName=*/"false_branch"))) {
    return failure();
  }
  return success();
}

static LogicalResult InlineIfConstantCondition(IfOp ifOp,
                                               PatternRewriter& rewriter) {
  DenseIntElementsAttr pred_attr;
  if (!matchPattern(ifOp.pred(), m_Constant(&pred_attr))) return failure();

  if (pred_attr.getSplatValue<BoolAttr>().getValue()) {
    ReplaceOpWithRegion(rewriter, ifOp, ifOp.true_branch());
  } else {
    ReplaceOpWithRegion(rewriter, ifOp, ifOp.false_branch());
  }
  return success();
}

void IfOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add(&InlineIfConstantCondition);
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

LogicalResult CaseOp::verify() {
  auto num_branches = branches().size();

  for (unsigned i = 0; i < num_branches; ++i)
    if (failed(VerifyConditionalBranch(*this, branches()[i],
                                       /*branchName=*/"branch " + Twine(i))))
      return failure();

  return success();
}

static LogicalResult InlineCaseConstantCondition(CaseOp caseOp,
                                                 PatternRewriter& rewriter) {
  DenseIntElementsAttr index_attr;
  if (!matchPattern(caseOp.index(), m_Constant(&index_attr))) {
    return failure();
  }
  int64_t index =
      index_attr.getSplatValue<IntegerAttr>().getValue().getSExtValue();
  // For an OOB index, the last branch is executed as the default branch:
  // https://www.tensorflow.org/xla/operation_semantics#conditional
  if (index < 0 || index >= caseOp.getNumRegions())
    index = caseOp.getNumRegions() - 1;

  Region& region = caseOp.getRegion(index);
  if (!llvm::hasSingleElement(region)) return failure();
  ReplaceOpWithRegion(rewriter, caseOp, region);
  return success();
}

void CaseOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add(&InlineCaseConstantCondition);
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult SqrtOp::fold(ArrayRef<Attribute> operands) {
  auto val = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  if (!val) return {};

  auto type = getElementTypeOrSelf(getType());
  if (!type.isF32() && !type.isF64()) return {};

  auto shaped_type = getType().cast<ShapedType>();
  if (!shaped_type.hasStaticShape()) return {};

  int bit_width = type.getIntOrFloatBitWidth();
  llvm::SmallVector<APFloat, 4> values;
  values.reserve(val.getNumElements());
  for (auto it : val.getValues<APFloat>()) {
    double value = bit_width == 32 ? it.convertToFloat() : it.convertToDouble();
    if (value < 0) return {};
    value = std::sqrt(value);
    if (bit_width == 32)
      values.emplace_back(static_cast<float>(value));
    else
      values.emplace_back(value);
  }
  return DenseFPElementsAttr::get(shaped_type, values);
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

struct round {
  APFloat operator()(const APFloat& f) {
    APFloat r = f;
    r.roundToIntegral(llvm::RoundingMode::NearestTiesToAway);
    return r;
  }
};

struct logical_not {
  APInt operator()(const APInt& i) {
    return APInt(i.getBitWidth(), static_cast<uint64_t>(!i));
  }
};

template <typename FloatOrInt>
struct sign {
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
UNARY_FOLDER(SignOp, sign);
UNARY_FOLDER_INT(NotOp, logical_not);
UNARY_FOLDER_FLOAT(RoundOp, round);

#undef UNARY_FOLDER
#undef UNARY_FOLDER_INT
#undef UNARY_FOLDER_FLOAT

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {

// Updates the element type of a (presumed) tensor type 'x', returning either
// a permuted UnrankedTensorType or RankedTensorType.
static Type UpdateResultElementType(Builder* builder, Type x,
                                    Type element_type) {
  auto x_ranked = x.dyn_cast<RankedTensorType>();
  if (!x_ranked) {
    return UnrankedTensorType::get(element_type);
  }

  auto shape_x = x_ranked.getShape();
  return RankedTensorType::get(shape_x, element_type);
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
  SplatElementsAttr splat_lhs = lhs.dyn_cast<SplatElementsAttr>();
  SplatElementsAttr splat_rhs = rhs.dyn_cast<SplatElementsAttr>();
  if (splat_lhs && splat_rhs) {
    return SplatElementsAttr::get(
        type, Convert()(splat_lhs.getSplatValue<ValType>(),
                        splat_rhs.getSplatValue<ValType>()));
  }

  // Prevent folding if lhs/rhs are too large.
  if (lhs.getNumElements() > kFoldBinaryOpEltLimit) {
    return {};
  }

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<ValType>(), rhs.getValues<ValType>())) {
    values.push_back(Convert()(std::get<0>(zip), std::get<1>(zip)));
  }

  return DenseElementsAttr::get(type, values);
}

template <typename T>
struct divide : std::divides<T> {};

template <>
struct divide<APInt> {
  APInt operator()(const APInt& a, const APInt& b) const { return a.sdiv(b); }
};

template <typename T>
struct remainder : std::modulus<T> {};

template <>
struct remainder<APInt> {
  APInt operator()(const APInt& a, const APInt& b) const { return a.srem(b); }
};

template <>
struct remainder<APFloat> {
  APFloat operator()(const APFloat& a, const APFloat& b) const {
    APFloat result(a);
    result.remainder(b);
    return result;
  }
};

template <typename T>
struct max {
  T operator()(const T& a, const T& b) const { return std::max<T>(a, b); }
};

template <>
struct max<APInt> {
  APInt operator()(const APInt& a, const APInt& b) const {
    return llvm::APIntOps::smax(a, b);
  }
};

template <typename T>
struct min {
  T operator()(const T& a, const T& b) const { return std::min<T>(a, b); }
};

template <>
struct min<APInt> {
  APInt operator()(const APInt& a, const APInt& b) const {
    return llvm::APIntOps::smin(a, b);
  }
};

#define BINARY_FOLDER_INTERNAL(Op, Func)                                     \
  if (getElementTypeOrSelf(getType()).isa<FloatType>())                      \
    return BinaryFolder<Op, FloatType, APFloat, Func<APFloat>>(this, attrs); \
  if (getElementTypeOrSelf(getType()).isa<IntegerType>())                    \
    return BinaryFolder<Op, IntegerType, APInt, Func<APInt>>(this, attrs);   \
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
BINARY_FOLDER(DivOp, divide);
BINARY_FOLDER(RemOp, remainder);
BINARY_FOLDER(MaxOp, max);
BINARY_FOLDER(MinOp, min);

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
    SplatElementsAttr splat_lhs =
        attrs[0].dyn_cast_or_null<SplatElementsAttr>();
    SplatElementsAttr splat_rhs =
        attrs[1].dyn_cast_or_null<SplatElementsAttr>();
    if (isSplatZero(splat_lhs))
      return splat_rhs ? (OpFoldResult)splat_rhs : rhs();
    if (isSplatZero(splat_rhs))
      return splat_lhs ? (OpFoldResult)splat_lhs : lhs();
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
    SplatElementsAttr splat_lhs =
        attrs[0].dyn_cast_or_null<SplatElementsAttr>();
    SplatElementsAttr splat_rhs =
        attrs[1].dyn_cast_or_null<SplatElementsAttr>();
    if (isSplatOne(splat_lhs))
      return splat_rhs ? (OpFoldResult)splat_rhs : rhs();
    if (isSplatOne(splat_rhs))
      return splat_lhs ? (OpFoldResult)splat_lhs : lhs();
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
static int64_t InferSliceDim(int64_t input_dim, int64_t start, int64_t end,
                             int64_t stride) {
  if (input_dim == -1 || start < 0 || start > end || end > input_dim ||
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
  RankedTensorType ranked_ty = ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) {
    // The operand type is unranked, so the best we can infer for the result
    // type is an unranked tensor with the same element type as the operand
    // type.
    inferredReturnTypes.assign({ty});
    return success();
  }

  ShapedType attr_ty = slice.start_indices().getType();
  if (attr_ty.getRank() != 1) {
    return emitOptionalError(location, "start_indices has rank ",
                             attr_ty.getRank(), " instead of required rank 1");
  }

  int64_t rank = ranked_ty.getRank();
  if (attr_ty.getNumElements() != rank) {
    return emitOptionalError(
        location, "the number of elements in start_indices (",
        attr_ty.getNumElements(), ") does not match the rank of the operand (",
        rank, ")");
  }

  if (!attr_ty.getElementType().isSignlessInteger(64) ||
      slice.limit_indices().getType() != attr_ty ||
      slice.strides().getType() != attr_ty) {
    // Unfortunately we can't rely on the AllTypesMatch trait for the SliceOp
    // having been verified at this point. Emit an error message that matches
    // the one that would be reported by AllTypesMatch for a more consistent
    // user experience.
    // TODO(b/171567182): Clean this up after AllTypesMatch has been refactored.
    return emitOptionalError(location,
                             "failed to verify that all of {start_indices, "
                             "limit_indices, strides} have same type");
  }

  SmallVector<int64_t, 4> start(slice.start_indices().getValues<int64_t>());
  SmallVector<int64_t, 4> limit(slice.limit_indices().getValues<int64_t>());
  SmallVector<int64_t, 4> stride_vals(slice.strides().getValues<int64_t>());

  SmallVector<int64_t, 4> shape;
  shape.reserve(rank);
  for (int64_t i = 0, e = rank; i != e; i++) {
    shape.push_back(InferSliceDim(ranked_ty.getDimSize(i), start[i], limit[i],
                                  stride_vals[i]));
  }
  inferredReturnTypes.assign(
      {RankedTensorType::get(shape, ranked_ty.getElementType())});
  return success();
}

template <typename I, typename E>
static void SliceElements(I values, ArrayRef<int64_t> sizes,
                          ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
                          ArrayRef<int64_t> strides,
                          llvm::SmallVectorImpl<E>* out_values) {
  assert(starts.size() == limits.size());
  assert(starts.size() == strides.size());
  if (starts.empty()) return;

  int64_t start = starts.front();
  int64_t limit = limits.front();
  int64_t stride = strides.front();
  if (starts.size() == 1) {
    for (int i = start; i < limit; i += stride) {
      out_values->push_back(*(values + i));
    }
    return;
  }

  for (; start < limit; start += stride) {
    auto begin = values + start * sizes.front();
    SliceElements<I, E>(begin, sizes.drop_front(), starts.drop_front(),
                        limits.drop_front(), strides.drop_front(), out_values);
  }
}

template <typename I, typename E>
static Attribute FoldSlice(SliceOp* op, I values) {
  auto start = llvm::to_vector<6>(op->start_indices().getValues<int64_t>());
  auto limit = llvm::to_vector<6>(op->limit_indices().getValues<int64_t>());
  auto stride = llvm::to_vector<6>(op->strides().getValues<int64_t>());

  auto result_type = op->operand().getType().cast<ShapedType>();
  if (!result_type.hasStaticShape()) return {};

  auto shape = result_type.getShape();
  int64_t count = result_type.getNumElements();
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

  llvm::SmallVector<E, 6> out_values;
  out_values.reserve(result_type.getNumElements());
  SliceElements<I, E>(values, sizes, start, limit, stride, &out_values);

  return DenseElementsAttr::get(op->getResult().getType().cast<ShapedType>(),
                                out_values);
}

OpFoldResult SliceOp::fold(ArrayRef<Attribute> operands) {
  // Check if the SliceOp is a NoOp operation.
  auto operand_type = getOperand().getType().cast<ShapedType>();
  auto result_type = getResult().getType().cast<ShapedType>();

  if (operand_type.hasStaticShape() && result_type.hasStaticShape() &&
      (operand_type.getShape() == result_type.getShape())) {
    return getOperand();
  }

  if (operands.empty() || !operands.front()) return {};

  // Evaluate for statically valued inputs.
  DenseElementsAttr elements = operands.front().dyn_cast<DenseElementsAttr>();
  if (!elements) return {};

  auto etype = elements.getType().getElementType();
  if (etype.isa<IntegerType>()) {
    return FoldSlice<DenseElementsAttr::IntElementIterator, APInt>(
        this, elements.value_begin<APInt>());
  }
  if (etype.isa<FloatType>()) {
    return FoldSlice<DenseElementsAttr::FloatElementIterator, APFloat>(
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
    auto result_ty = slice.getType().cast<ShapedType>();
    if (!result_ty.hasStaticShape()) {
      return failure();
    }

    auto slice_input = slice.operand();
    auto slice_input_ty = slice_input.getType().cast<ShapedType>();
    auto concat = slice_input.getDefiningOp<ConcatenateOp>();
    if (!concat) {
      return failure();
    }

    auto dimension = concat.dimension();

    auto start = slice.start_indices().getValues<APInt>();
    auto limit = slice.limit_indices().getValues<APInt>();

    auto slice_start = (*(start.begin() + dimension)).getSExtValue();
    auto slice_limit = (*(limit.begin() + dimension)).getSExtValue();

    // We need to determine what inputs from the concat affect the slice, and
    // how the bounds of the slice need to be updated for the minimally required
    // inputs.
    int64_t running_size = 0;
    int64_t front_offset = slice_input_ty.getShape()[dimension];

    auto subset_start = concat.operand_end();
    auto subset_end = concat.operand_end();
    for (auto it = concat.operand_begin(); it < concat.operand_end(); ++it) {
      auto input = *it;
      ShapedType input_ty = input.getType().cast<ShapedType>();
      if (input_ty.isDynamicDim(dimension)) {
        return failure();
      }
      auto dim_size = input_ty.getShape()[dimension];

      // If this position is in the slice its the start of the subset and we
      // need to update the start and limit values.
      if (running_size + dim_size > slice_start &&
          subset_start == concat.operand_end()) {
        subset_start = it;
        front_offset = running_size;
      }

      // Determine the last required offset.
      if (running_size < slice_limit) {
        subset_end = it + 1;
      }

      running_size += dim_size;
    }

    auto subset_size = subset_end - subset_start;
    // We need all inputs so no optimization.
    if (subset_size == concat.getNumOperands()) {
      return failure();
    }

    // If there's nothing to slice that means the output is an empty tensor and
    // there is dead code. We do nothing here and rely on other passes to clean
    // this up.
    if (subset_size == 0) {
      return failure();
    }

    if (subset_size > 1 && !concat.getResult().hasOneUse()) {
      return failure();
    }

    auto concat_range = OperandRange(subset_start, subset_end);
    auto new_concat = rewriter.create<ConcatenateOp>(
        concat.getLoc(), concat_range, concat.dimension());

    llvm::SmallVector<APInt, 6> new_start(start);
    llvm::SmallVector<APInt, 6> new_limit(limit);
    new_start[dimension] -= front_offset;
    new_limit[dimension] -= front_offset;

    auto attr_type = slice.start_indices().getType().cast<ShapedType>();
    auto create = rewriter.create<SliceOp>(
        slice.getLoc(), new_concat,
        DenseIntElementsAttr::get(attr_type, new_start),
        DenseIntElementsAttr::get(attr_type, new_limit), slice.strides());
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
                   ValueRange operands, int64_t dimension, bool is_stable) {
  state.addOperands(operands);
  state.addAttribute("dimension", builder.getI64IntegerAttr(dimension));
  state.addAttribute("is_stable", builder.getBoolAttr(is_stable));

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
    ArrayRef<int64_t> input_shape =
        (*operands.begin()).getType().cast<ShapedType>().getShape();

    if (llvm::any_of(llvm::drop_begin(operands, 1), [&](Value operand) {
          return operand.getType().cast<ShapedType>().getShape() != input_shape;
        }))
      return emitOpError("requires all inputs to have the same dimensions");

    int64_t rank = input_shape.size();
    int64_t cmp_dim = dimension();
    if (cmp_dim < -rank || cmp_dim >= rank)
      return emitOpError("dimension attribute value must be in range [-")
             << rank << ", " << rank << "), but found " << cmp_dim;
  }

  Block& block = comparator().front();
  size_t num_operands = getOperation()->getNumOperands();
  if (block.getNumArguments() != 2 * num_operands)
    return emitOpError("comparator block should have ")
           << 2 * num_operands << " arguments";

  for (const auto& indexed_operand : llvm::enumerate(operands)) {
    int index = indexed_operand.index();
    Type element_type =
        indexed_operand.value().getType().cast<ShapedType>().getElementType();
    Type tensor_type = RankedTensorType::get({}, element_type);
    for (int i : {2 * index, 2 * index + 1}) {
      Type arg_type = block.getArgument(i).getType();
      if (arg_type != tensor_type)
        return emitOpError("comparator block argument #")
               << i << " should be of type " << tensor_type << " but got "
               << arg_type;
    }
  }

  // Mapped computation must return single output.
  auto comparator_result = block.getTerminator()->getOperands();
  if (comparator_result.size() != 1)
    return emitOpError() << "comparator must return single output, but got: "
                         << comparator_result.size();

  // The output of computation must be 0-ranked tensor with element-type i1.
  auto comparator_result_type =
      comparator_result[0].getType().dyn_cast<RankedTensorType>();
  if (!comparator_result_type || comparator_result_type.getRank() != 0 ||
      !comparator_result_type.getElementType().isInteger(1))
    return emitOpError() << "comparator must return tensor<i1>, but got: "
                         << comparator_result[0].getType();

  // check number of return-values and their element-types.
  auto result_types = getResultTypes();
  if (result_types.size() != num_operands)
    return emitOpError() << "expects the number of results to be same as "
                            "number of operands. Got number of results = "
                         << result_types.size()
                         << " and number of operands = " << num_operands;

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
static LogicalResult SortDropEmptyUseArgs(SortOp op,
                                          PatternRewriter& rewriter) {
  DenseSet<unsigned> erased_args;
  unsigned num_operands = op.getNumOperands();
  for (unsigned i = 0; i < num_operands; ++i) {
    if (!op.getResult(i).use_empty()) continue;
    Block& block = op.comparator().front();
    if (!block.getArgument(i * 2).use_empty()) continue;
    if (!block.getArgument(i * 2 + 1).use_empty()) continue;
    erased_args.insert(i);
  }
  if (erased_args.empty()) return failure();

  SmallVector<Value> new_operands;
  SmallVector<unsigned> erased_block_args;
  for (const auto& en : llvm::enumerate(op.operands())) {
    if (erased_args.contains(en.index())) {
      erased_block_args.push_back(en.index() * 2);
      erased_block_args.push_back(en.index() * 2 + 1);
    } else {
      new_operands.push_back(en.value());
    }
  }

  auto new_op = rewriter.create<SortOp>(op.getLoc(), new_operands,
                                        op.dimension(), op.is_stable());
  Region& region = new_op.comparator();
  rewriter.inlineRegionBefore(op.comparator(), region, region.end());
  region.front().eraseArguments(erased_block_args);

  SmallVector<Value> results;
  for (unsigned i = 0, j = 0; i < num_operands; ++i) {
    if (erased_args.contains(i)) {
      results.push_back({});
    } else {
      results.push_back(new_op.getResult(j++));
    }
  }
  rewriter.replaceOp(op, results);

  return success();
}

/// Set the sorting dimension to the last dimension if it's not set and the rank
/// is known.
static LogicalResult SortOpInferDefaultDimension(SortOp op,
                                                 PatternRewriter& rewriter) {
  auto ty = op.getResultTypes()[0].dyn_cast<ShapedType>();
  if (!ty) {
    return failure();
  }
  if (op.dimension() != -1) {
    return failure();
  }

  IntegerAttr dim = rewriter.getI64IntegerAttr(ty.getRank() - 1);
  auto new_op = rewriter.create<SortOp>(op.getLoc(), op.getResultTypes(),
                                        op.operands(), dim, op.is_stableAttr());
  Region& region = new_op.comparator();
  rewriter.inlineRegionBefore(op.comparator(), region, region.end());
  rewriter.replaceOp(op, new_op.getResults());

  return success();
}

void SortOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* /*context*/) {
  results.add(SortDropEmptyUseArgs);
  results.add(SortOpInferDefaultDimension);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  for (const auto& it : llvm::enumerate(permutation().getValues<APInt>())) {
    if (it.index() != it.value()) {
      return {};
    }
  }
  return getOperand();
}

// transpose(transpose(X)) => transpose(X)
static LogicalResult EliminateRedundantTranspse(TransposeOp op,
                                                PatternRewriter& rewriter) {
  auto tranpose_operand = op.operand().getDefiningOp<TransposeOp>();
  if (!tranpose_operand) {
    return failure();
  }
  auto operand_permutation = tranpose_operand.permutation().getValues<APInt>();
  auto new_permutation =
      op.permutation()
          .mapValues(op.permutation().getElementType(),
                     [&operand_permutation](const APInt& index) -> APInt {
                       return operand_permutation[index.getSExtValue()];
                     })
          .cast<DenseIntElementsAttr>();
  rewriter.replaceOpWithNewOp<TransposeOp>(op, op.getResult().getType(),
                                           tranpose_operand.operand(),
                                           new_permutation);
  return success();
}

// transpose(broadcast_in_dim(X)) => broadcast_in_dim(X)
static LogicalResult EliminateBroadcastInDimTranspose(
    TransposeOp op, PatternRewriter& rewriter) {
  auto broadcast_in_dim_op = op.operand().getDefiningOp<BroadcastInDimOp>();
  if (!broadcast_in_dim_op) {
    return failure();
  }
  DenseIntElementsAttr broadcast_dimensions =
      broadcast_in_dim_op.broadcast_dimensions();
  DenseIntElementsAttr permutation = op.permutation();
  SmallVector<int64_t> new_broadcast_dimensions;
  for (auto dimension : broadcast_dimensions.getValues<int64_t>()) {
    int64_t index = 0;
    for (auto p : permutation.getValues<int64_t>()) {
      if (p == dimension) {
        new_broadcast_dimensions.push_back(index);
        break;
      }
      index++;
    }
  }
  rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
      op, op->getResultTypes(), broadcast_in_dim_op.operand(),
      rewriter.getI64TensorAttr(new_broadcast_dimensions));
  return success();
}

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* /*context*/) {
  results.add(EliminateRedundantTranspse);
  results.add(EliminateBroadcastInDimTranspose);
}

LogicalResult TransposeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  TransposeOp::Adaptor adaptor(operands);
  Value operand = adaptor.operand();

  auto operand_type = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operand_type) return failure();

  Location loc = this->getLoc();
  SmallVector<int64_t, 4> permutation(this->permutation().getValues<int64_t>());
  SmallVector<Value, 4> shape_values(permutation.size());

  Type shape_scalar_type = builder.getIndexType();
  auto to_shape_scalar_type = [&](Value v) {
    return MaybeCastTo(builder, loc, v, shape_scalar_type);
  };

  for (const auto& element : llvm::enumerate(operand_type.getShape())) {
    int64_t idx = element.index();
    auto* it = std::find(permutation.begin(), permutation.end(), idx);
    Value value_dim = to_shape_scalar_type(
        builder.createOrFold<tensor::DimOp>(loc, operand, element.index()));
    shape_values[std::distance(permutation.begin(), it)] = value_dim;
  }

  Value output_shape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shape_values.size())},
                            shape_scalar_type),
      shape_values);
  reifiedReturnShapes.push_back(output_shape);

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
  auto a_type = a().getType().dyn_cast<RankedTensorType>();

  // Skip verifier if a is unranked tensor.
  if (!a_type) return success();

  // Check that a should have rank >= 2
  auto a_rank = a_type.getRank();
  if (a_rank < 2)
    return emitOpError() << "operand 'a' must have rank >= 2, but got "
                         << a_type;

  // The two minor dimensions of a must have same size.
  if (a_type.getDimSize(a_rank - 2) != a_type.getDimSize(a_rank - 1))
    return emitOpError() << "two minor dimensions of operand 'a' must have "
                            "equal size, but got "
                         << a_type;

  auto b_type = b().getType().dyn_cast<RankedTensorType>();
  // If b is unranked skip remaining checks.
  if (!b_type) return success();

  // Check that a and b have same rank.
  auto b_rank = b_type.getRank();
  if (a_rank != b_rank)
    return emitOpError() << "operands must have equal rank, but got " << a_type
                         << " and " << b_type;

  // The shared dimension of a and b should match.
  if (a_type.getDimSize(a_rank - 1) !=
      b_type.getDimSize(b_rank - (left_side() ? 2 : 1)))
    return emitOpError() << "shared dimension of operands 'a' and 'b' does "
                            "not match, but got "
                         << a_type << " and " << b_type;

  // The leading batch dimensions of a and b must be equal.
  auto a_batch_dims = a_type.getShape().drop_back(2);
  auto b_batch_dims = b_type.getShape().drop_back(2);
  if (a_batch_dims != b_batch_dims)
    return emitOpError()
           << "leading batch dimensions of the operands must be same, but got "
           << a_type << " and " << b_type;

  // Result and argument b must have same shape.
  auto result_type = getType().dyn_cast<RankedTensorType>();
  if (!result_type) return success();
  if (result_type != b_type)
    return emitOpError()
           << "result and operand 'b' must have same shape, but got "
           << result_type << " and " << b_type;
  return success();
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleElementOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto tuple_type = operands[0].getType().dyn_cast<TupleType>();
  if (!tuple_type) return failure();

  auto index_attr = attributes.get("index").cast<IntegerAttr>();
  auto index = index_attr.getInt();
  if (index < 0 || index >= tuple_type.size()) return failure();

  inferredReturnTypes.push_back(tuple_type.getType(index));
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
                      Value rhs, ComparisonDirection comparison_direction,
                      ComparisonType compare_type) {
  build(
      builder, result, lhs, rhs,
      ComparisonDirectionAttr::get(builder.getContext(), comparison_direction),
      ComparisonTypeAttr::get(builder.getContext(), compare_type));
}

LogicalResult CompareOp::inferReturnTypeComponents(
    mlir::MLIRContext* ctx, llvm::Optional<mlir::Location>,
    ValueShapeRange operands, mlir::DictionaryAttr, mlir::RegionRange,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnTypes) {
  ShapedTypeComponents& components =
      inferredReturnTypes.emplace_back(IntegerType::get(ctx, /*width=*/1));
  auto arg_ty = operands.front().getType().cast<TensorType>();
  if (arg_ty.hasRank()) {
    components =
        ShapedTypeComponents(arg_ty.getShape(), components.getElementType());
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
struct less : std::less<T> {};

template <>
struct less<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.slt(b); }
};

template <typename T>
struct less_equal : std::less_equal<T> {};

template <>
struct less_equal<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.sle(b); }
};

template <typename T>
struct greater : std::greater<T> {};

template <>
struct greater<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.sgt(b); }
};

template <typename T>
struct greater_equal : std::greater_equal<T> {};

template <>
struct greater_equal<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.sge(b); }
};

template <typename Op, typename ElementType, typename SrcType, typename Convert>
static Attribute CompareFolder(CompareOp op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1]) return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs) return {};

  ShapedType operand_type =
      op.getOperand(0).getType().template cast<ShapedType>();
  if (!operand_type.hasStaticShape()) {
    return {};
  }

  if (!operand_type.getElementType().isa<ElementType>()) {
    return {};
  }

  SmallVector<bool, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<SrcType>(), rhs.getValues<SrcType>())) {
    values.push_back(Convert()(std::get<0>(zip), std::get<1>(zip)));
  }

  auto result_ty = op.getType().cast<ShapedType>();
  return DenseElementsAttr::get(result_ty, values);
}

OpFoldResult CompareOp::fold(ArrayRef<Attribute> operands) {
  auto result_ty = getType().cast<ShapedType>();
  if (!result_ty.hasStaticShape()) return {};

  auto direction = comparison_direction();
  auto lhs_ty = getElementTypeOrSelf(lhs());
  if (lhs() == rhs() && !lhs_ty.isa<FloatType>() &&
      (!lhs_ty.isa<ComplexType>() ||
       !lhs_ty.cast<ComplexType>().getElementType().isa<FloatType>())) {
    if (direction == ComparisonDirection::LE ||
        direction == ComparisonDirection::EQ ||
        direction == ComparisonDirection::GE) {
      return DenseIntElementsAttr::get(result_ty, {true});
    }
    return DenseIntElementsAttr::get(result_ty, {false});
  }

  auto op_el_type = lhs().getType().cast<ShapedType>().getElementType();
  // Fold tensor<*xi1> != false to just return tensor<*xi1>
  if (direction == ComparisonDirection::NE && op_el_type.isInteger(1)) {
    DenseIntElementsAttr cst_attr;
    if (matchPattern(lhs(), m_Constant(&cst_attr))) {
      if (cst_attr.isSplat() && !cst_attr.getSplatValue<bool>()) {
        return rhs();
      }
    }

    if (matchPattern(rhs(), m_Constant(&cst_attr))) {
      if (cst_attr.isSplat() && !cst_attr.getSplatValue<bool>()) {
        return lhs();
      }
    }
  }

  // Fold tensor<*xi1> == True to just return tensor<*xi1>
  if (direction == ComparisonDirection::EQ && op_el_type.isInteger(1)) {
    DenseIntElementsAttr cst_attr;
    if (matchPattern(lhs(), m_Constant(&cst_attr))) {
      if (cst_attr.isSplat() && cst_attr.getSplatValue<bool>()) {
        return rhs();
      }
    }

    if (matchPattern(rhs(), m_Constant(&cst_attr))) {
      if (cst_attr.isSplat() && cst_attr.getSplatValue<bool>()) {
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
  COMPARE_FOLDER(CompareOp, ComparisonDirection::LT, less);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::LE, less_equal);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::GT, greater);
  COMPARE_FOLDER(CompareOp, ComparisonDirection::GE, greater_equal);
#undef COMPARE_FOLDER

  return {};
}

//===----------------------------------------------------------------------===//
// SelectAndScatterOp
//===----------------------------------------------------------------------===//

namespace {
// Infer the return-type of SelectAndScatterOp.
TensorType inferSelectAndScatterOpReturnType(
    TensorType operand_type, const ArrayRef<WindowDimension> window) {
  if (!operand_type.hasRank())
    return UnrankedTensorType::get(operand_type.getElementType());

  return RankedTensorType::get(
      inferWindowOutputShape(operand_type.getShape(), window),
      operand_type.getElementType());
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
  auto operand_type = operand().getType().cast<TensorType>();
  auto init_value_type = init_value().getType().cast<TensorType>();
  auto source_type = source().getType().cast<TensorType>();
  auto result_type = getResult().getType().cast<TensorType>();

  // P1.
  Block& select_block = select().front();

  if (select_block.getArguments().size() != 2)
    return emitOpError()
           << "expects the select-region to take 2 parameters, but takes "
           << select_block.getArguments().size();

  Type expected_select_arg_type =
      RankedTensorType::get({}, operand_type.getElementType());
  for (const auto& select_arg_it : llvm::enumerate(select_block.getArguments()))
    if (!compatibleShapeAndElementType(expected_select_arg_type,
                                       select_arg_it.value().getType(),
                                       /*ignoreFpPrecision=*/true))
      return emitOpError()
             << "expects the type of select-region's parameter at index "
             << select_arg_it.index() << " to be " << expected_select_arg_type
             << ", but got " << select_arg_it.value().getType();

  auto select_result = select_block.getTerminator()->getOperands();
  if (select_result.size() != 1)
    return emitOpError()
           << "expects select-region to return single value, but got: "
           << select_result.size();

  auto select_result_type = select_result[0].getType().dyn_cast<TensorType>();
  if (!select_result_type ||
      !select_result_type.getElementType().isInteger(1) ||
      (select_result_type.hasRank() &&
       select_result_type.cast<RankedTensorType>().getRank() != 0))
    return emitOpError() << "expects the return-type of select-region to be "
                            "tensor<i1>, but got: "
                         << select_result[0].getType();

  // P2.
  Block& scatter_block = scatter().front();
  SmallVector<TensorType> accumulator_subshapes;
  if (failed(verifyReducerShape(
          this->getLoc(), scatter_block,
          {RankedTensorType::get({}, source_type.getElementType())},
          {init_value_type},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/false, accumulator_subshapes)))
    return failure();

  // P3.
  SmallVector<int64_t> window_dims =
      convertDenseIntAttr(this->window_dimensions());
  if (operand_type.hasRank()) {
    if (operand_type.getRank() != window_dims.size())
      return emitOpError()
             << "expects window-dimensions size == operand rank, but got "
                "window-dimensions size: "
             << window_dims.size() << " and operand-type: " << operand_type
             << " with rank = " << operand_type.getRank() << ".";
  }

  // P4.
  auto padding_or_err = convertNx2Attribute(this->padding(), getLoc());
  if (failed(padding_or_err)) return failure();
  SmallVector<std::pair<int64_t, int64_t>> padding = *padding_or_err;

  auto window_or_err = verifyWindowAttributesAndInferWindowDimensions(
      window_dims, convertDenseIntAttr(window_strides()), padding,
      /*lhs_dilation=*/{}, /*rhs_dilation=*/{}, getLoc());
  if (failed(window_or_err)) return failure();

  // P5.
  if (!compatibleShapeAndElementType(operand_type, result_type))
    return emitOpError()
           << "expects the return-type to match the operand-type, but got "
           << result_type << " and " << operand_type << " resp.";

  // P6.
  auto window_result_type =
      inferSelectAndScatterOpReturnType(operand_type, *window_or_err);

  if (!compatibleShapeAndElementType(window_result_type, source_type,
                                     /*ignoreFpPrecision=*/true))
    return emitOpError() << "expects source-type to be " << window_result_type
                         << ", but got" << source_type;

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
    ShapedType operand_type, ArrayRef<int64_t> scatter_indices_shape,
    ShapedType update_type, bool operand_type_ranked,
    bool scatter_indices_type_ranked, bool updates_type_ranked,
    ScatterDimensionNumbersAttr dim_numbers, Location loc) {
  const auto has_duplicates = [](SmallVector<int64_t>& nums) {
    if (!llvm::is_sorted(nums)) std::sort(nums.begin(), nums.end());
    auto last = std::unique(nums.begin(), nums.end());
    return last != nums.end();
  };

  // P1.
  auto update_window_dims = to_vector(dim_numbers.getUpdateWindowDims());
  if (!llvm::is_sorted(update_window_dims))
    return mlir::emitError(loc)
           << "Expects update_window_dims to be sorted; got: ["
           << update_window_dims << "].";

  if (has_duplicates(update_window_dims))
    return mlir::emitError(loc)
           << "Expects update_window_dims to not repeat; got: ["
           << update_window_dims << "].";

  if (updates_type_ranked) {
    for (int64_t window_dim : update_window_dims) {
      if (window_dim < 0 || window_dim >= update_type.getRank()) {
        return mlir::emitError(loc)
               << "Expects each element of update_window_dims to be in range "
                  "[0, "
                  "rank-of('updates') i.e. [0, "
               << update_type.getRank() << "). got: " << window_dim << ".";
      }
    }
  }

  // P2.
  auto inserted_window_dims = to_vector(dim_numbers.getInsertedWindowDims());
  if (!llvm::is_sorted(inserted_window_dims))
    return mlir::emitError(loc)
           << "Expects inserted_window_dims to be sorted; got: ["
           << inserted_window_dims << "].";

  if (has_duplicates(inserted_window_dims))
    return mlir::emitError(loc)
           << "Expects inserted_window_dims to not repeat; got: ["
           << inserted_window_dims << "].";

  if (operand_type_ranked) {
    for (int64_t inserted_dim : inserted_window_dims) {
      if (inserted_dim < 0 || inserted_dim >= operand_type.getRank()) {
        return mlir::emitError(loc)
               << "Expects each element of inserted_window_dims to be in range "
                  "[0, rank-of('operand') i.e. [0, "
               << operand_type.getRank() << "). got: " << inserted_dim << ".";
      }
    }
  }

  // P3.
  if (operand_type_ranked) {
    auto window_size = update_window_dims.size() + inserted_window_dims.size();
    if (operand_type.getRank() != window_size)
      return mlir::emitError(loc)
             << "Expects rank-of operand to match "
                "size-of('update_window_dims')  + "
                "size-of('inserted_window_dims') i.e. "
             << window_size << " but got " << operand_type.getRank() << ".";
  }

  // P4.
  auto scatter_dims_to_operand_dims =
      to_vector(dim_numbers.getScatterDimsToOperandDims());
  auto index_vector_dim = dim_numbers.getIndexVectorDim();
  if (scatter_indices_type_ranked) {
    if (!isDynamicDimSize(scatter_indices_shape[index_vector_dim]) &&
        scatter_dims_to_operand_dims.size() !=
            scatter_indices_shape[dim_numbers.getIndexVectorDim()])
      return mlir::emitError(loc)
             << "Scatter op has " << scatter_dims_to_operand_dims.size()
             << " elements in scatter_dims_to_operand_dims and the bound of "
                "dimension index_vector_dim="
             << dim_numbers.getIndexVectorDim() << " of scatter_indices is "
             << scatter_indices_shape[dim_numbers.getIndexVectorDim()]
             << ". These two numbers must be equal.";
  }

  if (operand_type_ranked) {
    for (int i = 0; i < scatter_dims_to_operand_dims.size(); ++i) {
      int64_t scatter_dim_to_operand_dim = scatter_dims_to_operand_dims[i];
      if (scatter_dim_to_operand_dim < 0 ||
          scatter_dim_to_operand_dim >= operand_type.getRank())
        return mlir::emitError(loc)
               << "Invalid scatter_dims_to_operand_dims mapping; domain is [0, "
               << operand_type.getRank() << "), got: " << i << "->"
               << scatter_dim_to_operand_dim << ".";
    }
  }

  if (has_duplicates(scatter_dims_to_operand_dims))
    return mlir::emitError(loc)
           << "Expects scatter_dims_to_operand_dims to not repeat; got: ["
           << scatter_dims_to_operand_dims << "].";

  return success();
}

/*
 * We intend to verify the following properties:
 *  P0. scatter_indices argument must be an integral tensor. Enforced by ODS.
 *  P1. Scatter index leaf dimension must be within [0, rank(scatter_indices)"
 *      " + 1).
 *  P2. Verify reducer shape.
 *  P3. rank-of('updates') == size-of('update_window_dims') +
 *      rank-of('scatter_indices') - 1, where 'scatter_indices' is expanded by a
 *      trailing 1 dimension if 'index_vector_dim' ==
 *      rank-of('scatter_indices')
 *  P4. Validate the scatter-dimensions-numbers.
 *  P5. Valide the bounds of 'updates' w.r.t the operand.
 *  P6. Validate the bounds of 'updates' w.r.t the 'scatter_indices'.
 *  P7. Check return type.
 */
LogicalResult ScatterOp::verify() {
  auto operand_type = operand().getType().cast<TensorType>();
  auto scatter_indices_type =
      scatter_indices().getType().dyn_cast<TensorType>();
  auto updates_type = updates().getType().dyn_cast<TensorType>();

  bool operand_type_ranked = operand_type.isa<RankedTensorType>();
  bool scatter_indices_type_ranked =
      scatter_indices_type.isa<RankedTensorType>();
  bool updates_type_ranked = updates_type.isa<RankedTensorType>();

  // P1.
  int64_t index_vector_dim = scatter_dimension_numbers().getIndexVectorDim();
  if (scatter_indices_type_ranked) {
    if (index_vector_dim > scatter_indices_type.getRank() ||
        index_vector_dim < 0)
      return emitOpError()
             << "expects scatter index leaf dimension to be within [0, "
                "rank(scatter_indices) + 1."
                " rank(scatter_indices) is "
             << scatter_indices_type.getRank()
             << " and scatter index leaf dimension is " << index_vector_dim
             << ".";
  }

  // P2.
  Block& block = update_computation().front();
  SmallVector<TensorType> accumulator_subshapes;
  if (failed(verifyReducerShape(
          this->getLoc(), block, {operand_type},
          {RankedTensorType::get({}, updates_type.getElementType())},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/!operand_type_ranked, accumulator_subshapes)))
    return failure();

  // P3.
  auto update_window_dims = scatter_dimension_numbers().getUpdateWindowDims();
  SmallVector<int64_t> expanded_scatter_indices_shape;
  if (scatter_indices_type_ranked) {
    expanded_scatter_indices_shape =
        llvm::to_vector(scatter_indices_type.getShape());
    if (expanded_scatter_indices_shape.size() == index_vector_dim)
      expanded_scatter_indices_shape.push_back(1);
  }

  if (scatter_indices_type_ranked && updates_type_ranked) {
    int64_t expected_updates_rank =
        expanded_scatter_indices_shape.size() - 1 + update_window_dims.size();
    if (updates_type.getRank() != expected_updates_rank)
      return emitOpError()
             << "expects updates tensor must be of rank "
             << expected_updates_rank
             << " ( == rank-of('scatter_indices') - 1 + "
                "size-of('update_window_dims'), where 'scatter_indices' is "
                "expanded by a trailing 1 dimension if 'index_vector_dim' == "
                "rank-of('scatter_indices')), but got "
             << updates_type.getRank() << ".";
  }

  // P4.
  if (failed(ValidateScatterDimensionNumbers(
          operand_type, expanded_scatter_indices_shape, updates_type,
          operand_type_ranked, scatter_indices_type_ranked, updates_type_ranked,
          scatter_dimension_numbers(), getLoc())))
    return failure();

  // P5.
  if (updates_type_ranked) {
    auto updates_shape = updates_type.getShape();
    if (operand_type_ranked) {
      auto operand_shape = operand_type.getShape();
      auto inserted_window_dims =
          scatter_dimension_numbers().getInsertedWindowDims();

      int64_t inserted_dims_seen = 0;
      SmallVector<int64_t> max_update_slice_sizes;
      const auto dimensions_size = operand_type.getRank();
      max_update_slice_sizes.reserve(dimensions_size);
      for (int i = 0; i < dimensions_size; ++i) {
        if (inserted_dims_seen < inserted_window_dims.size() &&
            inserted_window_dims[inserted_dims_seen] == i) {
          ++inserted_dims_seen;
        } else {
          max_update_slice_sizes.push_back(operand_shape[i]);
        }
      }

      for (int i = 0; i < update_window_dims.size(); ++i) {
        auto update_window_dim = update_window_dims[i];

        if (isDynamicDimSize(updates_shape[update_window_dim]) ||
            isDynamicDimSize(max_update_slice_sizes[i]))
          continue;

        if (updates_shape[update_window_dim] > max_update_slice_sizes[i]) {
          return emitOpError()
                 << "expects bounds of the window dimensions of "
                    "updates to not exceed the "
                    "bounds of the corresponding dimensions of "
                    "operand. For dimension "
                 << update_window_dim << ", updates bound is "
                 << updates_shape[update_window_dim] << ", operand bound is "
                 << max_update_slice_sizes[i] << ".";
        }
      }
    }

    // P6.
    if (scatter_indices_type_ranked) {
      int64_t scatter_dims_seen = 0;
      for (int64_t i = 0; i < updates_shape.size(); ++i) {
        bool is_update_window_dim = std::binary_search(
            update_window_dims.begin(), update_window_dims.end(), i);

        if (is_update_window_dim) continue;
        if (scatter_dims_seen == index_vector_dim) ++scatter_dims_seen;

        if (!isDynamicDimSize(updates_shape[i]) &&
            !isDynamicDimSize(
                expanded_scatter_indices_shape[scatter_dims_seen]) &&
            (updates_shape[i] !=
             expanded_scatter_indices_shape[scatter_dims_seen])) {
          return emitOpError()
                 << "expects bounds of the scatter dimensions of "
                    "updates to be same as the "
                    "bounds of the corresponding dimensions of "
                    "scatter indices. For "
                    "scatter dimension "
                 << i << ", updates bound is " << updates_shape[i]
                 << " , scatter_indices "
                    "bound is "
                 << expanded_scatter_indices_shape[scatter_dims_seen] << ".";
        }
        ++scatter_dims_seen;
      }
    }
  }

  // P7.
  if (!compatibleShapeAndElementType(operand_type, getResult().getType()))
    return emitOpError()
           << "expects the return type to be same as the operand type: "
           << operand_type << ", but got " << getResult().getType() << ".";

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

OpFoldResult ScatterOp::fold(ArrayRef<Attribute> operands) {
  auto index = operands[1].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!index) return {};

  auto base_type = operand().getType().dyn_cast<RankedTensorType>();
  auto update_type = updates().getType().dyn_cast<RankedTensorType>();
  auto index_type = index.getType().cast<RankedTensorType>();
  if (!base_type || !index_type || !update_type) return {};

  // TODO(b/228310289): Work around canonicalization crash for complex types.
  // Remove after upstream MLIR has been fixed.
  if (base_type.getElementType().isa<ComplexType>()) return {};

  // Catch a trivial full replacement of base with update, this does not require
  // these to be constant: just that we know the type.
  if (update_type == base_type && update_type.hasStaticShape() &&
      base_type.hasStaticShape() && index.isSplat() &&
      index.getSplatValue<uint32_t>() == 0 &&
      llvm::hasSingleElement(update_computation().front())) {
    return updates();
  }
  auto base = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto update = operands[2].dyn_cast_or_null<DenseElementsAttr>();
  if (!base || !update) return {};

  // Prevent splat to be expanded if too large.
  if (base.isSplat() && base.getNumElements() > kFoldExpandSplatEltLimit)
    return {};

  // Add the virtual trailing dimension of size 1 if index_vector_dim equals to
  // index_type.rank.
  const int64_t index_vector_dim =
      scatter_dimension_numbers().getIndexVectorDim();
  if (index_vector_dim == index_type.getRank()) {
    auto index_shape = index_type.getShape().vec();
    index_shape.push_back(1);
    index_type =
        RankedTensorType::get(index_shape, index_type.getElementType());
    index = index.reshape(index_type).cast<DenseIntElementsAttr>();
  }

  // Increment the multi-dimensional index vector based on the limits for each
  // dimension specified by shape and returns false if the index rolled around
  // with true otherwise.
  auto next_index = [](llvm::SmallVector<uint64_t, 8>& index,
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
  llvm::SmallVector<uint64_t, 8> update_index(update_type.getRank(), 0);
  llvm::SmallVector<uint64_t, 8> index_index;
  index_index.reserve(index_type.getRank());
  llvm::SmallVector<uint64_t, 8> base_index;
  base_index.reserve(base_type.getRank());
  do {
    // Compute the index for the slice of the indices tensor for this update
    // value.
    index_index.clear();
    if (index_vector_dim == 0) index_index.push_back(0);
    for (int64_t i = 0; i < update_index.size(); ++i) {
      if (llvm::count(scatter_dimension_numbers().getUpdateWindowDims(), i) ==
          0)
        index_index.push_back(update_index[i]);
      if (index_index.size() == index_vector_dim) index_index.push_back(0);
    }

    // Compute the index for the given update value in the base tensor.
    base_index.assign(base_type.getRank(), 0);
    uint64_t index_count = index_type.getShape()[index_vector_dim];
    for (uint64_t i = 0; i < index_count; ++i) {
      uint64_t operand_dim =
          scatter_dimension_numbers().getScatterDimsToOperandDims()[i];
      index_index[index_vector_dim] = i;
      base_index[operand_dim] +=
          index.getValues<APInt>()[index_index].getSExtValue();
    }
    uint64_t update_window_dim_index = 0;
    auto inserted_window_dims =
        scatter_dimension_numbers().getInsertedWindowDims();
    auto update_window_dims = scatter_dimension_numbers().getUpdateWindowDims();
    for (uint64_t i = 0; i < base_index.size(); ++i) {
      if (llvm::count(inserted_window_dims, i)) continue;
      base_index[i] +=
          update_index[update_window_dims[update_window_dim_index]];
      update_window_dim_index++;
    }

    // Compute the linear index for the index into the base tensor.
    int64_t linear_base_index = 0;
    int64_t linear_base_index_multiplyer = 1;
    for (int64_t i = base_index.size() - 1; i >= 0; --i) {
      // Out of bound index have backend specific behaviour so avoid folding it.
      if (base_index[i] < 0 || base_index[i] >= base_type.getShape()[i])
        return {};
      linear_base_index += base_index[i] * linear_base_index_multiplyer;
      linear_base_index_multiplyer *= base_type.getShape()[i];
    }

    // Evaluate update computation and update the value with the newly computed
    // attribute in the base tensor.
    auto lhs = DenseElementsAttr::get(
        RankedTensorType::get({}, base_type.getElementType()),
        results[linear_base_index]);
    auto rhs = DenseElementsAttr::get(
        RankedTensorType::get({}, base_type.getElementType()),
        update.getValues<Attribute>()[update_index]);
    auto new_value = evaluateMhloRegion(update_computation(), {lhs, rhs});
    if (new_value.size() != 1 || !new_value[0]) return {};
    results[linear_base_index] =
        new_value[0].cast<DenseElementsAttr>().getValues<Attribute>()[0];
  } while (next_index(update_index, update_type.getShape()));

  return DenseElementsAttr::get(base_type, results);
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
    parser.parseComma();
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
  StringRef data_type;
  if (parser.parseKeyword(&data_type)) return Type();

  if (data_type == "token") return TokenType::get(getContext());
  parser.emitError(parser.getNameLoc()) << "unknown mhlo type: " << data_type;
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
  StringRef attr_tag;
  if (failed(parser.parseKeyword(&attr_tag))) return Attribute();
  {
    Attribute attr;
    auto parse_result = generatedAttributeParser(parser, attr_tag, type, attr);
    if (parse_result.hasValue()) return attr;
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
    parser.parseOptionalComma();
  }
  return success();
}

static ParseResult parseDimsWithMinimumElements(AsmParser& parser,
                                                SmallVector<int64_t>& dims,
                                                int min_elements) {
  if (failed(parseDims(parser, dims))) return failure();
  if (dims.size() < min_elements)
    return parser.emitError(parser.getCurrentLocation())
           << "expected at least " << min_elements << " element(s), found "
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
    ArrayRef<bool> parse_equal = {}) {
  assert(keywords.size() == parseFuncs.size());
  assert(parse_equal.empty() || parse_equal.size() == keywords.size());
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
        if (parse_equal.empty() || parse_equal[index]) {
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
  SmallVector<int64_t> update_window_dims;
  SmallVector<int64_t> inserted_window_dims;
  SmallVector<int64_t> scatter_dims_to_operand_dims;
  int64_t index_vector_dim = 0;

  if (failed(parseStruct(
          parser,
          {"update_window_dims", "inserted_window_dims",
           "scatter_dims_to_operand_dims", "index_vector_dim"},
          {[&]() { return parseDims(parser, update_window_dims); },
           [&]() { return parseDims(parser, inserted_window_dims); },
           [&]() { return parseDims(parser, scatter_dims_to_operand_dims); },
           [&]() { return parser.parseInteger(index_vector_dim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing scatter dimension numbers attribute";
    return {};
  }

  return ScatterDimensionNumbersAttr::get(
      parser.getContext(), update_window_dims, inserted_window_dims,
      scatter_dims_to_operand_dims, index_vector_dim);
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

  SmallVector<int64_t> offset_dims;
  SmallVector<int64_t> collapsed_slice_dims;
  SmallVector<int64_t> start_index_map;
  int64_t index_vector_dim = 0;

  if (failed(parseStruct(
          parser,
          {"offset_dims", "collapsed_slice_dims", "start_index_map",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, offset_dims); },
           [&]() { return parseDims(parser, collapsed_slice_dims); },
           [&]() { return parseDims(parser, start_index_map); },
           [&]() { return parser.parseInteger(index_vector_dim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing gather dimension numbers attribute";
    return {};
  }

  return GatherDimensionNumbersAttr::get(parser.getContext(), offset_dims,
                                         collapsed_slice_dims, start_index_map,
                                         index_vector_dim);
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

  SmallVector<int64_t> lhs_batching_dimensions;
  SmallVector<int64_t> rhs_batching_dimensions;
  SmallVector<int64_t> lhs_contracting_dimensions;
  SmallVector<int64_t> rhs_contracting_dimensions;

  if (failed(parseStruct(
          parser,
          {"lhs_batching_dimensions", "rhs_batching_dimensions",
           "lhs_contracting_dimensions", "rhs_contracting_dimensions"},
          {[&]() { return parseDims(parser, lhs_batching_dimensions); },
           [&]() { return parseDims(parser, rhs_batching_dimensions); },
           [&]() { return parseDims(parser, lhs_contracting_dimensions); },
           [&]() { return parseDims(parser, rhs_contracting_dimensions); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return {};
  }
  return DotDimensionNumbersAttr::get(
      parser.getContext(), lhs_batching_dimensions, rhs_batching_dimensions,
      lhs_contracting_dimensions, rhs_contracting_dimensions);
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

  static unsigned getHashValue(const NonSpatialDim& Key) {
    return DenseMapInfo<int64_t>::getHashValue(Key);
  }

  static bool isEqual(const NonSpatialDim& LHS, const NonSpatialDim& RHS) {
    return LHS == RHS;
  }
};

char NonSpatialDimToString(NonSpatialDim dim) {
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
  auto print_dim =
      [&](ArrayRef<int64_t> spatial_dims,
          ArrayRef<std::pair<int64_t, NonSpatialDim>> non_spatial_dims) {
        int64_t num_dims = 0;
        if (!spatial_dims.empty()) {
          num_dims =
              *std::max_element(spatial_dims.begin(), spatial_dims.end()) + 1;
        }
        for (const auto& dim : non_spatial_dims) {
          num_dims = std::max(num_dims, dim.first + 1);
        }

        llvm::SmallVector<int64_t> dims(num_dims, kUnknownDim);
        // Fill each element of dims with a (< 0) NonSpatialDim enum or a (>=0)
        // spatial dimension index.
        for (const std::pair<int64_t, NonSpatialDim>& non_spatial_dim :
             non_spatial_dims) {
          dims[non_spatial_dim.first] = non_spatial_dim.second;
        }
        for (const auto& spatial_dim : llvm::enumerate(spatial_dims)) {
          dims[spatial_dim.value()] = static_cast<int64_t>(spatial_dim.index());
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
            p << NonSpatialDimToString(static_cast<NonSpatialDim>(dim));
          }
        });
        p << ']';
      };

  print_dim(dnums.getInputSpatialDimensions(),
            {{dnums.getInputBatchDimension(), IOBatch},
             {dnums.getInputFeatureDimension(), IOFeature}});
  p << "x";
  print_dim(dnums.getKernelSpatialDimensions(),
            {{dnums.getKernelInputFeatureDimension(), KIFeature},
             {dnums.getKernelOutputFeatureDimension(), KOFeature}});
  p << "->";
  print_dim(dnums.getOutputSpatialDimensions(),
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
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 0;
  SmallVector<int64_t> input_spatial_dimensions;
  int64_t kernel_input_feature_dimension = 0;
  int64_t kernel_output_feature_dimension = 0;
  SmallVector<int64_t> kernel_spatial_dimensions;
  int64_t output_batch_dimension = 0;
  int64_t output_feature_dimension = 0;
  SmallVector<int64_t> output_spatial_dimensions;
  if (failed(parseStruct(
          parser,
          {"input_batch_dimension", "input_feature_dimension",
           "input_spatial_dimensions", "kernel_input_feature_dimension",
           "kernel_output_feature_dimension", "kernel_spatial_dimensions",
           "output_batch_dimension", "output_feature_dimension",
           "output_spatial_dimensions"},
          {
              [&]() { return parser.parseInteger(input_batch_dimension); },
              [&]() { return parser.parseInteger(input_feature_dimension); },
              [&]() { return parseDims(parser, input_spatial_dimensions); },
              [&]() {
                return parser.parseInteger(kernel_input_feature_dimension);
              },
              [&]() {
                return parser.parseInteger(kernel_output_feature_dimension);
              },
              [&]() { return parseDims(parser, kernel_spatial_dimensions); },
              [&]() { return parser.parseInteger(output_batch_dimension); },
              [&]() { return parser.parseInteger(output_feature_dimension); },
              [&]() { return parseDims(parser, output_spatial_dimensions); },
          }))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return failure();
  }
  dnums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), input_batch_dimension,
      input_feature_dimension, input_spatial_dimensions,
      kernel_input_feature_dimension, kernel_output_feature_dimension,
      kernel_spatial_dimensions, output_batch_dimension,
      output_feature_dimension, output_spatial_dimensions);
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
  auto parse_dims =
      [&](std::set<NonSpatialDim, std::greater<>> allowed_non_spatial_dims,
          parse_dim_result_t& parsed_dims) -> ParseResult {
    auto& spatial_dims = std::get<0>(parsed_dims);
    auto& non_spatial_dims = std::get<1>(parsed_dims);
    spatial_dims.clear();
    non_spatial_dims.clear();

    // Parse the starting [
    if (parser.parseLSquare()) {
      return failure();
    }

    llvm::SmallDenseMap<int64_t, int64_t> spatial_dims_map;
    constexpr int64_t kInvalidDimension = -1;
    // Keep track of the maximum spatial dimension parsed as we expect to see
    // all the dimensions from 0 to maximum dimension parsed.
    int64_t max_parsed_spatial_dim = kInvalidDimension;

    int64_t index = 0;
    do {
      int64_t spatial_dim;
      auto dim_location = parser.getCurrentLocation();
      OptionalParseResult parseResult =
          parser.parseOptionalInteger(spatial_dim);
      if (parseResult.hasValue()) {
        if (parseResult.getValue().failed()) {
          return failure();
        }
        // We were successful in parsing an integer. Check if it is a valid
        // dimension (non-negative and no duplicate) and add its index to the
        // spatial dims map.
        if (spatial_dim < 0)
          return parser.emitError(dim_location)
                 << "Unexpected dimension " << spatial_dim;
        if (!spatial_dims_map
                 .insert(std::pair<int64_t, int64_t>(spatial_dim, index))
                 .second)
          return parser.emitError(dim_location)
                 << "Duplicate entries for spatial dimension " << spatial_dim;
        max_parsed_spatial_dim = std::max(spatial_dim, max_parsed_spatial_dim);
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
        if (keyword.size() != 1 || allowed_non_spatial_dims.empty()) {
          return parser.emitError(dim_location, "Unexpected keyword ")
                 << keyword;
        }
        // Check if the keyword matches one of the allowed non-spatial dims.
        // If so, add it to the non_spatial dims and remove it from the
        // allowed set so that it won't be allowed again.
        bool is_allowed = false;
        for (NonSpatialDim allowed : allowed_non_spatial_dims) {
          if (keyword[0] == NonSpatialDimToString(allowed)) {
            non_spatial_dims.insert({allowed, index});
            allowed_non_spatial_dims.erase(allowed);
            is_allowed = true;
            break;
          }
        }

        if (!is_allowed) {
          mlir::InFlightDiagnostic diag =
              parser.emitError(dim_location, "Unexpected dimension ");
          diag << keyword << ", expecting ";
          llvm::interleaveComma(
              allowed_non_spatial_dims, diag,
              [&](NonSpatialDim dim) { diag << NonSpatialDimToString(dim); });
          return diag;
        }
      }
      index++;
    } while (parser.parseOptionalComma().succeeded());

    // Make sure all expected non-spatial dimensions are parsed.
    if (!allowed_non_spatial_dims.empty()) {
      mlir::InFlightDiagnostic diag =
          parser.emitError(parser.getCurrentLocation(), "Expected dimensions ");
      llvm::interleaveComma(
          allowed_non_spatial_dims, diag,
          [&](NonSpatialDim dim) { diag << NonSpatialDimToString(dim); });
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
    int64_t num_spatial_dimensions = max_parsed_spatial_dim + 1;
    spatial_dims.resize(num_spatial_dimensions);
    // Store spatial dimensions in a vector which maps spatial dim (vector
    // index) -> index in the tensor dimensions. For example, for parsed
    // dimension numbers [0, 3, 2, b, i, 1] the spatial dimension vector would
    // be [0, 5, 2, 1].
    //
    // Get all the unspecified spatial dimensions to throw a more descriptive
    // error later.
    llvm::SmallVector<int64_t> unspecified_spatial_dims;
    constexpr int kPrintUnspecifiedDimsMax = 10;
    for (int dim = 0; dim < num_spatial_dimensions; ++dim) {
      auto it = spatial_dims_map.find(dim);
      if (it == spatial_dims_map.end()) {
        // Have an upper bound on the number of unspecified dimensions to print
        // in the error message.
        if (unspecified_spatial_dims.size() < kPrintUnspecifiedDimsMax)
          unspecified_spatial_dims.push_back(dim);
        continue;
      }
      spatial_dims[dim] = it->second;
    }

    // Verify that we got all spatial dimensions between 0 and maximum parsed
    // spatial dimension.
    if (!unspecified_spatial_dims.empty()) {
      mlir::InFlightDiagnostic diag = parser.emitError(
          parser.getCurrentLocation(), "Expected spatial dimensions ");
      llvm::interleaveComma(unspecified_spatial_dims, diag);
      diag << " not specified";
      return diag;
    }

    return success();
  };

  parse_dim_result_t parsed_dims;
  if (parse_dims({IOBatch, IOFeature}, parsed_dims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> input_spatial_dimensions = parsed_dims.first;
  int64_t input_batch_dimension = parsed_dims.second[IOBatch];
  int64_t input_feature_dimension = parsed_dims.second[IOFeature];
  if (parser.parseKeyword("x")) return failure();
  if (parse_dims({KIFeature, KOFeature}, parsed_dims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> kernel_spatial_dimensions = parsed_dims.first;
  int64_t kernel_input_feature_dimension = parsed_dims.second[KIFeature];
  int64_t kernel_output_feature_dimension = parsed_dims.second[KOFeature];
  if (parser.parseArrow()) {
    return failure();
  }
  if (parse_dims({IOBatch, IOFeature}, parsed_dims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> output_spatial_dimensions = parsed_dims.first;
  int64_t output_batch_dimension = parsed_dims.second[IOBatch];
  int64_t output_feature_dimension = parsed_dims.second[IOFeature];
  dnums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), input_batch_dimension,
      input_feature_dimension, input_spatial_dimensions,
      kernel_input_feature_dimension, kernel_output_feature_dimension,
      kernel_spatial_dimensions, output_batch_dimension,
      output_feature_dimension, output_spatial_dimensions);

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
  llvm::SmallVector<int64_t> arg_tuple_indices;
  // The first element of result indices holds the aliased result index and the
  // remaining elements are the result tuple indices.
  llvm::SmallVector<int64_t> result_indices;
  bool is_must_alias = false;

  // This conveys to parseStruct that keyword "must_alias" (3rd field) is not
  // followed by a "=", but other fields are.
  llvm::SmallVector<bool, 3> parse_equal = {true, true, false};

  if (failed(
          parseStruct(parser, {kArgTupleIndices, kResult, kMustAlias},
                      {[&]() { return parseDims(parser, arg_tuple_indices); },
                       [&]() {
                         // Since the first element is the index of result, at
                         // least one element is expected.
                         return parseDimsWithMinimumElements(
                             parser, result_indices, /*min_elements=*/1);
                       },
                       [&]() {
                         // always succeeds if the keyword "must_alias" was
                         // parsed
                         is_must_alias = true;
                         return success();
                       }},
                      parse_equal))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing argument-result alias attribute";
    return {};
  }

  int64_t result_index = result_indices[0];
  auto result_tuple_indices =
      ArrayRef<int64_t>{result_indices.begin() + 1, result_indices.end()};

  return ArgResultAliasAttr::get(parser.getContext(), arg_tuple_indices,
                                 result_index, result_tuple_indices,
                                 is_must_alias);
}

// Returns the element type pointed to by `indices` in type `t`. If the indices
// are invalid, returns nullptr.
static Type GetTypeFromTupleIndices(Type type, ArrayRef<int64_t> indices) {
  Type current = type;
  for (auto index : indices) {
    TupleType tuple_type = current.dyn_cast<TupleType>();
    if (!tuple_type || index >= tuple_type.size()) return {};
    current = tuple_type.getType(index);
  }
  return current;
}

static LogicalResult VerifyArgResultAliasAttr(StringAttr attr_name,
                                              ArgResultAliasAttr alias_attr,
                                              unsigned arg_index,
                                              Operation* op) {
  // The attribute can only be applied to function-like operations.
  if (!isa<mlir::FunctionOpInterface>(op))
    return op->emitOpError() << "attribute " << attr_name
                             << " can only be used on function-like operations";

  // Verify there are no negative indices.
  auto tuple_indices = llvm::concat<const int64_t>(
      alias_attr.getArgTupleIndices(), alias_attr.getResultTupleIndices());
  if (llvm::any_of(tuple_indices, [](const int64_t val) { return val < 0; }) ||
      alias_attr.getResultIndex() < 0)
    return op->emitOpError()
           << "attribute " << attr_name
           << " expects all argument and result indices to be >= 0";

  // Verify that the result index is not out of range. Since the attribute is a
  // function argument attribute, the argument index is always correct when this
  // verifier is called.
  FunctionOpInterface func_op = cast<FunctionOpInterface>(op);
  ArrayRef<Type> arg_types = func_op.getArgumentTypes();
  ArrayRef<Type> result_types = func_op.getResultTypes();
  if (alias_attr.getResultIndex() >= result_types.size())
    return op->emitOpError()
           << "attribute " << attr_name
           << " result index is out of range, must be <" << result_types.size();

  // Verify that argument and result types pointed to by the indices are valid
  // and compatible.
  Type arg_type = GetTypeFromTupleIndices(arg_types[arg_index],
                                          alias_attr.getArgTupleIndices());
  if (!arg_type)
    return op->emitOpError() << "attribute " << attr_name
                             << " argument tuple indices are invalid";
  Type result_type =
      GetTypeFromTupleIndices(result_types[alias_attr.getResultIndex()],
                              alias_attr.getResultTupleIndices());
  if (!result_type)
    return op->emitOpError()
           << "attribute " << attr_name << " result tuple indices are invalid";

  if (failed(mlir::verifyCompatibleShape(arg_type, result_type)) ||
      getElementTypeOrSelf(arg_type) != getElementTypeOrSelf(result_type))
    return op->emitOpError() << "attribute " << attr_name
                             << " aliases do not have compatible types, "
                             << arg_type << " vs. " << result_type;
  return success();
}

//===----------------------------------------------------------------------===//
// Type utilities
//===----------------------------------------------------------------------===//

Type getExpressedTypeOrSelf(Type type) {
  auto quant_type = type.dyn_cast<quant::QuantizedType>();
  return quant_type ? quant_type.getExpressedType() : type;
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
static void BuildSortComparisonBody(llvm::ArrayRef<Type> element_types,
                                    ComparisonDirection direction,
                                    llvm::Optional<StringRef> compare_type,
                                    Region* body, OpBuilder* builder) {
  OpBuilder::InsertionGuard insertion_point_gurad(*builder);

  Location loc = body->getLoc();
  Block* block = builder->createBlock(body);
  // Add two arguments for each element type.
  for (Type element_type : element_types) {
    TensorType tensor_type = RankedTensorType::get({}, element_type);
    block->addArguments({tensor_type, tensor_type},
                        SmallVector<Location, 2>(2, loc));
  }

  ComparisonType type_attr;
  if (compare_type)
    type_attr = symbolizeComparisonType(*compare_type).getValue();
  else
    type_attr = ComparisonType::NOTYPE;
  Value compare = builder->create<mhlo::CompareOp>(
      loc, block->getArgument(0), block->getArgument(1), direction, type_attr);

  builder->create<mhlo::ReturnOp>(loc, compare);
}

SortOp CreateSortOp(PatternRewriter* rewriter, const Location& loc,
                    const llvm::ArrayRef<Value>& operands,
                    const llvm::ArrayRef<Type>& element_types,
                    int64_t dimension, bool is_stable,
                    ComparisonDirection direction) {
  assert(!operands.empty() && "No operands to sort");
  // Create the sort op.
  auto sort_op =
      rewriter->create<mhlo::SortOp>(loc, operands, dimension, is_stable);

  // Use TOTALORDER comparison type instead of the default comparison if the
  // element type is of type float.
  llvm::Optional<StringRef> compare_type = llvm::None;
  for (auto const& element_type : element_types)
    if (element_type.isa<FloatType>()) {
      compare_type.emplace("TOTALORDER");
      break;
    }
  BuildSortComparisonBody(element_types, direction, compare_type,
                          &sort_op.comparator(), rewriter);
  return sort_op;
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult deriveShapeFromOperand(
    OpBuilder* builder, Operation* op, Value operand,
    SmallVectorImpl<Value>* reifiedReturnShapes) {
  auto shaped_ty = operand.getType().dyn_cast<ShapedType>();
  if (!shaped_ty) {
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
                                                    unsigned arg_index,
                                                    NamedAttribute attr) {
  if (auto alias_attr = attr.getValue().dyn_cast<ArgResultAliasAttr>()) {
    if (failed(VerifyArgResultAliasAttr(attr.getName(), alias_attr, arg_index,
                                        op)))
      return failure();
  }
  return success();
}

LogicalResult MhloDialect::verifyOperationAttribute(Operation* op,
                                                    NamedAttribute attr) {
  if (auto alias_attr = attr.getValue().dyn_cast<ArgResultAliasAttr>()) {
    if (!isa<mlir::FunctionOpInterface>(op))
      return op->emitOpError()
             << "attribute " << attr.getName()
             << " can only be used on function-like operations";
  }
  return success();
}

}  // namespace mhlo
}  // namespace mlir
