/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "Eigen/Core"  // from @eigen_archive
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/FoldInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/utils/arithmetic_count_util.h"
#include "tensorflow/compiler/mlir/lite/utils/size_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/util/padding.h"

namespace mlir {
namespace TFL {

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CeilOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CosOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(LocalResponseNormalizationOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(FloorOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RoundOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NegOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SinOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SqrtOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SquareOp);

namespace {

ParseResult parseOneResultSameOperandTypeOp(OpAsmParser& parser,
                                            OperationState& result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> ops;
  // If the operand list is in-between parentheses, then we have a generic form.
  // (see the fallback in `printOneResultOp`).
  if (!parser.parseOptionalLParen()) {
    if (parser.parseOperandList(ops) || parser.parseRParen()) return failure();
    return parser.parseGenericOperationAfterOpName(result, ops);
  }
  Type type;
  return failure(parser.parseOperandList(ops) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperands(ops, type, result.operands) ||
                 parser.addTypeToList(type, result.types));
}

void printOneResultOp(Operation* op, OpAsmPrinter& p) {
  assert(op->getNumResults() == 1 && "op should have one result");

  // If not all the operand and result types are the same, just use the
  // generic assembly form to avoid omitting information in printing.
  auto resultType = op->getResult(0).getType();
  if (llvm::any_of(op->getOperandTypes(),
                   [&](Type type) { return type != resultType; })) {
    p.printGenericOp(op, /*printOpName=*/false);
    return;
  }

  p << ' ';
  p.printOperands(op->getOperands());
  p.printOptionalAttrDict(op->getAttrs());
  // Now we can output only one type for all operands and the result.
  p << " : " << resultType;
}

Operation* getDefiningBroadcastArgsOp(Value operand) {
  auto* defining_op = operand.getDefiningOp();
  if (!llvm::dyn_cast_or_null<TF::BroadcastToOp>(defining_op) &&
      !llvm::dyn_cast_or_null<TFL::BroadcastToOp>(defining_op)) {
    return nullptr;
  }

  Value broadcast_shape = defining_op->getOperand(
      1);  // Broadcasted shape operand of BroadcastTo op.
  Operation* parent_of_defining_op = broadcast_shape.getDefiningOp();
  if (!llvm::dyn_cast_or_null<TF::BroadcastArgsOp>(parent_of_defining_op) &&
      !llvm::dyn_cast_or_null<TFL::BroadcastArgsOp>(parent_of_defining_op)) {
    return nullptr;
  }
  return parent_of_defining_op;
}

// Returns shape of a ranked tensor.
// Precondition: value_tensor's is ranked tensor.
// Returns a Squeezed shape. Truncation here means eliminating the redundant
// dimensions 1.
DenseElementsAttr GetSqueezedShape(Value value_tensor) {
  auto value_shape_type = llvm::dyn_cast<ShapedType>(value_tensor.getType());
  assert(value_shape_type.hasRank() && "value_tensor should be ranked tensor");

  auto value_shape = value_shape_type.getShape();
  SmallVector<int32_t> return_squeeze_shape;
  return_squeeze_shape.reserve(value_shape.size());

  for (size_t dim_idx = 0; dim_idx < value_shape.size(); ++dim_idx) {
    int64_t dim = value_shape[dim_idx];
    if (dim == 1) {
      continue;
    }
    return_squeeze_shape.push_back(
        ShapedType::isDynamic(dim) ? -1 : static_cast<int32_t>(dim));
  }

  return mlir::DenseElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int>(return_squeeze_shape.size())},
          mlir::IntegerType::get(value_tensor.getContext(), 32)),
      llvm::ArrayRef(return_squeeze_shape));
}

// This is a utility function to deduce the effective permutation to apply on
// TFL_TransposeOp when the tensor has some dimensions with value==1
// Example- "tfl.transpose"(tensor<56x8x56x1x1x1x7xf32>, [4, 5, 1, 2, 0, 6, 3])
// Permutation before squeese is [4, 5, 1, 2, 0, 6, 3] becomes [1, 2, 0, 3]
// after squeeze is perfomed to retain the relative ordering of the non-1 dims.
DenseElementsAttr GetSqueezedPermutation(Value input_value,
                                         Value input_permutation) {
  auto input_shape =
      mlir::dyn_cast<ShapedType>(input_value.getType()).getShape();
  absl::flat_hash_map<int32_t, int32_t> permutation_map;

  for (size_t before_dim_idx = 0, after_dim_idx = 0;
       before_dim_idx < input_shape.size(); ++before_dim_idx) {
    if (input_shape[before_dim_idx] == 1) {
      continue;
    }
    permutation_map.insert({before_dim_idx, after_dim_idx++});
  }

  SmallVector<int32_t> squeezed_permutation;
  DenseElementsAttr input_perm_const;
  if (matchPattern(input_permutation, m_Constant(&input_perm_const))) {
    for (int32_t idx = 0; idx < input_perm_const.getNumElements(); ++idx) {
      size_t perm = input_perm_const.getValues<APInt>()[idx].getSExtValue();
      if (input_shape[perm] == 1) {
        continue;
      }
      squeezed_permutation.push_back(permutation_map[perm]);
    }
  }

  return mlir::DenseElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int>(squeezed_permutation.size())},
          mlir::IntegerType::get(input_permutation.getContext(), 32)),
      llvm::ArrayRef(squeezed_permutation));
}

#include "tensorflow/compiler/mlir/lite/ir/tfl_canonicalize.inc"

}  // namespace

// Returns true when the given type lists contain a single element of shaped
// type with compatible shapes (unranked shape is compatible with any ranked
// shape, ranked shapes are compatible if their respective dimensions are
// compatible, dynamic dimensions are compatible with any size, static
// dimensions must be equal to be compatible) and identical element types.
bool VerifyCompatibleShapesSameElementType(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size() || lhs.size() != 1) return false;
  if (failed(mlir::verifyCompatibleShape(lhs[0], rhs[0]))) return false;
  auto lhsShaped = mlir::cast<ShapedType>(lhs[0]);
  auto rhsShaped = mlir::cast<ShapedType>(rhs[0]);
  return lhsShaped.getElementType() == rhsShaped.getElementType();
}

// Returns true when the given operand arguments have the same shape or
// broadcastable shape within the given rank. If any given shapes are
// non-static and maximum rank is within the given rank, this method returns
// true.
bool VerifyOperandsHaveSameShapesOrBroadcastableShape(
    Operation* op, ArrayRef<unsigned> indices, int max_bcast_rank) {
  if (indices.empty()) return true;

  // First, it checks there are any inputs that has unknown rank.
  bool has_unknown_shape_input = false;
  bool has_same_shape = true;
  bool reach_first_known_shape = false;
  int64_t max_rank = -1;

  ArrayRef<int64_t> pivot_shape;
  SmallVector<int64_t, 4> current_shape;
  SmallVector<int64_t, 4> result_shape;

  for (unsigned index : indices) {
    ShapedType shaped_type =
        mlir::dyn_cast<ShapedType>(op->getOperand(index).getType());
    if (!shaped_type || !shaped_type.hasRank()) {
      // Marks that we have an unknown rank input.
      has_unknown_shape_input = true;
      continue;
    }
    max_rank = std::max(max_rank, shaped_type.getRank());
    if (!shaped_type.hasStaticShape()) {
      // Marks that we have an unknown shape input.
      has_unknown_shape_input = true;
      continue;
    }

    ArrayRef<int64_t> shape = shaped_type.getShape();
    if (!reach_first_known_shape) {
      pivot_shape = shape;
      current_shape.assign(shape.begin(), shape.end());
      reach_first_known_shape = true;
      continue;
    }

    if (!pivot_shape.equals(shape)) {
      has_same_shape = false;
    }
    //  Checks if all the inputs are broadcastable since they have not all the
    //  same shapes.
    if (!OpTrait::util::getBroadcastedShape(current_shape, shape,
                                            result_shape)) {
      return false;
    }
    current_shape = result_shape;
  }

  // If all the shape is known and same, CPU kernels are able to handle inputs
  // regardless of dimension size.
  if (!has_unknown_shape_input) {
    return has_same_shape || max_rank <= max_bcast_rank;
  }

  // It will treat the unknown shape inputs as acceptable inputs for model
  // compatibility if all known ranks are no bigger than the allowed broadcast
  // maximum rank.
  if (max_rank <= max_bcast_rank) {
    return true;
  }

  // Checks if all operands are broadcasted by BroadcastTo ops with the shape
  // is calculated from the same BroadcastArgs op. In such case, all operands
  // will have the same shape.
  Operation* broadcast_args_pivot = nullptr;
  for (unsigned index : indices) {
    Operation* parent_broadcast_args =
        getDefiningBroadcastArgsOp(op->getOperand(index));
    if (parent_broadcast_args == nullptr) {
      return false;
    }

    if (broadcast_args_pivot == nullptr) {
      broadcast_args_pivot = parent_broadcast_args;
      continue;
    }

    if (broadcast_args_pivot != parent_broadcast_args) {
      return false;
    }
  }
  return true;
}

// Return true when the given element_type is QI8.
bool IsQI8Type(Type element_type) {
  auto quantized_type = mlir::dyn_cast<QuantizedType>(element_type);
  return quantized_type != nullptr &&
         quantized_type.getStorageTypeIntegralWidth() == 8 &&
         quantized_type.isSigned();
}

// Return true when the given element_type is QUI8.
bool IsQUI8Type(Type element_type) {
  auto quantized_type = mlir::dyn_cast<QuantizedType>(element_type);
  return quantized_type != nullptr &&
         quantized_type.getStorageTypeIntegralWidth() == 8 &&
         !quantized_type.isSigned();
}

// Return true when the given element_type is QI16.
bool IsQI16Type(Type element_type) {
  auto quantized_type = mlir::dyn_cast<QuantizedType>(element_type);
  return quantized_type != nullptr &&
         quantized_type.getStorageTypeIntegralWidth() == 16 &&
         quantized_type.isSigned();
}

// Return true when the given element_type is I16.
bool IsI16Type(Type element_type) {
  return element_type.isInteger(16) && !element_type.isUnsignedInteger();
}

// Return true when the given element_type is I32.
bool IsI32Type(Type element_type) {
  return element_type.isInteger(32) && !element_type.isUnsignedInteger();
}

// Return true when the given element_type is UI32.
bool IsUI32Type(Type element_type) {
  return element_type.isInteger(32) && element_type.isUnsignedInteger();
}

// Return true when the given element_type is I64.
bool IsI64Type(Type element_type) {
  return element_type.isInteger(64) && !element_type.isUnsignedInteger();
}

// Return true if the value is a splat tensor constant zero.
bool EqualsZero(Value value) {
  DenseElementsAttr constant;
  if (!matchPattern(value, m_Constant(&constant)) || !constant.isSplat()) {
    return false;
  }

  Type element_type = mlir::cast<ShapedType>(value.getType()).getElementType();
  if (mlir::isa<FloatType>(element_type)) {
    return constant.getSplatValue<APFloat>().isZero();
  } else {
    return false;
  }
}

// Replaces the bias operand with a "none" type value if the bias value is
// constant zero.
// `ConcreteOpType` must be an concrete MLIR op class that has an optional
// bias operand named 'bias'.
template <typename ConcreteOpType>
struct RemoveOptionalZeroBias : public OpRewritePattern<ConcreteOpType> {
  using OpRewritePattern<ConcreteOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcreteOpType op,
                                PatternRewriter& rewriter) const override {
    if (EqualsZero(op.getBias())) {
      auto none_value = rewriter.create<TFL::NoValueOp>(
          rewriter.getUnknownLoc(), rewriter.getNoneType(),
          rewriter.getUnitAttr());
      op.getBiasMutable().assign(none_value);
      return success();
    }

    return failure();
  }
};

struct SetAsymmetricQuantizeInput : public OpRewritePattern<FullyConnectedOp> {
  using OpRewritePattern<FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FullyConnectedOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getAsymmetricQuantizeInputs() == std::nullopt ||
        op.getAsymmetricQuantizeInputs() == false) {
      auto new_op = rewriter.create<FullyConnectedOp>(
          op.getLoc(), op.getOutput().getType(), op.getInput(), op.getFilter(),
          op.getBias(), op.getFusedActivationFunction(), op.getWeightsFormat(),
          op.getKeepNumDims(), rewriter.getBoolAttr(true));
      rewriter.replaceOp(op, new_op.getOutput());
      return success();
    }
    return failure();
  }
};

// Return true if the given Add operation has the CPU kernel supported shapes.
bool VerifyAddOpShapeConstraints(AddOp op) {
  auto element_type = getElementTypeOrSelf(op.getOutput().getType());

  // Allows F32, QI8, QUI8 and I32 outputs when the operands have valid shapes,
  // which are broadcastable shapes up to four dimensions or have same shapes.
  if (element_type.isF32() || IsQI8Type(element_type) ||
      IsQUI8Type(element_type) || IsI16Type(element_type) ||
      IsI32Type(element_type) || IsI64Type(element_type)) {
    return VerifyOperandsHaveSameShapesOrBroadcastableShape(
        /*op=*/op.getOperation(), /*indices=*/ArrayRef<unsigned>{0, 1},
        /*max_bcast_rank=*/6);
  }

  // Allows QI16 output when operands have the same shape.
  if (IsQI16Type(element_type)) {
    return succeeded(mlir::verifyCompatibleShape(op.getLhs().getType(),
                                                 op.getRhs().getType()));
  }
  return false;
}

// Return true if the given Sub operation has the CPU kernel supported shapes.
bool VerifySubOpShapeConstraints(SubOp op) {
  auto element_type = getElementTypeOrSelf(op.getOutput().getType());

  // Allows F32, QUI8, and QI16 outputs when the operands have valid shapes,
  // which are broadcastable shapes up to five dimension or have same shapes.
  if (element_type.isF32() || IsI32Type(element_type) ||
      IsI64Type(element_type) || IsQUI8Type(element_type) ||
      IsQI16Type(element_type)) {
    return VerifyOperandsHaveSameShapesOrBroadcastableShape(
        /*op=*/op.getOperation(), /*indices=*/ArrayRef<unsigned>{0, 1},
        /*max_bcast_rank=*/6);
  }

  // Allows QI8 output when the operands have valid shapes, which are
  // broadcastable shapes up to four dimension or have same shapes.
  if (IsQI8Type(element_type)) {
    return VerifyOperandsHaveSameShapesOrBroadcastableShape(
        /*op=*/op.getOperation(), /*indices=*/ArrayRef<unsigned>{0, 1},
        /*max_bcast_rank=*/6);
  }
  return false;
}

// Return true if the given Mul operation has the CPU kernel supported shapes.
bool VerifyMulOpShapeConstraints(MulOp op) {
  auto element_type = getElementTypeOrSelf(op.getOutput().getType());

  // Allows QI8 and QUI8 inputs up to five dimension broadcasting unless the
  // output type is not QI16. If the output type is Q16, allows only the same
  // shape operands.
  if (IsQI8Type(element_type) || IsQUI8Type(element_type)) {
    if (IsQI16Type(getElementTypeOrSelf(op.getLhs().getType()))) {
      return succeeded(mlir::verifyCompatibleShape(op.getLhs().getType(),
                                                   op.getRhs().getType()));
    }
    return VerifyOperandsHaveSameShapesOrBroadcastableShape(
        /*op=*/op.getOperation(), /*indices=*/ArrayRef<unsigned>{0, 1},
        /*max_bcast_rank=*/6);
  }

  // Allows I32, I64, QI16 and F32 outputs when the operands have valid shapes,
  // which are broadcastable shapes up to four dimension or have same shapes.
  if (IsI32Type(element_type) || IsUI32Type(element_type) ||
      IsI64Type(element_type) || IsQI16Type(element_type) ||
      IsI16Type(element_type) || mlir::isa<ComplexType>(element_type) ||
      element_type.isF32()) {
    return VerifyOperandsHaveSameShapesOrBroadcastableShape(
        /*op=*/op.getOperation(), /*indices=*/ArrayRef<unsigned>{0, 1},
        /*max_bcast_rank=*/6);
  }
  return false;
}

class FlatIndHelper {
 public:
  explicit FlatIndHelper(mlir::ShapedType type) : type_(type) {}

  llvm::SmallVector<int64_t> GetShapedInd(int64_t flat_ind) const {
    llvm::SmallVector<int64_t> result;
    result.reserve(type_.getRank());

    auto num_els = type_.getNumElements();

    for (auto d : type_.getShape()) {
      num_els /= d;
      result.push_back(flat_ind / num_els);
      flat_ind %= num_els;
    }

    return result;
  }

  int64_t GetFlatInd(llvm::ArrayRef<int64_t> shaped_ind) const {
    int64_t result = 0;

    auto num_els = type_.getNumElements();
    for (auto [max_dim, ind_dim] : llvm::zip(type_.getShape(), shaped_ind)) {
      num_els /= max_dim;
      result += num_els * ind_dim;
    }

    return result;
  }

  static void AddOffset(llvm::SmallVector<int64_t>& target,
                        const llvm::SmallVector<int64_t>& offset) {
    for (auto [ind, val] : llvm::enumerate(offset)) {
      target[ind] += val;
    }
  }

 private:
  mlir::ShapedType type_;
};

//===----------------------------------------------------------------------===//
// TensorFlowLiteDialect
//===----------------------------------------------------------------------===//

struct TensorFlowLiteInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Operation* op, Region* dest, bool wouldBeCloned,
                       IRMapping&) const final {
    // No TFLite op restricts inlining today, revise as needed in the future.
    return true;
  }
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       IRMapping& valueMapping) const final {
    return isa<WhileOp>(dest->getParentOp());
  }
};

struct TensorFlowLiteDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  // Registered hook to check if the given region, which is attached to an
  // operation that is *not* isolated from above (i.e. no internal regions
  // reference values defined in an enclosing region), should be used when
  // materializing constants.
  // In the TFLite dialect we materialize inside a while regions as slightly
  // more efficient computationally.
  bool shouldMaterializeInto(Region* region) const final {
    return isa<WhileOp>(region->getParentOp());
  }
};

void TFLDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (mlir::isa<ControlType>(type)) {
    os << "control";
    return;
  }
  os << "<unknown TFL type>";
}

Type TFLDialect::parseType(DialectAsmParser& parser) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type)) return Type();
  if (data_type == "control") return ControlType::get(getContext());
  parser.emitError(parser.getNameLoc()) << "unknown TFL type: " << data_type;
  return nullptr;
}

void TFLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_attrdefs.cc.inc"
      >();
  addInterfaces<TensorFlowLiteInlinerInterface,
                TensorFlowLiteDialectFoldInterface>();
  addTypes<ControlType>();
}

//===----------------------------------------------------------------------===//
// Common support logic
//===----------------------------------------------------------------------===//

namespace {

// Returns true if the dimensions in `a` is a suffix of the ones in `b`.
// For example, dimensions {2}, {1, 2}, and {3, 1, 2} are all suffixes to
// {5, 4, 3, 1, 2}, while {1}, {5, 4}, and {1, 3, 2} are all not.
inline bool IsTrailingDimensions(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  if (a.size() > b.size()) return false;

  return std::equal(a.rbegin(), a.rend(), b.rbegin());
}

// Returns true if it is a shaped type of f32 elements.
inline bool IsF32ShapedType(Type t) {
  if (auto shaped_type = mlir::dyn_cast_or_null<ShapedType>(t)) {
    return shaped_type.getElementType().isF32();
  }
  return false;
}

// Returns true if it is a shaped type of bf16 elements.
inline bool IsBF16ShapedType(Type t) {
  if (auto shaped_type = mlir::dyn_cast_or_null<ShapedType>(t)) {
    return shaped_type.getElementType().isBF16();
  }
  return false;
}

// Returns true if it is a shaped type of FloatType elements.
inline bool IsFloatShapedType(Type t) {
  if (auto shaped_type = mlir::dyn_cast_or_null<ShapedType>(t)) {
    return mlir::isa<FloatType>(shaped_type.getElementType());
  }
  return false;
}

// Returns new shape with rank 'new_dims' with padded ones on the
// left if needed.
inline std::vector<int64_t> GetPaddedShape(ArrayRef<int64_t> old_shape,
                                           int new_dims) {
  std::vector<int64_t> new_shape(new_dims, 1);
  std::copy_backward(old_shape.begin(), old_shape.end(), new_shape.end());
  return new_shape;
}

// Helper method that given and 'current_index' representing
// index in broadcasted tensor, get the index in the flat original tensor.
// 'shape' is the original shape with padding to match result shape.
int64_t GetElementIndex(const std::vector<int64_t>& shape,
                        const std::vector<int64_t>& current_index) {
  int64_t ind = 0;
  int64_t mul = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    ind += (current_index[i] % shape[i]) * mul;
    mul *= shape[i];
  }
  return ind;
}

// Helper method that increment index represented in 'current_index_ptr'
// in the shape of 'result_shape'.
void IncrementIndex(ArrayRef<int64_t> result_shape,
                    std::vector<int64_t>* current_index_ptr) {
  std::vector<int64_t>& current_index = *current_index_ptr;
  for (int i = result_shape.size() - 1; i >= 0; --i) {
    current_index[i]++;
    if (current_index[i] == result_shape[i]) {
      current_index[i] = 0;
    } else {
      break;
    }
  }
}

/// Performs const folding `calculate` with broadcast behavior on the two
/// attributes `operand1` and `operand2` and returns the result if possible.
/// This function assumes the both operands are verified to have value
/// attributes of broadcastable types.
template <class DenseElementsT,
          class ElementT = typename DenseElementsT::ValueType,
          class ResultT = ElementT,
          class CalculationT = llvm::function_ref<ResultT(ElementT, ElementT)>>
Attribute ConstFoldBinaryOpDenseDense(ShapedType result_type,
                                      DenseElementsAttr lhs,
                                      DenseElementsAttr rhs,
                                      const CalculationT& calculate) {
  auto type = llvm::dyn_cast_or_null<ShapedType>(
      OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType()));
  if (!type) {
    return {};
  }

  type = type.clone(result_type.getElementType());

  const bool rhs_is_splat = rhs.isSplat();
  const bool lhs_is_splat = lhs.isSplat();

  auto lhs_values = lhs.try_value_begin<ElementT>();
  auto rhs_values = rhs.try_value_begin<ElementT>();
  if (failed(lhs_values) || failed(rhs_values)) {
    return {};
  }

  // If both of them are splat, compute and return.
  if (lhs_is_splat && rhs_is_splat) {
    return DenseElementsT::get(
        type, calculate(*lhs_values.value(), *rhs_values.value()));
  }

  auto num_elements = type.getNumElements();

  SmallVector<ResultT> new_values;
  new_values.reserve(num_elements);
  const auto result_shape = type.getShape();
  std::vector<int64_t> current_index(type.getRank(), 0);

  // Create the new shape with ones padded to the left.
  const auto lhs_new_shape =
      GetPaddedShape(lhs.getType().getShape(), type.getRank());
  const auto rhs_new_shape =
      GetPaddedShape(rhs.getType().getShape(), type.getRank());

  // Add each pair of the corresponding values in the dense elements
  // attributes.
  for (int64_t i = 0; i < num_elements; ++i) {
    // current_index represents the index
    // in the N-dimension tensor. GetElementIndex returns
    // the index in the flat representation of the original tensor
    // to use.
    const int64_t lhs_index =
        lhs_is_splat ? 0 : GetElementIndex(lhs_new_shape, current_index);
    const int64_t rhs_index =
        rhs_is_splat ? 0 : GetElementIndex(rhs_new_shape, current_index);

    new_values.push_back(calculate(*(lhs_values.value() + lhs_index),
                                   *(rhs_values.value() + rhs_index)));
    IncrementIndex(result_shape, &current_index);
  }
  return DenseElementsT::get(type, new_values);
}

/// Performs const folding `calculate` with broadcast behavior on the two
/// attributes `operand1` and `operand2` and returns the result if possible.
/// This function assumes the two operands are verified to have value
/// attributes of broadcastable types.
template <class DenseElementsT,
          class ElementT = typename DenseElementsT::ValueType,
          class ResultT = ElementT,
          class CalculationT = llvm::function_ref<ResultT(ElementT, ElementT)>>
Attribute ConstFoldBinaryOp(ShapedType result_type, Attribute operand1,
                            Attribute operand2, const CalculationT& calculate) {
  if (mlir::dyn_cast_or_null<DenseElementsAttr>(operand1) &&
      mlir::dyn_cast_or_null<DenseElementsAttr>(operand2)) {
    return ConstFoldBinaryOpDenseDense<DenseElementsT, ElementT, ResultT,
                                       CalculationT>(
        result_type, mlir::cast<DenseElementsAttr>(operand1),
        mlir::cast<DenseElementsAttr>(operand2), calculate);
  }

  // TODO: support other attribute kinds

  return {};
}

/// Performs const folding with broadcast behavior on the two attributes in
/// `operands` and returns the result if possible.
/// Depending on the given `resultType`, either `floatCalculate` or
/// `intCalculate` is chosen to conduct the calculate.
Attribute ConstFoldBinaryOp(
    ShapedType type, ArrayRef<Attribute> operands,
    llvm::function_ref<APFloat(APFloat, APFloat)> float_calculate,
    llvm::function_ref<APInt(APInt, APInt)> int_calculate) {
  auto elemType = type.getElementType();

  if (mlir::isa<FloatType>(elemType))
    return ConstFoldBinaryOp<DenseFPElementsAttr, APFloat, APFloat>(
        type, operands[0], operands[1], float_calculate);

  if (elemType.isSignlessInteger())
    return ConstFoldBinaryOp<DenseIntElementsAttr, APInt, APInt>(
        type, operands[0], operands[1], int_calculate);

  return {};
}

/// Performs const folding a attributes `operand` and returns the result if
/// possible.
/// The function currently asserts that the `result_type` to be a f32 tensor
/// type.
/// TODO: Extend this function to handle integral tensor for ops like
/// "tfl.logical_not".
Attribute ConstFoldUnaryOp(Type result_type, Attribute operand,
                           llvm::function_ref<APFloat(APFloat)> calculate) {
  assert(IsF32ShapedType(result_type) || IsBF16ShapedType(result_type));
  auto result_shape_type = mlir::cast<ShapedType>(result_type);

  if (!result_shape_type.hasStaticShape()) return {};

  if (auto dense_elements =
          mlir::dyn_cast_or_null<DenseElementsAttr>(operand)) {
    SmallVector<APFloat, 16> new_values;
    const int num_elements = result_shape_type.getNumElements();
    new_values.reserve(num_elements);

    for (const APFloat& old_value : dense_elements.getValues<APFloat>()) {
      new_values.push_back(calculate(old_value));
    }

    return DenseElementsAttr::get(result_shape_type, new_values);
  }

  return {};
}

void buildComparisonBinOp(Builder* builder, OperationState& result, Value lhs,
                          Value rhs) {
  auto result_type =
      OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType());
  if (!result_type)
    emitError(result.location)
        << "non-broadcastable operands: " << lhs.getType() << " and "
        << rhs.getType();
  result.addOperands({lhs, rhs});
  // Comparison binary ops always return i1 tensor.
  if (auto shaped_type = mlir::dyn_cast<RankedTensorType>(result_type)) {
    auto result_shape = shaped_type.getShape();
    result.types.push_back(tensorflow::GetTypeFromTFTensorShape(
        result_shape, builder->getI1Type()));
  } else {
    result.types.push_back(UnrankedTensorType::get(builder->getI1Type()));
  }
}

void buildFusedBroadcastableBinOp(Builder* builder, OperationState& result,
                                  Value lhs, Value rhs,
                                  StringAttr fused_activation_function) {
  auto result_type =
      OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType());

  if (!result_type)
    emitError(result.location)
        << "non-broadcastable operands: " << lhs.getType() << " and "
        << rhs.getType();

  result.addOperands({lhs, rhs});
  result.addAttribute("fused_activation_function", fused_activation_function);
  result.types.push_back(result_type);
}

}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // TODO(b/142478136): Handle fused ops.
  if (getFusedActivationFunction() != "NONE") return {};

  auto lhs_elements = mlir::dyn_cast_or_null<DenseElementsAttr>(operands[0]);
  auto rhs_elements = mlir::dyn_cast_or_null<DenseElementsAttr>(operands[1]);
  if (lhs_elements && rhs_elements) {
    return ConstFoldBinaryOp(
        getType(), operands, [](APFloat a, APFloat b) { return a + b; },
        [](APInt a, APInt b) { return a + b; });
  }

  auto is_zero = [](Attribute a) {
    return matchPattern(a, m_Zero()) || matchPattern(a, m_AnyZeroFloat());
  };

  if (llvm::isa<quant::QuantizedType>(getType().getElementType())) {
    // Quantized folding not supported for the following.
    return {};
  }

  if (lhs_elements && is_zero(lhs_elements) &&
      getRhs().getType() == getType()) {
    return getRhs();
  }

  if (rhs_elements && is_zero(rhs_elements) &&
      getLhs().getType() == getType()) {
    return getLhs();
  }

  return {};
}

int64_t AddOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count)) return count;

  return -1;
}

//===----------------------------------------------------------------------===//
// FloorOp
//===----------------------------------------------------------------------===//

OpFoldResult FloorOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto result_type = getType();
  if (!IsF32ShapedType(result_type)) return {};

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::floor(f);
    return APFloat(result);
  };

  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// BitwiseXorOp
//===----------------------------------------------------------------------===//

OpFoldResult BitwiseXorOp::fold(FoldAdaptor adaptor) {
  auto compute = [](APInt lhs, APInt rhs) -> APInt {
    lhs ^= rhs;
    return lhs;
  };

  return ConstFoldBinaryOp<DenseElementsAttr, APInt>(
      getType(), adaptor.getLhs(), adaptor.getRhs(), compute);
}

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

OpFoldResult ExpOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto result_type = getType();
  if (!IsF32ShapedType(result_type)) return {};

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::exp(f);
    return APFloat(result);
  };

  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// LogicalNotOp
//===----------------------------------------------------------------------===//

OpFoldResult LogicalNotOp::fold(FoldAdaptor adaptor) {
  auto data = llvm::dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getLhs());
  if (!data) {
    return {};
  }

  auto compute = [](bool value) { return !value; };

  return DenseIntElementsAttr::get(
      data.getType(),
      llvm::to_vector(llvm::map_range(data.getValues<bool>(), compute)));
}

//===----------------------------------------------------------------------===//
// ConcatenationOp
//===----------------------------------------------------------------------===//
// TODO(ashwinm): Implement shape inference for Concatenation

namespace {

int64_t GetConcatenationOpAxis(ConcatenationOp op) {
  auto output_type = mlir::cast<RankedTensorType>(op.getOutput().getType());
  int32_t axis = op.getAxis();
  if (axis < 0) axis += output_type.getRank();
  return axis;
}

// Verify operand types and the result type:
//
// 1. Operand type ranks must be equal to the output type rank.
//
// 2. Operand dimension sizes (except dimension `axis`) must be equal to
//    previously seen dimension sizes of the same dimension.
//
// 3. Sum of operand dimension sizes of the `axis` dimension must be equal to
//    the dimension size of the `axis` dimension of output.
//
// Note: If an operand has unranked tensor type or has dynamic dimension size,
// those dimensions will be skipped.
LogicalResult VerifyConcatenationOpTypes(Operation* op,
                                         RankedTensorType output_type,
                                         ArrayRef<TensorType> operand_types,
                                         int64_t axis) {
  const int64_t output_rank = output_type.getRank();

  SmallVector<int64_t, 4> result_dim_sizes_loc(output_rank,
                                               ShapedType::kDynamic);
  SmallVector<int64_t, 4> result_dim_sizes(output_type.getShape().begin(),
                                           output_type.getShape().end());
  result_dim_sizes[axis] = 0;

  auto FormatLoc = [&result_dim_sizes_loc](int64_t dim) {
    const int64_t loc = result_dim_sizes_loc[dim];
    if (loc == ShapedType::kDynamic) return std::string("output");
    return llvm::formatv("operand #{0}", loc).str();
  };

  for (const auto& operand : llvm::enumerate(operand_types)) {
    auto operand_type = mlir::dyn_cast<RankedTensorType>(operand.value());
    if (!operand_type) {
      result_dim_sizes[axis] = ShapedType::kDynamic;
      continue;
    }

    const int64_t operand_rank = operand_type.getRank();
    if (operand_rank != output_rank)
      return op->emitOpError() << "rank of operand #" << operand.index()
                               << " must be equal to rank of output, expected "
                               << output_rank << ", got " << operand_rank;

    for (int64_t dim = 0; dim < output_rank; ++dim) {
      const int64_t operand_dim_size = operand_type.getDimSize(dim);
      const int64_t result_dim_size = result_dim_sizes[dim];

      if (dim == axis) {
        if (ShapedType::isDynamic(operand_dim_size) ||
            ShapedType::isDynamic(result_dim_size)) {
          result_dim_sizes[axis] = ShapedType::kDynamic;
        } else {
          result_dim_sizes[axis] += operand_dim_size;
        }
        continue;
      }

      if (ShapedType::isDynamic(operand_dim_size)) continue;

      if (ShapedType::isDynamic(result_dim_size)) {
        result_dim_sizes[dim] = operand_dim_size;
        result_dim_sizes_loc[dim] = operand.index();
        continue;
      }

      if (result_dim_size != operand_dim_size)
        return op->emitOpError()
               << "dimension size of dimension #" << dim << " of operand #"
               << operand.index() << " must be equal to "
               << "dimension size of dimension #" << dim << " of "
               << FormatLoc(dim) << ", expected " << result_dim_size << ", got "
               << operand_dim_size;
    }
  }

  const int64_t output_concated_dim_size = output_type.getDimSize(axis);
  if (!ShapedType::isDynamic(output_concated_dim_size) &&
      !ShapedType::isDynamic(result_dim_sizes[axis]) &&
      result_dim_sizes[axis] != output_concated_dim_size)
    return op->emitOpError()
           << "dimension size of dimension #" << axis << " of output "
           << "must be equal to the sum of dimension sizes of dimension #"
           << axis << ", expected " << result_dim_sizes[axis] << ", got "
           << output_concated_dim_size;

  return success();
}

// Returns true when all operands are instances of DenseElementsAttr and the
// output type has a static shape.
bool IsConcatenationOpConstFoldable(ConcatenationOp op,
                                    ArrayRef<Attribute> operands,
                                    RankedTensorType output_type,
                                    int64_t axis) {
  if (operands.empty()) return false;
  if (!output_type.hasStaticShape()) return false;
  if (axis < 0) return false;

  return llvm::all_of(operands, [](Attribute operand) {
    return operand && mlir::isa<DenseElementsAttr>(operand);
  });
}

DenseElementsAttr ConstFoldConcatenateOpDense(ArrayRef<Attribute> operands,
                                              RankedTensorType output_type,
                                              int64_t axis) {
  const auto outer_dims = output_type.getShape().take_front(axis);
  const int64_t outer_size = std::accumulate(
      outer_dims.begin(), outer_dims.end(), 1, std::multiplies<int64_t>());

  const auto base_inner_dims = output_type.getShape().drop_front(axis + 1);
  const int64_t base_inner_size =
      std::accumulate(base_inner_dims.begin(), base_inner_dims.end(), 1,
                      std::multiplies<int64_t>());

  // Splits each input operand into outer_size pieces and combines them in
  // round-robin ordering.
  std::vector<Attribute> out_attrs(output_type.getNumElements());
  int64_t out = 0;
  for (int64_t outer = 0; outer < outer_size; ++outer) {
    for (auto op : operands) {
      auto typed_attr = mlir::cast<TypedAttr>(op);
      const int64_t dim_size =
          mlir::cast<RankedTensorType>(typed_attr.getType()).getDimSize(axis);
      const int64_t inner_size = dim_size * base_inner_size;

      auto input_attrs =
          mlir::cast<DenseElementsAttr>(op).getValues<Attribute>();
      auto input_iter = input_attrs.begin() + outer * inner_size;
      for (int64_t inner = 0; inner < inner_size; ++inner)
        out_attrs[out++] = *input_iter++;
    }
  }

  return DenseElementsAttr::get(output_type, out_attrs);
}

}  // end anonymous namespace

LogicalResult ConcatenationOp::verify() {
  ConcatenationOp op = *this;
  auto output_type = mlir::dyn_cast<RankedTensorType>(op.getOutput().getType());

  // If the output type is unranked, there is nothing else to be verified.
  if (!output_type) return success();

  const int64_t axis = GetConcatenationOpAxis(op);
  if (axis < 0 || axis >= output_type.getRank())
    return op.emitOpError("concatenation dimension must be in [-rank, rank)");

  SmallVector<TensorType, 4> operand_types;
  for (Value operand : op.getValues())
    operand_types.push_back(mlir::cast<TensorType>(operand.getType()));

  return VerifyConcatenationOpTypes(op.getOperation(), output_type,
                                    operand_types, axis);
}

OpFoldResult ConcatenationOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  if (getFusedActivationFunction() == "NONE") {
    if (auto output_type =
            mlir::dyn_cast<RankedTensorType>(getOutput().getType())) {
      const int64_t axis = GetConcatenationOpAxis(*this);
      if (IsConcatenationOpConstFoldable(*this, operands, output_type, axis))
        return ConstFoldConcatenateOpDense(operands, output_type, axis);
    }
  }

  // Remove all empty values.
  SmallVector<Value, 4> non_empty_values;
  for (Value value : this->getValues()) {
    const auto shaped_type = mlir::cast<ShapedType>(value.getType());
    if (shaped_type.hasStaticShape() && shaped_type.getNumElements() == 0) {
      continue;
    }
    non_empty_values.push_back(value);
  }

  // All are not empty, do nothing.
  if (non_empty_values.size() == getNumOperands()) return nullptr;

  // If only one input is non-empty, just return it as the result of folding.
  if (non_empty_values.size() == 1) {
    return non_empty_values[0];
  }

  // Otherwise, build a new concatenation op with non-empty values.
  mlir::OpBuilder builder(getOperation());
  auto new_concat = builder.create<TFL::ConcatenationOp>(
      getLoc(), getType(), non_empty_values,
      builder.getIntegerAttr(builder.getIntegerType(32), getAxis()),
      builder.getStringAttr(getFusedActivationFunction()));
  return new_concat.getResult();
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

// TODO(b/241745316): Confirm that this is always valid
mlir::LogicalResult CustomOp::verify() {
  // Currently, this is always valid as it is a wrapper around a StringRef of 0
  // or more characters.
  return success();
}

//===----------------------------------------------------------------------===//
// CustomTfOp
//===----------------------------------------------------------------------===//

LogicalResult CustomTfOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attr, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  CustomTfOpAdaptor op(operands, attr, properties, regions);

  if (op.getRegions().empty()) return success();
  auto* real_op = &op.getBody().front().front();
  if (llvm::isa<TF::FakeQuantWithMinMaxArgsOp, TF::FakeQuantWithMinMaxVarsOp,
                TF::FakeQuantWithMinMaxVarsPerChannelOp>(real_op)) {
    Value input = *operands.begin();
    inferredReturnTypes.assign({input.getType()});
  }
  return success();
}

bool CustomTfOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  if (lhs.empty()) return true;
  if (lhs.size() != rhs.size() || lhs.size() != 1) return false;
  if (failed(mlir::verifyCompatibleShape(lhs[0], rhs[0]))) return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Gather op
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verify() {
  GatherOp op = *this;
  ShapedType params_type = mlir::cast<ShapedType>(op.getParams().getType());
  // TFLite gather kernel supports 1D string input only.
  if (isa<mlir::TF::StringType>(params_type.getElementType())) {
    if (params_type.hasRank() && params_type.getRank() != 1) {
      return op.emitOpError(
                 "expect 1d input when the given type is string, got ")
             << params_type;
    }
  }
  return mlir::success();
}

OpFoldResult GatherOp::fold(GatherOp::FoldAdaptor adaptor) {
  // Get the params tensor type/shape/data
  auto params = mlir::dyn_cast_or_null<DenseElementsAttr>(adaptor.getParams());
  if (!params) {
    return {};
  }
  auto params_type = params.getType();
  auto params_shape = params_type.getShape();
  auto params_data = params.getValues<Attribute>();
  // Get the indices tensor type/shape/data
  auto indices =
      mlir::dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getIndices());
  if (!indices) {
    return {};
  }
  auto indices_type = indices.getType();
  auto indices_shape = indices_type.getShape();
  auto indices_data = indices.getValues<IntegerAttr>();
  // Get the axis value
  int64_t axis = adaptor.getAxisAttr().getInt();
  if (axis < 0) {
    axis += params_shape.size();
  }
  // Get the batch_dims value
  int64_t batch_dims = adaptor.getBatchDimsAttr().getInt();
  if (axis < 0 || axis >= params_shape.size()) {
    return {};
  }
  if (batch_dims < 0) {
    batch_dims += params_shape.size();
  }
  // Check the values are valid
  if (batch_dims < 0) return {};
  if (batch_dims >= params_shape.size()) return {};
  if (batch_dims >= indices_shape.size()) return {};
  if (axis < batch_dims) return {};
  for (int i = 0; i < batch_dims; ++i) {
    if (params_shape[i] != indices_shape[i]) return {};
  }

  // Figure out the result shape. It will have this structure:
  // [batch dims.. , outer dims.., indexed dims.., inner dims..]
  // Where:
  // - batch dims are the first common batch_dims dimensions of params and
  //     indices
  // - outer dims are dimensions of params from batch_dims to the indexed axis
  // - indexed dims are all dimensions of indices
  // - inner dims are the dimensions of params after the indexed axis
  llvm::SmallVector<int64_t> result_shape;
  // batch dims:
  int64_t batch_flat_size = 1;
  for (int64_t i = 0; i < batch_dims; ++i) {
    batch_flat_size *= params_shape[i];
    result_shape.push_back(params_shape[i]);
  }
  // outer dims:
  int64_t outer_flat_size = 1;
  for (int64_t i = batch_dims; i < axis; ++i) {
    outer_flat_size *= params_shape[i];
    result_shape.push_back(params_shape[i]);
  }
  // indexed dims:
  const int64_t params_axis_size = params_shape[axis];
  int64_t indices_flat_size = 1;
  for (int64_t i = batch_dims; i < indices_shape.size(); ++i) {
    indices_flat_size *= indices_shape[i];
    result_shape.push_back(indices_shape[i]);
  }
  // inner dims:
  int64_t inner_flat_size = 1;
  for (int64_t i = axis + 1; i < params_shape.size(); ++i) {
    inner_flat_size *= params_shape[i];
    result_shape.push_back(params_shape[i]);
  }
  // flat size of the params tensor (used by a check below):
  int64_t params_flat_size = 1;
  for (auto params_dim : params_shape) {
    params_flat_size *= params_dim;
  }
  // flat size of the results tensor:
  int64_t results_flat_size = 1;
  for (auto result_dim : result_shape) {
    results_flat_size *= result_dim;
  }
  // Result type is the params element type with the result shape
  auto result_type = params_type.clone(result_shape);

  // Figure out the result value:
  // Follow the reference TFLite kernel implementation at
  // tensorflow/lite/kernels/internal/reference/gather.h
  std::vector<Attribute> result_values;
  result_values.reserve(results_flat_size);
  for (int64_t batch = 0; batch < batch_flat_size; ++batch) {
    for (int64_t outer = 0; outer < outer_flat_size; ++outer) {
      for (int64_t indices_idx = 0; indices_idx < indices_flat_size;
           ++indices_idx) {
        int64_t params_idx =
            indices_data[batch * indices_flat_size + indices_idx].getInt();
        int64_t from_pos =
            (((batch * outer_flat_size) + outer) * params_axis_size +
             params_idx) *
            inner_flat_size;

        if (from_pos < 0 || from_pos + inner_flat_size > params_flat_size) {
          return {};
        }
        std::copy_n(params_data.begin() + from_pos, inner_flat_size,
                    std::back_inserter(result_values));
      }
    }
  }
  return DenseElementsAttr::get(result_type, result_values);
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

// Canonicalizes BroadcastToOp to ReshapeOp if the input and output has the same
// number of elements.
struct ConvertBroadcastToReshape : public OpRewritePattern<BroadcastToOp> {
  using OpRewritePattern<BroadcastToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastToOp op,
                                PatternRewriter& rewriter) const override {
    auto input_type = mlir::cast<ShapedType>(op.getInput().getType());
    auto output_type = mlir::cast<ShapedType>(op.getType());
    if (!input_type.hasStaticShape() || !output_type.hasStaticShape() ||
        input_type.getNumElements() != output_type.getNumElements()) {
      return failure();
    }
    // Reshape op supports only new shape as I32. Add a cast op to I32 always
    // to make sure the introduced Reshape Op is a valid one.
    auto result_type = tensorflow::GetTypeFromTFTensorShape(
        mlir::cast<RankedTensorType>(op.getShape().getType()).getShape(),
        rewriter.getI32Type());
    auto cast_op =
        rewriter.create<TFL::CastOp>(op->getLoc(), result_type, op.getShape());

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getInput(),
                                           cast_op);
    return success();
  }
};

void BroadcastToOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<ConvertBroadcastToReshape>(context);
}

//===----------------------------------------------------------------------===//
// BatchMatMulOp
//===----------------------------------------------------------------------===//

// Verifier to check if following is true. Will perform the verification only if
// the BatchMatMulOp inputs and output are RankedTensors
// 1. The operands of a BatchMatMulOp are broadcast compatible
// 2. The result dimensions are correct
LogicalResult BatchMatMulOp::verify() {
  BatchMatMulOp op = *this;
  // batch size in lhs and rhs must be broadcastable
  RankedTensorType x_ty = mlir::dyn_cast<RankedTensorType>(op.getX().getType());
  RankedTensorType y_ty = mlir::dyn_cast<RankedTensorType>(op.getY().getType());

  if (!x_ty || !y_ty) return success();
  ArrayRef<int64_t> x_shape = x_ty.getShape();
  ArrayRef<int64_t> y_shape = y_ty.getShape();

  llvm::SmallVector<int64_t, 4> result_batch_shape;
  llvm::ArrayRef<int64_t> x_batches = x_shape.drop_back(2);
  llvm::ArrayRef<int64_t> y_batches = y_shape.drop_back(2);

  if (!OpTrait::util::getBroadcastedShape(x_batches, y_batches,
                                          result_batch_shape)) {
    return op.emitOpError()
           << "found incompatible broadcast batch dimensions for lhs shape "
           << x_ty << " and rhs shape " << y_ty;
  }

  RankedTensorType output_ty =
      mlir::dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!output_ty) return success();

  int64_t expected_output_rank = std::max(x_ty.getRank(), y_ty.getRank());
  if (output_ty.getRank() != expected_output_rank) {
    return op.emitOpError()
           << "found invalid output rank, expected " << expected_output_rank
           << " but got " << output_ty.getRank();
  }

  // Check output batch dim with potential broadcasting.
  ArrayRef<int64_t> output_shape = output_ty.getShape();
  for (int i = 0; i < result_batch_shape.size(); ++i) {
    if (output_shape[i] != ShapedType::kDynamic &&
        result_batch_shape[i] != ShapedType::kDynamic &&
        output_shape[i] != result_batch_shape[i])
      return op.emitOpError()
             << "has mismatching input batch dimension "
             << result_batch_shape[i] << " and output batch dimension "
             << output_shape[i];
  }

  // Check output shape for non-batch dimension, following documentation below.
  // https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-mat-mul
  int64_t x_row_dim = x_shape[x_shape.size() - 2];
  int64_t x_col_dim = x_shape[x_shape.size() - 1];
  int64_t y_row_dim = y_shape[y_shape.size() - 2];
  int64_t y_col_dim = y_shape[y_shape.size() - 1];
  int64_t out_row_dim = output_shape[output_shape.size() - 2];
  int64_t out_col_dim = output_shape[output_shape.size() - 1];

  int64_t expected_out_row_dim = op.getAdjX() ? x_col_dim : x_row_dim;
  int64_t expected_out_col_dim = op.getAdjY() ? y_row_dim : y_col_dim;

  if (expected_out_row_dim != ShapedType::kDynamic &&
      out_row_dim != ShapedType::kDynamic &&
      out_row_dim != expected_out_row_dim)
    return op.emitOpError()
           << "found invalid output dimension on row, expected "
           << expected_out_row_dim << " but got " << out_row_dim;
  if (expected_out_col_dim != ShapedType::kDynamic &&
      out_col_dim != ShapedType::kDynamic &&
      out_col_dim != expected_out_col_dim)
    return op.emitOpError()
           << "found invalid output dimension on col, expected "
           << expected_out_col_dim << " but got " << out_col_dim;

  return success();
}

//===----------------------------------------------------------------------===//
// FullyConnectedOp
//===----------------------------------------------------------------------===//

LogicalResult FullyConnectedOp::verify() {
  FullyConnectedOp op = *this;
  ShapedType input_type = mlir::cast<ShapedType>(op.getInput().getType());
  ShapedType filter_type = mlir::cast<ShapedType>(op.getFilter().getType());
  if (filter_type.hasRank() && filter_type.getRank() != 2) {
    return op.emitOpError("expect 2d filter, got ") << filter_type;
  }

  if (!input_type.hasStaticShape() || !filter_type.hasStaticShape()) {
    return mlir::success();
  }

  // Input's element size must be multiple of parameter's z_in dimension.
  const int z_in = filter_type.getDimSize(1);
  const int num_input_elements = input_type.getNumElements();
  if (z_in != 0 && num_input_elements % z_in != 0) {
    return op.emitOpError(llvm::formatv(
               "expect 'input' num_elements % {0} == 0, got input type ", z_in))
           << input_type;
  }

  // TODO(jpienaar): Include more shape verification for SHUFFLED4x16INT8
  // format.
  if (op.getWeightsFormat() == "DEFAULT") {
    ShapedType output_type =
        mlir::cast<ShapedType>((*op.getOutput().begin()).getType());
    if (!output_type.hasStaticShape()) {
      return mlir::success();
    }

    const int num_output_elements = output_type.getNumElements();
    const int z_out = filter_type.getDimSize(0);
    if (num_output_elements % z_out != 0) {
      return op.emitOpError(llvm::formatv(
                 "expect 'output' num_elements % {0} == 0, got ", z_out))
             << output_type;
    }

    if (z_in != 0 && num_input_elements / z_in != num_output_elements / z_out) {
      return op.emitOpError(
          "num_input_elements / z_in != num_output_elements / z_out");
    }
  }

  return mlir::success();
}

LogicalResult FullyConnectedOp::fold(FoldAdaptor adaptor,
                                     SmallVectorImpl<OpFoldResult>& results) {
  assert(adaptor.getOperands().size() == 3);

  // Folding not implemented with any activation function or any weight type
  // besides the default.
  if (getFusedActivationFunction() != "NONE") return failure();
  if (getWeightsFormat() != "DEFAULT") return failure();

  // Bias tensor is optional.
  const bool has_bias =
      !(!getBias() || mlir::isa<NoneType>(getBias().getType()));

  // Get the tensors.
  DenseElementsAttr input_tensor, weights_tensor, bias_tensor;
  if (!matchPattern(getInput(), m_Constant(&input_tensor)) ||
      !matchPattern(getFilter(), m_Constant(&weights_tensor)) ||
      (has_bias && !matchPattern(getBias(), m_Constant(&bias_tensor)))) {
    return failure();
  }

  // Get the tensor types.
  const auto input_type = mlir::cast<ShapedType>(input_tensor.getType());
  const auto weights_type = mlir::cast<ShapedType>(weights_tensor.getType());
  const auto bias_type =
      has_bias ? mlir::cast<ShapedType>(bias_tensor.getType()) : ShapedType{};

  const auto output_type = mlir::cast<ShapedType>(getType(0));

  // Folding only implemented for float tensors.
  if (!input_type.getElementType().isF32() ||
      !weights_type.getElementType().isF32() ||
      !output_type.getElementType().isF32() ||
      (has_bias && !bias_type.getElementType().isF32())) {
    return failure();
  }

  // Folding only implemented for static shapes
  if (!input_type.hasStaticShape() || !weights_type.hasStaticShape() ||
      (has_bias && !bias_type.hasStaticShape())) {
    return failure();
  }

  auto is_foldable = [](llvm::ArrayRef<int64_t> shape) {
    return shape.size() == 1 || (shape.size() == 2 && shape.front() == 1);
  };

  const bool weights_foldable = weights_type.getShape().size() == 2;
  const bool bias_foldable = !has_bias || is_foldable(bias_type.getShape());

  // Folding only implemented for 1D input, 2D weights and 1D bias
  if (!is_foldable(input_type.getShape()) || !bias_foldable ||
      !weights_foldable) {
    return failure();
  }

  // Get the sizes
  const auto input_size = input_type.getNumElements();
  const auto output_size = output_type.getNumElements();

  // Get iterators to the tensors.
  const auto input_values_it = input_tensor.getValues<float>().begin();
  const auto weights_values_ptr = weights_tensor.getValues<float>().begin();
  auto weights_row_it = weights_values_ptr;
  // The 'else' case could be nullptr, but the types don't match.
  auto bias_values_it =
      has_bias ? bias_tensor.getValues<float>().begin() : input_values_it;

  // Do the actual folding, one output at a time.
  std::vector<float> result_values;
  result_values.reserve(output_size);

  for (int i = 0; i < output_size; ++i) {
    // Dot product with Kahan/Neumaier summation to minimize numeric errors.
    float sum = has_bias ? *bias_values_it : 0.0f;
    float compensation = 0.0f;
    for (int j = 0; j < input_size; ++j) {
      const float addend = input_values_it[j] * weights_row_it[j];
      const float new_sum = sum + addend;
      // DO NOT enable -funsafe-math-optimizations here.
      // There is a test detecting unsafe optimizations.
      // Unsafe math optimizations can reorder float formulas, and set the
      // compensation to constant 0. The formula must be evaluated as written
      // for the algorithm to work.
      // (Note: -ffast-math is a superset of -funsafe-math-optimizations.)
      if (std::abs(sum) >= std::abs(addend)) {
        compensation += (sum - new_sum) + addend;
      } else {
        compensation += (addend - new_sum) + sum;
      }
      sum = new_sum;
    }
    result_values.push_back(sum + compensation);
    weights_row_it += input_size;
    bias_values_it++;
  }

  // Set result tensor
  const auto folded =
      DenseElementsAttr::get(output_type, ArrayRef<float>(result_values));
  results.assign({folded});

  return success();
}

void FullyConnectedOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                   MLIRContext* context) {
  results.add<RemoveOptionalZeroBias<FullyConnectedOp>>(context);
  results.add<SetAsymmetricQuantizeInput>(context);
}

int64_t FullyConnectedOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  if (ArithmeticCountUtilHelper::GetArithmeticCountForConvAndFullyconnectedOp(
          op, &count))
    return count;

  return -1;
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

void Conv2DOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  // TODO(b/180121750): Enable the pattern after the integration tests are
  // fixed.
  // results.add<RemoveOptionalZeroBias<Conv2DOp>>(context);
}

static LogicalResult ComputeConvWindowedOutputSize(
    int64_t input_size, int64_t filter_size, int64_t dilation_rate,
    int64_t stride, tensorflow::Padding padding, int64_t* output_size) {
  int64_t pad_low;
  int64_t pad_high;

  absl::Status status = tensorflow::GetWindowedOutputSizeVerbose(
      input_size, filter_size, dilation_rate, stride, padding, output_size,
      &pad_low, &pad_high);
  // Return failure if expected_output_size could not be calculated.
  if (!status.ok()) return failure();
  return success();
}

LogicalResult Conv2DOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attr, OpaqueProperties properties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  Conv2DOpAdaptor op(operands, attr, properties);

  const Value input = op.getInput();
  const Value filter = op.getFilter();

  const RankedTensorType input_ty =
      mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
  const RankedTensorType filter_ty =
      mlir::dyn_cast_or_null<RankedTensorType>(filter.getType());
  // If indeed both input type & filter type are ranked type and have ranks.
  // We will need to check their ranks are valid.
  if ((input_ty && input_ty.hasRank() && input_ty.getRank() != 4) ||
      (filter_ty && filter_ty.hasRank() && filter_ty.getRank() != 4)) {
    return emitOptionalError(location, "Invalid ranks");
  }

  // If either input or filter is unranked, we will just return unranked output
  // shape.
  if (!input_ty || !filter_ty || !input_ty.hasRank() || !filter_ty.hasRank()) {
    Type result_type;
    result_type = UnrankedTensorType::get(
        mlir::cast<ShapedType>(input.getType()).getElementType());
    inferredReturnTypes.assign({result_type});
    return success();
  }

  auto stride_h = op.getStrideHAttr().getInt();
  auto stride_w = op.getStrideWAttr().getInt();
  auto dilation_h = op.getDilationHFactorAttr().getInt();
  auto dilation_w = op.getDilationWFactorAttr().getInt();

  // We don't have EXPLICIT PADDING in TfLite.
  auto paddings = op.getPadding();
  tensorflow::Padding padding;
  auto padding_is_valid = GetPaddingFromString(paddings.str(), &padding);
  if (!padding_is_valid.ok()) {
    return emitOptionalError(location, "invalid padding format provided");
  }

  // Output always have rank 4. All dimensions are initialized to
  // dynamic size and can be partially inferred.
  // TFL's conv2d is always NHWC format & the filter is OHWI.
  SmallVector<int64_t, 4> return_shape(4, ShapedType::kDynamic);
  return_shape[0] = input_ty.getDimSize(0);
  return_shape[3] = filter_ty.getDimSize(0);

  // Spatial dimensions can be inferred only when both input and filter are
  // ranked because we need to get their spatial dimensions.

  // Height.
  if (!input_ty.isDynamicDim(1) && !filter_ty.isDynamicDim(1)) {
    int64_t output_height;
    if (failed(ComputeConvWindowedOutputSize(
            input_ty.getDimSize(1), filter_ty.getDimSize(1), dilation_h,
            stride_h, padding, &output_height))) {
      return failure();
    }
    return_shape[1] = output_height;
  }

  // Width.
  if (!input_ty.isDynamicDim(2) && !filter_ty.isDynamicDim(2)) {
    int64_t output_width;
    if (failed(ComputeConvWindowedOutputSize(
            input_ty.getDimSize(2), filter_ty.getDimSize(2), dilation_w,
            stride_w, padding, &output_width))) {
      return failure();
    }
    return_shape[2] = output_width;
  }

  auto result_type = tensorflow::GetTypeFromTFTensorShape(
      return_shape, input_ty.getElementType());

  inferredReturnTypes.assign({result_type});
  return success();
}

bool Conv2DOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size() || lhs.size() != 1) return false;
  if (failed(mlir::verifyCompatibleShape(lhs[0], rhs[0]))) return false;
  return true;
}

int64_t Conv2DOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  if (ArithmeticCountUtilHelper::GetArithmeticCountForConvAndFullyconnectedOp(
          op, &count))
    return count;

  return -1;
}

//===----------------------------------------------------------------------===//
// DepthwiseConv2DOp
//===----------------------------------------------------------------------===//

void DepthwiseConv2DOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                    MLIRContext* context) {
  // TODO(b/180121750): Enable the pattern after the integration tests are
  // fixed.
  // results.add<RemoveOptionalZeroBias<DepthwiseConv2DOp>>(context);
}

int64_t DepthwiseConv2DOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  if (ArithmeticCountUtilHelper::GetArithmeticCountForConvAndFullyconnectedOp(
          op, &count))
    return count;

  return -1;
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

static void BuildGatherOp(OpBuilder* builder, OperationState& result,
                          Value params, Value indices, IntegerAttr axis,
                          IntegerAttr batch_dims) {
  auto params_type = mlir::cast<TensorType>(params.getType());
  auto indices_type = mlir::cast<TensorType>(indices.getType());

  // If params/indices is unranked, then output is unranked.
  if (!params_type.hasRank() || !indices_type.hasRank())
    return TFL::GatherOp::build(
        *builder, result, UnrankedTensorType::get(params_type.getElementType()),
        params, indices, axis, batch_dims);

  int64_t params_rank = params_type.getRank();
  int64_t indices_rank = indices_type.getRank();

  // params rank is guaranteed to be at least 1.
  // Produces an output tensor with shape:
  // params.shape[:axis] + indices.shape + params.shape[axis + 1:]
  std::vector<int64_t> shape(params_type.getShape());
  int64_t axis_i = axis.getInt();

  // For neg axis values, we wrap around params, e.g. axis = -1 => params[:-1]
  if (axis_i < 0) {
    axis_i += params_rank;
  }

  // params must be at least rank axis + 1
  if (params_rank < axis_i + 1) {
    emitError(result.location, "params must be at least rank axis + 1");
  }

  int64_t batch_dims_i = batch_dims.getInt();
  if (batch_dims_i < 0) {
    batch_dims_i += indices_rank;
  }

  if (batch_dims_i > axis_i) {
    emitError(result.location,
              "axis should be bigger than or equal to batch_dims");
  }
  if (batch_dims_i >= params_rank || batch_dims_i > indices_rank) {
    emitError(result.location,
              "batch_dims must be smaller than params' rank and smaller than "
              "or equal to indices'rank");
  }
  for (int i = 0; i < batch_dims_i; ++i) {
    if (indices_type.getShape()[i] != params_type.getShape()[i]) {
      emitError(result.location,
                "batch dimensions of params must be equal to batch dimensions "
                "of indices");
    }
  }

  if ((indices_rank == 0) || (indices_rank == batch_dims_i)) {
    // Scalar indices (output is rank(params) - 1).
    // Erase shape[axis]
    shape.erase(shape.begin() + axis_i);
  } else if (indices_rank == 1) {
    // Vector indices (output is rank(params)).
    // Copy indices.shape into params.shape[axis]
    std::copy(std::begin(indices_type.getShape()),
              std::end(indices_type.getShape()), std::begin(shape) + axis_i);
  } else {
    // Higher rank indices (output is rank(params) + rank(indices) - 1).
    shape.resize(params_rank + indices_rank - 1 - batch_dims_i);
    // Copy params.shape[axis + 1: ] into shape[axis + indices_rank:]
    std::copy(std::begin(params_type.getShape()) + axis_i + 1,
              std::end(params_type.getShape()),
              std::begin(shape) + axis_i + indices_rank - batch_dims_i);

    // Copy indices.shape into params.shape[axis]
    std::copy(std::begin(indices_type.getShape()) + batch_dims_i,
              std::end(indices_type.getShape()), std::begin(shape) + axis_i);
  }

  TFL::GatherOp::build(
      *builder, result,
      tensorflow::GetTypeFromTFTensorShape(shape, params_type.getElementType()),
      params, indices, axis, batch_dims);
}

//===----------------------------------------------------------------------===//
// ScatterNdOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ScatterNdOp::verify() {
  ScatterNdOp op = *this;
  auto indices = op.getIndices();
  auto updates = op.getUpdates();
  auto shape = op.getShape();
  auto output = op.getOutput();

  auto updates_type = mlir::cast<ShapedType>(updates.getType());
  auto indices_type = mlir::cast<ShapedType>(indices.getType());

  if (!indices_type.hasStaticShape() || !updates_type.hasStaticShape()) {
    return success();
  }

  // Checks if the shape of `updates` is a tensor of shape
  // `indices.shape[:-1] + shape[indices.shape[-1]:]`, as described in
  // ScatterNd op description.

  auto outer_dims = indices_type.getRank() - 1;
  auto outermost_dim = indices_type.getDimSize(outer_dims);
  // Checks whether the first `outer_dims` dimensions of `indices` and
  // `updates` are equal.
  for (auto i = 0; i < outer_dims; i++) {
    if (indices_type.getDimSize(i) != updates_type.getDimSize(i)) {
      return op.emitOpError()
             << "indices.Dims(" << i << ") == " << indices_type.getDimSize(i)
             << ", but updates.Dims(" << i
             << ") == " << updates_type.getDimSize(i);
    }
  }

  auto output_type = mlir::cast<ShapedType>(output.getType());
  auto shape_type = mlir::cast<ShapedType>(shape.getType());
  if (shape_type.hasStaticShape()) {
    // Check the rank of `shape`.
    auto output_rank = outermost_dim + updates_type.getRank() - outer_dims;
    if (shape_type.getDimSize(0) != output_rank) {
      return op.emitOpError()
             << "shape must be a vector of length " << output_rank;
    }
    if (output_type.hasRank()) {
      if (output_type.getRank() != output_rank) {
        return op.emitOpError()
               << "output must have the same rank with the length of shape = "
               << output_rank;
      }
    }
  }

  DenseIntElementsAttr shape_value;
  if (matchPattern(shape, m_Constant(&shape_value))) {
    for (const auto shape_elem : shape_value) {
      if (shape_elem.getSExtValue() <= 0) {
        return op.emitOpError("all elements of shape must be > 0");
      }
    }

    // Checks whether the last `(shape_type.getDimSize(0) - outermost_dim)`
    // dimensions of `updates` and `shape` are equal.
    for (const auto& shape_it : llvm::enumerate(shape_value)) {
      int64_t i = shape_it.index();
      auto value = shape_it.value().getSExtValue();
      if (i >= outermost_dim) {
        auto corresponding_dim = i - outermost_dim + outer_dims;
        if (value != updates_type.getDimSize(corresponding_dim)) {
          return op.emitOpError()
                 << "updates.Dims(" << i
                 << ") == " << updates_type.getDimSize(corresponding_dim)
                 << ", but shape[" << i << "] == " << value;
        }
      }
    }

    // Checks if the output has the shape specified by `shape`.
    if (output_type.hasStaticShape()) {
      for (const auto& shape_it : llvm::enumerate(shape_value)) {
        int i = shape_it.index();
        auto value = shape_it.value().getSExtValue();
        if (output_type.getDimSize(i) != value) {
          return op.emitOpError()
                 << "output shape [" << output_type.getShape()
                 << "] must be equal to the value of shape " << shape_value;
        }
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // TODO(b/142478136): Handle fused ops.
  if (getFusedActivationFunction() != "NONE") return {};

  auto is_zero = [](Attribute a) {
    return matchPattern(a, m_Zero()) || matchPattern(a, m_AnyZeroFloat());
  };
  auto is_one = [](Attribute a) {
    return matchPattern(a, m_One()) || matchPattern(a, m_OneFloat());
  };

  // Quantized folding not supported.
  const bool is_quantized =
      llvm::isa<quant::QuantizedType>(getType().getElementType());

  auto lhs = llvm::dyn_cast_or_null<DenseElementsAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<DenseElementsAttr>(adaptor.getRhs());

  if (lhs && !is_quantized) {
    if (is_zero(lhs) && lhs.getType() == getType()) {
      return lhs;
    }
    if (is_one(lhs) && getRhs().getType() == getType()) {
      return getRhs();
    }
  }

  if (rhs && !is_quantized) {
    if (is_zero(rhs) && rhs.getType() == getType()) {
      return rhs;
    }
    if (is_one(rhs) && getLhs().getType() == getType()) {
      return getLhs();
    }
  }

  // This function is performance critical for op fusion patterns, e.g.
  // FuseBinaryOpToPrecedingAffine and FuseMulOrDivWithConv2dOrDepthwiseConv2d.
  // So a few specializations are provided to evaluate the math operation
  // more efficiently.

  // Specialization for f32 type.
  if (mlir::cast<ShapedType>(getType()).getElementType().isF32()) {
    return ConstFoldBinaryOp<DenseFPElementsAttr, float, float>(
        getType(), operands[0], operands[1],
        [](float a, float b) { return a * b; });
  }

  // Specialization for bf16 type.
  if (mlir::cast<ShapedType>(getType()).getElementType().isBF16()) {
    return ConstFoldBinaryOp<DenseFPElementsAttr, Eigen::bfloat16,
                             Eigen::bfloat16>(
        getType(), operands[0], operands[1],
        [](Eigen::bfloat16 a, Eigen::bfloat16 b) { return a * b; });
  }

  // Generic fallback with APFloat
  return ConstFoldBinaryOp(
      getType(), operands, [](APFloat a, APFloat b) { return a * b; },
      [](APInt a, APInt b) { return a * b; });
}

int64_t MulOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count)) return count;

  return -1;
}

//===----------------------------------------------------------------------===//
// PowOp
//===----------------------------------------------------------------------===//

OpFoldResult PowOp::fold(FoldAdaptor adaptor) {
  if (getType().getElementType().isF32()) {
    return ConstFoldBinaryOp<DenseFPElementsAttr, float>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](float lhs, float rhs) { return std::pow(lhs, rhs); });
  }
  if (getType().getElementType().isInteger(32)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int32_t>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int32_t lhs, int32_t rhs) { return std::pow(lhs, rhs); });
  }
  return {};
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // TODO(b/142478136): Handle fused ops.
  if (getFusedActivationFunction() != "NONE") return {};

  auto rhs = llvm::dyn_cast_or_null<DenseElementsAttr>(adaptor.getRhs());
  auto lhs = llvm::dyn_cast_or_null<DenseElementsAttr>(adaptor.getLhs());

  if (rhs && lhs) {
    return ConstFoldBinaryOp(
        getType(), operands, [](APFloat a, APFloat b) { return a / b; },
        [](APInt a, APInt b) { return a.sdiv(b); });
  }

  if (llvm::isa<quant::QuantizedType>(getType().getElementType())) {
    // Quantized folding not supported for the following.
    return {};
  }

  auto is_one = [](Attribute a) {
    return matchPattern(a, m_One()) || matchPattern(a, m_OneFloat());
  };

  if (rhs && is_one(rhs) && getLhs().getType() == getType()) {
    return getLhs();
  }

  return {};
}

int64_t DivOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count)) return count;

  return -1;
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

// TODO(b/133486129): Implement shape inference for pack

mlir::LogicalResult PackOp::verify() {
  PackOp op = *this;
  // TODO(antiagainst): Implement other checks as in
  // tensorflow/lite/kernels/pack.cc

  if (op.getOperation()->getNumOperands() != op.getValuesCount())
    return op.emitOpError("input count should match 'values_count' attribute");

  Value operand0 = op.getOperand(0);
  auto input_type = mlir::cast<ShapedType>(operand0.getType());

  // Check axis bounds.
  if (input_type.hasRank()) {
    int32_t axis_value = op.getAxis();
    if (axis_value < 0) axis_value += input_type.getRank() + 1;
    if (axis_value < 0 || axis_value >= input_type.getRank() + 1)
      return op.emitOpError()
             << "op attribute 'axis' should be in range [-rank - 1, rank + 1), "
             << "got rank = " << input_type.getRank()
             << ", and axis = " << op.getAxis();
  }

  // Make sure all inputs have the same shape and element type.
  // TODO(b/135032063): Simplify once fixed.
  for (Type operand_type : op.getOperandTypes()) {
    if (failed(mlir::verifyCompatibleShape(input_type, operand_type)))
      return op.emitOpError("operands should be of the same type. got ")
             << input_type << ", " << operand_type;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PReluOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult PReluOp::verify() {
  PReluOp op = *this;
  auto input_type = mlir::cast<ShapedType>(op.getInput().getType());
  auto alpha_type = mlir::cast<ShapedType>(op.getAlpha().getType());
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());

  if (input_type.hasStaticShape() && alpha_type.hasStaticShape()) {
    if (input_type.getRank() != alpha_type.getRank() + 1) {
      return op.emitOpError("'alpha' should have one less rank than 'input'.");
    }

    // Check if alpha is broadcastable
    for (int i = 0; i < alpha_type.getRank(); i++) {
      if (alpha_type.getDimSize(i) != input_type.getDimSize(i + 1) &&
          alpha_type.getDimSize(i) != 1) {
        return op.emitOpError(
            llvm::formatv("'alpha' is not broadcastable at dimension {0}.", i));
      }
    }
  }

  if (input_type.hasStaticShape() && output_type.hasStaticShape()) {
    if (input_type.getRank() != output_type.getRank()) {
      return op.emitOpError("'input' and 'output' should have the same rank.");
    }

    // Check if input and output shapes are same
    for (int i = 0; i < input_type.getRank(); i++) {
      if (input_type.getDimSize(i) != output_type.getDimSize(i)) {
        return op.emitOpError(
            "'input' and 'output' should have the same shape.");
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

namespace {
// This pattern matches and merges a tfl.reshape under the following
// condition:
// * The input's defining op is another tfl.reshape.
// TODO(antiagainst): This pattern probably should be moved to the peephole
// category, after we have the infra for peephole passes.
struct RemoveAdjacentReshape : public RewritePattern {
  explicit RemoveAdjacentReshape(MLIRContext* context)
      : RewritePattern(ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult match(Operation* op) const override {
    auto thisOp = cast<ReshapeOp>(op);
    auto prevOp = thisOp.getOperand(0).getDefiningOp();
    return isa_and_nonnull<ReshapeOp>(prevOp) ? success() : failure();
  }

  void rewrite(Operation* op, PatternRewriter& rewriter) const override {
    auto thisOp = cast<ReshapeOp>(op);
    auto prevOp = cast<ReshapeOp>(thisOp.getOperand(0).getDefiningOp());

    // Replace
    //   %1 = "tfl.reshape"(%0, %shape0)
    //   %2 = "tfl.reshape"(%1, %shape1)
    // With
    //   %2 = "tfl.reshape"(%0, %shape1)
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        op, thisOp.getType(), prevOp.getOperand(0), thisOp.getOperand(1));
  }
};

// The kernel expects an 1-D tensor for the shape operand if it presents. If all
// the dimensions are '1's except the last dimension, it will be reshaped to a
// 1-D tensor.
// Note that this pattern doesn't check or change the content of the shape
// tensor.
struct ConvertShapeTo1D : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshape,
                                PatternRewriter& rewriter) const override {
    if (!reshape.getShape().hasOneUse()) return failure();

    DenseIntElementsAttr shape;
    if (!matchPattern(reshape.getShape(), m_Constant(&shape))) {
      return failure();
    }
    // It is already a 1-D constant, no change.
    auto old_shape = shape.getShapedType().getShape();
    if (old_shape.size() == 1) {
      return failure();
    }
    // Verify all the leading dimensions are length one, except the last one.
    for (auto it = ++old_shape.rbegin(); it != old_shape.rend(); ++it) {
      if (*it != 1) {
        reshape->emitError(
            "Non-vector shape input is used, might cause runtime error");
        return failure();
      }
    }
    auto new_shape = shape.reshape(tensorflow::GetTypeFromTFTensorShape(
        {*old_shape.rbegin()}, shape.getShapedType().getElementType()));
    rewriter.replaceOpWithNewOp<TFL::ConstOp>(
        reshape.getShape().getDefiningOp(), new_shape);
    return success();
  }
};

bool InputOutputHasSameShape(mlir::Type input_type, mlir::Type output_type) {
  auto input_shaped_type = mlir::dyn_cast_or_null<ShapedType>(input_type);
  if (!input_shaped_type || !input_shaped_type.hasStaticShape()) return false;

  auto output_shaped_type = mlir::dyn_cast_or_null<ShapedType>(output_type);
  if (!output_shaped_type || !output_shaped_type.hasStaticShape()) return false;

  return input_shaped_type == output_shaped_type;
}

}  // end anonymous namespace

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // Remove identity reshape with both static result and input shape.
  auto result_type = mlir::cast<ShapedType>(getType());
  auto input_type = mlir::cast<ShapedType>(getOperand(0).getType());
  if (InputOutputHasSameShape(input_type, result_type)) return getInput();

  // Constant folding
  if (auto dense_elements =
          mlir::dyn_cast_or_null<DenseElementsAttr>(operands[0])) {
    // If the result type isn't static, tries to derive the result type from
    // the #2 operand.
    if (!result_type.hasStaticShape()) {
      auto shape_elements =
          mlir::dyn_cast_or_null<DenseElementsAttr>(operands[1]);
      if (!shape_elements) return nullptr;

      SmallVector<int64_t, 4> shape_data;
      for (const auto& it : shape_elements.getValues<APInt>()) {
        shape_data.push_back(it.getSExtValue());
      }
      result_type = tensorflow::GetTypeFromTFTensorShape(
          shape_data, input_type.getElementType());
    }
    return dense_elements.reshape(result_type);
  }

  return nullptr;
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<RemoveAdjacentReshape, ConvertShapeTo1D>(context);
}

using ReshapeErrorHandler =
    llvm::function_ref<LogicalResult(const llvm::Twine&)>;

LogicalResult GetReshapeOutputType(Value input, Value shape,
                                   ReshapeErrorHandler error_handler,
                                   TensorType& output_ty) {
  auto input_ty = mlir::cast<TensorType>(input.getType());
  auto element_ty = input_ty.getElementType();
  output_ty = UnrankedTensorType::get(element_ty);

  auto shape_ty = mlir::dyn_cast<RankedTensorType>(shape.getType());
  if (!shape_ty) return success();
  if (shape_ty.getRank() != 1)
    return error_handler(llvm::formatv(
        "requires 'shape' to be rank 1, but got {0}", shape_ty.getRank()));

  DenseIntElementsAttr shape_attr;
  if (!matchPattern(shape, m_Constant(&shape_attr))) {
    // If only shape of `shape` is known, return ranked but dynamic output
    // shape.
    if (shape_ty.hasStaticShape()) {
      llvm::SmallVector<int64_t, 8> dynamic_shape(shape_ty.getDimSize(0),
                                                  ShapedType::kDynamic);
      output_ty =
          tensorflow::GetTypeFromTFTensorShape(dynamic_shape, element_ty);
    }
    return success();
  }

  // Detect if reshape output shape is folded.
  bool shape_ty_zero_dim = false;
  int unknown_index = -1;
  // The product of constant shape argument excluding unknown dimension.
  int64_t shape_ty_size = 1;
  llvm::SmallVector<int64_t, 8> output_ty_shape;
  output_ty_shape.reserve(shape_attr.getNumElements());
  for (const auto& dim : llvm::enumerate(shape_attr.getValues<APInt>())) {
    const int64_t size = dim.value().getSExtValue();
    if (size == tensorflow::kTFDynamicSize ||  // NOLINT
        size == ShapedType::kDynamic) {        // NOLINT
      if (unknown_index != -1)
        return error_handler(llvm::formatv(
            "requires 'shape' to have at most one dynamic dimension, but got "
            "multiple dynamic dimensions at indices {0} and {1}. You need to "
            "set up the unspecified size(s) to avoid this problem, for example,"
            "setting batch size in keras model or setting unspecified input "
            "size(s) with fixed ones.",
            unknown_index, dim.index()));

      unknown_index = dim.index();
    } else if (size == 0) {
      shape_ty_zero_dim = true;
    } else if (size > 0) {
      shape_ty_size *= size;
    } else {
      return error_handler(
          llvm::formatv("requires 'shape' to have dimensions greater than -1, "
                        "but got {0} at index {1}",
                        size, dim.index()));
    }
    output_ty_shape.push_back(size);
  }

  if (!input_ty.hasStaticShape()) {
    output_ty =
        tensorflow::GetTypeFromTFTensorShape(output_ty_shape, element_ty);
    return success();
  }

  // Compute the value of the unknown dimension.
  if (unknown_index != -1) {
    // Compute number of elements in tensor shape.
    int64_t input_ty_size = 1;
    bool input_ty_zero_dim = false;
    for (const auto& dim : input_ty.getShape()) {
      if (dim > 0 || !shape_ty_zero_dim) {
        input_ty_size *= dim;
      } else {
        input_ty_zero_dim = true;
      }
    }

    const int64_t missing_dim = input_ty_size / shape_ty_size;
    if (!input_ty_zero_dim && shape_ty_size * missing_dim != input_ty_size)
      return error_handler(
          llvm::formatv("requires 'input' number of elements be a multiple of "
                        "{0}, but got {1}",
                        shape_ty_size, input_ty_size));

    // Set the unknown dimension such that total number of elements remain
    // constant.
    output_ty_shape[unknown_index] = missing_dim;
  }

  output_ty = tensorflow::GetTypeFromTFTensorShape(output_ty_shape, element_ty);

  return success();
}

mlir::LogicalResult ReshapeOp::verify() {
  ReshapeOp op = *this;
  auto error_handler = [&op](const llvm::Twine& message) -> LogicalResult {
    return op.emitOpError() << message;
  };
  TensorType expected_ty;
  if (failed(GetReshapeOutputType(op.getInput(), op.getShape(), error_handler,
                                  expected_ty)))
    return failure();

  auto output_ty = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!output_ty) return success();
  auto input_ty = mlir::cast<TensorType>(op.getInput().getType());
  if (output_ty.hasStaticShape() && input_ty.hasStaticShape()) {
    const int64_t output_ty_size = output_ty.getNumElements();
    const int64_t input_ty_size = input_ty.getNumElements();
    if (input_ty_size != output_ty_size)
      return op.emitOpError() << "requires 'output' number of elements to "
                                 "match 'input' number of elements, but got "
                              << output_ty_size << " and " << input_ty_size;
  }

  if (!TF::AreCastCompatible({output_ty, expected_ty}))
    return op.emitOpError()
           << "requires 'output' type " << output_ty
           << " to be cast compatible with expected type " << expected_ty;

  return success();
}

LogicalResult ReshapeOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attr, OpaqueProperties properties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ReshapeOpAdaptor op(operands, attr, properties);
  const Value input = op.getInput();
  const Value shape = op.getShape();

  auto error_handler = [&](const llvm::Twine& message) -> LogicalResult {
    // A dummy error handler.
    // Errors when computing the output shape will be raised in
    // ReshapeOp::verify call.
    return failure();
  };
  TensorType output_type;
  if (GetReshapeOutputType(input, shape, error_handler, output_type)
          .succeeded()) {
    inferredReturnTypes.assign({output_type});
    return success();
  }
  Type result_type;
  result_type = UnrankedTensorType::get(
      mlir::cast<ShapedType>(input.getType()).getElementType());
  inferredReturnTypes.assign({result_type});
  return success();
}

bool ReshapeOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size() || lhs.size() != 1) return false;
  if (failed(mlir::verifyCompatibleShape(lhs[0], rhs[0]))) return false;
  return true;
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

// Remove redundant unpack pack op.
// If a unpack op is followed by a pack op, we can remove the pack op, if the
// unpack op is only consumed by the pack op, it will be removed as well.
// An example illustration is:
//                  Unpack [5, 8, 9], axis = 1
//                /       \
//            value  ...  value [5, 9], 8 values in total
//              \           /
//                 pack,   axis = 1
//                   |
//               value   [5, 8, 9]
//
//   This can actually be simplified into just:
//
//           =>   Value [5, 8, 9]
// TODO(b/133341698): Move to tablegen when variadic is supported.
struct RemoveRedundantUnpackPack : public RewritePattern {
  explicit RemoveRedundantUnpackPack(MLIRContext* context)
      : RewritePattern(PackOp::getOperationName(), 2, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    TFL::PackOp pack_op = cast<TFL::PackOp>(op);
    Operation* first_input = pack_op.getOperand(0).getDefiningOp();
    if (!first_input) return failure();
    auto input_unpack_op = dyn_cast_or_null<TFL::UnpackOp>(first_input);
    if (!input_unpack_op) return failure();

    // The unpack & pack should have the same axis & num inputs/outputs.
    if (pack_op.getAxis() != input_unpack_op.getAxis() ||
        pack_op.getValuesCount() != input_unpack_op.getNum())
      return failure();

    const int total_pack_inputs = pack_op.getNumOperands();
    const int num_results = input_unpack_op.getNumResults();
    if (total_pack_inputs != num_results) return failure();
    for (auto input_output :
         llvm::zip(pack_op.getOperands(), input_unpack_op.getResults())) {
      Value pack_input = std::get<0>(input_output);
      Value unpack_output = std::get<1>(input_output);
      // Make sure the ordering is the same for the pack op & unpack op.
      if (pack_input != unpack_output) return failure();
    }

    // Replace the pack's output to the unpack's input.
    rewriter.replaceOp(pack_op, input_unpack_op.getOperand());
    // At this point, we don't manually remove the redundant pack op & unpack op
    // (we cannot actually), but trust the PatterRewriter to garbage collect
    // these two ops.
    return success();
  }
};

// Replace PackOp with a reshape when there is only one operand.
struct ReplacePackWithReshape : public RewritePattern {
  explicit ReplacePackWithReshape(MLIRContext* context)
      : RewritePattern(PackOp::getOperationName(), 2, context) {}
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    TFL::PackOp pack_op = cast<TFL::PackOp>(op);
    if (pack_op.getNumOperands() != 1) return failure();

    Location loc = pack_op.getLoc();
    auto output_type = mlir::cast<ShapedType>(pack_op.getType());
    if (!output_type.hasStaticShape()) return failure();

    // This is to workaround the unnecessary cast i64 -> i32.
    SmallVector<int32_t, 4> new_shape_array;
    for (auto size : output_type.getShape()) {
      new_shape_array.push_back(ConvertToTfliteSize(size));
    }

    auto new_shape = rewriter.create<TFL::ConstOp>(
        loc, DenseIntElementsAttr::get(
                 tensorflow::GetTypeFromTFTensorShape(
                     {static_cast<int64_t>(new_shape_array.size())},
                     rewriter.getIntegerType(32)),
                 new_shape_array));

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, output_type,
                                           pack_op.getOperand(0), new_shape);
    return success();
  }
};

void PackOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<RemoveRedundantUnpackPack, ReplacePackWithReshape>(context);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<int64_t> UnpackIndexOperand(Attribute operand) {
  auto data = cast<DenseIntElementsAttr>(operand);

  const bool is_i32 = data.getElementType().isSignlessInteger(32);

  if (data.isSplat()) {
    int64_t splat_val;

    if (is_i32) {
      splat_val = data.getSplatValue<int32_t>();
    } else {
      splat_val = data.getSplatValue<int64_t>();
    }

    return llvm::SmallVector<int64_t>(data.getShapedType().getNumElements(),
                                      splat_val);
  }

  if (!is_i32) {
    return llvm::to_vector(data.getValues<int64_t>());
  }

  return llvm::to_vector(llvm::map_range(data.getValues<int32_t>(), [](auto v) {
    return static_cast<int64_t>(v);
  }));
}

OpFoldResult SliceOp::fold(FoldAdaptor adaptor) {
  if (!getType().hasStaticShape()) {
    return {};
  }

  auto input_type = getInput().getType();
  if (input_type == getType()) {
    return getInput();
  }

  auto input = adaptor.getInput();
  if (!input) {
    return {};
  }
  auto begin = adaptor.getBegin();
  if (!begin) {
    return {};
  }
  auto size = adaptor.getSize();
  if (!size) {
    return {};
  }

  auto begin_type = getBegin().getType();
  auto size_type = getSize().getType();

  if (size_type.getRank() != 1 || begin_type.getRank() != 1) {
    return {};
  }

  if (begin_type.getDimSize(0) != input_type.getRank() ||
      size_type.getDimSize(0) != input_type.getRank()) {
    return {};
  }

  auto begin_vals = UnpackIndexOperand(begin);
  auto size_vals = UnpackIndexOperand(size);

  if (size_vals != getType().getShape()) {
    return {};
  }

  for (auto [begin, size, dim] :
       llvm::zip(begin_vals, size_vals, input_type.getShape())) {
    if (size < 0) {
      // TODO: b/351437662 - Add support for this case.
      return {};
    }
    if (begin + size > dim) {
      return {};
    }
  }

  auto input_dense = cast<DenseElementsAttr>(input);
  std::vector<Attribute> input_data(input_dense.value_begin<Attribute>(),
                                    input_dense.value_end<Attribute>());

  const FlatIndHelper write_inds(getType());
  const FlatIndHelper read_inds(input_type);

  std::vector<Attribute> result_data(getType().getNumElements());

  for (int w_flat_ind = 0; w_flat_ind < getType().getNumElements();
       ++w_flat_ind) {
    auto write_shaped_ind = write_inds.GetShapedInd(w_flat_ind);

    FlatIndHelper::AddOffset(write_shaped_ind, begin_vals);
    const auto read_flat_ind = read_inds.GetFlatInd(write_shaped_ind);

    result_data[w_flat_ind] = input_data[read_flat_ind];
  }

  return DenseElementsAttr::get(getType(), result_data);
}

mlir::LogicalResult SliceOp::verify() {
  SliceOp op = *this;
  auto input_type = mlir::cast<ShapedType>(op.getInput().getType());
  auto begin_type = mlir::cast<ShapedType>(op.getBegin().getType());
  auto size_type = mlir::cast<ShapedType>(op.getSize().getType());
  if (input_type.hasStaticShape() && begin_type.hasStaticShape() &&
      size_type.hasStaticShape()) {
    if (input_type.getRank() != begin_type.getNumElements()) {
      return op.emitError(
          "begin tensor elements size is not equal to input tensor rank");
    }

    if (input_type.getRank() != size_type.getNumElements()) {
      return op.emitError(
          "size tensor elements size is not equal to input tensor rank");
    }
  }

  DenseIntElementsAttr begin;
  if (matchPattern(op.getBegin(), m_Constant(&begin))) {
    int axis = 0;
    for (const auto& begin_i : llvm::enumerate(begin)) {
      if (begin_i.value().getSExtValue() < 0) {
        return op.emitError(
            llvm::formatv("begin[{0}] cannot be negative", axis));
      }
      axis++;
    }
  }

  DenseIntElementsAttr size;
  if (matchPattern(op.getSize(), m_Constant(&size))) {
    int axis = 0;
    for (const auto& size_i : llvm::enumerate(size)) {
      if (size_i.value().getSExtValue() < -1) {
        return op.emitError(
            llvm::formatv("size[{0}] cannot be negative other than -1", axis));
      }
      axis++;
    }
  }

  if (begin && size && input_type.hasStaticShape()) {
    for (uint64_t i = 0, end = begin.getNumElements(); i < end; i++) {
      int begin_i = begin.getValues<APInt>()[i].getSExtValue();
      int size_i = size.getValues<APInt>()[i].getSExtValue();
      int dim_i = input_type.getShape()[i];
      if (begin_i > dim_i) {
        return op.emitOpError(llvm::formatv(
            "begin[{0}] cannot exceed dimension length: {1}", i, dim_i));
      }
      if (size_i >= 0 && begin_i + size_i > dim_i) {
        return op.emitError(llvm::formatv(
            "begin[{0}] + size[{0}] cannot exceed dimension length: {1}", i,
            dim_i));
      }
    }
  }

  return success();
}

TFL::ConstOp NarrowDownInt64InputValuesForOp(Operation* input_op,
                                             RankedTensorType value_type,
                                             Location loc, OpBuilder* builder) {
  if (input_op == nullptr) return nullptr;

  mlir::DenseIntElementsAttr attr;
  if (!matchPattern(input_op, m_Constant(&attr))) {
    return nullptr;
  }

  auto value_shape_type = tensorflow::GetTypeFromTFTensorShape(
      value_type.getShape(), builder->getIntegerType(32));

  SmallVector<int32_t, 4> value_i32;
  value_i32.reserve(value_type.getRank());
  for (const auto& size : attr) {
    value_i32.push_back(ConvertToTfliteSize(size.getSExtValue()));
  }
  auto new_value_i32_attr =
      mlir::DenseIntElementsAttr::get(value_shape_type, value_i32);

  return builder->create<TFL::ConstOp>(loc, new_value_i32_attr);
}

// This will cast down int64 values for TFL slice op.
// This will require the begin & size are constants.
struct CastDonwInt64BeginEndToInt32 : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice_op,
                                PatternRewriter& rewriter) const override {
    auto begin = slice_op.getBegin();
    auto size = slice_op.getSize();
    auto begin_type = mlir::dyn_cast_or_null<RankedTensorType>(begin.getType());
    auto size_type = mlir::dyn_cast_or_null<RankedTensorType>(size.getType());
    auto begin_op = begin.getDefiningOp();
    auto size_op = size.getDefiningOp();

    if (begin_op == nullptr && size_op == nullptr) return failure();

    if (begin_type == nullptr && size_type == nullptr) return failure();

    // Handle begin.
    if (begin_op && begin_type && begin_type.getElementType().isInteger(64)) {
      auto new_begin = NarrowDownInt64InputValuesForOp(
          begin_op, begin_type, slice_op.getLoc(), &rewriter);
      if (new_begin != nullptr) {
        slice_op.setOperand(1, new_begin);
        return success();
      }
    }

    // Handle size.
    if (size_op && size_type && size_type.getElementType().isInteger(64)) {
      auto new_size = NarrowDownInt64InputValuesForOp(
          size_op, size_type, slice_op.getLoc(), &rewriter);
      if (new_size != nullptr) {
        slice_op.setOperand(2, new_size);
        return success();
      }
    }

    return failure();
  }
};

void SliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<CastDonwInt64BeginEndToInt32>(context);
}

//===----------------------------------------------------------------------===//
// SqueezeOp
//===----------------------------------------------------------------------===//

OpFoldResult SqueezeOp::fold(FoldAdaptor) {
  auto input_ty = mlir::dyn_cast<RankedTensorType>(getInput().getType());
  auto result_ty = mlir::dyn_cast<RankedTensorType>(getType());

  if (!input_ty || !result_ty) return {};
  if (input_ty == result_ty) return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // TODO(b/142478136): Handle fused ops.
  if (getFusedActivationFunction() != "NONE") return {};

  auto lhs = llvm::dyn_cast_or_null<DenseElementsAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<DenseElementsAttr>(adaptor.getRhs());

  if (lhs && rhs) {
    return ConstFoldBinaryOp(
        getType(), operands, [](APFloat a, APFloat b) { return a - b; },
        [](APInt a, APInt b) { return a - b; });
  }

  auto is_zero = [](Attribute a) {
    return matchPattern(a, m_Zero()) || matchPattern(a, m_AnyZeroFloat());
  };

  if (llvm::isa<quant::QuantizedType>(getType().getElementType())) {
    // Quant types not supported for the following.
    return {};
  }

  if (rhs && is_zero(rhs) && getLhs().getType() == getType()) {
    return getLhs();
  }

  return {};
}

int64_t SubOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count)) return count;

  return -1;
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

static void BuildTopKOp(OpBuilder* builder, OperationState& result, Value input,
                        Value k) {
  // Output size is only known if k is constant value. A negative dimension is
  // considered dynamic so use -1 here if k is not a constant value.
  int const_k = -1;
  ElementsAttr cst;
  if (matchPattern(k, m_Constant(&cst)))
    // These casts should all be valid due to how Tensor constants are stored.
    // TODO(jpienaar): This should use a helper function.
    const_k = cst.getValues<IntegerAttr>()[0].getValue().getSExtValue();

  auto val_type = mlir::cast<TensorType>(input.getType());
  // If value is unranked, then so is results.
  if (!val_type.hasRank())
    return TFL::TopKV2Op::build(
        *builder, result, UnrankedTensorType::get(val_type.getElementType()),
        UnrankedTensorType::get(k.getType()), input, k);

  // Resultant shape is value.shape[:-1] + [k]
  std::vector<int64_t> shape(val_type.getShape());
  shape[shape.size() - 1] = const_k;
  TFL::TopKV2Op::build(
      *builder, result,
      tensorflow::GetTypeFromTFTensorShape(shape, val_type.getElementType()),
      tensorflow::GetTypeFromTFTensorShape(shape, k.getType()), input, k);
}

//===----------------------------------------------------------------------===//
// FakeQuantOp
//===----------------------------------------------------------------------===//

// Return true if the op has non-empty "minmax" attribute.
static inline bool HasValidMinMaxAttribute(Operation* op) {
  auto minmax = op->getAttrOfType<ArrayAttr>("minmax");
  return minmax && minmax.getValue().size() == 2;
}

namespace {

/// This pattern matches and remove a tfl.fake_quant if all the users of this op
/// and itself have "minmax" attribute set.
struct DropFakeQuant : public RewritePattern {
  explicit DropFakeQuant(MLIRContext* context)
      : RewritePattern(FakeQuantOp::getOperationName(), 1, context) {}

  LogicalResult match(Operation* op) const override {
    // We only match the op with valid "minmax" attribute.
    if (!HasValidMinMaxAttribute(op)) return failure();

    // If all the users of this op have valid "minmax" attributes, it is matched
    // and can be removed.
    auto fakeQuantOp = cast<FakeQuantOp>(op);
    for (auto* operand : fakeQuantOp.getResult().getUsers())
      if (!HasValidMinMaxAttribute(operand)) return failure();

    return success();
  }

  void rewrite(Operation* op, PatternRewriter& rewriter) const override {
    // Replace the matched FakeQuantOp by its primary operand.
    rewriter.replaceOp(op, op->getOperand(0));
  }
};
}  // end anonymous namespace

void FakeQuantOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<DropFakeQuant>(context);
}

//===----------------------------------------------------------------------===//
// UnpackOp
//===----------------------------------------------------------------------===//

// TODO(b/133486129): Implement shape inference for unpack

LogicalResult UnpackOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  UnpackOpAdaptor op(operands, attributes, properties);
  // TODO(jpienaar): Refactor verify
  if (failed(op.verify(loc.has_value() ? *loc : UnknownLoc::get(context))))
    return failure();

  if (operands.size() != 1) {
    return emitOptionalError(loc, "input count should be equal to 1");
  }

  const int64_t num_value = op.getNumAttr().getInt();
  auto input_type = mlir::dyn_cast<ShapedType>(operands[0].getType());
  if (!input_type || !input_type.hasRank()) {
    // If input is unranked, then so is output.
    inferredReturnTypes.assign(
        num_value, UnrankedTensorType::get(input_type.getElementType()));
    return success();
  }

  if (input_type.hasStaticShape() && input_type.getNumElements() <= 0) {
    return emitOptionalError(
        loc, "number of elements in input should be larger than 0");
  }

  const int64_t rank = input_type.getRank();
  if (rank <= 0) {
    return emitOptionalError(loc, "input should be of rank larger than 0");
  }

  int64_t axis_value = op.getAxisAttr().getInt();
  if (axis_value < 0) {
    axis_value += rank;
  }
  if (axis_value < 0 || axis_value >= rank) {
    return emitOptionalError(
        loc, "attribute 'axis' should be in range [-rank, rank), got axis = ",
        op.getAxisAttr().getInt(), ", and rank = ", rank);
  }

  if (!ShapedType::isDynamic(input_type.getDimSize(axis_value)) &&
      input_type.getDimSize(axis_value) != num_value) {
    return emitOptionalError(loc, "output count should match 'num' attribute");
  }

  auto output_shape = llvm::to_vector<4>(input_type.getShape());
  output_shape.erase(output_shape.begin() + axis_value);

  auto output_type = tensorflow::GetTypeFromTFTensorShape(
      output_shape, input_type.getElementType());
  inferredReturnTypes.assign(num_value, output_type);

  return success();
}

bool UnpackOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto pair : llvm::zip(lhs, rhs)) {
    if (failed(
            mlir::verifyCompatibleShape(std::get<0>(pair), std::get<1>(pair))))
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

// Extracts and returns the signed integer constant in a 0-rank integer tensor
// or 1-element 1-rank integer tensor if 'value' is a constant.
static std::optional<int64_t> ExtractConstantIntFromTensor(Value value) {
  ElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) return {};
  if (attr.getNumElements() != 1) return {};
  IntegerAttr int_attr = *attr.getValues<IntegerAttr>().begin();
  return int_attr.getValue().getSExtValue();
}

// Returns a RankedTensorType which is similar to `input_type` but replaces the
// dimension size of `dim` with `dim_size`.  For example,
// `SubstituteRankedTensorTypeDimSize(tensor<3x4xi32>, 1, 2)` returns
// `tensor<3x2xi32>`.
static RankedTensorType SubstituteRankedTensorTypeDimSize(
    RankedTensorType input_type, int64_t dim, int64_t dim_size) {
  auto shape = input_type.getShape().vec();
  shape[dim] = dim_size;
  return tensorflow::GetTypeFromTFTensorShape(shape,
                                              input_type.getElementType());
}

// Verifies the output tensor types of SplitOp or SplitVOp.
template <typename ExpectedOutputTypeGetter>
static LogicalResult VerifySplitOpOutputTypes(
    Operation* op, int64_t num_splits,
    ExpectedOutputTypeGetter get_expected_output_type) {
  for (int64_t i = 0; i < num_splits; ++i) {
    auto expected_output_type = get_expected_output_type(i);
    Value output = op->getResult(i);
    if (failed(verifyCompatibleShape(output.getType(), expected_output_type)))
      return op->emitOpError()
             << "output #" << i << " should be " << expected_output_type
             << " instead got " << output.getType();
  }
  return success();
}

mlir::LogicalResult SplitOp::verify() {
  SplitOp op = *this;
  int64_t num_splits = op.getNumSplits();
  if (op.getNumResults() != num_splits)
    return op.emitOpError("output count should match 'num_splits' attribute");

  // If 'split_dim' is not a constant, there are no other checks.
  std::optional<int64_t> split_dim_opt =
      ExtractConstantIntFromTensor(op.getSplitDim());
  if (!split_dim_opt) return success();

  // If 'input' is not a ranked tensor, there are no other checks.
  auto input_type = mlir::dyn_cast<RankedTensorType>(op.getValue().getType());
  if (!input_type) return success();

  int64_t split_dim = split_dim_opt.value();
  const int64_t rank = input_type.getRank();
  if (split_dim < 0) split_dim += rank;
  if (split_dim < 0 || split_dim >= rank)
    return op.emitOpError("'split_dim' should be in [-rank, rank)");

  // If the 'split_dim' dimension of the 'input' tensor has a dynamic size,
  // there are no other checks.
  const int64_t dim_size = input_type.getDimSize(split_dim);
  if (ShapedType::isDynamic(dim_size)) return success();

  if (dim_size % num_splits != 0)
    return op.emitOpError("'num_splits' should evenly divide 'split_dim' axis");

  // Verifies output tensor types.
  RankedTensorType expected_output_type = SubstituteRankedTensorTypeDimSize(
      input_type, split_dim, dim_size / num_splits);
  return VerifySplitOpOutputTypes(
      op.getOperation(), num_splits,
      [expected_output_type](int64_t) { return expected_output_type; });
}

mlir::LogicalResult SplitVOp::verify() {
  SplitVOp op = *this;
  int64_t num_splits = op.getNumSplits();
  if (op.getNumResults() != num_splits)
    return op.emitOpError("output count should match 'num_splits' attribute");

  // If 'split_dim' is not a constant, there are no other checks.
  std::optional<int64_t> split_dim_opt =
      ExtractConstantIntFromTensor(op.getSplitDim());
  if (!split_dim_opt) return success();

  // If 'input' is not a ranked tensor, there are no other checks.
  auto input_type = mlir::dyn_cast<RankedTensorType>(op.getValue().getType());
  if (!input_type) return success();

  int64_t split_dim = split_dim_opt.value();
  const int64_t rank = input_type.getRank();
  if (split_dim < 0) split_dim += rank;
  if (split_dim < 0 || split_dim >= rank)
    return op.emitOpError("'split_dim' should be in [-rank, rank)");

  // If the 'split_dim' dimension of the 'input' tensor has a dynamic size,
  // there are no other checks.
  const int64_t dim_size = input_type.getDimSize(split_dim);
  if (ShapedType::isDynamic(dim_size)) return success();

  // If 'size_splits' is not a constant, there are no other checks.
  ElementsAttr size_splits_attr;
  if (!matchPattern(op.getSizeSplits(), m_Constant(&size_splits_attr)))
    return success();

  if (size_splits_attr.getNumElements() != num_splits) {
    auto size_splits_type =
        mlir::cast<RankedTensorType>(op.getSizeSplits().getType());
    RankedTensorType expected_size_splits_type =
        tensorflow::GetTypeFromTFTensorShape({num_splits},
                                             size_splits_type.getElementType());
    return op.emitOpError("'size_splits' should be ")
           << expected_size_splits_type;
  }

  // Normalizes and verifies 'size_splits'.
  // Note: TensorFlow allows one -1 element in 'size_splits'.  The -1 element
  // means the rest of the dimension size.
  llvm::SmallVector<int64_t, 4> size_splits;
  size_splits.reserve(num_splits);

  int64_t negative_size_split_loc = -1;
  int64_t total_size_splits = 0;

  for (int64_t i = 0; i < num_splits; ++i) {
    auto size_split_attr = size_splits_attr.getValues<IntegerAttr>()[i];
    int64_t size_split = size_split_attr.getValue().getSExtValue();
    size_splits.push_back(size_split);
    if (size_split >= 0) {
      total_size_splits += size_split;
      continue;
    }
    if (size_split < -1)
      return op.emitOpError(
          "elements of 'size_splits' should be greater than or equal to -1");
    if (negative_size_split_loc != -1)
      return op.emitOpError("'size_splits' can only have one -1");
    negative_size_split_loc = i;
  }

  if (negative_size_split_loc != -1) {
    if (total_size_splits > dim_size)
      return op.emitOpError(
          "sum of non-negative elements of 'size_splits' is greater than the "
          "dimension size of 'split_dim' axis");
    size_splits[negative_size_split_loc] = dim_size - total_size_splits;
    total_size_splits = dim_size;
  }

  if (total_size_splits != dim_size)
    return op.emitOpError(
        "sum of 'size_splits' should match the dimension size of 'split_dim' "
        "axis");

  // Verifies result tensor types.
  auto get_expected_output_type = [input_type, split_dim,
                                   &size_splits](int64_t i) {
    return SubstituteRankedTensorTypeDimSize(input_type, split_dim,
                                             size_splits[i]);
  };
  return VerifySplitOpOutputTypes(op.getOperation(), num_splits,
                                  get_expected_output_type);
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

// TODO(b/133854225): Implement shape inference to Mean

//===----------------------------------------------------------------------===//
// LSTMOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult LSTMOp::verify() {
  LSTMOp op = *this;
  auto operands = op.GetStatefulOperands();
  if (operands.size() != 2 || operands[0] != 18 || operands[1] != 19) {
    return op.emitOpError("LSTMOp expected to have two stateful operands");
  }

  const auto input_type = mlir::cast<ShapedType>(op.getInput().getType());
  // Since TFLite runtime generally supports dynamic shape/rank, if `input_type`
  // doesn't have static shape, we skip the shape check below.
  if (!input_type.hasStaticShape()) return success();
  // The input should be at least 2D tensor since it will go through fully
  // connected layer.
  if (!input_type.hasRank() || input_type.getRank() < 2)
    return op.emitOpError(
        "the first input operand should have more than 2 dimensions.");

  const auto activation_state =
      mlir::cast<ShapedType>(op.getInputActivationState().getType());
  const auto cell_state =
      mlir::cast<ShapedType>(op.getInputCellState().getType());
  const auto input_to_output_weights =
      mlir::cast<ShapedType>(op.getInputToOutputWeights().getType());
  const auto recurrent_to_output_weights =
      mlir::cast<ShapedType>(op.getRecurrentToOutputWeights().getType());
  if (activation_state.hasStaticShape() && cell_state.hasStaticShape() &&
      input_to_output_weights.hasStaticShape() &&
      recurrent_to_output_weights.hasStaticShape()) {
    const int n_input = input_type.getDimSize(input_type.getRank() - 1);
    const int n_cell = input_to_output_weights.getDimSize(0);
    const int n_output = recurrent_to_output_weights.getDimSize(1);
    const int output_state_size = activation_state.getNumElements();
    const int n_batch = input_type.getRank() == 2 ? input_type.getDimSize(0)
                                                  : input_type.getDimSize(1);
    const int state_size = cell_state.getNumElements();

    // Check if the dimension of the inputs matches.
    if ((output_state_size != n_batch * n_output) ||
        (state_size != n_batch * n_cell) ||
        (input_to_output_weights.getDimSize(1) != n_input) ||
        (recurrent_to_output_weights.getRank() != 2) ||
        (recurrent_to_output_weights.getDimSize(0) != n_cell) ||
        (input_to_output_weights.getRank() != 2)) {
      return op.emitOpError("inputs don't match with the dimensions.");
    }

    const bool is_layer_norm_lstm =
        !mlir::isa<NoneType>(op.getForgetLayerNormCoefficients().getType());
    if (is_layer_norm_lstm) {
      const auto forget_layer_norm_coefficients =
          mlir::cast<ShapedType>(op.getForgetLayerNormCoefficients().getType());
      // If this lstm has layer normalization, this input value,
      // "forget_layer_norm_coefficients" should be a 1D tensor.
      if (!forget_layer_norm_coefficients.hasRank() ||
          forget_layer_norm_coefficients.getRank() != 1 ||
          forget_layer_norm_coefficients.getDimSize(0) != n_cell)
        return op.emitOpError(
            "coefficient inputs have more than 2 dimensions or "
            "don't match the dimension with input operand "
            "`input_to_output_weights`.");
    }
  }

  return success();
}

namespace {

// Replaces the optional bias operands with a "none" type value if the bias
// values are constant zeros.
struct RemoveLSTMOpZeroBias : public OpRewritePattern<LSTMOp> {
  using OpRewritePattern<LSTMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LSTMOp op,
                                PatternRewriter& rewriter) const override {
    if (EqualsZero(op.getInputGateBias())) {
      auto none_value = rewriter.create<TFL::NoValueOp>(
          rewriter.getUnknownLoc(), rewriter.getNoneType(),
          rewriter.getUnitAttr());
      op.getInputGateBiasMutable().assign(none_value);
      return success();
    }

    if (EqualsZero(op.getProjectionBias())) {
      auto none_value = rewriter.create<TFL::NoValueOp>(
          rewriter.getUnknownLoc(), rewriter.getNoneType(),
          rewriter.getUnitAttr());
      op.getProjectionBiasMutable().assign(none_value);
      return success();
    }

    return failure();
  }
};

}  // namespace

void LSTMOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<RemoveLSTMOpZeroBias>(context);
}

//===----------------------------------------------------------------------===//
// UnidirectionalSequenceLSTMOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult UnidirectionalSequenceLSTMOp::verify() {
  UnidirectionalSequenceLSTMOp op = *this;
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 2 && operands[0] == 18 && operands[1] == 19) {
    return success();
  }
  return op.emitError(
      "UnidirectionalSequenceLSTMOp expected to have two stateful operands");
}

LogicalResult UnidirectionalSequenceLSTMOp::inferReturnTypes(
    MLIRContext*, std::optional<Location>, ValueRange operands,
    DictionaryAttr attr, OpaqueProperties properties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  Value input = operands[0];
  auto input_type = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());

  Value output_state = operands[18];
  auto output_state_type =
      mlir::dyn_cast_or_null<RankedTensorType>(output_state.getType());

  if (input_type && input_type.hasRank() && input_type.getRank() != 3) {
    return failure();
  }

  if (output_state_type && output_state_type.hasRank() &&
      output_state_type.getRank() != 2) {
    return failure();
  }

  if (!input_type || !input_type.hasRank() || !output_state_type ||
      !output_state_type.hasRank()) {
    // We cannot infer the output shape since we don't know the input shape or
    // the output state shape. We will set the output shape as unranked.
    Type result_type;
    result_type = UnrankedTensorType::get(
        mlir::cast<ShapedType>(input.getType()).getElementType());
    inferredReturnTypes.assign({result_type});
    return success();
  }

  // Default to non-time_major.
  std::optional<mlir::NamedAttribute> time_major_attr =
      attr.getNamed("time_major");
  bool time_majored =
      time_major_attr
          ? mlir::cast<BoolAttr>(time_major_attr->getValue()).getValue()
          : false;

  int64_t batch =
      time_majored ? input_type.getDimSize(1) : input_type.getDimSize(0);
  int64_t time =
      time_majored ? input_type.getDimSize(0) : input_type.getDimSize(1);
  int64_t n_output = output_state_type.getDimSize(1);

  // Build the output shape.
  SmallVector<int64_t, 3> output_shape;
  if (time_majored) {
    output_shape = {time, batch, n_output};
  } else {
    output_shape = {batch, time, n_output};
  }
  auto result_type = tensorflow::GetTypeFromTFTensorShape(
      output_shape, input_type.getElementType());

  inferredReturnTypes.assign({result_type});
  return success();
}

bool UnidirectionalSequenceLSTMOp::isCompatibleReturnTypes(TypeRange lhs,
                                                           TypeRange rhs) {
  if (lhs.size() != rhs.size() || lhs.size() != 1) return false;
  if (failed(mlir::verifyCompatibleShape(lhs[0], rhs[0]))) return false;
  return true;
}

//===----------------------------------------------------------------------===//
// BidirectionalSequenceLSTMOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult BidirectionalSequenceLSTMOp::verify() {
  BidirectionalSequenceLSTMOp op = *this;
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 4 && operands[0] == 35 && operands[1] == 36 &&
      operands[2] == 37 && operands[3] == 38) {
    return success();
  }
  return op.emitError(
      "BidirectionalSequenceLSTMOp expected to have four stateful operands");
}

//===----------------------------------------------------------------------===//
// UnidirectionalSequenceRNNOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult UnidirectionalSequenceRNNOp::verify() {
  UnidirectionalSequenceRNNOp op = *this;
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 1 && operands[0] == 4) {
    return success();
  }
  return op.emitError(
      "UnidirectionalSequenceRNNOp expected to have one stateful operand");
}

//===----------------------------------------------------------------------===//
// SvdfOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult SVDFOp::verify() {
  SVDFOp op = *this;
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 1 && operands[0] == 4) {
    return success();
  }
  return op.emitError("SvdfOp expected to have one stateful operand");
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

OpFoldResult AbsOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat { return llvm::abs(value); };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat { return llvm::neg(value); };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

OpFoldResult SinOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::sin(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

OpFoldResult CosOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::cos(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

OpFoldResult LogOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::log(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

OpFoldResult ShapeOp::fold(FoldAdaptor) {
  auto input_type = mlir::cast<ShapedType>(getInput().getType());
  if (!input_type.hasStaticShape()) return nullptr;

  ArrayRef<int64_t> shape = input_type.getShape();
  auto result_type = mlir::cast<ShapedType>(getType());
  if (result_type.getElementType().isInteger(64)) {
    return DenseElementsAttr::get<int64_t>(result_type, shape);
  } else if (result_type.getElementType().isInteger(32)) {
    SmallVector<int32_t, 4> shape_i32;
    shape_i32.reserve(shape.size());
    for (int64_t dim : shape) {
      shape_i32.push_back(dim);
    }
    return DenseElementsAttr::get<int32_t>(result_type, shape_i32);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult SqrtOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::sqrt(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult RsqrtOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32/bf16 is implemented.
  if (!IsF32ShapedType(result_type) && !IsBF16ShapedType(result_type))
    return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    bool loseInfo;
    const llvm::fltSemantics& original_float_semantics = value.getSemantics();
    value.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                  &loseInfo);
    float f = value.convertToFloat();
    APFloat result(1.f / std::sqrt(f));
    result.convert(original_float_semantics, APFloat::rmNearestTiesToEven,
                   &loseInfo);
    return result;
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// SquareOp
//===----------------------------------------------------------------------===//

OpFoldResult SquareOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat { return value * value; };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

template <typename T>
T ComputeRelu(T val) {
  return std::max(static_cast<T>(0), val);
}

// TODO: b/361137571 - Add folding for quantized types if it is needed.
OpFoldResult ReluOp::fold(FoldAdaptor adaptor) {
  auto data = mlir::dyn_cast_or_null<DenseElementsAttr>(adaptor.getX());
  if (!data) {
    return {};
  }

  if (getType().getElementType().isSignlessInteger(32)) {
    return DenseIntElementsAttr::get(
        data.getType(),
        llvm::map_to_vector(data.getValues<int32_t>(), ComputeRelu<int32_t>));
  }
  if (getType().getElementType().isF32()) {
    return DenseFPElementsAttr::get(
        data.getType(),
        llvm::map_to_vector(data.getValues<float>(), ComputeRelu<float>));
  }

  return {};
}

//===----------------------------------------------------------------------===//
// MaximumOp
//===----------------------------------------------------------------------===//

OpFoldResult MaximumOp::fold(FoldAdaptor adaptor) {
  auto lhs_type = getLhs().getType();
  auto rhs_type = getRhs().getType();

  auto lhs = mlir::dyn_cast_or_null<DenseElementsAttr>(adaptor.getLhs());
  auto rhs = mlir::dyn_cast_or_null<DenseElementsAttr>(adaptor.getRhs());

  if (lhs && rhs) {
    return ConstFoldBinaryOp(
        getType(), adaptor.getOperands(),
        [](APFloat a, APFloat b) { return llvm::maximum(a, b); },
        [](APInt a, APInt b) { return a.slt(b) ? b : a; });
  }
  // Only constant fold for float tensors of the same type is implemented.
  if (lhs_type != rhs_type || !IsFloatShapedType(lhs_type)) return nullptr;

  if (lhs && lhs.isSplat()) {
    APFloat lhs_value = lhs.getSplatValue<APFloat>();
    lhs_value.changeSign();
    if (lhs_value.isLargest() || lhs_value.isInfinity()) return getRhs();
  }
  if (rhs && rhs.isSplat()) {
    APFloat rhs_value = rhs.getSplatValue<APFloat>();
    rhs_value.changeSign();
    if (rhs_value.isLargest() || rhs_value.isInfinity()) return getLhs();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// MinimumOp
//===----------------------------------------------------------------------===//

OpFoldResult MinimumOp::fold(FoldAdaptor adaptor) {
  auto lhs_type = getLhs().getType();
  auto rhs_type = getRhs().getType();

  auto lhs = mlir::dyn_cast_or_null<DenseElementsAttr>(adaptor.getLhs());
  auto rhs = mlir::dyn_cast_or_null<DenseElementsAttr>(adaptor.getRhs());
  if (lhs && rhs) {
    return ConstFoldBinaryOp(
        getType(), adaptor.getOperands(),
        [](APFloat a, APFloat b) { return llvm::minimum(a, b); },
        [](APInt a, APInt b) { return a.slt(b) ? a : b; });
  }

  if (lhs_type != rhs_type || !IsFloatShapedType(lhs_type)) return nullptr;

  if (lhs && lhs.isSplat()) {
    auto splat = lhs.getSplatValue<APFloat>();
    if (splat.isLargest() || splat.isInfinity()) return getRhs();
  }
  if (rhs && rhs.isSplat()) {
    auto splat = rhs.getSplatValue<APFloat>();
    if (splat.isLargest() || splat.isInfinity()) return getLhs();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Comparison and Logical Ops
//===----------------------------------------------------------------------===//

OpFoldResult LessOp::fold(FoldAdaptor adaptor) {
  if (getLhs().getType().getElementType().isInteger(32)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int32_t, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int32_t lhs, int32_t rhs) { return lhs < rhs; });
  }
  if (getLhs().getType().getElementType().isInteger(64)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int64_t, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int64_t lhs, int64_t rhs) { return lhs < rhs; });
  }
  if (getLhs().getType().getElementType().isF32()) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, float, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](float lhs, float rhs) { return lhs < rhs; });
  }
  return {};
}

OpFoldResult LessEqualOp::fold(FoldAdaptor adaptor) {
  if (getLhs().getType().getElementType().isInteger(32)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int32_t, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int32_t lhs, int32_t rhs) { return lhs <= rhs; });
  }
  if (getLhs().getType().getElementType().isInteger(64)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int64_t, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int64_t lhs, int64_t rhs) { return lhs <= rhs; });
  }
  if (getLhs().getType().getElementType().isF32()) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, float, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](float lhs, float rhs) { return lhs <= rhs; });
  }
  return {};
}

OpFoldResult GreaterOp::fold(FoldAdaptor adaptor) {
  if (getLhs().getType().getElementType().isInteger(32)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int32_t, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int32_t lhs, int32_t rhs) { return lhs > rhs; });
  }
  if (getLhs().getType().getElementType().isInteger(64)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int64_t, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int64_t lhs, int64_t rhs) { return lhs > rhs; });
  }
  if (getLhs().getType().getElementType().isF32()) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, float, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](float lhs, float rhs) { return lhs > rhs; });
  }
  return {};
}

OpFoldResult GreaterEqualOp::fold(FoldAdaptor adaptor) {
  if (getLhs().getType().getElementType().isInteger(32)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int32_t, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int32_t lhs, int32_t rhs) { return lhs >= rhs; });
  }
  if (getLhs().getType().getElementType().isInteger(64)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int64_t, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int64_t lhs, int64_t rhs) { return lhs >= rhs; });
  }
  if (getLhs().getType().getElementType().isF32()) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, float, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](float lhs, float rhs) { return lhs >= rhs; });
  }
  return {};
}

OpFoldResult EqualOp::fold(FoldAdaptor adaptor) {
  if (getX().getType().getElementType().isInteger(32)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int32_t, bool>(
        getType(), adaptor.getX(), adaptor.getY(),
        [](int32_t lhs, int32_t rhs) { return lhs == rhs; });
  }
  if (getX().getType().getElementType().isInteger(64)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int64_t, bool>(
        getType(), adaptor.getX(), adaptor.getY(),
        [](int64_t lhs, int64_t rhs) { return lhs == rhs; });
  }
  if (getX().getType().getElementType().isF32()) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, float, bool>(
        getType(), adaptor.getX(), adaptor.getY(),
        [](float lhs, float rhs) { return lhs == rhs; });
  }
  return {};
}

OpFoldResult NotEqualOp::fold(FoldAdaptor adaptor) {
  if (getLhs().getType().getElementType().isInteger(32)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int32_t, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int32_t lhs, int32_t rhs) { return lhs != rhs; });
  }
  if (getLhs().getType().getElementType().isInteger(64)) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, int64_t, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](int64_t lhs, int64_t rhs) { return lhs != rhs; });
  }
  if (getLhs().getType().getElementType().isF32()) {
    return ConstFoldBinaryOp<DenseIntElementsAttr, float, bool>(
        getType(), adaptor.getLhs(), adaptor.getRhs(),
        [](float lhs, float rhs) { return lhs != rhs; });
  }
  return {};
}

OpFoldResult LogicalAndOp::fold(FoldAdaptor adaptor) {
  return ConstFoldBinaryOp<DenseIntElementsAttr, bool, bool>(
      getType(), adaptor.getLhs(), adaptor.getRhs(),
      [](bool lhs, bool rhs) { return lhs && rhs; });
}

OpFoldResult LogicalOrOp::fold(FoldAdaptor adaptor) {
  return ConstFoldBinaryOp<DenseIntElementsAttr, bool, bool>(
      getType(), adaptor.getLhs(), adaptor.getRhs(),
      [](bool lhs, bool rhs) { return lhs || rhs; });
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// TODO: b/359275356 - Expand this to handle the broadcast case similar
// to `ConstFoldBinaryOpDense`.
OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  auto lhs_type = getX().getType();
  auto rhs_type = getY().getType();
  auto condition_type = getCondition().getType();
  auto out_type = getType();

  if (lhs_type != rhs_type) {
    return {};
  }

  if (lhs_type.getShape() != condition_type.getShape()) {
    // "broadcasted" condition not yet supported.
    return {};
  }

  auto condition_vals =
      llvm::dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getCondition());
  if (!condition_vals || !condition_vals.getElementType().isInteger(1)) {
    return {};
  }

  if (condition_vals.isSplat()) {
    const bool val = condition_vals.getSplatValue<bool>();
    return val ? adaptor.getX() : adaptor.getY();
  }

  auto lhs_vals = llvm::dyn_cast_or_null<DenseElementsAttr>(adaptor.getX());
  auto rhs_vals = llvm::dyn_cast_or_null<DenseElementsAttr>(adaptor.getY());
  if (!lhs_vals || !rhs_vals) {
    return {};
  }

  llvm::SmallVector<Attribute> results;
  results.reserve(condition_type.getNumElements());

  auto lhs_it = lhs_vals.getValues<Attribute>().begin();
  auto lhs_end = lhs_vals.getValues<Attribute>().end();
  auto rhs_it = rhs_vals.getValues<Attribute>().begin();
  auto rhs_end = rhs_vals.getValues<Attribute>().end();

  auto condition_it = condition_vals.getValues<bool>().begin();
  auto condition_end = condition_vals.getValues<bool>().end();

  while (condition_it < condition_end && lhs_it < lhs_end && rhs_it < rhs_end) {
    if (*condition_it++) {
      results.push_back(*lhs_it);
    } else {
      results.push_back(*rhs_it);
    }

    if (!lhs_vals.isSplat()) {
      lhs_it++;
    }
    if (!rhs_vals.isSplat()) {
      rhs_it++;
    }
  }

  return DenseElementsAttr::get(out_type, results);
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

static size_t GetOutputFlatIndex(size_t input_flat_ind,
                                 const FlatIndHelper& input_flat_ind_helper,
                                 const FlatIndHelper& output_flat_ind_helper,
                                 const absl::flat_hash_set<int32_t>& axes_set,
                                 bool keep_dims) {
  auto input_shaped_ind = input_flat_ind_helper.GetShapedInd(input_flat_ind);

  std::vector<int64_t> output_shaped_ind;
  for (int dim = 0; dim < input_shaped_ind.size(); ++dim) {
    bool is_reduced_dim = axes_set.contains(dim);
    if (is_reduced_dim && keep_dims) {
      output_shaped_ind.push_back(0);
    } else if (!is_reduced_dim) {
      output_shaped_ind.push_back(input_shaped_ind[dim]);
    }
  }

  return output_flat_ind_helper.GetFlatInd(output_shaped_ind);
}

OpFoldResult SumOp::fold(FoldAdaptor adaptor) {
  auto input = adaptor.getInput();
  auto axes = adaptor.getAxes();

  if (!input || !axes) {
    return {};
  }

  auto input_type = getInput().getType();
  if (!input_type.getElementType().isF32()) {
    return {};
  }

  const auto input_rank = input_type.getRank();
  auto axes_data = llvm::cast<DenseIntElementsAttr>(axes);
  absl::flat_hash_set<int32_t> axes_set(axes_data.getValues<int32_t>().begin(),
                                        axes_data.getValues<int32_t>().end());

  llvm::SmallVector<int64_t> out_shape;
  for (int input_dim = 0; input_dim < input_rank; ++input_dim) {
    bool is_reduced_dim = axes_set.contains(input_dim);

    if (is_reduced_dim && adaptor.getKeepDims()) {
      out_shape.push_back(1);
    } else if (!is_reduced_dim) {
      out_shape.push_back(input_type.getDimSize(input_dim));
    }
  }

  auto out_type = RankedTensorType::get(out_shape, input_type.getElementType());
  int64_t output_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                        std::multiplies<int64_t>());

  std::vector<float> out_data(output_size, 0.0);
  FlatIndHelper input_flat_ind_helper(input_type);
  FlatIndHelper output_flat_ind_helper(out_type);

  auto in_data = llvm::cast<DenseFPElementsAttr>(input);

  size_t input_flat_ind = 0;
  for (auto it = in_data.value_begin<float>(); it < in_data.value_end<float>();
       ++it) {
    // Find the corresponding reduced output index.
    size_t output_flat_ind = GetOutputFlatIndex(
        input_flat_ind, input_flat_ind_helper, output_flat_ind_helper, axes_set,
        adaptor.getKeepDims());

    out_data[output_flat_ind] += *it;
    input_flat_ind++;
  }

  return DenseFPElementsAttr::get(out_type, out_data);
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

OpFoldResult RankOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.size() == 1);
  auto result_type = mlir::cast<ShapedType>(getType());
  if (auto elements_attr = mlir::dyn_cast_or_null<ElementsAttr>(operands[0])) {
    auto rank = static_cast<int32_t>(elements_attr.getShapedType().getRank());
    return DenseElementsAttr::get(result_type, {rank});
  }

  // Also fold if `input` has a known rank.
  auto input_type = mlir::cast<ShapedType>(getInput().getType());
  // Do not fold if rank is zero because the TFLite converter doesn't
  // distinguish between unranked input and scalar input due to b/138865275.
  // TODO(b/138865275): Remove `input_type.getRank() != 0` in the following
  // predicate and fold the op when rank is zero.
  if (input_type.hasRank() && input_type.getRank() != 0) {
    auto rank = static_cast<int32_t>(input_type.getRank());
    return DenseElementsAttr::get(result_type, {rank});
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  (void)operands;
  assert(operands.empty() && "constant has no operands");
  // Return the held attribute value.
  return getValue();
}

bool ConstOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  // Allow the type inferred to not match exactly the inferred type as the
  // inferred type is from the element attribute's type while the op may have
  // gotten constructed from TF const op or be in a partial state of shape
  // refinement, so allow it to only be compatible. The op will be refined
  // during shape inference and casts inserted as needed to satisfy type
  // constraints of consumers.
  return succeeded(verifyCompatibleShapes(l, r));
}

namespace {
struct FoldPseudoConstOp : public OpRewritePattern<ConstOp> {
  using OpRewritePattern<ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstOp const_op,
                                PatternRewriter& rewriter) const override {
    if (arith::ConstantOp::isBuildableWith(const_op.getValue(),
                                           const_op.getType())) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(const_op,
                                                     const_op.getValue());
      return success();
    } else if (TFL::NoValueOp::isBuildableWith(const_op.getValue(),
                                               const_op.getType())) {
      rewriter.replaceOpWithNewOp<NoValueOp>(
          const_op, rewriter.getNoneType(),
          mlir::cast<UnitAttr>(const_op.getValue()));
      return success();
    }
    return failure();
  }
};

}  // namespace

void ConstOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<FoldPseudoConstOp>(context);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

OpFoldResult CastIntToInt(DenseIntElementsAttr data, IntegerType in_type,
                          IntegerType out_type) {
  const bool is_unsigned = in_type.isUnsigned();
  const bool involves_bool =
      in_type.getWidth() == 1 || out_type.getWidth() == 1;
  const int output_bitwidth = out_type.getWidth();
  // The integer cast op is the same as C integer cast. Depends on the operand
  // type's signedness, we will determine whether or not sign extension is
  // needed.
  auto cast = [&](APInt value) {
    if (involves_bool) {
      // Handle boolean inputs or outputs explicitly as it doesn't have the same
      // behavior as extension or truncation.
      // true input should always be cast to 1 and not -1 as the sign extension
      // would do for signed outputs. Similarly, non-zero inputs should be cast
      // to true. Truncating even numbers to one bit will result in `false`.
      return APInt(out_type.getWidth(), value != 0);
    }
    return is_unsigned ? value.zextOrTrunc(output_bitwidth)
                       : value.sextOrTrunc(output_bitwidth);
  };

  return data.mapValues(out_type, cast);
}

OpFoldResult CastFloatToInt(DenseFPElementsAttr data, FloatType in_type,
                            IntegerType out_type) {
  if (!in_type.isF32()) {
    return {};
  }

  const auto out_width = out_type.getWidth();
  if (out_width != 64 && out_width != 32) {
    return {};
  }

  auto cast = [&](APFloat value) -> APInt {
    APSInt result(out_width, false);
    bool is_exact;
    value.convertToInteger(result, llvm::RoundingMode::TowardZero, &is_exact);
    return result;
  };

  return data.mapValues(out_type, cast);
}

template <typename InType, typename OutType>
llvm::SmallVector<OutType> MapStaticCast(DenseElementsAttr data) {
  return llvm::map_to_vector(data.getValues<InType>(),
                             [](InType v) { return static_cast<OutType>(v); });
}

OpFoldResult CastIntToFloat(DenseIntElementsAttr data, IntegerType in_type,
                            FloatType out_type) {
  auto result_type = data.getType().clone(out_type);
  if (!out_type.isF32()) {
    return {};
  }

  if (in_type.isSignlessInteger(32)) {
    return DenseFPElementsAttr::get(result_type,
                                    MapStaticCast<int32_t, float>(data));
  }
  if (in_type.isSignlessInteger(1)) {
    return DenseFPElementsAttr::get(result_type,
                                    MapStaticCast<bool, float>(data));
  }

  return {};
}

OpFoldResult CastFloatToFloat(DenseFPElementsAttr data, FloatType in_type,
                              FloatType out_type) {
  auto result_type = data.getType().clone(out_type);
  if (in_type.isF32() && out_type.isF64()) {
    return DenseFPElementsAttr::get(result_type,
                                    MapStaticCast<float, double>(data));
  }

  if (in_type.isF64() && out_type.isF32()) {
    return DenseFPElementsAttr::get(result_type,
                                    MapStaticCast<double, float>(data));
  }
  return {};
}

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  if (operands.size() != 1) {
    return {};
  }
  if (getInput().getType() == getType()) {
    return getInput();
  }

  auto input = operands[0];

  auto in_type = getInput().getType().getElementType();
  auto out_type = getType().getElementType();

  if (auto int_in_type = llvm::dyn_cast_or_null<IntegerType>(in_type)) {
    auto in_data = llvm::dyn_cast_or_null<DenseIntElementsAttr>(input);
    if (!in_data) {
      return {};
    }
    if (auto float_out_type = llvm::dyn_cast_or_null<FloatType>(out_type)) {
      return CastIntToFloat(in_data, int_in_type, float_out_type);
    }
    if (auto int_out_type = llvm::dyn_cast_or_null<IntegerType>(out_type)) {
      return CastIntToInt(in_data, int_in_type, int_out_type);
    }
  }

  if (auto float_in_type = llvm::dyn_cast_or_null<FloatType>(in_type)) {
    auto in_data = llvm::dyn_cast_or_null<DenseFPElementsAttr>(input);
    if (!in_data) {
      return {};
    }
    if (auto float_out_type = llvm::dyn_cast_or_null<FloatType>(out_type)) {
      return CastFloatToFloat(in_data, float_in_type, float_out_type);
    }
    if (auto int_out_type = llvm::dyn_cast_or_null<IntegerType>(out_type)) {
      return CastFloatToInt(in_data, float_in_type, int_out_type);
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SelectV2Op
//===----------------------------------------------------------------------===//

static void BuildSelectV2Op(Builder* builder, OperationState& result,
                            Value cond, Value x, Value y) {
  auto operand_type =
      OpTrait::util::getBroadcastedType(x.getType(), y.getType());

  if (!operand_type)
    emitError(result.location) << "non-broadcastable operands: " << x.getType()
                               << " and " << y.getType();

  bool has_static_cond_shape = false;
  bool has_static_operand_shape = false;
  ArrayRef<int64_t> cond_shape;
  ArrayRef<int64_t> operand_shape;

  if (auto shaped_type = mlir::dyn_cast<ShapedType>(cond.getType())) {
    if (shaped_type.hasStaticShape()) {
      has_static_cond_shape = true;
      cond_shape = shaped_type.getShape();
    }
  }
  if (auto shaped_type = mlir::dyn_cast<ShapedType>(operand_type)) {
    if (shaped_type.hasStaticShape()) {
      has_static_operand_shape = true;
      operand_shape = shaped_type.getShape();
    }
  }

  SmallVector<int64_t, 4> broadcastedShape;
  if (has_static_cond_shape && has_static_operand_shape &&
      !OpTrait::util::getBroadcastedShape(cond_shape, operand_shape,
                                          broadcastedShape)) {
    emitError(result.location) << "non-broadcastable operands: " << operand_type
                               << " and " << cond.getType();
  }

  result.addOperands({cond, x, y});

  auto elementType = mlir::dyn_cast<ShapedType>(x.getType()).getElementType();
  if (has_static_cond_shape && has_static_operand_shape) {
    result.types.push_back(
        tensorflow::GetTypeFromTFTensorShape(broadcastedShape, elementType));
  } else {
    result.types.push_back(UnrankedTensorType::get(elementType));
  }
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

namespace {

// Compute the length of a range (1-D) tensor given `start`, `limit`, `delta`.
// Template parameter `FloatOrInt` must be standard C integer or floating-point
// types.
template <typename FloatOrInt>
int GetLengthOfRange(FloatOrInt start, FloatOrInt limit, FloatOrInt delta) {
  // Refer to the implementation in
  // tensorflow/lite/kernels/range.cc.
  return std::is_integral<FloatOrInt>::value
             ? ((std::abs(limit - start) + std::abs(delta) - 1) /
                std::abs(delta))
             : std::ceil(std::abs((limit - start) / delta));
}

// Builds a constant range tensor of `result_elem_type` elements.
// Template parameter `FloatOrIntAtrr` must be mlir::IntegerAttr or
// mlir::FloatAttr.
template <typename FloatOrIntAtrr>
DenseElementsAttr BuildConstRangeTensor(Type result_elem_type, int num_elements,
                                        FloatOrIntAtrr start_attr,
                                        FloatOrIntAtrr delta_attr) {
  using ValueType = typename FloatOrIntAtrr::ValueType;  // APInt or APFloat
  ValueType start = start_attr.getValue();
  ValueType delta = delta_attr.getValue();

  SmallVector<ValueType, 16> new_values;
  new_values.reserve(num_elements);
  ValueType new_value = start;
  for (int i = 0; i < num_elements; ++i) {
    new_values.push_back(new_value);
    new_value = new_value + delta;
  }
  // Result is always a 1-D tensor.
  auto new_result_type =
      tensorflow::GetTypeFromTFTensorShape({num_elements}, result_elem_type);
  return DenseElementsAttr::get(new_result_type, new_values);
}
}  // namespace

OpFoldResult RangeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.size() == 3);
  auto start_tensor = mlir::dyn_cast_or_null<ElementsAttr>(operands[0]);
  auto limit_tensor = mlir::dyn_cast_or_null<ElementsAttr>(operands[1]);
  auto delta_tensor = mlir::dyn_cast_or_null<ElementsAttr>(operands[2]);
  if (start_tensor && limit_tensor && delta_tensor) {
    // Operands should all be scalars
    assert(start_tensor.getShapedType().getRank() == 0 &&
           limit_tensor.getShapedType().getRank() == 0 &&
           delta_tensor.getShapedType().getRank() == 0);
    Type elem_type = mlir::cast<ShapedType>(getType()).getElementType();
    if (elem_type.isSignlessInteger()) {
      auto start_attr = start_tensor.getValues<IntegerAttr>()[0];
      auto limit_attr = limit_tensor.getValues<IntegerAttr>()[0];
      auto delta_attr = delta_tensor.getValues<IntegerAttr>()[0];
      const int num_elements = GetLengthOfRange(
          start_attr.getInt(), limit_attr.getInt(), delta_attr.getInt());
      return BuildConstRangeTensor(elem_type, num_elements, start_attr,
                                   delta_attr);
    } else if (mlir::isa<FloatType>(elem_type)) {
      auto start_attr = start_tensor.getValues<FloatAttr>()[0];
      auto limit_attr = limit_tensor.getValues<FloatAttr>()[0];
      auto delta_attr = delta_tensor.getValues<FloatAttr>()[0];
      const int num_elements = GetLengthOfRange(start_attr.getValueAsDouble(),
                                                limit_attr.getValueAsDouble(),
                                                delta_attr.getValueAsDouble());
      return BuildConstRangeTensor(elem_type, num_elements, start_attr,
                                   delta_attr);
    }
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// TransposeConvOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult TransposeConvOp::verify() {
  TransposeConvOp op = *this;
  ShapedType output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  ShapedType output_shape_type =
      mlir::cast<ShapedType>(op.getOutputShape().getType());
  if (output_type.hasRank() && output_shape_type.hasStaticShape()) {
    if (output_type.getRank() != output_shape_type.getDimSize(0)) {
      return op.emitOpError(llvm::formatv(
          "expect output type has rank = {0}, got output type {1}",
          output_shape_type.getDimSize(0), output_type));
    }
  }

  DenseIntElementsAttr output_shape_elements;
  if (!matchPattern(op.getOutputShape(), m_Constant(&output_shape_elements))) {
    return success();
  }

  llvm::SmallVector<int64_t, 4> output_shape;
  output_shape.reserve(output_shape_elements.getNumElements());
  for (auto dim : output_shape_elements.getValues<int>()) {
    output_shape.push_back(dim);
  }

  auto expected_output_type = tensorflow::GetTypeFromTFTensorShape(
      output_shape, output_type.getElementType());
  if (failed(mlir::verifyCompatibleShape(output_type, expected_output_type))) {
    return op.emitOpError(llvm::formatv("expect output type {0}, got {1}",
                                        expected_output_type, output_type));
  }

  return success();
}

int64_t TransposeConvOp::GetArithmeticCount(Operation* op) {
  int64_t count = -1;
  auto transpose_conv = llvm::dyn_cast<TransposeConvOp>(op);
  auto input_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
      transpose_conv.getInput().getType());
  auto weight_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
      transpose_conv.getWeights().getType());
  if (input_type && weight_type && input_type.hasStaticShape() &&
      weight_type.hasStaticShape()) {
    // Compute op count from the seven nested loops of
    // tflite::reference_ops::TransposeConv():
    count = 2 * input_type.getNumElements() * weight_type.getDimSize(0) *
            weight_type.getDimSize(1) * weight_type.getDimSize(2);
  }

  return count;
}

//===----------------------------------------------------------------------===//
// StridedSliceOp
//===----------------------------------------------------------------------===//

LogicalResult StridedSliceOp::verify() {
  StridedSliceOp op = *this;
  auto ranked_input_type =
      mlir::dyn_cast<RankedTensorType>(op.getInput().getType());

  // If input is unranked, there is nothing else to be verified.
  if (!ranked_input_type) return success();
  int num_input_dims = ranked_input_type.getRank();

  if (auto begin_type =
          mlir::dyn_cast<RankedTensorType>(op.getBegin().getType())) {
    if (begin_type.getRank() != 1) return failure();
    if (begin_type.getDimSize(0) > num_input_dims) return failure();
  }

  if (auto end_type = mlir::dyn_cast<RankedTensorType>(op.getEnd().getType())) {
    if (end_type.getRank() != 1) return failure();
    if (end_type.getDimSize(0) > num_input_dims) return failure();
  }

  if (auto strides_type =
          mlir::dyn_cast<RankedTensorType>(op.getStrides().getType())) {
    if (strides_type.getRank() != 1) return failure();
    if (strides_type.getDimSize(0) > num_input_dims) return failure();
  }

  // The kernel will reshape the input tensor with new axis, it only supports
  // this reshaped tensor up to 5D.
  uint32_t ellipsis_mask = op.getEllipsisMask();
  uint32_t new_axis_mask = op.getNewAxisMask();
  int num_added_axis = 0;
  for (int i = 0; i < 8; ++i) {
    if (!((1 << i) & ellipsis_mask) && ((1 << i) & new_axis_mask)) {
      num_added_axis++;
    }
  }
  if (num_input_dims + num_added_axis > 5) return failure();
  return success();
}

OpFoldResult StridedSliceOp::fold(FoldAdaptor) {
  // Currently only support all masks being 0.
  if (getBeginMask() != 0 || getEndMask() != 0 || getEllipsisMask() != 0 ||
      getNewAxisMask() != 0 || getShrinkAxisMask() != 0)
    return {};

  auto input_type =
      mlir::dyn_cast_or_null<RankedTensorType>(getInput().getType());
  if (!input_type || !input_type.hasStaticShape()) return {};

  // Begin has to be all 0s.
  DenseIntElementsAttr begin_dense_elem_attr;
  if (!matchPattern(getBegin(), m_Constant(&begin_dense_elem_attr))) {
    return {};
  }
  for (auto begin_ele : begin_dense_elem_attr) {
    if (begin_ele.getSExtValue() != 0) {
      return {};
    }
  }

  // Strides has to be all 1s.
  DenseIntElementsAttr strides_dense_elem_attr;
  if (!matchPattern(getStrides(), m_Constant(&strides_dense_elem_attr))) {
    return {};
  }
  for (auto stride_ele : strides_dense_elem_attr) {
    if (stride_ele.getSExtValue() != 1) {
      return {};
    }
  }
  // End has to map the input shape.
  DenseIntElementsAttr end_dense_elem_attr;
  if (!matchPattern(getEnd(), m_Constant(&end_dense_elem_attr))) {
    return {};
  }
  int i = 0;
  for (auto end_ele : end_dense_elem_attr) {
    if (end_ele.getSExtValue() != input_type.getDimSize(i)) {
      return {};
    }
    ++i;
  }

  return getInput();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

namespace {

// The function recursively traverses the dimensions of the output tensor in
// a row-major order and writes the value of the output tensor into
// `output_element_addr`.
// TODO(@lukeboyer) make element byte size a template param.
void ComputePermutation(ArrayRef<int64_t> perms, ArrayRef<int64_t> output_shape,
                        const char* raw_input, const int element_byte_size,
                        const int64_t current_axis, char*& output_element_addr,
                        MutableArrayRef<uint64_t> current_input_index,
                        ShapedType input_shape_type) {
  const int64_t input_axis = perms[current_axis];
  const bool is_last_axis = current_axis == output_shape.size() - 1;
  for (int i = 0; i < output_shape[current_axis]; ++i) {
    // Update the input indices on `input_axis`.
    current_input_index[input_axis] = i;
    // Write the value from `input_tensor` if it is the last axis or
    // recurse into the next axis.
    if (is_last_axis) {
      int64_t input_flat_index = ElementsAttr::getFlattenedIndex(
          input_shape_type, current_input_index);
      // Address of input element to write raw data.
      const char* input_element_addr =
          raw_input + (input_flat_index * element_byte_size);
      std::memcpy(output_element_addr, input_element_addr, element_byte_size);
      // Increment the next output address to write to by bytes equal to
      // width of constiuent elements.
      output_element_addr += element_byte_size;
    } else {
      ComputePermutation(perms, output_shape, raw_input, element_byte_size,
                         current_axis + 1, output_element_addr,
                         current_input_index, input_shape_type);
    }
  }
}

}  // namespace

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<ConvertTransposeToDecreaseRank>(context);
}

OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();

  auto input_tensor = mlir::dyn_cast_or_null<DenseElementsAttr>(operands[0]);
  auto perm_tensor = mlir::dyn_cast_or_null<ElementsAttr>(operands[1]);
  if (!input_tensor || !perm_tensor) return nullptr;

  // Do not try to fold elements attr of a quant type because
  // DenseElementsAttr does not support it.
  if (!mlir::cast<ShapedType>(getType())
           .getElementType()
           .isSignlessIntOrFloat())
    return nullptr;

  // TODO(b/280099953) This algorithm only works for fixed width element types.
  // This is the usual case, but consider falling back to old approach
  // if transposing string tensors becomes needed while folding.
  if (!input_tensor.getElementType().isIntOrIndexOrFloat()) return nullptr;
  SmallVector<int64_t> perms;
  SmallVector<int64_t> output_shape;
  ArrayRef<int64_t> input_shape = input_tensor.getType().getShape();
  auto attr_iter = perm_tensor.getValues<IntegerAttr>();
  const int num_dimensions = input_tensor.getType().getRank();
  for (int i = 0; i < num_dimensions; ++i) {
    perms.push_back(attr_iter[i].getInt());
    output_shape.push_back(input_shape[perms[i]]);
  }

  // If the input tensor values are splat, then it has exactly one value.
  // It is sufficient then to just reshape the input data.
  if (input_tensor.isSplat()) {
    return input_tensor.reshape(input_tensor.getType().cloneWith(
        output_shape, input_tensor.getElementType()));
  }

  // MLIR implementation pads elements < 8 bits to 8 bits and pads non byte
  // aligned to the nearest byte. So this is allowed.
  const char* raw_input = input_tensor.getRawData().data();
  const int element_byte_size =
      input_tensor.getElementType().getIntOrFloatBitWidth() / 8;

  // Hold current ND index in input tensor when computing
  // permutation.
  llvm::OwningArrayRef<uint64_t> current_input_index(
      input_tensor.getType().getRank());

  // Allocate raw data and retrieve address of the first char in its raw
  // buffer.
  llvm::OwningArrayRef<char> raw_output_arr(input_tensor.getRawData());
  char* raw_output = (char*)raw_output_arr.data();

  // Compute the result and write to `raw_output`.
  ComputePermutation(perms, output_shape, raw_input, element_byte_size,
                     /*current_axis=*/0, raw_output, current_input_index,
                     input_tensor.getType());

  bool detected_splat = false;
  const bool valid_output_buffer = DenseElementsAttr::isValidRawBuffer(
      input_tensor.getType(), raw_output_arr, detected_splat);
  if (!valid_output_buffer || detected_splat) return nullptr;

  auto result_type =
      RankedTensorType::get(output_shape, input_tensor.getElementType());
  return DenseElementsAttr::getFromRawBuffer(result_type, raw_output_arr);
}

mlir::LogicalResult TransposeOp::verify() {
  TransposeOp op = *this;
  auto input_type = mlir::cast<ShapedType>(op.getInput().getType());
  auto perm_type = mlir::cast<ShapedType>(op.getPerm().getType());
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());
  if (input_type.hasStaticShape() && perm_type.hasStaticShape()) {
    if (perm_type.getNumElements() != input_type.getRank()) {
      return op.emitOpError(
          "perm tensor elements size is not equal to input tensor rank");
    }
  }

  DenseIntElementsAttr perm;
  if (!matchPattern(op.getPerm(), m_Constant(&perm))) {
    return success();
  }

  int index = 0;
  llvm::SmallVector<int64_t, 4> axes;
  for (const auto& axis_int : perm.getValues<APInt>()) {
    int64_t axis = axis_int.getSExtValue();
    if (axis < 0) {
      axis += input_type.getRank();
    }
    if (axis < 0 || (input_type.hasRank() && axis >= input_type.getRank())) {
      return op.emitOpError(
          llvm::formatv("perm[{0}] must be in [-rank, rank)", index));
    }
    if (std::count(axes.begin(), axes.end(), axis) > 0) {
      return op.emitOpError(
          llvm::formatv("perm[{0}] cannot have duplicated axis", index));
    }
    axes.push_back(axis);
    index++;
  }

  if (input_type.hasStaticShape() && output_type.hasStaticShape()) {
    llvm::SmallVector<int64_t, 4> transposed_shape;
    for (int64_t axis : axes) {
      transposed_shape.push_back(input_type.getDimSize(axis));
    }
    auto expected_output_type = tensorflow::GetTypeFromTFTensorShape(
        transposed_shape, input_type.getElementType());
    if (failed(
            mlir::verifyCompatibleShape(output_type, expected_output_type))) {
      return op.emitOpError(llvm::formatv("expect output type {0}, got {1}",
                                          expected_output_type, output_type));
    }
  }

  // Verify the quantized axis if the type is UniformQuantizedPerAxisType. Other
  // verifications to make sure the input and output has the same quantization
  // type, scale and zero point are performed by the SameOperandsAndResultsScale
  // trait.
  auto in_per_axis_qtype =
      mlir::dyn_cast_or_null<quant::UniformQuantizedPerAxisType>(
          QuantizedType::getQuantizedElementType(input_type));
  auto out_per_axis_qtype =
      mlir::dyn_cast_or_null<quant::UniformQuantizedPerAxisType>(
          QuantizedType::getQuantizedElementType(output_type));
  if (in_per_axis_qtype && out_per_axis_qtype) {
    if (out_per_axis_qtype.getQuantizedDimension() < axes.size() &&
        axes[out_per_axis_qtype.getQuantizedDimension()] !=
            in_per_axis_qtype.getQuantizedDimension()) {
      return op.emitOpError(
          "has mismatched quantized axes of input and output");
    }
  }

  return success();
}

static void BuildTransposeOp(OpBuilder* builder, OperationState& result,
                             Value input, Value perm) {
  // Output size is only known if input is ranked and perm is a constant.
  auto input_type = mlir::cast<TensorType>(input.getType());
  DenseIntElementsAttr perm_const;
  if (!input_type.hasRank() || !matchPattern(perm, m_Constant(&perm_const)) ||
      perm_const.empty()) {
    TFL::TransposeOp::build(
        *builder, result, UnrankedTensorType::get(input_type.getElementType()),
        input, perm);
    return;
  }

  const auto perm_value_it = perm_const.value_begin<APInt>();

  const ArrayRef<int64_t> input_shape = input_type.getShape();
  SmallVector<int64_t, 4> output_shape(input_shape.size());

  for (int i = 0; i < output_shape.size(); ++i) {
    const APInt perm_val = perm_value_it[i];
    output_shape[i] = input_shape[perm_val.getSExtValue()];
  }

  auto element_type = input_type.getElementType();
  // For UniformQuantizedPerAxisType element type, the quantized dimension
  // should be changed corresponding with the transpose.
  auto per_axis_qtype =
      mlir::dyn_cast_or_null<quant::UniformQuantizedPerAxisType>(
          QuantizedType::getQuantizedElementType(input_type));
  if (per_axis_qtype) {
    int32_t quantized_dimension = per_axis_qtype.getQuantizedDimension();
    for (int i = 0; i < output_shape.size(); ++i) {
      const APInt perm_val = perm_value_it[i];
      if (perm_val.getSExtValue() == quantized_dimension) {
        quantized_dimension = i;
        break;
      }
    }
    element_type = quant::UniformQuantizedPerAxisType::get(
        per_axis_qtype.getFlags(), per_axis_qtype.getStorageType(),
        per_axis_qtype.getExpressedType(), per_axis_qtype.getScales(),
        per_axis_qtype.getZeroPoints(), quantized_dimension,
        per_axis_qtype.getStorageTypeMin(), per_axis_qtype.getStorageTypeMax());
  }

  TFL::TransposeOp::build(
      *builder, result,
      tensorflow::GetTypeFromTFTensorShape(output_shape, element_type), input,
      perm);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void IfOp::getSuccessorRegions(RegionBranchPoint point,

                               SmallVectorImpl<RegionSuccessor>& regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // Don't consider the else region if it is empty.
  Region* else_reg = &getElseRegion();
  if (else_reg->empty()) else_reg = nullptr;

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getThenRegion()));
  // If the else region does not exist, it is not a viable successor.
  if (else_reg) regions.push_back(RegionSuccessor(else_reg));
}

void IfOp::getEntrySuccessorRegions(ArrayRef<Attribute> operands,
                                    SmallVectorImpl<RegionSuccessor>& regions) {
  // Don't consider the else region if it is empty.
  Region* else_reg = &getElseRegion();
  if (else_reg->empty()) else_reg = nullptr;

  bool condition;
  if (auto cond_attr = mlir::dyn_cast_or_null<IntegerAttr>(operands.front())) {
    condition = cond_attr.getValue().isOne();

    // Add the successor regions using the condition.
    regions.push_back(RegionSuccessor(condition ? &getThenRegion() : else_reg));
  } else {
    // If the condition isn't constant, both regions may be executed.
    regions.push_back(RegionSuccessor(&getThenRegion()));
    // If the else region does not exist, it is not a viable successor.
    if (else_reg) regions.push_back(RegionSuccessor(else_reg));
    return;
  }
}

//===----------------------------------------------------------------------===//
// PolyCallOp
//===----------------------------------------------------------------------===//

namespace {
// Canonicalize converted TF ops into PolymorphicCall op so different
// representations are preserved.
struct PolyCallResultOperandsMatchAndImplicitCapture
    : public OpRewritePattern<PolyCallOp> {
  using OpRewritePattern<PolyCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PolyCallOp while_op,
                                PatternRewriter& rewriter) const override {
    // Finish this.
    return success();
  }
};

}  // namespace

void PolyCallOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<PolyCallResultOperandsMatchAndImplicitCapture>(context);
}

void PolyCallOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor>& regions) {
  // Defaults to first region for TFLite execution.
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::verify() {
  WhileOp op = *this;
  if (op.getNumOperands() != op.getNumResults())
    return op.emitOpError(llvm::formatv(
        "number of operands does not match number of results ({0} != {1})",
        op.getNumOperands(), op.getNumResults()));
  if (op.getCond().front().getNumArguments() !=
      op.getBody().front().getNumArguments())
    return op.emitOpError(llvm::formatv(
        "number of arguments in condition function does not match number of "
        "arguments in body function ({0} != {1})",
        op.getCond().front().getNumArguments(),
        op.getBody().front().getNumArguments()));
  // Verify shapes are compatible.
  for (auto it : llvm::zip(op.getCond().front().getArgumentTypes(),
                           op.getBody().front().getArgumentTypes())) {
    if (failed(mlir::verifyCompatibleShape(std::get<0>(it), std::get<1>(it))))
      return op->emitOpError(llvm::formatv(
          "condition function's argument type does not match body "
          "function's argument type ({0} != {1})",
          std::get<0>(it), std::get<1>(it)));
  }

  return success();
}

namespace {
// Canonicalize While op so that results and operands match and external values
// are via implicit capture rather than via block args.
struct WhileResultOperandsMatchAndImplicitCapture
    : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp while_op,
                                PatternRewriter& rewriter) const override {
    // Replace values simply passed through the body with extern values
    // (in both body and condition regions as well as while result). The
    // block arguments of body and while match and so the corresponding cond
    // argument can be easily found.
    bool unchanged = true;
    auto& body_block = while_op.getBody().front();
    auto& cond_block = while_op.getCond().front();
    auto& yield = *body_block.getTerminator();
    for (auto ba : body_block.getArguments()) {
      int arg_no = ba.getArgNumber();
      // Skip removing resources that are not read-only variables.
      if (mlir::isa<TF::ResourceType>(getElementTypeOrSelf(ba.getType()))) {
        bool has_read_only_variables = true;
        for (auto user : ba.getUsers()) {
          // Terminator ops, for example, tfl::yield op, should be ignored since
          // the argument can be used for yielding as the `body` function result
          // and that does not give any meaningful points to the decision
          // whether the given argument is a read-only variable or not.
          if (user->hasTrait<OpTrait::IsTerminator>()) continue;
          if (!llvm::isa<mlir::TF::ReadVariableOp>(user)) {
            has_read_only_variables = false;
            break;
          }
        }
        if (!has_read_only_variables) continue;
      }
      if (ba == yield.getOperand(arg_no)) {
        unchanged = false;
        auto value = while_op.getOperand(arg_no);
        ba.replaceAllUsesWith(value);
        cond_block.getArgument(arg_no).replaceAllUsesWith(value);

        // This could be relaxed and casts inserted.
        if (while_op.getResult(arg_no).getType() == value.getType())
          while_op.getResult(arg_no).replaceAllUsesWith(value);
      }
    }

    // The While ops operands and result types need to match
    SmallVector<Value, 4> new_operands;
    SmallVector<Value, 4> new_body_yield;
    SmallVector<bool, 4> removed_operand(while_op.getNumOperands(), false);
    llvm::SmallVector<Type, 4> types;
    new_operands.reserve(while_op.getNumOperands());
    new_body_yield.reserve(while_op.getNumOperands());
    types.reserve(while_op.getNumOperands());

    // Remove block arguments not used in either cond or body. This leaves the
    // block arguments of body and cond matching still.
    int arg_index = 0;
    for (int while_index = 0, e = while_op.getNumOperands(); while_index < e;
         ++while_index) {
      auto value = while_op.getOperand(while_index);
      if (body_block.getArgument(arg_index).use_empty() &&
          cond_block.getArgument(arg_index).use_empty() &&
          // Note: since we are not erasing results, need to use while_index
          // to check if the corresponding result is unused.
          while_op.getResult(while_index).use_empty()) {
        unchanged = false;
        body_block.eraseArgument(arg_index);
        cond_block.eraseArgument(arg_index);

        // Mark operand for removal.
        removed_operand[while_index] = true;
      } else {
        new_operands.push_back(value);
        new_body_yield.push_back(yield.getOperand(while_index));
        auto type = while_op.getResult(while_index).getType();
        types.push_back(type);
        ++arg_index;
      }
    }

    // Done if no values removed from blocks and operands & results match.
    if (unchanged) return failure();

    // Replace with new While with matching operands and results.
    Operation* op = while_op.getOperation();
    Operation* new_op = rewriter.insert(Operation::create(
        op->getLoc(), op->getName(), types, new_operands, op->getAttrs(),
        op->getPropertiesStorage(), {}, /*numRegions=*/2));

    for (int i = 0; i < 2; ++i) new_op->getRegion(i).takeBody(op->getRegion(i));
    int new_index = 0;
    for (int op_index = 0, e = op->getNumResults(); op_index < e; ++op_index) {
      if (removed_operand[op_index]) continue;
      op->getResult(op_index).replaceAllUsesWith(new_op->getResult(new_index));
      ++new_index;
    }
    rewriter.eraseOp(op);

    Block& new_body_block = cast<WhileOp>(new_op).getBody().front();
    rewriter.setInsertionPointToEnd(&new_body_block);
    rewriter.replaceOpWithNewOp<YieldOp>(new_body_block.getTerminator(),
                                         new_body_yield);

    return success();
  }
};

}  // namespace

void WhileOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<WhileResultOperandsMatchAndImplicitCapture>(context);
}

SmallVector<Region*> WhileOp::getLoopRegions() { return {&getBody()}; }

bool WhileOp::isDefinedOutsideOfLoop(Value value) {
  // TODO(jpienaar): This is to overly conservative and disables anything other
  // than constant hoisting initially.
  return false;
}

//===----------------------------------------------------------------------===//
// LogisticOp
//===----------------------------------------------------------------------===//

int64_t LogisticOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  // As a very rough ballpark, the cost of evaluating a math function
  // such as tanh or logistic is about 32 multiplications, and about as
  // many additions/subtractions. (Just a power-of-two order-of-magnitude
  // from looking at actual implementations that we use in runtime/code).
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count))
    return 64 * count;

  return -1;
}

//===----------------------------------------------------------------------===//
// LogSoftmaxOp
//===----------------------------------------------------------------------===//

int64_t LogSoftmaxOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  // As a very rough ballpark, the cost of evaluating a math function
  // such as tanh or logistic is about 32 multiplications, and about as
  // many additions/subtractions. (Just a power-of-two order-of-magnitude
  // from looking at actual implementations that we use in runtime/code).
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count))
    return 64 * count;

  return -1;
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

int64_t SoftmaxOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  // As a very rough ballpark, the cost of evaluating a math function
  // such as tanh or logistic is about 32 multiplications, and about as
  // many additions/subtractions. (Just a power-of-two order-of-magnitude
  // from looking at actual implementations that we use in runtime/code).
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count))
    return 64 * count;

  return -1;
}

//===----------------------------------------------------------------------===//
// TanhOp
//===----------------------------------------------------------------------===//

int64_t TanhOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  // As a very rough ballpark, the cost of evaluating a math function
  // such as tanh or logistic is about 32 multiplications, and about as
  // many additions/subtractions. (Just a power-of-two order-of-magnitude
  // from looking at actual implementations that we use in runtime/code).
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count))
    return 64 * count;

  return -1;
}

//===----------------------------------------------------------------------===//
// AddNOp
//===----------------------------------------------------------------------===//

int64_t AddNOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count)) {
    // AddN cost is roughly the same cost as N-1 Adds.
    const int64_t num_adds = op->getNumOperands() - 1;
    return num_adds * count;
  }

  return -1;
}

//===----------------------------------------------------------------------===//
// AveragePool2DOp
//===----------------------------------------------------------------------===//

int64_t AveragePool2DOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count)) {
    auto avg_pool = llvm::dyn_cast<AveragePool2DOp>(op);
    return avg_pool.getFilterHeight() * avg_pool.getFilterWidth() * count;
  }

  return -1;
}

//===----------------------------------------------------------------------===//
// MaxPool2DOp
//===----------------------------------------------------------------------===//

int64_t MaxPool2DOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count)) {
    auto max_pool = llvm::dyn_cast<MaxPool2DOp>(op);
    return max_pool.getFilterHeight() * max_pool.getFilterWidth() * count;
  }

  return -1;
}

//===----------------------------------------------------------------------===//
// ReverseV2Op
//===----------------------------------------------------------------------===//

OpFoldResult ReverseV2Op::fold(FoldAdaptor adaptor) {
  auto input = adaptor.getInput();
  auto axis = adaptor.getAxis();

  if (!input || !axis) {
    return {};
  }

  auto axis_val = llvm::cast<DenseIntElementsAttr>(axis);
  llvm::SetVector<int32_t> axis_set(axis_val.value_begin<int32_t>(),
                                    axis_val.value_end<int32_t>());

  auto input_type = getInput().getType();
  const FlatIndHelper helper(input_type);

  std::vector<Attribute> new_data(input_type.getNumElements());
  auto input_data = llvm::cast<DenseElementsAttr>(input);

  for (auto [i, val] : llvm::enumerate(input_data.getValues<Attribute>())) {
    auto shaped_ind = helper.GetShapedInd(i);

    for (int d = 0; d < shaped_ind.size(); ++d) {
      if (!axis_set.contains(d)) {
        continue;
      }

      shaped_ind[d] = input_type.getDimSize(d) - 1 - shaped_ind[d];
    }

    new_data[helper.GetFlatInd(shaped_ind)] = val;
  }

  return DenseElementsAttr::get(getType(), new_data);
}

//===----------------------------------------------------------------------===//
// L2NormalizationOp
//===----------------------------------------------------------------------===//

int64_t L2NormalizationOp::GetArithmeticCount(Operation* op) {
  int64_t count;
  // Computing the squared L2 norm is N multiply-adds so 2N ops,
  // then the single inverse-sqrt is negligible, then we multiply each
  // value by the resulting multiplier, so an extra N ops. count 3N ops.
  if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count)) {
    return 3 * count;
  }

  return -1;
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

OpFoldResult PadOp::fold(FoldAdaptor) {
  if (InputOutputHasSameShape(getInput().getType(), getOutput().getType()))
    return getInput();

  return {};
}

// When padding amounts are constants, cast them to i32. XNN can only
// consume i32 pad amounts in some cases.
struct CastConstPadAmounts : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter& rewriter) const override {
    auto padding_amount_op = op.getPadding().getDefiningOp();
    if (!padding_amount_op ||
        !padding_amount_op->hasTrait<OpTrait::ConstantLike>()) {
      return failure();
    }

    auto padding_type = op.getPadding().getType();
    if (!padding_type.getElementType().isSignlessInteger(64)) {
      return failure();
    }

    auto cast = rewriter.createOrFold<CastOp>(
        padding_amount_op->getLoc(),
        padding_type.clone(rewriter.getIntegerType(32)), op.getPadding());

    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(1, cast); });

    return success();
  }
};

void PadOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<CastConstPadAmounts>(context);
}

//===----------------------------------------------------------------------===//
// PadV2Op
//===----------------------------------------------------------------------===//

OpFoldResult PadV2Op::fold(FoldAdaptor) {
  if (InputOutputHasSameShape(getInput().getType(), getOutput().getType()))
    return getInput();

  return {};
}

//===----------------------------------------------------------------------===//
// NoValueOp
//===----------------------------------------------------------------------===//

OpFoldResult NoValueOp::fold(FoldAdaptor) { return getValueAttr(); }

bool NoValueOp::isBuildableWith(Attribute value, Type type) {
  return mlir::isa<UnitAttr>(value) && mlir::isa<NoneType>(type);
}

YieldOp ControlNodeOp::GetYield() {
  return llvm::cast<YieldOp>(GetBody().back());
}

// Checks if a TFL.control_node wraps a single operation and the single
// operation results are perfectly forwarded to the wrapper's yield.
bool ControlNodeOp::WrapsSinglePerfectlyForwardedOp() {
  auto body = GetBody().without_terminator();
  if (!hasSingleElement(body)) return false;

  Operation& controlled_op = *body.begin();
  YieldOp yield = GetYield();
  return controlled_op.getNumResults() == yield.getNumOperands() &&
         std::equal(controlled_op.getResults().begin(),
                    controlled_op.getResults().end(),
                    yield.getOperands().begin());
}

mlir::LogicalResult ControlNodeOp::verify() {
  ControlNodeOp control_node = *this;
  if (!control_node.GetBody().args_empty())
    return control_node.emitOpError() << "expects body without any arguments";

  Operation& yield = control_node.GetBody().back();
  if (!isa<YieldOp>(yield))
    return yield.emitOpError()
           << "invalid TFL.control_node terminator, yield expected";

  // Ensure that the terminator's operands and the control_node results match in
  // types.
  const int result_count =
      control_node.getNumResults() - 1;  // 1 for control token
  const int num_operands = yield.getNumOperands();
  if (num_operands != result_count)
    return yield.emitOpError()
           << "has " << yield.getNumOperands()
           << " operand, but control_node returns " << result_count;
  for (const int operand_idx : llvm::seq<int>(0, yield.getNumOperands())) {
    if (control_node.getResult(operand_idx).getType() !=
        yield.getOperand(operand_idx).getType())
      return yield.emitOpError() << "operand #" << operand_idx
                                 << " type mismatch control_node results";
  }
  return success();
}

void ControlNodeOp::print(OpAsmPrinter& p) {
  if (getNumOperands()) {
    // These are always control operand, no explicit type needed.
    p << '(';
    p.printOperands(getOperands());
    p << ')';
  }
  // Check if we can print the short "controls" form: that is if the
  // control_node contains a single operation and the results of this operation
  // are perfectly forwarded to the yield.
  if (getOperation()->getAttrs().empty() && WrapsSinglePerfectlyForwardedOp()) {
    Operation& controlled_op = GetBody().front();
    // The "controls" syntax only encodes a single location.
    YieldOp yield_op = GetYield();
    // In order to correctly round-trip, we can only use this syntax when all
    // the locations are identical.
    if (controlled_op.getLoc() == getLoc() && yield_op.getLoc() == getLoc()) {
      p << " controls ";
      p.printGenericOp(&controlled_op);
      return;
    }
  }
  p << ' ';
  p.printRegion(getOperation()->getRegion(0));
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult ControlNodeOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse the body region.
  llvm::SMLoc loc = parser.getCurrentLocation();
  Type control_type = ControlType::get(parser.getBuilder().getContext());

  // Parse optional argument list (control dependencies only).
  SmallVector<OpAsmParser::UnresolvedOperand, 4> op_infos;
  if (parser.parseOperandList(op_infos, OpAsmParser::Delimiter::OptionalParen))
    return failure();
  if (!op_infos.empty()) {
    SmallVector<Type, 2> types(op_infos.size(), control_type);
    if (parser.resolveOperands(op_infos, types, loc, result.operands))
      return failure();
  }

  Region& body = *result.addRegion();

  if (succeeded(parser.parseOptionalKeyword("controls"))) {
    // If we parse the short version of the control node, we have an operation
    // in the generic form that follows the "controls" keyword. Parse it inside
    // the region and forward all of its results as-is to the yield operation.
    body.push_back(new Block);
    Block& block = body.back();
    Operation* controlled_op =
        parser.parseGenericOperation(&block, block.begin());
    if (!controlled_op) return failure();
    OpBuilder builder(parser.getBuilder().getContext());
    builder.setInsertionPointToEnd(&block);
    builder.create<YieldOp>(controlled_op->getLoc(),
                            controlled_op->getResults());
    result.location = controlled_op->getLoc();
  } else if (parser.parseRegion(body)) {
    return failure();
  }

  ControlNodeOp::ensureTerminator(body, parser.getBuilder(), result.location);

  // Get the results type for the control node from the terminator operands.
  Operation& yield = body.back().back();
  result.types.reserve(yield.getNumOperands() + 1);
  result.types.append(yield.operand_type_begin(), yield.operand_type_end());
  result.types.push_back(control_type);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// EmbeddingLookupOp
//===----------------------------------------------------------------------===//

OpFoldResult EmbeddingLookupOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto lookup_attr = mlir::dyn_cast_or_null<DenseIntElementsAttr>(operands[0]);
  auto value_attr = mlir::dyn_cast_or_null<DenseElementsAttr>(operands[1]);
  if (!lookup_attr || !value_attr) {
    return {};
  }

  // Hybrid lookup is not supported.
  if (value_attr.getType().getElementType() != getType().getElementType()) {
    return {};
  }

  // Expect 1d lookup tensor and a non-scalar input.
  if (lookup_attr.getType().getRank() != 1 ||
      value_attr.getType().getRank() == 0) {
    return {};
  }

  int64_t outer_dim_size = value_attr.getType().getShape()[0];
  int64_t inner_dims_size = value_attr.getNumElements() / outer_dim_size;

  std::vector<Attribute> new_values;
  for (auto outer_index : lookup_attr.getValues<int32_t>()) {
    // Fail if index is out of bounds.
    if (outer_index < 0 || outer_index >= outer_dim_size) {
      return {};
    }

    for (int i = 0; i < inner_dims_size; ++i) {
      new_values.push_back(
          value_attr.getValues<Attribute>()[outer_index * inner_dims_size + i]);
    }
  }

  std::vector<int64_t> new_shape = value_attr.getType().getShape().vec();
  new_shape[0] = lookup_attr.getType().getShape()[0];
  auto new_type = value_attr.getType().clone(new_shape);

  return DenseElementsAttr::get(new_type, new_values);
}

//===----------------------------------------------------------------------===//
// ConstBytesAttr
//===----------------------------------------------------------------------===//

Attribute ConstBytesAttr::parse(AsmParser& parser, Type type) {
  if (parser.parseColon()) {
    return nullptr;
  }

  std::string data;
  if (parser.parseString(&data)) {
    return nullptr;
  }
  if (data.size() >= 2 && data.substr(0, 2) == "0x") {
    std::string bytes_data = absl::HexStringToBytes(data.substr(2));
    return ConstBytesAttr::get(parser.getBuilder().getContext(), bytes_data);
  }

  return ConstBytesAttr::get(parser.getBuilder().getContext(), data);
}

void ConstBytesAttr::print(mlir::AsmPrinter& printer) const {
  StringRef bytes_str = getValue();
  // Elide the attribute if flag is set.
  std::optional<int64_t> limit = OpPrintingFlags().getLargeElementsAttrLimit();
  printer << " : \"";
  if (limit && limit.value() < bytes_str.size()) {
    printer << "__elided__";
  } else {
    printer << "0x" << llvm::toHex(bytes_str);
  }
  printer << "\"";
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

int64_t GetTypeBitWidth(mlir::Type type) {
  if (auto quant_type = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
    return quant_type.getStorageTypeIntegralWidth();
  }
  if (type.isIntOrFloat()) {
    return std::max(type.getIntOrFloatBitWidth(),
                    static_cast<unsigned>(CHAR_BIT));
  }
  return -1;
}

LogicalResult BitcastOp::verify() {
  BitcastOp op = *this;
  auto input_type = mlir::cast<ShapedType>(op.getInput().getType());
  auto output_type = mlir::cast<ShapedType>(op.getOutput().getType());

  auto input_element_type = input_type.getElementType();
  auto output_element_type = output_type.getElementType();

  if (input_type.hasStaticShape()) {
    const int input_element_type_bitwidth = GetTypeBitWidth(input_element_type);
    const int output_element_type_bitwidth =
        GetTypeBitWidth(output_element_type);

    if (input_element_type_bitwidth < 0 || output_element_type_bitwidth < 0) {
      // Only supports quantized type, int and float types.
      return op.emitOpError("Unsupported element type.");
    }

    if (input_element_type_bitwidth < output_element_type_bitwidth) {
      if (output_element_type_bitwidth % input_element_type_bitwidth != 0) {
        return op.emitOpError(
            "output element bitwidth is not multiple of input element "
            "bitwidth");
      }
      if (input_type.getShape().empty() ||
          input_type.getShape().back() % (output_element_type_bitwidth /
                                          input_element_type_bitwidth) !=
              0) {
        return op.emitOpError(
            "input rightmost dimension size is not multiple of the divisor");
      }
    } else if (input_element_type_bitwidth > output_element_type_bitwidth) {
      if (input_element_type_bitwidth % output_element_type_bitwidth != 0) {
        return op.emitOpError(
            "input element bitwidth is not multiple of output element "
            "bitwidth");
      }
    }
  }
  return success();
}

OpFoldResult BitcastOp::fold(FoldAdaptor adaptor) {
  if (getType() == getInput().getType()) return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// DynamicUpdateSliceOp
//===----------------------------------------------------------------------===//

OpFoldResult DynamicUpdateSliceOp::fold(FoldAdaptor) {
  // Check if update replaces the whole tensor, meaning operand and update has
  // the same shape and all start indices are zero.
  DenseIntElementsAttr indices_attr;
  if (matchPattern(getStartIndices(), m_Constant(&indices_attr)) &&
      indices_attr.isSplat() && indices_attr.getSplatValue<int>() == 0 &&
      getOperand().getType().hasStaticShape() &&
      getUpdate().getType().hasStaticShape() &&
      getOperand().getType() == getUpdate().getType()) {
    return getUpdate();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_interface.cc.inc"

static FailureOr<SmallVector<int32_t>> parseI32Array(AsmParser& parser) {
  SmallVector<int32_t> elements;
  auto elementParser = [&]() {
    int32_t element;
    if (failed(parser.parseInteger(element))) return failure();
    elements.push_back(element);
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     elementParser))
    return failure();
  return elements;
}

}  // namespace TFL
}  // namespace mlir

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_dialect.cc.inc"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_enums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_attrdefs.cc.inc"
#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.cc.inc"

namespace mlir {
namespace TFL {

#include "tensorflow/compiler/mlir/lite/runtime_verifiers.inc"

Operation* TFLDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                           Type type, Location loc) {
  // If this is a constant bytes attribute or the result type doesn't match the
  // attribute type, then generate a tfl.pseudo_const.
  if (mlir::isa<ConstBytesAttr>(value) ||
      (mlir::isa<ElementsAttr>(value) &&
       mlir::cast<ElementsAttr>(value).getType() != type))
    return builder.create<ConstOp>(loc, type, mlir::cast<ElementsAttr>(value));
  if (arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(value));
  if (NoValueOp::isBuildableWith(value, type))
    return builder.create<NoValueOp>(loc, type, mlir::cast<UnitAttr>(value));
  return nullptr;
}

}  // namespace TFL
}  // namespace mlir
