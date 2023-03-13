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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_arith_ops_folder.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_canonicalization_helper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_device_helper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_tensor_helper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/rewrite_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace TF {

namespace {
// Returns the equivalent Value skipping through identity nodes.
Value LookThroughIdentity(Value result) {
  while (isa_and_nonnull<IdentityOp, IdentityNOp>(result.getDefiningOp())) {
    auto op_result = result.cast<OpResult>();
    result = op_result.getOwner()->getOperand(op_result.getResultNumber());
  }
  return result;
}

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_canonicalize.inc"
}  // namespace

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NcclAllReduceOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NegOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(OnesLikeOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PreventGradientOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(QuantizeAndDequantizeOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RandomShuffleOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReciprocalOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReciprocalGradOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReluOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Relu6Op);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Relu6GradOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReluGradOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RintOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RoundOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RsqrtOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RsqrtGradOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SeluOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SeluGradOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SigmoidOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SigmoidGradOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SignOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SinOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SinhOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SnapshotOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SoftmaxOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SoftplusOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SoftplusGradOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SoftsignOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SoftsignGradOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SqrtOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SqrtGradOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SquareOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(StringStripOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanhOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanhGradOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ZerosLikeOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(_UnaryOpsCompositionOp);

//===----------------------------------------------------------------------===//
// NcclAllReduceOp
//===----------------------------------------------------------------------===//

// For `NcclAllReduceOp` ops the `device` attribute corresponds to the resource
// instance.
std::optional<std::string> NcclAllReduceOp::GetResourceInstanceStr() {
  auto device_attr = (*this)->getAttrOfType<StringAttr>("device");
  // Treat missing device attribute like unspecified (= empty string) attribute.
  // Note that different op instances with the same string (including empty
  // string) are seen as dependent (same resource instance).
  if (!device_attr) return "";
  return device_attr.str();
}

//===----------------------------------------------------------------------===//
// NotEqualOp
//===----------------------------------------------------------------------===//

LogicalResult NotEqualOp::verify() {
  NotEqualOp op = *this;
  // If we allow inputs to have incompatible type, then nothing to do.
  if (!op.getIncompatibleShapeError()) return success();

  // Otherwise, check inputs are broadcastable.
  return mlir::OpTrait::impl::verifyCompatibleOperandBroadcast(
      op.getOperation());
}

void NotEqualOp::build(OpBuilder &builder, OperationState &result, Value x,
                       Value y, BoolAttr incompatible_shape_error) {
  auto result_type = DeduceEqualCmpOpType(&builder, result.location, x, y,
                                          incompatible_shape_error);
  return build(builder, result, result_type, x, y, incompatible_shape_error);
}

//===----------------------------------------------------------------------===//
// OneHotOp
//===----------------------------------------------------------------------===//

LogicalResult OneHotOp::verify() {
  OneHotOp op = *this;
  int64_t axis = op.getAxis();

  auto indices_ty = op.getIndices().getType().dyn_cast<RankedTensorType>();
  if (indices_ty &&
      !(axis == -1 || (axis >= 0 && axis <= indices_ty.getShape().size()))) {
    return op.emitOpError()
           << "expected axis (" << axis << ") to be -1 or between [0, "
           << indices_ty.getShape().size() << "]";
  }

  if (axis < -1) {
    return op.emitOpError() << "expected axis (" << axis
                            << ") to be -1 or between [0, rank(indices()))";
  }

  if (!IsOfRankOrUnranked(op.getDepth(), 0)) {
    return op.emitOpError() << "requires depth to be a scalar";
  }
  if (!IsOfRankOrUnranked(op.getOnValue(), 0)) {
    return op.emitOpError() << "requires on_value to be a scalar";
  }
  if (!IsOfRankOrUnranked(op.getOffValue(), 0)) {
    return op.emitOpError() << "requires off_value to be a scalar";
  }

  DenseIntElementsAttr depth_attr;
  if (matchPattern(op.getDepth(), m_Constant(&depth_attr))) {
    if (depth_attr.getType().getRank() != 0)
      return op.emitOpError() << "requires depth to be a scalar";
    int64_t depth = depth_attr.getValues<APInt>()[0].getSExtValue();
    if (depth < 0) {
      return op.emitOpError() << "depth must be non-negative, got: " << depth;
    }
  }

  return success();
}

static TensorType InferOneHotOpType(Value indices, Value depth, Value on_value,
                                    Value off_value, IntegerAttr axis) {
  int64_t axis_val = axis.getInt();
  Type element_ty = on_value.getType().cast<TensorType>().getElementType();
  auto unranked_ty = UnrankedTensorType::get(element_ty);
  if (axis_val < -1) return unranked_ty;

  auto indices_ty = indices.getType().dyn_cast<RankedTensorType>();
  if (!indices_ty) return unranked_ty;

  auto shape = llvm::to_vector<2>(indices_ty.getShape());
  if (axis_val == -1) axis_val = shape.size();

  int64_t depth_val = ShapedType::kDynamic;
  DenseIntElementsAttr depth_attr;
  if (matchPattern(depth, m_Constant(&depth_attr)) &&
      depth_attr.getNumElements() == 1)
    depth_val = (*depth_attr.begin()).getSExtValue();
  shape.insert(shape.begin() + axis_val, depth_val);
  return tensorflow::GetTypeFromTFTensorShape(shape, element_ty);
}

void OneHotOp::build(OpBuilder &builder, OperationState &result, Value indices,
                     Value depth, Value on_value, Value off_value,
                     IntegerAttr axis) {
  build(builder, result,
        InferOneHotOpType(indices, depth, on_value, off_value, axis), indices,
        depth, on_value, off_value, axis);
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

LogicalResult PackOp::verify() {
  PackOp op = *this;
  // TODO(hinsu): Convert variadic length attributes to derived attributes.
  Operation::operand_range values = op.getValues();

  if (failed(VerifyTypesCompatibility(values,
                                      /*mask_one_dim=*/false,
                                      op.getOperation()))) {
    return failure();
  }

  int64_t inputs_rank = -1;
  for (Value value : values) {
    if (auto ty = value.getType().dyn_cast<RankedTensorType>()) {
      // Exit early as input types are verified to be compatible so all ranked
      // tensors have the same rank.
      inputs_rank = ty.getRank();
      break;
    }
  }
  if (inputs_rank == -1) return success();

  // The values can be packed along any of the dimensions between 0 and
  // inputs rank, inclusive. Also, as the negative axis values wrap around so
  // the axis value range is [-(R+1), R+1).
  int64_t range_begin = -inputs_rank - 1;  // Inclusive
  int64_t range_end = inputs_rank + 1;     // Exclusive
  int64_t axis = op.getAxis();
  if (axis < range_begin || axis >= range_end) {
    return op.emitError() << "attribute 'axis' should be within range ["
                          << range_begin << ", " << range_end
                          << "); actual value: " << axis;
  }

  return success();
}

OpFoldResult PackOp::fold(FoldAdaptor) {
  // Fold pack operation if it computes the input tensor shape:
  //
  //   %shape  = tf.Shape(%arg)                    // [? x ...]
  //   %dim0   = tf.StridedSlice(%shape, 0, 1, 1)  // get unknown dim0 value
  //   %pack   = tf.Pack(dim0, ...) { axis = 0 }   // [? x ...]
  //
  // Where `...` are some statically known dimensions. In this case %pack can be
  // replaced with a %shape. This is a common pattern in models with a dynamic
  // batch size.

  // Pack operation should pack at least two values.
  if (getValues().size() < 2) return {};

  // Dimensions packed along axis = 0 (pack scalars into vector).
  if (getAxis() != 0) return {};

  // First packed value is defined by a strided slice operation.
  auto slice_op =
      dyn_cast_or_null<StridedSliceOp>(getValues()[0].getDefiningOp());
  if (!slice_op) return {};

  // Input to the slice op is defined by shape operation.
  auto shape_op =
      dyn_cast_or_null<ShapeOp>(slice_op.getInput().getDefiningOp());
  if (!shape_op) return {};

  // Input tensor, which shape is reconstructed by the pack operation.
  Value tensor = shape_op.getInput();

  // All masks are `0` except `shrink_axis_mask` which is equal to `1` (slicing
  // scalar value from input vector).
  if (slice_op.getBeginMask() != 0 || slice_op.getEllipsisMask() != 0 ||
      slice_op.getEndMask() != 0 || slice_op.getNewAxisMask() != 0 ||
      slice_op.getShrinkAxisMask() != 1)
    return {};

  // Returns a value if the `value` is defined by a ConstOp with a single
  // integer element in it and has an expected rank.
  auto get_const_int = [](Value value,
                          int expected_rank) -> std::optional<int64_t> {
    auto const_op = dyn_cast_or_null<ConstOp>(value.getDefiningOp());
    if (!const_op) return std::nullopt;

    auto value_attr = const_op.getValue().dyn_cast<DenseIntElementsAttr>();
    if (!value_attr || value_attr.getNumElements() != 1) return std::nullopt;

    auto value_ty = value_attr.getType();
    if (!value_ty.hasRank() || value_ty.getRank() != expected_rank)
      return std::nullopt;

    auto splat = value_attr.getSplatValue<IntegerAttr>();
    return splat.getValue().getSExtValue();
  };

  // All other packed values are scalar constants.
  SmallVector<int64_t, 4> packed_dims;
  packed_dims.reserve(getValues().size() - 1);
  for (Value operand : llvm::drop_begin(getValues(), 1)) {
    if (auto dim = get_const_int(operand, /*expected_rank=*/0)) {
      packed_dims.push_back(*dim);
    } else {
      return {};
    }
  }

  // Slice exactly the first shape dimension:
  //   begin = [0] end = [1], strides = [1]
  auto begin = get_const_int(slice_op.getBegin(), /*expected_rank=*/1);
  auto end = get_const_int(slice_op.getEnd(), /*expected_rank=*/1);
  auto strides = get_const_int(slice_op.getStrides(), /*expected_rank=*/1);
  if (!begin.has_value() || !end.has_value() || !strides.has_value() ||
      *begin != 0 || *end != 1 || *strides != 1)
    return {};

  // First tensor dimension is dynamic.
  auto arg_ty = tensor.getType().dyn_cast<ShapedType>();
  if (!arg_ty || !arg_ty.hasRank() || arg_ty.getNumDynamicDims() != 1 ||
      !arg_ty.isDynamicDim(0))
    return {};

  // Argument tensor rank is equal to the number of packed dimensions.
  if (arg_ty.getRank() != getValues().size()) return {};

  // All other dimensions are statically known and equal to packed dims.
  auto arg_dims = llvm::drop_begin(arg_ty.getShape(), 1);
  if (!std::equal(arg_dims.begin(), arg_dims.end(), packed_dims.begin()))
    return {};

  // Replace %pack with %shape.
  return slice_op.getInput();
}

// Convert Pack to Reshape when there is only one operand to be packed.
// For example,
//
//   %0 = tf.Pack(%input) {axis = 0} // %input : tensor<2x3xf32>
//
// can be canonicalized to
//
//   %shape = "tf.Const"() {value = dense<[1, 2, 3]> : tensor<3xi64>}
//   %0 = tf.Reshape(%input, %shape)
struct ConvertPackToReshape : public OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOp pack_op,
                                PatternRewriter &rewriter) const override {
    // Check if there is only one operand to be packed.
    if (pack_op.getN() != 1) {
      return failure();
    }

    // Check if input and output are static.
    auto input_ty = pack_op.getOperand(0).getType().cast<ShapedType>();
    auto output_ty = pack_op.getOutput().getType().cast<ShapedType>();
    if (!input_ty.hasStaticShape() || !output_ty.hasStaticShape()) {
      return failure();
    }

    // Create constant shape for reshape.
    auto type = tensorflow::GetTypeFromTFTensorShape(
        output_ty.getRank(), rewriter.getIntegerType(64));
    auto shape_attr = DenseIntElementsAttr::get(type, output_ty.getShape());
    auto shape = rewriter.create<ConstOp>(pack_op.getLoc(), shape_attr);

    // TODO(b/173622615): Remove after fixed.
    ReplaceTfOpWithNewOp<ReshapeOp>(rewriter, pack_op, output_ty,
                                    pack_op.getOperand(0), shape);
    return success();
  }
};

void PackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<ConvertPackToReshape>(context);
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult PadOp::FoldOperandsPermutation(ArrayRef<int64_t> permutation) {
  // Paddings must be defined by a constant operation.
  auto paddings_op =
      dyn_cast_or_null<TF::ConstOp>(getPaddings().getDefiningOp());
  if (!paddings_op) return failure();

  auto paddings_value = paddings_op.getValue().dyn_cast<DenseElementsAttr>();
  if (!paddings_value ||
      paddings_value.getNumElements() != permutation.size() * 2)
    return failure();

  SmallVector<int32_t, 8> shuffled_paddings(paddings_value.getNumElements());
  for (const auto &index_pair :
       llvm::enumerate(paddings_value.getValues<APInt>())) {
    size_t outer_idx = index_pair.index() / 2;
    size_t inner_idx = index_pair.index() % 2;

    shuffled_paddings[permutation[outer_idx] * 2 + inner_idx] =
        index_pair.value().getSExtValue();
  }

  // Add constant operation with a new paddings.
  OpBuilder builder(getOperation());
  auto type = tensorflow::GetTypeFromTFTensorShape(
      paddings_value.getType().getShape(), builder.getIntegerType(32));
  auto values = mlir::DenseIntElementsAttr::get(type, shuffled_paddings);
  auto shuffled_paddings_op = builder.create<TF::ConstOp>(getLoc(), values);

  // Use new paddings.
  setOperand(1, shuffled_paddings_op);

  // Change the result type.
  getResult().setType(ShuffleRankedTensorType(getResult().getType(),
                                              ReversePermutation(permutation))
                          .cast<TensorType>());

  return success();
}

//===----------------------------------------------------------------------===//
// ParseExampleV2Op
//===----------------------------------------------------------------------===//

LogicalResult ParseExampleV2Op::verify() {
  ParseExampleV2Op op = *this;
  // NOTE(mrry): This validates properties of an op that would previously be
  // validated by the TensorFlow OpDef type checker. In addition to these
  // checks, the shape inference function for ParseExampleV2 validates the
  // consistency of the argument and result types.

  // Validate dense variadic input and output lengths.
  // NOTE(mrry): The Tdense attr is derived from dense_defaults, so we
  // do not need to validate dense_defaults.
  auto dense_types_count =
      std::distance(op.getTdense().begin(), op.getTdense().end());
  auto dense_values_count =
      std::distance(op.getDenseValues().begin(), op.getDenseValues().end());
  if (dense_values_count != dense_types_count) {
    return op.emitError() << "output 'dense_values' should have same length "
                          << "as attribute 'Tdense'";
  }

  // Validate sparse variadic output lengths.
  // NOTE(mrry): The sparse_types attr is derived from sparse_values, so we
  // do not need to validate sparse_values.
  auto sparse_types_count =
      std::distance(op.getSparseTypes().begin(), op.getSparseTypes().end());
  if (op.getNumSparse() != sparse_types_count) {
    return op.emitError() << "attribute 'num_sparse' should be the same as "
                          << "the length of attribute 'sparse_types'";
  }
  if (op.getSparseIndices().size() != sparse_types_count) {
    return op.emitError() << "output 'sparse_indices' should have same length "
                          << "as attribute 'sparse_types'";
  }
  if (op.getSparseShapes().size() != sparse_types_count) {
    return op.emitError() << "output 'sparse_shapes' should have same length "
                          << "as attribute 'sparse_types'";
  }

  // Validate ragged variadic output lengths.
  auto ragged_value_types_count = std::distance(
      op.getRaggedValueTypes().begin(), op.getRaggedValueTypes().end());
  auto ragged_split_types_count = std::distance(
      op.getRaggedSplitTypes().begin(), op.getRaggedSplitTypes().end());
  if (ragged_value_types_count != ragged_split_types_count) {
    return op.emitError() << "attribute 'ragged_value_types' should have same "
                          << "length as attribute 'ragged_split_types'";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PartitionedCallOp
//===----------------------------------------------------------------------===//

template <typename CallOpClass>
static LogicalResult VerifyPartitionedCall(CallOpClass op,
                                           SymbolTableCollection &symbolTable) {
  SymbolRefAttr func = op->getAttr("f").template cast<SymbolRefAttr>();
  auto function = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(op, func);
  if (!function) {
    return op.emitError("'f' attribute refers to an undefined function: ")
           << func;
  }

  FunctionType function_ty = function.getFunctionType();
  int func_arg_count = function_ty.getNumInputs();
  int arg_count = op.getArgs().size();

  if (arg_count != func_arg_count) {
    return op.emitError() << "argument count mismatch: 'args' has " << arg_count
                          << " arguments, but '" << func << "' expects "
                          << func_arg_count;
  }

  return success();
}

LogicalResult PartitionedCallOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  return VerifyPartitionedCall(*this, symbolTable);
}
LogicalResult StatefulPartitionedCallOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  return VerifyPartitionedCall(*this, symbolTable);
}
LogicalResult TPUPartitionedCallOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  return VerifyPartitionedCall(*this, symbolTable);
}

//===----------------------------------------------------------------------===//
// PowOp
//===----------------------------------------------------------------------===//

OpFoldResult PowOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto constant_y = operands[1].dyn_cast_or_null<DenseFPElementsAttr>();
  if (constant_y && constant_y.isSplat()) {
    APFloat y_value = constant_y.getSplatValue<APFloat>();
    auto output_type = getType().cast<ShapedType>();
    if (y_value.isZero() && output_type.hasStaticShape()) {
      return DenseElementsAttr::get(
          output_type,
          FloatAttr::get(output_type.getElementType(), /*value=*/1.0));
    }
    if (y_value.isExactlyValue(1.0)) {
      return getX();
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QuantizeAndDequantizeV2Op
//===----------------------------------------------------------------------===//

void QuantizeAndDequantizeV2Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<QuantizeAndDequantizeV2ToQuantizeAndDequantizeV4>(context);
}

//===----------------------------------------------------------------------===//
// QrOp
//===----------------------------------------------------------------------===//

// Verifies that,
//
// * Input type, if ranked, must have at least 2 dimensions and at most
//   INT32_MAX dimensions.
//
LogicalResult QrOp::verify() {
  QrOp op = *this;
  auto ttype = op.getInput().getType().cast<TensorType>();
  if (!ttype.hasRank()) return success();
  if (!HasRankAtLeast(op.getInput(), 2))
    return op.emitOpError(
        "requires ranked input tensor to be of rank 2 or more");
  if (!HasRankAtMost(op.getInput(), std::numeric_limits<int32_t>::max()))
    return op.emitOpError(
        "requires ranked input tensor to be of rank INT32_MAX or less");

  return success();
}

//===----------------------------------------------------------------------===//
// ReadVariableOp
//===----------------------------------------------------------------------===//

void ReadVariableOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<ReadVariableOfCast>(context);
}

//===----------------------------------------------------------------------===//
// RandomUniformOp
//===----------------------------------------------------------------------===//

LogicalResult RandomUniformOp::verify() {
  RandomUniformOp op = *this;
  if (!IsOfRankOrUnranked(op.getShape(), 1))
    return op.emitOpError("shape must be 1D tensor");
  return success();
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
  FloatOrInt diff = limit - start;
  if (std::is_integral<FloatOrInt>::value) {
    return ((std::abs(diff) + std::abs(delta) - 1) / std::abs(delta));
  }
  return std::ceil(std::abs(diff / delta));
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

void RangeOp::build(OpBuilder &builder, OperationState &result, Value start,
                    Value limit, Value delta) {
  assert(start.getType() == limit.getType());
  assert(start.getType() == delta.getType());
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
        tensorflow::GetTypeFromTFTensorShape(
            size.getSExtValue(),
            start.getType().cast<TensorType>().getElementType()),
        start, limit, delta);
  }
  return RangeOp::build(
      builder, result,
      tensorflow::GetTypeFromTFTensorShape(
          {-1}, start.getType().cast<TensorType>().getElementType()),
      start, limit, delta);
}

OpFoldResult RangeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.size() == 3);
  auto start_tensor = operands[0].dyn_cast_or_null<ElementsAttr>();
  auto limit_tensor = operands[1].dyn_cast_or_null<ElementsAttr>();
  auto delta_tensor = operands[2].dyn_cast_or_null<ElementsAttr>();
  if (!(start_tensor && limit_tensor && delta_tensor)) return nullptr;

  // Operands should all be scalars
  assert(start_tensor.getType().getRank() == 0 &&
         limit_tensor.getType().getRank() == 0 &&
         delta_tensor.getType().getRank() == 0);
  Type elem_type = getType().cast<ShapedType>().getElementType();
  if (elem_type.isSignlessInteger() || elem_type.isUnsignedInteger()) {
    auto start_attr = start_tensor.getValues<IntegerAttr>()[0];
    auto limit_attr = limit_tensor.getValues<IntegerAttr>()[0];
    auto delta_attr = delta_tensor.getValues<IntegerAttr>()[0];
    int num_elements;
    if (elem_type.isUnsignedInteger()) {
      uint64_t start = start_attr.getUInt();
      uint64_t limit = limit_attr.getUInt();
      uint64_t delta = delta_attr.getUInt();
      assert(start <= (uint64_t)INT_MAX);
      assert(limit <= (uint64_t)INT_MAX);
      assert(delta <= (uint64_t)INT_MAX);
      num_elements =
          GetLengthOfRange(static_cast<int>(start), static_cast<int>(limit),
                           static_cast<int>(delta));
    } else {
      num_elements = GetLengthOfRange(start_attr.getInt(), limit_attr.getInt(),
                                      delta_attr.getInt());
    }
    return BuildConstRangeTensor(elem_type, num_elements, start_attr,
                                 delta_attr);
  } else if (elem_type.isa<FloatType>()) {
    auto start_attr = start_tensor.getValues<FloatAttr>()[0];
    auto limit_attr = limit_tensor.getValues<FloatAttr>()[0];
    auto delta_attr = delta_tensor.getValues<FloatAttr>()[0];
    const int num_elements = GetLengthOfRange(start_attr.getValueAsDouble(),
                                              limit_attr.getValueAsDouble(),
                                              delta_attr.getValueAsDouble());
    return BuildConstRangeTensor(elem_type, num_elements, start_attr,
                                 delta_attr);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

void RankOp::build(OpBuilder &builder, OperationState &result, Value input) {
  return RankOp::build(
      builder, result,
      tensorflow::GetTypeFromTFTensorShape({}, builder.getIntegerType(32)),
      input);
}

// This will create a constant value for RankOp of a ranked tensor.
OpFoldResult RankOp::fold(FoldAdaptor) {
  auto type = getInput().getType();
  auto ranked_type = type.dyn_cast<RankedTensorType>();
  if (!ranked_type) return {};

  // DenseIntElementsAttr::get requires the output type be ranked with static
  // shape.
  auto output_type = getType().dyn_cast<RankedTensorType>();
  if (!output_type || !output_type.hasStaticShape()) return {};

  int32_t rank = ranked_type.getRank();
  return DenseIntElementsAttr::get(output_type, rank);
}

//===----------------------------------------------------------------------===//
// RealDivOp
//===----------------------------------------------------------------------===//

void RealDivOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<RealDivWithSqrtDivisor, RealDivWithConstDivisor>(context);
}

OpFoldResult RealDivOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return IdentityArithmeticOpFolder<RealDivOp>(*this, operands);
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

void ReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<ReluOfMinimum6ToRelu6>(context);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

namespace {
using ReshapeErrorHandler =
    llvm::function_ref<LogicalResult(const llvm::Twine &)>;

LogicalResult GetReshapeOutputType(Value tensor, Value shape,
                                   ReshapeErrorHandler error_handler,
                                   TensorType &output_ty) {
  auto tensor_ty = tensor.getType().cast<TensorType>();
  auto element_ty = tensor_ty.getElementType();
  output_ty = UnrankedTensorType::get(element_ty);

  auto shape_ty = shape.getType().dyn_cast<RankedTensorType>();
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
  for (const auto &dim : llvm::enumerate(shape_attr.getValues<APInt>())) {
    const int64_t size = dim.value().getSExtValue();
    if (size == tensorflow::kTFDynamicSize ||  // NOLINT
        size == ShapedType::kDynamic) {        // NOLINT
      if (unknown_index != -1)
        return error_handler(llvm::formatv(
            "requires 'shape' to have at most one dynamic dimension, but got "
            "multiple dynamic dimensions at indices {0} and {1}",
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

  if (!tensor_ty.hasStaticShape()) {
    output_ty =
        tensorflow::GetTypeFromTFTensorShape(output_ty_shape, element_ty);
    return success();
  }

  // Compute the value of the unknown dimension.
  if (unknown_index != -1) {
    // Compute number of elements in tensor shape.
    int64_t tensor_ty_size = 1;
    bool tensor_ty_zero_dim = false;
    for (const auto &dim : tensor_ty.getShape()) {
      if (dim > 0 || !shape_ty_zero_dim) {
        tensor_ty_size *= dim;
      } else {
        tensor_ty_zero_dim = true;
      }
    }

    const int64_t missing_dim = tensor_ty_size / shape_ty_size;
    if (!tensor_ty_zero_dim && shape_ty_size * missing_dim != tensor_ty_size)
      return error_handler(
          llvm::formatv("requires 'tensor' number of elements be a multiple of "
                        "{0}, but got {1}",
                        shape_ty_size, tensor_ty_size));

    // Set the unknown dimension such that total number of elements remain
    // constant.
    output_ty_shape[unknown_index] = missing_dim;
  }
  output_ty = tensorflow::GetTypeFromTFTensorShape(output_ty_shape, element_ty);

  return success();
}
}  // namespace

LogicalResult ReshapeOp::verify() {
  ReshapeOp op = *this;
  auto error_handler = [&op](const llvm::Twine &message) -> LogicalResult {
    return op.emitOpError() << message;
  };
  TensorType expected_ty;
  if (failed(GetReshapeOutputType(op.getTensor(), op.getShape(), error_handler,
                                  expected_ty)))
    return failure();

  auto output_ty = op.getType().dyn_cast<RankedTensorType>();
  if (!output_ty) return success();
  auto tensor_ty = op.getTensor().getType().cast<TensorType>();
  if (output_ty.hasStaticShape() && tensor_ty.hasStaticShape()) {
    const int64_t output_ty_size = output_ty.getNumElements();
    const int64_t tensor_ty_size = tensor_ty.getNumElements();
    if (tensor_ty_size != output_ty_size)
      return op.emitOpError() << "requires 'output' number of elements to "
                                 "match 'tensor' number of elements, but got "
                              << output_ty_size << " and " << tensor_ty_size;
  }

  if (!AreCastCompatible({output_ty, expected_ty}))
    return op.emitOpError()
           << "requires 'output' type " << output_ty
           << " to be cast compatible with expected type " << expected_ty;

  return success();
}

// Currently there are use cases that rely on partial evaluation of the `shape`
// operand, so InferTypeOpInterface is not used (along with generated builder of
// the same signature).
void ReshapeOp::build(OpBuilder &builder, OperationState &result, Value tensor,
                      Value shape) {
  auto error_handler = [&result](const llvm::Twine &message) {
    return mlir::emitError(result.location) << message;
  };
  TensorType output_ty;
  if (failed(GetReshapeOutputType(tensor, shape, error_handler, output_ty)))
    return;

  return ReshapeOp::build(builder, result, output_ty, tensor, shape);
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<RedundantReshape, ReshapeToSelfShape>(context);
}

OpFoldResult ReshapeOp::fold(FoldAdaptor) {
  Value tensor = this->getTensor();

  // Fold reshape if operand and result types are the same and all dimensions
  // are statically known (no-op reshape).
  auto result_ty = getType().dyn_cast<ShapedType>();
  if (result_ty && result_ty.hasStaticShape() &&
      result_ty == tensor.getType()) {
    return tensor;
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// Verifies a few extra requirements on SelectOp:
// (1) `then` and `else` must have same shape
// (2) At least one of the following must be true:
//     (a) `cond` has the same rank as `then` and `else`
//     (b) `cond` is a scalar
//     (c) `cond` is a vector AND `then` and `else` are non-scalar with their
//         first dimension equal to `cond`.
LogicalResult SelectOp::verify() {
  SelectOp op = *this;
  auto then_tensor = op.getThenValue().getType().cast<TensorType>();
  auto else_tensor = op.getElseValue().getType().cast<TensorType>();
  // Check (1).
  if (!AreCastCompatible({then_tensor, else_tensor}))
    return op.emitOpError() << "requires t and e have compatible shapes";

  // Get data rank (if exists).
  int data_rank;
  // If data is unranked or data_rank is 0, this will remain -2. Otherwise
  // refers to first dimension of then and/or else.
  int64_t data_first_dim = -2;
  bool then_has_rank = then_tensor.hasRank();
  bool else_has_rank = else_tensor.hasRank();
  if (then_has_rank && else_has_rank) {
    data_rank = then_tensor.getRank();
    if (then_tensor.getRank() > 0)
      data_first_dim = then_tensor.getShape().front();
    if (else_tensor.getRank() > 0)
      data_first_dim = std::max(else_tensor.getShape().front(), data_first_dim);
  } else if (then_has_rank) {
    data_rank = then_tensor.getRank();
    if (then_tensor.getRank() > 0)
      data_first_dim = then_tensor.getShape().front();
  } else if (else_has_rank) {
    data_rank = else_tensor.getRank();
    if (else_tensor.getRank() > 0)
      data_first_dim = else_tensor.getShape().front();
  } else {
    // Neither has a rank.
    return success();
  }

  auto cond_tensor = op.getCondition().getType().dyn_cast<RankedTensorType>();
  if (!cond_tensor) return success();
  auto cond_rank = cond_tensor.getRank();
  // Check (2a) and (2b).
  if (cond_rank == 0 || cond_rank == data_rank) return success();
  // Check (2c).
  if (cond_rank == 1) {
    auto cond_shape = cond_tensor.getShape().front();
    if (data_rank == 0) {
      return op.emitOpError()
             << "requires that t and e are nonscalar when pred is a vector";
    }
    // We know `data` tensor has a rank of at least 1.
    if (data_first_dim != ShapedType::kDynamic &&
        cond_shape != ShapedType::kDynamic && data_first_dim != cond_shape) {
      return op.emitOpError() << "requires that, when pred is a vector, the "
                                 "shape matches the first dimension of t and e";
    }
    return success();
  }
  // None of (2a,b,c) were true; fail.
  return op.emitOpError() << "requires that pred is a scalar OR has the same "
                             "rank as t and e OR is a vector";
}

//===----------------------------------------------------------------------===//
// SelectV2Op
//===----------------------------------------------------------------------===//

static Type InferSelectV2OpType(Value condition, Value e, Value t) {
  Type element_ty = e.getType().cast<TensorType>().getElementType();
  auto unranked_ty = UnrankedTensorType::get(element_ty);

  Type broadcasted_ty =
      OpTrait::util::getBroadcastedType(e.getType(), t.getType());
  if (!broadcasted_ty) return unranked_ty;

  auto cond_ranked_ty = condition.getType().dyn_cast<RankedTensorType>();
  auto broadcasted_ranked_ty = broadcasted_ty.dyn_cast<RankedTensorType>();
  if (!cond_ranked_ty || !broadcasted_ranked_ty) return unranked_ty;

  // Explicitly get broadcasted output type as element types of condition may
  // not be same as the broadcated type's element type.
  SmallVector<int64_t, 4> result_shape;
  if (!OpTrait::util::getBroadcastedShape(cond_ranked_ty.getShape(),
                                          broadcasted_ranked_ty.getShape(),
                                          result_shape))
    return unranked_ty;
  return tensorflow::GetTypeFromTFTensorShape(result_shape, element_ty);
}

void SelectV2Op::build(OpBuilder &builder, OperationState &result,
                       Value condition, Value e, Value t) {
  build(builder, result, InferSelectV2OpType(condition, e, t), condition, e, t);
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

namespace {
// Validates Shape/ShapeN/VariableShape operand and associated result types.
LogicalResult VerifyShapeOperandAndResult(Operation *op, Type operand_type,
                                          Type result_type,
                                          int variadic_idx = -1) {
  std::string variadic_idx_str =
      variadic_idx < 0 ? "" : llvm::formatv(" #{0}", variadic_idx).str();

  auto result_ranked_type = result_type.dyn_cast<RankedTensorType>();
  if (!result_ranked_type) return success();
  if (result_ranked_type.getShape().size() != 1)
    return op->emitOpError("requires 1D type for result") << variadic_idx_str;

  auto operand_ranked_type = operand_type.dyn_cast_or_null<RankedTensorType>();
  if (operand_ranked_type) {
    // The operand is a ranked tensor.
    if (result_ranked_type.hasStaticShape() &&
        !operand_ranked_type.getShape().empty() &&
        result_ranked_type.getDimSize(0) !=
            operand_ranked_type.getShape().size())
      return op->emitOpError("requires dimension size of result")
             << variadic_idx_str << " to match rank of operand"
             << variadic_idx_str;
  } else if (result_ranked_type.hasStaticShape()) {
    // The operand is an unranked tensor, print a warning if the result
    // is static.
    // Note: We do not handle this situation as an error, this would be too
    // restrictive due to incompleteness of shape inference at this point.
    mlir::InFlightDiagnostic diag =
        mlir::emitWarning(op->getLoc(), "has static shape result");
    if (op->getContext()->shouldPrintOpOnDiagnostic()) {
      diag.attachNote(op->getLoc())
          .append("see current operation: ")
          .appendOp(*op, OpPrintingFlags().printGenericOpForm());
    }
    diag << variadic_idx_str << " for unranked operand" << variadic_idx_str;
  }

  Type element_type = result_ranked_type.getElementType();
  if (!element_type.isSignlessInteger(32) &&
      !element_type.isSignlessInteger(64))
    return op->emitOpError("requires int32 or int64 return type for result")
           << variadic_idx_str;

  return success();
}
}  // anonymous namespace

LogicalResult ShapeOp::verify() {
  ShapeOp op = *this;
  return VerifyShapeOperandAndResult(op, op.getInput().getType(), op.getType());
}

// Converts shape of the given type to attribute if it is of ranked tensor type.
// Returned attribute has integer elements of the given width.
static Attribute ConvertShapeToAttr(Type input_ty, int out_width) {
  auto ranked_ty = input_ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty || !ranked_ty.hasStaticShape()) return {};

  auto shape = ranked_ty.getShape();
  int rank = shape.size();

  SmallVector<APInt, 4> dimensions;
  dimensions.reserve(rank);
  for (int i = 0; i < rank; ++i)
    dimensions.push_back(APInt(out_width, shape[i]));

  auto result_type = tensorflow::GetTypeFromTFTensorShape(
      {rank}, IntegerType::get(input_ty.getContext(), out_width));
  return DenseElementsAttr::get(result_type, dimensions);
}

OpFoldResult ShapeOp::fold(FoldAdaptor) {
  int width =
      getType().cast<ShapedType>().getElementType().getIntOrFloatBitWidth();
  return ConvertShapeToAttr(getOperand().getType(), width);
}

void ShapeOp::build(OpBuilder &builder, OperationState &result, Value input,
                    BoolAttr use32Bit) {
  auto rankedTensorType = input.getType().dyn_cast<RankedTensorType>();
  int64_t rank = rankedTensorType ? rankedTensorType.getRank() : -1;
  auto out_type = use32Bit.getValue() ? builder.getIntegerType(32)
                                      : builder.getIntegerType(64);
  return ShapeOp::build(builder, result,
                        tensorflow::GetTypeFromTFTensorShape({rank}, out_type),
                        input);
}

//===----------------------------------------------------------------------===//
// ShapeNOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeNOp::verify() {
  ShapeNOp op = *this;
  const size_t num_tensors = op.getN();

  if (op.getNumOperands() != num_tensors)
    return op.emitOpError() << "requires " << num_tensors << " operand(s), got "
                            << op.getNumOperands() << " operand(s)";

  if (op.getNumResults() != num_tensors)
    return op.emitOpError() << "requires " << num_tensors << " result(s), got "
                            << op.getNumResults() << " result(s)";

  for (auto i : llvm::seq<uint64_t>(0, num_tensors)) {
    auto verification = VerifyShapeOperandAndResult(
        op, op.getOperand(i).getType(), op.getResult(i).getType(), i);
    if (failed(verification)) return verification;
  }

  return success();
}

namespace {
// Canonicalization pattern for ShapeNOp that don't have all
// static input shapes. Replacing output values corresponding to static input
// types may enable optimizations in users of the values.
class ShapeNPartialStaticInputShape : public OpRewritePattern<ShapeNOp> {
  using OpRewritePattern<ShapeNOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ShapeNOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    int width = getElementTypeOrSelf(op.getType(0)).getIntOrFloatBitWidth();

    SmallVector<Value, 4> results(op.getNumOperands());
    SmallVector<int64_t, 4> dynamic_indices;
    SmallVector<Value, 4> dynamic_inputs;
    SmallVector<Type, 4> result_types;
    for (const auto &e : llvm::enumerate(op.getOperands())) {
      if (Attribute result = ConvertShapeToAttr(e.value().getType(), width)) {
        results[e.index()] = rewriter.create<TF::ConstOp>(op.getLoc(), result);
      } else {
        dynamic_indices.push_back(e.index());
        dynamic_inputs.push_back(e.value());
        result_types.push_back(op.getType(e.index()));
      }
    }

    if (dynamic_inputs.size() == op.getNumOperands()) {
      // Cannot canonicalize ShapeN if all inputs are dynamic.
      return failure();
    }

    // Create a ShapeNOp for all dynamic inputs.
    if (!dynamic_inputs.empty()) {
      auto dynamic_shape_n = rewriter.create<TF::ShapeNOp>(
          op.getLoc(), result_types, dynamic_inputs);
      for (auto index_result :
           llvm::zip(dynamic_indices, dynamic_shape_n.getResults())) {
        results[std::get<0>(index_result)] = std::get<1>(index_result);
      }
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

// Canonicalize ShapeNOp to ShapeOp if there is only one operand.
class ShapeNToShape : public OpRewritePattern<ShapeNOp> {
  using OpRewritePattern<ShapeNOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ShapeNOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 1) {
      return failure();
    }
    auto shape = rewriter.create<TF::ShapeOp>(op.getLoc(), op.getType(0),
                                              op.getOperand(0));
    rewriter.replaceOp(op, {shape});
    return success();
  }
};
}  // namespace

void ShapeNOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<ShapeNToShape, ShapeNPartialStaticInputShape>(context);
}

//===----------------------------------------------------------------------===//
// SizeOp
//===----------------------------------------------------------------------===//

// Verifies that,
//
// * Input type, if is a ranked tensor, has at most INT32_MAX dimensions.
//
LogicalResult SizeOp::verify() {
  SizeOp op = *this;
  if (!HasRankAtMost(op.getInput(), std::numeric_limits<int32_t>::max()))
    return op.emitOpError(
        "requires ranked input tensor to be of rank INT32_MAX or less");

  // Output type needs to be scalar.
  if (!IsOfRankOrUnranked(op.getOutput(), /*rank=*/0))
    return op.emitOpError("requires scalar output");

  return success();
}

OpFoldResult SizeOp::fold(FoldAdaptor) {
  ShapedType output_type = getType().cast<ShapedType>();
  if (!output_type.hasRank()) return {};
  ShapedType input_type = getOperand().getType().cast<ShapedType>();
  if (!input_type.hasStaticShape()) return {};
  int size = input_type.getNumElements();
  return DenseElementsAttr::get(
      output_type,
      IntegerAttr::get(output_type.getElementType(), /*value=*/size));
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// Verifies that:
//
// - operands begin and size are 1D with the same number of elements.
// - if the input is a ranked tensor, the rank of the input equals the number
//   of elements in operands begin and size.
// - if begin are constants, that
//   0 <= begin[i] <= begin[i] + size[i] <= input_ty.getShape()[i]
//   and
//   size[i] == output_ty.getShape()[i]
// - if begins aren't constant but the input is a ranked tensor, that
//   size[i] <= input_ty.getShape()[i]
// - output rank is the same as input rank
//
LogicalResult SliceOp::verify() {
  SliceOp op = *this;
  RankedTensorType begin_ty = GetRankedTensorTypeForOperand(op.getBegin());
  if (begin_ty && begin_ty.getRank() != 1) {
    return op.emitOpError() << "requires begin operand to be 1D tensor";
  }

  RankedTensorType size_ty = GetRankedTensorTypeForOperand(op.getSize());
  if (size_ty && size_ty.getRank() != 1) {
    return op.emitOpError() << "requires size operand to be 1D tensor";
  }

  if (!begin_ty || !size_ty || !begin_ty.hasStaticShape() ||
      !size_ty.hasStaticShape())
    return success();

  if (begin_ty.getNumElements() != size_ty.getNumElements()) {
    return op.emitOpError() << "requires begin and size operands to have the"
                               " same number of elements";
  }

  auto input_ty = op.getInput().getType().dyn_cast<RankedTensorType>();
  if (input_ty && begin_ty.getNumElements() != input_ty.getRank()) {
    return op.emitOpError() << "requires number of elements in begin and size "
                               "are equal to input rank";
  }

  auto output_ty = op.getOutput().getType().dyn_cast<RankedTensorType>();
  if (output_ty && input_ty && output_ty.getRank() != input_ty.getRank()) {
    return op.emitOpError()
           << "requires output to have the same rank as input, but got input "
              "rank "
           << input_ty.getRank() << " and output rank " << output_ty.getRank();
  }

  DenseIntElementsAttr begin_indices;
  if (matchPattern(op.getBegin(), m_Constant(&begin_indices))) {
    DenseIntElementsAttr slice_sizes;
    bool constant_slice_sizes =
        matchPattern(op.getSize(), m_Constant(&slice_sizes));
    int dim = 0;
    // TODO(jpienaar): Reformulate the shape verification below to not use magic
    // constants.
    for (const APInt &raw_begin_index : begin_indices.getValues<APInt>()) {
      int64_t begin_index = raw_begin_index.getSExtValue();
      int64_t input_size =
          input_ty ? input_ty.getShape()[dim] : ShapedType::kDynamic;
      int64_t slice_size =
          constant_slice_sizes
              ? slice_sizes.getValues<APInt>()[dim].getSExtValue()
              : 0;
      int64_t output_size =
          output_ty ? output_ty.getShape()[dim] : ShapedType::kDynamic;

      if (slice_size == -1 && input_size != ShapedType::kDynamic) {
        slice_size = input_size - begin_index;
      }
      if (output_size != ShapedType::kDynamic && constant_slice_sizes &&
          output_size != slice_size) {
        return op.emitOpError()
               << "requires output size to have the same size of slice, got "
                  "slice size "
               << slice_size << " and output size " << output_size;
      }
      if (begin_index < 0 || (input_size != ShapedType::kDynamic &&
                              begin_index + slice_size > input_size)) {
        return op.emitOpError()
               << "requires 0 <= begin[i] <= begin[i] + size[i] <= Di";
      }
      ++dim;
    }
  } else if (input_ty) {
    // If the inputs are ranked, we can do a few more sanity checks.
    DenseIntElementsAttr slice_sizes;
    if (matchPattern(op.getSize(), m_Constant(&slice_sizes))) {
      auto input_shape = input_ty.getShape();
      for (int64_t i = 0; i < input_ty.getRank(); ++i) {
        int64_t slice_size = slice_sizes.getValues<APInt>()[i].getSExtValue();
        int64_t input_size = input_shape[i];
        if (slice_size != -1 && input_size != ShapedType::kDynamic &&
            slice_size > input_size) {
          return op.emitOpError() << "requires size[i] <= Di, even if begin[i] "
                                     "is unknown at compile time";
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

LogicalResult SoftmaxOp::verify() {
  SoftmaxOp op = *this;
  if (!HasRankAtLeast(op.getLogits(), 1)) {
    return op.emitOpError("requires operand to have rank at least 1");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxCrossEntropyWithLogitsOp
//===----------------------------------------------------------------------===//

// Verifies that,
//
// * Input types are broadcast compatible and the broadcasted type has rank two.
//
LogicalResult SoftmaxCrossEntropyWithLogitsOp::verify() {
  SoftmaxCrossEntropyWithLogitsOp op = *this;
  auto broadcasted_ty =
      OpTrait::util::getBroadcastedType(op.getFeatures().getType(),
                                        op.getLabels().getType())
          .dyn_cast_or_null<ShapedType>();
  if (!broadcasted_ty ||
      (broadcasted_ty.hasRank() && broadcasted_ty.getRank() != 2))
    return op.emitOpError(
        "requires features and labels to be broadcast compatible to rank two");

  return success();
}

//===----------------------------------------------------------------------===//
// SpaceToBatchNDOp
//===----------------------------------------------------------------------===//

int64_t SpaceToBatchNDBlockRank(const TensorType block_shape_type,
                                const TensorType paddings_type) {
  if (block_shape_type.hasStaticShape()) {
    return block_shape_type.getShape()[0];
  } else if (paddings_type.hasStaticShape()) {
    return paddings_type.getShape()[0];
  } else {
    return -1;
  }
}

LogicalResult SpaceToBatchNDOp::verify() {
  SpaceToBatchNDOp op = *this;
  const auto input_type = op.getInput().getType().cast<TensorType>();
  const auto block_shape_type = op.getBlockShape().getType().cast<TensorType>();
  const auto paddings_type = op.getPaddings().getType().cast<TensorType>();

  // Check that block_shape has rank 1.
  if (!IsOfRankOrUnranked(op.getBlockShape(), 1)) {
    return op.emitOpError() << "requires rank of block_shape = 1; got "
                            << block_shape_type.getRank();
  }

  // Check that paddings has rank 2.
  if (!IsOfRankOrUnranked(op.getPaddings(), 2)) {
    return op.emitOpError()
           << "requires rank of paddings = 2; got " << paddings_type.getRank();
  }

  // Check that paddings.shape[1]=2.
  if (paddings_type.hasStaticShape() && paddings_type.getShape()[1] != 2) {
    return op.emitOpError() << "requires paddings.shape[1] to be 2; got "
                            << paddings_type.getShape()[1];
  }

  // Check that block_shape and paddings have consistent ranks.
  if (block_shape_type.hasStaticShape() && paddings_type.hasStaticShape() &&
      block_shape_type.getShape()[0] != paddings_type.getShape()[0]) {
    return op.emitOpError()
           << "requires block_shape.shape[0] must equal paddings.shape[0]";
  }

  const int64_t block_rank =
      SpaceToBatchNDBlockRank(block_shape_type, paddings_type);

  // Further checks require block_rank to be known.
  if (block_rank == -1) {
    return success();
  }

  // check that rank of input_type >= block_rank + 1
  if (input_type.hasRank() && input_type.getRank() < 1 + block_rank) {
    return op.emitOpError() << "requires rank of input >= 1 + rank of block";
  }

  ElementsAttr block_shape_attr = nullptr;
  ElementsAttr paddings_attr = nullptr;

  // Check that block_shape[*] >= 1.
  if (matchPattern(op.getBlockShape(), m_Constant(&block_shape_attr))) {
    uint64_t i = 0;
    for (auto block_len : block_shape_attr.getValues<APInt>()) {
      if (block_len.getSExtValue() < 1) {
        return op.emitOpError()
               << "requires all values of block_shape to be >= 1; "
                  "failed for dimension "
               << i;
      }
      ++i;
    }
  }

  // Check that paddings[*] >= 0.
  if (matchPattern(op.getPaddings(), m_Constant(&paddings_attr))) {
    for (uint64_t i = 0; i < block_rank; ++i) {
      const int64_t pad_start =
          paddings_attr.getValues<APInt>()[{i, 0}].getSExtValue();
      const int64_t pad_end =
          paddings_attr.getValues<APInt>()[{i, 1}].getSExtValue();
      if (pad_start < 0 || pad_end < 0) {
        return op.emitOpError()
               << "requires all values of paddings to be >= 0; "
                  "failed for dimension "
               << i;
      }
    }
  }

  // Check that block_shape divides the padded input.
  if (input_type.hasStaticShape() && block_shape_attr && paddings_attr) {
    for (uint64_t i = 0; i < block_rank; ++i) {
      const int64_t input_len = input_type.getShape()[1 + i];
      const int64_t pad_start =
          paddings_attr.getValues<APInt>()[{i, 0}].getSExtValue();
      const int64_t pad_end =
          paddings_attr.getValues<APInt>()[{i, 1}].getSExtValue();
      const int64_t block_len =
          block_shape_attr.getValues<APInt>()[i].getSExtValue();
      if ((input_len + pad_start + pad_end) % block_len != 0) {
        return op.emitOpError()
               << "requires block_shape[i] divides "
                  "input_shape[i + 1] + paddings[i, 0] + paddings[i, 1]; "
                  "failed for i="
               << i;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SparseSoftmaxCrossEntropyWithLogitsOp
//===----------------------------------------------------------------------===//

LogicalResult SparseSoftmaxCrossEntropyWithLogitsOp::verify() {
  SparseSoftmaxCrossEntropyWithLogitsOp op = *this;
  if (!IsOfRankOrUnranked(op.getFeatures(), 2)) {
    return op.emitOpError("requires features operand of rank two");
  }
  if (!IsOfRankOrUnranked(op.getLabels(), 1)) {
    return op.emitOpError("requires labels operand of rank one");
  }
  auto features_ty = op.getFeatures().getType().dyn_cast<RankedTensorType>();
  auto labels_ty = op.getLabels().getType().dyn_cast<RankedTensorType>();
  if (features_ty && labels_ty) {
    int64_t features_batches = features_ty.getDimSize(0);
    int64_t labels_batches = labels_ty.getDimSize(0);
    if (!ShapedType::isDynamic(features_batches) &&
        !ShapedType::isDynamic(labels_batches) &&
        features_batches != labels_batches)
      return op.emitOpError(
          "requires features and labels with matching first dimension");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

// Verifies the input and split dimension operands for tf.Split/tf.SplitV.
// Writes the split dimension's index (adjusted with input rank) via `dim_index`
// if it's a constant.
template <class Op>
LogicalResult VerifySplitInputAndSplitDim(Op op,
                                          std::optional<int64_t> *dim_index) {
  *dim_index = std::nullopt;

  Value split_dim = op.getSplitDim();
  if (auto split_dim_type = split_dim.getType().dyn_cast<RankedTensorType>())
    if (split_dim_type.getRank() != 0)
      return op.emitOpError(
          "split dimension should be an integer scalar tensor");

  // We can perform further verification if the input tensor to be split has
  // known rank and the split dimension tensor is a constant.

  auto input_type =
      op.getValue().getType().template dyn_cast<RankedTensorType>();
  if (!input_type) return success();

  int64_t input_rank = input_type.getRank();
  if (input_rank == 0)
    return op.emitOpError("cannot split scalar input tensor");

  DenseIntElementsAttr split_dim_attr;
  if (!matchPattern(split_dim, m_Constant(&split_dim_attr))) return success();

  int64_t index = (*split_dim_attr.begin()).getSExtValue();

  if (index + input_rank < 0 || index >= input_rank) {
    return op.emitOpError("split dimension must be in range [-")
           << input_rank << ", " << input_rank << ")";
  }

  if (index < 0) index += input_rank;
  *dim_index = index;

  return success();
}

LogicalResult SplitOp::verify() {
  SplitOp op = *this;
  std::optional<int64_t> dim_index;
  if (failed(VerifySplitInputAndSplitDim(op, &dim_index))) return failure();
  if (!dim_index) return success();

  int64_t input_dim_size =
      op.getValue().getType().cast<RankedTensorType>().getDimSize(*dim_index);
  if (ShapedType::isDynamic(input_dim_size)) return success();

  if (op.getNumResults() == 0) return failure();

  if (input_dim_size % op.getNumResults() != 0)
    return op.emitOpError("dimension #")
           << *dim_index << " not divisible by the number of result tensors";

  return success();
}

//===----------------------------------------------------------------------===//
// SplitVOp
//===----------------------------------------------------------------------===//

LogicalResult SplitVOp::verify() {
  SplitVOp op = *this;
  auto split_sizes_type =
      op.getSizeSplits().getType().dyn_cast<RankedTensorType>();
  if (!split_sizes_type) return success();

  if (split_sizes_type.getRank() != 1 ||
      (!ShapedType::isDynamic(split_sizes_type.getDimSize(0)) &&
       split_sizes_type.getDimSize(0) != op.getNumResults()))
    return op.emitOpError("split sizes should be a 1D tensor of ")
           << op.getNumResults() << " elements";

  std::optional<int64_t> dim_index = 0;
  if (failed(VerifySplitInputAndSplitDim(op, &dim_index))) return failure();
  if (!dim_index) return success();

  int64_t input_dim_size =
      op.getValue().getType().cast<RankedTensorType>().getDimSize(*dim_index);
  if (ShapedType::isDynamic(input_dim_size)) return success();

  // If split sizes come from a constant, they must sum to the dimension size
  // along split_dim, and we can have no more than one dynamic dimension.
  DenseIntElementsAttr split_sizes_attr;
  if (!matchPattern(op.getSizeSplits(), m_Constant(&split_sizes_attr)))
    return success();

  int64_t total_dim_size = 0;  // Total dimension size assigned to splits
  std::optional<int64_t> dynamic_dim_index;

  SmallVector<int64_t, 4> split_sizes;
  split_sizes.reserve(
      split_sizes_attr.getType().cast<ShapedType>().getNumElements());

  for (const auto &dim : llvm::enumerate(split_sizes_attr)) {
    int64_t dim_val = dim.value().getSExtValue();
    split_sizes.push_back(dim_val);
    if (dim_val == tensorflow::kTFDynamicSize) {
      // We cannot have more than one dynamic dimension.
      if (dynamic_dim_index)
        return op.emitOpError(
            "cannot have more than one dynamic dimension in split sizes");
      dynamic_dim_index = dim.index();
    } else {
      total_dim_size += dim_val;
    }
  }

  if (!dynamic_dim_index && total_dim_size != input_dim_size)
    return op.emitOpError(
               "split sizes must sum up to the dimension size along split "
               "dimension, found ")
           << total_dim_size << " vs " << input_dim_size;

  if (dynamic_dim_index && total_dim_size > input_dim_size)
    return op.emitOpError(
               "split sizes must sum up to be less than or equal to the "
               "dimension size along split dimension, found ")
           << total_dim_size << " vs " << input_dim_size;

  return success();
}

//===----------------------------------------------------------------------===//
// SquareOp
//===----------------------------------------------------------------------===//

void SquareOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<SquareOfSub>(context);
}

//===----------------------------------------------------------------------===//
// SqueezeOp
//===----------------------------------------------------------------------===//

LogicalResult SqueezeOp::verify() {
  SqueezeOp op = *this;
  auto input_type = op.getInput().getType().dyn_cast<RankedTensorType>();

  if (!input_type) return success();  // Can't verify squeeze dims.

  int64_t input_rank = input_type.getRank();
  for (const auto &squeeze_dim_apint :
       op.getSqueezeDims().getAsValueRange<IntegerAttr>()) {
    int64_t squeeze_dim = squeeze_dim_apint.getSExtValue();
    if (squeeze_dim < -input_rank || squeeze_dim >= input_rank) {
      return op.emitOpError()
             << "squeeze dimension " << squeeze_dim << " not in ["
             << -input_rank << ", " << input_rank << ")";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SubOfNeg>(context);
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return IdentityArithmeticOpFolder<SubOp>(*this, operands);
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

void SumOp::build(OpBuilder &builder, OperationState &result, Value input,
                  Value reduction_indices, BoolAttr keep_dims) {
  Type out_ty = InferReductionOpType(input, reduction_indices, keep_dims);
  build(builder, result, out_ty, input, reduction_indices, keep_dims);
}

// TODO: Templatize this fold for all reduction ops.
OpFoldResult SumOp::fold(FoldAdaptor) {
  auto input_ty = getInput().getType().template dyn_cast<RankedTensorType>();
  if (!input_ty) return {};
  auto result_ty = getType().template dyn_cast<RankedTensorType>();
  if (!result_ty) return {};

  // Bypass this op if the result has the same shape and type. This can happen
  // if the input tensor has size 0 or size 1.
  if (!getKeepDims() && input_ty == result_ty) {
    return getInput();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// StridedSliceOp
//===----------------------------------------------------------------------===//

// TODO(b/154160827): Add a canonicalization pattern from tf.StridedSliceOp to
// tf.SliceOp if both of the following are true:
// - All strides have a known value equal to 1
// - No masks are set (or masks can be applied by transforming the inputs to
//   Slice)

// Verifies that,
//
// - begin, end and strides operands are 1D and they have the same number of
//   elements. Here, the number of elements should be less than 32 to support
//   32-bit mask attributes.
// - None of the strides values are zero.
// - Ellipsis mask can have at most one bit set.

template <class OpTy>
static LogicalResult VerifyStridedSliceBase(OpTy op) {
  // Expected size for operands begin, end and strides vector operands.
  int64_t expected_size = -1;

  for (Value val : {op.getBegin(), op.getEnd(), op.getStrides()}) {
    auto operand_ty = val.getType().dyn_cast<ShapedType>();
    if (!operand_ty || !operand_ty.hasStaticShape()) {
      // TensorFlow constant ops may have non-static shape because the shape is
      // not propagated during constant folding. If the defining op for this
      // operand is a constant op, use the constant op's attribute to get the
      // actual shape.
      DenseIntElementsAttr attr;
      if (!matchPattern(val, m_Constant(&attr))) continue;
      operand_ty = attr.getType();
    }

    if (operand_ty.getRank() != 1)
      return op.emitOpError()
             << "requires begin, end and strides to be 1D tensors";

    int64_t length = operand_ty.getDimSize(0);
    if (length == -1) continue;

    if (expected_size == -1) {
      // This op uses 32-bit masks.
      if (length >= 32)
        return op.emitOpError(
            "requires begin, end and strides operands with less than 32 "
            "elements");

      expected_size = length;
    } else if (length != expected_size) {
      return op.emitOpError() << "requires begin, end and strides to have the "
                                 "same number of elements";
    }
  }

  // If strides are constants, verify that none of the element is zero.
  DenseIntElementsAttr strides;
  if (matchPattern(op.getStrides(), m_Constant(&strides))) {
    if (llvm::is_contained(strides.getValues<APInt>(), 0))
      return op.emitOpError("requires non-zero strides");
  }

  // Use bit compares to ensure ellipsis_mask is 0 or a power of 2, i.e. there
  // exists only no more than one ellipsis.
  uint32_t ellipsis_mask = op.getEllipsisMask();
  if (ellipsis_mask != 0 && !llvm::isPowerOf2_32(ellipsis_mask))
    return op.emitOpError("cannot have multiple ellipses");

  return success();
}

LogicalResult StridedSliceOp::verify() { return VerifyStridedSliceBase(*this); }

// Clamps the given `val`: returns `low` if `val` is less than `low`; returns
// `high` if `high` is less than `val`; otherwise returns `val`.
template <class T>
constexpr const T &Clamp(const T &val, const T &low, const T &high) {
  assert(!(high < low));
  return (val < low) ? low : (high < val) ? high : val;
}

// Checks if the `index` bit of `val` is set.
template <class T>
constexpr bool IsSet(const T &val, unsigned index) {
  return (val & (1 << index)) != 0;
}

// Sets the `index` bit of `val`.
template <class T>
constexpr void Set(T &val, unsigned index) {
  val |= (1 << index);
}

// Unset the `index` bit of `val`.
template <class T>
constexpr void Unset(T &val, unsigned index) {
  val &= ~(1 << index);
}

// Copy the `src_index` bit of `src` to `dst_index` bit of `dst`.
template <class T>
constexpr void CopyBit(const T &src, unsigned src_index, T &dst,
                       unsigned dst_index) {
  if (IsSet(src, src_index))
    Set(dst, dst_index);
  else
    Unset(dst, dst_index);
}

// The sparse spec of strided slice does not correspond to the number of
// dimensions. For example, sparse spec for foo[..., 3:10] for foo of shape (2,
// 4, 8) would have dims = 2.
struct SparseSliceSpec {
  int64_t dims;
  int32_t begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask;
  const ArrayRef<int64_t> &begin;
  const ArrayRef<int64_t> &end;
  const ArrayRef<int64_t> &strides;
};

// The dense spec of strided slice is the canonicalized version of sparse spec.
// The number of dimensions of dense spec correspond to the number of dimensions
// in operand tensor.
struct DenseSliceSpec {
  int64_t dims;
  int32_t begin_mask, end_mask, shrink_axis_mask;
  SmallVectorImpl<int64_t> &begin;
  SmallVectorImpl<int64_t> &end;
  SmallVectorImpl<int64_t> &strides;
};

// Make a sparse spec into a dense index spec.
// The sparse spec does not correspond to the number of dimensions
// Make a dense spec that corresponds to the number of dimensions
//
// For example suppose foo[...,3:, 2] on foo.shape=(2,2,3,4) then
// we need to produce the missing begin_mask, end_mask for the first two
// dimensions i.e. foo[:, :, 3:, 2].
static void BuildDenseSliceSpec(const SparseSliceSpec &sparse,
                                DenseSliceSpec *dense) {
  // Build expanded dense begin, end, strides, begin_mask, end_mask, and
  // shrink_axis_mask.
  dense->begin.resize(dense->dims);
  dense->end.resize(dense->dims);
  dense->strides.resize(dense->dims);
  dense->begin_mask = 0;
  dense->end_mask = 0;
  dense->shrink_axis_mask = 0;

  // Count number of new_axis after ellipsis. This helps in calculating the
  // number of dimensions ellipsis represents in the sparse spec.
  bool ellipsis_seen = false;
  int num_new_axis_after_ellipsis = 0;
  for (int sparse_index = 0; sparse_index < sparse.dims; ++sparse_index) {
    if (ellipsis_seen && IsSet(sparse.new_axis_mask, sparse_index))
      num_new_axis_after_ellipsis++;
    if (IsSet(sparse.ellipsis_mask, sparse_index)) ellipsis_seen = true;
  }

  int dense_index = 0;
  for (int sparse_index = 0; sparse_index < sparse.dims; ++sparse_index) {
    if (IsSet(sparse.new_axis_mask, sparse_index)) continue;
    if (IsSet(sparse.ellipsis_mask, sparse_index)) {
      auto next_index = std::min(dense->dims - (sparse.dims - sparse_index) +
                                     1 + num_new_axis_after_ellipsis,
                                 dense->dims);
      // Expand ellipsis into the appropriate dense indices. From current index
      // until next_index, all dimensions would have begin and end masks set and
      // stride 1, i.e., get all elements in those dimensions.
      for (; dense_index < next_index; ++dense_index) {
        dense->begin[dense_index] = dense->end[dense_index] = 0;
        dense->strides[dense_index] = 1;
        Set(dense->begin_mask, dense_index);
        Set(dense->end_mask, dense_index);
      }
      continue;
    }
    assert(dense_index < dense->dims);
    // Copy over the sparse indices to dense indices if ellipsis_mask and
    // new_axis_mask are not set.
    dense->begin[dense_index] = sparse.begin[sparse_index];
    dense->end[dense_index] = sparse.end[sparse_index];
    dense->strides[dense_index] = sparse.strides[sparse_index];
    CopyBit(sparse.begin_mask, sparse_index, dense->begin_mask, dense_index);
    CopyBit(sparse.end_mask, sparse_index, dense->end_mask, dense_index);
    CopyBit(sparse.shrink_axis_mask, sparse_index, dense->shrink_axis_mask,
            dense_index);
    dense_index++;
  }
}

// For the given `input_shape`, calculates the sliced shape using the given
// `begin`, `end`, and `stride` ranges and `begin_mask`, `end_mask`, and
// `shrink_axis_mask` masks. Updates the result back to `input_shape`. If
// `shrink_axis_mask` is not zero, this function will not drop the corresponding
// dimensions in `input_shape`; it will turn them into 1s. At the same time,
// canonicalizes `begin`, `end`, and `strides. The calculation follows
// tf.StridedSlice op semantics.
static void CalculateSlicedShapeFromDenseIndices(
    MutableArrayRef<int64_t> input_shape, int32_t begin_mask, int32_t end_mask,
    int32_t shrink_axis_mask, MutableArrayRef<int64_t> begin,
    MutableArrayRef<int64_t> end, MutableArrayRef<int64_t> stride) {
  assert(input_shape.size() <= 32);  // Only 32-bit masks are supported.

  // Make sure ranges' ranks are consistent with the input.
  assert(input_shape.size() == begin.size());
  assert(input_shape.size() == end.size());
  assert(input_shape.size() == stride.size());

  for (int i = 0, e = input_shape.size(); i < e; ++i) {
    if (ShapedType::isDynamic(input_shape[i])) continue;

    int64_t dim_i = input_shape[i];
    int64_t begin_i = begin[i];
    int64_t end_i = end[i];
    int64_t stride_i = stride[i];

    // [0]: mask for begin, [1]: mask for end
    int64_t masks[] = {begin_mask & (1 << i), end_mask & (1 << i)};
    // [0]: bound for begin, [1]: bound for end
    int64_t bounds[] = {stride_i > 0 ? 0 : -1,
                        stride_i > 0 ? dim_i : dim_i - 1};

    // Canonicalizes the given range `point` (begin/end) according to the
    // current dimension. `c` means case: 0 for begin, 1 for end.
    auto canonicalize = [&](int64_t point, int c) {
      if (masks[c]) return stride_i > 0 ? bounds[c] : bounds[(c + 1) & 1];

      // Add dim as offset to negative range point.
      point = point < 0 ? dim_i + point : point;
      return Clamp(point, bounds[0], bounds[1]);
    };

    begin_i = canonicalize(begin_i, 0);
    end_i = canonicalize(end_i, 1);

    int64_t interval_len = end_i - begin_i;
    int64_t size_i = 0;
    // If internal length is zero or has different sign from stride, it's a
    // degenerated case: we are slicing nothing. Otherwise, calculate the sliced
    // size.
    if (interval_len != 0 && (interval_len < 0) == (stride_i < 0))
      size_i = (interval_len / stride_i) + (interval_len % stride_i != 0);

    begin[i] = begin_i;
    if (IsSet(shrink_axis_mask, i)) {
      // Shrink this dimension. It means we only take the element at begin_i.
      input_shape[i] = 1;
      end[i] = begin_i + 1;
      stride[i] = 1;
    } else {
      input_shape[i] = size_i;
      end[i] = end_i;
      stride[i] = stride_i;
    }
  }
}

// For the given `input_shape`, calculates the sliced shape using the given
// `sparse_begin`, `sparse_end`, and `sparse_strides` ranges and `begin_mask`,
// `end_mask`, `ellipsis_mask` , `new_axis_mask` and `shrink_axis_mask` masks.
// Updates the result back to `input_shape`.
static void CalculateSlicedShapeFromSparseIndices(
    MutableArrayRef<int64_t> input_shape, ArrayRef<int64_t> sparse_begin,
    ArrayRef<int64_t> sparse_end, ArrayRef<int64_t> sparse_strides,
    int32_t begin_mask, int32_t end_mask, int32_t ellipsis_mask,
    int32_t new_axis_mask, int32_t shrink_axis_mask,
    SmallVectorImpl<int64_t> *begin, SmallVectorImpl<int64_t> *end,
    SmallVectorImpl<int64_t> *stride) {
  int64_t num_sparse_indices = sparse_begin.size();
  SparseSliceSpec sparse = {num_sparse_indices, begin_mask,    end_mask,
                            ellipsis_mask,      new_axis_mask, shrink_axis_mask,
                            sparse_begin,       sparse_end,    sparse_strides};

  // If no ellipsis_mask exists then an implicit ellipsis_mask at the end is
  // inserted. This handles cases where foo[2:4] (foo.shape() = [4, 8]) yields
  // a tensor of shape [2, 8], i.e., foo[2:4] is same as foo[2:4, ...].
  if (sparse.ellipsis_mask == 0) {
    Set(sparse.ellipsis_mask, sparse.dims);
    sparse.dims++;
  }

  int64_t dims = input_shape.size();
  DenseSliceSpec dense = {dims,
                          /*begin_mask = */ 0,
                          /*end_mask = */ 0,
                          /*shrink_axis_mask = */ 0,
                          *begin,
                          *end,
                          *stride};

  BuildDenseSliceSpec(sparse, &dense);
  CalculateSlicedShapeFromDenseIndices(input_shape, dense.begin_mask,
                                       dense.end_mask, dense.shrink_axis_mask,
                                       *begin, *end, *stride);
}

bool StridedSliceOp::GetSlicedBoundRanges(
    SmallVectorImpl<int64_t> *slice_begin, SmallVectorImpl<int64_t> *slice_end,
    SmallVectorImpl<int64_t> *slice_stride) {
  // TODO(hinsu): Support lowering for ops with dynamic begin and end values
  // when it is possible to derive indices based on mask attributes.
  DenseIntElementsAttr sparse_begin_attr, sparse_end_attr, sparse_strides_attr;
  if (!matchPattern(getBegin(), m_Constant(&sparse_begin_attr)) ||
      !matchPattern(getEnd(), m_Constant(&sparse_end_attr)) ||
      !matchPattern(getStrides(), m_Constant(&sparse_strides_attr)))
    return false;

  auto input_ty = this->getInput().getType().dyn_cast<RankedTensorType>();
  if (!input_ty || !input_ty.hasStaticShape()) return false;
  auto input_shape = llvm::to_vector<4>(input_ty.getShape());

  SmallVector<int64_t, 4> sparse_begin, sparse_end, sparse_strides;

  for (const APInt &index : sparse_begin_attr)
    sparse_begin.push_back(index.getSExtValue());
  for (const APInt &index : sparse_end_attr)
    sparse_end.push_back(index.getSExtValue());
  for (const APInt &stride : sparse_strides_attr)
    sparse_strides.push_back(stride.getSExtValue());

  CalculateSlicedShapeFromSparseIndices(
      input_shape, sparse_begin, sparse_end, sparse_strides, getBeginMask(),
      getEndMask(), getEllipsisMask(), getNewAxisMask(), getShrinkAxisMask(),
      slice_begin, slice_end, slice_stride);
  return true;
}

OpFoldResult StridedSliceOp::fold(FoldAdaptor) {
  // Fold StridedSlice operation if it extracts statically known dimensions.
  //
  // For example,
  //
  //   %shape  = tf.Shape(%arg)                   // %arg: tensor<?x2x3x1xf32>
  //   %height = tf.StridedSlice(%shape, 1, 2, 1)
  //
  // In this case %height can be replaced with a constant 2.
  //
  // Or,
  //
  //   %shape  = tf.Shape(%arg)                   // %arg: tensor<?x2x3x1xf32>
  //   %spatial_shape = tf.StridedSlice(%shape, 1, 3, 1)
  //
  // In this case %spatial_shape can be replaced with a constant [2, 3].

  // Input to strided slice op is defined by shape operation.
  auto shape_op = getInput().getDefiningOp<ShapeOp>();
  if (!shape_op) {
    return {};
  }

  // `begin`, `end` and `strides` should be constant in order to infer static
  // dimension.
  DenseIntElementsAttr begin_attr, end_attr, strides_attr;
  if (!matchPattern(getBegin(), m_Constant(&begin_attr)) ||
      !matchPattern(getEnd(), m_Constant(&end_attr)) ||
      !matchPattern(getStrides(), m_Constant(&strides_attr)) ||
      begin_attr.getNumElements() != 1 || end_attr.getNumElements() != 1 ||
      strides_attr.getNumElements() != 1) {
    return {};
  }

  // Do not fold when `new_axis_mask` is set. It's likely to break the shape
  // of output. Typically, `new_axis_mask` is not set in this canonicalization
  // pattern.
  if (getNewAxisMask() != 0) return {};

  auto tensor_ty = shape_op.getInput().getType().dyn_cast<RankedTensorType>();
  // Only ranked tensor can be folded.
  if (!tensor_ty) return {};

  int64_t rank = tensor_ty.getRank();
  int64_t begin_int = begin_attr.getValues<APInt>()[0].getSExtValue();
  int64_t end_int = end_attr.getValues<APInt>()[0].getSExtValue();
  int64_t strides_int = strides_attr.getValues<APInt>()[0].getSExtValue();

  // Canonicalize `begin` and `end` in case of negative index.
  if (begin_int < 0) begin_int += rank;
  if (end_int < 0) end_int += rank;

  // Create `begin` and `end` from `*_mask`. Note that we don't care about
  // `new_axis_mask` as it can be inferred from `output_ty`.
  if (getShrinkAxisMask() == 1) {
    // When `shrink_axis_mask` is set, output is always a scalar so only
    // one element is sliced.
    end_int = begin_int + 1;
  }
  if (getBeginMask() == 1) {
    begin_int = (strides_int > 0) ? 0 : rank - 1;
  }
  if (getEndMask() == 1) {
    end_int = (strides_int > 0) ? rank : -1;
  }
  if (getEllipsisMask() == 1) {
    begin_int = 0;
    end_int = rank;
  }

  // It's possible that `begin` and `end` are out of bound. See
  // https://docs.python.org/3/library/stdtypes.html#common-sequence-operations.
  if (strides_int > 0) {
    begin_int = std::min(begin_int, rank);
    end_int = std::min(end_int, rank);
  } else {
    begin_int = std::min(begin_int, rank - 1);
    end_int = std::min(end_int, rank - 1);
  }

  SmallVector<int64_t, 2> sub_shape;
  // Only handle cases that have something to slice to avoid infinite for-loop.
  if ((end_int > begin_int && strides_int > 0) ||
      (end_int < begin_int && strides_int < 0)) {
    // Extract sub-shape only if all of those dimensions are static.
    for (int64_t i = begin_int; (strides_int > 0) ? i < end_int : i > end_int;
         i += strides_int) {
      if (tensor_ty.isDynamicDim(i)) {
        return {};
      }
      sub_shape.push_back(tensor_ty.getDimSize(i));
    }
  }

  // For unranked or dynamic output, we infer the output type to either a
  // scalar or a vector based on `shrink_axis_mask` because we have rejected
  // the case of `new_axis_mask` != 0.
  auto output_elt_ty =
      getOutput().getType().cast<ShapedType>().getElementType();
  auto output_ty = getOutput().getType().dyn_cast<RankedTensorType>();
  if (!output_ty || !output_ty.hasStaticShape()) {
    if (getShrinkAxisMask() == 1) {
      output_ty = tensorflow::GetTypeFromTFTensorShape({}, output_elt_ty);
    } else {
      output_ty = tensorflow::GetTypeFromTFTensorShape(
          {static_cast<int64_t>(sub_shape.size())}, output_elt_ty);
    }
  }

  // Down-cast to 32 bit int if needed.
  if (output_elt_ty.isInteger(32)) {
    SmallVector<int32_t, 2> sub_shape_i32(sub_shape.size());
    std::transform(sub_shape.begin(), sub_shape.end(), sub_shape_i32.begin(),
                   [](int64_t d) { return static_cast<int32_t>(d); });
    return DenseIntElementsAttr::get(output_ty, sub_shape_i32);
  }
  return DenseIntElementsAttr::get(output_ty, sub_shape);
}

//===----------------------------------------------------------------------===//
// StridedSliceGradOp
//===----------------------------------------------------------------------===//

LogicalResult StridedSliceGradOp::verify() {
  StridedSliceGradOp op = *this;
  auto shape_type = op.getShape().getType().dyn_cast<RankedTensorType>();
  if (shape_type && shape_type.getRank() != 1)
    return op.emitOpError("'shape' operand must be 1D tensor, but got ")
           << shape_type.getRank() << "D tensor";

  if (failed(VerifyStridedSliceBase(op))) return failure();

  // TODO(antiagainst): verify the gradient op.dy()'s shape is consistent with
  // the sliced type from StridedSlice.

  return success();
}

bool StridedSliceGradOp::GetSlicedShapeAndBoundRanges(
    SmallVectorImpl<int64_t> *input_shape,
    SmallVectorImpl<int64_t> *slice_begin, SmallVectorImpl<int64_t> *slice_end,
    SmallVectorImpl<int64_t> *slice_stride) {
  DenseIntElementsAttr shape_attr;
  DenseIntElementsAttr sparse_begin_attr, sparse_end_attr, sparse_strides_attr;
  if (!matchPattern(getShape(), m_Constant(&shape_attr)) ||
      !matchPattern(getBegin(), m_Constant(&sparse_begin_attr)) ||
      !matchPattern(getEnd(), m_Constant(&sparse_end_attr)) ||
      !matchPattern(getStrides(), m_Constant(&sparse_strides_attr)))
    return false;

  int rank = std::distance(shape_attr.begin(), shape_attr.end());

  input_shape->clear();
  input_shape->reserve(rank);
  for (const APInt &dim : shape_attr)
    input_shape->push_back(dim.getSExtValue());

  SmallVector<int64_t, 4> sparse_begin, sparse_end, sparse_strides;

  for (const APInt &index : sparse_begin_attr)
    sparse_begin.push_back(index.getSExtValue());
  for (const APInt &index : sparse_end_attr)
    sparse_end.push_back(index.getSExtValue());
  for (const APInt &stride : sparse_strides_attr)
    sparse_strides.push_back(stride.getSExtValue());

  CalculateSlicedShapeFromSparseIndices(
      *input_shape, sparse_begin, sparse_end, sparse_strides, getBeginMask(),
      getEndMask(), getEllipsisMask(), getNewAxisMask(), getShrinkAxisMask(),
      slice_begin, slice_end, slice_stride);
  return true;
}

//===----------------------------------------------------------------------===//
// SummaryWriterOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<ResourceHandleValueAndId, 4>
SummaryWriterOp::GetResourceHandleValueAndIdList(
    llvm::SmallDenseMap<ResourceHandle, int64_t> &resource_handle_id_map,
    int64_t &next_id) {
  llvm::StringRef device = GetDeviceOrEmpty(getOperation());
  return {GetResourceHandleValueAndIdBase(getContainer(), getSharedName(),
                                          device, getWriter(),
                                          resource_handle_id_map, next_id)};
}

//===----------------------------------------------------------------------===//
// TPUExecuteOp
//===----------------------------------------------------------------------===//

void TPUExecuteOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.reserve(getArgs().size() + 1);
  effects.emplace_back(MemoryEffects::Write::get(),
                       ResourceEffects::TPUExecute::get());

  for (Value value : getArgs()) {
    if (value.getType()
            .cast<TensorType>()
            .getElementType()
            .isa<ResourceType>()) {
      // Conservatively mark resource handles as read and write, as without
      // analyzing TPUCompile, there is not sufficient information to determine
      // effects on resources. For the MLIR bridge, this op will never be
      // populated with resource handles and tf.TPUExecuteAndUpdateVariables is
      // used instead.
      effects.emplace_back(MemoryEffects::Read::get(), value,
                           ResourceEffects::Variable::get());
      effects.emplace_back(MemoryEffects::Write::get(), value,
                           ResourceEffects::Variable::get());
    }
  }
}

//===----------------------------------------------------------------------===//
// TPUExecuteAndUpdateVariablesOp
//===----------------------------------------------------------------------===//

LogicalResult TPUExecuteAndUpdateVariablesOp::verify() {
  TPUExecuteAndUpdateVariablesOp op = *this;
  int num_resource_args = 0;
  for (Type arg_type : op.getArgs().getTypes())
    if (arg_type.cast<TensorType>().getElementType().isa<ResourceType>())
      ++num_resource_args;

  auto check_attr = [&](ArrayAttr indices, llvm::StringRef name,
                        int min) -> LogicalResult {
    if (indices.size() != num_resource_args)
      return op.emitOpError()
             << "requires '" << name
             << "' to be the same size as number of resource handles in 'args' "
                "("
             << num_resource_args << "), but got " << indices.size();

    for (const auto &entry : llvm::enumerate(indices.getValue())) {
      auto int_attr = entry.value().cast<IntegerAttr>();
      if (int_attr.getInt() < min)
        return op.emitOpError()
               << "requires '" << name << "' to contain values of at least "
               << min << ", but got " << int_attr.getInt() << " at index "
               << entry.index();
    }

    return success();
  };

  return failure(
      failed(check_attr(op.getDeviceVarReadsIndices(),
                        /*name=*/"device_var_reads_indices", /*min=*/0)) ||
      failed(check_attr(op.getDeviceVarUpdatesIndices(),
                        /*name=*/"device_var_updates_indices", /*min=*/-1)));
}

void TPUExecuteAndUpdateVariablesOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.reserve(getDeviceVarReadsIndices().size() + 1);
  effects.emplace_back(MemoryEffects::Write::get(),
                       ResourceEffects::TPUExecute::get());
  auto resource_handles = llvm::make_filter_range(getArgs(), [](Value value) {
    return value.getType()
        .cast<TensorType>()
        .getElementType()
        .isa<ResourceType>();
  });

  for (auto &entry : llvm::enumerate(resource_handles)) {
    Value value = entry.value();
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         ResourceEffects::Variable::get());
    if (getDeviceVarUpdatesIndices()
            .getValue()[entry.index()]
            .cast<IntegerAttr>()
            .getInt() >= 0)
      effects.emplace_back(MemoryEffects::Write::get(), value,
                           ResourceEffects::Variable::get());
  }
}

//===----------------------------------------------------------------------===//
// TensorListGetItemOp
//===----------------------------------------------------------------------===//

namespace {
// If the input of TensorListGetItemOp is TensorListFromTensorOp and the
// TensorListFromTensorOp is only used by TensorListGetItemOp (not modified by
// other TensorList ops), we can convert it to a GatherOp.
class ConvertTensorListGetItemOpOfTensorListFromTensorOpToGather
    : public OpRewritePattern<TensorListGetItemOp> {
  using OpRewritePattern<TensorListGetItemOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorListGetItemOp op,
                                PatternRewriter &rewriter) const override {
    // Checks that the input is created by TensorListFromTensorOp and the input
    // is only used by TensorListGetItemOp.
    auto tensor_list_from_tensor_op = dyn_cast_or_null<TensorListFromTensorOp>(
        op.getInputHandle().getDefiningOp());
    if (!tensor_list_from_tensor_op ||
        llvm::any_of(
            tensor_list_from_tensor_op->getUsers(),
            [](Operation *user) { return !isa<TensorListGetItemOp>(user); })) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<GatherOp>(
        op, op.getType(), tensor_list_from_tensor_op.getTensor(),
        op.getIndex());
    return success();
  }
};
}  // namespace

void TensorListGetItemOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<ConvertTensorListGetItemOpOfTensorListFromTensorOpToGather>(
      context);
}

//===----------------------------------------------------------------------===//
// TensorListReserveOp
//===----------------------------------------------------------------------===//

LogicalResult TensorListReserveOp::verify() {
  TensorListReserveOp op = *this;
  // This is required to populate derived attributes during export in a
  // meaningful way. Else during export to GraphDef element_type() query
  // will result in out of bounds access/assert.
  if (handle_dtype().getSubtypes().size() != 1) {
    return emitOpError(
        "must have exactly one subtype in the result variant type");
  }
  if (!IsOfRankOrUnranked(op.getElementShape(), 0) &&
      !IsOfRankOrUnranked(op.getElementShape(), 1)) {
    return op.emitOpError("requires element_shape operand to be 0D/1D tensor");
  }

  if (!IsOfRankOrUnranked(op.getNumElements(), 0)) {
    return op.emitOpError("requires num_elements operand to be 0D tensor");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TensorListElementShapeOp
//===----------------------------------------------------------------------===//

OpFoldResult TensorListElementShapeOp::fold(FoldAdaptor) {
  int width =
      getType().cast<ShapedType>().getElementType().getIntOrFloatBitWidth();
  auto variant_type =
      getElementTypeOrSelf(getOperand().getType()).cast<TF::VariantType>();
  if (variant_type.getSubtypes().empty()) return {};
  return ConvertShapeToAttr(variant_type.getSubtypes()[0], width);
}

//===----------------------------------------------------------------------===//
// TensorListStackOp
//===----------------------------------------------------------------------===//

LogicalResult TensorListStackOp::verify() {
  TensorListStackOp op = *this;
  if (!IsOfRankOrUnranked(op.getElementShape(), 0) &&
      !IsOfRankOrUnranked(op.getElementShape(), 1)) {
    return op.emitOpError("requires element_shape operand to be 0D/1D tensor");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TensorScatterUpdateOp
//===----------------------------------------------------------------------===//

LogicalResult TensorScatterUpdateOp::verify() {
  TensorScatterUpdateOp op = *this;
  if (!HasRankAtLeast(op.getTensor(), 1))
    return op.emitOpError(
        "requires tensor operand to have at least 1 dimension");
  if (!HasRankAtLeast(op.getIndices(), 1))
    return op.emitOpError(
        "requires indices operand to have at least 1 dimension");

  auto tensor_ty = op.getTensor().getType().dyn_cast<RankedTensorType>();
  auto indices_ty = op.getIndices().getType().dyn_cast<RankedTensorType>();
  if (!tensor_ty || !indices_ty) return success();

  int64_t num_index_dims = indices_ty.getShape().back();
  if (ShapedType::isDynamic(num_index_dims)) return success();

  if (num_index_dims > tensor_ty.getRank())
    return op.emitOpError(
        "requires tensor operand with rank greater than or equal to the "
        "indices operand's last dimensions");
  return success();
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

// Verifies that,
//
// - input has at least rank 1
// - multiples is rank 1
// - multiples.size() == input.rank()
// - input.rank() == output.rank()
// - Elements in multiples are non-negative
// - input.shape[i] * multiples[i] == output.shape[i]
//   for i in [0, input.rank() - 1]

LogicalResult TileOp::verify() {
  TileOp op = *this;
  auto input_type = op.getInput().getType().dyn_cast<RankedTensorType>();
  auto multiples_type =
      op.getMultiples().getType().dyn_cast<RankedTensorType>();
  auto output_type = op.getOutput().getType().dyn_cast<RankedTensorType>();

  if (multiples_type && multiples_type.getRank() != 1) {
    return op.emitOpError() << "expected multiples to be rank 1, got rank = "
                            << multiples_type.getRank();
  }

  if (input_type && multiples_type && multiples_type.hasStaticShape() &&
      (input_type.getRank() != multiples_type.getNumElements() ||
       (input_type.getRank() == 0 && multiples_type.getNumElements() == 1))) {
    return op.emitOpError()
           << "expected size of multiples equal to rank of input"
           << ", got multiples of size " << multiples_type.getNumElements()
           << ", and input of rank " << input_type.getRank();
  }

  if (input_type && output_type) {
    if (input_type.getRank() != output_type.getRank()) {
      return op.emitOpError()
             << "expected rank of input to equal to rank of output"
             << ", got input of rank " << input_type.getRank()
             << ", and output of rank " << output_type.getRank();
    }

    DenseIntElementsAttr multiples_attr;
    if (matchPattern(op.getMultiples(), m_Constant(&multiples_attr))) {
      for (int32_t i = 0, e = input_type.getRank(); i < e; ++i) {
        const int64_t input_dim = input_type.getDimSize(i);
        const int64_t output_dim = output_type.getDimSize(i);
        const int64_t m = multiples_attr.getValues<APInt>()[i].getSExtValue();

        if (m < 0) {
          return op.emitOpError()
                 << "expected multiples to be non-negative, got "
                 << "multiples[" << i << "] = " << m;
        }

        if (!ShapedType::isDynamic(input_dim) &&
            !ShapedType::isDynamic(output_dim) && output_dim != input_dim * m) {
          return op.emitOpError()
                 << "requires input.shape[" << i << "] (" << input_dim << ")"
                 << " * " << m << " to be equal to "
                 << "output.shape[" << i << "] (" << output_dim << ")";
        }
      }
    }
  }

  return success();
}

OpFoldResult TileOp::fold(FoldAdaptor) {
  DenseIntElementsAttr multiples_attr;
  if (matchPattern(getMultiples(), m_Constant(&multiples_attr))) {
    // Return input directly when multiples are all ones,
    // regardless what input is.
    if (multiples_attr.isSplat() &&
        multiples_attr.getSplatValue<APInt>().getSExtValue() == 1) {
      return getInput();
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// TopKV2Op
//===----------------------------------------------------------------------===//

LogicalResult TopKV2Op::verify() {
  TopKV2Op op = *this;
  if (!HasRankAtLeast(op.getInput(), 1))
    return op.emitOpError(
        "requires input operand to have at least 1 dimension");

  if (!IsOfRankOrUnranked(op.getK(), 0))
    return op.emitOpError("requires k operand to be 0D tensor");

  return success();
}

//===----------------------------------------------------------------------===//
// ToBoolOp
//===----------------------------------------------------------------------===//

namespace {
// If the input to ToBoolOp is a ranked tensor, then the ToBoolOp can be folded
// into an identity or an equality comparison.
class ToBoolOfRankedTensor : public OpRewritePattern<ToBoolOp> {
  using OpRewritePattern<ToBoolOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ToBoolOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getOperand().getType().dyn_cast<RankedTensorType>();
    // If the input is an unranked tensor, cannpt rewrite.
    if (!type) return failure();

    // Expected return type of the ToBool operation. The return type of ToBool
    // operation is always 0D tensor of bool type.
    auto result_type = op.getResult().getType().cast<RankedTensorType>();

    // If input is already a tensor<i1>, it can be folded into an identity.
    if (type == result_type) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    if (type.getRank() == 0) {
      // If the input is a scalar tensor, the ToBool can be expanded to
      // element != 0 (for numerical values) or element == empty (for string).
      Type element_type = type.getElementType();
      Attribute zero_attr;
      if (element_type.isIntOrFloat())
        zero_attr = rewriter.getZeroAttr(type);
      else if (element_type.isa<TF::StringType>())
        zero_attr = DenseStringElementsAttr::get(type, {""});

      if (!zero_attr) return failure();

      auto zero_const = rewriter.create<TF::ConstOp>(op.getLoc(), zero_attr);
      rewriter.replaceOpWithNewOp<TF::NotEqualOp>(
          op, result_type, op.getOperand(), zero_const, false);
    } else {
      // If the input is a non-scalar ranked tensor, ToBool can be expanded
      // to numElements != 0. numElements will be 0 iff one of the dimensions is
      // zero.
      bool any_zero =
          llvm::any_of(type.getShape(), [](int64_t dim) { return dim == 0; });
      rewriter.replaceOpWithNewOp<TF::ConstOp>(
          op, result_type, DenseElementsAttr::get(result_type, {!any_zero}));
    }
    return success();
  }
};
}  // namespace

void ToBoolOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<ToBoolOfRankedTensor>(context);
}

LogicalResult ToBoolOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(
      tensorflow::GetTypeFromTFTensorShape({}, IntegerType::get(context, 1)));
  return success();
}

//===----------------------------------------------------------------------===//
// TPUPartitionedInputV2
//===----------------------------------------------------------------------===//

// This method mimics this op's core/TF-level shape inference logic
LogicalResult TPUPartitionedInputV2Op::verify() {
  TPUPartitionedInputV2Op op = *this;

  int num_partitions = 1;
  const mlir::ArrayAttr partition_dims = op.getPartitionDims();
  for (const mlir::Attribute &dim : partition_dims) {
    num_partitions *= dim.cast<IntegerAttr>().getInt();
  }

  const bool is_packed = op.getIsPacked();
  const bool replicated = partition_dims.empty();
  const int num_inputs_expected = is_packed ? 1 : num_partitions;

  if (!((replicated && !is_packed) || (op.getN() == num_inputs_expected))) {
    return op.emitOpError() << "expected " << num_inputs_expected
                            << " inputs, got " << op.getN();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  TransposeOp op = *this;
  auto perm_type = op.getPerm().getType().dyn_cast<RankedTensorType>();
  auto x_type = op.getX().getType().dyn_cast<RankedTensorType>();
  auto y_type = op.getY().getType().dyn_cast<RankedTensorType>();

  if (perm_type && perm_type.getRank() != 1) {
    return op.emitOpError()
           << "expected perm to be a 1-D Tensor, got perm of rank "
           << perm_type.getRank();
  }

  if (x_type && y_type && x_type.getRank() != y_type.getRank()) {
    return op.emitOpError() << "x should be of the same rank with y, got "
                            << "x of rank " << x_type.getRank()
                            << ", and y of rank " << y_type.getRank();
  }

  if (!x_type || !y_type || !perm_type || !perm_type.hasStaticShape()) {
    return success();
  }

  if (x_type.getRank() != perm_type.getNumElements()) {
    return op.emitOpError() << "expected perm to be a 1-D Tensor of size "
                            << "equal to the rank of x, got perm of size "
                            << perm_type.getNumElements() << ", and x of rank "
                            << x_type.getRank();
  }

  DenseIntElementsAttr attr_perm;
  if (matchPattern(op.getPerm(), m_Constant(&attr_perm))) {
    // y.shape[i] should be equal to x.shape[perm[i]]
    // for i = [0, 1, ..., rank(x) - 1]
    for (const auto &e : llvm::enumerate(attr_perm)) {
      const int64_t y_idx = e.index();
      const int64_t y_dim = y_type.getDimSize(y_idx);
      const int64_t x_idx = e.value().getSExtValue();
      const int64_t x_dim = x_type.getDimSize(x_idx);
      if (!ShapedType::isDynamic(y_dim) && !ShapedType::isDynamic(x_dim) &&
          y_dim != x_dim) {
        return op.emitOpError()
               << "requires y.shape[" << y_idx << "] (" << y_dim << ") "
               << "to be equal to x.shape[perm[" << x_idx << "]] "
               << "(" << x_dim << ")";
      }
    }
  }

  return success();
}

// TODO(jpienaar): perm could be optional too.
void TransposeOp::build(OpBuilder &builder, OperationState &result, Value x,
                        Value perm) {
  auto x_type = x.getType().cast<TensorType>();
  // If value is unranked, then so is results.
  if (!x_type.hasRank())
    return TransposeOp::build(builder, result,
                              UnrankedTensorType::get(x_type.getElementType()),
                              x, perm);

  // TODO(jpienaar): Handle unknown perm case.

  // TODO(jpienaar): Extract utility function.
  auto etype = x_type.cast<ShapedType>().getElementType();
  DenseIntElementsAttr attr_shape;
  if (matchPattern(perm, m_Constant(&attr_shape))) {
    llvm::SmallVector<int64_t, 4> const_shape;
    if (attr_shape.isSplat()) {
      const_shape.assign(
          attr_shape.getNumElements(),
          x_type.getDimSize((*attr_shape.begin()).getSExtValue()));
    } else {
      const_shape.reserve(attr_shape.getNumElements());
      for (const auto &dim : attr_shape)
        const_shape.push_back(x_type.getDimSize(dim.getSExtValue()));
    }
    return TransposeOp::build(
        builder, result,
        tensorflow::GetTypeFromTFTensorShape(const_shape, etype), x, perm);
  }
  return TransposeOp::build(builder, result, UnrankedTensorType::get(etype), x,
                            perm);
}

namespace {

OpFoldResult FoldIdentityTranspose(TransposeOp op) {
  DenseIntElementsAttr perm;
  if (!matchPattern(op.getPerm(), m_Constant(&perm))) return {};
  const auto elements = perm.getValues<APInt>();

  for (const auto &it : llvm::enumerate(elements)) {
    if (it.index() != it.value()) return {};
  }

  // TODO(jpienaar): Remove if/when we handle this more generally.
  if (op.getType() != op.getX().getType()) {
    // If the types don't match then only fold if all the operands are in the TF
    // dialect.
    for (auto user : op.getOperation()->getUsers())
      if (user->getDialect() != op->getDialect()) return {};
  }

  return op.getX();
}

OpFoldResult FoldCancellableTranspose(TransposeOp op) {
  // Operand is a TransposeOp.
  auto transpose = dyn_cast_or_null<TF::TransposeOp>(op.getX().getDefiningOp());
  if (!transpose) return {};

  // Permutations defined by constant operations.
  DenseIntElementsAttr perm0;
  DenseIntElementsAttr perm1;
  if (!matchPattern(op.getPerm(), m_Constant(&perm0)) ||
      !matchPattern(transpose.getPerm(), m_Constant(&perm1)))
    return {};

  // With permutation indices that cancel each other
  if (!AreCancellablePermutations(perm0, perm1)) return {};

  return transpose.getX();
}

}  // namespace

OpFoldResult TransposeOp::fold(FoldAdaptor) {
  if (auto folded = FoldIdentityTranspose(*this)) return folded;
  if (auto folded = FoldCancellableTranspose(*this)) return folded;
  return {};
}

//===----------------------------------------------------------------------===//
// TruncateDivOp
//===----------------------------------------------------------------------===//

void TruncateDivOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<TruncateDivWithSqrtDivisor>(context);
}

//===----------------------------------------------------------------------===//
// NonMaxSuppressionV3Op
//===----------------------------------------------------------------------===//

namespace {

// Canonicalize NonMaxSuppressionV3Op to NonMaxSuppressionV4Op.
class NMSV3ToNMSV4Op : public OpRewritePattern<NonMaxSuppressionV3Op> {
  using OpRewritePattern<NonMaxSuppressionV3Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(NonMaxSuppressionV3Op nms_op,
                                PatternRewriter &rewriter) const override {
    if (nms_op.getNumOperands() != 5) {
      return failure();
    }
    SmallVector<Type, 2> new_result_types;
    new_result_types.push_back(nms_op.getType());
    auto input_ty = nms_op.getType().template cast<ShapedType>();
    // corresponds to the second result type of nmsv4
    RankedTensorType valid_output_type =
        tensorflow::GetTypeFromTFTensorShape({}, input_ty.getElementType());
    new_result_types.push_back(valid_output_type);

    auto nmsv4 = rewriter.create<TF::NonMaxSuppressionV4Op>(
        nms_op.getLoc(), new_result_types, nms_op.getBoxes(),
        nms_op.getScores(), nms_op.getMaxOutputSize(), nms_op.getIouThreshold(),
        nms_op.getScoreThreshold());
    // Cannot replace the NMSv3 Op with NMSv4 since the outputs between the
    // two are different (v4 expects two output values vs v3 requires only one.
    nms_op.replaceAllUsesWith(nmsv4.getResult(0));
    return success();
  }
};
}  // namespace.

void NonMaxSuppressionV3Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<NMSV3ToNMSV4Op>(context);
}

//===----------------------------------------------------------------------===//
// FusedBatchNormOp
//===----------------------------------------------------------------------===//

namespace {

class ConvertFusedBatchNorm : public OpRewritePattern<TF::FusedBatchNormOp> {
  using OpRewritePattern<FusedBatchNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::FusedBatchNormOp tf_fused_batch_norm_op,
                                PatternRewriter &rewriter) const override {
    auto new_result_types =
        llvm::to_vector<6>(tf_fused_batch_norm_op.getResultTypes());
    // reserve_space_3
    new_result_types.push_back(
        UnrankedTensorType::get(FloatType::getF32(rewriter.getContext())));

    OperationState new_state(tf_fused_batch_norm_op.getLoc(),
                             TF::FusedBatchNormV3Op::getOperationName(),
                             tf_fused_batch_norm_op.getOperands(),
                             new_result_types,
                             tf_fused_batch_norm_op->getAttrs());
    Operation *tf_fused_batch_norm_op_v3 = rewriter.create(new_state);

    rewriter.replaceOp(tf_fused_batch_norm_op,
                       tf_fused_batch_norm_op_v3->getResults().drop_back());
    return success();
  }
};
}  // namespace.

void FusedBatchNormOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<ConvertFusedBatchNorm>(context);
}

//===----------------------------------------------------------------------===//
// UnpackOp
//===----------------------------------------------------------------------===//

LogicalResult UnpackOp::verify() {
  UnpackOp op = *this;
  auto value_type = op.getValue().getType().dyn_cast<RankedTensorType>();
  if (!value_type) return success();

  int64_t value_rank = value_type.getRank();
  int64_t axis = op.getAxis();
  if (axis < -value_rank || axis >= value_rank)
    return op.emitOpError("axis attribute must be in the range of [-")
           << value_rank << ", " << value_rank << ')';

  axis = GetDimForAxis(axis, value_rank);
  int64_t dim_size = value_type.getDimSize(axis);
  if (ShapedType::isDynamic(dim_size)) return success();

  if (dim_size != op.getNumResults())
    return op.emitOpError("result count must be equal to ") << dim_size;

  return success();
}

namespace {

// Hoist coefficient-wise unary operation out of the Unpack op:
//
//   %unpacked:N = "tf.Unpack"(%0)
//   %neg0 = "tf.Neg"(%unpacked#0)
//   %neg1 = "tf.Neg"(%unpacked#1)
//   ...
//   %negN-1 = "tf.Neg"(%unpacked:N-1)
//
// Rewrite it to:
//
//   %neg = "tf.Neg"(%0)
//   %unpacked:N = "tf.Unpack"(%neg)
class HoistCwiseUnaryOutOfUnpack : public OpRewritePattern<UnpackOp> {
 public:
  explicit HoistCwiseUnaryOutOfUnpack(MLIRContext *context)
      : OpRewritePattern<UnpackOp>(context) {}
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult HoistCwiseUnaryOutOfUnpack::matchAndRewrite(
    UnpackOp op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // First unpack user must be coeff-wise unary operation.
  Operation *first_user = *op->getUsers().begin();
  if (!first_user->hasTrait<OpTrait::TF::CwiseUnary>()) return failure();

  // All unpack users must be defined by the op of same kind.
  bool users_same_op = llvm::all_of(op->getUsers(), [&](Operation *user) {
    return user->getName() == first_user->getName();
  });
  if (!users_same_op) return failure();

  // Pass unpack operand to unary operation.
  OperationState new_unary_op_state(loc, first_user->getName().getStringRef(),
                                    op.getOperand(), op.getOperand().getType(),
                                    ArrayRef<NamedAttribute>());
  Operation *new_unary_op = rewriter.create(new_unary_op_state);

  // Unpack results after applying unary operation.
  auto unpack_unary_op = rewriter.create<UnpackOp>(
      loc, op.getResultTypes(), new_unary_op->getResult(0), op.getAxis());

  // Bypass all users of the original unpack operation and use `unpack_unary_op`
  // results instead.
  for (auto pair : llvm::zip(op.getResults(), unpack_unary_op.getResults())) {
    OpResult old_result = std::get<0>(pair);  // result of original Unpack
    OpResult new_result = std::get<1>(pair);  // result of transformed Unpack
    for (Operation *user : llvm::make_early_inc_range(old_result.getUsers()))
      rewriter.replaceOp(user, ValueRange(new_result));
  }

  // Erase original unpack operation.
  rewriter.eraseOp(op.getOperation());

  return success();
}

}  // namespace

void UnpackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<HoistCwiseUnaryOutOfUnpack>(context);
}

//===----------------------------------------------------------------------===//
// Unsorted segment reduction ops
//===----------------------------------------------------------------------===//

template <class Op>
static LogicalResult VerifyUnsortedSegmentReduction(Op op) {
  if (!HasRankAtMost(op.getNumSegments(), 0))
    return op.emitOpError("number of segments should be a 0-D tensor");

  auto data_type = op.getData().getType().template dyn_cast<RankedTensorType>();
  auto segment_ids_type =
      op.getSegmentIds().getType().template dyn_cast<RankedTensorType>();
  if (data_type && segment_ids_type) {
    if (data_type.getRank() < segment_ids_type.getRank())
      return op.emitOpError(
          "requires segment ids rank to be less than or equal to data's rank");

    int index = 0;
    for (auto shape_pair :
         llvm::zip_first(segment_ids_type.getShape(), data_type.getShape())) {
      int64_t segment_id_dim = std::get<0>(shape_pair);
      int64_t data_dim = std::get<1>(shape_pair);
      if (!ShapedType::isDynamic(segment_id_dim) &&
          !ShapedType::isDynamic(data_dim) && segment_id_dim != data_dim)
        return op.emitOpError(
                   "requires segment ids shape to be a prefix of data shape, "
                   "but dimension #")
               << index << " differs: " << segment_id_dim << " vs. "
               << data_dim;
      ++index;
    }
  }

  DenseIntElementsAttr num_segments_attr;
  if (matchPattern(op.getNumSegments(), m_Constant(&num_segments_attr))) {
    int64_t num_segments = (*num_segments_attr.begin()).getSExtValue();
    if (num_segments < 0)
      return op.emitOpError("num of segments cannot be negative");
  }

  return success();
}

LogicalResult UnsortedSegmentMaxOp::verify() {
  return VerifyUnsortedSegmentReduction(*this);
}
LogicalResult UnsortedSegmentMinOp::verify() {
  return VerifyUnsortedSegmentReduction(*this);
}
LogicalResult UnsortedSegmentProdOp::verify() {
  return VerifyUnsortedSegmentReduction(*this);
}
LogicalResult UnsortedSegmentSumOp::verify() {
  return VerifyUnsortedSegmentReduction(*this);
}

//===----------------------------------------------------------------------===//
// VarHandleOp
//===----------------------------------------------------------------------===//

LogicalResult VarHandleOp::verify() {
  // VarHandleOp requires the resource handle supply a single subtype from
  // which to derive the dtype and shape attributes.
  if (resource_type().getSubtypes().size() != 1) {
    return emitOpError(
        "must have exactly one subtype in the result resource type");
  }

  return success();
}

llvm::SmallVector<ResourceHandleValueAndId, 4>
VarHandleOp::GetResourceHandleValueAndIdList(
    llvm::SmallDenseMap<ResourceHandle, int64_t> &resource_handle_id_map,
    int64_t &next_id) {
  llvm::StringRef device = GetDeviceOrEmpty(getOperation());
  return {GetResourceHandleValueAndIdBase(getContainer(), getSharedName(),
                                          device, getResource(),
                                          resource_handle_id_map, next_id)};
}

//===----------------------------------------------------------------------===//
// VarIsInitializedOp
//===----------------------------------------------------------------------===//

namespace {

/// Erase VarIsInitializedOp operations with no uses. This op has side effect on
/// resources (read-only), but can still be deleted if it has zero uses.
struct EraseDeadVarIsInitializedOp
    : public OpRewritePattern<VarIsInitializedOp> {
  using OpRewritePattern<VarIsInitializedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VarIsInitializedOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.use_empty()) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};
}  // end anonymous namespace.

void VarIsInitializedOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<EraseDeadVarIsInitializedOp>(context);
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void VariableOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<VariableToVariableV2>(context);
}

//===----------------------------------------------------------------------===//
// VariableShapeOp
//===----------------------------------------------------------------------===//

LogicalResult VariableShapeOp::verify() {
  VariableShapeOp op = *this;
  auto input_type = op.getInput().getType().cast<TensorType>();
  if (input_type.hasStaticShape() && input_type.getNumElements() != 1)
    return op.emitOpError("requires input to have one resource");

  auto resource_type = input_type.getElementType().cast<TF::ResourceType>();
  auto subtypes = resource_type.getSubtypes();
  switch (subtypes.size()) {
    case 1:
      return VerifyShapeOperandAndResult(
          op, resource_type.getSubtypes().front(), op.getType());
    case 0:
      return VerifyShapeOperandAndResult(op, Type(), op.getType());
    default:
      return op.emitOpError(
          "requires resource input type to have at most 1 subtype");
  }
}

OpFoldResult VariableShapeOp::fold(FoldAdaptor) {
  int width =
      getType().cast<ShapedType>().getElementType().getIntOrFloatBitWidth();
  auto resource_type =
      getElementTypeOrSelf(getOperand().getType()).cast<TF::ResourceType>();
  if (resource_type.getSubtypes().empty()) return {};
  return ConvertShapeToAttr(resource_type.getSubtypes()[0], width);
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

static LogicalResult VerifyWhileTypes(Operation *op, TypeRange cond_input,
                                      TypeRange body_input,
                                      TypeRange body_result,
                                      bool shape_invariant) {
  const TypeRangeWithDesc input_type = {op->getOperandTypes(), "input"};
  const TypeRangeWithDesc result_type = {op->getResultTypes(), "result"};
  constexpr int kNumRegionTypeLists = 3;
  const std::array<TypeRangeWithDesc, kNumRegionTypeLists> region_types = {{
      {body_result, "body result"},
      {cond_input, "condition input"},
      {body_input, "body input"},
  }};

  // A pair of type lists should be cast compatible with each other if one is
  // converted to the another for a function call or assignment or there is a
  // common source of inputs for both. Therefore, the While op requires the
  // following pairs of type lists to be cast compatible for the tensor_cast
  // operation:
  //
  // * Operands and cond inputs to call the cond function before the
  //   first iteration.
  // * Operands and body inputs to call the body function for the first
  //   iteration if the cond functions returns True or equivalent result.
  // * Operands and results to assign cond function arguments to op results if
  //   the cond function returns False or equivalent result. If the op is shape
  //   invariant, this does not hold as shapes can differ.
  // * All three pairs using cond inputs, body inputs and results as operand is
  //   a common source for all three.
  // * Body result and cond inputs to call the cond function for the subsequent
  //   iterations. Similarly, Body result should be compatible with body inputs
  //   and op results.
  //
  // Note that the operands and body results need not be compatible as they are
  // never converted from one to the another nor there is a common source
  // tensors. Compatibility requirement is not transitive.

  if (!shape_invariant &&
      failed(VerifyTypeRangesAreCompatible(op, input_type, result_type)))
    return failure();

  // Skip the first pair as the While op operands and body function results does
  // not need to be compatible with each other.
  for (int i = 1; i < kNumRegionTypeLists; ++i)
    if (failed(VerifyTypeRangesAreCompatible(op, input_type, region_types[i])))
      return failure();

  for (int i = 0; i < kNumRegionTypeLists; ++i)
    if (failed(VerifyTypeRangesAreCompatible(op, result_type, region_types[i])))
      return failure();

  for (int i = 0; i < kNumRegionTypeLists; ++i)
    for (int j = i + 1; j < kNumRegionTypeLists; ++j)
      if (failed(VerifyTypeRangesAreCompatible(op, region_types[i],
                                               region_types[j])))
        return failure();

  return success();
}

LogicalResult WhileOp::verifySymbolUses(SymbolTableCollection &symbol_table) {
  auto cond_fn =
      symbol_table.lookupNearestSymbolFrom<func::FuncOp>(*this, getCondAttr());
  auto body_fn =
      symbol_table.lookupNearestSymbolFrom<func::FuncOp>(*this, getBodyAttr());
  if (!cond_fn) {
    return emitOpError("cond refers to an undefined function : ") << getCond();
  }
  if (!body_fn) {
    return emitOpError("body refers to an undefined function : ") << getBody();
  }

  auto cond_fn_type = cond_fn.getFunctionType();
  auto body_fn_type = body_fn.getFunctionType();

  // Verify that the cond function has exactly one result.
  if (cond_fn_type.getNumResults() != 1)
    return emitOpError("requires cond function to have exactly one result");

  return VerifyWhileTypes(*this, /*cond_input=*/cond_fn_type.getInputs(),
                          /*body_input=*/body_fn_type.getInputs(),
                          /*body_result=*/body_fn_type.getResults(),
                          getShapeInvariant());
}

//===----------------------------------------------------------------------===//
// WhileRegionOp
//===----------------------------------------------------------------------===//
LogicalResult WhileRegionOp::verify() {
  WhileRegionOp op = *this;
  // Verify that the condition generates a single tensor<i1> result.
  Operation *cond_yield = op.getCond().front().getTerminator();
  if (cond_yield->getNumOperands() != 1)
    return op.emitOpError()
           << "condition should have a single tensor<i1> result";

  auto cond_type =
      cond_yield->getOperand(0).getType().dyn_cast<RankedTensorType>();
  if (!cond_type || !cond_type.getShape().equals({}) ||
      !cond_type.getElementType().isInteger(/*width=*/1))
    return op.emitOpError()
           << "condition should have a single tensor<i1> result";

  Operation *body_yield = op.getBody().front().getTerminator();
  if (failed(VerifyWhileTypes(op,
                              /*cond_input=*/op.getCond().getArgumentTypes(),
                              /*body_input=*/op.getBody().getArgumentTypes(),
                              /*body_result=*/body_yield->getOperandTypes(),
                              op.getShapeInvariant())))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// WhileRegionOp LoopLikeOpInterface
//===----------------------------------------------------------------------===//

Region &WhileRegionOp::getLoopBody() { return getBody(); }

//===----------------------------------------------------------------------===//
// WhileRegionOp canonicalization
//===----------------------------------------------------------------------===//
namespace {

// Make casts before a `WhileRegion` be explicit. After this rewrite a
// `WhileRegion` operand will have the same type as its corresponding iteration
// variable. An operand and its iteration variables with the same type enables
// WhileRegionEliminatePassthrough.
struct WhileRegionExplicitCast : public OpRewritePattern<WhileRegionOp> {
  using OpRewritePattern<WhileRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileRegionOp while_op,
                                PatternRewriter &rewriter) const override {
    auto &body_block = while_op.getBody().front();
    auto &cond_block = while_op.getCond().front();
    bool changed = false;
    for (int op_idx : llvm::seq<int>(0, while_op.getNumOperands())) {
      auto body_arg = body_block.getArgument(op_idx);
      auto cond_arg = cond_block.getArgument(op_idx);
      auto while_operand = while_op.getOperand(op_idx);
      // Do not change if the body and cond type differ since there is no type
      // to cast to.
      if (body_arg.getType() == cond_arg.getType() &&
          body_arg.getType() != while_operand.getType()) {
        changed = true;
        rewriter.setInsertionPoint(while_op);
        auto cast_op = rewriter.create<CastOp>(
            while_op.getLoc(), body_arg.getType(), while_operand);
        while_op.setOperand(op_idx, cast_op);
      }
    }
    return success(changed);
  }
};

// Eliminate values that pass through the WhileRegionOp body.
struct WhileRegionEliminatePassThrough
    : public OpRewritePattern<WhileRegionOp> {
  using OpRewritePattern<WhileRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileRegionOp while_op,
                                PatternRewriter &rewriter) const override {
    // Remove any extern values that are explicitly captured and returned. Also
    // replace values that simply passthrough the body with extern values. The
    // block arguments of body and while match and so the corresponding cond
    // argument can be easily found.
    int old_num_operands = while_op.getNumOperands();
    int new_num_operands = old_num_operands;
    auto &body_block = while_op.getBody().front();
    auto &cond_block = while_op.getCond().front();
    auto &yield = *body_block.getTerminator();

    // Bit mask indicating which operands will be removed.
    llvm::BitVector removed_operand(old_num_operands);

    for (int op_idx : llvm::seq<int>(0, old_num_operands)) {
      auto body_arg = body_block.getArgument(op_idx);
      auto yield_operand = LookThroughIdentity(yield.getOperand(op_idx));
      auto while_operand = while_op.getOperand(op_idx);
      if (body_arg == yield_operand || while_operand == yield_operand) {
        // Replace the use of the passthrough value with the while operand
        // in the body and condition regions, as well as the while output (if
        // type match)
        // TODO(jurahul): Use PatternRewriter API for IR modification.
        if (body_arg.getType() == while_operand.getType())
          body_arg.replaceAllUsesWith(while_operand);

        auto cond_arg = cond_block.getArgument(op_idx);
        if (cond_arg.getType() == while_operand.getType())
          cond_arg.replaceAllUsesWith(while_operand);

        auto result = while_op.getResult(op_idx);
        if (result.getType() == while_operand.getType())
          result.replaceAllUsesWith(while_operand);
      }

      // Now check if the operand is unused in both regions as well as the
      // result. If so, mark it for removal.
      if (body_block.getArgument(op_idx).use_empty() &&
          cond_block.getArgument(op_idx).use_empty() &&
          while_op.getResult(op_idx).use_empty()) {
        removed_operand.set(op_idx);
        new_num_operands--;
      }
    }

    if (new_num_operands == old_num_operands) return failure();

    // Compress the operands, region arguments, and outputs.
    SmallVector<Value, 4> new_while_operands;
    SmallVector<Type, 4> new_result_types;
    new_while_operands.reserve(new_num_operands);
    new_result_types.reserve(new_num_operands);

    // Build new operands and result type.
    for (int op_idx : llvm::seq<int>(0, old_num_operands)) {
      if (removed_operand.test(op_idx)) continue;
      new_while_operands.push_back(while_op.getOperand(op_idx));
      new_result_types.push_back(while_op.getResult(op_idx).getType());
    }

    // Create the new while operation.
    auto new_while_op = rewriter.create<WhileRegionOp>(
        while_op.getLoc(), new_result_types, new_while_operands,
        while_op->getAttrs());

    // Move region bodies to the new while.
    rewriter.inlineRegionBefore(while_op.getCond(), new_while_op.getCond(),
                                new_while_op.getCond().end());
    rewriter.inlineRegionBefore(while_op.getBody(), new_while_op.getBody(),
                                new_while_op.getBody().end());

    auto &new_cond_block = new_while_op.getCond().front();
    auto &new_body_block = new_while_op.getBody().front();
    auto &new_yield = *new_body_block.getTerminator();

    // Patch up the region bodies and yield.
    new_cond_block.eraseArguments(removed_operand);
    new_body_block.eraseArguments(removed_operand);
    new_yield.eraseOperands(removed_operand);

    // Build a vector of new results. Also patch up the region bodies and
    // yield.
    SmallVector<Value, 4> new_results(old_num_operands);
    int next_idx = 0;
    for (int op_idx : llvm::seq<int>(0, old_num_operands))
      if (!removed_operand.test(op_idx))
        new_results[op_idx] = new_while_op.getResult(next_idx++);

    rewriter.replaceOp(while_op, new_results);
    return success();
  }
};

}  // anonymous namespace

void WhileRegionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<WhileRegionExplicitCast, WhileRegionEliminatePassThrough>(
      context);
}

//===----------------------------------------------------------------------===//
// XdivyOp
//===----------------------------------------------------------------------===//

void XdivyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<XdivyWithSqrtDivisor>(context);
}

//===----------------------------------------------------------------------===//
// XlaBroadcastHelperOp
//===----------------------------------------------------------------------===//

LogicalResult XlaBroadcastHelperOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  XlaBroadcastHelperOpAdaptor op(operands.getValues(), attributes);
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  auto set_unranked_results = [&]() {
    inferredReturnShapes.emplace_back(getElementTypeOrSelf(lhs));
    inferredReturnShapes.emplace_back(getElementTypeOrSelf(rhs));
    return success();
  };

  RankedTensorType lhs_ty = lhs.getType().dyn_cast<RankedTensorType>();
  RankedTensorType rhs_ty = rhs.getType().dyn_cast<RankedTensorType>();
  if (!lhs_ty || !rhs_ty) return set_unranked_results();

  int64_t lhs_rank = lhs_ty.getRank();
  int64_t rhs_rank = rhs_ty.getRank();

  DenseIntElementsAttr dims;
  if (!matchPattern(op.getBroadcastDims(), m_Constant(&dims))) {
    return set_unranked_results();
  }

  if (dims.size() == 0) {
    if (lhs_rank != rhs_rank && lhs_rank != 0 && rhs_rank != 0) {
      return emitOptionalError(
          location,
          "if broadcast_dims is empty, both arguments must have equal rank or "
          "at least one argument must be a scalar");
    }
    inferredReturnShapes.emplace_back(lhs_ty.cast<ShapedType>());
    inferredReturnShapes.emplace_back(rhs_ty.cast<ShapedType>());
    return success();
  }

  const bool broadcast_lhs = lhs_rank < rhs_rank;
  RankedTensorType min_rank_ty = broadcast_lhs ? lhs_ty : rhs_ty;
  RankedTensorType max_rank_ty = broadcast_lhs ? rhs_ty : lhs_ty;

  if (dims.size() != min_rank_ty.getRank()) {
    return emitOptionalError(
        location,
        "broadcast_dims must have size equal to the smaller argument rank");
  }

  int64_t output_rank = max_rank_ty.getRank();
  llvm::SmallVector<int64_t, 4> broadcast_shape(output_rank, 1LL);
  llvm::SmallVector<bool, 4> is_broadcasted(output_rank, false);
  for (const auto &item : llvm::enumerate(dims)) {
    int64_t index = item.index();
    int64_t dim = item.value().getSExtValue();
    if (dim < 0 || dim > output_rank) {
      return emitOptionalError(location, "out of range broadcast dim");
    }
    if (is_broadcasted[dim]) {
      return emitOptionalError(location, "broadcast_dims has duplicates");
    }
    broadcast_shape[dim] = min_rank_ty.getDimSize(index);
    is_broadcasted[dim] = true;
  }

  if (broadcast_lhs) {
    inferredReturnShapes.emplace_back(broadcast_shape, lhs_ty.getElementType());
    inferredReturnShapes.emplace_back(rhs_ty.cast<ShapedType>());
  } else {
    inferredReturnShapes.emplace_back(lhs_ty.cast<ShapedType>());
    inferredReturnShapes.emplace_back(broadcast_shape, rhs_ty.getElementType());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XlaConvOp
//===----------------------------------------------------------------------===//

class XlaConvToV2 : public OpRewritePattern<TF::XlaConvOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaConvOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> result_types{op.getResult().getType()};
    rewriter.replaceOpWithNewOp<TF::XlaConvV2Op>(
        op, op.getResult().getType(), op.getLhs(), op.getRhs(),
        op.getWindowStrides(), op.getPadding(), op.getLhsDilation(),
        op.getRhsDilation(), op.getFeatureGroupCount(),
        op.getDimensionNumbers(), op.getPrecisionConfig(), 1);
    return ::mlir::success();
  };
};

void XlaConvOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<XlaConvToV2>(context);
}

//===----------------------------------------------------------------------===//
// XlaConvV2Op
//===----------------------------------------------------------------------===//

LogicalResult XlaConvV2Op::verify() {
  XlaConvV2Op op = *this;
  DenseElementsAttr window_strides_attr, padding_attr, lhs_dilation_attr,
      rhs_dilation_attr, feature_group_count_attr;
  if (!(matchPattern(op.getWindowStrides(), m_Constant(&window_strides_attr)) &&
        matchPattern(op.getPadding(), m_Constant(&padding_attr)) &&
        matchPattern(op.getLhsDilation(), m_Constant(&lhs_dilation_attr)) &&
        matchPattern(op.getRhsDilation(), m_Constant(&rhs_dilation_attr)) &&
        matchPattern(op.getFeatureGroupCount(),
                     m_Constant(&feature_group_count_attr))))
    return success();

  if (window_strides_attr.getType().getRank() != 1)
    return op.emitOpError() << "expects window_stride to be a vector";

  const ShapedType &padding_ty = padding_attr.getType();
  if (padding_ty.getRank() != 2 || padding_ty.getDimSize(1) != 2)
    return op.emitOpError()
           << "expects padding to be a matrix with minor dimension 2";

  if (lhs_dilation_attr.getType().getRank() != 1)
    return op.emitOpError() << "expects lhs_dilation to be a vecotr";

  if (rhs_dilation_attr.getType().getRank() != 1)
    return op.emitOpError() << "expects rhs_dilation to be a vecotr";

  if (feature_group_count_attr.getType().getRank())
    return op.emitOpError() << "expects feature_group_count to be a scalar";

  return success();
}

//===----------------------------------------------------------------------===//
// XlaSetDynamicDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult XlaSetDynamicDimensionSizeOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  XlaSetDynamicDimensionSizeOpAdaptor op(operands.getValues(), attributes);

  TensorType operand_ty = op.getInput().getType().cast<TensorType>();
  Type element_ty = operand_ty.getElementType();

  TensorType result_ty;
  if (operand_ty.hasRank()) {
    auto shape = llvm::to_vector<4>(operand_ty.getShape());

    DenseIntElementsAttr dim_index_attr;
    if (matchPattern(op.getDimIndex(), m_Constant(&dim_index_attr))) {
      int64_t dim_index = dim_index_attr.getValues<APInt>()[0].getSExtValue();

      int64_t rank = operand_ty.getRank();
      if (dim_index < 0 || dim_index >= rank) {
        return emitOptionalError(location, "dim_index (", dim_index,
                                 ") is out of range [0, ", rank, ")");
      }
      shape[dim_index] = ShapedType::kDynamic;
    } else {
      shape.assign(shape.size(), ShapedType::kDynamic);
    }
    result_ty = tensorflow::GetTypeFromTFTensorShape(shape, element_ty);
  } else {
    result_ty = UnrankedTensorType::get(element_ty);
  }

  inferredReturnShapes.emplace_back(result_ty.cast<ShapedType>());
  return success();
}

//===----------------------------------------------------------------------===//
// XlaReduceOp
//===----------------------------------------------------------------------===//

class XlaReduceToXlaVariadicReduceV2
    : public OpRewritePattern<TF::XlaReduceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaReduceOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs{op.getInput()};
    SmallVector<Value> init_values{op.getInitValue()};
    SmallVector<Type> result_types{op.getResult().getType()};
    rewriter.replaceOpWithNewOp<TF::XlaVariadicReduceV2Op>(
        op, result_types, inputs, init_values, op.getDimensionsToReduce(),
        op.getReducer());
    return ::mlir::success();
  };
};

void XlaReduceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<XlaReduceToXlaVariadicReduceV2>(context);
}

//===----------------------------------------------------------------------===//
// XlaReduceWindowOp
//===----------------------------------------------------------------------===//

LogicalResult XlaReduceWindowOp::verify() {
  XlaReduceWindowOp op = *this;
  const auto &input_ty = op.getInput().getType().cast<ShapedType>();

  auto check = [&](mlir::Value val, std::string attr_name) -> LogicalResult {
    ElementsAttr attr;
    if (matchPattern(val, m_Constant(&attr))) {
      if (attr.getType().getRank() != 1) {
        return op.emitOpError() << "expects the rank of " << attr_name
                                << "to be 1, got " << attr.getType().getRank();
      }
      if (input_ty.hasRank()) {
        int64_t input_rank = input_ty.getRank();
        int64_t size = attr.size();
        if (input_rank != size) {
          return op.emitOpError() << "expects the size of " << attr_name
                                  << " to be equal to the input "
                                     "rank ("
                                  << size << " vs. " << input_rank << ")";
        }
      }
    }
    return success();
  };

  if (check(op.getWindowDimensions(), "window_dimensions").failed())
    return failure();

  if (check(op.getWindowStrides(), "window_strides").failed()) return failure();

  if (check(op.getBaseDilations(), "base_dilations").failed()) return failure();

  if (check(op.getWindowDilations(), "window_dilations").failed())
    return failure();

  ElementsAttr padding;
  if (matchPattern(op.getPadding(), m_Constant(&padding))) {
    const ShapedType &padding_ty = padding.getType();
    if (padding_ty.getRank() != 2 || padding_ty.getDimSize(1) != 2) {
      return op.emitOpError()
             << "expects padding to be a matrix with minor dimension 2, got "
             << padding.getType().getShape();
    }
  }

  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto func = dyn_cast_or_null<mlir::func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, op.getComputation()));
  if (!func) {
    return op.emitOpError() << "has no reduction function specified";
  }

  auto func_type = func.getFunctionType();

  if (func_type.getNumInputs() != 2) {
    return op.emitOpError()
           << "expects reduction function to take 2 parameters, but "
              "has "
           << func_type.getNumInputs() << " parameter(s)";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XlaSelectAndScatterOp
//===----------------------------------------------------------------------===//

LogicalResult XlaSelectAndScatterOp::verify() {
  XlaSelectAndScatterOp op = *this;
  auto input_ty = op.getOperand().getType().cast<ShapedType>();

  auto check = [&](mlir::Value val, std::string attr_name) -> LogicalResult {
    ElementsAttr attr;
    if (input_ty.hasRank() && matchPattern(val, m_Constant(&attr))) {
      int64_t input_rank = input_ty.getRank();
      int64_t size = attr.size();
      if (input_rank != size) {
        return op.emitOpError() << "expects the size of " << attr_name
                                << "to be equal to the input "
                                   "rank ("
                                << size << " vs. " << input_rank << ")";
      }
    }
    return success();
  };

  if (check(op.getWindowDimensions(), "window_dimensions").failed())
    return failure();

  if (check(op.getWindowStrides(), "window_strides").failed()) return failure();

  ElementsAttr padding;
  if (matchPattern(op.getPadding(), m_Constant(&padding))) {
    const ShapedType &padding_ty = padding.getType();
    if (padding_ty.getRank() != 2 || padding_ty.getDimSize(1) != 2) {
      return op.emitOpError()
             << "expects padding to be a matrix with minor dimension 2, got "
             << padding.getType().getShape();
    }
  }

  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto select_func = dyn_cast_or_null<mlir::func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, op.getSelect()));
  if (!select_func) {
    return op.emitOpError() << "has no select function specified";
  }
  auto select_func_type = select_func.getFunctionType();
  if (select_func_type.getNumInputs() != 2) {
    return op.emitOpError()
           << "expects select function to take 2 parameters, but has "
           << select_func_type.getNumInputs() << " parameter(s)";
  }
  if (select_func_type.getNumResults() != 1 ||
      !getElementTypeOrSelf(select_func_type.getResult(0)).isInteger(1)) {
    return op.emitOpError() << "expects select function to return a single "
                               "boolean result but got "
                            << select_func_type.getResult(0);
  }
  auto scatter_func = dyn_cast_or_null<mlir::func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, op.getScatter()));
  if (!scatter_func) {
    return op.emitOpError() << "has no scatter function specified";
  }
  auto scatter_func_type = scatter_func.getFunctionType();
  if (scatter_func_type.getNumInputs() != 2) {
    return op.emitOpError()
           << "expects scatter function to take 2 parameters, but has "
           << scatter_func_type.getNumInputs() << " parameter(s)";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XlaVariadicReduceOp
//===----------------------------------------------------------------------===//

LogicalResult XlaVariadicReduceOp::verify() {
  XlaVariadicReduceOp op = *this;
  // We rely on V2 for the majority of the checks.
  const auto &input_ty = op.getInput().getType();
  if (input_ty.empty()) return op.emitOpError() << "No input";
  const auto &dtype = input_ty[0].cast<TensorType>().getElementType();
  for (const auto &ty : input_ty) {
    if (ty.cast<TensorType>().getElementType() != dtype)
      return op.emitOpError()
             << "This version is limited to operands of the same dtype";
  }
  return success();
}

class XlaVariadicReduceToV2 : public OpRewritePattern<TF::XlaVariadicReduceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaVariadicReduceOp op,
                                PatternRewriter &rewriter) const override {
    mlir::TF::XlaVariadicReduceV2Op xla_variadic_reduce_v2_op =
        rewriter.create<::mlir::TF::XlaVariadicReduceV2Op>(
            op.getLoc(), op.getResults().getTypes(), op.getInput(),
            op.getInitValue(), op.getDimensionsToReduce(), op.getReducer());

    rewriter.replaceOp(op, xla_variadic_reduce_v2_op.getResults());
    return ::mlir::success();
  };
};

void XlaVariadicReduceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<XlaVariadicReduceToV2>(context);
}

//===----------------------------------------------------------------------===//
// XlaVariadicReduceV2Op
//===----------------------------------------------------------------------===//

LogicalResult XlaVariadicReduceV2Op::verify() {
  XlaVariadicReduceV2Op op = *this;
  const auto &inputs_ty = op.getInputs().getType();
  int n_inputs = inputs_ty.size();
  if (n_inputs < 1) return op.emitOpError() << "No inputs";

  const auto &init_values_ty = op.getInitValues().getType();
  int n_init_values = init_values_ty.size();
  if (n_init_values != n_inputs) {
    return op.emitOpError() << "Number of inputs (" << n_inputs
                            << ") is different than number of init_values ("
                            << n_init_values << ")";
  }

  auto input_ty_0 = inputs_ty[0].cast<ShapedType>();
  if (input_ty_0.hasStaticShape()) {
    for (int i = 1; i < n_inputs; ++i) {
      auto input_ty_i = inputs_ty[i].cast<ShapedType>();
      if (input_ty_i.hasStaticShape() &&
          input_ty_i.getShape() != input_ty_0.getShape()) {
        return op.emitOpError()
               << "inputs[" << i << "] has shape [" << input_ty_i.getShape()
               << "] different than the shape of inputs[0]: "
               << input_ty_0.getShape();
      }
    }

    if (op.getDimensionsToReduce().size() > input_ty_0.getRank()) {
      return op.emitOpError()
             << "Invalid dimensions_to_reduce argument to XlaVariadicReduceV2";
    }
  }

  for (int i = 0; i < n_inputs; ++i) {
    auto init_value_ty_i = init_values_ty[i].cast<ShapedType>();
    if (init_value_ty_i.hasRank() && init_value_ty_i.getRank() != 0) {
      return op.emitOpError()
             << "init_values[" << i << "] must be a scalar but got ["
             << init_value_ty_i.getShape() << "]";
    }
  }

  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto function = dyn_cast_or_null<mlir::func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, op.getReducer()));
  if (!function) return op.emitOpError() << "No reducer";
  if (!function.getBody().hasOneBlock())
    return op.emitOpError() << "reducer has more than one block";

  return success();
}

//===----------------------------------------------------------------------===//
// XlaVariadicSortOp
//===----------------------------------------------------------------------===//

LogicalResult XlaVariadicSortOp::verify() {
  XlaVariadicSortOp op = *this;
  const auto &inputs_ty = op.getInputs().getType();
  int n_inputs = inputs_ty.size();
  auto input_ty_0 = inputs_ty[0].cast<ShapedType>();
  if (input_ty_0.hasStaticShape()) {
    for (int i = 1; i < n_inputs; ++i) {
      auto input_ty_i = inputs_ty[i].cast<ShapedType>();
      if (input_ty_i.hasStaticShape() &&
          input_ty_i.getShape() != input_ty_0.getShape()) {
        return op.emitOpError()
               << "input[" << i << "] has shape [" << input_ty_i.getShape()
               << "] different than the shape of input[0]: "
               << input_ty_0.getShape();
      }
    }
  }

  ElementsAttr dimension;
  if (matchPattern(op.getDimension(), m_Constant(&dimension))) {
    if (dimension.getType().getRank() != 0 ||
        dimension.getType().getNumElements() != 1)
      return op.emitOpError() << "dimension must be a scalar";
  }

  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto function = dyn_cast_or_null<mlir::func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, op.getComparator()));
  if (!function) return op.emitOpError() << "No comparator";
  if (!function.getBody().hasOneBlock())
    return op.emitOpError() << "comparator has more than one block";

  return success();
}

//===----------------------------------------------------------------------===//
// SetStaticDimensionBoundsOp
//===----------------------------------------------------------------------===//
//

LogicalResult SetStaticDimensionBoundsOp::verify() {
  SetStaticDimensionBoundsOp op = *this;
  mlir::ShapedType input_type =
      op.getInput().getType().cast<mlir::ShapedType>();
  mlir::ShapedType static_shape_type =
      op.getStaticShape().getType().cast<mlir::ShapedType>();
  int input_type_rank = input_type.hasRank() ? input_type.getRank() : -1;
  if (input_type_rank > 2) {
    return op.emitOpError() << "was used with an input tensor with rank > 2, "
                               "only tensors of rank 1,2 are supported";
  }

  if (static_shape_type.hasRank() && static_shape_type.getRank() != 1) {
    return op.emitOpError("static shape must be of rank 1 (vector)");
  }
  if (input_type_rank != -1 && static_shape_type.hasStaticShape()) {
    if (static_shape_type.getShape()[0] != input_type_rank) {
      return op.emitOpError(
          "static shape must have num_elements == rank of input "
          "tensor");
    }
  }

  return success();
}

namespace {

template <typename UniformQuantizedOp>
LogicalResult VerifyScalesAndZeroPoints(UniformQuantizedOp op, Value scales,
                                        Value zero_points,
                                        int32_t quantization_axis) {
  ShapedType scales_type = scales.getType().cast<ShapedType>();
  ShapedType zero_points_type = zero_points.getType().cast<ShapedType>();

  if (quantization_axis == -1) {
    if (scales_type.hasRank() && scales_type.getRank() != 0) {
      return op.emitOpError(
          "quantization_axis is -1, scales must have 0 rank.");
    }
    if (zero_points_type.hasRank() && zero_points_type.getRank() != 0) {
      return op.emitOpError(
          "quantization_axis is -1, zero_points must have 0 rank.");
    }
  } else {
    if (scales_type.hasRank() && scales_type.getRank() != 1) {
      return op.emitOpError(
          "quantization_axis is not -1, scales must have 1 rank.");
    }
    if (zero_points_type.hasRank() && zero_points_type.getRank() != 1) {
      return op.emitOpError(
          "quantization_axis is not -1, zero_points must have 1 rank.");
    }
    if (scales_type.hasStaticShape() && zero_points_type.hasStaticShape() &&
        scales_type.getNumElements() != zero_points_type.getNumElements()) {
      return op.emitOpError(
          "scales and zero points must have same number of elements.");
    }
  }

  return success();
}

template <typename UniformQuantizedOp>
LogicalResult VerifyLhsRhsBothUniformQuantizedOp(UniformQuantizedOp op) {
  auto verify_lhs_params =
      VerifyScalesAndZeroPoints(op, op.getLhsScales(), op.getLhsZeroPoints(),
                                op.getLhsQuantizationAxis());
  if (failed(verify_lhs_params)) {
    return failure();
  }

  auto verify_rhs_params =
      VerifyScalesAndZeroPoints(op, op.getRhsScales(), op.getRhsZeroPoints(),
                                op.getRhsQuantizationAxis());
  if (failed(verify_rhs_params)) {
    return failure();
  }

  return VerifyScalesAndZeroPoints(op, op.getOutputScales(),
                                   op.getOutputZeroPoints(),
                                   op.getOutputQuantizationAxis());
}

}  // namespace

//===----------------------------------------------------------------------===//
// UniformQuantizedDotHybridOp
//===----------------------------------------------------------------------===//
//

LogicalResult UniformQuantizedDotHybridOp::verify() {
  UniformQuantizedDotHybridOp op = *this;
  return VerifyScalesAndZeroPoints(op, op.getRhsScales(), op.getRhsZeroPoints(),
                                   op.getRhsQuantizationAxis());
}

//===----------------------------------------------------------------------===//
// UniformQuantizedConvolutionHybridOp
//===----------------------------------------------------------------------===//
//

LogicalResult UniformQuantizedConvolutionHybridOp::verify() {
  UniformQuantizedConvolutionHybridOp op = *this;
  return VerifyScalesAndZeroPoints(op, op.getRhsScales(), op.getRhsZeroPoints(),
                                   op.getRhsQuantizationAxis());
}

//===----------------------------------------------------------------------===//
// UniformQuantizeOp
//===----------------------------------------------------------------------===//
//

LogicalResult UniformQuantizeOp::verify() {
  UniformQuantizeOp op = *this;
  return VerifyScalesAndZeroPoints(op, op.getScales(), op.getZeroPoints(),
                                   op.getQuantizationAxis());
}

//===----------------------------------------------------------------------===//
// UniformRequantizeOp
//===----------------------------------------------------------------------===//
//

LogicalResult UniformRequantizeOp::verify() {
  UniformRequantizeOp op = *this;
  auto verify_input_params = VerifyScalesAndZeroPoints(
      op, op.getInputScales(), op.getInputZeroPoints(),
      op.getInputQuantizationAxis());
  if (failed(verify_input_params)) {
    return failure();
  }
  return VerifyScalesAndZeroPoints(op, op.getOutputScales(),
                                   op.getOutputZeroPoints(),
                                   op.getOutputQuantizationAxis());
}

//===----------------------------------------------------------------------===//
// UniformDequantizeOp
//===----------------------------------------------------------------------===//
//

LogicalResult UniformDequantizeOp::verify() {
  UniformDequantizeOp op = *this;
  return VerifyScalesAndZeroPoints(op, op.getScales(), op.getZeroPoints(),
                                   op.getQuantizationAxis());
}

//===----------------------------------------------------------------------===//
// UniformQuantizedDotOp
//===----------------------------------------------------------------------===//
//

LogicalResult UniformQuantizedDotOp::verify() {
  return VerifyLhsRhsBothUniformQuantizedOp(*this);
}

//===----------------------------------------------------------------------===//
// UniformQuantizedConvolutionOp
//===----------------------------------------------------------------------===//
//

LogicalResult UniformQuantizedConvolutionOp::verify() {
  return VerifyLhsRhsBothUniformQuantizedOp(*this);
}

}  // namespace TF
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.cc.inc"
