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

// This transformation pass converts operations in TensorFlow dialect into
// operations that are legal in the TensorFlow Lite dialect.  Operations that
// can be legalized to TensorFlow Lite dialect with simple replacements are part
// of this pass and other operations that may create extra ops should be part of
// the PrepareTF pass which should be run before this pass.  That way any
// constant folding opportunities from the extra ops can be exploited by the
// constant folding support for the TensorFlow ops.

#include <algorithm>
#include <climits>
#include <complex>
#include <cstdint>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/constant_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual LegalizeTF Pass.
namespace {
#define GEN_PASS_DEF_LEGALIZETFPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

constexpr char kUnidirectionalSequenceLstm[] = "tf.UnidirectionalSequenceLstm";
constexpr char kUnidirectionalSequenceRnn[] = "tf.UnidirectionalSequenceRnn";
constexpr char kTfLiteInputIndices[] = "_tflite_input_indices";

// Legalize operations in functions.
class LegalizeTFPass : public impl::LegalizeTFPassBase<LegalizeTFPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeTFPass)
  LegalizeTFPass() = default;
  explicit LegalizeTFPass(bool run_tfl_runtime_verification,
                          bool preserve_assert_op) {
    this->run_tfl_runtime_verification_ = run_tfl_runtime_verification;
    this->preserve_assert_op_ = preserve_assert_op;
  }

  /// Performs the lowering to TFLite dialect.
  void runOnOperation() override;
};

// Util that casts 'val' to Int32 by adding a cast Op.
Value CreateCastToInt32(Value val, Location loc, PatternRewriter& rewriter) {
  IntegerType new_ele_type = rewriter.getIntegerType(32);
  if (auto shaped_type = val.getType().dyn_cast<RankedTensorType>()) {
    ShapedType new_type =
        RankedTensorType::get(shaped_type.getShape(), new_ele_type);
    return rewriter.createOrFold<TF::CastOp>(loc, new_type, val,
                                             rewriter.getBoolAttr(false));
  }
  return rewriter.createOrFold<TF::CastOp>(
      loc, UnrankedTensorType::get(new_ele_type), val,
      rewriter.getBoolAttr(false));
}

// Utility function to-
// 1. Create a tfl.const op with an int32_t values, from an MLIR Value, if the
// `Value` can be matched to a Constant DenseIntElementsAttr.
// This will make sure the dynamic dimensions are asigned to be `-1`
// 2. In the default case, cast the `Value` to an int32_t.
Value CreateInt32ConstOrCast(Value val, Location loc,
                             PatternRewriter& rewriter) {
  if (val.getType().cast<ShapedType>().hasStaticShape()) {
    DenseElementsAttr shape_value_attr;
    if (matchPattern(val, m_Constant(&shape_value_attr))) {
      SmallVector<int32_t, 4> new_shape_array_i32;
      auto shape_value_array = shape_value_attr.getValues<llvm::APInt>();
      for (int32_t idx = 0; idx < shape_value_array.size(); ++idx) {
        auto size = shape_value_array[idx].getSExtValue();
        new_shape_array_i32.push_back(
            ShapedType::isDynamic(size) ? -1 : static_cast<int32_t>(size));
      }
      return rewriter.create<arith::ConstantOp>(
          loc, DenseIntElementsAttr::get(
                   RankedTensorType::get(new_shape_array_i32.size(),
                                         rewriter.getIntegerType(32)),
                   new_shape_array_i32));
    }
  }

  return CreateCastToInt32(val, loc, rewriter);
}

// Get shape of an operand or result, support both dynamic and static shape.
Value GetShape(Value input, Location loc, PatternRewriter& rewriter) {
  auto shaped_type = input.getType().cast<ShapedType>();
  if (shaped_type.hasStaticShape()) {
    auto static_shape = shaped_type.getShape();
    auto static_shape_type =
        RankedTensorType::get(static_shape.size(), rewriter.getIntegerType(64));
    auto static_shape_attr =
        mlir::DenseIntElementsAttr::get(static_shape_type, static_shape);
    return rewriter.create<TF::ConstOp>(loc, static_shape_attr).getOutput();
  }

  // If the shape is not static, create a new ShapeOp.
  BoolAttr false_attr = rewriter.getBoolAttr(false);
  return rewriter
      .create<TF::ShapeOp>(loc, input,
                           /*use_32bit=*/false_attr)
      .getOutput();
}

mlir::TFL::MirrorPaddingType GetTFLMirrorPaddingFromString(
    mlir::StringAttr padding) {
  return llvm::StringSwitch<mlir::TFL::MirrorPaddingType>(padding.getValue())
      .Case("REFLECT", mlir::TFL::MirrorPaddingType::REFLECT)
      .Case("SYMMETRIC", mlir::TFL::MirrorPaddingType::SYMMETRIC);
}

#include "tensorflow/compiler/mlir/lite/transforms/generated_legalize_tf.inc"

#define DECL_CONVERT_OP(tf_op)                                               \
  struct ConvertTF##tf_op##Op : public RewritePattern {                      \
    explicit ConvertTF##tf_op##Op(MLIRContext* context)                      \
        : RewritePattern(TF::tf_op##Op::getOperationName(), 1, context) {}   \
    LogicalResult matchAndRewrite(Operation* op,                             \
                                  PatternRewriter& rewriter) const override; \
  }

// TODO(antiagainst): Define this pattern in a table-driven manner once variadic
// operands are properly supported in declarative rewrite rule specification.

DECL_CONVERT_OP(Assert);
DECL_CONVERT_OP(ConcatV2);
DECL_CONVERT_OP(BatchMatMul);
DECL_CONVERT_OP(BatchMatMulV2);
DECL_CONVERT_OP(BatchMatMulV3);
DECL_CONVERT_OP(MatMul);
DECL_CONVERT_OP(MatrixDiagV2);
DECL_CONVERT_OP(MatrixDiagV3);
DECL_CONVERT_OP(Pack);
DECL_CONVERT_OP(Split);
DECL_CONVERT_OP(SplitV);
DECL_CONVERT_OP(Unpack);
DECL_CONVERT_OP(Conv3D);
DECL_CONVERT_OP(Conv3DBackpropInputV2);

#undef DECL_CONVERT_OP

// Converts any IntegerAttr to an IntegerAttr of an i32 type.
// The value won't change in the new attribute, but if the value is out of
// the bound of i32, the function returns a failure.
LogicalResult ConvertToI32Attr(IntegerAttr attr, IntegerAttr* attr_i32) {
  if (attr.getType().isInteger(/*width=*/32)) {
    *attr_i32 = attr;
    return success();
  }

  int64_t value = attr.getInt();
  if (value > std::numeric_limits<int>::max() ||
      value < std::numeric_limits<int>::min()) {
    return failure();
  }

  *attr_i32 = IntegerAttr::get(
      IntegerType::get(attr.getContext(), /*width=*/32), value);
  return success();
}

LogicalResult ConvertTFConcatV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_concat_op = cast<TF::ConcatV2Op>(op);

  auto values = tf_concat_op.getValues();
  auto output_type = tf_concat_op.getOutput().getType();
  // Extract axis attribute from constant axis tensor
  ElementsAttr axis;
  if (!matchPattern(tf_concat_op.getAxis(), m_Constant(&axis)))
    return failure();
  IntegerAttr axis_int = ExtractSingleElementAsInteger(axis);

  // "axis" operand could be a i64 tensor. Resolve it here.
  IntegerAttr axis_i32;
  if (failed(ConvertToI32Attr(axis_int, &axis_i32))) return failure();

  StringAttr fused_activation_function =
      StringAttr::get(rewriter.getContext(), "NONE");
  rewriter.replaceOpWithNewOp<ConcatenationOp>(
      op, output_type, values, axis_i32, fused_activation_function);
  return success();
}

template <typename BatchMatMulOpType>
bool ConvertTFBatchMatMulOp2TFLFullyConnectedOp(Operation* bmm_op,
                                                PatternRewriter& rewriter) {
  // If `value` is produced by tf.Dequantize op, then return the Dequantize op's
  // input. Otherwise return `value`.
  auto get_real_input_value = [](Value value) -> Value {
    Operation* defining_op = value.getDefiningOp();
    if (auto dequantize = dyn_cast_or_null<TF::DequantizeOp>(defining_op)) {
      return dequantize.getInput();
    } else if (auto dequantize =
                   dyn_cast_or_null<TFL::DequantizeOp>(defining_op)) {
      return dequantize.getInput();
    } else {
      return value;
    }
  };

  // Returns true if the TF::BatchMatMul operation can be converted to
  // tfl.fully_connected.
  auto can_convert_to_fully_connected =
      [&](BatchMatMulOpType& batch_matmul_op) {
        Value input_rhs = get_real_input_value(batch_matmul_op.getY());

        DenseElementsAttr constant;
        if (!matchPattern(input_rhs, m_Constant(&constant))) {
          return false;
        }

        // The rhs matrix must be 2D for fully connected op.
        return (constant.getType().getRank() == 2);
      };

  auto op = cast<BatchMatMulOpType>(bmm_op);

  // Create a tfl.transpose op that performs ZX transpose on `input`.
  auto create_z_x_transpose_op = [&](Value input) -> Value {
    RankedTensorType input_type = input.getType().cast<RankedTensorType>();
    const int input_rank = input_type.getRank();

    // Create a 1D I32 tensor for representing the dimension permutation.
    auto permuation_tensor_type =
        RankedTensorType::get({input_rank}, rewriter.getIntegerType(32));
    llvm::SmallVector<Attribute, 4> permute;
    permute.reserve(input_rank);
    // First create an identity permutation tensor.
    for (int i = 0; i < input_rank; i++) {
      permute.push_back(rewriter.getI32IntegerAttr(i));
    }
    // Swaps the last two dimension since the last two dimension will be mapped
    // to X and Z dimension.
    std::iter_swap(permute.begin() + input_rank - 1,
                   permute.begin() + input_rank - 2);
    auto permutation_tensor_op = rewriter.create<arith::ConstantOp>(
        op->getLoc(), permuation_tensor_type,
        DenseElementsAttr::get(permuation_tensor_type, permute));

    auto input_shape = input_type.getShape();
    llvm::SmallVector<int64_t, 4> permuted_shape(input_shape.begin(),
                                                 input_shape.end());
    // Swaps z dimension and x dimension to get permuted shape.
    std::iter_swap(permuted_shape.begin() + input_rank - 1,
                   permuted_shape.begin() + input_rank - 2);
    return rewriter.create<TFL::TransposeOp>(
        op->getLoc(),
        RankedTensorType::get(permuted_shape, input_type.getElementType()),
        input, permutation_tensor_op.getResult());
  };

  if (!can_convert_to_fully_connected(op)) {
    return false;
  }

  Value input_lhs = get_real_input_value(op.getX());
  Value input_rhs = get_real_input_value(op.getY());

  Value legalized_lhs =
      op.getAdjX() ? create_z_x_transpose_op(input_lhs) : input_lhs;

  // The rhs need to be transposed if adj_y == false AND this matmul will be
  // legalized to tfl.fully_connected
  Value legalized_rhs =
      !op.getAdjY() ? create_z_x_transpose_op(input_rhs) : input_rhs;

  Type output_type = op.getResult().getType();
  auto no_input = rewriter.create<TFL::NoValueOp>(
      op->getLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());
  auto fc_op = rewriter.create<TFL::FullyConnectedOp>(
      op->getLoc(), ArrayRef<Type>{output_type},
      /*input=*/legalized_lhs, /*filter=*/legalized_rhs, /*bias=*/no_input,
      /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
      /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
      /*keep_num_dims=*/rewriter.getBoolAttr(true),
      /*asymmetric_quantize_inputs=*/mlir::BoolAttr());
  rewriter.replaceOp(op, {fc_op.getResult(0)});

  return true;
}

LogicalResult ConvertTFBatchMatMulOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (ConvertTFBatchMatMulOp2TFLFullyConnectedOp<TF::BatchMatMulOp>(op,
                                                                    rewriter))
    return success();
  return failure();
}

LogicalResult ConvertTFBatchMatMulV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (ConvertTFBatchMatMulOp2TFLFullyConnectedOp<TF::BatchMatMulV2Op>(op,
                                                                      rewriter))
    return success();
  return failure();
}

LogicalResult ConvertTFBatchMatMulV3Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (ConvertTFBatchMatMulOp2TFLFullyConnectedOp<TF::BatchMatMulV3Op>(op,
                                                                      rewriter))
    return success();
  return failure();
}

LogicalResult ConvertTFMatMulOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_matmul_op = cast<TF::MatMulOp>(op);
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  auto transpose = [&](Value input) -> std::pair<LogicalResult, Value> {
    RankedTensorType type =
        input.getType().dyn_cast_or_null<RankedTensorType>();
    if (!type || type.getRank() != 2) return {failure(), nullptr};

    auto permute_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI32Type()), {1, 0});
    auto permute = rewriter.create<arith::ConstantOp>(
        op->getLoc(), permute_attr.getType(), permute_attr);
    llvm::SmallVector<int64_t, 2> new_shape{type.getShape()[1],
                                            type.getShape()[0]};
    auto output = rewriter.create<TFL::TransposeOp>(
        op->getLoc(), RankedTensorType::get(new_shape, type.getElementType()),
        input, permute);
    return {success(), output};
  };

  // TODO(jpienaar): Remove once handled via dailect conversion.
  if (tf_matmul_op.getTransposeA()) {
    LogicalResult result = success();
    std::tie(result, lhs) = transpose(lhs);
    if (failed(result)) return failure();
  }
  if (!tf_matmul_op.getTransposeB()) {
    LogicalResult result = success();
    std::tie(result, rhs) = transpose(rhs);
    if (failed(result)) return failure();
  }

  Type output_type = tf_matmul_op.getResult().getType();
  auto no_input = rewriter.create<TFL::NoValueOp>(
      op->getLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());
  auto fc_op = rewriter.create<FullyConnectedOp>(
      op->getLoc(), ArrayRef<Type>{output_type},
      /*input=*/lhs, /*filter=*/rhs, /*bias=*/no_input,
      /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
      /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
      /*keep_num_dims=*/rewriter.getBoolAttr(false),
      /*asymmetric_quantize_inputs=*/mlir::BoolAttr());
  rewriter.replaceOp(op, {fc_op.getResult(0)});
  return success();
}

LogicalResult ConvertTFPackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_pack_op = cast<TF::PackOp>(op);

  SmallVector<Value, 4> values(tf_pack_op.getValues());
  auto output_type = tf_pack_op.getOutput().getType();
  auto values_count = rewriter.getI32IntegerAttr(tf_pack_op.getN());
  // Axis can be negative.
  auto axis = rewriter.getI32IntegerAttr(tf_pack_op.getAxis());

  rewriter.replaceOpWithNewOp<PackOp>(op, output_type, values, values_count,
                                      axis);
  return success();
}

LogicalResult ConvertTFSplitOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_split_op = cast<TF::SplitOp>(op);

  // Number of splits cannot be negative.
  auto num_split = rewriter.getI32IntegerAttr(tf_split_op.getNumSplit());

  rewriter.replaceOpWithNewOp<TFL::SplitOp>(
      op, tf_split_op.getOutput().getTypes(), tf_split_op.getSplitDim(),
      tf_split_op.getValue(), num_split);
  return success();
}

LogicalResult ConvertTFSplitVOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_splitv_op = cast<TF::SplitVOp>(op);

  // Number of splits cannot be negative.
  auto num_split = rewriter.getI32IntegerAttr(tf_splitv_op.getNumSplit());

  rewriter.replaceOpWithNewOp<TFL::SplitVOp>(
      op, tf_splitv_op.getOutput().getTypes(), tf_splitv_op.getValue(),
      tf_splitv_op.getSizeSplits(), tf_splitv_op.getSplitDim(), num_split);
  return success();
}

LogicalResult ConvertTFUnpackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_unpack_op = cast<TF::UnpackOp>(op);

  auto input = tf_unpack_op.getValue();
  auto num = rewriter.getI32IntegerAttr(tf_unpack_op.getNum());
  // Axis can be negative.
  auto axis = rewriter.getI32IntegerAttr(tf_unpack_op.getAxis());

  rewriter.replaceOpWithNewOp<UnpackOp>(op, tf_unpack_op.getOutput().getTypes(),
                                        input, num, axis);
  return success();
}

LogicalResult ConvertTFConv3DOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (!TFDataFormatIsNDHWC(op)) return failure();

  auto tf_op = cast<TF::Conv3DOp>(op);

  IntegerAttr stride_depth, stride_height, stride_width;
  if (!TFIntListIs1XYZ1(op, "strides", &stride_depth, &stride_height,
                        &stride_width))
    return failure();

  IntegerAttr dilation_depth_factor, dilation_height_factor,
      dilation_width_factor;
  if (!TFIntListIs1XYZ1(op, "dilations", &dilation_depth_factor,
                        &dilation_height_factor, &dilation_width_factor)) {
    // If the 'dilations' attribute is missing, we use the default value (1)
    // for all dilation depth, height and width factor.
    dilation_depth_factor = rewriter.getI32IntegerAttr(1);
    dilation_height_factor = rewriter.getI32IntegerAttr(1);
    dilation_width_factor = rewriter.getI32IntegerAttr(1);
  }

  StringAttr padding;
  if (!TFPaddingIsSameOrValid(op, &padding)) return failure();

  // TensorFlow Conv3D has no bias, optimization patterns will fuse Conv3D
  // with other ops can fill the bias.
  Value none = rewriter.create<TFL::NoValueOp>(
      op->getLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());

  rewriter.replaceOpWithNewOp<TFL::Conv3DOp>(
      op, tf_op.getType(), tf_op.getInput(), tf_op.getFilter(),
      /*bias=*/none, dilation_depth_factor, dilation_height_factor,
      dilation_width_factor,
      /*fused_activation_function=*/rewriter.getStringAttr("NONE"), padding,
      stride_depth, stride_height, stride_width);

  return success();
}

LogicalResult ConvertTFConv3DBackpropInputV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (!TFDataFormatIsNDHWC(op)) return failure();

  auto tf_op = cast<TF::Conv3DBackpropInputV2Op>(op);

  IntegerAttr stride_depth, stride_height, stride_width;
  if (!TFIntListIs1XYZ1(op, "strides", &stride_depth, &stride_height,
                        &stride_width))
    return failure();

  IntegerAttr dilation_depth_factor, dilation_height_factor,
      dilation_width_factor;
  if (!TFIntListIs1XYZ1(op, "dilations", &dilation_depth_factor,
                        &dilation_height_factor, &dilation_width_factor)) {
    // If the 'dilations' attribute is missing, we use the default value (1)
    // for all dilation depth, height and width factor.
    dilation_depth_factor = rewriter.getI32IntegerAttr(1);
    dilation_height_factor = rewriter.getI32IntegerAttr(1);
    dilation_width_factor = rewriter.getI32IntegerAttr(1);
  }

  StringAttr padding;
  if (!TFPaddingIsSameOrValid(op, &padding)) return failure();

  // TensorFlow Conv3D has no bias, optimization patterns will fuse Conv3D
  // with other ops can fill the bias.
  Value none = rewriter.create<TFL::NoValueOp>(
      op->getLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());

  Value output_shape =
      CreateCastToInt32(tf_op.getInputSizes(), op->getLoc(), rewriter);

  rewriter.replaceOpWithNewOp<TFL::Conv3DTransposeOp>(
      op, tf_op.getType(), output_shape, tf_op.getFilter(),
      tf_op.getOutBackprop(),
      /*bias=*/none, dilation_depth_factor, dilation_height_factor,
      dilation_width_factor,
      /*fused_activation_function=*/rewriter.getStringAttr("NONE"), padding,
      stride_depth, stride_height, stride_width);

  return success();
}

// MatrixDiagV3 is MatrixDiagV2 with an alignment attribute. This attribute
// only has effects when processing multiple diagonals. Since TFLite converts
// MatrixDiagV{2,3} to MatrixDiag, which only takes single-diagonal inputs, we
// can safely ignore this V3 attribute.
// We can't pass `rewriter` by reference because clang-tidy will want it to be
// constant (`const PatternRewriter& rewriter`). If we do that, we won't be able
// to call `rewriter::replaceOpWihNewOp`, which is not a const member function.
template <typename MatrixDiagV2OrV3Op>
bool ConvertTFMatrixDiagV2orV3(Operation* op, PatternRewriter* rewriter) {
  auto tf_matrix_diag_v2_or_v3_op = cast<MatrixDiagV2OrV3Op>(op);

  if (tf_matrix_diag_v2_or_v3_op.getNumOperands() != 5) return false;

  auto input = tf_matrix_diag_v2_or_v3_op.getDiagonal();
  auto output_type = tf_matrix_diag_v2_or_v3_op.getOutput().getType();

  // Extract k constant tensor and check value = 0.
  ElementsAttr k;
  if (!matchPattern(tf_matrix_diag_v2_or_v3_op.getK(), m_Constant(&k)))
    return false;
  if (ExtractSingleElementAsInteger(k).getInt() != 0) return false;

  // Extract num_rows constant tensor and check value = -1.
  ElementsAttr num_rows;
  if (!matchPattern(tf_matrix_diag_v2_or_v3_op.getNumRows(),
                    m_Constant(&num_rows)))
    return false;
  if (ExtractSingleElementAsInteger(num_rows).getInt() != -1) return false;

  // Extract num_cols constant tensor and check value = -1.
  ElementsAttr num_cols;
  if (!matchPattern(tf_matrix_diag_v2_or_v3_op.getNumCols(),
                    m_Constant(&num_cols)))
    return false;
  if (ExtractSingleElementAsInteger(num_cols).getInt() != -1) return false;

  // Verify padding_value is a tensor with all 0s.
  mlir::Value padding_value = tf_matrix_diag_v2_or_v3_op.getPaddingValue();
  mlir::Type element_type =
      padding_value.getType().cast<ShapedType>().getElementType();
  if (element_type.isa<FloatType>()) {
    DenseFPElementsAttr padding_attr;
    if (!matchPattern(padding_value, m_Constant(&padding_attr)) ||
        !padding_attr.isSplat() ||
        !padding_attr.getSplatValue<APFloat>().isZero()) {
      return false;
    }
  } else if (element_type.isa<IntegerType>()) {
    DenseIntElementsAttr padding_attr;
    if (!matchPattern(padding_value, m_Constant(&padding_attr)) ||
        !padding_attr.isSplat() ||
        !padding_attr.getSplatValue<APInt>().isZero()) {
      return false;
    }
  } else {
    // If the padding value is neither float nor int, conservatively assume it
    // contains nonzeros.
    return false;
  }

  rewriter->replaceOpWithNewOp<MatrixDiagOp>(op, output_type, input);
  return true;
}

LogicalResult ConvertTFMatrixDiagV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (ConvertTFMatrixDiagV2orV3<TF::MatrixDiagV2Op>(op, &rewriter))
    return success();
  return failure();
}

LogicalResult ConvertTFMatrixDiagV3Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (ConvertTFMatrixDiagV2orV3<TF::MatrixDiagV3Op>(op, &rewriter))
    return success();
  return failure();
}

// TF Lite doesn't support Assert, we just drop the assert from the graph.
LogicalResult ConvertTFAssertOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  rewriter.eraseOp(op);
  return success();
}

// Legalize unidirectional sequence lstm.
struct LegalizeUnidirectionalSequenceLstm : public RewritePattern {
  explicit LegalizeUnidirectionalSequenceLstm(MLIRContext* context)
      : RewritePattern(kUnidirectionalSequenceLstm, 1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    auto tflite_indices_attr =
        op->getAttrOfType<ArrayAttr>(kTfLiteInputIndices);
    if (!tflite_indices_attr) return failure();

    SmallVector<int64_t, 20> tflite_indices;
    for (auto index_attr : tflite_indices_attr.getValue()) {
      IntegerAttr index = index_attr.cast<IntegerAttr>();
      tflite_indices.push_back(index.getInt());
    }

    // Optional input placeholder.
    Value none = rewriter.create<TFL::NoValueOp>(
        op->getLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());

    // Populate inputs.
    // UnidirectionalSequenceLstm is expected to have 24 inputs.
    SmallVector<Value, 24> inputs;
    int count = 0;
    int total_ophint_converted_inputs = tflite_indices.size();
    for (int i = 0; i < 24; ++i) {
      if (count < total_ophint_converted_inputs && tflite_indices[count] == i) {
        // specified input.
        inputs.push_back(op->getOperand(i));
        count++;
      } else {
        // Non specified input.
        inputs.push_back(none);
      }
    }

    // Populate outputs.
    // UnidirectionalSequenceLstm should only have 1 output, and that is the
    // original ophint converted node's 3rd output.
    SmallVector<Type, 4> result_types;
    result_types.push_back(op->getOpResult(2).getType());

    // Populate attributes.
    SmallVector<NamedAttribute, 4> attributes;
    // Activation will always be tanh.
    attributes.push_back(rewriter.getNamedAttr("fused_activation_function",
                                               rewriter.getStringAttr("TANH")));
    // cell_clip.
    attributes.push_back(
        rewriter.getNamedAttr("cell_clip", rewriter.getF32FloatAttr(0.0)));
    // proj_clip.
    attributes.push_back(
        rewriter.getNamedAttr("proj_clip", rewriter.getF32FloatAttr(0.0)));
    // will always be time_majored.
    attributes.push_back(
        rewriter.getNamedAttr("time_major", rewriter.getBoolAttr(true)));

    Value lstm_result = rewriter.create<TFL::UnidirectionalSequenceLSTMOp>(
        op->getLoc(), result_types, inputs, attributes);

    // Rewire the output.
    rewriter.replaceOp(op, {nullptr, nullptr, lstm_result});
    return success();
  }
};

// Legalize unidirectional seqeucen rnn.
struct LegalizeUnidirectionalSequenceRnn : public RewritePattern {
  explicit LegalizeUnidirectionalSequenceRnn(MLIRContext* context)
      : RewritePattern(kUnidirectionalSequenceRnn, 1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    auto tflite_indices_attr =
        op->getAttrOfType<ArrayAttr>(kTfLiteInputIndices);
    if (!tflite_indices_attr) return failure();

    if (op->getNumOperands() != 5) {
      op->emitError()
          << "We're expecting 5 inputs for UnidirectionalSequenceRNN, only "
          << op->getNumOperands() << " provided";
      return failure();
    }

    if (op->getNumResults() != 2) {
      op->emitError()
          << "We're expecting 2 inputs for UnidirectionalSequenceRNN, only "
          << op->getNumResults() << " found";
      return failure();
    }

    // Populate inputs.
    // UnidirectionalSequenceRnn is expected to have 5 inputs, and none of them
    // are optional inputs.
    SmallVector<Value, 5> inputs;
    for (int i = 0; i < 5; ++i) {
      inputs.push_back(op->getOperand(i));
    }

    // Populate outputs.
    // UnidirectionalSequenceRnn should only have 1 output, and that is the
    // original ophint converted node's 2nd output.
    SmallVector<Type, 4> result_types;
    result_types.push_back(op->getOpResult(1).getType());

    // Populate attributes.
    SmallVector<NamedAttribute, 2> attributes;
    // Activation will always be tanh.
    attributes.push_back(rewriter.getNamedAttr("fused_activation_function",
                                               rewriter.getStringAttr("TANH")));

    // will always be time_majored.
    attributes.push_back(
        rewriter.getNamedAttr("time_major", rewriter.getBoolAttr(true)));

    Value rnn_result = rewriter.create<TFL::UnidirectionalSequenceRNNOp>(
        op->getLoc(), result_types, inputs, attributes);

    // Rewire the output.
    rewriter.replaceOp(op, {nullptr, rnn_result});

    return success();
  }
};

// Put two TFL BroadcastTo ops in front of the given TF binary broadcast op to
// to make binary broadcast-able op conversion always successful and does not
// require flex delegate.
template <typename SourceOp>
class ApplyExplicitBroadcasting : public OpRewritePattern<SourceOp> {
 public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult rewriteOpWithDynamicInput(Operation* op,
                                          PatternRewriter& rewriter) const {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto out = op->getResult(0);

    // Calculates symbolic broadcast shape that is only used in types.
    SmallVector<int64_t, 4> symbolic_broadcast_shape;
    // Matches fail when lhs or rhs is unranked tensor.
    // TODO(b/176202543): Support unranked tensor.
    if (!lhs.getType().cast<ShapedType>().hasRank() ||
        !rhs.getType().cast<ShapedType>().hasRank()) {
      return failure();
    }
    if (!OpTrait::util::getBroadcastedShape(
            lhs.getType().cast<ShapedType>().getShape(),
            rhs.getType().cast<ShapedType>().getShape(),
            symbolic_broadcast_shape)) {
      return failure();
    }

    // Calculates the broadcast shape using BroadcastArgs op.
    Value lhs_shape = GetShape(lhs, op->getLoc(), rewriter);
    Value rhs_shape = GetShape(rhs, op->getLoc(), rewriter);
    auto broadcast_shape =
        rewriter
            .create<TF::BroadcastArgsOp>(
                op->getLoc(),
                RankedTensorType::get(symbolic_broadcast_shape.size(),
                                      rewriter.getIntegerType(64)),
                lhs_shape, rhs_shape)
            .getR0();

    // Broadcasts inputs using BroadcastTo op.
    auto broadcast_type = RankedTensorType::get(
        symbolic_broadcast_shape, getElementTypeOrSelf(lhs.getType()));
    auto broadcasted_lhs =
        rewriter
            .create<TF::BroadcastToOp>(op->getLoc(), broadcast_type, lhs,
                                       broadcast_shape)
            .getOutput();
    auto broadcasted_rhs =
        rewriter
            .create<TF::BroadcastToOp>(op->getLoc(), broadcast_type, rhs,
                                       broadcast_shape)
            .getOutput();

    // Recreate an op with the above BroadcastTo op results.
    RankedTensorType result_type = RankedTensorType::get(
        symbolic_broadcast_shape, getElementTypeOrSelf(out.getType()));
    rewriter.replaceOpWithNewOp<SourceOp>(op, result_type, broadcasted_lhs,
                                          broadcasted_rhs);
    return success();
  }

  LogicalResult matchAndRewrite(SourceOp src_op,
                                PatternRewriter& rewriter) const override {
    Operation* op = static_cast<Operation*>(src_op);
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);

    if (!lhs.getType().cast<ShapedType>().hasStaticShape() ||
        !rhs.getType().cast<ShapedType>().hasStaticShape()) {
      return rewriteOpWithDynamicInput(op, rewriter);
    }

    auto lhs_shape = lhs.getType().cast<ShapedType>().getShape();
    auto rhs_shape = rhs.getType().cast<ShapedType>().getShape();

    if (lhs_shape == rhs_shape) {
      return failure();
    }

    // Calculate the broadcasted shape.
    SmallVector<int64_t, 4> result_shape;
    if (!OpTrait::util::getBroadcastedShape(lhs_shape, rhs_shape,
                                            result_shape)) {
      return failure();
    }

    RankedTensorType result_type = RankedTensorType::get(
        result_shape, getElementTypeOrSelf(op->getResult(0).getType()));

    // Create a const op, that stores the above broadcasted shape.
    auto new_shape_attr = mlir::DenseIntElementsAttr::get(
        RankedTensorType::get(result_shape.size(), rewriter.getIntegerType(64)),
        result_shape);
    auto new_shape = rewriter.create<TF::ConstOp>(op->getLoc(), new_shape_attr);

    // Apply BroadcastTo ops to each input.
    auto broadcast_type = RankedTensorType::get(
        result_shape, getElementTypeOrSelf(lhs.getType()));

    if (result_type.getShape() != lhs_shape) {
      lhs = rewriter
                .create<TF::BroadcastToOp>(op->getLoc(), broadcast_type, lhs,
                                           new_shape)
                .getOutput();
    }
    if (result_type.getShape() != rhs_shape) {
      rhs = rewriter
                .create<TF::BroadcastToOp>(op->getLoc(), broadcast_type, rhs,
                                           new_shape)
                .getOutput();
    }

    // Recreate an op with the above Broadcast op results.
    rewriter.replaceOpWithNewOp<SourceOp>(op, result_type, lhs, rhs);
    return success();
  }
};

// This specialization is for TF SelectV2 op. SelectV2 op have three inputs and
// they should have broadcastable shapes.
template <>
class ApplyExplicitBroadcasting<TF::SelectV2Op>
    : public OpRewritePattern<TF::SelectV2Op> {
 public:
  using OpRewritePattern<TF::SelectV2Op>::OpRewritePattern;

  LogicalResult rewriteOpWithDynamicInput(Operation* op,
                                          PatternRewriter& rewriter) const {
    auto cond = op->getOperand(0);
    auto lhs = op->getOperand(1);
    auto rhs = op->getOperand(2);
    auto out = op->getResult(0);

    // Matches fail when lhs|rhs|cond is unranked tensor.
    // TODO(b/176202543): Support unranked tensor.
    if (!lhs.getType().cast<ShapedType>().hasRank() ||
        !rhs.getType().cast<ShapedType>().hasRank() ||
        !cond.getType().cast<ShapedType>().hasRank()) {
      return failure();
    }

    // Calculates symbolic broadcast shape that is only used in types.
    SmallVector<int64_t, 4> symbolic_broadcast_lhs_rhs_shape;
    if (!OpTrait::util::getBroadcastedShape(
            lhs.getType().cast<ShapedType>().getShape(),
            rhs.getType().cast<ShapedType>().getShape(),
            symbolic_broadcast_lhs_rhs_shape)) {
      return failure();
    }
    SmallVector<int64_t, 4> symbolic_broadcast_shape;
    if (!OpTrait::util::getBroadcastedShape(
            cond.getType().cast<ShapedType>().getShape(),
            symbolic_broadcast_lhs_rhs_shape, symbolic_broadcast_shape)) {
      return failure();
    }

    // Calculates the broadcast shape using BroadcastArgs op.
    Value cond_shape = GetShape(cond, op->getLoc(), rewriter);
    Value lhs_shape = GetShape(lhs, op->getLoc(), rewriter);
    Value rhs_shape = GetShape(rhs, op->getLoc(), rewriter);
    auto broadcast_shape_value =
        rewriter
            .create<TF::BroadcastArgsOp>(op->getLoc(), lhs_shape.getType(),
                                         lhs_shape, rhs_shape)
            .getR0();
    broadcast_shape_value =
        rewriter
            .create<TF::BroadcastArgsOp>(op->getLoc(), lhs_shape.getType(),
                                         broadcast_shape_value, cond_shape)
            .getR0();

    // Broadcasting inputs using BroadcastTo op.
    auto broadcast_type = RankedTensorType::get(
        symbolic_broadcast_shape, getElementTypeOrSelf(out.getType()));
    auto broadcasted_cond =
        rewriter
            .create<TF::BroadcastToOp>(
                op->getLoc(),
                RankedTensorType::get(symbolic_broadcast_shape,
                                      rewriter.getIntegerType(1)),
                cond, broadcast_shape_value)
            .getOutput();
    auto broadcasted_lhs =
        rewriter
            .create<TF::BroadcastToOp>(op->getLoc(), broadcast_type, lhs,
                                       broadcast_shape_value)
            .getOutput();
    auto broadcasted_rhs =
        rewriter
            .create<TF::BroadcastToOp>(op->getLoc(), broadcast_type, rhs,
                                       broadcast_shape_value)
            .getOutput();

    // Recreate an op with the above BroadcastTo op results.
    rewriter.replaceOpWithNewOp<TF::SelectV2Op>(
        op, broadcast_type, broadcasted_cond, broadcasted_lhs, broadcasted_rhs);
    return success();
  }

  LogicalResult matchAndRewrite(TF::SelectV2Op src_op,
                                PatternRewriter& rewriter) const override {
    Operation* op = static_cast<Operation*>(src_op);
    auto cond = op->getOperand(0);
    auto lhs = op->getOperand(1);
    auto rhs = op->getOperand(2);

    // Should have static shapes to calculate the broadcasted shape.
    if (!lhs.getType().cast<ShapedType>().hasStaticShape() ||
        !rhs.getType().cast<ShapedType>().hasStaticShape() ||
        !cond.getType().cast<ShapedType>().hasStaticShape()) {
      return rewriteOpWithDynamicInput(op, rewriter);
    }

    auto lhs_shape = lhs.getType().cast<ShapedType>().getShape();
    auto rhs_shape = rhs.getType().cast<ShapedType>().getShape();
    auto cond_shape = cond.getType().cast<ShapedType>().getShape();

    if (lhs_shape == rhs_shape && cond_shape == lhs_shape) {
      return failure();
    }

    // Calculate the broadcasted shape.
    SmallVector<int64_t, 4> broadcasted_shape;
    if (!OpTrait::util::getBroadcastedShape(lhs_shape, rhs_shape,
                                            broadcasted_shape)) {
      return failure();
    }

    SmallVector<int64_t, 4> result_shape;
    if (!OpTrait::util::getBroadcastedShape(broadcasted_shape, cond_shape,
                                            result_shape)) {
      return failure();
    }

    // Create a const op, that stores the above broadcasted shape.
    auto shape_type =
        RankedTensorType::get(result_shape.size(), rewriter.getIntegerType(64));
    auto new_shape_attr =
        mlir::DenseIntElementsAttr::get(shape_type, result_shape);
    auto new_shape = rewriter.create<TF::ConstOp>(op->getLoc(), new_shape_attr);

    // Apply BroadcastTo ops to each input.
    auto cond_result_type =
        RankedTensorType::get(result_shape, rewriter.getIntegerType(1));
    auto result_type = RankedTensorType::get(
        result_shape, getElementTypeOrSelf(lhs.getType()));

    if (result_shape != cond_shape) {
      cond = rewriter
                 .create<TF::BroadcastToOp>(op->getLoc(), cond_result_type,
                                            cond, new_shape)
                 .getOutput();
    }
    if (result_shape != lhs_shape) {
      lhs = rewriter
                .create<TF::BroadcastToOp>(op->getLoc(), result_type, lhs,
                                           new_shape)
                .getOutput();
    }
    if (result_shape != rhs_shape) {
      rhs = rewriter
                .create<TF::BroadcastToOp>(op->getLoc(), result_type, rhs,
                                           new_shape)
                .getOutput();
    }

    // Recreate an op with the above Broadcast op results.
    rewriter.replaceOpWithNewOp<TF::SelectV2Op>(op, result_type, cond, lhs,
                                                rhs);
    return success();
  }
};

void addPatterns(MLIRContext* context, RewritePatternSet& patterns,
                 bool preserve_assert_op) {
  // Add TF->TF lowering patterns.
  TF::PopulateLoweringTFPatterns(context, &patterns);

  // Add the generated patterns to the list.
  populateWithGenerated(patterns);
  patterns.add<ConvertTFConcatV2Op, ConvertTFBatchMatMulOp,
               ConvertTFBatchMatMulV2Op, ConvertTFBatchMatMulV3Op,
               ConvertTFMatMulOp, ConvertTFMatrixDiagV2Op,
               ConvertTFMatrixDiagV3Op, ConvertTFPackOp, ConvertTFSplitOp,
               ConvertTFSplitVOp, ConvertTFUnpackOp, ConvertTFConv3DOp,
               ConvertTFConv3DBackpropInputV2Op>(context);
  if (!preserve_assert_op) patterns.add<ConvertTFAssertOp>(context);

  // Ophint python converter converted tf node pattern.
  patterns.add<LegalizeUnidirectionalSequenceLstm,
               LegalizeUnidirectionalSequenceRnn>(context);
}

bool applyPatterns(func::FuncOp func, ConversionTarget& target,
                   FrozenRewritePatternSet& frozenPatterns) {
  // Keep trying to convert.
  // TODO(karimnosseir): This is similar to what apply greedy patterns does.
  // Look if there is a function that tries until it converge.
  // Currently unit-test doesn't do multiple tries, so we need this.
  const int max_iterations = 15;
  for (int i = 0; i < max_iterations; ++i) {
    if (failed(applyPartialConversion(func, target, frozenPatterns))) {
      return false;
    }
  }
  return true;
}

void LegalizeTFPass::runOnOperation() {
  auto* context = &getContext();
  auto func = getOperation();

  ConversionTarget target(*context);
  // It is legal to have TF ops in the graph still which can be
  // used later or in the case of SELECT where we allow TF ops in the final
  // graph.
  target.addLegalOp<mlir::arith::ConstantOp>();
  target.addLegalOp<mlir::func::ConstantOp>();
  target.addLegalOp<TFL::NoValueOp>();
  target.addLegalOp<ConstOp>();
  target.addLegalOp<DequantizeOp>();
  target.addLegalOp<QConstOp>();
  if (run_tfl_runtime_verification_) {
    target.addDynamicallyLegalDialect<TensorFlowLiteDialect>([](Operation* op) {
      auto tfl_op = dyn_cast_or_null<TflRuntimeVerifyOpInterface>(op);
      if (!tfl_op) return false;
      return succeeded(tfl_op.VerifyTflRuntimeConstraints(
          op, /*emit_error_on_verify_fail=*/false));
    });
  } else {
    target.addLegalDialect<TensorFlowLiteDialect>();
  }

  RewritePatternSet stage1Patterns(&getContext());

  addPatterns(context, stage1Patterns, this->preserve_assert_op_);

  FrozenRewritePatternSet stage1FrozenPatterns(std::move(stage1Patterns));
  if (!applyPatterns(func, target, stage1FrozenPatterns)) {
    return signalPassFailure();
  }
  // Explict BroadcastTo addition for left-over broadcast-able ops.
  // The following pattern matchings should be done after the other legalization
  // rules in order not to add unnecessary BroadcastTo ops.
  RewritePatternSet stage2Patterns(&getContext());

  addPatterns(context, stage2Patterns, this->preserve_assert_op_);

  stage2Patterns.add<ApplyExplicitBroadcasting<TF::LessEqualOp>,
                     ApplyExplicitBroadcasting<TF::GreaterEqualOp>,
                     ApplyExplicitBroadcasting<TF::NotEqualOp>,
                     ApplyExplicitBroadcasting<TF::GreaterOp>,
                     ApplyExplicitBroadcasting<TF::LessOp>,
                     ApplyExplicitBroadcasting<TF::EqualOp>,
                     ApplyExplicitBroadcasting<TF::AddOp>,
                     ApplyExplicitBroadcasting<TF::AddV2Op>,
                     ApplyExplicitBroadcasting<TF::MulOp>,
                     ApplyExplicitBroadcasting<TF::DivOp>,
                     ApplyExplicitBroadcasting<TF::RealDivOp>,
                     ApplyExplicitBroadcasting<TF::SubOp>,
                     ApplyExplicitBroadcasting<TF::FloorDivOp>,
                     ApplyExplicitBroadcasting<TF::FloorModOp>,
                     ApplyExplicitBroadcasting<TF::PowOp>,
                     ApplyExplicitBroadcasting<TF::MaximumOp>,
                     ApplyExplicitBroadcasting<TF::MinimumOp>,
                     ApplyExplicitBroadcasting<TF::SquaredDifferenceOp>,
                     ApplyExplicitBroadcasting<TF::SelectV2Op>>(context);

  FrozenRewritePatternSet stage2FrozenPatterns(std::move(stage2Patterns));
  if (!applyPatterns(func, target, stage2FrozenPatterns)) {
    return signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect LegalizeTF pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeTFPass(
    bool run_tfl_runtime_verification, bool preserve_assert_op) {
  return std::make_unique<LegalizeTFPass>(run_tfl_runtime_verification,
                                          preserve_assert_op);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeTFPass() {
  return std::make_unique<LegalizeTFPass>();
}

}  // namespace TFL
}  // namespace mlir
