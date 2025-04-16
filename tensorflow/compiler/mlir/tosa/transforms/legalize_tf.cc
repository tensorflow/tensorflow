/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Legalize TensorFlow to TOSA

#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-legalize-tf"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {

#define GEN_PASS_DEF_TOSALEGALIZETFPASS
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

// Performs lowering to TOSA dialect
class LegalizeTF : public impl::TosaLegalizeTFPassBase<LegalizeTF> {
 public:
  explicit LegalizeTF() = default;
  void runOnOperation() override;
};

// All the Pat<> lowering mappings.
#include "tensorflow/compiler/mlir/tosa/transforms/tf_legalize_patterns.inc"

#define DECL_CONVERT_OP(tf_op)                                               \
  struct ConvertTF##tf_op##Op : public RewritePattern {                      \
    explicit ConvertTF##tf_op##Op(MLIRContext* context)                      \
        : RewritePattern(TF::tf_op##Op::getOperationName(), 1, context) {}   \
    LogicalResult matchAndRewrite(Operation* op,                             \
                                  PatternRewriter& rewriter) const override; \
  }

// All the explcitly implemented complex lowerings.
DECL_CONVERT_OP(MatMul);
DECL_CONVERT_OP(Relu);
DECL_CONVERT_OP(Relu6);
DECL_CONVERT_OP(Equal);
DECL_CONVERT_OP(NotEqual);
DECL_CONVERT_OP(Greater);
DECL_CONVERT_OP(GreaterEqual);
DECL_CONVERT_OP(Add);
DECL_CONVERT_OP(AddV2);
DECL_CONVERT_OP(AddN);
DECL_CONVERT_OP(Sub);
DECL_CONVERT_OP(Mul);
DECL_CONVERT_OP(Square);
DECL_CONVERT_OP(SquaredDifference);
DECL_CONVERT_OP(Sign);
DECL_CONVERT_OP(Round);
DECL_CONVERT_OP(FloorDiv);
DECL_CONVERT_OP(FloorMod);
DECL_CONVERT_OP(Assert);
DECL_CONVERT_OP(Maximum);
DECL_CONVERT_OP(Minimum);
DECL_CONVERT_OP(RealDiv);
DECL_CONVERT_OP(ArgMax);
DECL_CONVERT_OP(AvgPool);
DECL_CONVERT_OP(MaxPool);
DECL_CONVERT_OP(ConcatV2);
DECL_CONVERT_OP(Reshape);
DECL_CONVERT_OP(Rank);
DECL_CONVERT_OP(Shape);
DECL_CONVERT_OP(ExpandDims);
DECL_CONVERT_OP(Squeeze);
DECL_CONVERT_OP(Fill);
DECL_CONVERT_OP(Conv2D);
DECL_CONVERT_OP(Conv3D);
DECL_CONVERT_OP(DepthwiseConv2dNative);
DECL_CONVERT_OP(Conv2DBackpropInput);
DECL_CONVERT_OP(Elu);
DECL_CONVERT_OP(Softmax);
DECL_CONVERT_OP(LogSoftmax);
DECL_CONVERT_OP(All);
DECL_CONVERT_OP(Any);
DECL_CONVERT_OP(Max);
DECL_CONVERT_OP(Min);
DECL_CONVERT_OP(Mean);
DECL_CONVERT_OP(Prod);
DECL_CONVERT_OP(Sum);
DECL_CONVERT_OP(FusedBatchNorm);
DECL_CONVERT_OP(FusedBatchNormV3);
DECL_CONVERT_OP(BiasAdd);
DECL_CONVERT_OP(Split);
DECL_CONVERT_OP(SplitV);
DECL_CONVERT_OP(Pack);
DECL_CONVERT_OP(Unpack);
DECL_CONVERT_OP(Transpose);
DECL_CONVERT_OP(Tile);
DECL_CONVERT_OP(Slice);
DECL_CONVERT_OP(StridedSlice);
DECL_CONVERT_OP(Less);
DECL_CONVERT_OP(LessEqual);
DECL_CONVERT_OP(Pad);
DECL_CONVERT_OP(PadV2);
DECL_CONVERT_OP(MirrorPad);
DECL_CONVERT_OP(ResizeBilinear);
DECL_CONVERT_OP(ResizeNearestNeighbor);
DECL_CONVERT_OP(Gather);
DECL_CONVERT_OP(GatherV2);
DECL_CONVERT_OP(GatherNd);
DECL_CONVERT_OP(ScatterNd);
DECL_CONVERT_OP(SelectV2);
DECL_CONVERT_OP(SpaceToDepth);
DECL_CONVERT_OP(DepthToSpace);
DECL_CONVERT_OP(Sin);
DECL_CONVERT_OP(Cos);
DECL_CONVERT_OP(SpaceToBatchND);
DECL_CONVERT_OP(BatchToSpaceND);
DECL_CONVERT_OP(ZerosLike);
DECL_CONVERT_OP(Sigmoid);
DECL_CONVERT_OP(Tanh);
DECL_CONVERT_OP(LeakyRelu);
DECL_CONVERT_OP(Neg);
DECL_CONVERT_OP(StopGradient);
DECL_CONVERT_OP(ReverseV2);
DECL_CONVERT_OP(FakeQuantWithMinMaxArgs);
DECL_CONVERT_OP(FakeQuantWithMinMaxVars);
DECL_CONVERT_OP(LeftShift);
DECL_CONVERT_OP(RightShift);
DECL_CONVERT_OP(OneHot);
DECL_CONVERT_OP(BatchMatMulV2);
DECL_CONVERT_OP(BroadcastTo);
DECL_CONVERT_OP(BitwiseOr);
DECL_CONVERT_OP(BitwiseXor);
DECL_CONVERT_OP(BitwiseAnd);
DECL_CONVERT_OP(LogicalAnd);
DECL_CONVERT_OP(LogicalOr);
DECL_CONVERT_OP(Pow);
#undef DECL_CONVERT_OP

LogicalResult ConvertTFReluOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_relu_op = cast<TF::ReluOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_relu_op.getResult().getType());
  // Not a tensor output
  if (!output_type) return failure();

  auto element_type = output_type.getElementType();
  if (auto quant_type =
          dyn_cast<mlir::quant::UniformQuantizedType>(element_type)) {
    element_type = quant_type.getStorageType();
  }

  mlir::Attribute min_val, max_val;
  if (element_type.isa<mlir::FloatType>()) {
    min_val = rewriter.getFloatAttr(element_type, 0.0f);
    max_val =
        rewriter.getFloatAttr(element_type, std::numeric_limits<float>::max());
  } else {
    min_val = rewriter.getIntegerAttr(element_type, 0);
    max_val = rewriter.getIntegerAttr(element_type,
                                      std::numeric_limits<int32_t>::max());
  }

  CreateReplaceOpAndInfer<tosa::ClampOp>(
      rewriter, op, output_type, tf_relu_op.getFeatures(), min_val, max_val);
  return success();
}

LogicalResult ConvertTFRelu6Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_relu6_op = cast<TF::Relu6Op>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_relu6_op.getResult().getType());
  // Not a tensor output
  if (!output_type) return failure();

  auto element_type = output_type.getElementType();
  if (auto quant_type =
          dyn_cast<mlir::quant::UniformQuantizedType>(element_type)) {
    element_type = quant_type.getStorageType();
  }

  mlir::Attribute min_val, max_val;
  if (element_type.isa<mlir::FloatType>()) {
    min_val = rewriter.getFloatAttr(element_type, 0.0f);
    max_val = rewriter.getFloatAttr(element_type, 6.0f);
  } else {
    min_val = rewriter.getIntegerAttr(element_type, 0);
    max_val = rewriter.getIntegerAttr(element_type, 6);
  }

  CreateReplaceOpAndInfer<tosa::ClampOp>(
      rewriter, op, output_type, tf_relu6_op.getFeatures(), min_val, max_val);
  return success();
}

LogicalResult ConvertTFNotEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_not_equal_op = cast<TF::NotEqualOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_not_equal_op.getResult().getType());
  // Not a tensor output
  if (!output_type) return failure();

  Value x = tf_not_equal_op.getX();
  Value y = tf_not_equal_op.getY();

  auto op1_equal_in = CreateOpAndInfer<tosa::EqualOp>(rewriter, op->getLoc(),
                                                      output_type, x, y);

  auto op2_not_op1 = CreateOpAndInfer<tosa::LogicalNotOp>(
      rewriter, op->getLoc(), output_type, op1_equal_in.getResult());

  rewriter.replaceOp(op, {op2_not_op1.getResult()});

  return success();
}

LogicalResult ConvertTFSignOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_sign_op = cast<TF::SignOp>(op);

  RankedTensorType output_type =
      mlir::cast<RankedTensorType>(tf_sign_op.getResult().getType());

  std::optional<Value> result =
      convertSignOp(rewriter, op, tf_sign_op.getX(), output_type);
  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});
  return success();
}

LogicalResult ConvertTFSinOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_sin_op = cast<TF::SinOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tf_sin_op.getResult().getType());
  if (!output_type)
    return rewriter.notifyMatchFailure(op, "output_type required");

  CreateReplaceOpAndInfer<tosa::SinOp>(rewriter, op, output_type,
                                       tf_sin_op.getX());

  return success();
}

LogicalResult ConvertTFCosOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_cos_op = cast<TF::CosOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tf_cos_op.getResult().getType());
  if (!output_type)
    return rewriter.notifyMatchFailure(op, "output_type required");

  CreateReplaceOpAndInfer<tosa::CosOp>(rewriter, op, output_type,
                                       tf_cos_op.getX());

  return success();
}

LogicalResult ConvertTFAddOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_add_op = cast<TF::AddOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_add_op.getResult().getType());
  // Not a tensor output
  if (!output_type) return failure();

  Value x = tf_add_op.getX();
  Value y = tf_add_op.getY();

  CreateReplaceOpAndInfer<tosa::AddOp>(rewriter, op, output_type, x, y);
  return success();
}

LogicalResult ConvertTFAddV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_addv2_op = cast<TF::AddV2Op>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_addv2_op.getResult().getType());
  // Not a tensor output
  if (!output_type) return failure();

  Value x = tf_addv2_op.getX();
  Value y = tf_addv2_op.getY();

  CreateReplaceOpAndInfer<tosa::AddOp>(rewriter, op, output_type, x, y);
  return success();
}

// AddN is commutative
LogicalResult ConvertTFAddNOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_addn_op = cast<TF::AddNOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_addn_op.getResult().getType());
  // Not a tensor output
  if (!output_type) return failure();

  SmallVector<Value> inputs(tf_addn_op.getInputs());

  assert(inputs.size() >= 2);

  Value lhs = inputs[0];
  Value rhs = inputs[1];
  auto newOp = CreateOpAndInfer<tosa::AddOp>(rewriter, op->getLoc(),
                                             output_type, lhs, rhs);
  for (int i = 2; i < inputs.size(); i++) {
    Value lhs = inputs[i];
    Value rhs = newOp.getResult();
    newOp = CreateOpAndInfer<tosa::AddOp>(rewriter, op->getLoc(), output_type,
                                          lhs, rhs);
  }

  rewriter.replaceOp(op, {newOp.getResult()});

  return success();
}

LogicalResult ConvertTFSubOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_sub_op = cast<TF::SubOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_sub_op.getResult().getType());
  // Not a tensor output
  if (!output_type) return failure();

  Value lhs = tf_sub_op.getX();
  Value rhs = tf_sub_op.getY();

  CreateReplaceOpAndInfer<tosa::SubOp>(rewriter, op, output_type, lhs, rhs);
  return success();
}

LogicalResult ConvertTFMulOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_mul_op = cast<TF::MulOp>(op);

  std::optional<Value> result = convertMultiplyOp(
      rewriter, op, tf_mul_op.getResult(), tf_mul_op.getX(), tf_mul_op.getY());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});
  return success();
}

LogicalResult ConvertTFSquareOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_square_op = cast<TF::SquareOp>(op);

  std::optional<Value> result =
      convertMultiplyOp(rewriter, op, tf_square_op.getResult(),
                        tf_square_op.getX(), tf_square_op.getX());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});
  return success();
}

LogicalResult ConvertTFSquaredDifferenceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_squared_op = cast<TF::SquaredDifferenceOp>(op);

  std::optional<Value> result =
      convertSquaredDifferenceOp(rewriter, op, tf_squared_op.getResult(),
                                 tf_squared_op.getX(), tf_squared_op.getY());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});
  return success();
}

LogicalResult ConvertTFRoundOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_round_op = cast<TF::RoundOp>(op);

  TensorType input_type = dyn_cast<TensorType>(tf_round_op.getX().getType());
  if (!input_type) {
    return rewriter.notifyMatchFailure(op, "input not tensor type");
  }

  if (mlir::isa<FloatType>(input_type.getElementType())) {
    std::optional<Value> result = convertRoundOp(
        rewriter, op, tf_round_op.getResult(), tf_round_op.getX());

    if (!result) return failure();

    rewriter.replaceOp(op, {result.value()});
    return success();

  } else {
    tf_round_op.replaceAllUsesWith(tf_round_op.getX());
    return success();
  }
}

LogicalResult ConvertTFFloorDivOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_floordiv_op = cast<TF::FloorDivOp>(op);

  std::optional<Value> result =
      convertFloorDivOp(rewriter, op, tf_floordiv_op.getResult(),
                        tf_floordiv_op.getX(), tf_floordiv_op.getY());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFFloorModOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_floormod_op = cast<TF::FloorModOp>(op);

  std::optional<Value> result =
      convertFloorModOp(rewriter, op, tf_floormod_op.getResult(),
                        tf_floormod_op.getX(), tf_floormod_op.getY());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFAssertOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  op->dropAllReferences();
  op->erase();
  return success();
}

LogicalResult ConvertTFRealDivOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_div_op = cast<TF::RealDivOp>(op);

  TensorType y_type = dyn_cast<TensorType>(tf_div_op.getY().getType());
  TensorType output_type =
      dyn_cast<TensorType>(tf_div_op.getResult().getType());
  // Not a tensor output
  if (!output_type || !y_type) return failure();

  Type element_type = output_type.getElementType();

  Value x = tf_div_op.getX();
  Value y = tf_div_op.getY();

  if (mlir::isa<IntegerType>(element_type)) {
    CreateReplaceOpAndInfer<tosa::IntDivOp>(rewriter, op, output_type, x, y);
    return success();
  }

  auto reciprocal_op = CreateOpAndInfer<tosa::ReciprocalOp>(
      rewriter, op->getLoc(), y.getType(), y);

  auto mul_op = CreateMulOpAndInfer(
      rewriter, op, output_type, tf_div_op.getX(),
      reciprocal_op.getResult());
  rewriter.replaceOp(op, {mul_op.getResult()});

  return success();
}

LogicalResult ConvertTFArgMaxOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_argmax_op = cast<TF::ArgMaxOp>(op);

  TensorType input_type =
      dyn_cast<TensorType>(tf_argmax_op.getInput().getType());
  TensorType output_type =
      dyn_cast<TensorType>(tf_argmax_op.getResult().getType());
  // Not a tensor output
  if (!output_type || !input_type) return failure();

  ElementsAttr axis_elems;
  if (!matchPattern(tf_argmax_op.getDimension(), m_Constant(&axis_elems)))
    return failure();

  int32_t axis = axis_elems.getValues<IntegerAttr>()[0].getInt();
  if (axis < 0) {
    axis += input_type.getRank();
  }

  if (axis < 0 || axis >= input_type.getRank()) {
    return rewriter.notifyMatchFailure(op, "invalid axis value");
  }

  IntegerAttr axis_attr = rewriter.getI32IntegerAttr(axis);

  CreateReplaceOpAndInfer<tosa::ArgMaxOp>(rewriter, op, output_type,
                                          tf_argmax_op.getInput(), axis_attr,
                                          rewriter.getStringAttr("PROPAGATE"));

  return success();
}

LogicalResult ConvertTFAvgPoolOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_avgpool_op = cast<TF::AvgPoolOp>(op);

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tf_avgpool_op.getValue().getType());
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_avgpool_op.getResult().getType());
  // Not a ranked tensor output
  if (!input_type || !output_type) return failure();

  auto tmpAttr = tf_avgpool_op.getDataFormatAttr();
  if (tmpAttr && tmpAttr.getValue().str() != "NHWC") return failure();

  DenseI64ArrayAttr pad;
  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr kernel;
  {
    auto tmpAttr = tf_avgpool_op.getStrides();
    if (!tmpAttr) {
      stride = rewriter.getDenseI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t stride_h = dyn_cast<IntegerAttr>(tmpAttr[1]).getInt();
      int64_t stride_w = dyn_cast<IntegerAttr>(tmpAttr[2]).getInt();
      stride = rewriter.getDenseI64ArrayAttr({stride_h, stride_w});
    }
  }
  {
    auto tmpAttr = tf_avgpool_op.getKsize();
    if (!tmpAttr) {
      kernel = rewriter.getDenseI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t kernel_h = dyn_cast<IntegerAttr>(tmpAttr[1]).getInt();
      int64_t kernel_w = dyn_cast<IntegerAttr>(tmpAttr[2]).getInt();
      kernel = rewriter.getDenseI64ArrayAttr({kernel_h, kernel_w});
    }
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tf_avgpool_op.getPadding().str(), &tf_pad).ok())
      return failure();

    DenseI64ArrayAttr dilation = rewriter.getDenseI64ArrayAttr(
        {1, 1});  // Pooling has no non-unit dilation

    SmallVector<int64_t, 4> i64array;

    for (auto& elem : tf_avgpool_op.getKsize()) {
      int64_t value = dyn_cast<IntegerAttr>(elem).getInt();
      i64array.emplace_back(value);
    }

    RankedTensorType filter_type = tensorflow::GetTypeFromTFTensorShape(
        llvm::ArrayRef(i64array), rewriter.getIntegerType(64));

    if (!getPaddingValuesFromPadType(
            tf_pad,
            tensorflow::FORMAT_NHWC,  // TFLite only supports this
            1,                        // tensorflow::FORMAT_OHWI,
            input_type, filter_type, stride, dilation, rewriter, pad))
      return failure();
  }

  // Tosa supports FP16 and FP32 accumulator type for FP16 input. When the time
  // FP16 is supported, the accumulator type can be selected based on trade-off
  // between performance and accuracy. Set to FP32 by default.
  auto acc_attr = mlir::TypeAttr::get(rewriter.getF32Type());

  CreateReplaceOpAndInfer<tosa::AvgPool2dOp>(rewriter, op, output_type,
                                             tf_avgpool_op.getValue(), kernel,
                                             stride, pad, acc_attr);
  return success();
}

LogicalResult ConvertTFMaxPoolOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_maxpool_op = cast<TF::MaxPoolOp>(op);

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tf_maxpool_op.getInput().getType());
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_maxpool_op.getResult().getType());
  // Not a ranked tensor output
  if (!input_type || !output_type) return failure();

  auto tmpAttr = tf_maxpool_op.getDataFormatAttr();
  if (tmpAttr && tmpAttr.getValue().str() != "NHWC") return failure();

  DenseI64ArrayAttr pad;
  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr kernel;
  {
    auto tmpAttr = tf_maxpool_op.getStrides();
    if (!tmpAttr) {
      stride = rewriter.getDenseI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t stride_h = dyn_cast<IntegerAttr>(tmpAttr[1]).getInt();
      int64_t stride_w = dyn_cast<IntegerAttr>(tmpAttr[2]).getInt();
      stride = rewriter.getDenseI64ArrayAttr({stride_h, stride_w});
    }
  }
  {
    auto tmpAttr = tf_maxpool_op.getKsize();
    if (!tmpAttr) {
      kernel = rewriter.getDenseI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t kernel_h = dyn_cast<IntegerAttr>(tmpAttr[1]).getInt();
      int64_t kernel_w = dyn_cast<IntegerAttr>(tmpAttr[2]).getInt();
      kernel = rewriter.getDenseI64ArrayAttr({kernel_h, kernel_w});
    }
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tf_maxpool_op.getPadding().str(), &tf_pad).ok())
      return failure();

    // Pooling has no non-unit dilation
    DenseI64ArrayAttr dilation = rewriter.getDenseI64ArrayAttr({1, 1});

    SmallVector<int64_t, 4> i64array;

    for (auto& elem : tf_maxpool_op.getKsize()) {
      int64_t value = dyn_cast<IntegerAttr>(elem).getInt();
      i64array.emplace_back(value);
    }

    RankedTensorType filter_type = tensorflow::GetTypeFromTFTensorShape(
        llvm::ArrayRef(i64array), rewriter.getIntegerType(64));

    if (!getPaddingValuesFromPadType(
            tf_pad,
            tensorflow::FORMAT_NHWC,  // TFLite only supports this
            1,                        // tensorflow::FORMAT_OHWI,
            input_type, filter_type, stride, dilation, rewriter, pad))
      return failure();
  }

  CreateReplaceOpAndInfer<tosa::MaxPool2dOp>(
      rewriter, op, output_type, tf_maxpool_op.getInput(), kernel, stride, pad);
  return success();
}

LogicalResult ConvertTFConcatV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_concatv2_op = cast<TF::ConcatV2Op>(op);
  auto result_type =
      mlir::cast<ShapedType>(tf_concatv2_op.getResult().getType());
  SmallVector<Value> values(tf_concatv2_op.getValues());

  ElementsAttr axis_elems;
  if (!matchPattern(tf_concatv2_op.getAxis(), m_Constant(&axis_elems)))
    return failure();

  int32_t axis = axis_elems.getValues<IntegerAttr>()[0].getInt();

  std::optional<Value> result =
      convertConcatV2Op(rewriter, op, result_type, values, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFReshapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_reshape_op = cast<TF::ReshapeOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_reshape_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  // Regular way to match tensor as element attribute doesn't always work
  // use output_type.getShape() which is more stable
  SmallVector<int64_t> shape_vals;
  for (int i = 0; i < output_type.getShape().size(); i++) {
    shape_vals.push_back(output_type.getShape()[i]);
  }
  DenseI64ArrayAttr shape_attr = rewriter.getDenseI64ArrayAttr(shape_vals);

  auto shape_value = getTosaConstShape(rewriter, op->getLoc(), shape_vals);
  CreateReplaceOpAndInfer<tosa::ReshapeOp>(
      rewriter, op, output_type, tf_reshape_op.getTensor(), shape_value);
  return success();
}

LogicalResult ConvertTFRankOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_rank_op = cast<TF::RankOp>(op);

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tf_rank_op.getInput().getType());
  if (!input_type) return failure();

  int32_t rank = input_type.getRank();

  RankedTensorType rank_type =
      tensorflow::GetTypeFromTFTensorShape({1}, rewriter.getIntegerType(32));
  auto rank_attr = DenseI32ArrayAttr::get(rewriter.getContext(), {rank});
  auto rank_const = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), rank_type, cast<mlir::ElementsAttr>(rank_attr));

  rewriter.replaceOp(op, {rank_const.getResult()});

  return success();
}

LogicalResult ConvertTFShapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_shape_op = cast<TF::ShapeOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_shape_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tf_shape_op.getInput().getType());
  if (!input_type) return failure();

  auto input_shape = input_type.getShape();

  SmallVector<int32_t> shape_arr;
  for (int i = 0; i < input_shape.size(); i++) {
    shape_arr.emplace_back(input_shape[i]);
  }

  RankedTensorType shape_type = tensorflow::GetTypeFromTFTensorShape(
      {static_cast<int32_t>(shape_arr.size())}, rewriter.getIntegerType(32));
  auto shape_attr =
      DenseI32ArrayAttr::get(rewriter.getContext(), llvm::ArrayRef(shape_arr));
  auto shape_const = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), shape_type, cast<mlir::ElementsAttr>(shape_attr));

  rewriter.replaceOp(op, {shape_const.getResult()});

  return success();
}

LogicalResult ConvertTFExpandDimsOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_expanddims_op = cast<TF::ExpandDimsOp>(op);

  std::optional<Value> result = convertExpandDimsOp(
      rewriter, op, tf_expanddims_op.getResult(), tf_expanddims_op.getInput(),
      tf_expanddims_op.getDim());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFSqueezeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_squeeze_op = cast<TF::SqueezeOp>(op);

  // Copy squeeze_dims into int32_t array
  auto squeeze_dims_attr = tf_squeeze_op.getSqueezeDimsAttr();
  SmallVector<int32_t> squeeze_dims;
  for (auto& squeeze_dim : squeeze_dims_attr) {
    squeeze_dims.emplace_back(dyn_cast<IntegerAttr>(squeeze_dim).getInt());
  }

  std::optional<Value> result =
      convertSqueezeOp(rewriter, op, tf_squeeze_op.getResult(),
                       tf_squeeze_op.getInput(), squeeze_dims);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFFillOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_fill_op = cast<TF::FillOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_fill_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  ElementsAttr dims_elems;
  if (!matchPattern(tf_fill_op.getDims(), m_Constant(&dims_elems)))
    return failure();
  SmallVector<int64_t> dims_vals;
  uint32_t total_size = 1;
  for (int i = 0; i < dims_elems.getNumElements(); i++) {
    dims_vals.push_back(dims_elems.getValues<IntegerAttr>()[i].getInt());
    total_size *= dims_vals[i];
  }

  ElementsAttr value_elem;
  if (!matchPattern(tf_fill_op.getValue(), m_Constant(&value_elem)))
    return failure();

  RankedTensorType fill_type = tensorflow::GetTypeFromTFTensorShape(
      ArrayRef<int64_t>(dims_vals),
      value_elem.getShapedType().getElementType());
  DenseArrayAttr fill_attr;

  // Convert to a compatible zero type
  if (mlir::isa<FloatType>(value_elem.getShapedType().getElementType())) {
    SmallVector<float> fill_arr(
        total_size,
        value_elem.getValues<FloatAttr>()[0].getValue().convertToFloat());
    fill_attr =
        DenseF32ArrayAttr::get(rewriter.getContext(), llvm::ArrayRef(fill_arr));
  } else {
    SmallVector<int32_t> fill_arr(
        total_size,
        value_elem.getValues<IntegerAttr>()[0].getValue().getLimitedValue());
    fill_attr =
        DenseI32ArrayAttr::get(rewriter.getContext(), llvm::ArrayRef(fill_arr));
  }
  auto fill_const_op = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), fill_type, mlir::cast<ElementsAttr>(fill_attr));
  rewriter.replaceOp(op, {fill_const_op.getResult()});

  return success();
}

LogicalResult ConvertTFConv2DOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_conv2d_op = cast<TF::Conv2DOp>(op);

  RankedTensorType filter_type =
      dyn_cast<RankedTensorType>(tf_conv2d_op.getFilter().getType());
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_conv2d_op.getResult().getType());

  // Set up a zero attr for subsequent pattern replacement if required
  auto bias_dim = filter_type.getShape().back();
  RankedTensorType bias_type = tensorflow::GetTypeFromTFTensorShape(
      {bias_dim}, filter_type.getElementType());
  auto bias_attr = rewriter.getZeroAttr(bias_type);
  auto bias = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), bias_type, mlir::cast<ElementsAttr>(bias_attr));

  std::optional<Value> result = convertTFConv2DCommon(
      rewriter, op, output_type, tf_conv2d_op.getInput(),
      tf_conv2d_op.getFilter(), bias, tf_conv2d_op.getStrides(),
      tf_conv2d_op.getDilations(), tf_conv2d_op.getExplicitPaddings(),
      tf_conv2d_op.getPadding(), tf_conv2d_op.getDataFormat());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFConv3DOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_conv3d_op = cast<TF::Conv3DOp>(op);

  RankedTensorType filter_type =
      dyn_cast<RankedTensorType>(tf_conv3d_op.getFilter().getType());
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_conv3d_op.getResult().getType());

  if (!filter_type || !output_type) {
    return rewriter.notifyMatchFailure(
        op, "filter/output are not all a ranked tensor");
  }

  // Set up a zero attr for subsequent pattern replacement if required
  auto bias_dim = filter_type.getShape().back();
  RankedTensorType bias_type =
      RankedTensorType::get({bias_dim}, filter_type.getElementType());
  auto bias_attr = rewriter.getZeroAttr(bias_type);
  auto bias = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), bias_type, mlir::cast<ElementsAttr>(bias_attr));

  std::optional<Value> result = convertTFConv3DCommon(
      rewriter, op, output_type, tf_conv3d_op.getInput(),
      tf_conv3d_op.getFilter(), bias, tf_conv3d_op.getStrides(),
      tf_conv3d_op.getDilations(), tf_conv3d_op.getPadding(),
      tf_conv3d_op.getDataFormat());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFDepthwiseConv2dNativeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_dwconv2d_op = cast<TF::DepthwiseConv2dNativeOp>(op);

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tf_dwconv2d_op.getInput().getType());
  RankedTensorType filter_type =
      dyn_cast<RankedTensorType>(tf_dwconv2d_op.getFilter().getType());
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_dwconv2d_op.getResult().getType());
  // Not a ranked tensor output
  if (!input_type) return failure();
  if (!output_type) return failure();

  // Set up a zero attr for subsequent pattern replacement if required
  if (!filter_type) {
    return rewriter.notifyMatchFailure(op, "filter type unranked tensor");
  }

  auto tmpAttr = tf_dwconv2d_op.getDataFormatAttr();
  if (tmpAttr && tmpAttr.getValue().str() != "NHWC") return failure();

  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr dilation;
  DenseI64ArrayAttr pad;
  {
    auto tmpAttr = tf_dwconv2d_op.getStrides();
    if (!tmpAttr) {
      stride = rewriter.getDenseI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t stride_h = dyn_cast<IntegerAttr>(tmpAttr[1]).getInt();
      int64_t stride_w = dyn_cast<IntegerAttr>(tmpAttr[2]).getInt();
      stride = rewriter.getDenseI64ArrayAttr({stride_h, stride_w});
    }
  }
  {
    auto tmpAttr = tf_dwconv2d_op.getDilations();
    if (!tmpAttr) {
      dilation = rewriter.getDenseI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t dilation_h = dyn_cast<IntegerAttr>(tmpAttr[1]).getInt();
      int64_t dilation_w = dyn_cast<IntegerAttr>(tmpAttr[2]).getInt();
      dilation = rewriter.getDenseI64ArrayAttr({dilation_h, dilation_w});
    }
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tf_dwconv2d_op.getPadding().str(), &tf_pad).ok())
      return failure();

    tensorflow::TensorFormat data_format_tf;
    if (!FormatFromString(tf_dwconv2d_op.getDataFormat().str(),
                          &data_format_tf))
      return failure();

    if (tf_pad == tensorflow::Padding::EXPLICIT) {
      pad = getPaddingValuesFromExplicitPadAttr(
          tf_dwconv2d_op.getExplicitPaddings(), data_format_tf, rewriter);
    } else {
      if (!getPaddingValuesFromPadType(tf_pad, data_format_tf,
                                       0,  // tensorflow::FORMAT_HWIO
                                       input_type, filter_type, stride,
                                       dilation, rewriter, pad))
        return failure();
    }
  }

  auto filter_shape = filter_type.getShape();
  auto bias_dim = filter_shape[2] * filter_shape[3];
  RankedTensorType bias_type = tensorflow::GetTypeFromTFTensorShape(
      {bias_dim}, filter_type.getElementType());
  auto bias_attr = rewriter.getZeroAttr(bias_type);
  auto bias = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), bias_type, mlir::cast<ElementsAttr>(bias_attr));

  auto acc_type =
      getConvAccTypeAttr(rewriter,
                         /* input_etype = */ input_type.getElementType(),
                         /* output_etype = */ output_type.getElementType());

  CreateReplaceOpAndInfer<tosa::DepthwiseConv2DOp>(
      rewriter, op, output_type, tf_dwconv2d_op.getInput(),
      tf_dwconv2d_op.getFilter(), bias, pad, stride, dilation, acc_type);
  return success();
}

LogicalResult ConvertTFConv2DBackpropInputOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_conv_op = cast<TF::Conv2DBackpropInputOp>(op);

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tf_conv_op.getOutBackprop().getType());
  RankedTensorType filter_type =
      dyn_cast<RankedTensorType>(tf_conv_op.getFilter().getType());
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_conv_op.getResult().getType());
  // Not a ranked tensor output
  if (!input_type) return failure();
  if (!filter_type) return failure();
  if (!output_type) return failure();

  // Transpose [H, W, I, O] to [O, H, W, I]
  auto filter_shape = filter_type.getShape();
  SmallVector<int64_t, 4> a1_transpose_dims;
  a1_transpose_dims.push_back(filter_shape[2]);
  a1_transpose_dims.push_back(filter_shape[0]);
  a1_transpose_dims.push_back(filter_shape[1]);
  a1_transpose_dims.push_back(filter_shape[3]);

  auto a1_filter_transpose_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(ArrayRef<int64_t>(a1_transpose_dims),
                                           filter_type.getElementType()),
      tf_conv_op.getFilter(), rewriter.getDenseI32ArrayAttr({2, 0, 1, 3}));

  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr outpad;
  {
    auto tmpAttr = tf_conv_op.getStrides();
    if (!tmpAttr) {
      stride = rewriter.getDenseI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t stride_h = dyn_cast<IntegerAttr>(tmpAttr[1]).getInt();
      int64_t stride_w = dyn_cast<IntegerAttr>(tmpAttr[2]).getInt();
      stride = rewriter.getDenseI64ArrayAttr({stride_h, stride_w});
    }
  }
  {
    auto tmpAttr = tf_conv_op.getDilations();
    if (tmpAttr) {
      // Note: hardcoded to NHWC for now
      int64_t dilation_h = dyn_cast<IntegerAttr>(tmpAttr[1]).getInt();
      int64_t dilation_w = dyn_cast<IntegerAttr>(tmpAttr[2]).getInt();
      // TOSA transpose_conv2d does not support non-unit dilation
      if (dilation_h != 1 || dilation_w != 1) return failure();
    }
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tf_conv_op.getPadding().str(), &tf_pad).ok())
      return failure();

    tensorflow::TensorFormat data_format_tf;
    if (!FormatFromString(tf_conv_op.getDataFormat().str(), &data_format_tf))
      return failure();

    if (tf_pad == tensorflow::Padding::EXPLICIT) {
      outpad = getPaddingValuesFromExplicitPadAttr(
          tf_conv_op.getExplicitPaddings(), data_format_tf, rewriter);
    } else {
      if (!getTransposeConv2dPaddingValues(tf_pad, data_format_tf,
                                           0,  // tensorflow::FORMAT_HWIO,
                                           input_type, filter_type, output_type,
                                           stride, rewriter, outpad))
        return failure();
    }
  }

  int output_channel = output_type.getShape()[3];
  SmallVector<float> vec(output_channel, 0.0f);
  std::optional<Value> zero_bias =
      getConstTensor<float>(rewriter, op, vec, {output_channel});

  if (!zero_bias) return failure();

  auto acc_type =
      getConvAccTypeAttr(rewriter,
                         /* input_etype = */ input_type.getElementType(),
                         /* output_etype = */ output_type.getElementType());

  CreateReplaceOpAndInfer<tosa::TransposeConv2DOp>(
      rewriter, op, output_type, tf_conv_op.getOutBackprop(),
      a1_filter_transpose_op.getResult(), zero_bias.value(), outpad, stride,
      acc_type);

  return success();
}

LogicalResult ConvertTFAllOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_all_op = cast<TF::AllOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_all_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_all_op.getReductionIndices(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceAllOp(
      rewriter, op, output_type, tf_all_op.getInput(), axes_elems, tf_all_op.getKeepDims());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFAnyOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_any_op = cast<TF::AnyOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_any_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_any_op.getReductionIndices(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceAnyOp(
      rewriter, op, output_type, tf_any_op.getInput(), axes_elems, tf_any_op.getKeepDims());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFMaxOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_max_op = cast<TF::MaxOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_max_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_max_op.getReductionIndices(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceMaxOp(
      rewriter, op, output_type, tf_max_op.getInput(), axes_elems, tf_max_op.getKeepDims());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFMinOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_min_op = cast<TF::MinOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_min_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_min_op.getReductionIndices(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceMinOp(
      rewriter, op, output_type, tf_min_op.getInput(), axes_elems, tf_min_op.getKeepDims());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFMeanOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_mean_op = cast<TF::MeanOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_mean_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_mean_op.getReductionIndices(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceMeanOp(
      rewriter, op, output_type, tf_mean_op.getInput(), axes_elems, tf_mean_op.getKeepDims());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFProdOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_prod_op = cast<TF::ProdOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_prod_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_prod_op.getReductionIndices(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceProdOp(
      rewriter, op, output_type, tf_prod_op.getInput(), axes_elems, tf_prod_op.getKeepDims());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFSumOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_sum_op = cast<TF::SumOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_sum_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_sum_op.getReductionIndices(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceSumOp(
      rewriter, op, output_type, tf_sum_op.getInput(), axes_elems, tf_sum_op.getKeepDims());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFEluOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_elu_op = cast<TF::EluOp>(op);

  std::optional<Value> result = convertEluOp(
      rewriter, op, tf_elu_op.getResult(), tf_elu_op.getFeatures());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFSoftmaxOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_softmax_op = cast<TF::SoftmaxOp>(op);

  std::optional<Value> result =
      convertSoftmaxOp(rewriter, op, tf_softmax_op.getResult(),
                       tf_softmax_op.getLogits(), /*beta=*/1.0);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLogSoftmaxOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_logsoftmax_op = cast<TF::LogSoftmaxOp>(op);

  std::optional<Value> result = convertLogSoftmaxOp(
      rewriter, op, tf_logsoftmax_op.getResult(), tf_logsoftmax_op.getLogits());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFFusedBatchNormOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_batchnorm_op = cast<TF::FusedBatchNormOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_batchnorm_op.getResult(0).getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  // Lowering:
  // fused batchnorm = (input-mean) * scale * rsqrt(var+epsilon)) + offset
  //
  // shape_0 = ones(input.rank)
  // shape_0[input.rank-1] = input.shape[input.rank-1]
  // shape_1 = ones(1)
  //
  // bmean  = reshape(mean, shape_0)
  // bscale = reshape(scale, shape_0)
  // boffset= reshape(offset, shape_0)
  // beps   = reshape(epsilon, shape_1)
  //
  // op1 = sub(input, bmean)
  // op2 = add(var, beps)
  // op3 = rsqrt(op2)
  // bvar = reshape(op3, shape_0)
  // op4 = mul(op1, bvar)
  // op5 = mul(op4, bscale)
  // op6 = add(op5, boffset)

  Value mean = tf_batchnorm_op.getMean();
  Value variance = tf_batchnorm_op.getVariance();
  Value x = tf_batchnorm_op.getX();
  Value scale = tf_batchnorm_op.getScale();
  Value offset = tf_batchnorm_op.getOffset();

  RankedTensorType mean_type = dyn_cast<RankedTensorType>(mean.getType());
  RankedTensorType variance_type =
      dyn_cast<RankedTensorType>(variance.getType());
  RankedTensorType x_type = dyn_cast<RankedTensorType>(x.getType());
  RankedTensorType scale_type = dyn_cast<RankedTensorType>(scale.getType());
  RankedTensorType offset_type = dyn_cast<RankedTensorType>(offset.getType());
  if (!variance_type || !mean_type || !x_type || !scale_type || !offset_type)
    return failure();

  auto rank = x_type.getRank();

  Value mean_val, variance_val;

  if (mean_type.getNumElements() == 0) {
    mean_val = getTosaConstTensorSingleF32(rewriter, tf_batchnorm_op, 0, rank);
  } else {
    mean_val = tf_batchnorm_op.getMean();
  }

  if (variance_type.getNumElements() == 0) {
    variance_val =
        getTosaConstTensorSingleF32(rewriter, tf_batchnorm_op, 1.0, 1);
  } else {
    variance_val = tf_batchnorm_op.getVariance();
  }

  RankedTensorType epsilon_type =
      tensorflow::GetTypeFromTFTensorShape({1}, variance_type.getElementType());
  auto epsilon_attr =
      DenseFPElementsAttr::get(epsilon_type, {tf_batchnorm_op.getEpsilon()});
  Value epsilon_const = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), epsilon_type, epsilon_attr);

  Value op1_sub_input_mean = CreateOpAndInfer<tosa::SubOp>(
      rewriter, op->getLoc(), output_type, x, mean_val);

  Value op2_add_var_epsilon = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), variance_val.getType(), variance_val,
      epsilon_const);

  Value op3_rsqrt_op2 = CreateOpAndInfer<tosa::RsqrtOp>(
      rewriter, op->getLoc(), variance_val.getType(), op2_add_var_epsilon);

  Value op4_mul_op1_op3 = CreateMulOpAndInfer(
      rewriter, op, output_type, op1_sub_input_mean, op3_rsqrt_op2);

  Value op5_mul_op4_scale = CreateMulOpAndInfer(
      rewriter, op, output_type, op4_mul_op1_op3, scale);

  Value op6_add_op5_offset = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), output_type, op5_mul_op4_scale, offset);

  rewriter.replaceOp(op, {op6_add_op5_offset});
  return success();
}

LogicalResult ConvertTFFusedBatchNormV3Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_batchnorm_op = cast<TF::FusedBatchNormV3Op>(op);

  if (tf_batchnorm_op.getIsTraining())
    return rewriter.notifyMatchFailure(
        op, "unable to lower when is_training is set");

  for (auto value : tf_batchnorm_op.getResults().drop_front(1)) {
    if (!value.use_empty()) {
      // Really we should compute this still and let it DCE but I can't find
      // the math.
      return rewriter.notifyMatchFailure(
          op, "lowering does not support aggregate statistics");
    }
  }

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_batchnorm_op.getResult(0).getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  // Lowering:
  // fused batchnorm = (input-mean) * scale * rsqrt(var+epsilon)) + offset
  // op1 = sub(input, mean)
  // op2 = add(var, epsilon)
  // op3 = rsqrt(op2)
  // op4 = mul(op1, op3)
  // op5 = mul(op4, scale)
  // op6 = add(op5, offset)

  Value x = tf_batchnorm_op.getX();
  Value mean = tf_batchnorm_op.getMean();

  Value op1_sub_input_mean = CreateOpAndInfer<tosa::SubOp>(
      rewriter, op->getLoc(), output_type, x, mean);

  Value variance = tf_batchnorm_op.getVariance();
  RankedTensorType variance_type =
      dyn_cast<RankedTensorType>(variance.getType());
  if (!variance_type) return failure();

  auto epsilon_type =
      tensorflow::GetTypeFromTFTensorShape({1}, variance_type.getElementType());
  auto epsilon_attr =
      DenseFPElementsAttr::get(epsilon_type, {tf_batchnorm_op.getEpsilon()});
  auto epsilon_const = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), epsilon_type, epsilon_attr);

  variance_type = variance.getType().cast<RankedTensorType>();
  Value op2_add_var_epsilon = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), variance_type, variance, epsilon_const);

  Value op3_rsqrt_op2 = CreateOpAndInfer<tosa::RsqrtOp>(
      rewriter, op->getLoc(), variance_type, op2_add_var_epsilon);

  Value op4_mul_op1_op3 = CreateMulOpAndInfer(
      rewriter, op, tf_batchnorm_op.getResult(0).getType(),
      op1_sub_input_mean, op3_rsqrt_op2);

  Value scale = tf_batchnorm_op.getScale();
  Value op5_mul_op4_scale = CreateMulOpAndInfer(
      rewriter, op, output_type, op4_mul_op1_op3, scale);

  Value offset = tf_batchnorm_op.getOffset();
  Value op6_add_op5_offset = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), output_type, op5_mul_op4_scale, offset);

  llvm::SmallVector<Value> replacements = {
      op6_add_op5_offset, tf_batchnorm_op.getMean(),
      tf_batchnorm_op.getVariance(),
      // The last three are reserved spaces and have no purpose currently.
      tf_batchnorm_op.getMean(), tf_batchnorm_op.getVariance(),
      tf_batchnorm_op.getVariance()};
  rewriter.replaceOp(op, replacements);
  return success();
}

LogicalResult ConvertTFBiasAddOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_biasadd_op = cast<TF::BiasAddOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_biasadd_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  Value x = tf_biasadd_op.getValue();
  Value bias = tf_biasadd_op.getBias();

  RankedTensorType x_type = dyn_cast<RankedTensorType>(x.getType());
  RankedTensorType bias_type = dyn_cast<RankedTensorType>(bias.getType());
  if (!x_type || !bias_type) return failure();

  auto add_op = CreateOpAndInfer<tosa::AddOp>(rewriter, op->getLoc(),
                                              output_type, x, bias);

  rewriter.replaceOp(op, {add_op.getResult()});
  return success();
}

LogicalResult ConvertTFSliceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_slice_op = cast<TF::SliceOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_slice_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  ElementsAttr begin_elems, size_elems;

  SmallVector<int64_t> begin_vals, size_vals;

  // Assuming begin is always compile-time constant
  if (!matchPattern(tf_slice_op.getBegin(), m_Constant(&begin_elems))) {
    return rewriter.notifyMatchFailure(op, "begin is not constant");
  }

  for (int i = 0; i < begin_elems.getNumElements(); i++)
    begin_vals.push_back(begin_elems.getValues<IntegerAttr>()[i].getInt());

  // Try to match size as compile-time constant first,
  // if this fails, use the output tensor shape instead.
  if (matchPattern(tf_slice_op.getSize(), m_Constant(&size_elems))) {
    for (int i = 0; i < size_elems.getNumElements(); i++)
      size_vals.push_back(size_elems.getValues<IntegerAttr>()[i].getInt());
  } else {
    size_vals.assign(output_type.getShape().begin(),
                     output_type.getShape().end());
  }

  DenseI64ArrayAttr begin = rewriter.getDenseI64ArrayAttr(begin_vals);
  DenseI64ArrayAttr size = rewriter.getDenseI64ArrayAttr(size_vals);

  CreateReplaceOpAndInfer<tosa::SliceOp>(
      rewriter, op, output_type, tf_slice_op.getInput(),
      getTosaConstShape(rewriter, op->getLoc(), begin_vals),
      getTosaConstShape(rewriter, op->getLoc(), size_vals));
  return success();
}

LogicalResult ConvertTFTileOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_tile_op = cast<TF::TileOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_tile_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  ElementsAttr multiples_elems;
  if (!matchPattern(tf_tile_op.getMultiples(), m_Constant(&multiples_elems)))
    return failure();
  SmallVector<int64_t> multiples_vals;
  for (int i = 0; i < multiples_elems.getNumElements(); i++)
    multiples_vals.push_back(
        multiples_elems.getValues<IntegerAttr>()[i].getInt());

  auto multiples = getTosaConstShape(rewriter, op, multiples_vals);

  CreateReplaceOpAndInfer<tosa::TileOp>(rewriter, op, output_type,
                                        tf_tile_op.getInput(), multiples);

  return success();
}

LogicalResult ConvertTFTransposeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_transpose_op = cast<TF::TransposeOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_transpose_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) {
    return failure();
  }

  // Get the constant values for perm input
  ElementsAttr permsAttrElems;
  if (!matchPattern(tf_transpose_op.getPerm(), m_Constant(&permsAttrElems))) {
    return rewriter.notifyMatchFailure(op, "perm value is not constant");
  }

  SmallVector<int32_t> perms;
  for (auto v : permsAttrElems.getValues<APInt>()) {
    perms.push_back(v.getSExtValue());
  }

  CreateReplaceOpAndInfer<tosa::TransposeOp>(
      rewriter, op, output_type, tf_transpose_op.getX(),
      rewriter.getDenseI32ArrayAttr(perms));

  return success();
}

LogicalResult ConvertTFPackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_pack_op = cast<TF::PackOp>(op);

  SmallVector<Value> inputs(tf_pack_op.getValues());

  assert(inputs.size() >= 2);

  IntegerAttr axis_attr = tf_pack_op.getAxisAttr();
  if (!axis_attr) axis_attr = rewriter.getI32IntegerAttr(0);

  int32_t axis_i32 = axis_attr.getInt();

  std::optional<Value> result =
      convertPackOp(rewriter, op, tf_pack_op.getResult(), inputs, axis_i32);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFUnpackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_unpack_op = cast<TF::UnpackOp>(op);

  IntegerAttr axis_attr;
  {
    auto tmpAttr = tf_unpack_op.getAxisAttr();
    if (!tmpAttr) tmpAttr = rewriter.getI32IntegerAttr(0);
    axis_attr = tmpAttr;
  }
  int32_t axis_i32 = axis_attr.getInt();

  std::optional<SmallVector<Value>> results =
      convertUnpackOp(rewriter, op, tf_unpack_op.getValue(), axis_i32);

  if (!results) return failure();

  rewriter.replaceOp(op, results.value());

  return success();
}

// Splits in num_split parts along split_dim
LogicalResult ConvertTFSplitOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_split_op = cast<TF::SplitOp>(op);

  // Get the number of splits
  int32_t num_split = -1;

  auto range = tf_split_op.getODSResults(0);
  num_split = std::distance(range.begin(), range.end());

  // Get the axis
  int32_t axis = 0;
  ElementsAttr axisAttrElems;
  if (matchPattern(tf_split_op.getSplitDim(), m_Constant(&axisAttrElems))) {
    axis = axisAttrElems.getValues<IntegerAttr>()[0].getInt();
  }

  std::optional<SmallVector<Value>> results =
      convertSplitOp(rewriter, op, tf_split_op.getResult(0),
                     tf_split_op.getValue(), num_split, axis);

  if (!results) return failure();

  rewriter.replaceOp(op, results.value());

  return success();
}

// TFSplitV op splits based on a vector of sizes
LogicalResult ConvertTFSplitVOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_splitv_op = cast<TF::SplitVOp>(op);

  // Get the size_splits array
  SmallVector<int32_t> size_split;
  ElementsAttr size_split_elems;
  if (!matchPattern(tf_splitv_op.getSizeSplits(),
                    m_Constant(&size_split_elems))) {
    return failure();
  }

  for (int i = 0; i < size_split_elems.getNumElements(); i++) {
    size_split.push_back(size_split_elems.getValues<IntegerAttr>()[i].getInt());
  }

  // Get the axis
  ElementsAttr axisAttrElems;
  if (!matchPattern(tf_splitv_op.getSplitDim(), m_Constant(&axisAttrElems))) {
    return rewriter.notifyMatchFailure(op, "cannot read split_dim elems");
  }

  int32_t axis = axisAttrElems.getValues<IntegerAttr>()[0].getInt();

  std::optional<SmallVector<Value>> results =
      convertSplitVOp(rewriter, op, tf_splitv_op.getResult(0),
                      tf_splitv_op.getValue(), size_split, axis);

  if (!results) return failure();

  rewriter.replaceOp(op, results.value());

  return success();
}

LogicalResult ConvertTFLessOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_less_op = cast<TF::LessOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_less_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  Value x = tf_less_op.getX();
  Value y = tf_less_op.getY();

  // less(x, y) is not(greater_equal(x, y))
  auto greater_equal_op = CreateOpAndInfer<tosa::GreaterEqualOp>(
      rewriter, op->getLoc(), output_type, x, y);

  auto not_op = CreateOpAndInfer<tosa::LogicalNotOp>(
      rewriter, op->getLoc(), output_type, greater_equal_op.getResult());

  rewriter.replaceOp(op, {not_op.getResult()});
  return success();
}

LogicalResult ConvertTFLessEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_less_equal_op = cast<TF::LessEqualOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_less_equal_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  Value x = tf_less_equal_op.getX();
  Value y = tf_less_equal_op.getY();

  // less_equal(x, y) is not(greater(x, y))
  auto greater_op = CreateOpAndInfer<tosa::GreaterOp>(rewriter, op->getLoc(),
                                                      output_type, x, y);

  auto not_op = CreateOpAndInfer<tosa::LogicalNotOp>(
      rewriter, op->getLoc(), output_type, greater_op.getResult());

  rewriter.replaceOp(op, {not_op.getResult()});
  return success();
}

LogicalResult ConvertTFPadOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_pad_op = cast<TF::PadOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_pad_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  SmallVector<int64_t> padding_vals;
  if (failed(getVectorFromValue64(tf_pad_op.getPaddings(), padding_vals))) {
    return rewriter.notifyMatchFailure(op, "paddings is not a constant value");
  }

  Value padding = mlir::tosa::getTosaConstShape(rewriter, op->getLoc(), padding_vals);

  auto pad_op = CreateOpAndInfer<tosa::PadOp>(
      rewriter, op->getLoc(), output_type, tf_pad_op.getInput(), padding);

  rewriter.replaceOp(op, {pad_op.getResult()});
  return success();
}

LogicalResult ConvertTFPadV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_pad_op = cast<TF::PadV2Op>(op);

  RankedTensorType output_type =
      tf_pad_op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!output_type) {
    return rewriter.notifyMatchFailure(op, "output type not a ranked tensor");
  }

  Value input = tf_pad_op.getInput();
  Value constant_value = tf_pad_op.getConstantValues();

  Value rank1_scalar_value =
      reshapeScalarTo1D(rewriter, op->getLoc(), constant_value);
  if (!rank1_scalar_value) {
    return rewriter.notifyMatchFailure(
        op, "cannot reshape constant_value input to rank 1");
  }

  SmallVector<int64_t> padding_vals;
  if (failed(getVectorFromValue64(tf_pad_op.getPaddings(), padding_vals))) {
    return rewriter.notifyMatchFailure(op, "paddings is not a constant value");
  }

  Value padding = mlir::tosa::getTosaConstShape(rewriter, op->getLoc(), padding_vals);

  CreateReplaceOpAndInfer<tosa::PadOp>(rewriter, op, tf_pad_op.getType(), input,
                                       padding, rank1_scalar_value);

  return success();
}

LogicalResult ConvertTFMirrorPadOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_mirrorpad_op = cast<TF::MirrorPadOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_mirrorpad_op.getResult().getType());
  if (!output_type) {
    return rewriter.notifyMatchFailure(op, "output type isn't a ranked tensor");
  }

  TFTFLMirrorPaddingType mode;
  StringRef tf_mode = tf_mirrorpad_op.getMode();
  if (tf_mode == "REFLECT") {
    mode = TFTFLMirrorPaddingType::REFLECT;
  } else if (tf_mode == "SYMMETRIC") {
    mode = TFTFLMirrorPaddingType::SYMMETRIC;
  } else {
    return rewriter.notifyMatchFailure(
        op, "mode isn't one of REFLECT or SYMMETRIC");
  }

  std::optional<Value> result = convertMirrorPadCommon(
      rewriter, op, output_type, tf_mirrorpad_op.getInput(),
      tf_mirrorpad_op.getPaddings(), mode);

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFResizeBilinearOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_resize_op = cast<TF::ResizeBilinearOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_resize_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  std::optional<Value> result = convertResizeOp(
      rewriter, op, output_type, tf_resize_op.getImages(),
      StringRef("BILINEAR"), tf_resize_op.getAlignCornersAttr().getValue(),
      tf_resize_op.getHalfPixelCentersAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFResizeNearestNeighborOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_resize_op = cast<TF::ResizeNearestNeighborOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_resize_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  std::optional<Value> result =
      convertResizeOp(rewriter, op, output_type, tf_resize_op.getImages(),
                      StringRef("NEAREST_NEIGHBOR"),
                      tf_resize_op.getAlignCornersAttr().getValue(),
                      tf_resize_op.getHalfPixelCentersAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFMatMulOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_matmul_op = cast<TF::MatMulOp>(op);

  RankedTensorType a_type =
      dyn_cast<RankedTensorType>(tf_matmul_op.getA().getType());
  RankedTensorType b_type =
      dyn_cast<RankedTensorType>(tf_matmul_op.getB().getType());
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_matmul_op.getResult().getType());

  if (!(a_type && b_type && output_type)) {
    return rewriter.notifyMatchFailure(op, "a/b/output not ranked tensors");
  }

  if (a_type.getRank() != b_type.getRank() ||
      a_type.getRank() != output_type.getRank()) {
    return rewriter.notifyMatchFailure(op, "a/b/output rank must match");
  }

  // Can only handle rank 2 tensors for tf.MatMul.
  // Cases with rank > 2 tensors should be handled by tf.BatchMatMul or
  // tf.BatchMatMulV2
  if (a_type.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "a/b/output rank must be 2");
  }

  SmallVector<int64_t, 3> batch_a_shape(
      {1, a_type.getShape()[0], a_type.getShape()[1]});
  SmallVector<int64_t, 3> batch_b_shape(
      {1, b_type.getShape()[0], b_type.getShape()[1]});
  SmallVector<int64_t, 3> batch_output_shape(
      {1, output_type.getShape()[0], output_type.getShape()[1]});

  RankedTensorType batch_a_type = tensorflow::GetTypeFromTFTensorShape(
      batch_a_shape, a_type.getElementType());
  RankedTensorType batch_b_type = tensorflow::GetTypeFromTFTensorShape(
      batch_b_shape, b_type.getElementType());
  RankedTensorType batch_output_type = tensorflow::GetTypeFromTFTensorShape(
      batch_output_shape, output_type.getElementType());

  // Need to reshape input and output since TOSA matmul only supports
  // [N, H, C] * [N, C, W] -> [N, H, W].
  auto batch_a_shape_value = getTosaConstShape(rewriter, op->getLoc(), batch_a_shape);
  auto op1_reshape_a = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(), batch_a_type, tf_matmul_op.getA(),
      batch_a_shape_value);

  auto batch_b_shape_value = getTosaConstShape(rewriter, op->getLoc(), batch_b_shape);
  auto op2_reshape_b = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(), batch_b_type, tf_matmul_op.getB(),
      batch_b_shape_value);

  auto op3_matmul_op1_op2 = CreateOpAndInfer<tosa::MatMulOp>(
      rewriter, op->getLoc(), batch_output_type, op1_reshape_a.getResult(),
      op2_reshape_b.getResult());

  auto output_shape_value = getTosaConstShape(rewriter, op->getLoc(), output_type.getShape());
  CreateReplaceOpAndInfer<tosa::ReshapeOp>(
      rewriter, op, output_type, op3_matmul_op1_op2.getResult(),
      output_shape_value);

  return success();
}

LogicalResult ConvertTFGatherOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_gather_op = cast<TF::GatherOp>(op);

  // tf.Gather is equivalent to tf.GatherV2 with batch_dims = 0, axis = 0
  int32_t batch_dims = 0;
  int32_t axis = 0;

  std::optional<Value> result =
      convertGatherOp(rewriter, op, tf_gather_op.getParams(),
                      tf_gather_op.getIndices(), batch_dims, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFGatherV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_gather_op = cast<TF::GatherV2Op>(op);

  // Axis is a tensor.  Pull out the one integer value.
  ElementsAttr axis_elem;
  if (!matchPattern(tf_gather_op.getAxis(), m_Constant(&axis_elem)))
    return failure();
  assert(axis_elem.getNumElements() == 1);

  int32_t axis = axis_elem.getValues<IntegerAttr>()[0].getInt();
  int32_t batch_dims = tf_gather_op.getBatchDimsAttr().getInt();

  std::optional<Value> result =
      convertGatherOp(rewriter, op, tf_gather_op.getParams(),
                      tf_gather_op.getIndices(), batch_dims, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFGatherNdOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_gathernd_op = cast<TF::GatherNdOp>(op);

  std::optional<Value> result = convertGatherNdOp(
      rewriter, op, tf_gathernd_op.getResult(), tf_gathernd_op.getParams(),
      tf_gathernd_op.getIndices());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFScatterNdOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_scatternd_op = cast<TF::ScatterNdOp>(op);

  const std::optional<Value> result = convertScatterNdOp(
      rewriter, op, tfl_scatternd_op.getResult(), tfl_scatternd_op.getIndices(),
      tfl_scatternd_op.getUpdates(), tfl_scatternd_op.getShape());

  if (!result) {
    return failure();
  }
  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFSelectV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_sel_op = cast<TF::SelectV2Op>(op);

  std::optional<Value> result = convertSelectOp(
      rewriter, op, tf_sel_op.getResult(), tf_sel_op.getCondition(),
      tf_sel_op.getThenValue(), tf_sel_op.getElseValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFSpaceToDepthOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_s2d_op = cast<TF::SpaceToDepthOp>(op);

  std::optional<Value> result = convertSpaceToDepthOp(
      rewriter, op, tf_s2d_op.getResult(), tf_s2d_op.getInput(),
      tf_s2d_op.getBlockSizeAttr(), tf_s2d_op.getDataFormatAttr());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFDepthToSpaceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_d2s_op = cast<TF::DepthToSpaceOp>(op);

  std::optional<Value> result = convertDepthToSpaceOp(
      rewriter, op, tf_d2s_op.getResult(), tf_d2s_op.getInput(),
      tf_d2s_op.getBlockSizeAttr(), tf_d2s_op.getDataFormatAttr());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFSpaceToBatchNDOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_s2b_op = cast<TF::SpaceToBatchNDOp>(op);

  std::optional<Value> result = convertSpaceToBatchNDOp(
      rewriter, op, tf_s2b_op.getResult(), tf_s2b_op.getInput(),
      tf_s2b_op.getBlockShape(), tf_s2b_op.getPaddings());
  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFBatchToSpaceNDOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_b2s_op = cast<TF::BatchToSpaceNDOp>(op);

  std::optional<Value> result = convertBatchToSpaceNDOp(
      rewriter, op, tf_b2s_op.getResult(), tf_b2s_op.getInput(),
      tf_b2s_op.getBlockShape(), tf_b2s_op.getCrops());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFStridedSliceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_ss_op = cast<TF::StridedSliceOp>(op);

  std::optional<Value> result = convertStridedSliceOp(
      rewriter, op, tf_ss_op.getResult(), tf_ss_op.getInput(),
      tf_ss_op.getBegin(), tf_ss_op.getEnd(), tf_ss_op.getStrides(),
      tf_ss_op.getBeginMaskAttr().getInt(), tf_ss_op.getEndMaskAttr().getInt(),
      tf_ss_op.getEllipsisMaskAttr().getInt(),
      tf_ss_op.getNewAxisMaskAttr().getInt(),
      tf_ss_op.getShrinkAxisMaskAttr().getInt());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFZerosLikeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_zeroslike_op = cast<TF::ZerosLikeOp>(op);

  std::optional<Value> result = convertZerosLikeOp(
      rewriter, op, tf_zeroslike_op.getResult(), tf_zeroslike_op.getX());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFSigmoidOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_sigmoid_op = cast<TF::SigmoidOp>(op);
  TensorType output_type =
      dyn_cast<TensorType>(tf_sigmoid_op.getResult().getType());
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::SigmoidOp>(rewriter, op, output_type,
                                           tf_sigmoid_op.getX());

  return success();
}

LogicalResult ConvertTFTanhOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_tanh_op = cast<TF::TanhOp>(op);
  TensorType output_type =
      dyn_cast<TensorType>(tf_tanh_op.getResult().getType());
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::TanhOp>(rewriter, op, output_type,
                                        tf_tanh_op.getX());

  return success();
}

LogicalResult ConvertTFLeakyReluOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_leakyrelu_op = cast<TF::LeakyReluOp>(op);
  TensorType output_type =
      dyn_cast<TensorType>(tf_leakyrelu_op.getResult().getType());
  if (!output_type) return failure();

  // Implement LeakyRelu as element-wise:
  //   out = x > 0 ? x : alpha * x
  //
  // In TOSA ops:
  //
  //   const_zero = constant(0)
  //   a1 = mul(x, alpha)
  //   a2 = greater_equal(x, const_zero)
  //   out = select(a2, x, a1)
  //
  // If alpha can be constrained to 0.0 <= alpha <= 1.0, then
  // an alternative simpler lowering could be implemented with:
  //
  //   max(mul(x, alapha), x)
  //
  // But this alternative is not robust unless alpha meets those constraints.

  if (!output_type.getElementType().isF32()) {
    return rewriter.notifyMatchFailure(op, "only support F32");
  }

  FloatAttr tmpAttr = tf_leakyrelu_op.getAlphaAttr();
  // There is disagreement between the MLIR .td defaults and TF
  // documentation on 0.2 vs 0.3, but 0.2 will be used here.
  double alpha = 0.2;

  if (tmpAttr) {
    alpha = tmpAttr.getValueAsDouble();
  }

  int rank = output_type.getRank();
  Value const_zero = getTosaConstTensorSingleF32(rewriter, op, 0.0, rank);

  auto a1_mul = CreateMulOpAndInfer(
      rewriter, op, output_type, tf_leakyrelu_op.getFeatures(),
      getTosaConstTensorSingleF32(rewriter, op, alpha, rank));

  auto a2_ge = CreateOpAndInfer<tosa::GreaterEqualOp>(
      rewriter, op->getLoc(), UnrankedTensorType::get(rewriter.getI1Type()),
      tf_leakyrelu_op.getFeatures(), const_zero);

  auto a3_select = CreateOpAndInfer<tosa::SelectOp>(
      rewriter, op->getLoc(), output_type, a2_ge, tf_leakyrelu_op.getFeatures(),
      a1_mul.getResult());

  rewriter.replaceOp(op, {a3_select.getResult()});

  return success();
}

LogicalResult ConvertTFNegOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_neg_op = cast<TF::NegOp>(op);
  TensorType output_type =
      dyn_cast<TensorType>(tf_neg_op.getResult().getType());
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::NegateOp>(rewriter, op, output_type,
                                          tf_neg_op.getX());

  return success();
}

LogicalResult ConvertTFStopGradientOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_stopgrad_op = cast<TF::StopGradientOp>(op);
  TensorType output_type =
      dyn_cast<TensorType>(tf_stopgrad_op.getResult().getType());
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::IdentityOp>(rewriter, op, output_type,
                                            tf_stopgrad_op.getInput());

  return success();
}

LogicalResult ConvertTFReverseV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_reverse_op = cast<TF::ReverseV2Op>(op);
  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tf_reverse_op.getTensor().getType());
  TensorType output_type =
      dyn_cast<TensorType>(tf_reverse_op.getResult().getType());
  if (!input_type || !output_type) return failure();

  ElementsAttr axis_elems;
  if (!matchPattern(tf_reverse_op.getAxis(), m_Constant(&axis_elems)))
    return failure();

  auto input_rank = input_type.getShape().size();
  Value val = tf_reverse_op.getTensor();
  if (axis_elems.getNumElements() == 0) {
    auto identity_op = CreateOpAndInfer<tosa::IdentityOp>(
        rewriter, op->getLoc(), output_type, val);
    val = identity_op.getResult();
  } else {
    for (int i = 0; i < axis_elems.getNumElements(); i++) {
      int64_t axis_val = axis_elems.getValues<IntegerAttr>()[i].getInt();
      if (axis_val < 0) axis_val += input_rank;
      auto axis_attr = rewriter.getI32IntegerAttr(axis_val);
      auto reverse_op = CreateOpAndInfer<tosa::ReverseOp>(
          rewriter, op->getLoc(), output_type, val, axis_attr);

      val = reverse_op.getResult();
    }
  }

  rewriter.replaceOp(op, {val});

  return success();
}

LogicalResult ConvertTFFakeQuantWithMinMaxArgsOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_fakequant_op = cast<TF::FakeQuantWithMinMaxArgsOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_fakequant_op.getResult().getType());
  // Not a tensor output
  if (!output_type) return failure();

  std::optional<Value> result =
      convertFakeQuantOp(rewriter, op, output_type, tf_fakequant_op.getInputs(),
                         tf_fakequant_op.getMinAttr().getValueAsDouble(),
                         tf_fakequant_op.getMaxAttr().getValueAsDouble(),
                         tf_fakequant_op.getNumBitsAttr().getInt(),
                         tf_fakequant_op.getNarrowRangeAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFFakeQuantWithMinMaxVarsOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_fakequant_op = cast<TF::FakeQuantWithMinMaxVarsOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_fakequant_op.getResult().getType());
  // Not a tensor output
  if (!output_type) return failure();

  // Only support min/max that can be matched at compile time
  ElementsAttr min_elems, max_elems;
  if (!matchPattern(tf_fakequant_op.getMin(), m_Constant(&min_elems)))
    return failure();

  if (!matchPattern(tf_fakequant_op.getMax(), m_Constant(&max_elems)))
    return failure();

  if (min_elems.getNumElements() != 1 && max_elems.getNumElements() != 1)
    return failure();

  int64_t min_val = min_elems.getValues<IntegerAttr>()[0].getInt();
  int64_t max_val = max_elems.getValues<IntegerAttr>()[0].getInt();

  std::optional<Value> result = convertFakeQuantOp(
      rewriter, op, output_type, tf_fakequant_op.getInputs(), min_val, max_val,
      tf_fakequant_op.getNumBitsAttr().getInt(),
      tf_fakequant_op.getNarrowRangeAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLeftShiftOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_left_shift_op = cast<TF::LeftShiftOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_left_shift_op.getResult().getType());
  if (!output_type) return failure();

  Value x = tf_left_shift_op.getX();
  Value y = tf_left_shift_op.getY();

  RankedTensorType x_type = dyn_cast<RankedTensorType>(x.getType());
  RankedTensorType y_type = dyn_cast<RankedTensorType>(y.getType());
  if (!x_type || !y_type) return failure();

  CreateReplaceOpAndInfer<tosa::LogicalLeftShiftOp>(rewriter, op, output_type,
                                                    x, y);

  return success();
}

LogicalResult ConvertTFRightShiftOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  // Performs a logical shift for unsigned integer types, and an arithmetic
  // shift for signed integer types.
  auto tf_right_shift_op = cast<TF::RightShiftOp>(op);

  TensorType output_type =
      dyn_cast<TensorType>(tf_right_shift_op.getResult().getType());
  if (!output_type) return failure();

  Value x = tf_right_shift_op.getX();
  Value y = tf_right_shift_op.getY();

  RankedTensorType x_type = dyn_cast<RankedTensorType>(x.getType());
  RankedTensorType y_type = dyn_cast<RankedTensorType>(y.getType());
  if (!x_type || !y_type) return failure();

  Type output_element_type = output_type.getElementType();

  bool is_signed = false;
  if (!output_element_type.isUnsignedInteger()) is_signed = true;

  if (is_signed) {
    CreateReplaceOpAndInfer<tosa::ArithmeticRightShiftOp>(
        rewriter, op, output_type, x, y, false);
  } else {
    CreateReplaceOpAndInfer<tosa::LogicalRightShiftOp>(rewriter, op,
                                                       output_type, x, y);
  }

  return success();
}

LogicalResult ConvertTFOneHotOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_one_hot_op = cast<TF::OneHotOp>(op);

  ElementsAttr depth_elems;
  if (!matchPattern(tf_one_hot_op.getDepth(), m_Constant(&depth_elems)))
    return failure();
  int32_t depth = depth_elems.getValues<IntegerAttr>()[0].getInt();

  IntegerAttr axisAttr = tf_one_hot_op.getAxisAttr();
  int32_t axis = axisAttr.getInt();

  std::optional<Value> result = convertOneHotOp(
      rewriter, op, tf_one_hot_op.getResult(), tf_one_hot_op.getIndices(),
      tf_one_hot_op.getOnValue(), tf_one_hot_op.getOffValue(), depth, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFBatchMatMulV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_batch_matmul_op = cast<TF::BatchMatMulV2Op>(op);

  RankedTensorType x_type =
      dyn_cast<RankedTensorType>(tf_batch_matmul_op.getX().getType());
  RankedTensorType y_type =
      dyn_cast<RankedTensorType>(tf_batch_matmul_op.getY().getType());
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tf_batch_matmul_op.getResult().getType());

  if (!(x_type && y_type && output_type)) {
    return rewriter.notifyMatchFailure(op, "x/y/output not ranked tensors");
  }

  if (x_type.getRank() != y_type.getRank() ||
      x_type.getRank() != output_type.getRank()) {
    return rewriter.notifyMatchFailure(op, "x/y/output rank must match");
  }

  if (x_type.getRank() <= 2) {
    return rewriter.notifyMatchFailure(op, "x/y/output rank must > 2");
  }

  // Rank 3 batch matmul can be directly mapped to tosa.matmul trivially.
  if (x_type.getRank() == 3) {
    CreateReplaceOpAndInfer<tosa::MatMulOp>(rewriter, op, output_type,
                                            tf_batch_matmul_op.getX(),
                                            tf_batch_matmul_op.getY());
  } else {
    // 1. Reshape x from: (similar for y)
    //  [a0, a1, ... an, H, C] to [N, H, C].
    //  where N = a0 * a1 * ... * an.
    // 2. tosa.MatMul
    //  [N, H, C] * [N, C, W] -> [N, H, W].
    // 3. Reshape output from:
    //  [N, H, W] to [a0, a1, ... , an, H, W]
    int64_t rank = x_type.getRank();
    int64_t N = 1;
    for (int i = 0; i < (rank - 2); i++) {
      N *= x_type.getShape()[i];
    }
    int64_t H = x_type.getShape()[rank - 2];
    int64_t C = x_type.getShape()[rank - 1];
    int64_t W = y_type.getShape()[rank - 1];

    SmallVector<int64_t, 3> rank3_x_shape({N, H, C});
    SmallVector<int64_t, 3> rank3_y_shape({N, C, W});
    SmallVector<int64_t, 3> rank3_output_shape({N, H, W});

    RankedTensorType rank3_x_type = tensorflow::GetTypeFromTFTensorShape(
        rank3_x_shape, x_type.getElementType());
    RankedTensorType rank3_y_type = tensorflow::GetTypeFromTFTensorShape(
        rank3_y_shape, y_type.getElementType());
    RankedTensorType rank3_output_type = tensorflow::GetTypeFromTFTensorShape(
        rank3_output_shape, output_type.getElementType());

    auto rank3_x_shape_value = getTosaConstShape(rewriter, op->getLoc(), rank3_x_shape);
    auto op1_reshape_x = CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(), rank3_x_type, tf_batch_matmul_op.getX(),
        rank3_x_shape_value);

    auto rank3_y_shape_value = getTosaConstShape(rewriter, op->getLoc(), rank3_y_shape);
    auto op2_reshape_y = CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(), rank3_y_type, tf_batch_matmul_op.getY(),
        rank3_y_shape_value);

    auto op3_matmul_op1_op2 = CreateOpAndInfer<tosa::MatMulOp>(
        rewriter, op->getLoc(), rank3_output_type, op1_reshape_x.getResult(),
        op2_reshape_y.getResult());

    auto output_shape_value = getTosaConstShape(rewriter, op->getLoc(), output_type.getShape());
    CreateReplaceOpAndInfer<tosa::ReshapeOp>(
        rewriter, op, output_type, op3_matmul_op1_op2.getResult(),
        output_shape_value);
  }
  return success();
}

LogicalResult ConvertTFBroadcastToOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_broadcast_to_op = cast<TF::BroadcastToOp>(op);

  std::optional<Value> result =
      convertBroadcastToOp(rewriter, op, tf_broadcast_to_op.getInput(),
                           tf_broadcast_to_op.getShape());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::EqualOp>(op, rewriter);
}

LogicalResult ConvertTFGreaterOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::GreaterOp>(op, rewriter);
}

LogicalResult ConvertTFGreaterEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::GreaterEqualOp>(op, rewriter);
}

LogicalResult ConvertTFMaximumOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::MaximumOp>(op, rewriter);
}

LogicalResult ConvertTFMinimumOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::MinimumOp>(op, rewriter);
}

LogicalResult ConvertTFBitwiseOrOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::BitwiseOrOp>(op, rewriter);
}

LogicalResult ConvertTFBitwiseXorOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::BitwiseXorOp>(op, rewriter);
}

LogicalResult ConvertTFBitwiseAndOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::BitwiseAndOp>(op, rewriter);
}

LogicalResult ConvertTFLogicalAndOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::LogicalAndOp>(op, rewriter);
}

LogicalResult ConvertTFLogicalOrOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::LogicalOrOp>(op, rewriter);
}

LogicalResult ConvertTFPowOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::PowOp>(op, rewriter);
}

void LegalizeTF::runOnOperation() {
  auto* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  auto func = getOperation();
  populateLegalizeTFPatterns(ctx, patterns);

  if (ApplyPatternsWithShapeResolution(func, std::move(patterns)).failed()) {
    signalPassFailure();
  }
}

}  // anonymous namespace

void populateLegalizeTFPatterns(MLIRContext* ctx, RewritePatternSet& patterns) {
  // Add the generated patterns to the list.
  populateWithGenerated(patterns);
  patterns.add<ConvertTFMatMulOp>(ctx);
  patterns.add<ConvertTFReluOp>(ctx);
  patterns.add<ConvertTFRelu6Op>(ctx);
  patterns.add<ConvertTFEqualOp>(ctx);
  patterns.add<ConvertTFNotEqualOp>(ctx);
  patterns.add<ConvertTFGreaterOp>(ctx);
  patterns.add<ConvertTFGreaterEqualOp>(ctx);
  patterns.add<ConvertTFAddOp>(ctx);
  patterns.add<ConvertTFAddV2Op>(ctx);
  patterns.add<ConvertTFAddNOp>(ctx);
  patterns.add<ConvertTFSubOp>(ctx);
  patterns.add<ConvertTFMulOp>(ctx);
  patterns.add<ConvertTFSquareOp>(ctx);
  patterns.add<ConvertTFSquaredDifferenceOp>(ctx);
  patterns.add<ConvertTFSignOp>(ctx);
  patterns.add<ConvertTFRoundOp>(ctx);
  patterns.add<ConvertTFFloorDivOp>(ctx);
  patterns.add<ConvertTFFloorModOp>(ctx);
  patterns.add<ConvertTFAssertOp>(ctx);
  patterns.add<ConvertTFMaximumOp>(ctx);
  patterns.add<ConvertTFMinimumOp>(ctx);
  patterns.add<ConvertTFRealDivOp>(ctx);
  patterns.add<ConvertTFArgMaxOp>(ctx);
  patterns.add<ConvertTFAvgPoolOp>(ctx);
  patterns.add<ConvertTFMaxPoolOp>(ctx);
  patterns.add<ConvertTFConcatV2Op>(ctx);
  patterns.add<ConvertTFReshapeOp>(ctx);
  patterns.add<ConvertTFRankOp>(ctx);
  patterns.add<ConvertTFShapeOp>(ctx);
  patterns.add<ConvertTFExpandDimsOp>(ctx);
  patterns.add<ConvertTFSqueezeOp>(ctx);
  patterns.add<ConvertTFFillOp>(ctx);
  patterns.add<ConvertTFConv2DOp>(ctx);
  patterns.add<ConvertTFConv3DOp>(ctx);
  patterns.add<ConvertTFDepthwiseConv2dNativeOp>(ctx);
  patterns.add<ConvertTFConv2DBackpropInputOp>(ctx);
  patterns.add<ConvertTFEluOp>(ctx);
  patterns.add<ConvertTFSoftmaxOp>(ctx);
  patterns.add<ConvertTFLogSoftmaxOp>(ctx);
  patterns.add<ConvertTFAllOp>(ctx);
  patterns.add<ConvertTFAnyOp>(ctx);
  patterns.add<ConvertTFMaxOp>(ctx);
  patterns.add<ConvertTFMinOp>(ctx);
  patterns.add<ConvertTFMeanOp>(ctx);
  patterns.add<ConvertTFProdOp>(ctx);
  patterns.add<ConvertTFSumOp>(ctx);
  patterns.add<ConvertTFFusedBatchNormOp>(ctx);
  patterns.add<ConvertTFFusedBatchNormV3Op>(ctx);
  patterns.add<ConvertTFBiasAddOp>(ctx);
  patterns.add<ConvertTFSplitOp>(ctx);
  patterns.add<ConvertTFSplitVOp>(ctx);
  patterns.add<ConvertTFPackOp>(ctx);
  patterns.add<ConvertTFUnpackOp>(ctx);
  patterns.add<ConvertTFTransposeOp>(ctx);
  patterns.add<ConvertTFTileOp>(ctx);
  patterns.add<ConvertTFSliceOp>(ctx);
  patterns.add<ConvertTFStridedSliceOp>(ctx);
  patterns.add<ConvertTFLessOp>(ctx);
  patterns.add<ConvertTFLessEqualOp>(ctx);
  patterns.add<ConvertTFPadOp>(ctx);
  patterns.add<ConvertTFPadV2Op>(ctx);
  patterns.add<ConvertTFMirrorPadOp>(ctx);
  patterns.add<ConvertTFResizeBilinearOp>(ctx);
  patterns.add<ConvertTFResizeNearestNeighborOp>(ctx);
  patterns.add<ConvertTFGatherOp>(ctx);
  patterns.add<ConvertTFGatherV2Op>(ctx);
  patterns.add<ConvertTFGatherNdOp>(ctx);
  patterns.add<ConvertTFScatterNdOp>(ctx);
  patterns.add<ConvertTFSelectV2Op>(ctx);
  patterns.add<ConvertTFSpaceToDepthOp>(ctx);
  patterns.add<ConvertTFDepthToSpaceOp>(ctx);
  patterns.add<ConvertTFSinOp>(ctx);
  patterns.add<ConvertTFCosOp>(ctx);
  patterns.add<ConvertTFSpaceToBatchNDOp>(ctx);
  patterns.add<ConvertTFBatchToSpaceNDOp>(ctx);
  patterns.add<ConvertTFZerosLikeOp>(ctx);
  patterns.add<ConvertTFSigmoidOp>(ctx);
  patterns.add<ConvertTFTanhOp>(ctx);
  patterns.add<ConvertTFLeakyReluOp>(ctx);
  patterns.add<ConvertTFNegOp>(ctx);
  patterns.add<ConvertTFStopGradientOp>(ctx);
  patterns.add<ConvertTFReverseV2Op>(ctx);
  patterns.add<ConvertTFFakeQuantWithMinMaxArgsOp>(ctx);
  patterns.add<ConvertTFFakeQuantWithMinMaxVarsOp>(ctx);
  patterns.add<ConvertTFLeftShiftOp>(ctx);
  patterns.add<ConvertTFRightShiftOp>(ctx);
  patterns.add<ConvertTFOneHotOp>(ctx);
  patterns.add<ConvertTFBatchMatMulV2Op>(ctx);
  patterns.add<ConvertTFBroadcastToOp>(ctx);
  patterns.add<ConvertTFBitwiseOrOp>(ctx);
  patterns.add<ConvertTFBitwiseXorOp>(ctx);
  patterns.add<ConvertTFBitwiseAndOp>(ctx);
  patterns.add<ConvertTFLogicalAndOp>(ctx);
  patterns.add<ConvertTFLogicalOrOp>(ctx);
  patterns.add<ConvertTFPowOp>(ctx);
}

// Creates an instance of the TensorFlow dialect LegalizeTF pass.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFPass() {
  return std::make_unique<LegalizeTF>();
}

}  // namespace tosa

}  // namespace mlir
