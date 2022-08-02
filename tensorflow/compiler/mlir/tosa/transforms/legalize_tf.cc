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

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-legalize-tf"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {

// Performs lowering to TOSA dialect
class LegalizeTF : public TosaLegalizeTFPassBase<LegalizeTF> {
 public:
  explicit LegalizeTF() {}
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
DECL_CONVERT_OP(ResizeBilinear);
DECL_CONVERT_OP(ResizeNearestNeighbor);
DECL_CONVERT_OP(Gather);
DECL_CONVERT_OP(GatherV2);
DECL_CONVERT_OP(GatherNd);
DECL_CONVERT_OP(SelectV2);
DECL_CONVERT_OP(SpaceToDepth);
DECL_CONVERT_OP(DepthToSpace);
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
#undef DECL_CONVERT_OP

LogicalResult ConvertTFReluOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_relu_op = cast<TF::ReluOp>(op);

  TensorType output_type =
      tf_relu_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::ClampOp>(
      rewriter, op, output_type, tf_relu_op.features(),
      rewriter.getI64IntegerAttr(0),
      rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max()),
      rewriter.getF32FloatAttr(0.0f),
      rewriter.getF32FloatAttr(std::numeric_limits<float>::max()));
  return success();
}

LogicalResult ConvertTFRelu6Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_relu6_op = cast<TF::Relu6Op>(op);

  TensorType output_type =
      tf_relu6_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::ClampOp>(
      rewriter, op, output_type, tf_relu6_op.features(),
      rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(6),
      rewriter.getF32FloatAttr(0.0f), rewriter.getF32FloatAttr(6.0f));
  return success();
}

LogicalResult ConvertTFEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_equal_op = cast<TF::EqualOp>(op);

  TensorType output_type =
      tf_equal_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::EqualOp>(rewriter, op, output_type,
                                         tf_equal_op.x(), tf_equal_op.y());
  return success();
}

LogicalResult ConvertTFNotEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_not_equal_op = cast<TF::NotEqualOp>(op);

  TensorType output_type =
      tf_not_equal_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  auto op1_equal_in =
      CreateOpAndInfer<tosa::EqualOp>(rewriter, op->getLoc(), output_type,
                                      tf_not_equal_op.x(), tf_not_equal_op.y());

  auto op2_not_op1 = CreateOpAndInfer<tosa::LogicalNotOp>(
      rewriter, op->getLoc(), output_type, op1_equal_in.getResult());

  rewriter.replaceOp(op, {op2_not_op1.getResult()});

  return success();
}

LogicalResult ConvertTFGreaterOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_greater_op = cast<TF::GreaterOp>(op);

  TensorType output_type =
      tf_greater_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::GreaterOp>(
      rewriter, op, output_type, tf_greater_op.x(), tf_greater_op.y());
  return success();
}

LogicalResult ConvertTFGreaterEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_greater_equal_op = cast<TF::GreaterEqualOp>(op);

  TensorType output_type =
      tf_greater_equal_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::GreaterEqualOp>(rewriter, op, output_type,
                                                tf_greater_equal_op.x(),
                                                tf_greater_equal_op.y());
  return success();
}

LogicalResult ConvertTFAddOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_add_op = cast<TF::AddOp>(op);

  TensorType output_type =
      tf_add_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::AddOp>(rewriter, op, output_type, tf_add_op.x(),
                                       tf_add_op.y());
  return success();
}

LogicalResult ConvertTFAddV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_addv2_op = cast<TF::AddV2Op>(op);

  TensorType output_type =
      tf_addv2_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::AddOp>(rewriter, op, output_type,
                                       tf_addv2_op.x(), tf_addv2_op.y());
  return success();
}

// AddN is commutative
LogicalResult ConvertTFAddNOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_addn_op = cast<TF::AddNOp>(op);

  TensorType output_type =
      tf_addn_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  SmallVector<Value> inputs(tf_addn_op.inputs());

  assert(inputs.size() >= 2);

  auto newOp = CreateOpAndInfer<tosa::AddOp>(rewriter, op->getLoc(),
                                             output_type, inputs[0], inputs[1]);
  for (int i = 2; i < inputs.size(); i++) {
    newOp = CreateOpAndInfer<tosa::AddOp>(rewriter, op->getLoc(), output_type,
                                          inputs[i], newOp.getResult());
  }

  rewriter.replaceOp(op, {newOp.getResult()});

  return success();
}

LogicalResult ConvertTFSubOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_sub_op = cast<TF::SubOp>(op);

  TensorType output_type =
      tf_sub_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::SubOp>(rewriter, op, output_type, tf_sub_op.x(),
                                       tf_sub_op.y());
  return success();
}

LogicalResult ConvertTFMulOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_mul_op = cast<TF::MulOp>(op);

  llvm::Optional<Value> result = convertMultiplyOp(
      rewriter, op, tf_mul_op.getResult(), tf_mul_op.x(), tf_mul_op.y());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});
  return success();
}

LogicalResult ConvertTFSquareOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_square_op = cast<TF::SquareOp>(op);

  llvm::Optional<Value> result =
      convertMultiplyOp(rewriter, op, tf_square_op.getResult(),
                        tf_square_op.x(), tf_square_op.x());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});
  return success();
}

LogicalResult ConvertTFSquaredDifferenceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_squared_op = cast<TF::SquaredDifferenceOp>(op);

  llvm::Optional<Value> result =
      convertSquaredDifferenceOp(rewriter, op, tf_squared_op.getResult(),
                                 tf_squared_op.x(), tf_squared_op.y());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});
  return success();
}

LogicalResult ConvertTFRoundOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_round_op = cast<TF::RoundOp>(op);

  TensorType input_type = tf_round_op.x().getType().dyn_cast<TensorType>();
  if (!input_type) {
    return op->emitOpError("Round: input not tensor type");
  }

  if (input_type.getElementType().isa<FloatType>()) {
    llvm::Optional<Value> result =
        convertRoundOp(rewriter, op, tf_round_op.getResult(), tf_round_op.x());

    if (!result) return failure();

    rewriter.replaceOp(op, {result.getValue()});
    return success();

  } else {
    tf_round_op.replaceAllUsesWith(tf_round_op.x());
    return success();
  }
}

LogicalResult ConvertTFFloorDivOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_floordiv_op = cast<TF::FloorDivOp>(op);

  llvm::Optional<Value> result =
      convertFloorDivOp(rewriter, op, tf_floordiv_op.getResult(),
                        tf_floordiv_op.x(), tf_floordiv_op.y());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFFloorModOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_floormod_op = cast<TF::FloorModOp>(op);

  llvm::Optional<Value> result =
      convertFloorModOp(rewriter, op, tf_floormod_op.getResult(),
                        tf_floormod_op.x(), tf_floormod_op.y());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFAssertOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  op->dropAllReferences();
  op->erase();
  return success();
}

LogicalResult ConvertTFMaximumOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_maximum_op = cast<TF::MaximumOp>(op);

  TensorType output_type =
      tf_maximum_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::MaximumOp>(
      rewriter, op, output_type, tf_maximum_op.x(), tf_maximum_op.y());
  return success();
}

LogicalResult ConvertTFMinimumOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_minimum_op = cast<TF::MinimumOp>(op);

  TensorType output_type =
      tf_minimum_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::MinimumOp>(
      rewriter, op, output_type, tf_minimum_op.x(), tf_minimum_op.y());
  return success();
}

LogicalResult ConvertTFRealDivOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_div_op = cast<TF::RealDivOp>(op);

  TensorType y_type = tf_div_op.y().getType().dyn_cast<TensorType>();
  TensorType output_type =
      tf_div_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type || !y_type) return failure();

  Type element_type = output_type.getElementType();

  if (element_type.isa<IntegerType>()) {
    CreateReplaceOpAndInfer<tosa::DivOp>(rewriter, op, output_type,
                                         tf_div_op.x(), tf_div_op.y());
    return success();
  }

  auto reciprocal_op = CreateOpAndInfer<tosa::ReciprocalOp>(
      rewriter, op->getLoc(), tf_div_op.y().getType(), tf_div_op.y());

  auto mul_op = CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(),
                                              output_type, tf_div_op.x(),
                                              reciprocal_op.getResult(), 0);
  rewriter.replaceOp(op, {mul_op.getResult()});

  return success();
}

LogicalResult ConvertTFArgMaxOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_argmax_op = cast<TF::ArgMaxOp>(op);

  TensorType input_type = tf_argmax_op.input().getType().dyn_cast<TensorType>();
  TensorType output_type =
      tf_argmax_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type || !input_type) return failure();

  ElementsAttr axis_elems;
  if (!matchPattern(tf_argmax_op.dimension(), m_Constant(&axis_elems)))
    return failure();

  int32_t axis = axis_elems.getValues<IntegerAttr>()[0].getInt();
  if (axis < 0) {
    axis += input_type.getRank();
  }

  if (axis < 0 || axis >= input_type.getRank()) {
    return op->emitOpError("TFArgMax: invalid axis value");
  }

  IntegerAttr axis_attr = rewriter.getI64IntegerAttr(axis);

  CreateReplaceOpAndInfer<tosa::ArgMaxOp>(rewriter, op, output_type,
                                          tf_argmax_op.input(), axis_attr);

  return success();
}
LogicalResult ConvertTFAvgPoolOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_avgpool_op = cast<TF::AvgPoolOp>(op);

  RankedTensorType input_type =
      tf_avgpool_op.value().getType().dyn_cast<RankedTensorType>();
  RankedTensorType output_type =
      tf_avgpool_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!input_type || !output_type) return failure();

  auto tmpAttr = tf_avgpool_op.data_formatAttr();
  if (tmpAttr && tmpAttr.getValue().str() != "NHWC") return failure();

  ArrayAttr pad;
  ArrayAttr stride;
  ArrayAttr kernel;
  {
    auto tmpAttr = tf_avgpool_op.strides();
    if (!tmpAttr) {
      stride = rewriter.getI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t stride_h = tmpAttr[1].dyn_cast<IntegerAttr>().getInt();
      int64_t stride_w = tmpAttr[2].dyn_cast<IntegerAttr>().getInt();
      stride = rewriter.getI64ArrayAttr({stride_h, stride_w});
    }
  }
  {
    auto tmpAttr = tf_avgpool_op.ksize();
    if (!tmpAttr) {
      kernel = rewriter.getI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t kernel_h = tmpAttr[1].dyn_cast<IntegerAttr>().getInt();
      int64_t kernel_w = tmpAttr[2].dyn_cast<IntegerAttr>().getInt();
      kernel = rewriter.getI64ArrayAttr({kernel_h, kernel_w});
    }
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tf_avgpool_op.padding().str(), &tf_pad).ok())
      return failure();

    ArrayAttr dilation =
        rewriter.getI64ArrayAttr({1, 1});  // Pooling has no non-unit dilation

    SmallVector<int64_t, 4> i64array;

    for (auto& elem : tf_avgpool_op.ksize()) {
      int64_t value = elem.dyn_cast<IntegerAttr>().getInt();
      i64array.emplace_back(value);
    }

    RankedTensorType filter_type = RankedTensorType::get(
        llvm::makeArrayRef(i64array), rewriter.getIntegerType(64));

    if (!getPaddingValuesFromPadType(
            tf_pad,
            tensorflow::FORMAT_NHWC,  // TFLite only supports this
            1,                        // tensorflow::FORMAT_OHWI,
            input_type, filter_type, stride, dilation, rewriter, pad))
      return failure();
  }

  CreateReplaceOpAndInfer<tosa::AvgPool2dOp>(
      rewriter, op, output_type, tf_avgpool_op.value(), kernel, stride, pad);
  return success();
}

LogicalResult ConvertTFMaxPoolOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_maxpool_op = cast<TF::MaxPoolOp>(op);

  RankedTensorType input_type =
      tf_maxpool_op.input().getType().dyn_cast<RankedTensorType>();
  RankedTensorType output_type =
      tf_maxpool_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!input_type || !output_type) return failure();

  auto tmpAttr = tf_maxpool_op.data_formatAttr();
  if (tmpAttr && tmpAttr.getValue().str() != "NHWC") return failure();

  ArrayAttr pad;
  ArrayAttr stride;
  ArrayAttr kernel;
  {
    auto tmpAttr = tf_maxpool_op.strides();
    if (!tmpAttr) {
      stride = rewriter.getI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t stride_h = tmpAttr[1].dyn_cast<IntegerAttr>().getInt();
      int64_t stride_w = tmpAttr[2].dyn_cast<IntegerAttr>().getInt();
      stride = rewriter.getI64ArrayAttr({stride_h, stride_w});
    }
  }
  {
    auto tmpAttr = tf_maxpool_op.ksize();
    if (!tmpAttr) {
      kernel = rewriter.getI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t kernel_h = tmpAttr[1].dyn_cast<IntegerAttr>().getInt();
      int64_t kernel_w = tmpAttr[2].dyn_cast<IntegerAttr>().getInt();
      kernel = rewriter.getI64ArrayAttr({kernel_h, kernel_w});
    }
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tf_maxpool_op.padding().str(), &tf_pad).ok())
      return failure();

    // Pooling has no non-unit dilation
    ArrayAttr dilation = rewriter.getI64ArrayAttr({1, 1});

    SmallVector<int64_t, 4> i64array;

    for (auto& elem : tf_maxpool_op.ksize()) {
      int64_t value = elem.dyn_cast<IntegerAttr>().getInt();
      i64array.emplace_back(value);
    }

    RankedTensorType filter_type = RankedTensorType::get(
        llvm::makeArrayRef(i64array), rewriter.getIntegerType(64));

    if (!getPaddingValuesFromPadType(
            tf_pad,
            tensorflow::FORMAT_NHWC,  // TFLite only supports this
            1,                        // tensorflow::FORMAT_OHWI,
            input_type, filter_type, stride, dilation, rewriter, pad))
      return failure();
  }

  CreateReplaceOpAndInfer<tosa::MaxPool2dOp>(
      rewriter, op, output_type, tf_maxpool_op.input(), kernel, stride, pad);
  return success();
}

LogicalResult ConvertTFConcatV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_concatv2_op = cast<TF::ConcatV2Op>(op);
  SmallVector<Value> values(tf_concatv2_op.values());

  ElementsAttr axis_elems;
  if (!matchPattern(tf_concatv2_op.axis(), m_Constant(&axis_elems)))
    return failure();

  int32_t axis = axis_elems.getValues<IntegerAttr>()[0].getInt();

  llvm::Optional<Value> result =
      convertConcatV2Op(rewriter, op, tf_concatv2_op.getResult(), values, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFReshapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_reshape_op = cast<TF::ReshapeOp>(op);

  RankedTensorType output_type =
      tf_reshape_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  // Regular way to match tensor as element attribute doesn't always work
  // use output_type.getShape() which is more stable
  SmallVector<int64_t> shape_vals;
  for (int i = 0; i < output_type.getShape().size(); i++) {
    shape_vals.push_back(output_type.getShape()[i]);
  }
  ArrayAttr shape_attr = rewriter.getI64ArrayAttr(shape_vals);

  CreateReplaceOpAndInfer<tosa::ReshapeOp>(rewriter, op, output_type,
                                           tf_reshape_op.tensor(), shape_attr);
  return success();
}

LogicalResult ConvertTFRankOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_rank_op = cast<TF::RankOp>(op);

  RankedTensorType input_type =
      tf_rank_op.input().getType().dyn_cast<RankedTensorType>();
  if (!input_type) return failure();

  int32_t rank = input_type.getRank();

  RankedTensorType rank_type =
      RankedTensorType::get({1}, rewriter.getIntegerType(32));
  auto rank_attr = DenseElementsAttr::get(rank_type, {rank});
  auto rank_const = CreateOpAndInfer<tosa::ConstOp>(rewriter, op->getLoc(),
                                                    rank_type, rank_attr);

  rewriter.replaceOp(op, {rank_const.getResult()});

  return success();
}

LogicalResult ConvertTFShapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_shape_op = cast<TF::ShapeOp>(op);

  RankedTensorType output_type =
      tf_shape_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  RankedTensorType input_type =
      tf_shape_op.input().getType().dyn_cast<RankedTensorType>();
  if (!input_type) return failure();

  auto input_shape = input_type.getShape();

  SmallVector<int32_t> shape_arr;
  for (int i = 0; i < input_shape.size(); i++) {
    shape_arr.emplace_back(input_shape[i]);
  }

  RankedTensorType shape_type = RankedTensorType::get(
      {static_cast<int32_t>(shape_arr.size())}, rewriter.getIntegerType(32));
  auto shape_attr =
      DenseElementsAttr::get(shape_type, llvm::makeArrayRef(shape_arr));
  auto shape_const = CreateOpAndInfer<tosa::ConstOp>(rewriter, op->getLoc(),
                                                     shape_type, shape_attr);

  rewriter.replaceOp(op, {shape_const.getResult()});

  return success();
}

LogicalResult ConvertTFExpandDimsOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_expanddims_op = cast<TF::ExpandDimsOp>(op);

  llvm::Optional<Value> result =
      convertExpandDimsOp(rewriter, op, tf_expanddims_op.getResult(),
                          tf_expanddims_op.input(), tf_expanddims_op.dim());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFSqueezeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_squeeze_op = cast<TF::SqueezeOp>(op);

  // Copy squeeze_dims into int32_t array
  auto squeeze_dims_attr = tf_squeeze_op.squeeze_dimsAttr();
  SmallVector<int32_t> squeeze_dims;
  for (auto& squeeze_dim : squeeze_dims_attr) {
    squeeze_dims.emplace_back(squeeze_dim.dyn_cast<IntegerAttr>().getInt());
  }

  llvm::Optional<Value> result =
      convertSqueezeOp(rewriter, op, tf_squeeze_op.getResult(),
                       tf_squeeze_op.input(), squeeze_dims);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFFillOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_fill_op = cast<TF::FillOp>(op);

  RankedTensorType output_type =
      tf_fill_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  ElementsAttr dims_elems;
  if (!matchPattern(tf_fill_op.dims(), m_Constant(&dims_elems)))
    return failure();
  SmallVector<int64_t> dims_vals;
  uint32_t total_size = 1;
  for (int i = 0; i < dims_elems.getNumElements(); i++) {
    dims_vals.push_back(dims_elems.getValues<IntegerAttr>()[i].getInt());
    total_size *= dims_vals[i];
  }

  ElementsAttr value_elem;
  if (!matchPattern(tf_fill_op.value(), m_Constant(&value_elem)))
    return failure();

  RankedTensorType fill_type = RankedTensorType::get(
      ArrayRef<int64_t>(dims_vals), value_elem.getType().getElementType());
  DenseElementsAttr fill_attr;

  // Convert to a compatible zero type
  if (value_elem.getType().getElementType().isa<FloatType>()) {
    SmallVector<float> fill_arr(
        total_size,
        value_elem.getValues<FloatAttr>()[0].getValue().convertToFloat());
    fill_attr = DenseElementsAttr::get(fill_type, llvm::makeArrayRef(fill_arr));
  } else {
    SmallVector<int32_t> fill_arr(
        total_size,
        value_elem.getValues<IntegerAttr>()[0].getValue().getLimitedValue());
    fill_attr = DenseElementsAttr::get(fill_type, llvm::makeArrayRef(fill_arr));
  }
  auto fill_const_op = CreateOpAndInfer<tosa::ConstOp>(rewriter, op->getLoc(),
                                                       fill_type, fill_attr);
  rewriter.replaceOp(op, {fill_const_op.getResult()});

  return success();
}

LogicalResult ConvertTFConv2DOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_conv2d_op = cast<TF::Conv2DOp>(op);

  RankedTensorType filter_type =
      tf_conv2d_op.filter().getType().dyn_cast<RankedTensorType>();
  RankedTensorType output_type =
      tf_conv2d_op.getResult().getType().dyn_cast<RankedTensorType>();

  // Set up a zero attr for subsequent pattern replacement if required
  auto bias_dim = filter_type.getShape().back();
  RankedTensorType bias_type =
      RankedTensorType::get({bias_dim}, filter_type.getElementType());
  auto bias_attr = rewriter.getZeroAttr(bias_type);
  auto bias = CreateOpAndInfer<tosa::ConstOp>(rewriter, op->getLoc(), bias_type,
                                              bias_attr.cast<ElementsAttr>());

  llvm::Optional<Value> result = convertTFConv2DCommon(
      rewriter, op, output_type, tf_conv2d_op.input(), tf_conv2d_op.filter(),
      bias, tf_conv2d_op.strides(), tf_conv2d_op.dilations(),
      tf_conv2d_op.explicit_paddings(), tf_conv2d_op.padding(),
      tf_conv2d_op.data_format());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFDepthwiseConv2dNativeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_dwconv2d_op = cast<TF::DepthwiseConv2dNativeOp>(op);

  RankedTensorType input_type =
      tf_dwconv2d_op.input().getType().dyn_cast<RankedTensorType>();
  RankedTensorType filter_type =
      tf_dwconv2d_op.filter().getType().dyn_cast<RankedTensorType>();
  RankedTensorType output_type =
      tf_dwconv2d_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!input_type) return failure();
  if (!output_type) return failure();

  // Set up a zero attr for subsequent pattern replacement if required
  if (!filter_type) {
    return op->emitOpError("DepthwiseConv2d: filter type unranked tensor");
  }

  auto tmpAttr = tf_dwconv2d_op.data_formatAttr();
  if (tmpAttr && tmpAttr.getValue().str() != "NHWC") return failure();

  ArrayAttr stride;
  ArrayAttr dilation;
  ArrayAttr pad;
  {
    auto tmpAttr = tf_dwconv2d_op.strides();
    if (!tmpAttr) {
      stride = rewriter.getI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t stride_h = tmpAttr[1].dyn_cast<IntegerAttr>().getInt();
      int64_t stride_w = tmpAttr[2].dyn_cast<IntegerAttr>().getInt();
      stride = rewriter.getI64ArrayAttr({stride_h, stride_w});
    }
  }
  {
    auto tmpAttr = tf_dwconv2d_op.dilations();
    if (!tmpAttr) {
      dilation = rewriter.getI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t dilation_h = tmpAttr[1].dyn_cast<IntegerAttr>().getInt();
      int64_t dilation_w = tmpAttr[2].dyn_cast<IntegerAttr>().getInt();
      dilation = rewriter.getI64ArrayAttr({dilation_h, dilation_w});
    }
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tf_dwconv2d_op.padding().str(), &tf_pad).ok())
      return failure();

    tensorflow::TensorFormat data_format_tf;
    if (!FormatFromString(tf_dwconv2d_op.data_format().str(), &data_format_tf))
      return failure();

    if (tf_pad == tensorflow::Padding::EXPLICIT) {
      pad = getPaddingValuesFromExplicitPadAttr(
          tf_dwconv2d_op.explicit_paddings(), data_format_tf, rewriter);
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
  RankedTensorType bias_type =
      RankedTensorType::get({bias_dim}, filter_type.getElementType());
  auto bias_attr = rewriter.getZeroAttr(bias_type);
  auto bias = CreateOpAndInfer<tosa::ConstOp>(rewriter, op->getLoc(), bias_type,
                                              bias_attr.cast<ElementsAttr>());

  CreateReplaceOpAndInfer<tosa::DepthwiseConv2DOp>(
      rewriter, op, output_type, tf_dwconv2d_op.input(),
      tf_dwconv2d_op.filter(), bias, pad, stride, dilation);
  return success();
}

LogicalResult ConvertTFConv2DBackpropInputOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_conv_op = cast<TF::Conv2DBackpropInputOp>(op);

  RankedTensorType input_type =
      tf_conv_op.out_backprop().getType().dyn_cast<RankedTensorType>();
  RankedTensorType filter_type =
      tf_conv_op.filter().getType().dyn_cast<RankedTensorType>();
  RankedTensorType output_type =
      tf_conv_op.getResult().getType().dyn_cast<RankedTensorType>();
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
  llvm::Optional<Value> a1_filter_transpose_perm = getConstTensor<int32_t>(
      rewriter, op, /*vec=*/{2, 0, 1, 3}, /*shape=*/{4});

  if (!a1_filter_transpose_perm) return failure();

  auto a1_filter_transpose_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(),
      RankedTensorType::get(ArrayRef<int64_t>(a1_transpose_dims),
                            filter_type.getElementType()),
      tf_conv_op.filter(), a1_filter_transpose_perm.getValue());

  ArrayAttr stride;
  ArrayAttr outpad;
  ArrayAttr output_shape;
  {
    auto tmpAttr = tf_conv_op.strides();
    if (!tmpAttr) {
      stride = rewriter.getI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t stride_h = tmpAttr[1].dyn_cast<IntegerAttr>().getInt();
      int64_t stride_w = tmpAttr[2].dyn_cast<IntegerAttr>().getInt();
      stride = rewriter.getI64ArrayAttr({stride_h, stride_w});
    }
  }
  {
    auto tmpAttr = tf_conv_op.dilations();
    if (tmpAttr) {
      // Note: hardcoded to NHWC for now
      int64_t dilation_h = tmpAttr[1].dyn_cast<IntegerAttr>().getInt();
      int64_t dilation_w = tmpAttr[2].dyn_cast<IntegerAttr>().getInt();
      // TOSA transpose_conv2d does not support non-unit dilation
      if (dilation_h != 1 || dilation_w != 1) return failure();
    }
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tf_conv_op.padding().str(), &tf_pad).ok())
      return failure();

    tensorflow::TensorFormat data_format_tf;
    if (!FormatFromString(tf_conv_op.data_format().str(), &data_format_tf))
      return failure();

    if (tf_pad == tensorflow::Padding::EXPLICIT) {
      outpad = getPaddingValuesFromExplicitPadAttr(
          tf_conv_op.explicit_paddings(), data_format_tf, rewriter);
    } else {
      if (!getTransposeConv2dPaddingValues(tf_pad, data_format_tf,
                                           0,  // tensorflow::FORMAT_HWIO,
                                           input_type, filter_type, output_type,
                                           stride, rewriter, outpad))
        return failure();
    }
  }
  {
    ElementsAttr output_shape_elems;
    // Match from input_sizes tensor first.
    if (matchPattern(tf_conv_op.input_sizes(),
                     m_Constant(&output_shape_elems))) {
      SmallVector<int64_t> shape_vec;
      for (int i = 0; i < output_shape_elems.getNumElements(); i++)
        shape_vec.push_back(
            output_shape_elems.getValues<IntegerAttr>()[i].getInt());
      output_shape = rewriter.getI64ArrayAttr(shape_vec);
    } else {
      // Use output tensor's shape otherwise.
      output_shape = rewriter.getI64ArrayAttr(output_type.getShape());
    }
  }

  int output_channel = output_type.getShape()[3];
  SmallVector<float> vec(output_channel, 0.0f);
  llvm::Optional<Value> zero_bias =
      getConstTensor<float>(rewriter, op, vec, {output_channel});

  if (!zero_bias) return failure();

  CreateReplaceOpAndInfer<tosa::TransposeConv2DOp>(
      rewriter, op, output_type, tf_conv_op.out_backprop(),
      a1_filter_transpose_op.getResult(), zero_bias.getValue(), outpad, stride,
      output_shape);

  return success();
}

LogicalResult ConvertTFAllOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_all_op = cast<TF::AllOp>(op);

  RankedTensorType output_type =
      tf_all_op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_all_op.reduction_indices(), m_Constant(&axes_elems)))
    return failure();

  llvm::Optional<Value> result = convertReduceAllOp(
      rewriter, op, output_type, tf_all_op.input(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFAnyOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_any_op = cast<TF::AnyOp>(op);

  RankedTensorType output_type =
      tf_any_op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_any_op.reduction_indices(), m_Constant(&axes_elems)))
    return failure();

  llvm::Optional<Value> result = convertReduceAnyOp(
      rewriter, op, output_type, tf_any_op.input(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFMaxOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_max_op = cast<TF::MaxOp>(op);

  RankedTensorType output_type =
      tf_max_op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_max_op.reduction_indices(), m_Constant(&axes_elems)))
    return failure();

  llvm::Optional<Value> result = convertReduceMaxOp(
      rewriter, op, output_type, tf_max_op.input(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFMinOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_min_op = cast<TF::MinOp>(op);

  RankedTensorType output_type =
      tf_min_op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_min_op.reduction_indices(), m_Constant(&axes_elems)))
    return failure();

  llvm::Optional<Value> result = convertReduceMinOp(
      rewriter, op, output_type, tf_min_op.input(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFMeanOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_mean_op = cast<TF::MeanOp>(op);

  RankedTensorType output_type =
      tf_mean_op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_mean_op.reduction_indices(), m_Constant(&axes_elems)))
    return failure();

  llvm::Optional<Value> result = convertReduceMeanOp(
      rewriter, op, output_type, tf_mean_op.input(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFProdOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_prod_op = cast<TF::ProdOp>(op);

  RankedTensorType output_type =
      tf_prod_op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_prod_op.reduction_indices(), m_Constant(&axes_elems)))
    return failure();

  llvm::Optional<Value> result = convertReduceProdOp(
      rewriter, op, output_type, tf_prod_op.input(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFSumOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_sum_op = cast<TF::SumOp>(op);

  RankedTensorType output_type =
      tf_sum_op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tf_sum_op.reduction_indices(), m_Constant(&axes_elems)))
    return failure();

  llvm::Optional<Value> result = convertReduceSumOp(
      rewriter, op, output_type, tf_sum_op.input(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFEluOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_elu_op = cast<TF::EluOp>(op);

  llvm::Optional<Value> result =
      convertEluOp(rewriter, op, tf_elu_op.getResult(), tf_elu_op.features());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFSoftmaxOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_softmax_op = cast<TF::SoftmaxOp>(op);

  llvm::Optional<Value> result =
      convertSoftmaxOp(rewriter, op, tf_softmax_op.getResult(),
                       tf_softmax_op.logits(), /*beta=*/1.0);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFLogSoftmaxOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_logsoftmax_op = cast<TF::LogSoftmaxOp>(op);

  llvm::Optional<Value> result = convertLogSoftmaxOp(
      rewriter, op, tf_logsoftmax_op.getResult(), tf_logsoftmax_op.logits());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFFusedBatchNormOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_batchnorm_op = cast<TF::FusedBatchNormOp>(op);

  RankedTensorType output_type =
      tf_batchnorm_op.getResult(0).getType().dyn_cast<RankedTensorType>();
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

  RankedTensorType mean_type =
      tf_batchnorm_op.mean().getType().dyn_cast<RankedTensorType>();
  RankedTensorType variance_type =
      tf_batchnorm_op.variance().getType().dyn_cast<RankedTensorType>();
  if (!variance_type || !mean_type) return failure();

  Value mean_val, variance_val;

  if (mean_type.getNumElements() == 0) {
    mean_val = getTosaConstTensorSingleF32(rewriter, tf_batchnorm_op, 0);
  } else {
    mean_val = tf_batchnorm_op.mean();
  }

  if (variance_type.getNumElements() == 0) {
    variance_val = getTosaConstTensorSingleF32(rewriter, tf_batchnorm_op, 1.0);
  } else {
    variance_val = tf_batchnorm_op.variance();
  }

  RankedTensorType epsilon_type =
      RankedTensorType::get({1}, variance_type.getElementType());
  auto epsilon_attr =
      DenseFPElementsAttr::get(epsilon_type, {tf_batchnorm_op.epsilon()});
  auto epsilon_const = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), epsilon_type, epsilon_attr);

  auto op1_sub_input_mean = CreateOpAndInfer<tosa::SubOp>(
      rewriter, op->getLoc(), tf_batchnorm_op.getResult(0).getType(),
      tf_batchnorm_op.x(), mean_val);

  auto op2_add_var_epsilon = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), variance_val.getType(), variance_val,
      epsilon_const.getResult());

  auto op3_rsqrt_op2 = CreateOpAndInfer<tosa::RsqrtOp>(
      rewriter, op->getLoc(), variance_val.getType(),
      op2_add_var_epsilon.getResult());

  auto op4_mul_op1_op3 = CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(), tf_batchnorm_op.getResult(0).getType(),
      op1_sub_input_mean.getResult(), op3_rsqrt_op2.getResult(), 0);

  auto op5_mul_op4_scale = CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(), tf_batchnorm_op.getResult(0).getType(),
      op4_mul_op1_op3.getResult(), tf_batchnorm_op.scale(), 0);

  auto op6_add_op5_offset = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), tf_batchnorm_op.getResult(0).getType(),
      op5_mul_op4_scale.getResult(), tf_batchnorm_op.offset());

  rewriter.replaceOp(op, {op6_add_op5_offset.getResult()});
  return success();
}

LogicalResult ConvertTFFusedBatchNormV3Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_batchnorm_op = cast<TF::FusedBatchNormV3Op>(op);

  if (tf_batchnorm_op.is_training())
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
      tf_batchnorm_op.getResult(0).getType().dyn_cast<RankedTensorType>();
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

  auto op1_sub_input_mean = CreateOpAndInfer<tosa::SubOp>(
      rewriter, op->getLoc(), tf_batchnorm_op.getResult(0).getType(),
      tf_batchnorm_op.x(), tf_batchnorm_op.mean());

  RankedTensorType variance_type =
      tf_batchnorm_op.variance().getType().dyn_cast<RankedTensorType>();
  if (!variance_type) return failure();

  auto epsilon_type =
      RankedTensorType::get({1}, variance_type.getElementType());
  auto epsilon_attr =
      DenseFPElementsAttr::get(epsilon_type, {tf_batchnorm_op.epsilon()});
  auto epsilon_const = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), epsilon_type, epsilon_attr);

  auto op2_add_var_epsilon = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), tf_batchnorm_op.variance().getType(),
      tf_batchnorm_op.variance(), epsilon_const);

  auto op3_rsqrt_op2 = CreateOpAndInfer<tosa::RsqrtOp>(
      rewriter, op->getLoc(), tf_batchnorm_op.variance().getType(),
      op2_add_var_epsilon.getResult());

  auto op4_mul_op1_op3 = CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(), tf_batchnorm_op.getResult(0).getType(),
      op1_sub_input_mean.getResult(), op3_rsqrt_op2.getResult(), 0);

  auto op5_mul_op4_scale = CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(), tf_batchnorm_op.getResult(0).getType(),
      op4_mul_op1_op3.getResult(), tf_batchnorm_op.scale(), 0);

  auto op6_add_op5_offset = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), tf_batchnorm_op.getResult(0).getType(),
      op5_mul_op4_scale.getResult(), tf_batchnorm_op.offset());

  llvm::SmallVector<Value> replacements = {
      op6_add_op5_offset.getResult(), tf_batchnorm_op.mean(),
      tf_batchnorm_op.variance(),
      // The last three are reserved spaces and have no purpose currently.
      tf_batchnorm_op.mean(), tf_batchnorm_op.variance(),
      tf_batchnorm_op.variance()};
  rewriter.replaceOp(op, replacements);
  return success();
}

LogicalResult ConvertTFBiasAddOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_biasadd_op = cast<TF::BiasAddOp>(op);

  RankedTensorType output_type =
      tf_biasadd_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  auto add_op = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), output_type, tf_biasadd_op.value(),
      tf_biasadd_op.bias());

  rewriter.replaceOp(op, {add_op.getResult()});
  return success();
}

LogicalResult ConvertTFSliceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_slice_op = cast<TF::SliceOp>(op);

  RankedTensorType output_type =
      tf_slice_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  ElementsAttr begin_elems, size_elems;

  SmallVector<int64_t> begin_vals, size_vals;

  // Assuming begin is always compile-time constant
  if (!matchPattern(tf_slice_op.begin(), m_Constant(&begin_elems))) {
    return op->emitOpError("TF::Slice error: begin is not constant");
  }

  for (int i = 0; i < begin_elems.getNumElements(); i++)
    begin_vals.push_back(begin_elems.getValues<IntegerAttr>()[i].getInt());

  // Try to match size as compile-time constant first,
  // if this fails, use the output tensor shape instead.
  if (matchPattern(tf_slice_op.size(), m_Constant(&size_elems))) {
    for (int i = 0; i < size_elems.getNumElements(); i++)
      size_vals.push_back(size_elems.getValues<IntegerAttr>()[i].getInt());
  } else {
    size_vals.assign(output_type.getShape().begin(),
                     output_type.getShape().end());
  }

  ArrayAttr begin = rewriter.getI64ArrayAttr(begin_vals);
  ArrayAttr size = rewriter.getI64ArrayAttr(size_vals);

  CreateReplaceOpAndInfer<tosa::SliceOp>(rewriter, op, output_type,
                                         tf_slice_op.input(), begin, size);
  return success();
}

LogicalResult ConvertTFTileOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_tile_op = cast<TF::TileOp>(op);

  RankedTensorType output_type =
      tf_tile_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  ElementsAttr multiples_elems;
  if (!matchPattern(tf_tile_op.multiples(), m_Constant(&multiples_elems)))
    return failure();
  SmallVector<int64_t> multiples_vals;
  for (int i = 0; i < multiples_elems.getNumElements(); i++)
    multiples_vals.push_back(
        multiples_elems.getValues<IntegerAttr>()[i].getInt());

  ArrayAttr multiples_attr = rewriter.getI64ArrayAttr(multiples_vals);

  CreateReplaceOpAndInfer<tosa::TileOp>(rewriter, op, output_type,
                                        tf_tile_op.input(), multiples_attr);

  return success();
}

LogicalResult ConvertTFTransposeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_transpose_op = cast<TF::TransposeOp>(op);

  TensorType output_type =
      tf_transpose_op.getResult().getType().dyn_cast<TensorType>();
  // Not a ranked tensor output
  if (!output_type) {
    return failure();
  }

  CreateReplaceOpAndInfer<tosa::TransposeOp>(
      rewriter, op, output_type, tf_transpose_op.x(), tf_transpose_op.perm());

  return success();
}

LogicalResult ConvertTFPackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_pack_op = cast<TF::PackOp>(op);

  SmallVector<Value> inputs(tf_pack_op.values());

  assert(inputs.size() >= 2);

  IntegerAttr axis_attr = tf_pack_op.axisAttr();
  if (!axis_attr) axis_attr = rewriter.getI64IntegerAttr(0);

  int32_t axis_i32 = axis_attr.getInt();

  llvm::Optional<Value> result =
      convertPackOp(rewriter, op, tf_pack_op.getResult(), inputs, axis_i32);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFUnpackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_unpack_op = cast<TF::UnpackOp>(op);

  IntegerAttr axis_attr;
  {
    auto tmpAttr = tf_unpack_op.axisAttr();
    if (!tmpAttr) tmpAttr = rewriter.getI64IntegerAttr(0);
    axis_attr = tmpAttr;
  }
  int32_t axis_i32 = axis_attr.getInt();

  llvm::Optional<SmallVector<Value>> results =
      convertUnpackOp(rewriter, op, tf_unpack_op.value(), axis_i32);

  if (!results) return failure();

  rewriter.replaceOp(op, results.getValue());

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
  if (matchPattern(tf_split_op.split_dim(), m_Constant(&axisAttrElems))) {
    axis = axisAttrElems.getValues<IntegerAttr>()[0].getInt();
  }

  llvm::Optional<SmallVector<Value>> results =
      convertSplitOp(rewriter, op, tf_split_op.getResult(0),
                     tf_split_op.value(), num_split, axis);

  if (!results) return failure();

  rewriter.replaceOp(op, results.getValue());

  return success();
}

// TFSplitV op splits based on a vector of sizes
LogicalResult ConvertTFSplitVOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_splitv_op = cast<TF::SplitVOp>(op);

  // Get the size_splits array
  SmallVector<int32_t> size_split;
  ElementsAttr size_split_elems;
  if (!matchPattern(tf_splitv_op.size_splits(),
                    m_Constant(&size_split_elems))) {
    return failure();
  }

  for (int i = 0; i < size_split_elems.getNumElements(); i++) {
    size_split.push_back(size_split_elems.getValues<IntegerAttr>()[i].getInt());
  }

  // Get the axis
  ElementsAttr axisAttrElems;
  if (!matchPattern(tf_splitv_op.split_dim(), m_Constant(&axisAttrElems))) {
    return op->emitOpError("Cannot read split_dim elems");
  }

  int32_t axis = axisAttrElems.getValues<IntegerAttr>()[0].getInt();

  llvm::Optional<SmallVector<Value>> results =
      convertSplitVOp(rewriter, op, tf_splitv_op.getResult(0),
                      tf_splitv_op.value(), size_split, axis);

  if (!results) return failure();

  rewriter.replaceOp(op, results.getValue());

  return success();
}

LogicalResult ConvertTFLessOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_less_op = cast<TF::LessOp>(op);

  TensorType output_type =
      tf_less_op.getResult().getType().dyn_cast<TensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  // less(x, y) is not(greater_equal(x, y))
  auto greater_equal_op = CreateOpAndInfer<tosa::GreaterEqualOp>(
      rewriter, op->getLoc(), output_type, tf_less_op.x(), tf_less_op.y());

  auto not_op = CreateOpAndInfer<tosa::LogicalNotOp>(
      rewriter, op->getLoc(), output_type, greater_equal_op.getResult());

  rewriter.replaceOp(op, {not_op.getResult()});
  return success();
}

LogicalResult ConvertTFLessEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_less_equal_op = cast<TF::LessEqualOp>(op);

  TensorType output_type =
      tf_less_equal_op.getResult().getType().dyn_cast<TensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  // less_equal(x, y) is not(greater(x, y))
  auto greater_op = CreateOpAndInfer<tosa::GreaterOp>(
      rewriter, op->getLoc(), output_type, tf_less_equal_op.x(),
      tf_less_equal_op.y());
  auto not_op = CreateOpAndInfer<tosa::LogicalNotOp>(
      rewriter, op->getLoc(), output_type, greater_op.getResult());

  rewriter.replaceOp(op, {not_op.getResult()});
  return success();
}

LogicalResult ConvertTFPadOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_pad_op = cast<TF::PadOp>(op);

  TensorType output_type =
      tf_pad_op.getResult().getType().dyn_cast<TensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  auto pad_op =
      CreateOpAndInfer<tosa::PadOp>(rewriter, op->getLoc(), output_type,
                                    tf_pad_op.input(), tf_pad_op.paddings());

  rewriter.replaceOp(op, {pad_op.getResult()});
  return success();
}

LogicalResult ConvertTFResizeBilinearOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_resize_op = cast<TF::ResizeBilinearOp>(op);

  RankedTensorType output_type =
      tf_resize_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  llvm::Optional<Value> result = convertResizeOp(
      rewriter, op, output_type, tf_resize_op.images(), StringRef("BILINEAR"),
      tf_resize_op.align_cornersAttr().getValue(),
      tf_resize_op.half_pixel_centersAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFResizeNearestNeighborOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_resize_op = cast<TF::ResizeNearestNeighborOp>(op);

  RankedTensorType output_type =
      tf_resize_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  llvm::Optional<Value> result =
      convertResizeOp(rewriter, op, output_type, tf_resize_op.images(),
                      StringRef("NEAREST_NEIGHBOR"),
                      tf_resize_op.align_cornersAttr().getValue(),
                      tf_resize_op.half_pixel_centersAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFMatMulOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_matmul_op = cast<TF::MatMulOp>(op);

  RankedTensorType a_type =
      tf_matmul_op.a().getType().dyn_cast<RankedTensorType>();
  RankedTensorType b_type =
      tf_matmul_op.b().getType().dyn_cast<RankedTensorType>();
  RankedTensorType output_type =
      tf_matmul_op.getResult().getType().dyn_cast<RankedTensorType>();

  if (!(a_type && b_type && output_type)) {
    return op->emitOpError("MatMul: a/b/output not ranked tensors");
  }

  if (a_type.getRank() != b_type.getRank() ||
      a_type.getRank() != output_type.getRank()) {
    return op->emitOpError("MatMul: a/b/output rank must match");
  }

  // Can only handle rank 2 tensors for tf.MatMul.
  // Cases with rank > 2 tensors should be handled by tf.BatchMatMul or
  // tf.BatchMatMulV2
  if (a_type.getRank() != 2) {
    return op->emitOpError("MatMul: a/b/output rank must be 2");
  }

  SmallVector<int64_t, 3> batch_a_shape(
      {1, a_type.getShape()[0], a_type.getShape()[1]});
  SmallVector<int64_t, 3> batch_b_shape(
      {1, b_type.getShape()[0], b_type.getShape()[1]});
  SmallVector<int64_t, 3> batch_output_shape(
      {1, output_type.getShape()[0], output_type.getShape()[1]});

  RankedTensorType batch_a_type =
      RankedTensorType::get(batch_a_shape, a_type.getElementType());
  RankedTensorType batch_b_type =
      RankedTensorType::get(batch_b_shape, b_type.getElementType());
  RankedTensorType batch_output_type =
      RankedTensorType::get(batch_output_shape, output_type.getElementType());

  // Need to reshape input and output since TOSA matmul only supports
  // [N, H, C] * [N, C, W] -> [N, H, W].
  auto op1_reshape_a = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(), batch_a_type, tf_matmul_op.a(),
      rewriter.getI64ArrayAttr(batch_a_shape));

  auto op2_reshape_b = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(), batch_b_type, tf_matmul_op.b(),
      rewriter.getI64ArrayAttr(batch_b_shape));

  auto op3_matmul_op1_op2 = CreateOpAndInfer<tosa::MatMulOp>(
      rewriter, op->getLoc(), batch_output_type, op1_reshape_a.getResult(),
      op2_reshape_b.getResult());

  CreateReplaceOpAndInfer<tosa::ReshapeOp>(
      rewriter, op, output_type, op3_matmul_op1_op2.getResult(),
      rewriter.getI64ArrayAttr(output_type.getShape()));

  return success();
}

LogicalResult ConvertTFGatherOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_gather_op = cast<TF::GatherOp>(op);

  // tf.Gather is equivalent to tf.GatherV2 with batch_dims = 0, axis = 0
  int32_t batch_dims = 0;
  int32_t axis = 0;

  llvm::Optional<Value> result = convertGatherOp(
      rewriter, op, tf_gather_op.getResult(), tf_gather_op.params(),
      tf_gather_op.indices(), batch_dims, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFGatherV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_gather_op = cast<TF::GatherV2Op>(op);

  // Axis is a tensor.  Pull out the one integer value.
  ElementsAttr axis_elem;
  if (!matchPattern(tf_gather_op.axis(), m_Constant(&axis_elem)))
    return failure();
  assert(axis_elem.getNumElements() == 1);

  int32_t axis = axis_elem.getValues<IntegerAttr>()[0].getInt();
  int32_t batch_dims = tf_gather_op.batch_dimsAttr().getInt();

  llvm::Optional<Value> result = convertGatherOp(
      rewriter, op, tf_gather_op.getResult(), tf_gather_op.params(),
      tf_gather_op.indices(), batch_dims, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFGatherNdOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_gathernd_op = cast<TF::GatherNdOp>(op);

  llvm::Optional<Value> result =
      convertGatherNdOp(rewriter, op, tf_gathernd_op.getResult(),
                        tf_gathernd_op.params(), tf_gathernd_op.indices());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFSelectV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_sel_op = cast<TF::SelectV2Op>(op);

  llvm::Optional<Value> result =
      convertSelectOp(rewriter, op, tf_sel_op.getResult(),
                      tf_sel_op.condition(), tf_sel_op.t(), tf_sel_op.e());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFSpaceToDepthOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_s2d_op = cast<TF::SpaceToDepthOp>(op);

  llvm::Optional<Value> result = convertSpaceToDepthOp(
      rewriter, op, tf_s2d_op.getResult(), tf_s2d_op.input(),
      tf_s2d_op.block_sizeAttr(), tf_s2d_op.data_formatAttr());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFDepthToSpaceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_d2s_op = cast<TF::DepthToSpaceOp>(op);

  llvm::Optional<Value> result = convertDepthToSpaceOp(
      rewriter, op, tf_d2s_op.getResult(), tf_d2s_op.input(),
      tf_d2s_op.block_sizeAttr(), tf_d2s_op.data_formatAttr());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFSpaceToBatchNDOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_s2b_op = cast<TF::SpaceToBatchNDOp>(op);

  llvm::Optional<Value> result = convertSpaceToBatchNDOp(
      rewriter, op, tf_s2b_op.getResult(), tf_s2b_op.input(),
      tf_s2b_op.block_shape(), tf_s2b_op.paddings());
  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFBatchToSpaceNDOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_b2s_op = cast<TF::BatchToSpaceNDOp>(op);

  llvm::Optional<Value> result = convertBatchToSpaceNDOp(
      rewriter, op, tf_b2s_op.getResult(), tf_b2s_op.input(),
      tf_b2s_op.block_shape(), tf_b2s_op.crops());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFStridedSliceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_ss_op = cast<TF::StridedSliceOp>(op);

  llvm::Optional<Value> result = convertStridedSliceOp(
      rewriter, op, tf_ss_op.getResult(), tf_ss_op.input(), tf_ss_op.begin(),
      tf_ss_op.end(), tf_ss_op.strides(), tf_ss_op.begin_maskAttr().getInt(),
      tf_ss_op.end_maskAttr().getInt(), tf_ss_op.ellipsis_maskAttr().getInt(),
      tf_ss_op.new_axis_maskAttr().getInt(),
      tf_ss_op.shrink_axis_maskAttr().getInt());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFZerosLikeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_zeroslike_op = cast<TF::ZerosLikeOp>(op);

  llvm::Optional<Value> result = convertZerosLikeOp(
      rewriter, op, tf_zeroslike_op.getResult(), tf_zeroslike_op.x());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFSigmoidOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_sigmoid_op = cast<TF::SigmoidOp>(op);
  TensorType output_type =
      tf_sigmoid_op.getResult().getType().dyn_cast<TensorType>();
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::SigmoidOp>(rewriter, op, output_type,
                                           tf_sigmoid_op.x());

  return success();
}

LogicalResult ConvertTFTanhOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_tanh_op = cast<TF::TanhOp>(op);
  TensorType output_type =
      tf_tanh_op.getResult().getType().dyn_cast<TensorType>();
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::TanhOp>(rewriter, op, output_type,
                                        tf_tanh_op.x());

  return success();
}

LogicalResult ConvertTFLeakyReluOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_leakyrelu_op = cast<TF::LeakyReluOp>(op);
  TensorType output_type =
      tf_leakyrelu_op.getResult().getType().dyn_cast<TensorType>();
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
    op->emitOpError("ConvertTFLeakyReluOp: only support F32");
    return failure();
  }

  FloatAttr tmpAttr = tf_leakyrelu_op.alphaAttr();
  // There is disagreement between the MLIR .td defaults and TF
  // documentation on 0.2 vs 0.3, but 0.2 will be used here.
  double alpha = 0.2;

  if (tmpAttr) {
    alpha = tmpAttr.getValueAsDouble();
  }

  Value const_zero = getTosaConstTensorSingleF32(rewriter, op, 0.0);

  auto a1_mul = CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(), output_type, tf_leakyrelu_op.features(),
      getTosaConstTensorSingleF32(rewriter, op, alpha), 0);

  auto a2_ge = CreateOpAndInfer<tosa::GreaterEqualOp>(
      rewriter, op->getLoc(), UnrankedTensorType::get(rewriter.getI1Type()),
      tf_leakyrelu_op.features(), const_zero);

  auto a3_select = CreateOpAndInfer<tosa::SelectOp>(
      rewriter, op->getLoc(), output_type, a2_ge, tf_leakyrelu_op.features(),
      a1_mul.getResult());

  rewriter.replaceOp(op, {a3_select.getResult()});

  return success();
}

LogicalResult ConvertTFNegOp::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto tf_neg_op = cast<TF::NegOp>(op);
  TensorType output_type =
      tf_neg_op.getResult().getType().dyn_cast<TensorType>();
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::NegateOp>(rewriter, op, output_type,
                                          tf_neg_op.x());

  return success();
}

LogicalResult ConvertTFStopGradientOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_stopgrad_op = cast<TF::StopGradientOp>(op);
  TensorType output_type =
      tf_stopgrad_op.getResult().getType().dyn_cast<TensorType>();
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::IdentityOp>(rewriter, op, output_type,
                                            tf_stopgrad_op.input());

  return success();
}

LogicalResult ConvertTFReverseV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_reverse_op = cast<TF::ReverseV2Op>(op);
  RankedTensorType input_type =
      tf_reverse_op.tensor().getType().dyn_cast<RankedTensorType>();
  TensorType output_type =
      tf_reverse_op.getResult().getType().dyn_cast<TensorType>();
  if (!input_type || !output_type) return failure();

  ElementsAttr axis_elems;
  if (!matchPattern(tf_reverse_op.axis(), m_Constant(&axis_elems)))
    return failure();

  auto input_rank = input_type.getShape().size();
  Value val = tf_reverse_op.tensor();
  if (axis_elems.getNumElements() == 0) {
    auto identity_op = CreateOpAndInfer<tosa::IdentityOp>(
        rewriter, op->getLoc(), output_type, val);
    val = identity_op.getResult();
  } else {
    for (int i = 0; i < axis_elems.getNumElements(); i++) {
      int64_t axis_val = axis_elems.getValues<IntegerAttr>()[i].getInt();
      if (axis_val < 0) axis_val += input_rank;
      auto axis_attr = rewriter.getI64IntegerAttr(axis_val);
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
      tf_fakequant_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  llvm::Optional<Value> result =
      convertFakeQuantOp(rewriter, op, output_type, tf_fakequant_op.inputs(),
                         tf_fakequant_op.minAttr().getValueAsDouble(),
                         tf_fakequant_op.maxAttr().getValueAsDouble(),
                         tf_fakequant_op.num_bitsAttr().getInt(),
                         tf_fakequant_op.narrow_rangeAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFFakeQuantWithMinMaxVarsOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_fakequant_op = cast<TF::FakeQuantWithMinMaxVarsOp>(op);

  TensorType output_type =
      tf_fakequant_op.getResult().getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) return failure();

  // Only support min/max that can be matched at compile time
  ElementsAttr min_elems, max_elems;
  if (!matchPattern(tf_fakequant_op.min(), m_Constant(&min_elems)))
    return failure();

  if (!matchPattern(tf_fakequant_op.max(), m_Constant(&max_elems)))
    return failure();

  if (min_elems.getNumElements() != 1 && max_elems.getNumElements() != 1)
    return failure();

  int64_t min_val = min_elems.getValues<IntegerAttr>()[0].getInt();
  int64_t max_val = max_elems.getValues<IntegerAttr>()[0].getInt();

  llvm::Optional<Value> result = convertFakeQuantOp(
      rewriter, op, output_type, tf_fakequant_op.inputs(), min_val, max_val,
      tf_fakequant_op.num_bitsAttr().getInt(),
      tf_fakequant_op.narrow_rangeAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFLeftShiftOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_left_shift_op = cast<TF::LeftShiftOp>(op);

  TensorType output_type =
      tf_left_shift_op.getResult().getType().dyn_cast<TensorType>();
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::LogicalLeftShiftOp>(
      rewriter, op, output_type, tf_left_shift_op.x(), tf_left_shift_op.y());

  return success();
}

LogicalResult ConvertTFRightShiftOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  // Performs a logical shift for unsigned integer types, and an arithmetic
  // shift for signed integer types.
  auto tf_right_shift_op = cast<TF::RightShiftOp>(op);

  TensorType output_type =
      tf_right_shift_op.getResult().getType().dyn_cast<TensorType>();
  if (!output_type) return failure();

  Type output_element_type = output_type.getElementType();

  bool is_signed = false;
  if (!output_element_type.isUnsignedInteger()) is_signed = true;

  if (is_signed) {
    CreateReplaceOpAndInfer<tosa::ArithmeticRightShiftOp>(
        rewriter, op, output_type, tf_right_shift_op.x(), tf_right_shift_op.y(),
        false);
  } else {
    CreateReplaceOpAndInfer<tosa::LogicalRightShiftOp>(
        rewriter, op, output_type, tf_right_shift_op.x(),
        tf_right_shift_op.y());
  }

  return success();
}

LogicalResult ConvertTFOneHotOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_one_hot_op = cast<TF::OneHotOp>(op);

  ElementsAttr depth_elems;
  if (!matchPattern(tf_one_hot_op.depth(), m_Constant(&depth_elems)))
    return failure();
  int32_t depth = depth_elems.getValues<IntegerAttr>()[0].getInt();

  IntegerAttr axisAttr = tf_one_hot_op.axisAttr();
  int32_t axis = axisAttr.getInt();

  llvm::Optional<Value> result = convertOneHotOp(
      rewriter, op, tf_one_hot_op.getResult(), tf_one_hot_op.indices(),
      tf_one_hot_op.on_value(), tf_one_hot_op.off_value(), depth, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

LogicalResult ConvertTFBatchMatMulV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_batch_matmul_op = cast<TF::BatchMatMulV2Op>(op);

  RankedTensorType x_type =
      tf_batch_matmul_op.x().getType().dyn_cast<RankedTensorType>();
  RankedTensorType y_type =
      tf_batch_matmul_op.y().getType().dyn_cast<RankedTensorType>();
  RankedTensorType output_type =
      tf_batch_matmul_op.getResult().getType().dyn_cast<RankedTensorType>();

  if (!(x_type && y_type && output_type)) {
    return op->emitOpError("BatchMatMulV2: x/y/output not ranked tensors");
  }

  if (x_type.getRank() != y_type.getRank() ||
      x_type.getRank() != output_type.getRank()) {
    return op->emitOpError("BatchMatMulV2: x/y/output rank must match");
  }

  if (x_type.getRank() <= 2) {
    return op->emitOpError("BatchMatMulV2: x/y/output rank must > 2");
  }

  // Rank 3 batch matmul can be directly mapped to tosa.matmul trivially.
  if (x_type.getRank() == 3) {
    CreateReplaceOpAndInfer<tosa::MatMulOp>(rewriter, op, output_type,
                                            tf_batch_matmul_op.x(),
                                            tf_batch_matmul_op.y());
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

    RankedTensorType rank3_x_type =
        RankedTensorType::get(rank3_x_shape, x_type.getElementType());
    RankedTensorType rank3_y_type =
        RankedTensorType::get(rank3_y_shape, y_type.getElementType());
    RankedTensorType rank3_output_type =
        RankedTensorType::get(rank3_output_shape, output_type.getElementType());

    auto op1_reshape_x = CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(), rank3_x_type, tf_batch_matmul_op.x(),
        rewriter.getI64ArrayAttr(rank3_x_shape));

    auto op2_reshape_y = CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(), rank3_y_type, tf_batch_matmul_op.y(),
        rewriter.getI64ArrayAttr(rank3_y_shape));

    auto op3_matmul_op1_op2 = CreateOpAndInfer<tosa::MatMulOp>(
        rewriter, op->getLoc(), rank3_output_type, op1_reshape_x.getResult(),
        op2_reshape_y.getResult());

    CreateReplaceOpAndInfer<tosa::ReshapeOp>(
        rewriter, op, output_type, op3_matmul_op1_op2.getResult(),
        rewriter.getI64ArrayAttr(output_type.getShape()));
  }
  return success();
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
  patterns.add<ConvertTFResizeBilinearOp>(ctx);
  patterns.add<ConvertTFResizeNearestNeighborOp>(ctx);
  patterns.add<ConvertTFGatherOp>(ctx);
  patterns.add<ConvertTFGatherV2Op>(ctx);
  patterns.add<ConvertTFGatherNdOp>(ctx);
  patterns.add<ConvertTFSelectV2Op>(ctx);
  patterns.add<ConvertTFSpaceToDepthOp>(ctx);
  patterns.add<ConvertTFDepthToSpaceOp>(ctx);
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
}

// Creates an instance of the TensorFlow dialect LegalizeTF pass.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFPass() {
  return std::make_unique<LegalizeTF>();
}

}  // namespace tosa

}  // namespace mlir
