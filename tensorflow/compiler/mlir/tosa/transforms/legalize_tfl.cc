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

// Legalize TensorFlow Lite to TOSA

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"
#include "tensorflow/lite/kernels/internal/reference/gelu.h"

#define PASS_NAME "tosa-legalize-tfl"
#define DEBUG_TYPE PASS_NAME
#define HARDSWISH_EXPLICIT_RESCALING false

namespace mlir {
namespace tosa {
namespace {

#define GEN_PASS_DEF_TOSALEGALIZETFLPASS
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

// Performs lowering to TOSA dialect.
class LegalizeTFL : public impl::TosaLegalizeTFLPassBase<LegalizeTFL> {
 public:
  LegalizeTFL() = default;
  explicit LegalizeTFL(ArrayRef<std::string> disabled_patterns,
                       ArrayRef<std::string> enabled_patterns) {
    this->disabled_patterns_ = disabled_patterns;
    this->enabled_patterns_ = enabled_patterns;
  }
  void runOnOperation() override;
  LogicalResult initialize(MLIRContext* context) override;

 private:
  FrozenRewritePatternSet frozen_patterns_;
};

#include "tensorflow/compiler/mlir/tosa/transforms/tfl_legalize_patterns.inc"

// Input from tfl.conv2d takes 64 bits a bias, while tosa.conv2d expects 48
// bits. Need to do a customized truncate here instead of tablegen to handle
// attribute with negative value.
struct ConvertConstantOp : public RewritePattern {
  explicit ConvertConstantOp(MLIRContext* context)
      : RewritePattern(arith::ConstantOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

#define DECL_CONVERT_OP(tfl_op)                                              \
  struct ConvertTFL##tfl_op##Op : public RewritePattern {                    \
    explicit ConvertTFL##tfl_op##Op(MLIRContext* context)                    \
        : RewritePattern(TFL::tfl_op##Op::getOperationName(), 1, context) {} \
    LogicalResult matchAndRewrite(Operation* op,                             \
                                  PatternRewriter& rewriter) const override; \
  }
DECL_CONVERT_OP(Gelu);
DECL_CONVERT_OP(Relu);
DECL_CONVERT_OP(Relu1);
DECL_CONVERT_OP(Relu0To1);
DECL_CONVERT_OP(Relu6);
DECL_CONVERT_OP(Equal);
DECL_CONVERT_OP(NotEqual);
DECL_CONVERT_OP(Greater);
DECL_CONVERT_OP(GreaterEqual);
DECL_CONVERT_OP(Add);
DECL_CONVERT_OP(Sub);
DECL_CONVERT_OP(Mul);
DECL_CONVERT_OP(Square);
DECL_CONVERT_OP(SquaredDifference);
DECL_CONVERT_OP(Cast);
DECL_CONVERT_OP(Sign);
DECL_CONVERT_OP(Round);
DECL_CONVERT_OP(Div);
DECL_CONVERT_OP(Maximum);
DECL_CONVERT_OP(Minimum);
DECL_CONVERT_OP(FloorMod);
DECL_CONVERT_OP(FloorDiv);
DECL_CONVERT_OP(AddN);
DECL_CONVERT_OP(AveragePool2D);
DECL_CONVERT_OP(MaxPool2D);
DECL_CONVERT_OP(Concatenation);
DECL_CONVERT_OP(Reshape);
DECL_CONVERT_OP(Rank);
DECL_CONVERT_OP(Shape);
DECL_CONVERT_OP(ExpandDims);
DECL_CONVERT_OP(Squeeze);
DECL_CONVERT_OP(Fill);
DECL_CONVERT_OP(Elu);
DECL_CONVERT_OP(Softmax);
DECL_CONVERT_OP(LogSoftmax);
DECL_CONVERT_OP(Rsqrt);
DECL_CONVERT_OP(Sqrt);
DECL_CONVERT_OP(L2Normalization);
DECL_CONVERT_OP(ReduceAll);
DECL_CONVERT_OP(ReduceAny);
DECL_CONVERT_OP(ReduceMax);
DECL_CONVERT_OP(ReduceMin);
DECL_CONVERT_OP(Mean);
DECL_CONVERT_OP(ReduceProd);
DECL_CONVERT_OP(Sum);
DECL_CONVERT_OP(Conv2D);
DECL_CONVERT_OP(Conv3D);
DECL_CONVERT_OP(TransposeConv);
DECL_CONVERT_OP(DepthwiseConv2D);
DECL_CONVERT_OP(FullyConnected);
DECL_CONVERT_OP(BatchMatMul);
DECL_CONVERT_OP(Split);
DECL_CONVERT_OP(SplitV);
DECL_CONVERT_OP(Pack);
DECL_CONVERT_OP(Unpack);
DECL_CONVERT_OP(Transpose);
DECL_CONVERT_OP(Tile);
DECL_CONVERT_OP(Slice);
DECL_CONVERT_OP(StridedSlice);
DECL_CONVERT_OP(HardSwish);
DECL_CONVERT_OP(ZerosLike);
DECL_CONVERT_OP(Less);
DECL_CONVERT_OP(LessEqual);
DECL_CONVERT_OP(Pad);
DECL_CONVERT_OP(MirrorPad);
DECL_CONVERT_OP(PadV2);
DECL_CONVERT_OP(ResizeBilinear);
DECL_CONVERT_OP(ResizeNearestNeighbor);
DECL_CONVERT_OP(Select);
DECL_CONVERT_OP(SelectV2);
DECL_CONVERT_OP(SpaceToBatchNd);
DECL_CONVERT_OP(BatchToSpaceNd);
DECL_CONVERT_OP(SpaceToDepth);
DECL_CONVERT_OP(DepthToSpace);
DECL_CONVERT_OP(Bucketize);
DECL_CONVERT_OP(Sin);
DECL_CONVERT_OP(Cos);
DECL_CONVERT_OP(Atan2);
DECL_CONVERT_OP(Logistic);
DECL_CONVERT_OP(Tanh);
DECL_CONVERT_OP(PRelu);
DECL_CONVERT_OP(LeakyRelu);
DECL_CONVERT_OP(Neg);
DECL_CONVERT_OP(Yield);
DECL_CONVERT_OP(Custom);
DECL_CONVERT_OP(ReverseV2);
DECL_CONVERT_OP(Quantize);
DECL_CONVERT_OP(Dequantize);
DECL_CONVERT_OP(Const);
DECL_CONVERT_OP(QConst);
DECL_CONVERT_OP(Gather);
DECL_CONVERT_OP(GatherNd);
DECL_CONVERT_OP(SparseToDense);
DECL_CONVERT_OP(OneHot);
DECL_CONVERT_OP(ArgMax);
DECL_CONVERT_OP(ArgMin);
DECL_CONVERT_OP(FakeQuant);
DECL_CONVERT_OP(While);
DECL_CONVERT_OP(Real);
DECL_CONVERT_OP(Imag);
DECL_CONVERT_OP(RFFT2d);
DECL_CONVERT_OP(LogicalAnd);
DECL_CONVERT_OP(LogicalOr);
DECL_CONVERT_OP(Pow);
DECL_CONVERT_OP(BroadcastTo);

#undef DECL_CONVERT_OP

LogicalResult ConvertTFLGeluOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_gelu_op = cast<TFL::GeluOp>(op);
  Location loc = op->getLoc();

  auto approximate = tfl_gelu_op.getApproximateAttr().getValue();

  Value input = tfl_gelu_op.getInput();
  RankedTensorType input_type = dyn_cast<RankedTensorType>(input.getType());
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_gelu_op.getResult().getType());
  if (!input_type || !output_type) {
    return rewriter.notifyMatchFailure(
        op, "input/output are not all a ranked tensor");
  }

  UniformQuantizedType in_quant_type =
      dyn_cast<mlir::quant::UniformQuantizedType>(input_type.getElementType());
  UniformQuantizedType out_quant_type =
      dyn_cast<mlir::quant::UniformQuantizedType>(output_type.getElementType());

  if ((in_quant_type == nullptr) != (out_quant_type == nullptr)) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  if (out_quant_type) {
    if (in_quant_type.getStorageTypeIntegralWidth() != 8) {
      return rewriter.notifyMatchFailure(
          op, "current tfl.gelu only support 8-bit quantized type");
    }

    Value table_const = getTosaConst8bitTable(
        rewriter, op, in_quant_type.getScale(), in_quant_type.getZeroPoint(),
        out_quant_type.getScale(), out_quant_type.getZeroPoint(),
        approximate ? tflite::reference_ops::GeluTransformApproximate
                    : tflite::reference_ops::GeluTransform);

    CreateReplaceOpAndInfer<tosa::TableOp>(rewriter, op, output_type, input,
                                           table_const);
    return success();
  }

  // Following approximated implemention described in
  //   tensorflow/lite/kernels/internal/reference/gelu.h
  //
  // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  //
  // Lower the formula to the sequence of operators below:
  //   op0 = pow(x, 3)
  //   op1 = mul(op0, 0.044715)
  //   op2 = add(x, op1)
  //   op3 = mul(op2, sqrt(2/pi))
  //   op4 = tanh(op3)
  //   op5 = add(op4 ,1)
  //   op6 = mul(x, 0.5)
  //   op7 = mul(op6, op5)

  auto rank = input_type.getRank();
  Value cst_3 = getTosaConstTensorSingleF32(rewriter, op, 3.0f, rank);
  auto op0_pow_3 =
      CreateOpAndInfer<tosa::PowOp>(rewriter, loc, output_type, input, cst_3);

  Value cst_004 =
      getTosaConstTensorSingleF32(rewriter, op, 4.471500e-02f, rank);
  auto op1_mul_op0_004 =
      CreateMulOpAndInfer(rewriter, op, output_type, op0_pow_3, cst_004);

  auto op2_add_x_op1 = CreateOpAndInfer<tosa::AddOp>(rewriter, loc, output_type,
                                                     input, op1_mul_op0_004);

  Value cst_sqrt2pi =
      getTosaConstTensorSingleF32(rewriter, op, 0.797884583f, rank);
  auto op3_mul_op2_sqrt2pi = CreateMulOpAndInfer(rewriter, op, output_type,
                                                 op2_add_x_op1, cst_sqrt2pi);

  auto op4_tanh_op3 = CreateOpAndInfer<tosa::TanhOp>(rewriter, loc, output_type,
                                                     op3_mul_op2_sqrt2pi);

  Value cst_1 = getTosaConstTensorSingleF32(rewriter, op, 1.0f, rank);
  auto op5_add_op4_1 = CreateOpAndInfer<tosa::AddOp>(rewriter, loc, output_type,
                                                     op4_tanh_op3, cst_1);

  Value cst_05 = getTosaConstTensorSingleF32(rewriter, op, 0.5f, rank);
  auto op6_mul_x_05 =
      CreateMulOpAndInfer(rewriter, op, output_type, input, cst_05);

  auto op7_mul_op6_op5 = CreateMulOpAndInfer(rewriter, op, output_type,
                                             op6_mul_x_05, op5_add_op4_1);

  rewriter.replaceOp(op, {op7_mul_op6_op5.getResult()});

  return success();
}

LogicalResult ConvertTFLReluOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_relu_op = cast<TFL::ReluOp>(op);

  ShapedType input_type = dyn_cast<ShapedType>(tfl_relu_op.getX().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_relu_op.getResult().getType());
  // Not a ranked tensor output
  if (!input_type || !output_type) return failure();

  bool input_is_qtype =
      mlir::isa<mlir::quant::UniformQuantizedType>(input_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  int64_t clamp_min = 0;
  int64_t clamp_max = std::numeric_limits<int64_t>::max();
  Value clamp_in = tfl_relu_op.getX();

  if (output_is_qtype) {
    UniformQuantizedType input_qtype =
        dyn_cast<mlir::quant::UniformQuantizedType>(
            input_type.getElementType());
    UniformQuantizedType output_qtype =
        dyn_cast<mlir::quant::UniformQuantizedType>(
            output_type.getElementType());

    clamp_min = output_qtype.getZeroPoint();
    TrimQuantizedIntegerRangeMin(input_qtype, clamp_min);
    TrimQuantizedIntegerRangeMax(input_qtype, clamp_max);

    clamp_in =
        buildRescale(rewriter, op, output_type, tfl_relu_op.getX(),
                     input_qtype.getScale() / output_qtype.getScale(),
                     input_qtype.getZeroPoint(), output_qtype.getZeroPoint(),
                     /*double_round=*/"SINGLE_ROUND", /*scale32=*/true);
  }

  auto element_type = input_type.getElementType();
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
    min_val = rewriter.getIntegerAttr(element_type, clamp_min);
    max_val = rewriter.getIntegerAttr(element_type, clamp_max);
  }

  CreateReplaceOpAndInfer<tosa::ClampOp>(rewriter, op, output_type, clamp_in,
                                         min_val, max_val);

  return success();
}

LogicalResult ConvertTFLRelu1Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_relu1_op = cast<TFL::Relu1Op>(op);

  ShapedType input_type = dyn_cast<ShapedType>(tfl_relu1_op.getX().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_relu1_op.getResult().getType());
  // Not a ranked tensor output
  if (!input_type || !output_type) return failure();

  bool input_is_qtype =
      mlir::isa<mlir::quant::UniformQuantizedType>(input_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  int64_t clamp_min = -1;
  int64_t clamp_max = 1;
  Value clamp_in = tfl_relu1_op.getX();

  if (output_is_qtype && input_is_qtype) {
    UniformQuantizedType input_qtype =
        dyn_cast<mlir::quant::UniformQuantizedType>(
            input_type.getElementType());
    UniformQuantizedType output_qtype =
        dyn_cast<mlir::quant::UniformQuantizedType>(
            output_type.getElementType());

    clamp_min = output_qtype.getZeroPoint() -
                std::llround(1.0f / output_qtype.getScale());

    clamp_max = std::llround(1.0f / output_qtype.getScale()) +
                output_qtype.getZeroPoint();

    TrimQuantizedIntegerRange(input_qtype, clamp_min, clamp_max);

    clamp_in =
        buildRescale(rewriter, op, output_type, tfl_relu1_op.getX(),
                     input_qtype.getScale() / output_qtype.getScale(),
                     input_qtype.getZeroPoint(), output_qtype.getZeroPoint(),
                     /*double_round=*/"SINGLE_ROUND", /*scale32=*/true);
  }

  auto element_type = input_type.getElementType();
  if (auto quant_type =
          dyn_cast<mlir::quant::UniformQuantizedType>(element_type)) {
    element_type = quant_type.getStorageType();
  }

  mlir::Attribute min_val, max_val;
  if (element_type.isa<mlir::FloatType>()) {
    min_val = rewriter.getFloatAttr(element_type, -1.0f);
    max_val = rewriter.getFloatAttr(element_type, 1.0f);
  } else {
    min_val = rewriter.getIntegerAttr(element_type, clamp_min);
    max_val = rewriter.getIntegerAttr(element_type, clamp_max);
  }

  CreateReplaceOpAndInfer<tosa::ClampOp>(rewriter, op, output_type, clamp_in,
                                         min_val, max_val);

  return success();
}

LogicalResult ConvertTFLRelu0To1Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_relu0to1_op = cast<TFL::Relu0To1Op>(op);

  ShapedType input_type =
      mlir::cast<ShapedType>(tfl_relu0to1_op.getX().getType());
  ShapedType output_type =
      mlir::cast<ShapedType>(tfl_relu0to1_op.getResult().getType());

  bool input_is_qtype =
      mlir::isa<mlir::quant::UniformQuantizedType>(input_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  int64_t clamp_min = 0;
  int64_t clamp_max = 1;
  Value clamp_in = tfl_relu0to1_op.getX();

  if (output_is_qtype && input_is_qtype) {
    UniformQuantizedType input_qtype =
        mlir::cast<mlir::quant::UniformQuantizedType>(
            input_type.getElementType());
    UniformQuantizedType output_qtype =
        mlir::cast<mlir::quant::UniformQuantizedType>(
            output_type.getElementType());

    clamp_min = output_qtype.getZeroPoint();

    clamp_max = std::llround(1.0f / output_qtype.getScale()) +
                output_qtype.getZeroPoint();

    TrimQuantizedIntegerRange(input_qtype, clamp_min, clamp_max);

    clamp_in =
        buildRescale(rewriter, op, output_type, tfl_relu0to1_op.getX(),
                     input_qtype.getScale() / output_qtype.getScale(),
                     input_qtype.getZeroPoint(), output_qtype.getZeroPoint(),
                     /*double_round=*/"SINGLE_ROUND", /*scale32=*/true);
  }

  auto element_type = input_type.getElementType();
  if (auto quant_type =
          dyn_cast<mlir::quant::UniformQuantizedType>(element_type)) {
    element_type = quant_type.getStorageType();
  }

  mlir::Attribute min_val, max_val;
  if (element_type.isa<mlir::FloatType>()) {
    min_val = rewriter.getFloatAttr(element_type, 0.0f);
    max_val = rewriter.getFloatAttr(element_type, 1.0f);
  } else {
    min_val = rewriter.getIntegerAttr(element_type, clamp_min);
    max_val = rewriter.getIntegerAttr(element_type, clamp_max);
  }

  CreateReplaceOpAndInfer<tosa::ClampOp>(rewriter, op, output_type, clamp_in,
                                         min_val, max_val);

  return success();
}

LogicalResult ConvertTFLRelu6Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_relu6_op = cast<TFL::Relu6Op>(op);

  ShapedType input_type = dyn_cast<ShapedType>(tfl_relu6_op.getX().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_relu6_op.getResult().getType());
  // Not a ranked tensor output
  if (!input_type || !output_type) return failure();

  bool input_is_qtype =
      mlir::isa<mlir::quant::UniformQuantizedType>(input_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  int64_t clamp_min = 0;
  int64_t clamp_max = 6;
  Value clamp_in = tfl_relu6_op.getX();

  if (output_is_qtype && input_is_qtype) {
    UniformQuantizedType input_qtype =
        dyn_cast<mlir::quant::UniformQuantizedType>(
            input_type.getElementType());
    UniformQuantizedType output_qtype =
        dyn_cast<mlir::quant::UniformQuantizedType>(
            output_type.getElementType());

    clamp_min = output_qtype.getZeroPoint();
    clamp_max = std::llround(6.0f / output_qtype.getScale()) +
                output_qtype.getZeroPoint();

    TrimQuantizedIntegerRange(input_qtype, clamp_min, clamp_max);

    clamp_in =
        buildRescale(rewriter, op, output_type, tfl_relu6_op.getX(),
                     input_qtype.getScale() / output_qtype.getScale(),
                     input_qtype.getZeroPoint(), output_qtype.getZeroPoint(),
                     /*double_round=*/"SINGLE_ROUND", /*scale32=*/true);
  }

  auto element_type = input_type.getElementType();
  if (auto quant_type =
          dyn_cast<mlir::quant::UniformQuantizedType>(element_type)) {
    element_type = quant_type.getStorageType();
  }

  mlir::Attribute min_val, max_val;
  if (element_type.isa<mlir::FloatType>()) {
    min_val = rewriter.getFloatAttr(element_type, 0.0f);
    max_val = rewriter.getFloatAttr(element_type, 6.0f);
  } else {
    min_val = rewriter.getIntegerAttr(element_type, clamp_min);
    max_val = rewriter.getIntegerAttr(element_type, clamp_max);
  }

  CreateReplaceOpAndInfer<tosa::ClampOp>(rewriter, op, output_type, clamp_in,
                                         min_val, max_val);

  return success();
}

static LogicalResult prepareMatchAndRewriteComparison(
    Operation* op, mlir::OperandRange operands, PatternRewriter& rewriter,
    llvm::SmallVectorImpl<Value>& newOperands) {
  Value x = operands[0];
  Value y = operands[1];
  Value result = op->getResult(0);

  ShapedType input_x_type = dyn_cast<ShapedType>(x.getType());
  ShapedType input_y_type = dyn_cast<ShapedType>(y.getType());
  ShapedType output_type = dyn_cast<ShapedType>(result.getType());
  // Not a shaped tensor output
  if (!input_x_type || !input_y_type || !output_type) return failure();

  bool input_x_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      input_x_type.getElementType());
  bool input_y_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      input_y_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_x_is_qtype != input_y_is_qtype ||
      input_y_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  if (!output_is_qtype && !input_x_is_qtype && !input_y_is_qtype) {
    newOperands.push_back(x);
    newOperands.push_back(y);
    return success();
  }

  UniformQuantizedType input_x_qtype =
      dyn_cast<mlir::quant::UniformQuantizedType>(
          input_x_type.getElementType());
  UniformQuantizedType input_y_qtype =
      dyn_cast<mlir::quant::UniformQuantizedType>(
          input_y_type.getElementType());

  if (input_x_qtype.getScale() != input_y_qtype.getScale() ||
      input_x_qtype.getZeroPoint() != input_y_qtype.getZeroPoint()) {
    return rewriter.notifyMatchFailure(
        op, "input_x and input_y scale/zp must be the same");
  }

  x = removeZeroPointAndCastToInt32(rewriter, op, x,
                                    input_x_qtype.getZeroPoint());
  y = removeZeroPointAndCastToInt32(rewriter, op, y,
                                    input_y_qtype.getZeroPoint());

  newOperands.push_back(x);
  newOperands.push_back(y);
  return success();
}

LogicalResult ConvertTFLEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  llvm::SmallVector<Value, 2> newOperands;
  LogicalResult status = prepareMatchAndRewriteComparison(
      op, op->getOperands(), rewriter, newOperands);
  if (status.failed()) return failure();

  CreateReplaceOpAndInfer<tosa::EqualOp>(
      rewriter, op, op->getResult(0).getType(), newOperands[0], newOperands[1]);

  return success();
}

LogicalResult ConvertTFLNotEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  llvm::SmallVector<Value, 2> newOperands;
  LogicalResult status = prepareMatchAndRewriteComparison(
      op, op->getOperands(), rewriter, newOperands);
  if (status.failed()) return failure();

  auto equal_op = CreateOpAndInfer<tosa::EqualOp>(
      rewriter, op->getLoc(), op->getResult(0).getType(), newOperands[0],
      newOperands[1]);

  CreateReplaceOpAndInfer<tosa::LogicalNotOp>(
      rewriter, op, op->getResult(0).getType(), equal_op);

  return success();
}

LogicalResult ConvertTFLGreaterOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  llvm::SmallVector<Value, 2> newOperands;
  LogicalResult status = prepareMatchAndRewriteComparison(
      op, op->getOperands(), rewriter, newOperands);
  if (status.failed()) return failure();

  CreateReplaceOpAndInfer<tosa::GreaterOp>(
      rewriter, op, op->getResult(0).getType(), newOperands[0], newOperands[1]);

  return success();
}

LogicalResult ConvertTFLGreaterEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  llvm::SmallVector<Value, 2> newOperands;
  LogicalResult status = prepareMatchAndRewriteComparison(
      op, op->getOperands(), rewriter, newOperands);
  if (status.failed()) return failure();

  CreateReplaceOpAndInfer<tosa::GreaterEqualOp>(
      rewriter, op, op->getResult(0).getType(), newOperands[0], newOperands[1]);

  return success();
}

LogicalResult ConvertTFLLessOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  llvm::SmallVector<Value, 2> newOperands;
  LogicalResult status = prepareMatchAndRewriteComparison(
      op, op->getOperands(), rewriter, newOperands);
  if (status.failed()) return failure();

  CreateReplaceOpAndInfer<tosa::GreaterOp>(
      rewriter, op, op->getResult(0).getType(), newOperands[1], newOperands[0]);
  return success();
}

LogicalResult ConvertTFLLessEqualOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  llvm::SmallVector<Value, 2> newOperands;
  LogicalResult status = prepareMatchAndRewriteComparison(
      op, op->getOperands(), rewriter, newOperands);
  if (status.failed()) return failure();

  // Swapping the args handles the greater/less difference.
  CreateReplaceOpAndInfer<tosa::GreaterEqualOp>(
      rewriter, op, op->getResult(0).getType(), newOperands[1], newOperands[0]);

  return success();
}

template <typename TflOp, typename TosaOp>
static LogicalResult matchAndRewriteAddSub(Operation* op,
                                           mlir::OperandRange operands,
                                           PatternRewriter& rewriter) {
  auto tfl_add_op = cast<TflOp>(op);

  Value lhs = tfl_add_op.getLhs();
  Value rhs = tfl_add_op.getRhs();

  ShapedType input_lhs_type = mlir::dyn_cast<ShapedType>(lhs.getType());
  ShapedType input_rhs_type = mlir::dyn_cast<ShapedType>(rhs.getType());
  ShapedType output_type =
      mlir::dyn_cast<ShapedType>(tfl_add_op.getResult().getType());
  if (!input_lhs_type || !input_rhs_type || !output_type) return failure();

  bool input_lhs_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      input_lhs_type.getElementType());
  bool input_rhs_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      input_rhs_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_lhs_is_qtype != output_is_qtype ||
      input_rhs_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  Value output;
  if (output_is_qtype && input_lhs_is_qtype && input_rhs_is_qtype) {
    ShapedType rescale_type_output = output_type.clone(rewriter.getI32Type());
    ShapedType rescale_type_input_left =
        input_lhs_type.clone(rewriter.getI32Type());
    ShapedType rescale_type_input_right =
        input_rhs_type.clone(rewriter.getI32Type());

    UniformQuantizedType input_lhs_qtype =
        dyn_cast<mlir::quant::UniformQuantizedType>(
            input_lhs_type.getElementType());
    UniformQuantizedType input_rhs_qtype =
        dyn_cast<mlir::quant::UniformQuantizedType>(
            input_rhs_type.getElementType());
    UniformQuantizedType output_qtype =
        dyn_cast<mlir::quant::UniformQuantizedType>(
            output_type.getElementType());

    // Following quantization described in tensorflow/lite/kernels/add.cc
    // In details it does:
    // 1. Rescale inputs to scale = 2.0 x max(lhs.scale, rhs.scale)
    // 2. Extra left shift to input to increase precision
    // Where input_shift = 20 if input is 8-bit
    // input_shift = 15 if input is 16-bit
    double in_lhs_scale = input_lhs_qtype.getScale();
    double in_rhs_scale = input_rhs_qtype.getScale();
    double output_scale = output_qtype.getScale();
    double max_scale_2x = 2.0 * std::max(in_lhs_scale, in_rhs_scale);

    const int32_t SHIFT_8_BIT = 20;
    const int32_t SHIFT_16_BIT = 15;

    int32_t input_shift = (output_qtype.getStorageTypeIntegralWidth() == 16)
                              ? SHIFT_16_BIT
                              : SHIFT_8_BIT;

    double lhs_rescale_scale = in_lhs_scale / max_scale_2x;
    double rhs_rescale_scale = in_rhs_scale / max_scale_2x;
    double output_rescale_scale =
        max_scale_2x / (output_scale * static_cast<double>(1 << input_shift));

#if TFLITE_SINGLE_ROUNDING
    Value op1_rescale_lhs = buildRescaleToInt32(
        rewriter, op, lhs,
        lhs_rescale_scale * static_cast<double>(1 << input_shift),
        input_lhs_qtype.getZeroPoint());
    Value op2_rescale_rhs = buildRescaleToInt32(
        rewriter, op, rhs,
        rhs_rescale_scale * static_cast<double>(1 << input_shift),
        input_rhs_qtype.getZeroPoint());
#else  // TFLITE_DOUBLE_ROUNDING
    Value op1_rescale_lhs;
    Value op2_rescale_rhs;

    // Left side
    // Handle only 16-bit none_exact_half_rescale_scale as a special case here:
    // 1. The double rescale implementation will break the TOSA reference model
    //    because for 16-bit values, the minimum shift must be 16 which
    //    leaves a left shift of 15 impossible. Using a Cast + Shift instead.
    // 2. In other cases, we are scaling by lhs_rescale_scale*(1<<input_shift)
    if (lhs_rescale_scale == 0.5) {
      op1_rescale_lhs = buildRescaleToInt32(
          rewriter, op, lhs, lhs_rescale_scale * (1 << input_shift),
          input_lhs_qtype.getZeroPoint());
    } else {
      Value op1_none_half_scale_intermediate;
      if (output_qtype.getStorageTypeIntegralWidth() == 16) {
        auto tfl_add_lhs_casted = CreateOpAndInfer<tosa::CastOp>(
            rewriter, op->getLoc(), rescale_type_input_left, lhs);
        op1_none_half_scale_intermediate =
            CreateOpAndInfer<tosa::LogicalLeftShiftOp>(
                rewriter, op->getLoc(), rescale_type_input_left,
                tfl_add_lhs_casted.getResult(),
                getTosaConstTensorSingleI32(rewriter, op, input_shift,
                                            input_lhs_type.getRank()));
      } else {
        op1_none_half_scale_intermediate =
            buildRescaleToInt32(rewriter, op, lhs, (1 << input_shift),
                                input_lhs_qtype.getZeroPoint());
      }
      op1_rescale_lhs = buildRescaleToInt32(
          rewriter, op, op1_none_half_scale_intermediate, lhs_rescale_scale, 0);
    }

    // Right side
    // Handle only 16-bit none_exact_half_rescale_scale as a special case here:
    // 1. The double rescale implementation will break the TOSA reference model
    //    because for 16-bit values, the minimum shift must be 16 which
    //    leaves a left shift of 15 impossible. Using a Cast + Shift instead.
    // 2. In other cases, we are scaling by rhs_rescale_scale*(1<<input_shift)
    if (rhs_rescale_scale == 0.5) {
      op2_rescale_rhs = buildRescaleToInt32(
          rewriter, op, rhs, rhs_rescale_scale * (1 << input_shift),
          input_rhs_qtype.getZeroPoint());
    } else {
      Value op2_none_half_scale_intermediate;
      if (output_qtype.getStorageTypeIntegralWidth() == 16) {
        auto tfl_add_rhs_casted = CreateOpAndInfer<tosa::CastOp>(
            rewriter, op->getLoc(), rescale_type_input_right, rhs);
        op2_none_half_scale_intermediate =
            CreateOpAndInfer<tosa::LogicalLeftShiftOp>(
                rewriter, op->getLoc(), rescale_type_input_right,
                tfl_add_rhs_casted.getResult(),
                getTosaConstTensorSingleI32(rewriter, op, input_shift,
                                            input_rhs_type.getRank()));
      } else {
        op2_none_half_scale_intermediate =
            buildRescaleToInt32(rewriter, op, rhs, (1 << input_shift),
                                input_rhs_qtype.getZeroPoint());
      }
      op2_rescale_rhs = buildRescaleToInt32(
          rewriter, op, op2_none_half_scale_intermediate, rhs_rescale_scale, 0);
    }

#endif  // TFLITE_DOUBLE_ROUNDING
    auto op3_add_op1_op2 =
        CreateOpAndInfer<TosaOp>(rewriter, op->getLoc(), rescale_type_output,
                                 op1_rescale_lhs, op2_rescale_rhs);
    Value op4_rescale_op3 = buildRescaleFromInt32(
        rewriter, op, output_type, op3_add_op1_op2.getResult(),
        output_rescale_scale, output_qtype.getZeroPoint());

    output = op4_rescale_op3;
  } else {
    auto op1_add_in =
        CreateOpAndInfer<TosaOp>(rewriter, op->getLoc(), output_type, lhs, rhs);

    output = op1_add_in.getResult();
  }

  auto fused_activation_fn = tfl_add_op.getFusedActivationFunctionAttr();

  if (fused_activation_fn) {
    std::optional<Value> fused_activation_val =
        convertFusedActivation(rewriter, op, output, fused_activation_fn);

    if (!fused_activation_val) return failure();

    rewriter.replaceOp(op, {fused_activation_val.value()});
    return success();
  }

  rewriter.replaceOp(op, {output});
  return success();
}

LogicalResult ConvertTFLSignOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_sign_op = cast<TFL::SignOp>(op);

  RankedTensorType output_type =
      mlir::cast<RankedTensorType>(tfl_sign_op.getResult().getType());

  std::optional<Value> result =
      convertSignOp(rewriter, op, tfl_sign_op.getX(), output_type);
  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});
  return success();
}

LogicalResult ConvertTFLCastOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_cast_op = cast<TFL::CastOp>(op);

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_cast_op.getInput().getType());
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_cast_op.getOutput().getType());
  // Not a ranked tensor input or output
  if (!input_type || !output_type)
    return rewriter.notifyMatchFailure(
        op, "both operand and result must be ranked tensor type.");

  std::optional<Value> result =
      convertCastOp(rewriter, op, tfl_cast_op.getInput(), output_type);
  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});
  return success();
}

LogicalResult ConvertTFLAddOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return matchAndRewriteAddSub<TFL::AddOp, tosa::AddOp>(op, op->getOperands(),
                                                        rewriter);
}

LogicalResult ConvertTFLSubOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return matchAndRewriteAddSub<TFL::SubOp, tosa::SubOp>(op, op->getOperands(),
                                                        rewriter);
}

LogicalResult ConvertTFLMulOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_mul_op = cast<TFL::MulOp>(op);

  std::optional<Value> result =
      convertMultiplyOp(rewriter, op, tfl_mul_op.getResult(),
                        tfl_mul_op.getLhs(), tfl_mul_op.getRhs());

  if (!result) return failure();

  auto fused_activation_fn = tfl_mul_op.getFusedActivationFunctionAttr();

  if (fused_activation_fn) {
    std::optional<Value> fused_activation_val = convertFusedActivation(
        rewriter, op, result.value(), fused_activation_fn);

    if (!fused_activation_val) return failure();

    rewriter.replaceOp(op, {fused_activation_val.value()});
    return success();
  }

  rewriter.replaceOp(op, {result.value()});
  return success();
}

LogicalResult ConvertTFLSquareOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_square_op = cast<TFL::SquareOp>(op);

  std::optional<Value> result =
      convertMultiplyOp(rewriter, op, tfl_square_op.getResult(),
                        tfl_square_op.getX(), tfl_square_op.getX());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});
  return success();
}

LogicalResult ConvertTFLSquaredDifferenceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_squared_op = cast<TFL::SquaredDifferenceOp>(op);

  std::optional<Value> result = convertSquaredDifferenceOp(
      rewriter, op, tfl_squared_op.getResult(), tfl_squared_op.getLhs(),
      tfl_squared_op.getRhs());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});
  return success();
}

LogicalResult ConvertTFLRoundOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_round_op = cast<TFL::RoundOp>(op);

  ShapedType input_type = dyn_cast<ShapedType>(tfl_round_op.getX().getType());
  if (!input_type) {
    return rewriter.notifyMatchFailure(op, "input not shaped tensor type");
  }

  if (mlir::isa<FloatType>(input_type.getElementType())) {
    std::optional<Value> result = convertRoundOp(
        rewriter, op, tfl_round_op.getResult(), tfl_round_op.getX());

    if (!result) return failure();

    rewriter.replaceOp(op, {result.value()});
    return success();

  } else {
    // Round on int is nonsensical. Instead, replace uses of result with the
    // input.
    tfl_round_op.replaceAllUsesWith(tfl_round_op.getX());
    return success();
  }
}

LogicalResult ConvertTFLDivOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_div_op = cast<TFL::DivOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_div_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  auto fused_activation_fn = tfl_div_op.getFusedActivationFunctionAttr();

  Type element_type = output_type.getElementType();
  Value lhs = tfl_div_op.getLhs();
  Value rhs = tfl_div_op.getRhs();
  Value div_op;
  if (mlir::isa<IntegerType>(element_type)) {
    div_op = CreateOpAndInfer<tosa::IntDivOp>(rewriter, op->getLoc(),
                                              output_type, lhs, rhs)
                 .getResult();
  } else {
    auto reciprocal_op = CreateOpAndInfer<tosa::ReciprocalOp>(
        rewriter, op->getLoc(), rhs.getType(), rhs);
    div_op = CreateMulOpAndInfer(rewriter, op, output_type, lhs,
                                 reciprocal_op.getResult())
                 .getResult();
  }

  if (fused_activation_fn) {
    std::optional<Value> fused_activation_val =
        convertFusedActivation(rewriter, op, div_op, fused_activation_fn);

    if (!fused_activation_val) return failure();

    rewriter.replaceOp(op, {fused_activation_val.value()});
    return success();
  }

  rewriter.replaceOp(op, {div_op});

  return success();
}

LogicalResult ConvertTFLMaximumOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_max_op = cast<TFL::MaximumOp>(op);

  Value lhs = tfl_max_op.getLhs();
  Value rhs = tfl_max_op.getRhs();

  ShapedType input_lhs_type = dyn_cast<ShapedType>(lhs.getType());
  ShapedType input_rhs_type = dyn_cast<ShapedType>(rhs.getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_max_op.getResult().getType());

  // Not a shaped tensor output
  if (!input_lhs_type || !input_rhs_type || !output_type) return failure();

  bool input_lhs_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      input_lhs_type.getElementType());
  bool input_rhs_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      input_rhs_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_lhs_is_qtype != output_is_qtype ||
      input_rhs_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  Value output;
  if (output_is_qtype) {
    ShapedType rescale_type = output_type.clone(rewriter.getI32Type());

    Value op1_rescale_lhs = removeZeroPointAndCastToInt32(rewriter, op, lhs, 0);
    Value op2_rescale_rhs = removeZeroPointAndCastToInt32(rewriter, op, rhs, 0);

    auto op3_max_op1_op2 = CreateOpAndInfer<tosa::MaximumOp>(
        rewriter, op->getLoc(), rescale_type, op1_rescale_lhs, op2_rescale_rhs);

    Value op4_rescale_op3 = buildRescaleFromInt32(
        rewriter, op, output_type, op3_max_op1_op2.getResult(), 1.0f, 0);

    output = op4_rescale_op3;
  } else {
    auto op1_max_in = CreateOpAndInfer<tosa::MaximumOp>(rewriter, op->getLoc(),
                                                        output_type, lhs, rhs);

    output = op1_max_in.getResult();
  }

  rewriter.replaceOp(op, {output});

  return success();
}

LogicalResult ConvertTFLMinimumOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_min_op = cast<TFL::MinimumOp>(op);

  Value lhs = tfl_min_op.getLhs();
  Value rhs = tfl_min_op.getRhs();

  ShapedType input_lhs_type = dyn_cast<ShapedType>(lhs.getType());
  ShapedType input_rhs_type = dyn_cast<ShapedType>(rhs.getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_min_op.getResult().getType());
  // Not a shaped tensor output
  if (!input_lhs_type || !input_rhs_type || !output_type) return failure();

  bool input_lhs_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      input_lhs_type.getElementType());
  bool input_rhs_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      input_rhs_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_lhs_is_qtype != output_is_qtype ||
      input_rhs_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  Value output;
  if (output_is_qtype) {
    ShapedType rescale_type = output_type.clone(rewriter.getI32Type());

    Value op1_rescale_lhs = removeZeroPointAndCastToInt32(rewriter, op, lhs, 0);
    Value op2_rescale_rhs = removeZeroPointAndCastToInt32(rewriter, op, rhs, 0);
    auto op3_min_op1_op2 = CreateOpAndInfer<tosa::MinimumOp>(
        rewriter, op->getLoc(), rescale_type, op1_rescale_lhs, op2_rescale_rhs);

    Value op4_rescale_op3 = buildRescaleFromInt32(
        rewriter, op, output_type, op3_min_op1_op2.getResult(), 1.0f, 0);

    output = op4_rescale_op3;
  } else {
    auto op1_min_in = CreateOpAndInfer<tosa::MinimumOp>(rewriter, op->getLoc(),
                                                        output_type, lhs, rhs);

    output = op1_min_in.getResult();
  }

  rewriter.replaceOp(op, {output});

  return success();
}

LogicalResult ConvertTFLFloorDivOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_floordiv_op = cast<TFL::FloorDivOp>(op);

  std::optional<Value> result =
      convertFloorDivOp(rewriter, op, tfl_floordiv_op.getResult(),
                        tfl_floordiv_op.getLhs(), tfl_floordiv_op.getRhs());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLFloorModOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_floormod_op = cast<TFL::FloorModOp>(op);

  std::optional<Value> result =
      convertFloorModOp(rewriter, op, tfl_floormod_op.getResult(),
                        tfl_floormod_op.getLhs(), tfl_floormod_op.getRhs());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLAddNOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_addn_op = cast<TFL::AddNOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_addn_op.getResult().getType());
  // Not a shaped output
  if (!output_type) return failure();

  SmallVector<Value> inputs(tfl_addn_op.getInputs());

  assert(inputs.size() >= 2);

  Value x = inputs[0];
  Value y = inputs[1];
  Value newValue =
      CreateOpAndInfer<tosa::AddOp>(rewriter, op->getLoc(), output_type, x, y);
  for (int i = 2; i < inputs.size(); i++) {
    Value next = inputs[i];
    newValue = CreateOpAndInfer<tosa::AddOp>(rewriter, op->getLoc(),
                                             output_type, next, newValue);
  }

  rewriter.replaceOp(op, {newValue});

  return success();
}

LogicalResult ConvertTFLAveragePool2DOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_avgpool_op = cast<TFL::AveragePool2DOp>(op);

  ShapedType input_type =
      dyn_cast<ShapedType>(tfl_avgpool_op.getInput().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_avgpool_op.getResult().getType());
  // Not a shaped output
  if (!output_type) return failure();

  // Kernels and strides are dimensionally ordered
  SmallVector<int64_t, 4> i64array({1, 1, 1, 1});
  DenseI64ArrayAttr kernel_size;
  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr pad;
  // Pooling has no non-unit dilation
  DenseI64ArrayAttr dilation = rewriter.getDenseI64ArrayAttr({1, 1});
  {
    int64_t kernel_h = tfl_avgpool_op.getFilterHeight();
    int64_t kernel_w = tfl_avgpool_op.getFilterWidth();
    kernel_size = rewriter.getDenseI64ArrayAttr({kernel_h, kernel_w});
    // i64array is formatted as NHWC now
    i64array[1] = kernel_h;
    i64array[2] = kernel_w;
  }
  {
    int64_t stride_h = tfl_avgpool_op.getStrideH();
    int64_t stride_w = tfl_avgpool_op.getStrideW();
    stride = rewriter.getDenseI64ArrayAttr({stride_h, stride_w});
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tfl_avgpool_op.getPadding().str(), &tf_pad).ok())
      return failure();

    // Pooling has no non-unit dilation
    DenseI64ArrayAttr dilation = rewriter.getDenseI64ArrayAttr({1, 1});

    RankedTensorType filter_type = RankedTensorType::get(
        llvm::ArrayRef(i64array), rewriter.getIntegerType(64));

    // TFLite doesn't support explicit padding
    if (!getPaddingValuesFromPadType(
            tf_pad,
            tensorflow::FORMAT_NHWC,  // TFLite only supports this
            1,                        // tensorflow::FORMAT_OHWI,
            input_type, filter_type, stride, dilation, rewriter, pad))
      return failure();
  }

  Value avg_pool_input = getInputSlicedToItsUsedSize(
      rewriter, op, tensorflow::FORMAT_NHWC, input_type,
      tfl_avgpool_op.getInput(), kernel_size, pad, stride, dilation);

  auto average_etype = input_type.getElementType();
  auto average_type = output_type.clone(average_etype);

  // Tosa supports FP16 and FP32 accumulator type for FP16 input. When the time
  // FP16 is supported, the accumulator type can be selected based on trade-off
  // between performance and accuracy. Set to FP32 by default.
  TypeAttr acc_attr = mlir::isa<FloatType>(average_etype)
                          ? mlir::TypeAttr::get(rewriter.getF32Type())
                          : mlir::TypeAttr::get(rewriter.getIntegerType(32));

  Value result;
  if (mlir::isa<quant::UniformQuantizedType>(average_etype)) {
    // TensorFlow Lite doesn't use the zero point when calculating
    // quantized average pool, while TOSA does. Force the TOSA
    // zero_points to zero to ensure that the calculations match
    Location loc = op->getLoc();
    const std::optional<Value> input_zp =
      tosa::createZeroPointTensor(rewriter, loc, avg_pool_input.getType(), 0);
    if (!input_zp.has_value())
      return op->emitError("Failed to create input zero-point tensor for AvgPool2D op.");

    const Value empty_output_val = rewriter.create<tensor::EmptyOp>(loc,
      average_type.getShape(), average_type.getElementType());
    const std::optional<Value> output_zp =
      tosa::createZeroPointTensor(rewriter, loc, empty_output_val.getType(), 0);
    if (!output_zp.has_value())
      return op->emitError("Failed to create output zero-point tensor for AvgPool2D op.");

    result = CreateOpAndInfer<tosa::AvgPool2dOp>(
        rewriter, op->getLoc(), average_type, avg_pool_input, input_zp.value(),
        output_zp.value(), kernel_size, stride, pad, acc_attr);
  } else {
    result = CreateOpAndInfer<tosa::AvgPool2dOp>(
        rewriter, op->getLoc(), average_type, avg_pool_input, kernel_size,
        stride, pad, acc_attr);
  }

  if (average_type != output_type) {
    result = CreateOpAndInfer<tosa::CastOp>(rewriter, op->getLoc(), output_type,
                                            result);
  }

  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult ConvertTFLMaxPool2DOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_maxpool_op = cast<TFL::MaxPool2DOp>(op);

  ShapedType input_type =
      dyn_cast<ShapedType>(tfl_maxpool_op.getInput().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_maxpool_op.getResult().getType());
  // Not a shaped type
  if (!output_type) return failure();

  // Kernels and strides are dimensionally ordered
  SmallVector<int64_t, 4> i64array({1, 1, 1, 1});
  DenseI64ArrayAttr kernel_size;
  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr pad;
  {
    int64_t kernel_h = tfl_maxpool_op.getFilterHeight();
    int64_t kernel_w = tfl_maxpool_op.getFilterWidth();
    kernel_size = rewriter.getDenseI64ArrayAttr({kernel_h, kernel_w});
    // i64array is formatted as NHWC now
    i64array[1] = kernel_h;
    i64array[2] = kernel_w;
  }
  {
    int64_t stride_h = tfl_maxpool_op.getStrideH();
    int64_t stride_w = tfl_maxpool_op.getStrideW();
    stride = rewriter.getDenseI64ArrayAttr({stride_h, stride_w});
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tfl_maxpool_op.getPadding().str(), &tf_pad).ok())
      return failure();

    // Pooling has no non-unit dilation
    DenseI64ArrayAttr dilation = rewriter.getDenseI64ArrayAttr({1, 1});

    RankedTensorType filter_type =
        RankedTensorType::get(i64array, rewriter.getIntegerType(64));

    // TFLite doesn't support explicit padding
    if (!getPaddingValuesFromPadType(
            tf_pad,
            tensorflow::FORMAT_NHWC,  // TFLite only supports this
            1,                        // tensorflow::FORMAT_OHWI,
            input_type, filter_type, stride, dilation, rewriter, pad))
      return failure();
  }

  CreateReplaceOpAndInfer<tosa::MaxPool2dOp>(rewriter, op, output_type,
                                             tfl_maxpool_op.getInput(),
                                             kernel_size, stride, pad);
  return success();
}

// Returns a new type based on an input type and slicing information.
// If the input is quantized per axis, slices the scale and zp arrays.
// In any other case, returns the original type.
RankedTensorType getTypeForSlice(RankedTensorType type, int64_t slice_dim,
                                 int64_t slice_size, int64_t offset) {
  if (auto per_channel_qtype =
          dyn_cast<quant::UniformQuantizedPerAxisType>(type.getElementType())) {
    if (per_channel_qtype.getQuantizedDimension() != slice_dim) return type;
    SmallVector<double> output_scale_arr(
        per_channel_qtype.getScales().begin() + offset,
        per_channel_qtype.getScales().begin() + offset + slice_size);
    SmallVector<int64_t> output_zp_arr(
        per_channel_qtype.getZeroPoints().begin() + offset,
        per_channel_qtype.getZeroPoints().begin() + offset + slice_size);
    auto output_per_channel_qtype = quant::UniformQuantizedPerAxisType::get(
        per_channel_qtype.getFlags(), per_channel_qtype.getStorageType(),
        per_channel_qtype.getExpressedType(), output_scale_arr, output_zp_arr,
        per_channel_qtype.getQuantizedDimension(),
        per_channel_qtype.getStorageTypeMin(),
        per_channel_qtype.getStorageTypeMax());
    return RankedTensorType::get(type.getShape(), output_per_channel_qtype);
  }
  return type;
}

Value lowerGroupedConvolution(TFL::Conv2DOp op, PatternRewriter& rewriter) {
  auto input_type = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto filter_type = dyn_cast<RankedTensorType>(op.getFilter().getType());
  auto bias_type = dyn_cast<RankedTensorType>(op.getBias().getType());
  auto output_type = dyn_cast<RankedTensorType>(op.getResult().getType());

  // The slicing on the input/output is done on dim 3, while on the filter/bias
  // it's done on dim 0.
  int64_t input_slice_dim = 3;
  int64_t filter_slice_dim = 0;
  int64_t bias_slice_dim = 0;
  int64_t output_slice_dim = 3;
  int64_t input_channels = input_type.getDimSize(input_slice_dim);
  int64_t ic_slice_size = filter_type.getDimSize(input_slice_dim);
  int64_t num_groups = input_channels / ic_slice_size;

  // When slicing the filter/bias, the slice size is determined by the number of
  // slices on the input.
  int64_t oc_slice_size = filter_type.getDimSize(filter_slice_dim) / num_groups;

  SmallVector<Value> convolutions;
  convolutions.reserve(num_groups);
  auto rank = input_type.getRank();

  // Input size vector
  SmallVector<int64_t, 4> input_size_vals(input_type.getShape().begin(),
                                          input_type.getShape().end());
  input_size_vals.back() = ic_slice_size;
  auto input_slice_ty =
      RankedTensorType::get(input_size_vals, input_type.getElementType());

  // Filter size vector
  SmallVector<int64_t, 4> filter_size_vals(filter_type.getShape().begin(),
                                           filter_type.getShape().end());
  filter_size_vals.front() =
      filter_type.getDimSize(filter_slice_dim) / num_groups;
  DenseI64ArrayAttr filter_size =
      rewriter.getDenseI64ArrayAttr(filter_size_vals);
  auto filter_slice_ty =
      RankedTensorType::get(filter_size_vals, filter_type.getElementType());

  // Bias size vector
  int64_t bias_size_val = bias_type.getDimSize(bias_slice_dim) / num_groups;
  auto bias_slice_ty =
      RankedTensorType::get(bias_size_val, bias_type.getElementType());

  auto per_conv_out_ty = RankedTensorType::get(
      {output_type.getDimSize(0), output_type.getDimSize(1),
       output_type.getDimSize(2), oc_slice_size},
      output_type.getElementType());

  // Create a separate convolution for each group
  for (int i = 0; i < num_groups; ++i) {
    auto verified_input_slice_ty = getTypeForSlice(
        input_slice_ty, input_slice_dim, ic_slice_size, i * ic_slice_size);
    auto verified_filter_slice_ty = getTypeForSlice(
        filter_slice_ty, filter_slice_dim, oc_slice_size, i * oc_slice_size);
    auto verified_bias_slice_ty = getTypeForSlice(
        bias_slice_ty, bias_slice_dim, oc_slice_size, i * oc_slice_size);
    auto verified_per_conv_out_ty = getTypeForSlice(
        per_conv_out_ty, output_slice_dim, oc_slice_size, i * oc_slice_size);

    // Slice the input
    SmallVector<int64_t, 4> input_start_vals(rank, 0);
    input_start_vals.back() = i * ic_slice_size;

    auto slice_input = rewriter.createOrFold<tosa::SliceOp>(
        op->getLoc(), verified_input_slice_ty, op.getInput(),
        getTosaConstShape(rewriter, op->getLoc(), input_start_vals),
        getTosaConstShape(rewriter, op->getLoc(), input_size_vals));

    // Slice the filter
    SmallVector<int64_t, 4> filter_start_vals(rank, 0);
    filter_start_vals.front() = i * oc_slice_size;
    DenseI64ArrayAttr filter_start =
        rewriter.getDenseI64ArrayAttr(filter_start_vals);

    auto slice_filter = rewriter.createOrFold<tosa::SliceOp>(
        op->getLoc(), verified_filter_slice_ty, op.getFilter(),
        getTosaConstShape(rewriter, op->getLoc(), filter_start_vals),
        getTosaConstShape(rewriter, op->getLoc(), filter_size_vals));

    // Slice the bias
    int64_t bias_start_val = i * oc_slice_size;

    auto slice_bias = rewriter.createOrFold<tosa::SliceOp>(
        op->getLoc(), verified_bias_slice_ty, op.getBias(),
        getTosaConstShape(rewriter, op->getLoc(), bias_start_val),
        getTosaConstShape(rewriter, op->getLoc(), bias_size_val));

    // Create a convolution for each set of slices
    auto conv = rewriter.create<TFL::Conv2DOp>(
        op->getLoc(), verified_per_conv_out_ty, slice_input, slice_filter,
        slice_bias, op.getDilationHFactor(), op.getDilationWFactor(),
        op.getFusedActivationFunction(), op.getPadding(), op.getStrideH(),
        op.getStrideW());

    convolutions.push_back(conv.getResult());
  }

  return rewriter.createOrFold<tosa::ConcatOp>(op->getLoc(), output_type,
                                               convolutions, output_slice_dim);
}

LogicalResult ConvertTFLConv2DOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_conv2d_op = cast<TFL::Conv2DOp>(op);

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_conv2d_op.getInput().getType());
  RankedTensorType filter_type =
      dyn_cast<RankedTensorType>(tfl_conv2d_op.getFilter().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_conv2d_op.getResult().getType());
  // Not a ranked tensor output
  if (!input_type) return failure();
  if (!output_type) return failure();
  if (!filter_type) return failure();

  bool input_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(input_type.getElementType());
  bool filter_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(filter_type.getElementType());
  bool output_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(output_type.getElementType());

  if ((input_is_qtype != filter_is_qtype) ||
      (input_is_qtype != output_is_qtype)) {
    return rewriter.notifyMatchFailure(
        op,
        "input/filter/output tensor should "
        "be all quantized or all floating-point");
  }

  int64_t input_channels = input_type.getDimSize(3);
  int64_t filter_channels = filter_type.getDimSize(3);
  if (input_channels != filter_channels &&
      input_channels % filter_channels == 0) {
    rewriter.replaceOp(op, lowerGroupedConvolution(tfl_conv2d_op, rewriter));
    return success();
  }

  DenseI64ArrayAttr kernel_size;
  DenseI64ArrayAttr pad;
  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr dilation;
  {
    // TFLite filter format is tensorflow::FORMAT_OHWI
    int64_t kernel_h = filter_type.getDimSize(1);
    int64_t kernel_w = filter_type.getDimSize(2);
    kernel_size = rewriter.getDenseI64ArrayAttr({kernel_h, kernel_w});
  }
  {
    int64_t stride_h = tfl_conv2d_op.getStrideH();
    int64_t stride_w = tfl_conv2d_op.getStrideW();
    stride = rewriter.getDenseI64ArrayAttr({stride_h, stride_w});
  }
  {
    int64_t dilation_h = tfl_conv2d_op.getDilationHFactor();
    int64_t dilation_w = tfl_conv2d_op.getDilationWFactor();
    dilation = rewriter.getDenseI64ArrayAttr({dilation_h, dilation_w});
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tfl_conv2d_op.getPadding().str(), &tf_pad).ok())
      return failure();

    // TFLite doesn't support explicit padding
    if (!getPaddingValuesFromPadType(
            tf_pad,
            tensorflow::FORMAT_NHWC,  // TFLite only supports this
            1,                        // tensorflow::FORMAT_OHWI,
            input_type, filter_type, stride, dilation, rewriter, pad))
      return failure();
  }

  Value unquantized_bias = tfl_conv2d_op.getBias();
  Type bias_ety =
      output_is_qtype ? rewriter.getI32Type() : output_type.getElementType();
  if (unquantized_bias) {
    Type new_bias_ety = getElementTypeOrSelf(unquantized_bias.getType());
    if (auto qtype = mlir::dyn_cast<mlir::quant::QuantizedType>(new_bias_ety)) {
      new_bias_ety = qtype.getStorageType();
    }
    if (new_bias_ety.getIntOrFloatBitWidth() >
        bias_ety.getIntOrFloatBitWidth()) {
      bias_ety = new_bias_ety;
    }
  }

  // TFLite only supports NHWC format
  Value conv2d_input = getInputSlicedToItsUsedSize(
      rewriter, op, tensorflow::FORMAT_NHWC, input_type,
      tfl_conv2d_op.getInput(), kernel_size, pad, stride, dilation);

  auto acc_type =
      getConvAccTypeAttr(rewriter,
                         /* input_etype = */ input_type.getElementType(),
                         /* output_etype = */ bias_ety);

  auto a1_conv2d_op = CreateOpAndInfer<tosa::Conv2DOp>(
      rewriter, op->getLoc(), output_type.clone(bias_ety), conv2d_input,
      tfl_conv2d_op.getFilter(), unquantized_bias, pad, stride, dilation,
      acc_type);

  Value conv2d_output;
  if (input_is_qtype) {
    conv2d_output =
        buildRescaleOpConvOutput(rewriter, op, a1_conv2d_op.getResult(),
                                 input_type, filter_type, output_type);
  } else {
    conv2d_output = a1_conv2d_op.getResult();
  }

  auto fused_activation_fn = tfl_conv2d_op.getFusedActivationFunctionAttr();

  if (fused_activation_fn) {
    std::optional<Value> fused_activation_val = convertFusedActivation(
        rewriter, op, conv2d_output, fused_activation_fn);

    if (!fused_activation_val) return failure();

    rewriter.replaceOp(op, {fused_activation_val.value()});
    return success();
  }

  rewriter.replaceOp(op, {conv2d_output});

  return success();
}

LogicalResult ConvertTFLConv3DOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_conv3d_op = cast<TFL::Conv3DOp>(op);
  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_conv3d_op.getInput().getType());
  RankedTensorType filter_type =
      dyn_cast<RankedTensorType>(tfl_conv3d_op.getFilter().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_conv3d_op.getResult().getType());

  if (!input_type | !filter_type || !output_type) {
    return rewriter.notifyMatchFailure(
        op, "input/filter/output are not all a ranked tensor");
  }

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::QuantizedType>();
  bool filter_is_qtype =
      filter_type.getElementType().isa<mlir::quant::QuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::QuantizedType>();

  if ((input_is_qtype != filter_is_qtype) ||
      (input_is_qtype != output_is_qtype)) {
    return rewriter.notifyMatchFailure(
        op,
        "ConvertTFLConv3DOp: input/filter/output tensor should "
        "be all quantized or all floating-point.");
  }

  DenseI64ArrayAttr kernel_size;
  DenseI64ArrayAttr pad;
  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr dilation;
  {
    // TFLite filter format is DHWIO
    int64_t kernel_d = filter_type.getDimSize(0);
    int64_t kernel_h = filter_type.getDimSize(1);
    int64_t kernel_w = filter_type.getDimSize(2);
    kernel_size = rewriter.getDenseI64ArrayAttr({kernel_d, kernel_h, kernel_w});
  }
  {
    int64_t stride_d = tfl_conv3d_op.getStrideD();
    int64_t stride_h = tfl_conv3d_op.getStrideH();
    int64_t stride_w = tfl_conv3d_op.getStrideW();
    stride = rewriter.getDenseI64ArrayAttr({stride_d, stride_h, stride_w});
  }
  {
    int64_t dilation_d = tfl_conv3d_op.getDilationDFactor();
    int64_t dilation_h = tfl_conv3d_op.getDilationHFactor();
    int64_t dilation_w = tfl_conv3d_op.getDilationWFactor();
    dilation =
        rewriter.getDenseI64ArrayAttr({dilation_d, dilation_h, dilation_w});
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tfl_conv3d_op.getPadding().str(), &tf_pad).ok()) {
      return failure();
    }

    // TFLite doesn't support explicit padding
    if (!getPaddingValuesFromPadType(
            tf_pad,
            tensorflow::FORMAT_NHWC,  // TFLite only supports NDHWC format,
                                      // tensorflow::FORMAT_NHWC is used for
                                      // both rank 4 and rank 5 tensors
            0,                        // TFLite filter is DHWIO
            input_type, filter_type, stride, dilation, rewriter, pad)) {
      return failure();
    }
  }

  Value unquantized_bias = tfl_conv3d_op.getBias();
  if (!dyn_cast<RankedTensorType>(unquantized_bias.getType())) {
    // The bias may actually be typed "None" which has no value. TOSA requires
    // bias to be an array of output_channel_count values, so create a constant
    // of the appropriate number and type of zeros.
    auto bias_dim = filter_type.getShape().back();
    RankedTensorType bias_type =
        RankedTensorType::get({bias_dim}, filter_type.getElementType());
    auto bias_attr = rewriter.getZeroAttr(bias_type);
    unquantized_bias = CreateOpAndInfer<tosa::ConstOp>(
        rewriter, op->getLoc(), bias_type, bias_attr.cast<ElementsAttr>());
  }

  // TFLite only supports NDHWC format, tensorflow::FORMAT_NHWC is used for both
  // rank 4 and rank 5 tensors
  Value conv3d_input = getInputSlicedToItsUsedSize(
      rewriter, op, tensorflow::FORMAT_NHWC, input_type,
      tfl_conv3d_op.getInput(), kernel_size, pad, stride, dilation);

  Type bias_ety =
      unquantized_bias.getType().cast<ShapedType>().getElementType();

  auto acc_type =
      getConvAccTypeAttr(rewriter,
                         /* input_etype = */ input_type.getElementType(),
                         /* output_etype = */ bias_ety);

  std::optional<Value> a1_conv3d_op = convertConv3DCommon(
      rewriter, op, output_type.clone(bias_ety), conv3d_input,
      tfl_conv3d_op.getFilter(), unquantized_bias, pad, stride, dilation,
      acc_type, StringRef("NDHWC"));

  if (!a1_conv3d_op) return failure();

  Value conv3d_output =
      input_is_qtype
          ? buildRescaleOpConvOutput(rewriter, op, a1_conv3d_op.value(),
                                     input_type, filter_type, output_type)
          : a1_conv3d_op.value();

  if (auto fused_activation_fn =
          tfl_conv3d_op.getFusedActivationFunctionAttr()) {
    std::optional<Value> fused_activation_val = convertFusedActivation(
        rewriter, op, conv3d_output, fused_activation_fn);

    if (!fused_activation_val) return failure();

    rewriter.replaceOp(op, {fused_activation_val.value()});
    return success();
  }

  rewriter.replaceOp(op, {conv3d_output});

  return success();
}

LogicalResult ConvertTFLTransposeConvOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_conv_op = cast<TFL::TransposeConvOp>(op);

  ShapedType input_type =
      dyn_cast<ShapedType>(tfl_conv_op.getInput().getType());
  ShapedType filter_type =
      dyn_cast<ShapedType>(tfl_conv_op.getWeights().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_conv_op.getResult().getType());

  // Not a ranked tensor output
  if (!input_type) return failure();
  if (!output_type) return failure();
  if (!filter_type) return failure();

  bool input_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(input_type.getElementType());
  bool filter_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(filter_type.getElementType());
  bool output_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(output_type.getElementType());

  const bool has_bias =
      tfl_conv_op.getBias() && !isa<NoneType>(tfl_conv_op.getBias().getType());

  if (has_bias) {
    RankedTensorType bias_type =
        dyn_cast<RankedTensorType>(tfl_conv_op.getBias().getType());
    bool bias_is_qtype =
        isa<mlir::quant::QuantizedType>(bias_type.getElementType());

    if (input_is_qtype != bias_is_qtype) {
      return rewriter.notifyMatchFailure(
          op,
          "input/bias tensor should "
          "be all quantized or all floating-point");
    }
  }

  if ((input_is_qtype != filter_is_qtype) ||
      (input_is_qtype != output_is_qtype)) {
    return rewriter.notifyMatchFailure(
        op,
        "input/filter/output tensor should "
        "be all quantized or all floating-point");
  }

  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr outpad;
  {
    int64_t stride_h = tfl_conv_op.getStrideH();
    int64_t stride_w = tfl_conv_op.getStrideW();
    stride = rewriter.getDenseI64ArrayAttr({stride_h, stride_w});
  }

  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tfl_conv_op.getPadding().str(), &tf_pad).ok())
      return failure();

    if (!getTransposeConv2dPaddingValues(
            tf_pad,
            tensorflow::FORMAT_NHWC,  // TFLite only supports this
            1,                        // tensorflow::FORMAT_OHWI,
            input_type, filter_type, output_type, stride, rewriter, outpad))
      return failure();
  }

  int output_channel = 0;
  // TODO(suderman): We need to figure out how to guarantee output channel
  // propagation.
  if (output_type.hasRank()) {
    output_channel = output_type.getDimSize(3);
  } else if (filter_type.hasRank()) {
    output_channel = filter_type.getDimSize(0);
  } else {
    return failure();
  }

  Value bias_val;
  if (has_bias) {
    bias_val = tfl_conv_op.getBias();
  } else {
    std::optional<Value> zero_bias;
    if (input_is_qtype) {
      uint32_t input_bits =
          cast<mlir::quant::QuantizedType>(input_type.getElementType())
              .getStorageTypeIntegralWidth();
      uint32_t weight_bits =
          cast<mlir::quant::QuantizedType>(filter_type.getElementType())
              .getStorageTypeIntegralWidth();

      if (input_bits == 16 && weight_bits == 8) {
        // For signed 16x8, the output is accumulated into int48
        SmallVector<APInt> vec(output_channel, APInt(48, 0, true));
        zero_bias = getConstTensor<APInt>(rewriter, op, vec, {output_channel});
      } else {
        SmallVector<int32_t> vec(output_channel, 0);
        zero_bias =
            getConstTensor<int32_t>(rewriter, op, vec, {output_channel});
      }
    } else {
      SmallVector<float> vec(output_channel, 0.0f);
      zero_bias = getConstTensor<float>(rewriter, op, vec, {output_channel});
    }

    if (!zero_bias) return failure();
    bias_val = zero_bias.value();
  }

  Type bias_ety = cast<ShapedType>(bias_val.getType()).getElementType();

  auto acc_type =
      getConvAccTypeAttr(rewriter,
                         /* input_etype = */ input_type.getElementType(),
                         /* output_etype = */ bias_ety);

  auto a1_conv2d_op = CreateOpAndInfer<tosa::TransposeConv2DOp>(
      rewriter, op->getLoc(), output_type.clone(bias_ety),
      tfl_conv_op.getInput(), tfl_conv_op.getWeights(), bias_val,
      outpad, stride, acc_type);

  Value conv2d_output;
  if (input_is_qtype) {
    conv2d_output =
        buildRescaleOpConvOutput(rewriter, op, a1_conv2d_op.getResult(),
                                 input_type, filter_type, output_type);
  } else {
    conv2d_output = a1_conv2d_op.getResult();
  }

  auto fused_activation_fn = tfl_conv_op.getFusedActivationFunctionAttr();

  if (fused_activation_fn) {
    std::optional<Value> fused_activation_val = convertFusedActivation(
        rewriter, op, conv2d_output, fused_activation_fn);

    if (!fused_activation_val) return failure();

    rewriter.replaceOp(op, {fused_activation_val.value()});
    return success();
  }

  rewriter.replaceOp(op, {conv2d_output});

  return success();
}

LogicalResult ConvertTFLDepthwiseConv2DOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_conv2d_op = cast<TFL::DepthwiseConv2DOp>(op);

  ShapedType input_type =
      dyn_cast<ShapedType>(tfl_conv2d_op.getInput().getType());
  ShapedType filter_type =
      dyn_cast<ShapedType>(tfl_conv2d_op.getFilter().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_conv2d_op.getResult().getType());
  // Not a shaped output
  if (!input_type) return failure();
  if (!output_type) return failure();
  if (!filter_type) return failure();

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::QuantizedType>();
  bool filter_is_qtype =
      filter_type.getElementType().isa<mlir::quant::QuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::QuantizedType>();

  if ((input_is_qtype != filter_is_qtype) ||
      (input_is_qtype != output_is_qtype)) {
    return rewriter.notifyMatchFailure(
        op,
        "input/filter/output tensor should "
        "be all quantized or all floating-point");
  }

  // We need the filter shape to compute the transpose.
  if (!filter_type.hasRank()) return failure();
  auto filter_shape = filter_type.getShape();
  // Operator depthwiseConv2D
  // TFLite orders the depthwiseConv2D filter in IHWO, while TOSA orders
  // filter in HWIO
  //
  // The lowering reorders the filter.
  //
  // a1_transpose = tosa.transpose(filter, {1, 2, 3, 0})   // HWIO
  // a2_reshape = tosa.reshape(filter, H, W, depth_multiplier, I /
  // depth_multiplier)
  // a3_transpose_conv2d = tosa.transpose_conv2d(input, a2_reshape, padding,
  // stride, dilation)

  DenseI64ArrayAttr kernel_size;
  DenseI64ArrayAttr pad;
  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr dilation;
  auto depth_multiplier = tfl_conv2d_op.getDepthMultiplierAttr();

  {
    // DepthwiseConv2D TFLite filter format is tensorflow::FORMAT_IHWO
    int64_t kernel_h = filter_type.getDimSize(1);
    int64_t kernel_w = filter_type.getDimSize(2);
    kernel_size = rewriter.getDenseI64ArrayAttr({kernel_h, kernel_w});
  }
  {
    int64_t stride_h = tfl_conv2d_op.getStrideH();
    int64_t stride_w = tfl_conv2d_op.getStrideW();
    stride = rewriter.getDenseI64ArrayAttr({stride_h, stride_w});
  }
  {
    int64_t dilation_h = tfl_conv2d_op.getDilationHFactor();
    int64_t dilation_w = tfl_conv2d_op.getDilationWFactor();
    dilation = rewriter.getDenseI64ArrayAttr({dilation_h, dilation_w});
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(tfl_conv2d_op.getPadding().str(), &tf_pad).ok())
      return failure();

    if (!getPaddingValuesFromPadType(
            tf_pad,
            tensorflow::FORMAT_NHWC,  // TFLite only supports this
            1,                        // tensorflow::FORMAT_OHWI,
            input_type, filter_type, stride, dilation, rewriter, pad))
      return failure();
  }

  SmallVector<int64_t, 4> a1_transpose_dims;
  a1_transpose_dims.push_back(filter_shape[1]);
  a1_transpose_dims.push_back(filter_shape[2]);
  a1_transpose_dims.push_back(filter_shape[3]);
  a1_transpose_dims.push_back(filter_shape[0]);

  SmallVector<int64_t, 4> a2_reshape_dims;
  a2_reshape_dims.push_back(a1_transpose_dims[0]);
  a2_reshape_dims.push_back(a1_transpose_dims[1]);
  a2_reshape_dims.push_back(a1_transpose_dims[2] / depth_multiplier.getInt());
  a2_reshape_dims.push_back(depth_multiplier.getInt());

  auto a1_filter_transpose_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(),
      RankedTensorType::get(ArrayRef<int64_t>(a1_transpose_dims),
                            filter_type.getElementType()),
      tfl_conv2d_op.getFilter(), rewriter.getDenseI32ArrayAttr({1, 2, 3, 0}));

  auto a2_reshape_dims_value = getTosaConstShape(rewriter, op->getLoc(), a2_reshape_dims);
  auto a2_filter_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      RankedTensorType::get(ArrayRef<int64_t>(a2_reshape_dims),
                            filter_type.getElementType()),
      a1_filter_transpose_op.getResult(), a2_reshape_dims_value);

  Type bias_ety =
      output_is_qtype ? rewriter.getI32Type() : output_type.getElementType();

  Value unquantized_bias = tfl_conv2d_op.getBias();
  if (unquantized_bias) {
    Type new_bias_ety = getElementTypeOrSelf(unquantized_bias.getType());
    if (auto qtype = new_bias_ety.dyn_cast<mlir::quant::QuantizedType>()) {
      new_bias_ety = qtype.getStorageType();
    }
    if (new_bias_ety.getIntOrFloatBitWidth() >
        bias_ety.getIntOrFloatBitWidth()) {
      bias_ety = new_bias_ety;
    }
  }

  // TFLite only supports NHWC format
  Value conv2d_input = getInputSlicedToItsUsedSize(
      rewriter, op, tensorflow::FORMAT_NHWC, input_type,
      tfl_conv2d_op.getInput(), kernel_size, pad, stride, dilation);

  auto acc_type =
      getConvAccTypeAttr(rewriter,
                         /* input_etype = */ input_type.getElementType(),
                         /* output_etype = */ bias_ety);

  auto a3_depthwise_conv2d_op = CreateOpAndInfer<tosa::DepthwiseConv2DOp>(
      rewriter, op->getLoc(), output_type.clone(bias_ety), conv2d_input,
      a2_filter_reshape_op.getResult(), unquantized_bias, pad, stride, dilation,
      acc_type);

  Value conv2d_output;
  if (input_is_qtype) {
    conv2d_output = buildRescaleOpConvOutput(
        rewriter, op, a3_depthwise_conv2d_op.getResult(), input_type,
        filter_type, output_type);
  } else {
    conv2d_output = a3_depthwise_conv2d_op.getResult();
  }

  auto fused_activation_fn = tfl_conv2d_op.getFusedActivationFunctionAttr();

  if (fused_activation_fn) {
    std::optional<Value> fused_activation_val = convertFusedActivation(
        rewriter, op, conv2d_output, fused_activation_fn);

    if (!fused_activation_val) return failure();

    rewriter.replaceOp(op, {fused_activation_val.value()});
    return success();
  }

  rewriter.replaceOp(op, {conv2d_output});

  return success();
}

LogicalResult ConvertTFLBatchMatMulOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_mm_op = cast<TFL::BatchMatMulOp>(op);
  auto result_ty = mlir::cast<ShapedType>(tfl_mm_op.getType());
  Value lhs = tfl_mm_op.getX();
  Value rhs = tfl_mm_op.getY();

  if (broadcastLowRankTensor(rewriter, op, lhs, rhs).failed()) {
    return rewriter.notifyMatchFailure(
        op, "failed to broadcast low rank input tensor");
  }

  RankedTensorType lhs_ty = dyn_cast<RankedTensorType>(lhs.getType());
  RankedTensorType rhs_ty = dyn_cast<RankedTensorType>(rhs.getType());
  bool transpose_lhs = tfl_mm_op.getAdjX();
  bool transpose_rhs = tfl_mm_op.getAdjY();

  if (!lhs_ty || !rhs_ty) {
    return rewriter.notifyMatchFailure(op,
                                       "lhs/rhs tensor should be all ranked");
  }

  bool lhs_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(lhs_ty.getElementType());
  bool rhs_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(rhs_ty.getElementType());
  bool result_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(result_ty.getElementType());

  if ((lhs_is_qtype != rhs_is_qtype) || (lhs_is_qtype != result_is_qtype)) {
    return rewriter.notifyMatchFailure(
        op,
        "lhs/rhs/output tensor should "
        "be all quantized or all floating-point");
  }

  auto batch_dims = lhs_ty.getShape().drop_back(2);
  if (batch_dims.size() != 1) {
    int64_t N = 1;
    for (auto d : batch_dims) {
      N = N < 0 || d < 0 ? -1 : N * d;
    }

    llvm::SmallVector<int64_t> new_lhs_shape{N};
    llvm::SmallVector<int64_t> new_rhs_shape{N};
    auto lhs_shape_end = lhs_ty.getShape().take_back(2);
    auto rhs_shape_end = rhs_ty.getShape().take_back(2);

    new_lhs_shape.append(lhs_shape_end.begin(), lhs_shape_end.end());
    new_rhs_shape.append(rhs_shape_end.begin(), rhs_shape_end.end());

    auto new_lhs_shape_value = getTosaConstShape(rewriter, op->getLoc(), new_lhs_shape);
    lhs = CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(),
        UnrankedTensorType::get(lhs_ty.getElementType()), lhs,
        new_lhs_shape_value);

    auto new_rhs_shape_value = getTosaConstShape(rewriter, op->getLoc(), new_rhs_shape);
    rhs = CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(),
        UnrankedTensorType::get(rhs_ty.getElementType()), rhs,
        new_rhs_shape_value);
    lhs_ty = lhs.getType().cast<RankedTensorType>();
    rhs_ty = rhs.getType().cast<RankedTensorType>();
  }

  if (transpose_lhs) {
    Type output_type = UnrankedTensorType::get(lhs_ty.getElementType());
    lhs = CreateOpAndInfer<tosa::TransposeOp>(
              rewriter, op->getLoc(), output_type, lhs,
              rewriter.getDenseI32ArrayAttr({0, 2, 1}))
              .getResult();
  }

  if (transpose_rhs) {
    Type output_type = UnrankedTensorType::get(rhs_ty.getElementType());
    rhs = CreateOpAndInfer<tosa::TransposeOp>(
              rewriter, op->getLoc(), output_type, rhs,
              rewriter.getDenseI32ArrayAttr({0, 2, 1}))
              .getResult();
  }

  Type output_ety;
  if (result_is_qtype) {
    auto lhs_qty_width =
        mlir::cast<mlir::quant::QuantizedType>(lhs_ty.getElementType())
            .getStorageTypeIntegralWidth();
    auto rhs_qty_width =
        mlir::cast<mlir::quant::QuantizedType>(rhs_ty.getElementType())
            .getStorageTypeIntegralWidth();

    if (lhs_qty_width != rhs_qty_width) {
      return rewriter.notifyMatchFailure(
          op, "input tensors should have same qtype storage width");
    }

    if (lhs_qty_width == 8) {
      output_ety = rewriter.getI32Type();
    } else if (lhs_qty_width == 16) {
      output_ety = rewriter.getIntegerType(48);
    } else {
      return rewriter.notifyMatchFailure(
          op, "only support 8-bit or 16-bit quantized type");
    }
  } else {
    output_ety = result_ty.getElementType();
  }

  Value matmul =
      CreateOpAndInfer<tosa::MatMulOp>(
          rewriter, op->getLoc(), UnrankedTensorType::get(output_ety), lhs, rhs)
          .getResult();

  // Conditionally reshape rank back to expected rank.
  auto matmul_ty = mlir::cast<RankedTensorType>(matmul.getType());
  if (batch_dims.size() != 1) {
    llvm::SmallVector<int64_t> new_shape{};
    for (auto d : batch_dims) {
      new_shape.push_back(d);
    }

    for (auto d : matmul_ty.getShape().take_back(2)) {
      new_shape.push_back(d);
    }

    auto new_shape_value = getTosaConstShape(rewriter, op->getLoc(), new_shape);
    matmul = CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(),
        UnrankedTensorType::get(matmul_ty.getElementType()), matmul,
        new_shape_value);
  }

  if (lhs_is_qtype) {
    matmul = buildRescaleOpConvOutput(rewriter, op, matmul, lhs_ty, rhs_ty,
                                      result_ty);
  }

  rewriter.replaceOp(op, matmul);

  return success();
}

LogicalResult ConvertTFLFullyConnectedOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_fc_op = cast<TFL::FullyConnectedOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_fc_op.getResult(0).getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_fc_op.getInput().getType());
  RankedTensorType filter_type =
      dyn_cast<RankedTensorType>(tfl_fc_op.getFilter().getType());
  RankedTensorType bias_type =
      dyn_cast<RankedTensorType>(tfl_fc_op.getBias().getType());
  if (!input_type || !filter_type) return failure();

  bool input_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(input_type.getElementType());
  bool filter_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(filter_type.getElementType());
  bool output_is_qtype =
      mlir::isa<mlir::quant::QuantizedType>(output_type.getElementType());

  if ((input_is_qtype != filter_is_qtype) ||
      (input_is_qtype != output_is_qtype)) {
    return rewriter.notifyMatchFailure(
        op,
        "input/filter/output tensor should "
        "be all quantized or all floating-point");
  }

  Value input_val = tfl_fc_op.getInput();
  Value filter_val = tfl_fc_op.getFilter();

  // tfl.fully_connected() can takes various dimension tensor as input
  // need to reshape it to rank 2 tensor, which tosa.fully_connected only
  // supports if input tensor is rank 4.  It's not always reshaping to (dim[0] *
  // dim[1], dim[2] * dim[3]).

  // In some networks it's reshaping to (dim[0], dim[1] * dim[2] * dim[3]) so a
  // more general way to determine the reshape's shape is by looking at filter's
  // shape[1].
  if (input_type.getRank() != 2) {
    int64_t num_elems = filter_type.getShape()[1];
    int64_t num_batch = input_type.getNumElements() / num_elems;
    SmallVector<int64_t, 2> shape_vals({num_batch, num_elems});

    RankedTensorType reshape_type =
        RankedTensorType::get(shape_vals, input_type.getElementType());
    auto shape_vals_value =
        getTosaConstShape(rewriter, op->getLoc(), shape_vals);
    auto reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(), reshape_type, input_val, shape_vals_value);

    input_val = reshape_op.getResult();
    input_type = reshape_type;
  }

  // reshape input_val from [N, IC] to [N, 1, 1, IC] to use conv2d
  // reshape filter_val from [OC, IC] to [OC, 1, 1, IC] to use conv2d
  const int64_t N = input_type.getShape()[0];
  const int64_t IC = input_type.getShape()[1];
  const int64_t OC = filter_type.getShape()[0];

  SmallVector<int64_t> new_input_shape({N, 1, 1, IC});
  SmallVector<int64_t> new_filter_shape({OC, 1, 1, IC});

  auto new_input_shape_value =
      getTosaConstShape(rewriter, op->getLoc(),
                        tensorflow::ConvertTFShapeToMlir(new_input_shape));
  input_val = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(new_input_shape,
                                           input_type.getElementType()),
      input_val, new_input_shape_value);
  input_type = cast<RankedTensorType>(input_val.getType());

  auto new_filter_shape_value =
      getTosaConstShape(rewriter, op->getLoc(),
                        tensorflow::ConvertTFShapeToMlir(new_filter_shape));
  filter_val = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(new_filter_shape,
                                           filter_type.getElementType()),
      filter_val, new_filter_shape_value);
  filter_type = cast<RankedTensorType>(filter_val.getType());

  Value bias_val;
  if (!bias_type) {
    // For some matmuls, the bias may actually be a "UnitType" which has no
    // value. TOSA requires bias to be an array of output_channel_count values,
    // so create a constant of the appropriate number and type of zeros.
    SmallVector<int64_t, 1> bias_shape({filter_type.getShape()[0]});
    RankedTensorType new_bias_type;

    DenseElementsAttr bias_attr;
    if (mlir::isa<FloatType>(input_type.getElementType())) {
      SmallVector<float> bias_arr(bias_shape[0]);

      for (int i = 0; i < bias_shape[0]; i++) {
        bias_arr[i] = 0.0;
      }
      new_bias_type =
          RankedTensorType::get(bias_shape, input_type.getElementType());
      bias_attr =
          DenseElementsAttr::get(new_bias_type, llvm::ArrayRef(bias_arr));
    } else {
      SmallVector<int32_t> bias_arr(bias_shape[0]);

      for (int i = 0; i < bias_shape[0]; i++) {
        bias_arr[i] = 0;
      }
      if (!input_is_qtype) {
        return rewriter.notifyMatchFailure(
            op, "input must be quantized type if it's not float type");
      }
      auto input_qtype =
          mlir::cast<mlir::quant::QuantizedType>(input_type.getElementType());
      Type new_bias_ety = input_qtype.getStorageTypeIntegralWidth() == 16
                              ? rewriter.getIntegerType(48)
                              : rewriter.getI32Type();
      new_bias_type = RankedTensorType::get(bias_shape, new_bias_ety);
      bias_attr =
          DenseElementsAttr::get(new_bias_type, llvm::ArrayRef(bias_arr));
    }
    auto bias_op = CreateOpAndInfer<tosa::ConstOp>(rewriter, op->getLoc(),
                                                   new_bias_type, bias_attr);
    bias_val = bias_op.getResult();
    bias_type = new_bias_type;
  } else {
    bias_val = tfl_fc_op.getBias();
  }

  Type bias_ety = mlir::cast<ShapedType>(bias_val.getType()).getElementType();

  auto acc_type =
      getConvAccTypeAttr(rewriter,
                         /* input_etype = */ input_type.getElementType(),
                         /* output_etype = */ bias_ety);

  auto fc_op = CreateOpAndInfer<tosa::Conv2DOp>(
      rewriter, op->getLoc(), UnrankedTensorType::get(bias_ety), input_val,
      filter_val, bias_val,
      /* pad = */ rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
      /* stride = */ rewriter.getDenseI64ArrayAttr({1, 1}),
      /* dilation = */ rewriter.getDenseI64ArrayAttr({1, 1}), acc_type);

  Value fc_output;
  if (input_is_qtype) {
    fc_output = buildRescaleOpConvOutput(
        rewriter, op, fc_op.getResult(), input_type, filter_type,
        UnrankedTensorType::get(output_type.getElementType()));
  } else {
    fc_output = fc_op.getResult();
  }

  // If we know the output rank, we need to ensure the output shape is correct.
  ShapedType fc_type = mlir::cast<ShapedType>(fc_output.getType());

  DenseI64ArrayAttr output_shape_attr;
  if (output_type.hasRank()) {
    output_shape_attr = rewriter.getDenseI64ArrayAttr(output_type.getShape());
  } else {
    // set output_shape to {N, OC} to match previous results
    // with tosa::FullyConnectedOp
    output_shape_attr = rewriter.getDenseI64ArrayAttr({N, OC});
  }

  auto output_shape_value =
      (output_type.hasRank())
          ? getTosaConstShape(rewriter, op->getLoc(), output_type.getShape())
          : getTosaConstShape(rewriter, op->getLoc(), {N, OC});
  fc_output = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(), UnrankedTensorType::get(fc_type.getElementType()),
      fc_output, output_shape_value);

  auto fused_activation_fn = tfl_fc_op.getFusedActivationFunctionAttr();

  if (fused_activation_fn) {
    std::optional<Value> fused_activation_val =
        convertFusedActivation(rewriter, op, fc_output, fused_activation_fn);

    if (!fused_activation_val) return failure();

    rewriter.replaceOp(op, {fused_activation_val.value()});
    return success();
  }

  rewriter.replaceOp(op, {fc_output});

  return success();
}

LogicalResult ConvertTFLConcatenationOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_concat_op = cast<TFL::ConcatenationOp>(op);
  auto result_type = dyn_cast<ShapedType>(tfl_concat_op.getResult().getType());

  SmallVector<Value> values(tfl_concat_op.getValues());

  IntegerAttr axis_attr;
  {
    auto tmpAttr = tfl_concat_op.getAxisAttr();
    if (!tmpAttr) {
      tmpAttr = rewriter.getI32IntegerAttr(0);
    }
    axis_attr = tmpAttr;
  }
  int32_t axis = axis_attr.getInt();

  std::optional<Value> result =
      convertConcatV2Op(rewriter, op, result_type, values, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});
  return success();
}

LogicalResult ConvertTFLReshapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_reshape_op = cast<TFL::ReshapeOp>(op);

  Value shape = tfl_reshape_op.getShape();
  ShapedType shape_type = dyn_cast<ShapedType>(shape.getType());
  ShapedType output_type = dyn_cast<ShapedType>(tfl_reshape_op.getType());

  int64_t rank = ShapedType::kDynamic;
  if (output_type.hasRank()) rank = output_type.getRank();

  // Check the inferred rank from the shape tensor matches the output.
  if (shape_type.hasRank() && !shape_type.isDynamicDim(0)) {
    int64_t dim = shape_type.getDimSize(0);
    if (rank != ShapedType::kDynamic && rank != dim) {
      return rewriter.notifyMatchFailure(op,
                                         "static dim mismatch on tfl.reshape");
    }
    rank = dim;
  }

  if (rank == ShapedType::kDynamic) {
    return rewriter.notifyMatchFailure(op, "unknown rank for output shape");
  }

  // Extract the dynamically shaped values for each dimension.
  SmallVector<Value> shape_vals;
  shape_vals.reserve(rank);
  auto shape_ty = dyn_cast<ShapedType>(shape.getType());
  for (int i = 0; i < rank; i++) {
    auto e_ty = shape_ty.getElementType();
    Value dim = rewriter.createOrFold<tosa::SliceOp>(
        op->getLoc(), RankedTensorType::get({1}, e_ty), shape,
        getTosaConstShape(rewriter, op->getLoc(), {i}),
        getTosaConstShape(rewriter, op->getLoc(), {1}));

    auto shape_value = getTosaConstShape(rewriter, op->getLoc(), {});
    dim = rewriter.createOrFold<tosa::ReshapeOp>(
        op->getLoc(), RankedTensorType::get({}, e_ty), dim, shape_value);
    shape_vals.push_back(dim);
  }

  // Build the reshape operation with dynamic shapes.
  auto reshape =
      buildReshapeWithDynamicDims(rewriter, op, tfl_reshape_op.getInput(),
                                  tfl_reshape_op.getType(), shape_vals);

  if (!reshape.has_value()) return failure();

  rewriter.replaceOp(op, {reshape.value()});
  return success();
}

LogicalResult ConvertTFLRankOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_rank_op = cast<TFL::RankOp>(op);

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_rank_op.getInput().getType());
  if (!input_type) return failure();

  int32_t rank = input_type.getRank();

  RankedTensorType rank_type =
      RankedTensorType::get({1}, rewriter.getIntegerType(32));
  auto rank_attr = DenseI32ArrayAttr::get(rewriter.getContext(), {rank});
  auto rank_const = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), rank_type, mlir::cast<ElementsAttr>(rank_attr));

  rewriter.replaceOp(op, {rank_const.getResult()});

  return success();
}

LogicalResult ConvertTFLShapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_shape_op = cast<TFL::ShapeOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_shape_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_shape_op.getInput().getType());
  if (!input_type || !input_type.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "input shape not static");

  auto input_shape = input_type.getShape();

  SmallVector<int32_t> shape_arr;
  for (int i = 0; i < input_shape.size(); i++) {
    shape_arr.emplace_back(input_shape[i]);
  }

  RankedTensorType shape_type = RankedTensorType::get(
      {static_cast<int32_t>(shape_arr.size())}, rewriter.getIntegerType(32));
  auto shape_attr =
      DenseI32ArrayAttr::get(rewriter.getContext(), llvm::ArrayRef(shape_arr));
  auto shape_const = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), shape_type, mlir::cast<ElementsAttr>(shape_attr));

  rewriter.replaceOp(op, {shape_const.getResult()});

  return success();
}

LogicalResult ConvertTFLExpandDimsOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_expanddims_op = cast<TFL::ExpandDimsOp>(op);

  std::optional<Value> result = convertExpandDimsOp(
      rewriter, op, tfl_expanddims_op.getResult(), tfl_expanddims_op.getInput(),
      tfl_expanddims_op.getDim());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLSqueezeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_squeeze_op = cast<TFL::SqueezeOp>(op);

  // Copy squeeze_dims into int32_t array
  auto squeeze_dims_attr = tfl_squeeze_op.getSqueezeDimsAttr();
  SmallVector<int32_t> squeeze_dims;
  for (auto& squeeze_dim : squeeze_dims_attr) {
    squeeze_dims.emplace_back(dyn_cast<IntegerAttr>(squeeze_dim).getInt());
  }

  std::optional<Value> result =
      convertSqueezeOp(rewriter, op, tfl_squeeze_op.getResult(),
                       tfl_squeeze_op.getInput(), squeeze_dims);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLFillOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_fill_op = cast<TFL::FillOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_fill_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  ElementsAttr dims_elems;
  if (!matchPattern(tfl_fill_op.getDims(), m_Constant(&dims_elems)))
    return failure();
  SmallVector<int64_t> dims_vals;
  uint32_t total_size = 1;
  for (int i = 0; i < dims_elems.getNumElements(); i++) {
    dims_vals.push_back(dims_elems.getValues<APInt>()[i].getSExtValue());
    total_size *= dims_vals[i];
  }

  ElementsAttr value_elem;
  if (!matchPattern(tfl_fill_op.getInput(), m_Constant(&value_elem)))
    return failure();

  RankedTensorType fill_type =
      RankedTensorType::get(ArrayRef<int64_t>(dims_vals),
                            value_elem.getShapedType().getElementType());
  DenseArrayAttr fill_attr;

  // Convert to a compatible zero type.
  if (mlir::isa<FloatType>(value_elem.getShapedType().getElementType())) {
    SmallVector<float> fill_arr(
        total_size, value_elem.getValues<APFloat>()[0].convertToFloat());
    fill_attr =
        DenseF32ArrayAttr::get(rewriter.getContext(), llvm::ArrayRef(fill_arr));
  } else {
    SmallVector<int32_t> fill_arr(
        total_size, value_elem.getValues<APInt>()[0].getLimitedValue());
    fill_attr =
        DenseI32ArrayAttr::get(rewriter.getContext(), llvm::ArrayRef(fill_arr));
  }
  auto fill_const_op = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, op->getLoc(), fill_type, mlir::cast<ElementsAttr>(fill_attr));
  rewriter.replaceOp(op, {fill_const_op.getResult()});

  return success();
}

LogicalResult ConvertTFLReduceAllOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_all_op = cast<TFL::ReduceAllOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_all_op.getResult().getType());

  if (!output_type)
    return rewriter.notifyMatchFailure(op, "output not a ranked tensor");

  ElementsAttr axes_elems;
  if (!matchPattern(tfl_all_op.getReductionIndices(), m_Constant(&axes_elems)))
    return rewriter.notifyMatchFailure(op, "fail to get reduction indices");

  std::optional<Value> result = convertReduceAllOp(
      rewriter, op, output_type, tfl_all_op.getInput(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLReduceAnyOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_any_op = cast<TFL::ReduceAnyOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_any_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tfl_any_op.getReductionIndices(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceAnyOp(
      rewriter, op, output_type, tfl_any_op.getInput(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLReduceMaxOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_max_op = cast<TFL::ReduceMaxOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_max_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tfl_max_op.getAxes(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceMaxOp(
      rewriter, op, output_type, tfl_max_op.getInput(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLReduceMinOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_min_op = cast<TFL::ReduceMinOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_min_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tfl_min_op.getAxes(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceMinOp(
      rewriter, op, output_type, tfl_min_op.getInput(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLReduceProdOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_prod_op = cast<TFL::ReduceProdOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_prod_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tfl_prod_op.getAxes(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceProdOp(
      rewriter, op, output_type, tfl_prod_op.getInput(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLMeanOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_mean_op = cast<TFL::MeanOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_mean_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tfl_mean_op.getAxis(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceMeanOp(
      rewriter, op, output_type, tfl_mean_op.getInput(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLSumOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_sum_op = cast<TFL::SumOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_sum_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr axes_elems;
  if (!matchPattern(tfl_sum_op.getAxes(), m_Constant(&axes_elems)))
    return failure();

  std::optional<Value> result = convertReduceSumOp(
      rewriter, op, output_type, tfl_sum_op.getInput(), axes_elems);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLEluOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_elu_op = cast<TFL::EluOp>(op);

  std::optional<Value> result =
      convertEluOp(rewriter, op, tfl_elu_op.getResult(), tfl_elu_op.getX());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLSoftmaxOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_softmax_op = cast<TFL::SoftmaxOp>(op);

  std::optional<Value> result = convertSoftmaxOp(
      rewriter, op, tfl_softmax_op.getResult(), tfl_softmax_op.getInput(),
      tfl_softmax_op.getBetaAttr().getValueAsDouble());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLRsqrtOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_rsqrt_op = cast<TFL::RsqrtOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_rsqrt_op.getResult().getType());
  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_rsqrt_op.getX().getType());

  mlir::quant::UniformQuantizedType input_qtype =
      mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(
          input_type.getElementType());
  mlir::quant::UniformQuantizedType output_qtype =
      mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(
          output_type.getElementType());

  // Quantization case
  if (input_qtype && output_qtype) {
    // 16-bit is pending review for TFL
    // https://github.com/tensorflow/tensorflow/pull/58406
    if (input_qtype.getStorageTypeIntegralWidth() != 8) {
      return rewriter.notifyMatchFailure(op,
                                         "input qtype storage width is not 8");
    }

    // Implement with 8-bit table lookup.
    Value table_const = getTosaConstRsqrt8bitTable(
        rewriter, op, input_qtype.getScale(), input_qtype.getZeroPoint(),
        output_qtype.getScale(), output_qtype.getZeroPoint());

    CreateReplaceOpAndInfer<tosa::TableOp>(rewriter, op, output_type,
                                           tfl_rsqrt_op.getX(), table_const);
    return success();
  }

  CreateReplaceOpAndInfer<tosa::RsqrtOp>(rewriter, op, tfl_rsqrt_op.getType(),
                                         tfl_rsqrt_op.getX());

  return success();
}

LogicalResult ConvertTFLSqrtOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_rsqrt_op = cast<TFL::SqrtOp>(op);
  auto rsqrt = CreateOpAndInfer<tosa::RsqrtOp>(
      rewriter, op->getLoc(), tfl_rsqrt_op.getType(), tfl_rsqrt_op.getX());

  CreateReplaceOpAndInfer<tosa::ReciprocalOp>(rewriter, op, rsqrt.getType(),
                                              rsqrt);

  return success();
}

LogicalResult ConvertTFLL2NormalizationOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_l2norm_op = cast<TFL::L2NormalizationOp>(op);
  auto input = tfl_l2norm_op.getInput();
  auto input_ty = mlir::cast<ShapedType>(input.getType());
  auto loc = op->getLoc();

  if (!input_ty.hasRank()) return failure();

  auto rank = input_ty.getRank();

  if (input_ty.getElementType().isF32()) {
    auto result_ty = UnrankedTensorType::get(input_ty.getElementType());
    auto mul = CreateMulOpAndInfer(rewriter, op, result_ty, input, input);
    auto sum = CreateOpAndInfer<tosa::ReduceSumOp>(
        rewriter, loc, result_ty, mul,
        rewriter.getI32IntegerAttr(input_ty.getRank() - 1));

    SmallVector<float> min(1, sqrt(std::numeric_limits<float>::min()));
    Value min_val = getTosaConstTensorSingleF32(
        rewriter, op, sqrt(std::numeric_limits<float>::min()), rank);
    auto max = CreateOpAndInfer<tosa::MaximumOp>(rewriter, loc, result_ty, sum,
                                                 min_val);
    auto rsqrt = CreateOpAndInfer<tosa::RsqrtOp>(rewriter, loc, result_ty, max)
                     .getResult();
    Value result =
        CreateMulOpAndInfer(rewriter, op, result_ty, rsqrt, input).getResult();

    auto fused_activation_fn = tfl_l2norm_op.getFusedActivationFunctionAttr();

    if (fused_activation_fn) {
      std::optional<Value> fused_activation_val =
          convertFusedActivation(rewriter, op, result, fused_activation_fn);
      if (!fused_activation_val) return failure();
      result = fused_activation_val.value();
    }

    rewriter.replaceOp(op, result);
    return success();
  }

  return failure();
}

LogicalResult ConvertTFLLogSoftmaxOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_logsoftmax_op = cast<TFL::LogSoftmaxOp>(op);

  std::optional<Value> result =
      convertLogSoftmaxOp(rewriter, op, tfl_logsoftmax_op.getResult(),
                          tfl_logsoftmax_op.getInput());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLSliceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_slice_op = cast<TFL::SliceOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_slice_op.getResult().getType());
  // Not a shaped tensor output
  if (!output_type) return failure();

  ElementsAttr begin_elems, size_elems;

  SmallVector<int64_t> begin_vals, size_vals;

  if (!matchPattern(tfl_slice_op.getBegin(), m_Constant(&begin_elems)) ||
      !matchPattern(tfl_slice_op.getSize(), m_Constant(&size_elems))) {
    return failure();
  }

  auto input = tfl_slice_op.getInput();
  RankedTensorType input_type = cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> shape = input_type.getShape();
  size_t rank = shape.size();
  assert(begin_elems.getNumElements() == rank);
  assert(size_elems.getNumElements() == rank);

  const int64_t to_end = -1;
  for (int i = 0; i < rank; i++) {
    int64_t begin = begin_elems.getValues<APInt>()[i].getSExtValue();
    int64_t size = size_elems.getValues<APInt>()[i].getSExtValue();
    size = (size == to_end) ? shape[i] - begin : size;
    begin_vals.push_back(begin);
    size_vals.push_back(size);
  }

  CreateReplaceOpAndInfer<tosa::SliceOp>(
      rewriter, op, output_type, input,
      getTosaConstShape(rewriter, op->getLoc(), begin_vals),
      getTosaConstShape(rewriter, op->getLoc(), size_vals));

  return success();
}

LogicalResult ConvertTFLTileOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_tile_op = cast<TFL::TileOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_tile_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  ElementsAttr multiples_elems;
  if (!matchPattern(tfl_tile_op.getMultiples(), m_Constant(&multiples_elems)))
    return failure();
  SmallVector<int64_t> multiples_vals;
  for (int i = 0; i < multiples_elems.getNumElements(); i++)
    multiples_vals.push_back(
        multiples_elems.getValues<APInt>()[i].getSExtValue());

  auto multiples = getTosaConstShape(rewriter, op, multiples_vals);

  CreateReplaceOpAndInfer<tosa::TileOp>(rewriter, op, output_type,
                                        tfl_tile_op.getInput(), multiples);

  return success();
}

LogicalResult ConvertTFLTransposeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_transpose_op = cast<TFL::TransposeOp>(op);

  // Get the constant values for perm input
  ElementsAttr permsAttrElems;
  if (!matchPattern(tfl_transpose_op.getPerm(), m_Constant(&permsAttrElems))) {
    return rewriter.notifyMatchFailure(op, "perm value is not constant");
  }

  SmallVector<int32_t> perms;
  for (auto v : permsAttrElems.getValues<APInt>()) {
    perms.push_back(v.getSExtValue());
  }

  Type output_type = tfl_transpose_op.getResult().getType();
  CreateReplaceOpAndInfer<tosa::TransposeOp>(
      rewriter, op, output_type, tfl_transpose_op.getInput(),
      rewriter.getDenseI32ArrayAttr(perms));

  return success();
}

LogicalResult ConvertTFLPackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_pack_op = cast<TFL::PackOp>(op);

  SmallVector<Value> inputs(tfl_pack_op.getValues());
  assert(!inputs.empty());

  IntegerAttr axis_attr;
  {
    auto tmpAttr = tfl_pack_op.getAxisAttr();
    if (!tmpAttr) tmpAttr = rewriter.getI32IntegerAttr(0);
    axis_attr = tmpAttr;
  }
  int32_t axis_i32 = axis_attr.getInt();

  std::optional<Value> result =
      convertPackOp(rewriter, op, tfl_pack_op.getResult(), inputs, axis_i32);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLUnpackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_unpack_op = cast<TFL::UnpackOp>(op);

  IntegerAttr axis_attr;
  {
    auto tmpAttr = tfl_unpack_op.getAxisAttr();
    if (!tmpAttr) tmpAttr = rewriter.getI32IntegerAttr(0);
    axis_attr = tmpAttr;
  }
  int32_t axis_i32 = axis_attr.getInt();

  std::optional<SmallVector<Value>> results =
      convertUnpackOp(rewriter, op, tfl_unpack_op.getInput(), axis_i32);

  if (!results) return failure();

  rewriter.replaceOp(op, results.value());

  return success();
}

// Splits in num_split parts along split_dim
LogicalResult ConvertTFLSplitOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_split_op = cast<TFL::SplitOp>(op);

  // Get the number of splits
  int32_t num_split = -1;
  auto numSplitAttr = tfl_split_op.getNumSplitsAttr();
  if (numSplitAttr) {
    num_split = numSplitAttr.getInt();
  } else {
    return failure();
  }

  // Get the axis
  ElementsAttr axisAttrElems;
  if (!matchPattern(tfl_split_op.getSplitDim(), m_Constant(&axisAttrElems))) {
    return rewriter.notifyMatchFailure(op, "cannot read split_dim elems");
  }

  // The axis/split_dim parameter is stored as a 0D tensor instead of
  // an integer attribute in TFLite MLIR.
  int32_t axis = axisAttrElems.getValues<APInt>()[0].getSExtValue();

  std::optional<SmallVector<Value>> results =
      convertSplitOp(rewriter, op, tfl_split_op.getResult(0),
                     tfl_split_op.getValue(), num_split, axis);

  if (!results) return failure();

  rewriter.replaceOp(op, results.value());

  return success();
}

// Splits in num_split parts along split_dim
LogicalResult ConvertTFLSplitVOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_splitv_op = cast<TFL::SplitVOp>(op);

  // Get the size_splits array
  SmallVector<int32_t> size_split;
  ElementsAttr size_split_elems;
  if (!matchPattern(tfl_splitv_op.getSizeSplits(),
                    m_Constant(&size_split_elems))) {
    return failure();
  }

  for (int i = 0; i < size_split_elems.getNumElements(); i++) {
    size_split.push_back(size_split_elems.getValues<APInt>()[i].getSExtValue());
  }

  // Get the axis
  ElementsAttr axisAttrElems;
  if (!matchPattern(tfl_splitv_op.getSplitDim(), m_Constant(&axisAttrElems))) {
    return rewriter.notifyMatchFailure(op, "cannot read split_dim elems");
  }

  // The axis/split_dim parameter is stored as a 0D tensor instead of
  // an integer attribute in TFLite MLIR.
  int32_t axis = axisAttrElems.getValues<APInt>()[0].getSExtValue();

  std::optional<SmallVector<Value>> results =
      convertSplitVOp(rewriter, op, tfl_splitv_op.getResult(0),
                      tfl_splitv_op.getValue(), size_split, axis);

  if (!results) return failure();

  rewriter.replaceOp(op, results.value());

  return success();
}

LogicalResult ConvertTFLPadOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_pad_op = cast<TFL::PadOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_pad_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  SmallVector<int64_t> padding_vals;
  if (failed(getVectorFromValue64(tfl_pad_op.getPadding(), padding_vals))) {
    return rewriter.notifyMatchFailure(op, "padding is not a constant value");
  }
  Value padding = getTosaConstShape(rewriter, op->getLoc(), padding_vals);

  auto pad_op = CreateOpAndInfer<tosa::PadOp>(
      rewriter, op->getLoc(), output_type, tfl_pad_op.getInput(), padding);

  rewriter.replaceOp(op, {pad_op.getResult()});
  return success();
}

LogicalResult ConvertTFLMirrorPadOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_mirrorpad_op = cast<TFL::MirrorPadOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_mirrorpad_op.getResult().getType());
  if (!output_type) {
    return rewriter.notifyMatchFailure(op, "output type isn't a ranked tensor");
  }

  TFTFLMirrorPaddingType mode;
  switch (tfl_mirrorpad_op.getMode()) {
    case mlir::TFL::MirrorPaddingType::REFLECT:
      mode = TFTFLMirrorPaddingType::REFLECT;
      break;
    case mlir::TFL::MirrorPaddingType::SYMMETRIC:
      mode = TFTFLMirrorPaddingType::SYMMETRIC;
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "mode isn't one of REFLECT or SYMMETRIC");
  }

  std::optional<Value> result = convertMirrorPadCommon(
      rewriter, op, output_type, tfl_mirrorpad_op.getInput(),
      tfl_mirrorpad_op.getPad(), mode);

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLPadV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_pad_op = cast<TFL::PadV2Op>(op);

  Value input = tfl_pad_op.getInput();
  Value constant_value = tfl_pad_op.getConstantValues();

  Value rank1_scalar_value =
      reshapeScalarTo1D(rewriter, op->getLoc(), constant_value);
  if (!rank1_scalar_value) {
    return rewriter.notifyMatchFailure(
        op, "cannot reshape constant_value input to rank 1");
  }

  SmallVector<int64_t> padding_vals;
  if (failed(getVectorFromValue64(tfl_pad_op.getPadding(), padding_vals))) {
    return rewriter.notifyMatchFailure(op, "padding is not a constant value");
  }

  Value padding = getTosaConstShape(rewriter, op->getLoc(), padding_vals);

  CreateReplaceOpAndInfer<tosa::PadOp>(rewriter, op, tfl_pad_op.getType(),
                                       input, padding, rank1_scalar_value);

  return success();
}

LogicalResult ConvertTFLResizeBilinearOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_resize_op = cast<TFL::ResizeBilinearOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_resize_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  std::optional<Value> result = convertResizeOp(
      rewriter, op, output_type, tfl_resize_op.getInput(),
      StringRef("BILINEAR"), tfl_resize_op.getAlignCornersAttr().getValue(),
      tfl_resize_op.getHalfPixelCentersAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLResizeNearestNeighborOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_resize_op = cast<TFL::ResizeNearestNeighborOp>(op);

  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_resize_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  std::optional<Value> result =
      convertResizeOp(rewriter, op, output_type, tfl_resize_op.getInput(),
                      StringRef("NEAREST_NEIGHBOR"),
                      tfl_resize_op.getAlignCornersAttr().getValue(),
                      tfl_resize_op.getHalfPixelCentersAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLSelectOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_sel_op = cast<TFL::SelectOp>(op);

  std::optional<Value> result = convertSelectOp(
      rewriter, op, tfl_sel_op.getResult(), tfl_sel_op.getCondition(),
      tfl_sel_op.getX(), tfl_sel_op.getY());
  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLSelectV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_sel_op = cast<TFL::SelectV2Op>(op);

  std::optional<Value> result = convertSelectOp(
      rewriter, op, tfl_sel_op.getResult(), tfl_sel_op.getCondition(),
      tfl_sel_op.getX(), tfl_sel_op.getY());
  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLSpaceToBatchNdOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_s2b_op = cast<TFL::SpaceToBatchNdOp>(op);
  std::optional<Value> result = convertSpaceToBatchNDOp(
      rewriter, op, tfl_s2b_op.getResult(), tfl_s2b_op.getInput(),
      tfl_s2b_op.getBlockShape(), tfl_s2b_op.getPaddings());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLBatchToSpaceNdOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_b2s_op = cast<TFL::BatchToSpaceNdOp>(op);

  std::optional<Value> result = convertBatchToSpaceNDOp(
      rewriter, op, tfl_b2s_op.getResult(), tfl_b2s_op.getInput(),
      tfl_b2s_op.getBlockShape(), tfl_b2s_op.getIndices());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLSpaceToDepthOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_s2d_op = cast<TFL::SpaceToDepthOp>(op);

  auto block_size_attr = tfl_s2d_op.getBlockSizeAttr();
  std::optional<Value> result = convertSpaceToDepthOp(
      rewriter, op, tfl_s2d_op.getResult(), tfl_s2d_op.getInput(),
      block_size_attr, rewriter.getStringAttr("NHWC"));

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLDepthToSpaceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_d2s_op = cast<TFL::DepthToSpaceOp>(op);

  auto block_size_attr = tfl_d2s_op.getBlockSizeAttr();
  std::optional<Value> result = convertDepthToSpaceOp(
      rewriter, op, tfl_d2s_op.getResult(), tfl_d2s_op.getInput(),
      block_size_attr, rewriter.getStringAttr("NHWC"));

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLBucketizeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_bucketize_op = cast<TFL::BucketizeOp>(op);
  Location loc = op->getLoc();

  Value input = tfl_bucketize_op.getInput();
  auto boundaries_attr = tfl_bucketize_op.getBoundaries();
  RankedTensorType input_type = dyn_cast<RankedTensorType>(input.getType());
  if (!input_type) {
    return rewriter.notifyMatchFailure(op, "input is not a ranked tensor");
  }

  // The lowering is done by broadcasting the input and boundaries together, and
  // using GE comparison for each input against each boundary. Adding the
  // results of the comparison for each input generates the bucket it belongs
  // to, as the boundaries are sorted.
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_bucketize_op.getResult().getType());

  auto input_shape = input_type.getShape();

  SmallVector<APFloat> boundaries;
  for (auto& boundary : boundaries_attr) {
    boundaries.emplace_back(dyn_cast<FloatAttr>(boundary).getValue());
  }
  int64_t boundaries_size = boundaries.size();

  // Add a dim at the end of input shape for broadcasting with the boundaries.
  SmallVector<int64_t> broadcast_shape(input_shape.begin(), input_shape.end());
  broadcast_shape.push_back(boundaries_size);
  SmallVector<int64_t> new_input_shape(input_shape.begin(), input_shape.end());
  new_input_shape.push_back(1);

  auto boundaries_type =
      RankedTensorType::get({boundaries_size}, rewriter.getF32Type());

  auto boundaries_op = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, loc, boundaries_type,
      DenseElementsAttr::get(boundaries_type, boundaries));

  auto boundaries_input_type =
      boundaries_type.clone(input_type.getElementType());
  Value boundaries_op_casted = CreateOpAndInfer<tosa::CastOp>(
      rewriter, loc, boundaries_input_type, boundaries_op);

  auto new_input_shape_value = getTosaConstShape(rewriter, op->getLoc(), new_input_shape);
  Value reshaped_input = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, loc, input_type.clone(new_input_shape), input,
      new_input_shape_value);

  auto ge = CreateOpAndInfer<tosa::GreaterEqualOp>(
      rewriter, loc, UnrankedTensorType::get(rewriter.getIntegerType(1)),
      reshaped_input, boundaries_op_casted);

  auto casted = CreateOpAndInfer<tosa::CastOp>(
      rewriter, loc, UnrankedTensorType::get(rewriter.getIntegerType(32)), ge);

  auto sum = CreateOpAndInfer<tosa::ReduceSumOp>(
      rewriter, loc, output_type, casted,
      rewriter.getI32IntegerAttr(input_type.getRank()));

  auto output_shape_value =
      getTosaConstShape(rewriter, op->getLoc(), output_type.getShape());
  CreateReplaceOpAndInfer<tosa::ReshapeOp>(rewriter, op, output_type, sum,
                                           output_shape_value);

  return success();
}

LogicalResult ConvertTFLStridedSliceOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_ss_op = cast<TFL::StridedSliceOp>(op);

  std::optional<Value> result = convertStridedSliceOp(
      rewriter, op, tfl_ss_op.getResult(), tfl_ss_op.getInput(),
      tfl_ss_op.getBegin(), tfl_ss_op.getEnd(), tfl_ss_op.getStrides(),
      tfl_ss_op.getBeginMaskAttr().getInt(),
      tfl_ss_op.getEndMaskAttr().getInt(),
      tfl_ss_op.getEllipsisMaskAttr().getInt(),
      tfl_ss_op.getNewAxisMaskAttr().getInt(),
      tfl_ss_op.getShrinkAxisMaskAttr().getInt());
  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLZerosLikeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_zeroslike_op = cast<TFL::ZerosLikeOp>(op);

  std::optional<Value> result = convertZerosLikeOp(
      rewriter, op, tfl_zeroslike_op.getResult(), tfl_zeroslike_op.getInput());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLHardSwishOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_hardswish_op = cast<TFL::HardSwishOp>(op);
  RankedTensorType output_type =
      dyn_cast<RankedTensorType>(tfl_hardswish_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_hardswish_op.getInput().getType());
  // Not a ranked tensor output
  if (!input_type) return failure();

  // TFL hardswish: f(x) -> (x * relu6(x+3))/6

  if (mlir::isa<mlir::quant::QuantizedType>(input_type.getElementType()) &&
      mlir::isa<mlir::quant::QuantizedType>(output_type.getElementType())) {
    // Should match TFLite reference numerical behavior
    mlir::quant::UniformQuantizedType input_qtype =
        mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(
            input_type.getElementType());
    mlir::quant::UniformQuantizedType output_qtype =
        mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(
            output_type.getElementType());

    auto hardswish_func = [](double v) -> double {
      double w = v + 3.0;
      w = w < 0.0 ? 0.0 : w > 6.0 ? 6.0 : w;
      return v * w / 6.0;
    };

    if (input_qtype.getStorageTypeIntegralWidth() == 8) {
      // Implement with 8-bit table lookup.
      Value table_const = getTosaConst8bitTable(
          rewriter, op, input_qtype.getScale(), input_qtype.getZeroPoint(),
          output_qtype.getScale(), output_qtype.getZeroPoint(), hardswish_func);

      CreateReplaceOpAndInfer<tosa::TableOp>(
          rewriter, op, output_type, tfl_hardswish_op.getInput(), table_const);
    }

  } else {
    // op1 = constop(3)
    // op2 = add(x, op1)
    // op3 = clamp(op2, 0, 6)
    // op4 = mul(x, op3)
    // op5 = reciprocal(6)
    // op6 = mul (op4, op5)

    auto rank = input_type.getRank();

    Value op1_value = getTosaConstTensorSingleF32(rewriter, op, 3.0, rank);

    auto op2_add_x_op1 =
        CreateOpAndInfer<tosa::AddOp>(rewriter, op->getLoc(), output_type,
                                      tfl_hardswish_op.getInput(), op1_value);

    auto op3_relu_op2_6 = CreateOpAndInfer<tosa::ClampOp>(
        rewriter, op->getLoc(), output_type, op2_add_x_op1.getResult(),
        /* min_val = */ rewriter.getF32FloatAttr(0.0f),
        /* max_val = */ rewriter.getF32FloatAttr(6.0f));

    auto op4_mul_x_op3 = CreateMulOpAndInfer(rewriter, op, output_type,
                                             tfl_hardswish_op.getInput(),
                                             op3_relu_op2_6.getResult());

    auto const_6 = getTosaConstTensorSingleF32(rewriter, op, 6.0, rank);
    auto op5_reciprocal_6 = CreateOpAndInfer<tosa::ReciprocalOp>(
        rewriter, op->getLoc(), const_6.getType(), const_6);

    auto op6_mul_op4_op5 = CreateMulOpAndInfer(rewriter, op, output_type,
                                               op4_mul_x_op3.getResult(),
                                               op5_reciprocal_6.getResult());

    rewriter.replaceOp(op, {op6_mul_op4_op5.getResult()});
  }

  return success();
}

LogicalResult ConvertTFLSinOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_sin_op = cast<TFL::SinOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_sin_op.getResult().getType());

  if (!output_type)
    return rewriter.notifyMatchFailure(op, "output_type required");
  CreateReplaceOpAndInfer<tosa::SinOp>(rewriter, op, output_type,
                                       tfl_sin_op.getX());

  return success();
}

LogicalResult ConvertTFLCosOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_cos_op = cast<TFL::CosOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_cos_op.getResult().getType());
  if (!output_type)
    return rewriter.notifyMatchFailure(op, "output_type required");
  CreateReplaceOpAndInfer<tosa::CosOp>(rewriter, op, output_type,
                                       tfl_cos_op.getX());

  return success();
}

LogicalResult ConvertTFLAtan2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_atan2_op = cast<TFL::Atan2Op>(op);
  Location loc = op->getLoc();
  Value input_y = tfl_atan2_op.getY();
  Value input_x = tfl_atan2_op.getX();

  auto input_y_ty = dyn_cast<RankedTensorType>(input_y.getType());
  auto input_x_ty = dyn_cast<RankedTensorType>(input_x.getType());
  auto output_ty = dyn_cast<ShapedType>(tfl_atan2_op.getResult().getType());

  if (!input_y_ty || !input_x_ty || !output_ty) {
    return rewriter.notifyMatchFailure(op, "ranked inputs/output required");
  }

  if (!input_y_ty.getElementType().isF32()) {
    return rewriter.notifyMatchFailure(op, "input must be fp32");
  }

  // To perform an atan2 operation we make use of an atan lookup table,
  // then determine the correct quadrant for each output. To restrict the
  // input domain of the lookup table from [-inf, inf] to [0, 1], we make
  // use of two identities and undo the transformation later on:
  //
  // acrtan(z) = π/2 - arctan(1/z)                                  (0)
  //
  // and
  //
  // arctan(-z) = -arctan(z)                                        (1)

  auto rank = input_x_ty.getRank();
  Value pi = getTosaConstTensorSingleF32(rewriter, op, M_PI, rank);
  Value pi_2 = getTosaConstTensorSingleF32(rewriter, op, M_PI_2, rank);
  Value zero = getTosaConstTensorSingleF32(rewriter, op, 0.0, rank);
  Value one = getTosaConstTensorSingleF32(rewriter, op, 1.0, rank);
  Value two = getTosaConstTensorSingleF32(rewriter, op, 2.0, rank);

  // 1. Restrict the input to the atan lookup from [-inf, inf] to [0, 1].
  // By utilizing (0) and (1) we compute: min(|x|, |y|) / max(|x|, |y|).
  auto abs_y =
      CreateOpAndInfer<tosa::AbsOp>(rewriter, loc, input_y_ty, input_y);
  auto abs_x =
      CreateOpAndInfer<tosa::AbsOp>(rewriter, loc, input_y_ty, input_x);
  auto min_xy = CreateOpAndInfer<tosa::MinimumOp>(rewriter, loc, input_y_ty,
                                                  abs_y, abs_x);
  auto max_xy = CreateOpAndInfer<tosa::MaximumOp>(rewriter, loc, input_y_ty,
                                                  abs_y, abs_x);
  auto recip =
      CreateOpAndInfer<tosa::ReciprocalOp>(rewriter, loc, input_y_ty, max_xy);
  auto atan_input =
      CreateMulOpAndInfer(rewriter, op, input_y_ty, recip, min_xy);

  // 2. Scale and translate the normalized domain to the table domain. This
  // includes a translating and scaling to [-int16_max, int16_max] and casting
  // to an i16 as it is the highest precision the table operation supports.
  auto fp_scalar_ty = RankedTensorType::get({}, rewriter.getF32Type());
  auto scale_up =
      CreateMulOpAndInfer(rewriter, op, input_y_ty, atan_input, two);
  Value translate =
      CreateOpAndInfer<tosa::SubOp>(rewriter, loc, input_y_ty, scale_up, one);
  Value int_limit = rewriter.create<tosa::ConstOp>(
      loc, fp_scalar_ty,
      DenseElementsAttr::get(
          fp_scalar_ty,
          {static_cast<float>(std::numeric_limits<int16_t>::max())}));
  auto int_scaled =
      CreateMulOpAndInfer(rewriter, op, input_y_ty, translate, int_limit);

  auto int16_ty = input_y_ty.clone(rewriter.getIntegerType(16));
  auto casted =
      CreateOpAndInfer<tosa::CastOp>(rewriter, loc, int16_ty, int_scaled);

  // 3. Compute a lookup table using the domain of [0, 1] for atan.
  // Note: the implementation of std::atan2 may be different on
  // different machines, so may result in varying numerical results.
  auto atan_func = [](double x) -> double { return std::atan(x); };
  Value table_const = getTosaConst16bitTable(rewriter, op, atan_func, 0.0, 1.0);
  auto table_result = CreateOpAndInfer<tosa::TableOp>(
      rewriter, loc, output_ty.clone(rewriter.getIntegerType(32)), casted,
      table_const);

  // 4. The range of table is a 23-bit two's complement value. Normalize the
  // range by casting to an fp32 and dividing by 2^22.
  auto table_result_fp =
      CreateOpAndInfer<tosa::CastOp>(rewriter, loc, output_ty, table_result);
  auto output_scale = rewriter.create<ConstOp>(
      loc, fp_scalar_ty,
      DenseElementsAttr::get(
          fp_scalar_ty,
          {static_cast<float>(1.0 / static_cast<float>(1 << 22))}));
  auto table_output = CreateMulOpAndInfer(rewriter, op, output_ty,
                                          table_result_fp, output_scale);

  auto bool_ty = output_ty.clone(rewriter.getIntegerType(1));

  // 5. If (0) was applied to the atan input, apply π/2 - table_output.
  auto sub_pi_2 = CreateOpAndInfer<tosa::SubOp>(rewriter, loc, output_ty, pi_2,
                                                table_output);
  auto condition =
      CreateOpAndInfer<tosa::GreaterOp>(rewriter, loc, bool_ty, abs_y, abs_x);
  auto transform_output = CreateOpAndInfer<tosa::SelectOp>(
      rewriter, loc, output_ty, condition, sub_pi_2, table_output);

  // 6. Determine the correct atan2 quadrant.
  // If x < 0, apply π - transform_output.
  auto sub_pi = CreateOpAndInfer<tosa::SubOp>(rewriter, loc, output_ty, pi,
                                              transform_output);
  auto cond_1 =
      CreateOpAndInfer<tosa::GreaterOp>(rewriter, loc, bool_ty, zero, input_x);
  auto quadrant_select = CreateOpAndInfer<tosa::SelectOp>(
      rewriter, loc, output_ty, cond_1, sub_pi, transform_output);

  // 7. If (1) was applied to the atan input, negate output.
  auto neg_r = CreateOpAndInfer<tosa::NegateOp>(rewriter, loc, output_ty,
                                                quadrant_select);
  auto cond_2 =
      CreateOpAndInfer<tosa::GreaterOp>(rewriter, loc, bool_ty, zero, input_y);
  CreateReplaceOpAndInfer<tosa::SelectOp>(rewriter, op, output_ty, cond_2,
                                          neg_r, quadrant_select);

  return success();
}

LogicalResult ConvertTFLLogisticOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_logistic_op = cast<TFL::LogisticOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_logistic_op.getResult().getType());
  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_logistic_op.getX().getType());
  if (!input_type || !output_type) return failure();

  bool input_is_qtype =
      mlir::isa<mlir::quant::UniformQuantizedType>(input_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  if (input_is_qtype) {
    ShapedType int32_type = output_type.clone(rewriter.getIntegerType(32));
    mlir::quant::UniformQuantizedType input_qtype =
        mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(
            input_type.getElementType());
    mlir::quant::UniformQuantizedType output_qtype =
        mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(
            output_type.getElementType());

    auto sigmoid_func = [](double x) -> double {
      return 1.0 / (1.0 + std::exp(-x));
    };

    if (input_qtype.getStorageTypeIntegralWidth() == 8) {
      Value table_const = getTosaConst8bitTable(
          rewriter, op, input_qtype.getScale(), input_qtype.getZeroPoint(),
          output_qtype.getScale(), output_qtype.getZeroPoint(), sigmoid_func);

      CreateReplaceOpAndInfer<tosa::TableOp>(
          rewriter, op, output_type, tfl_logistic_op.getX(), table_const);
    } else {  // int16
      if (input_qtype.getZeroPoint() != 0 || output_qtype.getZeroPoint() != 0) {
        return rewriter.notifyMatchFailure(
            op, "input/output zeropoint should be 0 in 16-bit mode");
      }
      double input_min = -32768 * input_qtype.getScale();
      double input_max = 32767 * input_qtype.getScale();

      // Generate table with gen_lut() in
      // tensorflow/lite/kernels/internal/common.h
      Value table_const = getTosaConst16bitTable(rewriter, op, sigmoid_func,
                                                 input_min, input_max);

      auto op1_table_in =
          CreateOpAndInfer<tosa::TableOp>(rewriter, op->getLoc(), int32_type,
                                          tfl_logistic_op.getX(), table_const);

      Value op2_rescale_op1 =
          buildRescale(rewriter, op, output_type, op1_table_in.getResult(),
                       1.0 / 128.0, 0, 0, "SINGLE_ROUND", true);

      rewriter.replaceOp(op, {op2_rescale_op1});
    }
  } else {
    CreateReplaceOpAndInfer<tosa::SigmoidOp>(rewriter, op, output_type,
                                             tfl_logistic_op.getX());
  }

  return success();
}

LogicalResult ConvertTFLTanhOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_tanh_op = cast<TFL::TanhOp>(op);
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_tanh_op.getResult().getType());
  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_tanh_op.getInput().getType());
  if (!input_type || !output_type) return failure();

  bool input_is_qtype =
      mlir::isa<mlir::quant::UniformQuantizedType>(input_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
  }

  if (input_is_qtype) {
    ShapedType int32_type = output_type.clone(rewriter.getIntegerType(32));
    mlir::quant::UniformQuantizedType input_qtype =
        mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(
            input_type.getElementType());
    mlir::quant::UniformQuantizedType output_qtype =
        mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(
            output_type.getElementType());

    auto tanh_func = [](double x) -> double {
      x = std::exp(-2.0 * x);
      return (1.0 - x) / (1.0 + x);
    };

    if (input_qtype.getStorageTypeIntegralWidth() == 8) {
      Value table_const = getTosaConst8bitTable(
          rewriter, op, input_qtype.getScale(), input_qtype.getZeroPoint(),
          output_qtype.getScale(), output_qtype.getZeroPoint(), tanh_func);

      CreateReplaceOpAndInfer<tosa::TableOp>(
          rewriter, op, output_type, tfl_tanh_op.getInput(), table_const);
    } else {  // int16
      if (input_qtype.getZeroPoint() != 0 || output_qtype.getZeroPoint() != 0) {
        return rewriter.notifyMatchFailure(
            op, "input/output zeropoint should be 0 in 16-bit mode");
      }
      double input_min = -32768 * input_qtype.getScale();
      double input_max = 32767 * input_qtype.getScale();

      // Generate table with gen_lut() in
      // tensorflow/lite/kernels/internal/common.h
      Value table_const =
          getTosaConst16bitTable(rewriter, op, tanh_func, input_min, input_max);

      auto op1_table_in =
          CreateOpAndInfer<tosa::TableOp>(rewriter, op->getLoc(), int32_type,
                                          tfl_tanh_op.getInput(), table_const);

      Value op2_rescale_op1 =
          buildRescale(rewriter, op, output_type, op1_table_in.getResult(),
                       1.0 / 128.0, 0, 0, "SINGLE_ROUND", true);

      rewriter.replaceOp(op, {op2_rescale_op1});
    }

  } else {
    CreateReplaceOpAndInfer<tosa::TanhOp>(rewriter, op, output_type,
                                          tfl_tanh_op.getInput());
  }

  return success();
}

static LogicalResult LegalizeFloatingPointPrelu(Operation* op,
                                                PatternRewriter& rewriter,
                                                Value input, Value alpha,
                                                ShapedType output_type) {
  Value mul = CreateMulOpAndInfer(rewriter, op, output_type, input, alpha);
  auto rank = mul.getType().cast<ShapedType>().getRank();
  Value const_zero = getTosaConstTensorSingleF32(rewriter, op, 0.0, rank);
  auto ge = CreateOpAndInfer<tosa::GreaterEqualOp>(
      rewriter, op->getLoc(), output_type.clone(rewriter.getIntegerType(1)),
      input, const_zero);

  CreateReplaceOpAndInfer<tosa::SelectOp>(rewriter, op, output_type, ge, input,
                                          mul);

  return success();
}

static LogicalResult LegalizeQuantizedPrelu(Operation* op,
                                            PatternRewriter& rewriter,
                                            Value input, double alpha_scale,
                                            ShapedType output_type) {
  auto tfl_prelu_op = dyn_cast<TFL::PReluOp>(op);

  if (tfl_prelu_op == nullptr)
    return rewriter.notifyMatchFailure(op, "op is not PReLU");

  ShapedType rescale_type = output_type.clone(rewriter.getI32Type());
  ShapedType input_type = dyn_cast<ShapedType>(input.getType());

  UniformQuantizedType input_qtype =
      dyn_cast<UniformQuantizedType>(input_type.getElementType());
  UniformQuantizedType output_qtype =
      dyn_cast<UniformQuantizedType>(output_type.getElementType());

  if (!input_qtype || !output_qtype)
    return rewriter.notifyMatchFailure(
        op, "input or output is not an uniform quantized type");

  double scale_alpha =
      input_qtype.getScale() * alpha_scale / output_qtype.getScale();

  double scale_identity = input_qtype.getScale() / output_qtype.getScale();

  // Implement PReLU as:
  //   rescaled_in = rescale(in)
  //   rescaled_alpha = rescale(alpha)
  //   rescaled_identity_in = rescale(in, scale_identity)
  //   slope_in = mul(rescaled_in, rescaled_alpha)
  //   rescaled_slope_in = rescale(slope_in, scale_alpha)
  //   cond_result = greater_equal(rescaled_in, 0)
  //   output = select(cond_result, rescaled_identity_in, rescaled_slope_in)

  Value op_rescale_in = removeZeroPointAndCastToInt32(
      rewriter, op, input, input_qtype.getZeroPoint());
  auto rank = rescale_type.getRank();
  Value const_zero = getTosaConstTensorSingleI32(rewriter, op, 0, rank);
  Value op_ge = CreateOpAndInfer<tosa::GreaterEqualOp>(
      rewriter, op->getLoc(), rescale_type.clone(rewriter.getI1Type()),
      op_rescale_in, const_zero);

  // Initalize the negative values to the slope of leaky ReLU.
  Value op_rescale_slope_in = buildRescale(
      rewriter, op, output_type, input, scale_alpha, input_qtype.getZeroPoint(),
      output_qtype.getZeroPoint(), "DOUBLE_ROUND", true);

  // Perform an element-wise multiplication on rescaled alpha and input for
  // PReLU.
  Value alpha = tfl_prelu_op.getAlpha();
  ShapedType alpha_type = mlir::cast<ShapedType>(alpha.getType());
  UniformQuantizedType alpha_qtype =
      mlir::cast<UniformQuantizedType>(alpha_type.getElementType());

  Value op_rescale_alpha = removeZeroPointAndCastToInt32(
      rewriter, op, alpha, alpha_qtype.getZeroPoint());

  Value op_mul = CreateMulOpAndInfer(rewriter, op, rescale_type, op_rescale_in,
                                     op_rescale_alpha);

  op_rescale_slope_in =
      buildRescale(rewriter, op, output_type, op_mul, scale_alpha,
                   /* input_zp = */ 0, output_qtype.getZeroPoint(), "DOUBLE_ROUND", true);

  Value op_rescale_identity_in = buildRescale(
      rewriter, op, output_type, input, scale_identity,
      input_qtype.getZeroPoint(), output_qtype.getZeroPoint(), "DOUBLE_ROUND", true);

  CreateReplaceOpAndInfer<tosa::SelectOp>(rewriter, op, output_type, op_ge,
                                          op_rescale_identity_in,
                                          op_rescale_slope_in);

  return success();
}

LogicalResult ConvertTFLPReluOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_prelu_op = cast<TFL::PReluOp>(op);

  ShapedType input_type =
      dyn_cast<ShapedType>(tfl_prelu_op.getInput().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_prelu_op.getResult().getType());
  ShapedType alpha_type =
      dyn_cast<ShapedType>(tfl_prelu_op.getAlpha().getType());

  if (!input_type || !output_type || !alpha_type)
    return rewriter.notifyMatchFailure(
        op, "input, output, or alpha is not a ShapedType");

  if (auto alpha_qtype =
          dyn_cast<UniformQuantizedType>(alpha_type.getElementType())) {
    return LegalizeQuantizedPrelu(op, rewriter, tfl_prelu_op.getInput(),
                                  alpha_qtype.getScale(), output_type);
  }

  return LegalizeFloatingPointPrelu(op, rewriter, tfl_prelu_op.getInput(),
                                    tfl_prelu_op.getAlpha(), output_type);
}

static LogicalResult LegalizeQuantizedLeakyRelu(Operation* op,
                                                PatternRewriter& rewriter,
                                                Value input, double alpha,
                                                ShapedType output_type) {
  ShapedType input_type = mlir::cast<ShapedType>(input.getType());
  ShapedType rescale_type = input_type.clone(rewriter.getI32Type());

  UniformQuantizedType input_qtype =
      cast<UniformQuantizedType>(input_type.getElementType());

  UniformQuantizedType output_qtype =
      cast<UniformQuantizedType>(output_type.getElementType());

  double scale_alpha = input_qtype.getScale() * alpha / output_qtype.getScale();

  double scale_identity = input_qtype.getScale() / output_qtype.getScale();

  // Implement leaky ReLU as:
  //   op_rescale_identity_in = rescale(in, scale_identity)  to Int32
  //   op_rescale_alpha_in = rescale(in, scale_alpha)        to Int32
  //
  //   alpha <= 1: result_int32 = max(op_rescale_identity_in,
  //                                  op_rescale_alpha_in)
  //   alpha >= 1: result_int32 = min(op_rescale_identity_in,
  //                                  op_rescale_alpha_in)
  //
  //   output = rescaleFromInt32(result_int32) to output_type

  Value op_rescale_alpha_in =
      buildRescale(rewriter, op, rescale_type, input, scale_alpha,
                   input_qtype.getZeroPoint(), 0, "DOUBLE_ROUND", true);

  Value op_rescale_identity_in =
      buildRescale(rewriter, op, rescale_type, input, scale_identity,
                   input_qtype.getZeroPoint(), 0, "DOUBLE_ROUND", true);

  Value result_int32;
  if (alpha <= 1.0) {
    auto max_op = CreateOpAndInfer<tosa::MaximumOp>(
        rewriter, op->getLoc(), rescale_type, op_rescale_identity_in,
        op_rescale_alpha_in);
    result_int32 = max_op.getResult();
  } else {
    auto min_op = CreateOpAndInfer<tosa::MinimumOp>(
        rewriter, op->getLoc(), rescale_type, op_rescale_identity_in,
        op_rescale_alpha_in);
    result_int32 = min_op.getResult();
  }

  Value output = buildRescaleFromInt32(rewriter, op, output_type, result_int32,
                                       1.0, output_qtype.getZeroPoint());

  rewriter.replaceOp(op, {output});

  return success();
}

static LogicalResult LegalizeFloatingPointLeakyRelu(Operation* op,
                                                    PatternRewriter& rewriter,
                                                    Value input, double alpha,
                                                    ShapedType output_type) {
  auto rank = input.getType().cast<ShapedType>().getRank();
  Value const_alpha = getTosaConstTensorSingleF32(rewriter, op, alpha, rank);
  auto mul = CreateMulOpAndInfer(rewriter, op, output_type, input, const_alpha);
  if (alpha <= 1.0) {
    // leaky_relu( input ) = max(input*alpha, input)
    CreateReplaceOpAndInfer<tosa::MaximumOp>(rewriter, op, output_type, mul,
                                             input);
  } else {
    // leaky_relu( input ) = min(input*alpha, input)
    CreateReplaceOpAndInfer<tosa::MinimumOp>(rewriter, op, output_type, mul,
                                             input);
  }
  return success();
}

LogicalResult ConvertTFLLeakyReluOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_leakyrelu_op = cast<TFL::LeakyReluOp>(op);
  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_leakyrelu_op.getInput().getType());

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_leakyrelu_op.getResult().getType());

  if (!input_type || !output_type)
    return rewriter.notifyMatchFailure(op,
                                       "input or output is not a ShapedType");

  bool input_is_qtype =
      mlir::isa<mlir::quant::UniformQuantizedType>(input_type.getElementType());
  bool output_is_qtype = mlir::isa<mlir::quant::UniformQuantizedType>(
      output_type.getElementType());

  if (input_is_qtype != output_is_qtype) {
    return rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or "
        "all floating-point");
  }

  // Implement LeakyRelu as element-wise:
  //   out = x > 0 ? x : alpha * x
  //
  // alpha <= 1:  leaky_relu( x ) = max(x*alpha, x)
  // alpha >= 1:  leaky_relu( x ) = min(x*alpha, x)

  FloatAttr tmpAttr = tfl_leakyrelu_op.getAlphaAttr();
  // There is disagreement between the MLIR .td defaults and TF
  // documentation on 0.2 vs 0.3, but 0.2 will be used here.
  double alpha = 0.2;

  if (tmpAttr) {
    alpha = tmpAttr.getValueAsDouble();
  }

  Value input = tfl_leakyrelu_op.getInput();

  if (output_is_qtype) {
    return LegalizeQuantizedLeakyRelu(op, rewriter, input, alpha, output_type);
  }
  return LegalizeFloatingPointLeakyRelu(op, rewriter, input, alpha,
                                        output_type);
}

LogicalResult ConvertTFLNegOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_neg_op = cast<TFL::NegOp>(op);
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_neg_op.getResult().getType());
  if (!output_type) return failure();

  CreateReplaceOpAndInfer<tosa::NegateOp>(rewriter, op, output_type,
                                          tfl_neg_op.getX());

  return success();
}

LogicalResult ConvertTFLYieldOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<tosa::YieldOp>(op, op->getResultTypes(),
                                             op->getOperands());

  return success();
}

LogicalResult ConvertTFLCustomOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_custom_op = cast<TFL::CustomOp>(op);
  rewriter.replaceOpWithNewOp<tosa::CustomOp>(
      op, op->getResultTypes(), tfl_custom_op.getCustomCode(),
      rewriter.getStringAttr("TFL"),
      mlir::cast<mlir::TFL::ConstBytesAttr>(tfl_custom_op.getCustomOption())
          .getValue()
          .str(),
      op->getOperands());

  return success();
}

LogicalResult ConvertTFLReverseV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_reverse_op = cast<TFL::ReverseV2Op>(op);

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_reverse_op.getInput().getType());
  if (!input_type) {
    return rewriter.notifyMatchFailure(op, "input_type is unranked");
  }

  ElementsAttr axis_elems;
  if (!matchPattern(tfl_reverse_op.getAxis(), m_Constant(&axis_elems))) {
    return rewriter.notifyMatchFailure(op, "axis values not constant");
  }

  auto input_rank = input_type.getShape().size();
  llvm::SmallVector<int64_t> axis_vals;
  for (auto axis : axis_elems.getValues<APInt>()) {
    int64_t axis_val = axis.getSExtValue();
    if (axis_val < 0) axis_val += input_rank;
    if (axis_val < 0 || axis_val >= input_rank) {
      return rewriter.notifyMatchFailure(
          op, "axis values not within range of input shape");
    }
    axis_vals.push_back(axis_val);
  }

  Value val = tfl_reverse_op.getInput();
  if (axis_elems.getNumElements() == 0) {
    auto identity_op = CreateOpAndInfer<tosa::IdentityOp>(
        rewriter, op->getLoc(), input_type, val);
    val = identity_op.getResult();
  } else {
    for (auto axis_val : axis_vals) {
      auto axis_attr = rewriter.getI32IntegerAttr(axis_val);
      auto reverse_op = CreateOpAndInfer<tosa::ReverseOp>(
          rewriter, op->getLoc(), input_type, val, axis_attr);

      val = reverse_op.getResult();
    }
  }

  rewriter.replaceOp(op, {val});

  return success();
}

LogicalResult ConvertTFLQuantizeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_quantize_op = cast<TFL::QuantizeOp>(op);

  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(tfl_quantize_op.getInput().getType());
  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_quantize_op.getResult().getType());
  if (!input_type || !output_type) return failure();

  ShapedType qtype =
      dyn_cast<ShapedType>(tfl_quantize_op.getResult().getType());
  if (!qtype) return failure();

  UniformQuantizedType element_type =
      dyn_cast<UniformQuantizedType>(qtype.getElementType());
  if (!element_type) return failure();

  UniformQuantizedType input_element_type =
      dyn_cast<UniformQuantizedType>(input_type.getElementType());

  // If input is already a quantized type, this is basically a RESCALE (or
  // tensorflow::ops::Requantize)
  if (input_element_type) {
    double rescale_scale =
        input_element_type.getScale() / element_type.getScale();
    Value rescale_op =
        buildRescale(rewriter, op, output_type, tfl_quantize_op.getInput(),
                     rescale_scale, input_element_type.getZeroPoint(),
                     element_type.getZeroPoint(), "DOUBLE_ROUND", true);

    rewriter.replaceOp(op, {rescale_op});
    return success();
  } else {
    double scale = 1 / element_type.getScale();
    int64_t zp = element_type.getZeroPoint();
    int64_t num_bits = element_type.getStorageTypeIntegralWidth();
    zp = element_type.isSigned() ? zp : zp - (1 << (num_bits - 1));

    std::optional<Value> result = convertQuantizeOp(
        rewriter, op, output_type, tfl_quantize_op.getInput(), scale, zp);

    if (!result) return failure();

    rewriter.replaceOp(op, {result.value()});

    return success();
  }
}

LogicalResult ConvertTFLDequantizeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_dequantize_op = cast<TFL::DequantizeOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_dequantize_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  RankedTensorType qtype =
      dyn_cast<RankedTensorType>(tfl_dequantize_op.getInput().getType());
  if (!qtype) return failure();

  Type element_type = qtype.getElementType();
  if (mlir::isa<FloatType>(element_type)) {
    CreateReplaceOpAndInfer<tosa::CastOp>(rewriter, op, output_type,
                                          tfl_dequantize_op.getInput());
    return success();
  }

  if (auto eq_ty = dyn_cast<quant::UniformQuantizedType>(element_type)) {
    double scale = eq_ty.getScale();
    int64_t zp = eq_ty.getZeroPoint();
    int64_t num_bits = eq_ty.getStorageTypeIntegralWidth();
    zp = eq_ty.isSigned() ? zp : zp - (1 << (num_bits - 1));

    std::optional<Value> result = convertDequantizeOp(
        rewriter, op, output_type, tfl_dequantize_op.getInput(), scale, zp, 0);

    if (!result) return failure();

    rewriter.replaceOp(op, {result.value()});
    return success();
  }

  if (quant::UniformQuantizedPerAxisType eq_ty =
          dyn_cast<quant::UniformQuantizedPerAxisType>(element_type)) {
    SmallVector<float> zps;
    for (auto zp : eq_ty.getZeroPoints()) {
      int64_t num_bits = eq_ty.getStorageTypeIntegralWidth();
      zps.push_back(eq_ty.isSigned() ? zp : zp - (1 << (num_bits - 1)));
    }

    SmallVector<float> scales;
    for (auto scale : eq_ty.getScales()) {
      scales.push_back(scale);
    }

    std::optional<Value> result = convertDequantizeOp(
        rewriter, op, output_type, tfl_dequantize_op.getInput(), scales, zps,
        eq_ty.getQuantizedDimension());

    if (!result) return failure();

    rewriter.replaceOp(op, {result.value()});
    return success();
  }

  return failure();
}

LogicalResult ConvertTFLConstOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_const_op = cast<TFL::ConstOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_const_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr elements = tfl_const_op.getValue();
  Type element_type = elements.getShapedType().getElementType();
  if (mlir::isa<quant::QuantizedType>(output_type.getElementType())) {
    output_type = RankedTensorType::get(output_type.getShape(), element_type);
  }

  // If the output shape is unranked we can extract the result shape from the
  // attribute shape. This occurs as some TFLite folders create constants with
  // unranked shapes.
  if (!output_type.hasRank()) {
    output_type =
        mlir::cast<ShapedType>(elements.getType()).clone(element_type);
  }

  rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, output_type, elements);

  return success();
}

LogicalResult ConvertTFLQConstOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_qconst_op = cast<TFL::QConstOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_qconst_op.getResult().getType());
  if (!output_type) return failure();

  ElementsAttr elements = tfl_qconst_op.getValue();

  // If the output shape is unranked we can extract the result shape from the
  // attribute shape. This occurs as some TFLite folders create constants with
  // unranked shapes.
  if (!output_type.hasRank()) {
    output_type = mlir::cast<ShapedType>(elements.getType())
                      .clone(output_type.getElementType());
  }

  rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, output_type, elements);

  return success();
}

LogicalResult ConvertConstantOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_const_op = cast<arith::ConstantOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(tfl_const_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  ElementsAttr attr = dyn_cast<ElementsAttr>(tfl_const_op.getValueAttr());

  auto e_type = output_type.getElementType();
  // TOSA only support up to 48-bits
  // If source is higher than that, it's not representabble.
  // For data type like 64 bits, we need to truncate them into 48 bits.
  if (e_type.isInteger(64)) {
    e_type = rewriter.getIntegerType(48);
    attr = mlir::cast<DenseIntOrFPElementsAttr>(attr).mapValues(
        e_type, [](const APInt& x) -> APInt { return x.trunc(48); });
  }

  if (!output_type.hasRank()) {
    if (auto attr_type = dyn_cast<ShapedType>(attr.getType())) {
      output_type = attr_type.clone(e_type);
    }
  }

  output_type = output_type.clone(e_type);
  rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, output_type, attr);

  return success();
}

LogicalResult ConvertTFLGatherOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_gather_op = cast<TFL::GatherOp>(op);

  int32_t axis = tfl_gather_op.getAxisAttr().getInt();
  int32_t batch_dims = 0;
  if (auto batch_attr = tfl_gather_op.getBatchDimsAttr()) {
    batch_dims = static_cast<int32_t>(batch_attr.getInt());
  }

  std::optional<Value> result =
      convertGatherOp(rewriter, op, tfl_gather_op.getParams(),
                      tfl_gather_op.getIndices(), batch_dims, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLGatherNdOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_gathernd_op = cast<TFL::GatherNdOp>(op);

  std::optional<Value> result = convertGatherNdOp(
      rewriter, op, tfl_gathernd_op.getResult(), tfl_gathernd_op.getParams(),
      tfl_gathernd_op.getIndices());

  if (!result) return failure();
  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLSparseToDenseOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_sparse_to_dense_op = cast<TFL::SparseToDenseOp>(op);
  auto indices = tfl_sparse_to_dense_op.getSparseIndices();
  auto values = tfl_sparse_to_dense_op.getSparseValues();
  auto default_value = tfl_sparse_to_dense_op.getDefaultValue();
  auto indices_ty = mlir::cast<ShapedType>(indices.getType());
  auto indices_ety = indices_ty.getElementType();
  auto values_ty = mlir::cast<ShapedType>(values.getType());
  auto result_ty =
      mlir::cast<ShapedType>(tfl_sparse_to_dense_op.getResult().getType());
  auto result_ety = result_ty.getElementType();
  auto loc = op->getLoc();

  if (!result_ty.hasStaticShape()) return failure();
  auto result_rank = result_ty.getRank();

  // We want to generate the default tensor we need to scatter. Note that the
  // result_ty needs to be a statically shaped tensor.
  ElementsAttr default_value_attr;
  if (!matchPattern(default_value, m_Constant(&default_value_attr)))
    return failure();

  if (!default_value_attr.isSplat()) return failure();

  ShapedType scatter_ty =
      RankedTensorType::get({1, result_ty.getNumElements(), 1}, result_ety);

  Value default_const = rewriter.create<tosa::ConstOp>(
      loc, scatter_ty,
      DenseElementsAttr::get(scatter_ty,
                             default_value_attr.getSplatValue<APInt>().sext(
                                 result_ety.getIntOrFloatBitWidth())));

  // We need to determine what the index multiplier does
  llvm::SmallVector<int32_t> multiply_constant_ints;
  multiply_constant_ints.resize(result_rank, 1);
  for (int i = result_rank - 1; i > 0; i--) {
    multiply_constant_ints[i - 1] =
        result_ty.getDimSize(i) * multiply_constant_ints[i];
  }

  indices_ety = rewriter.getI32Type();
  indices_ty = RankedTensorType::get(indices_ty.getShape(), indices_ety);
  indices = CreateOpAndInfer<tosa::CastOp>(rewriter, loc, indices_ty, indices);

  auto multiply_constant_type =
      RankedTensorType::get({result_rank}, indices_ety);
  auto multiply_constant_attr = DenseElementsAttr::get(
      multiply_constant_type, llvm::ArrayRef(multiply_constant_ints));
  Value multiply_constant = CreateOpAndInfer<tosa::ConstOp>(
      rewriter, loc, multiply_constant_type, multiply_constant_attr);

  Value multiply_op =
      CreateMulOpAndInfer(rewriter, op, indices_ty, indices, multiply_constant);

  Value reduce_op = CreateOpAndInfer<tosa::ReduceSumOp>(
      rewriter, loc, UnrankedTensorType::get(indices_ety), multiply_op,
      rewriter.getI32IntegerAttr(1));

  auto shape_value = getTosaConstShape(
      rewriter, op->getLoc(),
      tensorflow::ConvertMlirShapeToTF({1, values_ty.getDimSize(0), 1}));
  auto values_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, loc, UnrankedTensorType::get(result_ety), values, shape_value);

  shape_value = getTosaConstShape(
      rewriter, op->getLoc(),
      tensorflow::ConvertMlirShapeToTF({1, indices_ty.getDimSize(0)}));
  auto index_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, loc, UnrankedTensorType::get(indices_ety), reduce_op,
      shape_value);

  auto scatter = CreateOpAndInfer<tosa::ScatterOp>(
      rewriter, loc, UnrankedTensorType::get(result_ety), default_const,
      index_reshape_op, values_reshape_op);

  auto result_shape_value =
      getTosaConstShape(rewriter, op->getLoc(),
                        tensorflow::ConvertMlirShapeToTF(result_ty.getShape()));
  CreateReplaceOpAndInfer<tosa::ReshapeOp>(rewriter, op, result_ty, scatter,
                                           result_shape_value);

  return success();
}

LogicalResult ConvertTFLOneHotOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_one_hot_op = cast<TFL::OneHotOp>(op);

  ElementsAttr depth_elems;
  if (!matchPattern(tfl_one_hot_op.getDepth(), m_Constant(&depth_elems)))
    return failure();
  int32_t depth = depth_elems.getValues<APInt>()[0].getSExtValue();

  IntegerAttr axisAttr = tfl_one_hot_op.getAxisAttr();
  int32_t axis = axisAttr.getInt();

  std::optional<Value> result = convertOneHotOp(
      rewriter, op, tfl_one_hot_op.getResult(), tfl_one_hot_op.getIndices(),
      tfl_one_hot_op.getOnValue(), tfl_one_hot_op.getOffValue(), depth, axis);

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult ConvertTFLArgMaxOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto arg_max_op = cast<TFL::ArgMaxOp>(op);

  ElementsAttr dim_elems;
  if (!matchPattern(arg_max_op.getDim(), m_Constant(&dim_elems)))
    return failure();

  int32_t dim = dim_elems.getValues<APInt>()[0].getSExtValue();

  if (dim < 0) {
    auto input_type = cast<RankedTensorType>(arg_max_op.getInput().getType());
    dim += input_type.getRank();
  }

  CreateReplaceOpAndInfer<tosa::ArgMaxOp>(
      rewriter, op, arg_max_op.getType(), arg_max_op.getInput(),
      rewriter.getI32IntegerAttr(dim), rewriter.getStringAttr("PROPAGATE"));

  return success();
}

LogicalResult ConvertTFLArgMinOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto arg_min_op = cast<TFL::ArgMinOp>(op);
  auto loc = arg_min_op.getLoc();
  Value input = arg_min_op.getInput();
  auto input_ty = mlir::cast<ShapedType>(input.getType());
  Type input_ety = input_ty.getElementType();
  bool input_type_casted = false;

  if (auto quantized_ty = dyn_cast<UniformQuantizedType>(input_ety)) {
    input_ety =
        rewriter.getIntegerType(quantized_ty.getStorageTypeIntegralWidth());
    if (quantized_ty.getZeroPoint() != 0) {
      // cast to input_ety type to get rid of zero point
      // so that sub operands are both of type input_ety
      input = CreateOpAndInfer<tosa::CastOp>(rewriter, loc,
                                             input_ty.clone(input_ety), input);
      input_type_casted = true;
    }
  }

  if (!input_ety.isIntOrFloat())
    return rewriter.notifyMatchFailure(op, "unsupported element type");

  ElementsAttr dim_elems;
  if (!matchPattern(arg_min_op.getDim(), m_Constant(&dim_elems)))
    return rewriter.notifyMatchFailure(op, "Non-constant dim");

  // When negative dim is measured from the back of the array.
  int32_t dim = dim_elems.getValues<APInt>()[0].getSExtValue();
  if (dim < 0) dim += input_ty.getRank();

  if (mlir::isa<FloatType>(input_ety)) {
    input = CreateOpAndInfer<tosa::NegateOp>(rewriter, loc, input_ty, input);
  } else if (mlir::isa<IntegerType>(input_ety)) {
    std::vector<int64_t> shape(input_ty.getRank(), 1);
    auto reverse_ty = RankedTensorType::get(shape, input_ety);
    Value reverse_val = rewriter.create<tosa::ConstOp>(
        loc, reverse_ty,
        DenseElementsAttr::get(reverse_ty,
                               rewriter.getIntegerAttr(input_ety, -1)));
    input = CreateOpAndInfer<tosa::SubOp>(
        rewriter, loc, input_ty.clone(input_ety), reverse_val, input);
  }

  // if output type has zero point, the input has been type cast to integer
  // so need to rescale ArgMax output to original output zero point
  int output_zp = 0;
  Type output_ty = arg_min_op.getType();
  Type output_ety = output_ty.cast<ShapedType>().getElementType();
  if (auto output_quantized_ty = dyn_cast<UniformQuantizedType>(output_ety)) {
    output_zp = output_quantized_ty.getZeroPoint();
    if (output_zp != 0) {
      // need to rescale arg_max output to output zero point
      output_ty = output_ty.cast<ShapedType>().clone(input_ety);
    }
  }

  // double check input/output type cast consistency
  assert(input_type_casted == (output_zp != 0));

  Value result = CreateOpAndInfer<tosa::ArgMaxOp>(
      rewriter, loc, output_ty, input, rewriter.getI32IntegerAttr(dim),
      rewriter.getStringAttr("PROPAGATE"));

  if (output_zp != 0) {
    // rescale result to output_zp
    result = buildRescale(rewriter, op, arg_min_op.getType(), result,
                          /* sclae = */ 1.0,
                          /* input_zp = */ 0,
                          /* output_zp = */ output_zp, "SINGLE_ROUND", true);
  }

  rewriter.replaceOp(op, {result});

  return success();
}

LogicalResult ConvertTFLFakeQuantOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto fakequant_op = cast<TFL::FakeQuantOp>(op);

  ShapedType output_type =
      dyn_cast<ShapedType>(fakequant_op.getResult().getType());
  // Not a ranked tensor output
  if (!output_type) return failure();

  std::optional<Value> result =
      convertFakeQuantOp(rewriter, op, output_type, fakequant_op.getInput(),
                         fakequant_op.getMinAttr().getValueAsDouble(),
                         fakequant_op.getMaxAttr().getValueAsDouble(),
                         fakequant_op.getNumBitsAttr().getInt(),
                         fakequant_op.getNarrowRangeAttr().getValue());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

// Clone block, convert yield from TFL to TOSA
static void inlineWhileCase(Region& srcRegion, Region& dstRegion,
                            PatternRewriter& rewriter) {
  rewriter.cloneRegionBefore(srcRegion, &dstRegion.back());
  rewriter.eraseBlock(&dstRegion.back());

  Block* headBlock = &dstRegion.front();

  auto yield = cast<mlir::TFL::YieldOp>(headBlock->getTerminator());
  rewriter.setInsertionPoint(yield);
  rewriter.create<mlir::tosa::YieldOp>(yield.getLoc(), yield.getOperands());
  rewriter.eraseOp(yield);
}

LogicalResult ConvertTFLWhileOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_while_op = cast<TFL::WhileOp>(op);

  auto while_op = rewriter.create<mlir::tosa::WhileOp>(
      op->getLoc(), op->getResultTypes(), op->getOperands());

  rewriter.createBlock(&while_op.getCondGraph());
  rewriter.createBlock(&while_op.getBodyGraph());

  inlineWhileCase(tfl_while_op.getCond(), while_op.getCondGraph(), rewriter);
  inlineWhileCase(tfl_while_op.getBody(), while_op.getBodyGraph(), rewriter);

  rewriter.replaceOp(tfl_while_op, while_op.getResults());

  return success();
}

LogicalResult ConvertTFLRealOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_real_op = cast<TFL::RealOp>(op);
  Value input = tfl_real_op.getInput();

  auto input_ty = dyn_cast<RankedTensorType>(input.getType());
  auto output_ty = dyn_cast<ShapedType>(tfl_real_op.getResult().getType());

  if (!input_ty || !output_ty) {
    return rewriter.notifyMatchFailure(op, "ranked input/output required");
  }

  Type input_ety = input_ty.getElementType();

  // For non-complex inputs, return the original tensor.
  if (!mlir::isa<ComplexType>(input_ety)) {
    CreateReplaceOpAndInfer<tosa::IdentityOp>(rewriter, op, input_ty, input);
    return success();
  }

  if (!mlir::cast<ComplexType>(input_ety).getElementType().isF32()) {
    return rewriter.notifyMatchFailure(
        op, "complex input must be of type complex64");
  }

  // Complex64 inputs of shape [x, ..., y] are lowered to TOSA
  // as a tensor of datatype float32 and shape [x, ..., y, 2].
  // This is because TOSA does not have support for complex
  // types. An unrealized conversion is inserted to handle
  // this discrepancy.
  llvm::SmallVector<int64_t> cast_size = to_vector(input_ty.getShape());
  cast_size.push_back(2);
  auto casted_ty = RankedTensorType::get(cast_size, rewriter.getF32Type());
  Value casted = rewriter
                     .create<mlir::UnrealizedConversionCastOp>(op->getLoc(),
                                                               casted_ty, input)
                     .getResult(0);

  // Slice the input to get the "real" values.
  const int64_t rank = input_ty.getRank();
  llvm::SmallVector<int64_t> start(rank + 1, 0);
  llvm::SmallVector<int64_t> size = to_vector(input_ty.getShape());
  size.push_back(1);
  auto slice_ty = RankedTensorType::get(size, rewriter.getF32Type());
  auto slice_op = CreateOpAndInfer<tosa::SliceOp>(
      rewriter, op->getLoc(), slice_ty, casted,
      getTosaConstShape(rewriter, op->getLoc(), start),
      getTosaConstShape(rewriter, op->getLoc(), size));
  auto output_shape_value =
      getTosaConstShape(rewriter, op->getLoc(), output_ty.getShape());
  CreateReplaceOpAndInfer<tosa::ReshapeOp>(rewriter, op, output_ty, slice_op,
                                           output_shape_value);

  return success();
}

LogicalResult ConvertTFLImagOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_imag_op = cast<TFL::ImagOp>(op);
  Value input = tfl_imag_op.getInput();

  auto input_ty = dyn_cast<RankedTensorType>(input.getType());
  auto output_ty = dyn_cast<ShapedType>(tfl_imag_op.getResult().getType());

  if (!input_ty || !output_ty) {
    return rewriter.notifyMatchFailure(op, "ranked input/output required");
  }

  Type input_ety = input_ty.getElementType();

  // For non-complex inputs return all zero's.
  if (!mlir::isa<ComplexType>(input_ety)) {
    CreateReplaceOpAndInfer<tosa::ConstOp>(
        rewriter, op, input_ty, DenseElementsAttr::get(input_ty, {0.0f}));
    return success();
  }

  if (!mlir::cast<ComplexType>(input_ety).getElementType().isF32()) {
    return rewriter.notifyMatchFailure(
        op, "complex input must be of type complex64");
  }

  // Complex64 inputs of shape [x, ..., y] are lowered to TOSA
  // as a tensor of datatype float32 and shape [x, ..., y, 2].
  // This is because TOSA does not have support for complex
  // types. An unrealized conversion is inserted to handle
  // this discrepancy.
  llvm::SmallVector<int64_t> cast_size = to_vector(input_ty.getShape());
  cast_size.push_back(2);
  auto casted_ty = RankedTensorType::get(cast_size, rewriter.getF32Type());
  Value casted = rewriter
                     .create<mlir::UnrealizedConversionCastOp>(op->getLoc(),
                                                               casted_ty, input)
                     .getResult(0);

  // Slice the input to the "imag" values.
  llvm::SmallVector<int64_t> start(input_ty.getRank(), 0);
  start.push_back(1);
  llvm::SmallVector<int64_t> size = to_vector(input_ty.getShape());
  size.push_back(1);

  auto slice_ty = RankedTensorType::get(size, rewriter.getF32Type());
  auto slice_op = CreateOpAndInfer<tosa::SliceOp>(
      rewriter, op->getLoc(), slice_ty, casted,
      getTosaConstShape(rewriter, op->getLoc(), start),
      getTosaConstShape(rewriter, op->getLoc(), size));
  auto output_shape_value =
      getTosaConstShape(rewriter, op->getLoc(), output_ty.getShape());
  CreateReplaceOpAndInfer<tosa::ReshapeOp>(rewriter, op, output_ty, slice_op,
                                           output_shape_value);

  return success();
}

LogicalResult ConvertTFLRFFT2dOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto rfft2d_op = cast<TFL::RFFT2dOp>(op);
  auto loc = op->getLoc();
  Value input = rfft2d_op.getInput();

  auto input_type = dyn_cast<RankedTensorType>(input.getType());
  auto output_type = dyn_cast<ShapedType>(rfft2d_op.getResult().getType());

  if (!input_type || !output_type) {
    return rewriter.notifyMatchFailure(op, "ranked input/output required");
  }

  if (!input_type.getElementType().isF32()) {
    return rewriter.notifyMatchFailure(op, "input type must be fp32");
  }

  Value fft_length_value = rfft2d_op.getFftLength();
  llvm::SmallVector<int32_t> fft_length;
  if (failed(getVectorFromValue32(fft_length_value, fft_length))) {
    return rewriter.notifyMatchFailure(op, "fft_length is not a constant");
  }

  auto fp32_ty = UnrankedTensorType::get(rewriter.getF32Type());

  // Padding is automatically inserted during the lowering when
  // fft_length > input shape. However, to take care of the
  // case fft_length < input shape we need to crop the input.
  const int64_t rank = input_type.getRank();
  auto input_shape = input_type.getShape();
  if (fft_length[0] < input_shape[rank - 2] ||
      fft_length[1] < input_shape[rank - 1]) {
    llvm::SmallVector<int64_t> slice_begin(rank, 0);
    llvm::SmallVector<int64_t> slice_size;
    for (auto dim : input_type.getShape().drop_back(2)) {
      slice_size.push_back(dim);
    }
    slice_size.push_back(fft_length[0]);
    slice_size.push_back(fft_length[1]);
    input = CreateOpAndInfer<tosa::SliceOp>(
        rewriter, loc, fp32_ty, input,
        getTosaConstShape(rewriter, op->getLoc(), slice_begin),
        getTosaConstShape(rewriter, op->getLoc(), slice_size));
  }

  auto rfft2d =
      CreateOpAndInfer<tosa::RFFT2dOp>(rewriter, loc, fp32_ty, fp32_ty, input);

  auto output_shape = output_type.getShape();
  llvm::SmallVector<int64_t> new_shape{output_shape};
  new_shape.push_back(1);
  auto new_shape_value = getTosaConstShape(rewriter, op->getLoc(), new_shape);
  auto reshape_1 = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, loc, fp32_ty, rfft2d.getResult(0), new_shape_value);
  auto reshape_2 = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, loc, fp32_ty, rfft2d.getResult(1), new_shape_value);

  llvm::SmallVector<Value, 2> values = {reshape_1, reshape_2};
  auto concat = CreateOpAndInfer<tosa::ConcatOp>(rewriter, loc, fp32_ty, values,
                                                 rewriter.getI32IntegerAttr(3));

  CreateReplaceOpAndInfer<mlir::UnrealizedConversionCastOp>(
      rewriter, op, output_type, concat.getResult());

  return success();
}

LogicalResult ConvertTFLLogicalAndOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::LogicalAndOp>(op, rewriter);
}

LogicalResult ConvertTFLLogicalOrOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::LogicalOrOp>(op, rewriter);
}

LogicalResult ConvertTFLPowOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  return ConvertBinaryOp<tosa::PowOp>(op, rewriter);
}

LogicalResult ConvertTFLBroadcastToOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_broadcast_to_op = cast<TFL::BroadcastToOp>(op);

  std::optional<Value> result =
      convertBroadcastToOp(rewriter, op, tfl_broadcast_to_op.getInput(),
                           tfl_broadcast_to_op.getShape());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.value()});

  return success();
}

LogicalResult LegalizeTFL::initialize(MLIRContext* context) {
  RewritePatternSet patterns(context);
  mlir::tosa::populateLegalizeTFLPatterns(context, patterns);
  frozen_patterns_ = FrozenRewritePatternSet(
      std::move(patterns), this->disabled_patterns_, this->enabled_patterns_);
  return success();
}

void LegalizeTFL::runOnOperation() {
  if (ApplyPatternsWithShapeResolution(getOperation(), this->frozen_patterns_)
          .failed()) {
    signalPassFailure();
  }
}

}  // namespace

void populateLegalizeTFLPatterns(MLIRContext* ctx,
                                 RewritePatternSet& patterns) {
#define DEF_PATTERN_INSERT(PAT) \
  patterns.addWithLabel<Convert##PAT##Op>({#PAT}, ctx);

  DEF_PATTERN_INSERT(TFLAbs);
  DEF_PATTERN_INSERT(TFLCeil);
  DEF_PATTERN_INSERT(TFLFloor);
  DEF_PATTERN_INSERT(TFLExp);
  DEF_PATTERN_INSERT(TFLLog);
  DEF_PATTERN_INSERT(TFLRsqrt);
  DEF_PATTERN_INSERT(TFLLogicalNot);
  DEF_PATTERN_INSERT(TFLCast);

  DEF_PATTERN_INSERT(QuantStat);

  DEF_PATTERN_INSERT(TFLLogicalAnd);
  DEF_PATTERN_INSERT(TFLLogicalOr);
  DEF_PATTERN_INSERT(TFLPow);

  DEF_PATTERN_INSERT(TFLGelu);
  DEF_PATTERN_INSERT(TFLRelu);
  DEF_PATTERN_INSERT(TFLRelu1);
  DEF_PATTERN_INSERT(TFLRelu0To1);
  DEF_PATTERN_INSERT(TFLRelu6);
  DEF_PATTERN_INSERT(TFLEqual);
  DEF_PATTERN_INSERT(TFLNotEqual);
  DEF_PATTERN_INSERT(TFLGreater);
  DEF_PATTERN_INSERT(TFLGreaterEqual);
  DEF_PATTERN_INSERT(TFLAdd);
  DEF_PATTERN_INSERT(TFLSub);
  DEF_PATTERN_INSERT(TFLMul);
  DEF_PATTERN_INSERT(TFLSquare);
  DEF_PATTERN_INSERT(TFLSquaredDifference);
  DEF_PATTERN_INSERT(TFLSign);
  DEF_PATTERN_INSERT(TFLRound);
  DEF_PATTERN_INSERT(TFLDiv);
  DEF_PATTERN_INSERT(TFLMaximum);
  DEF_PATTERN_INSERT(TFLMinimum);
  DEF_PATTERN_INSERT(TFLFloorMod);
  DEF_PATTERN_INSERT(TFLFloorDiv);
  DEF_PATTERN_INSERT(TFLAddN);
  DEF_PATTERN_INSERT(TFLAveragePool2D);
  DEF_PATTERN_INSERT(TFLMaxPool2D);
  DEF_PATTERN_INSERT(TFLConcatenation);
  DEF_PATTERN_INSERT(TFLReshape);
  DEF_PATTERN_INSERT(TFLRank);
  DEF_PATTERN_INSERT(TFLShape);
  DEF_PATTERN_INSERT(TFLExpandDims);
  DEF_PATTERN_INSERT(TFLSqueeze);
  DEF_PATTERN_INSERT(TFLFill);
  DEF_PATTERN_INSERT(TFLElu);
  DEF_PATTERN_INSERT(TFLSoftmax);
  DEF_PATTERN_INSERT(TFLLogSoftmax);
  DEF_PATTERN_INSERT(TFLSqrt);
  DEF_PATTERN_INSERT(TFLL2Normalization);
  DEF_PATTERN_INSERT(TFLReduceAll);
  DEF_PATTERN_INSERT(TFLReduceAny);
  DEF_PATTERN_INSERT(TFLReduceMax);
  DEF_PATTERN_INSERT(TFLReduceMin);
  DEF_PATTERN_INSERT(TFLMean);
  DEF_PATTERN_INSERT(TFLReduceProd);
  DEF_PATTERN_INSERT(TFLSum);
  DEF_PATTERN_INSERT(TFLConv2D);
  DEF_PATTERN_INSERT(TFLConv3D);
  DEF_PATTERN_INSERT(TFLTransposeConv);
  DEF_PATTERN_INSERT(TFLDepthwiseConv2D);
  DEF_PATTERN_INSERT(TFLFullyConnected);
  DEF_PATTERN_INSERT(TFLBatchMatMul);
  DEF_PATTERN_INSERT(TFLSplit);
  DEF_PATTERN_INSERT(TFLSplitV);
  DEF_PATTERN_INSERT(TFLPack);
  DEF_PATTERN_INSERT(TFLUnpack);
  DEF_PATTERN_INSERT(TFLTranspose);
  DEF_PATTERN_INSERT(TFLTile);
  DEF_PATTERN_INSERT(TFLSlice);
  DEF_PATTERN_INSERT(TFLStridedSlice);
  DEF_PATTERN_INSERT(TFLHardSwish);
  DEF_PATTERN_INSERT(TFLZerosLike);
  DEF_PATTERN_INSERT(TFLLess);
  DEF_PATTERN_INSERT(TFLLessEqual);
  DEF_PATTERN_INSERT(TFLPad);
  DEF_PATTERN_INSERT(TFLMirrorPad);
  DEF_PATTERN_INSERT(TFLPadV2);
  DEF_PATTERN_INSERT(TFLResizeBilinear);
  DEF_PATTERN_INSERT(TFLResizeNearestNeighbor);
  DEF_PATTERN_INSERT(TFLSelect);
  DEF_PATTERN_INSERT(TFLSelectV2);
  DEF_PATTERN_INSERT(TFLSpaceToBatchNd);
  DEF_PATTERN_INSERT(TFLBatchToSpaceNd);
  DEF_PATTERN_INSERT(TFLSpaceToDepth);
  DEF_PATTERN_INSERT(TFLDepthToSpace);
  DEF_PATTERN_INSERT(TFLBucketize);
  DEF_PATTERN_INSERT(TFLSin);
  DEF_PATTERN_INSERT(TFLCos);
  DEF_PATTERN_INSERT(TFLAtan2);
  DEF_PATTERN_INSERT(TFLLogistic);
  DEF_PATTERN_INSERT(TFLTanh);
  DEF_PATTERN_INSERT(TFLPRelu);
  DEF_PATTERN_INSERT(TFLLeakyRelu);
  DEF_PATTERN_INSERT(TFLNeg);
  DEF_PATTERN_INSERT(TFLYield);
  DEF_PATTERN_INSERT(TFLCustom);
  DEF_PATTERN_INSERT(TFLReverseV2);
  DEF_PATTERN_INSERT(TFLQuantize);
  DEF_PATTERN_INSERT(TFLDequantize);
  DEF_PATTERN_INSERT(TFLConst);
  DEF_PATTERN_INSERT(TFLQConst);
  DEF_PATTERN_INSERT(TFLGatherNd);
  DEF_PATTERN_INSERT(TFLSparseToDense);
  DEF_PATTERN_INSERT(Constant);
  DEF_PATTERN_INSERT(TFLOneHot);
  DEF_PATTERN_INSERT(TFLArgMax);
  DEF_PATTERN_INSERT(TFLArgMin);
  DEF_PATTERN_INSERT(TFLFakeQuant);
  DEF_PATTERN_INSERT(TFLWhile);
  DEF_PATTERN_INSERT(TFLReal);
  DEF_PATTERN_INSERT(TFLImag);
  DEF_PATTERN_INSERT(TFLRFFT2d);
  DEF_PATTERN_INSERT(TFLBroadcastTo);
#undef DEF_PATTERN_INSERT

  // Patterns which may optionally generate non-TOSA dialects
#define DEF_PATTERN_INSERT(PAT) \
  patterns.addWithLabel<Convert##PAT##Op>({#PAT "ToTensor"}, ctx);

  DEF_PATTERN_INSERT(TFLGather);

#undef DEF_PATTERN_INSERT
}

// Creates an instance of the TensorFlow Lite dialect LegalizeTFL pass.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFLPass(
    ArrayRef<std::string> disabled_patterns,
    ArrayRef<std::string> enabled_patterns) {
  return std::make_unique<LegalizeTFL>(disabled_patterns, enabled_patterns);
}

}  // namespace tosa
}  // namespace mlir
