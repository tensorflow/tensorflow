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

// This file implements logic for lowering TensorFlow dialect to XLA dialect.

#include <numeric>

#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

using namespace mlir;

namespace {
struct LegalizeTF : public FunctionPass<LegalizeTF> {
  /// Performs the lowering to XLA dialect.
  void runOnFunction() override;
};
}  // end anonymous namespace

std::unique_ptr<mlir::FunctionPassBase> mlir::xla_hlo::createLegalizeTFPass() {
  return std::make_unique<LegalizeTF>();
}

/// Returns if the given TF data format string is the default format.
static bool isDefaultDataFormat(StringRef format) { return format == "NHWC"; }

/// Returns the feature dimension for the given format and input type.
static size_t getFeatureDimension(StringAttr format,
                                  RankedTensorType inputType) {
  return isDefaultDataFormat(format.getValue()) ? inputType.getRank() - 1 : 1;
}

// Returns minimum value for the given int or float element type.
static ConstantOp GetMinValueForType(Type ty, Location loc,
                                     PatternRewriter *rewriter) {
  RankedTensorType scalar_ty = rewriter->getTensorType({}, ty);

  DenseElementsAttr attr;
  if (auto float_ty = ty.dyn_cast_or_null<FloatType>()) {
    APFloat neg_inf =
        APFloat::getInf(float_ty.getFloatSemantics(), /*negative=*/true);
    attr = DenseElementsAttr::get(scalar_ty, neg_inf);
  } else {
    auto int_ty = ty.cast<IntegerType>();
    APInt min_val = APInt::getSignedMinValue(int_ty.getWidth());
    attr = DenseElementsAttr::get(scalar_ty, min_val);
  }
  return rewriter->create<ConstantOp>(loc, attr);
}

// Builds body for reduce op by using the using the template binary op as the
// reducer op.
template <typename Op>
static void BuildReduceBody(Type element_type, Region *body,
                            OpBuilder *builder) {
  OpBuilder::InsertionGuard guard(*builder);
  Block *block = builder->createBlock(body);

  // Block arguments are scalars of the given element type.
  Type type = builder->getTensorType(/*shape=*/{}, element_type);
  block->addArguments({type, type});

  Location loc = body->getLoc();
  auto reducer = builder->create<Op>(loc, type, block->getArgument(0),
                                     block->getArgument(1),
                                     /*broadcast_dimensions=*/nullptr);
  builder->create<xla_hlo::ReturnOp>(loc, reducer.getResult());
}

//===----------------------------------------------------------------------===//
// BatchNorm op utilities.
//===----------------------------------------------------------------------===//

static IntegerAttr getFeatureDimensionAttr(Builder &b, StringAttr format,
                                           Value *input) {
  return b.getI64IntegerAttr(
      getFeatureDimension(format, input->getType().cast<RankedTensorType>()));
}

//===----------------------------------------------------------------------===//
// Bias op utilities.
//===----------------------------------------------------------------------===//

/// Returns whether the biasAdd feature dimension is valid or not.
static bool hasValidBiasFeatureDimension(StringAttr format, Value *input,
                                         Value *bias) {
  auto inputType = input->getType().cast<RankedTensorType>();
  auto biasType = bias->getType().cast<RankedTensorType>();

  // There must be enough biases as the feature dimension of the input tensor.
  size_t featureDim = getFeatureDimension(format, inputType);
  return biasType.getDimSize(0) == inputType.getDimSize(featureDim);
}

/// Return a 1D ElementsAttr for the feature dimension of a BiasAdd.
static ElementsAttr getBiasFeatureDimension(Builder &b, StringAttr format,
                                            Value *input) {
  return b.getDenseIntElementsAttr(
      b.getTensorType(1, b.getIntegerType(64)),
      getFeatureDimension(format, input->getType().cast<RankedTensorType>()));
}

//===----------------------------------------------------------------------===//
// Binary op utilities.
//===----------------------------------------------------------------------===//

/// Get a constant splat for the given value type.
template <typename T>
static ElementsAttr getSplat(Builder &b, Value *val, T constant) {
  auto valType = val->getType().cast<TensorType>();
  auto valElementType = valType.getElementType();

  // Handle integer elements.
  Attribute elementAttr;
  if (valElementType.isa<IntegerType>())
    elementAttr = b.getIntegerAttr(valElementType, constant);
  else if (valElementType.isa<FloatType>())
    elementAttr = b.getFloatAttr(valElementType, constant);
  else
    llvm_unreachable("unhandled element type");
  return DenseElementsAttr::get(valType, elementAttr);
}

static ElementsAttr getBroadcastDimensionsAttr(Builder &b, Value *x, Value *y) {
  TensorType xType = x->getType().dyn_cast<RankedTensorType>();
  TensorType yType = y->getType().dyn_cast<RankedTensorType>();
  if (xType == yType || !xType || !yType) return {};

  // If the shapes have the same rank, then there is nothing to do.
  auto xRank = xType.getRank(), yRank = yType.getRank();
  if (xRank == yRank) return {};

  // Otherwise if the ranks of the inputs don't match, TensorFlow automatically
  // reshapes the smaller by padding with dimensions of size 1 as a prefix. In
  // other words to pad a 5-vector to a 3-dimensional tensor it is reshaped to
  // have shape [1,1,5]. XLA's automatic broadcast code is able to broadcast
  // from lower to higher rank, but doesn't assume you want to pad as a prefix
  // of the dimensions, and instead needs to be told which dimensions of the
  // higher rank tensor to match to the lower rank tensor.
  auto maxRank = std::max(xRank, yRank);
  auto minRank = std::min(xRank, yRank);

  // Match the lower rank tensor along the larger-numbered dimensions of the
  // higher rank tensor.
  SmallVector<int64_t, 4> broadcastDimensions(minRank);
  std::iota(broadcastDimensions.begin(), broadcastDimensions.end(),
            maxRank - minRank);

  return b.getDenseIntElementsAttr(
      b.getTensorType({minRank}, b.getIntegerType(64)), broadcastDimensions);
}

//===----------------------------------------------------------------------===//
// Softmax op utilities.
//===----------------------------------------------------------------------===//

// Returns a 1-d i64 elements attribute populated with numbers from start to
// end, excluding.
static ElementsAttr GetI64AttrForSeq(int start, int end, Builder *builder) {
  int size = end - start;

  SmallVector<int64_t, 4> vals;
  vals.resize(size);
  std::iota(vals.begin(), vals.end(), start);

  TensorType ty = builder->getTensorType({size}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get<int64_t>(ty, vals);
}

// Returns the type to use for accumulating the given type.
static Type GetAccumulationType(Type ty) {
  // Upcast 16 bit sum reductions to 32 bit to reduce the precision loss from
  // repeated floating point additions.
  return (ty.isF16() || ty.isBF16()) ? FloatType::getF32(ty.getContext()) : ty;
}

//===----------------------------------------------------------------------===//
// Op converters.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace xla {
namespace {

// Converts MaxPool op to HLO ReduceWindow op by setting appropriate window
// dimensions with max as the reduction function.
//
// Sample result for VALID padding mode:
//
//   %init = constant dense<...> : tensor<i32>
//   %max_pool = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.max"]
//               {window_dimensions = ..., window_strides = ... }
//
class ConvertMaxPoolOp : public OpRewritePattern<TF::MaxPoolOp> {
 public:
  explicit ConvertMaxPoolOp(MLIRContext *context)
      : OpRewritePattern<TF::MaxPoolOp>(context, 1) {}

  PatternMatchResult matchAndRewrite(TF::MaxPoolOp op,
                                     PatternRewriter &rewriter) const override {
    // TODO(hinsu): Support 'SAME' padding mode.
    if (op.padding() != "VALID") return matchFailure();

    Type element_type =
        op.input()->getType().cast<TensorType>().getElementType();
    if (!element_type.isIntOrFloat()) return matchFailure();
    Location loc = op.getLoc();
    ConstantOp init = GetMinValueForType(element_type, loc, &rewriter);

    auto get_elements_attr = [&](ArrayAttr attr) {
      RankedTensorType ty = rewriter.getTensorType(
          static_cast<int64_t>(attr.size()), rewriter.getIntegerType(64));
      return DenseElementsAttr::get(ty, attr.getValue())
          .cast<DenseIntElementsAttr>();
    };

    auto reduce = rewriter.create<xla_hlo::ReduceWindowOp>(
        loc, op.getType(), op.input(), init.getResult(),
        get_elements_attr(op.ksize()), get_elements_attr(op.strides()),
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(),
        /*paddings=*/DenseIntElementsAttr());
    BuildReduceBody<xla_hlo::MaxOp>(element_type, &reduce.body(), &rewriter);

    rewriter.replaceOp(op.getOperation(), reduce.getResult(0));
    return matchSuccess();
  }
};

// Converts Softmax op to HLO ops computing softmax with the following formula:
//
//     softmax = div(exp(logits), sum(exp(logits)))
//
// Sample result with 2-d f16 inputs with B batches of with N elements each.
//
//    // Subtract each element by their batches' max to improve numerical
//    // stability.
//    %neg_infinity = constant dense<0xFF800000> : tensor<f16>
//    %max = "xla_hlo.reduce"(%input, %neg_infinity) ["xla_hlo.max"]
//             {dimensions = 1}
//           : (tensor<BxNxf16>, tensor<1xf16>) -> tensor<Bxf16>
//    %sub = "xla_hlo.sub"(%inp, %max) {broadcast_dimensions = 0}
//            : (tensor<BxNxf16>, tensor<Bxf16>) -> tensor<BxNxf16>
//
//    %exp = "xla_hlo.exp"(%sub) : (tensor<BxNxf16>) -> tensor<BxNxf16>
//
//    // Cast to f32 to avoid precision loss in summation.
//    %exp_f32 = "xla_hlo.convert"(%exp) : (tensor<BxNxbf16>) -> tensor<BxNxf32>
//    %zero = constant dense<0.000000e+00> : tensor<f32>
//    %sum = "xla_hlo.reduce"(%exp, %zero) ["xla_hlo.add"] {dimensions = 1}
//            : (tensor<BxNxf32>, tensor<1xf32>) -> tensor<Bxf32>
//
//    %sum_f16 = "xla_hlo.convert"(%sum) : (tensor<BxNxbf32>) -> tensor<BxNxf16>
//    %softmax = "xla_hlo.div"(%exp, %sum_f16) {broadcast_dimensions = 0}
//            : (tensor<BxNxf16>, tensor<Bxf16>) -> tensor<BxNxf16>
//
class ConvertSoftmaxOp : public OpRewritePattern<TF::SoftmaxOp> {
 public:
  explicit ConvertSoftmaxOp(MLIRContext *context)
      : OpRewritePattern<TF::SoftmaxOp>(context, 1) {}

  PatternMatchResult matchAndRewrite(TF::SoftmaxOp op,
                                     PatternRewriter &rewriter) const override {
    Value *logits = op.logits();

    // Softmax converter requires ranked type because the XLA reduce ops used
    // while lowering requires dimensions attribute to reduce along.
    RankedTensorType type = logits->getType().dyn_cast<RankedTensorType>();
    if (!type) return matchFailure();
    int rank = type.getRank();

    // Note that the TensorFlow Softmax op verifies that the input rank is
    // greater than or equal to one so both of the following sequences are
    // valid.
    ElementsAttr batch_dims = GetI64AttrForSeq(0, rank - 1, &rewriter);
    ElementsAttr reduce_dim = GetI64AttrForSeq(rank - 1, rank, &rewriter);
    Location loc = op.getLoc();

    // Exponential of input values and then their sum can be very large here.
    // Division with large denominator is numerically unstable. To improve
    // numerical stability, subtract each batch with their max element so that
    // the maximum input value is zero. It can be shown that softmax computed
    // after adding or subtracting all inputs in a batch using a common value
    // gives mathematically equivalent result.
    Type element_type = type.getElementType();
    ArrayRef<int64_t> reduce_shape = type.getShape().drop_back();
    RankedTensorType reduce_out_type =
        rewriter.getTensorType(reduce_shape, element_type);
    auto init = GetMinValueForType(element_type, loc, &rewriter);
    auto max_logits = rewriter.create<xla_hlo::ReduceOp>(
        loc, reduce_out_type, ArrayRef<Value *>{logits, init}, reduce_dim);
    BuildReduceBody<xla_hlo::MaxOp>(element_type, &max_logits.body(),
                                    &rewriter);
    auto shifted_logits = rewriter.create<xla_hlo::SubOp>(
        loc, type, logits, max_logits.getResult(0), batch_dims);

    // Exponentiate the inputs.
    Value *exp = rewriter.create<xla_hlo::ExpOp>(loc, type, shifted_logits);

    // Cast the exponentials to the appropriate accumulation type to avoid
    // precision loss during summation.
    Type sum_element_type = GetAccumulationType(element_type);
    Type sum_type = rewriter.getTensorType(type.getShape(), sum_element_type);
    auto casted_exp = rewriter.create<xla_hlo::ConvertOp>(loc, sum_type, exp);

    // Compute summation of the exponentials.
    init = rewriter.create<ConstantOp>(
        loc, DenseElementsAttr::get(rewriter.getTensorType({}, element_type),
                                    rewriter.getZeroAttr(element_type)));
    Type sum_out_type = rewriter.getTensorType(reduce_shape, sum_element_type);
    auto exp_sum = rewriter.create<xla_hlo::ReduceOp>(
        loc, sum_out_type, ArrayRef<Value *>{casted_exp, init}, reduce_dim);
    BuildReduceBody<xla_hlo::AddOp>(element_type, &exp_sum.body(), &rewriter);
    Value *sum = exp_sum.getResult(0);

    // Convert the summation result back to the original element type and divide
    // exponentials by the summations.
    sum = rewriter.create<xla_hlo::ConvertOp>(loc, reduce_out_type, sum);
    rewriter.replaceOpWithNewOp<xla_hlo::DivOp>(op.getOperation(), op.getType(),
                                                exp, sum, batch_dims);
    return matchSuccess();
  }
};

#include "tensorflow/compiler/mlir/xla/transforms/generated_legalize_tf.inc"
}  // end anonymous namespace
}  // end namespace xla
}  // end namespace mlir

void mlir::xla_hlo::legalizeTF(Operation *op) {
  // Add lowering patterns to the list.
  OwningRewritePatternList patterns;
  xla::populateWithGenerated(op->getContext(), &patterns);
  patterns.insert<mlir::xla::ConvertMaxPoolOp>(op->getContext());
  patterns.insert<mlir::xla::ConvertSoftmaxOp>(op->getContext());

  // Recursively applies rewrite patterns to nested operations.
  applyPatternsGreedily(op, patterns);
}

/// Performs the lowering to XLA dialect.
void LegalizeTF::runOnFunction() {
  auto func = getFunction();
  mlir::xla_hlo::legalizeTF(func);
}

static PassRegistration<LegalizeTF> pass(
    "xla-legalize-tf", "Legalize from TensorFlow to the XLA dialect");
