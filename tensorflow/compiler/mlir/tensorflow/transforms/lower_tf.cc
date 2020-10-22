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

#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"

#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace TF {
namespace {

// Returns 1D 64-bit dense elements attribute with the given values.
static DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                               Builder *builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

// Returns a 1-d i64 elements attribute populated with numbers from start to
// end, excluding.
static DenseIntElementsAttr GetI64ElementsAttrForSeq(int start, int end,
                                                     Builder *builder) {
  int size = end - start;

  SmallVector<int64_t, 4> vals;
  vals.resize(size);
  std::iota(vals.begin(), vals.end(), start);

  TensorType ty = RankedTensorType::get({size}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, vals);
}

static APFloat ConvertToAPFloat(double val, Type type) {
  if (type.getIntOrFloatBitWidth() == 32) {
    return APFloat(static_cast<float>(val));
  }

  return APFloat(val);
}

// Returns int, float, or complex DenseElementsAttr with scalar shape with the
// given element type and the integer value.
static DenseElementsAttr GetScalarOfType(Type ty, int64_t raw_value) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);
  if (auto float_ty = ty.dyn_cast_or_null<FloatType>()) {
    FloatAttr attr = FloatAttr::get(float_ty, raw_value);
    return DenseElementsAttr::get(scalar_ty, attr);
  } else if (auto int_ty = ty.dyn_cast_or_null<IntegerType>()) {
    IntegerAttr attr = IntegerAttr::get(int_ty, raw_value);
    return DenseElementsAttr::get(scalar_ty, attr);
  } else if (auto complex_ty = ty.dyn_cast_or_null<ComplexType>()) {
    Type complex_element_ty = complex_ty.getElementType();
    if (complex_element_ty.isF32()) {
      return DenseElementsAttr::get(
          scalar_ty, static_cast<std::complex<float>>(raw_value));
    } else if (complex_element_ty.isF64()) {
      return DenseElementsAttr::get(
          scalar_ty, static_cast<std::complex<double>>(raw_value));
    }
  }
  llvm_unreachable("unsupported type");
}

// Returns float DenseElementsAttr with scalar shape with the specified value.
static DenseElementsAttr GetScalarOfFloatType(Type ty, double raw_value) {
  auto float_ty = ty.cast<FloatType>();
  FloatAttr attr = FloatAttr::get(float_ty, raw_value);
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);
  return DenseElementsAttr::get(scalar_ty, attr);
}

// Returns reduction indices to use while lowering tf.BiasAddGrad op to tf.Sum
// op.
DenseIntElementsAttr GetBiasAddGradReductionIndices(int64_t rank,
                                                    StringAttr data_format,
                                                    Builder *builder) {
  tensorflow::TensorFormat format;
  if (!FormatFromString(data_format.getValue().str(), &format)) return {};

  // Reduce along all dimensions except the feature dimension.
  int64_t feature_dim = GetTensorFeatureDimIndex(rank, format);
  llvm::SmallVector<int64_t, 4> dims_to_reduce(rank - 1);
  std::iota(dims_to_reduce.begin(), dims_to_reduce.begin() + feature_dim, 0);
  std::iota(dims_to_reduce.begin() + feature_dim, dims_to_reduce.end(),
            feature_dim + 1);
  return GetI64ElementsAttr(dims_to_reduce, builder);
}

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_lower_tf.inc"

// Infers ExpandDims op output type for the given input type `ty` and dimension
// to expand at the given `axis`.
Type InferExpandDimsType(Type ty, int64_t axis, Builder *builder) {
  auto ranked_ty = ty.dyn_cast<RankedTensorType>();

  // Unranked type.
  if (!ranked_ty) return ty;

  auto shape = llvm::to_vector<4>(ranked_ty.getShape());
  if (axis < 0) axis += ranked_ty.getRank() + 1;

  shape.insert(shape.begin() + axis, 1);
  return RankedTensorType::get(shape, ranked_ty.getElementType());
}

// Converts individual Values to a tensor of rank 1. Each input Value has rank 1
// and size 1.
Value ValuesToRank1(PatternRewriter &rewriter, Location loc, Type dtype,
                    ArrayRef<Value> vals) {
  int64_t length = vals.size();
  auto type = RankedTensorType::get({length}, dtype);
  auto axis = rewriter.create<TF::ConstOp>(
      loc, GetScalarOfType(rewriter.getIntegerType(64), 0));
  return rewriter.create<TF::ConcatV2Op>(loc, type, ValueRange(vals), axis);
}

// Lowers AddN op to a sequence of AddV2 ops to accumulate operands.
//
// Note that to improve the parallelism, AddN op uses tree-based reduction.
// For example, tf.AddN([0, 1, 2, 3, 4]) behaves as follows:
//
//                 0     1     2     3     4
//                 |     |     |     |     |
//                 -------     -------     |
//                    |           |        |
//                    5           6        |
//                    |           |        |
//                    -------------        |
//                          |              |
//                          7              |
//                          |              |
//                          ----------------
//                                 |
//                                 8
//
// Example:
//
//   %result = "tf.AddN"(%0, %1, %2)
//
// is lowered to:
//
//   %sum0 = "tf.AddV2"(%0, %1)
//   %result = "tf.AddV2"(%sum0, %2)
//
// While
//
//   %result = "tf.AddN"(%0, %1, %2, %3, %4)
//
// is lowered to:
//
//   %sum0 = "tf.AddV2"(%0, %1)
//   %sum1 = "tf.AddV2"(%2, %3)
//   %sum2 = "tf.AddV2"(%sum0, %sum1)
//   %result = "tf.AddV2"(%sum2, %4)
//
class LowerAddNOp : public RewritePattern {
 public:
  explicit LowerAddNOp(MLIRContext *context)
      : RewritePattern(TF::AddNOp::getOperationName(),
                       {TF::AddV2Op::getOperationName()}, 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto addn_op = cast<TF::AddNOp>(op);

    // TODO(hinsu): Support variant with TensorList type. tf.AddV2 doesn't
    // support variant type so variant types require special handling.
    if (getElementTypeOrSelf(addn_op.getType()).isa<VariantType>())
      return failure();
    llvm::SmallVector<Value, 4> operands(addn_op.inputs().begin(),
                                         addn_op.inputs().end());

    int64_t n = operands.size();
    // Keep doing tree-based reduction when there are more than one operand.
    while (n > 1) {
      for (int64_t i = 0; i < n; i += 2) {
        // Add two adjacent operands if applicable.
        operands[i / 2] =
            (i + 1 < n) ? rewriter.create<TF::AddV2Op>(
                              addn_op.getLoc(), operands[i], operands[i + 1])
                        : operands[i];
      }
      n = (n + 1) / 2;
    }

    rewriter.replaceOp(addn_op, operands[0]);
    return success();
  }
};

// Lowers DynamicStitch op with constant indices and with static input and
// output shapes using Reshape, UnPack and Pack op.
//
//   %indices0 = "tf.Const"() {value = dense<4> : tensor<i32>}
//   %indices1 = "tf.Const"() {value = dense<[[3, 2], [1, 0]]> :
//   tensor<2x2xi32>} %0 = "tf.DynamicStitch"(%indices0, %indices1, %arg0,
//   %arg1)
//     : (tensor<i32>, tensor<2x2xi32>, tensor<2xf32>, tensor<2x2x2xf32>)
//     -> tensor<5x2xf32>
//
// is lowered to
//
//   %shape = "tf.Const"() {value = dense<[-1, 2]> : tensor<2xi64>}
//   %inp0 = "tf.Reshape"(%arg0, %shape)
//     : (tensor<2xf32>, tensor<2xi64>) -> tensor<1x2xf32>
//   %inp1 = "tf.Reshape"(%arg1, %shape)
//     : (tensor<2x2x2xf32>, tensor<2xi64>) -> tensor<4x2xf32>
//   %items0 = "tf.Unpack"(%[[INP0]]) {axis = 0 : i64}
//     : (tensor<1x2xf32>) -> tensor<2xf32>
//   %items1:4 = "tf.Unpack"(%[[INP1]]) {axis = 0 : i64}
//     : (tensor<4x2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
//     tensor<2xf32>)
//   %axis = "tf.Const"() {value = dense<0> : tensor<i64>}
//   %0 = "tf.Pack"(items1#3, items1#2, items1#1, items1#0, %items0, %axis)
//     : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
//        tensor<2xf32>, tensor<i64>) -> tensor<5x2xf32>
//
class LowerDynamicStitchOp : public OpRewritePattern<TF::DynamicStitchOp> {
 public:
  explicit LowerDynamicStitchOp(MLIRContext *context)
      : OpRewritePattern<TF::DynamicStitchOp>(context) {}

  LogicalResult matchAndRewrite(DynamicStitchOp op,
                                PatternRewriter &rewriter) const override {
    // Static output type is used to compute intermediate values. Note that the
    // output type doesn't have to be static but if input types and indices are
    // constant, then the output type can be statically determined.
    RankedTensorType out_ty = op.getType().dyn_cast<RankedTensorType>();
    if (!out_ty || !out_ty.hasStaticShape()) return failure();

    // Extract out all the constant indices' attributes and verify that data
    // types are static.
    SmallVector<DenseIntElementsAttr, 4> indices;
    indices.reserve(op.N());
    for (auto it : llvm::zip(op.indices(), op.data())) {
      Value index = std::get<0>(it);
      Value data = std::get<1>(it);

      DenseIntElementsAttr index_attr;
      if (!matchPattern(index, m_Constant(&index_attr))) return failure();
      indices.push_back(index_attr);

      RankedTensorType data_ty = data.getType().dyn_cast<RankedTensorType>();
      if (!data_ty || !data_ty.hasStaticShape()) return failure();
    }

    // Compute type of each of the items and shape to use while reshaping inputs
    // so that they can be unpacked to extract out individual items.
    ArrayRef<int64_t> item_shape = out_ty.getShape().drop_front(1);
    auto item_ty = RankedTensorType::get(item_shape, out_ty.getElementType());

    SmallVector<int64_t, 4> packed_shape;
    packed_shape.push_back(-1);
    packed_shape.append(item_shape.begin(), item_shape.end());
    Location loc = op.getLoc();
    auto packed_shape_val = rewriter.create<ConstOp>(
        loc, GetI64ElementsAttr(packed_shape, &rewriter));

    // Prepare each of the output item by unpacking data and then putting it to
    // the specified index.
    SmallVector<Value, 8> values(out_ty.getDimSize(0));
    for (auto it : llvm::zip(indices, op.data())) {
      DenseIntElementsAttr index_attr = std::get<0>(it);
      Value data = std::get<1>(it);

      auto reshaped_data =
          rewriter.create<ReshapeOp>(loc, data, packed_shape_val);
      auto num_items =
          reshaped_data.getType().cast<RankedTensorType>().getShape()[0];
      auto items = rewriter.create<UnpackOp>(
          loc, SmallVector<Type, 4>(num_items, item_ty), reshaped_data,
          /*axis=*/0);
      for (auto index_item : llvm::zip(index_attr, items.getResults())) {
        int64_t output_index = std::get<0>(index_item).getSExtValue();
        Value item = std::get<1>(index_item);
        values[output_index] = item;
      }
    }

    rewriter.replaceOpWithNewOp<PackOp>(op, op.getType(), values);
    return success();
  }
};

// This pass performs a manual conversion with FakeQuant, converting between
// floating point and quantized space. It is designed to reproduce TF's
// implementation, mirroring the previous XLA implementation.
//
// 1. Computing proper quantized bounds. This involves nudging the input bounds.
// 2. Converting the input bounds to quantized space, rounding values.
// 3. Convert back into floating point space.
class ConvertFakeQuantWithMinMaxVarsOp
    : public OpRewritePattern<TF::FakeQuantWithMinMaxVarsOp> {
  using OpRewritePattern<TF::FakeQuantWithMinMaxVarsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::FakeQuantWithMinMaxVarsOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.inputs();
    auto input_ty = input.getType().cast<ShapedType>();
    auto element_ty = input_ty.getElementType();
    auto scalar_ty = RankedTensorType::get({}, element_ty);

    auto num_bits = op.num_bits();
    auto narrow_range = op.narrow_range();
    const double bits_min = narrow_range ? 1 : 0;
    const double bits_max = (1 << num_bits) - 1;

    auto float_min = op.min();
    auto float_max = op.max();

    auto float_diff =
        rewriter.create<TF::SubOp>(op.getLoc(), float_max, float_min);

    // Compute the range when quantized.
    auto quant_min = rewriter.create<TF::ConstOp>(
        op.getLoc(), DenseElementsAttr::get(
                         scalar_ty, ConvertToAPFloat(bits_min, element_ty)));

    auto quant_max = rewriter.create<TF::ConstOp>(
        op.getLoc(), DenseElementsAttr::get(
                         scalar_ty, ConvertToAPFloat(bits_max, element_ty)));

    auto quant_diff = rewriter.create<TF::ConstOp>(
        op.getLoc(),
        DenseElementsAttr::get(
            scalar_ty, ConvertToAPFloat(bits_max - bits_min, element_ty)));

    auto quant_to_float =
        rewriter.create<TF::DivOp>(op.getLoc(), float_diff, quant_diff);

    auto float_to_quant =
        rewriter.create<TF::DivOp>(op.getLoc(), quant_diff, float_diff);

    // During quantization, the quantized min/max values may not line up
    // perfectly with the specified min/max. Nudge them into the right range.
    auto min_scaled =
        rewriter.create<TF::DivOp>(op.getLoc(), float_min, quant_to_float);
    auto min_scaled_sub =
        rewriter.create<TF::SubOp>(op.getLoc(), quant_min, min_scaled);

    auto mid_rounded =
        rewriter.create<TF::RoundOp>(op.getLoc(), scalar_ty, min_scaled_sub);

    auto nudged_zero_point_val = rewriter.create<TF::ClipByValueOp>(
        op.getLoc(), scalar_ty, mid_rounded, quant_min, quant_max);

    auto quant_min_sub = rewriter.create<TF::SubOp>(op.getLoc(), quant_min,
                                                    nudged_zero_point_val);
    auto quant_max_sub = rewriter.create<TF::SubOp>(op.getLoc(), quant_max,
                                                    nudged_zero_point_val);

    auto nudged_float_min =
        rewriter.create<TF::MulOp>(op.getLoc(), quant_min_sub, quant_to_float);

    auto nudged_float_max =
        rewriter.create<TF::MulOp>(op.getLoc(), quant_max_sub, quant_to_float);

    // Now quantize the input value with the approximated min/max values.

    // Move the input value into quantized space
    Value quantized_input = rewriter.create<TF::ClipByValueOp>(
        op.getLoc(), input_ty, input, nudged_float_min, nudged_float_max);

    quantized_input = rewriter.create<TF::SubOp>(
        op.getLoc(), input_ty, quantized_input, nudged_float_min);

    quantized_input = rewriter.create<TF::MulOp>(
        op.getLoc(), input_ty, quantized_input, float_to_quant);

    // Round the quantized input always to the positive direction.
    auto half_val = rewriter.create<TF::ConstOp>(
        op.getLoc(),
        DenseElementsAttr::get(scalar_ty, ConvertToAPFloat(0.5, element_ty)));

    quantized_input = rewriter.create<TF::AddOp>(op.getLoc(), input_ty,
                                                 quantized_input, half_val);

    quantized_input =
        rewriter.create<TF::FloorOp>(op.getLoc(), quantized_input);

    // Convert back into floating point spae.
    Value output = rewriter.create<TF::MulOp>(op.getLoc(), input_ty,
                                              quantized_input, quant_to_float);

    output = rewriter.create<TF::AddOp>(op.getLoc(), input_ty, output,
                                        nudged_float_min);

    rewriter.replaceOp(op, {output});
    return success();
  }
};

// Lowers InvertPermutation op to TensorScatterUpdate op.
//
// Example:
//
//   %x = "tf.Const"() {value = dense<[3, 4, 0, 1, 2]> : tensor<5xi32>}
//   "tf.InvertPermutation"(%x) : (tensor<5xi32>) -> tensor<5xi32>
//
// is lowered to
//
//   %x = "tf.Const"() {value = dense<[3, 4, 0, 1, 2]> : tensor<5xi32>}
//   %start = "tf.Const"() {value = dense<0> : tensor<i32>}
//   %limit = "tf.Const"() {value = dense<5> : tensor<i32>}
//   %delta = "tf.Const"() {value = dense<1> : tensor<i32>}
//   %updates = "tf.Range"(%start, %limit, %delta) :
//     (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5xi32>
//   %shape = "tf.Const"() {value = dense<[5, 1]> : tensor<2xi32>}
//   %indices = "tf.Reshape"(%x, %shape) : (tensor<5xi32, tensor<2xi32) ->
//     tensor<5x1xi32>
//   "tf.TensorScatterUpdate"(%x, %indices, %updates) :
//     (tensor<5xi32>, tensor<5x1xi32>, tensor<5xi32>) -> tensor<5xi32>
//
class LowerInvertPermutationOp
    : public OpRewritePattern<TF::InvertPermutationOp> {
 public:
  explicit LowerInvertPermutationOp(MLIRContext *context)
      : OpRewritePattern<TF::InvertPermutationOp>(context) {}

  LogicalResult matchAndRewrite(TF::InvertPermutationOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto x_type = op.x().getType().dyn_cast<RankedTensorType>();
    // x input must have static shape.
    if (!x_type || !x_type.hasStaticShape()) {
      return failure();
    }
    Type int_type = x_type.getElementType();  // Could be i32 or i64.

    auto result_type = x_type;
    auto start =
        rewriter.create<TF::ConstOp>(loc, GetScalarOfType(int_type, 0));
    Value limit = rewriter.create<TF::ConstOp>(
        loc, GetScalarOfType(int_type, x_type.getShape()[0]));
    auto delta =
        rewriter.create<TF::ConstOp>(loc, GetScalarOfType(int_type, 1));
    // Construct a sequence of numbers [0, 1, ... len(x)-1].
    auto updates =
        rewriter.create<TF::RangeOp>(loc, result_type, start, limit, delta);

    auto shape_type = RankedTensorType::get({2}, rewriter.getIntegerType(32));
    auto shape = rewriter.create<TF::ConstOp>(
        loc, DenseElementsAttr::get(
                 shape_type, {static_cast<int>(x_type.getDimSize(0)), 1}));
    auto indices = rewriter.create<TF::ReshapeOp>(loc, op.x(), shape);

    rewriter.replaceOpWithNewOp<TF::TensorScatterUpdateOp>(
        op, result_type, op.x(), indices, updates);
    return success();
  }
};

// Approximates lgamma using Lanczos' approximation from
// "A Precision Approximation of the Gamma Function". SIAM Journal on Numerical
// Analysis series B. Vol. 1:
// lgamma(z + 1) = (log(2) + log(pi)) / 2 + (z + 1/2) * log(t(z)) - t(z) + A(z)
// t(z) = z + kLanczosGamma + 1/2
// A(z) = kBaseLanczosCoeff
//       + sigma(k = 1, n, kLanczosCoefficients[i] / (z +  k))
//
// Coefficients for the Lanczos approximation of the gamma function. The
// coefficients are uniquely determined by the choice of g and n
// (kLanczosGamma and kLanczosCoefficients.size() + 1). The coefficients below
// correspond to [7, 9]. [5, 7], [7, 9], [9, 10], and [607/128.0, 15] were
// evaluated and [7, 9] seemed to be the least sensitive to the quality of the
// log function. In particular, [5, 7] is the only choice where -1.5e-5 <=
// lgamma(2) <= 1.5e-5 for a particularly inaccurate log function.
static constexpr double kLanczosGamma = 7;  // aka g
static constexpr double kBaseLanczosCoeff = 0.99999999999980993227684700473478;
static constexpr std::array<double, 8> kLanczosCoefficients = {
    676.520368121885098567009190444019, -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,   -176.61502916214059906584551354,
    12.507343278686904814458936853,     -0.13857109526572011689554707,
    9.984369578019570859563e-6,         1.50563273514931155834e-7};

class LowerLgammaOp : public OpRewritePattern<TF::LgammaOp> {
 public:
  using OpRewritePattern<TF::LgammaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::LgammaOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.x();
    TensorType original_tensor_type = op.x().getType().cast<TensorType>();

    // The approximation is not precise enough for float16. Do the computation
    // in float32 for that case.
    TensorType tensor_type = original_tensor_type;
    FloatType float_type = tensor_type.getElementType().cast<FloatType>();
    bool needs_cast = float_type.getWidth() < 32;
    if (needs_cast) {
      MLIRContext *context = rewriter.getContext();
      float_type = FloatType::getF32(context);
      if (original_tensor_type.hasRank()) {
        tensor_type =
            RankedTensorType::get(original_tensor_type.getShape(), float_type);
      } else {
        tensor_type = UnrankedTensorType::get(float_type);
      }
      input = rewriter.create<TF::CastOp>(loc, tensor_type, input);
    }

    // Helper lambda function for creating a ConstOp for a tensor filled with
    // the given constant float value.
    auto create_const_op = [&rewriter, loc, tensor_type,
                            float_type](double value) {
      return rewriter.create<TF::ConstOp>(
          loc, DenseElementsAttr::get(tensor_type,
                                      FloatAttr::get(float_type, value)));
    };

    Value one_half = create_const_op(0.5);
    Value one = create_const_op(1.0);
    Value infinity = create_const_op(std::numeric_limits<double>::infinity());
    Value pi = create_const_op(M_PI);
    Value log_pi = create_const_op(std::log(M_PI));
    Value log_sqrt_two_pi = create_const_op((std::log(2) + std::log(M_PI)) / 2);
    Value lanczos_gamma_plus_one_half = create_const_op(kLanczosGamma + 0.5);
    Value log_lanczos_gamma_plus_one_half =
        create_const_op(std::log(kLanczosGamma + 0.5));
    Value base_lanczos_coeff = create_const_op(kBaseLanczosCoeff);

    Value minus_input = rewriter.create<TF::NegOp>(loc, input);
    Value input_minus_one = rewriter.create<TF::SubOp>(loc, input, one);

    // If the input is less than 0.5 use Euler's reflection formula:
    // gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
    Value need_to_reflect = rewriter.create<TF::LessOp>(loc, input, one_half);
    Type tensor_bool_type = need_to_reflect.getType();
    Value z = rewriter.create<TF::SelectV2Op>(loc, need_to_reflect, minus_input,
                                              input_minus_one);

    Value x = base_lanczos_coeff;
    for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
      Value lanczos_coefficient = create_const_op(kLanczosCoefficients[i]);
      Value index = create_const_op(static_cast<double>(i));
      Value z_plus_index = rewriter.create<TF::AddV2Op>(loc, z, index);
      Value z_plus_index_plus_one =
          rewriter.create<TF::AddV2Op>(loc, z_plus_index, one);
      Value incr = rewriter.create<TF::DivOp>(loc, lanczos_coefficient,
                                              z_plus_index_plus_one);
      x = rewriter.create<TF::AddV2Op>(loc, x, incr);
    }

    // To improve accuracy on platforms with less-precise log implementations,
    // compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
    // the device.
    // log(t) = log(kLanczosGamma + 0.5 + z)
    //        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
    Value t = rewriter.create<TF::AddV2Op>(loc, lanczos_gamma_plus_one_half, z);
    Value z_div_lanczos_gamma_plus_one_half =
        rewriter.create<TF::DivOp>(loc, z, lanczos_gamma_plus_one_half);
    Value log1p_z_div_lanczos_gamma_plus_one_half =
        rewriter.create<TF::Log1pOp>(loc, z_div_lanczos_gamma_plus_one_half);
    Value log_t =
        rewriter.create<TF::AddV2Op>(loc, log_lanczos_gamma_plus_one_half,
                                     log1p_z_div_lanczos_gamma_plus_one_half);

    // Compute the final result (modulo reflection).  t(z) may be large, and we
    // need to be careful not to overflow to infinity in the first term of
    //
    //   (z + 1/2) * log(t(z)) - t(z).
    //
    // Therefore we compute this as
    //
    //   (z + 1/2 - t(z) / log(t(z))) * log(t(z)).
    //
    // log_y = log_sqrt_two_pi + (z + one_half - t / log_t) * log_t + Log(x);
    Value t_div_log_t = rewriter.create<TF::DivOp>(loc, t, log_t);
    Value one_half_minus_t_div_log_t =
        rewriter.create<TF::SubOp>(loc, one_half, t_div_log_t);
    Value z_plus_one_half_minus_t_div_log_t =
        rewriter.create<TF::AddV2Op>(loc, z, one_half_minus_t_div_log_t);
    Value z_plus_one_half_minus_t_div_log_t_mul_log_t =
        rewriter.create<TF::MulOp>(loc, z_plus_one_half_minus_t_div_log_t,
                                   log_t);
    Value log_x = rewriter.create<TF::LogOp>(loc, x);
    Value log_y_rhs = rewriter.create<TF::AddV2Op>(
        loc, z_plus_one_half_minus_t_div_log_t_mul_log_t, log_x);
    Value log_y = rewriter.create<TF::AddV2Op>(loc, log_sqrt_two_pi, log_y_rhs);

    // Compute the reflected value, used when x < 0.5:
    //
    //   lgamma(x) = log(pi) - lgamma(1-x) - log(abs(sin(pi * x))).
    //
    // (The abs is because lgamma is the log of the absolute value of the gamma
    // function.)
    //
    // We have to be careful when computing the final term above. gamma(x) goes
    // to +/-inf at every integer x < 0, and this is controlled by the
    // sin(pi * x) term.  The slope is large, so precision is particularly
    // important.
    //
    // Because abs(sin(pi * x)) has period 1, we can equivalently use
    // abs(sin(pi * frac(x))), where frac(x) is the fractional part of x.  This
    // is more numerically accurate: It doesn't overflow to inf like pi * x can,
    // and if x is an integer, it evaluates to 0 exactly, which is significant
    // because we then take the log of this value, and log(0) is inf.
    //
    // We don't have a frac(x) primitive in XLA and computing it is tricky, but
    // because abs(sin(pi * x)) = abs(sin(pi * abs(x))), it's good enough for
    // our purposes to use abs(frac(x)) = abs(x) - floor(abs(x)).
    //
    // Furthermore, pi * abs(frac(x)) loses precision when abs(frac(x)) is close
    // to 1.  To remedy this, we can use the fact that sin(pi * x) in the domain
    // [0, 1] is symmetric across the line Y=0.5.
    Value abs_input = rewriter.create<TF::AbsOp>(loc, input);
    Value abs_input_floor = rewriter.create<TF::FloorOp>(loc, abs_input);
    Value abs_frac_input =
        rewriter.create<TF::SubOp>(loc, abs_input, abs_input_floor);

    // Convert values of abs_frac_input > 0.5 to (1 - frac_input) to improve
    // precision of pi * abs_frac_input for values of abs_frac_input close to 1.
    Value one_minus_abs_frac_input =
        rewriter.create<TF::SubOp>(loc, one, abs_frac_input);
    Value abs_frac_input_gt_one_half =
        rewriter.create<TF::GreaterOp>(loc, abs_frac_input, one_half);
    Value reduced_frac_input = rewriter.create<TF::SelectV2Op>(
        loc, abs_frac_input_gt_one_half, one_minus_abs_frac_input,
        abs_frac_input);
    Value pi_mul_reduced_frac_input =
        rewriter.create<TF::MulOp>(loc, pi, reduced_frac_input);
    Value sin_pi_mul_reduced_frac_input =
        rewriter.create<TF::SinOp>(loc, pi_mul_reduced_frac_input);
    Value reflection_denom =
        rewriter.create<TF::LogOp>(loc, sin_pi_mul_reduced_frac_input);

    // Avoid computing -inf - inf, which is nan.  If reflection_denom is +/-inf,
    // then it "wins" and the result is +/-inf.
    Value is_finite = rewriter.create<TF::IsFiniteOp>(loc, tensor_bool_type,
                                                      reflection_denom);
    Value neg_reflection_denom =
        rewriter.create<TF::NegOp>(loc, reflection_denom);
    Value log_pi_minus_reflection_denom =
        rewriter.create<TF::SubOp>(loc, log_pi, reflection_denom);
    Value reflection_if_finite =
        rewriter.create<TF::SubOp>(loc, log_pi_minus_reflection_denom, log_y);
    Value reflection = rewriter.create<TF::SelectV2Op>(
        loc, is_finite, reflection_if_finite, neg_reflection_denom);

    Value result = rewriter.create<TF::SelectV2Op>(loc, need_to_reflect,
                                                   reflection, log_y);

    // lgamma(+/-inf) = +inf.
    Value is_inf = rewriter.create<TF::IsInfOp>(loc, tensor_bool_type, input);
    result = rewriter.create<SelectV2Op>(loc, is_inf, infinity, result);

    if (needs_cast) {
      result = rewriter.create<TF::CastOp>(loc, original_tensor_type, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lowers Pack op to ConcatV2 op after changing shape of the inputs with
// ExpandDims op.
//
// Sample result with 2 inputs to pack:
//
//   %axis = "tf.Const"() {value = dense<1> : tensor<i64>}
//   %inp0 = "tf.ExpandDims"(%operand0, %axis): tensor<2xf32> -> tensor<2x1xf32>
//   %inp1 = "tf.ExpandDims"(%operand1, %axis): tensor<2xf32> -> tensor<2x1xf32>
//   %result = "tf.ConcatV2"(%operand0, %operand1, %axis) { N = 2 : i64 }:
//
class LowerPackOp : public OpRewritePattern<TF::PackOp> {
 public:
  explicit LowerPackOp(MLIRContext *context)
      : OpRewritePattern<TF::PackOp>(context) {}

  LogicalResult matchAndRewrite(TF::PackOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto axis_value = rewriter.create<TF::ConstOp>(
        loc,
        DenseElementsAttr::get(
            RankedTensorType::get({}, rewriter.getIntegerType(64)), op.axis()));
    int64_t axis = op.axis();

    Type prev_input_ty, inferred_ty;
    SmallVector<Value, 4> expanded_inputs;
    expanded_inputs.reserve(op.N());
    for (Value input : op.values()) {
      // If input type is different than the previous input type, infer the
      // output type. Otherwise, use the already inferred output type from the
      // previous iteration.
      Type input_ty = input.getType();
      if (input_ty != prev_input_ty) {
        inferred_ty = InferExpandDimsType(input_ty, axis, &rewriter);
        prev_input_ty = input_ty;
      }
      expanded_inputs.push_back(rewriter.create<TF::ExpandDimsOp>(
          loc, inferred_ty, input, axis_value));
    }

    rewriter.replaceOpWithNewOp<TF::ConcatV2Op>(op, op.getType(),
                                                expanded_inputs, axis_value);
    return success();
  }
};

// Lowers SpaceToBatchND by reducing to reshape(transpose(reshape(pad(input)))).
//
// Before rewrite:
//   output = SpaceToBatchND(input, block_shape, paddings)
// Let:
//   [batch] + spatial_shape + remaining_shape = input.shape
//   M = spatial_shape.rank
// After rewrite:
//   padded = zero-pad input with paddings
//     The spatial_shape component of input.shape pads with paddings[*, 0]
//     before each dimension, and paddings[*, 1] after each dimension.
//   reshaped = reshape padded to:
//     [batch]
//     + [padded.shape[1]/block_shape[0], block_shape[0], ...,
//        padded.shape[M]/block_shape[M-1], block_shape[M-1]]
//     + remaining_shape
//   permuted = transpose reshaped to:
//     block_shape
//     + [batch]
//     + [padded.shape[1]/block_shape[0], ..., padded.shape[M]/block_shape[M-1]]
//     + remaining_shape
//   result = reshape permuted to:
//     [batch * product(block_shape)]
//     + [padded.shape[1]/block_shape[0], ..., padded.shape[M]/block_shape[M-1]]
//     + remaining_shape
class LowerSpaceToBatchNDOp : public OpRewritePattern<TF::SpaceToBatchNDOp> {
 public:
  using OpRewritePattern<TF::SpaceToBatchNDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SpaceToBatchNDOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto input_type = op.input().getType().cast<TensorType>();
    if (!input_type.hasStaticShape()) {
      return failure();
    }
    ArrayRef<int64_t> input_shape = input_type.getShape();
    auto block_shape_type = op.block_shape().getType().cast<TensorType>();
    if (!block_shape_type.hasStaticShape()) {
      return failure();
    }
    auto paddings_type = op.paddings().getType().cast<ShapedType>();

    int64_t input_rank = input_type.getRank();
    int64_t block_rank = block_shape_type.getNumElements();
    int64_t remaining_rank = input_rank - 1 - block_rank;
    if (remaining_rank < 0) {
      // TODO(b/157475606): Move this check to ::Verify
      return failure();
    }

    auto block_shape_i64_type = RankedTensorType::get(
        block_shape_type.getShape(), rewriter.getIntegerType(64));
    auto block_shape_i64 = rewriter.create<TF::CastOp>(
        loc, block_shape_i64_type, op.block_shape());

    auto paddings_i64_type = RankedTensorType::get(paddings_type.getShape(),
                                                   rewriter.getIntegerType(64));
    auto paddings_i64 =
        rewriter.create<TF::CastOp>(loc, paddings_i64_type, op.paddings());

    auto pad00 = rewriter.create<TF::ConstOp>(
        loc, DenseElementsAttr::get<int64_t>(
                 RankedTensorType::get({1, 2}, rewriter.getIntegerType(64)),
                 {0, 0}));
    SmallVector<Value, 4> full_paddings_list{pad00, paddings_i64};
    full_paddings_list.append(remaining_rank, pad00);
    auto full_paddings_type =
        RankedTensorType::get({input_rank, 2}, rewriter.getIntegerType(64));
    auto zero_i64 = rewriter.create<TF::ConstOp>(
        loc, GetScalarOfType(rewriter.getIntegerType(64), 0));
    // Extends paddings to all dimensions of input by adding 0s to non-block
    // dimensions.
    auto full_paddings = rewriter.create<TF::ConcatV2Op>(
        loc, full_paddings_type, full_paddings_list, zero_i64);

    SmallVector<int64_t, 4> padded_shape(input_rank, ShapedType::kDynamicSize);
    auto padded_type =
        RankedTensorType::get(padded_shape, rewriter.getF32Type());
    // padded = pad(input, full_paddings)
    auto padded =
        rewriter.create<TF::PadOp>(loc, padded_type, op.input(), full_paddings);

    auto paddings_sum_type =
        RankedTensorType::get({input_rank}, rewriter.getIntegerType(64));
    auto one_i64 = rewriter.create<TF::ConstOp>(
        loc, GetScalarOfType(rewriter.getIntegerType(64), 1));
    // paddings_sum = paddings[*,0] + paddings[*,1]
    auto paddings_sum = rewriter.create<TF::SumOp>(loc, paddings_sum_type,
                                                   full_paddings, one_i64);

    // input_shape_tensor = input.shape
    auto input_shape_tensor = rewriter.create<TF::ConstOp>(
        loc,
        DenseElementsAttr::get(
            RankedTensorType::get({input_rank}, rewriter.getIntegerType(64)),
            input_shape));

    // padded_shape_tensor is the shape of padded.
    auto padded_shape_tensor =
        rewriter.create<TF::AddOp>(loc, paddings_sum, input_shape_tensor);

    auto zero_i32 = rewriter.create<TF::ConstOp>(
        loc, GetScalarOfType(rewriter.getIntegerType(32), 0));
    SmallVector<Type, 4> padded_shape_splits_types(
        input_rank, RankedTensorType::get({1}, rewriter.getIntegerType(64)));
    SmallVector<Value, 4> padded_shape_splits(
        rewriter
            .create<TF::SplitOp>(loc, padded_shape_splits_types, zero_i32,
                                 padded_shape_tensor)
            .output());

    SmallVector<Type, 4> block_shape_splits_types(
        block_rank, RankedTensorType::get({1}, rewriter.getIntegerType(64)));
    SmallVector<Value, 4> block_shape_splits(
        rewriter
            .create<TF::SplitOp>(loc, block_shape_splits_types, zero_i32,
                                 block_shape_i64)
            .output());

    SmallVector<Value, 4> outer_shape_vals;
    for (int64_t i = 0; i < block_rank; ++i) {
      // TODO(b/157475606): Insert tf.Assert that the following division has
      // remainder 0.
      outer_shape_vals.push_back(rewriter.create<TF::DivOp>(
          loc, padded_shape_splits[1 + i], block_shape_splits[i]));
    }

    SmallVector<Value, 6> reshaped_shape_vals{padded_shape_splits[0]};
    for (int64_t i = 0; i < block_rank; ++i) {
      reshaped_shape_vals.push_back(outer_shape_vals[i]);
      reshaped_shape_vals.push_back(block_shape_splits[i]);
    }
    for (int64_t i = 1 + block_rank; i < input_rank; ++i) {
      reshaped_shape_vals.push_back(padded_shape_splits[i]);
    }
    auto reshaped_shape = ValuesToRank1(
        rewriter, loc, rewriter.getIntegerType(64), reshaped_shape_vals);

    SmallVector<int64_t, 6> permutation_vals;
    for (int64_t i = 0; i < block_rank; ++i) {
      permutation_vals.push_back(2 + 2 * i);
    }
    permutation_vals.push_back(0);
    for (int64_t i = 0; i < block_rank; ++i) {
      permutation_vals.push_back(1 + 2 * i);
    }
    for (int64_t i = 1 + block_rank; i < input_rank; ++i) {
      permutation_vals.push_back(block_rank + i);
    }
    auto permutation = rewriter.create<TF::ConstOp>(
        loc, GetI64ElementsAttr(permutation_vals, &rewriter));

    auto output_batch = padded_shape_splits[0];
    for (int64_t i = 0; i < block_rank; ++i) {
      output_batch =
          rewriter.create<TF::MulOp>(loc, output_batch, block_shape_splits[i]);
    }
    SmallVector<Value, 4> output_shape_vals{output_batch};
    for (int64_t i = 0; i < block_rank; ++i) {
      output_shape_vals.push_back(outer_shape_vals[i]);
    }
    for (int64_t i = 1 + block_rank; i < input_rank; ++i) {
      output_shape_vals.push_back(padded_shape_splits[i]);
    }
    auto output_shape = ValuesToRank1(
        rewriter, loc, rewriter.getIntegerType(64), output_shape_vals);
    auto reshaped = rewriter.create<TF::ReshapeOp>(loc, padded, reshaped_shape);
    auto permuted =
        rewriter.create<TF::TransposeOp>(loc, reshaped, permutation);

    // Sometimes the result type is more specific than what the reshape builder
    // can infer.
    auto result_type = op.getResult().getType();
    rewriter.replaceOpWithNewOp<TF::ReshapeOp>(op, result_type, permuted,
                                               output_shape);

    return success();
  }
};

// Lowers `TF::SparseMatMulOp` to `TF::MatMulOp`, ignoring the sparseness hints,
// since we currently don't have an implementation that can use this
// information. Adds appropriate casts where necessary to align element types
// of operands and result for `TF::MatMulOp`.
class LowerSparseMatMulOp : public OpRewritePattern<TF::SparseMatMulOp> {
 public:
  using OpRewritePattern<TF::SparseMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SparseMatMulOp op,
                                PatternRewriter &rewriter) const override {
    // Result type must be f32 for applying the pattern (currently this is
    // required by the op anyway but this might change).
    if (!op.product().getType().cast<TensorType>().getElementType().isF32()) {
      return failure();
    }
    MLIRContext *context = rewriter.getContext();
    llvm::SmallVector<Value, 2> operands{op.a(), op.b()};
    for (Value &operand : operands) {
      TensorType tensor_type = operand.getType().cast<TensorType>();
      Type element_type = tensor_type.getElementType();
      if (element_type.isF32()) continue;
      // Element type can either be f32 or bf16 for `SparseMatMulOp` so it
      // must be bf16 here.
      assert(element_type.isBF16());
      Type tensor_type_f32;
      if (tensor_type.hasRank()) {
        tensor_type_f32 = RankedTensorType::get(tensor_type.getShape(),
                                                FloatType::getF32(context));
      } else {
        tensor_type_f32 = UnrankedTensorType::get(FloatType::getF32(context));
      }
      // Add cast to f32 to conform with element type of result.
      operand =
          rewriter.create<TF::CastOp>(op.getLoc(), tensor_type_f32, operand);
    }
    Value result = rewriter.create<TF::MatMulOp>(
        op.getLoc(), op.product().getType(), operands[0], operands[1],
        op.transpose_a(), op.transpose_b());

    rewriter.replaceOp(op, {result});
    return success();
  }
};

// Lowers _UnaryOpsComposition op as a series of original TensorFlow ops that
// were fused together.
class Lower_UnaryOpsComposition
    : public OpRewritePattern<TF::_UnaryOpsCompositionOp> {
 public:
  using OpRewritePattern<TF::_UnaryOpsCompositionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::_UnaryOpsCompositionOp op,
                                PatternRewriter &rewriter) const override {
    Value result = op.x();
    for (StringRef op_name : op.op_names().getAsValueRange<StringAttr>()) {
      std::string full_name = "tf." + op_name.str();
      // All ops in the sequences have the same result type as the original
      // result type.
      OperationState state(op.getLoc(), full_name, /*operands=*/{result},
                           /*types=*/{op.getType()}, /*attributes=*/{});
      Operation *op = rewriter.createOperation(state);
      result = op->getResult(0);
    }
    rewriter.replaceOp(op, {result});
    return success();
  }
};

}  // namespace

void PopulateLoweringTFPatterns(MLIRContext *context,
                                OwningRewritePatternList *patterns) {
  patterns->insert<LowerAddNOp, ConvertFakeQuantWithMinMaxVarsOp,
                   LowerDynamicStitchOp, LowerInvertPermutationOp,
                   LowerLgammaOp, LowerPackOp, LowerSpaceToBatchNDOp,
                   LowerSparseMatMulOp, Lower_UnaryOpsComposition>(context);
  populateWithGenerated(context, *patterns);
}

}  // namespace TF
}  // namespace mlir
