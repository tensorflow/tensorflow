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
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
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

// Return an Attr representation of the value.
static DenseElementsAttr GetF32Scalar(OpBuilder *builder, float value) {
  return DenseElementsAttr::get(
      RankedTensorType::get({}, builder->getF32Type()),
      FloatAttr::get(builder->getF32Type(), value));
}

// Returns a TF_CastOp to F32. This function is used for CastOps that are
// intermediate nodes in a TableGen pattern result. In such a case, the
// destination type is not inferred and must be given explicitly.
//
// Preconditions: The given value must have a ShapedType.
static Value CreateTFCastOpF32(OpBuilder *builder, Location loc, Value x,
                               BoolAttr truncate) {
  auto x_type = x.getType().dyn_cast_or_null<ShapedType>();
  if (!x_type) llvm_unreachable("unsupported type");
  Type type = x_type.clone(builder->getF32Type());
  return builder->create<CastOp>(loc, type, x, truncate);
}

static APFloat ConvertToAPFloat(double val, Type type) {
  if (type.getIntOrFloatBitWidth() == 32) {
    return APFloat(static_cast<float>(val));
  }

  return APFloat(val);
}

// Returns int, float, or complex DenseElementsAttr with scalar shape with the
// given element type and the value.
template <typename T>
static DenseElementsAttr GetScalarOfType(Type ty, T raw_value) {
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

// Return true if the passed quantized type is unsigned.
bool QuantizedTypeIsUnsigned(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<mlir::TF::Qint8Type>([](Type) { return false; })
      .Case<mlir::TF::Qint16Type>([](Type) { return false; })
      .Case<mlir::TF::Qint32Type>([](Type) { return false; })
      .Case<mlir::TF::Quint8Type>([](Type) { return true; })
      .Case<mlir::TF::Quint16Type>([](Type) { return true; })
      .Default([](Type) {
        llvm_unreachable("QuantizedTypeIsUnsigned: not a quantized type");
        return false;
      });
}

// Return the half_range value that is used by DequantizeOp. half_range is used
// to offset the quantized representation before it gets scaled. In the case
// of negative quantize types, this offset is half the type's range.
static DenseElementsAttr DequantizeHalfRange(OpBuilder *builder, Value input) {
  auto input_type = input.getType().dyn_cast_or_null<ShapedType>();
  if (!input_type) llvm_unreachable("DequantizeHalfRange: not a ShapedType");
  bool is_unsigned = QuantizedTypeIsUnsigned(input_type.getElementType());
  float half_range = is_unsigned ? 0 : 128;
  return GetScalarOfType(builder->getF32Type(), half_range);
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
  auto axis = rewriter.create<ConstOp>(
      loc, GetScalarOfType(rewriter.getIntegerType(64), 0));
  return rewriter.create<ConcatV2Op>(loc, type, ValueRange(vals), axis);
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
      : RewritePattern(AddNOp::getOperationName(), 1, context,
                       {AddV2Op::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto addn_op = cast<AddNOp>(op);

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
            (i + 1 < n) ? rewriter.create<AddV2Op>(addn_op.getLoc(),
                                                   operands[i], operands[i + 1])
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
template <typename OpT>
class LowerDynamicStitchOp : public RewritePattern {
 public:
  explicit LowerDynamicStitchOp(MLIRContext *context)
      : RewritePattern(
            OpT::getOperationName(), 1, context,
            {ConstOp::getOperationName(), ReshapeOp::getOperationName(),
             UnpackOp::getOperationName(), PackOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *src_op,
                                PatternRewriter &rewriter) const override {
    auto op = cast<OpT>(src_op);

    // Static output type is used to compute intermediate values. Note that the
    // output type doesn't have to be static but if input types and indices are
    // constant, then the output type can be statically determined.
    RankedTensorType out_ty =
        op.getType().template dyn_cast<RankedTensorType>();
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

      RankedTensorType data_ty =
          data.getType().template dyn_cast<RankedTensorType>();
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
      auto num_items = reshaped_data.getType()
                           .template cast<RankedTensorType>()
                           .getShape()[0];
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
class ConvertFakeQuantWithMinMaxVarsOp : public RewritePattern {
 public:
  explicit ConvertFakeQuantWithMinMaxVarsOp(MLIRContext *context)
      : RewritePattern(
            FakeQuantWithMinMaxVarsOp::getOperationName(), 1, context,
            {AddV2Op::getOperationName(), SubOp::getOperationName(),
             ConstOp::getOperationName(), MulOp::getOperationName(),
             FloorOp::getOperationName(), ClipByValueOp::getOperationName(),
             DivOp::getOperationName(), RoundOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *src_op,
                                PatternRewriter &rewriter) const override {
    auto op = cast<FakeQuantWithMinMaxVarsOp>(src_op);

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

    auto float_diff = rewriter.create<SubOp>(op.getLoc(), float_max, float_min);

    // Compute the range when quantized.
    auto quant_min = rewriter.create<ConstOp>(
        op.getLoc(), DenseElementsAttr::get(
                         scalar_ty, ConvertToAPFloat(bits_min, element_ty)));

    auto quant_max = rewriter.create<ConstOp>(
        op.getLoc(), DenseElementsAttr::get(
                         scalar_ty, ConvertToAPFloat(bits_max, element_ty)));

    auto quant_diff = rewriter.create<ConstOp>(
        op.getLoc(),
        DenseElementsAttr::get(
            scalar_ty, ConvertToAPFloat(bits_max - bits_min, element_ty)));

    auto quant_to_float =
        rewriter.create<DivOp>(op.getLoc(), float_diff, quant_diff);

    auto float_to_quant =
        rewriter.create<DivOp>(op.getLoc(), quant_diff, float_diff);

    // During quantization, the quantized min/max values may not line up
    // perfectly with the specified min/max. Nudge them into the right range.
    auto min_scaled =
        rewriter.create<DivOp>(op.getLoc(), float_min, quant_to_float);
    auto min_scaled_sub =
        rewriter.create<SubOp>(op.getLoc(), quant_min, min_scaled);

    auto mid_rounded =
        rewriter.create<RoundOp>(op.getLoc(), scalar_ty, min_scaled_sub);

    auto nudged_zero_point_val = rewriter.create<ClipByValueOp>(
        op.getLoc(), scalar_ty, mid_rounded, quant_min, quant_max);

    auto quant_min_sub =
        rewriter.create<SubOp>(op.getLoc(), quant_min, nudged_zero_point_val);
    auto quant_max_sub =
        rewriter.create<SubOp>(op.getLoc(), quant_max, nudged_zero_point_val);

    auto nudged_float_min =
        rewriter.create<MulOp>(op.getLoc(), quant_min_sub, quant_to_float);

    auto nudged_float_max =
        rewriter.create<MulOp>(op.getLoc(), quant_max_sub, quant_to_float);

    // Now quantize the input value with the approximated min/max values.

    // Move the input value into quantized space
    Value quantized_input = rewriter.create<ClipByValueOp>(
        op.getLoc(), input_ty, input, nudged_float_min, nudged_float_max);

    quantized_input = rewriter.create<SubOp>(op.getLoc(), input_ty,
                                             quantized_input, nudged_float_min);

    quantized_input = rewriter.create<MulOp>(op.getLoc(), input_ty,
                                             quantized_input, float_to_quant);

    // Round the quantized input always to the positive direction.
    auto half_val = rewriter.create<ConstOp>(
        op.getLoc(),
        DenseElementsAttr::get(scalar_ty, ConvertToAPFloat(0.5, element_ty)));

    quantized_input = rewriter.create<AddV2Op>(op.getLoc(), input_ty,
                                               quantized_input, half_val);

    quantized_input = rewriter.create<FloorOp>(op.getLoc(), quantized_input);

    // Convert back into floating point spae.
    Value output = rewriter.create<MulOp>(op.getLoc(), input_ty,
                                          quantized_input, quant_to_float);

    output = rewriter.create<AddV2Op>(op.getLoc(), input_ty, output,
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
class LowerInvertPermutationOp : public RewritePattern {
 public:
  explicit LowerInvertPermutationOp(MLIRContext *context)
      : RewritePattern(
            InvertPermutationOp::getOperationName(), 1, context,
            {ConstOp::getOperationName(), RangeOp::getOperationName(),
             ReshapeOp::getOperationName(),
             TensorScatterUpdateOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *src_op,
                                PatternRewriter &rewriter) const override {
    auto op = cast<InvertPermutationOp>(src_op);

    Location loc = op.getLoc();
    auto x_type = op.x().getType().dyn_cast<RankedTensorType>();
    // x input must have static shape.
    if (!x_type || !x_type.hasStaticShape()) {
      return failure();
    }
    Type int_type = x_type.getElementType();  // Could be i32 or i64.

    auto result_type = x_type;
    auto start = rewriter.create<ConstOp>(loc, GetScalarOfType(int_type, 0));
    Value limit = rewriter.create<ConstOp>(
        loc, GetScalarOfType(int_type, x_type.getShape()[0]));
    auto delta = rewriter.create<ConstOp>(loc, GetScalarOfType(int_type, 1));
    // Construct a sequence of numbers [0, 1, ... len(x)-1].
    auto updates =
        rewriter.create<RangeOp>(loc, result_type, start, limit, delta);

    auto shape_type = RankedTensorType::get({2}, rewriter.getIntegerType(32));
    auto shape = rewriter.create<ConstOp>(
        loc, DenseElementsAttr::get(
                 shape_type, {static_cast<int>(x_type.getDimSize(0)), 1}));
    auto indices = rewriter.create<ReshapeOp>(loc, op.x(), shape);

    rewriter.replaceOpWithNewOp<TensorScatterUpdateOp>(op, result_type, op.x(),
                                                       indices, updates);
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

class LowerLgammaOp : public RewritePattern {
 public:
  explicit LowerLgammaOp(MLIRContext *context)
      : RewritePattern(LgammaOp::getOperationName(), 1, context,
                       {
                           CastOp::getOperationName(),
                           ConstOp::getOperationName(),
                           NegOp::getOperationName(),
                           SubOp::getOperationName(),
                           SelectV2Op::getOperationName(),
                           LessOp::getOperationName(),
                           AddV2Op::getOperationName(),
                           DivOp::getOperationName(),
                           SubOp::getOperationName(),
                           LogOp::getOperationName(),
                           Log1pOp::getOperationName(),
                           IsInfOp::getOperationName(),
                           MulOp::getOperationName(),
                           FloorOp::getOperationName(),
                           AbsOp::getOperationName(),
                           GreaterOp::getOperationName(),
                           SinOp::getOperationName(),
                           IsFiniteOp::getOperationName(),
                       }) {}

  LogicalResult matchAndRewrite(Operation *src_op,
                                PatternRewriter &rewriter) const override {
    auto op = cast<LgammaOp>(src_op);

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
      input = rewriter.create<CastOp>(loc, tensor_type, input);
    }

    // Helper lambda function for creating a ConstOp for a tensor filled with
    // the given constant float value.
    auto create_const_op = [&rewriter, loc, tensor_type,
                            float_type](double value) {
      return rewriter.create<ConstOp>(
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

    Value minus_input = rewriter.create<NegOp>(loc, input);
    Value input_minus_one = rewriter.create<SubOp>(loc, input, one);

    // If the input is less than 0.5 use Euler's reflection formula:
    // gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
    Value need_to_reflect = rewriter.create<LessOp>(loc, input, one_half);
    Type tensor_bool_type = need_to_reflect.getType();
    Value z = rewriter.create<SelectV2Op>(loc, need_to_reflect, minus_input,
                                          input_minus_one);

    Value x = base_lanczos_coeff;
    for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
      Value lanczos_coefficient = create_const_op(kLanczosCoefficients[i]);
      Value index = create_const_op(static_cast<double>(i));
      Value z_plus_index = rewriter.create<AddV2Op>(loc, z, index);
      Value z_plus_index_plus_one =
          rewriter.create<AddV2Op>(loc, z_plus_index, one);
      Value incr = rewriter.create<DivOp>(loc, lanczos_coefficient,
                                          z_plus_index_plus_one);
      x = rewriter.create<AddV2Op>(loc, x, incr);
    }

    // To improve accuracy on platforms with less-precise log implementations,
    // compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
    // the device.
    // log(t) = log(kLanczosGamma + 0.5 + z)
    //        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
    Value t = rewriter.create<AddV2Op>(loc, lanczos_gamma_plus_one_half, z);
    Value z_div_lanczos_gamma_plus_one_half =
        rewriter.create<DivOp>(loc, z, lanczos_gamma_plus_one_half);
    Value log1p_z_div_lanczos_gamma_plus_one_half =
        rewriter.create<Log1pOp>(loc, z_div_lanczos_gamma_plus_one_half);
    Value log_t =
        rewriter.create<AddV2Op>(loc, log_lanczos_gamma_plus_one_half,
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
    Value t_div_log_t = rewriter.create<DivOp>(loc, t, log_t);
    Value one_half_minus_t_div_log_t =
        rewriter.create<SubOp>(loc, one_half, t_div_log_t);
    Value z_plus_one_half_minus_t_div_log_t =
        rewriter.create<AddV2Op>(loc, z, one_half_minus_t_div_log_t);
    Value z_plus_one_half_minus_t_div_log_t_mul_log_t =
        rewriter.create<MulOp>(loc, z_plus_one_half_minus_t_div_log_t, log_t);
    Value log_x = rewriter.create<LogOp>(loc, x);
    Value log_y_rhs = rewriter.create<AddV2Op>(
        loc, z_plus_one_half_minus_t_div_log_t_mul_log_t, log_x);
    Value log_y = rewriter.create<AddV2Op>(loc, log_sqrt_two_pi, log_y_rhs);

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
    Value abs_input = rewriter.create<AbsOp>(loc, input);
    Value abs_input_floor = rewriter.create<FloorOp>(loc, abs_input);
    Value abs_frac_input =
        rewriter.create<SubOp>(loc, abs_input, abs_input_floor);

    // Convert values of abs_frac_input > 0.5 to (1 - frac_input) to improve
    // precision of pi * abs_frac_input for values of abs_frac_input close to 1.
    Value one_minus_abs_frac_input =
        rewriter.create<SubOp>(loc, one, abs_frac_input);
    Value abs_frac_input_gt_one_half =
        rewriter.create<GreaterOp>(loc, abs_frac_input, one_half);
    Value reduced_frac_input =
        rewriter.create<SelectV2Op>(loc, abs_frac_input_gt_one_half,
                                    one_minus_abs_frac_input, abs_frac_input);
    Value pi_mul_reduced_frac_input =
        rewriter.create<MulOp>(loc, pi, reduced_frac_input);
    Value sin_pi_mul_reduced_frac_input =
        rewriter.create<SinOp>(loc, pi_mul_reduced_frac_input);
    Value reflection_denom =
        rewriter.create<LogOp>(loc, sin_pi_mul_reduced_frac_input);

    // Avoid computing -inf - inf, which is nan.  If reflection_denom is +/-inf,
    // then it "wins" and the result is +/-inf.
    Value is_finite =
        rewriter.create<IsFiniteOp>(loc, tensor_bool_type, reflection_denom);
    Value neg_reflection_denom = rewriter.create<NegOp>(loc, reflection_denom);
    Value log_pi_minus_reflection_denom =
        rewriter.create<SubOp>(loc, log_pi, reflection_denom);
    Value reflection_if_finite =
        rewriter.create<SubOp>(loc, log_pi_minus_reflection_denom, log_y);
    Value reflection = rewriter.create<SelectV2Op>(
        loc, is_finite, reflection_if_finite, neg_reflection_denom);

    Value result =
        rewriter.create<SelectV2Op>(loc, need_to_reflect, reflection, log_y);

    // lgamma(+/-inf) = +inf.
    Value is_inf = rewriter.create<IsInfOp>(loc, tensor_bool_type, input);
    result = rewriter.create<SelectV2Op>(loc, is_inf, infinity, result);

    if (needs_cast) {
      result = rewriter.create<CastOp>(loc, original_tensor_type, result);
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
class LowerPackOp : public RewritePattern {
 public:
  explicit LowerPackOp(MLIRContext *context)
      : RewritePattern(
            PackOp::getOperationName(), 1, context,
            {ConstOp::getOperationName(), ConcatV2Op::getOperationName(),
             ExpandDimsOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *src_op,
                                PatternRewriter &rewriter) const override {
    auto op = cast<PackOp>(src_op);

    Location loc = op.getLoc();
    auto axis_value = rewriter.create<ConstOp>(
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
      expanded_inputs.push_back(
          rewriter.create<ExpandDimsOp>(loc, inferred_ty, input, axis_value));
    }

    rewriter.replaceOpWithNewOp<ConcatV2Op>(op, op.getType(), expanded_inputs,
                                            axis_value);
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
class LowerSpaceToBatchNDOp : public RewritePattern {
 public:
  explicit LowerSpaceToBatchNDOp(MLIRContext *context)
      : RewritePattern(SpaceToBatchNDOp::getOperationName(), 1, context,
                       {
                           CastOp::getOperationName(),
                           ConstOp::getOperationName(),
                           ConcatV2Op::getOperationName(),
                           AddV2Op::getOperationName(),
                           PadOp::getOperationName(),
                           SplitOp::getOperationName(),
                           UnpackOp::getOperationName(),
                           DivOp::getOperationName(),
                           MulOp::getOperationName(),
                           ReshapeOp::getOperationName(),
                           TransposeOp::getOperationName(),
                       }) {}

  LogicalResult matchAndRewrite(Operation *src_op,
                                PatternRewriter &rewriter) const override {
    auto op = cast<SpaceToBatchNDOp>(src_op);

    Location loc = op.getLoc();
    auto input_type = op.input().getType().cast<TensorType>();
    auto element_type = input_type.getElementType();
    if (!input_type.hasStaticShape()) {
      return failure();
    }
    ArrayRef<int64_t> input_shape = input_type.getShape();
    auto block_shape_type = op.block_shape().getType().cast<TensorType>();
    if (!block_shape_type.hasStaticShape()) {
      return failure();
    }
    auto paddings_type = op.paddings().getType().cast<ShapedType>();
    if (!paddings_type.hasRank()) {
      return failure();
    }

    int64_t input_rank = input_type.getRank();
    int64_t block_rank = block_shape_type.getNumElements();
    int64_t remaining_rank = input_rank - 1 - block_rank;
    if (remaining_rank < 0) {
      // TODO(b/157475606): Move this check to ::Verify
      return failure();
    }

    auto block_shape_i64_type = RankedTensorType::get(
        block_shape_type.getShape(), rewriter.getIntegerType(64));
    auto block_shape_i64 =
        rewriter.create<CastOp>(loc, block_shape_i64_type, op.block_shape());

    auto paddings_i64_type = RankedTensorType::get(paddings_type.getShape(),
                                                   rewriter.getIntegerType(64));
    auto paddings_i64 =
        rewriter.create<CastOp>(loc, paddings_i64_type, op.paddings());

    auto pad00 = rewriter.create<ConstOp>(
        loc, DenseElementsAttr::get<int64_t>(
                 RankedTensorType::get({1, 2}, rewriter.getIntegerType(64)),
                 {0, 0}));
    SmallVector<Value, 4> full_paddings_list{pad00, paddings_i64};
    full_paddings_list.append(remaining_rank, pad00);
    auto full_paddings_type =
        RankedTensorType::get({input_rank, 2}, rewriter.getIntegerType(64));
    auto zero_i64 = rewriter.create<ConstOp>(
        loc, GetScalarOfType(rewriter.getIntegerType(64), 0));
    // Extends paddings to all dimensions of input by adding 0s to non-block
    // dimensions.
    auto full_paddings = rewriter.create<ConcatV2Op>(
        loc, full_paddings_type, full_paddings_list, zero_i64);

    // Compute the result type here instead of using shape inference because the
    // full_paddings won't be available as a constant for shape inference.
    ElementsAttr block_shape;
    ElementsAttr paddings;
    llvm::SmallVector<int64_t, 4> block_shape_ints;
    auto padded_shape = llvm::to_vector<4>(input_shape);
    if (matchPattern(op.block_shape(), m_Constant(&block_shape)) &&
        matchPattern(op.paddings(), m_Constant(&paddings))) {
      for (uint64_t i = 0; i < block_rank; i++) {
        int64_t paddings_sum =
            paddings.getValue({i, 0}).cast<IntegerAttr>().getInt() +
            paddings.getValue({i, 1}).cast<IntegerAttr>().getInt();
        int64_t block_shape_i =
            block_shape.getValue({i}).cast<IntegerAttr>().getInt();
        padded_shape[i + 1] = (paddings_sum + input_shape[i + 1]);
        block_shape_ints.push_back(block_shape_i);
      }
    } else {
      for (int i = 0; i < block_rank; i++) {
        padded_shape[i + 1] = ShapedType::kDynamicSize;
      }
      block_shape_ints.resize(block_shape_type.getNumElements(), -1);
    }

    auto padded_type = RankedTensorType::get(padded_shape, element_type);
    // padded = pad(input, full_paddings)
    auto padded =
        rewriter.create<PadOp>(loc, padded_type, op.input(), full_paddings);

    auto paddings_sum_type =
        RankedTensorType::get({input_rank}, rewriter.getIntegerType(64));
    // paddings_sum = paddings[*,0] + paddings[*,1]
    auto paddings_split = rewriter.create<UnpackOp>(
        loc, TypeRange({paddings_sum_type, paddings_sum_type}), full_paddings,
        rewriter.getI64IntegerAttr(1));
    auto paddings_sum = rewriter.create<AddV2Op>(
        loc, paddings_split.getResult(0), paddings_split.getResult(1));

    auto input_shape_tensor = rewriter.create<ConstOp>(
        loc,
        DenseElementsAttr::get(
            RankedTensorType::get({input_rank}, rewriter.getIntegerType(64)),
            input_shape));

    // padded_shape_tensor is the shape of padded.
    auto padded_shape_tensor =
        rewriter.create<AddV2Op>(loc, paddings_sum, input_shape_tensor);

    auto zero_i32 = rewriter.create<ConstOp>(
        loc, GetScalarOfType(rewriter.getIntegerType(32), 0));
    SmallVector<Type, 4> padded_shape_splits_types(
        input_rank, RankedTensorType::get({1}, rewriter.getIntegerType(64)));
    SmallVector<Value, 4> padded_shape_splits(
        rewriter
            .create<SplitOp>(loc, padded_shape_splits_types, zero_i32,
                             padded_shape_tensor)
            .output());

    SmallVector<Type, 4> block_shape_splits_types(
        block_rank, RankedTensorType::get({1}, rewriter.getIntegerType(64)));
    SmallVector<Value, 4> block_shape_splits(
        rewriter
            .create<SplitOp>(loc, block_shape_splits_types, zero_i32,
                             block_shape_i64)
            .output());

    SmallVector<int64_t, 4> outer_shape_ints;
    SmallVector<Value, 4> outer_shape_vals;
    for (int64_t i = 0; i < block_rank; ++i) {
      // TODO(b/157475606): Insert tf.Assert that the following division has
      // remainder 0.
      outer_shape_vals.push_back(rewriter.create<DivOp>(
          loc, padded_shape_splits[1 + i], block_shape_splits[i]));

      auto padded_shape_i = padded_shape[1 + i];
      auto block_shape_ints_i = block_shape_ints[i];

      // Compute the outer_shape constant values to infer the reshape.
      if (padded_shape_i == -1 || block_shape_ints_i == -1) {
        outer_shape_ints.push_back(-1);
      } else {
        outer_shape_ints.push_back(padded_shape_i / block_shape_ints_i);
      }
    }

    SmallVector<Value, 6> reshaped_shape_vals{padded_shape_splits[0]};
    SmallVector<int64_t, 6> reshaped_shape_ints{padded_shape[0]};
    for (int64_t i = 0; i < block_rank; ++i) {
      reshaped_shape_vals.push_back(outer_shape_vals[i]);
      reshaped_shape_vals.push_back(block_shape_splits[i]);

      reshaped_shape_ints.push_back(outer_shape_ints[i]);
      reshaped_shape_ints.push_back(block_shape_ints[i]);
    }
    for (int64_t i = 1 + block_rank; i < input_rank; ++i) {
      reshaped_shape_vals.push_back(padded_shape_splits[i]);
      reshaped_shape_ints.push_back(padded_shape[i]);
    }
    auto reshaped_shape = ValuesToRank1(
        rewriter, loc, rewriter.getIntegerType(64), reshaped_shape_vals);

    auto reshaped = rewriter.create<ReshapeOp>(
        loc, RankedTensorType::get(reshaped_shape_ints, element_type), padded,
        reshaped_shape);

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
    auto permutation = rewriter.create<ConstOp>(
        loc, GetI64ElementsAttr(permutation_vals, &rewriter));

    auto permuted = rewriter.create<TransposeOp>(loc, reshaped, permutation);
    auto output_batch = padded_shape_splits[0];
    for (int64_t i = 0; i < block_rank; ++i) {
      output_batch =
          rewriter.create<MulOp>(loc, output_batch, block_shape_splits[i]);
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

    // Sometimes the result type is more specific than what the reshape builder
    // can infer.
    auto result_type = op.getResult().getType();
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, result_type, permuted,
                                           output_shape);

    return success();
  }
};

class LowerBatchToSpaceND : public RewritePattern {
 public:
  explicit LowerBatchToSpaceND(MLIRContext *context)
      : RewritePattern(BatchToSpaceNDOp::getOperationName(), 1, context,
                       {
                           ConstOp::getOperationName(),
                           ReshapeOp::getOperationName(),
                           SliceOp::getOperationName(),
                           TransposeOp::getOperationName(),
                       }) {}

  LogicalResult matchAndRewrite(Operation *src_op,
                                PatternRewriter &rewriter) const override {
    auto op = cast<BatchToSpaceNDOp>(src_op);
    auto input = op.input();
    auto input_ty = input.getType().cast<ShapedType>();
    auto element_ty = input_ty.getElementType();
    if (!input_ty.hasStaticShape()) {
      return failure();
    }

    const int input_rank = input_ty.getRank();
    auto input_shape = input_ty.getShape();

    DenseIntElementsAttr block_shape;
    DenseIntElementsAttr crops;
    if (!matchPattern(op.block_shape(), m_Constant(&block_shape)) ||
        !matchPattern(op.crops(), m_Constant(&crops))) {
      return failure();
    }

    auto block_shape_ty = block_shape.getType();
    if (!block_shape_ty.hasRank() || block_shape_ty.getRank() != 1) {
      return failure();
    }

    const int block_rank = block_shape_ty.getShape().front();
    auto remainder_shape = input_shape.drop_front(1 + block_rank);

    const int64_t batch_size = input_shape[0];

    // Compute the product of the block_shape values.
    int64_t block_num_elems = 1;

    for (auto val : block_shape.getIntValues()) {
      block_num_elems *= val.getSExtValue();
    }

    if (block_num_elems <= 0) {
      op.emitOpError()
          << "The product of the block dimensions must be positive";
      return failure();
    }

    // 1. Reshape `input` to `reshaped` of shape:
    //      [block_shape[0], ..., block_shape[M-1],
    //       batch / prod(block_shape),
    //       input_shape[1], ..., input_shape[N-1]]
    std::vector<int64_t> reshaped_shape;
    for (auto val : block_shape) {
      reshaped_shape.push_back(val.getSExtValue());
    }
    reshaped_shape.resize(input_rank + block_rank);

    reshaped_shape[block_rank] = batch_size / block_num_elems;
    std::copy(input_shape.begin() + 1, input_shape.end(),
              reshaped_shape.begin() + block_rank + 1);

    auto reshaped = rewriter.create<TF::ReshapeOp>(
        op.getLoc(), RankedTensorType::get(reshaped_shape, element_ty), input,
        rewriter.create<ConstOp>(op.getLoc(),
                                 rewriter.getI64TensorAttr(reshaped_shape)));

    // 2. Permute dimensions of `reshaped` to produce `permuted` of shape
    //      [batch / prod(block_shape),
    //
    //       input_shape[1], block_shape[0],
    //       ...,
    //       input_shape[M], block_shape[M-1],
    //
    //       input_shape[M+1], ..., input_shape[N-1]]
    std::vector<int64_t> permutation(reshaped_shape.size());
    permutation[0] = block_rank;
    for (int i = 0; i < block_rank; ++i) {
      permutation[1 + 2 * i] = block_rank + 1 + i;
      permutation[1 + 2 * i + 1] = i;
    }
    std::iota(permutation.begin() + 1 + block_rank * 2, permutation.end(),
              1 + block_rank * 2);

    std::vector<int64_t> transpose_shape(permutation.size());
    for (auto it : llvm::enumerate(permutation)) {
      transpose_shape[it.index()] = reshaped_shape[it.value()];
    }

    auto permuted = rewriter.create<TF::TransposeOp>(
        op.getLoc(), RankedTensorType::get(transpose_shape, element_ty),
        reshaped,
        rewriter.create<ConstOp>(op.getLoc(),
                                 rewriter.getI64TensorAttr(permutation)));

    // 3. Reshape `permuted` to produce `reshaped_permuted` of shape
    //      [batch / prod(block_shape),
    //
    //       input_shape[1] * block_shape[0],
    //       ...,
    //       input_shape[M] * block_shape[M-1],
    //
    //       input_shape[M+1],
    //       ...,
    //       input_shape[N-1]]
    std::vector<int64_t> reshaped_permuted_shape(input_rank);
    auto block_shape_values = llvm::to_vector<4>(block_shape.getIntValues());
    reshaped_permuted_shape[0] = batch_size / block_num_elems;
    for (int i = 0; i < block_rank; ++i) {
      reshaped_permuted_shape[1 + i] =
          block_shape_values[i].getSExtValue() * input_shape[1 + i];
    }
    std::copy(remainder_shape.begin(), remainder_shape.end(),
              reshaped_permuted_shape.begin() + 1 + block_rank);

    auto reshaped_permuted = rewriter.create<TF::ReshapeOp>(
        op.getLoc(), RankedTensorType::get(reshaped_permuted_shape, element_ty),
        permuted,
        rewriter.create<ConstOp>(
            op.getLoc(), rewriter.getI64TensorAttr(reshaped_permuted_shape)));

    // 4. Crop the start and end of dimensions `[1, ..., M]` of
    //    `reshaped_permuted` according to `crops` to produce the output of
    //    shape:
    //      [batch / prod(block_shape),
    //
    //       input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
    //       ...,
    //       input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],
    //
    //       input_shape[M+1], ..., input_shape[N-1]]
    std::vector<int64_t> start_indices(input_rank, 0);
    std::vector<int64_t> slice_sizes = reshaped_permuted_shape;
    std::vector<int64_t> strides(input_rank, 1);
    auto crop_values = llvm::to_vector<4>(crops.getIntValues());
    for (int i = 0; i < block_rank; ++i) {
      int64_t crop_start = crop_values[i * 2].getSExtValue();
      int64_t crop_end = crop_values[i * 2 + 1].getSExtValue();

      if (crop_start < 0 || crop_end < 0) {
        op.emitOpError() << "Crops must be non-negative";
        return failure();
      }

      start_indices[i + 1] = crop_start;
      slice_sizes[i + 1] -= crop_start + crop_end;

      if (slice_sizes[i + 1] < 0) {
        op.emitOpError() << "Cropped size must be non-negative: start: "
                         << crop_start << " end: " << crop_end << " size "
                         << reshaped_permuted_shape[1 + i];
      }
    }

    rewriter.replaceOpWithNewOp<TF::SliceOp>(
        op, RankedTensorType::get(slice_sizes, element_ty), reshaped_permuted,
        rewriter.create<ConstOp>(op.getLoc(),
                                 rewriter.getI64TensorAttr(start_indices)),
        rewriter.create<ConstOp>(op.getLoc(),
                                 rewriter.getI64TensorAttr(slice_sizes)));
    return success();
  }
};

// Lowers `SparseMatMulOp` to `MatMulOp`, ignoring the sparseness hints,
// since we currently don't have an implementation that can use this
// information. Adds appropriate casts where necessary to align element types
// of operands and result for `MatMulOp`.
class LowerSparseMatMulOp : public RewritePattern {
 public:
  explicit LowerSparseMatMulOp(MLIRContext *context)
      : RewritePattern(
            SparseMatMulOp::getOperationName(), 1, context,
            {CastOp::getOperationName(), MatMulOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *src_op,
                                PatternRewriter &rewriter) const override {
    auto op = cast<SparseMatMulOp>(src_op);

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
      operand = rewriter.create<CastOp>(op.getLoc(), tensor_type_f32, operand);
    }
    Value result = rewriter.create<MatMulOp>(
        op.getLoc(), op.product().getType(), operands[0], operands[1],
        op.transpose_a(), op.transpose_b());

    rewriter.replaceOp(op, {result});
    return success();
  }
};

// Lowers _UnaryOpsComposition op as a series of original TensorFlow ops that
// were fused together.
class Lower_UnaryOpsComposition
    : public OpRewritePattern<_UnaryOpsCompositionOp> {
 public:
  using OpRewritePattern<_UnaryOpsCompositionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(_UnaryOpsCompositionOp op,
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

// Lowers ResizeNearestNeighbor to an indices computations with a gather along
// the combined spatial dimensions. Generating the indices along the
// width/height index could be used to gather along each of W and H dimension
// of the input image array. To reduce to a single gather, these indices are
// combined, so a single gather can be performed along the combined spatial
// dimensions.
//
// Images must take the shape [b, h, w, c] and size is a rank-1 length-2 tensor
// containing the height and width values for the output tensor. This lowering
// should work with a dynamic images array.
//
// For example, a scaling with image shape [1, 3, 3, 1] to [2, 2] and unaligned
// corners would generate a [0, 1] lookup along both the x and y direction.
// Then when combined to form the 1-D spatial index the values would be
// [0, 1, 3, 4] which would gather along the reshape image tensor of shape
// [1, 9, 1], reshaped to the final [1, 3, 3, 1].
class LowerResizeNearestNeighbor : public RewritePattern {
 public:
  explicit LowerResizeNearestNeighbor(MLIRContext *context)
      : RewritePattern(ResizeNearestNeighborOp::getOperationName(), 1, context,
                       {
                           BroadcastToOp::getOperationName(),
                           ConstOp::getOperationName(),
                           DivOp::getOperationName(),
                           PackOp::getOperationName(),
                           RangeOp::getOperationName(),
                           ReshapeOp::getOperationName(),
                           ShapeOp::getOperationName(),
                           SplitOp::getOperationName(),
                           TransposeOp::getOperationName(),
                       }) {}

  LogicalResult matchAndRewrite(Operation *src_op,
                                PatternRewriter &rewriter) const override {
    auto op = cast<ResizeNearestNeighborOp>(src_op);
    auto loc = op.getLoc();
    auto result_ty = op.getType().cast<ShapedType>();

    auto input = op.images();
    auto input_ty = input.getType().cast<ShapedType>();
    auto input_element_ty = input_ty.getElementType();
    auto out_size = op.size();
    auto out_size_ty = out_size.getType().cast<ShapedType>();
    auto out_size_element_ty = out_size_ty.getElementType();

    // Input should be rank 4.
    if (!input_ty.hasRank() || input_ty.getRank() != 4) {
      return failure();
    }

    // Check that out_size is rank-1, length-2. Otherwise the size is not legal.
    if (!out_size_ty.hasRank() || out_size_ty.getRank() != 1 ||
        out_size_ty.getShape()[0] != 2) {
      return failure();
    }

    // Extract the output width / height dim size.
    int out_height_constant = -1;
    int out_width_constant = -1;
    DenseIntElementsAttr out_size_cst;
    if (matchPattern(out_size, m_Constant(&out_size_cst))) {
      llvm::SmallVector<int64_t, 2> cst_size;
      for (auto val : out_size_cst.getIntValues()) {
        cst_size.push_back(val.getSExtValue());
      }

      out_height_constant = cst_size[0];
      out_width_constant = cst_size[1];

      if (out_height_constant < 0 || out_width_constant < 0) return failure();
    }

    int out_spatial_cst = out_height_constant < 0 || out_width_constant < 0
                              ? -1
                              : out_height_constant * out_width_constant;

    // Input rank should be 4. Might be able to drop this requirement entirely
    // as its an input requirement.
    if (!input_ty.hasRank() || input_ty.getRank() != 4) {
      return failure();
    }

    int batch_cst = input_ty.getShape()[0];
    int channels_cst = input_ty.getShape()[3];

    int in_y_cst = input_ty.getShape()[1];
    int in_x_cst = input_ty.getShape()[2];
    int in_spatial_cst =
        in_y_cst < 0 || in_x_cst < 0 ? -1 : in_y_cst * in_x_cst;

    // TODO(suderman): Add support for these optional parameters.
    if (op.align_corners() == true || op.half_pixel_centers() == true) {
      return failure();
    }

    auto one =
        rewriter.create<ConstOp>(loc, GetScalarOfType(out_size_element_ty, 1));

    // Extract the image shape.
    Value input_shape = rewriter.create<ShapeOp>(
        loc, RankedTensorType::get({4}, rewriter.getI64Type()), input);
    input_shape = rewriter.create<CastOp>(
        loc, RankedTensorType::get({4}, out_size_element_ty), input_shape);

    auto scalar_dim_ty = RankedTensorType::get({}, out_size_element_ty);
    auto split_image_shape = rewriter.create<UnpackOp>(
        loc,
        TypeRange({scalar_dim_ty, scalar_dim_ty, scalar_dim_ty, scalar_dim_ty}),
        input_shape);

    // Extract the separate components from the input shape.
    auto batch = split_image_shape.getResult(0);
    auto in_y = split_image_shape.getResult(1);
    auto in_x = split_image_shape.getResult(2);
    auto channels = split_image_shape.getResult(3);

    auto in_count = rewriter.create<MulOp>(
        loc, RankedTensorType::get({}, out_size_element_ty), in_y, in_x);

    // Unpack and separate the out width/height.
    auto split_out_size = rewriter.create<UnpackOp>(
        loc, TypeRange({scalar_dim_ty, scalar_dim_ty}), out_size);

    auto out_y = split_out_size.getResult(0);
    auto out_x = split_out_size.getResult(1);

    auto out_count = rewriter.create<MulOp>(
        loc, RankedTensorType::get({}, out_size_element_ty), out_y, out_x);

    // Generate what the final output shape will look like.
    auto out_shape = rewriter.create<PackOp>(
        loc, RankedTensorType::get({4}, out_size_element_ty),
        ValueRange({batch, out_y, out_x, channels}));

    // Compute the indices along the vertical dimension.
    auto in_y_f32 = rewriter.create<CastOp>(
        loc, RankedTensorType::get({}, rewriter.getF32Type()), in_y);
    auto out_w_f32 = rewriter.create<CastOp>(
        loc, RankedTensorType::get({}, rewriter.getF32Type()), out_y);

    Value y_scale = rewriter.create<DivOp>(
        loc, RankedTensorType::get({}, rewriter.getF32Type()), in_y_f32,
        out_w_f32);

    Value zero_f32 = rewriter.create<ConstOp>(
        loc, GetScalarOfType(rewriter.getF32Type(), 0.0));
    Value one_f32 = rewriter.create<ConstOp>(
        loc, GetScalarOfType(rewriter.getF32Type(), 1.0));

    Value y_range = rewriter.create<RangeOp>(
        loc,
        RankedTensorType::get({out_height_constant}, rewriter.getF32Type()),
        zero_f32, out_w_f32, one_f32);

    y_range = rewriter.create<MulOp>(
        loc,
        RankedTensorType::get({out_height_constant}, rewriter.getF32Type()),
        y_range, y_scale);

    y_range = rewriter.create<CastOp>(
        loc, RankedTensorType::get({out_height_constant}, out_size_element_ty),
        y_range);

    y_range = rewriter.create<ReshapeOp>(
        loc,
        RankedTensorType::get({out_height_constant, 1}, out_size_element_ty),
        y_range,
        rewriter.create<PackOp>(loc,
                                RankedTensorType::get({2}, out_size_element_ty),
                                ValueRange({out_y, one})));

    Value y_indices = rewriter.create<MulOp>(
        loc,
        RankedTensorType::get({out_height_constant, 1}, out_size_element_ty),
        y_range, in_x);

    // Compute the indices for the nearest neighbour lookup across the width
    // dim.
    auto in_x_f32 = rewriter.create<CastOp>(
        loc, RankedTensorType::get({}, rewriter.getF32Type()), in_x);
    auto out_h_f32 = rewriter.create<CastOp>(
        loc, RankedTensorType::get({}, rewriter.getF32Type()), out_x);

    Value x_scale = rewriter.create<DivOp>(
        loc, RankedTensorType::get({}, rewriter.getF32Type()), in_x_f32,
        out_h_f32);

    Value x_range = rewriter.create<RangeOp>(
        loc, RankedTensorType::get({out_width_constant}, rewriter.getF32Type()),
        zero_f32, out_h_f32, one_f32);

    x_range = rewriter.create<MulOp>(
        loc, RankedTensorType::get({out_width_constant}, rewriter.getF32Type()),
        x_range, x_scale);

    x_range = rewriter.create<CastOp>(
        loc, RankedTensorType::get({out_width_constant}, out_size_element_ty),
        x_range);

    Value x_indices = rewriter.create<ReshapeOp>(
        loc,
        RankedTensorType::get({1, out_width_constant}, out_size_element_ty),
        x_range,
        rewriter.create<PackOp>(loc,
                                RankedTensorType::get({2}, out_size_element_ty),
                                ValueRange({one, out_x})));

    // Generate the combined index array, reshape to be 1-D.
    Value indices = rewriter.create<AddV2Op>(
        loc,
        RankedTensorType::get({out_height_constant, out_width_constant},
                              out_size_element_ty),
        y_indices, x_indices);

    indices = rewriter.create<ReshapeOp>(
        loc, RankedTensorType::get({out_spatial_cst}, out_size_element_ty),
        indices,
        rewriter.create<ReshapeOp>(
            loc, RankedTensorType::get({1}, out_size_element_ty), out_count,
            rewriter.create<ConstOp>(loc, rewriter.getI64TensorAttr({1}))));

    // Group the spatial indices and gather along that combined index.
    Value input_collapsed_spatial = rewriter.create<ReshapeOp>(
        loc,
        RankedTensorType::get({batch_cst, in_spatial_cst, channels_cst},
                              input_element_ty),
        input,
        rewriter.create<PackOp>(loc,
                                RankedTensorType::get({3}, out_size_element_ty),
                                ValueRange({batch, in_count, channels})));

    Value gathered_values = rewriter.create<GatherV2Op>(
        loc,
        RankedTensorType::get({batch_cst, out_spatial_cst, channels_cst},
                              input_element_ty),
        input_collapsed_spatial, indices, /*axis=*/one);

    gathered_values =
        rewriter.create<ReshapeOp>(loc, result_ty, gathered_values, out_shape);

    rewriter.replaceOp(op, gathered_values);
    return success();
  }
};

struct LowerRollOp : public RewritePattern {
  explicit LowerRollOp(MLIRContext *context)
      : RewritePattern(
            RollOp::getOperationName(), 1, context,
            {ConstOp::getOperationName(), SliceOp::getOperationName(),
             ConcatV2Op::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto tf_roll_op = cast<RollOp>(op);

    auto input_ty = tf_roll_op.input().getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "require the type of input to have static shapes");
    }

    DenseIntElementsAttr shift_attr;
    Value shift = tf_roll_op.shift();
    auto shift_ranked_attr_type = shift.getType().dyn_cast<RankedTensorType>();
    if (!shift_ranked_attr_type ||
        !matchPattern(shift, m_Constant(&shift_attr))) {
      return failure();
    }

    DenseIntElementsAttr axis_attr;
    Value axis = tf_roll_op.axis();
    auto axis_ranked_attr_type = axis.getType().dyn_cast<RankedTensorType>();
    if (!axis_ranked_attr_type || !matchPattern(axis, m_Constant(&axis_attr))) {
      return failure();
    }

    // Combine duplicate axis and make sure they are in [0, rank(input)) range.
    auto input_shape = input_ty.getShape();
    int input_rank = input_shape.size();
    SmallVector<int32_t, 4> shift_map(input_rank, 0);
    for (int i = 0; i < axis_attr.getNumElements(); ++i) {
      int32_t axis_i = axis_attr.getValue<int32_t>(i);
      if (axis_i < 0) axis_i += input_rank;
      int32_t shift_i = shift_attr.getValue<int32_t>(i);
      shift_map[axis_i] += shift_i;
    }

    SmallVector<int32_t, 4> adjusted_axis;
    SmallVector<int32_t, 4> adjusted_shift;
    for (int i = 0; i < input_rank; ++i) {
      int32_t input_dims_i = input_shape[i];
      int32_t shift_i = shift_map[i] % input_dims_i;
      if (shift_i < 0) shift_i += input_dims_i;
      if (shift_i == 0) continue;
      adjusted_axis.push_back(i);
      adjusted_shift.push_back(shift_i);
    }

    // Convert rolling in each dimension to two Slice ops and one Concat op.
    auto axis_type =
        RankedTensorType::get({input_rank}, rewriter.getIntegerType(64));
    auto create_slice_op = [&](int32_t axis_i, int32_t begin_i, int32_t size_i,
                               Value input) {
      SmallVector<int64_t, 4> begin_values(input_rank, 0);
      begin_values[axis_i] = begin_i;
      auto begin_attr = DenseIntElementsAttr::get(axis_type, begin_values);
      auto begin =
          rewriter.create<ConstOp>(op->getLoc(), axis_type, begin_attr);

      SmallVector<int64_t, 4> output_shape;
      output_shape.append(input_shape.begin(), input_shape.end());
      output_shape[axis_i] = size_i;
      auto size_attr = DenseIntElementsAttr::get(axis_type, output_shape);
      auto size = rewriter.create<ConstOp>(op->getLoc(), axis_type, size_attr);

      auto slice_op_ty =
          RankedTensorType::get(output_shape, input_ty.getElementType());
      return rewriter.create<SliceOp>(op->getLoc(), slice_op_ty, input, begin,
                                      size);
    };

    auto result = tf_roll_op.input();
    auto scalar_type =
        mlir::RankedTensorType::get({}, rewriter.getIntegerType(32));
    for (int i = 0; i < adjusted_axis.size(); ++i) {
      int32_t axis_i = adjusted_axis[i];
      int32_t shift_i = adjusted_shift[i];
      auto slice_op_1 = create_slice_op(axis_i, input_shape[axis_i] - shift_i,
                                        shift_i, result);
      auto slice_op_2 =
          create_slice_op(axis_i, 0, input_shape[axis_i] - shift_i, result);

      auto dim_attr = DenseIntElementsAttr::get(scalar_type, {axis_i});
      auto concat_dim =
          rewriter.create<ConstOp>(op->getLoc(), scalar_type, dim_attr);
      auto concat_op = rewriter.create<ConcatV2Op>(
          op->getLoc(), input_ty,
          ArrayRef<Value>({slice_op_1.output(), slice_op_2.output()}),
          concat_dim);
      result = concat_op.getResult();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Decomposes Softmax and LogSoftmax to primitive TF ops, using the following
// formulas:
//
//     softmax = div(exp(logits), sum(exp(logits)))
//     log_softmax = sub(logits, log(sum(exp(logits))))
//
// TODO(jpienaar): Evaluate benefit of templating here.
template <typename OpTy, bool use_log = true>
class LowerSoftmaxOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value logits = op.logits();
    auto loc = op.getLoc();

    // Note that the TensorFlow Softmax op verifies that the input rank is
    // greater than or equal to one so the following sequence is valid.
    auto reduce_dim =
        rewriter.create<TF::ConstOp>(loc, GetI64ElementsAttr({-1}, &rewriter));

    // Exponential of input values and then their sum can be very large here.
    // Division with large denominator is numerically unstable. To improve
    // numerical stability, subtract each batch with their max element so that
    // the maximum input value is zero. It can be shown that softmax computed
    // after adding or subtracting all inputs in a batch using a common value
    // gives mathematically equivalent result.
    auto max_logits =
        rewriter.create<TF::MaxOp>(loc, logits, reduce_dim,
                                   /*keep_dims=*/rewriter.getBoolAttr(true));
    auto shifted_logits = rewriter.create<TF::SubOp>(loc, logits, max_logits);

    // Exponentiate the inputs.
    Value exp = rewriter.create<TF::ExpOp>(loc, shifted_logits);

    // Compute summation of the exponentials.
    Value sum =
        rewriter.create<TF::SumOp>(loc, exp, reduce_dim,
                                   /*keep_dims=*/rewriter.getBoolAttr(true));

    if (use_log) {
      Value log = rewriter.create<TF::LogOp>(loc, sum);
      rewriter.replaceOpWithNewOp<TF::SubOp>(op, shifted_logits, log);
    } else {
      rewriter.replaceOpWithNewOp<TF::DivOp>(op, exp, sum);
    }
    return success();
  }
};

}  // namespace

void PopulateLoweringTFPatterns(MLIRContext *context,
                                OwningRewritePatternList *patterns) {
  // clang-format off
  patterns->insert<
      LowerAddNOp,
      ConvertFakeQuantWithMinMaxVarsOp,
      LowerDynamicStitchOp<DynamicStitchOp>,
      LowerDynamicStitchOp<ParallelDynamicStitchOp>,
      LowerInvertPermutationOp,
      LowerLgammaOp,
      LowerPackOp,
      LowerBatchToSpaceND,
      LowerSpaceToBatchNDOp,
      LowerResizeNearestNeighbor,
      LowerSparseMatMulOp,
      Lower_UnaryOpsComposition,
      LowerRollOp>(context);
  // clang-format on
  populateWithGenerated(*patterns);
}

void PopulateTFLoweringBeforeHLOPatterns(MLIRContext *context,
                                         OwningRewritePatternList *patterns) {
  // clang-format off
  patterns->insert<
      ConvertFakeQuantWithMinMaxVarsOp,
      LowerAddNOp,
      LowerBatchToSpaceND,
      LowerDynamicStitchOp<DynamicStitchOp>,
      LowerDynamicStitchOp<ParallelDynamicStitchOp>,
      LowerInvertPermutationOp,
      LowerPackOp,
      LowerResizeNearestNeighbor,
      LowerSoftmaxOp<TF::LogSoftmaxOp, /*use_log=*/true>,
      LowerSoftmaxOp<TF::SoftmaxOp, /*use_log=*/false>,
      LowerSpaceToBatchNDOp,
      LowerSparseMatMulOp,
      Lower_UnaryOpsComposition,
      LowerRollOp>(context);
  // clang-format on

  // Populate the relevant generated patterns.
  // clang-format off
  patterns->insert<
      LowerBiasAddGradOp,
      LowerDivNoNanOp,
      LowerEmptyOp,
      LowerFakeQuantWithMinMaxArgs,
      LowerFillOp,
      LowerIsNanOp,
      LowerL2LossOp,
      LowerMulNoNanOp,
      LowerPadOp,
      LowerReciprocal,
      LowerRintOp,
      LowerRoundOpOnFloatTensor,
      LowerRoundOpOnIntTensor,
      LowerRsqrtGradOp,
      LowerScatterNdOp,
      LowerSeluOp,
      LowerSeluGradOp,
      LowerSizeOp,
      LowerSoftmaxCrossEntropyWithLogitsOp,
      LowerSparseSoftmaxCrossEntropyWithLogitsOp,
      LowerSqrtGradOp,
      LowerSquareOp,
      LowerSquaredDifferenceOpOnRealTensors,
      LowerSquaredDifferenceOpOneComplexTensors,
      LowerTanhGradOp,
      LowerXdivyOp,
      LowerXlog1pyOp,
      LowerXlogyOp>(context);
  // clang-format on
}

void PopulateLoweringQuantizedPatterns(MLIRContext *context,
                                       OwningRewritePatternList *patterns) {
  // clang-format off
  patterns->insert<
      LowerDequantizeOp>(context);
  // clang-format on
}

}  // namespace TF
}  // namespace mlir
