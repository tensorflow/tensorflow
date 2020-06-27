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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/convert_op_folder.h"
#include "tensorflow/compiler/mlir/xla/ir/chlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"
#include "tensorflow/compiler/xla/client/lib/conv_grad_size_util.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace xla_hlo {
namespace {

constexpr char kShardingAttr[] = "xla_hlo.sharding";

class LegalizeTF : public PassWrapper<LegalizeTF, FunctionPass> {
 public:
  LegalizeTF() = default;
  LegalizeTF(const LegalizeTF &) {}
  explicit LegalizeTF(bool allow_partial_conversion, bool legalize_chlo) {
    allow_partial_conversion_ = allow_partial_conversion;
    legalize_chlo_ = legalize_chlo;
  }

  /// Performs the lowering to XLA dialect.
  void runOnFunction() override;

 private:
  Option<bool> allow_partial_conversion_{
      *this, "allow-partial-conversion",
      llvm::cl::desc("Allow operations that can't be legalized."),
      llvm::cl::init(false)};
  Option<bool> legalize_chlo_{
      *this, "legalize-chlo",
      llvm::cl::desc(
          "Also legalizes intermediate chlo ops to hlo (default true)"),
      llvm::cl::init(true)};
};

/// Returns if the given TF data format string is the default format.
static bool IsDefaultDataFormat(StringRef format) { return format == "NHWC"; }

/// Returns the feature dimension for the given format and input type.
static size_t GetFeatureDimension(StringAttr format,
                                  RankedTensorType inputType) {
  return IsDefaultDataFormat(format.getValue()) ? inputType.getRank() - 1 : 1;
}

// Gets all integer values from the given attribute and push them to `values`.
void GetI64ArrayAttrValues(Attribute attr, SmallVectorImpl<int64_t> *values) {
  auto array_attr = attr.cast<ArrayAttr>();
  values->reserve(array_attr.getValue().size());
  for (Attribute val : array_attr.getValue())
    values->push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
}

// Returns 1D 64-bit dense elements attribute with the given values.
static DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                               Builder *builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

// Converts an ArrayAttr to a 1D 64-bit dense elements attribute.
static DenseIntElementsAttr GetI64ElementsAttr(ArrayAttr attr) {
  RankedTensorType ty =
      RankedTensorType::get(static_cast<int64_t>(attr.size()),
                            IntegerType::get(64, attr.getContext()));
  return DenseIntElementsAttr::get(ty, attr.getValue());
}

// Returns 1D 32-bit dense elements attribute with the given values.
static DenseIntElementsAttr GetI32ElementsAttr(ArrayRef<int32_t> values,
                                               Builder *builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int32_t>(values.size())}, builder->getIntegerType(32));
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

// Returns a 1-d i64 elements attribute populated with `val` repeated `size`
// times.
static DenseIntElementsAttr GetI64ElementsAttrForValue(int size, int64_t val,
                                                       Builder *builder) {
  TensorType ty = RankedTensorType::get({size}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, val);
}

// Returns the corresponding type that should be used for performing sum
// accumulation over the given input type.
Type GetSumAccumulationType(Type input_type) {
  MLIRContext *ctx = input_type.getContext();
  if (input_type.isBF16() || input_type.isF16()) return FloatType::getF32(ctx);
  if (input_type.isSignlessInteger(8) || input_type.isSignlessInteger(16))
    return IntegerType::get(32, ctx);
  return input_type;
}

// Returns axis in HLO format from TF elements attr with exactly one element
// containing axis in the TensorFlow format. TensorFlow format supports negative
// indexing unlike HLO.
static IntegerAttr GetHLOAxisFromTFAxis(ElementsAttr attr, int64_t rank,
                                        Builder *b) {
  SmallVector<uint64_t, 1> index(attr.getType().getRank(), 0);
  int64_t axis = attr.getValue<IntegerAttr>(index).getInt();
  if (axis < 0) {
    axis += rank;
  }
  return b->getI64IntegerAttr(axis);
}

static IntegerAttr GetHLOAxisFromTFAxis(IntegerAttr attr, int64_t rank,
                                        Builder *b) {
  int64_t axis = attr.getInt();
  if (axis < 0) {
    axis += rank;
  }
  return b->getI64IntegerAttr(axis);
}

// If `value` is an IntegerAttr, returns the integer value for the HLO axis
// corresponding to the tensorflow axis. In particular, the tensorflow axis can
// be negative, in which case, the corresponding HLO axis is
// (axis + rank-of-the-tensor).
static llvm::Optional<int64_t> GetIntegerHLOAxisFromTFAxis(Value value,
                                                           int64_t rank) {
  DenseIntElementsAttr attrs;
  if (!matchPattern(value, m_Constant(&attrs)) ||
      attrs.getType().getRank() != 0) {
    return llvm::None;
  }
  int64_t axis = attrs.getValue<IntegerAttr>({}).getInt();
  return axis < 0 ? axis + rank : axis;
}

/// Returns a `ConvertOp` that casts the elements to a i64 type while retaining
/// the shape of the input value.
static ConvertOp CastValueToI64(Location loc, Value value,
                                PatternRewriter *rewriter) {
  return rewriter->create<ConvertOp>(loc, value, rewriter->getIntegerType(64));
}

// Creates an unpack op along the 0th dimension of the tensor. The `value` input
// must be a ranked tensor.
static TF::UnpackOp UnpackTensorAlongZeroDim(Location loc, Value value,
                                             PatternRewriter *rewriter) {
  auto indices_type = value.getType().cast<RankedTensorType>();
  int num_outputs = indices_type.getShape().front();
  SmallVector<Type, 2> unpacked_indices_type(
      num_outputs, RankedTensorType::get({}, indices_type.getElementType()));
  auto unpacked_indices = rewriter->create<TF::UnpackOp>(
      loc, unpacked_indices_type, value,
      IntegerAttr::get(rewriter->getIntegerType(64), 0));
  return unpacked_indices;
}

// Returns size of dimension at the specified index, if ranked tensor.
// Otherwise, returns -1.
//
// Aborts if the type is ranked but doesn't have the dimension.
int64_t GetDimSize(Type ty, int64_t index) {
  RankedTensorType ranked_ty = ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return -1;

  return ranked_ty.getDimSize(index);
}

template <typename T, int num_dims>
tensorflow::TensorShape ToTensorShape(llvm::ArrayRef<T> sizes) {
  return tensorflow::TensorShape(llvm::SmallVector<tensorflow::int64, num_dims>(
      sizes.begin(), sizes.end()));
}

template <typename T, int num_dims>
tensorflow::TensorShape ToTensorShape(
    llvm::iterator_range<DenseElementsAttr::ElementIterator<T>> sizes) {
  return tensorflow::TensorShape(llvm::SmallVector<tensorflow::int64, num_dims>(
      sizes.begin(), sizes.end()));
}

// Returns minimal value for the given int or float element type.
static ConstOp GetMinValueForType(Type ty, Location loc,
                                  PatternRewriter *rewriter) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);

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
  return rewriter->create<ConstOp>(loc, attr);
}

// Returns maximal value for the given int or float element type.
static ConstOp GetMaxValueForType(Type ty, Location loc,
                                  PatternRewriter *rewriter) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);

  DenseElementsAttr attr;
  if (auto float_ty = ty.dyn_cast_or_null<FloatType>()) {
    APFloat pos_inf =
        APFloat::getInf(float_ty.getFloatSemantics(), /*negative=*/false);
    attr = DenseElementsAttr::get(scalar_ty, pos_inf);
  } else {
    auto int_ty = ty.cast<IntegerType>();
    APInt max_val = APInt::getSignedMaxValue(int_ty.getWidth());
    attr = DenseElementsAttr::get(scalar_ty, max_val);
  }
  return rewriter->create<ConstOp>(loc, attr);
}

// Returns int or float scalar DenseElementsAttr attribute with the given
// element type and the value.
static ConstOp GetScalarConstOfType(Type ty, Location loc, int64_t raw_value,
                                    OpBuilder *builder) {
  return builder->create<ConstOp>(loc, xla::GetScalarOfType(ty, raw_value));
}

// Creates an xla_hlo::SliceOp where the major dimensions have full size, and
// the minor dimensions have the provided offsets and sizes.
static Value SliceInMinorDims(Location loc, Value v,
                              ArrayRef<int64_t> minor_starts,
                              ArrayRef<int64_t> minor_limits,
                              OpBuilder *builder) {
  auto type = v.getType().cast<RankedTensorType>();
  llvm::SmallVector<int64_t, 4> slice_starts(type.getRank(), 0);
  int64_t major_dims = type.getRank() - minor_starts.size();
  std::copy(minor_starts.begin(), minor_starts.end(),
            slice_starts.begin() + major_dims);
  auto slice_limits = llvm::to_vector<4>(type.getShape());
  std::copy(minor_limits.begin(), minor_limits.end(),
            slice_limits.begin() + major_dims);
  llvm::SmallVector<int64_t, 4> slice_strides(type.getRank(), 1);
  return builder->create<SliceOp>(loc, v,
                                  GetI64ElementsAttr(slice_starts, builder),
                                  GetI64ElementsAttr(slice_limits, builder),
                                  GetI64ElementsAttr(slice_strides, builder));
}

// Creates a vector of index values:
//  [0, 0, ..., minor_indices[0], minor_indices[1], ... minor_indices[-1]]
// with length `rank`.
static llvm::SmallVector<Value, 4> CreateFullIndexVectorFromMinorIndices(
    Location loc, ArrayRef<Value> minor_indices, int64_t rank,
    OpBuilder *builder) {
  auto zero =
      GetScalarConstOfType(getElementTypeOrSelf(minor_indices[0].getType()),
                           loc, 0, builder)
          .output();
  llvm::SmallVector<Value, 4> indices(rank, zero);
  std::copy(minor_indices.begin(), minor_indices.end(),
            indices.begin() + (rank - minor_indices.size()));
  return indices;
}

// Creates an xla_hlo::DynamicSliceOp where the major dimensions have full size,
// and the minor dimensions have the provided offsets and sizes.
static Value DynamicSliceInMinorDims(Location loc, Value v,
                                     ArrayRef<Value> minor_starts,
                                     ArrayRef<int64_t> minor_sizes,
                                     OpBuilder *builder) {
  if (minor_starts.empty()) return v;
  auto type = v.getType().cast<RankedTensorType>();
  auto slice_starts = CreateFullIndexVectorFromMinorIndices(
      loc, minor_starts, type.getRank(), builder);
  int64_t major_dims = type.getRank() - minor_starts.size();
  auto slice_sizes = llvm::to_vector<4>(type.getShape());
  std::copy(minor_sizes.begin(), minor_sizes.end(),
            slice_sizes.begin() + major_dims);
  auto slice_type = RankedTensorType::get(slice_sizes, type.getElementType());
  return builder->create<xla_hlo::DynamicSliceOp>(
      loc, slice_type, v, slice_starts,
      GetI64ElementsAttr(slice_sizes, builder));
}

// Creates an xla_hlo::DynamicUpdateSliceOp where the major dimensions have zero
// offsets, and the minor dimensions have the provided offsets.
static Value DynamicUpdateSliceInMinorDims(Location loc, Value v, Value update,
                                           ArrayRef<Value> minor_starts,
                                           OpBuilder *builder) {
  if (minor_starts.empty()) return v;
  auto type = v.getType().cast<RankedTensorType>();
  auto dus_starts = CreateFullIndexVectorFromMinorIndices(
      loc, minor_starts, type.getRank(), builder);
  return builder->create<DynamicUpdateSliceOp>(loc, type, v, update,
                                               llvm::makeArrayRef(dus_starts));
}

// Creates an xla_hlo::DynamicUpdateSliceOp where the major dimensions have zero
// offsets, and the minor dimensions have the provided static offsets.
static Value UpdateSliceInMinorDims(Location loc, Value v, Value update,
                                    ArrayRef<int64_t> minor_starts,
                                    OpBuilder *builder) {
  llvm::SmallVector<Value, 4> dus_starts(minor_starts.size());
  for (int64_t i = 0; i < minor_starts.size(); ++i) {
    dus_starts[i] = GetScalarConstOfType(builder->getIntegerType(32), loc,
                                         minor_starts[i], builder);
  }
  return DynamicUpdateSliceInMinorDims(loc, v, update, dus_starts, builder);
}

// Deprecated: This is maintained to aid in porting old code that is not yet
// dynamic shape aware and uses broadcasting modes that CHLO does not support.
// Gets the resulting type from a broadcast between two types for statically
// shaped types. This is to be used for legacy lowerings that both use non
// left-padded broadcasting and static shapes. Its use should not be permitted
// in new code.
// May return nullptr on invalid static broadcast dimensions.
// ABSL_DEPRECATED()
static RankedTensorType GetStaticBroadcastType(
    RankedTensorType x, RankedTensorType y,
    DenseIntElementsAttr broadcast_dimensions_attr) {
  auto element_type = x.getElementType();
  auto shape_x = x.getShape();
  auto shape_y = y.getShape();

  if (shape_x.size() == shape_y.size()) {
    llvm::SmallVector<int64_t, 4> out_shape(shape_x.size());
    for (int i = 0; i < shape_x.size(); i++) {
      auto x_val = shape_x[i];
      auto y_val = shape_y[i];
      out_shape[i] = std::max(x_val, y_val);
    }
    return RankedTensorType::get(out_shape, element_type);
  }

  auto shape_large = shape_x.size() > shape_y.size() ? shape_x : shape_y;
  auto shape_small = shape_x.size() <= shape_y.size() ? shape_x : shape_y;

  llvm::SmallVector<int64_t, 4> broadcast_dimensions;
  // Explicit broadcast dimensions.
  for (const APInt &int_value : broadcast_dimensions_attr) {
    broadcast_dimensions.push_back(int_value.getSExtValue());
  }
  if (broadcast_dimensions.size() != shape_small.size()) {
    return nullptr;
  }
  llvm::SmallVector<int64_t, 4> out_shape(shape_large.begin(),
                                          shape_large.end());

  // Update according to the broadcast dimensions.
  for (auto index_pair : llvm::enumerate(broadcast_dimensions)) {
    auto old_value = out_shape[index_pair.value()];
    auto new_value = shape_small[index_pair.index()];
    out_shape[index_pair.value()] = std::max(old_value, new_value);
  }
  return RankedTensorType::get(out_shape, element_type);
}

// Deprecated: This is maintained to aid in porting old code that is not yet
// dynamic shape aware and uses broadcasting modes that CHLO does not support.
// Applies static binary broadcasting to a binary elementwise op.
// This is a legacy helper to provide general broadcasting support in legacy,
// static shaped code that relies on non-left-padded broadcasting semantics.
template <typename BinaryOp>
static Value StaticBinaryBroadcast(Location loc, Value x, Value y,
                                   DenseIntElementsAttr broadcast_dims,
                                   OpBuilder &builder) {
  auto x_type = x.getType().cast<RankedTensorType>();
  auto y_type = y.getType().cast<RankedTensorType>();
  auto result_type = GetStaticBroadcastType(x_type, y_type, broadcast_dims);
  if (!result_type) {
    emitError(loc) << "could not binary broadcast " << x_type << ", " << y_type
                   << " with broadcast_dims = " << broadcast_dims;
    return nullptr;
  }
  auto larger_broadcast_dims =
      GetI64ElementsAttrForSeq(0, result_type.getRank(), &builder);
  if (x_type.getRank() < y_type.getRank()) {
    if (x_type != result_type) {
      x = builder.create<BroadcastInDimOp>(loc, result_type, x, broadcast_dims);
    }
    if (y_type != result_type) {
      y = builder.create<BroadcastInDimOp>(loc, result_type, y,
                                           larger_broadcast_dims);
    }
  } else {
    if (x_type != result_type) {
      x = builder.create<BroadcastInDimOp>(loc, result_type, x,
                                           larger_broadcast_dims);
    }
    if (y_type != result_type) {
      y = builder.create<BroadcastInDimOp>(loc, result_type, y, broadcast_dims);
    }
  }
  return builder.create<BinaryOp>(loc, x, y);
}

// Gets a 1D tensor type suitable for expressing extents of the given tensor
// value type. If the value type is ranked, the result will be statically
// shaped. Otherwise, it will have a dynamic dimension.
static RankedTensorType GetExtentsTensorTypeFor(TensorType value_type) {
  Builder b(value_type.getContext());
  int64_t dim = value_type.hasRank() ? value_type.getRank() : -1;
  return RankedTensorType::get({dim}, b.getIndexType());
}

// Broadcasts a 'lower_rank_value' to the shape of a 'higher_rank_value'
// by assuming that the shape of the lower ranked is a broadcast compatible
// prefix of the higher ranked.
// Values must be RankedTensorType (this restriction derives from the
// broadcast_dimensions attribute on DynamicBroadcastInDim).
//
// Example:
//   CommonPrefixBroadcast(tensor<4x3x256>, tensor<4, 3>) will broadcast the
//   lower rank value to [4, 3, 256] (i.e. the opposite of numpy-style
//   implicit broadcasting).
static Value CommonPrefixBroadcast(Location loc, Value higher_rank_value,
                                   Value lower_rank_value, OpBuilder &builder) {
  Value higher_rank_shape =
      builder.create<shape::ShapeOfOp>(loc, higher_rank_value);
  auto result_extents_type =
      GetExtentsTensorTypeFor(higher_rank_value.getType().cast<TensorType>());
  Value result_extents = builder.create<shape::ToExtentTensorOp>(
      loc, result_extents_type, higher_rank_shape);

  auto lower_rank_type = lower_rank_value.getType().cast<RankedTensorType>();
  auto lower_rank = lower_rank_type.getRank();
  auto prefix_dims = GetI64ElementsAttrForSeq(0, lower_rank, &builder);
  return builder.create<DynamicBroadcastInDimOp>(
      loc, higher_rank_value.getType(), lower_rank_value, result_extents,
      prefix_dims);
}

// Given a value (broadcast_to) and a feature dimension, broadcasts a 1D
// value (broadcast_from) along that feature dimension. This is a shortcut
// for the cases where a 1D tensor must be broadcast along a specific feature
// dimension, which can vary based on data layout, etc.
//
// The extent of `broadcast_from` dim0 must be equal to the extent of the
// feature_dim of `broadcast_to`.
//
// Example:
//   [1x2x3x4], [2], 1 -> [1x2x3x4]
// TODO(laurenzo): Swap the order of broadcast_to and broadcast_from for
// consistency. Possibly also rename for clarity.
static Value Broadcast1DToFeatureDim(Location loc, Value broadcast_to,
                                     Value broadcast_from, int64_t feature_dim,
                                     OpBuilder &builder) {
  auto broadcast_dims = GetI64ElementsAttr({feature_dim}, &builder);
  auto to_type = broadcast_to.getType().cast<RankedTensorType>();
  auto result_shape = builder.create<shape::ShapeOfOp>(loc, broadcast_to);
  auto result_extents_type = GetExtentsTensorTypeFor(to_type);
  auto result_extents = builder.create<shape::ToExtentTensorOp>(
      loc, result_extents_type, result_shape);
  return builder.create<DynamicBroadcastInDimOp>(
      loc, to_type, broadcast_from, result_extents, broadcast_dims);
}

// Broadcasts `input` to the shape of `broadcast_to` value following
// TF::BroadcastTo semantics.
//
// Requires that input is a ranked tensor.
//
// TODO(hinsu): Utilize TF::ShapeOp followed by TF::BroadcastTo once ShapeOp
// supports unranked inputs in the lowering.
static Value BroadcastToShapeOf(Location loc, Value input, Value broadcast_to,
                                OpBuilder &builder) {
  auto result_shape = builder.create<shape::ShapeOfOp>(loc, broadcast_to);
  auto to_type = broadcast_to.getType().cast<TensorType>();
  auto result_extents_type = GetExtentsTensorTypeFor(to_type);
  auto result_extents = builder.create<shape::ToExtentTensorOp>(
      loc, result_extents_type, result_shape);
  int64_t rank = input.getType().cast<RankedTensorType>().getRank();
  auto broadcast_dims = GetI64ElementsAttrForSeq(0, rank, &builder);
  return builder.create<DynamicBroadcastInDimOp>(
      loc, to_type, input, result_extents, broadcast_dims);
}

// Creates a batch dot using xla_hlo::DotGeneralOp.
Value BatchDot(Location loc, Value lhs, bool transpose_lhs, Value rhs,
               bool transpose_rhs, int64_t num_batch_dims,
               ArrayAttr precision_config, OpBuilder *builder) {
  auto batch_dimensions = GetI64ElementsAttr(
      llvm::to_vector<4>(llvm::seq<int64_t>(0, num_batch_dims)), builder);
  auto lhs_contracting_dimensions = GetI64ElementsAttr(
      llvm::makeArrayRef({transpose_lhs ? num_batch_dims : num_batch_dims + 1}),
      builder);
  auto rhs_contracting_dimensions = GetI64ElementsAttr(
      llvm::makeArrayRef({transpose_rhs ? num_batch_dims + 1 : num_batch_dims}),
      builder);
  auto dimension_numbers = DotDimensionNumbers::get(
      /*lhs_batching_dimensions=*/batch_dimensions,
      /*rhs_batching_dimensions=*/batch_dimensions,
      /*lhs_contracting_dimensions=*/lhs_contracting_dimensions,
      /*rhs_contracting_dimensions=*/rhs_contracting_dimensions,
      builder->getContext());
  auto lhs_shape = lhs.getType().cast<RankedTensorType>().getShape();
  auto rhs_shape = rhs.getType().cast<RankedTensorType>().getShape();
  auto shape = llvm::to_vector<4>(lhs_shape);
  shape[shape.size() - 2] =
      transpose_lhs ? lhs_shape.back() : lhs_shape[lhs_shape.size() - 2];
  shape[shape.size() - 1] =
      transpose_rhs ? rhs_shape[rhs_shape.size() - 2] : rhs_shape.back();
  Type element_type = getElementTypeOrSelf(lhs.getType());
  return builder->create<DotGeneralOp>(
      loc, RankedTensorType::get(shape, element_type), lhs, rhs,
      dimension_numbers, precision_config);
}

// Builds body for reduce op by using the template binary op as the
// reducer op.
template <typename Op>
static void BuildReduceBody(Type element_type, Region *body,
                            OpBuilder *builder) {
  OpBuilder::InsertionGuard guard(*builder);
  Block *block = builder->createBlock(body);

  // Block arguments are scalars of the given element type.
  Type type = RankedTensorType::get(/*shape=*/{}, element_type);
  block->addArguments({type, type});

  Location loc = body->getLoc();
  auto reducer =
      builder->create<Op>(loc, block->getArgument(0), block->getArgument(1));
  builder->create<ReturnOp>(loc, reducer.getResult());
}

// Builds region taking two arguments and returning second argument as the
// result. Corresponds to the function f(x, y) = y.
// Used in Scatter op's computation to update specific elements.
static void BuildBinaryAssignmentRegion(Type element_type, Region *region,
                                        OpBuilder *builder) {}

// Builds a set of operations for applying reduction on the input value. A
// tf.sum op is created and will be legalized to tfl ops automatically.
static Value ApplyReduction(Location loc, Value input,
                            DenseIntElementsAttr reduce_dims,
                            OpBuilder *builder) {
  auto reduce_dims_op = builder->create<ConstOp>(loc, reduce_dims);
  return builder->create<TF::SumOp>(loc, input, reduce_dims_op,
                                    builder->getBoolAttr(false));
}

// Creates a xla_hlo.rng_uniform op with `builder` to generate `num_elements`
// 32-bit integer numbers in the range of [`lower_limit`, `upper_limit`).
static xla_hlo::RngUniformOp CreateRngUniform32(Location loc, int num_elements,
                                                int lower_limit,
                                                int upper_limit,
                                                OpBuilder *builder) {
  auto i32_type = builder->getIntegerType(32);
  auto key_type = RankedTensorType::get({num_elements}, i32_type);
  auto shape_tensor = builder->create<xla_hlo::ConstOp>(
      loc, GetI64ElementsAttr({num_elements}, builder));

  auto lower = builder->create<xla_hlo::ConstOp>(
      loc, builder->getI32IntegerAttr(lower_limit));
  auto upper = builder->create<xla_hlo::ConstOp>(
      loc, builder->getI32IntegerAttr(upper_limit));

  return builder->create<xla_hlo::RngUniformOp>(loc, key_type, lower, upper,
                                                shape_tensor);
}

using WhileBodyFnType = llvm::function_ref<void(
    Location loc, Value iteration, ArrayRef<Value> old_values,
    SmallVectorImpl<Value> *new_values, OpBuilder *builder)>;

// Creates a xla_hlo.while op with `builder` to loop `num_interations` times,
// each time calling the given `body_fn` on a set of values to generate a new
// set of values. Returns the final set of values via `final_values`. The
// initial set of values is passed in via `init_values`.
//
// This effectively does:
//
// ```c++
// SmallVector<Values, 4> old_values = init_values;
// SmallVector<Values, 4> new_values;
// for (int i = 0; i < num_iterations; ++i) {
//   body_fn(old_values, &new_values, ...);
//   old_values = new_values;
// }
// ```
//
// Under the hood an induction variable is prepended to values to control the
// number of iterations, but that is transparent to `body_fn`, which does not
// need to care about that.
static void CreateWhile32(Location loc, int num_iterations,
                          WhileBodyFnType body_fn, ArrayRef<Value> init_values,
                          SmallVectorImpl<Value> *final_values,
                          OpBuilder *builder) {
  int value_count = init_values.size() + 1;

  // Prepend a loop induction variable to the initial values.
  SmallVector<Value, 2> init_values_with_loop_iv;
  init_values_with_loop_iv.reserve(value_count);
  // The initial value for the loop induction variable is 0.
  init_values_with_loop_iv.push_back(
      builder->create<xla_hlo::ConstOp>(loc, builder->getI32IntegerAttr(0)));
  init_values_with_loop_iv.append(init_values.begin(), init_values.end());

  // Prepare the initial tuple for the while op.
  auto init_tuple =
      builder->create<xla_hlo::TupleOp>(loc, init_values_with_loop_iv);
  auto tuple_type = init_tuple.getType();

  // Create the while op.
  auto while_op = builder->create<xla_hlo::WhileOp>(loc, init_tuple);

  {
    OpBuilder::InsertionGuard guard(*builder);

    // Build up the only block in the condition region. It should take one
    // argument of the loop's tuple type.
    Region &condition = while_op.cond();
    Block *block = builder->createBlock(&condition);
    BlockArgument arg = block->addArgument(tuple_type);

    // Get the loop induction variable and compare it against the upper limit.
    auto loop_iv = builder->create<GetTupleElementOp>(loc, arg, 0);
    auto upper_limit = builder->create<xla_hlo::ConstOp>(
        loc, builder->getI32IntegerAttr(num_iterations));
    StringAttr compare_direction = StringAttr::get("LT", builder->getContext());
    Value compare = builder->create<xla_hlo::CompareOp>(
        loc, loop_iv, upper_limit, compare_direction);

    builder->create<xla_hlo::ReturnOp>(loc, compare);
  }

  {
    OpBuilder::InsertionGuard guard(*builder);

    // Build up the only block in the body region. It should take one
    // argument of the loop's tuple type.
    Region &body = while_op.body();
    Block *block = builder->createBlock(&body);
    BlockArgument arg = block->addArgument(tuple_type);

    SmallVector<Value, 4> old_values;  // From the previous iteration
    SmallVector<Value, 4> new_values;  // Generated by this iteration
    old_values.reserve(value_count);
    new_values.reserve(value_count);

    // Unpack the tuple value from the last iteration.
    for (int i = 0; i < value_count; ++i)
      old_values.push_back(builder->create<GetTupleElementOp>(loc, arg, i));

    // Feed all values excluding the loop induction variable to body_fn.
    body_fn(loc, old_values[0], llvm::makeArrayRef(old_values).drop_front(),
            &new_values, builder);

    // Increment the loop induction variable by one.
    auto one =
        builder->create<xla_hlo::ConstOp>(loc, builder->getI32IntegerAttr(1));
    auto scalar_broadcast_dims = GetI64ElementsAttr({}, builder);
    auto plus_one = builder->create<xla_chlo::BroadcastAddOp>(
        loc, old_values[0], one, scalar_broadcast_dims);
    // Prepend with the updated loop induction variable.
    new_values.insert(new_values.begin(), plus_one);

    Value updated_tuple = builder->create<xla_hlo::TupleOp>(loc, new_values);

    builder->create<xla_hlo::ReturnOp>(loc, updated_tuple);
  }

  final_values->reserve(init_values.size());
  for (int i = 0, e = init_values.size(); i < e; ++i)
    final_values->push_back(
        builder->create<GetTupleElementOp>(loc, while_op, i + 1));
}

//===----------------------------------------------------------------------===//
// BatchNorm op utilities.
//===----------------------------------------------------------------------===//

static IntegerAttr getFeatureDimensionAttr(Builder &b, StringAttr format,
                                           Value input) {
  return b.getI64IntegerAttr(
      GetFeatureDimension(format, input.getType().cast<RankedTensorType>()));
}

//===----------------------------------------------------------------------===//
// MatMul op utilities.
//===----------------------------------------------------------------------===//

// If the 'transpose' attribute is true returns ElementsAttr to transpose 2D
// matrix. Otherwise, returns ElementsAttr for identity transpose.
static DenseIntElementsAttr Get2DTransposePerm(BoolAttr transpose, Builder *b) {
  if (transpose.getValue()) return GetI64ElementsAttr({1, 0}, b);
  return GetI64ElementsAttr({0, 1}, b);
}

//===----------------------------------------------------------------------===//
// MatrixBandPart op utilities.
//===----------------------------------------------------------------------===//

// Gets the size of the dimension `dim_from_end` from the end of `input`.
// Requires that `input` is a tensor.
static int GetDimensionSizeFromEnd(Value input, int dim_from_end) {
  // Note: the verifier enforces that `input` is a ranked tensor.
  auto input_type = input.getType().cast<TensorType>();
  auto input_shape = input_type.getShape();
  int dim = (input_shape.size() - 1) - dim_from_end;
  return input_shape[dim];
}

// Gets a 2D tensor type with shape {dim_0, dim_1}, where `dim_0` and `dim_1`
// have the same size as the last two dimensions of `input` (the second-to-last
// dimension and last dimension, respectively). The element type of the
// outputted RankedTensorType will match the element type of `input`.
// Requires that `input` is a tensor.
static RankedTensorType Get2DTensorType(Value input) {
  // `dim_0` refers to the second-to-last dimension; `dim_1` refers to the last.
  int dim_0 = GetDimensionSizeFromEnd(input, 1);
  int dim_1 = GetDimensionSizeFromEnd(input, 0);
  auto element_type = input.getType().cast<TensorType>().getElementType();
  return RankedTensorType::get({dim_0, dim_1}, element_type);
}

// Creates a HLO ConvertOp, converting `input` to have the same element type as
// `elem_type_tensor`. Requires `elem_type_tensor` to be a tensor.
static Value CreateConvertOp(OpBuilder *builder, Location loc, Value input,
                             Value elem_type_tensor) {
  auto element_type =
      elem_type_tensor.getType().cast<TensorType>().getElementType();
  return builder->create<xla_hlo::ConvertOp>(loc, input, element_type);
}

//===----------------------------------------------------------------------===//
// Pad op utilities.
//===----------------------------------------------------------------------===//

// Slices input attribute of rank two and returns the specified column.
//
// Always returns 64 bit integer attribute regardless of bitwidth of the input
// attribute.
static DenseIntElementsAttr SliceDenseIntElementsAttrColumn2D(
    ElementsAttr input, int column) {
  auto int_attr = input.cast<DenseIntElementsAttr>();
  auto shaped_type = int_attr.getType();
  auto shape = shaped_type.getShape();

  if (shape.size() != 2) return DenseIntElementsAttr();

  llvm::SmallVector<int64_t, 4> values;
  values.reserve(shaped_type.getNumElements() / shape[1]);

  for (auto it : llvm::enumerate(int_attr.getIntValues())) {
    if (it.index() % shape[1] == column) {
      values.push_back(it.value().getSExtValue());
    }
  }

  auto element_type = IntegerType::get(64, input.getContext());
  return DenseIntElementsAttr::get(
      RankedTensorType::get({shape[0]}, element_type), values);
}

// Returns interior padding to use in HLO Pad op based on the TensorFlow padding
// in TensorFlow PadV2 op.
static DenseIntElementsAttr GetInteriorPadding(ElementsAttr tf_padding) {
  auto length = tf_padding.getType().getShape()[0];
  auto element_type = IntegerType::get(64, tf_padding.getContext());
  return DenseIntElementsAttr::get<int64_t>(
      RankedTensorType::get({length}, element_type), 0);
}

//===----------------------------------------------------------------------===//
// Binary op utilities.
//===----------------------------------------------------------------------===//

// Returns whether the two values are guaranteed to be broadcastable to the
// same shape, this broadcasts size 1 tensors up to any rank. Dynamic dimensions
// must be broadcasted with a size 1 tensor or another dynamic dimension.
// Returns false on rankless.
static bool AreBroadcastCompatible(Value x, Value y) {
  auto x_rankless = x.getType().dyn_cast<RankedTensorType>();
  auto y_rankless = y.getType().dyn_cast<RankedTensorType>();
  if (!x_rankless || !y_rankless) {
    return false;
  }

  // Check that the shapes can be broadcasted.
  auto shape_x = x_rankless.getShape();
  auto shape_y = y_rankless.getShape();

  int rank_diff = shape_x.size() - shape_y.size();
  int offset_x = rank_diff > 0 ? rank_diff : 0;
  int offset_y = rank_diff < 0 ? -rank_diff : 0;
  for (int i = 0, s = std::min(shape_x.size(), shape_y.size()); i < s; i++) {
    int index_x = i + offset_x;
    int index_y = i + offset_y;
    if ((shape_x[index_x] == -1 && shape_y[index_y] != 1) ||
        (shape_y[index_y] == -1 && shape_x[index_x] != 1)) {
      return false;
    }
  }

  return true;
}

// Return a new TensorType the same rank and dimensions as the input with an
// updated element type.
static Type ChangeTensorElementType(Builder *b, Type tensor_type,
                                    Type element_type) {
  RankedTensorType ranked_type = tensor_type.dyn_cast<RankedTensorType>();
  if (ranked_type) {
    return RankedTensorType::get(ranked_type.getShape(), element_type);
  }

  return UnrankedTensorType::get(element_type);
}

//===----------------------------------------------------------------------===//
// Softmax op utilities.
//===----------------------------------------------------------------------===//

// Returns the type to use for accumulating the given type.
static Type GetAccumulationType(Type ty) {
  // Upcast 16 bit sum reductions to 32 bit to reduce the precision loss from
  // repeated floating point additions.
  return (ty.isF16() || ty.isBF16()) ? FloatType::getF32(ty.getContext()) : ty;
}

//===----------------------------------------------------------------------===//
// Softplus op utilities.
//===----------------------------------------------------------------------===//

static DenseElementsAttr GetEpsilonValue(Type ty) {
  auto element_ty = ty.cast<TensorType>().getElementType();
  auto scalar_ty = RankedTensorType::get({}, element_ty);
  if (element_ty.isF16()) {
    uint16_t raw_epsilon = Eigen::NumTraits<Eigen::half>::epsilon().x;
    auto value = APFloat(APFloat::IEEEhalf(), APInt(16, raw_epsilon));
    return DenseElementsAttr::get(scalar_ty, value);
  } else if (element_ty.isBF16()) {
    uint16_t raw_epsilon = tensorflow::bfloat16::epsilon().value;
    auto value = APFloat(APFloat::BFloat(), APInt(16, raw_epsilon));
    return DenseElementsAttr::get(scalar_ty, value);
  } else if (element_ty.isF32()) {
    auto value = APFloat(std::numeric_limits<float>::epsilon());
    return DenseElementsAttr::get(scalar_ty, value);
  } else if (element_ty.isF64()) {
    auto value = APFloat(std::numeric_limits<double>::epsilon());
    return DenseElementsAttr::get(scalar_ty, value);
  }
  llvm_unreachable("unsupported element type for tf.SoftPlus");
}

//===----------------------------------------------------------------------===//
// ArgMax/ArgMin op utilities.
//===----------------------------------------------------------------------===//

static void BuildArgMinMaxReductionBody(Type input_element_type,
                                        Type index_element_type,
                                        StringRef direction, Region *body,
                                        OpBuilder *builder) {
  OpBuilder::InsertionGuard insertion_point_gurad(*builder);

  Type input_type = RankedTensorType::get(/*shape=*/{}, input_element_type);
  Type index_type = RankedTensorType::get(/*shape=*/{}, index_element_type);
  Block *block = builder->createBlock(body);
  block->addArguments({input_type, index_type, input_type, index_type});

  Location loc = body->getLoc();
  StringAttr compare_direction =
      StringAttr::get(direction, builder->getContext());
  Value compare = builder->create<CompareOp>(
      loc, block->getArgument(0), block->getArgument(2), compare_direction);

  Value selected_input = builder->create<SelectOp>(
      loc, input_type, compare, block->getArgument(0), block->getArgument(2));
  Value selected_index = builder->create<SelectOp>(
      loc, index_type, compare, block->getArgument(1), block->getArgument(3));

  Value return_values[] = {selected_input, selected_index};
  builder->create<ReturnOp>(loc, return_values);
}

//===----------------------------------------------------------------------===//
// PartitionedCall op utilities.
//===----------------------------------------------------------------------===//

// Verify that the arguments to be passed into the function are the same types
// as the function paramter types.
static bool ArgTypesMatchCallee(mlir::Operation *op, OperandRange args,
                                SymbolRefAttr func) {
  auto module = op->getParentOfType<ModuleOp>();
  auto function =
      dyn_cast_or_null<FuncOp>(SymbolTable::lookupSymbolIn(module, func));
  FunctionType function_ty = function.getType();

  for (auto arg_in : llvm::zip(args, function_ty.getInputs())) {
    if (std::get<0>(arg_in).getType() != std::get<1>(arg_in)) {
      // Argument type and input type mismatch.
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Slice op utilities.
//===----------------------------------------------------------------------===//

static bool CanBeTranslatedToDynamicSlice(Value input, Value start_indices,
                                          DenseIntElementsAttr slice_sizes) {
  auto input_ty = input.getType().dyn_cast<RankedTensorType>();
  if (!input_ty) return false;
  auto start_indices_ty = start_indices.getType().dyn_cast<RankedTensorType>();
  if (!start_indices_ty) return false;

  int64_t input_rank = input_ty.getRank();
  ArrayRef<int64_t> input_shape = input_ty.getShape();
  DenseIntElementsAttr constant_start_indices;
  bool is_constant_start =
      matchPattern(start_indices, m_Constant(&constant_start_indices));

  for (int64_t i = 0; i < input_rank; ++i) {
    int64_t input_size = input_shape[i];
    int64_t slice_size = slice_sizes.getValue<IntegerAttr>(i).getInt();
    // A slice_size of -1 means "all elements from start_index to the end".
    // In order to support these semantics, we need to know both the start index
    // and the shape of the input dimension.
    if (slice_size < 0 && (!is_constant_start || input_size < 0)) return false;
  }
  return true;
}

// TF slice size can be -1, which represents all elements from start_index to
// the end. HLO slice size can't be -1. As such, we need to translate TF slice
// size -1 to HLO slice size.
static DenseIntElementsAttr TFSliceSizes2HLOSliceSizes(
    Value input, Value start_indices, DenseIntElementsAttr slice_sizes,
    Builder *builder) {
  DenseIntElementsAttr constant_start_indices;
  if (!matchPattern(start_indices, m_Constant(&constant_start_indices))) {
    return xla::ConvertElementsAttr(slice_sizes, builder->getIntegerType(64))
        .cast<DenseIntElementsAttr>();
  }

  auto input_ty = input.getType().dyn_cast<RankedTensorType>();
  int64_t input_rank = input_ty.getRank();
  ArrayRef<int64_t> input_shape = input_ty.getShape();
  SmallVector<int64_t, 4> normalized_sizes;

  for (int64_t i = 0; i < input_rank; ++i) {
    int64_t input_size = input_shape[i];
    int64_t start_index =
        constant_start_indices.getValue<IntegerAttr>(i).getInt();
    int64_t slice_size = slice_sizes.getValue<IntegerAttr>(i).getInt();
    normalized_sizes.push_back(slice_size == -1 ? input_size - start_index
                                                : slice_size);
  }

  return GetI64ElementsAttr(normalized_sizes, builder);
}

//===----------------------------------------------------------------------===//
// Sort op utilities.
//===----------------------------------------------------------------------===//

// Builds the region `body` for xla_hlo.sort's comparator: for each type in
// `element_types`, create two block arguments, one for lhs and one for rhs, and
// generates xla_hlo.compare op to compare them with the given `direction`.
//
// Note that this right now only does comparision on the first pair of block
// arguments.
static void BuildSortComparisonBody(llvm::ArrayRef<Type> element_types,
                                    StringRef direction, Region *body,
                                    OpBuilder *builder) {
  OpBuilder::InsertionGuard insertion_point_gurad(*builder);

  Block *block = builder->createBlock(body);
  // Add two arguments for each element type.
  for (Type element_type : element_types) {
    TensorType tensor_type = RankedTensorType::get({}, element_type);
    block->addArguments({tensor_type, tensor_type});
  }

  Location loc = body->getLoc();
  StringAttr compare_direction =
      StringAttr::get(direction, builder->getContext());
  Value compare = builder->create<xla_hlo::CompareOp>(
      loc, block->getArgument(0), block->getArgument(1), compare_direction);

  builder->create<xla_hlo::ReturnOp>(loc, compare);
}

//===----------------------------------------------------------------------===//
// Op converters.
//===----------------------------------------------------------------------===//

NamedAttribute GetConvDimensionNumbersAttr(
    ArrayRef<int64_t> spatial_dim_indices, tensorflow::TensorFormat format,
    Builder *builder) {
  int64_t num_spatial_dims = spatial_dim_indices.size();
  int64_t num_dims = num_spatial_dims + 2;

  IntegerAttr batch_dim =
      builder->getI64IntegerAttr(GetTensorBatchDimIndex(num_dims, format));
  IntegerAttr feature_dim =
      builder->getI64IntegerAttr(GetTensorFeatureDimIndex(num_dims, format));
  DenseIntElementsAttr spatial_dims =
      GetI64ElementsAttr(spatial_dim_indices, builder);

  // Filters data_format is always HWIO so input channels dimension is after
  // all spatial dimensions.
  IntegerAttr kernel_input_feature_dim =
      builder->getI64IntegerAttr(num_spatial_dims);
  IntegerAttr kernel_output_feature_dim =
      builder->getI64IntegerAttr(num_spatial_dims + 1);
  DenseIntElementsAttr kernel_spatial_dimensions =
      GetI64ElementsAttrForSeq(0, num_spatial_dims, builder);

  return builder->getNamedAttr(
      "dimension_numbers",
      ConvDimensionNumbers::get(
          batch_dim, feature_dim, spatial_dims, kernel_input_feature_dim,
          kernel_output_feature_dim, kernel_spatial_dimensions, batch_dim,
          feature_dim, spatial_dims, builder->getContext()));
}

// Converts a TF::BiasAddOp to HLO.
// This differs from a normal TF::AddOp with respect to how the data_format
// is handled, which can optionally require a general broadcast of the
// 'bias' term in a way that is not compatible with the standard left-padded
// broadcast semantics (i.e. NCHW will broadcast into dimension 1).
// The correct 'bias' broadcast will be synthesized manually.
class ConvertBiasAddOp : public OpRewritePattern<TF::BiasAddOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::BiasAddOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto feature_dim = GetFeatureDimension(
        op.data_formatAttr(), op.value().getType().cast<RankedTensorType>());
    auto bias_broadcast = Broadcast1DToFeatureDim(loc, op.value(), op.bias(),
                                                  feature_dim, rewriter);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.value(), bias_broadcast);
    return success();
  }
};

// Converts the TensorFlow conv op in template to the generic HLO conv op by
// converting TensorFlow op attributes to HLO op attributes.
//
// Sample result for Conv2D:
//
//   %conv = "xla_hlo.convolution"(%input, %filter) {
//     strides = [1, 2],
//     paddings = [[1, 0], [1, 1]],
//     ...
//   }
//
// This pattern is not defined using declarative rewrite rules as computation of
// the paddings attribute anyway requires multiple source op attributes and
// result op attributes. Defining it as declarative rewrite rule will introduce
// some duplication in the C++ helper methods.
template <typename OpTy, int num_spatial_dims, bool depthwise_conv = false>
class ConvertConvOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.data_format().str(), &data_format))
      return failure();

    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.padding().str(), &padding).ok())
      return failure();

    auto input_ty = op.input().getType().template dyn_cast<RankedTensorType>();
    auto filter_ty =
        op.filter().getType().template dyn_cast<RankedTensorType>();
    auto result_ty = op.getType().template dyn_cast<RankedTensorType>();

    // Input, filter and the result needs to have static shape for calculation
    // of HLO paddings and feature group count attributes.
    for (RankedTensorType ty : {input_ty, filter_ty, result_ty})
      if (!ty || !ty.hasStaticShape()) return failure();

    ArrayRef<Attribute> dilations = op.dilations().getValue();
    ArrayRef<Attribute> strides = op.strides().getValue();
    ArrayRef<Attribute> explicit_paddings;
    if (padding == tensorflow::Padding::EXPLICIT) {
      // EXPLICIT padding mode and the associated attribute is limited to
      // Conv2D. So, fetch attribute by identifier instead of the
      // op.explicit_paddings() attribute getter.
      explicit_paddings =
          op.template getAttrOfType<ArrayAttr>("explicit_paddings").getValue();
    }

    SmallVector<int64_t, num_spatial_dims> spatial_dim_indices;
    SmallVector<int64_t, num_spatial_dims> rhs_dilations;
    SmallVector<int64_t, num_spatial_dims> window_strides;
    SmallVector<int64_t, num_spatial_dims * 2> paddings;

    auto get_int = [](Attribute attr) {
      return attr.template cast<IntegerAttr>().getInt();
    };

    constexpr int num_dims = num_spatial_dims + 2;
    for (auto i : llvm::seq<int>(0, num_spatial_dims)) {
      const int64_t dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      spatial_dim_indices.push_back(dim);

      const int64_t dilation = get_int(dilations[dim]);
      rhs_dilations.push_back(dilation);
      const int64_t stride = get_int(strides[dim]);
      window_strides.push_back(stride);

      int64_t pad_low, pad_high;
      if (padding == tensorflow::Padding::EXPLICIT) {
        pad_low = get_int(explicit_paddings[2 * dim]);
        pad_high = get_int(explicit_paddings[2 * dim + 1]);
      } else {
        tensorflow::int64 output_size;
        tensorflow::int64 pad_low_int64;
        tensorflow::int64 pad_high_int64;
        tensorflow::Status status = tensorflow::GetWindowedOutputSizeVerboseV2(
            input_ty.getDimSize(dim), filter_ty.getDimSize(i), dilation, stride,
            padding, &output_size, &pad_low_int64, &pad_high_int64);
        if (!status.ok()) return failure();
        pad_low = pad_low_int64;
        pad_high = pad_high_int64;
      }
      paddings.push_back(pad_low);
      paddings.push_back(pad_high);
    }

    auto rhs_dilations_attr = rewriter.getNamedAttr(
        "rhs_dilation", GetI64ElementsAttr(rhs_dilations, &rewriter));

    auto window_strides_attr = rewriter.getNamedAttr(
        "window_strides", GetI64ElementsAttr(window_strides, &rewriter));

    auto dimension_numbers_attr = GetConvDimensionNumbersAttr(
        spatial_dim_indices, data_format, &rewriter);

    const int64_t input_channels =
        GetDimSize(input_ty, GetTensorFeatureDimIndex(num_dims, data_format));
    // Filters data_format is always HWIO so input channels dimension is after
    // all spatial dimensions.
    const int64_t filter_channels = GetDimSize(filter_ty, num_spatial_dims);
    // TensorFlow convolution op verifies that the number of input channels is
    // divisible by the number of filter channels.
    // For depthwise convolution the feature_group_count argument would be set
    // to the input feature dimension.
    const int64_t feature_group_count =
        depthwise_conv ? input_channels : input_channels / filter_channels;
    auto feature_group_count_attr = rewriter.getNamedAttr(
        "feature_group_count", rewriter.getI64IntegerAttr(feature_group_count));

    auto batch_group_count_attr = rewriter.getNamedAttr(
        "batch_group_count", rewriter.getI64IntegerAttr(1));

    RankedTensorType paddings_ty = RankedTensorType::get(
        {num_spatial_dims, 2}, rewriter.getIntegerType(64));
    auto paddings_attr = rewriter.getNamedAttr(
        "padding", DenseElementsAttr::get<int64_t>(paddings_ty, paddings));

    SmallVector<Value, 2> operands(op.getOperands());
    // Reshape the filter to {spatial_dims...., 1,in_channels *
    // channel_multiplier}
    if (depthwise_conv) {
      ArrayRef<int64_t> filter_shape = filter_ty.getShape();
      llvm::SmallVector<int64_t, num_dims> new_shape(
          filter_shape.begin(), filter_shape.begin() + num_spatial_dims);
      new_shape.push_back(1);
      new_shape.push_back(filter_shape[num_spatial_dims] *
                          filter_shape[num_spatial_dims + 1]);
      operands[1] = rewriter.create<xla_hlo::ReshapeOp>(
          op.getLoc(),
          RankedTensorType::get(new_shape, filter_ty.getElementType()),
          operands[1]);
    }
    NamedAttribute attrs[] = {rhs_dilations_attr,     window_strides_attr,
                              dimension_numbers_attr, feature_group_count_attr,
                              batch_group_count_attr, paddings_attr};
    rewriter.replaceOpWithNewOp<ConvOp>(op, op.getType(), operands,
                                        llvm::makeArrayRef(attrs));
    return success();
  }
};

using ConvertConv2DOp = ConvertConvOp<TF::Conv2DOp, /*num_spatial_dims=*/2>;
using ConvertConv3DOp = ConvertConvOp<TF::Conv3DOp, /*num_spatial_dims=*/3>;
using ConvertDepthConv2DOp =
    ConvertConvOp<TF::DepthwiseConv2dNativeOp, /*num_spatial_dims=*/2,
                  /*depthwise_conv=*/true>;

// Converts BF16 FloorDiv op to have casting operators on either end as BF16
// division can result in strange behavior.
//
//      floordiv = cast(floordiv(cast(left), cast(right))))
//
//   %left_cast = cast(%left)
//   %right_cast = cast(%right)
//   %div = div(%left, %left)
//   %floored = floor(%div)
//   %floored_cast = cast(%floored)
//
// Required to manually specify the intermediate types.
class ConvertBF16FloorDivOp : public OpRewritePattern<TF::FloorDivOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::FloorDivOp op,
                                PatternRewriter &rewriter) const override {
    auto l = op.x();
    auto r = op.y();
    auto element_type = getElementTypeOrSelf(l.getType());
    if (!element_type.isBF16()) return failure();

    auto out_type = op.z().getType().cast<TensorType>();

    l = rewriter.create<ConvertOp>(op.getLoc(), l, rewriter.getF32Type());
    r = rewriter.create<ConvertOp>(op.getLoc(), r, rewriter.getF32Type());

    auto intermediate = rewriter.create<TF::FloorDivOp>(
        op.getLoc(),
        ChangeTensorElementType(&rewriter, out_type, rewriter.getF32Type()), l,
        r);

    auto floor_op =
        rewriter.create<ConvertOp>(op.getLoc(), out_type, intermediate);
    rewriter.replaceOp(op, floor_op.getResult());
    return success();
  }
};

class ConvertBroadcastToOp : public OpRewritePattern<TF::BroadcastToOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::BroadcastToOp op,
                                PatternRewriter &rewriter) const override {
    auto input_type = op.input().getType().dyn_cast<RankedTensorType>();
    auto output_type = op.output().getType().dyn_cast<RankedTensorType>();
    if (!input_type || !output_type) {
      return rewriter.notifyMatchFailure(op, "requires ranked shape");
    }
    auto rank_diff = output_type.getRank() - input_type.getRank();
    // The tf.BroadcastTo op performs "right-aligned" numpy-style broadcasting.
    auto broadcast_dimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(rank_diff, output_type.getRank()));
    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        op, output_type, op.input(), op.shape(),
        rewriter.getI64TensorAttr(broadcast_dimensions));
    return success();
  }
};

// Converts TensorFlow DiagPartOp to HLO ops using reduction on masked matrix.
// For a Rank-2 input, it creates the following ops:
//   %1 = "xla_hlo.iota"() {iota_dimension = 0 : i64}
//   %2 = "xla_hlo.iota"() {iota_dimension = 1 : i64}
//   %3 = "xla_hlo.compare"(%1, %2) {comparison_direction = "EQ"}
//   %4 = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
//   %5 = "xla_hlo.broadcast"(%4)
//   %6 = "xla_hlo.select"(%3, %input, %5)
//   %7 = "xla_hlo.reduce"(%6, %4) ( {
//   ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
//     %9 = xla_hlo.add %arg1, %arg2 : tensor<f32>
//     "xla_hlo.return"(%9) : (tensor<f32>) -> ()
//   }) {dimensions = dense<0> : tensor<1xi64>}
//
// If the input's rank N is greater than 2, we will reshape it to R2 first and
// create the above ops, then reshape it back to rank N/2.
class ConvertDiagPartOp : public OpRewritePattern<TF::DiagPartOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::DiagPartOp op,
                                PatternRewriter &rewriter) const override {
    auto input_type = op.input().getType().dyn_cast<RankedTensorType>();
    if (!input_type || !input_type.hasStaticShape()) return failure();
    int64_t num_dims = input_type.getRank();
    if (num_dims < 2 || num_dims % 2 != 0) return failure();
    const int64_t out_dims = num_dims / 2;

    int64_t new_size = 1;
    llvm::SmallVector<int64_t, 4> new_dims;
    for (int i = 0; i < out_dims; i++) {
      if (input_type.getDimSize(i) != input_type.getDimSize(i + out_dims))
        return op.emitOpError("invalid dimensions size");
      new_size *= input_type.getDimSize(i);
      new_dims.push_back(input_type.getDimSize(i));
    }
    Value reshaped_input = rewriter.create<xla_hlo::ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get({new_size, new_size},
                              input_type.getElementType()),
        op.input());
    auto iota_type = RankedTensorType::get({new_size, new_size},
                                           rewriter.getIntegerType(32));
    auto iota0 = rewriter.create<IotaOp>(op.getLoc(), iota_type,
                                         rewriter.getI64IntegerAttr(0));
    auto iota1 = rewriter.create<IotaOp>(op.getLoc(), iota_type,
                                         rewriter.getI64IntegerAttr(1));
    Value compare = rewriter.create<CompareOp>(
        op.getLoc(), iota0, iota1,
        StringAttr::get("EQ", rewriter.getContext()));
    Value zero = GetScalarConstOfType(input_type.getElementType(), op.getLoc(),
                                      0, &rewriter);
    Value zero_matrix = rewriter.create<BroadcastOp>(
        op.getLoc(), reshaped_input.getType(), zero,
        GetI64ElementsAttr({new_size, new_size}, &rewriter));
    Value masked =
        rewriter.create<SelectOp>(op.getLoc(), reshaped_input.getType(),
                                  compare, reshaped_input, zero_matrix);
    auto reduce = rewriter.create<ReduceOp>(op.getLoc(), masked, zero,
                                            GetI64ElementsAttr({0}, &rewriter));
    assert(!input_type.getElementType().isInteger(1) &&
           "data type should not be i1");
    BuildReduceBody<AddOp>(input_type.getElementType(), &reduce.body(),
                           &rewriter);
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        op, RankedTensorType::get(new_dims, input_type.getElementType()),
        reduce.getResult(0));
    return success();
  }
};

// Converts TensorFlow EinsumOp to either HLO EinsumOp or UnaryEinsumOp
// depending on arity of the op.
class ConvertEinsumOp : public OpRewritePattern<TF::EinsumOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::EinsumOp op,
                                PatternRewriter &rewriter) const override {
    StringAttr equation = op.getAttrOfType<StringAttr>("equation");
    if (op.N() == 1) {
      rewriter.replaceOpWithNewOp<UnaryEinsumOp>(
          op, op.getType(), *op.inputs().begin(), equation);
    } else if (op.N() == 2) {
      ValueRange inputs = op.inputs();
      rewriter.replaceOpWithNewOp<EinsumOp>(op, op.getType(), inputs[0],
                                            inputs[1], equation);
    } else {
      // TensorFlow EinsumOp verifies that the number of operands are at most
      // two.
      return failure();
    }
    return success();
  }
};

// The base class to convert TensorFlow FusedBatchNormGrad*Op to HLO
// BatchNormGradOp for training and a sequence of binary ops for inference.
// TODO(b/145536565): move to legalize_tf_patterns.td if it applies.
template <typename FusedBatchNormGradOpT>
class ConvertFusedBatchNormGradBase
    : public OpRewritePattern<FusedBatchNormGradOpT> {
 public:
  using OpRewritePattern<FusedBatchNormGradOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(FusedBatchNormGradOpT op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value grad = op.y_backprop();
    Value act = op.x();
    Value scale = op.scale();
    Value mean = op.reserve_space_1();
    Value var = op.reserve_space_2();

    // TODO(b/141785544): Update this to not require static shapes.
    // activation shape needs to be static to convert negative indices in
    // TensorFlow to absolute indices required by HLO.
    RankedTensorType act_type =
        act.getType().template dyn_cast<RankedTensorType>();
    if (!act_type) return failure();
    Type act_ele_type = act_type.getElementType();
    // To support mixed precision, the statistics type, which maybe more
    // precise than the input types, are used for this op.
    Type kernel_type =
        scale.getType().template cast<TensorType>().getElementType();
    grad = rewriter.create<ConvertOp>(loc, grad, kernel_type);
    act = rewriter.create<ConvertOp>(loc, act, kernel_type);

    auto feature_dim_attr =
        getFeatureDimensionAttr(rewriter, op.data_formatAttr(), act);
    auto feature_dim = feature_dim_attr.getValue().getSExtValue();

    // Gets the result values.
    Value x_backprop, scale_backprop, offset_backprop;
    if (op.is_training()) {  // training
      // TODO(b/145536565): handle GPU logic separately.
      // Infers the output type with the converted `act`.
      Type feature_type = RankedTensorType::get(
          {GetDimSize(act_type, feature_dim)}, kernel_type);
      Type result_type = TupleType::get(
          {act.getType(), feature_type, feature_type}, rewriter.getContext());

      auto training_op = rewriter.create<BatchNormGradOp>(
          loc, result_type, act, scale, mean, var, grad, op.epsilon(),
          feature_dim_attr.getValue());

      x_backprop =
          rewriter.create<GetTupleElementOp>(loc, training_op.getResult(), 0);

      scale_backprop =
          rewriter.create<GetTupleElementOp>(loc, training_op.getResult(), 1);

      offset_backprop =
          rewriter.create<GetTupleElementOp>(loc, training_op.getResult(), 2);
    } else {  // inference
      SmallVector<int64_t, 4> non_feature_dims;
      for (int64_t i = 0; i < act_type.getRank(); ++i) {
        if (i == feature_dim) continue;
        non_feature_dims.push_back(i);
      }
      auto reduce_dims = GetI64ElementsAttr(non_feature_dims, &rewriter);
      auto scalar_broadcast_dims = GetI64ElementsAttr({}, &rewriter);

      // scratch1 = rsqrt(var + epsilon)
      RankedTensorType scalar_float = RankedTensorType::get({}, kernel_type);
      auto epsilon = rewriter.create<ConstOp>(
          loc, DenseFPElementsAttr::get(scalar_float, {op.epsilon()}));
      auto add_op = rewriter.create<xla_chlo::BroadcastAddOp>(
          loc, var, epsilon.getResult(), scalar_broadcast_dims);

      Value scratch1 = rewriter.create<RsqrtOp>(loc, add_op);

      // scratch2 = sum(y_backprop * (x - mean))
      auto sub_op = rewriter.create<xla_hlo::SubOp>(
          loc, act,
          Broadcast1DToFeatureDim(loc, act, mean, feature_dim, rewriter));
      auto weighted_grad = rewriter.create<xla_hlo::MulOp>(loc, grad, sub_op);
      Value scratch2 =
          ApplyReduction(loc, weighted_grad, reduce_dims, &rewriter);

      // x_backprop = y_backprop * (scale * scratch1)
      auto scaled_grad =
          rewriter.create<xla_hlo::MulOp>(loc, op.scale(), scratch1);
      x_backprop = rewriter.create<xla_hlo::MulOp>(
          loc, grad,
          Broadcast1DToFeatureDim(loc, act, scaled_grad, feature_dim,
                                  rewriter));

      // scale_backprop = scratch2 * scratch1
      scale_backprop = rewriter.create<xla_hlo::MulOp>(loc, scratch1, scratch2);

      // offset_backprop = sum(y_backprop)
      offset_backprop = ApplyReduction(loc, grad, reduce_dims, &rewriter);
    }

    x_backprop = rewriter.create<ConvertOp>(loc, x_backprop, act_ele_type);
    // It doesn't matter what values we provide for the last 2 results.
    rewriter.replaceOp(op,
                       {/*x_backprop=*/x_backprop,
                        /*scale_backprop=*/scale_backprop,
                        /*offset_backprop=*/offset_backprop, op.x(), op.x()});
    return success();
  }
};

using ConvertFusedBatchNormGradOp =
    ConvertFusedBatchNormGradBase<TF::FusedBatchNormGradOp>;
using ConvertFusedBatchNormGradV2Op =
    ConvertFusedBatchNormGradBase<TF::FusedBatchNormGradV2Op>;
using ConvertFusedBatchNormGradV3Op =
    ConvertFusedBatchNormGradBase<TF::FusedBatchNormGradV3Op>;

// Converts TensorFlow FusedBatchNormV3Op to either HLO BatchNormTrainingOp or
// HLO BatchNormInferenceOp, depending on the value of the 'is_training'
// parameter.
class ConvertFusedBatchNormV3Op
    : public OpRewritePattern<TF::FusedBatchNormV3Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::FusedBatchNormV3Op op,
                                PatternRewriter &rewriter) const override {
    auto feature_dim =
        getFeatureDimensionAttr(rewriter, op.data_formatAttr(), op.x());

    auto input_type_tensor = op.x().getType().cast<TensorType>();
    auto input_element_type = input_type_tensor.getElementType();

    auto scale_type_tensor = op.scale().getType().cast<TensorType>();
    auto scale_element_type = scale_type_tensor.getElementType();

    auto mean_type_tensor = op.mean().getType().cast<TensorType>();
    auto mean_element_type = mean_type_tensor.getElementType();
    // In the training case, dimensions of input tensors must be static.
    if (op.is_training() && (!input_type_tensor.hasStaticShape() ||
                             !scale_type_tensor.hasStaticShape() ||
                             !mean_type_tensor.hasStaticShape()))
      return failure();

    // TODO(b/69928690): Support mixed precision in the XLA batch
    // normalization operators. As a workaround, create a new x with the same
    // element type as scale (which may be more precise than the input type).
    Value bn_train_input = rewriter.create<xla_hlo::ConvertOp>(
        op.getLoc(), op.x(), scale_element_type);
    TensorType bn_train_input_type_tensor =
        bn_train_input.getType().cast<TensorType>();

    if (op.is_training()) {
      // Training case.
      auto operand_shape = bn_train_input_type_tensor.getShape();
      // The mean and variance are each 1 dimensional arrays the size of the
      // feature dimension, with the same element type as the operand (x).
      // This shape must be constructed manually because the mean and variance
      // inputs are empty in the training case.
      Type mean_var_type = RankedTensorType::get(
          {operand_shape[feature_dim.getInt()]}, scale_element_type);
      // Op result type is a tuple of 3 values: output with same shape as input;
      // batch_mean, and batch_var.
      SmallVector<Type, 3> operand_types = {bn_train_input_type_tensor,
                                            mean_var_type, mean_var_type};
      Type result_type = TupleType::get(operand_types, rewriter.getContext());

      auto bn_train_op = rewriter.create<xla_hlo::BatchNormTrainingOp>(
          op.getLoc(), result_type, bn_train_input, op.scale(), op.offset(),
          op.epsilon(), feature_dim.getValue());
      // HLO op outputs a tuple of tensors. Extract those results.
      auto bn_train_op_result = bn_train_op.getResult();
      Value y_out = rewriter.create<xla_hlo::GetTupleElementOp>(
          op.getLoc(), bn_train_op_result, 0);
      Value batch_mean = rewriter.create<xla_hlo::GetTupleElementOp>(
          op.getLoc(), bn_train_op_result, 1);
      Value reserve_space_1 = batch_mean;
      Value batch_variance = rewriter.create<xla_hlo::GetTupleElementOp>(
          op.getLoc(), bn_train_op_result, 2);

      // Apply Bessel's correction on the variance.
      int total_input_size = bn_train_input_type_tensor.getNumElements();
      int total_scale_size = scale_type_tensor.getNumElements();
      int sample_size = total_input_size / total_scale_size;
      int sample_size_minus_one = std::max(1, sample_size - 1);
      double factor = static_cast<double>(sample_size) /
                      static_cast<double>(sample_size_minus_one);
      auto factor_const_op = rewriter.create<xla_hlo::ConstOp>(
          op.getLoc(), rewriter.getFloatAttr(scale_element_type, factor));

      Value corrected_variance = rewriter.create<xla_chlo::BroadcastMulOp>(
          op.getLoc(), batch_variance.getType(), batch_variance,
          factor_const_op, /*broadcast_dimensions=*/DenseIntElementsAttr());

      // Convert back to input type to stay aligned with expected output type
      // for TF op.
      y_out = rewriter.create<xla_hlo::ConvertOp>(op.getLoc(), y_out,
                                                  input_element_type);

      float exponential_avg_factor =
          op.exponential_avg_factor().convertToFloat();
      if (exponential_avg_factor != 1.0f) {
        auto alpha = rewriter.create<xla_hlo::ConstOp>(
            op.getLoc(), rewriter.getFloatAttr(mean_element_type,
                                               1.0f - exponential_avg_factor));
        auto beta = rewriter.create<xla_hlo::ConstOp>(
            op.getLoc(),
            rewriter.getFloatAttr(mean_element_type, exponential_avg_factor));

        // new_running_mean = alpha * old_mean + beta * batch_mean.
        auto alpha_mul_old_mean = rewriter.create<xla_chlo::BroadcastMulOp>(
            op.getLoc(), op.mean().getType(), alpha, op.mean(),
            /*broadcast_dimensions=*/DenseIntElementsAttr());
        auto beta_mul_batch_mean = rewriter.create<xla_chlo::BroadcastMulOp>(
            op.getLoc(), batch_mean.getType(), beta, batch_mean,
            /*broadcast_dimensions=*/DenseIntElementsAttr());
        batch_mean = rewriter.create<xla_chlo::BroadcastAddOp>(
            op.getLoc(), alpha_mul_old_mean, beta_mul_batch_mean,
            /*broadcast_dimensions=*/DenseIntElementsAttr());

        // new_running_variance = alpha * old_variance + beta * batch_variance.
        auto alpha_mul_old_variance = rewriter.create<xla_chlo::BroadcastMulOp>(
            op.getLoc(), op.variance().getType(), alpha, op.variance(),
            /*broadcast_dimensions=*/DenseIntElementsAttr());
        auto beta_mul_batch_variance =
            rewriter.create<xla_chlo::BroadcastMulOp>(
                op.getLoc(), corrected_variance.getType(), beta,
                corrected_variance,
                /*broadcast_dimensions=*/DenseIntElementsAttr());
        corrected_variance = rewriter.create<xla_chlo::BroadcastAddOp>(
            op.getLoc(), alpha_mul_old_variance, beta_mul_batch_variance,
            /*broadcast_dimensions=*/DenseIntElementsAttr());
      }

      // TF FusedBatchNormV3 op expects 5 outputs. Outputs 3 and 4 are
      // currently marked as "reserved spaces 1 and 2". They are used to
      // pass the per-batch mean and variance to the gradiant. Here we
      // maintain the same behavior by setting them to the mean and variance
      // calculated by BatchNormTraining. Output 5 is unused; it doesn't
      // matter what we pass there.
      rewriter.replaceOp(op, {y_out, /*batch_mean=*/batch_mean,
                              /*batch_variance=*/corrected_variance,
                              /*reserve_space_1=*/reserve_space_1,
                              /*reserve_space_2=*/batch_variance,
                              /*reserve_space_3=*/op.x()});
    } else {  // Inference case.
      auto bn_train_op = rewriter.create<BatchNormInferenceOp>(
          op.getLoc(),
          /*result_type=*/bn_train_input_type_tensor, bn_train_input,
          op.scale(), op.offset(), op.mean(), op.variance(), op.epsilon(),
          feature_dim.getValue());

      // Convert back to input type to stay aligned with expected output type
      // for TF op.
      auto y_out = rewriter.create<xla_hlo::ConvertOp>(op.getLoc(), bn_train_op,
                                                       input_element_type);

      // The mean, variance, and reserved space outputs of the batch norm op are
      // not used for inference. It doesn't matter what values we provide for
      // the last 5 results as long as they are of the same type. Forward
      // input mean and variance to output mean, variance, reserved_space_1 and
      // reserver_space_2. Create a constant tensor to forward to last
      // reserve_space_3 output.
      auto reserve_space_3_type = op.getResult(5).getType().cast<TensorType>();
      int num_elements = reserve_space_3_type.hasStaticShape()
                             ? reserve_space_3_type.getNumElements()
                             : 0;
      auto const_attr_type = RankedTensorType::get(
          {num_elements}, getElementTypeOrSelf(reserve_space_3_type));

      Value dummy_const = rewriter.create<ConstOp>(
          op.getLoc(), DenseElementsAttr::get<float>(const_attr_type, 0.0));
      if (const_attr_type != reserve_space_3_type)
        dummy_const = rewriter.create<TensorCastOp>(
            op.getLoc(), reserve_space_3_type, dummy_const);
      rewriter.replaceOp(op, {/*y=*/y_out,
                              /*batch_mean=*/op.mean(),
                              /*batch_variance=*/op.variance(),
                              /*reserve_space_1=*/op.mean(),
                              /*reserve_space_2=*/op.variance(),
                              /*reserve_space_3=*/dummy_const});
    }
    return success();
  }
};

using PaddingArray =
    std::vector<std::pair<tensorflow::int64, tensorflow::int64>>;

// Returns padding values for ReduceWindow op as a vector of pairs.
//
// Requires padding to be either 'SAME' or 'VALID' and the number of input
// dimensions to be equal to the size of window dimensions and window strides.
template <int num_dims>
static PaddingArray GetReduceWindowPaddingAsArray(
    llvm::ArrayRef<int64_t> input_dims, ArrayAttr window_dims,
    ArrayAttr window_strides, StringRef padding, Builder *builder) {
  if (padding == "VALID") {
    return PaddingArray(num_dims, std::make_pair(0, 0));
  }
  assert(padding == "SAME");
  llvm::SmallVector<tensorflow::int64, num_dims> input_shape, window_shape,
      strides;
  input_shape.reserve(input_dims.size());
  window_shape.reserve(window_shape.size());
  strides.reserve(window_strides.size());

  for (const auto &dim : input_dims) input_shape.push_back(dim);
  for (Attribute attr : window_dims)
    window_shape.push_back(attr.cast<IntegerAttr>().getInt());
  for (Attribute attr : window_strides)
    strides.push_back(attr.cast<IntegerAttr>().getInt());

  PaddingArray paddings = ::xla::MakePadding(input_shape, window_shape, strides,
                                             ::xla::Padding::kSame);
  return paddings;
}

// Same as GetReduceWindowPaddingAsArray but returns padding as
// DenseIntElementsAttr. Returns empty attribute for `VALID` padding.
template <int num_dims>
static DenseIntElementsAttr GetReduceWindowPaddingAsAttr(
    llvm::ArrayRef<int64_t> input_dims, ArrayAttr window_dims,
    ArrayAttr window_strides, StringRef padding, Builder *builder) {
  if (padding == "VALID") return {};
  assert(padding == "SAME");
  PaddingArray paddings = GetReduceWindowPaddingAsArray<num_dims>(
      input_dims, window_dims, window_strides, padding, builder);
  int64_t rank = paddings.size();
  llvm::SmallVector<int64_t, num_dims * 2> flatten_paddings(rank * 2);
  for (int i = 0; i < rank; i++) {
    flatten_paddings[2 * i] = paddings[i].first;
    flatten_paddings[2 * i + 1] = paddings[i].second;
  }
  return DenseIntElementsAttr::get(
      RankedTensorType::get({rank, 2}, builder->getIntegerType(64)),
      flatten_paddings);
}

// Converts AvgPool op to HLO ReduceWindow op by setting appropriate window
// dimensions with add as the reduction function. The reduction result is
// then divided by the number of elements in the window.
class ConvertAvgPoolOp : public OpRewritePattern<TF::AvgPoolOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::AvgPoolOp op,
                                PatternRewriter &rewriter) const override {
    auto input_type = op.value().getType().dyn_cast<RankedTensorType>();
    if (!input_type) return failure();

    // TODO(b/147217034): support other data formats.
    if (!IsDefaultDataFormat(op.data_format())) return failure();
    // TODO(b/147217034): support "SAME" padding.
    if (op.padding() != "VALID") return failure();

    // We will do accumulation first; use a larger bitwidth if suitable.
    Type input_element_type = input_type.getElementType();
    Type sum_element_type = GetSumAccumulationType(input_element_type);
    Type result_type;

    // The result type for reduction and division with the proper element type.
    if (auto ranked_type = op.getType().dyn_cast<RankedTensorType>())
      result_type =
          RankedTensorType::get(ranked_type.getShape(), sum_element_type);
    else
      result_type = UnrankedTensorType::get(sum_element_type);

    Value input_value = op.value();

    // Convert if we need enlarge the element type's bitwidth.
    if (input_element_type != sum_element_type)
      input_value = rewriter.create<ConvertOp>(op.getLoc(), input_value,
                                               sum_element_type);

    // Create the tf.ReduceWindow op.
    Value init =
        GetScalarConstOfType(sum_element_type, op.getLoc(), 0, &rewriter);
    DenseIntElementsAttr paddings_attr =
        GetReduceWindowPaddingAsAttr<4>(input_type.getShape(), op.ksize(),
                                        op.strides(), op.padding(), &rewriter);
    auto reduce = rewriter.create<ReduceWindowOp>(
        op.getLoc(), result_type, input_value, init,
        GetI64ElementsAttr(op.ksize()), GetI64ElementsAttr(op.strides()),
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(), paddings_attr);
    BuildReduceBody<AddOp>(sum_element_type, &reduce.body(), &rewriter);

    // Count the number of elements in the window. The following calculation
    // is only valid for no paddings.
    SmallVector<int64_t, 4> ksize;
    GetI64ArrayAttrValues(op.ksize(), &ksize);
    int64_t count = std::accumulate(ksize.begin(), ksize.end(), 1,
                                    std::multiplies<int64_t>());

    // Divide by the number of elements in the window.
    Value divisor =
        GetScalarConstOfType(sum_element_type, op.getLoc(), count, &rewriter);
    auto scalar_broadcast_dims = GetI64ElementsAttr({}, &rewriter);
    Value result = rewriter.create<xla_chlo::BroadcastDivOp>(
        op.getLoc(), result_type, reduce, divisor, scalar_broadcast_dims);

    // Convert back if we enlarged the element type's bitwidth.
    if (input_element_type != sum_element_type)
      result =
          rewriter.create<ConvertOp>(op.getLoc(), result, input_element_type);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// `AvgPoolGradOp` is converted to the following operations:
// 1. Divide each entry of the output gradient (the gradient for the previous
//    layer in backpropagation order) by the count of the corresponding window
//    (i.e., the number of non-padding entries of the window which `AvgPool`
//    has mapped to this entry in forward propagation).
// 2. Add appropriate interior and exterior padding for step 3 (see example
//    below).
// 3. Convolve the result of step 2. with a kernel consisting of 1's (same shape
//    as windows) and stride 1 in each dimension. This is implemented as a
//    `ReduceWindowOp` with `AddOp` as body.
//
// Example:
// Let f : R^4 -> R^2 be an average pool function with window size 3, stride 2,
// and SAME padding with 0's. It is defined by
//    f(x) = [ (x_1 + x_2 + x_3) / 3 ]      ( x = (x_1, x_2, x_3, x_4) )
//           [ (x_3 + x_4 + 0)   / 2 ]      (the 0 results from right padding)
// Note that for SAME padding in `AvgPool` the padded entries are not counted
// for the average, this is why the second denominator is 2 and not 3.
// The Jacobian Df is
//    [ 1/3  1/3  1/3  0   ]
//    [ 0    0    1/2  1/2 ]
//
// Note that the Jacobian is constant (this is why `ConvertAvgPoolGradOp` only
// needs the original input shape and not the tensor as argument).
// Let v = [ 4  6 ]^T  be the output gradient (^T = transposed). Then the
// average pool gradient is given by
//    Df^T * v = [ 4/3  4/3  13/3  3 ]^T
// Instead of a matrix-vector-multiplication we can utilize the sparsity and
// structure of Df by using the 3-step approach from above:
// 1. Divide output gradient v by window counts: [ 4/3  6/2 ]^T
// 2. Add appropriate padding: [ 0  0  4/3  0  3  0 ]^T
// 3. Convolve with kernel [ 1  1  1 ]: [ 4/3  4/3  11/3  3 ]^T
//
// Note that the padding in step 2. is chosen in such a way that the subsequent
// convolution produces the gradient. Higher dimensions, different padding, and
// different windows/strides work in a similar way, the main difference is in
// the computation of the paddings in step 2.
//
// For more details on backpropagation for convolution of which `AvgPoolGrad`
// is a special case see `tensorflow/core/kernels/conv_grad_ops.h`.
// `tensorflow/compiler/mlir/xla/tests/legalize-tf.mlir` has more
// examples for different cases.
template <typename OpTy, int num_dims>
class ConvertAvgPoolGradOp : public OpRewritePattern<OpTy> {
  using DimVector = SmallVector<int64_t, num_dims>;

 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.data_format().str(), &data_format)) {
      return failure();
    }
    // `out_grad` is the gradient that was propagated via backpropagation from
    // the output layer.
    Value out_grad = op.grad();
    auto out_grad_type =
        out_grad.getType().template dyn_cast<RankedTensorType>();
    if (!out_grad_type) {
      return failure();
    }
    Type element_type = out_grad_type.getElementType();
    DenseIntElementsAttr orig_input_shape_attr;
    if (!matchPattern(op.orig_input_shape(),
                      m_Constant(&orig_input_shape_attr))) {
      return failure();
    }
    auto orig_input_shape_values = orig_input_shape_attr.getValues<int32_t>();
    DimVector orig_input_shape(orig_input_shape_values.begin(),
                               orig_input_shape_values.end());
    RankedTensorType orig_input_type =
        RankedTensorType::get(orig_input_shape, element_type);
    DimVector ksize, strides;
    GetI64ArrayAttrValues(op.ksize(), &ksize);
    GetI64ArrayAttrValues(op.strides(), &strides);
    Value zero = GetScalarConstOfType(element_type, loc, 0, &rewriter);

    Operation *out_grad_divided = nullptr;
    if (op.padding() == "VALID") {
      // All window counts are equal here because we don't have padding
      // (each entry of `out_grad` corresponds to a window that consists of
      //  original input entries only).
      int64_t window_count = std::accumulate(ksize.begin(), ksize.end(), 1,
                                             std::multiplies<int64_t>());
      // Divide `out_grad` by window counts.
      Value divisor =
          GetScalarConstOfType(element_type, loc, window_count, &rewriter);
      auto scalar_broadcast_dims = GetI64ElementsAttr({}, &rewriter);
      out_grad_divided = rewriter.create<xla_chlo::BroadcastDivOp>(
          loc, out_grad_type, out_grad, divisor, scalar_broadcast_dims);
    } else {
      assert(op.padding() == "SAME");
      // For SAME padding, only original entries that contributed to a window
      // are counted for the average of this window, not padded entries.

      // Build all-ones tensor of same shape as the original input.
      ElementsAttr splat = xla::getSplat(&rewriter, orig_input_type, 1);
      auto all_ones_tensor = rewriter.create<ConstOp>(loc, splat);

      // Get the same padding as for the original input.
      DenseIntElementsAttr orig_padding_attr =
          GetReduceWindowPaddingAsAttr<num_dims>(orig_input_shape, op.ksize(),
                                                 op.strides(), op.padding(),
                                                 &rewriter);

      // Count the 1's in each window, using the same padding as for the
      // original input, which gives us the window counts by which `out_grad`
      // needs to be divided.
      auto window_counts = rewriter.create<ReduceWindowOp>(
          loc, out_grad_type,
          /*operand=*/all_ones_tensor,
          /*init_value=*/zero,
          /*window_dimensions=*/GetI64ElementsAttr(op.ksize()),
          /*window_strides=*/GetI64ElementsAttr(op.strides()),
          /*base_dilations=*/DenseIntElementsAttr(),
          /*window_dilations=*/DenseIntElementsAttr(),
          /*padding=*/orig_padding_attr);
      BuildReduceBody<AddOp>(element_type, &window_counts.body(), &rewriter);

      // Divide `out_grad` by window counts.
      out_grad_divided = rewriter.create<xla_hlo::DivOp>(
          loc, out_grad_type, out_grad, window_counts);
    }

    // Get same padding as for original input.
    PaddingArray orig_padding = GetReduceWindowPaddingAsArray<num_dims>(
        orig_input_shape, op.ksize(), op.strides(), op.padding(), &rewriter);

    // Add padding around `out_grad_divided` values in such a way that the
    // subsequent `ReduceWindowOp` produces the gradient.
    DimVector out_grad_shape(
        llvm::to_vector<num_dims>(out_grad_type.getShape()));
    DimVector low_padding(num_dims, 0);
    DimVector high_padding(num_dims, 0);
    DimVector interior_padding(num_dims, 0);
    constexpr int num_spatial_dims = num_dims - 2;
    for (int i = 0; i < num_spatial_dims; ++i) {
      int dim = tensorflow::GetTensorSpatialDimIndex(num_dims, data_format, i);
      int orig_input_shape_padded_in_dim = orig_input_shape[dim] +
                                           orig_padding[dim].first +
                                           orig_padding[dim].second;
      // Set interior padding such that neighboring entries from
      // `out_grad_divided` have distance `strides[dim]` from each other in
      // every dimension.
      interior_padding[dim] = strides[dim] - 1;
      // Set exterior padding in the same way as for convolution gradient
      // computation.
      auto status = ::xla::ConvGradExtractAndVerifyDimension(
          /*input_size=*/orig_input_shape_padded_in_dim,
          /*filter_size=*/ksize[dim],
          /*output_size=*/out_grad_shape[dim],
          /*dilation=*/1,
          /*stride=*/strides[dim],
          /*padding=*/::xla::Padding::kValid);
      if (!status.ok()) {
        return failure();
      }
      ::xla::SpatialDimensionOutputSizeAndPadding &conv_grad_spatial_dim =
          status.ValueOrDie();
      // Subtract the original exterior padding since it doesn't contribute to
      // the gradient. Note that we save one `PadOp` and some unnecessary kernel
      // computations, compared to the `xla::AvgPoolGrad` implementation, by
      // subtracting the original exterior padding before `ReduceWindowOp`
      // instead of trimming the result of `ReduceWindowOp` (the final result is
      // the same because all strides are 1).
      low_padding[dim] =
          conv_grad_spatial_dim.pad_before - orig_padding[dim].first;
      high_padding[dim] =
          conv_grad_spatial_dim.pad_after - orig_padding[dim].second;

      // Update `out_grad_shape` to result shape of following `PadOp`.
      out_grad_shape[dim] = low_padding[dim] + high_padding[dim] +
                            (out_grad_shape[dim] - 1) * strides[dim] + 1;
    }
    Value reduce_window_input = rewriter.create<PadOp>(
        loc, RankedTensorType::get(out_grad_shape, element_type),
        /*operand=*/out_grad_divided->getOpResult(0),
        /*padding_value=*/zero,
        /*edge_padding_low=*/GetI64ElementsAttr(low_padding, &rewriter),
        /*edge_padding_high=*/GetI64ElementsAttr(high_padding, &rewriter),
        /*interior_padding=*/GetI64ElementsAttr(interior_padding, &rewriter));

    // Compute result by convolving `reduce_window_input` with an all-ones
    // kernel, using `ReduceWindowOp` with `AddOp` body.

    Type sum_element_type = GetSumAccumulationType(element_type);
    if (element_type != sum_element_type) {
      // Convert to appropriate sum accumulation type to avoid precision loss.
      reduce_window_input = rewriter.create<ConvertOp>(loc, reduce_window_input,
                                                       sum_element_type);
      zero = GetScalarConstOfType(sum_element_type, loc, 0, &rewriter);
    }
    auto ones = GetI64ElementsAttr(DimVector(num_dims, 1), &rewriter);
    auto reduce_window_op = rewriter.create<ReduceWindowOp>(
        loc, RankedTensorType::get(orig_input_shape, sum_element_type),
        /*operand=*/reduce_window_input,
        /*init_value=*/zero,
        /*window_dimensions=*/GetI64ElementsAttr(op.ksize()),
        /*window_strides=*/ones,
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(),
        /*padding=*/DenseIntElementsAttr());
    BuildReduceBody<AddOp>(sum_element_type, &reduce_window_op.body(),
                           &rewriter);
    Value result = reduce_window_op.getResult();

    if (element_type != sum_element_type) {
      // Convert back to original element type.
      result = rewriter.create<ConvertOp>(op.getLoc(), result, element_type);
    }
    rewriter.replaceOp(op, {result});
    return success();
  }
};

using ConvertAvgPool2DGradOp =
    ConvertAvgPoolGradOp<TF::AvgPoolGradOp, /*num_dims=*/4>;
using ConvertAvgPool3DGradOp =
    ConvertAvgPoolGradOp<TF::AvgPool3DGradOp, /*num_dims=*/5>;

// Converts MaxPool op to HLO ReduceWindow op by setting appropriate window
// dimensions with max as the reduction function.
//
// Sample result for VALID padding mode:
//
//   %init = constant dense<...> : tensor<i32>
//   %max_pool = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.maximum"]
//               {window_dimensions = ..., window_strides = ... }
//
template <typename OpTy, int num_dims>
class ConvertMaxPoolOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Type element_type =
        op.input().getType().template cast<TensorType>().getElementType();
    if (!element_type.isSignlessIntOrFloat()) return failure();
    Location loc = op.getLoc();
    ConstOp init = GetMinValueForType(element_type, loc, &rewriter);

    auto input_ty = op.input().getType().template dyn_cast<RankedTensorType>();
    if (!input_ty) return failure();
    DenseIntElementsAttr paddings_attr = GetReduceWindowPaddingAsAttr<num_dims>(
        input_ty.getShape(), op.ksize(), op.strides(), op.padding(), &rewriter);
    auto reduce = rewriter.create<ReduceWindowOp>(
        loc, op.getType(), op.input(), init.getResult(),
        GetI64ElementsAttr(op.ksize()), GetI64ElementsAttr(op.strides()),
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(), paddings_attr);
    BuildReduceBody<MaxOp>(element_type, &reduce.body(), &rewriter);

    rewriter.replaceOp(op, reduce.getResult());
    return success();
  }
};

using ConvertMaxPool2DOp = ConvertMaxPoolOp<TF::MaxPoolOp, /*num_dims=*/4>;
using ConvertMaxPool3DOp = ConvertMaxPoolOp<TF::MaxPool3DOp, /*num_dims=*/5>;

// Converts SelectV2 to HLO Select op and necessary BroadcastInDim ops on
// operands.
//
// For example, the following source IR:
//
//   %select = "tf.SelectV2"(%condition, %t, %e) :
//               (tensor<1xi1>, tensor<2xi32>, tensor<1xi32>) -> tensor<2xi32>
//
// will be converted into:
//
//   %pred = "xla_hlo.broadcast_in_dim"(%cond)
//             {broadcast_dimensions = dense<[0]> : tensor<1xi64>} :
//               (tensor<1xi1>) -> tensor<2xi1>
//   %on_false = "xla_hlo.broadcast_in_dim"(%e)
//                 {broadcast_dimensions = dense<[0]> : tensor<1xi64>} :
//                   (tensor<1xi32>) -> tensor<2xi32>
//   %select = "xla_hlo.select"(%pred, %t, %on_false) :
//               (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
class ConvertSelectV2Op : public OpRewritePattern<TF::SelectV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SelectV2Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<int64_t, 4> broadcast_then_else_shape;
    auto ranked_then_type = op.t().getType().dyn_cast<RankedTensorType>();
    auto ranked_else_type = op.e().getType().dyn_cast<RankedTensorType>();
    auto ranked_cond_type =
        op.condition().getType().dyn_cast<RankedTensorType>();
    if (!ranked_then_type || !ranked_then_type.hasStaticShape() ||
        !ranked_else_type || !ranked_else_type.hasStaticShape() ||
        !ranked_cond_type || !ranked_cond_type.hasStaticShape())
      return failure();

    if (!OpTrait::util::getBroadcastedShape(ranked_then_type.getShape(),
                                            ranked_else_type.getShape(),
                                            broadcast_then_else_shape))
      return failure();

    llvm::SmallVector<int64_t, 4> broadcast_shape;
    if (!OpTrait::util::getBroadcastedShape(broadcast_then_else_shape,
                                            ranked_cond_type.getShape(),
                                            broadcast_shape))
      return failure();

    auto broadcast_or_self = [&](Value value) {
      RankedTensorType type = value.getType().cast<RankedTensorType>();
      auto output_type =
          RankedTensorType::get(broadcast_shape, type.getElementType());
      if (output_type == type) return value;

      int64_t rank = type.getRank();
      SmallVector<int64_t, 4> broadcast_dimensions(rank);
      std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(),
                broadcast_shape.size() - rank);

      return rewriter
          .create<BroadcastInDimOp>(
              op.getLoc(), output_type, value,
              GetI64ElementsAttr(broadcast_dimensions, &rewriter))
          .getResult();
    };

    // HLO SelectOp supports broadcasting for predicate/condition if
    // predicate/condition is a scalar.
    Value pred = ranked_cond_type.getRank() == 0
                     ? op.condition()
                     : broadcast_or_self(op.condition());
    Value on_true = broadcast_or_self(op.t());
    Value on_false = broadcast_or_self(op.e());

    rewriter.replaceOpWithNewOp<SelectOp>(op, on_true.getType(), pred, on_true,
                                          on_false);

    return success();
  };
};

// Converts Sigmoid op to HLO ops computing sigmoid with the following formula:
//
//     sigmoid = add(mul(tanh(mul(logits, 0.5)), 0.5), 0.5)
//
// Sample result with 2-d f16 inputs with B batches of with N elements each.
//
//    // Create an array of 0.5 the shape of the input array.
//    %half = xla_hlo.constant dense<5.000000e-01> : tensor<f32>
//    %half_array = "xla_hlo.broadcast"(half)
//                           {broadcast_sizes = dense<2> : tensor<1xi64>}
//                           : (tensor<f32>) -> tensor<2xf32>
//
//    // Compute Tanh of half the logits of the values.
//    %halved_logits = xla_hlo.multiply %logits, %half_array : tensor<2xf32>
//    %tanh = "xla_hlo.tanh"(%halved_logits) : (tensor<2xf32>) -> tensor<2xf32>
//
//    // Have the result of Tanh and add 0.5.
//    %halved_tanh = xla_hlo.multiply %tanh, %half : tensor<2xf32>
//    %sigmoid = xla_hlo.add %halved_tanh, %half : tensor<2xf32>
//
class ConvertSigmoidOp : public OpRewritePattern<TF::SigmoidOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Create constant half with shape and element type same as the operand.
    Value operand = op.getOperand();
    auto operand_ty = operand.getType().cast<TensorType>();
    auto scalar_ty = RankedTensorType::get({}, operand_ty.getElementType());
    ElementsAttr attr = mlir::xla::getSplat(&rewriter, scalar_ty, 0.5);
    auto scalar_half = rewriter.create<ConstOp>(loc, attr);
    auto half = BroadcastToShapeOf(loc, scalar_half, operand, rewriter);

    auto scaled_input = rewriter.create<MulOp>(loc, operand, half);
    auto tanh_op = rewriter.create<TanhOp>(loc, scaled_input);
    auto mul_op = rewriter.create<MulOp>(loc, tanh_op, half);
    auto add_op = rewriter.create<AddOp>(loc, mul_op, half);

    rewriter.replaceOp(op, add_op.getResult());
    return success();
  }
};

// Converts Softmax and LogSoftmax to HLO ops, computing softmax with the
// following formulas:
//
//     softmax = div(exp(logits), sum(exp(logits)))

//     log_softmax = sub(logits, log(sum(exp(logits))))
//
// Sample result with 2-d f16 inputs with B batches of with N elements each.
//
//    %reduce_dim = tf.Const dense<[1]> : tensor<1xi64>
//
//    // Subtract each element by their batches' max to improve numerical
//    // stability.
//    %max = "tf.Max"(%input, %reduce_dim)
//           : (tensor<BxNxf16>, tensor<1xi64>) -> tensor<Bxf16>
//    %sub = "xla_hlo.subtract"(%inp, %max) {broadcast_dimensions = 0}
//            : (tensor<BxNxf16>, tensor<Bxf16>) -> tensor<BxNxf16>
//
//    %exp = "xla_hlo.exponential"(%sub) : (tensor<BxNxf16>) -> tensor<BxNxf16>
//    %sum = "tf.Sum"(%exp, %reduce_dim)
//            : (tensor<BxNxf32>, tensor<1xi64>) -> tensor<Bxf32>
//
//    // Softmax computation:
//    %softmax = "xla_hlo.divide"(%exp, %sum_f16) {broadcast_dimensions = 0}
//            : (tensor<BxNxf16>, tensor<Bxf16>) -> tensor<BxNxf16>
template <typename OpTy, bool use_log = true>
class ConvertSoftmaxOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Softmax converter requires ranked type because the XLA reduce ops used
    // while lowering requires dimensions attribute to reduce along.
    // Note that the input and output shape is equivalent, so we use 'logits'
    // and its type for shape calculations.
    Value logits = op.logits();
    RankedTensorType type = logits.getType().dyn_cast<RankedTensorType>();
    if (!type) return failure();
    auto loc = op.getLoc();
    int rank = type.getRank();

    // Note that the TensorFlow Softmax op verifies that the input rank is
    // greater than or equal to one so the following sequence is valid.
    auto reduce_dim = rewriter.create<TF::ConstOp>(
        loc, GetI64ElementsAttr({rank - 1}, &rewriter));

    // Exponential of input values and then their sum can be very large here.
    // Division with large denominator is numerically unstable. To improve
    // numerical stability, subtract each batch with their max element so that
    // the maximum input value is zero. It can be shown that softmax computed
    // after adding or subtracting all inputs in a batch using a common value
    // gives mathematically equivalent result.
    auto max_logits =
        rewriter.create<TF::MaxOp>(loc, logits, reduce_dim,
                                   /*keep_dims=*/rewriter.getBoolAttr(false));
    auto max_logits_broadcast =
        CommonPrefixBroadcast(loc, logits, max_logits, rewriter);
    auto shifted_logits = rewriter.create<xla_hlo::SubOp>(loc, type, logits,
                                                          max_logits_broadcast);

    // Exponentiate the inputs.
    Value exp = rewriter.create<ExpOp>(loc, type, shifted_logits);

    // Compute summation of the exponentials.
    auto exp_sum =
        rewriter.create<TF::SumOp>(loc, exp, reduce_dim,
                                   /*keep_dims=*/rewriter.getBoolAttr(false));
    Value sum = exp_sum.getResult();

    if (use_log) {
      Value log = rewriter.create<LogOp>(loc, sum);
      auto log_broadcast = CommonPrefixBroadcast(loc, logits, log, rewriter);
      rewriter.replaceOpWithNewOp<xla_hlo::SubOp>(op, shifted_logits,
                                                  log_broadcast);
    } else {
      auto sum_broadcast = CommonPrefixBroadcast(loc, logits, sum, rewriter);
      rewriter.replaceOpWithNewOp<xla_hlo::DivOp>(op, exp, sum_broadcast);
    }
    return success();
  }
};

// Converts Size to HLO ops, computing the size of a ranked input tensor.
// TODO(b/145253252): Update this to not require ranked input tensor shapes.
//
// The main logic of this pattern is to calculate the size by multiplying every
// dimension of the input tensor's shape together.
//
// For example, the following source IR:
//
//   %size = "tf.Size"(%input) : (tensor<2x?x8xf32>) -> tensor<i32>
//
// will be converted into:
//
//   %const = xla_hlo.constant dense<1> : tensor<i32>
//   %dim_0 = "xla_hlo.get_dimension_size"(%input) {dimension = 0 : i32} :
//                                         (tensor<2x?x8xf32>) -> tensor<i32>
//   %prod_0 = xla_hlo.multiply %const, %dim_0 : tensor<i32>
//   %dim_1 = "xla_hlo.get_dimension_size"(%input) {dimension = 1 : i32} :
//                                         (tensor<2x?x8xf32>) -> tensor<i32>
//   %prod_1 = xla_hlo.multiply %prod_0, %dim_1 : tensor<i32>
//   %dim_2 = "xla_hlo.get_dimension_size"(%input) {dimension = 2 : i32} :
//                                         (tensor<2x?x8xf32>) -> tensor<i32>
//   %size = xla_hlo.multiply %prod_1, %dim_2 : tensor<i32>
class ConvertSizeOp : public OpRewritePattern<TF::SizeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SizeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();
    auto input_ty = input.getType().dyn_cast<RankedTensorType>();
    if (!input_ty) return failure();

    const int64_t rank = input_ty.getRank();
    auto result_type = op.getResult().getType();
    Operation *size =
        GetScalarConstOfType(result_type.cast<TensorType>().getElementType(),
                             op.getLoc(), 1, &rewriter);
    for (int64_t i = 0; i < rank; ++i) {
      auto dim = rewriter.create<GetDimensionSizeOp>(
          op.getLoc(), result_type, input,
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), i));
      size = rewriter.create<xla_chlo::BroadcastMulOp>(
          op.getLoc(), size->getResult(0), dim.getResult(),
          /*DenseIntElementsAttr=*/DenseIntElementsAttr());
    }
    rewriter.replaceOp(op, size->getResult(0));

    return success();
  }
};

static void BroadcastBatchMatMulV2Operands(Value lhs, Value rhs, Location loc,
                                           Value *out_lhs, Value *out_rhs,
                                           PatternRewriter *rewriter) {
  // The dimension structure of the relevant operands to a tf.BatchMatMulV2 is:
  // - lhs: [LHSBATCHDIMS..., LHSROWS, LHSCOLS]
  // - rhs: [RHSBATCHDIMS..., RHSROWS, RHSCOLS]
  // - result: [broadcast(LHSBATCHDIMS, RHSBATCHDIMS)..., LHSROWS, RHSCOLS]
  // To perform the matmul, we need to first broadcast lhs and rhs to a common
  // set of leading dimensions before doing the actual matmul.
  // That's what the code below does.
  // In particular, we populate out_lhs and out_rhs to have dimension structure:
  // - out_lhs: [broadcast(LHSBATCHDIMS, RHSBATCHDIMS)..., LHSROWS, LHSCOLS]
  // - out_rhs: [broadcast(LHSBATCHDIMS, RHSBATCHDIMS)..., RHSROWS, RHSCOLS]
  // To do this, we need to calculate those output shapes, which involves
  // slicing off the leading batch dims of each operand, broadcasting them,
  // then concatenating the broadcasted leading dims back to the row/col dims.
  // Finally, we create a TF::BroadcastTo op that does the actual broadcast.

  // TODO(silvasean): Reduce duplication across reified shape calculations and
  // the static computation of output types needed to create ops.
  Value lhs_shape = rewriter->create<shape::ShapeOfOp>(loc, lhs);
  Value rhs_shape = rewriter->create<shape::ShapeOfOp>(loc, rhs);
  Value const_neg2 =
      rewriter->create<ConstantOp>(loc, rewriter->getI32IntegerAttr(-2));
  auto lhs_splitted =
      rewriter->create<shape::SplitAtOp>(loc, lhs_shape, const_neg2);
  auto rhs_splitted =
      rewriter->create<shape::SplitAtOp>(loc, rhs_shape, const_neg2);
  auto lhs_type = lhs.getType().cast<RankedTensorType>();
  auto rhs_type = rhs.getType().cast<RankedTensorType>();
  // The last two dimensions are the matrix row/col dimensions. Don't broadcast
  // them.
  SmallVector<int64_t, 6> result_batch_shape_compile_time_extents;
  OpTrait::util::getBroadcastedShape(lhs_type.getShape().drop_back(2),
                                     rhs_type.getShape().drop_back(2),
                                     result_batch_shape_compile_time_extents);
  auto result_batch_shape = rewriter->create<shape::BroadcastOp>(
      loc, lhs_splitted.head(), rhs_splitted.head(),
      /*error=*/nullptr);
  // Lambda which handles the broadcasting of one side to the common
  // leading-batch dimensions.
  auto broadcast_one_side = [&](Value side, RankedTensorType type,
                                Value tail_shape, Value *out_side) {
    ArrayRef<int64_t> matrix_dims = type.getShape().take_back(2);
    auto result_shape = result_batch_shape_compile_time_extents;
    result_shape.append(matrix_dims.begin(), matrix_dims.end());
    auto result_type =
        RankedTensorType::get(result_shape, type.getElementType());
    auto shape =
        rewriter->create<shape::ConcatOp>(loc, result_batch_shape, tail_shape);
    auto shape_tensor = rewriter->create<shape::ToExtentTensorOp>(
        loc,
        RankedTensorType::get({static_cast<int64_t>(result_shape.size())},
                              rewriter->getIndexType()),
        shape);
    *out_side = rewriter->create<TF::BroadcastToOp>(loc, result_type, side,
                                                    shape_tensor);
  };
  broadcast_one_side(lhs, lhs_type, lhs_splitted.tail(), out_lhs);
  broadcast_one_side(rhs, rhs_type, rhs_splitted.tail(), out_rhs);
}

class ConvertBatchMatMulV2Op : public OpRewritePattern<TF::BatchMatMulV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::BatchMatMulV2Op op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.x();
    Value rhs = op.y();
    auto lhs_type = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhs_type = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhs_type || !rhs_type) return failure();
    if (lhs_type.getElementType().isa<ComplexType>() && op.adj_x()) {
      lhs = rewriter.create<TF::ConjOp>(op.getLoc(), lhs_type, lhs);
    }
    if (rhs_type.getElementType().isa<ComplexType>() && op.adj_y()) {
      rhs = rewriter.create<TF::ConjOp>(op.getLoc(), rhs_type, rhs);
    }

    // Broadcast both operands.
    BroadcastBatchMatMulV2Operands(lhs, rhs, op.getLoc(), &lhs, &rhs,
                                   &rewriter);
    lhs_type = lhs.getType().cast<RankedTensorType>();
    rhs_type = rhs.getType().cast<RankedTensorType>();
    assert(lhs_type.getRank() == rhs_type.getRank());
    int64_t rank = lhs_type.getRank();
    auto batch_dimensions = GetI64ElementsAttr(
        llvm::to_vector<4>(llvm::seq<int64_t>(0, rank - 2)), &rewriter);
    auto lhs_contracting_dimensions = GetI64ElementsAttr(
        llvm::makeArrayRef({op.adj_x() ? rank - 2 : rank - 1}), &rewriter);
    auto rhs_contracting_dimensions = GetI64ElementsAttr(
        llvm::makeArrayRef({op.adj_y() ? rank - 1 : rank - 2}), &rewriter);
    auto dimension_numbers = DotDimensionNumbers::get(
        /*lhs_batching_dimensions=*/batch_dimensions,
        /*rhs_batching_dimensions=*/batch_dimensions,
        /*lhs_contracting_dimensions=*/lhs_contracting_dimensions,
        /*rhs_contracting_dimensions=*/rhs_contracting_dimensions,
        rewriter.getContext());
    // TODO(silvasean): Emit shape checks for contracting dimensions.
    // (The batch dimensions are checked by the broadcasting logic)
    rewriter.replaceOpWithNewOp<DotGeneralOp>(op, op.getType(), lhs, rhs,
                                              dimension_numbers,
                                              /*precision_config=*/nullptr);
    return success();
  }
};

// Converts the tf.Split op into a series of HLO slice ops when the tensor to be
// split has fully static shape and the dimension to split is a constant.
//
// The main logic of this pattern is to calculate the index start and end range
// for each slice. And this happens only on the dimension to be split; for all
// other dimensions, all resultant slices' index start and end range covers the
// input tensor's full range. Strides for all resultant slices are all one.
//
// For example, the following source IR:
//
//   %dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
//   %0:3 = "tf.Split"(%dim, %input) : (tensor<i32>, tensor<4x6xf32>) ->
//                (tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>)
//
// will be converted into:
//
//   %0 = "xla_hlo.slice"(%input) {
//             limit_indices = dense<[4, 2]> : tensor<2xi64>,
//             start_indices = dense<0> : tensor<2xi64>,
//             strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
//   %1 = "xla_hlo.slice"(%input) {
//             limit_indices = dense<4> : tensor<2xi64>,
//              start_indices = dense<[0, 2]> : tensor<2xi64>,
//            strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
//    %2 = "xla_hlo.slice"(%input) {
//            limit_indices = dense<[4, 6]> : tensor<2xi64>,
//            start_indices = dense<[0, 4]> : tensor<2xi64>,
//             strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
// TODO(antiagainst): consider lowering into TF ops so the pattern can be more
// applicable.
class ConvertSplitOp : public OpRewritePattern<TF::SplitOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SplitOp op,
                                PatternRewriter &rewriter) const override {
    // We can only split along static dimensions.
    auto input_type = op.value().getType().dyn_cast<RankedTensorType>();
    if (!input_type) return failure();

    // We can only match when the split dimension is a constant scalar.
    DenseIntElementsAttr split_dim_attr;
    if (!matchPattern(op.split_dim(), m_Constant(&split_dim_attr)))
      return failure();

    // Get the dimension we are splitting at. Offset properly if it's negative.
    int64_t input_rank = input_type.getRank();
    int64_t dim_index = (*split_dim_attr.begin()).getSExtValue();
    if (dim_index < 0) dim_index += input_rank;

    // Calculate the dimension size for each slice along the split dimension.
    int64_t input_dim_size = input_type.getDimSize(dim_index);
    // If we are splitting along the dynamic dimension then we cannot compute
    // the static dimension length.
    if (TensorType::isDynamic(input_dim_size)) return failure();

    int64_t num_splits = op.getNumResults();
    int64_t slice_size = input_dim_size / num_splits;

    // Get each slice's type.
    auto slice_shape = llvm::to_vector<4>(input_type.getShape());
    slice_shape[dim_index] = slice_size;
    Type slice_type =
        RankedTensorType::get(slice_shape, input_type.getElementType());

    // Parameters for constructing each slice.
    SmallVector<int64_t, 4> begin_indices(input_rank, 0);
    auto end_indices = llvm::to_vector<4>(input_type.getShape());
    SmallVector<int64_t, 4> strides(input_rank, 1);

    // All HLO slice results used to replace the original tf.Split op.
    SmallVector<Value, 4> slices;
    slices.reserve(num_splits);

    for (int i = 0; i < num_splits; ++i) {
      begin_indices[dim_index] = i * slice_size;
      end_indices[dim_index] = (i + 1) * slice_size;
      slices.push_back(
          rewriter.create<SliceOp>(op.getLoc(), slice_type, op.value(),
                                   GetI64ElementsAttr(begin_indices, &rewriter),
                                   GetI64ElementsAttr(end_indices, &rewriter),
                                   GetI64ElementsAttr(strides, &rewriter)));
    }

    rewriter.replaceOp(op, slices);
    return success();
  }
};

// Converts the tf.SplitV op into a series of HLO slice ops when the tensor to
// be split has fully static shape and the dimension to split and split sizes
// are constants.
//
// This is similar to the conversion for tf.Split op other than that the size of
// each chunk on the dimension to split is explicitly given as an op operand
// and they are not necessarily the same.
//
// For example, given the following IR:
//
// %split_sizes = "tf.Const"() {value = dense<[1, -1, 3]> : tensor<3xi32>}
// %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>}
// %0:3 = "tf.SplitV"(%input, %split_sizes, %split_dim) :
//                   (tensor<4x6xf32>, tensor<3xi32>, tensor<i32>) ->
//                   (tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>)
//
// We will generate slices following slices:
// %0 = "xla_hlo.slice"(%input) {
//        limit_indices = dense<[4, 1]> : tensor<2xi64>,
//        start_indices = dense<0> : tensor<2xi64>,
//        strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x1xf32>
// %1 = "xla_hlo.slice"(%input) {
//        limit_indices = dense<[4, 3]> : tensor<2xi64>,
//        start_indices = dense<[0, 1]> : tensor<2xi64>,
//        strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
// %2 = "xla_hlo.slice"(%input) {
//        limit_indices = dense<[4, 6]> : tensor<2xi64>,
//        start_indices = dense<[0, 3]> : tensor<2xi64>,
//        strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x3xf32>
class ConvertSplitVOp : public OpRewritePattern<TF::SplitVOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SplitVOp op,
                                PatternRewriter &rewriter) const override {
    // We can only split along static dimensions.
    // TODO(b/145731001): enhance to support dynamic-shaped inputs.
    auto input_type = op.value().getType().dyn_cast<RankedTensorType>();
    if (!input_type) return failure();

    // We can only match when the split dimension is a constant scalar.
    DenseIntElementsAttr split_dim_attr;
    if (!matchPattern(op.split_dim(), m_Constant(&split_dim_attr)))
      return failure();

    // We can only match when the split sizes is a constant int vector.
    DenseIntElementsAttr split_sizes_attr;
    if (!matchPattern(op.size_splits(), m_Constant(&split_sizes_attr)))
      return failure();

    // Get each chunck's size along the dimension to split. It may contain
    // dynamic sizes and we need to update it if so.
    SmallVector<int64_t, 4> split_sizes;
    int64_t total_dim_size = 0;  // Total dimension size assigned to splits
    llvm::Optional<int> dynamic_dim_index;
    split_sizes.reserve(
        split_sizes_attr.getType().cast<ShapedType>().getNumElements());
    for (auto dim : llvm::enumerate(split_sizes_attr)) {
      int64_t dim_val = dim.value().getSExtValue();
      split_sizes.push_back(dim_val);
      if (dim_val == ShapedType::kDynamicSize) {
        // We cannot have more than one dynamic dimension.
        assert(!dynamic_dim_index && "invalid split sizes");
        dynamic_dim_index = dim.index();
      } else {
        total_dim_size += dim_val;
      }
    }

    // Get the dimension we are splitting at. Offset properly if it's negative.
    int64_t input_rank = input_type.getRank();
    int64_t dim_index = (*split_dim_attr.begin()).getSExtValue();
    if (dim_index < 0) dim_index += input_rank;

    int64_t input_dim_size = input_type.getDimSize(dim_index);
    if (TensorType::isDynamic(input_dim_size)) return failure();

    assert(((dynamic_dim_index && total_dim_size <= input_dim_size) ||
            (!dynamic_dim_index && total_dim_size == input_dim_size)) &&
           "invalid split sizes");

    // Update the dynamic dimension with calculated concrete size.
    if (dynamic_dim_index)
      split_sizes[*dynamic_dim_index] = input_dim_size - total_dim_size;

    // Parameters for constructing each slice.
    SmallVector<int64_t, 4> begin_indices(input_rank, 0);
    auto end_indices = llvm::to_vector<4>(input_type.getShape());
    SmallVector<int64_t, 4> strides(input_rank, 1);

    // All HLO slice results used to replace the original tf.Split op.
    SmallVector<Value, 4> slices;
    slices.reserve(op.getNumResults());

    for (int i = 0; i < op.getNumResults(); ++i) {
      end_indices[dim_index] = begin_indices[dim_index] + split_sizes[i];
      slices.push_back(rewriter.create<xla_hlo::SliceOp>(
          op.getLoc(), op.value(), GetI64ElementsAttr(begin_indices, &rewriter),
          GetI64ElementsAttr(end_indices, &rewriter),
          GetI64ElementsAttr(strides, &rewriter)));
      // Prepare the begin indice for the next slice.
      begin_indices[dim_index] = end_indices[dim_index];
    }

    rewriter.replaceOp(op, slices);
    return success();
  }
};

// Converts StridedSlice op to HLO Slice op along with Reverse op to handle
// negative strides and Reshape op to update the output shape. Indices and
// strides operands are converted to attributes with non-negative indexing.
//
// If the begin input is not a compile time constant, the begin input needs to
// be sliced and the slice needs to be lowered to xla_hlo.DynamicSlice. In this
// case, strides must have a known value of 1 (otherwise we have insufficient
// information to conform to XLA's op semantics).
//
// For example with an op like following,
//   tf.StridedSlice(%input, %begin, %end, %strides) {shrink_axis_mask = 1}
//     : tensor<AxBxf32> -> tensor<Pxf32>
//
// If the %begin input is constant, output would be:
//   %reversed = "xla_hlo.Reverse" (%input) {dimensions = ...}
//   %sliced = "xla_hlo.Slice" (%input)
//             {start_indices = ..., limit_indices = ..., strides = ...}
//   %output = "xla_hlo.Reshape" (%sliced) : tensor<1xPxf32> -> tensor<Pxf32>
//
class ConvertStridedSliceOp : public OpRewritePattern<TF::StridedSliceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult rewriteWithConstantBegin(TF::StridedSliceOp op,
                                         ArrayRef<int64_t> begin_indices,
                                         ArrayRef<int64_t> end_indices,
                                         ArrayRef<int64_t> strides,
                                         RankedTensorType input_ty,
                                         PatternRewriter &rewriter) const {
    SmallVector<int64_t, 4> hlo_begin_indices, hlo_end_indices, hlo_strides,
        dims_to_reverse;
    int64_t input_rank = input_ty.getRank();
    ArrayRef<int64_t> input_shape = input_ty.getShape();
    hlo_begin_indices.reserve(input_rank);
    hlo_end_indices.reserve(input_rank);
    hlo_strides.reserve(input_rank);

    int64_t indices_elements = begin_indices.size();
    if (input_rank < indices_elements) return failure();

    // Convert from TensorFlow negative or out of range indices and strides
    // values to legal HLO Slice attributes.
    for (int i = 0, e = indices_elements; i != e; i++) {
      int64_t begin = begin_indices[i];
      int64_t end = end_indices[i];
      int64_t stride = strides[i];

      if (stride < 0) {
        // Negative stride means that the output values are computed starting
        // from end until begin. Mark the dimension for reversal before slice
        // and compute indices for the reversed input.
        dims_to_reverse.push_back(i);
        begin = (input_shape[i] - 1) - begin;
        end = (input_shape[i] - 1) - end;
        stride = -stride;
      }

      // Unlike TensorFlow, HLO requires begin and end values to be within
      // range.
      begin = std::max(int64_t(0), begin);
      end = std::max(begin, end);
      end = std::min(end, input_shape[i]);

      hlo_begin_indices.push_back(begin);
      hlo_end_indices.push_back(end);
      hlo_strides.push_back(stride);
    }

    Location loc = op.getLoc();
    Value input = op.input();
    if (!dims_to_reverse.empty())
      input = rewriter.create<ReverseOp>(
          loc, input_ty, op.input(),
          GetI64ElementsAttr(dims_to_reverse, &rewriter));
    auto sliced = rewriter.create<SliceOp>(
        loc, input, GetI64ElementsAttr(hlo_begin_indices, &rewriter),
        GetI64ElementsAttr(hlo_end_indices, &rewriter),
        GetI64ElementsAttr(hlo_strides, &rewriter));

    // Reshape slice result so that the shape is updated depending on
    // 'new_axis_mask' or 'shrink_axis_mask' attributes.
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), sliced);
    return success();
  }

  LogicalResult rewriteWithUnknownBegin(TF::StridedSliceOp op,
                                        RankedTensorType input_ty,
                                        RankedTensorType result_ty,
                                        PatternRewriter &rewriter) const {
    // If begin and end values are dynamic, we can only support this lowering
    // if strides are a known value of 1.
    DenseIntElementsAttr sparse_strides_attr;
    if (!matchPattern(op.strides(), m_Constant(&sparse_strides_attr))) {
      return rewriter.notifyMatchFailure(
          op,
          "requires that strides are known when begin/end values are dynamic");
    }
    SmallVector<int64_t, 4> strides;
    int64_t stride_value;
    for (const APInt &stride : sparse_strides_attr) {
      if ((stride_value = stride.getSExtValue()) != 1) {
        return rewriter.notifyMatchFailure(op,
                                           "requires that strides are all 1 "
                                           "when begin/end values are dynamic");
      }
      strides.push_back(stride_value);
    }

    ArrayRef<int64_t> input_shape = input_ty.getShape();
    int last_dim = std::max(static_cast<int>(input_shape.size()) - 1, 0);

    // When begin/end values are dynamic, we can only support shrinking a major
    // axis. For instance, if there are 4 dims, we can support a
    // shrink_axis_mask of 0001 (1), 0011 (3), 0111 (7), or 1111 (15), but no
    // other.
    bool shrink_axis_mask_ok = op.shrink_axis_mask().isMask();
    if (!shrink_axis_mask_ok)
      return rewriter.notifyMatchFailure(
          op,
          "requires that shrink_axis_mask, if set, refer to a major axis "
          "dimension (when begin/end values are dynamic)");

    // When begin/end values are dynamic, the ellipsis mask, if set, must refer
    // to the last dimension.
    int ellipsis_mask = op.ellipsis_mask().getZExtValue();
    if (!(ellipsis_mask == 0 || ellipsis_mask == (1 << last_dim)))
      return rewriter.notifyMatchFailure(
          op,
          "requires that ellipsis_mask, if set, refer to the last dimension of "
          "input (when begin/end values are dynamic)");

    APInt begin_mask = op.begin_mask();
    if (!begin_mask.isNullValue())
      return rewriter.notifyMatchFailure(
          op,
          "requires that begin_mask is either set to 0 or not set when "
          "begin/end values are dynamic");
    APInt end_mask = op.end_mask();
    if (!end_mask.isNullValue())
      return rewriter.notifyMatchFailure(
          op,
          "requires that end_mask is either set to 0 or not set when begin/end "
          "values are dynamic");
    APInt new_axis_mask = op.new_axis_mask();
    if (!new_axis_mask.isNullValue())
      return rewriter.notifyMatchFailure(
          op,
          "requires that new_axis_mask is either set to 0 or not set when "
          "begin/end values are dynamic");

    // In this case where the begin and end values are dynamic, the number of
    // output elements has to be equal to the number of input elements that
    // are sliced.
    int output_elements = result_ty.getNumElements();
    int input_elements_sliced = 1;

    // Begin must be a ranked, 1-dimensional tensor: This is checked by the
    // verifier.
    int64_t slicing_dim_size =
        op.begin().getType().cast<RankedTensorType>().getShape()[0];
    auto input_rank = input_shape.size();
    for (int d = slicing_dim_size; d < input_rank; ++d) {
      // We only support slicing major dimensions, so minor dimensions after
      // slicing dimensions are all sliced with their full sizes.
      input_elements_sliced *= input_shape[d];
    }
    if (input_elements_sliced != output_elements) {
      return rewriter.notifyMatchFailure(
          op,
          "requires the number of output elements to be equal to the number of "
          "input elements sliced (when begin/end values are dynamic)");
    }

    SmallVector<Value, 4> slice_begin_indices;
    // For the dimensions that are to be sliced, all have slice sizes of 1.
    SmallVector<int64_t, 4> slice_sizes(slicing_dim_size, 1);
    auto input_element_ty = input_ty.getElementType();
    // Scalar tensor type.
    TensorType type = RankedTensorType::get(/*shape=*/{}, input_element_ty);
    Location loc = op.getLoc();
    auto zero = GetScalarConstOfType(input_element_ty, loc, 0, &rewriter);
    for (int d = 0; d < slicing_dim_size; ++d) {
      auto index = rewriter.create<SliceOp>(
          loc, op.begin(), GetI64ElementsAttr({d}, &rewriter),
          GetI64ElementsAttr({d + 1}, &rewriter),
          GetI64ElementsAttr({1}, &rewriter));
      // Convert index to scalar.
      auto reshaped_index = rewriter.create<ReshapeOp>(loc, type, index);
      // If the index is negative, wrap it around with dimension size.
      auto index_negative =
          rewriter.create<TF::LessOp>(loc, reshaped_index, zero);
      auto input_val = GetScalarConstOfType(input_element_ty, loc,
                                            input_shape[d], &rewriter);
      auto wrapped_index =
          rewriter.create<TF::AddOp>(loc, input_val, reshaped_index);
      auto final_index = rewriter.create<SelectOp>(
          loc, type, index_negative, wrapped_index, reshaped_index);
      slice_begin_indices.push_back(final_index);
    }

    // For non-slice dims, get the full slice of that dimension.
    for (int d = slicing_dim_size; d < input_shape.size(); ++d) {
      slice_sizes.push_back(input_shape[d]);
      slice_begin_indices.push_back(zero);
    }

    auto slice_sizes_attr = GetI64ElementsAttr(slice_sizes, &rewriter);
    // This must be an xla DynamicSlice op due to the inputs that aren't
    // constant.
    auto sliced = rewriter.create<DynamicSliceOp>(
        loc, op.getType(), op.input(), slice_begin_indices, slice_sizes_attr);

    // Reshape slice result so that the shape is updated depending on
    // 'new_axis_mask' or 'shrink_axis_mask' attributes.
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), sliced);
    return success();
  }

  LogicalResult matchAndRewrite(TF::StridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    // Input shape needs to be static to convert negative indices in TensorFlow
    // to absolute indices required by HLO.
    //
    // TODO(hinsu): Relax this constraint for ops without negative indices and
    // strides.
    auto input_ty = op.input().getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) return failure();

    // Output shape needs to be static to apply 'new_axis_mask' or
    // 'shrink_axis_mask' by reshaping tensor after slice.
    //
    // TODO(hinsu): Relax this constraint for ops without the above masks.
    auto result_ty = op.getType().dyn_cast<RankedTensorType>();
    if (!result_ty || !result_ty.hasStaticShape()) return failure();

    DenseIntElementsAttr sparse_begin_attr, sparse_end_attr;
    if (!matchPattern(op.begin(), m_Constant(&sparse_begin_attr)) ||
        !matchPattern(op.end(), m_Constant(&sparse_end_attr))) {
      return rewriteWithUnknownBegin(op, input_ty, result_ty, rewriter);
    }

    SmallVector<int64_t, 4> begin_indices, end_indices, strides;
    if (!op.GetSlicedBoundRanges(&begin_indices, &end_indices, &strides)) {
      return failure();
    }
    return rewriteWithConstantBegin(op, begin_indices, end_indices, strides,
                                    input_ty, rewriter);
  }
};

// Converts tf.StridedSliceGrad to HLO reshape, reverse and padding ops.
//
// tf.StridedSlice is taking slice of the input tensor. tf.StridedSliceGrad does
// the reverse: it propagates the graident for the sliced tensor to the original
// input tensor by doing padding with zeros. The main logic is calculating the
// indices and strides for padding.
class ConvertStridedSliceGradOp
    : public OpRewritePattern<TF::StridedSliceGradOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::StridedSliceGradOp op,
                                PatternRewriter &rewriter) const override {
    // We need constant input shape to perform padding calculations later.
    DenseIntElementsAttr input_shape_attr;
    if (!matchPattern(op.shape(), m_Constant(&input_shape_attr)))
      return failure();

    // We also need constant begin/end indices and strides to perform padding
    // calculations.
    // Bounded shape after performing strided slice
    SmallVector<int64_t, 4> shape;
    // Bounded begin, end, and strides for strided slice
    SmallVector<int64_t, 4> begin_indices, end_indices, strides;
    if (!op.GetSlicedShapeAndBoundRanges(&shape, &begin_indices, &end_indices,
                                         &strides))
      return failure();

    Value grad = op.dy();
    Type element_type = grad.getType().cast<ShapedType>().getElementType();

    // Perform reshape to undo any new/shrink axes done by strided slice.
    grad = rewriter.create<xla_hlo::ReshapeOp>(
        op.getLoc(), RankedTensorType::get(shape, element_type), grad);

    SmallVector<int64_t, 4> padding_low, padding_high, padding_interm;
    SmallVector<int64_t, 4> dims_to_reverse;
    padding_low.reserve(shape.size());
    padding_high.reserve(shape.size());
    padding_interm.reserve(shape.size());

    // Prepare padding parameters for each dimension.
    for (int i = 0, e = shape.size(); i < e; ++i) {
      int64_t input_dim = (*(input_shape_attr.begin() + i)).getSExtValue();
      if (strides[i] > 0) {
        padding_low.push_back(begin_indices[i]);
        padding_interm.push_back(strides[i] - 1);

        // Pad the upper dimension up to the expected input shape. It's not
        // sufficient simply to use end_indices[i] to compute the padding in
        // cases where the stride does not divide evenly into the interval
        // between begin_indices[i] and end_indices[i].
        int64_t size =
            padding_low[i] + shape[i] + (shape[i] - 1) * padding_interm[i];
        padding_high.push_back(input_dim - size);
      } else {
        dims_to_reverse.push_back(i);
        padding_high.push_back(input_dim - begin_indices[i] - 1);
        padding_interm.push_back(-strides[i] - 1);

        // Pad the lower dimension up to the expected input shape.
        int64_t size =
            padding_high[i] + shape[i] + (shape[i] - 1) * padding_interm[i];
        padding_low.push_back(input_dim - size);
      }
    }

    if (!dims_to_reverse.empty()) {
      grad = rewriter.create<xla_hlo::ReverseOp>(
          op.getLoc(), grad.getType(), grad,
          GetI64ElementsAttr(dims_to_reverse, &rewriter));
    }

    auto zero = GetScalarConstOfType(element_type, op.getLoc(), 0, &rewriter);
    rewriter.replaceOpWithNewOp<xla_hlo::PadOp>(
        op, op.getType(), grad, zero,
        GetI64ElementsAttr(padding_low, &rewriter),
        GetI64ElementsAttr(padding_high, &rewriter),
        GetI64ElementsAttr(padding_interm, &rewriter));
    return success();
  }
};

/// Converts the RangeOp tensorflow op to a xla_hlo.iota op with a scaling and
/// offset applied to generate the range values. The output tensor needs to
/// have a static shape.
///
/// For example an op like the following:
///   %result = "tf.Range"(%start, %limit, %delta) {Tidx = "tfdtype$DT_FLOAT"}
///      : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<5xf32>
///
/// Output would be:
///   %iota = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5xf32>
///   %scaled = "xla_hlo.multiply"(%iota, %delta)
///       {broadcast_dimensions = dense<[]> : tensor<0xi64>} :
///       (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
///   %result = "xla_hlo.add"(%scaled, %offset)
///       {broadcast_dimensions = dense<[]> : tensor<0xi64>} :
///       (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
///
/// Implementation is defined in C++ due to no type interface for the iota op.
class ConvertRangeOp : public OpRewritePattern<TF::RangeOp> {
  using OpRewritePattern<TF::RangeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::RangeOp op,
                                PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    auto result_type = result.getType();
    if (!result_type.cast<ShapedType>().hasStaticShape()) {
      return failure();
    }

    auto iota = rewriter.create<IotaOp>(op.getLoc(), result_type,
                                        rewriter.getI64IntegerAttr(0));
    auto scaled = rewriter.create<xla_chlo::BroadcastMulOp>(
        op.getLoc(), result_type, iota, op.delta(),
        xla::getBroadcastDimensionsAttr(&rewriter, iota, op.delta()));
    rewriter.replaceOpWithNewOp<xla_chlo::BroadcastAddOp>(
        op, result_type, scaled, op.start(),
        xla::getBroadcastDimensionsAttr(&rewriter, scaled, op.start()));
    return success();
  }
};

// Converts RangeOp for cases with the length is a dynamic value. The shape of
// the resulting tensor computed, then the start and delta is used with the
// dynamic_iota value to compute the final range value.
//
// For example, the resulting range op value:
//   %range = "tf.range"(%start, %limit, %delta)
//
// Is converted to the following.
//   %start + %delta * iota(ceil(abs((%limit - %start) / %delta))
//
// Implementation is defined in C++ due to the complicated type behavior.
class ConvertDynamicRangeOp : public OpRewritePattern<TF::RangeOp> {
  using OpRewritePattern<TF::RangeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::RangeOp op,
                                PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    auto result_type = result.getType().cast<ShapedType>();
    if (result_type.hasStaticShape()) {
      return failure();
    }

    Value start = op.start();
    Value delta = op.delta();
    Value limit = op.limit();

    // To compute the length we need to use floating point calculations so that
    // ceil can be computed for the number of steps.
    auto compute_element_type =
        getElementTypeOrSelf(start.getType()).isa<FloatType>()
            ? getElementTypeOrSelf(start.getType())
            : rewriter.getF64Type();
    auto compute_type = RankedTensorType::get(
        limit.getType().cast<ShapedType>().getShape(), compute_element_type);

    // Compute the length of the sequence we are going to need. This includes
    // some conversion to float for the operations.
    //
    // %size = ceil(abs((%limit - %start) / %delta))
    auto range = rewriter.create<xla_hlo::SubOp>(op.getLoc(), limit, start);
    auto abs = rewriter.create<xla_hlo::AbsOp>(op.getLoc(), range);

    // Delta is not necessarily the same type as start and limit.
    auto abs_cast =
        rewriter.create<xla_hlo::ConvertOp>(op.getLoc(), compute_type, abs);
    auto delta_cast =
        rewriter.create<xla_hlo::ConvertOp>(op.getLoc(), compute_type, delta);

    // Compute the total number of integer steps and convert to the HLO
    // dimension tensor.
    auto normalized =
        rewriter.create<xla_hlo::DivOp>(op.getLoc(), abs_cast, delta_cast);
    auto ceil = rewriter.create<xla_hlo::CeilOp>(op.getLoc(), normalized);
    auto steps = rewriter.create<xla_hlo::ConvertOp>(
        op.getLoc(), RankedTensorType::get({}, rewriter.getI64Type()), ceil);
    auto reshape = rewriter.create<xla_hlo::ReshapeOp>(
        op.getLoc(), RankedTensorType::get({1}, rewriter.getI64Type()), steps);

    // Using the resulting length compute the correct range value:
    //
    // %range = %start + %delta * iota(%size)
    auto out_scalar_type =
        RankedTensorType::get({}, getElementTypeOrSelf(result_type));
    auto start_out_cast = rewriter.create<xla_hlo::ConvertOp>(
        op.getLoc(), out_scalar_type, start);
    auto delta_out_cast = rewriter.create<xla_hlo::ConvertOp>(
        op.getLoc(), out_scalar_type, delta);

    auto iota = rewriter.create<DynamicIotaOp>(
        op.getLoc(), result_type, reshape, rewriter.getI64IntegerAttr(0));
    auto scaled = rewriter.create<xla_chlo::BroadcastMulOp>(
        op.getLoc(), result_type, iota, delta_out_cast,
        xla::getBroadcastDimensionsAttr(&rewriter, iota, delta_cast));
    rewriter.replaceOpWithNewOp<xla_chlo::BroadcastAddOp>(
        op, result_type, scaled, start_out_cast,
        xla::getBroadcastDimensionsAttr(&rewriter, scaled, start_out_cast));
    return success();
  }
};

ElementsAttr ConvertAxisAttr(Value val, ElementsAttr attr, Builder *builder) {
  auto int_attr = attr.cast<DenseIntElementsAttr>();
  auto type = val.getType().cast<ShapedType>();

  SmallVector<int64_t, 6> axis;
  axis.reserve(int_attr.getNumElements());

  int64_t rank = type.getRank();
  for (auto val : int_attr.getValues<APInt>()) {
    axis.push_back((val.getSExtValue() + rank) % rank);
  }

  return builder->getI64TensorAttr(axis);
}

/// Converts the LinSpace tensorflow op to a xla_hlo.iota op with a scaling
/// and offset applied to generate the linspace values. The output tensor needs
/// to have a static shape.  The implementation is defined in C++ because there
/// is no type inference for the iota op.
class ConvertLinSpaceOp : public OpRewritePattern<TF::LinSpaceOp> {
  using OpRewritePattern<TF::LinSpaceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::LinSpaceOp op,
                                PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    auto result_type = result.getType().dyn_cast<ShapedType>();
    if (!result_type || !result_type.hasStaticShape()) {
      return failure();
    }

    DenseIntElementsAttr num_attr;
    if (!matchPattern(op.num(), m_Constant(&num_attr))) {
      return rewriter.notifyMatchFailure(op, "Num must be a constant scalar");
    }

    if (num_attr.begin() == num_attr.end()) {
      return rewriter.notifyMatchFailure(op, "Num must not be empty");
    }
    int64_t num = (*num_attr.begin()).getSExtValue();

    // Calculate the scaling that needs to be applied to the iota.
    auto step_numerator = rewriter.create<xla_chlo::BroadcastSubOp>(
        op.getLoc(), op.start().getType(), op.stop(), op.start(),
        xla::getBroadcastDimensionsAttr(&rewriter, op.stop(), op.start()));
    Value step_denominator = rewriter.create<ConvertOp>(
        op.getLoc(), op.num(), result_type.getElementType());
    if (num > 1) {
      Value one = GetScalarConstOfType(result_type.getElementType(),
                                       op.getLoc(), 1, &rewriter);
      step_denominator = rewriter.create<xla_chlo::BroadcastSubOp>(
          op.getLoc(), step_denominator.getType(), step_denominator, one,
          xla::getBroadcastDimensionsAttr(&rewriter, step_denominator, one));
    }
    auto step = rewriter.create<xla_chlo::BroadcastDivOp>(
        op.getLoc(), step_numerator.getType(), step_numerator, step_denominator,
        xla::getBroadcastDimensionsAttr(&rewriter, step_numerator,
                                        step_denominator));

    // Scale the iota and add the offset.
    auto iota = rewriter.create<IotaOp>(op.getLoc(), result_type,
                                        rewriter.getI64IntegerAttr(0));
    auto scaled = rewriter.create<xla_chlo::BroadcastMulOp>(
        op.getLoc(), result_type, iota, step,
        xla::getBroadcastDimensionsAttr(&rewriter, iota, step));
    rewriter.replaceOpWithNewOp<xla_chlo::BroadcastAddOp>(
        op, result_type, scaled, op.start(),
        xla::getBroadcastDimensionsAttr(&rewriter, scaled, op.start()));
    return success();
  }
};

/// Converts a generic OpTy tensorflow op to a xla_hlo.reduce op over
/// ReductionOp.
/// `is_accumulation` controls whether it uses higher precision for the actual
/// reduction. This is set to false for ops like max where there is no precision
/// concerns.
template <typename Derived, typename OpTy, typename ReductionOp,
          bool is_accumulation = true>
class GenericConvertReductionOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // TODO(b/141785544): Update this to not require static shapes.
    // Input shape needs to be static to convert negative indices in TensorFlow
    // to absolute indices required by HLO.
    auto input_ty = op.input().getType().template dyn_cast<RankedTensorType>();
    if (!input_ty) return failure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();

    DenseIntElementsAttr dimensions;
    if (!matchPattern(op.reduction_indices(), m_Constant(&dimensions)))
      return failure();

    // Build the final shape from input_shape and dimensions using a bitmap
    // to mark the reduced dimensions.
    SmallVector<bool, 4> reduced_dimensions_bitmap(input_shape.size(), false);
    SmallVector<int64_t, 4> xla_dimensions;
    for (const APInt &index_raw : dimensions.getValues<APInt>()) {
      int64_t index = index_raw.getSExtValue();
      int64_t rank = input_shape.size();
      if ((index < -rank || index >= rank)) return failure();
      index = (index + rank) % rank;
      reduced_dimensions_bitmap[index] = true;
      xla_dimensions.push_back(index);
    }

    Location loc = op.getLoc();
    Type element_type = input_ty.getElementType();
    // Convert to an accumulation type to not lose precision when doing
    // repeated arithmetic operations.
    Type reduce_element_type =
        is_accumulation ? GetAccumulationType(element_type) : element_type;
    auto casted_input =
        rewriter.create<ConvertOp>(loc, op.input(), reduce_element_type);

    // Each reduction op can have a different initial value.
    Value init = Derived::GetInitialValue(reduce_element_type, loc, &rewriter);

    auto reduction = rewriter.create<ReduceOp>(
        loc, casted_input.getResult(), init,
        GetI64ElementsAttr(xla_dimensions, &rewriter));
    BuildReduceBody<ReductionOp>(reduce_element_type, &reduction.body(),
                                 &rewriter);
    Value result = reduction.getResult(0);

    // The mean op needs to divide by the product of the reduced dimensions.
    if (std::is_same<OpTy, TF::MeanOp>::value) {
      int64_t divisor_count = 1;
      for (size_t i = 0; i < input_shape.size(); ++i) {
        if (reduced_dimensions_bitmap[i]) {
          if (TensorType::isDynamic(input_shape[i])) {
            return failure();
          }
          divisor_count *= input_shape[i];
        }
      }
      auto divisor = GetScalarConstOfType(reduce_element_type, loc,
                                          divisor_count, &rewriter);
      auto broadcast_dims = GetI64ElementsAttr({}, &rewriter);
      result = rewriter.create<xla_chlo::BroadcastDivOp>(
          loc, result, divisor.getResult(), broadcast_dims);
    }

    result = rewriter.create<ConvertOp>(loc, result, element_type);

    // Need to reshape back after the reduction if we're keeping the reduced
    // dimensions.
    if (op.keep_dims()) {
      result = rewriter.create<ReshapeOp>(loc, op.getType(), result);
    }
    rewriter.replaceOp(op, {result});

    return success();
  }
};

// Converts Mean op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %sum = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.add"]
//               {dimensions = ...}
//   %divisor = constant dense<...> : tensor<T>
//   %mean = "xla_hlo.divide"(%sum, %divisor)
class ConvertMeanOp
    : public GenericConvertReductionOp<ConvertMeanOp, TF::MeanOp, AddOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;
  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 0, rewriter);
  }
};

// Converts Sum op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %sum = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.add"]
//               {dimensions = ...}
class ConvertSumOp
    : public GenericConvertReductionOp<ConvertSumOp, TF::SumOp, AddOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 0, rewriter);
  }
};

// Converts Max op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %max = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.maximum"]
//               {dimensions = ...}
class ConvertMaxOp
    : public GenericConvertReductionOp<ConvertMaxOp, TF::MaxOp, MaxOp,
                                       /* is_accumulation= */ false> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetMinValueForType(reduce_element_type, loc, rewriter);
  }
};

// Converts Min op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %min = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.minimum"]
//               {dimensions = ...}
class ConvertMinOp
    : public GenericConvertReductionOp<ConvertMinOp, TF::MinOp, MinOp,
                                       /* is_accumulation= */ false> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetMaxValueForType(reduce_element_type, loc, rewriter);
  }
};

// Converts Prod op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %prod = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.multiply"]
//               {dimensions = ...}
class ConvertProdOp
    : public GenericConvertReductionOp<ConvertProdOp, TF::ProdOp, MulOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 1, rewriter);
  }
};

// Converts All op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %max = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.and"]
//               {dimensions = ...}
class ConvertAllOp
    : public GenericConvertReductionOp<ConvertAllOp, TF::AllOp, AndOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;
  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 1, rewriter);
  }
};

// Converts Any op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %max = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.or"]
//               {dimensions = ...}
class ConvertAnyOp
    : public GenericConvertReductionOp<ConvertAnyOp, TF::AnyOp, OrOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;
  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 0, rewriter);
  }
};

// Converts tensorflow ArgMin or ArgMax op to xla_hlo operations that perform
// a reduction on the original input and the corresponding index. The reduction
// sub-computation selects the max (or min) value and the index for the value.
//   Derived: is the resulting derived class of this class.
//   OpTy: is TF::ArgMaxOp or TF::ArgMinOp.
template <typename Derived, typename OpTy>
class ConvertArgMinMaxOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType input_type =
        op.input().getType().template dyn_cast<RankedTensorType>();
    if (!input_type) {
      return failure();
    }

    Type input_element_type = input_type.getElementType();
    // TODO(bixia): Clarify whether tf.ArgMax supports complex data types. If
    // tf.ArgMax doesn't support complex data types, this check can be removed.
    if (!input_element_type.isSignlessIntOrFloat()) return failure();

    Location loc = op.getLoc();
    Value init_value =
        Derived::GetInitialValue(input_element_type, loc, rewriter);

    RankedTensorType output_type =
        op.output().getType().template dyn_cast<RankedTensorType>();
    if (!output_type) {
      return failure();
    }

    Type index_element_type = output_type.getElementType();
    Value index_init_value =
        GetScalarConstOfType(index_element_type, loc, 0, &rewriter);

    RankedTensorType index_type =
        RankedTensorType::get(input_type.getShape(), index_element_type);

    llvm::Optional<int64_t> optional_axis =
        GetIntegerHLOAxisFromTFAxis(op.dimension(), input_type.getRank());
    if (!optional_axis.hasValue()) {
      return failure();
    }
    int64_t axis = optional_axis.getValue();

    IntegerAttr iota_dimension =
        IntegerAttr::get(rewriter.getIntegerType(64), axis);
    Value index_values =
        rewriter.create<IotaOp>(loc, index_type, iota_dimension);

    std::vector<int64_t> dimensions = input_type.getShape();
    dimensions.erase(dimensions.begin() + axis);
    ArrayRef<int64_t> reduction_result_shape(dimensions);

    Value operands[] = {op.input(), index_values};
    Value init_values[] = {init_value, index_init_value};
    DenseIntElementsAttr reduction_dimensions =
        GetI64ElementsAttr({axis}, &rewriter);

    auto reduction = rewriter.create<ReduceOp>(
        loc, llvm::ArrayRef<Value>(operands),
        llvm::ArrayRef<Value>(init_values), reduction_dimensions);
    StringRef direction = Derived::GetDirection();
    BuildArgMinMaxReductionBody(input_element_type, index_element_type,
                                direction, &reduction.body(), &rewriter);

    rewriter.replaceOp(op, {reduction.getResult(1)});
    return success();
  }
};

// Converts tensorflow ArgMax op to xla_hlo operations. The actual
// implementation is in class ConvertArgMinMaxOp:
//
//   %init_index = constant dense<...> : tensor<T>
//   %init = constant dense<...> : tensor<T>
//   %reduce = "xla_hlo.reduce"(%selected_input, %select_index, %init,
//                              %init_index) ["xla_hlo.arg_max"]
class ConvertArgMaxOp
    : public ConvertArgMinMaxOp<ConvertArgMaxOp, TF::ArgMaxOp> {
 public:
  using ConvertArgMinMaxOp::ConvertArgMinMaxOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter &rewriter) {
    return GetMinValueForType(reduce_element_type, loc, &rewriter);
  }

  static StringRef GetDirection() { return "GT"; }
};

// Converts TF TensorScatterUpdate op into Scatter Op with assignment:
//
//   %result = "xla_hlo.scatter"(%tensor, %indices, %updates)
//     { dimensions = ... }
//
class ConvertTensorScatterUpdateOp
    : public OpRewritePattern<TF::TensorScatterUpdateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::TensorScatterUpdateOp op,
                                PatternRewriter &rewriter) const override {
    auto tensor_ty = op.tensor().getType().dyn_cast<RankedTensorType>();
    auto indices_ty = op.indices().getType().dyn_cast<RankedTensorType>();
    auto updates_ty = op.updates().getType().dyn_cast<RankedTensorType>();

    if (!tensor_ty || !indices_ty || !updates_ty) return failure();
    // Last dimension of the indices needs to known at compile time for
    // computation of the 'update_window_dims' attribute in the dimensions
    // struct.
    int64_t num_index_dims = indices_ty.getShape().back();
    if (ShapedType::isDynamic(num_index_dims)) return failure();

    int64_t tensor_rank = tensor_ty.getRank();
    int64_t indices_rank = indices_ty.getRank();
    int64_t updates_rank = updates_ty.getRank();

    int64_t window_dims = tensor_rank - num_index_dims;
    auto dims_attr = ScatterDimensionNumbers::get(
        GetI64ElementsAttrForSeq(updates_rank - window_dims, updates_rank,
                                 &rewriter),
        GetI64ElementsAttrForSeq(0, num_index_dims, &rewriter),
        GetI64ElementsAttrForSeq(0, num_index_dims, &rewriter),
        rewriter.getI64IntegerAttr(indices_rank - 1), rewriter.getContext());

    Location loc = op.getLoc();
    auto scatter = rewriter.create<ScatterOp>(
        loc, op.getType(), op.tensor(), op.indices(), op.updates(), dims_attr);

    // Build region to assign the new value.
    [&](Region *region) {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block = rewriter.createBlock(region);

      // Block arguments are scalars of the given element type.
      Type type =
          RankedTensorType::get(/*shape=*/{}, tensor_ty.getElementType());
      block->addArguments({type, type});
      rewriter.create<ReturnOp>(loc, block->getArgument(1));
    }(&scatter.update_computation());

    rewriter.replaceOp(op, scatter.getResult());
    return success();
  }
};

// Converts Tile op to HLO BroadcastInDim and Reshape ops.
//   For shape [S1, S2] and multiples [M1, M2],
//     MS1 = M1 * S1; MS2 = M2 * S2
//
//   %broadcast = xla_hlo.broadcast_in_dim(%input) {
//     broadcast_dimensions = [0, 2]
//   }
//   %result = "xla_hlo.reshape"(%broadcast) : (tensor<S1xM1xS2xM2xf32>)
//      -> tensor<MS1xMS2xf32>
class ConvertTileOp : public OpRewritePattern<TF::TileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::TileOp op,
                                PatternRewriter &rewriter) const override {
    auto input_ty = op.input().getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) return failure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();
    Type element_type = input_ty.getElementType();

    DenseIntElementsAttr multiples;
    if (!matchPattern(op.multiples(), m_Constant(&multiples)) ||
        multiples.getType().getRank() != 1)
      return failure();

    if (multiples.getNumElements() != input_shape.size()) return failure();

    SmallVector<int64_t, 8> broadcasted_shape;
    SmallVector<int64_t, 4> broadcast_dimensions;
    broadcasted_shape.reserve(input_shape.size() * 2);
    broadcast_dimensions.reserve(input_shape.size());
    for (auto multiple_and_input :
         llvm::zip(multiples.getValues<APInt>(), input_shape)) {
      int64_t multiple = std::get<0>(multiple_and_input).getSExtValue();
      int64_t input_size = std::get<1>(multiple_and_input);

      if (multiple < 0) return failure();

      // Line input up with the next dimension in broadcasted_shape
      // when broadcasting.
      int64_t broadcast_dim;
      int64_t output_size = input_size * multiple;
      if (input_size == 1 || multiple == 1) {
        // Special case for when normal broadcasting will just work.
        broadcast_dim = broadcasted_shape.size();
        broadcasted_shape.push_back(output_size);
      } else {
        // Tiling will happen for this dimension during the ReshapeOp below.
        broadcasted_shape.push_back(multiple);
        broadcast_dim = broadcasted_shape.size();
        broadcasted_shape.push_back(input_size);
      }
      broadcast_dimensions.push_back(broadcast_dim);
    }
    Location loc = op.getLoc();
    Type broadcasted_type =
        RankedTensorType::get(broadcasted_shape, element_type);
    Type output_type = op.getType();

    Value result = rewriter.create<BroadcastInDimOp>(
        loc, broadcasted_type, op.input(),
        GetI64ElementsAttr(broadcast_dimensions, &rewriter));

    if (output_type != broadcasted_type) {
      result = rewriter.create<ReshapeOp>(loc, output_type, result);
    }

    rewriter.replaceOp(op, {result});

    return success();
  }
};

template <typename OpTy, int num_dims>
class ConvertMaxPoolGradOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type element_type =
        op.orig_input().getType().template cast<TensorType>().getElementType();

    // Compute paddings using the original input and kernel shape and strides.
    // Here, ReduceWindow op as used as the MaxPool op is lowered to the
    // ReduceWindow op.
    auto input_ty =
        op.orig_input().getType().template dyn_cast<RankedTensorType>();
    if (!input_ty) return failure();
    DenseIntElementsAttr paddings_attr = GetReduceWindowPaddingAsAttr<num_dims>(
        input_ty.getShape(), op.ksize(), op.strides(), op.padding(), &rewriter);

    auto result = rewriter.create<SelectAndScatterOp>(
        loc, op.getType(), op.orig_input(), op.grad(),
        GetScalarConstOfType(element_type, loc, 0, &rewriter),
        GetI64ElementsAttr(op.ksize()), GetI64ElementsAttr(op.strides()),
        paddings_attr);

    BuildReduceBody<AddOp>(element_type, &result.scatter(), &rewriter);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block = rewriter.createBlock(&result.select());

      // Block arguments are scalars of the given element type.
      Type type = RankedTensorType::get(/*shape=*/{}, element_type);
      block->addArguments({type, type});

      auto reducer = rewriter.create<CompareOp>(
          loc, block->getArgument(0), block->getArgument(1),
          StringAttr::get("GE", rewriter.getContext()));
      rewriter.create<ReturnOp>(loc, reducer.getResult());
    }

    rewriter.replaceOp(op, {result});

    return success();
  }
};

using ConvertMaxPool2DGradOp =
    ConvertMaxPoolGradOp<TF::MaxPoolGradOp, /*num_dims=*/4>;
using ConvertMaxPool3DGradOp =
    ConvertMaxPoolGradOp<TF::MaxPool3DGradOp, /*num_dims=*/5>;

// Converts tf.Conv?DBackpropInputOp into:
//   %rev_filter = "xla_hlo.reverse"(%filter)
//   %result = "xla_hlo.convolution"(%out_backprop, %rev_filter)
template <typename OpTy, int num_spatial_dims>
class ConvertConvBackpropInputOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Unpack all of the attributes.
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.data_format().str(), &data_format))
      return failure();

    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.padding().str(), &padding).ok())
      return failure();

    auto out_backprop_ty =
        op.out_backprop().getType().template dyn_cast<RankedTensorType>();
    auto filter_ty =
        op.filter().getType().template dyn_cast<RankedTensorType>();

    for (RankedTensorType ty : {out_backprop_ty, filter_ty})
      if (!ty || !ty.hasStaticShape()) return failure();

    DenseIntElementsAttr input_shape_attr;
    if (!matchPattern(op.input_sizes(), m_Constant(&input_shape_attr)) ||
        input_shape_attr.getType().getRank() != 1)
      return failure();

    auto input_shape = input_shape_attr.getValues<int32_t>();

    auto dilations_attr = GetI64ElementsAttr(op.dilations());
    std::vector<int> dilations{
        dilations_attr.template getValues<int64_t>().begin(),
        dilations_attr.template getValues<int64_t>().end()};
    auto strides_attr = GetI64ElementsAttr(op.strides());
    std::vector<tensorflow::int32> strides{
        strides_attr.template getValues<int64_t>().begin(),
        strides_attr.template getValues<int64_t>().end()};

    std::vector<tensorflow::int64> explicit_paddings;
    if (padding == tensorflow::Padding::EXPLICIT) {
      // EXPLICIT padding mode and the associated attribute is limited to
      // Conv2DBackpropInput. So, fetch attribute by identifier instead of the
      // op.explicit_paddings() attribute getter.
      ArrayRef<Attribute> explicit_paddings_attr =
          op.template getAttrOfType<ArrayAttr>("explicit_paddings").getValue();
      explicit_paddings.reserve(explicit_paddings_attr.size());
      for (Attribute explicit_padding : explicit_paddings_attr)
        explicit_paddings.push_back(
            explicit_padding.cast<IntegerAttr>().getInt());
    }

    constexpr int num_dims = num_spatial_dims + 2;
    ArrayRef<int64_t> filter_shape = filter_ty.getShape();

    // Reuse dimension computation logic from conv_grad_shape_utils.cc.
    tensorflow::ConvBackpropDimensions dims;
    if (!tensorflow::ConvBackpropComputeDimensionsV2(
             /*label=*/"", num_spatial_dims,
             ToTensorShape<int32_t, num_dims>(input_shape),
             ToTensorShape<int64_t, num_dims>(filter_shape),
             ToTensorShape<int64_t, num_dims>(out_backprop_ty.getShape()),
             dilations, strides, padding, explicit_paddings, data_format, &dims)
             .ok()) {
      return failure();
    }

    // Compute ConvDimensionNumbers, dilation, and padding.
    SmallVector<int64_t, num_spatial_dims> spatial_dims;
    SmallVector<int64_t, num_spatial_dims> lhs_dilation;
    SmallVector<int64_t, num_spatial_dims> rhs_dilation;
    SmallVector<int64_t, num_spatial_dims * 2> paddings;

    for (int i : llvm::seq<int>(0, num_spatial_dims)) {
      const int64_t dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      spatial_dims.push_back(dim);
      const auto &spatial_dim_i = dims.spatial_dims[i];
      lhs_dilation.push_back(spatial_dim_i.stride);
      rhs_dilation.push_back(dilations[dim]);
      paddings.push_back(spatial_dim_i.pad_before);
      paddings.push_back(spatial_dim_i.pad_after);
    }

    RankedTensorType paddings_ty = RankedTensorType::get(
        {num_spatial_dims, 2}, rewriter.getIntegerType(64));
    auto paddings_attr = DenseIntElementsAttr::get(paddings_ty, paddings);

    auto spatial_dims_attr = GetI64ElementsAttr(spatial_dims, &rewriter);

    Value filter = op.filter();

    const int feature_dim =
        tensorflow::GetTensorFeatureDimIndex(num_dims, data_format);
    const int64_t in_depth = *(input_shape.begin() + feature_dim);
    const int64_t filter_in_depth = filter_shape[num_spatial_dims];
    const int64_t feature_group_count = in_depth / filter_in_depth;

    if (feature_group_count != 1) {
      /*
      // TODO(parkers): Convert this code to mlir.
    filter = TransposeFilterForGroupConvolutionBackpropInput(
        filter, filter_shape, feature_group_count, attrs.num_spatial_dims);
        */
      return failure();
    }

    auto kernel_spatial_dims_attr =
        GetI64ElementsAttrForSeq(0, num_spatial_dims, &rewriter);

    // Mirror the filter in the spatial dimensions.
    filter = rewriter.create<ReverseOp>(op.getLoc(), filter,
                                        kernel_spatial_dims_attr);

    const int batch_dim =
        tensorflow::GetTensorBatchDimIndex(num_dims, data_format);
    auto batch_dim_attr = rewriter.getI64IntegerAttr(batch_dim);
    auto feature_dim_attr = rewriter.getI64IntegerAttr(feature_dim);

    // activation gradients
    //   = gradients (with padding and dilation) <conv> mirrored_weights
    Value result = rewriter.create<ConvOp>(
        op.getLoc(), op.getType(), op.out_backprop(), filter,
        /*window_strides=*/
        GetI64ElementsAttrForValue(/*size=*/num_spatial_dims, /*val=*/1,
                                   &rewriter),
        /*padding=*/paddings_attr, GetI64ElementsAttr(lhs_dilation, &rewriter),
        GetI64ElementsAttr(rhs_dilation, &rewriter),
        ConvDimensionNumbers::get(
            /*input_batch_dimension=*/batch_dim_attr,
            /*input_feature_dimension=*/feature_dim_attr,
            /*input_spatial_dimensions=*/spatial_dims_attr,
            // TF filter shape is [ H, W, ..., inC, outC ]
            // Transpose the input and output features for computing the
            // gradient.
            /*kernel_input_feature_dimension=*/
            rewriter.getI64IntegerAttr(num_spatial_dims + 1),
            /*kernel_output_feature_dimension=*/
            rewriter.getI64IntegerAttr(num_spatial_dims),
            /*kernel_spatial_dimensions=*/kernel_spatial_dims_attr,
            /*output_batch_dimension=*/batch_dim_attr,
            /*output_feature_dimension=*/feature_dim_attr,
            /*output_spatial_dimensions=*/spatial_dims_attr,
            rewriter.getContext()),
        rewriter.getI64IntegerAttr(feature_group_count),
        /*batch_group_count=*/rewriter.getI64IntegerAttr(1),
        /*precision_config=*/ArrayAttr());

    rewriter.replaceOp(op, {result});

    return success();
  }
};

using ConvertConv2DBackpropInputOp =
    ConvertConvBackpropInputOp<TF::Conv2DBackpropInputOp,
                               /*num_spatial_dims=*/2>;
using ConvertConv3DBackpropInputOp =
    ConvertConvBackpropInputOp<TF::Conv3DBackpropInputV2Op,
                               /*num_spatial_dims=*/3>;

// Converts tf.Conv?DBackpropFilterOp into:
//   %result = "xla_hlo.convolution"(%input, %out_backprop)
template <typename OpTy, int num_spatial_dims>
class ConvertConvBackpropFilterOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Unpack all of the attributes.
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.data_format().str(), &data_format))
      return failure();

    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.padding().str(), &padding).ok())
      return failure();

    auto out_backprop_ty =
        op.out_backprop().getType().template dyn_cast<RankedTensorType>();
    auto input_ty = op.input().getType().template dyn_cast<RankedTensorType>();

    for (RankedTensorType ty : {out_backprop_ty, input_ty})
      if (!ty || !ty.hasStaticShape()) return failure();

    ArrayRef<int64_t> out_backprop_shape = out_backprop_ty.getShape();
    ArrayRef<int64_t> input_shape = input_ty.getShape();

    DenseIntElementsAttr filter_shape_attr;
    if (!matchPattern(op.filter_sizes(), m_Constant(&filter_shape_attr)) ||
        filter_shape_attr.getType().getRank() != 1)
      return failure();

    auto dilations_attr = GetI64ElementsAttr(op.dilations());
    std::vector<int> dilations{
        dilations_attr.template getValues<int64_t>().begin(),
        dilations_attr.template getValues<int64_t>().end()};
    auto strides_attr = GetI64ElementsAttr(op.strides());
    std::vector<tensorflow::int32> strides{
        strides_attr.template getValues<int64_t>().begin(),
        strides_attr.template getValues<int64_t>().end()};

    std::vector<tensorflow::int64> explicit_paddings;
    if (padding == tensorflow::Padding::EXPLICIT) {
      // EXPLICIT padding mode and the associated attribute is limited to
      // Conv2DBackpropFilter. So, fetch attribute by identifier instead of the
      // op.explicit_paddings() attribute getter.
      ArrayRef<Attribute> explicit_paddings_attr =
          op.template getAttrOfType<ArrayAttr>("explicit_paddings").getValue();
      explicit_paddings.reserve(explicit_paddings_attr.size());
      for (Attribute explicit_padding : explicit_paddings_attr)
        explicit_paddings.push_back(
            explicit_padding.cast<IntegerAttr>().getInt());
    }

    constexpr int num_dims = num_spatial_dims + 2;
    auto filter_shape = filter_shape_attr.getValues<int32_t>();

    // Reuse dimension computation logic from conv_grad_shape_utils.cc.
    tensorflow::ConvBackpropDimensions dims;
    if (!tensorflow::ConvBackpropComputeDimensionsV2(
             /*label=*/"", num_spatial_dims,
             ToTensorShape<int64_t, num_dims>(input_shape),
             ToTensorShape<int32_t, num_dims>(filter_shape),
             ToTensorShape<int64_t, num_dims>(out_backprop_shape), dilations,
             strides, padding, explicit_paddings, data_format, &dims)
             .ok()) {
      return failure();
    }

    // The activations (inputs) form the LHS of the convolution.
    // Activations have shape: [batch, in_rows, in_cols, ..., in_depth]
    // For the gradient computation, we need to:
    // 1. In the case of group convolution, move the num_groups dimension before
    // the batch dimension
    // 2. Swap the roles of the batch and feature dimensions.
    const int feature_dim =
        tensorflow::GetTensorFeatureDimIndex(num_dims, data_format);
    const int64_t in_depth = input_shape[feature_dim];
    const int64_t filter_in_depth = *(filter_shape.begin() + num_spatial_dims);
    const int64_t feature_group_count = in_depth / filter_in_depth;

    if (feature_group_count != 1) {
      /*
          // TODO(parkers): translate this code to mlir.
          activations = TransposeInputForGroupConvolutionBackpropFilter(
              activations, input_shape, feature_group_count, batch_dim,
         feature_dim);
      */
      return failure();
    }

    // Compute ConvDimensionNumbers, dilation, and padding.
    SmallVector<int64_t, num_spatial_dims> spatial_dims;
    SmallVector<int64_t, num_spatial_dims> kernel_spatial_dims;
    SmallVector<int64_t, num_spatial_dims> rhs_dilation;
    SmallVector<int64_t, num_spatial_dims * 2> paddings;
    SmallVector<int64_t, num_spatial_dims> window_strides;

    // The filter gradients are computed by a convolution of the input
    // activations and the output gradients, with some appropriate padding.
    // See the comment at the top of conv_grad_ops.h for details.

    for (int i : llvm::seq<int>(0, num_spatial_dims)) {
      const int64_t dim =
          tensorflow::GetTensorSpatialDimIndex(num_dims, data_format, i);
      kernel_spatial_dims.push_back(dim);
      // Besides padding the input, we will also expand output_rows to
      //    expanded_out_rows = (output_rows - 1) * stride + 1
      // with zeros in between:
      //
      //      a . . . b . . . c . . . d . . . e
      //
      // This is done by specifying the window dilation factors in the
      // convolution HLO below.
      const auto &spatial_dim_i = dims.spatial_dims[i];
      rhs_dilation.push_back(spatial_dim_i.stride);
      window_strides.push_back(dilations[dim]);

      // We will also need to pad the input with zeros such that after the
      // convolution, we get the right size for the filter.
      // The padded_in_rows should be such that when we convolve this with the
      // expanded_out_rows as a filter, we should get filter_rows back.

      const int64_t padded_in_size =
          spatial_dim_i.expanded_output_size +
          (spatial_dim_i.filter_size - 1) * dilations[dim];

      // However it can be smaller than input_rows: in this
      // case it means some of the inputs are not used.
      //
      // An example is to have input_cols = 3, filter_cols = 2 and stride = 2:
      //
      // INPUT =  [ A  B  C ]
      //
      // FILTER = [ x y ]
      //
      // and the output will only have one column: a = A * x + B * y
      //
      // and input "C" is not used at all.
      //
      // We apply negative padding in this case.
      const int64_t pad_total = padded_in_size - spatial_dim_i.input_size;

      // + For the EXPLICIT padding, we pad the top/left side with the explicit
      //   padding and pad the bottom/right side with the remaining space.
      // + For the VALID padding, we don't pad anything on the top/left side
      //   and pad the bottom/right side with the remaining space.
      // + For the SAME padding, we pad top/left side the same as bottom/right
      //   side.
      //
      // In addition, if the padded input size is smaller than the input size,
      // we need to ignore some training elements of the input. We do this by
      // applying negative padding on the right/bottom.
      const int64_t pad_before = padding == tensorflow::Padding::EXPLICIT
                                     ? explicit_paddings[2 * dim]
                                     : padding == tensorflow::Padding::SAME
                                           ? std::max<int64_t>(pad_total / 2, 0)
                                           : 0;
      paddings.push_back(pad_before);
      paddings.push_back(pad_total - pad_before);
    }

    RankedTensorType paddings_ty = RankedTensorType::get(
        {num_spatial_dims, 2}, rewriter.getIntegerType(64));
    auto paddings_attr = DenseIntElementsAttr::get(paddings_ty, paddings);
    auto kernel_spatial_dims_attr =
        GetI64ElementsAttr(kernel_spatial_dims, &rewriter);

    const int batch_dim =
        tensorflow::GetTensorBatchDimIndex(num_dims, data_format);
    auto batch_dim_attr = rewriter.getI64IntegerAttr(batch_dim);
    auto feature_dim_attr = rewriter.getI64IntegerAttr(feature_dim);

    Value result = rewriter.create<ConvOp>(
        op.getLoc(), op.getType(), op.input(), op.out_backprop(),
        /*window_strides=*/GetI64ElementsAttr(window_strides, &rewriter),
        /*padding=*/paddings_attr, /*lhs_dilation=*/
        GetI64ElementsAttrForValue(/*size=*/num_spatial_dims, /*val=*/1,
                                   &rewriter),
        GetI64ElementsAttr(rhs_dilation, &rewriter),
        ConvDimensionNumbers::get(
            // Swap batch_dim and feature_dim in the activations.
            /*input_batch_dimension=*/feature_dim_attr,
            /*input_feature_dimension=*/batch_dim_attr,
            /*input_spatial_dimensions=*/kernel_spatial_dims_attr,
            // The gradients become the RHS of the convolution.
            // The gradients have shape [batch, out_rows, out_cols, ...,
            // out_depth] where the batch becomes the input feature for the
            // convolution.
            /*kernel_input_feature_dimension=*/batch_dim_attr,
            /*kernel_output_feature_dimension=*/feature_dim_attr,
            /*kernel_spatial_dimensions=*/kernel_spatial_dims_attr,
            /*output_batch_dimension=*/
            rewriter.getI64IntegerAttr(num_spatial_dims),
            /*output_feature_dimension=*/
            rewriter.getI64IntegerAttr(num_spatial_dims + 1),
            /*output_spatial_dimensions=*/
            GetI64ElementsAttrForSeq(0, num_spatial_dims, &rewriter),
            rewriter.getContext()),
        rewriter.getI64IntegerAttr(feature_group_count),
        /*batch_group_count=*/rewriter.getI64IntegerAttr(1),
        /*precision_config=*/ArrayAttr());

    rewriter.replaceOp(op, {result});

    return success();
  }
};

using ConvertConv2DBackpropFilterOp =
    ConvertConvBackpropFilterOp<TF::Conv2DBackpropFilterOp,
                                /*num_spatial_dims=*/2>;
using ConvertConv3DBackpropFilterOp =
    ConvertConvBackpropFilterOp<TF::Conv3DBackpropFilterV2Op,
                                /*num_spatial_dims=*/3>;

class ConvertOneHotOp : public OpRewritePattern<TF::OneHotOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::OneHotOp op,
                                PatternRewriter &rewriter) const override {
    auto indices_ty = op.indices().getType().dyn_cast<RankedTensorType>();
    if (!indices_ty || !indices_ty.hasStaticShape()) return failure();
    ArrayRef<int64_t> indices_shape = indices_ty.getShape();
    Type element_type = indices_ty.getElementType();

    DenseIntElementsAttr depth_attr;
    if (!matchPattern(op.depth(), m_Constant(&depth_attr))) {
      return failure();
    }

    int64_t depth = depth_attr.getValue<APInt>({}).getSExtValue();
    int64_t axis = op.axis().getSExtValue();
    if (axis == -1) axis = indices_shape.size();

    llvm::SmallVector<int64_t, 4> broadcast_dims(indices_shape.size());
    std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
    std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);

    llvm::SmallVector<int64_t, 4> output_dims =
        llvm::to_vector<4>(indices_shape);
    output_dims.insert(output_dims.begin() + axis, depth);

    Location loc = op.getLoc();

    // The iota result is the effective output shape of the computation,
    // and indices must be broadcast into it. At this point, this computation
    // would need to be reworked quite a bit to support dynamic shapes, so
    // just using static broadcasting.
    auto index_type = RankedTensorType::get(output_dims, element_type);
    auto iota = rewriter.create<IotaOp>(
        loc, index_type, IntegerAttr::get(rewriter.getIntegerType(64), axis));
    auto broadcast_indices = rewriter.create<BroadcastInDimOp>(
        loc, index_type, op.indices(),
        GetI64ElementsAttr(broadcast_dims, &rewriter));

    Value compare = rewriter.create<xla_hlo::CompareOp>(
        loc, broadcast_indices, iota,
        StringAttr::get("EQ", rewriter.getContext()));
    Value on_value = rewriter.create<BroadcastOp>(
        loc, op.getType(), op.on_value(),
        GetI64ElementsAttr(output_dims, &rewriter));
    Value off_value = rewriter.create<BroadcastOp>(
        loc, op.getType(), op.off_value(),
        GetI64ElementsAttr(output_dims, &rewriter));
    Value result = rewriter.create<SelectOp>(loc, op.getType(), compare,
                                             on_value, off_value);

    rewriter.replaceOp(op, {result});

    return success();
  }
};

// Converts InfeedDequeueTuple to XLA HLO create_token, infeed and
// get_tuple_element ops.
//
// All HLO infeed ops expect a HLO token type operand and produce a tuple
// containing a token. This HLO token type is used to order multiple infeed
// operations within a computation. The token type can come from other
// infeed/outfeed/send/recv ops or can be generated using create_token op with
// no operands. Here we emit a create_token op to generate the token type
// operand of infeed.
//
// For example the following IR:
// %0:2 = "tf.InfeedDequeueTuple"() : () -> (tensor<3xi32>, tensor<4xf32>)
//
// would be lowered to
//
// %token = "xla_hlo.create_token"() : () -> !xla_hlo.token
// %data_and_token = "xla_hlo.infeed"(%token) {infeed_config = ""} :
//      (!xla_hlo.token) -> tuple<tuple<tensor<3xi32>, tensor<4xf32>>,
//      !xla_hlo.token>
// %data = "xla_hlo.get_tuple_element"(%data_and_token) {index = 0}
// %0#0 = "xla_hlo.get_tuple_element"(%data) {index = 0}
// %0#1 = "xla_hlo.get_tuple_element"(%data) {index = 1}
//
class ConvertInfeedDequeueTupleOp
    : public OpRewritePattern<TF::InfeedDequeueTupleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::InfeedDequeueTupleOp op,
                                PatternRewriter &rewriter) const override {
    std::vector<Type> result_types(op.outputs().size());
    for (auto idx_and_output : llvm::enumerate(op.outputs())) {
      result_types[idx_and_output.index()] = (idx_and_output.value().getType());
    }
    // Infeed takes a single token operand. Generate the token using
    // create_token op to pass to the infeed op.
    auto token = rewriter.create<CreateTokenOp>(
        op.getLoc(), xla_hlo::TokenType::get(rewriter.getContext()));

    // Emit infeed op.
    // The result type of infeed is a tuple(tuple(result types), token type).
    auto data_tuple_type =
        mlir::TupleType::get(result_types, rewriter.getContext());
    auto data_and_token_type = mlir::TupleType::get(
        {data_tuple_type, token.getType()}, rewriter.getContext());

    auto data_and_token =
        rewriter.create<InfeedOp>(op.getLoc(), data_and_token_type, token,
                                  /*infeed_config=*/rewriter.getStringAttr(""));
    if (op._XlaSharding().hasValue()) {
      // _XlaSharding attribute in TF is a serialized string of the OpSharding
      // proto, so convert to a text form here.
      ::xla::OpSharding sharding_proto;
      if (!sharding_proto.ParseFromString(op._XlaSharding().getValue().str()))
        return failure();

      // Token is a control signal and not a real data, so arbitrarily assign
      // the token to device 0.
      if (sharding_proto.type() == ::xla::OpSharding::TUPLE) {
        *sharding_proto.add_tuple_shardings() =
            ::xla::sharding_builder::AssignDevice(0);
        data_and_token.setAttr(
            kShardingAttr,
            rewriter.getStringAttr(sharding_proto.SerializeAsString()));
      } else {
        data_and_token.setAttr(kShardingAttr, op._XlaShardingAttr());
      }
    }

    // The infeed instruction produces a tuple of the infeed data and a token
    // type. Emit get_tuple_element to get infeed data tuple.
    auto data_tuple = rewriter.create<GetTupleElementOp>(
        op.getLoc(), data_tuple_type, data_and_token,
        rewriter.getI32IntegerAttr(0));

    // Emit get_tuple_element for each result.
    std::vector<Value> results;
    for (auto idx_and_type : llvm::enumerate(result_types)) {
      auto tuple_element = rewriter.create<GetTupleElementOp>(
          op.getLoc(), idx_and_type.value(), data_tuple,
          rewriter.getI32IntegerAttr(idx_and_type.index()));
      results.push_back(tuple_element);
    }
    rewriter.replaceOp(op, ValueRange(results));
    return success();
  }
};

// Converts tf.OutfeedEnqueueTuple to XLA HLO tuple, create_token and outfeed
// ops.
//
// XLA HLO outfeed op expects a token, which we generate by emitting an
// create_token op.
//
// For example the following IR:
// "tf.OutfeedEnqueueTuple"(%val_1, %val_2) : (tensor<3xi32>, tensor<4xf32>) ->
//      ()
//
// would be lowered to
//
// %tuple = "xla_hlo.tuple"(%val_1, %val_2) : (tensor<3xi32>, tensor<4xf32>) ->
//      tuple<tensor<3xi32>, tensor<4xf32>>
// %token = "xla_hlo.create_token"() : () -> !xla_hlo.token
// %outfeed_token = "xla_hlo.outfeed"(%tuple, %token) {outfeed_config = ""} :
//      (tuple<tensor<3xi32>, tensor<4xf32>>, !xla_hlo.token) -> !xla_hlo.token
//
class ConvertOutfeedEnqueueTupleOp
    : public OpRewritePattern<TF::OutfeedEnqueueTupleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::OutfeedEnqueueTupleOp op,
                                PatternRewriter &rewriter) const override {
    auto token_type = xla_hlo::TokenType::get(rewriter.getContext());
    auto tuple = rewriter.create<TupleOp>(op.getLoc(), op.inputs());
    auto token = rewriter.create<CreateTokenOp>(op.getLoc(), token_type);
    rewriter.create<OutfeedOp>(op.getLoc(), token_type, tuple, token,
                               /*outfeed_config=*/rewriter.getStringAttr(""));
    rewriter.eraseOp(op);
    return success();
  }
};

// Converts tf.TopKV2 to XLA HLO iota, sort, and slice ops when k is a constant.
//
// tf.TopKV2 sorts along last dimension of the input tensor and then returns
// the top K components' values and indices. This is translated into a few
// ops in XLA HLO: first generating an integer sequence for the indices,
// then sort both the original input tensor and the indices togheter, and
// at last slice out the top K components.
//
// For example, for the following IR:
//
// %k = "tf.Const"() {value = dense<8> : tensor<i32>} : () -> tensor<i32>
// %0:2 = "tf.TopKV2"(%input, %k): (tensor<16x16xf32>, tensor<i32>) ->
//                                 (tensor<16x8xf32>, tensor<16x8xi32>)
//
// We will get:
//
// %1 = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<16x16xi32>
// %2 = "xla_hlo.sort"(%input, %1) ( {
// ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>,
//      %arg3: tensor<i32>, %arg4: tensor<i32>):
//   %7 = "xla_hlo.compare"(%arg1, %arg2) {comparison_direction = "GT"}: ...
//   "xla_hlo.return"(%7) : (tensor<i1>) -> ()
// }) {dimension = 1 : i64, is_stable = true} : ...
// %3 = "xla_hlo.get_tuple_element"(%2) {index = 0 : i32} : ...
// %4 = "xla_hlo.get_tuple_element"(%2) {index = 1 : i32} : ...
// %5 = "xla_hlo.slice"(%3) {limit_indices = dense<[16, 8]> : tensor<2xi64>,
//                           start_indices dense<0> : tensor<2xi64>,
//                           strides = dense<1> : tensor<2xi64>} :
//                              (tensor<16x16xf32>) -> tensor<16x8xf32>
// %6 = "xla_hlo.slice"(%4) ...
class ConvertTopKV2Op : public OpRewritePattern<TF::TopKV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::TopKV2Op op,
                                PatternRewriter &rewriter) const override {
    // We can only match when the `k` operand is a constant scalar.
    DenseIntElementsAttr k_attr;
    if (!matchPattern(op.k(), m_Constant(&k_attr))) return failure();

    // The last dimension of the input tensor's shape should be known so we can
    // have clamped end_indices for slices.
    TensorType input_type = op.input().getType().cast<TensorType>();
    if (!input_type.hasRank()) return failure();
    int64_t input_rank = input_type.getRank();
    int64_t last_dim_index = input_rank - 1;
    int64_t last_dim_size = input_type.getDimSize(last_dim_index);
    if (last_dim_size == ShapedType::kDynamicSize) return failure();

    // Create an Itoa op for indices.
    auto i32_type = rewriter.getIntegerType(32);
    Type iota_type = RankedTensorType::get(input_type.getShape(), i32_type);
    Value iota_op = rewriter.create<xla_hlo::IotaOp>(
        op.getLoc(), iota_type, rewriter.getI64IntegerAttr(last_dim_index));

    // Create the sort op. It takes two inputs, one for the original input, the
    // other for the indices.
    auto sort_op = rewriter.create<xla_hlo::SortOp>(
        op.getLoc(), llvm::ArrayRef<Value>{op.input(), iota_op}, last_dim_index,
        /*is_stable=*/true);
    BuildSortComparisonBody({input_type.getElementType(), i32_type},
                            /*direction=*/"GT", &sort_op.comparator(),
                            &rewriter);

    // Get the sorted input and index tuple element.
    auto tuple_first_element =
        rewriter.create<xla_hlo::GetTupleElementOp>(op.getLoc(), sort_op, 0);
    auto tuple_second_element =
        rewriter.create<xla_hlo::GetTupleElementOp>(op.getLoc(), sort_op, 1);

    SmallVector<int64_t, 4> begin_indices(input_rank, 0);
    auto end_indices = llvm::to_vector<4>(input_type.getShape());
    end_indices.back() =
        std::min((*k_attr.begin()).getSExtValue(), last_dim_size);
    SmallVector<int64_t, 4> strides(input_rank, 1);

    // Get the slice for the top K elements.

    Value values = rewriter.create<xla_hlo::SliceOp>(
        op.getLoc(), tuple_first_element,
        GetI64ElementsAttr(begin_indices, &rewriter),
        GetI64ElementsAttr(end_indices, &rewriter),
        GetI64ElementsAttr(strides, &rewriter));

    Value indices = rewriter.create<xla_hlo::SliceOp>(
        op.getLoc(), tuple_second_element,
        GetI64ElementsAttr(begin_indices, &rewriter),
        GetI64ElementsAttr(end_indices, &rewriter),
        GetI64ElementsAttr(strides, &rewriter));

    rewriter.replaceOp(op, {values, indices});
    return success();
  }
};

// Converts tf.Unpack to a series of XLA HLO slice ops.
//
// Each slice takes one element along the dimension to unpack and takes the full
// range for all other dimensions. Each slice is then reshaped to drop the
// dimension to unpack (which is always of size 1).
// TODO(antiagainst): consider changing this into a TF internal lowering pass.
class ConvertUnpackOp : public OpRewritePattern<TF::UnpackOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto value_type = op.value().getType().cast<RankedTensorType>();
    if (!value_type) return failure();

    int64_t value_rank = value_type.getRank();
    int64_t axis = op.axis().getSExtValue();
    if (axis < 0) axis += value_rank;

    // Parameters for constructing each slice.
    SmallVector<int64_t, 4> begin_indices(value_rank, 0);
    auto end_indices = llvm::to_vector<4>(value_type.getShape());
    SmallVector<int64_t, 4> strides(value_rank, 1);

    // All HLO slice+reshape results used to replace the original tf.Unpack op.
    SmallVector<Value, 4> results;
    results.reserve(op.getNumResults());

    for (int i = 0; i < op.getNumResults(); ++i) {
      begin_indices[axis] = i;
      end_indices[axis] = i + 1;

      auto slice_op = rewriter.create<xla_hlo::SliceOp>(
          op.getLoc(), op.value(), GetI64ElementsAttr(begin_indices, &rewriter),
          GetI64ElementsAttr(end_indices, &rewriter),
          GetI64ElementsAttr(strides, &rewriter));
      // Reshape to drop the axis dimension.
      auto reshape_op = rewriter.create<xla_hlo::ReshapeOp>(
          op.getLoc(), op.getType(i), slice_op);
      results.push_back(reshape_op);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

// Converts TF unsorted segment reduction ops to XLA HLO scatter op.
//
// TF unsorted segment reduction op peforms the following calculation:
//
// Assume segment ids' shape is [SI0, SI1, ..., SIm] and data's  shape is
// [D0, D1, ..., Dn]. Note that segment ids' shape must be a prefix of data's
// shape, so we can have data's shape represented as [SI0, SI1, ..., SIm,
// Dm+1, ..., Dn]. Then
//   output[segment_ids[SI_i0, SI_i1, ..., SI_im], D_im+1, ..., D_in] =
//      <ReductionOp> over data[SI_i0, SI_i1, ..., SI_im, D_im+1, ..., D_in]
// where SI_iN is in the range of [0, SIN) and D_iN is in the range of [0, DN).
//
// The op will be translated to XLA HLO scatter with the following parameters:
// * Update window dims is [segment_id_rank, data_rank).
// * Inserted window dims is {0}.
// * Scatter dims to operand dims mapping is {0}.
// * Index vector dim is segment_id_rank.
template <typename ConcreteClass, typename OpTy, typename ReductionOp>
class GenericConvertUnsortedSegmentReductionOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto data_type = op.data().getType().template dyn_cast<RankedTensorType>();
    if (!data_type) return failure();
    int64_t data_rank = data_type.getRank();

    auto segment_ids_type =
        op.segment_ids().getType().template dyn_cast<RankedTensorType>();
    if (!segment_ids_type) return failure();
    int64_t segment_ids_rank = segment_ids_type.getRank();

    DenseIntElementsAttr num_segments_attr;
    if (!matchPattern(op.num_segments(), m_Constant(&num_segments_attr)))
      return failure();

    // The final shape for TF unsorted segment reduction op is [num_segments] +
    // data_shape[segment_ids_rank:].
    SmallVector<int64_t, 4> output_shape;
    output_shape.push_back((*num_segments_attr.begin()).getSExtValue());
    auto suffix = data_type.getShape().drop_front(segment_ids_rank);
    output_shape.append(suffix.begin(), suffix.end());
    auto output_type =
        RankedTensorType::get(output_shape, data_type.getElementType());

    // Broadccast the initial value for reduction. This will become the
    // 'operand' parameter to scatter to for the final scatter op.
    Value init = ConcreteClass::GetInitialValue(data_type.getElementType(),
                                                op.getLoc(), &rewriter);
    auto broadcasted_init = rewriter.create<xla_hlo::BroadcastOp>(
        op.getLoc(), output_type, init,
        GetI64ElementsAttr(output_shape, &rewriter));

    // Parameters for the generated scatter op.
    SmallVector<int64_t, 1> inserted_window_dims(1, 0);
    SmallVector<int64_t, 1> scatter_dims_to_operand_dims(1, 0);
    int64_t index_vector_dim = segment_ids_rank;

    // Put all parameters in a StructAttr.
    auto dims_attr = ScatterDimensionNumbers::get(
        GetI64ElementsAttrForSeq(segment_ids_rank, data_rank, &rewriter),
        GetI64ElementsAttr(inserted_window_dims, &rewriter),
        GetI64ElementsAttr(scatter_dims_to_operand_dims, &rewriter),
        rewriter.getI64IntegerAttr(index_vector_dim), rewriter.getContext());

    auto scatter =
        rewriter.create<ScatterOp>(op.getLoc(), op.getType(), broadcasted_init,
                                   op.segment_ids(), op.data(), dims_attr);
    BuildReduceBody<ReductionOp>(data_type.getElementType(),
                                 &scatter.update_computation(), &rewriter);

    rewriter.replaceOp(op, scatter.getResult());
    return success();
  }
};

class ConvertUnsortedSegmentMaxOp
    : public GenericConvertUnsortedSegmentReductionOp<
          ConvertUnsortedSegmentMaxOp, TF::UnsortedSegmentMaxOp, MaxOp> {
 public:
  using GenericConvertUnsortedSegmentReductionOp::
      GenericConvertUnsortedSegmentReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetMinValueForType(reduce_element_type, loc, rewriter);
  }
};

class ConvertUnsortedSegmentMinOp
    : public GenericConvertUnsortedSegmentReductionOp<
          ConvertUnsortedSegmentMinOp, TF::UnsortedSegmentMinOp, MinOp> {
 public:
  using GenericConvertUnsortedSegmentReductionOp::
      GenericConvertUnsortedSegmentReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetMaxValueForType(reduce_element_type, loc, rewriter);
  }
};

class ConvertUnsortedSegmentProdOp
    : public GenericConvertUnsortedSegmentReductionOp<
          ConvertUnsortedSegmentProdOp, TF::UnsortedSegmentProdOp, MulOp> {
 public:
  using GenericConvertUnsortedSegmentReductionOp::
      GenericConvertUnsortedSegmentReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 1, rewriter);
  }
};

class ConvertUnsortedSegmentSumOp
    : public GenericConvertUnsortedSegmentReductionOp<
          ConvertUnsortedSegmentSumOp, TF::UnsortedSegmentSumOp, AddOp> {
 public:
  using GenericConvertUnsortedSegmentReductionOp::
      GenericConvertUnsortedSegmentReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 0, rewriter);
  }
};

// Converts tf.RandomShuffle op into a series of XLA HLO ops.
//
// tf.RandomShuffle shuffles tensors along the first dimension. If the input
// tensor's rank is 1, then it is translated into HLO sort op(s) according to
// indices randomly generated via HLO rng_uniform ops. Otherwise, it is
// translated into an HLO while op to first emulate shuffling indices using
// HLO dynamic_slice and dynamic_update_slice ops, then finally HLO gather
// with the shuffled indices.
class ConvertRandomShuffleOp : public OpRewritePattern<TF::RandomShuffleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::RandomShuffleOp op,
                                PatternRewriter &rewriter) const override {
    auto input_type = op.value().getType().dyn_cast<RankedTensorType>();
    if (!input_type) return failure();

    int64_t input_rank = input_type.getRank();
    int64_t first_dim_size = input_type.getDimSize(0);
    if (ShapedType::isDynamic(first_dim_size)) return failure();

    // We are shuffling along the first dimension. If its size is <= 1, then
    // shuffling is a no-op.
    if (first_dim_size <= 1) {
      rewriter.replaceOp(op, op.value());
      return success();
    }

    // For vectors, shuffle values by sorting instead of the obvious
    // Fisher-Yates algorithm. Fisher-Yates is simple to implement and correct,
    // but not easily parallelizable. For a sufficiently parallel architecture,
    // it is faster to sort many times, than Fisher-Yates shuffle once.
    if (input_rank == 1) {
      // Shuffle values by assigning each value a random key and sorting the
      // keys. Keys can collide causing detectable patterns in the shuffled
      // output. Collisions translates into more ascending sub-sequences in the
      // shuffled output than would be expected by chance. To avoid collisions,
      // the number of possible key values must be sufficiently large.

      // How are more than 2^32 keys created? In each loop iteration, the
      // algorithm sorts by random keys. Conceptually, the earlier iterations
      // are sorting on the lower-order bits of larger keys that are never
      // actually assembled.

      // The expected number of collisions is n - d + d(1 - 1/d)^n, where d is
      // the number of possible keys and n is the number of values. If d = n^2,
      // then the limit as n goes to infinity is 1/2. If d = n^3, then the limit
      // as n goes to infinity is zero.

      // This implementation ensures that the key-space is greater than or equal
      // to the cube of the number of values. The risk of collisions can be
      // further reduced by increasing Exponent at the expense of
      // performance.

      // For Exponent = 2, the expected number of collisions per shuffle is
      // maximized at n = floor((2^32-1)^(1/2)) = 65535 where the expectation is
      // about 1/2.

      // For Exponent = 3, the expected number of collisions per shuffle is
      // maximized at n = floor((2^32-1)^(1/3)) = 1625 where the expectation is
      // about 1/3255.

      // For Exponent = 4, the expected number of collisions per shuffle is
      // maximized at n = floor((2^32-1)^(1/4)) = 255 where the expectation is
      // about 1/132622.
      constexpr int exponent = 3;
      int64_t num_elements = input_type.getNumElements();
      uint32_t u32_max = std::numeric_limits<uint32_t>::max();
      int rounds =
          std::ceil(exponent * std::log(num_elements) / std::log(u32_max));

      Value current = op.value();
      for (int i = 0; i < rounds; ++i) {
        auto keys =
            CreateRngUniform32(op.getLoc(), num_elements, /*lower_limit=*/0,
                               /*upper_limit=*/u32_max, &rewriter);
        auto sorted = rewriter.create<xla_hlo::SortOp>(
            op.getLoc(), llvm::ArrayRef<Value>{keys, current});
        auto i32_type = rewriter.getIntegerType(32);
        BuildSortComparisonBody({i32_type, input_type.getElementType()},
                                /*direction=*/"LT", &sorted.comparator(),
                                &rewriter);
        current = rewriter.create<GetTupleElementOp>(op.getLoc(),
                                                     sorted.getResult(), 1);
      }
      rewriter.replaceOp(op, current);
      return success();
    }

    // The Fisher-Yates algorithm.

    // Generate range(n) as the initial value for the indices to be swapped.
    auto indices_type =
        RankedTensorType::get({first_dim_size}, rewriter.getIntegerType(32));
    Value indices = rewriter.create<xla_hlo::IotaOp>(
        op.getLoc(), indices_type, rewriter.getI64IntegerAttr(0));

    // Generate random numbers to be used as swaps for the indices.
    Value swaps = CreateRngUniform32(op.getLoc(), first_dim_size, 0,
                                     first_dim_size, &rewriter);

    // While loop body to perform index swaps.
    auto swap_body_fn = [&](Location loc, Value i, ArrayRef<Value> old_values,
                            SmallVectorImpl<Value> *new_values,
                            OpBuilder *builder) {
      Value swaps = old_values[0];
      Value indices = old_values[1];

      auto vec1_i32_type =
          RankedTensorType::get({1}, builder->getIntegerType(32));
      auto scalar_i32_type =
          RankedTensorType::get({}, builder->getIntegerType(32));
      auto scalar_i64_type =
          RankedTensorType::get({}, builder->getIntegerType(64));

      auto scalar_one =
          DenseIntElementsAttr::get(scalar_i64_type, ArrayRef<int64_t>(1));

      // We need to swap the indices[i] with indices[swaps[i]]. First get
      // these index values.
      Value source_index = builder->create<xla_hlo::DynamicSliceOp>(
          loc, vec1_i32_type, indices, i, scalar_one);
      Value swap_index = builder->create<xla_hlo::ReshapeOp>(
          loc, scalar_i32_type,
          builder->create<xla_hlo::DynamicSliceOp>(loc, vec1_i32_type, swaps, i,
                                                   scalar_one));
      Value target_index = builder->create<xla_hlo::DynamicSliceOp>(
          loc, vec1_i32_type, indices, swap_index, scalar_one);

      // Then perform the swap.
      // indices[i] <- indices[swaps[i]]
      indices = builder->create<xla_hlo::DynamicUpdateSliceOp>(
          loc, indices.getType(), indices, target_index, llvm::makeArrayRef(i));
      // indices[swaps[i]] <- indices[i]
      indices = builder->create<xla_hlo::DynamicUpdateSliceOp>(
          loc, indices.getType(), indices, source_index,
          llvm::makeArrayRef(swap_index));

      // Update new values.
      new_values->assign({swaps, indices});
    };

    // Create a while op to swap indices.
    SmallVector<Value, 2> while_output;
    CreateWhile32(op.getLoc(), first_dim_size, swap_body_fn, {swaps, indices},
                  &while_output, &rewriter);
    Value swaped_indices = while_output[1];

    // Gather the data using the swapped indices as the shuffled order.
    ArrayRef<int64_t> input_shape = input_type.getShape();
    SmallVector<int64_t, 4> slice_sizes(input_shape.begin(), input_shape.end());
    slice_sizes[0] = 1;
    auto dims_attr = GatherDimensionNumbers::get(
        /*offset_dims=*/GetI64ElementsAttrForSeq(1, first_dim_size, &rewriter),
        /*collapsed_slice_dims=*/GetI64ElementsAttr({0}, &rewriter),
        /*start_index_map=*/GetI64ElementsAttr({0}, &rewriter),
        /*index_vector_dim=*/rewriter.getI64IntegerAttr(1),
        rewriter.getContext());
    rewriter.replaceOpWithNewOp<xla_hlo::GatherOp>(
        op, op.getType(), op.value(), swaped_indices, dims_attr,
        GetI64ElementsAttr(slice_sizes, &rewriter));

    return success();
  }
};

// Converts an XlaSharding op to a XLA HLO shard op with sharding attributes.
class ConvertXlaShardingOp : public OpRewritePattern<TF::XlaShardingOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaShardingOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(b/148313088): define sharding attribute struct in MLIR intead of
    // using a string.
    if (!op._XlaSharding().hasValue()) return failure();

    auto custom_call = rewriter.create<xla_hlo::CustomCallOp>(
        op.getLoc(), op.getType(), op.input(),
        /*call_target_name=*/rewriter.getStringAttr("Sharding"),
        /*has_side_effect=*/rewriter.getBoolAttr(false),
        /*backend_config=*/rewriter.getStringAttr(""));
    custom_call.setAttr(kShardingAttr, op._XlaShardingAttr());
    rewriter.replaceOp(op, custom_call.getResult());

    return success();
  }
};

// Converts a TF InplaceUpdate op to DynamicUpdateSlice HLO.
class ConvertInplaceUpdateOp : public OpRewritePattern<TF::InplaceUpdateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::InplaceUpdateOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.x();
    auto indices = op.i();
    auto updates = op.v();

    // Slice each row of `i` and `v` to perform a separate dynamic-update-slice
    // on the contents of `x`.
    auto input_type = input.getType().cast<ShapedType>();
    auto updates_type = updates.getType().cast<ShapedType>();
    auto indices_type = indices.getType().cast<ShapedType>();
    if (!indices_type.hasStaticShape()) return failure();

    if (indices_type.getRank() != 1) return failure();

    SmallVector<Type, 4> unpacked_indices_type(
        indices_type.getDimSize(0),
        RankedTensorType::get({}, indices_type.getElementType()));
    auto zero_attr = IntegerAttr::get(rewriter.getIntegerType(64), 0);
    auto unpacked_indices = rewriter.create<TF::UnpackOp>(
        op.getLoc(), unpacked_indices_type, indices, zero_attr);

    SmallVector<int64_t, 4> split_updates_shape;
    split_updates_shape.append(updates_type.getShape().begin(),
                               updates_type.getShape().end());
    split_updates_shape.front() = 1;
    SmallVector<Type, 4> split_updates_type;
    split_updates_type.resize(
        updates_type.getShape().front(),
        RankedTensorType::get(split_updates_shape,
                              updates_type.getElementType()));

    auto cst =
        rewriter.create<xla_hlo::ConstOp>(op.getLoc(), zero_attr).getResult();
    auto split_updates = rewriter.create<TF::SplitOp>(
        op.getLoc(), split_updates_type, cst, updates);

    SmallVector<Value, 6> input_indices;
    input_indices.resize(input_type.getRank(), cst);

    SmallVector<int64_t, 6> starts(updates_type.getRank(), 0);
    SmallVector<int64_t, 6> strides(updates_type.getRank(), 1);
    SmallVector<int64_t, 6> limits(updates_type.getShape().begin(),
                                   updates_type.getShape().end());

    for (auto pair :
         llvm::zip(unpacked_indices.output(), split_updates.output())) {
      input_indices.front() = std::get<0>(pair);
      input = rewriter.create<xla_hlo::DynamicUpdateSliceOp>(
          op.getLoc(), op.getType(), input, std::get<1>(pair), input_indices);
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};

// Converts a TF XlaDynamicUpdateSlice op to DynamicUpdateSlice HLO.
class ConvertXlaDynamicUpdateSliceOp
    : public OpRewritePattern<TF::XlaDynamicUpdateSliceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaDynamicUpdateSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto indices_type = op.indices().getType().dyn_cast<RankedTensorType>();
    if (!indices_type || !indices_type.hasStaticShape() ||
        indices_type.getShape().size() != 1)
      return failure();

    SmallVector<Type, 4> unpacked_indices_type(
        indices_type.getDimSize(0),
        RankedTensorType::get({}, indices_type.getElementType()));
    auto unpacked_indices = rewriter.create<TF::UnpackOp>(
        op.getLoc(), unpacked_indices_type, op.indices(),
        IntegerAttr::get(rewriter.getIntegerType(64), 0));
    rewriter.replaceOpWithNewOp<xla_hlo::DynamicUpdateSliceOp>(
        op, op.getType(), op.input(), op.update(), unpacked_indices.output());
    return success();
  }
};

/// Converts the Cumsum TensorFlow op to the HLO ReduceWindow op by setting
/// appropriate window dimensions, with 'add' as the reduction function.  The
/// input tensor needs to have a static shape, and 'axis' must be const.  The
/// TableGen pattern is not used for this rewrite because it involves regions.
class ConvertCumsumOp : public OpRewritePattern<TF::CumsumOp> {
  using OpRewritePattern<TF::CumsumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::CumsumOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.x();
    auto input_type = input.getType().dyn_cast<ShapedType>();
    if (!input_type || !input_type.hasStaticShape()) {
      return failure();
    }

    // TODO(jennik): Add support for the optional 'exclusive' and 'reverse'
    // arguments.
    if (op.exclusive() || op.reverse()) {
      return failure();
    }

    // We can only match when the axis is a constant scalar.
    DenseIntElementsAttr axis_attr;
    if (!matchPattern(op.axis(), m_Constant(&axis_attr))) {
      return failure();
    }

    // Convert if we need to enlarge the element type's bitwidth to avoid
    // precision loss.
    Type input_element_type = input_type.getElementType();
    Type sum_element_type = GetSumAccumulationType(input_element_type);
    input = rewriter.create<ConvertOp>(op.getLoc(), input, sum_element_type);

    ArrayRef<int64_t> input_shape = input_type.getShape();
    int64_t rank = input_shape.size();

    // Get the dimension to apply the reduction on, and offset properly if it is
    // negative.
    int64_t axis = (*axis_attr.begin()).getSExtValue();
    if (axis < 0) {
      axis += rank;
    }

    SmallVector<int64_t, 4> window_dims(rank, 1);
    SmallVector<int64_t, 4> window_strides(rank, 1);
    window_dims[axis] = input_shape[axis];

    SmallVector<int64_t, 8> paddings(rank * 2, 0);
    paddings[axis * 2] = input_shape[axis] - 1;
    auto paddings_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({rank, 2}, rewriter.getIntegerType(64)),
        paddings);

    Value init =
        GetScalarConstOfType(sum_element_type, op.getLoc(), 0, &rewriter);

    auto reduce = rewriter.create<ReduceWindowOp>(
        op.getLoc(), input_type, input, init,
        GetI64ElementsAttr(rewriter.getI64ArrayAttr(window_dims)),
        GetI64ElementsAttr(rewriter.getI64ArrayAttr(window_strides)),
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(), paddings_attr);
    BuildReduceBody<AddOp>(sum_element_type, &reduce.body(), &rewriter);
    Value result = reduce.getResult();

    // Convert back if we enlarged the element type's bitwidth.
    result =
        rewriter.create<ConvertOp>(op.getLoc(), result, input_element_type);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Converts the Tensorflow ShapeOp to a sequence of Shape dialect and Standard
// dialect lowerings. This involves extracting the shape type, extracting and
// converting each dimension to a known integer type, and repacking into a final
// tensor.
class ConvertShapeOp : public OpRewritePattern<TF::ShapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::ShapeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();

    auto shape_op = rewriter.create<shape::ShapeOfOp>(op.getLoc(), input);
    auto result_ty = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!result_ty) {
      return failure();
    }

    auto index_tensor =
        RankedTensorType::get(result_ty.getShape(), rewriter.getIndexType());
    auto extent_tensor = rewriter.create<shape::ToExtentTensorOp>(
        op.getLoc(), index_tensor, shape_op);

    rewriter.replaceOpWithNewOp<IndexCastOp>(op, result_ty, extent_tensor);
    return success();
  }
};

// Converts a TF QR op to HLO.
class ConvertQrOp : public OpRewritePattern<TF::QrOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::QrOp op,
                                PatternRewriter &rewriter) const override {
    // Block Householder QR Factorization. Algorithm 5.2.2 of Golub and van
    // Loan. def qr_blocked(a, block_size):
    //   m = a.shape[0]
    //   n = a.shape[1]
    //   q = np.eye(m)
    //   for i in xrange(0, min(m, n), block_size):
    //     k = min(block_size, min(m, n) - s)
    //     (a, vs, taus) = qr(a[i:, i:i+k])
    //     y = vs
    //     w = ComputeWYRepresentation(vs, taus, m-i, k)
    //     a[i:, i+r:] += np.dot(y, np.dot(w.T, a[i:, i+k:]))
    //     q[:, i:] += np.dot(q[:, i:], np.dot(w, y.T))
    //   return (q, a)
    auto type = op.input().getType().dyn_cast<RankedTensorType>();
    if (!type || !type.hasStaticShape()) return failure();
    // The block size is chosen to match old bridge lowering.
    constexpr int64_t kBlockSize = 128;
    Value a = op.input();
    int64_t m = type.getDimSize(type.getRank() - 2);
    int64_t n = type.getDimSize(type.getRank() - 1);
    int64_t p = std::min(m, n);
    auto batch_dims = type.getShape().drop_back(2);
    auto iota_type = RankedTensorType::get({m, m}, rewriter.getIntegerType(32));
    auto iota0 = rewriter.create<IotaOp>(op.getLoc(), iota_type,
                                         rewriter.getI64IntegerAttr(0));
    auto iota1 = rewriter.create<IotaOp>(op.getLoc(), iota_type,
                                         rewriter.getI64IntegerAttr(1));
    Value compare = rewriter.create<CompareOp>(
        op.getLoc(), iota0, iota1,
        StringAttr::get("EQ", rewriter.getContext()));
    Value identity_matrix =
        rewriter.create<ConvertOp>(op.getLoc(), compare, type.getElementType());
    auto q_shape = llvm::to_vector<4>(type.getShape());
    q_shape.back() = m;
    Value q = rewriter.create<BroadcastOp>(
        op.getLoc(), RankedTensorType::get(q_shape, type.getElementType()),
        identity_matrix, GetI64ElementsAttr(batch_dims, &rewriter));
    auto precision_config = rewriter.getStrArrayAttr({"HIGHEST", "HIGHEST"});
    for (int64_t i = 0; i < p; i += kBlockSize) {
      int64_t k = std::min(kBlockSize, p - i);
      auto a_block =
          SliceInMinorDims(op.getLoc(), a, {i, i}, {m, i + k}, &rewriter);
      Value r_block;
      Value taus;
      Value vs;
      QRBlock(op.getLoc(), a_block, &r_block, &taus, &vs, &rewriter);
      a = UpdateSliceInMinorDims(op.getLoc(), a, r_block, {i, i}, &rewriter);

      // Compute the I-WY block representation of a product of Householder
      // matrices.
      Value w =
          ComputeWYRepresentation(op.getLoc(), type.getElementType(),
                                  batch_dims, vs, taus, m - i, k, &rewriter);
      auto y = vs;

      // a[i:, i+k:] += np.dot(Y, np.dot(W.T, a[i:, i+k:]))
      Value a_panel =
          SliceInMinorDims(op.getLoc(), a, {i, i + k}, {m, n}, &rewriter);
      auto a_update = BatchDot(op.getLoc(), w, true, a_panel, false,
                               batch_dims.size(), precision_config, &rewriter);
      a_update = BatchDot(op.getLoc(), y, false, a_update, false,
                          batch_dims.size(), precision_config, &rewriter);
      a_panel = rewriter.create<AddOp>(op.getLoc(), a_panel, a_update);
      a = UpdateSliceInMinorDims(op.getLoc(), a, a_panel, {i, i + k},
                                 &rewriter);

      // q[:, i:] += np.dot(np.dot(q[:, i:], W), Y.T))
      Value q_panel =
          SliceInMinorDims(op.getLoc(), q, {0, i}, {m, m}, &rewriter);
      Value q_update = BatchDot(op.getLoc(), q_panel, false, w, false,
                                batch_dims.size(), precision_config, &rewriter);
      q_update = BatchDot(op.getLoc(), q_update, false, y, true,
                          batch_dims.size(), precision_config, &rewriter);
      q_panel = rewriter.create<AddOp>(op.getLoc(), q_panel, q_update);
      q = UpdateSliceInMinorDims(op.getLoc(), q, q_panel, {i}, &rewriter);
    }
    // full_matrices is false when only a partial result in needed. Slice to the
    // needed dimensions here.
    if (!op.full_matrices()) {
      q = SliceInMinorDims(op.getLoc(), q, {0, 0}, {m, p}, &rewriter);
      a = SliceInMinorDims(op.getLoc(), a, {0, 0}, {p, n}, &rewriter);
    }
    rewriter.replaceOp(op, {q, a});
    return success();
  }

 private:
  // Computes a Householder reflection of the form:
  // H = I - tau v v.T.
  // such that
  // H . ( x1  ) = ( x1   )
  //     ( x2  ) = ( x2   )
  //     ( ... ) = ( ...  )
  //     ( xk  ) = ( beta )
  //     ( ... )   ( 0    )
  //     ( ... )   ( 0    )
  // Unlike the usual formulation, we allow the caller to supply 'k' rather than
  // only providing the relevant part of 'x' to maintain XLA's static shape
  // invariant. In addition, the implementation supports batching.
  // Pseudo-code, without batching:
  //   alpha = x[k]
  //   x_copy = np.copy(x)
  //   x_copy[:k+1] = 0
  //   xnorm = norm2(x_copy)
  //   if xnorm == 0:
  //     beta = alpha
  //     tau = 0
  //     v = np.zeros_like(x)
  //   else:
  //     beta = - np.sign(alpha) * dlapy2(alpha, xnorm)
  //     tau = (beta - alpha) / beta
  //     v = x / (alpha - beta)
  //   v[k] = 1
  //   return (v, tau, beta)
  void House(Location loc, Value x, Value k, ArrayRef<int64_t> batch_dims,
             const int64_t m, OpBuilder *builder, Value *v, Value *tau,
             Value *beta) const {
    auto x_type = x.getType().cast<RankedTensorType>();

    llvm::SmallVector<int64_t, 4> batch_dim_ids(batch_dims.size());
    std::iota(batch_dim_ids.begin(), batch_dim_ids.end(), 0);
    const int64_t minor_dim = batch_dims.size();

    Value zero = GetScalarConstOfType(x_type.getElementType(), loc, 0, builder);
    Value one = GetScalarConstOfType(x_type.getElementType(), loc, 1, builder);

    // alpha = x[k]
    Value alpha = DynamicSliceInMinorDims(loc, x, {k}, {1}, builder);
    alpha = builder->create<ReshapeOp>(
        loc, RankedTensorType::get(batch_dims, x_type.getElementType()), alpha);

    // Compute x[k+1:] (padded with zeros in elements 0..k)
    Value iota = builder->create<IotaOp>(
        loc, RankedTensorType::get({m}, builder->getIntegerType(32)),
        builder->getI64IntegerAttr(0));
    Value gtk = builder->create<xla_chlo::BroadcastCompareOp>(
        loc, iota, k, GetI64ElementsAttr({}, builder),
        StringAttr::get("GT", builder->getContext()));
    gtk = builder->create<ConvertOp>(loc, gtk, x_type.getElementType());
    Value x_after_k = builder->create<xla_chlo::BroadcastMulOp>(
        loc, x, gtk, GetI64ElementsAttr({minor_dim}, builder));
    Value x_after_k_sq = builder->create<MulOp>(loc, x_after_k, x_after_k);
    // sigma = np.dot(x[k+1:], x[k+1:])
    auto sigma = builder->create<ReduceOp>(
        loc, x_after_k_sq, zero, GetI64ElementsAttr({minor_dim}, builder));
    BuildReduceBody<AddOp>(x_type.getElementType(), &sigma.body(), builder);
    // mu = np.sqrt(x[k]*x[k] + sigma)
    Value alpha_sq = builder->create<MulOp>(loc, alpha, alpha);
    Value mu = builder->create<SqrtOp>(
        loc, builder->create<AddOp>(loc, alpha_sq, sigma.getResult(0)));

    Value sigma_is_zero = builder->create<xla_chlo::BroadcastCompareOp>(
        loc, sigma.getResult(0), zero, GetI64ElementsAttr({}, builder),
        StringAttr::get("EQ", builder->getContext()));
    Value alpha_is_negative = builder->create<xla_chlo::BroadcastCompareOp>(
        loc, alpha, zero, GetI64ElementsAttr({}, builder),
        StringAttr::get("LT", builder->getContext()));
    auto batch_size_one = builder->create<BroadcastOp>(
        loc, alpha.getType(), one, GetI64ElementsAttr(batch_dims, builder));
    Value signed_mu = builder->create<xla_chlo::BroadcastMulOp>(
        loc,
        builder->create<SelectOp>(loc, mu.getType(), alpha_is_negative,
                                  batch_size_one,
                                  builder->create<NegOp>(loc, batch_size_one)),
        mu, GetI64ElementsAttr({}, builder));
    *beta = builder->create<SelectOp>(loc, alpha.getType(), sigma_is_zero,
                                      alpha, signed_mu);
    *tau = builder->create<DivOp>(
        loc, builder->create<SubOp>(loc, *beta, alpha), *beta);
    Value zero_tau = builder->create<BroadcastOp>(
        loc, alpha.getType(), zero, GetI64ElementsAttr(batch_dims, builder));
    *tau = builder->create<SelectOp>(loc, alpha.getType(), sigma_is_zero,
                                     zero_tau, *tau);
    Value divisor = builder->create<SubOp>(loc, alpha, *beta);
    divisor = builder->create<SelectOp>(loc, divisor.getType(), sigma_is_zero,
                                        batch_size_one, divisor);

    Value eqk = builder->create<xla_chlo::BroadcastCompareOp>(
        loc, iota, k, GetI64ElementsAttr({}, builder),
        StringAttr::get("EQ", builder->getContext()));
    eqk = builder->create<ConvertOp>(loc, eqk, x_type.getElementType());
    llvm::SmallVector<int64_t, 4> e_k_shape(batch_dims.size(), 1);
    e_k_shape.push_back(m);
    auto e_k = builder->create<BroadcastOp>(
        loc, RankedTensorType::get(e_k_shape, x_type.getElementType()), eqk,
        GetI64ElementsAttr(llvm::SmallVector<int64_t, 4>(batch_dims.size(), 1),
                           builder));

    // Form v as [0, 0, ..., 1] ++ x[k+1:] / divisor
    // If sigma is zero, x[k+1:] is zero, so use any non-zero divisor.
    // Note that the add performs a degenerate broadcast.
    *v = builder->create<xla_chlo::BroadcastAddOp>(
        loc, e_k,
        StaticBinaryBroadcast<DivOp>(loc, x_after_k, divisor,
                                     GetI64ElementsAttr(batch_dim_ids, builder),
                                     *builder),
        /*broadcast_dimensions=*/nullptr);
  }

  // Householder QR decomposition. Algorithm 5.2.1 from Golub and Van
  // Loan "Matrix Computations", 4th Edition. This is an unblocked
  // implementation used as an inner routine of the blocked implementation.
  // Algorithm is adapted slightly so the shapes inside the loop are static, at
  // the cost of some redundant computation. Since this is used as an inner
  // block kernel, accumulates the Householder transformations (vs, taus) rather
  // than the matrix q. Equivalent Python code, without batching: def qr(a):
  //   m = a.shape[0]
  //   n = a.shape[1]
  //   vs = np.zeros([m, n])
  //   taus = np.zeros([n])
  //   for j in xrange(min(m, n)):
  //     v, tau, beta = house(a[:, j], j)
  //     # Unusually, we apply the Householder transformation to the entirety of
  //     # a, wasting FLOPs to maintain the static shape invariant that XLA
  //     # requires. For columns that precede j this has no effect.
  //     a[:, :] -= tau * np.dot(v[:, np.newaxis],
  //                              np.dot(v[np.newaxis, :], a[:, :]))
  //     # Form column j explicitly rather than relying on the precision of the
  //     # Householder update.
  //     a[j, j] = beta
  //     a[j+1:, j] = np.zeros([m - j - 1], dtype=a.dtype)
  //     vs[:, j] = v
  //     taus[j] = tau
  //   return (q, vs, taus)
  void QRBlock(Location loc, Value a, Value *r, Value *taus, Value *vs,
               PatternRewriter *rewriter) const {
    auto a_type = a.getType().cast<RankedTensorType>();
    const int num_dims = a_type.getRank();
    assert(num_dims >= 2 && "Argument to QR must have rank >= 2");

    const int64_t m = a_type.getDimSize(a_type.getRank() - 2);
    const int64_t n = a_type.getDimSize(a_type.getRank() - 1);

    const int64_t num_batch_dims = num_dims - 2;
    auto batch_dims = a_type.getShape().take_front(num_batch_dims);
    llvm::SmallVector<int64_t, 4> batch_dim_indices(batch_dims.size());
    std::iota(batch_dim_indices.begin(), batch_dim_indices.end(), 0);

    auto qr_body_fn = [&](Location loc, Value j, ArrayRef<Value> old_values,
                          SmallVectorImpl<Value> *new_values,
                          OpBuilder *builder) {
      auto a = old_values[0];
      auto vs = old_values[1];
      auto taus = old_values[2];

      // v, beta = house(a[:, j], j)
      auto x = DynamicSliceInMinorDims(loc, a, {j}, {1}, builder);
      auto x_collapsed_shape = llvm::to_vector<4>(batch_dims);
      x_collapsed_shape.push_back(m);
      auto x_collapsed = builder->create<ReshapeOp>(
          loc,
          RankedTensorType::get(x_collapsed_shape,
                                getElementTypeOrSelf(x.getType())),
          x);
      Value v, tau, beta;
      House(loc, x_collapsed, j, batch_dims, m, builder, &v, &tau, &beta);

      auto shape = llvm::to_vector<4>(batch_dims);
      shape.append({1, m});
      auto v_broadcast = builder->create<ReshapeOp>(
          loc, RankedTensorType::get(shape, getElementTypeOrSelf(v.getType())),
          v);
      // a[:, :] -= tau * np.dot(v[:, np.newaxis],
      //                          np.dot(v[np.newaxis, :], a[:, :]))
      auto precision = builder->getStrArrayAttr({"HIGHEST", "HIGHEST"});
      auto vva = BatchDot(loc, v_broadcast, false, a, false, num_batch_dims,
                          precision, builder);
      vva = BatchDot(loc, v_broadcast, true, vva, false, num_batch_dims,
                     precision, builder);
      auto tau_x_vva = StaticBinaryBroadcast<xla_hlo::MulOp>(
          loc, tau, vva, GetI64ElementsAttr(batch_dim_indices, builder),
          *builder);
      a = builder->create<SubOp>(loc, a, tau_x_vva);

      // It is more precise to populate column 'k' explicitly, rather than
      // computing it implicitly by applying the Householder transformation.
      // a[k,k] = beta
      // a[k+1:,k] = np.zeros([m-k-1], dtype=a.dtype)
      auto iota = builder->create<IotaOp>(
          loc, RankedTensorType::get({m, 1}, builder->getIntegerType(32)),
          builder->getI64IntegerAttr(0));
      Value predecessor_mask = builder->create<xla_chlo::BroadcastCompareOp>(
          loc, iota, j, GetI64ElementsAttr({}, builder),
          StringAttr::get("LT", builder->getContext()));
      predecessor_mask = builder->create<ConvertOp>(loc, predecessor_mask,
                                                    a_type.getElementType());
      Value mask = builder->create<xla_chlo::BroadcastCompareOp>(
          loc, iota, j, GetI64ElementsAttr({}, builder),
          StringAttr::get("EQ", builder->getContext()));
      mask = builder->create<ConvertOp>(loc, mask, a_type.getElementType());
      llvm::SmallVector<int64_t, 4> broadcast_mask_shape(a_type.getRank(), 1);
      broadcast_mask_shape[a_type.getRank() - 2] = m;
      mask = builder->create<BroadcastOp>(
          loc,
          RankedTensorType::get(broadcast_mask_shape, a_type.getElementType()),
          mask,
          GetI64ElementsAttr(llvm::SmallVector<int64_t, 4>(num_batch_dims, 1),
                             builder));
      Value predecessor_masked_x = StaticBinaryBroadcast<MulOp>(
          loc, x, predecessor_mask,
          GetI64ElementsAttr({num_dims - 2, num_dims - 1}, builder), *builder);
      Value masked_beta = StaticBinaryBroadcast<MulOp>(
          loc, beta, mask, GetI64ElementsAttr(batch_dim_indices, builder),
          *builder);
      Value new_x =
          builder->create<AddOp>(loc, predecessor_masked_x, masked_beta);
      // Update a[:,j]
      llvm::SmallVector<int64_t, 4> dim_ids(num_dims);
      std::iota(dim_ids.begin(), dim_ids.end(), 0);
      new_x = builder->create<BroadcastInDimOp>(
          loc, a_type, new_x, GetI64ElementsAttr(dim_ids, builder));
      const int64_t minor_dim = num_batch_dims;
      auto iota_mn = builder->create<IotaOp>(
          loc,
          RankedTensorType::get(a_type.getShape(), builder->getIntegerType(32)),
          builder->getI64IntegerAttr(minor_dim + 1));
      Value xa_mask = builder->create<xla_chlo::BroadcastCompareOp>(
          loc, iota_mn, j, GetI64ElementsAttr({}, builder),
          StringAttr::get("EQ", builder->getContext()));
      a = builder->create<SelectOp>(loc, a_type, xa_mask, new_x, a);

      // vs[:, j] = v
      llvm::SmallVector<int64_t, 4> vs_broadcast_dims(num_batch_dims + 1);
      std::iota(vs_broadcast_dims.begin(), vs_broadcast_dims.end(), 0);
      Value vs_zeros =
          GetScalarConstOfType(a_type.getElementType(), loc, 0, builder);
      vs_zeros = builder->create<BroadcastOp>(
          loc, vs.getType(), vs_zeros,
          GetI64ElementsAttr(vs.getType().cast<RankedTensorType>().getShape(),
                             builder));
      auto vs_update = builder->create<SelectOp>(
          loc, vs.getType(), xa_mask,
          StaticBinaryBroadcast<AddOp>(
              loc, vs_zeros, v, GetI64ElementsAttr(vs_broadcast_dims, builder),
              *builder),
          vs_zeros);
      vs = builder->create<AddOp>(loc, vs, vs_update);

      // taus[j] = tau
      llvm::SmallVector<int64_t, 4> tau_broadcast_dims(batch_dims.size());
      std::iota(tau_broadcast_dims.begin(), tau_broadcast_dims.end(), 0);

      auto iota_shape = llvm::to_vector<4>(batch_dims);
      iota_shape.push_back(n);
      auto iota_n = builder->create<IotaOp>(
          loc, RankedTensorType::get(iota_shape, builder->getIntegerType(32)),
          builder->getI64IntegerAttr(minor_dim));
      Value taus_zeros =
          GetScalarConstOfType(a_type.getElementType(), loc, 0, builder);
      taus_zeros = builder->create<BroadcastOp>(
          loc, taus.getType(), taus_zeros,
          GetI64ElementsAttr(taus.getType().cast<RankedTensorType>().getShape(),
                             builder));
      Value taus_mask = builder->create<xla_chlo::BroadcastCompareOp>(
          loc, iota_n, j, GetI64ElementsAttr({}, builder),
          StringAttr::get("EQ", builder->getContext()));
      auto taus_update = builder->create<SelectOp>(
          loc, taus.getType(), taus_mask,
          StaticBinaryBroadcast<AddOp>(
              loc, taus_zeros, tau,
              GetI64ElementsAttr(tau_broadcast_dims, builder), *builder),
          taus_zeros);
      taus = builder->create<AddOp>(loc, taus, taus_update);
      new_values->assign({a, vs, taus});
    };

    Value zero =
        GetScalarConstOfType(a_type.getElementType(), loc, 0, rewriter);
    *vs = rewriter->create<BroadcastOp>(
        loc, a_type, zero, GetI64ElementsAttr(a_type.getShape(), rewriter));
    auto taus_shape = llvm::to_vector<4>(batch_dims);
    taus_shape.push_back(n);
    *taus = rewriter->create<BroadcastOp>(
        loc, RankedTensorType::get(taus_shape, a_type.getElementType()), zero,
        GetI64ElementsAttr(taus_shape, rewriter));

    SmallVector<Value, 4> while_output;
    CreateWhile32(loc, std::min(m, n), qr_body_fn, {a, *vs, *taus},
                  &while_output, rewriter);
    *r = while_output[0];
    *vs = while_output[1];
    *taus = while_output[2];
  }

  // Computes W and Y such that I-WY is equivalent to the sequence of
  // Householder
  // transformations given by vs and taus.
  // Golub and van Loan, "Matrix Computations", algorithm 5.1.2.
  // Y = np.zeros([m, n])
  // W = np.zeros([m, n])
  // Y[:, 0] = vs[:, 0]
  // W[:, 0] = -taus[0] * vs[:, 0]
  // for j in xrange(1, n):
  //   v = vs[:, j]
  //   z = -taus[j] * v - taus[j] * np.dot(W, np.dot(Y.T, v))
  //   W[:, j] = z
  //   Y[:, j] = v
  // return W
  // There is no need to return Y since at termination of the loop it is equal
  // to vs.
  Value ComputeWYRepresentation(Location loc, Type data_type,
                                ArrayRef<int64_t> batch_dims, Value vs,
                                Value taus, int64_t m, int64_t n,
                                PatternRewriter *rewriter) const {
    int64_t n_index = batch_dims.size() + 1;
    llvm::SmallVector<int64_t, 4> batch_dim_indices(batch_dims.size());
    std::iota(batch_dim_indices.begin(), batch_dim_indices.end(), 0);

    auto body_fn = [&](Location loc, Value j, ArrayRef<Value> old_values,
                       SmallVectorImpl<Value> *new_values, OpBuilder *builder) {
      // w has shape [..., m, n]
      auto w = old_values[0];
      const auto vs = old_values[1];
      const auto taus = old_values[2];

      // Want j values in range [1, ... n).
      j = builder->create<AddOp>(
          loc, j,
          GetScalarConstOfType(getElementTypeOrSelf(j.getType()), loc, 1,
                               builder));
      // vs has shape [..., m, 1]
      auto v = DynamicSliceInMinorDims(loc, vs, {j}, {1}, builder);
      // beta has shape [..., 1]
      auto beta = DynamicSliceInMinorDims(loc, taus, {j}, {1}, builder);

      auto iota_shape = llvm::to_vector<4>(batch_dims);
      iota_shape.append({m, n});
      auto iota_mn = builder->create<IotaOp>(
          loc, RankedTensorType::get(iota_shape, builder->getIntegerType(32)),
          builder->getI64IntegerAttr(n_index));

      // y has shape [..., m, n]
      Value zero = GetScalarConstOfType(getElementTypeOrSelf(vs.getType()), loc,
                                        0, builder);
      zero = builder->create<BroadcastOp>(
          loc, vs.getType(), zero,
          GetI64ElementsAttr(vs.getType().cast<RankedTensorType>().getShape(),
                             builder));
      auto compare = builder->create<xla_chlo::BroadcastCompareOp>(
          loc, iota_mn, j, GetI64ElementsAttr({}, builder),
          StringAttr::get("GE", builder->getContext()));
      auto y = builder->create<SelectOp>(loc, vs.getType(), compare, zero, vs);

      // yv has shape [..., n, 1]
      auto precision = builder->getStrArrayAttr({"HIGHEST", "HIGHEST"});
      auto yv = BatchDot(loc, y, true, v, false, batch_dims.size(), precision,
                         builder);
      // wyv has shape [..., m, 1]
      auto wyv = BatchDot(loc, w, false, yv, false, batch_dims.size(),
                          precision, builder);

      // z = -beta * (v + wyv)
      auto neg_beta = builder->create<NegOp>(loc, beta);
      auto v_wyv = builder->create<AddOp>(loc, v, wyv);
      auto beta_broadcast_dims = llvm::to_vector<4>(batch_dim_indices);
      beta_broadcast_dims.push_back(n_index);
      auto z = StaticBinaryBroadcast<MulOp>(
          loc, neg_beta, v_wyv,
          GetI64ElementsAttr(beta_broadcast_dims, builder), *rewriter);

      w = DynamicUpdateSliceInMinorDims(loc, w, z, {j}, builder);
      new_values->assign({w, vs, taus});
    };

    Value w =
        GetScalarConstOfType(getElementTypeOrSelf(data_type), loc, 0, rewriter);
    auto w_shape = llvm::to_vector<4>(batch_dims);
    w_shape.append({m, n});
    w = rewriter->create<BroadcastOp>(loc,
                                      RankedTensorType::get(w_shape, data_type),
                                      w, GetI64ElementsAttr(w_shape, rewriter));
    auto v = SliceInMinorDims(loc, vs, {0}, {1}, rewriter);
    auto beta = SliceInMinorDims(loc, taus, {0}, {1}, rewriter);
    auto neg_beta = rewriter->create<NegOp>(loc, beta);
    auto beta_broadcast_dims = llvm::to_vector<4>(batch_dim_indices);
    beta_broadcast_dims.push_back(n_index);
    auto bv = StaticBinaryBroadcast<MulOp>(
        loc, neg_beta, v, GetI64ElementsAttr(beta_broadcast_dims, rewriter),
        *rewriter);
    w = UpdateSliceInMinorDims(loc, w, bv, {0}, rewriter);

    SmallVector<Value, 4> while_output;
    CreateWhile32(loc, n - 1, body_fn, {w, vs, taus}, &while_output, rewriter);
    return while_output[0];
  }
};

// Emits debug information which includes the number of ops of each type which
// failed to legalize.
void EmitLegalizationErrors(Operation *op,
                            const DenseSet<Operation *> &nonlegalized_ops) {
  // Track the legalization failures by mapping op name to information about
  // that failure: the number of unlegalized occurances of the op, and one
  // example operation that failed.
  std::map<StringRef, std::pair<int, Operation *>> op_name_to_error_info;
  DenseSet<Operation *> error_ops;
  for (Operation *nonlegalized_op : nonlegalized_ops) {
    // Increment count of this legalization failure.
    StringRef op_name = nonlegalized_op->getName().getStringRef();
    // If this emplace is successful, it's the first time we've encountered
    // this op type. Initialize count to 0 so that after increment, it is 1.
    auto insertion_result = op_name_to_error_info.emplace(
        op_name, std::make_pair(0, nonlegalized_op));
    ++insertion_result.first->second.first;
  }
  std::vector<std::string> error_messages;
  error_messages.reserve(op_name_to_error_info.size());
  for (const auto &op_info : op_name_to_error_info) {
    error_messages.push_back(
        llvm::formatv("{0} (count: {1})", op_info.first, op_info.second.first));
  }
  Location loc = op->getLoc();
  emitError(loc) << "The following operations cannot be legalized: "
                 << llvm::join(error_messages, "; ")
                 << ". These legalization failure(s) may be due to missing TF "
                    "to HLO lowerings and/or unsupported attributes, etc.";
  // Emit more information about the missing ops. This error message
  // contains useful details beyond the op name (input and output shapes,
  // attributes, etc.).
  if (!VLOG_IS_ON(1) && nonlegalized_ops.size() != 1) {
    emitError(loc)
        << "Emitting more detail about one op that failed to legalize...";
  } else if (VLOG_IS_ON(1)) {
    emitError(loc) << "Emitting more detail about one of each type of op "
                      "that failed to legalize...";
  }
  for (const auto &op_info : op_name_to_error_info) {
    op_info.second.second->emitOpError() << "is not legalizable";
    if (!VLOG_IS_ON(1)) break;
  }
}

// Performs the lowering to XLA dialect.
void LegalizeTF::runOnFunction() {
  if (failed(
          legalizeTF(getFunction(), allow_partial_conversion_, legalize_chlo_)))
    signalPassFailure();
}

static PassRegistration<LegalizeTF> pass(
    "xla-legalize-tf", "Legalize from TensorFlow to the XLA dialect");

}  // end namespace

#include "tensorflow/compiler/mlir/xla/transforms/generated_legalize_tf.inc"

LogicalResult legalizeTF(Operation *op, bool allow_partial_conversion,
                         bool legalize_chlo) {
  MLIRContext *context = op->getContext();

  // Add lowering patterns to the list.
  OwningRewritePatternList patterns;
  populateWithGenerated(context, &patterns);

  // Add patterns that lower some of the high level TensorFlow ops to lower
  // level TensorFlow ops. So, we don't have to target all the TensorFlow ops
  // here for lowering to HLO.
  TF::PopulateLoweringTFPatterns(context, &patterns);
  patterns.insert<
      ConvertAllOp, ConvertAnyOp, ConvertArgMaxOp, ConvertBatchMatMulV2Op,
      ConvertBiasAddOp, ConvertBroadcastToOp, ConvertBF16FloorDivOp,
      ConvertConv2DOp, ConvertConv3DOp, ConvertDepthConv2DOp,
      ConvertConv2DBackpropFilterOp, ConvertConv3DBackpropFilterOp,
      ConvertConv2DBackpropInputOp, ConvertConv3DBackpropInputOp,
      ConvertCumsumOp, ConvertDiagPartOp, ConvertEinsumOp,
      ConvertFusedBatchNormGradOp, ConvertFusedBatchNormGradV2Op,
      ConvertFusedBatchNormGradV3Op, ConvertFusedBatchNormV3Op,
      ConvertInfeedDequeueTupleOp, ConvertInplaceUpdateOp, ConvertLinSpaceOp,
      ConvertMaxOp, ConvertMinOp, ConvertAvgPoolOp, ConvertAvgPool2DGradOp,
      ConvertAvgPool3DGradOp, ConvertMaxPool2DOp, ConvertMaxPool3DOp,
      ConvertMaxPool2DGradOp, ConvertMaxPool3DGradOp, ConvertMeanOp,
      ConvertOneHotOp, ConvertOutfeedEnqueueTupleOp, ConvertProdOp, ConvertQrOp,
      ConvertDynamicRangeOp, ConvertRangeOp, ConvertSelectV2Op,
      ConvertSigmoidOp, ConvertShapeOp, ConvertSizeOp,
      ConvertSoftmaxOp<TF::LogSoftmaxOp, true>,
      ConvertSoftmaxOp<TF::SoftmaxOp, false>, ConvertSplitOp, ConvertSplitVOp,
      ConvertStridedSliceOp, ConvertStridedSliceGradOp, ConvertSumOp,
      ConvertTensorScatterUpdateOp, ConvertTileOp, ConvertTopKV2Op,
      ConvertUnpackOp, ConvertUnsortedSegmentMaxOp, ConvertUnsortedSegmentMinOp,
      ConvertUnsortedSegmentProdOp, ConvertUnsortedSegmentSumOp,
      ConvertRandomShuffleOp, ConvertXlaShardingOp,
      ConvertXlaDynamicUpdateSliceOp>(op->getContext());

  // Populate with CHLO->HLO lowerings to account for TF ops legalized to
  // CHLO first.
  if (legalize_chlo) {
    xla_chlo::PopulateLegalizeChloToHloPatterns(context, &patterns);
  }

  ConversionTarget target(*context);
  if (legalize_chlo) {
    target.addIllegalDialect<xla_chlo::XlaHloClientDialect>();
  } else {
    target.addLegalDialect<xla_chlo::XlaHloClientDialect>();
  }
  target.addLegalDialect<XlaHloDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<shape::ShapeDialect>();
  target.addLegalOp<CallOp>();
  target.addLegalOp<TensorCastOp>();

  if (!allow_partial_conversion) {
    // Fully qualify ReturnOp here as xla_hlo dialect also defines a ReturnOp.
    target.addLegalOp<ModuleOp, FuncOp, ModuleTerminatorOp, ::mlir::ReturnOp>();
    DenseSet<Operation *> nonlegalized_ops;
    LogicalResult result =
        applyPartialConversion(op, target, patterns, &nonlegalized_ops);
    // In order to enforce that the conversion result is fully converted,
    // fail if there are any nonlegalized ops in the set.
    if (failed(result) || !nonlegalized_ops.empty()) {
      EmitLegalizationErrors(op, nonlegalized_ops);
      return failure();
    }
    return result;
  }

  return applyPartialConversion(op, target, patterns);
}

std::unique_ptr<OperationPass<FuncOp>> createLegalizeTFPass(
    bool allow_partial_conversion, bool legalize_chlo) {
  return std::make_unique<LegalizeTF>(allow_partial_conversion, legalize_chlo);
}

}  // end namespace xla_hlo
}  // end namespace mlir
