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

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/utils.h"
#include "tensorflow/compiler/xla/client/lib/conv_grad_size_util.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/utils/convert_op_folder.h"
#include "tensorflow/compiler/xla/mlir_hlo/utils/hlo_utils.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/attribute_importer.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/tsl/platform/bfloat16.h"
#include "tensorflow/tsl/platform/status.h"

namespace mlir {
namespace mhlo {
namespace {

constexpr char kShardingAttr[] = "mhlo.sharding";

/// Returns the feature dimension for the given format and input type.
static size_t GetFeatureDimension(tensorflow::TensorFormat format,
                                  RankedTensorType input_ty) {
  return GetTensorFeatureDimIndex(input_ty.getRank(), format);
}

// Gets all integer values from the given attribute and push them to `values`.
void GetI64ArrayAttrValues(Attribute attr, SmallVectorImpl<int64_t> *values) {
  auto array_attr = attr.cast<ArrayAttr>();
  values->reserve(array_attr.getValue().size());
  for (Attribute val : array_attr.getValue())
    values->push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
}

// Returns 1D 32-bit dense elements attribute with the given values.
static DenseIntElementsAttr GetI32ElementsAttr(ArrayRef<int32_t> values,
                                               Builder *builder) {
  RankedTensorType ty = tensorflow::GetTypeFromTFTensorShape(
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

  TensorType ty =
      tensorflow::GetTypeFromTFTensorShape({size}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, vals);
}

// Returns a 1-d i64 elements attribute populated with `val` repeated `size`
// times.
static DenseIntElementsAttr GetI64ElementsAttrForValue(int size, int64_t val,
                                                       Builder *builder) {
  TensorType ty =
      tensorflow::GetTypeFromTFTensorShape({size}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, val);
}

// Returns the corresponding type that should be used for performing sum
// accumulation over the given input type.
Type GetSumAccumulationType(Type input_type) {
  MLIRContext *ctx = input_type.getContext();
  if (input_type.isBF16() || input_type.isF16()) return FloatType::getF32(ctx);
  if (input_type.isSignlessInteger(8) || input_type.isSignlessInteger(16))
    return IntegerType::get(ctx, 32);
  return input_type;
}

// Returns axis in HLO format from TF elements attr with exactly one element or
// is an IntegerAttr, containing axis in the TensorFlow format. TensorFlow
// format supports negative indexing unlike HLO.
static IntegerAttr GetHLOAxisFromTFAxis(Attribute attr, int64_t rank,
                                        Builder *b) {
  IntegerAttr intAttr = attr.dyn_cast_or_null<IntegerAttr>();
  if (auto elementAttr = attr.dyn_cast_or_null<ElementsAttr>()) {
    SmallVector<uint64_t, 1> index(elementAttr.getType().getRank(), 0);
    intAttr = elementAttr.getValues<IntegerAttr>()[index];
  }

  assert(intAttr && "Invalid attribute passed to GetHLOAxisFromTFAxis");

  int64_t axis = intAttr.getInt();
  if (axis < 0) {
    axis += rank;
  }
  return b->getI64IntegerAttr(axis);
}

// If `value` is an IntegerAttr, returns the integer value for the HLO axis
// corresponding to the tensorflow axis. In particular, the tensorflow axis can
// be negative, in which case, the corresponding HLO axis is
// (axis + rank-of-the-tensor).
static std::optional<int64_t> GetIntegerHLOAxisFromTFAxis(Value value,
                                                          int64_t rank) {
  DenseIntElementsAttr attrs;
  if (!matchPattern(value, m_Constant(&attrs)) ||
      attrs.getType().getRank() != 0) {
    return std::nullopt;
  }
  int64_t axis = attrs.getValues<IntegerAttr>()[0].getInt();
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
      num_outputs,
      tensorflow::GetTypeFromTFTensorShape({}, indices_type.getElementType()));
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
  return tensorflow::TensorShape(
      llvm::SmallVector<int64_t, num_dims>(sizes.begin(), sizes.end()));
}

template <typename T, int num_dims>
tensorflow::TensorShape ToTensorShape(
    llvm::iterator_range<DenseElementsAttr::ElementIterator<T>> sizes) {
  return tensorflow::TensorShape(
      llvm::SmallVector<int64_t, num_dims>(sizes.begin(), sizes.end()));
}

// Returns a limit scalar const op for the given type.
// Requires FloatType or IntegerType
static ConstantOp GetScalarLimitConstOfType(Type ty, Location loc,
                                            hlo::ScalarLimit limit,
                                            OpBuilder *builder) {
  return builder->create<ConstantOp>(loc, hlo::getScalarLimitOfType(ty, limit));
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
    return tensorflow::GetTypeFromTFTensorShape(out_shape, element_type);
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
  for (auto &index_pair : llvm::enumerate(broadcast_dimensions)) {
    auto old_value = out_shape[index_pair.value()];
    auto new_value = shape_small[index_pair.index()];
    out_shape[index_pair.value()] = std::max(old_value, new_value);
  }
  return tensorflow::GetTypeFromTFTensorShape(out_shape, element_type);
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
  return tensorflow::GetTypeFromTFTensorShape({dim}, b.getIndexType());
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

// Builds a set of operations for applying reduction on the input value. A
// tf.sum op is created and will be legalized to tfl ops automatically.
static Value ApplyReduction(Location loc, Value input,
                            DenseIntElementsAttr reduce_dims,
                            OpBuilder *builder) {
  auto reduce_dims_op = builder->create<ConstantOp>(loc, reduce_dims);
  return builder->create<TF::SumOp>(loc, input, reduce_dims_op,
                                    builder->getBoolAttr(false));
}

// Creates a mhlo.rng_uniform op with `builder` to generate `num_elements`
// 32-bit integer numbers in the range of [`lower_limit`, `upper_limit`).
static mhlo::RngOp CreateRngUniform32(Location loc, int num_elements,
                                      int lower_limit, int upper_limit,
                                      OpBuilder *builder) {
  auto shape_tensor = builder->create<mhlo::ConstantOp>(
      loc, GetI64ElementsAttr({num_elements}, builder));

  auto lower = builder->create<mhlo::ConstantOp>(
      loc, builder->getI32IntegerAttr(lower_limit));
  auto upper = builder->create<mhlo::ConstantOp>(
      loc, builder->getI32IntegerAttr(upper_limit));

  return builder->create<mhlo::RngOp>(loc, lower, upper, shape_tensor,
                                      ::mlir::mhlo::RngDistribution::UNIFORM);
}

using WhileBodyFnType = llvm::function_ref<void(
    Location loc, Value iteration, ArrayRef<Value> old_values,
    SmallVectorImpl<Value> *new_values, OpBuilder *builder)>;

// Creates a mhlo.while op with `builder` to loop `num_interations` times,
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
  SmallVector<Type, 2> init_types_with_loop_iv;
  init_values_with_loop_iv.reserve(value_count);
  init_types_with_loop_iv.reserve(value_count);

  // The initial value for the loop induction variable is 0.
  init_values_with_loop_iv.push_back(
      builder->create<mhlo::ConstantOp>(loc, builder->getI32IntegerAttr(0)));
  init_values_with_loop_iv.append(init_values.begin(), init_values.end());

  // Accumulate types of all the init values.
  for (const auto &init_value_with_loop_iv : init_values_with_loop_iv)
    init_types_with_loop_iv.push_back(init_value_with_loop_iv.getType());

  // Create the while op.
  auto while_op = builder->create<mhlo::WhileOp>(loc, init_types_with_loop_iv,
                                                 init_values_with_loop_iv);
  auto ivs_count = init_types_with_loop_iv.size();

  {
    OpBuilder::InsertionGuard guard(*builder);

    // Build up the only block in the condition region.
    Region &condition = while_op.getCond();
    Block *block = builder->createBlock(&condition);
    block->addArguments(init_types_with_loop_iv,
                        SmallVector<Location>(ivs_count, loc));

    // Get the loop induction variable and compare it against the upper limit.
    auto loop_iv = block->getArgument(0);
    auto upper_limit = builder->create<mhlo::ConstantOp>(
        loc, builder->getI32IntegerAttr(num_iterations));
    Value compare = builder->create<mhlo::CompareOp>(loc, loop_iv, upper_limit,
                                                     ComparisonDirection::LT);

    builder->create<mhlo::ReturnOp>(loc, compare);
  }

  {
    OpBuilder::InsertionGuard guard(*builder);

    // Build up the only block in the body region.
    Region &body = while_op.getBody();
    Block *block = builder->createBlock(&body);
    block->addArguments(init_types_with_loop_iv,
                        SmallVector<Location>(ivs_count, loc));

    SmallVector<Value, 4> new_values;  // Generated by this iteration
    new_values.reserve(value_count);

    // Feed all values excluding the loop induction variable to body_fn.
    body_fn(loc, block->getArgument(0),
            ArrayRef<Value>(block->getArguments().begin() + 1,
                            block->getArguments().end()),
            &new_values, builder);

    // Increment the loop induction variable by one.
    auto one =
        builder->create<mhlo::ConstantOp>(loc, builder->getI32IntegerAttr(1));
    auto scalar_broadcast_dims = GetI64ElementsAttr({}, builder);
    auto plus_one = builder->create<chlo::BroadcastAddOp>(
        loc, block->getArgument(0), one, scalar_broadcast_dims);
    // Prepend with the updated loop induction variable.
    new_values.insert(new_values.begin(), plus_one);

    builder->create<mhlo::ReturnOp>(loc, new_values);
  }

  // TODO(jpienaar): Support multi-operand while op.
  final_values->reserve(init_values.size());
  for (int i = 0, e = init_values.size(); i < e; ++i)
    final_values->push_back(while_op.getResult(i + 1));
}

//===----------------------------------------------------------------------===//
// BatchNorm op utilities.
//===----------------------------------------------------------------------===//

static IntegerAttr getFeatureDimensionAttr(Builder &b,
                                           tensorflow::TensorFormat format,
                                           Value input) {
  return b.getI64IntegerAttr(
      GetFeatureDimension(format, input.getType().cast<RankedTensorType>()));
}

//===----------------------------------------------------------------------===//
// FFT op utilities.
//===----------------------------------------------------------------------===//

// Returns the 1D i64 elements attribute populated with the inner-most dim of
// the value.
static DenseIntElementsAttr GetInnerDimFromValue(ShapedType type,
                                                 Builder *builder) {
  if (type.getRank() == 0) {
    return builder->getI64TensorAttr({});
  }
  return builder->getI64TensorAttr(type.getShape().back());
}

// Returns True if the inner-most dim is static.
bool CheckInnerDimStatic(ShapedType type, Builder *builder) {
  if (!type.hasRank()) {
    return false;
  }
  return !type.isDynamicDim(type.getShape().size() - 1);
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

  for (auto &it : llvm::enumerate(int_attr.getValues<APInt>())) {
    if (static_cast<int>(it.index() % shape[1]) == column) {
      values.push_back(it.value().getSExtValue());
    }
  }

  auto element_type = IntegerType::get(input.getContext(), 64);
  return DenseIntElementsAttr::get(
      tensorflow::GetTypeFromTFTensorShape({shape[0]}, element_type), values);
}

// Returns interior padding to use in HLO Pad op based on the TensorFlow padding
// in TensorFlow PadV2 op.
static DenseIntElementsAttr GetInteriorPadding(ElementsAttr tf_padding) {
  auto length = tf_padding.getType().getShape()[0];
  auto element_type = IntegerType::get(tf_padding.getContext(), 64);
  return DenseIntElementsAttr::get<int64_t>(
      tensorflow::GetTypeFromTFTensorShape({length}, element_type), 0);
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
    return tensorflow::GetTypeFromTFTensorShape(ranked_type.getShape(),
                                                element_type);
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
  auto scalar_ty = tensorflow::GetTypeFromTFTensorShape({}, element_ty);
  if (element_ty.isF16()) {
    uint16_t raw_epsilon = Eigen::numext::bit_cast<uint16_t>(
        Eigen::NumTraits<Eigen::half>::epsilon());
    auto value = APFloat(APFloat::IEEEhalf(), APInt(16, raw_epsilon));
    return DenseElementsAttr::get(scalar_ty, value);
  } else if (element_ty.isBF16()) {
    uint16_t raw_epsilon = Eigen::numext::bit_cast<uint16_t>(
        Eigen::NumTraits<Eigen::bfloat16>::epsilon());
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
                                        ComparisonDirection direction,
                                        Region *body, OpBuilder *builder) {
  OpBuilder::InsertionGuard insertion_point_gurad(*builder);

  Type input_type =
      tensorflow::GetTypeFromTFTensorShape(/*shape=*/{}, input_element_type);
  Type index_type =
      tensorflow::GetTypeFromTFTensorShape(/*shape=*/{}, index_element_type);
  Block *block = builder->createBlock(body);
  Location loc = body->getLoc();
  block->addArguments({input_type, index_type, input_type, index_type},
                      SmallVector<Location, 4>(4, loc));

  Value lhs_val = block->getArgument(0);
  Value lhs_index = block->getArgument(1);
  Value rhs_val = block->getArgument(2);
  Value rhs_index = block->getArgument(3);

  ImplicitLocOpBuilder b(loc, *builder);
  Value compare_dt = b.create<CompareOp>(lhs_val, rhs_val, direction);
  Value selected_input =
      b.create<SelectOp>(input_type, compare_dt, lhs_val, rhs_val);

  Value compare_eq =
      b.create<CompareOp>(lhs_val, rhs_val, ComparisonDirection::EQ);
  Value min_index = b.create<MinOp>(lhs_index, rhs_index);
  Value min_val_index =
      b.create<SelectOp>(index_type, compare_dt, lhs_index, rhs_index);
  Value selected_index =
      b.create<SelectOp>(index_type, compare_eq, min_index, min_val_index);

  Value return_values[] = {selected_input, selected_index};
  b.create<ReturnOp>(return_values);
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
      dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(module, func));
  FunctionType function_ty = function.getFunctionType();

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
    int64_t slice_size = slice_sizes.getValues<IntegerAttr>()[i].getInt();
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
    return hlo::convertElementsAttr(slice_sizes, builder->getIntegerType(64))
        .cast<DenseIntElementsAttr>();
  }

  auto input_ty = input.getType().dyn_cast<RankedTensorType>();
  int64_t input_rank = input_ty.getRank();
  ArrayRef<int64_t> input_shape = input_ty.getShape();
  SmallVector<int64_t, 4> normalized_sizes;

  for (int64_t i = 0; i < input_rank; ++i) {
    int64_t input_size = input_shape[i];
    int64_t start_index =
        constant_start_indices.getValues<IntegerAttr>()[i].getInt();
    int64_t slice_size = slice_sizes.getValues<IntegerAttr>()[i].getInt();
    normalized_sizes.push_back(slice_size == -1 ? input_size - start_index
                                                : slice_size);
  }

  return GetI64ElementsAttr(normalized_sizes, builder);
}

//===----------------------------------------------------------------------===//
// XlaGather op utilities.
//===----------------------------------------------------------------------===//

bool HasValidGatherDims(StringAttr attr) {
  ::xla::GatherDimensionNumbers dims;
  return dims.ParseFromString(attr.getValue().str());
}

GatherDimensionNumbersAttr GetGatherDimNumsAttr(StringAttr attr,
                                                Builder *builder) {
  ::xla::GatherDimensionNumbers dims;
  if (!dims.ParseFromString(attr.getValue().str())) return {};
  return ::xla::ConvertGatherDimensionNumbers(dims, builder);
}

//===----------------------------------------------------------------------===//
// XlaDot op utilities.
//===----------------------------------------------------------------------===//

bool HasValidDotDims(StringAttr attr) {
  ::xla::DotDimensionNumbers dims;
  return dims.ParseFromString(attr.getValue().str());
}

DotDimensionNumbersAttr GetDotDimNumsAttr(StringAttr attr, Builder *builder) {
  ::xla::DotDimensionNumbers dims;
  if (!dims.ParseFromString(attr.getValue().str())) return {};
  return ::xla::ConvertDotDimensionNumbers(dims, builder);
}

bool HasValidPrecisionConfig(StringAttr attr) {
  ::xla::PrecisionConfig precision;
  return precision.ParseFromString(attr.getValue().str());
}

mlir::ArrayAttr GetPrecisionConfigAttr(StringAttr attr, Builder *builder) {
  ::xla::PrecisionConfig precision;
  if (!precision.ParseFromString(attr.getValue().str())) return {};
  return ::xla::ConvertPrecisionConfig(&precision, builder);
}

//===----------------------------------------------------------------------===//
// XlaVariadicReduceV2 op utilities.
//===----------------------------------------------------------------------===//

static void BuildBodyWithCall(PatternRewriter &rewriter, const Location &loc,
                              mlir::SymbolRefAttr func,
                              mlir::FunctionType func_ty, Region *body) {
  OpBuilder::InsertionGuard guard(rewriter);

  Block *block = rewriter.createBlock(body);
  auto inputs = func_ty.getInputs();
  block->addArguments(inputs, SmallVector<Location>(inputs.size(), loc));
  mlir::func::CallOp call_op = rewriter.create<mlir::func::CallOp>(
      loc, func, func_ty.getResults(), block->getArguments());
  rewriter.create<mhlo::ReturnOp>(loc, call_op.getResults());
}

//===----------------------------------------------------------------------===//
// Op converters.
//===----------------------------------------------------------------------===//

NamedAttribute GetConvDimensionNumbersAttr(ArrayRef<int64_t> spatial_dims,
                                           tensorflow::TensorFormat format,
                                           Builder *builder) {
  int64_t num_spatial_dims = spatial_dims.size();
  int64_t num_dims = num_spatial_dims + 2;

  int64_t batch_dim = GetTensorBatchDimIndex(num_dims, format);
  int64_t feature_dim = GetTensorFeatureDimIndex(num_dims, format);

  // Filters data_format is always HWIO so input channels dimension is after
  // all spatial dimensions.
  int64_t kernel_input_feature_dim = num_spatial_dims;
  int64_t kernel_output_feature_dim = num_spatial_dims + 1;
  SmallVector<int64_t, 4> kernel_spatial_dimensions;
  kernel_spatial_dimensions.resize(num_spatial_dims);
  std::iota(kernel_spatial_dimensions.begin(), kernel_spatial_dimensions.end(),
            0);

  return builder->getNamedAttr(
      "dimension_numbers",
      ConvDimensionNumbersAttr::get(
          builder->getContext(), batch_dim, feature_dim, spatial_dims,
          kernel_input_feature_dim, kernel_output_feature_dim,
          kernel_spatial_dimensions, batch_dim, feature_dim, spatial_dims));
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
    Location loc = op.getLoc();
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.getDataFormat().str(), &data_format))
      return op.emitOpError("invalid data format");

    auto value_type = op.getValue().getType().dyn_cast<RankedTensorType>();
    if (!value_type) return failure();
    auto feature_dim = GetFeatureDimension(data_format, value_type);
    auto bias_broadcast = Broadcast1DToFeatureDim(
        loc, op.getValue(), op.getBias(), feature_dim, rewriter);
    Value add = rewriter.create<AddOp>(loc, op.getValue(), bias_broadcast);
    if (add.getType() != op.getType()) {
      add = rewriter.create<tensor::CastOp>(loc, op.getType(), add);
    }
    rewriter.replaceOp(op, {add});
    return success();
  }
};

// Conterts tf.Conv2D to mhlo.dynamic_conv.
// TODO(disc): To recover static special case's performance with adding folding,
// canonicalization func and removing ConvertConvOp.
template <typename OpT, int num_spatial_dims, bool depthwise_conv = false>
class ConvertConvDynamic : public OpRewritePattern<OpT> {
 public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  bool GetPaddingValues(OpT &op, PatternRewriter &rewriter, Value input_size,
                        Value filter_size, int64_t dilation_rate,
                        int64_t stride, tensorflow::Padding padding_type,
                        Type shape_scalar_type, Value *padding_low,
                        Value *padding_high) const {
    // Stride must be > 0
    if (stride <= 0) return false;
    // Dilation rate must be >= 1
    if (dilation_rate < 1) return false;

    Location loc = op.getLoc();
    switch (padding_type) {
      case tensorflow::Padding::VALID: {
        auto zero =
            rewriter.create<arith::ConstantIntOp>(loc, 0, shape_scalar_type);
        *padding_low = *padding_high = zero;
        break;
      }
      case tensorflow::Padding::EXPLICIT:
        break;
      case tensorflow::Padding::SAME: {
        auto zero =
            rewriter.create<arith::ConstantIntOp>(loc, 0, shape_scalar_type);
        auto one =
            rewriter.create<arith::ConstantIntOp>(loc, 1, shape_scalar_type);
        auto two =
            rewriter.create<arith::ConstantIntOp>(loc, 2, shape_scalar_type);
        // See also the parallel implementation in
        // GetWindowedOutputSizeFromDimsV2. effective_filter_size = (filter_size
        // - 1) * dilation_rate + 1
        Value stride_value = rewriter.create<arith::ConstantIntOp>(
            loc, stride, shape_scalar_type);
        Value dilation_rate_value = rewriter.create<arith::ConstantIntOp>(
            loc, dilation_rate, shape_scalar_type);
        Value effective_filter_size_op = rewriter.create<arith::AddIOp>(
            loc, one,
            rewriter.create<arith::MulIOp>(
                loc, dilation_rate_value,
                rewriter.create<arith::SubIOp>(loc, filter_size, one)));
        // output_size = (input_size + stride - 1) / stride;
        Value output_size = rewriter.create<arith::DivUIOp>(
            loc,
            rewriter.create<arith::AddIOp>(
                loc, input_size,
                rewriter.create<arith::SubIOp>(loc, stride_value, one)),
            stride_value);
        // std::max(int64{0}, (output_size - 1) * stride +
        //     effective_filter_size - input_size);
        Value padding_needed = rewriter.create<arith::SubIOp>(
            loc,
            rewriter.create<arith::AddIOp>(
                loc, effective_filter_size_op,
                rewriter.create<arith::MulIOp>(
                    loc, stride_value,
                    rewriter.create<arith::SubIOp>(loc, output_size, one))),
            input_size);
        Value cond = rewriter.create<mlir::arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, padding_needed, zero);
        padding_needed = rewriter.create<mlir::arith::SelectOp>(
            loc, padding_needed.getType(), cond, padding_needed, zero);
        *padding_low =
            rewriter.create<arith::DivUIOp>(loc, padding_needed, two);
        *padding_high =
            rewriter.create<arith::SubIOp>(loc, padding_needed, *padding_low);
        break;
      }
    }
    return true;
  }

  LogicalResult matchAndRewriteDynamicConv(OpT op,
                                           PatternRewriter &rewriter) const {
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.getDataFormat().str(), &data_format))
      return op.emitOpError("invalid data format");

    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.getPadding().str(), &padding).ok())
      return failure();

    auto input_ty =
        op.getInput().getType().template dyn_cast<RankedTensorType>();
    auto filter_ty =
        op.getFilter().getType().template dyn_cast<RankedTensorType>();
    auto result_ty = op.getType().template dyn_cast<RankedTensorType>();
    if (!input_ty || !filter_ty || !result_ty) return failure();
    // TODO(disc): Remove this constraint once fold and canonicalization
    // implemented.
    if (input_ty.hasStaticShape() && filter_ty.hasStaticShape())
      return failure();

    ArrayRef<Attribute> dilations = op.getDilations().getValue();
    ArrayRef<Attribute> strides = op.getStrides().getValue();
    ArrayRef<Attribute> explicit_paddings;
    if (padding == tensorflow::Padding::EXPLICIT) {
      // EXPLICIT padding mode and the associated attribute is attached to
      // Conv2D.
      explicit_paddings =
          op->template getAttrOfType<ArrayAttr>("explicit_paddings").getValue();
    }

    SmallVector<int64_t, num_spatial_dims> spatial_dim_indices;
    SmallVector<int64_t, num_spatial_dims> rhs_dilations;
    SmallVector<int64_t, num_spatial_dims> window_strides;
    SmallVector<Value, num_spatial_dims * 2> paddings;

    auto get_int = [](Attribute attr) {
      return attr.template cast<IntegerAttr>().getInt();
    };

    constexpr int num_dims = num_spatial_dims + 2;

    Location loc = op.getLoc();
    auto shape_scalar_type = rewriter.getIntegerType(32);

    auto get_const = [&](int64_t val) {
      return rewriter.create<mlir::arith::ConstantIntOp>(loc, val,
                                                         shape_scalar_type);
    };
    auto get_dim_value = [&](Value val, int64_t dim) {
      Value dim_value = rewriter.create<tensor::DimOp>(loc, val, dim);
      return rewriter.create<arith::IndexCastOp>(loc, shape_scalar_type,
                                                 dim_value);
    };

    for (auto i : llvm::seq<int>(0, num_spatial_dims)) {
      const int64_t dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      spatial_dim_indices.push_back(dim);

      const int64_t dilation = get_int(dilations[dim]);
      rhs_dilations.push_back(dilation);
      const int64_t stride = get_int(strides[dim]);
      window_strides.push_back(stride);

      Value pad_low, pad_high;
      if (padding == tensorflow::Padding::EXPLICIT) {
        pad_low = get_const(get_int(explicit_paddings[2 * dim]));
        pad_high = get_const(get_int(explicit_paddings[2 * dim + 1]));
      } else {
        auto input_size = get_dim_value(op.getInput(), dim);
        auto filter_size = get_dim_value(op.getFilter(), i);
        if (!GetPaddingValues(op, rewriter, input_size, filter_size, dilation,
                              stride, padding, shape_scalar_type, &pad_low,
                              &pad_high)) {
          return failure();
        }
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

    Value paddings_op = rewriter.create<tensor::FromElementsOp>(
        op.getLoc(),
        tensorflow::GetTypeFromTFTensorShape(2 * num_spatial_dims,
                                             rewriter.getI32Type()),
        paddings);

    SmallVector<Value, 3> operands(op.getOperands());
    operands.push_back(paddings_op);
    // Reshape the filter to {spatial_dims...., 1,in_channels *
    // channel_multiplier}
    if (depthwise_conv) {
      ArrayRef<int64_t> filter_shape = filter_ty.getShape();
      llvm::SmallVector<int64_t, num_dims> new_shape(
          filter_shape.begin(), filter_shape.begin() + num_spatial_dims);
      new_shape.push_back(1);
      new_shape.push_back(filter_shape[num_spatial_dims] *
                          filter_shape[num_spatial_dims + 1]);
      operands[1] = rewriter.create<mhlo::ReshapeOp>(
          op.getLoc(),
          tensorflow::GetTypeFromTFTensorShape(new_shape,
                                               filter_ty.getElementType()),
          operands[1]);
    }
    NamedAttribute attrs[] = {rhs_dilations_attr, window_strides_attr,
                              dimension_numbers_attr, feature_group_count_attr,
                              batch_group_count_attr};
    rewriter.replaceOpWithNewOp<mhlo::DynamicConvOp>(op, op.getType(), operands,
                                                     llvm::ArrayRef(attrs));
    return success();
  }

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteDynamicConv(op, rewriter);
  }
};

using ConvertConv2DDynamic =
    ConvertConvDynamic<TF::Conv2DOp, /*num_spatial_dims=*/2>;

// Converts the TensorFlow conv op in template to the generic HLO conv op by
// converting TensorFlow op attributes to HLO op attributes.
//
// Sample result for Conv2D:
//
//   %conv = "mhlo.convolution"(%input, %filter) {
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
    if (!FormatFromString(op.getDataFormat().str(), &data_format))
      return op.emitOpError("invalid data format");

    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.getPadding().str(), &padding).ok())
      return failure();

    auto input_ty =
        op.getInput().getType().template dyn_cast<RankedTensorType>();
    auto filter_ty =
        op.getFilter().getType().template dyn_cast<RankedTensorType>();

    // With the exception of input's batch dimension, input and filter need to
    // have static shape for calculation of HLO paddings and feature group count
    // attributes. Filter is validated here, input is mostly validated at use.
    if (!input_ty || !filter_ty || !filter_ty.hasStaticShape())
      return failure();

    ArrayRef<Attribute> dilations = op.getDilations().getValue();
    ArrayRef<Attribute> strides = op.getStrides().getValue();
    ArrayRef<Attribute> explicit_paddings;
    if (padding == tensorflow::Padding::EXPLICIT) {
      // EXPLICIT padding mode and the associated attribute is limited to
      // Conv2D. So, fetch attribute by identifier instead of the
      // op.explicit_paddings() attribute getter.
      explicit_paddings =
          op->template getAttrOfType<ArrayAttr>("explicit_paddings").getValue();
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
        int64_t output_size;
        int64_t pad_low_int64;
        int64_t pad_high_int64;
        int64_t input_size = input_ty.getDimSize(dim);
        if (input_size == ShapedType::kDynamic) return failure();
        tsl::Status status = tensorflow::GetWindowedOutputSizeVerboseV2(
            input_size, filter_ty.getDimSize(i), dilation, stride, padding,
            &output_size, &pad_low_int64, &pad_high_int64);
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
    if (input_channels == ShapedType::kDynamic) return failure();
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

    RankedTensorType paddings_ty = tensorflow::GetTypeFromTFTensorShape(
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
      operands[1] = rewriter.create<mhlo::ReshapeOp>(
          op.getLoc(),
          tensorflow::GetTypeFromTFTensorShape(new_shape,
                                               filter_ty.getElementType()),
          operands[1]);
    }
    NamedAttribute attrs[] = {rhs_dilations_attr,     window_strides_attr,
                              dimension_numbers_attr, feature_group_count_attr,
                              batch_group_count_attr, paddings_attr};
    rewriter.replaceOpWithNewOp<ConvolutionOp>(op, op.getType(), operands,
                                               llvm::ArrayRef(attrs));
    return success();
  }
};

using ConvertConv2DOp = ConvertConvOp<TF::Conv2DOp, /*num_spatial_dims=*/2>;
using ConvertConv3DOp = ConvertConvOp<TF::Conv3DOp, /*num_spatial_dims=*/3>;
using ConvertDepthConv2DOp =
    ConvertConvOp<TF::DepthwiseConv2dNativeOp, /*num_spatial_dims=*/2,
                  /*depthwise_conv=*/true>;

// Converts tf.PadV2Op to mhlo.DynamicPadOp. Padding values must be const.
class ConvertPadOpDynamic : public OpRewritePattern<TF::PadV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;
  // TODO(disc): To recover static special case's performance with folding and
  // canonicalization.
  LogicalResult matchAndRewrite(TF::PadV2Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto input = op.getInput();
    auto paddings = op.getPaddings();
    auto constant_values = op.getConstantValues();
    auto input_type = input.getType().dyn_cast<RankedTensorType>();
    auto paddings_type = paddings.getType().dyn_cast<RankedTensorType>();
    if (!input_type || !paddings_type || !paddings_type.hasStaticShape())
      return failure();

    // TODO(disc): Remove this constraint once fold and canonicalization is
    // implemented.
    if (input_type.hasStaticShape()) return failure();

    int input_rank = input_type.getRank();
    // interior padding
    std::vector<int64_t> interior_values(input_rank, 0);
    auto interior_attr = GetI64ElementsAttr(interior_values, &rewriter);

    Value interior_padding_tensor =
        rewriter.create<mhlo::ConstantOp>(loc, interior_attr);
    Type paddings_elem_ty = paddings_type.getElementType();
    if (!paddings_elem_ty.isInteger(64)) {
      interior_padding_tensor = rewriter.create<mhlo::ConvertOp>(
          loc, interior_padding_tensor, paddings_elem_ty);
    }
    llvm::SmallVector<int64_t, 2> transposed_shape = {2, input_rank};
    auto transpose_attr = GetI64ElementsAttr({1, 0}, &rewriter);
    Value transposed_paddings =
        rewriter.create<mhlo::TransposeOp>(loc, paddings, transpose_attr);
    Value reshaped_paddings = rewriter.create<mhlo::ReshapeOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape({input_rank * 2},
                                             paddings_elem_ty),
        transposed_paddings);

    auto left_padding_start_attr = GetI64ElementsAttr({0}, &rewriter);
    auto left_padding_limit_attr = GetI64ElementsAttr({input_rank}, &rewriter);
    auto left_padding_stride_attr = GetI64ElementsAttr({1}, &rewriter);
    Value left_padding_tensor = rewriter.create<mhlo::SliceOp>(
        loc, reshaped_paddings, left_padding_start_attr,
        left_padding_limit_attr, left_padding_stride_attr);

    auto right_padding_start_attr = GetI64ElementsAttr({input_rank}, &rewriter);
    auto right_padding_limit_attr =
        GetI64ElementsAttr({2 * input_rank}, &rewriter);
    auto right_padding_stride_attr = GetI64ElementsAttr({1}, &rewriter);
    Value right_padding_tensor = rewriter.create<mhlo::SliceOp>(
        loc, reshaped_paddings, right_padding_start_attr,
        right_padding_limit_attr, right_padding_stride_attr);

    rewriter.replaceOpWithNewOp<mhlo::DynamicPadOp>(
        op, op.getType(), input, constant_values, left_padding_tensor,
        right_padding_tensor, interior_padding_tensor);

    return success();
  }
};

class ConvertGatherNdOpDynamic : public OpRewritePattern<TF::GatherNdOp> {
  using OpRewritePattern<TF::GatherNdOp>::OpRewritePattern;
  // Converts tf.GatherNdOp to mhlo.DynamicGatherOp.
  // Here we leave 'slice_sizes' as an Attr, without defining a new
  // DynamicGatherOp, since GatherDimensionNumbers has already provide enough
  // information for shape inference and code generation of mhlo::GatherOp. '?'
  // will be filled into slice_sizes for dimensions that are dynamic sized.
  // TODO(disc): To recover static special case's performance with folding and
  // canonicalization.
  LogicalResult matchAndRewrite(TF::GatherNdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto params = op.getParams();
    auto params_ty = params.getType().dyn_cast<RankedTensorType>();
    auto indices = op.getIndices();
    auto indices_ty = indices.getType().dyn_cast<RankedTensorType>();
    auto params_rank = params_ty.getRank();
    auto indices_rank = indices_ty.getRank();
    int64_t num_index_dims = indices_ty.getDimSize(indices_rank - 1);
    if (!params_ty || !indices_ty) return failure();
    // the last dim of indices of GatherNdOp must be fixed shaped
    if (num_index_dims == ShapedType::kDynamic) return failure();

    SmallVector<int64_t, 4> slice_sizes;
    slice_sizes.reserve(params_rank);
    for (int64_t i = 0; i < params_rank; ++i) {
      if (i < num_index_dims) {
        slice_sizes.push_back(1);
      } else {
        // potentially dynamic
        int64_t dim_size = params_ty.getDimSize(i);
        slice_sizes.push_back(dim_size);
      }
    }
    SmallVector<Value, 4> slice_sizes_vals;
    Value slice_sizes_value = nullptr;
    for (int64_t i = 0; i < params_rank; ++i) {
      if (i < num_index_dims) {
        slice_sizes_vals.push_back(rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(indices_ty.getElementType(), 1)));
      } else {
        int64_t dim_size = params_ty.getDimSize(i);
        if (dim_size != ShapedType::kDynamic) {
          slice_sizes_vals.push_back(rewriter.create<arith::ConstantOp>(
              loc,
              rewriter.getIntegerAttr(indices_ty.getElementType(), dim_size)));
        } else {
          slice_sizes_vals.push_back(rewriter.create<arith::IndexCastOp>(
              loc, indices_ty.getElementType(),
              rewriter.create<tensor::DimOp>(loc, params, i)));
        }
      }
    }
    slice_sizes_value =
        rewriter.create<tensor::FromElementsOp>(loc, slice_sizes_vals);

    // collapsed_slice_dims
    SmallVector<int64_t, 4> collapsed_slice_dims;
    collapsed_slice_dims.reserve(num_index_dims);
    for (int64_t i = 0; i < num_index_dims; ++i) {
      collapsed_slice_dims.push_back(i);
    }
    // offset_dims
    SmallVector<int64_t, 4> offset_dims;
    offset_dims.reserve(params_rank - num_index_dims);
    for (int64_t i = num_index_dims; i < params_rank; i++) {
      offset_dims.push_back(i + indices_rank - 1 - num_index_dims);
    }
    // start_index_map
    SmallVector<int64_t, 4> start_index_map;
    offset_dims.reserve(num_index_dims);
    for (int64_t i = 0; i < num_index_dims; i++) {
      start_index_map.push_back(i);
    }
    // index_vector_dim
    int64_t index_vector_dim = indices_rank - 1;

    auto dims_attr = GatherDimensionNumbersAttr::get(
        rewriter.getContext(), offset_dims, collapsed_slice_dims,
        start_index_map, index_vector_dim);
    // TODO(disc): Remove this if-statement once fold and canonicalization is
    // implemented.
    if (params_ty.hasStaticShape() && indices_ty.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::GatherOp>(
          op, op.getType(), op.getParams(), op.getIndices(), dims_attr,
          GetI64ElementsAttr(slice_sizes, &rewriter));
    } else {
      rewriter.replaceOpWithNewOp<mhlo::DynamicGatherOp>(
          op, op.getType(), op.getParams(), op.getIndices(), slice_sizes_value,
          dims_attr);
    }
    return success();
  }
};

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
    auto l = op.getX();
    auto r = op.getY();
    auto element_type = getElementTypeOrSelf(l.getType());
    if (!element_type.isBF16()) return failure();

    auto out_type = op.getZ().getType().cast<TensorType>();

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
    auto input_type = op.getInput().getType().dyn_cast<RankedTensorType>();
    auto output_type = op.getOutput().getType();
    if (!input_type) {
      return rewriter.notifyMatchFailure(op, "requires ranked input shape");
    }
    llvm::SmallVector<int64_t, 4> broadcast_dimensions;
    if (input_type.getRank() > 0) {
      auto ranked_output_type = output_type.dyn_cast<RankedTensorType>();
      if (!ranked_output_type) {
        return rewriter.notifyMatchFailure(op, "requires ranked output shape");
      }
      auto rank_diff = ranked_output_type.getRank() - input_type.getRank();
      // The tf.BroadcastTo op performs "right-aligned" numpy-style
      // broadcasting.
      broadcast_dimensions = llvm::to_vector<4>(
          llvm::seq<int64_t>(rank_diff, ranked_output_type.getRank()));
    }
    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        op, output_type, op.getInput(), op.getShape(),
        rewriter.getI64TensorAttr(broadcast_dimensions));
    return success();
  }
};

/// Converts a TF::RollOp to HLO. Only support 0D axis and shift case, and axis
/// have to be a constant.
class ConvertRollOp : public OpRewritePattern<TF::RollOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::RollOp op,
                                PatternRewriter &rewriter) const override {
    auto shift_ty = op.getShift().getType().dyn_cast<RankedTensorType>();
    if (!shift_ty || shift_ty.getRank() != 0) {
      return rewriter.notifyMatchFailure(
          op, "require the type of shift to be 0D tensor");
    }

    APInt val;
    if (!matchPattern(op.getAxis(), m_ConstantInt(&val))) {
      return rewriter.notifyMatchFailure(op, "require axis to be constant");
    }
    int axis = val.getSExtValue();

    auto input_ty = op.getInput().getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "require the type of input to have static shapes");
    }
    ArrayRef<int64_t> input_shape = input_ty.getShape();
    int input_rank = input_ty.getRank();
    if (axis < 0) axis += input_rank;

    // Adjust large offsets into [0, axis_size). This also makes negative
    // offsets positive.
    // offset = ((offset % axis_size) + axis_size) % axis_size
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value offset = op.getShift();
    auto axis_size = b.create<mhlo::ConstantOp>(b.getIntegerAttr(
        getElementTypeOrSelf(offset.getType()), input_shape[axis]));
    offset = b.create<RemOp>(
        b.create<AddOp>(b.create<RemOp>(offset, axis_size), axis_size),
        axis_size);

    // Stack two copies of the dimension, then slice from the calculated
    // offset. This also works if shift is not constant.
    // DynamicSliceOp requires the sizes being integer, and we can get the
    // information from input shape.
    auto concat = b.create<ConcatenateOp>(
        ValueRange{op.getInput(), op.getInput()}, b.getI64IntegerAttr(axis));
    Value zero = b.create<mhlo::ConstantOp>(
        b.getIntegerAttr(getElementTypeOrSelf(offset.getType()), 0));
    SmallVector<Value> slice_begin_indices(input_rank, zero);
    slice_begin_indices[axis] = b.create<SubtractOp>(axis_size, offset);
    rewriter.replaceOpWithNewOp<DynamicSliceOp>(
        op, input_ty, concat, slice_begin_indices,
        rewriter.getI64TensorAttr(input_shape));
    return success();
  }
};

/// Converts a TF::LeakyReluOp to HLO.
/// LeakyRelu(x) = alpha * x if x < 0 else x.
class ConvertLeakyReluOp : public OpRewritePattern<TF::LeakyReluOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::LeakyReluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value features = op.getFeatures();

    // Use ConstantLike for `alpha` to match the shape of feature.
    auto alphaVal = chlo::getConstantLike(
        rewriter, loc, op.getAlpha().convertToFloat(), features);
    Value zeroVal = chlo::getConstantLike(rewriter, loc, 0.0, features);

    Value leakyActivationVal =
        rewriter.create<mhlo::MulOp>(loc, features, alphaVal);

    Value compareGtZero = rewriter.create<mhlo::CompareOp>(
        loc, features, zeroVal, ComparisonDirection::GT);

    rewriter.replaceOpWithNewOp<SelectOp>(op, compareGtZero, features,
                                          leakyActivationVal);
    return success();
  }
};

/// Converts a TF::LeakyReluGradOp to HLO.
/// LeakyReluGrad(gradient, inputs) = gradient if input > 0
/// else alpha * gradient.
class ConvertLeakyReluGradOp : public OpRewritePattern<TF::LeakyReluGradOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::LeakyReluGradOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value gradients = op.getGradients();
    Value features = op.getFeatures();
    auto featureType = features.getType();

    // Use ConstantLike for `alpha` to match the shape of feature.
    auto alphaVal = chlo::getConstantLike(
        rewriter, loc, op.getAlpha().convertToFloat(), features);
    Value zeroVal = chlo::getConstantLike(rewriter, loc, 0.0, features);

    Value leakyGradientVal =
        rewriter.create<mhlo::MulOp>(loc, gradients, alphaVal);

    Value compareGtZero = rewriter.create<mhlo::CompareOp>(
        loc, features, zeroVal, ComparisonDirection::GT);

    rewriter.replaceOpWithNewOp<SelectOp>(op, featureType, compareGtZero,
                                          gradients, leakyGradientVal);
    return success();
  }
};

// Converts TensorFlow DiagPartOp to HLO ops using reduction on masked matrix.
// For a Rank-2 input, it creates the following ops:
//   %1 = "mhlo.iota"() {iota_dimension = 0 : i64}
//   %2 = "mhlo.iota"() {iota_dimension = 1 : i64}
//   %3 = "mhlo.compare"(%1, %2) {comparison_direction = "EQ"}
//   %4 = mhlo.constant dense<0.000000e+00> : tensor<f32>
//   %5 = "mhlo.broadcast"(%4)
//   %6 = "mhlo.select"(%3, %input, %5)
//   %7 = "mhlo.reduce"(%6, %4) ({
//   ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
//     %9 = mhlo.add %arg1, %arg2 : tensor<f32>
//     "mhlo.return"(%9) : (tensor<f32>) -> ()
//   }) {dimensions = dense<0> : tensor<1xi64>}
//
// If the input's rank N is greater than 2, we will reshape it to R2 first and
// create the above ops, then reshape it back to rank N/2.
class ConvertDiagPartOp : public OpRewritePattern<TF::DiagPartOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::DiagPartOp op,
                                PatternRewriter &rewriter) const override {
    auto input_type = op.getInput().getType().dyn_cast<RankedTensorType>();
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
    Value reshaped_input = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(),
        tensorflow::GetTypeFromTFTensorShape({new_size, new_size},
                                             input_type.getElementType()),
        op.getInput());
    auto iota_type = tensorflow::GetTypeFromTFTensorShape(
        {new_size, new_size}, rewriter.getIntegerType(32));
    auto iota0 = rewriter.create<IotaOp>(op.getLoc(), iota_type,
                                         rewriter.getI64IntegerAttr(0));
    auto iota1 = rewriter.create<IotaOp>(op.getLoc(), iota_type,
                                         rewriter.getI64IntegerAttr(1));
    Value compare = rewriter.create<CompareOp>(op.getLoc(), iota0, iota1,
                                               ComparisonDirection::EQ);
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
    BuildReduceBody<AddOp>(input_type.getElementType(), &reduce.getBody(),
                           &rewriter);
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        op,
        tensorflow::GetTypeFromTFTensorShape(new_dims,
                                             input_type.getElementType()),
        reduce.getResult(0));
    return success();
  }
};

// Converts TensorFlow MatrixDiagPartOp to HLO ops.
class ConvertMatrixDiagPartV3Op
    : public OpRewritePattern<TF::MatrixDiagPartV3Op> {
  using Shape = llvm::SmallVector<int64_t, 4>;

  // Parse the "k" parameter. MatrixDiagPartV3 allows to specify the diagonal(s)
  // with k. This can be either a single value (for a single diagonal) or a
  // tuple of two values (starting and ending diagonal, for a band).
  LogicalResult ExtractK(TF::MatrixDiagPartV3Op op, int64_t (*k)[2]) const {
    DenseIntElementsAttr kattr;
    if (!matchPattern(op.getK(), m_Constant(&kattr))) {
      return failure();
    }
    DenseIntElementsAttr::iterator it = kattr.begin();
    (*k)[0] = (*it).getSExtValue();
    it++;
    if (it == kattr.end()) {
      // Handle input like e.g. "k = 5", in which case we extract a single
      // diagonal.
      (*k)[1] = (*k)[0];
    } else {
      // Handle input like e.g. "k = [-1, 1]", in which case we extract a
      // band (multiple diagonals).
      (*k)[1] = (*it).getSExtValue();
    }
    return success();
  }

  // Utility method for broadcasting integer constants to a given shape.
  BroadcastOp BroadcastConstant(Location loc, Shape shape, int32_t constant,
                                int int_size, PatternRewriter &rewriter) const {
    return rewriter.create<BroadcastOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape(shape,
                                             rewriter.getIntegerType(int_size)),
        GetScalarConstOfType(rewriter.getIntegerType(int_size), loc, constant,
                             &rewriter),
        GetI64ElementsAttr(shape, &rewriter));
  }

 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::MatrixDiagPartV3Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ShapedType input_type = op.getInput().getType().dyn_cast<ShapedType>();

    // Align is a string specifying how superdiagonals and subdiagonals should
    // be aligned/padded for diagonals that are shorter than max_diag_len. The
    // format is "{super}_{sub}", with {super} the superdiagonal alignment and
    // {sub} the subdiagonal alignment. "LEFT" means rows will be padded to the
    // left, "RIGHT" means rows will be padded ot the right.  The default is
    // "RIGHT_LEFT".
    StringRef align = op->getAttrOfType<StringAttr>("align").getValue();
    enum Alignment { kLeft, kRight };

    // default is RIGHT_LEFT
    Alignment superdiagonal_align = kRight;
    Alignment subdiagonal_align = kLeft;

    if (align == "RIGHT_LEFT") {
      superdiagonal_align = kRight;
      subdiagonal_align = kLeft;
    } else if (align == "RIGHT_RIGHT") {
      superdiagonal_align = kRight;
      subdiagonal_align = kRight;
    } else if (align == "LEFT_RIGHT") {
      superdiagonal_align = kLeft;
      subdiagonal_align = kRight;
    } else if (align == "LEFT_LEFT") {
      superdiagonal_align = kLeft;
      subdiagonal_align = kLeft;
    } else {
      return failure();  // unsupported alignment
    }

    // MatrixDiagPart operates on a matrix of shape [I, J, ..., L, M, N], and
    // will extract the diagonal(s) out of [M, N], for all [I, J, ..., L].
    if (!input_type || !input_type.hasStaticShape()) return failure();
    int64_t num_dims = input_type.getRank();
    if (num_dims < 2) return failure();
    int64_t rows = input_type.getDimSize(num_dims - 2);  // rows
    int64_t cols = input_type.getDimSize(num_dims - 1);  // cols

    // We extract the diagonals from k[0] up to and including k[1].
    // Addressing is 0 for the main diagonal. (So k = [0, 0] would just extract
    // the main diagonal). It's negative for subdiagonals (under and to the left
    // of the main diagonal) and positive for superdiagonals (above and to the
    // right of the main diagonal).
    int64_t k[2];
    if (failed(ExtractK(op, &k))) return failure();
    int num_diags = k[1] - k[0] + 1;

    // Shifting diagonals away from the main diagonal might shorten them. This
    // is the longest diagonal we will see. We make this the last dimension of
    // the output shape.
    int64_t max_diag_len =
        std::min(rows + std::min(k[1], static_cast<int64_t>(0)),
                 cols + std::min(-k[0], static_cast<int64_t>(0)));

    // The first dimension is the index vector dimension we'll use for gather.
    // It's 1 here, but will be 2 once we glue x and y together.
    Shape indices_shape({1, num_diags, max_diag_len});

    RankedTensorType iota_type = tensorflow::GetTypeFromTFTensorShape(
        indices_shape, rewriter.getIntegerType(32));
    Value iotaM =
        rewriter.create<IotaOp>(loc, iota_type, rewriter.getI64IntegerAttr(1));
    Value iotaN =
        rewriter.create<IotaOp>(loc, iota_type, rewriter.getI64IntegerAttr(2));

    // Boradcasted constants, of the same shape as iotaM and iotaN.
    Value b_zero = BroadcastConstant(loc, indices_shape, 0, 32, rewriter);
    Value b_false = BroadcastConstant(loc, indices_shape, 0, 1, rewriter);
    Value b_true = BroadcastConstant(loc, indices_shape, 1, 1, rewriter);
    Value b_k1 = BroadcastConstant(loc, indices_shape, k[1], 32, rewriter);
    Value b_rows = BroadcastConstant(loc, indices_shape, rows, 32, rewriter);
    Value b_cols = BroadcastConstant(loc, indices_shape, cols, 32, rewriter);
    Value b_max_diag_len =
        BroadcastConstant(loc, indices_shape, max_diag_len, 32, rewriter);

    // d = k[1] - m
    // (A.k.a. the number of the diagonal, depending on m. Note that we
    //  subtract m here. This means we start with the superdiagonals and
    //  move downwards towards the subdiagonals. So the start indices will
    //  be decreasing.)
    Value d = rewriter.create<SubtractOp>(loc, b_k1, iotaM);
    Value neg_d = rewriter.create<NegOp>(loc, d);

    // diag_len_d = min(rows + min(d, 0), cols - max(d, 0))
    // (Length of a diagonal for a given d. Same as max_diag_len for m = 0.)
    Value diag_len_d = rewriter.create<MinOp>(
        loc,
        rewriter.create<AddOp>(loc, b_rows,
                               rewriter.create<MinOp>(loc, d, b_zero)),
        rewriter.create<SubtractOp>(loc, b_cols,
                                    rewriter.create<MaxOp>(loc, d, b_zero)));

    // offset is max_diag_len - diag_len_d if we're padding, 0 otherwise.
    Value cmp;
    if (subdiagonal_align == kRight && superdiagonal_align == kRight) {
      cmp = b_true;
    } else if (superdiagonal_align == kRight) {
      // offset = d>=0 ? max_diag_len - diag_len_d : 0
      cmp = rewriter.create<TF::GreaterEqualOp>(loc, d, b_zero);
    } else if (subdiagonal_align == kRight) {
      // offset = d<=0 ? max_diag_len - diag_len_d : 0
      cmp = rewriter.create<TF::LessEqualOp>(loc, d, b_zero);
    } else {
      // offset = 0
      cmp = b_false;
    }

    // This offset shifts the diagonals to the "left" or "right", depending
    // on alignment.
    Value offset = rewriter.create<SelectOp>(
        loc, b_zero.getType(), cmp,
        rewriter.create<SubtractOp>(loc, b_max_diag_len, diag_len_d), b_zero);

    // x = max(d, 0) - offset
    // y = max(-d, 0) - offset
    Value x = rewriter.create<SubtractOp>(
        loc, rewriter.create<MaxOp>(loc, d, b_zero), offset);
    Value y = rewriter.create<SubtractOp>(
        loc, rewriter.create<MaxOp>(loc, neg_d, b_zero), offset);

    Value n_plus_x = rewriter.create<AddOp>(loc, iotaN, x);
    Value n_plus_y = rewriter.create<AddOp>(loc, iotaN, y);

    // GatherOp is happy about letting us index out of bounds values, but those
    // values will be undefined. So we mask them later. Set up the boolean
    // expression that tells us which entries, in the output shape, are out of
    // bounds and thus become the padding_value.
    Value x_in_bounds = rewriter.create<AndOp>(
        loc,
        rewriter.create<TF::GreaterEqualOp>(loc, b_false.getType(), n_plus_x,
                                            b_zero),
        rewriter.create<TF::LessOp>(loc, b_false.getType(), n_plus_x, b_cols));
    Value y_in_bounds = rewriter.create<AndOp>(
        loc,
        rewriter.create<TF::GreaterEqualOp>(loc, b_false.getType(), n_plus_y,
                                            b_zero),
        rewriter.create<TF::LessOp>(loc, b_false.getType(), n_plus_y, b_rows));
    Value in_bounds = rewriter.create<ReshapeOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape(Shape({num_diags, max_diag_len}),
                                             rewriter.getIntegerType(1)),
        rewriter.create<AndOp>(loc, x_in_bounds, y_in_bounds));

    // Now combine x and y into the index data structure needed for gather.
    Shape concat_shape({2, num_diags, max_diag_len});
    Value start_indices = rewriter.create<ConcatenateOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape(concat_shape,
                                             rewriter.getIntegerType(32)),
        mlir::ValueRange({n_plus_y, n_plus_x}),
        mlir::IntegerAttr::get(rewriter.getIntegerType(64), 0));

    // Shape of the final output. (Except for dimension folding in the
    // single diagonal case.)
    Shape output_shape;
    for (int i = 0; i < num_dims - 2; i++) {
      output_shape.push_back(input_type.getDimSize(i));
    }
    output_shape.push_back(num_diags);
    output_shape.push_back(max_diag_len);

    // A slice is the shape of what GatherOp copies per lookup. So the last
    // two dimensions (M, N in the matrix-diag-part docs) are where we go
    // through entry by entry.
    ArrayRef<int64_t> input_shape = input_type.getShape();
    Shape slice_sizes(input_shape.begin(), input_shape.end());
    int slice_dimensions = slice_sizes.size();
    slice_sizes[slice_dimensions - 2] = 1;
    slice_sizes[slice_dimensions - 1] = 1;

    // Dimensions of the input we won't see in the output (M and N).
    SmallVector<int64_t, 2> collapsed_dims(
        {slice_dimensions - 2, slice_dimensions - 1});

    // Which dimensions (in the input) the two offset "columns" map to.
    SmallVector<int64_t, 2> start_index_map({num_dims - 2, num_dims - 1});

    // Gather the diagonal entries.
    // TODO(kramm): For a single diagonal, this might be slower than the
    //              mask + sum approach. Special-case num_diags==1?
    auto dims_attr = GatherDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*offset_dims=*/llvm::to_vector<4>(llvm::seq<int64_t>(0, num_dims - 2)),
        /*collapsed_slice_dims=*/collapsed_dims, start_index_map,
        /*index_vector_dim=*/0);
    Value gather = rewriter.create<mhlo::GatherOp>(
        loc, op.getInput(), start_indices, dims_attr,
        GetI64ElementsAttr(slice_sizes, &rewriter));

    // We now need to broadcast the "in_bounds" boolean expression, as well as
    // the padding value, to do the final select.
    Shape broadcast_bounds;
    for (int i = 0; i < output_shape.size() - 2; i++) {
      broadcast_bounds.push_back(output_shape[i]);
    }
    Value b_in_bounds = rewriter.create<BroadcastOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape(output_shape,
                                             rewriter.getIntegerType(1)),
        in_bounds, GetI64ElementsAttr(broadcast_bounds, &rewriter));
    Value b_padding = rewriter.create<BroadcastOp>(
        loc, op.getPaddingValue(), GetI64ElementsAttr(output_shape, &rewriter));

    // Replace all out-of-bounds values in the result with padding_value.
    Value result =
        rewriter.create<SelectOp>(loc, b_in_bounds, gather, b_padding);

    if (num_diags == 1) {
      // matrix_diag_part folds away the 1-sized band dimension if we only
      // extract a single diagonal.
      result = rewriter.create<ReshapeOp>(loc, op.getType(), result);
    }

    rewriter.replaceOp(op, result);
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
    StringAttr equation = op->getAttrOfType<StringAttr>("equation");
    if (op.getN() == 1) {
      rewriter.replaceOpWithNewOp<UnaryEinsumOp>(
          op, op.getType(), *op.getInputs().begin(), equation);
    } else if (op.getN() == 2) {
      ValueRange inputs = op.getInputs();
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

// Bypasses IdentityN op.
class ConvertIdentityNOp : public OpRewritePattern<TF::IdentityNOp> {
 public:
  using OpRewritePattern<TF::IdentityNOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::IdentityNOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getOperands());
    return success();
  }
};

template <typename OpTy>
class ConvertFFTOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto input_ty = op.getInput().getType().template cast<ShapedType>();
    if (!input_ty.hasRank()) {
      return failure();
    }
    auto input_shape = input_ty.getShape();
    DenseIntElementsAttr fft_length_attr;
    if (!matchPattern(op.getFftLength(), m_Constant(&fft_length_attr))) {
      return failure();
    }
    int64_t fft_length;
    if (fft_length_attr.getNumElements() != 0) {
      fft_length = fft_length_attr.getValues<IntegerAttr>()[0].getInt();
    } else {
      return failure();
    }

    int64_t expected_dim = fft_length;
    std::string fft_string = "RFFT";
    if (typeid(OpTy) == typeid(TF::IRFFTOp)) {
      expected_dim = fft_length / 2 + 1;
      fft_string = "IRFFT";
    }
    Location loc = op.getLoc();

    // The inner-most dim cannot be dynamic.
    if (input_ty.isDynamicDim(input_shape.size() - 1)) {
      return failure();
    }

    auto expected_shape = llvm::to_vector<4>(input_shape.drop_back());
    expected_shape.push_back(expected_dim);

    // Zero pad or truncate the last axis
    Value reshaped = op.getInput();
    SmallVector<int64_t, 4> begin_indices(input_shape.size(), 0);
    SmallVector<int64_t, 4> strides(input_shape.size(), 1);

    // Last dim larger than expected_dim, slice the input
    if (input_shape.back() > expected_dim) {
      reshaped = rewriter.create<SliceOp>(
          op.getLoc(),
          tensorflow::GetTypeFromTFTensorShape(expected_shape,
                                               input_ty.getElementType()),
          op.getInput(), GetI64ElementsAttr(begin_indices, &rewriter),
          GetI64ElementsAttr(expected_shape, &rewriter),
          GetI64ElementsAttr(strides, &rewriter));

      // Last dim smaller than expected_dim, zero-pad the input
    } else if (input_ty.getShape().back() < expected_dim) {
      SmallVector<int64_t, 4> no_padding(input_shape.size(), 0);
      SmallVector<int64_t, 4> padding(input_shape.size() - 1, 0);
      padding.push_back(expected_dim - input_shape.back());
      Value zero =
          GetScalarConstOfType(input_ty.getElementType(), loc, 0, &rewriter);
      reshaped = rewriter.create<PadOp>(
          loc,
          tensorflow::GetTypeFromTFTensorShape(expected_shape,
                                               input_ty.getElementType()),
          op.getInput(), zero, GetI64ElementsAttr(no_padding, &rewriter),
          GetI64ElementsAttr(padding, &rewriter),
          GetI64ElementsAttr(no_padding, &rewriter));
    }

    rewriter.replaceOpWithNewOp<FftOp>(
        op, op.getType(), reshaped,
        FftTypeAttr::get(rewriter.getContext(),
                         symbolizeFftType(fft_string).value()),
        rewriter.getI64TensorAttr(fft_length));
    return success();
  }
};

using ConvertRFFTOp = ConvertFFTOp<TF::RFFTOp>;
using ConvertIRFFTOp = ConvertFFTOp<TF::IRFFTOp>;

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
    Value grad = op.getYBackprop();
    Value act = op.getX();
    Value scale = op.getScale();
    Value mean = op.getReserveSpace_1();
    Value var = op.getReserveSpace_2();

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

    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.getDataFormat().str(), &data_format))
      return op.emitOpError("invalid data format");

    auto feature_dim_attr = getFeatureDimensionAttr(rewriter, data_format, act);
    auto feature_dim = feature_dim_attr.getValue().getSExtValue();

    // Gets the result values.
    Value x_backprop, scale_backprop, offset_backprop;
    if (op.getIsTraining()) {  // training
      // TODO(b/145536565): handle GPU logic separately.
      // Infers the output type with the converted `act`.
      Type feature_type = tensorflow::GetTypeFromTFTensorShape(
          {GetDimSize(act_type, feature_dim)}, kernel_type);

      SmallVector<Type, 3> operand_types = {act.getType(), feature_type,
                                            feature_type};
      auto training_op = rewriter.create<BatchNormGradOp>(
          loc, operand_types, act, scale, mean, var, grad, op.getEpsilon(),
          feature_dim);

      x_backprop = training_op.getResult(0);

      scale_backprop = training_op.getResult(1);

      offset_backprop = training_op.getResult(2);
    } else {  // inference
      SmallVector<int64_t, 4> non_feature_dims;
      for (int64_t i = 0; i < act_type.getRank(); ++i) {
        if (i == feature_dim) continue;
        non_feature_dims.push_back(i);
      }
      auto reduce_dims = GetI64ElementsAttr(non_feature_dims, &rewriter);
      auto scalar_broadcast_dims = GetI64ElementsAttr({}, &rewriter);

      // scratch1 = rsqrt(var + epsilon)
      RankedTensorType scalar_float =
          tensorflow::GetTypeFromTFTensorShape({}, kernel_type);
      auto epsilon = rewriter.create<ConstantOp>(
          loc, DenseFPElementsAttr::get(scalar_float, {op.getEpsilon()}));
      auto add_op = rewriter.create<chlo::BroadcastAddOp>(
          loc, var, epsilon.getResult(), scalar_broadcast_dims);

      Value scratch1 = rewriter.create<RsqrtOp>(loc, add_op);

      // scratch2 = sum(y_backprop * (x - mean))
      auto sub_op = rewriter.create<mhlo::SubtractOp>(
          loc, act,
          Broadcast1DToFeatureDim(loc, act, mean, feature_dim, rewriter));
      auto weighted_grad = rewriter.create<mhlo::MulOp>(loc, grad, sub_op);
      Value scratch2 =
          ApplyReduction(loc, weighted_grad, reduce_dims, &rewriter);

      // x_backprop = y_backprop * (scale * scratch1)
      auto scaled_grad =
          rewriter.create<mhlo::MulOp>(loc, op.getScale(), scratch1);
      x_backprop = rewriter.create<mhlo::MulOp>(
          loc, grad,
          Broadcast1DToFeatureDim(loc, act, scaled_grad, feature_dim,
                                  rewriter));

      // scale_backprop = scratch2 * scratch1
      scale_backprop = rewriter.create<mhlo::MulOp>(loc, scratch1, scratch2);

      // offset_backprop = sum(y_backprop)
      offset_backprop = ApplyReduction(loc, grad, reduce_dims, &rewriter);
    }

    x_backprop = rewriter.create<ConvertOp>(loc, x_backprop, act_ele_type);
    Value last_val[2];
    if (op.getResult(3).use_empty() && op.getResult(4).use_empty()) {
      // It doesn't matter what values we provide for the last 2 results.
      last_val[0] = last_val[1] = op.getX();
    } else {
      auto const_val = rewriter.create<ConstantOp>(
          op.getLoc(), DenseElementsAttr::get<float>(
                           tensorflow::GetTypeFromTFTensorShape(
                               {0}, getElementTypeOrSelf(op.getResult(3))),
                           0.0));
      auto maybe_cast = [&](Value val, Type t) -> Value {
        if (val.getType() == t) return val;
        return rewriter.create<tensor::CastOp>(op.getLoc(), t, val);
      };
      last_val[0] = maybe_cast(const_val, op.getResult(3).getType());
      last_val[1] = maybe_cast(const_val, op.getResult(4).getType());
    }
    rewriter.replaceOp(
        op, {/*x_backprop=*/x_backprop,
             /*scale_backprop=*/scale_backprop,
             /*offset_backprop=*/offset_backprop, last_val[0], last_val[1]});
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
template <typename FusedBatchNormOpT>
class ConvertFusedBatchNormBase : public OpRewritePattern<FusedBatchNormOpT> {
 public:
  using OpRewritePattern<FusedBatchNormOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(FusedBatchNormOpT op,
                                PatternRewriter &rewriter) const override {
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.getDataFormat().str(), &data_format))
      return op.emitOpError("invalid data format");

    auto feature_dim =
        getFeatureDimensionAttr(rewriter, data_format, op.getX());

    auto input_type_tensor = op.getX().getType().template cast<TensorType>();
    auto input_element_type = input_type_tensor.getElementType();

    auto scale_type_tensor =
        op.getScale().getType().template cast<TensorType>();
    auto scale_element_type = scale_type_tensor.getElementType();

    auto mean_type_tensor = op.getMean().getType().template cast<TensorType>();
    auto mean_element_type = mean_type_tensor.getElementType();
    // In the training case, dimensions of input tensors must be static.
    if (op.getIsTraining() && (!input_type_tensor.hasStaticShape() ||
                               !scale_type_tensor.hasStaticShape() ||
                               !mean_type_tensor.hasStaticShape()))
      return failure();

    // TODO(b/69928690): Support mixed precision in the XLA batch
    // normalization operators. As a workaround, create a new x with the same
    // element type as scale (which may be more precise than the input type).
    Value bn_train_input = rewriter.create<mhlo::ConvertOp>(
        op.getLoc(), op.getX(), scale_element_type);
    TensorType bn_train_input_type_tensor =
        bn_train_input.getType().template cast<TensorType>();

    if (op.getIsTraining()) {
      // Training case.
      auto operand_shape = bn_train_input_type_tensor.getShape();
      // The mean and variance are each 1 dimensional arrays the size of the
      // feature dimension, with the same element type as the operand (x).
      // This shape must be constructed manually because the mean and variance
      // inputs are empty in the training case.
      Type mean_var_type = tensorflow::GetTypeFromTFTensorShape(
          {operand_shape[feature_dim.getInt()]}, scale_element_type);
      // Op result type is a tuple of 3 values: output with same shape as input;
      // batch_mean, and batch_var.
      SmallVector<Type, 3> operand_types = {bn_train_input_type_tensor,
                                            mean_var_type, mean_var_type};
      auto bn_train_op = rewriter.create<mhlo::BatchNormTrainingOp>(
          op.getLoc(), operand_types, bn_train_input, op.getScale(),
          op.getOffset(), op.getEpsilon(), feature_dim.getInt());
      // HLO op outputs a tuple of tensors. Extract those results.
      Value y_out = bn_train_op.getResult(0);
      Value batch_mean = bn_train_op.getResult(1);
      Value reserve_space_1 = batch_mean;
      Value batch_variance = bn_train_op.getResult(2);

      // Apply Bessel's correction on the variance.
      int total_input_size = bn_train_input_type_tensor.getNumElements();
      int total_scale_size = scale_type_tensor.getNumElements();
      int sample_size = total_input_size / total_scale_size;
      int sample_size_minus_one = std::max(1, sample_size - 1);
      double factor = static_cast<double>(sample_size) /
                      static_cast<double>(sample_size_minus_one);
      auto factor_const_op = rewriter.create<mhlo::ConstantOp>(
          op.getLoc(), rewriter.getFloatAttr(scale_element_type, factor));

      Value corrected_variance = rewriter.create<chlo::BroadcastMulOp>(
          op.getLoc(), batch_variance.getType(), batch_variance,
          factor_const_op, /*broadcast_dimensions=*/DenseIntElementsAttr());

      // Convert back to input type to stay aligned with expected output type
      // for TF op.
      y_out = rewriter.create<mhlo::ConvertOp>(op.getLoc(), y_out,
                                               input_element_type);

      float exponential_avg_factor =
          op.getExponentialAvgFactor().convertToFloat();
      if (exponential_avg_factor != 1.0f) {
        auto alpha = rewriter.create<mhlo::ConstantOp>(
            op.getLoc(), rewriter.getFloatAttr(mean_element_type,
                                               1.0f - exponential_avg_factor));
        auto beta = rewriter.create<mhlo::ConstantOp>(
            op.getLoc(),
            rewriter.getFloatAttr(mean_element_type, exponential_avg_factor));

        // new_running_mean = alpha * old_mean + beta * batch_mean.
        auto alpha_mul_old_mean = rewriter.create<chlo::BroadcastMulOp>(
            op.getLoc(), op.getMean().getType(), alpha, op.getMean(),
            /*broadcast_dimensions=*/DenseIntElementsAttr());
        auto beta_mul_batch_mean = rewriter.create<chlo::BroadcastMulOp>(
            op.getLoc(), batch_mean.getType(), beta, batch_mean,
            /*broadcast_dimensions=*/DenseIntElementsAttr());
        batch_mean = rewriter.create<chlo::BroadcastAddOp>(
            op.getLoc(), alpha_mul_old_mean, beta_mul_batch_mean,
            /*broadcast_dimensions=*/DenseIntElementsAttr());

        // new_running_variance = alpha * old_variance + beta * batch_variance.
        auto alpha_mul_old_variance = rewriter.create<chlo::BroadcastMulOp>(
            op.getLoc(), op.getVariance().getType(), alpha, op.getVariance(),
            /*broadcast_dimensions=*/DenseIntElementsAttr());
        auto beta_mul_batch_variance = rewriter.create<chlo::BroadcastMulOp>(
            op.getLoc(), corrected_variance.getType(), beta, corrected_variance,
            /*broadcast_dimensions=*/DenseIntElementsAttr());
        corrected_variance = rewriter.create<chlo::BroadcastAddOp>(
            op.getLoc(), alpha_mul_old_variance, beta_mul_batch_variance,
            /*broadcast_dimensions=*/DenseIntElementsAttr());
      }

      if (std::is_same<FusedBatchNormOpT, TF::FusedBatchNormV2Op>::value) {
        // FusedBatchNormV2 expects 4 outputs.
        // Outputs 3 and 4 are currently marked as "reserved spaces 1 and 2".
        // They are used to pass the per-batch mean and variance to the
        // gradiant. Here we maintain the same behavior by setting them to the
        // mean and variance calculated by BatchNormTraining.
        rewriter.replaceOp(op, {y_out, /*batch_mean=*/batch_mean,
                                /*batch_variance=*/corrected_variance,
                                /*reserve_space_1=*/reserve_space_1,
                                /*reserve_space_2=*/batch_variance});
      } else {  // TF::FusedBatchNormV3Op
        // For FusedBatchNormV3Op, also create a constant tensor to forward to
        // last reserve_space_3 output.
        auto reserve_space_3_type =
            op.getResult(5).getType().template cast<TensorType>();
        int num_elements = reserve_space_3_type.hasStaticShape()
                               ? reserve_space_3_type.getNumElements()
                               : 0;
        auto const_attr_type = tensorflow::GetTypeFromTFTensorShape(
            {num_elements}, getElementTypeOrSelf(reserve_space_3_type));
        Value dummy_const = rewriter.create<ConstantOp>(
            op.getLoc(), DenseElementsAttr::get<float>(const_attr_type, 0.0));
        if (const_attr_type != reserve_space_3_type)
          dummy_const = rewriter.create<tensor::CastOp>(
              op.getLoc(), reserve_space_3_type, dummy_const);
        rewriter.replaceOp(op, {y_out, /*batch_mean=*/batch_mean,
                                /*batch_variance=*/corrected_variance,
                                /*reserve_space_1=*/reserve_space_1,
                                /*reserve_space_2=*/batch_variance,
                                /*reserve_space_3=*/dummy_const});
      }
    } else {  // Inference case.
      auto bn_train_op = rewriter.create<BatchNormInferenceOp>(
          op.getLoc(),
          /*result_type=*/bn_train_input_type_tensor, bn_train_input,
          op.getScale(), op.getOffset(), op.getMean(), op.getVariance(),
          op.getEpsilon(), feature_dim.getInt());

      // Convert back to input type to stay aligned with expected output type
      // for TF op.
      auto y_out = rewriter.create<mhlo::ConvertOp>(op.getLoc(), bn_train_op,
                                                    input_element_type);

      // The mean, variance, and reserved space outputs of the batch norm op are
      // not used for inference. It doesn't matter what values we provide for
      // the last 5 results as long as they are of the same type. Forward
      // input mean and variance to output mean, variance, reserved_space_1 and
      // reserved_space_2.
      if (std::is_same<FusedBatchNormOpT, TF::FusedBatchNormV2Op>::value) {
        rewriter.replaceOp(op, {/*y=*/y_out,
                                /*batch_mean=*/op.getMean(),
                                /*batch_variance=*/op.getVariance(),
                                /*reserve_space_1=*/op.getMean(),
                                /*reserve_space_2=*/op.getVariance()});
      } else {
        // For FusedBatchNormV3Op, also create a constant tensor to forward to
        // last reserve_space_3 output.
        auto reserve_space_3_type =
            op.getResult(5).getType().template cast<TensorType>();
        int num_elements = reserve_space_3_type.hasStaticShape()
                               ? reserve_space_3_type.getNumElements()
                               : 0;
        auto const_attr_type = tensorflow::GetTypeFromTFTensorShape(
            {num_elements}, getElementTypeOrSelf(reserve_space_3_type));
        Value dummy_const = rewriter.create<ConstantOp>(
            op.getLoc(), DenseElementsAttr::get<float>(const_attr_type, 0.0));
        if (const_attr_type != reserve_space_3_type)
          dummy_const = rewriter.create<tensor::CastOp>(
              op.getLoc(), reserve_space_3_type, dummy_const);
        rewriter.replaceOp(op, {/*y=*/y_out,
                                /*batch_mean=*/op.getMean(),
                                /*batch_variance=*/op.getVariance(),
                                /*reserve_space_1=*/op.getMean(),
                                /*reserve_space_2=*/op.getVariance(),
                                /*reserve_space_3=*/dummy_const});
      }
    }
    return success();
  }
};

using ConvertFusedBatchNormV2Op =
    ConvertFusedBatchNormBase<TF::FusedBatchNormV2Op>;
using ConvertFusedBatchNormV3Op =
    ConvertFusedBatchNormBase<TF::FusedBatchNormV3Op>;

using PaddingArray = std::vector<std::pair<int64_t, int64_t>>;

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
  llvm::SmallVector<int64_t, num_dims> input_shape, window_shape, strides;
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
  return DenseIntElementsAttr::get(tensorflow::GetTypeFromTFTensorShape(
                                       {rank, 2}, builder->getIntegerType(64)),
                                   flatten_paddings);
}

// Helper function for dividing each entry of `pooled` by the count of its
// corresponding window, i.e., the number of non-padding entries of the window
// which an `AvgPool` operation performed on an `input_shape`-tensor would map
// to this entry, depending on `ksize` and `strides`. This function is used for
// `AvgPool` and `AvgPoolGrad` legalizations.
// `zero` is passed as a parameter because it can be reused from caller level.
// `pooled` must have `RankedTensorType`.
template <typename OpTy, int num_dims>
Operation *AvgPoolDivideByCount(
    Value pooled, const SmallVector<int64_t, num_dims> &input_shape,
    const SmallVector<int64_t, num_dims> &ksize,
    const SmallVector<int64_t, num_dims> &strides, OpTy op, Value zero,
    PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  RankedTensorType pooled_type =
      pooled.getType().template cast<RankedTensorType>();
  Type element_type = pooled_type.getElementType();
  Operation *result = nullptr;
  RankedTensorType orig_input_type =
      tensorflow::GetTypeFromTFTensorShape(input_shape, element_type);

  if (op.getPadding() == "VALID") {
    // All window counts are equal here because we don't have padding
    // (each entry of `pooled` corresponds to a window that consists of
    //  original input entries only).
    int64_t window_count = std::accumulate(ksize.begin(), ksize.end(), 1,
                                           std::multiplies<int64_t>());
    // Divide `pooled` by window counts.
    Value divisor =
        GetScalarConstOfType(element_type, loc, window_count, &rewriter);
    auto scalar_broadcast_dims = GetI64ElementsAttr({}, &rewriter);
    result = rewriter.create<chlo::BroadcastDivOp>(
        loc, pooled_type, pooled, divisor, scalar_broadcast_dims);
  } else {
    assert(op.getPadding() == "SAME");
    // For SAME padding, only original entries that contributed to a window
    // are counted for the average of this window, not padded entries.

    // Build all-ones tensor of same shape as the original input.
    ElementsAttr splat = hlo::getSplat(&rewriter, orig_input_type, 1);
    auto all_ones_tensor = rewriter.create<ConstantOp>(loc, splat);

    // Get padding for the input.
    DenseIntElementsAttr input_padding_attr =
        GetReduceWindowPaddingAsAttr<num_dims>(input_shape, op.getKsize(),
                                               op.getStrides(), op.getPadding(),
                                               &rewriter);

    // Count the 1's in each window, using the same padding as for the input,
    // which gives us the window counts by which `pooled` needs to be divided.
    auto divisor = rewriter.create<ReduceWindowOp>(
        loc, pooled_type,
        /*operand=*/all_ones_tensor,
        /*init_value=*/zero,
        /*window_dimensions=*/GetI64ElementsAttr(op.getKsize()),
        /*window_strides=*/GetI64ElementsAttr(op.getStrides()),
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(),
        /*padding=*/input_padding_attr);
    BuildReduceBody<AddOp>(element_type, &divisor.getBody(), &rewriter);

    // Divide `pooled` by window counts.
    result = rewriter.create<mhlo::DivOp>(loc, pooled_type, pooled,
                                          divisor.getResult(0));
  }
  return result;
}

Value GetAvgPoolInput(TF::AvgPoolOp op) { return op.getValue(); }
Value GetAvgPoolInput(TF::AvgPool3DOp op) { return op.getInput(); }

// Converts AvgPool op to HLO ReduceWindow op by setting appropriate window
// dimensions with add as the reduction function. The reduction result is
// then divided by the number of elements in the window.
template <typename OpTy, int num_dims>
class ConvertAvgPoolOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value input_value = GetAvgPoolInput(op);
    auto input_type =
        input_value.getType().template dyn_cast<RankedTensorType>();
    if (!input_type) return failure();

    // We will do accumulation first; use a larger bitwidth if suitable.
    Type input_element_type = input_type.getElementType();
    Type sum_element_type = GetSumAccumulationType(input_element_type);
    Type result_type;

    // The result type for reduction and division with the proper element type.
    if (auto ranked_type = op.getType().template dyn_cast<RankedTensorType>())
      result_type = tensorflow::GetTypeFromTFTensorShape(ranked_type.getShape(),
                                                         sum_element_type);
    else
      result_type = UnrankedTensorType::get(sum_element_type);

    // Convert if we need enlarge the element type's bitwidth.
    if (input_element_type != sum_element_type)
      input_value = rewriter.create<ConvertOp>(op.getLoc(), input_value,
                                               sum_element_type);

    // Create the ReduceWindow op.
    Value init =
        GetScalarConstOfType(sum_element_type, op.getLoc(), 0, &rewriter);
    DenseIntElementsAttr paddings_attr = GetReduceWindowPaddingAsAttr<num_dims>(
        input_type.getShape(), op.getKsize(), op.getStrides(), op.getPadding(),
        &rewriter);
    auto reduce = rewriter.create<ReduceWindowOp>(
        op.getLoc(), result_type, input_value, init,
        GetI64ElementsAttr(op.getKsize()), GetI64ElementsAttr(op.getStrides()),
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(), paddings_attr);
    BuildReduceBody<AddOp>(sum_element_type, &reduce.getBody(), &rewriter);

    // Count the number of elements in the window. The following calculation
    // is only valid for no paddings.
    SmallVector<int64_t, num_dims> input_shape(
        llvm::to_vector<num_dims>(input_type.getShape()));
    SmallVector<int64_t, num_dims> ksize, strides;
    GetI64ArrayAttrValues(op.getKsize(), &ksize);
    GetI64ArrayAttrValues(op.getStrides(), &strides);

    Operation *result_op = AvgPoolDivideByCount<OpTy, num_dims>(
        reduce.getResult(0), input_shape, ksize, strides, op, init, rewriter);

    // Convert back if we enlarged the element type's bitwidth.
    Value result = result_op->getOpResult(0);
    if (input_element_type != sum_element_type)
      result =
          rewriter.create<ConvertOp>(op.getLoc(), result, input_element_type);

    rewriter.replaceOp(op, result);
    return success();
  }
};

using ConvertAvgPool2DOp = ConvertAvgPoolOp<TF::AvgPoolOp, /*num_dims=*/4>;
using ConvertAvgPool3DOp = ConvertAvgPoolOp<TF::AvgPool3DOp, /*num_dims=*/5>;

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
    if (!FormatFromString(op.getDataFormat().str(), &data_format)) {
      return op.emitOpError("invalid data format");
    }
    // `out_grad` is the gradient that was propagated via backpropagation from
    // the output layer.
    Value out_grad = op.getGrad();
    auto out_grad_type =
        out_grad.getType().template dyn_cast<RankedTensorType>();
    if (!out_grad_type) {
      return failure();
    }
    Type element_type = out_grad_type.getElementType();
    DenseIntElementsAttr orig_input_shape_attr;
    if (!matchPattern(op.getOrigInputShape(),
                      m_Constant(&orig_input_shape_attr))) {
      return failure();
    }
    auto orig_input_shape_values = orig_input_shape_attr.getValues<int32_t>();
    DimVector orig_input_shape(orig_input_shape_values.begin(),
                               orig_input_shape_values.end());
    DimVector ksize, strides;
    GetI64ArrayAttrValues(op.getKsize(), &ksize);
    GetI64ArrayAttrValues(op.getStrides(), &strides);
    Value zero = GetScalarConstOfType(element_type, loc, 0, &rewriter);

    auto out_grad_divided = AvgPoolDivideByCount<OpTy, num_dims>(
        out_grad, orig_input_shape, ksize, strides, op, zero, rewriter);

    // Get same padding as for original input.
    PaddingArray orig_padding = GetReduceWindowPaddingAsArray<num_dims>(
        orig_input_shape, op.getKsize(), op.getStrides(), op.getPadding(),
        &rewriter);

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
          status.value();
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
        loc, tensorflow::GetTypeFromTFTensorShape(out_grad_shape, element_type),
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
        loc,
        tensorflow::GetTypeFromTFTensorShape(orig_input_shape,
                                             sum_element_type),
        /*operand=*/reduce_window_input,
        /*init_value=*/zero,
        /*window_dimensions=*/GetI64ElementsAttr(op.getKsize()),
        /*window_strides=*/ones,
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(),
        /*padding=*/DenseIntElementsAttr());
    BuildReduceBody<AddOp>(sum_element_type, &reduce_window_op.getBody(),
                           &rewriter);
    Value result = reduce_window_op.getResult(0);

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
//   %init = arith.constant dense<...> : tensor<i32>
//   %max_pool = "mhlo.reduce"(%inp, %init) ["mhlo.maximum"]
//               {window_dimensions = ..., window_strides = ... }
//
template <typename OpTy, int num_dims>
class ConvertMaxPoolOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Type element_type =
        op.getInput().getType().template cast<TensorType>().getElementType();
    if (!element_type.isSignlessIntOrFloat()) return failure();
    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.getPadding().str(), &padding).ok())
      return failure();
    if (padding == tensorflow::Padding::EXPLICIT) {
      return failure();
    }
    Location loc = op.getLoc();
    ConstantOp init = GetScalarLimitConstOfType(
        element_type, loc, hlo::kInfinityLowest, &rewriter);

    auto input_ty =
        op.getInput().getType().template dyn_cast<RankedTensorType>();
    if (!input_ty) return failure();
    DenseIntElementsAttr paddings_attr = GetReduceWindowPaddingAsAttr<num_dims>(
        input_ty.getShape(), op.getKsize(), op.getStrides(), op.getPadding(),
        &rewriter);
    auto reduce = rewriter.create<ReduceWindowOp>(
        loc, op.getType(), op.getInput(), init,
        GetI64ElementsAttr(op.getKsize()), GetI64ElementsAttr(op.getStrides()),
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(), paddings_attr);
    BuildReduceBody<MaxOp>(element_type, &reduce.getBody(), &rewriter);

    rewriter.replaceOp(op, reduce.getResult(0));
    return success();
  }
};

using ConvertMaxPool2DOp = ConvertMaxPoolOp<TF::MaxPoolOp, /*num_dims=*/4>;
using ConvertMaxPool3DOp = ConvertMaxPoolOp<TF::MaxPool3DOp, /*num_dims=*/5>;

// Converts tf.Select (SelectV1) to mhlo.select. It has optional broadcasting on
// the condition only.
class ConvertSelectOp : public OpRewritePattern<TF::SelectOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SelectOp op,
                                PatternRewriter &rewriter) const override {
    // This lowering only works on ranked types.
    auto cond_type = op.getCondition().getType().dyn_cast<RankedTensorType>();
    auto then_type = op.getThenValue().getType().dyn_cast<RankedTensorType>();
    auto else_type = op.getElseValue().getType().dyn_cast<RankedTensorType>();
    if (!cond_type || !then_type || !else_type) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value cond_shape = b.createOrFold<shape::ShapeOfOp>(op.getCondition());
    Value then_shape = b.createOrFold<shape::ShapeOfOp>(op.getThenValue());
    Value else_shape = b.createOrFold<shape::ShapeOfOp>(op.getElseValue());

    // First check that the `then` and `else` shapes are the equal.
    Value assumption =
        b.createOrFold<shape::CstrEqOp>(ValueRange{then_shape, else_shape});
    // For a vector cond we also verify that the majormost dim of `then` matches
    // the vector size. To do that split off the first dim of `then`.
    bool needs_broadcast = cond_type.getRank() == 1 && then_type.getRank() != 1;
    Value then_shape_split = then_shape;
    if (needs_broadcast) {
      Value const_one = b.create<arith::ConstantIndexOp>(1);
      Type extent_first = shape::getExtentTensorType(b.getContext(), 1);
      Type extent_second =
          shape::getExtentTensorType(b.getContext(), then_type.getRank() - 1);
      SmallVector<Value, 2> then_split;
      b.createOrFold<shape::SplitAtOp>(then_split,
                                       TypeRange{extent_first, extent_second},
                                       then_shape, const_one);
      then_shape_split = then_split[0];
    }
    // If the condition is not a scalar, check that it matches the other shapes.
    if (cond_type.getRank() > 0) {
      Value eq_cstr = b.createOrFold<shape::CstrEqOp>(
          ValueRange{cond_shape, then_shape_split});
      auto witness = shape::WitnessType::get(b.getContext());
      assumption = b.createOrFold<shape::AssumingAllOp>(
          witness, ValueRange{assumption, eq_cstr});
    }
    auto result_type = op.getResult().getType().cast<TensorType>();
    auto assuming_op =
        b.create<shape::AssumingOp>(ArrayRef<Type>{result_type}, assumption);

    OpBuilder::InsertionGuard guard(b);
    b.createBlock(&assuming_op.getDoRegion());

    // Broadcast the cond if necessary.
    Value cond = op.getCondition();
    if (needs_broadcast) {
      Value result_extents = b.create<shape::ToExtentTensorOp>(
          GetExtentsTensorTypeFor(result_type), then_shape);
      cond = b.create<mhlo::DynamicBroadcastInDimOp>(
          tensorflow::GetTypeFromTFTensorShape(result_type.getShape(),
                                               b.getI1Type()),
          cond, result_extents,
          GetI64ElementsAttrForSeq(0, cond_type.getRank(), &b));
    }
    Value select = b.create<mhlo::SelectOp>(
        result_type, cond, op.getThenValue(), op.getElseValue());
    b.create<shape::AssumingYieldOp>(select);
    rewriter.replaceOp(op, {assuming_op.getResult(0)});
    return success();
  }
};

// Converts Sigmoid op to HLO ops computing sigmoid with the following formula:
//
//     sigmoid = add(mul(tanh(mul(logits, 0.5)), 0.5), 0.5)
//
// Sample result with 2-d f16 inputs with B batches of with N elements each.
//
//    // Create an array of 0.5 the shape of the input array.
//    %half = mhlo.constant dense<5.000000e-01> : tensor<f32>
//    %half_array = "mhlo.broadcast"(half)
//                           {broadcast_sizes = dense<2> : tensor<1xi64>}
//                           : (tensor<f32>) -> tensor<2xf32>
//
//    // Compute Tanh of half the logits of the values.
//    %halved_logits = mhlo.multiply %logits, %half_array : tensor<2xf32>
//    %tanh = "mhlo.tanh"(%halved_logits) : (tensor<2xf32>) -> tensor<2xf32>
//
//    // Have the result of Tanh and add 0.5.
//    %halved_tanh = mhlo.multiply %tanh, %half : tensor<2xf32>
//    %sigmoid = mhlo.add %halved_tanh, %half : tensor<2xf32>
//
class ConvertSigmoidOp : public RewritePattern {
 public:
  explicit ConvertSigmoidOp(MLIRContext *context)
      : RewritePattern(
            TF::SigmoidOp::getOperationName(), 0, context,
            {mhlo::ConstantOp::getOperationName(),
             shape::ShapeOfOp::getOperationName(),
             shape::ToExtentTensorOp::getOperationName(),
             mhlo::DynamicBroadcastInDimOp::getOperationName(),
             mhlo::MulOp::getOperationName(), mhlo::TanhOp::getOperationName(),
             mhlo::AddOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *sigmoid_op,
                                PatternRewriter &rewriter) const override {
    auto op = cast<TF::SigmoidOp>(sigmoid_op);
    Location loc = op.getLoc();

    // Create constant half with shape and element type same as the operand.
    Value operand = op.getOperand();
    auto operand_ty = operand.getType().cast<TensorType>();
    auto scalar_ty =
        tensorflow::GetTypeFromTFTensorShape({}, operand_ty.getElementType());
    ElementsAttr attr = mlir::hlo::getSplat(&rewriter, scalar_ty, 0.5);
    auto scalar_half = rewriter.create<ConstantOp>(loc, attr);
    auto half = BroadcastToShapeOf(loc, scalar_half, operand, rewriter);

    auto scaled_input = rewriter.create<MulOp>(loc, operand, half);
    auto tanh_op = rewriter.create<TanhOp>(loc, scaled_input);
    auto mul_op = rewriter.create<MulOp>(loc, tanh_op, half);
    auto add_op = rewriter.create<AddOp>(loc, mul_op, half);

    rewriter.replaceOp(op, add_op.getResult());
    return success();
  }
};

// Converts the tf.Slice op into mhlo.real_dynamic_slice
// TODO(disc): To recover static special case's performance with folding and
// canonicalization.
class ConvertSliceOpDynamic : public OpRewritePattern<TF::SliceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SliceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value begin_indices = op.getBegin();
    Value sizes = op.getSize();

    auto input_ty = input.getType().dyn_cast<RankedTensorType>();
    auto begin_type = begin_indices.getType().dyn_cast<RankedTensorType>();
    auto size_type = sizes.getType().dyn_cast<RankedTensorType>();

    if (!input_ty || !begin_type || !size_type ||
        !begin_type.hasStaticShape() || !size_type.hasStaticShape() ||
        begin_type.getRank() != 1 || size_type.getRank() != 1) {
      return failure();
    }
    // TODO(disc): remove static shape check once folding/canonicalization func
    // added
    DenseIntElementsAttr size_attr;
    if (matchPattern(op.getSize(), m_Constant(&size_attr))) {
      return failure();
    }

    int rank = begin_type.getDimSize(0);
    auto shape_scalar_type = begin_type.getElementType();
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value, 4> stride_values(rank, one);
    SmallVector<Value, 4> end_values;
    SmallVector<Value, 4> begin_values;
    end_values.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      SmallVector<Value, 4> indices;
      indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
      auto begin_value =
          rewriter.create<tensor::ExtractOp>(loc, begin_indices, indices);
      auto size_value = rewriter.create<tensor::ExtractOp>(loc, sizes, indices);
      Value minus_one = rewriter.create<arith::IndexCastOp>(
          loc, shape_scalar_type,
          rewriter.create<arith::ConstantIndexOp>(loc, -1));
      auto is_minus_one = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, size_value, minus_one);
      Value end_value =
          rewriter.create<arith::AddIOp>(loc, begin_value, size_value);
      auto dim_value = rewriter.create<arith::IndexCastOp>(
          loc, shape_scalar_type,
          rewriter.create<tensor::DimOp>(loc, input, i));
      end_value = rewriter.create<mlir::arith::SelectOp>(loc, is_minus_one,
                                                         dim_value, end_value);
      auto end_value_casted = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), end_value);
      end_values.push_back(end_value_casted);

      auto begin_value_casted = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), begin_value);
      begin_values.push_back(begin_value_casted);
    }
    auto index_ty = rewriter.getIndexType();
    auto start_indices = rewriter.create<tensor::FromElementsOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape(
            {static_cast<int64_t>(begin_values.size())}, index_ty),
        begin_values);
    auto end_indices = rewriter.create<tensor::FromElementsOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape(
            {static_cast<int64_t>(end_values.size())}, index_ty),
        end_values);
    auto stride_indices = rewriter.create<tensor::FromElementsOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape(
            {static_cast<int64_t>(stride_values.size())}, index_ty),
        stride_values);

    auto d_slice = rewriter.create<mhlo::RealDynamicSliceOp>(
        loc, op.getOperation()->getResult(0).getType(), input, start_indices,
        end_indices, stride_indices);
    rewriter.replaceOp(op, d_slice.getOperation()->getResults());
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
      rewriter->create<arith::ConstantOp>(loc, rewriter->getIndexAttr(-2));
  auto shape_type = shape::ShapeType::get(rewriter->getContext());
  auto lhs_splitted = rewriter->create<shape::SplitAtOp>(
      loc, TypeRange{shape_type, shape_type}, lhs_shape, const_neg2);
  auto rhs_splitted = rewriter->create<shape::SplitAtOp>(
      loc, TypeRange{shape_type, shape_type}, rhs_shape, const_neg2);
  auto lhs_type = lhs.getType().cast<RankedTensorType>();
  auto rhs_type = rhs.getType().cast<RankedTensorType>();
  // The last two dimensions are the matrix row/col dimensions. Don't broadcast
  // them.
  SmallVector<int64_t, 6> result_batch_shape_compile_time_extents;
  mlir::OpTrait::util::getBroadcastedShape(
      lhs_type.getShape().drop_back(2), rhs_type.getShape().drop_back(2),
      result_batch_shape_compile_time_extents);
  auto result_batch_shape = rewriter->create<shape::BroadcastOp>(
      loc, shape_type, lhs_splitted.getHead(), rhs_splitted.getHead(),
      /*error=*/nullptr);
  // Lambda which handles the broadcasting of one side to the common
  // leading-batch dimensions.
  auto broadcast_one_side = [&](Value side, RankedTensorType type,
                                Value tail_shape, Value *out_side) {
    ArrayRef<int64_t> matrix_dims = type.getShape().take_back(2);
    auto result_shape = result_batch_shape_compile_time_extents;
    result_shape.append(matrix_dims.begin(), matrix_dims.end());
    auto result_type = tensorflow::GetTypeFromTFTensorShape(
        result_shape, type.getElementType());
    auto shape = rewriter->create<shape::ConcatOp>(
        loc, shape_type, result_batch_shape, tail_shape);
    auto shape_tensor = rewriter->create<shape::ToExtentTensorOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape(
            {static_cast<int64_t>(result_shape.size())},
            rewriter->getIndexType()),
        shape);
    *out_side = rewriter->create<TF::BroadcastToOp>(loc, result_type, side,
                                                    shape_tensor);
  };
  broadcast_one_side(lhs, lhs_type, lhs_splitted.getTail(), out_lhs);
  broadcast_one_side(rhs, rhs_type, rhs_splitted.getTail(), out_rhs);
}

class ConvertBatchMatMulV2Op : public OpRewritePattern<TF::BatchMatMulV2Op> {
 public:
  // TODO(hinsu): Legalize this op to Einsum op. HLO Einsum op needs to be moved
  // to CHLO and it is missing legalization to MHLO. Once that is done, this
  // pattern's benefit can be changed back to one as well as the fallback
  // lowering pattern for the op can be removed.
  //
  // Set benefit of this pattern to zero to prefer the fallback pattern when
  // available and applicable. That pattern avoids broadcast on operands and is
  // therefore faster.
  //
  // Native legalization for BatchMatMulV3 needs to be added as well.
  explicit ConvertBatchMatMulV2Op(MLIRContext *context)
      : OpRewritePattern<TF::BatchMatMulV2Op>(context, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(TF::BatchMatMulV2Op op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getX();
    Value rhs = op.getY();
    auto lhs_type = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhs_type = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhs_type || !rhs_type) return failure();
    if (lhs_type.getElementType().isa<ComplexType>() && op.getAdjX()) {
      lhs = rewriter.create<TF::ConjOp>(op.getLoc(), lhs_type, lhs);
    }
    if (rhs_type.getElementType().isa<ComplexType>() && op.getAdjY()) {
      rhs = rewriter.create<TF::ConjOp>(op.getLoc(), rhs_type, rhs);
    }

    // Broadcast both operands.
    BroadcastBatchMatMulV2Operands(lhs, rhs, op.getLoc(), &lhs, &rhs,
                                   &rewriter);
    lhs_type = lhs.getType().cast<RankedTensorType>();
    rhs_type = rhs.getType().cast<RankedTensorType>();
    assert(lhs_type.getRank() == rhs_type.getRank());
    int64_t rank = lhs_type.getRank();
    auto batch_dimensions = llvm::to_vector<4>(llvm::seq<int64_t>(0, rank - 2));
    auto lhs_contracting_dimensions = llvm::to_vector<4>(
        llvm::ArrayRef({op.getAdjX() ? rank - 2 : rank - 1}));
    auto rhs_contracting_dimensions = llvm::to_vector<4>(
        llvm::ArrayRef({op.getAdjY() ? rank - 1 : rank - 2}));
    auto dimension_numbers = DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*lhs_batching_dimensions=*/batch_dimensions,
        /*rhs_batching_dimensions=*/batch_dimensions,
        /*lhs_contracting_dimensions=*/lhs_contracting_dimensions,
        /*rhs_contracting_dimensions=*/rhs_contracting_dimensions);
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
//   %0 = "mhlo.slice"(%input) {
//             limit_indices = dense<[4, 2]> : tensor<2xi64>,
//             start_indices = dense<0> : tensor<2xi64>,
//             strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
//   %1 = "mhlo.slice"(%input) {
//             limit_indices = dense<4> : tensor<2xi64>,
//              start_indices = dense<[0, 2]> : tensor<2xi64>,
//            strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
//    %2 = "mhlo.slice"(%input) {
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
    // We can only split inputs that have fully static shape.
    auto input_type = op.getValue().getType().dyn_cast<RankedTensorType>();
    if (!input_type || !input_type.hasStaticShape()) return failure();

    // We can only match when the split dimension is a constant scalar.
    DenseIntElementsAttr split_dim_attr;
    if (!matchPattern(op.getSplitDim(), m_Constant(&split_dim_attr)))
      return failure();

    // Get the dimension we are splitting at. Offset properly if it's negative.
    int64_t input_rank = input_type.getRank();
    int64_t dim_index = (*split_dim_attr.begin()).getSExtValue();
    if (dim_index < 0) dim_index += input_rank;

    // Calculate the dimension size for each slice along the split dimension.
    int64_t input_dim_size = input_type.getDimSize(dim_index);

    int64_t num_splits = op.getNumResults();
    int64_t slice_size = input_dim_size / num_splits;

    // Get each slice's type.
    auto slice_shape = llvm::to_vector<4>(input_type.getShape());
    slice_shape[dim_index] = slice_size;
    Type slice_type = tensorflow::GetTypeFromTFTensorShape(
        slice_shape, input_type.getElementType());

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
          rewriter.create<SliceOp>(op.getLoc(), slice_type, op.getValue(),
                                   GetI64ElementsAttr(begin_indices, &rewriter),
                                   GetI64ElementsAttr(end_indices, &rewriter),
                                   GetI64ElementsAttr(strides, &rewriter)));
    }

    rewriter.replaceOp(op, slices);
    return success();
  }
};

// Converts the tf.Split op into a series of mhlo.real_dynamic_slice ops the
// dimension to split is a constant.
// TODO(disc): To recover static special case's performance with folding and
// canonicalization. delete ConvertSplitOp
class ConvertSplitOpDynamic : public OpRewritePattern<TF::SplitOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SplitOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getValue();
    auto input_type = input.getType().dyn_cast<RankedTensorType>();
    if (!input_type) return failure();

    // TODO(disc): remove static shape check once folding/canonicalization func
    // added and ConvertSplitOp deleted. Calculate the dimension size for each
    // slice along the split dimension. We are splitting along the dynamic
    // dimension, or using static pattern transform
    if (input_type.hasStaticShape()) return failure();

    // We can only match when the split dimension is a constant scalar.
    DenseIntElementsAttr split_dim_attr;
    if (!matchPattern(op.getSplitDim(), m_Constant(&split_dim_attr)))
      return failure();

    // Get the dimension we are splitting at. Offset properly if it's negative.
    int64_t input_rank = input_type.getRank();
    int64_t dim_index = (*split_dim_attr.begin()).getSExtValue();
    if (dim_index < 0) dim_index += input_rank;

    Value input_dim_size =
        rewriter.create<tensor::DimOp>(loc, input, dim_index);
    // Calculate the dimension size for each slice along the split dimension.
    int num_splits = op.getNumResults();
    Value num_splits_value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(num_splits));
    Value slice_size =
        rewriter.create<arith::DivSIOp>(loc, input_dim_size, num_splits_value);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value, 4> begin_indices(input_rank, zero);
    SmallVector<Value, 4> end_indices;
    end_indices.reserve(input_rank);
    SmallVector<Value, 4> strides(input_rank, one);
    for (int i = 0; i < input_rank; ++i) {
      end_indices.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
    }

    // All HLO d_slice results used to replace the original tf.Split op.
    SmallVector<Value, 4> slices;
    slices.reserve(num_splits);

    for (int i = 0; i < num_splits; ++i) {
      begin_indices[dim_index] = rewriter.create<arith::MulIOp>(
          loc, slice_size, rewriter.create<arith::ConstantIndexOp>(loc, i));
      end_indices[dim_index] = rewriter.create<arith::MulIOp>(
          loc, slice_size, rewriter.create<arith::ConstantIndexOp>(loc, i + 1));

      Type index_ty = rewriter.getIndexType();
      auto begin_value = rewriter.create<tensor::FromElementsOp>(
          loc,
          tensorflow::GetTypeFromTFTensorShape(
              {static_cast<int64_t>(begin_indices.size())}, index_ty),
          begin_indices);
      auto end_value = rewriter.create<tensor::FromElementsOp>(
          loc,
          tensorflow::GetTypeFromTFTensorShape(
              {static_cast<int64_t>(end_indices.size())}, index_ty),
          end_indices);
      auto stride_value = rewriter.create<tensor::FromElementsOp>(
          loc,
          tensorflow::GetTypeFromTFTensorShape(
              {static_cast<int64_t>(strides.size())}, index_ty),
          strides);
      slices.push_back(rewriter.create<RealDynamicSliceOp>(
          loc, op.getOperation()->getResult(i).getType(), input, begin_value,
          end_value, stride_value));
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
// %0 = "mhlo.slice"(%input) {
//        limit_indices = dense<[4, 1]> : tensor<2xi64>,
//        start_indices = dense<0> : tensor<2xi64>,
//        strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x1xf32>
// %1 = "mhlo.slice"(%input) {
//        limit_indices = dense<[4, 3]> : tensor<2xi64>,
//        start_indices = dense<[0, 1]> : tensor<2xi64>,
//        strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
// %2 = "mhlo.slice"(%input) {
//        limit_indices = dense<[4, 6]> : tensor<2xi64>,
//        start_indices = dense<[0, 3]> : tensor<2xi64>,
//        strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x3xf32>
class ConvertSplitVOp : public OpRewritePattern<TF::SplitVOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SplitVOp op,
                                PatternRewriter &rewriter) const override {
    // We can only split inputs that have fully static shape.
    // TODO(b/145731001): enhance to support dynamic-shaped inputs.
    auto input_type = op.getValue().getType().dyn_cast<RankedTensorType>();
    if (!input_type || !input_type.hasStaticShape()) return failure();

    // We can only match when the split dimension is a constant scalar.
    DenseIntElementsAttr split_dim_attr;
    if (!matchPattern(op.getSplitDim(), m_Constant(&split_dim_attr)))
      return failure();

    // We can only match when the split sizes is a constant int vector.
    DenseIntElementsAttr split_sizes_attr;
    if (!matchPattern(op.getSizeSplits(), m_Constant(&split_sizes_attr)))
      return failure();

    // Get each chunck's size along the dimension to split. It may contain
    // dynamic sizes and we need to update it if so.
    SmallVector<int64_t, 4> split_sizes;
    int64_t total_dim_size = 0;  // Total dimension size assigned to splits
    std::optional<int> dynamic_dim_index;
    split_sizes.reserve(
        split_sizes_attr.getType().cast<ShapedType>().getNumElements());
    for (auto &dim : llvm::enumerate(split_sizes_attr)) {
      int64_t dim_val = dim.value().getSExtValue();
      split_sizes.push_back(dim_val);
      if (dim_val == -1) {
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

    for (int i = 0, end = op.getNumResults(); i < end; ++i) {
      end_indices[dim_index] = begin_indices[dim_index] + split_sizes[i];
      slices.push_back(rewriter.create<mhlo::SliceOp>(
          op.getLoc(), op.getValue(),
          GetI64ElementsAttr(begin_indices, &rewriter),
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
// be sliced and the slice needs to be lowered to mhlo.DynamicSlice. In this
// case, strides must have a known value of 1 (otherwise we have insufficient
// information to conform to XLA's op semantics).
//
// For example with an op like following,
//   tf.StridedSlice(%input, %begin, %end, %strides) {shrink_axis_mask = 1}
//     : tensor<AxBxf32> -> tensor<Pxf32>
//
// If the %begin input is constant, output would be:
//   %reversed = "mhlo.Reverse" (%input) {dimensions = ...}
//   %sliced = "mhlo.Slice" (%input)
//             {start_indices = ..., limit_indices = ..., strides = ...}
//   %output = "mhlo.Reshape" (%sliced) : tensor<1xPxf32> -> tensor<Pxf32>
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
    Value input = op.getInput();
    if (!dims_to_reverse.empty())
      input = rewriter.create<ReverseOp>(
          loc, input_ty, op.getInput(),
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
    if (!matchPattern(op.getStrides(), m_Constant(&sparse_strides_attr))) {
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

    // When begin/end values are dynamic, the ellipsis mask, if set, must refer
    // to the last dimension.
    int ellipsis_mask = op.getEllipsisMask();
    if (!(ellipsis_mask == 0 || ellipsis_mask == (1 << last_dim)))
      return rewriter.notifyMatchFailure(
          op,
          "requires that ellipsis_mask, if set, refer to the last dimension of "
          "input (when begin/end values are dynamic)");

    // In this case where the begin and end values are dynamic, we only support
    // cases where the number of output elements has to be equal to the number
    // of input elements that are sliced. Each dimension is either sliced fully
    // or sliced with a size of one.
    int output_elements = result_ty.getNumElements();
    int input_elements_sliced = 1;

    // Begin must be a ranked, 1-dimensional tensor: This is checked by the
    // verifier.
    int64_t slicing_dim_size =
        op.getBegin().getType().cast<RankedTensorType>().getDimSize(0);
    uint64_t begin_mask = op.getBeginMask();
    uint64_t end_mask = op.getEndMask();
    const int input_rank = input_shape.size();
    for (int d = 0; d < input_rank; ++d) {
      // Each dimension is either sliced fully or has size of one.
      if ((((begin_mask >> d) & 1) && ((end_mask >> d) & 1)) ||
          (d >= slicing_dim_size)) {
        input_elements_sliced *= input_shape[d];
      }
    }
    if (input_elements_sliced != output_elements) {
      return rewriter.notifyMatchFailure(
          op,
          "requires the number of output elements to be equal to the number of "
          "input elements sliced (when begin/end values are dynamic)");
    }

    SmallVector<Value, 4> slice_begin_indices;
    // For the dimensions that are to be sliced, all have slice sizes of 1.
    SmallVector<int64_t, 4> slice_sizes;
    auto begin_element_ty =
        op.getBegin().getType().cast<ShapedType>().getElementType();
    // Scalar tensor type.
    TensorType type =
        tensorflow::GetTypeFromTFTensorShape(/*shape=*/{}, begin_element_ty);
    Location loc = op.getLoc();
    auto zero = GetScalarConstOfType(begin_element_ty, loc, 0, &rewriter);
    for (int d = 0; d < input_rank; ++d) {
      if ((((begin_mask >> d) & 1) && ((end_mask >> d) & 1)) ||
          (d >= slicing_dim_size)) {
        slice_begin_indices.push_back(zero);
        slice_sizes.push_back(input_shape[d]);
        continue;
      }

      auto index = rewriter.create<SliceOp>(
          loc, op.getBegin(), GetI64ElementsAttr({d}, &rewriter),
          GetI64ElementsAttr({d + 1}, &rewriter),
          GetI64ElementsAttr({1}, &rewriter));
      // Convert index to scalar.
      auto reshaped_index = rewriter.create<ReshapeOp>(loc, type, index);
      // If the index is negative, wrap it around with dimension size.
      auto index_negative =
          rewriter.create<TF::LessOp>(loc, reshaped_index, zero);
      auto input_val = GetScalarConstOfType(begin_element_ty, loc,
                                            input_shape[d], &rewriter);
      auto wrapped_index =
          rewriter.create<TF::AddV2Op>(loc, input_val, reshaped_index);
      auto final_index = rewriter.create<SelectOp>(
          loc, type, index_negative, wrapped_index, reshaped_index);
      slice_begin_indices.push_back(final_index);
      slice_sizes.push_back(1);
    }

    auto slice_sizes_attr = GetI64ElementsAttr(slice_sizes, &rewriter);
    auto sliced_type = tensorflow::GetTypeFromTFTensorShape(
        slice_sizes, op.getType().getElementType());
    // This must be an xla DynamicSlice op due to the inputs that aren't
    // constant.
    auto sliced = rewriter.create<DynamicSliceOp>(
        loc, sliced_type, op.getInput(), slice_begin_indices, slice_sizes_attr);

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
    auto input_ty = op.getInput().getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) return failure();

    // Output shape needs to be static to apply 'new_axis_mask' or
    // 'shrink_axis_mask' by reshaping tensor after slice.
    //
    // TODO(hinsu): Relax this constraint for ops without the above masks.
    auto result_ty = op.getType().dyn_cast<RankedTensorType>();
    if (!result_ty || !result_ty.hasStaticShape()) return failure();

    DenseIntElementsAttr sparse_begin_attr, sparse_end_attr;
    if (!matchPattern(op.getBegin(), m_Constant(&sparse_begin_attr)) ||
        !matchPattern(op.getEnd(), m_Constant(&sparse_end_attr))) {
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
    if (!matchPattern(op.getShape(), m_Constant(&input_shape_attr)))
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

    Value grad = op.getDy();
    Type element_type = grad.getType().cast<ShapedType>().getElementType();

    // Perform reshape to undo any new/shrink axes done by strided slice.
    grad = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), tensorflow::GetTypeFromTFTensorShape(shape, element_type),
        grad);

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
      grad = rewriter.create<mhlo::ReverseOp>(
          op.getLoc(), grad.getType(), grad,
          GetI64ElementsAttr(dims_to_reverse, &rewriter));
    }

    auto zero = GetScalarConstOfType(element_type, op.getLoc(), 0, &rewriter);
    rewriter.replaceOpWithNewOp<mhlo::PadOp>(
        op, op.getType(), grad, zero,
        GetI64ElementsAttr(padding_low, &rewriter),
        GetI64ElementsAttr(padding_high, &rewriter),
        GetI64ElementsAttr(padding_interm, &rewriter));
    return success();
  }
};

/// Converts the RangeOp tensorflow op to a mhlo.iota op with a scaling and
/// offset applied to generate the range values. The output tensor needs to
/// have a static shape.
///
/// For example an op like the following:
///   %result = "tf.Range"(%start, %limit, %delta) {Tidx = "tfdtype$DT_FLOAT"}
///      : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<5xf32>
///
/// Output would be:
///   %iota = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5xf32>
///   %scaled = "mhlo.multiply"(%iota, %delta)
///       {broadcast_dimensions = dense<[]> : tensor<0xi64>} :
///       (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
///   %result = "mhlo.add"(%scaled, %offset)
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
    auto scaled = rewriter.create<chlo::BroadcastMulOp>(
        op.getLoc(), result_type, iota, op.getDelta(),
        hlo::getBroadcastDimensionsAttr(&rewriter, iota, op.getDelta()));
    rewriter.replaceOpWithNewOp<chlo::BroadcastAddOp>(
        op, result_type, scaled, op.getStart(),
        hlo::getBroadcastDimensionsAttr(&rewriter, scaled, op.getStart()));
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

    Value start = op.getStart();
    Value delta = op.getDelta();
    Value limit = op.getLimit();

    // To compute the length we need to use floating point calculations so that
    // ceil can be computed for the number of steps.
    auto compute_element_type =
        getElementTypeOrSelf(start.getType()).isa<FloatType>()
            ? getElementTypeOrSelf(start.getType())
            : rewriter.getF64Type();
    auto compute_type = tensorflow::GetTypeFromTFTensorShape(
        limit.getType().cast<ShapedType>().getShape(), compute_element_type);

    // Compute the length of the sequence we are going to need. This includes
    // some conversion to float for the operations.
    //
    // %size = ceil(abs((%limit - %start) / %delta))
    auto range = rewriter.create<mhlo::SubtractOp>(op.getLoc(), limit, start);
    auto abs = rewriter.create<mhlo::AbsOp>(op.getLoc(), range);

    // Delta is not necessarily the same type as start and limit.
    auto abs_cast =
        rewriter.create<mhlo::ConvertOp>(op.getLoc(), compute_type, abs);
    auto delta_cast =
        rewriter.create<mhlo::ConvertOp>(op.getLoc(), compute_type, delta);

    // Compute the total number of integer steps and convert to the HLO
    // dimension tensor.
    auto normalized =
        rewriter.create<mhlo::DivOp>(op.getLoc(), abs_cast, delta_cast);
    auto ceil = rewriter.create<mhlo::CeilOp>(op.getLoc(), normalized);
    auto steps = rewriter.create<mhlo::ConvertOp>(
        op.getLoc(),
        tensorflow::GetTypeFromTFTensorShape({}, rewriter.getI64Type()), ceil);
    auto reshape = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(),
        tensorflow::GetTypeFromTFTensorShape({1}, rewriter.getI64Type()),
        steps);

    // Using the resulting length compute the correct range value:
    //
    // %range = %start + %delta * iota(%size)
    auto out_scalar_type = tensorflow::GetTypeFromTFTensorShape(
        {}, getElementTypeOrSelf(result_type));
    auto start_out_cast =
        rewriter.create<mhlo::ConvertOp>(op.getLoc(), out_scalar_type, start);
    auto delta_out_cast =
        rewriter.create<mhlo::ConvertOp>(op.getLoc(), out_scalar_type, delta);

    auto iota = rewriter.create<DynamicIotaOp>(
        op.getLoc(), result_type, reshape, rewriter.getI64IntegerAttr(0));
    auto scaled = rewriter.create<chlo::BroadcastMulOp>(
        op.getLoc(), result_type, iota, delta_out_cast,
        hlo::getBroadcastDimensionsAttr(&rewriter, iota, delta_cast));
    rewriter.replaceOpWithNewOp<chlo::BroadcastAddOp>(
        op, result_type, scaled, start_out_cast,
        hlo::getBroadcastDimensionsAttr(&rewriter, scaled, start_out_cast));
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

/// Converts the LinSpace tensorflow op to a mhlo.iota op with a scaling
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
    if (!matchPattern(op.getNum(), m_Constant(&num_attr))) {
      return rewriter.notifyMatchFailure(op, "Num must be a constant scalar");
    }

    if (num_attr.begin() == num_attr.end()) {
      return rewriter.notifyMatchFailure(op, "Num must not be empty");
    }
    int64_t num = (*num_attr.begin()).getSExtValue();

    // Calculate the scaling that needs to be applied to the iota.
    auto step_numerator = rewriter.create<chlo::BroadcastSubOp>(
        op.getLoc(), op.getStart().getType(), op.getStop(), op.getStart(),
        hlo::getBroadcastDimensionsAttr(&rewriter, op.getStop(),
                                        op.getStart()));
    Value step_denominator = rewriter.create<ConvertOp>(
        op.getLoc(), op.getNum(), result_type.getElementType());
    if (num > 1) {
      Value one = GetScalarConstOfType(result_type.getElementType(),
                                       op.getLoc(), 1, &rewriter);
      step_denominator = rewriter.create<chlo::BroadcastSubOp>(
          op.getLoc(), step_denominator.getType(), step_denominator, one,
          hlo::getBroadcastDimensionsAttr(&rewriter, step_denominator, one));
    }
    auto step = rewriter.create<chlo::BroadcastDivOp>(
        op.getLoc(), step_numerator.getType(), step_numerator, step_denominator,
        hlo::getBroadcastDimensionsAttr(&rewriter, step_numerator,
                                        step_denominator));

    // Scale the iota and add the offset.
    auto iota = rewriter.create<IotaOp>(op.getLoc(), result_type,
                                        rewriter.getI64IntegerAttr(0));
    auto scaled = rewriter.create<chlo::BroadcastMulOp>(
        op.getLoc(), result_type, iota, step,
        hlo::getBroadcastDimensionsAttr(&rewriter, iota, step));
    rewriter.replaceOpWithNewOp<chlo::BroadcastAddOp>(
        op, result_type, scaled, op.getStart(),
        hlo::getBroadcastDimensionsAttr(&rewriter, scaled, op.getStart()));
    return success();
  }
};

/// Converts a generic OpTy tensorflow op to a mhlo.reduce op over
/// ReductionOp.
/// `is_accumulation` controls whether it uses higher precision for the actual
/// reduction. This is set to false for ops like max where there is no precision
/// concerns.
//
// The Derived class should have a static method to return the initial value to
// use for reduction:
//   static Value GetInitialValue(Type reduce_element_type, Location loc,
//                                PatternRewriter *rewriter);
// The reduce_element_type is guaranteed to be a float, int, or complex type
// suitable for use with GetScalarConstOfType or GetScalarLimitConstOfType.
template <typename Derived, typename OpTy, typename ReductionOp,
          bool is_accumulation = true>
class GenericConvertReductionOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // TODO(b/141785544): Update this to not require ranked shapes.
    // Input shape needs to be ranked to convert negative indices in TensorFlow
    // to absolute indices required by HLO.
    auto input_ty =
        op.getInput().getType().template dyn_cast<RankedTensorType>();
    if (!input_ty) return failure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();

    DenseIntElementsAttr dimensions;
    if (!matchPattern(op.getReductionIndices(), m_Constant(&dimensions)))
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

    // Only float, int, and complex types are currently supported.
    if (!element_type.isa<FloatType>() && !element_type.isa<IntegerType>() &&
        !element_type.isa<ComplexType>()) {
      return rewriter.notifyMatchFailure(
          op, "element type must be float, int, or complex type");
    }

    // Convert to an accumulation type to not lose precision when doing
    // repeated arithmetic operations.
    Type reduce_element_type =
        is_accumulation ? GetAccumulationType(element_type) : element_type;
    auto casted_input =
        rewriter.create<ConvertOp>(loc, op.getInput(), reduce_element_type);

    // Each reduction op can have a different initial value.
    Value init = Derived::GetInitialValue(reduce_element_type, loc, &rewriter);

    auto reduction = rewriter.create<ReduceOp>(
        loc, casted_input.getResult(), init,
        GetI64ElementsAttr(xla_dimensions, &rewriter));
    BuildReduceBody<ReductionOp>(reduce_element_type, &reduction.getBody(),
                                 &rewriter);
    Value result = reduction.getResult(0);

    // The mean op needs to divide by the product of the reduced dimensions.
    if (std::is_same<OpTy, TF::MeanOp>::value) {
      Value in_shape = rewriter.create<shape::ShapeOfOp>(loc, op.getInput());
      Value divisor_count = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      for (size_t i = 0; i < input_shape.size(); ++i) {
        if (reduced_dimensions_bitmap[i]) {
          Value index = rewriter.create<arith::ConstantIndexOp>(loc, i);
          auto dim = rewriter.create<tensor::ExtractOp>(loc, in_shape, index);
          divisor_count =
              rewriter.create<arith::MulIOp>(loc, divisor_count, dim);
        }
      }
      // HLO ops are only defined on tensors, so we cast the divisor from
      // index -> i64 -> tensor<1xi64> -> tensor<i64> -> tensor<reduction type>
      Value divisor_casted = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(), divisor_count);
      Value divisor_tensor = rewriter.create<tensor::FromElementsOp>(
          loc, tensorflow::GetTypeFromTFTensorShape({}, rewriter.getI64Type()),
          divisor_casted);
      Value divisor = rewriter.create<ConvertOp>(
          loc, tensorflow::GetTypeFromTFTensorShape({}, reduce_element_type),
          divisor_tensor);
      auto broadcast_dims = GetI64ElementsAttr({}, &rewriter);
      result = rewriter.create<chlo::BroadcastDivOp>(loc, result, divisor,
                                                     broadcast_dims);
    }

    result = rewriter.create<ConvertOp>(loc, result, element_type);

    // Need to reshape back after the reduction if we're keeping the reduced
    // dimensions. Note that we do this through successive (nominally 1)
    // applications of the TF ExpandDims op vs a more labor intensive
    // reshape. Various code generation techniques benefit from the knowledge
    // that this is a restricted form of shape manipulation that is just adding
    // unit dims.
    if (op.getKeepDims()) {
      for (auto &dim_is_reduced : llvm::enumerate(reduced_dimensions_bitmap)) {
        if (dim_is_reduced.value()) {
          auto index_attr = GetI32ElementsAttr(
              {static_cast<int>(dim_is_reduced.index())}, &rewriter);
          Value index = rewriter.create<arith::ConstantOp>(loc, index_attr);
          result = rewriter.create<TF::ExpandDimsOp>(loc, result, index);
        }
      }
    }
    rewriter.replaceOp(op, {result});

    return success();
  }
};

// Converts Mean op to HLO Reduce op.
//
//   %init = arith.constant dense<...> : tensor<T>
//   %sum = "mhlo.reduce"(%inp, %init) ["mhlo.add"]
//               {dimensions = ...}
//   %divisor = arith.constant dense<...> : tensor<T>
//   %mean = "mhlo.divide"(%sum, %divisor)
class ConvertMeanOp
    : public GenericConvertReductionOp<ConvertMeanOp, TF::MeanOp, AddOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;
  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarNegZeroOfType(reduce_element_type, loc, rewriter);
  }
};

// Converts Sum op to HLO Reduce op.
//
//   %init = arith.constant dense<...> : tensor<T>
//   %sum = "mhlo.reduce"(%inp, %init) ["mhlo.add"]
//               {dimensions = ...}
class ConvertSumOp
    : public GenericConvertReductionOp<ConvertSumOp, TF::SumOp, AddOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    // The neutral element of fp addition is -0.0, not 0.0: '0.0 + -0.0 = 0.0'.
    return GetScalarNegZeroOfType(reduce_element_type, loc, rewriter);
  }
};

// Converts Max op to HLO Reduce op.
//
//   %init = arith.constant dense<...> : tensor<T>
//   %max = "mhlo.reduce"(%inp, %init) ["mhlo.maximum"]
//               {dimensions = ...}
class ConvertMaxOp
    : public GenericConvertReductionOp<ConvertMaxOp, TF::MaxOp, MaxOp,
                                       /* is_accumulation= */ false> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarLimitConstOfType(reduce_element_type, loc,
                                     hlo::kInfinityLowest, rewriter);
  }
};

// Converts Min op to HLO Reduce op.
//
//   %init = arith.constant dense<...> : tensor<T>
//   %min = "mhlo.reduce"(%inp, %init) ["mhlo.minimum"]
//               {dimensions = ...}
class ConvertMinOp
    : public GenericConvertReductionOp<ConvertMinOp, TF::MinOp, MinOp,
                                       /* is_accumulation= */ false> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarLimitConstOfType(reduce_element_type, loc,
                                     hlo::kInfinityMax, rewriter);
  }
};

// Converts Prod op to HLO Reduce op.
//
//   %init = arith.constant dense<...> : tensor<T>
//   %prod = "mhlo.reduce"(%inp, %init) ["mhlo.multiply"]
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
//   %init = arith.constant dense<...> : tensor<T>
//   %max = "mhlo.reduce"(%inp, %init) ["mhlo.and"]
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
//   %init = arith.constant dense<...> : tensor<T>
//   %max = "mhlo.reduce"(%inp, %init) ["mhlo.or"]
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

// Converts tensorflow ArgMin or ArgMax op to mhlo operations that perform
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
        op.getInput().getType().template dyn_cast<RankedTensorType>();
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
        op.getOutput().getType().template dyn_cast<RankedTensorType>();
    if (!output_type) {
      return rewriter.notifyMatchFailure(op, "requires known rank");
    }

    Type index_element_type = output_type.getElementType();
    Value index_init_value =
        GetScalarConstOfType(index_element_type, loc, 0, &rewriter);

    RankedTensorType index_type = tensorflow::GetTypeFromTFTensorShape(
        input_type.getShape(), index_element_type);

    std::optional<int64_t> optional_axis =
        GetIntegerHLOAxisFromTFAxis(op.getDimension(), input_type.getRank());
    if (!optional_axis.has_value())
      return rewriter.notifyMatchFailure(op, "required axis");
    int64_t axis = optional_axis.value();

    IntegerAttr iota_dimension =
        IntegerAttr::get(rewriter.getIntegerType(64), axis);
    Value input_shape = rewriter.create<shape::ShapeOfOp>(loc, op.getInput());
    Value index_values = rewriter.create<DynamicIotaOp>(
        loc, index_type, input_shape, iota_dimension);

    Value operands[] = {op.getInput(), index_values};
    Value init_values[] = {init_value, index_init_value};
    DenseIntElementsAttr reduction_dimensions =
        GetI64ElementsAttr({axis}, &rewriter);

    auto reduction = rewriter.create<ReduceOp>(
        loc, llvm::ArrayRef<Value>(operands),
        llvm::ArrayRef<Value>(init_values), reduction_dimensions);
    auto direction = Derived::GetDirection();
    BuildArgMinMaxReductionBody(input_element_type, index_element_type,
                                direction, &reduction.getBody(), &rewriter);

    rewriter.replaceOp(op, {reduction.getResult(1)});
    return success();
  }
};

// Converts tensorflow ArgMax op to mhlo operations. The actual
// implementation is in class ConvertArgMinMaxOp:
//
//   %init_index = arith.constant dense<...> : tensor<T>
//   %init = arith.constant dense<...> : tensor<T>
//   %reduce = "mhlo.reduce"(%selected_input, %select_index, %init,
//                              %init_index) ["mhlo.arg_max"]
class ConvertArgMaxOp
    : public ConvertArgMinMaxOp<ConvertArgMaxOp, TF::ArgMaxOp> {
 public:
  using ConvertArgMinMaxOp::ConvertArgMinMaxOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter &rewriter) {
    return GetScalarLimitConstOfType(reduce_element_type, loc,
                                     hlo::kInfinityLowest, &rewriter);
  }

  static ComparisonDirection GetDirection() { return ComparisonDirection::GE; }
};

// Converts tensorflow ArgMin op to mhlo operations. The actual
// implementation is in class ConvertArgMinMaxOp:
//
//   %init_index = arith.constant dense<...> : tensor<T>
//   %init = arith.constant dense<...> : tensor<T>
//   %reduce = "mhlo.reduce"(%selected_input, %select_index, %init,
//                              %init_index) ["mhlo.arg_min"]
class ConvertArgMinOp
    : public ConvertArgMinMaxOp<ConvertArgMinOp, TF::ArgMinOp> {
 public:
  using ConvertArgMinMaxOp::ConvertArgMinMaxOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter &rewriter) {
    return GetScalarLimitConstOfType(reduce_element_type, loc,
                                     hlo::kInfinityMax, &rewriter);
  }

  static ComparisonDirection GetDirection() { return ComparisonDirection::LE; }
};

// Converts TF TensorScatterUpdate/Min/Max/Add/Sub op into Scatter Op with
// assignment:
//
//   %result = "mhlo.scatter"(%tensor, %indices, %updates)
//     { dimensions = ... }
//
template <typename Derived, typename OpTy>
class ConvertTensorScatterOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto tensor_ty =
        op.getTensor().getType().template dyn_cast<RankedTensorType>();
    auto indices_ty =
        op.getIndices().getType().template dyn_cast<RankedTensorType>();
    auto updates_ty =
        op.getUpdates().getType().template dyn_cast<RankedTensorType>();

    if (!tensor_ty || !indices_ty || !updates_ty) return failure();
    // Last dimension of the indices needs to known at compile time for
    // computation of the 'update_window_dims' attribute in the dimensions
    // struct.
    int64_t num_index_dims = indices_ty.getShape().back();
    if (ShapedType::isDynamic(num_index_dims)) return failure();

    auto updates = op.getUpdates();

    // Broadcast scalar `updates` in into expected shape as following shape:
    // updates.shape == indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
    if (updates_ty.getRank() == 0 &&
        (std::is_same<OpTy, TF::TensorScatterUpdateOp>::value ||
         std::is_same<OpTy, TF::TensorScatterAddOp>::value)) {
      if (!tensor_ty.hasStaticShape()) {
        return failure();
      }

      if (!indices_ty.hasStaticShape()) {
        return failure();
      }

      auto tensor_shape = tensor_ty.getShape();
      auto indices_shape = indices_ty.getShape();
      auto index_depth = indices_shape.back();
      llvm::SmallVector<int64_t> expected_update_shape;

      // create the expected update shape which scalar update is broadcasted to
      expected_update_shape.append(indices_shape.begin(),
                                   std::prev(indices_shape.end()));

      expected_update_shape.append(std::next(tensor_shape.begin(), index_depth),
                                   tensor_shape.end());

      auto const_type = tensorflow::GetTypeFromTFTensorShape(
          {static_cast<int>(expected_update_shape.size())},
          rewriter.getIntegerType(64));

      auto const_attr = GetI64ElementsAttr(expected_update_shape, &rewriter);

      auto const_op =
          rewriter.create<TF::ConstOp>(op->getLoc(), const_type, const_attr);

      auto broadcast_to_type = tensorflow::GetTypeFromTFTensorShape(
          llvm::ArrayRef<int64_t>(expected_update_shape),
          updates_ty.getElementType());

      updates = rewriter.create<TF::BroadcastToOp>(
          op->getLoc(), broadcast_to_type, op.getUpdates(), const_op);

      updates_ty = updates.getType().template dyn_cast<RankedTensorType>();
    }

    int64_t tensor_rank = tensor_ty.getRank();
    int64_t indices_rank = indices_ty.getRank();
    int64_t updates_rank =
        updates.getType().template dyn_cast<RankedTensorType>().getRank();

    int64_t window_dims = tensor_rank - num_index_dims;
    auto dims_attr = ScatterDimensionNumbersAttr::get(
        rewriter.getContext(),
        llvm::to_vector<4>(
            llvm::seq<int64_t>(updates_rank - window_dims, updates_rank)),
        llvm::to_vector<4>(llvm::seq<int64_t>(0, num_index_dims)),
        llvm::to_vector<4>(llvm::seq<int64_t>(0, num_index_dims)),
        indices_rank - 1);

    Location loc = op.getLoc();
    auto scatter = rewriter.create<ScatterOp>(
        loc, op.getType(), ValueRange(Value(op.getTensor())), op.getIndices(),
        updates, dims_attr);
    Derived::BuildScatterBody(tensor_ty.getElementType(),
                              &scatter.getUpdateComputation(), loc, rewriter);

    rewriter.replaceOp(op, scatter.getResult(0));
    return success();
  }
};

class ConvertTensorScatterUpdateOp
    : public ConvertTensorScatterOp<ConvertTensorScatterUpdateOp,
                                    TF::TensorScatterUpdateOp> {
 public:
  using ConvertTensorScatterOp::ConvertTensorScatterOp;

  static void BuildScatterBody(Type element_type, Region *region, Location loc,
                               OpBuilder &builder) {
    OpBuilder::InsertionGuard guard(builder);
    Block *block = builder.createBlock(region);
    Type type =
        tensorflow::GetTypeFromTFTensorShape(/*shape=*/{}, element_type);
    block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));
    builder.create<ReturnOp>(loc, block->getArgument(1));
  }
};

class ConvertTensorScatterAddOp
    : public ConvertTensorScatterOp<ConvertTensorScatterAddOp,
                                    TF::TensorScatterAddOp> {
 public:
  using ConvertTensorScatterOp::ConvertTensorScatterOp;

  static void BuildScatterBody(Type element_type, Region *region, Location loc,
                               OpBuilder &builder) {
    OpBuilder::InsertionGuard guard(builder);
    Block *block = builder.createBlock(region);
    Type type =
        tensorflow::GetTypeFromTFTensorShape(/*shape=*/{}, element_type);
    block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));
    auto add_op = builder.create<AddOp>(loc, block->getArgument(0),
                                        block->getArgument(1));
    builder.create<ReturnOp>(loc, add_op.getResult());
  }
};

class ConvertTensorScatterSubOp
    : public ConvertTensorScatterOp<ConvertTensorScatterSubOp,
                                    TF::TensorScatterSubOp> {
 public:
  using ConvertTensorScatterOp::ConvertTensorScatterOp;

  static void BuildScatterBody(Type element_type, Region *region, Location loc,
                               OpBuilder &builder) {
    OpBuilder::InsertionGuard guard(builder);
    Block *block = builder.createBlock(region);
    Type type =
        tensorflow::GetTypeFromTFTensorShape(/*shape=*/{}, element_type);
    block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));
    auto sub_op = builder.create<SubtractOp>(loc, block->getArgument(0),
                                             block->getArgument(1));
    builder.create<ReturnOp>(loc, sub_op.getResult());
  }
};

class ConvertTensorScatterMinOp
    : public ConvertTensorScatterOp<ConvertTensorScatterMinOp,
                                    TF::TensorScatterMinOp> {
 public:
  using ConvertTensorScatterOp::ConvertTensorScatterOp;

  static void BuildScatterBody(Type element_type, Region *region, Location loc,
                               OpBuilder &builder) {
    OpBuilder::InsertionGuard guard(builder);
    Block *block = builder.createBlock(region);
    Type type =
        tensorflow::GetTypeFromTFTensorShape(/*shape=*/{}, element_type);
    block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));
    auto min_op = builder.create<MinOp>(loc, block->getArgument(0),
                                        block->getArgument(1));
    builder.create<ReturnOp>(loc, min_op.getResult());
  }
};

class ConvertTensorScatterMaxOp
    : public ConvertTensorScatterOp<ConvertTensorScatterMaxOp,
                                    TF::TensorScatterMaxOp> {
 public:
  using ConvertTensorScatterOp::ConvertTensorScatterOp;

  static void BuildScatterBody(Type element_type, Region *region, Location loc,
                               OpBuilder &builder) {
    OpBuilder::InsertionGuard guard(builder);
    Block *block = builder.createBlock(region);
    Type type =
        tensorflow::GetTypeFromTFTensorShape(/*shape=*/{}, element_type);
    block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));
    auto max_op = builder.create<MaxOp>(loc, block->getArgument(0),
                                        block->getArgument(1));
    builder.create<ReturnOp>(loc, max_op.getResult());
  }
};

// Converts Tile op to HLO BroadcastInDim and Reshape ops.
//   For shape [S1, S2] and multiples [M1, M2],
//     MS1 = M1 * S1; MS2 = M2 * S2
//
//   %broadcast = mhlo.broadcast_in_dim(%input) {
//     broadcast_dimensions = [0, 2]
//   }
//   %result = "mhlo.reshape"(%broadcast) : (tensor<S1xM1xS2xM2xf32>)
//      -> tensor<MS1xMS2xf32>
class ConvertTileOp : public OpRewritePattern<TF::TileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::TileOp op,
                                PatternRewriter &rewriter) const override {
    auto input_ty = op.getInput().getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) return failure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();
    Type element_type = input_ty.getElementType();

    DenseIntElementsAttr multiples;
    if (!matchPattern(op.getMultiples(), m_Constant(&multiples)) ||
        multiples.getType().getRank() != 1)
      return failure();

    const int64_t input_shape_size = input_shape.size();
    if (multiples.getNumElements() != input_shape_size) return failure();

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
        tensorflow::GetTypeFromTFTensorShape(broadcasted_shape, element_type);
    Type output_type = op.getType();

    Value result = rewriter.create<BroadcastInDimOp>(
        loc, broadcasted_type, op.getInput(),
        GetI64ElementsAttr(broadcast_dimensions, &rewriter));

    if (output_type != broadcasted_type) {
      result = rewriter.create<ReshapeOp>(loc, output_type, result);
    }

    rewriter.replaceOp(op, {result});

    return success();
  }
};

// Converts the tf.TileOp op into mhlo.dynamic_reshape
// TODO(disc): To recover static special case's performance with folding and
// canonicalization.
class ConvertTileOpDynamic : public OpRewritePattern<TF::TileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  // clang-format off
  // Converts Tile op to HLO DBroadcastInDim and DReshape ops.
  //   For shape [S1, S2] and multiples [M1, M2],
  //     MS1 = M1 * S1; MS2 = M2 * S2
  //
  //   %out_dim_size = [S1, M1, S2, M2]
  //   %broadcast_dimensions = [1, 3];
  //   %broadcast = mhlo.d_broadcast_in_dim(%input, %out_dim_size, %braodcast_dimensions);
  //   %shape = [MS1, MS2]
  //   %result = "mhlo.d_reshape"(%broadcast, %shape) : (tensor<S1xM1xS2xM2xf32>) -> tensor<MS1xMS2xf32>
  // clang-format on
  LogicalResult matchAndRewrite(TF::TileOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value multiples = op.getMultiples();
    auto input_ty = input.getType().dyn_cast<RankedTensorType>();
    if (!input_ty) return failure();
    // TODO(disc): Remove this constraint once fold and canonicalization
    // implemented.
    if (input_ty.hasStaticShape()) return failure();

    Type element_type = input_ty.getElementType();
    int64_t input_rank = input_ty.getRank();
    SmallVector<Value, 4> input_shape_values;
    for (int64_t i = 0; i < input_rank; ++i) {
      auto dim_size = input_ty.getDimSize(i);
      if (dim_size == ShapedType::kDynamic) {
        input_shape_values.push_back(
            rewriter.create<tensor::DimOp>(loc, input, i));
      } else {
        input_shape_values.push_back(rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(dim_size)));
      }
    }

    auto multiples_ty = multiples.getType().dyn_cast<RankedTensorType>();
    int64_t multiples_rank = multiples_ty.getRank();
    // rank of multiples input of tf.TileOp must be 1
    if (multiples_rank != 1) return failure();
    // multiples input of tf.TileOp must be fixed shaped
    if ((!multiples_ty.hasStaticShape()) ||
        (multiples_ty.getDimSize(0) != input_rank)) {
      return failure();
    }
    Type index_ty = rewriter.getIndexType();
    // %out_dim_size
    SmallVector<Value, 4> out_dim_size;
    out_dim_size.reserve(input_rank * 2);
    for (int64_t dim_idx = 0; dim_idx < input_rank; ++dim_idx) {
      Value index = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(dim_idx));
      Value multiples_size =
          rewriter.create<tensor::ExtractOp>(loc, multiples, ValueRange{index});
      Value multiples_size_casted =
          rewriter.create<arith::IndexCastOp>(loc, index_ty, multiples_size);
      out_dim_size.push_back(multiples_size_casted);
      out_dim_size.push_back(input_shape_values[dim_idx]);
    }
    SmallVector<int64_t, 4> broadcast_dimensions;
    broadcast_dimensions.reserve(input_rank);
    for (int64_t dim_idx = 0; dim_idx < input_rank; ++dim_idx) {
      broadcast_dimensions.push_back(1 + 2 * dim_idx);
    }
    auto broadcast_dims_attr =
        GetI64ElementsAttr(broadcast_dimensions, &rewriter);

    Value out_dim_size_tensor = rewriter.create<tensor::FromElementsOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape(
            {static_cast<int64_t>(out_dim_size.size())}, index_ty),
        out_dim_size);
    SmallVector<int64_t, 4> broadcast_shape(input_rank * 2,
                                            ShapedType::kDynamic);
    RankedTensorType broadcast_type =
        tensorflow::GetTypeFromTFTensorShape(broadcast_shape, element_type);
    Value broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, broadcast_type, input, out_dim_size_tensor, broadcast_dims_attr);

    // %shape = [MS1, MS2]
    SmallVector<Value, 4> shape_values;
    shape_values.reserve(input_rank);
    for (int64_t i = 0; i < input_rank; ++i) {
      Value dim_size_value = rewriter.create<mlir::arith::MulIOp>(
          loc, out_dim_size[2 * i], out_dim_size[2 * i + 1]);
      shape_values.push_back(dim_size_value);
    }
    Value shape = rewriter.create<tensor::FromElementsOp>(
        loc, tensorflow::GetTypeFromTFTensorShape({input_rank}, index_ty),
        shape_values);
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(op, op.getType(),
                                                        broadcast, shape);
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

    Type element_type = op.getOrigInput()
                            .getType()
                            .template cast<TensorType>()
                            .getElementType();

    // Compute paddings using the original input and kernel shape and strides.
    // Here, ReduceWindow op as used as the MaxPool op is lowered to the
    // ReduceWindow op.
    auto input_ty =
        op.getOrigInput().getType().template dyn_cast<RankedTensorType>();
    if (!input_ty) return failure();
    DenseIntElementsAttr paddings_attr = GetReduceWindowPaddingAsAttr<num_dims>(
        input_ty.getShape(), op.getKsize(), op.getStrides(), op.getPadding(),
        &rewriter);

    auto result = rewriter.create<SelectAndScatterOp>(
        loc, op.getType(), op.getOrigInput(), op.getGrad(),
        GetScalarConstOfType(element_type, loc, 0, &rewriter),
        GetI64ElementsAttr(op.getKsize()), GetI64ElementsAttr(op.getStrides()),
        paddings_attr);

    BuildReduceBody<AddOp>(element_type, &result.getScatter(), &rewriter);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block = rewriter.createBlock(&result.getSelect());

      // Block arguments are scalars of the given element type.
      Type type =
          tensorflow::GetTypeFromTFTensorShape(/*shape=*/{}, element_type);
      block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));

      auto reducer = rewriter.create<CompareOp>(loc, block->getArgument(0),
                                                block->getArgument(1),
                                                ComparisonDirection::GE);
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
//   %rev_filter = "mhlo.reverse"(%filter)
//   %result = "mhlo.convolution"(%out_backprop, %rev_filter)
template <typename OpTy, int num_spatial_dims>
class ConvertConvBackpropInputOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Unpack all of the attributes.
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.getDataFormat().str(), &data_format))
      return op.emitOpError("invalid data format");
    constexpr int num_dims = num_spatial_dims + 2;
    int batch_dim = GetTensorBatchDimIndex(num_dims, data_format);

    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.getPadding().str(), &padding).ok())
      return failure();

    auto out_backprop_ty =
        op.getOutBackprop().getType().template dyn_cast<RankedTensorType>();
    auto filter_ty =
        op.getFilter().getType().template dyn_cast<RankedTensorType>();

    // With the exception of out_backprop's batch dimension, out_backprop and
    // filter need to have static shape. Filter is validated here, out_backprop
    // is mostly validated at use.
    if (!out_backprop_ty || !filter_ty || !filter_ty.hasStaticShape())
      return failure();

    // Compute input_shape by supporting either:
    //   1) Fully static shapes, represented as constants.
    //   2) Static shapes with a dynamic batch dimension, represented as
    //      1D tf.Pack of a batch dimension (can be static or dynamic)
    //      and other dimensions (can only be static), for example:
    //      "tf.Pack"(%142, %cst_301, %cst_301, %cst_300) {axis = 0 : i64, ...}
    std::vector<int64_t> input_shape;
    DenseIntElementsAttr input_shape_attr;
    if (matchPattern(op.getInputSizes(), m_Constant(&input_shape_attr)) &&
        input_shape_attr.getType().getRank() == 1) {
      input_shape.insert(input_shape.end(),
                         input_shape_attr.getValues<int32_t>().begin(),
                         input_shape_attr.getValues<int32_t>().end());
    } else {
      auto pack = op.getInputSizes().template getDefiningOp<TF::PackOp>();
      if (!pack || pack.getAxis() != 0) return failure();
      auto pack_ty = pack.getType().template dyn_cast<RankedTensorType>();
      if (!pack_ty || pack_ty.getRank() != 1) return failure();
      for (auto i = 0; i < pack_ty.getDimSize(0); ++i) {
        if (i == batch_dim) {
          // We don't use the batch dimension below, so we don't care about
          // its size. Might as well populate it with -1.
          input_shape.push_back(ShapedType::kDynamic);
        } else {
          DenseIntElementsAttr input_dims_attr;
          if (matchPattern(pack.getValues()[i], m_Constant(&input_dims_attr)) &&
              input_dims_attr.getType().getRank() == 0) {
            input_shape.push_back(input_dims_attr.getSplatValue<int32_t>());
          } else {
            return failure();
          }
        }
      }
    }

    auto dilations_attr = GetI64ElementsAttr(op.getDilations());
    std::vector<int> dilations{
        dilations_attr.template getValues<int64_t>().begin(),
        dilations_attr.template getValues<int64_t>().end()};
    auto strides_attr = GetI64ElementsAttr(op.getStrides());
    std::vector<tensorflow::int32> strides{
        strides_attr.template getValues<int64_t>().begin(),
        strides_attr.template getValues<int64_t>().end()};

    std::vector<int64_t> explicit_paddings;
    if (padding == tensorflow::Padding::EXPLICIT) {
      // EXPLICIT padding mode and the associated attribute is limited to
      // Conv2DBackpropInput. So, fetch attribute by identifier instead of the
      // op.explicit_paddings() attribute getter.
      ArrayRef<Attribute> explicit_paddings_attr =
          op->template getAttrOfType<ArrayAttr>("explicit_paddings").getValue();
      explicit_paddings.reserve(explicit_paddings_attr.size());
      for (Attribute explicit_padding : explicit_paddings_attr)
        explicit_paddings.push_back(
            explicit_padding.cast<IntegerAttr>().getInt());
    }

    ArrayRef<int64_t> filter_shape = filter_ty.getShape();

    // Compute ConvDimensionNumbers, dilation, and padding.
    SmallVector<int64_t, num_spatial_dims> spatial_dims;
    SmallVector<int64_t, num_spatial_dims> lhs_dilation;
    SmallVector<int64_t, num_spatial_dims> rhs_dilation;
    SmallVector<int64_t, num_spatial_dims * 2> paddings;

    for (int i : llvm::seq<int>(0, num_spatial_dims)) {
      const int64_t spatial_dim =
          GetTensorSpatialDimIndex(num_dims, data_format, i);
      spatial_dims.push_back(spatial_dim);

      // Prepare metadata indexed by spatial_dim for computing pad_before
      // and pad_after.
      int64_t input_size = input_shape[spatial_dim];
      if (input_size == ShapedType::kDynamic) return failure();
      int64_t output_size = out_backprop_ty.getDimSize(spatial_dim);
      if (output_size == ShapedType::kDynamic) return failure();
      int64_t filter_size = filter_ty.getDimSize(i);
      int64_t stride = strides[spatial_dim];
      int64_t dilation = dilations[spatial_dim];

      // Compute pad_before and pad_after following the logic from
      // ConvBackpropComputeDimensionsV2. (Unfortunately, we cannot call
      // the function in question because it doesn't work with dynamic dims).
      int64_t padding_before = -1, padding_after = -1;
      if (padding == tensorflow::Padding::EXPLICIT) {
        padding_before = explicit_paddings[2 * spatial_dim];
        padding_after = explicit_paddings[2 * spatial_dim + 1];
      }
      int64_t expected_output_size = 0;
      auto status = GetWindowedOutputSizeVerboseV2(
          input_size, filter_size, dilation, stride, padding,
          &expected_output_size, &padding_before, &padding_after);
      if (!status.ok()) return failure();
      if (output_size != expected_output_size) return failure();
      int64_t effective_filter_size = (filter_size - 1) * dilation + 1;
      int64_t pad_before = effective_filter_size - 1 - padding_before;
      int64_t padded_out_size = input_size + effective_filter_size - 1;
      int64_t expanded_output_size = (output_size - 1) * stride + 1;
      int64_t pad_after = padded_out_size - expanded_output_size - pad_before;

      // Populate metadata for the upcoming mhlo.conv op using the result of
      // the computations performed above.
      lhs_dilation.push_back(stride);
      rhs_dilation.push_back(dilation);
      paddings.push_back(pad_before);
      paddings.push_back(pad_after);
    }

    RankedTensorType paddings_ty = tensorflow::GetTypeFromTFTensorShape(
        {num_spatial_dims, 2}, rewriter.getIntegerType(64));
    auto paddings_attr = DenseIntElementsAttr::get(paddings_ty, paddings);

    Value filter = op.getFilter();

    const int feature_dim =
        tensorflow::GetTensorFeatureDimIndex(num_dims, data_format);
    const int64_t in_depth = *(input_shape.begin() + feature_dim);
    if (in_depth == ShapedType::kDynamic) return failure();
    const int64_t filter_in_depth = filter_shape[num_spatial_dims];
    const int64_t feature_group_count = in_depth / filter_in_depth;

    if (feature_group_count != 1) {
      // 1. Reshape filter from
      //   [H, W, ..., filter_in_depth, out_depth] to
      //   [H, W, ..., filter_in_depth, G, out_depth / G].
      auto new_shape = llvm::to_vector<6>(filter_shape);
      new_shape.back() = feature_group_count;
      new_shape.push_back(filter_shape.back() / feature_group_count);
      Type filter_element_ty = filter_ty.getElementType();
      auto ty =
          tensorflow::GetTypeFromTFTensorShape(new_shape, filter_element_ty);
      filter = rewriter.create<ReshapeOp>(op.getLoc(), ty, filter);

      // 2. Transpose to [H, W, ..., G, filter_in_depth, out_depth / G].
      llvm::SmallVector<int64_t, 6> perm(num_dims + 1);
      std::iota(perm.begin(), perm.end(), 0);
      std::swap(perm[num_spatial_dims], perm[num_spatial_dims + 1]);
      std::swap(new_shape[num_spatial_dims], new_shape[num_spatial_dims + 1]);
      ty = tensorflow::GetTypeFromTFTensorShape(new_shape, filter_element_ty);
      filter = rewriter.create<TransposeOp>(
          op.getLoc(), ty, filter, GetI64ElementsAttr(perm, &rewriter));

      // 3. Reshape to [H, W, ..., in_depth, out_depth / G].
      new_shape[num_spatial_dims] *= new_shape[num_spatial_dims + 1];
      new_shape[num_spatial_dims + 1] = new_shape.back();
      new_shape.pop_back();
      ty = tensorflow::GetTypeFromTFTensorShape(new_shape, filter_element_ty);
      filter = rewriter.create<ReshapeOp>(op.getLoc(), ty, filter);
    }

    SmallVector<int64_t, 4> kernel_spatial_dims;
    kernel_spatial_dims.resize(num_spatial_dims);
    std::iota(kernel_spatial_dims.begin(), kernel_spatial_dims.end(), 0);

    // Mirror the filter in the spatial dimensions.
    filter = rewriter.create<ReverseOp>(
        op.getLoc(), filter,
        GetI64ElementsAttr(kernel_spatial_dims, &rewriter));

    // activation gradients
    //   = gradients (with padding and dilation) <conv> mirrored_weights
    Value result = rewriter.create<ConvolutionOp>(
        op.getLoc(), op.getType(), op.getOutBackprop(), filter,
        /*window_strides=*/
        GetI64ElementsAttrForValue(/*size=*/num_spatial_dims, /*val=*/1,
                                   &rewriter),
        /*padding=*/paddings_attr, GetI64ElementsAttr(lhs_dilation, &rewriter),
        GetI64ElementsAttr(rhs_dilation, &rewriter),
        /*window_reversal=*/nullptr,
        ConvDimensionNumbersAttr::get(
            rewriter.getContext(),
            /*input_batch_dimension=*/batch_dim,
            /*input_feature_dimension=*/feature_dim,
            /*input_spatial_dimensions=*/spatial_dims,
            // TF filter shape is [ H, W, ..., inC, outC ]
            // Transpose the input and output features for computing the
            // gradient.
            /*kernel_input_feature_dimension=*/
            num_spatial_dims + 1,
            /*kernel_output_feature_dimension=*/
            num_spatial_dims,
            /*kernel_spatial_dimensions=*/kernel_spatial_dims,
            /*output_batch_dimension=*/batch_dim,
            /*output_feature_dimension=*/feature_dim,
            /*output_spatial_dimensions=*/spatial_dims),
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
//   %result = "mhlo.convolution"(%input, %out_backprop)
template <typename OpTy, int num_spatial_dims>
class ConvertConvBackpropFilterOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Unpack all of the attributes.
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.getDataFormat().str(), &data_format))
      return op.emitOpError("invalid data format");

    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.getPadding().str(), &padding).ok())
      return failure();

    auto out_backprop_ty =
        op.getOutBackprop().getType().template dyn_cast<RankedTensorType>();
    auto input_ty =
        op.getInput().getType().template dyn_cast<RankedTensorType>();

    for (RankedTensorType ty : {out_backprop_ty, input_ty})
      if (!ty || !ty.hasStaticShape()) return failure();

    ArrayRef<int64_t> out_backprop_shape = out_backprop_ty.getShape();
    ArrayRef<int64_t> input_shape = input_ty.getShape();

    DenseIntElementsAttr filter_shape_attr;
    if (!matchPattern(op.getFilterSizes(), m_Constant(&filter_shape_attr)) ||
        filter_shape_attr.getType().getRank() != 1)
      return failure();

    auto dilations_attr = GetI64ElementsAttr(op.getDilations());
    std::vector<int> dilations{
        dilations_attr.template getValues<int64_t>().begin(),
        dilations_attr.template getValues<int64_t>().end()};
    auto strides_attr = GetI64ElementsAttr(op.getStrides());
    std::vector<tensorflow::int32> strides{
        strides_attr.template getValues<int64_t>().begin(),
        strides_attr.template getValues<int64_t>().end()};

    std::vector<int64_t> explicit_paddings;
    if (padding == tensorflow::Padding::EXPLICIT) {
      // EXPLICIT padding mode and the associated attribute is limited to
      // Conv2DBackpropFilter. So, fetch attribute by identifier instead of the
      // op.explicit_paddings() attribute getter.
      ArrayRef<Attribute> explicit_paddings_attr =
          op->template getAttrOfType<ArrayAttr>("explicit_paddings").getValue();
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
    const int64_t batch_group_count = in_depth / filter_in_depth;

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

    RankedTensorType paddings_ty = tensorflow::GetTypeFromTFTensorShape(
        {num_spatial_dims, 2}, rewriter.getIntegerType(64));
    auto paddings_attr = DenseIntElementsAttr::get(paddings_ty, paddings);

    SmallVector<int64_t, 4> output_spatial_dimensions;
    output_spatial_dimensions.resize(num_spatial_dims);
    std::iota(output_spatial_dimensions.begin(),
              output_spatial_dimensions.end(), 0);

    const int batch_dim =
        tensorflow::GetTensorBatchDimIndex(num_dims, data_format);

    Value result = rewriter.create<ConvolutionOp>(
        op.getLoc(), op.getType(), op.getInput(), op.getOutBackprop(),
        /*window_strides=*/GetI64ElementsAttr(window_strides, &rewriter),
        /*padding=*/paddings_attr, /*lhs_dilation=*/
        GetI64ElementsAttrForValue(/*size=*/num_spatial_dims, /*val=*/1,
                                   &rewriter),
        GetI64ElementsAttr(rhs_dilation, &rewriter),
        /*window_reversal=*/nullptr,
        ConvDimensionNumbersAttr::get(
            rewriter.getContext(),
            // Swap batch_dim and feature_dim in the activations.
            /*input_batch_dimension=*/feature_dim,
            /*input_feature_dimension=*/batch_dim,
            /*input_spatial_dimensions=*/kernel_spatial_dims,
            // The gradients become the RHS of the convolution.
            // The gradients have shape [batch, out_rows, out_cols, ...,
            // out_depth] where the batch becomes the input feature for the
            // convolution.
            /*kernel_input_feature_dimension=*/batch_dim,
            /*kernel_output_feature_dimension=*/feature_dim,
            /*kernel_spatial_dimensions=*/kernel_spatial_dims,
            /*output_batch_dimension=*/num_spatial_dims,
            /*output_feature_dimension=*/num_spatial_dims + 1,
            /*output_spatial_dimensions=*/output_spatial_dimensions),
        /*feature_group_count=*/rewriter.getI64IntegerAttr(1),
        rewriter.getI64IntegerAttr(batch_group_count),
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
    auto indices_ty = op.getIndices().getType().dyn_cast<RankedTensorType>();
    if (!indices_ty || !indices_ty.hasStaticShape()) return failure();
    ArrayRef<int64_t> indices_shape = indices_ty.getShape();
    Type element_type = indices_ty.getElementType();

    DenseIntElementsAttr depth_attr;
    if (!matchPattern(op.getDepth(), m_Constant(&depth_attr))) {
      return failure();
    }

    int64_t depth = depth_attr.getValues<APInt>()[0].getSExtValue();
    int64_t axis = op.getAxis();
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
    auto index_type =
        tensorflow::GetTypeFromTFTensorShape(output_dims, element_type);
    auto iota = rewriter.create<IotaOp>(
        loc, index_type, IntegerAttr::get(rewriter.getIntegerType(64), axis));
    auto broadcast_indices = rewriter.create<BroadcastInDimOp>(
        loc, index_type, op.getIndices(),
        GetI64ElementsAttr(broadcast_dims, &rewriter));

    Value compare = rewriter.create<mhlo::CompareOp>(
        loc, broadcast_indices, iota, ComparisonDirection::EQ);
    Value on_value = rewriter.create<BroadcastOp>(
        loc, op.getType(), op.getOnValue(),
        GetI64ElementsAttr(output_dims, &rewriter));
    Value off_value = rewriter.create<BroadcastOp>(
        loc, op.getType(), op.getOffValue(),
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
// operand of infeed. The mhlo.InfeedOp can produce multiple results and later
// will be exported to XLA infeed op with single tuple return type.
//
// For example the following IR:
// %0:2 = "tf.InfeedDequeueTuple"() : () -> (tensor<3xi32>, tensor<4xf32>)
//
// would be lowered to
//
// %token = "mhlo.create_token"() : () -> !mhlo.token
// %data_and_token = "mhlo.infeed"(%token) {infeed_config = ""} :
//      (!mhlo.token) -> tensor<3xi32>, tensor<4xf32>, !mhlo.token>
//
class ConvertInfeedDequeueTupleOp
    : public OpRewritePattern<TF::InfeedDequeueTupleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::InfeedDequeueTupleOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> result_types;
    result_types.reserve(op.getOutputs().size() + 1);
    for (const auto &output : op.getOutputs()) {
      Type ty = output.getType();
      if (auto tensor_ty = ty.dyn_cast<RankedTensorType>()) {
        if (!tensor_ty.hasStaticShape()) return failure();
      }
      result_types.push_back(ty);
    }

    // Infeed takes a single token operand. Generate the token using
    // create_token op to pass to the infeed op.
    auto token = rewriter.create<CreateTokenOp>(
        op.getLoc(), mhlo::TokenType::get(rewriter.getContext()));

    result_types.push_back(token.getType());

    ArrayAttr layout;  // filled in during the xla-adjust-layout pass
    auto data_and_token =
        rewriter.create<InfeedOp>(op.getLoc(), result_types, token,
                                  /*infeed_config=*/rewriter.getStringAttr(""),
                                  /*layout=*/layout);

    result_types.pop_back();  // remove the token type.

    if (op.get_XlaSharding().has_value()) {
      // _XlaSharding attribute in TF is a serialized string of the OpSharding
      // proto, so convert to a text form here.
      ::xla::OpSharding sharding_proto;
      if (!sharding_proto.ParseFromString(op.get_XlaSharding().value().str()))
        return failure();

      // Token is a control signal and not a real data, so arbitrarily assign
      // the token to device 0.
      if (sharding_proto.type() == ::xla::OpSharding::TUPLE) {
        *sharding_proto.add_tuple_shardings() =
            ::xla::sharding_builder::AssignDevice(0);
        data_and_token->setAttr(
            kShardingAttr,
            rewriter.getStringAttr(sharding_proto.SerializeAsString()));
      } else {
        data_and_token->setAttr(kShardingAttr, op.get_XlaShardingAttr());
      }
    }

    if (op->hasAttr("layouts")) {
      // Append a UnitAttr for the "token" operand of the mhlo.infeed op here to
      // avoid compilation failure when exporting "layouts" attribute of the
      // corresponding InfeedDequeueTupleOp to a graph node.
      data_and_token->setAttr("layout", op->getAttr("layouts"));
    }
    llvm::SmallVector<Value> results;
    results.reserve(result_types.size());
    for (auto &idx_and_type : llvm::enumerate(result_types)) {
      results.push_back(data_and_token.getResult(idx_and_type.index()));
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
// %token = "mhlo.create_token"() : () -> !mhlo.token
// %outfeed_token = "mhlo.outfeed"(%val_1, %val_2, %token) {outfeed_config = ""}
// :
//      (tensor<3xi32>, tensor<4xf32>, !mhlo.token) -> !mhlo.token
//
class ConvertOutfeedEnqueueTupleOp
    : public OpRewritePattern<TF::OutfeedEnqueueTupleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::OutfeedEnqueueTupleOp op,
                                PatternRewriter &rewriter) const override {
    auto token_type = mhlo::TokenType::get(rewriter.getContext());
    auto token = rewriter.create<CreateTokenOp>(op.getLoc(), token_type);

    rewriter.create<OutfeedOp>(op.getLoc(), token_type, op.getInputs(), token,
                               /*outfeed_config=*/rewriter.getStringAttr(""));
    rewriter.eraseOp(op);
    return success();
  }
};

// Converts tf.TopKV2 to chlo.top_k.
class ConvertTopKV2Op : public OpRewritePattern<TF::TopKV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::TopKV2Op op,
                                PatternRewriter &rewriter) const override {
    // We can only match when the `k` operand is a constant scalar.
    DenseIntElementsAttr k_attr;
    if (!matchPattern(op.getK(), m_Constant(&k_attr))) return failure();
    int64_t k = (*k_attr.begin()).getSExtValue();

    TensorType input_type = op.getInput().getType().cast<TensorType>();
    if (!input_type.hasRank()) return failure();
    int64_t input_rank = input_type.getRank();
    int64_t last_dim_index = input_rank - 1;
    int64_t last_dim_size = input_type.getDimSize(last_dim_index);
    if (last_dim_size == ShapedType::kDynamic) return failure();

    rewriter.replaceOpWithNewOp<chlo::TopKOp>(op, op.getInput(), k);
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
    auto value_type = op.getValue().getType().dyn_cast<RankedTensorType>();
    if (!value_type) return failure();

    int64_t value_rank = value_type.getRank();
    int64_t axis = op.getAxis();
    if (axis < 0) axis += value_rank;

    // Parameters for constructing each slice.
    SmallVector<int64_t, 4> begin_indices(value_rank, 0);
    auto end_indices = llvm::to_vector<4>(value_type.getShape());
    SmallVector<int64_t, 4> strides(value_rank, 1);

    // All HLO slice+squeeze results used to replace the original tf.Unpack op.
    SmallVector<Value, 4> results;
    results.reserve(op.getNumResults());

    for (int i = 0, end = op.getNumResults(); i < end; ++i) {
      begin_indices[axis] = i;
      end_indices[axis] = i + 1;

      auto slice_op = rewriter.create<mhlo::SliceOp>(
          op.getLoc(), op.getValue(),
          GetI64ElementsAttr(begin_indices, &rewriter),
          GetI64ElementsAttr(end_indices, &rewriter),
          GetI64ElementsAttr(strides, &rewriter));
      // Reshape to drop the axis dimension.
      auto result = rewriter.create<TF::SqueezeOp>(
          op.getLoc(), op.getType(i), slice_op,
          rewriter.getI64ArrayAttr(op.getAxis()));
      results.push_back(result);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

// Converts tf.Unpack to a series of XLA HLO Slice ops.
// TODO(disc): To recover static special case's performance with folding and
// canonicalization.
class ConvertUnpackOpDynamic : public OpRewritePattern<TF::UnpackOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto value_type = op.getValue().getType().dyn_cast<RankedTensorType>();
    if (!value_type) return failure();
    // TODO(disc): Remove this constraint once fold and canonicalization
    // implemented.
    if (value_type.hasStaticShape()) return failure();

    int64_t value_rank = value_type.getRank();
    int64_t axis = op.getAxis();
    if (axis < 0) axis += value_rank;
    Location loc = op.getLoc();

    auto shape_scalar_type = rewriter.getIntegerType(32);
    // Parameters for constructing each slice.
    SmallVector<Value, 4> begin_indices, end_indices, strides;
    begin_indices.reserve(value_rank);
    end_indices.reserve(value_rank);
    strides.reserve(value_rank);
    // final output shape
    SmallVector<Value, 4> shape_values;
    shape_values.reserve(value_rank - 1);
    // slice shape before reshape, should be like{?, 1, ?, ?} if axis = 1
    SmallVector<int64_t, 4> slice_shape(value_rank, ShapedType::kDynamic);
    for (int64_t dim_idx = 0; dim_idx < value_rank; ++dim_idx) {
      int64_t dim_size = value_type.getDimSize(dim_idx);
      if (dim_size == ShapedType::kDynamic) {
        Value dim_i = rewriter.create<arith::IndexCastOp>(
            loc, shape_scalar_type,
            rewriter.create<tensor::DimOp>(loc, op.getOperand(), dim_idx));
        end_indices.push_back(dim_i);
        if (dim_idx != axis) {
          shape_values.push_back(dim_i);
        }
      } else {
        Value dim_i = rewriter.create<arith::ConstantOp>(
            loc, shape_scalar_type,
            rewriter.getIntegerAttr(shape_scalar_type, dim_size));
        end_indices.push_back(dim_i);
        if (dim_idx != axis) {
          shape_values.push_back(dim_i);
          slice_shape[dim_idx] = dim_size;
        } else {
          slice_shape[dim_idx] = 1;
        }
      }
      begin_indices.push_back(
          rewriter.create<arith::ConstantIntOp>(loc, 0, 32));
      strides.push_back(rewriter.create<arith::ConstantIntOp>(loc, 1, 32));
    }

    SmallVector<Value, 4> results;
    results.reserve(op.getNumResults());
    Type i32_ty = rewriter.getI32Type();
    for (int64_t i = 0; i < op.getNumResults(); ++i) {
      begin_indices[axis] = rewriter.create<arith::ConstantIntOp>(loc, i, 32);
      end_indices[axis] = rewriter.create<arith::ConstantIntOp>(loc, i + 1, 32);
      Value slice_op = rewriter.create<RealDynamicSliceOp>(
          loc,
          tensorflow::GetTypeFromTFTensorShape(slice_shape,
                                               value_type.getElementType()),
          op.getValue(),
          rewriter.create<tensor::FromElementsOp>(
              loc,
              tensorflow::GetTypeFromTFTensorShape(
                  {static_cast<int64_t>(begin_indices.size())}, i32_ty),
              begin_indices),
          rewriter.create<tensor::FromElementsOp>(
              loc,
              tensorflow::GetTypeFromTFTensorShape(
                  {static_cast<int64_t>(end_indices.size())}, i32_ty),
              end_indices),
          rewriter.create<tensor::FromElementsOp>(
              loc,
              tensorflow::GetTypeFromTFTensorShape(
                  {static_cast<int64_t>(strides.size())}, i32_ty),
              strides));
      // Reshape to drop the axis dimension.
      Value new_shape = rewriter.create<tensor::FromElementsOp>(
          loc,
          tensorflow::GetTypeFromTFTensorShape(
              {static_cast<int64_t>(shape_values.size())}, i32_ty),
          shape_values);
      Value reshape_op = rewriter.create<DynamicReshapeOp>(loc, op.getType(i),
                                                           slice_op, new_shape);
      results.push_back(reshape_op);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

// Converts the tf.SigmoidGradOp
// TODO(disc): To recover static special case's performance with folding and
// canonicalization.
class ConvertSigmoidGradOpDynamic : public OpRewritePattern<TF::SigmoidGradOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SigmoidGradOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value y = op.getY();
    Value dy = op.getDy();
    auto tp_y = y.getType().dyn_cast<RankedTensorType>();
    auto tp_dy = dy.getType().dyn_cast<RankedTensorType>();
    if (!tp_y || !tp_dy) return failure();

    // TODO(disc): Remove this constraint once fold and canonicalization
    // implemented.
    if (tp_y.hasStaticShape() || tp_dy.hasStaticShape()) return failure();

    Attribute attr;
    Type elem_tp = tp_y.getElementType();
    if (elem_tp.isSignlessInteger()) {
      attr = rewriter.getIntegerAttr(elem_tp, 1);
    } else {
      assert(elem_tp.isa<FloatType>());
      attr = rewriter.getFloatAttr(elem_tp, 1);
    }
    Value one = rewriter.create<mhlo::ConstantOp>(
        loc, DenseElementsAttr::get(
                 tensorflow::GetTypeFromTFTensorShape({}, elem_tp), attr));

    auto v0 = rewriter.create<chlo::BroadcastMulOp>(
        loc, dy, y, hlo::getBroadcastDimensionsAttr(&rewriter, dy, y));
    auto v1 = rewriter.create<chlo::BroadcastSubOp>(
        loc, one, y, hlo::getBroadcastDimensionsAttr(&rewriter, one, y));
    auto result = rewriter.create<chlo::BroadcastMulOp>(
        loc, v0, v1, hlo::getBroadcastDimensionsAttr(&rewriter, v0, v1));

    rewriter.replaceOp(op, result.getOperation()->getResults());
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
    auto data_type =
        op.getData().getType().template dyn_cast<RankedTensorType>();
    if (!data_type) return failure();
    int64_t data_rank = data_type.getRank();

    auto segment_ids_type =
        op.getSegmentIds().getType().template dyn_cast<RankedTensorType>();
    if (!segment_ids_type) return failure();
    int64_t segment_ids_rank = segment_ids_type.getRank();

    DenseIntElementsAttr num_segments_attr;
    if (!matchPattern(op.getNumSegments(), m_Constant(&num_segments_attr)))
      return failure();

    // The final shape for TF unsorted segment reduction op is [num_segments] +
    // data_shape[segment_ids_rank:].
    SmallVector<int64_t, 4> output_shape;
    output_shape.push_back((*num_segments_attr.begin()).getSExtValue());
    auto suffix = data_type.getShape().drop_front(segment_ids_rank);
    output_shape.append(suffix.begin(), suffix.end());
    auto output_type = tensorflow::GetTypeFromTFTensorShape(
        output_shape, data_type.getElementType());

    // Broadcast the initial value for reduction. This will become the
    // 'operand' parameter to scatter to for the final scatter op.
    Value init = ConcreteClass::GetInitialValue(data_type.getElementType(),
                                                op.getLoc(), &rewriter);
    auto broadcasted_init = rewriter.create<mhlo::BroadcastOp>(
        op.getLoc(), output_type, init,
        GetI64ElementsAttr(output_shape, &rewriter));

    // Parameters for the generated scatter op.
    SmallVector<int64_t, 1> inserted_window_dims(1, 0);
    SmallVector<int64_t, 1> scatter_dims_to_operand_dims(1, 0);
    int64_t index_vector_dim = segment_ids_rank;

    // Put all parameters in a StructAttr.
    auto dims_attr = ScatterDimensionNumbersAttr::get(
        rewriter.getContext(),
        llvm::to_vector<4>(llvm::seq<int64_t>(segment_ids_rank, data_rank)),
        inserted_window_dims, scatter_dims_to_operand_dims, index_vector_dim);

    auto scatter = rewriter.create<ScatterOp>(
        op.getLoc(), op.getType(), ValueRange(Value(broadcasted_init)),
        op.getSegmentIds(), op.getData(), dims_attr);
    BuildReduceBody<ReductionOp>(data_type.getElementType(),
                                 &scatter.getUpdateComputation(), &rewriter);

    rewriter.replaceOp(op, scatter.getResult(0));
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
    return GetScalarLimitConstOfType(reduce_element_type, loc, hlo::kLowest,
                                     rewriter);
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
    return GetScalarLimitConstOfType(reduce_element_type, loc, hlo::kMax,
                                     rewriter);
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
    auto no_op = [&]() {
      rewriter.replaceOp(op, op.getValue());
      return success();
    };

    auto input_type = op.getValue().getType().dyn_cast<RankedTensorType>();
    if (!input_type) return failure();
    if (input_type.hasStaticShape() && input_type.getNumElements() <= 1)
      // No shuffling is required, so copy input directly to output.
      return no_op();

    int64_t input_rank = input_type.getRank();
    int64_t first_dim_size = input_type.getDimSize(0);
    if (ShapedType::isDynamic(first_dim_size)) return failure();

    if (first_dim_size <= 1)
      // No shuffling is required, so copy input directly to output.
      return no_op();

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

      Value current = op.getValue();
      for (int i = 0; i < rounds; ++i) {
        auto keys =
            CreateRngUniform32(op.getLoc(), num_elements, /*lower_limit=*/0,
                               /*upper_limit=*/u32_max, &rewriter);
        auto sorted = createSortOp(
            &rewriter, op.getLoc(), {keys, current},
            {rewriter.getIntegerType(32), input_type.getElementType()},
            /*dimension=*/-1, /*is_stable=*/false,
            /*direction=*/ComparisonDirection::LT);
        current = sorted.getResult(1);
      }
      rewriter.replaceOp(op, current);
      return success();
    }

    // The Fisher-Yates algorithm.

    // Generate range(n) as the initial value for the indices to be swapped.
    auto indices_type = tensorflow::GetTypeFromTFTensorShape(
        {first_dim_size}, rewriter.getIntegerType(32));
    Value indices = rewriter.create<mhlo::IotaOp>(
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

      auto scalar_i32_type =
          tensorflow::GetTypeFromTFTensorShape({}, builder->getIntegerType(32));
      auto one_cross_i64_type = tensorflow::GetTypeFromTFTensorShape(
          {1}, builder->getIntegerType(64));

      auto scalar_one =
          DenseIntElementsAttr::get(one_cross_i64_type, ArrayRef<int64_t>(1));

      // We need to swap the indices[i] with indices[swaps[i]]. First get
      // these index values.
      Value source_index =
          builder->create<mhlo::DynamicSliceOp>(loc, indices, i, scalar_one);
      Value swap_index = builder->create<mhlo::ReshapeOp>(
          loc, scalar_i32_type,
          builder->create<mhlo::DynamicSliceOp>(loc, swaps, i, scalar_one));
      Value target_index = builder->create<mhlo::DynamicSliceOp>(
          loc, indices, swap_index, scalar_one);

      // Then perform the swap.
      // indices[i] <- indices[swaps[i]]
      indices = builder->create<mhlo::DynamicUpdateSliceOp>(
          loc, indices.getType(), indices, target_index, llvm::ArrayRef(i));
      // indices[swaps[i]] <- indices[i]
      indices = builder->create<mhlo::DynamicUpdateSliceOp>(
          loc, indices.getType(), indices, source_index,
          llvm::ArrayRef(swap_index));

      // Update new values.
      new_values->assign({swaps, indices});
    };

    // Create a while op to swap indices.
    SmallVector<Value, 2> while_output;
    CreateWhile32(op.getLoc(), first_dim_size, swap_body_fn, {swaps, indices},
                  &while_output, &rewriter);
    Value swaped_indices = while_output[1];

    // Gather the data using the swapped indices as the shuffled order.
    auto slice_sizes = tensorflow::ConvertMlirShapeToTF(input_type.getShape());
    slice_sizes[0] = 1;
    auto dims_attr = GatherDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*offset_dims=*/llvm::to_vector<4>(llvm::seq<int64_t>(1, input_rank)),
        /*collapsed_slice_dims=*/{0},
        /*start_index_map=*/{0},
        /*index_vector_dim=*/1);

    SmallVector<Value> slice_sizes_values;
    for (auto i = 0; i < slice_sizes.size(); ++i) {
      if (slice_sizes[i] == tensorflow::kTFDynamicSize) {
        Value i_const = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(i));
        Value slice_size_index =
            rewriter.create<shape::DimOp>(op.getLoc(), op.getValue(), i_const);
        Value index_to_i64 = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getI64Type(), slice_size_index);
        Value i64_to_tensor = rewriter.create<tensor::FromElementsOp>(
            op.getLoc(),
            tensorflow::GetTypeFromTFTensorShape({1}, rewriter.getI64Type()),
            index_to_i64);
        slice_sizes_values.push_back(i64_to_tensor);
      } else {
        slice_sizes_values.push_back(rewriter.create<mhlo::ConstantOp>(
            op.getLoc(), GetI64ElementsAttr({slice_sizes[i]}, &rewriter)));
      }
    }

    auto slice_sizes_concat = rewriter.create<mhlo::ConcatenateOp>(
        op.getLoc(), slice_sizes_values, rewriter.getI64IntegerAttr(0));
    rewriter.replaceOpWithNewOp<mhlo::DynamicGatherOp>(
        op, op.getType(), op.getValue(), swaped_indices, slice_sizes_concat,
        dims_attr);

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
    if (!op.get_XlaSharding().has_value()) return failure();

    NamedAttribute call_target_name = rewriter.getNamedAttr(
        "call_target_name", rewriter.getStringAttr("Sharding"));

    auto custom_call = rewriter.create<mhlo::CustomCallOp>(
        op.getLoc(), op.getType(), op.getInput(),
        ArrayRef<NamedAttribute>{call_target_name});
    custom_call->setAttr(kShardingAttr, op.get_XlaShardingAttr());
    rewriter.replaceOp(op, custom_call.getResult(0));

    return success();
  }
};

// Converts a TF InplaceUpdate op to DynamicUpdateSlice HLO.
class ConvertInplaceUpdateOp : public OpRewritePattern<TF::InplaceUpdateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::InplaceUpdateOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getX();
    auto indices = op.getI();
    auto updates = op.getV();

    // Slice each row of `i` and `v` to perform a separate dynamic-update-slice
    // on the contents of `x`.
    auto input_type = input.getType().cast<ShapedType>();
    auto updates_type = updates.getType().cast<ShapedType>();
    auto indices_type = indices.getType().cast<ShapedType>();
    if (!input_type.hasRank()) return failure();
    if (!updates_type.hasRank() || updates_type.isDynamicDim(0))
      return failure();
    if (!indices_type.hasStaticShape()) return failure();

    if (indices_type.getRank() != 1) return failure();

    SmallVector<Type, 4> unpacked_indices_type(
        indices_type.getDimSize(0), tensorflow::GetTypeFromTFTensorShape(
                                        {}, indices_type.getElementType()));
    // Note on zero_attr integer type: DynamicUpdateSlice op start_indices are
    // required to have matching types. This rewrite rule creates
    // DynamicUpdateSlice ops where the first "start index" is always i32 and
    // subsequent ones are constructed based on zero_attr. Thus the type
    // for zero_attr needs to be i32 as well.
    auto zero_attr = IntegerAttr::get(rewriter.getIntegerType(32), 0);
    auto unpacked_indices = rewriter.create<TF::UnpackOp>(
        op.getLoc(), unpacked_indices_type, indices, zero_attr);

    SmallVector<int64_t, 4> split_updates_shape;
    split_updates_shape.append(updates_type.getShape().begin(),
                               updates_type.getShape().end());
    split_updates_shape.front() = 1;
    SmallVector<Type, 4> split_updates_type;
    split_updates_type.resize(
        updates_type.getShape().front(),
        tensorflow::GetTypeFromTFTensorShape(split_updates_shape,
                                             updates_type.getElementType()));

    auto cst =
        rewriter.create<mhlo::ConstantOp>(op.getLoc(), zero_attr).getResult();
    auto split_updates = rewriter.create<TF::SplitOp>(
        op.getLoc(), split_updates_type, cst, updates);

    SmallVector<Value, 6> input_indices;
    input_indices.resize(input_type.getRank(), cst);

    for (auto pair :
         llvm::zip(unpacked_indices.getOutput(), split_updates.getOutput())) {
      input_indices.front() = std::get<0>(pair);
      input = rewriter.create<mhlo::DynamicUpdateSliceOp>(
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
    auto indices_type = op.getIndices().getType().dyn_cast<RankedTensorType>();
    if (!indices_type || !indices_type.hasStaticShape() ||
        indices_type.getShape().size() != 1)
      return failure();

    SmallVector<Type, 4> unpacked_indices_type(
        indices_type.getDimSize(0), tensorflow::GetTypeFromTFTensorShape(
                                        {}, indices_type.getElementType()));
    auto unpacked_indices = rewriter.create<TF::UnpackOp>(
        op.getLoc(), unpacked_indices_type, op.getIndices(),
        IntegerAttr::get(rewriter.getIntegerType(64), 0));
    rewriter.replaceOpWithNewOp<mhlo::DynamicUpdateSliceOp>(
        op, op.getType(), op.getInput(), op.getUpdate(),
        unpacked_indices.getOutput());
    return success();
  }
};

// Converts a TF XlaReduceScatter op to ReduceScatter HLO.
class ConvertXlaReduceScatterOp
    : public OpRewritePattern<TF::XlaReduceScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaReduceScatterOp op,
                                PatternRewriter &rewriter) const override {
    DenseIntElementsAttr group_assignment;
    if (!matchPattern(op.getGroupAssignment(), m_Constant(&group_assignment)))
      return failure();
    auto replica_groups =
        hlo::convertElementsAttr(group_assignment, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>();
    if (replica_groups.getType().getRank() != 2) return failure();

    APInt scatter_dimension;
    if (!matchPattern(op.getScatterDimension(),
                      m_ConstantInt(&scatter_dimension)))
      return failure();

    Location loc = op.getLoc();
    Type element_type = getElementTypeOrSelf(op.getInput().getType());

    auto reduce_scatter = rewriter.create<ReduceScatterOp>(
        loc, op.getType(), op.getInput(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                scatter_dimension.getSExtValue()),
        replica_groups, ChannelHandleAttr());
    StringRef reduce_op = op.getReduceOp();
    if (reduce_op == "Add") {
      BuildReduceBody<AddOp>(element_type, &reduce_scatter.getComputation(),
                             &rewriter);
    } else if (reduce_op == "Mul") {
      BuildReduceBody<MulOp>(element_type, &reduce_scatter.getComputation(),
                             &rewriter);
    } else if (reduce_op == "Min") {
      BuildReduceBody<MinOp>(element_type, &reduce_scatter.getComputation(),
                             &rewriter);
    } else if (reduce_op == "Max") {
      BuildReduceBody<MaxOp>(element_type, &reduce_scatter.getComputation(),
                             &rewriter);
    } else {
      // For mean, add replicas in the same group. Then divide the sum by the
      // number of replicas in each group below.
      assert(reduce_op == "Mean");
      BuildReduceBody<AddOp>(element_type, &reduce_scatter.getComputation(),
                             &rewriter);
    }
    Value result = reduce_scatter.getResult();

    // For mean, divide the merge result by group size.
    if (reduce_op == "Mean") {
      int64_t replica_group_size = replica_groups.getType().getDimSize(1);
      if (replica_group_size == 0) return failure();
      auto divisor = GetScalarConstOfType(element_type, loc, replica_group_size,
                                          &rewriter);
      auto broadcast_dims = GetI64ElementsAttr({}, &rewriter);
      result = rewriter.create<chlo::BroadcastDivOp>(
          loc, result, divisor.getResult(), broadcast_dims);
    }

    rewriter.replaceOp(op, {result});
    return success();
  }
};

// Converts tf.XlaReduceWindow to mhlo.ReduceWindow
class ConvertXlaReduceWindowOp
    : public OpRewritePattern<TF::XlaReduceWindowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaReduceWindowOp op,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr window_dimensions, window_strides, base_dilations,
        window_dilations, padding;
    if (!(matchPattern(op.getWindowDimensions(),
                       m_Constant(&window_dimensions)) &&
          matchPattern(op.getWindowStrides(), m_Constant(&window_strides)) &&
          matchPattern(op.getBaseDilations(), m_Constant(&base_dilations)) &&
          matchPattern(op.getWindowDilations(),
                       m_Constant(&window_dilations)) &&
          matchPattern(op.getPadding(), m_Constant(&padding))))
      return failure();

    Location loc = op.getLoc();

    SmallVector<Type> result_types{op.getResult().getType()};
    // Create the mhlo.SelectAndScatter op.
    auto reduce_window_op = rewriter.create<mhlo::ReduceWindowOp>(
        loc, result_types, op.getInput(), op.getInitValue(),
        hlo::convertElementsAttr(window_dimensions, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>(),
        hlo::convertElementsAttr(window_strides, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>(),
        hlo::convertElementsAttr(base_dilations, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>(),
        hlo::convertElementsAttr(window_dilations, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>(),
        hlo::convertElementsAttr(padding, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>());
    // Insert a call to the reducer in the region of the mhlo op.
    mlir::SymbolRefAttr func = op.getComputation();
    auto func_op = cast<mlir::func::FuncOp>(SymbolTable::lookupSymbolIn(
        op->getParentOfType<mlir::ModuleOp>(), func));
    auto func_ty = func_op.getFunctionType();
    BuildBodyWithCall(rewriter, loc, func, func_ty,
                      &reduce_window_op.getBody());

    rewriter.replaceOp(op, reduce_window_op.getResults());

    return success();
  }
};

// Converts ClipByValue to XLA's clamp operation. Includes the broadcasting
// semantics for static and dynamic cases.
class ConvertClipByValueOp : public OpRewritePattern<TF::ClipByValueOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::ClipByValueOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getX();
    Value min = op.getClipValueMin();
    Value max = op.getClipValueMax();

    auto input_ty = input.getType().cast<ShapedType>();
    auto min_ty = min.getType().cast<ShapedType>();
    auto max_ty = max.getType().cast<ShapedType>();

    if (!input_ty.hasRank() || !min_ty.hasRank() || !max_ty.hasRank()) {
      return failure();
    }

    auto shape = rewriter.create<TF::ShapeOp>(
        op.getLoc(),
        tensorflow::GetTypeFromTFTensorShape({input_ty.getRank()},
                                             rewriter.getI32Type()),
        input);

    if (min_ty != input_ty) {
      min =
          rewriter.create<TF::BroadcastToOp>(op.getLoc(), input_ty, min, shape);
    }

    if (max_ty != input_ty) {
      max =
          rewriter.create<TF::BroadcastToOp>(op.getLoc(), input_ty, max, shape);
    }

    rewriter.replaceOpWithNewOp<mhlo::ClampOp>(op, input_ty, min, input, max);
    return success();
  }
};

// Converts ConstOp to XLA's constant operation and introduces a tensor cast if
// needed.
class ConvertConstOp : public OpRewritePattern<TF::ConstOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::ConstOp op,
                                PatternRewriter &rewriter) const override {
    // Convert only for valid HLO tensors.
    auto ty = op.getType().dyn_cast<TensorType>();
    if (!ty || !ty.getElementType().isa<FloatType, IntegerType, ComplexType>())
      return failure();

    Location loc = op.getLoc();
    Value result = rewriter.create<mhlo::ConstantOp>(loc, op.getValue());
    if (result.getType() != op.getType())
      result = rewriter.create<tensor::CastOp>(loc, op.getType(), result);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Converts the Cumsum or Cumprod TensorFlow op to the HLO ReduceWindow op by
// setting appropriate window dimensions, with the given aggregation op as the
// reduction function. The input tensor needs to have a static shape, and 'axis'
// must be const. The TableGen pattern is not used for this rewrite because it
// involves regions.
template <typename OpT, typename AggregationOp>
class ConvertCumOp : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getX();
    auto input_type = input.getType().template dyn_cast<ShapedType>();
    if (!input_type || !input_type.hasStaticShape()) {
      return failure();
    }

    ArrayRef<int64_t> input_shape = input_type.getShape();
    int64_t rank = input_shape.size();

    // We can only match when the axis is a constant scalar.
    DenseIntElementsAttr axis_attr;
    if (!matchPattern(op.getAxis(), m_Constant(&axis_attr))) {
      return failure();
    }

    // Get the dimension to apply the reduction on, and offset properly if it is
    // negative.
    int64_t axis = (*axis_attr.begin()).getSExtValue();
    if (axis < 0) {
      axis += rank;
    }

    // If we're supposed to sum things up in the reverse direction, we reverse
    // the input and then later reverse the output.
    if (op.getReverse()) {
      llvm::SmallVector<int64_t, 4> dims_to_reverse({axis});
      input = rewriter.create<ReverseOp>(
          op.getLoc(), input, GetI64ElementsAttr(dims_to_reverse, &rewriter));
    }

    // Convert if we need to enlarge the element type's bitwidth to avoid
    // precision loss.
    Type input_element_type = input_type.getElementType();

    // TODO(hinsu): Handle complex element types.
    if (!input_element_type.isIntOrFloat()) return failure();

    Type sum_element_type = GetSumAccumulationType(input_element_type);
    input = rewriter.create<ConvertOp>(op.getLoc(), input, sum_element_type);

    SmallVector<int64_t, 4> window_dims(rank, 1);
    SmallVector<int64_t, 4> window_strides(rank, 1);
    window_dims[axis] = input_shape[axis];

    SmallVector<int64_t, 8> paddings(rank * 2, 0);
    paddings[axis * 2] =
        std::max(input_shape[axis] - 1, static_cast<int64_t>(0));
    auto paddings_attr =
        DenseIntElementsAttr::get(tensorflow::GetTypeFromTFTensorShape(
                                      {rank, 2}, rewriter.getIntegerType(64)),
                                  paddings);

    int64_t init_value = (std::is_same<AggregationOp, AddOp>::value) ? 0 : 1;
    Value init = GetScalarConstOfType(sum_element_type, op.getLoc(), init_value,
                                      &rewriter);

    auto reduce = rewriter.create<ReduceWindowOp>(
        op.getLoc(), input.getType(), input, init,
        GetI64ElementsAttr(rewriter.getI64ArrayAttr(window_dims)),
        GetI64ElementsAttr(rewriter.getI64ArrayAttr(window_strides)),
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(), paddings_attr);
    BuildReduceBody<AggregationOp>(sum_element_type, &reduce.getBody(),
                                   &rewriter);
    Value result = reduce.getResult(0);

    if (op.getExclusive()) {
      // In "exclusive" operation, the output will start with the "init" (0)
      // values. There is no way to express that as a ReduceWindowOp, so run the
      // normal operation, and then use a PadOp to add the 0 "column" on the
      // left and cut away the last column on the right.
      llvm::SmallVector<int64_t, 4> low_padding(rank, 0);
      llvm::SmallVector<int64_t, 4> high_padding(rank, 0);
      llvm::SmallVector<int64_t, 4> interior_padding(rank, 0);
      low_padding[axis] = 1;
      high_padding[axis] = -1;
      result = rewriter.create<PadOp>(
          op.getLoc(), result, init, GetI64ElementsAttr(low_padding, &rewriter),
          GetI64ElementsAttr(high_padding, &rewriter),
          GetI64ElementsAttr(interior_padding, &rewriter));
    }

    // Convert back if we enlarged the element type's bitwidth.
    result =
        rewriter.create<ConvertOp>(op.getLoc(), result, input_element_type);

    if (op.getReverse()) {
      llvm::SmallVector<int64_t, 4> dims_to_reverse({axis});
      result = rewriter.create<ReverseOp>(
          op.getLoc(), result, GetI64ElementsAttr(dims_to_reverse, &rewriter));
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

using ConvertCumsumOp = ConvertCumOp<TF::CumsumOp, AddOp>;
using ConvertCumprodOp = ConvertCumOp<TF::CumprodOp, MulOp>;

// Converts the Tensorflow ShapeOp to a sequence of Shape dialect and Standard
// dialect lowerings. This involves extracting the shape type, extracting and
// converting each dimension to a known integer type, and repacking into a final
// tensor.
class ConvertShapeOp : public OpRewritePattern<TF::ShapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::ShapeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();

    auto result_ty = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!result_ty) {
      return failure();
    }

    auto index_tensor = tensorflow::GetTypeFromTFTensorShape(
        result_ty.getShape(), rewriter.getIndexType());
    auto shape_op =
        rewriter.create<shape::ShapeOfOp>(op.getLoc(), index_tensor, input);
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, result_ty, shape_op);
    return success();
  }
};

class ConvertDynamicExpandDimsOp : public OpRewritePattern<TF::ExpandDimsOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::ExpandDimsOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto input_ty = input.getType().cast<ShapedType>();
    auto result_ty = op.getType().cast<ShapedType>();
    if (!result_ty.hasRank() || !input_ty.hasRank() ||
        result_ty.hasStaticShape()) {
      return failure();
    }

    DenseIntElementsAttr expand_dims_attr;
    if (!matchPattern(op.getDim(), m_Constant(&expand_dims_attr))) {
      return failure();
    }

    auto shape = rewriter.create<shape::ShapeOfOp>(
        op.getLoc(),
        tensorflow::GetTypeFromTFTensorShape({input_ty.getRank()},
                                             rewriter.getIndexType()),
        input);
    auto expand_dims = llvm::to_vector<6>(expand_dims_attr.getValues<APInt>());

    llvm::SmallVector<Value, 4> dims;
    dims.resize(result_ty.getRank());

    auto inserted_dim = expand_dims[0].getSExtValue();

    // Handle the negative value use case.
    if (inserted_dim < 0) {
      inserted_dim += result_ty.getRank();
      // This means the value is completely incorrect, just return.
      if (inserted_dim < 0) {
        return failure();
      }
    }

    dims[inserted_dim] =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);

    for (int i = 0; i < dims.size() - 1; i++) {
      // Add the extracted dim.
      Value index = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
      Value dim = rewriter.create<tensor::ExtractOp>(op.getLoc(), shape, index);
      dims[i >= inserted_dim ? i + 1 : i] = dim;
    }

    auto from_extents =
        rewriter.create<tensor::FromElementsOp>(op.getLoc(), dims);
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(op, result_ty, input,
                                                        from_extents);
    return success();
  }
};

class ConvertDynamicSqueezeOp : public OpRewritePattern<TF::SqueezeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SqueezeOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto input_ty = input.getType().cast<ShapedType>();
    auto result_ty = op.getType().cast<ShapedType>();
    if (!result_ty.hasRank() || !input_ty.hasRank() ||
        result_ty.hasStaticShape()) {
      return failure();
    }

    // The fully dynamic case is unsupported.
    if (op.getSqueezeDims().empty()) {
      return failure();
    }

    SmallVector<int64_t> squeeze_dims;
    int64_t input_rank = input_ty.getRank();
    for (const auto &squeeze_dim_apint :
         op.getSqueezeDims().getAsValueRange<IntegerAttr>()) {
      int64_t squeeze_dim = squeeze_dim_apint.getSExtValue();
      // Handle negative inputs.
      if (squeeze_dim < 0) squeeze_dim += input_rank;
      assert(squeeze_dim >= 0 && squeeze_dim < input_rank &&
             "squeeze dim out of bounds");

      squeeze_dims.push_back(squeeze_dim);
    }

    // Collect the unsqueezed dimensions.
    llvm::SmallVector<Value> dims;
    for (int64_t i = 0; i != input_rank; ++i) {
      if (llvm::is_contained(squeeze_dims, i)) continue;
      dims.push_back(rewriter.create<tensor::DimOp>(op.getLoc(), input, i));
    }

    auto from_extents =
        rewriter.create<tensor::FromElementsOp>(op.getLoc(), dims);
    // chlo::DynamicReshapeOp checks if the reshape is legal and will fail if
    // any non-1 dimension is squeezed.
    rewriter.replaceOpWithNewOp<chlo::DynamicReshapeOp>(op, result_ty, input,
                                                        from_extents);
    return success();
  }
};

// Converts tf.XlaConvV2 to mhlo.Conv
class ConvertXlaConvV2Op : public OpRewritePattern<TF::XlaConvV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaConvV2Op op,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr window_strides_attr, padding_attr, lhs_dilation_attr,
        rhs_dilation_attr, feature_group_count_attr;
    if (!(matchPattern(op.getWindowStrides(),
                       m_Constant(&window_strides_attr)) &&
          matchPattern(op.getPadding(), m_Constant(&padding_attr)) &&
          matchPattern(op.getLhsDilation(), m_Constant(&lhs_dilation_attr)) &&
          matchPattern(op.getRhsDilation(), m_Constant(&rhs_dilation_attr)) &&
          matchPattern(op.getFeatureGroupCount(),
                       m_Constant(&feature_group_count_attr))))
      return failure();

    auto window_strides_named_attr = rewriter.getNamedAttr(
        "window_strides", hlo::convertElementsAttr(window_strides_attr,
                                                   rewriter.getIntegerType(64))
                              .cast<DenseIntElementsAttr>());

    auto padding_named_attr = rewriter.getNamedAttr(
        "padding",
        hlo::convertElementsAttr(padding_attr, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>());

    auto lhs_dilation_named_attr = rewriter.getNamedAttr(
        "lhs_dilation",
        hlo::convertElementsAttr(lhs_dilation_attr, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>());

    auto rhs_dilation_named_attr = rewriter.getNamedAttr(
        "rhs_dilation",
        hlo::convertElementsAttr(rhs_dilation_attr, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>());

    int64_t feature_group_count_val =
        feature_group_count_attr.getValues<IntegerAttr>()[0].getInt();
    auto feature_group_count_named_attr = rewriter.getNamedAttr(
        "feature_group_count",
        rewriter.getI64IntegerAttr(feature_group_count_val));

    auto batch_group_count_named_attr =
        rewriter.getNamedAttr("batch_group_count", op.getBatchGroupCountAttr());

    xla::ConvolutionDimensionNumbers dnums;
    dnums.ParseFromString(op.getDimensionNumbersAttr().getValue().str());
    auto dimension_numbers_named_attr = rewriter.getNamedAttr(
        "dimension_numbers",
        xla::ConvertConvDimensionNumbers(dnums, &rewriter));

    xla::PrecisionConfig precision_config;
    precision_config.ParseFromString(
        op.getPrecisionConfigAttr().getValue().str());
    auto precision_config_named_attr = rewriter.getNamedAttr(
        "precision_config",
        xla::ConvertPrecisionConfig(&precision_config, &rewriter));

    SmallVector<Value, 2> operands{op.getLhs(), op.getRhs()};
    NamedAttribute attrs[] = {
        window_strides_named_attr,      padding_named_attr,
        lhs_dilation_named_attr,        rhs_dilation_named_attr,
        feature_group_count_named_attr, batch_group_count_named_attr,
        dimension_numbers_named_attr,   precision_config_named_attr};
    rewriter.replaceOpWithNewOp<mhlo::ConvolutionOp>(op, op.getType(), operands,
                                                     llvm::ArrayRef(attrs));
    return success();
  }
};

// Converts tf.XlaSelectAndScatter to mhlo.SelectAndScatter
class ConvertXlaSelectAndScatterOp
    : public OpRewritePattern<TF::XlaSelectAndScatterOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaSelectAndScatterOp op,
                                PatternRewriter &rewriter) const override {
    ElementsAttr window_dimensions, window_strides, padding;
    if (!(matchPattern(op.getWindowDimensions(),
                       m_Constant(&window_dimensions)) &&
          matchPattern(op.getWindowStrides(), m_Constant(&window_strides)) &&
          matchPattern(op.getPadding(), m_Constant(&padding))))
      return failure();

    Location loc = op.getLoc();

    SmallVector<Type> result_types{op.getResult().getType()};
    // Create the mhlo.SelectAndScatter op.
    auto select_and_scatter_op = rewriter.create<mhlo::SelectAndScatterOp>(
        loc, result_types, op.getOperand(), op.getSource(), op.getInitValue(),
        hlo::convertElementsAttr(window_dimensions, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>(),
        hlo::convertElementsAttr(window_strides, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>(),
        hlo::convertElementsAttr(padding, rewriter.getIntegerType(64))
            .cast<DenseIntElementsAttr>());

    auto insert_call_to = [&](const mlir::SymbolRefAttr &func, Region *region) {
      auto func_op = cast<mlir::func::FuncOp>(SymbolTable::lookupSymbolIn(
          op->getParentOfType<mlir::ModuleOp>(), func));
      auto func_ty = func_op.getFunctionType();
      BuildBodyWithCall(rewriter, loc, func, func_ty, region);
    };

    // Insert a call to the select function in the select region of the mhlo op.
    insert_call_to(op.getSelect(), &select_and_scatter_op.getSelect());
    // Insert a call to the scatter function in the scatter region of the mhlo
    // op.
    insert_call_to(op.getScatter(), &select_and_scatter_op.getScatter());

    rewriter.replaceOp(op, select_and_scatter_op.getResult());

    return success();
  }
};

// Convert tf.XlaSort to mhlo.Sort
class ConvertXlaSortOp : public OpRewritePattern<TF::XlaSortOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaSortOp op,
                                PatternRewriter &rewriter) const override {
    // Create the sort op.
    Type element_type = getElementTypeOrSelf(op.getInput().getType());
    auto sort_op =
        createSortOp(&rewriter, op.getLoc(), {op.getInput()}, {element_type},
                     /*dimension=*/-1, /*is_stable=*/false,
                     /*direction=*/ComparisonDirection::LT);
    rewriter.replaceOp(op, sort_op.getResult(0));
    return success();
  }
};

inline std::optional<xla::RandomAlgorithm> TensorFlowRngAlgToXla(
    tensorflow::Algorithm alg) {
  if (alg == tensorflow::RNG_ALG_PHILOX) {
    return xla::RandomAlgorithm::RNG_PHILOX;
  } else if (alg == tensorflow::RNG_ALG_THREEFRY) {
    return xla::RandomAlgorithm::RNG_THREE_FRY;
  } else if (alg == tensorflow::RNG_ALG_AUTO_SELECT) {
    return xla::RandomAlgorithm::RNG_DEFAULT;
  }
  return std::nullopt;
}

// Converts tf.XlaRngBitGenerator op to mhlo.RngBitGenerator op.
class ConvertXlaRngBitGeneratorOp
    : public OpRewritePattern<TF::XlaRngBitGeneratorOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaRngBitGeneratorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    DenseElementsAttr algorithm;
    if (!(matchPattern(op.getAlgorithm(), m_Constant(&algorithm))) ||
        algorithm.getType().getRank()) {
      return op.emitOpError() << "algorithm must be a constant scalar";
    }
    auto alg = static_cast<tensorflow::Algorithm>(
        algorithm.getValues<IntegerAttr>()[0].getInt());
    auto xla_alg = TensorFlowRngAlgToXla(alg);
    if (!xla_alg) {
      return op.emitOpError() << "unknown algorithm";
    }

    auto algorithm_attr = mlir::mhlo::RngAlgorithmAttr::get(
        rewriter.getContext(),
        *mlir::mhlo::symbolizeRngAlgorithm(xla_alg.value()));
    auto rng_bit_generator_op = rewriter.create<mhlo::RngBitGeneratorOp>(
        loc, op.getResultTypes(), algorithm_attr, op.getInitialState());

    rewriter.replaceOp(op, rng_bit_generator_op.getResults());

    return success();
  }
};

// Converts tf.XlaVariadicReduceV2 to mhlo.Reduce
class ConvertXlaVariadicReduceV2Op
    : public OpRewritePattern<TF::XlaVariadicReduceV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaVariadicReduceV2Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Create the mhlo.reduce op.
    auto reduce_op = rewriter.create<mhlo::ReduceOp>(
        loc, op.getInputs(), op.getInitValues(),
        GetI64ElementsAttr(op.getDimensionsToReduce()));
    mlir::SymbolRefAttr func = op.getReducer();
    auto func_op = cast<mlir::func::FuncOp>(SymbolTable::lookupSymbolIn(
        op->getParentOfType<mlir::ModuleOp>(), func));
    auto func_ty = func_op.getFunctionType();
    // Insert a call to the reducer in the region of the mhlo op.
    BuildBodyWithCall(rewriter, loc, func, func_ty, &reduce_op.getBody());

    rewriter.replaceOp(op, reduce_op.getResults());

    return success();
  }
};

// Convert tf.XlaVariadicSort to mhlo.Sort
class ConvertXlaVariadicSortOp
    : public OpRewritePattern<TF::XlaVariadicSortOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaVariadicSortOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ElementsAttr dimension;
    matchPattern(op.getDimension(), m_Constant(&dimension));
    // Create the mhlo.sort op.
    auto sort_op = rewriter.create<mhlo::SortOp>(
        loc, op.getInputs(), dimension.getValues<IntegerAttr>()[0].getInt(),
        op.getIsStable());
    mlir::SymbolRefAttr func = op.getComparator();
    auto func_op = cast<mlir::func::FuncOp>(SymbolTable::lookupSymbolIn(
        op->getParentOfType<mlir::ModuleOp>(), func));
    auto func_ty = func_op.getFunctionType();
    // Insert a call to the reducer in the region of the mhlo op.
    BuildBodyWithCall(rewriter, loc, func, func_ty, &sort_op.getComparator());

    rewriter.replaceOp(op, sort_op.getResults());
    return success();
  }
};

// Convert tf.XlaReducePrecision to mhlo.ReducePrecision
class ConvertXlaReducePrecisionOp
    : public OpRewritePattern<TF::XlaReducePrecisionOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaReducePrecisionOp op,
                                PatternRewriter &rewriter) const override {
    IntegerType int32_type = rewriter.getIntegerType(32);
    APInt exponent_bits = op.getExponentBitsAttr().getValue();
    // Truncating to 32-bits is safe, since pasing any number above the dtype
    // size (which is at most 64, for float64) is equivalent to passing the
    // dtype size.
    IntegerAttr new_exponent_attr =
        IntegerAttr::get(int32_type, exponent_bits.truncSSat(32));
    APInt mantissa_bits = op.getMantissaBitsAttr().getValue();
    IntegerAttr new_mantissa_attr =
        IntegerAttr::get(int32_type, mantissa_bits.truncSSat(32));
    rewriter.replaceOpWithNewOp<mhlo::ReducePrecisionOp>(
        op, op.getType(), op.getOperand(), new_exponent_attr,
        new_mantissa_attr);
    return success();
  }
};

class LowerYieldOp : public OpConversionPattern<TF::YieldOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::YieldOp op, TF::YieldOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mhlo::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

// Returns a new tensor type from the given type with element type updated to
// the given type.
TensorType UpdateElementTypeTo(Type ty, Type element_ty) {
  auto ranked_ty = ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) {
    return UnrankedTensorType::get(element_ty);
  }
  return RankedTensorType::get(ranked_ty.getShape(), element_ty,
                               ranked_ty.getEncoding());
}

template <typename SrcOpT, typename DstOpT>
class LowerControlFlowOp : public OpConversionPattern<SrcOpT> {
 public:
  using OpConversionPattern<SrcOpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpT op, typename SrcOpT::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    DstOpT mhlo_op;
    Location loc = op.getLoc();

    // To handle quant type conversions, use the converted operands' element
    // types and original source op's shapes and encoding to get converted op's
    // result types. This is only done for the While op for now.
    llvm::SmallVector<Type, 4> element_types;
    int64_t num_results = op.getNumResults();
    if constexpr (std::is_same<DstOpT, mhlo::WhileOp>::value) {
      element_types.reserve(num_results);
      for (Value value : adaptor.getOperands()) {
        element_types.push_back(getElementTypeOrSelf(value.getType()));
      }
    }

    if constexpr (std::is_same<DstOpT, mhlo::CaseOp>::value) {
      // Explicitly handle the Case op because it has variadic regions and takes
      // the number of regions as an input along with the operands.
      mhlo_op = rewriter.create<DstOpT>(loc, op.getResultTypes(),
                                        adaptor.getBranchIndex(),
                                        op.getBranches().size());
    } else if constexpr (std::is_same<DstOpT, mhlo::WhileOp>::value) {
      llvm::SmallVector<Type, 4> while_result_types;
      while_result_types.reserve(num_results);
      for (int64_t idx = 0; idx < num_results; ++idx) {
        auto ty = UpdateElementTypeTo(op.getType(idx), element_types[idx]);
        while_result_types.push_back(ty);
      }

      mhlo_op = rewriter.create<DstOpT>(loc, TypeRange(while_result_types),
                                        adaptor.getOperands());
    } else {
      mhlo_op = rewriter.create<DstOpT>(loc, op.getResultTypes(),
                                        adaptor.getOperands());
    }

    // Replace all uses of `op` results with the newly created op.
    rewriter.replaceOp(op, mhlo_op.getResults());

    int64_t num_regions = op.getNumRegions();
    for (int64_t idx = 0; idx < num_regions; ++idx) {
      Region &region = mhlo_op.getBodyRegion(idx);
      rewriter.inlineRegionBefore(op.getBodyRegion(idx), region, region.end());

      // Update region's entry blocks argument types to handle quantized element
      // types.
      if constexpr (std::is_same<DstOpT, mhlo::WhileOp>::value) {
        TypeConverter::SignatureConversion signature(num_results);
        Block &block = region.front();
        for (auto &[block_idx, original_ty] :
             llvm::enumerate(block.getArgumentTypes())) {
          TensorType updated_ty =
              UpdateElementTypeTo(original_ty, element_types[block_idx]);
          signature.addInputs(block_idx, {updated_ty});
        }
        rewriter.applySignatureConversion(&region, signature);
      }
    }
    return success();
  }
};
}  // end namespace

#include "tensorflow/compiler/mlir/xla/transforms/generated_legalize_tf.inc"
// LINT.IfChange
void PopulateLegalizeTfPatterns(MLIRContext *context,
                                RewritePatternSet *patterns) {
  populateWithGenerated(*patterns);
  // clang-format off
  patterns->add<
    ConvertAllOp,
    ConvertAnyOp,
    ConvertArgMaxOp,
    ConvertArgMinOp,
    ConvertBatchMatMulV2Op,
    ConvertBiasAddOp,
    ConvertBroadcastToOp,
    ConvertBF16FloorDivOp,
    ConvertClipByValueOp,
    ConvertConstOp,
    ConvertConv2DOp,
    ConvertConv3DOp,
    ConvertDepthConv2DOp,
    ConvertConv2DBackpropFilterOp,
    ConvertConv3DBackpropFilterOp,
    ConvertConv2DBackpropInputOp,
    ConvertConv3DBackpropInputOp,
    ConvertCumprodOp,
    ConvertCumsumOp,
    ConvertDiagPartOp,
    ConvertDynamicExpandDimsOp,
    ConvertDynamicSqueezeOp,
    ConvertEinsumOp,
    ConvertRFFTOp,
    ConvertIRFFTOp,
    ConvertFusedBatchNormGradOp,
    ConvertFusedBatchNormGradV2Op,
    ConvertFusedBatchNormGradV3Op,
    ConvertFusedBatchNormV2Op,
    ConvertFusedBatchNormV3Op,
    ConvertInfeedDequeueTupleOp,
    ConvertIdentityNOp,
    ConvertInplaceUpdateOp,
    ConvertLinSpaceOp,
    ConvertMaxOp,
    ConvertMinOp,
    ConvertAvgPool2DOp,
    ConvertAvgPool3DOp,
    ConvertAvgPool2DGradOp,
    ConvertAvgPool3DGradOp,
    ConvertMaxPool2DOp,
    ConvertMaxPool3DOp,
    ConvertMaxPool2DGradOp,
    ConvertMaxPool3DGradOp,
    ConvertMeanOp,
    ConvertOneHotOp,
    ConvertOutfeedEnqueueTupleOp,
    ConvertProdOp,
    ConvertDynamicRangeOp,
    ConvertMatrixDiagPartV3Op,
    ConvertRangeOp,
    ConvertSelectOp,
    ConvertSigmoidOp,
    ConvertShapeOp,
    ConvertSplitOp,
    ConvertSplitVOp,
    ConvertStridedSliceOp,
    ConvertStridedSliceGradOp,
    ConvertSumOp,
    ConvertTensorScatterAddOp,
    ConvertTensorScatterSubOp,
    ConvertTensorScatterMinOp,
    ConvertTensorScatterMaxOp,
    ConvertTensorScatterUpdateOp,
    ConvertTileOp,
    ConvertTopKV2Op,
    ConvertUnpackOp,
    ConvertUnsortedSegmentMaxOp,
    ConvertUnsortedSegmentMinOp,
    ConvertUnsortedSegmentProdOp,
    ConvertUnsortedSegmentSumOp,
    ConvertRandomShuffleOp,
    ConvertXlaShardingOp,
    ConvertXlaDynamicUpdateSliceOp,
    ConvertXlaConvV2Op,
    ConvertXlaReducePrecisionOp,
    ConvertXlaReduceScatterOp,
    ConvertXlaReduceWindowOp,
    ConvertXlaRngBitGeneratorOp,
    ConvertXlaSelectAndScatterOp,
    ConvertXlaSortOp,
    ConvertXlaVariadicReduceV2Op,
    ConvertXlaVariadicSortOp,
    ConvertRollOp,
    ConvertLeakyReluOp,
    ConvertLeakyReluGradOp,
    ConvertSplitOpDynamic,
    ConvertSliceOpDynamic,
    ConvertTileOpDynamic,
    ConvertUnpackOpDynamic,
    ConvertSigmoidGradOpDynamic,
    ConvertConv2DDynamic,
    ConvertPadOpDynamic,
    ConvertGatherNdOpDynamic,
    LowerControlFlowOp<TF::CaseRegionOp, mhlo::CaseOp>,
    LowerControlFlowOp<TF::IfRegionOp, mhlo::IfOp>,
    LowerControlFlowOp<TF::WhileRegionOp, mhlo::WhileOp>,
    LowerYieldOp>(context);
  // clang-format on
}
// LINT.ThenChange(:MlirPreferredOps)
}  // end namespace mhlo
}  // end namespace mlir
