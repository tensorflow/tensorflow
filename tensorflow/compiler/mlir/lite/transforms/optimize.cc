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

// This transformation pass takes operations in TensorFlowLite dialect and
// optimizes them to resulting operations in TensorFlowLite dialect.

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Optimize Pass.
namespace {
#define GEN_PASS_DEF_OPTIMIZEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

constexpr char kRelu[] = "RELU";
constexpr char kRelu6[] = "RELU6";
constexpr char kRelu1[] = "RELU_N1_TO_1";

ElementsAttr FlattenTo1D(Attribute a) {
  auto elements = a.cast<DenseElementsAttr>();
  const std::array<int64_t, 1> flattened_shape = {elements.getNumElements()};
  auto new_type = RankedTensorType::get(flattened_shape,
                                        elements.getType().getElementType());
  return elements.reshape(new_type);
}

bool L2NormalizeReduceAxis(Value sq_op, DenseElementsAttr axis) {
  if (axis.getNumElements() == 0) {
    return false;
  }
  if (sq_op.getType().cast<ShapedType>().getRank() - 1 ==
          *axis.getValues<int>().begin() ||
      *axis.getValues<int>().begin() == -1) {
    return true;
  }
  if (sq_op.getType().cast<ShapedType>().getRank() != axis.getNumElements()) {
    return false;
  }
  auto shape = sq_op.getType().cast<ShapedType>();
  SmallVector<int, 4> elems{axis.getValues<int>().begin(),
                            axis.getValues<int>().end()};
  for (int i = 0; i < shape.getRank(); ++i) {
    if (i != elems[i]) return false;
  }
  return true;
}

using ::llvm::cast;

// Optimize TFLite operations in functions.
class OptimizePass : public impl::OptimizePassBase<OptimizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizePass)

  OptimizePass() = default;
  OptimizePass(const OptimizePass &) {}
  explicit OptimizePass(bool enable_canonicalization,
                        bool disable_fuse_mul_and_fc = false) {
    this->enable_canonicalization_ = enable_canonicalization;
    this->disable_fuse_mul_and_fc_ = disable_fuse_mul_and_fc;
  }

  void runOnOperation() override;
};

// Returns whether the given type `a` is broadcast-compatible with `b`.
bool IsBroadcastableElementsAttrAndType(Type a, Type b) {
  return OpTrait::util::getBroadcastedType(a, b) != Type();
}

// Returns whether the resultant type of any broadcastable operation with
// operands `a` and `b` matches `expected_output`. Returns false if `a` is not
// broadcast-compatible with `b`.
bool OperandsBroadcastToOutputType(Type a, Type b, Type expected_output) {
  Type output_element_type =
      expected_output.cast<ShapedType>().getElementType();
  Type broadcasted_type =
      OpTrait::util::getBroadcastedType(a, b, output_element_type);
  return broadcasted_type != Type() && broadcasted_type == expected_output;
}

// Returns whether if `type1` dimensions are the same as the ending dimensions
// of `type2`. This is more restricted than broadcastable.
bool IsTailOfShape(Type type1, Type type2) {
  auto tail_type = type1.dyn_cast<ShapedType>();
  auto full_type = type2.dyn_cast<ShapedType>();
  if (!tail_type || !full_type || !tail_type.hasRank() ||
      !full_type.hasRank() || tail_type.getRank() > full_type.getRank())
    return false;
  auto i1 = tail_type.getShape().rbegin(), e1 = tail_type.getShape().rend();
  auto i2 = full_type.getShape().rbegin();
  return std::equal(i1, e1, i2);
}

// This function removes explicit broadcasting on type1 and returns whether if
// the reduced `type1` dimensions are the same as the ending dimensions
// of `type2`.
bool IsReducedTailOfShape(Type type1, Type type2) {
  auto tail_type = type1.dyn_cast<ShapedType>();
  auto full_type = type2.dyn_cast<ShapedType>();
  if (!tail_type || !full_type || !tail_type.hasRank() || !full_type.hasRank())
    return false;

  auto i1 = tail_type.getShape().rbegin();
  auto reduced_e1 = tail_type.getShape().rend();
  auto i2 = full_type.getShape().rbegin();

  while ((std::distance(i1, reduced_e1) > 0) && (*(reduced_e1 - 1) == 1)) {
    reduced_e1--;
  }

  return (std::distance(i1, reduced_e1) > 0) &&
         (std::distance(i1, reduced_e1) <= full_type.getRank()) &&
         (std::equal(i1, reduced_e1, i2));
}

// Check if the value of the last dimension of type1 is equal to the number of
// elements in type2. This is a required condition to flatten type2 to form a
// 1D array and allow the binaryOp handle the broadcasting implicitly.
bool IsLastDimEqualToNumElements(Type type1, Type type2) {
  return (type1.cast<ShapedType>().getRank() >= 1 &&
          type1.cast<ShapedType>().getDimSize(
              type1.cast<ShapedType>().getRank() - 1) ==
              type2.cast<ShapedType>().getNumElements());
}

bool CanFuseConvOrDepthwiseConvShapes(const ArrayRef<int64_t> filter_shape,
                                      const ArrayRef<int64_t> elements_shape,
                                      bool is_depthwise) {
  // Val tensor must be a scalar or of a shape [1, ... , 1, elements_depth].
  const int elements_rank = elements_shape.size();
  for (int i = 0; i < elements_rank - 1; ++i) {
    if (elements_shape[i] != 1) {
      return false;
    }
  }

  auto elements_depth = elements_shape.empty() ? 1 : elements_shape.back();
  // If elements depth equals 1 (i.e., scalar or tensor with 1 element), then we
  // can let binary op to broadcast elements.
  if (elements_depth == 1) {
    return true;
  }

  // In TFLite Conv2D uses OHWI format for filter, and 1HWO for Depthwise Conv.
  // For conv:
  // Check if last dimension in filter equals the first dimension
  // For depthwise conv:
  // Check if the first in filter dimension equals the first dimension.
  if (filter_shape.empty() ||
      (is_depthwise ? filter_shape.back() != elements_depth
                    : filter_shape[0] != elements_depth))
    return false;
  return true;
}

bool CanFuseConvOrDepthwiseConv(Value filter, Attribute val,
                                bool is_depthwise) {
  const auto elements = val.dyn_cast<DenseElementsAttr>();
  if (!elements) {
    return false;
  }
  const auto elements_shape = elements.getType().getShape();
  const auto filter_shape = filter.getType().cast<ShapedType>().getShape();
  return CanFuseConvOrDepthwiseConvShapes(filter_shape, elements_shape,
                                          is_depthwise);
}

bool CanFuseConvOrDepthwiseConv(Attribute filter, Attribute val,
                                bool is_depthwise) {
  if (const auto elements = val.dyn_cast<DenseElementsAttr>()) {
    if (const auto filter_elements = filter.dyn_cast<DenseElementsAttr>()) {
      return CanFuseConvOrDepthwiseConvShapes(
          filter_elements.getType().getShape(), elements.getType().getShape(),
          is_depthwise);
    }
  }
  return false;
}

// Retuns true if we can eliminate the GatherNdOp or ScatterNdOp. When the value
// of `indices` are from 0 to n-1, the output tensor are identical to the
// `params`.
bool CanOptimizeIdentityGatherNdOrScatterNdOp(Value params,
                                              DenseIntElementsAttr indices,
                                              Type output_type) {
  auto params_type = params.getType().dyn_cast<RankedTensorType>();
  auto indices_type = indices.getType().dyn_cast<RankedTensorType>();
  // Checks the shape of `params` is [n, ...], shape of `indices` is [n, 1]. 2D
  // `indices` means it gets the first row of `params`. As long as indices
  // iterate the first row of `params`, the output is identical to input.
  if (!params_type || !indices_type || indices_type.getRank() != 2 ||
      indices_type.getDimSize(0) != params_type.getDimSize(0) ||
      indices_type.getDimSize(1) != 1)
    return false;

  // Checks the `params_type` is equal to `output_type`. If not equal, we
  // cannot replace the scatter_nd/gather_nd op with `params`.
  if (params_type != output_type) return false;

  // Checks the value in `indices` is from 0 to n-1.
  int cur_value = 0;
  for (const auto &v : indices.getValues<APInt>()) {
    if (v.getSExtValue() != cur_value) return false;
    ++cur_value;
  }

  return true;
}

// Returns true if we can eliminate the SliceOp. When the values of `begin` are
// all 0s and `size[i]` is equal to either -1 or `input.shape[i]`
// for each dim i, the output tensor is identical to `input`.
bool CanOptimizeIdentitySliceOp(Value input, Attribute begin, Attribute size) {
  // Checks if `begin` and `size` are i32 or i64.
  auto begin_attr = begin.dyn_cast<DenseIntElementsAttr>();
  auto size_attr = size.dyn_cast<DenseIntElementsAttr>();
  if (!begin_attr || !size_attr) {
    return false;
  }

  auto begin_elem_ty = begin_attr.getType().getElementType();
  if (!begin_elem_ty.isInteger(32) && !begin_elem_ty.isInteger(64)) {
    return false;
  }
  auto size_elem_ty = size_attr.getType().getElementType();
  if (!size_elem_ty.isInteger(32) && !size_elem_ty.isInteger(64)) {
    return false;
  }

  // Checks if `input` is ranked and its rank is equal to number of elements in
  // `begin` and `size`.
  auto input_ty = input.getType().cast<ShapedType>();
  if (!input_ty.hasRank()) {
    return false;
  }

  int64_t rank = input_ty.getRank();
  if (rank != begin_attr.getNumElements() ||
      rank != size_attr.getNumElements()) {
    return false;
  }

  // Checks if `begin` is all 0s, and `size[i]` is equal to either -1 or
  // `input.shape[i]`.
  for (uint64_t i = 0; i < rank; ++i) {
    if (begin_attr.getValues<APInt>()[i].getSExtValue() != 0) return false;
    int64_t si = size_attr.getValues<APInt>()[i].getSExtValue();
    if (si != -1 && si != input_ty.getDimSize(i)) return false;
  }

  return true;
}

// Expand Attribute 'a' to 4D with all 1s except 1 dimension.
// Which dimension depends on 'is_depthwise' is true or false.
ElementsAttr ExpandTo4DForConvImpl(Attribute a, bool is_depthwise) {
  auto elements = a.dyn_cast<DenseElementsAttr>();
  auto shape = elements.getType().getShape();
  if (!shape.empty()) {
    // Checks that elements are essentially 1d.
    assert(elements.getNumElements() == shape.back());
  }
  std::vector<int64_t> shape_data = {1, 1, 1, 1};
  const int vector_length = elements.getNumElements();
  if (is_depthwise)
    shape_data[3] = vector_length;
  else
    shape_data[0] = vector_length;
  auto new_shape =
      RankedTensorType::get(shape_data, elements.getType().getElementType());
  return elements.reshape(new_shape);
}

ElementsAttr ExpandTo4DForConv(Attribute a) {
  return ExpandTo4DForConvImpl(a, false);
}

ElementsAttr ExpandTo4DForDepthwiseConv(Attribute a) {
  return ExpandTo4DForConvImpl(a, true);
}

TypeAttr RescaleQtype(Type input, Attribute factor) {
  return quant::RescaleQuantizedType(input, factor);
}

// Returns shape of a ranked tensor.
// Precondition: output_val's is ranked tensor.
DenseElementsAttr GetShape(Value output_val) {
  auto output_type = output_val.getType().cast<RankedTensorType>();

  SmallVector<int32_t> shape;
  shape.reserve(output_type.getRank());
  for (int64_t dim : output_type.getShape()) {
    shape.push_back(ShapedType::isDynamic(dim) ? -1
                                               : static_cast<int32_t>(dim));
  }
  return mlir::DenseElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int>(shape.size())},
          mlir::IntegerType::get(output_val.getContext(), 32)),
      llvm::ArrayRef(shape));
}

// Returns `true` if reducing `axes` in `input` with `keep_dims=true` results in
// the specified `shape` and `false` otherwise.
static bool ShapeMatchesReduceWithKeepAxes(Value input,
                                           const mlir::Attribute &axes,
                                           const mlir::Attribute &shape) {
  RankedTensorType type = input.getType().dyn_cast_or_null<RankedTensorType>();
  if (!type) return false;

  DenseIntElementsAttr axes_attr =
      axes.dyn_cast_or_null<DenseIntElementsAttr>();
  DenseIntElementsAttr shape_attr =
      shape.dyn_cast_or_null<DenseIntElementsAttr>();
  if (!axes_attr || !shape_attr) return false;

  if (shape_attr.getNumElements() != type.getRank()) return false;

  llvm::SmallSet<uint64_t, 4> axes_set;
  for (auto a : axes_attr.getValues<APInt>()) {
    axes_set.insert(a.getZExtValue());
  }

  auto type_shape = type.getShape();
  for (uint64_t i = 0; i < type.getRank(); ++i) {
    if (axes_set.contains(i)) {
      if (shape_attr.getValues<APInt>()[i] != 1) return false;
    } else {
      if (shape_attr.getValues<APInt>()[i] != type_shape[i]) return false;
    }
  }
  return true;
}

// Returns `true` if all the `axes` dimensions of `input` are 1.
static bool AreInputDimensionsOneInAxes(Value input,
                                        const mlir::Attribute &axes) {
  RankedTensorType input_type =
      input.getType().dyn_cast_or_null<RankedTensorType>();
  if (!input_type) return false;
  auto type_shape = input_type.getShape();

  DenseIntElementsAttr axes_attr =
      axes.dyn_cast_or_null<DenseIntElementsAttr>();
  if (!axes_attr) return false;

  for (auto a : axes_attr.getValues<APInt>()) {
    int64_t axis = a.getSExtValue();
    if (axis < 0) {
      axis += type_shape.size();
    }
    if (axis < 0 || axis >= type_shape.size()) {
      // `axis` is not a valid axis in input.
      return false;
    }
    if (type_shape[axis] != 1) {
      return false;
    }
  }

  return true;
}

static bool FloatValueEquals(const Attribute &attr, double value) {
  auto fp_attr = attr.dyn_cast_or_null<DenseFPElementsAttr>();
  if (!fp_attr) return false;

  if (fp_attr.isSplat()) {
    return fp_attr.getSplatValue<APFloat>().isExactlyValue(value);
  }
  return llvm::all_of(fp_attr.getValues<APFloat>(), [value](const APFloat &f) {
    return f.isExactlyValue(value);
  });
}

// Returns true if `value` is compile-time constant and its splat value equals
// to `raw_value`.
template <typename T>
bool IsConstantValueOf(mlir::TypedAttr value, T raw_value) {
  auto element_type = value.getType().cast<ShapedType>().getElementType();

  if (element_type.isa<FloatType>()) {
    return FloatValueEquals(value, raw_value);
  } else if (element_type.isa<IntegerType>()) {
    auto int_attr = value.dyn_cast_or_null<DenseIntElementsAttr>();
    if (!int_attr) return false;

    if (int_attr.isSplat()) {
      return int_attr.getSplatValue<APInt>() == raw_value;
    }
    return llvm::all_of(int_attr.getValues<APInt>(),
                        [raw_value](const APInt &f) { return f == raw_value; });
  }

  return false;
}

// Returns true if the value's element type is F32.
bool IsF32Value(Value value) {
  return value.getType().cast<ShapedType>().getElementType().isF32();
}

// Returns the number of elements in attr if it is a static shape, 1 otherwise,
// as an unranked int32 Attribute.
TypedAttr GetNumElementsOrOne(Type type) {
  auto shaped_type = type.cast<ShapedType>();
  int32_t num_elements =
      shaped_type.hasStaticShape() ? shaped_type.getNumElements() : 1;

  OpBuilder builder(type.getContext());

  return DenseIntElementsAttr::get(
      RankedTensorType::get({}, builder.getI32Type()),
      {llvm::APInt(32, num_elements, true)});
}

// Reshapes value to a given shape.
Value ReshapeValueDroppingLastDim(OpBuilder &builder, Value value) {
  // This function is always guarded with HasTrivialShapeExceptSecondLastDim(),
  // so we could cast safely here.
  auto type = value.getType().cast<ShapedType>();
  SmallVector<int> new_shape;
  for (int64_t dim : type.getShape().drop_back()) {
    new_shape.push_back(dim);
  }
  return builder.create<ReshapeOp>(
      value.getLoc(), value,
      builder.create<arith::ConstantOp>(
          value.getLoc(),
          DenseIntElementsAttr::get(
              RankedTensorType::get(type.getRank() - 1, builder.getI32Type()),
              new_shape)));
}

// Returns true if val has a static shape and the last dimension equals 1.
bool IsLastDimensionEqualOne(Value val) {
  const auto val_type = val.getType().cast<ShapedType>();
  if (!val_type.hasStaticShape()) return false;
  const auto val_shape = val_type.getShape();
  if (val_shape.empty()) return false;
  const auto last_element = *val_shape.rbegin();
  return last_element == 1;
}

// Returns true if the supplied value-
// 1) Has only one use or
// 2) Is only used by binary op like AddOp, SubOp, MulOp or DivOp.
bool HasOneUseOrUsedByOnlyBinaryOps(Value out_value) {
  if (out_value.hasOneUse()) {
    return true;
  }

  for (auto &use : out_value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (!llvm::isa<mlir::TFL::AddOp>(owner) &&
        !llvm::isa<mlir::TFL::SubOp>(owner) &&
        !llvm::isa<mlir::TFL::DivOp>(owner) &&
        !llvm::isa<mlir::TFL::MulOp>(owner)) {
      return false;
    }
  }

  return true;
}

// Returns true if attr is a DenseIntElementsAttr of int32 or int64 values or an
// incrementing sequence from 0 to N-1.
//
// If such a value is used in an Equal operator, it can be replaced with OneHot.
bool IsOneHotIndexAttribute(Attribute attr) {
  const auto dense_attr = attr.dyn_cast_or_null<DenseIntElementsAttr>();
  if (!dense_attr) {
    return false;
  }
  auto index_type = dense_attr.getType();
  const auto index_elem_bits = index_type.getElementTypeBitWidth();
  if (index_elem_bits != 32 && index_elem_bits != 64) {
    return false;
  }
  // Checks that the index has shape of [1, 1, 1, ..., 1, N].
  if (index_type.getRank() < 1 ||
      llvm::any_of(index_type.getShape().drop_back(),
                   [](int64_t dim) { return dim != 1; })) {
    return false;
  }
  const auto elems = dense_attr.value_begin<APInt>();
  for (int i = 0; i < dense_attr.getNumElements(); ++i) {
    if (i != elems[i]) {
      return false;
    }
  }
  return true;
}

Value Get1DShapeValue(OpBuilder &builder, Value value) {
  auto type = value.getType().cast<ShapedType>();
  if (!type.hasStaticShape()) {
    return nullptr;
  }
  auto output_type = RankedTensorType::get({1}, builder.getI32Type());
  const int num_elements = type.getNumElements();
  return builder.create<ConstOp>(
      value.getLoc(), output_type,
      DenseIntElementsAttr::get(output_type, num_elements));
}

Type GetEmbeddingLookupShape(Value lookup, Value value) {
  auto lookup_type = lookup.getType().cast<ShapedType>();
  if (!lookup_type.hasStaticShape()) {
    return nullptr;
  }
  auto value_type = value.getType().cast<ShapedType>();
  if (!value_type.hasStaticShape() || value_type.getRank() != 2) {
    return nullptr;
  }
  SmallVector<int64_t> new_shape = {lookup_type.getNumElements(),
                                    value_type.getDimSize(0)};
  return value_type.clone(new_shape);
}

// Creates FullyConnected op from params and returns the output.
mlir::Value GetFcOutput(OpBuilder *builder,
                        ::mlir::Operation::result_range result, Value input,
                        Value filter, Value bias,
                        StringAttr fused_activation_function,
                        StringAttr weights_format, BoolAttr keep_num_dims,
                        BoolAttr asymmetric_quantize_inputs) {
  auto fc_op = builder->create<FullyConnectedOp>(
      result[0].getLoc(), result.getTypes(), input, filter, bias,
      fused_activation_function, weights_format, keep_num_dims,
      asymmetric_quantize_inputs);
  return fc_op->getResult(0);
}

// Returns true if 'value' represents a const ElementsAttr with all values
// equals to 0.0.
bool AllValuesAreZero(mlir::Value value) {
  if (!value) return false;
  DenseElementsAttr vals;
  if (!matchPattern(value, m_Constant(&vals))) return false;
  for (auto elem : vals.getValues<float>())
    if (elem != 0.0f) return false;
  return true;
}

// Converts an Attribute with a single value of float or integral type to an
// Attribute holding a single value of float type. If attr has no elements, the
// result is 0.0f.
TypedAttr ConvertSingleElementAttrToFloatAttr(Attribute attr) {
  const auto dense_fp_attr = attr.dyn_cast_or_null<DenseFPElementsAttr>();
  if (dense_fp_attr) {
    // Already float => return
    return dense_fp_attr;
  }

  OpBuilder builder(attr.getContext());

  const auto dense_int_attr = attr.dyn_cast<DenseIntElementsAttr>();
  const auto int_values = dense_int_attr.getValues<APInt>();
  float float_val = 0.0f;
  if (!int_values.empty()) {
    const APInt apint_val = *int_values.begin();
    if (dense_int_attr.getType().getElementType().isSignedInteger()) {
      // Get the sign-extended value (=>int64) if the type is signed.
      float_val = apint_val.getSExtValue();
    } else {
      // Get the zero-extended value (=>uint64) if unsigned or signless.
      float_val = apint_val.getZExtValue();
    }
  }
  return DenseFPElementsAttr::get(
      RankedTensorType::get({}, builder.getF32Type()),
      {llvm::APFloat(float_val)});
}

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize.inc"

// Fuse Add with proceeding FullyConnected.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseFullyConnectedAndAdd : public OpRewritePattern<TFL::AddOp> {
  using OpRewritePattern<TFL::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::AddOp add_op,
                                PatternRewriter &rewriter) const override {
    // Match Add.
    DenseElementsAttr added_value;
    Value constant_val = add_op.getRhs();
    if (!matchPattern(constant_val, m_Constant(&added_value))) return failure();

    // Match Fully Connected.
    auto fc_op = dyn_cast_or_null<TFL::FullyConnectedOp>(
        add_op.getLhs().getDefiningOp());
    if (!fc_op) return failure();

    // Check if the constant RHS is either 0D (scalar), or a 1D with
    // `{num_channels}` shape.
    auto constant_val_type = constant_val.getType().cast<TensorType>();

    // In TFLite FullyConnect definition, bias must be a 1D tensor where
    // the number of elements is equal to the number of channels.
    // If it's not 1D or 0D (which can be broadcasted to 1D), reject the
    // matching.
    bool is_scalar_rhs = false;
    if (constant_val_type.getRank() == 0) {
      is_scalar_rhs = true;
    }

    Value filter = fc_op.getFilter();
    Value bias = fc_op.getBias();
    ElementsAttr bias_value;
    const bool is_none_bias = bias.getType().isa<NoneType>();
    if (fc_op.getFusedActivationFunction() != "NONE") return failure();

    if (!is_none_bias && !matchPattern(bias, m_Constant(&bias_value)))
      return failure();

    // Rewrite
    if (is_none_bias) {
      if (constant_val_type.getRank() == 1) {
        // If there no pre-existing bias and the `constant_val` is 1D, simply
        // use `constant_val` as bias.
        bias = constant_val;
      } else {
        if (!is_scalar_rhs &&
            !(IsReducedTailOfShape(constant_val.getType(), filter.getType()) &&
              IsLastDimEqualToNumElements(filter.getType(),
                                          constant_val.getType()))) {
          return failure();
        }

        // If the `constant_val` is scalar, we must the shape of filter
        // to properly broadcast the scalar to `{num_channels}` shape.

        // Get the number of channels if possible.
        auto filter_type = filter.getType().dyn_cast<RankedTensorType>();
        // Filter must be a `2D` tensor with `{num_channels, num_features}`
        // shape. The following check is rejecting unknown rank (-1).
        if (filter_type == nullptr || filter_type.getRank() != 2) {
          return failure();
        }
        int num_channels = filter_type.getShape()[0];

        // Create a zero tensor with shape {num_channels}, and the type need
        // to be the same as constant_val. This is a way to gracefully handle
        // scalar tensor. The Add will always be constant-folded away
        // regardless if `constant_val` is a scalar or not.
        RankedTensorType type = RankedTensorType::get(
            {num_channels}, constant_val_type.getElementType());
        auto attr = rewriter.getZeroAttr(type);
        bias = rewriter.create<arith::ConstantOp>(add_op.getLoc(), type, attr);
        auto none_af = rewriter.getStringAttr("NONE");
        if (is_scalar_rhs) {
          bias =
              rewriter
                  .create<AddOp>(add_op.getLoc(), bias, constant_val, none_af)
                  .getOutput();
        } else {
          // If the RHS is neither a scalar constant nor a 1d constant, look
          // if there is opportunity to reduce the dimentionality and allow
          // implicit broadcasting

          auto new_added_value = added_value.reshape(RankedTensorType::get(
              {added_value.getType().cast<ShapedType>().getNumElements()},
              added_value.getType().cast<ShapedType>().getElementType()));

          ::mlir::arith::ConstantOp new_constant_val =
              rewriter.create<::mlir::arith::ConstantOp>(
                  add_op.getLoc(),
                  /*value=*/new_added_value);

          bias = rewriter
                     .create<::mlir::TFL::AddOp>(
                         add_op.getLoc(),
                         /*lhs=*/bias,
                         /*rhs=*/new_constant_val.getResult(),
                         /*fused_activation_function=*/none_af)
                     .getOutput();
        }
      }
    } else {
      bias = rewriter
                 .create<AddOp>(add_op.getLoc(), bias, constant_val,
                                rewriter.getStringAttr("NONE"))
                 .getOutput();
    }

    auto fc = rewriter.create<TFL::FullyConnectedOp>(
        FusedLoc::get(fc_op.getContext(), {fc_op.getLoc(), add_op.getLoc()}),
        add_op.getType(),
        /*input=*/fc_op.getInput(),
        /*filter=*/filter,
        /*bias=*/bias,
        /*fused_activation_function=*/
        rewriter.getStringAttr(add_op.getFusedActivationFunction()),
        /*weights_format=*/rewriter.getStringAttr(fc_op.getWeightsFormat()),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.getKeepNumDims()),
        /*asymmetric_quantize_inputs=*/
        fc_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(add_op, fc.getOutput());

    return success();
  }
};

// Replace ..
// FC(Add(lhs, rhs), filter, bias)
// .. with ..
// FC(lhs, filter, FC(rhs, filter, bias))
// .. if rhs, filter, and bias are all constants.
// The second FC will be constant folded to a single vector.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseAddAndFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc_op,
                                PatternRewriter &rewriter) const override {
    // This only works with default format.
    if (fc_op.getWeightsFormat() != "DEFAULT") return failure();

    // Match Add.
    auto add_op =
        dyn_cast_or_null<TFL::AddOp>(fc_op.getInput().getDefiningOp());
    if (!add_op) return failure();
    if (add_op.getFusedActivationFunction() != "NONE") return failure();

    // Don't match adds where the added constant is not 1D.
    {
      auto addend_shape = add_op.getRhs().getType().cast<ShapedType>();
      if (!addend_shape.hasStaticShape()) return failure();
      if (addend_shape.getShape().size() != 1) return failure();
    }

    // Calculate new bias.  Generate a new FC; it will be constant folded.
    auto old_bias = fc_op.getBias();
    if (!old_bias || old_bias.getType().isa<NoneType>()) {
      // TODO(b/180752069): Figure out new bias' type when old bias is empty.
      return failure();
    }

    // The FC relies on constant folding, which is implemented on F32. Checks
    // types to be F32.
    {
      if (!IsF32Value(add_op.getRhs()) || !IsF32Value(fc_op.getFilter()) ||
          !IsF32Value(old_bias))
        return failure();
    }

    auto new_bias = rewriter.create<TFL::FullyConnectedOp>(
        fc_op.getLoc(), old_bias.getType(),
        /*input=*/add_op.getRhs(),
        /*filter=*/fc_op.getFilter(),
        /*bias=*/old_bias,
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(true),
        /*asymmetric_quantize_inputs=*/fc_op.getAsymmetricQuantizeInputsAttr());

    // Create the updated FC.
    auto new_fc = rewriter.create<TFL::FullyConnectedOp>(
        FusedLoc::get(add_op.getContext(), {add_op.getLoc(), fc_op.getLoc()}),
        fc_op.getOutput().getTypes(),
        /*input=*/add_op.getLhs(),
        /*filter=*/fc_op.getFilter(),
        /*bias=*/*new_bias.getOutput().begin(),
        /*fused_activation_function=*/
        rewriter.getStringAttr(fc_op.getFusedActivationFunction()),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.getKeepNumDims()),
        /*asymmetric_quantize_inputs=*/fc_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(fc_op.getOperation(), new_fc.getOutput());

    return success();
  }
};

// Replace ..
// FC(Mul(lhs, rhs), filter, bias)
// .. with ..
// FC(lhs, Mul(filter, rhs), bias)
// .. if rhs, filter, and bias are all constants.
// The generated Mul will be constant folded to a single matrix.
struct FuseMulAndFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc_op,
                                PatternRewriter &rewriter) const override {
    // This only works with default format.
    if (fc_op.getWeightsFormat() != "DEFAULT") return failure();

    // Match Mul.
    auto mul_op =
        dyn_cast_or_null<TFL::MulOp>(fc_op.getInput().getDefiningOp());
    if (!mul_op) return failure();
    if (mul_op.getFusedActivationFunction() != "NONE") return failure();

    // Don't match muls where the multiplier constant is not 1D.
    {
      auto multiplier_shape = mul_op.getRhs().getType().cast<ShapedType>();
      if (!multiplier_shape.hasStaticShape()) return failure();
      if (multiplier_shape.getShape().size() != 1) return failure();
    }

    // We rely on constant folding, implemented only for F32. Check types.
    if (!IsF32Value(mul_op.getRhs()) || !IsF32Value(fc_op.getFilter())) {
      return failure();
    }

    auto location =
        FusedLoc::get(mul_op.getContext(), {mul_op.getLoc(), fc_op.getLoc()});

    auto new_filter = rewriter.create<TFL::MulOp>(
        location,
        /*lhs=*/fc_op.getFilter(),
        /*rhs=*/mul_op.getRhs(),
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"));
    // Create the updated FC.
    auto new_fc = rewriter.create<TFL::FullyConnectedOp>(
        location, fc_op.getOutput().getTypes(),
        /*input=*/mul_op.getLhs(),
        /*filter=*/new_filter,
        /*bias=*/fc_op.getBias(),
        /*fused_activation_function=*/
        rewriter.getStringAttr(fc_op.getFusedActivationFunction()),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.getKeepNumDims()),
        /*asymmetric_quantize_inputs=*/fc_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(fc_op.getOperation(), new_fc.getOutput());

    return success();
  }
};

// TODO(b/136285429): Move to tablegen when variadic is supported.
template <typename ReluXOp, char const *Act>
struct FuseFullyConnectedAndReluX : public OpRewritePattern<ReluXOp> {
  using OpRewritePattern<ReluXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluXOp relu_op,
                                PatternRewriter &rewriter) const override {
    Operation *input = relu_op.getOperand().getDefiningOp();
    if (!isa_and_nonnull<FullyConnectedOp>(input)) return failure();
    auto fully_connected_op = cast<FullyConnectedOp>(input);
    if (fully_connected_op.getFusedActivationFunction() != "NONE")
      return failure();

    auto new_activation_func = rewriter.getStringAttr(Act);
    auto new_weights_format =
        rewriter.getStringAttr(fully_connected_op.getWeightsFormat());
    auto new_keep_num_dims =
        rewriter.getBoolAttr(fully_connected_op.getKeepNumDims());
    auto fc = rewriter.create<FullyConnectedOp>(
        FusedLoc::get(relu_op.getContext(),
                      {fully_connected_op.getLoc(), relu_op.getLoc()}),
        relu_op.getType(), /*input=*/fully_connected_op.getInput(),
        /*filter=*/fully_connected_op.getFilter(),
        /*bias=*/fully_connected_op.getBias(),
        /*fused_activation_function=*/new_activation_func,
        /*weights_format=*/new_weights_format,
        /*keep_num_dims=*/new_keep_num_dims,
        /*asymmetric_quantize_inputs=*/
        fully_connected_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(relu_op, fc.getOutput());

    return success();
  }
};

// Fuse Mul with proceeding FullyConnected.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseFullyConnectedAndMul : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mul_op,
                                PatternRewriter &rewriter) const override {
    // If we are broadcasting on the lhs then don't fold the multiply as it
    // would increase the amount of compute done by the fully connected op.
    if (mul_op.getLhs().getType() != mul_op.getType()) return failure();

    // Mul.
    DenseElementsAttr cst;
    Value constant_val = mul_op.getRhs();
    if (!matchPattern(constant_val, m_Constant(&cst))) return failure();

    // Fully Connected.
    auto fc_op = dyn_cast_or_null<TFL::FullyConnectedOp>(
        mul_op.getLhs().getDefiningOp());
    if (!fc_op) return failure();
    Value filter = fc_op.getFilter();
    Value bias = fc_op.getBias();
    ElementsAttr cst_tmp;
    if (!matchPattern(filter, m_Constant(&cst_tmp))) return failure();
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&cst_tmp)))
      return failure();
    if (fc_op.getFusedActivationFunction() != "NONE") return failure();

    // Only fuse multiplier if all dimensions other than the depth dimension
    // are equal to 1 since otherwise
    // `matmul(x, filter) * cst != matmul(x, filter * cst)`
    // even if `filter` and `cst` are be broadcastable.
    auto shape = cst.getType().getShape();
    if (!IsDimensionsDegenerateExceptLastOne(shape)) return failure();

    int64_t element_size = shape.empty() ? 1 : shape[shape.size() - 1];
    // Expand and transpose the multiplier since weights are using the
    // OHWI data format in TFLite.
    int64_t normalized_shape[2] = {element_size, 1};
    auto new_cst = cst.reshape(RankedTensorType::get(
        normalized_shape, cst.getType().getElementType()));
    Type new_type = new_cst.getType();
    if (!IsBroadcastableElementsAttrAndType(new_type, filter.getType())) {
      return failure();
    }

    auto new_op =
        rewriter.create<arith::ConstantOp>(mul_op.getLoc(), new_type, new_cst);
    Value new_const_val = new_op.getResult();

    // Rewrite. Since the folder of TFL::MulOp couldn't broadcast the operands,
    // TF::MulOp is used to fold the constant.
    // TODO(b/139192933): switch to the TFL constant folding
    auto new_filter =
        rewriter.create<TF::MulOp>(mul_op.getLoc(), filter, new_const_val)
            .getZ();
    // If bias isn't None, it needs to be multiplied as well.
    if (!bias.getType().isa<NoneType>()) {
      bias = rewriter.create<TF::MulOp>(mul_op.getLoc(), bias, constant_val)
                 .getZ();
    }

    auto fc = rewriter.create<TFL::FullyConnectedOp>(
        FusedLoc::get(fc_op.getContext(), {fc_op.getLoc(), mul_op.getLoc()}),
        mul_op.getType(),
        /*input=*/fc_op.getInput(),
        /*filter=*/new_filter,
        /*bias=*/bias,
        /*fused_activation_function=*/
        rewriter.getStringAttr(mul_op.getFusedActivationFunction()),
        /*weights_format=*/rewriter.getStringAttr(fc_op.getWeightsFormat()),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.getKeepNumDims()),
        /*asymmetric_quantize_inputs=*/fc_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(mul_op, fc.getOutput());

    return success();
  }
};

// Fuse Mul with proceeding Affine ops. This is an C++ implementation of the
// following table gen implementation, which doesn't derived the result type of
// the TFL_DequantizeOp.
// def : Pat<(TFL_MulOp (TFL_Conv2DOp:$conv_output $input,
//                          (TFL_DequantizeOp (TFL_QuantizeOp
//                              (Arith_ConstantOp F32ElementsAttr:$filter),
//                              $qtype)),
//                          (Arith_ConstantOp F32ElementsAttr:$bias),
//                          $h_factor, $w_factor, TFL_AF_None,
//                          $padding, $stride_h, $stride_w),
//                      (Arith_ConstantOp F32ElementsAttr:$value), $act_fn),
//           (TFL_Conv2DOp $input,
//                      (TFL_DequantizeOp (TFL_QuantizeOp
//                          (TFL_MulOp (Arith_ConstantOp $filter),
//                                     (Arith_ConstantOp (ExpandTo4DForConv
//                                     $value)),
//                                      TFL_AF_None),
//                          (RescaleQtype $qtype, $value))),
//                      (TFL_MulOp (Arith_ConstantOp $bias), (Arith_ConstantOp
//                      $value),
//                          TFL_AF_None),
//                      $h_factor, $w_factor, $act_fn,
//                      $padding, $stride_h, $stride_w),
//         [(CanFuseConvOrDepthwiseConv<"false"> $filter, $value),
//          (HasOneUse $conv_output),
//          (IsPerAxisQuantization $qtype), // per-axis quantization
//         ]>;
template <typename AffineOpType>
struct FuseAffinOpAndMulWithQDQs : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mul_op,
                                PatternRewriter &rewriter) const override {
    // Mul. Required 1-D rhs for batch normalization.
    DenseElementsAttr gamma_cst;
    Value gamma = mul_op.getRhs();
    if (!matchPattern(gamma, m_Constant(&gamma_cst))) return failure();
    if (gamma_cst.getType().getRank() != 1) return failure();

    // Affine op
    Operation *mul_op_lhs = mul_op.getLhs().getDefiningOp();
    auto fc_op = dyn_cast_or_null<AffineOpType>(mul_op_lhs);
    if (!fc_op) return failure();
    Value filter = fc_op.getFilter();
    Value bias = fc_op.getBias();

    // QDQs
    auto dq_op = dyn_cast_or_null<TFL::DequantizeOp>(filter.getDefiningOp());
    if (!dq_op) return failure();
    auto q_op =
        dyn_cast_or_null<TFL::QuantizeOp>(dq_op.getInput().getDefiningOp());
    if (!q_op) return failure();
    filter = q_op.getInput();

    // weight constant
    ElementsAttr cst_tmp;
    if (!matchPattern(filter, m_Constant(&cst_tmp))) return failure();
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&cst_tmp)))
      return failure();
    if (fc_op.getFusedActivationFunction() != "NONE") return failure();

    // Broadcast the constant operand of Mul if it isn't compatible to the
    // filter input. We only support broadcasting the operand along the depth
    // dimension, when the operand's depth is 1.
    rewriter.setInsertionPoint(q_op);
    Location loc = fc_op.getLoc();
    Value broadcasted_gamma;
    if (isa<TFL::Conv2DOp>(mul_op_lhs)) {
      auto mul_rhs = ExpandTo4DForConv(gamma_cst);
      broadcasted_gamma = rewriter.create<ConstOp>(loc, mul_rhs);
    } else if (isa<TFL::DepthwiseConv2DOp>(mul_op_lhs)) {
      auto mul_rhs = ExpandTo4DForDepthwiseConv(gamma_cst);
      broadcasted_gamma = rewriter.create<ConstOp>(loc, mul_rhs);
    } else {
      return failure();
    }

    // Make sure that the fused bias will be a 1D tensor.
    auto gamma_shape = gamma.getType().cast<ShapedType>();
    if (!gamma_shape.hasRank() || gamma_shape.getRank() != 1) {
      return failure();
    }

    // Rewrite filter constant. Since the folder of TFL::MulOp couldn't
    // broadcast the operands, TF::MulOp is used to fold the constant.
    auto new_filter =
        rewriter.create<TF::MulOp>(loc, filter, broadcasted_gamma).getZ();
    // Update the scale in the quantize op.
    auto new_qtype = RescaleQtype(q_op.getQtype(), gamma_cst);
    if (!new_qtype) return failure();
    rewriter.replaceOpWithNewOp<TFL::QuantizeOp>(q_op, new_qtype.getValue(),
                                                 new_filter, new_qtype);

    // If bias isn't None, it needs to be multiplied as well.
    if (!bias.getType().isa<NoneType>()) {
      rewriter.setInsertionPoint(fc_op);
      auto new_bias = rewriter.create<TF::MulOp>(loc, bias, gamma);
      fc_op.getOperation()->replaceUsesOfWith(bias, new_bias);
    }

    // Remove the tailing mul op.
    mul_op.replaceAllUsesWith(fc_op.getResult());
    return success();
  }
};

using FuseConv2DAndMulWithQDQs = FuseAffinOpAndMulWithQDQs<TFL::Conv2DOp>;
using FuseDepthwiseConv2DAndMulWithQDQs =
    FuseAffinOpAndMulWithQDQs<TFL::DepthwiseConv2DOp>;

// Fuse Binary Op with following Affine operation.
template <typename AffineOpType>
struct FuseBinaryOpToFollowingAffineOp : public OpRewritePattern<AffineOpType> {
  using OpRewritePattern<AffineOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineOpType fc_op,
                                PatternRewriter &rewriter) const override {
    // Binary op.
    Operation *binary_op = fc_op.getInput().getDefiningOp();
    if (!binary_op || binary_op->getNumOperands() != 2) return failure();
    // We only handle the cases the RHS is a scalar.
    // TODO(fengliuai): Currently the canonicalizer pass couldn't guarantee that
    // the constant operands are on the RHS, we need to consider LHS constant
    // operand if necessary.
    DenseFPElementsAttr cst;
    if (!matchPattern(binary_op->getOperand(1), m_Constant(&cst)))
      return failure();
    if (cst.getNumElements() != 1) return failure();
    APFloat cst_value = *cst.value_begin<APFloat>();

    // Affine op.
    Value filter = fc_op.getFilter();
    Value bias = fc_op.getBias();
    DenseFPElementsAttr filter_cst, bias_cst;
    if (!matchPattern(filter, m_Constant(&filter_cst))) {
      // The filter maybe quantized, then we should set it to the real constant.
      auto dq = llvm::dyn_cast_or_null<DequantizeOp>(filter.getDefiningOp());
      if (!dq) return failure();
      auto q =
          llvm::dyn_cast_or_null<QuantizeOp>(dq.getInput().getDefiningOp());
      if (!q || !matchPattern(q.getInput(), m_Constant(&filter_cst))) {
        return failure();
      }
      filter = q.getInput();
    }
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&bias_cst)))
      return failure();
    auto binary_op_activation_func =
        binary_op->template getAttrOfType<StringAttr>(
            "fused_activation_function");
    if (!binary_op_activation_func ||
        binary_op_activation_func.getValue() != "NONE")
      return failure();
    ShapedType filter_type = filter_cst.getType();

    if (llvm::isa<AddOp, SubOp>(binary_op)) {
      auto padding = fc_op->template getAttrOfType<StringAttr>("padding");
      if (padding && padding.getValue() != "VALID") return failure();

      // The fusion of add/sub is actually applying the following
      // transformation:
      // w * (x + c) + b => w * x + (w * c + b)
      // so we have to update the bias.
      if (llvm::isa<SubOp>(binary_op)) cst_value.changeSign();

      auto bias_and_slice =
          GetBiasDimAndSliceSize(filter_type.getShape(), fc_op);
      int64_t bias_size = bias_and_slice.first;
      int64_t slice_size = bias_and_slice.second;
      ShapedType new_bias_type =
          RankedTensorType::get({bias_size}, filter_type.getElementType());

      // The new bias should be a 1-D tensor with length equals to the bias
      // dimension of the weight.
      SmallVector<APFloat, 4> new_bias_values;
      if (bias.getType().isa<NoneType>()) {  // none bias, a list of zeros
        new_bias_values.resize(bias_size,
                               APFloat::getZero(cst_value.getSemantics()));
      } else if (bias_cst.getNumElements() == 1) {  // scalar bias, broadcast it
        new_bias_values.resize(bias_size, *bias_cst.value_begin<APFloat>());
      } else if (bias_cst.getNumElements() == bias_size) {  // 1-d bias, copy it
        new_bias_values.insert(new_bias_values.begin(),
                               bias_cst.value_begin<APFloat>(),
                               bias_cst.value_end<APFloat>());
      } else {
        return failure();
      }

      int64_t flatten_index = 0;
      for (auto fp_it = filter_cst.value_begin<APFloat>(),
                fp_end = filter_cst.value_end<APFloat>();
           fp_it != fp_end; ++fp_it) {
        int bias_index = (flatten_index++ / slice_size) % bias_size;

        new_bias_values[bias_index] =
            new_bias_values[bias_index] + *fp_it * cst_value;
      }
      auto new_bias = DenseFPElementsAttr::get(new_bias_type, new_bias_values);
      auto new_bias_op =
          rewriter.create<ConstOp>(fc_op.getLoc(), new_bias_type, new_bias);
      fc_op.setOperand(0, binary_op->getOperand(0));
      fc_op.setOperand(2, new_bias_op);
    } else if (llvm::isa<MulOp, DivOp>(binary_op)) {
      // The fusion of mul/div is actually applying the following
      // transformation:
      // w * (x ' c) + b => (w ' c) x + b
      // so we have to update the weight.
      bool is_mul = llvm::isa<MulOp>(binary_op);
      auto new_filter =
          filter_cst.mapValues(filter_type.getElementType(), [&](APFloat it) {
            return (is_mul ? it * cst_value : it / cst_value).bitcastToAPInt();
          });
      // We recreate the constant op in case it is shared by the other ops. This
      // might increase the model size.
      auto new_filter_op = rewriter.create<ConstOp>(
          fc_op.getLoc(), filter.getType(), new_filter);
      fc_op.setOperand(0, binary_op->getOperand(0));
      if (fc_op.getFilter() != filter) {
        // This filter goes through quantize and dequantize ops. Then we just
        // need to update the weight to the quantize op.
        filter.replaceAllUsesWith(new_filter_op);
      } else {
        // This filter doesn't go through quantize and dequantize ops, Then
        // we update the weight of the affine op directly.
        fc_op.setOperand(1, new_filter_op);
      }
    } else {
      return failure();
    }
    return success();
  }

 private:
  // Returns the dimension length of the channel dimension and also the slide
  // size by each position in the channel dimension accordingly. tfl.conv2d and
  // tfl.fully_connected has heading channel dimension, but tfl.depthwise_conv2d
  // has tailing channel dimension. This function is to provide a utility to
  // create the above information from the op property.
  static std::pair<int64_t, int64_t> GetBiasDimAndSliceSize(
      ArrayRef<int64_t> filter_shape, AffineOpType op) {
    // Channel dimension index is specified as op property
    auto channel_index_iter = filter_shape.begin();
    std::advance(channel_index_iter, op.GetChannelDimIndex());
    // The slide size is the size of the data in higher dimensions.
    int64_t slice_size =
        std::accumulate(std::next(channel_index_iter), filter_shape.end(), 1,
                        std::multiplies<int64_t>());
    return {*channel_index_iter, slice_size};
  }
};

// If the operand to a broadcastable op is a splat constant, try to replace it
// with a 0-d constant, e.g. before this optimization,
//   %cst = arith.constant dense<1.0> : tensor<16x16x4xf32>
//   %0 = "tfl.conv_2d"...
//   %1 = "tfl.add"(%0, %cst) : (tensor<16x16x4xf32>, tensor<16x16x4xf32>)
// After this optimization:
//   %cst = arith.constant dense<1.0> : tensor<f32>
//   %0 = "tfl.conv_2d"...
//   %1 = "tfl.add"(%0, %cst) : (tensor<16x16x4xf32>, tensor<f32>)
// This pattern can enable more fusing opportunities when the binary op is
// following conv ops.
template <typename BinaryOpType>
struct ScalarizeSplatConstantForBroadcastableOps
    : public OpRewritePattern<BinaryOpType> {
  using OpRewritePattern<BinaryOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOpType binary_op,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr splat_elements_attr;
    if (!IsScalarizableSplatConstant(binary_op.getRhs(),
                                     &splat_elements_attr)) {
      return failure();
    }

    constexpr int kSplatOperandIndex = 1;
    auto result_type =
        binary_op.getResult().getType().template cast<ShapedType>();
    mlir::Value non_splat_operand =
        binary_op.getOperand(1 - kSplatOperandIndex);
    auto non_splat_operand_type =
        non_splat_operand.getType().cast<ShapedType>();
    // If the other operand's shape does not equal to the result shape, then we
    // cannot scalarize the splat constant because the result shape relies on
    // the splat constant op's shape for broadcasting.
    if (!non_splat_operand_type.hasStaticShape() ||
        non_splat_operand_type.getShape() != result_type.getShape() ||
        non_splat_operand_type.getRank() > 4) {
      return failure();
    }

    // If non-splat operand is not fusable affine ops, then no need to apply
    // this transformation.
    if (!CanFuseAffineOp(non_splat_operand.getDefiningOp(), binary_op)) {
      return failure();
    }

    // Creates a new scalar constant op using the splat value.
    mlir::Value splat_operand = binary_op.getOperand(kSplatOperandIndex);
    auto scalar_elements_attr = DenseElementsAttr::get(
        RankedTensorType::get({},
                              splat_elements_attr.getType().getElementType()),
        splat_elements_attr.getSplatValue<mlir::Attribute>());

    auto scalar_constant_op = rewriter.create<arith::ConstantOp>(
        splat_operand.getLoc(), scalar_elements_attr.getType(),
        scalar_elements_attr);

    binary_op.setOperand(kSplatOperandIndex, scalar_constant_op);
    return success();
  }

 private:
  // Returns true if this value is a splat constant op which can be scalarized.
  // Also returns the elements attr if this value is indeed a splat constant.
  bool IsScalarizableSplatConstant(mlir::Value value,
                                   DenseElementsAttr *elements_attr) const {
    if (!matchPattern(value, m_Constant(elements_attr))) {
      return false;
    }
    auto element_type = value.getType().cast<ShapedType>().getElementType();
    // Ignore per-axis quantized constants because after converting to scalar,
    // we will lose per-axis qantization parameter.
    if (element_type.isa<quant::UniformQuantizedPerAxisType>()) {
      return false;
    }
    if (IsScalar(value)) {
      return false;
    }
    return elements_attr->isSplat();
  }

  // If this type is a scalar shaped type.
  bool IsScalar(mlir::Value value) const {
    auto type = value.getType().dyn_cast<ShapedType>();
    if (!type) {
      return false;
    }
    if (!type.hasStaticShape()) {
      return false;
    }
    return type.getNumElements() == 1;
  }

  // Returns true if we can fuse an affine op with consuming binary op.
  bool CanFuseAffineOp(Operation *affine_op, Operation *binary_op) const {
    if (!isa_and_nonnull<TFL::Conv2DOp, TFL::DepthwiseConv2DOp,
                         TFL::FullyConnectedOp>(affine_op)) {
      return false;
    }
    DenseElementsAttr value;
    // Check that bias are constants if not none.
    Value bias = affine_op->getOperand(2);
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&value))) {
      return false;
    }
    // If the binary op is mul/div, also check that filter is constant.
    if (isa<TFL::MulOp, TFL::DivOp>(binary_op) &&
        !matchPattern(affine_op->getOperand(1), m_Constant(&value))) {
      return false;
    }

    // We can only fuse F32/BF16.
    auto is_fusable_type = [](Type t) {
      Type element_type = t;
      if (auto shaped_type = t.dyn_cast<ShapedType>()) {
        element_type = shaped_type.getElementType();
      }
      return element_type.isBF16() || element_type.isF32();
    };
    for (Type t : binary_op->getOperandTypes()) {
      if (!is_fusable_type(t)) {
        return false;
      }
    }

    return true;
  }
};

using ScalarizeSplatConstantForSub =
    ScalarizeSplatConstantForBroadcastableOps<TFL::SubOp>;
using ScalarizeSplatConstantForAdd =
    ScalarizeSplatConstantForBroadcastableOps<TFL::AddOp>;
using ScalarizeSplatConstantForMul =
    ScalarizeSplatConstantForBroadcastableOps<TFL::MulOp>;
using ScalarizeSplatConstantForDiv =
    ScalarizeSplatConstantForBroadcastableOps<TFL::DivOp>;

struct ConvertTrivialTransposeOpToReshapeOp
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp transpose_op,
                                PatternRewriter &rewriter) const override {
    auto input_type = transpose_op.getInput().getType().cast<ShapedType>();
    auto output_type = transpose_op.getOutput().getType().cast<ShapedType>();
    // It's possible to know if the transformation is safe only if the input
    // & output shapes are fully known and permutation is a constant.
    if (!input_type.hasStaticShape() || !output_type.hasStaticShape())
      return failure();
    Value perm = transpose_op.getPerm();
    DenseElementsAttr perm_values_attr;
    if (!matchPattern(perm, m_Constant(&perm_values_attr))) return failure();

    auto input_shape = input_type.getShape();
    SmallVector<int64_t, 8> perm_values;
    for (const auto &dim : perm_values_attr.getValues<APInt>())
      perm_values.push_back(dim.getSExtValue());

    // This should never happen unless the input graph is malformed.
    if (input_shape.size() != perm_values.size()) {
      transpose_op.emitError(
          "TransposeOP has inconsistent input and perm values.");
    }

    SmallVector<int, 8> old_major_index_ordering;
    SmallVector<int, 8> new_major_index_ordering;
    for (int i = 0, end = input_shape.size(); i < end; i++) {
      if (input_shape[i] != 1) {
        old_major_index_ordering.push_back(i);
      }

      if (input_shape[perm_values[i]] != 1) {
        new_major_index_ordering.push_back(perm_values[i]);
      }
    }
    if (old_major_index_ordering != new_major_index_ordering) {
      return failure();
    }

    // Rewrite.
    Location loc = transpose_op.getLoc();

    SmallVector<int32_t, 8> output_shape_values;
    for (auto dim : output_type.getShape()) {
      output_shape_values.push_back(
          ShapedType::isDynamic(dim) ? -1 : static_cast<int32_t>(dim));
    }
    auto type = mlir::RankedTensorType::get(output_shape_values.size(),
                                            rewriter.getIntegerType(32));
    auto new_shape_attr =
        mlir::DenseIntElementsAttr::get(type, output_shape_values);
    auto new_shape = rewriter.create<TF::ConstOp>(loc, new_shape_attr);

    rewriter.replaceOpWithNewOp<TFL::ReshapeOp>(
        transpose_op, transpose_op.getOutput().getType(),
        transpose_op.getInput(), new_shape);

    return success();
  }
};

// Remove Reshape before FullyConnected when `keep_num_dims=false` and Reshape
// does not alter the last dimension as FullyConnected will collapse all other
// dimensions into a single dimension. For example,
//
//   %shape = arith.constant dense<[1, 128, 64]> : tensor<3xi32>
//   %reshape = tfl.reshape(%input, %shape) // %input: tensor<128x64xf32>
//   %fc = tfl.fully_connected(%reshape, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
//
// can be canonicalized to
//
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
struct RemoveReshapeBeforeFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fully_connected_op,
                                PatternRewriter &) const override {
    auto input = fully_connected_op.getInput();
    auto input_ty = input.getType().dyn_cast<ShapedType>();
    auto output_ty = fully_connected_op.getOutput()[0]
                         .getType()
                         .template dyn_cast<ShapedType>();
    if (!input_ty.hasStaticShape() ||
        fully_connected_op.getWeightsFormat() != "DEFAULT" ||
        fully_connected_op.getKeepNumDims() || !output_ty.hasStaticShape() ||
        output_ty.getRank() != 2) {
      return failure();
    }

    auto reshape_op = input.getDefiningOp<TFL::ReshapeOp>();
    if (!reshape_op) return failure();

    // Check if the last dimension does not change after reshape.
    auto reshape_input = reshape_op.getInput();
    auto reshape_input_ty = reshape_input.getType().dyn_cast<ShapedType>();
    if (!reshape_input_ty.hasStaticShape() || input_ty.getRank() == 0 ||
        reshape_input_ty.getRank() == 0 ||
        input_ty.getDimSize(input_ty.getRank() - 1) !=
            reshape_input_ty.getDimSize(reshape_input_ty.getRank() - 1)) {
      return failure();
    }

    // Connect the input to the one of reshape.
    fully_connected_op.setOperand(0, reshape_input);
    return success();
  }
};

// Remove Reshape after FullyConnected when `keep_num_dims=false`, the Reshape
// does not alter the last dimension and it restores the batch dimensions
// collapsed by the FullyConnected op due to `keep_num_dims=false`. For example,
//
//   // %input: tensor<4x16x32xf32>
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
//   %shape = arith.constant dense<[4, 16, 32]> : tensor<3xi32>
//   %rs = tfl.reshape(%fc, %shape)
//
// can be canonicalized to
//
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = true, weights_format = "DEFAULT"}
struct RemoveReshapeAfterFullyConnected
    : public OpRewritePattern<TFL::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ReshapeOp reshape_op,
                                PatternRewriter &rewriter) const override {
    auto fully_connected_op = llvm::dyn_cast_or_null<TFL::FullyConnectedOp>(
        reshape_op.getInput().getDefiningOp());
    if (!fully_connected_op || fully_connected_op.getNumResults() != 1 ||
        fully_connected_op.getWeightsFormat() != "DEFAULT" ||
        fully_connected_op.getKeepNumDims())
      return failure();
    if (!reshape_op.getInput().hasOneUse()) return failure();

    auto input_shape =
        fully_connected_op.getInput().getType().cast<ShapedType>();
    auto output_shape = fully_connected_op.getType(0).cast<ShapedType>();
    auto reshape_shape = reshape_op.getType().cast<ShapedType>();
    if (!input_shape.hasStaticShape() || !output_shape.hasStaticShape() ||
        !reshape_shape.hasStaticShape())
      return failure();

    // Check that the reshape doesn't modify the last dimension and it restores
    // the input (batch) dimension with the exception of the feature (last)
    // dimension.
    if (output_shape.getShape().empty() || reshape_shape.getShape().empty() ||
        output_shape.getShape().back() != reshape_shape.getShape().back() ||
        input_shape.getShape().drop_back() !=
            reshape_shape.getShape().drop_back())
      return failure();

    llvm::SmallVector<Type, 1> output_type{reshape_op.getType()};
    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        reshape_op, output_type, /*input=*/fully_connected_op.getInput(),
        /*filter=*/fully_connected_op.getFilter(),
        /*bias=*/fully_connected_op.getBias(),
        /*fused_activation_function=*/
        fully_connected_op.getFusedActivationFunction(),
        /*weights_format=*/fully_connected_op.getWeightsFormat(),
        /*keep_num_dims=*/true,
        /*asymmetric_quantize_inputs=*/
        fully_connected_op.getAsymmetricQuantizeInputsAttr());
    return success();
  }
};

// Fuses Unpack with proceeding Concatenation to Reshape if output type has
// static shape and activation function is none. For example:
//
//   // %input: tensor<1x3x2xf32>
//   %unpack:3 = "tfl.unpack"(%input) {axis = 1 : i32, num = 3 : i32}
//   %res = "tfl.concatenation"(%unpack#0, %unpack#1, %unpack#2)
//        {axis = -1 : i32, fused_activation_function = "NONE"}
//
// can be optimized to
//
//   %cst = arith.constant dense<[1, 6]> : tensor<2xi32>
//   %res = "tfl.reshape"(%input, %cst)
struct FuseUnpackAndConcatToReshape
    : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concat_op,
                                PatternRewriter &rewriter) const override {
    if (concat_op.getFusedActivationFunction() != "NONE") {
      return failure();
    }

    // Checks all operands come from the same unpack op.
    auto first_operand = concat_op.getValues().front();
    auto unpack_op =
        dyn_cast_or_null<TFL::UnpackOp>(first_operand.getDefiningOp());
    if (!unpack_op || unpack_op.getNumResults() != concat_op.getNumOperands()) {
      return failure();
    }
    for (const auto &index_and_value : llvm::enumerate(concat_op.getValues())) {
      if (index_and_value.value() !=
          unpack_op.getResult(index_and_value.index())) {
        return failure();
      }
    }

    auto output_type = concat_op.getType().cast<ShapedType>();
    if (!output_type.hasStaticShape()) {
      return failure();
    }

    auto new_shape_array = output_type.getShape();
    // This is to workaround the unnecessary cast i64 -> i32.
    SmallVector<int32_t, 4> new_shape_array_i32;
    for (auto size : new_shape_array) {
      new_shape_array_i32.push_back(
          ShapedType::isDynamic(size) ? -1 : static_cast<int32_t>(size));
    }
    auto new_shape = rewriter.create<TFL::ConstOp>(
        concat_op.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get(new_shape_array_i32.size(),
                                  rewriter.getIntegerType(32)),
            new_shape_array_i32));

    rewriter.replaceOpWithNewOp<TFL::ReshapeOp>(
        concat_op, output_type, unpack_op.getInput(), new_shape);
    return success();
  }
};

// Reduce the K of a TopKV2Op for the following case.
//
// values, indices = tfl.topkv2(%inputs, K)
// %1 = tfl.slice(values, 0, k)
// %2 = tfl.slice(indices,0, k)
// .... (values and indices only used for %1 and %2)
//
// %1 or %2 can be absent. If values and indices are only used here,
// this pattern can be replaced with (conceptually)
//
// %values, %indices = tfl.topkv2(%inputs, k)
// replace all use of %1 with values
// replace all use of %2 with indices
//
struct OptimizeTopK : public OpRewritePattern<TFL::TopKV2Op> {
  using OpRewritePattern::OpRewritePattern;

  // It computes the last dim k of slice size of value.user.
  // If value has no use then return 0.
  std::optional<int32_t> ComputeSliceK(Value value) const {
    if (value.use_empty()) return 0;
    auto slice_op =
        llvm::dyn_cast_or_null<TFL::SliceOp>(value.getUses().begin().getUser());
    // We only match for the case where value is used by SliceOp.
    if (!slice_op) return std::nullopt;
    DenseElementsAttr begin;
    DenseElementsAttr size;
    if (!matchPattern(slice_op->getOperand(1), m_Constant(&begin)) ||
        !matchPattern(slice_op->getOperand(2), m_Constant(&size)))
      return std::nullopt;

    // Check if "begin" is a zero tensor.
    for (auto begin_idx : begin.getValues<APInt>())
      if (begin_idx != 0) return std::nullopt;

    // Check if "size" is equal to slice_op.input.shape except
    // for last dimension.
    // It can be done  by verifying the number of elements:
    // i.e., num_input/input_last_dim = num_result/k
    auto input_ty = value.getType().dyn_cast_or_null<ShapedType>();
    auto result_ty = slice_op.getType().dyn_cast<ShapedType>();
    if (!input_ty || !result_ty) return std::nullopt;
    if (!input_ty.hasStaticShape() || !result_ty.hasStaticShape())
      return std::nullopt;
    if (!input_ty.getRank() || !result_ty.getRank()) return std::nullopt;
    int num_input = input_ty.getNumElements();
    int input_last_dim = input_ty.getShape().back();
    if (input_last_dim < 1) return std::nullopt;
    int num_result = result_ty.getNumElements();
    auto size_last = *(--size.value_end<APInt>());
    int32_t k = size_last.getSExtValue();
    if (num_input / input_last_dim * k != num_result) return std::nullopt;
    // We don't match sliceOp with last dim size = 0.
    if (!k) return std::nullopt;
    return k;
  }

  LogicalResult matchAndRewrite(TFL::TopKV2Op op,
                                PatternRewriter &rewriter) const override {
    auto values = op.getValues();
    auto indices = op.getIndices();
    // op.getValues() and op.getIndices() cannot be used more than once.
    if (!values.hasOneUse() && !values.use_empty()) return failure();
    if (!indices.hasOneUse() && !indices.use_empty()) return failure();

    auto k_values_or = ComputeSliceK(values);
    auto k_indices_or = ComputeSliceK(indices);
    if (!k_values_or.has_value() || !k_indices_or.has_value()) return failure();
    int32_t k_values = k_values_or.value();
    int32_t k_indices = k_indices_or.value();
    // We don't match two SliceOp with different sizes.
    if (k_values != k_indices && !values.use_empty() && !indices.use_empty())
      return failure();

    // Start replacing.
    auto k = !values.use_empty() ? k_values : k_indices;
    // Build scalar tensor k.
    auto k_ty = mlir::RankedTensorType::get({}, rewriter.getIntegerType(32));
    Value k_cst = rewriter.create<TFL::ConstOp>(
        op.getLoc(), DenseElementsAttr::get(k_ty, k));
    // Compute new result types.
    auto values_ty = values.getType().dyn_cast<ShapedType>();
    auto indices_ty = indices.getType().dyn_cast<ShapedType>();
    auto shape = std::vector<int64_t>();
    for (auto d : values_ty.getShape().drop_back()) {
      shape.push_back(d);
    }
    shape.push_back(static_cast<int64_t>(k));
    auto new_values_ty =
        mlir::RankedTensorType::get(shape, values_ty.getElementType());
    auto new_indices_ty =
        mlir::RankedTensorType::get(shape, indices_ty.getElementType());
    TFL::TopKV2Op top_k_op = rewriter.create<TFL::TopKV2Op>(
        op.getLoc(), new_values_ty, new_indices_ty, op->getOperand(0), k_cst);

    // Remove original ops (topk, Slice, Slice).
    if (!values.use_empty()) {
      auto values_slice_op = llvm::dyn_cast_or_null<TFL::SliceOp>(
          values.getUses().begin().getUser());
      values_slice_op.getResult().replaceAllUsesWith(top_k_op.getValues());
      values_slice_op.erase();
    }
    if (!indices.use_empty()) {
      auto indices_slice_op = llvm::dyn_cast_or_null<TFL::SliceOp>(
          indices.getUses().begin().getUser());
      indices_slice_op.getResult().replaceAllUsesWith(top_k_op.getIndices());
      indices_slice_op.erase();
    }
    op.erase();
    return success();
  }
};

using FuseBinaryOpToFollowingFullyConnected =
    FuseBinaryOpToFollowingAffineOp<FullyConnectedOp>;
using FuseBinaryOpToFollowingDepthwiseConv2D =
    FuseBinaryOpToFollowingAffineOp<DepthwiseConv2DOp>;
using FuseBinaryOpToFollowingConv2D = FuseBinaryOpToFollowingAffineOp<Conv2DOp>;

// Adds canonicalization patterns to the list of patterns.
void AddCanonicalizationPatterns(MLIRContext *context,
                                 RewritePatternSet *patterns) {
  for (auto op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(*patterns, context);
}

void OptimizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto *ctx = &getContext();
  auto func = getOperation();

  // Merge reshapes into fully connected ops before we start moving them past
  // binary ops.
  RewritePatternSet phase_0_patterns(&getContext());
  phase_0_patterns
      .add<RemoveReshapeAfterFullyConnected, RemoveReshapeBeforeFullyConnected>(
          ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(phase_0_patterns));

  // Potentially the binary ops might be fused together, like hard_swish, thus
  // we explore these potentially first and then fuse the binary ops with the
  // following ops in a second pattern match.
  TFL::populateWithGenerated(patterns);
  patterns.add<FuseFullyConnectedAndAdd, FuseAddAndFullyConnected,
               FuseFullyConnectedAndMul,
               FuseFullyConnectedAndReluX<TFL::ReluOp, kRelu>,
               FuseFullyConnectedAndReluX<TFL::Relu6Op, kRelu6>,
               FuseFullyConnectedAndReluX<TFL::Relu1Op, kRelu1>>(ctx);
  if (!this->disable_fuse_mul_and_fc_) {
    patterns.add<FuseMulAndFullyConnected>(ctx);
  }
  if (this->enable_canonicalization_)
    AddCanonicalizationPatterns(ctx, &patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Fuse the binary ops with the following ops.
  RewritePatternSet phase_2_patterns(&getContext());
  TFL::populateWithGenerated(phase_2_patterns);
  phase_2_patterns.add<
      ScalarizeSplatConstantForAdd, ScalarizeSplatConstantForSub,
      ScalarizeSplatConstantForMul, ScalarizeSplatConstantForDiv,
      FuseFullyConnectedAndAdd, FuseAddAndFullyConnected,
      FuseFullyConnectedAndMul, FuseFullyConnectedAndReluX<TFL::ReluOp, kRelu>,
      FuseFullyConnectedAndReluX<TFL::Relu6Op, kRelu6>,
      FuseFullyConnectedAndReluX<TFL::Relu1Op, kRelu1>,
      FuseBinaryOpToFollowingConv2D, FuseBinaryOpToFollowingDepthwiseConv2D,
      FuseBinaryOpToFollowingFullyConnected, FuseConv2DAndMulWithQDQs,
      FuseDepthwiseConv2DAndMulWithQDQs, ConvertTrivialTransposeOpToReshapeOp,
      RemoveReshapeAfterFullyConnected, RemoveReshapeBeforeFullyConnected,
      FuseUnpackAndConcatToReshape, OptimizeTopK>(ctx);
  if (!this->disable_fuse_mul_and_fc_) {
    phase_2_patterns.add<FuseMulAndFullyConnected>(ctx);
  }
  if (this->enable_canonicalization_)
    AddCanonicalizationPatterns(ctx, &phase_2_patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(phase_2_patterns));
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizePass(
    bool enable_canonicalization, bool disable_fuse_mul_and_fc) {
  return std::make_unique<OptimizePass>(enable_canonicalization,
                                        disable_fuse_mul_and_fc);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizePass() {
  return std::make_unique<OptimizePass>();
}

}  // namespace TFL
}  // namespace mlir
