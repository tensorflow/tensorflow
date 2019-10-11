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

#include <cstdint>
#include <iterator>
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Transforms/DialectConversion.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

using namespace mlir;

namespace {
struct LegalizeTF : public FunctionPass<LegalizeTF> {
  /// Performs the lowering to XLA dialect.
  void runOnFunction() override;
};
}  // end anonymous namespace

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>>
mlir::xla_hlo::createLegalizeTFPass() {
  return std::make_unique<LegalizeTF>();
}

/// Returns if the given TF data format string is the default format.
static bool isDefaultDataFormat(StringRef format) { return format == "NHWC"; }

/// Returns the feature dimension for the given format and input type.
static size_t getFeatureDimension(StringAttr format,
                                  RankedTensorType inputType) {
  return isDefaultDataFormat(format.getValue()) ? inputType.getRank() - 1 : 1;
}

// Returns 1D 64-bit dense elements attribute with the given values.
static DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                               Builder *builder) {
  RankedTensorType ty = builder->getTensorType(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseElementsAttr::get<int64_t>(ty, values)
      .cast<DenseIntElementsAttr>();
}

static IntegerAttr GetHLOAxisFromTFAxis(ElementsAttr attr, int64_t rank,
                                        Builder *b) {
  SmallVector<uint64_t, 1> index(attr.getType().getRank(), 0);
  int64_t axis = attr.getValue<IntegerAttr>(index).getInt();
  if (axis < 0) {
    axis += rank;
  }
  return b->getI64IntegerAttr(axis);
}

// If `value` is an IntegerAttr, returns the integer value for the HLO axis
// corresponding to the tensorflow axis. In particular, the tensorflow axis can
// be negative, in which case, the corresponding HLO axis is
// (axis + rank-of-the-tensor).
static llvm::Optional<int64_t> GetIntegerHLOAxisFromTFAxis(Value *value,
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
static xla_hlo::ConvertOp CastElementsToI64(Location loc, Value *value,
                                            PatternRewriter *rewriter) {
  auto type = value->getType().cast<RankedTensorType>();
  assert(type && "CastElementsToI64 requires a shaped tensor as input.");
  ArrayRef<int64_t> shape = type.getShape();
  auto i64_type = rewriter->getTensorType(shape, rewriter->getIntegerType(64));
  return rewriter->create<xla_hlo::ConvertOp>(loc, i64_type, value);
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

// Returns minimum value for the given int or float element type.
static xla_hlo::ConstOp GetMinValueForType(Type ty, Location loc,
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
  return rewriter->create<xla_hlo::ConstOp>(loc, attr);
}

// Returns an integer constant for the given int or float element type.
static xla_hlo::ConstOp GetScalarForType(Type ty, Location loc,
                                         int64_t raw_value,
                                         PatternRewriter *rewriter) {
  RankedTensorType scalar_ty = rewriter->getTensorType({}, ty);

  DenseElementsAttr attr;
  if (auto float_ty = ty.dyn_cast_or_null<FloatType>()) {
    APFloat value(float_ty.getFloatSemantics(), raw_value);
    attr = DenseElementsAttr::get(scalar_ty, value);
  } else {
    auto int_ty = ty.cast<IntegerType>();
    APInt value(int_ty.getWidth(), raw_value, true);
    attr = DenseElementsAttr::get(scalar_ty, value);
  }
  return rewriter->create<xla_hlo::ConstOp>(loc, attr);
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

/// Return a 1D DenseIntElementsAttr for the feature dimension of a BiasAdd.
static DenseIntElementsAttr getBiasFeatureDimension(Builder &b,
                                                    StringAttr format,
                                                    Value *input) {
  auto inputType = input->getType().cast<RankedTensorType>();
  size_t featureDim = getFeatureDimension(format, inputType);
  RankedTensorType type = b.getTensorType(1, b.getIntegerType(64));
  return DenseIntElementsAttr::get(type, featureDim)
      .cast<DenseIntElementsAttr>();
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

// Returns whether the two values are guaranteed to be broadcastable to the
// same shape, this broadcasts size 1 tensors up to any rank. Dynamic dimensions
// must be broadcasted with a size 1 tensor or another dynamic dimension.
// Returns false on rankless.
static bool AreBroadcastCompatible(Value *x, Value *y) {
  auto x_rankless = x->getType().dyn_cast<RankedTensorType>();
  auto y_rankless = y->getType().dyn_cast<RankedTensorType>();
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

static DenseIntElementsAttr getBroadcastDimensionsAttr(Builder &b, Value *x,
                                                       Value *y) {
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

  RankedTensorType type = b.getTensorType({minRank}, b.getIntegerType(64));
  return DenseIntElementsAttr::get<int64_t>(type, broadcastDimensions)
      .cast<DenseIntElementsAttr>();
}

//===----------------------------------------------------------------------===//
// Softmax op utilities.
//===----------------------------------------------------------------------===//

// Returns a 1-d i64 elements attribute populated with numbers from start to
// end, excluding.
static DenseIntElementsAttr GetI64ElementsAttrForSeq(int start, int end,
                                                     Builder *builder) {
  int size = end - start;

  SmallVector<int64_t, 4> vals;
  vals.resize(size);
  std::iota(vals.begin(), vals.end(), start);

  TensorType ty = builder->getTensorType({size}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get<int64_t>(ty, vals)
      .cast<DenseIntElementsAttr>();
}

// Returns the type to use for accumulating the given type.
static Type GetAccumulationType(Type ty) {
  // Upcast 16 bit sum reductions to 32 bit to reduce the precision loss from
  // repeated floating point additions.
  return (ty.isF16() || ty.isBF16()) ? FloatType::getF32(ty.getContext()) : ty;
}

//===----------------------------------------------------------------------===//
// ArgMax/ArgMin op utilities.
//===----------------------------------------------------------------------===//

static void BuildArgMinMaxReductionBody(Type input_element_type,
                                        Type index_element_type,
                                        StringRef direction, Region *body,
                                        OpBuilder *builder) {
  OpBuilder::InsertionGuard insertion_point_gurad(*builder);

  Type input_type = builder->getTensorType(/*shape=*/{}, input_element_type);
  Type index_type = builder->getTensorType(/*shape=*/{}, index_element_type);
  Block *block = builder->createBlock(body);
  block->addArguments({input_type, index_type, input_type, index_type});

  Location loc = body->getLoc();
  Type compare_type =
      builder->getTensorType(/*shape=*/{}, builder->getIntegerType(1));
  StringAttr compare_direction =
      StringAttr::get(direction, builder->getContext());
  Value *compare = builder->create<xla_hlo::CompareOp>(
      loc, compare_type, block->getArgument(0), block->getArgument(2),
      /*broadcast_dimensions=*/nullptr, compare_direction);

  Value *selected_input = builder->create<xla_hlo::SelectOp>(
      loc, input_type, compare, block->getArgument(0), block->getArgument(2));
  Value *selected_index = builder->create<xla_hlo::SelectOp>(
      loc, index_type, compare, block->getArgument(1), block->getArgument(3));

  Value *return_values[] = {selected_input, selected_index};
  builder->create<xla_hlo::ReturnOp>(loc, return_values);
}

//===----------------------------------------------------------------------===//
// Slice op utilities.
//===----------------------------------------------------------------------===//

static bool CanBeTranslatedToDynamicSlice(Value *input, Value *start_indices,
                                          DenseIntElementsAttr slice_sizes) {
  auto input_ty = input->getType().dyn_cast<RankedTensorType>();
  int64_t input_rank = input_ty.getRank();
  ArrayRef<int64_t> input_shape = input_ty.getShape();
  DenseIntElementsAttr constant_start_indices;
  if (!matchPattern(start_indices, m_Constant(&constant_start_indices))) {
    for (int64_t i = 0; i < input_rank; ++i) {
      int64_t slice_size = slice_sizes.getValue<IntegerAttr>(i).getInt();
      int64_t input_size = input_shape[i];
      if (slice_size < 0 || (input_size != -1 && slice_size > input_size)) {
        return false;
      }
    }
    return true;
  }

  for (int64_t i = 0; i < input_rank; ++i) {
    int64_t input_size = input_shape[i];
    int64_t start_index =
        constant_start_indices.getValue<IntegerAttr>(i).getInt();
    int64_t slice_size = slice_sizes.getValue<IntegerAttr>(i).getInt();
    if (start_index < 0) return false;
    // A slice_size of -1 means "all elements from start_index to the end".
    // We can't support this semantics for dynamic shapes.
    if (slice_size == -1) {
      if (input_size == -1) return false;
      slice_size = input_size - start_index;
    }
    if (input_size != -1 && start_index + slice_size > input_size) {
      return false;
    }
  }

  return true;
}

// TF slice size can be -1, which represents all elements from start_index to
// the end. HLO slice size can't be -1. As such, we need to translate TF slice
// size -1 to HLO slice size.
static DenseIntElementsAttr TFSliceSizes2HLOSliceSizes(
    Value *input, Value *start_indices, DenseIntElementsAttr slice_sizes,
    Builder *builder) {
  DenseIntElementsAttr constant_start_indices;
  if (!matchPattern(start_indices, m_Constant(&constant_start_indices))) {
    return slice_sizes;
  }

  auto input_ty = input->getType().dyn_cast<RankedTensorType>();
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
// Op converters.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace xla {
namespace {

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
      mlir::xla_hlo::ConvDimensionNumbers::get(
          batch_dim, feature_dim, spatial_dims, kernel_input_feature_dim,
          kernel_output_feature_dim, kernel_spatial_dimensions, batch_dim,
          feature_dim, spatial_dims, builder->getContext()));
}

// Converts the TensorFlow conv op in template to the generic HLO conv op by
// converting TensorFlow op attributes to HLO op attributes.
//
// Sample result for Conv2D:
//
//   %conv = "xla_hlo.conv"(%input, %filter) {
//     strides = [1, 2],
//     paddings = [[1, 0], [1, 1]],
//     ...
//   }
//
// This pattern is not defined using declarative rewrite rules as computation of
// the paddings attribute anyway requires multiple source op attributes and
// result op attributes. Defining it as declarative rewrite rule will introduce
// some duplication in the C++ helper methods.
template <typename OpT, int num_spatial_dims>
class ConvertConv : public OpRewritePattern<OpT> {
 public:
  explicit ConvertConv(MLIRContext *context) : OpRewritePattern<OpT>(context) {}

  PatternMatchResult matchAndRewrite(OpT op,
                                     PatternRewriter &rewriter) const override {
    tensorflow::TensorFormat format;
    std::string data_format = op.data_format().str();
    if (!FormatFromString(data_format, &format)) return Pattern::matchFailure();

    auto input_ty = op.input()->getType().template dyn_cast<RankedTensorType>();
    auto filter_ty =
        op.filter()->getType().template dyn_cast<RankedTensorType>();
    auto result_ty = op.getType().template dyn_cast<RankedTensorType>();

    // Input, filter and the result needs to have static shape for calculation
    // of HLO paddings and feature group count attributes.
    for (RankedTensorType ty : {input_ty, filter_ty, result_ty}) {
      if (!ty || !ty.hasStaticShape()) return Pattern::matchFailure();
    }

    int num_dims = num_spatial_dims + 2;
    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.padding().str(), &padding).ok())
      return Pattern::matchFailure();

    auto get_int = [](Attribute attr) {
      return attr.template cast<IntegerAttr>().getInt();
    };

    SmallVector<int64_t, 4> spatial_dim_indices;
    SmallVector<int64_t, 4> rhs_dilations;
    SmallVector<int64_t, 4> window_strides;
    SmallVector<int64_t, 8> paddings;

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

    for (int i = 0; i < num_spatial_dims; ++i) {
      int64_t dim = GetTensorSpatialDimIndex(num_dims, format, i);
      spatial_dim_indices.push_back(dim);

      int64_t stride = get_int(strides[dim]);
      int64_t dilation = get_int(dilations[dim]);
      window_strides.push_back(stride);
      rhs_dilations.push_back(dilation);

      int64_t pad_low, pad_high;
      if (padding == tensorflow::Padding::EXPLICIT) {
        pad_low = get_int(explicit_paddings[2 * dim]);
        pad_high = get_int(explicit_paddings[2 * dim + 1]);
      } else {
        tensorflow::int64 output_size;
        tensorflow::int64 pad_low_int64;
        tensorflow::int64 pad_high_int64;
        tensorflow::Status status = tensorflow::GetWindowedOutputSizeVerboseV2(
            input_ty.getDimSize(i), filter_ty.getDimSize(i), dilation, stride,
            padding, &output_size, &pad_low_int64, &pad_high_int64);
        if (!status.ok()) return Pattern::matchFailure();
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

    auto dimension_numbers_attr =
        GetConvDimensionNumbersAttr(spatial_dim_indices, format, &rewriter);

    int64_t input_channels =
        GetDimSize(input_ty, GetTensorFeatureDimIndex(num_dims, format));
    // Filters data_format is always HWIO so input channels dimension is after
    // all spatial dimensions.
    int64_t filter_channels = GetDimSize(filter_ty, num_spatial_dims);
    // TensorFlow convolution op verifies that the number of input channels is
    // divisible by the number of filter channels.
    int64_t feature_group_count = input_channels / filter_channels;
    auto feature_group_count_attr = rewriter.getNamedAttr(
        "feature_group_count", rewriter.getI64IntegerAttr(feature_group_count));

    auto batch_group_count_attr = rewriter.getNamedAttr(
        "batch_group_count", rewriter.getI64IntegerAttr(1));

    RankedTensorType paddings_ty = rewriter.getTensorType(
        {num_spatial_dims, 2}, rewriter.getIntegerType(64));
    auto paddings_attr = rewriter.getNamedAttr(
        "padding", DenseElementsAttr::get<int64_t>(paddings_ty, paddings));

    SmallVector<Value *, 2> operands(op.getOperands());
    NamedAttribute attrs[] = {rhs_dilations_attr,     window_strides_attr,
                              dimension_numbers_attr, feature_group_count_attr,
                              batch_group_count_attr, paddings_attr};
    rewriter.replaceOpWithNewOp<xla_hlo::ConvOp>(op, op.getType(), operands,
                                                 llvm::makeArrayRef(attrs));
    return Pattern::matchSuccess();
  }
};

using ConvertConv2D = ConvertConv<TF::Conv2DOp, /*num_spatial_dims=*/2>;

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
    xla_hlo::ConstOp init = GetMinValueForType(element_type, loc, &rewriter);

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

    rewriter.replaceOp(op, reduce.getResult(0));
    return matchSuccess();
  }
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
//    %halved_logits = xla_hlo.mul %logits, %half_array : tensor<2xf32>
//    %tanh = "xla_hlo.tanh"(%halved_logits) : (tensor<2xf32>) -> tensor<2xf32>
//
//    // Have the result of Tanh and add 0.5.
//    %halved_tanh = xla_hlo.mul %tanh, %half : tensor<2xf32>
//    %sigmoid = xla_hlo.add %halved_tanh, %half : tensor<2xf32>
//
class ConvertSigmoidOp : public OpRewritePattern<TF::SigmoidOp> {
 public:
  explicit ConvertSigmoidOp(MLIRContext *context)
      : OpRewritePattern<TF::SigmoidOp>(context, 1) {}

  PatternMatchResult matchAndRewrite(TF::SigmoidOp op,
                                     PatternRewriter &rewriter) const override {
    auto operand = op.getOperand();

    auto scalar_one = rewriter.create<xla_hlo::ConstOp>(
        op.getLoc(),
        rewriter.getFloatAttr(getElementTypeOrSelf(operand->getType()), 0.5));

    auto shaped_type = operand->getType().cast<ShapedType>();
    auto constant_ones = rewriter.create<xla_hlo::BroadcastOp>(
        op.getLoc(), shaped_type, scalar_one,
        rewriter
            .getDenseIntElementsAttr(
                rewriter.getTensorType({shaped_type.getRank()},
                                       rewriter.getIntegerType(64)),
                shaped_type.getShape())
            .cast<DenseIntElementsAttr>());

    auto scaled_input = rewriter.create<xla_hlo::MulOp>(
        op.getLoc(), operand->getType(), operand, constant_ones,
        DenseIntElementsAttr());
    auto tanh_op = rewriter.create<xla_hlo::TanhOp>(
        op.getLoc(), operand->getType(), scaled_input);
    auto mul_op = rewriter.create<xla_hlo::MulOp>(
        op.getLoc(), operand->getType(), tanh_op, constant_ones,
        /*DenseIntElementsAttr=*/DenseIntElementsAttr());
    auto add_op = rewriter.create<xla_hlo::AddOp>(
        op.getLoc(), operand->getType(), mul_op, constant_ones,
        /*DenseIntElementsAttr=*/DenseIntElementsAttr());

    rewriter.replaceOp(op, add_op.getResult());
    return matchSuccess();
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
//
//    // Softmax computation:
//    %softmax = "xla_hlo.div"(%exp, %sum_f16) {broadcast_dimensions = 0}
//            : (tensor<BxNxf16>, tensor<Bxf16>) -> tensor<BxNxf16>
//
// TODO(hinsu): Use tf.Max and tf.Sum instead of lowering directly to xla.
template <typename OpTy, bool use_log = true>
class ConvertSoftmaxOp : public OpRewritePattern<OpTy> {
 public:
  explicit ConvertSoftmaxOp(MLIRContext *context)
      : OpRewritePattern<OpTy>(context, 1) {}

  PatternMatchResult matchAndRewrite(OpTy op,
                                     PatternRewriter &rewriter) const override {
    Value *logits = op.logits();

    // Softmax converter requires ranked type because the XLA reduce ops used
    // while lowering requires dimensions attribute to reduce along.
    RankedTensorType type = logits->getType().dyn_cast<RankedTensorType>();
    if (!type) return Pattern::matchFailure();

    auto loc = op.getLoc();

    int rank = type.getRank();

    // Note that the TensorFlow Softmax op verifies that the input rank is
    // greater than or equal to one so both of the following sequences are
    // valid.
    auto batch_dims = GetI64ElementsAttrForSeq(0, rank - 1, &rewriter);
    auto reduce_dim = GetI64ElementsAttrForSeq(rank - 1, rank, &rewriter);

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
        loc, reduce_out_type, logits, init.getResult(), reduce_dim);
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
    init = rewriter.create<xla_hlo::ConstOp>(
        loc, DenseElementsAttr::get(rewriter.getTensorType({}, element_type),
                                    rewriter.getZeroAttr(element_type)));
    Type sum_out_type = rewriter.getTensorType(reduce_shape, sum_element_type);
    auto exp_sum = rewriter.create<xla_hlo::ReduceOp>(
        loc, sum_out_type, casted_exp.getResult(), init.getResult(),
        reduce_dim);
    BuildReduceBody<xla_hlo::AddOp>(element_type, &exp_sum.body(), &rewriter);
    Value *sum = exp_sum.getResult(0);

    // Convert the summation result back to the original element type and divide
    // exponentials by the summations.
    sum = rewriter.create<xla_hlo::ConvertOp>(loc, reduce_out_type, sum);

    if (use_log) {
      Value *log = rewriter.create<xla_hlo::LogOp>(loc, reduce_out_type, sum);
      rewriter.replaceOpWithNewOp<xla_hlo::SubOp>(
          op, op.getType(), shifted_logits, log, batch_dims);
    } else {
      rewriter.replaceOpWithNewOp<xla_hlo::DivOp>(op, op.getType(), exp, sum,
                                                  batch_dims);
    }

    return Pattern::matchSuccess();
  }
};

// Converts StridedSlice op to HLO Slice op along with Reverse op to handle
// negative strides and Reshape op to update the output shape. Indices and
// strides operands are converted to attributes with non-negative indexing.
//
// For example with an op like following,
//   tf.StridedSlice(%input, %begin, %end, %strides) {shrink_axis_mask = 1}
//     : tensor<AxBxf32> -> tensor<Pxf32>
//
// Output would be:
//   %reversed = "xla_hlo.Reverse" (%input) {dimensions = ...}
//   %sliced = "xla_hlo.Slice" (%input)
//             {start_indices = ..., limit_indices = ..., strides = ...}
//   %output = "xla_hlo.Reshape" (%sliced) : tensor<1xPxf32> -> tensor<Pxf32>
//
class ConvertStridedSliceOp : public OpRewritePattern<TF::StridedSliceOp> {
 public:
  explicit ConvertStridedSliceOp(MLIRContext *context)
      : OpRewritePattern<TF::StridedSliceOp>(context, 1) {}

  PatternMatchResult matchAndRewrite(TF::StridedSliceOp op,
                                     PatternRewriter &rewriter) const override {
    // Input shape needs to be static to convert negative indices in TensorFlow
    // to absolute indices required by HLO.
    //
    // TODO(hinsu): Relax this constraint for ops without negative indices and
    // strides.
    auto input_ty = op.input()->getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) return matchFailure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();

    // Output shape needs to be static to apply 'new_axis_mask' or
    // 'shrink_axis_mask' by reshaping tensor after slice.
    //
    // TODO(hinsu): Relax this constraint for ops without the above masks.
    auto result_ty = op.getType().dyn_cast<RankedTensorType>();
    if (!result_ty || !result_ty.hasStaticShape()) return matchFailure();

    // TODO(hinsu): Support non-zero mask values. Currently only
    // 'shrink_axis_mask' is supported.
    for (StringRef mask :
         {"begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask"}) {
      auto attr = op.getAttrOfType<IntegerAttr>(mask);
      if (attr && attr.getValue() != 0) return matchFailure();
    }

    // TODO(hinsu): Support lowering for ops with dynamic begin and end values
    // when it is possible to derive indices based on mask attributes.
    DenseIntElementsAttr begin_indices, end_indices, strides;
    if (!matchPattern(op.begin(), m_Constant(&begin_indices)) ||
        !matchPattern(op.end(), m_Constant(&end_indices)) ||
        !matchPattern(op.strides(), m_Constant(&strides)))
      return matchFailure();

    SmallVector<int64_t, 4> hlo_begin_indices, hlo_end_indices, hlo_strides,
        dims_to_reverse;
    int64_t input_rank = input_ty.getRank();
    for (auto *vec : {&hlo_begin_indices, &hlo_end_indices, &hlo_strides}) {
      vec->reserve(input_rank);
    }

    int64_t indices_elements = begin_indices.getNumElements();
    if (input_rank < indices_elements) return matchFailure();

    // Convert from TensorFlow negative or out of range indices and strides
    // values to legal HLO Slice attributes.
    for (int i = 0, e = indices_elements; i != e; i++) {
      int64_t begin = begin_indices.getValue<IntegerAttr>(i).getInt();
      int64_t end = end_indices.getValue<IntegerAttr>(i).getInt();
      int64_t stride = strides.getValue<IntegerAttr>(i).getInt();

      if (begin < 0) begin = input_shape[i] + begin;
      if (end < 0) end = input_shape[i] + end;

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
    auto reversed = rewriter.create<xla_hlo::ReverseOp>(
        loc, input_ty, op.input(),
        GetI64ElementsAttr(dims_to_reverse, &rewriter));
    auto sliced = rewriter.create<xla_hlo::SliceOp>(
        loc, reversed.getResult(),
        GetI64ElementsAttr(hlo_begin_indices, &rewriter),
        GetI64ElementsAttr(hlo_end_indices, &rewriter),
        GetI64ElementsAttr(hlo_strides, &rewriter));

    // Reshape slice result so that the shape is updated depending on
    // 'new_axis_mask' or 'shrink_axis_mask' attributes.
    rewriter.replaceOpWithNewOp<xla_hlo::ReshapeOp>(op, op.getType(), sliced);
    return matchSuccess();
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

  PatternMatchResult matchAndRewrite(OpTy op,
                                     PatternRewriter &rewriter) const override {
    // TODO(b/141785544): Update this to not require static shapes.
    // Input shape needs to be static to convert negative indices in TensorFlow
    // to absolute indices required by HLO.
    auto input_ty = op.input()->getType().template dyn_cast<RankedTensorType>();
    if (!input_ty) return this->matchFailure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();

    DenseIntElementsAttr dimensions;
    if (!matchPattern(op.reduction_indices(), m_Constant(&dimensions)) ||
        dimensions.getType().getRank() != 1)
      return this->matchFailure();

    // Build the final shape from input_shape and dimensions using a bitmap
    // to mark the reduced dimensions.
    SmallVector<bool, 4> reduced_dimensions_bitmap(input_shape.size(), false);
    SmallVector<int64_t, 4> xla_dimensions;
    for (APInt index_raw : dimensions.getValues<APInt>()) {
      int64_t index = index_raw.getSExtValue();
      int64_t rank = input_shape.size();
      if ((index < -rank || index >= rank)) return this->matchFailure();
      index = (index + rank) % rank;
      reduced_dimensions_bitmap[index] = true;
      xla_dimensions.push_back(index);
    }
    SmallVector<int64_t, 4> reduced_shape;
    reduced_shape.reserve(input_shape.size());
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (!reduced_dimensions_bitmap[i]) {
        // If we are not reducing along dimension i.
        int64_t dim = input_shape[i];
        reduced_shape.push_back(dim);
      }
    }

    Location loc = op.getLoc();
    Type element_type = input_ty.getElementType();
    // Convert to an accumulation type to not lose precision when doing
    // repeated arithmetic operations.
    Type reduce_element_type =
        is_accumulation ? GetAccumulationType(element_type) : element_type;
    auto casted_input = rewriter.create<xla_hlo::ConvertOp>(
        loc, rewriter.getTensorType(input_shape, reduce_element_type),
        op.input());

    // Each reduction op can have a different initial value.
    Value *init = Derived::GetInitialValue(reduce_element_type, loc, rewriter);

    Type reduced_out_type =
        rewriter.getTensorType(reduced_shape, reduce_element_type);
    // TODO(hinsu): Infer reduced_out_type.
    auto reduction = rewriter.create<xla_hlo::ReduceOp>(
        loc, reduced_out_type, casted_input.getResult(), init,
        GetI64ElementsAttr(xla_dimensions, &rewriter));
    BuildReduceBody<ReductionOp>(reduce_element_type, &reduction.body(),
                                 &rewriter);
    Value *result = reduction.getResult(0);

    // The mean op needs to divide by the product of the reduced dimensions.
    if (std::is_same<OpTy, TF::MeanOp>::value) {
      int64_t divisor_count = 1;
      for (size_t i = 0; i < input_shape.size(); ++i) {
        if (reduced_dimensions_bitmap[i]) {
          if (TensorType::isDynamic(input_shape[i])) {
            return this->matchFailure();
          }
          divisor_count *= input_shape[i];
        }
      }
      auto divisor =
          GetScalarForType(reduce_element_type, loc, divisor_count, &rewriter);
      result = rewriter.create<xla_hlo::DivOp>(
          loc, reduced_out_type, result, divisor.getResult(),
          /* broadcast_dimensions= */ DenseIntElementsAttr());
    }

    Type reduced_final_type =
        rewriter.getTensorType(reduced_shape, element_type);
    result =
        rewriter.create<xla_hlo::ConvertOp>(loc, reduced_final_type, result);

    // Need to reshape back after the reduction if we're keeping the reduced
    // dimensions.
    if (op.keep_dims()) {
      result = rewriter.create<xla_hlo::ReshapeOp>(loc, op.getType(), result);
    }
    rewriter.replaceOp(op, {result}, {op.reduction_indices()});

    return this->matchSuccess();
  }
};

// Converts Mean op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %sum = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.add"]
//               {dimensions = ...}
//   %divisor = constant dense<...> : tensor<T>
//   %mean = "xla_hlo.div"(%sum, %divisor)
class ConvertMeanOp
    : public GenericConvertReductionOp<ConvertMeanOp, TF::MeanOp,
                                       xla_hlo::AddOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value *GetInitialValue(Type reduce_element_type, Location loc,
                                PatternRewriter &rewriter) {
    return GetScalarForType(reduce_element_type, loc, 0, &rewriter);
  }
};

// Converts Sum op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %sum = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.add"]
//               {dimensions = ...}
class ConvertSumOp : public GenericConvertReductionOp<ConvertSumOp, TF::SumOp,
                                                      xla_hlo::AddOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value *GetInitialValue(Type reduce_element_type, Location loc,
                                PatternRewriter &rewriter) {
    return GetScalarForType(reduce_element_type, loc, 0, &rewriter);
  }
};

// Converts Max op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %max = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.max"]
//               {dimensions = ...}
class ConvertMaxOp
    : public GenericConvertReductionOp<ConvertMaxOp, TF::MaxOp, xla_hlo::MaxOp,
                                       /* is_accumulation= */ false> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value *GetInitialValue(Type reduce_element_type, Location loc,
                                PatternRewriter &rewriter) {
    return GetMinValueForType(reduce_element_type, loc, &rewriter);
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

  PatternMatchResult matchAndRewrite(OpTy op,
                                     PatternRewriter &rewriter) const override {
    RankedTensorType input_type =
        op.input()->getType().template dyn_cast<RankedTensorType>();
    if (!input_type) {
      return this->matchFailure();
    }

    Type input_element_type = input_type.getElementType();
    // TODO(bixia): Clarify whether tf.ArgMax supports complex data types. If
    // tf.ArgMax doesn't support complex data types, this check can be removed.
    if (!input_element_type.isIntOrFloat()) return this->matchFailure();

    Location loc = op.getLoc();
    Value *init_value =
        Derived::GetInitialValue(input_element_type, loc, rewriter);

    RankedTensorType output_type =
        op.output()->getType().template dyn_cast<RankedTensorType>();
    if (!output_type) {
      return this->matchFailure();
    }

    Type index_element_type = output_type.getElementType();
    Value *index_init_value =
        GetScalarForType(index_element_type, loc, 0, &rewriter);

    RankedTensorType index_type =
        rewriter.getTensorType(input_type.getShape(), index_element_type);

    llvm::Optional<int64_t> optional_axis =
        GetIntegerHLOAxisFromTFAxis(op.dimension(), input_type.getRank());
    if (!optional_axis.hasValue()) {
      return this->matchFailure();
    }
    int64_t axis = optional_axis.getValue();

    IntegerAttr iota_dimension =
        IntegerAttr::get(rewriter.getIntegerType(64), axis);
    Value *index_values =
        rewriter.create<xla_hlo::IotaOp>(loc, index_type, iota_dimension);

    std::vector<int64_t> dimensions = input_type.getShape();
    dimensions.erase(dimensions.begin() + axis);
    ArrayRef<int64_t> reduction_result_shape(dimensions);

    Type input_reduction_result_type = rewriter.getTensorType(
        reduction_result_shape, input_type.getElementType());
    Type index_reduction_result_type = rewriter.getTensorType(
        reduction_result_shape, index_type.getElementType());

    Type result_types[] = {input_reduction_result_type,
                           index_reduction_result_type};
    Value *operands[] = {op.input(), index_values};
    Value *init_values[] = {init_value, index_init_value};
    DenseIntElementsAttr reduction_dimensions =
        GetI64ElementsAttr({axis}, &rewriter);

    auto reduction = rewriter.create<xla_hlo::ReduceOp>(
        loc, llvm::ArrayRef<Type>(result_types),
        llvm::ArrayRef<Value *>(operands), llvm::ArrayRef<Value *>(init_values),
        reduction_dimensions);
    StringRef direction = Derived::GetDirection();
    BuildArgMinMaxReductionBody(input_element_type, index_element_type,
                                direction, &reduction.body(), &rewriter);

    rewriter.replaceOp(op, {reduction.getResult(1)});
    return this->matchSuccess();
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

  static Value *GetInitialValue(Type reduce_element_type, Location loc,
                                PatternRewriter &rewriter) {
    return GetMinValueForType(reduce_element_type, loc, &rewriter);
  }

  static StringRef GetDirection() { return "GT"; }
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

  PatternMatchResult matchAndRewrite(TF::TileOp op,
                                     PatternRewriter &rewriter) const override {
    auto input_ty = op.input()->getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) return matchFailure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();
    Type element_type = input_ty.getElementType();

    DenseIntElementsAttr multiples;
    if (!matchPattern(op.multiples(), m_Constant(&multiples)) ||
        multiples.getType().getRank() != 1)
      return matchFailure();

    if (multiples.getNumElements() != input_shape.size()) return matchFailure();

    SmallVector<int64_t, 8> broadcasted_shape;
    SmallVector<int64_t, 4> broadcast_dimensions;
    broadcasted_shape.reserve(input_shape.size() * 2);
    broadcast_dimensions.reserve(input_shape.size());
    for (auto multiple_and_input :
         llvm::zip(multiples.getValues<APInt>(), input_shape)) {
      int64_t multiple = std::get<0>(multiple_and_input).getSExtValue();
      int64_t input_size = std::get<1>(multiple_and_input);

      if (multiple < 0) return matchFailure();

      // Line input up with the next dimension in broadcasted_shape
      // when broadcasting.
      broadcast_dimensions.push_back(broadcasted_shape.size());
      int64_t output_size = input_size * multiple;
      if (input_size == 1 || multiple == 1) {
        // Special case for when normal broadcasting will just work.
        broadcasted_shape.push_back(output_size);
      } else {
        // Tiling will happen for this dimension during the ReshapeOp below.
        broadcasted_shape.push_back(input_size);
        broadcasted_shape.push_back(multiple);
      }
    }
    Location loc = op.getLoc();
    Type broadcasted_type =
        rewriter.getTensorType(broadcasted_shape, element_type);
    Type output_type = op.getType();

    Value *result = rewriter.create<xla_hlo::BroadcastInDimOp>(
        loc, broadcasted_type, op.input(),
        GetI64ElementsAttr(broadcast_dimensions, &rewriter));

    if (output_type != broadcasted_type) {
      result = rewriter.create<xla_hlo::ReshapeOp>(loc, output_type, result);
    }

    rewriter.replaceOp(op, {result}, {op.multiples()});

    return matchSuccess();
  }
};

#include "tensorflow/compiler/mlir/xla/transforms/generated_legalize_tf.inc"
}  // end anonymous namespace
}  // end namespace xla
}  // end namespace mlir

LogicalResult mlir::xla_hlo::legalizeTF(Operation *op) {
  MLIRContext *context = op->getContext();

  // Add lowering patterns to the list.
  OwningRewritePatternList patterns;
  xla::populateWithGenerated(context, &patterns);

  // Add patterns that lower some of the high level TensorFlow ops to lower
  // level TensorFlow ops. So, we don't have to target all the TensorFlow ops
  // here for lowering to HLO.
  mlir::TF::PopulateLoweringTFPatterns(context, &patterns);
  patterns.insert<mlir::xla::ConvertArgMaxOp, mlir::xla::ConvertConv2D,
                  mlir::xla::ConvertMaxPoolOp, mlir::xla::ConvertSigmoidOp,
                  mlir::xla::ConvertSoftmaxOp<TF::LogSoftmaxOp, true>,
                  mlir::xla::ConvertSoftmaxOp<TF::SoftmaxOp, false>,
                  mlir::xla::ConvertStridedSliceOp, mlir::xla::ConvertMeanOp,
                  mlir::xla::ConvertSumOp, mlir::xla::ConvertMaxOp,
                  mlir::xla::ConvertTileOp>(op->getContext());

  ConversionTarget target(*context);
  target.addLegalDialect<XlaHloDialect>();

  return applyPartialConversion(op, target, patterns);
}

/// Performs the lowering to XLA dialect.
void LegalizeTF::runOnFunction() {
  if (failed(mlir::xla_hlo::legalizeTF(getFunction()))) signalPassFailure();
}

static PassRegistration<LegalizeTF> pass(
    "xla-legalize-tf", "Legalize from TensorFlow to the XLA dialect");
