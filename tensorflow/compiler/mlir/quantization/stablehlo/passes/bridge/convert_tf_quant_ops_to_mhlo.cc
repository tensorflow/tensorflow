/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/tf_type_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/utils.h"
#include "xla/hlo/translate/hlo_to_mhlo/attribute_importer.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_attr.pb.h"

namespace mlir::quant::stablehlo {
namespace {

using quant::tensorflow::GetDenseAttrFromTensorProtoAttr;
using quant::tensorflow::GetIntTypeFromTFQint;
using quant::tensorflow::IsTFQintType;

#define GEN_PASS_DEF_CONVERTTFQUANTOPSTOMHLO
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"

template <typename UniformQuantizedOp>
FailureOr<TensorType> GetUniformQuantizedType(
    UniformQuantizedOp op, Type original_type,
    TypedValue<TensorType> scales_value,
    TypedValue<TensorType> zero_points_value, FloatType expressed_type,
    int64_t storage_type_min, int64_t storage_type_max,
    int64_t quantized_dimension, PatternRewriter &rewriter) {
  // Check whether the scales operand has constant op.
  DenseFPElementsAttr scales;
  if (!matchPattern(scales_value, m_Constant(&scales))) {
    return rewriter.notifyMatchFailure(op, "scales must be constant");
  }

  // Check whether the zero_points operand has constant op.
  DenseIntElementsAttr zero_points;
  if (!matchPattern(zero_points_value, m_Constant(&zero_points))) {
    return rewriter.notifyMatchFailure(op, "zero_points must be constant");
  }

  auto original_element_type = getElementTypeOrSelf(original_type);
  if (!mlir::isa<TF::Qint8Type, TF::Quint8Type, TF::Qint32Type>(
          original_element_type)) {
    return rewriter.notifyMatchFailure(
        op, "Quantized type must be qint8, quint8 or qint32.");
  }
  auto storage_type = GetIntTypeFromTFQint(original_element_type);

  const unsigned flags = mlir::isa<TF::Quint8Type>(original_element_type)
                             ? 0
                             : quant::QuantizationFlags::Signed;
  Type elem_ty;
  if (quantized_dimension == -1) {
    elem_ty = quant::UniformQuantizedType::get(
        flags, storage_type, expressed_type, scales.getValues<float>()[0],
        zero_points.getValues<int32_t>()[0], storage_type_min,
        storage_type_max);
  } else {
    SmallVector<double> scales_vec;
    SmallVector<int64_t> zero_points_vec;
    for (auto elem : scales.getValues<float>()) scales_vec.push_back(elem);
    for (auto elem : zero_points.getValues<int32_t>())
      zero_points_vec.push_back(elem);
    elem_ty = quant::UniformQuantizedPerAxisType::get(
        flags, storage_type, expressed_type, scales_vec, zero_points_vec,
        quantized_dimension, storage_type_min, storage_type_max);
  }

  return mlir::cast<TensorType>(original_type).clone(elem_ty);
}

// If operand is TF const op, create MHLO constant op from the contents.
// Otherwise convert the operand to the desired type.
FailureOr<Value> CreateConstantOrConvertOp(Operation *op, Value operand,
                                           TensorType new_operand_type,
                                           PatternRewriter &rewriter) {
  // Check whether the rhs operand has constant op.
  TF::TensorProtoAttr tensor_proto_attr;
  if (!matchPattern(operand, m_Constant(&tensor_proto_attr))) {
    return Value(mhlo::BitcastConvertOp::create(rewriter, op->getLoc(),
                                                new_operand_type, operand));
  }

  auto dense_attr_or = GetDenseAttrFromTensorProtoAttr(
      tensor_proto_attr.getValue(), new_operand_type);
  if (failed(dense_attr_or)) return failure();

  return Value(mhlo::ConstantOp::create(rewriter, op->getLoc(),
                                        new_operand_type, *dense_attr_or));
}

xla::ConvolutionDimensionNumbers ConvertConvolutionDimensionNumbers(
    const ::tensorflow::UniformQuantizedConvolutionDimensionNumbersAttr
        &dnums_input) {
  xla::ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(dnums_input.input_batch_dimension());
  dnums.set_input_feature_dimension(dnums_input.input_feature_dimension());
  for (auto value : dnums_input.input_spatial_dimensions()) {
    dnums.add_input_spatial_dimensions(value);
  }
  dnums.set_kernel_input_feature_dimension(
      dnums_input.kernel_input_feature_dimension());
  dnums.set_kernel_output_feature_dimension(
      dnums_input.kernel_output_feature_dimension());
  for (auto value : dnums_input.kernel_spatial_dimensions()) {
    dnums.add_kernel_spatial_dimensions(value);
  }
  dnums.set_output_batch_dimension(dnums_input.output_batch_dimension());
  dnums.set_output_feature_dimension(dnums_input.output_feature_dimension());
  for (auto value : dnums_input.output_spatial_dimensions()) {
    dnums.add_output_spatial_dimensions(value);
  }
  return dnums;
}

DenseIntElementsAttr ConvertToDenseElementsAttr(ArrayAttr array_attr,
                                                PatternRewriter &rewriter) {
  SmallVector<int64_t> array;
  array.reserve(array_attr.size());
  for (auto elem : array_attr.getAsRange<IntegerAttr>()) {
    array.push_back(elem.getInt());
  }
  return DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(array_attr.size())},
                            rewriter.getIntegerType(64)),
      array);
}

LogicalResult ValidateDimensionNumbersPermutation(
    Operation* op, int64_t rank, int64_t dim0, int64_t dim1,
    llvm::ArrayRef<int64_t> spatial_dims, llvm::StringRef tensor_name) {
  const int64_t num_spatial_dims = rank - 2;
  if (static_cast<int64_t>(spatial_dims.size()) != num_spatial_dims) {
    return op->emitError() << tensor_name
                           << " spatial dimensions size must be rank - 2 ("
                           << num_spatial_dims << "), got "
                           << spatial_dims.size() << ".";
  }
  SmallVector<bool, 8> seen(static_cast<size_t>(rank), false);
  auto mark = [&](int64_t dim) -> LogicalResult {
    if (dim < 0 || dim >= rank || seen[static_cast<size_t>(dim)]) {
      return op->emitError()
             << tensor_name
             << " dimension numbers must be a permutation of [0, " << rank
             << "), got invalid or duplicate dimension " << dim << ".";
    }
    seen[static_cast<size_t>(dim)] = true;
    return success();
  };
  if (failed(mark(dim0)) || failed(mark(dim1))) {
    return failure();
  }
  for (int64_t dim : spatial_dims) {
    if (failed(mark(dim))) {
      return failure();
    }
  }
  return success();
}

LogicalResult ValidateConvolutionDimensionNumbers(
    Operation* op, const xla::ConvolutionDimensionNumbers& dnums,
    int64_t rank) {
  if (failed(ValidateDimensionNumbersPermutation(
          op, rank, dnums.input_batch_dimension(),
          dnums.input_feature_dimension(), dnums.input_spatial_dimensions(),
          "input"))) {
    return failure();
  }
  if (failed(ValidateDimensionNumbersPermutation(
          op, rank, dnums.kernel_input_feature_dimension(),
          dnums.kernel_output_feature_dimension(),
          dnums.kernel_spatial_dimensions(), "kernel"))) {
    return failure();
  }
  if (failed(ValidateDimensionNumbersPermutation(
          op, rank, dnums.output_batch_dimension(),
          dnums.output_feature_dimension(), dnums.output_spatial_dimensions(),
          "output"))) {
    return failure();
  }
  return success();
}

// Returns per-spatial-dim stride or dilation values. An omitted or empty
// attribute means 1 for every spatial dim, matching the TF kernel and the
// mhlo default.
FailureOr<SmallVector<int64_t>> GetSpatialAttrOrOnes(Operation* op,
                                                     ArrayAttr attr,
                                                     int64_t num_spatial_dims,
                                                     llvm::StringRef name) {
  SmallVector<int64_t> values(static_cast<size_t>(num_spatial_dims), 1);
  if (!attr || attr.empty()) {
    return values;
  }
  if (static_cast<int64_t>(attr.size()) != num_spatial_dims) {
    return op->emitError() << name << " must have size " << num_spatial_dims
                           << ", got " << attr.size() << ".";
  }
  for (size_t i = 0; i < attr.size(); ++i) {
    const int64_t value = mlir::cast<IntegerAttr>(attr[i]).getInt();
    if (value < 1) {
      return op->emitError()
             << name << " elements must be >= 1, got " << value << ".";
    }
    values[i] = value;
  }
  return values;
}

// Requires size >= 0 and dilation >= 1.
FailureOr<int64_t> ComputeDilatedSize(Operation* op, int64_t size,
                                      int64_t dilation,
                                      llvm::StringRef attr_name) {
  if (size <= 1) {
    return size;
  }
  if (dilation - 1 >
      (std::numeric_limits<int64_t>::max() - size) / (size - 1)) {
    return op->emitError() << "dilated size overflow for " << attr_name << ".";
  }
  return size + (dilation - 1) * (size - 1);
}

template <typename UniformQuantizedConvolutionOp>
FailureOr<ElementsAttr> ConvertPaddingAttr(
    UniformQuantizedConvolutionOp op, llvm::ArrayRef<int64_t> window_strides,
    llvm::ArrayRef<int64_t> lhs_dilated, llvm::ArrayRef<int64_t> rhs_dilated,
    PatternRewriter& rewriter) {
  // window_strides, lhs_dilated, and rhs_dilated have one entry per spatial
  // dim; the caller builds all three with the same size.
  const int64_t num_spatial_dims = static_cast<int64_t>(lhs_dilated.size());
  const int64_t padding_nums_size = 2 * num_spatial_dims;
  SmallVector<int64_t> padding_nums;
  const llvm::StringRef padding_str = op.getPaddingAttr().getValue();
  if (padding_str != "EXPLICIT" && padding_str != "VALID" &&
      padding_str != "SAME") {
    return op.emitError(
               "padding must be one of EXPLICIT, VALID, or SAME, got '")
           << padding_str << "'.";
  }
  ArrayAttr explicit_padding = op.getExplicitPaddingAttr();
  if (padding_str != "EXPLICIT" && explicit_padding &&
      !explicit_padding.empty()) {
    return op.emitError("explicit_padding must be empty unless padding is ")
           << "EXPLICIT, got size " << explicit_padding.size() << ".";
  }
  if (padding_str == "EXPLICIT") {
    const size_t explicit_padding_size =
        explicit_padding ? explicit_padding.size() : 0;
    if (explicit_padding_size != static_cast<size_t>(padding_nums_size)) {
      return op.emitError("explicit_padding must have size ")
             << padding_nums_size << ", got " << explicit_padding_size << ".";
    }
    padding_nums.reserve(explicit_padding_size);
    for (auto padding_elem : explicit_padding.getAsRange<IntegerAttr>()) {
      const int64_t value = padding_elem.getInt();
      if (value < 0) {
        return op.emitError("explicit_padding elements must be >= 0, got ")
               << value << ".";
      }
      padding_nums.push_back(value);
    }
  } else if (padding_str == "VALID") {
    padding_nums.assign(static_cast<size_t>(padding_nums_size), 0);
  } else {
    padding_nums.resize(static_cast<size_t>(padding_nums_size));
    for (size_t i = 0; i < static_cast<size_t>(num_spatial_dims); ++i) {
      const int64_t stride = window_strides[i];
      // Compute max((ceil(lhs / stride) - 1) * stride + rhs - lhs, 0) without
      // intermediate signed integer overflow.
      const int64_t lhs_offset =
          lhs_dilated[i] == 0 ? stride : ((lhs_dilated[i] - 1) % stride) + 1;
      const int64_t total_padding =
          std::max(rhs_dilated[i] - lhs_offset, static_cast<int64_t>(0));
      const int64_t padding_begin = total_padding / 2;
      const int64_t padding_end = total_padding - padding_begin;
      padding_nums[2 * i] = padding_begin;
      padding_nums[2 * i + 1] = padding_end;
    }
  }

  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < static_cast<size_t>(num_spatial_dims); ++i) {
    const int64_t low = padding_nums[2 * i];
    const int64_t high = padding_nums[2 * i + 1];
    if (low > kMax - lhs_dilated[i] || high > kMax - lhs_dilated[i] - low) {
      return op.emitError("padded lhs size overflow.");
    }
  }

  ElementsAttr padding_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({num_spatial_dims, 2}, rewriter.getIntegerType(64)),
      padding_nums);
  return padding_attr;
}

template <typename UniformQuantizedConvolutionOp>
FailureOr<SmallVector<NamedAttribute>> ConvertToMhloConvolutionOpAttrs(
    UniformQuantizedConvolutionOp op, PatternRewriter& rewriter) {
  // TODO(b/261005147): Update the lowering logic after migration to mhlo
  // ConvolutionDimensionNumbers.
  ::tensorflow::UniformQuantizedConvolutionDimensionNumbersAttr dnums_input;
  if (!dnums_input.ParseFromString(std::string(op.getDimensionNumbers()))) {
    return op->emitError("Parse dimension_numbers failed.");
  }
  xla::ConvolutionDimensionNumbers dnums =
      ConvertConvolutionDimensionNumbers(dnums_input);

  ShapedType lhs_shape = mlir::cast<ShapedType>(op.getLhs().getType());
  ShapedType rhs_shape = mlir::cast<ShapedType>(op.getRhs().getType());

  // Handle only static shape cases.
  // TODO(b/260284866): Handle dynamic shape cases.
  if (!lhs_shape.hasStaticShape()) {
    return op.emitError("lhs must have static shape.");
  }
  if (!rhs_shape.hasStaticShape()) {
    return op.emitError("rhs must have static shape.");
  }
  const int64_t lhs_rank = lhs_shape.getRank();
  const int64_t rhs_rank = rhs_shape.getRank();
  if (lhs_rank != rhs_rank) {
    return op.emitError("lhs and rhs must have the same rank, got ")
           << lhs_rank << " and " << rhs_rank << ".";
  }
  if (rhs_rank < 3) {
    return op.emitError("lhs and rhs rank must be at least 3, got ")
           << rhs_rank << ".";
  }
  // TODO(b/261005147): Empty dimension_numbers means the default
  // NCHW/OIHW/NCHW layout in the TF kernel; this lowering rejects it.
  if (failed(ValidateConvolutionDimensionNumbers(op, dnums, rhs_rank))) {
    return failure();
  }

  const int64_t num_spatial_dims = rhs_rank - 2;
  const FailureOr<SmallVector<int64_t>> window_strides = GetSpatialAttrOrOnes(
      op, op.getWindowStridesAttr(), num_spatial_dims, "window_strides");
  if (failed(window_strides)) {
    return failure();
  }
  const FailureOr<SmallVector<int64_t>> lhs_dilation = GetSpatialAttrOrOnes(
      op, op.getLhsDilationAttr(), num_spatial_dims, "lhs_dilation");
  if (failed(lhs_dilation)) {
    return failure();
  }
  const FailureOr<SmallVector<int64_t>> rhs_dilation = GetSpatialAttrOrOnes(
      op, op.getRhsDilationAttr(), num_spatial_dims, "rhs_dilation");
  if (failed(rhs_dilation)) {
    return failure();
  }

  SmallVector<int64_t> lhs_dilated;
  SmallVector<int64_t> rhs_dilated;
  lhs_dilated.reserve(static_cast<size_t>(num_spatial_dims));
  rhs_dilated.reserve(static_cast<size_t>(num_spatial_dims));
  for (int64_t i = 0; i < num_spatial_dims; ++i) {
    const size_t idx = static_cast<size_t>(i);
    const FailureOr<int64_t> lhs_size = ComputeDilatedSize(
        op,
        lhs_shape.getDimSize(
            dnums.input_spatial_dimensions(static_cast<int>(i))),
        (*lhs_dilation)[idx], "lhs_dilation");
    if (failed(lhs_size)) {
      return failure();
    }
    const FailureOr<int64_t> rhs_size = ComputeDilatedSize(
        op,
        rhs_shape.getDimSize(
            dnums.kernel_spatial_dimensions(static_cast<int>(i))),
        (*rhs_dilation)[idx], "rhs_dilation");
    if (failed(rhs_size)) {
      return failure();
    }
    lhs_dilated.push_back(*lhs_size);
    rhs_dilated.push_back(*rhs_size);
  }

  // Read IntegerAttr::getInt() directly: the ODS getters return uint64_t
  // via getZExtValue(), so negative values would pass the < 1 check.
  auto group_count_or_one = [](IntegerAttr attr) -> int64_t {
    return attr ? attr.getInt() : 1;
  };
  const int64_t feature_group_count =
      group_count_or_one(op.getFeatureGroupCountAttr());
  const int64_t batch_group_count =
      group_count_or_one(op.getBatchGroupCountAttr());
  if (feature_group_count < 1 || batch_group_count < 1) {
    return op.emitError("feature_group_count and batch_group_count must be ")
           << ">= 1, got " << feature_group_count << " and "
           << batch_group_count << ".";
  }

  SmallVector<NamedAttribute> converted_attrs;
  converted_attrs.push_back(
      rewriter.getNamedAttr(op.getFeatureGroupCountAttrName(),
                            rewriter.getI64IntegerAttr(feature_group_count)));
  converted_attrs.push_back(
      rewriter.getNamedAttr(op.getBatchGroupCountAttrName(),
                            rewriter.getI64IntegerAttr(batch_group_count)));
  for (auto attr : op->getAttrs()) {
    if (attr.getName() == op.getDimensionNumbersAttrName()) {
      attr.setValue(xla::ConvertConvDimensionNumbers(dnums, &rewriter));
      converted_attrs.push_back(attr);
    } else if (attr.getName() == op.getPaddingAttrName()) {
      auto value_or = ConvertPaddingAttr(op, *window_strides, lhs_dilated,
                                         rhs_dilated, rewriter);
      if (failed(value_or)) {
        return failure();
      }
      attr.setValue(*value_or);
      converted_attrs.push_back(attr);
    } else if (attr.getName() == op.getWindowStridesAttrName() ||
               attr.getName() == op.getLhsDilationAttrName() ||
               attr.getName() == op.getRhsDilationAttrName()) {
      auto array_attr = mlir::cast<ArrayAttr>(attr.getValue());
      if (!array_attr.empty()) {
        attr.setValue(ConvertToDenseElementsAttr(array_attr, rewriter));
        converted_attrs.push_back(attr);
      }
    }
  }
  return converted_attrs;
}

// TODO(hinsu): Move this pattern to legalize_tf after resolving the dependency
// on the tensor proto.
class ConvertUniformQuantizedDotHybridOp
    : public OpConversionPattern<TF::UniformQuantizedDotHybridOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::UniformQuantizedDotHybridOp op,
      TF::UniformQuantizedDotHybridOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Uniform Quantized type for the rhs.
    int64_t rhs_quantized_dimension = op.getRhsQuantizationAxis();
    // Currently for dot, PTQ supports per-tensor quantization.
    if (rhs_quantized_dimension != -1) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports only rhs_quantization_axis -1.");
    }
    auto rhs_type = GetUniformQuantizedType(
        op, op.getRhs().getType(), op.getRhsScales(), op.getRhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), rhs_quantized_dimension, rewriter);
    if (failed(rhs_type)) {
      return failure();
    }

    auto rhs_or =
        CreateConstantOrConvertOp(op, adaptor.getRhs(), *rhs_type, rewriter);
    if (failed(rhs_or)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mhlo::DotOp>(op, op.getType(), op.getLhs(),
                                             *rhs_or,
                                             /*precision_config=*/nullptr);
    return success();
  }
};

class ConvertUniformQuantizedConvolutionHybridOp
    : public OpConversionPattern<TF::UniformQuantizedConvolutionHybridOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::UniformQuantizedConvolutionHybridOp op,
      TF::UniformQuantizedConvolutionHybridOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto converted_attrs_or = ConvertToMhloConvolutionOpAttrs(op, rewriter);
    if (failed(converted_attrs_or)) {
      return failure();
    }

    // Uniform Quantized type for the rhs.
    auto rhs_type = GetUniformQuantizedType(
        op, op.getRhs().getType(), op.getRhsScales(), op.getRhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), op.getRhsQuantizationAxis(), rewriter);
    if (failed(rhs_type)) {
      return failure();
    }

    auto rhs_or =
        CreateConstantOrConvertOp(op, adaptor.getRhs(), *rhs_type, rewriter);
    if (failed(rhs_or)) {
      return failure();
    }

    SmallVector<Value, 2> operands{op.getLhs(), *rhs_or};
    rewriter.replaceOpWithNewOp<mhlo::ConvolutionOp>(op, op.getType(), operands,
                                                     *converted_attrs_or);
    return success();
  }
};

class ConvertUniformQuantizeOp
    : public OpRewritePattern<TF::UniformQuantizeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::UniformQuantizeOp op,
                                PatternRewriter &rewriter) const override {
    auto output_type = GetUniformQuantizedType(
        op, op.getOutput().getType(), op.getScales(), op.getZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getQuantizationMinVal(),
        op.getQuantizationMaxVal(), op.getQuantizationAxis(), rewriter);
    if (failed(output_type)) {
      return failure();
    }

    auto result = mhlo::UniformQuantizeOp::create(rewriter, op->getLoc(),
                                                  *output_type, op.getInput());
    rewriter.replaceOpWithNewOp<mhlo::BitcastConvertOp>(
        op,
        output_type->clone(
            mlir::dyn_cast<quant::QuantizedType>(output_type->getElementType())
                .getStorageType()),
        result);

    return success();
  }
};

// UniformDequantizeOp takes TF quantized types as input which would have been
// converted to the mhlo quantized types. Use OpConversionPattern in order to
// retrieve the operand type *after* conversion, using OpAdaptor operand
// accessor.
// Same for other Uniform Quant Ops that take TF quantized types as input.
class ConvertUniformDequantizeOp
    : public OpConversionPattern<TF::UniformDequantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::UniformDequantizeOp op, TF::UniformDequantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto input_quant_type = GetUniformQuantizedType(
        op, op.getInput().getType(), op.getScales(), op.getZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getQuantizationMinVal(),
        op.getQuantizationMaxVal(), op.getQuantizationAxis(), rewriter);
    if (failed(input_quant_type)) {
      return failure();
    }
    input = mhlo::BitcastConvertOp::create(rewriter, op->getLoc(),
                                           *input_quant_type, input);

    rewriter.replaceOpWithNewOp<mhlo::UniformDequantizeOp>(
        op, op.getOutput().getType(), input);
    return success();
  }
};

class ConvertUniformRequantizeOp
    : public OpConversionPattern<TF::UniformRequantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::UniformRequantizeOp op, TF::UniformRequantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto input_quant_type = GetUniformQuantizedType(
        op, op.getInput().getType(), op.getInputScales(),
        op.getInputZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(),
        op.getInputQuantizationMinVal(), op.getInputQuantizationMaxVal(),
        op.getInputQuantizationAxis(), rewriter);
    if (failed(input_quant_type)) {
      return failure();
    }

    auto output_type = GetUniformQuantizedType(
        op, op.getOutput().getType(), op.getOutputScales(),
        op.getOutputZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(),
        op.getOutputQuantizationMinVal(), op.getOutputQuantizationMaxVal(),
        op.getOutputQuantizationAxis(), rewriter);
    if (failed(output_type)) {
      return failure();
    }

    auto input_quant = mhlo::BitcastConvertOp::create(rewriter, op->getLoc(),
                                                      *input_quant_type, input);
    auto result = mhlo::UniformQuantizeOp::create(rewriter, op->getLoc(),
                                                  *output_type, input_quant);
    rewriter.replaceOpWithNewOp<mhlo::BitcastConvertOp>(
        op,
        output_type->clone(
            mlir::dyn_cast<quant::QuantizedType>(output_type->getElementType())
                .getStorageType()),
        result);
    return success();
  }
};

class ConvertUniformQuantizedDotOp
    : public OpConversionPattern<TF::UniformQuantizedDotOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::UniformQuantizedDotOp op, TF::UniformQuantizedDotOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();

    auto lhs_quant_type = GetUniformQuantizedType(
        op, op.getLhs().getType(), op.getLhsScales(), op.getLhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getLhsQuantizationMinVal(),
        op.getLhsQuantizationMaxVal(), op.getLhsQuantizationAxis(), rewriter);
    if (failed(lhs_quant_type)) {
      return failure();
    }
    lhs = mhlo::BitcastConvertOp::create(rewriter, op->getLoc(),
                                         *lhs_quant_type, adaptor.getLhs());

    // Uniform Quantized type for the rhs.
    int64_t rhs_quantized_dimension = op.getRhsQuantizationAxis();
    // Currently for dot, PTQ supports per-tensor quantization.
    if (rhs_quantized_dimension != -1) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports only rhs_quantization_axis -1.");
    }
    auto rhs_type = GetUniformQuantizedType(
        op, op.getRhs().getType(), op.getRhsScales(), op.getRhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), rhs_quantized_dimension, rewriter);
    if (failed(rhs_type)) {
      return failure();
    }

    auto rhs_or =
        CreateConstantOrConvertOp(op, adaptor.getRhs(), *rhs_type, rewriter);
    if (failed(rhs_or)) {
      return failure();
    }

    auto output_type = GetUniformQuantizedType(
        op, op.getOutput().getType(), op.getOutputScales(),
        op.getOutputZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(),
        op.getOutputQuantizationMinVal(), op.getOutputQuantizationMaxVal(),
        op.getOutputQuantizationAxis(), rewriter);
    if (failed(output_type)) {
      return failure();
    }

    auto result =
        mhlo::DotOp::create(rewriter, op->getLoc(), *output_type, lhs, *rhs_or,
                            /*precision_config=*/nullptr);
    rewriter.replaceOpWithNewOp<mhlo::BitcastConvertOp>(
        op,
        output_type->clone(
            mlir::dyn_cast<quant::QuantizedType>(output_type->getElementType())
                .getStorageType()),
        result);
    return success();
  }
};

class ConvertUniformQuantizedConvolutionOp
    : public OpConversionPattern<TF::UniformQuantizedConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::UniformQuantizedConvolutionOp op,
      TF::UniformQuantizedConvolutionOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto converted_attrs_or = ConvertToMhloConvolutionOpAttrs(op, rewriter);
    if (failed(converted_attrs_or)) {
      return failure();
    }

    auto lhs_quant_type = GetUniformQuantizedType(
        op, op.getLhs().getType(), op.getLhsScales(), op.getLhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getLhsQuantizationMinVal(),
        op.getLhsQuantizationMaxVal(), op.getLhsQuantizationAxis(), rewriter);
    if (failed(lhs_quant_type)) {
      return failure();
    }

    auto rhs_type = GetUniformQuantizedType(
        op, op.getRhs().getType(), op.getRhsScales(), op.getRhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), op.getRhsQuantizationAxis(), rewriter);
    if (failed(rhs_type)) {
      return failure();
    }

    auto output_type = GetUniformQuantizedType(
        op, op.getOutput().getType(), op.getOutputScales(),
        op.getOutputZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(),
        op.getOutputQuantizationMinVal(), op.getOutputQuantizationMaxVal(),
        op.getOutputQuantizationAxis(), rewriter);
    if (failed(output_type)) {
      return failure();
    }

    auto rhs_or =
        CreateConstantOrConvertOp(op, adaptor.getRhs(), *rhs_type, rewriter);
    if (failed(rhs_or)) {
      return failure();
    }

    Value lhs = mhlo::BitcastConvertOp::create(
        rewriter, op->getLoc(), *lhs_quant_type, adaptor.getLhs());

    SmallVector<Value, 2> operands{lhs, *rhs_or};
    auto result = mhlo::ConvolutionOp::create(
        rewriter, op->getLoc(), *output_type, operands, *converted_attrs_or);
    rewriter.replaceOpWithNewOp<mhlo::BitcastConvertOp>(
        op,
        output_type->clone(
            mlir::dyn_cast<quant::QuantizedType>(output_type->getElementType())
                .getStorageType()),
        result);
    return success();
  }
};

class ConvertUniformQuantizedAddOp
    : public OpConversionPattern<TF::UniformQuantizedAddOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::UniformQuantizedAddOp op, TF::UniformQuantizedAddOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();

    auto lhs_type = mlir::cast<ShapedType>(lhs.getType());
    if (!lhs_type.hasRank()) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports cases where only lhs rank known.");
    }

    auto lhs_quant_type = GetUniformQuantizedType(
        op, op.getLhs().getType(), op.getLhsScales(), op.getLhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getLhsQuantizationMinVal(),
        op.getLhsQuantizationMaxVal(), op.getLhsQuantizationAxis(), rewriter);
    if (failed(lhs_quant_type)) {
      return failure();
    }
    lhs = mhlo::BitcastConvertOp::create(rewriter, op->getLoc(),
                                         *lhs_quant_type, adaptor.getLhs());

    // rhs (bias) is always 1D that broadcasts to the last dim of lhs.
    auto broadcast_dims =
        rewriter.getDenseI64ArrayAttr({lhs_type.getRank() - 1});

    auto rhs_type = GetUniformQuantizedType(
        op, op.getRhs().getType(), op.getRhsScales(), op.getRhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), op.getRhsQuantizationAxis(), rewriter);
    if (failed(rhs_type)) {
      return failure();
    }

    auto rhs_or =
        CreateConstantOrConvertOp(op, adaptor.getRhs(), *rhs_type, rewriter);
    if (failed(rhs_or)) {
      return failure();
    }

    auto output_type = GetUniformQuantizedType(
        op, op.getOutput().getType(), op.getOutputScales(),
        op.getOutputZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(),
        op.getOutputQuantizationMinVal(), op.getOutputQuantizationMaxVal(),
        op.getOutputQuantizationAxis(), rewriter);
    if (failed(output_type)) {
      return failure();
    }

    // lhs, rhs, output scales and zero_points are guaranteed (by the TF
    // quantizer) to be identical, respectively.
    auto result = chlo::BroadcastAddOp::create(
        rewriter, op->getLoc(), *output_type, lhs, *rhs_or, broadcast_dims);
    rewriter.replaceOpWithNewOp<mhlo::BitcastConvertOp>(
        op,
        output_type->clone(
            mlir::dyn_cast<quant::QuantizedType>(output_type->getElementType())
                .getStorageType()),
        result);
    return success();
  }
};

class ConvertUniformQuantizedClipByValueOp
    : public OpConversionPattern<TF::UniformQuantizedClipByValueOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::UniformQuantizedClipByValueOp op,
      TF::UniformQuantizedClipByValueOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value operand = adaptor.getOperand();

    const int64_t quantization_axis = op.getQuantizationAxis();
    llvm::SmallVector<int64_t> broadcast_dims_values = {};
    if (quantization_axis >= 0) {
      broadcast_dims_values.push_back(quantization_axis);
    }
    auto broadcast_dims = rewriter.getDenseI64ArrayAttr(broadcast_dims_values);

    auto min_max_type = GetUniformQuantizedType(
        op, op.getMin().getType(), op.getScales(), op.getZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getQuantizationMinVal(),
        op.getQuantizationMaxVal(), op.getQuantizationAxis(), rewriter);
    if (failed(min_max_type)) {
      return failure();
    }
    auto min_or = CreateConstantOrConvertOp(op, adaptor.getMin(), *min_max_type,
                                            rewriter);
    if (failed(min_or)) {
      return failure();
    }
    auto max_or = CreateConstantOrConvertOp(op, adaptor.getMax(), *min_max_type,
                                            rewriter);
    if (failed(max_or)) {
      return failure();
    }

    auto output_type = GetUniformQuantizedType(
        op, op.getOutput().getType(), op.getScales(), op.getZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getQuantizationMinVal(),
        op.getQuantizationMaxVal(), op.getQuantizationAxis(), rewriter);
    if (failed(output_type)) {
      return failure();
    }
    operand = mhlo::BitcastConvertOp::create(rewriter, op->getLoc(),
                                             *output_type, operand);

    Value res_min_clipped = chlo::BroadcastMaxOp::create(
        rewriter, op->getLoc(), *output_type, operand, *min_or, broadcast_dims);
    Value res_max_clipped =
        chlo::BroadcastMinOp::create(rewriter, op->getLoc(), *output_type,
                                     res_min_clipped, *max_or, broadcast_dims);
    rewriter.replaceOpWithNewOp<mhlo::BitcastConvertOp>(
        op,
        output_type->clone(
            mlir::dyn_cast<quant::QuantizedType>(output_type->getElementType())
                .getStorageType()),
        res_max_clipped);
    return success();
  }
};

// This pattern converts qint <-> int CastOp to int -> int ConvertOps.
// The former are introduced in ConvertTFQuantTypes pass. The resulting int <->
// int ConvertOps are no-ops and can be removed later in a Canonicalizer pass.
class ConvertTfCastOp : public OpConversionPattern<TF::CastOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::CastOp op, TF::CastOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type output_type = op.getDstT();
    if (!IsTFQintType(output_type) && !IsTFQintType(op.getSrcT())) {
      // skip CastOps with no qint types.
      return failure();
    }
    Value input = adaptor.getX();
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
        op, input, GetIntTypeFromTFQint(output_type));
    return success();
  }
};

class ConvertTFQuantOpsToMHLO
    : public impl::ConvertTFQuantOpsToMHLOBase<ConvertTFQuantOpsToMHLO> {
 public:
  ConvertTFQuantOpsToMHLO() = default;
  ConvertTFQuantOpsToMHLO(const ConvertTFQuantOpsToMHLO &) = default;

  // Performs conversion of MHLO quant ops to primitive ops.
  void runOnOperation() override;
};

void ConvertTFQuantOpsToMHLO::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp func = getOperation();
  ConversionTarget target(*ctx);
  target.addLegalDialect<TF::TensorFlowDialect, mhlo::MhloDialect,
                         chlo::ChloDialect>();
  target.addIllegalOp<
      TF::UniformQuantizeOp, TF::UniformRequantizeOp, TF::UniformDequantizeOp,
      TF::UniformQuantizedDotOp, TF::UniformQuantizedDotHybridOp,
      TF::UniformQuantizedConvolutionOp,
      TF::UniformQuantizedConvolutionHybridOp, TF::UniformQuantizedAddOp,
      TF::UniformQuantizedClipByValueOp>();
  target.addDynamicallyLegalOp<TF::CastOp>([](Operation *op) {
    auto cast_op = llvm::dyn_cast<TF::CastOp>(op);
    return !IsTFQintType(cast_op.getSrcT()) && !IsTFQintType(cast_op.getDstT());
  });

  RewritePatternSet patterns(ctx);
  PopulateLegalizeTfQuantizationPatterns(ctx, &patterns);
  if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

void PopulateLegalizeTfQuantizationPatterns(MLIRContext *context,
                                            RewritePatternSet *patterns) {
  patterns
      ->add<ConvertUniformQuantizedDotHybridOp,
            ConvertUniformQuantizedConvolutionHybridOp,
            ConvertUniformQuantizeOp, ConvertUniformRequantizeOp,
            ConvertUniformDequantizeOp, ConvertUniformQuantizedDotOp,
            ConvertUniformQuantizedConvolutionOp, ConvertUniformQuantizedAddOp,
            ConvertUniformQuantizedClipByValueOp, ConvertTfCastOp>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertTFQuantOpsToMHLOPass() {
  return std::make_unique<ConvertTFQuantOpsToMHLO>();
}

}  // namespace mlir::quant::stablehlo
