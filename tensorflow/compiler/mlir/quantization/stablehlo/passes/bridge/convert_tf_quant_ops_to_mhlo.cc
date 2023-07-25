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
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/utils.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/attribute_importer.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_attr.pb.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_params.h"

// TODO: b/289560952 - Move to mlir::quant::stablehlo namespace.
namespace mlir {
namespace stablehlo {
namespace {

#define GEN_PASS_DEF_CONVERTTFQUANTOPSTOMHLO
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"

FailureOr<IntegerType> GetStorageType(Operation *op,
                                      Type original_output_element_type,
                                      PatternRewriter &rewriter) {
  if (original_output_element_type.isa<TF::Qint8Type>()) {
    return rewriter.getIntegerType(8);
  } else if (original_output_element_type.isa<TF::Qint32Type>()) {
    return rewriter.getIntegerType(32);
  } else {
    return rewriter.notifyMatchFailure(
        op, "Quantized type must be qint8 or qint32.");
  }
}

TensorType GetSameShapeTensorType(TensorType tensor_type, Type element_type) {
  if (auto ranked_tensor_ty =
          tensor_type.dyn_cast_or_null<RankedTensorType>()) {
    return RankedTensorType::get(ranked_tensor_ty.getShape(), element_type);
  }
  if (auto unranked_tensor_ty =
          tensor_type.dyn_cast_or_null<UnrankedTensorType>()) {
    return UnrankedTensorType::get(element_type);
  }
  llvm_unreachable("unhandled type");
}

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

  auto storage_type_or =
      GetStorageType(op, getElementTypeOrSelf(original_type), rewriter);
  if (failed(storage_type_or)) {
    return failure();
  }

  const unsigned flags = quant::QuantizationFlags::Signed;
  Type elem_ty;
  if (quantized_dimension == -1) {
    elem_ty = quant::UniformQuantizedType::get(
        flags, *storage_type_or, expressed_type, scales.getValues<float>()[0],
        zero_points.getValues<int32_t>()[0], storage_type_min,
        storage_type_max);
  } else {
    SmallVector<double> scales_vec;
    SmallVector<int64_t> zero_points_vec;
    for (auto elem : scales.getValues<float>()) scales_vec.push_back(elem);
    for (auto elem : zero_points.getValues<int32_t>())
      zero_points_vec.push_back(elem);
    elem_ty = quant::UniformQuantizedPerAxisType::get(
        flags, *storage_type_or, expressed_type, scales_vec, zero_points_vec,
        quantized_dimension, storage_type_min, storage_type_max);
  }

  return GetSameShapeTensorType(original_type.cast<TensorType>(), elem_ty);
}

template <typename TFQuantizedType, typename UniformQuantizedOp>
FailureOr<mhlo::ConstantOp> CreateConstantOp(UniformQuantizedOp op,
                                             Value original_operand,
                                             TensorType new_operand_type,
                                             PatternRewriter &rewriter) {
  // Check whether the rhs operand has constant op.
  TF::TensorProtoAttr tensor_proto_attr;
  if (!matchPattern(original_operand, m_Constant(&tensor_proto_attr))) {
    return rewriter.notifyMatchFailure(op, "operand must be constant.");
  }

  llvm::StringRef mangled_tensor = tensor_proto_attr.getValue();
  absl::string_view tensor_view(mangled_tensor.data(), mangled_tensor.size());
  // TODO(hinsu): Instead of getting the weight from TensorProto, use MLIR
  // constant attribute to avoid depending on the Tensor proto.
  tensorflow::TensorProto tensor_proto;
  tensorflow::Status status =
      tensorflow::mangling_util::DemangleTensor(tensor_view, &tensor_proto);
  if (!status.ok()) {
    return rewriter.notifyMatchFailure(op, status.message());
  }

  tensorflow::Tensor t;
  if (!t.FromProto(tensor_proto)) {
    return op.emitError("Failed to convert tensor proto to Tensor.");
  }

  auto arr = t.flat<TFQuantizedType>();
  auto dense_attr = mlir::DenseElementsAttr::get(
      GetSameShapeTensorType(
          new_operand_type,
          rewriter.getIntegerType(8 * sizeof(TFQuantizedType))),
      llvm::ArrayRef(arr.data(), arr.size()));
  return rewriter.create<mhlo::ConstantOp>(op.getLoc(), new_operand_type,
                                           dense_attr);
}

xla::ConvolutionDimensionNumbers ConvertConvolutionDimensionNumbers(
    const tensorflow::UniformQuantizedConvolutionDimensionNumbersAttr
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

template <typename UniformQuantizedConvolutionOp>
FailureOr<ElementsAttr> ConvertPaddingAttr(
    UniformQuantizedConvolutionOp op,
    const xla::ConvolutionDimensionNumbers &dnums, PatternRewriter &rewriter) {
  StringAttr conv_padding = op.getPaddingAttr();
  SmallVector<int64_t> padding_nums;
  ShapedType lhs_shape = op.getLhs().getType().template cast<ShapedType>();
  ShapedType rhs_shape = op.getRhs().getType().template cast<ShapedType>();

  // Handle only static shape cases.
  // TODO(b/260284866): Handle dynamic shape cases.
  if (!lhs_shape.hasStaticShape()) {
    return op.emitError("lhs must have static shape.");
  }
  if (!rhs_shape.hasStaticShape()) {
    return op.emitError("rhs must have static shape.");
  }

  const int64_t padding_nums_size = 2 * (rhs_shape.getRank() - 2);
  padding_nums.reserve(padding_nums_size);
  if (conv_padding.strref().equals("EXPLICIT")) {
    for (auto padding_elem :
         op.getExplicitPaddingAttr().template getAsRange<IntegerAttr>()) {
      padding_nums.push_back(padding_elem.getInt());
    }
  } else if (conv_padding.strref().equals("VALID")) {
    padding_nums.resize(padding_nums_size, 0);
  } else {
    padding_nums.resize(padding_nums_size);
    for (int i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
      const int64_t stride =
          op.getWindowStridesAttr()[i].template cast<IntegerAttr>().getInt();
      const int64_t lhs_size_dilated =
          tensorflow::UniformQuantizedConvolutionParams::DilatedSize(
              lhs_shape.getDimSize(dnums.input_spatial_dimensions(i)),
              op.getLhsDilationAttr()[i].template cast<IntegerAttr>().getInt());
      const int64_t rhs_size_dilated =
          tensorflow::UniformQuantizedConvolutionParams::DilatedSize(
              rhs_shape.getDimSize(dnums.kernel_spatial_dimensions(i)),
              op.getRhsDilationAttr()[i].template cast<IntegerAttr>().getInt());

      const int64_t output_size = (lhs_size_dilated + stride - 1) / stride;
      const int64_t total_padding = std::max(
          (output_size - 1) * stride + rhs_size_dilated - lhs_size_dilated,
          static_cast<int64_t>(0));
      const int64_t padding_end = total_padding / 2;
      const int64_t padding_begin = total_padding - padding_end;
      padding_nums[2 * i] = padding_begin;
      padding_nums[2 * i + 1] = padding_end;
    }
  }

  ElementsAttr padding_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int32_t>(padding_nums.size() / 2), 2},
                            rewriter.getIntegerType(64)),
      padding_nums);
  return padding_attr;
}

template <typename UniformQuantizedConvolutionOp>
FailureOr<SmallVector<NamedAttribute>> ConvertToMhloConvolutionOpAttrs(
    UniformQuantizedConvolutionOp op, PatternRewriter &rewriter) {
  // TODO(b/261005147): Update the lowering logic after migration to mhlo
  // ConvolutionDimensionNumbers.
  tensorflow::UniformQuantizedConvolutionDimensionNumbersAttr dnums_input;
  if (!dnums_input.ParseFromString(std::string(op.getDimensionNumbers()))) {
    return op->emitError("Parse dimension_numbers failed.");
  }
  xla::ConvolutionDimensionNumbers dnums =
      ConvertConvolutionDimensionNumbers(dnums_input);

  SmallVector<NamedAttribute> converted_attrs;
  for (auto attr : op->getAttrs()) {
    if (attr.getName() == op.getFeatureGroupCountAttrName() ||
        attr.getName() == op.getBatchGroupCountAttrName()) {
      converted_attrs.push_back(attr);
    } else if (attr.getName() == op.getDimensionNumbersAttrName()) {
      attr.setValue(xla::ConvertConvDimensionNumbers(dnums, &rewriter));
      converted_attrs.push_back(attr);
    } else if (attr.getName() == op.getPaddingAttrName()) {
      auto value_or = ConvertPaddingAttr(op, dnums, rewriter);
      if (failed(value_or)) {
        return failure();
      }
      attr.setValue(*value_or);
      converted_attrs.push_back(attr);
    } else if (attr.getName() == op.getWindowStridesAttrName() ||
               attr.getName() == op.getLhsDilationAttrName() ||
               attr.getName() == op.getRhsDilationAttrName()) {
      attr.setValue(ConvertToDenseElementsAttr(
          attr.getValue().template cast<ArrayAttr>(), rewriter));
      converted_attrs.push_back(attr);
    }
  }
  return converted_attrs;
}

// TODO(hinsu): Move this pattern to legalize_tf after resolving the dependency
// on the tensor proto.
class ConvertUniformQuantizedDotHybridOp
    : public OpRewritePattern<TF::UniformQuantizedDotHybridOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::UniformQuantizedDotHybridOp op,
                                PatternRewriter &rewriter) const override {
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

    auto rhs = CreateConstantOp<tensorflow::qint8>(op, op.getRhs(), *rhs_type,
                                                   rewriter);
    if (failed(rhs)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mhlo::DotOp>(op, op.getType(), op.getLhs(),
                                             *rhs,
                                             /*precision_config=*/nullptr);
    return success();
  }
};

class ConvertUniformQuantizedConvolutionHybridOp
    : public OpRewritePattern<TF::UniformQuantizedConvolutionHybridOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::UniformQuantizedConvolutionHybridOp op,
                                PatternRewriter &rewriter) const override {
    // Uniform Quantized type for the rhs.
    auto rhs_type = GetUniformQuantizedType(
        op, op.getRhs().getType(), op.getRhsScales(), op.getRhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), op.getRhsQuantizationAxis(), rewriter);
    if (failed(rhs_type)) {
      return failure();
    }

    auto rhs = CreateConstantOp<tensorflow::qint8>(op, op.getRhs(), *rhs_type,
                                                   rewriter);
    if (failed(rhs)) {
      return failure();
    }

    auto converted_attrs_or = ConvertToMhloConvolutionOpAttrs(op, rewriter);
    if (failed(converted_attrs_or)) {
      return failure();
    }
    SmallVector<Value, 2> operands{op.getLhs(), *rhs};
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

    rewriter.replaceOpWithNewOp<mhlo::UniformQuantizeOp>(op, *output_type,
                                                         op.getInput());
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

    auto output_type = GetUniformQuantizedType(
        op, op.getOutput().getType(), op.getOutputScales(),
        op.getOutputZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(),
        op.getOutputQuantizationMinVal(), op.getOutputQuantizationMaxVal(),
        op.getOutputQuantizationAxis(), rewriter);
    if (failed(output_type)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mhlo::UniformQuantizeOp>(op, *output_type,
                                                         input);
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

    // Uniform Quantized type for the rhs.
    int64_t rhs_quantized_dimension = op.getRhsQuantizationAxis();
    // Currently for dot, PTQ supports per-tensor quantization.
    if (rhs_quantized_dimension != -1) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports only rhs_quantization_axis -1.");
    }
    auto rhs_type = GetUniformQuantizedType(
        op, adaptor.getRhs().getType(), op.getRhsScales(),
        op.getRhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), rhs_quantized_dimension, rewriter);
    if (failed(rhs_type)) {
      return failure();
    }

    auto rhs_or = CreateConstantOp<tensorflow::qint8>(op, op.getRhs(),
                                                      *rhs_type, rewriter);
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

    rewriter.replaceOpWithNewOp<mhlo::DotOp>(op, *output_type, lhs, *rhs_or,
                                             /*precision_config=*/nullptr);
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
      ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();

    auto rhs_type = GetUniformQuantizedType(
        op, adaptor.getRhs().getType(), op.getRhsScales(),
        op.getRhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), op.getRhsQuantizationAxis(), rewriter);
    if (failed(rhs_type)) {
      return failure();
    }

    auto rhs_or = CreateConstantOp<tensorflow::qint8>(op, op.getRhs(),
                                                      *rhs_type, rewriter);
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

    auto converted_attrs_or = ConvertToMhloConvolutionOpAttrs(op, rewriter);
    if (failed(converted_attrs_or)) {
      return failure();
    }
    SmallVector<Value, 2> operands{lhs, *rhs_or};
    rewriter.replaceOpWithNewOp<mhlo::ConvolutionOp>(op, *output_type, operands,
                                                     *converted_attrs_or);
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

    auto lhs_type = lhs.getType().cast<ShapedType>();
    if (!lhs_type.hasRank()) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports cases where only lhs rank known.");
    }
    // rhs (bias) is always 1D that broadcasts to the last dim of lhs.
    auto broadcast_dims =
        mhlo::GetI64ElementsAttr({lhs_type.getRank() - 1}, &rewriter);

    auto rhs_type = GetUniformQuantizedType(
        op, adaptor.getRhs().getType(), op.getRhsScales(),
        op.getRhsZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), op.getRhsQuantizationAxis(), rewriter);
    if (failed(rhs_type)) {
      return failure();
    }

    auto rhs_or = CreateConstantOp<tensorflow::qint32>(op, op.getRhs(),
                                                       *rhs_type, rewriter);
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
    rewriter.replaceOpWithNewOp<chlo::BroadcastAddOp>(op, *output_type, lhs,
                                                      *rhs_or, broadcast_dims);
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
    auto broadcast_dims =
        mhlo::GetI64ElementsAttr(broadcast_dims_values, &rewriter);

    auto min_max_type = GetUniformQuantizedType(
        op, adaptor.getMin().getType(), op.getScales(), op.getZeroPoints(),
        /*expressed_type=*/rewriter.getF32Type(), op.getQuantizationMinVal(),
        op.getQuantizationMaxVal(), op.getQuantizationAxis(), rewriter);
    if (failed(min_max_type)) {
      return failure();
    }
    auto min_or = CreateConstantOp<tensorflow::qint32>(op, op.getMin(),
                                                       *min_max_type, rewriter);
    if (failed(min_or)) {
      return failure();
    }
    auto max_or = CreateConstantOp<tensorflow::qint32>(op, op.getMax(),
                                                       *min_max_type, rewriter);
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

    Value res_min_clipped = rewriter.create<chlo::BroadcastMaxOp>(
        op->getLoc(), *output_type, operand, *min_or, broadcast_dims);
    rewriter.replaceOpWithNewOp<chlo::BroadcastMinOp>(
        op, *output_type, res_min_clipped, *max_or, broadcast_dims);
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
            ConvertUniformQuantizedClipByValueOp>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertTFQuantOpsToMHLOPass() {
  return std::make_unique<ConvertTFQuantOpsToMHLO>();
}

}  // namespace stablehlo
}  // namespace mlir
