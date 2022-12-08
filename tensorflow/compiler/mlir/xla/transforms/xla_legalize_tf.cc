/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/attribute_importer.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_attr.pb.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_params.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_LEGALIZETF
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes.h.inc"

class LegalizeTF : public impl::LegalizeTFBase<LegalizeTF> {
 public:
  explicit LegalizeTF(bool allow_partial_conversion, bool legalize_chlo,
                      llvm::Optional<StringRef> tf2xla_fallback_device_type,
                      bool prefer_tf2xla) {
    allow_partial_conversion_ = allow_partial_conversion;
    legalize_chlo_ = legalize_chlo;
    prefer_tf2xla_ = prefer_tf2xla;
    use_tf2xla_fallback_ = tf2xla_fallback_device_type.has_value();
    if (tf2xla_fallback_device_type.has_value()) {
      device_type_ = tf2xla_fallback_device_type.value().str();
    }
  }
  /// Performs the lowering to XLA dialect.
  void runOnOperation() override;
};

#define GEN_PASS_DEF_LEGALIZETFMODULEPASS
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes.h.inc"

class LegalizeTFModulePass
    : public impl::LegalizeTFModulePassBase<LegalizeTFModulePass> {
 public:
  explicit LegalizeTFModulePass(StringRef tf2xla_fallback_device_type) {
    device_type_ = tf2xla_fallback_device_type.str();
  }

  /// Performs the lowering to XLA dialect.
  void runOnOperation() override;
};

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
    UniformQuantizedOp op, TensorType operand_type,
    TypedValue<TensorType> scales_value,
    TypedValue<TensorType> zero_points_value, IntegerType storage_type,
    FloatType expressed_type, int64_t storage_type_min,
    int64_t storage_type_max, int32_t quantized_dimension,
    PatternRewriter &rewriter) {
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
  const unsigned flags = quant::QuantizationFlags::Signed;
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

  return GetSameShapeTensorType(operand_type, elem_ty);
}

template <typename UniformQuantizedOp>
FailureOr<mhlo::ConstantOp> CreateConstantOpForQint8Rhs(
    UniformQuantizedOp op, TensorType new_rhs_type, PatternRewriter &rewriter) {
  // Check whether the rhs operand has constant op.
  TF::TensorProtoAttr tensor_proto_attr;
  if (!matchPattern(op.getRhs(), m_Constant(&tensor_proto_attr))) {
    return rewriter.notifyMatchFailure(op, "rhs must be constant.");
  }

  llvm::StringRef mangled_tensor = tensor_proto_attr.getValue();
  absl::string_view tensor_view(mangled_tensor.data(), mangled_tensor.size());
  // TODO(hinsu): Instead of getting the weight from TensorProto, use MLIR
  // constant attribute to avoid depending on the Tensor proto.
  tensorflow::TensorProto tensor_proto;
  tensorflow::Status status =
      tensorflow::mangling_util::DemangleTensor(tensor_view, &tensor_proto);
  if (!status.ok()) {
    return rewriter.notifyMatchFailure(op, status.error_message());
  }

  tensorflow::Tensor t;
  if (!t.FromProto(tensor_proto)) {
    return op.emitError("Failed to convert tensor proto to Tensor.");
  }

  auto arr = t.flat<tensorflow::qint8>();
  auto dense_attr = mlir::DenseElementsAttr::get(
      GetSameShapeTensorType(new_rhs_type, rewriter.getIntegerType(8)),
      llvm::makeArrayRef(arr.data(), arr.size()));
  return rewriter.create<mhlo::ConstantOp>(op.getLoc(), new_rhs_type,
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

FailureOr<ElementsAttr> ConvertPaddingAttr(
    TF::UniformQuantizedConvolutionHybridOp op,
    const xla::ConvolutionDimensionNumbers &dnums, PatternRewriter &rewriter) {
  StringAttr conv_padding = op.getPaddingAttr();
  SmallVector<int64_t> padding_nums;
  ShapedType lhs_shape = op.getLhs().getType().cast<ShapedType>();
  ShapedType rhs_shape = op.getRhs().getType().cast<ShapedType>();

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
         op.getExplicitPaddingAttr().getAsRange<IntegerAttr>()) {
      padding_nums.push_back(padding_elem.getInt());
    }
  } else if (conv_padding.strref().equals("VALID")) {
    padding_nums.resize(padding_nums_size, 0);
  } else {
    padding_nums.resize(padding_nums_size);
    for (int i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
      const int64_t stride =
          op.getWindowStridesAttr()[i].cast<IntegerAttr>().getInt();
      const int64_t lhs_size_dilated =
          tensorflow::UniformQuantizedConvolutionParams::DilatedSize(
              lhs_shape.getDimSize(dnums.input_spatial_dimensions(i)),
              op.getLhsDilationAttr()[i].cast<IntegerAttr>().getInt());
      const int64_t rhs_size_dilated =
          tensorflow::UniformQuantizedConvolutionParams::DilatedSize(
              rhs_shape.getDimSize(dnums.kernel_spatial_dimensions(i)),
              op.getRhsDilationAttr()[i].cast<IntegerAttr>().getInt());

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

// TODO(hinsu): Move this pattern to legalize_tf after resolving the dependency
// on the tensor proto.
class ConvertUniformQuantizedDotHybridOp
    : public OpRewritePattern<TF::UniformQuantizedDotHybridOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::UniformQuantizedDotHybridOp op,
                                PatternRewriter &rewriter) const override {
    // f32 x qi8 -> f32
    IntegerType rhs_storage_type = rewriter.getIntegerType(8);
    // Uniform Quantized type for the rhs.
    int32_t rhs_quantized_dimension = op.getRhsQuantizationAxis();
    // Currently for dot, PTQ supports per-tensor quantization.
    if (rhs_quantized_dimension != -1) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports only rhs_quantization_axis -1.");
    }
    auto rhs_type = GetUniformQuantizedType(
        op, op.getRhs().getType().cast<TensorType>(), op.getRhsScales(),
        op.getRhsZeroPoints(), rhs_storage_type,
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), rhs_quantized_dimension, rewriter);
    if (failed(rhs_type)) {
      return rhs_type;
    }
    auto rhs = CreateConstantOpForQint8Rhs(op, *rhs_type, rewriter);
    if (failed(rhs)) {
      return rhs;
    }
    rewriter.replaceOpWithNewOp<mhlo::DotOp>(op, op.getLhs(), *rhs,
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
    IntegerType rhs_storage_type = rewriter.getIntegerType(8);

    // Uniform Quantized type for the rhs.
    auto rhs_type = GetUniformQuantizedType(
        op, op.getRhs().getType().cast<TensorType>(), op.getRhsScales(),
        op.getRhsZeroPoints(), rhs_storage_type,
        /*expressed_type=*/rewriter.getF32Type(), op.getRhsQuantizationMinVal(),
        op.getRhsQuantizationMaxVal(), op.getRhsQuantizationAxis(), rewriter);
    if (failed(rhs_type)) {
      return rhs_type;
    }
    auto rhs = CreateConstantOpForQint8Rhs(op, *rhs_type, rewriter);
    if (failed(rhs)) {
      return rhs;
    }

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
          return value_or;
        }
        attr.setValue(*value_or);
        converted_attrs.push_back(attr);
      } else if (attr.getName() == op.getWindowStridesAttrName() ||
                 attr.getName() == op.getLhsDilationAttrName() ||
                 attr.getName() == op.getRhsDilationAttrName()) {
        attr.setValue(ConvertToDenseElementsAttr(
            attr.getValue().cast<ArrayAttr>(), rewriter));
        converted_attrs.push_back(attr);
      }
    }

    SmallVector<Value, 2> operands{op.getLhs(), *rhs};
    rewriter.replaceOpWithNewOp<mhlo::ConvolutionOp>(op, op.getType(), operands,
                                                     converted_attrs);
    return success();
  }
};

// Emits debug information which includes the number of ops of each type which
// failed to legalize.
void EmitLegalizationErrors(Operation *op,
                            const DenseSet<Operation *> &nonlegalized_ops) {
  // Track the legalization failures by mapping op name to information about
  // that failure: the number of unlegalized occurrences of the op, and one
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

/// Returns ops that should use MLIR legalization only in the case of
/// prefer_tf2xla. All other ops not in this list should use XlaOpKernel
/// legalization only or not be legalized by the new bridge.
const llvm::DenseSet<mlir::TypeID> &MlirPreferredOps() {
  // The static variable is a pointer in order to avoid destruction upon thread
  // termination.

  // clang-format off
  static const llvm::DenseSet<mlir::TypeID>* ops =
      new llvm::DenseSet<mlir::TypeID>{
    // Ops that are legalized in the old bridge using MlirXlaOpKernel
    TypeID::get<TF::AbsOp>(),
    TypeID::get<TF::AtanOp>(),
    TypeID::get<TF::AvgPool3DOp>(),
    TypeID::get<TF::BiasAddGradOp>(),
    TypeID::get<TF::CeilOp>(),
    TypeID::get<TF::CheckNumericsOp>(),
    TypeID::get<TF::ComplexOp>(),
    TypeID::get<TF::CosOp>(),
    TypeID::get<TF::DiagPartOp>(),
    TypeID::get<TF::DivOp>(),
    TypeID::get<TF::EinsumOp>(),
    TypeID::get<TF::ExpOp>(),
    TypeID::get<TF::Expm1Op>(),
    TypeID::get<TF::FakeQuantWithMinMaxArgsOp>(),
    TypeID::get<TF::FloorOp>(),
    TypeID::get<TF::GreaterEqualOp>(),
    TypeID::get<TF::IFFTOp>(),
    TypeID::get<TF::ImagOp>(),
    TypeID::get<TF::IsFiniteOp>(),
    TypeID::get<TF::IsInfOp>(),
    TypeID::get<TF::IsNanOp>(),
    TypeID::get<TF::LessEqualOp>(),
    TypeID::get<TF::LgammaOp>(),
    TypeID::get<TF::Log1pOp>(),
    TypeID::get<TF::LogicalOrOp>(),
    TypeID::get<TF::LogSoftmaxOp>(),
    TypeID::get<TF::MatrixBandPartOp>(),
    TypeID::get<TF::MaxPool3DGradOp>(),
    TypeID::get<TF::PreventGradientOp>(),
    TypeID::get<TF::RandomShuffleOp>(),
    TypeID::get<TF::RealOp>(),
    TypeID::get<TF::ReciprocalOp>(),
    TypeID::get<TF::ReluOp>(),
    TypeID::get<TF::Relu6Op>(),
    TypeID::get<TF::ReluGradOp>(),
    TypeID::get<TF::RsqrtOp>(),
    TypeID::get<TF::SelectOp>(),
    TypeID::get<TF::SigmoidOp>(),
    TypeID::get<TF::SignOp>(),
    TypeID::get<TF::SoftmaxOp>(),
    TypeID::get<TF::SqrtOp>(),
    TypeID::get<TF::SqrtGradOp>(),
    TypeID::get<TF::SquaredDifferenceOp>(),
    TypeID::get<TF::TanhOp>(),
    TypeID::get<TF::TanhGradOp>(),
    TypeID::get<TF::XlaConvV2Op>(),
    TypeID::get<TF::XlaDotOp>(),
    TypeID::get<TF::XlaDotV2Op>(),
    TypeID::get<TF::XlaDynamicSliceOp>(),
    TypeID::get<TF::XlaEinsumOp>(),
    TypeID::get<TF::XlaReduceWindowOp>(),
    TypeID::get<TF::XlaReplicaIdOp>(),
    TypeID::get<TF::XlaRngBitGeneratorOp>(),
    TypeID::get<TF::XlaSelectAndScatterOp>(),
    TypeID::get<TF::XlaSortOp>(),
    TypeID::get<TF::XlaVariadicReduceV2Op>(),
    TypeID::get<TF::XlaVariadicSortOp>(),
    TypeID::get<TF::XlogyOp>(),
    TypeID::get<TF::ZetaOp>(),

    // Ops that have no XlaOpKernel.
    TypeID::get<TF::RiscAddOp>(),
    TypeID::get<TF::RiscDotOp>(),

    // Const op has a simple legalization and it is much more efficient to lower
    // within MLIR.
    TypeID::get<TF::ConstOp>(),

    // AssertOp with string types are not supported by the fallback.
    TypeID::get<TF::AssertOp>(),

    // TF2XLA fallback pattern doesn't support these op as MLIR hlo builder
    // doesn't override the necessary builder methods. These ops have simple
    // lowering pattern so this should be safe.
    TypeID::get<TF::CrossReplicaSumOp>(),
    TypeID::get<TF::InfeedDequeueTupleOp>(),
    TypeID::get<TF::OutfeedEnqueueTupleOp>(),
    TypeID::get<TF::XlaShardingOp>(),

    // These ops have undetermined bugs, may not be legalizable with XlaOpKernel
    // legalization in TF2XLA fallback. By legalization with MLIR, we can fix
    // the bug. b/195583695 describes the motivation of this change.
    // See b/216355804 how to reproduce the bug regarding tf.RandomUniform Op
    // See b/216353817 how to reproduce the bug regarding tf.StridedSlice Op
    // See b/245615401 how to reproduce the bug regarding tf.SliceOp
    TypeID::get<TF::RandomUniformOp>(),
    TypeID::get<TF::StridedSliceOp>(),
    TypeID::get<TF::SliceOp>(),
  };
  // clang-format on
  return *ops;
}

// Patterns whose root op is in the set `include_ops` are moved from the set
// `from` to the returned set. This is used to partition patterns by op so they
// can be cleanly migrated from the old bridge to the MLIR bridge.
RewritePatternSet PatternsIncludeOps(
    RewritePatternSet &from, const llvm::DenseSet<mlir::TypeID> &include_ops) {
  RewritePatternSet to(from.getContext());
  // Filter NativePatterns.
  for (auto &pattern : from.getNativePatterns()) {
    Optional<OperationName> pat_op_name = pattern->getRootKind();
    // If the pattern does not have a specific operation, always include it,
    // If the pattern is in include_ops then include it.
    bool include =
        !pat_op_name ||
        include_ops.count(pat_op_name->getRegisteredInfo()->getTypeID());
    if (include) to.add(std::move(pattern));
  }

  // Don't filter PDLPatterns.
  to.add(std::move(from.getPDLPatterns()));

  return to;
}

mlir::LogicalResult ApplyPatterns(Operation *op, RewritePatternSet &patterns,
                                  bool legalize_chlo,
                                  bool allow_partial_conversion) {
  ConversionTarget target(*op->getContext());
  if (legalize_chlo) {
    target.addIllegalDialect<chlo::ChloDialect>();
  } else {
    target.addLegalDialect<chlo::ChloDialect>();
  }
  target.addLegalDialect<MhloDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<shape::ShapeDialect>();
  target.addLegalOp<func::CallOp>();

  if (!allow_partial_conversion) {
    // Fully qualify ReturnOp here as mhlo dialect also defines a ReturnOp.
    target.addLegalOp<ModuleOp, ::mlir::func::FuncOp, ::mlir::func::ReturnOp>();
    DenseSet<Operation *> nonlegalized_ops;
    LogicalResult result = applyPartialConversion(
        op, target, std::move(patterns), &nonlegalized_ops);
    // In order to enforce that the conversion result is fully converted,
    // fail if there are any nonlegalized ops in the set.
    if (failed(result) || !nonlegalized_ops.empty()) {
      EmitLegalizationErrors(op, nonlegalized_ops);
      return failure();
    }
    return result;
  }

  return applyPartialConversion(op, target, std::move(patterns));
}

/// When `tf2xla_fallback_device_type` is not `None`, also uses legalization
/// patterns from TF2XLA fallback for provided device type (see
/// legalize_tf_with_tf2xla.cc for details). By default, TF2XLA fallback is not
/// used.
LogicalResult legalizeTF(Operation *op, bool allow_partial_conversion,
                         bool legalize_chlo,
                         llvm::Optional<StringRef> tf2xla_fallback_device_type,
                         bool prefer_tf2xla) {
  MLIRContext *context = op->getContext();
  RewritePatternSet legalize_lower_patterns(context);
  // Note that the `OperationConverter` orders patterns lexicographically by:
  // 1) Ascending legalization depth (i.e., minimum number of patterns necessary
  //    to arrive at conversion target). This requires relevant patterns to
  //    specify the list of ops generated by it which most of patterns
  //    implemented in C++ don't do so this comparison doesn't work in those
  //    cases.
  // 2) Descending pattern benefit.
  // 3) Op specific patterns over patterns with MatchAnyOpTypeTag.
  // 4) Order of patterns in `RewritePatternSet`.

  // Add TF->HLO legalization patterns.
  PopulateLegalizeTfPatterns(context, &legalize_lower_patterns);
  PopulateLegalizeTfQuantizationPatterns(context, &legalize_lower_patterns);

  // Add TF->TF lowering patterns.
  TF::PopulateTFLoweringBeforeHLOPatterns(context, &legalize_lower_patterns);

  if (tf2xla_fallback_device_type && prefer_tf2xla) {
    VLOG(1) << "TF to XLA legalization patterns are partitioned by op into "
               "either native MLIR legalization, or TF2XLA fallback "
               "legalzation, with a preference toward TF2XLA.";
  } else if (tf2xla_fallback_device_type) {
    VLOG(1) << "TF to XLA legalization patterns include all native patterns "
               "and TF2XLA fallback patterns.";
  } else {
    VLOG(1) << "TF to XLA legalization patterns are native patterns only.";
  }

  // Set patterns to legalize_lower_patters, where in the prefer_tf2xla case
  // only patterns whose ops are in the set MlirPreferredOps are kept.
  RewritePatternSet patterns =
      (tf2xla_fallback_device_type && prefer_tf2xla)
          ? PatternsIncludeOps(legalize_lower_patterns, MlirPreferredOps())
          : std::move(legalize_lower_patterns);

  if (tf2xla_fallback_device_type) {
    // Add TF->HLO legalization patterns via TF2XLA fallback.
    PopulateLegalizeTfWithTf2XlaPatterns(tf2xla_fallback_device_type.value(),
                                         patterns, context, prefer_tf2xla);
  }

  // Populate with CHLO->HLO lowerings to account for TF ops legalized to
  // CHLO first.
  if (legalize_chlo) {
    chlo::populateDecomposeChloPatterns(context, &patterns);
    chlo::populateChloBroadcastingPatterns(context, &patterns);
  }
  // ConstantLike op is convenient to create splat constants, but is
  // canonicalized to plain HLO constant if statically shaped. Add the
  // canonicalization pattern to pattern list to enable multi-hop lowering.
  chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

  return ApplyPatterns(op, patterns, legalize_chlo, allow_partial_conversion);
}

// Performs the lowering to XLA dialect.
void LegalizeTF::runOnOperation() {
  llvm::Optional<StringRef> tf2xla_fallback_device_type = llvm::None;
  if (use_tf2xla_fallback_) {
    tf2xla_fallback_device_type = device_type_;
  }
  if (failed(legalizeTF(getOperation(), allow_partial_conversion_,
                        legalize_chlo_, tf2xla_fallback_device_type,
                        prefer_tf2xla_))) {
    signalPassFailure();
  }
}

void LegalizeTFModulePass::runOnOperation() {
  // This pass should only be run when a fallback device is present.
  if (!device_type_.hasValue()) {
    return;
  }
  VLOG(1) << "TF to XLA legalization patterns include TF2XLA fallback "
             "patterns for Ops that need to create functions.";
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);
  PopulateLegalizeTfWithTf2XlaPatterns(device_type_, patterns, context,
                                       /*prefer_tf2xla=*/false,
                                       /*is_module_pass=*/true);

  if (failed(ApplyPatterns(op, patterns,
                           /*legalize_chlo=*/false,
                           /*allow_partial_conversion=*/true))) {
    signalPassFailure();
  }
}

}  // end namespace

void PopulateLegalizeTfQuantizationPatterns(MLIRContext *context,
                                            RewritePatternSet *patterns) {
  patterns->add<ConvertUniformQuantizedDotHybridOp,
                ConvertUniformQuantizedConvolutionHybridOp>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFPass(
    bool allow_partial_conversion, bool legalize_chlo,
    llvm::Optional<StringRef> tf2xla_fallback_device_type, bool prefer_tf2xla) {
  return std::make_unique<LegalizeTF>(allow_partial_conversion, legalize_chlo,
                                      tf2xla_fallback_device_type,
                                      prefer_tf2xla);
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeTFModulePass(
    StringRef tf2xla_fallback_device_type) {
  return std::make_unique<LegalizeTFModulePass>(tf2xla_fallback_device_type);
}

}  // end namespace mhlo
}  // end namespace mlir
