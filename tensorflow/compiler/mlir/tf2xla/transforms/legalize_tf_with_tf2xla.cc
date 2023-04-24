/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tpu_embedding_ops_registry.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/tf2xla_rewriter.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/mlir_hlo_builder.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace mlir {
namespace mhlo {

// LINT.IfChange
bool IsOpAllowedTf2XlaFallback(Operation* op) {
  // Allowlisted TensorFlow ops are known to have well behaved tf2xla kernels
  // building valid MLIR using MlirHloBuilder.
  // TODO(hinsu): Drop explicit allowlist when MLIR based bridge is enabled for
  // all tf2xla kernels.
  // Use a pointer for the static set, so the set is not destructed upon thread
  // end, which would not be thread safe.

  static auto* ops = [] {
    llvm::SmallDenseSet<mlir::TypeID, 512>* ops_set =
        new llvm::SmallDenseSet<mlir::TypeID, 512>{
            TypeID::get<TF::AcoshOp>(),
            TypeID::get<TF::AcosOp>(),
            TypeID::get<TF::AddNOp>(),
            TypeID::get<TF::AddV2Op>(),
            TypeID::get<TF::AngleOp>(),
            TypeID::get<TF::AdjustContrastv2Op>(),
            TypeID::get<TF::AdjustHueOp>(),
            TypeID::get<TF::AdjustSaturationOp>(),
            TypeID::get<TF::ApproximateEqualOp>(),
            TypeID::get<TF::ArgMaxOp>(),
            TypeID::get<TF::ArgMinOp>(),
            TypeID::get<TF::AsinhOp>(),
            TypeID::get<TF::AsinOp>(),
            TypeID::get<TF::Atan2Op>(),
            TypeID::get<TF::AtanhOp>(),
            TypeID::get<TF::BatchMatMulV2Op>(),
            TypeID::get<TF::BatchMatMulV3Op>(),
            TypeID::get<TF::BatchToSpaceOp>(),
            TypeID::get<TF::BesselI0eOp>(),
            TypeID::get<TF::BesselI1eOp>(),
            TypeID::get<TF::BetaincOp>(),
            TypeID::get<TF::BiasAddOp>(),
            TypeID::get<TF::BitwiseAndOp>(),
            TypeID::get<TF::BitwiseOrOp>(),
            TypeID::get<TF::BitwiseXorOp>(),
            TypeID::get<TF::BucketizeOp>(),
            // CaseOp isn't actually supported but is enabled for testing to
            // make sure ops with symbol ref attributes are filtered out.
            TypeID::get<TF::CaseOp>(),
            TypeID::get<TF::CastOp>(),
            TypeID::get<TF::ClipByValueOp>(),
            TypeID::get<TF::CholeskyOp>(),
            TypeID::get<TF::ComplexAbsOp>(),
            TypeID::get<TF::ConjugateTransposeOp>(),
            TypeID::get<TF::CoshOp>(),
            TypeID::get<TF::CrossOp>(),
            TypeID::get<TF::CumulativeLogsumexpOp>(),
            TypeID::get<TF::DataFormatDimMapOp>(),
            TypeID::get<TF::DataFormatVecPermuteOp>(),
            TypeID::get<TF::DepthToSpaceOp>(),
            TypeID::get<TF::DepthwiseConv2dNativeBackpropFilterOp>(),
            TypeID::get<TF::DepthwiseConv2dNativeBackpropInputOp>(),
            TypeID::get<TF::DiagOp>(),
            TypeID::get<TF::DigammaOp>(),
            TypeID::get<TF::DivNoNanOp>(),
            TypeID::get<TF::DynamicPartitionOp>(),
            TypeID::get<TF::EluGradOp>(),
            TypeID::get<TF::EluOp>(),
            TypeID::get<TF::EnsureShapeOp>(),
            TypeID::get<TF::EqualOp>(),
            TypeID::get<TF::ErfcOp>(),
            TypeID::get<TF::ErfinvOp>(),
            TypeID::get<TF::ErfOp>(),
            TypeID::get<TF::ExtractImagePatchesOp>(),
            TypeID::get<TF::FFT2DOp>(),
            TypeID::get<TF::FFT3DOp>(),
            TypeID::get<TF::FFTOp>(),
            TypeID::get<TF::FakeParamOp>(),
            TypeID::get<TF::FakeQuantWithMinMaxArgsGradientOp>(),
            TypeID::get<TF::FakeQuantWithMinMaxVarsGradientOp>(),
            TypeID::get<TF::FakeQuantWithMinMaxVarsPerChannelOp>(),
            TypeID::get<TF::FakeQuantWithMinMaxVarsPerChannelGradientOp>(),
            TypeID::get<TF::FloorDivOp>(),
            TypeID::get<TF::FloorModOp>(),
            TypeID::get<TF::GreaterOp>(),
            TypeID::get<TF::HSVToRGBOp>(),
            TypeID::get<TF::IFFT2DOp>(),
            TypeID::get<TF::IFFT3DOp>(),
            TypeID::get<TF::IRFFT2DOp>(),
            TypeID::get<TF::IRFFT3DOp>(),
            TypeID::get<TF::IgammaOp>(),
            TypeID::get<TF::IgammacOp>(),
            TypeID::get<TF::IgammaGradAOp>(),
            TypeID::get<TF::InplaceAddOp>(),
            TypeID::get<TF::InTopKV2Op>(),
            TypeID::get<TF::InvertOp>(),
            TypeID::get<TF::InvOp>(),
            TypeID::get<TF::KthOrderStatisticOp>(),
            TypeID::get<TF::LRNOp>(),
            TypeID::get<TF::LRNGradOp>(),
            TypeID::get<TF::LeakyReluGradOp>(),
            TypeID::get<TF::LeakyReluOp>(),
            TypeID::get<TF::LeftShiftOp>(),
            TypeID::get<TF::LessOp>(),
            TypeID::get<TF::ListDiffOp>(),
            TypeID::get<TF::LogicalAndOp>(),
            TypeID::get<TF::LogicalNotOp>(),
            TypeID::get<TF::LogOp>(),
            TypeID::get<TF::LowerBoundOp>(),
            TypeID::get<TF::MakeUniqueOp>(),
            TypeID::get<TF::MatMulOp>(),
            TypeID::get<TF::MatrixDiagV3Op>(),
            TypeID::get<TF::MatrixInverseOp>(),
            TypeID::get<TF::MatrixSetDiagV3Op>(),
            TypeID::get<TF::MatrixSolveOp>(),
            TypeID::get<TF::MatrixTriangularSolveOp>(),
            TypeID::get<TF::MaxPool3DGradGradOp>(),
            TypeID::get<TF::MaxPoolGradGradOp>(),
            TypeID::get<TF::MirrorPadOp>(),
            TypeID::get<TF::MirrorPadGradOp>(),
            TypeID::get<TF::MulOp>(),
            TypeID::get<TF::MultinomialOp>(),
            TypeID::get<TF::NdtriOp>(),
            TypeID::get<TF::NegOp>(),
            TypeID::get<TF::NextAfterOp>(),
            TypeID::get<TF::NonMaxSuppressionV4Op>(),
            TypeID::get<TF::NotEqualOp>(),
            TypeID::get<TF::PadOp>(),
            TypeID::get<TF::ParameterizedTruncatedNormalOp>(),
            TypeID::get<TF::PlaceholderWithDefaultOp>(),
            TypeID::get<TF::PolygammaOp>(),
            TypeID::get<TF::PopulationCountOp>(),
            TypeID::get<TF::PowOp>(),
            TypeID::get<TF::QrOp>(),
            // TODO(hinsu): Canonicalize QuantizeAndDequantize and
            // QuantizeAndDequantizeV2 to QuantizeAndDequantizeV3 by converting
            // attributes to operands.
            TypeID::get<TF::QuantizeAndDequantizeOp>(),
            TypeID::get<TF::QuantizeAndDequantizeV2Op>(),
            TypeID::get<TF::QuantizeAndDequantizeV3Op>(),
            TypeID::get<TF::QuantizeAndDequantizeV4Op>(),
            TypeID::get<TF::RFFT2DOp>(),
            TypeID::get<TF::RFFT3DOp>(),
            TypeID::get<TF::RGBToHSVOp>(),
            TypeID::get<TF::RandomUniformIntOp>(),
            TypeID::get<TF::RealDivOp>(),
            TypeID::get<TF::ReciprocalGradOp>(),
            TypeID::get<TF::Relu6GradOp>(),
            TypeID::get<TF::ResizeBilinearOp>(),
            TypeID::get<TF::ResizeBilinearGradOp>(),
            TypeID::get<TF::ResizeNearestNeighborOp>(),
            TypeID::get<TF::ResizeNearestNeighborGradOp>(),
            TypeID::get<TF::ReverseSequenceOp>(),
            TypeID::get<TF::RightShiftOp>(),
            TypeID::get<TF::RintOp>(),
            TypeID::get<TF::RollOp>(),
            TypeID::get<TF::RoundOp>(),
            TypeID::get<TF::SegmentSumV2Op>(),
            TypeID::get<TF::SegmentProdV2Op>(),
            TypeID::get<TF::SegmentMinV2Op>(),
            TypeID::get<TF::SegmentMaxV2Op>(),
            TypeID::get<TF::SelectV2Op>(),
            TypeID::get<TF::SelfAdjointEigV2Op>(),
            TypeID::get<TF::SeluGradOp>(),
            TypeID::get<TF::SeluOp>(),
            TypeID::get<TF::SigmoidGradOp>(),
            TypeID::get<TF::SinOp>(),
            TypeID::get<TF::SoftplusGradOp>(),
            TypeID::get<TF::SoftsignGradOp>(),
            TypeID::get<TF::SoftsignOp>(),
            TypeID::get<TF::SpaceToBatchNDOp>(),
            TypeID::get<TF::SpaceToBatchOp>(),
            TypeID::get<TF::SpaceToDepthOp>(),
            TypeID::get<TF::SparseToDenseOp>(),
            TypeID::get<TF::SquareOp>(),
            TypeID::get<TF::StatelessMultinomialOp>(),
            TypeID::get<TF::StatelessParameterizedTruncatedNormalOp>(),
            TypeID::get<TF::StatelessRandomGetAlgOp>(),
            TypeID::get<TF::StatelessRandomGetKeyCounterOp>(),
            TypeID::get<TF::StatelessRandomGetKeyCounterAlgOp>(),
            TypeID::get<TF::StatelessRandomNormalOp>(),
            TypeID::get<TF::StatelessRandomNormalV2Op>(),
            TypeID::get<TF::StatelessRandomUniformOp>(),
            TypeID::get<TF::StatelessRandomUniformFullIntOp>(),
            TypeID::get<TF::StatelessRandomUniformFullIntV2Op>(),
            TypeID::get<TF::StatelessRandomUniformV2Op>(),
            TypeID::get<TF::StatelessRandomUniformIntOp>(),
            TypeID::get<TF::StatelessRandomUniformIntV2Op>(),
            TypeID::get<TF::StatelessTruncatedNormalOp>(),
            TypeID::get<TF::StatelessTruncatedNormalV2Op>(),
            TypeID::get<TF::SubOp>(),
            TypeID::get<TF::SvdOp>(),
            TypeID::get<TF::TanOp>(),
            TypeID::get<TF::TensorScatterAddOp>(),
            TypeID::get<TF::TensorScatterSubOp>(),
            TypeID::get<TF::TPUEmbeddingActivationsOp>(),
            TypeID::get<TF::TopKUniqueOp>(),
            TypeID::get<TF::TopKWithUniqueOp>(),
            TypeID::get<TF::TransposeOp>(),
            TypeID::get<TF::TridiagonalSolveOp>(),
            TypeID::get<TF::TridiagonalMatMulOp>(),
            TypeID::get<TF::TruncateDivOp>(),
            TypeID::get<TF::TruncatedNormalOp>(),
            TypeID::get<TF::TruncateModOp>(),
            TypeID::get<TF::UniqueOp>(),
            TypeID::get<TF::UnpackOp>(),
            TypeID::get<TF::UpperBoundOp>(),
            TypeID::get<TF::WhereOp>(),
            TypeID::get<TF::XlaBroadcastHelperOp>(),
            TypeID::get<TF::XlaCustomCallV2Op>(),
            TypeID::get<TF::XlaDynamicUpdateSliceOp>(),
            TypeID::get<TF::XlaKeyValueSortOp>(),
            TypeID::get<TF::XlaPadOp>(),
            TypeID::get<TF::XlaSetBoundOp>(),
            TypeID::get<TF::XlaSetDynamicDimensionSizeOp>(),
            TypeID::get<TF::XlaSpmdFullToShardShapeOp>(),
            TypeID::get<TF::XlaSpmdShardToFullShapeOp>(),
            TypeID::get<TF::XlaSvdOp>(),
        };

    // Add the ops from the TPUEmbeddingOpsRegistry.
    for (auto op_type_id :
         TF::TPUEmbeddingOpsRegistry::Global().GetOpsTypeIds()) {
      ops_set->insert(op_type_id);
    }
    return ops_set;
  }();

  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return false;
  return ops->count(abstractOp->getTypeID());
}
// LINT.ThenChange(:Tf2XlaPreferred)

/// List of ops that should use XlaOpKernel legalization only in the case of
/// prefer_tf2xla. All other ops not in this list should use MLIR legalization
/// only or not be legalized by the new bridge.
// LINT.IfChange(Tf2XlaPreferred)
bool IsOpAllowedTf2XlaPreferred(Operation* op) {
  // Use a pointer for the static set, so the set is not destructed upon thread
  // end, which would not be thread safe.
  // clang-format off
  static auto* ops =
      new llvm::SmallDenseSet<mlir::TypeID, 512>{
    TypeID::get<TF::AllOp>(),
    TypeID::get<TF::AllToAllOp>(),
    TypeID::get<TF::AnyOp>(),
    TypeID::get<TF::AvgPoolOp>(),
    TypeID::get<TF::AvgPool3DGradOp>(),
    TypeID::get<TF::AvgPoolGradOp>(),
    TypeID::get<TF::BatchToSpaceNDOp>(),
    TypeID::get<TF::BitcastOp>(),
    TypeID::get<TF::BroadcastToOp>(),
    TypeID::get<TF::CollectivePermuteOp>(),
    TypeID::get<TF::ComplexOp>(),
    TypeID::get<TF::ConcatV2Op>(),
    TypeID::get<TF::ConjOp>(),
    TypeID::get<TF::Conv2DOp>(),
    TypeID::get<TF::Conv2DBackpropFilterOp>(),
    TypeID::get<TF::Conv2DBackpropInputOp>(),
    TypeID::get<TF::Conv3DOp>(),
    TypeID::get<TF::Conv3DBackpropFilterV2Op>(),
    TypeID::get<TF::Conv3DBackpropInputV2Op>(),
    TypeID::get<TF::CumprodOp>(),
    TypeID::get<TF::CumsumOp>(),
    TypeID::get<TF::DepthwiseConv2dNativeOp>(),
    TypeID::get<TF::DivOp>(),
    TypeID::get<TF::DynamicStitchOp>(),
    TypeID::get<TF::_EagerConstOp>(),
    TypeID::get<TF::EmptyOp>(),
    TypeID::get<TF::ExpandDimsOp>(),
    TypeID::get<TF::FakeQuantWithMinMaxVarsOp>(),
    TypeID::get<TF::FillOp>(),
    TypeID::get<TF::FusedBatchNormOp>(),
    TypeID::get<TF::FusedBatchNormGradOp>(),
    TypeID::get<TF::FusedBatchNormGradV2Op>(),
    TypeID::get<TF::FusedBatchNormV2Op>(),
    TypeID::get<TF::_FusedConv2DOp>(),
    TypeID::get<TF::GatherNdOp>(),
    TypeID::get<TF::GatherV2Op>(),
    TypeID::get<TF::GreaterEqualOp>(),
    TypeID::get<TF::IdentityOp>(),
    TypeID::get<TF::IdentityNOp>(),
    TypeID::get<TF::InplaceUpdateOp>(),
    TypeID::get<TF::InvertPermutationOp>(),
    TypeID::get<TF::IRFFTOp>(),
    TypeID::get<TF::L2LossOp>(),
    TypeID::get<TF::LegacyCallOp>(),
    TypeID::get<TF::LessEqualOp>(),
    TypeID::get<TF::LinSpaceOp>(),
    TypeID::get<TF::LogicalOrOp>(),
    TypeID::get<TF::MatrixDiagPartV3Op>(),
    TypeID::get<TF::MaxOp>(),
    TypeID::get<TF::MaximumOp>(),
    TypeID::get<TF::MaxPoolOp>(),
    TypeID::get<TF::MaxPool3DOp>(),
    TypeID::get<TF::MeanOp>(),
    TypeID::get<TF::MinOp>(),
    TypeID::get<TF::MinimumOp>(),
    TypeID::get<TF::MulNoNanOp>(),
    TypeID::get<TF::OneHotOp>(),
    TypeID::get<TF::OnesLikeOp>(),
    TypeID::get<TF::PackOp>(),
    TypeID::get<TF::PadV2Op>(),
    TypeID::get<TF::ParallelDynamicStitchOp>(),
    TypeID::get<TF::PartitionedCallOp>(),
    TypeID::get<TF::ProdOp>(),
    TypeID::get<TF::QrOp>(),
    TypeID::get<TF::RandomStandardNormalOp>(),
    TypeID::get<TF::RangeOp>(),
    TypeID::get<TF::ReshapeOp>(),
    TypeID::get<TF::ReverseV2Op>(),
    TypeID::get<TF::RFFTOp>(),
    TypeID::get<TF::RsqrtGradOp>(),
    TypeID::get<TF::ScatterNdOp>(),
    TypeID::get<TF::ShapeOp>(),
    TypeID::get<TF::SinhOp>(),
    TypeID::get<TF::SizeOp>(),
    TypeID::get<TF::SliceOp>(),
    TypeID::get<TF::SoftmaxCrossEntropyWithLogitsOp>(),
    TypeID::get<TF::SoftplusOp>(),
    TypeID::get<TF::SparseMatMulOp>(),
    TypeID::get<TF::SparseSoftmaxCrossEntropyWithLogitsOp>(),
    TypeID::get<TF::SplitOp>(),
    TypeID::get<TF::SplitVOp>(),
    TypeID::get<TF::SqrtGradOp>(),
    TypeID::get<TF::SquaredDifferenceOp>(),
    TypeID::get<TF::SqueezeOp>(),
    TypeID::get<TF::StatelessParameterizedTruncatedNormalOp>(),
    TypeID::get<TF::StatefulPartitionedCallOp>(),
    TypeID::get<TF::StopGradientOp>(),
    TypeID::get<TF::StridedSliceGradOp>(),
    TypeID::get<TF::SumOp>(),
    TypeID::get<TF::TanhGradOp>(),
    TypeID::get<TF::TensorScatterUpdateOp>(),
    TypeID::get<TF::TileOp>(),
    TypeID::get<TF::TopKV2Op>(),
    TypeID::get<TF::_UnaryOpsCompositionOp>(),
    TypeID::get<TF::UnsortedSegmentMaxOp>(),
    TypeID::get<TF::UnsortedSegmentMinOp>(),
    TypeID::get<TF::UnsortedSegmentProdOp>(),
    TypeID::get<TF::UnsortedSegmentSumOp>(),
    TypeID::get<TF::XdivyOp>(),
    TypeID::get<TF::XlaAllReduceOp>(),
    TypeID::get<TF::XlaGatherOp>(),
    TypeID::get<TF::Xlog1pyOp>(),
    TypeID::get<TF::XlogyOp>(),
    TypeID::get<TF::ZerosLikeOp>(),
    TypeID::get<TF::ZetaOp>(),
  };
  // clang-format on

  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return false;
  return ops->count(abstractOp->getTypeID());
}
// LINT.ThenChange()

// List of ops that require falling back to XlaOpKernel legalizations and also
// require the ability to create functions.
bool IsOpAllowedTf2XlaFallbackAndCreateFunctions(Operation* op) {
  static auto* ops = new llvm::SmallDenseSet<mlir::TypeID, 16>{
      TypeID::get<TF::ApproxTopKOp>(),
  };
  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return false;
  return ops->count(abstractOp->getTypeID());
}

bool HasTf2XlaFallback(Operation* op) {
  return IsOpAllowedTf2XlaFallback(op) ||
         IsOpAllowedTf2XlaFallbackAndCreateFunctions(op) ||
         IsOpAllowedTf2XlaPreferred(op);
}

namespace {

// Returns true if the given type is a ranked tensor type with static or bounded
// dimensions.
bool IsBounded(Type ty) {
  auto ranked_ty = ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return false;

  if (ranked_ty.hasStaticShape()) return true;

  auto encoding =
      ranked_ty.getEncoding().dyn_cast_or_null<TypeExtensionsAttr>();
  if (!encoding) return false;

  for (int i = 0; i < ranked_ty.getRank(); ++i) {
    if (ranked_ty.isDynamicDim(i) &&
        encoding.getBounds()[i] == ShapedType::kDynamic) {
      return false;
    }
  }
  return true;
}

bool HasSymbolRefAttr(Operation* op) {
  for (const auto& attr : op->getAttrs()) {
    Attribute attr_value = attr.getValue();
    if (attr_value.isa<SymbolRefAttr>()) {
      return true;
    } else if (auto array_attr = attr_value.dyn_cast<ArrayAttr>()) {
      if (!array_attr.empty() && array_attr.begin()->isa<SymbolRefAttr>()) {
        return true;
      }
    }
  }
  return false;
}

class Tf2XlaRewritePattern : public ConversionPattern {
 public:
  explicit Tf2XlaRewritePattern(MLIRContext* ctx, TypeConverter& converter,
                                const std::string& device_type,
                                bool prefer_tf2xla, bool is_module_pass,
                                bool use_tf2xla_hlo_importer)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        device_type_(device_type),
        prefer_tf2xla_(prefer_tf2xla),
        is_module_pass_(is_module_pass),
        use_tf2xla_hlo_importer_(use_tf2xla_hlo_importer) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    // This pattern is a conversion pattern because we want to specify a type
    // converter. However, this pattern still uses the original op's operands
    // while creating the ops so make sure there aren't any type changes between
    // the original op operands and the operands during the conversion.
    for (auto&& [old_val, new_val] : llvm::zip(op->getOperands(), operands)) {
      if (old_val.getType() != new_val.getType()) return failure();
    }

    if (is_module_pass_) {
      // Module passes should only ever legalize ops that have been specifically
      // whitelisted for legalization within a module pass. They will never
      // legalize any ops whitelisted for legalization within a func pass.
      if (!IsOpAllowedTf2XlaFallbackAndCreateFunctions(op)) {
        return failure();
      }
    } else if (!(IsOpAllowedTf2XlaFallback(op) ||
                 (prefer_tf2xla_ && IsOpAllowedTf2XlaPreferred(op)))) {
      return failure();
    }
    return Tf2XlaRewriter::RewriteOp(op, rewriter, device_type_,
                                     is_module_pass_, use_tf2xla_hlo_importer_);
  }

 private:
  std::string device_type_;
  bool prefer_tf2xla_;
  bool is_module_pass_;
  bool use_tf2xla_hlo_importer_;
};

bool ShouldRefineTypeTo(Type original_ty, Type updated_ty) {
  auto updated = updated_ty.dyn_cast<ShapedType>();
  auto original = original_ty.dyn_cast<ShapedType>();

  // Both types must be shaped types.
  if (!original || !updated) return false;

  // Element types must match.
  if (original.getElementType() != updated.getElementType()) return false;

  // If the updated type doesn't have a rank, then it can't be a more refined
  // type.
  if (!updated.hasRank()) return false;

  // If the original type doesn't have a rank, then refine as the updated type
  // has a rank.
  if (!original.hasRank()) return true;

  // Both types must have the same rank.
  if (original.getRank() != updated.getRank()) return false;

  // Refine if the updated type is bounded.
  return IsBounded(updated);
}

// Propagates more refined type by cloning op using the new operands. This
// allows all rewrite patterns that requires refined types to work without
// requiring a rewrite to the conversion pattern. Declarative rewrite pattern
// (DRR) doesn't even support conversion patterns with TableGen.
class TypePropagator : public ConversionPattern {
 public:
  explicit TypePropagator(MLIRContext* ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    // This could be generalized to other ops as needs arise. We could even
    // remove this restriction altogether except for the terminators that
    // require function signature change and shouldn't be
    if (op->getName().getDialectNamespace() !=
        TF::TensorFlowDialect::getDialectNamespace())
      return failure();

    // Refining types may have implications to the attached regions or symbol
    // references so do not update such ops.
    if (!op->getRegions().empty() || HasSymbolRefAttr(op)) return failure();

    IRMapping mapper;
    bool has_type_change = false;
    for (auto [original, updated] : llvm::zip(op->getOperands(), operands)) {
      Type original_ty = original.getType();
      Type updated_ty = updated.getType();
      if (original_ty != updated_ty) has_type_change = true;

      if (!ShouldRefineTypeTo(original_ty, updated_ty)) return failure();
      mapper.map(original, updated);
    }
    if (!has_type_change) return failure();

    Operation* cloned_op = rewriter.clone(*op, mapper);
    rewriter.replaceOp(op, cloned_op->getResults());
    return success();
  }
};

}  // end namespace

Tf2XlaTypeConverter::Tf2XlaTypeConverter() {
  // Currently, we don't do any type conversions. Any TensorFlow op with a type
  // that is not supported in MHLO will fail conversion. Quantized types are
  // going to handled separately so we don't need to handle those.
  addConversion([](Type ty) { return ty; });

  // This materialization is helpful in cases where we have more refined types
  // after conversion to mhlo compared to the original type in TF. For example,
  // a TF op with result type tensor<*xf32> will have a bounded type after
  // fallback legalization.
  auto cast_value = [&](OpBuilder& builder, Type result_type, ValueRange inputs,
                        Location loc) -> Value {
    return builder.create<mlir::tensor::CastOp>(loc, result_type,
                                                inputs.front());
  };
  addSourceMaterialization(cast_value);
}

void PopulateLegalizeTfWithTf2XlaPatterns(
    llvm::StringRef device_type, RewritePatternSet& patterns, MLIRContext* ctx,
    Tf2XlaTypeConverter& converter, bool prefer_tf2xla, bool is_module_pass,
    bool use_tf2xla_hlo_importer) {
  patterns.add<TypePropagator>(ctx);
  patterns.add<Tf2XlaRewritePattern>(ctx, converter, device_type.str(),
                                     prefer_tf2xla, is_module_pass,
                                     use_tf2xla_hlo_importer);
}

}  // end namespace mhlo
}  // end namespace mlir
