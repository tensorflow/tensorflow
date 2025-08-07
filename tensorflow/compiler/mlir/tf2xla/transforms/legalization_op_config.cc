/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/transforms/legalization_op_config.h"

#include "llvm/ADT/DenseSet.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tpu_embedding_ops_registry.h"

namespace mlir {
namespace hlo {

namespace {

// Returns ops that should use MLIR legalization.
// All other ops not in this list should use XlaOpKernel.
const llvm::DenseSet<mlir::TypeID>& MlirAlwaysOps() {
  // The static variable is a pointer in order to avoid destruction upon thread
  // termination.
  static const llvm::DenseSet<mlir::TypeID>* ops = new llvm::DenseSet<
      mlir::TypeID>{
      // Ops that should always use the MLIR legalization.
      TypeID::get<TF::FusedBatchNormV3Op>(),
      TypeID::get<TF::FusedBatchNormGradV3Op>(),
      TypeID::get<TF::XlaReduceScatterOp>(),
      TypeID::get<TF::ModOp>(),

      // MatrixDiagPartV3 should use the MLIR implementation due to performance.
      TypeID::get<TF::MatrixDiagPartV3Op>(),

      // Ops that are legalized in the old bridge using MlirXlaOpKernel
      TypeID::get<TF::AbsOp>(),
      TypeID::get<TF::AtanOp>(),
      TypeID::get<TF::AvgPool3DOp>(),
      TypeID::get<TF::BiasAddGradOp>(),
      TypeID::get<TF::CeilOp>(),
      TypeID::get<TF::CheckNumericsOp>(),
      TypeID::get<TF::CosOp>(),
      TypeID::get<TF::TanOp>(),
      TypeID::get<TF::DiagPartOp>(),
      TypeID::get<TF::EinsumOp>(),
      TypeID::get<TF::ExpOp>(),
      TypeID::get<TF::Expm1Op>(),
      TypeID::get<TF::FakeQuantWithMinMaxArgsOp>(),
      TypeID::get<TF::FloorOp>(),
      TypeID::get<TF::IFFTOp>(),
      TypeID::get<TF::ImagOp>(),
      TypeID::get<TF::IsFiniteOp>(),
      TypeID::get<TF::IsInfOp>(),
      TypeID::get<TF::IsNanOp>(),
      TypeID::get<TF::LgammaOp>(),
      TypeID::get<TF::Log1pOp>(),
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
      TypeID::get<TF::TanhOp>(),
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

      // Ops that have no XlaOpKernel.
      TypeID::get<TF::RiscAddOp>(),
      TypeID::get<TF::RiscDotOp>(),

      // Const op has a simple legalization and it is much more efficient to
      // lower
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

      // These ops have undetermined bugs, may not be legalizable with
      // XlaOpKernel
      // legalization in TF2XLA fallback. By legalization with MLIR, we can fix
      // the bug. b/195583695 describes the motivation of this change.
      // See b/216355804 how to reproduce the bug regarding tf.RandomUniform Op

      // Conditional ops
      TypeID::get<TF::IfRegionOp>(),
      TypeID::get<TF::WhileRegionOp>(),
      TypeID::get<TF::CaseRegionOp>(),
      TypeID::get<TF::YieldOp>(),
  };
  return *ops;
}

bool IsOpTypeAllowedTf2XlaFallback(const TypeID& type_id) {
  // Allowlisted TensorFlow ops are known to have well behaved tf2xla kernels
  // building valid MLIR using MlirHloBuilder.
  // TODO(hinsu): Drop explicit allowlist when MLIR based bridge is enabled for
  // all tf2xla kernels.
  // Use a pointer for the static set, so the set is not destructed upon thread
  // end, which would not be thread safe.

  static auto* ops = [] {
    llvm::SmallDenseSet<mlir::TypeID, 512>* ops_set = new llvm::SmallDenseSet<
        mlir::TypeID, 512>{
        TypeID::get<TF::AcoshOp>(),
        TypeID::get<TF::AcosOp>(),
        TypeID::get<TF::AddNOp>(),
        TypeID::get<TF::AddV2Op>(),
        TypeID::get<TF::AngleOp>(),
        TypeID::get<TF::AdjustContrastv2Op>(),
        TypeID::get<TF::AdjustHueOp>(),
        TypeID::get<TF::AdjustSaturationOp>(),
        TypeID::get<TF::ApproximateEqualOp>(),
        TypeID::get<TF::ApproxTopKOp>(),
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
        TypeID::get<TF::CollectiveReduceV2Op>(),
        TypeID::get<TF::ComplexAbsOp>(),
        TypeID::get<TF::ConjugateTransposeOp>(),
        TypeID::get<TF::ConcatV2Op>(),
        TypeID::get<TF::ConvOp>(),
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
        TypeID::get<TF::GetMinibatchesInCsrWithPhysicalReplicaOp>(),
        TypeID::get<TF::GetMinibatchSplitsWithPhysicalReplicaOp>(),
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
        TypeID::get<TF::MaxPoolGradOp>(),
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
        TypeID::get<TF::RandomUniformOp>(),
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
        TypeID::get<TF::SliceOp>(),
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
        TypeID::get<TF::StoreMinibatchStatisticsInFdoOp>(),
        TypeID::get<TF::StridedSliceOp>(),
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
        TypeID::get<TF::XlaSendTPUEmbeddingGradientsOp>(),
        TypeID::get<TF::XlaBroadcastHelperOp>(),
        TypeID::get<TF::XlaCallModuleOp>(),
        TypeID::get<TF::XlaCustomCallV2Op>(),
        TypeID::get<TF::XlaDynamicUpdateSliceOp>(),
        TypeID::get<TF::XlaKeyValueSortOp>(),
        TypeID::get<TF::XlaPadOp>(),
        TypeID::get<TF::XlaSetBoundOp>(),
        TypeID::get<TF::XlaSetDynamicDimensionSizeOp>(),
        TypeID::get<TF::XlaSparseActivationsUnstackOp>(),
        TypeID::get<TF::XlaSparseCoreAdagradMomentumOp>(),
        TypeID::get<TF::XlaSparseCoreAdagradOp>(),
        TypeID::get<TF::XlaSparseCoreAdamOp>(),
        TypeID::get<TF::XlaSparseCoreFtrlOp>(),
        TypeID::get<TF::XlaSparseCoreSgdOp>(),
        TypeID::get<TF::XlaSparseDenseMatmulGradWithAdagradAndCsrInputOp>(),
        TypeID::get<
            TF::XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOp>(),
        TypeID::get<TF::XlaSparseDenseMatmulGradWithAdamAndCsrInputOp>(),
        TypeID::get<TF::XlaSparseDenseMatmulGradWithFtrlAndCsrInputOp>(),
        TypeID::get<TF::XlaSparseDenseMatmulGradWithSgdAndCsrInputOp>(),
        TypeID::get<TF::XlaSparseDenseMatmulWithCsrInputOp>(),
        TypeID::get<TF::XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInputOp>(),
        TypeID::get<TF::XlaSparseDenseMatmulWithStaticBufferSizeOp>(),
        TypeID::get<
            TF::XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSizeOp>(),
        TypeID::get<
            TF::XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSizeOp>(),  // NOLINT
        TypeID::get<
            TF::XlaSparseDenseMatmulGradWithAdamAndStaticBufferSizeOp>(),
        TypeID::get<
            TF::XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSizeOp>(),
        TypeID::get<
            TF::XlaSparseDenseMatmulGradWithSgdAndStaticBufferSizeOp>(),  // NOLINT
        TypeID::get<TF::XlaSparseDenseMatmulGradWithCsrInputOp>(),
        TypeID::get<
            TF::XlaSparseDenseMatmulCustomCombinerOnTcGradWithSgdAndCsrInputOp>(),  // NOLINT
        TypeID::get<
            TF::XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdagradAndCsrInputOp>(),  // NOLINT
        TypeID::get<
            TF::XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdagradMomentumAndCsrInputOp>(),  // NOLINT
        TypeID::get<
            TF::XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdamAndCsrInputOp>(),  // NOLINT
        TypeID::get<
            TF::XlaSparseDenseMatmulCustomCombinerOnTcGradWithFtrlAndCsrInputOp>(),  // NOLINT
        TypeID::get<
            TF::XlaSparseDenseMatmulCustomCombinerOnTcGradWithCsrInputOp>(),
        TypeID::get<TF::XlaLocalSparseDenseMatmulOp>(),  // NOLINT
        TypeID::get<TF::XlaSparseGradientsStackOp>(),
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

  return ops->count(type_id);
}

/// List of ops that should use XlaOpKernel legalization only in the case of
/// prefer_tf2xla. All other ops not in this list should use MLIR legalization
/// only or not be legalized by the new bridge.
bool IsOpTypeAllowedTf2XlaPreferred(const TypeID& type_id) {
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
    TypeID::get<TF::RandomUniformOp>(),
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
    TypeID::get<TF::StridedSliceOp>(),
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
    TypeID::get<TF::XlaSendTPUEmbeddingGradientsOp>(),
    TypeID::get<TF::XlaAllReduceOp>(),
    TypeID::get<TF::XlaGatherOp>(),
    TypeID::get<TF::Xlog1pyOp>(),
    TypeID::get<TF::XlogyOp>(),
    TypeID::get<TF::ZerosLikeOp>(),
    TypeID::get<TF::ZetaOp>(),
  };
  // clang-format on

  return ops->contains(type_id);
}

const llvm::DenseSet<mlir::TypeID>& DynamicTensorflowOps() {
  // The static variable is a pointer in order to avoid destruction upon thread
  // termination.
  static const llvm::DenseSet<mlir::TypeID>* ops =
      new llvm::DenseSet<mlir::TypeID>{
          TypeID::get<mlir::TF::DynamicPartitionOp>(),
          TypeID::get<mlir::TF::UniqueOp>(),
          TypeID::get<mlir::TF::WhereOp>(),
          TypeID::get<mlir::TF::XlaSetDynamicDimensionSizeOp>(),
      };
  return *ops;
}

}  // namespace

bool HasTf2XlaFallback(const TypeID& type_id) {
  return IsOpTypeAllowedTf2XlaFallback(type_id) ||
         IsOpTypeAllowedTf2XlaPreferred(type_id);
}

bool IsTypeLegalizedWithMlir(const TypeID& type_id) {
  return MlirAlwaysOps().contains(type_id);
}

bool IsOpAllowedTf2xlaFallback(const TypeID& type_id) {
  return IsOpTypeAllowedTf2XlaFallback(type_id);
}

bool IsOpAllowedTf2xlaPreferred(const TypeID& type_id) {
  return IsOpTypeAllowedTf2XlaPreferred(type_id);
}

bool IsDynamicPadderOp(const TypeID& type_id) {
  return DynamicTensorflowOps().contains(type_id);
}

}  // namespace hlo
}  // namespace mlir
