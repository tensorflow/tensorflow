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
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.h"
#include "tensorflow/compiler/mlir/xla/transforms/tf_xla_passes_detail.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
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
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/stream_executor.h"

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
  // clang-format off

  static auto* ops =
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
    TypeID::get<TF::CastOp>(),
    TypeID::get<TF::ClipByValueOp>(),
    TypeID::get<TF::CholeskyOp>(),
    TypeID::get<TF::ComplexAbsOp>(),
    TypeID::get<TF::ConjugateTransposeOp>(),
    TypeID::get<TF::CoshOp>(),
    TypeID::get<TF::CrossOp>(),
    TypeID::get<TF::DataFormatDimMapOp>(),
    TypeID::get<TF::DataFormatVecPermuteOp>(),
    TypeID::get<TF::DepthToSpaceOp>(),
    TypeID::get<TF::DepthwiseConv2dNativeBackpropFilterOp>(),
    TypeID::get<TF::DepthwiseConv2dNativeBackpropInputOp>(),
    TypeID::get<TF::DiagOp>(),
    TypeID::get<TF::DigammaOp>(),
    TypeID::get<TF::DivNoNanOp>(),
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
    TypeID::get<TF::TruncateDivOp>(),
    TypeID::get<TF::TruncatedNormalOp>(),
    TypeID::get<TF::TruncateModOp>(),
    TypeID::get<TF::UnpackOp>(),
    TypeID::get<TF::UpperBoundOp>(),
    TypeID::get<TF::XlaBroadcastHelperOp>(),
    TypeID::get<TF::XlaConvOp>(),
    TypeID::get<TF::XlaConvV2Op>(),
    TypeID::get<TF::XlaDynamicUpdateSliceOp>(),
    TypeID::get<TF::XlaKeyValueSortOp>(),
    TypeID::get<TF::XlaPadOp>(),
    TypeID::get<TF::XlaSvdOp>(),
  };
  // clang-format on

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
    TypeID::get<TF::DynamicStitchOp>(),
    TypeID::get<TF::_EagerConstOp>(),
    TypeID::get<TF::EmptyOp>(),
    TypeID::get<TF::ExpandDimsOp>(),
    TypeID::get<TF::FakeQuantWithMinMaxVarsOp>(),
    TypeID::get<TF::FillOp>(),
    TypeID::get<TF::FusedBatchNormOp>(),
    TypeID::get<TF::FusedBatchNormGradOp>(),
    TypeID::get<TF::FusedBatchNormGradV2Op>(),
    TypeID::get<TF::FusedBatchNormGradV3Op>(),
    TypeID::get<TF::FusedBatchNormV2Op>(),
    TypeID::get<TF::FusedBatchNormV3Op>(),
    TypeID::get<TF::GatherNdOp>(),
    TypeID::get<TF::GatherV2Op>(),
    TypeID::get<TF::IdentityOp>(),
    TypeID::get<TF::IdentityNOp>(),
    TypeID::get<TF::InplaceUpdateOp>(),
    TypeID::get<TF::InvertPermutationOp>(),
    TypeID::get<TF::IRFFTOp>(),
    TypeID::get<TF::L2LossOp>(),
    TypeID::get<TF::LegacyCallOp>(),
    TypeID::get<TF::LinSpaceOp>(),
    TypeID::get<TF::MatrixDiagPartV3Op>(),
    TypeID::get<TF::MaxOp>(),
    TypeID::get<TF::MaximumOp>(),
    TypeID::get<TF::MaxPoolOp>(),
    TypeID::get<TF::MaxPool3DOp>(),
    TypeID::get<TF::MaxPoolGradOp>(),
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
    TypeID::get<TF::SqueezeOp>(),
    TypeID::get<TF::StatelessParameterizedTruncatedNormalOp>(),
    TypeID::get<TF::StatefulPartitionedCallOp>(),
    TypeID::get<TF::StopGradientOp>(),
    TypeID::get<TF::StridedSliceGradOp>(),
    TypeID::get<TF::SumOp>(),
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
    TypeID::get<TF::ZerosLikeOp>(),
  };
  // clang-format on
  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return false;
  return ops->count(abstractOp->getTypeID());
}
// LINT.ThenChange()

bool IsOpAllowedForTesting(Operation* op) {
  // clang-format off
  static auto* ops =
      new llvm::SmallDenseSet<mlir::TypeID, 16>{
    // Op used to verify handling of XlaExpression of kind constant.
    TypeID::get<TF::ConstOp>(),
  };
  // clang-format on
  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return false;
  return ops->count(abstractOp->getTypeID());
}

namespace {

template <typename T, size_t N>
using InlinedVector = tensorflow::gtl::InlinedVector<T, N>;  // non-absl ok

static std::unique_ptr<tensorflow::StaticDeviceMgr> CreateDeviceMgr(
    const std::string& device_type) {
  // Register compilation kernels for all registered XLA backends.
  tensorflow::XlaOpRegistry::RegisterCompilationKernels();

  auto device = std::make_unique<tensorflow::XlaCompilationDevice>(
      tensorflow::SessionOptions(), tensorflow::DeviceType(device_type));
  return std::make_unique<tensorflow::StaticDeviceMgr>(std::move(device));
}

class Tf2XlaRewriter {
 public:
  static LogicalResult RewriteOp(Operation* op, PatternRewriter& rewriter,
                                 const std::string& device_type) {
    Tf2XlaRewriter tf2xla_rewriter(op, rewriter, device_type);
    return tf2xla_rewriter.LegalizeOp();
  }

 private:
  Tf2XlaRewriter(Operation* op, PatternRewriter& rewriter,
                 const std::string& device_type)
      : op_(op),
        device_type_(device_type),
        rewriter_(rewriter),
        hlo_builder_(op->getName().getStringRef().str(), rewriter_,
                     op->getLoc()),
        context_(nullptr) {}

  ~Tf2XlaRewriter() {
    if (context_) context_->Unref();
  }

  // Prepares OpKernelContext params common to all the ops.
  // Emits an error on failure.
  LogicalResult PrepareParams();

  // Tries to legalize the specified TensorFlow op, if supported.
  //
  // Emits an error and returns failure if an error is encountered during
  // conversion. Note that success return value doesn't mean successful
  // legalization.
  LogicalResult LegalizeOp();

  // Converts the given operand to expression of kind kConstant or kXlaOp.
  // Emits a remark and returns expression of kind kInvalid on failure.
  tensorflow::XlaExpression GetExprForOperand(Value operand, Operation* op);

  Operation* op_;
  std::string device_type_;

  PatternRewriter& rewriter_;
  ::xla::MlirHloBuilder hlo_builder_;
  tensorflow::OpOrArgLocNameMapper name_mapper_;

  tensorflow::XlaContext* context_;  // Ref-counted.

  std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr_;
  tensorflow::Device* device_;  // Owned by device_mgr_;
  std::unique_ptr<tensorflow::ScopedStepContainer> step_container_;
  std::unique_ptr<tensorflow::FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<tensorflow::ProcessFunctionLibraryRuntime> pflr_;
  tensorflow::OpKernelContext::Params params_;
};

LogicalResult Tf2XlaRewriter::PrepareParams() {
  // XlaCompiler within the context is only used by the functional ops to
  // compile functions. We are not handling those at the moment so XlaCompiler
  // is not required.
  context_ = new tensorflow::XlaContext(/*compiler=*/nullptr, &hlo_builder_,
                                        /*graph=*/nullptr);
  context_->Ref();

  device_mgr_ = CreateDeviceMgr(device_type_);
  if (!device_mgr_) return failure();

  // Type of params_.device is DeviceBase* so store it as Device* to access
  // derived class method.
  device_ = device_mgr_->ListDevices().front();
  params_.device = device_;
  params_.resource_manager = device_->resource_manager();

  // Resources are cleared at the time of device manager destruction so pass
  // no-op cleanup function.
  auto cleanup = [](const std::string& name) {};
  // Use step_id zero as we only have a single context concurrently and
  // concurrently running each of the MLIR functions create a new device.
  step_container_ = std::make_unique<tensorflow::ScopedStepContainer>(
      /*step_id=*/0, cleanup);
  tensorflow::Status status = step_container_->Create(
      device_->resource_manager(),
      tensorflow::XlaContext::kXlaContextResourceName, context_);
  if (!status.ok()) {
    return emitRemark(op_->getLoc())
           << "failed to create XlaContext resource: " << status.ToString();
  }
  params_.step_container = step_container_.get();

  tensorflow::StatusOr<int64_t> version_or =
      tensorflow::GetTfGraphProducerVersion(
          op_->getParentOfType<mlir::ModuleOp>());
  if (!version_or.ok()) {
    return emitError(op_->getLoc()) << version_or.status().ToString();
  }

  flib_def_ = std::make_unique<tensorflow::FunctionLibraryDefinition>(
      tensorflow::OpRegistry::Global(), tensorflow::FunctionDefLibrary());
  pflr_ = std::make_unique<tensorflow::ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), tensorflow::Env::Default(), /*config=*/nullptr,
      version_or.ValueOrDie(), flib_def_.get(), tensorflow::OptimizerOptions());
  params_.function_library = pflr_->GetFLR(device_->name());
  return success();
}

LogicalResult Tf2XlaRewriter::LegalizeOp() {
  // Only static shaped operands are supported in XLA builders for now.
  for (Type ty : op_->getOperandTypes()) {
    auto ranked_ty = ty.dyn_cast<ShapedType>();
    if (!ranked_ty || !ranked_ty.hasStaticShape()) {
      return op_->emitRemark()
             << "lowering requires static shaped tensor operands";
    }
  }

  for (const auto& attr : op_->getAttrs()) {
    if (attr.getValue().isa<SymbolRefAttr>()) {
      return op_->emitRemark()
             << "ops with symbol references are not supported";
    }
  }

  auto nodedef_or = tensorflow::ConvertTFDialectOpToNodeDef(
      op_, name_mapper_.GetUniqueName(op_), /*ignore_unregistered_attrs=*/true);
  if (!nodedef_or.ok()) {
    return op_->emitRemark() << "failed to convert op to NodeDef: "
                             << nodedef_or.status().ToString();
  }

  if (failed(PrepareParams())) return failure();

  std::shared_ptr<const tensorflow::NodeProperties> props;
  tensorflow::Status status = tensorflow::NodeProperties::CreateFromNodeDef(
      *nodedef_or.ValueOrDie(),
      params_.function_library->GetFunctionLibraryDefinition(), &props);
  if (!status.ok()) {
    return op_->emitRemark()
           << "failed to create NodeProperties: " << status.ToString();
  }
  tensorflow::OpKernel* op_kernel_raw;
  status = params_.function_library->CreateKernel(props, &op_kernel_raw);
  if (!status.ok()) {
    return op_->emitRemark()
           << "failed to create tf2xla kernel: " << status.ToString();
  }
  // Transfer ownership of the kernel to a local smart pointer.
  auto op_kernel = absl::WrapUnique(op_kernel_raw);

  std::vector<int> required_constants;
  status = tensorflow::XlaOpRegistry::CompileTimeConstantInputs(
      *op_kernel, &required_constants);
  if (!status.ok()) {
    return op_->emitRemark()
           << "failed to compute required constants: " << status.ToString();
  }
  llvm::SmallDenseSet<int, 4> required_consts;
  required_consts.insert(required_constants.begin(), required_constants.end());

  // TensorValue in inputs are backed by tensors which in turn depend on
  // expressions. So, pre-allocate them to the required size.
  InlinedVector<tensorflow::XlaExpression, 4> expressions;
  InlinedVector<tensorflow::Tensor, 4> tensors;
  InlinedVector<tensorflow::TensorValue, 4> inputs;
  expressions.reserve(op_->getNumOperands());
  tensors.reserve(op_->getNumOperands());
  inputs.reserve(op_->getNumOperands());

  // Prepare the list of Tensor inputs for the kernel.
  for (auto it : llvm::enumerate(op_->getOperands())) {
    Value operand = it.value();
    size_t idx = it.index();

    tensorflow::XlaExpression expr = GetExprForOperand(operand, op_);
    tensorflow::XlaExpression::Kind kind = expr.kind();
    if (kind == tensorflow::XlaExpression::Kind::kInvalid) return failure();
    if (required_consts.count(idx) &&
        kind != tensorflow::XlaExpression::Kind::kConstant) {
      return op_->emitRemark()
             << "lowering requires operand #" << idx << " to be a constant";
    }
    expressions.push_back(expr);

    if (!tensorflow::DataTypeCanUseMemcpy(expr.dtype())) {
      return op_->emitRemark()
             << "skipping legalization due to unsupported type "
             << operand.getType();
    }

    auto shape_or = expr.GetShape();
    if (!shape_or.ok()) {
      return op_->emitRemark()
             << "failed to get shape for expression. " << expr.HumanString();
    }

    tensors.emplace_back(
        device_->GetAllocator(tensorflow::AllocatorAttributes()), expr.dtype(),
        shape_or.ValueOrDie());
    tensorflow::Tensor& tensor = tensors.back();
    tensorflow::XlaExpression::AssignExpressionToTensor(expr, &tensor);
    inputs.emplace_back(&tensor);
  }

  params_.inputs = &inputs;
  params_.op_kernel = op_kernel.get();
  llvm::SmallVector<tensorflow::AllocatorAttributes, 4> output_attr(
      op_->getNumResults());
  params_.output_attr_array = output_attr.data();

  hlo_builder_.setInsertionPoint(op_);
  hlo_builder_.SetLocation(op_->getLoc());

  // Execute the kernel.
  tensorflow::OpKernelContext op_context(&params_, op_->getNumResults());
  device_->Compute(params_.op_kernel, &op_context);

  status = op_context.status();
  status.Update(hlo_builder_.GetCurrentStatus());
  if (!status.ok()) {
    return op_->emitRemark()
           << "compilation to HLO failed: " << status.ToString();
  }

  // Replace uses of old results using the corresponding value after the
  // lowering.
  llvm::SmallVector<Value, 2> values;
  values.reserve(op_->getNumResults());
  for (int i = 0, e = op_->getNumResults(); i < e; i++) {
    tensorflow::Tensor* output = op_context.mutable_output(i);
    const tensorflow::XlaExpression* expr =
        tensorflow::XlaExpression::CastExpressionFromTensor(*output);
    if (expr->kind() != tensorflow::XlaExpression::Kind::kXlaOp &&
        expr->kind() != tensorflow::XlaExpression::Kind::kConstant) {
      return op_->emitRemark(
          "expects XlaExpression of kind kXlaOp or kConstant in compiled "
          "output");
    }
    mlir::Value value = hlo_builder_.GetValue(expr->AsXlaOp(&hlo_builder_));
    mlir::OpResult old_result = op_->getResult(i);
    if (value.getType() != old_result.getType()) {
      value = hlo_builder_.create<mlir::tensor::CastOp>(old_result.getType(),
                                                        value);
    }
    values.push_back(value);
  }
  rewriter_.replaceOp(op_, values);
  return success();
}

tensorflow::XlaExpression Tf2XlaRewriter::GetExprForOperand(Value operand,
                                                            Operation* op) {
  ElementsAttr const_attr;
  auto defining_op = operand.getDefiningOp();
  if (defining_op && matchPattern(defining_op, m_Constant(&const_attr))) {
    tensorflow::Tensor tensor;
    auto status = tensorflow::ConvertToTensor(const_attr, &tensor);
    if (!status.ok()) {
      op->emitRemark() << "skipping legalization due to failed const conversion"
                       << status.ToString();
      return tensorflow::XlaExpression::Invalid();
    }
    return tensorflow::XlaExpression::Constant(tensor);
  }

  // Skip this op if XLA doesn't support this operand type.
  auto xla_op_or = hlo_builder_.MakeXlaOp(operand);
  if (!xla_op_or.ok()) {
    op->emitRemark() << "skipping legalization due to "
                     << xla_op_or.status().ToString();
    return tensorflow::XlaExpression::Invalid();
  }
  ::xla::XlaOp xla_op = xla_op_or.ValueOrDie();

  tensorflow::DataType dtype;
  auto status = tensorflow::ConvertToDataType(operand.getType(), &dtype);
  if (!status.ok()) {
    op->emitRemark() << "skipping legalization due to " << status.ToString();
    return tensorflow::XlaExpression::Invalid();
  }
  return tensorflow::XlaExpression::XlaOp(xla_op, dtype);
}

class Tf2XlaRewritePattern : public RewritePattern {
 public:
  explicit Tf2XlaRewritePattern(MLIRContext* ctx,
                                const std::string& device_type,
                                bool prefer_tf2xla, bool legalize_test_only_ops)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        device_type_(device_type),
        prefer_tf2xla_(prefer_tf2xla),
        legalize_test_only_ops_(legalize_test_only_ops) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (!(IsOpAllowedTf2XlaFallback(op) ||
          (prefer_tf2xla_ && IsOpAllowedTf2XlaPreferred(op)) ||
          (legalize_test_only_ops_ && IsOpAllowedForTesting(op))))
      return failure();
    return Tf2XlaRewriter::RewriteOp(op, rewriter, device_type_);
  }

 private:
  std::string device_type_;
  bool prefer_tf2xla_;
  bool legalize_test_only_ops_;
};

class LegalizeTF : public LegalizeTFPassBase<LegalizeTF> {
 public:
  LegalizeTF() = default;
  explicit LegalizeTF(llvm::StringRef device_type, bool prefer_tf2xla) {
    device_type_ = device_type.str();
    prefer_tf2xla_ = prefer_tf2xla;
  }

  LegalizeTF(const LegalizeTF&) {}

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<Tf2XlaRewritePattern>(&getContext(), device_type_,
                                       prefer_tf2xla_, legalize_test_only_ops_);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

 private:
};

}  // end namespace

void PopulateLegalizeTfWithTf2XlaPatterns(llvm::StringRef device_type,
                                          RewritePatternSet& patterns,
                                          MLIRContext* ctx,
                                          bool prefer_tf2xla) {
  patterns.add<Tf2XlaRewritePattern>(ctx, device_type.str(), prefer_tf2xla,
                                     /*legalize_test_only_ops=*/false);
}

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTfWithTf2XlaPass(
    llvm::StringRef device_type, bool prefer_tf2xla) {
  return std::make_unique<LegalizeTF>(device_type, prefer_tf2xla);
}

}  // end namespace mhlo
}  // end namespace mlir
