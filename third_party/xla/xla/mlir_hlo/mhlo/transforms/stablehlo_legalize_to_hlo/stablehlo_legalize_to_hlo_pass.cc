/* Copyright 2022 The OpenXLA Authors.

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

#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZETOHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

// Keep this list until Direct StableHLO to HLO Path is well tested in prod.
/*
void legalDirectStablehloToHloConversionOps(
    ConversionTarget& target) {
  target.addLegalOp<
      stablehlo::AbsOp, stablehlo::CbrtOp, stablehlo::SqrtOp, stablehlo::TanOp,
      stablehlo::AddOp, stablehlo::AddOp, stablehlo::AllGatherOp,
      stablehlo::AfterAllOp, stablehlo::AndOp, stablehlo::BatchNormInferenceOp,
      stablehlo::Atan2Op, stablehlo::BroadcastInDimOp, stablehlo::BroadcastOp,
      stablehlo::CeilOp, stablehlo::ClzOp, stablehlo::ConvertOp,
      stablehlo::CholeskyOp, stablehlo::CollectivePermuteOp,
      stablehlo::ComplexOp, stablehlo::ConvolutionOp, stablehlo::CosineOp,
      stablehlo::ConcatenateOp, stablehlo::ConstantOp, stablehlo::DivOp,
      stablehlo::MaxOp, stablehlo::EinsumOp, stablehlo::FftOp,
      stablehlo::DynamicUpdateSliceOp, stablehlo::DynamicBroadcastInDimOp,
      stablehlo::ExpOp, stablehlo::IsFiniteOp, stablehlo::Expm1Op,
      stablehlo::CrossReplicaSumOp, stablehlo::FloorOp,
      stablehlo::GetDimensionSizeOp, stablehlo::NegOp, stablehlo::NotOp,
      stablehlo::ImagOp, stablehlo::DynamicSliceOp, stablehlo::LogOp,
      stablehlo::LogisticOp, stablehlo::Log1pOp, stablehlo::MinOp,
      stablehlo::MulOp, stablehlo::PowOp, stablehlo::OrOp,
      stablehlo::PopulationCountOp, stablehlo::RsqrtOp, stablehlo::SelectOp,
      stablehlo::ReplicaIdOp, stablehlo::RealOp, stablehlo::RoundNearestEvenOp,
      stablehlo::RoundOp, stablehlo::ReverseOp, stablehlo::RemOp,
      stablehlo::ShiftRightArithmeticOp, stablehlo::ShiftRightLogicalOp,
      stablehlo::SliceOp, stablehlo::TanhOp, stablehlo::TransposeOp,
      stablehlo::SubtractOp, stablehlo::SignOp, stablehlo::SineOp,
      stablehlo::TorchIndexSelectOp, stablehlo::ShiftLeftOp,
      stablehlo::TriangularSolveOp, stablehlo::XorOp, stablehlo::CreateTokenOp,
      stablehlo::TupleOp, stablehlo::SendOp, stablehlo::RecvOp,
      stablehlo::InfeedOp, stablehlo::OutfeedOp, stablehlo::GetTupleElementOp,
      stablehlo::OptimizationBarrierOp, stablehlo::WhileOp, stablehlo::CaseOp,
      stablehlo::IfOp, stablehlo::AllReduceOp, stablehlo::ReduceOp,
      stablehlo::MapOp, stablehlo::ReturnOp, stablehlo::AllToAllOp,
      stablehlo::BatchNormGradOp, stablehlo::BatchNormTrainingOp,
      stablehlo::BitcastConvertOp, stablehlo::ClampOp,
      stablehlo::CollectiveBroadcastOp, stablehlo::CompareOp, stablehlo::SortOp,
      stablehlo::CompositeOp, stablehlo::CustomCallOp, stablehlo::DotGeneralOp,
      stablehlo::DotOp, stablehlo::DynamicConvOp, stablehlo::DynamicGatherOp,
      stablehlo::DynamicPadOp, stablehlo::DynamicReshapeOp,
      stablehlo::DynamicIotaOp, stablehlo::ReshapeOp, stablehlo::GatherOp,
      stablehlo::IotaOp, stablehlo::PadOp, stablehlo::PartitionIdOp,
      stablehlo::RealDynamicSliceOp, stablehlo::ReduceWindowOp,
      stablehlo::ReducePrecisionOp, stablehlo::ReduceScatterOp,
      stablehlo::RngBitGeneratorOp, stablehlo::RngOp, stablehlo::ScatterOp,
      stablehlo::SelectAndScatterOp, stablehlo::SetDimensionSizeOp,
      stablehlo::UniformDequantizeOp, stablehlo::UniformQuantizeOp>();
}
*/

struct StablehloLegalizeToHloPass
    : public impl::StablehloLegalizeToHloPassBase<StablehloLegalizeToHloPass> {
  using StablehloLegalizeToHloPassBase::StablehloLegalizeToHloPassBase;
  void runOnOperation() override {
    ConversionTarget target(getContext());
    // All StableHLO ops can go on Direct StableHLO to HLO path. This pass is
    // is NO-OP for Direct StableHLO to HLO path.
    if (!convert_xla_supported_stablehlo_) {
      target.addLegalDialect<stablehlo::StablehloDialect>();
      return;
    }
    stablehlo::setupStablehloToHloConversionTarget(target);

    stablehlo::StablehloToHloTypeConverter converter(
        convert_xla_supported_stablehlo_);

    RewritePatternSet patterns(&getContext());
    stablehlo::populateStablehloToHloPatterns(&patterns, &converter,
                                              &getContext());
    stablehlo::registerFuncOpsForTypeConversion(target, patterns, converter);

    // Our guiding principle is to support all StableHLO functionality in MHLO.
    // This check is here only for exceptional situations, e.g. when we added
    // a new StableHLO op and forgot to update the conversion patterns.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

}  // namespace mhlo
}  // namespace mlir
