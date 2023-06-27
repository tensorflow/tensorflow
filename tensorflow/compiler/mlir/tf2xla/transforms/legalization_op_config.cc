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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace mhlo {

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
      TypeID::get<TF::ConcatV2Op>(),

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
      // See b/216353817 how to reproduce the bug regarding tf.StridedSlice Op
      // See b/245615401 how to reproduce the bug regarding tf.SliceOp
      TypeID::get<TF::RandomUniformOp>(),
      TypeID::get<TF::StridedSliceOp>(),
      TypeID::get<TF::SliceOp>(),

      // Conditional ops
      TypeID::get<TF::IfRegionOp>(),
      TypeID::get<TF::WhileRegionOp>(),
      TypeID::get<TF::CaseRegionOp>(),
      TypeID::get<TF::YieldOp>(),
  };
  return *ops;
}

}  // namespace

bool IsOpLegalizedWithMlir(Operation& op) {
  auto abstractOp = op.getRegisteredInfo();
  if (!abstractOp) return false;
  return IsTypeLegalizedWithMlir(abstractOp->getTypeID());
}

bool IsTypeLegalizedWithMlir(const TypeID& type_id) {
  return MlirAlwaysOps().contains(type_id);
}

}  // namespace mhlo
}  // namespace mlir
