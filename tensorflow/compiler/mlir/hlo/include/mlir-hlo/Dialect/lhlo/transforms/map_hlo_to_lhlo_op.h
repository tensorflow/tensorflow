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

#ifndef MLIR_HLO_DIALECT_LHLO_TRANSFORMS_MAP_HLO_TO_LHLO_OP_H
#define MLIR_HLO_DIALECT_LHLO_TRANSFORMS_MAP_HLO_TO_LHLO_OP_H

#include <type_traits>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace mhlo {

template <typename HloOpTy>
struct HloToLhloOpImpl {
  using Type = std::false_type;
};
template <typename HloOpTy>
using HloToLhloOp = typename HloToLhloOpImpl<HloOpTy>::Type;

#define MAP_HLO_TO_LHLO(OpName)          \
  template <>                            \
  struct HloToLhloOpImpl<mhlo::OpName> { \
    using Type = lmhlo::OpName;          \
  }

MAP_HLO_TO_LHLO(AbsOp);
MAP_HLO_TO_LHLO(AddOp);
MAP_HLO_TO_LHLO(AndOp);
MAP_HLO_TO_LHLO(Atan2Op);
MAP_HLO_TO_LHLO(BatchNormGradOp);
MAP_HLO_TO_LHLO(BatchNormTrainingOp);
MAP_HLO_TO_LHLO(BitcastConvertOp);
MAP_HLO_TO_LHLO(BroadcastInDimOp);
MAP_HLO_TO_LHLO(CeilOp);
MAP_HLO_TO_LHLO(ClampOp);
MAP_HLO_TO_LHLO(ConstantOp);
MAP_HLO_TO_LHLO(CompareOp);
MAP_HLO_TO_LHLO(ComplexOp);
MAP_HLO_TO_LHLO(ConcatenateOp);
MAP_HLO_TO_LHLO(ConvOp);
MAP_HLO_TO_LHLO(ConvertOp);
MAP_HLO_TO_LHLO(CopyOp);
MAP_HLO_TO_LHLO(CosOp);
MAP_HLO_TO_LHLO(CustomCallOp);
MAP_HLO_TO_LHLO(DivOp);
MAP_HLO_TO_LHLO(DotOp);
MAP_HLO_TO_LHLO(DynamicBroadcastInDimOp);
MAP_HLO_TO_LHLO(DynamicGatherOp);
MAP_HLO_TO_LHLO(DynamicIotaOp);
MAP_HLO_TO_LHLO(DynamicPadOp);
MAP_HLO_TO_LHLO(DynamicReshapeOp);
MAP_HLO_TO_LHLO(ExpOp);
MAP_HLO_TO_LHLO(Expm1Op);
MAP_HLO_TO_LHLO(FloorOp);
MAP_HLO_TO_LHLO(GatherOp);
MAP_HLO_TO_LHLO(ImagOp);
MAP_HLO_TO_LHLO(IotaOp);
MAP_HLO_TO_LHLO(IsFiniteOp);
MAP_HLO_TO_LHLO(LogOp);
MAP_HLO_TO_LHLO(LogisticOp);
MAP_HLO_TO_LHLO(Log1pOp);
MAP_HLO_TO_LHLO(MaxOp);
MAP_HLO_TO_LHLO(MinOp);
MAP_HLO_TO_LHLO(MulOp);
MAP_HLO_TO_LHLO(NegOp);
MAP_HLO_TO_LHLO(NotOp);
MAP_HLO_TO_LHLO(OrOp);
MAP_HLO_TO_LHLO(PowOp);
MAP_HLO_TO_LHLO(RealDynamicSliceOp);
MAP_HLO_TO_LHLO(RealOp);
MAP_HLO_TO_LHLO(ReduceOp);
MAP_HLO_TO_LHLO(ReduceWindowOp);
MAP_HLO_TO_LHLO(ReshapeOp);
MAP_HLO_TO_LHLO(RemOp);
MAP_HLO_TO_LHLO(RsqrtOp);
MAP_HLO_TO_LHLO(SelectOp);
MAP_HLO_TO_LHLO(ShiftLeftOp);
MAP_HLO_TO_LHLO(ShiftRightArithmeticOp);
MAP_HLO_TO_LHLO(ShiftRightLogicalOp);
MAP_HLO_TO_LHLO(SignOp);
MAP_HLO_TO_LHLO(SinOp);
MAP_HLO_TO_LHLO(SliceOp);
MAP_HLO_TO_LHLO(SqrtOp);
MAP_HLO_TO_LHLO(SubOp);
MAP_HLO_TO_LHLO(TanhOp);
MAP_HLO_TO_LHLO(TransposeOp);
MAP_HLO_TO_LHLO(XorOp);

#undef MAP_HLO_TO_LHLO

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_LHLO_TRANSFORMS_MAP_HLO_TO_LHLO_OP_H
