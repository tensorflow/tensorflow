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

#ifndef MLIR_HLO_DIALECT_LHLO_TRANSFORMS_MAP_LHLO_TO_HLO_OP_H
#define MLIR_HLO_DIALECT_LHLO_TRANSFORMS_MAP_LHLO_TO_HLO_OP_H

#include <type_traits>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace lmhlo {

template <typename LhloOpTy>
struct LhloToHloOpImpl {
  using Type = std::false_type;
};
template <typename LhloOpTy>
using LhloToHloOp = typename LhloToHloOpImpl<LhloOpTy>::Type;

#define MAP_LHLO_TO_HLO(OpName)           \
  template <>                             \
  struct LhloToHloOpImpl<lmhlo::OpName> { \
    using Type = mhlo::OpName;            \
  }

MAP_LHLO_TO_HLO(AbsOp);
MAP_LHLO_TO_HLO(AddOp);
MAP_LHLO_TO_HLO(AndOp);
MAP_LHLO_TO_HLO(Atan2Op);
MAP_LHLO_TO_HLO(BitcastConvertOp);
MAP_LHLO_TO_HLO(BroadcastInDimOp);
MAP_LHLO_TO_HLO(CeilOp);
MAP_LHLO_TO_HLO(ClampOp);
MAP_LHLO_TO_HLO(ConstantOp);
MAP_LHLO_TO_HLO(CompareOp);
MAP_LHLO_TO_HLO(ComplexOp);
MAP_LHLO_TO_HLO(ConcatenateOp);
MAP_LHLO_TO_HLO(ConvolutionOp);
MAP_LHLO_TO_HLO(ConvertOp);
MAP_LHLO_TO_HLO(CopyOp);
MAP_LHLO_TO_HLO(CosOp);
MAP_LHLO_TO_HLO(CustomCallOp);
MAP_LHLO_TO_HLO(DivOp);
MAP_LHLO_TO_HLO(DotOp);
MAP_LHLO_TO_HLO(DynamicBroadcastInDimOp);
MAP_LHLO_TO_HLO(DynamicGatherOp);
MAP_LHLO_TO_HLO(DynamicIotaOp);
MAP_LHLO_TO_HLO(DynamicPadOp);
MAP_LHLO_TO_HLO(DynamicReshapeOp);
MAP_LHLO_TO_HLO(ExpOp);
MAP_LHLO_TO_HLO(Expm1Op);
MAP_LHLO_TO_HLO(FloorOp);
MAP_LHLO_TO_HLO(GatherOp);
MAP_LHLO_TO_HLO(ImagOp);
MAP_LHLO_TO_HLO(IotaOp);
MAP_LHLO_TO_HLO(IsFiniteOp);
MAP_LHLO_TO_HLO(LogOp);
MAP_LHLO_TO_HLO(LogisticOp);
MAP_LHLO_TO_HLO(Log1pOp);
MAP_LHLO_TO_HLO(MaxOp);
MAP_LHLO_TO_HLO(MinOp);
MAP_LHLO_TO_HLO(MulOp);
MAP_LHLO_TO_HLO(NegOp);
MAP_LHLO_TO_HLO(NotOp);
MAP_LHLO_TO_HLO(OrOp);
MAP_LHLO_TO_HLO(PowOp);
MAP_LHLO_TO_HLO(RealDynamicSliceOp);
MAP_LHLO_TO_HLO(RealOp);
MAP_LHLO_TO_HLO(ReduceOp);
MAP_LHLO_TO_HLO(ReshapeOp);
MAP_LHLO_TO_HLO(RemOp);
MAP_LHLO_TO_HLO(RsqrtOp);
MAP_LHLO_TO_HLO(SelectOp);
MAP_LHLO_TO_HLO(ShiftLeftOp);
MAP_LHLO_TO_HLO(ShiftRightArithmeticOp);
MAP_LHLO_TO_HLO(ShiftRightLogicalOp);
MAP_LHLO_TO_HLO(SignOp);
MAP_LHLO_TO_HLO(SinOp);
MAP_LHLO_TO_HLO(SliceOp);
MAP_LHLO_TO_HLO(SqrtOp);
MAP_LHLO_TO_HLO(SubOp);
MAP_LHLO_TO_HLO(TanhOp);
MAP_LHLO_TO_HLO(TransposeOp);
MAP_LHLO_TO_HLO(XorOp);

#undef MAP_LHLO_TO_HLO

}  // namespace lmhlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_LHLO_TRANSFORMS_MAP_LHLO_TO_HLO_OP_H
