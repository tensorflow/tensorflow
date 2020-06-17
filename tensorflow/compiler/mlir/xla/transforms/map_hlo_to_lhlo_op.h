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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_HLO_TO_LHLO_OP_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_HLO_TO_LHLO_OP_H_

#include <type_traits>

#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"

namespace mlir {
namespace xla_hlo {

template <typename HloOpTy>
struct HloToLhloOpImpl {
  using Type = std::false_type;
};
template <typename HloOpTy>
using HloToLhloOp = typename HloToLhloOpImpl<HloOpTy>::Type;

#define MAP_HLO_TO_LHLO(OpName)             \
  template <>                               \
  struct HloToLhloOpImpl<xla_hlo::OpName> { \
    using Type = xla_lhlo::OpName;          \
  }

MAP_HLO_TO_LHLO(AbsOp);
MAP_HLO_TO_LHLO(AddOp);
MAP_HLO_TO_LHLO(AndOp);
MAP_HLO_TO_LHLO(BroadcastInDimOp);
MAP_HLO_TO_LHLO(CeilOp);
MAP_HLO_TO_LHLO(ConstOp);
MAP_HLO_TO_LHLO(CompareOp);
MAP_HLO_TO_LHLO(ComplexOp);
MAP_HLO_TO_LHLO(ConvOp);
MAP_HLO_TO_LHLO(ConvertOp);
MAP_HLO_TO_LHLO(CopyOp);
MAP_HLO_TO_LHLO(CosOp);
MAP_HLO_TO_LHLO(DivOp);
MAP_HLO_TO_LHLO(DotOp);
MAP_HLO_TO_LHLO(ExpOp);
MAP_HLO_TO_LHLO(ImagOp);
MAP_HLO_TO_LHLO(IotaOp);
MAP_HLO_TO_LHLO(LogOp);
MAP_HLO_TO_LHLO(MaxOp);
MAP_HLO_TO_LHLO(MinOp);
MAP_HLO_TO_LHLO(MulOp);
MAP_HLO_TO_LHLO(NegOp);
MAP_HLO_TO_LHLO(RealOp);
MAP_HLO_TO_LHLO(ReduceOp);
MAP_HLO_TO_LHLO(RemOp);
MAP_HLO_TO_LHLO(RsqrtOp);
MAP_HLO_TO_LHLO(SelectOp);
MAP_HLO_TO_LHLO(SignOp);
MAP_HLO_TO_LHLO(SinOp);
MAP_HLO_TO_LHLO(SqrtOp);
MAP_HLO_TO_LHLO(SubOp);
MAP_HLO_TO_LHLO(TanhOp);

#undef MAP_HLO_TO_LHLO

}  // namespace xla_hlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_HLO_TO_LHLO_OP_H_
