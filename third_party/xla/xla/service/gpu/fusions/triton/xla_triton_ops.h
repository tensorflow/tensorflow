/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_OPS_H_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_OPS_H_

#include "mlir/IR/Attributes.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"  // IWYU pragma: keep
#include "mlir/IR/Dialect.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "mlir/Interfaces/InferTypeOpInterface.h"  // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h"  // IWYU pragma: keep
#include "xla/service/gpu/fusions/triton/xla_triton_dialect.h.inc"  // IWYU pragma: keep
#include "triton/Dialect/Triton/IR/Dialect.h"  // IWYU pragma: keep
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"  // IWYU pragma: keep
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"  // IWYU pragma: keep

namespace mlir::triton::xla {
class SparseDotOp;
}
namespace mlir::OpTrait {
// Template specialization for DotLike<SparseDotOp> to skip verification, which
// would fail because the sparse dot has different shapes and operands.
template <>
class DotLike<triton::xla::SparseDotOp>
    : public TraitBase<triton::xla::SparseDotOp, DotLike> {
 public:
  // TODO (b/350928208) : Add a proper verifier for SparseDotOp.
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};
}  // namespace mlir::OpTrait

#define GET_ATTRDEF_CLASSES
#include "xla/service/gpu/fusions/triton/xla_triton_attrs.h.inc"
#define GET_OP_CLASSES
#include "xla/service/gpu/fusions/triton/xla_triton_ops.h.inc"

#endif  // XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_OPS_H_
