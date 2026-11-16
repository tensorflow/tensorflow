/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_MOSAIC_DIALECT_TPU_TPU_DIALECT_H_
#define XLA_MOSAIC_DIALECT_TPU_TPU_DIALECT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"  // IWYU pragma: keep
#include "xla/mosaic/dialect/tpu/stringify_util.h"
#include "xla/mosaic/dialect/tpu/tpu_enums.h.inc"

namespace mlir::tpu {
class TPUDialect;
}  // namespace mlir::tpu

#define GET_ATTRDEF_CLASSES
#include "xla/mosaic/dialect/tpu/tpu_attr_defs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "xla/mosaic/dialect/tpu/tpu_type_defs.h.inc"

#define GET_OP_CLASSES
#include "xla/mosaic/dialect/tpu/tpu_dialect.h.inc"
#include "xla/mosaic/dialect/tpu/tpu_ops.h.inc"

namespace mlir {
namespace tpu {

inline constexpr StringRef kTpuVectorSubcoreGlobalBarrierName =
    "__tpu_vector_subcore_global_barrier";
inline constexpr StringRef kTpuScalarSubcoreGlobalBarrierName =
    "__tpu_scalar_subcore_global_barrier";

DEFINE_ABSL_STRINGIFY_FOR_ENUMS();

std::pair<bool, bool> mightCommunicateBetweenChips(Operation* op);

#define GEN_PASS_DECL_MOSAICSERDEPASS
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

// Finds the first parent op that has the `tpu.core_type` annotation.
// If no such annotation is found, returns nullptr.
Operation* GetParentOpWithCoreType(Operation& op);

// Determine the core type of the given op based on the `tpu.core_type`
// annotation of its first parent op that has the annotation. If no such
// annotation is found, returns kTc.
CoreType GetCoreTypeOfParentOp(Operation& op);

// Returns the main function in the module for the given core type.
absl::StatusOr<func::FuncOp> GetMainFunc(ModuleOp module, CoreType core_type);

// In Mosaic, we often strip tiled layouts from memrefs, for compatibility with
// vector ops. This functions inverts the layout erasure applied to the value.
MemRefType getMemRefType(Value value);

DotDimensionNumbersAttr defaultDimensionNumbers(Builder& builder,
                                                bool transpose_lhs,
                                                bool transpose_rhs);

// True if source represents a shared memory space and target represents a local
// memory space. TC VMEM is "shared" here but it is not shared between TCs.
FailureOr<bool> isGather(Operation& op, MemorySpace source_memory_space,
                         std::optional<CoreType> source_core_type,
                         MemorySpace target_memory_space,
                         std::optional<CoreType> target_core_type);

LogicalResult verifyGather(Operation* op, ArrayRef<int64_t> operand_shape,
                           ArrayRef<int64_t> offsets_shape,
                           ArrayRef<int64_t> result_shape);

LogicalResult verifyScatter(Operation* op, ArrayRef<int64_t> updates_shape,
                            ArrayRef<int64_t> offsets_shape,
                            ArrayRef<int64_t> operand_shape);

#define GEN_PASS_REGISTRATION
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

}  // namespace tpu
}  // namespace mlir

#endif  // XLA_MOSAIC_DIALECT_TPU_TPU_DIALECT_H_
