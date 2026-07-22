/* Copyright 2026 The OpenXLA Authors.

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
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/layout.h"  // IWYU pragma: keep
#include "xla/mosaic/dialect/tpu/stringify_util.h"
#include "xla/mosaic/dialect/tpu/tpu_enums.h.inc"
#include "xla/layout.h"  // IWYU pragma: keep

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

DEFINE_ABSL_STRINGIFY_FOR_ENUMS();

struct TpuTilingFlags {
  bool use_x16_large_second_minor = false;
  bool use_x8_large_second_minor = false;
  bool use_x4_large_second_minor = false;
};

std::pair<bool, bool> mightCommunicateBetweenChips(Operation* op);

// Creates a pass that infers the layout of memrefs in the given function.
//
// The `target_shape` can either be
// * 1D -- (lane count) SparseCore tiling; or
// * 2D -- (sublane count, lane count) TensorCore tiling.
std::unique_ptr<OperationPass<func::FuncOp>> createInferMemRefLayoutPass(
    int hardware_generation, absl::Span<const int64_t> target_shape,
    const TpuTilingFlags& tpu_tiling_flags);

#define GEN_PASS_DECL_MOSAICSERDEPASS
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

// Finds the first parent op that has the `tpu.core_type` annotation.
// If no such annotation is found, returns nullptr.
Operation* GetParentOpWithCoreType(Operation& op);

// Determine the core type of the given op based on the `tpu.core_type`
// annotation of its first parent op that has the annotation. If no such
// annotation is found, returns kTc.
CoreType GetCoreTypeOfParentOp(Operation& op);

// Returns the function in the module with the given core type.
absl::StatusOr<func::FuncOp> GetFuncWithCoreType(ModuleOp module,
                                                 CoreType core_type);

// Changes the memory space of the value and propagates it through the program.
LogicalResult specializeMemorySpace(TypedValue<MemRefType> value,
                                    MemorySpace memory_space);

// In Mosaic, we often strip tiled layouts from memrefs, for compatibility with
// vector ops. This functions inverts the layout erasure applied to the value.
MemRefType getMemRefType(Value value);

// Returns the remainder of the given value when divided by the given divisor.
// Returns nullopt if the remainder is not known.
std::optional<int64_t> getRemainder(Value val, int64_t divisor,
                                    int64_t fuel = 128);

// Returns true if `value` is guaranteed to be divisible by `divisor`, false if
// value is known to not be divisible by `divisor`, and nullopt if the
// divisibility is not known.
std::optional<bool> isDivisible(Value value, int64_t divisor,
                                int64_t fuel = 128);

bool isGuaranteedDivisible(Value value, int64_t divisor, int64_t fuel = 128);

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

constexpr std::string_view kLeadingTileRows = "leading_tile_rows";

}  // namespace tpu
}  // namespace mlir

#endif  // XLA_MOSAIC_DIALECT_TPU_TPU_DIALECT_H_
