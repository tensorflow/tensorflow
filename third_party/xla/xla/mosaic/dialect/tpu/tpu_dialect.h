/* Copyright 2023 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_DIALECT_TPU_DIALECT_H_
#define JAXLIB_MOSAIC_DIALECT_TPU_DIALECT_H_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"  // IWYU pragma: keep
#include "xla/mosaic/dialect/tpu/layout.h"  // IWYU pragma: keep
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

struct TpuTilingFlags {
  bool use_x16_large_second_minor = false;
  bool use_x8_large_second_minor = false;
  bool use_x4_large_second_minor = false;
};

struct ApplyVectorLayoutContext {
  // TODO(tlongeri): target_shape should be determined from hardware_generation
  int hardware_generation = -1;
  std::array<int64_t, 2> target_shape = {8, 128};
  // mxu_shape = {contracting_size, non_contracting_size}
  std::array<int64_t, 2> mxu_shape = {128, 128};
  int64_t max_sublanes_in_scratch = 0;
  int64_t vmem_banks = -1;                  // -1 means "unspecified".
  int32_t max_shuffle_sublane_offset = -1;  // -1 means "unspecified".
};

std::pair<bool, bool> mightCommunicateBetweenChips(Operation *op);

std::unique_ptr<OperationPass<func::FuncOp>> createInferMemRefLayoutPass(
    int hardware_generation = -1,
    std::array<int64_t, 2> target_shape = {8, 128},
    const TpuTilingFlags &tpu_tiling_flags = {});

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeMosaicPass(
    int hardware_generation = -1, bool compatibility_mode = true,
    std::array<int64_t, 2> target_shape = {8, 128});

std::unique_ptr<OperationPass<func::FuncOp>> createInferVectorLayoutPass(
    int hardware_generation = -1,
    std::array<int64_t, 2> target_shape = {8, 128},
    const TpuTilingFlags &tpu_tiling_flags = {});

std::unique_ptr<OperationPass<func::FuncOp>> createRelayoutInsertionPass(
    int hardware_generation = -1,
    std::array<int64_t, 2> target_shape = {8, 128});

std::unique_ptr<OperationPass<func::FuncOp>> createApplyVectorLayoutPass(
    const ApplyVectorLayoutContext &ctx = ApplyVectorLayoutContext{});

std::unique_ptr<OperationPass<func::FuncOp>>
createLogicalToPhysicalDeviceIdPass(int64_t total_devices);

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgVectorizationPass(
    bool supports_bf16_alu_instructions = false,
    bool supports_bf16_matmul = false);

std::unique_ptr<OperationPass<func::FuncOp>> createDebugAssertInsertionPass();

#define GEN_PASS_DECL_MOSAICSERDEPASS
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

// Determine the core type of the given op based on the `tpu.core_type`
// annotation of its parent function. If no such annotation is found, returns
// kTc.
FailureOr<CoreType> GetCoreTypeOfParentFunc(Operation &op);

// Changes the memory space of the value and propagates it through the program.
LogicalResult specializeMemorySpace(TypedValue<MemRefType> value,
                                    MemorySpace memory_space);

// In Mosaic, we often strip tiled layouts from memrefs, for compatibility with
// vector ops. This functions inverts the layout erasure applied to the value.
MemRefType getMemRefType(Value value);

bool isGuaranteedDivisible(Value value, int64_t divisor, int64_t fuel = 8);

DotDimensionNumbersAttr defaultDimensionNumbers(Builder &builder,
                                                bool transpose_lhs,
                                                bool transpose_rhs);

#define GEN_PASS_REGISTRATION
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

}  // namespace tpu
}  // namespace mlir

#endif  // JAXLIB_MOSAIC_DIALECT_TPU_DIALECT_H_
