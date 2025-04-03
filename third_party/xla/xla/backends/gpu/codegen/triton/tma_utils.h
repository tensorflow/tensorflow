/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TMA_UTILS_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TMA_UTILS_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/shape.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla::gpu {

// Returns a TmaDescriptor for a 2D tensor to be emitted in Triton.
absl::StatusOr<stream_executor::gpu::TmaDescriptor> Create2DTmaDescriptor(
    Shape global_shape, llvm::ArrayRef<int64_t> block_shape,
    mlir::Type element_type);

// Emit a TmaDescriptor for the given argument & tensor type. It can then be
// used to load a tensor using the DescriptorLoadOp.
mlir::Value EmitTmaDescriptor(EmitterLocOpBuilder& b, mlir::Value arg,
                              mlir::RankedTensorType tensor_type);

// Loading arguments by TMA changes the kernel signature and must be updated
// appropriately.
void RewriteFunctionForTma(
    EmitterLocOpBuilder& b, mlir::triton::FuncOp fn,
    std::optional<stream_executor::gpu::TmaMetadata> tma_metadata);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TMA_UTILS_H_
