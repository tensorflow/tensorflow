/* Copyright 2024 The JAX Authors.

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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_GPU_MOSAIC_GPU_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_GPU_MOSAIC_GPU_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"

// Generated definitions.
#include "xla/mosaic/dialect/gpu/mosaic_gpu_dialect.h.inc"  // IWYU pragma: keep
#include "xla/mosaic/dialect/gpu/mosaic_gpu_enums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/mosaic/dialect/gpu/mosaic_gpu_attrdefs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "xla/mosaic/dialect/gpu/mosaic_gpu_types.h.inc"
#define GET_OP_CLASSES
#include "xla/mosaic/dialect/gpu/mosaic_gpu_ops.h.inc"

namespace mosaic_gpu {

using Memref = ::mlir::TypedValue<::mlir::MemRefType>;
using Pointer = ::mlir::TypedValue<::mlir::LLVM::LLVMPointerType>;

constexpr absl::string_view kRuntimeTmaDescriptorInitializerName =
    "mosaic_gpu_init_tma_desc";

template <typename T>
std::string MlirToString(T&& value) {
  std::string result;
  llvm::raw_string_ostream os(result);
  value.print(os);
  return result;
}

// Declares the runtime functions that can be called from the generated code.
void DeclareRuntimeFunctions(mlir::OpBuilder& builder);

// Given a target host pointer, a memref corresponding to the tensor we intend
// to describe, and the shape of the slice we intend to load using the resulting
// TMA descriptor, `InitTmaDescriptor` generates the TMA descriptor
// initialization logic on the host.  The resulting TMA descriptor will be
// stored at `host_pointer_to_descriptor`.
absl::Status InitTmaDescriptor(mlir::OpBuilder& builder,
                               Pointer host_pointer_to_descriptor,
                               Memref gmem_ref,
                               mlir::ArrayRef<int64_t> slice_shape);

}  // namespace mosaic_gpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_GPU_MOSAIC_GPU_H_
