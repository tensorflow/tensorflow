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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_LAYOUT_UTILS_H_
#define XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_LAYOUT_UTILS_H_

#include <string>

#include "tensor_ir/Dialect/TensorIR.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/shape.h"

namespace xla::gpu::tensor_ir {

// Returns the CuTe stride tuple for `shape`'s layout, e.g. "(16,1)" for a
// row-major 2-D shape with 16 columns. Strides are in dimension order
// (dimension 0 first) and counted in elements, not bytes.
// For rank-0 scalar shapes, this returns "()".
std::string ComputeStrideString(const Shape& shape);

// True when `shape` is an array with the default descending minor-to-major
// layout (or with no explicit layout, which is treated as the default one), in
// which case no stride attribute needs to be attached.
bool HasDefaultLayout(const Shape& shape);

// Attaches `nv_tensor_ir.stride` to every argument and result of `graph` whose
// corresponding shape does not have the default layout. Results correspond to
// the elements of the root tuple, or to the root itself if it is not a tuple.
//
// Strides are intrinsic to `computation`: they are derived purely from the
// layouts of its parameters and its root, with no buffer assignment involved.
// This is therefore part of importing a computation, and is called by
// `ImportAndLegalizeComputation`, so that everyone who imports a computation
// (the emitter, the autotuner, tools) sees the same graph.
//
// `graph` must have exactly one argument per computation parameter and one
// result per computation result; this mirrors what the importer produces.
absl::Status AttachLayoutStrides(mlir::nv_tensor_ir::GraphOp graph,
                                 const HloComputation& computation);

// Attaches `nv_tensor_ir.alignment` to every argument and result of `graph`.
//
// Alignments come from buffer assignment, so unlike strides they are a
// property of the buffers the computation happens to be assigned rather than
// of the computation itself. This can only run in the emitter, after kernel
// arguments exist.
//
// `kernel_args` must hold one argument per graph argument followed by one per
// graph result.
absl::Status AttachBufferAlignments(
    mlir::nv_tensor_ir::GraphOp graph,
    absl::Span<const emitters::KernelArgument> kernel_args);

}  // namespace xla::gpu::tensor_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_LAYOUT_UTILS_H_
