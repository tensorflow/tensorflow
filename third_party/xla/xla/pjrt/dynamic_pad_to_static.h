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

#ifndef XLA_PJRT_DYNAMIC_PAD_TO_STATIC_H_
#define XLA_PJRT_DYNAMIC_PAD_TO_STATIC_H_

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "xla/shape.h"

namespace xla {

class HloModule;

// Standard dynamic PJRT buffers prepend a dimension-size header sized by
// `Layout::dynamic_shape_metadata_prefix_bytes()`. `PadRealToStatic` is an
// exception: operand 0 is a compact payload buffer with no prefix, while the
// runtime dimension sizes arrive as separate scalar operands.
//
// Finds dynamic `PadRealToStatic` custom calls across `hlo_module`, traces
// each operand 0 back to its entry parameter leaf across zero-copy tuple
// plumbing, calls, async wrappers, and offloaded subcomputation dispatchers,
// and sets `dynamic_shape_metadata_prefix_bytes(0)` on the matching
// `parameter_shapes` entry for flat or single-tuple entry parameters.
//
// Returns `InvalidArgument` without modifying `parameter_shapes` if operand 0
// comes from a copy or computed value rather than an entry parameter, or if
// that parameter has other readers, is returned, or aliases an output.
absl::Status ClearPadRealToStaticParameterMetadataPrefixes(
    const HloModule& hlo_module, absl::Span<Shape> parameter_shapes);

// Checks the same single-use, unaliased contract on pre-HLO MLIR when async
// compilation infers dispatch info before an `HloModule` exists. Returns true
// when `main` argument `arg_index` feeds `PadRealToStatic` operand 0, and
// `InvalidArgument` if `arg_index` is out of bounds, has other users, or
// carries `tf.aliasing_output`.
absl::StatusOr<bool> ShouldClearPadRealToStaticMetadataPrefix(
    mlir::func::FuncOp main, size_t arg_index);

}  // namespace xla

#endif  // XLA_PJRT_DYNAMIC_PAD_TO_STATIC_H_
