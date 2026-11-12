/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_PRESERVE_REGIONS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_PRESERVE_REGIONS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

// Preserve regions let a frontend keep a subgraph out of the converter's TFL
// optimization pipeline.
//
// A frontend marks the subgraph by wrapping it in a
// `stablehlo.composite "odml.preserve_region"`. The StableHLO pipeline lowers
// the whole module to TFL as usual (the decomposition is an ordinary function
// at that point, so StableHLO-level canonicalization and CSE apply to it like
// to everything else). It then moves every region body into a separate module,
// runs the TFL optimization pipeline on the main module and a fixed, minimal
// pipeline on the region bodies, and inlines the processed bodies back in
// place of their composites before export.
//
// No pass of the main TFL optimization pipeline ever sees a region body, so
// TFL rewrite patterns (fusions, canonicalizations, layout changes) do not
// apply to ops the frontend authored directly as `tfl.*`. The region pipeline
// only fuses Q/DQ ops the frontend authored (and quantizes the bias of ops
// whose other operands are already quantized); it does not quantize float ops.
// Its quantization passes run op folders, so identity ops and ops whose
// operands are all constants can still fold away.

namespace mlir::TFL {

// Name of the composite that marks a preserve region.
inline constexpr llvm::StringLiteral kPreserveRegionCompositeName =
    "odml.preserve_region";

// Returns true if `module` contains at least one preserve-region composite.
bool HasPreserveRegions(ModuleOp module);

// Moves the bodies of the preserve regions in `module` into a new module and
// returns it.
//
// Every call site gets its own copy of the decomposition, under a fresh symbol;
// the composite in `module` is retargeted to a private declaration of that
// symbol, which keeps it valid while the main pipeline runs. In the returned
// module:
//   * constant operands of the composite are cloned into the body, so that
//     rewrites that need constant operands (e.g. quantization of weights) can
//     see them. Only operands defined directly by a constant op are cloned;
//     computed constants stay operands;
//   * preserve regions nested in a region body are inlined into it.
// Private decompositions that are no longer referenced are erased.
//
// Emits an error and fails, without modifying `module`, if a decomposition is
// missing or recursive, if a region body refers to any symbol other than a
// nested preserve region (it would dangle in the returned module), or if a
// decomposition with nested regions is public or also used outside of
// preserve regions.
FailureOr<OwningOpRef<ModuleOp>> SplitPreserveRegions(ModuleOp module);

// Inlines the processed region bodies in `regions` (as produced by
// `SplitPreserveRegions` and transformed since) in place of the composites in
// `module` that refer to them, then erases the placeholder declarations and
// the ops that only fed the regions.
//
// Emits an error and fails if a composite has no body in `regions` or if the
// body's signature does not match the composite's operand and result types,
// which happens if the two pipelines legalized boundary types differently.
// `module` may be partially stitched on failure.
LogicalResult StitchPreserveRegions(ModuleOp module, ModuleOp regions);

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_PRESERVE_REGIONS_H_
