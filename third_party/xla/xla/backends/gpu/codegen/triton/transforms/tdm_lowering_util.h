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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TRANSFORMS_TDM_LOWERING_UTIL_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TRANSFORMS_TDM_LOWERING_UTIL_H_

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/service/decision.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

// Returns the original dim indices of all size-1 tile dims.
llvm::SmallVector<int64_t> GetSingletonTileDims(
    llvm::ArrayRef<int64_t> tile_sizes,
    llvm::ArrayRef<int64_t> minor_to_major_layout);

// Whether the tile shape is compatible with AMD TDM lowering. Rejects:
//   - dynamic tile sizes or strides
//   - non-unit tile strides
//   - tiles the singleton fold cannot turn into a valid TDM box
//   - tiles whose surviving minor-most dim spans fewer than 2 dwords
//
// Singleton (size-1) tile dims are folded into the base pointer (at any
// position, not just trailing) and dropped from the descriptor.
// TDM requires the descriptor's minor-most dim to have global stride 1. After
// folding, that dim is the minor-most survivor, which keeps stride 1 only if no
// folded singleton was more-minor than it. A batch-major operand [1, M, K]
// keeps K's stride 1 (accept); a batch-minor operand [M, K, 1] makes K inherit
// the batch's non-unit stride (reject).
//
// TDM encodes LDS padding as a power-of-two dword interval from the minor-most
// extent (Triton's TDMUtility.cpp); a single-dword extent underflows that field
// to garbage, so require >= 2 dwords (e.g. f8[.,.,4] = 1 dword is rejected).
//
// Not checked here, deferred to upstream Triton / hardware legalization:
//   - Hardware rank cap (TDM supports ranks 1 to 5).
//   - Per-dim box size limits.
//   - Padding mode compatibility (this pass hardcodes PAD_ZERO).
::xla::Decision CanUseTdm(bool allow_tdm,
                          llvm::ArrayRef<int64_t> original_shape,
                          llvm::ArrayRef<int64_t> tile_sizes,
                          llvm::ArrayRef<int64_t> tile_strides,
                          llvm::ArrayRef<int64_t> minor_to_major_layout,
                          int64_t element_bit_width);

// Wraps CanUseTdm, logging the reason when TDM is declined.
bool TdmAllowed(const ::xla::Decision& decision);

// Operands for a TDM tensor descriptor. Constructed from the full tile, then
// optionally reduced via DropSingletonTileDims.
struct TdmDescriptorOperands {
  TdmDescriptorOperands(Value pointer, llvm::ArrayRef<int64_t> shape,
                        llvm::ArrayRef<int64_t> layout,
                        llvm::ArrayRef<int64_t> sizes, ValueRange offsets);

  // Returns a copy with dims_to_drop (the size-1 tile dims, at any position)
  // folded out: their fixed offset is baked into the base pointer via tt.addptr
  // and they are dropped from every field, with layout renumbered to the
  // reduced rank. Returns *this unchanged if empty. Survivors keep the pre-fold
  // strides (recomputing from the reduced shape would lose the dropped dims'
  // extents).
  TdmDescriptorOperands DropSingletonTileDims(
      ImplicitLocOpBuilder& builder,
      llvm::ArrayRef<int64_t> dims_to_drop) const;

  Value pointer;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> layout;
  llvm::SmallVector<int64_t> strides;
  llvm::SmallVector<int64_t> sizes;
  llvm::SmallVector<Value> offsets;

 private:
  TdmDescriptorOperands() = default;
};

// Builds a tt.make_tensor_descriptor for a contiguous box load/store from the
// given operands, reordered major-to-minor for Triton's TDM legalizer. Padding
// is hardcoded to PAD_ZERO.
MakeTensorDescOp BuildTensorDescriptor(ImplicitLocOpBuilder& builder,
                                       const TdmDescriptorOperands& operands);

}  // namespace mlir::triton::xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TRANSFORMS_TDM_LOWERING_UTIL_H_
