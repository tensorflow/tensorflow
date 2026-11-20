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
#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWER_TENSORS_COMMON_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWER_TENSORS_COMMON_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/codegen/emitters/transforms/atomic_rmw_utils.h"

namespace xla {
namespace emitters {

struct LowerTensorsConfig {
  // Signature lowering policy: CPU allows mixed tensor and non-tensor function
  // results, whereas GPU requires all-or-nothing tensor results.
  bool allow_mixed_tensor_results = false;
  // Sub-byte `tensor.insert` policy: `false` emits a byte load/mask/store
  // sequence (CPU), whereas `true` emits an `xla.atomic_rmw` on an `i8` tensor
  // cast so concurrent threads updating adjacent sub-byte elements synchronize
  // via atomic CAS (GPU).
  bool atomic_sub_byte_inserts = false;
  // Minimum atomic word width in bits for CAS lowering (CPU: 8, GPU: 32 with
  // 4-byte `ptrmask` alignment).
  int64_t min_atomic_bit_width = 32;
  // Maximum supported CAS word width in bits (64, or 128 on Hopper+).
  int64_t max_cas_bit_width = 64;
  // Target LLVM atomic syncscope string ("" default, "agent-one-as" on AMD).
  std::string atomic_sync_scope;
  // LLVM atomic ordering for direct atomic stores and atomic loads (CAS seed
  // loads and pre-check loads). AMDGPU uses `monotonic` so SIMemoryLegalizer
  // applies agent-scope L1 cache bypass; CPU uses `monotonic` so LLVM's
  // LoopIdiomRecognizePass does not form unresolved
  // `__llvm_memcpy_element_unordered_atomic_*` JIT calls; CUDA and Intel GPU
  // use `unordered` so atomic loads remain Ref-only in alias analysis.
  mlir::LLVM::AtomicOrdering atomic_load_store_ordering =
      mlir::LLVM::AtomicOrdering::unordered;
  // Enables direct `llvm.store atomic` and integer `llvm.atomicrmw` lowering
  // for 8/16/32/64-bit scalars and 8-byte-aligned `complex<f32>` (CPU: true;
  // GPU: false, handled by backend patterns).
  bool direct_scalar_atomics = false;
};

// Returns the standard target-neutral configuration for XLA:CPU tensor
// lowering.
LowerTensorsConfig GetCpuLowerTensorsConfig();

// Runs the two-stage tensor-to-LLVM lowering (`tensor_patterns`, unlowered
// atomic verification, and `function_patterns`) on `module`.
mlir::LogicalResult LowerTensors(
    mlir::ModuleOp module, const LowerTensorsConfig& config,
    absl::FunctionRef<void(mlir::RewritePatternSet&)>
        populate_backend_tensor_patterns = [](mlir::RewritePatternSet&) {});

// Traces `dest` backward through tensor update and control-flow forwarding
// operations to find the underlying destination buffer SSA value.
mlir::Value TraceDestinationBuffer(mlir::Value dest);

// Returns the byte alignment proven for element `indices` of `dest`, or 1 when
// no alignment > 1 can be proven.
int64_t GetGuaranteedByteAlignment(
    mlir::TypedValue<mlir::RankedTensorType> dest, mlir::ValueRange indices,
    mlir::Operation* op);

// Returns the packed bit-width (`2` or `4`) if `element_type` is a packed
// sub-byte type, or `std::nullopt` otherwise.
std::optional<int> GetSubByteBitWidth(mlir::Type element_type);

// Maps integer `AtomicRMWKind` values to `mlir::LLVM::AtomicBinOp`, or returns
// `std::nullopt` for floating-point kinds.
std::optional<mlir::LLVM::AtomicBinOp> GetIntegralAtomicBinOp(
    AtomicRMWKind kind);

// Builds an `inbounds` GEP to element `indices` of `tensor`'s traced
// destination buffer.
mlir::LLVM::GEPOp CreateAtomicGep(
    mlir::TypedValue<mlir::RankedTensorType> tensor, mlir::ValueRange indices,
    mlir::ImplicitLocOpBuilder& b);

// Creates a naturally aligned atomic `llvm.load` with `ordering` and optional
// `sync_scope`.
mlir::Value CreateAtomicLoad(mlir::OpBuilder& b, mlir::Location loc,
                             mlir::Type type, mlir::Value addr,
                             mlir::LLVM::AtomicOrdering ordering,
                             absl::string_view sync_scope);

// Bitcasts `value` to `ty`, bridging across `arith.bitcast`, `llvm.bitcast`,
// `complex`, and `vector` types.
mlir::Value CreateBitcast(mlir::ImplicitLocOpBuilder& b, mlir::Value value,
                          mlir::Type ty);

// Creates a private `llvm.mlir.global` array in `module` for `shaped_ty`
// (packing 4-bit elements into bytes when needed).
mlir::LLVM::GlobalOp CreateGlobalOp(mlir::Attribute value,
                                    absl::string_view name_prefix,
                                    mlir::ShapedType shaped_ty,
                                    mlir::ModuleOp module, bool is_constant,
                                    int addr_space,
                                    mlir::ImplicitLocOpBuilder& b);

// Offsets `ptr` by `offset_bytes` via `ptrtoint` + `add` + `inttoptr`.
mlir::Value AddByteOffsetToPointer(mlir::ImplicitLocOpBuilder& b,
                                   mlir::Value ptr, int64_t offset_bytes);

}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWER_TENSORS_COMMON_H_
