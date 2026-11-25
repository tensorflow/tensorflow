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
#include "mlir/Dialect/Arith/IR/Arith.h"  // IWYU pragma: keep
#include "mlir/Dialect/Complex/IR/Complex.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/SCF/IR/SCF.h"  // IWYU pragma: keep
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // IWYU pragma: keep
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LogicalResult.h"
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h"  // IWYU pragma: keep
#include "xla/codegen/emitters/ir/xla_dialect.h"  // IWYU pragma: keep
#include "xla/codegen/emitters/transforms/lower_tensors_common.h"

namespace xla::cpu {

#define GEN_PASS_DEF_LOWERTENSORSPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {

class LowerTensorsPass : public impl::LowerTensorsPassBase<LowerTensorsPass> {
 public:
  void runOnOperation() override {
    emitters::LowerTensorsConfig config;
    config.allow_mixed_tensor_results = true;
    config.atomic_sub_byte_inserts = false;
    config.min_atomic_bit_width = 8;
    // CPU targets without a native 128 bit CAS call
    // __atomic_compare_exchange_16, which the XLA:CPU JIT does not provide.
    config.max_cas_bit_width = 64;
    // LLVM's LoopIdiomRecognizePass rewrites contiguous loops with unordered
    // atomic loads or stores into @llvm.memcpy.element.unordered.atomic,
    // emitting unresolved __llvm_memcpy_element_unordered_atomic_* calls in the
    // XLA:CPU JIT; monotonic loads and stores block that rewrite with identical
    // codegen.
    config.atomic_load_store_ordering = mlir::LLVM::AtomicOrdering::monotonic;
    config.direct_scalar_atomics = true;
    if (mlir::failed(emitters::LowerTensors(getOperation(), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace xla::cpu
