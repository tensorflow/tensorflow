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
#include <cstdint>
#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"  // IWYU pragma: keep
#include "mlir/Dialect/Complex/IR/Complex.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/LLVMMemoryOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"  // IWYU pragma: keep
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // IWYU pragma: keep
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"  // IWYU pragma: keep
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/lower_tensors_common.h"

namespace xla::cpu {

#define GEN_PASS_DEF_LOWERTENSORSPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {

class RewriteCpuDirect128BitAtomicStore
    : public mlir::OpRewritePattern<AtomicStoreOp> {
 public:
  explicit RewriteCpuDirect128BitAtomicStore(mlir::MLIRContext* context)
      : mlir::OpRewritePattern<AtomicStoreOp>(context, /*benefit=*/2) {}

  mlir::LogicalResult matchAndRewrite(
      AtomicStoreOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getIndices().size() > 1 || op.getDest() == op.getResult()) {
      return rewriter.notifyMatchFailure(
          op, "only acyclic 0D and 1D tensors are supported");
    }
    if (!emitters::TraceDestinationBuffer(op.getDest())) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot trace the destination buffer");
    }
    mlir::Type element_type = op.getValue().getType();
    bool is_128_bit = false;
    if (auto complex_ty = mlir::dyn_cast<mlir::ComplexType>(element_type)) {
      is_128_bit = complex_ty.getElementType().getIntOrFloatBitWidth() == 64;
    } else if (element_type.isIntOrFloat()) {
      is_128_bit = element_type.getIntOrFloatBitWidth() == 128;
    }
    if (!is_128_bit) {
      return rewriter.notifyMatchFailure(
          op, "requires a 128 bit scalar or complex element");
    }
    const int64_t alignment =
        emitters::GetGuaranteedByteAlignment(op.getDest(), op.getIndices(), op);
    if (alignment < 16) {
      return rewriter.notifyMatchFailure(
          op, "requires at least 16 byte destination alignment");
    }

    mlir::Location loc = op.getLoc();
    mlir::ImplicitLocOpBuilder b(loc, rewriter);
    mlir::Value addr =
        emitters::CreateAtomicGep(op.getDest(), op.getIndices(), b);
    mlir::Value store_value =
        emitters::CreateBitcast(b, op.getValue(), b.getIntegerType(128));
    mlir::LLVM::StoreOp::create(rewriter, loc, store_value, addr,
                                /*alignment=*/static_cast<unsigned>(alignment),
                                /*isVolatile=*/false, /*isNonTemporal=*/false,
                                /*isInvariantGroup=*/false,
                                mlir::LLVM::AtomicOrdering::monotonic);
    rewriter.replaceOp(op, op.getDest());
    return mlir::success();
  }
};

class LowerTensorsPass : public impl::LowerTensorsPassBase<LowerTensorsPass> {
 public:
  using LowerTensorsPassBase::LowerTensorsPassBase;

  void runOnOperation() override {
    emitters::LowerTensorsConfig config;
    config.allow_mixed_tensor_results = true;
    config.atomic_sub_byte_inserts = false;
    config.min_atomic_bit_width = 8;
    // CPU targets without a native 128 bit CAS call
    // __atomic_compare_exchange_16, which the XLA:CPU JIT does not provide.
    config.max_cas_bit_width = has_atomic_cmpxchg_128_ ? 128 : 64;
    // LLVM's LoopIdiomRecognizePass rewrites contiguous loops with unordered
    // atomic loads or stores into @llvm.memcpy.element.unordered.atomic,
    // emitting unresolved __llvm_memcpy_element_unordered_atomic_* calls in the
    // XLA:CPU JIT; monotonic loads and stores block that rewrite with identical
    // codegen.
    config.atomic_load_store_ordering = mlir::LLVM::AtomicOrdering::monotonic;
    config.direct_scalar_atomics = true;
    const bool direct_128_bit_store = has_atomic_load_store_128_;
    if (mlir::failed(emitters::LowerTensors(
            getOperation(), config, [&](mlir::RewritePatternSet& patterns) {
              if (direct_128_bit_store) {
                patterns.add<RewriteCpuDirect128BitAtomicStore>(
                    patterns.getContext());
              }
            }))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLowerTensorsPass(
    const CpuDeviceSpec& cpu_device_spec) {
  LowerTensorsPassOptions options;
  options.has_atomic_cmpxchg_128_ = cpu_device_spec.has_atomic_cmpxchg_128;
  options.has_atomic_load_store_128_ =
      cpu_device_spec.has_atomic_load_store_128;
  return createLowerTensorsPass(options);
}

}  // namespace xla::cpu
