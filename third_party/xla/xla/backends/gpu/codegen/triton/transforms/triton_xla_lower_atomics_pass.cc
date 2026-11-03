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

// Generic lowering of atomic operations for Triton XLA using
// tt.extern_elementwise. This implementation uses tt.extern_elementwise to call
// custom atomic functions that will be implemented in platform-specific passes
// later in the Triton pipeline.

#include <string>
#include <utility>

#include "absl/log/log.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/extern_function_helper.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWERATOMICSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

mlir::Type GetResultType(mlir::Type ptr_type, PatternRewriter& rewriter) {
  mlir::Type result_type = rewriter.getI32Type();
  auto ranked_tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(ptr_type);
  // Tensor arguments must have tensor result type.
  if (ranked_tensor_type) {
    result_type = mlir::RankedTensorType::get(ranked_tensor_type.getShape(),
                                              rewriter.getI32Type());
  }
  return result_type;
}

// tt.extern_elementwise requires all operands to have the same shape: when
// lowered to LLVM, each operand is unpacked into per-thread elements, and a
// scalar operand contributes only one element. Splat scalar operands to the
// pointer tensor shape so that every thread gets one value per pointer.
mlir::Value SplatToPtrShape(mlir::ImplicitLocOpBuilder& builder,
                            mlir::Value scalar, mlir::Type ptr_type) {
  auto ptr_tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(ptr_type);
  if (!ptr_tensor_type || mlir::isa<mlir::RankedTensorType>(scalar.getType())) {
    return scalar;
  }
  return triton::SplatOp::create(
      builder,
      mlir::RankedTensorType::get(ptr_tensor_type.getShape(), scalar.getType()),
      scalar);
}

// Lower AtomicWriteOp to tt.extern_elementwise.
// This creates extern calls that will be implemented in a separate ROCm pass.
LogicalResult LowerAtomicWriteOp(AtomicWriteOp atomic_write,
                                 PatternRewriter& rewriter) {
  VLOG(3) << "LowerAtomicWriteOp: Starting tt.extern_elementwise lowering";
  mlir::ImplicitLocOpBuilder builder(atomic_write.getLoc(), rewriter);

  mlir::Value ptr = atomic_write.getPtr();
  mlir::Value value = atomic_write.getValue();
  mlir::Value mask = atomic_write.getMask();

  // Validate memory semantics
  triton::MemSemantic semantic = atomic_write.getMemSyncSemantic();
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::RELEASE) {
    return rewriter.notifyMatchFailure(
        atomic_write,
        "AtomicWriteOp only supports RELAXED or RELEASE semantics");
  }

  // Build function name using helper
  AtomicWriteInstruction instruction{
      /* .semantic= */ semantic,
      /* .scope= */ atomic_write.getMemSyncScope(),
      /* .has_mask= */ mask ? true : false,
  };
  std::string func_name = SerializeExternFunctionName(instruction);
  VLOG(3) << "LowerAtomicWriteOp: Creating extern_elementwise call to "
          << func_name;
  // Get result type (handles both tensor and scalar pointers)
  mlir::Type result_type = GetResultType(ptr.getType(), rewriter);
  // Prepare operands: ptr (tensor or scalar), value splat to ptr's shape.
  llvm::SmallVector<mlir::Value> operands = {
      ptr, SplatToPtrShape(builder, value, ptr.getType())};

  // If mask is provided, pass it as third argument
  if (mask) {
    operands.push_back(mask);
  }

  // Create tt.extern_elementwise call
  // The function will perform atomic exchange and return the old value
  builder.create<triton::ExternElementwiseOp>(
      /*resultType=*/result_type,
      /*srcs=*/operands,
      /*libname=*/"",
      /*libpath=*/"",
      /*symbol=*/func_name,
      /*pure=*/false);

  rewriter.eraseOp(atomic_write);
  return success();
}

// Lower AtomicSpinWaitOp to tt.extern_elementwise.
// This creates extern calls that will be implemented in a separate ROCm pass.
LogicalResult LowerAtomicSpinWaitOp(AtomicSpinWaitOp atomic_wait,
                                    PatternRewriter& rewriter) {
  VLOG(3) << "LowerAtomicSpinWaitOp: Starting tt.extern_elementwise lowering";
  mlir::ImplicitLocOpBuilder builder(atomic_wait.getLoc(), rewriter);

  mlir::Value ptr = atomic_wait.getPtr();
  mlir::Value expected = atomic_wait.getExpected();
  mlir::Value mask = atomic_wait.getMask();

  // Validate memory semantics
  triton::MemSemantic semantic = atomic_wait.getMemSyncSemantic();
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::ACQUIRE) {
    return rewriter.notifyMatchFailure(
        atomic_wait,
        "AtomicSpinWaitOp only supports RELAXED or ACQUIRE semantics");
  }

  // Build function name using helper
  AtomicSpinWaitInstruction instruction{
      /* .semantic= */ semantic,
      /* .scope= */ atomic_wait.getMemSyncScope(),
      /* .comparator= */ atomic_wait.getComparator(),
      /* .has_mask= */ mask ? true : false,
  };
  std::string func_name = SerializeExternFunctionName(instruction);

  VLOG(3) << "LowerAtomicSpinWaitOp: Creating extern_elementwise call to "
          << func_name;

  // Get result type (handles both tensor and scalar pointers)
  mlir::Type result_type = GetResultType(ptr.getType(), rewriter);

  // Prepare operands: ptr (tensor or scalar), expected splat to ptr's shape.
  llvm::SmallVector<mlir::Value> operands = {
      ptr, SplatToPtrShape(builder, expected, ptr.getType())};

  // If mask is provided, pass it as third argument
  if (mask) {
    operands.push_back(mask);
  }

  // Create tt.extern_elementwise call
  // The function will spin-wait until the condition is met
  builder.create<triton::ExternElementwiseOp>(
      /*resultType=*/result_type,
      /*srcs=*/operands,
      /*libname=*/"",
      /*libpath=*/"",
      /*symbol=*/func_name,
      /*pure=*/false);

  rewriter.eraseOp(atomic_wait);
  return success();
}

class TritonXLALowerAtomicsPass
    : public impl::TritonXLALowerAtomicsPassBase<TritonXLALowerAtomicsPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerAtomicWriteOp);
    patterns.add(LowerAtomicSpinWaitOp);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::triton::xla
