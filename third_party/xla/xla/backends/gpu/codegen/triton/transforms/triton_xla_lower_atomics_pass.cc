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

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWERATOMICSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

absl::string_view GetMemorySemanticStr(triton::MemSemantic semantic) {
  switch (semantic) {
    case triton::MemSemantic::RELAXED:
      return "relaxed";
    case triton::MemSemantic::ACQUIRE:
      return "acquire";
    case triton::MemSemantic::RELEASE:
      return "release";
    case triton::MemSemantic::ACQUIRE_RELEASE:
      return "acq_rel";
  }
}

absl::string_view GetMemSyncScopeStr(triton::MemSyncScope scope) {
  switch (scope) {
    case triton::MemSyncScope::GPU:
      return "gpu";
    case triton::MemSyncScope::SYSTEM:
      return "sys";
    case triton::MemSyncScope::CTA:
      return "cta";
  }
}

absl::string_view GetComparatorStr(Comparator comparator) {
  switch (comparator) {
    case Comparator::EQ:
      return "eq";
    case Comparator::LT:
      return "lt";
  }
}

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

LogicalResult LowerAtomicWriteOp(AtomicWriteOp atomic_write,
                                 PatternRewriter& rewriter) {
  mlir::ImplicitLocOpBuilder builder(atomic_write.getLoc(), rewriter);

  mlir::Value ptr = atomic_write.getPtr();
  mlir::Value value = atomic_write.getValue();
  triton::MemSemantic semantic = atomic_write.getMemSyncSemantic();
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::RELEASE) {
    return rewriter.notifyMatchFailure(
        atomic_write, absl::StrFormat("Unsupported memory semantic: %s",
                                      stringifyMemSemantic(semantic)));
  }
  absl::string_view memory_semantic = GetMemorySemanticStr(semantic);
  absl::string_view scope = GetMemSyncScopeStr(atomic_write.getMemSyncScope());

  // Predicate ASM to check if the mask is set.
  // NB: The arguments start from 1 because 0 is reserved to be the output.
  // Even though we don't care about result($0) in this case it must be there
  // for ElementwiseInlineAsmOp verifiers to work
  constexpr absl::string_view kAtomicWriteAsmWithMaskTemplate = R"(
    {
    .reg .pred %%p<>;
    setp.ne.u32 %%p<>, $3, 0;
    @%%p st.global.%s.%s.u32 [$1], $2;
    }
  )";
  constexpr absl::string_view kAtomicWriteAsmTemplate = R"(
    st.global.%s.%s.u32 [$1], $2;
  )";

  mlir::Type result_type = GetResultType(ptr.getType(), rewriter);
  mlir::Value mask = atomic_write.getMask();
  if (mask) {
    const std::string atomic_write_asm_with_mask = absl::StrFormat(
        kAtomicWriteAsmWithMaskTemplate, scope, memory_semantic);
    triton::ElementwiseInlineAsmOp::create(
        builder,
        /*result_types=*/result_type,
        /*asm_string=*/rewriter.getStringAttr(atomic_write_asm_with_mask),
        /*constraints=*/rewriter.getStringAttr("=r,l,r,r"),
        /*pure=*/rewriter.getBoolAttr(false),
        /*packed_element=*/rewriter.getI32IntegerAttr(1),
        /*args=*/mlir::ValueRange{ptr, value, mask});
  } else {
    const std::string atomic_write_asm =
        absl::StrFormat(kAtomicWriteAsmTemplate, scope, memory_semantic);
    triton::ElementwiseInlineAsmOp::create(
        builder,
        /*result_types=*/result_type,
        /*asm_string=*/rewriter.getStringAttr(atomic_write_asm),
        /*constraints=*/rewriter.getStringAttr("=r,l,r"),
        /*pure=*/rewriter.getBoolAttr(false),
        /*packed_element=*/rewriter.getI32IntegerAttr(1),
        /*args=*/mlir::ValueRange{ptr, value});
  }
  // No results to replace; just erase the op.
  rewriter.eraseOp(atomic_write);
  return success();
}

LogicalResult LowerAtomicSpinWaitOp(AtomicSpinWaitOp atomic_wait,
                                    PatternRewriter& rewriter) {
  mlir::ImplicitLocOpBuilder builder(atomic_wait.getLoc(), rewriter);

  mlir::Value ptr = atomic_wait.getPtr();
  mlir::Value expected = atomic_wait.getExpected();
  triton::MemSemantic semantic = atomic_wait.getMemSyncSemantic();
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::ACQUIRE) {
    return rewriter.notifyMatchFailure(
        atomic_wait, absl::StrFormat("Unsupported memory semantic: %s",
                                     stringifyMemSemantic(semantic)));
  }
  absl::string_view memory_semantic = GetMemorySemanticStr(semantic);
  absl::string_view scope = GetMemSyncScopeStr(atomic_wait.getMemSyncScope());

  absl::string_view comparator = GetComparatorStr(atomic_wait.getComparator());
  constexpr absl::string_view kAtomicSpinWaitAsmTemplate = R"(
    {
    .reg .pred %%p<1>;
    .reg .b32 %%r<1>;
    wait:
      ld.global.%s.%s.u32 %%r0, [$1];
      setp.%s.u32 %%p0, %%r0, $2;
      @%%p0 bra wait;
    }
  )";
  constexpr absl::string_view kAtomicSpinWaitAsmWithMaskTemplate = R"(
    {
    .reg .pred %%p<2>;
    .reg .b32 %%r<1>;
    setp.ne.u32 %%p0, $3, 0;
    @%%!p0 bra done;
    wait:
      ld.global.%s.%s.u32 %%r0, [$1];
      setp.%s.u32 %%p1, %%r0, $2;
      @%%p1 bra wait;
    done:
    }
  )";
  mlir::Type result_type = GetResultType(ptr.getType(), rewriter);
  Value mask = atomic_wait.getMask();
  if (mask) {
    const std::string atomic_wait_asm_with_mask = absl::StrFormat(
        kAtomicSpinWaitAsmWithMaskTemplate, scope, memory_semantic, comparator);
    triton::ElementwiseInlineAsmOp::create(
        builder,
        /*result_types=*/result_type,
        /*asm_string=*/rewriter.getStringAttr(atomic_wait_asm_with_mask),
        /*constraints=*/rewriter.getStringAttr("=r,l,r,r"),
        /*pure=*/rewriter.getBoolAttr(false),
        /*packed_element=*/rewriter.getI32IntegerAttr(1),
        /*args=*/mlir::ValueRange{ptr, expected, mask});
  } else {
    const std::string atomic_wait_asm = absl::StrFormat(
        kAtomicSpinWaitAsmTemplate, scope, memory_semantic, comparator);
    triton::ElementwiseInlineAsmOp::create(
        builder,
        /*result_types=*/result_type,
        /*asm_string=*/rewriter.getStringAttr(atomic_wait_asm),
        /*constraints=*/rewriter.getStringAttr("=r,l,r"),
        /*pure=*/rewriter.getBoolAttr(false),
        /*packed_element=*/rewriter.getI32IntegerAttr(1),
        /*args=*/mlir::ValueRange{ptr, expected});
  }
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

std::unique_ptr<Pass> CreateTritonXLALowerAtomicsPass() {
  return std::make_unique<TritonXLALowerAtomicsPass>();
}

}  // namespace mlir::triton::xla
