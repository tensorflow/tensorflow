/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <cassert>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/transforms/memref/passes.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/xla/mlir/transforms/memref/passes.h.inc"

struct AlignedAllocationsPass
    : public AlignedAllocationsPassBase<AlignedAllocationsPass> {
  explicit AlignedAllocationsPass(int64_t alignment) { alignment_ = alignment; }
  void runOnOperation() override;
};

void AlignedAllocationsPass::runOnOperation() {
  assert(alignment_ >= 0 && "alignment must be larger or equal to 0");
  if (alignment_ == 0) return;

  auto i64 = IntegerType::get(&getContext(), 64);
  auto alignment_attr = IntegerAttr::get(i64, alignment_);

  getOperation().walk([&](memref::AllocOp alloc) {
    // Add alignment attribute only if the alignment attribute is missing or the
    // current alignment is smaller.
    if (!alloc.alignment().has_value() || *alloc.alignment() < alignment_)
      alloc.alignmentAttr(alignment_attr);
  });
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateAlignedAllocationsPass(
    int64_t alignment) {
  return std::make_unique<AlignedAllocationsPass>(alignment);
}

}  // namespace runtime
}  // namespace xla
