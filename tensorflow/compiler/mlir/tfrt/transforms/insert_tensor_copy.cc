/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback.h"
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback_async.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime
#include "tfrt/compiler/stream_analysis.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_compiler {
namespace {

// This pass inserts copy kernels for fallback tensors when they are passed to
// multiple threads, to avoid atomic contention on their refcounts.
class InsertFallbackTensorCopy
    : public mlir::PassWrapper<InsertFallbackTensorCopy,
                               mlir::OperationPass<mlir::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<tfrt::fallback_async::FallbackAsyncDialect>();
  }

  llvm::StringRef getArgument() const final {
    return "tfrt-insert-fallback-tensor-copy";
  }

  llvm::StringRef getDescription() const final {
    return "Inserts copy kernels for fallback tensors when they are passed to "
           "multiple threads, to avoid atomic contention on refcounts.";
  }

 public:
  void runOnOperation() override {
    mlir::FuncOp func_op = getOperation();

    // Use stream analysis to know whether a value is passed to different
    // threads.
    tfrt::compiler::StreamAnalysis stream_analysis(func_op);

    auto builder = mlir::OpBuilder::atBlockBegin(&func_op.front());

    // Process function arguments first.
    for (auto arg : func_op.getArguments()) {
      if (!arg.getType().isa<tfrt::fallback::TFTensorType>()) continue;
      InsertFallbackTensorCopyForValue(arg, func_op->getLoc(), builder,
                                       stream_analysis);
    }

    // Then process each operations in the block.
    for (mlir::Operation& op : llvm::make_early_inc_range(func_op.front())) {
      if (llvm::isa<tfrt::fallback_async::ExecuteOp,
                    tfrt::fallback_async::ExecuteOpSeq>(&op)) {
        InsertFallbackTensorCopyForFallbackOp(&op, builder, stream_analysis);
      }
    }
  }

 private:
  void InsertFallbackTensorCopyForFallbackOp(
      mlir::Operation* op, mlir::OpBuilder& builder,
      const tfrt::compiler::StreamAnalysis& stream_analysis) {
    builder.setInsertionPointAfter(op);

    // Process each result value.
    for (auto result : op->getResults()) {
      if (!result.getType().isa<tfrt::fallback::TFTensorType>()) continue;
      InsertFallbackTensorCopyForValue(result, op->getLoc(), builder,
                                       stream_analysis);
    }
  }

  // Insert copy kernels to copy the result, and allocate new atomic refcount
  // if the value is going to be used by different streams/threads, in order to
  // avoid contention on the atomic counter.
  void InsertFallbackTensorCopyForValue(
      mlir::Value value, mlir::Location loc, mlir::OpBuilder& builder,
      const tfrt::compiler::StreamAnalysis& stream_analysis) {
    llvm::DenseMap<int, llvm::SmallVector<mlir::OpOperand*, 4>> stream_map;

    // Find out streams that use this value and the corresponding uses.
    for (mlir::OpOperand& use : value.getUses()) {
      // Skip return op as there should not be atomic contention on the return
      // op.
      if (llvm::isa<tfrt::compiler::ReturnOp>(use.getOwner())) continue;

      int stream_id = stream_analysis.GetStream(use.getOwner()).id();
      stream_map[stream_id].push_back(&use);
    }

    // Organize these uses into groups. If a stream has many uses of this value,
    // put these uses into one stream. Otherwise, streams with small number
    // of uses are grouped with each other to form groups with enough uses.
    constexpr int kCopyGroupThreshold = 16;
    llvm::SmallVector<llvm::SmallVector<mlir::OpOperand*, 4>, 4> small_copies;
    llvm::SmallVector<llvm::SmallVector<mlir::OpOperand*, 4>, 4> copies;
    for (const auto& iter : stream_map) {
      if (iter.second.size() >= kCopyGroupThreshold) {
        copies.push_back(iter.second);
      } else {
        if (small_copies.empty() ||
            small_copies.back().size() >= kCopyGroupThreshold) {
          small_copies.push_back(iter.second);
        } else {
          small_copies.back().append(iter.second.begin(), iter.second.end());
        }
      }
    }

    if (!small_copies.empty())
      copies.append(small_copies.begin(), small_copies.end());

    // If it is only used by one group, then we don't need to copy.
    if (copies.size() <= 1) return;

    // Remove one group from the candidates, as we can just use the original
    // value for this group.
    copies.pop_back();

    // For each stream, we will create one new value that replaces the uses in
    // that stream.

    assert(value.getType().isa<tfrt::fallback::TFTensorType>());

    // The number of results is the number candidate streams.
    llvm::SmallVector<mlir::Type, 4> result_types(copies.size(),
                                                  value.getType());
    assert(!result_types.empty());

    // Create the tfrt_fallback_async.copy_if_small kernel.
    auto copy_op = builder.create<tfrt::fallback_async::CopyIfSmallOp>(
        loc, result_types, value);

    // Finally, replaces all uses with the new value.
    for (int i = 0; i < copies.size(); ++i) {
      const auto& uses = copies[i];
      auto new_value = copy_op.getResult(i);
      for (auto* use : uses) {
        use->set(new_value);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
CreateInsertFallbackTensorCopyPass() {
  return std::make_unique<InsertFallbackTensorCopy>();
}

static mlir::PassRegistration<InsertFallbackTensorCopy> register_pass(
    CreateInsertFallbackTensorCopyPass);

}  // namespace tfrt_compiler
}  // namespace tensorflow
