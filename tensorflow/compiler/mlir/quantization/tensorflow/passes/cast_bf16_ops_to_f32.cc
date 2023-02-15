/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "absl/algorithm/container.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/remove_identity_op_pattern.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

class CastBf16OpsToF32Pass
    : public PassWrapper<CastBf16OpsToF32Pass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CastBf16OpsToF32Pass)
  explicit CastBf16OpsToF32Pass() = default;

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-cast-bf16-ops-to-f32";
  }
  StringRef getDescription() const final {
    return "Cast BF16 operations to F32.";
  }

  void runOnOperation() override;
};

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/cast_bf16_ops_to_f32.inc"

void CastBf16OpsToF32Pass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  auto module_op = getOperation();

  patterns.add<RemoveIdentity>(ctx);
  populateWithGenerated(patterns);

  if (failed(applyPatternsAndFoldGreedily(module_op, std::move(patterns)))) {
    module_op.emitError() << "quant-internal-cast-bf16-to-f32 failed.";
    signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the Cast BF16 ops to F32 pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateCastBf16OpsToF32Pass() {
  return std::make_unique<CastBf16OpsToF32Pass>();
}

static PassRegistration<CastBf16OpsToF32Pass> pass;

}  // namespace quant
}  // namespace mlir
