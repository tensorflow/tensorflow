/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/odml_converter/folders.h"

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_SHLOSIMPLIFYPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/odml_converter/passes.h.inc"

// Performs misc odml "cleanup" on shlo dialect. This is a functional standin
// for canonicalization and folding which is not offered directly by the
// shlo implementation.
class SHLOSimplifyPass : public impl::SHLOSimplifyPassBase<SHLOSimplifyPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SHLOSimplifyPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    PopulateFolderPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateSHLOSimplifyPass() {
  return std::make_unique<SHLOSimplifyPass>();
}

}  // namespace odml
}  // namespace mlir
