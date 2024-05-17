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

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/test_passes.h"

namespace mlir {
namespace tf_test {
namespace {

#define GEN_PASS_DEF_RESOURCEANALYZERTESTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/test_passes.h.inc"

class ResourceAnalyzerTestPass
    : public impl::ResourceAnalyzerTestPassBase<ResourceAnalyzerTestPass> {
 public:
  void runOnOperation() override;
};

// A set of values that identifies a resource.
struct ResourceKey {
  StringRef device;
  StringRef container;
  StringRef shared_name;
};

ResourceKey GetResourceKey(TF::VarHandleOp var_handle_op) {
  ResourceKey resource_key;

  if (auto attr = var_handle_op->getAttrOfType<StringAttr>("device")) {
    resource_key.device = attr.getValue();
  }

  resource_key.container = var_handle_op.getContainer();
  resource_key.shared_name = var_handle_op.getSharedName();

  return resource_key;
}

// Prints the analysis result for each resource ops found in `module_op` for
// testing purposes.
void PrintAnalysisResults(const TF::ResourceAnalyzer& analyzer,
                          ModuleOp module_op) {
  module_op.getRegion().walk([&analyzer](Operation* op) {
    if (auto var_handle_op = dyn_cast<TF::VarHandleOp>(op)) {
      const ResourceKey resource_key = GetResourceKey(var_handle_op);
      op->emitRemark(llvm::formatv(
          "device: \"{0}\", container: \"{1}\", shared_name: \"{2}\", "
          "is_potentially_written: {3}",
          resource_key.device, resource_key.container, resource_key.shared_name,
          analyzer.IsPotentiallyWritten(var_handle_op.getResource())));
    }
  });
}

void ResourceAnalyzerTestPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  TF::ResourceAnalyzer resource_analyzer(module_op);

  PrintAnalysisResults(resource_analyzer, module_op);
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateResourceAnalyzerTestPass() {
  return std::make_unique<ResourceAnalyzerTestPass>();
}

}  // namespace tf_test
}  // namespace mlir
