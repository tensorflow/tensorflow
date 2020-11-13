/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {
namespace {

// InitTextFileToImportTestPass generates a temporary file and run the
// InitTextFileToImportPass for testing purpose.
class InitTextFileToImportTestPass
    : public mlir::PassWrapper<InitTextFileToImportTestPass,
                               OperationPass<ModuleOp>> {
 public:
  explicit InitTextFileToImportTestPass() {}

 private:
  void runOnOperation() override;
};

void InitTextFileToImportTestPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Create a temporary vocab file.
  int fd;
  SmallString<256> filename;
  std::error_code error_code =
      llvm::sys::fs::createTemporaryFile("text", "vocab", fd, filename);
  if (error_code) return signalPassFailure();

  llvm::ToolOutputFile temp_file(filename, fd);
  const char* dictionary_in_lines =
      "apple\n"
      "banana\n"
      "grape";
  temp_file.os() << dictionary_in_lines;
  temp_file.os().flush();

  // Replace filename constant ops to use the temporary file.
  MLIRContext* context = &getContext();

  for (FuncOp func : module.getOps<FuncOp>()) {
    llvm::SmallVector<ConstantOp, 4> constant_ops(func.getOps<ConstantOp>());
    for (auto op : constant_ops) {
      ShapedType shaped_type =
          RankedTensorType::get({1}, StringType::get(context));

      DenseStringElementsAttr attr;
      if (!matchPattern(op.getOperation(), m_Constant(&attr))) {
        continue;
      }

      ArrayRef<StringRef> values = attr.getRawStringData();
      if (values.size() != 1 || values[0] != "%FILE_PLACEHOLDER") {
        continue;
      }

      op.valueAttr(DenseStringElementsAttr::get(shaped_type, {filename}));
    }
  }

  // Run the lowering pass.
  PassManager pm(context);
  pm.addNestedPass<FuncOp>(CreateInitTextFileToImportPass());
  if (failed(pm.run(module))) return signalPassFailure();
}

}  // namespace

static PassRegistration<InitTextFileToImportTestPass> pass(
    "tf-init-text-file-to-import-test",
    "generate a temporary file and invoke InitTextFileToImportPass");

}  // namespace TF
}  // namespace mlir
