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

#include <memory>

#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/test_passes_detail.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace mlir {
namespace TF {
namespace {

// InitTextFileToImportTestPass generates a temporary file and run the
// InitTextFileToImportPass for testing purpose.
class InitTextFileToImportTestPass
    : public tf_test::InitTextFileToImportTestPassBase<
          InitTextFileToImportTestPass> {
 public:
  explicit InitTextFileToImportTestPass() {}

  StringRef getArgument() const final {
    return "tf-init-text-file-to-import-test";
  }

  StringRef getDescription() const final {
    return "generate a temporary file and invoke InitTextFileToImportPass";
  }

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
  temp_file.os() << "apple\n";
  temp_file.os() << "banana\n";
  temp_file.os() << "grape";
  temp_file.os().flush();

  // Replace filename constant ops to use the temporary file.
  MLIRContext* context = &getContext();

  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    llvm::SmallVector<arith::ConstantOp, 4> constant_ops(
        func.getOps<arith::ConstantOp>());
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

      op.setValueAttr(DenseStringElementsAttr::get(shaped_type, {filename}));
    }
  }

  // Run the lowering pass.
  PassManager pm(context);
  pm.addNestedPass<func::FuncOp>(CreateInitTextFileToImportPass(""));
  if (failed(pm.run(module))) return signalPassFailure();
}

// InitTextFileToImportSavedModelTestPass mimicks a temporary saved model and
// run the InitTextFileToImportPass for testing purpose.
class InitTextFileToImportSavedModelTestPass
    : public tf_test::InitTextFileToImportSavedModelTestPassBase<
          InitTextFileToImportSavedModelTestPass> {
 public:
  explicit InitTextFileToImportSavedModelTestPass() {}

 private:
  void runOnOperation() override;
};

void InitTextFileToImportSavedModelTestPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Create a temporary saved model's asset file.
  SmallString<256> tempdir;
  std::error_code error_code =
      llvm::sys::fs::createUniqueDirectory("saved-model", tempdir);
  if (error_code) return signalPassFailure();
  error_code =
      llvm::sys::fs::create_directories(Twine(tempdir) + "/assets", false);
  if (error_code) return signalPassFailure();

  std::string filename = std::string(tempdir) + "/assets/tokens.txt";

  std::string error_message;
  auto temp_file = openOutputFile(filename, &error_message);
  if (!error_message.empty()) return;
  temp_file->os() << "apple\n";
  temp_file->os() << "banana\n";
  temp_file->os() << "grape";
  temp_file->os().flush();

  // Run the lowering pass.
  MLIRContext* context = &getContext();
  PassManager pm(context);
  pm.addNestedPass<func::FuncOp>(
      CreateInitTextFileToImportPass(std::string(tempdir)));
  if (failed(pm.run(module))) return signalPassFailure();
}

}  // namespace
}  // namespace TF

namespace tf_test {
std::unique_ptr<OperationPass<ModuleOp>> CreateInitTextFileToImportTestPass() {
  return std::make_unique<TF::InitTextFileToImportTestPass>();
}
std::unique_ptr<OperationPass<ModuleOp>>
CreateInitTextFileToImportSavedModelTestPass() {
  return std::make_unique<TF::InitTextFileToImportSavedModelTestPass>();
}
}  // namespace tf_test

}  // namespace mlir
