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

#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_PRINTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class PrintPass : public impl::PrintPassBase<PrintPass> {
 public:
  explicit PrintPass(raw_ostream* os = nullptr);
  PrintPass(const PrintPass& other);
  void runOnOperation() override;

 private:
  llvm::sys::SmartMutex<true> mutex_;
  raw_ostream* os_;
};

PrintPass::PrintPass(raw_ostream* os) {
  if (os) {
    os_ = os;
  } else {
    os_ = &llvm::errs();
  }
}

PrintPass::PrintPass(const PrintPass& other) : PrintPass(other.os_) {}

void PrintPass::runOnOperation() {
  llvm::sys::SmartScopedLock<true> instrumentationLock(mutex_);
  OpPrintingFlags flags =
      OpPrintingFlags().elideLargeElementsAttrs().enableDebugInfo(false);
  getOperation()->print(*os_, flags);
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreatePrintPass(raw_ostream* os) {
  return std::make_unique<PrintPass>(os);
}

}  // namespace TF
}  // namespace mlir
