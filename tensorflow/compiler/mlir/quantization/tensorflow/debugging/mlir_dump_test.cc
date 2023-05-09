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
#include "tensorflow/compiler/mlir/quantization/tensorflow/debugging/mlir_dump.h"

#include <memory>
#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/test.h"

namespace tensorflow {
namespace quantization {
namespace {

class NoOpPass
    : public mlir::PassWrapper<NoOpPass, mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NoOpPass)

  NoOpPass() = default;

  llvm::StringRef getArgument() const final { return "no-op-pass"; }

  void runOnOperation() override {
    // Noop pass does nothing on the operation.
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateNoOpPass() {
  return std::make_unique<NoOpPass>();
}

TEST(EnableIrPrintingTest, PassSuccessfullyRuns) {
  mlir::MLIRContext ctx{};

  mlir::PassManager pm = {&ctx};
  pm.addPass(CreateNoOpPass());

  std::error_code ec{};  // NOLINT: Required to create llvm::raw_fd_ostream
  const std::string tmp_dump_filename =
      tsl::io::GetTempFilename(/*extension=*/".mlir");
  llvm::raw_fd_ostream dump_file{tmp_dump_filename, ec};

  EnableIrPrinting(dump_file, pm);

  mlir::OpBuilder builder(&ctx);
  auto module_op = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  // Destroy by calling destroy() to avoid memory leak since it is allocated
  // with malloc().
  const absl::Cleanup module_op_cleanup = [module_op] { module_op->destroy(); };

  const mlir::LogicalResult result = pm.run(module_op);
  EXPECT_FALSE(failed(result));
}

}  // namespace
}  // namespace quantization
}  // namespace tensorflow
