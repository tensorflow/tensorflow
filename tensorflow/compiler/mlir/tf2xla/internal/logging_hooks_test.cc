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

#include "tensorflow/compiler/mlir/tf2xla/internal/logging_hooks.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/file_statistics.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {
namespace {

using mlir::DialectRegistry;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OwningOpRef;
using mlir::PassManager;
using mlir::func::FuncOp;

std::string TestDataPath() {
  return tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tf2xla/internal/testdata/");
}

class LoggingHooksTest : public ::testing::Test {
 public:
  LoggingHooksTest() {
    mlir::RegisterCommonToolingDialects(registry_);
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();
    env_ = Env::Default();

    test_group_name_ = "TestGroup";
    test_dir_ = testing::TmpDir();
    setenv(/*name=*/"TF_DUMP_GRAPH_PREFIX", /*value=*/test_dir_.c_str(),
           /*overwrite=*/1);
  }

  absl::Status CreateMlirModule(std::string mlir_module_filename) {
    std::string mlir_module_path = TestDataPath() + mlir_module_filename;
    mlir_module_ =
        mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);
    if (!mlir_module_) {
      return absl::Status(
          absl::StatusCode::kNotFound,
          absl::StrCat("Could not find MLIR module at ", mlir_module_path));
    }
    return absl::OkStatus();
  }

  DialectRegistry registry_;
  MLIRContext context_;
  OwningOpRef<mlir::ModuleOp> mlir_module_;
  Env* env_;
  std::string test_dir_;
  std::string test_group_name_;
};

TEST_F(LoggingHooksTest, DumpsPassData) {
  std::vector<std::string> files;
  TF_ASSERT_OK(env_->GetChildren(test_dir_, &files));
  EXPECT_THAT(files, ::testing::IsEmpty());

  TF_ASSERT_OK(CreateMlirModule("dead_const.mlir"));
  PassManager pass_manager(&context_);
  pass_manager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  EnablePassIRPrinting(pass_manager, test_group_name_);

  LogicalResult pass_status = pass_manager.run(mlir_module_.get());
  EXPECT_TRUE(pass_status.succeeded());

  TF_ASSERT_OK(env_->GetChildren(test_dir_, &files));
  EXPECT_THAT(files, ::testing::SizeIs(2));
}

};  // namespace
};  // namespace internal
};  // namespace tf2xla
};  // namespace tensorflow
