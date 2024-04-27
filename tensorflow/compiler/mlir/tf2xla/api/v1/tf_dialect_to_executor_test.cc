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

#include "tensorflow/compiler/mlir/tf2xla/api/v1/tf_dialect_to_executor.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {
namespace {

using mlir::ModuleOp;

using mlir::DialectRegistry;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OwningOpRef;
using tensorflow::monitoring::testing::CellReader;

static constexpr char kCompilationStreamz[] =
    "/tensorflow/core/tf2xla/api/v1/tf_dialect_to_executor_dialect_status";

std::string TestDataPath() {
  return tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tf2xla/api/v1/testdata/");
}

class TensorflowDialectToExecutorTest : public ::testing::Test {
 public:
  TensorflowDialectToExecutorTest() {
    mlir::RegisterCommonToolingDialects(registry_);
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();
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
};

TEST_F(TensorflowDialectToExecutorTest, ConvertsToExecutor) {
  CellReader<int64_t> compilation_status(kCompilationStreamz);

  TF_ASSERT_OK(CreateMlirModule("empty_func.mlir"));

  TF_EXPECT_OK(ExportFromTensorflowDialectToExecutor(*mlir_module_));

  EXPECT_EQ(compilation_status.Delta("success"), 1);
}

TEST_F(TensorflowDialectToExecutorTest, ErrorsWhenCannotConvert) {
  CellReader<int64_t> compilation_status(kCompilationStreamz);

  TF_ASSERT_OK(CreateMlirModule("invalid_executor.mlir"));

  EXPECT_FALSE(ExportFromTensorflowDialectToExecutor(*mlir_module_).ok());

  EXPECT_EQ(compilation_status.Delta("failed"), 1);
}

}  // namespace
}  // namespace v1
}  // namespace tf2xla
}  // namespace tensorflow
