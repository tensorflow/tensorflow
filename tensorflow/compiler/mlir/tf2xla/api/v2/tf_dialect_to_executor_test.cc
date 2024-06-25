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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/tf_dialect_to_executor.h"

#include <stdlib.h>

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tsl/lib/core/status_test_util.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {
namespace {

constexpr char kExportStreamzName[] =
    "/tensorflow/core/tf2xla/api/v2/tf_dialect_to_executor_dialect_status";
constexpr char kExportSuccess[] = "success";
constexpr char kExportFailed[] = "failed";

using mlir::DialectRegistry;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OwningOpRef;
using ::tensorflow::monitoring::testing::CellReader;

std::string TestDataPath() {
  return tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tf2xla/api/v2/testdata/");
}

size_t CountSubstring(absl::string_view str, absl::string_view substr) {
  size_t count = 0;
  size_t idx = str.find(substr);
  while (idx != std::string::npos) {
    count++;
    idx = str.find(substr, idx + 1);
  }
  return count;
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
  CellReader<int64_t> compilation_status(kExportStreamzName);

  TF_ASSERT_OK(CreateMlirModule("empty_func.mlir"));

  TF_EXPECT_OK(ExportFromTensorflowDialectToExecutor(*mlir_module_));

  EXPECT_EQ(compilation_status.Delta(kExportSuccess), 1);
  EXPECT_EQ(compilation_status.Delta(kExportFailed), 0);
}

TEST_F(TensorflowDialectToExecutorTest, ErrorsWhenCannotConvert) {
  CellReader<int64_t> compilation_status(kExportStreamzName);

  TF_ASSERT_OK(CreateMlirModule("invalid_executor.mlir"));

  EXPECT_FALSE(ExportFromTensorflowDialectToExecutor(*mlir_module_).ok());

  EXPECT_EQ(compilation_status.Delta(kExportSuccess), 0);
  EXPECT_EQ(compilation_status.Delta(kExportFailed), 1);
}

TEST_F(TensorflowDialectToExecutorTest, PrunesDeadOps) {
  CellReader<int64_t> compilation_status(kExportStreamzName);

  TF_ASSERT_OK(CreateMlirModule("func_with_dead_ops.mlir"));

  TF_EXPECT_OK(ExportFromTensorflowDialectToExecutor(*mlir_module_));

  std::string module_dump;
  llvm::raw_string_ostream raw_stream(module_dump);
  mlir_module_->print(raw_stream);

  EXPECT_EQ(compilation_status.Delta(kExportSuccess), 1);
  EXPECT_EQ(compilation_status.Delta(kExportFailed), 0);
  EXPECT_EQ(
      CountSubstring(module_dump, "tf_executor.island wraps \"tf.Concat\""), 2);
}

}  // namespace
}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
