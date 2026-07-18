/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/dump/mlir.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace pjrt {
namespace {

TEST(MlirTest, MlirModuleToFile) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::mhlo::MhloDialect>();
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      xla::llvm_ir::CreateMlirModuleOp(builder.getUnknownLoc());

  std::string file_path =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "test_mlir_module.mlir");
  TF_ASSERT_OK(MlirModuleToFile(module.get(), file_path));

  std::string file_content;
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), file_path, &file_content));
  EXPECT_THAT(file_content, ::testing::HasSubstr("module"));
}

}  // namespace
}  // namespace pjrt
