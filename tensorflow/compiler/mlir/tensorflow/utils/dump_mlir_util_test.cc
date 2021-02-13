/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(DumpMlirModuleTest, NoEnvPrefix) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  unsetenv("TF_DUMP_GRAPH_PREFIX");

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  EXPECT_EQ(filepath, "(TF_DUMP_GRAPH_PREFIX not specified)");
}

TEST(DumpMlirModuleTest, LogInfo) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  setenv("TF_DUMP_GRAPH_PREFIX", "-", 1);

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  EXPECT_EQ(filepath, "(stderr)");
}

TEST(DumpMlirModuleTest, Valid) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  std::string expected_txt_module;
  {
    llvm::raw_string_ostream os(expected_txt_module);
    module_ref->getOperation()->print(
        os, mlir::OpPrintingFlags().useLocalScope().printGenericOpForm());
    os.flush();
  }

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  ASSERT_NE(filepath, "(TF_DUMP_GRAPH_PREFIX not specified)");
  ASSERT_NE(filepath, "LOG(INFO)");
  ASSERT_NE(filepath, "(unavailable)");

  Env* env = Env::Default();
  std::string file_txt_module;
  TF_ASSERT_OK(ReadFileToString(env, filepath, &file_txt_module));
  EXPECT_EQ(file_txt_module, expected_txt_module);
}

TEST(DumpRawStringToFileTest, Valid) {
  llvm::StringRef example = "module {\n}";
  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);

  std::string filepath = DumpRawStringToFile("example", example);
  ASSERT_NE(filepath, "(TF_DUMP_GRAPH_PREFIX not specified)");
  ASSERT_NE(filepath, "LOG(INFO)");
  ASSERT_NE(filepath, "(unavailable)");

  Env* env = Env::Default();
  std::string file_txt_module;
  TF_ASSERT_OK(ReadFileToString(env, filepath, &file_txt_module));
  EXPECT_EQ(file_txt_module, example);
}

}  // namespace
}  // namespace tensorflow
