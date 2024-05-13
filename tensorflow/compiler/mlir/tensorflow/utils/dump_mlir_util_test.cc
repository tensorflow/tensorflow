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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::testing::IsNull;

TEST(DumpMlirModuleTest, NoEnvPrefix) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  unsetenv("TF_DUMP_GRAPH_PREFIX");

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  EXPECT_EQ(filepath, "(TF_DUMP_GRAPH_PREFIX not specified)");
}

TEST(DumpMlirModuleTest, LogInfo) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  setenv("TF_DUMP_GRAPH_PREFIX", "-", 1);

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  EXPECT_EQ(filepath, "(stderr; requested filename: 'module')");
}

TEST(DumpMlirModuleTest, Valid) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  std::string expected_txt_module;
  {
    llvm::raw_string_ostream os(expected_txt_module);
    module_ref->getOperation()->print(os,
                                      mlir::OpPrintingFlags().useLocalScope());
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

TEST(DumpCrashReproducerTest, RoundtripDumpAndReadValid) {
  mlir::registerPassManagerCLOptions();
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  std::string filepath =
      testing::TmpDir() + "/" + mlir::TF::kStandardPipelineBefore + ".mlir";

  std::string output_dump = testing::TmpDir() + "/" + "output_dump.txt";

  TF_ASSERT_OK(mlir::TF::RunBridgeWithStandardPipeline(
      module_ref.get(),
      /*enable_logging=*/true, /*enable_inliner=*/false));

  std::string errorMessage;
  auto input_file = mlir::openInputFile(filepath, &errorMessage);
  EXPECT_THAT(input_file, Not(IsNull()));

  auto output_stream = mlir::openOutputFile(output_dump, &errorMessage);
  EXPECT_THAT(output_stream, Not(IsNull()));

  mlir::PassPipelineCLParser passPipeline(
      /*arg=*/"", /*description=*/"Compiler passes to run", /*alias=*/"p");
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::registerAllPasses();
  mlir::registerTensorFlowPasses();

  EXPECT_TRUE(mlir::MlirOptMain(output_stream->os(), std::move(input_file),
                                registry,
                                mlir::MlirOptMainConfig{}
                                    .splitInputFile("")
                                    .verifyDiagnostics(false)
                                    .verifyPasses(false)
                                    .allowUnregisteredDialects(false)
                                    .setPassPipelineParser(passPipeline))
                  .succeeded());
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
