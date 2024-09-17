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

#include "tensorflow/compiler/mlir/lite/debug/debug.h"

#include <stdint.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/debug/debug_options.pb.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace tensorflow {
namespace debug_test {

class NopPass : public mlir::PassWrapper<NopPass, mlir::OperationPass<>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NopPass)

  void runOnOperation() override {}
};

class MutatePass : public mlir::PassWrapper<MutatePass, mlir::OperationPass<>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MutatePass)

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());
    getOperation()->setAttr("tfl.random_attr", builder.getUnitAttr());
  }
};

class AlwaysFailPass
    : public mlir::PassWrapper<AlwaysFailPass, mlir::OperationPass<>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlwaysFailPass)

  void runOnOperation() override { signalPassFailure(); }
};

}  // namespace debug_test

namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using namespace tensorflow::debug_test;

class InitPassManagerTest : public testing::Test {
 protected:
  InitPassManagerTest()
      : path_(GetOutputPath()), context_([]() {
          mlir::registerPassManagerCLOptions();
          mlir::DialectRegistry registry;
          registry.insert<mlir::BuiltinDialect>();
          registry.insert<mlir::arith::ArithDialect>();
          registry.insert<mlir::func::FuncDialect>();
          registry.insert<mlir::TFL::TensorFlowLiteDialect>();
          return registry;
        }()) {
    context_.loadAllAvailableDialects();

    mlir::OpBuilder builder(&context_);
    module_ = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

    builder.setInsertionPointToStart(module_->getBody());
    auto func = builder.create<mlir::func::FuncOp>(  //
        builder.getUnknownLoc(), "main", builder.getFunctionType({}, {}));
    func->setAttr("tfl.func", builder.getUnitAttr());
    builder.setInsertionPointToStart(func.addEntryBlock());
    llvm::SmallVector<int> shape{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(),
        mlir::DenseIntElementsAttr::get(
            mlir::RankedTensorType::get(shape.size(), builder.getI32Type()),
            shape));
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  }

  absl::Status GetDumpDir(std::string* dump_dir) {
    std::vector<string> files;
    if (auto status = tsl::Env::Default()->GetChildren(path_, &files);
        !status.ok()) {
      return status;
    }
    if (files.size() != 1) {
      return absl::FailedPreconditionError(
          "Expecting directory to have one child.");
    }
    *dump_dir = tsl::io::JoinPath(path_, files[0]);
    return absl::OkStatus();
  }

  std::string path_;
  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;

 private:
  std::string GetOutputPath() {
    const auto* const test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    return tsl::io::JoinPath(
        getenv("TEST_UNDECLARED_OUTPUTS_DIR"),
        absl::StrCat(test_info->test_suite_name(), ".", test_info->name()));
  }
};

TEST_F(InitPassManagerTest, CrashReproducer) {
  converter::DebugOptions debug_options;
  *debug_options.mutable_ir_dump_dir() = path_;

  mlir::PassManager pm(&context_);
  InitPassManager(pm, debug_options);
  pm.addPass(std::make_unique<AlwaysFailPass>());
  ASSERT_TRUE(mlir::failed(pm.run(*module_)));

  std::string dump_dir;
  TF_ASSERT_OK(GetDumpDir(&dump_dir));

  std::string mlir_dump;
  TF_ASSERT_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(dump_dir, "tfl_mlir_crash_repro.mlir"), &mlir_dump));
  EXPECT_THAT(mlir_dump, Not(IsEmpty()));
}

TEST_F(InitPassManagerTest, DumpToDir) {
  converter::DebugOptions debug_options;
  *debug_options.mutable_ir_dump_dir() = path_;
  *debug_options.mutable_ir_dump_pass_regex() = R"(.*NopPass)";

  mlir::PassManager pm(&context_);
  InitPassManager(pm, debug_options);
  pm.addPass(std::make_unique<NopPass>());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module_)));
  std::string dump_dir;
  TF_ASSERT_OK(GetDumpDir(&dump_dir));

  {
    std::string mlir_dump;
    TF_ASSERT_OK(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(
            dump_dir, "00000000.main.tensorflow_debug_test_NopPass_after.mlir"),
        &mlir_dump));
    EXPECT_THAT(mlir_dump, Not(IsEmpty()));
  }
  {
    std::string mlir_dump;
    TF_ASSERT_OK(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(
            dump_dir,
            "00000000.main.tensorflow_debug_test_NopPass_before.mlir"),
        &mlir_dump));
    EXPECT_THAT(mlir_dump, Not(IsEmpty()));
  }
}

TEST_F(InitPassManagerTest, PrintIRBeforeEverything) {
  converter::DebugOptions debug_options;
  *debug_options.mutable_print_ir_before() = R"(.*)";
  std::string captured_out;
  llvm::raw_string_ostream out(captured_out);

  mlir::PassManager pm(&context_);
  InitPassManager(pm, debug_options, out);
  pm.addPass(std::make_unique<NopPass>());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module_)));

  EXPECT_THAT(captured_out,
              HasSubstr("IR Dump Before tensorflow::debug_test::NopPass"));
  EXPECT_THAT(captured_out,
              Not(HasSubstr("IR Dump After tensorflow::debug_test::NopPass")));
}

TEST_F(InitPassManagerTest, PrintIRAfterEverything) {
  converter::DebugOptions debug_options;
  *debug_options.mutable_print_ir_after() = R"(.*)";
  std::string captured_out;
  llvm::raw_string_ostream out(captured_out);

  mlir::PassManager pm(&context_);
  InitPassManager(pm, debug_options, out);
  pm.addPass(std::make_unique<MutatePass>());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module_)));

  EXPECT_THAT(captured_out,
              HasSubstr("IR Dump After tensorflow::debug_test::MutatePass"));
  EXPECT_THAT(
      captured_out,
      Not(HasSubstr("IR Dump Before tensorflow::debug_test::MutatePass")));
}

TEST_F(InitPassManagerTest, PrintIRBeforeAndAfterEverything) {
  converter::DebugOptions debug_options;
  *debug_options.mutable_print_ir_before() = R"(.*)";
  *debug_options.mutable_print_ir_after() = R"(.*)";
  std::string captured_out;
  llvm::raw_string_ostream out(captured_out);

  mlir::PassManager pm(&context_);
  InitPassManager(pm, debug_options, out);
  pm.addPass(std::make_unique<MutatePass>());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module_)));

  EXPECT_THAT(captured_out,
              HasSubstr("IR Dump After tensorflow::debug_test::MutatePass"));
  EXPECT_THAT(captured_out,
              HasSubstr("IR Dump Before tensorflow::debug_test::MutatePass"));
}

TEST_F(InitPassManagerTest, ElideLargeElementAttrs) {
  converter::DebugOptions debug_options;
  *debug_options.mutable_print_ir_before() = R"(.*)";
  debug_options.set_elide_elementsattrs_if_larger(5);
  std::string captured_out;
  llvm::raw_string_ostream out(captured_out);

  mlir::PassManager pm(&context_);
  InitPassManager(pm, debug_options, out);
  pm.addPass(std::make_unique<MutatePass>());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module_)));

  EXPECT_THAT(captured_out, HasSubstr("dense_resource<__elided__>"));
}

TEST_F(InitPassManagerTest, DontElideSmallerElementAttrs) {
  converter::DebugOptions debug_options;
  *debug_options.mutable_print_ir_before() = R"(.*)";
  debug_options.set_elide_elementsattrs_if_larger(11);
  std::string captured_out;
  llvm::raw_string_ostream out(captured_out);

  mlir::PassManager pm(&context_);
  InitPassManager(pm, debug_options, out);
  pm.addPass(std::make_unique<MutatePass>());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module_)));

  EXPECT_THAT(captured_out,
              HasSubstr("dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]>"));
}

}  // namespace
}  // namespace tensorflow
