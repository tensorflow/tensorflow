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

#include "xla/python/ifrt/ir/transforms/debug.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "xla/python/ifrt/support/module_parsing.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::Contains;
using ::testing::ContainsRegex;
using ::tsl::testing::IsOkAndHolds;

class NopPass : public mlir::PassWrapper<NopPass, mlir::OperationPass<>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NopPass)

  void runOnOperation() override {}
};

class AlwaysFailPass
    : public mlir::PassWrapper<AlwaysFailPass, mlir::OperationPass<>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlwaysFailPass)

  void runOnOperation() override { signalPassFailure(); }
};

class InitPassManagerTest : public testing::Test {
 protected:
  InitPassManagerTest() {
    xla::ifrt::support::RegisterMlirDialects(context_);
    context_.loadAllAvailableDialects();

    mlir::OpBuilder builder(&context_);
    module_ = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

    builder.setInsertionPointToStart(module_->getBody());
    auto func = builder.create<mlir::func::FuncOp>(  //
        builder.getUnknownLoc(), "program", builder.getFunctionType({}, {}));
    func->setAttr("pw.program", builder.getUnitAttr());

    builder.setInsertionPointToStart(func.addEntryBlock());
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  }

  absl::StatusOr<std::vector<std::string>> MatchUndeclaredOutputs() {
    std::vector<std::string> paths;
    TF_RETURN_IF_ERROR(tsl::Env::Default()->GetMatchingPaths(
        tsl::io::JoinPath(
            absl::NullSafeStringView(getenv("TEST_UNDECLARED_OUTPUTS_DIR")),
            "*.mlir"),
        &paths));
    return paths;
  }

  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

TEST_F(InitPassManagerTest, CrashReproducer) {
  mlir::PassManager pm(&context_);
  InitPassManager(pm, "foo");
  pm.addPass(std::make_unique<AlwaysFailPass>());
  ASSERT_TRUE(mlir::failed(pm.run(*module_)));

  EXPECT_THAT(
      MatchUndeclaredOutputs(),
      IsOkAndHolds(Contains(ContainsRegex(R"(ifrt_ir_mlir_repro_.*\.mlir$)"))));
}

TEST_F(InitPassManagerTest, Dump) {
  mlir::PassManager pm(&context_);
  InitPassManager(pm, "foo", /*dump_dir=*/"sponge",
                  /*dump_pass_re=*/".*NopPass");
  pm.addPass(std::make_unique<NopPass>());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module_)));

  EXPECT_THAT(MatchUndeclaredOutputs(),
              IsOkAndHolds(Contains(
                  ContainsRegex(R"(.*\.program\..*NopPass.*\.mlir)"))));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
