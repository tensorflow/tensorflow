/* Copyright 2024 The OpenXLA Authors.

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
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/dialect/Version.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/support/module_parsing.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;
using ::tsl::testing::StatusIs;

std::string PrintModule(mlir::ModuleOp module) {
  std::string module_str;
  llvm::raw_string_ostream os(module_str);
  module->print(os, mlir::OpPrintingFlags().enableDebugInfo());
  return module_str;
}

class IfrtIRProgramSerDesTest : public testing::TestWithParam<SerDesVersion> {
 public:
  IfrtIRProgramSerDesTest() : version_(GetParam()) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

TEST_P(IfrtIRProgramSerDesTest, RoundTrip) {
  static constexpr absl::string_view kMlirModuleStr = R"(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
  )";

  Serialized serialized;
  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      support::ParseMlirModuleString(kMlirModuleStr, *context));
  auto initial_program =
      std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));

  // TODO(hyeontaek): Use `version()` to fill in
  // `SerializeIfrtIRProgramOptions::ifrt_version`.
  TF_ASSERT_OK_AND_ASSIGN(serialized,
                          Serialize(*initial_program, /*options=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IfrtIRProgram> deserialized_program,
      Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr));

  EXPECT_EQ(PrintModule(initial_program->mlir_module),
            PrintModule(deserialized_program->mlir_module));
}

TEST_P(IfrtIRProgramSerDesTest, VersioningRoundTrip) {
  static constexpr absl::string_view kMlirModuleStr = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module @multiple_calls_of_same_module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array) -> !array
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) on devices [0,1]
        : (!array) -> !array
    %2, %ctrl_2 = ifrt.Call @add_one::@main(%1) after %ctrl_1 on devices [0,1]
        : (!array) -> !array
    return %2 : !array
  }

  module @add_one attributes {sym_visibility = "private"} {
    func.func private @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}
  )";

  Serialized serialized;
  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      support::ParseMlirModuleString(kMlirModuleStr, *context));
  auto initial_program =
      std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));

  // TODO(hyeontaek): Use `version()` to fill in
  // `SerializeIfrtIRProgramOptions::ifrt_version`.
  auto options = std::make_unique<SerializeIfrtIRProgramOptions>(
      Version::getCurrentVersion().toString(),
      ::mlir::vhlo::Version::getCurrentVersion().toString(),
      /*version_in_place=*/false);
  TF_ASSERT_OK_AND_ASSIGN(serialized,
                          Serialize(*initial_program, std::move(options)));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IfrtIRProgram> deserialized_program,
      Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr));

  EXPECT_EQ(PrintModule(initial_program->mlir_module),
            PrintModule(deserialized_program->mlir_module));
}

TEST_P(IfrtIRProgramSerDesTest, VersioningOfAtomProgramInMhloShouldFail) {
  static constexpr absl::string_view kMlirModuleStr = R"(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
  )";

  Serialized serialized;
  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      support::ParseMlirModuleString(kMlirModuleStr, *context));
  auto initial_program =
      std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));

  // TODO(hyeontaek): Use `version()` to fill in
  // `SerializeIfrtIRProgramOptions::ifrt_version`.
  auto options = std::make_unique<SerializeIfrtIRProgramOptions>(
      Version::getCurrentVersion().toString(),
      ::mlir::vhlo::Version::getCurrentVersion().toString());
  EXPECT_THAT(Serialize(*initial_program, std::move(options)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Failed to version IFRT IR program")));
}

TEST_P(IfrtIRProgramSerDesTest, DeserializationError) {
  static constexpr absl::string_view kMlirModuleStr = R"(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
  )";
  Serialized serialized;
  {
    auto context = std::make_unique<mlir::MLIRContext>();
    TF_ASSERT_OK_AND_ASSIGN(
        mlir::OwningOpRef<mlir::ModuleOp> module,
        support::ParseMlirModuleString(kMlirModuleStr, *context));
    auto program =
        std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));
    // TODO(hyeontaek): Use `version()` to fill in
    // `SerializeIfrtIRProgramOptions::ifrt_version`.
    TF_ASSERT_OK_AND_ASSIGN(serialized,
                            Serialize(*program, /*options=*/nullptr));
  }

  serialized.set_data("invalid data");

  EXPECT_THAT(Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr),
              StatusIs(Not(absl::StatusCode::kOk),
                       HasSubstr("Failed to parse IfrtIrProgramProto")));
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, IfrtIRProgramSerDesTest,
    testing::ValuesIn(test_util::Week4OldOrLaterSerDesVersions()));

}  // namespace
}  // namespace ifrt
}  // namespace xla
