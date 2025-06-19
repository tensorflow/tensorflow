/* Copyright 2023 The OpenXLA Authors.

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
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;
using ::tsl::testing::StatusIs;

class HloProgramSerDesTest : public testing::TestWithParam<SerDesVersion> {
 public:
  HloProgramSerDesTest() : version_(GetParam()) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

TEST_P(HloProgramSerDesTest, RoundTrip) {
  static constexpr absl::string_view kMlirModuleStr = R"(
module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "stablehlo.broadcast_in_dim"(%0) {broadcast_dimensions = array<i64>} : (tensor<f32>) -> tensor<2x3xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }
})";

  Serialized serialized;
  {
    auto context = std::make_unique<mlir::MLIRContext>();
    TF_ASSERT_OK_AND_ASSIGN(
        mlir::OwningOpRef<mlir::ModuleOp> module,
        xla::ParseMlirModuleString(kMlirModuleStr, *context));
    auto program =
        std::make_unique<HloProgram>(std::move(context), std::move(module));
    auto options = std::make_unique<SerializeOptions>();
    options->version = version();
    TF_ASSERT_OK_AND_ASSIGN(serialized,
                            Serialize(*program, std::move(options)));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloProgram> xla_program,
      Deserialize<HloProgram>(serialized, /*options=*/nullptr));

  // Verify that the deserialized program has no MHLO ops.
  bool has_unsupported_dialect = false;
  xla_program->mlir_module()->walk([&](mlir::Operation *op) {
    if (!llvm::isa<mlir::BuiltinDialect, mlir::func::FuncDialect,
                   mlir::stablehlo::StablehloDialect>(op->getDialect())) {
      LOG(ERROR) << "Found an op with an unsupported dialect: "
                 << mlir::debugString(*op);
      has_unsupported_dialect = true;
    }
  });
  EXPECT_FALSE(has_unsupported_dialect);
}

TEST_P(HloProgramSerDesTest, SerializationError) {
  static constexpr absl::string_view kMlirModuleStr = R"(
module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = "UnknownOp"(%arg0) : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
})";

  Serialized serialized;
  {
    auto context = std::make_unique<mlir::MLIRContext>();
    context->allowUnregisteredDialects();
    TF_ASSERT_OK_AND_ASSIGN(
        mlir::OwningOpRef<mlir::ModuleOp> module,
        xla::ParseMlirModuleString(kMlirModuleStr, *context));
    auto program =
        std::make_unique<HloProgram>(std::move(context), std::move(module));
    auto options = std::make_unique<SerializeOptions>();
    options->version = version();
    EXPECT_THAT(Serialize(*program, std::move(options)),
                StatusIs(Not(absl::StatusCode::kOk),
                         HasSubstr("Failed to serialize StableHLO")));
  }
}

TEST_P(HloProgramSerDesTest, DeserializationError) {
  static constexpr absl::string_view kMlirModuleStr = R"(
module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
  }
})";

  Serialized serialized;
  {
    auto context = std::make_unique<mlir::MLIRContext>();
    TF_ASSERT_OK_AND_ASSIGN(
        mlir::OwningOpRef<mlir::ModuleOp> module,
        xla::ParseMlirModuleString(kMlirModuleStr, *context));
    auto program =
        std::make_unique<HloProgram>(std::move(context), std::move(module));
    auto options = std::make_unique<SerializeOptions>();
    options->version = version();
    TF_ASSERT_OK_AND_ASSIGN(serialized,
                            Serialize(*program, std::move(options)));
  }

  serialized.set_data("invalid data");

  EXPECT_THAT(Deserialize<HloProgram>(serialized, /*options=*/nullptr),
              StatusIs(Not(absl::StatusCode::kOk),
                       HasSubstr("Failed to deserialize StableHLO module")));
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, HloProgramSerDesTest,
    testing::ValuesIn(test_util::Week4OldOrLaterSerDesVersions()));

}  // namespace
}  // namespace ifrt
}  // namespace xla
