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
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;
using ::tsl::testing::StatusIs;

TEST(HloProgramSerDesTest, RoundTrip) {
  static constexpr absl::string_view kMlirModuleStr = R"(
module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "mhlo.copy"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast"(%1) {broadcast_sizes = dense<[2, 3]> : tensor<2xi64>} : (tensor<f32>) -> tensor<2x3xf32>
    %3 = mhlo.add %0, %2 : tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
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
    TF_ASSERT_OK_AND_ASSIGN(serialized,
                            Serialize(*program, /*options=*/nullptr));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloProgram> xla_program,
      Deserialize<HloProgram>(serialized, /*options=*/nullptr));

  // Verify that the deserialized program has no StableHLO ops.
  bool has_unsupported_dialect = false;
  xla_program->mlir_module->walk([&](mlir::Operation *op) {
    if (!llvm::isa<mlir::BuiltinDialect, mlir::func::FuncDialect,
                   mlir::mhlo::MhloDialect>(op->getDialect())) {
      LOG(ERROR) << "Found an op with an unsupported dialect: "
                 << mlir::debugString(op);
      has_unsupported_dialect = true;
    }
  });
  EXPECT_FALSE(has_unsupported_dialect);
}

TEST(HloProgramSerDesTest, SerializationError) {
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
    EXPECT_THAT(Serialize(*program, /*options=*/nullptr),
                StatusIs(Not(absl::StatusCode::kOk),
                         HasSubstr("Failed to serialize StableHLO")));
  }
}

TEST(HloProgramSerDesTest, DeserializationError) {
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
    TF_ASSERT_OK_AND_ASSIGN(serialized,
                            Serialize(*program, /*options=*/nullptr));
  }

  serialized.set_data("invalid data");

  EXPECT_THAT(Deserialize<HloProgram>(serialized, /*options=*/nullptr),
              StatusIs(Not(absl::StatusCode::kOk),
                       HasSubstr("Failed to deserialize StableHLO module")));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
