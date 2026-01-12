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

#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

#include <string>

#include <gmock/gmock.h>
#include "absl/status/status_matchers.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "xla/hlo/translate/register.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

// This file should contain tests for interfaces that can't be tested at the
// MLIR level.

namespace mlir {
namespace {

using testing::_;
using testing::AllOf;
using testing::HasSubstr;

TEST(ConvertMlirHloToHloModuleTest, PropagatesDiagnostics) {
  const std::string mlir_source = R"mlir(
func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>) -> tensor<?xf32> {
  %0 = shape.const_shape [14, 1] : tensor<2xindex>
  %1 = "stablehlo.real_dynamic_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}
)mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::BaseScopedDiagnosticHandler handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
    ASSERT_OK(handler.ConsumeStatus());
  }

  ASSERT_THAT(ConvertMlirHloToHloModule(*module),
              absl_testing::StatusIs(
                  _, AllOf(HasSubstr("Unable to prepare for XLA export"),
                           HasSubstr("real_dynamic_slice"))));
}

TEST(ConvertMlirHloToHloModuleTest, ConvertsDotGeneralPrecisionConfig) {
  const std::string mlir_source = R"mlir(
func.func @main(%arg0: tensor<5x10xbf16>, %arg1: tensor<10x5xbf16>) -> tensor<5x5xbf16> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<5x10xbf16>, tensor<10x5xbf16>) -> tensor<5x5xbf16>
  return %0 : tensor<5x5xbf16>
}
)mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::BaseScopedDiagnosticHandler handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
    ASSERT_OK(handler.ConsumeStatus());
  }

  ASSERT_OK(ConvertMlirHloToHloModule(*module));
}
TEST(ConvertMlirHloToHloModuleTest, ConvertsConvolutionPrecisionConfig) {
  const std::string mlir_source = R"mlir(
func.func @main(%arg0: tensor<3x3x3x3xf32>, %arg1: tensor<3x3x3x3xf32>) -> tensor<3x3x3x3xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<3x3x3x3xf32>, tensor<3x3x3x3xf32>) -> tensor<3x3x3x3xf32>
  return %0 : tensor<3x3x3x3xf32>
}
)mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::BaseScopedDiagnosticHandler handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
    ASSERT_OK(handler.ConsumeStatus());
  }

  ASSERT_OK(ConvertMlirHloToHloModule(*module));
}
}  // namespace
}  // namespace mlir
