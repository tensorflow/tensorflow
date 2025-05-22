/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_saved_model_import.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_test_base.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"

namespace mlir::tf_quant::stablehlo {
namespace {

using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

using UpdateFunctionAliasesTest = ::mlir::tf_quant::QuantizationTestBase;

TEST_F(UpdateFunctionAliasesTest, NoAliasesReturnsEmptyMap) {
  // MLIR @main function corresponds to the TF function "main_original".
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(R"mlir(
    func.func private @main(%arg: tensor<1x2xf32>) -> (tensor<1x2xf32>) attributes {tf._original_func_name = "main_original"} {
      return %arg : tensor<1x2xf32>
    }
  )mlir");
  ASSERT_TRUE(module_op);

  absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases;
  UpdateFunctionAliases(function_aliases, *module_op);
  EXPECT_THAT(function_aliases, IsEmpty());
}

TEST_F(UpdateFunctionAliasesTest, AliasUpdatedByMlirFunctionName) {
  // MLIR @main function corresponds to the TF function "main_original".
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(R"mlir(
    func.func private @main(%arg: tensor<1x2xf32>) -> (tensor<1x2xf32>) attributes {tf._original_func_name = "main_original"} {
      return %arg : tensor<1x2xf32>
    }
  )mlir");
  ASSERT_TRUE(module_op);

  absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases{
      {"main_original", "main_alias"}};
  UpdateFunctionAliases(function_aliases, *module_op);

  EXPECT_THAT(function_aliases,
              UnorderedElementsAre(Pair("main", "main_alias")));
}

TEST_F(UpdateFunctionAliasesTest, IgnoresUnmatchedFunctions) {
  // MLIR @main function corresponds to the TF function "main_original".
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(R"mlir(
    func.func private @main(%arg: tensor<1x2xf32>) -> (tensor<1x2xf32>) attributes {tf._original_func_name = "main_original"} {
      return %arg : tensor<1x2xf32>
    }
  )mlir");
  ASSERT_TRUE(module_op);

  // There is no alias corresponding to "main_original". The existing entry
  // without a corresponding function is ignored.
  absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases{
      {"not_main", "not_main_alias"}};
  UpdateFunctionAliases(function_aliases, *module_op);

  EXPECT_THAT(function_aliases, IsEmpty());
}

TEST_F(UpdateFunctionAliasesTest,
       SkipsFunctionsWithNoOriginalFuncNameAttribute) {
  // @main does not have the "tf._original_func_name" attribute.
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(R"mlir(
    func.func private @main(%arg: tensor<1x2xf32>) -> (tensor<1x2xf32>) {
      return %arg : tensor<1x2xf32>
    }
  )mlir");
  ASSERT_TRUE(module_op);

  // The existing entry without a corresponding function is ignored.
  absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases{
      {"main_original", "main_alias"}};
  UpdateFunctionAliases(function_aliases, *module_op);

  EXPECT_THAT(function_aliases, IsEmpty());
}

TEST_F(UpdateFunctionAliasesTest, FunctionNameNotChanged) {
  // @main does not have the "tf._original_func_name" attribute.
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(R"mlir(
    func.func private @main_original(%arg: tensor<1x2xf32>) -> (tensor<1x2xf32>) {
      return %arg : tensor<1x2xf32>
    }
  )mlir");
  ASSERT_TRUE(module_op);

  // The existing entry without a corresponding function is ignored.
  absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases{
      {"main_original", "main_alias"}};
  UpdateFunctionAliases(function_aliases, *module_op);

  EXPECT_THAT(function_aliases,
              UnorderedElementsAre(Pair("main_original", "main_alias")));
}

}  // namespace
}  // namespace mlir::tf_quant::stablehlo
