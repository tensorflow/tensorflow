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
// Test cases for the StableHLO Quantizer adaptor functions.

#include "tensorflow/compiler/mlir/lite/quantization/stablehlo/quantization.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tsl/platform/status_matchers.h"

namespace tensorflow {
namespace {

using ::stablehlo::quantization::QuantizationConfig;
using ::stablehlo::quantization::io::CreateTmpDir;
using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

// Test cases for `RunQuantization` mainly tests for error cases because testing
// for successful cases require passing python implementation to
// `quantization_py_function_lib`, which requires testing from the python level.
// Internal integration tests exist for testing successful quantization.

TEST(RunQuantizationTest,
     WhenSavedModelBundleIsNullptrReturnsInvalidArgumentError) {
  const absl::StatusOr<std::string> tmp_saved_model_dir = CreateTmpDir();
  ASSERT_THAT(tmp_saved_model_dir, IsOk());

  QuantizationConfig config;
  const absl::StatusOr<mlir::ModuleOp> quantized_module_op = RunQuantization(
      /*saved_model_bundle=*/nullptr, *tmp_saved_model_dir,
      /*saved_model_tags=*/{}, config,
      /*quantization_py_function_lib=*/nullptr, /*module_op=*/{});
  EXPECT_THAT(
      quantized_module_op,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("`saved_model_bundle` should not be nullptr")));
}

TEST(RunQuantizationTest,
     WhenPyFunctionLibIsNullptrReturnsInvalidArgumentError) {
  const absl::StatusOr<std::string> tmp_saved_model_dir = CreateTmpDir();
  ASSERT_THAT(tmp_saved_model_dir, IsOk());

  // Dummy SavedModelBundle to pass a non-nullptr argument.
  SavedModelBundle bundle{};
  QuantizationConfig config;
  const absl::StatusOr<mlir::ModuleOp> quantized_module_op = RunQuantization(
      /*saved_model_bundle=*/&bundle, *tmp_saved_model_dir,
      /*saved_model_tags=*/{}, config,
      /*quantization_py_function_lib=*/nullptr, /*module_op=*/{});
  EXPECT_THAT(
      quantized_module_op,
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("`quantization_py_function_lib` should not be nullptr")));
}

}  // namespace
}  // namespace tensorflow
