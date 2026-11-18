/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/export/types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_types.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#include "tensorflow/core/ir/types/dialect.h"

namespace mlir::TFL::fb_export {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

class GetTensorTypeTest : public ::testing::Test {
 protected:
  GetTensorTypeTest() {
    context_.loadDialect<TensorFlowLiteDialect, tf_type::TFTypeDialect>();
  }

  MLIRContext context_;
};

TEST_F(GetTensorTypeTest, MapsTflTypes) {
  EXPECT_THAT(GetTensorType(StringType::get(&context_)),
              IsOkAndHolds(tflite::TensorType_STRING));
  EXPECT_THAT(GetTensorType(Quint8Type::get(&context_)),
              IsOkAndHolds(tflite::TensorType_UINT8));
  EXPECT_THAT(GetTensorType(ResourceType::get(&context_)),
              IsOkAndHolds(tflite::TensorType_RESOURCE));
  EXPECT_THAT(GetTensorType(ResourceType::get(
                  &context_,
                  {RankedTensorType::get({2}, Float32Type::get(&context_))})),
              IsOkAndHolds(tflite::TensorType_RESOURCE));
  EXPECT_THAT(GetTensorType(VariantType::get(&context_)),
              IsOkAndHolds(tflite::TensorType_VARIANT));
}

// The matching `tf_type` types are exported the same way (matched by name).
TEST_F(GetTensorTypeTest, MapsTfTypesByName) {
  EXPECT_THAT(GetTensorType(tf_type::StringType::get(&context_)),
              IsOkAndHolds(tflite::TensorType_STRING));
  EXPECT_THAT(GetTensorType(tf_type::Quint8Type::get(&context_)),
              IsOkAndHolds(tflite::TensorType_UINT8));
  EXPECT_THAT(GetTensorType(tf_type::ResourceType::get(&context_)),
              IsOkAndHolds(tflite::TensorType_RESOURCE));
  EXPECT_THAT(GetTensorType(tf_type::VariantType::get(&context_)),
              IsOkAndHolds(tflite::TensorType_VARIANT));
}

TEST_F(GetTensorTypeTest, RejectsTfRefTypes) {
  EXPECT_THAT(GetTensorType(tf_type::StringRefType::get(&context_)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetTensorType(tf_type::ResourceRefType::get(&context_)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace mlir::TFL::fb_export
