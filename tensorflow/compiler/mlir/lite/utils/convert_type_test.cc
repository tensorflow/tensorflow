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

#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_types.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/utils/tf_convert_type.h"
#include "tensorflow/core/ir/types/dialect.h"

namespace tflite {
namespace {

class ConvertTypeTest : public ::testing::Test {
 protected:
  ConvertTypeTest() : builder_(&context_) {
    context_.loadDialect<mlir::TFL::TensorFlowLiteDialect,
                         mlir::tf_type::TFTypeDialect>();
  }

  mlir::MLIRContext context_;
  mlir::Builder builder_;
};

TEST_F(ConvertTypeTest, ConvertElementTypeReturnsTflTypes) {
  EXPECT_TRUE(mlir::isa<mlir::TFL::StringType>(
      ConvertElementType(TensorType_STRING, builder_)));
  EXPECT_TRUE(mlir::isa<mlir::TFL::ResourceType>(
      ConvertElementType(TensorType_RESOURCE, builder_)));
  EXPECT_TRUE(mlir::isa<mlir::TFL::VariantType>(
      ConvertElementType(TensorType_VARIANT, builder_)));
  EXPECT_TRUE(ConvertElementType(TensorType_INT8, builder_).isInteger(8));
}

TEST_F(ConvertTypeTest, ConvertElementTypeToTfTypeReturnsTfTypes) {
  EXPECT_TRUE(mlir::isa<mlir::tf_type::StringType>(
      ConvertElementTypeToTfType(TensorType_STRING, builder_)));
  EXPECT_TRUE(mlir::isa<mlir::tf_type::ResourceType>(
      ConvertElementTypeToTfType(TensorType_RESOURCE, builder_)));
  EXPECT_TRUE(mlir::isa<mlir::tf_type::VariantType>(
      ConvertElementTypeToTfType(TensorType_VARIANT, builder_)));
  EXPECT_TRUE(ConvertElementTypeToTfType(TensorType_FLOAT32, builder_).isF32());
}

TEST_F(ConvertTypeTest, ConvertTypeToTensorTypeAcceptsTflAndTfTypes) {
  EXPECT_EQ(ConvertTypeToTensorType(mlir::TFL::StringType::get(&context_)),
            TensorType_STRING);
  EXPECT_EQ(ConvertTypeToTensorType(mlir::TFL::Quint8Type::get(&context_)),
            TensorType_UINT8);
  EXPECT_EQ(ConvertTypeToTensorType(mlir::TFL::ResourceType::get(&context_)),
            TensorType_RESOURCE);
  EXPECT_EQ(ConvertTypeToTensorType(mlir::TFL::VariantType::get(&context_)),
            TensorType_VARIANT);
  EXPECT_EQ(ConvertTypeToTensorType(mlir::tf_type::StringType::get(&context_)),
            TensorType_STRING);
  EXPECT_EQ(ConvertTypeToTensorType(mlir::tf_type::Quint8Type::get(&context_)),
            TensorType_UINT8);
  EXPECT_EQ(
      ConvertTypeToTensorType(mlir::tf_type::ResourceType::get(&context_)),
      TensorType_RESOURCE);
  EXPECT_EQ(ConvertTypeToTensorType(mlir::tf_type::VariantType::get(&context_)),
            TensorType_VARIANT);
}

}  // namespace
}  // namespace tflite
