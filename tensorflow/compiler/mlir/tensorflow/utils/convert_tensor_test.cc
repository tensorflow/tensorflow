/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"

#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

TEST(ConvertTypeToTensorTypeTest, UnrankedTensorType) {
  mlir::MLIRContext context;
  mlir::Builder b(&context);

  PartialTensorShape output_shape =
      ConvertTypeToTensorShape(mlir::UnrankedTensorType::get(b.getF32Type()));
  EXPECT_TRUE(output_shape.IsIdenticalTo(PartialTensorShape()));
}

TEST(ConvertTypeToTensorTypeTest, NonFullyDefinedRankedTensorType) {
  mlir::MLIRContext context;
  mlir::Builder b(&context);

  PartialTensorShape output_shape = ConvertTypeToTensorShape(
      mlir::RankedTensorType::get({-1, 2, 3}, b.getF32Type()));
  EXPECT_TRUE(output_shape.IsIdenticalTo(PartialTensorShape({-1, 2, 3})));
}

TEST(ConvertTypeToTensorTypeTest, FullyDefinedRankedTensorType) {
  mlir::MLIRContext context;
  mlir::Builder b(&context);

  PartialTensorShape output_shape = ConvertTypeToTensorShape(
      mlir::RankedTensorType::get({1, 2, 3}, b.getF32Type()));
  EXPECT_TRUE(output_shape.IsIdenticalTo(PartialTensorShape({1, 2, 3})));
}

TEST(ConvertTypeToTensorTypeTest, ScalarTensorType) {
  mlir::MLIRContext context;
  mlir::Builder b(&context);

  PartialTensorShape output_shape = ConvertTypeToTensorShape(b.getF32Type());
  EXPECT_TRUE(output_shape.IsIdenticalTo(TensorShape()));
}

}  // namespace
}  // namespace tensorflow
