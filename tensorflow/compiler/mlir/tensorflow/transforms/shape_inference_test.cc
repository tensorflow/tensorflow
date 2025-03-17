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
#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"

#include <memory>

#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace TF {
namespace {

TEST(ShapeInferenceTest, CreateMultiThreadedMLIRContext) {
  std::unique_ptr<MLIRContext> ctx = MakeMLIRContextWithThreading();
  EXPECT_TRUE(ctx->isMultithreadingEnabled());
}

TEST(ShapeInferenceTest, CreateSingleThreadedMLIRContext) {
  setenv(kMLIRContextSingleThreadVar, "true", 1);
  std::unique_ptr<MLIRContext> ctx = MakeMLIRContextWithThreading();
  EXPECT_FALSE(ctx->isMultithreadingEnabled());
}

}  // namespace
}  // namespace TF
}  // namespace mlir
