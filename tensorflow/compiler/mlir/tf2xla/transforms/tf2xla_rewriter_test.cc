/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/transforms/tf2xla_rewriter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"

namespace mlir {
namespace mhlo {

static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1438 : i32}} {
  func.func @main() -> tensor<10000000xf64> attributes {tf.entry_function = {control_outputs = "", inputs = "_arg0", outputs = "_retval0"}} {
    %cst = "tf.Const"() {value = dense<10000000> : tensor<1xi32>} : () -> tensor<1xi32>
    %0 = "tf.TruncatedNormal"(%cst) {_XlaHasReferenceVars = false, device = "/job:localhost/replica:0/task:0/device:XLA_CPU:0", seed = 0 : i64, seed2 = 0 : i64} : (tensor<1xi32>) -> tensor<10000000xf64>
    return %0 : tensor<10000000xf64>
  }
})";

class Tf2XlaRewriterTest : public ::testing::Test {};

TEST_F(Tf2XlaRewriterTest, LegalizesOp) {
  MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      OwningOpRef<ModuleOp> module,
      test::GetMlirModuleFromString(kMlirModuleStr, &context));
}

}  // namespace mhlo
}  // namespace mlir
