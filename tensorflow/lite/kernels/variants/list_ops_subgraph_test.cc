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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_subgraph_test_util.h"

namespace tflite {
namespace {

TEST_F(ListOpsSubgraphTest, SimpleAddConst) {
  builder_.AddConstSubgraph(&interpreter_.primary_subgraph());

  TfLiteTensor* cst1 = interpreter_.tensor(0);
  ASSERT_THAT(cst1, DimsAre({2}));
  EXPECT_EQ(cst1->data.i32[0], 2);
  EXPECT_EQ(cst1->data.i32[1], 2);

  TfLiteTensor* cst2 = interpreter_.tensor(1);
  ASSERT_THAT(cst2, DimsAre({2}));
  EXPECT_EQ(cst2->data.i32[0], 3);
  EXPECT_EQ(cst2->data.i32[1], 3);

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  TfLiteTensor* out = interpreter_.tensor(2);
  ASSERT_THAT(out, DimsAre({2}));
  EXPECT_EQ(out->data.i32[0], 5);
  EXPECT_EQ(out->data.i32[1], 5);
}

}  // namespace
}  // namespace tflite
