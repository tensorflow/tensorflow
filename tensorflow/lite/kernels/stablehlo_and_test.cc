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

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

class AndOpModel : public SingleOpModel {
 public:
  AndOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_STABLEHLO_AND, BuiltinOptions_NONE, 0);
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(StablehloElementwise, AndInt32) {
  AndOpModel model({TensorType_INT32, {1, 2, 2, 1}},
                   {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {2, 3, 7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {4, 5, 7, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int32_t>(), ElementsAre(0, 1, 7, 0));
}

TEST(StablehloElementwise, AndInt8) {
  AndOpModel model({TensorType_INT8, {1, 3, 1}}, {TensorType_INT8, {1, 3, 1}},
                   {TensorType_INT8, {}});
  model.PopulateTensor<int8_t>(model.input1(), {7, -8, -8});
  model.PopulateTensor<int8_t>(model.input2(), {0, 7, -8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int8_t>(), ElementsAre(0, 0, -8));
}

TEST(StablehloElementwise, AndInt16) {
  AndOpModel model({TensorType_INT16, {1, 1, 3}}, {TensorType_INT16, {1, 1, 3}},
                   {TensorType_INT16, {}});
  model.PopulateTensor<int16_t>(model.input1(), {32767, -32768, -32768});
  model.PopulateTensor<int16_t>(model.input2(), {32767, -32768, -32768});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int16_t>(), ElementsAre(32767, -32768, -32768));
}

TEST(StablehloElementwise, AndBool) {
  AndOpModel model({TensorType_BOOL, {2, 1, 2, 1}},
                   {TensorType_BOOL, {2, 1, 2, 1}}, {TensorType_BOOL, {}});
  model.PopulateTensor<bool>(model.input1(), {false, false, true, true});
  model.PopulateTensor<bool>(model.input2(), {false, true, false, true});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<bool>(), ElementsAre(false, false, false, true));
}

}  // namespace
}  // namespace tflite
