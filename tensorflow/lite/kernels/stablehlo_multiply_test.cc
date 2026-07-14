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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

class MultiplyOpModel : public SingleOpModel {
 public:
  MultiplyOpModel(const TensorData& input1, const TensorData& input2,
                  const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_STABLEHLO_MULTIPLY, BuiltinOptions_NONE, 0);
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(StablehloElementwise, MultiplyWorks) {
  MultiplyOpModel model({TensorType_FLOAT32, {1, 2, 2, 1}},
                        {TensorType_FLOAT32, {1, 2, 2, 1}},
                        {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {1.2, 2.5, -1.2, 1});
  model.PopulateTensor<float>(model.input2(), {0.1, 3, 2, 0.5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {0.12, 7.5, -2.4, 0.5};
  std::vector<float> actual_values = model.GetOutput();
  ASSERT_EQ(actual_values.size(), expected_values.size());
  for (int idx = 0; idx < expected_values.size(); ++idx) {
    ASSERT_NEAR(actual_values[idx], expected_values[idx], 1e-6);
  }
}

}  // namespace
}  // namespace tflite
