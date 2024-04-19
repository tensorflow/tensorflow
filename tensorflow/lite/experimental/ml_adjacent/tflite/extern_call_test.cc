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
#include "tensorflow/lite/experimental/ml_adjacent/tflite/extern_call.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

class ExternCallModel : public SingleOpModel {
 public:
  ExternCallModel(std::vector<TensorData> inputs,
                  std::vector<TensorData> outputs, uint8_t func_id) {
    std::vector<std::vector<int>> input_shapes;
    for (const auto& data : inputs) {
      input_ids_.push_back(AddInput(data));
      input_shapes.push_back(GetShape(input_ids_.back()));
    }
    for (const auto& data : outputs) {
      output_ids_.push_back(AddOutput(data));
    }
    SetCustomOp("ExternCall", {func_id}, extern_call::Register_EXTERN_CALL);
    BuildInterpreter(input_shapes);
  }

  const TfLiteTensor* Output(int output_id) {
    return interpreter_->tensor(output_ids_[output_id]);
  }

 private:
  std::vector<int> input_ids_;
  std::vector<int> output_ids_;
};

namespace {

// This custom op is a simple wrapper and most of the meaningful logic
// with be tested in the contained `Algo` implementations. It is
// sufficient to just test that calling these `Algo` succeeds.

// TODO(b/290283768) If the complexity if this custom op grows consider
// devising a mechanism allowing for a "mock" `Algo` to call from these tests.

TEST(ExternCallTest, CropFunc) {
  std::vector<TensorData> inputs = {{TensorType_FLOAT32, {1, 5, 5, 1}},
                                    {TensorType_FLOAT64, {}}};
  std::vector<TensorData> output = {{TensorType_FLOAT32, {}}};

  ExternCallModel model(inputs, output, 0);
  model.PopulateTensor<double>(1, {0.5});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_NE(model.Output(0), nullptr);
  ASSERT_THAT(model.Output(0), DimsAre({1, 3, 3, 1}));
}

TEST(ExternCallTest, ResizeTest) {
  std::vector<TensorData> inputs = {{TensorType_FLOAT32, {1, 5, 5, 1}},
                                    {TensorType_UINT32, {2}}};
  std::vector<TensorData> output = {{TensorType_FLOAT32, {}}};

  ExternCallModel model(inputs, output, 1);
  model.PopulateTensor<uint32_t>(1, {3, 3});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_NE(model.Output(0), nullptr);
  ASSERT_THAT(model.Output(0), DimsAre({1, 3, 3, 1}));
}

}  // namespace
}  // namespace tflite
