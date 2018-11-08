/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {

namespace ops {
namespace custom {
TfLiteRegistration* Register_PREDICT();

namespace {

using ::testing::ElementsAreArray;

class PredictOpModel : public SingleOpModel {
 public:
  PredictOpModel(std::initializer_list<int> input_signature_shape,
                 std::initializer_list<int> key_shape,
                 std::initializer_list<int> labelweight_shape, int num_output,
                 float threshold) {
    input_signature_ = AddInput(TensorType_INT32);
    model_key_ = AddInput(TensorType_INT32);
    model_label_ = AddInput(TensorType_INT32);
    model_weight_ = AddInput(TensorType_FLOAT32);
    output_label_ = AddOutput(TensorType_INT32);
    output_weight_ = AddOutput(TensorType_FLOAT32);

    std::vector<uint8_t> predict_option;
    writeInt32(num_output, &predict_option);
    writeFloat32(threshold, &predict_option);
    SetCustomOp("Predict", predict_option, Register_PREDICT);
    BuildInterpreter({{input_signature_shape, key_shape, labelweight_shape,
                       labelweight_shape}});
  }

  void SetInputSignature(std::initializer_list<int> data) {
    PopulateTensor<int>(input_signature_, data);
  }

  void SetModelKey(std::initializer_list<int> data) {
    PopulateTensor<int>(model_key_, data);
  }

  void SetModelLabel(std::initializer_list<int> data) {
    PopulateTensor<int>(model_label_, data);
  }

  void SetModelWeight(std::initializer_list<float> data) {
    PopulateTensor<float>(model_weight_, data);
  }

  std::vector<int> GetLabel() { return ExtractVector<int>(output_label_); }
  std::vector<float> GetWeight() {
    return ExtractVector<float>(output_weight_);
  }

  void writeFloat32(float value, std::vector<uint8_t>* data) {
    union {
      float v;
      uint8_t r[4];
    } float_to_raw;
    float_to_raw.v = value;
    for (unsigned char i : float_to_raw.r) {
      data->push_back(i);
    }
  }

  void writeInt32(int32_t value, std::vector<uint8_t>* data) {
    union {
      int32_t v;
      uint8_t r[4];
    } int32_to_raw;
    int32_to_raw.v = value;
    for (unsigned char i : int32_to_raw.r) {
      data->push_back(i);
    }
  }

 private:
  int input_signature_;
  int model_key_;
  int model_label_;
  int model_weight_;
  int output_label_;
  int output_weight_;
};

TEST(PredictOpTest, AllLabelsAreValid) {
  PredictOpModel m({4}, {5}, {5, 2}, 2, 0.0001);
  m.SetInputSignature({1, 3, 7, 9});
  m.SetModelKey({1, 2, 4, 6, 7});
  m.SetModelLabel({11, 12, 11, 12, 11, 12, 11, 12, 11, 12});
  m.SetModelWeight({0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2});
  m.Invoke();
  EXPECT_THAT(m.GetLabel(), ElementsAreArray({12, 11}));
  EXPECT_THAT(m.GetWeight(), ElementsAreArray(ArrayFloatNear({0.1, 0.05})));
}

TEST(PredictOpTest, MoreLabelsThanRequired) {
  PredictOpModel m({4}, {5}, {5, 2}, 1, 0.0001);
  m.SetInputSignature({1, 3, 7, 9});
  m.SetModelKey({1, 2, 4, 6, 7});
  m.SetModelLabel({11, 12, 11, 12, 11, 12, 11, 12, 11, 12});
  m.SetModelWeight({0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2});
  m.Invoke();
  EXPECT_THAT(m.GetLabel(), ElementsAreArray({12}));
  EXPECT_THAT(m.GetWeight(), ElementsAreArray(ArrayFloatNear({0.1})));
}

TEST(PredictOpTest, OneLabelDoesNotPassThreshold) {
  PredictOpModel m({4}, {5}, {5, 2}, 2, 0.07);
  m.SetInputSignature({1, 3, 7, 9});
  m.SetModelKey({1, 2, 4, 6, 7});
  m.SetModelLabel({11, 12, 11, 12, 11, 12, 11, 12, 11, 12});
  m.SetModelWeight({0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2});
  m.Invoke();
  EXPECT_THAT(m.GetLabel(), ElementsAreArray({12, -1}));
  EXPECT_THAT(m.GetWeight(), ElementsAreArray(ArrayFloatNear({0.1, 0})));
}

TEST(PredictOpTest, NoneLabelPassThreshold) {
  PredictOpModel m({4}, {5}, {5, 2}, 2, 0.6);
  m.SetInputSignature({1, 3, 7, 9});
  m.SetModelKey({1, 2, 4, 6, 7});
  m.SetModelLabel({11, 12, 11, 12, 11, 12, 11, 12, 11, 12});
  m.SetModelWeight({0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2});
  m.Invoke();
  EXPECT_THAT(m.GetLabel(), ElementsAreArray({-1, -1}));
  EXPECT_THAT(m.GetWeight(), ElementsAreArray(ArrayFloatNear({0, 0})));
}

TEST(PredictOpTest, OnlyOneLabelGenerated) {
  PredictOpModel m({4}, {5}, {5, 2}, 2, 0.0001);
  m.SetInputSignature({1, 3, 7, 9});
  m.SetModelKey({1, 2, 4, 6, 7});
  m.SetModelLabel({11, 0, 11, 0, 11, 0, 11, 0, 11, 0});
  m.SetModelWeight({0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0});
  m.Invoke();
  EXPECT_THAT(m.GetLabel(), ElementsAreArray({11, -1}));
  EXPECT_THAT(m.GetWeight(), ElementsAreArray(ArrayFloatNear({0.05, 0})));
}

TEST(PredictOpTest, NoLabelGenerated) {
  PredictOpModel m({4}, {5}, {5, 2}, 2, 0.0001);
  m.SetInputSignature({5, 3, 7, 9});
  m.SetModelKey({1, 2, 4, 6, 7});
  m.SetModelLabel({11, 0, 11, 0, 11, 0, 11, 0, 0, 0});
  m.SetModelWeight({0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetLabel(), ElementsAreArray({-1, -1}));
  EXPECT_THAT(m.GetWeight(), ElementsAreArray(ArrayFloatNear({0, 0})));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
