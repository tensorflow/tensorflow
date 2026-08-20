/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_MFCC();

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class BaseMfccOpModel : public SingleOpModel {
 public:
  BaseMfccOpModel(const TensorData& input1, const TensorData& input2,
                  const TensorData& output, int dct_coefficient_count = 13,
                  bool allocate_and_delegate = true) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("upper_frequency_limit", 4000);
      fbb.Int("lower_frequency_limit", 20);
      fbb.Int("filterbank_channel_count", 40);
      fbb.Int("dct_coefficient_count", dct_coefficient_count);
    });
    fbb.Finish();
    SetCustomOp("Mfcc", fbb.GetBuffer(), Register_MFCC);

    BuildInterpreter({GetShape(input1_), GetShape(input2_)}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true,
                     /*allocate_and_delegate=*/allocate_and_delegate);
  }

  int input1() { return input1_; }
  int input2() { return input2_; }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(MfccOpTest, SimpleTest) {
  BaseMfccOpModel m({TensorType_FLOAT32, {1, 1, 513}}, {TensorType_INT32, {1}},
                    {TensorType_FLOAT32, {}});

  std::vector<float> data(513);
  for (int i = 0; i < data.size(); ++i) {
    data[i] = i + 1;
  }
  m.PopulateTensor<float>(m.input1(), 0, data.data(),
                          data.data() + data.size());
  m.PopulateTensor<int>(m.input2(), {22050});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::vector<int> output_shape = m.GetOutputShape();
  EXPECT_THAT(output_shape, ElementsAre(1, 1, 13));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {29.13970072, -6.41568601, -0.61903012, -0.96778652, -0.26819878,
           -0.40907028, -0.15614748, -0.23203119, -0.10481487, -0.1543029,
           -0.0769791, -0.10806114, -0.06047613},
          1e-3)));
}

TEST(MfccOpTest, ScalarInputRateTest) {
  BaseMfccOpModel m({TensorType_FLOAT32, {1, 1, 513}}, {TensorType_INT32, {}},
                    {TensorType_FLOAT32, {}});

  std::vector<float> data(513);
  for (int i = 0; i < data.size(); ++i) {
    data[i] = i + 1;
  }
  m.PopulateTensor<float>(m.input1(), 0, data.data(),
                          data.data() + data.size());
  m.PopulateTensor<int>(m.input2(), {22050});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::vector<int> output_shape = m.GetOutputShape();
  EXPECT_THAT(output_shape, ElementsAre(1, 1, 13));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {29.13970072, -6.41568601, -0.61903012, -0.96778652, -0.26819878,
           -0.40907028, -0.15614748, -0.23203119, -0.10481487, -0.1543029,
           -0.0769791, -0.10806114, -0.06047613},
          1e-3)));
}

TEST(MfccOpTest, MultiChannelTest) {
  BaseMfccOpModel m({TensorType_FLOAT32, {2, 2, 513}}, {TensorType_INT32, {1}},
                    {TensorType_FLOAT32, {}});

  std::vector<float> data(513 * 4);
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 513; ++i) {
      data[j * 513 + i] = i + 1;
    }
  }
  m.PopulateTensor<float>(m.input1(), 0, data.data(),
                          data.data() + data.size());
  m.PopulateTensor<int>(m.input2(), {22050});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::vector<int> output_shape = m.GetOutputShape();
  EXPECT_THAT(output_shape, ElementsAre(2, 2, 13));
  std::vector<float> expected = {
      29.13970072, -6.41568601, -0.61903012, -0.96778652, -0.26819878,
      -0.40907028, -0.15614748, -0.23203119, -0.10481487, -0.1543029,
      -0.0769791,  -0.10806114, -0.06047613};
  std::vector<float> expected_all;
  for (int j = 0; j < 4; ++j) {
    expected_all.insert(expected_all.end(), expected.begin(), expected.end());
  }
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(expected_all, 1e-3)));
}

TEST(MfccOpTest, InvalidDctCoefficientCountFails) {
  BaseMfccOpModel m({TensorType_FLOAT32, {1, 1, 513}}, {TensorType_INT32, {1}},
                    {TensorType_FLOAT32, {}}, /*dct_coefficient_count=*/0,
                    /*allocate_and_delegate=*/false);
  EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
}

TEST(MfccOpTest, InvalidInputDimensionFails) {
  BaseMfccOpModel m({TensorType_FLOAT32, {0, 1, 513}}, {TensorType_INT32, {1}},
                    {TensorType_FLOAT32, {}}, /*dct_coefficient_count=*/13,
                    /*allocate_and_delegate=*/false);
  EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
