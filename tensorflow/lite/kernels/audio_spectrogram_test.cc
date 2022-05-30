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

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_AUDIO_SPECTROGRAM();

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class BaseAudioSpectrogramOpModel : public SingleOpModel {
 public:
  BaseAudioSpectrogramOpModel(const TensorData& input1,
                              const TensorData& output, int window_size,
                              int stride, bool magnitude_squared) {
    input1_ = AddInput(input1);
    output_ = AddOutput(output);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("window_size", window_size);
      fbb.Int("stride", stride);
      fbb.Bool("magnitude_squared", magnitude_squared);
    });
    fbb.Finish();
    SetCustomOp("AudioSpectrogram", fbb.GetBuffer(),
                Register_AUDIO_SPECTROGRAM);
    BuildInterpreter({GetShape(input1_)});
  }

  int input1() { return input1_; }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int output_;
};

TEST(BaseAudioSpectrogramOpModel, NonSquaredTest) {
  BaseAudioSpectrogramOpModel m({TensorType_FLOAT32, {8, 1}},
                                {TensorType_FLOAT32, {}}, 8, 1, false);
  m.PopulateTensor<float>(m.input1(),
                          {-1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::vector<int> output_shape = m.GetOutputShape();
  EXPECT_EQ(3, output_shape.size());
  EXPECT_THAT(output_shape, ElementsAre(1, 1, 5));

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {0.0f, 1.0f, 2.0f, 1.0f, 0.0f}, 1e-3)));
}

TEST(SpectrogramOpTest, SquaredTest) {
  BaseAudioSpectrogramOpModel m({TensorType_FLOAT32, {8, 1}},
                                {TensorType_FLOAT32, {}}, 8, 1, true);
  m.PopulateTensor<float>(m.input1(),
                          {-1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::vector<int> output_shape = m.GetOutputShape();
  EXPECT_EQ(3, output_shape.size());
  EXPECT_THAT(output_shape, ElementsAre(1, 1, 5));

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {0.f, 1.f, 4.f, 1.f, 0.f}, 1e-3)));
}

TEST(SpectrogramOpTest, StrideTest) {
  BaseAudioSpectrogramOpModel m({TensorType_FLOAT32, {10, 1}},
                                {TensorType_FLOAT32, {}}, 8, 2, true);
  m.PopulateTensor<float>(m.input1(), {-1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f,
                                       1.0f, 0.0f, 1.0f, 0.0f});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::vector<int> output_shape = m.GetOutputShape();
  EXPECT_THAT(output_shape, ElementsAre(1, 2, 5));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {0, 1, 4, 1, 0, 1, 2, 1, 2, 1}, 1e-3)));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
