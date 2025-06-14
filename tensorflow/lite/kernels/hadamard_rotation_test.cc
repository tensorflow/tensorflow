/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include <initializer_list>
#include <random>
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

TfLiteRegistration* Register_HADAMARD_ROTATION();

namespace {

using ::testing::ElementsAre;
using ::testing::Values;

class BaseHadamardRotationOpModel : public SingleOpModel {
 public:
  BaseHadamardRotationOpModel(const int size, const TensorData& input,
                              const TensorData& output) {
    input1_ = AddInput(input);
    output1_ = AddOutput(output);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("hadamard_size", size);
      auto start = fbb.StartVector("random_binary_vector");
      for (int i = 0; i < size; ++i) {
        fbb.Add(1);
      }
      fbb.EndVector(start, false, false);
    });
    fbb.Finish();
    SetCustomOp("aeq.hadamard_rotation", fbb.GetBuffer(),
                Register_HADAMARD_ROTATION);
    BuildInterpreter({GetShape(input1_)});
  }

  int input1() { return input1_; }

  template <class T>
  void SetInput1(std::vector<T> data) {
    PopulateTensor<T>(input1_, data);
  }

  template <class T>
  std::vector<T> GetOutput1() {
    return ExtractVector<T>(output1_);
  }

  std::vector<int> GetOutputShape1() { return GetTensorShape(output1_); }

 protected:
  int input1_;
  int output1_;
};

class HadamardRotationOpTest : public ::testing::TestWithParam<int> {};

TEST_P(HadamardRotationOpTest, BasicTest) {
  int size = GetParam();
  // Use a batch of 2 vectors to be transformed.
  BaseHadamardRotationOpModel m(size, {TensorType_FLOAT32, {1, size * 2}},
                                {TensorType_FLOAT32, {1, size * 2}});

  std::vector<float> ones;
  for (int i = 0; i < size * 2; ++i) {
    ones.push_back(1.0);
  }

  m.SetInput1<float>(ones);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, size * 2));
  std::vector<float> output = m.GetOutput1<float>();
  EXPECT_FLOAT_EQ(output[0], std::sqrt(size));
  EXPECT_FLOAT_EQ(output[size], std::sqrt(size));
  for (int i = 1; i < size; ++i) {
    EXPECT_FLOAT_EQ(output[i], 0.0);
    EXPECT_FLOAT_EQ(output[i + size], 0.0);
  }
}

void hadamard(float* p, int len) {
  float tmp = 0.0;

  if (len == 2) {
    tmp = p[0];
    p[0] = (tmp + p[1]) / std::sqrt(2);
    p[1] = (tmp - p[1]) / std::sqrt(2);
  } else {
    hadamard(p, len / 2);
    hadamard(p + len / 2, len / 2);

    for (int i = 0; i < len / 2; i++) {
      tmp = p[i];
      p[i] = (tmp + p[i + len / 2]) / std::sqrt(2);
      p[i + len / 2] = (tmp - p[i + len / 2]) / std::sqrt(2);
    }
  }
}

TEST_P(HadamardRotationOpTest, RandomInputTest) {
  int size = GetParam();
  // Use a batch of 2 vectors to be transformed.
  BaseHadamardRotationOpModel m(size, {TensorType_FLOAT32, {1, size * 2}},
                                {TensorType_FLOAT32, {1, size * 2}});

  std::mt19937 gen(12345);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  std::vector<float> randoms;
  for (int i = 0; i < size * 2; ++i) {
    randoms.push_back(dist(gen));
  }

  m.SetInput1<float>(randoms);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, size * 2));
  std::vector<float> output = m.GetOutput1<float>();
  hadamard(randoms.data(), size);
  hadamard(randoms.data() + size, size);
  for (int i = 0; i < size * 2; ++i) {
    EXPECT_NEAR(output[i], randoms[i], 1e-5);
  }
}

INSTANTIATE_TEST_SUITE_P(HadamardSizes, HadamardRotationOpTest,
                         Values(4, 16, 64));

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
