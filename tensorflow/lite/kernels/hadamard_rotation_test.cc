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

#include <cmath>
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

  // All ones input vector should generate an impulse output, with the first
  // element being the only non-zero element with value of sqrt(size).
  // Here we test a batch of 2 such inputs.
  std::vector<float> ones;
  for (int i = 0; i < size * 2; ++i) {
    ones.push_back(1.0);
  }

  m.SetInput1<float>(ones);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, size * 2));
  std::vector<float> output = m.GetOutput1<float>();
  // First element of both outputs should be sqrt(size).
  EXPECT_FLOAT_EQ(output[0], std::sqrt(size));
  EXPECT_FLOAT_EQ(output[size], std::sqrt(size));
  // All other elements should be zero.
  for (int i = 1; i < size; ++i) {
    EXPECT_FLOAT_EQ(output[i], 0.0);
    EXPECT_FLOAT_EQ(output[i + size], 0.0);
  }
}

// Recursive implementation of the Fast Walsh-Hadamard Transform.
// Runtime is a faster, iterative version of the algorithm, but this may be
// more intuitive.
void recursive_FWHT(float* inout_vector, int len) {
  // Transform the vector in place with the FWHT algorithm of size len.
  float tmp = 0.0;

  if (len == 2) {
    tmp = inout_vector[0];
    inout_vector[0] = (tmp + inout_vector[1]) / std::sqrt(2);
    inout_vector[1] = (tmp - inout_vector[1]) / std::sqrt(2);
  } else {
    int half_len = len / 2;
    recursive_FWHT(inout_vector, half_len);
    recursive_FWHT(inout_vector + half_len, half_len);

    for (int i = 0; i < half_len; i++) {
      tmp = inout_vector[i];
      inout_vector[i] = (tmp + inout_vector[i + half_len]) / std::sqrt(2);
      inout_vector[i + half_len] =
          (tmp - inout_vector[i + half_len]) / std::sqrt(2);
    }
  }
}

TEST_P(HadamardRotationOpTest, RandomInputTest) {
  int size = GetParam();
  // Compare simple recursive FWHT implementation with the TFLite op.
  // Use a batch of 2 random vectors to be transformed.
  BaseHadamardRotationOpModel m(size, {TensorType_FLOAT32, {1, size * 2}},
                                {TensorType_FLOAT32, {1, size * 2}});

  std::mt19937 gen(12345);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  // Container for input vectors.
  std::vector<float> randoms;
  // Initialize both input vectors with random numbers.
  for (int i = 0; i < size * 2; ++i) {
    randoms.push_back(dist(gen));
  }

  m.SetInput1<float>(randoms);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, size * 2));
  std::vector<float> output = m.GetOutput1<float>();
  // Transform both input vectors with the recursive FWHT algorithm.
  recursive_FWHT(randoms.data(), size);
  recursive_FWHT(randoms.data() + size, size);
  // After transforming both vectors in-place, they should be the same as the
  // output of the TFLite op.
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
