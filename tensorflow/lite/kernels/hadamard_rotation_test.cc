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
using ::testing::ElementsAreArray;

class BaseHadamardRotationOpModel : public SingleOpModel {
 public:
  BaseHadamardRotationOpModel(const TensorData& input,
                              const TensorData& output) {
    input1_ = AddInput(input);
    output1_ = AddOutput(output);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("hadamard_size", 2);
      auto start = fbb.StartVector("random_binary_vector");
      fbb.Add(1);
      fbb.Add(1);
      fbb.EndVector(start, false, false);
    });
    fbb.Finish();
    SetCustomOp("aeq.hadamard_rotation", fbb.GetBuffer(),
                Register_HADAMARD_ROTATION);
    BuildInterpreter({GetShape(input1_)});
  }

  int input1() { return input1_; }

  template <class T>
  void SetInput1(std::initializer_list<T> data) {
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

TEST(HadamardRotationOpTest, BasicTest) {
  BaseHadamardRotationOpModel m({TensorType_FLOAT32, {1, 4}},
                                {TensorType_FLOAT32, {1, 4}});

  m.SetInput1<float>({
      1.0,
      1.0,
      1.0,
      1.0,
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, 4));
  EXPECT_THAT(
      m.GetOutput1<float>(),
      // Equals to [FWHT([1.0, 1.0]), FWHT([1.0, 1.0])] normalized by sqrt(2)
      ElementsAreArray(ArrayFloatNear({1.41421354, 0, 1.41421354, 0}, 1e-6)));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
