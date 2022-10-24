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
// Unit test for TFLite SOFTMAX op.

#include "tensorflow/lite/kernels/internal/reference/softmax.h"

#include <initializer_list>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

class SoftmaxOpModel : public SingleOpModel {
 public:
  SoftmaxOpModel(int batches, int size, float beta)
      : batches_(batches), input_size_(size), beta_(beta) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, beta_).Union());
    BuildInterpreter({{batches_, input_size_}});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  void SetInput(int offset, float* begin, float* end) {
    PopulateTensor(input_, offset, begin, end);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;

  int batches_;
  int input_size_;
  float beta_;
};

TEST(SoftmaxOpTest, SimpleTest) {
  SoftmaxOpModel m(/*batches=*/2, /*size=*/5, /*beta=*/1.0);
  m.SetInput({
      1.0, 2.0, 3.0, 4.0, 5.0,       // b = 0
      -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 0
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {0.011656231, 0.031684921, 0.086128544, 0.234121657, 0.636408647,
           0.636408647, 0.234121657, 0.086128544, 0.031684921, 0.011656231},
          1e-6)));
}

TEST(SoftmaxOpTest, CompareWithTFminiBetaEq1) {
  const int batch_size = 2;
  const int input_size = 5;
  const float beta = 1.0;
  static float input_buffer[] = {
      1.0,  2.0,  3.0,  4.0,  5.0,   // b = 0
      -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 1
  };

  SoftmaxOpModel m(batch_size, input_size, beta);

  m.SetInput(0, input_buffer, input_buffer + input_size * batch_size);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::unique_ptr<float[]> output_buffer(new float[input_size * batch_size]);
  auto input_shape = RuntimeShape({batch_size, 1, 1, input_size});
  SoftmaxParams params;
  params.beta = beta;
  tflite::reference_ops::Softmax(params, input_shape, input_buffer, input_shape,
                                 output_buffer.get());

  std::vector<float> expected;
  expected.insert(expected.end(), output_buffer.get(),
                  output_buffer.get() + input_size * batch_size);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(expected, 1e-6)));
}

TEST(SoftmaxOpTest, CompareWithTFminiBetaNotEq1) {
  const int batch_size = 2;
  const int input_size = 5;
  const float beta = 0.5;
  static float input_buffer[] = {
      1.0,  2.0,  3.0,  4.0,  5.0,   // b = 0
      -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 1
  };

  SoftmaxOpModel m(batch_size, input_size, beta);

  m.SetInput(0, input_buffer, input_buffer + input_size * batch_size);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::unique_ptr<float[]> output_buffer(new float[input_size * batch_size]);
  auto input_shape = RuntimeShape({batch_size, 1, 1, input_size});
  SoftmaxParams params;
  params.beta = beta;
  tflite::reference_ops::Softmax(params, input_shape, input_buffer, input_shape,
                                 output_buffer.get());

  std::vector<float> expected;
  expected.insert(expected.end(), output_buffer.get(),
                  output_buffer.get() + input_size * batch_size);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(expected, 1e-6)));
}

}  // namespace
}  // namespace tflite
