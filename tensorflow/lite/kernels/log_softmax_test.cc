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
// Unit test for TFLite LOG_SOFTMAX op.

#include <initializer_list>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

class LogSoftmaxOpModel : public SingleOpModel {
 public:
  LogSoftmaxOpModel(int batches, int size)
      : batches_(batches), input_size_(size) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_LOG_SOFTMAX, BuiltinOptions_LogSoftmaxOptions,
                 CreateLogSoftmaxOptions(builder_).Union());
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
};

TEST(LogSoftmaxOpTest, SimpleTest) {
  LogSoftmaxOpModel m(/*batches=*/2, /*size=*/5);
  m.SetInput({
      1.0, 2.0, 3.0, 4.0, 5.0,       // b = 0
      -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 1
  });

  m.Invoke();

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {-4.45191431, -3.45191431, -2.45191431, -1.45191443, -0.4519144,
           -0.4519144, -1.45191443, -2.45191431, -3.45191431, -4.45191431},
          1e-6)));
}

TEST(LogSoftmaxOpTest, CompareWithTFmini) {
  const int batch_size = 2;
  const int input_size = 5;
  static float input_buffer[] = {
      1.0,  2.0,  3.0,  4.0,  5.0,   // b = 0
      -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 1
  };

  LogSoftmaxOpModel m(batch_size, input_size);

  m.SetInput(0, input_buffer, input_buffer + input_size * batch_size);

  m.Invoke();

  std::unique_ptr<float[]> output_buffer(new float[input_size * batch_size]);
  auto input_shape = RuntimeShape({batch_size, 1, 1, input_size});
  SoftmaxParams params;
  tflite::reference_ops::LogSoftmax(params, input_shape, input_buffer,
                                    input_shape, output_buffer.get());

  std::vector<float> expected;
  expected.insert(expected.end(), output_buffer.get(),
                  output_buffer.get() + input_size * batch_size);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(expected, 1e-6)));
}

}  // namespace
}  // namespace tflite
