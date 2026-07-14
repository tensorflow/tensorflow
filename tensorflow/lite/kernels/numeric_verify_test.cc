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
#include <string.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "Eigen/Core"  // from @eigen_archive
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace ops {
namespace custom {

TfLiteRegistration* Register_NUMERIC_VERIFY();

}  // namespace custom
}  // namespace ops

namespace {

class NumericVerifyOpModel : public SingleOpModel {
 public:
  NumericVerifyOpModel(TensorType type, std::initializer_list<int> shape,
                       float scale, int32_t zero_point, int version,
                       float tolerance = 5.0, bool log_if_failed = true) {
    const TensorData input_tensor_data = {type, shape, 0, 0, scale, zero_point};
    input_ = AddInput(input_tensor_data);
    ref_ = AddInput({TensorType_FLOAT32, shape});
    // The output tensor has the same shape with that of the input tensor.
    output_ = AddOutput({TensorType_FLOAT32, shape});

    std::vector<uint8_t> custom_options(sizeof(float));

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Float("tolerance", tolerance);
      fbb.Bool("log_if_failed", log_if_failed);
    });
    fbb.Finish();

    SetCustomOp("NUMERIC_VERIFY", fbb.GetBuffer(),
                ops::custom::Register_NUMERIC_VERIFY);

    BuildInterpreter({GetShape(input_), GetShape(ref_)});
  }

  template <typename T>
  void SetInputs(std::initializer_list<T> data,
                 std::initializer_list<float> ref_data) {
    PopulateTensor(input_, data);
    PopulateTensor(ref_, ref_data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int ref_;
  int output_;
};

TEST(NumericVerifyOpTest, Uint8) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  NumericVerifyOpModel m(TensorType_UINT8, {2, 5}, 0.5, 127, 1);

  m.SetInputs<uint8_t>({0, 1, 2, 3, 4, 251, 252, 253, 254, 255},
                       {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  EXPECT_EQ(m.Invoke(), kTfLiteOk);
}

TEST(NumericVerifyOpTest, Int8) {
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  NumericVerifyOpModel m(TensorType_INT8, {2, 5}, 0.5, -1, 2);

  m.SetInputs<int8_t>({-128, -127, -126, -125, -124, 123, 124, 125, 126, 127},
                      {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  EXPECT_EQ(m.Invoke(), kTfLiteOk);
}

TEST(NumericVerifyOpTest, Float16) {
  NumericVerifyOpModel m(TensorType_FLOAT16, {2, 3}, 1.0f, 0, 3,
                         /*tolerance=*/0.1f);

  std::vector<Eigen::half> half{Eigen::half{-535.54f}, Eigen::half{-100.0f},
                                Eigen::half{-1.0f},    Eigen::half{0.f},
                                Eigen::half{1.0f},     Eigen::half{100.32f}};
  m.PopulateTensor(0, 0, reinterpret_cast<TfLiteFloat16*>(half.data()),
                   reinterpret_cast<TfLiteFloat16*>(half.data()) + half.size());
  m.PopulateTensor(1, {-535.54f, -100.0f, -1.0f, 0.f, 1.0f, 100.32f});
  EXPECT_EQ(m.Invoke(), kTfLiteOk);
}

TEST(NumericVerifyOpTest, Int16) {
  NumericVerifyOpModel m(TensorType_INT16, {2, 5}, 0.5, -1, 4);
  m.SetInputs<int16_t>(
      {-130, -127, -126, -125, -124, 123, 124, 125, 126, 130},
      {-64.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 65.5});
  EXPECT_EQ(m.Invoke(), kTfLiteOk);
}

TEST(NumericVerifyOpFailedTest, Int8) {
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  NumericVerifyOpModel m(TensorType_INT8, {2, 5}, 0.5, -1, 2);

  // The 5th element is set to 0.
  m.SetInputs<int8_t>({-128, -127, -126, -125, -124, 0, 124, 125, 126, 127},
                      {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  EXPECT_EQ(m.Invoke(), kTfLiteError);
}

TEST(NumericVerifyOpDebugModeTest, Int8) {
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  NumericVerifyOpModel m(TensorType_INT8, {2, 5}, 0.5, -1, 2, 5.0, false);

  // The 5th element is set to 0.
  m.SetInputs<int8_t>({-128, -127, -126, -125, -124, 0, 124, 125, 126, 127},
                      {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  EXPECT_EQ(m.Invoke(), kTfLiteOk);
  // The 5th element has discrepancy -61.5 (=dequantized - reference=0-(61.5)).
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear({0, 0, 0, 0, 0, -61.5, 0, 0, 0, 0})));
}
}  // namespace
}  // namespace tflite
