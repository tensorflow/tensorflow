/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_TABLE();

namespace {

using ::testing::ElementsAreArray;

class TableOpModel : public SingleOpModel {
 public:
  TableOpModel(const TensorData& input, const TensorData& table,
               const TensorData& output) {
    input_ = AddInput(input);
    table_ = AddInput(table);
    output_ = AddOutput(output);
    SetCustomOp("Table", {}, Register_TABLE);
    BuildInterpreter({GetShape(input_), GetShape(table_)});
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

  int input() { return input_; }
  int table() { return table_; }
  int output() { return output_; }

 protected:
  int input_;
  int table_;
  int output_;
};

// A LUT of 256 values is used in the int8 case. For the int16 case a 513 LUT is
// used but as the last value is only used for interpolation we only have 512
// quantized steps.
template <typename T>
inline float GetLUTTolerance(float input_min, float input_max, float output_min,
                             float output_max) {
  static_assert(
      std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value,
      "T must be an int8_t or int16_t.");

  const float range_sum = (input_max - input_min) + (output_max - output_min);
  if (std::is_same<T, int8_t>::value) {
    return range_sum / 256.0f;
  } else {
    return range_sum / 512.0f;
  }
}

template <typename T>
void TableWithExpLUTTest() {
  float input_min = -0.5f;
  float input_max = 0.8f;
  // Use symmetric inputs for int16 cases, nudge max for null zero-point
  if (std::is_same<T, int16_t>::value) {
    input_min = -0.8f;
    input_max = 0.8f * std::numeric_limits<T>::max() /
                static_cast<float>(std::numeric_limits<T>::max() + 1);
  }

  float output_min = 0.0f;
  float output_max = 2.4f;
  // Use symmetric outputs  for int16 cases, nudge max for null zero-point
  if (std::is_same<T, int16_t>::value) {
    output_min = -2.4f;
    output_max = 2.4f * std::numeric_limits<T>::max() /
                 static_cast<float>(std::numeric_limits<T>::max() + 1);
  }

  const float kQuantizedTolerance =
      GetLUTTolerance<T>(input_min, input_max, output_min, output_max);

  TableOpModel m({GetTensorType<T>(), {1, 2, 3, 1}, input_min, input_max},
                 {GetTensorType<T>(), {LUTSize<T>()}},
                 {GetTensorType<T>(), {}, output_min, output_max});
  T table[LUTSize<T>()];
  LUTPopulate<T>(
      m.GetScale(m.input()), m.GetZeroPoint(m.input()), m.GetScale(m.output()),
      m.GetZeroPoint(m.output()), [](float v) { return std::exp(v); }, table);

  m.QuantizeAndPopulate<T>(m.input(), {-0.5f, -0.2f, 0.0f, 0.1f, 0.3f, 0.8f});
  m.PopulateTensor<T>(m.table(), 0, table, table + LUTSize<T>());
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<T>(),
              ElementsAreArray(ArrayFloatNear(
                  {std::exp(-0.5f), std::exp(-0.2f), std::exp(0.0f),
                   std::exp(0.1f), std::exp(0.3f), std::exp(0.8f)},
                  kQuantizedTolerance)));
}

TEST(TableOpTest, Int8ExpLUT) { TableWithExpLUTTest<int8_t>(); }

TEST(TableOpTest, Int16ExpLUT) { TableWithExpLUTTest<int16_t>(); }

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
