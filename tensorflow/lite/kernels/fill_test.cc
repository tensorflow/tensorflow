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
#include <stdint.h>

#include <initializer_list>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename dims_type, typename value_type>
class FillOpModel : public SingleOpModel {
 public:
  explicit FillOpModel(TensorType dims_tensor_type,
                       std::initializer_list<int> dims_shape,
                       std::initializer_list<dims_type> dims_data,
                       value_type value, TestType input_tensor_types) {
    if (input_tensor_types == TestType::kDynamic) {
      dims_ = AddInput(dims_tensor_type);
      value_ = AddInput(GetTensorType<value_type>());
    } else {
      dims_ = AddConstInput(dims_tensor_type, dims_data, dims_shape);
      value_ = AddConstInput(GetTensorType<value_type>(), {value}, {});
    }
    output_ = AddOutput(GetTensorType<value_type>());
    SetBuiltinOp(BuiltinOperator_FILL, BuiltinOptions_FillOptions,
                 CreateFillOptions(builder_).Union());
    BuildInterpreter({dims_shape, {}});

    if (input_tensor_types == TestType::kDynamic) {
      if (dims_data.size() > 0) {
        PopulateTensor<dims_type>(dims_, dims_data);
      }
      PopulateTensor<value_type>(value_, {value});
    }
  }

  std::vector<value_type> GetOutput() {
    return ExtractVector<value_type>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int dims_;
  int value_;
  int output_;
};

template <typename dims_type, typename quant_type>
class QuantizedFillOpModel : public SingleOpModel {
 public:
  explicit QuantizedFillOpModel(TensorType dims_tensor_type,
                                std::initializer_list<int> dims_shape,
                                std::initializer_list<dims_type> dims_data,
                                const TensorData& tensor_data, float value) {
    dims_ = AddInput(dims_tensor_type);
    value_ = AddInput(tensor_data);
    output_ = AddOutput(tensor_data);
    SetBuiltinOp(BuiltinOperator_FILL, BuiltinOptions_FillOptions,
                 CreateFillOptions(builder_).Union());
    BuildInterpreter({dims_shape, {}});

    if (dims_data.size() > 0) {
      PopulateTensor<dims_type>(dims_, dims_data);
    }
    QuantizeAndPopulate<quant_type>(value_, {value});
  }

  std::vector<quant_type> GetOutput() {
    return ExtractVector<quant_type>(output_);
  }
  std::vector<float> GetDequantizedOutput() {
    TfLiteTensor* t = interpreter_->tensor(output_);
    return Dequantize(GetOutput(), t->params.scale, t->params.zero_point);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int dims_;
  int value_;
  int output_;
};

class FillOpTest : public ::testing::TestWithParam<TestType> {};

TEST_P(FillOpTest, FillInt32) {
  FillOpModel<int32_t, int32_t> m(TensorType_INT32, {2}, {2, 3}, -11,
                                  GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-11, -11, -11, -11, -11, -11}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
}

TEST_P(FillOpTest, FillInt64) {
  FillOpModel<int64_t, int64_t> m(TensorType_INT64, {2}, {2, 4}, 1LL << 45,
                                  GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1LL << 45, 1LL << 45, 1LL << 45, 1LL << 45,
                                1LL << 45, 1LL << 45, 1LL << 45, 1LL << 45}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST_P(FillOpTest, FillFloat) {
  FillOpModel<int64_t, float> m(TensorType_INT64, {3}, {2, 2, 2}, 4.0,
                                GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST_P(FillOpTest, FillFloatInt32Dims) {
  FillOpModel<int32_t, float> m(TensorType_INT32, {3}, {2, 2, 2}, 4.0,
                                GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST_P(FillOpTest, FillOutputScalar) {
  FillOpModel<int64_t, float> m(TensorType_INT64, {0}, {}, 4.0, GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4.0}));
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
}

TEST_P(FillOpTest, FillBool) {
  FillOpModel<int64_t, bool> m(TensorType_INT64, {3}, {2, 2, 2}, true,
                               GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({true, true, true, true, true,
                                               true, true, true}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST(FillOpTest, FillString) {
  FillOpModel<int64_t, std::string> m(TensorType_INT64, {3}, {2, 2, 2}, "AB",
                                      TestType::kDynamic);
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({"AB", "AB", "AB", "AB", "AB",
                                               "AB", "AB", "AB"}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST_P(FillOpTest, FillInt8) {
  FillOpModel<int64_t, int8_t> m(TensorType_INT64, {3}, {2, 2, 2}, 5,
                                 GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

template <typename quant_type>
void QuantizedFill(float value) {
  // Prepare TensorData for quantization of value
  const float kMin = -1;
  // Workaround to get a zero-point of 0
  const float kMax =
      std::numeric_limits<quant_type>::max() /
      static_cast<float>(std::numeric_limits<quant_type>::max() + 1);
  const TensorData tensor_data(GetTensorType<quant_type>(), {},
                               std::abs(value) * kMin, std::abs(value) * kMax);

  QuantizedFillOpModel<int32_t, quant_type> m(TensorType_INT32, {2}, {2, 3},
                                              tensor_data, value);
  m.Invoke();

  constexpr float epsilon = 0.01f;
  const float min_value = tensor_data.min - epsilon;
  const float max_value = tensor_data.max + epsilon;
  const float kQuantizedTolerance =
      (max_value - min_value) / (std::numeric_limits<quant_type>::max() -
                                 std::numeric_limits<quant_type>::min());
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(
          {value, value, value, value, value, value}, kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
}

TEST(FillOpTest, QuantizedFillInt8) { QuantizedFill<int8_t>(3.14f); }

TEST(FillOpTest, QuantizedFillInt16) { QuantizedFill<int16_t>(3.14f); }

INSTANTIATE_TEST_SUITE_P(FillOpTest, FillOpTest,
                         ::testing::Values(TestType::kConst,
                                           TestType::kDynamic));

}  // namespace
}  // namespace tflite
