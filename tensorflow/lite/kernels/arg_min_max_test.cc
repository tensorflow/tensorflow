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
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class ArgBaseOpModel : public SingleOpModel {
 public:
  ArgBaseOpModel(TensorType input_type, int axis_value, TensorType axis_type,
                 bool constant_axis, TensorType output_type)
      : axis_value_(axis_value),
        axis_type_(axis_type),
        constant_axis_(constant_axis) {
    input_ = AddInput(input_type);
    if (constant_axis) {
      if (axis_type == TensorType_INT64) {
        axis_ =
            AddConstInput(axis_type, {static_cast<int64_t>(axis_value)}, {1});
      } else {
        axis_ = AddConstInput(axis_type, {axis_value}, {1});
      }
    } else {
      axis_ = AddInput(axis_type);
    }
    output_ = AddOutput(output_type);
  }

  int input() const { return input_; }
  int axis() const { return axis_; }

  std::vector<int32_t> GetInt32Output() const {
    return ExtractVector<int32_t>(output_);
  }
  std::vector<int64_t> GetInt64Output() const {
    return ExtractVector<int64_t>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  void PopulateAxisIfNeeded() {
    if (constant_axis_) return;
    if (axis_type_ == TensorType_INT32) {
      PopulateTensor<int32_t>(axis(), {axis_value_});
    } else {
      PopulateTensor<int64_t>(axis(), {axis_value_});
    }
  }

  const int axis_value_;
  const TensorType axis_type_;
  const bool constant_axis_;

  int input_;
  int axis_;
  int output_;
};

class ArgMaxOpModel : public ArgBaseOpModel {
 public:
  ArgMaxOpModel(std::initializer_list<int> input_shape, TensorType input_type,
                int axis_value, TensorType axis_type, bool constant_axis,
                TensorType output_type)
      : ArgBaseOpModel(input_type, axis_value, axis_type, constant_axis,
                       output_type) {
    ArgBaseOpModel::SetBuiltinOp(
        BuiltinOperator_ARG_MAX, BuiltinOptions_ArgMaxOptions,
        CreateArgMaxOptions(ArgBaseOpModel::builder_, output_type).Union());
    ArgBaseOpModel::BuildInterpreter({input_shape, {1}});
    PopulateAxisIfNeeded();
  }
};

class ArgMinOpModel : public ArgBaseOpModel {
 public:
  ArgMinOpModel(std::initializer_list<int> input_shape, TensorType input_type,
                int axis_value, TensorType axis_type, bool constant_axis,
                TensorType output_type)
      : ArgBaseOpModel(input_type, axis_value, axis_type, constant_axis,
                       output_type) {
    ArgBaseOpModel::SetBuiltinOp(
        BuiltinOperator_ARG_MIN, BuiltinOptions_ArgMinOptions,
        CreateArgMinOptions(ArgBaseOpModel::builder_, output_type).Union());
    ArgBaseOpModel::BuildInterpreter({input_shape, {1}});
    PopulateAxisIfNeeded();
  }
};

// Declare ArgMinMaxOpTest as a parameterized test, where the parameter is a
// tuple with:
// - boolean indicating whether to use a constant axis or not.
// - axis type (TensorType_INT32 or TensorType_INT64)
// - output type (TensorType_INT32 or TensorType_INT64)
class ArgMinMaxOpTest : public ::testing::TestWithParam<
                            std::tuple<bool, TensorType, TensorType>> {
 public:
  bool ConstantAxis() const { return std::get<0>(GetParam()); }

  TensorType AxisType() const { return std::get<1>(GetParam()); }

  TensorType OutputType() const { return std::get<2>(GetParam()); }

  void ValidateOutput(const ArgBaseOpModel& model,
                      const std::vector<int>& expected_output) {
    if (OutputType() == TensorType_INT32) {
      EXPECT_THAT(model.GetInt32Output(), ElementsAreArray(expected_output));
    } else {
      EXPECT_THAT(model.GetInt64Output(), ElementsAreArray(expected_output));
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    ArgMinMaxOpTest, ArgMinMaxOpTest,
    ::testing::Combine(::testing::Bool(),
                       ::testing::Values(TensorType_INT32, TensorType_INT64),
                       ::testing::Values(TensorType_INT32, TensorType_INT64)));

TEST_P(ArgMinMaxOpTest, GetMaxArgFloat) {
  ArgMaxOpModel model({1, 1, 1, 4}, TensorType_FLOAT32, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<float>(model.input(), {0.1, 0.9, 0.7, 0.3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {1});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1}));
}

TEST_P(ArgMinMaxOpTest, GetMaxArgUInt8) {
  ArgMaxOpModel model({1, 1, 1, 4}, TensorType_UINT8, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<uint8_t>(model.input(), {1, 9, 7, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {1});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1}));
}

TEST_P(ArgMinMaxOpTest, GetMaxArgInt8) {
  ArgMaxOpModel model({1, 1, 1, 4}, TensorType_INT8, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<int8_t>(model.input(), {-1, -9, 7, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {2});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1}));
}

TEST_P(ArgMinMaxOpTest, GetMaxArgInt) {
  ArgMaxOpModel model({1, 1, 1, 4}, TensorType_INT32, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<int>(model.input(), {1, 9, 7, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {1});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1}));
}

TEST_P(ArgMinMaxOpTest, GetMaxArgBool) {
  ArgMaxOpModel model({1, 1, 1, 4}, TensorType_BOOL, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<bool>(model.input(), {true, false, false, false});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {0});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1}));
}

TEST_P(ArgMinMaxOpTest, GetMaxArgMulDimensions) {
  ArgMaxOpModel model({1, 1, 2, 4}, TensorType_INT32, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<int>(model.input(), {1, 2, 7, 8, 1, 9, 7, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {3, 1});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 2}));
}

TEST_P(ArgMinMaxOpTest, GetMaxArgNegativeAxis) {
  ArgMaxOpModel model({1, 1, 2, 4}, TensorType_INT32, -2, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<int>(model.input(), {1, 2, 7, 8, 1, 9, 7, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {0, 1, 0, 0});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 4}));
}

TEST_P(ArgMinMaxOpTest, GetMaxArgOutput64) {
  ArgMaxOpModel model({1, 1, 2, 4}, TensorType_INT32, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<int>(model.input(), {10, 2, 7, 8, 1, 9, 7, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {0, 1});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 2}));
}

TEST_P(ArgMinMaxOpTest, GetMaxArgFloatLastAxis) {
  std::vector<float> input{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  for (int i = 1; i < 10; ++i) {
    ArgMaxOpModel model({i}, TensorType_FLOAT32, 0, AxisType(), ConstantAxis(),
                        OutputType());
    model.PopulateTensor<float>(
        model.input(), std::vector<float>(input.begin(), input.begin() + i));
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

    ValidateOutput(model, {i - 1});
  }
}

TEST_P(ArgMinMaxOpTest, GetMinArgFloat) {
  ArgMinOpModel model({1, 1, 1, 4}, TensorType_FLOAT32, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<float>(model.input(), {0.1, 0.9, 0.7, 0.3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {0});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1}));
}

TEST_P(ArgMinMaxOpTest, GetMinArgInt) {
  ArgMinOpModel model({1, 1, 1, 4}, TensorType_INT32, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<int>(model.input(), {1, 9, 7, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {0});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1}));
}

TEST_P(ArgMinMaxOpTest, GetMinArgBool) {
  ArgMinOpModel model({1, 1, 1, 4}, TensorType_BOOL, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<bool>(model.input(), {true, false, true, true});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {1});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1}));
}

TEST_P(ArgMinMaxOpTest, GetMinArgMulDimensions) {
  ArgMinOpModel model({1, 1, 2, 4}, TensorType_INT32, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<int>(model.input(), {1, 2, 7, 8, 1, 9, 7, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {0, 0});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 2}));
}

TEST_P(ArgMinMaxOpTest, GetMinArgNegativeAxis) {
  ArgMinOpModel model({1, 1, 2, 4}, TensorType_INT32, -2, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<int>(model.input(), {1, 2, 7, 8, 1, 9, 7, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {0, 0, 0, 1});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 4}));
}

TEST_P(ArgMinMaxOpTest, GetMinArgOutput64) {
  ArgMinOpModel model({1, 1, 2, 4}, TensorType_INT32, 3, AxisType(),
                      ConstantAxis(), OutputType());
  model.PopulateTensor<int>(model.input(), {10, 2, 7, 8, 1, 9, 7, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  ValidateOutput(model, {1, 0});
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 2}));
}

TEST_P(ArgMinMaxOpTest, GetMinArgFloatLastAxis) {
  std::vector<float> input{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
  for (int i = 1; i < 10; ++i) {
    ArgMinOpModel model({i}, TensorType_FLOAT32, 0, AxisType(), ConstantAxis(),
                        OutputType());
    model.PopulateTensor<float>(
        model.input(), std::vector<float>(input.begin(), input.begin() + i));
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

    ValidateOutput(model, {i - 1});
  }
}

TEST_P(ArgMinMaxOpTest, GetMaxArgInt8LastAxis) {
  // Vector size for int8 is 16 elements, so 35 covers two SIMD widths
  // Plus extras for testing
  constexpr int INPUT_SIZE = 35;
  std::vector<int8_t> input;
  input.reserve(INPUT_SIZE);
  for (int i = 0; i < INPUT_SIZE; i++) {
    input.push_back(INPUT_SIZE - i);
  }
  for (int i = 1; i < INPUT_SIZE; ++i) {
    ArgMinOpModel model({i}, TensorType_INT8, 0, AxisType(), ConstantAxis(),
                        OutputType());
    model.PopulateTensor<int8_t>(
        model.input(), std::vector<int8_t>(input.begin(), input.begin() + i));
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

    ValidateOutput(model, {i - 1});
  }
}

TEST_P(ArgMinMaxOpTest, GetMaxArgUInt8LastAxis) {
  // Vector size for int8 is 16 elements, so 35 covers two SIMD widths
  // Plus extras for testing
  constexpr int INPUT_SIZE = 35;
  std::vector<uint8_t> input;
  input.reserve(INPUT_SIZE);
  for (unsigned int i = 0; i < INPUT_SIZE; i++) {
    input.push_back(INPUT_SIZE - i);
  }
  for (int i = 1; i < INPUT_SIZE; ++i) {
    ArgMinOpModel model({i}, TensorType_UINT8, 0, AxisType(), ConstantAxis(),
                        OutputType());
    model.PopulateTensor<uint8_t>(
        model.input(), std::vector<uint8_t>(input.begin(), input.begin() + i));
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

    ValidateOutput(model, {i - 1});
  }
}

}  // namespace
}  // namespace tflite
