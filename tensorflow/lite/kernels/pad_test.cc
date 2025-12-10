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
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Matcher;

template <typename RegularInputOutput, typename PaddingIntegerType>
class PadOpModel : public SingleOpModel {
 public:
  void SetInput(std::initializer_list<RegularInputOutput> data) {
    PopulateTensor<RegularInputOutput>(input_, data);
  }

  template <typename QuantizedInputOutput>
  void SetQuantizedInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<QuantizedInputOutput>(input_, data);
  }

  template <typename QuantizedInputOutput>
  void SetQuantizedPadValue(float data) {
    QuantizeAndPopulate<QuantizedInputOutput>(constant_values_, {data});
  }

  void SetPaddings(std::initializer_list<PaddingIntegerType> paddings) {
    PopulateTensor<PaddingIntegerType>(paddings_, paddings);
  }

  std::vector<RegularInputOutput> GetOutput() {
    return ExtractVector<RegularInputOutput>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename QuantizedInputOutput>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<QuantizedInputOutput>(
        ExtractVector<QuantizedInputOutput>(output_), GetScale(output_),
        GetZeroPoint(output_));
  }

 protected:
  int input_;
  int output_;
  int paddings_;
  int constant_values_;
};

// Tests case where paddings is a const tensor. Type T1 is the dtype. Type T2 is
// the padding dtype.
template <typename T1, typename T2>
class PadV2OpConstModel : public PadOpModel<T1, T2> {
 public:
  PadV2OpConstModel(const TensorData& input,
                    std::initializer_list<int> paddings_shape,
                    std::initializer_list<T2> paddings, T1 constant_values,
                    const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ =
        this->AddConstInput(GetTensorType<T2>(), paddings, paddings_shape);
    this->constant_values_ =
        this->AddConstInput(GetTensorType<T1>(), {constant_values}, {1});

    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PADV2, BuiltinOptions_PadV2Options,
                       CreatePadV2Options(this->builder_).Union());
    this->BuildInterpreter({input.shape});
  }

  PadV2OpConstModel(const TensorData& input,
                    std::initializer_list<int> paddings_shape,
                    std::initializer_list<T2> paddings,
                    const TensorData& constant_values,
                    const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ =
        this->AddConstInput(GetTensorType<T2>(), paddings, paddings_shape);
    this->constant_values_ = this->AddInput(constant_values);

    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PADV2, BuiltinOptions_PadV2Options,
                       CreatePadV2Options(this->builder_).Union());
    this->BuildInterpreter({input.shape});
  }
};

// Tests case where paddings is a const tensor. Type T is the padding dtype.
//
// Example usage is as follows:
//    PadOpDynamicModel m(input_shape, paddings_shape, paddings_data);
//    m.SetInput(input_data);
//    m.Invoke();
template <typename T>
class PadOpConstModel : public PadOpModel<float, T> {
 public:
  PadOpConstModel(const TensorData& input,
                  std::initializer_list<int> paddings_shape,
                  std::initializer_list<T> paddings, const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ =
        this->AddConstInput(GetTensorType<T>(), paddings, paddings_shape);
    this->constant_values_ = this->AddNullInput();
    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PAD, BuiltinOptions_PadOptions,
                       CreatePadOptions(this->builder_).Union());
    this->BuildInterpreter({input.shape});
  }
};

// Test case where paddings is a non-const tensor.
template <typename RegularInputOutput, typename PaddingIntegerType>
class PadV2OpDynamicModel
    : public PadOpModel<RegularInputOutput, PaddingIntegerType> {
 public:
  PadV2OpDynamicModel(const TensorData& input,
                      std::initializer_list<int> paddings_shape,
                      RegularInputOutput constant_values,
                      const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ = this->AddInput(GetTensorType<PaddingIntegerType>());
    this->constant_values_ = this->AddConstInput(
        GetTensorType<RegularInputOutput>(), {constant_values}, {1});
    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PADV2, BuiltinOptions_PadV2Options,
                       CreatePadV2Options(this->builder_).Union());
    this->BuildInterpreter({input.shape, paddings_shape});
  }
  PadV2OpDynamicModel(const TensorData& input,
                      std::initializer_list<int> paddings_shape,
                      const TensorData& constant_values,
                      const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ = this->AddInput(GetTensorType<PaddingIntegerType>());
    this->constant_values_ = this->AddInput(constant_values);
    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PADV2, BuiltinOptions_PadV2Options,
                       CreatePadV2Options(this->builder_).Union());
    this->BuildInterpreter({input.shape, paddings_shape});
  }
};

// Test case where paddings is a non-const tensor.
//
// Example usage is as follows:
//    PadOpDynamicModel m(input_shape, paddings_shape);
//    m.SetInput(input_data);
//    m.SetPaddings(paddings_data);
//    m.Invoke();
template <typename T>
class PadOpDynamicModel : public PadOpModel<float, T> {
 public:
  PadOpDynamicModel(const TensorData& input,
                    std::initializer_list<int> paddings_shape,
                    const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ = this->AddInput(GetTensorType<T>());
    this->constant_values_ = this->AddNullInput();
    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PAD, BuiltinOptions_PadOptions,
                       CreatePadOptions(this->builder_).Union());
    this->BuildInterpreter({input.shape, paddings_shape});
  }
};

class PadOpTest : public ::testing::Test {};

#if GTEST_HAS_DEATH_TEST
template <typename padding_integer_type>
void TooFewDimensions() {
  EXPECT_DEATH(PadOpConstModel<padding_integer_type>(
                   {TensorType_FLOAT32, {1, 2, 3, 4, 5, 6, 7, 8, 9}}, {9, 2},
                   {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9},
                   {TensorType_FLOAT32}),
               "dims <= reference_ops::PadKernelMaxDimensionCount()");
}

TEST_F(PadOpTest, Int32PaddingTooFewDimensions) { TooFewDimensions<int32_t>(); }

TEST_F(PadOpTest, Int64PaddingTooFewDimensions) { TooFewDimensions<int64_t>(); }

TEST_F(PadOpTest, Int8PaddingTooFewDimensions) { TooFewDimensions<int8_t>(); }

TEST_F(PadOpTest, Int16PaddingTooFewDimensions) { TooFewDimensions<int16_t>(); }

template <typename padding_integer_type>
void UnequalDimensions() {
  EXPECT_DEATH(PadOpConstModel<padding_integer_type>(
                   {TensorType_FLOAT32, {1, 1, 2, 1}}, {3, 2},
                   {1, 1, 2, 2, 3, 3}, {TensorType_FLOAT32}),
               "3 != 4");
}

TEST_F(PadOpTest, Int32PaddingUnequalDimensions) {
  UnequalDimensions<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingUnequalDimensions) {
  UnequalDimensions<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingUnequalDimensions) { UnequalDimensions<int8_t>(); }

TEST_F(PadOpTest, Int16PaddingUnequalDimensions) {
  UnequalDimensions<int16_t>();
}

template <typename padding_integer_type>
void InvalidPadValue() {
  EXPECT_DEATH(PadOpConstModel<int32_t>({TensorType_FLOAT32, {1, 1, 2, 1}},
                                        {4, 2}, {0, 0, 1, -1, 2, -1, 0, 0},
                                        {TensorType_FLOAT32}),
               "Pad value has to be greater than equal to 0.");
}

TEST_F(PadOpTest, Int32PaddingInvalidPadValue) { InvalidPadValue<int32_t>(); }

TEST_F(PadOpTest, Int64PaddingInvalidPadValue) { InvalidPadValue<int64_t>(); }

TEST_F(PadOpTest, Int8PaddingInvalidPadValue) { InvalidPadValue<int8_t>(); }

TEST_F(PadOpTest, Int16PaddingInvalidPadValue) { InvalidPadValue<int16_t>(); }

TEST_F(PadOpTest, Int64PaddingOverflow) {
  EXPECT_DEATH(PadOpConstModel<int64_t>(
                   {TensorType_FLOAT32, {1, 1, 2, 1}}, {4, 2},
                   {std::numeric_limits<int64_t>::min(), 0, 1, -1, 2, -1, 0, 0},
                   {TensorType_FLOAT32}),
               "INT64 padding overflow. Only support value between INT32_MIN "
               "and INT32_MAX.");
  EXPECT_DEATH(PadOpConstModel<int64_t>(
                   {TensorType_FLOAT32, {1, 1, 2, 1}}, {4, 2},
                   {0, 0, 1, -1, 2, -1, std::numeric_limits<int64_t>::max(), 0},
                   {TensorType_FLOAT32}),
               "INT64 padding overflow. Only support value between INT32_MIN "
               "and INT32_MAX.");
}
#endif

template <typename padding_integer_type>
void SimpleConstTest() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadOpConstModel<padding_integer_type> m({TensorType_FLOAT32, {1, 2, 2, 1}},
                                          {4, 2}, {1, 1, 0, 0, 1, 1, 0, 0},
                                          {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                                0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2, 4, 1}));
}

TEST_F(PadOpTest, Int32PaddingSimpleConstTest) { SimpleConstTest<int32_t>(); }

TEST_F(PadOpTest, Int64PaddingSimpleConstTest) { SimpleConstTest<int64_t>(); }

TEST_F(PadOpTest, Int8PaddingSimpleConstTest) { SimpleConstTest<int8_t>(); }

TEST_F(PadOpTest, Int16PaddingSimpleConstTest) { SimpleConstTest<int16_t>(); }

template <typename padding_integer_type>
void SimpleConstImageStyleTest() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadOpConstModel<padding_integer_type> m({TensorType_FLOAT32, {1, 2, 2, 1}},
                                          {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0},
                                          {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadOpTest, Int32PaddingSimpleConstImageStyleTest) {
  SimpleConstImageStyleTest<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingSimpleConstImageStyleTest) {
  SimpleConstImageStyleTest<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingSimpleConstImageStyleTest) {
  SimpleConstImageStyleTest<int8_t>();
}

TEST_F(PadOpTest, Int16PaddingSimpleConstImageStyleTest) {
  SimpleConstImageStyleTest<int16_t>();
}

// Optimized versions may choose to handle zero-sized images differently.
template <typename padding_integer_type>
void ZeroHeightConstImageStyleTest() {
  PadOpConstModel<padding_integer_type> m({TensorType_FLOAT32, {1, 0, 2, 1}},
                                          {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0},
                                          {TensorType_FLOAT32});
  // Nothing to SetInput().
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 4, 1}));
}

TEST_F(PadOpTest, Int32PaddingZeroHeightConstImageStyleTest) {
  ZeroHeightConstImageStyleTest<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingZeroHeightConstImageStyleTest) {
  ZeroHeightConstImageStyleTest<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingZeroHeightConstImageStyleTest) {
  ZeroHeightConstImageStyleTest<int8_t>();
}

TEST_F(PadOpTest, Int16PaddingZeroHeightConstImageStyleTest) {
  ZeroHeightConstImageStyleTest<int16_t>();
}

// Optimized versions may choose to handle zero-sized images differently.
template <typename padding_integer_type>
void ZeroWidthConstImageStyleTest() {
  PadOpConstModel<padding_integer_type> m({TensorType_FLOAT32, {1, 2, 0, 1}},
                                          {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0},
                                          {TensorType_FLOAT32});
  // Nothing to SetInput().
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 2, 1}));
}

TEST_F(PadOpTest, Int32PaddingZeroWidthConstImageStyleTest) {
  ZeroWidthConstImageStyleTest<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingZeroWidthConstImageStyleTest) {
  ZeroWidthConstImageStyleTest<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingZeroWidthConstImageStyleTest) {
  ZeroWidthConstImageStyleTest<int8_t>();
}

TEST_F(PadOpTest, Int16PaddingZeroWidthConstImageStyleTest) {
  ZeroWidthConstImageStyleTest<int16_t>();
}

template <typename padding_integer_type>
void SimpleConst1DTest() {
  PadOpConstModel<padding_integer_type> m({TensorType_FLOAT32, {2}}, {1, 2},
                                          {1, 2}, {TensorType_FLOAT32});
  m.SetInput({2, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 3, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({5}));
}

TEST_F(PadOpTest, Int32PaddingSimpleConst1DTest) {
  SimpleConst1DTest<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingSimpleConst1DTest) {
  SimpleConst1DTest<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingSimpleConst1DTest) { SimpleConst1DTest<int8_t>(); }

TEST_F(PadOpTest, Int16PaddingSimpleConst1DTest) {
  SimpleConst1DTest<int16_t>();
}

template <typename padding_integer_type>
void SimpleConst1DDim0Test() {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  PadOpConstModel<int32_t> m({TensorType_FLOAT32, {0}}, {1, 2}, {1, 2},
                             {TensorType_FLOAT32});
  // NumElements(input) = 0, so there is no input data.
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
}

TEST_F(PadOpTest, Int32PaddingSimpleConst1DDim0Test) {
  SimpleConst1DDim0Test<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingSimpleConst1DDim0Test) {
  SimpleConst1DDim0Test<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingSimpleConst1DDim0Test) {
  SimpleConst1DDim0Test<int8_t>();
}

TEST_F(PadOpTest, Int16PaddingSimpleConst1DDim0Test) {
  SimpleConst1DDim0Test<int16_t>();
}

template <typename padding_integer_type>
void SimpleDynamicTest() {
  PadOpDynamicModel<padding_integer_type> m({TensorType_FLOAT32, {1, 2, 2, 1}},
                                            {4, 2}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadOpTest, Int32PaddingSimpleDynamicTest) {
  SimpleDynamicTest<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingSimpleDynamicTest) {
  SimpleDynamicTest<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingSimpleDynamicTest) { SimpleDynamicTest<int8_t>(); }

TEST_F(PadOpTest, Int16PaddingSimpleDynamicTest) {
  SimpleDynamicTest<int16_t>();
}

template <typename padding_integer_type>
void DynamicUnequalDimensions() {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  PadOpDynamicModel<padding_integer_type> m({TensorType_FLOAT32, {}}, {3, 2},
                                            {TensorType_FLOAT32});
  // Skip invoking m.SetInput() since the method doesn't work with dynamic
  // shapes.
  m.SetPaddings({0, 0, 1, 1, 1, 1});
  ASSERT_NE(m.Invoke(), kTfLiteOk) << "Unequal dimensions.";
}

TEST_F(PadOpTest, Int32PaddingDynamicUnequalDimensions) {
  DynamicUnequalDimensions<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingDynamicUnequalDimensions) {
  DynamicUnequalDimensions<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingDynamicUnequalDimensions) {
  DynamicUnequalDimensions<int8_t>();
}

TEST_F(PadOpTest, Int16PaddingDynamicUnequalDimensions) {
  DynamicUnequalDimensions<int16_t>();
}

template <typename padding_integer_type>
void AdvancedConstTestV2() {
  PadOpConstModel<padding_integer_type> m({TensorType_FLOAT32, {1, 2, 3, 1}},
                                          {4, 2}, {1, 0, 0, 2, 0, 3, 0, 0},
                                          {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 4, 5,
                        6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 4, 6, 1}));
}

TEST_F(PadOpTest, Int32PaddingAdvancedConstTest) {
  AdvancedConstTestV2<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingAdvancedConstTest) {
  AdvancedConstTestV2<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingAdvancedConstTest) {
  AdvancedConstTestV2<int8_t>();
}

TEST_F(PadOpTest, Int16PaddingAdvancedConstTest) {
  AdvancedConstTestV2<int16_t>();
}

template <typename padding_integer_type>
void AdvancedConstImageStyleTest() {
  PadOpConstModel<int32_t> m({TensorType_FLOAT32, {1, 2, 3, 1}}, {4, 2},
                             {0, 0, 0, 2, 1, 3, 0, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(PadOpTest, Int32PaddingAdvancedConstImageStyleTest) {
  AdvancedConstImageStyleTest<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingAdvancedConstImageStyleTest) {
  AdvancedConstImageStyleTest<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingAdvancedConstImageStyleTest) {
  AdvancedConstImageStyleTest<int8_t>();
}

TEST_F(PadOpTest, Int16PaddingAdvancedConstImageStyleTest) {
  AdvancedConstImageStyleTest<int16_t>();
}

template <typename padding_integer_type>
void AdvancedDynamicTest() {
  PadOpDynamicModel<padding_integer_type> m({TensorType_FLOAT32, {1, 2, 3, 1}},
                                            {4, 2}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(PadOpTest, Int32PaddingAdvancedDynamicTest) {
  AdvancedDynamicTest<int32_t>();
}

TEST_F(PadOpTest, Int64PaddingAdvancedDynamicTest) {
  AdvancedDynamicTest<int64_t>();
}

TEST_F(PadOpTest, Int8PaddingAdvancedDynamicTest) {
  AdvancedDynamicTest<int8_t>();
}

TEST_F(PadOpTest, Int16PaddingAdvancedDynamicTest) {
  AdvancedDynamicTest<int16_t>();
}

std::vector<Matcher<float>> DequantizedArrayNear(
    const std::vector<float>& values, const float min, const float max) {
  const float quantization_tolerance = (max - min) / 255.0;
  return ArrayFloatNear(values, quantization_tolerance);
}

class QuantizedPadOpTest : public ::testing::Test {};

#if GTEST_HAS_DEATH_TEST
template <typename integer_type, TensorType tensor_dtype>
void ZeroNotInQuantizationRange() {
  // The test_util and actual quantization code currently ensure that the range
  // must include zero, but if that ever changes, this test will catch it.
  EXPECT_DEATH(PadOpConstModel<int32_t> m(
                   {tensor_dtype, {1, 2, 2, 1}, 1.0, 2.0}, {4, 2},
                   {0, 0, 1, 1, 1, 1, 0, 0}, {tensor_dtype, {}, 1.0, 2.0}),
               ".*Check failed: f_min <= 0.*");
}

TEST_F(QuantizedPadOpTest, UInt8ZeroNotInQuantizationRange) {
  ZeroNotInQuantizationRange<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadOpTest, Int8ZeroNotInQuantizationRange) {
  ZeroNotInQuantizationRange<int8_t, TensorType_INT8>();
}
TEST_F(QuantizedPadOpTest, Int16ZeroNotInQuantizationRange) {
  ZeroNotInQuantizationRange<int16_t, TensorType_INT16>();
}
#endif

template <typename integer_type, TensorType tensor_dtype>
void SimpleConstTest() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).

  const float kMin = -1.f;
  const float kMax = tensor_dtype == TensorType_INT16 ? 32767.f / 32768.f : 1.f;

  PadOpConstModel<int32_t> m({tensor_dtype, {1, 2, 2, 1}, kMin, kMax}, {4, 2},
                             {0, 0, 1, 1, 1, 1, 0, 0},
                             {tensor_dtype, {}, kMin, kMax});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0, 0, 0, 0, -0.8, 0.2, 0, 0, 0.9, 0.7, 0, 0, 0, 0, 0},
                  kMin, kMax)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadOpTest, UInt8SimpleConstTest) {
  SimpleConstTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadOpTest, Int8SimpleConstTest) {
  SimpleConstTest<int8_t, TensorType_INT8>();
}
TEST_F(QuantizedPadOpTest, Int16SimpleConstTest) {
  SimpleConstTest<int16_t, TensorType_INT16>();
}

template <typename integer_type, TensorType tensor_dtype>
void SimpleDynamicTest() {
  const float kMin = -1.f;
  const float kMax = tensor_dtype == TensorType_INT16 ? 32767.f / 32768.f : 1.f;

  PadOpDynamicModel<int32_t> m({tensor_dtype, {1, 2, 2, 1}, kMin, kMax}, {4, 2},
                               {tensor_dtype, {}, kMin, kMax});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0, 0, 0, 0, -0.8, 0.2, 0, 0, 0.9, 0.7, 0, 0, 0, 0, 0},
                  kMin, kMax)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadOpTest, UInt8SimpleDynamicTest) {
  SimpleDynamicTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadOpTest, Int8SimpleDynamicTest) {
  SimpleDynamicTest<int8_t, TensorType_INT8>();
}
TEST_F(QuantizedPadOpTest, Int16SimpleDynamicTest) {
  SimpleDynamicTest<int16_t, TensorType_INT16>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedConstTest() {
  const float kMin = -1.f;
  const float kMax = tensor_dtype == TensorType_INT16 ? 32767.f / 32768.f : 1.f;

  PadOpConstModel<int32_t> m({tensor_dtype, {1, 2, 3, 1}, kMin, kMax}, {4, 2},
                             {0, 0, 0, 2, 1, 3, 0, 0},
                             {tensor_dtype, {}, kMin, kMax});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, -0.8, 0.2, 0.9, 0, 0, 0, 0, 0.7, 0.1, -0.3, 0, 0, 0,
                   0, 0,    0,   0,   0, 0, 0, 0, 0,   0,   0,    0, 0, 0},
                  kMin, kMax)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadOpTest, UInt8AdvancedConstTest) {
  AdvancedConstTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadOpTest, Int8AdvancedConstTest) {
  AdvancedConstTest<int8_t, TensorType_INT8>();
}
TEST_F(QuantizedPadOpTest, Int16AdvancedConstTest) {
  AdvancedConstTest<int16_t, TensorType_INT16>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedDynamicTest() {
  const float kMin = -1.f;
  const float kMax = tensor_dtype == TensorType_INT16 ? 32767.f / 32768.f : 1.f;

  PadOpDynamicModel<int32_t> m({tensor_dtype, {1, 2, 3, 1}, kMin, kMax}, {4, 2},
                               {tensor_dtype, {}, kMin, kMax});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, -0.8, 0.2, 0.9, 0, 0, 0, 0, 0.7, 0.1, -0.3, 0, 0, 0,
                   0, 0,    0,   0,   0, 0, 0, 0, 0,   0,   0,    0, 0, 0},
                  kMin, kMax)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadOpTest, UInt8AdvancedDynamicTest) {
  AdvancedDynamicTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadOpTest, Int8AdvancedDynamicTest) {
  AdvancedDynamicTest<int8_t, TensorType_INT8>();
}
TEST_F(QuantizedPadOpTest, Int16AdvancedDynamicTest) {
  AdvancedDynamicTest<int16_t, TensorType_INT16>();
}

class PadV2OpTest : public ::testing::Test {};

#if GTEST_HAS_DEATH_TEST
template <typename padding_integer_type>
void TooManyDimensions() {
  typedef PadV2OpConstModel<float, padding_integer_type> f;
  EXPECT_DEATH(f({TensorType_FLOAT32, {1, 2, 3, 4, 5, 6, 7, 8, 9}}, {9, 2},
                 {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9}, 0.0,
                 {TensorType_FLOAT32}),
               "dims <= reference_ops::PadKernelMaxDimensionCount()");
}

TEST_F(PadV2OpTest, Int32PaddingTooManyDimensions) {
  TooManyDimensions<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingTooManyDimensions) {
  TooManyDimensions<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingTooManyDimensions) {
  TooManyDimensions<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingTooManyDimensions) {
  TooManyDimensions<int16_t>();
}

template <typename padding_integer_type>
void UnequalDimensionsV2() {
  typedef PadV2OpConstModel<float, padding_integer_type> f;
  EXPECT_DEATH(f({TensorType_FLOAT32, {1, 1, 2, 1}}, {3, 2}, {1, 1, 2, 2, 3, 3},
                 0.0, {TensorType_FLOAT32}),
               "3 != 4");
}

TEST_F(PadV2OpTest, Int32PaddingUnequalDimensions) {
  UnequalDimensionsV2<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingUnequalDimensions) {
  UnequalDimensionsV2<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingUnequalDimensions) {
  UnequalDimensionsV2<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingUnequalDimensions) {
  UnequalDimensionsV2<int16_t>();
}

template <typename padding_integer_type>
void InvalidPadValueV2() {
  typedef PadV2OpConstModel<float, padding_integer_type> f;
  EXPECT_DEATH(f({TensorType_FLOAT32, {1, 1, 2, 1}}, {4, 2},
                 {0, 0, 1, -1, 2, -1, 0, 0}, 0.0, {TensorType_FLOAT32}),
               "Pad value has to be greater than equal to 0.");
}

TEST_F(PadV2OpTest, Int32PaddingInvalidPadValue) {
  InvalidPadValueV2<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingInvalidPadValue) {
  InvalidPadValueV2<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingInvalidPadValue) { InvalidPadValueV2<int8_t>(); }

TEST_F(PadV2OpTest, Int16PaddingInvalidPadValue) {
  InvalidPadValueV2<int16_t>();
}

TEST_F(PadV2OpTest, Int64PaddingOverflow) {
  EXPECT_DEATH(PadOpConstModel<int64_t>(
                   {TensorType_FLOAT32, {1, 1, 2, 1}}, {4, 2},
                   {std::numeric_limits<int64_t>::min(), 0, 1, -1, 2, -1, 0, 0},
                   {TensorType_FLOAT32}),
               "INT64 padding overflow. Only support value between INT32_MIN "
               "and INT32_MAX.");
}

TEST_F(PadV2OpTest, UnsupportedPaddingType) {
  EXPECT_DEATH(
      PadOpConstModel<float>({TensorType_FLOAT32, {1, 1, 2, 1}}, {4, 2},
                             {0, 0, 1, 1, 2, 1, 0, 0}, {TensorType_FLOAT32}),
      "Padding type FLOAT32 is currently not supported by Pad.");
}

#endif

template <typename padding_integer_type>
void SimpleConstTestUint8() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0}, 0.0,
      {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleConstTestUint8) {
  SimpleConstTestUint8<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleConstTestUint8) {
  SimpleConstTestUint8<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleConstTestUint8) {
  SimpleConstTestUint8<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleConstTestUint8) {
  SimpleConstTestUint8<int16_t>();
}

template <typename padding_integer_type>
void SimpleConstTestInt8() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0}, 0.0,
      {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleConstTestInt8) {
  SimpleConstTestInt8<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleConstTestInt8) {
  SimpleConstTestInt8<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleConstTestInt8) {
  SimpleConstTestInt8<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleConstTestInt8) {
  SimpleConstTestInt8<int16_t>();
}

template <typename padding_integer_type>
void SimpleConstFloat32ValuedTestUint8() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0}, 5,
      {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4,
                                               5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleConstFloat32ValuedTestUint8) {
  SimpleConstFloat32ValuedTestUint8<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleConstFloat32ValuedTestUint8) {
  SimpleConstFloat32ValuedTestUint8<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleConstFloat32ValuedTestUint8) {
  SimpleConstFloat32ValuedTestUint8<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleConstFloat32ValuedTestUint8) {
  SimpleConstFloat32ValuedTestUint8<int16_t>();
}

template <typename padding_integer_type>
void SimpleConstFloat32ValuedTestInt8() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0}, 5,
      {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4,
                                               5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleConstFloat32ValuedTestInt8) {
  SimpleConstFloat32ValuedTestInt8<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleConstFloat32ValuedTestInt8) {
  SimpleConstFloat32ValuedTestInt8<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleConstFloat32ValuedTestInt8) {
  SimpleConstFloat32ValuedTestInt8<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleConstFloat32ValuedTestInt8) {
  SimpleConstFloat32ValuedTestInt8<int16_t>();
}

template <typename padding_integer_type>
void SimpleConstFloat16ValuedTest() {
  PadV2OpConstModel<Eigen::half, padding_integer_type> m(
      {TensorType_FLOAT16, {1, 2, 2, 1}}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0},
      Eigen::half{4.0f}, {TensorType_FLOAT16});
  m.SetInput({Eigen::half{1.5f}, Eigen::half{2.5f}, Eigen::half{3.5f},
              Eigen::half{4.5}});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {Eigen::half{4}, Eigen::half{4}, Eigen::half{4}, Eigen::half{4},
           Eigen::half{4}, Eigen::half{1.5}, Eigen::half{2.5}, Eigen::half{4},
           Eigen::half{4}, Eigen::half{3.5}, Eigen::half{4.5}, Eigen::half{4},
           Eigen::half{4}, Eigen::half{4}, Eigen::half{4}, Eigen::half{4}})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleConstFloat16) {
  SimpleConstFloat16ValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleConstFloat16) {
  SimpleConstFloat16ValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleConstFloat16) {
  SimpleConstFloat16ValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleConstFloat16) {
  SimpleConstFloat16ValuedTest<int16_t>();
}

template <typename padding_integer_type>
void SimpleConstBFloat16ValuedTest() {
  PadV2OpConstModel<Eigen::bfloat16, padding_integer_type> m(
      {TensorType_BFLOAT16, {1, 2, 2, 1}}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0},
      Eigen::bfloat16{6.0f}, {TensorType_BFLOAT16});
  m.SetInput({Eigen::bfloat16{1.0f}, Eigen::bfloat16{2.0f},
              Eigen::bfloat16{3.0f}, Eigen::bfloat16{4.0}});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 6, 6, 6, 6, 1, 2, 6, 6, 3, 4,
                                               6, 6, 6, 6, 6}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleConstBFloat16) {
  SimpleConstBFloat16ValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleConstBFloat16) {
  SimpleConstBFloat16ValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleConstBFloat16) {
  SimpleConstBFloat16ValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleConstBFloat16) {
  SimpleConstBFloat16ValuedTest<int16_t>();
}

template <typename padding_integer_type>
void SimpleConstBoolValuedTest() {
  PadV2OpConstModel<bool, padding_integer_type> m(
      {TensorType_BOOL, {1, 2, 2, 1}}, {4, 2},
      {false, false, true, true, true, true, false, false}, true,
      {TensorType_BOOL});
  m.SetInput({true, true, false, false});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({true, true, true, true, true, true, true, true, true,
                        false, false, true, true, true, true, true}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleConstBool) {
  SimpleConstBoolValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleConstBool) {
  SimpleConstBoolValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleConstBool) {
  SimpleConstBoolValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleConstBool) {
  SimpleConstBoolValuedTest<int16_t>();
}

template <typename padding_integer_type>
void Simple4DConstFloat32ValuedTest() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 1, 2, 1}}, {4, 2}, {0, 1, 0, 0, 0, 0, 0, 1}, 5,
      {TensorType_FLOAT32});
  m.SetInput({3, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 5, 3, 5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 2, 2}));
}

TEST_F(PadV2OpTest, Int32PaddingSimple4DConstFloat32ValuedTest) {
  Simple4DConstFloat32ValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimple4DConstFloat32ValuedTest) {
  Simple4DConstFloat32ValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimple4DConstFloat32ValuedTest) {
  Simple4DConstFloat32ValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimple4DConstFloat32ValuedTest) {
  Simple4DConstFloat32ValuedTest<int16_t>();
}

template <typename padding_integer_type>
void Simple4DConstFloat16ValuedTest() {
  PadV2OpConstModel<Eigen::half, padding_integer_type> m(
      {TensorType_FLOAT16, {1, 1, 2, 1}}, {4, 2}, {0, 1, 0, 0, 0, 0, 0, 1},
      Eigen::half{7.0}, {TensorType_FLOAT16});
  m.SetInput({Eigen::half{3.0f}, Eigen::half{6.0f}});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 7, 6, 7, 7, 7, 7, 7}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 2, 2}));
}

TEST_F(PadV2OpTest, Int32PaddingSimple4DConstFloat16ValuedTest) {
  Simple4DConstFloat16ValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimple4DConstFloat16ValuedTest) {
  Simple4DConstFloat16ValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimple4DConstFloat16ValuedTest) {
  Simple4DConstFloat16ValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimple4DConstFloat16ValuedTest) {
  Simple4DConstFloat16ValuedTest<int16_t>();
}

template <typename padding_integer_type>
void Simple4DConstBFloat16ValuedTest() {
  PadV2OpConstModel<Eigen::bfloat16, padding_integer_type> m(
      {TensorType_BFLOAT16, {1, 1, 2, 1}}, {4, 2}, {0, 1, 0, 0, 0, 0, 0, 1},
      Eigen::bfloat16{5.0}, {TensorType_BFLOAT16});
  m.SetInput({Eigen::bfloat16{3.2f}, Eigen::bfloat16{6.4f}});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {Eigen::bfloat16{3.2f}, Eigen::bfloat16{5.0f}, Eigen::bfloat16{6.4f},
           Eigen::bfloat16{5.0f}, Eigen::bfloat16{5.0f}, Eigen::bfloat16{5.0f},
           Eigen::bfloat16{5.0f}, Eigen::bfloat16{5.0f}})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 2, 2}));
}

TEST_F(PadV2OpTest, Int32PaddingSimple4DConstBFloat16ValuedTest) {
  Simple4DConstBFloat16ValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimple4DConstBFloat16ValuedTest) {
  Simple4DConstBFloat16ValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimple4DConstBFloat16ValuedTest) {
  Simple4DConstBFloat16ValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimple4DConstBFloat16ValuedTest) {
  Simple4DConstBFloat16ValuedTest<int16_t>();
}

template <typename padding_integer_type>
void SimpleConstInt32ValuedTest() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<int32_t, padding_integer_type> m(
      {TensorType_INT32, {1, 2, 2, 1}}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0}, 5,
      {TensorType_INT32});
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4,
                                               5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleConstInt32ValuedTest) {
  SimpleConstInt32ValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleConstInt32ValuedTest) {
  SimpleConstInt32ValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleConstInt32ValuedTest) {
  SimpleConstInt32ValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleConstInt32ValuedTest) {
  SimpleConstInt32ValuedTest<int16_t>();
}

template <typename padding_integer_type>
void SimpleDynamicTestV2() {
  PadV2OpDynamicModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2}, 0.0, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleDynamicTest) {
  SimpleDynamicTestV2<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleDynamicTest) {
  SimpleDynamicTestV2<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleDynamicTest) {
  SimpleDynamicTestV2<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleDynamicTest) {
  SimpleDynamicTestV2<int16_t>();
}

template <typename padding_integer_type>
void SimpleDynamicTestV2Float16() {
  PadV2OpDynamicModel<Eigen::half, padding_integer_type> m(
      {TensorType_FLOAT16, {1, 2, 2, 1}}, {4, 2}, Eigen::half{0.0},
      {TensorType_FLOAT16});
  m.SetInput({Eigen::half{1.0f}, Eigen::half{2.0f}, Eigen::half{3.0f},
              Eigen::half{4.0f}});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleDynamicTestFloat16) {
  SimpleDynamicTestV2Float16<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleDynamicTestFloat16) {
  SimpleDynamicTestV2Float16<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleDynamicTestFloat16) {
  SimpleDynamicTestV2Float16<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleDynamicTestFloat16) {
  SimpleDynamicTestV2Float16<int16_t>();
}

template <typename padding_integer_type>
void SimpleDynamicTestV2BFloat16() {
  PadV2OpDynamicModel<Eigen::bfloat16, padding_integer_type> m(
      {TensorType_BFLOAT16, {1, 2, 2, 1}}, {4, 2}, Eigen::bfloat16{2.0},
      {TensorType_BFLOAT16});
  m.SetInput({Eigen::bfloat16{5.0f}, Eigen::bfloat16{6.0f},
              Eigen::bfloat16{7.0f}, Eigen::bfloat16{8.0f}});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 2, 2, 2, 2, 5, 6, 2, 2, 7, 8,
                                               2, 2, 2, 2, 2}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleDynamicTestBFloat16) {
  SimpleDynamicTestV2BFloat16<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleDynamicTestBFloat16) {
  SimpleDynamicTestV2BFloat16<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleDynamicTestBFloat16) {
  SimpleDynamicTestV2BFloat16<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleDynamicTestBFloat16) {
  SimpleDynamicTestV2BFloat16<int16_t>();
}

template <typename padding_integer_type>
void SimpleDynamicTestBoolV2() {
  PadV2OpDynamicModel<bool, padding_integer_type> m(
      {TensorType_BOOL, {1, 2, 2, 1}}, {4, 2}, false, {TensorType_BOOL});
  m.SetInput({true, false, true, false});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({false, false, false, false, false, true, false,
                                false, false, true, false, false, false, false,
                                false, false}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleDynamicTestBoolV2) {
  SimpleDynamicTestBoolV2<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleDynamicTestBoolV2) {
  SimpleDynamicTestBoolV2<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleDynamicTestBoolV2) {
  SimpleDynamicTestBoolV2<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleDynamicTestBoolV2) {
  SimpleDynamicTestBoolV2<int16_t>();
}

template <typename padding_integer_type>
void PadV2OpDynamicUnequalDimensions() {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  PadV2OpDynamicModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {}}, {4, 2}, 0.0, {TensorType_FLOAT32});
  // Skip invoking m.SetInput() since the method doesn't work with dynamic
  // shapes.
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_NE(m.Invoke(), kTfLiteOk) << "Unequal dimensions";
}

TEST_F(PadV2OpTest, Int32PaddingDynamicUnequalDimensions) {
  PadV2OpDynamicUnequalDimensions<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingDynamicUnequalDimensions) {
  PadV2OpDynamicUnequalDimensions<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingDynamicUnequalDimensions) {
  PadV2OpDynamicUnequalDimensions<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingDynamicUnequalDimensions) {
  PadV2OpDynamicUnequalDimensions<int16_t>();
}

template <typename padding_integer_type>
void SimpleDynamicValuedTest() {
  PadV2OpDynamicModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2}, 5, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4,
                                               5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleDynamicValuedTest) {
  SimpleDynamicValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleDynamicValuedTest) {
  SimpleDynamicValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleDynamicValuedTest) {
  SimpleDynamicValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleDynamicValuedTest) {
  SimpleDynamicValuedTest<int16_t>();
}

template <typename padding_integer_type>
void SimpleTensorWithDim0Test() {
  PadV2OpDynamicModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 2, 2, 0}}, {4, 2}, 5, {TensorType_FLOAT32});
  // NumElements(input) = 0, so there is no input data.
  m.SetPaddings({0, 0, 1, 1, 0, 0, 1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                               5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 2, 2}));

  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // Since NumElements(output) = 0 in this case, there is no data.
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 0}));
}

TEST_F(PadV2OpTest, Int32PaddingSimpleTensorWithDim0Test) {
  SimpleTensorWithDim0Test<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimpleTensorWithDim0Test) {
  SimpleTensorWithDim0Test<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimpleTensorWithDim0Test) {
  SimpleTensorWithDim0Test<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimpleTensorWithDim0Test) {
  SimpleTensorWithDim0Test<int16_t>();
}

template <typename padding_integer_type>
void Simple5DConstFloat32ValuedTest() {
  PadV2OpConstModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 1, 2, 1, 1}}, {5, 2},
      {0, 1, 0, 0, 1, 1, 0, 0, 0, 1}, 5, {TensorType_FLOAT32});
  m.SetInput({3, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 4, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 3, 5, 3, 5, 5, 5, 5, 5, 5,
                                               5, 5, 5, 5, 5}));
}

TEST_F(PadV2OpTest, Int32PaddingSimple5DConstFloat32ValuedTest) {
  Simple5DConstFloat32ValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimple5DConstFloat32ValuedTest) {
  Simple5DConstFloat32ValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimple5DConstFloat32ValuedTest) {
  Simple5DConstFloat32ValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimple5DConstFloat32ValuedTest) {
  Simple5DConstFloat32ValuedTest<int16_t>();
}

template <typename padding_integer_type>
void Simple5DConstInt32ValuedTest() {
  PadV2OpConstModel<int32_t, padding_integer_type> m(
      {TensorType_INT32, {1, 2, 2, 1, 1}}, {5, 2},
      {0, 0, 1, 1, 1, 1, 0, 0, 1, 1}, 5, {TensorType_INT32});
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1, 3}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                        1, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 4,
                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}));
}

TEST_F(PadV2OpTest, Int32PaddingSimple5DConstInt32ValuedTest) {
  Simple5DConstInt32ValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimple5DConstInt32ValuedTest) {
  Simple5DConstInt32ValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimple5DConstInt32ValuedTest) {
  Simple5DConstInt32ValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimple5DConstInt32ValuedTest) {
  Simple5DConstInt32ValuedTest<int16_t>();
}

template <typename padding_integer_type>
void Simple5DDynamicValuedTest() {
  PadV2OpDynamicModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 2, 2, 1, 1}}, {5, 2}, 5, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0, 1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1, 3}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                        1, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 4,
                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}));
}

TEST_F(PadV2OpTest, Int32PaddingSimple5DDynamicValuedTest) {
  Simple5DDynamicValuedTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingSimple5DDynamicValuedTest) {
  Simple5DDynamicValuedTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingSimple5DDynamicValuedTest) {
  Simple5DDynamicValuedTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingSimple5DDynamicValuedTest) {
  Simple5DDynamicValuedTest<int16_t>();
}

template <typename padding_integer_type>
void AdvancedConstTest() {
  PadV2OpConstModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 2, 3, 1}}, {4, 2}, {0, 0, 0, 2, 1, 3, 0, 0}, 0,
      {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingAdvancedConstTest) {
  AdvancedConstTest<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingAdvancedConstTest) {
  AdvancedConstTest<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingAdvancedConstTest) {
  AdvancedConstTest<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingAdvancedConstTest) {
  AdvancedConstTest<int16_t>();
}

template <typename padding_integer_type>
void AdvancedDynamicTestV2() {
  PadV2OpDynamicModel<float, padding_integer_type> m(
      {TensorType_FLOAT32, {1, 2, 3, 1}}, {4, 2}, 0, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(PadV2OpTest, Int32PaddingAdvancedDynamicTest) {
  AdvancedDynamicTestV2<int32_t>();
}

TEST_F(PadV2OpTest, Int64PaddingAdvancedDynamicTest) {
  AdvancedDynamicTestV2<int64_t>();
}

TEST_F(PadV2OpTest, Int8PaddingAdvancedDynamicTest) {
  AdvancedDynamicTestV2<int8_t>();
}

TEST_F(PadV2OpTest, Int16PaddingAdvancedDynamicTest) {
  AdvancedDynamicTestV2<int16_t>();
}

class QuantizedPadV2OpTest : public ::testing::Test {
 protected:
  std::vector<Matcher<float>> DequantizedArrayNear(
      const std::vector<float>& values, const float min, const float max) {
    const float quantization_tolerance = (max - min) / 255.0;
    return ArrayFloatNear(values, quantization_tolerance);
  }
};

#if GTEST_HAS_DEATH_TEST
template <TensorType tensor_dtype>
void ZeroNotInQuantizationRangeV2() {
  // The test_util and actual quantization code currently ensure that the range
  // must include zero, but if that ever changes, this test will catch it.
  typedef PadV2OpConstModel<float, int32_t> f;
  EXPECT_DEATH(f({tensor_dtype, {1, 2, 2, 1}, 1.0, 2.0}, {4, 2},
                 {0, 0, 1, 1, 1, 1, 0, 0}, 0, {tensor_dtype, {}, 1.0, 2.0}),
               ".*Check failed: f_min <= 0.*");
}

TEST_F(QuantizedPadV2OpTest, UInt8ZeroNotInQuantizationRange) {
  ZeroNotInQuantizationRangeV2<TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8ZeroNotInQuantizationRange) {
  ZeroNotInQuantizationRangeV2<TensorType_INT8>();
}
#endif

template <typename integer_type, TensorType tensor_dtype>
void SimpleConstTestV2() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<integer_type, int32_t> m(
      {tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.template SetQuantizedPadValue<integer_type>(0);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0, 0, 0, 0, -0.8, 0.2, 0, 0, 0.9, 0.7, 0, 0, 0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8SimpleConstTest) {
  SimpleConstTestV2<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8SimpleConstTest) {
  SimpleConstTestV2<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void SimpleDynamicTestV2() {
  PadV2OpDynamicModel<integer_type, int32_t> m(
      {tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0}, {4, 2},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.template SetQuantizedPadValue<integer_type>(0);
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0, 0, 0, 0, -0.8, 0.2, 0, 0, 0.9, 0.7, 0, 0, 0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8SimpleDynamicTest) {
  SimpleDynamicTestV2<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8SimpleDynamicTest) {
  SimpleDynamicTestV2<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedConstTestV2() {
  PadV2OpConstModel<integer_type, int32_t> m(
      {tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0}, {4, 2}, {0, 0, 0, 2, 1, 3, 0, 0},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.template SetQuantizedPadValue<integer_type>(0);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, -0.8, 0.2, 0.9, 0, 0, 0, 0, 0.7, 0.1, -0.3, 0, 0, 0,
                   0, 0,    0,   0,   0, 0, 0, 0, 0,   0,   0,    0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8AdvancedConstTest) {
  AdvancedConstTestV2<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8AdvancedConstTest) {
  AdvancedConstTestV2<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedDynamicTestV2() {
  PadV2OpDynamicModel<integer_type, int32_t> m(
      {tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0}, {4, 2},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.template SetQuantizedPadValue<integer_type>(0);
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, -0.8, 0.2, 0.9, 0, 0, 0, 0, 0.7, 0.1, -0.3, 0, 0, 0,
                   0, 0,    0,   0,   0, 0, 0, 0, 0,   0,   0,    0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8AdvancedDynamicTest) {
  AdvancedDynamicTestV2<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8AdvancedDynamicTest) {
  AdvancedDynamicTestV2<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void SimpleConstValuedTest() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<integer_type, int32_t> m(
      {tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.template SetQuantizedPadValue<integer_type>(-0.5);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {-0.5, -0.5, -0.5, -0.5, -0.5, -0.8, 0.2, -0.5, -0.5, 0.9,
                   0.7, -0.5, -0.5, -0.5, -0.5, -0.5},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8SimpleConstValuedTest) {
  SimpleConstValuedTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8SimpleConstValuedTest) {
  SimpleConstValuedTest<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void SimpleDynamicValuedTest() {
  PadV2OpDynamicModel<integer_type, int32_t> m(
      {tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0}, {4, 2},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.template SetQuantizedPadValue<integer_type>(-0.5);
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {-0.5, -0.5, -0.5, -0.5, -0.5, -0.8, 0.2, -0.5, -0.5, 0.9,
                   0.7, -0.5, -0.5, -0.5, -0.5, -0.5},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8SimpleDynamicValuedTest) {
  SimpleDynamicValuedTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8SimpleDynamicValuedTest) {
  SimpleDynamicValuedTest<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedConstValuedTest() {
  PadV2OpConstModel<integer_type, int32_t> m(
      {tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0}, {4, 2}, {0, 0, 0, 2, 1, 3, 0, 0},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.template SetQuantizedPadValue<integer_type>(-0.5);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {-0.5, -0.8, 0.2,  0.9,  -0.5, -0.5, -0.5, -0.5, 0.7,  0.1,
                   -0.3, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                   -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8AdvancedConstValuedTest) {
  AdvancedConstValuedTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8AdvancedConstValuedTest) {
  AdvancedConstValuedTest<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedDynamicValuedTest() {
  PadV2OpDynamicModel<integer_type, int32_t> m(
      {tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0}, {4, 2},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.template SetQuantizedPadValue<integer_type>(-0.5);
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {-0.5, -0.8, 0.2,  0.9,  -0.5, -0.5, -0.5, -0.5, 0.7,  0.1,
                   -0.3, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                   -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8AdvancedDynamicValuedTest) {
  AdvancedDynamicValuedTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8AdvancedDynamicValuedTest) {
  AdvancedDynamicValuedTest<int8_t, TensorType_INT8>();
}

}  // namespace
}  // namespace tflite
