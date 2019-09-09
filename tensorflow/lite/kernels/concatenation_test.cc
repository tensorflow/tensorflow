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
#include <cstdarg>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseConcatenationOpModel : public SingleOpModel {
 public:
  // TODO(ahentz): Also test different activation types, axis, input
  // dimensions.
  BaseConcatenationOpModel() {}
  BaseConcatenationOpModel(const std::vector<TensorData>& input_template,
                           int axis, int num_inputs,
                           const TensorData& output_template) {
    std::vector<std::vector<int>> all_input_shapes;
    CHECK_EQ(input_template.size(), num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      all_input_shapes.push_back(input_template[i].shape);
      AddInput(input_template[i]);
    }
    output_ = AddOutput({output_template.type, /*shape=*/{},
                         output_template.min, output_template.max});
    SetBuiltinOp(
        BuiltinOperator_CONCATENATION, BuiltinOptions_ConcatenationOptions,
        CreateConcatenationOptions(builder_, axis, ActivationFunctionType_NONE)
            .Union());
    BuildInterpreter(all_input_shapes);
  }
  BaseConcatenationOpModel(const TensorData& input_template, int axis,
                           int num_inputs)
      : BaseConcatenationOpModel(
            std::vector<TensorData>(num_inputs, input_template), axis,
            num_inputs, input_template) {}

 protected:
  int output_;
};

class ConcatenationOpModel : public BaseConcatenationOpModel {
 public:
  using BaseConcatenationOpModel::BaseConcatenationOpModel;
  void SetInput(int index, std::initializer_list<float> data) {
    PopulateTensor(index, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class QuantizedConcatenationOpModel : public BaseConcatenationOpModel {
 public:
  using BaseConcatenationOpModel::BaseConcatenationOpModel;

  template <typename T>
  void SetInput(int index, std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(index, data);
  }
  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }
};

TEST(ConcatenationOpTest, ThreeDimensionalOneInput) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 1, 2}}, /*axis=*/1,
                          /*num_inputs=*/1);
  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 3, 4, 7}));
}

TEST(ConcatenationOpTest, FiveDimensionalOneInput) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 1, 2, 1, 3}}, /*axis=*/2,
                          /*num_inputs=*/1);
  m0.SetInput(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInput) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 1, 2, 1, 3}}, /*axis=*/0,
                          /*num_inputs=*/2);
  m0.SetInput(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f});
  m0.SetInput(1, {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f,
                  22.0f, 23.0f, 24.0f});
  m0.Invoke();
  EXPECT_THAT(
      m0.GetOutput(),
      ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInputNegativeAxes) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 1, 2, 1, 3}}, /*axis=*/-2,
                          /*num_inputs=*/2);
  m0.SetInput(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f});
  m0.SetInput(1, {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f,
                  22.0f, 23.0f, 24.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({1, 2, 3, 13, 14, 15, 4,  5,  6,  16, 17, 18,
                                7, 8, 9, 19, 20, 21, 10, 11, 12, 22, 23, 24}));
}

template <typename Type>
struct QuantizedConcatenationOpTest : public testing::Test {
  using TestType = Type;

  enum TensorType tensor_type =
      std::is_same<Type, int16_t>::value ? TensorType_INT16 : TensorType_UINT8;
};

using TestTypes = testing::Types<uint8_t, int16_t>;
TYPED_TEST_CASE(QuantizedConcatenationOpTest, TestTypes);

TYPED_TEST(QuantizedConcatenationOpTest, FiveDimensionalTwoInputQuantized) {
  using TestType = typename TestFixture::TestType;

  QuantizedConcatenationOpModel m0(
      {TestFixture::tensor_type, {2, 1, 2, 1, 3}, -12.7, 12.8},
      /*axis=*/0,
      /*num_inputs=*/2);

  m0.SetInput<TestType>(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f});
  m0.SetInput<TestType>(1, {1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
                            9.1f, 10.1f, 11.1f, 12.1f});
  m0.Invoke();

  EXPECT_THAT(m0.GetDequantizedOutput<TestType>(),
              ElementsAreArray(ArrayFloatNear({
                  1.0f, 2.0f,  3.0f,  4.0f,  5.0f, 6.0f,  7.0f,  8.0f,
                  9.0f, 10.0f, 11.0f, 12.0f, 1.1f, 2.1f,  3.1f,  4.1f,
                  5.1f, 6.1f,  7.1f,  8.1f,  9.1f, 10.1f, 11.1f, 12.1f,
              })));

  if (TestFixture::tensor_type == TensorType_UINT8) {
    EXPECT_THAT(
        m0.GetOutput<uint8_t>(),
        ElementsAreArray({
            137, 147, 157, 167, 177, 187, 197, 207, 217, 227, 237, 247, 138,  //
            148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248,
        }));
  }

  if (TestFixture::tensor_type == TensorType_INT16) {
    EXPECT_THAT(m0.GetOutput<int16_t>(),
                ElementsAreArray({
                    2441,  5011,  7581,  10151, 12721, 15291,
                    17861, 20431, 23001, 25571, 28141, 30711,  //
                    2698,  5268,  7838,  10408, 12978, 15548,
                    18118, 20688, 23258, 25828, 28398, 30968,
                }));
  }
}

TEST(ConcatenationOpTest, ThreeDimensionalTwoInputsDifferentShapes) {
  ConcatenationOpModel m0(
      {{TensorType_FLOAT32, {2, 1, 2}}, {TensorType_FLOAT32, {2, 3, 2}}},
      /*axis=*/1, /*num_inputs=*/2, TensorType_FLOAT32);
  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput(1, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 3, 1, 2, 3, 4, 5, 6, 4, 7, 7,
                                                8, 9, 10, 11, 12}));
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(ConcatenationOpTest, ThreeDimensionalTwoInputsDifferentShapesWrongAxis) {
  EXPECT_DEATH(
      ConcatenationOpModel m0(
          {{TensorType_FLOAT32, {2, 1, 2}}, {TensorType_FLOAT32, {2, 3, 2}}},
          /*axis=*/0, /*num_inputs=*/2, TensorType_FLOAT32),
      "Cannot allocate tensors");
}
#endif

TEST(ConcatenationOpTest, OneTrivialInput) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {1}}, /*axis=*/0,
                          /*num_inputs=*/1);
  m0.SetInput(0, {5.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(), ::testing::ElementsAre(5));
}

TEST(ConcatenationOpTest, TwoDimensionalOneInput) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 3}}, /*axis=*/0,
                          /*num_inputs=*/1);
  m0.SetInput(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(ConcatenationOpTest, TwoInputsTwoAxesNegativeAxes) {
  // We will concatenate two tensors along different dimensions.
  auto tensor0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto tensor1 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 3}}, /*axis=*/0,
                          /*num_inputs=*/2);
  m0.SetInput(0, tensor0);
  m0.SetInput(1, tensor1);
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  ConcatenationOpModel m0_negative({TensorType_FLOAT32, {2, 3}}, /*axis=*/-2,
                                   /*num_inputs=*/2);
  m0_negative.SetInput(0, tensor0);
  m0_negative.SetInput(1, tensor1);
  m0_negative.Invoke();
  EXPECT_THAT(m0_negative.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  ConcatenationOpModel m1({TensorType_FLOAT32, {2, 3}}, /*axis=*/1,
                          /*num_inputs=*/2);
  m1.SetInput(0, tensor0);
  m1.SetInput(1, tensor1);
  m1.Invoke();
  EXPECT_THAT(m1.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));

  ConcatenationOpModel m1_negative({TensorType_FLOAT32, {2, 3}}, /*axis=*/-1,
                                   /*num_inputs=*/2);
  m1_negative.SetInput(0, tensor0);
  m1_negative.SetInput(1, tensor1);
  m1_negative.Invoke();
  EXPECT_THAT(m1_negative.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

TEST(ConcatenationOpTest, FourInputs) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 1, 2}}, /*axis=*/2,
                          /*num_inputs=*/4);
  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              }));
}

TYPED_TEST(QuantizedConcatenationOpTest, FourInputsQuantizedUint8) {
  QuantizedConcatenationOpModel m0(
      {TestFixture::tensor_type, {2, 1, 2}, -12.7, 12.8},
      /*axis=*/2,
      /*num_inputs=*/4);

  using TestType = typename TestFixture::TestType;

  m0.SetInput<TestType>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput<TestType>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<TestType>(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput<TestType>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetDequantizedOutput<TestType>(),
              ElementsAreArray(ArrayFloatNear({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              })));

  if (TestFixture::tensor_type == TensorType_UINT8) {
    EXPECT_THAT(m0.GetOutput<TestType>(),
                ElementsAreArray({
                    137, 157, 138, 158, 139, 159, 140, 160,  //
                    167, 197, 168, 198, 169, 199, 170, 200,  //
                }));
  }

  if (TestFixture::tensor_type == TensorType_INT16) {
    EXPECT_THAT(m0.GetOutput<TestType>(),
                ElementsAreArray({
                    2441, 7581, 2698, 7838, 2955, 8095,      //
                    3212, 8352, 10151, 17861, 10408, 18118,  //
                    10665, 18375, 10922, 18632,              //
                }));
  }
}

TEST(ConcatenationOpTest, FourInputsQuantizedInt8) {
  QuantizedConcatenationOpModel m0({TensorType_INT8, {2, 1, 2}, -12.7, 12.8},
                                   /*axis=*/2,
                                   /*num_inputs=*/4);

  m0.SetInput<int8_t>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput<int8_t>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<int8_t>(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput<int8_t>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  1, 3, 1.1, 3.1, 1.2, 3.2, 1.3, 3.3,  //
                  4, 7, 4.1, 7.1, 4.2, 7.2, 4.3, 7.3   //
              })));
  EXPECT_THAT(m0.GetOutput<int8_t>(), ElementsAreArray({
                                          9, 29, 10, 30, 11, 31, 12, 32,   //
                                          39, 69, 40, 70, 41, 71, 42, 72,  //
                                      }));
}

TYPED_TEST(QuantizedConcatenationOpTest, FourInputsQuantizedMixedRange) {
  QuantizedConcatenationOpModel m0(
      {{TestFixture::tensor_type, {2, 1, 2}, -10.7, 10.8},
       {TestFixture::tensor_type, {2, 1, 2}, 0, 12.8},
       {TestFixture::tensor_type, {2, 1, 2}, -11, 11.8},
       {TestFixture::tensor_type, {2, 1, 2}, 0, 7.4}},
      /*axis=*/2, /*num_inputs=*/4,
      {TestFixture::tensor_type, {2, 1, 2}, -12.7, 12.8});

  using TestType = typename TestFixture::TestType;

  m0.SetInput<TestType>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput<TestType>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<TestType>(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput<TestType>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetDequantizedOutput<TestType>(),
              ElementsAreArray(ArrayFloatNear({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              })));

  if (TestFixture::tensor_type == TensorType_UINT8) {
    EXPECT_THAT(m0.GetOutput<TestType>(),
                ElementsAreArray({
                    137, 157, 138, 158, 139, 159, 140, 160,  //
                    167, 197, 168, 198, 169, 199, 170, 200,  //
                }));
  }

  if (TestFixture::tensor_type == TensorType_INT16) {
    EXPECT_THAT(m0.GetOutput<TestType>(),
                ElementsAreArray({
                    2441, 7581, 2698, 7838, 2955, 8095, 3212, 8352,          //
                    10151, 17861, 10408, 18118, 10665, 18375, 10922, 18632,  //
                }));
  }
}

TYPED_TEST(QuantizedConcatenationOpTest,
           FourInputsQuantizedMixedRangeClampingLogic) {
  QuantizedConcatenationOpModel m0(
      {{TestFixture::tensor_type, {2, 1, 2}, -10.7, 10.8},
       {TestFixture::tensor_type, {2, 1, 2}, 0, 12.8},
       {TestFixture::tensor_type, {2, 1, 2}, -11, 11.8},
       {TestFixture::tensor_type, {2, 1, 2}, 0, 7.4}},
      /*axis=*/2, /*num_inputs=*/4,
      {TestFixture::tensor_type, {2, 1, 2}, -1., 1.});

  using TestType = typename TestFixture::TestType;

  m0.SetInput<TestType>(0, {1.0f, -3.0f, -4.0f, -7.0f});
  m0.SetInput<TestType>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<TestType>(2, {1.2f, -3.2f, -4.2f, 7.2f});
  m0.SetInput<TestType>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetDequantizedOutput<TestType>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f,   //
                      -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,  //
                  },
                  4e-3)));

  if (TestFixture::tensor_type == TensorType_UINT8) {
    EXPECT_THAT(m0.GetOutput<TestType>(),
                ElementsAreArray({
                    255, 0, 255, 255, 255, 0, 255, 255,  //
                    0, 0, 255, 255, 0, 255, 255, 255,    //
                }));
  }

  if (TestFixture::tensor_type == TensorType_INT16) {
    EXPECT_THAT(
        m0.GetOutput<TestType>(),
        ElementsAreArray({
            32765, -32768, 32767, 32767, 32767, -32768, 32767, 32767,   //
            -32768, -32768, 32767, 32767, -32768, 32767, 32767, 32767,  //
        }));
  }
}

TYPED_TEST(QuantizedConcatenationOpTest, ThreeDimensionalNonQuantizedOneInput) {
  using TestType = typename TestFixture::TestType;

  QuantizedConcatenationOpModel m0({TestFixture::tensor_type,
                                    {2, 1, 2},
                                    std::numeric_limits<TestType>::min(),
                                    std::numeric_limits<TestType>::max()},
                                   /*axis=*/1,
                                   /*num_inputs=*/1);

  m0.SetInput<TestType>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput<TestType>(),
              ElementsAreArray(ArrayFloatNear({1.0f, 3.0f, 4.0f, 7.0f})));
}

TYPED_TEST(QuantizedConcatenationOpTest, OneTrivialNonQuantizedInput) {
  using TestType = typename TestFixture::TestType;

  QuantizedConcatenationOpModel m0({TestFixture::tensor_type,
                                    {1},
                                    std::numeric_limits<TestType>::min(),
                                    std::numeric_limits<TestType>::max()},
                                   /*axis=*/0,
                                   /*num_inputs=*/1);
  m0.SetInput<TestType>(0, {5.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput<TestType>(), ::testing::ElementsAre(5));
}

TYPED_TEST(QuantizedConcatenationOpTest, TwoDimensionalNonQuantizedOneInput) {
  using TestType = typename TestFixture::TestType;

  QuantizedConcatenationOpModel m0({TestFixture::tensor_type,
                                    {2, 3},
                                    std::numeric_limits<TestType>::min(),
                                    std::numeric_limits<TestType>::max()},
                                   /*axis=*/0,
                                   /*num_inputs=*/1);
  m0.SetInput<TestType>(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput<TestType>(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(QuantizedConcatenationOpTest,
           TwoInputsTwoAxesNegativeAxesNonQuantized) {
  using TestType = typename TestFixture::TestType;

  // We will concatenate two tensors along different dimensions.
  auto tensor0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto tensor1 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  QuantizedConcatenationOpModel m0({TestFixture::tensor_type,
                                    {2, 3},
                                    std::numeric_limits<TestType>::min(),
                                    std::numeric_limits<TestType>::max()},
                                   /*axis=*/0,
                                   /*num_inputs=*/2);
  m0.SetInput<TestType>(0, tensor0);
  m0.SetInput<TestType>(1, tensor1);
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput<TestType>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  QuantizedConcatenationOpModel m0_negative(
      {TestFixture::tensor_type,
       {2, 3},
       std::numeric_limits<TestType>::min(),
       std::numeric_limits<TestType>::max()},
      /*axis=*/-2,
      /*num_inputs=*/2);
  m0_negative.SetInput<TestType>(0, tensor0);
  m0_negative.SetInput<TestType>(1, tensor1);
  m0_negative.Invoke();
  EXPECT_THAT(m0_negative.GetOutput<TestType>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  QuantizedConcatenationOpModel m1({TestFixture::tensor_type,
                                    {2, 3},
                                    std::numeric_limits<TestType>::min(),
                                    std::numeric_limits<TestType>::max()},
                                   /*axis=*/1,
                                   /*num_inputs=*/2);
  m1.SetInput<TestType>(0, tensor0);
  m1.SetInput<TestType>(1, tensor1);
  m1.Invoke();
  EXPECT_THAT(m1.GetOutput<TestType>(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));

  QuantizedConcatenationOpModel m1_negative(
      {TestFixture::tensor_type,
       {2, 3},
       std::numeric_limits<TestType>::min(),
       std::numeric_limits<TestType>::max()},
      /*axis=*/-1,
      /*num_inputs=*/2);
  m1_negative.SetInput<TestType>(0, tensor0);
  m1_negative.SetInput<TestType>(1, tensor1);
  m1_negative.Invoke();
  EXPECT_THAT(m1_negative.GetOutput<TestType>(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

}  // namespace
}  // namespace tflite
