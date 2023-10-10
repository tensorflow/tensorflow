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
#include <stdint.h>

#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

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

template <typename T>
class ConcatenationOpModel : public BaseConcatenationOpModel {
 public:
  using BaseConcatenationOpModel::BaseConcatenationOpModel;
  void SetInput(int index, std::initializer_list<T> data) {
    PopulateTensor<T>(index, data);
  }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
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

class BoolConcatenationOpModel : public BaseConcatenationOpModel {
 public:
  using BaseConcatenationOpModel::BaseConcatenationOpModel;
  void SetInput(int index, std::initializer_list<bool> data) {
    PopulateTensor(index, data);
  }
  std::vector<bool> GetOutput() { return ExtractVector<bool>(output_); }
};

TEST(ConcatenationOpTest, ThreeDimensionalOneInput) {
  ConcatenationOpModel<float> m0({TensorType_FLOAT32, {2, 1, 2}}, /*axis=*/1,
                                 /*num_inputs=*/1);
  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 3, 4, 7}));
}

TEST(ConcatenationOpTest, ThreeDimensionalOneInputUInt32) {
  ConcatenationOpModel<uint32_t> m0({TensorType_UINT32, {2, 1, 2}}, /*axis=*/1,
                                    /*num_inputs=*/1);
  m0.SetInput(0, {1, 3, 4, 7});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 3, 4, 7}));
}

TEST(ConcatenationOpTest, FiveDimensionalOneInput) {
  ConcatenationOpModel<float> m0({TensorType_FLOAT32, {2, 1, 2, 1, 3}},
                                 /*axis=*/2,
                                 /*num_inputs=*/1);
  m0.SetInput(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST(ConcatenationOpTest, FiveDimensionalOneInputUInt32) {
  ConcatenationOpModel<uint32_t> m0({TensorType_UINT32, {2, 1, 2, 1, 3}},
                                    /*axis=*/2,
                                    /*num_inputs=*/1);
  m0.SetInput(0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInput) {
  ConcatenationOpModel<float> m0({TensorType_FLOAT32, {2, 1, 2, 1, 3}},
                                 /*axis=*/0,
                                 /*num_inputs=*/2);
  m0.SetInput(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f});
  m0.SetInput(1, {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f,
                  22.0f, 23.0f, 24.0f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m0.GetOutput(),
      ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInputUInt32) {
  ConcatenationOpModel<uint32_t> m0({TensorType_UINT32, {2, 1, 2, 1, 3}},
                                    /*axis=*/0,
                                    /*num_inputs=*/2);
  m0.SetInput(0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m0.SetInput(1, {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m0.GetOutput(),
      ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInputNegativeAxes) {
  ConcatenationOpModel<float> m0({TensorType_FLOAT32, {2, 1, 2, 1, 3}},
                                 /*axis=*/-2,
                                 /*num_inputs=*/2);
  m0.SetInput(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f});
  m0.SetInput(1, {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f,
                  22.0f, 23.0f, 24.0f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({1, 2, 3, 13, 14, 15, 4,  5,  6,  16, 17, 18,
                                7, 8, 9, 19, 20, 21, 10, 11, 12, 22, 23, 24}));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInputNegativeAxesUInt32) {
  ConcatenationOpModel<uint32_t> m0({TensorType_UINT32, {2, 1, 2, 1, 3}},
                                    /*axis=*/-2,
                                    /*num_inputs=*/2);
  m0.SetInput(0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m0.SetInput(1, {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({1, 2, 3, 13, 14, 15, 4,  5,  6,  16, 17, 18,
                                7, 8, 9, 19, 20, 21, 10, 11, 12, 22, 23, 24}));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInputQuantizedUint8) {
  QuantizedConcatenationOpModel m0(
      {TensorType_UINT8, {2, 1, 2, 1, 3}, -12.7, 12.8},
      /*axis=*/0,
      /*num_inputs=*/2);

  m0.SetInput<uint8_t>(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                           10.0f, 11.0f, 12.0f});
  m0.SetInput<uint8_t>(1, {1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f, 9.1f,
                           10.1f, 11.1f, 12.1f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  1.0f, 2.0f,  3.0f,  4.0f,  5.0f, 6.0f,  7.0f,  8.0f,
                  9.0f, 10.0f, 11.0f, 12.0f, 1.1f, 2.1f,  3.1f,  4.1f,
                  5.1f, 6.1f,  7.1f,  8.1f,  9.1f, 10.1f, 11.1f, 12.1f,
              })));
  EXPECT_THAT(
      m0.GetOutput<uint8_t>(),
      ElementsAreArray({
          137, 147, 157, 167, 177, 187, 197, 207, 217, 227, 237, 247, 138,  //
          148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248,
      }));
}

TEST(ConcatenationOpTest, ThreeDimensionalTwoInputsDifferentShapes) {
  ConcatenationOpModel<float> m0(
      {{TensorType_FLOAT32, {2, 1, 2}}, {TensorType_FLOAT32, {2, 3, 2}}},
      /*axis=*/1, /*num_inputs=*/2, TensorType_FLOAT32);
  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput(1, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 3, 1, 2, 3, 4, 5, 6, 4, 7, 7,
                                                8, 9, 10, 11, 12}));
}

TEST(ConcatenationOpTest, ThreeDimensionalTwoInputsDifferentShapesUInt32) {
  ConcatenationOpModel<uint32_t> m0(
      {{TensorType_UINT32, {2, 1, 2}}, {TensorType_UINT32, {2, 3, 2}}},
      /*axis=*/1, /*num_inputs=*/2, TensorType_UINT32);
  m0.SetInput(0, {1, 3, 4, 7});
  m0.SetInput(1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 3, 1, 2, 3, 4, 5, 6, 4, 7, 7,
                                                8, 9, 10, 11, 12}));
}

#if GTEST_HAS_DEATH_TEST
TEST(ConcatenationOpTest, ThreeDimensionalTwoInputsDifferentShapesWrongAxis) {
  EXPECT_DEATH(
      ConcatenationOpModel<float> m0(
          {{TensorType_FLOAT32, {2, 1, 2}}, {TensorType_FLOAT32, {2, 3, 2}}},
          /*axis=*/0, /*num_inputs=*/2, TensorType_FLOAT32),
      "Cannot allocate tensors");
}

TEST(ConcatenationOpTest,
     ThreeDimensionalTwoInputsDifferentShapesWrongAxisUInt32) {
  EXPECT_DEATH(
      ConcatenationOpModel<uint32_t> m0(
          {{TensorType_UINT32, {2, 1, 2}}, {TensorType_UINT32, {2, 3, 2}}},
          /*axis=*/0, /*num_inputs=*/2, TensorType_UINT32),
      "Cannot allocate tensors");
}
#endif

TEST(ConcatenationOpTest, OneTrivialInput) {
  ConcatenationOpModel<float> m0({TensorType_FLOAT32, {1}}, /*axis=*/0,
                                 /*num_inputs=*/1);
  m0.SetInput(0, {5.0f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(), ::testing::ElementsAre(5));
}

TEST(ConcatenationOpTest, OneTrivialInputUInt32) {
  ConcatenationOpModel<uint32_t> m0({TensorType_UINT32, {1}}, /*axis=*/0,
                                    /*num_inputs=*/1);
  m0.SetInput(0, {5});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(), ::testing::ElementsAre(5));
}

TEST(ConcatenationOpTest, TwoDimensionalOneInput) {
  ConcatenationOpModel<float> m0({TensorType_FLOAT32, {2, 3}}, /*axis=*/0,
                                 /*num_inputs=*/1);
  m0.SetInput(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(ConcatenationOpTest, TwoDimensionalOneInputUInt32) {
  ConcatenationOpModel<uint32_t> m0({TensorType_UINT32, {2, 3}}, /*axis=*/0,
                                    /*num_inputs=*/1);
  m0.SetInput(0, {1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(ConcatenationOpTest, TwoInputsTwoAxesNegativeAxes) {
  // We will concatenate two tensors along different dimensions.
  auto tensor0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto tensor1 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  ConcatenationOpModel<float> m0({TensorType_FLOAT32, {2, 3}}, /*axis=*/0,
                                 /*num_inputs=*/2);
  m0.SetInput(0, tensor0);
  m0.SetInput(1, tensor1);
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  ConcatenationOpModel<float> m0_negative({TensorType_FLOAT32, {2, 3}},
                                          /*axis=*/-2,
                                          /*num_inputs=*/2);
  m0_negative.SetInput(0, tensor0);
  m0_negative.SetInput(1, tensor1);
  ASSERT_EQ(m0_negative.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0_negative.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  ConcatenationOpModel<float> m1({TensorType_FLOAT32, {2, 3}}, /*axis=*/1,
                                 /*num_inputs=*/2);
  m1.SetInput(0, tensor0);
  m1.SetInput(1, tensor1);
  ASSERT_EQ(m1.Invoke(), kTfLiteOk);
  EXPECT_THAT(m1.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));

  ConcatenationOpModel<float> m1_negative({TensorType_FLOAT32, {2, 3}},
                                          /*axis=*/-1,
                                          /*num_inputs=*/2);
  m1_negative.SetInput(0, tensor0);
  m1_negative.SetInput(1, tensor1);
  ASSERT_EQ(m1_negative.Invoke(), kTfLiteOk);
  EXPECT_THAT(m1_negative.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

TEST(ConcatenationOpTest, TwoInputsTwoAxesNegativeAxesUInt32) {
  // We will concatenate two tensors along different dimensions.
  std::initializer_list<uint32_t> tensor0 = {1, 2, 3, 4, 5, 6};
  std::initializer_list<uint32_t> tensor1 = {7, 8, 9, 10, 11, 12};

  ConcatenationOpModel<uint32_t> m0({TensorType_UINT32, {2, 3}}, /*axis=*/0,
                                    /*num_inputs=*/2);
  m0.SetInput(0, tensor0);
  m0.SetInput(1, tensor1);
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  ConcatenationOpModel<uint32_t> m0_negative({TensorType_UINT32, {2, 3}},
                                             /*axis=*/-2,
                                             /*num_inputs=*/2);
  m0_negative.SetInput(0, tensor0);
  m0_negative.SetInput(1, tensor1);
  ASSERT_EQ(m0_negative.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0_negative.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  ConcatenationOpModel<uint32_t> m1({TensorType_UINT32, {2, 3}}, /*axis=*/1,
                                    /*num_inputs=*/2);
  m1.SetInput(0, tensor0);
  m1.SetInput(1, tensor1);
  ASSERT_EQ(m1.Invoke(), kTfLiteOk);
  EXPECT_THAT(m1.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));

  ConcatenationOpModel<uint32_t> m1_negative({TensorType_UINT32, {2, 3}},
                                             /*axis=*/-1,
                                             /*num_inputs=*/2);
  m1_negative.SetInput(0, tensor0);
  m1_negative.SetInput(1, tensor1);
  ASSERT_EQ(m1_negative.Invoke(), kTfLiteOk);
  EXPECT_THAT(m1_negative.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

TEST(ConcatenationOpTest, FourInputs) {
  ConcatenationOpModel<float> m0({TensorType_FLOAT32, {2, 1, 2}}, /*axis=*/2,
                                 /*num_inputs=*/4);
  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput(3, {1.3f, 3.3f, 4.3f, 7.3f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              }));
}

TEST(ConcatenationOpTest, FourInputsUInt32) {
  ConcatenationOpModel<uint32_t> m0({TensorType_UINT32, {2, 1, 2}}, /*axis=*/2,
                                    /*num_inputs=*/4);
  m0.SetInput(0, {1, 3, 4, 7});
  m0.SetInput(1, {1, 3, 4, 7});
  m0.SetInput(2, {1, 3, 4, 7});
  m0.SetInput(3, {1, 3, 4, 7});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({
                                  1, 3, 1, 3, 1, 3, 1, 3,  //
                                  4, 7, 4, 7, 4, 7, 4, 7,  //
                              }));
}

TEST(ConcatenationOpTest, FourInputsQuantizedUint8) {
  QuantizedConcatenationOpModel m0({TensorType_UINT8, {2, 1, 2}, -12.7, 12.8},
                                   /*axis=*/2,
                                   /*num_inputs=*/4);

  m0.SetInput<uint8_t>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput<uint8_t>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<uint8_t>(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput<uint8_t>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              })));
  EXPECT_THAT(m0.GetOutput<uint8_t>(),
              ElementsAreArray({
                  137, 157, 138, 158, 139, 159, 140, 160,  //
                  167, 197, 168, 198, 169, 199, 170, 200,  //
              }));
}

template <typename Type>
struct ConcatenationOpTestTyped : public testing::Test {
  using TestType = Type;

  enum TensorType tensor_type =
      (std::is_same<Type, int16_t>::value ? TensorType_INT16 : TensorType_INT8);
};

using TestTypes = testing::Types<int8_t, int16_t>;
TYPED_TEST_CASE(ConcatenationOpTestTyped, TestTypes);

TYPED_TEST(ConcatenationOpTestTyped, FourInputsQuantizedInt8) {
  using TestType = typename TestFixture::TestType;

  const float kMin = -1;
  const float kMax =
      std::numeric_limits<TestType>::max() /
      static_cast<float>(std::numeric_limits<TestType>::max() + 1);

  QuantizedConcatenationOpModel m0(
      {TestFixture::tensor_type, {2, 1, 2}, 12.8f * kMin, 12.8f * kMax},
      /*axis=*/2,
      /*num_inputs=*/4);

  m0.SetInput<TestType>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput<TestType>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<TestType>(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput<TestType>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetDequantizedOutput<TestType>(),
              ElementsAreArray(ArrayFloatNear({
                  1, 3, 1.1, 3.1, 1.2, 3.2, 1.3, 3.3,  //
                  4, 7, 4.1, 7.1, 4.2, 7.2, 4.3, 7.3   //
              })));
}

TEST(ConcatenationOpTest, FourInputsQuantizedMixedRange) {
  QuantizedConcatenationOpModel m0({{TensorType_UINT8, {2, 1, 2}, -10.7, 10.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 12.8},
                                    {TensorType_UINT8, {2, 1, 2}, -11, 11.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 7.4}},
                                   /*axis=*/2, /*num_inputs=*/4,
                                   {TensorType_UINT8, {2, 1, 2}, -12.7, 12.8});

  m0.SetInput<uint8_t>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput<uint8_t>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<uint8_t>(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput<uint8_t>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              })));
  EXPECT_THAT(m0.GetOutput<uint8_t>(),
              ElementsAreArray({
                  137, 157, 138, 158, 139, 159, 140, 160,  //
                  167, 197, 168, 198, 169, 199, 170, 200,  //
              }));
}

TEST(ConcatenationOpTest, FourInputsQuantizedMixedRangeClampingLogic) {
  QuantizedConcatenationOpModel m0({{TensorType_UINT8, {2, 1, 2}, -10.7, 10.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 12.8},
                                    {TensorType_UINT8, {2, 1, 2}, -11, 11.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 7.4}},
                                   /*axis=*/2, /*num_inputs=*/4,
                                   {TensorType_UINT8, {2, 1, 2}, -1., 1.});

  m0.SetInput<uint8_t>(0, {1.0f, -3.0f, -4.0f, -7.0f});
  m0.SetInput<uint8_t>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<uint8_t>(2, {1.2f, -3.2f, -4.2f, 7.2f});
  m0.SetInput<uint8_t>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f,   //
                      -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,  //
                  },
                  4e-3)));
  EXPECT_THAT(m0.GetOutput<uint8_t>(),
              ElementsAreArray({
                  255, 0, 255, 255, 255, 0, 255, 255,  //
                  0, 0, 255, 255, 0, 255, 255, 255,    //
              }));
}

TEST(ConcatenationOpTest, ThreeDimensionalNonQuantizedOneInput) {
  QuantizedConcatenationOpModel m0(
      {TensorType_UINT8, {2, 1, 2}, 0, std::numeric_limits<uint8_t>::max()},
      /*axis=*/1,
      /*num_inputs=*/1);
  m0.SetInput<uint8_t>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({1.0f, 3.0f, 4.0f, 7.0f})));
}

TEST(ConcatenationOpTest, OneTrivialNonQuantizedInput) {
  QuantizedConcatenationOpModel m0(
      {TensorType_UINT8, {1}, 0, std::numeric_limits<uint8_t>::max()},
      /*axis=*/0,
      /*num_inputs=*/1);
  m0.SetInput<uint8_t>(0, {5.0f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput<uint8_t>(), ::testing::ElementsAre(5));
}

TEST(ConcatenationOpTest, TwoDimensionalNonQuantizedOneInput) {
  QuantizedConcatenationOpModel m0(
      {TensorType_UINT8, {2, 3}, 0, std::numeric_limits<uint8_t>::max()},
      /*axis=*/0,
      /*num_inputs=*/1);
  m0.SetInput<uint8_t>(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput<uint8_t>(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(ConcatenationOpTest, TwoInputsTwoAxesNegativeAxesNonQuantized) {
  // We will concatenate two tensors along different dimensions.
  auto tensor0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto tensor1 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  QuantizedConcatenationOpModel m0(
      {TensorType_UINT8, {2, 3}, 0, std::numeric_limits<uint8_t>::max()},
      /*axis=*/0,
      /*num_inputs=*/2);
  m0.SetInput<uint8_t>(0, tensor0);
  m0.SetInput<uint8_t>(1, tensor1);
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput<uint8_t>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  QuantizedConcatenationOpModel m0_negative(
      {TensorType_UINT8, {2, 3}, 0, std::numeric_limits<uint8_t>::max()},
      /*axis=*/-2,
      /*num_inputs=*/2);
  m0_negative.SetInput<uint8_t>(0, tensor0);
  m0_negative.SetInput<uint8_t>(1, tensor1);
  ASSERT_EQ(m0_negative.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0_negative.GetOutput<uint8_t>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  QuantizedConcatenationOpModel m1(
      {TensorType_UINT8, {2, 3}, 0, std::numeric_limits<uint8_t>::max()},
      /*axis=*/1,
      /*num_inputs=*/2);
  m1.SetInput<uint8_t>(0, tensor0);
  m1.SetInput<uint8_t>(1, tensor1);
  ASSERT_EQ(m1.Invoke(), kTfLiteOk);
  EXPECT_THAT(m1.GetOutput<uint8_t>(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));

  QuantizedConcatenationOpModel m1_negative(
      {TensorType_UINT8, {2, 3}, 0, std::numeric_limits<uint8_t>::max()},
      /*axis=*/-1,
      /*num_inputs=*/2);
  m1_negative.SetInput<uint8_t>(0, tensor0);
  m1_negative.SetInput<uint8_t>(1, tensor1);
  ASSERT_EQ(m1_negative.Invoke(), kTfLiteOk);
  EXPECT_THAT(m1_negative.GetOutput<uint8_t>(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

TEST(ConcatenationOpTest, BoolTypeOneInput) {
  BoolConcatenationOpModel m0({TensorType_BOOL, {2, 1, 2}}, /*axis=*/1,
                              /*num_inputs=*/1);
  m0.SetInput(0, {true, false, false, true});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({true, false, false, true}));
}

TEST(ConcatenationOpTest, BoolTypeTwoInputs) {
  BoolConcatenationOpModel m0(
      {{TensorType_BOOL, {2, 1, 2}}, {TensorType_BOOL, {2, 3, 2}}},
      /*axis=*/1, /*num_inputs=*/2, TensorType_BOOL);
  m0.SetInput(0, {false, false, false, false});
  m0.SetInput(1, {true, true, true, true, true, true, true, true, true, true,
                  true, true});
  ASSERT_EQ(m0.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m0.GetOutput(),
      ElementsAreArray({false, false, true, true, true, true, true, true, false,
                        false, true, true, true, true, true, true}));
}

enum class TestInputType {
  kPersistentRo = 0,
  kOnePersistentRo = 1,
  kDefault = 2,
};

struct PersistentTestCase {
  TestInputType test_type;
  TensorType tensor_type;
  bool is_quantized = false;
};

template <typename T>
class PersistentConcatenationOpModel : public SingleOpModel {
 public:
  PersistentConcatenationOpModel(const std::vector<TensorData>& input_template,
                                 int axis, const TensorData& output_template,
                                 PersistentTestCase test_case,
                                 std::vector<std::vector<T>> input_data_list)
      : input_data_list_(input_data_list), test_case_(test_case) {
    const int num_inputs = input_data_list.size();
    std::vector<std::vector<int>> all_input_shapes;
    CHECK_EQ(input_template.size(), num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      int id;
      all_input_shapes.push_back(input_template[i].shape);
      id = AddInput(input_template[i]);
      concat_inputs_.push_back(id);
    }
    output_ = AddOutput(output_template);
    SetBuiltinOp(
        BuiltinOperator_CONCATENATION, BuiltinOptions_ConcatenationOptions,
        CreateConcatenationOptions(builder_, axis, ActivationFunctionType_NONE)
            .Union());
    BuildInterpreter(all_input_shapes, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true,
                     /*allocate_and_delegate=*/false);

    int num_persistent_inputs = 0;
    if (test_case_.test_type == TestInputType::kPersistentRo) {
      num_persistent_inputs = num_inputs;
    } else if (test_case_.test_type == TestInputType::kOnePersistentRo) {
      num_persistent_inputs = 1;
    }

    for (int i = 0; i < num_persistent_inputs; ++i) {
      interpreter_->tensor(concat_inputs_[i])->allocation_type =
          kTfLitePersistentRo;
      std::vector<T>& input_data = input_data_list[i];
      interpreter_->ResizeInputTensorStrict(concat_inputs_[i],
                                            input_template[i].shape);
      if (test_case.is_quantized) {
        QuantizeAndPopulate<int8_t>(concat_inputs_[i], FloatVector(input_data));
      } else {
        PopulateTensor(concat_inputs_[i], input_data);
      }
    }
    AllocateAndDelegate(true);
  }

  std::vector<float> FloatVector(std::vector<T> data) {
    std::vector<float> ret;
    for (T t : data) {
      ret.push_back(static_cast<float>(t));
    }
    return ret;
  }

  void PopulateInputTensors() {
    int start = -1;
    if (test_case_.test_type == TestInputType::kDefault) {
      start = 0;
    } else if (test_case_.test_type == TestInputType::kOnePersistentRo) {
      start = 1;
    }
    if (start < 0) {
      return;
    }
    for (int i = start; i < input_data_list_.size(); ++i) {
      if (test_case_.is_quantized) {
        QuantizeAndPopulate<int8_t>(concat_inputs_[i],
                                    FloatVector(input_data_list_[i]));
      } else {
        std::vector<T> v(input_data_list_[i]);
        PopulateTensor(concat_inputs_[i], v);
      }
    }
  }

  bool IsPersistentOutput() {
    const TfLiteTensor* tensor = interpreter_->tensor(output_);
    return tensor->allocation_type == kTfLitePersistentRo;
  }

  std::vector<float> GetOutput() {
    if (test_case_.is_quantized) {
      return Dequantize<int8_t>(ExtractVector<int8_t>(output_),
                                GetScale(output_), GetZeroPoint(output_));
    }
    return FloatVector(ExtractVector<T>(output_));
  }

 protected:
  int output_;
  std::vector<std::vector<T>> input_data_list_;
  PersistentTestCase test_case_;
  std::vector<int> concat_inputs_;
};

template <typename T>
class ConcatenationOpPersistentModelTest : public ::testing::Test {
 public:
  static std::vector<PersistentTestCase> Range(bool is_quantized = false) {
    TensorType tensor_type = TensorType_FLOAT32;
    if (std::is_same<T, int32_t>::value) {
      tensor_type = TensorType_INT32;
    }
    if (std::is_same<T, uint32_t>::value) {
      tensor_type = TensorType_UINT32;
    }
    if (is_quantized) {
      tensor_type = TensorType_INT8;
    }
    return {{TestInputType::kDefault, tensor_type, is_quantized},
            {TestInputType::kPersistentRo, tensor_type, is_quantized}};
  }
};

using DataTypes = ::testing::Types<float, int32_t, uint32_t>;
TYPED_TEST_SUITE(ConcatenationOpPersistentModelTest, DataTypes);

TYPED_TEST(ConcatenationOpPersistentModelTest, PersistentTest) {
  for (PersistentTestCase test_case :
       ConcatenationOpPersistentModelTest<TypeParam>::Range()) {
    std::vector<std::vector<TypeParam>> input_data_lists = {
        {1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}};
    std::vector<TensorData> input_template = {{test_case.tensor_type, {2, 3}},
                                              {test_case.tensor_type, {2, 3}}};
    TensorData output_template = {test_case.tensor_type, {4, 3}};
    PersistentConcatenationOpModel<TypeParam> m0(input_template, /*axis=*/0,
                                                 output_template, test_case,
                                                 input_data_lists);
    m0.PopulateInputTensors();
    ASSERT_EQ(m0.Invoke(), kTfLiteOk);
    ASSERT_EQ(m0.IsPersistentOutput(),
              test_case.test_type == TestInputType::kPersistentRo);
    EXPECT_THAT(
        m0.GetOutput(),
        ElementsAreArray(ArrayFloatNear(
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0})));
  }
}

TYPED_TEST(ConcatenationOpPersistentModelTest, QuantizedPersistentTest) {
  const bool is_quantized = true;
  for (PersistentTestCase test_case :
       ConcatenationOpPersistentModelTest<TypeParam>::Range(is_quantized)) {
    std::vector<std::vector<TypeParam>> input_data_lists = {
        {1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}};
    float scale = 12.0 / 255.0;
    int zero_point = -128;
    std::vector<TensorData> input_template = {
        {test_case.tensor_type, {2, 3}, 0.0, 12.0, scale, zero_point},
        {test_case.tensor_type, {2, 3}, 0.0, 12.0, scale, zero_point},
    };
    TensorData output_template = {
        test_case.tensor_type, {4, 3}, 0.0, 12.0, scale, zero_point};
    PersistentConcatenationOpModel<TypeParam> m0(input_template, /*axis=*/0,
                                                 output_template, test_case,
                                                 input_data_lists);
    m0.PopulateInputTensors();
    ASSERT_EQ(m0.Invoke(), kTfLiteOk);
    ASSERT_EQ(m0.IsPersistentOutput(),
              test_case.test_type == TestInputType::kPersistentRo);
    EXPECT_THAT(
        m0.GetOutput(),
        ElementsAreArray(ArrayFloatNear(
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
            1e-1)));
  }
}

}  // namespace
}  // namespace tflite
