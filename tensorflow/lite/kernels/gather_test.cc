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
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename InputType, typename PositionsType>
class GatherOpModel : public SingleOpModel {
 public:
  GatherOpModel(const TensorData& input, const TensorData& positions,
                bool constant_tensor, const std::vector<InputType>& input_data,
                const std::vector<PositionsType>& positions_data, int axis = 0,
                int batch_dims = 0) {
    if (constant_tensor) {
      input_ = AddConstInput(input, input_data);
      positions_ = AddConstInput(positions, positions_data);
    } else {
      input_ = AddInput(input);
      positions_ = AddInput(positions);
    }
    output_ = AddOutput(input.type);
    SetBuiltinOp(BuiltinOperator_GATHER, BuiltinOptions_GatherOptions,
                 CreateGatherOptions(builder_, axis, batch_dims).Union());
    BuildInterpreter({GetShape(input_), GetShape(positions_)});
    if (!constant_tensor) {
      if (input.type == TensorType_INT4) {
        SetInputInt4(input_, input_data,
                     std::is_same<std::string, InputType>());
      } else {
        SetInput(input_, input_data, std::is_same<std::string, InputType>());
      }
      SetPositions(positions_data);
    }
  }

  template <typename T>
  void SetInput(int input, const std::vector<T> data, std::false_type) {
    PopulateTensor<T>(input, data);
  }

  // Overload for string inputs.
  template <typename T>
  void SetInput(int input, const std::vector<T> data, std::true_type) {
    PopulateStringTensor(input_, data);
  }

  template <typename T>
  void SetInputInt4(int input, const std::vector<T> data, std::false_type) {
    auto non_const = *const_cast<std::vector<T>*>(&data);
    std::vector<int8_t> data_int8(non_const.size());
    std::copy(non_const.begin(), non_const.end(), data_int8.begin());
    PopulateTensor4bit(input, 0, data_int8.data(),
                       data_int8.data() + data_int8.size());
  }

  template <typename T>
  void SetInputInt4(int input, const std::vector<T> data, std::true_type) {
    // Unsupported
  }

  void SetPositions(const std::vector<PositionsType>& data) {
    PopulateTensor<PositionsType>(positions_, data);
  }

  std::vector<InputType> GetOutput() {
    return ExtractVector<InputType>(output_);
  }

  std::vector<std::string> GetStringOutput() {
    return ExtractVector<std::string>(output_);
  }

  std::vector<int8_t> GetInt4Output() {
    const auto* tensor = interpreter_->tensor(output_);
    const std::vector<int8_t> data_int8 = std::vector<int8_t>(
        tensor->data.raw, tensor->data.raw + GetTensorSize(output_));
    int num_elements = 1;
    auto shape = GetTensorShape(output_);
    for (int i = 0; i < shape.size(); i++) {
      num_elements *= shape[i];
    }
    std::vector<int8_t> inflated_output(num_elements);
    tensor_utils::UnpackPackedIntToInt8(data_int8.data(), num_elements,
                                        /*bit_width=*/4,
                                        inflated_output.data());
    return inflated_output;
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int positions_;
  int output_;
};

struct GatherOpTest : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(ConstantTensor, GatherOpTest, testing::Bool());

TEST_P(GatherOpTest, Shuffle) {
  bool constant_tensor = GetParam();
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {2, 2}},
                                  {TensorType_INT32, {2}}, constant_tensor,
                                  {-2.0, 0.2, 0.7, 0.8}, {1, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({0.7, 0.8, -2, 0.2})));
}

TEST_P(GatherOpTest, Test0DIndex) {
  bool constant_tensor = GetParam();
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {2, 2}},
                                  {TensorType_INT32, {}}, constant_tensor,
                                  {-2.0, 0.2, 0.7, 0.8}, {1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({0.7, 0.8})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
}

TEST_P(GatherOpTest, Test0DIndexWith0DResult) {
  bool constant_tensor = GetParam();
  // 0D tensor is special case in current TFLite. Test it once to make sure
  // existing workarounds are fine with it.
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {3}},
                                  {TensorType_INT32, {}}, constant_tensor,
                                  {1.0, 2.0, 3.0}, {1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({2.0})));
  EXPECT_TRUE(m.GetOutputShape().empty());
}

TEST_P(GatherOpTest, Test1DInput1DIndex) {
  bool constant_tensor = GetParam();
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {3}},
                                  {TensorType_INT32, {1}}, constant_tensor,
                                  {1.0, 3.0, 5.0}, {1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({3.0})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
}

TEST_P(GatherOpTest, Test2DIndexWith2DResult) {
  bool constant_tensor = GetParam();
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {3}},
                                  {TensorType_INT32, {1, 2}}, constant_tensor,
                                  {1.0, 2.0, 3.0}, {1, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({2.0, 1.0})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
}

TEST_P(GatherOpTest, Duplicate) {
  bool constant_tensor = GetParam();
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {1, 2, 2}},
                                  {TensorType_INT32, {2}}, constant_tensor,
                                  {-2.0, 0.2, 0.7, 0.8}, {0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear({-2, 0.2, 0.7, 0.8, -2, 0.2, 0.7, 0.8})));
}

TEST_P(GatherOpTest, Slice) {
  bool constant_tensor = GetParam();
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {4, 1}},
                                  {TensorType_INT32, {2}}, constant_tensor,
                                  {-2.0, 0.2, 0.7, 0.8}, {1, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({0.2, 0.8})));
}

TEST_P(GatherOpTest, Axis1) {
  bool constant_tensor = GetParam();
  const int axis = 1;
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {1, 2, 3}},
                                  {TensorType_INT32, {2}}, constant_tensor,
                                  {1, 2, 3, 4, 5, 6}, {1, 0}, axis);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({4, 5, 6, 1, 2, 3})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3}));
}

TEST_P(GatherOpTest, Axis10DIndex) {
  bool constant_tensor = GetParam();
  const int axis = 1;
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {1, 3, 2}},
                                  {TensorType_INT32, {}}, constant_tensor,
                                  {1, 2, 3, 4, 5, 6}, {1}, axis);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({3, 4})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
}

TEST_P(GatherOpTest, Axis1Slice) {
  bool constant_tensor = GetParam();
  const int axis = 1;
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {1, 4, 2}},
                                  {TensorType_INT32, {2}}, constant_tensor,
                                  {1, 2, 3, 4, 5, 6, 7, 8}, {3, 1}, axis);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({7, 8, 3, 4})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2}));
}

TEST_P(GatherOpTest, LastAxis) {
  const int axis = -1;
  bool constant_tensor = GetParam();
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {1, 2, 3}},
                                  {TensorType_INT32, {2}}, constant_tensor,
                                  {1, 2, 3, 4, 5, 6}, {2, 0}, axis);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({3, 1, 6, 4})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2}));
}

TEST_P(GatherOpTest, LastAxis0DIndex) {
  bool constant_tensor = GetParam();
  const int axis = -1;
  GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {1, 2, 3}},
                                  {TensorType_INT32, {}}, constant_tensor,
                                  {1, 2, 3, 4, 5, 6}, {2}, axis);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({3, 6})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
}

using TestTypes = testing::Types<int8_t, uint8_t, int16_t, int32_t, int64_t,
                                 float, Eigen::half, Eigen::bfloat16>;

template <typename T>
struct TypedGatherOpTest : public testing::Test {};

TYPED_TEST_CASE(TypedGatherOpTest, TestTypes);

TYPED_TEST(TypedGatherOpTest, Int32Indices) {
  for (bool constant_tensor : {true, false}) {
    TensorType tensor_type = GetTensorType<TypeParam>();
    GatherOpModel<TypeParam, int32_t> m(
        {tensor_type, {2, 2}}, {TensorType_INT32, {2}}, constant_tensor,
        {TypeParam(13), TypeParam(120), TypeParam(14), TypeParam(15)}, {1, 0});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);

    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({TypeParam(14), TypeParam(15), TypeParam(13),
                                  TypeParam(120)}));
  }
}

TYPED_TEST(TypedGatherOpTest, Int64Indices) {
  for (bool constant_tensor : {true, false}) {
    TensorType tensor_type = GetTensorType<TypeParam>();
    GatherOpModel<TypeParam, int64_t> m(
        {tensor_type, {2, 2}}, {TensorType_INT64, {2}}, constant_tensor,
        {TypeParam(13), TypeParam(120), TypeParam(14), TypeParam(15)}, {1, 0});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);

    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({TypeParam(14), TypeParam(15), TypeParam(13),
                                  TypeParam(120)}));
  }
}

TEST(GatherOpTest, SimpleString) {
  GatherOpModel<std::string, int32_t> m(
      {TensorType_STRING, {3}}, {TensorType_INT32, {2}},
      /*constant_tensor=*/false, {"A", "B", "C"}, {0, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  ASSERT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetStringOutput(), ElementsAreArray({"A", "C"}));
}

TEST_P(GatherOpTest, 2DIndexString) {
  GatherOpModel<std::string, int32_t> m(
      {TensorType_STRING, {3}}, {TensorType_INT32, {2, 3}},
      /*constant_tensor=*/false, {"A", "B", "C"}, {0, 2, 1, 1, 0, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  ASSERT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetStringOutput(),
              ElementsAreArray({"A", "C", "B", "B", "A", "C"}));
}

TYPED_TEST(TypedGatherOpTest, BatchDims2) {
  for (bool constant_tensor : {true, false}) {
    TensorType tensor_type = GetTensorType<TypeParam>();
    GatherOpModel<TypeParam, int32_t> m(
        {tensor_type, {2, 2, 3, 5}}, {TensorType_INT32, {2, 2, 2}},
        constant_tensor,
        {TypeParam(0),  TypeParam(1),  TypeParam(2),  TypeParam(3),
         TypeParam(4),  TypeParam(5),  TypeParam(6),  TypeParam(7),
         TypeParam(8),  TypeParam(9),  TypeParam(10), TypeParam(11),
         TypeParam(12), TypeParam(13), TypeParam(14), TypeParam(15),
         TypeParam(16), TypeParam(17), TypeParam(18), TypeParam(19),
         TypeParam(20), TypeParam(21), TypeParam(22), TypeParam(23),
         TypeParam(24), TypeParam(25), TypeParam(26), TypeParam(27),
         TypeParam(28), TypeParam(29), TypeParam(30), TypeParam(31),
         TypeParam(32), TypeParam(33), TypeParam(34), TypeParam(35),
         TypeParam(36), TypeParam(37), TypeParam(38), TypeParam(39),
         TypeParam(40), TypeParam(41), TypeParam(42), TypeParam(43),
         TypeParam(44), TypeParam(45), TypeParam(46), TypeParam(47),
         TypeParam(48), TypeParam(49), TypeParam(50), TypeParam(51),
         TypeParam(52), TypeParam(53), TypeParam(54), TypeParam(55),
         TypeParam(56), TypeParam(57), TypeParam(58), TypeParam(59)},
        {1, 0, 0, 1, 1, 0, 0, 1},
        /*axis=*/2,
        /*batch_dims=*/2);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);

    ASSERT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2, 5}));
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(
            {TypeParam(5),  TypeParam(6),  TypeParam(7),  TypeParam(8),
             TypeParam(9),  TypeParam(0),  TypeParam(1),  TypeParam(2),
             TypeParam(3),  TypeParam(4),  TypeParam(15), TypeParam(16),
             TypeParam(17), TypeParam(18), TypeParam(19), TypeParam(20),
             TypeParam(21), TypeParam(22), TypeParam(23), TypeParam(24),
             TypeParam(35), TypeParam(36), TypeParam(37), TypeParam(38),
             TypeParam(39), TypeParam(30), TypeParam(31), TypeParam(32),
             TypeParam(33), TypeParam(34), TypeParam(45), TypeParam(46),
             TypeParam(47), TypeParam(48), TypeParam(49), TypeParam(50),
             TypeParam(51), TypeParam(52), TypeParam(53), TypeParam(54)}));
  }
}

TEST_P(GatherOpTest, BatchDims1) {
  bool constant_tensor = GetParam();
  GatherOpModel<int8_t, int32_t> m(
      {TensorType_INT8, {2, 2, 3, 5}}, {TensorType_INT32, {2, 2, 2}},
      constant_tensor,
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
       15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
       30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
       45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59},
      {1, 0, 0, 1, 1, 0, 0, 1},
      /*axis=*/2, /*batch_dims=*/1);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  ASSERT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2, 2, 5}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  0,  1,  2,  3,
                        4,  5,  6,  7,  8,  9,  20, 21, 22, 23, 24, 15, 16, 17,
                        18, 19, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 35, 36,
                        37, 38, 39, 30, 31, 32, 33, 34, 30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 50, 51, 52, 53, 54, 45, 46, 47, 48, 49,
                        45, 46, 47, 48, 49, 50, 51, 52, 53, 54}));
}

TEST_P(GatherOpTest, NegativeBatchDims) {
  bool constant_tensor = GetParam();
  GatherOpModel<int8_t, int32_t> m(
      {TensorType_INT8, {2, 2, 3, 5}}, {TensorType_INT32, {2, 2, 2}},
      constant_tensor,
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
       15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
       30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
       45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59},
      {1, 0, 0, 1, 1, 0, 0, 1},
      /*axis=*/2, /*batch_dims=*/-2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  ASSERT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2, 2, 5}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  0,  1,  2,  3,
                        4,  5,  6,  7,  8,  9,  20, 21, 22, 23, 24, 15, 16, 17,
                        18, 19, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 35, 36,
                        37, 38, 39, 30, 31, 32, 33, 34, 30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 50, 51, 52, 53, 54, 45, 46, 47, 48, 49,
                        45, 46, 47, 48, 49, 50, 51, 52, 53, 54}));
}

TEST_P(GatherOpTest, BatchDimsEqualIndexDims) {
  bool constant_tensor = GetParam();
  GatherOpModel<int8_t, int32_t> m(
      {TensorType_INT8, {2, 2, 2, 5}}, {TensorType_INT32, {2, 2, 2}},
      constant_tensor, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39},
      {1, 0, 0, 1, 1, 0, 0, 1},
      /*axis=*/3, /*batch_dims=*/3);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  ASSERT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 5, 10, 16, 21, 25, 30, 36}));
}

TEST_P(GatherOpTest, ErrorOnOutOfBoundsTooLarge) {
  bool constant_tensor = GetParam();
  if (constant_tensor) {
#if GTEST_HAS_DEATH_TEST
    EXPECT_DEATH(
        (GatherOpModel<float, int32_t>({TensorType_FLOAT32, {2, 2}},
                                       {TensorType_INT32, {2}}, constant_tensor,
                                       {
                                           -2.f, 0.2f,  //
                                           0.7f, 0.8f   //
                                       },
                                       {3, 1})),
        "Cannot allocate tensors");
#endif
  } else {
    GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {2, 2}},
                                    {TensorType_INT32, {2}}, constant_tensor,
                                    {
                                        -2.f, 0.2f,  //
                                        0.7f, 0.8f   //
                                    },
                                    {3, 1});
    EXPECT_EQ(m.Invoke(), kTfLiteError);
  }
}

TEST_P(GatherOpTest, ErrorOnOutOfBoundsNegative) {
  bool constant_tensor = GetParam();
  if (constant_tensor) {
#if GTEST_HAS_DEATH_TEST
    EXPECT_DEATH(
        (GatherOpModel<float, int32_t>({TensorType_FLOAT32, {2, 2}},
                                       {TensorType_INT32, {2}}, constant_tensor,
                                       {
                                           -2.f, 0.2f,  //
                                           0.7f, 0.8f   //
                                       },
                                       {-1, 0})),
        "Cannot allocate tensors");
#endif
  } else {
    GatherOpModel<float, int32_t> m({TensorType_FLOAT32, {2, 2}},
                                    {TensorType_INT32, {2}}, constant_tensor,
                                    {
                                        -2.f, 0.2f,  //
                                        0.7f, 0.8f   //
                                    },
                                    {-1, 0});
    ASSERT_EQ(m.Invoke(), kTfLiteError);
    m.SetPositions({-1, 0});
    EXPECT_EQ(m.Invoke(), kTfLiteError);
  }
}

TEST(GatherOpTest, BatchDims1Int4) {
  GatherOpModel<int8_t, int32_t> m(
      {TensorType_INT4, {2, 2, 3, 4}}, {TensorType_INT32, {2, 2, 2}}, false,
      {1,  2,  3,  4,  -1, -2, -3, -4, 0,  0,  0,  0,  1,  2,  3,  4,
       -1, -2, -3, -4, 0,  0,  0,  0,  4,  5,  6,  7,  -5, -6, -7, -8,
       0,  0,  0,  0,  4,  5,  6,  7,  -5, -6, -7, -8, 0,  0,  0,  0},
      {1, 0, 0, 1, 1, 0, 0, 1},
      /*axis=*/2, /*batch_dims=*/1);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  ASSERT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2, 2, 4}));

  EXPECT_THAT(m.GetInt4Output(),
              ElementsAreArray(
                  {-1, -2, -3, -4, 1, 2, 3, 4, 1, 2, 3, 4, -1, -2, -3, -4,
                   -1, -2, -3, -4, 1, 2, 3, 4, 1, 2, 3, 4, -1, -2, -3, -4,
                   -5, -6, -7, -8, 4, 5, 6, 7, 4, 5, 6, 7, -5, -6, -7, -8,
                   -5, -6, -7, -8, 4, 5, 6, 7, 4, 5, 6, 7, -5, -6, -7, -8}));
}

}  // namespace
}  // namespace tflite
