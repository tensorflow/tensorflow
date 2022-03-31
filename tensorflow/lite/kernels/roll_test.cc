/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {

using ::testing::ElementsAreArray;

class BaseRollOpModel : public SingleOpModel {
 public:
  BaseRollOpModel(TensorData input, const std::vector<int32_t>& shift,
                  const std::vector<int64_t>& axis, TensorData output) {
    if (input.type == TensorType_FLOAT32 || input.type == TensorType_INT64) {
      // Clear quantization params.
      input.min = input.max = 0.f;
      output.min = output.max = 0.f;
    }
    input_ = AddInput(input);
    shift_ = AddInput(
        TensorData(TensorType_INT32, {static_cast<int>(shift.size())}));
    axis_ =
        AddInput(TensorData(TensorType_INT64, {static_cast<int>(axis.size())}));
    output_ = AddOutput(output);

    SetCustomOp("Roll", {}, ops::custom::Register_ROLL);
    BuildInterpreter({GetShape(input_), GetShape(shift_), GetShape(axis_)});

    PopulateTensor(shift_, shift);
    PopulateTensor(axis_, axis);
  }

  template <typename T>
  inline typename std::enable_if<is_small_integer<T>::value, void>::type
  SetInput(const std::initializer_list<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  inline typename std::enable_if<!is_small_integer<T>::value, void>::type
  SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }

  template <typename T>
  inline typename std::enable_if<is_small_integer<T>::value,
                                 std::vector<float>>::type
  GetOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  template <typename T>
  inline
      typename std::enable_if<!is_small_integer<T>::value, std::vector<T>>::type
      GetOutput() {
    return ExtractVector<T>(output_);
  }

  void SetStringInput(std::initializer_list<std::string> data) {
    PopulateStringTensor(input_, data);
  }

 protected:
  int input_;
  int shift_;
  int axis_;
  int output_;
};

#ifdef GTEST_HAS_DEATH_TEST
TEST(RollOpTest, MismatchSize) {
  EXPECT_DEATH(BaseRollOpModel m(/*input=*/{TensorType_FLOAT32, {1, 2, 4, 2}},
                                 /*shift=*/{2, 3}, /*axis=*/{2},
                                 /*output=*/{TensorType_FLOAT32, {}}),
               "NumElements.shift. != NumElements.axis.");
}
#endif

template <typename T>
class RollOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, int8_t, int16_t, int64_t>;
TYPED_TEST_SUITE(RollOpTest, DataTypes);

TYPED_TEST(RollOpTest, Roll1D) {
  BaseRollOpModel m(
      /*input=*/{GetTensorType<TypeParam>(), {10}, 0, 31.875},
      /*shift=*/{3}, /*axis=*/{0},
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 31.875});
  m.SetInput<TypeParam>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(),
              ElementsAreArray({7, 8, 9, 0, 1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(RollOpTest, Roll3D) {
  BaseRollOpModel m(
      /*input=*/{GetTensorType<TypeParam>(), {2, 4, 4}, 0, 31.875},
      /*shift=*/{2, 6}, /*axis=*/{1, 2},
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 31.875});
  m.SetInput<TypeParam>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(),
              ElementsAreArray({10, 11, 8,  9,  14, 15, 12, 13, 2,  3,  0,
                                1,  6,  7,  4,  5,  26, 27, 24, 25, 30, 31,
                                28, 29, 18, 19, 16, 17, 22, 23, 20, 21}));
}

TYPED_TEST(RollOpTest, Roll3DNegativeShift) {
  BaseRollOpModel m(
      /*input=*/{GetTensorType<TypeParam>(), {2, 4, 4}, 0, 31.875},
      /*shift=*/{2, -5}, /*axis=*/{1, -1},
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 31.875});
  m.SetInput<TypeParam>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(),
              ElementsAreArray({9,  10, 11, 8,  13, 14, 15, 12, 1,  2,  3,
                                0,  5,  6,  7,  4,  25, 26, 27, 24, 29, 30,
                                31, 28, 17, 18, 19, 16, 21, 22, 23, 20}));
}

TYPED_TEST(RollOpTest, DuplicatedAxis) {
  BaseRollOpModel m(
      /*input=*/{GetTensorType<TypeParam>(), {2, 4, 4}, 0, 31.875},
      /*shift=*/{2, 3}, /*axis=*/{1, 1},
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 31.875});
  m.SetInput<TypeParam>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(),
              ElementsAreArray({12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,
                                7,  8,  9,  10, 11, 28, 29, 30, 31, 16, 17,
                                18, 19, 20, 21, 22, 23, 24, 25, 26, 27}));
}

TEST(RollOpTest, Roll3DTring) {
  BaseRollOpModel m(/*input=*/{TensorType_STRING, {2, 4, 4}},
                    /*shift=*/{2, 5}, /*axis=*/{1, 2},
                    /*output=*/{TensorType_STRING, {}});
  m.SetStringInput({"0",  "1",  "2",  "3",  "4",  "5",  "6",  "7",
                    "8",  "9",  "10", "11", "12", "13", "14", "15",
                    "16", "17", "18", "19", "20", "21", "22", "23",
                    "24", "25", "26", "27", "28", "29", "30", "31"});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<std::string>(),
      ElementsAreArray({"11", "8",  "9",  "10", "15", "12", "13", "14",
                        "3",  "0",  "1",  "2",  "7",  "4",  "5",  "6",
                        "27", "24", "25", "26", "31", "28", "29", "30",
                        "19", "16", "17", "18", "23", "20", "21", "22"}));
}

TEST(RollOpTest, BoolRoll3D) {
  BaseRollOpModel m(/*input=*/{TensorType_BOOL, {2, 4, 4}},
                    /*shift=*/{2, 3}, /*axis=*/{1, 2},
                    /*output=*/{TensorType_BOOL, {}});
  m.SetInput<bool>({true,  false, false, true,  true,  false, false, true,
                    false, false, false, true,  false, false, true,  true,
                    false, false, true,  false, false, false, true,  false,
                    false, true,  true,  false, false, true,  false, false});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<bool>(),
              ElementsAreArray({false, false, true,  false, false, true,  true,
                                false, false, false, true,  true,  false, false,
                                true,  true,  true,  true,  false, false, true,
                                false, false, false, false, true,  false, false,
                                false, true,  false, false}));
}

}  // namespace tflite
