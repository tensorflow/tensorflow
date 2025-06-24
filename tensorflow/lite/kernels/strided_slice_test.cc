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
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

template <typename input_type>
class StridedSliceOpModel : public SingleOpModel {
 public:
  StridedSliceOpModel(std::initializer_list<int> input_shape,
                      std::initializer_list<int> begin_shape,
                      std::initializer_list<int> end_shape,
                      std::initializer_list<int> strides_shape,
                      const std::vector<input_type> input_data,
                      const std::vector<int> begin_data,
                      const std::vector<int> end_data,
                      const std::vector<int> strides_data, int begin_mask,
                      int end_mask, int ellipsis_mask, int new_axis_mask,
                      int shrink_axis_mask, bool constant_tensors,
                      bool offset = false) {
    if (constant_tensors) {
      input_ =
          AddConstInput(GetTensorType<input_type>(), input_data, input_shape);
      begin_ = AddConstInput(TensorType_INT32, begin_data, begin_shape);
      end_ = AddConstInput(TensorType_INT32, end_data, end_shape);
      strides_ = AddConstInput(TensorType_INT32, strides_data, strides_shape);
    } else if (offset) {
      input_ = AddInput(GetTensorType<input_type>());
      begin_ = AddInput(TensorType_INT32);
      end_ = AddConstInput(TensorType_INT32, end_data, end_shape);
      strides_ = AddConstInput(TensorType_INT32, strides_data, strides_shape);
    } else {
      input_ = AddInput(GetTensorType<input_type>());
      begin_ = AddInput(TensorType_INT32);
      end_ = AddInput(TensorType_INT32);
      strides_ = AddInput(TensorType_INT32);
    }
    output_ = AddOutput(GetTensorType<input_type>());
    SetBuiltinOp(
        BuiltinOperator_STRIDED_SLICE, BuiltinOptions_StridedSliceOptions,
        CreateStridedSliceOptions(builder_, begin_mask, end_mask, ellipsis_mask,
                                  new_axis_mask, shrink_axis_mask, offset)
            .Union());
    BuildInterpreter({input_shape, begin_shape, end_shape, strides_shape});
    if (!constant_tensors) {
      if (!input_data.empty()) {
        SetInput(input_data, std::is_same<std::string, input_type>());
      }
      SetBegin(begin_data);
      SetEnd(end_data);
      SetStrides(strides_data);
    } else if (offset) {
      if (!input_data.empty()) {
        SetInput(input_data, std::is_same<std::string, input_type>());
      }
      SetBegin(begin_data);
    }
  }

  // Constant input, strides and end with offset.
  StridedSliceOpModel(std::initializer_list<int> input_shape,
                      std::initializer_list<int> begin_shape,
                      std::initializer_list<int> end_shape,
                      std::initializer_list<int> strides_shape,
                      const std::vector<input_type> input_data,
                      const std::vector<int> begin_data,
                      const std::vector<int> end_data,
                      const std::vector<int> strides_data, int begin_mask,
                      int end_mask, int ellipsis_mask, int new_axis_mask,
                      int shrink_axis_mask) {
    input_ =
        AddConstInput(GetTensorType<input_type>(), input_data, input_shape);
    begin_ = AddInput(TensorType_INT32);
    end_ = AddConstInput(TensorType_INT32, end_data, end_shape);
    strides_ = AddConstInput(TensorType_INT32, strides_data, strides_shape);
    output_ = AddOutput(GetTensorType<input_type>());
    SetBuiltinOp(BuiltinOperator_STRIDED_SLICE,
                 BuiltinOptions_StridedSliceOptions,
                 CreateStridedSliceOptions(builder_, begin_mask, end_mask,
                                           ellipsis_mask, new_axis_mask,
                                           shrink_axis_mask, /*offset=*/true)
                     .Union());
    BuildInterpreter({input_shape, begin_shape, end_shape, strides_shape});
    SetBegin(begin_data);
  }

  template <typename T>
  void SetInput(const std::vector<T> data, std::false_type) {
    PopulateTensor<input_type>(input_, data);
  }
  template <typename T>
  void SetInput(const std::vector<T> data, std::true_type) {
    PopulateStringTensor(input_, data);
  }
  void SetBegin(const std::vector<int32_t> data) {
    PopulateTensor<int32_t>(begin_, data);
  }
  void SetEnd(const std::vector<int32_t> data) {
    PopulateTensor<int32_t>(end_, data);
  }
  void SetStrides(const std::vector<int32_t> data) {
    PopulateTensor<int32_t>(strides_, data);
  }

  std::vector<input_type> GetOutput() {
    return ExtractVector<input_type>(output_);
  }
  std::vector<std::string> GetStringOutput() {
    return ExtractVector<std::string>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  const TfLiteTensor* GetOutputTensor(int index) {
    return interpreter_->output_tensor(index);
  }

 private:
  int input_;
  int begin_;
  int end_;
  int strides_;
  int output_;
};

template <typename T>
class StridedSliceOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, Eigen::half, Eigen::bfloat16, uint8_t,
                                   uint32_t, int8_t, int16_t, int32_t>;
TYPED_TEST_SUITE(StridedSliceOpTest, DataTypes);

template <typename TypeParam, typename T = TypeParam>
auto ElementsAreTypedArray(std::vector<T> x) {
  if constexpr (std::is_floating_point_v<TypeParam>) {
    return ElementsAreArray(ArrayFloatNear(std::move(x)));
  } else {
    return ElementsAreArray(std::move(x));
  }
}

// Casts input vector to specified type, converting to string for std::string
// type.
template <typename T>
std::vector<T> CastVector(const std::vector<int>& input_data) {
  std::vector<T> casted_input(input_data.size());

  if constexpr (std::is_same_v<T, std::string>) {
    std::transform(input_data.begin(), input_data.end(), casted_input.begin(),
                   [](int x) { return std::to_string(x); });
  } else if constexpr (std::is_same_v<T, int>) {
    return input_data;
  } else {
    std::transform(input_data.begin(), input_data.end(), casted_input.begin(),
                   [](int x) { return static_cast<T>(x); });
  }
  return casted_input;
}

#if GTEST_HAS_DEATH_TEST
TYPED_TEST(StridedSliceOpTest, UnsupportedInputSize) {
  EXPECT_DEATH(StridedSliceOpModel<TypeParam>({2, 2, 2, 2, 2, 2}, {5}, {5}, {5},
                                              {TypeParam{}}, {}, {}, {}, 0, 0,
                                              0, 0, 0, false),
               "StridedSlice op only supports 1D-5D input arrays.");
}
#endif

TYPED_TEST(StridedSliceOpTest, In1DEmpty) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    StridedSliceOpModel<TypeParam> m({0}, {1}, {1}, {1},
                                     std::vector<TypeParam>{}, {1}, {3}, {1}, 0,
                                     0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0}));
  }
}

TYPED_TEST(StridedSliceOpTest, Offset) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    StridedSliceOpModel<TypeParam> m({10}, {1}, {1}, {1}, input_data, {1}, {3},
                                     {1}, 0, 0, 0, 0, 0, constant_tensors,
                                     /*offset=*/true);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3})));
    if (constant_tensors) {
      EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
    } else {
      EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
    }
  }
}

TYPED_TEST(StridedSliceOpTest, OffsetArray) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    StridedSliceOpModel<TypeParam> m({3, 4}, {2}, {2}, {2}, input_data, {0, 1},
                                     {2, 2}, {1, 1}, 0, 0, 0, 0, 0,
                                     constant_tensors, /*offset=*/true);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 5, 6})));
    if (constant_tensors) {
      EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
    } else {
      EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
    }
  }
}

TYPED_TEST(StridedSliceOpTest, OffsetConstant) {
  const std::vector<TypeParam> input_data =
      CastVector<TypeParam>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  StridedSliceOpModel<TypeParam> m({3, 4}, {2}, {2}, {2}, input_data, {0, 1},
                                   {2, 2}, {1, 1}, 0, 0, 0, 0, 0,
                                   /*constant_tensors*/ false,
                                   /*offset=*/true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                 CastVector<TypeParam>({1, 2, 5, 6})));
  EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
}

TYPED_TEST(StridedSliceOpTest, OffsetConstantStride) {
  const int height = 5;
  const int width = 6;
  std::vector<int> input_data(height * width);
  std::iota(input_data.begin(), input_data.end(), 0);

  auto casted_input_data = CastVector<TypeParam>(input_data);

  StridedSliceOpModel<TypeParam> m({height, width}, {2}, {2}, {2},
                                   casted_input_data, {0, 1}, {4, 3}, {2, 2}, 0,
                                   0, 0, 0, 0,
                                   /*constant_tensors*/ false,
                                   /*offset=*/true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                 CastVector<TypeParam>({1, 3, 13, 15})));
  EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
}

TYPED_TEST(StridedSliceOpTest, OffsetConstantNegativeStride) {
  const int height = 5;
  const int width = 6;
  std::vector<int> input_data(height * width);
  std::iota(input_data.begin(), input_data.end(), 0);

  auto casted_input_data = CastVector<TypeParam>(input_data);

  StridedSliceOpModel<TypeParam> m({height, width}, {2}, {2}, {2},
                                   casted_input_data, {4, 4}, {-4, -3},
                                   {-2, -2}, 0, 0, 0, 0, 0,
                                   /*constant_tensors*/ false,
                                   /*offset=*/true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                 CastVector<TypeParam>({28, 26, 16, 14})));
  EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
}

TYPED_TEST(StridedSliceOpTest, In1D) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {1}, {3},
                                     {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({2, 3})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1DConst) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {1}, {3},
                                     {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({2, 3})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_Int32End) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    std::vector<TypeParam> values(32768);
    std::iota(values.begin(), values.end(), TypeParam(0));

    StridedSliceOpModel<TypeParam> m({32768}, {1}, {1}, {1}, values, {0},
                                     {32768}, {1}, 0, 0, 0, 0, 0,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({32768}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(values));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_EmptyOutput) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {10}, {3},
                                     {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0}));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_NegativeBegin) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {-3}, {3},
                                     {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({2, 3})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_OutOfRangeBegin) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {-5}, {3},
                                     {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_NegativeEnd) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {1}, {-2},
                                     {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({2})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_OutOfRangeEnd) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {-3}, {5},
                                     {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({2, 3, 4})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_BeginMask) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {1}, {3},
                                     {1}, 1, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_NegativeBeginNegativeStride) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {-2}, {-3},
                                     {-1}, 0, 0, 0, 0, 0, constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({3})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_OutOfRangeBeginNegativeStride) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {5}, {2},
                                     {-1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({4})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_NegativeEndNegativeStride) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {2}, {-4},
                                     {-1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({3, 2})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_OutOfRangeEndNegativeStride) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {-3}, {-5},
                                     {-1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({2, 1})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_EndMask) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {1}, {3},
                                     {1}, 0, 1, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({2, 3, 4})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_NegStride) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({1, 2, 3});
    StridedSliceOpModel<TypeParam> m({3}, {1}, {1}, {1}, input_data, {-1}, {-4},
                                     {-1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({3, 2, 1})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_EvenLenStride2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({1, 2});
    StridedSliceOpModel<TypeParam> m({2}, {1}, {1}, {1}, input_data, {0}, {2},
                                     {2}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({1})));
  }
}

TYPED_TEST(StridedSliceOpTest, In1D_OddLenStride2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({1, 2, 3});
    StridedSliceOpModel<TypeParam> m({3}, {1}, {1}, {1}, input_data, {0}, {3},
                                     {2}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 3})));
  }
}

TYPED_TEST(StridedSliceOpTest, In2D_Identity) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {0, 0},
                                     {2, 3}, {1, 1}, 0, 0, 0, 0, 0,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3, 4, 5, 6})));
  }
}

TYPED_TEST(StridedSliceOpTest, In2D) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {1, 0},
                                     {2, 2}, {1, 1}, 0, 0, 0, 0, 0,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({4, 5})));
  }
}

TYPED_TEST(StridedSliceOpTest, In2D_Stride2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {0, 0},
                                     {2, 3}, {2, 2}, 0, 0, 0, 0, 0,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 3})));
  }
}

TYPED_TEST(StridedSliceOpTest, In2D_NegStride) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {1, -1},
                                     {2, -4}, {2, -1}, 0, 0, 0, 0, 0,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({6, 5, 4})));
  }
}

TYPED_TEST(StridedSliceOpTest, In2D_BeginMask) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {1, 0},
                                     {2, 2}, {1, 1}, 1, 0, 0, 0, 0,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 4, 5})));
  }
}

TYPED_TEST(StridedSliceOpTest, In2D_EndMask) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {1, 0},
                                     {2, 2}, {1, 1}, 0, 2, 0, 0, 0,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({4, 5, 6})));
  }
}
TYPED_TEST(StridedSliceOpTest, In2D_NegStrideBeginMask) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {1, -2},
                                     {2, -4}, {1, -1}, 2, 0, 0, 0, 0,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({6, 5, 4})));
  }
}
TYPED_TEST(StridedSliceOpTest, In2D_NegStrideEndMask) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {1, -2},
                                     {2, -3}, {1, -1}, 0, 2, 0, 0, 0,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({5, 4})));
  }
}

TYPED_TEST(StridedSliceOpTest, In3D_Identity) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {2, 3, 2}, {1, 1, 1}, 0, 0, 0,
                                     0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_NegStride) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {-1, -1, -1}, {-3, -4, -3}, {-1, -1, -1},
                                     0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_Strided2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {2, 3, 2}, {2, 2, 2}, 0, 0, 0,
                                     0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 1}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 5})));
  }
}
TYPED_TEST(StridedSliceOpTest, In1D_ShrinkAxisMask1) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {1}, {2},
                                     {1}, 0, 0, 0, 0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_TRUE(m.GetOutputShape().empty());
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({2})));
  }
}
TYPED_TEST(StridedSliceOpTest, In1D_ShrinkAxisMask1_NegativeSlice) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    // This is equivalent to tf.range(4)[-1].
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({0, 1, 2, 3});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {-1}, {0},
                                     {1}, 0, 0, 0, 0, 1, constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_TRUE(m.GetOutputShape().empty());
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({3})));
  }
}
TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxis3_NegativeSlice) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    // This is equivalent to tf.range(4)[:, tf.newaxis][-2, -1].
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({0, 1, 2, 3});
    StridedSliceOpModel<TypeParam> m({4, 1}, {2}, {2}, {2}, input_data,
                                     {-2, -1}, {-1, 0}, {1, 1}, 0, 0, 0, 0, 3,
                                     constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_TRUE(m.GetOutputShape().empty());
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({2})));
  }
}
TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxis2_BeginEndAxis1_NegativeSlice) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    // This is equivalent to tf.range(4)[:, tf.newaxis][:, -1].
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({0, 1, 2, 3});
    StridedSliceOpModel<TypeParam> m({4, 1}, {2}, {2}, {2}, input_data, {0, -1},
                                     {0, 0}, {1, 1}, 1, 1, 0, 0, 2,
                                     constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({0, 1, 2, 3})));
  }
}
TYPED_TEST(StridedSliceOpTest, In1D_BeginMaskShrinkAxisMask1) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {1}, {1},
                                     {1}, 1, 0, 0, 0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_TRUE(m.GetOutputShape().empty());
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({1})));
  }
}
TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxisMask1) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {0, 0},
                                     {1, 3}, {1, 1}, 0, 0, 0, 0, 1,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3})));
  }
}
TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxisMask2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {0, 0},
                                     {2, 1}, {1, 1}, 0, 0, 0, 0, 2,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 4})));
  }
}
TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxisMask3) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {0, 0},
                                     {1, 1}, {1, 1}, 0, 0, 0, 0, 3,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_TRUE(m.GetOutputShape().empty());
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({1})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis1) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 3, 2}, {1, 1, 1}, 0, 0, 0,
                                     0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3, 4, 5, 6})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {2, 1, 2}, {1, 1, 1}, 0, 0, 0,
                                     0, 2, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 7, 8})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis3) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 1, 2}, {1, 1, 1}, 0, 0, 0,
                                     0, 3, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis4) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {2, 3, 1}, {1, 1, 1}, 0, 0, 0,
                                     0, 4, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 3, 5, 7, 9, 11})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis5) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 3, 1}, {1, 1, 1}, 0, 0, 0,
                                     0, 5, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 3, 5})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis6) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {2, 1, 1}, {1, 1, 1}, 0, 0, 0,
                                     0, 6, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 7})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis7) {
  for (bool constant_tensors : {true, false}) {
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 0, 0, 0,
                                     0, 7, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_TRUE(m.GetOutputShape().empty());
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({1})));
  }

  // This tests catches a very subtle bug that was fixed by cl/188403234.
}
TYPED_TEST(StridedSliceOpTest, RunTwice) {
  const std::vector<TypeParam> input_data =
      CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {1, 0},
                                   {2, 2}, {1, 1}, 1, 0, 0, 0, 0, false);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                 CastVector<TypeParam>({1, 2, 4, 5})));

  auto setup_inputs = [&m, &input_data]() {
    m.template SetInput<TypeParam>(input_data,
                                   std::is_same<std::string, TypeParam>());
    m.SetBegin({1, 0});
    m.SetEnd({2, 2});
    m.SetStrides({1, 1});
  };

  setup_inputs();
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // Prior to cl/188403234 this was {4, 5}.
  EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                 CastVector<TypeParam>({1, 2, 4, 5})));
}
TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis1Uint8) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 3, 2}, {1, 1, 1}, 0, 0, 0,
                                     0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3, 4, 5, 6})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis1int8) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 3, 2}, {1, 1, 1}, 0, 0, 0,
                                     0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3, 4, 5, 6})));
  }
}
TYPED_TEST(StridedSliceOpTest, In5D_Identity) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    StridedSliceOpModel<TypeParam> m(
        {2, 2, 2, 1, 2}, {5}, {5}, {5}, input_data, {0, 0, 0, 0, 0},
        {2, 1, 2, 1, 2}, {1, 1, 1, 1, 1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 2, 1, 2}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(
                    CastVector<TypeParam>({1, 2, 3, 4, 9, 10, 11, 12})));
  }
}
TYPED_TEST(StridedSliceOpTest, In5D_IdentityShrinkAxis1) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    StridedSliceOpModel<TypeParam> m(
        {2, 2, 2, 1, 2}, {5}, {5}, {5}, input_data, {0, 0, 0, 0, 0},
        {2, 1, 2, 1, 2}, {1, 1, 1, 1, 1}, 0, 0, 0, 0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3, 4})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_SmallBegin) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {1}, {1}, {1}, input_data, {0},
                                     {1}, {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3, 4, 5, 6})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_SmallBeginWithhrinkAxis1) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {1}, {1}, {1}, input_data, {0},
                                     {1}, {1}, 0, 0, 0, 0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3, 4, 5, 6})));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_BackwardSmallBeginEndMask) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({1, 2});
    StridedSliceOpModel<TypeParam> m({1, 1, 2}, {1}, {1}, {1}, input_data, {1},
                                     {0}, {1}, 0, 1, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0, 1, 2}));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_BackwardSmallBegin) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({1, 2});
    StridedSliceOpModel<TypeParam> m({1, 1, 2}, {1}, {1}, {1}, input_data, {1},
                                     {0}, {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0, 1, 2}));
  }
}
TYPED_TEST(StridedSliceOpTest, In3D_Backward) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({1, 2});
    StridedSliceOpModel<TypeParam> m({1, 1, 2}, {3}, {3}, {3}, input_data,
                                     {1, 0, 0}, {0, -1, -1}, {1, 1, 1}, 6, 7, 0,
                                     0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0, 1, 2}));
  }
}

TEST(StridedSliceOpTest, In1D_String_NegativeBegin) {
  std::vector<std::string> input_data = CastVector<std::string>(
      {1, 2, 3, 4});  // input_data = {"a", "b", "c", "d"}
  StridedSliceOpModel<std::string> m({4}, {1}, {1}, {1}, input_data, {-3}, {3},
                                     {1}, 0, 0, 0, 0, 0, false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<std::string> output_data =
      CastVector<std::string>({2, 3});  // output_data = {"b", "c"}
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetStringOutput(), ElementsAreArray(output_data));
}

TEST(StridedSliceOpTest, In3D_String_BackwardSmallBegin) {
  std::vector<std::string> input_data =
      CastVector<std::string>({1, 2});  // input_data = {"a", "b"}

  StridedSliceOpModel<std::string> m({1, 1, 2}, {1}, {1}, {1}, input_data, {1},
                                     {0}, {1}, 0, 1, 0, 0, 0, false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0, 1, 2}));
}

TEST(StridedSliceOpTest, In3D_String_SmallBeginWithhrinkAxis1) {
  std::vector<std::string> input_data =
      CastVector<std::string>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  StridedSliceOpModel<std::string> m({2, 3, 2}, {1}, {1}, {1}, input_data, {0},
                                     {1}, {1}, 0, 0, 0, 0, 1, false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
  EXPECT_THAT(m.GetStringOutput(),
              ElementsAreArray({"1", "2", "3", "4", "5", "6"}));
}

TEST(StridedSliceOpTest, In5D_String_IdentityShrinkAxis1) {
  std::vector<std::string> input_data = CastVector<std::string>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  StridedSliceOpModel<std::string> m({2, 2, 2, 1, 2}, {5}, {5}, {5}, input_data,
                                     {0, 0, 0, 0, 0}, {2, 1, 2, 1, 2},
                                     {1, 1, 1, 1, 1}, 0, 0, 0, 0, 1, false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 1, 2}));
  EXPECT_THAT(m.GetStringOutput(), ElementsAreArray({"1", "2", "3", "4"}));
}
TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxis_Endmask_AtSameAxis) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({0, 1, 2, 3});
    StridedSliceOpModel<TypeParam> m({2, 2}, {2}, {2}, {2}, input_data, {0, -1},
                                     {0, 0}, {1, -1}, 1, 1, 0, 0, 1,
                                     constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({1})));
  }
}
TYPED_TEST(StridedSliceOpTest, EllipsisMask1_NewAxisMask2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 2, 1}, {1, 1, 1}, 0, 0, 1,
                                     2, 0, constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 1, 1}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 3, 5, 7, 9, 11})));
  }
}
TYPED_TEST(StridedSliceOpTest, EllipsisMask2_NewAxisMask1) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 2, 1}, {1, 1, 1}, 0, 0, 2,
                                     1, 0, constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 1}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 3, 5, 7, 9, 11})));
  }
}
TYPED_TEST(StridedSliceOpTest, EllipsisMask2_NewAxisMask5) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 2, 1}, {1, 1, 1}, 0, 0, 2,
                                     5, 0, constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 2, 1}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
  }
}
TYPED_TEST(StridedSliceOpTest, EllipsisMask2_NewAxisMask2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 2, 1}, {1, 1, 1}, 0, 0, 2,
                                     2, 0, constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 3, 5})));
  }
}
TYPED_TEST(StridedSliceOpTest, EllipsisMask4_NewAxisMask2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 2, 1}, {1, 1, 1}, 0, 0, 4,
                                     2, 0, constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 3, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 3, 4, 5, 6})));
  }
}
TYPED_TEST(StridedSliceOpTest, EllipsisMask2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 2, 1}, {1, 1, 1}, 0, 0, 2,
                                     0, 0, constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 3, 5})));
  }
}
TYPED_TEST(StridedSliceOpTest, NewAxisMask2) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 3, 1}, {1, 1, 1}, 0, 0, 0,
                                     2, 0, constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2})));
  }
}
TYPED_TEST(StridedSliceOpTest, NewAxisMask1) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, input_data,
                                     {0, 0, 0}, {1, 3, 1}, {1, 1, 1}, 0, 0, 0,
                                     1, 0, constant_tensors);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({1, 2, 7, 8})));
  }
}
TYPED_TEST(StridedSliceOpTest, NoInfiniteLoop) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    StridedSliceOpModel<TypeParam> m(
        {1, 1}, {6}, {6}, {6}, {}, {1, 1, 1, 1, 1, 1}, {3, 3, 3, 3, 3, 3},
        {1, 1, 1, 1, 1, 1}, 1, 2, 1, 6, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
  }
}
TYPED_TEST(StridedSliceOpTest, MinusThreeMinusFourMinusOne) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {-3}, {-4},
                                     {-1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({2})));
  }
}
TYPED_TEST(StridedSliceOpTest, MinusFourMinusThreeOne) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {-4}, {-3},
                                     {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({1})));
  }
}
TYPED_TEST(StridedSliceOpTest, OneOneOne) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({2});
    StridedSliceOpModel<TypeParam> m({1}, {1}, {1}, {1}, input_data, {1}, {1},
                                     {1}, 0, 0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0}));
  }
}
TYPED_TEST(StridedSliceOpTest, OneOneOneShrinkAxis) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({1, 2, 3});
    StridedSliceOpModel<TypeParam> m({3}, {1}, {1}, {1}, input_data, {1}, {1},
                                     {1}, 0, 0, 0, 0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), IsEmpty());
    EXPECT_THAT(m.GetOutput(),
                ElementsAreTypedArray<TypeParam>(CastVector<TypeParam>({2})));
  }
}
TYPED_TEST(StridedSliceOpTest, OneOneOneShrinkAxisOOB) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({2});
    StridedSliceOpModel<TypeParam> m({1}, {1}, {1}, {1}, input_data, {1}, {1},
                                     {1}, 0, 0, 0, 0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  }
}
TYPED_TEST(StridedSliceOpTest, OutOfBounds) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    StridedSliceOpModel<TypeParam> m({1}, {1}, {1}, {1}, {}, {1}, {2}, {1}, 0,
                                     0, 0, 0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  }
}
TYPED_TEST(StridedSliceOpTest, StrideOutOfBounds) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    StridedSliceOpModel<TypeParam> m({1}, {1}, {1}, {1}, {}, {1}, {4}, {7}, 0,
                                     0, 0, 0, 1, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  }
}
TYPED_TEST(StridedSliceOpTest, NegEndMask) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
    StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {0, -1},
                                     {2, -3}, {1, -1}, 0, 0b10, 0, 0, 0,
                                     constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({3, 2, 1, 6, 5, 4})));
  }
}
TYPED_TEST(StridedSliceOpTest, NoopOffset) {
  const std::vector<TypeParam> input_data =
      CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, input_data, {0, -1},
                                   {2, -3}, {1, -1}, 0, 0b10, 0, 0, 0);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                 CastVector<TypeParam>({3, 2, 1, 6, 5, 4})));
}
}  // namespace
}  // namespace tflite
