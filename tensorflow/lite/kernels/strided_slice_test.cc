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
#include "tensorflow/lite/kernels/internal/reference/strided_slice.h"

#include <stdint.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <ios>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "Eigen/Core"  // from @eigen_archive  // IWYU pragma: keep
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/types/half.h"

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

using DataTypes = ::testing::Types<float, half, Eigen::bfloat16, uint8_t,
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

TYPED_TEST(StridedSliceOpTest, In6D) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({0, 1, 2, 3, 4, 5});
    StridedSliceOpModel<TypeParam> m({2, 1, 1, 1, 1, 3}, {6}, {6}, {6},
                                     input_data, {1, 0, 0, 0, 0, 1},
                                     {2, 1, 1, 1, 1, 3}, {1, 1, 1, 1, 1, 1}, 0,
                                     0, 0, 0, 0, constant_tensors);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreTypedArray<TypeParam>(
                                   CastVector<TypeParam>({4, 5})));
  }
}

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
    if (m.GetNumberOfAppliedDelegates() == 0) {
      if (constant_tensors) {
        EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
      } else {
        EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
      }
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
    if (m.GetNumberOfAppliedDelegates() == 0) {
      if (constant_tensors) {
        EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
      } else {
        EXPECT_THAT(m.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
      }
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
    for (int i = 0; i < 32768; ++i) {
      values[i] = static_cast<TypeParam>(i);
    }

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

TYPED_TEST(StridedSliceOpTest, StrideOverflowAndEdgeCases) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  {
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({1});
    StridedSliceOpModel<TypeParam> m({1}, {1}, {1}, {1}, input_data, {0},
                                     {2147483647}, {126322568}, 0, 0, 0, 0, 0,
                                     /*constant_tensors=*/false,
                                     /*offset=*/true);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({17}));
  }
  {
    const std::vector<TypeParam> input_data = CastVector<TypeParam>({1});
    StridedSliceOpModel<TypeParam> m({1}, {1}, {1}, {1}, input_data, {0},
                                     {-2147483648}, {-1000000000}, 0, 0, 0, 0,
                                     0, /*constant_tensors=*/false,
                                     /*offset=*/true);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  }
}

TYPED_TEST(StridedSliceOpTest, OutputDimOverflowCheck) {
  // Non-constant INT32_MIN stride is rejected during Invoke.
  {
    const std::vector<TypeParam> input_data =
        CastVector<TypeParam>({1, 2, 3, 4});
    StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, input_data, {1}, {3},
                                     {std::numeric_limits<int32_t>::min()}, 0,
                                     0, 0, 0, 0,
                                     /*constant_tensors=*/false);
    EXPECT_NE(m.Invoke(), kTfLiteOk);
  }

#if GTEST_HAS_DEATH_TEST
  // Constant INT32_MIN stride is rejected during Prepare.
  EXPECT_DEATH(
      {
        const std::vector<TypeParam> input_data =
            CastVector<TypeParam>({1, 2, 3, 4});
        StridedSliceOpModel<TypeParam> m(
            {4}, {1}, {1}, {1}, input_data, {1}, {3},
            {std::numeric_limits<int32_t>::min()}, 0, 0, 0, 0, 0,
            /*constant_tensors=*/true);
      },
      "stride value INT32_MIN is not supported");

  // offset mode rejects an end value whose output dim overflows int32.
  EXPECT_DEATH(
      {
        const std::vector<TypeParam> input_data =
            CastVector<TypeParam>({1, 2, 3, 4});
        StridedSliceOpModel<TypeParam> m(
            {4}, {1}, {1}, {1}, input_data, {0},
            {std::numeric_limits<int32_t>::min()}, {-1}, 0, 0, 0, 0, 0,
            /*constant_tensors=*/false, /*offset=*/true);
      },
      "StridedSlice: integer overflow computing output dim at axis 0");
#endif
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

// Verbatim copy of the legacy element-by-element std::function recursive
// StridedSlice implementation used as a reference oracle to verify 100%
// bit-for-bit identical output across all tensor shapes, strides, and masks.
template <typename T>
void LegacyReferenceStridedSlice(
    const reference_ops::DynamicStridedSliceParams& op_params,
    const RuntimeShape& input_shape, const RuntimeShape& output_shape,
    SequentialTensorWriter<T>* writer) {
  ruy::profiler::ScopeLabel label("StridedSlice");
  const int dims = input_shape.DimensionsCount();
  std::vector<int> starts(dims);
  std::vector<int> stops(dims);
  std::vector<int64_t> input_strides(dims);
  if (dims == 0) {
    writer->Write(0);
    return;
  }
  input_strides[dims - 1] = 1;
  for (int i = dims - 2; i >= 0; --i) {
    input_strides[i] = input_strides[i + 1] * input_shape.Dims(i + 1);
  }
  for (int axis = 0; axis < dims; ++axis) {
    starts[axis] = reference_ops::StartForAxis(op_params, input_shape, axis);
    stops[axis] =
        reference_ops::EndForAxis(op_params, input_shape, axis, starts[axis]);
  }
  auto loop_condition = [](int64_t index, int64_t stop, int stride) {
    return stride > 0 ? index < stop : index > stop;
  };
  std::function<void(int, int64_t)> write_slice = [&](int axis,
                                                      int64_t input_index) {
    if (axis == dims) {
      writer->Write(input_index);
      return;
    }
    for (int64_t offset = starts[axis];
         loop_condition(offset, stops[axis], op_params.strides[axis]);
         offset += op_params.strides[axis]) {
      write_slice(axis + 1, input_index + offset * input_strides[axis]);
    }
  };
  write_slice(/*axis=*/0, /*input_index=*/0);
}

// Dual-sentinel bit-for-bit and write-cardinality verification helper.
// 1. Runs both LegacyReferenceStridedSlice and reference_ops::StridedSlice
//    twice using bitwise-complementary sentinel byte patterns (0xA5 and 0x5A).
// 2. Independently computes the closed-form analytical element count K from
//    StartForAxis / EndForAxis and verifies that:
//    - Every byte in [0, K * sizeof(T)) is identical between the 0xA5 run and
//      the 0x5A run (proving every byte in [0, K * sizeof(T)) was written),
//    - Every byte in [K * sizeof(T), max_out * sizeof(T)) equals 0xA5 in the
//      first run and 0x5A in the second run (proving zero over-writes), and
//    - The full output buffer matches LegacyReferenceStridedSlice bit-for-bit.
template <typename T>
void VerifyBitForBitIdentical(
    const reference_ops::DynamicStridedSliceParams& params,
    const RuntimeShape& shape, const std::vector<T>& input) {
  const int dims = shape.DimensionsCount();
  int64_t expected_elements = 1;
  for (int axis = 0; axis < dims; ++axis) {
    const int64_t start = reference_ops::StartForAxis(params, shape, axis);
    const int64_t stop = reference_ops::EndForAxis(params, shape, axis, start);
    const int64_t stride = params.strides[axis];
    if ((stride > 0 && start >= stop) || (stride < 0 && start <= stop)) {
      expected_elements = 0;
      break;
    }
    const int64_t span = stride > 0 ? (stop - start) : (start - stop);
    const int64_t step = stride > 0 ? stride : -stride;
    expected_elements *= (span + step - 1) / step;
  }

  const int64_t max_out =
      std::max<int64_t>(4, static_cast<int64_t>(input.size()) * 2 + 4);
  const size_t total_bytes = static_cast<size_t>(max_out) * sizeof(T);
  const size_t expected_bytes =
      static_cast<size_t>(expected_elements) * sizeof(T);

  std::vector<T> opt_out_a5(max_out);
  std::vector<T> opt_out_5a(max_out);

  for (int sentinel : {0xA5, 0x5A}) {
    std::vector<T> legacy_out(max_out);
    std::vector<T>& opt_out = (sentinel == 0xA5) ? opt_out_a5 : opt_out_5a;
    std::memset(legacy_out.data(), sentinel, total_bytes);
    std::memset(opt_out.data(), sentinel, total_bytes);

    SequentialTensorWriter<T> legacy_writer(input.data(), legacy_out.data());
    SequentialTensorWriter<T> opt_writer(input.data(), opt_out.data());

    LegacyReferenceStridedSlice<T>(params, shape, shape, &legacy_writer);
    reference_ops::StridedSlice<T>(params, shape, shape, &opt_writer);

    EXPECT_EQ(std::memcmp(legacy_out.data(), opt_out.data(), total_bytes), 0)
        << "Bit-for-bit mismatch for shape with dims=" << dims
        << " and sentinel=0x" << std::hex << sentinel;
  }

  if (expected_bytes > 0) {
    EXPECT_EQ(std::memcmp(opt_out_a5.data(), opt_out_5a.data(), expected_bytes),
              0)
        << "Under-write detected inside [0, " << expected_bytes << ")";
  }
  const auto* bytes_a5 = reinterpret_cast<const uint8_t*>(opt_out_a5.data());
  const auto* bytes_5a = reinterpret_cast<const uint8_t*>(opt_out_5a.data());
  for (size_t b = expected_bytes; b < total_bytes; ++b) {
    ASSERT_EQ(bytes_a5[b], 0xA5u) << "Over-write at byte " << b;
    ASSERT_EQ(bytes_5a[b], 0x5Au) << "Over-write at byte " << b;
  }
}

template <typename T>
void RunExhaustiveBitForBitSuite() {
  // 0D scalar tensor.
  {
    RuntimeShape shape0d;
    std::vector<T> in0d = CastVector<T>({42});
    reference_ops::DynamicStridedSliceParams params0d;
    VerifyBitForBitIdentical<T>(params0d, shape0d, in0d);
  }

  // 1D, 2D, 3D, 4D, and 5D shapes exercising:
  // - Zero-extent dimensions (Dims(axis) == 0) with positive & negative strides
  // - Full trailing contiguous dimensions (1, 2, 3, and 4 coalesced axes)
  // - Trailing singleton dimensions (Dims(axis) == 1) where
  //   input_strides[last_loop_axis] == 1 on a non-innermost axis
  // - Partial unit-stride inner dimension (starts > 0, stops < Dims)
  // - Strided inner dimension (stride = 2, 3, -1, -2, -3)
  // - Strided or negative-strided outer/middle dimension with full contiguous
  //   inner dimensions (where input_strides[last_loop_axis] > 1)
  // - Empty slices on outer, middle, and inner axes (positive & negative
  //   stride)
  // - begin_mask, end_mask, shrink_axis_mask, and offset = true.
  struct SliceCase {
    std::vector<int32_t> dims;
    std::vector<int32_t> starts;
    std::vector<int32_t> stops;
    std::vector<int32_t> strides;
    uint32_t begin_mask = 0;
    uint32_t end_mask = 0;
    uint32_t shrink_axis_mask = 0;
    bool offset = false;
  };

  const std::vector<SliceCase> cases = {
      // Zero-extent dimension cases (Dims(axis) == 0).
      {{0}, {0}, {0}, {1}},
      {{0}, {0}, {0}, {-1}},
      {{0, 8, 16}, {0, 0, 0}, {0, 8, 16}, {1, 1, 1}},
      {{3, 0, 16}, {0, 0, 0}, {3, 0, 16}, {1, 1, 1}},
      {{3, 8, 0}, {0, 0, 0}, {3, 8, 0}, {1, -1, -1}},
      {{3, 0, 16}, {0, 0, 0}, {3, 0, 16}, {1, 1, 1}, 0, 0, 0b010},
      // 1D cases: full, partial, strided, reversed, empty.
      {{16}, {0}, {16}, {1}},
      {{16}, {3}, {13}, {1}},
      {{16}, {1}, {15}, {2}},
      {{16}, {15}, {-1}, {-1}, 0, 1},
      {{16}, {14}, {2}, {-3}},
      {{16}, {10}, {5}, {1}},
      {{16}, {5}, {10}, {-1}},
      // 2D cases: coalesced inner, partial inner, strided outer + full inner,
      // reversed outer + full inner, reversed inner.
      {{6, 12}, {0, 0}, {6, 12}, {1, 1}},
      {{6, 12}, {1, 0}, {5, 12}, {1, 1}},
      {{6, 12}, {1, 2}, {5, 10}, {1, 1}},
      {{6, 12}, {0, 0}, {6, 12}, {2, 1}},
      {{6, 12}, {5, 0}, {-1, 12}, {-1, 1}, 0, 0b01},
      {{6, 12}, {5, 0}, {0, 12}, {-2, 1}},
      {{6, 12}, {1, 11}, {5, -1}, {2, -1}, 0, 0b10},
      {{6, 12}, {1, 10}, {5, 1}, {1, -2}},
      {{6, 12}, {4, 0}, {2, 12}, {1, 1}},
      {{6, 12}, {0, 8}, {6, 3}, {1, 1}},
      // Trailing singleton dimension cases (1 < inner_contig_axis < dims with
      // input_strides[last_loop_axis] == 1 and last_loop_stride_is_1 == false).
      {{3, 6, 1}, {0, 0, 0}, {3, 6, 1}, {1, 2, 1}},
      {{3, 6, 1}, {0, 5, 0}, {3, -1, 1}, {1, -2, 1}, 0, 0b010},
      {{2, 6, 1, 1}, {0, 5, 0, 0}, {2, -1, 1, 1}, {1, -1, 1, 1}, 0, 0b0010},
      // 3D speech-like feature tensor cases:
      // [B, T, F] with full T & F coalesced, sliced T with full F coalesced,
      // strided/reversed T with full F coalesced, partial F, strided F,
      // and empty middle/inner axes.
      {{3, 8, 16}, {0, 0, 0}, {3, 8, 16}, {1, 1, 1}},
      {{3, 8, 16}, {1, 0, 0}, {3, 8, 16}, {1, 1, 1}},
      {{3, 8, 16}, {0, 2, 0}, {3, 7, 16}, {1, 1, 1}},
      {{3, 8, 16}, {0, 1, 0}, {3, 8, 16}, {1, 2, 1}},
      {{3, 8, 16}, {0, 7, 0}, {3, -1, 16}, {1, -1, 1}, 0, 0b010},
      {{3, 8, 16}, {2, 7, 0}, {-1, 0, 16}, {-1, -2, 1}, 0, 0b001},
      {{3, 8, 16}, {0, 1, 3}, {3, 6, 14}, {1, 2, 1}},
      {{3, 8, 16}, {0, 1, 15}, {3, 6, 0}, {1, 2, -3}},
      {{3, 8, 16}, {0, 5, 0}, {3, 2, 16}, {1, 1, 1}},
      {{3, 8, 16}, {0, 1, 10}, {3, 6, 4}, {1, 1, 1}},
      {{3, 8, 16}, {1, 2, 0}, {2, 5, 16}, {1, 1, 1}, 0, 0, 0b001},
      {{3, 8, 16}, {0, 3, 0}, {3, 4, 16}, {1, 1, 1}, 0, 0, 0b010},
      {{3, 8, 16}, {0, 2, 5}, {3, 5, 6}, {1, 1, 1}, 0, 0, 0b100},
      {{3, 8, 16}, {1, 2, 3}, {2, 4, 10}, {1, 1, 1}, 0, 0, 0, true},
      // 4D and 5D cases: multi-axis coalescing across 3 and 4 trailing axes,
      // mixed positive/negative strides, and shrink_axis_mask combinations.
      {{2, 3, 4, 5}, {0, 1, 0, 0}, {2, 3, 4, 5}, {1, 1, 1, 1}},
      {{2, 3, 4, 5}, {0, 2, 0, 0}, {2, -1, 4, 5}, {1, -1, 1, 1}, 0, 0b0010},
      {{2, 3, 4, 5}, {1, 0, 1, 0}, {-1, 3, 3, 5}, {-1, 2, 1, 1}, 0, 0b0001},
      {{2, 2, 3, 2, 4}, {0, 0, 1, 0, 0}, {2, 2, 3, 2, 4}, {1, 1, 1, 1, 1}},
      {{2, 2, 3, 2, 4},
       {1, 1, 2, 0, 0},
       {-1, -1, 0, 2, 4},
       {-1, -1, -1, 1, 1},
       0,
       0b00011},
  };

  for (const auto& tc : cases) {
    RuntimeShape shape(static_cast<int>(tc.dims.size()), tc.dims.data());
    const int total_elements = shape.FlatSize();
    std::vector<int> raw_vals(total_elements);
    for (int i = 0; i < total_elements; ++i) {
      if (i == 0) {
        raw_vals[i] = -91;  // 0xA5 in two's complement int8_t
      } else if (i == 1) {
        raw_vals[i] = 90;  // 0x5A in int8_t
      } else {
        raw_vals[i] = (i * 37 + 13) % 127 - 63;
      }
    }
    const std::vector<T> input = CastVector<T>(raw_vals);
    reference_ops::DynamicStridedSliceParams params;
    params.start_indices = tc.starts;
    params.stop_indices = tc.stops;
    params.strides = tc.strides;
    params.begin_mask = tc.begin_mask;
    params.end_mask = tc.end_mask;
    params.shrink_axis_mask = tc.shrink_axis_mask;
    params.offset = tc.offset;

    VerifyBitForBitIdentical<T>(params, shape, input);
  }
}

TYPED_TEST(StridedSliceOpTest, ExhaustiveBitForBitEquivalenceWithLegacyOracle) {
  RunExhaustiveBitForBitSuite<TypeParam>();
}

TEST(StridedSliceBitExactInt64Test,
     ExhaustiveBitForBitEquivalenceWithLegacyOracleInt64) {
  RunExhaustiveBitForBitSuite<int64_t>();
}

// Verify strict bit-for-bit equality on float tensors containing special IEEE
// 754 bit patterns (-0.0f, +0.0f, quiet NaNs with distinct payloads, signaling
// NaNs, subnormals, +-Inf, and sentinel bit patterns 0xa5a5a5a5 / 0x5a5a5a5a)
// where operator== would either return false (NaN != NaN) or conflate distinct
// bit representations (-0.0f == +0.0f).
TEST(StridedSliceBitExactFloatTest, PreservesSpecialFloatBitPatternsExactly) {
  const std::vector<uint32_t> special_bits = {
      0x00000000u,  // +0.0f
      0x80000000u,  // -0.0f
      0x7f800000u,  // +Inf
      0xff800000u,  // -Inf
      0x7fc00001u,  // Quiet NaN payload 1
      0x7fcabcdeu,  // Quiet NaN payload 0xabcde (8 hex digits)
      0xffc01234u,  // Negative Quiet NaN payload 0x1234
      0x7f800001u,  // Signaling NaN payload 1
      0x00000001u,  // Smallest positive subnormal
      0x80000001u,  // Smallest negative subnormal
      0x007fffffu,  // Largest positive subnormal
      0x3f800000u,  // 1.0f
      0xbf800000u,  // -1.0f
      0xa5a5a5a5u,  // Sentinel bit pattern A5
      0x5a5a5a5au,  // Sentinel bit pattern 5A
  };

  RuntimeShape shape({2, 4, 8});
  const int total = shape.FlatSize();
  std::vector<float> input(total);
  for (int i = 0; i < total; ++i) {
    uint32_t bits =
        special_bits[i % special_bits.size()] ^ static_cast<uint32_t>(i & 0x3);
    std::memcpy(&input[i], &bits, sizeof(float));
  }

  for (int stride_1 : {1, 2, -1, -2}) {
    for (int stride_2 : {1, 2, -1}) {
      reference_ops::DynamicStridedSliceParams params;
      params.start_indices = {0, stride_1 > 0 ? 0 : 3, stride_2 > 0 ? 0 : 7};
      params.stop_indices = {2, stride_1 > 0 ? 4 : -1, stride_2 > 0 ? 8 : -1};
      params.strides = {1, stride_1, stride_2};
      params.end_mask =
          (stride_1 < 0 ? 0b010u : 0u) | (stride_2 < 0 ? 0b100u : 0u);
      VerifyBitForBitIdentical<float>(params, shape, input);
    }
  }
}

// Verify bit-for-bit identical serialized string buffers on multi-dimensional
// std::string tensors across coalesced, partial, strided/reversed, and empty
// slices, comparing both deserialized strings and raw serialized TfLiteTensor
// byte buffers against LegacyReferenceStridedSlice<std::string>.
TEST(StridedSliceBitExactStringTest, MultiDimCoalescedAndStridedStrings) {
  std::vector<std::string> input_data;
  input_data.reserve(24);
  for (int i = 0; i < 24; ++i) {
    if (i % 5 == 0) {
      input_data.push_back("");  // Empty string
    } else if (i % 7 == 0) {
      input_data.push_back(std::string("nul\0byte_", 9) + std::to_string(i));
    } else {
      input_data.push_back("token_str_" + std::to_string(i * 101));
    }
  }

  auto verify_raw_string_tensor_bit_exact =
      [&](const RuntimeShape& shape,
          const reference_ops::DynamicStridedSliceParams& params) {
        DynamicBuffer in_buf;
        for (const std::string& s : input_data) {
          in_buf.AddString(s.data(), s.size());
        }
        TfLiteTensor input_tensor{};
        input_tensor.type = kTfLiteString;
        input_tensor.allocation_type = kTfLiteDynamic;
        in_buf.WriteToTensor(&input_tensor, /*new_shape=*/nullptr);

        TfLiteTensor legacy_tensor{};
        legacy_tensor.type = kTfLiteString;
        legacy_tensor.allocation_type = kTfLiteDynamic;

        TfLiteTensor opt_tensor{};
        opt_tensor.type = kTfLiteString;
        opt_tensor.allocation_type = kTfLiteDynamic;

        {
          SequentialTensorWriter<std::string> legacy_writer(&input_tensor,
                                                            &legacy_tensor);
          LegacyReferenceStridedSlice<std::string>(params, shape, shape,
                                                   &legacy_writer);
        }
        {
          SequentialTensorWriter<std::string> opt_writer(&input_tensor,
                                                         &opt_tensor);
          reference_ops::StridedSlice<std::string>(params, shape, shape,
                                                   &opt_writer);
        }

        ASSERT_EQ(legacy_tensor.bytes, opt_tensor.bytes);
        if (legacy_tensor.bytes > 0) {
          EXPECT_EQ(std::memcmp(legacy_tensor.data.raw, opt_tensor.data.raw,
                                legacy_tensor.bytes),
                    0);
        }

        free(input_tensor.data.raw);
        free(legacy_tensor.data.raw);
        free(opt_tensor.data.raw);

        // Also verify a non-trivially-copyable type with the raw-pointer
        // overload (SequentialTensorWriter<T>(const T*, T*)), which exercises
        // the !std::is_trivially_copyable_v<T> element-by-element Write
        // path in write_contiguous without invoking memcpy.
        struct NonTriviallyCopyableVal {
          int v = -1;
          const NonTriviallyCopyableVal* self = this;
          NonTriviallyCopyableVal() = default;
          explicit NonTriviallyCopyableVal(int x) : v(x), self(this) {}
          NonTriviallyCopyableVal(const NonTriviallyCopyableVal& o)
              : v(o.v), self(this) {}
          NonTriviallyCopyableVal& operator=(const NonTriviallyCopyableVal& o) {
            v = o.v;
            self = this;
            return *this;
          }
          bool operator==(const NonTriviallyCopyableVal& o) const {
            return v == o.v && self == this && o.self == &o;
          }
        };
        static_assert(!std::is_trivially_copyable_v<NonTriviallyCopyableVal>,
                      "Must be non-trivially copyable");
        std::vector<NonTriviallyCopyableVal> ntc_in(shape.FlatSize());
        for (int i = 0; i < shape.FlatSize(); ++i) {
          ntc_in[i] = NonTriviallyCopyableVal(i + 100);
        }
        std::vector<NonTriviallyCopyableVal> ntc_legacy(
            shape.FlatSize(), NonTriviallyCopyableVal(-999));
        std::vector<NonTriviallyCopyableVal> ntc_actual(
            shape.FlatSize(), NonTriviallyCopyableVal(-999));
        SequentialTensorWriter<NonTriviallyCopyableVal> ntc_legacy_writer(
            ntc_in.data(), ntc_legacy.data());
        LegacyReferenceStridedSlice<NonTriviallyCopyableVal>(
            params, shape, shape, &ntc_legacy_writer);
        reference_ops::StridedSlice<NonTriviallyCopyableVal>(
            params, shape, ntc_in.data(), shape, ntc_actual.data());
        EXPECT_EQ(ntc_actual, ntc_legacy);
      };

  const RuntimeShape shape3d({2, 3, 4});

  // Case 1: Coalesced inner dimension [2, 3, 4] -> [0:2, 1:3, 0:4]
  {
    StridedSliceOpModel<std::string> m({2, 3, 4}, {3}, {3}, {3}, input_data,
                                       {0, 1, 0}, {2, 3, 4}, {1, 1, 1}, 0, 0, 0,
                                       0, 0, false);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 4}));
    std::vector<std::string> expected;
    for (int b = 0; b < 2; ++b) {
      for (int r = 1; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
          expected.push_back(input_data[b * 12 + r * 4 + c]);
        }
      }
    }
    EXPECT_THAT(m.GetStringOutput(), ElementsAreArray(expected));

    reference_ops::DynamicStridedSliceParams params;
    params.start_indices = {0, 1, 0};
    params.stop_indices = {2, 3, 4};
    params.strides = {1, 1, 1};
    verify_raw_string_tensor_bit_exact(shape3d, params);
  }

  // Case 2: Negative stride on middle axis with full unit-stride inner axis
  // (exercises input_strides[last_loop_axis] > 1 with WriteN on strings).
  {
    StridedSliceOpModel<std::string> m({2, 3, 4}, {3}, {3}, {3}, input_data,
                                       {0, 2, 0}, {2, -1, 4}, {1, -1, 1}, 0,
                                       0b010, 0, 0, 0, false);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 4}));
    std::vector<std::string> expected;
    for (int b = 0; b < 2; ++b) {
      for (int r = 2; r >= 0; --r) {
        for (int c = 0; c < 4; ++c) {
          expected.push_back(input_data[b * 12 + r * 4 + c]);
        }
      }
    }
    EXPECT_THAT(m.GetStringOutput(), ElementsAreArray(expected));

    reference_ops::DynamicStridedSliceParams params;
    params.start_indices = {0, 2, 0};
    params.stop_indices = {2, -1, 4};
    params.strides = {1, -1, 1};
    params.end_mask = 0b010;
    verify_raw_string_tensor_bit_exact(shape3d, params);
  }

  // Case 3: Partial unit-stride inner axis (1:3:1), strided/reversed inner axis
  // (3:-1:-1), and empty string slice (2:1:1).
  {
    reference_ops::DynamicStridedSliceParams partial_inner;
    partial_inner.start_indices = {0, 0, 1};
    partial_inner.stop_indices = {2, 3, 3};
    partial_inner.strides = {1, 1, 1};
    verify_raw_string_tensor_bit_exact(shape3d, partial_inner);

    reference_ops::DynamicStridedSliceParams reversed_inner;
    reversed_inner.start_indices = {0, 0, 3};
    reversed_inner.stop_indices = {2, 3, -1};
    reversed_inner.strides = {1, 2, -1};
    reversed_inner.end_mask = 0b100;
    verify_raw_string_tensor_bit_exact(shape3d, reversed_inner);

    reference_ops::DynamicStridedSliceParams empty_slice;
    empty_slice.start_indices = {0, 2, 0};
    empty_slice.stop_indices = {2, 1, 4};
    empty_slice.strides = {1, 1, 1};
    verify_raw_string_tensor_bit_exact(shape3d, empty_slice);
  }
}

// Microbenchmarks comparing LegacyReferenceStridedSlice vs optimized
// reference_ops::StridedSlice on representative speech_detector_alt_service
// feature tensors ([1, 100, 256] and [4, 64, 128]).
void BM_StridedSlice_SpeechFeature_Legacy(benchmark::State& state) {
  const RuntimeShape input_shape({1, 100, 256});
  const RuntimeShape output_shape({1, 99, 256});
  std::vector<float> input(input_shape.FlatSize(), 1.25f);
  std::vector<float> output(output_shape.FlatSize(), 0.0f);
  reference_ops::DynamicStridedSliceParams params;
  params.start_indices = {0, 1, 0};
  params.stop_indices = {1, 100, 256};
  params.strides = {1, 1, 1};

  for (auto _ : state) {
    SequentialTensorWriter<float> writer(input.data(), output.data());
    LegacyReferenceStridedSlice<float>(params, input_shape, output_shape,
                                       &writer);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_StridedSlice_SpeechFeature_Legacy)->MinTime(0.01);

void BM_StridedSlice_SpeechFeature_Optimized(benchmark::State& state) {
  const RuntimeShape input_shape({1, 100, 256});
  const RuntimeShape output_shape({1, 99, 256});
  std::vector<float> input(input_shape.FlatSize(), 1.25f);
  std::vector<float> output(output_shape.FlatSize(), 0.0f);
  reference_ops::DynamicStridedSliceParams params;
  params.start_indices = {0, 1, 0};
  params.stop_indices = {1, 100, 256};
  params.strides = {1, 1, 1};

  for (auto _ : state) {
    SequentialTensorWriter<float> writer(input.data(), output.data());
    reference_ops::StridedSlice<float>(params, input_shape, output_shape,
                                       &writer);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_StridedSlice_SpeechFeature_Optimized)->MinTime(0.01);

void BM_StridedSlice_BatchedSpeechFeature_Legacy(benchmark::State& state) {
  const RuntimeShape input_shape({4, 64, 128});
  const RuntimeShape output_shape({4, 60, 128});
  std::vector<float> input(input_shape.FlatSize(), 1.25f);
  std::vector<float> output(output_shape.FlatSize(), 0.0f);
  reference_ops::DynamicStridedSliceParams params;
  params.start_indices = {0, 2, 0};
  params.stop_indices = {4, 62, 128};
  params.strides = {1, 1, 1};

  for (auto _ : state) {
    SequentialTensorWriter<float> writer(input.data(), output.data());
    LegacyReferenceStridedSlice<float>(params, input_shape, output_shape,
                                       &writer);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_StridedSlice_BatchedSpeechFeature_Legacy)->MinTime(0.01);

void BM_StridedSlice_BatchedSpeechFeature_Optimized(benchmark::State& state) {
  const RuntimeShape input_shape({4, 64, 128});
  const RuntimeShape output_shape({4, 60, 128});
  std::vector<float> input(input_shape.FlatSize(), 1.25f);
  std::vector<float> output(output_shape.FlatSize(), 0.0f);
  reference_ops::DynamicStridedSliceParams params;
  params.start_indices = {0, 2, 0};
  params.stop_indices = {4, 62, 128};
  params.strides = {1, 1, 1};

  for (auto _ : state) {
    SequentialTensorWriter<float> writer(input.data(), output.data());
    reference_ops::StridedSlice<float>(params, input_shape, output_shape,
                                       &writer);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_StridedSlice_BatchedSpeechFeature_Optimized)->MinTime(0.01);

}  // namespace
}  // namespace tflite
