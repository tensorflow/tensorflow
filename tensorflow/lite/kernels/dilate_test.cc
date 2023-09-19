/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

namespace tflite {
namespace {

// A reference implementation of the dilation operation.
template <class T>
std::vector<T> DilateReference(const std::vector<T>& input,
                               const std::vector<int32_t>& shape,
                               const std::vector<int32_t>& dilations,
                               const T padding_value) {
  constexpr int kMaxDilateDims = 6;

  // Compute the output shape.
  std::vector<int> output_shape(kMaxDilateDims, 0);
  for (size_t i = 0; i < shape.size(); ++i) {
    output_shape[i] = (shape[i] - 1) * dilations[i] + 1;
  }

  // Compute the input strides.
  std::vector<int> strides(kMaxDilateDims, 0);
  strides[shape.size() - 1] = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = shape[i] * strides[i];
  }

  // Compute the output strides.
  std::vector<int> output_strides(kMaxDilateDims, 0);
  output_strides[shape.size() - 1] = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    output_strides[i - 1] = output_shape[i] * output_strides[i];
  }

  // Copy the dilations to a buffer that can hold the maximum ranks.
  std::vector<int> safe_dilations(kMaxDilateDims, 0);
  absl::c_copy(dilations, safe_dilations.begin());

  // Copy the input shape to a buffer that can hold the maximum ranks.
  std::vector<int> safe_input_shape(kMaxDilateDims, 0);
  absl::c_copy(shape, safe_input_shape.begin());

  // Create a buffer that can hold the output data filled with 0.
  std::vector<T> output(
      std::accumulate(output_shape.begin(), output_shape.begin() + shape.size(),
                      1, std::multiplies<>()),
      padding_value);

  int a = 0;
  do {
    int b = 0;
    do {
      int c = 0;
      do {
        int d = 0;
        do {
          int e = 0;
          do {
            int f = 0;
            do {
              const int i_idx = a * strides[0] + b * strides[1] +
                                c * strides[2] + d * strides[3] +
                                e * strides[4] + f * strides[5];
              const int o_idx = a * safe_dilations[0] * output_strides[0] +
                                b * safe_dilations[1] * output_strides[1] +
                                c * safe_dilations[2] * output_strides[2] +
                                d * safe_dilations[3] * output_strides[3] +
                                e * safe_dilations[4] * output_strides[4] +
                                f * safe_dilations[5] * output_strides[5];
              output[o_idx] = input[i_idx];
            } while (++f < safe_input_shape[5]);
          } while (++e < safe_input_shape[4]);
        } while (++d < safe_input_shape[3]);
      } while (++c < safe_input_shape[2]);
    } while (++b < safe_input_shape[1]);
  } while (++a < safe_input_shape[0]);

  return output;
}

template <class T>
struct TensorTypeFor;

#define TENSOR_TYPE_ASSOC(CPP_TYPE, TENSORTYPE_VALUE)     \
  template <>                                             \
  struct TensorTypeFor<CPP_TYPE> {                        \
    static constexpr TensorType value = TENSORTYPE_VALUE; \
  };

TENSOR_TYPE_ASSOC(int8_t, TensorType_INT8);
TENSOR_TYPE_ASSOC(int16_t, TensorType_INT16);
TENSOR_TYPE_ASSOC(int32_t, TensorType_INT32);
TENSOR_TYPE_ASSOC(int64_t, TensorType_INT64);

TENSOR_TYPE_ASSOC(uint8_t, TensorType_UINT8);
TENSOR_TYPE_ASSOC(uint16_t, TensorType_UINT16);
TENSOR_TYPE_ASSOC(uint32_t, TensorType_UINT32);
TENSOR_TYPE_ASSOC(uint64_t, TensorType_UINT64);

TENSOR_TYPE_ASSOC(float, TensorType_FLOAT32);
static_assert(sizeof(float) == 4, "float type is expected to be 32 bit long");
TENSOR_TYPE_ASSOC(double, TensorType_FLOAT64);
static_assert(sizeof(double) == 8, "double type is expected to be 64 bit long");

template <class T, bool IsDilationTensorConst>
class DilateOpModel : public SingleOpModel {
  static constexpr TensorType kTensorType = TensorTypeFor<T>::value;

 public:
  void SetInput(absl::Span<const int32_t> shape,
                absl::Span<const T> data = {}) {
    input_shape_.assign(shape.begin(), shape.end());
    if (data.empty()) {
      input_data_.resize(absl::c_accumulate(shape, 1, std::multiplies<int>()));
      absl::c_iota(input_data_, 1);
    } else {
      input_data_.assign(data.begin(), data.end());
    }
  }

  void SetDilations(absl::Span<const int32_t> dilations) {
    dilations_shape_ = std::vector<int>(1, dilations.size());
    dilations_data_.assign(dilations.begin(), dilations.end());
  }

  void SetPaddingValue(const T& val) { padding_value_data_ = val; }

  void Build() {
    input_ = AddInput({kTensorType, input_shape_});
    if (IsDilationTensorConst) {
      dilations_ = AddConstInput(TensorType_INT32, dilations_data_,
                                 {static_cast<int>(dilations_data_.size())});
    } else {
      dilations_ = AddInput({TensorType_INT32, dilations_shape_});
    }
    padding_value_ = AddConstInput(kTensorType, &padding_value_data_, {1});
    output_ = AddOutput(kTensorType);
    SetBuiltinOp(BuiltinOperator_DILATE, BuiltinOptions2_DilateOptions,
                 CreateDilateOptions(builder_).Union());
    BuildInterpreter({input_shape_});
    PopulateTensor(input_, input_data_);
    if (!IsDilationTensorConst) {
      PopulateTensor(dilations_, dilations_data_);
    }
  }

  TfLiteStatus BuildAndInvoke() {
    Build();
    return Invoke();
  }

  absl::Span<const T> GetOutputData() {
    return absl::Span<const T>(interpreter_->typed_tensor<T>(output_),
                               GetTensorSize(output_));
  }

  absl::Span<const int> GetOutputShape() {
    const TfLiteIntArray& shape = *(interpreter_->tensor(output_)->dims);
    return absl::Span<const int>(shape.data, shape.size);
  }

  const std::vector<T>& GetInput() const { return input_data_; }
  const std::vector<int>& GetInputShape() const { return input_shape_; }
  const std::vector<int>& GetDilations() const { return dilations_data_; }
  const T& GetPaddingValue() const { return padding_value_data_; }

 protected:
  int input_ = -1;
  int dilations_ = -1;
  int padding_value_ = -1;
  int output_ = -1;
  std::vector<T> input_data_;
  std::vector<int32_t> input_shape_;
  std::vector<int32_t> dilations_data_;
  std::vector<int32_t> dilations_shape_;
  T padding_value_data_ = 0;
};

template <class Configuration>
class DilateTest;

template <class StorageType, class IsDilationTensorConst>
class DilateTest<testing::Types<StorageType, IsDilationTensorConst>>
    : public testing::Test {
 protected:
  DilateOpModel<StorageType, IsDilationTensorConst::value> model_;
};

struct ConstantDilation : std::true_type {};
struct VariableDilation : std::false_type {};

using TestList = testing::Types<testing::Types<int8_t, ConstantDilation>,
                                testing::Types<int16_t, ConstantDilation>,
                                testing::Types<int32_t, ConstantDilation>,
                                testing::Types<int64_t, ConstantDilation>,
                                testing::Types<uint8_t, ConstantDilation>,
                                testing::Types<uint16_t, ConstantDilation>,
                                testing::Types<uint32_t, ConstantDilation>,
                                testing::Types<uint64_t, ConstantDilation>,
                                testing::Types<float, ConstantDilation>,
                                testing::Types<double, ConstantDilation>,
                                testing::Types<int8_t, VariableDilation>,
                                testing::Types<int16_t, VariableDilation>,
                                testing::Types<int32_t, VariableDilation>,
                                testing::Types<int64_t, VariableDilation>,
                                testing::Types<uint8_t, VariableDilation>,
                                testing::Types<uint16_t, VariableDilation>,
                                testing::Types<uint32_t, VariableDilation>,
                                testing::Types<uint64_t, VariableDilation>,
                                testing::Types<float, VariableDilation>,
                                testing::Types<double, VariableDilation>>;

TYPED_TEST_SUITE(DilateTest, TestList);

TYPED_TEST(DilateTest, DilationManualTest) {
  this->model_.SetInput(/*shape=*/{2, 2});
  this->model_.SetDilations(/*dilations=*/{2, 3});

  const std::vector<int> expected{
      /* clang-format off */
      1, 0, 0, 2,
      0, 0, 0, 0,
      3, 0, 0, 4
      /* clang-format on */
  };

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(3, 4));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAreArray(expected));
}

TYPED_TEST(DilateTest, DilationManualTest2) {
  this->model_.SetInput(/*shape=*/{2, 3});
  this->model_.SetDilations(/*dilations=*/{2, 3});

  const std::vector<int> expected{
      /* clang-format off */
      1, 0, 0, 2, 0, 0, 3,
      0, 0, 0, 0, 0, 0, 0,
      4, 0, 0, 5, 0, 0, 6
      /* clang-format on */
  };

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(3, 7));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAreArray(expected));
}

TYPED_TEST(DilateTest, DilationManualTest3) {
  this->model_.SetInput(/*shape=*/{4, 2, 3});
  this->model_.SetDilations({2, 3, 4});

  const std::vector<int> expected{
      /* clang-format off */
       1,  0,  0,  0,  2,  0,  0,  0,  3,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       4,  0,  0,  0,  5,  0,  0,  0,  6,

       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,

       7,  0,  0,  0,  8,  0,  0,  0,  9,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
      10,  0,  0,  0, 11,  0,  0,  0, 12,

       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,

      13,  0,  0,  0, 14,  0,  0,  0, 15,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
      16,  0,  0,  0, 17,  0,  0,  0, 18,

       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,

      19,  0,  0,  0, 20,  0,  0,  0, 21,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,
      22,  0,  0,  0, 23,  0,  0,  0, 24,
      /* clang-format on */
  };

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(7, 4, 9));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAreArray(expected));
}

TYPED_TEST(DilateTest, TrailingDilationOptimizationWorks) {
  this->model_.SetInput(/*shape=*/{2, 2, 2, 2});
  this->model_.SetDilations(/*dilations=*/{2, 1, 1, 1});

  const std::vector<int> expected{
      /* clang-format off */
      1,  2,  3,  4,  5,  6,  7,  8,
      0,  0,  0,  0,  0,  0,  0,  0,
      9, 10, 11, 12, 13, 14, 15, 16
      /* clang-format on */
  };

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(3, 2, 2, 2));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAreArray(expected));
}

TYPED_TEST(DilateTest, TrailingDilationOptimizationDegenerateCaseWorks) {
  this->model_.SetInput(/*shape=*/{2, 2, 2, 2});
  this->model_.SetDilations(/*dilations=*/{1, 1, 1, 1});

  const std::vector<int> expected{
      /* clang-format off */
      1,  2,  3,  4,  5,  6,  7,  8,
      9, 10, 11, 12, 13, 14, 15, 16
      /* clang-format on */
  };

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(2, 2, 2, 2));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAreArray(expected));
}

TYPED_TEST(DilateTest, CheckAgainstReferenceImplementation) {
  auto& model = this->model_;
  model.SetInput(/*shape=*/{5, 4, 2});
  model.SetDilations(/*dilations=*/{2, 3, 5});
  model.SetPaddingValue(-1);

  const auto expected =
      DilateReference(model.GetInput(), model.GetInputShape(),
                      model.GetDilations(), model.GetPaddingValue());
  EXPECT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected));
}

}  // namespace
}  // namespace tflite
