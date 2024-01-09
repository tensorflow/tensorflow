/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
         //
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <functional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using testing::ElementsAre;

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

template <class Container>
int32_t intsize(const Container& c) {
  return static_cast<int32_t>(c.size());
}

template <class T>
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

  void SetWindowShape(absl::Span<const int64_t> shape) {
    window_shape_data_.assign(shape.begin(), shape.end());
  }

  // Note: the strides are counted in elements on the tensor grid not in the
  // underlying buffer.
  //
  // For instance, {2,2} on the following matrix strting at element 1 will reach
  // elements 3 (+2 horizontally), 7 (+2 vertically) and 9 (+2 vertically, +2
  // horizontally):
  //
  // 1 2 3
  // 4 5 6
  // 7 8 9
  void SetWindowStrides(absl::Span<const int64_t> strides) {
    window_strides_data_.assign(strides.begin(), strides.end());
  }

  void SetWindowDilations(absl::Span<const int64_t> dilations) {
    window_dilations_data_.assign(dilations.begin(), dilations.end());
  }

  void SetInitValue(const T& val) { init_value_data_ = val; }

  void Build() {
    input_ = AddInput({kTensorType, input_shape_});
    init_value_ = AddConstInput(kTensorType, {init_value_data_}, {1});
    window_shape_ = AddConstInput(TensorType_INT64, window_shape_data_,
                                  {intsize(window_shape_data_)});
    window_strides_ = AddConstInput(TensorType_INT64, window_strides_data_,
                                    {intsize(window_strides_data_)});
    window_dilations_ = AddConstInput(TensorType_INT64, window_dilations_data_,
                                      {intsize(window_dilations_data_)});
    output_ = AddOutput(kTensorType);
    SetBuiltinOp(
        BuiltinOperator_REDUCE_WINDOW, BuiltinOptions2_ReduceWindowOptions,
        CreateReduceWindowOptions(builder_, ReduceWindowFunction_ADD).Union());
    BuildInterpreter({input_shape_});
    PopulateTensor(input_, input_data_);
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
  const std::vector<int32_t>& GetInputShape() const { return input_shape_; }
  const std::vector<int64_t>& GetWindowShape() const {
    return window_shape_data_;
  }
  const std::vector<int64_t>& GetWindowStrides() const {
    return window_strides_data_;
  }
  const std::vector<int64_t>& GetWindowDilations() const {
    return window_dilations_data_;
  }
  const T& GetInitValue() const { return init_value_data_; }

 protected:
  int input_ = -1;
  int window_shape_ = -1;
  int window_strides_ = -1;
  int window_dilations_ = -1;
  int init_value_ = -1;
  int output_ = -1;
  std::vector<T> input_data_;
  T init_value_data_;
  std::vector<int32_t> input_shape_;
  std::vector<int64_t> window_shape_data_;
  std::vector<int64_t> window_strides_data_;
  std::vector<int64_t> window_dilations_data_;
};

template <class StorageType>
class ReduceWindowTest : public testing::Test {
 protected:
  DilateOpModel<StorageType> model_;
};

using TestList =
    testing::Types<int8_t, int16_t, int32_t, int64_t, uint8_t, float, double>;

TYPED_TEST_SUITE(ReduceWindowTest, TestList);

TYPED_TEST(ReduceWindowTest, FullWindow) {
  auto& model = this->model_;
  model.SetInput(/*shape=*/{3, 3});
  model.SetWindowShape({3, 3});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(1, 1));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAre(45));
}

TYPED_TEST(ReduceWindowTest, NoDilation) {
  auto& model = this->model_;
  model.SetInput(/*shape=*/{3, 3});
  model.SetWindowShape({2, 2});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(2, 2));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAre(12, 16, 24, 28));
}

TYPED_TEST(ReduceWindowTest, FullWindowWithDilation) {
  auto& model = this->model_;
  model.SetInput(/*shape=*/{3, 3});
  model.SetWindowShape({2, 2});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({2, 2});
  model.SetInitValue(0);

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(1, 1));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAre(20));
}

TYPED_TEST(ReduceWindowTest, WithDilation) {
  auto& model = this->model_;
  model.SetInput(/*shape=*/{4, 4});
  model.SetWindowShape({2, 2});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({2, 2});
  model.SetInitValue(0);

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(2, 2));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAre(24, 28, 40, 44));
}

TYPED_TEST(ReduceWindowTest, WithStrides) {
  auto& model = this->model_;
  model.SetInput(/*shape=*/{4, 4});
  model.SetWindowShape({2, 2});
  model.SetWindowStrides({2, 2});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(2, 2));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAre(14, 22, 46, 54));
}

TYPED_TEST(ReduceWindowTest, WithDilationAndStrides) {
  auto& model = this->model_;
  model.SetInput(/*shape=*/{5, 5});
  model.SetWindowShape({2, 2});
  model.SetWindowStrides({2, 2});
  model.SetWindowDilations({2, 2});
  model.SetInitValue(2);

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(2, 2));
  EXPECT_THAT(this->model_.GetOutputData(), ElementsAre(30, 38, 70, 78));
}

TYPED_TEST(ReduceWindowTest, OutputShapeRoundingIsCorrect) {
  auto& model = this->model_;
  model.SetInput(/*shape=*/{1, 64, 114, 114});
  model.SetWindowShape({1, 1, 3, 3});
  model.SetWindowStrides({1, 1, 2, 2});
  model.SetWindowDilations({1, 1, 1, 1});
  model.SetInitValue(2);

  EXPECT_EQ(this->model_.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(this->model_.GetOutputShape(), ElementsAre(1, 64, 56, 56));
}

}  // namespace
}  // namespace tflite
