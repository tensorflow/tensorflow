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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/stablehlo_reduce_window_test_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace reduce_window {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

// TF_LITE_ENSURE* family of macros require a context to be passed, which we do
// not have when building the model.
#define REDUCE_WINDOW_ENSURE_OK(expr)                        \
  do {                                                       \
    if (TfLiteStatus status = (expr); status != kTfLiteOk) { \
      ABSL_LOG(ERROR) << #expr " failed.\n";                 \
      return status;                                         \
    }                                                        \
  } while (false)

// Returns kTfLiteError if the expression evaluates to false.
#define REDUCE_WINDOW_ENSURE_IMPL(expr, msg) \
  do {                                       \
    if (!(expr)) {                           \
      ABSL_LOG(ERROR) << #msg " failed.\n";  \
      return kTfLiteError;                   \
    }                                        \
  } while (false)

#define REDUCE_WINDOW_ENSURE(expr) REDUCE_WINDOW_ENSURE_IMPL((expr), #expr)

#define REDUCE_WINDOW_ENSURE_EQ(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) == (b), #a " == " #b)
#define REDUCE_WINDOW_ENSURE_NE(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) != (b), #a " != " #b)
#define REDUCE_WINDOW_ENSURE_GE(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) >= (b), #a " >= " #b)
#define REDUCE_WINDOW_ENSURE_LE(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) <= (b), #a " <= " #b)
#define REDUCE_WINDOW_ENSURE_GT(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) > (b), #a " > " #b)
#define REDUCE_WINDOW_ENSURE_LT(a, b) \
  REDUCE_WINDOW_ENSURE_IMPL((a) < (b), #a " < " #b)
#define REDUCE_WINDOW_ENSURE_UNREACHABLE(msg) \
  REDUCE_WINDOW_ENSURE_IMPL(false, msg)

// Maps the native C++ types to the corresponding TFLite tensor type enum
// values.
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

enum class BodyFunction {
  kUnset,
  kUnsupported,
  kAdd,
  kMul,
  kMax,
  kMin,
  kAll,
  kAny
};

std::ostream& operator<<(std::ostream& os, const BodyFunction& f) {
  switch (f) {
    case BodyFunction::kUnset:
      return os << "unset";
    case BodyFunction::kUnsupported:
      return os << "unsupported";
    case BodyFunction::kAdd:
      return os << "add";
    case BodyFunction::kMul:
      return os << "mul";
    case BodyFunction::kMax:
      return os << "max";
    case BodyFunction::kMin:
      return os << "min";
    case BodyFunction::kAll:
      return os << "all";
    case BodyFunction::kAny:
      return os << "any";
  }
  return os;
}

template <class T>
class ReduceWindowOpModel : public SingleOpModel {
  static constexpr TensorType kTensorType = TensorTypeFor<T>::value;

 public:
  // Sets the input tensor shape and data.
  //
  // If the data isn't provided, the buffer is filled with `iota`.
  void SetInput(absl::Span<const int64_t> shape) {
    input_shape_.assign(shape.begin(), shape.end());
    input_data_.resize(absl::c_accumulate(shape, 1, std::multiplies<>()));
    absl::c_iota(input_data_, 1);
  }

  void SetInput(absl::Span<const int64_t> shape, absl::Span<const T> data) {
    input_shape_.assign(shape.begin(), shape.end());
    input_data_.assign(data.begin(), data.end());
  }

  void SetInput(absl::Span<const int64_t> shape, absl::BitGenRef bitgen, T min,
                T max) {
    input_shape_.assign(shape.begin(), shape.end());
    input_data_.resize(absl::c_accumulate(shape, 1, std::multiplies<>()));
    absl::c_generate(input_data_, [&] {
      return absl::Uniform(absl::IntervalClosed, bitgen, min, max);
    });
  }

  void SetWindowDimensions(absl::Span<const int64_t> dimensions) {
    window_dimensions_.assign(dimensions.begin(), dimensions.end());
  }

  // Note: the strides are counted in elements on the tensor grid not in the
  // underlying buffer.
  //
  // For instance, with {2,2} window strides on the following matrix, the window
  // anchored at element 1 will reach elements 3 (+2 horizontally), 7 (+2
  // vertically) and 9 (+2 vertically, +2 horizontally):
  //
  // 1 2 3
  // 4 5 6
  // 7 8 9
  void SetWindowStrides(absl::Span<const int64_t> strides) {
    window_strides_.assign(strides.begin(), strides.end());
  }

  void SetBaseDilations(absl::Span<const int64_t> dilations) {
    base_dilations_.assign(dilations.begin(), dilations.end());
  }

  void SetWindowDilations(absl::Span<const int64_t> dilations) {
    window_dilations_.assign(dilations.begin(), dilations.end());
  }

  void SetPadding(absl::Span<const int64_t> padding) {
    padding_.assign(padding.begin(), padding.end());
  }

  void SetInitValue(const T& val) { init_value_ = val; }

  void SetBody(const BodyFunction func) { body_function_ = func; }

  TfLiteStatus Build() {
    constexpr int kBodySubGraphIndex = 1;

    REDUCE_WINDOW_ENSURE(!input_shape_.empty());
    REDUCE_WINDOW_ENSURE_EQ(window_dimensions_.size(), input_shape_.size());
    REDUCE_WINDOW_ENSURE_EQ(window_strides_.size(), input_shape_.size());
    REDUCE_WINDOW_ENSURE_EQ(base_dilations_.size(), input_shape_.size());
    REDUCE_WINDOW_ENSURE_EQ(window_dilations_.size(), input_shape_.size());
    REDUCE_WINDOW_ENSURE_EQ(padding_.size(), 2 * input_shape_.size());
    REDUCE_WINDOW_ENSURE_NE(body_function_, BodyFunction::kUnset);
    REDUCE_WINDOW_ENSURE_NE(body_function_, BodyFunction::kUnsupported);

    input_tensor_id_ =
        AddInput({kTensorType,
                  std::vector<int>(input_shape_.begin(), input_shape_.end())});
    init_value_tensor_id_ = AddConstInput(kTensorType, {init_value_}, {1});
    output_tensor_id_ = AddOutput(kTensorType);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_REDUCE_WINDOW,
                 BuiltinOptions2_StablehloReduceWindowOptions,
                 CreateStablehloReduceWindowOptions(
                     builder_, builder_.CreateVector(window_dimensions_),
                     builder_.CreateVector(window_strides_),
                     builder_.CreateVector(base_dilations_),
                     builder_.CreateVector(window_dilations_),
                     builder_.CreateVector(padding_), kBodySubGraphIndex)
                     .Union());

    BuildInterpreter(
        /*input_shapes=*/{std::vector<int>(input_shape_.begin(),
                                           input_shape_.end())},
        /*num_threads=*/-1, /*allow_fp32_relax_to_fp16=*/false,
        /*apply_delegate=*/true, /*allocate_and_delegate=*/false,
        /*use_simple_allocator=*/false);

    int body_subgraph_index;
    AddSubgraphs(1, &body_subgraph_index);
    REDUCE_WINDOW_ENSURE_EQ(body_subgraph_index, kBodySubGraphIndex);
    switch (body_function_) {
      case BodyFunction::kAdd:
        subgraph_builder_.BuildAddSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      case BodyFunction::kMul:
        subgraph_builder_.BuildMulSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      case BodyFunction::kMax:
        subgraph_builder_.BuildMaximumSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      case BodyFunction::kMin:
        subgraph_builder_.BuildMinimumSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      case BodyFunction::kAll:
        subgraph_builder_.BuildLogicalAndSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      case BodyFunction::kAny:
        subgraph_builder_.BuildLogicalOrSubgraph(
            interpreter_->subgraph(body_subgraph_index));
        break;
      default:
        REDUCE_WINDOW_ENSURE_UNREACHABLE("Unhandled body function enum value.");
    }

    AllocateAndDelegate(/*apply_delegate=*/true);

    PopulateTensor(input_tensor_id_, input_data_);
    return kTfLiteOk;
  }

  TfLiteStatus BuildAndInvoke() {
    REDUCE_WINDOW_ENSURE_OK(Build());
    return Invoke();
  }

  absl::Span<const T> GetOutputData() {
    return absl::Span<const T>(interpreter_->typed_tensor<T>(output_tensor_id_),
                               GetTensorSize(output_tensor_id_));
  }

  absl::Span<const int> GetOutputShape() {
    const TfLiteIntArray& shape =
        *(interpreter_->tensor(output_tensor_id_)->dims);
    return absl::Span<const int>(shape.data, shape.size);
  }

  const std::vector<T>& GetInput() const { return input_data_; }

  const std::vector<int64_t>& GetInputShape() const { return input_shape_; }

  const std::vector<int64_t>& GetWindowDimensions() const {
    return window_dimensions_;
  }

  const std::vector<int64_t>& GetWindowStrides() const {
    return window_strides_;
  }

  const std::vector<int64_t>& GetBaseDilations() const {
    return base_dilations_;
  }

  const std::vector<int64_t>& GetWindowDilations() const {
    return window_dilations_;
  }

  const std::vector<int64_t>& GetPadding() const { return padding_; }

  const T& GetInitValue() const { return init_value_; }

  const BodyFunction& GetBodyFunction() const { return body_function_; }

  friend std::ostream& operator<<(std::ostream& os,
                                  const ReduceWindowOpModel& model) {
    using Adapt = ReduceWindowOpModel::VectorOutputAdapter;
    os << "input dimensions: {" << Adapt{model.GetInputShape()} << "}\n";
    os << "  base dilations: {" << Adapt{model.GetBaseDilations()} << "}\n";
    os << "  padding: {" << Adapt{model.GetPadding()} << "}\n";
    os << "  window dimensions: {" << Adapt{model.GetWindowDimensions()}
       << "}\n";
    os << "  window dilations: {" << Adapt{model.GetWindowDilations()} << "}\n";
    os << "  window strides: {" << Adapt{model.GetWindowStrides()} << "}\n";
    os << "  init value: " << +model.GetInitValue() << "\n";
    os << "  body function: " << model.GetBodyFunction() << "\n";
    return os;
  }

 protected:
  struct VectorOutputAdapter {
    const std::vector<int64_t>& data;
    friend std::ostream& operator<<(std::ostream& os,
                                    const VectorOutputAdapter& vec) {
      if (!vec.data.empty()) {
        os << +vec.data[0];
        for (size_t i = 1; i < vec.data.size(); ++i) {
          os << ", " << +vec.data[i];
        }
      }
      return os;
    }
  };

  int input_tensor_id_ = -1;
  int init_value_tensor_id_ = -1;
  int output_tensor_id_ = -1;
  std::vector<T> input_data_;
  T init_value_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> window_dimensions_;
  std::vector<int64_t> window_strides_;
  std::vector<int64_t> base_dilations_;
  std::vector<int64_t> window_dilations_;
  std::vector<int64_t> padding_;
  BodyFunction body_function_{};
  subgraph_test_util::SubgraphBuilder subgraph_builder_;
};

template <class StorageType>
class StablehloReduceWindowTest : public testing::Test {};

using TestList =
    testing::Types<int8_t, int16_t, int32_t, int64_t, uint8_t, float, double>;
TYPED_TEST_SUITE(StablehloReduceWindowTest, TestList);

TYPED_TEST(StablehloReduceWindowTest, Identity) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, 0, 0});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 3));
  EXPECT_THAT(model.GetOutputData(), ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9));
}

TYPED_TEST(StablehloReduceWindowTest, Dilate) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({2, 2});
  model.SetPadding({0, 0, 0, 0});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(5, 5));
  EXPECT_THAT(model.GetOutputData(),
              ElementsAreArray({1, 0, 2, 0, 3, 0, 0, 0, 0, 0, 4, 0, 5,
                                0, 6, 0, 0, 0, 0, 0, 7, 0, 8, 0, 9}));
}

TYPED_TEST(StablehloReduceWindowTest, IdentityPadTop) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({1, 0, 0, 0});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3));
  EXPECT_THAT(model.GetOutputData(),
              ElementsAreArray({0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TYPED_TEST(StablehloReduceWindowTest, IdentityPadBottom) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 1, 0, 0});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3));
  EXPECT_THAT(model.GetOutputData(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0}));
}

TYPED_TEST(StablehloReduceWindowTest, IdentityPadLeft) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, 1, 0});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 4));
  EXPECT_THAT(model.GetOutputData(),
              ElementsAreArray({0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9}));
}

TYPED_TEST(StablehloReduceWindowTest, IdentityPadRight) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, 0, 1});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 4));
  EXPECT_THAT(model.GetOutputData(),
              ElementsAreArray({1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0}));
}

TYPED_TEST(StablehloReduceWindowTest, IdentityPadAll) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({1, 1, 1, 1});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(5, 5));
  EXPECT_THAT(model.GetOutputData(),
              ElementsAreArray({0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5,
                                6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0}));
}

TYPED_TEST(StablehloReduceWindowTest, IdentityCropTop) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({-1, 0, 0, 0});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray({4, 5, 6, 7, 8, 9}));
}

TYPED_TEST(StablehloReduceWindowTest, IdentityCropBottom) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, -1, 0, 0});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(StablehloReduceWindowTest, IdentityCropLeft) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, -1, 0});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray({2, 3, 5, 6, 8, 9}));
}

TYPED_TEST(StablehloReduceWindowTest, IdentityCropRight) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, 0, -1});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray({1, 2, 4, 5, 7, 8}));
}

TYPED_TEST(StablehloReduceWindowTest, IdentityCropAll) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({-1, -1, -1, -1});
  model.SetWindowDimensions({1, 1});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1));
  EXPECT_THAT(model.GetOutputData(), ElementsAre(5));
}

TYPED_TEST(StablehloReduceWindowTest, ReduceWindowFullWindow) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, 0, 0});
  model.SetWindowDimensions({3, 3});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);
  model.SetBody(BodyFunction::kAdd);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1));
  EXPECT_THAT(model.GetOutputData(), ElementsAre(45));
}

TYPED_TEST(StablehloReduceWindowTest, ReduceWindowNoDilation) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, 0, 0});
  model.SetBody(BodyFunction::kAdd);
  model.SetWindowDimensions({2, 2});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2));
  EXPECT_THAT(model.GetOutputData(), ElementsAre(12, 16, 24, 28));
}

TYPED_TEST(StablehloReduceWindowTest, ReduceWindowFullWindowWithDilation) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{3, 3});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, 0, 0});
  model.SetBody(BodyFunction::kAdd);
  model.SetWindowDimensions({2, 2});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({2, 2});
  model.SetInitValue(0);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1));
  EXPECT_THAT(model.GetOutputData(), ElementsAre(20));
}

TYPED_TEST(StablehloReduceWindowTest, ReduceWindowWithDilation) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{4, 4});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, 0, 0});
  model.SetBody(BodyFunction::kAdd);
  model.SetWindowDimensions({2, 2});
  model.SetWindowStrides({1, 1});
  model.SetWindowDilations({2, 2});
  model.SetInitValue(0);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2));
  EXPECT_THAT(model.GetOutputData(), ElementsAre(24, 28, 40, 44));
}

TYPED_TEST(StablehloReduceWindowTest, ReduceWindowWithStrides) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{4, 4});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, 0, 0});
  model.SetBody(BodyFunction::kAdd);
  model.SetWindowDimensions({2, 2});
  model.SetWindowStrides({2, 2});
  model.SetWindowDilations({1, 1});
  model.SetInitValue(0);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2));
  EXPECT_THAT(model.GetOutputData(), ElementsAre(14, 22, 46, 54));
}

TYPED_TEST(StablehloReduceWindowTest, ReduceWindowWithDilationAndStrides) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{5, 5});
  model.SetBaseDilations({1, 1});
  model.SetPadding({0, 0, 0, 0});
  model.SetBody(BodyFunction::kAdd);
  model.SetWindowDimensions({2, 2});
  model.SetWindowStrides({2, 2});
  model.SetWindowDilations({2, 2});
  model.SetInitValue(2);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2));
  EXPECT_THAT(model.GetOutputData(), ElementsAre(30, 38, 70, 78));
}

TYPED_TEST(StablehloReduceWindowTest,
           ReduceWindowOutputShapeRoundingIsCorrect) {
  ReduceWindowOpModel<TypeParam> model;
  model.SetInput(/*shape=*/{1, 64, 114, 114});
  model.SetBaseDilations({1, 1, 1, 1});
  model.SetPadding({0, 0, 0, 0, 0, 0, 0, 0});
  model.SetBody(BodyFunction::kAdd);
  model.SetWindowDimensions({1, 1, 3, 3});
  model.SetWindowStrides({1, 1, 2, 2});
  model.SetWindowDilations({1, 1, 1, 1});
  model.SetInitValue(2);

  ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 64, 56, 56));
}

// Returns a vector of given size with elements in the range [min, max].
template <class T>
std::vector<T> RandomVector(absl::BitGen& bitgen, size_t size, T min, T max) {
  std::vector<T> vec(size);
  for (T& v : vec) {
    v = absl::Uniform(absl::IntervalClosed, bitgen, min, max);
  }
  return vec;
}

struct Body {
  static Body GetRandomSupported(absl::BitGen& bitgen, bool allow_mul) {
    Body b;
    b = Body{/*.body=*/static_cast<BodyFunction>(absl::Uniform<int>(
        absl::IntervalClosed, bitgen, static_cast<int>(BodyFunction::kAdd),
        static_cast<int>(BodyFunction::kAny)))};
    // This skews the uniformity of the random generation in favor of add. We
    // only need to ensure that all the cases are tested.
    if (!allow_mul && b.func == BodyFunction::kMul) {
      b.func = BodyFunction::kAdd;
    }
    return b;
  }

  template <class T>
  T operator()(const T& a, const T& b) const noexcept {
    switch (func) {
      case BodyFunction::kUnset:
      case BodyFunction::kUnsupported:
        return -1;
      case BodyFunction::kAdd:
        return a + b;
      case BodyFunction::kMul:
        return a * b;
      case BodyFunction::kMin:
        return a <= b ? a : b;
      case BodyFunction::kMax:
        return a >= b ? a : b;
      case BodyFunction::kAll:
        return a && b;
      case BodyFunction::kAny:
        return a || b;
    }
  }

  template <class T>
  T init_value() const noexcept {
    switch (func) {
      case BodyFunction::kUnset:
      case BodyFunction::kUnsupported:
        return -1;
      case BodyFunction::kAdd:
        return 0;
      case BodyFunction::kMul:
        return 1;
      case BodyFunction::kMin:
        return std::numeric_limits<T>::max();
      case BodyFunction::kMax:
        return std::numeric_limits<T>::lowest();
      case BodyFunction::kAll:
        return true;
      case BodyFunction::kAny:
        return false;
    }
  }

  BodyFunction func;
};

TYPED_TEST(StablehloReduceWindowTest, FuzzyTest) {
  absl::BitGen bitgen;

  for (size_t iteration = 0; iteration < 1000; ++iteration) {
    const int rank = absl::Uniform(absl::IntervalClosed, bitgen, 1, 3);

    ReduceWindowOpModel<TypeParam> model;
    // To avoid reduction overflows, we only test mul with floating point types.
    Body body = Body::GetRandomSupported(
        bitgen, /*allow_mul=*/std::is_floating_point<TypeParam>::value);
    model.SetInput(
        /*shape=*/RandomVector<int64_t>(bitgen, rank, /*min=*/1, /*max=*/10),
        bitgen, /*min=*/-5, /*max=*/5);
    model.SetBaseDilations(
        RandomVector<int64_t>(bitgen, rank, /*min=*/1, /*max=*/3));
    model.SetPadding(
        RandomVector<int64_t>(bitgen, 2 * rank, /*min=*/-5, /*max=*/5));
    model.SetWindowDimensions(
        RandomVector<int64_t>(bitgen, rank, /*min=*/1, /*max=*/3));
    model.SetWindowStrides(
        RandomVector<int64_t>(bitgen, rank, /*min=*/1, /*max=*/3));
    model.SetWindowDilations(
        RandomVector<int64_t>(bitgen, rank, /*min=*/1, /*max=*/3));
    model.SetInitValue(body.init_value<TypeParam>());
    model.SetBody(body.func);

    // Skip invalid specifications.
    const std::vector<int64_t> padded_shape = reference::PadCropShape(
        reference::DilateShape(model.GetInputShape(), model.GetBaseDilations()),
        model.GetPadding());
    if (absl::c_any_of(padded_shape, [](int64_t d) { return d <= 0; })) {
      iteration = iteration > 1 ? iteration - 1 : 0;
      continue;
    }

    const reference::Tensor<TypeParam> expected = reference::ReduceWindow(
        reference::Tensor<TypeParam>{/*shape=*/model.GetInputShape(),
                                     /*data=*/model.GetInput()},
        model.GetBaseDilations(), model.GetPadding(), model.GetInitValue(),
        model.GetWindowDimensions(), model.GetWindowDilations(),
        model.GetWindowStrides(), body);

    ASSERT_EQ(model.BuildAndInvoke(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape))
        << model;
    EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data))
        << model;
  }
}

}  // namespace
}  // namespace reduce_window
}  // namespace tflite
