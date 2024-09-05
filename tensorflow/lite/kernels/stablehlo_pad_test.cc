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
// #include <gmock/gmock.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/stablehlo_reduce_window_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_pad {
namespace {

using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::HasSubstr;

template <class T>
class StablehloPadModel : public SingleOpModel {
 public:
  static constexpr TensorType kTensorType = GetTensorType<T>();

  void SetEdgePadding(std::vector<int64_t> low, std::vector<int64_t> high) {
    edge_padding_low_ = std::move(low);
    edge_padding_high_ = std::move(high);
  }

  const std::vector<int64_t>& GetEdgePaddingLow() const {
    return edge_padding_low_;
  }

  const std::vector<int64_t>& GetEdgePaddingHigh() const {
    return edge_padding_high_;
  }

  void SetInteriorPadding(std::vector<int64_t> padding) {
    interior_padding_ = std::move(padding);
  }

  const std::vector<int64_t>& GetInteriorPadding() const {
    return interior_padding_;
  }

  void SetInput(std::vector<int64_t> shape) {
    input_.shape = shape;
    input_.data.resize(absl::c_accumulate(shape, 1, std::multiplies<>()));
    absl::c_iota(input_.data, static_cast<T>(1));
  }

  void SetInput(std::vector<int64_t> shape, std::vector<T> data) {
    input_.shape = shape;
    input_.data = data;
  }

  void SetInput(absl::Span<const int64_t> shape, absl::BitGenRef bitgen, T min,
                T max) {
    input_.shape.assign(shape.begin(), shape.end());
    input_.data.resize(absl::c_accumulate(shape, 1, std::multiplies<>()));
    absl::c_generate(input_.data, [&] {
      return absl::Uniform(absl::IntervalClosed, bitgen, min, max);
    });
  }

  const reduce_window::reference::Tensor<T>& GetInput() const { return input_; }

  void SetPaddingValue(const T& v) { padding_value_ = v; }

  T GetPaddingValue() const { return padding_value_; }

  absl::Span<const T> GetOutputData() {
    return absl::Span<const T>(interpreter_->typed_tensor<T>(output_tensor_id_),
                               GetTensorSize(output_tensor_id_));
  }

  absl::Span<const int> GetOutputShape() {
    const TfLiteIntArray& shape =
        *(interpreter_->tensor(output_tensor_id_)->dims);
    return absl::Span<const int>(shape.data, shape.size);
  }

  absl::Status CheckPreconditions() {
    const size_t rank = input_.shape.size();
    if (rank == 0) {
      return absl::FailedPreconditionError("Input rank is 0.");
    }
    if (edge_padding_low_.empty()) {
      edge_padding_low_ = std::vector<int64_t>(rank, 0);
    } else if (edge_padding_low_.size() != rank) {
      return absl::FailedPreconditionError(
          "Low edge padding does not have the right size.");
    }
    if (edge_padding_high_.empty()) {
      edge_padding_high_ = std::vector<int64_t>(rank, 0);
    } else if (edge_padding_high_.size() != rank) {
      return absl::FailedPreconditionError(
          "High edge padding does not have the right size.");
    }
    if (interior_padding_.empty()) {
      interior_padding_ = std::vector<int64_t>(rank, 0);
    } else if (interior_padding_.size() != rank) {
      return absl::FailedPreconditionError(
          "Interior padding does not have the right size.");
    }
    return absl::OkStatus();
  }

  absl::Status Build() {
    if (absl::Status status = CheckPreconditions(); !status.ok()) {
      return status;
    }
    input_tensor_id_ =
        AddInput({kTensorType,
                  std::vector<int>(input_.shape.begin(), input_.shape.end())});
    padding_value_tensor_id_ =
        AddConstInput(kTensorType, /*data=*/{padding_value_}, /*shape=*/{1});
    output_tensor_id_ = AddOutput(kTensorType);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_PAD,
                 BuiltinOptions2_StablehloPadOptions,
                 CreateStablehloPadOptions(
                     builder_, builder_.CreateVector(edge_padding_low_),
                     builder_.CreateVector(edge_padding_high_),
                     builder_.CreateVector(interior_padding_))
                     .Union());
    BuildInterpreter(
        /*input_shapes=*/{std::vector<int>(input_.shape.begin(),
                                           input_.shape.end())},
        /*num_threads=*/-1, /*allow_fp32_relax_to_fp16=*/false,
        /*apply_delegate=*/true, /*allocate_and_delegate=*/false,
        /*use_simple_allocator=*/false);
    AllocateAndDelegate(/*apply_delegate=*/true);
    PopulateTensor(input_tensor_id_, input_.data);
    return absl::OkStatus();
  }

  absl::Status BuildAndInvoke() {
    if (absl::Status status = Build(); !status.ok()) {
      return status;
    }
    if (TfLiteStatus status = Invoke(); status != kTfLiteOk) {
      const std::string msg =
          absl::StrFormat("Invoke failed with status %d.", status);
      return absl::InternalError(msg);
    }
    return absl::OkStatus();
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const StablehloPadModel& model) {
    auto print_vec = [&os](const auto& vec) {
      os << "[";
      if (!vec.empty()) {
        auto it = vec.begin();
        os << +*(it++);
        for (; it != vec.end(); ++it) {
          os << ", " << +*it;
        }
      }
      os << "]";
    };
    os << "  edge_padding_low: ";
    print_vec(model.GetEdgePaddingLow());
    os << "\n  edge_padding_high: ";
    print_vec(model.GetEdgePaddingHigh());
    os << "\n  interior_padding: ";
    print_vec(model.GetInteriorPadding());
    os << "\n  padding_value: " << +model.GetPaddingValue();
    os << "\n  input shape: ";
    print_vec(model.GetInput().shape);
    return os;
  }

 private:
  std::vector<int64_t> edge_padding_low_;
  std::vector<int64_t> edge_padding_high_;
  std::vector<int64_t> interior_padding_;
  reduce_window::reference::Tensor<T> input_;
  T padding_value_ = 0;

  int input_tensor_id_;
  int padding_value_tensor_id_;
  int output_tensor_id_;
};

template <class T>
absl::StatusOr<reduce_window::reference::Tensor<T>> ComputeReference(
    StablehloPadModel<T>& model) {
  if (absl::Status status = model.CheckPreconditions(); !status.ok()) {
    return status;
  }
  std::vector<int64_t> dilations, padding;
  for (size_t i = 0; i < model.GetInput().shape.size(); ++i) {
    padding.push_back(model.GetEdgePaddingLow()[i]);
    padding.push_back(model.GetEdgePaddingHigh()[i]);
    dilations.push_back(model.GetInteriorPadding()[i] + 1);
  }

  auto dilated_tensor = reduce_window::reference::Dilate(
      model.GetInput(), dilations, model.GetPaddingValue());
  auto padded_tensor = reduce_window::reference::Pad(dilated_tensor, padding,
                                                     model.GetPaddingValue());
  return reduce_window::reference::Crop(padded_tensor, padding);
}

TEST(StablehloPadModelTest, DefaultModelFails) {
  StablehloPadModel<int> model;
  const auto expected_status = ComputeReference(model);
  EXPECT_FALSE(expected_status.ok());
  EXPECT_EQ(expected_status.status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(expected_status.status().message(),
              HasSubstr("Input rank is 0."));
}

TEST(StablehloPadModelTest, DefaultModelReturnsIdentity) {
  StablehloPadModel<int> model;
  model.SetInput({3, 1});
  EXPECT_THAT(model.GetInput().shape, ElementsAre(3, 1));
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  EXPECT_THAT(expected_status.value().data,
              ElementsAreArray(model.GetInput().data));
}

TEST(StablehloPadModelTest, WrongEdgePaddingSizeIsAnError) {
  StablehloPadModel<int> model;
  model.SetInput({3, 1});
  model.SetEdgePadding(/*low=*/{3, 4, 5}, /*high=*/{6, 7});
  {
    const auto expected_status = ComputeReference(model);
    EXPECT_FALSE(expected_status.ok());
    EXPECT_EQ(expected_status.status().code(),
              absl::StatusCode::kFailedPrecondition);
    EXPECT_THAT(expected_status.status().message(),
                HasSubstr("Low edge padding does not have the right size."));
  }
  model.SetEdgePadding(/*low=*/{3, 4}, /*high=*/{5, 6, 7});
  {
    const auto expected_status = ComputeReference(model);
    EXPECT_FALSE(expected_status.ok());
    EXPECT_EQ(expected_status.status().code(),
              absl::StatusCode::kFailedPrecondition);
    EXPECT_THAT(expected_status.status().message(),
                HasSubstr("High edge padding does not have the right size."));
  }
}

TEST(StablehloPadModelTest, WrongInteriorPaddingSizeIsAnError) {
  StablehloPadModel<int> model;
  model.SetInput({3, 1});
  model.SetInteriorPadding({3, 4, 5});
  const auto expected_status = ComputeReference(model);
  EXPECT_FALSE(expected_status.ok());
  EXPECT_EQ(expected_status.status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(expected_status.status().message(),
              HasSubstr("Interior padding does not have the right size."));
}

TEST(StablehloPadTest, IdentityParams) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(model.GetInput().shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(model.GetInput().data));
}

TEST(StablehloPadTest, InteriorPad) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  model.SetInteriorPadding({1, 2});
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  const auto& expected = expected_status.value();
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data));
}

TEST(StablehloPadTest, LowPad) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  model.SetEdgePadding({1, 1}, {0, 0});
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  const auto& expected = expected_status.value();
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data));
}

TEST(StablehloPadTest, HighPad) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  model.SetEdgePadding({0, 0}, {1, 1});
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  const auto& expected = expected_status.value();
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data));
}

TEST(StablehloPadTest, AllPad) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  model.SetEdgePadding({1, 1}, {1, 1});
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  const auto& expected = expected_status.value();
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data));
}

TEST(StablehloPadTest, LowCrop) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  model.SetEdgePadding({-1, -1}, {0, 0});
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  const auto& expected = expected_status.value();
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data));
}

TEST(StablehloPadTest, HighCrop) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  model.SetEdgePadding({0, 0}, {-1, -1});
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  const auto& expected = expected_status.value();
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data));
}

TEST(StablehloPadTest, AllCrop) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  model.SetEdgePadding({-1, -1}, {-1, -1});
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  const auto& expected = expected_status.value();
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data));
}

TEST(StablehloPadTest, PadCrop) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  model.SetEdgePadding({1, -1}, {1, -1});
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  const auto& expected = expected_status.value();
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data));
}

TEST(StablehloPadTest, InteriorEdgePadding) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  model.SetEdgePadding({-1, -4}, {0, 0});
  model.SetInteriorPadding({1, 2});
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  const auto& expected = expected_status.value();
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data));
}

TEST(StablehloPadTest, CallPrepareTwiceDoesNotFail) {
  StablehloPadModel<int> model;
  model.SetInput({3, 3});
  model.SetEdgePadding({-1, -4}, {0, 0});
  model.SetInteriorPadding({1, 2});
  const auto expected_status = ComputeReference(model);
  ASSERT_TRUE(expected_status.ok());
  const auto& expected = expected_status.value();
  // Applying delegates forces Prepare to be called twice.
  model.SetApplyDefaultDelegates();
  ASSERT_TRUE(model.BuildAndInvoke().ok());
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape));
  EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data));
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

template <class T>
class StablehloPadFuzzyTest : public testing::Test {};

using TestList =
    testing::Types<int8_t, int16_t, int32_t, int64_t, uint8_t, float, double>;
TYPED_TEST_SUITE(StablehloPadFuzzyTest, TestList);

TYPED_TEST(StablehloPadFuzzyTest, FuzzyTest) {
  absl::BitGen bitgen;

  for (size_t iteration = 0; iteration < 10000; ++iteration) {
    const int rank = absl::Uniform(absl::IntervalClosed, bitgen, 1, 2);

    StablehloPadModel<TypeParam> model;
    model.SetInput(
        /*shape=*/RandomVector<int64_t>(bitgen, rank, /*min=*/1, /*max=*/3),
        bitgen, /*min=*/-5, /*max=*/5);
    model.SetInteriorPadding(
        RandomVector<int64_t>(bitgen, rank, /*min=*/0, /*max=*/2));
    model.SetEdgePadding(
        RandomVector<int64_t>(bitgen, rank, /*min=*/-5, /*max=*/5),
        RandomVector<int64_t>(bitgen, rank, /*min=*/-5, /*max=*/5));
    model.SetPaddingValue(
        absl::Uniform(absl::IntervalClosed, bitgen, -127, 127));

    const auto expected_status = ComputeReference(model);
    ASSERT_TRUE(expected_status.ok());
    const auto& expected = expected_status.value();
    ASSERT_TRUE(model.BuildAndInvoke().ok());
    EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(expected.shape))
        << model;
    EXPECT_THAT(model.GetOutputData(), ElementsAreArray(expected.data))
        << model;
  }
}

}  // namespace
}  // namespace stablehlo_pad
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
