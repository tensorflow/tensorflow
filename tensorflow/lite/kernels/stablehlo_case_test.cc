/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
tflite::TensorType GetTTEnum();

// NOLINTBEGIN

template <>
tflite::TensorType GetTTEnum<float>() {
  return tflite::TensorType_FLOAT32;
}

template <>
tflite::TensorType GetTTEnum<int8_t>() {
  return tflite::TensorType_INT8;
}

template <>
tflite::TensorType GetTTEnum<int16_t>() {
  return tflite::TensorType_INT16;
}

template <>
tflite::TensorType GetTTEnum<int32_t>() {
  return tflite::TensorType_INT32;
}

// NOLINTEND

class StablehloCaseOpModel : public SingleOpModel {
 public:
  StablehloCaseOpModel(const TensorData& input, const TensorData& input1,
                       const TensorData& input2, const TensorData& output,
                       const TfLiteStablehloCaseParams& params) {
    InitializeCommonInputs(input, input1, input2, output, params);
  }

  template <typename T>
  void SetInput(int index, std::initializer_list<T> data) {
    PopulateTensor<T>(index, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

  int input() { return input_; }

  int subgraph_input1() { return subgraph_input1_; }

  int subgraph_input2() { return subgraph_input2_; }

 protected:
  void InitializeCommonInputs(const TensorData& input, const TensorData& input1,
                              const TensorData& input2,
                              const TensorData& output,
                              const TfLiteStablehloCaseParams& params) {
    input_ = AddInput(input);
    subgraph_input1_ = AddInput(SymmetricInt16Scaling(input1));
    subgraph_input2_ = AddInput(SymmetricInt16Scaling(input2));
    output_ = AddOutput(SymmetricInt16Scaling(output));
    SetBuiltinOp(BuiltinOperator_STABLEHLO_CASE,
                 BuiltinOptions2_StablehloCaseOptions,
                 CreateStablehloCaseOptions(
                     builder_,
                     builder_.CreateVector(std::vector<int>(
                         params.branch_subgraph_indices,
                         params.branch_subgraph_indices + params.num_branches)))
                     .Union());
    BuildInterpreter({GetShape(input_)}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/false,
                     /*use_simple_allocator=*/false);
    AddSubgraphs(params.num_branches);
  }

  TensorData SymmetricInt16Scaling(TensorData tensor) {
    if (tensor.type == TensorType_INT16) {
      ABSL_CHECK_EQ(std::abs(tensor.min), tensor.max);
      tensor.scale = tensor.max / std::numeric_limits<int16_t>::max();
      tensor.zero_point = 0;
      tensor.min = 0;
      tensor.max = 0;
    }
    return tensor;
  }

  int input_;
  int subgraph_input1_;
  int subgraph_input2_;
  int output_;
  subgraph_test_util::SubgraphBuilder subgraph_builder_;
};

class StablehloCaseStaticOpModel : public StablehloCaseOpModel {
 public:
  StablehloCaseStaticOpModel(const TensorData& input, const TensorData& input1,
                             const TensorData& input2, const TensorData& output,
                             const TfLiteStablehloCaseParams& params)
      : StablehloCaseOpModel(input, input1, input2, output, params) {
    subgraph_builder_.BuildAddSubgraph(interpreter_->subgraph(1));
    subgraph_builder_.BuildMulSubgraph(interpreter_->subgraph(2));
    subgraph_builder_.BuildMaximumSubgraph(interpreter_->subgraph(3));
    subgraph_builder_.BuildMinimumSubgraph(interpreter_->subgraph(4));
    AllocateAndDelegate(true);
  }
  int output() { return output_; }
};

class StablehloCaseDynamicOpModel : public StablehloCaseOpModel {
 public:
  StablehloCaseDynamicOpModel(const TensorData& input, const TensorData& input1,
                              const TensorData& input2,
                              const TensorData& output,
                              const TfLiteStablehloCaseParams& params)
      : StablehloCaseOpModel(input, input1, input2, output, params) {
    subgraph_builder_.BuildAddSubgraph(interpreter_->subgraph(1));
    subgraph_builder_.BuildPadSubgraph(interpreter_->subgraph(2));
    AllocateAndDelegate(true);
  }
  int output() { return output_; }
};

template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep =
      2.0 * (max - min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  return kQuantizedStep;
}

template <typename Float>
class StablehloCaseTestFloat : public ::testing::Test {
 public:
  using FloatType = Float;
};

using FloatTestTypes = ::testing::Types<float>;

TYPED_TEST_SUITE(StablehloCaseTestFloat, FloatTestTypes);

TYPED_TEST(StablehloCaseTestFloat, CaseFloatMul) {
  using Float = typename TestFixture::FloatType;
  TfLiteStablehloCaseParams params = {
      {1, 2, 3, 4},
      4,
  };

  StablehloCaseStaticOpModel model(
      {TensorType_INT32, {}}, {GetTTEnum<Float>(), {1, 2}},
      {GetTTEnum<Float>(), {1, 2}}, {GetTTEnum<Float>(), {1, 2}}, params);
  model.SetInput<int>(model.input(), {1});
  model.SetInput<Float>(model.subgraph_input1(),
                        {static_cast<Float>(5.5), static_cast<Float>(2.5)});
  model.SetInput<Float>(model.subgraph_input2(),
                        {static_cast<Float>(5.5), static_cast<Float>(2.5)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<Float>(),
              Pointwise(FloatingPointEq(), {Float(30.25), Float(6.25)}));
}

TYPED_TEST(StablehloCaseTestFloat, CaseFloatAdd) {
  using Float = typename TestFixture::FloatType;
  TfLiteStablehloCaseParams params = {
      {1, 2, 3, 4},
      4,
  };

  StablehloCaseStaticOpModel model(
      {TensorType_INT32, {}}, {GetTTEnum<Float>(), {1, 2}},
      {GetTTEnum<Float>(), {1, 2}}, {GetTTEnum<Float>(), {1, 2}}, params);
  model.SetInput<int>(model.input(), {0});
  model.SetInput<Float>(model.subgraph_input1(),
                        {static_cast<Float>(5.5), static_cast<Float>(2.4)});
  model.SetInput<Float>(model.subgraph_input2(),
                        {static_cast<Float>(5), static_cast<Float>(2)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<Float>(),
              Pointwise(FloatingPointEq(),
                        {static_cast<Float>(10.5), static_cast<Float>(4.4)}));
}

template <typename Int>
class StablehloCaseTestInt : public ::testing::Test {
 public:
  using IntType = Int;
};

using IntTestTypes = ::testing::Types<int, int16_t>;

TYPED_TEST_SUITE(StablehloCaseTestInt, IntTestTypes);

TYPED_TEST(StablehloCaseTestInt, CaseIntMaximum) {
  using Int = typename TestFixture::IntType;
  TfLiteStablehloCaseParams params = {
      {1, 2, 3, 4},
      4,
  };

  StablehloCaseStaticOpModel model(
      {TensorType_INT32, {}}, {GetTTEnum<Int>(), {1, 2}},
      {GetTTEnum<Int>(), {1, 2}}, {GetTTEnum<Int>(), {1, 2}}, params);
  model.SetInput<int>(model.input(), {2});
  model.SetInput<Int>(model.subgraph_input1(), {5, 20});
  model.SetInput<Int>(model.subgraph_input2(), {15, 2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<Int>(), ElementsAreArray({15, 20}));
}

TYPED_TEST(StablehloCaseTestInt, CaseIntMinimum) {
  using Int = typename TestFixture::IntType;
  TfLiteStablehloCaseParams params = {
      {1, 2, 3, 4},
      4,
  };

  StablehloCaseStaticOpModel model(
      {TensorType_INT32, {}}, {GetTTEnum<Int>(), {1, 2}},
      {GetTTEnum<Int>(), {1, 2}}, {GetTTEnum<Int>(), {1, 2}}, params);
  model.SetInput<int>(
      model.input(),
      {-1});  // when index is out of bounds, case op executes the last branch
  model.SetInput<Int>(model.subgraph_input1(),
                      {static_cast<Int>(5), static_cast<Int>(20)});
  model.SetInput<Int>(model.subgraph_input2(),
                      {static_cast<Int>(15), static_cast<Int>(2)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<Int>(),
              ElementsAreArray({static_cast<Int>(5), static_cast<Int>(2)}));
}

TEST(StablehloCaseTest, CaseQuantizedMul) {
  float kQuantizedTolerance = GetTolerance<int8_t>(-127.0, 127.0);
  TfLiteStablehloCaseParams params = {
      {1, 2, 3, 4},
      4,
  };

  StablehloCaseStaticOpModel model(
      {TensorType_INT32, {}}, {TensorType_INT8, {1, 2}, -127.0f, 127.0f},
      {TensorType_INT8, {1, 2}, -127.0f, 127.0f},
      {TensorType_INT8, {1, 2}, -127.0f, 127.0f}, params);
  model.SetInput<int>(model.input(), {0});
  model.QuantizeAndPopulate<int8_t>(model.subgraph_input1(), {5.0, 2.0});
  model.QuantizeAndPopulate<int8_t>(model.subgraph_input2(), {5.0, 2.0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({10, 4}, kQuantizedTolerance)));
}

TEST(StablehloCaseTest, DynamicCaseTestAdd) {
  TfLiteStablehloCaseParams params = {
      {1, 2},
      2,
  };

  StablehloCaseDynamicOpModel model(
      {TensorType_INT32, {}}, {TensorType_INT32, {2}},
      {TensorType_INT32, {1, 2}}, {TensorType_INT32, {}}, params);
  model.SetInput<int>(model.input(), {0});
  model.SetInput<int>(model.subgraph_input1(), {5, 7});
  model.SetInput<int>(model.subgraph_input2(), {1, 2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_TRUE(IsDynamicTensor(model.GetOutputTensor(0)));
  EXPECT_THAT(model.GetOutput<int>(), ElementsAreArray({6, 9}));
}

TEST(StablehloCaseTest, DynamicCaseTestPad) {
  TfLiteStablehloCaseParams params = {
      {1, 2},
      2,
  };

  StablehloCaseDynamicOpModel model(
      {TensorType_INT32, {}}, {TensorType_INT32, {2}},
      {TensorType_INT32, {1, 2}}, {TensorType_INT32, {}}, params);
  model.SetInput<int>(model.input(),
                      {-1});  // when index value is out of bounds, case op
                              // executes the last branch
  model.SetInput<int>(model.subgraph_input1(), {5, 7});
  model.SetInput<int>(model.subgraph_input2(), {1, 2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_TRUE(IsDynamicTensor(model.GetOutputTensor(0)));
  EXPECT_THAT(model.GetOutput<int>(), ElementsAreArray({0, 5, 7, 0, 0}));
}

}  // namespace
}  // namespace tflite
