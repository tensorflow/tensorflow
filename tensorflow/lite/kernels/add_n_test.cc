/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/add_n_test_common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

// Builds the interpreter without allocating tensors (allocate_and_delegate =
// false) so AllocateTensors() can be called explicitly and its failure
// observed, instead of CHECK-failing inside BuildInterpreter().
class DynamicAddNOpModel : public SingleOpModel {
 public:
  DynamicAddNOpModel(const std::vector<TensorData>& inputs,
                     const TensorData& output) {
    int num_inputs = inputs.size();
    std::vector<std::vector<int>> input_shapes;
    for (int i = 0; i < num_inputs; ++i) {
      inputs_.push_back(AddInput(inputs[i]));
      input_shapes.push_back(GetShape(inputs_[i]));
    }
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD_N, BuiltinOptions_AddNOptions,
                 CreateAddNOptions(builder_).Union());
    BuildInterpreter(input_shapes, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false,
                     /*allocate_and_delegate=*/false);
  }

 private:
  std::vector<int> inputs_;
  int output_;
};

TEST(FloatAddNOpModel, AddMultipleTensors) {
  FloatAddNOpModel m({{TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}}},
                     {TensorType_FLOAT32, {}});
  m.PopulateTensor<float>(m.input(0), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input(1), {0.1, 0.2, 0.3, 0.5});
  m.PopulateTensor<float>(m.input(2), {0.5, 0.1, 0.1, 0.2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              Pointwise(FloatingPointEq(), {-1.4, 0.5, 1.1, 1.5}));
}

TEST(FloatAddNOpModel, Add2Tensors) {
  FloatAddNOpModel m(
      {{TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}}},
      {TensorType_FLOAT32, {}});
  m.PopulateTensor<float>(m.input(0), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input(1), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              Pointwise(FloatingPointEq(), {-1.9, 0.4, 1.0, 1.3}));
}

TEST(IntegerAddNOpModel, AddMultipleTensors) {
  IntegerAddNOpModel m({{TensorType_INT32, {1, 2, 2, 1}},
                        {TensorType_INT32, {1, 2, 2, 1}},
                        {TensorType_INT32, {1, 2, 2, 1}}},
                       {TensorType_INT32, {}});
  m.PopulateTensor<int32_t>(m.input(0), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input(1), {1, 2, 3, 5});
  m.PopulateTensor<int32_t>(m.input(2), {10, -5, 1, -2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-9, -1, 11, 11}));
}

TEST(IntegerAddNOpModel, OverflowScratchBuffer) {
  // 2 * 1,073,741,825 = 2,147,483,650 > INT32_MAX, so the input element count
  // overflows int32_t (independent of the host thread count) and Prepare() must
  // reject the shape rather than size an undersized scratch buffer.
  // allocate_and_delegate=false lets us observe the AllocateTensors() failure
  // instead of BuildInterpreter() CHECK-failing, and Prepare() returns the
  // error before any large buffer is allocated.
  DynamicAddNOpModel m({{TensorType_INT32, {2, 1073741825}},
                        {TensorType_INT32, {2, 1073741825}}},
                       {TensorType_INT32, {}});
  EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
}

}  // namespace
}  // namespace tflite
