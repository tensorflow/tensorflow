/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

class TransposeOpModel : public SingleOpModelWithHexagon {
 public:
  TransposeOpModel(const TensorData& input,
                   std::initializer_list<int> perm_shape,
                   std::initializer_list<int> perm, bool const_perm,
                   const TensorData& output) {
    input_ = AddInput(input);
    if (const_perm) {
      perm_ = AddConstInput(TensorType_INT32, perm, perm_shape);
    } else {
      perm_ = AddInput({TensorType_INT32, perm_shape});
    }
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
    if (!const_perm) {
      PopulateTensor<int32_t>(perm_, perm);
    }
  }

  void SetInput(const std::vector<uint8_t>& data) {
    PopulateTensor<uint8_t>(input_, data);
  }

  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }

 protected:
  int input_;
  int perm_;
  int output_;
};

void ComputeExpectedTransposeResult(const std::vector<int>& shape,
                                    const std::vector<int>& perms,
                                    std::vector<uint8_t>* input,
                                    std::vector<uint8_t>* input_transposed) {
  // Count elements and allocate output.
  int count = 1;
  for (auto factor : shape) count *= factor;
  input_transposed->resize(count);

  // Create the dummy data
  (*input).resize(count);
  for (int i = 0; i < count; i++) {
    (*input)[i] = i;
  }

  // Make input and output shapes.
  const RuntimeShape input_shape = ::tflite::GetTensorShape(shape);
  RuntimeShape output_shape(perms.size());
  for (int i = 0; i < perms.size(); i++) {
    output_shape.SetDim(i, input_shape.Dims(perms[i]));
  }

  TransposeParams params;
  params.perm_count = perms.size();
  for (int i = 0; i < perms.size(); ++i) {
    params.perm[i] = perms[i];
  }

  reference_ops::Transpose<uint8_t>(params, input_shape, input->data(),
                                    output_shape, input_transposed->data());
}

TEST(TransposeOpTest, Test1D) {
  // Basic 1D identity.
  std::vector<uint8_t> expected_output, input;
  std::vector<int> input_shape = {3};
  ComputeExpectedTransposeResult(input_shape, {0}, &input, &expected_output);

  TransposeOpModel model({TensorType_UINT8, input_shape, -10, 10}, {1}, {0},
                         true, {TensorType_UINT8, {}, -10, 10});
  model.SetInput(input);
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_output));
}

TEST(TransposeOpTest, Test2D) {
  std::vector<uint8_t> expected_output, input;
  std::vector<int> input_shape = {3, 2};
  std::vector<int> perm = {1, 0};
  ComputeExpectedTransposeResult(input_shape, perm, &input, &expected_output);

  TransposeOpModel model({TensorType_UINT8, input_shape, -10, 10}, {2}, {1, 0},
                         true, {TensorType_UINT8, {}, -10, 10});
  model.SetInput(input);
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_output));
}

TEST(TransposeOpTest, Test4D) {
  std::vector<uint8_t> expected_output, input;
  std::vector<int> input_shape = {2, 2, 3, 1};
  std::vector<int> perm = {3, 0, 1, 2};
  ComputeExpectedTransposeResult(input_shape, perm, &input, &expected_output);

  TransposeOpModel model({TensorType_UINT8, input_shape, -10, 10}, {4},
                         {3, 0, 1, 2}, true, {TensorType_UINT8, {}, -10, 10});
  model.SetInput(input);
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_output));
}
}  // namespace tflite
