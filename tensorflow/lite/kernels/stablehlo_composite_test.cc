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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"

using testing::ElementsAreArray;
using testing::FloatEq;
using testing::FloatNear;
using testing::Pointwise;

namespace tflite {
namespace {

void AddStablehloCompositeNode(Subgraph* subgraph, const char* name,
                               const std::vector<int>& inputs,
                               const std::vector<int>& outputs,
                               const std::vector<uint8_t>& attributes = {}) {
  TfLiteStablehloCompositeParams* params =
      reinterpret_cast<TfLiteStablehloCompositeParams*>(
          std::malloc(sizeof(TfLiteStablehloCompositeParams)));
  params->name = name;
  params->subgraph_index = 0;
  params->attributes = attributes.empty() ? nullptr : attributes.data();
  params->attributes_size = attributes.size();
  params->version = 1;

  auto* composite_reg = ops::builtin::Register_STABLEHLO_COMPOSITE();
  composite_reg->builtin_code = kTfLiteBuiltinStablehloComposite;

  int node_index;
  ASSERT_EQ(subgraph->AddNodeWithParameters(inputs, outputs, {}, nullptr, 0,
                                            params, composite_reg, &node_index),
            kTfLiteOk);
}

template <typename T>
void AssignTensor(absl::Span<T> tensor, std::initializer_list<T> data) {
  ASSERT_EQ(tensor.size(), data.size());
  std::copy(data.begin(), data.end(), tensor.begin());
}

void SetTensorQuantization(Subgraph* subgraph, int tensor_index, float scale,
                           int zero_point = 0) {
  TfLiteTensor* tensor = subgraph->tensor(tensor_index);
  tensor->params.scale = scale;
  tensor->params.zero_point = zero_point;
}

class CompositeTest : public subgraph_test_util::ControlFlowOpTest {
 protected:
  template <class IndirectionVector>
  TfLiteTensor* GetTensorWithIndirection(int id,
                                         const IndirectionVector& tensor_map) {
    return interpreter_->tensor(tensor_map[id]);
  }

  TfLiteTensor* GetInputTensor(int id) {
    return GetTensorWithIndirection(id, interpreter_->inputs());
  }

  TfLiteTensor* GetOutputTensor(int id) {
    return GetTensorWithIndirection(id, interpreter_->outputs());
  }

  template <class T, class IndirectionVector>
  absl::Span<T> GetTensorDataWithIndirection(
      int id, const IndirectionVector& tensor_map) {
    TfLiteTensor* const tensor = GetTensorWithIndirection(id, tensor_map);
    const size_t size = NumElements(tensor);
    return absl::Span<T>(GetTensorData<T>(tensor), size);
  }

  template <class T>
  absl::Span<T> GetInputData(int id) {
    return GetTensorDataWithIndirection<T>(id, interpreter_->inputs());
  }

  template <class T>
  absl::Span<T> GetOutputData(int id) {
    return GetTensorDataWithIndirection<T>(id, interpreter_->outputs());
  }

  void RunRecurrentLinearAttentionChunkedPrefill(int chunk_size,
                                                 std::vector<float>* output,
                                                 std::vector<float>* state) {
    constexpr int kQuery = 0;
    constexpr int kKey = 1;
    constexpr int kValue = 2;
    constexpr int kPastState = 3;
    constexpr int kDecay = 4;
    constexpr int kBeta = 5;
    constexpr int kOutput = 6;
    constexpr int kPresentState = 7;

    interpreter_ = std::make_unique<Interpreter>();

    flexbuffers::Builder fbb;
    size_t map = fbb.StartMap();
    fbb.Int("q_num_heads", 2);
    fbb.Int("kv_num_heads", 1);
    fbb.Int("chunk_size", chunk_size);
    fbb.Float("scale", 1.0f);
    fbb.String("update_rule", "gated_delta");
    fbb.EndMap(map);
    fbb.Finish();

    Subgraph& subgraph = interpreter_->primary_subgraph();
    int first_new_tensor_index;
    ASSERT_EQ(subgraph.AddTensors(8, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(first_new_tensor_index, 0);
    ASSERT_EQ(
        subgraph.SetInputs({kQuery, kKey, kValue, kPastState, kDecay, kBeta}),
        kTfLiteOk);
    ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
    for (int i = 0; i < 8; ++i) {
      subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
    }
    AddStablehloCompositeNode(&subgraph, "odml.recurrent_linear_attention",
                              {kQuery, kKey, kValue, kPastState, kDecay, kBeta},
                              {kOutput, kPresentState}, fbb.GetBuffer());

    ASSERT_EQ(interpreter_->ResizeInputTensor(kQuery, {1, 3, 2, 1}), kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(kKey, {1, 3, 1, 1}), kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(kValue, {1, 3, 1, 1}), kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 1, 1, 1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(kDecay, {1, 3, 1}), kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(kBeta, {1, 3, 1}), kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    AssignTensor(GetInputData<float>(0), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    AssignTensor(GetInputData<float>(1), {1.0f, 2.0f, 3.0f});
    AssignTensor(GetInputData<float>(2), {10.0f, 20.0f, 30.0f});
    AssignTensor(GetInputData<float>(3), {0.0f});
    AssignTensor(GetInputData<float>(4), {0.0f, 0.0f, 0.0f});
    AssignTensor(GetInputData<float>(5), {1.0f, 1.0f, 1.0f});

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

    absl::Span<float> output_data = GetOutputData<float>(0);
    absl::Span<float> state_data = GetOutputData<float>(1);
    output->assign(output_data.begin(), output_data.end());
    state->assign(state_data.begin(), state_data.end());
  }
};

TEST_F(CompositeTest, TestInvokeWorks) {
  AddSubgraphs(1);
  builder_->BuildAddSubgraph(interpreter_->subgraph(1));
  builder_->BuildCompositeSubgraph(&interpreter_->primary_subgraph(),
                                   interpreter_->subgraph(1));

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {2, 3});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2, 3});

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  subgraph_test_util::FillIntTensor(GetInputTensor(0), {1, 2, 3, 4, 5, 6});
  subgraph_test_util::FillIntTensor(GetInputTensor(1), {7, 8, 9, 10, 11, 12});

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  const TfLiteTensor* const output = GetOutputTensor(0);
  ASSERT_THAT(output, DimsAre({2, 3}));
  EXPECT_THAT(GetOutputData<int>(0), ElementsAreArray({8, 10, 12, 14, 16, 18}));
}

TEST_F(CompositeTest, OdmlCausalConvWithState1dNativeCpu) {
  constexpr int kInput = 0;
  constexpr int kWeight = 1;
  constexpr int kBias = 2;
  constexpr int kPastState = 3;
  constexpr int kOutput = 4;
  constexpr int kPresentState = 5;

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(6, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kInput, kWeight, kBias, kPastState}),
            kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 6; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(&subgraph, "odml.causal_conv_with_state_1d",
                            {kInput, kWeight, kBias, kPastState},
                            {kOutput, kPresentState});

  ASSERT_EQ(interpreter_->ResizeInputTensor(kInput, {1, 3, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kWeight, {3, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kBias, {2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 2, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f});
  AssignTensor(GetInputData<float>(1), {0.1f, 0.01f, 1.0f, 0.1f, 10.0f, 1.0f});
  AssignTensor(GetInputData<float>(2), {0.5f, 5.0f});
  AssignTensor(GetInputData<float>(3), {10.0f, 100.0f, 20.0f, 200.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0),
              Pointwise(FloatEq(), {31.5f, 36.0f, 23.5f, 28.0f, 32.6f, 37.1f}));
  EXPECT_THAT(GetOutputData<float>(1),
              Pointwise(FloatEq(), {2.0f, 20.0f, 3.0f, 30.0f}));
}

TEST_F(CompositeTest, OdmlCausalConvWithState1dSupportsLegacyWeightLayout) {
  constexpr int kInput = 0;
  constexpr int kWeight = 1;
  constexpr int kBias = 2;
  constexpr int kPastState = 3;
  constexpr int kOutput = 4;
  constexpr int kPresentState = 5;

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(6, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kInput, kWeight, kBias, kPastState}),
            kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 6; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(&subgraph, "odml.causal_conv_with_state_1d",
                            {kInput, kWeight, kBias, kPastState},
                            {kOutput, kPresentState});

  ASSERT_EQ(interpreter_->ResizeInputTensor(kInput, {1, 3, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kWeight, {2, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kBias, {2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 2, 3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f});
  AssignTensor(GetInputData<float>(1),
               {0.1f, 1.0f, 10.0f, 100.0f, 0.01f, 0.1f, 1.0f, 10.0f});
  AssignTensor(GetInputData<float>(2), {0.5f, 5.0f});
  AssignTensor(GetInputData<float>(3),
               {10.0f, 20.0f, 30.0f, 100.0f, 200.0f, 300.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(
      GetOutputData<float>(0),
      Pointwise(FloatEq(), {421.5f, 426.0f, 242.5f, 247.0f, 324.5f, 329.0f}));
  EXPECT_THAT(GetOutputData<float>(1),
              Pointwise(FloatEq(), {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f}));
}

TEST_F(CompositeTest, OdmlCausalConvWithState1dAppliesSiluActivation) {
  constexpr int kInput = 0;
  constexpr int kWeight = 1;
  constexpr int kOutput = 2;
  constexpr int kPresentState = 3;

  flexbuffers::Builder fbb;
  size_t map = fbb.StartMap();
  fbb.String("activation", "silu");
  fbb.EndMap(map);
  fbb.Finish();

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(4, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kInput, kWeight}), kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 4; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(&subgraph, "odml.causal_conv_with_state_1d",
                            {kInput, kWeight}, {kOutput, kPresentState},
                            fbb.GetBuffer());

  ASSERT_EQ(interpreter_->ResizeInputTensor(kInput, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kWeight, {1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f});
  AssignTensor(GetInputData<float>(1), {1.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0),
              Pointwise(FloatNear(1e-6f), {0.7310586f}));
  EXPECT_EQ(GetOutputTensor(1)->bytes, 0);
}

TEST_F(CompositeTest, OdmlCausalConvWithState1dSupportsQuantizedIo) {
  constexpr int kInput = 0;
  constexpr int kWeight = 1;
  constexpr int kBias = 2;
  constexpr int kPastState = 3;
  constexpr int kOutput = 4;
  constexpr int kPresentState = 5;

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(6, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kInput, kWeight, kBias, kPastState}),
            kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  subgraph_test_util::SetupTensor(&subgraph, kInput, kTfLiteInt8);
  subgraph_test_util::SetupTensor(&subgraph, kWeight, kTfLiteInt8);
  subgraph_test_util::SetupTensor(&subgraph, kBias, kTfLiteInt32);
  subgraph_test_util::SetupTensor(&subgraph, kPastState, kTfLiteInt8);
  subgraph_test_util::SetupTensor(&subgraph, kOutput, kTfLiteInt8);
  subgraph_test_util::SetupTensor(&subgraph, kPresentState, kTfLiteInt8);
  for (int i = 0; i < 6; ++i) {
    SetTensorQuantization(&subgraph, i, 1.0f);
  }
  AddStablehloCompositeNode(&subgraph, "odml.causal_conv_with_state_1d",
                            {kInput, kWeight, kBias, kPastState},
                            {kOutput, kPresentState});

  ASSERT_EQ(interpreter_->ResizeInputTensor(kInput, {1, 3, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kWeight, {3, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kBias, {1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 2, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor<int8_t>(GetInputData<int8_t>(0), {1, 2, 3});
  AssignTensor<int8_t>(GetInputData<int8_t>(1), {1, 2, 3});
  AssignTensor<int32_t>(GetInputData<int32_t>(2), {1});
  AssignTensor<int8_t>(GetInputData<int8_t>(3), {4, 5});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<int8_t>(0),
              ElementsAreArray(std::vector<int8_t>{18, 14, 15}));
  EXPECT_THAT(GetOutputData<int8_t>(1),
              ElementsAreArray(std::vector<int8_t>{2, 3}));
}

TEST_F(CompositeTest, OdmlRecurrentLinearAttentionNativeCpu) {
  constexpr int kQuery = 0;
  constexpr int kKey = 1;
  constexpr int kValue = 2;
  constexpr int kPastState = 3;
  constexpr int kDecay = 4;
  constexpr int kBeta = 5;
  constexpr int kOutput = 6;
  constexpr int kPresentState = 7;

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(8, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(
      subgraph.SetInputs({kQuery, kKey, kValue, kPastState, kDecay, kBeta}),
      kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 8; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(&subgraph, "odml.recurrent_linear_attention",
                            {kQuery, kKey, kValue, kPastState, kDecay, kBeta},
                            {kOutput, kPresentState});

  ASSERT_EQ(interpreter_->ResizeInputTensor(kQuery, {1, 2, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kKey, {1, 2, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kValue, {1, 2, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 1, 1, 1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDecay, {1, 2, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kBeta, {1, 2, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f, 5.0f});
  AssignTensor(GetInputData<float>(1), {1.0f, 3.0f});
  AssignTensor(GetInputData<float>(2), {2.0f, 4.0f});
  AssignTensor(GetInputData<float>(3), {0.0f});
  AssignTensor(GetInputData<float>(4), {0.0f, 0.0f});
  AssignTensor(GetInputData<float>(5), {1.0f, 1.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0), Pointwise(FloatEq(), {2.0f, -20.0f}));
  EXPECT_THAT(GetOutputData<float>(1), Pointwise(FloatEq(), {-4.0f}));
}

TEST_F(CompositeTest, OdmlRecurrentLinearAttentionSupportsQuantizedIo) {
  constexpr int kQuery = 0;
  constexpr int kKey = 1;
  constexpr int kValue = 2;
  constexpr int kPastState = 3;
  constexpr int kDecay = 4;
  constexpr int kBeta = 5;
  constexpr int kOutput = 6;
  constexpr int kPresentState = 7;

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(8, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(
      subgraph.SetInputs({kQuery, kKey, kValue, kPastState, kDecay, kBeta}),
      kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  subgraph_test_util::SetupTensor(&subgraph, kQuery, kTfLiteInt8);
  subgraph_test_util::SetupTensor(&subgraph, kKey, kTfLiteInt8);
  subgraph_test_util::SetupTensor(&subgraph, kValue, kTfLiteInt8);
  subgraph_test_util::SetupTensor(&subgraph, kPastState, kTfLiteInt8);
  subgraph_test_util::SetupTensor(&subgraph, kDecay, kTfLiteFloat32);
  subgraph_test_util::SetupTensor(&subgraph, kBeta, kTfLiteFloat32);
  subgraph_test_util::SetupTensor(&subgraph, kOutput, kTfLiteInt8);
  subgraph_test_util::SetupTensor(&subgraph, kPresentState, kTfLiteInt8);
  SetTensorQuantization(&subgraph, kQuery, 1.0f);
  SetTensorQuantization(&subgraph, kKey, 1.0f);
  SetTensorQuantization(&subgraph, kValue, 1.0f);
  SetTensorQuantization(&subgraph, kPastState, 1.0f);
  SetTensorQuantization(&subgraph, kOutput, 1.0f);
  SetTensorQuantization(&subgraph, kPresentState, 1.0f);
  AddStablehloCompositeNode(&subgraph, "odml.recurrent_linear_attention",
                            {kQuery, kKey, kValue, kPastState, kDecay, kBeta},
                            {kOutput, kPresentState});

  ASSERT_EQ(interpreter_->ResizeInputTensor(kQuery, {1, 2, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kKey, {1, 2, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kValue, {1, 2, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 1, 1, 1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDecay, {1, 2, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kBeta, {1, 2, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor<int8_t>(GetInputData<int8_t>(0), {1, 5});
  AssignTensor<int8_t>(GetInputData<int8_t>(1), {1, 3});
  AssignTensor<int8_t>(GetInputData<int8_t>(2), {2, 4});
  AssignTensor<int8_t>(GetInputData<int8_t>(3), {0});
  AssignTensor<float>(GetInputData<float>(4), {0.0f, 0.0f});
  AssignTensor<float>(GetInputData<float>(5), {1.0f, 1.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<int8_t>(0),
              ElementsAreArray(std::vector<int8_t>{2, -20}));
  EXPECT_THAT(GetOutputData<int8_t>(1),
              ElementsAreArray(std::vector<int8_t>{-4}));
}

TEST_F(CompositeTest, OdmlRecurrentLinearAttentionUsesHeadAttributes) {
  constexpr int kQuery = 0;
  constexpr int kKey = 1;
  constexpr int kValue = 2;
  constexpr int kPastState = 3;
  constexpr int kDecay = 4;
  constexpr int kBeta = 5;
  constexpr int kOutput = 6;
  constexpr int kPresentState = 7;

  flexbuffers::Builder fbb;
  size_t map = fbb.StartMap();
  fbb.Int("q_num_heads", 2);
  fbb.Int("kv_num_heads", 1);
  fbb.Float("scale", 1.0f);
  fbb.String("update_rule", "gated_delta");
  fbb.EndMap(map);
  fbb.Finish();

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(8, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(
      subgraph.SetInputs({kQuery, kKey, kValue, kPastState, kDecay, kBeta}),
      kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 8; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(&subgraph, "odml.recurrent_linear_attention",
                            {kQuery, kKey, kValue, kPastState, kDecay, kBeta},
                            {kOutput, kPresentState}, fbb.GetBuffer());

  ASSERT_EQ(interpreter_->ResizeInputTensor(kQuery, {1, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kKey, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kValue, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 1, 1, 1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDecay, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kBeta, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f, 5.0f});
  AssignTensor(GetInputData<float>(1), {1.0f});
  AssignTensor(GetInputData<float>(2), {2.0f});
  AssignTensor(GetInputData<float>(3), {0.0f});
  AssignTensor(GetInputData<float>(4), {0.0f});
  AssignTensor(GetInputData<float>(5), {1.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0), Pointwise(FloatEq(), {2.0f, 10.0f}));
  EXPECT_THAT(GetOutputData<float>(1), Pointwise(FloatEq(), {2.0f}));
}

TEST_F(CompositeTest, OdmlRecurrentLinearAttentionAllowsMissingPastState) {
  constexpr int kQuery = 0;
  constexpr int kKey = 1;
  constexpr int kValue = 2;
  constexpr int kDecay = 3;
  constexpr int kBeta = 4;
  constexpr int kOutput = 5;
  constexpr int kPresentState = 6;

  flexbuffers::Builder fbb;
  size_t map = fbb.StartMap();
  fbb.Int("q_num_heads", 1);
  fbb.Int("kv_num_heads", 1);
  fbb.Float("scale", 1.0f);
  fbb.String("update_rule", "gated_delta");
  fbb.EndMap(map);
  fbb.Finish();

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(7, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kQuery, kKey, kValue, kDecay, kBeta}),
            kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 7; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(
      &subgraph, "odml.recurrent_linear_attention",
      {kQuery, kKey, kValue, kTfLiteOptionalTensor, kDecay, kBeta},
      {kOutput, kPresentState}, fbb.GetBuffer());

  ASSERT_EQ(interpreter_->ResizeInputTensor(kQuery, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kKey, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kValue, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDecay, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kBeta, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f});
  AssignTensor(GetInputData<float>(1), {1.0f});
  AssignTensor(GetInputData<float>(2), {2.0f});
  AssignTensor(GetInputData<float>(3), {0.0f});
  AssignTensor(GetInputData<float>(4), {1.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0), Pointwise(FloatEq(), {2.0f}));
  EXPECT_THAT(GetOutputData<float>(1), Pointwise(FloatEq(), {2.0f}));
}

TEST_F(CompositeTest, OdmlRecurrentLinearAttentionAllowsPerKeyDimDecay) {
  constexpr int kQuery = 0;
  constexpr int kKey = 1;
  constexpr int kValue = 2;
  constexpr int kPastState = 3;
  constexpr int kDecay = 4;
  constexpr int kOutput = 5;
  constexpr int kPresentState = 6;

  flexbuffers::Builder fbb;
  size_t map = fbb.StartMap();
  fbb.Int("q_num_heads", 1);
  fbb.Int("kv_num_heads", 1);
  fbb.Float("scale", 1.0f);
  fbb.String("update_rule", "gated");
  fbb.EndMap(map);
  fbb.Finish();

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(7, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kQuery, kKey, kValue, kPastState, kDecay}),
            kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 7; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(&subgraph, "odml.recurrent_linear_attention",
                            {kQuery, kKey, kValue, kPastState, kDecay},
                            {kOutput, kPresentState}, fbb.GetBuffer());

  ASSERT_EQ(interpreter_->ResizeInputTensor(kQuery, {1, 1, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kKey, {1, 1, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kValue, {1, 1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 1, 2, 1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDecay, {1, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f, 1.0f});
  AssignTensor(GetInputData<float>(1), {0.0f, 0.0f});
  AssignTensor(GetInputData<float>(2), {0.0f});
  AssignTensor(GetInputData<float>(3), {2.0f, 3.0f});
  AssignTensor(GetInputData<float>(4), {std::log(2.0f), std::log(3.0f)});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0), Pointwise(FloatEq(), {13.0f}));
  EXPECT_THAT(GetOutputData<float>(1), Pointwise(FloatEq(), {4.0f, 9.0f}));
}

TEST_F(CompositeTest,
       OdmlRecurrentLinearAttentionAllowsFlattenedPerHeadKeyDimDecay) {
  constexpr int kQuery = 0;
  constexpr int kKey = 1;
  constexpr int kValue = 2;
  constexpr int kPastState = 3;
  constexpr int kDecay = 4;
  constexpr int kOutput = 5;
  constexpr int kPresentState = 6;

  flexbuffers::Builder fbb;
  size_t map = fbb.StartMap();
  fbb.Int("q_num_heads", 2);
  fbb.Int("kv_num_heads", 2);
  fbb.Float("scale", 1.0f);
  fbb.String("update_rule", "gated");
  fbb.EndMap(map);
  fbb.Finish();

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(7, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kQuery, kKey, kValue, kPastState, kDecay}),
            kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 7; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(&subgraph, "odml.recurrent_linear_attention",
                            {kQuery, kKey, kValue, kPastState, kDecay},
                            {kOutput, kPresentState}, fbb.GetBuffer());

  ASSERT_EQ(interpreter_->ResizeInputTensor(kQuery, {1, 1, 2, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kKey, {1, 1, 2, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kValue, {1, 1, 2, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 2, 2, 1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDecay, {1, 1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f, 1.0f, 1.0f, 1.0f});
  AssignTensor(GetInputData<float>(1), {0.0f, 0.0f, 0.0f, 0.0f});
  AssignTensor(GetInputData<float>(2), {0.0f, 0.0f});
  AssignTensor(GetInputData<float>(3), {2.0f, 3.0f, 5.0f, 7.0f});
  AssignTensor(GetInputData<float>(4), {std::log(2.0f), std::log(3.0f),
                                        std::log(5.0f), std::log(7.0f)});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0), Pointwise(FloatEq(), {13.0f, 74.0f}));
  EXPECT_THAT(GetOutputData<float>(1),
              Pointwise(FloatEq(), {4.0f, 9.0f, 25.0f, 49.0f}));
}

TEST_F(CompositeTest, OdmlRecurrentLinearAttentionChunkedPrefillShortChunk) {
  std::vector<float> output;
  std::vector<float> state;
  RunRecurrentLinearAttentionChunkedPrefill(/*chunk_size=*/2, &output, &state);

  EXPECT_THAT(output,
              Pointwise(FloatEq(), {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}));
  EXPECT_THAT(state, Pointwise(FloatEq(), {10.0f}));
}

TEST_F(CompositeTest, OdmlRecurrentLinearAttentionChunkedPrefillExactChunk) {
  std::vector<float> output;
  std::vector<float> state;
  RunRecurrentLinearAttentionChunkedPrefill(/*chunk_size=*/3, &output, &state);

  EXPECT_THAT(output,
              Pointwise(FloatEq(), {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}));
  EXPECT_THAT(state, Pointwise(FloatEq(), {10.0f}));
}

TEST_F(CompositeTest, OdmlRecurrentLinearAttentionChunkedPrefillLongChunk) {
  std::vector<float> output;
  std::vector<float> state;
  RunRecurrentLinearAttentionChunkedPrefill(/*chunk_size=*/8, &output, &state);

  EXPECT_THAT(output,
              Pointwise(FloatEq(), {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}));
  EXPECT_THAT(state, Pointwise(FloatEq(), {10.0f}));
}

TEST_F(CompositeTest,
       OdmlRecurrentLinearAttentionChunkedPrefillMatchesReference) {
  constexpr int kQuery = 0;
  constexpr int kKey = 1;
  constexpr int kValue = 2;
  constexpr int kPastState = 3;
  constexpr int kDecay = 4;
  constexpr int kBeta = 5;
  constexpr int kOutput = 6;
  constexpr int kPresentState = 7;

  flexbuffers::Builder fbb;
  size_t map = fbb.StartMap();
  fbb.Int("q_num_heads", 1);
  fbb.Int("kv_num_heads", 1);
  fbb.Int("chunk_size", 2);
  fbb.Bool("use_chunked_prefill", true);
  fbb.Float("scale", 1.0f);
  fbb.String("update_rule", "gated_delta");
  fbb.EndMap(map);
  fbb.Finish();

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(8, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(
      subgraph.SetInputs({kQuery, kKey, kValue, kPastState, kDecay, kBeta}),
      kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 8; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(&subgraph, "odml.recurrent_linear_attention",
                            {kQuery, kKey, kValue, kPastState, kDecay, kBeta},
                            {kOutput, kPresentState}, fbb.GetBuffer());

  ASSERT_EQ(interpreter_->ResizeInputTensor(kQuery, {1, 3, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kKey, {1, 3, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kValue, {1, 3, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 1, 2, 2}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDecay, {1, 3, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kBeta, {1, 3, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0),
               {1.0f, 0.5f, -0.25f, 1.5f, 0.75f, -1.0f});
  AssignTensor(GetInputData<float>(1), {0.6f, -0.2f, 1.0f, 0.5f, -0.4f, 0.8f});
  AssignTensor(GetInputData<float>(2), {0.7f, -0.3f, -0.2f, 0.9f, 0.5f, 0.25f});
  AssignTensor(GetInputData<float>(3), {0.2f, -0.1f, 0.3f, 0.4f});
  AssignTensor(GetInputData<float>(4),
               {std::log(0.5f), std::log(0.25f), std::log(0.75f)});
  AssignTensor(GetInputData<float>(5), {0.5f, 0.75f, 1.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0),
              Pointwise(FloatNear(1e-5f),
                        {0.3425f, -0.0075f, -0.094796875f, 0.429265625f,
                         -0.574537891f, -0.160108203f}));
  EXPECT_THAT(GetOutputData<float>(1),
              Pointwise(FloatNear(1e-5f), {-0.308276563f, 0.401942188f,
                                           0.343330469f, 0.461564844f}));
}

TEST_F(CompositeTest, OdmlSelectiveStateSpaceMamba2StyleNativeCpu) {
  constexpr int kX = 0;
  constexpr int kDelta = 1;
  constexpr int kA = 2;
  constexpr int kB = 3;
  constexpr int kC = 4;
  constexpr int kPastState = 5;
  constexpr int kD = 6;
  constexpr int kDeltaBias = 7;
  constexpr int kOutput = 8;
  constexpr int kPresentState = 9;

  flexbuffers::Builder fbb;
  size_t map = fbb.StartMap();
  fbb.Int("num_groups", 1);
  fbb.EndMap(map);
  fbb.Finish();

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(10, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(
      subgraph.SetInputs({kX, kDelta, kA, kB, kC, kPastState, kD, kDeltaBias}),
      kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 10; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(
      &subgraph, "odml.selective_state_space",
      {kX, kDelta, kA, kB, kC, kPastState, kD, kDeltaBias},
      {kOutput, kPresentState}, fbb.GetBuffer());

  ASSERT_EQ(interpreter_->ResizeInputTensor(kX, {1, 2, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDelta, {1, 2, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kA, {1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kB, {1, 2, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kC, {1, 2, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 1, 2, 2}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kD, {1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDeltaBias, {1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f, 2.0f, 3.0f, 4.0f});
  AssignTensor(GetInputData<float>(1), {1.0f, 2.0f});
  AssignTensor(GetInputData<float>(2), {0.0f});
  AssignTensor(GetInputData<float>(3), {1.0f, 2.0f, 3.0f, 4.0f});
  AssignTensor(GetInputData<float>(4), {1.0f, 1.0f, 1.0f, 1.0f});
  AssignTensor(GetInputData<float>(5), {0.0f, 0.0f, 0.0f, 0.0f});
  AssignTensor(GetInputData<float>(6), {10.0f, 20.0f});
  AssignTensor(GetInputData<float>(7), {0.0f, 0.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0),
              Pointwise(FloatEq(), {13.0f, 46.0f, 75.0f, 142.0f}));
  EXPECT_THAT(GetOutputData<float>(1),
              Pointwise(FloatEq(), {19.0f, 26.0f, 26.0f, 36.0f}));
}

TEST_F(CompositeTest, OdmlSelectiveStateSpaceMambaV1Rank3NativeCpu) {
  constexpr int kX = 0;
  constexpr int kDelta = 1;
  constexpr int kA = 2;
  constexpr int kB = 3;
  constexpr int kC = 4;
  constexpr int kPastState = 5;
  constexpr int kOutput = 6;
  constexpr int kPresentState = 7;

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(8, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kX, kDelta, kA, kB, kC, kPastState}),
            kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 8; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(&subgraph, "odml.selective_state_space",
                            {kX, kDelta, kA, kB, kC, kPastState},
                            {kOutput, kPresentState});

  ASSERT_EQ(interpreter_->ResizeInputTensor(kX, {1, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDelta, {1, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kA, {2, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kB, {1, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kC, {1, 1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 2, 1, 2}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f, 2.0f});
  AssignTensor(GetInputData<float>(1), {1.0f, 1.0f});
  AssignTensor(GetInputData<float>(2), {0.0f, 0.0f, 0.0f, 0.0f});
  AssignTensor(GetInputData<float>(3), {3.0f, 4.0f});
  AssignTensor(GetInputData<float>(4), {5.0f, 6.0f});
  AssignTensor(GetInputData<float>(5), {0.0f, 0.0f, 0.0f, 0.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0), Pointwise(FloatEq(), {39.0f, 78.0f}));
  EXPECT_THAT(GetOutputData<float>(1),
              Pointwise(FloatEq(), {3.0f, 4.0f, 6.0f, 8.0f}));
}

TEST_F(CompositeTest, OdmlSelectiveStateSpaceAppliesDeltaSoftplus) {
  constexpr int kX = 0;
  constexpr int kDelta = 1;
  constexpr int kA = 2;
  constexpr int kB = 3;
  constexpr int kC = 4;
  constexpr int kPastState = 5;
  constexpr int kOutput = 6;
  constexpr int kPresentState = 7;

  flexbuffers::Builder fbb;
  size_t map = fbb.StartMap();
  fbb.String("delta_transform", "softplus");
  fbb.EndMap(map);
  fbb.Finish();

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(8, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kX, kDelta, kA, kB, kC, kPastState}),
            kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 8; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  AddStablehloCompositeNode(&subgraph, "odml.selective_state_space",
                            {kX, kDelta, kA, kB, kC, kPastState},
                            {kOutput, kPresentState}, fbb.GetBuffer());

  ASSERT_EQ(interpreter_->ResizeInputTensor(kX, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDelta, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kA, {1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kB, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kC, {1, 1, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 1, 1, 1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {5.0f});
  AssignTensor(GetInputData<float>(1), {0.0f});
  AssignTensor(GetInputData<float>(2), {0.0f});
  AssignTensor(GetInputData<float>(3), {2.0f});
  AssignTensor(GetInputData<float>(4), {3.0f});
  AssignTensor(GetInputData<float>(5), {0.0f});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0),
              Pointwise(FloatNear(1e-5f), {20.794415f}));
  EXPECT_THAT(GetOutputData<float>(1),
              Pointwise(FloatNear(1e-5f), {6.9314718f}));
}

TEST_F(CompositeTest, OdmlSelectiveStateSpaceSupportsMaskAndReset) {
  constexpr int kX = 0;
  constexpr int kDelta = 1;
  constexpr int kA = 2;
  constexpr int kB = 3;
  constexpr int kC = 4;
  constexpr int kPastState = 5;
  constexpr int kTokenMask = 6;
  constexpr int kResetMask = 7;
  constexpr int kOutput = 8;
  constexpr int kPresentState = 9;

  Subgraph& subgraph = interpreter_->primary_subgraph();
  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(10, &first_new_tensor_index), kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs(
                {kX, kDelta, kA, kB, kC, kPastState, kTokenMask, kResetMask}),
            kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput, kPresentState}), kTfLiteOk);
  for (int i = 0; i < 10; ++i) {
    subgraph_test_util::SetupTensor(&subgraph, i, kTfLiteFloat32);
  }
  subgraph_test_util::SetupTensor(&subgraph, kTokenMask, kTfLiteBool);
  subgraph_test_util::SetupTensor(&subgraph, kResetMask, kTfLiteBool);
  AddStablehloCompositeNode(
      &subgraph, "odml.selective_state_space",
      {kX, kDelta, kA, kB, kC, kPastState, kTfLiteOptionalTensor,
       kTfLiteOptionalTensor, kTokenMask, kResetMask},
      {kOutput, kPresentState});

  ASSERT_EQ(interpreter_->ResizeInputTensor(kX, {1, 3, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kDelta, {1, 3, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kA, {1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kB, {1, 3, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kC, {1, 3, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kPastState, {1, 1, 1, 1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kTokenMask, {1, 3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(kResetMask, {1, 3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignTensor(GetInputData<float>(0), {1.0f, 2.0f, 3.0f});
  AssignTensor(GetInputData<float>(1), {1.0f, 1.0f, 1.0f});
  AssignTensor(GetInputData<float>(2), {0.0f});
  AssignTensor(GetInputData<float>(3), {1.0f, 1.0f, 1.0f});
  AssignTensor(GetInputData<float>(4), {1.0f, 1.0f, 1.0f});
  AssignTensor(GetInputData<float>(5), {5.0f});
  AssignTensor<bool>(GetInputData<bool>(6), {true, false, true});
  AssignTensor<bool>(GetInputData<bool>(7), {false, false, true});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<float>(0),
              Pointwise(FloatEq(), {6.0f, 0.0f, 3.0f}));
  EXPECT_THAT(GetOutputData<float>(1), Pointwise(FloatEq(), {3.0f}));
}

TEST_F(CompositeTest, TestXNNPACKDelegation) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(1);
  builder_->BuildXNNPACKSubgraph(interpreter_->subgraph(1));
  builder_->BuildCompositeSubgraph(&interpreter_->primary_subgraph(),
                                   interpreter_->subgraph(1));

  const auto opt = TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> xnnpack_delegate(
      TfLiteXNNPackDelegateCreate(&opt), TfLiteXNNPackDelegateDelete);
  interpreter_->primary_subgraph().MarkAsDelegationSkippable();
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(std::move(xnnpack_delegate)),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {2, 3}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2, 3}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  absl::Span<float> input0 = GetInputData<float>(0);
  std::iota(input0.begin(), input0.end(), 1.0f);
  absl::Span<float> input1 = GetInputData<float>(1);
  std::iota(input1.begin(), input1.end(), 7.0f);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  const std::vector<float> expected_values = {16, 20, 24, 28, 32, 36};

  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  const absl::Span<float> output0_data(GetTensorData<float>(output0), 6);
  ASSERT_THAT(output0, DimsAre({2, 3}));
  EXPECT_THAT(output0_data, Pointwise(FloatEq(), expected_values));

  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  const absl::Span<float> output1_data(GetTensorData<float>(output1), 6);
  ASSERT_THAT(output1, DimsAre({2, 3}));
  EXPECT_THAT(output1_data, Pointwise(FloatEq(), expected_values));

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

}  // namespace
}  // namespace tflite
