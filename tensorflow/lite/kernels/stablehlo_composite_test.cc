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

#include <cstddef>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"

using testing::ElementsAreArray;
using testing::FloatEq;
using testing::Pointwise;

namespace tflite {
namespace {

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
