/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/async/async_subgraph.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/backend_async_kernel_interface.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/interop/attribute_map_internal.h"
#include "tensorflow/lite/core/async/interop/c/types.h"
#include "tensorflow/lite/core/async/task_internal.h"
#include "tensorflow/lite/core/async/testing/mock_async_kernel.h"
#include "tensorflow/lite/core/async/testing/test_backend.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"

using ::testing::_;

namespace tflite {
namespace async {

class AsyncSubgraphTestPeer {
 public:
  explicit AsyncSubgraphTestPeer(AsyncSubgraph* subgraph)
      : subgraph_(subgraph) {}

  bool IsFullyDelegated() const { return subgraph_->IsFullyDelegated(); }

 private:
  AsyncSubgraph* subgraph_;
};

class AsyncSubgraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    kernel_ = std::make_unique<testing::MockAsyncKernel>();
    backend_ = std::make_unique<testing::TestBackend>(kernel_->kernel());

    interpreter_ = std::make_unique<Interpreter>();
    interpreter_->AddTensors(5);
    interpreter_->SetInputs({0, 1});
    interpreter_->SetOutputs({3, 4});
    TfLiteQuantizationParams quant;
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(3, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(4, kTfLiteFloat32, "", {3},
                                               quant);
    TfLiteRegistration* reg = ops::builtin::Register_ADD();
    void* builtin_data_1 = malloc(sizeof(int));
    void* builtin_data_2 = malloc(sizeof(int));
    void* builtin_data_3 = malloc(sizeof(int));
    interpreter_->AddNodeWithParameters({0, 0}, {2}, nullptr, 0, builtin_data_1,
                                        reg);
    interpreter_->AddNodeWithParameters({1, 1}, {3}, nullptr, 0, builtin_data_2,
                                        reg);
    interpreter_->AddNodeWithParameters({2, 1}, {4}, nullptr, 0, builtin_data_3,
                                        reg);
  }

  void BuildAsyncSubgraph() {
    interpreter_->ModifyGraphWithDelegate(backend_->get_delegate());
    subgraph_ = std::make_unique<AsyncSubgraph>(interpreter_->subgraph(0));
  }

  void TearDown() override { subgraph_.reset(); }

 protected:
  std::unique_ptr<testing::MockAsyncKernel> kernel_;
  std::unique_ptr<testing::TestBackend> backend_;
  std::unique_ptr<Interpreter> interpreter_;
  std::unique_ptr<AsyncSubgraph> subgraph_;
};

TEST_F(AsyncSubgraphTest, FullyDelegated) {
  BuildAsyncSubgraph();
  EXPECT_TRUE(AsyncSubgraphTestPeer(subgraph_.get()).IsFullyDelegated());
}

TEST_F(AsyncSubgraphTest, NotFullyDelegated) {
  // Don't do delegation.
  backend_->SetMinPartitionedNodes(42);
  BuildAsyncSubgraph();
  EXPECT_FALSE(AsyncSubgraphTestPeer(subgraph_.get()).IsFullyDelegated());
}

TEST_F(AsyncSubgraphTest, BasicTest) {
  BuildAsyncSubgraph();

  EXPECT_CALL(*kernel_, RegisterBuffer(_, _, _, _, _));
  EXPECT_CALL(*kernel_, RegisterBufferSlice(_, _, _, _));
  EXPECT_CALL(*kernel_, UnregisterBuffer(_, _));
  EXPECT_CALL(*kernel_, ReconcileRestrictions(_, _, _, _, _, _));
  EXPECT_CALL(*kernel_, SetAttributes(_, _, _, _));
  EXPECT_CALL(*kernel_, Prepare(_, _));
  EXPECT_CALL(*kernel_, Eval(_, _, _));
  EXPECT_CALL(*kernel_, Wait(_, _));
  EXPECT_CALL(*kernel_, Finish(_, _));

  auto* buffer = TfLiteBackendBufferCreate();
  auto* attrs = new TfLiteAttributeMap(kTfLiteBufferAttrMap);
  TfLiteBufferHandle handle = 1;
  TfLiteBufferHandle another_handle = 1;
  auto* task = new TfLiteExecutionTask;
  EXPECT_FALSE(task->task->Scheduled());

  subgraph_->RegisterBuffer(kTfLiteIoTypeInput, buffer, attrs, &handle);
  subgraph_->RegisterBufferSlice(handle, attrs, &another_handle);
  subgraph_->UnregisterBuffer(handle);
  subgraph_->ReconcileRestrictions(0, attrs, attrs, attrs);
  subgraph_->SetAttributes(0, attrs);
  subgraph_->Prepare();
  EXPECT_EQ(kTfLiteOk, subgraph_->InvokeAsync(task));
  EXPECT_TRUE(task->task->Scheduled());
  // Scheduling another execution w/o waiting on the task should return error.
  EXPECT_EQ(kTfLiteError, subgraph_->InvokeAsync(task));
  EXPECT_TRUE(task->task->Scheduled());
  EXPECT_EQ(kTfLiteOk, task->task->Status());
  EXPECT_EQ(kTfLiteOk, subgraph_->Wait(task));

  // If waiting the task failed, all successive `Wait` should also fail.
  task->task->SetStatus(kTfLiteError);
  EXPECT_EQ(kTfLiteError, subgraph_->Wait(task));
  EXPECT_EQ(kTfLiteError, subgraph_->Wait(task));

  EXPECT_FALSE(task->task->Scheduled());
  // Deletes `task`
  subgraph_->Finish(task);

  TfLiteBackendBufferDelete(buffer);
  delete attrs;

  EXPECT_NE(handle, another_handle);
}

}  // namespace async
}  // namespace tflite
