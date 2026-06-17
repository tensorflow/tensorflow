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
#include "tensorflow/lite/profiling/subgraph_tensor_profiler.h"

#include <functional>
#include <string>
#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"

namespace tflite::profiling {
namespace {

using ::testing::IsSupersetOf;
using ::testing::Not;

constexpr const char* kIfSubgraphTensorNames[] = {
    "if_cond",
    "if_input2",
    "if_input3",
    "if_output1",
};

constexpr const char* kAddSubgraphTensorNames[] = {
    "add_input1",
    "add_input2",
    "add_output1",
};

constexpr const char* kMulSubgraphTensorNames[] = {
    "mul_input1",
    "mul_input2",
    "mul_output1",
};

// A functor which captures all tensor names to a set for later consumption.
struct TensorGatherer {
  void operator()(const TfLiteTensor* tensor) { tensors.insert(tensor->name); }

  std::unordered_set<std::string> tensors;
};

// A simple test that performs `ADD` if condition is true, and `MUL` otherwise.
// The computation is: `cond ? a + b : a * b`.
class SubgraphTensorProfilerTest
    : public subgraph_test_util::ControlFlowOpTest {
 protected:
  void SetUp() override {
    AddSubgraphs(2);
    builder_->BuildAddSubgraph(interpreter_->subgraph(1));
    builder_->BuildMulSubgraph(interpreter_->subgraph(2));
    builder_->BuildIfSubgraph(&interpreter_->primary_subgraph());

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1, 2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    subgraph_test_util::FillIntTensor(
        interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});
    subgraph_test_util::FillIntTensor(
        interpreter_->tensor(interpreter_->inputs()[2]), {1, 2});

    NameTensors();
  }

 private:
  // Assign a non-null name to all tensors in every subgraph such that they can
  // be uniquely identified later on.
  void NameTensors() {
    auto set_names = [](Subgraph* subgraph, auto names) {
      for (int j = 0; j < subgraph->tensors_size(); ++j) {
        subgraph->tensor(j)->name = names[j];
      }
    };

    set_names(interpreter_->subgraph(0), kIfSubgraphTensorNames);
    set_names(interpreter_->subgraph(1), kAddSubgraphTensorNames);
    set_names(interpreter_->subgraph(2), kMulSubgraphTensorNames);
  }
};

TEST_F(SubgraphTensorProfilerTest, TestMulSubgraph) {
  TensorGatherer tensor_gatherer;
  tflite::profiling::SubgraphTensorProfiler profiler(*interpreter_,
                                                     std::ref(tensor_gatherer));

  interpreter_->AddProfiler(&profiler);
  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  // Ensure that only subgraphs that were invoked by the interpreter had their
  // tensors captured.
  EXPECT_THAT(tensor_gatherer.tensors, IsSupersetOf(kIfSubgraphTensorNames));
  EXPECT_THAT(tensor_gatherer.tensors, IsSupersetOf(kMulSubgraphTensorNames));
  EXPECT_THAT(tensor_gatherer.tensors,
              Not(IsSupersetOf(kAddSubgraphTensorNames)));
}

TEST_F(SubgraphTensorProfilerTest, TestAddSubgraph) {
  TensorGatherer tensor_gatherer;
  tflite::profiling::SubgraphTensorProfiler profiler(*interpreter_,
                                                     std::ref(tensor_gatherer));

  interpreter_->AddProfiler(&profiler);
  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  // Ensure that only subgraphs that were invoked by the interpreter had their
  // tensors captured.
  EXPECT_THAT(tensor_gatherer.tensors, IsSupersetOf(kIfSubgraphTensorNames));
  EXPECT_THAT(tensor_gatherer.tensors, IsSupersetOf(kAddSubgraphTensorNames));
  EXPECT_THAT(tensor_gatherer.tensors,
              Not(IsSupersetOf(kMulSubgraphTensorNames)));
}

TEST_F(SubgraphTensorProfilerTest, TestBeginEvent) {
  TensorGatherer tensor_gatherer;
  tflite::profiling::SubgraphTensorProfiler profiler(*interpreter_,
                                                     std::ref(tensor_gatherer));
  const int subgraph_id = 1;
  uint32_t valid_event = profiler.BeginEvent(
      "Invoke", Profiler::EventType::DEFAULT, 0, subgraph_id);
  EXPECT_EQ(valid_event, 1);

  uint32_t invalid_event = profiler.BeginEvent(
      "NotInvoke", Profiler::EventType::DEFAULT, 0, subgraph_id);
  EXPECT_EQ(invalid_event, 0);
}

}  // namespace
}  // namespace tflite::profiling
