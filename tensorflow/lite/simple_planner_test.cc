/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/simple_planner.h"

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/graph_info.h"

namespace tflite {
namespace {

// A simple op to be used in tests, as syntactic sugar.
class TestOp {
 public:
  TestOp(std::initializer_list<int> inputs, std::initializer_list<int> outputs,
         std::initializer_list<int> temporaries)
      : inputs_(inputs), outputs_(outputs), temporaries_(temporaries) {}

  const std::vector<int>& inputs() const { return inputs_; }
  const std::vector<int>& outputs() const { return outputs_; }
  const std::vector<int>& temporaries() const { return temporaries_; }
  const TfLiteRegistration& registration() const { return registration_; }

 private:
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<int> temporaries_;
  TfLiteRegistration registration_{};
};

// A test graph where inputs are processed by the given nodes to produce
// outputs.
class TestGraph {
 public:
  TestGraph(std::initializer_list<int> inputs,
            std::initializer_list<TestOp> nodes,
            std::initializer_list<int> outputs)
      : inputs_(inputs), outputs_(outputs) {
    int max_tensor_index = 0;

    for (int t : inputs) {
      max_tensor_index = std::max(max_tensor_index, t);
    }
    for (int t : outputs) {
      max_tensor_index = std::max(max_tensor_index, t);
    }
    for (const auto& node : nodes) {
      auto int_array = [](const std::vector<int>& x) {
        TfLiteIntArray* lite = TfLiteIntArrayCreate(x.size());
        for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];
        return lite;
      };

      registrations_.push_back(node.registration());
      nodes_.push_back(TfLiteNode());
      nodes_.back().inputs = int_array(node.inputs());
      for (int t : node.inputs()) {
        max_tensor_index = std::max(max_tensor_index, t);
      }
      nodes_.back().outputs = int_array(node.outputs());
      for (int t : node.outputs()) {
        max_tensor_index = std::max(max_tensor_index, t);
      }
      nodes_.back().temporaries = int_array(node.temporaries());
      for (int t : node.temporaries()) {
        max_tensor_index = std::max(max_tensor_index, t);
      }
    }

    for (int i = 0; i <= max_tensor_index; ++i) {
      tensors_.push_back(TfLiteTensor());
      // Set some default values for allocation_type and bytes, which are the
      // only fields used by the arena planner.
      tensors_.back().allocation_type = kTfLiteArenaRw;
      tensors_.back().bytes = (i + 1) * 3;
    }
  }

  ~TestGraph() {
    for (auto node : nodes_) {
      TfLiteIntArrayFree(node.inputs);
      TfLiteIntArrayFree(node.outputs);
      TfLiteIntArrayFree(node.temporaries);
    }
  }

  const std::vector<TfLiteNode>& nodes() { return nodes_; }
  std::vector<TfLiteTensor>* tensors() { return &tensors_; }
  const std::vector<int>& inputs() { return inputs_; }
  const std::vector<int>& outputs() { return outputs_; }
  const std::vector<int>& variables() { return variables_; }
  const std::vector<TfLiteRegistration>& registrations() {
    return registrations_;
  }

  void SetVariables(const std::vector<int>& variables) {
    variables_ = variables;
  }

  void Swap(TestGraph* other) {
    std::swap(nodes_, other->nodes_);
    std::swap(tensors_, other->tensors_);
    std::swap(inputs_, other->inputs_);
    std::swap(outputs_, other->outputs_);
    std::swap(variables_, other->variables_);
  }

 private:
  std::vector<TfLiteNode> nodes_;
  std::vector<TfLiteTensor> tensors_;
  std::vector<TfLiteRegistration> registrations_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<int> variables_;
};

// The GraphInfo for a TestGraph.
class TestGraphInfo : public GraphInfo {
 public:
  explicit TestGraphInfo(TestGraph* graph) : graph_(graph) {}

  size_t num_tensors() const override { return graph_->tensors()->size(); }
  const TfLiteRegistration& registration(size_t index) const override {
    return graph_->registrations()[index];
  }
  TfLiteTensor* tensor(size_t index) override {
    return &graph_->tensors()->at(index);
  }
  TfLiteTensor* tensors() override { return graph_->tensors()->data(); }
  size_t num_execution_nodes() const override { return graph_->nodes().size(); }
  size_t num_total_nodes() const override { return graph_->nodes().size(); }
  const TfLiteNode& node(size_t index) const override {
    return graph_->nodes()[index];
  }
  size_t node_index(size_t index) const override { return index; }
  const std::vector<int>& inputs() const override { return graph_->inputs(); }
  const std::vector<int>& outputs() const override { return graph_->outputs(); }
  const std::vector<int>& variables() const override {
    return graph_->variables();
  }

 private:
  TestGraph* graph_;
};

void ReportError(TfLiteContext* context, const char* format, ...) {
  const size_t kBufferSize = 1024;
  char temp_buffer[kBufferSize];

  va_list args;
  va_start(args, format);
  vsnprintf(temp_buffer, kBufferSize, format, args);
  va_end(args);

  LOG(INFO) << temp_buffer;
}

class SimplePlannerTest : public ::testing::Test {
 protected:
  void SetGraph(TestGraph* graph, bool preserve_all_tensors = false) {
    graph_ = graph;
    context_.ReportError = ReportError;
    planner_ = std::make_unique<SimplePlanner>(
        &context_, std::unique_ptr<GraphInfo>(new TestGraphInfo(graph)));
    CHECK(planner_->ResetAllocations() == kTfLiteOk);
    CHECK(planner_->PlanAllocations() == kTfLiteOk);
  }

  void SwapGraph(TestGraph* graph) {
    graph_->Swap(graph);
    CHECK(planner_->PlanAllocations() == kTfLiteOk);
  }

  void Execute(int start, int end) {
    CHECK(planner_->ExecuteAllocations(start, end) == kTfLiteOk);
  }

  void ReleaseNonPersistentMemory() {
    CHECK(planner_->ReleaseNonPersistentMemory() == kTfLiteOk);
  }

  void AcquireNonPersistentMemory() {
    CHECK(planner_->AcquireNonPersistentMemory() == kTfLiteOk);
  }

  void ResetAllocationsAfter(int node) {
    CHECK(planner_->ResetAllocationsAfter(node) == kTfLiteOk);
  }

  bool HasNonPersistentMemory() {
    return planner_ && planner_->HasNonPersistentMemory();
  }

  // Returns if the given tensor is allocated or not.
  bool IsAllocated(int tensor_index) {
    return (*graph_->tensors())[tensor_index].data.raw != nullptr;
  }

  TfLiteContext context_;
  TestGraph* graph_;
  std::unique_ptr<SimplePlanner> planner_;
};

TEST_F(SimplePlannerTest, EmptyGraph) {
  TestGraph graph({}, {}, {});
  SetGraph(&graph);
  Execute(0, 10);
}

TEST_F(SimplePlannerTest, GraphWithNoOps) {
  TestGraph graph({0, 10}, {}, {5, 11});
  SetGraph(&graph);
  Execute(0, 10);
  // The outputs are never allocated because they are not connected to any
  // inputs.
  EXPECT_FALSE(IsAllocated(5));
  EXPECT_FALSE(IsAllocated(11));
}

TEST_F(SimplePlannerTest, ZeroSizedTensors) {
  TestGraph graph({1}, {{{1}, {2}, {}}}, {2});
  (*graph.tensors())[1].bytes = 0;
  SetGraph(&graph);
  ASSERT_EQ(planner_->ExecuteAllocations(0, 10), kTfLiteOk);
  EXPECT_FALSE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
}

TEST_F(SimplePlannerTest, SimpleGraph) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, 10);

  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_TRUE(IsAllocated(3));
  EXPECT_TRUE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));
}

TEST_F(SimplePlannerTest, SimpleGraphInputsPreserved) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, 10);

  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_TRUE(IsAllocated(3));
  EXPECT_TRUE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));
}

TEST_F(SimplePlannerTest, SimpleGraphWithTemporary) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with temporary
                      {{4}, {3}, {}}       // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, 10);

  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_TRUE(IsAllocated(3));
  EXPECT_TRUE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));
}

TEST_F(SimplePlannerTest, SimpleGraphWithResetAllocationsAfter) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with temporary
                      {{4}, {3}, {}}       // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, 10);

  EXPECT_TRUE(IsAllocated(2));
  EXPECT_TRUE(IsAllocated(3));
  EXPECT_TRUE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));
  // Reset allocations after the first node
  ResetAllocationsAfter(0);

  EXPECT_TRUE(IsAllocated(0));
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_FALSE(IsAllocated(3));
  EXPECT_FALSE(IsAllocated(4));
  EXPECT_FALSE(IsAllocated(5));
}

TEST_F(SimplePlannerTest, SimpleGraphWithPersistentResetAllocationsAfter) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with temporary
                      {{4}, {3}, {}}       // Third op
                  },
                  {3});
  // Make the tensor #5 persistent.
  (*graph.tensors())[5].allocation_type = kTfLiteArenaRwPersistent;
  SetGraph(&graph);
  Execute(0, 10);

  // Save the pointer of the persistent temporary tensor #5.
  void* tensor5_ptr = (*graph.tensors())[5].data.raw;

  // Reset allocations after the first node
  ResetAllocationsAfter(0);

  EXPECT_TRUE(IsAllocated(0));
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_FALSE(IsAllocated(3));
  EXPECT_FALSE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));

  // Second run
  Execute(0, 10);

  // Check if the persistent pointer isn't changed.
  EXPECT_TRUE(tensor5_ptr == (*graph.tensors())[5].data.raw);
}

TEST_F(SimplePlannerTest, SimpleGraphOptionalOutput) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op
                  },
                  {-1, 3});
  SetGraph(&graph);
  Execute(0, 10);

  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
  EXPECT_TRUE(IsAllocated(3));
  EXPECT_TRUE(IsAllocated(4));
  EXPECT_TRUE(IsAllocated(5));
}

}  // namespace
}  // namespace tflite
