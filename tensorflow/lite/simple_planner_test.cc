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
  void SetGraph(TestGraph* graph, bool enable_reclamation = false,
                bool preserve_all = false,
                TfLiteAllocator* allocator = nullptr) {
    graph_ = graph;
    context_.ReportError = ReportError;
    planner_ = std::make_unique<SimplePlanner>(
        &context_, std::unique_ptr<GraphInfo>(new TestGraphInfo(graph)),
        preserve_all, enable_reclamation, allocator);
    CHECK(planner_->ResetAllocations() == kTfLiteOk);
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

TEST_F(SimplePlannerTest, UAFWhenResizedToZero) {
  TestGraph graph({0}, {{{0}, {1}, {}}}, {1});
  SetGraph(&graph);
  Execute(0, 10);
  EXPECT_TRUE(IsAllocated(1));

  // Resize tensor 1 to 0 bytes and re-execute.
  (*graph.tensors())[1].bytes = 0;
  Execute(0, 10);
  EXPECT_FALSE(IsAllocated(1));
}

TEST_F(SimplePlannerTest, OptionalTensorsInOutputsAndTemporaries) {
  TestGraph graph({0}, {{{0}, {1, -1}, {-1, 2}}}, {1});
  SetGraph(&graph);
  Execute(0, 10);
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
}

TEST_F(SimplePlannerTest, NonPersistentMemoryLifecycle) {
  TestGraph graph({0, 1}, {{{0, 1}, {2}, {}}}, {2});
  SetGraph(&graph);
  Execute(0, 10);
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));

  ReleaseNonPersistentMemory();
  EXPECT_FALSE(IsAllocated(1));
  EXPECT_FALSE(IsAllocated(2));

  AcquireNonPersistentMemory();
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));
}

TEST_F(SimplePlannerTest, DeterministicChainEagerVsReclamation) {
  // Chain: X (0) -> op0 -> A (1) -> op1 -> B (2) -> op2 -> C (3) -> op3 -> D
  // (4) -> op4 -> Y (5)
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {}},  // op0: A = op(X)
                      {{1}, {2}, {}},  // op1: B = op(A)
                      {{2}, {3}, {}},  // op2: C = op(B)
                      {{3}, {4}, {}},  // op3: D = op(C)
                      {{4}, {5}, {}},  // op4: Y = op(D)
                  },
                  {5});
  constexpr size_t kTensorBytes = 256;
  for (int i = 0; i <= 5; ++i) {
    (*graph.tensors())[i].bytes = kTensorBytes;
  }

  // Part 1: Eager Mode (Baseline)
  {
    SetGraph(&graph, /*enable_reclamation=*/false);
    Execute(0, 4);

    EXPECT_TRUE(IsAllocated(0));
    EXPECT_TRUE(IsAllocated(1));
    EXPECT_TRUE(IsAllocated(2));
    EXPECT_TRUE(IsAllocated(3));
    EXPECT_TRUE(IsAllocated(4));
    EXPECT_TRUE(IsAllocated(5));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 6 * kTensorBytes);
    EXPECT_EQ(planner_->peak_outstanding_bytes(), 6 * kTensorBytes);
  }

  // Part 2: Reclamation Mode (Deferred Allocation + Last-Use Reclamation)
  {
    SetGraph(&graph, /*enable_reclamation=*/true);
    Execute(0, 4);

    // After preparation, ONLY pinned buffers (X and Y) are allocated.
    // Intermediates A, B, C, D are deferred!
    EXPECT_TRUE(IsAllocated(0));
    EXPECT_FALSE(IsAllocated(1));
    EXPECT_FALSE(IsAllocated(2));
    EXPECT_FALSE(IsAllocated(3));
    EXPECT_FALSE(IsAllocated(4));
    EXPECT_TRUE(IsAllocated(5));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 2 * kTensorBytes);

    // Simulate 1st invocation
    ASSERT_EQ(planner_->BeginInvocation(), kTfLiteOk);

    // op0: before node, A (1) is allocated.
    ASSERT_EQ(planner_->BeforeNode(0), kTfLiteOk);
    EXPECT_TRUE(IsAllocated(1));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 3 * kTensorBytes);
    ASSERT_EQ(planner_->AfterNode(0), kTfLiteOk);
    EXPECT_TRUE(IsAllocated(1));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 3 * kTensorBytes);

    // op1: before node, B (2) is allocated.
    ASSERT_EQ(planner_->BeforeNode(1), kTfLiteOk);
    EXPECT_TRUE(IsAllocated(2));
    EXPECT_EQ(planner_->current_outstanding_bytes(),
              4 * kTensorBytes);  // Peak = 4 * S!
    ASSERT_EQ(planner_->AfterNode(1), kTfLiteOk);
    // A (1) is freed after op1!
    EXPECT_FALSE(IsAllocated(1));
    EXPECT_TRUE(IsAllocated(2));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 3 * kTensorBytes);

    // op2: before node, C (3) is allocated.
    ASSERT_EQ(planner_->BeforeNode(2), kTfLiteOk);
    EXPECT_TRUE(IsAllocated(3));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 4 * kTensorBytes);  // Peak
    ASSERT_EQ(planner_->AfterNode(2), kTfLiteOk);
    EXPECT_FALSE(IsAllocated(2));
    EXPECT_TRUE(IsAllocated(3));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 3 * kTensorBytes);

    // op3: before node, D (4) is allocated.
    ASSERT_EQ(planner_->BeforeNode(3), kTfLiteOk);
    EXPECT_TRUE(IsAllocated(4));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 4 * kTensorBytes);  // Peak
    ASSERT_EQ(planner_->AfterNode(3), kTfLiteOk);
    EXPECT_FALSE(IsAllocated(3));
    EXPECT_TRUE(IsAllocated(4));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 3 * kTensorBytes);

    // op4: before node, Y (5) is already allocated (pinned).
    ASSERT_EQ(planner_->BeforeNode(4), kTfLiteOk);
    EXPECT_TRUE(IsAllocated(5));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 3 * kTensorBytes);
    ASSERT_EQ(planner_->AfterNode(4), kTfLiteOk);
    EXPECT_FALSE(IsAllocated(4));
    EXPECT_TRUE(IsAllocated(5));
    EXPECT_EQ(planner_->current_outstanding_bytes(), 2 * kTensorBytes);

    planner_->EndInvocation(/*completed_successfully=*/true);

    // Verify final stats of 1st invocation:
    EXPECT_EQ(planner_->current_outstanding_bytes(), 2 * kTensorBytes);
    EXPECT_EQ(planner_->peak_outstanding_bytes(), 4 * kTensorBytes);
    EXPECT_EQ(planner_->total_allocations(), 6);
    EXPECT_EQ(planner_->total_deallocations(), 4);

    // Simulate 2nd invocation without calling ExecuteAllocations again:
    ASSERT_EQ(planner_->BeginInvocation(), kTfLiteOk);
    for (int node = 0; node <= 4; ++node) {
      ASSERT_EQ(planner_->BeforeNode(node), kTfLiteOk);
      ASSERT_EQ(planner_->AfterNode(node), kTfLiteOk);
    }
    planner_->EndInvocation(/*completed_successfully=*/true);

    // Memory between invocations is bounded to 2 * kTensorBytes!
    EXPECT_EQ(planner_->current_outstanding_bytes(), 2 * kTensorBytes);
    EXPECT_EQ(planner_->total_allocations(), 10);
    EXPECT_EQ(planner_->total_deallocations(), 8);
  }
}

TEST_F(SimplePlannerTest, FanOutLiveness) {
  // op0: produces A (1)
  // op1: consumes A (1), produces B (2)
  // op2: consumes A (1) and B (2), produces Y (3)
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {}},     // op0: A = op(X)
                      {{1}, {2}, {}},     // op1: B = op(A)
                      {{1, 2}, {3}, {}},  // op2: Y = op(A, B)
                  },
                  {3});
  for (int i = 0; i <= 3; ++i) {
    (*graph.tensors())[i].bytes = 100;
  }

  SetGraph(&graph, /*enable_reclamation=*/true);
  Execute(0, 2);

  ASSERT_EQ(planner_->BeginInvocation(), kTfLiteOk);

  // op0
  ASSERT_EQ(planner_->BeforeNode(0), kTfLiteOk);
  EXPECT_TRUE(IsAllocated(1));
  ASSERT_EQ(planner_->AfterNode(0), kTfLiteOk);
  EXPECT_TRUE(IsAllocated(1));

  // op1
  ASSERT_EQ(planner_->BeforeNode(1), kTfLiteOk);
  EXPECT_TRUE(IsAllocated(2));
  ASSERT_EQ(planner_->AfterNode(1), kTfLiteOk);
  // A (1) MUST NOT be freed after op1, because op2 still consumes A!
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));

  // op2
  ASSERT_EQ(planner_->BeforeNode(2), kTfLiteOk);
  EXPECT_TRUE(IsAllocated(3));  // Y is pinned
  ASSERT_EQ(planner_->AfterNode(2), kTfLiteOk);
  // Now that op2 has completed, BOTH A (1) and B (2) are freed!
  EXPECT_FALSE(IsAllocated(1));
  EXPECT_FALSE(IsAllocated(2));
  EXPECT_TRUE(IsAllocated(3));

  planner_->EndInvocation(/*completed_successfully=*/true);
}

TEST_F(SimplePlannerTest, DuplicateInputLiveness) {
  // op0: produces A (1)
  // op1: consumes A, A (like Add(A, A)), produces Y (2)
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {}},     // op0
                      {{1, 1}, {2}, {}},  // op1: duplicate inputs
                  },
                  {2});
  for (int i = 0; i <= 2; ++i) {
    (*graph.tensors())[i].bytes = 100;
  }

  SetGraph(&graph, /*enable_reclamation=*/true);
  Execute(0, 1);

  ASSERT_EQ(planner_->BeginInvocation(), kTfLiteOk);

  ASSERT_EQ(planner_->BeforeNode(0), kTfLiteOk);
  EXPECT_TRUE(IsAllocated(1));
  ASSERT_EQ(planner_->AfterNode(0), kTfLiteOk);

  ASSERT_EQ(planner_->BeforeNode(1), kTfLiteOk);
  ASSERT_EQ(planner_->AfterNode(1), kTfLiteOk);
  // A (1) freed exactly once
  EXPECT_FALSE(IsAllocated(1));
  EXPECT_EQ(planner_->total_deallocations(), 1);

  planner_->EndInvocation(/*completed_successfully=*/true);
}

TEST_F(SimplePlannerTest, UnconsumedOutputReclaimedImmediately) {
  // op0: produces A (1) and unconsumed dead tensor (2)
  // op1: consumes A (1), produces Y (3)
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1, 2}, {}},  // op0 produces 1 and 2
                      {{1}, {3}, {}},     // op1 consumes 1, produces 3
                  },
                  {3});
  for (int i = 0; i <= 3; ++i) {
    (*graph.tensors())[i].bytes = 100;
  }

  SetGraph(&graph, /*enable_reclamation=*/true);
  Execute(0, 1);

  ASSERT_EQ(planner_->BeginInvocation(), kTfLiteOk);

  // Before op0: A (1) and dead (2) are allocated
  ASSERT_EQ(planner_->BeforeNode(0), kTfLiteOk);
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));

  // After op0: dead (2) has no consumers and is not a graph output,
  // so it is reclaimed immediately after its producer!
  ASSERT_EQ(planner_->AfterNode(0), kTfLiteOk);
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_FALSE(IsAllocated(2));

  // op1
  ASSERT_EQ(planner_->BeforeNode(1), kTfLiteOk);
  ASSERT_EQ(planner_->AfterNode(1), kTfLiteOk);
  EXPECT_FALSE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(3));

  planner_->EndInvocation(/*completed_successfully=*/true);
}

TEST_F(SimplePlannerTest, TemporaryTensorLifecycle) {
  // op0: produces A (1) with temporary (2)
  // op1: consumes A (1), produces Y (3)
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {2}},  // op0 has temporary 2
                      {{1}, {3}, {}},   // op1
                  },
                  {3});
  for (int i = 0; i <= 3; ++i) {
    (*graph.tensors())[i].bytes = 100;
  }

  SetGraph(&graph, /*enable_reclamation=*/true);
  Execute(0, 1);

  // After preparation, temporary 2 is NOT allocated
  EXPECT_FALSE(IsAllocated(2));

  ASSERT_EQ(planner_->BeginInvocation(), kTfLiteOk);

  // Before op0, temporary 2 is allocated
  ASSERT_EQ(planner_->BeforeNode(0), kTfLiteOk);
  EXPECT_TRUE(IsAllocated(2));

  // After op0, temporary 2 is freed
  ASSERT_EQ(planner_->AfterNode(0), kTfLiteOk);
  EXPECT_FALSE(IsAllocated(2));

  planner_->EndInvocation(/*completed_successfully=*/true);
}

TEST_F(SimplePlannerTest, PersistentTemporaryNotReclaimed) {
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {2}},  // op0 has temporary 2
                      {{1}, {3}, {}},   // op1
                  },
                  {3});
  (*graph.tensors())[2].allocation_type = kTfLiteArenaRwPersistent;
  for (int i = 0; i <= 3; ++i) {
    (*graph.tensors())[i].bytes = 100;
  }

  SetGraph(&graph, /*enable_reclamation=*/true);
  Execute(0, 1);

  // Persistent temporary is allocated during preparation
  EXPECT_TRUE(IsAllocated(2));

  ASSERT_EQ(planner_->BeginInvocation(), kTfLiteOk);
  ASSERT_EQ(planner_->BeforeNode(0), kTfLiteOk);
  ASSERT_EQ(planner_->AfterNode(0), kTfLiteOk);
  // NOT freed after op0
  EXPECT_TRUE(IsAllocated(2));

  planner_->EndInvocation(/*completed_successfully=*/true);
  // Still allocated between invocations
  EXPECT_TRUE(IsAllocated(2));
}

TEST_F(SimplePlannerTest, PreserveAllTensorsOverridesReclamation) {
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {}},
                      {{1}, {2}, {}},
                  },
                  {2});
  for (int i = 0; i <= 2; ++i) {
    (*graph.tensors())[i].bytes = 100;
  }

  SetGraph(&graph, /*enable_reclamation=*/true, /*preserve_all=*/true);
  Execute(0, 1);

  // With preserve_all=true, all tensors allocated during preparation
  EXPECT_TRUE(IsAllocated(0));
  EXPECT_TRUE(IsAllocated(1));
  EXPECT_TRUE(IsAllocated(2));

  ASSERT_EQ(planner_->BeginInvocation(), kTfLiteOk);
  ASSERT_EQ(planner_->BeforeNode(0), kTfLiteOk);
  ASSERT_EQ(planner_->AfterNode(0), kTfLiteOk);
  ASSERT_EQ(planner_->BeforeNode(1), kTfLiteOk);
  ASSERT_EQ(planner_->AfterNode(1), kTfLiteOk);
  // Intermediate 1 is NOT freed!
  EXPECT_TRUE(IsAllocated(1));

  planner_->EndInvocation(/*completed_successfully=*/true);
  EXPECT_TRUE(IsAllocated(1));
}

TEST_F(SimplePlannerTest, HasNonPersistentMemoryState) {
  TestGraph graph({0}, {{{0}, {1}, {}}}, {1});
  SetGraph(&graph, /*enable_reclamation=*/true);
  Execute(0, 0);

  EXPECT_TRUE(planner_->HasNonPersistentMemory());
  EXPECT_EQ(planner_->BeginInvocation(), kTfLiteOk);
  planner_->EndInvocation(/*completed_successfully=*/true);

  // Release non-persistent memory
  ReleaseNonPersistentMemory();
  EXPECT_FALSE(planner_->HasNonPersistentMemory());

  // BeginInvocation must fail when non-persistent memory is not available
  EXPECT_NE(planner_->BeginInvocation(), kTfLiteOk);

  // Reacquire
  AcquireNonPersistentMemory();
  EXPECT_TRUE(planner_->HasNonPersistentMemory());
  EXPECT_EQ(planner_->BeginInvocation(), kTfLiteOk);
  planner_->EndInvocation(/*completed_successfully=*/true);
}

TEST_F(SimplePlannerTest, AllocationFailureRollback) {
  // Custom allocator that fails after 2 allocations.
  struct FailingAllocatorState {
    int count = 0;
    int max_success = 2;
  };
  static FailingAllocatorState state;
  state.count = 0;
  state.max_success = 2;

  static TfLiteAllocator failing_allocator = {
      &state,
      [](void* data, size_t bytes, size_t alignment) -> void* {
        auto* s = static_cast<FailingAllocatorState*>(data);
        if (++s->count > s->max_success) {
          return nullptr;  // Inject failure!
        }
        return malloc(bytes);
      },
      nullptr,
      [](void* data, void* ptr, size_t bytes, size_t alignment) {
        ::free(ptr);
      }};

  // op0 produces A (1) and B (2).
  // Preparation allocates X (0) [alloc 1] and Y (3) [alloc 2].
  // Then BeforeNode(0) allocates A (1) [alloc 3 -> FAILS!].
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1, 2}, {}},
                      {{1, 2}, {3}, {}},
                  },
                  {3});
  for (int i = 0; i <= 3; ++i) {
    (*graph.tensors())[i].bytes = 100;
  }

  SetGraph(&graph, /*enable_reclamation=*/true, /*preserve_all=*/false,
           &failing_allocator);
  Execute(0, 1);

  ASSERT_EQ(planner_->BeginInvocation(), kTfLiteOk);

  // BeforeNode(0) will attempt to allocate tensors 1 and 2.
  // One of them will fail, and BeforeNode must return kTfLiteError
  // and roll back any partial allocations.
  EXPECT_EQ(planner_->BeforeNode(0), kTfLiteError);

  // Pinned tensors X (0) and Y (3) remain intact
  EXPECT_TRUE(IsAllocated(0));
  EXPECT_TRUE(IsAllocated(3));

  planner_->EndInvocation(/*completed_successfully=*/false);
  planner_.reset();
}

TEST_F(SimplePlannerTest, EndInvocationFailureCleanup) {
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {}},
                      {{1}, {2}, {}},
                  },
                  {2});
  for (int i = 0; i <= 2; ++i) {
    (*graph.tensors())[i].bytes = 100;
  }

  SetGraph(&graph, /*enable_reclamation=*/true);
  Execute(0, 1);

  ASSERT_EQ(planner_->BeginInvocation(), kTfLiteOk);
  ASSERT_EQ(planner_->BeforeNode(0), kTfLiteOk);
  EXPECT_TRUE(IsAllocated(1));

  // Simulate an error or cancellation before node 1 executes:
  planner_->EndInvocation(/*completed_successfully=*/false);

  // Intermediate 1 must be cleaned up to avoid leaking across invocations!
  EXPECT_FALSE(IsAllocated(1));
  // Pinned inputs and outputs are preserved
  EXPECT_TRUE(IsAllocated(0));
  EXPECT_TRUE(IsAllocated(2));
}

}  // namespace
}  // namespace tflite
