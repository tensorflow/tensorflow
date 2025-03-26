/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/arena_planner.h"

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/graph_info.h"

namespace tflite {

// Profiler allocation hook for testing.
int gNumAlloc = 0;
void OnTfLiteArenaAlloc(int subgraph_index, int arena_id, size_t num_bytes) {
  gNumAlloc++;
}

// Profiler deallocation hook for testing.
int gNumDealloc = 0;
void OnTfLiteArenaDealloc(int subgraph_index, int arena_id, size_t num_bytes) {
  gNumDealloc++;
}

namespace {

constexpr const int kTensorAlignment = 4;

// A simple op to be used in tests, as syntactic sugar.
class TestOp {
 public:
  TestOp(std::initializer_list<int> inputs, std::initializer_list<int> outputs,
         std::initializer_list<int> temporaries,
         int builtin_code = kTfLiteBuiltinAdd,
         int inplace_operator = kTfLiteInplaceOpInput0Shared)
      : inputs_(inputs),
        outputs_(outputs),
        temporaries_(temporaries),
        registration_{} {
    registration_.builtin_code = builtin_code;
    registration_.inplace_operator = inplace_operator;
  }

  const std::vector<int>& inputs() const { return inputs_; }
  const std::vector<int>& outputs() const { return outputs_; }
  const std::vector<int>& temporaries() const { return temporaries_; }
  const TfLiteRegistration& registration() const { return registration_; }

 private:
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<int> temporaries_;
  TfLiteRegistration registration_;
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
  TfLiteTensor* tensors() override { return graph_->tensors()->data(); }
  TfLiteTensor* tensor(size_t index) override {
    return &graph_->tensors()->at(index);
  }
  size_t num_execution_nodes() const override { return graph_->nodes().size(); }
  size_t num_total_nodes() const override { return graph_->nodes().size(); }
  const TfLiteNode& node(size_t index) const override {
    return graph_->nodes()[index];
  }
  const TfLiteRegistration& registration(size_t index) const override {
    return graph_->registrations()[index];
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

class ArenaPlannerTest : public ::testing::Test {
 protected:
  void SetGraph(TestGraph* graph, bool preserve_all_tensors = false) {
    graph_ = graph;
    context_.ReportError = ReportError;
    planner_ = std::make_unique<ArenaPlanner>(
        &context_, std::unique_ptr<GraphInfo>(new TestGraphInfo(graph)),
        preserve_all_tensors, kTensorAlignment);
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

  void ResetAllocations() { CHECK(planner_->ResetAllocations() == kTfLiteOk); }

  void ResetAllocationsAfter(int node) {
    CHECK(planner_->ResetAllocationsAfter(node) == kTfLiteOk);
  }

  bool HasNonPersistentMemory() {
    return planner_ && planner_->HasNonPersistentMemory();
  }

  void Destroy() { planner_.reset(); }

  // Returns the actual offset of a given tensor, relative to the start of its
  // arena.
  std::ptrdiff_t GetOffset(int tensor_index) {
    const TfLiteTensor& tensor = (*graph_->tensors())[tensor_index];
    return reinterpret_cast<std::intptr_t>(tensor.data.raw) -
           planner_->BasePointer(tensor.allocation_type);
  }

  // Returns the first aligned offset after a given tensor.
  std::ptrdiff_t GetOffsetAfter(int tensor_index) {
    const TfLiteTensor& tensor = (*graph_->tensors())[tensor_index];
    std::ptrdiff_t offset = GetOffset(tensor_index) + tensor.bytes;
    // We must make sure the offset is aligned to kDefaultArenaAlignment.
    if (offset % kTensorAlignment != 0) {
      offset += kTensorAlignment - offset % kTensorAlignment;
    }
    return offset;
  }

  // Returns if the given tensor is unallocated or not.
  bool IsUnallocated(int tensor_index) {
    return (*graph_->tensors())[tensor_index].data.raw == nullptr;
  }

  TfLiteContext context_;
  TestGraph* graph_;
  std::unique_ptr<ArenaPlanner> planner_;
};

TEST_F(ArenaPlannerTest, EmptyGraph) {
  TestGraph graph({}, {}, {});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);
}

TEST_F(ArenaPlannerTest, GraphWithOneOp) {
  TestGraph graph({0, 10}, {{{0}, {}, {}}, {{10}, {}, {}}}, {5, 11});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(10), GetOffsetAfter(0));
  // The outputs are never allocated because they are not connected to any
  // inputs.
  EXPECT_TRUE((*graph.tensors())[5].data.raw == nullptr);
  EXPECT_TRUE((*graph.tensors())[11].data.raw == nullptr);
}

TEST_F(ArenaPlannerTest, GraphWithOneOp2) {
  TestGraph graph({1}, {{{1}, {2}, {}}}, {2});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);
  EXPECT_EQ(GetOffset(2), 8);
  EXPECT_EQ(GetOffsetAfter(2), 20);
}

TEST_F(ArenaPlannerTest, ZeroSizedTensors) {
  TestGraph graph({1}, {{{1}, {2}, {}}}, {2});
  (*graph.tensors())[1].bytes = 0;
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);
  EXPECT_EQ((*graph_->tensors())[1].data.raw, nullptr);
}

TEST_F(ArenaPlannerTest, SimpleGraph) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +4 +5 -2 +3 -4 -5
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(1), 4);
}

TEST_F(ArenaPlannerTest, AllocsCorrectlyReset) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +4 +5 -2 +3 -4 -5
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(1), 4);

  // Increase tensor sizes to trigger a reallocation, but not enough to change
  // their offsets. Adding one byte will fill the space left empty by alignment
  // requirements. If the allocs have not been reset, then the offsets will have
  // increased since the old allocs are still present preventing memory reuse.
  ResetAllocations();
  std::vector<TfLiteTensor>& tensors = *graph.tensors();
  tensors[0].bytes += 1;
  tensors[1].bytes += 1;
  tensors[2].bytes += 1;
  tensors[3].bytes += 1;
  tensors[4].bytes += 1;
  tensors[5].bytes += 1;
  Execute(0, graph.nodes().size() - 1);

  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(1), 4);
}

TEST_F(ArenaPlannerTest, SimpleGraphInputsPreserved) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +4 +5 -2 +3 -4 -5
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), GetOffsetAfter(0));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(1));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
}

TEST_F(ArenaPlannerTest, SimpleGraphWithTemporary) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with temporary
                      {{4}, {3}, {}}       // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +5 +4 -2 -5 +3 -4
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(1), 4);
}

TEST_F(ArenaPlannerTest, SimpleGraphWithInplaceReshape) {
  TestGraph graph(
      {0, 1},
      {
          /* in, out, tmp */
          {{0}, {2}, {}},  // First op
          {{1}, {3}, {}},  // Second op
          // Third op, with in-place reshape.
          {{2, 3},
           {4},
           {},
           kTfLiteBuiltinReshape,
           kTfLiteInplaceOpInput0Shared | kTfLiteInplaceOpDataUnmodified},
          {{4}, {5}, {}}  // Fourth Op, output
      },
      {5});
  (*graph.tensors())[2].bytes = 24;
  (*graph.tensors())[4].bytes = 24;
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Tensors two and 4 should have the same offset.
  EXPECT_EQ(GetOffset(2), GetOffset(4));
}

TEST_F(ArenaPlannerTest, SimpleGraphWithChainOfInplaceOps) {
  TestGraph graph(
      {0, 1},
      {
          /* in, out, tmp */
          {{0}, {2}, {}},
          {{2, 3},
           {4},
           {},
           kTfLiteBuiltinReshape,
           kTfLiteInplaceOpInput0Shared | kTfLiteInplaceOpDataUnmodified},
          {{4, 3},
           {5},
           {},
           kTfLiteBuiltinExpandDims,
           kTfLiteInplaceOpInput0Shared | kTfLiteInplaceOpDataUnmodified},
          {{5, 3},
           {6},
           {},
           kTfLiteBuiltinSqueeze,
           kTfLiteInplaceOpInput0Shared | kTfLiteInplaceOpDataUnmodified},
          {{6, 3},
           {7},
           {},
           kTfLiteBuiltinReshape,
           kTfLiteInplaceOpInput0Shared | kTfLiteInplaceOpDataUnmodified},
          {{7}, {8}, {}},
      },
      {8});
  (*graph.tensors())[2].bytes = 24;
  (*graph.tensors())[4].bytes = 24;
  (*graph.tensors())[5].bytes = 24;
  (*graph.tensors())[6].bytes = 24;
  (*graph.tensors())[7].bytes = 24;
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Tensors 2,4 5, 6, 7 should have the same offset.
  EXPECT_EQ(GetOffset(2), GetOffset(2));
  EXPECT_EQ(GetOffset(2), GetOffset(4));
  EXPECT_EQ(GetOffset(2), GetOffset(5));
  EXPECT_EQ(GetOffset(2), GetOffset(6));
  EXPECT_EQ(GetOffset(2), GetOffset(7));
}

TEST_F(ArenaPlannerTest, SimpleGraphsWithReshapeInputOutput) {
  TestGraph graph(
      {0, 1},
      {/* in, out, tmp */
       {{0}, {2}, {}},
       // Reshape's input and output are not graph inputs or outputs.
       {{2, 1},
        {3},
        {},
        kTfLiteBuiltinReshape,
        kTfLiteInplaceOpInput0Shared | kTfLiteInplaceOpDataUnmodified},
       {{3}, {4}, {}}},
      {4});
  (*graph.tensors())[2].bytes = 24;
  (*graph.tensors())[3].bytes = 24;
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Tensors 2 and 3 should have the same offset.
  EXPECT_EQ(GetOffset(2), GetOffset(3));
}

TEST_F(ArenaPlannerTest, SimpleGraphsWithReshapeInputTensor) {
  TestGraph graph({0, 1},
                  {/* in, out, tmp */
                   {{0, 1},
                    {2},
                    {},
                    kTfLiteBuiltinReshape,
                    kTfLiteInplaceOpInput0Shared |
                        kTfLiteInplaceOpDataUnmodified},  // First op is reshape
                   {{4}, {3}, {}}},
                  {3});
  SetGraph(&graph);
  // Only arena allocated tensors have shared buffer.
  (*graph.tensors())[0].allocation_type = kTfLiteDynamic;
  Execute(0, graph.nodes().size() - 1);

  // Tensors 0 and 2 should have different offsets.
  EXPECT_NE(GetOffset(0), GetOffset(2));
}

TEST_F(ArenaPlannerTest, SimpleGraphsWithReshapeOutputTensor) {
  TestGraph graph(
      {0, 1},
      {
          /* in, out, tmp */
          {{0}, {2}, {}},
          {{2, 1},
           {3},
           {},
           kTfLiteBuiltinReshape,
           kTfLiteInplaceOpInput0Shared |
               kTfLiteInplaceOpDataUnmodified},  // Last op is reshape
      },
      {3});
  SetGraph(&graph);
  // Only arena allocated tensors have shared buffer.
  (*graph.tensors())[0].allocation_type = kTfLiteDynamic;
  Execute(0, graph.nodes().size() - 1);

  // Tensors 2 and 3 should have different offsets.
  EXPECT_NE(GetOffset(2), GetOffset(3));
}

TEST_F(ArenaPlannerTest, SimpleGraphsWithReshapeDynamicInput) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1},
                       {2},
                       {},
                       kTfLiteBuiltinReshape,
                       kTfLiteInplaceOpDataUnmodified}  // First op is reshape
                  },
                  {2});
  SetGraph(&graph);
  // Only arena allocated tensors have shared buffer.
  (*graph.tensors())[0].allocation_type = kTfLiteDynamic;
  Execute(0, graph.nodes().size() - 1);

  // Tensors 0 and 2 should have different offsets.
  EXPECT_NE(GetOffset(0), GetOffset(2));
}

TEST_F(ArenaPlannerTest, SimpleGraphsWithBroadcastingAddInPlace) {
  TestGraph graph(
      {0, 1},
      {
          /* in, out, tmp */
          {{0, 1}, {3}, {}},
          {{1, 2}, {4}, {}},
          {{3, 4},
           {5},
           {},
           kTfLiteBuiltinAdd,
           kTfLiteInplaceOpInput0Shared | kTfLiteInplaceOpInput1Shared},
          {{5}, {6}, {}},
      },
      {6});
  // Only arena allocated tensors have shared buffer.
  (*graph.tensors())[3].bytes = 8;   // shape [8]
  (*graph.tensors())[4].bytes = 16;  // shape [2, 8]
  (*graph.tensors())[5].bytes = 16;  // shape [2, 8]

  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Tensors 3 and 5 should have different offsets.
  EXPECT_NE(GetOffset(3), GetOffset(5));
  // Tensors 4 and 5 should have the same offsets.
  EXPECT_EQ(GetOffset(4), GetOffset(5));
}

TEST_F(ArenaPlannerTest, SimpleGraphsWithBroadcastingAddNotInPlace) {
  TestGraph graph(
      {0, 1},
      {
          /* in, out, tmp */
          {{0, 1}, {3}, {}},
          {{1, 2}, {4}, {}},
          {{3, 4}, {5}, {}, kTfLiteBuiltinAdd, kTfLiteInplaceOpInput0Shared},
          {{5}, {6}, {}},
      },
      {6});
  // Only arena allocated tensors have shared buffer.
  (*graph.tensors())[3].bytes = 8;   // shape [8, 1]
  (*graph.tensors())[4].bytes = 8;   // shape [1, 8]
  (*graph.tensors())[5].bytes = 64;  // shape [7, 8]

  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Tensors 3 and 5 should have different offsets.
  EXPECT_NE(GetOffset(3), GetOffset(5));
  // Tensors 4 and 5 should have different offsets.
  EXPECT_NE(GetOffset(4), GetOffset(5));
}

TEST_F(ArenaPlannerTest, SimpleGraphWithResetAllocationsAfter) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with temporary
                      {{4}, {3}, {}}       // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +5 +4 -2 -5 +3 -4
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));

  // Reset allocations after the first node
  ResetAllocationsAfter(0);

  EXPECT_FALSE(IsUnallocated(0));
  EXPECT_FALSE(IsUnallocated(1));
  EXPECT_FALSE(IsUnallocated(2));
  EXPECT_TRUE(IsUnallocated(3));
  EXPECT_TRUE(IsUnallocated(4));
  EXPECT_TRUE(IsUnallocated(5));

  // Trigger a reallocation after the 2nd op so that allocations are calculated
  // from an intermediate node focing active allocs to be re-generated.
  (*graph.tensors())[4].bytes += 64;
  Execute(1, graph.nodes().size() - 1);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(2), 48);
}

TEST_F(ArenaPlannerTest, SimpleGraphWithPersistentResetAllocationsAfter) {
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
  Execute(0, graph.nodes().size() - 1);

  // Save the pointer of the persistent temporary tensor #5.
  void* tensor5_ptr = (*graph.tensors())[5].data.raw;

  // Reset allocations after the first node
  ResetAllocationsAfter(0);

  EXPECT_FALSE(IsUnallocated(0));
  EXPECT_FALSE(IsUnallocated(1));
  EXPECT_FALSE(IsUnallocated(2));
  EXPECT_TRUE(IsUnallocated(3));
  EXPECT_TRUE(IsUnallocated(4));
  EXPECT_FALSE(IsUnallocated(5));

  // Second run
  Execute(0, graph.nodes().size() - 1);

  // Check if the persistent pointer isn't changed.
  EXPECT_TRUE(tensor5_ptr == (*graph.tensors())[5].data.raw);
}

TEST_F(ArenaPlannerTest, SimpleGraphWithOptionals) {
  TestGraph graph({0, -1, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, -1, 5}, {3}, {}}  // Third op, with optional
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +4 +5 -2 +3 -4 -5
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
}

TEST_F(ArenaPlannerTest, SimpleGraphWithOptionalOutput) {
  TestGraph graph({0, -1, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op, with optional
                  },
                  {-1, 3});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +4 +5 -2 +3 -4 -5
  EXPECT_EQ(GetOffset(5), 12);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
}

TEST_F(ArenaPlannerTest, SimpleGraphWithLargeTensor) {
  TestGraph graph({0, -1},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {}},      // First op
                      {{1}, {2}, {}},      // Second op
                      {{2, 0}, {4}, {5}},  // Third op, with temporary
                      {{4, -1}, {3}, {}}   // Fourth op, with optional
                  },
                  {3});

  // Make #1 very large so its vacancy can be filled with #5 and #4.
  (*graph.tensors())[1].bytes = 40;

  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 -1 +5 +4 -2 -5 +3 -4
  EXPECT_EQ(GetOffset(1), 4);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(1));
  EXPECT_EQ(GetOffset(3), 4);
  EXPECT_EQ(GetOffset(5), 4);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
}

TEST_F(ArenaPlannerTest, SimpleGraphWithPersistentTensor) {
  TestGraph graph({0, -1, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with persistent
                      {{4, -1}, {3}, {}}   // Third op, with optional
                  },
                  {3});

  // Make #1 persistent so it goes into its own arena.
  (*graph.tensors())[1].allocation_type = kTfLiteArenaRwPersistent;
  // The only use case for kTfLiteArenaRwPersistent is variable tensor now.
  graph.SetVariables({1});

  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Make sure #0 and #1 were given different memory locations (because they
  // will both have offset=0, in different arenas.)
  EXPECT_NE((*graph.tensors())[0].data.raw, (*graph.tensors())[1].data.raw);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +5 +4 -2 -5 +3 -4
  EXPECT_EQ(GetOffset(5), 4);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), 4);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), 0);
}

TEST_F(ArenaPlannerTest, SimpleGraphWithDynamicTensor) {
  TestGraph graph({0, -1, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},   // First op
                      {{2, 0}, {4}, {5}},  // Second op, with temporary
                      {{4, -1}, {3}, {}}   // Third op, with optional
                  },
                  {3});

  // Make #1 dynamic so it does not get allocated.
  (*graph.tensors())[1].allocation_type = kTfLiteDynamic;

  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  EXPECT_EQ((*graph.tensors())[1].data.raw, nullptr);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +5 +4 -2 -5 +3 -4
  EXPECT_EQ(GetOffset(5), 4);
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), 4);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(4));
}

TEST_F(ArenaPlannerTest, LargerGraphAndStepwiseAllocation) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2, 3}, {}},
                      {{2, 0}, {4, 5}, {6}},
                      {{1, -1}, {7}, {}},
                      {{7, 3}, {8}, {9}},
                      {{4, 5, 8}, {10}, {}},
                  },
                  {10});
  SetGraph(&graph);

  // The allocation plan is made at the beginning and is independent of
  // the execution steps. Here's the allocation order:
  //   Op0: +0 +1 +2 +3
  //   Op1: +6 +4 +5 -6 -2
  //   Op2: +7
  //   Op3: +9 +8 -9 -3 -7
  //   Op4: +10 -4 -5 -8

  Execute(0, 0);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(3));
  EXPECT_TRUE(IsUnallocated(6));
  EXPECT_TRUE(IsUnallocated(4));
  EXPECT_TRUE(IsUnallocated(5));
  EXPECT_TRUE(IsUnallocated(7));
  EXPECT_TRUE(IsUnallocated(9));
  EXPECT_TRUE(IsUnallocated(8));
  EXPECT_TRUE(IsUnallocated(10));

  Execute(1, 1);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(6));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_TRUE(IsUnallocated(7));
  EXPECT_TRUE(IsUnallocated(9));
  EXPECT_TRUE(IsUnallocated(8));
  EXPECT_TRUE(IsUnallocated(10));

  Execute(2, 2);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(6));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(7), GetOffsetAfter(3));
  EXPECT_TRUE(IsUnallocated(9));
  EXPECT_TRUE(IsUnallocated(8));
  EXPECT_TRUE(IsUnallocated(10));

  Execute(3, 3);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(6));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(7), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(9), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(8), GetOffsetAfter(9));
  EXPECT_TRUE(IsUnallocated(10));

  Execute(4, 4);
  EXPECT_EQ(GetOffset(3), 12);
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(6));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(7), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(9), GetOffsetAfter(4));
  EXPECT_EQ(GetOffset(8), GetOffsetAfter(9));
  EXPECT_EQ(GetOffset(10), 12);
}

TEST_F(ArenaPlannerTest, ModifiedGraph) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Now update the graph data used by the existing allocator. It should behave
  // as if it had been recreated with the new graph.
  TestGraph pruned_graph({0, 1},
                         {
                             /* in, out, tmp */
                             {{0, 1}, {3}, {}},  // First op
                         },
                         {3});
  SwapGraph(&pruned_graph);
  Execute(0, graph.nodes().size() - 1);

  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), GetOffsetAfter(0));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(1));
}

TEST_F(ArenaPlannerTest, ModifiedGraph_DeallocateNonPersistentArena) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {}},     // First op
                      {{2, 0}, {4, 5}, {}},  // Second op
                      {{4, 5}, {3}, {}}      // Third op
                  },
                  {3});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Should be no-ops, since ReleaseNonPersistentMemory() hasn't been called.
  AcquireNonPersistentMemory();
  AcquireNonPersistentMemory();

  EXPECT_TRUE(HasNonPersistentMemory());

  // Release non-persistent arena.
  ReleaseNonPersistentMemory();
  EXPECT_FALSE(HasNonPersistentMemory());
  // Offsets should be zero.
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), 0);
  EXPECT_EQ(GetOffset(3), 0);

  // Now update the graph data used by the existing allocator. It should behave
  // as if it had been recreated with the new graph.
  TestGraph pruned_graph({0, 1},
                         {
                             /* in, out, tmp */
                             {{0, 1}, {3}, {}},  // First op
                         },
                         {3});
  SwapGraph(&pruned_graph);
  Execute(0, graph.nodes().size() - 1);

  // Should be a no-op.
  AcquireNonPersistentMemory();
  EXPECT_TRUE(HasNonPersistentMemory());

  // Release & acquire non-persistent memory.
  ReleaseNonPersistentMemory();
  AcquireNonPersistentMemory();
  // Offset checks from previous test should still apply.
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), GetOffsetAfter(0));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(1));
}

TEST_F(ArenaPlannerTest, ComplexGraph) {
  TestGraph graph({0},
                  {
                      /* in, out, tmp */
                      {{0}, {1}, {}},
                      {{1}, {2}, {}},
                      {{1}, {3}, {}},
                      {{1}, {4}, {}},
                      {{2, 3, 4}, {5}, {}},
                      {{5}, {6}, {}},
                      {{5}, {7}, {}},
                      {{6, 7}, {8}, {}},
                  },
                  {8});
  (*graph.tensors())[0].bytes = 32;
  (*graph.tensors())[1].bytes = 28;
  (*graph.tensors())[2].bytes = 36;
  (*graph.tensors())[3].bytes = 16;
  (*graph.tensors())[4].bytes = 8;
  (*graph.tensors())[5].bytes = 64;
  (*graph.tensors())[6].bytes = 10;
  (*graph.tensors())[7].bytes = 40;
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Alloc(+) and dealloc(-) order: +0 +1 +2 +3 +4 -1 +5 -2 -3 -4 +6 +7 -5 +8
  EXPECT_EQ(GetOffset(5), 32);
  EXPECT_EQ(GetOffset(7), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(7));
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(5));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(2));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(3));
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), GetOffsetAfter(0));
  EXPECT_EQ(GetOffset(8), 32);
}

TEST_F(ArenaPlannerTest, GraphWithIntermediates) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0}, {2}, {3}},
                      {{1, 2}, {4, 5}, {}},
                      {{5}, {6, 7}, {8, 9, 10}},
                      {{4, 6}, {11}, {12}},
                      {{11}, {13}, {}},
                      {{7, 13}, {14}, {15}},
                  },
                  {11, 14});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  // Alloc(+) and dealloc(-) order by operation:
  // Op0: +0 +1 +2 +3 -3
  // Op1: +4 +5 -2 -4
  // Op2: +6 +7 +8 +9 +10 -8 -9 -10 -5
  // Op3: +11 +12 -12 -4 -6
  // Op4: +13
  // Op5: +14 +15 -7 -13 -15
  EXPECT_EQ(GetOffset(0), 0);
  EXPECT_EQ(GetOffset(1), GetOffsetAfter(0));
  EXPECT_EQ(GetOffset(15), GetOffsetAfter(1));
  EXPECT_EQ(GetOffset(14), GetOffsetAfter(15));
  EXPECT_EQ(GetOffset(13), GetOffsetAfter(14));
  EXPECT_EQ(GetOffset(12), GetOffsetAfter(1));
  EXPECT_EQ(GetOffset(11), GetOffsetAfter(13));
  EXPECT_EQ(GetOffset(10), GetOffsetAfter(1));
  EXPECT_EQ(GetOffset(9), GetOffsetAfter(10));
  EXPECT_EQ(GetOffset(8), GetOffsetAfter(9));
  EXPECT_EQ(GetOffset(7), GetOffsetAfter(11));
  EXPECT_EQ(GetOffset(6), GetOffsetAfter(8));
  EXPECT_EQ(GetOffset(5), GetOffsetAfter(6));
  EXPECT_EQ(GetOffset(4), GetOffsetAfter(7));
  EXPECT_EQ(GetOffset(3), GetOffsetAfter(1));

  // 2 is allocated in the smallest suitable gap, which is not equal to the
  // first available one.
  EXPECT_EQ(GetOffset(2), GetOffsetAfter(5));
}

TEST_F(ArenaPlannerTest, DebugTensors) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2}, {5}},  // First op, with temporary
                      {{2, 0}, {4}, {6}},  // Second op, with temporary
                      {{4}, {3}, {7}}      // Third op, with temporary
                  },
                  {3});
  SetGraph(&graph, /*preserve_all_tensors=*/false);
  Execute(0, graph.nodes().size() - 1);

  // Memory of temporary tensors are shared by default.
  EXPECT_EQ(GetOffset(5), GetOffset(6));
  EXPECT_EQ(GetOffset(6), GetOffset(7));

  SetGraph(&graph, /*preserve_all_tensors=*/true);
  Execute(0, graph.nodes().size() - 1);

  std::set<std::ptrdiff_t> tensorOffsets;
  for (int i = 0; i < 8; i++) {
    tensorOffsets.insert(GetOffset(i));
  }
  // Every tensor should have unique memory allocation with
  // preserve_all_tensors.
  EXPECT_EQ(tensorOffsets.size(), 8);
}

TEST_F(ArenaPlannerTest, DebugTensorsInputReuse) {
  TestGraph graph({0, 1},
                  {
                      /* in, out, tmp */
                      {{0, 1}, {2, 3}, {}},
                      {{2, 3}, {4}, {}, kTfLiteBuiltinMul},
                      {{4, 2}, {5}, {}, kTfLiteBuiltinSub},
                      {{5}, {6}, {}},
                  },
                  {6});

  (*graph.tensors())[4].bytes = 200;
  (*graph.tensors())[5].bytes = 200;

  SetGraph(&graph, /*preserve_all_tensors=*/false);
  Execute(0, graph.nodes().size() - 1);

  // Output of mul node should be reused for output of sub node.
  EXPECT_EQ(GetOffset(4), GetOffset(5));

  SetGraph(&graph, /*preserve_all_tensors=*/true);
  Execute(0, graph.nodes().size() - 1);

  // Output of mul node should not be reused for output of sub node.
  EXPECT_NE(GetOffset(4), GetOffset(5));
}

TEST_F(ArenaPlannerTest, SimpleProfilerTest) {
  gNumAlloc = 0;
  gNumDealloc = 0;
  TestGraph graph({1}, {{{1}, {2}, {}}}, {2});
  SetGraph(&graph);
  Execute(0, graph.nodes().size() - 1);

  EXPECT_EQ(gNumAlloc, 1);
  EXPECT_EQ(gNumDealloc, 0);
  Destroy();
  EXPECT_EQ(gNumDealloc, 1);
}

}  // namespace
}  // namespace tflite
