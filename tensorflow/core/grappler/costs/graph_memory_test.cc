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

#include "tensorflow/core/grappler/costs/graph_memory.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class GraphMemoryTest : public ::testing::Test {
 protected:
  std::unordered_map<string, DeviceProperties> devices_;

 public:
  GraphMemoryTest() {
    devices_["/CPU:0"].set_type("CPU");
    devices_["/CPU:0"].set_num_cores(1);
    devices_["/CPU:0"].set_frequency(1);
    devices_["/CPU:0"].set_bandwidth(1);

    devices_["/GPU:0"].set_type("GPU");
    devices_["/GPU:0"].set_num_cores(1);
    devices_["/GPU:0"].set_frequency(1);
    devices_["/CPU:0"].set_bandwidth(1);
    (*devices_["/GPU:0"].mutable_environment())["architecture"] = "3";
  }
};

TEST_F(GraphMemoryTest, Basic) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"/CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));
  item.feed.clear();

  GraphMemory memory(item);
  Status s = memory.InferStatically(devices_);
  TF_CHECK_OK(s);
  const GraphMemory::MemoryUsage& mem_usage =
      memory.GetPeakMemoryUsage("/CPU:0");
  EXPECT_EQ(120, mem_usage.used_memory);

  std::set<string> tensors;
  for (const auto& t : mem_usage.live_tensors) {
    tensors.insert(strings::StrCat(t.node, ":", t.output_id));
  }
  // When the execution of the 'Square' node completes, TF can start executing
  // 'Square_1' and release the memory used by 'x'. Since we can't be sure of
  // the order in which this takes place, in the worst case the 3 tensors are in
  // memory.
  std::set<string> expected;
  expected.insert("Square:0");
  expected.insert("Square_1:0");
  expected.insert("x:0");
  EXPECT_EQ(expected, tensors);
}

TEST_F(GraphMemoryTest, UnknownBatchSize) {
  TrivialTestGraphInputYielder fake_input(4, 1, -1, false, {"/CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));
  item.feed.clear();

  GraphMemory memory(item);
  Status s = memory.InferStatically(devices_);
  TF_CHECK_OK(s);
  // Same maths as before, except that batch size is unknown and therefore
  // assumed to be one.
  const GraphMemory::MemoryUsage& mem_usage =
      memory.GetPeakMemoryUsage("/CPU:0");
  EXPECT_EQ(16, mem_usage.used_memory);

  std::set<string> tensors;
  for (const auto& t : mem_usage.live_tensors) {
    tensors.insert(strings::StrCat(t.node, ":", t.output_id));
  }
  std::set<string> expected;
  expected.insert("Const/Const:0");
  expected.insert("Square:0");
  expected.insert("x:0");
  EXPECT_EQ(expected, tensors);
}

TEST_F(GraphMemoryTest, MultiDevice) {
  TrivialTestGraphInputYielder fake_input(4, 2, 1024 * 1024, false,
                                          {"/CPU:0", "/GPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));
  item.feed.clear();

  GraphMemory memory(item);
  Status s = memory.InferStatically(devices_);
  TF_CHECK_OK(s);

  const GraphMemory::MemoryUsage& cpu_mem = memory.GetPeakMemoryUsage("/CPU:0");
  EXPECT_EQ(16777216, cpu_mem.used_memory);
  std::set<string> cpu_tensors;
  for (const auto& t : cpu_mem.live_tensors) {
    cpu_tensors.insert(strings::StrCat(t.node, ":", t.output_id));
  }
  std::set<string> cpu_expected;
  cpu_expected.insert("Recv_Square_1_0_on_/CPU_0:0");
  cpu_expected.insert("Square:0");
  cpu_expected.insert("x:0");
  cpu_expected.insert("AddN:0");
  EXPECT_EQ(cpu_expected, cpu_tensors);

  const GraphMemory::MemoryUsage& gpu_mem = memory.GetPeakMemoryUsage("/GPU:0");
  EXPECT_EQ(16777216, gpu_mem.used_memory);
  std::set<string> gpu_tensors;
  for (const auto& t : gpu_mem.live_tensors) {
    gpu_tensors.insert(strings::StrCat(t.node, ":", t.output_id));
  }
  std::set<string> gpu_expected;
  gpu_expected.insert("Recv_AddN_0_on_/GPU_0:0");
  gpu_expected.insert("Square_1:0");
  gpu_expected.insert("AddN_1:0");
  gpu_expected.insert("AddN_3:0");
  EXPECT_EQ(gpu_expected, gpu_tensors);
}

TEST_F(GraphMemoryTest, GpuSwapping) {
  TrivialTestGraphInputYielder fake_input(4, 2, 1024 * 1024, false, {"/GPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));
  item.feed.clear();

  {
    // Estimate the max memory usage for the graph.
    GraphMemory memory(item);
    Status s = memory.InferStatically(devices_);
    TF_CHECK_OK(s);

    const GraphMemory::MemoryUsage& gpu_mem =
        memory.GetPeakMemoryUsage("/GPU:0");
    EXPECT_EQ(20971520, gpu_mem.used_memory);
    std::set<string> gpu_tensors;
    for (const auto& t : gpu_mem.live_tensors) {
      gpu_tensors.insert(strings::StrCat(t.node, ":", t.output_id));
    }
    std::set<string> gpu_expected;
    gpu_expected.insert("Square:0");
    gpu_expected.insert("Square_1:0");
    gpu_expected.insert("AddN:0");
    gpu_expected.insert("AddN_1:0");
    gpu_expected.insert("AddN_2:0");
    EXPECT_EQ(gpu_expected, gpu_tensors);
  }

  {
    // Swap the first input to node AddN_1: its fanin (the square nodes) should
    // not appear in the max cut anymore.
    for (auto& node : *item.graph.mutable_node()) {
      if (node.name() == "AddN_1") {
        (*node.mutable_attr())["_swap_to_host"].mutable_list()->add_i(0);
      }
    }
    GraphMemory memory(item);
    Status s = memory.InferStatically(devices_);
    TF_CHECK_OK(s);
    const GraphMemory::MemoryUsage& new_gpu_mem =
        memory.GetPeakMemoryUsage("/GPU:0");
    EXPECT_EQ(20971520, new_gpu_mem.used_memory);
    std::set<string> new_gpu_tensors;
    for (const auto& t : new_gpu_mem.live_tensors) {
      new_gpu_tensors.insert(strings::StrCat(t.node, ":", t.output_id));
    }
    std::set<string> new_gpu_expected;
    new_gpu_expected.insert("AddN:0");
    new_gpu_expected.insert("AddN_1:0");
    new_gpu_expected.insert("AddN_2:0");
    new_gpu_expected.insert("AddN_3:0");
    new_gpu_expected.insert("AddN_4:0");
    EXPECT_EQ(new_gpu_expected, new_gpu_tensors);
  }
}

TEST_F(GraphMemoryTest, CtrlDependencies) {
  // Build a simple graph with a control dependency.
  Scope s = Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a").WithDevice("/CPU:0"), 10.0f, {3});
  Output v =
      ops::Variable(s.WithOpName("v").WithDevice("/CPU:0"), {3}, DT_FLOAT);
  Output assign =
      ops::Assign(s.WithOpName("assign").WithDevice("/CPU:0"), v, a);
  ops::NoOp init(
      s.WithOpName("init").WithDevice("/CPU:0").WithControlDependencies(
          assign));

  GrapplerItem item;
  item.fetch.push_back("init");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphMemory memory(item);
  Status status = memory.InferStatically(devices_);
  TF_CHECK_OK(status);

  const GraphMemory::MemoryUsage& mem = memory.GetPeakMemoryUsage("/CPU:0");
  EXPECT_EQ(36, mem.used_memory);
  std::set<string> tensors;
  for (const auto& t : mem.live_tensors) {
    tensors.insert(strings::StrCat(t.node, ":", t.output_id));
  }
  std::set<string> expected;
  expected.insert("a:0");
  expected.insert("v:0");
  expected.insert("assign:0");
  EXPECT_EQ(expected, tensors);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
