/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/partitioning_utils.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

class PartitioningUtilsTest : public ::testing::Test {
 public:
  void SetUp() override {
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 2});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, "/job:a/replica:0/task:0",
                                          &devices));
    device0_ = devices[0].get();
    device1_ = devices[1].get();
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));

    for (auto d : device_mgr_->ListDevices()) {
      device_set_.AddDevice(d);
    }
  }

  void SwapGraph(Graph* graph, bool assign_device = false) {
    Scope s = Scope::NewRootScope();
    if (assign_device) {
      s = s.WithDevice(device0_->name());
    }
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto y = ops::_Arg(s.WithOpName("y"), DT_FLOAT, 1);
    auto id_x = ops::Identity(s.WithOpName("id_x"), x);
    auto id_y = ops::Identity(s.WithOpName("id_y"), y);
    auto dx_retval = ops::_Retval(s.WithOpName("retval1"), id_y, 0);
    auto dy_retval = ops::_Retval(s.WithOpName("retval2"), id_x, 1);
    TF_ASSERT_OK(s.ToGraph(graph));

    if (assign_device) {
      Placer placer(graph, "", &device_set_, device0_);
      TF_ASSERT_OK(placer.Run());
    }
  }

  void TwoDeviceSwapGraph(Graph* graph) {
    Scope s = Scope::NewRootScope();
    Scope s1 = s.WithDevice("/job:a/replica:0/task:0/device:CPU:0");
    Scope s2 = s.WithDevice("/job:a/replica:0/task:0/device:CPU:1");
    auto x = ops::_Arg(s1.WithOpName("x"), DT_FLOAT, 0);
    auto y = ops::_Arg(s2.WithOpName("y"), DT_FLOAT, 1);
    auto id_x = ops::Identity(s1.WithOpName("id_x"), x);
    auto id_y = ops::Identity(s2.WithOpName("id_y"), y);
    auto dx_retval = ops::_Retval(s2.WithOpName("retval1"), id_y, 0);
    auto dy_retval = ops::_Retval(s1.WithOpName("retval2"), id_x, 1);
    TF_ASSERT_OK(s.ToGraph(graph));
    Placer placer(graph, "", &device_set_, device0_);
    TF_ASSERT_OK(placer.Run());
  }

  // Fills subgraph with an identify function arg->identity->ret
  // where each node has type `dtype` and arg/ret nodes have
  // indices `arg_index` and `ret_index`.
  void SubGraph(Graph* subgraph, DataType dtype,
                gtl::ArraySlice<int> arg_indices,
                gtl::ArraySlice<int> ret_indices) {
    Scope s = Scope::NewRootScope();
    Scope s1 = s.WithDevice("/job:a/replica:0/task:0/device:CPU:0");
    CHECK_EQ(arg_indices.size(), ret_indices.size());
    for (size_t i = 0; i < arg_indices.size(); ++i) {
      auto x = ops::_Arg(s1.WithOpName("x"), dtype, arg_indices[i]);
      auto id_x = ops::Identity(s1.WithOpName("id_x"), x);
      auto dx_retval =
          ops::_Retval(s1.WithOpName("retval1"), id_x, ret_indices[i]);
    }
    TF_ASSERT_OK(s.ToGraph(subgraph));
    Placer placer(subgraph, "", &device_set_, device0_);
    TF_ASSERT_OK(placer.Run());
  }

  std::unique_ptr<DeviceMgr> device_mgr_;
  Device* device0_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  Device* device1_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  DeviceSet device_set_;
};

TEST_F(PartitioningUtilsTest, GraphWithoutAssignedDevicesFails) {
  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  SwapGraph(graph.get());

  std::unordered_map<string, std::unique_ptr<Graph>> subgraphs;
  Status status =
      PartitionFunctionGraph(device_set_, std::move(graph), &subgraphs);
  ASSERT_TRUE(errors::IsInvalidArgument(status)) << status.ToString();
}

TEST_F(PartitioningUtilsTest, OneDevice) {
  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  SwapGraph(graph.get(), true);
  int num_nodes = graph->num_op_nodes();

  std::unordered_map<string, std::unique_ptr<Graph>> subgraphs;
  Status status =
      PartitionFunctionGraph(device_set_, std::move(graph), &subgraphs);
  ASSERT_TRUE(status.ok()) << status.ToString();

  ASSERT_EQ(1, subgraphs.size());
  const auto& pair = *subgraphs.begin();
  ASSERT_EQ("/job:a/replica:0/task:0/device:CPU:0", pair.first);
  ASSERT_EQ(num_nodes, pair.second->num_op_nodes());
}

TEST_F(PartitioningUtilsTest, TwoDevices) {
  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  TwoDeviceSwapGraph(graph.get());

  std::unordered_map<string, std::unique_ptr<Graph>> subgraphs;
  Status status =
      PartitionFunctionGraph(device_set_, std::move(graph), &subgraphs);
  ASSERT_TRUE(status.ok()) << status.ToString();

  ASSERT_EQ(2, subgraphs.size());

  const auto& part1 = subgraphs["/job:a/replica:0/task:0/device:CPU:0"];
  ASSERT_EQ(3, part1->num_op_nodes());
  const auto& part2 = subgraphs["/job:a/replica:0/task:0/device:CPU:1"];
  ASSERT_EQ(3, part2->num_op_nodes());
}

void CheckIndices(const std::vector<int>& expected,
                  const std::vector<int>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (int i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i], actual[i]) << " at index " << i;
  }
}

void CheckAlloc(const std::vector<bool>& expected,
                const std::vector<AllocatorAttributes>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (int i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i], actual[i].on_host()) << " at index " << i;
  }
}

void CheckIndex(const Node& node, int expected_index) {
  const AttrValue* attr_value;
  TF_ASSERT_OK(node.attrs().Find("index", &attr_value));
  int index = static_cast<int>(attr_value->i());
  ASSERT_EQ(expected_index, index);
}

TEST_F(PartitioningUtilsTest, UpdateArgsAndRets) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  SubGraph(graph.get(), DT_FLOAT, {3}, {5});

  std::vector<int> arg_indices;
  std::vector<int> ret_indices;
  std::vector<AllocatorAttributes> arg_alloc_attrs;
  std::vector<AllocatorAttributes> ret_alloc_attrs;

  string device_type = "CPU";

  Status status = UpdateArgAndRetvalMetadata(
      graph.get(), device_type, &arg_indices, &ret_indices, &arg_alloc_attrs,
      &ret_alloc_attrs);
  ASSERT_TRUE(status.ok()) << status.ToString();

  CheckIndices({3}, arg_indices);
  CheckIndices({5}, ret_indices);
  CheckAlloc({false}, arg_alloc_attrs);
  CheckAlloc({false}, ret_alloc_attrs);

  std::unordered_map<string, Node*> nodes = graph->BuildNodeNameIndex();
  ASSERT_EQ(1, nodes.count("x"));
  CheckIndex(*nodes["x"], 0);
  ASSERT_EQ(1, nodes.count("retval1"));
  CheckIndex(*nodes["retval1"], 0);
}

TEST_F(PartitioningUtilsTest, UpdateArgsAndRets_Order) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  SubGraph(graph.get(), DT_FLOAT, {9, 7, 5, 3, 1}, {2, 4, 6, 8, 10});

  std::vector<int> arg_indices;
  std::vector<int> ret_indices;
  std::vector<AllocatorAttributes> arg_alloc_attrs;
  std::vector<AllocatorAttributes> ret_alloc_attrs;

  string device_type = "CPU";

  Status status = UpdateArgAndRetvalMetadata(
      graph.get(), device_type, &arg_indices, &ret_indices, &arg_alloc_attrs,
      &ret_alloc_attrs);
  ASSERT_TRUE(status.ok()) << status.ToString();

  CheckIndices({1, 3, 5, 7, 9}, arg_indices);
  CheckIndices({2, 4, 6, 8, 10}, ret_indices);
  CheckAlloc({false, false, false, false, false}, arg_alloc_attrs);
  CheckAlloc({false, false, false, false, false}, ret_alloc_attrs);
}

}  // anonymous namespace
}  // namespace tensorflow
