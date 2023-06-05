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

#include "tensorflow/core/common_runtime/replicate_per_replica_nodes.h"

#include <map>
#include <vector>

#include "absl/strings/match.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class GraphHelper {
 public:
  explicit GraphHelper(const Graph& graph) : graph_(graph) {
    for (Node* node : graph.nodes()) {
      nodes_by_name_[node->name()] = node;
    }
  }

  Node* GetNodeByName(const string& name) {
    const auto it = nodes_by_name_.find(name);
    if (it != nodes_by_name_.end()) {
      return it->second;
    }
    for (const auto& entry : nodes_by_name_) {
      if (absl::StartsWith(entry.first, name)) {
        return entry.second;
      }
    }
    return nullptr;
  }

  void SetAssignedDevice(const string& node_name, const string& device_name) {
    CHECK_NOTNULL(GetNodeByName(node_name))
        ->set_assigned_device_name(device_name);
  }

  void CheckArgNum(const int expected_num) {
    int arg_num = 0;
    for (Node* node : graph_.op_nodes()) {
      if (node->IsArg()) {
        arg_num++;
      }
    }
    EXPECT_EQ(arg_num, expected_num);
  }

  void CheckAssignedDevice(const string& node_name,
                           const string& expected_device_name) {
    EXPECT_EQ(expected_device_name,
              CHECK_NOTNULL(GetNodeByName(node_name))->assigned_device_name());
  }

  void CheckAssignedDevicePrefix(const string& node_name,
                                 const string& expected_device_name) {
    auto assigned =
        CHECK_NOTNULL(GetNodeByName(node_name))->assigned_device_name();
    EXPECT_EQ(assigned.rfind(expected_device_name, 0), 0);
  }

 private:
  const Graph& graph_;
  // Maps from a node name to a Node* in the graph. We use an ordered map here
  // to ensure stability of GetNodeByName().
  std::map<string, Node*> nodes_by_name_;
};

TEST(ReplicatePerReplicaNodesTest, SingleCompositeDevice) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto read = ops::ReadVariableOp(scope.WithOpName("read"), arg, DT_INT32);
  auto one = ops::Const<int32>(scope.WithOpName("one"), 1);
  auto write = ops::AssignVariableOp(scope.WithOpName("write"), arg, one);
  auto ret = ops::_Retval(
      scope.WithOpName("ret").WithControlDependencies({write}), read, 0);

  const std::vector<string> underlying_devices = {"/device:TPU:0",
                                                  "/device:TPU:1"};
  const absl::flat_hash_map<string, const std::vector<string>*>
      composite_devices = {{"/device:TPU_COMPOSITE:0", &underlying_devices}};

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  {
    // _Arg(TPU_COMPOSITE:0) -> ReadVariableOp(TPU:0);
    // Const(CPU:0) -> AssignVariableOp(TPU_COMPOSITE:0);
    // ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    ASSERT_EQ(graph.num_op_nodes(), 5);
    GraphHelper helper(graph);
    helper.SetAssignedDevice("arg", "/device:TPU_COMPOSITE:0");
    helper.SetAssignedDevice("read", "/device:TPU:0");
    helper.SetAssignedDevice("one", "/device:CPU:0");
    helper.SetAssignedDevice("write", "/device:TPU_COMPOSITE:0");
    helper.SetAssignedDevice("ret", "/device:CPU:0");
  }

  TF_EXPECT_OK(
      ReplicatePerReplicaNodesInFunctionGraph(composite_devices, &graph));

  {
    // _Arg(TPU:0, TPU:1) -> ReadVariableOp(TPU:0);
    // Const(CPU:0) -> AssignVariableOp(TPU:0, TPU:1);
    // ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    EXPECT_EQ(graph.num_op_nodes(), 9);
    GraphHelper helper(graph);
    helper.CheckArgNum(2);
    helper.CheckAssignedDevicePrefix("arg/R0", "/device:TPU");
    helper.CheckAssignedDevicePrefix("arg/R1", "/device:TPU");
    helper.CheckAssignedDevicePrefix("write/R0", "/device:TPU");
    helper.CheckAssignedDevicePrefix("write/R1", "/device:TPU");
    helper.CheckAssignedDevice("read", "/device:TPU:0");
    helper.CheckAssignedDevice("one", "/device:CPU:0");
    helper.CheckAssignedDevice("ret", "/device:CPU:0");
  }
}

TEST(ReplicatePerReplicaNodesTest, SingleCompositeDeviceToSingleDevice) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto read = ops::ReadVariableOp(scope.WithOpName("read"), arg, DT_INT32);
  auto ret = ops::_Retval(scope.WithOpName("ret"), read, 0);

  const std::vector<string> underlying_devices = {"/device:TPU:0"};
  const absl::flat_hash_map<string, const std::vector<string>*>
      composite_devices = {{"/device:TPU_COMPOSITE:0", &underlying_devices}};

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  {
    // _Arg(TPU_COMPOSITE:0) -> ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    ASSERT_EQ(graph.num_op_nodes(), 3);
    GraphHelper helper(graph);
    helper.SetAssignedDevice("arg", "/device:TPU_COMPOSITE:0");
    helper.SetAssignedDevice("read", "/device:TPU:0");
    helper.SetAssignedDevice("ret", "/device:CPU:0");
  }

  TF_EXPECT_OK(
      ReplicatePerReplicaNodesInFunctionGraph(composite_devices, &graph));

  {
    // _Arg(TPU:0) -> ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    EXPECT_EQ(graph.num_op_nodes(), 3);
    GraphHelper helper(graph);
    helper.CheckArgNum(1);
    helper.CheckAssignedDevice("arg", "/device:TPU:0");
    helper.CheckAssignedDevice("read", "/device:TPU:0");
    helper.CheckAssignedDevice("ret", "/device:CPU:0");
  }
}

TEST(ReplicatePerReplicaNodesTest, MultipleCompositeDevices) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_RESOURCE, 0);
  Output arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_RESOURCE, 0);
  auto read0 = ops::ReadVariableOp(scope.WithOpName("read0"), arg0, DT_INT32);
  auto read1 = ops::ReadVariableOp(scope.WithOpName("read1"), arg1, DT_INT32);
  auto identity0 = ops::Identity(scope.WithOpName("identity0"), read0);
  auto identity1 = ops::Identity(scope.WithOpName("identity1"), read1);
  auto add = ops::Add(scope.WithOpName("add"), identity0, identity1);
  auto ret = ops::_Retval(scope.WithOpName("ret"), add, 0);

  const std::vector<string> underlying_devices_0 = {"/device:TPU:0",
                                                    "/device:TPU:1"};
  const std::vector<string> underlying_devices_1 = {"/device:TPU:2",
                                                    "/device:TPU:3"};
  const absl::flat_hash_map<string, const std::vector<string>*>
      composite_devices = {{"/device:TPU_COMPOSITE:0", &underlying_devices_0},
                           {"/device:TPU_COMPOSITE:1", &underlying_devices_1}};

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  {
    // _Arg(TPU_COMPOSITE:0) -> ReadVariableOp(TPU_COMPOSITE:0) ->
    // Identity(TPU:1)
    // _Arg(TPU_COMPOSITE:1) -> ReadVariableOp(TPU_COMPOSITE:1)
    // -> Identity(TPU:3)
    // Identity(TPU:1), Identity(TPU:3) -> Add(TPU:0)-> _Retval(CPU:0)
    ASSERT_EQ(graph.num_op_nodes(), 8);
    GraphHelper helper(graph);
    helper.SetAssignedDevice("arg0", "/device:TPU_COMPOSITE:0");
    helper.SetAssignedDevice("read0", "/device:TPU_COMPOSITE:0");
    helper.SetAssignedDevice("identity0", "/device:TPU:1");
    helper.SetAssignedDevice("arg1", "/device:TPU_COMPOSITE:1");
    helper.SetAssignedDevice("read1", "/device:TPU_COMPOSITE:1");
    helper.SetAssignedDevice("identity1", "/device:TPU:3");
    helper.SetAssignedDevice("add", "/device:TPU:0");
    helper.SetAssignedDevice("ret", "/device:CPU:0");
  }

  TF_EXPECT_OK(
      ReplicatePerReplicaNodesInFunctionGraph(composite_devices, &graph));

  {
    // _Arg(TPU:0, TPU:3) -> ReadVariableOp(TPU:1, TPU:3) -> Identity(TPU:1,
    // TPU:3) -> Add(TPU:0)-> _Retval(CPU:0)
    EXPECT_EQ(graph.num_op_nodes(), 8);
    GraphHelper helper(graph);
    helper.CheckArgNum(2);
    helper.CheckAssignedDevice("arg0/R1", "/device:TPU:1");
    helper.CheckAssignedDevice("arg1/R1", "/device:TPU:3");
    helper.CheckAssignedDevice("read0/R1", "/device:TPU:1");
    helper.CheckAssignedDevice("read1/R1", "/device:TPU:3");
    helper.CheckAssignedDevice("identity0", "/device:TPU:1");
    helper.CheckAssignedDevice("identity1", "/device:TPU:3");
    helper.CheckAssignedDevice("add", "/device:TPU:0");
    helper.CheckAssignedDevice("ret", "/device:CPU:0");
  }
}

TEST(ReplicatePerReplicaNodesTest, NestedFunctions) {
  const std::vector<string> underlying_devices = {"/device:TPU:0",
                                                  "/device:TPU:1"};
  const absl::flat_hash_map<string, const std::vector<string>*>
      composite_devices = {{"/device:TPU_COMPOSITE:0", &underlying_devices}};

  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
    auto read = ops::ReadVariableOp(scope.WithOpName("read"), arg, DT_INT32);
    auto ret = ops::_Retval(scope.WithOpName("ret"), read, 0);
    Graph graph(OpRegistry::Global());
    TF_ASSERT_OK(scope.ToGraph(&graph));
    GraphHelper helper(graph);
    helper.SetAssignedDevice("arg", "/device:TPU_COMPOSITE:0");
    helper.SetAssignedDevice("read", "/device:TPU:0");
    helper.SetAssignedDevice("ret", "/device:CPU:0");
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(graph, "Func", &fdef));
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));
  }

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));
  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("func", "Func", &flib_def)
                   .Input(arg.name(), 0, DT_RESOURCE)
                   .Finalize(&def));
  Status status;
  Node* func = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(arg.node(), 0, func, 0);
  auto ret = ops::_Retval(scope.WithOpName("ret"), Output(func), 0);
  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  {
    // _Arg(TPU_COMPOSITE:0) -> Func(CPU:0) -> _Retval(CPU:0)
    GraphHelper helper(graph);
    EXPECT_EQ(graph.num_op_nodes(), 3);
    helper.SetAssignedDevice("arg", "/device:TPU_COMPOSITE:0");
    helper.SetAssignedDevice("func", "/device:CPU:0");
    helper.SetAssignedDevice("ret", "/device:CPU:0");
  }

  TF_EXPECT_OK(
      ReplicatePerReplicaNodesInFunctionGraph(composite_devices, &graph));

  {
    // _Arg(TPU:0), _Arg(TPU:1) -> Pack(CPU:0) -> Func(CPU:0) -> _Retval(CPU:0)
    EXPECT_EQ(graph.num_op_nodes(), 5);
    GraphHelper helper(graph);
    helper.CheckArgNum(2);
    helper.CheckAssignedDevice("arg/R0", "/device:TPU:0");
    helper.CheckAssignedDevice("arg/R1", "/device:TPU:1");
    helper.CheckAssignedDevice("arg/Packed", "/device:CPU:0");
    helper.CheckAssignedDevice("func", "/device:CPU:0");
    helper.CheckAssignedDevice("ret", "/device:CPU:0");
    const EdgeSet& packed_in_edges =
        helper.GetNodeByName("arg/Packed")->in_edges();
    EXPECT_EQ(packed_in_edges.size(), 2);
    auto it = packed_in_edges.begin();
    EXPECT_EQ(helper.GetNodeByName("arg/R0"), (*it++)->src());
    EXPECT_EQ(helper.GetNodeByName("arg/R1"), (*it)->src());
    const EdgeSet& func_in_edges = helper.GetNodeByName("func")->in_edges();
    EXPECT_EQ(func_in_edges.size(), 1);
    EXPECT_EQ(helper.GetNodeByName("arg/Packed"),
              (*func_in_edges.begin())->src());
  }
}

TEST(ReplicatePerReplicaNodesTest, DeadArgNodes) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto read = ops::ReadVariableOp(scope.WithOpName("read"), arg, DT_INT32);
  auto ret = ops::_Retval(scope.WithOpName("ret"), read, 0);

  const std::vector<string> underlying_devices = {"/device:TPU:0",
                                                  "/device:TPU:1"};
  const absl::flat_hash_map<string, const std::vector<string>*>
      composite_devices = {{"/device:TPU_COMPOSITE:0", &underlying_devices}};

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  {
    // _Arg(TPU_COMPOSITE:0) -> ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    ASSERT_EQ(graph.num_op_nodes(), 3);
    GraphHelper helper(graph);
    helper.SetAssignedDevice("arg", "/device:TPU_COMPOSITE:0");
    helper.SetAssignedDevice("read", "/device:TPU:0");
    helper.SetAssignedDevice("ret", "/device:CPU:0");
  }

  TF_EXPECT_OK(
      ReplicatePerReplicaNodesInFunctionGraph(composite_devices, &graph));

  {
    // _Arg(TPU:0) -> ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    // "arg/R1" is a dead node, so gets removed.
    EXPECT_EQ(graph.num_op_nodes(), 3);
    GraphHelper helper(graph);
    helper.CheckArgNum(1);
    helper.CheckAssignedDevice("arg/R0", "/device:TPU:0");
    helper.CheckAssignedDevice("read", "/device:TPU:0");
    helper.CheckAssignedDevice("ret", "/device:CPU:0");
  }
}

}  // namespace
}  // namespace tensorflow
