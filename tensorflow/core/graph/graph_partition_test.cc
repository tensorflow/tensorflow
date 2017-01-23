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

#include "tensorflow/core/graph/graph_partition.h"

#include <unordered_map>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/equal_graph_def.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

const char gpu_device[] = "/job:a/replica:0/task:0/gpu:0";

string SplitByDevice(const Node* node) { return node->assigned_device_name(); }

string DeviceName(const Node* node) {
  char first = node->name()[0];
  if (first == 'G') {
    return gpu_device;
  } else {
    const string cpu_prefix = "/job:a/replica:0/task:0/cpu:";
    int index = first - 'A';
    return strings::StrCat(cpu_prefix, index);
  }
}

void Partition(const GraphDef& graph_def,
               std::unordered_map<string, GraphDef>* partitions) {
  Graph g(OpRegistry::Global());
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, &g));

  // Assigns devices to each node. Uses 1st letter of the node name as
  // the device index.
  for (Node* node : g.nodes()) {
    node->set_assigned_device_name(DeviceName(node));
  }

  PartitionOptions popts;
  popts.node_to_loc = SplitByDevice;
  popts.new_name = [&g](const string& prefix) { return g.NewName(prefix); };
  popts.get_incarnation = [](const string& name) {
    return (name[0] - 'A') + 100;
  };
  Status s = Partition(popts, &g, partitions);
  CHECK(s.ok()) << s;

  // Check versions
  EXPECT_EQ(graph_def.versions().producer(), TF_GRAPH_DEF_VERSION);
  EXPECT_EQ(graph_def.versions().min_consumer(),
            TF_GRAPH_DEF_VERSION_MIN_CONSUMER);
  for (auto& it : *partitions) {
    EXPECT_EQ(graph_def.versions().producer(), it.second.versions().producer());
    EXPECT_EQ(graph_def.versions().min_consumer(),
              it.second.versions().min_consumer());
  }
}

void CheckLoopConstruction(const GraphDef& graph_def) {
  std::unordered_map<string, GraphDef> partitions;
  Partition(graph_def, &partitions);
  for (const auto& kv : partitions) {
    const GraphDef& gdef = kv.second;
    bool has_control_enter = false;
    bool has_control_merge = false;
    bool has_control_switch = false;
    bool has_control_next = false;
    for (const NodeDef& ndef : gdef.node()) {
      // _recvs must have a control input
      if (ndef.op() == "_Recv") {
        bool has_control = false;
        for (const string& input_name : ndef.input()) {
          if (StringPiece(input_name).starts_with("^")) {
            has_control = true;
            break;
          }
        }
        EXPECT_TRUE(has_control);
      }
      // Must have a control loop
      if (StringPiece(ndef.name()).starts_with("_cloop")) {
        if (ndef.op() == "Enter") {
          has_control_enter = true;
        }
        if (ndef.op() == "Merge") {
          has_control_merge = true;
        }
        if (ndef.op() == "Switch") {
          has_control_switch = true;
        }
        if (ndef.op() == "NextIteration") {
          has_control_next = true;
        }
      }
    }
    EXPECT_TRUE(has_control_enter);
    EXPECT_TRUE(has_control_merge);
    EXPECT_TRUE(has_control_switch);
    EXPECT_TRUE(has_control_next);
  }
}

REGISTER_OP("FloatInput").Output("o: float");
REGISTER_OP("BoolInput").Output("o: bool");
REGISTER_OP("Combine").Input("a: float").Input("b: float").Output("o: float");

Output ConstructOp(const Scope& scope, const string& op_type,
                   const gtl::ArraySlice<Input>& inputs) {
  if (!scope.ok()) return Output();
  const string unique_name = scope.GetUniqueNameForOp(op_type);
  auto builder = NodeBuilder(unique_name, op_type);
  for (auto const& input : inputs) {
    builder.Input(ops::NodeOut(input.node(), input.index()));
  }
  scope.UpdateBuilder(&builder);
  Node* ret;
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return Output();
  return Output(ret);
}

Output FloatInput(const Scope& scope) {
  return ConstructOp(scope, "FloatInput", {});
}

Output BoolInput(const Scope& scope) {
  return ConstructOp(scope, "BoolInput", {});
}

Output Combine(const Scope& scope, Input a, Input b) {
  return ConstructOp(scope, "Combine", {a, b});
}

class GraphPartitionTest : public ::testing::Test {
 protected:
  GraphPartitionTest()
      : in_(Scope::NewRootScope().ExitOnError()),
        scope_a_(Scope::NewRootScope().ExitOnError().WithDevice(
            "/job:a/replica:0/task:0/cpu:0")),
        scope_b_(Scope::NewRootScope().ExitOnError().WithDevice(
            "/job:a/replica:0/task:0/cpu:1")) {}

  const GraphDef& ToGraphDef() {
    in_.ToGraphDef(&in_graph_def_);
    return in_graph_def_;
  }

  void ExpectMatchA() {
    GraphDef graph_def;
    scope_a_.ToGraphDef(&graph_def);
    string a = "/job:a/replica:0/task:0/cpu:0";
    TF_EXPECT_GRAPH_EQ(graph_def, partitions_[a]);
  }

  void ExpectMatchB() {
    GraphDef graph_def;
    scope_b_.ToGraphDef(&graph_def);
    string b = "/job:a/replica:0/task:0/cpu:1";
    TF_EXPECT_GRAPH_EQ(graph_def, partitions_[b]);
  }

  Scope in_;
  GraphDef in_graph_def_;
  Scope scope_a_;
  Scope scope_b_;
  std::unordered_map<string, GraphDef> partitions_;
};

TEST_F(GraphPartitionTest, SingleDevice) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  auto a1 = FloatInput(in_.WithOpName("A1"));
  Combine(in_.WithOpName("A2"), a1, a1);

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(1, partitions_.size());

  a1 = FloatInput(scope_a_.WithOpName("A1"));
  Combine(scope_a_.WithOpName("A2"), a1, a1);
  ExpectMatchA();
}

TEST_F(GraphPartitionTest, CrossDeviceData) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2"), a1, b1);

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  _Send(scope_a_.WithOpName("A1/_0"), a1, "edge_1_A1", a, 82, b);
  ExpectMatchA();

  b1 = FloatInput(scope_b_.WithOpName("B1"));
  auto recv =
      _Recv(scope_b_.WithOpName("A1/_1"), DT_FLOAT, "edge_1_A1", a, 82, b);
  Combine(scope_b_.WithOpName("B2"), recv, b1);
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceControl) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2").WithControlDependencies(a1), b1, b1);

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  auto c = Const(scope_a_.WithOpName("A1/_0").WithControlDependencies(a1), {});
  _Send(scope_a_.WithOpName("A1/_1"), c, "edge_3_A1", a, 82, b);
  ExpectMatchA();

  auto recv =
      _Recv(scope_b_.WithOpName("A1/_2"), DT_FLOAT, "edge_3_A1", a, 82, b);
  auto id = Identity(scope_b_.WithOpName("A1/_3"), recv);
  b1 = FloatInput(scope_b_.WithOpName("B1"));
  Combine(scope_b_.WithOpName("B2").WithControlDependencies(id), b1, b1);
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceData_MultiUse) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2"), a1, b1);
  Combine(in_.WithOpName("B3"), a1, a1);

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  _Send(scope_a_.WithOpName("A1/_0"), a1, "edge_1_A1", a, 82, b);
  ExpectMatchA();

  auto recv =
      _Recv(scope_b_.WithOpName("A1/_1"), DT_FLOAT, "edge_1_A1", a, 82, b);
  b1 = FloatInput(scope_b_.WithOpName("B1"));
  Combine(scope_b_.WithOpName("B2"), recv, b1);
  Combine(scope_b_.WithOpName("B3"), recv, recv);
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceControl_MultiUse) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2").WithControlDependencies(a1), b1, b1);
  FloatInput(in_.WithOpName("B3").WithControlDependencies(a1));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  auto c = Const(scope_a_.WithOpName("A1/_0").WithControlDependencies(a1), {});
  _Send(scope_a_.WithOpName("A1/_1"), c, "edge_1_A1", a, 82, b);
  ExpectMatchA();

  auto recv =
      _Recv(scope_b_.WithOpName("A1/_2"), DT_FLOAT, "edge_1_A1", a, 82, b);
  auto id = Identity(scope_b_.WithOpName("A1/_3"), recv);
  b1 = FloatInput(scope_b_.WithOpName("B1"));
  Combine(scope_b_.WithOpName("B2").WithControlDependencies(id), b1, b1);
  FloatInput(scope_b_.WithOpName("B3").WithControlDependencies(id));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDevice_DataControl) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2"), a1, b1);
  FloatInput(in_.WithOpName("B3").WithControlDependencies(a1));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  auto c = Const(scope_a_.WithOpName("A1/_0").WithControlDependencies(a1), {});
  // NOTE: Send 0 A1/_1 -> A1/_2 is not necessarily needed. We could
  // use A1/_0 -> A1/_4 as the control as a minor optimization.
  _Send(scope_a_.WithOpName("A1/_1"), c, "edge_1_A1", a, 82, b);
  _Send(scope_a_.WithOpName("A1/_4"), a1, "edge_2_A1", a, 82, b);
  ExpectMatchA();

  auto recv1 =
      _Recv(scope_b_.WithOpName("A1/_2"), DT_FLOAT, "edge_1_A1", a, 82, b);
  auto id1 = Identity(scope_b_.WithOpName("A1/_3"), recv1);
  auto recv2 =
      _Recv(scope_b_.WithOpName("A1/_5"), DT_FLOAT, "edge_2_A1", a, 82, b);
  b1 = FloatInput(scope_b_.WithOpName("B1"));
  Combine(scope_b_.WithOpName("B2"), recv2, b1);
  FloatInput(scope_b_.WithOpName("B3").WithControlDependencies(id1));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceLoop) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  auto a1 = BoolInput(in_.WithOpName("A1"));
  auto a2 = Enter(in_.WithOpName("A2"), a1, "foo");
  auto a3 = Merge(in_.WithOpName("A3"), {a2, Input("A5", 0, DT_BOOL)}).output;
  LoopCond(in_.WithOpName("A4"), a3);
  auto b1 = Identity(in_.WithOpName("B1"), a3);
  NextIteration(in_.WithOpName("A5"), b1);

  CheckLoopConstruction(ToGraphDef());
}

TEST_F(GraphPartitionTest, CrossDeviceLoop1) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  auto a1 = BoolInput(in_.WithOpName("A1"));
  auto a2 = Enter(in_.WithOpName("B2"), a1, "foo");
  auto a3 = Merge(in_.WithOpName("A3"), {a2, Input("B5", 0, DT_BOOL)}).output;
  LoopCond(in_.WithOpName("A4"), a3);
  auto b1 = Identity(in_.WithOpName("B1"), a3);
  NextIteration(in_.WithOpName("B5"), b1);

  std::unordered_map<string, GraphDef> partitions;
  Partition(ToGraphDef(), &partitions);
  for (const auto& kv : partitions) {
    const GraphDef& gdef = kv.second;
    for (const NodeDef& ndef : gdef.node()) {
      if (ndef.name() == "A3") {
        // A3, B2, and B5 are on the same device.
        EXPECT_EQ(ndef.input(0), "B2");
        EXPECT_EQ(ndef.input(1), "B5");
      }
    }
  }
}

TEST_F(GraphPartitionTest, PartitionIncompleteGraph) {
  NodeDef ndef;
  Graph g(OpRegistry::Global());
  // Invalid graph since the Combine node requires an input.
  bool parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
      name: "N"
      op: "Combine"
      )EOF",
      &ndef);
  ASSERT_TRUE(parsed);
  Status status;
  g.AddNode(ndef, &status);
  TF_ASSERT_OK(status);

  PartitionOptions popts;
  popts.node_to_loc = SplitByDevice;
  popts.new_name = [&g](const string& prefix) { return g.NewName(prefix); };
  popts.get_incarnation = [](const string&) { return 1; };

  std::unordered_map<string, GraphDef> partitions;
  status = Partition(popts, &g, &partitions);
  // Partitioning should fail, but not crash like it did before the
  // changes that accompanied the addition of this test.
  EXPECT_EQ(error::INVALID_ARGUMENT, status.code()) << status;
}

}  // namespace
}  // namespace tensorflow
