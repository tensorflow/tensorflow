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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_debug_info_builder.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {

// from graph_partition.cc
extern absl::Status TopologicalSortNodesWithTimePriority(
    const GraphDef* gdef,
    std::vector<std::pair<const NodeDef*, int64_t>>* nodes,
    std::unordered_map<const NodeDef*, int64_t>* node_to_start_time_out);

namespace {

using ops::_Recv;
using ops::_Send;
using ops::Const;
using ops::Identity;
using ops::LoopCond;
using ops::NextIteration;
using ::testing::Eq;
using ::testing::Ne;

const char gpu_device[] = "/job:a/replica:0/task:0/device:GPU:0";

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

  // Assigns devices to each node. Uses 1st letter of the node name as the
  // device index if no device is specified.
  for (Node* node : g.nodes()) {
    string device_name = !node->requested_device().empty()
                             ? node->requested_device()
                             : DeviceName(node);
    node->set_assigned_device_name(device_name);
  }

  PartitionOptions popts;
  popts.node_to_loc = SplitByDevice;
  popts.new_name = [&g](const string& prefix) { return g.NewName(prefix); };
  popts.get_incarnation = [](const string& name) {
    return (name[0] - 'A') + 100;
  };
  absl::Status s = Partition(popts, &g, partitions);
  CHECK(s.ok()) << s;

  // Check versions.
  EXPECT_EQ(graph_def.versions().producer(), TF_GRAPH_DEF_VERSION);
  // Partitions must inherit the versions of the original graph.
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
          if (absl::StartsWith(input_name, "^")) {
            has_control = true;
            break;
          }
        }
        EXPECT_TRUE(has_control);
      }
      // Must have a control loop
      if (absl::StartsWith(ndef.name(), "_cloop")) {
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

REGISTER_OP("FloatInput")
    .Output("o: float")
    .SetShapeFn(shape_inference::UnknownShape);
REGISTER_OP("BoolInput")
    .Output("o: bool")
    .SetShapeFn(shape_inference::UnknownShape);
REGISTER_OP("Combine")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float")
    .SetShapeFn(shape_inference::UnknownShape);

Output ConstructOp(const Scope& scope, const string& op_type,
                   const absl::Span<const Input>& inputs) {
  if (!scope.ok()) return Output();
  const string unique_name = scope.GetUniqueNameForOp(op_type);
  auto builder =
      NodeBuilder(unique_name, op_type, scope.graph()->op_registry());
  for (auto const& input : inputs) {
    builder.Input(ops::NodeOut(input.node(), input.index()));
  }
  scope.UpdateBuilder(&builder);
  Node* ret;
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return Output();
  scope.UpdateStatus(scope.DoShapeInference(ret));
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
  return ConstructOp(scope, "Combine", {std::move(a), std::move(b)});
}

std::string FormatStackTrace(const GraphDebugInfo::StackTrace& stack_trace,
                             const GraphDebugInfo& debug_info) {
  std::string result;
  for (const GraphDebugInfo::FileLineCol& file_line_col :
       stack_trace.file_line_cols()) {
    const std::string& file = debug_info.files(file_line_col.file_index());
    absl::StrAppend(&result, file_line_col.func(), "@", file, ":",
                    file_line_col.line(), ".", file_line_col.col(), "\n");
  }
  return result;
}

class GraphPartitionTest : public ::testing::Test {
 protected:
  GraphPartitionTest()
      : in_(Scope::NewRootScope().ExitOnError()),
        scope_a_(Scope::NewRootScope().ExitOnError().WithDevice(
            "/job:a/replica:0/task:0/cpu:0")),
        scope_b_(Scope::NewRootScope().ExitOnError().WithDevice(
            "/job:a/replica:0/task:0/cpu:1")) {}

  const GraphDef& ToGraphDef(bool include_debug_info = false) {
    TF_EXPECT_OK(in_.ToGraphDef(&in_graph_def_, include_debug_info));
    return in_graph_def_;
  }

  void ExpectMatchA() {
    GraphDef graph_def;
    TF_EXPECT_OK(scope_a_.ToGraphDef(&graph_def));
    string a = "/job:a/replica:0/task:0/cpu:0";
    TF_EXPECT_GRAPH_EQ(graph_def, partitions_[a]);
  }

  void ExpectMatchB() {
    GraphDef graph_def;
    TF_EXPECT_OK(scope_b_.ToGraphDef(&graph_def));
    string b = "/job:a/replica:0/task:0/cpu:1";
    TF_EXPECT_GRAPH_EQ(graph_def, partitions_[b]);
  }

  void ExpectFunctions(const FunctionDefLibrary& library,
                       const std::set<string>& expected_names) {
    std::set<string> actual_names;
    for (const FunctionDef& fdef : library.function()) {
      actual_names.insert(fdef.signature().name());
    }
    EXPECT_EQ(actual_names, expected_names);
  }

  Scope in_;
  GraphDef in_graph_def_;
  Scope scope_a_;
  Scope scope_b_;
  std::unordered_map<string, GraphDef> partitions_;
};

TEST_F(GraphPartitionTest, SingleDevice) {
  auto a1 = FloatInput(in_.WithOpName("A1"));
  Combine(in_.WithOpName("A2"), a1, a1);

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(1, partitions_.size());

  a1 = FloatInput(scope_a_.WithOpName("A1"));
  Combine(scope_a_.WithOpName("A2"), a1, a1);
  ExpectMatchA();
}

TEST_F(GraphPartitionTest, CrossDeviceData) {
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
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2").WithControlDependencies(a1), b1, b1);

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  auto c =
      Const(scope_a_.WithOpName("A1/ctrl/_0").WithControlDependencies(a1), {});
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
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2").WithControlDependencies(a1), b1, b1);
  FloatInput(in_.WithOpName("B3").WithControlDependencies(a1));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  auto c =
      Const(scope_a_.WithOpName("A1/ctrl/_0").WithControlDependencies(a1), {});
  _Send(scope_a_.WithOpName("A1/_1"), c, "edge_3_A1", a, 82, b);
  ExpectMatchA();

  auto recv =
      _Recv(scope_b_.WithOpName("A1/_2"), DT_FLOAT, "edge_3_A1", a, 82, b);
  auto id = Identity(scope_b_.WithOpName("A1/_3"), recv);
  b1 = FloatInput(scope_b_.WithOpName("B1"));
  Combine(scope_b_.WithOpName("B2").WithControlDependencies(id), b1, b1);
  FloatInput(scope_b_.WithOpName("B3").WithControlDependencies(id));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDevice_DataControl) {
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2"), a1, b1);
  FloatInput(in_.WithOpName("B3").WithControlDependencies(a1));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  _Send(scope_a_.WithOpName("A1/_0"), a1, "edge_1_A1", a, 82, b);
  auto c =
      Const(scope_a_.WithOpName("A1/ctrl/_2").WithControlDependencies(a1), {});
  // NOTE: Send 0 A1/_1 -> A1/_2 is not necessarily needed. We could
  // use A1/_0 -> A1/_4 as the control as a minor optimization.
  _Send(scope_a_.WithOpName("A1/_3"), c, "edge_3_A1", a, 82, b);
  ExpectMatchA();

  auto recv1 =
      _Recv(scope_b_.WithOpName("A1/_4"), DT_FLOAT, "edge_3_A1", a, 82, b);
  auto id1 = Identity(scope_b_.WithOpName("A1/_5"), recv1);
  auto recv2 =
      _Recv(scope_b_.WithOpName("A1/_1"), DT_FLOAT, "edge_1_A1", a, 82, b);
  b1 = FloatInput(scope_b_.WithOpName("B1"));
  Combine(scope_b_.WithOpName("B2"), recv2, b1);
  FloatInput(scope_b_.WithOpName("B3").WithControlDependencies(id1));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceLoopSimple) {
  auto a1 = BoolInput(in_.WithOpName("A1"));
  auto a2 = ::tensorflow::ops::internal::Enter(in_.WithOpName("A2"), a1, "foo");
  auto a3 = ::tensorflow::ops::Merge(in_.WithOpName("A3"),
                                     {a2, Input("A5", 0, DT_BOOL)})
                .output;
  LoopCond(in_.WithOpName("A4"), a3);
  auto b1 = Identity(in_.WithOpName("B1"), a3);
  NextIteration(in_.WithOpName("A5"), b1);

  CheckLoopConstruction(ToGraphDef());
}

TEST_F(GraphPartitionTest, CrossDeviceLoopSimple1) {
  auto a1 = BoolInput(in_.WithOpName("A1"));
  auto a2 = ::tensorflow::ops::internal::Enter(in_.WithOpName("B2"), a1, "foo");
  auto a3 = ::tensorflow::ops::Merge(in_.WithOpName("A3"),
                                     {a2, Input("B5", 0, DT_BOOL)})
                .output;
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

TEST_F(GraphPartitionTest, CrossDeviceLoopFull) {
  Scope cpu0 = in_.WithDevice("/job:a/replica:0/task:0/cpu:0");
  auto p1 = ops::Placeholder(cpu0, DT_INT32);
  auto p2 = ops::Placeholder(cpu0, DT_INT32);
  OutputList outputs;
  // while i1 < 10: i1 += i2
  TF_ASSERT_OK(ops::BuildWhileLoop(
      cpu0, {p1, p2},
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::Less(s, inputs[0], 10);
        return s.status();
      },
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) {
        Scope cpu1 = s.WithDevice("/job:a/replica:0/task:0/cpu:1");
        outputs->push_back(ops::AddN(cpu1, {inputs[0], inputs[1]}));
        outputs->push_back(inputs[1]);
        return s.status();
      },
      "test_loop", &outputs));
  CheckLoopConstruction(ToGraphDef());
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
  absl::Status status;
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

TEST_F(GraphPartitionTest, Functions) {
  FunctionDefLibrary fdef_lib;
  *fdef_lib.add_function() = test::function::XTimesTwo();
  *fdef_lib.add_function() = test::function::XTimesFour();
  TF_ASSERT_OK(in_.graph()->AddFunctionLibrary(fdef_lib));

  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  ConstructOp(in_.WithOpName("A2"), "XTimesTwo", {a1});
  ConstructOp(in_.WithOpName("B2"), "XTimesFour", {b1});

  // The `Partition()` helper function uses the first letter of the op name ('A'
  // or 'B') to choose a device for each node.
  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  // Test that partition graphs inherit function library from original graph.
  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";

  // Node "A2" is placed in part `a`, and uses only "XTimesTwo".
  ExpectFunctions(partitions_[a].library(), {"XTimesTwo"});
  // Node "B2" is placed in part `b`, and uses both "XTimesFour" directly,
  // and "XTimesTwo" in the body of "XTimesFour".
  ExpectFunctions(partitions_[b].library(), {"XTimesTwo", "XTimesFour"});
}

TEST_F(GraphPartitionTest, SetIncarnation) {
  GraphDef gdef;
  const char* const kSendRecvAttrs = R"pb(
    attr {
      key: 'T'
      value { type: DT_FLOAT }
    }
    attr {
      key: 'client_terminated'
      value { b: false }
    }
    attr {
      key: 'recv_device'
      value { s: 'B' }
    }
    attr {
      key: 'send_device'
      value { s: 'A' }
    }
    attr {
      key: 'send_device_incarnation'
      value { i: 0 }
    }
    attr {
      key: 'tensor_name'
      value { s: 'test' }
    }
  )pb";
  CHECK(protobuf::TextFormat::ParseFromString(
      strings::StrCat(
          "node { name: 'A/Pi' op: 'Const' ",
          "  attr { key: 'dtype' value { type: DT_FLOAT } } ",
          "  attr { key: 'value' value { tensor { ",
          "    dtype: DT_FLOAT tensor_shape {} float_val: 3.14 } } } }",
          "node { name: 'A' op: '_Send' input: 'A/Pi' ", kSendRecvAttrs, "}",
          "node { name: 'B' op: '_Recv' ", kSendRecvAttrs,
          "  attr { key: 'tensor_type' value { type:DT_FLOAT}}}"),
      &gdef));
  gdef.mutable_versions()->set_producer(TF_GRAPH_DEF_VERSION);
  Partition(gdef, &partitions_);
  EXPECT_EQ(2, partitions_.size());

  for (const auto& kv : partitions_) {
    const GraphDef& gdef = kv.second;
    for (const NodeDef& ndef : gdef.node()) {
      if (ndef.name() == "A" || ndef.name() == "B") {
        int64_t val;
        TF_CHECK_OK(GetNodeAttr(ndef, "send_device_incarnation", &val));
        EXPECT_EQ(val, 100);  // Send device is "A".
      }
    }
  }
}

TEST_F(GraphPartitionTest, GraphDebugInfo) {
  GraphDef graph_def;
  Output a1 = FloatInput(in_.WithOpName("A1"));
  Output b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2"), a1, b1);

  Node *a1_node = nullptr, *b1_node = nullptr, *b2_node = nullptr;
  for (Node* node : in_.graph()->op_nodes()) {
    if (node->name() == "A1") {
      a1_node = node;
    } else if (node->name() == "B1") {
      b1_node = node;
    } else if (node->name() == "B2") {
      b2_node = node;
    }
  }
  EXPECT_NE(a1_node, nullptr);
  EXPECT_NE(b1_node, nullptr);
  EXPECT_NE(b2_node, nullptr);

  std::vector<StackFrame> a1_stack_trace{{"main.cc", 20, "x"},
                                         {"alpha.cc", 30, "a1"}};
  std::vector<StackFrame> b1_stack_trace{{"window.cc", 21, "y"},
                                         {"beta.cc", 35, "b1"}};
  std::vector<StackFrame> b2_stack_trace{{"cache.cc", 22, "bar"},
                                         {"beta.cc", 39, "b2"}};
  a1_node->SetStackTrace(std::make_shared<FrozenStackTrace>(a1_stack_trace));
  b1_node->SetStackTrace(std::make_shared<FrozenStackTrace>(b1_stack_trace));
  b2_node->SetStackTrace(std::make_shared<FrozenStackTrace>(b2_stack_trace));

  TF_EXPECT_OK(in_.ToGraphDef(&graph_def, /*include_debug_info=*/true));

  // `Partition()` uses the first letter of the op name ('A' or 'B') to choose a
  // device for each node. It calls the function under test, also named
  // `Partition()`, to do the actual partitioning.
  Partition(ToGraphDef(/*include_debug_info=*/true), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  // Expect each partitioned graph to contain the stack traces for its nodes.
  // A stack trace for A1 should be in the A partition (".../cpu:0").
  string a = "/job:a/replica:0/task:0/cpu:0";
  const GraphDebugInfo& a_debug_info = partitions_[a].debug_info();
  StackTracesMap traces = LoadTracesFromDebugInfo(a_debug_info);
  const auto& a_it = traces.find("A1");
  EXPECT_THAT(a_it, Ne(traces.end()));
  EXPECT_THAT(a_it->second->ToString({}),
              ::testing::ContainsRegex("alpha.cc.*30"));

  // Stack traces for B1 and B2 should be in the B partition (".../cpu:1").
  string b = "/job:a/replica:0/task:0/cpu:1";
  const GraphDebugInfo& b_debug_info = partitions_[b].debug_info();
  traces = LoadTracesFromDebugInfo(b_debug_info);
  const auto& b1_it = traces.find("B1");
  const auto& b2_it = traces.find("B2");
  EXPECT_THAT(b1_it, Ne(traces.end()));
  EXPECT_THAT(b2_it, Ne(traces.end()));
  EXPECT_THAT(b1_it->second->ToString({}),
              ::testing::ContainsRegex("beta.cc.*35"));
  EXPECT_THAT(b2_it->second->ToString({}),
              ::testing::ContainsRegex("beta.cc.*39"));
}

TEST(TopologicalSortNodesWithTimePriorityTest, NoDependencies) {
  // Create placeholders, shuffle them so the order in the graph is not strictly
  // increasing.
  Scope root = Scope::NewRootScope().ExitOnError();
  std::vector<int> indexes;
  for (int i = 0; i < 20; ++i) {
    indexes.push_back((i + 2001) % 20);
  }
  std::vector<ops::Placeholder> placeholders;
  for (int i : indexes) {
    placeholders.emplace_back(root.WithOpName(strings::StrCat("p", i)),
                              DT_FLOAT);
    placeholders.back().node()->AddAttr("_start_time", i + 1);
  }

  GraphDef gdef;
  TF_EXPECT_OK(root.ToGraphDef(&gdef));

  std::vector<std::pair<const NodeDef*, int64_t>> nodes;
  std::unordered_map<const NodeDef*, int64_t> node_to_start_time;
  TF_CHECK_OK(
      TopologicalSortNodesWithTimePriority(&gdef, &nodes, &node_to_start_time));
  ASSERT_EQ(nodes.size(), 20);
  for (int i = 0; i < nodes.size(); ++i) {
    EXPECT_EQ(strings::StrCat("p", i), nodes[i].first->name());
    EXPECT_EQ(i + 1, nodes[i].second);
  }
}

TEST(TopologicalSortNodesWithTimePriority, Dependencies) {
  // Create placeholders, shuffle them so the order in the graph is not strictly
  // increasing.
  Scope root = Scope::NewRootScope().ExitOnError();
  std::vector<int> indexes;
  std::vector<ops::Placeholder> placeholders_in_order;
  const int num_leaves = 20;
  for (int i = 0; i < num_leaves; ++i) {
    indexes.push_back((i + 2001) % num_leaves);
    placeholders_in_order.emplace_back(root.WithOpName(strings::StrCat("p", i)),
                                       DT_FLOAT);
    placeholders_in_order.back().node()->AddAttr("_start_time", i + 1);
  }
  std::vector<ops::Placeholder> placeholders;
  for (int i : indexes) {
    placeholders.push_back(placeholders_in_order[i]);
  }

  // Create ops that depend on the placeholders. We give start times to these
  // that are in descending order (e.g., the op that depends on the first
  // placeholder runs last).
  std::vector<ops::Square> squares;
  for (int i : indexes) {
    squares.emplace_back(root.WithOpName(strings::StrCat("s", i)),
                         placeholders[i]);
    squares.back().node()->AddAttr("_start_time", 50 - (i + 1));
  }

  // Create addn to sum all squares.
  std::vector<Input> inputs;
  for (const auto& s : squares) inputs.push_back(s);
  ops::AddN addn =
      ops::AddN(root.WithOpName("addn"), absl::Span<const Input>(inputs));
  // Start times is actually listed earlier than the nodes it depends on.
  // But because of dependency ordering, it is last in the list.
  addn.node()->AddAttr("_start_time", 1);

  GraphDef gdef;
  TF_EXPECT_OK(root.ToGraphDef(&gdef));

  std::vector<std::pair<const NodeDef*, int64_t>> nodes;
  std::unordered_map<const NodeDef*, int64_t> node_to_start_time;
  TF_CHECK_OK(
      TopologicalSortNodesWithTimePriority(&gdef, &nodes, &node_to_start_time));
  ASSERT_EQ(1 + squares.size() + placeholders.size(), nodes.size());
  for (int i = 0; i < placeholders.size(); ++i) {
    const NodeDef* node = nodes[i].first;
    EXPECT_EQ(strings::StrCat("p", i), node->name());
    EXPECT_EQ(i + 1, nodes[i].second);
    EXPECT_EQ(i + 1, node_to_start_time[node]);
  }
  for (int i = 0; i < squares.size(); ++i) {
    int node_index = placeholders.size() + i;
    int square_index = num_leaves - 1 - i;
    const NodeDef* node = nodes[node_index].first;
    EXPECT_EQ(strings::StrCat("s", square_index), node->name());
    EXPECT_EQ(50 - (square_index + 1), nodes[node_index].second);
    EXPECT_EQ(50 - (square_index + 1), node_to_start_time[node]);
  }
  EXPECT_EQ("addn", nodes.back().first->name());
  EXPECT_EQ(50, nodes.back().second);
  EXPECT_EQ(50, node_to_start_time[nodes.back().first]);
}

TEST(TopologicalSortNodesWithTimePriority, WhileLoop) {
  using namespace ::tensorflow::ops;            // NOLINT(build/namespaces)
  using namespace ::tensorflow::ops::internal;  // NOLINT(build/namespaces)

  // Create placeholders.
  Scope root = Scope::NewRootScope().ExitOnError();
  std::vector<int> indexes;
  std::vector<Placeholder> placeholders_in_order;
  const int num_leaves = 20;
  for (int i = 0; i < num_leaves; ++i) {
    indexes.push_back((i + 2001) % num_leaves);
    placeholders_in_order.emplace_back(root.WithOpName(strings::StrCat("p", i)),
                                       DT_FLOAT);
    placeholders_in_order.back().node()->AddAttr("_start_time", i + 1);
  }
  std::vector<Placeholder> placeholders;
  placeholders.reserve(indexes.size());
  for (int i : indexes) {
    placeholders.push_back(placeholders_in_order[i]);
  }

  // Add a while loop above each placeholder.
  std::vector<Exit> while_exits;
  const int nodes_per_loop = 8;
  for (int i : indexes) {
    Scope scope = root.NewSubScope(strings::StrCat("while", i));
    auto dummy = Placeholder(scope, DT_FLOAT);

    Enter enter(scope, placeholders[i], strings::StrCat("frame", i));
    Merge merge(scope, std::initializer_list<Input>{enter, dummy});
    auto cv = Const(scope.WithControlDependencies({merge.output}), false);
    LoopCond loop_cond(scope, cv);
    Switch switch_node(scope, merge.output, loop_cond);
    Identity identity(scope, switch_node.output_true);
    NextIteration next_iteration(scope, identity);
    while_exits.emplace_back(scope.WithOpName("exit"),
                             switch_node.output_false);

    // Complete loop by removing dummy node and attaching NextIteration to
    // that input of the merge node.
    scope.graph()->RemoveNode(dummy.node());
    scope.graph()->AddEdge(next_iteration.node(), 0, merge.output.node(), 1);

    int base_start_time = i * 10 + 100;
    for (const auto& op : std::initializer_list<Output>{
             enter, merge.output, cv, loop_cond, switch_node.output_false,
             identity, next_iteration, while_exits.back()}) {
      op.node()->AddAttr("_start_time", base_start_time++);
    }
  }

  // Create ops that depend on the loop exits.
  std::vector<Square> squares;
  squares.reserve(indexes.size());
  for (int i : indexes) {
    squares.emplace_back(root.WithOpName(strings::StrCat("s", i)),
                         while_exits[i]);
    squares.back().node()->AddAttr("_start_time", 500 - (i + 1));
  }

  GraphDef gdef;
  TF_EXPECT_OK(root.ToGraphDef(&gdef));

  // Run the sort. The while loop nodes do not appear in the output <nodes>.
  std::vector<std::pair<const NodeDef*, int64_t>> nodes;
  std::unordered_map<const NodeDef*, int64_t> node_to_start_time;
  TF_CHECK_OK(
      TopologicalSortNodesWithTimePriority(&gdef, &nodes, &node_to_start_time));
  ASSERT_LT(while_exits.size() + squares.size() + placeholders.size(),
            nodes.size());
  int node_index = 0;
  for (int i = 0; i < placeholders.size(); ++i, ++node_index) {
    const NodeDef* node = nodes[i].first;
    EXPECT_EQ(strings::StrCat("p", i), node->name());
    EXPECT_EQ(i + 1, nodes[i].second);
    EXPECT_EQ(i + 1, node_to_start_time[node]);
  }
  for (int i = 0; i < while_exits.size(); ++i, node_index += nodes_per_loop) {
    const NodeDef* node = nodes[node_index].first;
    EXPECT_EQ(strings::StrCat("while", i, "/Enter"), node->name());
    EXPECT_EQ(100 + i * 10, nodes[node_index].second);
    EXPECT_EQ(100 + i * 10, node_to_start_time[node]);
  }
  for (int i = 0; i < squares.size(); ++i, ++node_index) {
    int square_index = num_leaves - 1 - i;
    const NodeDef* node = nodes[node_index].first;
    EXPECT_EQ(strings::StrCat("s", square_index), node->name());
    EXPECT_EQ(500 - (square_index + 1), nodes[node_index].second);
    EXPECT_EQ(500 - (square_index + 1), node_to_start_time[node]);
  }
}

}  // namespace
}  // namespace tensorflow
