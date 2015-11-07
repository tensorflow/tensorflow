#include "tensorflow/core/graph/graph_partition.h"

#include <unordered_map>

#include <gtest/gtest.h>
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
  popts.control_flow_added = false;
  Status s = Partition(popts, &g, partitions);
  CHECK(s.ok()) << s;
}

void CheckLoopConstruction(const GraphDef& graph_def) {
  std::unordered_map<string, GraphDef> partitions;
  Partition(graph_def, &partitions);
  GraphConstructorOptions opts;
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

REGISTER_OP("Input").Output("o: float");
REGISTER_OP("BoolInput").Output("o: bool");
REGISTER_OP("Cross").Input("a: float").Input("b: float").Output("o: float");

Node* Input(const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("Input", opts);
}

Node* BoolInput(const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("BoolInput", opts);
}

Node* Cross(ops::NodeOut a, ops::NodeOut b,
            const GraphDefBuilder::Options& opts) {
  return ops::BinaryOp("Cross", a, b, opts);
}

class GraphPartitionTest : public ::testing::Test {
 protected:
  GraphPartitionTest()
      : in_(GraphDefBuilder::kFailImmediately),
        builder_a_(GraphDefBuilder::kFailImmediately),
        builder_b_(GraphDefBuilder::kFailImmediately),
        a_opts_(builder_a_.opts().WithDevice("/job:a/replica:0/task:0/cpu:0")),
        b_opts_(builder_b_.opts().WithDevice("/job:a/replica:0/task:0/cpu:1")) {
    RequireDefaultOps();
  }

  const GraphDef& ToGraphDef() {
    in_.ToGraphDef(&in_graph_def_);
    return in_graph_def_;
  }

  void ExpectMatchA() {
    GraphDef graph_def;
    builder_a_.ToGraphDef(&graph_def);
    string a = "/job:a/replica:0/task:0/cpu:0";
    TF_EXPECT_GRAPH_EQ(graph_def, partitions_[a]);
  }

  void ExpectMatchB() {
    GraphDef graph_def;
    builder_b_.ToGraphDef(&graph_def);
    string b = "/job:a/replica:0/task:0/cpu:1";
    TF_EXPECT_GRAPH_EQ(graph_def, partitions_[b]);
  }

  GraphDefBuilder in_;
  GraphDef in_graph_def_;
  GraphDefBuilder builder_a_;
  GraphDefBuilder builder_b_;
  GraphDefBuilder::Options a_opts_;
  GraphDefBuilder::Options b_opts_;
  std::unordered_map<string, GraphDef> partitions_;
};

TEST_F(GraphPartitionTest, SingleDevice) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  Node* a1 = Input(in_.opts().WithName("A1"));
  Cross(a1, a1, in_.opts().WithName("A2"));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(1, partitions_.size());

  a1 = Input(a_opts_.WithName("A1"));
  Cross(a1, a1, a_opts_.WithName("A2"));
  ExpectMatchA();
}

TEST_F(GraphPartitionTest, CrossDeviceData) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  Node* a1 = Input(in_.opts().WithName("A1"));
  Node* b1 = Input(in_.opts().WithName("B1"));
  Cross(a1, b1, in_.opts().WithName("B2"));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = Input(a_opts_.WithName("A1"));
  _Send(a1, "edge_1_A1", a, 82, b, a_opts_.WithName("A1/_0"));
  ExpectMatchA();

  b1 = Input(b_opts_.WithName("B1"));
  Node* recv =
      _Recv(DT_FLOAT, "edge_1_A1", a, 82, b, b_opts_.WithName("A1/_1"));
  Cross(recv, b1, b_opts_.WithName("B2"));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceControl) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  Node* a1 = Input(in_.opts().WithName("A1"));
  Node* b1 = Input(in_.opts().WithName("B1"));
  Cross(b1, b1, in_.opts().WithName("B2").WithControlInput(a1));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = Input(a_opts_.WithName("A1"));
  Node* c = EmptyConst<float>(a_opts_.WithName("A1/_0").WithControlInput(a1));
  _Send(c, "edge_3_A1", a, 82, b, a_opts_.WithName("A1/_1"));
  ExpectMatchA();

  Node* recv =
      _Recv(DT_FLOAT, "edge_3_A1", a, 82, b, b_opts_.WithName("A1/_2"));
  Node* id = Identity(recv, b_opts_.WithName("A1/_3"));
  b1 = Input(b_opts_.WithName("B1"));
  Cross(b1, b1, b_opts_.WithName("B2").WithControlInput(id));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceData_MultiUse) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  Node* a1 = Input(in_.opts().WithName("A1"));
  Node* b1 = Input(in_.opts().WithName("B1"));
  Cross(a1, b1, in_.opts().WithName("B2"));
  Cross(a1, a1, in_.opts().WithName("B3"));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = Input(a_opts_.WithName("A1"));
  _Send(a1, "edge_1_A1", a, 82, b, a_opts_.WithName("A1/_0"));
  ExpectMatchA();

  Node* recv =
      _Recv(DT_FLOAT, "edge_1_A1", a, 82, b, b_opts_.WithName("A1/_1"));
  b1 = Input(b_opts_.WithName("B1"));
  Cross(recv, b1, b_opts_.WithName("B2"));
  Cross(recv, recv, b_opts_.WithName("B3"));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceControl_MultiUse) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  Node* a1 = Input(in_.opts().WithName("A1"));
  Node* b1 = Input(in_.opts().WithName("B1"));
  Cross(b1, b1, in_.opts().WithName("B2").WithControlInput(a1));
  Input(in_.opts().WithName("B3").WithControlInput(a1));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = Input(a_opts_.WithName("A1"));
  Node* c = EmptyConst<float>(a_opts_.WithName("A1/_0").WithControlInput(a1));
  _Send(c, "edge_1_A1", a, 82, b, a_opts_.WithName("A1/_1"));
  ExpectMatchA();

  Node* recv =
      _Recv(DT_FLOAT, "edge_1_A1", a, 82, b, b_opts_.WithName("A1/_2"));
  Node* id = Identity(recv, b_opts_.WithName("A1/_3"));
  b1 = Input(b_opts_.WithName("B1"));
  Cross(b1, b1, b_opts_.WithName("B2").WithControlInput(id));
  Input(b_opts_.WithName("B3").WithControlInput(id));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDevice_DataControl) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  Node* a1 = Input(in_.opts().WithName("A1"));
  Node* b1 = Input(in_.opts().WithName("B1"));
  Cross(a1, b1, in_.opts().WithName("B2"));
  Input(in_.opts().WithName("B3").WithControlInput(a1));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = Input(a_opts_.WithName("A1"));
  Node* c = EmptyConst<float>(a_opts_.WithName("A1/_0").WithControlInput(a1));
  // NOTE: Send 0 A1/_1 -> A1/_2 is not necessarily needed. We could
  // use A1/_0 -> A1/_4 as the control as a minor optimization.
  _Send(c, "edge_1_A1", a, 82, b, a_opts_.WithName("A1/_1"));
  _Send(a1, "edge_2_A1", a, 82, b, a_opts_.WithName("A1/_4"));
  ExpectMatchA();

  Node* recv1 =
      _Recv(DT_FLOAT, "edge_1_A1", a, 82, b, b_opts_.WithName("A1/_2"));
  Node* id1 = Identity(recv1, b_opts_.WithName("A1/_3"));
  Node* recv2 =
      _Recv(DT_FLOAT, "edge_2_A1", a, 82, b, b_opts_.WithName("A1/_5"));
  b1 = Input(b_opts_.WithName("B1"));
  Cross(recv2, b1, b_opts_.WithName("B2"));
  Input(b_opts_.WithName("B3").WithControlInput(id1));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceLoop) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  Node* a1 = BoolInput(in_.opts().WithName("A1"));
  Node* a2 = Enter(a1, "foo", in_.opts().WithName("A2"));
  Node* a3 = Merge({a2, {"A5", 0, DT_BOOL}}, in_.opts().WithName("A3"));
  LoopCond(a3, in_.opts().WithName("A4"));
  Node* b1 = Identity(a3, in_.opts().WithName("B1"));
  NextIteration(b1, in_.opts().WithName("A5"));

  CheckLoopConstruction(ToGraphDef());
}

}  // namespace
}  // namespace tensorflow
