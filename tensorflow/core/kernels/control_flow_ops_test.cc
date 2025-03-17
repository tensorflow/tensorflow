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

#include <string>

#include "tensorflow/core/common_runtime/type_inference.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Tests for the switch op
class SwitchOpTest : public OpsTestBase {
 protected:
  void Initialize(DataType dt) {
    TF_ASSERT_OK(NodeDefBuilder("op", "Switch")
                     .Input(FakeInput(dt))
                     .Input(FakeInput())
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SwitchOpTest, Int32Success_6_s0) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {false});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
  EXPECT_EQ(nullptr, GetOutput(1));
}

TEST_F(SwitchOpTest, Int32Success_6_s1) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {true});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(1));
  EXPECT_EQ(nullptr, GetOutput(0));
}

TEST_F(SwitchOpTest, Int32Success_2_3_s0) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {false});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
  EXPECT_EQ(nullptr, GetOutput(1));
}

TEST_F(SwitchOpTest, StringSuccess_s1) {
  Initialize(DT_STRING);
  AddInputFromArray<tstring>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  AddInputFromArray<bool>(TensorShape({}), {true});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<tstring>(&expected, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(1));
  EXPECT_EQ(nullptr, GetOutput(0));
}

class AbortOpTest : public OpsTestBase {
 protected:
};

#ifdef PLATFORM_WINDOWS
#define SIGABRT 3

class KilledBySignal {
 public:
  explicit KilledBySignal(int signum) : signum_(signum) {}
  bool operator()(int exit_status) const { return exit_status == signum_; }

 private:
  const int signum_;
};
#else
#define KilledBySignal ::testing::KilledBySignal
#endif

// Pass an error message to the op.
TEST_F(AbortOpTest, pass_error_msg) {
  TF_ASSERT_OK(NodeDefBuilder("abort_op", "Abort")
                   .Attr("error_msg", "abort_op_test")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  EXPECT_EXIT(RunOpKernel().IgnoreError(), KilledBySignal(SIGABRT),
              "Abort_op intentional failure; abort_op_test");
}

// Use the default error message.
TEST_F(AbortOpTest, default_msg) {
  TF_ASSERT_OK(NodeDefBuilder("abort_op", "Abort").Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  EXPECT_EXIT(RunOpKernel().IgnoreError(), KilledBySignal(SIGABRT),
              "Abort_op intentional failure; ");
}

// Exit normally.
TEST_F(AbortOpTest, exit_normally) {
  TF_ASSERT_OK(NodeDefBuilder("abort_op", "Abort")
                   .Attr("exit_without_error", true)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  EXPECT_EXIT(RunOpKernel().IgnoreError(), ::testing::ExitedWithCode(0), "");
}

// Adds identity notes to all outputs of this node
static void add_identity_nodes(Node* node, Graph& graph,
                               std::vector<Node*>& identity_nodes) {
  for (int i = 0; i < node->num_outputs(); i++) {
    Node* new_node;
    std::string name = absl::StrCat("Identity", i);
    TF_EXPECT_OK(NodeBuilder(name, "Identity")
                     .Attr("T", node->output_type(i))
                     .Input(node, i)
                     .Finalize(&graph, &new_node));
    identity_nodes.push_back(new_node);
  }
}

// Runs type inference pass on graph
static absl::Status type_inference(Graph& graph) {
  GraphOptimizationPassOptions opt_options;
  std::unique_ptr<Graph> graph_ptr(new Graph(OpRegistry::Global()));
  graph_ptr->Copy(graph);
  opt_options.graph = &graph_ptr;
  opt_options.flib_def = graph.mutable_flib_def();
  TypeInferencePass pass;
  return pass.Run(opt_options);
}

TEST(MergeOpTest, TypeInference) {
  GTEST_SKIP() << "TODO(b/222556864) fix \"Merge\" forward type inference "
               << "to support \"value_index\" special case";
  Graph graph(OpRegistry::Global());  // NOLINT(*-unreachable-code)
  protobuf::TextFormat::Parser parser;

  FullTypeDef input_dataset_t;
  Node* input_dataset1;
  Node* input_dataset2;
  Node* merge;
  CHECK(parser.ParseFromString(
      R"pb(type_id: TFT_PRODUCT
           args {
             type_id: TFT_DATASET
             args {
               type_id: TFT_PRODUCT
               args {
                 type_id: TFT_RAGGED
                 args { type_id: TFT_STRING }
               }
             }
           })pb",
      &input_dataset_t));
  TensorProto tensor_proto;
  TF_EXPECT_OK(NodeBuilder("input_dataset1", "Const")
                   .Attr("value", tensor_proto)
                   .Attr("dtype", DT_VARIANT)
                   .Finalize(&graph, &input_dataset1));
  (*input_dataset1->mutable_def()->mutable_experimental_type()) =
      input_dataset_t;
  TF_EXPECT_OK(NodeBuilder("input_dataset2", "Const")
                   .Attr("value", tensor_proto)
                   .Attr("dtype", DT_VARIANT)
                   .Finalize(&graph, &input_dataset2));
  (*input_dataset1->mutable_def()->mutable_experimental_type()) =
      input_dataset_t;

  TF_EXPECT_OK(NodeBuilder("Merge", "Merge")
                   .Attr("T", DT_VARIANT)
                   .Attr("N", 2)
                   .Input({input_dataset1, input_dataset2})
                   .Finalize(&graph, &merge));
  std::vector<Node*> identity_nodes;
  add_identity_nodes(merge, graph, identity_nodes);
  TF_EXPECT_OK(type_inference(graph));
  EXPECT_TRUE(full_type::IsEqual(identity_nodes[0]->def().experimental_type(),
                                 input_dataset1->def().experimental_type()))
      << "fulltype is\n"
      << identity_nodes[0]->def().experimental_type().DebugString()
      << "\nexpected\n"
      << input_dataset1->def().experimental_type().DebugString();
}

// Tests for Enter op.
class EnterOpTest : public OpsTestBase {
 protected:
  void Initialize(DataType dt) {
    TF_ASSERT_OK(NodeDefBuilder("op", "Enter")
                     .Input(FakeInput(dt))
                     .Attr("frame_name", "EnterOp")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(EnterOpTest, QUInt8_Success) {
  Initialize(DT_QUINT8);
  AddInputFromArray<quint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({2, 3}));
  test::FillValues<quint8>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
}

TEST_F(EnterOpTest, String_Success) {
  Initialize(DT_STRING);
  AddInputFromArray<tstring>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<tstring>(&expected, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

// Tests for Exit op.
class ExitOpTest : public OpsTestBase {
 protected:
  void Initialize(DataType dt) {
    TF_ASSERT_OK(
        NodeDefBuilder("op", "Exit").Input(FakeInput(dt)).Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ExitOpTest, QUInt8_Success) {
  Initialize(DT_QUINT8);
  AddInputFromArray<quint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({2, 3}));
  test::FillValues<quint8>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
}

TEST_F(ExitOpTest, String_Success) {
  Initialize(DT_STRING);
  AddInputFromArray<tstring>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<tstring>(&expected, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

}  // namespace
}  // namespace tensorflow
