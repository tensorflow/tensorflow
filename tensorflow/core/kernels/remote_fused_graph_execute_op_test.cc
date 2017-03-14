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

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/remote_fused_graph_execute_info.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_executor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_op_test_utils.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class RemoteFusedGraphExecuteTest : public OpsTestBase {};

TEST_F(RemoteFusedGraphExecuteTest, ExecuteAddGraph) {
  TF_ASSERT_OK(
      NodeDefBuilder("remote_fused_graph_execute_op", "RemoteFusedGraphExecute")
          .Input(FakeInput(2, DT_FLOAT))
          .Attr("M", 2)
          .Attr("N", 1)
          .Attr("T", DataTypeToEnum<float>::v())
          .Attr("U", DataTypeToEnum<float>::v())
          .Attr("serialized_graph_transfer_info", "")
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // TODO(satok): Add benchmark
}

////////////////////////////
// End-to-end test: Begin //
////////////////////////////
// This test does a end-to-end test for a simple usage of
// RemoteFusedGraphExecuteOp.

constexpr const char* const NAME_A = "a";
constexpr const char* const NAME_B = "b";
constexpr const char* const NAME_A_PLUS_B = "a_plus_b";
constexpr const char* const REMOTE_FUSED_EXECUTE_OP_NODE_NAME =
    "remote_fused_execute_op";
constexpr const char* const REMOTE_FUSED_EXECUTOR_NAME =
    "build_test_remote_fused_graph_executor";

constexpr float NODE_A_VAL = 2.0f;
constexpr float NODE_A_VAL2 = 10.0f;
constexpr float NODE_B_VAL = 3.0f;
constexpr float FLOAT_VALUE_TOLERANCE = 1e-8f;

// Utility functions //
static Output BuildPlaceHolderOp(const string& name, const DataType dt,
                                 const TensorShape& tensor_shape, Scope* root) {
  const Scope& scope = root->WithOpName(name);
  Node* ret;
  const string unique_name = scope.GetUniqueNameForOp("PlaceholderV2");
  NodeBuilder builder = NodeBuilder(unique_name, "PlaceholderV2")
                            .Attr("dtype", dt)
                            .Attr("shape", tensor_shape);
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  CHECK(scope.ok());
  return Output(ret, 0);
}

static Output BuildRemoteFusedGraphExecuteOp(
    const string& name, const std::vector<Output>& output_list,
    const int output_node_count,
    const RemoteFusedGraphExecuteInfo& execute_info, Scope* root) {
  const Scope& scope = root->WithOpName(name);
  Node* ret;
  CHECK(scope.ok());
  auto node_out_list = ops::AsNodeOutList(scope, InputList(output_list));
  const auto unique_name = scope.GetUniqueNameForOp("RemoteFusedGraphExecute");
  auto builder = NodeBuilder(unique_name, "RemoteFusedGraphExecute")
                     .Input(node_out_list)
                     .Attr("M", static_cast<int64>(output_list.size()))
                     .Attr("N", static_cast<int64>(output_node_count))
                     .Attr("T", DT_FLOAT)
                     .Attr("U", DT_FLOAT)
                     .Attr("serialized_graph_transfer_info",
                           StringPiece(execute_info.SerializeAsString()));
  CHECK(scope.ok());
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  CHECK(scope.ok());
  return Output(ret, 0);
}

static RemoteFusedGraphExecuteInfo BuildRemoteFusedGraphExecuteInfo(
    const GraphDef& original_graph) {
  RemoteFusedGraphExecuteInfo execute_info;
  execute_info.set_executor_name(REMOTE_FUSED_EXECUTOR_NAME);

  // In this example, simply copy all nodes. Basically, you don't need to add
  // unused node for inference.
  for (const NodeDef& node : original_graph.node()) {
    NodeDef& copied_node = *execute_info.add_node();
    copied_node = node;
    // Adding tensor shape type to the node
    // TODO(satok): Use TensorShapeMap to detime tensor shape type
    RemoteFusedGraphExecuteUtils::AddOutputTensorShapeType(
        std::vector<DataType>({DT_FLOAT}),
        std::vector<TensorShape>({TensorShape()}), &copied_node);
  }

  // Add node A as input
  execute_info.add_graph_input_node_name(NAME_A);
  RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& shape_a =
      *execute_info.add_default_graph_input_tensor_shape();
  shape_a.set_dtype(DT_FLOAT);
  // (skip setting shape to shape_a as it's shape is rank = 0.)

  // Add node A + B as output
  execute_info.add_graph_output_node_name(NAME_A_PLUS_B);
  RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& shape_a_plus_b =
      *execute_info.add_default_graph_output_tensor_shape();
  shape_a_plus_b.set_dtype(DT_FLOAT);
  // (skip setting shape to shape_a_plus_b as it's shape is rank = 0.)

  return execute_info;
}

// 1. Create TestRemoteFusedGraphExecutor to execute your fused graph
class TestRemoteFusedGraphExecutor final : public IRemoteFusedGraphExecutor {
 public:
  int GetVersion() final { return 1; }
  bool Init(const RemoteFusedGraphExecuteInfo& info) final {
    info_ = &info;
    for (const NodeDef& node_def : info.node()) {
      node_def_map_.emplace(node_def.name(), &node_def);
    }
    return true;
  }
  bool Finalize() final { return true; }
  bool SetupGraph() final { return true; }
  bool ExecuteGraph() final {
    CHECK(info_ != nullptr);
    // TODO(satok): Add utilities to implement this function more easily.
    // CAVEAT: This test only handles add op. You can implement here as you
    // like.
    CHECK_EQ(1, info_->graph_input_node_name_size());
    const string& input_node_name = info_->graph_input_node_name(0);
    const Tensor& input_tensor = input_tensor_cache_[input_node_name];
    const float input_val = *input_tensor.scalar<float>().data();
    // TODO(satok): Read NAME_B from node_a_plus_b
    const NodeDef& node_b = *node_def_map_.at(NAME_B);
    const TensorProto* proto = nullptr;
    TF_CHECK_OK(GetNodeAttr(node_b, "value", &proto));
    Tensor const_tensor;
    TF_CHECK_OK(RemoteFusedGraphExecuteUtils::MakeTensorFromProto(
        *proto, &const_tensor));
    const float b_val = *const_tensor.scalar<float>().data();
    Tensor output_a_plus_b(DT_FLOAT, {});
    output_a_plus_b.flat<float>().data()[0] = input_val + b_val;
    output_tensor_buf_.emplace(info_->graph_output_node_name(0),
                               output_a_plus_b);
    return true;
  }

  bool TeardownGraph() final { return true; }

  bool FillInputNode(const string& node_name, const Tensor& tensor) final {
    input_tensor_cache_[node_name] = tensor;
    return true;
  }

  bool ReadOutputNode(const string& node_name,
                      TensorAllocatorFunc tensor_allocator) final {
    // TODO(satok): Specify tensor shape by using default_graph_tensor_shape.
    const Tensor& buffered_output_tensor = output_tensor_buf_.at(node_name);
    const TensorShape& output_shape = buffered_output_tensor.shape();
    Tensor* output_tensor = tensor_allocator(output_shape);
    CHECK_EQ(buffered_output_tensor.dtype(), output_tensor->dtype());
    CHECK(output_tensor->CopyFrom(buffered_output_tensor, output_shape));
    return true;
  }

 private:
  const RemoteFusedGraphExecuteInfo* info_;
  std::unordered_map<string, Tensor> input_tensor_cache_;
  std::unordered_map<string, const NodeDef*> node_def_map_;
  std::unordered_map<string, Tensor> output_tensor_buf_;
};

// 2. Register a builder of your custom executor
namespace remote_fused_graph_execute_op {
Status BuildRemoteFusedGraphExecutor(
    std::unique_ptr<IRemoteFusedGraphExecutor>* executor) {
  executor->reset(new TestRemoteFusedGraphExecutor());
  return Status::OK();
}

// This class instantiation registers executor to the
// RemoteFusedGraphExecuteOp. This architecture makes executors to be
// pluggable in order not to link unnecessary libraries.
static RemoteFusedGraphExecuteUtils::ExecutorBuildRegistrar
    k_test_remote_fused_graph_executor_build(REMOTE_FUSED_EXECUTOR_NAME,
                                             BuildRemoteFusedGraphExecutor);
}  // namespace remote_fused_graph_execute_op

// 3. Create Graph transform function to fuse your graph
static Status RewriteGraphToFusedGraph(const GraphDef& original_graph,
                                       GraphDef* fused_graph) {
  Scope root = Scope::NewRootScope();
  std::vector<Output> output_list;
  const Output op_a = BuildPlaceHolderOp(NAME_A, DT_FLOAT, {}, &root);
  output_list.emplace_back(op_a);
  const RemoteFusedGraphExecuteInfo execute_info =
      BuildRemoteFusedGraphExecuteInfo(original_graph);
  BuildRemoteFusedGraphExecuteOp(REMOTE_FUSED_EXECUTE_OP_NODE_NAME, output_list,
                                 1, execute_info, &root);
  GraphDef fused_graph_def;
  TF_CHECK_OK(root.ToGraphDef(&fused_graph_def));
  *fused_graph = fused_graph_def;
  return Status::OK();
}

// 4. Register transform function
// You can register transform function by REGISTER_GRAPH_TRANSFORM.
// In this test, we don't use graph transform tool to avoid linking to
// the graph transform library.
// To register transform function, you need to change the interface of
// BuildFusedGraphDefOfAddGraph to
// Status BuildFusedGraphDefOfAddGraph(
// const GraphDef& original_graph, const TransformFuncContext& context,
// GraphDef* output_graph_def);
// Then register the function like:
// REGISTER_GRAPH_TRANSFORM("rewrite_graph", RewriteGraph);

// 5. Fuse the original graph and run the inference the new fused graph
TEST(RemoteFusedExecuteGraphOp, EndToEndTest) {
  // 5.1 Load original graph
  const GraphDef original_graph =
      RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
          NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B);

  // 5.2 Fuse graph
  GraphDef fused_graph;
  TF_CHECK_OK(RewriteGraphToFusedGraph(original_graph, &fused_graph));

  // 5.3 Setup session
  std::vector<Tensor> output_tensors;
  SessionOptions session_options;
  session_options.env = Env::Default();
  std::unique_ptr<Session> session =
      std::unique_ptr<Session>(NewSession(session_options));
  Status status = session->Create(fused_graph);
  ASSERT_TRUE(status.ok());
  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;

  // 5.4 Setup input
  Tensor input_a(DT_FLOAT, {});
  input_a.flat<float>().data()[0] = NODE_A_VAL2;
  std::vector<std::pair<string, Tensor>> inputs;
  inputs.emplace_back(NAME_A, input_a);

  // 5.5 Setup output
  const std::vector<string> outputs{REMOTE_FUSED_EXECUTE_OP_NODE_NAME};

  // 5.6 Run inference with all node as output
  status = session->Run(run_options, inputs, outputs, {}, &output_tensors,
                        &run_metadata);
  ASSERT_TRUE(status.ok());

  // 5.7 Check output tensor value
  ASSERT_EQ(1, output_tensors.size());
  EXPECT_NEAR(NODE_A_VAL2 + NODE_B_VAL,
              output_tensors.at(0).flat<float>().data()[0],
              FLOAT_VALUE_TOLERANCE);
}

////////////////////////////
// End-to-end test: End   //
////////////////////////////

}  // namespace tensorflow
