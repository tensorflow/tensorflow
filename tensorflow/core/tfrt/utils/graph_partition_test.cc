/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/utils/graph_partition.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::testing::IsEmpty;
using ::testing::NotNull;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class GraphPartitionTest : public grappler::GrapplerTest {
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
    device_mgr_ = std::make_unique<DynamicDeviceMgr>(std::move(devices));

    for (auto d : device_mgr_->ListDevices()) {
      device_set_.AddDevice(d);
    }
  }

  std::unique_ptr<DeviceMgr> device_mgr_;
  Device* device0_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  Device* device1_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  DeviceSet device_set_;
};

TEST_F(GraphPartitionTest, InsertTransferOpsWithOneDevice) {
  // A graph with three nodes that are on the same device.
  // input(Placeholder, device0) -> id_x(Identity, device0) -> output(Identity,
  // device0)
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Scope scope = Scope::NewRootScope().WithDevice(device0_->name());

  auto input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);
  auto id_x = ops::Identity(scope.WithOpName("identity"), input);
  auto output = ops::Identity(scope.WithOpName("output"), id_x);
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  GraphDef original_graphdef;
  TF_ASSERT_OK(scope.ToGraphDef(&original_graphdef));

  FunctionLibraryDefinition flib_def(OpRegistry::Global());
  Placer placer(graph.get(), "", &flib_def, &device_set_, device0_);
  TF_ASSERT_OK(placer.Run());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Graph> new_graph,
      InsertTransferOps(/*graph_func_name=*/"test_graph", device_set_, device0_,
                        /*inputs=*/{"input"},
                        /*outputs=*/{"output"}, /*control_outputs=*/{},
                        std::move(graph)));

  GraphDef new_graphdef;
  new_graph->ToGraphDef(&new_graphdef);

  // The graph remains the same.
  CompareGraphs(original_graphdef, new_graphdef);
}

TEST_F(GraphPartitionTest, InsertTransferOpsWithTwoDevice) {
  // A graph with four nodes that are on different devices.
  //
  //            -----> id_x(device0) ------
  //           /                           \
  // input(device0)                     output(device0)
  //           \                           /
  //            -----> id_y(device1) ------
  //
  // After inserting transfer ops, we should get the following graph, where id_x
  // is wrapped in the function invoked by PartitionedCallOp_0, and id_y is
  // wrapped in the function invoked by PartitionedCallOp_1. Both of them have
  // a data dependency with the StatefulPartitionedCallOp.
  //
  // input ---> PartitionedCallOp_0 ----
  //                                    \
  //                         StatefulPartitionedCallOp ---> output
  //                                    /
  //            PartitionedCallOp_1 ----
  //
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Scope scope = Scope::NewRootScope();
  Scope scope0 = scope.WithDevice(device0_->name());
  Scope scope1 = scope.WithDevice(device1_->name());

  auto input = ops::Placeholder(scope0.WithOpName("input"), DT_FLOAT);
  Output id_x = ops::Identity(scope0.WithOpName("id_x"), input);
  Output id_y = ops::Identity(scope1.WithOpName("id_y"), input);
  auto output = ops::IdentityN(scope0.WithOpName("output"), {id_x, id_y});
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  FunctionLibraryDefinition flib_def(OpRegistry::Global());
  Placer placer(graph.get(), "", &flib_def, &device_set_, device0_);
  TF_ASSERT_OK(placer.Run());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Graph> new_graph,
      InsertTransferOps(/*graph_func_name=*/"test_graph", device_set_, device0_,
                        /*inputs=*/{"input"},
                        /*outputs=*/{"output"}, /*control_outputs=*/{},
                        std::move(graph)));

  GraphDef new_graphdef;
  new_graph->ToGraphDef(&new_graphdef);

  NodeDef* input_node = nullptr;
  NodeDef* output_node = nullptr;
  NodeDef* stateful_partitioned_call_node = nullptr;
  std::vector<NodeDef*> partitioned_call_nodes;

  for (NodeDef& node : *new_graphdef.mutable_node()) {
    if (node.op() == "PartitionedCall") {
      partitioned_call_nodes.push_back(&node);
    } else if (node.op() == "StatefulPartitionedCall") {
      stateful_partitioned_call_node = &node;
    } else if (node.name() == "input") {
      input_node = &node;
    } else if (node.name() == "output") {
      output_node = &node;
    }
  }
  ASSERT_THAT(input_node, NotNull());
  ASSERT_THAT(output_node, NotNull());
  ASSERT_THAT(partitioned_call_nodes, SizeIs(2));
  ASSERT_THAT(stateful_partitioned_call_node, NotNull());

  // Verify the inputs of StatefulPartitionedCallOp, which should be the two
  // PartitionedCallOps.
  EXPECT_THAT(stateful_partitioned_call_node->input(),
              UnorderedElementsAre(partitioned_call_nodes[0]->name(),
                                   partitioned_call_nodes[1]->name()));

  absl::flat_hash_map<std::string, FunctionDef> func_name_to_func;

  EXPECT_THAT(new_graphdef.library().function(), SizeIs(3));
  for (const FunctionDef& fdef : new_graphdef.library().function()) {
    // The functions are set with the attribute that indicates they
    // should not be inlined.
    ASSERT_TRUE(fdef.attr().contains(tensorflow::kNoInlineAttr));
    EXPECT_TRUE(fdef.attr().at(tensorflow::kNoInlineAttr).b());

    func_name_to_func[fdef.signature().name()] = fdef;
  }

  for (NodeDef* node : partitioned_call_nodes) {
    ASSERT_TRUE(node->attr().contains("f"));
    ASSERT_TRUE(func_name_to_func.contains(node->attr().at("f").func().name()));
    const FunctionDef& fdef =
        func_name_to_func.at(node->attr().at("f").func().name());

    // Verify the inputs of PartitionedCallOp.
    ASSERT_TRUE(fdef.attr().contains("device"));
    if (fdef.attr().at("device").s() == device0_->name()) {
      EXPECT_THAT(node->input(), UnorderedElementsAre(input_node->name()));
    } else if (fdef.attr().at("device").s() == device1_->name()) {
      EXPECT_THAT(node->input(), IsEmpty());
    }

    ASSERT_TRUE(node->attr().contains(tensorflow::kNoInlineAttr));
    EXPECT_TRUE(node->attr().at(tensorflow::kNoInlineAttr).b());

    // Each partition contains a _Send op and a _Recv op.
    int send_count = 0, recv_count = 0;
    for (const NodeDef& node : fdef.node_def()) {
      if (node.op() == "_Send")
        ++send_count;
      else if (node.op() == "_Recv")
        ++recv_count;
    }
    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);
  }
}

}  // anonymous namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
