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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
absl::Status AddDefaultAttributes(const GraphDef& input_graph_def,
                                  const TransformFuncContext& context,
                                  GraphDef* output_graph_def);

class AddDefaultAttributesTest : public ::testing::Test {
 protected:
  void TestAddDefaultAttributes() {
    GraphDef graph_def;

    NodeDef* lrn_node1 = graph_def.add_node();
    lrn_node1->set_name("lrn_node1");
    lrn_node1->set_op("LRN");

    NodeDef* lrn_node2 = graph_def.add_node();
    lrn_node2->set_name("lrn_node2");
    lrn_node2->set_op("LRN");
    SetNodeAttr("depth_radius", 7, lrn_node2);
    SetNodeAttr("bias", 2.0f, lrn_node2);
    SetNodeAttr("alpha", 2.0f, lrn_node2);
    SetNodeAttr("beta", 1.0f, lrn_node2);

    GraphDef result;
    TF_ASSERT_OK(AddDefaultAttributes(graph_def, {}, &result));

    std::map<string, const NodeDef*> nodes;
    MapNamesToNodes(result, &nodes);
    EXPECT_EQ(5, nodes.at("lrn_node1")->attr().at("depth_radius").i());
    EXPECT_NEAR(1.0f, nodes.at("lrn_node1")->attr().at("bias").f(), 1e-5f);
    EXPECT_NEAR(1.0f, nodes.at("lrn_node1")->attr().at("alpha").f(), 1e-5f);
    EXPECT_NEAR(0.5f, nodes.at("lrn_node1")->attr().at("beta").f(), 1e-5f);
    EXPECT_EQ(7, nodes.at("lrn_node2")->attr().at("depth_radius").i());
    EXPECT_NEAR(2.0f, nodes.at("lrn_node2")->attr().at("bias").f(), 1e-5f);
    EXPECT_NEAR(2.0f, nodes.at("lrn_node2")->attr().at("alpha").f(), 1e-5f);
    EXPECT_NEAR(1.0f, nodes.at("lrn_node2")->attr().at("beta").f(), 1e-5f);
  }
};

TEST_F(AddDefaultAttributesTest, TestAddDefaultAttributes) {
  TestAddDefaultAttributes();
}

}  // namespace graph_transforms
}  // namespace tensorflow
