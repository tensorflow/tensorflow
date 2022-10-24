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

#include "tensorflow/core/grappler/optimizers/auto_parallel.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class AutoParallelTest : public ::testing::Test {};

TEST_F(AutoParallelTest, SimpleParallel) {
  tensorflow::Scope s = tensorflow::Scope::DisabledShapeInferenceScope();
  Output constant_a = ops::Const(s.WithOpName("constant_a"), 1.0f, {1});
  Output constant_b = ops::Const(s.WithOpName("constant_b"), 1, {1});
  Output var = ops::Variable(s.WithOpName("var"), {1}, DT_FLOAT);
  Output assign = ops::Assign(s.WithOpName("assign"), {var}, {constant_a});
  Output identity = ops::Identity(s.WithOpName("identity"), {var});
  Output fifo_queue = ops::FIFOQueue(s.WithOpName("fifo_queue"), {DT_FLOAT});
  auto dequeue = ops::QueueDequeueMany(s.WithOpName("dequeue"), {fifo_queue},
                                       {constant_b}, {DT_FLOAT});
  Output add = ops::AddN(s.WithOpName("add"), {constant_a, dequeue[0]});
  Output learning_rate = ops::Const(s.WithOpName("learning_rate"), 0.01f, {1});
  Output apply_gradient = ops::ApplyGradientDescent(
      s.WithOpName("apply_gradient"), {var}, {learning_rate}, {add});

  GrapplerItem item;
  item.init_ops.push_back("assign");
  item.fetch.push_back("apply_gradient");
  item.init_ops.push_back("assign");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  AutoParallel parallel(2);
  GraphDef output;
  Status status = parallel.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_EQ(21, output.node_size());

  const NodeDef& node_assign = output.node(0);
  EXPECT_EQ("assign", node_assign.name());
  EXPECT_EQ("AutoParallel-Replica-0/constant_a", node_assign.input(1));

  const NodeDef& node_constant_b = output.node(1);
  EXPECT_EQ("constant_b", node_constant_b.name());

  const NodeDef& node_fifo_queue = output.node(2);
  EXPECT_EQ("fifo_queue", node_fifo_queue.name());

  const NodeDef& node_identity = output.node(3);
  EXPECT_EQ("identity", node_identity.name());
  EXPECT_EQ("var", node_identity.input(0));

  const NodeDef& node_var = output.node(4);
  EXPECT_EQ("var", node_var.name());

  const NodeDef& node_div_const0 = output.node(5);
  EXPECT_EQ("AutoParallel-Replica-0/AutoParallel-Div-Const",
            node_div_const0.name());

  const NodeDef& node_div0 = output.node(6);
  EXPECT_EQ("AutoParallel-Replica-0/AutoParallel-Div-apply_gradient",
            node_div0.name());
  const NodeDef& node_add0 = output.node(7);
  EXPECT_EQ("AutoParallel-Replica-0/add", node_add0.name());

  const NodeDef& node_gradient0 = output.node(8);
  EXPECT_EQ("AutoParallel-Replica-0/apply_gradient", node_gradient0.name());

  const NodeDef& node_constant_a0 = output.node(9);
  EXPECT_EQ("AutoParallel-Replica-0/constant_a", node_constant_a0.name());

  const NodeDef& node_dequeue0 = output.node(10);
  EXPECT_EQ("AutoParallel-Replica-0/dequeue", node_dequeue0.name());

  const NodeDef& node_learning_rate0 = output.node(11);
  EXPECT_EQ("AutoParallel-Replica-0/learning_rate", node_learning_rate0.name());

  const NodeDef& node_div_const1 = output.node(12);
  EXPECT_EQ("AutoParallel-Replica-1/AutoParallel-Div-Const",
            node_div_const1.name());

  const NodeDef& node_div1 = output.node(13);
  EXPECT_EQ("AutoParallel-Replica-1/AutoParallel-Div-apply_gradient",
            node_div1.name());

  const NodeDef& node_add1 = output.node(14);
  EXPECT_EQ("AutoParallel-Replica-1/add", node_add1.name());

  const NodeDef& node_gradient1 = output.node(15);
  EXPECT_EQ("AutoParallel-Replica-1/apply_gradient", node_gradient1.name());

  const NodeDef& node_constant_a1 = output.node(16);
  EXPECT_EQ("AutoParallel-Replica-1/constant_a", node_constant_a1.name());

  const NodeDef& node_dequeue1 = output.node(17);
  EXPECT_EQ("AutoParallel-Replica-1/dequeue", node_dequeue1.name());

  const NodeDef& node_learning_rate1 = output.node(18);
  EXPECT_EQ("AutoParallel-Replica-1/learning_rate", node_learning_rate1.name());

  const NodeDef& node_fetch = output.node(19);
  EXPECT_EQ("AutoParallel-Control-Fetch", node_fetch.name());
  EXPECT_EQ("^AutoParallel-Replica-0/apply_gradient", node_fetch.input(0));
  EXPECT_EQ("^AutoParallel-Replica-1/apply_gradient", node_fetch.input(1));

  const NodeDef& node_gradient = output.node(20);
  EXPECT_EQ("apply_gradient", node_gradient.name());
  EXPECT_EQ("^AutoParallel-Control-Fetch", node_gradient.input(0));
}

TEST_F(AutoParallelTest, SimpleParallelNoDequeue) {
  tensorflow::Scope s = tensorflow::Scope::DisabledShapeInferenceScope();
  Output constant_a = ops::Const(s.WithOpName("constant_a"), 1.0f, {1});
  Output constant_c = ops::Const(s.WithOpName("constant_c"), 1.0f, {1});
  Output constant_b = ops::Const(s.WithOpName("constant_b"), 1, {1});
  Output var = ops::Variable(s.WithOpName("var"), {1}, DT_FLOAT);
  Output assign = ops::Assign(s.WithOpName("assign"), {var}, {constant_a});
  Output add = ops::AddN(s.WithOpName("add"), {constant_a, constant_c});
  Output learning_rate = ops::Const(s.WithOpName("learning_rate"), 0.01f, {1});
  Output apply_gradient = ops::ApplyGradientDescent(
      s.WithOpName("apply_gradient"), {var}, {learning_rate}, {add});

  GrapplerItem item;
  item.init_ops.push_back("assign");
  item.fetch.push_back("apply_gradient");
  item.init_ops.push_back("assign");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  AutoParallel parallel(2);
  GraphDef output;
  Status status = parallel.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
