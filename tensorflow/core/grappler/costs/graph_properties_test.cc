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

#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/inputs/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

const char kTestDataPath[] = "core/grappler/costs/graph_properties_testdata";

class GraphPropertiesTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Provision a single machine with 3 cpu cores
    cluster_.reset(new SingleMachine(5 * 60, 3, 0));
    TF_CHECK_OK(cluster_->Provision());
  }

  void TearDown() override {
    TF_CHECK_OK(cluster_->Shutdown());
    cluster_.reset();
  }

 protected:
  // Returns a string form of <p>, suitable for comparing type and shape.
  // Example output for 4-d float tensor: "float: [10,2,30,4]"
  string PropToString(const OpInfo::TensorProperties& p) {
    string s = strings::StrCat(DataTypeString(p.dtype()), ": ");
    if (p.shape().unknown_rank()) {
      strings::StrAppend(&s, "?");
    } else {
      strings::StrAppend(&s, "[");
      for (int i = 0; i < p.shape().dim_size(); ++i) {
        strings::StrAppend(&s, i == 0 ? "" : ",",
                           std::max<int64>(p.shape().dim(i).size(), -1));
      }
      strings::StrAppend(&s, "]");
    }
    return s;
  }

  std::unique_ptr<SingleMachine> cluster_;
};

TEST_F(GraphPropertiesTest, StaticProperties) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphProperties properties(item);
  Status s = properties.InferStatically(true);
  TF_CHECK_OK(s);

  for (const auto& node : item.graph.node()) {
    if (node.op() == "RandomStandardNormal") {
      // The node has one input (the shape of the tensor to generate).
      EXPECT_EQ(1, properties.GetInputProperties(node.name()).size());
      // The const node has one output.
      const auto props = properties.GetOutputProperties(node.name());
      EXPECT_EQ(1, props.size());
      const OpInfo::TensorProperties& prop = props[0];
      EXPECT_EQ(DT_FLOAT, prop.dtype());
      EXPECT_FALSE(prop.shape().unknown_rank());
      EXPECT_EQ(2, prop.shape().dim_size());
      EXPECT_EQ(10, prop.shape().dim(0).size());
      EXPECT_EQ(1, prop.shape().dim(1).size());
    } else if (node.op() == "AddN") {
      const auto in_props = properties.GetInputProperties(node.name());
      EXPECT_EQ(1, in_props.size());
      const OpInfo::TensorProperties& in_prop = in_props[0];
      EXPECT_EQ(DT_FLOAT, in_prop.dtype());
      EXPECT_FALSE(in_prop.shape().unknown_rank());
      EXPECT_EQ(2, in_prop.shape().dim_size());
      EXPECT_EQ(10, in_prop.shape().dim(0).size());
      EXPECT_EQ(1, in_prop.shape().dim(1).size());
      const auto out_props = properties.GetOutputProperties(node.name());
      EXPECT_EQ(1, out_props.size());
      string in_prop_str;
      ::tensorflow::protobuf::TextFormat::PrintToString(in_prop, &in_prop_str);
      string out_prop_str;
      ::tensorflow::protobuf::TextFormat::PrintToString(out_props[0],
                                                        &out_prop_str);
      EXPECT_EQ(in_prop_str, out_prop_str);
    }
  }
}

TEST_F(GraphPropertiesTest, ClearProperties) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphProperties properties(item);
  Status s = properties.InferStatically(true);
  TF_CHECK_OK(s);

  for (const auto& node : item.graph.node()) {
    if (node.op() == "RandomStandardNormal") {
      EXPECT_EQ(1, properties.GetInputProperties(node.name()).size());
      const auto props = properties.GetOutputProperties(node.name());
      properties.ClearOutputProperties(node.name());
      const auto cleared_props = properties.GetOutputProperties(node.name());
      EXPECT_TRUE(cleared_props.empty());
    } else if (node.op() == "AddN") {
      const auto in_props = properties.GetInputProperties(node.name());
      EXPECT_EQ(1, in_props.size());
      properties.ClearInputProperties(node.name());
      const auto cleared_props = properties.GetInputProperties(node.name());
      EXPECT_TRUE(cleared_props.empty());
    }
  }
}

TEST_F(GraphPropertiesTest, DynamicProperties) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphProperties properties(item);
  TF_CHECK_OK(cluster_->Initialize(item));
  Status s = properties.InferDynamically(cluster_.get());
  TF_CHECK_OK(s);

  for (const auto& node : item.graph.node()) {
    if (node.op() == "RandomStandardNormal") {
      // The random node is missing from the cost graph (why ?)
      EXPECT_EQ(0, properties.GetInputProperties(node.name()).size());
    } else if (node.op() == "AddN") {
      // Since the random node is missing, we can't infer the input properties
      // of the first AddN node. The other AddN nodes have the expected
      // properties.
      if (node.name() == "AddN") {
        const auto props = properties.GetInputProperties(node.name());
        EXPECT_EQ(1, props.size());
        const OpInfo::TensorProperties& prop = props[0];
        EXPECT_EQ(DT_INVALID, prop.dtype());
        EXPECT_TRUE(prop.shape().unknown_rank());
      } else {
        const auto props = properties.GetInputProperties(node.name());
        EXPECT_EQ(1, props.size());
        const OpInfo::TensorProperties& prop = props[0];
        EXPECT_EQ(DT_FLOAT, prop.dtype());
        EXPECT_FALSE(prop.shape().unknown_rank());
        EXPECT_EQ(2, prop.shape().dim_size());
        EXPECT_EQ(10, prop.shape().dim(0).size());
        EXPECT_EQ(1, prop.shape().dim(1).size());
        const auto out_props = properties.GetOutputProperties(node.name());
        EXPECT_EQ(1, out_props.size());
        string prop_str;
        ::tensorflow::protobuf::TextFormat::PrintToString(prop, &prop_str);
        string out_prop_str;
        ::tensorflow::protobuf::TextFormat::PrintToString(out_props[0],
                                                          &out_prop_str);
        EXPECT_EQ(prop_str, out_prop_str);
      }
    }
  }
}

TEST_F(GraphPropertiesTest, Variables) {
  GrapplerItem item;
  TF_CHECK_OK(NodeDefBuilder("Var", "Variable")
                  .Attr("dtype", DT_FLOAT)
                  .Attr("shape", TensorShape({3, 7}))
                  .Finalize(item.graph.add_node()));
  item.fetch.push_back("Var");

  Tensor initial_val(DT_FLOAT, TensorShape({3, 7}));
  test::FillIota<float>(&initial_val, 0);
  TF_CHECK_OK(NodeDefBuilder("InitialVal", "Const")
                  .Attr("dtype", DT_FLOAT)
                  .Attr("value", initial_val)
                  .Finalize(item.graph.add_node()));
  TF_CHECK_OK(NodeDefBuilder("InitVar", "Assign")
                  .Input("Var", 0, DT_FLOAT_REF)
                  .Input("InitialVal", 0, DT_FLOAT)
                  .Finalize(item.graph.add_node()));
  item.init_ops.push_back("InitVar");

  {
    GraphProperties static_properties(item);
    TF_CHECK_OK(static_properties.InferStatically(false));

    const auto props = static_properties.GetOutputProperties("Var");
    EXPECT_EQ(1, props.size());
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT_REF, prop.dtype());
    EXPECT_FALSE(prop.shape().unknown_rank());
    EXPECT_EQ(2, prop.shape().dim_size());
    EXPECT_EQ(3, prop.shape().dim(0).size());
    EXPECT_EQ(7, prop.shape().dim(1).size());
  }
  {
    TF_CHECK_OK(cluster_->Initialize(item));
    GraphProperties dynamic_properties(item);
    TF_CHECK_OK(dynamic_properties.InferDynamically(cluster_.get()));

    const auto props = dynamic_properties.GetOutputProperties("Var");
    EXPECT_EQ(1, props.size());
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT_REF, prop.dtype());
    EXPECT_FALSE(prop.shape().unknown_rank());
    EXPECT_EQ(2, prop.shape().dim_size());
    EXPECT_EQ(3, prop.shape().dim(0).size());
    EXPECT_EQ(7, prop.shape().dim(1).size());
  }
}

TEST_F(GraphPropertiesTest, VarHandles) {
  GrapplerItem item;
  TF_CHECK_OK(NodeDefBuilder("Var", "VarHandleOp")
                  .Attr("dtype", DT_FLOAT)
                  .Attr("shape", TensorShape({3, 7}))
                  .Finalize(item.graph.add_node()));

  TF_CHECK_OK(NodeDefBuilder("VarRead", "ReadVariableOp")
                  .Attr("dtype", DT_FLOAT)
                  .Input("Var", 0, DT_RESOURCE)
                  .Finalize(item.graph.add_node()));

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));

  const auto props = properties.GetOutputProperties("VarRead");
  EXPECT_EQ(1, props.size());
  const OpInfo::TensorProperties& prop = props[0];
  EXPECT_EQ(DT_FLOAT, prop.dtype());
  EXPECT_FALSE(prop.shape().unknown_rank());
  EXPECT_EQ(2, prop.shape().dim_size());
  EXPECT_EQ(3, prop.shape().dim(0).size());
  EXPECT_EQ(7, prop.shape().dim(1).size());
}

TEST_F(GraphPropertiesTest, Queues) {
  // Create a graph with known input shapes, and propagate the shapes through a
  // couple of queues.
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  auto q1 = ops::FIFOQueue(root.WithOpName("Queue1"), {DataType::DT_FLOAT});
  Output rnd =
      ops::RandomNormal(root.WithOpName("rnd"), {3, 7}, DataType::DT_FLOAT);
  Output square1 = ops::Square(root.WithOpName("Square1"), rnd);
  auto enqueue1 = ops::QueueEnqueue(root.WithOpName("Enqueue1"), q1, {square1});
  auto dequeue1 =
      ops::QueueDequeue(root.WithOpName("Dequeue1"), q1, {DataType::DT_FLOAT});

  auto q2 =
      ops::RandomShuffleQueue(root.WithOpName("Queue2"), {DataType::DT_FLOAT});
  Output square2 = ops::Square(root.WithOpName("Square2"), dequeue1[0]);
  auto enqueue2 = ops::QueueEnqueue(root.WithOpName("Enqueue2"), q2, {square2});
  auto dequeue2 =
      ops::QueueDequeue(root.WithOpName("Dequeue2"), q2, {DataType::DT_FLOAT});

  // Create a queue that feeds itself.
  auto q3 =
      ops::RandomShuffleQueue(root.WithOpName("Queue3"), {DataType::DT_FLOAT});
  auto dequeue3 =
      ops::QueueDequeue(root.WithOpName("Dequeue3"), q3, {DataType::DT_FLOAT});
  auto merge3 = ops::Merge(root.WithOpName("Merge3"), {dequeue3[0], square2});
  auto enqueue3 =
      ops::QueueEnqueue(root.WithOpName("Enqueue3"), q3, {merge3.output});

  auto q4 =
      ops::RandomShuffleQueue(root.WithOpName("Queue4"), {DataType::DT_FLOAT});
  auto enqueue4 = ops::QueueEnqueue(root.WithOpName("Enqueue4"), q4, {square2});
  auto enqueue4_2 =
      ops::QueueEnqueue(root.WithOpName("Enqueue4_2"), q4, {dequeue3[0]});
  auto dequeue4 =
      ops::QueueDequeue(root.WithOpName("Dequeue4"), q4, {DataType::DT_FLOAT});

  // Create a queue that takes in three tensors.
  auto q5 = ops::RandomShuffleQueue(
      root.WithOpName("Queue5"),
      {DataType::DT_FLOAT, DataType::DT_DOUBLE, DataType::DT_FLOAT});
  Output rnd2 =
      ops::RandomNormal(root.WithOpName("rnd"), {10}, DataType::DT_DOUBLE);
  Output rnd3 =
      ops::RandomNormal(root.WithOpName("rnd"), {1, 2, 3}, DataType::DT_FLOAT);
  auto enqueue5 =
      ops::QueueEnqueue(root.WithOpName("Enqueue5"), q5, {rnd, rnd2, rnd3});
  auto dequeue5 = ops::QueueDequeue(
      root.WithOpName("Dequeue5"), q5,
      {DataType::DT_FLOAT, DataType::DT_DOUBLE, DataType::DT_FLOAT});

  GrapplerItem item;
  TF_CHECK_OK(root.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));

  const auto props1 = properties.GetOutputProperties("Dequeue1");
  ASSERT_EQ(1, props1.size());
  EXPECT_EQ("float: [3,7]", PropToString(props1[0]));

  const auto props2 = properties.GetOutputProperties("Dequeue2");
  ASSERT_EQ(1, props2.size());
  EXPECT_EQ("float: [3,7]", PropToString(props2[0]));

  const auto props3 = properties.GetOutputProperties("Dequeue3");
  ASSERT_EQ(1, props3.size());
  EXPECT_EQ("float: [3,7]", PropToString(props3[0]));

  // The dequeue3 op shape is unknown. The square2 op shape is known. Verify
  // that we merge the 2 properly to determine the shape of the data coming out
  // of the queue.
  const auto props4 = properties.GetOutputProperties("Dequeue4");
  ASSERT_EQ(1, props4.size());
  EXPECT_EQ("float: [3,7]", PropToString(props4[0]));

  // The dequeue5 op shape is known.
  const auto props5 = properties.GetOutputProperties("Dequeue5");
  ASSERT_EQ(3, props5.size());
  EXPECT_EQ("float: [3,7]", PropToString(props5[0]));
  EXPECT_EQ("double: [10]", PropToString(props5[1]));
  EXPECT_EQ("float: [1,2,3]", PropToString(props5[2]));
}

TEST_F(GraphPropertiesTest, MergeWithoutLoops) {
  // Test graph produced in python using:
  /*
    with tf.Graph().as_default():
      x = tf.constant(2)
      y = tf.constant(5)
      z = tf.ones([1,1,1])
      def f1(): return tf.concat([z, z], axis=0)
      def f2(): return tf.concat([z, z], axis=1)
      r = tf.cond(tf.less(x, y), f1, f2)
      tf.concat([r, r], axis=2)
      with open('/tmp/graph.pbtxt', 'w') as f:
        f.write(str(tf.get_default_graph().as_graph_def()))
   */

  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "merge_without_loops.pbtxt");
  TF_CHECK_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));

  std::vector<string> nodes{"cond/Merge", "cond/concat", "cond/concat_1"};
  std::vector<string> expected_outputs{"float: [-1,-1,1]", "float: [2,1,1]",
                                       "float: [1,2,1]"};
  for (int i = 0; i < nodes.size(); i++) {
    const auto props = properties.GetOutputProperties(nodes[i]);
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT, prop.dtype());
    EXPECT_EQ(expected_outputs[i], PropToString(prop));
  }

  // The "Less" node should be fed by 2 int32 scalar constant values.
  const auto props = properties.GetInputProperties("Less");
  EXPECT_EQ(2, props.size());
  for (int i = 0; i < props.size(); ++i) {
    EXPECT_EQ(DT_INT32, props[i].dtype());
    EXPECT_TRUE(props[i].has_value());
    EXPECT_EQ("int32: []", PropToString(props[i]));
  }
}

TEST_F(GraphPropertiesTest, WhileLoop) {
  // Test graph produced in python using:
  /*
     with tf.Graph().as_default():
       i0 = tf.constant(0)
       m0 = tf.placeholder([-1, 2])
       c = lambda i, m: i < 10
       b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
       r = tf.while_loop(
              c, b, loop_vars=[i0, m0],
              shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])
       with open('/tmp/graph.pbtxt', 'w') as f:
         f.write(str(tf.get_default_graph().as_graph_def()))
  */

  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "while_loop.pbtxt");
  TF_CHECK_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));

  std::vector<string> nodes{"while/Merge_1", "while/NextIteration_1",
                            "while/Exit_1"};
  for (const string& node : nodes) {
    const auto props = properties.GetOutputProperties(node);
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT, prop.dtype());
    EXPECT_EQ("float: [-1,2]", PropToString(prop));
  }

  // The loop outputs batch dim should be different from the input batch dim
  // since we concatenated along the batch dim.
  auto shape_in = properties.GetOutputProperties("ones").at(0).shape();
  auto shape_out = properties.GetOutputProperties("while/Exit_1").at(0).shape();
  EXPECT_GE(-2, shape_in.dim(0).size());
  EXPECT_GE(-2, shape_out.dim(0).size());
  EXPECT_NE(shape_in.dim(0).size(), shape_out.dim(0).size());
}

TEST_F(GraphPropertiesTest, NestedLoop) {
  // Test graph produced in python using:
  /*
    with tf.Graph().as_default():
      i0 = tf.constant(0)

      def inner(j, y):
        def inner_cond(j, y):
          return j < 3

        def inner_body(j, y):
          return j+1, tf.concat([y, y], axis=2)

        return tf.while_loop(inner_cond, inner_body, loop_vars=[j, y],
                             shape_invariants=[i0.get_shape(),
                                              tf.TensorShape([None, 1, None])])

      def outer_cond(i, x):
        return i < 3

      def outer_body(i, x):
        j, y = inner(0, x)
        return i+1, tf.concat([x, x], axis=0)

      r = tf.while_loop(outer_cond, outer_body,
                        loop_vars=[i0, tf.ones([1, 1, 1])],
                        shape_invariants=[i0.get_shape(),
                                          tf.TensorShape([None, 1, None])])

      with open('/tmp/graph.pbtxt', 'w') as f:
        f.write(str(tf.get_default_graph().as_graph_def()))
  */

  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "nested_loop.pbtxt");
  TF_CHECK_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));

  std::vector<string> outer_nodes{"while/Merge_1", "while/NextIteration_1",
                                  "while/Exit_1"};
  std::vector<string> inner_nodes{"while/while/Merge_1",
                                  "while/while/NextIteration_1",
                                  "while/while/Exit_1"};
  for (const string& node : outer_nodes) {
    const auto props = properties.GetOutputProperties(node);
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT, prop.dtype());
    EXPECT_EQ("float: [-1,1,1]", PropToString(prop));
  }
  for (const string& node : inner_nodes) {
    const auto props = properties.GetOutputProperties(node);
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT, prop.dtype());
    EXPECT_EQ("float: [-1,1,-1]", PropToString(prop));
  }
}

TEST_F(GraphPropertiesTest, LoopsAndQueues) {
  // Test graph produced in python using:
  /*
    with tf.Graph().as_default():
      i0 = tf.constant(0)
      q = tf.FIFOQueue(1, "float")

      def inner(j, y):
        def inner_cond(j, y):
          return j < 3

        def inner_body(j, y):
          return j+1, tf.concat([y, y], axis=0)

        return tf.while_loop(inner_cond, inner_body,
                             loop_vars=[j, y],
                             shape_invariants=[i0.get_shape(),
                                               tf.TensorShape(None)])

      def outer_cond(i, x):
        return i < 3

      def outer_body(i, x):
        q.enqueue(x)
        y = tf.concat([x, x], axis=2)
        inner(0, q.dequeue())
        return i+1, y

      i, z = tf.while_loop(outer_cond, outer_body,
                           loop_vars=[i0, tf.ones([1, 1, 1])],
                           shape_invariants=[i0.get_shape(),
                                             tf.TensorShape([None, 1, None])])

      with open('/tmp/graph.pbtxt', 'w') as f:
        f.write(str(tf.get_default_graph().as_graph_def()))
   */

  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "loops_and_queues.pbtxt");
  TF_CHECK_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));

  std::vector<string> outer_nodes{"while/Merge_1", "while/NextIteration_1",
                                  "while/Exit_1"};
  std::vector<string> inner_nodes{"while/while/Merge_1",
                                  "while/while/NextIteration_1",
                                  "while/while/Exit_1"};
  for (const string& node : outer_nodes) {
    const auto props = properties.GetOutputProperties(node);
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT, prop.dtype());
    EXPECT_EQ("float: [1,1,-1]", PropToString(prop));
  }
  for (const string& node : inner_nodes) {
    const auto props = properties.GetOutputProperties(node);
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT, prop.dtype());
    EXPECT_EQ("float: [-1,1,-1]", PropToString(prop));
  }
}

TEST_F(GraphPropertiesTest, LoopsAndResourceVars) {
  // Test graph produced in python using:
  /*
    with tf.Graph().as_default():
      i0 = tf.constant(0)
      with tf.variable_scope(VariableScope(reuse=None, use_resource=True)):
        v = tf.get_variable(initializer=i0, name='loop_var')

      def inner(j, y):
        def inner_cond(j, y):
          return j < 3

        def inner_body(j, y):
          return j + 1, y + y

        return tf.while_loop(inner_cond, inner_body, loop_vars=[j, y])

      def outer_cond(i, x):
        return i < 3

      def outer_body(i, x):
        y = x + x
        inner(0, v)
        return i + 1, y

      v, z = tf.while_loop(outer_cond, outer_body,
                           loop_vars=[v, tf.constant(1)])

      with open('/tmp/graph.pbtxt', 'w') as f:
        f.write(str(tf.get_default_graph().as_graph_def()))
  */

  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "loops_and_resource_vars.pbtxt");
  TF_CHECK_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));

  std::vector<string> outer_nodes{"while/Merge_1", "while/NextIteration_1",
                                  "while/Exit_1"};
  std::vector<string> inner_nodes{"while/while/Merge_1",
                                  "while/while/NextIteration_1",
                                  "while/while/Exit_1"};
  for (const string& node : outer_nodes) {
    const auto props = properties.GetOutputProperties(node);
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_INT32, prop.dtype());
    EXPECT_EQ("int32: []", PropToString(prop));
  }
  for (const string& node : inner_nodes) {
    const auto props = properties.GetOutputProperties(node);
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_INT32, prop.dtype());
    EXPECT_EQ("int32: []", PropToString(prop));
  }
}

TEST_F(GraphPropertiesTest, QueuesAndLoops) {
  // Test graph produced in python using:
  /*
    with tf.Graph().as_default():
      i0 = tf.constant(0)
      q0 = tf.FIFOQueue(1, "float")
      q0.enqueue(tf.ones([2, 2]))
      q1 = tf.FIFOQueue(1, "float")

      def c(i, m):
        return i < 10

      def b(i, m):
        return i+1, tf.concat([m, m], axis=0)

      i, m = tf.while_loop(
          c, b, loop_vars=[i0,  q0.dequeue()],
          shape_invariants=[i0.get_shape(), tf.TensorShape(None)])

      q1.enqueue(m)
      v = q1.dequeue();
      tf.concat([v, v], axis=1)
      with open('/tmp/graph.pbtxt', 'w') as f:
        f.write(str(tf.get_default_graph().as_graph_def()))
  */

  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "queues_and_loops.pbtxt");
  TF_CHECK_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));

  std::vector<string> nodes{"while/Merge_1", "while/NextIteration_1",
                            "while/Exit_1"};

  for (const string& node : nodes) {
    const auto props = properties.GetOutputProperties(node);
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT, prop.dtype());
    EXPECT_EQ("float: [-1,2]", PropToString(prop));
  }

  const auto props = properties.GetOutputProperties("concat");
  const OpInfo::TensorProperties& prop = props[0];
  EXPECT_EQ(DT_FLOAT, prop.dtype());
  EXPECT_EQ("float: [-1,4]", PropToString(prop));
}

TEST_F(GraphPropertiesTest, InferRestoreOpShape) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output var = ops::Variable(s.WithOpName("var"), TensorShape({128, 256}),
                             DataType::DT_FLOAT);
  Output filename =
      ops::Const(s.WithOpName("filename"), string("model"), TensorShape());
  Output tensor_name =
      ops::Const(s.WithOpName("tensorname"), string("a"), TensorShape());
  Output restore = ops::Restore(s.WithOpName("restore"), filename, tensor_name,
                                DataType::DT_FLOAT);
  Output init_restore = ops::Assign(s.WithOpName("init_restore"), var, restore);

  Output shape_and_slice = ops::Const(s.WithOpName("shape_and_slice"),
                                      string("256 256 0,128:-"), TensorShape());
  Output restore_slice =
      ops::RestoreSlice(s.WithOpName("restore_slice"), filename, tensor_name,
                        shape_and_slice, DataType::DT_FLOAT);
  Output init_restore_slice =
      ops::Assign(s.WithOpName("init_restore_slice"), var, restore_slice);

  Output restore_v2 =
      ops::RestoreSlice(s.WithOpName("restore_v2"), filename, tensor_name,
                        shape_and_slice, DataType::DT_FLOAT);
  Output init_restore_v2 =
      ops::Assign(s.WithOpName("init_restore_v2"), var, restore_v2);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("init_restore");

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));

  const auto restore_props = properties.GetOutputProperties("restore");
  const OpInfo::TensorProperties& restore_prop = restore_props[0];
  EXPECT_EQ(DT_FLOAT, restore_prop.dtype());
  EXPECT_EQ("float: [128,256]", PropToString(restore_prop));

  const auto restore_slice_props =
      properties.GetOutputProperties("restore_slice");
  const OpInfo::TensorProperties& restore_slice_prop = restore_slice_props[0];
  EXPECT_EQ(DT_FLOAT, restore_slice_prop.dtype());
  EXPECT_EQ("float: [128,256]", PropToString(restore_slice_prop));

  const auto restorev2_props = properties.GetOutputProperties("restore_v2");
  const OpInfo::TensorProperties& restorev2_prop = restorev2_props[0];
  EXPECT_EQ(DT_FLOAT, restorev2_prop.dtype());
  EXPECT_EQ("float: [128,256]", PropToString(restorev2_prop));

  // Check input shapes of assign op are propagted correctly.
  const auto input_props = properties.GetInputProperties("init_restore");
  ASSERT_EQ(2, input_props.size());
  const OpInfo::TensorProperties& input_prop = input_props[1];
  EXPECT_EQ(DT_FLOAT, input_prop.dtype());
  EXPECT_EQ("float: [128,256]", PropToString(input_prop));
}

TEST_F(GraphPropertiesTest, InferRestoreOpShape_WithTwoNodesShareSameOutput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output var = ops::Variable(s.WithOpName("var"), PartialTensorShape(),
                             DataType::DT_FLOAT);
  Output var2 = ops::Variable(s.WithOpName("var2"), TensorShape({128, 256}),
                              DataType::DT_FLOAT);
  Output filename =
      ops::Const(s.WithOpName("filename"), string("model"), TensorShape());
  Output tensor_name =
      ops::Const(s.WithOpName("tensorname"), string("a"), TensorShape());
  Output restore = ops::Restore(s.WithOpName("restore"), filename, tensor_name,
                                DataType::DT_FLOAT);
  Output init = ops::Assign(s.WithOpName("init"), var, restore);
  Output init2 = ops::Assign(s.WithOpName("init2"), var2, restore);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("init");
  item.fetch.push_back("init2");

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));

  const auto props = properties.GetOutputProperties("restore");
  const OpInfo::TensorProperties& prop = props[0];
  EXPECT_EQ(DT_FLOAT, prop.dtype());
  EXPECT_EQ("float: [128,256]", PropToString(prop));
}

TEST_F(GraphPropertiesTest, FunctionStaticShapeInference) {
  // Test graph produced in python using:
  /*
    @function.Defun(*[tf.float32] * 2, noinline=True)
    def MyAdd(x, y):
      return tf.add(x,y)

    with tf.Graph().as_default():
      x = tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      y = tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      z = MyAdd(x, y)
      z = MyAdd(x, z)
  */
  // Check that the shape inference code infers what it can.
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "simple_function.pbtxt");
  TF_CHECK_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));
  const auto out_props = properties.GetOutputProperties("MyAdd_55e046a8");
  const OpInfo::TensorProperties& out_prop = out_props[0];
  EXPECT_EQ(DT_FLOAT, out_prop.dtype());
  EXPECT_TRUE(out_prop.shape().unknown_rank());

  const auto in_props = properties.GetInputProperties("MyAdd_55e046a8");
  const OpInfo::TensorProperties& in_prop = in_props[0];
  EXPECT_EQ(DT_FLOAT, in_prop.dtype());
  EXPECT_FALSE(in_prop.shape().unknown_rank());
  EXPECT_EQ(2, in_prop.shape().dim_size());
  EXPECT_EQ(1, in_prop.shape().dim(0).size());
  EXPECT_EQ(2, in_prop.shape().dim(1).size());
}

TEST_F(GraphPropertiesTest, SymbolicShapes) {
  // Build a simple graph with placeholders of unknown dimensions. These
  // dimensions will be encoded symbolically.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a =
      ops::Placeholder(s.WithOpName("a"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1, -1})));
  Output b =
      ops::Placeholder(s.WithOpName("b"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1})));
  Output c = ops::Identity(s.WithOpName("c"), a);
  Output d = ops::Identity(s.WithOpName("d"), b);
  Output e = ops::Add(s.WithOpName("e"), c, d);
  Output f = ops::Add(s.WithOpName("f"), a, c);

  Output zero = ops::Const(s.WithOpName("zero"), 0.0f, {});
  Output g = ops::Shape(s.WithOpName("g"), c);
  Output h = ops::Fill(s.WithOpName("h"), g, zero);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));
  const auto shape_a = properties.GetOutputProperties("a").at(0).shape();
  const auto shape_c = properties.GetOutputProperties("c").at(0).shape();
  EXPECT_EQ(2, shape_a.dim_size());
  EXPECT_EQ(shape_a.dim_size(), shape_c.dim_size());
  EXPECT_GE(-2, shape_a.dim(0).size());
  EXPECT_EQ(shape_a.dim(0).size(), shape_c.dim(0).size());
  EXPECT_GE(-2, shape_a.dim(1).size());
  EXPECT_EQ(shape_a.dim(1).size(), shape_c.dim(1).size());

  PartialTensorShape shape(shape_a);
  EXPECT_FALSE(shape.IsFullyDefined());
  EXPECT_FALSE(shape.unknown_rank());

  const auto shape_b = properties.GetOutputProperties("b").at(0).shape();
  const auto shape_d = properties.GetOutputProperties("d").at(0).shape();
  EXPECT_EQ(1, shape_b.dim_size());
  EXPECT_EQ(shape_b.dim_size(), shape_d.dim_size());
  EXPECT_GE(-2, shape_b.dim(0).size());
  EXPECT_NE(shape_a.dim(0).size(), shape_b.dim(0).size());
  EXPECT_EQ(shape_b.dim(0).size(), shape_d.dim(0).size());

  const auto shape_e = properties.GetOutputProperties("e").at(0).shape();
  ASSERT_EQ(2, shape_e.dim_size());
  EXPECT_EQ(shape_e.dim(0).size(), shape_c.dim(0).size());
  EXPECT_NE(shape_e.dim(1).size(), shape_c.dim(1).size());
  EXPECT_NE(shape_e.dim(0).size(), shape_d.dim(0).size());

  const auto shape_f = properties.GetOutputProperties("f").at(0).shape();
  ASSERT_EQ(2, shape_f.dim_size());
  EXPECT_EQ(shape_f.dim(0).size(), shape_a.dim(0).size());
  EXPECT_EQ(shape_f.dim(1).size(), shape_a.dim(1).size());

  const auto shape_h = properties.GetOutputProperties("h").at(0).shape();
  ASSERT_EQ(2, shape_f.dim_size());
  EXPECT_EQ(shape_h.dim(0).size(), shape_c.dim(0).size());
  EXPECT_EQ(shape_h.dim(1).size(), shape_c.dim(1).size());
}

TEST_F(GraphPropertiesTest, DoNotValidateColocationConstraints) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 1.0f, {1});
  Output b = ops::Const(s.WithOpName("b"), 2.0f, {1});
  Output c = ops::Const(s.WithOpName("c").ColocateWith(a), 3.0f, {1});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  // Create a graph with node a removed (say by some graph optimization
  // pass), noting that node c is colocated with a. This is fine as it
  // is in the late stage of graph execution, the colocation constraints have
  // been validated previously and the device placement of nodes has completed.
  GraphDef optimized_graph;
  for (const auto& node : item.graph.node()) {
    if (node.name() != "a") {
      *optimized_graph.add_node() = node;
    }
  }
  item.graph.Swap(&optimized_graph);
  GraphProperties properties(item);
  // This function should return OK, since it doesn't validate the colocation
  // constraints internally.
  TF_EXPECT_OK(properties.InferStatically(false));
}

TEST_F(GraphPropertiesTest, ShapeTracking) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a =
      ops::Placeholder(s.WithOpName("a"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1, -1})));
  Output b =
      ops::Placeholder(s.WithOpName("b"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1})));
  Output zero = ops::Const(s.WithOpName("zero"), 0.0f, {});
  auto shp = ops::ShapeN(s.WithOpName("shapes"), {a, b});
  Output o1 = ops::Fill(s.WithOpName("o1"), shp[0], zero);
  Output o2 = ops::Fill(s.WithOpName("o2"), shp[1], zero);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));
  const auto shape_a = properties.GetOutputProperties("a").at(0).shape();
  const auto shape_b = properties.GetOutputProperties("b").at(0).shape();
  const auto shape_o1 = properties.GetOutputProperties("o1").at(0).shape();
  const auto shape_o2 = properties.GetOutputProperties("o2").at(0).shape();
  EXPECT_EQ(shape_a.DebugString(), shape_o1.DebugString());
  EXPECT_EQ(shape_b.DebugString(), shape_o2.DebugString());
}

TEST_F(GraphPropertiesTest, FedNodes) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  {
    // Conservative shape analysis: the shape of fed ports should be unknown
    GraphProperties properties(item);
    Status s = properties.InferStatically(false);
    TF_CHECK_OK(s);
    for (const auto& node : item.graph.node()) {
      if (node.op() == "Const") {
        continue;
      }
      const auto in_props = properties.GetInputProperties(node.name());
      EXPECT_EQ(1, in_props.size());
      const OpInfo::TensorProperties& in_prop = in_props[0];
      const auto out_props = properties.GetOutputProperties(node.name());
      EXPECT_EQ(1, out_props.size());
      const OpInfo::TensorProperties& out_prop = out_props[0];

      if (node.name() == "x") {
        // x is fed: its input should have a known shape, while its output
        // doesn't
        EXPECT_FALSE(in_prop.shape().unknown_rank());
        EXPECT_EQ(1, in_prop.shape().dim_size());
        EXPECT_EQ(2, in_prop.shape().dim(0).size());
        EXPECT_TRUE(out_prop.shape().unknown_rank());
      } else if (node.op() == "Square" || node.op() == "AddN") {
        // These nodes are in the fanout of x: their shapes should be unknown.
        EXPECT_TRUE(in_prop.shape().unknown_rank());
        EXPECT_TRUE(out_prop.shape().unknown_rank());
      }
    }
  }
  {
    // Optimistic shape analysis: the shape of fed ports should be derived from
    // the shape of the fanin.
    GraphProperties properties(item);
    Status s = properties.InferStatically(true);
    TF_CHECK_OK(s);
    for (const auto& node : item.graph.node()) {
      if (node.op() == "Square" || node.op() == "AddN") {
        const auto in_props = properties.GetInputProperties(node.name());
        EXPECT_EQ(1, in_props.size());
        const OpInfo::TensorProperties& in_prop = in_props[0];
        EXPECT_EQ(DT_FLOAT, in_prop.dtype());
        EXPECT_FALSE(in_prop.shape().unknown_rank());
        EXPECT_EQ(2, in_prop.shape().dim_size());
        const auto out_props = properties.GetOutputProperties(node.name());
        EXPECT_EQ(1, out_props.size());
        const OpInfo::TensorProperties& out_prop = out_props[0];
        EXPECT_EQ(in_prop.DebugString(), out_prop.DebugString());
      }
    }
  }
}

TEST_F(GraphPropertiesTest, Performance) {
  // Load a large graph with many nested loops to make sure we can infer shapes
  // quickly.
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "large_graph.pbtxt.html");
  TF_CHECK_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically(false));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
