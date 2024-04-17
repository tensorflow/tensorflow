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
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/inputs/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#ifdef INTEL_MKL
#include "tensorflow/core/graph/mkl_graph_util.h"
#endif

namespace tensorflow {
namespace grappler {
namespace {

using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

const char kTestDataPath[] = "core/grappler/costs/graph_properties_testdata";

REGISTER_OP("TestOpWithNoInferenceFn")
    .Input("x: float")
    .Output("y: float")
    .Doc(R"doc(
Test op with no Inference Function registered.
x: input
y: output
)doc");

class GraphPropertiesTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Provision a single machine with 3 cpu cores
    cluster_.reset(new SingleMachine(5 * 60, 3, 0));
    TF_ASSERT_OK(cluster_->Provision());

    // This function is simply
    // out = Fill(shape, value), but
    // Fill requires values in the shape input, not just shape of it, to infer
    // output shape.
    auto f = FunctionDefHelper::Create(
        // Name
        "MyFillFunc",
        // Inputs
        {"shape: int32", "value: float"},
        // Outputs
        {"out: float"},
        // Attrs
        {},
        // Nodes
        {
            {{"a"},
             "Fill",
             {"shape", "value"},
             {{"T", DataType::DT_FLOAT}, {"index_type", DataType::DT_INT32}}},
        },
        // Returns
        {{"out", "a:output:0"}});
    function_lib_.add_function()->Swap(&f);
  }

  void TearDown() override {
    TF_ASSERT_OK(cluster_->Shutdown());
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
                           std::max<int64_t>(p.shape().dim(i).size(), -1));
      }
      strings::StrAppend(&s, "]");
    }
    return s;
  }

  // Compare values of integer (DT_INT32 or DT_INT64) tensor against expected
  // ones.
  void ExpectTensorValues(const std::vector<int64_t>& expected,
                          const TensorProto& tensor_proto_to_compare) {
    Tensor tensor;
    ASSERT_TRUE(tensor.FromProto(tensor_proto_to_compare));
    EXPECT_EQ(expected.size(), tensor.NumElements());
    // We're interested in only integer tensors as only shapes are exported as
    // graph properties values.
    ASSERT_TRUE(tensor.dtype() == DT_INT32 || tensor.dtype() == DT_INT64);
    if (tensor.dtype() == DT_INT32) {
      for (int i = 0; i < tensor.NumElements(); i++) {
        EXPECT_EQ(expected[i], tensor.flat<int32>()(i));
      }
    } else {
      for (int i = 0; i < tensor.NumElements(); i++) {
        EXPECT_EQ(expected[i], tensor.flat<int64_t>()(i));
      }
    }
  }

  // Compare values of float (DT_FLOAT) tensor against expected
  // ones.
  void ExpectFloatTensorValues(const std::vector<float>& expected,
                               const TensorProto& tensor_proto_to_compare) {
    Tensor tensor;
    ASSERT_TRUE(tensor.FromProto(tensor_proto_to_compare));
    EXPECT_EQ(expected.size(), tensor.NumElements());
    ASSERT_EQ(tensor.dtype(), DT_FLOAT);
    for (int i = 0; i < tensor.NumElements(); i++) {
      EXPECT_EQ(expected[i], tensor.flat<float>()(i));
    }
  }

  std::unique_ptr<SingleMachine> cluster_;
  FunctionDefLibrary function_lib_;
};

TEST_F(GraphPropertiesTest, StaticProperties) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphProperties properties(item);
  Status s = properties.InferStatically(true);
  TF_ASSERT_OK(s);

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
      EXPECT_EQ(in_prop.dtype(), out_props[0].dtype());
      EXPECT_EQ(in_prop.shape().DebugString(),
                out_props[0].shape().DebugString());
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
  TF_ASSERT_OK(s);

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

TEST_F(GraphPropertiesTest, Clear) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphProperties properties(item);
  Status s = properties.InferStatically(true);
  TF_ASSERT_OK(s);

  EXPECT_TRUE(properties.has_properties());
  properties.Clear();
  EXPECT_FALSE(properties.has_properties());
}

TEST_F(GraphPropertiesTest, DynamicProperties) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphProperties properties(item);
  TF_ASSERT_OK(cluster_->Initialize(item));
  Status s = properties.InferDynamically(cluster_.get());
  TF_ASSERT_OK(s);

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
#ifdef INTEL_MKL
        if (!NativeFormatEnabled()) {
          // Intel MKL AddN OP would have two output.
          // One is the real output, another one for MKL metadata
          EXPECT_EQ(2, out_props.size());
        } else {
          EXPECT_EQ(1, out_props.size());
        }
#else
        EXPECT_EQ(1, out_props.size());
#endif  // INTEL_MKL
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

// A test op that outputs different shape based on input_tensor in the shape
// inference context.
REGISTER_OP("DetectInputValueInShapeInferenceOp")
    .Input("a: T")
    .Output("o: T")
    .Attr("T: {numbertype, bool}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (c->input_tensor(0)) {
        // 10x10 if input_tensor is given to the inference context.
        c->set_output(0, c->Matrix(10, 10));
        return absl::OkStatus();
      }
      // unknown rank if input_tensor is not provided.
      return shape_inference::UnknownShape(c);
    });

// Helper class for testing Const tensor skip.
class ConstTensorSkipTestCase {
 public:
  ConstTensorSkipTestCase(const DataType data_type,
                          const std::vector<int64_t> shape, const double value,
                          const bool expected)
      : data_type_(data_type),
        shape_(shape),
        value_(value),
        expected_(expected) {}

  void RunTestAndValidate() const {
    LOG(INFO) << "Run Const tensor skip test: "
              << "data_type: " << data_type_ << ", shape: {"
              << absl::StrJoin(shape_, ",") << "}, value: " << value_
              << ", expected: " << expected_;
    // Build a graph with Const --> Identity --> Detect.
    GrapplerItem item;
    const absl::Span<const int64_t> shape_array_slice(shape_);
    Tensor const_tensor_value(data_type_, TensorShape(shape_array_slice));
    // Fill the const tensor value based on data type.
    switch (data_type_) {
      case DT_INT32:
        test::FillIota<int32>(&const_tensor_value, static_cast<int32>(value_));
        break;
      case DT_INT64:
        test::FillIota<int64_t>(&const_tensor_value,
                                static_cast<int64_t>(value_));
        break;
      case DT_FLOAT:
        test::FillIota<float>(&const_tensor_value, static_cast<float>(value_));
        break;
      case DT_DOUBLE:
        test::FillIota<double>(&const_tensor_value,
                               static_cast<double>(value_));
        break;
      case DT_BFLOAT16:
        test::FillIota<Eigen::bfloat16>(&const_tensor_value,
                                        static_cast<Eigen::bfloat16>(value_));
        break;
      default:
        CHECK(false) << "Unsupported data type (" << data_type_
                     << ") in this test.";
        break;
    }
    TF_ASSERT_OK(NodeDefBuilder("const", "Const")
                     .Attr("dtype", data_type_)
                     .Attr("value", const_tensor_value)
                     .Finalize(item.graph.add_node()));
    TF_ASSERT_OK(NodeDefBuilder("const_identity", "Identity")
                     .Attr("dtype", data_type_)
                     .Input("const", 0, data_type_)
                     .Finalize(item.graph.add_node()));
    TF_ASSERT_OK(NodeDefBuilder("detect", "DetectInputValueInShapeInferenceOp")
                     .Attr("T", data_type_)
                     .Input("const_identity", 0, data_type_)
                     .Finalize(item.graph.add_node()));
    item.fetch.push_back("const");
    item.fetch.push_back("const_identity");
    item.fetch.push_back("detect");

    // Run static shape inference.
    GraphProperties graph_properties(item);
    TF_ASSERT_OK(graph_properties.InferStatically(false));

    // Extract input / output properties of interest.
    const auto& const_output = graph_properties.GetOutputProperties("const");
    EXPECT_EQ(1, const_output.size());
    const OpInfo::TensorProperties& const_output0 = const_output[0];
    const auto& const_identity_input =
        graph_properties.GetInputProperties("const_identity");
    EXPECT_EQ(1, const_identity_input.size());
    const OpInfo::TensorProperties& const_identity_input0 =
        const_identity_input[0];
    const auto& const_identity_output =
        graph_properties.GetOutputProperties("const_identity");
    EXPECT_EQ(1, const_identity_output.size());
    const OpInfo::TensorProperties& const_identity_output0 =
        const_identity_output[0];
    EXPECT_TRUE(const_output0.has_value());
    EXPECT_TRUE(const_identity_input0.has_value());
    EXPECT_TRUE(const_identity_output0.has_value());
    const auto& detect_input = graph_properties.GetInputProperties("detect");
    EXPECT_EQ(1, detect_input.size());
    const OpInfo::TensorProperties& detect_input0 = detect_input[0];
    const auto& detect_output = graph_properties.GetOutputProperties("detect");
    EXPECT_EQ(1, detect_output.size());
    const OpInfo::TensorProperties& detect_output0 = detect_output[0];

    // Tensor protos are propagated, regardless of types and sizes.
    EXPECT_TRUE(const_output0.has_value());
    EXPECT_TRUE(const_identity_input0.has_value());
    EXPECT_TRUE(const_identity_output0.has_value());
    EXPECT_TRUE(detect_input0.has_value());

    // Detect op outputs 10x10 matrix if it has input_tensor in the shape
    // inference context. Otherwise, unknown rank.
    if (expected_) {
      EXPECT_EQ(detect_output0.shape().dim_size(), 2);
      EXPECT_EQ(detect_output0.shape().dim(0).size(), 10);
      EXPECT_EQ(detect_output0.shape().dim(1).size(), 10);
    } else {
      EXPECT_TRUE(detect_output0.shape().unknown_rank());
    }
  }

 private:
  DataType data_type_;
  std::vector<int64_t> shape_;
  double value_;
  bool expected_;
};

TEST_F(GraphPropertiesTest, SkipInstantiatingConstTensor) {
  // We skip const tensor value propagation in shape inference, if a const
  // tensor is too large.
  std::vector<ConstTensorSkipTestCase> test_cases = {
      // data_type, shape, value, bool: propagate const?
      {DT_INT32, {16, 8}, 1, true},      // 128 elements; smaller than threshold
      {DT_INT32, {1, 129}, 2, false},    // 129 elements; larger than threshold
      {DT_INT64, {8, 8}, 3, true},       // 64 elements; smaller than threshold
      {DT_INT64, {128, 2}, 0, false},    // 256 elements; larger than threshold
      {DT_FLOAT, {16, 8}, 1.0, true},    // integer value for float tensor
      {DT_FLOAT, {16, 8}, 1.3, true},    // fractional value (1.3)
      {DT_FLOAT, {1, 129}, 0.7, false},  // fractional value (0.7)
      {DT_DOUBLE, {16, 8}, 1.0, true},   // integer value for float tensor
      {DT_DOUBLE, {16, 8}, 1.3, true},   // fractional value (1.3)
      {DT_DOUBLE, {1, 129}, 0.7, false},    // fractional value (0.7)
      {DT_BFLOAT16, {16, 8}, 1.0, true},    // integer value for float tensor
      {DT_BFLOAT16, {16, 8}, 1.3, true},    // fractional value (1.3)
      {DT_BFLOAT16, {1, 129}, 0.7, false},  // fractional value (0.7)
  };
  for (const auto& test_case : test_cases) {
    test_case.RunTestAndValidate();
  }
}

TEST_F(GraphPropertiesTest, Variables) {
  GrapplerItem item;
  TF_ASSERT_OK(NodeDefBuilder("Var", "Variable")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", TensorShape({3, 7}))
                   .Finalize(item.graph.add_node()));
  item.fetch.push_back("Var");

  Tensor initial_val(DT_FLOAT, TensorShape({3, 7}));
  test::FillIota<float>(&initial_val, 0);
  TF_ASSERT_OK(NodeDefBuilder("InitialVal", "Const")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("value", initial_val)
                   .Finalize(item.graph.add_node()));
  TF_ASSERT_OK(NodeDefBuilder("InitVar", "Assign")
                   .Input("Var", 0, DT_FLOAT_REF)
                   .Input("InitialVal", 0, DT_FLOAT)
                   .Finalize(item.graph.add_node()));
  item.init_ops.push_back("InitVar");

  {
    GraphProperties static_properties(item);
    TF_ASSERT_OK(static_properties.InferStatically(false));

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
    TF_ASSERT_OK(cluster_->Initialize(item));
    GraphProperties dynamic_properties(item);
    TF_ASSERT_OK(dynamic_properties.InferDynamically(cluster_.get()));

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

TEST_F(GraphPropertiesTest, ReadVariableOpAfterEnter) {
  GrapplerItem item;
  TF_ASSERT_OK(NodeDefBuilder("Var", "VarHandleOp")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", TensorShape({3, 7}))
                   .Finalize(item.graph.add_node()));
  TF_ASSERT_OK(NodeDefBuilder("Enter", "Enter")
                   .Attr("T", DT_RESOURCE)
                   .Attr("frame_name", "while_context")
                   .Attr("is_constant", true)
                   .Attr("parallel_iterations", 10)
                   .Input("Var", 0, DT_RESOURCE)
                   .Finalize(item.graph.add_node()));
  TF_ASSERT_OK(NodeDefBuilder("ReadVariableOpAfterEnter", "ReadVariableOp")
                   .Attr("dtype", DT_FLOAT)
                   .Input("Enter", 0, DT_RESOURCE)
                   .Finalize(item.graph.add_node()));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto props = properties.GetOutputProperties("ReadVariableOpAfterEnter");
  EXPECT_EQ(1, props.size());
  const OpInfo::TensorProperties& prop = props[0];
  EXPECT_EQ(DT_FLOAT, prop.dtype());
  EXPECT_FALSE(prop.shape().unknown_rank());
  EXPECT_EQ(2, prop.shape().dim_size());
  EXPECT_EQ(3, prop.shape().dim(0).size());
  EXPECT_EQ(7, prop.shape().dim(1).size());
}

TEST_F(GraphPropertiesTest, VarHandles) {
  GrapplerItem item;
  TF_ASSERT_OK(NodeDefBuilder("Var", "VarHandleOp")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", TensorShape({3, 7}))
                   .Finalize(item.graph.add_node()));

  TF_ASSERT_OK(NodeDefBuilder("VarRead", "ReadVariableOp")
                   .Attr("dtype", DT_FLOAT)
                   .Input("Var", 0, DT_RESOURCE)
                   .Finalize(item.graph.add_node()));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  const auto props = properties.GetOutputProperties("VarRead");
  EXPECT_EQ(1, props.size());
  const OpInfo::TensorProperties& prop = props[0];
  EXPECT_EQ(DT_FLOAT, prop.dtype());
  EXPECT_FALSE(prop.shape().unknown_rank());
  EXPECT_EQ(2, prop.shape().dim_size());
  EXPECT_EQ(3, prop.shape().dim(0).size());
  EXPECT_EQ(7, prop.shape().dim(1).size());
}

TEST_F(GraphPropertiesTest, WhileLoopWithVarHandleOpInput) {
  // Test graph is first generated in python using:
  /*
    i0 = tf.constant(0)
    v = tf.get_variable(initializer=i0, name='loop_var', use_resource=True)
    def cond(i, x):
      return i < 3
    def body(i, x):
      return i + 1, x + x
    v, y = tf.while_loop(cond, body, loop_vars=[v, tf.constant(1)])
  */
  // and then modified by hand such that the ReadVariableOp is inside the loop
  // body instead of outside the while loop (which is the case when constructed
  // using the python API), such that we have the following pattern: VarHandleOp
  // -> Enter -> Switch -> ReadVariableOp -> other parts of loop body. Note
  // DT_RESOURCE is passed all the way until ReadVariableOp.
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "while_loop_var_handle_op.pbtxt");
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  std::vector<string> resource_nodes{
      "loop_var",       "while/Enter",         "while/Merge", "while/Switch",
      "while/Identity", "while/NextIteration", "while/Exit"};
  for (const string& node : resource_nodes) {
    const auto props = properties.GetOutputProperties(node);
    EXPECT_GE(props.size(), 1);  // Merge has 2 outputs.
    EXPECT_EQ("resource: []", PropToString(props[0]));
  }

  // After ReadVariableOp, the shape should be recovered.
  const auto props = properties.GetOutputProperties("while/ReadVariableOp");
  EXPECT_EQ(1, props.size());
  EXPECT_EQ("int32: []", PropToString(props[0]));
}

TEST_F(GraphPropertiesTest, QueueWithOnlyDequeue_NoShapeAttr) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto q1 = ops::FIFOQueue(root.WithOpName("Queue1"), {DataType::DT_FLOAT});
  auto dequeue1 =
      ops::QueueDequeue(root.WithOpName("Dequeue1"), q1, {DataType::DT_FLOAT});

  GrapplerItem item;
  TF_ASSERT_OK(root.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  const auto props1 = properties.GetOutputProperties("Dequeue1");
  ASSERT_EQ(1, props1.size());
  EXPECT_EQ("float: ?", PropToString(props1[0]));
}

TEST_F(GraphPropertiesTest, QueueWithOnlyDequeue_ShapeAttr) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto q1 = ops::FIFOQueue(root.WithOpName("Queue1"), {DataType::DT_FLOAT},
                           ops::FIFOQueue::Attrs().Shapes({{3, 7, 1}}));
  auto dequeue1 =
      ops::QueueDequeue(root.WithOpName("Dequeue1"), q1, {DataType::DT_FLOAT});

  GrapplerItem item;
  TF_ASSERT_OK(root.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  const auto props1 = properties.GetOutputProperties("Dequeue1");
  ASSERT_EQ(1, props1.size());
  EXPECT_EQ("float: [3,7,1]", PropToString(props1[0]));
}

TEST_F(GraphPropertiesTest, QueueWithOnlyDequeue_PartialShapeAttr) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto q1 = ops::FIFOQueue(root.WithOpName("Queue1"), {DataType::DT_FLOAT},
                           ops::FIFOQueue::Attrs().Shapes({{3, 7, -1}}));
  auto dequeue1 =
      ops::QueueDequeue(root.WithOpName("Dequeue1"), q1, {DataType::DT_FLOAT});

  GrapplerItem item;
  TF_ASSERT_OK(root.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  const auto props1 = properties.GetOutputProperties("Dequeue1");
  ASSERT_EQ(1, props1.size());
  EXPECT_EQ("float: [3,7,-1]", PropToString(props1[0]));
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

  auto q4 =
      ops::RandomShuffleQueue(root.WithOpName("Queue4"), {DataType::DT_FLOAT});
  auto enqueue4 = ops::QueueEnqueue(root.WithOpName("Enqueue4"), q4, {square2});
  auto enqueue4_2 =
      ops::QueueEnqueue(root.WithOpName("Enqueue4_2"), q4, {dequeue2[0]});
  auto dequeue4 =
      ops::QueueDequeue(root.WithOpName("Dequeue4"), q4, {DataType::DT_FLOAT});

  // Create a queue that takes in three tensors.
  auto q5 = ops::RandomShuffleQueue(
      root.WithOpName("Queue5"),
      {DataType::DT_FLOAT, DataType::DT_DOUBLE, DataType::DT_FLOAT});
  Output rnd2 =
      ops::RandomNormal(root.WithOpName("rnd2"), {10}, DataType::DT_DOUBLE);
  Output rnd3 =
      ops::RandomNormal(root.WithOpName("rnd3"), {1, 2, 3}, DataType::DT_FLOAT);
  auto enqueue5 =
      ops::QueueEnqueue(root.WithOpName("Enqueue5"), q5, {rnd, rnd2, rnd3});
  auto dequeue5 = ops::QueueDequeue(
      root.WithOpName("Dequeue5"), q5,
      {DataType::DT_FLOAT, DataType::DT_DOUBLE, DataType::DT_FLOAT});

  GrapplerItem item;
  TF_ASSERT_OK(root.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  const auto props1 = properties.GetOutputProperties("Dequeue1");
  ASSERT_EQ(1, props1.size());
  EXPECT_EQ("float: [3,7]", PropToString(props1[0]));

  const auto props2 = properties.GetOutputProperties("Dequeue2");
  ASSERT_EQ(1, props2.size());
  EXPECT_EQ("float: [3,7]", PropToString(props2[0]));

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
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

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
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

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
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

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
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

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
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

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
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

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
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("init_restore");

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

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

  // Check input shapes of assign op are propagated correctly.
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
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("init");
  item.fetch.push_back("init2");

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  const auto props = properties.GetOutputProperties("restore");
  const OpInfo::TensorProperties& prop = props[0];
  EXPECT_EQ(DT_FLOAT, prop.dtype());
  EXPECT_EQ("float: [128,256]", PropToString(prop));
}

TEST_F(GraphPropertiesTest, TensorAsShapesPropagation) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), {5, 7}, {2});
  Output a1 = ops::Identity(s.WithOpName("a1"), a);
  Output b = ops::Const(s.WithOpName("b"), 99, {});
  Output b1 = ops::Identity(s.WithOpName("b1"), b);
  Output c = ops::Const(s.WithOpName("c"), 1, {4, 4, 4});
  Output c1 = ops::Identity(s.WithOpName("c1"), c);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  // Check output shapes.
  EXPECT_EQ("int32: [2]", PropToString(properties.GetOutputProperties("a")[0]));
  EXPECT_EQ("int32: [2]",
            PropToString(properties.GetOutputProperties("a1")[0]));
  EXPECT_EQ("int32: []", PropToString(properties.GetOutputProperties("b")[0]));
  EXPECT_EQ("int32: []", PropToString(properties.GetOutputProperties("b1")[0]));
  EXPECT_EQ("int32: [4,4,4]",
            PropToString(properties.GetOutputProperties("c")[0]));
  EXPECT_EQ("int32: [4,4,4]",
            PropToString(properties.GetOutputProperties("c1")[0]));

  // Check has_value.
  EXPECT_TRUE(properties.GetOutputProperties("a")[0].has_value());
  EXPECT_TRUE(properties.GetInputProperties("a1")[0].has_value());
  EXPECT_TRUE(properties.GetOutputProperties("a1")[0].has_value());
  EXPECT_TRUE(properties.GetOutputProperties("b")[0].has_value());
  EXPECT_TRUE(properties.GetInputProperties("b1")[0].has_value());
  EXPECT_TRUE(properties.GetOutputProperties("b1")[0].has_value());
  EXPECT_TRUE(properties.GetOutputProperties("c")[0].has_value());
  EXPECT_TRUE(properties.GetInputProperties("c1")[0].has_value());
  // Note that we propagate tensor value of only 1D vector and scalar.
  EXPECT_TRUE(properties.GetOutputProperties("c1")[0].has_value());

  // Check values.
  ExpectTensorValues({5, 7}, properties.GetOutputProperties("a")[0].value());
  ExpectTensorValues({5, 7}, properties.GetInputProperties("a1")[0].value());
  ExpectTensorValues({5, 7}, properties.GetOutputProperties("a1")[0].value());
  ExpectTensorValues({99}, properties.GetOutputProperties("b")[0].value());
  ExpectTensorValues({99}, properties.GetInputProperties("b1")[0].value());
  ExpectTensorValues({99}, properties.GetOutputProperties("b1")[0].value());
  std::vector<int64_t> c_values;
  for (int i = 0; i < 4 * 4 * 4; i++) {
    c_values.push_back(1);
  }
  ExpectTensorValues({c_values},
                     properties.GetOutputProperties("c")[0].value());
  ExpectTensorValues({c_values},
                     properties.GetInputProperties("c1")[0].value());
  ExpectTensorValues({c_values},
                     properties.GetOutputProperties("c1")[0].value());
}

TEST_F(GraphPropertiesTest, IdentityPassingShape) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 5, {2});
  Output b = ops::Identity(s.WithOpName("b"), a);
  Output c = ops::Const(s.WithOpName("const"), 0.1f, {});
  // Fill needs not only e's shape but also the value of e to figure out output
  // shape; hence, Identity op (b) should pass a's value as
  // output_tensors_as_shape.
  Output d = ops::Fill(s.WithOpName("fill"), b, c);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto out_props = properties.GetOutputProperties("fill");
  const OpInfo::TensorProperties out_prop0 = out_props[0];
  EXPECT_EQ("float: [5,5]", PropToString(out_prop0));
}

TEST_F(GraphPropertiesTest, SkippingValueInferenceForLargeTensors) {
  // When using aggressive_shape_inference, we run EvaluateNode() for
  // allowlisted ops and small input / output tensors. For instance, Fill op is
  // evaluated and produces output tensor value if output tensor size is small
  // (currently, fewer than 17 elements); otherwise we don't run EvaluateNode().
  // This is to avoid wasting time and memory for producing huge tensors (e.g.,
  // initializing a large table using Fill.
  // Note that we do not propagate float const tensors with fractional values
  // (even if they're small); so this test should use integer values.
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output a = ops::Const(s.WithOpName("a"), 4, {2});  // 4x4
    Output b = ops::Const(s.WithOpName("const"), 7, {});
    // Shape described by a is small; expect output values of Fill op.
    Output c = ops::Fill(s.WithOpName("fill"), a, b);

    GrapplerItem item;
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
    GraphProperties properties(item);
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/false,
        /*aggressive_shape_inference=*/true,
        /*include_tensor_values=*/true));
    const auto out_props = properties.GetOutputProperties("fill");
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("int32: [4,4]", PropToString(out_prop0));
    EXPECT_TRUE(out_prop0.has_value());
  }
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output a = ops::Const(s.WithOpName("a"), 1000, {4});  // 1000x1000x1000x1000
    Output b = ops::Const(s.WithOpName("const"), 7, {});
    // Shape described by a is huge; in that case we skip value inference.
    // Otherwise, it'd be too much overhead.
    Output c = ops::Fill(s.WithOpName("fill"), a, b);

    GrapplerItem item;
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
    GraphProperties properties(item);
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/false,
        /*aggressive_shape_inference=*/true,
        /*include_tensor_values=*/true));
    const auto out_props = properties.GetOutputProperties("fill");
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("int32: [1000,1000,1000,1000]", PropToString(out_prop0));
    EXPECT_FALSE(out_prop0.has_value());
  }
}

TEST_F(GraphPropertiesTest, PackWithConstInput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 1, {});
  Output b = ops::Const(s.WithOpName("b"), 2, {});
  Output c = ops::Const(s.WithOpName("c"), 3, {});
  Output d = ops::Const(s.WithOpName("d"), 4, {});
  // Note ops::Stack instantiates Pack op.
  Output e = ops::Stack(s.WithOpName("pack"), {a, b, c, d});
  // e is rank 1 tensor: shape = {4}, and its value is {1, 2, 3, 4}
  Output f = ops::Const(s.WithOpName("const"), 0.1f, {});
  // Fill needs not only e's shape but also its value to figure out output
  // shape.
  Output g = ops::Fill(s.WithOpName("fill"), e, f);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto out_props = properties.GetOutputProperties("fill");
  const OpInfo::TensorProperties out_prop0 = out_props[0];
  EXPECT_EQ("float: [1,2,3,4]", PropToString(out_prop0));
}

TEST_F(GraphPropertiesTest, RankOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(s.WithOpName("Const"), 1, {4, 4, 4});
  Output r = ops::Rank(s.WithOpName("Rank"), c);
  Output i = ops::Identity(s.WithOpName("Identity"), r);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto rank_props = properties.GetOutputProperties("Rank");
  const OpInfo::TensorProperties rank_prop0 = rank_props[0];
  EXPECT_EQ("int32: []", PropToString(rank_prop0));
  EXPECT_TRUE(rank_prop0.has_value());
  ExpectTensorValues({3}, rank_prop0.value());
  const auto identity_props = properties.GetOutputProperties("Identity");
  const OpInfo::TensorProperties identity_props0 = identity_props[0];
  EXPECT_EQ("int32: []", PropToString(identity_props0));
  EXPECT_TRUE(identity_props0.has_value());
  ExpectTensorValues({3}, identity_props0.value());
}

TEST_F(GraphPropertiesTest, SizeOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(s.WithOpName("Const"), 1, {1, 2, 3, 4});
  Output r = ops::Size(s.WithOpName("Size"), c);
  Output i = ops::Identity(s.WithOpName("Identity"), r);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto size_props = properties.GetOutputProperties("Size");
  const OpInfo::TensorProperties size_props0 = size_props[0];
  EXPECT_EQ("int32: []", PropToString(size_props0));
  EXPECT_TRUE(size_props0.has_value());
  ExpectTensorValues({24}, size_props0.value());
  const auto identity_props = properties.GetOutputProperties("Identity");
  const OpInfo::TensorProperties identity_props0 = identity_props[0];
  EXPECT_EQ("int32: []", PropToString(identity_props0));
  EXPECT_TRUE(identity_props0.has_value());
  ExpectTensorValues({24}, identity_props0.value());
}

TEST_F(GraphPropertiesTest, PackWithConstMinus1AndReshapes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output shape0 = ops::Const(s.WithOpName("shape0"), 4, {});
  Output shape1 = ops::Const(s.WithOpName("shape1"), -1, {});
  Output pack = ops::Stack(s.WithOpName("pack"), {shape0, shape1});
  // pack is [2], with values {4, -1}.

  Output x0_ = ops::Placeholder(s.WithOpName("x0_"), DataType::DT_FLOAT);
  Output x1_ = ops::Placeholder(s.WithOpName("x1_"), DataType::DT_FLOAT);

  Output x0 = ops::Reshape(s.WithOpName("x0"), x0_, pack);
  Output x1 = ops::Reshape(s.WithOpName("x1"), x1_, pack);
  // Two unknown rank tensors (x0_ and x1_) are reshaped with pack {4, -1},
  // their output shapes would be [4, -1]. However, though we use the same
  // shape input to the Reshape ops, their output shapes can be different;
  // i.e., unknown dim values (-1) of x0 and x1 shapes are not necessarily
  // the same.

  // if input to the Select ops. Note that s0 has a fully defined shape, while
  // s1 has unknown shape.
  Output s0 = ops::Const(s.WithOpName("s0"), true, {4, 16});
  Output s1 = ops::Placeholder(s.WithOpName("s1"), DataType::DT_BOOL);

  Output y0 = ops::Placeholder(s.WithOpName("y0"), DataType::DT_FLOAT);
  Output y1 = ops::Placeholder(s.WithOpName("y1"), DataType::DT_FLOAT);

  // We instantiate SelectV2, but will replace it with Select. The shape
  // inference function for Select links all inputs and outputs as they should
  // have the same shapes.
  Output z0 = ops::SelectV2(s.WithOpName("z0"), s0, x0, y0);
  Output z1 = ops::SelectV2(s.WithOpName("z1"), s1, x1, y1);

  // For z0, as we know the shape of s0, symbolic shape manager in shape
  // inference will make the shapes of x0, y0, and z0 equal to the shape of s0,
  // which is [4, 16].
  // For z1, s0 and y1 are all unknown shapes, so we can infer they're [4, -1]
  // at best.
  // Note that x0 and x1 share the same shape input to the Reshape op, but
  // -1 in the shape input should not be treated as the same symoblic unknown
  // dim; it is merely a constant value -1 for identitying unknown dim for
  // Reshape operation.

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Replace SelectV2 op with Select op.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    auto* node = item.graph.mutable_node(i);
    if (node->op() == "SelectV2") {
      node->set_op("Select");
    }
  }

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  for (const auto& node_name : {"x0", "y0", "z0"}) {
    const auto out_props = properties.GetOutputProperties(node_name);
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("float: [4,16]", PropToString(out_prop0));
  }
  {
    const auto out_props = properties.GetOutputProperties("s0");
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("bool: [4,16]", PropToString(out_prop0));
  }

  for (const auto& node_name : {"x1", "y1", "z1"}) {
    const auto out_props = properties.GetOutputProperties(node_name);
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("float: [4,-1]", PropToString(out_prop0));
  }
  // if input of Select can be either vector or the same shape to the
  // input/output; in this case, even if we know input and output are
  // [4, ?], we can't say it's [4, ?] or a vector; hence, it should be
  // unknown.
  {
    const auto out_props = properties.GetOutputProperties("s1");
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("bool: ?", PropToString(out_prop0));
  }
}

TEST_F(GraphPropertiesTest, PackWithIdentityInput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  // Same to PackWithConstInput test case, but a, b, c, and d are Identity ops
  // from Const.
  // If output_tensors_as_shape is not set for those Shape ops or Pack op
  // doesn't take input_tensors_as_shape, Fill op's input doesn't have value;
  // hence, its output shape becomes unknown.
  Output a0 = ops::Const(s.WithOpName("a0"), 1, {});
  Output b0 = ops::Const(s.WithOpName("b0"), 2, {});
  Output c0 = ops::Const(s.WithOpName("c0"), 3, {});
  Output d0 = ops::Const(s.WithOpName("d0"), 4, {});
  Output a = ops::Identity(s.WithOpName("a"), a0);
  Output b = ops::Identity(s.WithOpName("b"), b0);
  Output c = ops::Identity(s.WithOpName("c"), c0);
  Output d = ops::Identity(s.WithOpName("d"), d0);
  // Note ops::Stack instantiates Pack op.
  Output e = ops::Stack(s.WithOpName("pack"), {a, b, c, d});
  // e is rank 1 tensor: shape = {4}, and its value is {1, 2, 3, 4}
  Output f = ops::Const(s.WithOpName("const"), 0.1f, {});
  // Fill needs not only e's shape but also its value to figure out output
  // shape.
  Output g = ops::Fill(s.WithOpName("fill"), e, f);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto out_props = properties.GetOutputProperties("fill");
  const OpInfo::TensorProperties out_prop0 = out_props[0];
  EXPECT_EQ("float: [1,2,3,4]", PropToString(out_prop0));
}

TEST_F(GraphPropertiesTest, FunctionWithDtResourceInput) {
  // Function ops may have DT_RESOURCE input; if not properly set shapes and
  // dtypes through the DT_RESOURCE _Arg, we cannot infer output shapes of such
  // function ops.
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "function_with_dt_resource_input.pbtxt");
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));

  // This graph evaluates FunctionWithDtResourceInput with two inputs:
  // x [DT_FLOAT Const],
  // _Arg [DT_RESOURCE _Arg]
  // and has two outputs:
  // z1 = x + _Arg
  // z2 = x
  {
    GraphProperties properties(item);
    TF_ASSERT_OK(properties.InferStatically(false));
    const auto out_props =
        properties.GetOutputProperties("FunctionWithDtResourceInput");
    EXPECT_EQ(out_props.size(), 2);
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("float: [1,3]", PropToString(out_prop0));
    const OpInfo::TensorProperties out_prop1 = out_props[1];
    EXPECT_EQ("float: [1,3]", PropToString(out_prop1));
  }

  {
    // Delete _handle_dtypes and _handle_shapes attr for the input _Arg node.
    for (int i = 0; i < item.graph.node_size(); i++) {
      auto* node = item.graph.mutable_node(i);
      if (node->name() == "y") {  // _Arg node with DT_RESOURCE
        node->mutable_attr()->erase("_handle_dtypes");
        node->mutable_attr()->erase("_handle_shapes");
        break;
      }
    }
    // We cannot infer the function output shape correctly without those attr,
    // but still it shouldn't fail; also, there can be some shapes we can
    // infer in such a case. In this test graph,
    // z2 of the function node just returns x input; hence, even if _Arg's shape
    // cannot be inferred, we can infer z2 output shape.
    GraphProperties properties(item);
    TF_ASSERT_OK(properties.InferStatically(false));
    const auto out_props =
        properties.GetOutputProperties("FunctionWithDtResourceInput");
    EXPECT_EQ(out_props.size(), 2);
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    // Without shape and dtype attr, we don't know _Arg's shape; hence, unknown
    // for x + _Arg.
    EXPECT_EQ("float: ?", PropToString(out_prop0));
    // The 2nd output is just x, so even if _Arg's shape is unknown, we can
    // infer this output shape.
    const OpInfo::TensorProperties out_prop1 = out_props[1];
    EXPECT_EQ("float: [1,3]", PropToString(out_prop1));
  }
}

TEST_F(GraphPropertiesTest, FunctionWithConstInput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  TF_ASSERT_OK(s.graph()->AddFunctionLibrary(function_lib_));
  Output shape = ops::Const(s.WithOpName("shape"), {1, 2, 3, 4});
  Output value = ops::Const(s.WithOpName("value"), 0.1f, {});
  auto builder = tensorflow::NodeBuilder("MyFillFunc", "MyFillFunc",
                                         s.graph()->op_registry());
  tensorflow::Node* func_op;
  auto _shape = tensorflow::ops::AsNodeOut(s, shape);
  auto _value = tensorflow::ops::AsNodeOut(s, value);
  TF_ASSERT_OK(
      builder.Input(_shape).Input(_value).Finalize(s.graph(), &func_op));
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto out_props = properties.GetOutputProperties("MyFillFunc");
  const OpInfo::TensorProperties out_prop0 = out_props[0];
  EXPECT_EQ("float: [1,2,3,4]", PropToString(out_prop0));
}

TEST_F(GraphPropertiesTest, FunctionWithIdentityOfConstInput) {
  // Same to FunctionWithConstInput, but function inputs are Identity of Const,
  // so tensor shapes, not tensor value, should be used as Const input to
  // function.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  TF_ASSERT_OK(s.graph()->AddFunctionLibrary(function_lib_));
  Output shape_ = ops::Const(s.WithOpName("shape_"), {1, 2, 3, 4});
  Output shape = ops::Identity(s.WithOpName("shape"), shape_);
  Output value = ops::Const(s.WithOpName("value"), 0.1f, {});
  auto builder = tensorflow::NodeBuilder("MyFillFunc", "MyFillFunc",
                                         s.graph()->op_registry());
  tensorflow::Node* func_op;
  auto _shape = tensorflow::ops::AsNodeOut(s, shape);
  auto _value = tensorflow::ops::AsNodeOut(s, value);
  TF_ASSERT_OK(
      builder.Input(_shape).Input(_value).Finalize(s.graph(), &func_op));
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto out_props = properties.GetOutputProperties("MyFillFunc");
  const OpInfo::TensorProperties out_prop0 = out_props[0];
  EXPECT_EQ("float: [1,2,3,4]", PropToString(out_prop0));
}

TEST_F(GraphPropertiesTest, FunctionReturnTensorValue) {
  FunctionDefLibrary library;
  *library.add_function() = FunctionDefHelper::Create(
      "MyFunc",                                                   // Name
      {"x: int32"},                                               // Inputs
      {"out: int32"},                                             // Outputs
      {},                                                         // Attrs
      {{{"a"}, "Identity", {"x"}, {{"T", DataType::DT_INT32}}}},  // Nodes
      {{"out", "a:output:0"}});                                   // Returns
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  TF_ASSERT_OK(s.graph()->AddFunctionLibrary(library));

  // MyFunc takes Const (shape) and passes it with Identity. Expect function
  // output has the same shape as well as value (output_tensors_as_shape) as
  // input Const tensor.
  Output shape = ops::Const(s.WithOpName("shape"), {5, 7}, {2});
  auto _shape = tensorflow::ops::AsNodeOut(s, shape);
  auto builder =
      tensorflow::NodeBuilder("MyFunc", "MyFunc", s.graph()->op_registry());
  tensorflow::Node* func_op;
  TF_ASSERT_OK(builder.Input(_shape).Finalize(s.graph(), &func_op));

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(true));
  const auto out_props = properties.GetOutputProperties("MyFunc");
  const OpInfo::TensorProperties out_prop0 = out_props[0];
  EXPECT_EQ("int32: [2]", PropToString(out_prop0));
  EXPECT_TRUE(out_prop0.has_value());
  ExpectTensorValues({5, 7}, out_prop0.value());
  ExpectTensorValues({5, 7},
                     properties.GetInputProperties("MyFunc")[0].value());
}

TEST_F(GraphPropertiesTest, ArithmeticFunctionReturnTensorValue) {
  FunctionDefLibrary library;
  // Function that adds two input values.
  *library.add_function() = FunctionDefHelper::Create(
      "MyFunc",                                                   // Name
      {"x: int32", "y: int32"},                                   // Inputs
      {"out: int32"},                                             // Outputs
      {},                                                         // Attrs
      {{{"a"}, "Add", {"x", "y"}, {{"T", DataType::DT_INT32}}}},  // Nodes
      {{"out", "a:z:0"}});                                        // Returns
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  TF_ASSERT_OK(s.graph()->AddFunctionLibrary(library));

  Output shape = ops::Const(s.WithOpName("shape"), {5, 7}, {2});
  auto _shape = tensorflow::ops::AsNodeOut(s, shape);
  auto builder =
      tensorflow::NodeBuilder("MyFunc", "MyFunc", s.graph()->op_registry());
  tensorflow::Node* func_op;
  TF_ASSERT_OK(
      builder.Input(_shape).Input(_shape).Finalize(s.graph(), &func_op));

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  {
    GraphProperties properties(item);
    // Without aggressive_shape_inference, the internal function does not
    // evaluate output value.
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/true,
        /*aggressive_shape_inference=*/false,
        /*include_tensor_values=*/true));
    const auto out_props = properties.GetOutputProperties("MyFunc");
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("int32: [2]", PropToString(out_prop0));
    EXPECT_FALSE(out_prop0.has_value());
  }

  {
    GraphProperties properties(item);
    // With aggressive_shape_inference, output value is evaluated.
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/true,
        /*aggressive_shape_inference=*/true,
        /*include_tensor_values=*/true));
    const auto out_props = properties.GetOutputProperties("MyFunc");
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("int32: [2]", PropToString(out_prop0));
    EXPECT_TRUE(out_prop0.has_value());

    ExpectTensorValues({10, 14}, out_prop0.value());
    ExpectTensorValues({5, 7},
                       properties.GetInputProperties("MyFunc")[0].value());
    ExpectTensorValues({5, 7},
                       properties.GetInputProperties("MyFunc")[1].value());
  }
}

// Same as the above, but float values; also, one of the function input is
// Identity of Const.
TEST_F(GraphPropertiesTest, ArithmeticFunctionReturnTensorValueFloat) {
  FunctionDefLibrary library;
  // Function that adds two input values.
  *library.add_function() = FunctionDefHelper::Create(
      "MyFunc",                                                   // Name
      {"x: float", "y: float"},                                   // Inputs
      {"out: float"},                                             // Outputs
      {},                                                         // Attrs
      {{{"a"}, "Add", {"x", "y"}, {{"T", DataType::DT_FLOAT}}}},  // Nodes
      {{"out", "a:z:0"}});                                        // Returns
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  TF_ASSERT_OK(s.graph()->AddFunctionLibrary(library));

  Output x1 = ops::Const(s.WithOpName("x1"), {5.0f, 7.0f}, {2});
  Output x2 = ops::Identity(s.WithOpName("x1"), x1);
  auto _x1 = tensorflow::ops::AsNodeOut(s, x1);
  auto _x2 = tensorflow::ops::AsNodeOut(s, x2);
  auto builder =
      tensorflow::NodeBuilder("MyFunc", "MyFunc", s.graph()->op_registry());
  tensorflow::Node* func_op;
  TF_ASSERT_OK(builder.Input(_x1).Input(_x2).Finalize(s.graph(), &func_op));

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  {
    GraphProperties properties(item);
    // Without aggressive_shape_inference, the internal function does not
    // evaluate output value.
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/true,
        /*aggressive_shape_inference=*/false,
        /*include_tensor_values=*/true));
    const auto out_props = properties.GetOutputProperties("MyFunc");
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("float: [2]", PropToString(out_prop0));
    EXPECT_FALSE(out_prop0.has_value());
  }

  {
    GraphProperties properties(item);
    // With aggressive_shape_inference, output value is evaluated.
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/true,
        /*aggressive_shape_inference=*/true,
        /*include_tensor_values=*/true));
    const auto out_props = properties.GetOutputProperties("MyFunc");
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("float: [2]", PropToString(out_prop0));
    EXPECT_TRUE(out_prop0.has_value());

    ExpectFloatTensorValues({10.0, 14.0}, out_prop0.value());
    ExpectFloatTensorValues({5.0, 7.0},
                            properties.GetInputProperties("MyFunc")[0].value());
    ExpectFloatTensorValues({5.0, 7.0},
                            properties.GetInputProperties("MyFunc")[1].value());
  }
}

TEST_F(GraphPropertiesTest, FunctionWithScalarInput) {
  // Create graph with a function that takes a scalar value so that we use
  // Placeholder with scalar as for input to the function shape inference.
  // Placeholder -> Identity -> MyFunc, where MyFunc simply takes Identity of
  // the input; all tensors are scalars.
  FunctionDefLibrary library;
  *library.add_function() = FunctionDefHelper::Create(
      "MyFunc",                                                   // Name
      {"x: float"},                                               // Inputs
      {"out: float"},                                             // Outputs
      {},                                                         // Attrs
      {{{"a"}, "Identity", {"x"}, {{"T", DataType::DT_FLOAT}}}},  // Nodes
      {{"out", "a:output:0"}});                                   // Returns
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  TF_ASSERT_OK(s.graph()->AddFunctionLibrary(library));
  Output placeholder =
      ops::Placeholder(s.WithOpName("Placeholder"), DataType::DT_FLOAT,
                       ops::Placeholder::Shape(TensorShape({})));
  Output identity = ops::Identity(s.WithOpName("Identity"), placeholder);
  auto _identity = tensorflow::ops::AsNodeOut(s, identity);
  auto builder =
      tensorflow::NodeBuilder("MyFunc", "MyFunc", s.graph()->op_registry());
  tensorflow::Node* func_op;
  TF_ASSERT_OK(builder.Input(_identity).Finalize(s.graph(), &func_op));
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Tensorflow version < 21 infers output shape of Placeholder with empty shape
  // as unknown, instead of scalar.
  EXPECT_GT(item.graph.versions().producer(), 21);

  // MyFunc output shouldn't be unknown rank.
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(true));
  const auto out_props = properties.GetOutputProperties("MyFunc");
  const OpInfo::TensorProperties out_prop0 = out_props[0];
  EXPECT_EQ(DT_FLOAT, out_prop0.dtype());
  EXPECT_FALSE(out_prop0.shape().unknown_rank());
}

TEST_F(GraphPropertiesTest, SimpleFunctionStaticShapeInference) {
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
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "simple_function.pbtxt");
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto out_props = properties.GetOutputProperties("MyAdd_55e046a8");
  const OpInfo::TensorProperties& out_prop = out_props[0];
  EXPECT_EQ(DT_FLOAT, out_prop.dtype());
  EXPECT_FALSE(out_prop.shape().unknown_rank());
  EXPECT_EQ(2, out_prop.shape().dim_size());
  EXPECT_EQ(1, out_prop.shape().dim(0).size());
  EXPECT_EQ(2, out_prop.shape().dim(1).size());

  const auto in_props = properties.GetInputProperties("MyAdd_55e046a8");
  EXPECT_EQ(2, in_props.size());

  const OpInfo::TensorProperties& in_prop = in_props[0];
  EXPECT_EQ("float: [1,2]", PropToString(in_prop));

  const OpInfo::TensorProperties& in_prop1 = in_props[1];
  EXPECT_EQ("float: [1,2]", PropToString(in_prop1));
}

TEST_F(GraphPropertiesTest, LargeFunctionStaticShapeInference) {
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "large_function_graph.pbtxt");
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  const auto out_props = properties.GetOutputProperties("y0");
  EXPECT_EQ(2, out_props.size());

  const OpInfo::TensorProperties& out_prop0 = out_props[0];
  EXPECT_EQ("float: [128,112,112,64]", PropToString(out_prop0));

  const OpInfo::TensorProperties& out_prop1 = out_props[1];
  EXPECT_EQ("float: [128,112,112,24]", PropToString(out_prop1));

  const auto in_props = properties.GetInputProperties("y0");
  EXPECT_EQ(4, in_props.size());

  const OpInfo::TensorProperties& in_prop0 = in_props[0];
  EXPECT_EQ("float: [64]", PropToString(in_prop0));

  const OpInfo::TensorProperties& in_prop1 = in_props[1];
  EXPECT_EQ("float: [1,1,24,64]", PropToString(in_prop1));

  const OpInfo::TensorProperties& in_prop2 = in_props[2];
  EXPECT_EQ("float: [128,224,224,3]", PropToString(in_prop2));

  const OpInfo::TensorProperties& in_prop3 = in_props[3];
  EXPECT_EQ("float: [7,7,3,8]", PropToString(in_prop3));
}

TEST_F(GraphPropertiesTest, LargeFunctionWithMultipleOutputs) {
  // Test graph produced in python using:
  /*
    @function.Defun(noinline=True)
    def MyFunc():
      @function.Defun(*[tf.float32] * 2)
      def Cond(n, unused_x):
        return n > 0

      @function.Defun(*[tf.float32] * 2)
      def Body(n, x):
        return n - 1, x + n

      i = tf.constant(10)
      return functional_ops.While([i, 0.], Cond, Body)

    with tf.Graph().as_default():
      z = MyFunc()
  */
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "function_functional_while.pbtxt");
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  const auto out_props = properties.GetOutputProperties("MyFunc_AenMyWWx1Us");
  EXPECT_EQ(2, out_props.size());

  const OpInfo::TensorProperties& out_prop0 = out_props[0];
  EXPECT_EQ(DT_INT32, out_prop0.dtype());
  EXPECT_FALSE(out_prop0.shape().unknown_rank());

  const OpInfo::TensorProperties& out_prop1 = out_props[1];
  EXPECT_EQ(DT_FLOAT, out_prop1.dtype());
  EXPECT_FALSE(out_prop1.shape().unknown_rank());
}

TEST_F(GraphPropertiesTest, FunctionWithErrorStaticShapeInference) {
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "function_error.pbtxt");
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));

  const auto out_props = properties.GetOutputProperties("MyAdd_yabA4wXEdM4");
  EXPECT_EQ(1, out_props.size());

  const OpInfo::TensorProperties& out_prop = out_props[0];
  EXPECT_EQ(DT_FLOAT, out_prop.dtype());
  EXPECT_TRUE(out_prop.shape().unknown_rank());

  const auto in_props = properties.GetInputProperties("MyAdd_yabA4wXEdM4");
  EXPECT_EQ(2, in_props.size());

  const OpInfo::TensorProperties& in_prop = in_props[0];
  EXPECT_EQ("float: [1,2]", PropToString(in_prop));

  const OpInfo::TensorProperties& in_prop1 = in_props[1];
  EXPECT_EQ("float: [1,2]", PropToString(in_prop1));
}

TEST_F(GraphPropertiesTest, FunctionSwitchStaticShapeInference) {
  // Test graph produced in python using:
  /*
    @function.Defun(*[tf.float32] * 2, noinline=True)
    def MyAdd(x, y):
      return tf.add(x, y)

    with tf.Graph().as_default():
      x = lambda: tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      y = lambda: tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      z = tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      z2 = MyAdd(tf.case([(tf.less(0, 1), x)], default=y), z)
  */
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "function_switch.pbtxt");
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto out_props = properties.GetOutputProperties("MyAdd_MPaeanipb7o");
  const OpInfo::TensorProperties& out_prop = out_props[0];
  EXPECT_EQ(DT_FLOAT, out_prop.dtype());
  EXPECT_EQ("float: [1,2]", PropToString(out_prop));

  const auto in_props = properties.GetInputProperties("MyAdd_MPaeanipb7o");
  EXPECT_EQ(2, in_props.size());

  const OpInfo::TensorProperties& in_prop = in_props[0];
  EXPECT_EQ("float: [1,2]", PropToString(in_prop));

  const OpInfo::TensorProperties& in_prop1 = in_props[1];
  EXPECT_EQ("float: [1,2]", PropToString(in_prop1));
}

TEST_F(GraphPropertiesTest, FunctionSwitch2StaticShapeInference) {
  // Test graph produced in python using:
  /*
    @function.Defun(*[tf.float32] * 2, noinline=True)
    def MyAdd(x, y):
      return tf.add(x, y)

    with tf.Graph().as_default():
      x = lambda: tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      y = lambda: tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      z = tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      z2 = MyAdd(tf.case([(tf.less(1, 0), x)], default=y), z)
  */
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "function_switch_2.pbtxt");
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto out_props = properties.GetOutputProperties("MyAdd_MPaeanipb7o");
  const OpInfo::TensorProperties& out_prop = out_props[0];
  EXPECT_EQ("float: [1,2]", PropToString(out_prop));

  const auto in_props = properties.GetInputProperties("MyAdd_MPaeanipb7o");
  EXPECT_EQ(2, in_props.size());

  const OpInfo::TensorProperties& in_prop = in_props[0];
  EXPECT_EQ("float: [1,2]", PropToString(in_prop));

  const OpInfo::TensorProperties& in_prop1 = in_props[1];
  EXPECT_EQ("float: [1,2]", PropToString(in_prop1));
}

TEST_F(GraphPropertiesTest, FunctionSwitchShapesStaticShapeInference) {
  // Test graph produced in python using:
  /*
    @function.Defun(*[tf.float32] * 2, noinline=True)
    def MyAdd(x, y):
      a = tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      b = tf.constant(2.0, shape=[1, 3], dtype=tf.float32)
      c = tf.add(x, a)
      d = tf.add(y, b)
      return c

    with tf.Graph().as_default():
      x = lambda: tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      y = lambda: tf.constant(2.0, shape=[1, 2], dtype=tf.float32)
      z = tf.constant(2.0, shape=[1, 3], dtype=tf.float32)
      z2 = MyAdd(tf.case([(tf.less(1, 0), x)], default=y), z)
  */
  GrapplerItem item;
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPath,
                                 "function_switch_shapes.pbtxt");
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto out_props = properties.GetOutputProperties("MyAdd_lEKAAnIwI5I");
  const OpInfo::TensorProperties& out_prop = out_props[0];
  EXPECT_EQ("float: [1,2]", PropToString(out_prop));

  const auto in_props = properties.GetInputProperties("MyAdd_lEKAAnIwI5I");
  EXPECT_EQ(2, in_props.size());

  const OpInfo::TensorProperties& in_prop = in_props[0];
  EXPECT_EQ("float: [1,2]", PropToString(in_prop));

  const OpInfo::TensorProperties& in_prop1 = in_props[1];
  EXPECT_EQ("float: [1,3]", PropToString(in_prop1));
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
  Output zero_idx = ops::Const(s.WithOpName("zero_idx"), {0}, {1});
  Output j = ops::Sum(s.WithOpName("j"), a, zero_idx);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
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

  const auto shape_j = properties.GetOutputProperties("j").at(0).shape();
  ASSERT_EQ(1, shape_j.dim_size());
  EXPECT_EQ(shape_j.dim(0).size(), shape_a.dim(1).size());
}

TEST_F(GraphPropertiesTest, DoNotValidateColocationConstraints) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 1.0f, {1});
  Output b = ops::Const(s.WithOpName("b"), 2.0f, {1});
  Output c = ops::Const(s.WithOpName("c").ColocateWith(a), 3.0f, {1});
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
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
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
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
    TF_ASSERT_OK(s);
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
    TF_ASSERT_OK(s);
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
        EXPECT_EQ(in_prop.dtype(), out_prop.dtype());
        EXPECT_EQ(in_prop.shape().DebugString(),
                  out_prop.shape().DebugString());
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
  TF_ASSERT_OK(ReadGraphDefFromFile(filename, &item.graph));
  TF_ASSERT_OK(AddDefaultAttrsToGraphDef(
      &item.graph,
      FunctionLibraryDefinition(OpRegistry::Global(), item.graph.library()), 0,
      true));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
}

TEST_F(GraphPropertiesTest, StridedSlicesOfShapes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a =
      ops::Placeholder(s.WithOpName("a"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1, -1})));
  auto shp = ops::Shape(s.WithOpName("shape"), {a});

  Output index1 = ops::Const(s.WithOpName("index1"), 0, {1});
  Output index2 = ops::Const(s.WithOpName("index2"), 1, {1});
  Output index3 = ops::Const(s.WithOpName("index3"), 2, {1});

  Output b = ops::StridedSlice(s.WithOpName("b"), shp, index1, index2, index2);
  Output c = ops::StridedSlice(s.WithOpName("c"), shp, index2, index3, index2);

  Output zero = ops::Const(s.WithOpName("zero"), 0.0f, {});
  Output o1 = ops::Fill(s.WithOpName("o1"), b, zero);
  Output o2 = ops::Fill(s.WithOpName("o2"), c, zero);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(false));
  const auto shape_a = properties.GetOutputProperties("a").at(0).shape();
  const auto shape_o1 = properties.GetOutputProperties("o1").at(0).shape();
  const auto shape_o2 = properties.GetOutputProperties("o2").at(0).shape();
  EXPECT_EQ(2, shape_a.dim_size());
  EXPECT_EQ(1, shape_o1.dim_size());
  EXPECT_EQ(1, shape_o2.dim_size());
  EXPECT_EQ(shape_a.dim(0).size(), shape_o1.dim(0).size());
  EXPECT_EQ(shape_a.dim(1).size(), shape_o2.dim(0).size());
}

TEST_F(GraphPropertiesTest, StridedSliceOfShapeWithShrinkAxisMask) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output placeholder =
      ops::Placeholder(scope.WithOpName("input_placeholder"), DT_FLOAT,
                       ops::Placeholder::Shape(TensorShape({5, 480, 40, 1})));
  auto input_shape = ops::Shape(scope.WithOpName("input_shape"), placeholder);

  Output begin = ops::Const(scope.WithOpName("begin"), {0}, {1});
  Output end = ops::Const(scope.WithOpName("end"), {3}, {1});
  Output stride = ops::Const(scope.WithOpName("stride"), {1}, {1});

  Output slice =
      ops::StridedSlice(scope.WithOpName("slice"), input_shape, begin, end,
                        stride, ops::StridedSlice::ShrinkAxisMask(1));

  GrapplerItem item;
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  // Without aggressive shape inference, it cannot infer output value of
  // StridedSlice with ShrinkAxisMask.
  {
    GraphProperties properties(item);
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/false,
        /*aggressive_shape_inference=*/false,
        /*include_tensor_values=*/true));
    EXPECT_FALSE(properties.GetOutputProperties("slice").at(0).has_value());
  }

  // InferStatically with aggressive shape inference can infer output value of
  // StridedSlice with ShrinkAxisMask.
  {
    GraphProperties properties(item);
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/false,
        /*aggressive_shape_inference=*/true,
        /*include_tensor_values=*/true));
    EXPECT_TRUE(properties.GetOutputProperties("slice").at(0).has_value());
    const auto slice_value =
        properties.GetOutputProperties("slice").at(0).value();
    ExpectTensorValues({5}, slice_value);
  }
}

TEST_F(GraphPropertiesTest, ValuePropagationThroughArithmeticOps) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), {5, 7}, {2});
  Output b = ops::Const(s.WithOpName("b"), {8, 8}, {2});
  Output c = ops::Const(s.WithOpName("c"), {2, 2}, {2});

  Output a1 = ops::OnesLike(s.WithOpName("a1"), a);
  Output a_plus_one = ops::Add(s.WithOpName("a_plus_one"), a, a1);
  Output a_plus_a = ops::Add(s.WithOpName("a_plus_a"), a, a);
  Output b_plus_2a = ops::Add(s.WithOpName("b_plus_2a"), b, a_plus_a);
  Output c_plus_b_plus_2a =
      ops::Add(s.WithOpName("c_plus_b_plus_2a"), c, b_plus_2a);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(
      /*assume_valid_feeds=*/false,
      /*aggressive_shape_inference=*/true,
      /*include_tensor_values=*/true));

  // Check output shapes and values.
  const auto& a_plus_one_prop = properties.GetOutputProperties("a_plus_one")[0];
  EXPECT_EQ("int32: [2]", PropToString(a_plus_one_prop));
  EXPECT_TRUE(a_plus_one_prop.has_value());
  ExpectTensorValues({6, 8}, a_plus_one_prop.value());

  const auto& a_plus_a_prop = properties.GetOutputProperties("a_plus_a")[0];
  EXPECT_EQ("int32: [2]", PropToString(a_plus_a_prop));
  EXPECT_TRUE(a_plus_a_prop.has_value());
  ExpectTensorValues({10, 14}, a_plus_a_prop.value());

  const auto& b_plus_2a_prop = properties.GetOutputProperties("b_plus_2a")[0];
  EXPECT_EQ("int32: [2]", PropToString(b_plus_2a_prop));
  EXPECT_TRUE(b_plus_2a_prop.has_value());
  ExpectTensorValues({18, 22}, b_plus_2a_prop.value());

  const auto& c_plus_b_plus_2a_prop =
      properties.GetOutputProperties("c_plus_b_plus_2a")[0];
  EXPECT_EQ("int32: [2]", PropToString(c_plus_b_plus_2a_prop));
  EXPECT_TRUE(c_plus_b_plus_2a_prop.has_value());
  ExpectTensorValues({20, 24}, c_plus_b_plus_2a_prop.value());
}

TEST_F(GraphPropertiesTest, ShapeAnnotation) {
  GrapplerItem item;
  TF_ASSERT_OK(NodeDefBuilder("Input", "Placeholder")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", PartialTensorShape({-1, -1}))
                   .Finalize(item.graph.add_node()));
  // Annotate shapes.
  TF_ASSERT_OK(NodeDefBuilder("Identity", "Identity")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("_same_output_for_iterations", true)
                   .Attr("_output_shape_vector", {TensorShape({5, 7})})
                   .Input("Input", 0, DT_FLOAT)
                   .Finalize(item.graph.add_node()));
  {
    GraphProperties properties(item);
    // Without aggressive_shape_inference, ignore annotated information.
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/false,
        /*aggressive_shape_inference=*/false,
        /*include_tensor_values=*/true));
    const auto props = properties.GetOutputProperties("Identity");
    EXPECT_EQ(1, props.size());
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT, prop.dtype());
    EXPECT_EQ(2, prop.shape().dim_size());
    // Get unknown shapes without using annotated information.
    EXPECT_EQ("float: [-1,-1]", PropToString(prop));
  }
  {
    GraphProperties properties(item);
    // Use annotated information.
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/false,
        /*aggressive_shape_inference=*/true,
        /*include_tensor_values=*/true));
    const auto props = properties.GetOutputProperties("Identity");
    EXPECT_EQ(1, props.size());
    const OpInfo::TensorProperties& prop = props[0];
    EXPECT_EQ(DT_FLOAT, prop.dtype());
    EXPECT_EQ(2, prop.shape().dim_size());
    // Update output shape using annotated shapes.
    EXPECT_EQ("float: [5,7]", PropToString(prop));
  }
}

TEST_F(GraphPropertiesTest, ShapeAnnotationWithCompatibleShapes) {
  GrapplerItem item;
  TF_ASSERT_OK(NodeDefBuilder("Input", "Placeholder")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", PartialTensorShape({-1, 100}))
                   .Finalize(item.graph.add_node()));
  // Annotate shapes.
  TF_ASSERT_OK(NodeDefBuilder("Identity", "Identity")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("_same_output_for_iterations", true)
                   .Attr("_output_shape_vector", {TensorShape({10, 100})})
                   .Input("Input", 0, DT_FLOAT)
                   .Finalize(item.graph.add_node()));
  GraphProperties properties(item);
  // Use annotated information.
  TF_ASSERT_OK(properties.InferStatically(
      /*assume_valid_feeds=*/false,
      /*aggressive_shape_inference=*/true,
      /*include_tensor_values=*/true));
  const auto props = properties.GetOutputProperties("Identity");
  EXPECT_EQ(1, props.size());
  const OpInfo::TensorProperties& prop = props[0];
  EXPECT_EQ(DT_FLOAT, prop.dtype());
  EXPECT_EQ(2, prop.shape().dim_size());
  // Compatible shapes. Update output shape using annotated shapes.
  EXPECT_EQ("float: [10,100]", PropToString(prop));
}

TEST_F(GraphPropertiesTest, ShapeAnnotationWithIncompatibleShapes) {
  GrapplerItem item;
  TF_ASSERT_OK(NodeDefBuilder("Input", "Placeholder")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", PartialTensorShape({-1, 100}))
                   .Finalize(item.graph.add_node()));
  // Annotate shapes.
  TF_ASSERT_OK(NodeDefBuilder("Identity", "Identity")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("_same_output_for_iterations", true)
                   .Attr("_output_shape_vector", {TensorShape({10, 10})})
                   .Input("Input", 0, DT_FLOAT)
                   .Finalize(item.graph.add_node()));
  GraphProperties properties(item);
  // Use annotated information.
  TF_ASSERT_OK(properties.InferStatically(
      /*assume_valid_feeds=*/false,
      /*aggressive_shape_inference=*/true,
      /*include_tensor_values=*/true));
  const auto props = properties.GetOutputProperties("Identity");
  EXPECT_EQ(1, props.size());
  const OpInfo::TensorProperties& prop = props[0];
  EXPECT_EQ(DT_FLOAT, prop.dtype());
  EXPECT_EQ(2, prop.shape().dim_size());
  // Incompatible shapes. Do not use annotated shapes.
  EXPECT_EQ("float: [-1,100]", PropToString(prop));
}

TEST_F(GraphPropertiesTest, ShapeAnnotationWithoutInferenceFn) {
  GrapplerItem item;
  TF_ASSERT_OK(NodeDefBuilder("Input", "Placeholder")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", PartialTensorShape({-1, -1}))
                   .Finalize(item.graph.add_node()));
  // Annotate shapes.
  TF_ASSERT_OK(
      NodeDefBuilder("TestOpWithNoInferenceFn", "TestOpWithNoInferenceFn")
          .Attr("_same_output_for_iterations", true)
          .Attr("_output_shape_vector", {TensorShape({10, 100})})
          .Input("Input", 0, DT_FLOAT)
          .Finalize(item.graph.add_node()));
  GraphProperties properties(item);
  // Use annotated information.
  TF_ASSERT_OK(properties.InferStatically(
      /*assume_valid_feeds=*/false,
      /*aggressive_shape_inference=*/true,
      /*include_tensor_values=*/true));
  const auto props = properties.GetOutputProperties("TestOpWithNoInferenceFn");
  EXPECT_EQ(1, props.size());
  const OpInfo::TensorProperties& prop = props[0];
  EXPECT_EQ(DT_FLOAT, prop.dtype());
  EXPECT_EQ(2, prop.shape().dim_size());
  EXPECT_EQ("float: [10,100]", PropToString(prop));
}

TEST_F(GraphPropertiesTest, PartitionedCallOp) {
  Scope root = Scope::NewRootScope().ExitOnError();
  FunctionDefLibrary library;
  FunctionDef called_func = FunctionDefHelper::Create(
      "identity_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int32"},
      /*attr_def=*/{},
      {{{"Identity"}, "Identity", {"arg0"}, {{"T", DT_INT32}}}},
      /*ret_def=*/{{"ret0", "Identity:output:0"}});
  *library.add_function() = called_func;
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(library));

  Output in = ops::Const(root, {3, 1, 2, 0});
  NameAttrList b_name_attr;
  b_name_attr.set_name("identity_function");
  ops::PartitionedCall call(root.WithOpName("identity_call"), {in}, {DT_INT32},
                            b_name_attr);

  GrapplerItem item;
  TF_ASSERT_OK(root.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(
      /*assume_valid_feeds=*/true,
      /*aggressive_shape_inference=*/false,
      /*include_tensor_values=*/true));

  EXPECT_EQ("int32: [4]",
            PropToString(properties.GetOutputProperties("identity_call")[0]));
}

TEST_F(GraphPropertiesTest, NonTrivialInputPartitionedCallOp) {
  auto f = FunctionDefHelper::Create(
      // Name
      "FunctionWhichAdds",
      // Inputs
      {"arg0: int32", "arg1: int32"},
      // Outputs
      {"ret0: int32"},
      /*attr_def=*/{},
      // Nodes
      {{{"a"}, "Add", {"arg0", "arg1"}, {{"T", DT_INT32}}}},
      /*ret_def=*/{{"ret0", "a:z:0"}});

  FunctionDefLibrary function_lib;
  function_lib.add_function()->Swap(&f);
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(function_lib));

  PartialTensorShape input_shape({2, 2, -1});
  Output in1 =
      ops::Placeholder(root, DT_INT32, ops::Placeholder::Shape(input_shape));
  Output in2 =
      ops::Placeholder(root, DT_INT32, ops::Placeholder::Shape(input_shape));
  NameAttrList b_name_attr;
  b_name_attr.set_name("FunctionWhichAdds");
  ops::PartitionedCall call(root.WithOpName("add_call"), {in1, in2}, {DT_INT32},
                            b_name_attr);

  GrapplerItem item;
  TF_ASSERT_OK(root.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(
      /*assume_valid_feeds=*/true,
      /*aggressive_shape_inference=*/false,
      /*include_tensor_values=*/true));

  EXPECT_EQ("int32: [2,2,-1]",
            PropToString(properties.GetOutputProperties("add_call")[0]));
}

TEST_F(GraphPropertiesTest, ShapeAnnotatedFunctionOp) {
  // A function, which we cannot infer output shape statically.
  auto f = FunctionDefHelper::Create(
      // Name
      "FuncShapeCannotBeInferred",
      // Inputs
      {},
      // Outputs
      {"output: float"},
      // Attrs
      {},
      // Nodes
      {
          // Placeholder without shape attr; unknown rank.
          {{"p"}, "Placeholder", {}, {{"dtype", DataType::DT_FLOAT}}},
      },
      // Returns
      {{"output", "p:output:0"}});
  FunctionDefLibrary function_lib;
  function_lib.add_function()->Swap(&f);
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  TF_ASSERT_OK(s.graph()->AddFunctionLibrary(function_lib));
  tensorflow::Node* func_op;
  TensorShapeProto output_shape;
  output_shape.set_unknown_rank(false);
  output_shape.add_dim()->set_size(1);
  output_shape.add_dim()->set_size(2);
  output_shape.add_dim()->set_size(3);
  output_shape.add_dim()->set_size(4);
  // The function node, f, includes shape annotation.
  TF_ASSERT_OK(tensorflow::NodeBuilder("f", "FuncShapeCannotBeInferred",
                                       s.graph()->op_registry())
                   .Attr("_execution_count", 1)
                   .Attr("_same_output_for_iterations", true)
                   .Attr("_output_dtype_vector", {DataType::DT_FLOAT})
                   .Attr("_output_shape_vector", {output_shape})
                   .Finalize(s.graph(), &func_op));
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // InferStatically with aggressive_shape_inference would fail to infer
  // the output shape of the node f.
  {
    GraphProperties properties(item);
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/false,
        /*aggressive_shape_inference=*/false,
        /*include_tensor_values=*/false));
    const auto out_props = properties.GetOutputProperties("f");
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("float: ?", PropToString(out_prop0));
  }
  // With aggressive_shape_inference, it skips recursively callying
  // InferStatically for the function node and outputs annotated shape info.
  {
    GraphProperties properties(item);
    TF_ASSERT_OK(properties.InferStatically(
        /*assume_valid_feeds=*/false,
        /*aggressive_shape_inference=*/true,
        /*include_tensor_values=*/true));
    const auto out_props = properties.GetOutputProperties("f");
    const OpInfo::TensorProperties out_prop0 = out_props[0];
    EXPECT_EQ("float: [1,2,3,4]", PropToString(out_prop0));
  }
}

TEST_F(GraphPropertiesTest,
       SymbolicShapeInferenceWithReshapeOpsSharingShapeVector) {
  GrapplerItem item;
  // This graph creates a shape vector [-1, 10] from Concat(Const, Const)
  // used for two reshape ops. One reshape op is segment_ids input to
  // UnsortedSegmentSum op, which applies MergePrefix from its shape function.
  // segment_ids has a shape [-1, 10] (from reshape), but MergePrefix with
  // data input ([10, 10, 10, 10]) makes -1, or unknown dim, 10, with
  // SymbolicShapeRefiner.
  // This dim value (10), however, should not affect the other reshape op, even
  // though it shares the shape input; -1 in the shape input of Reshape op is
  // a special case of computed output dim, not unknown dim.
  // data and num_segments are inputs to UnsortedSegmenetSum.

  TF_ASSERT_OK(NodeDefBuilder("data", "Placeholder")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", TensorShape({10, 10, 10, 10}))
                   .Finalize(item.graph.add_node()));
  Tensor num_segments(DT_INT32, TensorShape({}));
  // Build semgent_ids input to UnsortedSegmentSum from Const ops, ConcatV2,
  // and Reshape ops. tensors_as_shape from Const ops are propagated to ConcatV2
  // output to form shape vector [-1, 10] to Reshape.
  test::FillIota<int>(&num_segments, 3);
  TF_ASSERT_OK(NodeDefBuilder("num_segments", "Const")
                   .Attr("dtype", DT_INT32)
                   .Attr("value", num_segments)
                   .Finalize(item.graph.add_node()));
  Tensor minus_one(DT_INT32, TensorShape({1}));
  test::FillIota<int>(&minus_one, -1);
  TF_ASSERT_OK(NodeDefBuilder("minus_one", "Const")
                   .Attr("dtype", DT_INT32)
                   .Attr("value", minus_one)
                   .Finalize(item.graph.add_node()));
  Tensor plus_ten(DT_INT32, TensorShape({1}));
  test::FillIota<int>(&plus_ten, 10);
  TF_ASSERT_OK(NodeDefBuilder("plus_ten", "Const")
                   .Attr("dtype", DT_INT32)
                   .Attr("value", plus_ten)
                   .Finalize(item.graph.add_node()));
  Tensor axis(DT_INT32, TensorShape({}));
  test::FillIota<int>(&axis, -1);
  TF_ASSERT_OK(NodeDefBuilder("axis", "Const")
                   .Attr("dtype", DT_INT32)
                   .Attr("value", axis)
                   .Finalize(item.graph.add_node()));
  std::vector<NodeDefBuilder::NodeOut> inputs(2);
  inputs[0] = NodeDefBuilder::NodeOut{"minus_one", 0, DT_INT32};
  inputs[1] = NodeDefBuilder::NodeOut{"plus_ten", 0, DT_INT32};
  TF_ASSERT_OK(NodeDefBuilder("concat", "ConcatV2")
                   .Input(inputs)
                   .Input("axis", 0, DT_INT32)
                   .Attr("N", 2)
                   .Attr("T", DT_INT32)
                   .Attr("Tidx", DT_INT32)
                   .Finalize(item.graph.add_node()));
  TF_ASSERT_OK(NodeDefBuilder("segment_ids_", "Placeholder")
                   .Attr("dtype", DT_FLOAT)
                   .Finalize(item.graph.add_node()));
  TF_ASSERT_OK(NodeDefBuilder("segment_ids_shape_before_reshape", "Shape")
                   .Input("segment_ids_", 0, DT_FLOAT)
                   .Attr("T", DT_FLOAT)
                   .Attr("out_type", DT_INT32)
                   .Finalize(item.graph.add_node()));
  TF_ASSERT_OK(NodeDefBuilder("segment_ids", "Reshape")
                   .Input("segment_ids_", 0, DT_FLOAT)
                   .Input("concat", 0, DT_INT32)
                   .Attr("T", DT_FLOAT)
                   .Attr("Tshape", DT_INT32)
                   .Finalize(item.graph.add_node()));
  // Shape function of UnsortedSegmentSum applies MergePrefix to data and
  // segment_ids (the latter being prefix). data shape is [10,10,10,10] and
  // segment_ids shape is [-1, 10], but MergePrefix and symbolic shape inference
  // assign 10 from data shape to the unknown dim in segment_ids.
  TF_ASSERT_OK(NodeDefBuilder("y", "UnsortedSegmentSum")
                   .Input("data", 0, DT_FLOAT)
                   .Input("segment_ids", 0, DT_INT32)
                   .Input("num_segments", 0, DT_INT32)
                   .Attr("T", DT_FLOAT)
                   .Attr("Tindices", DT_INT32)
                   .Attr("Tnumsegments", DT_INT32)
                   .Finalize(item.graph.add_node()));
  // Note that y2=Reshape(x1) using the same shape vector as segment_ids, but
  // y2 shape shouldn't be affected by symbolic shape inference w/ segment_ids.
  TF_ASSERT_OK(NodeDefBuilder("x1", "Placeholder")
                   .Attr("dtype", DT_FLOAT)
                   .Finalize(item.graph.add_node()));
  TF_ASSERT_OK(NodeDefBuilder("y1", "Reshape")
                   .Input("x1", 0, DT_FLOAT)
                   .Input("concat", 0, DT_INT32)
                   .Attr("T", DT_FLOAT)
                   .Attr("Tshape", DT_INT32)
                   .Finalize(item.graph.add_node()));

  GraphProperties properties(item);
  TF_ASSERT_OK(properties.InferStatically(true));
  const auto& y1_output_properties = properties.GetOutputProperties("y1");
  // y1=reshape(x1), but x1's shape in unknown, so y1 should be [-1, 10].
  // The first dimension should not be 10.
  EXPECT_EQ(y1_output_properties.size(), 1);
  EXPECT_EQ(y1_output_properties[0].shape().dim_size(), 2);
  EXPECT_LT(y1_output_properties[0].shape().dim(0).size(), 0);
  EXPECT_EQ(y1_output_properties[0].shape().dim(1).size(), 10);
}

TEST(HelperFunctions, IsShapeFullyDefinedIntegerVectorOrScalar) {
  // Make a dummy InferenceContext.
  NodeDef node_def;
  OpRegistrationData op_reg_data;
  OpDefBuilder b("dummy");
  CHECK(b.Finalize(&op_reg_data).ok());
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
      input_handle_shapes_and_types;
  InferenceContext ic(/*graph_def_version=*/0, node_def, op_reg_data.op_def,
                      /*input_shapes=*/{},
                      /*input_tensors=*/{},
                      /*input_tensors_as_shapes=*/{},
                      std::move(input_handle_shapes_and_types));

  // ShapeHandles for testing.
  ShapeHandle fully_defined_vector = ic.MakeShape(
      {ic.MakeDim(4), ic.MakeDim(5), ic.MakeDim(6), ic.MakeDim(7)});
  ShapeHandle vector_with_unknown = ic.MakeShape(
      {ic.MakeDim(4), ic.MakeDim(5), ic.UnknownDim(), ic.MakeDim(7)});
  // INT64_MAX is used as unknown from Const. See kUnknownFromConst const in
  // graph_properties.cc
  ShapeHandle vector_with_unknown_from_const = ic.MakeShape(
      {ic.MakeDim(4), ic.MakeDim(INT64_MAX), ic.MakeDim(6), ic.MakeDim(7)});
  ShapeHandle rank_1_vector = ic.MakeShape({ic.MakeDim(4)});

  // Rank-1 shape and fully defined tensor_as_shape with INT32 or INT64.
  EXPECT_TRUE(IsShapeFullyDefinedIntegerVectorOrScalar(
      &ic, rank_1_vector, fully_defined_vector, DT_INT32));
  EXPECT_TRUE(IsShapeFullyDefinedIntegerVectorOrScalar(
      &ic, rank_1_vector, fully_defined_vector, DT_INT64));

  // Non-integer data type.
  EXPECT_FALSE(IsShapeFullyDefinedIntegerVectorOrScalar(
      &ic, rank_1_vector, fully_defined_vector, DT_FLOAT));

  // tensor_as_shape including Unknown or UnknownFromConst.
  EXPECT_FALSE(IsShapeFullyDefinedIntegerVectorOrScalar(
      &ic, rank_1_vector, vector_with_unknown, DT_INT32));
  EXPECT_FALSE(IsShapeFullyDefinedIntegerVectorOrScalar(
      &ic, rank_1_vector, vector_with_unknown_from_const, DT_INT32));
  EXPECT_FALSE(IsShapeFullyDefinedIntegerVectorOrScalar(
      &ic, rank_1_vector, ic.UnknownShape(), DT_INT32));
  EXPECT_FALSE(IsShapeFullyDefinedIntegerVectorOrScalar(
      &ic, ic.UnknownShape(), fully_defined_vector, DT_INT32));

  // shape rank > 1.
  EXPECT_FALSE(IsShapeFullyDefinedIntegerVectorOrScalar(
      &ic, fully_defined_vector, vector_with_unknown_from_const, DT_INT32));
}
}  // namespace
}  // namespace grappler
}  // namespace tensorflow
