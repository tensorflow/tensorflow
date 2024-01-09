/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>

#include "tensorflow/core/common_runtime/type_inference.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ControlFlowOpsTest, Merge_ShapeFn) {
  ShapeInferenceTestOp op("Merge");

  int n = 3;
  std::vector<NodeDefBuilder::NodeOut> src_list;
  src_list.reserve(n);
  for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_FLOAT);
  TF_ASSERT_OK(NodeDefBuilder("test", "Merge")
                   .Input(src_list)
                   .Attr("N", n)
                   .Finalize(&op.node_def));

  // The second output should always be scalar.
  // The first output should be unknown if any of the inputs are unknown, or
  // if two inputs disagree about rank.
  INFER_OK(op, "?;?;?", "?;[]");
  INFER_OK(op, "[2,1];?;[2,1]", "?;[]");
  INFER_OK(op, "[2,1];[2,1];?", "?;[]");
  INFER_OK(op, "[2,1];[2,1];[3,1,2]", "?;[]");
  // If inputs on rank, but disagree on specific dimensions, those dimensions
  // should be unknown.
  INFER_OK(op, "[2,1];[2,1];[3,1]", "[?,d0_1];[]");
  INFER_OK(op, "[2,1];[2,2];[3,1]", "[?,?];[]");
  // Otherwise, all inputs agree and we return the first input.
  INFER_OK(op, "[2,1];[2,1];[2,1]", "in0;[]");
}

TEST(ControlFlowOpsTest, SwitchN_ShapeFn) {
  ShapeInferenceTestOp op("_SwitchN");

  int n = 5;
  TF_ASSERT_OK(NodeDefBuilder("test", "_SwitchN")
                   .Input({"d", 0, DT_FLOAT})
                   .Input({"bi", 0, DT_INT32})
                   .Attr("num_outs", n)
                   .Finalize(&op.node_def));

  // Non-scalar output_index.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[2]");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[1]");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[?]");
  // The second input should always be scalar. Outputs are 5x the first input.
  INFER_OK(op, "?;?", "in0;in0;in0;in0;in0");
  INFER_OK(op, "[2,?];?", "in0;in0;in0;in0;in0");
  INFER_OK(op, "[2,?];[]", "in0;in0;in0;in0;in0");
  INFER_OK(op, "[2,3];[]", "in0;in0;in0;in0;in0");
}

TEST(ControlFlowOpsTest, RefSelect_ShapeFn) {
  ShapeInferenceTestOp op("RefSelect");

  int n = 3;
  std::vector<NodeDefBuilder::NodeOut> src_list;
  src_list.reserve(n);
  for (int i = 0; i < n; ++i) src_list.emplace_back("a", 1, DT_FLOAT_REF);
  TF_ASSERT_OK(NodeDefBuilder("test", "RefSelect")
                   .Input("index", 0, DT_INT32)
                   .Input(src_list)
                   .Attr("N", n)
                   .Finalize(&op.node_def));

  // The first argument should be scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[2];?;?;?");

  // If any inputs aren't fully defined, we return an unknown shape.
  INFER_OK(op, "?;?;?;?", "?");
  INFER_OK(op, "[];?;?;?", "?");
  INFER_OK(op, "[];[1,2,3];?;?", "?");
  INFER_OK(op, "[];[1,2,3];[1,2,?];[1,2,3]", "?");
  // If inputs disagree on rank or dimension, we return an unknown shape.
  INFER_OK(op, "[];[1,2,3];[1,2];[1,2,3]", "?");
  INFER_OK(op, "[];[1,2,3];[1,2,4];[1,2,3]", "?");
  // Otherwise, all inputs agree and we return the first input.
  INFER_OK(op, "[];[1,2,3];[1,2,3];[1,2,3]", "in1");
}

// Runs type inference pass on graph
static Status type_inference(Graph& graph) {
  GraphOptimizationPassOptions opt_options;
  std::unique_ptr<Graph> graph_ptr(new Graph(OpRegistry::Global()));
  graph_ptr->Copy(graph);
  opt_options.graph = &graph_ptr;
  opt_options.flib_def = graph.mutable_flib_def();
  TypeInferencePass pass;
  return pass.Run(opt_options);
}

// TODO(b/222556529) when Const has a type constructor, remove the following
// REGISTER_OP definiton for ControlFlowOpsTest>ConstTypeCtor and use the Const
// op instead of ControlFlowOpsTest>ConstTypeCtor in the Shape_TypeCtor test.
REGISTER_OP("ControlFlowOpsTest>ConstTypeCtor")
    .Output("output: dtype")
    .Attr("value: tensor")
    .Attr("dtype: type")
    .SetTypeConstructor(full_type::Unary(TFT_TENSOR, "dtype"))
    .SetShapeFn(shape_inference::UnknownShape);

TEST(ControlFlowOpsTest, Merge_TypeInfrnc) {
  Graph graph(OpRegistry::Global());
  Node* input_tensor_op1;
  TensorProto tensor_proto1;
  TF_EXPECT_OK(
      NodeBuilder("input_tensor_op1", "ControlFlowOpsTest>ConstTypeCtor")
          .Attr("value", tensor_proto1)
          .Attr("dtype", DT_FLOAT)
          .Finalize(&graph, &input_tensor_op1));
  Node* input_tensor_op2;
  TensorProto tensor_proto2;
  TF_EXPECT_OK(
      NodeBuilder("input_tensor_op2", "ControlFlowOpsTest>ConstTypeCtor")
          .Attr("value", tensor_proto2)
          .Attr("dtype", DT_FLOAT)
          .Finalize(&graph, &input_tensor_op2));
  Node* shape_op;
  TF_EXPECT_OK(NodeBuilder("merge_op", "Merge")
                   .Input({input_tensor_op1, input_tensor_op2})
                   .Attr("T", DT_FLOAT)
                   .Finalize(&graph, &shape_op));
  TF_EXPECT_OK(type_inference(graph));
  FullTypeDef expected_shape_op_t;
  protobuf::TextFormat::Parser parser;
  CHECK(parser.ParseFromString(
      R"pb(type_id: TFT_PRODUCT
           args {
             type_id: TFT_TENSOR
             args { type_id: TFT_FLOAT }
           }
           args {
             type_id: TFT_TENSOR
             args { type_id: TFT_INT32 }
           })pb",
      &expected_shape_op_t));
  EXPECT_TRUE(full_type::IsEqual(shape_op->def().experimental_type(),
                                 expected_shape_op_t))
      << "fulltype is\n"
      << shape_op->def().experimental_type().DebugString() << "\nexpected\n"
      << expected_shape_op_t.DebugString();
}

}  // end namespace tensorflow
