/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/shape_inference_testutil.h"

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace shape_inference {

namespace {

static OpShapeInferenceFn* global_fn_ptr = nullptr;
REGISTER_OP("OpOneOut")
    .Input("inputs: N * T")
    .Output("o1: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* c) { return (*global_fn_ptr)(c); });
REGISTER_OP("OpTwoOut")
    .Input("inputs: N * T")
    .Output("o1: T")
    .Output("o2: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* c) { return (*global_fn_ptr)(c); });

string RunInferShapes(const string& op_name, const string& ins,
                      const string& expected_outs, OpShapeInferenceFn fn) {
  ShapeInferenceTestOp op(op_name);
  const int num_inputs = 1 + std::count(ins.begin(), ins.end(), ';');
  std::vector<NodeDefBuilder::NodeOut> src_list;
  for (int i = 0; i < num_inputs; ++i) src_list.emplace_back("a", 0, DT_FLOAT);
  NodeDef node_def;
  TF_CHECK_OK(NodeDefBuilder("dummy", op_name)
                  .Input(src_list)
                  .Attr("N", num_inputs)
                  .Finalize(&op.node_def));
  global_fn_ptr = &fn;
  return ShapeInferenceTestutil::InferShapes(op, ins, expected_outs)
      .error_message();
}

}  // namespace

TEST(ShapeInferenceTestutilTest, Failures) {
  auto fn_copy_input_0 = [](InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  };
  auto fn_copy_input_2 = [](InferenceContext* c) {
    c->set_output(0, c->input(2));
    return Status::OK();
  };
  auto fn_output_unknown_shapes = [](InferenceContext* c) {
    for (int i = 0; i < c->num_outputs(); ++i) {
      c->set_output(i, c->UnknownShape());
    }
    return Status::OK();
  };
  auto fn_output_1_2 = [](InferenceContext* c) {
    c->set_output(0, c->Matrix(1, 2));
    return Status::OK();
  };
  auto fn_output_u_2 = [](InferenceContext* c) {
    c->set_output(0, c->Matrix(InferenceContext::kUnknownDim, 2));
    return Status::OK();
  };
  const string& op = "OpOneOut";

  EXPECT_EQ("Shape inference should have returned error",
            RunInferShapes(op, "[1];[2];[1]", "e", fn_copy_input_0));
  EXPECT_EQ("Wrong number of expected outputs (2 vs 1)",
            RunInferShapes(op, "[1];[2];[1]", "[1];[2]", fn_copy_input_0));
  EXPECT_EQ("Op type not registered 'NoSuchOp'",
            ShapeInferenceTestutil::InferShapes(
                ShapeInferenceTestOp("NoSuchOp"), "", "")
                .error_message());

  // Wrong shape error messages.
  EXPECT_EQ(
      "Output 0 matched input 0 and should have not matched an input shape; "
      "output shape was [1]",
      RunInferShapes(op, "[1];[2];[1]", "?", fn_copy_input_0));
  EXPECT_EQ(
      "Output 0 matched input 0 and should have matched one of (in2); output "
      "shape was [1]",
      RunInferShapes(op, "[1];[2];[1]", "in2", fn_copy_input_0));
  EXPECT_EQ(
      "Output 0 matched input 0 and should have matched one of (in1|in2); "
      "output shape was [1]",
      RunInferShapes(op, "[1];[2];[1]", "in1|in2", fn_copy_input_0));
  EXPECT_EQ(
      "Output 0 matched input 2 and should have not matched an input shape; "
      "output shape was [1]",
      RunInferShapes(op, "[1];[2];[1]", "[1]", fn_copy_input_2));
  EXPECT_EQ("Output 0 did not match any input shape; output shape was [1,2]",
            RunInferShapes(op, "[1];[2];[1]", "in0|in1", fn_output_1_2));
  EXPECT_EQ("Output 0 expected to be unknown; output shape was [1,2]",
            RunInferShapes(op, "[1];[2];[1]", "?", fn_output_1_2));
  EXPECT_EQ("Output 0 expected rank 3 but was 2; output shape was [1,2]",
            RunInferShapes(op, "[1];[2];[1]", "[1,2,3]", fn_output_1_2));
  EXPECT_EQ(
      "Output 0 expected rank 2 but was ?; output shape was ?",
      RunInferShapes(op, "[1];[2];[1]", "[1,2]", fn_output_unknown_shapes));

  // Wrong shape error messages on the second output.
  EXPECT_EQ("Output 1 expected rank 3 but was ?; output shape was ?",
            RunInferShapes("OpTwoOut", "[1];[2];[1]", "?;[1,2,3]",
                           fn_output_unknown_shapes));

  // Wrong dimension error messages.
  EXPECT_EQ("Output dim 0,1 expected to be 3 but was 2; output shape was [1,2]",
            RunInferShapes(op, "[1];[2];[1]", "[1,3]", fn_output_1_2));
  EXPECT_EQ("Output dim 0,0 expected to be 2 but was 1; output shape was [1,2]",
            RunInferShapes(op, "[1];[2];[1]", "[2,2]", fn_output_1_2));
  EXPECT_EQ(
      "Output dim 0,0 expected to be unknown but was 1; output shape was [1,2]",
      RunInferShapes(op, "[1];[2];[1]", "[?,2]", fn_output_1_2));
  EXPECT_EQ("Output dim 0,1 expected to be 1 but was 2; output shape was [?,2]",
            RunInferShapes(op, "[1];[2];[1]", "[?,1]", fn_output_u_2));
  EXPECT_EQ("Output dim 0,0 expected to be 1 but was ?; output shape was [?,2]",
            RunInferShapes(op, "[0,1,?];[2];[1]", "[1,2]", fn_output_u_2));
  auto fn = [](InferenceContext* c) {
    c->set_output(0, c->MakeShape({c->Dim(c->input(0), 1), c->MakeDim(2),
                                   c->UnknownDim(), c->Dim(c->input(2), 0)}));
    return Status::OK();
  };
  const string ins = "[0,1,?];[2];[1]";
  EXPECT_EQ(
      "Output dim 0,0 expected to be unknown but matched input d0_1; output "
      "shape was [1,2,?,1]",
      RunInferShapes(op, ins, "[?,2,?,d2_0]", fn));
  EXPECT_EQ(
      "Output dim 0,0 expected to be 0 but matched input d0_1; output shape "
      "was [1,2,?,1]",
      RunInferShapes(op, ins, "[0,2,?,d2_0]", fn));
  EXPECT_EQ(
      "Output dim 0,0 matched input d0_1 and should have matched one of d0_0; "
      "output shape was [1,2,?,1]",
      RunInferShapes(op, ins, "[d0_0,2,?,d2_0]", fn));
  EXPECT_EQ(
      "Output dim 0,0 expected dim failed to parse as int64; output shape was "
      "[1,2,?,1]",
      RunInferShapes(op, ins, "[x,2,?,d2_0]", fn));
  EXPECT_EQ(
      "Output dim 0,0 matched input d0_1 and should have matched one of "
      "d0_0|d0_2; output shape was [1,2,?,1]",
      RunInferShapes(op, ins, "[d0_0|d0_2,2,?,d2_0]", fn));
  EXPECT_EQ(
      "Output dim 0,1 expected to be unknown but was 2; output shape was "
      "[1,2,?,1]",
      RunInferShapes(op, ins, "[d0_1,?,?,d0_0|d2_0]", fn));
  EXPECT_EQ(
      "Output dim 0,2 expected to be 8 but was ?; output shape was [1,2,?,1]",
      RunInferShapes(op, ins, "[d0_1,2,8,d0_0|d2_0]", fn));
  EXPECT_EQ(
      "Output dim 0,2 did not match any input dim; output shape was [1,2,?,1]",
      RunInferShapes(op, ins, "[d0_1,2,d0_1|d2_0,d0_0|d2_0]", fn));
  EXPECT_EQ("",  // OK, no error.
            RunInferShapes(op, ins, "[d0_1,2,?,d0_0|d2_0]", fn));
}

}  // namespace shape_inference
}  // namespace tensorflow
