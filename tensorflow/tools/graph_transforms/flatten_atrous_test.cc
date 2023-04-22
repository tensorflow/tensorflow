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

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
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
Status FlattenAtrousConv(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def);

class FlattenAtrousConvTest : public ::testing::Test {
 protected:
  template <class TConvOp>
  void TestFlattenAtrousConv() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 3, 3, 2}));
    test::FillValues<float>(
        &input_data, {.1f, .4f, .2f, .5f, .3f, .6f, -1.0f, -.4f, -.2f, -.5f,
                      -.3f, -.6f, .1f, .4f, .2f, .5f, .3f, .6f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor block_shape_data(DT_INT32, TensorShape({2}));
    test::FillValues<int>(&block_shape_data, {2, 2});
    Output block_shape_op = Const(root.WithOpName("block_shape_op"),
                                  Input::Initializer(block_shape_data));

    Tensor paddings_data(DT_INT32, TensorShape({2, 2}));
    test::FillValues<int>(&paddings_data, {1, 2, 1, 2});
    Output paddings_op = Const(root.WithOpName("paddings_op"),
                               Input::Initializer(paddings_data));

    Output space_to_batch_op =
        SpaceToBatchND(root.WithOpName("space_to_batch_op"), input_op,
                       block_shape_op, paddings_op);

    Tensor weights_data(DT_FLOAT, TensorShape({2, 2, 2, 1}));
    test::FillValues<float>(&weights_data,
                            {.1f, .2f, .3f, .4f, .1f, .2f, .3f, .4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = TConvOp(root.WithOpName("conv_op"), space_to_batch_op,
                             weights_op, {1, 1, 1, 1}, "VALID");

    Tensor crops_data(DT_INT32, TensorShape({2, 2}));
    test::FillValues<int>(&crops_data, {0, 1, 0, 1});
    Output crops_op =
        Const(root.WithOpName("crops_op"), Input::Initializer(crops_data));

    Output batch_to_space_op = BatchToSpaceND(
        root.WithOpName("output"), conv_op, block_shape_op, crops_op);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef modified_graph_def;
    TF_ASSERT_OK(FlattenAtrousConv(original_graph_def, {{}, {"output"}},
                                   &modified_graph_def));

    std::unique_ptr<Session> modified_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(modified_session->Create(modified_graph_def));
    std::vector<Tensor> modified_outputs;
    TF_ASSERT_OK(modified_session->Run({}, {"output"}, {}, &modified_outputs));

    EXPECT_EQ(3, modified_graph_def.node_size());

    EXPECT_EQ("input_op", modified_graph_def.node(0).name());
    EXPECT_EQ("weights_op", modified_graph_def.node(1).name());
    EXPECT_EQ("output", modified_graph_def.node(2).name());

    EXPECT_EQ("Const", modified_graph_def.node(0).op());
    EXPECT_EQ("Const", modified_graph_def.node(1).op());
    EXPECT_EQ(conv_op.node()->type_string(), modified_graph_def.node(2).op());

    test::ExpectTensorNear<float>(original_outputs[0], modified_outputs[0],
                                  1e-6);
  }
};

TEST_F(FlattenAtrousConvTest, TestFlattenAtrousConv2D) {
  TestFlattenAtrousConv<::tensorflow::ops::Conv2D>();
}
TEST_F(FlattenAtrousConvTest, TestFlattenAtrousDepthwiseConv2dNative) {
  TestFlattenAtrousConv<::tensorflow::ops::DepthwiseConv2dNative>();
}

}  // namespace graph_transforms
}  // namespace tensorflow
