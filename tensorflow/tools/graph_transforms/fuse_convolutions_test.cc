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
Status FuseResizePadAndConv(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def);
Status FuseResizeAndConv(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def);
Status FusePadAndConv(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef* output_graph_def);

class FuseConvolutionsTest : public ::testing::Test {
 protected:
  void TestFuseResizePadAndConv() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 2, 3, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Output resize_op = ResizeBilinear(root.WithOpName("resize_op"), input_op,
                                      Const(root.WithOpName("size"), {12, 4}),
                                      ResizeBilinear::AlignCorners(false));

    Tensor pad_dims_data(DT_INT32, TensorShape({4, 2}));
    test::FillValues<int32>(&pad_dims_data, {0, 0, 1, 1, 2, 2, 0, 0});
    Output pad_dims_op = Const(root.WithOpName("pad_dims_op"),
                               Input::Initializer(pad_dims_data));
    Output pad_op =
        MirrorPad(root.WithOpName("pad_op"), resize_op, pad_dims_op, "REFLECT");

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("output"), pad_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(FuseResizePadAndConv(original_graph_def, {{}, {"output"}},
                                      &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 1e-5);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("Conv2D", node.op());
      EXPECT_NE("MirrorPad", node.op());
      EXPECT_NE("ResizeBilinear", node.op());
    }
  }

  void TestFuseResizeAndConv() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 2, 3, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Output resize_op = ResizeBilinear(root.WithOpName("resize_op"), input_op,
                                      Const(root.WithOpName("size"), {12, 4}),
                                      ResizeBilinear::AlignCorners(false));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("output"), resize_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(FuseResizeAndConv(original_graph_def, {{}, {"output"}},
                                   &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 1e-5);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("Conv2D", node.op());
      EXPECT_NE("ResizeBilinear", node.op());
    }
  }

  void TestFusePadAndConv() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 2, 3, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor pad_dims_data(DT_INT32, TensorShape({4, 2}));
    test::FillValues<int32>(&pad_dims_data, {0, 0, 1, 1, 2, 2, 0, 0});
    Output pad_dims_op = Const(root.WithOpName("pad_dims_op"),
                               Input::Initializer(pad_dims_data));
    Output pad_op =
        MirrorPad(root.WithOpName("pad_op"), input_op, pad_dims_op, "REFLECT");

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("output"), pad_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(
        FusePadAndConv(original_graph_def, {{}, {"output"}}, &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 1e-5);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("Conv2D", node.op());
      EXPECT_NE("MirrorPad", node.op());
    }
  }
};

TEST_F(FuseConvolutionsTest, TestFuseResizePadAndConv) {
  TestFuseResizePadAndConv();
}

TEST_F(FuseConvolutionsTest, TestFuseResizeAndConv) { TestFuseResizeAndConv(); }

TEST_F(FuseConvolutionsTest, TestFusePadAndConv) { TestFusePadAndConv(); }

}  // namespace graph_transforms
}  // namespace tensorflow
