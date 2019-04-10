/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/remapper.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class RemapperTest : public GrapplerTest {
 protected:
  // TODO(b/119765980): Upgrade upstream Eigen to set `m_can_use_xsmm=false` for
  // contractions with non-default contraction output kernels.
  bool EigenSupportsContractionOutputKernel() {
#if defined(EIGEN_USE_LIBXSMM)
    return false;
#endif
    return true;
  }
};

TEST_F(RemapperTest, FusedBatchNorm) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output dflt = ops::Const(s.WithOpName("dflt"), {3.14f, 2.7f}, {2, 1, 1, 1});
  Output x = ops::PlaceholderWithDefault(s.WithOpName("x"), dflt, {2, 1, 1, 1});
  Output scale = ops::Const(s.WithOpName("scale"), {0.3f}, {1});
  Output offset = ops::Const(s.WithOpName("offset"), {0.123f}, {1});
  Output mean = ops::Const(s.WithOpName("mean"), {7.3f}, {1});
  Output variance = ops::Const(s.WithOpName("variance"), {0.57f}, {1});
  ops::FusedBatchNorm::Attrs attr;
  attr = attr.IsTraining(false);
  ops::FusedBatchNorm bn(s.WithOpName("batch_norm"), x, scale, offset, mean,
                         variance, attr);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"batch_norm"};

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(1, tensors_expected.size());

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(RemapperTest, FusedBatchNormNCHW) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output dflt =
      ops::Const(s.WithOpName("dflt"), {3.14f, 2.7f, 1.0f, 2.0f, 3.0f, 100.0f},
                 {1, 3, 1, 2});
  Output x = ops::PlaceholderWithDefault(s.WithOpName("x"), dflt, {1, 3, 1, 2});
  Output scale = ops::Const(s.WithOpName("scale"), {0.3f, 7.0f, 123.0f}, {3});
  Output offset =
      ops::Const(s.WithOpName("offset"), {0.123f, 2.1f, 0.55f}, {3});
  Output mean = ops::Const(s.WithOpName("mean"), {7.3f, 8.3f, 3.1f}, {3});
  Output variance =
      ops::Const(s.WithOpName("variance"), {0.57f, 1.0f, 2.0f}, {3});
  ops::FusedBatchNorm::Attrs attr;
  attr = attr.IsTraining(false);
  attr = attr.DataFormat("NCHW");
  ops::FusedBatchNorm bn(s.WithOpName("batch_norm").WithDevice("/device:GPU:0"),
                         x, scale, offset, mean, variance, attr);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"batch_norm"};

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

  if (GetNumAvailableGPUs() > 0) {
    // NCHW batch norm is only supported on GPU.
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
    EXPECT_EQ(1, tensors_expected.size());
    auto tensors = EvaluateNodes(output, item.fetch);
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-5);
  }
}

TEST_F(RemapperTest, FuseConv2DWithBias) {
  if (!EigenSupportsContractionOutputKernel()) return;

  using ::tensorflow::ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
  auto bias_shape = ops::Placeholder::Shape({128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv = ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);
  auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "bias_add") {
      EXPECT_EQ("_FusedConv2D", node.op());
      EXPECT_EQ("input", node.input(0));
      EXPECT_EQ("filter", node.input(1));

      EXPECT_EQ(1, node.attr().at("num_args").i());
      EXPECT_EQ("bias", node.input(2));

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      EXPECT_EQ(1, fused_ops.size());
      EXPECT_EQ("BiasAdd", fused_ops[0]);
      found++;
    }
  }
  EXPECT_EQ(1, found);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(RemapperTest, FuseMatMulWithBias) {
  if (!EigenSupportsContractionOutputKernel()) return;

  using ::tensorflow::ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto lhs_shape = ops::Placeholder::Shape({8, 32});
  auto rhs_shape = ops::Placeholder::Shape({32, 64});
  auto bias_shape = ops::Placeholder::Shape({64});

  auto lhs = Placeholder(s.WithOpName("lhs"), DT_FLOAT, lhs_shape);
  auto rhs = Placeholder(s.WithOpName("rhs"), DT_FLOAT, rhs_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);
  auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);

  auto lhs_t = GenerateRandomTensor<DT_FLOAT>({8, 32});
  auto rhs_t = GenerateRandomTensor<DT_FLOAT>({32, 64});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({64});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}, {"bias", bias_t}};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "bias_add") {
      EXPECT_EQ("_FusedMatMul", node.op());
      EXPECT_EQ("lhs", node.input(0));
      EXPECT_EQ("rhs", node.input(1));

      EXPECT_EQ(1, node.attr().at("num_args").i());
      EXPECT_EQ("bias", node.input(2));

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      EXPECT_EQ(1, fused_ops.size());
      EXPECT_EQ("BiasAdd", fused_ops[0]);
      found++;
    }
  }
  EXPECT_EQ(1, found);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(RemapperTest, FuseConv2DWithBiasAndActivation) {
  if (!EigenSupportsContractionOutputKernel()) return;

  using ::tensorflow::ops::Placeholder;

  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = Placeholder::Shape({8, 32, 32, 3});
    auto filter_shape = Placeholder::Shape({1, 1, 3, 128});
    auto bias_shape = Placeholder::Shape({128});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
    auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

    std::vector<int> strides = {1, 1, 1, 1};
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);

    ops::Identity fetch = [&]() -> ops::Identity {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");

      if (activation == "Relu") {
        return ops::Identity(fetch, ops::Relu(activate, bias_add));
      } else if (activation == "Relu6") {
        return ops::Identity(fetch, ops::Relu6(activate, bias_add));
      } else if (activation == "Elu") {
        return ops::Identity(fetch, ops::Elu(activate, bias_add));
      }

      return ops::Identity(fetch, bias);
    }();

    auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
    auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
    auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "activation") {
        EXPECT_EQ("_FusedConv2D", node.op());
        EXPECT_EQ("input", node.input(0));
        EXPECT_EQ("filter", node.input(1));

        EXPECT_EQ(1, node.attr().at("num_args").i());
        EXPECT_EQ("bias", node.input(2));

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(2, fused_ops.size());
        EXPECT_EQ("BiasAdd", fused_ops[0]);
        EXPECT_EQ(activation, fused_ops[1]);
        found++;
      }
    }
    EXPECT_EQ(1, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    EXPECT_EQ(1, tensors_expected.size());
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
  }
}

TEST_F(RemapperTest, FuseMatMulWithBiasAndActivation) {
  if (!EigenSupportsContractionOutputKernel()) return;

  using ::tensorflow::ops::Placeholder;

  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto lhs_shape = ops::Placeholder::Shape({8, 32});
    auto rhs_shape = ops::Placeholder::Shape({32, 64});
    auto bias_shape = ops::Placeholder::Shape({64});

    auto lhs = Placeholder(s.WithOpName("lhs"), DT_FLOAT, lhs_shape);
    auto rhs = Placeholder(s.WithOpName("rhs"), DT_FLOAT, rhs_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

    auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);

    ops::Identity fetch = [&]() -> ops::Identity {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");

      if (activation == "Relu") {
        return ops::Identity(fetch, ops::Relu(activate, bias_add));
      } else if (activation == "Relu6") {
        return ops::Identity(fetch, ops::Relu6(activate, bias_add));
      } else if (activation == "Elu") {
        return ops::Identity(fetch, ops::Elu(activate, bias_add));
      }

      return ops::Identity(fetch, bias);
    }();

    auto lhs_t = GenerateRandomTensor<DT_FLOAT>({8, 32});
    auto rhs_t = GenerateRandomTensor<DT_FLOAT>({32, 64});
    auto bias_t = GenerateRandomTensor<DT_FLOAT>({64});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}, {"bias", bias_t}};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "activation") {
        EXPECT_EQ("_FusedMatMul", node.op());
        EXPECT_EQ("lhs", node.input(0));
        EXPECT_EQ("rhs", node.input(1));

        EXPECT_EQ(1, node.attr().at("num_args").i());
        EXPECT_EQ("bias", node.input(2));

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(2, fused_ops.size());
        EXPECT_EQ("BiasAdd", fused_ops[0]);
        EXPECT_EQ(activation, fused_ops[1]);
        found++;
      }
    }
    EXPECT_EQ(1, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    EXPECT_EQ(1, tensors_expected.size());
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
  }
}

TEST_F(RemapperTest, FuseConv2DWithBatchNorm) {
  if (!EigenSupportsContractionOutputKernel()) return;

  using ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
  auto scale_shape = ops::Placeholder::Shape({128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT, scale_shape);
  auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT, scale_shape);
  auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT, scale_shape);
  auto variance = Placeholder(s.WithOpName("variance"), DT_FLOAT, scale_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv = ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
  ops::FusedBatchNorm::Attrs attrs;
  attrs = attrs.IsTraining(false);
  auto batch_norm = ops::FusedBatchNorm(s.WithOpName("batch_norm"), conv, scale,
                                        offset, mean, variance, attrs);
  auto fetch = ops::Identity(s.WithOpName("fetch"), batch_norm.y);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
  auto scale_t = GenerateRandomTensor<DT_FLOAT>({128});
  auto offset_t = GenerateRandomTensor<DT_FLOAT>({128});
  auto mean_t = GenerateRandomTensor<DT_FLOAT>({128});
  auto variance_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}, {"filter", filter_t},
               {"scale", scale_t}, {"offset", offset_t},
               {"mean", mean_t},   {"variance", variance_t}};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "batch_norm") {
      EXPECT_EQ("_FusedConv2D", node.op());
      EXPECT_EQ("input", node.input(0));
      EXPECT_EQ("filter", node.input(1));

      EXPECT_EQ(4, node.attr().at("num_args").i());
      EXPECT_EQ("scale", node.input(2));
      EXPECT_EQ("offset", node.input(3));
      EXPECT_EQ("mean", node.input(4));
      EXPECT_EQ("variance", node.input(5));

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      EXPECT_EQ(1, fused_ops.size());
      EXPECT_EQ("FusedBatchNorm", fused_ops[0]);
      found++;
    }
  }
  EXPECT_EQ(1, found);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(RemapperTest, FuseConv2DWithBatchNormAndActivation) {
  if (!EigenSupportsContractionOutputKernel()) return;

  using ops::Placeholder;

  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
    auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
    auto scale_shape = ops::Placeholder::Shape({128});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
    auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
    auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT, scale_shape);
    auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT, scale_shape);
    auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT, scale_shape);
    auto variance =
        Placeholder(s.WithOpName("variance"), DT_FLOAT, scale_shape);

    std::vector<int> strides = {1, 1, 1, 1};
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
    ops::FusedBatchNorm::Attrs attrs;
    attrs = attrs.IsTraining(false);
    auto batch_norm = ops::FusedBatchNorm(s.WithOpName("batch_norm"), conv,
                                          scale, offset, mean, variance, attrs);

    ops::Identity fetch = [&]() -> ops::Identity {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");

      if (activation == "Relu") {
        return ops::Identity(fetch, ops::Relu(activate, batch_norm.y));
      } else if (activation == "Relu6") {
        return ops::Identity(fetch, ops::Relu6(activate, batch_norm.y));
      } else if (activation == "Elu") {
        return ops::Identity(fetch, ops::Elu(activate, batch_norm.y));
      }

      return ops::Identity(fetch, batch_norm.y);
    }();

    auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
    auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
    auto scale_t = GenerateRandomTensor<DT_FLOAT>({128});
    auto offset_t = GenerateRandomTensor<DT_FLOAT>({128});
    auto mean_t = GenerateRandomTensor<DT_FLOAT>({128});
    auto variance_t = GenerateRandomTensor<DT_FLOAT>({128});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"filter", filter_t},
                 {"scale", scale_t}, {"offset", offset_t},
                 {"mean", mean_t},   {"variance", variance_t}};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "activation") {
        EXPECT_EQ("_FusedConv2D", node.op());
        EXPECT_EQ("input", node.input(0));
        EXPECT_EQ("filter", node.input(1));

        EXPECT_EQ(4, node.attr().at("num_args").i());
        EXPECT_EQ("scale", node.input(2));
        EXPECT_EQ("offset", node.input(3));
        EXPECT_EQ("mean", node.input(4));
        EXPECT_EQ("variance", node.input(5));

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        EXPECT_EQ(2, fused_ops.size());
        EXPECT_EQ("FusedBatchNorm", fused_ops[0]);
        EXPECT_EQ(activation, fused_ops[1]);
        found++;
      }
    }
    EXPECT_EQ(1, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    EXPECT_EQ(1, tensors_expected.size());
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
  }
}

TEST_F(RemapperTest, FuseConv2DWithSqueezeAndBias) {
  if (!EigenSupportsContractionOutputKernel()) return;

  using ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 32, 1, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
  auto bias_shape = ops::Placeholder::Shape({128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv = ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");

  ops::Squeeze::Attrs attrs;
  attrs = attrs.Axis({2});
  auto squeeze = ops::Squeeze(s.WithOpName("squeeze"), conv, attrs);

  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), squeeze, bias);
  auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 1, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "conv") {
      EXPECT_EQ("_FusedConv2D", node.op());
      EXPECT_EQ("input", node.input(0));
      EXPECT_EQ("filter", node.input(1));

      EXPECT_EQ(1, node.attr().at("num_args").i());
      EXPECT_EQ("bias", node.input(2));

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(1, fused_ops.size());
      EXPECT_EQ("BiasAdd", fused_ops[0]);
      found++;
    } else if (node.name() == "bias_add") {
      EXPECT_EQ("Squeeze", node.op());
      EXPECT_EQ("conv", node.input(0));
      found++;
    }
  }
  EXPECT_EQ(2, found);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

}  // namespace grappler
}  // namespace tensorflow
