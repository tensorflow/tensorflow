/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/remapper.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class MklRemapperTest : public GrapplerTest {};

TEST_F(MklRemapperTest, FuseConv2DWithBiasAndAddN) {
  using ::tensorflow::ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
  auto input_shape_addn = ops::Placeholder::Shape({8, 32, 32, 128});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
  auto bias_shape = ops::Placeholder::Shape({128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto input_addn =
      Placeholder(s.WithOpName("input_addn"), DT_FLOAT, input_shape_addn);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv = ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);
  auto addn = ops::AddN(s.WithOpName("addn"),
                        std::initializer_list<Input>{input_addn, bias_add});
  auto fetch = ops::Identity(s.WithOpName("fetch"), addn);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
  auto input_addn_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 128});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t},
               {"filter", filter_t},
               {"bias", bias_t},
               {"input_addn", input_addn_t}};
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
    if (node.name() == "addn") {
      EXPECT_EQ("_FusedConv2D", node.op());
      EXPECT_EQ("input", node.input(0));
      EXPECT_EQ("filter", node.input(1));

      EXPECT_EQ(2, node.attr().at("num_args").i());
      EXPECT_EQ("bias", node.input(2));
      EXPECT_EQ("input_addn", node.input(3));

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      EXPECT_EQ(2, fused_ops.size());
      EXPECT_EQ("BiasAdd", fused_ops[0]);
      EXPECT_EQ("Add", fused_ops[1]);
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

TEST_F(MklRemapperTest, FuseConv2DWithBiasAndAddNRelu) {
  using ::tensorflow::ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
  auto input_shape_addn = ops::Placeholder::Shape({8, 32, 32, 128});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
  auto bias_shape = ops::Placeholder::Shape({128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto input_addn =
      Placeholder(s.WithOpName("input_addn"), DT_FLOAT, input_shape_addn);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv = ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);
  auto addn = ops::AddN(s.WithOpName("addn"),
                        std::initializer_list<Input>{input_addn, bias_add});
  auto relu = ops::Relu(s.WithOpName("relu"), addn);
  auto fetch = ops::Identity(s.WithOpName("fetch"), relu);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
  auto input_addn_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 128});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t},
               {"filter", filter_t},
               {"bias", bias_t},
               {"input_addn", input_addn_t}};
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
    if (node.name() == "relu") {
      EXPECT_EQ("_FusedConv2D", node.op());
      EXPECT_EQ("input", node.input(0));
      EXPECT_EQ("filter", node.input(1));

      EXPECT_EQ(2, node.attr().at("num_args").i());
      EXPECT_EQ("bias", node.input(2));
      EXPECT_EQ("input_addn", node.input(3));

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      EXPECT_EQ(3, fused_ops.size());
      EXPECT_EQ("BiasAdd", fused_ops[0]);
      EXPECT_EQ("Add", fused_ops[1]);
      EXPECT_EQ("Relu", fused_ops[2]);
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

#ifdef ENABLE_MKLDNN_V1
TEST_F(MklRemapperTest, FuseBatchNormWithRelu) {
  using ::tensorflow::ops::Placeholder;

  for (bool is_training : {true, false}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    const int num_channels = 24;

    TensorShape channel_shape({num_channels});
    TensorShape empty_shape({0});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT,
                             ops::Placeholder::Shape({2, 8, 8, num_channels}));
    auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_FLOAT);
    auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT);
    auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT);
    auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT);
    auto var = Placeholder(s.WithOpName("var"), DT_FLOAT);

    float epsilon = 0.1f;
    auto fbn = ops::FusedBatchNormV3(
        s.WithOpName("fused_batch_norm"), input_cast, scale, offset, mean, var,
        ops::FusedBatchNormV3::IsTraining(is_training)
            .Epsilon(epsilon)
            .DataFormat("NHWC"));
    auto relu = ops::Relu(s.WithOpName("relu"), fbn.y);
    auto fetch = ops::Identity(s.WithOpName("fetch"), relu);

    auto input_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});
    auto scale_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
    auto offset_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
    auto mean_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                             : channel_shape);
    auto var_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                            : channel_shape);

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t},
                 {"scale", scale_t},
                 {"offset", offset_t},
                 {"mean", mean_t},
                 {"var", var_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::AGGRESSIVE);  // trust placeholders shape
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "relu") {
        EXPECT_EQ(node.op(), "Identity");
        ASSERT_EQ(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "fused_batch_norm");
        found++;
      }
      if (node.name() == "fused_batch_norm") {
        EXPECT_EQ(node.op(), "_FusedBatchNormEx");
        ASSERT_EQ(node.input_size(), 5);
        EXPECT_EQ(node.input(0), "input_cast");
        EXPECT_EQ(node.input(1), "scale");
        EXPECT_EQ(node.input(2), "offset");
        EXPECT_EQ(node.input(3), "mean");
        EXPECT_EQ(node.input(4), "var");

        auto attr = node.attr();
        EXPECT_EQ(attr["num_side_inputs"].i(), 0);
        EXPECT_EQ(attr["activation_mode"].s(), "Relu");
        found++;
      }
    }
    EXPECT_EQ(found, 2);
  }
}

TEST_F(MklRemapperTest, FuseBatchNormWithAddAndRelu) {
  using ::tensorflow::ops::Placeholder;

  for (bool is_training : {true, false}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    const int num_channels = 24;

    TensorShape input_shape({2, 8, 8, num_channels});
    TensorShape channel_shape({num_channels});
    TensorShape empty_shape({0});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT,
                             ops::Placeholder::Shape(input_shape));
    auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_FLOAT);
    auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT);
    auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT);
    auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT);
    auto var = Placeholder(s.WithOpName("var"), DT_FLOAT);
    auto side_input = Placeholder(s.WithOpName("side_input"), DT_FLOAT,
                                  ops::Placeholder::Shape(input_shape));
    auto side_input_cast =
        ops::Cast(s.WithOpName("side_input_cast"), side_input, DT_FLOAT);

    float epsilon = 0.1f;
    auto fbn = ops::FusedBatchNormV3(
        s.WithOpName("fused_batch_norm"), input_cast, scale, offset, mean, var,
        ops::FusedBatchNormV3::IsTraining(is_training)
            .Epsilon(epsilon)
            .DataFormat("NHWC"));
    auto add = ops::Add(s.WithOpName("add"), fbn.y, side_input_cast);
    auto relu = ops::Relu(s.WithOpName("relu"), add);
    auto fetch = ops::Identity(s.WithOpName("fetch"), relu);

    auto input_t = GenerateRandomTensor<DT_FLOAT>(input_shape);
    auto scale_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
    auto offset_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
    auto mean_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                             : channel_shape);
    auto var_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                            : channel_shape);
    auto side_input_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t},   {"scale", scale_t},
                 {"offset", offset_t}, {"mean", mean_t},
                 {"var", var_t},       {"side_input", side_input_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::AGGRESSIVE);  // trust placeholders shape
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "add") {
        EXPECT_EQ(node.op(), "Add");
        ASSERT_EQ(node.input_size(), 2);
        EXPECT_EQ(node.input(0), "fused_batch_norm");
        EXPECT_EQ(node.input(1), "side_input_cast");
        found++;
      }
      if (node.name() == "relu") {
        EXPECT_EQ(node.op(), "Relu");
        ASSERT_EQ(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "add");
        found++;
      }
      if (node.name() == "fused_batch_norm") {
        EXPECT_EQ(node.op(), "FusedBatchNormV3");
        ASSERT_EQ(node.input_size(), 5);
        EXPECT_EQ(node.input(0), "input_cast");
        EXPECT_EQ(node.input(1), "scale");
        EXPECT_EQ(node.input(2), "offset");
        EXPECT_EQ(node.input(3), "mean");
        EXPECT_EQ(node.input(4), "var");
        found++;
      }
    }
    EXPECT_EQ(found, 3);
  }
}
#endif  // ENABLE_MKLDNN_V1

}  // namespace grappler
}  // namespace tensorflow
#endif  // INTEL_MKL
