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

#define REGISTER_TEST_FLOAT32(TEST) REGISTER_TEST(TEST, DT_FLOAT, Float32Input);

#ifdef ENABLE_INTEL_MKL_BFLOAT16
#define REGISTER_TEST_BFLOAT16(TEST) \
  REGISTER_TEST(TEST, DT_BFLOAT16, BFloat16Input);

#define REGISTER_TEST_ALL_TYPES(TEST) \
  REGISTER_TEST_FLOAT32(TEST);        \
  REGISTER_TEST_BFLOAT16(TEST);
#else
#define REGISTER_TEST_ALL_TYPES(TEST) REGISTER_TEST_FLOAT32(TEST);
#endif  // ENABLE_INTEL_MKL_BFLOAT16

namespace tensorflow {
namespace grappler {

class MklRemapperTest : public GrapplerTest {
 public:
  const string kAddNOp = "AddN";
  const string kAddOp = "Add";
  const string kAddV2Op = "AddV2";

 protected:
  void FuseConv2DWithBiasAndAddNOrAdd(const string& data_format, bool has_relu,
                                      string add_op, bool add_with_bcast) {
    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = (data_format == "NHWC")
                           ? ops::Placeholder::Shape({8, 32, 32, 3})
                           : ops::Placeholder::Shape({8, 3, 32, 32});
    auto input_shape_addn = ops::Placeholder::Shape({});
    if (data_format == "NHWC") {
      if (add_with_bcast)
        input_shape_addn = ops::Placeholder::Shape({128});
      else
        input_shape_addn = ops::Placeholder::Shape({8, 32, 32, 128});
    } else {
      if (add_with_bcast)
        input_shape_addn = ops::Placeholder::Shape({32});
      else
        input_shape_addn = ops::Placeholder::Shape({8, 128, 32, 32});
    }
    auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
    auto bias_shape = ops::Placeholder::Shape({128});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
    auto input_addn =
        Placeholder(s.WithOpName("input_addn"), DT_FLOAT, input_shape_addn);
    auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

    std::vector<int> strides = {1, 1, 1, 1};
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME",
                    ops::Conv2D::Attrs().DataFormat(data_format));
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias,
                                 ops::BiasAdd::Attrs().DataFormat(data_format));
    if (add_op == kAddNOp) {
      auto addn = ops::AddN(s.WithOpName(add_op),
                            std::initializer_list<Input>{input_addn, bias_add});
      if (has_relu) {
        auto relu = ops::Relu(s.WithOpName("relu"), addn);
        ops::Identity(s.WithOpName("fetch"), relu);
      } else {
        ops::Identity(s.WithOpName("fetch"), addn);
      }
    } else if (add_op == kAddV2Op) {
      auto add = ops::AddV2(s.WithOpName(add_op), input_addn, bias_add);
      if (has_relu) {
        auto relu = ops::Relu(s.WithOpName("relu"), add);
        ops::Identity(s.WithOpName("fetch"), relu);
      } else {
        ops::Identity(s.WithOpName("fetch"), add);
      }
    } else {
      auto add = ops::Add(s.WithOpName(add_op), input_addn, bias_add);
      if (has_relu) {
        auto relu = ops::Relu(s.WithOpName("relu"), add);
        ops::Identity(s.WithOpName("fetch"), relu);
      } else {
        ops::Identity(s.WithOpName("fetch"), add);
      }
    }
    auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(input_shape.shape_.dim_sizes()));
    auto input_addn_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(input_shape_addn.shape_.dim_sizes()));
    auto filter_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(filter_shape.shape_.dim_sizes()));
    auto bias_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(bias_shape.shape_.dim_sizes()));

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_tensor},
                 {"filter", filter_tensor},
                 {"bias", bias_tensor},
                 {"input_addn", input_addn_tensor}};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    // Set Rewriter config to AGGRESSIVE so that we can use Placeholder shape
    // to test that Add with both inputs having same shape get fused with
    // Conv2D. Setting this config to AGGRESSIVE is not required for the feature
    // though.
    Remapper optimizer(RewriterConfig::AGGRESSIVE);
    GraphDef output;
    TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

    bool check_fusion = !add_with_bcast;
    int found = 0;
    for (const NodeDef& node : output.node()) {
      auto fetch_node_name = has_relu ? "relu" : add_op;
      if (node.name() == fetch_node_name) {
        if (check_fusion) {
          EXPECT_EQ("_FusedConv2D", node.op());
          EXPECT_EQ("input", node.input(0));
          EXPECT_EQ("filter", node.input(1));

          EXPECT_EQ(2, node.attr().at("num_args").i());
          EXPECT_EQ("bias", node.input(2));
          EXPECT_EQ("input_addn", node.input(3));

          const auto fused_ops = node.attr().at("fused_ops").list().s();
          if (has_relu) {
            EXPECT_EQ(3, fused_ops.size());
            EXPECT_EQ("BiasAdd", fused_ops[0]);
            EXPECT_EQ("Add", fused_ops[1]);
            EXPECT_EQ("Relu", fused_ops[2]);
          } else {
            EXPECT_EQ(2, fused_ops.size());
            EXPECT_EQ("BiasAdd", fused_ops[0]);
            EXPECT_EQ("Add", fused_ops[1]);
          }
        } else {
          if (has_relu) {
            EXPECT_EQ(node.op(), "Relu");
            ASSERT_EQ(node.input_size(), 1);
            EXPECT_EQ(node.input(0), add_op);
          } else {
            EXPECT_EQ(node.op(), add_op);
            ASSERT_EQ(node.input_size(), 2);
          }
        }
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
};

#define CREATE_CONV2DFUSION_TEST(data_format, addop, relu, bcast)                    \
  TEST_F(                                                                            \
      MklRemapperTest,                                                               \
      FuseConv2DWithBiasAnd##addop##_##data_format##_relu##relu##_addbcast##bcast) { \
    const bool kShouldFuseRelu = relu;                                               \
    const bool kIsAddWithBcast = bcast;                                              \
    FuseConv2DWithBiasAndAddNOrAdd(#data_format, relu, #addop, bcast);               \
  }

#define CREATE_CONV2DFUSION_ADD_NOBCAST_TEST(addop)    \
  CREATE_CONV2DFUSION_TEST(NHWC, addop, false, false); \
  CREATE_CONV2DFUSION_TEST(NHWC, addop, true, false);  \
  CREATE_CONV2DFUSION_TEST(NCHW, addop, false, false); \
  CREATE_CONV2DFUSION_TEST(NCHW, addop, true, false);

CREATE_CONV2DFUSION_ADD_NOBCAST_TEST(AddN);

#define CREATE_CONV2DFUSION_ADD_BCAST_TEST(addop)      \
  CREATE_CONV2DFUSION_TEST(NHWC, addop, false, false); \
  CREATE_CONV2DFUSION_TEST(NHWC, addop, true, false);  \
  CREATE_CONV2DFUSION_TEST(NCHW, addop, false, false); \
  CREATE_CONV2DFUSION_TEST(NCHW, addop, true, false);  \
  CREATE_CONV2DFUSION_TEST(NHWC, addop, false, true);  \
  CREATE_CONV2DFUSION_TEST(NHWC, addop, true, true);   \
  CREATE_CONV2DFUSION_TEST(NCHW, addop, false, true);  \
  CREATE_CONV2DFUSION_TEST(NCHW, addop, true, true);

CREATE_CONV2DFUSION_ADD_BCAST_TEST(Add);
CREATE_CONV2DFUSION_ADD_BCAST_TEST(AddV2);

#undef CREATE_CONV2DFUSION_ADD_NOBCAST_TEST
#undef CREATE_CONV2DFUSION_ADD_BCAST_TEST
#undef CREATE_CONV2DFUSION_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklRemapperTest, NAME##_##T) {                                       \
    using ::tensorflow::ops::Placeholder;                                     \
                                                                              \
    for (const string& activation : {"Relu", "Relu6", "Elu", "None"}) {       \
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();                \
                                                                              \
      auto input_shape = Placeholder::Shape({8, 32, 32, 3});                  \
      auto filter_shape = Placeholder::Shape({1, 1, 3, 1});                   \
      auto bias_shape = Placeholder::Shape({3});                              \
                                                                              \
      auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape); \
      auto filter =                                                           \
          Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);        \
      auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);    \
                                                                              \
      std::vector<int> strides = {1, 1, 1, 1};                                \
      auto conv = ops::DepthwiseConv2dNative(s.WithOpName("depthwise_conv"),  \
                                             input, filter, strides, "SAME"); \
      auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);     \
                                                                              \
      ops::Identity fetch = [&]() -> ops::Identity {                          \
        auto activate = s.WithOpName("activation");                           \
        auto fetch = s.WithOpName("fetch");                                   \
                                                                              \
        if (activation == "Relu") {                                           \
          return ops::Identity(fetch, ops::Relu(activate, bias_add));         \
        } else if (activation == "Relu6") {                                   \
          return ops::Identity(fetch, ops::Relu6(activate, bias_add));        \
        } else if (activation == "Elu") {                                     \
          return ops::Identity(fetch, ops::Elu(activate, bias_add));          \
        }                                                                     \
                                                                              \
        DCHECK(activation == "None");                                         \
        return ops::Identity(fetch, bias_add);                                \
      }();                                                                    \
                                                                              \
      auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});          \
      auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 1});           \
      auto bias_t = GenerateRandomTensor<DT_FLOAT>({3});                      \
                                                                              \
      GrapplerItem item;                                                      \
      item.fetch = {"fetch"};                                                 \
      item.feed = {                                                           \
          {"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};        \
      TF_CHECK_OK(s.ToGraphDef(&item.graph));                                 \
                                                                              \
      for (int i = 0; i < item.graph.node_size(); ++i) {                      \
        item.graph.mutable_node(i)->set_device("/device:CPU:0");              \
      }                                                                       \
                                                                              \
      Remapper optimizer(RewriterConfig::ON);                                 \
      GraphDef output;                                                        \
      TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));                \
                                                                              \
      int found = 0;                                                          \
      for (const NodeDef& node : output.node()) {                             \
        if (node.name() != "bias_add" && node.name() != "activation")         \
          continue;                                                           \
                                                                              \
        EXPECT_EQ(node.op(), "_FusedDepthwiseConv2dNative");                  \
        ASSERT_EQ(node.input_size(), 3);                                      \
        EXPECT_EQ(node.input(0), "input");                                    \
        EXPECT_EQ(node.input(1), "filter");                                   \
                                                                              \
        EXPECT_EQ(node.attr().at("num_args").i(), 1);                         \
        EXPECT_EQ(node.input(2), "bias");                                     \
                                                                              \
        const auto fused_ops = node.attr().at("fused_ops").list().s();        \
        if (node.name() == "bias_add") {                                      \
          ASSERT_EQ(fused_ops.size(), 1);                                     \
          EXPECT_EQ(fused_ops[0], "BiasAdd");                                 \
          found++;                                                            \
        }                                                                     \
        if (node.name() == "activation") {                                    \
          ASSERT_EQ(fused_ops.size(), 2);                                     \
          EXPECT_EQ(fused_ops[0], "BiasAdd");                                 \
          EXPECT_EQ(fused_ops[1], activation);                                \
          found++;                                                            \
        }                                                                     \
      }                                                                       \
      EXPECT_EQ(found, 1);                                                    \
                                                                              \
      auto tensors_expected =                                                 \
          EvaluateNodes(item.graph, item.fetch, item.feed);                   \
      ASSERT_EQ(tensors_expected.size(), 1);                                  \
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);            \
      ASSERT_EQ(tensors.size(), 1);                                           \
      test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);   \
    }                                                                         \
  }
REGISTER_TEST_ALL_TYPES(FuseDepthwiseConv2DWithBiasAndActivation);
#undef REGISTER_TEST

#ifdef ENABLE_MKLDNN_V1
TEST_F(MklRemapperTest, FuseBatchNormWithRelu) {
  using ::tensorflow::ops::Placeholder;

  for (bool is_training : {true, false}) {
    for (bool has_side_input : {true, false}) {
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();

      const int num_channels = 24;

      TensorShape channel_shape({num_channels});
      TensorShape empty_shape({0});

      auto input =
          Placeholder(s.WithOpName("input"), DT_FLOAT,
                      ops::Placeholder::Shape({2, 8, 8, num_channels}));
      auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_FLOAT);
      auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT);
      auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT);
      auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT);
      auto var = Placeholder(s.WithOpName("var"), DT_FLOAT);

      float epsilon = 0.1f;
      auto fbn =
          ops::FusedBatchNormV3(s.WithOpName("fused_batch_norm"), input_cast,
                                scale, offset, mean, var,
                                ops::FusedBatchNormV3::IsTraining(is_training)
                                    .Epsilon(epsilon)
                                    .DataFormat("NHWC"));

      if (has_side_input) {
        auto side_input =
            Placeholder(s.WithOpName("side_input"), DT_FLOAT,
                        ops::Placeholder::Shape({2, 8, 8, num_channels}));
        auto side_input_cast =
            ops::Cast(s.WithOpName("side_input_cast"), side_input, DT_FLOAT);
        auto add = ops::Add(s.WithOpName("add"), fbn.y, side_input_cast);
        auto relu = ops::Relu(s.WithOpName("relu"), add);
      } else {
        auto relu = ops::Relu(s.WithOpName("relu"), fbn.y);
      }

      auto input_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});
      auto scale_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
      auto offset_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
      auto mean_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                               : channel_shape);
      auto var_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                              : channel_shape);
      auto side_input_t =
          GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});

      GrapplerItem item;
      item.fetch = {"relu"};
      if (has_side_input)
        item.feed = {{"input", input_t},   {"scale", scale_t},
                     {"offset", offset_t}, {"mean", mean_t},
                     {"var", var_t},       {"side_input", side_input_t}};
      else
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

      Remapper optimizer(RewriterConfig::AGGRESSIVE);
      GraphDef output;
      TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

      int found = 0;
      if (has_side_input) {
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
      } else {
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

      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
    }
  }
}
#endif  // ENABLE_MKLDNN_V1

}  // namespace grappler
}  // namespace tensorflow
#endif  // INTEL_MKL
