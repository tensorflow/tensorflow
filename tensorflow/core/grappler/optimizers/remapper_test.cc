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

#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/util.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace grappler {

class RemapperTest : public GrapplerTest {
 protected:
  void SetUp() override {
    // This is a requirement for fusing FusedBatchNorm + SideInput + Activation.
    setenv("TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT", "1", 1 /* replace */);
    // This is a requirement for fusing FusedMatmul + BiasAdd (+ Activation).
    setenv("TF_USE_CUBLASLT", "1", 1 /* replace */);
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
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"batch_norm"};

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(RemapperTest, FusedBatchNormNCHW) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !GOOGLE_CUDA || TENSORFLOW_USE_ROCM
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
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"batch_norm"};

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;

  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  if (GetNumAvailableGPUs() > 0) {
    // NCHW batch norm is only supported on GPU.
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch);
    ASSERT_EQ(tensors.size(), 1);
    test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-3);
  }
}

TEST_F(RemapperTest, FuseBatchNormWithRelu) {
  using ::tensorflow::ops::Placeholder;

  for (bool is_training : {true, false}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

#if !defined(GOOGLE_CUDA) || !(CUDNN_VERSION >= 7402)
    if (is_training) {
      LOG(INFO) << "Skip FuseBatchNormWithRelu"
                << "[is_training=" << is_training << "] "
                << "test. It requires CUDNN_VERSION >= 7402.";
      continue;
    }
#endif

#if !defined(GOOGLE_CUDA)
    if (!is_training) {
      LOG(INFO) << "Skip FuseBatchNormWithRelu"
                << "[is_training=" << is_training << "]";
      continue;
    }
#endif

    const int num_channels = 24;

    TensorShape channel_shape({num_channels});
    TensorShape empty_shape({0});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT,
                             ops::Placeholder::Shape({2, 8, 8, num_channels}));
    auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_HALF);
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

    // Place all nodes on GPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:GPU:0");
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

    if (GetNumAvailableGPUs() > 0) {
      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, /*rtol=*/1e-2);
    }
  }
}

#if defined(GOOGLE_CUDA) && CUDNN_VERSION >= 7402
TEST_F(RemapperTest, FuseBatchNormGradWithReluGrad) {
  if (IsMKLEnabled()) GTEST_SKIP() << "Fusion not available with oneDNN.";
  using ::tensorflow::ops::Placeholder;
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  bool is_training = true;

  const int num_channels = 24;

  TensorShape channel_shape({num_channels});
  TensorShape empty_shape({0});

  // Forward pass.
  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT,
                           ops::Placeholder::Shape({2, 8, 8, num_channels}));
  auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_HALF);
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

  // Backward pass.
  auto output_grad =
      Placeholder(s.WithOpName("output_grad"), DT_FLOAT,
                  ops::Placeholder::Shape({2, 8, 8, num_channels}));
  auto output_grad_cast =
      ops::Cast(s.WithOpName("output_grad_cast"), output_grad, DT_HALF);
  auto relu_grad = ops::internal::ReluGrad(s.WithOpName("relu_grad"),
                                           output_grad_cast, relu);
  auto fbn_grad = ops::FusedBatchNormGradV3(
      s.WithOpName("fused_batch_norm_grad"), relu_grad, input_cast, scale,
      fbn.reserve_space_1, fbn.reserve_space_2, fbn.reserve_space_3,
      ops::FusedBatchNormGradV3::IsTraining(is_training)
          .Epsilon(epsilon)
          .DataFormat("NHWC"));
  auto fetch0 = ops::Identity(s.WithOpName("fetch0"), fbn_grad.x_backprop);
  auto fetch1 = ops::Identity(s.WithOpName("fetch1"), fbn_grad.scale_backprop);
  auto fetch2 = ops::Identity(s.WithOpName("fetch2"), fbn_grad.offset_backprop);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});
  auto scale_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
  auto offset_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
  auto mean_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
  auto var_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
  auto output_grad_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});

  GrapplerItem item;
  item.fetch = {"fetch0", "fetch1", "fetch2"};
  item.feed = {{"input", input_t},   {"scale", scale_t},
               {"offset", offset_t}, {"mean", mean_t},
               {"var", var_t},       {"output_grad", output_grad_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on GPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:GPU:0");
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

    if (node.name() == "fused_batch_norm_grad") {
      EXPECT_EQ(node.op(), "_FusedBatchNormGradEx");
      ASSERT_EQ(node.input_size(), 8);
      EXPECT_EQ(node.input(0), "output_grad_cast");
      EXPECT_EQ(node.input(1), "input_cast");
      EXPECT_EQ(node.input(2), "scale");
      EXPECT_EQ(node.input(3), "fused_batch_norm:3");
      EXPECT_EQ(node.input(4), "fused_batch_norm:4");
      EXPECT_EQ(node.input(5), "fused_batch_norm:5");
      EXPECT_EQ(node.input(6), "offset");
      EXPECT_EQ(node.input(7), "relu");

      auto attr = node.attr();
      EXPECT_EQ(attr["num_side_inputs"].i(), 0);
      EXPECT_EQ(attr["activation_mode"].s(), "Relu");
      found++;
    }
  }
  EXPECT_EQ(found, 3);

  if (GetNumAvailableGPUs() > 0) {
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 3);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 3);
    test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, /*rtol=*/1e-2);
    test::ExpectClose(tensors[1], tensors_expected[1], 1e-2, /*rtol=*/1e-2);
    test::ExpectClose(tensors[2], tensors_expected[2], 1e-2, /*rtol=*/1e-2);
  }
}
#endif  // defined(GOOGLE_CUDA) && CUDNN_VERSION >= 7402

TEST_F(RemapperTest, FuseBatchNormWithAddAndRelu) {
  using ::tensorflow::ops::Placeholder;

  for (bool is_training : {true, false}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

#if !defined(GOOGLE_CUDA) || !(CUDNN_VERSION >= 7402)
    if (is_training) {
      LOG(INFO) << "Skip FuseBatchNormWithAddAndRelu"
                << "[is_training=" << is_training << "] "
                << "test. It requires CUDNN_VERSION >= 7402.";
      continue;
    }
#endif

#if !defined(GOOGLE_CUDA)
    if (!is_training) {
      LOG(INFO) << "Skip FuseBatchNormWithAddAndRelu"
                << "[is_training=" << is_training << "]";
      continue;
    }
#endif

    const int num_channels = 24;

    TensorShape input_shape({2, 8, 8, num_channels});
    TensorShape channel_shape({num_channels});
    TensorShape empty_shape({0});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT,
                             ops::Placeholder::Shape(input_shape));
    auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_HALF);
    auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT);
    auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT);
    auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT);
    auto var = Placeholder(s.WithOpName("var"), DT_FLOAT);
    auto side_input = Placeholder(s.WithOpName("side_input"), DT_FLOAT,
                                  ops::Placeholder::Shape(input_shape));
    auto side_input_cast =
        ops::Cast(s.WithOpName("side_input_cast"), side_input, DT_HALF);

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

    // Place all nodes on GPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:GPU:0");
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
        ASSERT_EQ(node.input_size(), 6);
        EXPECT_EQ(node.input(0), "input_cast");
        EXPECT_EQ(node.input(1), "scale");
        EXPECT_EQ(node.input(2), "offset");
        EXPECT_EQ(node.input(3), "mean");
        EXPECT_EQ(node.input(4), "var");
        EXPECT_EQ(node.input(5), "side_input_cast");

        auto attr = node.attr();
        EXPECT_EQ(attr["num_side_inputs"].i(), 1);
        EXPECT_EQ(attr["activation_mode"].s(), "Relu");
        found++;
      }
    }
    EXPECT_EQ(found, 2);

    if (GetNumAvailableGPUs() > 0) {
      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, /*rtol=*/1e-2);
    }
  }
}

#if defined(GOOGLE_CUDA) && CUDNN_VERSION >= 7402
TEST_F(RemapperTest, FuseBatchNormGradWithAddAndReluGrad) {
  if (IsMKLEnabled()) GTEST_SKIP() << "Fusion not available with oneDNN.";
  using ::tensorflow::ops::Placeholder;
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  bool is_training = true;

  const int num_channels = 24;

  TensorShape input_shape({2, 8, 8, num_channels});
  TensorShape channel_shape({num_channels});
  TensorShape empty_shape({0});

  // Forward pass.
  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT,
                           ops::Placeholder::Shape(input_shape));
  auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_HALF);
  auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT);
  auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT);
  auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT);
  auto var = Placeholder(s.WithOpName("var"), DT_FLOAT);
  auto side_input = Placeholder(s.WithOpName("side_input"), DT_FLOAT,
                                ops::Placeholder::Shape(input_shape));
  auto side_input_cast =
      ops::Cast(s.WithOpName("side_input_cast"), side_input, DT_HALF);

  float epsilon = 0.1f;
  auto fbn = ops::FusedBatchNormV3(
      s.WithOpName("fused_batch_norm"), input_cast, scale, offset, mean, var,
      ops::FusedBatchNormV3::IsTraining(is_training)
          .Epsilon(epsilon)
          .DataFormat("NHWC"));
  auto fbn_side_input =
      ops::FusedBatchNormV3(s.WithOpName("fused_batch_norm_side_input"),
                            side_input_cast, scale, offset, mean, var,
                            ops::FusedBatchNormV3::IsTraining(is_training)
                                .Epsilon(epsilon)
                                .DataFormat("NHWC"));
  // Since fbn.y is the first argument of "add" op, "fused_batch_norm" will be
  // fused (not "fused_batch_norm_side_input"). Correspondingly,
  // "fused_batch_norm_grad" will be fused (not
  // "fused_batch_norm_side_input_grad").
  auto add = ops::Add(s.WithOpName("add"), fbn.y, fbn_side_input.y);
  auto relu = ops::Relu(s.WithOpName("relu"), add);

  // Backward pass.
  auto output_grad =
      Placeholder(s.WithOpName("output_grad"), DT_FLOAT,
                  ops::Placeholder::Shape({2, 8, 8, num_channels}));
  auto output_grad_cast =
      ops::Cast(s.WithOpName("output_grad_cast"), output_grad, DT_HALF);
  auto relu_grad = ops::internal::ReluGrad(s.WithOpName("relu_grad"),
                                           output_grad_cast, relu);
  auto fbn_grad = ops::FusedBatchNormGradV3(
      s.WithOpName("fused_batch_norm_grad"), relu_grad, input_cast, scale,
      fbn.reserve_space_1, fbn.reserve_space_2, fbn.reserve_space_3,
      ops::FusedBatchNormGradV3::IsTraining(is_training)
          .Epsilon(epsilon)
          .DataFormat("NHWC"));
  auto fbn_side_input_grad = ops::FusedBatchNormGradV3(
      s.WithOpName("fused_batch_norm_side_input_grad"), relu_grad,
      side_input_cast, scale, fbn_side_input.reserve_space_1,
      fbn_side_input.reserve_space_2, fbn_side_input.reserve_space_3,
      ops::FusedBatchNormGradV3::IsTraining(is_training)
          .Epsilon(epsilon)
          .DataFormat("NHWC"));
  auto fetch0 = ops::Identity(s.WithOpName("fetch0"), fbn_grad.x_backprop);
  auto fetch1 = ops::Identity(s.WithOpName("fetch1"), fbn_grad.scale_backprop);
  auto fetch2 = ops::Identity(s.WithOpName("fetch2"), fbn_grad.offset_backprop);
  auto fetch3 =
      ops::Identity(s.WithOpName("fetch3"), fbn_side_input_grad.x_backprop);
  auto fetch4 =
      ops::Identity(s.WithOpName("fetch4"), fbn_side_input_grad.scale_backprop);
  auto fetch5 = ops::Identity(s.WithOpName("fetch5"),
                              fbn_side_input_grad.offset_backprop);

  auto input_t = GenerateRandomTensor<DT_FLOAT>(input_shape);
  auto scale_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
  auto offset_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
  auto mean_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
  auto var_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
  auto side_input_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});
  auto output_grad_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});

  GrapplerItem item;
  item.fetch = {"fetch0", "fetch1", "fetch2", "fetch3", "fetch4", "fetch5"};
  item.feed = {{"input", input_t},
               {"scale", scale_t},
               {"offset", offset_t},
               {"mean", mean_t},
               {"var", var_t},
               {"side_input", side_input_t},
               {"output_grad", output_grad_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on GPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:GPU:0");
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
      ASSERT_EQ(node.input_size(), 6);
      EXPECT_EQ(node.input(0), "input_cast");
      EXPECT_EQ(node.input(1), "scale");
      EXPECT_EQ(node.input(2), "offset");
      EXPECT_EQ(node.input(3), "mean");
      EXPECT_EQ(node.input(4), "var");
      EXPECT_EQ(node.input(5), "fused_batch_norm_side_input");

      auto attr = node.attr();
      EXPECT_EQ(attr["num_side_inputs"].i(), 1);
      EXPECT_EQ(attr["activation_mode"].s(), "Relu");
      found++;
    }

    if (node.name() == "relu_grad") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "fused_batch_norm_grad:5");
      found++;
    }

    if (node.name() == "fused_batch_norm_grad") {
      EXPECT_EQ(node.op(), "_FusedBatchNormGradEx");
      ASSERT_EQ(node.input_size(), 8);
      EXPECT_EQ(node.input(0), "output_grad_cast");
      EXPECT_EQ(node.input(1), "input_cast");
      EXPECT_EQ(node.input(2), "scale");
      EXPECT_EQ(node.input(3), "fused_batch_norm:3");
      EXPECT_EQ(node.input(4), "fused_batch_norm:4");
      EXPECT_EQ(node.input(5), "fused_batch_norm:5");
      EXPECT_EQ(node.input(6), "offset");
      EXPECT_EQ(node.input(7), "relu");

      auto attr = node.attr();
      EXPECT_EQ(attr["num_side_inputs"].i(), 1);
      EXPECT_EQ(attr["activation_mode"].s(), "Relu");
      found++;
    }
  }
  EXPECT_EQ(found, 4);

  if (GetNumAvailableGPUs() > 0) {
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 6);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 6);
    test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, /*rtol=*/1e-2);
    test::ExpectClose(tensors[1], tensors_expected[1], 1e-2, /*rtol=*/1e-2);
    test::ExpectClose(tensors[2], tensors_expected[2], 1e-2, /*rtol=*/1e-2);
    test::ExpectClose(tensors[3], tensors_expected[3], 1e-2, /*rtol=*/1e-2);
    test::ExpectClose(tensors[4], tensors_expected[4], 1e-2, /*rtol=*/1e-2);
    test::ExpectClose(tensors[5], tensors_expected[5], 1e-2, /*rtol=*/1e-2);
  }
}
#endif  // defined(GOOGLE_CUDA) && CUDNN_VERSION >= 7402

class RemapperFuseConvWithBias : public RemapperTest {
 public:
  template <int dim, DataType DTYPE>
  void RunTest() {
    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
    auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
    auto bias_shape = ops::Placeholder::Shape({128});
    std::vector<int> strides = {1, 1, 1, 1};

    auto input_t = GenerateTensorWithSetRandom<DTYPE>({8, 32, 32, 3});
    auto filter_t = GenerateTensorWithSetRandom<DTYPE>({1, 1, 3, 128});
    auto bias_t = GenerateTensorWithSetRandom<DTYPE>({128});

    if (dim == 3) {
      if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to oneDNN.";
      input_shape = ops::Placeholder::Shape({8, 4, 32, 32, 3});
      filter_shape = ops::Placeholder::Shape({1, 1, 1, 3, 128});
      bias_shape = ops::Placeholder::Shape({128});
      strides = {1, 1, 1, 1, 1};

      input_t = GenerateTensorWithSetRandom<DTYPE>({8, 4, 32, 32, 3});
      filter_t = GenerateTensorWithSetRandom<DTYPE>({1, 1, 1, 3, 128});
      bias_t = GenerateTensorWithSetRandom<DTYPE>({128});
    }

    auto input = Placeholder(s.WithOpName("input"), DTYPE, input_shape);
    auto filter = Placeholder(s.WithOpName("filter"), DTYPE, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);
    if (dim == 2) {
      auto conv =
          ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
      auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);
      auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);
    } else if (dim == 3) {
      auto conv =
          ops::Conv3D(s.WithOpName("conv"), input, filter, strides, "SAME");
      auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);
      auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);
    }

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "bias_add") {
        if (dim == 2) {
          EXPECT_EQ(node.op(), "_FusedConv2D");
        } else if (dim == 3) {
          EXPECT_EQ(node.op(), "_FusedConv3D");
        }
        ASSERT_GE(node.input_size(), 3);
        EXPECT_EQ(node.input(0), "input");
        EXPECT_EQ(node.input(1), "filter");

        EXPECT_EQ(node.attr().at("num_args").i(), 1);
        EXPECT_EQ(node.input(2), "bias");

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(fused_ops.size(), 1);
        EXPECT_EQ(fused_ops[0], "BiasAdd");
        found++;
      }
    }
    EXPECT_EQ(found, 1);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    if (DTYPE == DT_BFLOAT16)
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
    else
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
  }
};

TEST_F(RemapperFuseConvWithBias, Conv2D_F32) { RunTest<2, DT_FLOAT>(); }
TEST_F(RemapperFuseConvWithBias, Conv3D_F32) { RunTest<3, DT_FLOAT>(); }
TEST_F(RemapperFuseConvWithBias, Conv2D_BF16) {
  if (!IsMKLEnabled())
    GTEST_SKIP() << "Intel MKL with bfloat16 support is not enabled, skipping "
                    "FuseConv2DWithBias with bfloat16.";
  RunTest<2, DT_BFLOAT16>();
}
TEST_F(RemapperFuseConvWithBias, Conv3D_BF16) {
  if (!IsMKLEnabled())
    GTEST_SKIP() << "Intel MKL with bfloat16 support is not enabled, skipping "
                    "FuseConv3DWithBias with bfloat16.";
  RunTest<3, DT_BFLOAT16>();
}

class RemapperFuseConvWithBiasAndActivation : public RemapperTest {
 public:
  template <int dim, DataType DTYPE>
  void RunTest() {
    using ::tensorflow::ops::Placeholder;

    for (const string& activation : {"Relu", "Relu6", "Elu", "LeakyRelu"}) {
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();
      auto input_shape = Placeholder::Shape({8, 32, 32, 3});
      auto filter_shape = Placeholder::Shape({1, 1, 3, 128});
      auto bias_shape = Placeholder::Shape({128});
      std::vector<int> strides = {1, 1, 1, 1};

      auto input_t = GenerateTensorWithSetRandom<DTYPE>({8, 32, 32, 3});
      auto filter_t = GenerateTensorWithSetRandom<DTYPE>({1, 1, 3, 128});
      auto bias_t = GenerateTensorWithSetRandom<DTYPE>({128});

      if (dim == 3) {
        if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to oneDNN.";
        input_shape = Placeholder::Shape({8, 4, 32, 32, 3});
        filter_shape = Placeholder::Shape({1, 1, 1, 3, 128});
        bias_shape = Placeholder::Shape({128});
        strides = {1, 1, 1, 1, 1};

        input_t = GenerateTensorWithSetRandom<DTYPE>({8, 4, 32, 32, 3});
        filter_t = GenerateTensorWithSetRandom<DTYPE>({1, 1, 1, 3, 128});
        bias_t = GenerateTensorWithSetRandom<DTYPE>({128});
      }

      float leakyrelu_alpha = 0.5;

      auto input = Placeholder(s.WithOpName("input"), DTYPE, input_shape);
      auto filter = Placeholder(s.WithOpName("filter"), DTYPE, filter_shape);
      auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

      if (dim == 2) {
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
          } else if (activation == "LeakyRelu") {
            auto attr = ops::internal::LeakyRelu::Alpha(leakyrelu_alpha);
            return ops::Identity(
                fetch, ops::internal::LeakyRelu(activate, bias_add, attr));
          }

          return ops::Identity(fetch, bias);
        }();
      } else if (dim == 3) {
        auto conv =
            ops::Conv3D(s.WithOpName("conv"), input, filter, strides, "SAME");
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
          } else if (activation == "LeakyRelu") {
            auto attr = ops::internal::LeakyRelu::Alpha(leakyrelu_alpha);
            return ops::Identity(
                fetch, ops::internal::LeakyRelu(activate, bias_add, attr));
          }

          return ops::Identity(fetch, bias);
        }();
      }

      GrapplerItem item;
      item.fetch = {"fetch"};
      item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
      TF_ASSERT_OK(s.ToGraphDef(&item.graph));

      // Place all nodes on CPU.
      for (int i = 0; i < item.graph.node_size(); ++i) {
        item.graph.mutable_node(i)->set_device("/device:CPU:0");
      }

      Remapper optimizer(RewriterConfig::ON);
      GraphDef output;
      TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

      int found = 0;
      for (const NodeDef& node : output.node()) {
        if (node.name() == "activation") {
          if (dim == 2) {
            EXPECT_EQ(node.op(), "_FusedConv2D");
          } else if (dim == 3) {
            EXPECT_EQ(node.op(), "_FusedConv3D");
          }
          ASSERT_GE(node.input_size(), 3);
          EXPECT_EQ(node.input(0), "input");
          EXPECT_EQ(node.input(1), "filter");

          EXPECT_EQ(node.attr().at("num_args").i(), 1);
          EXPECT_EQ(node.input(2), "bias");

          const auto fused_ops = node.attr().at("fused_ops").list().s();
          ASSERT_EQ(fused_ops.size(), 2);
          EXPECT_EQ(fused_ops[0], "BiasAdd");
          EXPECT_EQ(fused_ops[1], activation);

          if (activation == "LeakyRelu") {
            EXPECT_EQ(node.attr().at("leakyrelu_alpha").f(), leakyrelu_alpha);
          }
          found++;
        }
      }
      EXPECT_EQ(found, 1);

      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      if (DTYPE == DT_BFLOAT16)
        test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
      else
        test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
    }
  }
};

TEST_F(RemapperFuseConvWithBiasAndActivation, Conv2D_F32) {
  RunTest<2, DT_FLOAT>();
}
TEST_F(RemapperFuseConvWithBiasAndActivation, Conv3D_F32) {
  RunTest<3, DT_FLOAT>();
}
TEST_F(RemapperFuseConvWithBiasAndActivation, Conv2D_BF16) {
  if (!IsMKLEnabled())
    GTEST_SKIP() << "Intel MKL with bfloat16 support is not enabled, skipping "
                    "FuseConv2DWithBiasAndActivation with bfloat16.";
  RunTest<2, DT_BFLOAT16>();
}
TEST_F(RemapperFuseConvWithBiasAndActivation, Conv3D_BF16) {
  if (!IsMKLEnabled())
    GTEST_SKIP() << "Intel MKL with bfloat16 support is not enabled, skipping "
                    "FuseConv3DWithBiasAndActivation with bfloat16.";
  RunTest<3, DT_BFLOAT16>();
}

class RemapperFuseConvWithSqueezeAndBias : public RemapperTest {
 public:
  template <int dim, DataType DTYPE>
  void RunTest() {
    using ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    auto input_shape = ops::Placeholder::Shape({8, 32, 1, 3});
    auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
    auto bias_shape = ops::Placeholder::Shape({128});
    std::vector<int> strides = {1, 1, 1, 1};

    auto input_t = GenerateTensorWithSetRandom<DTYPE>({8, 32, 1, 3});
    auto filter_t = GenerateTensorWithSetRandom<DTYPE>({1, 1, 3, 128});
    auto bias_t = GenerateTensorWithSetRandom<DTYPE>({128});

    if (dim == 3) {
      if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to oneDNN.";
      input_shape = ops::Placeholder::Shape({8, 4, 32, 1, 3});
      filter_shape = ops::Placeholder::Shape({1, 1, 1, 3, 128});
      bias_shape = ops::Placeholder::Shape({128});
      strides = {1, 1, 1, 1, 1};

      input_t = GenerateTensorWithSetRandom<DTYPE>({8, 4, 32, 1, 3});
      filter_t = GenerateTensorWithSetRandom<DTYPE>({1, 1, 1, 3, 128});
      bias_t = GenerateTensorWithSetRandom<DTYPE>({128});
    }

    auto input = Placeholder(s.WithOpName("input"), DTYPE, input_shape);
    auto filter = Placeholder(s.WithOpName("filter"), DTYPE, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

    if (dim == 2) {
      auto conv =
          ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");

      auto squeeze = ops::Squeeze(s.WithOpName("squeeze"), conv,
                                  ops::Squeeze::Attrs().Axis({2}));

      auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), squeeze, bias);
      auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);
    } else if (dim == 3) {
      auto conv =
          ops::Conv3D(s.WithOpName("conv"), input, filter, strides, "SAME");

      auto squeeze = ops::Squeeze(s.WithOpName("squeeze"), conv,
                                  ops::Squeeze::Attrs().Axis({3}));

      auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), squeeze, bias);
      auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);
    }

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "conv") {
        if (dim == 2) {
          EXPECT_EQ(node.op(), "_FusedConv2D");
        } else if (dim == 3) {
          EXPECT_EQ(node.op(), "_FusedConv3D");
        }
        ASSERT_GE(node.input_size(), 3);
        EXPECT_EQ(node.input(0), "input");
        EXPECT_EQ(node.input(1), "filter");

        EXPECT_EQ(node.attr().at("num_args").i(), 1);
        EXPECT_EQ(node.input(2), "bias");

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(fused_ops.size(), 1);
        EXPECT_EQ(fused_ops[0], "BiasAdd");
        found++;
      } else if (node.name() == "bias_add") {
        EXPECT_EQ(node.op(), "Squeeze");
        ASSERT_GE(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "conv");
        found++;
      }
    }
    EXPECT_EQ(found, 2);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    if (DTYPE == DT_BFLOAT16)
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
    else
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
  }
};

TEST_F(RemapperFuseConvWithSqueezeAndBias, Conv2D_FP32) {
  RunTest<2, DT_FLOAT>();
}
TEST_F(RemapperFuseConvWithSqueezeAndBias, Conv3D_FP32) {
  RunTest<3, DT_FLOAT>();
}
TEST_F(RemapperFuseConvWithSqueezeAndBias, Conv2D_BF16) {
  if (!IsMKLEnabled())
    GTEST_SKIP() << "Intel MKL with bfloat16 support is not enabled, skipping "
                    "FuseConvWithSqueezeAndBias with bfloat16.";
  RunTest<2, DT_BFLOAT16>();
}
TEST_F(RemapperFuseConvWithSqueezeAndBias, Conv3D_BF16) {
  if (!IsMKLEnabled())
    GTEST_SKIP() << "Intel MKL with bfloat16 support is not enabled, skipping "
                    "FuseConvWithSqueezeAndBias with bfloat16.";
  RunTest<3, DT_BFLOAT16>();
}

#ifdef INTEL_MKL
TEST_F(RemapperTest, FuseConv3DWithBias) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to MKL.";
  using ::tensorflow::ops::Placeholder;
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 4, 32, 32, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 1, 3, 6});
  auto add_shape = ops::Placeholder::Shape({6});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);

  std::vector<int> strides = {1, 1, 1, 1, 1};
  auto conv =
      ops::Conv3D(s.WithOpName("conv"), input, filter, strides, "VALID");
  auto add_const = ops::Const(s.WithOpName("add_const"),
                              {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {6});
  auto add = ops::Add(s.WithOpName("b_add"), add_const, conv);
  auto fetch = ops::Identity(s.WithOpName("fetch"), add);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 4, 32, 32, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 1, 3, 6});
  auto add_t = GenerateRandomTensor<DT_FLOAT>({6});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}, {"filter", filter_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "b_add") {
      EXPECT_EQ(node.op(), "_FusedConv3D");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 1);
      EXPECT_EQ(node.input(2), "add_const");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      found++;
    }
  }
  EXPECT_EQ(found, 1);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

// Fuse conv3d + add {1,1,1,1,C}
TEST_F(RemapperTest, FuseConv3DWithAdd) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to MKL.";
  using ::tensorflow::ops::Placeholder;
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 4, 32, 32, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 1, 3, 6});
  auto add_shape = ops::Placeholder::Shape({1, 1, 1, 1, 6});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto a_placeholder =
      Placeholder(s.WithOpName("add_placeholder"), DT_FLOAT, add_shape);

  std::vector<int> strides = {1, 1, 1, 1, 1};
  auto conv =
      ops::Conv3D(s.WithOpName("conv"), input, filter, strides, "VALID");
  auto add_const = ops::Const(s.WithOpName("add_const"), 1.0f, {1, 1, 1, 1, 6});
  auto add = ops::Add(s.WithOpName("add"), add_const, conv);
  auto fetch = ops::Identity(s.WithOpName("fetch"), add);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 4, 32, 32, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 1, 3, 6});
  auto add_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 1, 1, 6});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}, {"filter", filter_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "add") {
      EXPECT_EQ(node.op(), "_FusedConv3D");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 1);
      EXPECT_EQ(node.input(2), "add_const");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      found++;
    }
  }
  EXPECT_EQ(found, 1);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

// Fuse conv2d + add {1,1,C}
TEST_F(RemapperTest, FuseConv2DWithAdd) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to MKL.";
  using ::tensorflow::ops::Placeholder;
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 6});
  auto add_shape = ops::Placeholder::Shape({1, 1, 6});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto a_placeholder =
      Placeholder(s.WithOpName("add_placeholder"), DT_FLOAT, add_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv =
      ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "VALID");
  auto add_const = ops::Const(s.WithOpName("add_const"), 1.0f, {1, 1, 6});
  auto add = ops::Add(s.WithOpName("add"), add_const, conv);
  auto fetch = ops::Identity(s.WithOpName("fetch"), add);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 6});
  auto add_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 6});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}, {"filter", filter_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "add") {
      EXPECT_EQ(node.op(), "_FusedConv2D");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 1);
      EXPECT_EQ(node.input(2), "add_const");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      found++;
    }
  }
  EXPECT_EQ(found, 1);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

// Fuse  matmul + add {1,C}
TEST_F(RemapperTest, FuseMatmulWithAdd) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to MKL.";

  using ::tensorflow::ops::Placeholder;
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto lhs_shape = ops::Placeholder::Shape({8, 32});
  auto rhs_shape = ops::Placeholder::Shape({32, 64});

  auto lhs = Placeholder(s.WithOpName("lhs"), DT_FLOAT, lhs_shape);
  auto rhs = Placeholder(s.WithOpName("rhs"), DT_FLOAT, rhs_shape);

  auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
  auto add_const = ops::Const(s.WithOpName("add_const"), 1.0f, {1, 64});
  auto add = ops::Add(s.WithOpName("add"), matmul, add_const);
  auto fetch = ops::Identity(s.WithOpName("fetch"), add);

  auto lhs_t = GenerateTensorWithSetRandom<DT_FLOAT>({8, 32});
  auto rhs_t = GenerateTensorWithSetRandom<DT_FLOAT>({32, 64});
  auto add_t = GenerateTensorWithSetRandom<DT_FLOAT>({1, 64});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "add") {
      EXPECT_EQ(node.op(), "_FusedMatMul");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "lhs");
      EXPECT_EQ(node.input(1), "rhs");

      EXPECT_EQ(node.attr().at("num_args").i(), 1);
      EXPECT_EQ(node.input(2), "add_const");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      found++;
    }
  }
  EXPECT_EQ(1, found);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
}

class RemapperFuseSoftplusTanhMul : public RemapperTest {
 public:
  template <DataType DTYPE>
  void RunTest() {
    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
    auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
    auto bias_shape = ops::Placeholder::Shape({128});

    auto input = Placeholder(s.WithOpName("input"), DTYPE, input_shape);
    auto filter = Placeholder(s.WithOpName("filter"), DTYPE, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

    std::vector<int> strides = {1, 1, 1, 1};
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);
    auto softplus = ops::Softplus(s.WithOpName("softplus"), bias_add);
    auto tanh = ops::Tanh(s.WithOpName("tanh"), softplus);
    auto mul = ops::Mul(s.WithOpName("mul"), tanh, bias_add);
    auto fetch = ops::Identity(s.WithOpName("fetch"), mul);

    auto input_t = GenerateTensorWithSetRandom<DTYPE>({8, 32, 32, 3});
    auto filter_t = GenerateTensorWithSetRandom<DTYPE>({1, 1, 3, 128});
    auto bias_t = GenerateTensorWithSetRandom<DTYPE>({128});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "mul") {
        EXPECT_EQ(node.op(), "_MklFusedMish");
        ASSERT_EQ(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "bias_add");
        found++;
      }
    }
    EXPECT_EQ(found, 1);
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    if (DTYPE == DT_BFLOAT16) {
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
    } else {
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
    }
  }
};

TEST_F(RemapperFuseSoftplusTanhMul, FP32) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to MKL.";
  RunTest<DT_FLOAT>();
}
TEST_F(RemapperFuseSoftplusTanhMul, BF16) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to MKL.";
  RunTest<DT_BFLOAT16>();
}
#endif

class RemapperTensorToHashBucketTest : public RemapperTest {
 public:
  template <DataType DTYPE>
  void RunTest() {
    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
    auto input = Placeholder(s.WithOpName("input"), DTYPE, input_shape);

    int num_buckets = 100;
    auto to_string = ops::AsString(s.WithOpName("to_string"), input);
    auto to_bucket = ops::StringToHashBucketFast(s.WithOpName("to_bucket"),
                                                 to_string, num_buckets);
    auto fetch = ops::Identity(s.WithOpName("fetch"), to_bucket);

    auto input_t = GenerateRandomTensor<DTYPE>({8, 32, 32, 3});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // For CPU tests, we place all nodes on CPU. For GPU tests, we place the
    // "input" node on GPU to determine the fused op to be on GPU.
    const string input_device =
        GetNumAvailableGPUs() > 0 ? "/device:GPU:0" : "/device:CPU:0";
    for (int i = 0; i < item.graph.node_size(); ++i) {
      if (item.graph.node(i).name() == "input") {
        item.graph.mutable_node(i)->set_device(input_device);
      } else {
        item.graph.mutable_node(i)->set_device("/device:CPU:0");
      }
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "to_bucket") {
        EXPECT_EQ(node.op(), "_TensorToHashBucketFast");
        ASSERT_GE(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "input");
        EXPECT_EQ(node.attr().at("num_buckets").i(), num_buckets);
        found++;
      }
    }
    EXPECT_EQ(found, 1);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    test::ExpectTensorEqual<int64_t>(tensors[0], tensors_expected[0]);
  }
};

TEST_F(RemapperTensorToHashBucketTest, I8) { RunTest<DT_INT8>(); }

TEST_F(RemapperTensorToHashBucketTest, I16) { RunTest<DT_INT16>(); }

TEST_F(RemapperTensorToHashBucketTest, I32) { RunTest<DT_INT32>(); }

TEST_F(RemapperTensorToHashBucketTest, I64) { RunTest<DT_INT64>(); }

class RemapperFuseMatMulWithBiasTest : public RemapperTest {
 public:
  template <DataType DTYPE>
  void RunTest() {
    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto lhs_shape = ops::Placeholder::Shape({8, 32});
    auto rhs_shape = ops::Placeholder::Shape({32, 64});
    auto bias_shape = ops::Placeholder::Shape({64});

    auto lhs = Placeholder(s.WithOpName("lhs"), DTYPE, lhs_shape);
    auto rhs = Placeholder(s.WithOpName("rhs"), DTYPE, rhs_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

    auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);
    auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);

    auto lhs_t = GenerateTensorWithSetRandom<DTYPE>({8, 32});
    auto rhs_t = GenerateTensorWithSetRandom<DTYPE>({32, 64});
    auto bias_t = GenerateTensorWithSetRandom<DTYPE>({64});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}, {"bias", bias_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    const string device =
        GetNumAvailableGPUs() > 0 && (DTYPE == DT_HALF || DTYPE == DT_FLOAT)
            ? "/device:GPU:0"
            : "/device:CPU:0";

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device(device);
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "bias_add") {
        EXPECT_EQ(node.op(), "_FusedMatMul");
        ASSERT_GE(node.input_size(), 3);
        EXPECT_EQ(node.input(0), "lhs");
        EXPECT_EQ(node.input(1), "rhs");

        EXPECT_EQ(node.attr().at("num_args").i(), 1);
        EXPECT_EQ(node.input(2), "bias");

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(fused_ops.size(), 1);
        EXPECT_EQ(fused_ops[0], "BiasAdd");
        found++;
      }
    }
    EXPECT_EQ(1, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    if (DTYPE == DT_BFLOAT16 || DTYPE == DT_HALF)
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
    else
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
  }
};

TEST_F(RemapperFuseMatMulWithBiasTest, F16) {
  bool skip_test = false;
#if !defined(GOOGLE_CUDA)
  skip_test = true;
#endif
  if (skip_test || GetNumAvailableGPUs() == 0) {
    GTEST_SKIP() << "Skipping FuseMatMulWithBias with half, which is only "
                    "supported in CUDA.";
  }
  RunTest<DT_HALF>();
}

TEST_F(RemapperFuseMatMulWithBiasTest, F32) {
  bool skip_test = false;
#if !defined(GOOGLE_CUDA)
  skip_test = true;
#endif
  if (skip_test || GetNumAvailableGPUs() == 0) {
    GTEST_SKIP() << "Skipping FuseMatMulWithBias with float, which is only "
                    "supported in CUDA.";
  }
  RunTest<DT_FLOAT>();
}

TEST_F(RemapperFuseMatMulWithBiasTest, Bf16) {
#if !defined(ENABLE_MKL)
  GTEST_SKIP() << "Intel MKL with bfloat16 support is not enabled, skipping "
                  "FuseMatMulWithBias with bfloat16.";
#endif
  RunTest<DT_BFLOAT16>();  // NOLINT
}

// TODO(b/161005848): Fix flaky test.
TEST_F(RemapperTest, DISABLED_FuseConv2DWithBiasAndActivationOnGPU) {
#if !(GOOGLE_CUDA)
  GTEST_SKIP() << "No CUDA, skip FuseConv2DWithBiasAndActivation on GPU";
#endif  // !GOOGLE_CUDA
  using ::tensorflow::ops::Placeholder;
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = Placeholder::Shape({8, 32, 32, 3});
  auto filter_shape = Placeholder::Shape({3, 3, 3, 128});
  auto bias_shape = Placeholder::Shape({128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv = ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);

  ops::Identity fetch = [&]() -> ops::Identity {
    auto activate = s.WithOpName("activation");
    auto fetch = s.WithOpName("fetch");
    return ops::Identity(fetch, ops::Relu(activate, bias_add));
  }();

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({3, 3, 3, 128});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on GPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:GPU:0");
  }

  Remapper optimizer(RewriterConfig::AGGRESSIVE);  // trust placeholders shape
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "activation") {
      EXPECT_EQ(node.op(), "_FusedConv2D");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 1);
      EXPECT_EQ(node.input(2), "bias");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 2);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      EXPECT_EQ(fused_ops[1], "Relu");
      found++;
    }
  }
  EXPECT_EQ(found, 1);

  if (GetNumAvailableGPUs() > 0) {
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  }
}

class RemapperFuseMatMulWithBiasAndActivationTest : public RemapperTest {
 public:
  template <DataType DTYPE>
  void RunTest() {
    using ::tensorflow::ops::Placeholder;

#if defined(INTEL_MKL) && defined(ENABLE_MKL)
    std::vector<string> activations = {"Relu", "Relu6", "Elu", "Tanh",
                                       "LeakyRelu"};
#else
    std::vector<string> activations = {"Relu", "Relu6", "Elu", "LeakyRelu"};
#endif

    for (const string& activation : activations) {
      if (DTYPE == DT_HALF && activation != "Relu") continue;
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();

      auto lhs_shape = ops::Placeholder::Shape({8, 32});
      auto rhs_shape = ops::Placeholder::Shape({32, 64});
      auto bias_shape = ops::Placeholder::Shape({64});

      auto lhs = Placeholder(s.WithOpName("lhs"), DTYPE, lhs_shape);
      auto rhs = Placeholder(s.WithOpName("rhs"), DTYPE, rhs_shape);
      auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

      auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
      auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);

      float leakyrelu_alpha = 0.5;

      ops::Identity fetch = [&]() -> ops::Identity {
        auto activate = s.WithOpName("activation");
        auto fetch = s.WithOpName("fetch");

        if (activation == "Relu") {
          return ops::Identity(fetch, ops::Relu(activate, bias_add));
        } else if (activation == "Relu6") {
          return ops::Identity(fetch, ops::Relu6(activate, bias_add));
        } else if (activation == "Elu") {
          return ops::Identity(fetch, ops::Elu(activate, bias_add));
#if defined(INTEL_MKL) && defined(ENABLE_MKL)
        } else if (activation == "Tanh") {
          return ops::Identity(fetch, ops::Tanh(activate, bias_add));
#endif
        } else if (activation == "LeakyRelu") {
          auto attr = ops::internal::LeakyRelu::Alpha(leakyrelu_alpha);
          return ops::Identity(
              fetch, ops::internal::LeakyRelu(activate, bias_add, attr));
        }

        return ops::Identity(fetch, bias);
      }();

      auto lhs_t = GenerateTensorWithSetRandom<DTYPE>({8, 32});
      auto rhs_t = GenerateTensorWithSetRandom<DTYPE>({32, 64});
      auto bias_t = GenerateTensorWithSetRandom<DTYPE>({64});

      GrapplerItem item;
      item.fetch = {"fetch"};
      item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}, {"bias", bias_t}};
      TF_ASSERT_OK(s.ToGraphDef(&item.graph));

      const string device = GetNumAvailableGPUs() > 0 &&
                                    (DTYPE == DT_HALF || DTYPE == DT_FLOAT) &&
                                    activation == "Relu"
                                ? "/device:GPU:0"
                                : "/device:CPU:0";

      // Place all nodes on CPU.
      for (int i = 0; i < item.graph.node_size(); ++i) {
        item.graph.mutable_node(i)->set_device(device);
      }

      Remapper optimizer(RewriterConfig::ON);
      GraphDef output;
      TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

      int found = 0;
      for (const NodeDef& node : output.node()) {
        if (node.name() == "activation") {
          EXPECT_EQ(node.op(), "_FusedMatMul");
          ASSERT_GE(node.input_size(), 3);
          EXPECT_EQ(node.input(0), "lhs");
          EXPECT_EQ(node.input(1), "rhs");

          EXPECT_EQ(node.attr().at("num_args").i(), 1);
          EXPECT_EQ(node.input(2), "bias");

          const auto fused_ops = node.attr().at("fused_ops").list().s();
          ASSERT_EQ(fused_ops.size(), 2);
          EXPECT_EQ(fused_ops[0], "BiasAdd");
          EXPECT_EQ(fused_ops[1], activation);

          if (activation == "LeakyRelu") {
            EXPECT_EQ(node.attr().at("leakyrelu_alpha").f(), leakyrelu_alpha);
          }
          found++;
        }
      }
      EXPECT_EQ(1, found);

      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      if (DTYPE == DT_BFLOAT16 || DTYPE == DT_HALF)
        test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
      else
        test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
    }
  }
};

TEST_F(RemapperFuseMatMulWithBiasAndActivationTest, F16) {
  bool skip_test = false;
#if !defined(GOOGLE_CUDA)
  skip_test = true;
#endif
  if (skip_test || GetNumAvailableGPUs() == 0) {
    GTEST_SKIP() << "Skipping FuseMatMulWithBiasAndActivationTest with half, "
                    "which is only supported in CUDA.";
  }
  RunTest<DT_HALF>();
}

TEST_F(RemapperFuseMatMulWithBiasAndActivationTest, F32) {
  bool skip_test = false;
#if !defined(GOOGLE_CUDA)
  skip_test = true;
#endif
  if (skip_test || GetNumAvailableGPUs() == 0) {
    GTEST_SKIP() << "Skipping FuseMatMulWithBiasAndActivationTest with float, "
                    "which is only supported in CUDA.";
  }
  RunTest<DT_FLOAT>();
}

TEST_F(RemapperFuseMatMulWithBiasAndActivationTest, Bf16) {
#if !defined(ENABLE_MKL)
  GTEST_SKIP() << "Intel MKL with bfloat16 support is not enabled, skipping "
                  "FuseMatMulWithBiasAndActivation with bfloat16.";
#endif
  RunTest<DT_BFLOAT16>();  // NOLINT
}

TEST_F(RemapperTest, FuseConv2DWithBatchNorm) {
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
  auto conv = ops::Conv2D(
      s.WithOpName("conv"), input, filter, strides, "EXPLICIT",
      ops::Conv2D::Attrs().ExplicitPaddings({0, 0, 1, 2, 3, 4, 0, 0}));
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
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "batch_norm") {
      EXPECT_EQ(node.op(), "_FusedConv2D");
      ASSERT_GE(node.input_size(), 6);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 4);
      EXPECT_EQ(node.input(2), "scale");
      EXPECT_EQ(node.input(3), "offset");
      EXPECT_EQ(node.input(4), "mean");
      EXPECT_EQ(node.input(5), "variance");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "FusedBatchNorm");
      found++;
    }
  }
  EXPECT_EQ(found, 1);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectClose(tensors[0], tensors_expected[0], 1e-6, 1e-4);
}

TEST_F(RemapperTest, FuseConv2DWithBatchNormAndActivation) {
  using ops::Placeholder;

  for (const string& activation : {"Relu", "Relu6", "Elu", "LeakyRelu"}) {
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

    float leakyrelu_alpha = 0.5;

    ops::Identity fetch = [&]() -> ops::Identity {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");

      if (activation == "Relu") {
        return ops::Identity(fetch, ops::Relu(activate, batch_norm.y));
      } else if (activation == "Relu6") {
        return ops::Identity(fetch, ops::Relu6(activate, batch_norm.y));
      } else if (activation == "Elu") {
        return ops::Identity(fetch, ops::Elu(activate, batch_norm.y));
      } else if (activation == "LeakyRelu") {
        auto attr = ops::internal::LeakyRelu::Alpha(leakyrelu_alpha);
        return ops::Identity(
            fetch, ops::internal::LeakyRelu(activate, batch_norm.y, attr));
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
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "activation") {
        EXPECT_EQ(node.op(), "_FusedConv2D");
        ASSERT_GE(node.input_size(), 6);
        EXPECT_EQ(node.input(0), "input");
        EXPECT_EQ(node.input(1), "filter");

        EXPECT_EQ(node.attr().at("num_args").i(), 4);
        EXPECT_EQ(node.input(2), "scale");
        EXPECT_EQ(node.input(3), "offset");
        EXPECT_EQ(node.input(4), "mean");
        EXPECT_EQ(node.input(5), "variance");

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(fused_ops.size(), 2);
        EXPECT_EQ(fused_ops[0], "FusedBatchNorm");
        EXPECT_EQ(fused_ops[1], activation);

        if (activation == "LeakyRelu") {
          EXPECT_EQ(node.attr().at("leakyrelu_alpha").f(), leakyrelu_alpha);
        }
        found++;
      }
    }
    EXPECT_EQ(found, 1);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    test::ExpectClose(tensors[0], tensors_expected[0], 1e-6, 1e-4);
  }
}

#ifdef INTEL_MKL
TEST_F(RemapperTest, FuseConv3DWithBiasAndAddN) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to oneDNN.";
  using ::tensorflow::ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 4, 32, 32, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 1, 3, 128});
  auto bias_shape = ops::Placeholder::Shape({128});
  auto add_shape = ops::Placeholder::Shape({8, 4, 32, 32, 128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);
  auto input_add = Placeholder(s.WithOpName("input_add"), DT_FLOAT, add_shape);

  std::vector<int> strides = {1, 1, 1, 1, 1};
  auto conv = ops::Conv3D(s.WithOpName("conv"), input, filter, strides, "SAME");
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);
  auto add = ops::AddN(s.WithOpName("add_op"),
                       std::initializer_list<Input>{input_add, bias_add});
  auto fetch = ops::Identity(s.WithOpName("fetch"), add);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 4, 32, 32, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 1, 3, 128});
  auto add_t = GenerateRandomTensor<DT_FLOAT>({8, 4, 32, 32, 128});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t},
               {"filter", filter_t},
               {"bias", bias_t},
               {"input_add", add_t}};
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
    if (node.name() == "add_op") {
      EXPECT_EQ(node.op(), "_FusedConv3D");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 2);
      EXPECT_EQ(node.input(2), "bias");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 2);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      EXPECT_EQ(fused_ops[1], "Add");
      found++;
    }
  }
  EXPECT_EQ(found, 1);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(RemapperTest, FuseConv3DWithBiasAndAdd) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to oneDNN.";
  using ::tensorflow::ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 4, 32, 32, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 1, 3, 128});
  auto bias_shape = ops::Placeholder::Shape({128});
  auto add_shape = ops::Placeholder::Shape({8, 4, 32, 32, 128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);
  auto input_add = Placeholder(s.WithOpName("input_add"), DT_FLOAT, add_shape);

  std::vector<int> strides = {1, 1, 1, 1, 1};
  auto conv = ops::Conv3D(s.WithOpName("conv"), input, filter, strides, "SAME");
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);
  auto add = ops::Add(s.WithOpName("add_op"), input_add, bias_add);
  auto fetch = ops::Identity(s.WithOpName("fetch"), add);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 4, 32, 32, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 1, 3, 128});
  auto add_t = GenerateRandomTensor<DT_FLOAT>({8, 4, 32, 32, 128});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t},
               {"filter", filter_t},
               {"bias", bias_t},
               {"input_add", add_t}};
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
    if (node.name() == "add_op") {
      EXPECT_EQ(node.op(), "_FusedConv3D");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 2);
      EXPECT_EQ(node.input(2), "bias");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 2);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      EXPECT_EQ(fused_ops[1], "Add");
      found++;
    }
  }
  EXPECT_EQ(found, 1);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(RemapperTest, FuseConv3DWithBiasAndAddActivation) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to oneDNN.";
  using ::tensorflow::ops::Placeholder;

  for (const string& activation : {"Relu", "Relu6", "Elu", "LeakyRelu"}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = Placeholder::Shape({8, 4, 32, 32, 3});
    auto filter_shape = Placeholder::Shape({1, 1, 1, 3, 128});
    auto bias_shape = Placeholder::Shape({128});
    auto add_shape = ops::Placeholder::Shape({8, 4, 32, 32, 128});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
    auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);
    auto input_add =
        Placeholder(s.WithOpName("input_add"), DT_FLOAT, add_shape);

    float leakyrelu_alpha = 0.5;

    std::vector<int> strides = {1, 1, 1, 1, 1};
    auto conv =
        ops::Conv3D(s.WithOpName("conv"), input, filter, strides, "SAME");
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);
    auto add = ops::Add(s.WithOpName("add_op"), input_add, bias_add);

    ops::Identity fetch = [&]() -> ops::Identity {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");

      if (activation == "Relu") {
        return ops::Identity(fetch, ops::Relu(activate, add));
      } else if (activation == "Relu6") {
        return ops::Identity(fetch, ops::Relu6(activate, add));
      } else if (activation == "Elu") {
        return ops::Identity(fetch, ops::Elu(activate, add));
      } else if (activation == "LeakyRelu") {
        auto attr = ops::internal::LeakyRelu::Alpha(leakyrelu_alpha);
        return ops::Identity(fetch,
                             ops::internal::LeakyRelu(activate, add, attr));
      }

      return ops::Identity(fetch, bias);
    }();

    auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 4, 32, 32, 3});
    auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 1, 3, 128});
    auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});
    auto add_t = GenerateRandomTensor<DT_FLOAT>({8, 4, 32, 32, 128});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t},
                 {"filter", filter_t},
                 {"bias", bias_t},
                 {"input_add", add_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::AGGRESSIVE);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "activation") {
        EXPECT_EQ(node.op(), "_FusedConv3D");
        ASSERT_GE(node.input_size(), 3);
        EXPECT_EQ(node.input(0), "input");
        EXPECT_EQ(node.input(1), "filter");

        EXPECT_EQ(node.attr().at("num_args").i(), 2);
        EXPECT_EQ(node.input(2), "bias");

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(fused_ops.size(), 3);
        EXPECT_EQ("BiasAdd", fused_ops[0]);
        EXPECT_EQ("Add", fused_ops[1]);
        EXPECT_EQ(activation, fused_ops[2]);
        found++;
      }
    }
    EXPECT_EQ(found, 1);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  }
}

// Conv2D + Add {6,} + Conv2D + Biasadd fusion.
TEST_F(RemapperTest, FuseConv2DWithSemanticAdd) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to MKL.";
  using ::tensorflow::ops::Placeholder;
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 6});
  auto filter_shape_1 = ops::Placeholder::Shape({1, 1, 6, 6});
  auto semanticadd_shape = ops::Placeholder::Shape({6});
  auto bias_shape = ops::Placeholder::Shape({6});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto filter_1 =
      Placeholder(s.WithOpName("filter_1"), DT_FLOAT, filter_shape_1);
  auto semanticadd =
      Placeholder(s.WithOpName("semanticadd"), DT_FLOAT, semanticadd_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv =
      ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "VALID");
  auto add = ops::Add(s.WithOpName("add"), semanticadd, conv);
  auto conv_1 =
      ops::Conv2D(s.WithOpName("conv_1"), add, filter_1, strides, "VALID");
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv_1, bias);
  auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);

  auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
      TensorShape(input_shape.shape_.dim_sizes()));
  auto filter_tensor = GenerateRandomTensor<DT_FLOAT>(
      TensorShape(filter_shape.shape_.dim_sizes()));
  auto filter_tensor_1 = GenerateRandomTensor<DT_FLOAT>(
      TensorShape(filter_shape_1.shape_.dim_sizes()));
  auto semanticadd_tensor = GenerateRandomTensor<DT_FLOAT>(
      TensorShape(semanticadd_shape.shape_.dim_sizes()));
  auto bias_tensor = GenerateRandomTensor<DT_FLOAT>(
      TensorShape(bias_shape.shape_.dim_sizes()));

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_tensor},
               {"filter", filter_tensor},
               {"filter_1", filter_tensor_1},
               {"semanticadd", semanticadd_tensor},
               {"bias", bias_tensor}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "bias_add") {
      EXPECT_EQ(node.op(), "_FusedConv2D");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "add");
      EXPECT_EQ(node.input(1), "filter_1");

      EXPECT_EQ(node.attr().at("num_args").i(), 1);
      EXPECT_EQ(node.input(2), "bias");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      found++;
    }
    if (node.name() == "add") {
      EXPECT_EQ(node.op(), "_FusedConv2D");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 1);
      EXPECT_EQ(node.input(2), "semanticadd");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      found++;
    }
  }
  EXPECT_EQ(found, 2);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

class RemapperFusePadConv3D : public RemapperTest {
 public:
  template <DataType DTYPE>
  void RunTest() {
    if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to MKL.";
    using ::tensorflow::ops::Placeholder;
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = ops::Placeholder::Shape({8, 4, 32, 32, 3});
    auto filter_shape = ops::Placeholder::Shape({1, 1, 1, 3, 6});
    auto paddings_shape = ops::Placeholder::Shape({5, 2});

    auto input = Placeholder(s.WithOpName("input"), DTYPE, input_shape);
    auto filter = Placeholder(s.WithOpName("filter"), DTYPE, filter_shape);

    std::vector<int> strides = {1, 1, 1, 1, 1};
    auto padding_const = ops::Const(s.WithOpName("padding"),
                                    {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}, {5, 2});
    auto pad = ops::Pad(s.WithOpName("pad"), input, padding_const);
    auto conv =
        ops::Conv3D(s.WithOpName("conv"), pad, filter, strides, "VALID");
    auto fetch = ops::Identity(s.WithOpName("fetch"), conv);

    auto input_t = GenerateTensorWithSetRandom<DTYPE>({8, 4, 32, 32, 3});
    auto filter_t = GenerateTensorWithSetRandom<DTYPE>({1, 1, 1, 3, 6});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"filter", filter_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::AGGRESSIVE);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "conv") {
        EXPECT_EQ(node.op(), "_FusedConv3D");
        ASSERT_GE(node.input_size(), 2);
        EXPECT_EQ(node.input(0), "input");
        EXPECT_EQ(node.input(1), "filter");
        found++;
      }
    }
    EXPECT_EQ(found, 1);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    if (DTYPE == DT_BFLOAT16)
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
    else
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
  }
};

TEST_F(RemapperFusePadConv3D, Conv3D_FP32) {
  if (!IsMKLEnabled())
    GTEST_SKIP()
        << "Pad fusion with Conv3D is only enabled with oneDNN, skipping "
           "RemapperFusePadConv3D with FP32.";
  RunTest<DT_FLOAT>();
}
TEST_F(RemapperFusePadConv3D, Conv3D_BF16) {
  if (!IsMKLEnabled())
    GTEST_SKIP() << "Intel MKL with bfloat16 support is not enabled, skipping "
                    "RemapperFusePadConv3D with bfloat16.";
  RunTest<DT_BFLOAT16>();
}

class RemapperFusePadWithFusedConv3D : public RemapperTest {
 public:
  template <DataType DTYPE>
  void RunTest() {
    if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to oneDNN.";
    using ::tensorflow::ops::Placeholder;

    // Empty string denotes no activation.
    for (const string& activation : {"", "Relu", "Relu6", "Elu", "LeakyRelu"}) {
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();
      auto input_shape = ops::Placeholder::Shape({8, 4, 32, 32, 3});
      auto filter_shape = ops::Placeholder::Shape({1, 1, 1, 3, 128});
      auto bias_shape = ops::Placeholder::Shape({128});
      auto paddings_shape = ops::Placeholder::Shape({5, 2});
      auto strides = {1, 1, 1, 1, 1};

      auto input_t = GenerateTensorWithSetRandom<DTYPE>({8, 4, 32, 32, 3});
      auto filter_t = GenerateTensorWithSetRandom<DTYPE>({1, 1, 1, 3, 128});
      auto bias_t = GenerateTensorWithSetRandom<DTYPE>({128});

      auto input = Placeholder(s.WithOpName("input"), DTYPE, input_shape);
      auto filter = Placeholder(s.WithOpName("filter"), DTYPE, filter_shape);
      auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

      auto padding_const = ops::Const(s.WithOpName("padding"),
                                      {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}, {5, 2});
      auto pad = ops::Pad(s.WithOpName("pad"), input, padding_const);
      auto conv =
          ops::Conv3D(s.WithOpName("conv"), pad, filter, strides, "SAME");
      auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);

      float leakyrelu_alpha = 0.5;
      ops::Identity fetch = [&]() -> ops::Identity {
        auto activate = s.WithOpName("activation");
        auto fetch = s.WithOpName("fetch");

        if (activation == "Relu") {
          return ops::Identity(fetch, ops::Relu(activate, bias_add));
        } else if (activation == "Relu6") {
          return ops::Identity(fetch, ops::Relu6(activate, bias_add));
        } else if (activation == "Elu") {
          return ops::Identity(fetch, ops::Elu(activate, bias_add));
        } else if (activation == "LeakyRelu") {
          auto attr = ops::internal::LeakyRelu::Alpha(leakyrelu_alpha);
          return ops::Identity(
              fetch, ops::internal::LeakyRelu(activate, bias_add, attr));
        }

        return ops::Identity(fetch, bias);
      }();

      GrapplerItem item;
      item.fetch = {"fetch"};
      item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
      TF_ASSERT_OK(s.ToGraphDef(&item.graph));

      // Place all nodes on CPU.
      for (int i = 0; i < item.graph.node_size(); ++i) {
        item.graph.mutable_node(i)->set_device("/device:CPU:0");
      }

      Remapper optimizer(RewriterConfig::ON);
      GraphDef output_1;
      TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output_1));
      item.graph = std::move(output_1);
      GraphDef output;
      TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

      string fused_node_name;
      std::vector<string> expected_fused_ops = {"BiasAdd"};
      if (activation.empty()) {
        fused_node_name = "bias_add";
      } else {
        fused_node_name = "activation";
        expected_fused_ops.push_back(activation);
      }
      int found = 0;
      for (const NodeDef& node : output.node()) {
        if (node.name() == fused_node_name) {
          EXPECT_EQ(node.op(), "_FusedConv3D");
          ASSERT_GE(node.input_size(), 3);
          EXPECT_EQ(node.input(0), "input");
          EXPECT_EQ(node.input(1), "filter");
          EXPECT_EQ(node.attr().at("num_args").i(), 1);
          EXPECT_EQ(node.input(2), "bias");
          const auto fused_ops = node.attr().at("fused_ops").list().s();
          ASSERT_EQ(fused_ops.size(), expected_fused_ops.size());
          for (int i = 0; i < fused_ops.size(); ++i) {
            EXPECT_EQ(fused_ops[i], expected_fused_ops[i]);
          }
          if (activation == "LeakyRelu") {
            EXPECT_EQ(node.attr().at("leakyrelu_alpha").f(), leakyrelu_alpha);
          }
          found++;
        }
      }
      EXPECT_EQ(found, 1);

      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      if (DTYPE == DT_BFLOAT16)
        test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
      else
        test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
    }
  }
};

TEST_F(RemapperFusePadWithFusedConv3D, FusedConv3D_FP32) {
  if (!IsMKLEnabled())
    GTEST_SKIP()
        << "Pad fusion with FusedConv3D is only enabled with oneDNN, skipping "
           "RemapperFusePadWithFusedConv3D with FP32.";
  RunTest<DT_FLOAT>();
}
TEST_F(RemapperFusePadWithFusedConv3D, FusedConv3D_BF16) {
  if (!IsMKLEnabled())
    GTEST_SKIP() << "Intel MKL with bfloat16 support is not enabled, skipping "
                    "RemapperFusePadWithFusedConv3D with bfloat16.";
  RunTest<DT_BFLOAT16>();
}
#endif

class RemapperLeakyReluTest : public GrapplerTest {
 protected:
  template <DataType DTYPE>
  void RunTest() {
    if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to oneDNN.";
    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    auto max_shape = ops::Placeholder::Shape({64, 64});

    // y = maximum(x, alpha * x)
    auto input = Placeholder(s.WithOpName("input"), DTYPE, max_shape);
    float epsilon = 0.3f;

    typedef typename EnumToDataType<DTYPE>::Type CType;
    auto leakyrelu_alpha = ops::Const<CType>(s.WithOpName("alpha"), epsilon);

    auto mul = ops::Mul(s.WithOpName("Mul"), input, leakyrelu_alpha);
    auto max = ops::Maximum(s.WithOpName("Maximum"), mul, input);

    auto fetch = ops::Identity(s.WithOpName("fetch"), max);
    auto max_t = GenerateTensorWithSetRandom<DTYPE>({64, 64});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", max_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "Maximum") {
        EXPECT_EQ(node.op(), "LeakyRelu");
        ASSERT_EQ(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "input");
        ++found;
      }
    }
    EXPECT_EQ(found, 1);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    float atol = 1e-6, rtol = 1e-6;
    if (DTYPE == DT_BFLOAT16) {
      atol = 1e-2;
      rtol = 1e-2;
    }
    test::ExpectClose(tensors[0], tensors_expected[0], atol, rtol);
  }
};

TEST_F(RemapperLeakyReluTest, F32) { RunTest<DT_FLOAT>(); }
TEST_F(RemapperLeakyReluTest, BF16) { RunTest<DT_BFLOAT16>(); }

}  // namespace grappler
}  // namespace tensorflow
