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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h"

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops_internal.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

namespace {

using ::tensorflow::test::ExpectTensorEqual;

constexpr int kBatchSize = 32;
constexpr int kWidth = 10;
constexpr int kHeight = 10;
constexpr int kDepthIn = 8;
constexpr int kKernel = 2;
constexpr int kStride1 = 2;
constexpr int kStride2 = 4;
constexpr int kOutWidth = 5;
constexpr int kOutHeight = 5;
constexpr int kDepthOut = 16;
constexpr int kDilation = 2;
constexpr int kPaddingTop = 1;
constexpr int kPaddingBottom = 2;
constexpr int kPaddingLeft = 3;
constexpr int kPaddingRight = 4;
constexpr char kSrcFormat[] = "NHWC";
constexpr char kDstFormat[] = "NCHW";
constexpr char kGPU[] = "GPU";
constexpr char kAttrOutputShapes[] = "_output_shapes";
constexpr char kAttrDataFormat[] = "data_format";
constexpr char kOpTranspose[] = "Transpose";

class TransposerImpl : public Transposer {
 public:
  explicit TransposerImpl() : Transposer() {}
  Status TransposeNode(TransposeContext*, utils::MutableNodeView*) override {
    return Status::OK();
  }
};

void VerifyRegularFaninMatch(const utils::MutableNodeView* node, int port,
                             absl::string_view fanin_name, int fanin_port) {
  ASSERT_GT(node->NumRegularFanins(), port);
  const auto& fanin = node->GetRegularFanin(port);
  EXPECT_EQ(fanin.node_view()->GetName(), fanin_name);
  EXPECT_EQ(fanin.index(), fanin_port);
}

void VerifyShapeAttributeMatch(const utils::MutableNodeView* node,
                               absl::string_view attr_value) {
  const auto* attr = node->GetAttr(kAttrOutputShapes);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->shape().DebugString(), attr_value);
}

void VerifyShapeAttributeMatch(const utils::MutableNodeView* node,
                               int shape_index, absl::string_view attr_value) {
  const auto* attr = node->GetAttr(kAttrOutputShapes);
  ASSERT_NE(attr, nullptr);
  ASSERT_GT(attr->list().shape_size(), shape_index);
  EXPECT_EQ(attr->list().shape(shape_index).DebugString(), attr_value);
}

void VerifyDataFormatAttributeMatch(const utils::MutableNodeView* node,
                                    absl::string_view attr_value) {
  const auto* attr = node->GetAttr(kAttrDataFormat);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->s(), attr_value);
}

Output SimpleConv2D(const Scope* scope, const DataType& data_type = DT_FLOAT) {
  auto input =
      ops::RandomUniform(scope->WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, data_type);
  auto filter =
      ops::RandomUniform(scope->WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, data_type);
  auto conv2d = ops::Conv2D(
      scope->WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, kStride1, kStride2, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

  return conv2d;
}

Status CreateSimpleConv2DGraph(GraphDef* graph,
                               const DataType& data_type = DT_FLOAT) {
  Scope scope = Scope::NewRootScope();
  auto conv2d = SimpleConv2D(&scope, data_type);
  auto output = ops::Identity(scope.WithOpName("output"), conv2d);

  return scope.ToGraphDef(graph);
}

Status CreateSimpleFusedBatchNorm(GraphDef* graph,
                                  const DataType& data_type = DT_FLOAT) {
  Scope scope = Scope::NewRootScope();
  auto x =
      ops::RandomUniform(scope.WithOpName("x"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, data_type);
  auto scale =
      ops::RandomUniform(scope.WithOpName("scale"), {kDepthIn}, DT_FLOAT);
  auto offset =
      ops::RandomUniform(scope.WithOpName("offset"), {kDepthIn}, DT_FLOAT);
  auto mean =
      ops::RandomUniform(scope.WithOpName("mean"), {kDepthIn}, DT_FLOAT);
  auto var = ops::RandomUniform(scope.WithOpName("var"), {kDepthIn}, DT_FLOAT);
  auto batch_norm = ops::FusedBatchNormV2(
      scope.WithOpName("bn").WithDevice("/device:GPU:0"), x, scale, offset,
      mean, var, ops::FusedBatchNormV2::IsTraining(false).Epsilon(0.1f));

  auto output_y = ops::Identity(scope.WithOpName("output_y"), batch_norm.y);
  auto output_mean =
      ops::Identity(scope.WithOpName("output_mean"), batch_norm.batch_mean);
  auto output_variance = ops::Identity(scope.WithOpName("output_variance"),
                                       batch_norm.batch_variance);

  return scope.ToGraphDef(graph);
}

Status CreateSimpleMaxPoolGrad(GraphDef* graph, bool use_grad_grad) {
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("orig_input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto output_data = ops::RandomUniform(
      scope.WithOpName("orig_output"),
      {kBatchSize, kOutHeight, kOutWidth, kDepthIn}, DT_FLOAT);
  auto output_grad =
      ops::RandomUniform(scope.WithOpName("grad"),
                         {kBatchSize, use_grad_grad ? kHeight : kOutHeight,
                          use_grad_grad ? kWidth : kOutWidth, kDepthIn},
                         DT_FLOAT);
  Output maxpool_grad;
  if (use_grad_grad) {
    maxpool_grad = ops::MaxPoolGradGrad(
        scope.WithOpName("maxpool_grad").WithDevice("/device:GPU:0"), input,
        output_data, output_grad, {1, kKernel, kKernel, 1},
        {1, kStride1, kStride1, 1}, "VALID");
  } else {
    maxpool_grad = ops::internal::MaxPoolGrad(
        scope.WithOpName("maxpool_grad").WithDevice("/device:GPU:0"), input,
        output_data, output_grad, {1, kKernel, kKernel, 1},
        {1, kStride1, kStride1, 1}, "VALID");
  }

  auto output = ops::Identity(scope.WithOpName("output"), maxpool_grad);

  return scope.ToGraphDef(graph);
}

Status CreateSimpleBiasAddGrad(GraphDef* graph, const Input& shape) {
  Scope scope = Scope::NewRootScope();
  auto input = ops::RandomUniform(scope.WithOpName("input"), shape, DT_FLOAT);
  auto bag =
      ops::BiasAddGrad(scope.WithOpName("bag").WithDevice("/device:GPU:0"),
                       input, ops::BiasAddGrad::DataFormat(kSrcFormat));
  auto output = ops::Identity(scope.WithOpName("output"), bag);

  return scope.ToGraphDef(graph);
}

Status CreateSimpleConv2DBackpropFilter(GraphDef* graph,
                                        const DataType& data_type = DT_FLOAT,
                                        absl::string_view padding = "SAME") {
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, data_type);
  auto out_backprop =
      ops::RandomUniform(scope.WithOpName("out_backprop"),
                         {kBatchSize, kHeight, kWidth, kDepthOut}, data_type);
  if (padding == "EXPLICIT") {
    auto conv2d_backprop_filter = ops::Conv2DBackpropFilter(
        scope.WithOpName("conv2d_backprop_filter").WithDevice("/device:GPU:0"),
        input, {kHeight, kWidth, kDepthIn, kDepthOut}, out_backprop,
        {1, 2, 4, 1}, padding,
        ops::Conv2DBackpropFilter::Attrs()
            .Dilations({1, kDilation, kDilation, 1})
            .ExplicitPaddings({0, 0, kPaddingTop, kPaddingBottom, kPaddingLeft,
                               kPaddingRight, 0, 0})
            .DataFormat(kSrcFormat));
    auto output =
        ops::Identity(scope.WithOpName("output"), conv2d_backprop_filter);
  } else {
    auto conv2d_backprop_filter = ops::Conv2DBackpropFilter(
        scope.WithOpName("conv2d_backprop_filter").WithDevice("/device:GPU:0"),
        input, {kHeight, kWidth, kDepthIn, kDepthOut}, out_backprop,
        {1, 2, 4, 1}, padding,
        ops::Conv2DBackpropFilter::DataFormat(kSrcFormat));
    auto output =
        ops::Identity(scope.WithOpName("output"), conv2d_backprop_filter);
  }

  return scope.ToGraphDef(graph);
}

Status CreateSimpleConv2DBackpropInput(GraphDef* graph,
                                       const DataType& data_type = DT_FLOAT) {
  Scope scope = Scope::NewRootScope();
  auto input_sizes = ops::Const(scope.WithOpName("input_sizes"),
                                {kBatchSize, kHeight, kWidth, kDepthIn});
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, data_type);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, data_type);
  auto out_backprop =
      ops::RandomUniform(scope.WithOpName("out_backprop"),
                         {kBatchSize, kHeight, kWidth, kDepthOut}, data_type);
  auto conv2d_backprop_input = ops::Conv2DBackpropInput(
      scope.WithOpName("conv2d_backprop_input").WithDevice("/device:GPU:0"),
      input_sizes, filter, out_backprop, {1, kStride1, kStride1, 1}, "VALID");
  auto output =
      ops::Identity(scope.WithOpName("output"), conv2d_backprop_input);

  return scope.ToGraphDef(graph);
}

Status CreateSimpleFusedBatchNormGrad(GraphDef* graph, bool is_training,
                                      const DataType& data_type = DT_FLOAT) {
  Scope scope = Scope::NewRootScope();
  auto y_backprop =
      ops::RandomUniform(scope.WithOpName("y_backprop"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, data_type);
  auto x =
      ops::RandomUniform(scope.WithOpName("x"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, data_type);
  auto scale =
      ops::RandomUniform(scope.WithOpName("scale"), {kDepthIn}, DT_FLOAT);
  auto reserve_space_1 = ops::RandomUniform(scope.WithOpName("reserve_space_1"),
                                            {kDepthIn}, DT_FLOAT);
  auto reserve_space_2 = ops::RandomUniform(scope.WithOpName("reserve_space_2"),
                                            {kDepthIn}, DT_FLOAT);
  auto fused_batch_norm_grad = ops::FusedBatchNormGradV2(
      scope.WithOpName("fused_batch_norm_grad").WithDevice("/device:GPU:0"),
      y_backprop, x, scale, reserve_space_1, reserve_space_2,
      ops::FusedBatchNormGradV2::DataFormat(kSrcFormat)
          .IsTraining(is_training)
          .Epsilon(0.1f));
  auto x_backprop = ops::Identity(scope.WithOpName("x_backprop"),
                                  fused_batch_norm_grad.x_backprop);
  auto scale_backprop = ops::Identity(scope.WithOpName("scale_backprop"),
                                      fused_batch_norm_grad.scale_backprop);
  auto offset_backprop = ops::Identity(scope.WithOpName("offset_backprop"),
                                       fused_batch_norm_grad.offset_backprop);
  auto reserve_space_3 = ops::Identity(scope.WithOpName("reserve_space_3"),
                                       fused_batch_norm_grad.reserve_space_3);
  auto reserve_space_4 = ops::Identity(scope.WithOpName("reserve_space_4"),
                                       fused_batch_norm_grad.reserve_space_4);

  return scope.ToGraphDef(graph);
}

Status CreateSimpleAddN(GraphDef* graph) {
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  Output a = ops::RandomUniform(scope.WithOpName("a"),
                                {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  Output b = ops::RandomUniform(scope.WithOpName("b"),
                                {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  Output c = ops::RandomUniform(scope.WithOpName("c"),
                                {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  auto add_n = ops::AddN(scope.WithOpName("add_n").WithDevice("/device:GPU:0"),
                         {a, b, c, conv2d});
  auto output = ops::Identity(scope.WithOpName("output"), add_n);

  return scope.ToGraphDef(graph);
}

Status CreateSimpleIdentityN(GraphDef* graph) {
  Scope scope = Scope::NewRootScope();
  auto conv2d_1_input =
      ops::RandomUniform(scope.WithOpName("conv2d_1_input"),
                         {kBatchSize, kDepthIn, kHeight, kWidth}, DT_FLOAT);
  auto conv2d_1_filter =
      ops::RandomUniform(scope.WithOpName("conv2d_1_filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d_1 =
      ops::Conv2D(scope.WithOpName("conv2d_1").WithDevice("/device:GPU:0"),
                  conv2d_1_input, conv2d_1_filter, {1, 1, 2, 4}, "SAME",
                  ops::Conv2D::DataFormat(kDstFormat));
  auto conv2d_2_input =
      ops::RandomUniform(scope.WithOpName("conv2d_2_input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto conv2d_2_filter =
      ops::RandomUniform(scope.WithOpName("conv2d_2_filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d_2 =
      ops::Conv2D(scope.WithOpName("conv2d_2").WithDevice("/device:GPU:0"),
                  conv2d_2_input, conv2d_2_filter, {1, 2, 4, 1}, "SAME",
                  ops::Conv2D::DataFormat(kSrcFormat));
  Output a = ops::RandomUniform(
      scope.WithOpName("a"), {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  Output b = ops::RandomUniform(scope.WithOpName("b"), {kBatchSize, kDepthIn},
                                DT_FLOAT);
  auto identity_n =
      ops::IdentityN(scope.WithOpName("identity_n").WithDevice("/device:GPU:0"),
                     {conv2d_1, conv2d_2, a, b});
  auto conv2d_1_output =
      ops::Identity(scope.WithOpName("conv2d_1_output"), identity_n.output[0]);
  auto conv2d_2_output =
      ops::Identity(scope.WithOpName("conv2d_2_output"), identity_n.output[1]);
  auto a_output =
      ops::Identity(scope.WithOpName("a_output"), identity_n.output[2]);
  auto b_output =
      ops::Identity(scope.WithOpName("b_output"), identity_n.output[3]);

  return scope.ToGraphDef(graph);
}

class TransposerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    bool gpu_available = GetNumAvailableGPUs() > 0;

    if (gpu_available) {
      virtual_cluster_ =
          absl::make_unique<SingleMachine>(/*timeout_s=*/10, 1, 1);
    } else {
      DeviceProperties gpu_device;
      gpu_device.set_type(kGPU);
      gpu_device.mutable_environment()->insert({"architecture", "6"});
      virtual_cluster_ =
          absl::WrapUnique(new VirtualCluster({{"/GPU:1", gpu_device}}));
    }
    TF_ASSERT_OK(virtual_cluster_->Provision());
  }

  void TearDown() override { TF_ASSERT_OK(virtual_cluster_->Shutdown()); }

  template <typename T>
  void ReduceTransposerKeepDims() {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
    GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
    GrapplerItem item;
    Scope scope = Scope::NewRootScope();

    auto input =
        ops::RandomUniform(scope.WithOpName("input"),
                           {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
    auto filter =
        ops::RandomUniform(scope.WithOpName("filter"),
                           {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
    Output conv2d = ops::Conv2D(
        scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
        {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

    auto axis = ops::Const<T>(scope.WithOpName("axis"), {0, 1, 2}, {3});
    auto attrs = ops::Sum::Attrs().KeepDims(true);
    auto sum_op = ops::Sum(scope.WithOpName("sum").WithDevice("/device:GPU:0"),
                           conv2d, axis, attrs);

    auto z = ops::Identity(scope.WithOpName("z"), sum_op);
    TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

    TransposeContext context;
    TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
        item, virtual_cluster_.get(), &context));
    context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

    DefaultLayoutSensitiveOpTransposer conv2d_transposer;
    auto* c2d = context.graph_view->GetNode("conv2d");
    ASSERT_NE(c2d, nullptr);
    TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

    ReduceTransposer reducer_transposer;
    auto* sum = context.graph_view->GetNode("sum");
    ASSERT_NE(sum, nullptr);
    TF_ASSERT_OK(reducer_transposer.TransposeNode(&context, sum));

    auto* input_transpose_node = context.graph_view->GetNode(
        "sum-0-TransposeNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(input_transpose_node, nullptr);

    auto* updated_sum_node = context.graph_view->GetNode("sum");
    ASSERT_NE(updated_sum_node, nullptr);
    ASSERT_EQ(updated_sum_node->NumRegularFanins(), 2);
    VerifyRegularFaninMatch(updated_sum_node, 0,
                            input_transpose_node->GetName(), 0);

    auto* axis_node = context.graph_view->GetNode(
        "sum-1-DataFormatDimMapNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(axis_node, nullptr);
    ASSERT_EQ(axis_node->NumRegularFanins(), 1);
    VerifyRegularFaninMatch(axis_node, 0, "axis", 0);

    auto* output_transpose_node = context.graph_view->GetNode(
        "sum-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
    ASSERT_NE(output_transpose_node, nullptr);

    auto* z_output_node = context.graph_view->GetNode("z");
    ASSERT_NE(z_output_node, nullptr);
    ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
    VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                            0);
  }

  template <typename T>
  void ReduceTransposerValidAxisNode() {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
    GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
    GrapplerItem item;
    Scope scope = Scope::NewRootScope();

    auto input =
        ops::RandomUniform(scope.WithOpName("input"),
                           {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
    auto filter =
        ops::RandomUniform(scope.WithOpName("filter"),
                           {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
    Output conv2d = ops::Conv2D(
        scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
        {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

    auto axis = ops::Const<T>(scope.WithOpName("axis"), {0, 1, 2}, {3});
    auto sum_op = ops::Max(scope.WithOpName("max").WithDevice("/device:GPU:0"),
                           conv2d, axis);

    auto z = ops::Identity(scope.WithOpName("z"), sum_op);
    TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

    TransposeContext context;
    TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
        item, virtual_cluster_.get(), &context));
    context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

    DefaultLayoutSensitiveOpTransposer conv2d_transposer;
    auto* c2d = context.graph_view->GetNode("conv2d");
    ASSERT_NE(c2d, nullptr);
    TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

    ReduceTransposer reducer_transposer;
    auto* max = context.graph_view->GetNode("max");
    ASSERT_NE(max, nullptr);
    TF_ASSERT_OK(reducer_transposer.TransposeNode(&context, max));

    auto* input_transpose_node = context.graph_view->GetNode(
        "max-0-TransposeNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(input_transpose_node, nullptr);

    auto* updated_max_node = context.graph_view->GetNode("max");
    ASSERT_NE(updated_max_node, nullptr);
    ASSERT_EQ(updated_max_node->NumRegularFanins(), 2);
    VerifyRegularFaninMatch(updated_max_node, 0,
                            input_transpose_node->GetName(), 0);

    auto* axis_node = context.graph_view->GetNode(
        "max-1-DataFormatDimMapNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(axis_node, nullptr);
    ASSERT_EQ(axis_node->NumRegularFanins(), 1);
    VerifyRegularFaninMatch(axis_node, 0, "axis", 0);

    auto* z_output_node = context.graph_view->GetNode("z");
    ASSERT_NE(z_output_node, nullptr);
    ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
    VerifyRegularFaninMatch(z_output_node, 0, updated_max_node->GetName(), 0);
  }

  std::unique_ptr<Cluster> virtual_cluster_;
};

TEST_F(TransposerTest, CreateConstPermNode) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleConv2DGraph(&item.graph));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  TransposerImpl transposer;
  constexpr char kNodeName[] = "const_perm_node";
  constexpr char kDevice[] = "/device:GPU:0";
  utils::MutationNewNode added_node;
  EXPECT_FALSE(context.graph_view->HasNode(kNodeName));
  TF_ASSERT_OK(transposer.CreateConstPermNode(&context, kNodeName, kDevice,
                                              {0, 3, 1, 2}, "", &added_node));
  TF_ASSERT_OK(context.graph_view->GetMutationBuilder()->Apply());

  utils::MutableNodeView* const_perm_node =
      context.graph_view->GetNode(kNodeName);
  EXPECT_EQ(const_perm_node->GetName(), kNodeName);
  EXPECT_EQ(const_perm_node->GetDevice(), kDevice);
  const auto* value_attr = const_perm_node->GetAttr("value");
  ASSERT_NE(value_attr, nullptr);

  Tensor tensor;
  ASSERT_TRUE(tensor.FromProto(value_attr->tensor()));
  Tensor expected(DT_INT32, {4});
  ::tensorflow::test::FillValues<int32>(&expected, {0, 3, 1, 2});
  ExpectTensorEqual<int32>(tensor, expected);
}

TensorShapeProto MakeTensorShapeFromDimensions(absl::Span<const int> dims) {
  TensorShapeProto shape_proto = TensorShapeProto();
  for (const int dim : dims) {
    TensorShapeProto_Dim dim_proto = TensorShapeProto_Dim();
    dim_proto.set_size(dim);
    *shape_proto.add_dim() = std::move(dim_proto);
  }
  return shape_proto;
}

TEST_F(TransposerTest, CreateTransposeNode) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleConv2DGraph(&item.graph));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  TransposerImpl transposer;
  constexpr char kNodeNameFormat[] =
      "transpose_node-0-$0-NWCHToNCWH-LayoutOptimizer";
  constexpr char kDevice[] = "/device:GPU:0";
  TensorShapeProto input_shape = MakeTensorShapeFromDimensions({1, 2, 3, 4});
  TensorShapeProto expected_shape = MakeTensorShapeFromDimensions({1, 4, 2, 3});
  utils::MutationNewNode added_node;
  string transpose_node_name;
  TF_ASSERT_OK(transposer.CreateTransposeNode(
      &context, kNodeNameFormat, DT_DOUBLE, kDevice, input_shape, {0, 3, 1, 2},
      "", &added_node, &transpose_node_name));

  EXPECT_EQ(transpose_node_name,
            "transpose_node-0-Transpose-NWCHToNCWH-LayoutOptimizer");
  utils::Mutation* mutation = context.graph_view->GetMutationBuilder();
  Status status;
  // Placeholder node with empty name as transpose node is created with it's
  // first input not set.
  mutation->AddNode({}, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(context.graph_view->GetMutationBuilder()->Apply());
  auto* transpose_node = context.graph_view->GetNode(transpose_node_name);
  ASSERT_NE(transpose_node, nullptr);
  EXPECT_EQ(transpose_node->GetDevice(), kDevice);
  const auto* output_shapes_attr = transpose_node->GetAttr("_output_shapes");
  EXPECT_EQ(output_shapes_attr->list().shape(0).DebugString(),
            expected_shape.DebugString());
}

TEST_F(TransposerTest, UpdateNode) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleConv2DGraph(&item.graph));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer transposer;
  auto* conv2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(conv2d, nullptr);
  TF_ASSERT_OK(transposer.UpdateNode(&context, conv2d));
  TF_ASSERT_OK(context.graph_view->GetMutationBuilder()->Apply());

  auto* updated_conv2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(updated_conv2d, nullptr);
  VerifyDataFormatAttributeMatch(updated_conv2d, kDstFormat);
}

AttrValue_ListValue MakeAttrValueListValueFromVector(
    absl::Span<const int> vec) {
  AttrValue_ListValue list_proto = AttrValue_ListValue();
  for (const int i : vec) {
    list_proto.add_i(i);
  }
  return list_proto;
}

TEST_F(TransposerTest, UpdateStrides) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleConv2DGraph(&item.graph));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, "ABCD", "ACBD");

  AttrValue_ListValue expected_original_strides =
      MakeAttrValueListValueFromVector({1, 2, 4, 1});
  AttrValue_ListValue expected_updated_strides =
      MakeAttrValueListValueFromVector({1, 4, 2, 1});
  auto* conv2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(conv2d, nullptr);
  const auto& strides_attr = conv2d->GetAttr("strides");
  ASSERT_NE(strides_attr, nullptr);
  EXPECT_EQ(strides_attr->list().DebugString(),
            expected_original_strides.DebugString());
  AttrValue data_format_attr;
  data_format_attr.set_s("ABCD");
  context.graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(
      conv2d, "data_format", data_format_attr);
  TF_ASSERT_OK(context.graph_view->GetMutationBuilder()->Apply());

  DefaultLayoutSensitiveOpTransposer transposer;
  TF_ASSERT_OK(transposer.UpdateNode(&context, conv2d));
  TF_ASSERT_OK(context.graph_view->GetMutationBuilder()->Apply());

  auto* updated_conv2d = context.graph_view->GetNode("conv2d");
  const auto& updated_strides_attr = updated_conv2d->GetAttr("strides");
  ASSERT_NE(updated_strides_attr, nullptr);
  EXPECT_EQ(updated_strides_attr->list().DebugString(),
            expected_updated_strides.DebugString());
}

TEST_F(TransposerTest, UpdateFaninEdgesTranspose) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleFusedBatchNormGrad(&item.graph, true));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  FusedBatchNormGradTransposer transposer;
  auto* fbng = context.graph_view->GetNode("fused_batch_norm_grad");
  ASSERT_NE(fbng, nullptr);
  const auto& fbng_output_shapes_attr = fbng->GetAttr("_output_shapes");
  ASSERT_NE(fbng_output_shapes_attr, nullptr);
  const TensorShapeProto& expected_shape = fbng_output_shapes_attr->shape();
  TF_ASSERT_OK(
      transposer.UpdateFaninEdgesWithOp(&context, {0, 1}, fbng, kOpTranspose));
  TF_ASSERT_OK(context.graph_view->GetMutationBuilder()->Apply());

  // Verify output shape matches input shape.
  auto* transpose_node1 = context.graph_view->GetNode(
      "fused_batch_norm_grad-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(transpose_node1, nullptr);
  VerifyShapeAttributeMatch(transpose_node1, expected_shape.DebugString());
  auto* transpose_node2 = context.graph_view->GetNode(
      "fused_batch_norm_grad-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(transpose_node2, nullptr);
  VerifyShapeAttributeMatch(transpose_node2, expected_shape.DebugString());

  // Validate a const perm node is created.
  auto* const_node1 = context.graph_view->GetNode(
      "fused_batch_norm_grad-0-PermConstNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(const_node1, nullptr);
  auto* const_node2 = context.graph_view->GetNode(
      "fused_batch_norm_grad-1-PermConstNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(const_node2, nullptr);

  // Validate nodes connected correctly.
  auto* y_backprop = context.graph_view->GetNode("y_backprop");
  ASSERT_NE(y_backprop, nullptr);
  ASSERT_EQ(transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(transpose_node1, 0, y_backprop->GetName(), 0);
  VerifyRegularFaninMatch(transpose_node1, 1, const_node1->GetName(), 0);

  auto* x = context.graph_view->GetNode("x");
  ASSERT_NE(x, nullptr);
  ASSERT_EQ(transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(transpose_node2, 0, x->GetName(), 0);
  VerifyRegularFaninMatch(transpose_node2, 1, const_node2->GetName(), 0);

  auto* updated_fbng = context.graph_view->GetNode("fused_batch_norm_grad");
  ASSERT_NE(updated_fbng, nullptr);
  ASSERT_EQ(updated_fbng->NumRegularFanins(), 5);
  VerifyRegularFaninMatch(updated_fbng, 0, transpose_node1->GetName(), 0);
  VerifyRegularFaninMatch(updated_fbng, 1, transpose_node2->GetName(), 0);
}

TEST_F(TransposerTest, UpdateFanoutEdgesTranspose) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleConv2DGraph(&item.graph));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  TransposerImpl transposer;
  TensorShapeProto expected_original_shape =
      MakeTensorShapeFromDimensions({32, 5, 3, 16});
  TensorShapeProto expected_updated_shape =
      MakeTensorShapeFromDimensions({32, 16, 5, 3});

  auto* conv2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(conv2d, nullptr);
  VerifyShapeAttributeMatch(conv2d, 0, expected_original_shape.DebugString());

  TF_ASSERT_OK(
      transposer.UpdateFanoutEdgesWithOp(&context, {0}, conv2d, kOpTranspose));
  TF_ASSERT_OK(context.graph_view->GetMutationBuilder()->Apply());

  auto* updated_conv2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(updated_conv2d, nullptr);
  VerifyShapeAttributeMatch(updated_conv2d, 0,
                            expected_updated_shape.DebugString());

  // Verify output shape matches original shape.
  auto* transpose_node = context.graph_view->GetNode(
      "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(transpose_node, nullptr);
  VerifyShapeAttributeMatch(transpose_node, 0,
                            expected_original_shape.DebugString());

  // Verify a const perm node is created for transpose node.
  auto* const_node = context.graph_view->GetNode(
      "conv2d-0-0-PermConstNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(const_node, nullptr);

  // Verify nodes connected correctly.
  ASSERT_EQ(transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(transpose_node, 0, updated_conv2d->GetName(), 0);
  VerifyRegularFaninMatch(transpose_node, 1, const_node->GetName(), 0);

  auto* output = context.graph_view->GetNode("output");
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output, 0, transpose_node->GetName(), 0);
}

TEST_F(TransposerTest, DefaultLayoutSensitiveOpTransposerTestFusedBatchNorm) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  // Use FusedBatchNorm for default transposer test
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleFusedBatchNorm(&item.graph));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer transposer;
  auto* bn = context.graph_view->GetNode("bn");
  TF_ASSERT_OK(transposer.TransposeNode(&context, bn));

  // The expected optimized graph contains 2 extra sets of Transpose nodes and
  // has the FusedBatchNorm's data_format set to "NCHW".
  auto* input_transpose_node =
      context.graph_view->GetNode("bn-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0, "x", 0);

  auto* bn_node = context.graph_view->GetNode("bn");
  ASSERT_NE(bn_node, nullptr);
  ASSERT_EQ(bn_node->NumRegularFanins(), 5);
  VerifyRegularFaninMatch(bn_node, 0, input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(bn_node, 1, "scale", 0);
  VerifyRegularFaninMatch(bn_node, 2, "offset", 0);
  VerifyRegularFaninMatch(bn_node, 3, "mean", 0);
  VerifyRegularFaninMatch(bn_node, 4, "var", 0);
  VerifyDataFormatAttributeMatch(bn_node, kDstFormat);

  auto* output_transpose_node =
      context.graph_view->GetNode("bn-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, bn_node->GetName(), 0);

  auto* output_y = context.graph_view->GetNode("output_y");
  ASSERT_NE(output_y, nullptr);
  ASSERT_EQ(output_y->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_y, 0, output_transpose_node->GetName(), 0);

  auto* output_mean = context.graph_view->GetNode("output_mean");
  ASSERT_NE(output_mean, nullptr);
  ASSERT_EQ(output_mean->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_mean, 0, bn_node->GetName(), 1);

  auto* output_variance = context.graph_view->GetNode("output_variance");
  ASSERT_NE(output_variance, nullptr);
  ASSERT_EQ(output_variance->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_variance, 0, bn_node->GetName(), 2);
}

TEST_F(TransposerTest, DefaultLayoutSensitiveOpTransposerTestConv2D) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  // Use Conv2D for default transposer test
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleConv2DGraph(&item.graph));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer transposer;
  auto* conv2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(conv2d, nullptr);
  TF_ASSERT_OK(transposer.TransposeNode(&context, conv2d));

  // The expected optimized graph contains 2 extra sets of Transpose nodes and
  // has the Conv2D's data_format set to "NCHW".
  auto* input_transpose_node = context.graph_view->GetNode(
      "conv2d-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0, "input", 0);

  auto* conv2d_node = context.graph_view->GetNode("conv2d");
  ASSERT_NE(conv2d_node, nullptr);
  ASSERT_EQ(conv2d_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(conv2d_node, 0, input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(conv2d_node, 1, "filter", 0);
  VerifyDataFormatAttributeMatch(conv2d_node, kDstFormat);
  const auto* strides_attr = conv2d_node->GetAttr("strides");
  ASSERT_NE(strides_attr, nullptr);
  ASSERT_EQ(strides_attr->list().i_size(), 4);
  EXPECT_EQ(strides_attr->list().i(0), 1);
  EXPECT_EQ(strides_attr->list().i(1), 1);
  EXPECT_EQ(strides_attr->list().i(2), kStride1);
  EXPECT_EQ(strides_attr->list().i(3), kStride2);

  auto* output_transpose_node = context.graph_view->GetNode(
      "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, conv2d_node->GetName(), 0);

  auto* output_node = context.graph_view->GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, output_transpose_node->GetName(), 0);
}

TEST_F(TransposerTest, MaxPoolGradTransposerTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  for (bool use_grad_grad : {false, true}) {
    GrapplerItem item;
    TransposeContext context;
    TF_ASSERT_OK(CreateSimpleMaxPoolGrad(&item.graph, use_grad_grad));
    TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
        item, virtual_cluster_.get(), &context));
    context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

    MaxPoolGradTransposer transposer;
    auto* maxpool_grad = context.graph_view->GetNode("maxpool_grad");
    ASSERT_NE(maxpool_grad, nullptr);
    TF_ASSERT_OK(transposer.TransposeNode(&context, maxpool_grad));

    auto* input_transpose_node1 = context.graph_view->GetNode(
        "maxpool_grad-0-TransposeNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(input_transpose_node1, nullptr);
    ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
    VerifyRegularFaninMatch(input_transpose_node1, 0, "orig_input", 0);

    auto* input_transpose_node2 = context.graph_view->GetNode(
        "maxpool_grad-1-TransposeNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(input_transpose_node2, nullptr);
    ASSERT_EQ(input_transpose_node2->NumRegularFanins(), 2);
    VerifyRegularFaninMatch(input_transpose_node2, 0, "orig_output", 0);

    auto* input_transpose_node3 = context.graph_view->GetNode(
        "maxpool_grad-2-TransposeNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(input_transpose_node3, nullptr);
    ASSERT_EQ(input_transpose_node3->NumRegularFanins(), 2);
    VerifyRegularFaninMatch(input_transpose_node3, 0, "grad", 0);

    auto* updated_maxpool_grad = context.graph_view->GetNode("maxpool_grad");
    VerifyDataFormatAttributeMatch(updated_maxpool_grad, kDstFormat);
    ASSERT_EQ(updated_maxpool_grad->NumRegularFanins(), 3);
    VerifyRegularFaninMatch(updated_maxpool_grad, 0,
                            input_transpose_node1->GetName(), 0);
    VerifyRegularFaninMatch(updated_maxpool_grad, 1,
                            input_transpose_node2->GetName(), 0);
    VerifyRegularFaninMatch(updated_maxpool_grad, 2,
                            input_transpose_node3->GetName(), 0);

    auto* output_transpose_node = context.graph_view->GetNode(
        "maxpool_grad-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
    ASSERT_NE(output_transpose_node, nullptr);
    ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
    VerifyRegularFaninMatch(output_transpose_node, 0,
                            updated_maxpool_grad->GetName(), 0);
  }
}

TEST_F(TransposerTest, BiasAddGradTransposerTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleBiasAddGrad(
      &item.graph, {kBatchSize, kHeight, kWidth, kDepthIn}));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  BiasAddGradTransposer transposer;
  auto* bag = context.graph_view->GetNode("bag");
  ASSERT_NE(bag, nullptr);
  TF_ASSERT_OK(transposer.TransposeNode(&context, bag));

  // The expected optimized graph contains 1 extra Transpose node and has the
  // BiasAddGrad's data_format set to "NCHW".
  auto* input_transpose_node =
      context.graph_view->GetNode("bag-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0, "input", 0);

  auto* bag_node = context.graph_view->GetNode("bag");
  ASSERT_NE(bag_node, nullptr);
  VerifyDataFormatAttributeMatch(bag_node, kDstFormat);

  auto* output_transpose_node = context.graph_view->GetNode(
      "bag-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* output_node = context.graph_view->GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, bag_node->GetName(), 0);
}

TEST_F(TransposerTest, BiasAddGradTransposerIncorrectInputTest) {
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(
      CreateSimpleBiasAddGrad(&item.graph, {kHeight, kWidth, kDepthIn}));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  BiasAddGradTransposer transposer;
  auto* bag = context.graph_view->GetNode("bag");
  ASSERT_NE(bag, nullptr);
  TF_ASSERT_OK(transposer.TransposeNode(&context, bag));

  // Optimization should not occur because of incorrect input dimensions.
  auto* input_transpose_node =
      context.graph_view->GetNode("bag-0-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node, nullptr);

  auto* bag_node = context.graph_view->GetNode("bag");
  ASSERT_NE(bag_node, nullptr);
  VerifyDataFormatAttributeMatch(bag_node, kSrcFormat);

  auto* output_transpose_node = context.graph_view->GetNode(
      "bag-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* output_node = context.graph_view->GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, bag_node->GetName(), 0);
}

TEST_F(TransposerTest, Conv2DBackpropFilterTransposerTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleConv2DBackpropFilter(&item.graph));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  Conv2DBackpropFilterTransposer transposer;
  auto* conv2d_bf = context.graph_view->GetNode("conv2d_backprop_filter");
  ASSERT_NE(conv2d_bf, nullptr);
  TF_ASSERT_OK(transposer.TransposeNode(&context, conv2d_bf));

  // The expected optimized graph contains 2 extra sets of Transpose nodes and
  // has the Conv2DBackpropFilter's data_format set to "NCHW".
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "conv2d_backprop_filter-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0, "input", 0);

  auto* input_transpose_node_filter_sizes = context.graph_view->GetNode(
      "conv2d_backprop_filter-1-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node_filter_sizes, nullptr);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "conv2d_backprop_filter-2-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node2, nullptr);
  ASSERT_EQ(input_transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node2, 0, "out_backprop", 0);

  auto* conv2d_bf_node = context.graph_view->GetNode("conv2d_backprop_filter");
  ASSERT_NE(conv2d_bf_node, nullptr);
  ASSERT_EQ(conv2d_bf_node->NumRegularFanins(), 3);
  VerifyRegularFaninMatch(conv2d_bf_node, 0, input_transpose_node1->GetName(),
                          0);
  VerifyRegularFaninMatch(conv2d_bf_node, 2, input_transpose_node2->GetName(),
                          0);
  VerifyDataFormatAttributeMatch(conv2d_bf_node, kDstFormat);

  auto* output_transpose_node = context.graph_view->GetNode(
      "conv2d_backprop_filter-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* output_node = context.graph_view->GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, conv2d_bf_node->GetName(), 0);
}

TEST_F(TransposerTest, NodeAttributes) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(
      CreateSimpleConv2DBackpropFilter(&item.graph, DT_FLOAT, "EXPLICIT"));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  Conv2DBackpropFilterTransposer transposer;
  auto* conv2d_bf = context.graph_view->GetNode("conv2d_backprop_filter");
  ASSERT_NE(conv2d_bf, nullptr);
  TF_ASSERT_OK(transposer.TransposeNode(&context, conv2d_bf));

  auto* conv2d_bf_node = context.graph_view->GetNode("conv2d_backprop_filter");
  ASSERT_NE(conv2d_bf_node, nullptr);
  ASSERT_EQ(conv2d_bf_node->NumRegularFanins(), 3);
  VerifyDataFormatAttributeMatch(conv2d_bf_node, kDstFormat);
  auto* dilations_attr = conv2d_bf_node->GetAttr("dilations");
  ASSERT_NE(dilations_attr, nullptr);
  ASSERT_EQ(dilations_attr->list().i_size(), 4);
  EXPECT_EQ(dilations_attr->list().i(0), 1);
  EXPECT_EQ(dilations_attr->list().i(1), 1);
  EXPECT_EQ(dilations_attr->list().i(2), kDilation);
  EXPECT_EQ(dilations_attr->list().i(3), kDilation);
  auto* explicit_paddings_attr = conv2d_bf_node->GetAttr("explicit_paddings");
  ASSERT_NE(explicit_paddings_attr, nullptr);
  ASSERT_EQ(explicit_paddings_attr->list().i_size(), 8);
  EXPECT_EQ(explicit_paddings_attr->list().i(0), 0);
  EXPECT_EQ(explicit_paddings_attr->list().i(1), 0);
  EXPECT_EQ(explicit_paddings_attr->list().i(2), 0);
  EXPECT_EQ(explicit_paddings_attr->list().i(3), 0);
  EXPECT_EQ(explicit_paddings_attr->list().i(4), kPaddingTop);
  EXPECT_EQ(explicit_paddings_attr->list().i(5), kPaddingBottom);
  EXPECT_EQ(explicit_paddings_attr->list().i(6), kPaddingLeft);
  EXPECT_EQ(explicit_paddings_attr->list().i(7), kPaddingRight);
}

TEST_F(TransposerTest, Conv2DBackpropInputTransposerTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleConv2DBackpropInput(&item.graph));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  Conv2DBackpropInputTransposer transposer;
  auto* conv2d_i = context.graph_view->GetNode("conv2d_backprop_input");
  ASSERT_NE(conv2d_i, nullptr);
  TF_ASSERT_OK(transposer.TransposeNode(&context, conv2d_i));

  // The expected optimized graph contains 1 extra set of Transpose nodes,
  // 1 DataFormatVecPermute node and has the Conv2DBackpropInput's data_format
  // set to "NCHW".
  auto* input_vec_permute_node = context.graph_view->GetNode(
      "conv2d_backprop_input-0-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_vec_permute_node, nullptr);
  ASSERT_EQ(input_vec_permute_node->NumRegularFanins(), 1);
  const auto* src_format_attr = input_vec_permute_node->GetAttr(kAttrSrcFormat);
  ASSERT_NE(src_format_attr, nullptr);
  EXPECT_EQ(src_format_attr->s(), kSrcFormat);
  const auto* dst_format_attr = input_vec_permute_node->GetAttr(kAttrDstFormat);
  ASSERT_NE(dst_format_attr, nullptr);
  EXPECT_EQ(dst_format_attr->s(), kDstFormat);

  auto* input_transpose_node = context.graph_view->GetNode(
      "conv2d_backprop_input-2-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0, "out_backprop", 0);

  auto* conv2d_i_node = context.graph_view->GetNode("conv2d_backprop_input");
  ASSERT_NE(conv2d_i_node, nullptr);
  ASSERT_EQ(conv2d_i_node->NumRegularFanins(), 3);
  VerifyRegularFaninMatch(conv2d_i_node, 0, input_vec_permute_node->GetName(),
                          0);
  VerifyRegularFaninMatch(conv2d_i_node, 1, "filter", 0);
  VerifyRegularFaninMatch(conv2d_i_node, 2, input_transpose_node->GetName(), 0);
  VerifyDataFormatAttributeMatch(conv2d_i_node, kDstFormat);

  auto* output_transpose_node = context.graph_view->GetNode(
      "conv2d_backprop_input-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, conv2d_i_node->GetName(),
                          0);

  auto* output_node = context.graph_view->GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, output_transpose_node->GetName(), 0);
}

TEST_F(TransposerTest, FusedBatchNormGradTransposerIsTrainingTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleFusedBatchNormGrad(&item.graph, true));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  FusedBatchNormGradTransposer transposer;
  auto* fbng = context.graph_view->GetNode("fused_batch_norm_grad");
  ASSERT_NE(fbng, nullptr);
  TF_ASSERT_OK(transposer.TransposeNode(&context, fbng));

  // The expected optimized graph contains 3 extra sets of Transpose nodes and
  // has the FusedBatchNormGrad's data_format set to "NCHW".
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "fused_batch_norm_grad-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0, "y_backprop", 0);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "fused_batch_norm_grad-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node2, nullptr);
  ASSERT_EQ(input_transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node2, 0, "x", 0);

  auto* fbng_node = context.graph_view->GetNode("fused_batch_norm_grad");
  ASSERT_NE(fbng_node, nullptr);
  ASSERT_EQ(fbng_node->NumRegularFanins(), 5);
  VerifyRegularFaninMatch(fbng_node, 0, input_transpose_node1->GetName(), 0);
  VerifyRegularFaninMatch(fbng_node, 1, input_transpose_node2->GetName(), 0);
  VerifyRegularFaninMatch(fbng_node, 2, "scale", 0);
  VerifyRegularFaninMatch(fbng_node, 3, "reserve_space_1", 0);
  VerifyRegularFaninMatch(fbng_node, 4, "reserve_space_2", 0);
  VerifyDataFormatAttributeMatch(fbng_node, kDstFormat);

  auto* output_transpose_node = context.graph_view->GetNode(
      "fused_batch_norm_grad-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, fbng_node->GetName(), 0);

  auto* x_backprop = context.graph_view->GetNode("x_backprop");
  ASSERT_NE(x_backprop, nullptr);
  ASSERT_EQ(x_backprop->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(x_backprop, 0, output_transpose_node->GetName(), 0);

  auto* scale_backprop = context.graph_view->GetNode("scale_backprop");
  ASSERT_NE(scale_backprop, nullptr);
  ASSERT_EQ(scale_backprop->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(scale_backprop, 0, fbng_node->GetName(), 1);

  auto* offset_backprop = context.graph_view->GetNode("offset_backprop");
  ASSERT_NE(offset_backprop, nullptr);
  ASSERT_EQ(offset_backprop->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(offset_backprop, 0, fbng_node->GetName(), 2);

  auto* reserve_space_3 = context.graph_view->GetNode("reserve_space_3");
  ASSERT_NE(reserve_space_3, nullptr);
  ASSERT_EQ(reserve_space_3->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(reserve_space_3, 0, fbng_node->GetName(), 3);

  auto* reserve_space_4 = context.graph_view->GetNode("reserve_space_4");
  ASSERT_NE(reserve_space_4, nullptr);
  ASSERT_EQ(reserve_space_4->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(reserve_space_4, 0, fbng_node->GetName(), 4);
}

TEST_F(TransposerTest, FusedBatchNormGradTransposerNotTrainingTest) {
  GrapplerItem item;
  TransposeContext context;
  TF_ASSERT_OK(CreateSimpleFusedBatchNormGrad(&item.graph, false));
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  FusedBatchNormGradTransposer transposer;
  auto* fbng = context.graph_view->GetNode("fused_batch_norm_grad");
  ASSERT_NE(fbng, nullptr);
  TF_ASSERT_OK(transposer.TransposeNode(&context, fbng));

  // Optimization should not occur because FusedBatchNormGrad is not set to
  // training.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "fused_batch_norm_grad-0-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node1, nullptr);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "fused_batch_norm_grad-1-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node2, nullptr);

  auto* fbng_node = context.graph_view->GetNode("fused_batch_norm_grad");
  ASSERT_NE(fbng_node, nullptr);
  ASSERT_EQ(fbng_node->NumRegularFanins(), 5);
  VerifyRegularFaninMatch(fbng_node, 0, "y_backprop", 0);
  VerifyRegularFaninMatch(fbng_node, 1, "x", 0);
  VerifyRegularFaninMatch(fbng_node, 2, "scale", 0);
  VerifyRegularFaninMatch(fbng_node, 3, "reserve_space_1", 0);
  VerifyRegularFaninMatch(fbng_node, 4, "reserve_space_2", 0);
  VerifyDataFormatAttributeMatch(fbng_node, kSrcFormat);

  auto* output_transpose_node = context.graph_view->GetNode(
      "fused_batch_norm_grad-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* x_backprop = context.graph_view->GetNode("x_backprop");
  ASSERT_NE(x_backprop, nullptr);
  ASSERT_EQ(x_backprop->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(x_backprop, 0, fbng_node->GetName(), 0);

  auto* scale_backprop = context.graph_view->GetNode("scale_backprop");
  ASSERT_NE(scale_backprop, nullptr);
  ASSERT_EQ(scale_backprop->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(scale_backprop, 0, fbng_node->GetName(), 1);

  auto* offset_backprop = context.graph_view->GetNode("offset_backprop");
  ASSERT_NE(offset_backprop, nullptr);
  ASSERT_EQ(offset_backprop->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(offset_backprop, 0, fbng_node->GetName(), 2);

  auto* reserve_space_3 = context.graph_view->GetNode("reserve_space_3");
  ASSERT_NE(reserve_space_3, nullptr);
  ASSERT_EQ(reserve_space_3->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(reserve_space_3, 0, fbng_node->GetName(), 3);

  auto* reserve_space_4 = context.graph_view->GetNode("reserve_space_4");
  ASSERT_NE(reserve_space_4, nullptr);
  ASSERT_EQ(reserve_space_4->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(reserve_space_4, 0, fbng_node->GetName(), 4);
}

TEST_F(TransposerTest, DefaultLayoutAgnosticOpTransposerIdentityTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto conv2d = SimpleConv2D(&scope);
  auto identity = ops::Identity(
      scope.WithOpName("identity").WithDevice("/device:GPU:0"), conv2d);
  auto output = ops::Identity(scope.WithOpName("output"), identity);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  DefaultLayoutAgnosticOpTransposer transposer;
  auto* i = context.graph_view->GetNode("identity");
  ASSERT_NE(i, nullptr);
  TF_ASSERT_OK(transposer.TransposeNode(&context, i));

  // The expected optimized graph contains 2 extra sets of Transpose nodes.
  auto* input_transpose_node = context.graph_view->GetNode(
      "identity-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* i_node = context.graph_view->GetNode("identity");
  ASSERT_NE(i_node, nullptr);
  ASSERT_EQ(i_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(i_node, 0, input_transpose_node->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "identity-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, i_node->GetName(), 0);

  auto* output_node = context.graph_view->GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, output_transpose_node->GetName(), 0);
}

TEST_F(TransposerTest, DefaultLayoutAgnosticOpTransposerIdentityBadInputTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto conv2d = SimpleConv2D(&scope);
  auto sum = ops::Sum(scope.WithOpName("sum"), conv2d, {0, 1});
  auto identity = ops::Identity(
      scope.WithOpName("identity").WithDevice("/device:GPU:0"), sum);
  auto output = ops::Identity(scope.WithOpName("output"), identity);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  DefaultLayoutAgnosticOpTransposer transposer;
  auto* i = context.graph_view->GetNode("identity");
  ASSERT_NE(i, nullptr);
  TF_ASSERT_OK(transposer.TransposeNode(&context, i));

  // Optimization should not occur because input is not the right shape (needs
  // to be 4D).
  auto* input_transpose_node = context.graph_view->GetNode(
      "identity-0-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node, nullptr);

  auto* i_node = context.graph_view->GetNode("identity");
  ASSERT_NE(i_node, nullptr);
  ASSERT_EQ(i_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(i_node, 0, "sum", 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "identity-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* output_node = context.graph_view->GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, i_node->GetName(), 0);
}

TEST_F(TransposerTest, AddNTransposerTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TF_ASSERT_OK(CreateSimpleAddN(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* conv2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(conv2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, conv2d));

  AddNTransposer addn_transposer;
  auto* an = context.graph_view->GetNode("add_n");
  ASSERT_NE(an, nullptr);
  TF_ASSERT_OK(addn_transposer.TransposeNode(&context, an));

  // The expected optimized graph contains 5 extra sets of Transpose nodes.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "add_n-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0, "a", 0);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "add_n-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node2, nullptr);
  ASSERT_EQ(input_transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node2, 0, "b", 0);

  auto* input_transpose_node3 = context.graph_view->GetNode(
      "add_n-2-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node3, nullptr);
  ASSERT_EQ(input_transpose_node3->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node3, 0, "c", 0);

  auto* input_transpose_node4 = context.graph_view->GetNode(
      "add_n-3-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node4, nullptr);
  ASSERT_EQ(input_transpose_node4->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node4, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* an_node = context.graph_view->GetNode("add_n");
  ASSERT_NE(an_node, nullptr);
  ASSERT_EQ(an_node->NumRegularFanins(), 4);
  VerifyRegularFaninMatch(an_node, 0, input_transpose_node1->GetName(), 0);
  VerifyRegularFaninMatch(an_node, 1, input_transpose_node2->GetName(), 0);
  VerifyRegularFaninMatch(an_node, 2, input_transpose_node3->GetName(), 0);
  VerifyRegularFaninMatch(an_node, 3, input_transpose_node4->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "add_n-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, an_node->GetName(), 0);

  auto* output_node = context.graph_view->GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, output_transpose_node->GetName(), 0);
}

TEST_F(TransposerTest, AddNTransposerNotAfterTransformTest) {
  GrapplerItem item;
  TF_ASSERT_OK(CreateSimpleAddN(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  AddNTransposer addn_transposer;
  auto* an = context.graph_view->GetNode("add_n");
  ASSERT_NE(an, nullptr);
  TF_ASSERT_OK(addn_transposer.TransposeNode(&context, an));

  // Optimization should not occur because AddN does not follow a transform.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "add_n-0-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node1, nullptr);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "add_n-1-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node2, nullptr);

  auto* input_transpose_node3 = context.graph_view->GetNode(
      "add_n-2-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node3, nullptr);

  auto* input_transpose_node4 = context.graph_view->GetNode(
      "add_n-3-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node4, nullptr);

  auto* an_node = context.graph_view->GetNode("add_n");
  ASSERT_NE(an_node, nullptr);
  ASSERT_EQ(an_node->NumRegularFanins(), 4);
  VerifyRegularFaninMatch(an_node, 0, "a", 0);
  VerifyRegularFaninMatch(an_node, 1, "b", 0);
  VerifyRegularFaninMatch(an_node, 2, "c", 0);
  VerifyRegularFaninMatch(an_node, 3, "conv2d", 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "add_n-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* output_node = context.graph_view->GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, an_node->GetName(), 0);
}

TEST_F(TransposerTest, IdentityNTransposerTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  TF_ASSERT_OK(CreateSimpleIdentityN(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* conv2d_1 = context.graph_view->GetNode("conv2d_1");
  ASSERT_NE(conv2d_1, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, conv2d_1));
  auto* conv2d_2 = context.graph_view->GetNode("conv2d_2");
  ASSERT_NE(conv2d_2, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, conv2d_2));

  IdentityNTransposer identityn_transposer;
  auto* in = context.graph_view->GetNode("identity_n");
  ASSERT_NE(in, nullptr);
  TF_ASSERT_OK(identityn_transposer.TransposeNode(&context, in));

  // The expected optimized graph contains 4 extra sets of Transpose nodes.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "identity_n-0-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node1, nullptr);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "identity_n-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node2, nullptr);
  ASSERT_EQ(input_transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node2, 0,
                          "conv2d_2-0-0-TransposeNCHWToNHWC-LayoutOptimizer",
                          0);

  auto* input_transpose_node3 = context.graph_view->GetNode(
      "identity_n-2-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node3, nullptr);

  auto* input_transpose_node4 = context.graph_view->GetNode(
      "identity_n-3-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node4, nullptr);

  auto* in_node = context.graph_view->GetNode("identity_n");
  ASSERT_NE(in_node, nullptr);
  ASSERT_EQ(in_node->NumRegularFanins(), 4);
  VerifyRegularFaninMatch(in_node, 0, "conv2d_1", 0);
  VerifyRegularFaninMatch(in_node, 1, input_transpose_node2->GetName(), 0);
  VerifyRegularFaninMatch(in_node, 2, "a", 0);
  VerifyRegularFaninMatch(in_node, 3, "b", 0);

  auto* output_transpose_node1 = context.graph_view->GetNode(
      "identity_n-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node1, nullptr);

  auto* output_transpose_node2 = context.graph_view->GetNode(
      "identity_n-1-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node2, nullptr);
  ASSERT_EQ(output_transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node2, 0, in_node->GetName(), 1);

  auto* output_transpose_node3 = context.graph_view->GetNode(
      "identity_n-2-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node3, nullptr);

  auto* output_transpose_node4 = context.graph_view->GetNode(
      "identity_n-3-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node4, nullptr);

  auto* conv2d_1_output_node = context.graph_view->GetNode("conv2d_1_output");
  ASSERT_NE(conv2d_1_output_node, nullptr);
  ASSERT_EQ(conv2d_1_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(conv2d_1_output_node, 0, in_node->GetName(), 0);

  auto* conv2d_2_output_node = context.graph_view->GetNode("conv2d_2_output");
  ASSERT_NE(conv2d_2_output_node, nullptr);
  ASSERT_EQ(conv2d_2_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(conv2d_2_output_node, 0,
                          output_transpose_node2->GetName(), 0);

  auto* a_output_node = context.graph_view->GetNode("a_output");
  ASSERT_NE(a_output_node, nullptr);
  ASSERT_EQ(a_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(a_output_node, 0, in_node->GetName(), 2);

  auto* b_output_node = context.graph_view->GetNode("b_output");
  ASSERT_NE(b_output_node, nullptr);
  ASSERT_EQ(b_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(b_output_node, 0, in_node->GetName(), 3);
}

TEST_F(TransposerTest, MergeTransposerTestMergeBothInputsConvertible) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto conv2d = SimpleConv2D(&scope);
  Output i1 = ops::Identity(scope.WithOpName("i1"), conv2d);
  auto merge = ops::Merge(scope.WithOpName("merge").WithDevice("/device:GPU:0"),
                          {conv2d, i1});
  auto i2 = ops::Identity(scope.WithOpName("i2"), merge.output);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  MergeTransposer merge_transposer;
  auto* m = context.graph_view->GetNode("merge");
  ASSERT_NE(m, nullptr);
  TF_ASSERT_OK(merge_transposer.TransposeNode(&context, m));

  // The expected optimized graph contains 3 extra sets of Transpose nodes.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "merge-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0,
                          "conv2d-0-1-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "merge-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node2, nullptr);
  ASSERT_EQ(input_transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node2, 0, "i1", 0);

  auto* m_node = context.graph_view->GetNode("merge");
  ASSERT_NE(m_node, nullptr);
  ASSERT_EQ(m_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(m_node, 0, input_transpose_node1->GetName(), 0);
  VerifyRegularFaninMatch(m_node, 1, input_transpose_node2->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "merge-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, m_node->GetName(), 0);

  auto* output_node = context.graph_view->GetNode("i2");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, output_transpose_node->GetName(), 0);
}

TEST_F(TransposerTest, MergeTransposerTestMergeOneInputNotConvertible) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto conv2d = SimpleConv2D(&scope);
  auto tensor_4d =
      ops::Const(scope.WithOpName("tensor_4d"), 3.0f, {1, 1, 1, 3});
  auto merge = ops::Merge(scope.WithOpName("merge").WithDevice("/device:GPU:0"),
                          {conv2d, tensor_4d});
  auto i2 = ops::Identity(scope.WithOpName("i2"), merge.output);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  MergeTransposer merge_transposer;
  auto* m = context.graph_view->GetNode("merge");
  ASSERT_NE(m, nullptr);
  TF_ASSERT_OK(merge_transposer.TransposeNode(&context, m));

  // Optimization should not occur because not every input is a transform or
  // after transform.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "merge-0-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node1, nullptr);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "merge-1-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node2, nullptr);

  auto* m_node = context.graph_view->GetNode("merge");
  ASSERT_NE(m_node, nullptr);
  ASSERT_EQ(m_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(m_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);
  VerifyRegularFaninMatch(m_node, 1, "tensor_4d", 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "merge-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* output_node = context.graph_view->GetNode("i2");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, m_node->GetName(), 0);
}

TEST_F(TransposerTest, PadTransposerTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto conv2d = SimpleConv2D(&scope);
  auto c = ops::Const(scope.WithOpName("c"), {1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
  auto p =
      ops::Pad(scope.WithOpName("p").WithDevice("/device:GPU:0"), conv2d, c);
  auto o = ops::Identity(scope.WithOpName("o"), p);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  PadTransposer pad_transposer;
  auto* pad = context.graph_view->GetNode("p");
  ASSERT_NE(pad, nullptr);
  TF_ASSERT_OK(pad_transposer.TransposeNode(&context, pad));

  // The expected optimized graph contains 2 extra sets of Transpose nodes and 1
  // DataFormatVecPermute node.
  auto* input_transpose_node =
      context.graph_view->GetNode("p-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* padding_transpose_node = context.graph_view->GetNode(
      "p-1-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(padding_transpose_node, nullptr);
  ASSERT_EQ(padding_transpose_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(padding_transpose_node, 0, "c", 0);

  auto* pad_node = context.graph_view->GetNode("p");
  ASSERT_NE(pad_node, nullptr);
  ASSERT_EQ(pad_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(pad_node, 0, input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(pad_node, 1, padding_transpose_node->GetName(), 0);

  auto* output_transpose_node =
      context.graph_view->GetNode("p-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, pad_node->GetName(), 0);

  auto* output_node = context.graph_view->GetNode("o");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, output_transpose_node->GetName(), 0);
}

TEST_F(TransposerTest, SwitchTransposerTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto conv2d = SimpleConv2D(&scope);
  ops::Variable ctrl(scope.WithOpName("ctrl"), {}, DT_BOOL);
  auto sw = ops::Switch(scope.WithOpName("switch").WithDevice("/device:GPU:0"),
                        conv2d, ctrl);
  auto i1 = ops::Identity(scope.WithOpName("i1"), sw.output_false);
  auto i2 = ops::Identity(scope.WithOpName("i2"), sw.output_true);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SwitchTransposer switch_transposer;
  auto* sw_node = context.graph_view->GetNode("switch");
  ASSERT_NE(sw_node, nullptr);
  TF_ASSERT_OK(switch_transposer.TransposeNode(&context, sw_node));

  // The expected optimized graph contains 3 extra sets of Transpose nodes.
  auto* input_transpose_node = context.graph_view->GetNode(
      "switch-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* switch_node = context.graph_view->GetNode("switch");
  ASSERT_NE(switch_node, nullptr);
  ASSERT_EQ(switch_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(switch_node, 0, input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(switch_node, 1, "ctrl", 0);

  auto* output_transpose_node1 = context.graph_view->GetNode(
      "switch-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node1, nullptr);
  ASSERT_EQ(output_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node1, 0, switch_node->GetName(), 0);

  auto* output_transpose_node2 = context.graph_view->GetNode(
      "switch-1-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node2, nullptr);
  ASSERT_EQ(output_transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node2, 0, switch_node->GetName(), 1);

  auto* i1_node = context.graph_view->GetNode("i1");
  ASSERT_NE(i1_node, nullptr);
  ASSERT_EQ(i1_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(i1_node, 0, output_transpose_node1->GetName(), 0);

  auto* i2_node = context.graph_view->GetNode("i2");
  ASSERT_NE(i2_node, nullptr);
  ASSERT_EQ(i2_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(i2_node, 0, output_transpose_node2->GetName(), 0);
}

TEST_F(TransposerTest, TernaryOpTransposerTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto conv2d = SimpleConv2D(&scope);
  auto a = ops::RandomUniform(scope.WithOpName("a"),
                              {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  auto b = ops::RandomUniform(scope.WithOpName("b"),
                              {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  auto beta_inc = ops::Betainc(
      scope.WithOpName("beta_inc").WithDevice("/device:GPU:0"), a, b, conv2d);
  auto z = ops::Identity(scope.WithOpName("z"), beta_inc);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  TernaryOpTransposer ternary_op_transposer;
  auto* bi = context.graph_view->GetNode("beta_inc");
  ASSERT_NE(bi, nullptr);
  TF_ASSERT_OK(ternary_op_transposer.TransposeNode(&context, bi));

  // The expected optimized graph contains 4 extra sets of Transpose nodes.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "beta_inc-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0, "a", 0);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "beta_inc-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node2, nullptr);
  ASSERT_EQ(input_transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node2, 0, "b", 0);

  auto* input_transpose_node3 = context.graph_view->GetNode(
      "beta_inc-2-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node3, nullptr);
  ASSERT_EQ(input_transpose_node3->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node3, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* bi_node = context.graph_view->GetNode("beta_inc");
  ASSERT_NE(bi_node, nullptr);
  ASSERT_EQ(bi_node->NumRegularFanins(), 3);
  VerifyRegularFaninMatch(bi_node, 0, input_transpose_node1->GetName(), 0);
  VerifyRegularFaninMatch(bi_node, 1, input_transpose_node2->GetName(), 0);
  VerifyRegularFaninMatch(bi_node, 2, input_transpose_node3->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "beta_inc-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, bi_node->GetName(), 0);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, UnaryGradTransposerTestTanhGrad) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto conv2d = SimpleConv2D(&scope);
  auto a = ops::RandomUniform(scope.WithOpName("a"),
                              {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  auto tanh_grad_op = ops::internal::TanhGrad(
      scope.WithOpName("tanh_grad").WithDevice("/device:GPU:0"), conv2d, a);
  auto z = ops::Identity(scope.WithOpName("z"), tanh_grad_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  UnaryGradTransposer unary_grad_transposer;
  auto* tanh_grad = context.graph_view->GetNode("tanh_grad");
  ASSERT_NE(tanh_grad, nullptr);
  TF_ASSERT_OK(unary_grad_transposer.TransposeNode(&context, tanh_grad));

  // The expected optimized graph contains 4 extra sets of Transpose nodes.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "tanh_grad-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "tanh_grad-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node2, nullptr);
  ASSERT_EQ(input_transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node2, 0, "a", 0);

  auto* tanh_grad_node = context.graph_view->GetNode("tanh_grad");
  ASSERT_NE(tanh_grad_node, nullptr);
  ASSERT_EQ(tanh_grad_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(tanh_grad_node, 0, input_transpose_node1->GetName(),
                          0);
  VerifyRegularFaninMatch(tanh_grad_node, 1, input_transpose_node2->GetName(),
                          0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "tanh_grad-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, tanh_grad_node->GetName(),
                          0);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, UnaryGradTransposerTestRelu6Grad) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto conv2d = SimpleConv2D(&scope);
  auto a = ops::RandomUniform(scope.WithOpName("a"),
                              {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  auto relu6_grad_op = ops::internal::SigmoidGrad(
      scope.WithOpName("relu6_grad").WithDevice("/device:GPU:0"), conv2d, a);
  auto z = ops::Identity(scope.WithOpName("z"), relu6_grad_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  UnaryGradTransposer unary_grad_transposer;
  auto* relu6_grad = context.graph_view->GetNode("relu6_grad");
  ASSERT_NE(relu6_grad, nullptr);
  TF_ASSERT_OK(unary_grad_transposer.TransposeNode(&context, relu6_grad));

  // The expected optimized graph contains 4 extra sets of Transpose nodes.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "relu6_grad-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* input_transpose_node2 = context.graph_view->GetNode(
      "relu6_grad-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node2, nullptr);
  ASSERT_EQ(input_transpose_node2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node2, 0, "a", 0);

  auto* relu6_grad_node = context.graph_view->GetNode("relu6_grad");
  ASSERT_NE(relu6_grad_node, nullptr);
  ASSERT_EQ(relu6_grad_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(relu6_grad_node, 0, input_transpose_node1->GetName(),
                          0);
  VerifyRegularFaninMatch(relu6_grad_node, 1, input_transpose_node2->GetName(),
                          0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "relu6_grad-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, relu6_grad_node->GetName(),
                          0);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, SqueezeTransposerTest) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"), {32, 1, 1, 8}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"), {1, 1, 8, 16}, DT_FLOAT);
  auto conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 1, 1, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

  auto squeeze_op = ops::Squeeze(
      scope.WithOpName("squeeze").WithDevice("/device:GPU:0"), conv2d);
  auto z = ops::Identity(scope.WithOpName("z"), squeeze_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SqueezeTransposer squeeze_transposer;
  auto* squeeze = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze, nullptr);
  TF_ASSERT_OK(squeeze_transposer.TransposeNode(&context, squeeze));

  auto* input_transpose_node1 = context.graph_view->GetNode(
      "squeeze-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* squeeze_node = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze_node, nullptr);
  ASSERT_EQ(squeeze_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(squeeze_node, 0, input_transpose_node1->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "squeeze-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, squeeze_node->GetName(), 0);
}

TEST_F(TransposerTest, SqueezeTransposerTestUnsupportedInputShape) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"), {32, 5, 5, 8}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"), {5, 5, 8, 16}, DT_FLOAT);
  auto conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 1, 1, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

  auto squeeze_op = ops::Squeeze(
      scope.WithOpName("squeeze").WithDevice("/device:GPU:0"), conv2d);
  auto z = ops::Identity(scope.WithOpName("z"), squeeze_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SqueezeTransposer squeeze_transposer;
  auto* squeeze = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze, nullptr);
  TF_ASSERT_OK(squeeze_transposer.TransposeNode(&context, squeeze));

  // Expect no changes to the input edge.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "squeeze-0-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node1, nullptr);
}

TEST_F(TransposerTest, SqueezeTransposerTestInvalidHWAxis) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"), {32, 1, 1, 8}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"), {1, 1, 8, 16}, DT_FLOAT);
  auto conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 1, 1, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

  auto squeeze_op =
      ops::Squeeze(scope.WithOpName("squeeze").WithDevice("/device:GPU:0"),
                   conv2d, ops::Squeeze::Attrs().Axis({1}));
  auto z = ops::Identity(scope.WithOpName("z"), squeeze_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SqueezeTransposer squeeze_transposer;
  auto* squeeze = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze, nullptr);
  TF_ASSERT_OK(squeeze_transposer.TransposeNode(&context, squeeze));

  // Expect no changes to the input edge.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "squeeze-0-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node1, nullptr);
}

TEST_F(TransposerTest, SqueezeTransposerTestInvalidNHWAxis) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"), {32, 1, 1, 8}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"), {1, 1, 8, 1}, DT_FLOAT);
  auto conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 1, 1, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

  auto squeeze_op =
      ops::Squeeze(scope.WithOpName("squeeze").WithDevice("/device:GPU:0"),
                   conv2d, ops::Squeeze::Attrs().Axis({1, 2, 3}));
  auto z = ops::Identity(scope.WithOpName("z"), squeeze_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SqueezeTransposer squeeze_transposer;
  auto* squeeze = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze, nullptr);
  TF_ASSERT_OK(squeeze_transposer.TransposeNode(&context, squeeze));

  // Expect no changes to the input edge.
  auto* input_transpose_node1 = context.graph_view->GetNode(
      "squeeze-0-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(input_transpose_node1, nullptr);
}

TEST_F(TransposerTest, SqueezeTransposerTestSqueezeDimsUpdated) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"), {1, 1, 1, 8}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"), {1, 1, 8, 1}, DT_FLOAT);
  auto conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 1, 1, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

  auto squeeze_op =
      ops::Squeeze(scope.WithOpName("squeeze").WithDevice("/device:GPU:0"),
                   conv2d, ops::Squeeze::Attrs().Axis({1, 2}));
  auto z = ops::Identity(scope.WithOpName("z"), squeeze_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SqueezeTransposer squeeze_transposer;
  auto* squeeze = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze, nullptr);
  TF_ASSERT_OK(squeeze_transposer.TransposeNode(&context, squeeze));

  auto* input_transpose_node1 = context.graph_view->GetNode(
      "squeeze-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* squeeze_node = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze_node, nullptr);
  ASSERT_EQ(squeeze_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(squeeze_node, 0, input_transpose_node1->GetName(), 0);
  const auto* squeeze_dims_attr = squeeze_node->GetAttr("squeeze_dims");
  const auto& list = squeeze_dims_attr->list();
  ASSERT_EQ(list.i_size(), 2);
  EXPECT_EQ(list.i(0), 2);
  EXPECT_EQ(list.i(1), 3);

  auto* output_transpose_node = context.graph_view->GetNode(
      "squeeze-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, squeeze_node->GetName(), 0);
}

// Same as SqueezeTransposerTestSqueezeDimsUpdated but with squeeze dims
// specified with negative values.
TEST_F(TransposerTest, SqueezeTransposerTestNegativeSqueezeDimsUpdated) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"), {1, 1, 1, 8}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"), {1, 1, 8, 1}, DT_FLOAT);
  auto conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 1, 1, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

  auto squeeze_op =
      ops::Squeeze(scope.WithOpName("squeeze").WithDevice("/device:GPU:0"),
                   conv2d, ops::Squeeze::Attrs().Axis({-3, -2}));
  auto z = ops::Identity(scope.WithOpName("z"), squeeze_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SqueezeTransposer squeeze_transposer;
  auto* squeeze = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze, nullptr);
  TF_ASSERT_OK(squeeze_transposer.TransposeNode(&context, squeeze));

  auto* input_transpose_node1 = context.graph_view->GetNode(
      "squeeze-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* squeeze_node = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze_node, nullptr);
  ASSERT_EQ(squeeze_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(squeeze_node, 0, input_transpose_node1->GetName(), 0);
  const auto* squeeze_dims_attr = squeeze_node->GetAttr("squeeze_dims");
  const auto& list = squeeze_dims_attr->list();
  ASSERT_EQ(list.i_size(), 2);
  EXPECT_EQ(list.i(0), 2);
  EXPECT_EQ(list.i(1), 3);

  auto* output_transpose_node = context.graph_view->GetNode(
      "squeeze-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, squeeze_node->GetName(), 0);
}

// Same as SqueezeTransposerTestSqueezeDimsUpdated but with the source and
// destination formats swapped (as is used in some cases when the data type is
// DT_HALF).
TEST_F(TransposerTest, SqueezeTransposerTestNCHWToNHWCSqueezeDimsUpdated) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"), {1, 8, 1, 1}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"), {1, 1, 8, 1}, DT_FLOAT);
  auto conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 1, 1, 1}, "SAME", ops::Conv2D::DataFormat(kDstFormat));

  auto squeeze_op =
      ops::Squeeze(scope.WithOpName("squeeze").WithDevice("/device:GPU:0"),
                   conv2d, ops::Squeeze::Attrs().Axis({2, 3}));
  auto z = ops::Identity(scope.WithOpName("z"), squeeze_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kDstFormat, kSrcFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SqueezeTransposer squeeze_transposer;
  auto* squeeze = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze, nullptr);
  TF_ASSERT_OK(squeeze_transposer.TransposeNode(&context, squeeze));

  auto* input_transpose_node1 = context.graph_view->GetNode(
      "squeeze-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0,
                          "conv2d-0-0-TransposeNHWCToNCHW-LayoutOptimizer", 0);

  auto* squeeze_node = context.graph_view->GetNode("squeeze");
  ASSERT_NE(squeeze_node, nullptr);
  ASSERT_EQ(squeeze_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(squeeze_node, 0, input_transpose_node1->GetName(), 0);
  const auto* squeeze_dims_attr = squeeze_node->GetAttr("squeeze_dims");
  const auto& list = squeeze_dims_attr->list();
  ASSERT_EQ(list.i_size(), 2);
  EXPECT_EQ(list.i(0), 1);
  EXPECT_EQ(list.i(1), 2);

  auto* output_transpose_node = context.graph_view->GetNode(
      "squeeze-0-0-TransposeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, squeeze_node->GetName(), 0);
}

TEST_F(TransposerTest, MaxPoolV2Transposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kWidth, kHeight, kDepthIn}, DT_FLOAT);
  auto ksize = ops::Const(scope.WithOpName("ksize"), {1, kKernel, kKernel, 1});
  auto strides =
      ops::Const(scope.WithOpName("strides"), {1, kKernel, kKernel, 1});
  auto maxpool_op =
      ops::MaxPoolV2(scope.WithOpName("maxpoolv2").WithDevice("/device:GPU:0"),
                     input, ksize, strides, "VALID");
  auto z = ops::Identity(scope.WithOpName("z"), maxpool_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  MaxPoolV2Transposer maxpool_transposer;
  auto* maxpool = context.graph_view->GetNode("maxpoolv2");
  ASSERT_NE(maxpool, nullptr);
  TF_ASSERT_OK(maxpool_transposer.TransposeNode(&context, maxpool));

  auto* input_transpose_node1 = context.graph_view->GetNode(
      "maxpoolv2-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  auto* input_transpose_node2 = context.graph_view->GetNode(
      "maxpoolv2-1-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node2, nullptr);
  auto* input_transpose_node3 = context.graph_view->GetNode(
      "maxpoolv2-2-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node3, nullptr);

  auto* updated_maxpool = context.graph_view->GetNode("maxpoolv2");
  ASSERT_NE(updated_maxpool, nullptr);
  ASSERT_EQ(updated_maxpool->NumRegularFanins(), 3);
  VerifyRegularFaninMatch(updated_maxpool, 0, input_transpose_node1->GetName(),
                          0);
  VerifyRegularFaninMatch(updated_maxpool, 1, input_transpose_node2->GetName(),
                          0);
  VerifyRegularFaninMatch(updated_maxpool, 2, input_transpose_node3->GetName(),
                          0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "maxpoolv2-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, MaxPoolGradV2Transposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  for (bool use_grad_grad : {false, true}) {
    GrapplerItem item;
    Scope scope = Scope::NewRootScope();
    auto orig_input =
        ops::RandomUniform(scope.WithOpName("orig_input"),
                           {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
    auto orig_output =
        ops::RandomUniform(scope.WithOpName("orig_output"),
                           {kBatchSize, use_grad_grad ? kOutHeight : kHeight,
                            use_grad_grad ? kOutWidth : kWidth, kDepthIn},
                           DT_FLOAT);
    auto grad =
        ops::RandomUniform(scope.WithOpName("grad_input"),
                           {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
    auto ksize =
        ops::Const(scope.WithOpName("ksize"), {1, kKernel, kKernel, 1});
    auto strides =
        ops::Const(scope.WithOpName("strides"), {1, kKernel, kKernel, 1});
    Output maxpoolgrad_op;
    if (use_grad_grad) {
      maxpoolgrad_op = ops::MaxPoolGradGradV2(
          scope.WithOpName("maxpoolgradv2").WithDevice("/device:GPU:0"),
          orig_input, orig_output, grad, ksize, strides, "VALID");
    } else {
      maxpoolgrad_op = ops::MaxPoolGradV2(
          scope.WithOpName("maxpoolgradv2").WithDevice("/device:GPU:0"),
          orig_input, orig_output, grad, ksize, strides, "VALID");
    }
    auto z = ops::Identity(scope.WithOpName("z"), maxpoolgrad_op);
    TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
    TransposeContext context;
    TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
        item, virtual_cluster_.get(), &context));
    context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

    MaxPoolGradV2Transposer maxpoolgrad_transposer;
    auto* maxpoolgrad = context.graph_view->GetNode("maxpoolgradv2");
    ASSERT_NE(maxpoolgrad, nullptr);
    TF_ASSERT_OK(maxpoolgrad_transposer.TransposeNode(&context, maxpoolgrad));

    auto* orig_input_transpose_node = context.graph_view->GetNode(
        "maxpoolgradv2-0-TransposeNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(orig_input_transpose_node, nullptr);
    auto* orig_output_transpose_node = context.graph_view->GetNode(
        "maxpoolgradv2-1-TransposeNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(orig_output_transpose_node, nullptr);
    auto* grad_input_transpose_node = context.graph_view->GetNode(
        "maxpoolgradv2-2-TransposeNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(grad_input_transpose_node, nullptr);
    auto* size_node = context.graph_view->GetNode(
        "maxpoolgradv2-3-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(size_node, nullptr);
    auto* stride_node = context.graph_view->GetNode(
        "maxpoolgradv2-4-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
    ASSERT_NE(stride_node, nullptr);

    auto* updated_maxpoolgrad = context.graph_view->GetNode("maxpoolgradv2");
    ASSERT_NE(updated_maxpoolgrad, nullptr);
    ASSERT_EQ(updated_maxpoolgrad->NumRegularFanins(), 5);
    VerifyRegularFaninMatch(updated_maxpoolgrad, 0,
                            orig_input_transpose_node->GetName(), 0);
    VerifyRegularFaninMatch(updated_maxpoolgrad, 1,
                            orig_output_transpose_node->GetName(), 0);
    VerifyRegularFaninMatch(updated_maxpoolgrad, 2,
                            grad_input_transpose_node->GetName(), 0);
    VerifyRegularFaninMatch(updated_maxpoolgrad, 3, size_node->GetName(), 0);
    VerifyRegularFaninMatch(updated_maxpoolgrad, 4, stride_node->GetName(), 0);

    auto* output_transpose_node = context.graph_view->GetNode(
        "maxpoolgradv2-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
    ASSERT_NE(output_transpose_node, nullptr);

    auto* z_output_node = context.graph_view->GetNode("z");
    ASSERT_NE(z_output_node, nullptr);
    ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
    VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                            0);
  }
}

TEST_F(TransposerTest, BinaryOpTransposerAdd) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  auto conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto a = ops::RandomUniform(scope.WithOpName("a"), {1}, DT_FLOAT);
  auto add =
      ops::Add(scope.WithOpName("Add").WithDevice("/device:GPU:0"), a, conv2d);
  auto z = ops::Identity(scope.WithOpName("z"), add);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  auto* addop = context.graph_view->GetNode("Add");
  ASSERT_NE(addop, nullptr);
  BinaryOpTransposer binaryop_transposer;
  TF_ASSERT_OK(binaryop_transposer.TransposeNode(&context, addop));

  auto* input_const_node =
      context.graph_view->GetNode("Add-0-ReshapeConst-LayoutOptimizer");
  ASSERT_NE(input_const_node, nullptr);
  EXPECT_EQ(input_const_node->NumRegularFanins(), 0);

  auto* input_reshape_node =
      context.graph_view->GetNode("Add-0-ReshapeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_reshape_node, nullptr);
  ASSERT_EQ(input_reshape_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_reshape_node, 0, "a", 0);
  VerifyRegularFaninMatch(input_reshape_node, 1, input_const_node->GetName(),
                          0);

  auto* input_transpose_node =
      context.graph_view->GetNode("Add-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* updated_add = context.graph_view->GetNode("Add");
  ASSERT_NE(updated_add, nullptr);
  ASSERT_EQ(updated_add->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(updated_add, 0, input_reshape_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_add, 1, input_transpose_node->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "Add-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, BinaryOpTransposerMul) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  auto conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto a = ops::RandomUniform(scope.WithOpName("a"), {1}, DT_FLOAT);
  auto mul =
      ops::Mul(scope.WithOpName("Mul").WithDevice("/device:GPU:0"), conv2d, a);
  auto z = ops::Identity(scope.WithOpName("z"), mul);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  auto* mulop = context.graph_view->GetNode("Mul");
  ASSERT_NE(mulop, nullptr);
  BinaryOpTransposer binaryop_transposer;
  TF_ASSERT_OK(binaryop_transposer.TransposeNode(&context, mulop));

  auto* input_const_node =
      context.graph_view->GetNode("Mul-1-ReshapeConst-LayoutOptimizer");
  ASSERT_NE(input_const_node, nullptr);
  EXPECT_EQ(input_const_node->NumRegularFanins(), 0);

  auto* input_reshape_node =
      context.graph_view->GetNode("Mul-1-ReshapeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_reshape_node, nullptr);
  ASSERT_EQ(input_reshape_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_reshape_node, 0, "a", 0);
  VerifyRegularFaninMatch(input_reshape_node, 1, input_const_node->GetName(),
                          0);

  auto* input_transpose_node =
      context.graph_view->GetNode("Mul-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* updated_mul = context.graph_view->GetNode("Mul");
  ASSERT_NE(updated_mul, nullptr);
  ASSERT_EQ(updated_mul->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(updated_mul, 1, input_reshape_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_mul, 0, input_transpose_node->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "Mul-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, BinaryOpTransposerPolygamma) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  auto conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto a = ops::RandomUniform(scope.WithOpName("a"),
                              {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);

  auto polygamma = ops::Polygamma(
      scope.WithOpName("polygamma").WithDevice("/device:GPU:0"), conv2d, a);
  auto z = ops::Identity(scope.WithOpName("z"), polygamma);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  BinaryOpTransposer binaryop_transposer;
  auto* polygamma_op = context.graph_view->GetNode("polygamma");
  ASSERT_NE(polygamma_op, nullptr);
  TF_ASSERT_OK(binaryop_transposer.TransposeNode(&context, polygamma_op));

  auto* input_transpose_node1 = context.graph_view->GetNode(
      "polygamma-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node1, nullptr);
  ASSERT_EQ(input_transpose_node1->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node1, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* updated_polygamma = context.graph_view->GetNode("polygamma");
  ASSERT_NE(updated_polygamma, nullptr);
  ASSERT_EQ(updated_polygamma->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(updated_polygamma, 0,
                          input_transpose_node1->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "polygamma-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

bool CreateConcatV1Op(const Scope& scope, const InputList& tensors,
                      const Input& concat_axis, Output* output) {
  if (!scope.ok()) {
    return false;
  }
  auto values = ops::AsNodeOutList(scope, tensors);
  if (!scope.ok()) {
    return false;
  }
  auto axis = ops::AsNodeOut(scope, concat_axis);
  if (!scope.ok()) {
    return false;
  }
  Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("Concat");
  auto builder = NodeBuilder(unique_name, "Concat").Input(axis).Input(values);
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) {
    return false;
  }
  scope.UpdateStatus(scope.DoShapeInference(ret));
  *output = Output(ret, 0);
  return true;
}

TEST_F(TransposerTest, ConcatOpTransposerConcat) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  Output input_1 = ops::RandomUniform(scope.WithOpName("input_1"),
                                      {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  Output input_2 = ops::RandomUniform(scope.WithOpName("input_2"),
                                      {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto axis = ops::Const(scope.WithOpName("axis"), 2, {});
  Output concat_op;
  ASSERT_TRUE(
      CreateConcatV1Op(scope.WithOpName("concat").WithDevice("/device:GPU:0"),
                       {input_1, input_2, conv2d}, axis, &concat_op));
  auto z = ops::Identity(scope.WithOpName("z"), concat_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  ConcatOpTransposer concat_transposer;
  auto* concat = context.graph_view->GetNode("concat");
  ASSERT_NE(concat, nullptr);
  TF_ASSERT_OK(concat_transposer.TransposeNode(&context, concat));

  auto* conv2d_transpose_node = context.graph_view->GetNode(
      "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(conv2d_transpose_node, nullptr);
  auto* conv2d_concat_input_node = context.graph_view->GetNode(
      "concat-3-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(conv2d_concat_input_node, nullptr);
  ASSERT_EQ(conv2d_concat_input_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(conv2d_concat_input_node, 0,
                          conv2d_transpose_node->GetName(), 0);

  auto* axis_dim_node = context.graph_view->GetNode(
      "concat-0-DataFormatDimMapNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(axis_dim_node, nullptr);

  auto* updated_concat = context.graph_view->GetNode("concat");
  ASSERT_NE(updated_concat, nullptr);
  ASSERT_EQ(updated_concat->NumRegularFanins(), 4);
  VerifyRegularFaninMatch(updated_concat, 0, axis_dim_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_concat, 1,
                          "concat-1-TransposeNHWCToNCHW-LayoutOptimizer", 0);
  VerifyRegularFaninMatch(updated_concat, 2,
                          "concat-2-TransposeNHWCToNCHW-LayoutOptimizer", 0);
  VerifyRegularFaninMatch(updated_concat, 3,
                          conv2d_concat_input_node->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "concat-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, ConcatOpTransposerConcatV2) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  Output input_1 = ops::RandomUniform(scope.WithOpName("input_1"),
                                      {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  Output input_2 = ops::RandomUniform(scope.WithOpName("input_2"),
                                      {kBatchSize, 5, 3, kDepthOut}, DT_FLOAT);
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto axis = ops::Const(scope.WithOpName("axis"), 2, {});
  auto concat_op =
      ops::Concat(scope.WithOpName("concat").WithDevice("/device:GPU:0"),
                  {input_1, input_2, conv2d}, axis);
  auto z = ops::Identity(scope.WithOpName("z"), concat_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  ConcatOpTransposer concat_transposer;
  auto* concat = context.graph_view->GetNode("concat");
  ASSERT_NE(concat, nullptr);
  TF_ASSERT_OK(concat_transposer.TransposeNode(&context, concat));

  auto* conv2d_transpose_node = context.graph_view->GetNode(
      "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(conv2d_transpose_node, nullptr);
  auto* conv2d_concat_input_node = context.graph_view->GetNode(
      "concat-2-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(conv2d_concat_input_node, nullptr);
  ASSERT_EQ(conv2d_concat_input_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(conv2d_concat_input_node, 0,
                          conv2d_transpose_node->GetName(), 0);

  auto* axis_dim_node = context.graph_view->GetNode(
      "concat-3-DataFormatDimMapNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(axis_dim_node, nullptr);

  auto* updated_concat = context.graph_view->GetNode("concat");
  ASSERT_NE(updated_concat, nullptr);
  ASSERT_EQ(updated_concat->NumRegularFanins(), 4);
  VerifyRegularFaninMatch(updated_concat, 0,
                          "concat-0-TransposeNHWCToNCHW-LayoutOptimizer", 0);
  VerifyRegularFaninMatch(updated_concat, 1,
                          "concat-1-TransposeNHWCToNCHW-LayoutOptimizer", 0);
  VerifyRegularFaninMatch(updated_concat, 2,
                          conv2d_concat_input_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_concat, 3, axis_dim_node->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "concat-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, ReverseV2Transposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();

  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto axis = ops::Const(scope.WithOpName("axis"), {0, 3}, {2});
  auto reverse_op = ops::Reverse(
      scope.WithOpName("reverse_v2").WithDevice("/device:GPU:0"), conv2d, axis);
  auto z = ops::Identity(scope.WithOpName("z"), reverse_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  ReverseV2Transposer reverse_v2_transposer;
  auto* reverse_v2 = context.graph_view->GetNode("reverse_v2");
  ASSERT_NE(reverse_v2, nullptr);
  TF_ASSERT_OK(reverse_v2_transposer.TransposeNode(&context, reverse_v2));

  auto* input_transpose_node = context.graph_view->GetNode(
      "reverse_v2-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* axis_node = context.graph_view->GetNode(
      "reverse_v2-1-DataFormatDimMapNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(axis_node, nullptr);
  ASSERT_EQ(axis_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(axis_node, 0, "axis", 0);

  auto* updated_reverse_v2 = context.graph_view->GetNode("reverse_v2");
  ASSERT_NE(updated_reverse_v2, nullptr);
  ASSERT_EQ(updated_reverse_v2->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(updated_reverse_v2, 0,
                          input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_reverse_v2, 1, axis_node->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "reverse_v2-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, TileTransposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();

  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto multiple = ops::Const(scope.WithOpName("multiple"), {1, 1, 2, 3}, {4});
  auto tile_op = ops::Tile(scope.WithOpName("tile").WithDevice("/device:GPU:0"),
                           conv2d, multiple);
  auto z = ops::Identity(scope.WithOpName("z"), tile_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  TileTransposer tile_transposer;
  auto* tile = context.graph_view->GetNode("tile");
  ASSERT_NE(tile, nullptr);
  TF_ASSERT_OK(tile_transposer.TransposeNode(&context, tile));

  auto* input_transpose_node =
      context.graph_view->GetNode("tile-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* multiple_node = context.graph_view->GetNode(
      "tile-1-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(multiple_node, nullptr);
  ASSERT_EQ(multiple_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(multiple_node, 0, "multiple", 0);

  auto* updated_tile = context.graph_view->GetNode("tile");
  ASSERT_NE(updated_tile, nullptr);
  ASSERT_EQ(updated_tile->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(updated_tile, 0, input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_tile, 1, multiple_node->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "tile-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, ShapeTransposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto shape =
      ops::Shape(scope.WithOpName("shape").WithDevice("/device:GPU:0"), conv2d);
  auto z = ops::Identity(scope.WithOpName("z"), shape);

  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  ShapeTransposer shape_transposer;
  auto* shape_node = context.graph_view->GetNode("shape");
  ASSERT_NE(shape_node, nullptr);
  TF_ASSERT_OK(shape_transposer.TransposeNode(&context, shape_node));

  auto* conv2d_transpose_node = context.graph_view->GetNode(
      "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(conv2d_transpose_node, nullptr);

  auto* shape_input_node = context.graph_view->GetNode(
      "shape-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(shape_input_node, nullptr);
  ASSERT_EQ(shape_input_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(shape_input_node, 0, conv2d_transpose_node->GetName(),
                          0);

  auto* output_vec_perm_node = context.graph_view->GetNode(
      "shape-0-0-DataFormatVecPermuteNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_vec_perm_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_vec_perm_node->GetName(), 0);
}

TEST_F(TransposerTest, ShapeNTransposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d_1 = ops::Conv2D(
      scope.WithOpName("conv2d_1").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  Output conv2d_2 = ops::Conv2D(
      scope.WithOpName("conv2d_2").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  Output conv2d_3 = ops::Conv2D(
      scope.WithOpName("conv2d_3").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto shape =
      ops::ShapeN(scope.WithOpName("shape").WithDevice("/device:GPU:0"),
                  {conv2d_1, conv2d_2, conv2d_3});
  auto z_1 = ops::Identity(scope.WithOpName("z_1"), shape.output[0]);
  auto z_2 = ops::Identity(scope.WithOpName("z_2"), shape.output[1]);
  auto z_3 = ops::Identity(scope.WithOpName("z_3"), shape.output[2]);

  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d_1 = context.graph_view->GetNode("conv2d_1");
  ASSERT_NE(c2d_1, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d_1));
  auto* c2d_2 = context.graph_view->GetNode("conv2d_2");
  ASSERT_NE(c2d_2, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d_2));

  ShapeNTransposer shape_transposer;
  auto* shape_node = context.graph_view->GetNode("shape");
  ASSERT_NE(shape_node, nullptr);
  TF_ASSERT_OK(shape_transposer.TransposeNode(&context, shape_node));

  auto* conv2d_1_transpose_node = context.graph_view->GetNode(
      "conv2d_1-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(conv2d_1_transpose_node, nullptr);
  auto* conv2d_2_transpose_node = context.graph_view->GetNode(
      "conv2d_2-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(conv2d_2_transpose_node, nullptr);

  auto* shape_input_1_node = context.graph_view->GetNode(
      "shape-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(shape_input_1_node, nullptr);
  ASSERT_EQ(shape_input_1_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(shape_input_1_node, 0,
                          conv2d_1_transpose_node->GetName(), 0);

  auto* shape_input_2_node = context.graph_view->GetNode(
      "shape-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(shape_input_2_node, nullptr);
  ASSERT_EQ(shape_input_2_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(shape_input_2_node, 0,
                          conv2d_2_transpose_node->GetName(), 0);

  auto* updated_shape_node = context.graph_view->GetNode("shape");
  ASSERT_NE(updated_shape_node, nullptr);
  ASSERT_EQ(updated_shape_node->NumRegularFanins(), 3);
  VerifyRegularFaninMatch(updated_shape_node, 0, shape_input_1_node->GetName(),
                          0);
  VerifyRegularFaninMatch(updated_shape_node, 1, shape_input_2_node->GetName(),
                          0);
  VerifyRegularFaninMatch(updated_shape_node, 2, "conv2d_3", 0);

  auto* output_vec_perm_node_1 = context.graph_view->GetNode(
      "shape-0-0-DataFormatVecPermuteNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_vec_perm_node_1, nullptr);
  auto* output_vec_perm_node_2 = context.graph_view->GetNode(
      "shape-1-0-DataFormatVecPermuteNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_vec_perm_node_2, nullptr);

  auto* z_output_node_1 = context.graph_view->GetNode("z_1");
  ASSERT_NE(z_output_node_1, nullptr);
  ASSERT_EQ(z_output_node_1->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node_1, 0, output_vec_perm_node_1->GetName(),
                          0);

  auto* z_output_node_2 = context.graph_view->GetNode("z_2");
  ASSERT_NE(z_output_node_2, nullptr);
  ASSERT_EQ(z_output_node_2->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node_2, 0, output_vec_perm_node_2->GetName(),
                          0);

  auto* z_output_node_3 = context.graph_view->GetNode("z_3");
  ASSERT_NE(z_output_node_3, nullptr);
  ASSERT_EQ(z_output_node_3->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node_3, 0, updated_shape_node->GetName(), 2);
}

TEST_F(TransposerTest, FillOpTransposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();
  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto shape = ops::Shape(scope.WithOpName("conv2d"), conv2d);
  auto value = ops::Const(scope.WithOpName("value"), 0, {});
  auto fill = ops::Fill(scope.WithOpName("fill").WithDevice("/device:GPU:0"),
                        shape, value);
  auto z = ops::Identity(scope.WithOpName("z"), fill);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  FillOpTransposer fill_op_transposer;
  auto* fill_node = context.graph_view->GetNode("fill");
  ASSERT_NE(fill_node, nullptr);
  TF_ASSERT_OK(fill_op_transposer.TransposeNode(&context, fill_node));

  auto* input_node = context.graph_view->GetNode(
      "fill-0-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_node, nullptr);

  auto* updated_fill_node = context.graph_view->GetNode("fill");
  ASSERT_NE(updated_fill_node, nullptr);
  ASSERT_EQ(updated_fill_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(updated_fill_node, 0, input_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_fill_node, 1, "value", 0);

  auto* output_node = context.graph_view->GetNode(
      "fill-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_node, 0, updated_fill_node->GetName(), 0);

  auto* z_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_node, nullptr);
  ASSERT_EQ(z_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_node, 0, output_node->GetName(), 0);
}

TEST_F(TransposerTest, SliceTransposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();

  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto begin = ops::Const(scope.WithOpName("begin"), {0, 0, 2, 1}, {4});
  auto size = ops::Const(scope.WithOpName("size"), {1, 1, 2, 3}, {4});
  auto slice_op =
      ops::Slice(scope.WithOpName("slice").WithDevice("/device:GPU:0"), conv2d,
                 begin, size);
  auto z = ops::Identity(scope.WithOpName("z"), slice_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SliceTransposer slice_transposer;
  auto* slice = context.graph_view->GetNode("slice");
  ASSERT_NE(slice, nullptr);
  TF_ASSERT_OK(slice_transposer.TransposeNode(&context, slice));

  auto* input_transpose_node = context.graph_view->GetNode(
      "slice-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* begin_node = context.graph_view->GetNode(
      "slice-1-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(begin_node, nullptr);
  ASSERT_EQ(begin_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(begin_node, 0, "begin", 0);

  auto* size_node = context.graph_view->GetNode(
      "slice-2-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(size_node, nullptr);
  ASSERT_EQ(size_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(size_node, 0, "size", 0);

  auto* updated_slice_node = context.graph_view->GetNode("slice");
  ASSERT_NE(updated_slice_node, nullptr);
  ASSERT_EQ(updated_slice_node->NumRegularFanins(), 3);
  VerifyRegularFaninMatch(updated_slice_node, 0,
                          input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_slice_node, 1, begin_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_slice_node, 2, size_node->GetName(), 0);

  auto* output_transpose_node = context.graph_view->GetNode(
      "slice-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, SplitTransposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();

  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto axis = ops::Const(scope.WithOpName("axis"), 2, {});
  auto split_op = ops::Split(
      scope.WithOpName("split").WithDevice("/device:GPU:0"), axis, conv2d, 3);
  auto z_1 = ops::Identity(scope.WithOpName("z_1"), split_op.output[0]);
  auto z_2 = ops::Identity(scope.WithOpName("z_2"), split_op.output[1]);
  auto z_3 = ops::Identity(scope.WithOpName("z_3"), split_op.output[2]);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SplitTransposer split_transposer;
  auto* split = context.graph_view->GetNode("split");
  ASSERT_NE(split, nullptr);
  TF_ASSERT_OK(split_transposer.TransposeNode(&context, split));

  auto* input_transpose_node = context.graph_view->GetNode(
      "split-1-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* axis_node = context.graph_view->GetNode(
      "split-0-DataFormatDimMapNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(axis_node, nullptr);
  ASSERT_EQ(axis_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(axis_node, 0, "axis", 0);

  auto* updated_split_node = context.graph_view->GetNode("split");
  ASSERT_NE(updated_split_node, nullptr);
  ASSERT_EQ(updated_split_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(updated_split_node, 0, axis_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_split_node, 1,
                          input_transpose_node->GetName(), 0);

  auto* output_transpose_node_1 = context.graph_view->GetNode(
      "split-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node_1, nullptr);
  auto* output_transpose_node_2 = context.graph_view->GetNode(
      "split-1-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node_2, nullptr);
  auto* output_transpose_node_3 = context.graph_view->GetNode(
      "split-2-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node_3, nullptr);

  auto* z_output_node_1 = context.graph_view->GetNode("z_1");
  ASSERT_NE(z_output_node_1, nullptr);
  ASSERT_EQ(z_output_node_1->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node_1, 0,
                          output_transpose_node_1->GetName(), 0);
  auto* z_output_node_2 = context.graph_view->GetNode("z_2");
  ASSERT_NE(z_output_node_2, nullptr);
  ASSERT_EQ(z_output_node_2->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node_2, 0,
                          output_transpose_node_2->GetName(), 0);
  auto* z_output_node_3 = context.graph_view->GetNode("z_3");
  ASSERT_NE(z_output_node_3, nullptr);
  ASSERT_EQ(z_output_node_3->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node_3, 0,
                          output_transpose_node_3->GetName(), 0);
}

TEST_F(TransposerTest, SplitVTransposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();

  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));
  auto axis = ops::Const(scope.WithOpName("axis"), 1, {});
  auto size_splits =
      ops::Const(scope.WithOpName("size_splits"), {2, 2, 1}, {3});
  auto splitv_op =
      ops::SplitV(scope.WithOpName("splitv").WithDevice("/device:GPU:0"),
                  conv2d, size_splits, axis, 3);
  auto z_1 = ops::Identity(scope.WithOpName("z_1"), splitv_op.output[0]);
  auto z_2 = ops::Identity(scope.WithOpName("z_2"), splitv_op.output[1]);
  auto z_3 = ops::Identity(scope.WithOpName("z_3"), splitv_op.output[2]);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  SplitVTransposer splitv_transposer;
  auto* splitv = context.graph_view->GetNode("splitv");
  ASSERT_NE(splitv, nullptr);
  TF_ASSERT_OK(splitv_transposer.TransposeNode(&context, splitv));

  auto* input_transpose_node = context.graph_view->GetNode(
      "splitv-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* axis_node = context.graph_view->GetNode(
      "splitv-2-DataFormatDimMapNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(axis_node, nullptr);
  ASSERT_EQ(axis_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(axis_node, 0, "axis", 0);

  auto* updated_splitv_node = context.graph_view->GetNode("splitv");
  ASSERT_NE(updated_splitv_node, nullptr);
  ASSERT_EQ(updated_splitv_node->NumRegularFanins(), 3);
  VerifyRegularFaninMatch(updated_splitv_node, 0,
                          input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_splitv_node, 1, "size_splits", 0);
  VerifyRegularFaninMatch(updated_splitv_node, 2, axis_node->GetName(), 0);

  auto* output_transpose_node_1 = context.graph_view->GetNode(
      "splitv-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node_1, nullptr);
  auto* output_transpose_node_2 = context.graph_view->GetNode(
      "splitv-1-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node_2, nullptr);
  auto* output_transpose_node_3 = context.graph_view->GetNode(
      "splitv-2-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node_3, nullptr);

  auto* z_output_node_1 = context.graph_view->GetNode("z_1");
  ASSERT_NE(z_output_node_1, nullptr);
  ASSERT_EQ(z_output_node_1->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node_1, 0,
                          output_transpose_node_1->GetName(), 0);
  auto* z_output_node_2 = context.graph_view->GetNode("z_2");
  ASSERT_NE(z_output_node_2, nullptr);
  ASSERT_EQ(z_output_node_2->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node_2, 0,
                          output_transpose_node_2->GetName(), 0);
  auto* z_output_node_3 = context.graph_view->GetNode("z_3");
  ASSERT_NE(z_output_node_3, nullptr);
  ASSERT_EQ(z_output_node_3->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node_3, 0,
                          output_transpose_node_3->GetName(), 0);
}

TEST_F(TransposerTest, StridedSliceTransposer) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();

  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

  auto attrs = ops::StridedSlice::Attrs().BeginMask(0xB).EndMask(0x7);

  auto begin = ops::Const(scope.WithOpName("begin"), {2, 0, 2, 1}, {4});
  auto end = ops::Const(scope.WithOpName("end"), {34, 4, 3, 1}, {4});
  auto strides = ops::Const(scope.WithOpName("strides"), {7, 2, 1, 1}, {4});

  auto strided_slice_op = ops::StridedSlice(
      scope.WithOpName("stridedslice").WithDevice("/device:GPU:0"), conv2d,
      begin, end, strides, attrs);
  auto z = ops::Identity(scope.WithOpName("z"), strided_slice_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  StridedSliceTransposer stridedslice_transposer;
  auto* stridedslice = context.graph_view->GetNode("stridedslice");
  ASSERT_NE(stridedslice, nullptr);
  TF_ASSERT_OK(stridedslice_transposer.TransposeNode(&context, stridedslice));

  auto* input_transpose_node = context.graph_view->GetNode(
      "stridedslice-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* begin_node = context.graph_view->GetNode(
      "stridedslice-1-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(begin_node, nullptr);
  auto* end_node = context.graph_view->GetNode(
      "stridedslice-2-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(end_node, nullptr);
  auto* strides_node = context.graph_view->GetNode(
      "stridedslice-3-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(strides_node, nullptr);

  auto* updated_stridedslice_node = context.graph_view->GetNode("stridedslice");
  ASSERT_NE(updated_stridedslice_node, nullptr);
  ASSERT_EQ(updated_stridedslice_node->NumRegularFanins(), 4);
  VerifyRegularFaninMatch(updated_stridedslice_node, 0,
                          input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_stridedslice_node, 1, begin_node->GetName(),
                          0);
  VerifyRegularFaninMatch(updated_stridedslice_node, 2, end_node->GetName(), 0);
  VerifyRegularFaninMatch(updated_stridedslice_node, 3, strides_node->GetName(),
                          0);
  const auto* begin_mask_attr =
      updated_stridedslice_node->GetAttr("begin_mask");
  ASSERT_NE(begin_mask_attr, nullptr);
  EXPECT_EQ(begin_mask_attr->i(), 0x7);
  const auto* end_mask_attr = updated_stridedslice_node->GetAttr("end_mask");
  ASSERT_NE(end_mask_attr, nullptr);
  EXPECT_EQ(end_mask_attr->i(), 0xD);

  auto* output_transpose_node = context.graph_view->GetNode(
      "stridedslice-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0, output_transpose_node->GetName(),
                          0);
}

TEST_F(TransposerTest, StridedSliceTransposerEllipsisMaskPresent) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();

  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

  auto attrs =
      ops::StridedSlice::Attrs().BeginMask(0xB).EndMask(0x7).EllipsisMask(0x2);

  auto begin = ops::Const(scope.WithOpName("begin"), {2, 0, 2, 1}, {4});
  auto end = ops::Const(scope.WithOpName("end"), {34, 4, 3, 1}, {4});
  auto strides = ops::Const(scope.WithOpName("strides"), {7, 2, 1, 1}, {4});

  auto strided_slice_op = ops::StridedSlice(
      scope.WithOpName("stridedslice").WithDevice("/device:GPU:0"), conv2d,
      begin, end, strides, attrs);
  auto z = ops::Identity(scope.WithOpName("z"), strided_slice_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  StridedSliceTransposer stridedslice_transposer;
  auto* stridedslice = context.graph_view->GetNode("stridedslice");
  ASSERT_NE(stridedslice, nullptr);
  TF_ASSERT_OK(stridedslice_transposer.TransposeNode(&context, stridedslice));

  // Expect StridedSlice Node to remain unchanged because of the ellipsis mask.
  auto* updated_stridedslice_node = context.graph_view->GetNode("stridedslice");
  ASSERT_NE(updated_stridedslice_node, nullptr);
  ASSERT_EQ(updated_stridedslice_node->NumRegularFanins(), 4);
  VerifyRegularFaninMatch(updated_stridedslice_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);
  VerifyRegularFaninMatch(updated_stridedslice_node, 1, "begin", 0);
  VerifyRegularFaninMatch(updated_stridedslice_node, 2, "end", 0);
  VerifyRegularFaninMatch(updated_stridedslice_node, 3, "strides", 0);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0,
                          updated_stridedslice_node->GetName(), 0);
}

TEST_F(TransposerTest, StridedSliceTransposerConstFaninBadRank) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  Scope scope = Scope::NewRootScope();

  auto input =
      ops::RandomUniform(scope.WithOpName("input"),
                         {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
  auto filter =
      ops::RandomUniform(scope.WithOpName("filter"),
                         {kHeight, kWidth, kDepthIn, kDepthOut}, DT_FLOAT);
  Output conv2d = ops::Conv2D(
      scope.WithOpName("conv2d").WithDevice("/device:GPU:0"), input, filter,
      {1, 2, 4, 1}, "SAME", ops::Conv2D::DataFormat(kSrcFormat));

  auto attrs = ops::StridedSlice::Attrs().BeginMask(0xB).EndMask(0x7);

  auto begin = ops::Const(scope.WithOpName("begin"), {2, 0, 2}, {3});
  auto end = ops::Const(scope.WithOpName("end"), {34, 4, 3}, {3});
  auto strides = ops::Const(scope.WithOpName("strides"), {7, 2, 1}, {3});

  auto strided_slice_op = ops::StridedSlice(
      scope.WithOpName("stridedslice").WithDevice("/device:GPU:0"), conv2d,
      begin, end, strides, attrs);
  auto z = ops::Identity(scope.WithOpName("z"), strided_slice_op);
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  TransposeContext context;
  TF_ASSERT_OK(TransposeContext::InitializeTransposeContext(
      item, virtual_cluster_.get(), &context));
  context.AssignDeviceAndDataFormats(kGPU, kSrcFormat, kDstFormat);

  DefaultLayoutSensitiveOpTransposer conv2d_transposer;
  auto* c2d = context.graph_view->GetNode("conv2d");
  ASSERT_NE(c2d, nullptr);
  TF_ASSERT_OK(conv2d_transposer.TransposeNode(&context, c2d));

  StridedSliceTransposer stridedslice_transposer;
  auto* stridedslice = context.graph_view->GetNode("stridedslice");
  ASSERT_NE(stridedslice, nullptr);
  TF_ASSERT_OK(stridedslice_transposer.TransposeNode(&context, stridedslice));

  auto* input_transpose_node = context.graph_view->GetNode(
      "stridedslice-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_EQ(input_transpose_node, nullptr);

  auto* begin_node = context.graph_view->GetNode(
      "stridedslice-1-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_EQ(begin_node, nullptr);
  auto* end_node = context.graph_view->GetNode(
      "stridedslice-2-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_EQ(end_node, nullptr);
  auto* strides_node = context.graph_view->GetNode(
      "stridedslice-3-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_EQ(strides_node, nullptr);

  auto* updated_stridedslice_node = context.graph_view->GetNode("stridedslice");
  ASSERT_NE(updated_stridedslice_node, nullptr);
  ASSERT_EQ(updated_stridedslice_node->NumRegularFanins(), 4);
  VerifyRegularFaninMatch(updated_stridedslice_node, 0,
                          "conv2d-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);
  VerifyRegularFaninMatch(updated_stridedslice_node, 1, "begin", 0);
  VerifyRegularFaninMatch(updated_stridedslice_node, 2, "end", 0);
  VerifyRegularFaninMatch(updated_stridedslice_node, 3, "strides", 0);
  const auto* begin_mask_attr =
      updated_stridedslice_node->GetAttr("begin_mask");
  ASSERT_NE(begin_mask_attr, nullptr);
  EXPECT_EQ(begin_mask_attr->i(), 0xB);
  const auto* end_mask_attr = updated_stridedslice_node->GetAttr("end_mask");
  ASSERT_NE(end_mask_attr, nullptr);
  EXPECT_EQ(end_mask_attr->i(), 0x7);

  auto* output_transpose_node = context.graph_view->GetNode(
      "stridedslice-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_EQ(output_transpose_node, nullptr);

  auto* z_output_node = context.graph_view->GetNode("z");
  ASSERT_NE(z_output_node, nullptr);
  ASSERT_EQ(z_output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(z_output_node, 0,
                          updated_stridedslice_node->GetName(), 0);
}

TEST_F(TransposerTest, ReduceTransposerKeepDims) {
  ReduceTransposerKeepDims<int32>();
  ReduceTransposerKeepDims<int64_t>();
}

TEST_F(TransposerTest, ReduceTransposerValidAxisNode) {
  ReduceTransposerValidAxisNode<int32>();
  ReduceTransposerValidAxisNode<int64_t>();
}

TEST(PermutationTest, PermutesVector) {
  std::vector<int64_t> input{32, 16, 8, 4};
  std::vector<int64_t> expected{4, 8, 16, 32};
  TF_ASSERT_OK(PermuteSingle("test", {3, 2, 1, 0}, &input));
  ASSERT_EQ(input.size(), 4);
  for (int i = 0; i < input.size(); ++i) {
    EXPECT_EQ(input[i], expected[i]);
  }
}

TEST(PermutationTest, PermutesRepeatedField) {
  TensorShapeProto input_shape = MakeTensorShapeFromDimensions({1, 2, 3, 4});
  TensorShapeProto expected_shape = MakeTensorShapeFromDimensions({1, 4, 2, 3});

  TF_ASSERT_OK(PermuteSingle("test", {0, 3, 1, 2}, input_shape.mutable_dim()));
  EXPECT_EQ(input_shape.DebugString(), expected_shape.DebugString());
}

TEST(PermutationTest, PermutesDoubleRepeatedField) {
  {
    // NHWC -> NCHW
    TensorShapeProto input =
        MakeTensorShapeFromDimensions({1, 2, 3, 4, 5, 6, 7, 8});
    TensorShapeProto expected =
        MakeTensorShapeFromDimensions({1, 2, 7, 8, 3, 4, 5, 6});

    TF_ASSERT_OK(PermuteDouble("test", {0, 3, 1, 2}, input.mutable_dim()));
    EXPECT_EQ(input.DebugString(), expected.DebugString());
  }
  {
    // NCHW -> NHWC
    TensorShapeProto input =
        MakeTensorShapeFromDimensions({1, 2, 3, 4, 5, 6, 7, 8});
    TensorShapeProto expected =
        MakeTensorShapeFromDimensions({1, 2, 5, 6, 7, 8, 3, 4});
    TF_ASSERT_OK(PermuteDouble("test", {0, 2, 3, 1}, input.mutable_dim()));
    EXPECT_EQ(input.DebugString(), expected.DebugString());
  }
}

TEST(PermutationTest, PermutesDataFormat) {
  string input = "NHWC";
  string expected = "NCHW";
  TF_ASSERT_OK(PermuteSingle("test", {0, 3, 1, 2}, &input));
  EXPECT_EQ(input, expected);
}

TEST(PermutationTest, PermutesString) {
  string input = "ABCD";
  string expected = "ACBD";
  TF_ASSERT_OK(PermuteSingle("test", {0, 2, 1, 3}, &input));
  EXPECT_EQ(input, expected);
}

TEST(PermutationTest, GetNHWCToNCHWPermutation) {
  string src_format = "NHWC";
  absl::flat_hash_map<char, int> src_dim_indices =
      GetDimensionIndices(src_format);
  EXPECT_EQ(src_dim_indices.size(), 4);
  EXPECT_EQ(src_dim_indices['N'], 0);
  EXPECT_EQ(src_dim_indices['H'], 1);
  EXPECT_EQ(src_dim_indices['W'], 2);
  EXPECT_EQ(src_dim_indices['C'], 3);
  string dst_format = "NCHW";
  std::vector<int> permutation = GetPermutation(src_dim_indices, dst_format);
  ASSERT_EQ(permutation.size(), 4);
  EXPECT_EQ(permutation[0], 0);
  EXPECT_EQ(permutation[1], 3);
  EXPECT_EQ(permutation[2], 1);
  EXPECT_EQ(permutation[3], 2);
}

TEST(PermutationTest, GetNCHWToNHWCPermutation) {
  string src_format = "NCHW";
  absl::flat_hash_map<char, int> src_dim_indices =
      GetDimensionIndices(src_format);
  EXPECT_EQ(src_dim_indices.size(), 4);
  EXPECT_EQ(src_dim_indices['N'], 0);
  EXPECT_EQ(src_dim_indices['C'], 1);
  EXPECT_EQ(src_dim_indices['H'], 2);
  EXPECT_EQ(src_dim_indices['W'], 3);
  string dst_format = "NHWC";
  std::vector<int> permutation = GetPermutation(src_dim_indices, dst_format);
  ASSERT_EQ(permutation.size(), 4);
  EXPECT_EQ(permutation[0], 0);
  EXPECT_EQ(permutation[1], 2);
  EXPECT_EQ(permutation[2], 3);
  EXPECT_EQ(permutation[3], 1);
}

// TODO(yanzha): Add frame related tests.
}  // namespace
}  // namespace grappler
}  // namespace tensorflow
