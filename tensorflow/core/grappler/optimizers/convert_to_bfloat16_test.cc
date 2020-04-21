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

#if INTEL_MKL
#ifdef ENABLE_INTEL_MKL_BFLOAT16

#include "tensorflow/core/grappler/optimizers/convert_to_bfloat16.h"
#include "tensorflow/core/grappler/optimizers/remapper.h"

#include <utility>
#include <vector>

#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/list_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace grappler {
namespace {

class BFloat16ConverterTest : public GrapplerTest {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

void VerifyGraphsEquivalent(const GraphDef& original_graph,
                            const GraphDef& optimized_graph,
                            const string& func) {
  EXPECT_EQ(original_graph.node_size(), optimized_graph.node_size()) << func;
  GraphView optimized_view(&optimized_graph);
  for (int i = 0; i < original_graph.node_size(); ++i) {
    const NodeDef& original = original_graph.node(i);
    const NodeDef& optimized = *optimized_view.GetNode(original.name());
    EXPECT_EQ(original.name(), optimized.name()) << func;
    EXPECT_EQ(original.op(), optimized.op()) << func;
    EXPECT_EQ(original.input_size(), optimized.input_size()) << func;
    if (original.input_size() == optimized.input_size()) {
      for (int j = 0; j < original.input_size(); ++j) {
        EXPECT_EQ(original.input(j), optimized.input(j)) << func;
      }
    }
  }
}

#define CHECK_TYPE(NAME, ATTR, T)                                    \
  do {                                                               \
    EXPECT_EQ(output_view.GetNode(NAME)->attr().at(ATTR).type(), T); \
  } while (0);

#define CHECK_FP32ToBF16Cast(NAME)         \
  do {                                     \
    CHECK_TYPE(NAME, "SrcT", DT_FLOAT);    \
    CHECK_TYPE(NAME, "DstT", DT_BFLOAT16); \
  } while (0);

#define CHECK_BF16ToFP32Cast(NAME)         \
  do {                                     \
    CHECK_TYPE(NAME, "SrcT", DT_BFLOAT16); \
    CHECK_TYPE(NAME, "DstT", DT_FLOAT);    \
  } while (0);

// Test involving operators that will not be rewritten into BFloat16.
// Graph should not change at all after running the optimizer.
TEST_F(BFloat16ConverterTest, NoOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.234f, {32});
  Output b = ops::Exp(s.WithOpName("b"), input);
  Output c = ops::Sqrt(s.WithOpName("c"), b);
  Output fetch = ops::Identity(s.WithOpName("fetch"), c);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  BFloat16Converter converter;
  GraphDef output;
  TF_ASSERT_OK(converter.Optimize(nullptr, item, &output));
  VLOG(1) << output.DebugString();

  VerifyGraphsEquivalent(item.graph, output, __FUNCTION__);

  GraphView output_view(&output);
  // We don't expect any change to the graph.
  int expected_num_cast_nodes = 0;
  EXPECT_EQ(output.node_size(),
            item.graph.node_size() + expected_num_cast_nodes);
  CHECK_TYPE("input", "dtype", DT_FLOAT);
  CHECK_TYPE("b", "T", DT_FLOAT);
  CHECK_TYPE("c", "T", DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

// Test involving graph that contains some operators in BFloat16.
TEST_F(BFloat16ConverterTest, AlreadyBFloat16) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f, {32, 32});
  Output b = ops::Cast(s.WithOpName("b"), input, DT_BFLOAT16);
  Output c = ops::MatMul(s.WithOpName("c"), b, b);
  Output d = ops::Relu(s.WithOpName("d"), c);
  Output e = ops::Cast(s.WithOpName("e"), d, DT_FLOAT);
  Output fetch = ops::Identity(s.WithOpName("fetch"), e);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  BFloat16Converter converter;
  GraphDef output;
  TF_ASSERT_OK(converter.Optimize(nullptr, item, &output));
  VLOG(1) << output.DebugString();

  VerifyGraphsEquivalent(item.graph, output, __FUNCTION__);
  GraphView output_view(&output);
  // We don't expect any change to the graph.
  int expected_num_cast_nodes = 0;
  EXPECT_EQ(output.node_size(),
            item.graph.node_size() + expected_num_cast_nodes);
  CHECK_TYPE("input", "dtype", DT_FLOAT);
  CHECK_FP32ToBF16Cast("b");
  CHECK_TYPE("c", "T", DT_BFLOAT16);
  CHECK_TYPE("d", "T", DT_BFLOAT16);
  CHECK_BF16ToFP32Cast("e");

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

// Graph involving operators that should be rewritten with BFloat16 type
TEST_F(BFloat16ConverterTest, Simple) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output b = ops::Exp(s.WithOpName("b"), input);
  Output c = ops::Relu(s.WithOpName("c"), b);
  Output d = ops::Sqrt(s.WithOpName("d"), c);
  Output e = ops::Relu(s.WithOpName("e"), d);
  Output f = ops::MatMul(s.WithOpName("f"), e, e);
  Output g = ops::Relu(s.WithOpName("g"), f);
  Output fetch = ops::Log(s.WithOpName("fetch"), g);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  BFloat16Converter converter;
  GraphDef output;
  TF_ASSERT_OK(converter.Optimize(nullptr, item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  // We expect 4 cast nodes will be inserted: between [b->c], [c->d], [d->e],
  // and [g->fetch]
  int expected_num_cast_nodes = 4;
  EXPECT_EQ(output.node_size(),
            item.graph.node_size() + expected_num_cast_nodes);
  CHECK_TYPE("input", "dtype", DT_FLOAT);
  CHECK_TYPE("b", "T", DT_FLOAT);
  CHECK_TYPE("c", "T", DT_BFLOAT16);
  CHECK_TYPE("d", "T", DT_FLOAT);
  CHECK_TYPE("e", "T", DT_BFLOAT16);
  CHECK_TYPE("f", "T", DT_BFLOAT16);
  CHECK_TYPE("g", "T", DT_BFLOAT16);

  CHECK_FP32ToBF16Cast("b_0_0_cast/_0");
  CHECK_BF16ToFP32Cast("c_0_0_cast/_1");
  CHECK_FP32ToBF16Cast("d_0_0_cast/_2");
  CHECK_BF16ToFP32Cast("g_0_0_cast/_3");

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-4);
  }
}

// Test for context-based rewrite feature. Mul operator in the graph should be
// rewritten with BFloat16 type since it has all the inputs coming from MKL
// supported operators.
TEST_F(BFloat16ConverterTest, ContextBasedRewritePositive) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output b = ops::Exp(s.WithOpName("b"), input);
  Output c = ops::Relu(s.WithOpName("c"), b);
  Output d = ops::Sqrt(s.WithOpName("d"), c);
  Output e = ops::Relu(s.WithOpName("e"), d);
  // Using Mul op here which goes through Context-based rewrite
  Output f = ops::Mul(s.WithOpName("f"), e, e);
  Output g = ops::Relu(s.WithOpName("g"), f);
  Output fetch = ops::Log(s.WithOpName("fetch"), g);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  BFloat16Converter converter;
  GraphDef output;
  TF_ASSERT_OK(converter.Optimize(nullptr, item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  // We expect 4 cast nodes will be inserted: between [b->c], [c->d], [d->e],
  // and [g->fetch]
  int expected_num_cast_nodes = 4;
  EXPECT_EQ(output.node_size(),
            item.graph.node_size() + expected_num_cast_nodes);
  CHECK_TYPE("input", "dtype", DT_FLOAT);
  CHECK_TYPE("b", "T", DT_FLOAT);
  CHECK_TYPE("c", "T", DT_BFLOAT16);
  CHECK_TYPE("d", "T", DT_FLOAT);
  CHECK_TYPE("e", "T", DT_BFLOAT16);
  CHECK_TYPE("f", "T", DT_BFLOAT16);
  CHECK_TYPE("g", "T", DT_BFLOAT16);

  CHECK_FP32ToBF16Cast("b_0_0_cast/_0");
  CHECK_BF16ToFP32Cast("c_0_0_cast/_1");
  CHECK_FP32ToBF16Cast("d_0_0_cast/_2");
  CHECK_BF16ToFP32Cast("g_0_0_cast/_3");

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    // TODO(nhasabni): check tolerance here.
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-2);
  }
}

#if 0
// Test for context-based rewrite feature. Mul operator in the graph should not
// be rewritten with BFloat16 type since it has all the inputs coming from
// non-MKL operators.
TEST_F(BFloat16ConverterTest, ContextBasedRewriteNegative) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output b = ops::Exp(s.WithOpName("b"), input);
  Output c = ops::Relu(s.WithOpName("c"), b);
  Output d = ops::Sqrt(s.WithOpName("d"), c);
  // Mul op getting input from non-BFLOAT16 op.
  Output e = ops::Mul(s.WithOpName("e"), d, d);
  Output f = ops::MatMul(s.WithOpName("f"), e, e);
  Output g = ops::Relu(s.WithOpName("g"), f);
  Output fetch = ops::Log(s.WithOpName("fetch"), g);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  BFloat16Converter converter;
  GraphDef output;
  TF_ASSERT_OK(converter.Optimize(nullptr, item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  // We expect 4 cast nodes will be inserted: between [b->c], [c->d], [e->f],
  // [e->f], and [g->fetch]
  int expected_num_cast_nodes = 5;
  EXPECT_EQ(output.node_size(),
            item.graph.node_size() + expected_num_cast_nodes);
  CHECK_TYPE("input", "dtype", DT_FLOAT);
  CHECK_TYPE("b", "T", DT_FLOAT);
  CHECK_TYPE("c", "T", DT_BFLOAT16);
  CHECK_TYPE("d", "T", DT_FLOAT);
  CHECK_TYPE("e", "T", DT_FLOAT);
  CHECK_TYPE("f", "T", DT_BFLOAT16);
  CHECK_TYPE("g", "T", DT_BFLOAT16);

  CHECK_FP32ToBF16Cast("b_0_0_cast/_0");
  CHECK_BF16ToFP32Cast("c_0_0_cast/_1");
  CHECK_FP32ToBF16Cast("e_0_0_cast/_2");
  CHECK_FP32ToBF16Cast("e_0_1_cast/_3");
  CHECK_BF16ToFP32Cast("g_0_0_cast/_4");

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    // TODO(nhasabni): check tolerance here.
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-2);
  }
}
#endif

// Check that Conv2D is fused even in BFloat16 format.
//
// Patterns tested:
// Conv2D(FP32) + BiasAdd(FP32) -> _FusedConv2D(BFloat16)
// Conv2D(FP32) + BiasAdd(FP32) + Relu(FP32) -> _FusedConv2D(BFloat16)
// Conv2D(FP32) + BiasAdd(FP32) + Relu6(FP32) -> _FusedConv2D(BFloat16)
// Conv2D(FP32) + BiasAdd(FP32) + Elu(FP32) -> _FusedConv2D(BFloat16)
TEST_F(BFloat16ConverterTest, FusionTest_Conv2DWithBiasAndActivation) {
  for (const string& fused_op_name : {"BiasAdd", "Relu", "Relu6", "Elu"}) {
    using ::tensorflow::ops::Placeholder;
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
    auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
    auto bias_shape = ops::Placeholder::Shape({128});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
    auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);
    std::vector<int> strides = {1, 1, 1, 1};
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
    auto bias_add = ops::BiasAdd(s.WithOpName("BiasAdd"), conv, bias);

    ops::Log fetch = [&]() {
      auto fetch = s.WithOpName("fetch");
      if (fused_op_name == "BiasAdd") {
        return ops::Log(fetch, bias_add);
      } else if (fused_op_name == "Relu") {
        auto fused_op = s.WithOpName(fused_op_name);
        return ops::Log(fetch, ops::Relu(fused_op, bias_add));
      } else if (fused_op_name == "Relu6") {
        auto fused_op = s.WithOpName(fused_op_name);
        return ops::Log(fetch, ops::Relu6(fused_op, bias_add));
      } else if (fused_op_name == "Elu") {
        auto fused_op = s.WithOpName(fused_op_name);
        return ops::Log(fetch, ops::Elu(fused_op, bias_add));
      }
      return ops::Log(fetch, bias);
    }();

    auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
    auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
    auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

    GrapplerItem item;
    item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
    item.fetch = {"fetch"};

    TF_CHECK_OK(s.ToGraphDef(&item.graph));
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef rewriter_output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &rewriter_output));
    item.graph = rewriter_output;

    BFloat16Converter converter;
    GraphDef converter_output;
    TF_ASSERT_OK(converter.Optimize(nullptr, item, &converter_output));

    GraphView output_view(&converter_output);

    // We expect 4 cast nodes will be inserted: for input, filter and bias,
    // and output from _FusedConv2D
    int expected_num_cast_nodes = 4;
    EXPECT_EQ(converter_output.node_size(),
              item.graph.node_size() + expected_num_cast_nodes);
    CHECK_TYPE("input", "dtype", DT_FLOAT);
    CHECK_TYPE("filter", "dtype", DT_FLOAT);
    CHECK_TYPE("bias", "dtype", DT_FLOAT);
    CHECK_TYPE(fused_op_name, "T", DT_BFLOAT16);
    CHECK_FP32ToBF16Cast("input_0_0_cast/_0");
    CHECK_FP32ToBF16Cast("filter_0_1_cast/_1");
    CHECK_FP32ToBF16Cast("bias_0_2_cast/_2");
    CHECK_BF16ToFP32Cast(fused_op_name + "_0_0_cast/_3");

    int found = 0;
    for (const NodeDef& node : converter_output.node()) {
      if (node.name() == fused_op_name) {
        EXPECT_EQ(node.op(), "_FusedConv2D");
        const auto fused_ops = node.attr().at("fused_ops").list().s();
        EXPECT_EQ(fused_ops[0], "BiasAdd");
        if (fused_op_name == "BiasAdd") {
          ASSERT_EQ(fused_ops.size(), 1);
        } else {
          // If fused_op is Relu, Relu6 or Elu, we fuse BiasAdd + activation
          // also. So there should be 2 fused ops.
          ASSERT_EQ(fused_ops.size(), 2);
          EXPECT_EQ(fused_ops[1], fused_op_name);
        }
        found++;
      }
    }
    EXPECT_EQ(found, 1);

    auto tensors = EvaluateNodes(converter_output, item.fetch, item.feed);
    EXPECT_EQ(tensors.size(), tensors_expected.size());
    EXPECT_EQ(tensors.size(), item.fetch.size());
    for (int i = 0; i < item.fetch.size(); ++i) {
      test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-2);
    }
  }
}

// Check that MatMul is fused even in BFloat16 format.
//
// Patterns tested:
// MatMul(FP32) + BiasAdd(FP32) -> _FusedMatMul(BFloat16)
// MKL-DNN does not support MatMul + BiasAdd + Activation yet.
TEST_F(BFloat16ConverterTest, FusionTest_MatMulWithBias) {
  for (const string& fused_op_name : {"BiasAdd"}) {
    using ::tensorflow::ops::Placeholder;
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto lhs_shape = ops::Placeholder::Shape({8, 32});
    auto rhs_shape = ops::Placeholder::Shape({32, 64});
    auto bias_shape = ops::Placeholder::Shape({64});

    auto lhs = Placeholder(s.WithOpName("lhs"), DT_FLOAT, lhs_shape);
    auto rhs = Placeholder(s.WithOpName("rhs"), DT_FLOAT, rhs_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);
    auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
    auto bias_add = ops::BiasAdd(s.WithOpName("BiasAdd"), matmul, bias);

    ops::Log fetch = [&]() {
      auto fetch = s.WithOpName("fetch");
      if (fused_op_name == "BiasAdd") {
        return ops::Log(fetch, bias_add);
      }

      return ops::Log(fetch, bias);
      // Add other rules here when we support activation.
    }();

    auto lhs_t = GenerateRandomTensor<DT_FLOAT>({8, 32});
    auto rhs_t = GenerateRandomTensor<DT_FLOAT>({32, 64});
    auto bias_t = GenerateRandomTensor<DT_FLOAT>({64});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}, {"bias", bias_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef rewriter_output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &rewriter_output));
    item.graph = rewriter_output;

    BFloat16Converter converter;
    GraphDef converter_output;
    TF_ASSERT_OK(converter.Optimize(nullptr, item, &converter_output));

    GraphView output_view(&converter_output);

    // We expect 4 cast nodes will be inserted: for input, filter and bias,
    // and output from _FusedConv2D
    int expected_num_cast_nodes = 4;
    EXPECT_EQ(converter_output.node_size(),
              item.graph.node_size() + expected_num_cast_nodes);
    CHECK_TYPE("lhs", "dtype", DT_FLOAT);
    CHECK_TYPE("rhs", "dtype", DT_FLOAT);
    CHECK_TYPE("bias", "dtype", DT_FLOAT);
    CHECK_TYPE(fused_op_name, "T", DT_BFLOAT16);
    CHECK_FP32ToBF16Cast("lhs_0_0_cast/_0");
    CHECK_FP32ToBF16Cast("rhs_0_1_cast/_1");
    CHECK_FP32ToBF16Cast("bias_0_2_cast/_2");
    CHECK_BF16ToFP32Cast(fused_op_name + "_0_0_cast/_3");

    int found = 0;
    for (const NodeDef& node : converter_output.node()) {
      if (node.name() == fused_op_name) {
        EXPECT_EQ(node.op(), "_FusedMatMul");
        const auto fused_ops = node.attr().at("fused_ops").list().s();
        EXPECT_EQ(fused_ops[0], "BiasAdd");
        if (fused_op_name == "BiasAdd") {
          ASSERT_EQ(fused_ops.size(), 1);
        }
        found++;
      }
    }
    EXPECT_EQ(found, 1);

    auto tensors = EvaluateNodes(converter_output, item.fetch, item.feed);
    EXPECT_EQ(tensors.size(), tensors_expected.size());
    EXPECT_EQ(tensors.size(), item.fetch.size());
    for (int i = 0; i < item.fetch.size(); ++i) {
      test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-2);
    }
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

#endif  // ENABLE_INTEL_MKL_BFLOAT16
#endif  // INTEL_MKL
