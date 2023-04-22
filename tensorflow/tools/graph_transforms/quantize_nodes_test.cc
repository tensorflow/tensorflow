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

#define EIGEN_USE_THREADS

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
Status QuantizeNodes(const GraphDef& input_graph_def,
                     const TransformFuncContext& context,
                     GraphDef* output_graph_def);
Status RemoveRedundantQuantizations(const GraphDef& input_graph_def,
                                    const TransformFuncContext& context,
                                    GraphDef* output_graph_def);
Status QuantizePlaceholders(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def);
Status ConvertFakeQuantsToRequantize(const GraphDef& input_graph_def,
                                     const TransformFuncContext& context,
                                     GraphDef* output_graph_def);
Status MergeAdjacentRequantizes(const GraphDef& input_graph_def,
                                const TransformFuncContext& context,
                                GraphDef* output_graph_def);
Status HoistFakeQuants(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def);
Status MergeDuplicateNodes(const GraphDef& input_graph_def,
                           const TransformFuncContext& context,
                           GraphDef* output_graph_def);

class QuantizeNodesTest : public ::testing::Test {
 protected:
  void TestTransformedVersusFloatGraph(
      const TransformFunc& transform_function, const GraphDef& float_graph_def,
      const std::vector<std::pair<string, Tensor>>& float_inputs,
      const std::vector<std::pair<string, Tensor>>& transformed_inputs,
      const std::vector<string>& output_names,
      const TransformFuncContext& in_context, double threshold,
      GraphDef* transformed_graph_def) {
    std::unique_ptr<Session> float_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(float_session->Create(float_graph_def));
    std::vector<Tensor> float_outputs;
    TF_ASSERT_OK(
        float_session->Run(float_inputs, output_names, {}, &float_outputs));

    TransformFuncContext context(in_context);
    std::vector<string> input_names;
    for (const std::pair<const string&, const Tensor&> float_input :
         float_inputs) {
      context.input_names.push_back(float_input.first);
    }

    context.output_names = output_names;
    TF_ASSERT_OK(
        transform_function(float_graph_def, context, transformed_graph_def));

    std::unique_ptr<Session> transformed_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(transformed_session->Create(*transformed_graph_def));
    std::vector<Tensor> transformed_outputs;
    TF_ASSERT_OK(transformed_session->Run(transformed_inputs, output_names, {},
                                          &transformed_outputs));

    const int output_count = output_names.size();
    EXPECT_EQ(output_count, float_outputs.size());
    EXPECT_EQ(output_count, transformed_outputs.size());
    for (int i = 0; i < output_count; ++i) {
      test::ExpectTensorNear<float>(float_outputs[i], transformed_outputs[i],
                                    threshold);
    }
  }

  void TestQuantizedVersusFloatGraph(
      const GraphDef& float_graph_def,
      const std::vector<std::pair<string, Tensor>>& inputs,
      const std::vector<string>& output_names) {
    GraphDef quantized_graph_def;
    TestTransformedVersusFloatGraph(QuantizeNodes, float_graph_def, inputs,
                                    inputs, output_names, {}, 1.0,
                                    &quantized_graph_def);
    // Reshape is not included here because it can be added as part of the
    // quantization process.
    const std::set<string> quantizable_ops = {
        "Add",   "BiasAdd",        "Concat",  "Conv2D",  "MatMul", "Relu",
        "Relu6", "ResizeBilinear", "AvgPool", "MaxPool", "Mul"};
    for (const NodeDef& node : quantized_graph_def.node()) {
      EXPECT_EQ(0, quantizable_ops.count(node.op()))
          << "Found quantizable node " << node.op() << " for node named "
          << node.name();
    }
  }

  void TestGraphWithInputRange(
      const GraphDef& float_graph_def,
      const std::vector<std::pair<string, Tensor>>& float_inputs,
      const std::vector<string>& output_names, float range_min,
      float range_max) {
    TransformFuncContext context;
    context.params["input_min"] = {strings::StrCat(range_min)};
    context.params["input_max"] = {strings::StrCat(range_max)};

    std::vector<std::pair<string, Tensor>> quantized_inputs;
    for (const std::pair<string, Tensor>& float_input : float_inputs) {
      const Tensor& float_tensor = float_input.second;
      Tensor quantized_tensor(DT_QUINT8, float_tensor.shape());
      FloatTensorToQuantizedInPlace<quint8>(float_tensor, range_min, range_max,
                                            &quantized_tensor);
      quantized_inputs.push_back({float_input.first, quantized_tensor});
    }

    GraphDef quantized_graph_def;
    TestTransformedVersusFloatGraph(
        QuantizeNodes, float_graph_def, float_inputs, quantized_inputs,
        output_names, context, 1.0, &quantized_graph_def);
  }

  void TestGraphWithFallbackRange(
      const GraphDef& float_graph_def,
      const std::vector<std::pair<string, Tensor>>& float_inputs,
      const std::vector<string>& output_names, float range_min, float range_max,
      GraphDef* quantized_graph_def) {
    TransformFuncContext context;
    context.params["fallback_min"] = {strings::StrCat(range_min)};
    context.params["fallback_max"] = {strings::StrCat(range_max)};
    TestTransformedVersusFloatGraph(QuantizeNodes, float_graph_def,
                                    float_inputs, float_inputs, output_names,
                                    context, 2.0, quantized_graph_def);
  }

  void TestIgnoreOps(std::initializer_list<string> ops_to_ignore) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    // A small helper to construct a Const op.
    auto const_op = [&](const string& name, const TensorShape& shape,
                        std::initializer_list<float> values) {
      Tensor tensor(DT_FLOAT, shape);
      test::FillValues<float>(&tensor, values);
      return Const(root.WithOpName(name), Input::Initializer(tensor));
    };

    // A simple graph with two different quantizable ops.
    int m = 1;
    int n = 1;
    int k = 1;
    Output a_op = const_op("a_op", {m, k}, {2});
    Output b_op = const_op("b_op", {k, n}, {3});
    Output c_op = const_op("c_op", {m, k}, {1});
    Output d_op = const_op("d_op", {k, n}, {4});
    Output mat_mul_op = MatMul(root.WithOpName("mat_mul_op"), a_op, b_op);
    Output mul_op = Mul(root.WithOpName("mul"), c_op, d_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TransformFuncContext context;
    if (ops_to_ignore.size() > 0) {
      context.params["ignore_op"] = ops_to_ignore;
    }

    GraphDef quantized_graph_def;
    TestTransformedVersusFloatGraph(QuantizeNodes, float_graph_def, {}, {},
                                    {"mat_mul_op", "mul"}, context, 1.0,
                                    &quantized_graph_def);

    // Make sure the quantized graph still contains the op that should have
    // been ignored by QuantizeNodes.
    for (const string& op_name : ops_to_ignore) {
      bool exists_in_quantized_graph = false;
      for (const NodeDef& node : quantized_graph_def.node()) {
        if (node.op() == op_name) {
          exists_in_quantized_graph = true;
          break;
        }
      }
      EXPECT_TRUE(exists_in_quantized_graph)
          << "Op " << op_name
          << " should not have been replace by a quantized version";
    }
  }

  void TestQuantizeMatMul(int m, int n, int k,
                          const std::vector<float>& a_values,
                          const std::vector<float>& b_values) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor a_tensor(DT_FLOAT, TensorShape({m, k}));
    test::FillValues<float>(&a_tensor, a_values);
    Output a_op = Const(root.WithOpName("a_op"), Input::Initializer(a_tensor));

    Tensor b_tensor(DT_FLOAT, TensorShape({k, n}));
    test::FillValues<float>(&b_tensor, b_values);
    Output b_op = Const(root.WithOpName("b_op"), Input::Initializer(b_tensor));

    Output mat_mul_op = MatMul(root.WithOpName("mat_mul_op"), a_op, b_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"mat_mul_op"});
  }

  void TestQuantizeMatMulTiny() {
    // These tests are added to test the generate case where
    // min(matrix) == max(matrix), which used to cause problems.
    TestQuantizeMatMul(1, 1, 1, {2}, {3});
    TestQuantizeMatMul(1, 2, 1, {1}, {2, 3});
    TestQuantizeMatMul(1, 1, 2, {1, 1}, {1, 1});
    TestQuantizeMatMul(1, 1, 2, {0, 0}, {1, 1});
    // The general case.
    TestQuantizeMatMul(1, 1, 2, {1, 2}, {1, 2});
  }

  void TestQuantizeMatMulSmall() {
    TestQuantizeMatMul(2, 4, 3, {1, 2, 3, 4, 5, 6},
                       {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  }

  void TestQuantizeMul() {
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    std::vector<int64> x_shape({10, 100});
    const size_t x_num_elements = TensorShape(x_shape).num_elements();
    std::vector<float> x_values(x_num_elements);
    for (int i = 0; i < x_num_elements; ++i) {
      x_values[i] = (i % 256) / 256.0f;
    }

    std::vector<int64> y_shape({100});
    const size_t y_num_elements = TensorShape(y_shape).num_elements();
    std::vector<float> y_values(y_num_elements);
    for (int i = 0; i < y_num_elements; ++i) {
      y_values[i] = ((i + 23) % 123) - 50;
    }

    Scope root = Scope::NewRootScope();

    Tensor x_float_tensor(DT_FLOAT, TensorShape(x_shape));
    test::FillValues<float>(&x_float_tensor, x_values);
    Output x = Const(root.WithOpName("x"), Input::Initializer(x_float_tensor));

    Tensor y_float_tensor(DT_FLOAT, TensorShape(y_shape));
    test::FillValues<float>(&y_float_tensor, y_values);
    Output y = Const(root.WithOpName("y"), Input::Initializer(y_float_tensor));

    Mul mul = Mul(root.WithOpName("mul"), x, y);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"mul"});
  }

  void TestQuantizeAdd() {
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    std::vector<int64> x_shape({10, 100});
    const size_t x_num_elements = TensorShape(x_shape).num_elements();
    std::vector<float> x_values(x_num_elements);
    for (int i = 0; i < x_num_elements; ++i) {
      x_values[i] = (i % 256) / 256.0f;
    }

    std::vector<int64> y_shape({100});
    const size_t y_num_elements = TensorShape(y_shape).num_elements();
    std::vector<float> y_values(y_num_elements);
    for (int i = 0; i < y_num_elements; ++i) {
      y_values[i] = ((i + 23) % 123) - 50;
    }

    Scope root = Scope::NewRootScope();

    Tensor x_float_tensor(DT_FLOAT, TensorShape(x_shape));
    test::FillValues<float>(&x_float_tensor, x_values);
    Output x = Const(root.WithOpName("x"), Input::Initializer(x_float_tensor));

    Tensor y_float_tensor(DT_FLOAT, TensorShape(y_shape));
    test::FillValues<float>(&y_float_tensor, y_values);
    Output y = Const(root.WithOpName("y"), Input::Initializer(y_float_tensor));

    Add add = Add(root.WithOpName("add"), x, y);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"add"});
  }

  void TestQuantizeConv2D(int depth, int input_width, int input_height,
                          int input_batch_count, int filter_size,
                          int filter_count, int stride, const string& padding,
                          const std::vector<float>& input_values,
                          const std::vector<float>& filter_values) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_tensor(DT_FLOAT, TensorShape({input_batch_count, input_height,
                                               input_width, depth}));
    test::FillValues<float>(&input_tensor, input_values);
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_tensor));

    Tensor filter_tensor(
        DT_FLOAT, TensorShape({filter_size, filter_size, depth, filter_count}));
    test::FillValues<float>(&filter_tensor, filter_values);
    Output filter_op =
        Const(root.WithOpName("filter_op"), Input::Initializer(filter_tensor));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, filter_op,
                            {1, stride, stride, 1}, padding);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"conv_op"});
  }

  void TestQuantizeBiasAdd() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_tensor(DT_FLOAT, TensorShape({1, 1, 2, 6}));
    test::FillIota<float>(&input_tensor, 1);
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_tensor));

    Tensor offset_tensor(DT_FLOAT, TensorShape({6}));
    test::FillIota<float>(&offset_tensor, 1);
    Output offset_op =
        Const(root.WithOpName("offset_op"), Input::Initializer(offset_tensor));

    Output bias_add_op =
        BiasAdd(root.WithOpName("bias_add_op"), input_op, offset_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"bias_add_op"});
  }

  void TestQuantizeConcat() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor shape_tensor(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&shape_tensor, {0});
    Output shape_op =
        Const(root.WithOpName("shape_op"), Input::Initializer(shape_tensor));

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2, 3}));
    test::FillValues<float>(&a_tensor, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Output a_op = Const(root.WithOpName("a_op"), Input::Initializer(a_tensor));

    Tensor b_tensor(DT_FLOAT, TensorShape({2, 2, 3}));
    test::FillValues<float>(&b_tensor,
                            {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    Output b_op = Const(root.WithOpName("b_op"), Input::Initializer(b_tensor));

    Output concat_op =
        Concat(root.WithOpName("concat_op"), {a_op, b_op}, shape_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"concat_op"});
  }

  void TestQuantizeRelu() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor constant_tensor(DT_FLOAT, TensorShape({1, 2, 6, 1}));
    test::FillValues<float>(&constant_tensor,
                            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Output constant_op = Const(root.WithOpName("constant_op"),
                               Input::Initializer(constant_tensor));

    Output relu_op = Relu(root.WithOpName("relu_op"), constant_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"relu_op"});
  }

  void TestQuantizeRelu6() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor constant_tensor(DT_FLOAT, TensorShape({1, 2, 6, 1}));
    test::FillValues<float>(&constant_tensor,
                            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Output constant_op = Const(root.WithOpName("constant_op"),
                               Input::Initializer(constant_tensor));

    Output relu6_op = Relu6(root.WithOpName("relu6_op"), constant_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"relu6_op"});
  }

  void TestQuantizeMaxPool() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor constant_tensor(DT_FLOAT, TensorShape({1, 2, 6, 1}));
    test::FillValues<float>(&constant_tensor,
                            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Output constant_op = Const(root.WithOpName("constant_op"),
                               Input::Initializer(constant_tensor));

    Output max_pool_op = MaxPool(root.WithOpName("max_pool_op"), constant_op,
                                 {1, 2, 2, 1}, {1, 1, 1, 1}, "SAME");

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"max_pool_op"});
  }

  void TestQuantizeAvgPool() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor constant_tensor(DT_FLOAT, TensorShape({1, 2, 6, 1}));
    test::FillValues<float>(&constant_tensor,
                            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Output constant_op = Const(root.WithOpName("constant_op"),
                               Input::Initializer(constant_tensor));

    Output avg_pool_op = AvgPool(root.WithOpName("avg_pool_op"), constant_op,
                                 {1, 2, 2, 1}, {1, 1, 1, 1}, "SAME");

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"avg_pool_op"});
  }

  void TestQuantizeReshape() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor constant_tensor(DT_FLOAT, TensorShape({4, 5}));
    test::FillValues<float>(&constant_tensor,
                            {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                             11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    Output constant_op = Const(root.WithOpName("constant_op"),
                               Input::Initializer(constant_tensor));

    Output reshape_op =
        Reshape(root.WithOpName("reshape_op"), constant_op, {10, 2});

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TestQuantizedVersusFloatGraph(float_graph_def, {}, {"reshape_op"});
  }

  void TestRemoveRedundantQuantization() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor quantized_tensor(DT_QUINT8, TensorShape({}));
    test::FillValues<quint8>(&quantized_tensor, {0});
    Output quantized_op = Const(root.WithOpName("quantized_op"),
                                Input::Initializer(quantized_tensor));

    Tensor quantized_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantized_min_tensor, {2.0f});
    Output quantized_min_op = Const(root.WithOpName("quantized_min_op"),
                                    Input::Initializer(quantized_min_tensor));

    Tensor quantized_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantized_max_tensor, {2.0f});
    Output quantized_max_op = Const(root.WithOpName("quantized_max_op"),
                                    Input::Initializer(quantized_min_tensor));

    Output dequantize_op =
        Dequantize(root.WithOpName("dequantize_op"), quantized_op,
                   quantized_min_op, quantized_max_op);

    Tensor dequantize_reshape_dims_tensor(DT_INT32, TensorShape({1}));
    test::FillValues<int32>(&dequantize_reshape_dims_tensor, {-1});
    Output dequantize_reshape_dims =
        Const(root.WithOpName("dequantize_reshape_dims"),
              Input::Initializer(dequantize_reshape_dims_tensor));

    Tensor dequantize_reduction_dims_tensor(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&dequantize_reduction_dims_tensor, {0});
    Output dequantize_reduction_dims =
        Const(root.WithOpName("dequantize_reduction_dims"),
              Input::Initializer(dequantize_reduction_dims_tensor));

    Output dequantize_reshape = Reshape(root.WithOpName("dequantize_reshape"),
                                        dequantize_op, dequantize_reshape_dims);

    Output dequantize_min =
        Min(root.WithOpName("dequantize_min"), dequantize_reshape,
            dequantize_reduction_dims, Min::Attrs().KeepDims(false));

    Output dequantize_max =
        Max(root.WithOpName("dequantize_max"), dequantize_reshape,
            dequantize_reduction_dims, Max::Attrs().KeepDims(false));

    QuantizeV2 quantize_op(root.WithOpName("quantize_op"), dequantize_op,
                           dequantize_min, dequantize_max, DT_QUINT8,
                           QuantizeV2::Attrs().Mode("MIN_FIRST"));

    Output final_dequantize =
        Dequantize(root.WithOpName("final_dequantize"), quantize_op.output,
                   quantize_op.output_min, quantize_op.output_max);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef removed_graph_def;
    TestTransformedVersusFloatGraph(
        RemoveRedundantQuantizations, float_graph_def, {}, {},
        {"final_dequantize"}, {}, 1.0, &removed_graph_def);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(removed_graph_def, &node_map);
    EXPECT_EQ(1, node_map.count("final_dequantize"));
    EXPECT_EQ("quantized_op", node_map.at("final_dequantize")->input(0));
  }

  void TestRemoveRedundantQuantizationWithBiasAdd() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor quantized_tensor(DT_QUINT8, TensorShape({1, 6}));
    test::FillValues<quint8>(&quantized_tensor, {0, 0, 0, 0, 0, 0});
    Output quantized_op = Const(root.WithOpName("quantized_op"),
                                Input::Initializer(quantized_tensor));

    Tensor quantized_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantized_min_tensor, {2.0f});
    Output quantized_min_op = Const(root.WithOpName("quantized_min_op"),
                                    Input::Initializer(quantized_min_tensor));

    Tensor quantized_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantized_max_tensor, {2.0f});
    Output quantized_max_op = Const(root.WithOpName("quantized_max_op"),
                                    Input::Initializer(quantized_min_tensor));

    Tensor offset_tensor(DT_QUINT8, TensorShape({6}));
    test::FillValues<quint8>(&offset_tensor, {1, 2, 3, 4, 5, 6});
    Output offset_op =
        Const(root.WithOpName("offset_op"), Input::Initializer(offset_tensor));

    Tensor offset_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&offset_min_tensor, {0.0f});
    Output offset_min_op = Const(root.WithOpName("offset_min_op"),
                                 Input::Initializer(offset_min_tensor));

    Tensor offset_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&offset_max_tensor, {255.0f});
    Output offset_max_op = Const(root.WithOpName("offset_max_op"),
                                 Input::Initializer(offset_max_tensor));

    QuantizedBiasAdd quantized_bias_add_op(
        root.WithOpName("bias_add_op"), quantized_op, offset_op,
        quantized_min_op, quantized_max_op, offset_min_op, offset_max_op,
        DT_QINT32);

    RequantizationRange requantization_range_op(
        root.WithOpName("requantization_range_op"),
        quantized_bias_add_op.output, quantized_bias_add_op.min_out,
        quantized_bias_add_op.max_out);

    Requantize requantize_op(
        root.WithOpName("requantize_op"), quantized_bias_add_op.output,
        quantized_bias_add_op.min_out, quantized_bias_add_op.max_out,
        requantization_range_op.output_min, requantization_range_op.output_max,
        DT_QUINT8);

    Output dequantize_op =
        Dequantize(root.WithOpName("dequantize_op"), requantize_op.output,
                   requantize_op.output_min, requantize_op.output_max);

    Tensor dequantize_reshape_dims_tensor(DT_INT32, TensorShape({1}));
    test::FillValues<int32>(&dequantize_reshape_dims_tensor, {-1});
    Output dequantize_reshape_dims =
        Const(root.WithOpName("dequantize_reshape_dims"),
              Input::Initializer(dequantize_reshape_dims_tensor));

    Tensor dequantize_reduction_dims_tensor(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&dequantize_reduction_dims_tensor, {0});
    Output dequantize_reduction_dims =
        Const(root.WithOpName("dequantize_reduction_dims"),
              Input::Initializer(dequantize_reduction_dims_tensor));

    Output dequantize_reshape = Reshape(root.WithOpName("dequantize_reshape"),
                                        dequantize_op, dequantize_reshape_dims);

    Output dequantize_min =
        Min(root.WithOpName("dequantize_min"), dequantize_reshape,
            dequantize_reduction_dims, Min::Attrs().KeepDims(false));

    Output dequantize_max =
        Max(root.WithOpName("dequantize_max"), dequantize_reshape,
            dequantize_reduction_dims, Max::Attrs().KeepDims(false));

    QuantizeV2 quantize_op(root.WithOpName("quantize_op"), dequantize_op,
                           dequantize_min, dequantize_max, DT_QUINT8,
                           QuantizeV2::Attrs().Mode("MIN_FIRST"));

    Output final_dequantize =
        Dequantize(root.WithOpName("final_dequantize"), quantize_op.output,
                   quantize_op.output_min, quantize_op.output_max);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef removed_graph_def;
    TestTransformedVersusFloatGraph(
        RemoveRedundantQuantizations, float_graph_def, {}, {},
        {"final_dequantize"}, {}, 1.0, &removed_graph_def);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(removed_graph_def, &node_map);
    EXPECT_EQ(1, node_map.count("final_dequantize"));
    EXPECT_EQ("requantize_op", node_map.at("final_dequantize")->input(0));
  }

  void TestQuantizeResizeBilinear() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor size_tensor(DT_INT32, TensorShape({2}));
    test::FillValues<int32>(&size_tensor, {256, 256});

    Output constant_op = Const(root.WithOpName("size_tensor_op"),
                               Input::Initializer(size_tensor));

    Output placeholder_op =
        Placeholder(root.WithOpName("placeholder_op"), DT_FLOAT);

    Output resize_bilinear_op = ResizeBilinear(
        root.WithOpName("resize_bilinear_op"), placeholder_op, constant_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    Tensor input_tensor(DT_FLOAT, {1, 128, 128, 3});
    test::FillFn<float>(&input_tensor, [](int) { return 100.0f; });

    TestQuantizedVersusFloatGraph(float_graph_def,
                                  {{"placeholder_op", input_tensor}},
                                  {"resize_bilinear_op"});
  }

  void TestRemoveRedundantQuantizationWithMultipleOutputs() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor quantized_tensor(DT_QUINT8, TensorShape({1, 6}));
    test::FillValues<quint8>(&quantized_tensor, {0, 0, 0, 0, 0, 0});
    Output quantized_op = Const(root.WithOpName("quantized_op"),
                                Input::Initializer(quantized_tensor));

    Tensor quantized_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantized_min_tensor, {2.0f});
    Output quantized_min_op = Const(root.WithOpName("quantized_min_op"),
                                    Input::Initializer(quantized_min_tensor));

    Tensor quantized_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantized_max_tensor, {2.0f});
    Output quantized_max_op = Const(root.WithOpName("quantized_max_op"),
                                    Input::Initializer(quantized_min_tensor));

    Tensor offset_tensor(DT_QUINT8, TensorShape({6}));
    test::FillValues<quint8>(&offset_tensor, {1, 2, 3, 4, 5, 6});
    Output offset_op =
        Const(root.WithOpName("offset_op"), Input::Initializer(offset_tensor));

    Tensor offset_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&offset_min_tensor, {0.0f});
    Output offset_min_op = Const(root.WithOpName("offset_min_op"),
                                 Input::Initializer(offset_min_tensor));

    Tensor offset_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&offset_max_tensor, {255.0f});
    Output offset_max_op = Const(root.WithOpName("offset_max_op"),
                                 Input::Initializer(offset_max_tensor));

    QuantizedBiasAdd quantized_bias_add_op(
        root.WithOpName("bias_add_op"), quantized_op, offset_op,
        quantized_min_op, quantized_max_op, offset_min_op, offset_max_op,
        DT_QINT32);

    RequantizationRange requantization_range_op(
        root.WithOpName("requantization_range_op"),
        quantized_bias_add_op.output, quantized_bias_add_op.min_out,
        quantized_bias_add_op.max_out);

    Requantize requantize_op(
        root.WithOpName("requantize_op"), quantized_bias_add_op.output,
        quantized_bias_add_op.min_out, quantized_bias_add_op.max_out,
        requantization_range_op.output_min, requantization_range_op.output_max,
        DT_QUINT8);

    Output dequantize_op =
        Dequantize(root.WithOpName("dequantize_op"), requantize_op.output,
                   requantize_op.output_min, requantize_op.output_max);

    Tensor dequantize_reshape_dims_tensor(DT_INT32, TensorShape({1}));
    test::FillValues<int32>(&dequantize_reshape_dims_tensor, {-1});
    Output dequantize_reshape_dims =
        Const(root.WithOpName("dequantize_reshape_dims"),
              Input::Initializer(dequantize_reshape_dims_tensor));

    Tensor dequantize_reduction_dims_tensor(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&dequantize_reduction_dims_tensor, {0});
    Output dequantize_reduction_dims =
        Const(root.WithOpName("dequantize_reduction_dims"),
              Input::Initializer(dequantize_reduction_dims_tensor));

    Output dequantize_reshape = Reshape(root.WithOpName("dequantize_reshape"),
                                        dequantize_op, dequantize_reshape_dims);

    Output dequantize_min =
        Min(root.WithOpName("dequantize_min"), dequantize_reshape,
            dequantize_reduction_dims, Min::Attrs().KeepDims(false));

    Output dequantize_max =
        Max(root.WithOpName("dequantize_max"), dequantize_reshape,
            dequantize_reduction_dims, Max::Attrs().KeepDims(false));

    QuantizeV2 quantize_op(root.WithOpName("quantize_op"), dequantize_op,
                           dequantize_min, dequantize_max, DT_QUINT8,
                           QuantizeV2::Attrs().Mode("MIN_FIRST"));

    Output final_dequantize =
        Dequantize(root.WithOpName("final_dequantize"), quantize_op.output,
                   quantize_op.output_min, quantize_op.output_max);

    Output relu_op = Relu(root.WithOpName("relu_op"), dequantize_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef removed_graph_def;
    TestTransformedVersusFloatGraph(
        RemoveRedundantQuantizations, float_graph_def, {}, {},
        {"final_dequantize", "relu_op"}, {}, 1.0, &removed_graph_def);

    std::map<string, int> op_type_count;
    for (const NodeDef& node : removed_graph_def.node()) {
      ++op_type_count[node.op()];
    }
    EXPECT_EQ(2, op_type_count["Dequantize"]);
  }

  void TestQuantizePlaceholders() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Output placeholder_op =
        Placeholder(root.WithOpName("placeholder_op"), DT_FLOAT);

    Output relu_op = Relu(root.WithOpName("relu_op"), placeholder_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    TransformFuncContext context;
    context.input_names = {"placeholder_op"};
    context.output_names = {"relu_op"};
    context.params = {{"input_min", {"-10.0"}}, {"input_max", {"10.0"}}};

    GraphDef quantized_graph_def;
    TF_ASSERT_OK(
        QuantizePlaceholders(float_graph_def, context, &quantized_graph_def));

    Tensor input_tensor(DT_FLOAT, {});
    input_tensor.flat<float>()(0) = 5.0f;

    TestQuantizedVersusFloatGraph(
        float_graph_def, {{"placeholder_op", input_tensor}}, {"relu_op"});

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(quantized_graph_def, &node_map);
    EXPECT_NE("placeholder_op", node_map.at("relu_op")->input(0));
    EXPECT_EQ("Placeholder", node_map.at("placeholder_op")->op());
    EXPECT_EQ(DT_QUINT8,
              node_map.at("placeholder_op")->attr().at("dtype").type());
  }

  void TestInputRange() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({1, width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output bias_add =
        BiasAdd(root.WithOpName("bias_add"), a_const, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    Tensor placeholder_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&placeholder_tensor, 1.0f);

    TestGraphWithInputRange(graph_def, {{"placeholder", placeholder_tensor}},
                            {"bias_add"}, 0.0f, 100.0f);
  }

  void TestFallbackRange() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({1, width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output bias_add =
        BiasAdd(root.WithOpName("bias_add"), a_const, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    Tensor placeholder_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&placeholder_tensor, 1.0f);

    GraphDef quantized_graph_def;
    TestGraphWithFallbackRange(graph_def, {{"placeholder", placeholder_tensor}},
                               {"bias_add"}, 0.0f, 200.0f,
                               &quantized_graph_def);

    for (const NodeDef& node : quantized_graph_def.node()) {
      EXPECT_NE("RequantizationRange", node.op());
    }
  }

  void TestConvertFakeQuantsToRequantize() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_tensor(DT_FLOAT, TensorShape({1, 1, 2, 6}));
    test::FillIota<float>(&input_tensor, 1);
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_tensor));

    Tensor offset_tensor(DT_FLOAT, TensorShape({6}));
    test::FillIota<float>(&offset_tensor, 1);
    Output offset_op =
        Const(root.WithOpName("offset_op"), Input::Initializer(offset_tensor));

    Output bias_add_op =
        BiasAdd(root.WithOpName("bias_add_op"), input_op, offset_op);

    Tensor fake_quant_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&fake_quant_min_tensor, {0.0f});
    Output fake_quant_min_op = Const(root.WithOpName("fake_quant_min_op"),
                                     Input::Initializer(fake_quant_min_tensor));

    Tensor fake_quant_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&fake_quant_max_tensor, {18.0f});
    Output fake_quant_max_op = Const(root.WithOpName("fake_quant_max_op"),
                                     Input::Initializer(fake_quant_max_tensor));

    Output fake_quant_op =
        FakeQuantWithMinMaxVars(root.WithOpName("fake_quant_op"), bias_add_op,
                                fake_quant_min_op, fake_quant_max_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef converted_graph_def;
    TestTransformedVersusFloatGraph(ConvertFakeQuantsToRequantize,
                                    float_graph_def, {}, {}, {"fake_quant_op"},
                                    {}, 1.0, &converted_graph_def);

    for (const NodeDef& node : converted_graph_def.node()) {
      EXPECT_NE("FakeQuantWithMinMaxVars", node.op());
    }
  }

  void TestMergeAdjacentRequantizes() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_tensor(DT_QUINT8, TensorShape({1, 1, 2, 6}));
    test::FillValues<quint8>(&input_tensor,
                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_tensor));

    Tensor input_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&input_min_tensor, {0.0f});
    Output input_min_op = Const(root.WithOpName("input_min_op"),
                                Input::Initializer(input_min_tensor));

    Tensor input_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&input_max_tensor, {255.0f});
    Output input_max_op = Const(root.WithOpName("input_max_op"),
                                Input::Initializer(input_max_tensor));

    Tensor offset_tensor(DT_QUINT8, TensorShape({6}));
    test::FillValues<quint8>(&offset_tensor, {1, 2, 3, 4, 5, 6});
    Output offset_op =
        Const(root.WithOpName("offset_op"), Input::Initializer(offset_tensor));

    Tensor offset_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&offset_min_tensor, {0.0f});
    Output offset_min_op = Const(root.WithOpName("offset_min_op"),
                                 Input::Initializer(offset_min_tensor));

    Tensor offset_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&offset_max_tensor, {255.0f});
    Output offset_max_op = Const(root.WithOpName("offset_max_op"),
                                 Input::Initializer(offset_max_tensor));

    QuantizedBiasAdd quantized_bias_add_op(
        root.WithOpName("quantized_bias_add_op"), input_op, offset_op,
        input_min_op, input_max_op, offset_min_op, offset_max_op, DT_QINT32);

    RequantizationRange requantization_range_op(
        root.WithOpName("requantization_range_op"),
        quantized_bias_add_op.output, quantized_bias_add_op.min_out,
        quantized_bias_add_op.max_out);

    Requantize requantize_op(
        root.WithOpName("requantize_op"), quantized_bias_add_op.output,
        quantized_bias_add_op.min_out, quantized_bias_add_op.max_out,
        requantization_range_op.output_min, requantization_range_op.output_max,
        DT_QUINT8);

    Output dequantize_op =
        Dequantize(root.WithOpName("dequantize_op"), requantize_op.output,
                   requantize_op.output_min, requantize_op.output_max,
                   Dequantize::Attrs().Mode("MIN_FIRST"));

    Tensor quantize_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantize_min_tensor, {0.0f});
    Output quantize_min_op = Const(root.WithOpName("quantize_min_op"),
                                   Input::Initializer(quantize_min_tensor));

    Tensor quantize_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantize_max_tensor, {255.0f});
    Output quantize_max_op = Const(root.WithOpName("quantize_max_op"),
                                   Input::Initializer(quantize_max_tensor));

    QuantizeV2 quantize_op(root.WithOpName("quantize_op"), dequantize_op,
                           quantize_min_op, quantize_max_op, DT_QINT32,
                           QuantizeV2::Attrs().Mode("MIN_FIRST"));

    Tensor fake_requantize_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&fake_requantize_min_tensor, {0.0f});
    Output fake_requantize_min_op =
        Const(root.WithOpName("fake_requantize_min_op"),
              Input::Initializer(fake_requantize_min_tensor));

    Tensor fake_requantize_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&fake_requantize_max_tensor, {255.0f});
    Output fake_requantize_max_op =
        Const(root.WithOpName("fake_requantize_max_op"),
              Input::Initializer(fake_requantize_max_tensor));

    Requantize fake_requantize_op(
        root.WithOpName("fake_requantize_op"), quantize_op.output,
        quantize_op.output_min, quantize_op.output_max, fake_requantize_min_op,
        fake_requantize_max_op, DT_QUINT8);

    Output fake_dequantize_op = Dequantize(
        root.WithOpName("fake_dequantize_op"), fake_requantize_op.output,
        fake_requantize_op.output_min, fake_requantize_op.output_max,
        Dequantize::Attrs().Mode("MIN_FIRST"));

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef converted_graph_def;
    TestTransformedVersusFloatGraph(MergeAdjacentRequantizes, float_graph_def,
                                    {}, {}, {"fake_dequantize_op"}, {}, 1.0,
                                    &converted_graph_def);

    int requantize_count = 0;
    for (const NodeDef& node : converted_graph_def.node()) {
      if (node.op() == "Requantize") {
        ++requantize_count;
      }
    }
    EXPECT_EQ(1, requantize_count);
  }

  void TestConvertFakeQuantsEndToEnd() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_tensor(DT_FLOAT, TensorShape({1, 1, 2, 6}));
    test::FillIota<float>(&input_tensor, 1);
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_tensor));

    Tensor offset_tensor(DT_FLOAT, TensorShape({6}));
    test::FillIota<float>(&offset_tensor, 1);
    Output offset_op =
        Const(root.WithOpName("offset_op"), Input::Initializer(offset_tensor));

    Output bias_add_op =
        BiasAdd(root.WithOpName("bias_add_op"), input_op, offset_op);

    Tensor fake_quant_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&fake_quant_min_tensor, {0.0f});
    Output fake_quant_min_op = Const(root.WithOpName("fake_quant_min_op"),
                                     Input::Initializer(fake_quant_min_tensor));

    Tensor fake_quant_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&fake_quant_max_tensor, {18.0f});
    Output fake_quant_max_op = Const(root.WithOpName("fake_quant_max_op"),
                                     Input::Initializer(fake_quant_max_tensor));

    Output fake_quant_op =
        FakeQuantWithMinMaxVars(root.WithOpName("fake_quant_op"), bias_add_op,
                                fake_quant_min_op, fake_quant_max_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef converted_graph_def;
    TestTransformedVersusFloatGraph(QuantizeNodes, float_graph_def, {}, {},
                                    {"fake_quant_op"}, {}, 1.0,
                                    &converted_graph_def);

    int requantize_count = 0;
    for (const NodeDef& node : converted_graph_def.node()) {
      EXPECT_NE("FakeQuantWithMinMaxVars", node.op());
      if (node.op() == "Requantize") {
        ++requantize_count;
      }
    }
    EXPECT_EQ(1, requantize_count);
  }

  void TestHoistFakeQuants() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_tensor(DT_FLOAT, TensorShape({1, 1, 2, 6}));
    test::FillIota<float>(&input_tensor, 1);
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_tensor));

    Tensor offset_tensor(DT_FLOAT, TensorShape({6}));
    test::FillIota<float>(&offset_tensor, 1);
    Output offset_op =
        Const(root.WithOpName("offset_op"), Input::Initializer(offset_tensor));

    Output bias_add_op =
        BiasAdd(root.WithOpName("bias_add_op"), input_op, offset_op);

    Output relu_op = Relu(root.WithOpName("relu_op"), bias_add_op);

    Output max_pool_op = MaxPool(root.WithOpName("max_pool_op"), relu_op,
                                 {1, 2, 2, 1}, {1, 1, 1, 1}, "SAME");

    Tensor fake_quant_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&fake_quant_min_tensor, {0.0f});
    Output fake_quant_min_op = Const(root.WithOpName("fake_quant_min_op"),
                                     Input::Initializer(fake_quant_min_tensor));

    Tensor fake_quant_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&fake_quant_max_tensor, {18.0f});
    Output fake_quant_max_op = Const(root.WithOpName("fake_quant_max_op"),
                                     Input::Initializer(fake_quant_max_tensor));

    Output fake_quant_op =
        FakeQuantWithMinMaxVars(root.WithOpName("fake_quant_op"), max_pool_op,
                                fake_quant_min_op, fake_quant_max_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef converted_graph_def;
    TestTransformedVersusFloatGraph(HoistFakeQuants, float_graph_def, {}, {},
                                    {"fake_quant_op"}, {}, 1.0,
                                    &converted_graph_def);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(converted_graph_def, &node_map);
    EXPECT_EQ("MaxPool", node_map.at("fake_quant_op")->op());
    EXPECT_EQ("FakeQuantWithMinMaxVars",
              node_map.at(node_map.at("relu_op")->input(0))->op());
  }

  void TestMergeDuplicateQuantizes() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor quantized_tensor(DT_QUINT8, TensorShape({}));
    test::FillValues<quint8>(&quantized_tensor, {0});
    Output quantized_op = Const(root.WithOpName("quantized_op"),
                                Input::Initializer(quantized_tensor));

    Tensor quantized_min_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantized_min_tensor, {2.0f});
    Output quantized_min_op = Const(root.WithOpName("quantized_min_op"),
                                    Input::Initializer(quantized_min_tensor));

    Tensor quantized_max_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&quantized_max_tensor, {2.0f});
    Output quantized_max_op = Const(root.WithOpName("quantized_max_op"),
                                    Input::Initializer(quantized_min_tensor));

    Output dequantize_op =
        Dequantize(root.WithOpName("dequantize_op"), quantized_op,
                   quantized_min_op, quantized_max_op);

    Tensor quantize_reshape_dims1_tensor(DT_INT32, TensorShape({1}));
    test::FillValues<int32>(&quantize_reshape_dims1_tensor, {-1});
    Output quantize_reshape_dims1 =
        Const(root.WithOpName("dequantize_reshape_dims1"),
              Input::Initializer(quantize_reshape_dims1_tensor));

    Tensor quantize_reduction_dims1_tensor(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&quantize_reduction_dims1_tensor, {0});
    Output quantize_reduction_dims1 =
        Const(root.WithOpName("quantize_reduction_dims1"),
              Input::Initializer(quantize_reduction_dims1_tensor));

    Output quantize_reshape1 = Reshape(root.WithOpName("quantize_reshape1"),
                                       dequantize_op, quantize_reshape_dims1);

    Output quantize_min1 =
        Min(root.WithOpName("quantize_min1"), quantize_reshape1,
            quantize_reduction_dims1, Min::Attrs().KeepDims(false));

    Output quantize_max1 =
        Max(root.WithOpName("quantize_max1"), quantize_reshape1,
            quantize_reduction_dims1, Max::Attrs().KeepDims(false));

    QuantizeV2 quantize_op1(root.WithOpName("quantize_op1"), dequantize_op,
                            quantize_min1, quantize_max1, DT_QUINT8,
                            QuantizeV2::Attrs().Mode("MIN_FIRST"));

    Tensor quantize_reshape_dims2_tensor(DT_INT32, TensorShape({1}));
    test::FillValues<int32>(&quantize_reshape_dims2_tensor, {-1});
    Output quantize_reshape_dims2 =
        Const(root.WithOpName("dequantize_reshape_dims2"),
              Input::Initializer(quantize_reshape_dims2_tensor));

    Tensor quantize_reduction_dims2_tensor(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&quantize_reduction_dims2_tensor, {0});
    Output quantize_reduction_dims2 =
        Const(root.WithOpName("quantize_reduction_dims2"),
              Input::Initializer(quantize_reduction_dims2_tensor));

    Output quantize_reshape2 = Reshape(root.WithOpName("quantize_reshape2"),
                                       dequantize_op, quantize_reshape_dims2);

    Output quantize_min2 =
        Min(root.WithOpName("quantize_min2"), quantize_reshape2,
            quantize_reduction_dims2, Min::Attrs().KeepDims(false));

    Output quantize_max2 =
        Max(root.WithOpName("quantize_max2"), quantize_reshape2,
            quantize_reduction_dims2, Max::Attrs().KeepDims(false));

    QuantizeV2 quantize_op2(root.WithOpName("quantize_op2"), dequantize_op,
                            quantize_min1, quantize_max1, DT_QUINT8,
                            QuantizeV2::Attrs().Mode("MIN_FIRST"));

    Output final_dequantize1 =
        Dequantize(root.WithOpName("final_dequantize1"), quantize_op1.output,
                   quantize_op1.output_min, quantize_op1.output_max);

    Output final_dequantize2 =
        Dequantize(root.WithOpName("final_dequantize2"), quantize_op2.output,
                   quantize_op2.output_min, quantize_op2.output_max);

    Output add_op =
        Add(root.WithOpName("add_op"), final_dequantize1, final_dequantize2);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef merged_graph_def;
    TestTransformedVersusFloatGraph(MergeDuplicateNodes, float_graph_def, {},
                                    {}, {"add_op"}, {}, 1.0, &merged_graph_def);

    std::map<string, int> op_map;
    for (const NodeDef& node : merged_graph_def.node()) {
      ++op_map[node.op()];
    }
    EXPECT_EQ(1, op_map["QuantizeV2"]);
  }

  void TestMergeDuplicateConsts() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 10;

    Tensor a_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_tensor, 1.0f);
    Output a_op = Const(root.WithOpName("a_op"), Input::Initializer(a_tensor));

    Tensor b_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_tensor, 1.0f);
    Output b_op = Const(root.WithOpName("b_op"), Input::Initializer(b_tensor));

    Output add_op = Add(root.WithOpName("add_op"), a_op, b_op);

    Tensor c_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&c_tensor, 2.0f);
    Output c_op = Const(root.WithOpName("c_op"), Input::Initializer(c_tensor));

    Output mul_op = Mul(root.WithOpName("mul_op"), add_op, c_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef merged_graph_def;
    TestTransformedVersusFloatGraph(MergeDuplicateNodes, float_graph_def, {},
                                    {}, {"mul_op"}, {}, 1.0, &merged_graph_def);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(merged_graph_def, &node_map);
    EXPECT_EQ(1, (node_map.count("a_op") + node_map.count("b_op")));
    string remaining_const;
    if (node_map.count("a_op")) {
      remaining_const = "a_op";
    } else {
      remaining_const = "b_op";
    }
    EXPECT_EQ(remaining_const, node_map["add_op"]->input(0));
    EXPECT_EQ(remaining_const, node_map["add_op"]->input(1));
    EXPECT_EQ(1, node_map.count("c_op"));
    EXPECT_EQ("add_op", node_map["mul_op"]->input(0));
    EXPECT_EQ("c_op", node_map["mul_op"]->input(1));
  }

  void TestMergeDuplicatesNested() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 10;

    Tensor a_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_tensor, 1.0f);
    Output a_op = Const(root.WithOpName("a_op"), Input::Initializer(a_tensor));

    Output a_relu_op = Relu(root.WithOpName("a_relu_op"), a_op);

    Tensor b_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_tensor, 1.0f);
    Output b_op = Const(root.WithOpName("b_op"), Input::Initializer(b_tensor));

    Output b_relu_op = Relu(root.WithOpName("b_relu_op"), b_op);

    Output add_op = Add(root.WithOpName("add_op"), a_relu_op, b_relu_op);

    Tensor c_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&c_tensor, 2.0f);
    Output c_op = Const(root.WithOpName("c_op"), Input::Initializer(c_tensor));

    Output mul_op = Mul(root.WithOpName("mul_op"), add_op, c_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef merged_graph_def;
    TestTransformedVersusFloatGraph(MergeDuplicateNodes, float_graph_def, {},
                                    {}, {"mul_op"}, {}, 1.0, &merged_graph_def);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(merged_graph_def, &node_map);
    EXPECT_EQ(1, (node_map.count("a_op") + node_map.count("b_op")));
    EXPECT_EQ(1, (node_map.count("a_relu_op") + node_map.count("b_relu_op")));
    string remaining_relu;
    if (node_map.count("a_relu_op")) {
      remaining_relu = "a_relu_op";
    } else {
      remaining_relu = "b_relu_op";
    }
    EXPECT_EQ(remaining_relu, node_map["add_op"]->input(0));
    EXPECT_EQ(remaining_relu, node_map["add_op"]->input(1));
    EXPECT_EQ(1, node_map.count("c_op"));
    EXPECT_EQ("add_op", node_map["mul_op"]->input(0));
    EXPECT_EQ("c_op", node_map["mul_op"]->input(1));
  }

  void TestMergeDuplicatesInOut() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 10;

    Tensor a_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_tensor, 1.0f);
    Output a_op = Const(root.WithOpName("a_op"), Input::Initializer(a_tensor));

    Output a_relu_op = Relu(root.WithOpName("a_relu_op"), a_op);

    Tensor b_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_tensor, 1.0f);
    Output b_op = Const(root.WithOpName("b_op"), Input::Initializer(b_tensor));

    Output b_relu_op = Relu(root.WithOpName("b_relu_op"), b_op);

    Output add_op = Add(root.WithOpName("add_op"), a_relu_op, b_relu_op);

    Tensor c_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&c_tensor, 2.0f);
    Output c_op = Const(root.WithOpName("c_op"), Input::Initializer(c_tensor));

    Output mul_op1 = Mul(root.WithOpName("mul_op1"), add_op, c_op);
    Output mul_op2 = Mul(root.WithOpName("mul_op2"), add_op, c_op);
    Output mul_op3 = Mul(root.WithOpName("mul_op3"), add_op, c_op);

    Output final_mul_op =
        Mul(root.WithOpName("final_mul_op"), mul_op2, mul_op3);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef merged_graph_def;
    TestTransformedVersusFloatGraph(MergeDuplicateNodes, float_graph_def,
                                    {{"a_op", a_tensor}}, {{"a_op", a_tensor}},
                                    {"mul_op1", "final_mul_op"}, {}, 1.0,
                                    &merged_graph_def);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(merged_graph_def, &node_map);
    EXPECT_EQ(1, node_map.count("a_op"));
    EXPECT_EQ(1, node_map.count("b_op"));
    EXPECT_EQ(1, node_map.count("a_relu_op"));
    EXPECT_EQ(1, node_map.count("b_relu_op"));
    EXPECT_EQ(1, node_map.count("mul_op1"));
    EXPECT_EQ(1, node_map.count("final_mul_op"));
    EXPECT_EQ(1, (node_map.count("mul_op2") + node_map.count("mul_op3")));
    string remaining_mul;
    if (node_map.count("mul_op2")) {
      remaining_mul = "mul_op2";
    } else {
      remaining_mul = "mul_op3";
    }
    EXPECT_EQ(remaining_mul, node_map["final_mul_op"]->input(0));
    EXPECT_EQ(remaining_mul, node_map["final_mul_op"]->input(1));
    EXPECT_EQ(1, node_map.count("c_op"));
    EXPECT_EQ("add_op", node_map["mul_op1"]->input(0));
    EXPECT_EQ("c_op", node_map["mul_op1"]->input(1));
  }

  void TestExcludeNonFloat() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor int_constant_tensor(DT_INT32, TensorShape({4, 5}));
    test::FillIota<int32>(&int_constant_tensor, 1);
    Output int_constant = Const(root.WithOpName("int_constant"),
                                Input::Initializer(int_constant_tensor));

    Tensor float_constant_tensor(DT_FLOAT, TensorShape({4, 5}));
    test::FillIota<float>(&float_constant_tensor, 2.0f);
    Output float_constant = Const(root.WithOpName("float_constant"),
                                  Input::Initializer(float_constant_tensor));

    Output excluded_reshape_op =
        Reshape(root.WithOpName("excluded_reshape_op"), int_constant, {10, 2});

    Output included_reshape_op = Reshape(root.WithOpName("included_reshape_op"),
                                         float_constant, {10, 2});

    Output excluded_relu_op =
        Relu(root.WithOpName("excluded_relu_op"), excluded_reshape_op);

    Output excluded_float_caster = Cast(
        root.WithOpName("excluded_float_caster"), excluded_relu_op, DT_FLOAT);

    Output included_relu_op =
        Relu(root.WithOpName("included_relu_op"), included_reshape_op);

    GraphDef float_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&float_graph_def));

    GraphDef quantized_graph_def;
    TestTransformedVersusFloatGraph(
        QuantizeNodes, float_graph_def, {}, {},
        {"excluded_float_caster", "included_relu_op"}, {}, 1.0,
        &quantized_graph_def);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(quantized_graph_def, &node_map);
    ASSERT_EQ(1, node_map.count("excluded_reshape_op"));
    EXPECT_EQ("Reshape", node_map.at("excluded_reshape_op")->op());
    ASSERT_EQ(1, node_map.count("included_reshape_op"));
    EXPECT_EQ("Dequantize", node_map.at("included_reshape_op")->op());
  }
};

TEST_F(QuantizeNodesTest, TestIgnoreOps) {
  TestIgnoreOps({});
  TestIgnoreOps({"MatMul"});
  TestIgnoreOps({"MatMul", "Mul"});
}

TEST_F(QuantizeNodesTest, TestQuantizeMatMulTiny) { TestQuantizeMatMulTiny(); }

TEST_F(QuantizeNodesTest, TestQuantizeMatMulSmall) {
  TestQuantizeMatMulSmall();
}

TEST_F(QuantizeNodesTest, TestQuantizeMul) { TestQuantizeMul(); }

TEST_F(QuantizeNodesTest, TestQuantizeAdd) { TestQuantizeAdd(); }

TEST_F(QuantizeNodesTest, TestOddPaddingProblem) {
  // Tests one error case we ran into in a real graph.
  TestQuantizeConv2D(1, 4, 4, 1, 3, 1, 2, "SAME",
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9});
}

TEST_F(QuantizeNodesTest, TestQuantizeConv2D) {
  TestQuantizeConv2D(1, 4, 3, 1, 3, 1, 1, "SAME",
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                     {1, 4, 7, 2, 5, 8, 3, 6, 9});
}

TEST_F(QuantizeNodesTest, TestQuantizeBiasAdd) { TestQuantizeBiasAdd(); }

TEST_F(QuantizeNodesTest, TestQuantizeConcat) { TestQuantizeConcat(); }

TEST_F(QuantizeNodesTest, TestQuantizeRelu) { TestQuantizeRelu(); }

TEST_F(QuantizeNodesTest, TestQuantizeRelu6) { TestQuantizeRelu6(); }

TEST_F(QuantizeNodesTest, TestQuantizeMaxPool) { TestQuantizeMaxPool(); }

TEST_F(QuantizeNodesTest, TestQuantizeAvgPool) { TestQuantizeAvgPool(); }

TEST_F(QuantizeNodesTest, TestQuantizeReshape) { TestQuantizeReshape(); }

TEST_F(QuantizeNodesTest, TestQuantizeResizeBilinear) {
  TestQuantizeResizeBilinear();
}

TEST_F(QuantizeNodesTest, TestRemoveRedundantQuantization) {
  TestRemoveRedundantQuantization();
}

TEST_F(QuantizeNodesTest, TestRemoveRedundantQuantizationWithBiasAdd) {
  TestRemoveRedundantQuantizationWithBiasAdd();
}

TEST_F(QuantizeNodesTest, TestRemoveRedundantQuantizationWithMultipleOutputs) {
  TestRemoveRedundantQuantizationWithMultipleOutputs();
}

TEST_F(QuantizeNodesTest, TestQuantizePlaceholders) {
  TestQuantizePlaceholders();
}

TEST_F(QuantizeNodesTest, TestInputRange) { TestInputRange(); }

TEST_F(QuantizeNodesTest, TestFallbackRange) { TestFallbackRange(); }

TEST_F(QuantizeNodesTest, TestConvertFakeQuantsToRequantize) {
  TestConvertFakeQuantsToRequantize();
}

TEST_F(QuantizeNodesTest, TestMergeAdjacentRequantizes) {
  TestMergeAdjacentRequantizes();
}

TEST_F(QuantizeNodesTest, TestConvertFakeQuantsEndToEnd) {
  TestConvertFakeQuantsEndToEnd();
}

TEST_F(QuantizeNodesTest, TestHoistFakeQuants) { TestHoistFakeQuants(); }

TEST_F(QuantizeNodesTest, TestMergeDuplicateQuantizes) {
  TestMergeDuplicateQuantizes();
}

TEST_F(QuantizeNodesTest, TestMergeDuplicateConsts) {
  TestMergeDuplicateConsts();
}

TEST_F(QuantizeNodesTest, TestMergeDuplicatesNested) {
  TestMergeDuplicatesNested();
}

TEST_F(QuantizeNodesTest, TestMergeDuplicateInOut) {
  TestMergeDuplicatesInOut();
}

TEST_F(QuantizeNodesTest, TestExcludeNonFloat) { TestExcludeNonFloat(); }

}  // namespace graph_transforms
}  // namespace tensorflow
