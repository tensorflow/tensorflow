/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#if defined(INTEL_MKL)

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/tsl/platform/default/integral_types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

template <typename T>
class TransposeOpTest : public OpsTestBase {
 protected:
  void RunRegularTransposeOp(const Tensor& input,
                             const gtl::ArraySlice<int32_t> perm_list,
                             Tensor* result_tensor) {
    auto root = tensorflow::Scope::NewRootScope();
    Output x = ops::Const(root.WithOpName("input"), Input::Initializer(input));
    Output perm = ops::Const(root.WithOpName("perm"),
                             Input::Initializer(test::AsTensor(perm_list)));
    Output output = ops::Transpose(root.WithOpName("output"), x, perm);
    std::vector<Tensor> outputs;
    RunAndFetch(root, {output.name()}, &outputs);
    *result_tensor = outputs[0];
  }

  void RunQuantizedTransposeOp(const Tensor& input,
                               const gtl::ArraySlice<int32_t> perm_list,
                               const Tensor& min_input, const Tensor& max_input,
                               Tensor* result_tensor, Tensor* min_tensor,
                               Tensor* max_tensor) {
    DataType dtype = DataTypeToEnum<T>::v();
    auto root = tensorflow::Scope::NewRootScope();
    Output x = ops::Const(root.WithOpName("input"), Input::Initializer(input));
    Output perm = ops::Const(root.WithOpName("perm"),
                             Input::Initializer(test::AsTensor(perm_list)));
    Output min =
        ops::Const(root.WithOpName("min"), Input::Initializer(min_input));
    Output max =
        ops::Const(root.WithOpName("max"), Input::Initializer(max_input));

    NodeDef quant_transpose_node;
    std::vector<const NodeDef*> add_nodes;
    TF_EXPECT_OK(NodeDefBuilder("quant_transpose", "_QuantizedTranspose")
                     .Input({x.name(), 0, dtype})
                     .Input({perm.name(), 0, DT_INT32})
                     .Input({min.name(), 0, DT_FLOAT})
                     .Input({max.name(), 0, DT_FLOAT})
                     .Finalize(&quant_transpose_node));
    add_nodes = {&quant_transpose_node};
    std::vector<Tensor> outputs;
    RunAndFetch(root,
                {"quant_transpose:0", "quant_transpose:1", "quant_transpose:2"},
                &outputs, add_nodes);
    *result_tensor = outputs[0];
    *min_tensor = outputs[1];
    *max_tensor = outputs[2];
  }

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the outputs. Optional `add_nodes` parameter
  // allows to define nodes directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
  void RunAndFetch(const tensorflow::Scope& root,
                   const std::vector<string>& fetch,
                   std::vector<Tensor>* outputs,
                   const std::vector<const NodeDef*> add_nodes = {}) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    for (const NodeDef* add_node : add_nodes) {
      *graph.add_node() = *add_node;
    }

    // We really want to make sure that graph executed exactly as we passed it
    // to the session, so we disable various optimizations.
    tensorflow::SessionOptions session_options;

    // Disable common runtime constant folding.
    session_options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(OptimizerOptions::L0);

    // Disable Grappler optimizations for tests.
    tensorflow::RewriterConfig* cfg =
        session_options.config.mutable_graph_options()
            ->mutable_rewrite_options();
    cfg->set_constant_folding(tensorflow::RewriterConfig::OFF);
    cfg->set_layout_optimizer(tensorflow::RewriterConfig::OFF);
    cfg->set_remapping(tensorflow::RewriterConfig::OFF);

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(session_options));

    const string device = "/device:CPU:0";
    for (NodeDef& mutable_node : *graph.mutable_node()) {
      mutable_node.set_device(device);
    }

    TF_ASSERT_OK(session->Create(graph));
    TF_ASSERT_OK(session->Run({}, fetch, {}, outputs));
  }

  void VerifyQuantizedTranspose(const gtl::ArraySlice<int64_t>& input_dims,
                                const gtl::ArraySlice<int32_t>& permutation) {
    DataType dtype = DataTypeToEnum<T>::v();
    TensorShape input_shape = TensorShape(input_dims);
    std::vector<int64_t> output_dims(input_dims.size());
    for (size_t i = 0; i < input_dims.size(); ++i) {
      output_dims[i] = input_dims[permutation[i]];
    }
    TensorShape output_shape = TensorShape(output_dims);

    Tensor input(dtype, input_shape);
    input.flat<T>().setRandom();
    Tensor expected(dtype, output_shape);
    if (dtype == DT_QINT8) {
      // Regular Transpose Op does not accecpt qint8 input. It accepts
      // int8(signed char).
      Tensor int8_input(DT_INT8, input_shape);
      int8_input.flat<int8>() = input.flat<T>().template cast<int8>();
      Tensor int8_output(DT_INT8, output_shape);
      RunRegularTransposeOp(int8_input, permutation, &int8_output);
      expected.flat<T>() = int8_output.flat<int8>().template cast<T>();
    } else if (dtype == DT_QUINT8) {
      // Regular Transpose Op does not accecpt quint8 input. It accepts
      // uint8(unsgined char).
      Tensor uint8_input(DT_UINT8, input_shape);
      uint8_input.flat<uint8>() = input.flat<T>().template cast<uint8>();
      Tensor uint8_output(DT_UINT8, output_shape);
      RunRegularTransposeOp(uint8_input, permutation, &uint8_output);
      expected.flat<T>() = uint8_output.flat<uint8>().template cast<T>();
    }

    Tensor min_input(DT_FLOAT, TensorShape({}));
    Tensor max_input(DT_FLOAT, TensorShape({}));
    min_input.flat<float>()(0) = -3.5;  // The output should match this value.
    max_input.flat<float>()(0) = -4.5;  // The output should match this value.
    Tensor result(dtype, input_shape);
    Tensor min_result(DT_FLOAT, TensorShape({}));
    Tensor max_result(DT_FLOAT, TensorShape({}));
    RunQuantizedTransposeOp(input, permutation, min_input, max_input, &result,
                            &min_result, &max_result);
    test::ExpectEqual(expected, result);
    test::ExpectEqual(min_input, min_result);
    test::ExpectEqual(max_input, max_result);
  }
};

TYPED_TEST_SUITE_P(TransposeOpTest);

TYPED_TEST_P(TransposeOpTest, small) {
  this->VerifyQuantizedTranspose(/*input shape*/ {3, 4, 5, 6},
                                 /*permutation list*/ {0, 2, 1, 3});
}

REGISTER_TYPED_TEST_SUITE_P(TransposeOpTest, small);

using InputDataTypes = ::testing::Types<qint8, quint8>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, TransposeOpTest, InputDataTypes);

}  // namespace tensorflow

#endif  // INTEL_MKL
