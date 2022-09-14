/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL

#ifdef INTEL_MKL

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

class MklTestingUtil {
 public:
  static void RunMklQuantizeOp(const Tensor& input, const float input_min,
                               const float input_max, DataType type,
                               string mode, Tensor* output) {
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    Node* input_node = test::graph::Constant(&*graph, input, "input");

    Tensor min(DT_FLOAT, TensorShape());
    Tensor max(DT_FLOAT, TensorShape());
    min.scalar<float>()() = input_min;
    max.scalar<float>()() = input_max;
    Node* min_node = test::graph::Constant(&*graph, Tensor(min), "min");
    Node* max_node = test::graph::Constant(&*graph, Tensor(max), "max");

    Node* quantize_op;
    TF_CHECK_OK(NodeBuilder("mkl_quantizeV2", "_MklQuantizeV2")
                    .Input(input_node)
                    .Input(min_node)
                    .Input(max_node)
                    .Attr("T", type)
                    .Attr("mode", mode)
                    .Attr("round_mode", "HALF_TO_EVEN")
                    .Attr("_kernel", "QuantizedMklOp")
                    .Finalize(&*graph, &quantize_op));

    GraphDef graph_def;
    graph->ToGraphDef(&graph_def);
    RunGraph(graph_def, "mkl_quantizeV2", output);
  }

  static void RunDequantizeOp(const Tensor& input, const Tensor& input_min,
                              const Tensor& input_max, string mode,
                              Tensor* output) {
    auto root = tensorflow::Scope::NewRootScope();
    string op_name = "dequantize_op";
    auto input_op =
        ops::Const(root.WithOpName("input"), Input::Initializer(input));
    auto input_min_op =
        ops::Const(root.WithOpName("input_min"), Input::Initializer(input_min));
    auto input_max_op =
        ops::Const(root.WithOpName("input_max"), Input::Initializer(input_max));

    ops::Dequantize::Attrs attrs;
    attrs = attrs.Mode(mode);

    auto out_op = ops::Dequantize(root.WithOpName(op_name), input_op,
                                  input_min_op, input_max_op, attrs);
    tensorflow::GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    RunGraph(graph_def, op_name, output);
  }

  static void RunGraph(const tensorflow::GraphDef graph_def,
                       const string& fetch, Tensor* output) {
    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(session->Create(graph_def));

    std::vector<Tensor> output_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &output_tensors));

    *output = output_tensors[0];
  }

  template <typename T>
  static void ComputeMinMax(const Tensor& tf_tensor, T* tenosr_min,
                            T* tensor_max) {
    auto eigen_tensor = tf_tensor.flat<T>();
    Eigen::Tensor<T, 0, Eigen::RowMajor> min = eigen_tensor.minimum();
    Eigen::Tensor<T, 0, Eigen::RowMajor> max = eigen_tensor.maximum();
    *tenosr_min = min();
    *tensor_max = max();
  }
};

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL
