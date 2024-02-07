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

#include "tensorflow/core/kernels/mkl/mkl_kernel_util.h"

#ifdef INTEL_MKL

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

void MklTestingUtil::RunMklQuantizeOp(const Tensor& input,
                                      const float input_min,
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

void MklTestingUtil::RunDequantizeOp(const Tensor& input,
                                     const Tensor& input_min,
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
  TF_CHECK_OK(root.ToGraphDef(&graph_def));
  RunGraph(graph_def, op_name, output);
}

void MklTestingUtil::RunGraph(const tensorflow::GraphDef graph_def,
                              const string& fetch, Tensor* output) {
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));

  std::vector<Tensor> output_tensors;
  TF_CHECK_OK(session->Run({}, {fetch}, {}, &output_tensors));

  *output = output_tensors[0];
}
#endif  // INTEL_MKL
}  // namespace tensorflow
