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
#ifndef TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_MATMUL_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_MATMUL_BUILDER_H_

#include <vector>

#include "tensorflow/lite/delegates/hexagon/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace hexagon {

// Builder for FullyConnected op in Hexagon with weights as const.
class MatMulWithConstWeightsOpBuilder : public OpBuilder {
 public:
  explicit MatMulWithConstWeightsOpBuilder(GraphBuilder* graph_builder,
                                           int op_type)
      : OpBuilder(graph_builder, op_type) {}
  TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                const TfLiteIntArray* outputs,
                                TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

 private:
  TensorID node_output_;
  std::vector<int> weights_shape_, bias_shape_;
  std::vector<float> transposed_weights_;
  float weights_min_, weights_max_;
};

// Builder for FullyConnected op in Hexagon with non const weights.
class MatMulOpBuilder : public OpBuilder {
 public:
  explicit MatMulOpBuilder(GraphBuilder* graph_builder, int op_type)
      : OpBuilder(graph_builder, op_type) {}
  TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                const TfLiteIntArray* outputs,
                                TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

 private:
  // Adds Fully connected op related ops to the graph.
  TfLiteStatus AddFullyConnected(const TfLiteIntArray* inputs,
                                 const TfLiteIntArray* outputs,
                                 const TensorID weights_id,
                                 const TensorID weights_min_id,
                                 const TensorID weights_max_id,
                                 TfLiteContext* context, OpBuilder* matmul_op);

  TensorID node_output_;
  std::vector<int> weights_shape_, bias_shape_;
  std::vector<float> transposed_weights_;
  float weights_min_, weights_max_;
};

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_MATMUL_BUILDER_H_
