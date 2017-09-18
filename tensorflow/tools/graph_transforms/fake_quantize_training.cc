/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/quantize_training.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// EXPERIMENTAL: This can change without warning.
// Rewrites the GraphDef for quantized training.
// Rewrites the forward pass to include the precision loss with quantization so
// the model can learn to deal with such loss and achieve better accuracy when
// it is quantized later for inference.
// Quantization range information is collected in FakeQuantizeWithMinMaxVars
// ops.
//
// TODO(suharshs): Provide instructions on converting the resulting graph for
// inference.
// TODO(suharshs): Implement this using the GTT rather than calling the old
// prototype function.
Status FakeQuantizeTraining(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def) {
  // TODO(suharshs): Make num_bits a parameter.
  const int32 num_bits = 8;
  // TODO(suharshs): Make quantization op a parameter?
  const string quant_op_type = "FakeQuantWithMinMaxVars";

  return DoQuantizeTrainingOnGraphDef(input_graph_def, num_bits, quant_op_type,
                                      output_graph_def);
}

REGISTER_GRAPH_TRANSFORM("fake_quantize_training", FakeQuantizeTraining);

}  // namespace graph_transforms
}  // namespace tensorflow
