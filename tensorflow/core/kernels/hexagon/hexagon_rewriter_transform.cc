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

// Wraps the hexagon rewriter in a transform so it can be used as part of the
// graph transform tool.
// A usage example, based on the Image Understanding pipeline:
/*
bazel build tensorflow/tools/graph_transforms:transform_graph
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=/tmp/tensorflow_inception_v3_stripped_optimized_quantized.pb \
--out_graph=\
/tmp/tensorflow_inception_v3_stripped_optimized_quantized_fused_hexagon.pb \
--inputs='Mul' \
--outputs='softmax' \
--transforms='\
rewrite_quantized_stripped_model_for_hexagon(
input_shape0="1,299,299,3" \
input_type0="float" \
)'
*/

#include "tensorflow/core/kernels/hexagon/graph_transfer_utils.h"
#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {
constexpr const char* const INPUT_SHAPE_PREFIX = "input_shape";
constexpr const char* const INPUT_TYPE_PREFIX = "input_type";

Status RewriteQuantizedStrippedModelForHexagon(
    const GraphDef& input_graph_def, const TransformFuncContext& context,
    GraphDef* output_graph_def) {
  LOG(INFO) << "Transforming quantized stripped model to a remote fused "
               "graph execute op...";
  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<string> outputs;
  for (auto i = 0; i < context.input_names.size(); ++i) {
    const string& input_name = context.input_names.at(i);

    // Get input shape
    string shape_string;
    TF_RETURN_IF_ERROR(context.GetOneStringParameter(
        INPUT_SHAPE_PREFIX + std::to_string(i), "", &shape_string));
    std::vector<int64> dims;
    CHECK(str_util::SplitAndParseAsInts(shape_string, ',', &dims));

    // Get input data type
    string data_type_string;
    TF_RETURN_IF_ERROR(context.GetOneStringParameter(
        INPUT_TYPE_PREFIX + std::to_string(i), "", &data_type_string));
    DataType data_type;
    CHECK(DataTypeFromString(data_type_string, &data_type))
        << "\"" << data_type_string << "\" was an invalid type";

    LOG(INFO) << "Input(" << i << "): name = " << input_name
              << ", shape = " << shape_string
              << ", type = " << data_type_string;

    inputs.emplace_back(input_name, Tensor(data_type, TensorShape(dims)));
  }

  for (const string& output_name : context.output_names) {
    outputs.emplace_back(output_name);
  }
  GraphTransferer gt;
  gt.EnableStrictCheckMode(false);
  *output_graph_def = GraphTransferUtils::BuildFusedGraphDef(
      HexagonOpsDefinitions::getInstance(), "remote_fused_graph_execute_node",
      inputs, outputs, input_graph_def, &gt);
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("rewrite_quantized_stripped_model_for_hexagon",
                         RewriteQuantizedStrippedModelForHexagon);

}  // namespace graph_transforms
}  // namespace tensorflow
