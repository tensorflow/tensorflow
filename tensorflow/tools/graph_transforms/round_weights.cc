/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Rounds any large float constants to the specified number of levels.
Status RoundWeights(const GraphDef& input_graph_def,
                    const TransformFuncContext& context,
                    GraphDef* output_graph_def) {
  int32 num_steps;
  TF_RETURN_IF_ERROR(
      context.GetOneInt32Parameter("num_steps", 256, &num_steps));
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def, {"Const"},
      [num_steps](const NodeMatch& match, const std::set<string>& input_nodes,
                  const std::set<string>& output_nodes,
                  std::vector<NodeDef>* new_nodes) {
        const NodeDef& old_const_node = match.node;
        if (!old_const_node.attr().count("dtype")) {
          return errors::InvalidArgument("No 'dtype' attribute for Const node ",
                                         old_const_node.name());
        }
        if (!old_const_node.attr().count("value")) {
          return errors::InvalidArgument("No 'value' attribute for Const node ",
                                         old_const_node.name());
        }
        const DataType old_dtype = old_const_node.attr().at("dtype").type();
        Tensor old_tensor;
        if (!old_tensor.FromProto(old_const_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         old_const_node.name());
        }
        const size_t num_elements = old_tensor.NumElements();
        // If this isn't a float constant, or it's too small, then reuse the
        // same node with no changes. The size is important because small
        // constants tend to be used for more accuracy-sensitive calculations,
        // and the benefit of shrinking them is very marginal.
        if ((old_dtype != DT_FLOAT) || (num_elements < 16)) {
          new_nodes->push_back(old_const_node);
          return Status::OK();
        }
        const float* old_values = old_tensor.flat<float>().data();
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();
        for (int i = 0; i < num_elements; ++i) {
          const float value = old_values[i];
          min = std::min(min, value);
          max = std::max(max, value);
        }
        // min_value == max_value is a tricky case. It can occur for general
        // tensors, and of course for scalars. The quantized ops cannot deal
        // with this case, so we set max_value to something else.
        // It's a tricky question what is the numerically best solution to
        // deal with this degeneracy.
        // TODO(petewarden): Better use a tolerance than a hard comparison?
        if (min == max) {
          if (std::abs(min) < 0.000001f) {
            max = min + 1.0f;
          } else if (min > 0) {
            max = 2.0f * min;
          } else {
            min = 2.0f * max;
          }
        }
        Tensor rounded_tensor(DT_FLOAT, old_tensor.shape());
        float* rounded_values = rounded_tensor.flat<float>().data();
        const float bucket_width = (max - min) / num_steps;
        for (int i = 0; i < num_elements; ++i) {
          const int32 bucket = std::floor((old_values[i] - min) / bucket_width);
          rounded_values[i] = min + (bucket_width * (bucket + 0.5f));
        }

        NodeDef rounded_const_node;
        rounded_const_node.set_op("Const");
        rounded_const_node.set_name(old_const_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &rounded_const_node);
        SetNodeTensorAttr<float>("value", rounded_tensor, &rounded_const_node);
        new_nodes->push_back(rounded_const_node);

        return Status::OK();
      },
      {}, output_graph_def));

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("round_weights", RoundWeights);

}  // namespace graph_transforms
}  // namespace tensorflow
