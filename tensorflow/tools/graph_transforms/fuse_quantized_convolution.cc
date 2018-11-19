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
#ifdef INTEL_MKL
#include <algorithm>

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status FuseQuantizedConvolutionAndRequantize(
    const GraphDef& input_graph_def, const TransformFuncContext& context,
    GraphDef* output_graph_def) {
  std::map<string, const NodeDef*> node_map;
  MapNamesToNodes(input_graph_def, &node_map);
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off

      {"Requantize",
        {
          {"QuantizedConv2D|QuantizedConv2DWithBias|QuantizedConv2DWithRelu|"
            "QuantizedConv2DWithBiasAndRelu|QuantizedConv2DWithBiasSumAndRelu"},
          {"QuantizedConv2D|QuantizedConv2DWithBias|QuantizedConv2DWithRelu|"
           "QuantizedConv2DWithBiasAndRelu|QuantizedConv2DWithBiasSumAndRelu"},
          {"QuantizedConv2D|QuantizedConv2DWithBias|QuantizedConv2DWithRelu|"
           "QuantizedConv2DWithBiasAndRelu|QuantizedConv2DWithBiasSumAndRelu"},
          {"Const"},
          {"Const"}
        }
      },  // clang-format on */
      [&node_map](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        // TODO(mdfaijul/sheng): Current implementation assumed all
        // requantization cases have bias. Index of inputs need to be updated
        // for non-bias cases.

        // Find all the nodes we expect in the subgraph.
        const NodeDef& requantize_node = match.node;
        CHECK_EQ("Requantize", requantize_node.op());
        const NodeDef& quantized_conv2D_node = match.inputs[0].node;
        const NodeDef& const_requantize_range_min_node = match.inputs[3].node;
        CHECK_EQ("Const", const_requantize_range_min_node.op());
        const NodeDef& const_requantize_range_max_node = match.inputs[4].node;
        CHECK_EQ("Const", const_requantize_range_max_node.op());

        string quantized_conv2D_op_name = quantized_conv2D_node.op();
        // Set up the new fused version of the convolution op.
        NodeDef fused_conv;
        fused_conv.set_op(quantized_conv2D_op_name + "AndRequantize");
        fused_conv.set_name(match.node.name());
        int n_input = quantized_conv2D_node.input_size();
        if (quantized_conv2D_op_name.compare(
                "QuantizedConv2DWithBiasSumAndRelu") == 0)
          n_input -= 1;  // -1 since summand is moved after frozen min-max

        for (int i=0; i < n_input; i++)
          AddNodeInput(quantized_conv2D_node.input(i), &fused_conv);

        AddNodeInput(const_requantize_range_min_node.name(), &fused_conv);
        AddNodeInput(const_requantize_range_max_node.name(), &fused_conv);

        // Add additional inputs to
        // QuantizedConv2DWithBiasSumAndReluAndRequantize
        if (quantized_conv2D_op_name.compare(
              "QuantizedConv2DWithBiasSumAndRelu") == 0) {
          const NodeDef *in_requantize = node_map[node_map[
              quantized_conv2D_node.input(n_input)]->input(0)];
          string summand(in_requantize->name());
          string min_summand(in_requantize->name() + ":1");
          string max_summand(in_requantize->name() + ":2");
          AddNodeInput(summand, &fused_conv);
          AddNodeInput(min_summand, &fused_conv);
          AddNodeInput(max_summand, &fused_conv);

          // Signed version QuantizedConv2DWithBiasSumAndReluAndRequantize
          // if Relu does not follow the convolution operation
          std::vector<string> signed_ops = {
              "QuantizedConv2DWithBias",
              "QuantizedConv2D"
              };
          bool is_signed_summand =
              std::find(signed_ops.begin(), signed_ops.end(),
              node_map[in_requantize->input(0)]->op()) != signed_ops.end();
          if (is_signed_summand) {
            fused_conv.set_op(
                "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize");
            SetNodeAttr("Tsummand", DT_QINT8, &fused_conv);
          } else {
            SetNodeAttr("Tsummand", DT_QUINT8, &fused_conv);
          }
        }
        CopyNodeAttr(quantized_conv2D_node, "Tinput", "Tinput", &fused_conv);
        CopyNodeAttr(quantized_conv2D_node, "Tfilter", "Tfilter", &fused_conv);
        CopyNodeAttr(quantized_conv2D_node, "strides", "strides", &fused_conv);
        CopyNodeAttr(quantized_conv2D_node, "padding", "padding", &fused_conv);

        // Copy dilation attribute if exsit in the orginal node
        if (HasNodeAttr(quantized_conv2D_node, "dilations"))
          CopyNodeAttr(quantized_conv2D_node, "dilations",
                       "dilations", &fused_conv);
        if (quantized_conv2D_op_name.compare("QuantizedConv2D") == 0 ||
           quantized_conv2D_op_name.compare("QuantizedConv2DWithBias") == 0)
          SetNodeAttr("out_type", DT_QINT8, &fused_conv);
        else
          SetNodeAttr("out_type", DT_QUINT8, &fused_conv);
        new_nodes->push_back(fused_conv);
        new_nodes->push_back(const_requantize_range_min_node);
        new_nodes->push_back(const_requantize_range_max_node);

        return Status::OK();
      },
      {}, &replaced_graph_def));

  // Convert bias float -> int32 on replaced_graph_def
  std::vector<string> fused_requantized_bias_ops = {
      "QuantizedConv2DWithBiasAndRequantize",
      "QuantizedConv2DWithBiasAndReluAndRequantize",
      "QuantizedConv2DWithBiasSumAndReluAndRequantize",
      "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"
      };
  node_map.clear();
  MapNamesToNodes(replaced_graph_def, &node_map);
  for (auto& node_pair : node_map) {
    const NodeDef *node = node_pair.second;
    bool is_fused_requantized_conv_op =
        std::find(fused_requantized_bias_ops.begin(),
                  fused_requantized_bias_ops.end(),
                  node->op()) != fused_requantized_bias_ops.end();
    if (is_fused_requantized_conv_op) {
      // If the op is not fed by Another Requantize op,
      // then we coonvert bias as Int32
      string input_op = node_map[NodeNameFromInput(node->input(0))]->op();
      if (str_util::StartsWith(input_op, "QuantizedConv2D") &&
          str_util::EndsWith(input_op, "AndRequantize")) {
        NodeDef *bias_node = const_cast<NodeDef*>(node_map[NodeNameFromInput(
            node->input(2))]);
        const NodeDef *min_input_node = node_map[NodeNameFromInput(
            node_map[node->input(0)]->input(7))];
        const NodeDef *max_input_node = node_map[NodeNameFromInput(
            node_map[node->input(0)]->input(8))];
        const NodeDef *min_filter_node = node_map[NodeNameFromInput(
            node->input(5))];
        const NodeDef *max_filter_node = node_map[NodeNameFromInput(
            node->input(6))];
        const float min_input =
            GetNodeTensorAttr(*min_input_node, "value").flat<float>()(0);
        const float max_input =
            GetNodeTensorAttr(*max_input_node, "value").flat<float>()(0);
        const float min_filter =
            GetNodeTensorAttr(*min_filter_node, "value").flat<float>()(0);
        const float max_filter =
            GetNodeTensorAttr(*max_filter_node, "value").flat<float>()(0);

        TensorProto float_tensor_proto = bias_node->attr().at("value").tensor();
        Tensor float_tensor;
        if(!float_tensor.FromProto(float_tensor_proto)) {
          TF_RETURN_IF_ERROR(::tensorflow::errors::InvalidArgument(
              "TensorProto object is not valid."));
        }
        if (float_tensor.dtype() != DT_FLOAT) {
          TF_RETURN_IF_ERROR(::tensorflow::errors::Unimplemented(
              "Expected float tensor."));
        }
        float *p_bias_float = float_tensor.flat<float>().data();

        Tensor int32_tensor = Tensor(DT_QINT32, float_tensor.shape());
        qint32 *p_bias_int32 = int32_tensor.flat<qint32>().data();

        float bias_scale = 255.0 * 127.0 /
            (std::max(std::abs(max_input), std::abs(min_input)) *
            std::max(std::abs(max_filter), std::abs(min_filter)));
        int64 nelems = float_tensor.NumElements();
        for (int64 n = 0; n < nelems; n++)
          p_bias_int32[n] = (int32_t) (p_bias_float[n] * bias_scale);

        bias_node->clear_attr();
        AttrValue attr_type;
        attr_type.set_type(int32_tensor.dtype());
        bias_node->mutable_attr()->insert({"dtype", attr_type});

        AttrValue attr_tensor;
        TensorProto* t = attr_tensor.mutable_tensor();
        int32_tensor.AsProtoTensorContent(t);
        bias_node->mutable_attr()->insert({"value", attr_tensor});
        SetNodeAttr("Tbias", DT_QINT32, const_cast<NodeDef*>(node));
      } else {
        SetNodeAttr("Tbias", DT_FLOAT, const_cast<NodeDef*>(node));
      }
    }
  }
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fuse_quantized_conv_and_requantize",
                         FuseQuantizedConvolutionAndRequantize);

}  // namespace graph_transforms
}  // namespace tensorflow
#endif // INTEL_MKL
