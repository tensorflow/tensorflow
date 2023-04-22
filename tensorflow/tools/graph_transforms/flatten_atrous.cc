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

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status FlattenAtrousConv(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def) {
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"BatchToSpaceND",
          {
              {"Conv2D|DepthwiseConv2dNative",
                  {
                      {"SpaceToBatchND",
                          {
                              {"*"},          // Input to the flattened op.
                              {"*"},          // block_shape
                              {"*"}           // paddings
                          }
                      },
                      {"*"}                   // filter
                  }
              },
              {"*"},                          // block_shape
              {"*"}                           // crops
          }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& batch_to_space_node = match.node;
        const NodeDef& conv_node = match.inputs[0].node;
        const NodeDef& filter_node = match.inputs[0].inputs[1].node;
        const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].node;
        const NodeDef& space_to_batch_block_shape_node =
            match.inputs[0].inputs[0].inputs[1].node;

        // The atrous rate value is inferred from the block shape.
        Tensor block_shape =
            GetNodeTensorAttr(space_to_batch_block_shape_node, "value");
        const int32 block_height = block_shape.flat<int32>()(0);
        const int32 block_width = block_shape.flat<int32>()(1);

        // Compute the upsampled filter.
        const Tensor& filter = GetNodeTensorAttr(filter_node, "value");
        const int32 filter_height = filter.dim_size(0);
        const int32 filter_width = filter.dim_size(1);
        const int32 in_channels = filter.dim_size(2);
        const int32 out_channels = filter.dim_size(3);

        const int32 upsampled_filter_height =
            (filter_height - 1) * block_height + 1;
        const int32 upsampled_filter_width =
            (filter_width - 1) * block_width + 1;
        Tensor upsampled_filter(
            DT_FLOAT,
            TensorShape({upsampled_filter_height, upsampled_filter_width,
                         in_channels, out_channels}));

        auto filter_eigen = filter.tensor<float, 4>();
        auto upsampled_filter_eigen = upsampled_filter.tensor<float, 4>();

        upsampled_filter_eigen.setZero();
        for (int h = 0; h < filter_height; ++h) {
          for (int w = 0; w < filter_width; ++w) {
            for (int c_in = 0; c_in < in_channels; ++c_in) {
              for (int c_out = 0; c_out < out_channels; ++c_out) {
                upsampled_filter_eigen(block_height * h, block_width * w, c_in,
                                       c_out) = filter_eigen(h, w, c_in, c_out);
              }
            }
          }
        }

        NodeDef upsampled_filter_node;
        upsampled_filter_node.set_op("Const");
        upsampled_filter_node.set_name(filter_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &upsampled_filter_node);
        SetNodeTensorAttr<float>("value", upsampled_filter,
                                 &upsampled_filter_node);

        // Set up the new flattened version of the convolution op.
        NodeDef flattened_conv_node;

        flattened_conv_node.set_name(batch_to_space_node.name());
        flattened_conv_node.set_op(conv_node.op());
        flattened_conv_node.set_device(conv_node.device());

        AddNodeInput(input_node.name(), &flattened_conv_node);
        AddNodeInput(upsampled_filter_node.name(), &flattened_conv_node);

        CopyNodeAttr(conv_node, "T", "T", &flattened_conv_node);
        CopyNodeAttr(conv_node, "strides", "strides", &flattened_conv_node);
        SetNodeAttr("padding", "SAME", &flattened_conv_node);
        CopyNodeAttr(conv_node, "data_format", "data_format",
                     &flattened_conv_node);

        if (conv_node.op() == "Conv2D") {
          CopyNodeAttr(conv_node, "use_cudnn_on_gpu", "use_cudnn_on_gpu",
                       &flattened_conv_node);
        }

        new_nodes->push_back(input_node);
        new_nodes->push_back(upsampled_filter_node);
        new_nodes->push_back(flattened_conv_node);

        return Status::OK();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("flatten_atrous_conv", FlattenAtrousConv);

}  // namespace graph_transforms
}  // namespace tensorflow
