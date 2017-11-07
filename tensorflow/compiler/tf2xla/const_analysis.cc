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

#include "tensorflow/compiler/tf2xla/const_analysis.h"

#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

// Backwards dataflow analysis that finds arguments to a graph that must be
// compile-time constants.
Status BackwardsConstAnalysis(const Graph& g,
                              std::vector<bool>* compile_time_const_args) {
  // TODO(phawkins): annotate these on the kernel registrations, rather than
  // using a hard-coded list.
  // (operator, argument) pairs that must be compile-time constants.
  const std::unordered_multimap<string, string> compile_time_const_inputs = {
      {"All", "reduction_indices"},
      {"Any", "reduction_indices"},
      {"ArgMin", "dimension"},
      {"ArgMax", "dimension"},
      {"AvgPoolGrad", "orig_input_shape"},
      {"AvgPool3DGrad", "orig_input_shape"},
      {"BatchToSpace", "crops"},
      {"BatchToSpaceND", "block_shape"},
      {"BatchToSpaceND", "crops"},
      {"BroadcastArgs", "s0"},
      {"BroadcastArgs", "s1"},
      {"BroadcastGradientArgs", "s0"},
      {"BroadcastGradientArgs", "s1"},
      {"Concat", "concat_dim"},
      {"ConcatV2", "axis"},
      {"ConcatOffset", "concat_dim"},
      {"ConcatOffset", "shape"},
      {"Conv2DBackpropFilter", "filter_sizes"},
      {"Conv2DBackpropInput", "input_sizes"},
      {"Conv3DBackpropFilterV2", "filter_sizes"},
      {"Conv3DBackpropInputV2", "input_sizes"},
      {"DepthwiseConv2dNativeBackpropFilter", "filter_sizes"},
      {"DepthwiseConv2dNativeBackpropInput", "input_sizes"},
      {"DynamicStitch", "indices"},
      {"ExpandDims", "dim"},
      {"Fill", "dims"},
      {"GatherV2", "axis"},
      {"InvertPermutation", "x"},
      {"LinSpace", "start"},
      {"LinSpace", "stop"},
      {"LinSpace", "num"},
      {"Max", "reduction_indices"},
      {"Mean", "reduction_indices"},
      {"Min", "reduction_indices"},
      {"OneHot", "depth"},
      {"Pad", "paddings"},
      {"PadV2", "paddings"},
      {"MirrorPad", "paddings"},
      {"Multinomial", "num_samples"},
      {"Prod", "reduction_indices"},
      {"RandomStandardNormal", "shape"},
      {"RandomUniform", "shape"},
      {"RandomUniformInt", "shape"},
      {"Range", "start"},
      {"Range", "limit"},
      {"Range", "delta"},
      {"Reshape", "shape"},
      {"ResourceStridedSliceAssign", "begin"},
      {"ResourceStridedSliceAssign", "end"},
      {"ResourceStridedSliceAssign", "strides"},
      {"Reverse", "dims"},
      {"ReverseV2", "axis"},
      {"Slice", "begin"},
      {"Slice", "size"},
      {"SpaceToBatch", "paddings"},
      {"SpaceToBatchND", "block_shape"},
      {"SpaceToBatchND", "paddings"},
      {"Split", "split_dim"},
      {"SplitV", "split_dim"},
      {"SplitV", "size_splits"},
      {"StackV2", "max_size"},
      {"StridedSlice", "begin"},
      {"StridedSlice", "end"},
      {"StridedSlice", "strides"},
      {"StridedSliceGrad", "shape"},
      {"StridedSliceGrad", "begin"},
      {"StridedSliceGrad", "end"},
      {"StridedSliceGrad", "strides"},
      {"Sum", "reduction_indices"},
      {"TensorArrayV3", "size"},
      {"TensorArraySplitV3", "lengths"},
      {"Tile", "multiples"},
      {"Transpose", "perm"}};

  // Operators that don't look at the data of their inputs, just the shapes.
  const std::unordered_set<string> metadata_ops = {
      "Rank", "Shape", "ShapeN", "Size",
  };

  Status status;
  std::unordered_set<Node*> must_be_const;
  auto visit = [&status, &metadata_ops, &compile_time_const_inputs,
                &must_be_const, compile_time_const_args](Node* node) {
    if (!status.ok()) return;

    // If this is a metadata-only op, don't propagate the const requirement.
    if (metadata_ops.find(node->type_string()) != metadata_ops.end()) return;

    // If this node must be const, and it isn't a metadata op, then all of its
    // parents must be const.
    if (must_be_const.find(node) != must_be_const.end()) {
      if (node->type_string() == "_Arg") {
        int index;
        status = GetNodeAttr(node->attrs(), "index", &index);
        if (!status.ok()) return;
        compile_time_const_args->at(index) = true;
        return;
      }
      for (Node* pred : node->in_nodes()) {
        must_be_const.insert(pred);
      }
      return;
    }

    // Mark any compile-time constant operator arguments as const.
    auto range = compile_time_const_inputs.equal_range(node->type_string());
    if (range.first == range.second) return;

    NameRangeMap input_name_ranges;
    status =
        NameRangesForNode(*node, node->op_def(), &input_name_ranges, nullptr);
    if (!status.ok()) return;

    for (auto it = range.first; it != range.second; ++it) {
      auto name_range = input_name_ranges.find(it->second);
      if (name_range == input_name_ranges.end()) continue;

      for (Edge const* edge : node->in_edges()) {
        if (edge->dst_input() >= name_range->second.first &&
            edge->dst_input() < name_range->second.second) {
          must_be_const.insert(edge->src());
        }
      }
    }
  };

  // Post-order traversal visits nodes in reverse topological order for an
  // acyclic graph.
  DFS(g, {}, visit);
  return status;
}

}  // namespace tensorflow
