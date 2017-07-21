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
// A usage example, based on inception v3 model:
/*
bazel build tensorflow/tools/graph_transforms:transform_graph


// Specify remote graph by node names
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=/tmp/tensorflow_inception_v3_stripped_optimized_quantized.pb \
--out_graph=\
/tmp/tensorflow_inception_v3_stripped_optimized_quantized_fused_hexagon.pb \
--inputs='Mul' \
--outputs='softmax' \
--transforms='\
fuse_remote_graph(
input_types="float" \
input_shapes="1,299,299,3" \
fused_nodes="NodeA,NodeB,NodeC",
remote_fused_graph_executor_name="executor" \
remote_fused_graph_node_name="node_name" \
)'

// Specify remote graph by border inputs and outputs
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=/tmp/tensorflow_inception_v3_stripped_optimized_quantized.pb \
--out_graph=\
/tmp/tensorflow_inception_v3_stripped_optimized_quantized_fused_hexagon.pb \
--inputs='Mul' \
--outputs='softmax' \
--transforms='\
fuse_remote_graph(
input_types="float" \
input_shapes="1,299,299,3" \
border_inputs="NodeA:0,NodeB:0" \
border_outputs="NodeC" \
remote_fused_graph_executor_name="executor" \
remote_fused_graph_node_name="node_name" \
)'
*/

#include <unordered_set>

#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

static Status ParseArguments(const TransformFuncContext& context,
                             string* input_types_str, string* input_shapes_str,
                             string* fused_nodes_str, string* border_inputs_str,
                             string* border_outputs_str,
                             string* fused_op_types_str, bool* fuse_by_executor,
                             string* remote_fused_graph_node_name,
                             string* remote_graph_executor_name) {
  TF_RETURN_IF_ERROR(context.GetOneStringParameter(
      RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_INPUT_TYPES, "",
      input_types_str));
  TF_RETURN_IF_ERROR(context.GetOneStringParameter(
      RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_INPUT_SHAPES, "",
      input_shapes_str));
  TF_RETURN_IF_ERROR(context.GetOneStringParameter(
      RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_FUSED_NODES, "",
      fused_nodes_str));
  TF_RETURN_IF_ERROR(context.GetOneStringParameter(
      RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_BORDER_INPUTS, "",
      border_inputs_str));
  TF_RETURN_IF_ERROR(context.GetOneStringParameter(
      RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_BORDER_OUTPUTS, "",
      border_outputs_str));
  TF_RETURN_IF_ERROR(context.GetOneStringParameter(
      RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_FUSED_OP_TYPES, "",
      fused_op_types_str));
  TF_RETURN_IF_ERROR(context.GetOneBoolParameter(
      RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_FUSE_BY_EXECUTOR, false,
      fuse_by_executor));
  TF_RETURN_IF_ERROR(context.GetOneStringParameter(
      RemoteFusedGraphExecuteUtils::
          TRANSFORM_ARG_REMOTE_FUSED_GRAPH_EXECUTOR_NAME,
      "", remote_graph_executor_name));
  TF_RETURN_IF_ERROR(context.GetOneStringParameter(
      RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_REMOTE_FUSED_GRAPH_NODE_NAME,
      "", remote_fused_graph_node_name));

  CHECK(!remote_graph_executor_name->empty());
  return Status::OK();
}

static Status PlaceShapeType(const std::vector<string>& inputs,
                             const std::vector<string>& outputs,
                             const string& input_types_str,
                             const string& input_shapes_str,
                             GraphDef* mutable_input_graph_def) {
  const std::vector<string> input_types_strs =
      str_util::Split(input_types_str, ",");
  const std::vector<string> input_shapes_strs =
      str_util::Split(input_shapes_str, ":");
  CHECK_EQ(inputs.size(), input_types_strs.size());
  CHECK_EQ(inputs.size(), input_shapes_strs.size());
  std::vector<std::pair<string, Tensor>> input_tensors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const string& name = inputs.at(i);
    std::vector<int64> dims;
    CHECK(str_util::SplitAndParseAsInts(input_shapes_strs.at(i), ',', &dims));
    DataType data_type;
    CHECK(DataTypeFromString(input_types_strs.at(i), &data_type))
        << "\"" << input_types_strs.at(i) << "\" was an invalid type";
    input_tensors.emplace_back(
        std::make_pair(name, Tensor(data_type, TensorShape(dims))));
  }
  TF_RETURN_IF_ERROR(RemoteFusedGraphExecuteUtils::BuildAndAddTensorShapes(
      input_tensors, /*dry_run_inference=*/true, mutable_input_graph_def));
  return Status::OK();
}

Status FuseRemoteGraph(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def) {
  GraphDef mutable_input_graph_def = input_graph_def;

  const std::vector<string>& inputs = context.input_names;
  const std::vector<string>& outputs = context.output_names;

  string input_types_str;
  string input_shapes_str;
  string fused_nodes_str;
  string border_inputs_str;
  string border_outputs_str;
  string fused_op_types_str;
  bool fuse_by_executor = false;
  string remote_fused_graph_node_name;
  string remote_graph_executor_name;
  TF_RETURN_IF_ERROR(ParseArguments(
      context, &input_types_str, &input_shapes_str, &fused_nodes_str,
      &border_inputs_str, &border_outputs_str, &fused_op_types_str,
      &fuse_by_executor, &remote_fused_graph_node_name,
      &remote_graph_executor_name));

  if (!input_types_str.empty()) {
    TF_RETURN_IF_ERROR(PlaceShapeType(inputs, outputs, input_types_str,
                                      input_shapes_str,
                                      &mutable_input_graph_def));
  }

  const bool require_shape_type = !input_types_str.empty();
  if (!fused_nodes_str.empty()) {
    const std::vector<string> fused_node_name_vector =
        str_util::Split(fused_nodes_str, ",");
    const std::unordered_set<string> fused_node_names(
        fused_node_name_vector.begin(), fused_node_name_vector.end());
    TF_RETURN_IF_ERROR(RemoteFusedGraphExecuteUtils::FuseRemoteGraphByNodeNames(
        mutable_input_graph_def, inputs, outputs, remote_fused_graph_node_name,
        fused_node_names, remote_graph_executor_name, require_shape_type,
        output_graph_def));
  } else if (!border_inputs_str.empty() && !border_outputs_str.empty()) {
    const std::vector<string> border_inputs =
        str_util::Split(border_inputs_str, ",");
    const std::vector<string> border_outputs =
        str_util::Split(border_outputs_str, ",");
    for (size_t i = 0; i < border_inputs.size(); ++i) {
      VLOG(2) << "Border Input(" << i << "): " << border_inputs.at(i);
    }
    for (size_t i = 0; i < border_outputs.size(); ++i) {
      VLOG(2) << "Border Output(" << i << "): " << border_outputs.at(i);
    }
    TF_RETURN_IF_ERROR(RemoteFusedGraphExecuteUtils::FuseRemoteGraphByBorder(
        mutable_input_graph_def, inputs, outputs, remote_fused_graph_node_name,
        border_inputs, border_outputs, remote_graph_executor_name,
        require_shape_type, output_graph_def));
  } else if (!fused_op_types_str.empty()) {
    const std::vector<string> fused_op_type_vector =
        str_util::Split(fused_op_types_str, ",");
    const std::unordered_set<string> fused_op_types(
        fused_op_type_vector.begin(), fused_op_type_vector.end());
    TF_RETURN_IF_ERROR(RemoteFusedGraphExecuteUtils::FuseRemoteGraphByOpTypes(
        mutable_input_graph_def, inputs, outputs, remote_fused_graph_node_name,
        fused_op_types, remote_graph_executor_name, require_shape_type,
        output_graph_def));
  } else if (fuse_by_executor) {
    TF_RETURN_IF_ERROR(RemoteFusedGraphExecuteUtils::FuseRemoteGraphByExecutor(
        mutable_input_graph_def, inputs, outputs, remote_graph_executor_name,
        output_graph_def));
  } else {
    CHECK(false) << "Fuse targets are not specified.";
  }

  return Status::OK();
}

Status PlaceRemoteGraphArguments(const GraphDef& input_graph_def,
                                 const TransformFuncContext& context,
                                 GraphDef* output_graph_def) {
  *output_graph_def = input_graph_def;

  const std::vector<string>& inputs = context.input_names;
  const std::vector<string>& outputs = context.output_names;

  string input_types_str;
  string input_shapes_str;
  string fused_nodes_str;
  string border_inputs_str;
  string border_outputs_str;
  string fused_op_types_str;
  bool fuse_by_executor = false;
  string remote_fused_graph_node_name;
  string remote_graph_executor_name;
  TF_RETURN_IF_ERROR(ParseArguments(
      context, &input_types_str, &input_shapes_str, &fused_nodes_str,
      &border_inputs_str, &border_outputs_str, &fused_op_types_str,
      &fuse_by_executor, &remote_fused_graph_node_name,
      &remote_graph_executor_name));

  if (!input_types_str.empty()) {
    TF_RETURN_IF_ERROR(PlaceShapeType(inputs, outputs, input_types_str,
                                      input_shapes_str, output_graph_def));
  }

  const std::vector<string> fused_node_name_vector =
      str_util::Split(fused_nodes_str, ",");
  const std::unordered_set<string> fused_node_names(
      fused_node_name_vector.begin(), fused_node_name_vector.end());
  const std::vector<string> border_inputs =
      str_util::Split(border_inputs_str, ",");
  const std::vector<string> border_outputs =
      str_util::Split(border_outputs_str, ",");
  const std::vector<string> fused_op_type_vector =
      str_util::Split(fused_op_types_str, ",");
  const std::unordered_set<string> fused_op_types(fused_op_type_vector.begin(),
                                                  fused_op_type_vector.end());

  TF_RETURN_IF_ERROR(RemoteFusedGraphExecuteUtils::PlaceRemoteGraphArguments(
      inputs, outputs, fused_node_names, border_inputs, border_outputs,
      fused_op_types, remote_fused_graph_node_name, remote_graph_executor_name,
      output_graph_def));

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fuse_remote_graph", FuseRemoteGraph);

REGISTER_GRAPH_TRANSFORM("place_remote_graph_arguments",
                         PlaceRemoteGraphArguments);

}  // namespace graph_transforms
}  // namespace tensorflow
