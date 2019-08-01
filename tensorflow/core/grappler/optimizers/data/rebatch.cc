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

#include "tensorflow/core/grappler/optimizers/data/rebatch.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {

Status RebatchOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  if (!config)
    return errors::InvalidArgument(
        "Cannot initialize RebatchOptimizer without config.");

  num_workers_ = config->parameter_map().at("num_workers").i();
  use_fallback_ = config->parameter_map().at("use_fallback").b();
  return Status::OK();
}

namespace {

constexpr char kAddOp[] = "Add";
constexpr char kConstOp[] = "Const";
constexpr char kIdentityOp[] = "Identity";
constexpr char kSubOp[] = "Sub";
constexpr char kTruncateDivOp[] = "TruncateDiv";

constexpr std::array<const char*, 6> kBatchDatasetOps = {
    "BatchDataset",
    "BatchDatasetV2",
    "ExperimentalMapAndBatchDataset",
    "MapAndBatchDataset",
    "PaddedBatchDataset",
    "PaddedBatchDatasetV2"};

constexpr std::array<const char*, 2> kMultipleInputsDatasetOps = {
    "ConcatenateDataset",
    "ZipDataset"
};

// TODO(rachelim): We might want to be more conservative here and not allow
// passthrough for ops like "Map", "ParallelMap" etc which may change the
// batch dimension. Furthermore, transformations like "Skip" may change
// the semantics of the dataset (since we'd be skipping N minibatches instead
// of N batches).
constexpr std::array<const char*, 21> kPassThroughOps = {
    "CacheDataset",
    "ExperimentalScanDataset",
    "ExperimentalParseExampleDataset",
    "FilterDataset",
    "Identity",
    "MapDataset",
    "ModelDataset",
    "OptimizeDataset",
    "ParallelMapDataset",
    "ParseExampleDataset",
    "PrefetchDataset",
    "ReduceDataset",
    "RepeatDataset",
    "ScanDataset",
    "ShardDataset",
    "ShuffleAndRepeatDataset",
    "ShuffleDataset",
    "ShuffleDatasetV2",
    "SkipDataset",
    "TakeDataset",
    "WindowDataset"};

constexpr std::array<const char*, 5> kFuncDatasetOps = {
    "ExperimentalGroupByWindowDataset",
    "FlatMapDataset",
    "GroupByWindowDataset",
    "InterleaveDataset",
    "ParallelInterleaveDatasetV2",
};

const std::map<string, const char*>* kFuncDatasetOpFuncs =
    new std::map<string, const char*>({
        {"ExperimentalGroupByWindowDataset", "reduce_func"},
        {"FlatMapDataset", "f"},
        {"GroupByWindowDataset", "reduce_func"},
        {"InterleaveDataset", "f"},
        {"ParallelInterleaveDatasetV2", "f"},
    });

constexpr std::array<const char*, 9> kSourceDatasetOps = {
    "FixedLengthRecordDataset",  "FixedLengthRecordDatasetV2",
    "GeneratorDataset",          "RangeDataset",
    "SparseTensorsSliceDataset", "TensorDataset",
    "TensorSliceDataset",        "TextLineDataset",
    "TFRecordDataset",
};

NodeDef* AddBinaryNode(const string& input_x, const string& input_y,
                       const string& op, DataType type,
                       MutableGraphView* graph) {
  NodeDef node;
  node.set_op(op);
  node.add_input(input_x);
  node.add_input(input_y);
  graph_utils::SetUniqueGraphNodeName(op, graph->graph(), &node);
  AddNodeAttr("T", type, &node);

  return graph->AddNode(std::move(node));
}

// Adds a Const node to the FunctionDef.
Status AddConstIntNode(gtl::ArraySlice<int32> values, const TensorShape& shape,
                       FunctionDef* fdef, NodeDef** result) {
  if (shape.dims() > 1) {
    return errors::InvalidArgument("Cannot add const node with rank > 1");
  }
  *result = fdef->add_node_def();
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  if (shape.dims() == 0) {
    // Scalar
    DCHECK_EQ(values.size(), 1);
  } else {
    // vector
    DCHECK_EQ(values.size(), shape.dim_size(0));
    tensor_proto.mutable_tensor_shape()->add_dim()->set_size(shape.dim_size(0));
  }

  for (int value : values) {
    *tensor_proto.mutable_int_val()->Add() = value;
  }

  TF_RETURN_IF_ERROR(NodeDefBuilder("", "Const")
                         .Attr("dtype", DT_INT32)
                         .Attr("value", tensor_proto)
                         .Finalize(*result));
  function_utils::SetUniqueFunctionNodeName("rebatch/const", fdef, *result);

  return Status::OK();
}

Status AddShapeNode(const NodeDefBuilder::NodeOut& input, FunctionDef* fdef,
                    NodeDef** result) {
  *result = fdef->add_node_def();
  TF_RETURN_IF_ERROR(
      NodeDefBuilder("", "Shape").Input(input).Finalize(*result));
  function_utils::SetUniqueFunctionNodeName("rebatch/shape", fdef, *result);
  return Status::OK();
}

Status AddStridedSliceNode(const NodeDefBuilder::NodeOut& input,
                           const NodeDefBuilder::NodeOut& begin,
                           const NodeDefBuilder::NodeOut& end,
                           const NodeDefBuilder::NodeOut& strides,
                           DataType index, int32 begin_mask,
                           int32 ellipsis_mask, int32 end_mask,
                           int32 new_axis_mask, int32 shrink_axis_mask,
                           FunctionDef* fdef, NodeDef** result) {
  *result = fdef->add_node_def();
  TF_RETURN_IF_ERROR(NodeDefBuilder("", "StridedSlice")
                         .Input(input)
                         .Input(begin)
                         .Input(end)
                         .Input(strides)
                         .Attr("Index", index)
                         .Attr("begin_mask", begin_mask)
                         .Attr("ellipsis_mask", ellipsis_mask)
                         .Attr("end_mask", end_mask)
                         .Attr("new_axis_mask", new_axis_mask)
                         .Attr("shrink_axis_mask", shrink_axis_mask)
                         .Finalize(*result));
  function_utils::SetUniqueFunctionNodeName("rebatch/strided_slice", fdef,
                                            *result);
  return Status::OK();
}

Status AddConcatNode(gtl::ArraySlice<NodeDefBuilder::NodeOut> values,
                     NodeDefBuilder::NodeOut axis, int32 n, FunctionDef* fdef,
                     NodeDef** result) {
  *result = fdef->add_node_def();
  TF_RETURN_IF_ERROR(NodeDefBuilder("", "ConcatV2")
                         .Input(values)
                         .Input(axis)
                         .Attr("N", n)
                         .Finalize(*result));
  function_utils::SetUniqueFunctionNodeName("rebatch/concat", fdef, *result);
  return Status::OK();
}

Status AddReshapeNode(NodeDefBuilder::NodeOut tensor,
                      NodeDefBuilder::NodeOut shape, FunctionDef* fdef,
                      NodeDef** result) {
  *result = fdef->add_node_def();
  TF_RETURN_IF_ERROR(NodeDefBuilder("", "Reshape")
                         .Input(tensor)
                         .Input(shape)
                         .Finalize(*result));
  function_utils::SetUniqueFunctionNodeName("rebatch/reshape", fdef, *result);
  return Status::OK();
}

template <std::size_t SIZE>
bool IsDatasetNodeOfType(const NodeDef& node,
                         const std::array<const char*, SIZE>& arr) {
  for (const auto& dataset_op_name : arr) {
    if (node.op() == dataset_op_name) return true;
  }
  return false;
}

void SetUnknownShapes(int num_components, AttrValue* output_shapes) {
  for (int i = 0; i < num_components; ++i) {
    output_shapes->mutable_list()->mutable_shape()->Add()->set_unknown_rank(
        true);
  }
}

Status GetBatchDim(AttrValue output_shapes, int* batch_dim) {
  const auto& shape_0 = output_shapes.list().shape(0);
  if (shape_0.unknown_rank() || shape_0.dim(0).size() == -1) {
    return errors::InvalidArgument(
        "Cannot use rebatching fallback when 0th dimensions of dataset "
        "components are not fully known. Component 0 has shape: ",
        shape_0.ShortDebugString());
  }

  *batch_dim = output_shapes.list().shape(0).dim(0).size();

  for (int i = 1; i < output_shapes.list().shape_size(); ++i) {
    const auto& shape_i = output_shapes.list().shape(i);

    if (shape_i.unknown_rank() || shape_i.dim(0).size() == -1) {
      return errors::InvalidArgument(
          "Cannot use rebatching fallback when 0th dimensions of dataset "
          "components are not fully known. Component ",
          i, " has shape: ", shape_i.ShortDebugString());
    }
    if (shape_i.dim(0).size() != *batch_dim) {
      return errors::InvalidArgument(
          "Cannot use rebatching fallback when 0th dimensions of dataset "
          "components don't match. Component ",
          i, " has batch dimension: ", shape_i.dim(0).size(),
          " while previous components have batch dimension: ", *batch_dim);
    }
  }
  return Status::OK();
}

Status UpdateOutputShapes(const string& node_name, int64 num_workers,
                          MutableGraphView* graph) {
  NodeDef* node = graph->GetNode(node_name);
  if (node->attr().contains("output_shapes")) {
    AttrValue output_shapes = node->attr().at("output_shapes");
    for (auto& shape : *output_shapes.mutable_list()->mutable_shape()) {
      if (!shape.unknown_rank() && shape.dim(0).size() != -1) {
        shape.mutable_dim(0)->set_size(shape.dim(0).size() / num_workers);
      }
    }
    (*node->mutable_attr())["output_shapes"] = output_shapes;
  }
  return Status::OK();
}

// Given a "batch" dataset node, we replace the `batch_size` input with a new
// input that corresponds to the original input divided by `num_workers`. If
// `num_workers` does not divide `batch_size` evenly, the value is rounded up.
Status MutateBatchSize(const NodeDef& node, int64 num_workers,
                       MutableGraphView* graph) {
  // For all the batching datasets the batch_size is input number 1 except for
  // MapAndBatchDataset.
  int64 batch_size_arg_index = 1;
  if (node.op() == "ExperimentalMapAndBatchDataset" ||
      node.op() == "MapAndBatchDataset") {
    // For MapAndBatch we take the 3rd last input.
    batch_size_arg_index = node.input_size() - 3;
  }
  NodeDef* batch_size_node =
      graph_utils::GetInputNode(node, *graph, batch_size_arg_index);
  NodeDef* new_batch_size_node;
  if (batch_size_node->op() == kConstOp) {
    Tensor batch_size_tensor;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(*batch_size_node, "value", &batch_size_tensor));
    if (!TensorShapeUtils::IsScalar(batch_size_tensor.shape())) {
      return errors::Internal("Batch size node shape should be scalar");
    }
    int64 batch_size = batch_size_tensor.scalar<int64>()();
    batch_size = (batch_size + num_workers - 1) / num_workers;
    new_batch_size_node =
        graph_utils::AddScalarConstNode<int64>(batch_size, graph);
  } else {
    NodeDef* one_node = graph_utils::AddScalarConstNode<int64>(1, graph);
    NodeDef* num_workers_node =
        graph_utils::AddScalarConstNode<int64>(num_workers, graph);
    NodeDef* numerator_node =
        AddBinaryNode(batch_size_node->name(), num_workers_node->name(), kAddOp,
                      DT_INT64, graph);
    numerator_node = AddBinaryNode(numerator_node->name(), one_node->name(),
                                   kSubOp, DT_INT64, graph);
    new_batch_size_node =
        AddBinaryNode(numerator_node->name(), num_workers_node->name(),
                      kTruncateDivOp, DT_INT64, graph);
  }
  // We don't call UpdateFanouts here because CSE elimination might lead to
  // multiple nodes sharing the same batch size constant node. This is also
  // why we don't delete batch_size_node as well.
  TF_RETURN_IF_ERROR(graph->UpdateRegularFaninByPort(
      node.name(), batch_size_arg_index, {new_batch_size_node->name(), 0}));
  return Status::OK();
}

Status OptimizeGraph(const GrapplerItem& item, int64 num_workers,
                     bool use_fallback, GraphDef* output);

// Helper function that starts from a node in the graph and recurses into its
// inputs trying to find a BatchDataset type operation to modify. During the
// recursion it handles four kinds of cases.
// 1. BatchDataset type ops: Mutates the batch_size input node and stops.
// 2. Zip / Concatenate dataset ops: Recurses into all inputs to these ops
//      as they are datasets themselves.
// 3. Core dataset ops + Identity op: Recurses into first input parameter.
// 4. FlatMap type mapping dataset ops: Recurses into the function definition.
Status RecursivelyHandleOp(const NodeDef& node, int64 num_workers,
                           bool use_fallback, FunctionLibraryDefinition* flib,
                           MutableGraphView* graph) {
  if (IsDatasetNodeOfType(node, kBatchDatasetOps)) {
    TF_RETURN_IF_ERROR(MutateBatchSize(node, num_workers, graph));
  } else if (IsDatasetNodeOfType(node, kMultipleInputsDatasetOps)) {
    // For all multiple input datasets, all inputs are datasets themselves.
    for (int i = 0; i < node.input_size(); ++i) {
      NodeDef* input_node = graph_utils::GetInputNode(node, *graph, i);
      TF_RETURN_IF_ERROR(RecursivelyHandleOp(*input_node, num_workers,
                                             use_fallback, flib, graph));
    }
  } else if (IsDatasetNodeOfType(node, kPassThroughOps) || IsRetval(node)) {
    // For all the dataset ops that are passthrough, or _Retvals added to the
    // function body graph in place of function outputs, the input dataset is
    // input 0.
    NodeDef* input_node = graph_utils::GetInputNode(node, *graph, 0);
    TF_RETURN_IF_ERROR(RecursivelyHandleOp(*input_node, num_workers,
                                           use_fallback, flib, graph));
  } else if (IsDatasetNodeOfType(node, kFuncDatasetOps)) {
    const string func_name =
        node.attr().at(kFuncDatasetOpFuncs->at(node.op())).func().name();
    const FunctionDef* fdef = flib->Find(func_name);
    GrapplerFunctionItem f_item;
    TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(
        *fdef, *flib, graph->graph()->versions().producer(), &f_item));
    GraphDef optimized_func_graph;
    TF_RETURN_IF_ERROR(OptimizeGraph(f_item, num_workers, use_fallback,
                                     &optimized_func_graph));

    // Function body optimization might have created new specialized
    // functions for each instantiation context. Add them to the library.
    for (const FunctionDef& func_def :
         optimized_func_graph.library().function()) {
      if (flib->Find(func_def.signature().name()) == nullptr) {
        TF_RETURN_IF_ERROR(flib->AddFunctionDef(func_def));
      }
    }

    // Convert optimized graph back to FunctionDef.
    FunctionDef optimized_func;
    f_item.SwapFunctionBody(std::move(optimized_func_graph));
    TF_RETURN_IF_ERROR(MakeFunctionDef(f_item, *flib, &optimized_func));

    // Replace optimized function with a new FunctionDef.
    TF_RETURN_IF_ERROR(flib->ReplaceFunction(func_name, optimized_func));
  } else if (IsDatasetNodeOfType(node, kSourceDatasetOps)) {
    return errors::InvalidArgument(
        "Reached a source dataset: ", node.op(),
        " without encountering a batch transformation.");
  } else {
    return errors::InvalidArgument("Encountered an unsupported op: ",
                                   node.op());
  }
  // If we've successfully updated the batch size of this node or any nodes
  // in the dataset tree rooted in this node, we update the output_shapes attr.
  TF_RETURN_IF_ERROR(UpdateOutputShapes(node.name(), num_workers, graph));
  return Status::OK();
}

// Add nodes to the function to reshape arg to shape (-1, new_batch_dim, ...)
Status ReshapeComponent(int new_batch_dim, StringPiece arg, DataType dtype,
                        FunctionDef* fdef, string* result) {
  // Const with value [0]
  NodeDef* const_vec_0;
  TF_RETURN_IF_ERROR(AddConstIntNode({0}, {1}, fdef, &const_vec_0));

  // Const with value [1]
  NodeDef* const_vec_1;
  TF_RETURN_IF_ERROR(AddConstIntNode({1}, {1}, fdef, &const_vec_1));

  // Const with value 0
  NodeDef* const_0;
  TF_RETURN_IF_ERROR(AddConstIntNode({0}, {}, fdef, &const_0));

  // Const with value [-1, new_batch_dim]
  NodeDef* first_two_dims;
  TF_RETURN_IF_ERROR(
      AddConstIntNode({-1, new_batch_dim}, {2}, fdef, &first_two_dims));

  // shape = tf.shape(arg)
  NodeDef* shape;
  TF_RETURN_IF_ERROR(AddShapeNode({arg, 0, dtype}, fdef, &shape));

  // later_dimensions = tf.shape(arg)[1:]
  NodeDef* later_dimensions;
  TF_RETURN_IF_ERROR(AddStridedSliceNode(
      {strings::StrCat(shape->name(), ":output"), 0, DT_INT32},
      {strings::StrCat(const_vec_1->name(), ":output"), 0, DT_INT32},
      {strings::StrCat(const_vec_0->name(), ":output"), 0, DT_INT32},
      {strings::StrCat(const_vec_1->name(), ":output"), 0, DT_INT32}, DT_INT32,
      0, 0, 1, 0, 0, fdef, &later_dimensions));

  // new_shape = tf.concat([pack, later_dimensions], 0)
  NodeDef* new_shape;
  TF_RETURN_IF_ERROR(AddConcatNode(
      {{strings::StrCat(first_two_dims->name(), ":output"), 0, DT_INT32},
       {strings::StrCat(later_dimensions->name(), ":output"), 0, DT_INT32}},
      {strings::StrCat(const_0->name(), ":output"), 0, DT_INT32}, 2, fdef,
      &new_shape));

  NodeDef* reshape;
  TF_RETURN_IF_ERROR(AddReshapeNode(
      {arg, 0, dtype},
      {strings::StrCat(new_shape->name(), ":output"), 0, DT_INT32}, fdef,
      &reshape));
  *result = reshape->name();

  return Status::OK();
}

Status CreateFlatMapFn(int new_batch_dim, const AttrValue& types,
                       FunctionDef* result) {
  std::vector<NodeDefBuilder::NodeOut> tensor_slice_dataset_inputs;

  // For each component of the dataset, we reshape it from shape
  // (old_batch_size, ...) to (-1, new_batch_size, ...)
  // where new_batch_size = (old_batch_size + num_workers - 1) // num_workers
  for (int i = 0; i < types.list().type_size(); ++i) {
    string arg = strings::StrCat("args_", i);
    auto* input_arg = result->mutable_signature()->mutable_input_arg()->Add();
    input_arg->set_type(types.list().type(i));
    input_arg->set_name(arg);

    string reshape_node_name;
    TF_RETURN_IF_ERROR(ReshapeComponent(
        new_batch_dim, arg, types.list().type(i), result, &reshape_node_name));

    tensor_slice_dataset_inputs.emplace_back(
        strings::StrCat(reshape_node_name, ":output"), 0, types.list().type(i));
  }

  // The output_shapes attr here doesn't make a difference, since we
  // set the output_shapes of the external FlatMap node.
  AttrValue shapes;
  SetUnknownShapes(types.list().type_size(), &shapes);

  NodeDef* tensor_slice_dataset = result->add_node_def();
  TF_RETURN_IF_ERROR(NodeDefBuilder("", "TensorSliceDataset")
                         .Input(tensor_slice_dataset_inputs)
                         .Attr("Toutput_types", types)
                         .Attr("output_shapes", shapes)
                         .Finalize(tensor_slice_dataset));
  function_utils::SetUniqueFunctionNodeName("rebatch/tensor_slice_dataset",
                                            result, tensor_slice_dataset);

  auto* output_arg = result->mutable_signature()->mutable_output_arg()->Add();
  output_arg->set_name("output");
  output_arg->set_type(DT_VARIANT);
  result->mutable_signature()->set_is_stateful(true);
  (*result->mutable_ret())["output"] =
      strings::StrCat(tensor_slice_dataset->name(), ":handle:0");

  return Status::OK();
}

// We fallback to the following rewrite:
// ```
//   dataset = ...fetch_node...
//   def fn(x):
//     return tf.data.Dataset.from_tensor_slices(
//       tf.reshape(
//         x,
//         tf.concat([[-1, old_batch_dim / num_workers], tf.shape(x)[1:]], 0)
//       )
//     )
//
//   dataset = dataset.flat_map(fn)
// ```
Status RebatchWithFallback(const NodeDef* fetch_node, int64 num_workers,
                           FunctionLibraryDefinition* flib,
                           MutableGraphView* graph) {
  if (IsRetval(*fetch_node) || fetch_node->op() == kIdentityOp) {
    // Get the last dataset in the pipeline
    fetch_node = graph_utils::GetInputNode(*fetch_node, *graph, 0);
  }

  // Note: Here, we are conservative with only using the fallback when
  // the output_shapes attr has the 0th dimension defined for every component.
  // This because the flat_map_fn will fail if the batch does not divide evenly
  // because of the use of the "Reshape" op. This ensures that the error is
  // surfaced correctly.
  AttrValue output_shapes;
  if (!fetch_node->attr().contains("output_shapes")) {
    return errors::InvalidArgument(
        "Cannot use rebatching fallback without output_shapes attr. Node: ",
        fetch_node->name(), " Op: ", fetch_node->op());
  } else {
    output_shapes = fetch_node->attr().at("output_shapes");
  }
  int batch_dim;
  TF_RETURN_IF_ERROR(GetBatchDim(output_shapes, &batch_dim));
  if (batch_dim % num_workers != 0) {
    return errors::InvalidArgument(
        "Cannot use rebatching fallback when batch dimension doesn't divide "
        "num_workers evenly.");
  }

  // Create the flat map fn
  FunctionDef flat_map_fn;
  FunctionDefLibrary lib = flib->ToProto();
  graph_utils::SetUniqueGraphFunctionName("flat_map_fn", &lib, &flat_map_fn);

  // Get types of input arguments from the output types of the final dataset.
  AttrValue output_types;
  TF_RETURN_IF_ERROR(
      graph_utils::GetDatasetOutputTypesAttr(*fetch_node, &output_types));
  TF_RETURN_IF_ERROR(
      CreateFlatMapFn(batch_dim / num_workers, output_types, &flat_map_fn));

  TF_RETURN_IF_ERROR(flib->AddFunctionDef(flat_map_fn));
  AttrValue fn;
  fn.mutable_func()->set_name(flat_map_fn.signature().name());

  NodeDef flat_map_node;
  TF_RETURN_IF_ERROR(
      NodeDefBuilder("", "FlatMapDataset")
          .Input(fetch_node->name(), 0, DT_VARIANT)
          .Input(std::vector<NodeDefBuilder::NodeOut>())  // other_arguments
          .Attr("f", fn)
          .Attr("Targuments", std::vector<DataType>())
          .Attr("output_types", output_types)
          .Attr("output_shapes", output_shapes)
          .Finalize(&flat_map_node));
  graph_utils::SetUniqueGraphNodeName("rebatch/flat_map", graph->graph(),
                                      &flat_map_node);
  NodeDef* added = graph->AddNode(std::move(flat_map_node));
  TF_RETURN_IF_ERROR(UpdateOutputShapes(added->name(), num_workers, graph));

  TF_RETURN_IF_ERROR(graph->UpdateFanouts(fetch_node->name(), added->name()));

  return Status::OK();
}

// Helper function that given a GrapplerItem generates a mutated graph def
// with the batch size changed. The GrapplerItem could be generated from the
// main graph or could be a function graph.
Status OptimizeGraph(const GrapplerItem& item, int64 num_workers,
                     bool use_fallback, GraphDef* output) {
  *output = item.graph;
  MutableGraphView graph(output);

  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());

  NodeDef* sink_node;
  TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));

  Status s =
      RecursivelyHandleOp(*sink_node, num_workers, use_fallback, &flib, &graph);
  if (!s.ok()) {
    if (use_fallback) {
      VLOG(1) << "Couldn't find a batch transformation. Using a fallback method"
                 " to rebatch dataset.";
      // If RecursivelyHandleOp fails, we reset `graph` to use the original,
      // graph, since that function may have mutated `graph`.
      *output = item.graph;
      graph = MutableGraphView(output);
      TF_RETURN_IF_ERROR(
          RebatchWithFallback(sink_node, num_workers, &flib, &graph));
    } else {
      // Return the error
      return s;
    }
  }
  *output->mutable_library() = flib.ToProto();
  return Status::OK();
}

}  // anonymous namespace

Status RebatchOptimizer::OptimizeAndCollectStats(Cluster* cluster,
                                                 const GrapplerItem& item,
                                                 GraphDef* output,
                                                 OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);

  TF_RETURN_IF_ERROR(OptimizeGraph(item, num_workers_, use_fallback_, output));
  stats->num_changes++;
  return Status::OK();
}

void RebatchOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                const GraphDef& optimize_output,
                                double result) {}

REGISTER_GRAPH_OPTIMIZER_AS(RebatchOptimizer, "tf_data_rebatcher");

}  // namespace grappler
}  // namespace tensorflow
