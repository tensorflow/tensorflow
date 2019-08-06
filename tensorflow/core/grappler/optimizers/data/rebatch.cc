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
#include "tensorflow/core/util/padding.h"

namespace tensorflow {
namespace grappler {

Status RebatchOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  if (!config)
    return errors::InvalidArgument(
        "Cannot initialize RebatchOptimizer without config.");

  num_replicas_ = config->parameter_map().at("num_replicas").i();
  use_fallback_ = config->parameter_map().at("use_fallback").b();
  return Status::OK();
}

namespace {

constexpr char kAddOp[] = "Add";
constexpr char kConstOp[] = "Const";
constexpr char kIdentityOp[] = "Identity";
constexpr char kSubOp[] = "Sub";
constexpr char kTruncateDivOp[] = "TruncateDiv";
constexpr char kOutputShapesAttr[] = "output_shapes";
constexpr char kOutputTypesAttr[] = "output_types";
constexpr char kTOutputTypesAttr[] = "Toutput_types";
constexpr char kBatchOp[] = "BatchDataset";
constexpr char kBatchV2Op[] = "BatchDatasetV2";
constexpr char kPaddedBatchOp[] = "PaddedBatchDataset";
constexpr char kPaddedBatchV2Op[] = "PaddedBatchDatasetV2";
constexpr char kMapAndBatchOp[] = "MapAndBatchDataset";
constexpr char kExperimentalMapAndBatchOp[] = "ExperimentalMapAndBatchDataset";

constexpr std::array<const char*, 6> kBatchDatasetOps = {
    kBatchOp,       kBatchV2Op,      kMapAndBatchOp, kExperimentalMapAndBatchOp,
    kPaddedBatchOp, kPaddedBatchV2Op};

constexpr std::array<const char*, 2> kMultipleInputsDatasetOps = {
    "ConcatenateDataset",
    "ZipDataset"
};

// TODO(rachelim): We might want to be more conservative here and not allow
// passthrough for ops like "Map", "ParallelMap" etc which may change the
// batch dimension. Furthermore, transformations like "Skip" may change
// the semantics of the dataset (since we'd be skipping N minibatches instead
// of N batches).
constexpr std::array<const char*, 22> kPassThroughOps = {
    "CacheDataset",
    "CacheDatasetV2",
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
    "WindowDataset",
};

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

NodeDef MakeBinaryNode(const string& input_x, const string& input_y,
                       const string& op, DataType dtype) {
  NodeDef node;
  node.set_op(op);
  node.add_input(input_x);
  node.add_input(input_y);
  AddNodeAttr("T", dtype, &node);

  return node;
}

NodeDef* AddBinaryNode(const string& input_x, const string& input_y,
                       const string& op, DataType type, FunctionDef* fdef) {
  NodeDef* node = fdef->add_node_def();
  *node = MakeBinaryNode(input_x, input_y, op, type);
  function_utils::SetUniqueFunctionNodeName(op, fdef, node);

  return node;
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

Status AddConstInt64Node(int64 value, FunctionDef* fdef, NodeDef** result) {
  *result = fdef->add_node_def();
  Tensor t(value);
  TF_RETURN_IF_ERROR(NodeDefBuilder("", "Const")
                         .Attr("dtype", DT_INT64)
                         .Attr("value", t)
                         .Finalize(*result));
  function_utils::SetUniqueFunctionNodeName("rebatch/const", fdef, *result);

  return Status::OK();
}

Status AddConstBoolNode(bool value, FunctionDef* fdef, NodeDef** result) {
  *result = fdef->add_node_def();
  Tensor t(value);
  TF_RETURN_IF_ERROR(NodeDefBuilder("", "Const")
                         .Attr("dtype", DT_BOOL)
                         .Attr("value", t)
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

Status UpdateOutputShapes(const string& node_name, int64 num_replicas,
                          MutableGraphView* graph) {
  NodeDef* node = graph->GetNode(node_name);
  if (node->attr().contains(kOutputShapesAttr)) {
    AttrValue output_shapes = node->attr().at(kOutputShapesAttr);
    for (auto& shape : *output_shapes.mutable_list()->mutable_shape()) {
      if (!shape.unknown_rank() && shape.dim(0).size() != -1) {
        shape.mutable_dim(0)->set_size(shape.dim(0).size() / num_replicas);
      }
    }
    (*node->mutable_attr())[kOutputShapesAttr] = output_shapes;
  }
  return Status::OK();
}

// Helper function to get the batch_size input node for a give batch node.
int64 GetBatchSizeArgIndex(const NodeDef& batch_node) {
  if (batch_node.op() == kExperimentalMapAndBatchOp ||
      batch_node.op() == kMapAndBatchOp) {
    // For MapAndBatch we take the 3rd last input.
    return batch_node.input_size() - 3;
  }
  // For all the batching datasets the batch_size is input number 1 except for
  // MapAndBatchDataset.
  return 1;
}

Status MakeNewBatchSizeNode(const string& global_batch_size_name,
                            int64 num_replicas, FunctionDef* fdef,
                            NodeDef** result) {
  NodeDef* one_node;
  TF_RETURN_IF_ERROR(AddConstInt64Node(1, fdef, &one_node));
  NodeDef* num_replicas_node;
  TF_RETURN_IF_ERROR(AddConstInt64Node(num_replicas, fdef, &num_replicas_node));

  NodeDef* numerator_node =
      AddBinaryNode(global_batch_size_name,
                    strings::StrCat(num_replicas_node->name(), ":output:0"),
                    kAddOp, DT_INT64, fdef);
  numerator_node = AddBinaryNode(
      strings::StrCat(numerator_node->name(), ":z:0"),
      strings::StrCat(one_node->name(), ":output:0"), kSubOp, DT_INT64, fdef);

  *result =
      AddBinaryNode(strings::StrCat(numerator_node->name(), ":z:0"),
                    strings::StrCat(num_replicas_node->name(), ":output:0"),
                    kTruncateDivOp, DT_INT64, fdef);
  return Status::OK();
}

// Given a "batch" dataset node, we replace the `batch_size` input with a new
// input that corresponds to the original input divided by `num_replicas`.
Status MutateBatchSize(const NodeDef& node, int64 num_replicas,
                       MutableGraphView* graph) {
  // For all the batching datasets the batch_size is input number 1 except for
  // MapAndBatchDataset.
  int64 batch_size_arg_index = GetBatchSizeArgIndex(node);
  NodeDef* batch_size_node =
      graph_utils::GetInputNode(node, *graph, batch_size_arg_index);
  int64 batch_size;
  TF_RETURN_IF_ERROR(
      graph_utils::GetScalarConstNodeValue(*batch_size_node, &batch_size));
  DCHECK_EQ(batch_size % num_replicas, 0);
  batch_size = batch_size / num_replicas;
  NodeDef* new_batch_size_node =
      graph_utils::AddScalarConstNode<int64>(batch_size, graph);
  // We don't call UpdateFanouts here because CSE elimination might lead to
  // multiple nodes sharing the same batch size constant node. This is also
  // why we don't delete batch_size_node as well.
  TF_RETURN_IF_ERROR(graph->UpdateRegularFaninByPort(
      node.name(), batch_size_arg_index, {new_batch_size_node->name(), 0}));
  return Status::OK();
}

Status AddFlatMapNode(const string& input_dataset,
                      gtl::ArraySlice<string> other_arguments,
                      gtl::ArraySlice<DataType> t_arguments,
                      const FunctionDef& flat_map_fn,
                      const AttrValue& output_shapes,
                      const DataTypeVector& output_types,
                      FunctionLibraryDefinition* flib, MutableGraphView* graph,
                      NodeDef** result) {
  TF_RETURN_IF_ERROR(flib->AddFunctionDef(flat_map_fn));
  AttrValue f;
  f.mutable_func()->set_name(flat_map_fn.signature().name());

  NodeDef flat_map_node;
  flat_map_node.set_op("FlatMapDataset");
  flat_map_node.add_input(input_dataset);
  for (const auto& arg : other_arguments) {
    flat_map_node.add_input(arg);
  }
  AddNodeAttr("f", f, &flat_map_node);
  AddNodeAttr("Targuments", t_arguments, &flat_map_node);
  AddNodeAttr(kOutputShapesAttr, output_shapes, &flat_map_node);
  AddNodeAttr(kOutputTypesAttr, output_types, &flat_map_node);

  graph_utils::SetUniqueGraphNodeName("rebatch/flat_map", graph->graph(),
                                      &flat_map_node);
  *result = graph->AddNode(std::move(flat_map_node));
  return Status::OK();
}

// def flat_map_fn(*batched_components):
//   ds = tf.data.Dataset.from_tensor_slices(batched_components)
//   return ds.batch(minibatch_size, drop_remainder=False)
Status CreateFlatMapFnWithBatch(const DataTypeVector& dtypes,
                                int64 num_replicas, FunctionDef* result) {
  NodeDef* tensor_slice_node = result->add_node_def();
  tensor_slice_node->set_op("TensorSliceDataset");
  for (int i = 0; i < dtypes.size(); ++i) {
    auto* input_arg = function_utils::AddFunctionInput(
        strings::StrCat("args_", i), result, dtypes.at(i));
    tensor_slice_node->add_input(input_arg->name());
  }
  AddNodeAttr(kTOutputTypesAttr, dtypes, tensor_slice_node);

  // The output_shapes attr here doesn't make a difference, since we
  // set the output_shapes of the external FlatMap node.
  AttrValue shapes;
  SetUnknownShapes(dtypes.size(), &shapes);
  AddNodeAttr(kOutputShapesAttr, shapes, tensor_slice_node);
  function_utils::SetUniqueFunctionNodeName("rebatch/from_tensor_slices",
                                            result, tensor_slice_node);

  NodeDef* false_node;
  TF_RETURN_IF_ERROR(AddConstBoolNode(false, result, &false_node));
  NodeDef* batch_node = result->add_node_def();
  batch_node->set_op(kBatchV2Op);
  batch_node->add_input(
      strings::StrCat(tensor_slice_node->name(), ":handle:0"));

  // `batch_size` input
  // Here, we capture the original batch size from outside the flat map fn.
  auto* original_batch_size =
      function_utils::AddFunctionInput("captured_batch_size", result, DT_INT64);
  NodeDef* new_batch_size;
  TF_RETURN_IF_ERROR(MakeNewBatchSizeNode(
      original_batch_size->name(), num_replicas, result, &new_batch_size));
  batch_node->add_input(strings::StrCat(new_batch_size->name(), ":z:0"));

  // `drop_remainder` input
  batch_node->add_input(strings::StrCat(false_node->name(), ":output:0"));
  AddNodeAttr(kOutputTypesAttr, dtypes, batch_node);
  AddNodeAttr(kOutputShapesAttr, shapes, batch_node);
  function_utils::SetUniqueFunctionNodeName("rebatch/batch", result,
                                            batch_node);
  function_utils::AddFunctionOutputWithUniqueName(
      "output", strings::StrCat(batch_node->name(), ":handle:0"), result,
      DT_VARIANT);
  // Because TensorSliceDataset is stateful, we set the function to stateful.
  result->mutable_signature()->set_is_stateful(true);

  return Status::OK();
}

// Rewrite graph to add
// `.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x).
//     batch(minibatch_size, drop_remainder=False))`
// after the batch node. This ensures that the sum of the minibatch sizes
// in a step adds up to the global batch size. However, since this adds
// additional data copies (both from_tensor_slices and batch), we only use
// this approach when necessary, i.e. when we need to drop remainder on the
// global batch, or when the global batch size does not divide num_replicas
// evenly.
Status AppendFlatMap(const NodeDef& batch_node, int64 num_replicas,
                     FunctionLibraryDefinition* flib, MutableGraphView* graph) {
  // `.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x).
  //     batch(minibatch_size, drop_remainder=False))`
  FunctionDef flat_map_fn;
  FunctionDefLibrary lib = flib->ToProto();
  graph_utils::SetUniqueGraphFunctionName("rebatch/flat_map_fn", &lib,
                                          &flat_map_fn);
  DataTypeVector dtypes;
  TF_RETURN_IF_ERROR(
      graph_utils::GetDatasetOutputTypesAttr(batch_node, &dtypes));
  TF_RETURN_IF_ERROR(
      CreateFlatMapFnWithBatch(dtypes, num_replicas, &flat_map_fn));

  int64 batch_size_index = GetBatchSizeArgIndex(batch_node);

  NodeDef* flat_map_node;

  AttrValue output_shapes = batch_node.attr().at(kOutputShapesAttr);
  for (auto& shape : *output_shapes.mutable_list()->mutable_shape()) {
    if (!shape.unknown_rank() && shape.dim(0).size() != -1) {
      // Because the flat map function uses drop_remainder = False,
      // the shape might be unknown
      auto old_dim = shape.dim(0).size();
      auto new_dim = old_dim % num_replicas == 0 ? old_dim / num_replicas : -1;
      shape.mutable_dim(0)->set_size(new_dim);
    }
  }

  TF_RETURN_IF_ERROR(AddFlatMapNode(strings::StrCat(batch_node.name(), ":0"),
                                    {batch_node.input(batch_size_index)},
                                    {DT_INT64}, flat_map_fn, output_shapes,
                                    dtypes, flib, graph, &flat_map_node));

  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(batch_node.name(), flat_map_node->name()));

  return Status::OK();
}

// There are several things we do here, depending on the values of
// batch_size and drop_remainder.
// (1) If batch size is known and divisible by num_replicas, and drop_remainder
// is known to be False, we mutate the batch size directly.
//   .batch(global_batch_size) -> .batch(global_batch_size // num_replicas)
// (2) Otherwise, we add a flat_map transformation to preserve the global batch
// size across the replicas and to preserve the drop remainder behavior.
bool ShouldMutateBatchSizeDirectly(const NodeDef& batch_node,
                                   int64 num_replicas,
                                   MutableGraphView* graph) {
  int64 batch_size_arg_index = GetBatchSizeArgIndex(batch_node);
  NodeDef* batch_size_node =
      graph_utils::GetInputNode(batch_node, *graph, batch_size_arg_index);

  int64 batch_size;
  Status s =
      graph_utils::GetScalarConstNodeValue(*batch_size_node, &batch_size);
  // If batch size is unknown or indivisible by num replicas, we don't
  // mutate it directly
  if (!s.ok() || batch_size % num_replicas != 0) return false;

  if (batch_node.op() == kBatchOp || batch_node.op() == kPaddedBatchOp) {
    // These ops don't have a `drop_remainder` input, and behave like
    // drop_remainder is False.
    return true;
  }

  // drop_remainder is the final input on the other batch nodes.
  NodeDef* drop_remainder_node = graph_utils::GetInputNode(
      batch_node, *graph, batch_node.input_size() - 1);
  bool drop_remainder;
  s = graph_utils::GetScalarConstNodeValue(*drop_remainder_node,
                                           &drop_remainder);
  return s.ok() && !drop_remainder;
}

Status RewriteBatchNode(const NodeDef& batch_node, int64 num_replicas,
                        FunctionLibraryDefinition* flib,
                        MutableGraphView* graph) {
  if (ShouldMutateBatchSizeDirectly(batch_node, num_replicas, graph)) {
    return MutateBatchSize(batch_node, num_replicas, graph);
  }
  return AppendFlatMap(batch_node, num_replicas, flib, graph);
}

Status OptimizeGraph(const GrapplerItem& item, int64 num_replicas,
                     bool use_fallback, GraphDef* output);

// Helper function that starts from a node in the graph and recurses into its
// inputs trying to find a BatchDataset type operation to modify. During the
// recursion it handles four kinds of cases.
// 1. BatchDataset type ops: Mutates the batch_size input node and stops.
// 2. Zip / Concatenate dataset ops: Recurses into all inputs to these ops
//      as they are datasets themselves.
// 3. Core dataset ops + Identity op: Recurses into first input parameter.
// 4. FlatMap type mapping dataset ops: Recurses into the function definition.
Status RecursivelyHandleOp(const NodeDef& node, int64 num_replicas,
                           bool use_fallback, FunctionLibraryDefinition* flib,
                           MutableGraphView* graph) {
  if (IsDatasetNodeOfType(node, kBatchDatasetOps)) {
    TF_RETURN_IF_ERROR(RewriteBatchNode(node, num_replicas, flib, graph));
  } else if (IsDatasetNodeOfType(node, kMultipleInputsDatasetOps)) {
    // For all multiple input datasets, all inputs are datasets themselves.
    for (int i = 0; i < node.input_size(); ++i) {
      NodeDef* input_node = graph_utils::GetInputNode(node, *graph, i);
      TF_RETURN_IF_ERROR(RecursivelyHandleOp(*input_node, num_replicas,
                                             use_fallback, flib, graph));
    }
  } else if (IsDatasetNodeOfType(node, kPassThroughOps) || IsRetval(node)) {
    // For all the dataset ops that are passthrough, or _Retvals added to the
    // function body graph in place of function outputs, the input dataset is
    // input 0.
    NodeDef* input_node = graph_utils::GetInputNode(node, *graph, 0);
    TF_RETURN_IF_ERROR(RecursivelyHandleOp(*input_node, num_replicas,
                                           use_fallback, flib, graph));
  } else if (IsDatasetNodeOfType(node, kFuncDatasetOps)) {
    const string func_name =
        node.attr().at(kFuncDatasetOpFuncs->at(node.op())).func().name();
    const FunctionDef* fdef = flib->Find(func_name);
    GrapplerFunctionItem f_item;
    TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(
        *fdef, *flib, graph->graph()->versions().producer(), &f_item));
    GraphDef optimized_func_graph;
    TF_RETURN_IF_ERROR(OptimizeGraph(f_item, num_replicas, use_fallback,
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
  TF_RETURN_IF_ERROR(UpdateOutputShapes(node.name(), num_replicas, graph));
  return Status::OK();
}

// Add nodes to the function to reshape arg to shape (-1, new_batch_dim, ...)
Status ReshapeComponent(int new_batch_dim, const string& arg, DataType dtype,
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

// def flat_map_fn(*batched_components):
//   return tf.data.Dataset.from_tensor_slices(
//     [tf.reshape(c, (-1, new_batch_size, ...))
//      for c in batched_components])
Status CreateFlatMapFnWithReshape(int new_batch_dim,
                                  const DataTypeVector& types,
                                  FunctionDef* result) {
  std::vector<NodeDefBuilder::NodeOut> tensor_slice_dataset_inputs;

  // For each component of the dataset, we reshape it from shape
  // (old_batch_size, ...) to (-1, new_batch_size, ...)
  // where new_batch_size = (old_batch_size + num_replicas - 1) // num_replicas
  for (int i = 0; i < types.size(); ++i) {
    auto* input_arg = function_utils::AddFunctionInput(
        strings::StrCat("args_", i), result, types.at(i));

    string reshape_node_name;
    TF_RETURN_IF_ERROR(ReshapeComponent(new_batch_dim, input_arg->name(),
                                        types.at(i), result,
                                        &reshape_node_name));

    tensor_slice_dataset_inputs.emplace_back(
        strings::StrCat(reshape_node_name, ":output"), 0, types.at(i));
  }

  // The output_shapes attr here doesn't make a difference, since we
  // set the output_shapes of the external FlatMap node.
  AttrValue shapes;
  SetUnknownShapes(types.size(), &shapes);

  NodeDef* tensor_slice_dataset = result->add_node_def();
  TF_RETURN_IF_ERROR(NodeDefBuilder("", "TensorSliceDataset")
                         .Input(tensor_slice_dataset_inputs)
                         .Attr("Toutput_types", types)
                         .Attr(kOutputShapesAttr, shapes)
                         .Finalize(tensor_slice_dataset));
  function_utils::SetUniqueFunctionNodeName("rebatch/tensor_slice_dataset",
                                            result, tensor_slice_dataset);

  function_utils::AddFunctionOutputWithUniqueName(
      "output", strings::StrCat(tensor_slice_dataset->name(), ":handle:0"),
      result, DT_VARIANT);
  // Because TensorSliceDataset is stateful, we set the function to stateful.
  result->mutable_signature()->set_is_stateful(true);

  return Status::OK();
}

// We fallback to the following rewrite:
// ```
//   dataset = ...fetch_node...
//   def fn(x):
//     return tf.data.Dataset.from_tensor_slices(
//       tf.reshape(
//         x,
//         tf.concat([[-1, old_batch_dim / num_replicas], tf.shape(x)[1:]], 0)
//       )
//     )
//
//   dataset = dataset.flat_map(fn)
// ```
Status RebatchWithFallback(const NodeDef* fetch_node, int64 num_replicas,
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
  if (!fetch_node->attr().contains(kOutputShapesAttr)) {
    return errors::InvalidArgument(
        "Cannot use rebatching fallback without output_shapes attr. Node: ",
        fetch_node->name(), " Op: ", fetch_node->op());
  } else {
    output_shapes = fetch_node->attr().at(kOutputShapesAttr);
  }
  int batch_dim;
  TF_RETURN_IF_ERROR(GetBatchDim(output_shapes, &batch_dim));
  if (batch_dim % num_replicas != 0) {
    return errors::InvalidArgument(
        "Cannot use rebatching fallback when batch dimension doesn't divide "
        "num_replicas evenly.");
  }

  // Create the flat map fn
  FunctionDef flat_map_fn;
  FunctionDefLibrary lib = flib->ToProto();
  graph_utils::SetUniqueGraphFunctionName("rebatch/flat_map_fn", &lib,
                                          &flat_map_fn);

  // Get types of input arguments from the output types of the final dataset.
  DataTypeVector output_types;
  TF_RETURN_IF_ERROR(
      graph_utils::GetDatasetOutputTypesAttr(*fetch_node, &output_types));
  TF_RETURN_IF_ERROR(CreateFlatMapFnWithReshape(batch_dim / num_replicas,
                                                output_types, &flat_map_fn));

  NodeDef* flat_map_node;
  TF_RETURN_IF_ERROR(AddFlatMapNode(strings::StrCat(fetch_node->name(), ":0"),
                                    {}, {}, flat_map_fn, output_shapes,
                                    output_types, flib, graph, &flat_map_node));
  TF_RETURN_IF_ERROR(
      UpdateOutputShapes(flat_map_node->name(), num_replicas, graph));

  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(fetch_node->name(), flat_map_node->name()));

  return Status::OK();
}

// Helper function that given a GrapplerItem generates a mutated graph def
// with the batch size changed. The GrapplerItem could be generated from the
// main graph or could be a function graph.
Status OptimizeGraph(const GrapplerItem& item, int64 num_replicas,
                     bool use_fallback, GraphDef* output) {
  *output = item.graph;
  MutableGraphView graph(output);

  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());

  NodeDef* sink_node;
  TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));

  Status s = RecursivelyHandleOp(*sink_node, num_replicas, use_fallback, &flib,
                                 &graph);
  if (!s.ok()) {
    if (use_fallback) {
      VLOG(1) << "Failed to rebatch by rewriting the batch transformation ("
              << s << "). Using a fallback method instead.";
      // If RecursivelyHandleOp fails, we reset `graph` to use the original,
      // graph, since that function may have mutated `graph`.
      *output = item.graph;
      graph = MutableGraphView(output);
      TF_RETURN_IF_ERROR(
          RebatchWithFallback(sink_node, num_replicas, &flib, &graph));
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

  TF_RETURN_IF_ERROR(OptimizeGraph(item, num_replicas_, use_fallback_, output));
  stats->num_changes++;
  return Status::OK();
}

void RebatchOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                const GraphDef& optimize_output,
                                double result) {}

REGISTER_GRAPH_OPTIMIZER_AS(RebatchOptimizer, "tf_data_rebatcher");

}  // namespace grappler
}  // namespace tensorflow
