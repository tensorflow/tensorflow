/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/map_vectorization.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization_utils.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

// Returns a FunctionDef containing a MapDefun op that wraps the original
// function.
FunctionDef* CreateMapDefunWrapper(const NodeDef& map_node,
                                   const FunctionDef& orig_func,
                                   FunctionDefLibrary* library) {
  FunctionDef* vectorized_func = library->add_function();
  // Function inputs and outputs are the same as original, just
  // with different shapes.
  *vectorized_func->mutable_signature() = orig_func.signature();
  graph_utils::SetUniqueGraphFunctionName("vectorized_function", library,
                                          vectorized_func);

  // Add MapDefun node
  NodeDef* map_defun_node = vectorized_func->mutable_node_def()->Add();
  map_defun_node->set_op("MapDefun");
  function_utils::SetUniqueFunctionNodeName(map_defun_node->op(),
                                            vectorized_func, map_defun_node);

  // Set attrs and inputs
  for (const string& k : {"f", "output_types", "output_shapes"}) {
    // Function, output types and (unbatched) shapes are the same as the
    // original map node.
    graph_utils::CopyAttribute(k, map_node, map_defun_node);
  }

  // Get types of input arguments from original map function
  AttrValue t_args;
  for (const auto& input : vectorized_func->signature().input_arg()) {
    t_args.mutable_list()->add_type(input.type());
    map_defun_node->add_input(input.name());
  }
  (*map_defun_node->mutable_attr())["Targuments"] = t_args;

  // Set return values to match output names
  string output_prefix = strings::StrCat(map_defun_node->name(), ":output:");
  for (size_t i = 0; i < vectorized_func->signature().output_arg_size(); ++i) {
    const auto& output_arg = vectorized_func->signature().output_arg(i);
    (*vectorized_func->mutable_ret())[output_arg.name()] =
        strings::StrCat(output_prefix, i);
  }

  return vectorized_func;
}

FunctionDef* AddVectorizedFunction(const NodeDef& map_node,
                                   const FunctionDef& orig_func,
                                   FunctionDefLibrary* library) {
  // Vectorizes orig_func naively by wrapping in a MapDefun op, then performing
  // efficient vectorization with VectorizeMapDefun.
  FunctionDef* vectorized_func =
      CreateMapDefunWrapper(map_node, orig_func, library);
  const NodeDef& map_defun_node = vectorized_func->node_def(0);
  DCHECK_EQ(map_defun_node.op(), "MapDefun");

  // TODO(b/116285210): Unreferenced functions should get cleaned up later
  FunctionDef* result;
  Status s = vectorization_utils::VectorizeMapDefun(
      *vectorized_func, map_defun_node, library, &result);

  if (!s.ok()) {
    LOG(ERROR) << "VectorizeMapDefun failed: " << s;
    return vectorized_func;
  }
  return result;
}

bool IsOutputShapesFullyDefined(const NodeDef& node) {
  auto* shapes_attr = gtl::FindOrNull(node.attr(), "output_shapes");
  if (shapes_attr == nullptr) return false;
  const auto& shapes = shapes_attr->list().shape();

  for (const TensorShapeProto& shape : shapes) {
    for (const auto& dim : shape.dim()) {
      if (dim.size() == -1) {
        return false;
      }
    }
  }
  return true;
}

bool IsStatefulFn(const FunctionLibraryDefinition& library,
                  const FunctionDef& function_def) {
  for (const NodeDef& node_def : function_def.node_def()) {
    const OpDef* op_def;
    Status s = library.LookUpOpDef(node_def.op(), &op_def);
    if (!s.ok() || op_def->is_stateful()) {
      return true;
    }
  }
  return false;
}

bool HasCapturedInputs(const NodeDef& map_node) {
  return map_node.attr().at("Targuments").list().type_size() > 0;
}

NodeDef MakeNewBatchNode(const NodeDef& old_batch_node,
                         const NodeDef& input_node,
                         const FunctionDef& vectorized_func,
                         MutableGraphView* graph) {
  NodeDef batch_node;
  batch_node.set_op(old_batch_node.op());
  graph_utils::SetUniqueGraphNodeName(batch_node.op(), graph->GetGraph(),
                                      &batch_node);

  // Set the `input_dataset` input argument
  batch_node.add_input(input_node.name());
  // Set the `batch_size` input_argument
  batch_node.add_input(old_batch_node.input(1));
  if (batch_node.op() == "BatchDatasetV2") {
    // Set the `drop_remainder` input argument
    batch_node.add_input(old_batch_node.input(2));
  }

  // Set attrs
  AttrValue output_types;
  for (const auto& input : vectorized_func.signature().input_arg()) {
    output_types.mutable_list()->add_type(input.type());
  }
  (*batch_node.mutable_attr())["output_types"] = output_types;

  auto& output_shapes_attr = (*batch_node.mutable_attr())["output_shapes"];
  const auto& input_shapes =
      input_node.attr().at("output_shapes").list().shape();
  int64 batch_size =
      old_batch_node.attr().at("output_shapes").list().shape()[0].dim(0).size();
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    TensorShapeProto* shape = output_shapes_attr.mutable_list()->add_shape();
    TensorShapeProto_Dim* dim = shape->add_dim();
    dim->set_size(batch_size);
    shape->MergeFrom(input_shapes.Get(i));
  }
  return batch_node;
}

NodeDef MakeNewMapNode(const NodeDef& old_map_node,
                       const NodeDef& old_batch_node,
                       const NodeDef& new_batch_node,
                       const FunctionDef& vectorized_func,
                       MutableGraphView* graph) {
  NodeDef map_node;
  map_node.set_op(old_map_node.op());
  graph_utils::SetUniqueGraphNodeName(map_node.op(), graph->GetGraph(),
                                      &map_node);

  // Set the `input_dataset` input argument
  map_node.add_input(new_batch_node.name());
  for (int i = 1; i < old_map_node.input_size(); i++) {
    // Set the `other_arguments` and `num_parallel_calls` input arguments
    map_node.add_input(old_map_node.input(i));
  }

  // Set attrs
  graph_utils::CopyAttribute("Targuments", old_map_node, &map_node);
  auto& func_attr = (*map_node.mutable_attr())["f"];
  func_attr.mutable_func()->set_name(vectorized_func.signature().name());

  for (auto key : {"output_shapes", "output_types"}) {
    graph_utils::CopyAttribute(key, old_batch_node, &map_node);
  }

  (*map_node.mutable_attr())["use_inter_op_parallelism"].set_b(true);

  return map_node;
}

}  // namespace

Status MapVectorization::Optimize(Cluster* cluster, const GrapplerItem& item,
                                  GraphDef* output) {
  *output = item.graph;
  MutableGraphView graph(output);
  std::set<string> nodes_to_delete;

  for (const NodeDef& node : item.graph.node()) {
    // Find Map->Batch nodes.
    // TODO(rachelim): Optimize MapAndBatchDataset[V2] as well.
    if (node.op() != "BatchDataset" && node.op() != "BatchDatasetV2") {
      continue;
    }

    const NodeDef& batch_node(node);
    NodeDef* node2 = graph_utils::GetInputNode(batch_node, graph);
    if (node2->op() != "MapDataset" && node2->op() != "ParallelMapDataset") {
      continue;
    }

    // Use a more descriptive variable name now that we know the node type.
    NodeDef* map_node = node2;
    // Input to the map node
    NodeDef* input_node = graph_utils::GetInputNode(*map_node, graph);
    CHECK_NOTNULL(input_node);

    FunctionDefLibrary* library = output->mutable_library();

    FunctionLibraryDefinition function_library(OpRegistry::Global(), *library);
    const FunctionDef* orig_func =
        function_library.Find(map_node->attr().at("f").func().name());

    // Check that this is a valid optimization.
    if (!IsOutputShapesFullyDefined(*input_node) ||
        !IsOutputShapesFullyDefined(*map_node) ||
        IsStatefulFn(function_library, *orig_func) ||
        HasCapturedInputs(*map_node)) {
      // 1. If any of the inputs have an unknown shape, don't optimize, since
      // inputs might not be batchable.
      // 2. If any of the map func outputs have an unknown shape, don't
      // optimize, so that batching errors surface as before.
      // 3. If the function is stateful, don't vectorize it.
      // 4. TODO(rachelim): Make this work for MapDataset with captured inputs
      // by tiling inputs or modifying the signature of MapDefun.
      continue;
    }

    FunctionDef* vectorized_func =
        AddVectorizedFunction(*map_node, *orig_func, library);
    CHECK_NOTNULL(vectorized_func);

    auto* new_batch_node = graph.AddNode(
        MakeNewBatchNode(batch_node, *input_node, *vectorized_func, &graph));

    auto* new_map_node = graph.AddNode(MakeNewMapNode(
        *map_node, batch_node, *new_batch_node, *vectorized_func, &graph));
    graph.ReplaceInput(batch_node, *new_map_node);

    // Mark the `Map` and `Batch` nodes for removal.
    nodes_to_delete.insert(map_node->name());
    nodes_to_delete.insert(batch_node.name());
  }
  graph.DeleteNodes(nodes_to_delete);
  return Status::OK();
}

void MapVectorization::Feedback(Cluster* cluster, const GrapplerItem& item,
                                const GraphDef& optimize_output,
                                double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(MapVectorization, "map_vectorization");

}  // end namespace grappler
}  // end namespace tensorflow
