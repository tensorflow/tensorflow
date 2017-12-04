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
#include "tensorflow/core/common_runtime/shape_refiner.h"

#include <deque>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

ShapeRefiner::ShapeRefiner(int graph_def_version,
                           const OpRegistryInterface* ops)
    : graph_def_version_(graph_def_version),
      ops_registry_(ops),
      graph_runner_(Env::Default()) {}

ShapeRefiner::ShapeRefiner(const VersionDef& versions,
                           const OpRegistryInterface* ops)
    : ShapeRefiner(versions.producer(), ops) {}

ShapeRefiner::~ShapeRefiner() {
  // The lifetime of the tensors are bound to the GraphRunner, so the tensors
  // should be deleted before it.
  const_tensor_map_.clear();
}

namespace {

constexpr char kArgOp[] = "_Arg";
constexpr char kRetvalOp[] = "_Retval";

// Runs shape inference for the given node using the given ShapeRefiner.
// The node must be a sub-node of a function node and the outer_context is
// the inference context of that function node in the outer graph.
Status InferShapesForFunctionSubNode(const Node* node, ShapeRefiner* refiner,
                                     InferenceContext* outer_context) {
  TF_RETURN_IF_ERROR(refiner->AddNode(node));
  InferenceContext* node_context = CHECK_NOTNULL(refiner->GetContext(node));

  if (StringPiece(node->type_string()) == kArgOp) {
    // Handle special node: function input.
    // Shapes for these nodes are provided in the outer inference
    // context.

    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node->def()), "index", &index));

    if (index < 0 || outer_context->num_inputs() <= index) {
      return errors::Internal(
          "Function instantiation included invalid input index: ", index,
          " not in [0, ", outer_context->num_inputs(), ").");
    }

    node_context->set_output(0, outer_context->input(index));

    auto* resource = outer_context->input_handle_shapes_and_types(index);
    if (resource) {
      node_context->set_output_handle_shapes_and_types(0, *resource);
    }
  } else if (StringPiece(node->type_string()) == kRetvalOp) {
    // Handle special node: function output.
    // Shapes inferred for these nodes go into the outer inference
    // context.

    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node->def()), "index", &index));

    if (index < 0 || outer_context->num_outputs() <= index) {
      return errors::Internal(
          "Function instantiation included invalid output index: ", index,
          " not in [0, ", outer_context->num_outputs(), ").");
    }

    // outer_context outlives node_context, therefore we need to create
    // a new shape handle owned by outer_context instead.
    ShapeHandle handle;
    TensorShapeProto proto;
    node_context->ShapeHandleToProto(node_context->input(0), &proto);
    TF_RETURN_IF_ERROR(outer_context->MakeShapeFromShapeProto(proto, &handle));
    outer_context->set_output(index, handle);

    auto* resource = node_context->input_handle_shapes_and_types(0);
    if (resource) {
      outer_context->set_output_handle_shapes_and_types(index, *resource);
    }
  }

  return Status::OK();
}

}  // namespace

// TODO(cwhipkey): When an inference context inside function has
// requested_input_tensor(i) or requested_input_tensor_as_partial_shape(i)
// set when input(i) is an _Arg op, then this request should propagate to
// context, and vice versa.
//
// NOTE: Recursive user-defined functions are not supported.
// Maybe we won't support recursive functions at all in TF, because of
// other maintanabilty issues.
Status ShapeRefiner::InferShapesForFunction(
    const tensorflow::FunctionDef* function_def, bool keep_nested_shapes,
    ExtendedInferenceContext* outer_context) {
  const Graph* graph;
  auto it = functions_.find(function_def);
  if (it != functions_.end()) {
    graph = it->second.get();
  } else {
    InstantiationResult result;
    TF_RETURN_IF_ERROR(InstantiateFunction(
        *function_def, outer_context->get_context()->attrs(),
        [this](const string& op, const OpDef** sig) {
          return this->function_library_->LookUpOpDef(op, sig);
        },
        &result));

    Graph* new_graph = new Graph(function_library_);
    GraphConstructorOptions options;
    options.allow_internal_ops = true;
    TF_RETURN_IF_ERROR(
        ConvertNodeDefsToGraph(options, result.nodes, new_graph));
    functions_[function_def].reset(new_graph);
    graph = new_graph;
  }

  std::unordered_set<const Node*> function_nodes;
  Status inference_status = Status::OK();
  {
    auto node_shape_inference_lambda = [this, &outer_context, &function_nodes,
                                        &inference_status](const Node* node) {
      if (!inference_status.ok()) return;
      inference_status = InferShapesForFunctionSubNode(
          node, this, outer_context->get_context());
      function_nodes.insert(node);
    };

    // Calls inference lambda for each node after visiting all predecessors.
    // Ensures that we are adding nodes to ShapeRefiner in the topological
    // order.
    ReverseDFS(*graph, {}, node_shape_inference_lambda);
  }

  if (keep_nested_shapes && inference_status.ok()) {
    // Fill the nested inferences map.
    //
    // The materialized function graph has extra nodes for arguments and
    // return values, which are not explicitly listed in the FunctionDef,
    // we filter out these special nodes here to not expose the implementation
    // details and keep only inferences for the nodes listed in the FunctionDef.
    std::unordered_map<string, const NodeDef*> user_defined_nodes;
    for (const auto& node_def : function_def->node_def()) {
      user_defined_nodes[node_def.name()] = &node_def;
    }

    std::unordered_map<string, std::unique_ptr<ExtendedInferenceContext>>
        nested_inferences;
    for (const Node* node : function_nodes) {
      const string& node_name = node->name();
      if (user_defined_nodes.find(node_name) != user_defined_nodes.end()) {
        nested_inferences[node_name] = std::move(node_to_context_[node]);
        node_to_context_.erase(node);
        // By default InferenceContext refers to a NodeDef from Graph.
        // Change it to the publicly accessible NodeDef of the function
        // definition.
        nested_inferences[node_name]->get_context()->node_def_ =
            user_defined_nodes[node_name];
      }
    }
    outer_context->set_nested_inferences(std::move(nested_inferences));
  } else {
    // Delete the contexts created for the functions nodes to save memory.
    for (const Node* node : function_nodes) {
      node_to_context_.erase(node);
    }
  }

  return inference_status;
}

Status ShapeRefiner::AddNode(const Node* node) {
  // For each 'input' of this node, fetch the corresponding shape
  // from 'input's InferenceContext, and store into a vector
  // indexed by 'node's input.
  std::vector<Node*> input_nodes(node->num_inputs());
  std::vector<ShapeHandle> input_shapes(node->num_inputs());
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
      input_handle_shapes_and_types(node->num_inputs());
  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) continue;

    Node* input = e->src();
    auto it = node_to_context_.find(input);
    if (it == node_to_context_.end()) {
      return errors::FailedPrecondition(
          "Input ", e->dst_input(), " ('", input->name(), "') for '",
          node->name(), "' was not previously added to ShapeRefiner.");
    }

    InferenceContext* c = it->second->get_context();
    DCHECK_GE(e->dst_input(), 0);
    input_nodes[e->dst_input()] = input;
    input_shapes[e->dst_input()] = c->output(e->src_output());

    // Only propagate handle data of edges which are carrying resource handles.
    if (e->src()->output_type(e->src_output()) == DT_RESOURCE) {
      const auto* in_v = c->output_handle_shapes_and_types(e->src_output());
      if (in_v != nullptr) {
        input_handle_shapes_and_types[e->dst_input()].reset(
            new std::vector<ShapeAndType>(*in_v));
      }
    }
  }

  // Get the shape function for this node
  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(ops_registry_->LookUp(node->type_string(), &op_reg_data));
  if (op_reg_data->shape_inference_fn == nullptr &&
      require_shape_inference_fns_) {
    return errors::InvalidArgument(
        "No shape inference function exists for op '", node->type_string(),
        "', did you forget to define it?");
  }

  // This needs to be filled in with real data in a second pass.
  std::vector<const Tensor*> input_tensors(node->num_inputs(), nullptr);
  std::vector<ShapeHandle> input_tensors_as_shapes;

  // Create the inference context for this node with the existing input shapes.
  std::unique_ptr<InferenceContext> c(
      new InferenceContext(graph_def_version_, &node->def(), node->op_def(),
                           input_shapes, input_tensors, input_tensors_as_shapes,
                           std::move(input_handle_shapes_and_types)));
  if (!c->construction_status().ok()) {
    return c->construction_status();
  }

  std::unique_ptr<ExtendedInferenceContext> ec(
      new ExtendedInferenceContext(std::move(c), node));

  // Run the shape inference function, and return if there was an error.
  TF_RETURN_IF_ERROR(RunShapeFn(node, op_reg_data, ec.get()));

  // Store the resulting context object in the map.
  node_to_context_[node].swap(ec);

  return Status::OK();
}

Status ShapeRefiner::SetShape(const Node* node, int output_port,
                              ShapeHandle shape) {
  auto c = GetContext(node);
  if (c == nullptr) {
    return errors::Internal("Could not find context for ", node->name());
  }

  if (output_port < 0 || output_port >= node->num_outputs()) {
    return errors::InvalidArgument(
        "output_port '", output_port, "' is out of range, ", "node '",
        node->name(), "' has ", node->num_outputs(), " outputs");
  }

  // Check compatibility, and merge the shapes.
  ShapeHandle existing_shape = c->output(output_port);
  TF_RETURN_IF_ERROR(c->Merge(existing_shape, shape, &shape));
  c->set_output(output_port, shape);

  // TODO(vrv): Do we need to propagate the new shape through all
  // consumers that change their outputs?  At the moment, python
  // does not do this, but this seems like a nice feature.

  // TODO(vrv): We might need to keep track of the fact that the
  // existing shape is invalidated, in case we need to propagate
  // this information to remote workers.
  return Status::OK();
}

Status ShapeRefiner::UpdateNode(const Node* node, bool relax, bool* refined) {
  auto it = node_to_context_.find(node);
  if (it == node_to_context_.end()) {
    *refined = true;
    return AddNode(node);
  }
  ExtendedInferenceContext* node_ext_context = it->second.get();
  InferenceContext* node_context = node_ext_context->get_context();

  // Give up if the context wasn't successfully built by the AddNode() method.
  TF_RETURN_IF_ERROR(node_context->construction_status());

  // Check if the shapes of the nodes in the fan-in of this node have changed,
  // and if they have update the node input shapes.
  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) continue;

    int dst_input = e->dst_input();
    int src_output = e->src_output();

    Node* input = e->src();
    auto iter = node_to_context_.find(input);
    if (iter == node_to_context_.end()) {
      return errors::FailedPrecondition(
          "Input ", dst_input, " ('", input->name(), "') for '", node->name(),
          "' was not previously added to ShapeRefiner.");
    }

    InferenceContext* c = iter->second->get_context();
    DCHECK_GE(dst_input, 0);
    ShapeHandle existing_input = node_context->input(dst_input);
    if (!relax && node_context->MergeInput(dst_input, c->output(src_output)) &&
        !existing_input.SameHandle(node_context->input(dst_input))) {
      *refined = true;
    } else if (relax) {
      if (node_context->RelaxInput(dst_input, c->output(src_output))) {
        if (!SameDefinedShape(node_context, node_context->input(dst_input),
                              existing_input)) {
          *refined = true;
        }
      }
    }

    // Also propagate handle shape and dtype of edges which are carrying
    // resource handles.
    if (e->src()->output_type(src_output) == DT_RESOURCE) {
      auto* outputs = c->output_handle_shapes_and_types(src_output);
      if (!outputs) continue;

      if (!relax &&
          node_context->MergeInputHandleShapesAndTypes(dst_input, *outputs)) {
        *refined = true;
      } else if (relax) {
        std::vector<ShapeAndType> existing_inputs;
        const std::vector<ShapeAndType>* inputs =
            node_context->input_handle_shapes_and_types(dst_input);
        if (inputs) {
          existing_inputs = *inputs;
        }
        if (node_context->RelaxInputHandleShapesAndMergeTypes(dst_input,
                                                              *outputs)) {
          if (IsUpdatedShapesOrTypes(
                  node_context, existing_inputs,
                  *node_context->input_handle_shapes_and_types(dst_input))) {
            *refined = true;
          }
        }
      }
    }
  }

  if (!*refined) {
    // No input shape has changed, we're done
    return Status::OK();
  }

  // Get and run the shape function for this node to update the shapes of the
  // outputs.
  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(ops_registry_->LookUp(node->type_string(), &op_reg_data));
  if (op_reg_data->shape_inference_fn == nullptr &&
      require_shape_inference_fns_) {
    return errors::InvalidArgument(
        "No shape inference function exists for op '", node->type_string(),
        "', did you forget to define it?");
  }

  if (!op_reg_data->shape_inference_fn) {
    // There is nothing more we can infer
    return Status::OK();
  }

  return RunShapeFn(node, op_reg_data, node_ext_context);
}

Status ShapeRefiner::EvaluateConstantTensorForEdge(const Node* node,
                                                   int dst_idx, bool* evaluated,
                                                   Tensor* result) {
  *evaluated = false;

  const Edge* input_edge;
  TF_RETURN_IF_ERROR(node->input_edge(dst_idx, &input_edge));

  // Simple case: the source node is a constant
  const Node* src = input_edge->src();
  if (src->IsConstant()) {
    if (result->FromProto(src->def().attr().at("value").tensor())) {
      *evaluated = true;
      return Status::OK();
    }
  }

  if (disable_constant_propagation_) {
    return Status::OK();
  }

  bool is_constant_graph = false;
  Graph subgraph(ops_registry_);
  auto versions = subgraph.versions();
  versions.set_producer(graph_def_version_);
  subgraph.set_versions(versions);

  // We identify the possibly constant subgraph to evaluate by
  // recursively iterating backwards through the inputs to 'node'
  // until we either 1) find an already existing input to our subgraph
  // (filled in `const_inputs`), 2) Discover our graph is not constant,
  // or 3) Hit a root node.
  std::vector<std::pair<string, Tensor>> const_inputs;
  TF_RETURN_IF_ERROR(ExtractConstantSubgraph(
      input_edge->src(), &subgraph, &is_constant_graph, &const_inputs));
  if (!is_constant_graph) {
    return Status::OK();
  }
  const string output_tensor_name =
      strings::StrCat(input_edge->src()->name(), ":", input_edge->src_output());
  std::vector<Tensor> outputs;

  // NOTE; we should pass in a function library runtime if we want
  // to support constant-expression evaluation on functions.
  Status s = graph_runner_.Run(&subgraph, nullptr /* function_library */,
                               const_inputs, {output_tensor_name}, &outputs);

  // If all kernels in the constant graph are not registered
  // in the process, GraphRunner::Run may fail, in which case
  // we cannot propagate constants, so this is best-effort.
  if (s.ok()) {
    *result = outputs[0];
    *evaluated = true;

    // We memoize (small) constants evaluated so far, so
    // ExtractConstantSubgraph can avoid extracting the full
    // subgraph.  As we build up large graphs, this avoids
    // repeated computation of the early parts of a constant
    // graph.
    if (outputs[0].TotalBytes() <= kMaxTensorSize) {
      const_tensor_map_[output_tensor_name] = outputs[0];
    }
  }
  return Status::OK();
}

Status ShapeRefiner::TryToInferTensorOutputFromInputShapes(const Edge* edge,
                                                           Tensor* output,
                                                           bool* success) {
  *success = false;
  const Node* node = edge->src();
  auto it = node_to_context_.find(node);
  if (it == node_to_context_.end()) {
    return errors::FailedPrecondition("Node does not have context.");
  }
  InferenceContext* c = it->second->get_context();

  if (node->type_string() == "Shape") {
    // If input shapes to the shape op are fully defined,
    // we can infer the shape op's output tensor.
    bool fully_defined_inputs = c->FullyDefined(c->input(0));
    if (fully_defined_inputs) {
      int input_rank = c->Rank(c->input(0));
      Tensor t(node->output_type(0), TensorShape({input_rank}));
      if (node->output_type(0) == DT_INT32) {
        auto flat = t.flat<int>();
        for (int i = 0; i < input_rank; i++) {
          int64 dimension = c->Value(c->Dim(c->input(0), i));
          if (!FastBoundsCheck(dimension, std::numeric_limits<int32>::max())) {
            return errors::FailedPrecondition(
                "Shape has output type int32, but dimension exceeds maximum "
                "int32 value");
          }
          flat(i) = static_cast<int32>(dimension);
        }
      } else if (node->output_type(0) == DT_INT64) {
        auto flat = t.flat<int64>();
        for (int i = 0; i < input_rank; i++) {
          flat(i) = c->Value(c->Dim(c->input(0), i));
        }
      } else {
        return errors::FailedPrecondition(
            "Shape has output type that is not int32 or int64");
      }
      *output = t;
      *success = true;
    }
  } else if (node->type_string() == "Rank") {
    bool rank_known = c->RankKnown(c->input(0));
    if (rank_known) {
      int32 input_rank = c->Rank(c->input(0));
      Tensor t(node->output_type(0), TensorShape({}));
      t.flat<int32>()(0) = input_rank;
      *output = t;
      *success = true;
    }
  } else if (node->type_string() == "Size") {
    bool fully_defined_inputs = c->FullyDefined(c->input(0));
    if (fully_defined_inputs) {
      int32 rank = c->Rank(c->input(0));
      Tensor t(node->output_type(0), TensorShape({}));
      int64 size = 1;
      for (int i = 0; i < rank; i++) {
        size *= c->Value(c->Dim(c->input(0), i));
      }
      if (node->output_type(0) == DT_INT32) {
        if (!FastBoundsCheck(size, std::numeric_limits<int32>::max())) {
          return errors::FailedPrecondition(
              "Size has output type int32, but size exceeds maximum int32 "
              "value");
        }
        t.flat<int32>()(0) = static_cast<int32>(size);
      } else if (node->output_type(0) == DT_INT64) {
        t.flat<int64>()(0) = size;
      } else {
        return errors::FailedPrecondition(
            "Size has output type that is not int32 or int64");
      }
      *output = t;
      *success = true;
    }
  }
  return Status::OK();
}

Status ShapeRefiner::ExtractConstantSubgraph(
    Node* target_node, Graph* out_graph, bool* is_constant_graph,
    std::vector<std::pair<string, Tensor>>* const_inputs) {
  *is_constant_graph = false;
  std::unordered_set<string> const_inputs_added;

  if (target_node->op_def().is_stateful()) {
    return Status::OK();
  }

  struct NodeAndRecursed {
    Node* new_node = nullptr;
    bool recursed = false;
  };

  std::map<Node*, NodeAndRecursed> old_to_new_and_recursed;
  Node* target_node_copy = out_graph->CopyNode(target_node);
  old_to_new_and_recursed[target_node].new_node = target_node_copy;
  old_to_new_and_recursed[target_node].recursed = true;

  // Add the target node's inputs to seed the recursion.
  std::deque<const Edge*> edges_to_visit;
  for (const Edge* e : target_node->in_edges()) {
    // TODO(vrv): What do we do about control edges?  Based on our
    // definition of a constant graph, we should be free to ignore
    // control edges since the order in which a constant graph is
    // executed should be the same regardless of when nodes run: we
    // should only need to recurse down data edges.
    if (e->IsControlEdge()) continue;
    edges_to_visit.push_back(e);
  }

  *is_constant_graph = true;

  // Iterate over the set of edges to visit (backwards).
  while (!edges_to_visit.empty()) {
    const Edge* current_edge = edges_to_visit.front();
    edges_to_visit.pop_front();
    Node* current_node = current_edge->src();

    // If the node is stateful, assume the graph is not constant.
    if (current_node->op_def().is_stateful()) {
      *is_constant_graph = false;
      return Status::OK();
    }

    // During construction or import from GraphConstructor, back edges may not
    // be filled in.  Don't constant fold through merges at all for now.
    if (IsMerge(current_node)) {
      *is_constant_graph = false;
      return Status::OK();
    }

    // Don't constant fold enter/exit currently either, as it's easy to end
    // up with a partial frame.
    if (IsEnter(current_node) || IsExit(current_node)) {
      *is_constant_graph = false;
      return Status::OK();
    }

    // If there is nothing more to recurse down, see if
    // the generator node is a constant.
    if (current_node->num_inputs() == 0) {
      if (!current_node->IsConstant()) {
        // Generator node is not a constant, so subgraph is not
        // constant.
        *is_constant_graph = false;
        return Status::OK();
      }
    }

    // Either the node is a constant, or the node is a potential
    // intermediate node on the path from a constant.
    //
    // Add a copy of its node and a new edge to the new subgraph.

    // Get or create the version of 'current_node' in the new graph.
    Node* current_node_copy;
    // This gets or creates the NodeAndRecursed entry for current_node.
    NodeAndRecursed* node_and_recursed = &old_to_new_and_recursed[current_node];
    if (node_and_recursed->new_node == nullptr) {
      // First time processing this node.
      current_node_copy = out_graph->CopyNode(current_node);
      // Track the mapping from the original node to the new one.
      node_and_recursed->new_node = current_node_copy;
    } else {
      current_node_copy = node_and_recursed->new_node;
    }

    // Add the edge to the destination node.
    {
      auto it = old_to_new_and_recursed.find(current_edge->dst());
      if (it == old_to_new_and_recursed.end()) {
        return errors::Internal(
            "Could not find mapping from old to new copy of destination node: ",
            current_edge->dst()->name());
      }
      Node* dst_copy = it->second.new_node;

      out_graph->AddEdge(current_node_copy, current_edge->src_output(),
                         dst_copy, current_edge->dst_input());
    }

    const string& output_tensor_name =
        strings::StrCat(current_node->name(), ":", current_edge->src_output());

    // Some tensor values can be inferred. For example, a shape op
    // with input shapes fully defined can have its output tensor inferred.
    Tensor tensor_inferred;
    bool successfully_inferred_tensor = false;
    TF_RETURN_IF_ERROR(TryToInferTensorOutputFromInputShapes(
        current_edge, &tensor_inferred, &successfully_inferred_tensor));
    if (successfully_inferred_tensor) {
      const_inputs->emplace_back(output_tensor_name, tensor_inferred);
      const_inputs_added.insert(output_tensor_name);
      continue;
    }

    // If we have a copy of the input tensor materialized already,
    // then add to the list of inputs to feed and do not recurse further.
    auto it = const_tensor_map_.find(output_tensor_name);
    if (it != const_tensor_map_.end() &&
        const_inputs_added.count(output_tensor_name) == 0) {
      const_inputs->emplace_back(output_tensor_name, it->second);
      const_inputs_added.insert(output_tensor_name);
      continue;
    }

    // If this node's inputs have not been processed already, do so now.
    if (!node_and_recursed->recursed) {
      node_and_recursed->recursed = true;
      for (const Edge* e : current_node->in_edges()) {
        if (e->IsControlEdge()) continue;
        edges_to_visit.push_back(e);
      }
    }
  }

  return Status::OK();
}

Status ShapeRefiner::ConstantPartialShape(InferenceContext* target_context,
                                          const Node* node, int dst_idx,
                                          ShapeHandle* result) {
  const Edge* input_edge;
  TF_RETURN_IF_ERROR(node->input_edge(dst_idx, &input_edge));

  InferenceContext* src_context = GetContext(input_edge->src());
  if (src_context == nullptr) return errors::Internal("Missing src context");
  ShapeHandle src_shape = src_context->output(input_edge->src_output());
  TF_RETURN_IF_ERROR(src_context->WithRank(src_shape, 1, &src_shape));

  const string& src_op = input_edge->src()->type_string();
  if (src_context->Value(src_context->Dim(src_shape, 0)) == 0) {
    // Source tensor is a vector of length 0, so the shape it
    // represents is as scalar.
    *result = target_context->Scalar();
  } else if (src_op == "Shape") {
    *result = src_context->input(0);
  } else if (src_op == "Pack") {
    std::vector<DimensionHandle> dims;
    // Pack is concatenating its input scalars to form the shape tensor vector.
    for (int i = 0; i < src_context->num_inputs(); ++i) {
      Tensor scalar;
      bool evaluated = false;
      TF_RETURN_IF_ERROR(EvaluateConstantTensorForEdge(input_edge->src(), i,
                                                       &evaluated, &scalar));
      if (evaluated) {
        int64 size;
        if (scalar.dtype() == DT_INT32) {
          size = scalar.scalar<int32>()();
        } else if (scalar.dtype() == DT_INT64) {
          size = scalar.scalar<int64>()();
        } else {
          return errors::InvalidArgument("Pack input must be int32 or int64");
        }
        dims.push_back(size < 0 ? target_context->UnknownDim()
                                : target_context->MakeDim(size));
      } else {
        dims.push_back(target_context->UnknownDim());
      }
    }
    *result = target_context->MakeShape(dims);
  } else if (src_op == "Concat" || src_op == "ConcatV2") {
    *result = target_context->Scalar();
    // For Concat, input 0 is concat dim; for V2 it is the last input.
    const int concat_dim =
        src_op == "Concat" ? 0 : src_context->num_inputs() - 1;
    // Concat is concatenating its input shape vectors.
    for (int i = 0; i < src_context->num_inputs(); ++i) {
      // Concat dim is ignored (and will always be a scalar).
      if (i == concat_dim) continue;
      ShapeHandle sub_result;
      TF_RETURN_IF_ERROR(ConstantPartialShape(target_context, input_edge->src(),
                                              i, &sub_result));
      if (!target_context->RankKnown(sub_result)) {
        // Failed to evaluate. Treat the output as completely unknown.
        // TODO(cwhipkey): we could rely on all inputs being the same rank, so
        // figure that rank out and append the right number of unknown dims.
        *result = target_context->UnknownShape();
        return Status::OK();
      }
      TF_RETURN_IF_ERROR(
          target_context->Concatenate(*result, sub_result, result));
    }
  } else {
    Tensor t;
    bool evaluated = false;
    TF_RETURN_IF_ERROR(
        EvaluateConstantTensorForEdge(node, dst_idx, &evaluated, &t));
    TF_RETURN_IF_ERROR(target_context->MakeShapeFromTensor(
        evaluated ? &t : nullptr, src_shape, result));
  }
  return Status::OK();
}

Status ShapeRefiner::RunShapeFn(const Node* node,
                                const OpRegistrationData* op_reg_data,
                                ExtendedInferenceContext* ec) {
  // This will be filled in with real data in a second pass.
  std::vector<const Tensor*> input_tensors(node->num_inputs(), nullptr);
  std::vector<Tensor> real_tensors(node->num_inputs());
  std::vector<bool> attempted_materialization(node->num_inputs());
  std::vector<bool> attempted_tensor_as_shape_conversion(node->num_inputs());
  std::vector<ShapeHandle> input_tensors_as_shapes;

  auto* c = ec->get_context();

  c->set_input_tensors(input_tensors);
  c->set_input_tensors_as_shapes(input_tensors_as_shapes);

  // Run the shape inference function, and return if there was an error.
  // Capture as lambda, because we might need to re-run inference later on.
  auto run_inference_lambda = [&]() {
    if (function_library_ && op_reg_data->is_function_op) {
      // Special inference logic for user-defined functions.

      auto* func_def = function_library_->Find(op_reg_data->op_def.name());
      if (func_def) {
        return InferShapesForFunction(func_def, keep_nested_shape_inferences_,
                                      ec);
      }
    }

    if (op_reg_data->shape_inference_fn) {
      TF_RETURN_IF_ERROR(c->Run(op_reg_data->shape_inference_fn));
    } else {
      TF_RETURN_IF_ERROR(c->Run(shape_inference::UnknownShape));
    }
    return Status::OK();
  };
  TF_RETURN_IF_ERROR(run_inference_lambda());

  // We must run the shape function repeatedly, in case users write
  // shape functions where they only conditionally call input_tensor()
  // based on the values of another input tensor.
  bool rerun_shape_fn;
  do {
    // If the result of running shape inference would have benefitted
    // from knowing the values of input tensors, try to materialize
    // the results of those tensors, and then run the shape inference
    // function again using those known tensors.
    rerun_shape_fn = false;

    // NOTE: It is possible to batch the extraction and
    // materialization of inputs, instead of materializing one input
    // at a time like we do below.  If input-at-a-time computation
    // becomes a bottleneck, we could separate ExtractConstantSubgraph
    // into two functions: one that returns true if an input is
    // derivable from constants, and another function that extracts
    // the subgraph for multiple target nodes and executes the whole
    // subgraph once.

    for (int i = 0; i < c->num_inputs(); ++i) {
      if (!c->requested_input_tensor(i)) {
        continue;
      }
      // Check if we have not already filled in the requested input,
      // and if not, try to materialize the tensors.
      if (!attempted_materialization[i]) {
        attempted_materialization[i] = true;

        Tensor result;
        bool evaluated = false;
        TF_RETURN_IF_ERROR(
            EvaluateConstantTensorForEdge(node, i, &evaluated, &result));
        if (evaluated) {
          real_tensors[i] = result;
          input_tensors[i] = &real_tensors[i];
          // We have more concrete information about a shape,
          // so re-run shape inference.
          rerun_shape_fn = true;
        }
      }
      if (c->requested_input_tensor_as_partial_shape(i) &&
          !attempted_tensor_as_shape_conversion[i]) {
        attempted_tensor_as_shape_conversion[i] = true;
        if (i >= input_tensors_as_shapes.size()) {
          input_tensors_as_shapes.resize(i + 1);
        }
        ShapeHandle s;
        TF_RETURN_IF_ERROR(ConstantPartialShape(c, node, i, &s));
        input_tensors_as_shapes[i] = s;
        rerun_shape_fn = true;
      }
    }

    if (rerun_shape_fn) {
      // We have more information about the shapes on this pass,
      // so re-run shape inference.
      c->set_input_tensors(input_tensors);
      c->set_input_tensors_as_shapes(input_tensors_as_shapes);
      TF_RETURN_IF_ERROR(run_inference_lambda());
    }
  } while (rerun_shape_fn);

  return Status::OK();
}

bool ShapeRefiner::SameDefinedShape(InferenceContext* c, ShapeHandle s0,
                                    ShapeHandle s1) {
  if (!c->RankKnown(s0)) {
    return !c->RankKnown(s1);
  } else if (!c->RankKnown(s1) || c->Rank(s0) != c->Rank(s1)) {
    return false;
  }

  for (int i = 0; i < c->Rank(s0); ++i) {
    if (c->Value(c->Dim(s0, i)) != c->Value(c->Dim(s1, i))) {
      return false;
    }
  }

  return true;
}

bool ShapeRefiner::IsUpdatedShapesOrTypes(
    InferenceContext* c, const std::vector<ShapeAndType>& existing,
    const std::vector<ShapeAndType>& updated) {
  if (existing.size() != updated.size()) {
    return true;
  }
  for (int i = 0; i < existing.size(); i++) {
    if (!SameDefinedShape(c, existing[i].shape, updated[i].shape) ||
        existing[i].dtype != updated[i].dtype) {
      return true;
    }
  }
  return false;
}

}  // namespace tensorflow
