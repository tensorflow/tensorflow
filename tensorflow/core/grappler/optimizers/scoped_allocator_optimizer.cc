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
#include "tensorflow/core/grappler/optimizers/scoped_allocator_optimizer.h"

#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

// Like TF_RETURN_IF_ERROR, but also logs a WARNING.
#define LOG_WARNING_AND_RETURN_IF_ERROR(...)            \
  do {                                                  \
    const ::tensorflow::Status _status = (__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_status.ok())) {              \
      LOG(WARNING) << "error: " << _status;             \
      return _status;                                   \
    }                                                   \
  } while (0)

namespace tensorflow {
namespace grappler {

namespace {

const char kScopedAllocatorAttrName[] = "_scoped_allocator";

// Node names often have some kind of name_scope prefix, with slashes,
// and a _nn numeric suffix.  Returns true if the main part of the node_name
// matches op_name, i.e. it looks from the name like this node is
// of that op type.
bool HasOpName(const string& node_name, const string& op_name) {
  size_t begin = node_name.rfind("/");
  if (begin == string::npos) {
    begin = 0;
  } else {
    ++begin;
  }
  size_t end = node_name.rfind("_");
  if (end != string::npos) {
    size_t p = end + 1;
    while (p < node_name.size()) {
      if (!isdigit(node_name[p])) {
        end = node_name.size();
        break;
      }
      ++p;
    }
  } else {
    end = node_name.size();
  }
  return node_name.substr(begin, end - begin) == op_name;
}

Status GetOutputDataType(
    const std::vector<OpInfo::TensorProperties>& output_props, int output_index,
    DataType* dtype) {
  int output_props_size = output_props.size();
  if (output_index >= output_props_size) {
    return errors::Internal("Invalid output index ", output_index,
                            " size of output_props ", output_props.size());
  }
  *dtype = output_props[output_index].dtype();
  return Status::OK();
}

// After shape inference has been done each op should be annotated
// with its output shape(s).  This function iterates over a collection
// of ops that are a potential application of a ScopedAllocator.  It
// verifies whether they all have the same output type and if so
// gathers a vector of their output shapes.  It returns an error if
// any of the ops doesn't have type or shape data, or if it has more
// than one output, of if the output type of all ops is not the same.
// If it returns OK then *type and *shapes should be correctly populated.
Status CheckTypesAndGetShapes(const GraphProperties& graph_properties,
                              const std::vector<NodeDef*>& ops, DataType* type,
                              std::vector<TensorShape>* shapes) {
  VLOG(1) << "CheckTypesAndGetShapes";
  *type = DT_INVALID;
  for (NodeDef* n : ops) {
    AttrSlice n_attrs = AttrSlice(*n);
    DataType dtype;
    // Check that op has an explicit data type attr "T".
    LOG_WARNING_AND_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "T", &dtype));
    VLOG(2) << "op " << n->name() << " has type " << dtype << " shapes.size() "
            << shapes->size();
    if (!graph_properties.HasOutputProperties(n->name())) {
      LOG(ERROR) << "Node " << n->DebugString() << " lacks output shape.";
      return errors::Internal("Node ", n->name(), " lacks output shape.");
    }
    const std::vector<OpInfo::TensorProperties>& prop_list =
        graph_properties.GetOutputProperties(n->name());
    if (prop_list.size() != 1) {
      return errors::Internal("Node ", n->name(),
                              " does not have exactly one output as expected "
                              "by ScopedAllocatorOptimizer");
    }
    const OpInfo::TensorProperties& props = prop_list[0];
    if (shapes->empty()) {
      *type = props.dtype();
    } else if (*type != props.dtype()) {
      return errors::Internal("Group ops don't all have same type");
    } else if (!TensorShape::IsValid(props.shape()) ||
               props.shape().unknown_rank()) {
      // TensorShape::IsValid may return true if unknown_rank is True, i.e.
      // number of dimensions is unknown.  But for ScopedAllocatorOptimizer we
      // need to know the shape fully.
      return errors::Internal("Complete shape not known for ", n->name());
    }
    if (*type != dtype) {
      return errors::Internal(
          "Type mismatch: type in op attr = ", DataTypeString(dtype),
          ", type in output props = ", DataTypeString(*type));
    }
    VLOG(2) << "Adding shape " << props.shape().DebugString();
    shapes->push_back(TensorShape(props.shape()));
  }
  return Status::OK();
}

// Describes an existing input edge in the graph.
struct InputDesc {
  NodeDef* from_node_def;
  int output_slot;
  NodeDef* to_node_def;
  InputDesc(NodeDef* f, int os, NodeDef* t)
      : from_node_def(f), output_slot(os), to_node_def(t) {}
};

// Remove the NodeDef nd from node_map and graph.  It must be the case
// that nd no longer has any input or output edges, though that is not
// checked.
void RemoveNode(NodeDef* nd, GraphDef* graph, NodeMap* node_map) {
  node_map->RemoveNode(nd->name());
  // TODO(tucker): The efficiency of this routine is poor.
  // Change to accumulate and do a bulk removal, maybe refactoring
  // some code from dependency_optimizer.
  protobuf::RepeatedPtrField<NodeDef>* nodes = graph->mutable_node();
  for (int i = 0; i < nodes->size(); ++i) {
    if (nd->name() == (*nodes)[i].name()) {
      nodes->SwapElements(i, nodes->size() - 1);
      nodes->RemoveLast();
      return;
    }
  }
  LOG(FATAL) << "Failed to find node " << nd->name() << " in graph";
}

// Removes a named edge from between two nodes.
Status RemoveEdge(const string& input_edge_name, const string& from_node_name,
                  NodeDef* to_node, NodeMap* node_map) {
  protobuf::RepeatedPtrField<string>* inputs = to_node->mutable_input();
  int edge_index = -1;
  for (edge_index = 0; edge_index < inputs->size(); ++edge_index) {
    VLOG(2) << " consider edge " << (*inputs)[edge_index];
    if ((*inputs)[edge_index] == input_edge_name) {
      break;
    }
  }
  if (edge_index >= inputs->size()) {
    return errors::Internal("Could not find input name ", input_edge_name,
                            " at node ", to_node->name());
  }
  if (node_map) {
    node_map->RemoveOutput(from_node_name, to_node->name());
  }
  inputs->DeleteSubrange(edge_index, 1);
  return Status::OK();
}

// In certain cases, we would like to insert an identity op between `input` and
// `op` to ensure correctness.  We currently do this in 2 cases: when `input` is
// Exit node, or when `input` is already marked for allocation with another
// scoped allocator op.
//
// If `input` is an Exit node, we add an identity to avoid the case when Exit
// has inputs from different frames.
//
// If `input` is in `sa_opti->repeated_outputs()`, this means that it will be
// potentially used by multiple scope ids.  Since there can be only one scope id
// per output, we insert an identity between the input and op.  This will ensure
// that the identity becomes the new input to op, and this identity can be
// marked with a new scope id different from `input`.
//
// If the graph is rewritten, this function will perform the following change:
//
//  input                                  input
//   |                                      |
//   op                                  Identity
//                                          |
//                                          op
//
// This function returns the input to op in `new_input`, and the output index
// from input to op in `new_output_index`.
// `edge_name` gives the name of the edge from `input` to `op`, and
// `output_index` is the output index of this edge on `input`.
Status MaybeRewriteInput(ScopedAllocatorOptimizer* sa_opti,
                         int64 invocation_count, GraphDef* graph,
                         NodeMap* node_map, const DataType& dtype,
                         NodeDef* input, const string& edge_name,
                         int output_index, NodeDef* op, NodeDef** new_input,
                         int* new_output_index, bool* rewrite) {
  *rewrite = IsExit(*input) || (sa_opti->repeated_outputs().find(edge_name) !=
                                sa_opti->repeated_outputs().end());
  if (!(*rewrite)) {
    *new_input = input;
    *new_output_index = output_index;
    return Status::OK();
  }

  // Create new Identity op.
  int unique_id;
  LOG_WARNING_AND_RETURN_IF_ERROR(sa_opti->NewIdentityId(&unique_id));
  string identity_name = strings::StrCat("scoped_allocator_identity_",
                                         unique_id, "_", invocation_count);
  NodeDefBuilder identity_builder(identity_name, "Identity");
  identity_builder.Device(op->device());
  identity_builder.Attr("T", dtype);
  // Connect output at `output_index` from `input` to `identity`.
  identity_builder.Input(
      NodeDefBuilder::NodeOut(input->name(), output_index, dtype));
  NodeDef* identity = graph->add_node();
  LOG_WARNING_AND_RETURN_IF_ERROR(identity_builder.Finalize(identity));
  node_map->AddNode(identity_name, identity);
  node_map->AddOutput(input->name(), identity_name);
  node_map->UpdateInput(op->name(), input->name(), identity_name);
  *op->mutable_input(0) = identity_name;
  *new_input = identity;
  *new_output_index = 0;
  VLOG(1) << "Rewrite input " << edge_name << " op " << op->name()
          << " old output index " << output_index << " with identity "
          << identity_name << " new output index 0";
  return Status::OK();
}

// Populates *inputs with all of the non-control inputs of ops.
// Returns error if it fails to find exactly one input for each op,
// or if some input is not of type dtype.
Status GetInputs(ScopedAllocatorOptimizer* sa_opti, int64 invocation_count,
                 GraphDef* graph, const GraphProperties& graph_properties,
                 NodeMap* node_map, const std::vector<NodeDef*>& ops,
                 DataType dtype, std::vector<InputDesc>* inputs) {
  VLOG(1) << "Getinputs";
  for (NodeDef* n : ops) {
    NodeDef* inode = nullptr;
    int output_index = 0;
    DataType inode_dtype = DT_INVALID;
    VLOG(2) << "for node " << n->name();
    for (const auto& input_name : n->input()) {
      if (!IsControlInput(input_name)) {
        if (inode) {
          return errors::Internal("Found more than one input for node ",
                                  n->name());
        }
        ParseNodeName(input_name, &output_index);
        inode = node_map->GetNode(input_name);
        if (inode == nullptr) {
          return errors::Internal("Did not find node ", input_name);
        }
        VLOG(2) << "inode " << inode->DebugString() << " output_index "
                << output_index;
        bool rewrite;
        LOG_WARNING_AND_RETURN_IF_ERROR(MaybeRewriteInput(
            sa_opti, invocation_count, graph, node_map, dtype, inode,
            input_name, output_index, n, &inode, &output_index, &rewrite));
        // If `inode` was rewritten, don't try to get output properties from the
        // input node below.
        if (rewrite) {
          inode_dtype = dtype;
        }
        VLOG(2) << "inode after rewrite " << inode->DebugString()
                << " output_index " << output_index;
      }
    }
    if (inode_dtype == DT_INVALID) {
      if (!graph_properties.HasOutputProperties(inode->name())) {
        return errors::Internal("Input node ", inode->name(),
                                " does not have output properties");
      }
      const auto& inode_output_props =
          graph_properties.GetOutputProperties(inode->name());
      LOG_WARNING_AND_RETURN_IF_ERROR(
          GetOutputDataType(inode_output_props, output_index, &inode_dtype));
    }
    if (inode_dtype != dtype) {
      return errors::Internal("ScopedAllocatorOptimizer expected input type ",
                              dtype, " but found ", inode_dtype);
    }
    inputs->emplace_back(inode, output_index, n);
  }
  return Status::OK();
}

// Return non-control inputs of `op` in `inputs`.
Status GetDataInputs(GraphDef* graph, NodeMap* node_map, NodeDef* op,
                     std::vector<InputDesc>* inputs) {
  VLOG(2) << "GetDataInputs for node " << op->name();
  NodeDef* inode = nullptr;
  int output_index = 0;
  for (const auto& input_name : op->input()) {
    if (IsControlInput(input_name)) {
      continue;
    }
    ParseNodeName(input_name, &output_index);
    inode = nullptr;
    inode = node_map->GetNode(input_name);
    if (inode == nullptr) {
      return errors::Internal("Did not find node ", input_name);
    }
    VLOG(2) << "inode " << inode->DebugString() << " output_index "
            << output_index;
    inputs->emplace_back(inode, output_index, op);
  }
  return Status::OK();
}

void DumpGraphToVLOG(const GraphDef& graph, int log_level) {
  if (VLOG_IS_ON(log_level)) {
    // VLOG may truncate lines so we print line by line.
    for (const auto& line : str_util::Split(graph.DebugString(), "\n\r")) {
      VLOG(log_level) << line;
    }
  }
}

}  // namespace

void ScopedAllocatorOptimizer::ExtendNodeAttr(StringPiece name,
                                              const std::vector<int32>& values,
                                              NodeDef* node_def) {
  if (HasNodeAttr(*node_def, name)) {
    VLOG(2) << "extending";
    AttrValue* existing = &(*node_def->mutable_attr())[string(name)];
    for (int32 i : values) {
      existing->mutable_list()->add_i(i);
    }
  } else {
    VLOG(2) << "setting new attr value";
    AddNodeAttr(name, values, node_def);
  }
}

class UnaryElementwiseRewriter : public ScopedAllocatorOptimizer::Rewriter {
 public:
  ~UnaryElementwiseRewriter() override {}

  // Return non-OK if any input is already committed to a ScopedAllocator.
  //
  // We insert an identity to ensure that inputs are not committed to different
  // scope ids in `MaybeRewriteInput`, so this function is basically a sanity
  // check.
  Status CheckExistingScopedAllocator(const std::vector<InputDesc>& inputs) {
    for (const InputDesc& nd : inputs) {
      VLOG(2) << "get attrs for " << nd.from_node_def->name();
      AttrSlice n_attrs = AttrSlice(*nd.from_node_def);
      std::vector<int32> scope_ids;
      Status ss = GetNodeAttr(n_attrs, kScopedAllocatorAttrName, &scope_ids);
      // Check that both output name and output slot match.  It is okay to have
      // different outputs of the input committed to different scope ids.
      if (ss.ok() && scope_ids[0] == nd.output_slot) {
        LOG(INFO) << "Abandoning ScopedAllocatorOptimizer because input "
                  << nd.from_node_def->name() << " output " << scope_ids[0]
                  << " is already assigned to scope_id " << scope_ids[1];
        return errors::Internal(
            "Abandoning ScopedAllocatorOptimizer because input ",
            nd.from_node_def->name(), " output ", scope_ids[0], " is already ",
            "assigned to scope_id ", scope_ids[1]);
      }
    }
    return Status::OK();
  }

  // Return non-OK if any input is a member of op_set.
  Status CheckInternalDataDependency(const std::set<string>& op_set,
                                     const std::vector<InputDesc>& inputs) {
    for (const InputDesc& nd : inputs) {
      if (op_set.find(nd.from_node_def->name()) != op_set.end()) {
        if (nd.output_slot != tensorflow::Graph::kControlSlot) {
          return errors::Internal("Data edge exists between ",
                                  nd.from_node_def->name(),
                                  " and another "
                                  "node in the set");
        }
      }
    }
    return Status::OK();
  }

  // Remove all control edges between members of ops.
  void ClearInternalControlInputs(const std::set<string>& op_set,
                                  const std::vector<NodeDef*>& ops,
                                  NodeMap* node_map) {
    for (NodeDef* n : ops) {
      for (const auto& input_name : n->input()) {
        if (IsControlInput(input_name)) {
          int position = 0;
          string input_node_name = ParseNodeName(input_name, &position);
          CHECK_EQ(position, -1);
          if (op_set.find(input_node_name) != op_set.end()) {
            // This is an internal control edge.  Remove it.
            VLOG(1) << "Remove control output from " << input_node_name
                    << " via edge " << input_name << " to " << n->name();
            TF_CHECK_OK(RemoveEdge(input_name, input_node_name, n, node_map));
          }
        }
      }
    }
  }

  // Examine the input set of an op set, gathering their shapes and types
  // and checking whether there are any considerations that prevent use
  // of a single ScopedAllocator for all of those inputs.
  Status AnalyzeInputs(ScopedAllocatorOptimizer* sa_opti,
                       int64 invocation_count, GraphDef* graph,
                       NodeMap* node_map, const std::vector<NodeDef*>& ops,
                       const std::set<string>& op_instance_names,
                       string* device_name, DataType* dtype,
                       std::vector<TensorShape>* input_shapes,
                       std::vector<InputDesc>* inputs, TensorShape* sa_shape) {
    CHECK(graph_properties_);
    LOG_WARNING_AND_RETURN_IF_ERROR(
        CheckTypesAndGetShapes(*graph_properties_, ops, dtype, input_shapes));
    LOG_WARNING_AND_RETURN_IF_ERROR(
        GetInputs(sa_opti, invocation_count, graph, *graph_properties_,
                  sa_opti->node_map(), ops, *dtype, inputs));
    LOG_WARNING_AND_RETURN_IF_ERROR(CheckExistingScopedAllocator(*inputs));
    LOG_WARNING_AND_RETURN_IF_ERROR(
        CheckInternalDataDependency(op_instance_names, *inputs));
    ClearInternalControlInputs(op_instance_names, ops, node_map);
    *device_name = ops[0]->device();
    CHECK(!device_name->empty());
    CHECK(!input_shapes->empty());
    CHECK_EQ(0, Allocator::kAllocatorAlignment % DataTypeSize(*dtype))
        << "ScopedAllocatorOptimizer only applies to types that evenly "
        << "divide kAllocatorAlignment";
    std::vector<ScopedAllocator::Field> sa_fields;
    // Calculate the field embedding boundaries and thereby the
    // required size of the backing tensor.
    int64 num_bytes = ScopedAllocatorMgr::PopulateFields(
        0 /*scope_id*/, *input_shapes, *dtype, &sa_fields);
    int64 num_elts = num_bytes / DataTypeSize(*dtype);
    VLOG(2) << "num_bytes " << num_bytes << " num_elts=" << num_elts;
    *sa_shape = TensorShape({num_elts});
    return Status::OK();
  }

  // Returns the set of all nodes that are transitively reachable via data or
  // control edges starting at `source_nodes`.  Stop at the boundary of a frame.
  Status TransitiveFanoutWithinFrame(
      GraphDef* graph, NodeMap* node_map,
      const std::vector<const NodeDef*>& source_nodes,
      absl::flat_hash_set<const NodeDef*>* fanout) {
    std::deque<const NodeDef*> queue(source_nodes.begin(), source_nodes.end());
    absl::flat_hash_set<const NodeDef*> visited;
    while (!queue.empty()) {
      const NodeDef* node = queue.front();
      queue.pop_front();
      if (!visited.insert(node).second) {
        continue;
      }
      fanout->insert(node);
      for (const NodeDef* output : node_map->GetOutputs(node->name())) {
        if (!ModifiesFrameInfo(*output)) {
          queue.push_back(output);
        }
        VLOG(2) << "TransitiveFanout parent: " << node->name()
                << " child: " << output->name() << " of type " << output->op();
      }
    }

    return Status::OK();
  }

  // Build the ScopedAllocator node that will be assigned to allocate
  // the output tensors of the input node set.
  Status ConstructScopedAllocatorNode(
      ScopedAllocatorOptimizer* sa_opti, GraphDef* graph, NodeMap* node_map,
      const std::vector<NodeDef*>& ops, const string& device_name,
      DataType dtype, int sa_id, const string& sa_name,
      const std::vector<TensorShape>& input_shapes,
      const std::vector<InputDesc>& inputs, const TensorShape& sa_shape) {
    VLOG(2) << "ConstructScopedAllocatorNode " << sa_name;
    NodeDefBuilder sa_builder(sa_name, "_ScopedAllocator");
    sa_builder.Device(device_name);
    sa_builder.Attr("sa_name", sa_name);
    sa_builder.Attr("T", dtype);
    sa_builder.Attr("id", sa_id);
    sa_builder.Attr("shapes", input_shapes);
    sa_builder.Attr("shape", sa_shape);
    sa_builder.Attr("expected_call_count", static_cast<int64>(ops.size()));
    NodeDef* sa_node = graph->add_node();
    LOG_WARNING_AND_RETURN_IF_ERROR(sa_builder.Finalize(sa_node));
    node_map->AddNode(sa_name, sa_node);

    std::vector<const NodeDef*> fanout_sources;
    fanout_sources.reserve(inputs.size());
    for (const auto& input : inputs) {
      fanout_sources.push_back(input.from_node_def);
    }
    absl::flat_hash_set<const NodeDef*> fanout;
    TF_RETURN_IF_ERROR(
        TransitiveFanoutWithinFrame(graph, node_map, fanout_sources, &fanout));

    // Add control edges from the ScopedAllocatorOp to all of the
    // input nodes and mark them for allocation from backing tensor.
    for (int i = 0, iter_limit = inputs.size(); i < iter_limit; ++i) {
      auto& nd = inputs[i];
      if (IsArg(*nd.from_node_def)) {
        return errors::Internal(
            "ScopedAllocatorOptimizer does not work well when the op inputs "
            "are _Arg ops; skipping this optimizer for this function");
      }
      VLOG(2) << "To input " << i << ": " << nd.from_node_def->name()
              << " add control input "
              << "^" << sa_name;
      nd.from_node_def->add_input(strings::StrCat("^", sa_name));
      // This attribute says: allocate output_slot from
      // ScopedAllocator instance sa_id + 1 + i.
      ScopedAllocatorOptimizer::ExtendNodeAttr(kScopedAllocatorAttrName,
                                               {nd.output_slot, sa_id + 1 + i},
                                               nd.from_node_def);
      node_map->AddOutput(sa_name, nd.from_node_def->name());
    }

    // We add control edges in order to (1) delay execution of the
    // ScopedAllocatorOp until just before first use in order to conserve memory
    // (2) ensure correctness in the presence of control flow related ops.
    bool added_delay_edge = false;
    for (auto& nd : inputs) {
      std::vector<InputDesc> inputs_to_first;
      LOG_WARNING_AND_RETURN_IF_ERROR(GetDataInputs(
          graph, sa_opti->node_map(), nd.from_node_def, &inputs_to_first));
      for (int i = 0, iter_limit = inputs_to_first.size(); i < iter_limit;
           ++i) {
        if (fanout.find(inputs_to_first[i].from_node_def) != fanout.end()) {
          VLOG(2) << "Found node " << inputs_to_first[i].from_node_def->name()
                  << " in the fanout of " << sa_name;
          continue;
        }
        sa_node->add_input(
            strings::StrCat("^", inputs_to_first[i].from_node_def->name()));
        node_map->AddOutput(inputs_to_first[i].from_node_def->name(), sa_name);
        added_delay_edge = true;
        VLOG(2) << "Adding control dependency from "
                << inputs_to_first[i].from_node_def->name() << " to "
                << sa_node->name();
        break;
      }
      if (added_delay_edge) {
        break;
      }
    }

    if (!added_delay_edge) {
      LOG(WARNING) << "Found no node from which a control edge can be added to "
                      "scoped allocator node.  If you run into issues with "
                      "graphs that contain control flow, turn off the "
                      "ScopedAllocatorOptimizer and file a bug.";
    }

    return Status::OK();
  }

  Status BuildSAConcatNode(GraphDef* graph, NodeMap* node_map,
                           const std::vector<NodeDef*>& ops,
                           const std::set<string>& op_instance_names,
                           const string& device_name, DataType dtype, int sa_id,
                           const string& sa_name, const string& sac_name,
                           const TensorShape& sa_shape,
                           std::vector<NodeDefBuilder::NodeOut>* sac_inputs) {
    VLOG(2) << "BuildSAConcatNode " << sac_name;
    // control input: edge name -> source node name
    absl::flat_hash_map<string, string> sac_ctl_inputs;
    for (int i = 0, iter_limit = ops.size(); i < iter_limit; ++i) {
      NodeDef* old_op = ops[i];
      for (const string& old_op_input : old_op->input()) {
        int position = 0;
        string input_name = ParseNodeName(old_op_input, &position);
        if (position == -1) {
          // A control input: drop if from another member of the op set.
          if (op_instance_names.find(old_op_input) == op_instance_names.end()) {
            sac_ctl_inputs.emplace(old_op_input, input_name);
          }
        } else {
          // TODO(tucker): remove redundant check.
          // A data input: illegal if from another member of the op set.
          if (op_instance_names.find(old_op_input) != op_instance_names.end()) {
            LOG(ERROR) << "Data edge between " << old_op_input << " and "
                       << old_op->name() << " cannot build ScopedAllocator.";
            return errors::Internal("Data edge between ", old_op_input, " and ",
                                    old_op->name(),
                                    " cannot build ScopedAllocator.");
          }
          sac_inputs->push_back(
              NodeDefBuilder::NodeOut(old_op_input, 0, dtype));
        }
        VLOG(3) << "from op " << i << ": " << old_op->name()
                << " sac_inputs append " << old_op_input;
      }
    }
    NodeDefBuilder sac_builder(sac_name, "_ScopedAllocatorConcat");
    VLOG(2) << "New sac_name " << sac_name << " shape "
            << sa_shape.DebugString();
    sac_builder.Device(device_name);
    sac_builder.Attr("sa_name", sa_name);
    sac_builder.Attr("id", sa_id);
    sac_builder.Attr("T", dtype);
    sac_builder.Attr("shape", sa_shape);
    sac_builder.Attr("N", static_cast<int>(sac_inputs->size()));
    sac_builder.Input(NodeDefBuilder::NodeOut(sa_name, 0, dtype));
    sac_builder.Input(*sac_inputs);
    NodeDef* sac_node = graph->add_node();
    LOG_WARNING_AND_RETURN_IF_ERROR(sac_builder.Finalize(sac_node));
    node_map->AddNode(sac_name, sac_node);
    node_map->AddOutput(sa_name, sac_name);

    // Attach the old control inputs to the new sac node.
    for (const auto& ctl_input : sac_ctl_inputs) {
      const auto& ctl_edge = ctl_input.first;
      const auto& input_name = ctl_input.second;
      sac_node->add_input(ctl_edge);
      node_map->AddOutput(input_name, sac_node->name());
    }
    return Status::OK();
  }

  Status BuildReplacementOp(GraphDef* graph, NodeMap* node_map,
                            const std::vector<NodeDef*>& ops,
                            const string& device_name, DataType dtype,
                            const string& op_name, const string& sac_name,
                            const string& sa_op_name) {
    VLOG(2) << "BuildReplacementOp " << sa_op_name;
    NodeDefBuilder op_builder(sa_op_name, op_name);
    op_builder.Device(device_name);

    // Transfer the Node Attr from the first replaced Node to the new
    // Node.  TODO(tucker): In principle we should verify that
    // the Attr are consistent and compatible across all op instances.
    // Unfortunately that will probably require op-specific tests, so
    // punt on that for the time being.
    AttrSlice first_slice(*ops[0]);
    for (auto& it : first_slice) {
      op_builder.Attr(it.first, it.second);
    }
    op_builder.Attr("_forward_input", {0, 0});
    op_builder.Input(sac_name, 0, dtype);
    NodeDef* sa_op_node = graph->add_node();
    LOG_WARNING_AND_RETURN_IF_ERROR(op_builder.Finalize(sa_op_node));
    node_map->AddNode(sa_op_name, sa_op_node);
    node_map->AddOutput(sac_name, sa_op_name);
    return Status::OK();
  }

  Status BuildSplitNode(GraphDef* graph, NodeMap* node_map,
                        const std::vector<NodeDef*>& ops,
                        const std::vector<TensorShape>& input_shapes,
                        const std::vector<NodeDefBuilder::NodeOut>& sac_inputs,
                        const string& device_name, DataType dtype,
                        const string& op_name, int sa_id,
                        const string& sas_name, const string& sa_name,
                        const string& sa_op_name) {
    VLOG(2) << "new ScopedAllocatorSplit " << sas_name;
    NodeDefBuilder sas_builder(sas_name, "_ScopedAllocatorSplit");
    sas_builder.Device(device_name);
    sas_builder.Attr("sa_name", sa_name);
    sas_builder.Attr("id", sa_id);
    sas_builder.Attr("T", dtype);
    sas_builder.Attr("shapes", input_shapes);
    std::vector<NodeDefBuilder::NodeOut> sas_inputs = sac_inputs;
    sas_builder.Attr("N", static_cast<int>(sas_inputs.size()));
    sas_builder.Input(NodeDefBuilder::NodeOut(sa_op_name, 0, dtype));
    sas_builder.Input(sas_inputs);
    NodeDef* sas_node = graph->add_node();
    LOG_WARNING_AND_RETURN_IF_ERROR(sas_builder.Finalize(sas_node));
    node_map->AddNode(sas_name, sas_node);
    node_map->AddOutput(sa_op_name, sas_name);
    for (const auto& input : sas_inputs) {
      node_map->AddOutput(input.node, sas_name);
    }
    return Status::OK();
  }

  // After the new ScopedAllocator and its corresponding Concat and
  // Split nodes have been built, and a new single Op instance
  // constructed, rewire the graph: Remove input edges to the old Op
  // nodes and replace the old Op node outputs with the corresponding
  // ScopedAllocatorSplit node outputs.  After this the old Op nodes
  // should no longer have any input or output edges and they can be
  // removed from the graph.
  Status RewireSubgraph(GraphDef* graph, NodeMap* node_map,
                        const std::vector<NodeDef*>& ops,
                        const std::set<string>& op_instance_names,
                        const string& op_name, const string& sas_name) {
    VLOG(2) << "RewireSubgraph";
    for (int op_idx = 0, idx_limit = ops.size(); op_idx < idx_limit; ++op_idx) {
      NodeDef* old_op = ops[op_idx];
      // Copy the output node set since we'll be modifying the version
      // maintained by NodeMap in the loop.
      auto output_nodes = node_map->GetOutputs(old_op->name());
      VLOG(3) << "old_op " << old_op->name() << " had " << output_nodes.size()
              << " outputs.  Moving them to the ScopedAllocatorSplit node.";
      if (VLOG_IS_ON(2)) {
        for (NodeDef* n : output_nodes) {
          VLOG(3) << "    output: " << n->name();
        }
      }
      for (NodeDef* n : output_nodes) {
        VLOG(3) << "really checking old output " << n->name()
                << " for corresponding input.";
        if (op_instance_names.find(n->name()) != op_instance_names.end()) {
          // If this output node is a member of the ops set, it must have
          // been an internal control edge so drop it.
          VLOG(3) << "Dropping control output from " << old_op->name() << " to "
                  << n->name();
          // However, we may already have dropped it at the clear() below,
          // so if we fail to find it, that's okay.
          Status ignore = RemoveEdge(strings::StrCat("^", old_op->name()),
                                     old_op->name(), n, node_map);
          continue;
        }
        bool found = false;
        VLOG(3) << "about to iterate over " << n->input_size() << " inputs";
        for (int i = 0; i < n->input_size(); ++i) {
          VLOG(3) << "input " << n->input(i);
          int position = 0;
          string input_node = ParseNodeName(n->input(i), &position);
          if (input_node == old_op->name()) {
            found = true;
            VLOG(3) << "match pos=" << position;
            if (position == -1) {
              // It was a control edge
              *n->mutable_input(i) = strings::StrCat("^", sas_name);
            } else {
              CHECK_EQ(0, position)
                  << "name " << n->input(i) << " pos " << position;
              *n->mutable_input(i) = strings::StrCat(sas_name, ":", op_idx);
            }
            node_map->UpdateInput(n->name(), old_op->name(), sas_name);
            VLOG(3) << "breaking on success";
            break;
          } else {
            VLOG(3) << "other input " << n->input(i);
          }
        }
        // In general it's required that we found the output node's old
        // input and replaced it, but one exception is if the output node
        // is of the same type being coalesced and the edge is a control
        // input.  In that case it probably got eliminated in an earlier
        // pass.
        VLOG(3) << "before HasOp";
        if (!HasOpName(n->name(), op_name)) {
          CHECK(found) << "old_op " << old_op->name() << " node "
                       << " could not find input edge on " << n->DebugString()
                       << " to replace."
                       << " " << op_name << " not in " << n->name();
        }
        VLOG(3) << "bottom of for output_nodes";
      }
      VLOG(3) << "Clearing all inputs of " << old_op->name();
      node_map->RemoveInputs(old_op->name());
      old_op->clear_input();
      node_map->RemoveOutputs(old_op->name());
      VLOG(3) << "after clear: " << old_op->DebugString();
      // old_op should be dead, with no further inputs or outputs.
      // It needs to be removed altogether before the graph is generated,
      // but we need to leave it around until this Optimizer is done,
      // because there may be some
      // Remove.
      RemoveNode(old_op, graph, node_map);
    }
    return Status::OK();
  }

  // Given a collection of instances of op_name, presumed to be
  // logically parallel and operating on tensors of the same type,
  // replace them by a single instance.  First find the upstream Ops
  // generating their inputs. Create a new ScopedAllocatorOp that
  // outputs a single backing_tensor pre-arranged for sub-allocation
  // of all of those input tensors.  Then insert a new
  // ScopedAllocatorConcatOp below the upstream Ops to make explicit
  // the materialization of a concatenation of their outputs.  Put the
  // new op_name instance below the new concat op and follow with a
  // ScopedAllocatorSplitOp that restores the correct shape outputs
  // for the consumers of the old op_name instances.
  //
  // There must be no non-control edges between Nodes in 'ops'.
  // Control edges among these nodes will be dropped.
  Status Rewrite(ScopedAllocatorOptimizer* sa_opti, int64 invocation_count,
                 GraphDef* graph, const string& op_name,
                 const std::vector<NodeDef*>& ops, bool* applied) override {
    if (VLOG_IS_ON(1)) {
      VLOG(1) << "Rewrite";
      string op_names;
      for (auto& nd : ops) {
        strings::StrAppend(&op_names, nd->name(), ", ");
      }
      VLOG(1) << "UnaryElementwiseRewriter::Rewrite " << op_name
              << " to: " << op_names;
    }
    NodeMap* node_map = sa_opti->node_map();

    // Make a set of the node names for faster membership testing.
    std::set<string> op_instance_names;
    for (auto& nd : ops) {
      op_instance_names.insert(nd->name());
      VLOG(2) << "op_instance_name " << nd->name();
    }
    DataType dtype;
    std::vector<TensorShape> input_shapes;
    std::vector<InputDesc> inputs;
    TensorShape sa_shape;
    string device_name;

    TF_RETURN_IF_ERROR(AnalyzeInputs(
        sa_opti, invocation_count, graph, node_map, ops, op_instance_names,
        &device_name, &dtype, &input_shapes, &inputs, &sa_shape));

    int sa_id = sa_opti->NewScopedAllocatorId(input_shapes.size());
    string sa_name =
        strings::StrCat("scoped_allocator_", sa_id, "_", invocation_count);
    TF_RETURN_IF_ERROR(ConstructScopedAllocatorNode(
        sa_opti, graph, node_map, ops, device_name, dtype, sa_id, sa_name,
        input_shapes, inputs, sa_shape));

    // Build a ScopedAllocatorConcat below all of the input nodes.
    std::vector<NodeDefBuilder::NodeOut> sac_inputs;
    string sac_name = strings::StrCat("scoped_allocator_concat_", sa_id, "_",
                                      invocation_count);
    TF_RETURN_IF_ERROR(BuildSAConcatNode(
        graph, node_map, ops, op_instance_names, device_name, dtype, sa_id,
        sa_name, sac_name, sa_shape, &sac_inputs));

    // Construct a new instance of the parallel op and insert it
    // immediately below the new ScopedAllocatorConcat.
    string sa_op_name = strings::StrCat(sa_name, "_", op_name);
    TF_RETURN_IF_ERROR(BuildReplacementOp(graph, node_map, ops, device_name,
                                          dtype, op_name, sac_name,
                                          sa_op_name));

    // Build a ScopedAllocatorSplit split below the new Op.
    string sas_name = strings::StrCat("scoped_allocator_split_", sa_id, "_",
                                      invocation_count);
    TF_RETURN_IF_ERROR(BuildSplitNode(graph, node_map, ops, input_shapes,
                                      sac_inputs, device_name, dtype, op_name,
                                      sa_id, sas_name, sa_name, sa_op_name));

    // Rewire the graph.
    TF_RETURN_IF_ERROR(RewireSubgraph(graph, node_map, ops, op_instance_names,
                                      op_name, sas_name));

    *applied = true;
    return Status::OK();
  }
};

ScopedAllocatorOptimizer::ScopedAllocatorOptimizer(
    RewriterConfig::Toggle opt_level, const ScopedAllocatorOptions& opts)
    : opt_level_(opt_level) {
  VLOG(1) << "ScopedAllocatorOptimizer::ScopedAllocatorOptimizer";
  Rewriter* r = new UnaryElementwiseRewriter();
  to_delete_.push_back(r);
  if (opts.enable_op_size() == 0) {
    // Opts handled by default:
    for (const auto& op_name : {"CollectiveReduce"}) {
      op_name_set_.insert(op_name);
      rewriters_[op_name] = r;
    }
  } else {
    for (const auto& op_name : opts.enable_op()) {
      op_name_set_.insert(op_name);
      rewriters_[op_name] = r;
    }
  }
}

Status ScopedAllocatorOptimizer::Optimize(Cluster* /*cluster*/,
                                          const GrapplerItem& item,
                                          GraphDef* optimized_graph) {
  VLOG(3) << "Input graph:";
  DumpGraphToVLOG(item.graph, /*log_level=*/3);

  // Nodes that cannot be removed from the graph without damaging correctness,
  // typically fetch nodes.
  nodes_to_preserve_ = item.NodesToPreserve();

  GraphProperties graph_properties(item);
  const bool assume_valid_feeds = opt_level_ == RewriterConfig::AGGRESSIVE;
  LOG_WARNING_AND_RETURN_IF_ERROR(graph_properties.InferStatically(
      assume_valid_feeds, /*aggressive_shape_inference=*/false,
      /*include_tensor_values=*/false));
  *optimized_graph = item.graph;
  node_map_ = absl::make_unique<NodeMap>(optimized_graph);

  LOG_WARNING_AND_RETURN_IF_ERROR(ScopedAllocatorOptimizer::ProcessGraphDef(
      optimized_graph, graph_properties));

  VLOG(1) << "ScopedAllocatorOptimizer::Optimize() done";
  VLOG(3) << "Optimized graph:";
  DumpGraphToVLOG(*optimized_graph, /*log_level=*/3);
  return Status::OK();
}

ScopedAllocatorOptimizer::Rewriter* ScopedAllocatorOptimizer::GetRewriter(
    const string& op_name) {
  auto it = rewriters_.find(op_name);
  if (it != rewriters_.end()) {
    return it->second;
  }
  return nullptr;
}

int ScopedAllocatorOptimizer::NewScopedAllocatorId(int num_fields) {
  CHECK_GT(num_fields, 0);
  int id = next_sa_id_;
  next_sa_id_ += (num_fields + 1);
  CHECK_GT(next_sa_id_, 0);
  return id;
}

Status ScopedAllocatorOptimizer::NewIdentityId(int* id) {
  *id = next_identity_id_++;
  if (next_identity_id_ < 0) {
    return errors::Internal("NewIdentityId overflow");
  }
  return Status::OK();
}

ScopedAllocatorOptimizer::~ScopedAllocatorOptimizer() {
  for (auto ptr : to_delete_) {
    delete ptr;
  }
}

void ScopedAllocatorOptimizer::FindOpOccurrences(GraphDef* graph,
                                                 const OpNameSet& op_names,
                                                 GraphOpOccurrences* occs) {
  VLOG(1) << "FindOpOccurrences ";
  for (const auto& it : op_names) {
    VLOG(1) << "search target " << it;
  }
  for (int ni = 0; ni < graph->node_size(); ++ni) {
    NodeDef* node = graph->mutable_node(ni);
    const string& op_name = node->op();
    if (op_names.find(op_name) != op_names.end()) {
      VLOG(1) << "found " << op_name << " on dev " << node->device();
      (*occs)[node->device()][op_name].push_back(node);
    }
  }
}

namespace {
struct OpNameOrder {
  bool operator()(const NodeDef* a, const NodeDef* b) {
    return a->name() <= b->name();
  }
};

class Tree {
 public:
  Tree(const string& edge, int depth) : edge_(edge), depth_(depth) {}
  ~Tree() {
    for (const auto& it : subtrees_) delete it.second;
  }

  Tree* GetSubTree(const string& edge) {
    auto it = subtrees_.find(edge);
    if (it != subtrees_.end()) {
      return it->second;
    }
    Tree* t = new Tree(edge, depth_ + 1);
    subtrees_[edge] = t;
    return t;
  }

  void InsertNode(NodeDef* n) { nodes_.push_back(n); }

  string edge_;
  int depth_;
  std::vector<NodeDef*> nodes_;
  absl::flat_hash_map<string, Tree*> subtrees_;
};

// Applies a function to every Tree in DFS order.  Terminates early
// on any non-OK Status.
Status ApplyToAll(Tree* tree, const std::function<Status(Tree*)>& func) {
  Status s;
  for (const auto& it : tree->subtrees_) {
    s = ApplyToAll(it.second, func);
    if (!s.ok()) return s;
  }
  s = func(tree);
  return s;
}

Tree* ComputeScopeTree(const string& op_name,
                       const std::vector<NodeDef*>& node_vec) {
  Tree* root = new Tree("", 0);
  for (NodeDef* n : node_vec) {
    std::vector<string> pieces = str_util::Split(n->name(), "/");
    // last piece is node name proper.
    int depth = pieces.size() - 1;
    Tree* subtree = root;
    for (int i = 0; i < depth; ++i) {
      subtree = subtree->GetSubTree(pieces[i]);
    }
    subtree->InsertNode(n);
  }
  return root;
}

void PartitionByLoopStructure(const FrameView& frame_view,
                              std::vector<NodeDef*> nodes,
                              std::vector<std::vector<NodeDef*>>* loop_groups) {
  // It is assumed that two nodes with identical loop containment have
  // identical integer vectors. Represent those by 64 bit hashes.
  absl::flat_hash_map<uint64, std::vector<NodeDef*>> loop_sets;
  for (NodeDef* nd : nodes) {
    uint64 hash = 0;
    const std::vector<int>& loop_ids = frame_view.Frames(*nd);
    for (int id : loop_ids) {
      hash = Hash64Combine(hash, static_cast<uint64>(id));
    }
    loop_sets[hash].push_back(nd);
  }
  for (auto it : loop_sets) {
    loop_groups->push_back(std::move(it.second));
  }
}

// Identify outputs that are inputs to multiple sets of nodes.
void IdentifyRepeatedInputs(const std::vector<NodeDef*>& nodes,
                            absl::flat_hash_set<string>* seen_outputs,
                            absl::flat_hash_set<string>* repeated_outputs) {
  for (NodeDef* node : nodes) {
    for (const auto& input_name : node->input()) {
      if (!seen_outputs->insert(input_name).second) {
        repeated_outputs->insert(input_name);
      }
    }
  }
}

}  // namespace

Status ScopedAllocatorOptimizer::ProcessGraphDef(
    GraphDef* graph, const GraphProperties& graph_properties) {
  // Nodes created by this optimizer have the IsStateful() property
  // which means their names must be globally unique within a process,
  // so we include an optimizer invocation count in every generated
  // name.
  static std::atomic<int64> invocation_counter(1);
  const int64 invocation_count =
      invocation_counter.fetch_add(1, std::memory_order_seq_cst);
  VLOG(1) << "ProcessGraphDef " << invocation_count;
  Status status;
  GraphOpOccurrences occ;
  FindOpOccurrences(graph, op_name_set_, &occ);
  if (!occ.empty()) {
    FrameView frame_view;
    // TODO(ezhulenev): Pass a GraphView when this optimizer will be migrated
    // from NodeMap.
    LOG_WARNING_AND_RETURN_IF_ERROR(frame_view.InferFromGraph(*graph));

    for (auto& dt : occ) {
      VLOG(2) << "Processing device " << dt.first;
      const DevOpOccurrences& dev_occ = dt.second;
      for (auto& it : dev_occ) {
        string op_name = it.first;
        VLOG(1) << "Processing " << op_name << " set size " << it.second.size();
        Rewriter* rewriter = GetRewriter(op_name);
        if (!rewriter) {
          LOG(ERROR) << "Failed to find Rewriter in ScopedAllocatorOptimizer "
                     << "for op_name " << op_name;
          continue;
        }
        rewriter->SetGraphProperties(graph_properties);
        std::unique_ptr<Tree> root(ComputeScopeTree(it.first, it.second));
        // Record outputs that are inputs to multiple Tree nodes.
        absl::flat_hash_set<string> seen_outputs;
        status = ApplyToAll(root.get(), [this, &seen_outputs](Tree* t) {
          IdentifyRepeatedInputs(t->nodes_, &seen_outputs, &repeated_outputs_);
          return Status::OK();
        });
        if (!status.ok()) {
          break;
        }
        // Nodes with a common depth and root path are now grouped
        // in the same Tree struct.  Split those groups into subgroups that
        // share identical loop nesting.
        status = ApplyToAll(root.get(), [this, rewriter, graph, &frame_view,
                                         &op_name, invocation_count](Tree* t) {
          VLOG(2) << "applied to tree node " << t->edge_ << " at depth "
                  << t->depth_ << " of size " << t->nodes_.size();
          if (t->nodes_.size() > 1) {
            std::vector<std::vector<NodeDef*>> loop_groups;
            PartitionByLoopStructure(frame_view, t->nodes_, &loop_groups);
            for (auto& lg : loop_groups) {
              if (lg.size() > 1) {
                bool applied = false;
                Status s = OrderNodeSet(&lg);
                TF_RETURN_IF_ERROR(s);
                VLOG(1) << "Applying Rewriter for " << op_name;
                s = rewriter->Rewrite(this, invocation_count, graph, op_name,
                                      lg, &applied);
                LOG_WARNING_AND_RETURN_IF_ERROR(s);
              }
            }
          }
          return Status::OK();
        });
        if (!status.ok()) {
          break;
        }
      }
      if (!status.ok()) {
        break;
      }
    }
  }
  VLOG(1) << "ScopedAllocatorOptimizer returning " << status;
  if (!status.ok()) {
    LOG(ERROR) << "ScopedAllocatorOptimizer: " << status;
  }
  return status;
}

namespace {
struct InstanceKeyLess {
  bool operator()(const NodeDef* a, const NodeDef* b) const {
    AttrSlice a_attrs = AttrSlice(*a);
    AttrSlice b_attrs = AttrSlice(*b);
    int32 a_key = -1;
    int32 b_key = -1;
    Status s = GetNodeAttr(a_attrs, "instance_key", &a_key);
    CHECK(s.ok());
    s = GetNodeAttr(b_attrs, "instance_key", &b_key);
    CHECK(s.ok());
    return a_key < b_key;
  }
};

struct NameLess {
  bool operator()(const NodeDef* a, const NodeDef* b) const {
    return a->name() < b->name();
  }
};

bool IsCollectiveNode(const NodeDef& n) {
  AttrSlice attrs = AttrSlice(n);
  int key = -1;
  if (!IsCollective(n)) return false;
  Status s = GetNodeAttr(attrs, "instance_key", &key);
  if (s.ok() && key >= 0) {
    return true;
  }
  return false;
}
}  // namespace

Status ScopedAllocatorOptimizer::OrderNodeSet(
    std::vector<NodeDef*>* nodes) const {
  // Nodes should be identical type.  Default order is by name but for
  // collectives we order by increasing instance_key so each group gets
  // the same instance_key.
  if (nodes->size() <= 1) return Status::OK();
  if (IsCollectiveNode(*nodes->at(0))) {
    sort(nodes->begin(), nodes->end(), InstanceKeyLess());
  } else {
    sort(nodes->begin(), nodes->end(), NameLess());
  }
  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow

#undef LOG_WARNING_AND_RETURN_IF_ERROR
