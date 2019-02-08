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

#include "tensorflow/core/debug/debug_graph_utils.h"

#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/debug.pb.h"

namespace tensorflow {

namespace {

// TODO(cais): Switch to safe_strtob when available.
Status ParseBoolString(const string& bool_str, bool* bool_val) {
  const string lower_bool_str = str_util::Lowercase(bool_str);
  if (lower_bool_str == "false" || lower_bool_str == "f" ||
      lower_bool_str == "0") {
    *bool_val = false;
  } else if (lower_bool_str == "true" || lower_bool_str == "t" ||
             lower_bool_str == "1") {
    *bool_val = true;
  } else {
    return errors::InvalidArgument("Invalid string for bool value: ", bool_str);
  }
  return Status::OK();
}

}  // namespace

// static
Status DebugNodeInserter::InsertNodes(
    const protobuf::RepeatedPtrField<DebugTensorWatch>& watches, Graph* graph,
    Device* device) {
  // TODO(cais): This method is getting too large in size.
  // Refactor it with helpers.

  if (watches.empty()) {
    // Nothing to do: Return OK right away.
    return Status::OK();
  }

  // A map from tensor name (e.g., "node_a:0") to list of debug op names
  // (e.g., {"DebugIdentity", "DebugNanCount"})
  std::unordered_map<string, std::vector<string>> tensor_watches;
  // A map from tensor name to debug_url.
  std::unordered_map<string, std::vector<string>> tensor_watch_urls;
  std::unordered_map<string, bool> tensor_tolerate_failures;

  // Cache the proto content for fast lookup later
  for (const DebugTensorWatch& watch : watches) {
    if (watch.output_slot() < 0) {
      // The semantics of output_slot == -1 is that the node is watched only
      // for completion, but not for output tensor values (see
      // NodeCompletionCallback in debug_gateway.h).
      continue;
    }
    if (watch.debug_ops().empty()) {
      continue;
    }

    string tensor_name =
        strings::StrCat(watch.node_name(), ":", watch.output_slot());

    std::vector<string> debug_ops;
    for (const string& debug_op : watch.debug_ops()) {
      debug_ops.push_back(debug_op);
    }

    tensor_watches[tensor_name] = debug_ops;
    tensor_tolerate_failures[tensor_name] =
        watch.tolerate_debug_op_creation_failures();

    std::vector<string> urls;
    for (const string& url : watch.debug_urls()) {
      urls.push_back(url);
    }
    tensor_watch_urls[tensor_name] = urls;
  }

  if (tensor_watches.empty()) {
    return Status::OK();
  }

  DeviceType device_type = DeviceType{device->device_type()};

  // Keep track of all edges to be removed.
  std::vector<const Edge*> edges_to_remove;

  for (Node* src_node : graph->nodes()) {
    // Make a map from output slot to outgoing edges from the slot.
    std::unordered_map<int, std::vector<const Edge*>> output_slot_to_edges;
    for (const Edge* edge : src_node->out_edges()) {
      const int src_output = edge->src_output();
      if (output_slot_to_edges.find(src_output) == output_slot_to_edges.end()) {
        output_slot_to_edges[src_output] = {edge};
      } else {
        output_slot_to_edges[src_output].push_back(edge);
      }
    }

    // Iterate through all output slots of the node.
    for (int src_output_slot = 0; src_output_slot < src_node->num_outputs();
         ++src_output_slot) {
      const string tensor_name =
          strings::StrCat(src_node->name(), ":", src_output_slot);
      if (tensor_watches.find(tensor_name) == tensor_watches.end()) {
        // Add debug nodes only for edges with matching source node and source
        // output slot.
        continue;
      }

      // Now we have encountered a watched tensor. We will:
      //   1) Mark this edge as to be removed, iff this is a non-Reference
      //      tensor
      //   2) Create a Copy node for the tensor
      //   3) Add a new edge, from the source tensor to the Copy node
      //   4) Add a new edge, from the Copy node to the destination node, iff
      //      this is a non-Reference tensor.
      //   5) Create all the requested debug nodes and their edges to the Copy
      //      node.
      //   6) Add control edges from the debug nodes to the destination nodes
      //      to ensure that the tensors values exported by the debug nodes
      //      to the debug URLs reflect the values before the execution of
      //      the destination nodes.

      const DataType src_dt = src_node->output_type(src_output_slot);
      MemoryType memory_type;
      TF_RETURN_IF_ERROR(MemoryTypeForOutput(device_type, graph, src_node,
                                             src_output_slot, &memory_type));

      // Create the copy node for the watched tensor.
      Node* copy_node;
      Status copy_s = CreateCopyNode(
          graph, device_type, memory_type == HOST_MEMORY, src_node->name(),
          src_output_slot, src_dt, tensor_name, tensor_watches[tensor_name],
          tensor_watch_urls[tensor_name], &copy_node);
      if (!copy_s.ok()) {
        return Status(
            error::FAILED_PRECONDITION,
            strings::StrCat("Failed to create Copy/CopyHost node for tensor ",
                            tensor_name, ", due to: ", copy_s.error_message()));
      }

      // Add edge from watched tensor to the copy node.
      graph->AddEdge(src_node, src_output_slot, copy_node, 0);

      // Create all requested debug nodes and their edges to the Copy node.
      std::vector<Node*> debug_nodes;
      for (size_t i = 0; i < tensor_watches[tensor_name].size(); ++i) {
        const string& debug_op_name = tensor_watches[tensor_name][i];

        Node* debug_node;
        Status debug_s = CreateDebugNode(
            graph, *device, copy_node->name(), src_dt, tensor_name,
            tensor_watch_urls[tensor_name], i, debug_op_name, &debug_node);
        if (debug_s.ok()) {
          graph->AddEdge(copy_node, 0, debug_node, 0);
          debug_nodes.push_back(debug_node);
        } else {
          if (tensor_tolerate_failures[tensor_name]) {
            LOG(INFO) << "Tolerating failure to create debug node: "
                      << "tensor name = " << tensor_name << "; "
                      << "debug op name = " << debug_op_name;
          } else {
            return Status(
                error::FAILED_PRECONDITION,
                strings::StrCat("Failed to create debug node ", debug_op_name,
                                " for tensor ", tensor_name,
                                ", due to: ", debug_s.error_message()));
          }
        }
      }

      // Is the output a reference?
      const bool is_ref = IsRefType(src_node->output_type(src_output_slot));

      // Iterate through all outgoing edges attached to the slot.
      for (const Edge* edge : output_slot_to_edges[src_output_slot]) {
        // Mark the edge for removal.
        if (!is_ref) {
          edges_to_remove.push_back(edge);
          graph->AddEdge(copy_node, 0, edge->dst(), edge->dst_input());
        }

        // Add control edges from the debug nodes to the destination node
        // to ensure that the debug nodes are executed before the destination
        // node. Skip Enter and NextIteration ops to avoid hanging.
        for (Node* debug_node : debug_nodes) {
          if (!src_node->IsEnter() && !src_node->IsNextIteration()) {
            graph->AddEdge(debug_node, Graph::kControlSlot, edge->dst(),
                           Graph::kControlSlot);
          }
        }
      }
    }
  }

  // Remove all edges marked for removal.
  for (const Edge* edge : edges_to_remove) {
    graph->RemoveEdge(edge);
  }

  return Status::OK();
}

void DebugNodeInserter::DeparallelizeWhileLoops(Graph* graph, Device* device) {
  bool deparallelized_a_loop = false;
  for (Node* node : graph->nodes()) {
    if (node->IsEnter()) {
      const AttrValue* parallel_iterations =
          node->attrs().Find("parallel_iterations");
      if (parallel_iterations && parallel_iterations->i() > 1) {
        deparallelized_a_loop = true;
        VLOG(1) << "Changing the parallel_iterations attribute of the "
                << "Enter/RefEnter node \"" << node->name() << "\" on device \""
                << device->name() << "\" from " << parallel_iterations->i()
                << " to 1.";
        node->AddAttr<int64>("parallel_iterations", 1);
      }
    }
  }
  if (deparallelized_a_loop) {
    LOG(INFO) << "For debugging, tfdbg has set the parallel_iterations "
              << "attribute of all scheduled Enter/RefEnter nodes to 1. (This "
              << "does not affect subsequent non-debug runs.)";
  }
}

// static
const string DebugNodeInserter::GetCopyNodeName(const string& node_name,
                                                const int output_slot) {
  // For example, if the watched node is named "node1" and the output slot
  // is 0, the debug node will be called: __copy_node1_0
  return strings::StrCat("__copy_", node_name, "_", output_slot);
}

// static
const string DebugNodeInserter::GetDebugNodeName(const string& tensor_name,
                                                 const int debug_op_num,
                                                 const string& debug_op_name) {
  // For example, if the watched node is named "node1" and the debug op that
  // watches the output slot of node1 is of the type "DebugNanCount", the
  // debug node will be called: __dbg_node1_0_0_DebugNanCount.
  return strings::StrCat("__dbg_", tensor_name, "_", debug_op_num, "_",
                         debug_op_name);
}

// static
Status DebugNodeInserter::CreateCopyNode(
    Graph* graph, const DeviceType device_type, const bool is_host_memory,
    const string& src_node_name, const int src_output, const DataType src_dt,
    const string& tensor_name, const std::vector<string>& debug_ops,
    const std::vector<string>& debug_urls, Node** copy_node) {
  const string kGatedGrpcAttributeKey = "gated_grpc";

  NodeDef node_def;
  const KernelDef* kdef;

  const string copy_op_name = is_host_memory ? "CopyHost" : "Copy";
  const string copy_node_name = GetCopyNodeName(src_node_name, src_output);

  // Cross debug_ops and debug_urls to get the list of debug ops and watches.
  std::vector<string> debug_ops_spec;
  for (const string& debug_op : debug_ops) {
    for (const string& debug_url : debug_urls) {
      string debug_op_name_proper;
      std::unordered_map<string, string> custom_attributes;
      TF_RETURN_IF_ERROR(ParseDebugOpName(debug_op, &debug_op_name_proper,
                                          &custom_attributes));

      bool gated_grpc_value = false;
      if (custom_attributes.find(kGatedGrpcAttributeKey) !=
          custom_attributes.end()) {
        TF_RETURN_IF_ERROR(ParseBoolString(
            custom_attributes[kGatedGrpcAttributeKey], &gated_grpc_value));
      }
      debug_ops_spec.push_back(strings::StrCat(debug_op_name_proper, ";",
                                               debug_url, ";",
                                               gated_grpc_value ? "1" : "0"));
    }
  }

  auto builder = NodeDefBuilder(copy_node_name, copy_op_name)
                     .Input(src_node_name, src_output, src_dt)
                     .Attr("debug_ops_spec", debug_ops_spec);

  if (!builder.Finalize(&node_def).ok()) {
    return Status(
        error::FAILED_PRECONDITION,
        strings::StrCat("Failed to create node definition ", "for copy op ",
                        copy_node_name, " on watched tensor ", tensor_name));
  }
  Status s = FindKernelDef(device_type, node_def, &kdef, nullptr);

  if (!s.ok()) {
    return Status(
        error::FAILED_PRECONDITION,
        strings::StrCat("Failed to find kernel definition ", "for copy op ",
                        copy_node_name, " on watched tensor ", tensor_name));
  }
  if (!NodeBuilder(builder).Finalize(graph, copy_node).ok()) {
    return Status(error::FAILED_PRECONDITION,
                  strings::StrCat("Failed to create copy node ", copy_node_name,
                                  " on watched tensor ", tensor_name));
  }

  return Status::OK();
}

// static
Status DebugNodeInserter::ParseDebugOpName(
    const string& debug_op_name, string* debug_op_name_proper,
    std::unordered_map<string, string>* attributes) {
  const size_t l_index = debug_op_name.find('(');
  const size_t r_index = debug_op_name.find(')');
  if (l_index == string::npos && r_index == string::npos) {
    *debug_op_name_proper = debug_op_name;
  } else {
    if (l_index == string::npos || l_index == 0 ||
        r_index != debug_op_name.size() - 1) {
      return errors::InvalidArgument("Malformed debug op name \"",
                                     debug_op_name, "\"");
    }

    *debug_op_name_proper = debug_op_name.substr(0, l_index);
    string arguments = debug_op_name.substr(l_index + 1, r_index - l_index - 1);

    std::vector<string> attribute_segs = str_util::Split(arguments, ";");
    for (const string& attribute_seg : attribute_segs) {
      StringPiece seg(attribute_seg);
      str_util::RemoveWhitespaceContext(&seg);
      if (seg.empty()) {
        continue;
      }

      const size_t eq_index = seg.find('=');
      if (eq_index == string::npos) {
        return errors::InvalidArgument(
            "Malformed attributes in debug op name \"", debug_op_name, "\"");
      }

      const string key(seg.substr(0, eq_index));
      const string value(
          seg.substr(eq_index + 1, attribute_seg.size() - eq_index - 1));
      if (key.empty() || value.empty()) {
        return errors::InvalidArgument(
            "Malformed attributes in debug op name \"", debug_op_name, "\"");
      }

      if (attributes->find(key) == attributes->end()) {
        (*attributes)[key] = value;
      } else {
        return errors::InvalidArgument("Duplicate attribute name \"", key,
                                       "\" found in the debug op: \"",
                                       debug_op_name, "\"");
      }
    }
  }
  return Status::OK();
}

// static
Status DebugNodeInserter::SetDebugNodeAttributes(
    Node* debug_node, const std::unordered_map<string, string>& attributes) {
  std::unordered_set<string> unfulfilled_keys;
  for (const auto& item : attributes) {
    unfulfilled_keys.insert(item.first);
  }

  for (const auto& attr : debug_node->op_def().attr()) {
    if (attributes.find(attr.name()) != attributes.end()) {
      const string& attr_value = attributes.at(attr.name());
      if (attr.type() == "string") {
        debug_node->AddAttr<string>(attr.name(), attr_value);
      } else if (attr.type() == "float") {
        float float_value = 0.0;
        if (!::tensorflow::strings::safe_strtof(attr_value.c_str(),
                                                &float_value)) {
          return errors::InvalidArgument(
              "Invalid value string for float-type attribute ", attr.name(),
              "of debug node ", debug_node->name(), ": \"", attr_value, "\"");
        }
        debug_node->AddAttr<float>(attr.name(), float_value);
      } else if (attr.type() == "int") {
        int64 int_value = 0;
        if (!::tensorflow::strings::safe_strto64(attr_value, &int_value)) {
          return errors::InvalidArgument(
              "Invalid value string for int-type attribute ", attr.name(),
              "of debug node ", debug_node->name(), ": \"", attr_value, "\"");
        }
        debug_node->AddAttr<int>(attr.name(), int_value);
      } else if (attr.type() == "bool") {
        bool bool_value;
        if (!ParseBoolString(attr_value, &bool_value).ok()) {
          return errors::InvalidArgument(
              "Invalid value string for bool-type attribute ", attr.name(),
              "of debug node ", debug_node->name(), ": \"", attr_value, "\"");
        }
        debug_node->AddAttr<bool>(attr.name(), bool_value);
      } else {
        return errors::InvalidArgument(
            "Unsupported type of custom attribute for debug ops: ",
            attr.type());
      }

      unfulfilled_keys.erase(attr.name());
    }
  }

  if (unfulfilled_keys.empty()) {
    return Status::OK();
  } else {
    return errors::InvalidArgument(
        unfulfilled_keys.size(),
        " attribute key(s) were not valid for debug node ", debug_node->name(),
        ": ", str_util::Join(unfulfilled_keys, ", "));
  }
}

// static
Status DebugNodeInserter::CreateDebugNode(
    Graph* graph, const Device& device, const string& src_copy_node_name,
    const DataType src_dt, const string& tensor_name,
    const std::vector<string>& debug_urls, const int debug_op_num,
    const string& debug_op_name, Node** debug_node) {
  NodeDef node_def;
  const KernelDef* kdef;

  string debug_op_name_proper;
  std::unordered_map<string, string> custom_attributes;
  TF_RETURN_IF_ERROR(ParseDebugOpName(debug_op_name, &debug_op_name_proper,
                                      &custom_attributes));

  const string debug_node_name =
      GetDebugNodeName(tensor_name, debug_op_num, debug_op_name_proper);
  auto builder = NodeDefBuilder(debug_node_name, debug_op_name_proper)
                     .Input(src_copy_node_name, 0, src_dt)
                     .Attr("device_name", device.name())
                     .Attr("tensor_name", tensor_name)
                     .Attr("debug_urls", debug_urls);

  if (!builder.Finalize(&node_def).ok()) {
    return errors::FailedPrecondition(
        "Failed to create node definition for debug op ", debug_op_name_proper,
        " on watched tensor ", tensor_name);
  }
  if (!FindKernelDef(DeviceType(device.device_type()), node_def, &kdef, nullptr)
           .ok()) {
    return errors::FailedPrecondition(
        "Failed to find kernel definition for debug op ", debug_op_name_proper,
        " on watched tensor ", tensor_name);
  }
  if (!NodeBuilder(builder).Finalize(graph, debug_node).ok()) {
    return errors::FailedPrecondition("Failed to create debug node ",
                                      debug_op_name_proper,
                                      " on watched tensor ", tensor_name);
  }

  // Set custom attributes (if any).
  if (!custom_attributes.empty()) {
    TF_RETURN_IF_ERROR(SetDebugNodeAttributes(*debug_node, custom_attributes));
  }

  return Status::OK();
}

}  // namespace tensorflow
