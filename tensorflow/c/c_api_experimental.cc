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

#include "tensorflow/c/c_api_experimental.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/compiler/jit/legacy_flags/mark_for_compilation_pass_flags.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/config.pb.h"

using tensorflow::Node;
using tensorflow::NodeBuilder;
using tensorflow::NodeDef;
using tensorflow::Status;
using tensorflow::string;

namespace {

const char* const DEVICE_TPU_REPLICATED_CORE = "TPU_REPLICATED_CORE";
const char* const DEVICE_TPU_SYSTEM = "TPU_SYSTEM";

TF_Operation* ToTF_Operation(Node* node) {
  return static_cast<TF_Operation*>(static_cast<void*>(node));
}

// Graph rewrite algorithm (modeled after the python TPU graph rewrite path):
//
// 1. For each input node I, with C being the consumer node of I's output:
//
// a) When infeed is not specified, feed I to a new TPUReplicatedInput node
// (both running on CPU), which in turn feeds a new Identity node N, and N feeds
// C (both running on TPU).
//
// b) Otherwise, feed I to a new InfeedEnqueueTuple node IE, both running on
// CPU. Also set an InfeedDequeueTuple node ID to feed C, both running on
// TPU.
//
// In case b), if we have multiple input nodes, they all feed into the same
// InfeedEnqueueTuple node, so that the graph has a single pair of infeed
// enqueue and dequeue nodes. The list of output tensors from the dequeue node
// can go to different consumer nodes. For example, say the original graph has
// input nodes I1 and I2 respectively feeding nodes C1 and C2. After the rewrite
// with infeed ops, we will have: I1 and I2 feed a single infeed enqueue node
// IE, and a corresponding infeed dequeue node ID produces a list of two
// tensors, respectively feeding C1 and C2.
//
// 2. Rewrite all existing graph nodes by adding an attribute on TPU
// cluster. For each node C reading some input node I, rewire it to read from a
// new input node generated in step #1 above.
//
// 3. For each output node O, feed it to a new Identity node, which in turn
// feeds a new TPUReplicatedOutput node, which in turn feeds a new Identity node
// M. Return the set of new output nodes (the "M" nodes) for caller to fetch
// from.
//
// Limitations compared to the python TPU rewrite path:
// - # replicas is always 1.
// - Less error checking.
//
// TODO(hongm): Simplify the graph rewrite to generating fewer TPUReplicate
// related nodes.
class GraphRewriter {
 public:
  GraphRewriter(TF_Graph* graph, int num_input_nodes,
                const TF_Output* input_nodes, int num_output_nodes,
                const TF_Output* output_nodes)
      EXCLUSIVE_LOCKS_REQUIRED(graph->mu)
      : graph_(graph), input_nodes_(input_nodes) {
    for (int i = 0; i < num_input_nodes; ++i) {
      //  Will fill in the value part later when we create the associated new
      //  input node.
      input_node_map_[input_nodes[i].oper->node.name()] =
          NodeBuilder::NodeOut(nullptr, -1);
    }

    // Grab all existing nodes for the upcoming rewrite, before mutating the
    // graph.
    for (Node* n : graph->graph.nodes()) {
      nodes_to_rewrite_.push_back(n);
    }

    for (int i = 0; i < num_output_nodes; ++i) {
      output_node_map_.emplace(output_nodes[i].oper->node.name(),
                               PortIndexPair{output_nodes[i].index, i});
    }
  }

  // On success, sets `config_op` and `shutdown_op` to the corresponding
  // "ConfigureDistributedTPU" and "ShutdownDistributedTPU" nodes added to the
  // graph.
  tensorflow::Status Rewrite(TF_Output* new_output_nodes,
                             TF_Operation** infeed_enqueue_node,
                             TF_Output* config_op, TF_Output* shutdown_op)
      EXCLUSIVE_LOCKS_REQUIRED(graph_->mu) {
    TF_RETURN_IF_ERROR(ProcessInputNodes(infeed_enqueue_node));

    return RewriteGraphAndAddOutputNodes(new_output_nodes, config_op,
                                         shutdown_op);
  }

 private:
  // Synthesizes new graph nodes (infeed enqueue or TPU replicated input
  // nodes) for the input nodes, and creates a replicated metadata node.
  //
  // When `infeed_enqueue_node` is non-NULL and there are some input nodes,
  // also adds the infeed dequeue node.
  tensorflow::Status ProcessInputNodes(TF_Operation** infeed_enqueue_node)
      EXCLUSIVE_LOCKS_REQUIRED(graph_->mu) {
    Node* metadata_node;
    TF_RETURN_IF_ERROR(
        NodeBuilder(metadata_node_name_.c_str(), "TPUReplicateMetadata")
            .Attr("num_replicas", 1)
            .Attr("_tpu_replicate", cluster_name_.c_str())
            .Finalize(&graph_->graph, &metadata_node));

    Node* dequeue_node = nullptr;
    // Be deterministic in the corner case where `use_infeed` below is false.
    if (infeed_enqueue_node) *infeed_enqueue_node = nullptr;
    const bool use_infeed =
        infeed_enqueue_node != nullptr && !input_node_map_.empty();
    if (use_infeed) {
      std::vector<NodeBuilder::NodeOut> new_input_list;
      new_input_list.reserve(input_node_map_.size());
      std::vector<tensorflow::DataType> input_dtypes;
      input_dtypes.reserve(input_node_map_.size());
      std::vector<tensorflow::TensorShape> input_shapes;
      input_shapes.reserve(input_node_map_.size());
      for (int i = 0; i < input_node_map_.size(); ++i) {
        Node& input_node = input_nodes_[i].oper->node;
        new_input_list.push_back(
            NodeBuilder::NodeOut(&input_node, input_nodes_[i].index));
        input_dtypes.push_back(input_node.output_type(input_nodes_[i].index));
        tensorflow::TensorShapeProto shape;
        TF_RETURN_IF_ERROR(
            tensorflow::GetNodeAttr(input_node.attrs(), "shape", &shape));
        VLOG(1) << "Input node " << i << " has shape " << shape.DebugString();
        input_shapes.push_back(shape);
      }
      // Enqueue always runs on CPU.
      Node* enqueue_node;
      TF_RETURN_IF_ERROR(NodeBuilder("InfeedEnqueueTuple", "InfeedEnqueueTuple")
                             .Input(new_input_list)
                             .Device("/device:CPU:0")
                             .Attr("device_ordinal", 0)
                             .Attr("dtypes", input_dtypes)
                             .Attr("shapes", input_shapes)
                             .Finalize(&graph_->graph, &enqueue_node));
      *infeed_enqueue_node = ToTF_Operation(enqueue_node);
      // The dequeue node should be put onto the "_tpu_replicate" cluster.
      TF_RETURN_IF_ERROR(
          NodeBuilder("TPUReplicate/InfeedDequeueTuple", "InfeedDequeueTuple")
              .ControlInput(metadata_node)
              .Attr("_tpu_replicate", cluster_name_.c_str())
              .Attr("dtypes", input_dtypes)
              .Attr("shapes", input_shapes)
              .Finalize(&graph_->graph, &dequeue_node));
    }

    for (int i = 0; i < input_node_map_.size(); ++i) {
      VLOG(1) << "Handling input node " << input_nodes_[i].oper->node.name();
      if (use_infeed) {
        DCHECK(dequeue_node);
        input_node_map_[input_nodes_[i].oper->node.name()] =
            NodeBuilder::NodeOut(dequeue_node, i);
      } else {
        Node* replicated_input_node;
        {
          std::string replicated_input_name("TPUReplicate/input" +
                                            std::to_string(i));
          NodeBuilder::NodeOut input(&input_nodes_[i].oper->node,
                                     input_nodes_[i].index);
          std::vector<NodeBuilder::NodeOut> input_list;
          input_list.push_back(input);
          TF_RETURN_IF_ERROR(
              NodeBuilder(replicated_input_name.c_str(), "TPUReplicatedInput")
                  // This op requires an input list.
                  .Input(input_list)
                  .Finalize(&graph_->graph, &replicated_input_node));
        }

        {
          Node* new_input_node;
          const std::string new_input_name("TPUReplicate/replicated_input_" +
                                           std::to_string(i));
          TF_RETURN_IF_ERROR(NodeBuilder(new_input_name.c_str(), "Identity")
                                 .Input(replicated_input_node, 0)
                                 .ControlInput(metadata_node)
                                 .Attr("_tpu_replicate", cluster_name_.c_str())
                                 .Finalize(&graph_->graph, &new_input_node));
          DCHECK_GT(input_node_map_.count(input_nodes_[i].oper->node.name()),
                    0);
          input_node_map_[input_nodes_[i].oper->node.name()] =
              NodeBuilder::NodeOut(new_input_node, 0);
        }
      }
    }
    return Status::OK();
  }

  // On success, sets `config_op` and `shutdown_op` to the corresponding
  // "ConfigureDistributedTPU" and "ShutdownDistributedTPU" nodes added to the
  // graph.
  tensorflow::Status RewriteGraphAndAddOutputNodes(TF_Output* new_output_nodes,
                                                   TF_Output* config_op,
                                                   TF_Output* shutdown_op)
      EXCLUSIVE_LOCKS_REQUIRED(graph_->mu) {
    tensorflow::Status s;
    // For each non-input node in the input graph, place the node in a "TPU
    // replicate cluster" via an attribute, and with the above metadata node
    // as a control dependency.
    //
    // Although we have handled the input nodes in ProcessInputNodes(), some
    // of those nodes may also serve as output nodes, which we will handle
    // below.
    for (Node* n : nodes_to_rewrite_) {
      if (n->IsSource()) continue;
      VLOG(1) << "Rewriting node " << n->name();

      if (n->IsSink()) {
        // TODO(hongm): Rewire SINK to be control dependent on the new input
        // nodes created above?
        continue;
      }

      const NodeDef& old_def = n->def();
      // Let node C be the consumer of `n`'s output in the original graph.
      // This new node will feed into C in the rewritten graph.
      NodeBuilder::NodeOut new_node;
      if (input_node_map_.count(n->name())) {
        new_node = input_node_map_[n->name()];
      } else {
        // This node is to replace `n` in the graph.
        NodeDef new_def = n->def();
        const std::string new_node_name = "TPUReplicate/" + n->name();
        new_def.set_name(new_node_name);
        new_def.clear_input();
        for (int i = 0; i < old_def.input_size(); ++i) {
          const string old_input_name = old_def.input(i);
          // When there are multiple input nodes that get mapped to the same
          // infeed dequeue node, use different output ports of the dequeue
          // node. e.g. Say in the original graph, input I1 feeds C1, and I2
          // feeds C2. After the rewrite, I1 and I2 both feed a new infeed
          // enqueue node, and the corresponding dequeue node has its output
          // port 0 feeding C1, and output port 1 feeding C2. Note C1 and C2
          // could be the same node (e.g. an Add that takes 2 inputs).
          const string new_input_name =
              input_node_map_.count(old_input_name) > 0
                  ? tensorflow::strings::StrCat(
                        input_node_map_[old_input_name].node->name(), ":",
                        input_node_map_[old_input_name].index)
                  : "TPUReplicate/" + old_input_name;
          new_def.add_input(new_input_name);
        }
        if (old_def.input_size() == 0) {
          // It is sufficient to only set control dependency of nodes without
          // input. Other nodes with input(s) with inherit such control
          // dependency.
          // e.g. say the graph computes add(x, y). Once we make nodes x and y
          // control-dependent on the metadata node, node add will inherit
          // such control dependency indirectly.
          new_def.add_input(
              tensorflow::strings::StrCat("^", metadata_node_name_.c_str()));
        }
        tensorflow::AddNodeAttr("_tpu_replicate", cluster_name_.c_str(),
                                &new_def);
        new_node = NodeBuilder::NodeOut(graph_->graph.AddNode(new_def, &s), 0);
        if (!s.ok()) {
          return s;
        }
        VLOG(1) << "The rewritten node node is "
                << new_node.node->DebugString();
      }

      if (output_node_map_.count(n->name()) > 0) {
        VLOG(1) << "Handling output node " << n->name();
        auto range_it = output_node_map_.equal_range(n->name());
        for (auto it = range_it.first; it != range_it.second; ++it) {
          const PortIndexPair& pair = it->second;
          Node* out_identity_node;
          {
            // If this output node is also an input, use the input_node_map_'s
            // stored port, which would also work for an infeed dequeue op.
            // Otherwise use pair.port.
            // An example of the former: Say the graph has input nodes I1 and
            // I2, and the output nodes are also I1 and I2. In the rewritten
            // graph with infeed, the 2 output nodes will both come from a
            // single infeed dequeue node ID, with output ports respectively
            // set to 0 and 1.
            const int output_port =
                input_node_map_.count(n->name()) ? new_node.index : pair.port;
            VLOG(1) << "Handling its output port " << output_port
                    << " at output index " << pair.index;
            std::string output_node_name = "TPUReplicate/Identity";
            if (pair.index > 0) {
              output_node_name += "_" + std::to_string(pair.index);
            }
            TF_RETURN_IF_ERROR(
                NodeBuilder(output_node_name.c_str(), "Identity")
                    .Input(new_node.node, output_port)
                    .Device(!old_def.device().empty()
                                ? old_def.device()
                                : tensorflow::strings::StrCat(
                                      "/device:", DEVICE_TPU_REPLICATED_CORE))
                    .Attr("_tpu_replicate", cluster_name_.c_str())
                    .Finalize(&graph_->graph, &out_identity_node));
            VLOG(1) << "out_identity_node: "
                    << out_identity_node->DebugString();
          }

          Node* replicated_output_node;
          {
            const std::string replicated_output_node_name =
                "TPUReplicate/output" + std::to_string(pair.index);
            TF_RETURN_IF_ERROR(
                NodeBuilder(replicated_output_node_name.c_str(),
                            "TPUReplicatedOutput")
                    .Input(out_identity_node, 0)
                    .Attr("num_replicas", 1)
                    .Finalize(&graph_->graph, &replicated_output_node));
            VLOG(1) << "replicated_output_node: "
                    << replicated_output_node->DebugString();
          }

          Node* final_output_node;
          const std::string final_output_node_name =
              "TPUReplicate/output_" + std::to_string(pair.index) + "_shard_" +
              std::to_string(0);
          TF_RETURN_IF_ERROR(
              NodeBuilder(final_output_node_name.c_str(), "Identity")
                  .Input(replicated_output_node, 0)
                  .Finalize(&graph_->graph, &final_output_node));
          VLOG(1) << "new_output_node: " << final_output_node->DebugString();
          auto oper = ToTF_Operation(final_output_node);
          new_output_nodes[pair.index] = {oper, 0};
        }
      }

      if (input_node_map_.count(n->name()) == 0) {
        graph_->graph.RemoveNode(n);
      }
    }

    {
      Node* config_node;
      TF_RETURN_IF_ERROR(
          NodeBuilder("ConfigureDistributedTPU", "ConfigureDistributedTPU")
              .Device(DEVICE_TPU_SYSTEM)
              .Finalize(&graph_->graph, &config_node));
      *config_op = {ToTF_Operation(config_node), 0};
    }

    {
      Node* shutdown_node;
      TF_RETURN_IF_ERROR(
          NodeBuilder("ShutdownDistributedTPU", "ShutdownDistributedTPU")
              .Device(DEVICE_TPU_SYSTEM)
              .Finalize(&graph_->graph, &shutdown_node));
      *shutdown_op = {ToTF_Operation(shutdown_node), 0};
    }

    return Status::OK();
  }

  TF_Graph* const graph_;

  const TF_Output* const input_nodes_;

  const std::string cluster_name_ = "TPUReplicate/cluster";
  const std::string metadata_node_name_ = "TPUReplicate/TPUReplicateMetadata";

  // Keep mappings from the current input nodes to newly created input nodes,
  // which we will use to rewrite existing nodes that read these
  // inputs. e.g. A node that reads input node PlaceHolder could be rewired to
  // read the created TPUReplicate/replicated_input_0 node or some output port
  // of the created TPUReplicate/InfeedDequeueTuple node. Because of the latter
  // case, we the map entries store NodeBuilder::NodeOut, and not just Node*.
  std::unordered_map<std::string, NodeBuilder::NodeOut> input_node_map_;

  std::vector<Node*> nodes_to_rewrite_;

  // Map from name to set{(output port, output tensor idx)}.
  // e.g. Say there are 3 output tensors, respectively produced by (node 0,
  // port 0), (node 0, port 1), (node 1, port 0). Then the mapping entries
  // are: node 0 -> {(port 0, idx 0), (port 1, idx 1)} node 1 -> {(port 0, idx
  // 2)} Based on these mappings, we will generate 3 new output nodes.
  struct PortIndexPair {
    int port;
    int index;
  };
  std::multimap<std::string, PortIndexPair> output_node_map_;
};

}  // namespace

void TF_EnableXLACompilation(TF_SessionOptions* options, unsigned char enable) {
  tensorflow::ConfigProto& config = options->options.config;
  auto* optimizer_options =
      config.mutable_graph_options()->mutable_optimizer_options();
  if (enable) {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::ON_1);

    // These XLA flags are needed to trigger XLA properly from C (more generally
    // non-Python) clients. If this API is called again with `enable` set to
    // false, it is safe to keep these flag values as is.
    tensorflow::legacy_flags::MarkForCompilationPassFlags* flags =
        tensorflow::legacy_flags::GetMarkForCompilationPassFlags();
    flags->tf_xla_cpu_global_jit = true;
    flags->tf_xla_min_cluster_size = 1;
  } else {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::OFF);
  }
}

TF_Output TF_SetupTPUExecution(TF_Session* session, int num_input_nodes,
                               const TF_Output* input_nodes,
                               int num_output_nodes,
                               const TF_Output* output_nodes,
                               TF_Output* new_output_nodes,
                               TF_Operation** infeed_enqueue_node,
                               TF_Status* status) {
  TF_Output config_op, shutdown_op;
  {
    auto graph = session->graph;
    tensorflow::mutex_lock c(graph->mu);

    VLOG(1) << "Graph before TPU rewrite: "
            << graph->graph.ToGraphDefDebug().DebugString();
    GraphRewriter rewriter(graph, num_input_nodes, input_nodes,
                           num_output_nodes, output_nodes);
    status->status = rewriter.Rewrite(new_output_nodes, infeed_enqueue_node,
                                      &config_op, &shutdown_op);
    if (!status->status.ok()) {
      return shutdown_op;
    }
    VLOG(1) << "Graph after TPU rewrite: "
            << graph->graph.ToGraphDefDebug().DebugString();
  }

  VLOG(1) << "Initializing TPU";
  TF_Tensor* dummy_output;
  TF_SessionRun(session, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ nullptr, /*input_values*/ nullptr, /*ninputs*/ 0,
                // output related parameters
                /*outputs*/ &config_op, /*output_values*/ &dummy_output,
                /*noutputs*/ 1,
                /*targets*/ nullptr, /*ntargets*/ 0,
                /*run_metadata*/ nullptr, status);
  if (status->status.ok()) {
    TF_DeleteTensor(dummy_output);
  }
  return shutdown_op;
}

void TF_ShutdownTPUExecution(TF_Session* session, TF_Output shutdown_node,
                             TF_Status* status) {
  {
    tensorflow::mutex_lock c(session->graph->mu);
    VLOG(1) << "Shutting down TPU, with input graph: "
            << session->graph->graph.ToGraphDefDebug().DebugString();
  }

  TF_SessionRun(session, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ nullptr, /*input_values*/ nullptr, /*ninputs*/ 0,
                // output related parameters
                /*outputs*/ nullptr, /*output_values*/ nullptr,
                /*noutputs*/ 0,
                /*targets*/ &shutdown_node.oper, /*ntargets*/ 1,
                /*run_metadata*/ nullptr, status);
}
