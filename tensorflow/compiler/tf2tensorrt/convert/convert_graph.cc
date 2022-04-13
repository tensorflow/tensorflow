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

#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"

#include <fstream>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/logger_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/segment/segment.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/protobuf/config.pb.h"             // NOLINT
#include "tensorflow/core/protobuf/device_properties.pb.h"  // NOLINT
#include "tensorflow/core/protobuf/rewriter_config.pb.h"    // NOLINT
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/tensorrt/NvInfer.h"
namespace tensorflow {
namespace tensorrt {
namespace convert {

using absl::StrAppend;
using absl::StrCat;
using ::tensorflow::tensorrt::segment::ClusterProperty;
using ::tensorflow::tensorrt::segment::NodePtrCompare;
using ::tensorflow::tensorrt::segment::Segment;

namespace {

Status BuildNodeMap(const Graph& graph,
                    std::unordered_map<string, Node*>* node_map) {
  for (auto* node : graph.op_nodes()) {
    if (!node_map->insert({node->name(), node}).second) {
      return errors::AlreadyExists("Node name is not unique in graph: " +
                                   node->name());
    }
  }
  return Status::OK();
}

EngineInfo::EngineType GetEngineType(
    const TRTOptimizationPass::ConversionParams& params) {
  return (params.is_dynamic_op || params.use_calibration)
             ? EngineInfo::EngineType::TRTDynamic
             : EngineInfo::EngineType::TRTStatic;
}

// Returns true when use_implicit_batch is false or when we are building dynamic
// engine, to allow unknown size for dimensions rather than dimension 0.
bool AllowDynamicNonBatchDimension(
    const TRTOptimizationPass::ConversionParams& params) {
  return !params.use_implicit_batch ||
         GetEngineType(params) == EngineInfo::EngineType::TRTDynamic;
}

struct EdgePtrCompare {
  bool operator()(const Edge* lhs, const Edge* rhs) const {
    return lhs->id() < rhs->id();
  }
};

// TODO(laigd): instead of deciding the device here, the converter should accept
// a device name as one of the conversion parameter so users can control on
// which device they want to run the conversion.
std::pair<TfDeviceId, PlatformDeviceId> GetFirstValidDeviceId() {
  for (int tf_device_id_value = 0; tf_device_id_value < 100;
       ++tf_device_id_value) {
    TfDeviceId tf_device_id(tf_device_id_value);
    PlatformDeviceId platform_device_id;
    Status s =
        GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id);
    if (s.ok()) {
      VLOG(1) << "Found TF GPU " << tf_device_id.value() << " at cuda device "
              << platform_device_id.value();
      return std::make_pair(tf_device_id, platform_device_id);
    }
  }
  LOG(ERROR) << "Could not find any TF GPUs";
  return std::make_pair(TfDeviceId(-1), PlatformDeviceId(-1));
}

// Returns false for const nodes (we intend to drop control edges from those).
bool ShallKeepControlEdgeFrom(const Node* input_node) {
  if (!input_node) {
    LOG(ERROR) << "Node pointer is null, this should not happen";
    return false;
  }
  return input_node->type_string() != "Const";
}

// Function to get subsegment information structure.
Status GetEngineInfo(const Graph* g,
                     const grappler::GraphProperties& graph_properties,
                     const Segment& segment,
                     const std::vector<Node*>& reverse_topo_order,
                     EngineInfo* info) {
  std::vector<const Node*> subgraph_nodes;  // Topologically sorted nodes.
  std::set<const Node*> added_const_nodes;  // Used to prevent double insertion.

  const ClusterProperty& segment_property = segment.property;
  const std::set<const Node*, NodePtrCompare>& segment_nodes = segment.nodes;

  // The device assignment accumulated from the compatible device assignments
  // for the nodes in the segment.
  const DeviceNameUtils::ParsedName segment_device =
      segment_property.DeviceName();
  info->max_batch_size = segment_property.BatchSize().GetOptionalMaxBatchSize();

  // Map from src_node_name+port to the unique port numbers of the TRT op, where
  // the src_node_name is the name of the source node of the input/output
  // edge, thus there must not be any duplicates since source nodes of
  // input/output edges must be in different split of the graph.
  // TODO(aaroey): consider using node id and port instead.
  // TODO(aaroey): using topo order instead of reverting reverse topo order.
  std::unordered_map<string, int> input_to_engine_port, output_to_engine_port;
  for (auto it = reverse_topo_order.rbegin(); it != reverse_topo_order.rend();
       ++it) {
    const Node* node = *it;
    if (segment_nodes.count(node) == 0) continue;
    subgraph_nodes.push_back(node);

    const int node_id = node->id();
    const string& node_name = node->name();

    // Create input connections. Sort edges first to make deterministic since
    // in_edges is a set of pointers.
    std::vector<const Edge*> in_edges(node->in_edges().begin(),
                                      node->in_edges().end());
    std::sort(in_edges.begin(), in_edges.end(), EdgePtrCompare());
    for (const auto edge : in_edges) {
      auto input_node = edge->src();
      if (input_node->IsSource() || segment_nodes.count(input_node)) {
        continue;
      }
      if (edge->IsControlEdge()) {
        if (ShallKeepControlEdgeFrom(input_node)) {
          // Non-Const control input.
          info->connections.emplace_back(input_node->name(), input_node->id(),
                                         node_name, node_id,
                                         /*input_edge=*/true);
        }
      } else if (input_node->type_string() == "Const") {
        // Add constant data input nodes into the segment graphdef (thus also in
        // the engine). We don't care if it has other output edges going into
        // other engines or TF nodes. Since we add it only to the segment
        // graphdef, not the segment itself, it won't be removed from the graph.
        // If it doesn't have any edges, TF will prune it out.
        //
        // Note that the segmenter already ensure that the constant data input
        // is valid and supported by the engine.
        if (!added_const_nodes.insert(input_node).second) {
          // Already added before.
          continue;
        }
        VLOG(1) << "Adding const node " << input_node->name();
      } else {
        // Non-const data input.
        int port = Graph::kControlSlot - 1;
        // Use the source non-segment node name/port as key.
        const string s = StrCat(input_node->name(), ":", edge->src_output());
        VLOG(1) << "Input edge = " << s;
        if (input_to_engine_port.count(s)) {
          port = input_to_engine_port.at(s);
        } else {
          port = input_to_engine_port.size();
          input_to_engine_port.insert({s, port});
        }
        info->connections.emplace_back(
            input_node->name(), input_node->id(), edge->src_output(), node_name,
            node_id, edge->dst_input(), /*input_edge=*/true, port);
      }
    }
    // Create output connections. Sort edges first to make deterministic since
    // out_edges is a set of pointers.
    std::vector<const Edge*> out_edges(node->out_edges().begin(),
                                       node->out_edges().end());
    std::sort(out_edges.begin(), out_edges.end(), EdgePtrCompare());
    for (const auto edge : out_edges) {
      auto output_node = edge->dst();
      if (output_node->IsSink() || segment_nodes.count(output_node)) {
        continue;
      }
      if (edge->IsControlEdge()) {
        // Control output.
        if (ShallKeepControlEdgeFrom(node)) {
          info->connections.emplace_back(output_node->name(), output_node->id(),
                                         node_name, node_id,
                                         /*input_edge=*/false);
        }
      } else {
        // Data output.
        int port = Graph::kControlSlot - 1;
        // Use the source segment node name/port as key.
        const string s = StrCat(node_name, ":", edge->src_output());
        VLOG(1) << "Output edge = " << s;
        if (output_to_engine_port.count(s)) {
          port = output_to_engine_port.at(s);
        } else {
          port = output_to_engine_port.size();
          output_to_engine_port.insert({s, port});
        }
        info->connections.emplace_back(
            output_node->name(), output_node->id(), edge->dst_input(),
            node_name, node_id, edge->src_output(), /*input_edge=*/false, port);
      }
    }
  }  // For each segment node in topological order.

  // Construct the const nodes first.
  subgraph_nodes.insert(subgraph_nodes.begin(), added_const_nodes.begin(),
                        added_const_nodes.end());
  TF_RETURN_IF_ERROR(
      ConvertSegmentToGraphDef(g, graph_properties, subgraph_nodes, info));
  VLOG(1) << "Converted TensorRT candidate segment '" << info->engine_name
          << "' to a GraphDef";
  if (segment_device.has_type) {
    // If the accumulated device assignment for the segment has a device type,
    // the segmenter guarantees the device type is GPU. Use the device
    // assignment in this case.
    if (segment_device.type != "GPU") {
      return errors::Internal(
          "segment device is not GPU: ",
          DeviceNameUtils::ParsedNameToString(segment_device));
    }
    info->device = DeviceNameUtils::ParsedNameToString(segment_device);
  } else {
    TfDeviceId tf_device_id;
    PlatformDeviceId platform_device_id;
    std::tie(tf_device_id, platform_device_id) = GetFirstValidDeviceId();
    if (tf_device_id.value() >= 0) {
      DeviceNameUtils::ParsedName parsed_name;
      parsed_name.type = "GPU";
      parsed_name.has_type = true;
      parsed_name.id = tf_device_id.value();
      parsed_name.has_id = true;
      info->device = DeviceNameUtils::ParsedNameToString(parsed_name);
    } else {
      VLOG(1) << "No device is assigned to the segment. A device will be "
                 "assigned during graph execution (inference).";
    }
  }
  return Status::OK();
}

// Helper function to update edge connection from the removed node to the
// engine node. If an outside node is gone, it must have been absorbed into
// an engine node. Find the engine node.
void UpdateToEngineNode(const std::vector<EngineInfo>& infos,
                        const size_t my_engine_id,
                        const std::vector<Node*>& engine_nodes,
                        const bool is_input_edge, const string& node_name,
                        Node** node, int* port) {
  for (size_t t = 0; t < infos.size(); ++t) {
    if (t == my_engine_id) {
      continue;
    }
    const auto& info = infos.at(t);
    for (const auto& eng_conn : info.connections) {
      // If the connection being updated is an input connection, the source of
      // the connection must be an output connection of another engine. And vise
      // versa.
      if (is_input_edge == eng_conn.is_input_edge) continue;
      if (eng_conn.inside_node_name == node_name &&
          eng_conn.inside_port == *port) {
        *node = CHECK_NOTNULL(engine_nodes[t]);
        QCHECK_EQ(info.engine_name, (**node).name())
            << "Engine name mismatch: " << info.engine_name << " vs "
            << (**node).name();
        *port = eng_conn.port_number;
        return;
      }
    }
  }
  LOG(FATAL) << "Node " << node_name << " not found in any engine.";
}

tensorflow::TensorShapeProto ComputeTRTNodeIOShape(
    std::vector<PartialTensorShape>& partial_tensorshape_vect,
    std::vector<tensorflow::TensorShapeProto>& shape_proto_vect,
    const PartialTensorShape& conn_shape, int port_number) {
  tensorflow::TensorShapeProto tmp_shape_proto;
  conn_shape.AsProto(&tmp_shape_proto);

  if (partial_tensorshape_vect.size() <= port_number) {
    shape_proto_vect.resize(port_number + 1);
    partial_tensorshape_vect.resize(port_number + 1);
  }

  return tmp_shape_proto;
}

// Function to insert a TRT engine node into the graph.
// Create engine nodes in the following way:
// 1. Each invocation of CreateTRTNode creates an engine node for infos[pos]
// 2. When an engine node is created, add it into the graph with necessary
//    re-wiring.
//    2.1. If the outside connected node is existing, connect the engine
//         node to it.
//    2.2. If the outside connected node is gone, it must have been absorted
//         into another engine node (which was processed before the processing
//         one). Connect to the pre-existing engine node instead.
// 3. In this way, we ensure the graph is topologically sort-able after each
//    invocation of CreateTRTNode().
Status CreateTRTNode(const TRTOptimizationPass::ConversionParams& params,
                     const std::vector<EngineInfo>& infos, int pos,
                     int default_max_batch_size, Graph* graph,
                     std::vector<Node*>* engine_nodes,
                     grappler::Cluster* cluster) {
  const auto& info = infos.at(pos);
  std::vector<tensorflow::TensorShapeProto> input_shape_protos;
  std::vector<tensorflow::TensorShapeProto> output_shape_protos;
  std::vector<PartialTensorShape> input_shapes;
  std::vector<PartialTensorShape> output_shapes;
  std::vector<NodeDefBuilder::NodeOut> inputs;
  std::vector<Node*> input_nodes;
  std::vector<Node*> control_input_nodes;
  std::unordered_set<string> control_input_names;
  std::vector<DataType> out_types;

  VLOG(1) << "Processing " << info.engine_name;
  // Collect needed info for creating the engine node in the graph
  for (const auto& conn : info.connections) {
    // Control edges
    if (conn.is_control_edge()) {
      // Skip control outputs for now. control output info are not needed for
      // node creation and will be processed later.
      if (!conn.is_input_edge) continue;

      // Rewrire control input if it's not found in original graph.
      Node* input_node = graph->FindNodeId(conn.outside_id);
      int port = Graph::kControlSlot;
      if (!input_node) {
        UpdateToEngineNode(infos, pos, *engine_nodes, /*is_input_edge=*/true,
                           conn.outside_node_name, &input_node, &port);
        QCHECK_EQ(Graph::kControlSlot, port);
      }
      if (!control_input_names.insert(input_node->name()).second) {
        continue;
      }
      control_input_nodes.push_back(input_node);
      VLOG(1) << "Engine Control Input " << input_node->name() << " -> "
              << info.engine_name;
    } else {
      // Data edges
      if (!conn.is_input_edge) {
        // Set the shapes and data types of the output edge.
        tensorflow::TensorShapeProto out_shape = ComputeTRTNodeIOShape(
            /*partial_tensorshape_vect=*/output_shapes,
            /*shape_proto_vect=*/output_shape_protos,
            /*conn_shape=*/conn.inside_shape,
            /*port_number=*/conn.port_number);

        output_shape_protos.at(conn.port_number) = out_shape;
        output_shapes.at(conn.port_number) = conn.inside_shape;

        if (out_types.size() <= conn.port_number) {
          out_types.resize(conn.port_number + 1);
        }
        out_types.at(conn.port_number) = conn.connection_type;
        VLOG(2) << "Collected output shape "
                << output_shape_protos.at(conn.port_number).DebugString();
      } else {
        // Set the shapes of the input edge.
        tensorflow::TensorShapeProto in_shape = ComputeTRTNodeIOShape(
            /*partial_tensorshape_vect=*/input_shapes,
            /*shape_proto_vect=*/input_shape_protos,
            /*conn_shape=*/conn.outside_shape,
            /*port_number=*/conn.port_number);

        input_shape_protos.at(conn.port_number) = in_shape;
        input_shapes.at(conn.port_number) = conn.outside_shape;

        // Shape must be fully defined (excluding batch dimension) for static
        // mode.
        if (params.use_implicit_batch &&
            info.engine_type == EngineInfo::EngineType::TRTStatic) {
          for (int i = 1; i < conn.outside_shape.dims(); i++) {
            if (conn.outside_shape.dim_size(i) <= 0) {
              return errors::Internal(
                  "Not fully defined input shape when in static mode which "
                  "should have been excluded by the segmenter. ");
            }
          }
        }

        // Rewrire data input if it's not found in original graph.
        Node* input_node = graph->FindNodeId(conn.outside_id);
        int port = conn.outside_port;
        if (!input_node) {
          UpdateToEngineNode(infos, pos, *engine_nodes, /*is_input_edge=*/true,
                             conn.outside_node_name, &input_node, &port);
        }
        if (std::find_if(
                std::begin(inputs), std::end(inputs),
                [input_node, &port](const NodeDefBuilder::NodeOut& inp) {
                  return inp.node == input_node->name() && inp.index == port;
                }) == std::end(inputs)) {
          inputs.emplace_back(input_node->name(), port, conn.connection_type);
          input_nodes.push_back(CHECK_NOTNULL(input_node));
          VLOG(1) << "Engine Input " << input_node->name() << ":" << port
                  << " -> " << info.engine_name << ":" << inputs.size() - 1;
        }
      }
    }
  }
  // We don't support segments with no inputs. Fall back to native TF here to
  // avoid crash later. Constant folding should've folded the ops that make up
  // these segments.
  if (inputs.empty()) {
    return errors::Internal(
        "Segment has no inputs (possible constfold failure)");
  }

  // Build the engine and get its serialized representation.
  string segment_string;

  int max_batch_size = info.max_batch_size.has_value()
                           ? info.max_batch_size.value()
                           : default_max_batch_size;

  if (info.engine_type == EngineInfo::EngineType::TRTStatic) {
    TF_RETURN_IF_ERROR(CreateStaticEngine(params, info, max_batch_size,
                                          input_shapes, nullptr,
                                          &segment_string, cluster));
  }

  string prec_string;
  TF_RETURN_IF_ERROR(TrtPrecisionModeToName(info.precision_mode, &prec_string));
  NodeDefBuilder node_builder(info.engine_name, "TRTEngineOp");
  if (!info.device.empty()) node_builder.Device(info.device);
  if (VLOG_IS_ON(1)) {
    string ins = StrCat(info.engine_name, " inputs= ");
    for (const auto& ii : inputs) {
      StrAppend(&ins, ii.node, ":", ii.index, " ");
    }
    VLOG(1) << ins;
  }
  node_builder.Input(inputs);
  for (const string& c : control_input_names) {
    node_builder.ControlInput(c);
  }

  NodeDef trt_node;
  NameAttrList function;
  function.set_name(StrCat(info.engine_name, "_native_segment"));

  node_builder.Attr("input_shapes", input_shape_protos)
      .Attr("output_shapes", output_shape_protos)
      .Attr("static_engine",
            info.engine_type == EngineInfo::EngineType::TRTStatic)
      .Attr("segment_func", function)
      .Attr("serialized_segment", segment_string)
      .Attr("calibration_data", "")
      .Attr("max_cached_engines_count", info.maximum_cached_engines)
      .Attr("workspace_size_bytes", info.max_workspace_size_bytes)
      .Attr("max_batch_size", max_batch_size)
      .Attr("precision_mode", prec_string)
      .Attr("use_calibration", info.use_calibration)
      .Attr("_use_implicit_batch", params.use_implicit_batch)
      .Attr("use_explicit_precision", params.use_explicit_precision)
      .Attr("_allow_build_at_runtime", info.allow_build_at_runtime)
      .Attr("OutT", out_types);

  if (!params.use_implicit_batch) {
    node_builder.Attr("profile_strategy",
                      ProfileStrategyToName(params.profile_strategy));
  }

  Status status = node_builder.Finalize(&trt_node);

  if (!status.ok()) {
    LOG(ERROR) << "Node construction failed with" << status;
    return status;
  }
  VLOG(1) << "Adding TRTEngine " << info.engine_name << " to graph";

  // Up until this point, graph is not modified. If we return !status.ok() from
  // here, this segment will be skipped
  // TODO(aaroey): let it return proper error status for the following logic
  // instead of checking fail.
  TF_ASSIGN_OR_RETURN(Node * engine_node, graph->AddNode(trt_node));
  (*engine_nodes)[pos] = engine_node;
  // Add control input and input edges to the engine node.
  for (const auto in : control_input_nodes) {
    VLOG(1) << "Connecting control edge from " << in->name() << " to "
            << engine_node->name();
    graph->AddControlEdge(in, engine_node);
  }
  VLOG(1) << "input_nodes size = " << input_nodes.size();
  for (int i = 0; i < input_nodes.size(); ++i) {
    Node* n = CHECK_NOTNULL(input_nodes[i]);
    const auto& in = inputs[i];
    VLOG(1) << "Connecting data edge from " << n->name() << ":" << in.index
            << " to " << engine_node->name() << ":" << i;
    graph->AddEdge(n, in.index, engine_node, i);
  }

  // Updates the inputs of output edges destination nodes, and point them to the
  // engine node.
  for (auto& conn : info.connections) {
    if (conn.is_input_edge) {
      continue;
    }
    Node* output_node = graph->FindNodeId(conn.outside_id);
    int port = conn.outside_port;
    if (!output_node) {
      UpdateToEngineNode(infos, pos, *engine_nodes, /*is_input_edge=*/false,
                         conn.outside_node_name, &output_node, &port);
    }
    if (conn.is_control_edge()) {
      VLOG(1) << "Updating control edge from " << engine_node->name() << " to "
              << output_node->name();
      QCHECK_EQ(Graph::kControlSlot, port);
      graph->AddControlEdge(engine_node, output_node);
    } else {
      VLOG(1) << "Updating data edge from " << engine_node->name() << ":"
              << conn.port_number << " to " << output_node->name() << ":"
              << port;
      // Use UpdateEdge() to avoid adding the same edge multiple times.
      TF_CHECK_OK(
          graph->UpdateEdge(engine_node, conn.port_number, output_node, port));
    }
  }
  return Status::OK();
}

int64 GetNextGraphSequenceNumber() {
  static std::atomic<int64_t> graph_sequence_num;
  return graph_sequence_num++;
}

constexpr char kCastInputTypeAttrName[] = "SrcT";

// Transforms node = cast(x, fp32) where datatype(x) != fp16 to:
//   castToFp16 = cast(x, fp16)
//   node = cast(castToFp16, fp32)
//
Status MaybeRewriteCastToFp32(GraphDef* graph_def, NodeDef* node_def) {
  if (node_def->op() != "Cast") {
    return Status::OK();
  }

  DataTypeVector input_types;
  DataTypeVector output_types;
  TF_RETURN_IF_ERROR(
      graph_transforms::GetInOutTypes(*node_def, &input_types, &output_types));

  if (input_types.size() != 1 || output_types.size() != 1) {
    return errors::Internal("Bad cast operation");
  }

  if (input_types[0] == DT_HALF || output_types[0] != DT_FLOAT) {
    return Status::OK();
  }

  VLOG(2) << "Rewriting cast to FP32 " << node_def->DebugString();

  NodeDef* castToFp16 = graph_def->add_node();
  for (auto attr_value : node_def->attr()) {
    (*castToFp16->mutable_attr())[attr_value.first] = attr_value.second;
  }
  castToFp16->set_name(node_def->name() + "_split");
  castToFp16->set_op("Cast");
  castToFp16->set_device(node_def->device());
  castToFp16->add_input(node_def->input(0));
  (*castToFp16->mutable_attr())[kCastOutputTypeAttrName].set_type(DT_HALF);

  node_def->set_input(0, castToFp16->name() + ":0");
  (*node_def->mutable_attr())[kCastInputTypeAttrName].set_type(DT_HALF);

  VLOG(2) << castToFp16->DebugString();
  VLOG(2) << node_def->DebugString();

  return Status::OK();
}

}  // namespace

Status RegisterGraphToFunctionLibrary(const GraphDef& segment_graph_def,
                                      Graph* graph, const string& engine_name) {
  Graph segment_graph(graph->flib_def());
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(GraphConstructorOptions(),
                                            segment_graph_def, &segment_graph));
  FunctionDefLibrary library;
  auto segment_func = library.add_function();
  TF_RETURN_IF_ERROR(GraphToFunctionDef(
      segment_graph, StrCat(engine_name, "_native_segment"), segment_func));
  if (VLOG_IS_ON(7)) {
    VLOG(7) << engine_name << " Function_Def ";
    VLOG(7) << segment_func->DebugString();
  }
  VLOG(1) << "Adding funcdef " << segment_func->signature().name()
          << " to graphlib";
  TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(library));
  return Status::OK();
}

std::pair<int, Allocator*> GetDeviceAndAllocator(
    const grappler::Cluster* cluster, const EngineInfo& engine) {
  int cuda_device_id = -1;
  Allocator* dev_allocator = nullptr;
  if (cluster == nullptr || cluster->GetDeviceSet() == nullptr ||
      engine.device.empty()) {
    // If device is not set, use the first found GPU device for the conversion.
    TfDeviceId tf_device_id;
    PlatformDeviceId platform_device_id;
    std::tie(tf_device_id, platform_device_id) = GetFirstValidDeviceId();
    cuda_device_id = platform_device_id.value();
    if (cuda_device_id >= 0) {
      GPUOptions gpu_options;
      // If the TF to Cuda gpu id mapping exist, the device and corresponding
      // allocator must have been initialized already, so the
      // GetGPUAllocator() call won't create a new allocator.
      dev_allocator = GPUProcessState::singleton()->GetGPUAllocator(
          gpu_options, tf_device_id, /*total_bytes=*/1, /*peer_gpu_ids=*/{});
    }
    return std::make_pair(cuda_device_id, dev_allocator);
  }

  // Use the device requested by the engine.
  auto device_set = cluster->GetDeviceSet();
  std::vector<Device*> devices;
  DeviceNameUtils::ParsedName parsed_name;
  if (DeviceNameUtils::ParseFullName(engine.device, &parsed_name) &&
      parsed_name.has_id) {
    device_set->FindMatchingDevices(parsed_name, &devices);
  }
  if (!devices.empty()) {
    if (devices.size() > 1) {
      string msg = "Found multiple matching devices using name '";
      StrAppend(&msg, engine.device, "': ");
      for (auto d : devices) StrAppend(&msg, d->name(), ", ");
      StrAppend(&msg, ". Will get the allocator from first one.");
      LOG_WARNING_WITH_PREFIX << msg;
    }
    AllocatorAttributes alloc_attr;
    cuda_device_id = devices[0]->tensorflow_accelerator_device_info()->gpu_id;
    dev_allocator = devices[0]->GetAllocator(alloc_attr);
    VLOG(1) << "Using allocator " << dev_allocator->Name()
            << " and cuda_device_id " << cuda_device_id;
  } else {
    LOG_WARNING_WITH_PREFIX << "Cluster is set but device '" << engine.device
                            << "' is not found in the cluster";
  }
  return std::make_pair(cuda_device_id, dev_allocator);
}

Status CreateStaticEngine(const TRTOptimizationPass::ConversionParams& params,
                          const EngineInfo& info, int max_batch_size,
                          const std::vector<PartialTensorShape>& input_shapes,
                          TrtShapeOptimizationProfile* profile,
                          string* segment_string, grappler::Cluster* cluster) {
  std::pair<int, Allocator*> device_allocator =
      GetDeviceAndAllocator(cluster, info);
  int cuda_device_id = 0;
  std::unique_ptr<TRTBaseAllocator> trt_allocator;
  if (device_allocator.first >= 0) {
    cuda_device_id = device_allocator.first;
    trt_allocator.reset(new TRTDeviceAllocator(device_allocator.second));
  } else {
    // The value in trt_allocator is a nullptr and cudamalloc will be used.
    LOG_WARNING_WITH_PREFIX << "Can't identify the cuda device. Running on "
                               "device 0 and use cudamalloc as an allocator";
  }
  cudaSetDevice(cuda_device_id);

  auto trt_logger = GetLoggerRegistry()->LookUp(params.trt_logger_name);
  const bool calibrate_int8 =
      (info.precision_mode == TrtPrecisionMode::INT8 && info.use_calibration);

  // Create static engines with precision_mode fp32/fp16.
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine;
  TF_RETURN_IF_ERROR(ConvertGraphDefToEngine(
      info.segment_graph_def, nullptr,
      calibrate_int8 ? TrtPrecisionMode::FP32 : info.precision_mode,
      max_batch_size, info.max_workspace_size_bytes, input_shapes, trt_logger,
      trt_allocator.get(), /*calibrator=*/nullptr, &engine,
      info.use_calibration, params.use_implicit_batch,
      /*convert_successfully=*/nullptr, profile, info.engine_name,
      /*use_explicit_precision=*/params.use_explicit_precision, cluster));
  TrtUniquePtrType<nvinfer1::IHostMemory> engine_data(engine->serialize());
  *segment_string = string(static_cast<const char*>(engine_data->data()),
                           engine_data->size());
  return Status::OK();
}

Status ConvertGraph(const TRTOptimizationPass::ConversionParams& params,
                    grappler::GrapplerItem& grappler_item,
                    const std::vector<string>& input_output_names,
                    grappler::Cluster* cluster, GraphDef* output) {
  // Sanity checks.
  TRT_ENSURE(output != nullptr)
  if (params.precision_mode != TrtPrecisionMode::INT8 &&
      params.use_calibration) {
    return errors::InvalidArgument(
        "Calibration with FP32 or FP16 is not supported.");
  }

  GraphDef& graph_def = grappler_item.graph;

  // When precision_mode is FP16, transform cast(x, fp32) to
  // cast(cast(x, fp16), fp32). This creates cast(fp16, f32) that can be
  // included in the TRTEngineOp as an TensorRT Identity layer for performance:
  //  . Avoid cast(fp32, fp16) in the TRT engine implementation for fp16
  //    precision.
  //  . Changing the input to the TRTEngine from fp32 to fp16 may reduce data
  //    moving from the host to the GPU.
  if (params.precision_mode == TrtPrecisionMode::FP16) {
    for (int i = 0; i < graph_def.node_size(); i++) {
      NodeDef* node_def = graph_def.mutable_node(i);
      TF_RETURN_IF_ERROR(MaybeRewriteCastToFp32(&graph_def, node_def));
    }
  }

  // Construct a GrapplerItem using the modified graph_def and the input
  // grappler_item.
  grappler::GraphProperties static_graph_properties(grappler_item);
  TF_RETURN_IF_ERROR(static_graph_properties.InferStatically(true));

  // Convert graphdef to graph.
  FunctionLibraryDefinition flib(OpRegistry::Global(), graph_def.library());
  Graph graph(flib);
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(GraphConstructorOptions(), graph_def, &graph));

  // Segment the graph into subgraphs that can be converted to TensorRT
  segment::SegmentOptions segment_options;
  // TODO(ben,jie,sami): exclude output nodes (DISCUSS IT)
  for (const auto& node : input_output_names) {
    segment_options.exclude_node_list.insert(node);
  }
  segment_options.minimum_segment_size = params.minimum_segment_size;
  segment_options.use_implicit_batch = params.use_implicit_batch;
  if (segment_options.use_implicit_batch)
    segment_options.maximum_batch_size = params.max_batch_size;
  segment_options.allow_dynamic_non_batch_dim =
      AllowDynamicNonBatchDimension(params);

  segment::SegmentVector initial_segments;
  TrtNodeValidator validator(static_graph_properties, params.precision_mode,
                             params.use_calibration, params.use_implicit_batch,
                             params.use_explicit_precision);
  TF_RETURN_IF_ERROR(segment::SegmentGraph(
      &graph, &static_graph_properties,
      std::bind(&TrtNodeValidator::IsTensorRTCandidate, &validator,
                std::placeholders::_1),
      // Input validation is already done by TrtNodeValidator, so we don't
      // need to check the input edges.
      [](const Edge* edge) { return true; }, OutputEdgeValidator(),
      segment_options, &initial_segments));
  LOG(INFO) << "Number of TensorRT candidate segments: "
            << initial_segments.size();

  // Get the EngineInfo for each segment.
  std::unordered_map<string, Node*> node_map;
  TF_RETURN_IF_ERROR(BuildNodeMap(graph, &node_map));
  std::vector<EngineInfo> engine_segments;
  engine_segments.reserve(initial_segments.size());
  std::vector<Node*> reverse_topo_order;
  GetPostOrder(graph, &reverse_topo_order);
  segment::SegmentVector converted_segments;
  converted_segments.reserve(initial_segments.size());
  string engine_name_prefix =
      StrCat("TRTEngineOp_",
             absl::StrFormat("%0*d", 3, GetNextGraphSequenceNumber()), "_");
  for (size_t t = 0; t < initial_segments.size(); t++) {
    auto& curr_segment = initial_segments.at(t);
    EngineInfo curr_engine;
    curr_engine.engine_name =
        StrCat(engine_name_prefix, absl::StrFormat("%0*d", 3, t));

    bool int8_no_calib = (!params.use_calibration &&
                          params.precision_mode == TrtPrecisionMode::INT8);
    bool has_qdq = false;
    if (int8_no_calib) {
      has_qdq = absl::c_any_of(reverse_topo_order, IsQuantizeAndDequantizeOp);
    }

    Status status = GetEngineInfo(&graph, static_graph_properties, curr_segment,
                                  reverse_topo_order, &curr_engine);
    if (!status.ok()) {
      LOG_WARNING_WITH_PREFIX << "Failed to get engine info for segment " << t
                              << ": " << status;
      continue;
    }

    curr_engine.engine_type = GetEngineType(params);
    curr_engine.use_calibration = params.use_calibration;
    // Building cuda engines for INT8 without calibration and without dynamic
    // range info cause TRT failure. Avoid this situation by setting the
    // precision to FP16.
    if (int8_no_calib && !has_qdq) {
      LOG(WARNING) << "Set engine precision to FP16 due to missing QDQ OP";
      curr_engine.precision_mode = TrtPrecisionMode::FP16;
    } else {
      curr_engine.precision_mode = params.precision_mode;
    }
    curr_engine.maximum_cached_engines = params.max_cached_engines;
    curr_engine.allow_build_at_runtime = params.allow_build_at_runtime;
    if (!curr_engine.max_batch_size.has_value()) {
      curr_engine.max_batch_size = params.max_batch_size;
    }

    status = RegisterGraphToFunctionLibrary(curr_engine.segment_graph_def,
                                            &graph, curr_engine.engine_name);

    if (!status.ok()) {
      LOG_WARNING_WITH_PREFIX
          << "Failed to register segment graphdef to the library " << t << ": "
          << status;
      continue;
    }

    engine_segments.push_back(std::move(curr_engine));
    converted_segments.push_back(std::move(curr_segment));

    if (VLOG_IS_ON(8)) {
      string fname = engine_segments.back().engine_name;
      StrAppend(&fname, ".pb");
      std::fstream f;
      f.open(fname.c_str(), std::fstream::out | std::fstream::binary);
      f << engine_segments.at(t).segment_graph_def.SerializeAsString();
      f.close();
    }
  }

  // Save the cuda device since we may need to switch to another cuda device to
  // build static engines.
  absl::optional<int> old_cuda_device = absl::nullopt;
  if (!params.is_dynamic_op) {
    int cuda_device_id;
    cudaError_t cuda_error = cudaGetDevice(&cuda_device_id);
    if (cuda_error != cudaSuccess) {
      LOG_WARNING_WITH_PREFIX << "Couldn't get current device: "
                              << cudaGetErrorString(cuda_error);
    } else {
      VLOG(1) << "Current cuda device is " << cuda_device_id;
      old_cuda_device = cuda_device_id;
    }
  }

  auto restore_cuda_device = gtl::MakeCleanup([old_cuda_device] {
    if (old_cuda_device.has_value()) {
      cudaSetDevice(old_cuda_device.value());
    }
  });

  std::vector<Node*> engine_nodes;
  engine_nodes.resize(engine_segments.size());
  for (int i = 0; i < engine_segments.size(); ++i) {
    auto& engine = engine_segments.at(i);
    // TODO(b/170762693): implement the heuristic to calculate
    // max_workspace_size_bytes.
    engine.max_workspace_size_bytes = params.max_workspace_size_bytes;
    VLOG(1) << "Assigned " << engine.max_workspace_size_bytes << " bytes to "
            << engine.engine_name;
    auto status =
        CreateTRTNode(params, engine_segments, i, params.max_batch_size, &graph,
                      &engine_nodes, cluster);

    string msg = StrCat("segment ", i, " consisting of ",
                        converted_segments.at(i).nodes.size(), " nodes by ",
                        engine.engine_name);
    if (status.ok()) {
      LOG(INFO) << "Replaced " << msg << ".";
    } else {
      // Graph is not modified.
      LOG_WARNING_WITH_PREFIX << "Cannot replace " << msg
                              << " reason: " << status.error_message()
                              << " (keeping original segment).";
    }
    if (VLOG_IS_ON(1)) {
      msg = "Segment consists of nodes: ";
      for (const Node* node : converted_segments.at(i).nodes) {
        StrAppend(&msg, node->name(), ", ");
      }
      VLOG(1) << msg;
    }

    // If status is ok, we successfully added the node to the graph and can
    // remove segment ops. Otherwise graph is not modified.
    if (status.ok()) {
      for (const Node* node : converted_segments.at(i).nodes) {
        graph.RemoveNode(const_cast<Node*>(node));
      }
    }
  }
  graph.ToGraphDef(output);
  return Status::OK();
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
