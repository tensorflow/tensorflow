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

#include "tensorflow/contrib/tensorrt/convert/convert_graph.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"

#include <fstream>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/contrib/tensorrt/convert/convert_nodes.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resource_manager.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resources.h"
#include "tensorflow/contrib/tensorrt/segment/segment.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"  // NOLINT

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include <cuda/include/cuda_runtime_api.h>
#include "tensorrt/include/NvInfer.h"
namespace tensorflow {
namespace tensorrt {
namespace convert {
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;
std::vector<int> GetLinkedTensorRTVersion() {
  return {NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH};
}
std::vector<int> GetLoadedTensorRTVersion() {
  int ver = getInferLibVersion();
  int ver_major = ver / 1000;
  ver = ver - ver_major * 1000;
  int ver_minor = ver / 100;
  int ver_patch = ver - ver_minor * 100;
  return {ver_major, ver_minor, ver_patch};
}
namespace {

bool IsTensorRTCandidate(const tensorflow::Node* node) {
  // LINT.IfChange
  // TODO(jie): Segmentation shouldn't associated with op name.
  //            Split it into a registration for each kernel.
  static const std::set<string> candidate_ops = {
      "Identity",
      "Snapshot",
      "Const",
      "Conv2D",
      "MaxPool",
      "BiasAdd",
      "Relu",
      "Add",
      "Mul",
      "Sub",
      "Rsqrt",
      "Pad",
      "Mean",
      "AvgPool",
      "ConcatV2",
      "DepthwiseConv2dNative",
      "FusedBatchNorm",
      "FusedBatchNormV2",
      // TODO(ben,jie): ...
  };
  // LINT.ThenChange(//tensorflow/contrib/tensorrt/convert/convert_nodes.h)
  return (candidate_ops.count(node->type_string()) ||
          PluginFactoryTensorRT::GetInstance()->IsPlugin(node->type_string()));
}

void GetSubGraphIncomingEdges(const tensorflow::Graph& graph,
                              const std::set<int>& subgraph_node_ids,
                              tensorflow::EdgeSet* incoming_edges) {
  for (int node_id : subgraph_node_ids) {
    const tensorflow::Node* node = graph.FindNodeId(node_id);
    for (const tensorflow::Edge* edge : node->in_edges()) {
      if (!subgraph_node_ids.count(edge->src()->id()) &&
          !edge->src()->IsSource() && !edge->IsControlEdge()) {
        incoming_edges->insert(edge);
        VLOG(2) << "INCOMING " << edge->src()->name() << " -> " << node->name()
                << " Y, ";
      } else {
        VLOG(2) << "INCOMING " << edge->src()->name() << " -> " << node->name()
                << " N, ";
      }
    }
  }
}

void GetSubGraphOutgoingEdges(const tensorflow::Graph& graph,
                              const std::set<int>& subgraph_node_ids,
                              tensorflow::EdgeSet* outgoing_edges) {
  for (int node_id : subgraph_node_ids) {
    const tensorflow::Node* node = graph.FindNodeId(node_id);
    for (const tensorflow::Edge* edge : node->out_edges()) {
      if (!subgraph_node_ids.count(edge->dst()->id()) &&
          !edge->dst()->IsSink() && !edge->IsControlEdge()) {
        VLOG(2) << "OUTGOING " << node->name() << " -> " << edge->dst()->name()
                << " Y, ";
        outgoing_edges->insert(edge);
      } else {
        VLOG(2) << "OUTGOING " << node->name() << " -> " << edge->dst()->name()
                << " N, ";
      }
    }
  }
}

std::pair<string, int> ParseTensorName(const string& name,
                                       int default_idx = 0) {
  string name_no_idx = name;
  int idx = default_idx;
  const size_t sep = name_no_idx.find_last_of(':');
  if (sep != string::npos) {
    name_no_idx = name_no_idx.substr(0, sep);
    idx = std::stoi(name.substr(sep + 1));
  }
  return std::make_pair(name_no_idx, idx);
}

std::unordered_map<string, std::vector<int>> BuildTensorNameMap(
    const std::vector<string>& tensor_names) {
  std::unordered_map<string, std::vector<int>> result;
  for (const string& tensor_name : tensor_names) {
    string node_name;
    int index;
    std::tie(node_name, index) = ParseTensorName(tensor_name);
    result[node_name].push_back(index);
  }
  return result;
}

// TODO(sami): convert references to pointers
struct ConvertGraphParams {
  ConvertGraphParams(
      tensorflow::Graph& inp_graph,
      const std::vector<string>& output_node_names,
      const std::set<int>& subgraph_node_id_numbers,
      size_t max_supported_batch_size, size_t max_consumed_workspace_size_bytes,
      const tensorflow::grappler::GraphProperties& current_graph_properties,
      std::unordered_map<string, std::pair<int, string>>* output_edges,
      int engine_precision_mode, const string& device_name,
      std::shared_ptr<nvinfer1::IGpuAllocator> allocator, int cuda_gpu_id)
      : graph(inp_graph),
        output_names(output_node_names),
        subgraph_node_ids(subgraph_node_id_numbers),
        max_batch_size(max_supported_batch_size),
        max_workspace_size_bytes(max_consumed_workspace_size_bytes),
        graph_properties(current_graph_properties),
        output_edge_map(output_edges),
        precision_mode(engine_precision_mode),
        device_name_(device_name),
        allocator_(allocator),
        cuda_gpu_id_(cuda_gpu_id) {}
  tensorflow::Graph& graph;
  const std::vector<string>& output_names;
  const std::set<int>& subgraph_node_ids;
  size_t max_batch_size;
  size_t max_workspace_size_bytes;
  const tensorflow::grappler::GraphProperties& graph_properties;
  std::unordered_map<string, std::pair<int, string>>* output_edge_map;
  int precision_mode;
  string device_name_;
  std::shared_ptr<nvinfer1::IGpuAllocator> allocator_;
  int cuda_gpu_id_;
  std::vector<std::pair<int, int>> subgraph_inputs;
  std::vector<std::pair<int, int>> subgraph_outputs;
  tensorflow::EdgeSet subgraph_incoming_edges;
  tensorflow::EdgeSet subgraph_outgoing_edges;
};

static tensorflow::Status FillSubGraphEdgeSets(ConvertGraphParams* p) {
  GetSubGraphIncomingEdges(p->graph, p->subgraph_node_ids,
                           &p->subgraph_incoming_edges);

  std::set<std::pair<int, int>> unique_tensors;
  // Add only unique input source nodes. If output of an outside node is shared
  // between multiple nodes inside the engine, only one edge should be created
  for (const tensorflow::Edge* edge : p->subgraph_incoming_edges) {
    unique_tensors.insert({edge->src()->id(), edge->src_output()});
  }
  p->subgraph_inputs.insert(p->subgraph_inputs.begin(), unique_tensors.begin(),
                            unique_tensors.end());
  GetSubGraphOutgoingEdges(p->graph, p->subgraph_node_ids,
                           &p->subgraph_outgoing_edges);
  unique_tensors.clear();
  // Similar to above, if multiple ouside nodes are sharing the output of an
  // internal node only one output port should be created and shared between
  // outputs
  for (const tensorflow::Edge* edge : p->subgraph_outgoing_edges) {
    unique_tensors.insert({edge->src()->id(), edge->src_output()});
  }
  p->subgraph_outputs.reserve(unique_tensors.size());
  p->subgraph_outputs.insert(p->subgraph_outputs.begin(),
                             unique_tensors.begin(), unique_tensors.end());
  return tensorflow::Status::OK();
}

tensorflow::Status GetCalibNode(ConvertGraphParams* params) {
  TF_RETURN_IF_ERROR(FillSubGraphEdgeSets(params));
  tensorflow::NodeDef trt_node_def;
  SubGraphParams s(params->graph, params->subgraph_node_ids,
                   params->subgraph_inputs, params->subgraph_outputs,
                   params->max_batch_size, params->max_workspace_size_bytes,
                   params->graph_properties, params->output_edge_map,
                   &trt_node_def, params->precision_mode, params->device_name_,
                   params->allocator_, params->cuda_gpu_id_);
  TF_RETURN_IF_ERROR(InjectCalibrationNode(s));
  tensorflow::Status status;
  tensorflow::Node* trt_node = params->graph.AddNode(trt_node_def, &status);

  TF_RETURN_IF_ERROR(status);

  for (auto in_edge :
       params->subgraph_incoming_edges) {  // loop over incoming edges and
                                           // attach them to calib node
    auto src_output = in_edge->src_output();
    auto dst_node = in_edge->dst();
    auto dst_input = in_edge->dst_input();
    VLOG(0) << " update edge " << trt_node->name() << ":" << src_output
            << " -> " << dst_node->name() << ":" << dst_input;
    TF_RETURN_IF_ERROR(
        params->graph.UpdateEdge(trt_node, src_output, dst_node, dst_input));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSubGraphToTensorRT(ConvertGraphParams* params) {
  TF_RETURN_IF_ERROR(FillSubGraphEdgeSets(params));
  tensorflow::NodeDef trt_node_def;

  SubGraphParams s(params->graph, params->subgraph_node_ids,
                   params->subgraph_inputs, params->subgraph_outputs,
                   params->max_batch_size, params->max_workspace_size_bytes,
                   params->graph_properties, params->output_edge_map,
                   &trt_node_def, params->precision_mode, params->device_name_,
                   params->allocator_, params->cuda_gpu_id_);
  TF_RETURN_IF_ERROR(ConvertSubGraphToTensorRTNodeDef(s));
  tensorflow::Status status;
  tensorflow::Node* trt_node = params->graph.AddNode(trt_node_def, &status);

  // AddNode does not wire edges.
  // Re-map incoming edges to use the new TRT node instead of the orig subgraph
  std::map<std::pair<int, int>, int> subgraph_edge_to_input_map;
  for (size_t i = 0; i < params->subgraph_inputs.size(); ++i) {
    subgraph_edge_to_input_map.insert({params->subgraph_inputs.at(i), i});
  }
  std::set<std::pair<int, int>> unique_tensors;
  for (const tensorflow::Edge* edge : params->subgraph_incoming_edges) {
    std::pair<int, int> old_src = {edge->src()->id(), edge->src_output()};
    if (unique_tensors.count(old_src)) continue;
    unique_tensors.insert(old_src);
    int new_src_output = subgraph_edge_to_input_map.at(old_src);
    params->graph.AddEdge(edge->src(), edge->src_output(), trt_node,
                          new_src_output);
    VLOG(1) << "Wire " << edge->src()->name() << ":" << edge->src_output()
            << " -> " << trt_node->name() << ":" << new_src_output;
    params->graph.RemoveEdge(edge);
  }
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "new edge count: " << trt_node->in_edges().size();
    for (const tensorflow::Edge* edge : trt_node->in_edges()) {
      VLOG(2) << edge->src()->name() << " port: " << edge->src_output();
    }
  }
  TF_RETURN_IF_ERROR(status);

  // Re-map outgoing edges to use the new TRT node instead of the orig subgraph
  std::map<std::pair<int, int>, int> subgraph_edge_to_output_map;
  for (size_t i = 0; i < params->subgraph_outputs.size(); ++i) {
    subgraph_edge_to_output_map.insert({params->subgraph_outputs.at(i), i});
  }
  TF_RETURN_IF_ERROR(status);
  for (const tensorflow::Edge* edge : params->subgraph_outgoing_edges) {
    std::pair<int, int> old_src = {edge->src()->id(), edge->src_output()};
    int new_src_output = subgraph_edge_to_output_map.at(old_src);
    TF_RETURN_IF_ERROR(params->graph.UpdateEdge(
        trt_node, new_src_output, edge->dst(), edge->dst_input()));
    VLOG(1) << "Wire " << trt_node->name() << ":" << new_src_output << " -> "
            << edge->dst()->name() << ":" << edge->dst_input();
  }
  // Remove the original subgraph
  for (int node_id : params->subgraph_node_ids) {
    tensorflow::Node* node = params->graph.FindNodeId(node_id);
    // Don't remove the input placeholders
    if (node->type_string() == "Placeholder") {
      continue;
    }
    params->graph.RemoveNode(node);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status BuildNodeMap(
    const tensorflow::Graph& graph,
    std::unordered_map<string, tensorflow::Node*>* node_map) {
  for (auto* node : graph.op_nodes()) {
    if (!node_map->insert({node->name(), node}).second) {
      return tensorflow::errors::AlreadyExists(
          "Node name is not unique in graph: " + node->name());
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace
tensorflow::Status ConvertCalibGraphToInferGraph(
    const tensorflow::GraphDef& graph_def, tensorflow::GraphDef* infer_graph) {
  VLOG(0) << "Starting Calib Conversion";
  infer_graph->CopyFrom(graph_def);
  auto trt_rm = tensorflow::tensorrt::TRTResourceManager::instance();
  auto calib_rm = trt_rm->getManager("TRTCalibration");
  int num_nodes=infer_graph->node_size();
  for (int i=0;i<num_nodes;++i){
    auto n=infer_graph->mutable_node(i);
    if (n->op() == "TRTEngineOp") {
      VLOG(1)<<"Processing "<<n->name();
      string container_name = n->attr().at("segment_funcdef_name").s();
      tensorflow::tensorrt::TRTCalibrationResource* cres = nullptr;
      auto status = calib_rm->Lookup(container_name, "Calibrator", &cres);
      if (!status.ok()) {
        LOG(ERROR) << "Could not get Calibration information. Did you run with "
                      "calibration data?";
        return tensorflow::errors::FailedPrecondition(
            "Need to run graph with calibration data first!");
      }
      if (cres->calibrator_) {
        cres->calibrator_->setDone();
        cres->thr_->join();
        auto calibration_table =
            cres->calibrator_->getCalibrationTableAsString();
        if (!calibration_table.size()) {
          LOG(ERROR) << "Calibration table is empty";
          return tensorflow::errors::Unknown(
              "Calibration table is missing. This shouldn't have happened!");
        }
        n->mutable_attr()->at("calibration_data").set_s(calibration_table);
      } else {
        LOG(ERROR) << "Can't get TRTCalibrator from resource manager!";
        return tensorflow::errors::Unknown(
            "Can't get TRTCalibrator from resource manager!");
      }
      cres->Unref();
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertGraphDefToTensorRT(
    const tensorflow::GraphDef& graph_def,
    const std::vector<string>& output_names, size_t max_batch_size,
    size_t max_workspace_size_bytes, tensorflow::GraphDef* new_graph_def,
    int precision_mode, int minimum_segment_size, bool is_dyn_op,
    int max_cached_engines, std::vector<int> cached_engine_batches) {
  // optimization pass
  tensorflow::grappler::GrapplerItem item;
  item.fetch = output_names;
  item.graph = graph_def;

  tensorflow::DeviceProperties device_properties;
  device_properties.set_type("GPU");
  device_properties.mutable_environment()->insert({"architecture", "6"});
  tensorflow::grappler::Cluster* cluster =
      new tensorflow::grappler::VirtualCluster({{"/GPU:0", device_properties}});

  // single machine
  int num_cpu_cores = tensorflow::grappler::GetNumAvailableLogicalCPUCores();
  int num_gpus = tensorflow::grappler::GetNumAvailableGPUs();
  VLOG(2) << "cpu_cores: " << num_cpu_cores;
  VLOG(2) << "gpus: " << num_gpus;
  tensorflow::RewriterConfig rw_cfg;
  tensorflow::grappler::MetaOptimizer meta_opt(nullptr, rw_cfg);
  tensorflow::GraphDef gdef;
  TF_RETURN_IF_ERROR(meta_opt.Optimize(cluster, item, &gdef));
  item.graph = gdef;

  // AJ refactoring shape inference through grappler/GraphProperties.
  tensorflow::grappler::GraphProperties static_graph_properties(item);
  TF_RETURN_IF_ERROR(static_graph_properties.InferStatically(true));
  // Build full graph
  ConversionParams cp;
  cp.input_graph_def = &gdef;
  cp.output_names = &output_names;
  cp.max_batch_size = max_batch_size;
  cp.output_graph_def = new_graph_def;
  cp.precision_mode = precision_mode;
  cp.is_dyn_op = is_dyn_op;
  cp.max_cached_engines = max_cached_engines;
  cp.cached_engine_batches = cached_engine_batches;
  cp.minimum_segment_size = minimum_segment_size;
  cp.graph_properties = &static_graph_properties;
  cp.max_workspace_size_bytes = max_workspace_size_bytes;
  // return ConvertAfterShapes(gdef, output_names, max_batch_size,
  //                           max_workspace_size_bytes, new_graph_def,
  //                           precision_mode, minimum_segment_size,
  //                           static_graph_properties, nullptr);
  return ConvertAfterShapes(cp);
}

EngineInfo GetEngineInfo(
    const tensorflow::Graph* g,
    const tensorflow::grappler::GraphProperties& graph_properties,
    const std::set<string>& segment_nodes,
    const std::unordered_map<string, tensorflow::Node*>& node_map,
    const std::vector<tensorflow::Node*>& topological_order) {
  std::vector<int> subgraph_node_ids;
  EngineInfo info;
  std::set<string> segment_devices;
  int input_port = 0;
  int output_port = 0;
  std::unordered_map<string, int> created_edges;
  for (auto it = topological_order.rbegin(); it != topological_order.rend();
       ++it) {
    auto node_name = (*it)->name();

    if (segment_nodes.count(node_name) == 0) continue;
    auto node = node_map.at(node_name);
    auto node_device = node->requested_device();
    if (!node_device.empty()) {
      segment_devices.insert(node_device);
    }
    int node_id = node->id();
    subgraph_node_ids.push_back(node_id);
    for (const auto edge : node->in_edges()) {
      auto input_node = edge->src();
      if (segment_nodes.count(input_node->name()) == 0) {
        if (input_node->type_string() ==
            "Const") {  // Add constant input into segment
          subgraph_node_ids.push_back(input_node->id());
        } else if (!edge->IsControlEdge() && !input_node->IsSource()) {
          string s(input_node->name());
          StrAppend(&s, ":", edge->src_output());
          VLOG(1) << "Input edge = " << s;
          int port = input_port;
          if (created_edges.count(s)) {
            port = created_edges.at(s);
          } else {
            created_edges.insert({s, port});
            input_port++;
          }
          info.connections.emplace_back(input_node->name(), input_node->id(),
                                        edge->src_output(), node_name, node_id,
                                        edge->dst_input(), true, port);
        }
      }
    }
    for (const auto edge : node->out_edges()) {
      auto output_node = edge->dst();
      if (segment_nodes.count(output_node->name()) == 0 &&
          !edge->IsControlEdge() && !output_node->IsSink()) {
        string s(node_name);
        StrAppend(&s, ":", edge->src_output());
        VLOG(1) << "Output edge = " << s;
        int port = output_port;
        if (created_edges.count(s)) {
          port = created_edges.at(s);
        } else {
          created_edges.insert({s, port});
          output_port++;
        }
        info.connections.emplace_back(output_node->name(), output_node->id(),
                                      edge->dst_input(), node_name, node_id,
                                      edge->src_output(), false, port);
      }
    }
  }

  ConvertSegmentToGraphDef(g, graph_properties, subgraph_node_ids,
                           &info.connections, &info.segment_graph_def,
                           &info.engine_name);
  info.engine_type = EngineInfo::EngineType::TRTStatic;
  if (segment_devices.size() > 1) {
    LOG(WARNING) << "Detected multiple(" << segment_devices.size()
                 << ") devices for the segment. Picking first one to continue "
                 << "but this shouldn't have happened";
    info.device = *segment_devices.begin();
  }
  return info;
}

tensorflow::Status CreateTRTNode(tensorflow::Graph* graph,
                                 const std::vector<EngineInfo>& infos, int pos,
                                 tensorflow::NodeDef* trtNode,
                                 nvinfer1::IGpuAllocator* alloc,
                                 int max_batch_size) {
  auto& info = infos.at(pos);
  std::vector<tensorflow::TensorShapeProto> out_shapes;
  std::vector<tensorflow::TensorShapeProto> input_shapes;
  std::vector<tensorflow::PartialTensorShape> shapes;
  std::vector<tensorflow::NodeDefBuilder::NodeOut> inputs;
  std::vector<tensorflow::DataType> out_types;
  VLOG(1) << "Processing " << info.engine_name;
  for (const auto conn : info.connections) {
    if (!conn.is_input_edge) {  // output edge
      tensorflow::TensorShapeProto out_shape;
      conn.inside_shape.AsProto(
          &out_shape);  // shape of the output node inside segment
      if (out_shapes.size() <= conn.port_number) {
        out_shapes.resize(conn.port_number + 1);
        out_types.resize(conn.port_number + 1);
      }
      out_shapes.at(conn.port_number) = out_shape;
      out_types.at(conn.port_number) = conn.inside_type;
      continue;
    } else {  // input edge
      tensorflow::TensorShapeProto in_shape;
      conn.outside_shape.AsProto(&in_shape);

      if (input_shapes.size() <= conn.port_number) {
        input_shapes.resize(conn.port_number + 1);
        shapes.resize(conn.port_number + 1);
      }
      input_shapes.at(conn.port_number) = in_shape;
      shapes.at(conn.port_number) = conn.outside_shape;
    }
    string input_node = conn.outside_node_name;
    int input_port = conn.outside_port;
    auto dtype =
        graph->FindNodeId(conn.outside_id)->output_type(conn.outside_port);
    bool found_engine = false;
    // Rewire the inputs to other engines if they contain original input node
    for (size_t t = 0; t < infos.size(); ++t) {
      if (t == pos) {
        continue;
      }
      auto& engine_info = infos.at(t);
      for (const auto& eng_conn : engine_info.connections) {
        if (eng_conn.is_input_edge) {
          continue;
        }
        if (eng_conn.inside_node_name == input_node) {
          input_node = engine_info.engine_name;
          if (eng_conn.inside_port == input_port) {
            input_port = eng_conn.port_number;
            found_engine = true;
            break;
          }
        }
      }
      if (found_engine) break;
    }
    VLOG(1) << "Engine Input " << input_node << ":" << input_port << " -> "
            << info.engine_name << ":" << inputs.size();
    bool new_input = true;
    for (const auto& inp : inputs) {
      if (inp.node == input_node && inp.index == input_port) {
        new_input = false;
        break;
      }
    }
    if (new_input) {
      inputs.emplace_back(input_node, input_port, dtype);
    }
  }
  string segment_string;
  if (info.engine_type == EngineInfo::EngineType::TRTStatic) {
    // add static engine creation here
    tensorflow::tensorrt::Logger trt_logger;
    auto builder = std::shared_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(trt_logger), [](nvinfer1::IBuilder* p) {
          if (p) p->destroy();
        });
    builder->setMaxBatchSize(max_batch_size);
    if (info.precision_mode == tensorflow::tensorrt::convert::FP16MODE) {
      builder->setHalf2Mode(true);
    }
    builder->setMaxWorkspaceSize(info.max_workspace_size_bytes);
    nvinfer1::ICudaEngine* engine = nullptr;
    // TODO(sami): What happens if 1st dim is not batch?
    auto status = ConvertSubgraphToEngine(info.segment_graph_def, builder.get(),
                                          shapes, &engine, info.precision_mode);
    if (!status.ok()) {
      LOG(ERROR) << "Engine conversion failed with " << status;
      return status;
    }
    if (engine) {
      auto engine_data = std::shared_ptr<nvinfer1::IHostMemory>(
          engine->serialize(), [](nvinfer1::IHostMemory* p) {
            if (p) p->destroy();
          });
      segment_string =
          string((const char*)engine_data->data(), engine_data->size());
      engine->destroy();
    }
  } else {
    segment_string = info.segment_graph_def.SerializeAsString();
  }
  string prec_string;
  switch (info.precision_mode) {
    case FP32MODE: {
      prec_string = "FP32";
      break;
    }
    case FP16MODE: {
      prec_string = "FP16";
      break;
    }
    case INT8MODE: {
      prec_string = "INT8";
      auto trt_rm = tensorflow::tensorrt::TRTResourceManager::instance();
      auto calib_rm = trt_rm->getManager("TRTCalibration");
      if (!calib_rm) {
        LOG(ERROR) << "Failed to construct calibration storage";
      }
      break;
    }
    default: {
      return tensorflow::errors::OutOfRange("Unknown precision mode");
    }
  }
  tensorflow::Status status;
  tensorflow::Node* engine_node = nullptr;
  tensorflow::NodeDefBuilder node_builder(info.engine_name, "TRTEngineOp");
  if (!info.device.empty()) {
    node_builder.Device(info.device);
  }
  if (VLOG_IS_ON(1)) {
    string ins(info.engine_name);
    for (const auto& ii : inputs) {
      StrAppend(&ins, ii.node, ":", ii.index, " ");
    }
    VLOG(1) << ins;
  }
  node_builder.Input(inputs);
  if (info.engine_type == EngineInfo::EngineType::TRTStatic) {
    if (info.cached_engine_batches.size()) {
      LOG(WARNING) << "Cached engine batches are ignored for static engines";
    }
  }
  status = node_builder.Attr("input_shapes", input_shapes)
               .Attr("output_shapes", out_shapes)
               .Attr("static_engine",
                     info.engine_type == EngineInfo::EngineType::TRTStatic)
               .Attr("segment_funcdef_name",
                     StrCat(info.engine_name, "_native_segment"))
               .Attr("serialized_segment", segment_string)
               .Attr("calibration_data", "")
               .Attr("max_cached_engines_count", info.maximum_cached_engines)
               .Attr("cached_engine_batches", {max_batch_size})
               .Attr("workspace_size_bytes", info.max_workspace_size_bytes)
               .Attr("precision_mode", prec_string)
               .Attr("OutT", out_types)
               .Finalize(trtNode);
  if (!status.ok()) {
    LOG(ERROR) << "Node construction failed with" << status;
    return status;
  }
  VLOG(1) << "Adding TRTEngine " << info.engine_name << " to graph";
  engine_node = graph->AddNode(*trtNode, &status);
  if (!status.ok()) {
    LOG(ERROR) << "Adding node failed " << status;
    return status;
  }

  for (auto& conn : info.connections) {
    if (conn.is_input_edge) continue;
    VLOG(1) << " Updating DBG " << engine_node->name() << " out_port "
            << conn.port_number << " out_id " << conn.outside_id
            << " name=" << conn.outside_node_name;
    auto dst_node = graph->FindNodeId(conn.outside_id);
    if (!dst_node) {  // node removed skip.
      continue;
    }
    VLOG(1) << "Updating " << engine_node->name() << ":" << conn.port_number
            << " to " << dst_node->name() << ":" << conn.outside_port;
    status = graph->UpdateEdge(engine_node, conn.port_number, dst_node,
                               conn.outside_port);
    if (!status.ok()) {
      LOG(ERROR) << "Edge update failed " << engine_node->name() << ":"
                 << conn.port_number << " -> " << dst_node->name() << ":"
                 << conn.outside_port << " status= " << status;
    }
  }
  return status;
}

// tensorflow::Status ConvertAfterShapes(
//     const tensorflow::GraphDef& gdef, const std::vector<string>&
//     output_names, size_t max_batch_size, size_t max_workspace_size_bytes,
//     tensorflow::GraphDef* new_graph_def, int precision_mode,
//     int minimum_segment_size,
//     const tensorflow::grappler::GraphProperties& graph_properties,
//     const tensorflow::grappler::Cluster* cluster) {
tensorflow::Status RegisterSegmentFunctionToFunctionLibrary(
    tensorflow::Graph* graph, const tensorflow::GraphDef& segment,
    const string& name) {
  tensorflow::Graph sgraph(graph->flib_def());
  tensorflow::GraphConstructorOptions gcopts;
  TF_RETURN_IF_ERROR(
      tensorflow::ConvertGraphDefToGraph(gcopts, segment, &sgraph));
  VLOG(1) << " SAMI OPNODES  ";
  std::map<string, tensorflow::Node*> io_nodes;
  int num_inputs = 0;
  for (auto n : sgraph.op_nodes()) {
    VLOG(1) << n->type_string();
    if (tensorflow::str_util::StartsWith(n->name(), "InputPH_")) {
      num_inputs++;
      io_nodes.insert({n->name(), n});
    } else if (tensorflow::str_util::StartsWith(n->name(), "OutputPH_")) {
      io_nodes.insert({n->name(), n});
    }
  }
  for (int i = 0; i < num_inputs; ++i) {
    auto name = StrCat("InputPH_", i);
    auto node = io_nodes[name];
    tensorflow::NodeDef nd;
    tensorflow::NodeDefBuilder node_builder(
        StrCat(name, "_Arg"), tensorflow::FunctionLibraryDefinition::kArgOp);
    VLOG(1) << "Adding " << StrCat(name, "_Arg");
    node_builder.Attr("T", node->output_type(0)).Attr("index", i).Finalize(&nd);
    tensorflow::Status s;
    auto nArg = sgraph.AddNode(nd, &s);
    if (!s.ok()) {
      LOG(ERROR) << "Couldn't add _Arg node for " << name;
    }
    for (auto edge : node->out_edges()) {
      sgraph.AddEdge(nArg, 0, edge->dst(), edge->dst_input());
      VLOG(1) << "Updating funcdef input " << nArg->name() << ":" << 0
              << " - > " << edge->dst()->name() << ":" << edge->dst_input();
      // s = sgraph.UpdateEdge(nArg, 0, edge->dst(), edge->dst_input());
      if (!s.ok()) {
        LOG(ERROR) << "Failed to update edge from " << nArg->name() << " to "
                   << edge->dst()->name() << ":" << edge->dst_input();
      }
    }
    sgraph.RemoveNode(node);
  }
  for (int i = 0; i < io_nodes.size() - num_inputs; ++i) {
    auto name = StrCat("OutputPH_", i);
    auto node = io_nodes[name];
    tensorflow::NodeDef nd;
    tensorflow::NodeDefBuilder node_builder(
        StrCat(name, "_Ret"), tensorflow::FunctionLibraryDefinition::kRetOp);
    auto edge = *(node->in_edges().begin());
    tensorflow::NodeDefBuilder::NodeOut nout(
        edge->src()->name(), edge->src_output(),
        edge->src()->output_type(edge->src_output()));
    VLOG(1) << " input " << nout.node << ":" << nout.index
            << " dtype=" << tensorflow::DataTypeString(nout.data_type);
    node_builder.Input({nout});
    node_builder.Attr("T", node->output_type(0)).Attr("index", i).Finalize(&nd);
    if (VLOG_IS_ON(3)) {
      VLOG(3) << nd.DebugString();
    }
    tensorflow::Status s;
    auto nRet = sgraph.AddNode(nd, &s);
    if (!s.ok()) {
      LOG(ERROR) << "Couldn't add _Ret node for " << name;
    }
    VLOG(1) << "Update edge from " << edge->src()->name() << ":"
            << edge->src_output() << " - > " << nRet->name() << ":" << 0;
    sgraph.AddEdge(edge->src(), edge->src_output(), nRet, 0);
    s = sgraph.UpdateEdge(edge->src(), edge->src_output(), nRet, 0);
    if (!s.ok()) {
      LOG(ERROR) << "Failed to update edge from " << edge->src()->name() << ":"
                 << edge->src_output() << " - > " << nRet->name() << ":" << 0;
    }
    sgraph.RemoveNode(node);
  }
  tensorflow::FunctionDefLibrary fdeflib;
  auto native_segment = fdeflib.add_function();
  TF_RETURN_IF_ERROR(tensorflow::GraphToFunctionDef(
      sgraph, StrCat(name, "_native_segment"), native_segment));
  // for (int i = 0; i < num_inputs; i++) {
  //   auto arg = native_segment->mutable_signature()->add_input_arg();
  //   arg->set_type(io_nodes[StrCat("InputPH_", i)]->output_type(0));
  //   arg->set_name(io_nodes[StrCat("InputPH_", i)]->name());
  // }
  // for (int i = 0; i < io_nodes.size() - num_inputs; ++i) {
  //   auto arg = native_segment->mutable_signature()->add_output_arg();
  //   arg->set_type(io_nodes[StrCat("OutputPH_", i)]->output_type(0));
  //   arg->set_name(io_nodes[StrCat("OutputPH_", i)]->name());
  //   (*native_segment->mutable_ret())[StrCat("OutputPH_", i)] =
  //       StrCat("OutputPH_", i, ":", 0);
  // }
  if (VLOG_IS_ON(3)) {
    VLOG(3) << name << " Function_Def ";
    VLOG(3) << native_segment->DebugString();
  }
  TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(fdeflib));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertAfterShapes(ConversionParams& params) {
  // Segment the graph into subgraphs that can be converted to TensorRT
  tensorflow::tensorrt::segment::SegmentOptions segment_options;
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             params.input_graph_def->library());
  tensorflow::Graph graph(flib);
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), *params.input_graph_def, &graph));

  // TODO(ben,jie,sami): exclude output nodes (DISCUSS IT)
  for (auto node : *(params.output_names)) {
    segment_options.exclude_node_list.insert(node);
  }

  segment_options.minimum_segment_size = params.minimum_segment_size;
  tensorflow::tensorrt::segment::SegmentNodesVector segments;
  TF_RETURN_IF_ERROR(tensorrt::segment::SegmentGraph(
      &graph, IsTensorRTCandidate, segment_options, &segments));
  if (segments.size() > 1) {
    VLOG(0) << "MULTIPLE tensorrt candidate conversion: " << segments.size();
  }
  std::unordered_map<string, tensorflow::Node*> node_map;
  TF_RETURN_IF_ERROR(BuildNodeMap(graph, &node_map));
  std::unordered_map<string, std::pair<int, string>> output_edge_map;
  float total_num_nodes_in_segments = 0.;
  std::vector<EngineInfo> engine_segments;
  engine_segments.reserve(segments.size());
  std::vector<tensorflow::Node*> topo_order;
  tensorflow::GetPostOrder(graph, &topo_order);
  size_t total_engine_size = 0;
  std::vector<size_t> engine_sizes;
  for (size_t t = 0; t < segments.size(); t++) {
    auto& s = segments.at(t);
    engine_segments.emplace_back(GetEngineInfo(&graph, *params.graph_properties,
                                               s.first, node_map, topo_order));
    auto& curr_engine = engine_segments.back();
    curr_engine.precision_mode = params.precision_mode;
    engine_sizes.push_back(curr_engine.segment_graph_def.ByteSizeLong());
    curr_engine.engine_type =
        (params.is_dyn_op || params.precision_mode == INT8MODE
             ? EngineInfo::EngineType::TRTDynamic
             : EngineInfo::EngineType::TRTStatic);
    curr_engine.cached_engine_batches = params.cached_engine_batches;
    curr_engine.maximum_cached_engines = params.max_cached_engines;
    total_engine_size += engine_sizes.back();
    total_num_nodes_in_segments += s.first.size();
    StrAppend(&curr_engine.engine_name, "my_trt_op_", t);
    RegisterSegmentFunctionToFunctionLibrary(
        &graph, curr_engine.segment_graph_def, curr_engine.engine_name);
    if (VLOG_IS_ON(8)) {
      string fname = curr_engine.engine_name;
      StrAppend(&fname, ".pb");
      std::fstream f;
      f.open(fname.c_str(), std::fstream::out | std::fstream::binary);
      f << engine_segments.at(t).segment_graph_def.SerializeAsString();
      f.close();
    }
  }
  std::vector<tensorflow::NodeDef*> trt_nodes;
  trt_nodes.reserve(engine_segments.size());
  int old_cuda_device = 0;
  cudaGetDevice(&old_cuda_device);
  for (int i = 0; i < engine_segments.size(); ++i) {
    auto trt_node = new tensorflow::NodeDef;
    trt_nodes.push_back(trt_node);
    auto& engine = engine_segments.at(i);
    // Partition the workspace size by the average of node ratio and segment
    // graphdef size
    engine.max_workspace_size_bytes =
        params.max_workspace_size_bytes *
        (engine_sizes.at(i) / total_engine_size +
         segments.at(i).first.size() / total_num_nodes_in_segments) /
        2.0;
    std::shared_ptr<nvinfer1::IGpuAllocator> alloc(new TRTCudaAllocator());
    int cuda_device_id = 0;
    if (params.cluster) {  // get allocator
      const auto device =
          params.cluster->GetDeviceSet()->FindDeviceByName(engine.device);
      if (device) {
        tensorflow::TfGpuId tf_gpu_id(device->parsed_name().id);
        CudaGpuId cuda_gpu_id;
        Status s = GpuIdManager::TfToCudaGpuId(tf_gpu_id, &cuda_gpu_id);
        if (!s.ok()) {
          LOG(ERROR) << "Cuda device identification failed, using device "
                        "0. Error= "
                     << s;
          cuda_device_id = 0;
        } else {
          cuda_device_id = cuda_gpu_id.value();
        }
        tensorflow::GPUOptions gpuoptions;
        // we need to us PM here since in python path there is no way to get
        // to allocators
        auto pm = tensorflow::ProcessState::singleton();
        // this should be instantiated by now
        auto dev_allocator = pm->GetGPUAllocator(gpuoptions, tf_gpu_id, 1);
        VLOG(0) << "Got an allocator for device tf_device=" << tf_gpu_id.value()
                << " cuda device= " << cuda_device_id << " at "
                << dev_allocator;
        alloc.reset(new TRTDeviceAllocator(dev_allocator));
      }
    }
    cudaSetDevice(cuda_device_id);
    CreateTRTNode(&graph, engine_segments, i, trt_node, alloc.get(),
                  params.max_batch_size);
    const auto& internal_nodes = segments.at(i).first;
    for (auto node_id : internal_nodes) {
      graph.RemoveNode(node_map.at(node_id));
    }
  }
  cudaSetDevice(old_cuda_device);
  graph.ToGraphDef(params.output_graph_def);
  for (auto tn : trt_nodes) delete tn;
  return tensorflow::Status::OK();
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
