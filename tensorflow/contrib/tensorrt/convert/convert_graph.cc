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

#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/contrib/tensorrt/convert/convert_nodes.h"
#include "tensorflow/contrib/tensorrt/segment/segment.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/layout_optimizer.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"  // NOLINT

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {
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
  return candidate_ops.count(node->type_string());
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
      } else {
        VLOG(2) << node->name() << " -> " << edge->src()->name() << " N, ";
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
        VLOG(2) << node->name() << " -> " << edge->dst()->name() << " Y, ";
        outgoing_edges->insert(edge);
      } else {
        VLOG(2) << node->name() << " -> " << edge->dst()->name() << " N, ";
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
      std::shared_ptr<nvinfer1::IGpuAllocator> allocator, int cuda_device_id)
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
        cuda_device_id_(cuda_device_id) {}
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
  int cuda_device_id_;
  std::vector<std::pair<int, int>> subgraph_inputs;
  std::vector<std::pair<int, int>> subgraph_outputs;
  tensorflow::EdgeSet subgraph_incoming_edges;
  tensorflow::EdgeSet subgraph_outgoing_edges;
};

static tensorflow::Status FillSubGraphEdgeSets(ConvertGraphParams* p) {
  GetSubGraphIncomingEdges(p->graph, p->subgraph_node_ids,
                           &p->subgraph_incoming_edges);
  for (const tensorflow::Edge* edge : p->subgraph_incoming_edges) {
    p->subgraph_inputs.push_back({edge->src()->id(), edge->src_output()});
  }
  auto output_name_to_index_map = BuildTensorNameMap(p->output_names);
  std::set<std::pair<int, int>> subgraph_outputs_set;
  // Collect outputs referenced from output_names
  for (int node_id : p->subgraph_node_ids) {
    tensorflow::Node* node = p->graph.FindNodeId(node_id);
    if (output_name_to_index_map.count(node->name())) {
      for (int index : output_name_to_index_map.at(node->name())) {
        subgraph_outputs_set.insert({node_id, index});
      }
    }
  }
  GetSubGraphOutgoingEdges(p->graph, p->subgraph_node_ids,
                           &p->subgraph_outgoing_edges);
  for (const tensorflow::Edge* edge : p->subgraph_outgoing_edges) {
    subgraph_outputs_set.insert({edge->src()->id(), edge->src_output()});
  }
  p->subgraph_outputs.reserve(subgraph_outputs_set.size());
  p->subgraph_outputs.insert(p->subgraph_outputs.begin(),
                             subgraph_outputs_set.begin(),
                             subgraph_outputs_set.end());
  return tensorflow::Status::OK();
};

tensorflow::Status GetCalibNode(ConvertGraphParams* params) {
  TF_RETURN_IF_ERROR(FillSubGraphEdgeSets(params));
  tensorflow::NodeDef trt_node_def;
  SubGraphParams s(params->graph, params->subgraph_node_ids,
                   params->subgraph_inputs, params->subgraph_outputs,
                   params->max_batch_size, params->max_workspace_size_bytes,
                   params->graph_properties, params->output_edge_map,
                   &trt_node_def, params->precision_mode, params->device_name_,
                   params->allocator_, params->cuda_device_id_);
  TF_RETURN_IF_ERROR(InjectCalibrationNode(s));
  tensorflow::Status status;
  tensorflow::Node* trt_node = params->graph.AddNode(trt_node_def, &status);

  TF_RETURN_IF_ERROR(status);

  for (auto in_edge :
       params->subgraph_incoming_edges) {  // loop over incoming edges and
                                           // attach them to calib node
    // tensorflow::Node* src_node = in_edge->src();
    auto src_output = in_edge->src_output();
    auto dst_node = in_edge->dst();
    auto dst_input = in_edge->dst_input();
    VLOG(1) << " update edge " << trt_node->name() << ":" << src_output
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
                   params->allocator_, params->cuda_device_id_);
  TF_RETURN_IF_ERROR(ConvertSubGraphToTensorRTNodeDef(s));
  tensorflow::Status status;
  tensorflow::Node* trt_node = params->graph.AddNode(trt_node_def, &status);

  // AddNode does not wire edges.
  // Re-map incoming edges to use the new TRT node instead of the orig subgraph
  std::map<std::pair<int, int>, int> subgraph_edge_to_input_map;
  for (size_t i = 0; i < params->subgraph_inputs.size(); ++i) {
    subgraph_edge_to_input_map.insert({params->subgraph_inputs.at(i), i});
  }
  for (const tensorflow::Edge* edge : params->subgraph_incoming_edges) {
    std::pair<int, int> old_src = {edge->src()->id(), edge->src_output()};
    int new_src_output = subgraph_edge_to_input_map.at(old_src);
    params->graph.AddEdge(edge->src(), edge->src_output(), trt_node,
                          new_src_output);
    params->graph.RemoveEdge(edge);
  }

  VLOG(2) << "new wiring edges: " << trt_node->in_edges().size();
  for (const tensorflow::Edge* edge : trt_node->in_edges()) {
    VLOG(2) << edge->src()->name() << " port: " << edge->src_output();
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
  tensorflow::Graph graph(tensorflow::OpRegistry::Global());
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), graph_def, &graph));
  //  get calib nodes
  std::vector<tensorflow::Node*> calib_nodes;
  for (auto node : graph.op_nodes()) {
    if (node->type_string() == "TRTCalibOp") {
      VLOG(1) << "Found Calib Node";
      calib_nodes.push_back(node);
    }
  }
  VLOG(0) << "Num Calib nodes in graph= " << calib_nodes.size();
  if (calib_nodes.size() == 0)
    return tensorflow::errors::FailedPrecondition(
        "Graph doesn't contain any calibration nodes!."
        " Please generate calibration graph and run calibration first");
  for (auto n : calib_nodes) {
    TF_RETURN_IF_ERROR(
        tensorrt::convert::ConvertCalibrationNodeToEngineNode(graph, n));
  }
  graph.ToGraphDef(infer_graph);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertGraphDefToTensorRT(
    const tensorflow::GraphDef& graph_def,
    const std::vector<string>& output_names, size_t max_batch_size,
    size_t max_workspace_size_bytes, tensorflow::GraphDef* new_graph_def,
    int precision_mode = FP32MODE, int minimum_segment_size = 3) {
  // optimization pass
  tensorflow::grappler::GrapplerItem item;
  item.fetch = output_names;
  tensorflow::GraphDef gdef;

  // Layout optimization
  item.graph = graph_def;
  tensorflow::grappler::LayoutOptimizer optimizer;
  tensorflow::grappler::Cluster* cluster;

  // virtual cluster
  tensorflow::DeviceProperties device_properties;

  device_properties.set_type("GPU");
  device_properties.mutable_environment()->insert({"architecture", "6"});
  cluster =
      new tensorflow::grappler::VirtualCluster({{"/GPU:0", device_properties}});

  // single machine
  int num_cpu_cores = tensorflow::grappler::GetNumAvailableLogicalCPUCores();
  int num_gpus = tensorflow::grappler::GetNumAvailableGPUs();
  VLOG(2) << "cpu_cores: " << num_cpu_cores;
  VLOG(2) << "gpus: " << num_gpus;
  tensorflow::RewriterConfig rw_cfg;
  tensorflow::grappler::MetaOptimizer meta_opt(nullptr, rw_cfg);
  TF_RETURN_IF_ERROR(meta_opt.Optimize(cluster, item, &gdef));
  item.graph = gdef;

  // AJ refactoring shape inference through grappler/GraphProperties.
  tensorflow::grappler::GraphProperties static_graph_properties(item);
  TF_RETURN_IF_ERROR(static_graph_properties.InferStatically(true));
  // Build full graph

  return ConvertAfterShapes(gdef, output_names, max_batch_size,
                            max_workspace_size_bytes, new_graph_def,
                            precision_mode, minimum_segment_size,
                            static_graph_properties, nullptr);
}

tensorflow::Status ConvertAfterShapes(
    const tensorflow::GraphDef& gdef, const std::vector<string>& output_names,
    size_t max_batch_size, size_t max_workspace_size_bytes,
    tensorflow::GraphDef* new_graph_def, int precision_mode,
    int minimum_segment_size,
    const tensorflow::grappler::GraphProperties& graph_properties,
    const tensorflow::grappler::Cluster* cluster) {
  // Segment the graph into subgraphs that can be converted to TensorRT
  tensorflow::tensorrt::segment::SegmentOptions segment_options;
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             gdef.library());
  tensorflow::Graph graph(flib);
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), gdef, &graph));

  // TODO(ben,jie,sami): exclude output nodes (DISCUSS IT)
  for (auto node : output_names) {
    segment_options.exclude_node_list.insert(node);
  }

  // TODO(sami): this should be passed as a knob!!!!
  segment_options.minimum_segment_size = minimum_segment_size;
  tensorflow::tensorrt::segment::SegmentNodesVector segments;
  TF_RETURN_IF_ERROR(tensorrt::segment::SegmentGraph(
      &graph, IsTensorRTCandidate, segment_options, &segments));
  if (segments.size() > 1) {
    VLOG(0) << "MULTIPLE tensorrt candidate conversion: " << segments.size();
  }
  std::unordered_map<string, tensorflow::Node*> node_map;
  TF_RETURN_IF_ERROR(BuildNodeMap(graph, &node_map));
  std::unordered_map<string, std::pair<int, string>> output_edge_map;
  int count = 0;
  float total_num_nodes_in_segments = 0.;
  for (auto s : segments) {
    total_num_nodes_in_segments += s.first.size();
  }
  std::map<string, tensorflow::Device*> name_to_device_map;
  if (cluster) {
    for (const auto dm : cluster->GetDeviceSet()->devices()) {
      name_to_device_map[dm->name()] = dm;
    }
  }
  for (const auto& segment_nodes_and_device : segments) {
    const std::set<string>& subgraph_node_names =
        segment_nodes_and_device.first;
    std::set<int> subgraph_node_ids;
    size_t max_mem_per_engine =
        max_workspace_size_bytes *
        ((float)subgraph_node_names.size() / total_num_nodes_in_segments);
    std::stringstream oss;
    for (const string& node_name : subgraph_node_names) {
      oss << " " << node_name;
      subgraph_node_ids.insert(node_map.at(node_name)->id());
    }
    VLOG(1) << "Subgraph nodes at device " << segment_nodes_and_device.second
            << " : " << oss.str();
    auto target_device =
        name_to_device_map.find(segment_nodes_and_device.second);
    std::shared_ptr<nvinfer1::IGpuAllocator> allocator(0);

    int cuda_device_id = 0;
    if (target_device != name_to_device_map.end()) {
      tensorflow::TfGpuId tf_gpu_id(target_device->second->parsed_name().id);
      CudaGpuId cuda_gpu_id;
      Status s = GpuIdManager::TfToCudaGpuId(tf_gpu_id, &cuda_gpu_id);
      if (!s.ok()) {
        LOG(ERROR)
            << "Cuda device identification failed, using device 0. Error= "
            << s;
      } else {
        cuda_device_id = cuda_gpu_id.value();
      }
      tensorflow::GPUOptions gpuoptions;
      auto pm = tensorflow::ProcessState::singleton();
      // this should be instantiated by now
      auto dev_allocator = pm->GetGPUAllocator(gpuoptions, tf_gpu_id, 1);
      VLOG(1) << "Got an allocator for device tf_device=" << tf_gpu_id.value()
              << " cuda device= " << cuda_device_id << " at " << dev_allocator;
      allocator = std::make_shared<TRTDeviceAllocator>(dev_allocator);
    } else {  // device unknown or not available
      allocator = std::make_shared<TRTCudaAllocator>();
    }
    ConvertGraphParams p(graph, output_names, subgraph_node_ids, max_batch_size,
                         max_mem_per_engine, graph_properties, &output_edge_map,
                         precision_mode, segment_nodes_and_device.second,
                         allocator, cuda_device_id);
    if (precision_mode == INT8MODE) {
      tensorflow::Status status = GetCalibNode(&p);
      if (status != tensorflow::Status::OK()) {
        LOG(WARNING) << "subgraph conversion error for subgraph_index:" << count
                     << " due to: \"" << status.ToString()
                     << "\" SKIPPING......( " << subgraph_node_names.size()
                     << " nodes)";
      }
    } else {
      tensorflow::Status status = ConvertSubGraphToTensorRT(&p);
      if (status != tensorflow::Status::OK()) {
        LOG(WARNING) << "subgraph conversion error for subgraph_index:" << count
                     << " due to: \"" << status.ToString()
                     << "\" SKIPPING......( " << subgraph_node_names.size()
                     << " nodes)";
      }
    }
    count++;
  }
  graph.ToGraphDef(new_graph_def);
  return tensorflow::Status::OK();
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
