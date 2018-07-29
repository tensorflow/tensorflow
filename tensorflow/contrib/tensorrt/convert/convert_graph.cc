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

#include <fstream>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/contrib/tensorrt/convert/convert_nodes.h"
#include "tensorflow/contrib/tensorrt/convert/utils.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resource_manager.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resources.h"
#include "tensorflow/contrib/tensorrt/segment/segment.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
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
#include "tensorflow/core/protobuf/config.pb.h"  // NOLINT
#include "tensorflow/core/protobuf/device_properties.pb.h"  // NOLINT
#include "tensorflow/core/protobuf/rewriter_config.pb.h"  // NOLINT
#include "tensorflow/core/util/device_name_utils.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"
namespace tensorflow {
namespace tensorrt {
namespace convert {
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

// Returns compiled TRT version information {Maj, Min, Patch}
std::vector<int> GetLinkedTensorRTVersion() {
  return {NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH};
}

// Returns loaded TRT library version {Maj, Min, Patch}
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
    "Div",
    "RealDiv",
    "Rsqrt",
    "Reciprocal",
    "Exp",
    "Log",
    "Sqrt",
    "Abs",
    "Neg",
#if NV_TENSORRT_MAJOR > 3
    "MatMul",
    "BatchMatMul",
    "Softmax",
    "Minimum",
    "Maximum",
    "TopKV2",
    "Sum",
    "Prod",
    "Max",
    "Min",
#endif
    // TODO(ben,jie): ...
  };
  // LINT.ThenChange(//tensorflow/contrib/tensorrt/convert/convert_nodes.cc)
  return (candidate_ops.count(node->type_string()) ||
          PluginFactoryTensorRT::GetInstance()->IsPlugin(node->type_string()));
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

// Function to get calibration from ResourceMgr and put them into nodedef.
tensorflow::Status ConvertCalibGraphToInferGraph(
    const tensorflow::GraphDef& graph_def, tensorflow::GraphDef* infer_graph,
    bool is_dyn_op) {
  VLOG(0) << "Starting Calib Conversion";
  infer_graph->CopyFrom(graph_def);
  auto trt_rm = TRTResourceManager::instance();
  auto calib_rm = trt_rm->getManager("TRTCalibration");
  int num_nodes = infer_graph->node_size();
  if (!is_dyn_op) {
    LOG(WARNING) << "Construction of static int8 engine is not implemented "
                    "yet!. Dynamic engine will be constructed";
  }
  for (int i = 0; i < num_nodes; ++i) {
    auto n = infer_graph->mutable_node(i);
    if (n->op() == "TRTEngineOp") {
      VLOG(1) << "Processing " << n->name();
      const string& container_name = n->attr().at("segment_funcdef_name").s();
      TRTCalibrationResource* cres = nullptr;
      auto status = calib_rm->Lookup(container_name, "Calibrator", &cres);
      if (!status.ok()) {
        LOG(ERROR) << "Could not get Calibration information. Did you run with "
                      "calibration data?";
        return tensorflow::errors::FailedPrecondition(
            "Need to run graph with calibration data first!");
      }
      if (cres->calibrator_) {
        cres->calibrator_->waitAndSetDone();
        cres->thr_->join();
        const auto& calibration_table =
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
      TF_RETURN_IF_ERROR(calib_rm->Cleanup(container_name));
    }
  }
  return tensorflow::Status::OK();
}

// Entry function from Python.
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
  // grappler requires a virtual cluster with a proper GPU device
  // in order to calculate flops>0 or fails with FATAL
  // We add numbers from a Pascal card here to have flops>0
  tensorflow::DeviceProperties device_properties;
  device_properties.set_type("GPU");
  device_properties.mutable_environment()->insert({"architecture", "6"});
  device_properties.set_num_cores(3584);
  device_properties.set_frequency(1531);
  std::unique_ptr<tensorflow::grappler::Cluster> cluster(
      new tensorflow::grappler::VirtualCluster(
          {{"/GPU:0", device_properties}}));

  // single machine
  int num_cpu_cores = tensorflow::grappler::GetNumAvailableLogicalCPUCores();
  int num_gpus = tensorflow::grappler::GetNumAvailableGPUs();
  VLOG(2) << "cpu_cores: " << num_cpu_cores;
  VLOG(2) << "gpus: " << num_gpus;
  tensorflow::RewriterConfig rw_cfg;
  // use only const folding and layout for the time being since new optimizers
  // break the graph for us
  rw_cfg.add_optimizers("constfold");
  rw_cfg.add_optimizers("layout");
  rw_cfg.set_meta_optimizer_iterations(tensorflow::RewriterConfig::ONE);
  tensorflow::grappler::MetaOptimizer meta_opt(nullptr, rw_cfg);
  tensorflow::GraphDef gdef;
  TF_RETURN_IF_ERROR(meta_opt.Optimize(cluster.get(), item, &gdef));
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
  if (VLOG_IS_ON(5)) {
    std::fstream f;
    f.open("TRTConversionInput.pb",
           std::fstream::out | std::fstream::binary | std::fstream::trunc);
    f << gdef.SerializeAsString();
    f.close();
  }
  return ConvertAfterShapes(cp);
}

// Function to get subsegment information structure.
tensorflow::Status GetEngineInfo(
    const tensorflow::Graph* g,
    const tensorflow::grappler::GraphProperties& graph_properties,
    const std::set<string>& segment_nodes,
    const std::unordered_map<string, tensorflow::Node*>& node_map,
    const std::vector<tensorflow::Node*>& reverse_topo_order,
    EngineInfo* info) {
  std::vector<int> subgraph_node_ids;
  std::set<int> added_const_node_ids;  // Used to prevent double insertion.
  std::set<string> segment_devices;
  int input_port = 0;
  int output_port = 0;

  // Map from src_node_name+port to the unique port numbers of the TRT op, where
  // the src_node_name is the name of the source node of the input/output
  // edge, thus there must not be any duplicates since source nodes of
  // input/output edges must be in different split of the graph.
  // TODO(aaroey): consider using node id and port instead.
  // TODO(aaroey): using topo order instead of reverting reverse topo order.
  std::unordered_map<string, int> created_edges;
  for (auto it = reverse_topo_order.rbegin(); it != reverse_topo_order.rend();
       ++it) {
    const auto& node_name = (*it)->name();

    if (segment_nodes.count(node_name) == 0) continue;
    auto node = node_map.at(node_name);
    auto node_device = node->requested_device();
    if (!node_device.empty()) {
      segment_devices.insert(node_device);
    } else {
      if (node->has_assigned_device_name()) {
        segment_devices.insert(node->assigned_device_name());
      } else {
        VLOG(2) << "Node " << node->name()
                << " neither have requested device nor assigned device";
      }
    }
    const int node_id = node->id();
    for (const auto edge : node->in_edges()) {
      auto input_node = edge->src();
      if (segment_nodes.count(input_node->name()) == 0 &&
          !edge->IsControlEdge() && !input_node->IsSource()) {
        // Add constant input node into the segment. We don't care if it has
        // other output edges going into other engines or TF nodes. Since we add
        // it only to the subsegment node list, not the subsegment itself, it
        // won't be removed from the graph. If it doesn't have any edges, TF
        // will prune it out.
        if (input_node->type_string() == "Const") {
          if (added_const_node_ids.count(input_node->id()) == 0) {
            added_const_node_ids.insert(input_node->id());
            subgraph_node_ids.push_back(input_node->id());
          }
        } else {
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
          info->connections.emplace_back(input_node->name(), input_node->id(),
                                         edge->src_output(), node_name, node_id,
                                         edge->dst_input(), true, port);
        }
      }
    }
    // We need to add possible const input nodes before adding this node in
    // order to keep the topological order.
    subgraph_node_ids.push_back(node_id);
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
        info->connections.emplace_back(output_node->name(), output_node->id(),
                                       edge->dst_input(), node_name, node_id,
                                       edge->src_output(), false, port);
      }
    }
  }

  TF_RETURN_IF_ERROR(ConvertSegmentToGraphDef(
      g, graph_properties, subgraph_node_ids, &info->connections,
      &info->segment_graph_def, &info->engine_name));
  // TODO(sami): This should not happen once segmenter is updated.
  if (segment_devices.size() == 1) {
    info->device = *segment_devices.begin();
  } else if (segment_devices.size() > 1) {
    LOG(WARNING) << "Detected multiple(" << segment_devices.size()
                 << ") devices for the segment. Picking first one to continue "
                 << "but this shouldn't have happened";
    info->device = *segment_devices.begin();
  } else {
    VLOG(1) << "Segment devices size is 0";
  }
  return Status::OK();
}

// Function to insert a TRT node into the graph. The graph is not modified if
// the returned status is not ok.
// 'alloc' is only used for creating static engine.
tensorflow::Status CreateTRTNode(tensorflow::Graph* graph,
                                 const std::vector<EngineInfo>& infos, int pos,
                                 nvinfer1::IGpuAllocator* alloc,
                                 int max_batch_size) {
  const auto& info = infos.at(pos);
  std::vector<tensorflow::TensorShapeProto> output_shape_protos;
  std::vector<tensorflow::TensorShapeProto> input_shape_protos;
  std::vector<tensorflow::PartialTensorShape> input_shapes;
  std::vector<tensorflow::NodeDefBuilder::NodeOut> inputs;
  std::vector<tensorflow::DataType> out_types;
  VLOG(1) << "Processing " << info.engine_name;

  // Update the shape and data types of input/output nodes, and find all unique
  // inputs.
  for (const auto& conn : info.connections) {
    if (!conn.is_input_edge) {
      // Set the shapes and data types of output edge.
      tensorflow::TensorShapeProto out_shape;
      // shape of the output node inside segment
      conn.inside_shape.AsProto(&out_shape);
      if (output_shape_protos.size() <= conn.port_number) {
        output_shape_protos.resize(conn.port_number + 1);
        out_types.resize(conn.port_number + 1);
      }
      output_shape_protos.at(conn.port_number) = out_shape;
      out_types.at(conn.port_number) = conn.connection_type;
      continue;
    }

    // Set the shapes and data types of input edge.
    tensorflow::TensorShapeProto in_shape;
    conn.outside_shape.AsProto(&in_shape);
    if (input_shape_protos.size() <= conn.port_number) {
      input_shape_protos.resize(conn.port_number + 1);
      input_shapes.resize(conn.port_number + 1);
    }
    input_shape_protos.at(conn.port_number) = in_shape;
    input_shapes.at(conn.port_number) = conn.outside_shape;

    string input_node = conn.outside_node_name;
    int input_port = conn.outside_port;
    bool found_engine = false;
    // Rewire the inputs to other engines if they contain original input node.
    // Note that we use the information of the engine here, not the information
    // of the created TRT nodes, so we're able to find all the connections to
    // any other engines beforehand.
    for (size_t t = 0; t < infos.size(); ++t) {
      if (t == pos) continue;
      auto& engine_info = infos.at(t);
      for (const auto& eng_conn : engine_info.connections) {
        if (eng_conn.is_input_edge) continue;
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
    // Skip duplicate inputs.
    // TODO(aaroey): use std::find instead. GetEngineInfo already remove
    // duplicate connections, so here we should never find any duplicate?
    bool new_input = true;
    for (const auto& inp : inputs) {
      if (inp.node == input_node && inp.index == input_port) {
        new_input = false;
        break;
      }
    }
    if (new_input) {
      inputs.emplace_back(input_node, input_port, conn.connection_type);
    }
  }

  // Build the engine and get its serialized representation.
  string segment_string;
  if (info.engine_type == EngineInfo::EngineType::TRTStatic ||
      info.precision_mode == INT8MODE) {
    // Create static engine for fp32/fp16 mode, and test validity of the engine
    // for int8 mode. We don't want engine to fail at the calibration time.
    // So we are constructing a FP32 engine here to check its validity, and if
    // it is a valid engine then we put the serialized graphdef to the op.
    // Otherwise we skip node creation for this engine.
    Logger trt_logger;
    TrtUniquePtrType<nvinfer1::ICudaEngine> engine;
    // TODO(sami): What happens if 1st dim is not batch?
    TF_RETURN_IF_ERROR(ConvertGraphDefToEngine(
        info.segment_graph_def,
        info.precision_mode == INT8MODE ? FP32MODE : info.precision_mode,
        max_batch_size, info.max_workspace_size_bytes, input_shapes,
        &trt_logger, alloc, /*calibrator=*/nullptr, &engine,
        /*convert_successfully=*/nullptr));
    TrtUniquePtrType<nvinfer1::IHostMemory> engine_data(engine->serialize());
    segment_string =
        string((const char*)engine_data->data(), engine_data->size());
    if (info.precision_mode == INT8MODE) {
      // See above comment about why not putting this inside the 'else' branch.
      segment_string = info.segment_graph_def.SerializeAsString();
    }
  } else {
    segment_string = info.segment_graph_def.SerializeAsString();
  }

  // TODO(aaroey): use enum instead, and add a helper method to do the
  // conversion.
  string prec_string;
  switch (info.precision_mode) {
    case FP32MODE:
      prec_string = "FP32";
      break;
    case FP16MODE:
      prec_string = "FP16";
      break;
    case INT8MODE:
      prec_string = "INT8";
      if (!TRTResourceManager::instance()->getManager("TRTCalibration")) {
        LOG(ERROR) << "Failed to construct calibration storage";
      }
      break;
    default:
      return tensorflow::errors::OutOfRange("Unknown precision mode");
  }
  tensorflow::NodeDefBuilder node_builder(info.engine_name, "TRTEngineOp");
  if (!info.device.empty()) node_builder.Device(info.device);
  if (VLOG_IS_ON(1)) {
    string ins = StrCat(info.engine_name, " inputs= ");
    for (const auto& ii : inputs) {
      StrAppend(&ins, ii.node, ":", ii.index, " ");
    }
    VLOG(1) << ins;
  }
  node_builder.Input(inputs);
  if (info.engine_type == EngineInfo::EngineType::TRTStatic &&
      info.cached_engine_batches.size()) {
    LOG(WARNING) << "Cached engine batches are ignored for static engines";
  }
  tensorflow::NodeDef trt_node;
  tensorflow::Status status =
      node_builder.Attr("input_shapes", input_shape_protos)
          .Attr("output_shapes", output_shape_protos)
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
          .Finalize(&trt_node);
  if (!status.ok()) {
    LOG(ERROR) << "Node construction failed with" << status;
    return status;
  }
  VLOG(1) << "Adding TRTEngine " << info.engine_name << " to graph";

  // Up until this point, graph is not modified. If we return !status.ok() from
  // here, this segment will be skipped
  tensorflow::Node* engine_node = graph->AddNode(trt_node, &status);
  if (!status.ok()) {
    LOG(ERROR) << "Adding node failed " << status;
    return status;
  }
  // Updates the inputs of output edges destination nodes, and point them to the
  // engine node.
  for (auto& conn : info.connections) {
    if (conn.is_input_edge) continue;
    VLOG(1) << " Updating DBG " << engine_node->name() << " out_port "
            << conn.port_number << " out_id " << conn.outside_id
            << " name=" << conn.outside_node_name;
    auto dst_node = graph->FindNodeId(conn.outside_id);
    // dst_node can only be removed if it is an input node of another engine.
    // In this case, other engines input edge is updated in nodedef to point to
    // this engine. Even though edge doesn't exists in the graph, when it is
    // deserialized again, correct edges will be constructed. This is a problem
    // of graph->AddNode().
    if (!dst_node) continue;
    VLOG(1) << "Updating " << engine_node->name() << ":" << conn.port_number
            << " to " << dst_node->name() << ":" << conn.outside_port;
    auto new_edge = graph->AddEdge(engine_node, conn.port_number, dst_node,
                                   conn.outside_port);
    CHECK(new_edge) << "Adding a new edge failed " << engine_node->name() << ":"
                    << conn.port_number << " -> " << dst_node->name() << ":"
                    << conn.outside_port;
  }
  return status;
}

// Function to construct a funcdef from the segment and add it to the graph.
tensorflow::Status RegisterSegmentFunctionToFunctionLibrary(
    tensorflow::Graph* graph, const tensorflow::GraphDef& segment,
    const string& name) {
  tensorflow::Graph sgraph(graph->flib_def());
  tensorflow::GraphConstructorOptions gcopts;
  TF_RETURN_IF_ERROR(
      tensorflow::ConvertGraphDefToGraph(gcopts, segment, &sgraph));
  std::map<string, tensorflow::Node*> io_nodes;
  int num_inputs = 0;
  for (auto n : sgraph.op_nodes()) {
    if (tensorflow::str_util::StartsWith(n->name(), kInputPHName)) {
      num_inputs++;
      io_nodes.insert({n->name(), n});
    } else if (tensorflow::str_util::StartsWith(n->name(), kOutputPHName)) {
      io_nodes.insert({n->name(), n});
    }
  }

  for (int i = 0; i < num_inputs; ++i) {
    auto name = StrCat(kInputPHName, i);
    auto node = io_nodes[name];
    tensorflow::NodeDef nd;
    tensorflow::NodeDefBuilder node_builder(
        StrCat(name, "_Arg"), tensorflow::FunctionLibraryDefinition::kArgOp);
    VLOG(1) << "Adding " << StrCat(name, "_Arg");
    TF_RETURN_IF_ERROR(node_builder.Attr("T", node->output_type(0))
                           .Attr("index", i)
                           .Finalize(&nd));
    tensorflow::Status s;
    auto node_arg = sgraph.AddNode(nd, &s);
    if (!s.ok()) {
      LOG(ERROR) << "Couldn't add _Arg node for " << name;
    }
    for (auto edge : node->out_edges()) {
      sgraph.AddEdge(node_arg, 0, edge->dst(), edge->dst_input());
      VLOG(1) << "Updating funcdef input " << node_arg->name() << ":" << 0
              << " - > " << edge->dst()->name() << ":" << edge->dst_input();
      if (!s.ok()) {
        LOG(ERROR) << "Failed to update edge from " << node_arg->name()
                   << " to " << edge->dst()->name() << ":" << edge->dst_input();
      }
    }
    sgraph.RemoveNode(node);
  }

  for (int i = 0; i < io_nodes.size() - num_inputs; ++i) {
    auto name = StrCat(kOutputPHName, i);
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
    // nvcc complains that Input(<brace-enclosed initializer list>) is
    // ambiguous, so do not use Input({nout}).
    node_builder.Input(nout);
    TF_RETURN_IF_ERROR(node_builder.Attr("T", node->output_type(0))
                           .Attr("index", i)
                           .Finalize(&nd));
    if (VLOG_IS_ON(3)) {
      VLOG(3) << nd.DebugString();
    }
    tensorflow::Status s;
    auto node_ret = sgraph.AddNode(nd, &s);
    if (!s.ok()) {
      LOG(ERROR) << "Couldn't add _Ret node for " << name;
    }
    VLOG(1) << "Update edge from " << edge->src()->name() << ":"
            << edge->src_output() << " - > " << node_ret->name() << ":" << 0;
    sgraph.AddEdge(edge->src(), edge->src_output(), node_ret, 0);
    s = sgraph.UpdateEdge(edge->src(), edge->src_output(), node_ret, 0);
    if (!s.ok()) {
      LOG(ERROR) << "Failed to update edge from " << edge->src()->name() << ":"
                 << edge->src_output() << " - > " << node_ret->name() << ":"
                 << 0;
    }
    sgraph.RemoveNode(node);
  }
  tensorflow::FunctionDefLibrary fdeflib;
  auto native_segment = fdeflib.add_function();
  TF_RETURN_IF_ERROR(tensorflow::GraphToFunctionDef(
      sgraph, StrCat(name, "_native_segment"), native_segment));
  if (VLOG_IS_ON(7)) {
    VLOG(7) << name << " Function_Def ";
    VLOG(7) << native_segment->DebugString();
  }
  VLOG(1) << "Adding funcdef to graphlib";
  TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(fdeflib));
  return tensorflow::Status::OK();
}

std::pair<int, tensorflow::Allocator*> GetDeviceAndAllocator(
    ConversionParams& params, EngineInfo& engine) {
  int cuda_device_id = -1;
  auto check_device_id = [](int tfid) -> int {
    tensorflow::TfGpuId tf_gpu_id(tfid);
    CudaGpuId cuda_gpu_id;
    Status s = GpuIdManager::TfToCudaGpuId(tf_gpu_id, &cuda_gpu_id);
    if (s.ok()) {
      VLOG(1) << "Found TF GPU " << tf_gpu_id.value() << " at cuda device "
              << cuda_gpu_id.value();
      return cuda_gpu_id.value();
    }
    VLOG(2) << "TF GPU with id " << tfid << " do not exist " << s;
    return -1;
  };
  tensorflow::Allocator* dev_allocator = nullptr;
  // we need to us PM here since in python path there is no way to get
  // to allocators.
  // TODO(sami): when grappler devices become available else path will not be
  // necessary
  auto pm = tensorflow::GPUProcessState::singleton();
  if (params.cluster) {  // get allocator
    tensorflow::Device* device = nullptr;
    if (params.cluster->GetDeviceSet()) {
      device = params.cluster->GetDeviceSet()->FindDeviceByName(engine.device);
    }
    if (device) {
      tensorflow::AllocatorAttributes alloc_attr;
      dev_allocator = device->GetAllocator(alloc_attr);
      VLOG(1) << "Using allocator " << dev_allocator->Name();
    } else {
      LOG(WARNING) << "Cluster is set but device '" << engine.device
                   << "' is not found in the cluster";
    }
  } else {  // cluster not found, possibly a python call
    VLOG(1) << "Cluster is not set, probably called from python";
    int found_device = 0;
    bool try_gpu_ids = true;
    // if device is set, try to find the device. Might be a problem for multi
    // host case but TensorRT do not support multi host setups yet.
    if (!engine.device.empty()) {
      DeviceNameUtils::ParsedName parsed_name;
      if (DeviceNameUtils::ParseFullName(engine.device, &parsed_name)) {
        cuda_device_id = parsed_name.has_id ? parsed_name.id : -1;
      }
      try_gpu_ids = !parsed_name.has_id;
    }
    if (try_gpu_ids) {
      while (found_device < 100) {
        cuda_device_id = check_device_id(found_device);
        if (cuda_device_id >= 0) break;
        found_device++;
      }
    }
    if (found_device == 100) {
      LOG(ERROR) << " Can't find a GPU device to work with. Please "
                    "instantiate a session to initialize devices";
      return std::make_pair(cuda_device_id, dev_allocator);
    }
    LOG(WARNING)
        << "Can't determine the device, constructing an allocator at device "
        << found_device;
    tensorflow::GPUOptions gpuoptions;
    // this will be a noop if device is already initialized
    gpuoptions.set_allow_growth(true);
    tensorflow::TfGpuId tf_gpu_id(found_device);
    dev_allocator = pm->GetGPUAllocator(gpuoptions, tf_gpu_id, 1);
  }
  return std::make_pair(cuda_device_id, dev_allocator);
}

// Entry function from optimization pass.
// TODO(aaeory): parameter should use pointer type.
tensorflow::Status ConvertAfterShapes(ConversionParams& params) {
  // Convert graphdef to graph.
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             params.input_graph_def->library());
  tensorflow::Graph graph(flib);
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), *params.input_graph_def, &graph));

  // Segment the graph into subgraphs that can be converted to TensorRT
  tensorflow::tensorrt::segment::SegmentOptions segment_options;
  // TODO(ben,jie,sami): exclude output nodes (DISCUSS IT)
  for (auto node : *(params.output_names)) {
    segment_options.exclude_node_list.insert(node);
  }
  segment_options.minimum_segment_size = params.minimum_segment_size;
  tensorflow::tensorrt::segment::SegmentNodesVector initial_segments;
  TF_RETURN_IF_ERROR(tensorrt::segment::SegmentGraph(
      &graph, IsTensorRTCandidate, InputEdgeValidator(*params.graph_properties),
      OutputEdgeValidator(), segment_options, &initial_segments));
  if (initial_segments.size() > 1) {
    VLOG(0) << "MULTIPLE tensorrt candidate conversion: "
            << initial_segments.size();
  }

  // Get the EngineInfo for each segment.
  std::unordered_map<string, tensorflow::Node*> node_map;
  TF_RETURN_IF_ERROR(BuildNodeMap(graph, &node_map));
  float total_num_nodes_in_segments = 0.;
  std::vector<EngineInfo> engine_segments;
  engine_segments.reserve(initial_segments.size());
  std::vector<tensorflow::Node*> reverse_topo_order;
  tensorflow::GetPostOrder(graph, &reverse_topo_order);
  size_t total_engine_bytes_size = 0;
  std::vector<size_t> engine_bytes_size;
  tensorflow::tensorrt::segment::SegmentNodesVector converted_segments;
  converted_segments.reserve(initial_segments.size());
  for (size_t t = 0; t < initial_segments.size(); t++) {
    auto& curr_segment = initial_segments.at(t);
    EngineInfo curr_engine;
    Status status =
        GetEngineInfo(&graph, *params.graph_properties, curr_segment.first,
                      node_map, reverse_topo_order, &curr_engine);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to get engine info for segment " << t << ": "
                   << status;
      continue;
    }
    curr_engine.precision_mode = params.precision_mode;
    curr_engine.engine_type =
        (params.is_dyn_op || params.precision_mode == INT8MODE
             ? EngineInfo::EngineType::TRTDynamic
             : EngineInfo::EngineType::TRTStatic);
    curr_engine.cached_engine_batches = params.cached_engine_batches;
    curr_engine.maximum_cached_engines = params.max_cached_engines;
    StrAppend(&curr_engine.engine_name, "my_trt_op_", t);
    status = RegisterSegmentFunctionToFunctionLibrary(
        &graph, curr_engine.segment_graph_def, curr_engine.engine_name);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to register segment graphdef as a function " << t
                   << ": " << status;
      continue;
    }

    engine_bytes_size.push_back(curr_engine.segment_graph_def.ByteSizeLong());
    total_engine_bytes_size += engine_bytes_size.back();
    total_num_nodes_in_segments += curr_segment.first.size();
    engine_segments.push_back(std::move(curr_engine));
    converted_segments.push_back(std::move(curr_segment));

    if (VLOG_IS_ON(8)) {
      string fname = curr_engine.engine_name;
      StrAppend(&fname, ".pb");
      std::fstream f;
      f.open(fname.c_str(), std::fstream::out | std::fstream::binary);
      f << engine_segments.at(t).segment_graph_def.SerializeAsString();
      f.close();
    }
  }

  // Create a TRT node for each segment using its EngineInfo.
  int old_cuda_device = 0;
  auto err = cudaGetDevice(&old_cuda_device);
  if (err != cudaSuccess) {
    LOG(ERROR) << "Couldn't get current device: " << cudaGetErrorString(err);
  }
  VLOG(1) << "Current cuda device is " << old_cuda_device;
  for (int i = 0; i < engine_segments.size(); ++i) {
    auto& engine = engine_segments.at(i);
    // Partition the workspace size by the average of node ratio and segment
    // graphdef size
    engine.max_workspace_size_bytes =
        params.max_workspace_size_bytes *
        (engine_bytes_size.at(i) / total_engine_bytes_size +
         converted_segments.at(i).first.size() / total_num_nodes_in_segments) /
        2.0;
    // The allocator is used to build the engine. The build and the built engine
    // will be destroyed after we get the serialized engine string, so it's fine
    // to use unique_ptr here.
    std::unique_ptr<TRTBaseAllocator> alloc;
    auto device_alloc = GetDeviceAndAllocator(params, engine);
    int cuda_device_id = 0;
    if (device_alloc.first >= 0) {
      cuda_device_id = device_alloc.first;
      alloc.reset(new TRTDeviceAllocator(device_alloc.second));
    } else {
      // Setting allocator as nullptr should get revert to the cudamalloc
      LOG(WARNING) << "Can't identify the cuda device. Running on device 0 ";
    }
    cudaSetDevice(cuda_device_id);
    auto status = CreateTRTNode(&graph, engine_segments, i, alloc.get(),
                                params.max_batch_size);
    // If status is ok, we successfully added the node to the graph and can
    // remove segment ops. Otherwise graph is not modified.
    if (status.ok()) {
      for (auto node_name : converted_segments.at(i).first) {
        graph.RemoveNode(node_map.at(node_name));
      }
    } else {
      // Graph is not modified.
      LOG(WARNING) << "Engine creation for segment " << i << ", composed of "
                   << converted_segments.at(i).first.size()
                   << " nodes failed: " << status << ". Skipping...";
    }
  }
  cudaSetDevice(old_cuda_device);
  graph.ToGraphDef(params.output_graph_def);
  VLOG(1) << "Returning from conversion";
  return tensorflow::Status::OK();
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
