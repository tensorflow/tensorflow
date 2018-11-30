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
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/contrib/tensorrt/convert/convert_nodes.h"
#include "tensorflow/contrib/tensorrt/convert/utils.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resource_manager.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resources.h"
#include "tensorflow/contrib/tensorrt/segment/segment.h"
#include "tensorflow/contrib/tensorrt/test/utils.h"
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
#include "tensorrt/NvInfer.h"
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

TrtCandidateSelector::TrtCandidateSelector(
    const grappler::GraphProperties& graph_properties, int precision_mode)
    : graph_properties_(graph_properties), precision_mode_(precision_mode) {}

Status TrtCandidateSelector::IsTensorRTCandidate(const tensorflow::Node* node) {
  // TODO(laigd): move this set to TrtNodeValidator where it should belong.
  // LINT.IfChange
  static const std::set<string> candidate_ops = {
      "Identity",
      "Snapshot",
      "Const",
      "Conv2D",
      "MaxPool",
      "BiasAdd",
      "Relu",
      "Sigmoid",
      "Tanh",
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
      "Transpose",
      "Reshape",
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
      "Relu6",
      "Square",
  };
  bool is_supported_op_type =
      (candidate_ops.count(node->type_string()) ||
       PluginFactoryTensorRT::GetInstance()->IsPlugin(node->type_string()));
  static const std::set<string> quantize_ops = {
      "QuantizeAndDequantizeV2",
      "QuantizeAndDequantizeV3",
      "FakeQuantWithMinMaxVars",
      "FakeQuantWithMinMaxArgs",
  };
  // In INT8 mode, we will always apply the quantization ranges provided by
  // these ops to the relevant tensors. This happens regardless of the value of
  // use_calibration.
  if (precision_mode_ == INT8MODE && quantize_ops.count(node->type_string())) {
    is_supported_op_type = true;
  }
  // LINT.ThenChange(//tensorflow/contrib/tensorrt/convert/convert_nodes.cc)
  if (!is_supported_op_type) {
    return errors::Unimplemented("Op type ", node->type_string(),
                                 " is not supported");
  }

  std::vector<const Edge*> input_edges;
  TF_RETURN_IF_ERROR(node->input_edges(&input_edges));
  std::vector<std::pair<const NodeDef*, int>> input_node_and_ports;
  input_node_and_ports.reserve(input_edges.size());
  for (const Edge* input_edge : input_edges) {
    input_node_and_ports.emplace_back(&input_edge->src()->def(),
                                      input_edge->src_output());
  }
  return validator_.ValidateNode(node->def(), input_node_and_ports,
                                 graph_properties_);
}

namespace {

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
  LOG(INFO) << "Starting Calib Conversion";
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

tensorflow::Status ConvertGraphDefToTensorRT(
    const tensorflow::GraphDef& graph_def,
    const std::vector<string>& output_names, size_t max_batch_size,
    size_t max_workspace_size_bytes, tensorflow::GraphDef* new_graph_def,
    int precision_mode, int minimum_segment_size, bool is_dyn_op,
    int max_cached_engines, std::vector<int> cached_engine_batches,
    bool use_calibration) {
  // Create GrapplerItem.
  tensorflow::grappler::GrapplerItem item;
  item.fetch = output_names;
  item.graph = graph_def;

// TODO(aaroey): we should have used single machine cluster like the
// following, but the problem is then wrap_conversion will depend on
// direct_session and cause double linking problems. To fix this we need to
// fix or get rid of the swig dependency. Here we use VirtualCluster
// as a work around, and we need to create a session to initialize the
// underlying device before calling this method.
#if 0
  // Create single machine cluster. Note that this will create a session and
  // initialize the gpu devices.
  const int num_cpu_cores =
      tensorflow::grappler::GetNumAvailableLogicalCPUCores();
  const int num_gpus = tensorflow::grappler::GetNumAvailableGPUs();
  VLOG(2) << "cpu_cores: " << num_cpu_cores;
  VLOG(2) << "gpus: " << num_gpus;
  const int timeout_s = 60 * 10;
  std::unique_ptr<tensorflow::grappler::Cluster> cluster(
      new tensorflow::grappler::SingleMachine(
          timeout_s, num_cpu_cores, num_gpus));
  // These settings are the defaults in tensorflow/python/grappler/cluster.py.
  cluster->DisableDetailedStats(true);
  cluster->AllowSoftPlacement(true);
  cluster->SetNumWarmupSteps(10);
  TF_RETURN_IF_ERROR(cluster->Provision());
#else
  // Create virtual cluster. Grappler requires a virtual cluster with a proper
  // GPU device in order to calculate flops>0 or fails with FATAL in dbg mode.
  // We add numbers from a Pascal card here to have flops>0.
  tensorflow::DeviceProperties device_properties;
  device_properties.set_type("GPU");
  device_properties.mutable_environment()->insert({"architecture", "6"});
  device_properties.set_num_cores(3584);
  device_properties.set_frequency(1531);
  std::unique_ptr<tensorflow::grappler::Cluster> cluster(
      new tensorflow::grappler::VirtualCluster(
          {{"/GPU:0", device_properties}}));
#endif

  // Create RewriterConfig.
  tensorflow::ConfigProto config_proto;
  auto& rw_cfg =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  // TODO(aaroey): use only const folding and layout for the time being since
  // new optimizers break the graph for trt.
  rw_cfg.add_optimizers("constfold");
  rw_cfg.add_optimizers("layout");
  auto optimizer = rw_cfg.add_custom_optimizers();
  optimizer->set_name("TensorRTOptimizer");
  auto& parameters = *(optimizer->mutable_parameter_map());
  parameters["minimum_segment_size"].set_i(minimum_segment_size);
  parameters["max_batch_size"].set_i(max_batch_size);
  parameters["is_dynamic_op"].set_b(is_dyn_op);
  parameters["max_workspace_size_bytes"].set_i(max_workspace_size_bytes);
  TF_RETURN_IF_ERROR(GetPrecisionModeName(
      precision_mode, parameters["precision_mode"].mutable_s()));
  parameters["maximum_cached_engines"].set_i(max_cached_engines);
  if (!cached_engine_batches.empty()) {
    auto list = parameters["cached_engine_batches"].mutable_list();
    for (const int batch : cached_engine_batches) {
      list->add_i(batch);
    }
  }
  parameters["use_calibration"].set_b(use_calibration);

  // Run optimizer.
  tensorflow::grappler::MetaOptimizer meta_opt(nullptr, config_proto);
  TF_RETURN_IF_ERROR(meta_opt.Optimize(cluster.get(), item, new_graph_def));

  if (VLOG_IS_ON(5)) {
    std::fstream f;
    f.open("TRTConversionInput.pb",
           std::fstream::out | std::fstream::binary | std::fstream::trunc);
    f << new_graph_def->SerializeAsString();
    f.close();
  }
  return Status::OK();
}

// Function to get subsegment information structure.
tensorflow::Status GetEngineInfo(
    const tensorflow::Graph* g,
    const tensorflow::grappler::GraphProperties& graph_properties,
    const std::set<string>& segment_nodes,
    const std::unordered_map<string, tensorflow::Node*>& node_map,
    const std::vector<tensorflow::Node*>& reverse_topo_order,
    EngineInfo* info) {
  std::vector<int> subgraph_node_ids;  // Topologically sorted node ids.
  std::set<string> subgraph_node_names = segment_nodes;
  std::set<int> added_const_node_ids;  // Used to prevent double insertion.
  std::set<string> segment_devices;

  // Map from src_node_name+port to the unique port numbers of the TRT op, where
  // the src_node_name is the name of the source node of the input/output
  // edge, thus there must not be any duplicates since source nodes of
  // input/output edges must be in different split of the graph.
  // TODO(aaroey): consider using node id and port instead.
  // TODO(aaroey): using topo order instead of reverting reverse topo order.
  std::unordered_map<string, int> input_to_engine_port, output_to_engine_port;
  for (auto it = reverse_topo_order.rbegin(); it != reverse_topo_order.rend();
       ++it) {
    const auto& node_name = (*it)->name();
    if (segment_nodes.count(node_name) == 0) continue;
    auto node = *it;
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
    subgraph_node_ids.push_back(node_id);
    // Create input connections.
    for (const auto edge : node->in_edges()) {
      auto input_node = edge->src();
      if (input_node->IsSource() || segment_nodes.count(input_node->name())) {
        continue;
      }
      if (edge->IsControlEdge()) {
        // Control input.
        info->connections.emplace_back(input_node->name(), input_node->id(),
                                       node_name, node_id,
                                       /*input_edge=*/true);
      } else if (input_node->type_string() == "Const") {
        // Add constant data input nodes into the segment graphdef (thus also in
        // the engine). We don't care if it has other output edges going into
        // other engines or TF nodes. Since we add it only to the segment
        // graphdef, not the segment itself, it won't be removed from the graph.
        // If it doesn't have any edges, TF will prune it out.
        //
        // Note that the segmenter already ensure that the constant data input
        // is valid and suppported by the engine.
        if (!added_const_node_ids.insert(input_node->id()).second) {
          // Already added before.
          continue;
        }
        VLOG(1) << "Adding const node " << input_node->name();
        QCHECK(subgraph_node_names.insert(input_node->name()).second);
        // Since we already add (duplicate) the const input node to the segment
        // graphdef, it's now not a data dependency any more, but to make the
        // dependency correct we still add a control dependency.
        info->connections.emplace_back(input_node->name(), input_node->id(),
                                       node_name, node_id,
                                       /*input_edge=*/true);
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
    // Create output connections.
    for (const auto edge : node->out_edges()) {
      auto output_node = edge->dst();
      if (output_node->IsSink() || segment_nodes.count(output_node->name())) {
        continue;
      }
      if (edge->IsControlEdge()) {
        // Control output.
        info->connections.emplace_back(output_node->name(), output_node->id(),
                                       node_name, node_id,
                                       /*input_edge=*/false);
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
  subgraph_node_ids.insert(subgraph_node_ids.begin(),
                           added_const_node_ids.begin(),
                           added_const_node_ids.end());
  TF_RETURN_IF_ERROR(ConvertSegmentToGraphDef(
      g, graph_properties, subgraph_node_names, subgraph_node_ids,
      &info->connections, &info->segment_graph_def, &info->engine_name));
  // TODO(sami): This should not happen once segmenter is updated.
  if (segment_devices.size() == 1) {
    info->device = *segment_devices.begin();
  } else if (segment_devices.size() > 1) {
    LOG(WARNING) << "Detected multiple(" << segment_devices.size()
                 << ") devices for the segment. Picking first one to continue "
                 << "but this shouldn't have happened";
    info->device = *segment_devices.begin();
  } else {
    VLOG(1) << "No device is assigned to the segment. "
            << "A device will be assigned during graph execution (inference).";
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
                        tensorflow::Node** node, int* port) {
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
  LOG(FATAL) << "Node " << (**node).name() << " not found in any engine.";
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
tensorflow::Status CreateTRTNode(const std::vector<EngineInfo>& infos, int pos,
                                 int max_batch_size, tensorflow::Graph* graph,
                                 nvinfer1::IGpuAllocator* alloc,
                                 std::vector<Node*>* engine_nodes) {
  const auto& info = infos.at(pos);
  TRT_RETURN_IF_TEST_VALUE(StrCat(info.engine_name, ":CreateTRTNode"), "fail");
  std::vector<tensorflow::TensorShapeProto> output_shape_protos;
  std::vector<tensorflow::TensorShapeProto> input_shape_protos;
  std::vector<tensorflow::PartialTensorShape> input_shapes;
  std::vector<tensorflow::NodeDefBuilder::NodeOut> inputs;
  std::vector<tensorflow::Node*> input_nodes;
  std::vector<tensorflow::Node*> control_input_nodes;
  std::unordered_set<string> control_input_names;
  std::vector<tensorflow::DataType> out_types;

  VLOG(1) << "Processing " << info.engine_name;
  // Collect needed info for creating the engine node in the graph
  for (const auto& conn : info.connections) {
    // Control edges
    if (conn.is_control_edge()) {
      // Skip control outputs for now. control output info are not needed for
      // node creation and will be processed later.
      if (!conn.is_input_edge) continue;

      // Rewrire control input if it's not found in original graph.
      tensorflow::Node* input_node = graph->FindNodeId(conn.outside_id);
      int port = tensorflow::Graph::kControlSlot;
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
      } else {
        // Set the shapes and data types of input edge.
        tensorflow::TensorShapeProto in_shape;
        conn.outside_shape.AsProto(&in_shape);
        if (input_shape_protos.size() <= conn.port_number) {
          input_shape_protos.resize(conn.port_number + 1);
          input_shapes.resize(conn.port_number + 1);
        }
        input_shape_protos.at(conn.port_number) = in_shape;
        input_shapes.at(conn.port_number) = conn.outside_shape;

        // Rewrire data input if it's not found in original graph.
        tensorflow::Node* input_node = graph->FindNodeId(conn.outside_id);
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

  const bool calibrate_int8 =
      (info.precision_mode == INT8MODE && info.use_calibration);
  // Build the engine and get its serialized representation.
  string segment_string;
  if (info.engine_type == EngineInfo::EngineType::TRTStatic || calibrate_int8) {
    // Create static engine for fp32/fp16 mode, and test validity of the engine
    // for int8 calibration mode. We don't want engine to fail at the
    // calibration time. So we are constructing a FP32 engine here to check its
    // validity, and if it is a valid engine then we put the serialized graphdef
    // to the op. Otherwise we skip node creation for this engine.
    Logger trt_logger;
    TrtUniquePtrType<nvinfer1::ICudaEngine> engine;
    // TODO(sami): What happens if 1st dim is not batch?
    TF_RETURN_IF_ERROR(ConvertGraphDefToEngine(
        info.segment_graph_def, calibrate_int8 ? FP32MODE : info.precision_mode,
        max_batch_size, info.max_workspace_size_bytes, input_shapes,
        &trt_logger, alloc, /*calibrator=*/nullptr, &engine,
        info.use_calibration,
        /*convert_successfully=*/nullptr));
    TrtUniquePtrType<nvinfer1::IHostMemory> engine_data(engine->serialize());
    segment_string =
        string((const char*)engine_data->data(), engine_data->size());
    if (calibrate_int8) {
      // See above comment about why not putting this inside the 'else' branch.
      segment_string = info.segment_graph_def.SerializeAsString();
    }
  } else {
    segment_string = info.segment_graph_def.SerializeAsString();
  }

  // TODO(aaroey): use enum instead, and add a helper method to do the
  // conversion.
  string prec_string;
  TF_RETURN_IF_ERROR(GetPrecisionModeName(info.precision_mode, &prec_string));
  if (info.precision_mode == INT8MODE && calibrate_int8 &&
      !TRTResourceManager::instance()->getManager("TRTCalibration")) {
    LOG(ERROR) << "Failed to construct calibration storage";
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
  for (const string& c : control_input_names) {
    node_builder.ControlInput(c);
  }

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
          .Attr("use_calibration", info.use_calibration)
          .Attr("OutT", out_types)
          .Finalize(&trt_node);
  if (!status.ok()) {
    LOG(ERROR) << "Node construction failed with" << status;
    return status;
  }
  VLOG(1) << "Adding TRTEngine " << info.engine_name << " to graph";

  // Up until this point, graph is not modified. If we return !status.ok() from
  // here, this segment will be skipped
  // TODO(aaroey): let it return proper error status for the following logic
  // instead of checking fail.
  tensorflow::Node* engine_node = graph->AddNode(trt_node, &status);
  (*engine_nodes)[pos] = engine_node;
  if (!status.ok()) {
    LOG(ERROR) << "Adding node failed " << status;
    return status;
  }
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
    tensorflow::Node* output_node = graph->FindNodeId(conn.outside_id);
    int port = conn.outside_port;
    if (!output_node) {
      UpdateToEngineNode(infos, pos, *engine_nodes, /*is_input_edge=*/false,
                         conn.outside_node_name, &output_node, &port);
    }
    VLOG(1) << "Updating " << engine_node->name() << ":" << conn.port_number
            << " to " << output_node->name() << ":" << port;
    if (conn.is_control_edge()) {
      QCHECK_EQ(Graph::kControlSlot, port);
      graph->AddControlEdge(engine_node, output_node);
    } else {
      auto new_edge =
          graph->AddEdge(engine_node, conn.port_number, output_node, port);
      QCHECK(new_edge) << "Adding a new edge failed " << engine_node->name()
                       << ":" << conn.port_number << " -> "
                       << output_node->name() << ":" << conn.outside_port;
    }
  }
  return Status::OK();
}

// Function to construct a funcdef from the segment and add it to the graph.
tensorflow::Status RegisterSegmentFunctionToFunctionLibrary(
    tensorflow::Graph* graph, const tensorflow::GraphDef& segment,
    const string& engine_name) {
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
      sgraph, StrCat(engine_name, "_native_segment"), native_segment));
  if (VLOG_IS_ON(7)) {
    VLOG(7) << engine_name << " Function_Def ";
    VLOG(7) << native_segment->DebugString();
  }
  VLOG(1) << "Adding funcdef to graphlib";
  TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(fdeflib));
  return tensorflow::Status::OK();
}

std::pair<int, tensorflow::Allocator*> GetDeviceAndAllocator(
    const ConversionParams& params, const EngineInfo& engine) {
  int cuda_device_id = -1;
  tensorflow::Allocator* dev_allocator = nullptr;
  if (params.cluster == nullptr || params.cluster->GetDeviceSet() == nullptr ||
      engine.device.empty()) {
    // If device is not set, use the first found GPU device for the conversion.
    for (int tf_gpu_id_value = 0; tf_gpu_id_value < 100; ++tf_gpu_id_value) {
      TfGpuId tf_gpu_id(tf_gpu_id_value);
      PlatformGpuId platform_gpu_id;
      Status s = GpuIdManager::TfToPlatformGpuId(tf_gpu_id, &platform_gpu_id);
      if (s.ok()) {
        VLOG(1) << "Found TF GPU " << tf_gpu_id.value() << " at cuda device "
                << platform_gpu_id.value();
        cuda_device_id = platform_gpu_id.value();
        GPUOptions gpu_options;
        // If the TF to Cuda gpu id mapping exist, the device and corresponding
        // allocator must have been initialized already, so the
        // GetGPUAllocator() call won't create a new allocator.
        dev_allocator = GPUProcessState::singleton()->GetGPUAllocator(
            gpu_options, tf_gpu_id, 1);
        break;
      }
      LOG(ERROR) << "TF GPU with id " << tf_gpu_id_value << " does not exist "
                 << s;
    }
    return std::make_pair(cuda_device_id, dev_allocator);
  }

  // Use the device requested by the engine.
  auto device_set = params.cluster->GetDeviceSet();
  std::vector<tensorflow::Device*> devices;
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
      LOG(WARNING) << msg;
    }
    tensorflow::AllocatorAttributes alloc_attr;
    cuda_device_id = devices[0]->tensorflow_gpu_device_info()->gpu_id;
    dev_allocator = devices[0]->GetAllocator(alloc_attr);
    VLOG(1) << "Using allocator " << dev_allocator->Name()
            << " and cuda_device_id " << cuda_device_id;
  } else {
    LOG(WARNING) << "Cluster is set but device '" << engine.device
                 << "' is not found in the cluster";
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
  TrtCandidateSelector candidate_selector(*params.graph_properties,
                                          params.precision_mode);
  TF_RETURN_IF_ERROR(tensorrt::segment::SegmentGraph(
      &graph,
      std::bind(&TrtCandidateSelector::IsTensorRTCandidate, &candidate_selector,
                std::placeholders::_1),
      // Input validation is already done by TrtCandidateSelector, so we don't
      // need to check the input edges.
      [](const Edge* edge) { return true; }, OutputEdgeValidator(),
      segment_options, &initial_segments));
  LOG(INFO) << "Number of TensorRT candidate segments: "
            << initial_segments.size();

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
    if (params.use_calibration && params.precision_mode != INT8MODE) {
      return errors::InvalidArgument(
          "Calibration with FP32 or FP16 is not supported.");
    }
    curr_engine.engine_type = ((params.is_dyn_op || params.use_calibration)
                                   ? EngineInfo::EngineType::TRTDynamic
                                   : EngineInfo::EngineType::TRTStatic);
    curr_engine.use_calibration = params.use_calibration;
    curr_engine.cached_engine_batches = params.cached_engine_batches;
    curr_engine.maximum_cached_engines = params.max_cached_engines;
    StrAppend(&curr_engine.engine_name, "TRTEngineOp_", t);
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
      string fname = engine_segments.back().engine_name;
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
  std::vector<Node*> engine_nodes;
  engine_nodes.resize(engine_segments.size());
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
    auto status = CreateTRTNode(engine_segments, i, params.max_batch_size,
                                &graph, alloc.get(), &engine_nodes);
    // If status is ok, we successfully added the node to the graph and can
    // remove segment ops. Otherwise graph is not modified.
    string msg = StrCat("TensorRT node ", engine.engine_name,
                        " added for segment ", i, " consisting of ",
                        converted_segments.at(i).first.size(), " nodes");
    if (status.ok()) {
      LOG(INFO) << msg << " succeeded.";
      for (auto node_name : converted_segments.at(i).first) {
        graph.RemoveNode(node_map.at(node_name));
      }
    } else {
      // Graph is not modified.
      LOG(WARNING) << msg << " failed: " << status << ". Fallback to TF...";
    }
    if (VLOG_IS_ON(1)) {
      msg = "Segment consists of nodes: ";
      for (const string& node_name : converted_segments.at(i).first) {
        StrAppend(&msg, node_name, ", ");
      }
      VLOG(1) << msg;
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
