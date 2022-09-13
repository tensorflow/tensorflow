/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/experimental/utils/model_optim.h"

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/experimental/utils/session_utils.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/platform/errors.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

// Creates and provisions a new cluster. The caller must call Shutdown before
// the cluster is destroyed.
Status NewCluster(grappler::Cluster** cluster) {
  int num_cpu_cores = grappler::GetNumAvailableLogicalCPUCores();
  int num_gpus = grappler::GetNumAvailableGPUs();
  int timeout_s = 60 * 10;
  *cluster = new grappler::SingleMachine(timeout_s, num_cpu_cores, num_gpus);
  (*cluster)->DisableDetailedStats(true);
  (*cluster)->AllowSoftPlacement(true);
  (*cluster)->SetNumWarmupSteps(10);
  TF_RETURN_IF_ERROR((*cluster)->Provision());
  return OkStatus();
}

Status RunGrappler(const MetaGraphDef& meta_graph_def,
                   const std::vector<std::string>& input_names,
                   const std::vector<std::string>& output_names,
                   const ConfigProto& config_proto, GraphDef* out_graph_def) {
  std::unique_ptr<grappler::Cluster> cluster;
  grappler::Cluster* p_cluster;
  mutex mu_cluster;  // There can be only one provisioned cluster per process.
  mutex_lock lock(mu_cluster);
  TF_RETURN_IF_ERROR(NewCluster(&p_cluster));
  cluster.reset(p_cluster);

  grappler::ItemConfig item_config;
  item_config.ignore_user_placement = false;

  for (const std::string& name : input_names) {
    item_config.feed_nodes.insert(name);
  }
  for (const std::string& name : output_names) {
    item_config.fetch_nodes.insert(name);
  }

  std::unique_ptr<grappler::GrapplerItem> item =
      grappler::GrapplerItemFromMetaGraphDef("tf_graph", meta_graph_def,
                                             item_config);
  Status status = OkStatus();
  if (!item) {
    status =
        errors::Internal("Failed to create grappler item from MetaGraphDef.");
  } else {
    tensorflow::DeviceBase* cpu_device = nullptr;
    status =
        grappler::RunMetaOptimizer(std::move(*item), config_proto, cpu_device,
                                   cluster.get(), out_graph_def);
  }

  VLOG(2) << "Grappler finished\n";
  TF_RETURN_IF_ERROR(cluster->Shutdown());
  return status;
}

Status ApplyInlining(MetaGraphDef& meta_graph_def,
                     const std::string& saved_model_dir,
                     const std::vector<std::string>& input_names,
                     const std::vector<std::string>& output_names,
                     std::unique_ptr<Session>* session) {
  // Remove "api_implements" attribute
  auto* library = meta_graph_def.mutable_graph_def()->mutable_library();
  for (auto& function : *(library->mutable_function())) {
    auto* attr = function.mutable_attr();
    attr->erase("api_implements");
  }

  // Clear initializer names
  std::vector<std::string> variable_collections = {
      "variables", "model_variables", "trainable_variables", "local_variables"};
  auto* collections = meta_graph_def.mutable_collection_def();
  for (const auto& name : variable_collections) {
    if (collections->find("variables") == collections->end() ||
        collections->find(name) == collections->end()) {
      continue;
    }
    std::vector<std::string> raw_list;
    for (auto& raw : collections->at("variables").bytes_list().value()) {
      VariableDef variable;
      variable.ParseFromString(raw);
      variable.clear_initializer_name();
      std::string variable_str;
      variable.SerializeToString(&variable_str);
      raw_list.push_back(variable_str);
    }
    google::protobuf::RepeatedPtrField<std::string> data(raw_list.begin(),
                                                         raw_list.end());
    collections->at(name).mutable_bytes_list()->mutable_value()->Swap(&data);
  }

  // Initialize config with only function inlining
  ConfigProto config_proto;
  auto* rewriter_config =
      config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config->set_min_graph_nodes(-1);
  rewriter_config->add_optimizers("function");

  GraphDef inlined_graph_def;
  TF_RETURN_IF_ERROR(RunGrappler(meta_graph_def, input_names, output_names,
                                 config_proto, &inlined_graph_def));

  // Restore variables
  meta_graph_def.mutable_graph_def()->CopyFrom(inlined_graph_def);
  RunOptions run_options;
  TF_RETURN_IF_ERROR(
      RestoreSession(run_options, meta_graph_def, saved_model_dir, session));
  return OkStatus();
}

// Returns info for nodes listed in the signature definition.
std::vector<std::string> GetNodeNames(
    const google::protobuf::Map<std::string, tensorflow::TensorInfo>&
        signature) {
  std::vector<std::string> names;
  for (const auto& item : signature) {
    absl::string_view name = item.second.name();
    names.push_back(std::string(name));
  }
  return names;
}

Status LoadSavedModel(const std::string& model_dir,
                      const std::string& signature_key,
                      const std::unordered_set<std::string>& tags,
                      SavedModelBundle* bundle,
                      std::vector<std::string>* input_names,
                      std::vector<std::string>* output_names) {
  RunOptions run_options;
  SessionOptions sess_options = GetSessionConfig();
  TF_RETURN_IF_ERROR(
      LoadSavedModel(sess_options, run_options, model_dir, tags, bundle));

  // Get input and output names
  auto signature_map = bundle->GetSignatures();
  const SignatureDef& signature = signature_map[signature_key];
  std::unordered_map<std::string, std::string> node_rename_map;
  for (auto& item : signature.inputs()) {
    absl::string_view name = item.second.name();
    // Remove tensor suffix like ":0".
    size_t last_colon = name.find_last_of(':');
    if (last_colon != absl::string_view::npos) {
      name.remove_suffix(name.size() - last_colon);
    }
    if (item.first.compare(name)) {
      node_rename_map[std::string(name)] = item.first;
    }
    input_names->push_back(item.first + ":0");
  }
  *output_names = GetNodeNames(signature.outputs());

  // Rename inputs - WAR for kwargs changing when loading
  // this graph in Python
  std::unordered_set<std::string> node_names;
  auto* graph_def = bundle->meta_graph_def.mutable_graph_def();
  for (const auto& node : graph_def->node()) {
    node_names.insert(node.name());
  }
  for (auto& node : *(graph_def->mutable_node())) {
    if (node_rename_map.find(node.name()) != node_rename_map.end()) {
      auto new_name = node_rename_map.at(node.name());
      if (node_names.find(new_name) != node_names.end()) {
        LOG(WARNING) << "Model has input " << new_name << " for node "
                     << node.name() << ". If executing using the Python API, "
                     << "the converted model's keyword arguments may differ "
                     << "from what is expected.";
        continue;
      }
      auto* id_node = graph_def->add_node();
      id_node->set_name(node.name());
      id_node->set_op("Identity");
      id_node->add_input(new_name);
      (*id_node->mutable_attr())["T"] = node.attr().at("dtype");
      node.set_name(new_name);
    }
  }

  return OkStatus();
}

Status AnnotateVariableOps(GraphDef* graph_def) {
  // Construct a mapping of node names to nodes
  std::unordered_map<std::string, NodeDef*> name_to_node;
  for (size_t i = 0; i < graph_def->node_size(); i++) {
    auto* node = graph_def->mutable_node(i);
    name_to_node[node->name()] = node;
  }
  // Go through all the ReadVariableOp in the graph def
  for (size_t i = 0; i < graph_def->node_size(); i++) {
    auto* node = graph_def->mutable_node(i);
    if (!node->op().compare("ReadVariableOp") ||
        !node->op().compare("ResourceGather")) {
      auto* var_node = node;
      // Go up the chain of identities to find a placeholder
      while (!name_to_node[var_node->input().at(0)]->op().compare("Identity")) {
        var_node = name_to_node[var_node->input().at(0)];
      }
      var_node = name_to_node[var_node->input().at(0)];
      auto* attr = node->mutable_attr();
      (*attr)["_shape"] = var_node->attr().at("shape");
    }
  }
  return OkStatus();
}

Status GetTrtRewriterConfig(const TrtConversionParams& params,
                            RewriterConfig* opt_config,
                            bool disable_non_trt_optimizers) {
  opt_config->set_meta_optimizer_iterations(tensorflow::RewriterConfig::ONE);
  opt_config->set_min_graph_nodes(-1);  // do not skip small graphs

  // Turn off remapping.
  opt_config->set_remapping(RewriterConfig_Toggle::RewriterConfig_Toggle_OFF);

  // If the graph has QDQ nodes, then we need to disable folding of the
  // QDQ with constants. Otherwise, the conversion will not work corectly.
  // Ideally, we do this after segmentation and outlining of TRT regions to
  // functions, but we currently lack that capability. Disabling QDQ-const
  // folding doesn't matter if you don't have QDQ nodes, so we always enable
  // this.
  opt_config->set_experimental_disable_folding_quantization_emulation(
      IS_TRT_VERSION_GE(8, 0, 0, 0));

  // Initial transformations before TensorRTOptimizer is called
  if (!disable_non_trt_optimizers) {
    opt_config->add_optimizers("pruning");
    opt_config->add_optimizers("debug_stripper");
    opt_config->add_optimizers("layout");
    opt_config->add_optimizers("dependency");
    opt_config->add_optimizers("constfold");
    opt_config->add_optimizers("common_subgraph_elimination");
  }

  // Parameters for TensorRTOptimizer
  auto trt_optimizer = opt_config->add_custom_optimizers();
  trt_optimizer->set_name("TensorRTOptimizer");

  auto trt_parameter_map = trt_optimizer->mutable_parameter_map();
  (*trt_parameter_map)["dla_core"].set_i(-1);
  (*trt_parameter_map)["dla_fallback_layers"].set_i(-1);
  (*trt_parameter_map)["enable_sparse_compute"].set_b(true);
  (*trt_parameter_map)["is_dynamic_op"].set_b(true);
  (*trt_parameter_map)["minimum_segment_size"].set_i(
      params.minimum_segment_size);
  std::string prec_string;
  TF_RETURN_IF_ERROR(
      TrtPrecisionModeToName(params.precision_mode, &prec_string));
  (*trt_parameter_map)["precision_mode"].set_s(prec_string);
  (*trt_parameter_map)["max_workspace_size_bytes"].set_i(
      params.max_workspace_size_bytes);
  (*trt_parameter_map)["maximum_cached_engines"].set_i(
      params.maximum_cached_engines);
  (*trt_parameter_map)["use_calibration"].set_b(params.use_calibration);
  (*trt_parameter_map)["use_implicit_batch"].set_b(!params.use_dynamic_shape);
  if (params.use_dynamic_shape) {
    std::string strategy =
        ProfileStrategyToName(params.dynamic_shape_profile_strategy);
    std::transform(strategy.cbegin(), strategy.cend(), strategy.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    (*trt_parameter_map)["profile_strategy"].set_s(strategy);
  }
  // Always set this to true for offline TRT engine building.
  (*trt_parameter_map)["allow_build_at_runtime"].set_b(true);

  if (!disable_non_trt_optimizers) {
    opt_config->add_custom_optimizers()->set_name("constfold");
  } else {
    auto off = RewriterConfig_Toggle::RewriterConfig_Toggle_OFF;
    opt_config->set_arithmetic_optimization(off);
    opt_config->set_auto_mixed_precision(off);
    opt_config->mutable_auto_parallel()->set_enable(false);
    opt_config->set_constant_folding(off);
    opt_config->set_debug_stripper(off);
    opt_config->set_dependency_optimization(off);
    opt_config->set_disable_meta_optimizer(false);
    opt_config->set_disable_model_pruning(true);
    opt_config->set_function_optimization(off);
    opt_config->set_implementation_selector(off);
    opt_config->set_layout_optimizer(off);
    opt_config->set_loop_optimization(off);
    opt_config->set_memory_optimization(
        RewriterConfig_MemOptType::RewriterConfig_MemOptType_NO_MEM_OPT);
    opt_config->set_pin_to_host_optimization(off);
    opt_config->set_scoped_allocator_optimization(off);
    opt_config->set_shape_optimization(off);
  }
  return OkStatus();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT