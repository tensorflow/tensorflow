/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
// This file extends/implements core graph optimizer base classes in terms of
// the C API defined in grappler.h. A class "CSomething" represents a
// "Something" that can be manipulated via calls in the C interface and a C
// struct called "TP_Something".

#include "tensorflow/c/experimental/grappler/grappler.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/experimental/grappler/grappler_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"

namespace {

#define VALIDATE_STRUCT_SIZE(STRUCT_NAME, STRUCT_OBJ, SIZE_VALUE_NAME)    \
  do {                                                                    \
    if (STRUCT_OBJ.struct_size == 0) {                                    \
      return tensorflow::Status(tensorflow::error::FAILED_PRECONDITION,   \
                                "struct_size field in " #STRUCT_NAME      \
                                " must be set to " #SIZE_VALUE_NAME "."); \
    }                                                                     \
  } while (0)

#define VALIDATE_MEMBER(STRUCT_NAME, STRUCT_OBJ, NAME)                  \
  do {                                                                  \
    if (STRUCT_OBJ.NAME == 0) {                                         \
      return tensorflow::Status(tensorflow::error::FAILED_PRECONDITION, \
                                "'" #NAME "' field in " #STRUCT_NAME    \
                                " must be set.");                       \
    }                                                                   \
  } while (0)

tensorflow::Status ValidateTPOptimizerRegistrationParams(
    const TP_OptimizerRegistrationParams& params) {
  VALIDATE_STRUCT_SIZE(TP_OptimizerRegistrationParams, params,
                       TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE);
  VALIDATE_MEMBER(TP_OptimizerRegistrationParams, params, device_type);
  return tensorflow::Status::OK();
}

tensorflow::Status ValidateTPOptimizer(const TP_Optimizer& optimizer) {
  VALIDATE_STRUCT_SIZE(TP_Optimizer, optimizer, TP_OPTIMIZER_STRUCT_SIZE);
  VALIDATE_MEMBER(TP_Optimizer, optimizer, optimize_func);
  return tensorflow::Status::OK();
}

tensorflow::Status ValidateTPOptimizerConfigs(
    const TP_OptimizerConfigs& configs) {
  VALIDATE_STRUCT_SIZE(TP_OptimizerConfigs, configs,
                       TP_OPTIMIZER_CONFIGS_STRUCT_SIZE);
  return tensorflow::Status::OK();
}

#undef VALIDATE_MEMBER
#undef VALIDATE_STRUCT_SIZE
}  // namespace

namespace tensorflow {
namespace grappler {

Status CGraphOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* optimized_graph_def) {
  OwnedTFStatus c_status(TF_NewStatus());
  OwnedTFBuffer graph_buf(TF_NewBuffer());
  OwnedTFBuffer optimized_graph_buf(TF_NewBuffer());
  TF_RETURN_IF_ERROR(MessageToBuffer(item.graph, graph_buf.get()));

  optimizer_.optimize_func(c_optimizer_, graph_buf.get(),
                           reinterpret_cast<const TF_GrapplerItem*>(&item),
                           optimized_graph_buf.get(), c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(
      BufferToMessage(optimized_graph_buf.get(), optimized_graph_def));

  return Status::OK();
}

#define CONFIG_TOGGLE(optimizer)                             \
  if (tp_configs.optimizer == TF_TriState_Off)               \
    configs.toggle_config[#optimizer] = RewriterConfig::OFF; \
  else                                                       \
    configs.toggle_config[#optimizer] = RewriterConfig::ON;

void CGraphOptimizerRegister(
    const PluginGraphOptimizerRegistry::Creator& creator,
    const TP_OptimizerConfigs tp_configs, const char* device_type) {
  ConfigList configs;
  // disable_model_pruning is turned off by default.
  if (tp_configs.disable_model_pruning == TF_TriState_On)
    configs.disable_model_pruning = true;
  else
    configs.disable_model_pruning = false;
  // The other configs are turned on by default.
  CONFIG_TOGGLE(implementation_selector);
  CONFIG_TOGGLE(function_optimization);
  CONFIG_TOGGLE(common_subgraph_elimination);
  CONFIG_TOGGLE(arithmetic_optimization);
  CONFIG_TOGGLE(debug_stripper);
  CONFIG_TOGGLE(constant_folding);
  CONFIG_TOGGLE(shape_optimization);
  CONFIG_TOGGLE(auto_mixed_precision);
  CONFIG_TOGGLE(auto_mixed_precision_mkl);
  CONFIG_TOGGLE(pin_to_host_optimization);
  CONFIG_TOGGLE(layout_optimizer);
  CONFIG_TOGGLE(remapping);
  CONFIG_TOGGLE(loop_optimization);
  CONFIG_TOGGLE(dependency_optimization);
  CONFIG_TOGGLE(auto_parallel);
  CONFIG_TOGGLE(memory_optimization);
  CONFIG_TOGGLE(scoped_allocator_optimization);
  PluginGraphOptimizerRegistry::RegisterPluginOptimizerOrDie(
      creator, device_type, configs);
}

#undef CONFIG_TOGGLE

tensorflow::Status InitGraphPlugin(void* dso_handle) {
  tensorflow::Env* env = tensorflow::Env::Default();

  // Step 1: Load symbol for `TF_InitPlugin`
  void* dso_symbol;
  TF_RETURN_IF_ERROR(
      env->GetSymbolFromLibrary(dso_handle, "TF_InitGraph", &dso_symbol));

  // Step 2: Call `TF_InitPlugin`
  auto init_fn = reinterpret_cast<TFInitGraphPluginFn>(dso_symbol);
  return InitGraphPlugin(init_fn);
}

tensorflow::Status InitGraphPlugin(TFInitGraphPluginFn init_fn) {
  TP_OptimizerRegistrationParams params{
      TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE};
  TP_Optimizer optimizer{TP_OPTIMIZER_STRUCT_SIZE};
  TP_OptimizerConfigs optimizer_configs{TP_OPTIMIZER_CONFIGS_STRUCT_SIZE};
  params.major_version = GO_MAJOR;
  params.minor_version = GO_MINOR;
  params.patch_version = GO_PATCH;
  params.optimizer = &optimizer;
  params.optimizer_configs = &optimizer_configs;

  OwnedTFStatus c_status(TF_NewStatus());
  init_fn(&params, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateTPOptimizerRegistrationParams(params));
  TF_RETURN_IF_ERROR(ValidateTPOptimizer(optimizer));
  TF_RETURN_IF_ERROR(ValidateTPOptimizerConfigs(optimizer_configs));

  CGraphOptimizerRegister(
      [=]() { return new CGraphOptimizer(optimizer, params.device_type); },
      optimizer_configs, params.device_type);

  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow

void TF_GetNodesToPreserveListSize(const TF_GrapplerItem* item, int* num_values,
                                   size_t* storage_size, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  const std::unordered_set<std::string>& nodes =
      reinterpret_cast<const tensorflow::grappler::GrapplerItem*>(item)
          ->NodesToPreserve();
  *num_values = nodes.size();
  *storage_size = 0;
  for (const std::string& str : nodes) {
    *storage_size += str.size();
  }
}

void TF_GetNodesToPreserveList(const TF_GrapplerItem* item, char** values,
                               size_t* lengths, int num_values, void* storage,
                               size_t storage_size, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  const std::unordered_set<std::string>& nodes =
      reinterpret_cast<const tensorflow::grappler::GrapplerItem*>(item)
          ->NodesToPreserve();
  char* p = static_cast<char*>(storage);

  int index = 0;
  for (const std::string& s : nodes) {
    if (index >= num_values) break;
    values[index] = p;
    lengths[index] = s.size();
    if ((p + s.size()) > (static_cast<char*>(storage) + storage_size)) {
      status->status = tensorflow::errors::InvalidArgument(
          "Not enough storage to hold the requested list of nodes");
      return;
    }
    memcpy(values[index], s.data(), s.size());
    p += s.size();
    index++;
  }
}

void TF_GetFetchNodesListSize(const TF_GrapplerItem* item, int* num_values,
                              size_t* storage_size, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  const std::vector<std::string>& nodes =
      reinterpret_cast<const tensorflow::grappler::GrapplerItem*>(item)->fetch;
  *num_values = nodes.size();
  *storage_size = 0;
  for (const std::string& str : nodes) {
    *storage_size += str.size();
  }
}

void TF_GetFetchNodesList(const TF_GrapplerItem* item, char** values,
                          size_t* lengths, int num_values, void* storage,
                          size_t storage_size, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  const std::vector<std::string>& nodes =
      reinterpret_cast<const tensorflow::grappler::GrapplerItem*>(item)->fetch;

  const int len = std::min(num_values, static_cast<int>(nodes.size()));
  char* p = static_cast<char*>(storage);
  for (int index = 0; index < len; ++index) {
    const std::string& s = nodes[index];
    values[index] = p;
    lengths[index] = s.size();
    if ((p + s.size()) > (static_cast<char*>(storage) + storage_size)) {
      status->status = tensorflow::errors::InvalidArgument(
          "Not enough storage to hold the requested list of nodes");
      return;
    }
    memcpy(values[index], s.data(), s.size());
    p += s.size();
  }
}

TF_GraphProperties* TF_NewGraphProperties(const TF_GrapplerItem* item) {
  return reinterpret_cast<TF_GraphProperties*>(
      new tensorflow::grappler::GraphProperties(
          *reinterpret_cast<const tensorflow::grappler::GrapplerItem*>(item)));
}

void TF_DeleteGraphProperties(TF_GraphProperties* graph_properties) {
  if (graph_properties == nullptr) return;
  delete reinterpret_cast<tensorflow::grappler::GraphProperties*>(
      graph_properties);
}

void TF_InferStatically(TF_GraphProperties* graph_properties,
                        TF_Bool assume_valid_feeds,
                        TF_Bool aggressive_shape_inference,
                        TF_Bool include_input_tensor_values,
                        TF_Bool include_output_tensor_values,
                        TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  tensorflow::Status s =
      reinterpret_cast<tensorflow::grappler::GraphProperties*>(graph_properties)
          ->InferStatically(assume_valid_feeds, aggressive_shape_inference,
                            include_input_tensor_values,
                            include_output_tensor_values);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
  }
}

void TF_GetInputPropertiesListSize(TF_GraphProperties* graph_properties,
                                   const char* name, int* num_values,
                                   TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  *num_values =
      reinterpret_cast<tensorflow::grappler::GraphProperties*>(graph_properties)
          ->GetInputProperties(name)
          .size();
}

void TF_GetOutputPropertiesListSize(TF_GraphProperties* graph_properties,
                                    const char* name, int* num_values,
                                    TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  *num_values =
      reinterpret_cast<tensorflow::grappler::GraphProperties*>(graph_properties)
          ->GetOutputProperties(name)
          .size();
}

void TF_GetInputPropertiesList(TF_GraphProperties* graph_properties,
                               const char* name, TF_Buffer** properties,
                               int num_values, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  const std::vector<tensorflow::OpInfo::TensorProperties>& tensor_properties =
      reinterpret_cast<tensorflow::grappler::GraphProperties*>(graph_properties)
          ->GetInputProperties(name);
  const int len =
      std::min(num_values, static_cast<int>(tensor_properties.size()));
  for (int i = 0; i < len; ++i) {
    tensorflow::Status s =
        tensorflow::MessageToBuffer(tensor_properties[i], properties[i]);
    if (!s.ok()) {
      ::tensorflow::Set_TF_Status_from_Status(status, s);
      return;
    }
  }
}

void TF_GetOutputPropertiesList(TF_GraphProperties* graph_properties,
                                const char* name, TF_Buffer** properties,
                                int num_values, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  const std::vector<tensorflow::OpInfo::TensorProperties>& tensor_properties =
      reinterpret_cast<tensorflow::grappler::GraphProperties*>(graph_properties)
          ->GetOutputProperties(name);
  const int len =
      std::min(num_values, static_cast<int>(tensor_properties.size()));
  for (int i = 0; i < len; ++i) {
    tensorflow::Status s =
        tensorflow::MessageToBuffer(tensor_properties[i], properties[i]);
    if (!s.ok()) {
      ::tensorflow::Set_TF_Status_from_Status(status, s);
      return;
    }
  }
}

TF_FunctionLibraryDefinition* TF_NewFunctionLibraryDefinition(
    TF_Buffer* graph_buf, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  tensorflow::GraphDef graph_def;
  tensorflow::Status s = tensorflow::BufferToMessage(graph_buf, &graph_def);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  return reinterpret_cast<TF_FunctionLibraryDefinition*>(
      new tensorflow::FunctionLibraryDefinition(
          tensorflow::OpRegistry::Global(), graph_def.library()));
}

void TF_DeleteFunctionLibraryDefinition(TF_FunctionLibraryDefinition* fn_lib) {
  if (fn_lib == nullptr) return;
  delete reinterpret_cast<tensorflow::FunctionLibraryDefinition*>(fn_lib);
}

void TF_LookUpOpDef(TF_FunctionLibraryDefinition* fn_lib, const char* name,
                    TF_Buffer* buf, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  const tensorflow::OpDef* op_def_ptr = nullptr;
  tensorflow::Status s =
      reinterpret_cast<tensorflow::FunctionLibraryDefinition*>(fn_lib)
          ->LookUpOpDef(name, &op_def_ptr);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return;
  }

  s = tensorflow::MessageToBuffer(*op_def_ptr, buf);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return;
  }
}
