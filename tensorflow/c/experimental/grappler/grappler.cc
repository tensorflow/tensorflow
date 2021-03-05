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

// A map containing the input graph as its key, and TF_GrapplerItem as the
// value. Users can fetch GrapplerItem for additional info to transform the
// graph.
absl::flat_hash_map<TF_Buffer*, const TF_GrapplerItem*>* GrapplerItemMap() {
  static absl::flat_hash_map<TF_Buffer*, const TF_GrapplerItem*>*
      grappler_items =
          new absl::flat_hash_map<TF_Buffer*, const TF_GrapplerItem*>;
  return grappler_items;
}
}  // namespace

namespace tensorflow {
namespace grappler {

Status CGraphOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* optimized_graph_def) {
  OwnedTFStatus c_status(TF_NewStatus());
  OwnedTFBuffer graph_buf(TF_NewBuffer());
  OwnedTFBuffer optimized_graph_buf(TF_NewBuffer());
  TF_RETURN_IF_ERROR(MessageToBuffer(item.graph, graph_buf.get()));

  const auto it = GrapplerItemMap()->find(graph_buf.get());
  if (it == GrapplerItemMap()->end())
    GrapplerItemMap()->insert(
        {graph_buf.get(), reinterpret_cast<const TF_GrapplerItem*>(&item)});

  optimizer_.optimize_func(c_optimizer_, graph_buf.get(),
                           optimized_graph_buf.get(), c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(
      BufferToMessage(optimized_graph_buf.get(), optimized_graph_def));

  GrapplerItemMap()->erase(graph_buf.get());
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
  params.major_version = SE_MAJOR;
  params.minor_version = SE_MINOR;
  params.patch_version = SE_PATCH;
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
