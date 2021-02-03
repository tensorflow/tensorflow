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
#ifndef TENSORFLOW_C_EXPERIMENTAL_GRAPPLER_GRAPPLER_H_
#define TENSORFLOW_C_EXPERIMENTAL_GRAPPLER_GRAPPLER_H_
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_status.h"

// --------------------------------------------------------------------------
// C API for Graph. The API is under active development and eventually
// should allow registering a plugin graph optimizer with TensorFlow.
//
// Conventions:
//   * Struct prefix indicates whether struct fields should be filled by the
//     plugin or core implementation:
//     * Struct that should be filled by the plugin: `TP_OptimizerConfigs`,
//       `TP_Optimizer`, `TP_OptimizerRegistrationParams`
//     * Struct that should be filled by the proper: `TF_GrapplerItem`,
//       `TF_GraphProperties`, `TF_FunctionLibraryDefinition`
//   * We use `struct_size` for version checking. It should be set both by
//     core and the plugin.
//     * For example, `TF_InitGraph` function receives
//       `TP_OptimizerRegistrationParams*` as input with `struct_size`
//       populated by core. The plugin is responsible for setting
//       `struct_size` as well, along with all other fields.
//     * Refer to "TensorFlow Versioning Strategy" section at
//       https://github.com/tensorflow/community/pull/257/files.
//     * Note that the API is still under active development and doesn't have
//       versioning guarantees yet.
//   * `void* ext` is a free-form field that can be populated by
//     a plugin in `TP_*` structs or potential future extension points .
//
// Example usage:
//
//   /* Sample TensorFlow code below, exact implementation might differ. */
//   // Version checking uses `struct_size`. It should be set both by core
//   // and the plugin.
//   TP_OptimizerRegistrationParams params{
//       TP_OPTIMIZER_REGISTRARION_PARAMS_STRUCT_SIZE};
//   TP_Optimizer optimizer{TP_OPTIMIZER_STRUCT_SIZE};
//   TP_OptimizerConfigs configs{TP_OPTIMIZER_CONFIGS_STRUCT_SIZE};
//   params.optimizer = &optimizer;
//   params.configs = &configs;
//
//   /* Plugin code below */
//    void TF_InitGraph(TP_OptimizerRegistrationParams* params,
//                            TF_Status* status) {
//      params->struct_size = TP_OPTIMIZER_REGISTRARION_PARAMS_STRUCT_SIZE;
//      params->device_type = "MY_DEVICE";
//
//      // Disable certain optimizer.
//      params->optimizer_configs->struct_size =
//      TP_OPTIMIZER_CONFIGS_STRUCT_SIZE; params->optimizer_configs->remapping =
//      TF_TriState_Off;
//
//      // Set functions to create a new optimizer.
//      params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;
//      params->optimizer->create_func = (My_optimizer::create_func);
//    }

#define SE_MAJOR 0
#define SE_MINOR 0
#define SE_PATCH 1

#ifdef __cplusplus
extern "C" {
#endif

// TF_TriState is the C API typedef for tri-state.
typedef enum TF_TriState {
  TF_TriState_Default = 0,
  TF_TriState_Off,
  TF_TriState_On,
} TF_TriState;

// Flags indicating whether existing optimizers should be turned off.
// It's optional for plugin to set functions to return true/false. If not
// set, proper uses configuration set by user.
typedef struct TP_OptimizerConfigs {
  size_t struct_size;
  void* ext;  // reserved for future use
  TF_TriState disable_model_pruning;
  TF_TriState implementation_selector;
  TF_TriState function_optimization;
  TF_TriState common_subgraph_elimination;
  TF_TriState arithmetic_optimization;
  TF_TriState debug_stripper;
  TF_TriState constant_folding;
  TF_TriState shape_optimization;
  TF_TriState auto_mixed_precision;
  TF_TriState auto_mixed_precision_mkl;
  TF_TriState pin_to_host_optimization;
  TF_TriState layout_optimizer;
  TF_TriState remapping;
  TF_TriState loop_optimization;
  TF_TriState dependency_optimization;
  TF_TriState auto_parallel;
  TF_TriState memory_optimization;
  TF_TriState scoped_allocator_optimization;
} TP_OptimizerConfigs;

#define TP_OPTIMIZER_CONFIGS_STRUCT_SIZE \
  TF_OFFSET_OF_END(TP_OptimizerConfigs, scoped_allocator_optimization)

// Struct for Optimizer. Plugin authors must provide an optimize function.
// Creation and deletion functions are optional.
typedef struct TP_Optimizer {
  size_t struct_size;
  void* ext;  // reserved for future use

  // [Optional]
  // Create function for optimizer.
  void* (*create_func)();

  // Optimizer function for optimizer. The first param is an optimizer created
  // by create_func. The second param is input graph. The third param is output
  // graph.
  void (*optimize_func)(void*, TF_Buffer*, TF_Buffer*, TF_Status*);

  // [Optional]
  // Destroy function for optimizer. If Create function is provided, destroy
  // function is must.
  void (*destroy_func)(void*);
} TP_Optimizer;

#define TP_OPTIMIZER_STRUCT_SIZE TF_OFFSET_OF_END(TP_Optimizer, destroy_func)

typedef struct TP_OptimizerRegistrationParams {
  size_t struct_size;
  void* ext;  // reserved for future use

  // Graph C API version.
  int32_t major_version;
  int32_t minor_version;
  int32_t patch_version;

  // Backend device type supported by the optimizer.
  const char* device_type;
  TP_OptimizerConfigs* optimizer_configs;  // output, set by plugin
  TP_Optimizer* optimizer;                 // output, set by plugin
} TP_OptimizerRegistrationParams;

#define TP_OPTIMIZER_REGISTRARION_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(TP_OptimizerRegistrationParams, optimizer)

// TF_InitGraph is used to do graph optimizer registration.
// Plugin should implement TF_InitGraph to register graph optimizers.
void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status);

// TF_GrapplerItem represents a combination of a graph, one of more fetch nodes,
// and potentially a set of nodes to feed.
typedef struct TF_GrapplerItem TF_GrapplerItem;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_C_EXPERIMENTAL_GRAPPLER_GRAPPLER_H_
