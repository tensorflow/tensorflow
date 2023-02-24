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
// This file contains util functions related to function graph instantiation and
// optimizations.
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZE_FUNCTION_GRAPH_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZE_FUNCTION_GRAPH_UTILS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/optimized_function_graph_info.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
// TODO(b/246646753): add more tests.

// Generates graph and return information given the input function name,
// attributes and function definition.
Status GetGraphAndArgRets(
    const string& function_name, AttrSlice attrs, const FunctionDef* fdef,
    const FunctionLibraryDefinition* lib_def, std::unique_ptr<Graph>* graph,
    std::vector<Node*>* arg_nodes, std::vector<Node*>* ret_nodes,
    std::vector<string>* ret_node_names, DataTypeVector* ret_types,
    std::vector<string>* control_ret_node_names);

// TODO(iga): Reword
// Pins each arg that emits a `DT_RESOURCE` tensor to the device on which the
// corresponding resource lives. This ensures that the Placer assigns ops that
// access these resources to the appropriate devices.
Status PinArgsAndRets(const std::vector<string>& input_devices,
                      const std::vector<string>& output_devices,
                      const DeviceSet& device_set,
                      const std::vector<Node*>& arg_nodes,
                      const std::vector<Node*>& ret_nodes,
                      const FunctionLibraryDefinition* lib_def,
                      Device* default_device);

// Outputs graph optimization result after all the graph optimization (up till
// before graph partitioning); returns error if optimization fails. Note that
// the `input_lib_def` will be used only if the lib_def in `options` is nullptr.
StatusOr<OptimizedFunctionGraphInfo> OptimizeFunctionGraph(
    const string& function_name, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    const DeviceSet& dev_set, const FunctionLibraryDefinition* input_lib_def,
    const std::vector<CompositeDevice*>& composite_devices, Device* cpu_device,
    Device* default_device, Env* env);

// Pre-processes, partitions and post-optimizes the input graph; returns
// subgraph result (maps from device name to the subgraph); returns error if any
// optimization or partitioning step fails.
StatusOr<std::unique_ptr<std::unordered_map<string, std::unique_ptr<Graph>>>>
PreprocessAndPartitionGraph(
    OptimizedFunctionGraphInfo& input_optimized_graph,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    const DeviceSet& dev_set, const FunctionLibraryDefinition* input_lib_def,
    const std::vector<CompositeDevice*>& composite_devices, Env* env);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZE_FUNCTION_GRAPH_UTILS_H_
