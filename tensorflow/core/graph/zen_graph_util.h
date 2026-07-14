/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPH_ZEN_GRAPH_UTIL_H_
#define TENSORFLOW_CORE_GRAPH_ZEN_GRAPH_UTIL_H_
#ifdef AMD_ZENDNN

#include <string>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace zen_op_registry {

// Prefix that we add to Tensorflow op name to construct Zen op name.
static const char* const kZenNodePrefix = "_Zen";

// Get the name of Zen op from original TensorFlow op.
// We prefix the original op with "Zen" to get Zen op.
inline string GetZenOpName(const string& name) {
  return string(kZenNodePrefix) + name;
}

// Check whether op name with type T is registered as Zen operator
// that will go through name change or layout change pass.
//
// @input  op_name - name of the op.
// @input  T - datatype to be used for checking op.
// @return true if op name is registered as Zen op that will go through name
// change or layout change pass; false otherwise.
static inline bool IsZenOpKernelRegistered(const string& op_name, DataType T) {
  string registered_kernels_key = op_name + string(DataType_Name(T));
  thread_local static auto* registered_kernels_map =
      new absl::flat_hash_map<string, bool>();
  auto kernel_element = registered_kernels_map->find(registered_kernels_key);
  bool kernel_registered = false;

  if (kernel_element == registered_kernels_map->end()) {
    string registered_kernels = KernelsRegisteredForOp(op_name);
    // String returned by KernelsRegisteredForOp looks like below:
    //
    // Op = ZenMatMul, kernels =
    // device='CPU'; T in [DT_FLOAT]
    // device='CPU'; T in [DT_DOUBLE]

    // If we have multiple kernels registered for the op. We need to verify
    // our datatype
    if (registered_kernels.find(string(DataType_Name(T))) != string::npos) {
      kernel_registered = true;
    }
    registered_kernels_map->insert(
        std::make_pair(registered_kernels_key, kernel_registered));
  } else {
    // Kernel is visited at least once. Return stored registration result.
    kernel_registered = kernel_element->second;
  }
  return kernel_registered;
}

}  // namespace zen_op_registry
}  // namespace tensorflow

#endif  // AMD_ZENDNN
#endif  // TENSORFLOW_CORE_GRAPH_ZEN_GRAPH_UTIL_H_
