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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_VARIABLE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_VARIABLE_H_

#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {

class Tensor;

// A helper base class that wraps tensorflow::VariableInfo for the convenience
// of passing between plugin and tensorflow. Similar to `PluginOpKernelContext`,
// the implementations can accomodate for "Internal build" and "External build",
// meaning the plugin is built with TensorFlow either together or separately. In
// repsective build modes, the implementations can either include
// tensorflow::VariableInfo and use C++ API directly, or include the C structure
// `TF_VariableInfo` and use the corresponding C API.
class PluginVariable {
 public:
  PluginVariable() = default;
  virtual ~PluginVariable() = default;

  // `result_tensor` will point to the tensor possessed by the variable if
  // status is ok.
  virtual tsl::Status GetTensor(const Tensor** result_tensor) = 0;

  virtual tsl::Status GetMutableTensor(Tensor** result_tensor) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_VARIABLE_H_
