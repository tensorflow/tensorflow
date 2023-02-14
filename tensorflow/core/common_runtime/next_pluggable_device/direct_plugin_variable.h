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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_DIRECT_PLUGIN_VARIABLE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_DIRECT_PLUGIN_VARIABLE_H_

#include <string>

#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_variable.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {

class DirectPluginOpKernelContext;

class DirectPluginVariable : public PluginVariable {
 public:
  DirectPluginVariable(int index, const std::string& name, Var* var);
  tsl::Status GetTensor(const Tensor** result_tensor) override {
    *result_tensor = var_info_.var()->tensor();
    return tsl::OkStatus();
  }

  tsl::Status GetMutableTensor(Tensor** result_tensor) override {
    *result_tensor = var_info_.var()->tensor();
    return tsl::OkStatus();
  }

  VariableInfo* GetVariableInfo() { return &var_info_; }

  friend DirectPluginOpKernelContext;

 private:
  VariableInfo var_info_{0, "", nullptr};
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_DIRECT_PLUGIN_VARIABLE_H_
