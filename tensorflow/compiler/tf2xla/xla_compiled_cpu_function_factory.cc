/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function_factory.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"

namespace tensorflow {
namespace xla_compiled_cpu_function_factory {

// Weak symbol to allow for the thunk factory to be registered by the
// xla_compiled_cpu_function_thunk_factory_registerer. This is a workaround that
// allows us to link in the thunk runtime without breaking AOT size constraints.
std::unique_ptr<XlaCompiledCpuFunction> CreateXlaCompiledCpuFunctionThunks(
    const XlaCompiledCpuFunction::StaticData& static_data,
    XlaCompiledCpuFunction::AllocMode alloc_mode) __attribute__((weak));

absl::StatusOr<std::unique_ptr<XlaCompiledCpuFunction>> Create(
    const XlaCompiledCpuFunction::StaticData& static_data,
    XlaCompiledCpuFunction::AllocMode alloc_mode) {
  if (static_data.has_thunk_sequence()) {
    if (CreateXlaCompiledCpuFunctionThunks == nullptr) {
      return absl::InternalError(
          "XlaCompiledCpuFunctionThunks factory is not registered");
    }
    return CreateXlaCompiledCpuFunctionThunks(static_data, alloc_mode);
  } else {
    return std::make_unique<XlaCompiledCpuFunction>(static_data, alloc_mode);
  }
}

}  // namespace xla_compiled_cpu_function_factory
}  // namespace tensorflow
