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

#include <memory>

#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function_thunks.h"

namespace tensorflow {
namespace xla_compiled_cpu_function_factory {

std::unique_ptr<XlaCompiledCpuFunction> CreateXlaCompiledCpuFunctionThunks(
    const XlaCompiledCpuFunction::StaticData& static_data,
    XlaCompiledCpuFunction::AllocMode alloc_mode) {
  return std::make_unique<XlaCompiledCpuFunctionThunks>(static_data,
                                                        alloc_mode);
}

}  // namespace xla_compiled_cpu_function_factory
}  // namespace tensorflow
