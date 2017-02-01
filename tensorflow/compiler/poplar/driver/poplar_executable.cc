/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/poplar/driver/poplar_executable.h"

#include <stdint.h>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace se = ::perftools::gputools;

namespace xla {
namespace poplar {

PoplarExecutable::PoplarExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloModuleConfig> module_config)
    : Executable(std::move(hlo_module), std::move(module_config)) {
}

StatusOr<perftools::gputools::DeviceMemoryBase> PoplarExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return tensorflow::errors::Unimplemented(
          "ExecuteOnStream is not yet supported on Poplar.");
}

StatusOr<std::unique_ptr<ShapedBuffer>> PoplarExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return tensorflow::errors::Unimplemented(
          "ExecuteOnStream is not yet supported on Poplar.");
}

Status PoplarExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    ShapedBuffer* result_buffer, HloExecutionProfile* hlo_execution_profile) {
  return tensorflow::errors::Unimplemented(
          "ExecuteOnStream is not yet supported on Poplar.");
}

StatusOr<perftools::gputools::DeviceMemoryBase>
PoplarExecutable::ExecuteAsyncOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments) {
  return tensorflow::errors::Unimplemented(
          "ExecuteAsyncOnStream is not yet supported on Poplar.");
}

}  // namespace poplar
}  // namespace xla

