/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/mlir_gpu/failover_compiler.h"

#include <memory>

#include "tensorflow/core/lib/core/errors.h"

namespace xla {

template <typename T>
bool IsUnimplemented(StatusOr<T>& result) {
  return result.status().code() == tensorflow::error::Code::UNIMPLEMENTED;
}

StatusOr<std::unique_ptr<HloModule>> FailoverCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  auto result =
      primary_->RunHloPasses(module->Clone(), stream_exec, device_allocator);
  if (IsUnimplemented(result)) {
    VLOG(2) << "RunHloPasses resulted in " << result.status()
            << ", falling back to secondary backend";
    return secondary_->RunHloPasses(std::move(module), stream_exec,
                                    device_allocator);
  }
  return result;
}

StatusOr<std::unique_ptr<Executable>> FailoverCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  auto result =
      primary_->RunBackend(module->Clone(), stream_exec, device_allocator);
  if (IsUnimplemented(result)) {
    VLOG(2) << "RunBackend resulted in " << result.status()
            << ", falling back to secondary backend";
    return secondary_->RunBackend(std::move(module), stream_exec,
                                  device_allocator);
  }
  return result;
}

StatusOr<std::vector<std::unique_ptr<Executable>>> FailoverCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_execs,
    se::DeviceMemoryAllocator* device_allocator) {
  std::vector<std::unique_ptr<Executable>> result;
  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();
  for (size_t i = 0; i < modules.size(); i++) {
    if (stream_execs[i].size() != 1) {
      // This is not supported by GPU compiler anyway.
      return Unimplemented(
          "Model partitioning not implemented for the failover compiler!");
    }
    auto executable = [stream_execs, device_allocator, i,
                       this](std::unique_ptr<HloModule> module)
        -> StatusOr<std::unique_ptr<Executable>> {
      TF_ASSIGN_OR_RETURN(
          auto processed_module,
          primary_->RunHloPasses(std::move(module), stream_execs[i][0],
                                 device_allocator));
      TF_ASSIGN_OR_RETURN(
          auto result,
          primary_->RunBackend(std::move(processed_module), stream_execs[i][0],
                               device_allocator));
      return result;
    }(modules[i]->Clone());

    if (IsUnimplemented(executable)) {
      VLOG(2) << "Compile resulted in " << executable.status()
              << ", falling back to secondary backend";
      TF_ASSIGN_OR_RETURN(
          modules[i],
          secondary_->RunHloPasses(std::move(modules[i]), stream_execs[i][0],
                                   device_allocator));
      TF_ASSIGN_OR_RETURN(
          executable,
          secondary_->RunBackend(std::move(modules[i]), stream_execs[i][0],
                                 device_allocator));
    }

    if (!executable.ok()) {
      return executable.status();
    }

    result.push_back(std::move(executable.ValueOrDie()));
  }

  return {std::move(result)};
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
FailoverCompiler::CompileAheadOfTime(
    std::unique_ptr<HloModuleGroup> module_group,
    const AotCompilationOptions& options) {
  // This is not supported by GPU compiler anyway.
  return Unimplemented(
      "CompileAheadOfTime not implemented in failover compiler!");
}

HloCostAnalysis::ShapeSizeFunction FailoverCompiler::ShapeSizeBytesFunction()
    const {
  auto prim_fun = primary_->ShapeSizeBytesFunction();
  auto second_fun = secondary_->ShapeSizeBytesFunction();
  return [prim_fun, second_fun](const Shape& shape) -> int64 {
    int64 primary = prim_fun(shape);
    assert(primary == second_fun(shape));
    return primary;
  };
}

}  // namespace xla
