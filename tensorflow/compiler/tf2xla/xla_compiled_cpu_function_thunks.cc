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

#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function_thunks.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "xla/backends/cpu/codegen/aot_compiled_function_library.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/status.h"

namespace tensorflow {

XlaCompiledCpuFunctionThunks::XlaCompiledCpuFunctionThunks(
    const StaticData& static_data, AllocMode alloc_mode)
    : XlaCompiledCpuFunction(static_data, alloc_mode) {
  CHECK(static_data.compilation_result_proto_ != nullptr);

  std::unique_ptr<xla::cpu::FunctionLibrary> function_library =
      std::make_unique<xla::cpu::AotCompiledFunctionLibrary>(
          function_library_symbol_map());

  auto aot_compilation_result =
      xla::cpu::CpuAotCompilationResultThunks::FromString(
          static_data.compilation_result_proto_->SerializeAsString(),
          function_library.release());

  // To load a CPU executable we don't need a compiler or a stream executor.
  TF_CHECK_OK(aot_compilation_result.status());
  // NO_CDC: aot_compilation_result is checked to be OK above.
  auto cpu_executable = std::move(*aot_compilation_result.value())
                            .LoadExecutable(nullptr, nullptr);

  TF_CHECK_OK(cpu_executable.status());
  auto executable_or_err =
      // NO_CDC: cpu_executable is checked to be OK above.
      xla::cpu::NanoRtExecutable::Create(std::move(cpu_executable.value()));

  TF_CHECK_OK(executable_or_err.status());
  // NO_CDC: executable_or_err is checked to be OK above.
  executable_ = std::move(executable_or_err.value());
}

bool XlaCompiledCpuFunctionThunks::Run() {
  auto ret = Execute(GenerateNanortArgs(), GenerateNanortResults(),
                     GenerateNanortPreallocatedTemp());

  if (!ret.ok()) {
    set_error_msg(ret.message());
  }

  return ret.ok();
}

std::vector<xla::cpu::NanoRtExecutable::Argument>
XlaCompiledCpuFunctionThunks::GenerateNanortArgs() {
  std::vector<xla::cpu::NanoRtExecutable::Argument> arguments;
  arguments.reserve(num_args());
  for (int i = 0; i < num_args(); ++i) {
    arguments.push_back(
        xla::cpu::NanoRtExecutable::Argument(arg_data(i), arg_size(i)));
  }

  return arguments;
}

std::vector<xla::cpu::NanoRtExecutable::Result>
XlaCompiledCpuFunctionThunks::GenerateNanortResults() {
  std::vector<xla::cpu::NanoRtExecutable::Result> results;
  results.reserve(num_results());
  for (int i = 0; i < num_results(); ++i) {
    results.push_back(
        xla::cpu::NanoRtExecutable::Result(result_data(i), result_size(i)));
  }

  return results;
}

xla::cpu::NanoRtExecutable::PreallocatedTemp
XlaCompiledCpuFunctionThunks::GenerateNanortPreallocatedTemp() {
  xla::cpu::NanoRtExecutable::PreallocatedTemp temp;

  auto temp_allocation_index = this->temp_allocation_index();
  if (temp_allocation_index.has_value()) {
    temp = xla::cpu::NanoRtExecutable::PreallocatedTemp(
        static_cast<std::byte*>(buffer_table()[*temp_allocation_index]),
        buffer_infos()[*temp_allocation_index].size());
  }

  return temp;
}

absl::Status XlaCompiledCpuFunctionThunks::Execute(
    absl::Span<const xla::cpu::NanoRtExecutable::Argument> arguments,
    absl::Span<const xla::cpu::NanoRtExecutable::Result> results,
    xla::cpu::NanoRtExecutable::PreallocatedTemp temp) {
  auto event =
      executable_->Execute(arguments, results, temp, thunk_run_options_);
  tsl::BlockUntilReady(event);

  if (!event.IsConcrete()) {
    return event.GetError();
  }

  return absl::OkStatus();
}

}  // namespace tensorflow
