/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"

#include <memory>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_parser.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

absl::Status RunHloBenchmark(benchmark::State& state,
                             std::string_view hlo_module,
                             absl::Span<const Literal* const> args,
                             StrToStrMapping replacements) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      GetTfrtCpuClient(CpuClientOptions()));
  PjRtDevice* device = client->devices().front();

  HloModuleConfig config;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnUnverifiedModule(
          absl::StrReplaceAll(hlo_module, replacements), HloModuleConfig()));

  XlaComputation computation(module->ToProto());

  // Compile HLO module to executable.
  CompileOptions compile_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                      client->Compile(computation, compile_options));

  // Convert literals to PjRtBuffers.
  std::vector<std::unique_ptr<PjRtBuffer>> args_buffers;
  args_buffers.reserve(args.size());

  for (const Literal* arg : args) {
    TF_ASSIGN_OR_RETURN(args_buffers.emplace_back(),
                        client->BufferFromHostLiteral(*arg, device));
    TF_RETURN_IF_ERROR(args_buffers.back()->GetReadyFuture().Await());
  }

  // Execute in synchronous mode to avoid thread hops.
  ExecuteOptions execute_options;
  execute_options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;

  std::vector<PjRtBuffer*> args_ptrs;
  args_ptrs.reserve(args_buffers.size());
  for (const auto& arg : args_buffers) {
    args_ptrs.push_back(arg.get());
  }

  // Warmup executable.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<PjRtBuffer>> results,
      executable->ExecuteSharded(args_ptrs, device, execute_options));

  // Benchmark executable.
  for (auto _ : state) {
    TF_ASSIGN_OR_RETURN(results, executable->ExecuteSharded(args_ptrs, device,
                                                            execute_options));
  }

  return absl::OkStatus();
}

}  // namespace xla::cpu
