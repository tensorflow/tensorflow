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

#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {

absl::Status RunHloBenchmark(benchmark::State& state,
                             absl::string_view hlo_module,
                             absl::Span<const Literal* const> args,
                             StrToStrMapping replacements,
                             const HloBenchmarkOptions& benchmark_options) {
  xla::CpuClientOptions client_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      xla::GetXlaPjrtCpuClient(client_options));
  PjRtDevice* device = client->devices().front();
  TF_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                      device->default_memory_space());

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnUnverifiedModule(
                          absl::StrReplaceAll(hlo_module, replacements),
                          HloModuleConfig() /* unused */));

  XlaComputation computation(module->ToProto());

  // Compile HLO module to executable.
  CompileOptions compile_options;
  if (benchmark_options.disable_parallel_task_assigner) {
    compile_options.executable_build_options.mutable_debug_options()
        ->add_xla_disable_hlo_passes("cpu-parallel-task-assigner");
  }
  std::unique_ptr<PjRtLoadedExecutable> executable;
  if (benchmark_options.aot_options) {
    auto* cpu_client = tsl::down_cast<TfrtCpuClient*>(client.get());
    TF_ASSIGN_OR_RETURN(executable, cpu_client->CompileAheadOfTimeAndLoad(
                                        computation, compile_options,
                                        *benchmark_options.aot_options));
  } else {
    TF_ASSIGN_OR_RETURN(executable,
                        client->CompileAndLoad(computation, compile_options));
  }

  // Convert literals to PjRtBuffers.
  std::vector<std::unique_ptr<PjRtBuffer>> args_buffers;

  size_t expected_arg_count =
      module->entry_computation()->parameter_instructions().size();

  // If the user has not passed any arguments we need to generate
  // fake arguments based on the number of inputs to the hlo module.
  if (args.empty()) {
    TF_ASSIGN_OR_RETURN(std::vector<Literal> fake_args,
                        MakeFakeArguments(module.get()));
    args_buffers.reserve(fake_args.size());
    for (const Literal& arg : fake_args) {
      TF_ASSIGN_OR_RETURN(args_buffers.emplace_back(),
                          client->BufferFromHostLiteral(arg, memory_space));
      TF_RETURN_IF_ERROR(args_buffers.back()->GetReadyFuture().Await());
    }
  } else {
    if (expected_arg_count != args.size()) {
      return absl::InvalidArgumentError(
          "Number of arguments does not match the number of parameters in "
          "the HLO module.");
    }

    args_buffers.reserve(args.size());
    for (const Literal* arg : args) {
      TF_ASSIGN_OR_RETURN(args_buffers.emplace_back(),
                          client->BufferFromHostLiteral(*arg, memory_space));
      TF_RETURN_IF_ERROR(args_buffers.back()->GetReadyFuture().Await());
    }
  }

  // Execute in synchronous mode to avoid thread hops, as we anyway use our own
  // thread pool if we need to run multiple executions in parallel.
  ExecuteOptions execute_options;
  execute_options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;

  std::vector<PjRtBuffer*> args_ptrs;
  args_ptrs.reserve(args_buffers.size());
  for (const auto& arg : args_buffers) {
    args_ptrs.push_back(arg.get());
  }

  CHECK_GE(benchmark_options.num_executions, 1);
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results(
      benchmark_options.num_executions);

  // Thread pool for dispatching multiple executions in parallel.
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "hlo_benchmark_runner",
                                  benchmark_options.num_executions);

  // Warmup executable.
  TF_ASSIGN_OR_RETURN(results[0], executable->ExecuteSharded(args_ptrs, device,
                                                             execute_options));

  // Benchmark executable.
  for (auto _ : state) {
    if (benchmark_options.num_executions == 1) {
      // Single execution always runs in the caller thread.
      results[0] =
          executable->ExecuteSharded(args_ptrs, device, execute_options)
              .value();
    } else {
      // Multiple executions run in parallel.
      absl::BlockingCounter counter(benchmark_options.num_executions);

      for (size_t i = 0; i < benchmark_options.num_executions; ++i) {
        threads.Schedule([&, i]() {
          results[i] =
              executable->ExecuteSharded(args_ptrs, device, execute_options)
                  .value();
          counter.DecrementCount();
        });
      }

      counter.Wait();
    }

    // Wait for all results to be ready.
    for (size_t i = 0; i < benchmark_options.num_executions; ++i) {
      for (const auto& result : results[i]) {
        CHECK_OK(result->GetReadyFuture().Await());
      }
    }
  }

  return absl::OkStatus();
}

absl::Status CompileHloBenchmark(benchmark::State& state,
                                 absl::string_view hlo_module,
                                 StrToStrMapping replacements,
                                 const HloBenchmarkOptions& benchmark_options) {
  xla::CpuClientOptions client_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      xla::GetXlaPjrtCpuClient(client_options));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnUnverifiedModule(
                          absl::StrReplaceAll(hlo_module, replacements),
                          HloModuleConfig() /* unused */));

  XlaComputation computation(module->ToProto());

  CompileOptions compile_options;
  if (benchmark_options.disable_parallel_task_assigner) {
    compile_options.executable_build_options.mutable_debug_options()
        ->add_xla_disable_hlo_passes("cpu-parallel-task-assigner");
  }

  for (auto _ : state) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                        client->CompileAndLoad(computation, compile_options));
    tsl::testing::DoNotOptimize(executable);
  }

  return absl::OkStatus();
}

}  // namespace xla::cpu
