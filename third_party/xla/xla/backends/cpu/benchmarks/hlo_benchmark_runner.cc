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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/pjrt/cpu/abstract_tfrt_cpu_buffer.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/cpu/cpu_device.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {

namespace {

// Helper class for replacing output buffers with input buffers if there is
// aliasing.
// This class is required for benchmarking models where outputs are aliased to
// inputs. This returns ownership of the aliased memory to the input buffers
// so that it can be reused for the next iteration, otherwise we'd have invalid
// input buffers.
// Alternatively, we could turn off aliasing but that wouldn't be representative
// of production performance.
class AliasHelper {
 public:
  AliasHelper(HloModule* hlo_module, PjRtClient* client, PjRtDevice* device,
              PjRtMemorySpace* memory_space)
      : client_(client), device_(device), memory_space_(memory_space) {
    hlo_module->input_output_alias_config().ForEachAlias(
        [this](const ShapeIndex& output_index,
               const HloInputOutputAliasConfig::Alias& alias) {
          aliased_output_index_to_argument_index_.push_back(
              std::make_pair(output_index, alias.parameter_number));
        });
  }

  bool ComputationHasAliasing() const {
    return !aliased_output_index_to_argument_index_.empty();
  }

  absl::Status SwapOutputAliasedBuffersToArgumentBuffers(
      PjRtBuffer* result,
      std::vector<std::unique_ptr<PjRtBuffer>>& args_buffers,
      std::vector<PjRtBuffer*>& args_ptrs) {
    if (!ComputationHasAliasing()) {
      return absl::OkStatus();
    }
    TfrtCpuBuffer* result_tfrt_cpu_buffer =
        tsl::down_cast<TfrtCpuBuffer*>(result);

    TF_ASSIGN_OR_RETURN(
        AbstractTfrtCpuBuffer::DonationTransaction buffer_donation,
        result_tfrt_cpu_buffer->AcquireDonation());
    TrackedCpuDeviceBuffer* tracked_tfrt_cpu_device_buffer =
        buffer_donation.device_buffer();

    for (const auto& [output_index, arg_index] :
         aliased_output_index_to_argument_index_) {
      // we don't need the entire buffer just the one at the output index
      tsl::AsyncValuePtr<CpuDeviceMemory> output_cpu_memory =
          tracked_tfrt_cpu_device_buffer->Buffer(output_index);

      auto tracked_device_buffer = std::make_unique<TrackedCpuDeviceBuffer>(
          /*is_tuple=*/false, /*owns_buffers=*/true,
          absl::InlinedVector<tsl::AsyncValueRef<CpuDeviceMemory>, 4>{
              output_cpu_memory.CopyRef()},
          tsl::MakeAvailableAsyncValueRef<CpuEvent>());

      args_buffers[arg_index] = std::make_unique<TfrtCpuBuffer>(
          tsl::down_cast<AbstractTfrtCpuBuffer*>(args_buffers[arg_index].get())
              ->on_device_shape(),
          std::move(tracked_device_buffer),
          tsl::down_cast<TfrtCpuClient*>(client_),
          tsl::down_cast<TfrtCpuDevice*>(device_), memory_space_);

      args_ptrs[arg_index] = args_buffers[arg_index].get();
    }
    return absl::OkStatus();
  }

 private:
  std::vector<std::pair<ShapeIndex, int64_t>>
      aliased_output_index_to_argument_index_;

  PjRtClient* client_;
  PjRtDevice* device_;
  PjRtMemorySpace* memory_space_;
};

}  // namespace

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

  CHECK_GE(benchmark_options.num_executions, 1);

  // For every parallel execution, we need to have a copy of the arguments in
  // case there is aliasing. This also makes the benchmark more realistic,
  // since in production each call would have separate argument buffers.
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> execution_args_buffers(
      benchmark_options.num_executions);

  size_t expected_arg_count =
      module->entry_computation()->parameter_instructions().size();

  // If the user has not passed any arguments we need to generate
  // fake arguments based on the number of inputs to the hlo module.
  if (args.empty()) {
    TF_ASSIGN_OR_RETURN(std::vector<Literal> fake_args,
                        MakeFakeArguments(module.get()));
    for (auto& args_buffers : execution_args_buffers) {
      args_buffers.reserve(fake_args.size());
      for (const Literal& arg : fake_args) {
        TF_ASSIGN_OR_RETURN(args_buffers.emplace_back(),
                            client->BufferFromHostLiteral(arg, memory_space));
        TF_RETURN_IF_ERROR(args_buffers.back()->GetReadyFuture().Await());
      }
    }
  } else {
    if (expected_arg_count != args.size()) {
      return absl::InvalidArgumentError(
          "Number of arguments does not match the number of parameters in "
          "the HLO module.");
    }

    for (auto& args_buffers : execution_args_buffers) {
      args_buffers.reserve(args.size());
      for (const Literal* arg : args) {
        TF_ASSIGN_OR_RETURN(args_buffers.emplace_back(),
                            client->BufferFromHostLiteral(*arg, memory_space));
        TF_RETURN_IF_ERROR(args_buffers.back()->GetReadyFuture().Await());
      }
    }
  }

  // Execute in synchronous mode to avoid thread hops, as we anyway use our own
  // thread pool if we need to run multiple executions in parallel.
  ExecuteOptions execute_options;
  execute_options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;

  std::vector<std::vector<PjRtBuffer*>> execution_args_ptrs(
      benchmark_options.num_executions);
  for (int i = 0; i < benchmark_options.num_executions; ++i) {
    std::vector<PjRtBuffer*>& args_ptrs = execution_args_ptrs[i];
    const std::vector<std::unique_ptr<PjRtBuffer>>& args_buffers =
        execution_args_buffers[i];
    args_ptrs.reserve(args_buffers.size());
    for (const auto& arg : args_buffers) {
      args_ptrs.push_back(arg.get());
    }
  }

  AliasHelper alias_helper(module.get(), client.get(), device, memory_space);
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> execution_results(
      benchmark_options.num_executions);

  // Thread pool for dispatching multiple executions in parallel.
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "hlo_benchmark_runner",
                                  benchmark_options.num_executions);

  auto run_benchmark_once = [&]() -> absl::Status {
    if (benchmark_options.num_executions == 1) {
      // Single execution always runs in the caller thread.
      execution_results[0] =
          executable
              ->ExecuteSharded(execution_args_ptrs[0], device, execute_options)
              .value();
    } else {
      // Multiple executions run in parallel.
      absl::BlockingCounter counter(benchmark_options.num_executions);

      for (size_t i = 0; i < benchmark_options.num_executions; ++i) {
        threads.Schedule([&, i]() {
          const std::vector<PjRtBuffer*>& args_ptrs = execution_args_ptrs[i];
          std::vector<std::unique_ptr<PjRtBuffer>>& results =
              execution_results[i];
          results =
              executable->ExecuteSharded(args_ptrs, device, execute_options)
                  .value();
          counter.DecrementCount();
        });
      }

      counter.Wait();
    }

    // Wait for all results to be ready.
    for (size_t i = 0; i < benchmark_options.num_executions; ++i) {
      for (const auto& result : execution_results[i]) {
        CHECK_OK(result->GetReadyFuture().Await());
        CHECK(!alias_helper.ComputationHasAliasing() ||
              result->IsTuple() && execution_results[i].size() == 1)
            << "Only single output tuple is supported in benchmarking aliased "
               "models. "
               "result->IsTuple(): "
            << result->IsTuple()
            << " execution_results size: " << execution_results[i].size();
        std::vector<std::unique_ptr<PjRtBuffer>>& args_buffers =
            execution_args_buffers[i];
        std::vector<PjRtBuffer*>& args_ptrs = execution_args_ptrs[i];
        TF_RETURN_IF_ERROR(
            alias_helper.SwapOutputAliasedBuffersToArgumentBuffers(
                result.get(), args_buffers, args_ptrs));
      }
    }

    return absl::OkStatus();
  };

  // Warm up executable.
  TF_RETURN_IF_ERROR(run_benchmark_once());

  // Benchmark executable.
  for (auto _ : state) {
    TF_RETURN_IF_ERROR(run_benchmark_once());
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
