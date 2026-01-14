/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuning/autotuner_compile_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

namespace {

std::vector<ExecutionInput> ExecutionInputsFromBuffers(
    absl::Span<se::DeviceAddressBase const> buffers,
    absl::Span<Shape const> shapes) {
  CHECK_EQ(buffers.size(), shapes.size());
  std::vector<ExecutionInput> inputs;
  for (int i = 0; i < buffers.size(); ++i) {
    inputs.emplace_back(shapes.at(i));
    // Our executable doesn't have input-output aliasing, so we can pass
    // unowned input buffers.
    inputs.back().SetUnownedBuffer(
        /*index=*/{}, MaybeOwningDeviceAddress(/*unowned=*/buffers.at(i)));
  }
  return inputs;
}

}  // namespace

AutotunerCompileUtil::AutotunerCompileUtil(
    std::unique_ptr<Compiler> compiler, se::StreamExecutor& stream_executor,
    se::Stream& stream, se::DeviceAddressAllocator& allocator,
    const DebugOptions& opts)
    : compiler_(std::move(compiler)),
      stream_executor_(stream_executor),
      stream_(stream),
      allocator_(allocator),
      opts_(opts) {
  GpuCodegenBackend::AdjustDebugOptionsForAutotuning(opts_);
}

absl::StatusOr<AutotunerCompileUtil::ProfilingOutput>
AutotunerCompileUtil::ProfileExecutable(
    Executable* executable, se::Stream* stream,
    absl::Span<se::DeviceAddressBase const> input_buffers,
    absl::Span<Shape const> input_shapes) {
  tsl::profiler::TraceMe traceme("ProfileExecutable");
  {
    std::vector<ExecutionInput> execution_inputs =
        ExecutionInputsFromBuffers(input_buffers, input_shapes);
    // Warmup: in and out buffers are reused while probing different configs,
    // so GPU caches should be in some comparable states during measurements.
    TF_ASSIGN_OR_RETURN(ExecutionOutput execution_output,
                        Execute(*executable, std::move(execution_inputs)));

    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  }
  std::vector<ExecutionInput> execution_inputs =
      ExecutionInputsFromBuffers(input_buffers, input_shapes);
  ExecutionProfile profile;
  // Flag that a warm-up run was executed so that GpuTimer can use the, more
  // accurate, delay kernel implementation.
  profile.set_warmup_run_executed(true);
  TF_ASSIGN_OR_RETURN(
      ExecutionOutput execution_output,
      Execute(*executable, std::move(execution_inputs), &profile));
  return ProfilingOutput(absl::Nanoseconds(profile.compute_time_ns()),
                         execution_output.Commit().ConsumeResult());
}

absl::StatusOr<std::unique_ptr<Executable>> AutotunerCompileUtil::Compile(
    GenerateModuleFn extractor) {
  tsl::profiler::TraceMe traceme("AutotunerCompile");
  absl::StatusOr<std::unique_ptr<HloModule>> new_hlo_module = extractor(opts_);
  if (new_hlo_module.status().GetPayload(kUncompilableFusion).has_value()) {
    // Incompatible value of split-k is an example of an expected failure.
    VLOG(5) << "Module with uncompilable fusion";
    return std::unique_ptr<Executable>();
  }
  if (!new_hlo_module.status().ok()) {
    return new_hlo_module.status();
  }
  Compiler::CompileOptions compile_options;
  compile_options.device_allocator = &allocator_;
  compile_options.embed_hlo_module = false;
  absl::StatusOr<std::unique_ptr<Executable>> out = compiler_->RunBackend(
      std::move(*new_hlo_module), &stream_executor_, compile_options);
  if (out.status().code() == absl::StatusCode::kResourceExhausted ||
      out.status().code() == absl::StatusCode::kCancelled) {
    // Being out of shared memory budget or registers is an expected failure.
    // Cancelling upon register spilling is also an expected failure.
    VLOG(5) << "Compilation failed with status " << out.status()
            << " that is ignored";
    return std::unique_ptr<Executable>();
  }
  if (!out.ok()) {
    return out.status();
  }
  if (opts_.xla_gpu_filter_kernels_spilling_registers_on_autotuning()) {
    const ModuleStats& module_stats = out.value()->module_stats();
    const auto spills_registers = [](const auto& pair) {
      const KernelStats& kernel_stats = pair.second;
      return kernel_stats.store_bytes_spilled > 0 ||
             kernel_stats.load_bytes_spilled > 0;
    };

    if (absl::c_any_of(module_stats, spills_registers)) {
      return std::unique_ptr<Executable>();
    }
  }
  return out;
}

absl::StatusOr<std::unique_ptr<HloModule>> AutotunerCompileUtil::ExtractModule(
    GenerateModuleFn extractor) {
  return extractor(opts_);
}

/*static*/ absl::StatusOr<AutotunerCompileUtil> AutotunerCompileUtil::Create(
    const DeviceOrDevicelessConfig& config, const DebugOptions& opts) {
  tsl::profiler::TraceMe traceme("AutotunerCreate");
  if (config.IsDeviceless()) {
    return absl::InvalidArgumentError(
        "Deviceless autotuning is not supported.");
  }
  se::StreamExecutor* stream_exec = config.GetExecutor();
  se::DeviceAddressAllocator* allocator = config.GetAllocator();
  TF_ASSIGN_OR_RETURN(se::Stream* const stream, config.GetStream());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Compiler> compiler,
      Compiler::GetForPlatform(stream_exec->GetPlatform()->id()));
  return AutotunerCompileUtil(std::move(compiler), *stream_exec, *stream,
                              *allocator, opts);
}

absl::StatusOr<ExecutionOutput> AutotunerCompileUtil::Execute(
    Executable& executable, std::vector<ExecutionInput> arguments,
    ExecutionProfile* profile) {
  tsl::profiler::TraceMe traceme("AutotunerExecute");
  // Require exclusive GPU lock to prevent other runs during autotuning.
  GpuExecutableRunOptions gpu_opts;
  gpu_opts.set_requires_exclusive_lock_on_gpu();

  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(stream_executor_.device_ordinal());
  run_options.set_stream(&stream_);
  run_options.set_allocator(&allocator_);
  run_options.set_gpu_executable_run_options(&gpu_opts);
  run_options.set_execution_profile(profile);
  ServiceExecutableRunOptions service_run_options(run_options);
  TF_ASSIGN_OR_RETURN(ExecutionOutput output,
                      executable.ExecuteAsyncOnStreamWrapper(
                          &service_run_options, std::move(arguments)));

  return std::move(output);
}

}  // namespace gpu
}  // namespace xla
