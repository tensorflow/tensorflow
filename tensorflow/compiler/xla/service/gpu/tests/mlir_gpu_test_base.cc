/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/register.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"

namespace xla {
namespace gpu {

MlirGpuTestBase::MlirGpuTestBase() {
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(tensorflow::GpuPlatformName())
          .value();
  BackendOptions options;
  options.set_platform(platform);
  backend_ = xla::Backend::CreateBackend(options).value();
}

StatusOr<std::unique_ptr<Executable>> MlirGpuTestBase::CompileMlirModule(
    mlir::ModuleOp module, se::Stream* stream) {
  llvm::LLVMContext llvm_context;
  auto llvm_module = std::make_unique<llvm::Module>("", llvm_context);
#if TENSORFLOW_USE_ROCM
  llvm_module->setTargetTriple(amdgpu::TargetTriple());
  llvm_module->setDataLayout(amdgpu::DataLayout());
#else
  llvm_module->setTargetTriple(nvptx::TargetTriple());
  llvm_module->setDataLayout(nvptx::DataLayout());
#endif

  se::StreamExecutor* stream_exec = stream->parent();
  GpuDeviceInfo gpu_device_info = GetGpuDeviceInfo(stream_exec);
  IrEmitterContext ir_emitter_context(
      /*hlo_module=*/nullptr, /*buffer_assignment=*/nullptr,
      backend_->platform()->Name(), gpu_device_info,
      stream_exec->GetDeviceDescription().cuda_compute_capability(),
      stream_exec->GetDeviceDescription().rocm_compute_capability(),
      /*mlir_context=*/nullptr, llvm_module.get());

  HloModuleConfig module_config;
  module_config.set_debug_options(GetDebugOptionsFromFlags());
  return CompileLmhloToExecutable(
      static_cast<GpuCompiler*>(backend_->compiler()), module, "TestModule",
      module_config, Compiler::CompileOptions(), "main", stream_exec,
      std::move(llvm_module), &ir_emitter_context);
}

StatusOr<ExecutionOutput> MlirGpuTestBase::RunMlirModule(
    mlir::ModuleOp module, se::Stream* stream,
    absl::Span<const se::DeviceMemoryBase> arguments) {
  TF_ASSIGN_OR_RETURN(auto executable, CompileMlirModule(module, stream));

  ExecutableRunOptions executable_run_options;
  executable_run_options.set_stream(stream);
  executable_run_options.set_allocator(backend_->memory_allocator());
  ServiceExecutableRunOptions run_options(executable_run_options,
                                          backend_->StreamBorrower());
  std::vector<ExecutionInput> execution_inputs;
  execution_inputs.reserve(arguments.size());

  for (auto arg : arguments) {
    Shape shape =
        ShapeUtil::MakeShape(xla::U8, {static_cast<int64_t>(arg.size())});
    execution_inputs.emplace_back(shape);
    execution_inputs.back().SetBuffer({}, MaybeOwningDeviceMemory(arg));
  }

  TF_ASSIGN_OR_RETURN(auto output,
                      executable->ExecuteAsyncOnStream(
                          &run_options, std::move(execution_inputs),
                          /*hlo_execution_profile=*/nullptr));

  TF_CHECK_OK(stream->BlockHostUntilDone());

  return std::move(output);
}

StatusOr<std::vector<std::vector<uint8_t>>>
MlirGpuTestBase::RunMlirModuleWithHostBuffers(
    mlir::ModuleOp module, std::vector<absl::Span<uint8_t>> arguments) {
  auto* allocator = backend_->memory_allocator();
  std::vector<se::OwningDeviceMemory> owning_memory;
  owning_memory.reserve(arguments.size());
  for (auto host_buffer : arguments) {
    owning_memory.push_back(
        allocator
            ->Allocate(backend_->default_device_ordinal(), host_buffer.size())
            .value());
  }
  auto stream =
      backend_->BorrowStream(backend_->default_device_ordinal()).value();
  std::vector<se::DeviceMemoryBase> args;
  for (int i = 0; i < owning_memory.size(); i++) {
    se::DeviceMemoryBase memory(*owning_memory[i]);
    stream->ThenMemcpy(&memory, static_cast<void*>(arguments[i].data()),
                       memory.size());
    args.push_back(memory);
  }
  TF_ASSIGN_OR_RETURN(ExecutionOutput output,
                      RunMlirModule(module, stream.get(), args));

  std::vector<std::vector<uint8_t>> host_outputs;
  for (const auto& result : output.Result().buffers().leaves()) {
    host_outputs.emplace_back();
    host_outputs.back().resize(result.second.size());
    stream->ThenMemcpy(static_cast<void*>(host_outputs.back().data()),
                       result.second, result.second.size());
  }
  TF_CHECK_OK(stream->BlockHostUntilDone());
  return host_outputs;
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> MlirGpuTestBase::ParseMlirModule(
    absl::string_view module_text, mlir::MLIRContext& context) {
  context
      .loadDialect<mlir::arith::ArithmeticDialect, mlir::lmhlo::LmhloDialect,
                   mlir::mhlo::MhloDialect, mlir::func::FuncDialect,
                   mlir::gpu::GPUDialect, mlir::lmhlo_gpu::LmhloGpuDialect>();
  llvm::SourceMgr source_mgr;
  std::string diagnostic_str;
  llvm::raw_string_ostream os(diagnostic_str);
  mlir::SourceMgrDiagnosticHandler handler(source_mgr, &context, os);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(module_text.data(), module_text.size()), &context);
  if (!module) {
    return InvalidArgument("Failed to parse MLIR module: %s", diagnostic_str);
  }
  return module;
}

StatusOr<std::vector<std::vector<uint8_t>>>
MlirGpuTestBase::RunMlirTextWithHostBuffers(
    absl::string_view module_text, std::vector<absl::Span<uint8_t>> arguments) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModule(module_text, context));
  return RunMlirModuleWithHostBuffers(*module, arguments);
}

StatusOr<std::unique_ptr<Executable>> MlirGpuTestBase::CompileMlirText(
    absl::string_view module_text) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModule(module_text, context));
  auto stream =
      backend_->BorrowStream(backend_->default_device_ordinal()).value();
  return CompileMlirModule(*module, stream.get());
}

}  // namespace gpu
}  // namespace xla
