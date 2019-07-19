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

#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_compiler.h"

#include "mlir/LLVMIR/LLVMDialect.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nvptx_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/failover_compiler.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace mlir {

using ::mlir::MLIRContext;
using ::mlir::LLVM::LLVMDialect;

namespace {
int64 ConfigureLLVMModuleAndGetPointerSize(MLIRContext* context) {
  LLVMDialect* dialect = context->getRegisteredDialect<LLVMDialect>();
  llvm::Module& module = dialect->getLLVMModule();
  module.setTargetTriple(gpu::nvptx::kTargetTriple);
  module.setDataLayout(gpu::nvptx::kDataLayout);
  return module.getDataLayout().getPointerSize();
}
}  // namespace

MlirCompiler::MlirCompiler()
    : pointer_size_(ConfigureLLVMModuleAndGetPointerSize(&context_)) {}

se::Platform::Id MlirCompiler::PlatformId() const {
  return stream_executor::cuda::kCudaPlatformId;
}

StatusOr<std::unique_ptr<HloModule>> MlirCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  // Until we find a reason to do something different, run the same passes
  // that the normal GPU backend runs.
  TF_RETURN_IF_ERROR(xla::gpu::impl::OptimizeHloModule(
      module.get(), stream_exec, device_allocator));

  TF_RETURN_IF_ERROR(
      xla::gpu::impl::PrepareHloModuleForIrEmitting(module.get()));

  return std::move(module);
}

StatusOr<std::unique_ptr<Executable>> MlirCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("Not yet implemented in MLIR compiler");
}

Status MlirCompiler::RunHloPassesOnModuleGroup(
    HloModuleGroup* module_group,
    absl::Span<se::StreamExecutor* const> executors,
    se::DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("Not yet implemented in MLIR compiler");
}

StatusOr<std::vector<std::unique_ptr<Executable>>>
MlirCompiler::RunBackendOnModuleGroup(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("Not yet implemented in MLIR compiler");
}

StatusOr<std::vector<std::unique_ptr<Executable>>> MlirCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_execs,
    se::DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("Not yet implemented in MLIR compiler");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
MlirCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                 const AotCompilationOptions& options) {
  return Unimplemented("Not yet implemented in MLIR compiler");
}

}  // namespace mlir
}  // namespace xla

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::cuda::kCudaPlatformId, []() {
        return absl::make_unique<xla::FailoverCompiler>(
            absl::make_unique<xla::mlir::MlirCompiler>(),
            absl::make_unique<xla::gpu::NVPTXCompiler>(
                stream_executor::cuda::kCudaPlatformId));
      });
  return true;
}
static bool module_initialized = InitModule();
