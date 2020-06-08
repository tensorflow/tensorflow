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

#include <memory>

#include "llvm/IR/Module.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace mlir_gpu {
namespace {

using ::mlir::MLIRContext;
using ::mlir::LLVM::LLVMDialect;

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

void MlirCompiler::SetModuleHook(IRHook module_hook) {
  module_hook_ = module_hook;
}

void MlirCompiler::RemoveModuleHook() {
  module_hook_ = {nullptr, IRHook::LoweringStage::LHLO};
}

void MlirCompiler::SetErrorHandler(ErrorHandler error_handler) {
  error_handler_ = error_handler;
}

void MlirCompiler::RemoveErrorHandler() { error_handler_ = nullptr; }

}  // namespace mlir_gpu
}  // namespace xla
