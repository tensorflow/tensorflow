/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_CUBIN_CUSTOM_KERNEL_COMPILER_H_
#define XLA_BACKENDS_GPU_CODEGEN_CUBIN_CUSTOM_KERNEL_COMPILER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// LlvmIrCompiler abstracts compilation of LLVM IR to target binary.
// Takes the LLVM module, device description, and debug options as input.
// Returns the compiled binary as a vector of bytes or an error status.
using LlvmIrCompiler = absl::AnyInvocable<absl::StatusOr<std::vector<uint8_t>>(
    llvm::Module& module, const stream_executor::DeviceDescription& descr,
    const DebugOptions& opts)>;

// Implementation of KernelCompiler that compiles LLVM IR to CUBIN format using
// a provided compilation function.
//
// Note: CubinCustomKernelCompiler utilizes provided threadpool.
// If threadpool is not provided, the compilation happens
// fully within this call, and the result is returned as an immediately ready
// Future.
class CubinCustomKernelCompiler : public KernelCompiler {
 public:
  CubinCustomKernelCompiler(LlvmIrCompiler compiler,
                            const se::DeviceDescription& gpu_device_info,
                            const DebugOptions& debug_options,
                            tsl::thread::ThreadPool* thread_pool = nullptr)
      : compiler_(std::move(compiler)),
        device_info_(gpu_device_info),
        debug_options_(debug_options),
        thread_pool_(thread_pool) {}

  xla::Future<std::unique_ptr<Thunk>> Compile(
      Thunk::ThunkInfo thunk_info, LlvmKernelSource kernel_source,
      const std::string& sanitized_kernel_name,
      const emitters::KernelArguments& kernel_arguments,
      const LaunchDimensions& launch_dimensions) override;

  xla::Future<LlvmKernelSource> CompileMlirToLlvm(
      const se::DeviceDescription& device, const HloModule& hlo_module,
      const std::string& entry_function_name, int unroll_factor,
      MlirKernelSource source, BorrowedMlirContext borrowed_context) override;

  xla::Future<std::vector<uint8_t>> CompileToPtx(
      LlvmKernelSource kernel_source) override;

 private:
  absl::StatusOr<std::vector<uint8_t>> CompileToPtxImpl(
      LlvmKernelSource kernel_source);

  absl::StatusOr<std::unique_ptr<Thunk>> CompileImpl(
      Thunk::ThunkInfo thunk_info, LlvmKernelSource kernel_source,
      const std::string& sanitized_kernel_name,
      const emitters::KernelArguments& kernel_arguments,
      const LaunchDimensions& launch_dimensions);

  LlvmIrCompiler compiler_;
  const se::DeviceDescription device_info_;
  const DebugOptions debug_options_;
  tsl::thread::ThreadPool* thread_pool_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_CUBIN_CUSTOM_KERNEL_COMPILER_H_
