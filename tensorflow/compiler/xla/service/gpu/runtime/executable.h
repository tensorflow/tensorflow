/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_EXECUTABLE_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/ffi.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/compiler/xla/runtime/module_registry.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/collectives.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/conv.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/cublas_lt_matmul.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/fft.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/gemm.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/graph_launch.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/kernel_launch.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace gpu {

// Register custom calls implementing Xla Gpu runtime.
void RegisterXlaGpuRuntimeCustomCalls(
    runtime::DirectCustomCallRegistry& registry);

// Register mapping from XLA (SE) enums/structs type ids to symbol names.
void RegisterXlaGpuTypeIdNames(runtime::TypeIDNameRegistry& registry);

// Register encoding for (L)MHLO attributes required by the runtime functions.
void RegisterXlaGpuAttrEncoding(runtime::CustomCallAttrEncodingSet& encoding);

// Xla Gpu program lowered to the Xla runtime dialects. Gpu runtime executable
// jit-compiles this program to an executable artifact (via lowering to LLVM).
//
// We have this program as an intermediate step between lowering from HLO to
// runtime executable to be able to introspect the compilation process. Once we
// have this program, the Xla gpu compiler job is done, and lowering to LLVM is
// the responsibility of backend-agnostic Xla runtime passes. This is the last
// stage when IR is still at a fairly high level of abstraction and has a lot of
// Gpu specific details in it.
struct GpuRuntimeProgram {
  GpuRuntimeProgram(std::string entry_point, std::string module,
                    std::vector<int64_t> buffer_sizes,
                    DebugOptions debug_options)
      : entry_point(std::move(entry_point)),
        module(std::move(module)),
        buffer_sizes(std::move(buffer_sizes)),
        debug_options(std::move(debug_options)) {}

  std::string entry_point;
  std::string module;
  std::vector<int64_t> buffer_sizes;
  DebugOptions debug_options;
};

// Gpu runtime executable encapsulates the Xla runtime executable compiled from
// an Xla program and owns all the state required for running it (e.g. it owns
// various caches required for performance).
//
// TODO(ezhulenev): Once thunks are removed from Xla, it might make sense to
// merge this executable into GpuExecutable. Today we keep it separate to manage
// the complexity of mixing two execution modes in the same file. GpuExecutable
// provides an API at XLA level of abstraction (streams and buffers), and this
// executable provides a lower level API exposing some of the implementation
// details.
class GpuRuntimeExecutable {
  using ModulesState = ::xla::runtime::ModulesState;
  using FfiModulesState = ::xla::runtime::ffi::FfiModulesState;

 public:
  // Creates GpuRuntimeExecutable from the Xla Gpu Program.
  static StatusOr<std::unique_ptr<GpuRuntimeExecutable>> Create(
      std::unique_ptr<GpuRuntimeProgram> program);

  // Creates GpuRuntimeExecutable from the AOT compiled binary.
  static StatusOr<std::unique_ptr<GpuRuntimeExecutable>> Create(
      absl::Span<const int64_t> buffer_sizes, runtime::Executable executable,
      DebugOptions debug_options);

  // Executes entry function with the given buffer arguments.
  Status Execute(const ServiceExecutableRunOptions* run_options,
                 const std::string& asm_text,
                 const std::vector<uint8_t>& binary,
                 const BufferAllocations& buffer_allocations,
                 const BufferAllocation* temp_alloc = nullptr);

  // Returns object file behind the runtime executable. This object file can
  // be exported and loaded later to instantiate another executable.
  StatusOr<std::string_view> GetObjFile() const;

  // Returns MLIR module behind this executable if it is available.
  StatusOr<std::string_view> GetMlirModule() const;

 private:
  GpuRuntimeExecutable(std::vector<int64_t> buffer_sizes,
                       std::unique_ptr<runtime::JitExecutable> jit_executable,
                       DebugOptions debug_options, ModulesState modules_state,
                       FfiModulesState ffi_modules_state);

  GpuRuntimeExecutable(std::vector<int64_t> buffer_sizes,
                       std::unique_ptr<runtime::Executable> aot_executable,
                       DebugOptions debug_options, ModulesState modules_state,
                       FfiModulesState ffi_modules_state);

  // Depending on the state of `executable_` returns a reference to active
  // Xla runtime executable.
  runtime::Executable& executable();

  std::vector<int64_t> buffer_sizes_;

  // In JIT compilation mode `JitExecutable` is used. In AOT compilation mode
  // `Executable` is used.
  std::variant<std::unique_ptr<runtime::JitExecutable>,
               std::unique_ptr<runtime::Executable>>
      executable_;

  const DebugOptions debug_options_;

  // Keep gpu kernels loaded by this executable.
  GpuExecutableKernels gpu_kernels_;

  // Keep gemm configs for all gemm operation in the program.
  GemmConfigs gemm_configs_;

  // Keep a cache for conv configs for all conv operations in the program.
  ConvRunners conv_runners_;

  // Support for running collective operations.
  CollectivesSupport collectives_;

  // Keep a cache of fft plans for all FFT operations in the program.
  FftPlans fft_plans_;

#if GOOGLE_CUDA
  // Keep matmul execution plans (only if cuBLASLt is available).
  MatmulPlans cublas_lt_matmul_plans_;

  // Keep captured and instantiated CUDA graphs instances.
  GraphInstances graph_instances_;
#endif  // GOOGLE_CUDA

  // Keep an executable state for all registered runtime modules.
  ModulesState modules_state_;

  // Keeps an executable state for all registered FFI modules.
  FfiModulesState ffi_modules_state_;

  // Dynamic custom calls exported from XLA runtime modules (and FFI modules).
  runtime::DynamicCustomCallRegistry dynamic_custom_calls_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_EXECUTABLE_H_
