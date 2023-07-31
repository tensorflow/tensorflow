/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_OPENXLA_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_OPENXLA_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

//===----------------------------------------------------------------------===//
// TODO(ezhulenev): We currently do not build with OpenXLA runtime in open
// source because we do not have bazel dependency from XLA to IREE.
#if XLA_DISABLE_OPENXLA_RUNTIME
//===----------------------------------------------------------------------===//

namespace xla::gpu {
struct OpenXlaRuntimeProgram {
  OpenXlaRuntimeProgram(std::unique_ptr<mlir::MLIRContext>,
                        mlir::OwningOpRef<mlir::ModuleOp>, std::string,
                        std::vector<int64_t>, DebugOptions) {}
};

struct OpenXlaRuntimeExecutable {
  static StatusOr<std::unique_ptr<OpenXlaRuntimeExecutable>> Create(
      std::unique_ptr<OpenXlaRuntimeProgram>, std::string_view,
      const std::vector<uint8_t>&) {
    return absl::UnimplementedError(
        "OpenXLA runtime is not supported in OSS build");
  }

  Status Execute(const ServiceExecutableRunOptions* run_options,
                 const BufferAllocations& buffer_allocations,
                 const BufferAllocation* temp_alloc) {
    return absl::UnimplementedError(
        "OpenXLA runtime is not supported in OSS build");
  }
};

}  // namespace xla::gpu

//===----------------------------------------------------------------------===//
#else  // !XLA_DISABLE_OPENXLA_RUNTIME
//===----------------------------------------------------------------------===//

#include <string>
#include <utility>

#include "third_party/iree/runtime/src/iree/vm/api.h"  // IWYU pragma: keep
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/openxla/compiler.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace gpu {

// Forward declare.
struct OpenXlaDevice;

// Xla Gpu program lowered to the OpenXLA dialects (IREE input dialects).
// OpenXLA runtime executable jit-compiles this program to an executable
// artifact (via lowering to IREE VM executable).
//
// Top level module has a single HAL executable source that contains all device
// kernels for an XLA module. After lowering from LMHLO executable source is
// just a placeholder; it gets updated with a real device kernels source only
// before we pass IR to OpenXLA/IREE compiler (see OpenXlaRuntimeExecutable
// below), because in the XLA compilation pipeline backend compiler runs last.
//
// We have this program as an intermediate step between lowering from LMHLO to
// VM executable to be able to introspect the compilation process. Once we have
// this program, the Xla gpu compiler job is done, and lowering to IREE VM is
// the responsibility of OpenXLA/IREE compiler.
struct OpenXlaRuntimeProgram {
  OpenXlaRuntimeProgram(std::unique_ptr<mlir::MLIRContext> ctx,
                        mlir::OwningOpRef<mlir::ModuleOp> module,
                        std::string entry_point,
                        std::vector<int64_t> buffer_sizes,
                        DebugOptions debug_options)
      : ctx(std::move(ctx)),
        module(std::move(module)),
        entry_point(std::move(entry_point)),
        buffer_sizes(std::move(buffer_sizes)),
        debug_options(std::move(debug_options)) {}

  std::unique_ptr<mlir::MLIRContext> ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string entry_point;
  std::vector<int64_t> buffer_sizes;
  DebugOptions debug_options;
};

// Gpu runtime executable encapsulates the Xla runtime executable compiled from
// an Xla program and owns all the state required for running it (e.g. it owns
// various caches required for performance).
class OpenXlaRuntimeExecutable {
 public:
  using Bytecode = OpenXlaCompiler::Bytecode;

  // Creates OpenXlaRuntimeExecutable from the OpenXLA program.
  static StatusOr<std::unique_ptr<OpenXlaRuntimeExecutable>> Create(
      std::unique_ptr<OpenXlaRuntimeProgram> program, std::string_view asm_text,
      const std::vector<uint8_t>& binary);

  ~OpenXlaRuntimeExecutable();

  // Executes entry function with the given buffer arguments.
  Status Execute(const ServiceExecutableRunOptions* run_options,
                 const BufferAllocations& buffer_allocations,
                 const BufferAllocation* temp_alloc);

 private:
  OpenXlaRuntimeExecutable(
      std::unique_ptr<OpenXlaDevice> device, std::unique_ptr<Bytecode> bytecode,
      std::vector<int64_t> buffer_sizes, DebugOptions debug_options,
      std::string_view asm_text, absl::Span<const uint8_t> binary,
      iree::vm::ref<iree_vm_context_t> context,
      iree::vm::ref<iree_vm_instance_t> instance,
      std::unique_ptr<std::vector<iree_vm_module_t*>> modules,
      iree_vm_function_t function);

  // TODO(ezhulenev): Devices should be created lazily for each StreamExecutor
  // and share underlying resources. For now we create a CUDA driver and CUDA
  // HAL device for each executable. And we assume that we have just once GPU
  // attached to the host, and always run on device with ordinal 0.
  std::unique_ptr<OpenXlaDevice> device_;

  std::unique_ptr<Bytecode> bytecode_;

  std::vector<int64_t> buffer_sizes_;
  const DebugOptions debug_options_;

  std::string_view asm_text_;
  absl::Span<const uint8_t> binary_;

  // TODO(ezhulenev): VM context and instance should be shared between multiple
  // executables. Also HAL module should be loaded just once. This has to be
  // fixed together with efficient device sharing, because HAL VM module
  // requires HAL device for loading.
  iree::vm::ref<iree_vm_context_t> context_;
  iree::vm::ref<iree_vm_instance_t> instance_;
  std::unique_ptr<std::vector<iree_vm_module_t*>> modules_;
  iree_vm_function_t function_;
};

}  // namespace gpu
}  // namespace xla

#endif  // !XLA_DISABLE_OPENXLA_RUNTIME
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_OPENXLA_EXECUTABLE_H_
