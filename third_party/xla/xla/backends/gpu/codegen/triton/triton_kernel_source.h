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
#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TRITON_KERNEL_SOURCE_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TRITON_KERNEL_SOURCE_H_

#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/DebugStringHelper.h"
#include "xla/codegen/kernel_source.h"

namespace xla::gpu {

// Kernel JIT source that is backed by triton.
//
// The triton source is typically created by a fusion emitter from
// GPU backend.
class TritonKernelSource final : public KernelSource {
 public:
  explicit TritonKernelSource(mlir::OwningOpRef<mlir::ModuleOp> module)
      : module_(std::move(module)) {}

  TritonKernelSource(const TritonKernelSource& other) noexcept = delete;
  TritonKernelSource& operator=(TritonKernelSource& other) noexcept = delete;
  TritonKernelSource(TritonKernelSource&& other) noexcept = default;
  TritonKernelSource& operator=(TritonKernelSource&& other) noexcept = default;

  mlir::ModuleOp module() const { return *module_; }

  // Moves ownership of the module to the caller.
  mlir::OwningOpRef<mlir::ModuleOp> TakeModule() && {
    DCHECK(module_.get() != nullptr)
        << "Can't move ownership of the module owned by the TritonKernelSource";
    return std::move(module_);
  }

  std::string ToString() const final { return mlir::debugString(*module_); }

  void set_error_context_provider(
      absl::AnyInvocable<std::string() const> provider) {
    error_ctx_provider_ = std::move(provider);
  }

  const absl::AnyInvocable<std::string() const>& error_context_provider()
      const {
    return error_ctx_provider_;
  }

 private:
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  absl::AnyInvocable<std::string() const> error_ctx_provider_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TRITON_KERNEL_SOURCE_H_
