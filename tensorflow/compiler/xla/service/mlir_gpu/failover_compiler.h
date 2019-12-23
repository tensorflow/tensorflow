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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_FAILOVER_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_FAILOVER_COMPILER_H_

#include <memory>

#include "tensorflow/compiler/xla/service/compiler.h"

namespace xla {

// FailoverCompiler implements a compiler that fails over between a primary
// and secondary compiler.
//
// For all methods, first the primary compiler is invoked. If that compiler's
// implementation of the method fails with an unimplemented error, the
// secondary's compiler method is invoked. In all other cases, the result of
// the primary compiler's method is returned.
//
// The primary compiler is invoked on a clone of the supplied HloModule. This
// ensures that partial updates to the module by one compiler to not leak into
// the other compiler.
//
// The FailoverCompiler is used to layer a partial compiler implementation on
// top of a full implementation.
class FailoverCompiler final : public Compiler {
 public:
  FailoverCompiler(std::unique_ptr<Compiler> primary,
                   std::unique_ptr<Compiler> secondary)
      : primary_(std::move(primary)), secondary_(std::move(secondary)) {
    // Both compilers should serve the same platform id.
    assert(primary_->PlatformId() == secondary_->PlatformId());
  }

  se::Platform::Id PlatformId() const override {
    return primary_->PlatformId();
  }

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_execs,
      se::DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  Compiler* GetPrimary() const { return primary_.get(); }
  Compiler* GetSecondary() const { return secondary_.get(); }

 private:
  std::unique_ptr<Compiler> primary_;
  std::unique_ptr<Compiler> secondary_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_FAILOVER_COMPILER_H_
