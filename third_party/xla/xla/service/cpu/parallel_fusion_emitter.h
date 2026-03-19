/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_PARALLEL_FUSION_EMITTER_H_
#define XLA_SERVICE_CPU_PARALLEL_FUSION_EMITTER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::cpu {

// Manages the parallel compilation of multiple fusion instructions.
// Using the provided thread pool to compile fusions concurrently and a pool of
// FusionCompiler instances to reduce initialization overhead.
class ParallelFusionEmitter {
 public:
  ParallelFusionEmitter(tsl::thread::ThreadPool& thread_pool,
                        FusionCompiler::Options options,
                        FusionCompiler::CompilationHooks hooks,
                        const BufferAssignment* buffer_assignment,
                        bool use_unique_c_name, bool enable_tiled_emitter);

  ~ParallelFusionEmitter();

  // Adds a fusion to the queue of fusions to be compiled.
  // Returns the kernel spec for the fusion.
  absl::StatusOr<KernelSpec> AddFusion(const HloFusionInstruction* fusion);

  // Returns the kernels for all the added fusions, blocks until all kernels
  // have been compiled.
  absl::StatusOr<std::vector<KernelDefinition<LlvmKernelSource>>>
  ConsumeKernels();

 private:
  struct CompilerInstance;
  class FusionCompilerPool;

  void CompileFusion(
      std::shared_ptr<KernelDefinition<MlirKernelSource>> mlir_kernel,
      std::shared_ptr<CompilerInstance> compiler_instance);

  tsl::thread::ThreadPool& thread_pool_;
  std::unique_ptr<FusionCompilerPool> fusion_compiler_pool_;
  const BufferAssignment* buffer_assignment_;
  bool use_unique_c_name_;
  bool enable_tiled_emitter_;

  absl::Mutex kernels_mutex_;
  int64_t outstanding_kernels_ ABSL_GUARDED_BY(kernels_mutex_) = 0;
  absl::Status kernels_status_ ABSL_GUARDED_BY(kernels_mutex_);
  std::vector<KernelDefinition<LlvmKernelSource>> kernels_
      ABSL_GUARDED_BY(kernels_mutex_);
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_PARALLEL_FUSION_EMITTER_H_
