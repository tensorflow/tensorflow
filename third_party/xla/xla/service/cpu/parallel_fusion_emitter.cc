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

#include "xla/service/cpu/parallel_fusion_emitter.h"

#include <cstdint>
#include <memory>
#include <stack>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/bind_front.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/fusion_emitter.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::cpu {

struct ParallelFusionEmitter::CompilerInstance {
  std::unique_ptr<mlir::MLIRContext> mlir_context;
  std::unique_ptr<FusionCompiler> compiler;
};

class ParallelFusionEmitter::FusionCompilerPool {
 public:
  explicit FusionCompilerPool(FusionCompiler::Options options,
                              FusionCompiler::CompilationHooks hooks)
      : options_(options), hooks_(std::move(hooks)) {}

  ~FusionCompilerPool();

  // Get a single fusion compiler instance from the pool.
  // When the shared_ptr is destroyed, the instance is returned to the pool.
  std::shared_ptr<CompilerInstance> GetInstance();

 private:
  std::shared_ptr<CompilerInstance> CreateSharedInstance(
      CompilerInstance instance)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(instances_mutex_);
  void RecycleCompilerInstance(CompilerInstance* instance);
  // Wrap the held hooks to be able to pass them to the nested compiler without
  // a copy.
  FusionCompiler::CompilationHooks GetNestedHooks() const;

  FusionCompiler::Options options_;
  FusionCompiler::CompilationHooks hooks_;

  absl::Mutex instances_mutex_;
  int64_t outstanding_instances_ ABSL_GUARDED_BY(instances_mutex_) = 0;
  std::stack<CompilerInstance> instances_ ABSL_GUARDED_BY(instances_mutex_);
};

ParallelFusionEmitter::FusionCompilerPool::~FusionCompilerPool() {
  // We must wait for all instances to be returned to the pool before
  // destroying it.
  absl::MutexLock lock(instances_mutex_);
  instances_mutex_.Await(absl::Condition(
      +[](int64_t* outstanding_instances) {
        return *outstanding_instances == 0;
      },
      &outstanding_instances_));
}

auto ParallelFusionEmitter::FusionCompilerPool::GetInstance()
    -> std::shared_ptr<CompilerInstance> {
  absl::MutexLock lock(instances_mutex_);
  if (!instances_.empty()) {
    CompilerInstance instance = std::move(instances_.top());
    instances_.pop();
    return CreateSharedInstance(std::move(instance));
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();

  auto compiler = std::make_unique<FusionCompiler>(mlir_context.get(), options_,
                                                   GetNestedHooks());

  return CreateSharedInstance({std::move(mlir_context),
                               std::move(compiler)});
}

auto ParallelFusionEmitter::FusionCompilerPool::CreateSharedInstance(
    CompilerInstance instance) -> std::shared_ptr<CompilerInstance> {
  outstanding_instances_++;
  return std::shared_ptr<CompilerInstance>(
      new CompilerInstance(std::move(instance)),
      absl::bind_front(&FusionCompilerPool::RecycleCompilerInstance, this));
}

void ParallelFusionEmitter::FusionCompilerPool::RecycleCompilerInstance(
    CompilerInstance* instance) {
  absl::MutexLock lock(instances_mutex_);
  outstanding_instances_--;
  instances_.push(std::move(*instance));
  delete instance;
}

FusionCompiler::CompilationHooks
ParallelFusionEmitter::FusionCompilerPool::GetNestedHooks() const {
  using HookFnRef = absl::FunctionRef<void(mlir::ModuleOp) const>;

  FusionCompiler::CompilationHooks new_hooks;
  if (hooks_.pre_optimization) {
    new_hooks.pre_optimization = HookFnRef(hooks_.pre_optimization);
  }
  if (hooks_.post_optimization) {
    new_hooks.post_optimization = HookFnRef(hooks_.post_optimization);
  }
  if (hooks_.post_lowering) {
    new_hooks.post_lowering = HookFnRef(hooks_.post_lowering);
  }

  return new_hooks;
}

ParallelFusionEmitter::ParallelFusionEmitter(
    tsl::thread::ThreadPool& thread_pool, FusionCompiler::Options options,
    FusionCompiler::CompilationHooks hooks,
    const BufferAssignment* buffer_assignment, bool use_unique_c_name,
    bool enable_tiled_emitter)
    : thread_pool_(thread_pool),
      fusion_compiler_pool_(
          std::make_unique<FusionCompilerPool>(options, std::move(hooks))),
      buffer_assignment_(buffer_assignment),
      use_unique_c_name_(use_unique_c_name),
      enable_tiled_emitter_(enable_tiled_emitter) {}

ParallelFusionEmitter::~ParallelFusionEmitter() {
  absl::MutexLock lock(kernels_mutex_);
  kernels_mutex_.Await(absl::Condition(
      +[](int64_t* outstanding_kernels) { return *outstanding_kernels == 0; },
      &outstanding_kernels_));
}

absl::StatusOr<KernelSpec> ParallelFusionEmitter::AddFusion(
    const HloFusionInstruction* fusion) {
  // Ideally we would do the emitting in addition to the compilation in the
  // background thread, but as the ThunkEmitter requires the kernel spec to be
  // returned immediately, we have to do it in the main thread. This can be
  // fixed but will require a rework of the ThunkEmitter.
  auto compiler_instance = fusion_compiler_pool_->GetInstance();
  TF_ASSIGN_OR_RETURN(
      KernelDefinition mlir_kernel_definition,
      EmitFusionKernel(*compiler_instance->mlir_context, *fusion,
                       buffer_assignment_, use_unique_c_name_,
                       enable_tiled_emitter_));

  {
    absl::MutexLock lock(kernels_mutex_);
    outstanding_kernels_++;
  }

  KernelSpec spec = mlir_kernel_definition.spec();
  auto shared_source = std::make_shared<KernelDefinition<MlirKernelSource>>(
      std::move(mlir_kernel_definition));

  thread_pool_.Schedule(absl::bind_front(&ParallelFusionEmitter::CompileFusion,
                                         this, std::move(shared_source),
                                         std::move(compiler_instance)));

  return spec;
}

absl::StatusOr<std::vector<KernelDefinition<LlvmKernelSource>>>
ParallelFusionEmitter::ConsumeKernels() {
  absl::MutexLock lock(kernels_mutex_);

  kernels_mutex_.Await(absl::Condition(
      +[](int64_t* outstanding_kernels) { return *outstanding_kernels == 0; },
      &outstanding_kernels_));

  if (!kernels_status_.ok()) {
    return kernels_status_;
  }

  // Sort the kernels by name to ensure a deterministic order.
  absl::c_sort(kernels_, [](const KernelDefinition<LlvmKernelSource>& lhs,
                            const KernelDefinition<LlvmKernelSource>& rhs) {
    return lhs.spec().name() < rhs.spec().name();
  });

  return std::move(kernels_);
}

void ParallelFusionEmitter::CompileFusion(
    std::shared_ptr<KernelDefinition<MlirKernelSource>> mlir_kernel,
    std::shared_ptr<CompilerInstance> compiler_instance) {
  KernelSpec spec = mlir_kernel->spec();
  absl::StatusOr<LlvmKernelSource> llvm_kernel_source =
      compiler_instance->compiler->Compile(
          std::move(*mlir_kernel).TakeSource());

  absl::MutexLock lock(kernels_mutex_);
  outstanding_kernels_--;

  if (!llvm_kernel_source.ok()) {
    kernels_status_.Update(llvm_kernel_source.status());
    return;
  }

  kernels_.emplace_back(std::move(spec), std::move(*llvm_kernel_source));
}

}  // namespace xla::cpu
