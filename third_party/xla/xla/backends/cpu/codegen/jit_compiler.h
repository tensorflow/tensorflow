/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_JIT_COMPILER_H_
#define XLA_BACKENDS_CPU_CODEGEN_JIT_COMPILER_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {

// Jit compiler that compiles LLVM modules added to it into a FunctionLibrary.
// Jit-compiled function library will be backed by multiple dynamic libraries
// compiled from LLVM modules using LLVM ORC APIs.
//
// JitCompiler is an opinionated JIT compiler built on top of LLVM ORC stack,
// optimized for compiling LLVM modules produced by XLA:CPU. LLVM itself
// has another pre-fabricated ORC JIT stack called `llvm::orc::LLJIT`.
class JitCompiler {
 public:
  using Symbol = FunctionLibrary::Symbol;

  // Task and a TaskRunner are used to run compilation tasks in parallel.
  using Task = std::function<void()>;  // NOLINT (must be copyable)
  using TaskRunner = absl::AnyInvocable<void(Task)>;

  // A callback that returns a definition generator that will be added to all
  // dynamic libraries created by the jit compiler. Definition generator enables
  // linking host runtime symbols into the jit-compiled function library.
  using DefinitionGenerator =
      std::function<std::unique_ptr<llvm::orc::DefinitionGenerator>(
          llvm::TargetMachine*)>;

  JitCompiler(JitCompiler&&) = default;
  JitCompiler& operator=(JitCompiler&&) = default;

  ~JitCompiler();

  // Infers the `llvm::TargetMachine` for the current host. If `max_cpu_feature`
  // is provided, it will be used to constrain the set of features that LLVM
  // codegen (instruction selection) is allowed to use, e.g. it can be used to
  // explicitly disable certain AVX512 extensions, in case the compiled
  // executable will be serialized and later loaded on a different machine.
  static absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
  InferTargetMachine(const llvm::TargetOptions& target_options,
                     llvm::CodeGenOptLevel opt_level,
                     std::optional<tsl::port::CPUFeature> max_cpu_feature);

  // Returns a target machine builder that uses `InferTargetMachine` defined
  // above to infer the target machine for the given options.
  static IrCompiler::TargetMachineBuilder InferTargetMachineBuilder(
      const llvm::TargetOptions& target_options,
      llvm::CodeGenOptLevel opt_level,
      std::optional<tsl::port::CPUFeature> max_cpu_feature);

  struct Options {
    // Options for the underlying IR compiler instance.
    IrCompiler::Options ir_compiler_options;
    IrCompiler::CompilationHooks ir_compiler_hooks;

    // The number of dynamic libraries to create for the jit compiler instance.
    // We compile XLA:CPU program into multiple LLVM modules, and by using
    // multiple dynamic libraries we enable parallel compilation.
    size_t num_dylibs = 1;

    // Optional definition generator to inject host runtime symbols into the
    // jit-compiled function library.
    DefinitionGenerator definition_generator;

    // Maximum CPU instruction set for wich the compiler should generate code.
    // If instruction set is empty, compiler will generate code for all ISA
    // extensions detected on the current machine.
    std::optional<tsl::port::CPUFeature> max_cpu_feature;
  };

  // Creates a new instance of the JitCompiler.
  static absl::StatusOr<JitCompiler> Create(llvm::TargetOptions target_options,
                                            Options options,
                                            TaskRunner task_runner);

  // Adds a LLVM module to the dynamic library at `dylib_index`.
  absl::Status AddModule(llvm::orc::ThreadSafeModule module,
                         size_t dylib_index = 0);

  // Adds an object file to the dynamic library at `dylib_index`.
  absl::Status AddObjFile(std::unique_ptr<llvm::MemoryBuffer> obj_file,
                          size_t dylib_index = 0);

  // Compiles all added LLVM modules and object files into the FunctionLibrary
  // by resolving all symbols in `symbols`.
  //
  // After this method returns, the FunctionLibrary will contain compiled
  // functions that can be invoked via function calls. Returned FunctionLibrary
  // tracks type ids of the resolved symbols, but the compiler doesn't verify
  // that LLVM IR function signature matches the type id, and it's up to the
  // user to make sure that function types actually match, otherwise it will
  // lead to run-time crashes.
  //
  // TODO(ezhulenev): Add an option to pass symbol (function) types at compile
  // time together with names and type-check LLVM function signature against the
  // function type to make compilation process type safe. Currently we only keep
  // track of type ids, but we don't track function signatures for type ids, and
  // have a simple run-time type checking inside of the FunctionLibrary.
  absl::StatusOr<std::unique_ptr<FunctionLibrary>> Compile(
      absl::Span<const Symbol> symbols) &&;

  llvm::TargetMachine* target_machine() { return target_machine_.get(); }

 private:
  // LLVM ORC task dispatcher that uses `TaskRunner` to run compilation tasks.
  class TaskDispatcher : public llvm::orc::TaskDispatcher {
   public:
    explicit TaskDispatcher(TaskRunner task_runner);
    ~TaskDispatcher() final;

    void dispatch(std::unique_ptr<llvm::orc::Task> T) final;
    void shutdown() final;

   private:
    TaskRunner task_runner_;

    absl::Mutex mu_;
    absl::CondVar cv_;
    size_t num_dispatched_tasks_ ABSL_GUARDED_BY(mu_) = 0;
  };

  // Function library constructed from the set of jit-compiled symbols.
  class CompiledFunctionLibrary : public FunctionLibrary {
   public:
    struct ResolvedSymbol {
      TypeId type_id;
      void* ptr;
    };

    CompiledFunctionLibrary(
        std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
        std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer,
        absl::flat_hash_map<std::string, ResolvedSymbol> symbols_map);

    ~CompiledFunctionLibrary() final;

    absl::StatusOr<void*> ResolveFunction(TypeId type_id,
                                          std::string_view name) final;

   private:
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session_;
    std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer_;
    absl::flat_hash_map<std::string, ResolvedSymbol> symbols_map_;
  };

  JitCompiler(IrCompiler::TargetMachineBuilder target_machine_builder,
              std::shared_ptr<llvm::TargetMachine> target_machine,
              TaskDispatcher* task_dispatcher,
              std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
              std::unique_ptr<IrCompiler> ir_compiler, size_t num_dylibs,
              DefinitionGenerator definition_generator);

  // Target machine builder that is used to construct target machines for this
  // instance of `JitCompiler` (when compiling LLVM modules in parallel).
  IrCompiler::TargetMachineBuilder target_machine_builder_;
  std::shared_ptr<llvm::TargetMachine> target_machine_;

  TaskDispatcher* task_dispatcher_;  // owned by `execution_session_`

  std::unique_ptr<llvm::orc::ExecutionSession> execution_session_;
  std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer_;
  std::unique_ptr<llvm::orc::IRCompileLayer> compile_layer_;

  // Non-owning pointers to dynamic libraries created for the execution session.
  std::vector<llvm::orc::JITDylib*> dylibs_;

  // Non owning pointer to JIT event listeners for gdb and perf.
  llvm::JITEventListener* gdb_;   // not owned
  llvm::JITEventListener* perf_;  // not owned
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_JIT_COMPILER_H_
