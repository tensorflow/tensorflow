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

#ifndef XLA_RUNTIME_JIT_EXECUTABLE_H_
#define XLA_RUNTIME_JIT_EXECUTABLE_H_

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "tensorflow/compiler/xla/mlir/transforms/runtime/jit_compiler.h"
#include "tensorflow/compiler/xla/runtime/async_values_cache.h"
#include "tensorflow/compiler/xla/runtime/constraints.h"

namespace xla {
namespace runtime {

// JitExecutable owns a default executable compiled from the MLIR module (if
// operands constraints allow that), and orchestrates on-demand re-compilation
// for specific argument ranks, shapes or values depending on the operands
// constraints.
class JitExecutable {
 public:
  using UserData = std::any;

  // XLA program can be specialized and recompiled at runtime to the concrete
  // input shapes and sometimes values (e.g. reduction dimension).
  enum class Specialization {
    // Recompile specialized kernels when needed.
    kEnabled,
    // Completely disable specialized kernels (always call default executable).
    kDisabled,
    // Always use specialized kernels, and never call default executable (only
    // required for getting reproducible results in benchmarks).
    kAlways,
  };

  struct Options {
    // What level of specialization is enabled at runtime.
    Specialization specialization = Specialization::kAlways;

    // Options for the XLA runtime JitCompiler.
    JitCompiler::Options compiler;
  };

  // We use `llvm::unique_function` to represent compilation task because it
  // allows to capture move-only values.
  using CompilationTask = llvm::unique_function<void()>;

  // Compilation task runner called at runtime when specialization compilation
  // is required with the `TaskFunction` that does the compilation, and updates
  // the internal state of the `JitExecutable`. This runner can be used by the
  // caller to offload compilation task to the specialized thread pool and
  // add tracing events (e.g. add Tensorflow profiler tracing). Task runner must
  // call the `TaskFunction`, otherwise it will lead to deadlock.
  //
  // Caller can pass arbitrary user data to the `GetExecutable` method, and it
  // will be passed to the runner if recompilation is required. It is guaranteed
  // that the runner will be called in the same thread as `GetExecutable`.
  //
  using CompilationTaskRunner =
      llvm::unique_function<void(size_t, absl::Span<const ArgumentConstraint>,
                                 ArgumentsRef, CompilationTask, UserData)>;

  // Inline compilation task runner runs compilation task in the caller thread.
  static void InlineCompilationTaskRunner(
      size_t num_specializations,
      absl::Span<const ArgumentConstraint> constraints, ArgumentsRef arguments,
      CompilationTask task, UserData user_data);

  static llvm::Expected<JitExecutable> Instantiate(
      std::string_view mlir_module, std::string_view entrypoint, Options opts,
      std::string_view memory_region_name = "",
      CompilationTaskRunner runner = InlineCompilationTaskRunner);

  // Returns entrypoint operands constraints after resolving them using the
  // statically known information in the entrypoint function signature.
  absl::Span<const ArgumentConstraint> constraints() const;

  // Returns default executable that accepts all compatible operands
  // (operands rank and all static dimensions should match the operands).
  tfrt::AsyncValuePtr<Executable> DefaultExecutable() const;

  // Returns an executable that may be specialized for the arguments. Can return
  // default executable if no specialization is required, or if the specialized
  // executable is not yet available.
  //
  // Caller can pass arbitrary data via the `user_data` argument, and it will be
  // available to the compilation task runner. This can be used for tracing,
  // e.g. to track what user-level requests triggered recompilation.
  //
  // Returns an error if the arguments do not match the expected function
  // signature and specialization is not possible (without trying to compile).
  // If specialization is disabled, returns the default executable without
  // checking the arguments (the default executable itself will check arguments
  // when called).
  //
  // Async values holding compilation results (executables) cached in the
  // JitExecutable, and successive calls with the same arguments are cheap (the
  // definition of "same" depend on the argument type specialization and chosen
  // hash function, e.g. shaped arguments compared using their symbolic shape).
  // If compilation fails, then the returned async value will hold a compilation
  // error message. Compilation errors are never retried.
  //
  // Note: This function never falls back on the default executable if
  // specialization compilation fails.
  llvm::Expected<tfrt::AsyncValuePtr<Executable>> GetExecutable(
      ArgumentsRef arguments, UserData user_data = {},
      const SpecializationListener* listener = nullptr);

  // Returns an async value that becomes ready when all executables owned by
  // this JitExecutable are compiled (no pending compilation tasks).
  tfrt::AsyncValueRef<tfrt::Chain> AllExecutablesCompiled() const;

  // JitExecutable is move-only type.
  JitExecutable(const JitExecutable&) = delete;
  JitExecutable(JitExecutable&&) = default;

 private:
  JitExecutable(std::string_view mlir_module, std::string_view entrypoint,
                std::string_view memory_region_name, Options opts,
                absl::Span<const ArgumentConstraint> constraints,
                FunctionType signature,
                std::optional<Executable> default_executable,
                CompilationTaskRunner runner);

  std::string mlir_module_;
  std::string entrypoint_;

  // Name of the memory region where JIT'ed code is compiled to.
  // This allows profilers to correctly label JIT-executed code.
  // Note: this feature might only be available on some platforms, e.g. Linux.
  std::string memory_region_name_;

  Options opts_;

  // Entrypoint operands constraints after resolving them using the statically
  // known information in the entrypoint function signature. If constraint
  // specified by the argument attribute known to be statically satisfied by the
  // operand type (e.g. rank constraint with an operand of statically known
  // rank), then the constraint value for that operand will be updated to
  // `kResolved`.
  llvm::SmallVector<ArgumentConstraint> constraints_;

  // True if any of the operands has `ArgumentConstraint::kValue` constraint.
  bool has_value_constraints_;

  // Signature of the compiled module entrypoint function.
  //
  // This function signature is allowed to have operands and results types
  // without a well-defined ABI (e.g. it can have tensors when compiled module
  // defined in Tensorflow dialect), and it corresponds to the kernel definition
  // in one of the high level dialects (e.g. Tensorflow or mHLO).
  //
  // When compiled module prepared for execution, function operands and results
  // are mapped to the types with well-defined ABI (e.g. tensors mapped to
  // memrefs). See `signature_` documentation in the `Executable` type.
  FunctionType signature_;

  // Symbolic shape resolver assigns symbolic dimensions to runtime operands
  // based on the entrypoint function signature.
  SymbolicShapesResolver symbolic_shapes_resolver_;

  // Default executable that was not specialized to any of the arguments.
  AsyncValueRef<Executable> default_executable_;
  bool has_default_executable_;

  // A custom runner for compiling specializations.
  CompilationTaskRunner runner_;

  // Executables specialized for the arguments shapes or/and values.
  using Specializations = AsyncValuesCache<llvm::hash_code, Executable>;
  std::unique_ptr<Specializations> specializations_;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_JIT_EXECUTABLE_H_
