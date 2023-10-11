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

#ifndef XLA_RUNTIME_EXECUTABLE_H_
#define XLA_RUNTIME_EXECUTABLE_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "xla/runtime/arguments.h"
#include "xla/runtime/async_runtime.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/diagnostics.h"
#include "xla/runtime/execution_engine.h"
#include "xla/runtime/logical_result.h"
#include "xla/runtime/memory_mapper.h"
#include "xla/runtime/results.h"
#include "xla/runtime/type_id.h"
#include "xla/runtime/types.h"

namespace xla {
namespace runtime {

struct ExecutionContext;

struct DestroyExecutionContext {
  void operator()(ExecutionContext* ctx);
};

// If executable has async results, ExecutionReference keeps that
// execution context alive. For sync executables `Execute` always returns
// ExecutionReference with nullptr.
class ExecutionReference
    : public std::unique_ptr<ExecutionContext, DestroyExecutionContext> {
  // Bring std::unique_ptr constructors in scope.
  using std::unique_ptr<ExecutionContext, DestroyExecutionContext>::unique_ptr;
};

class FunctionRef;
class JitCompiler;

// Returns a symbols binding for running XLA executable with a custom symbols
// provided by the user.
ExecutionEngine::SymbolsBinding RuntimeSymbolsBinding(
    ExecutionEngine::SymbolsBinding custom_binding);

// Converts a direct custom call and custom type id name registration functions
// (types required by the library) to the execution engine symbols binding.
// Returned symbols binding always includes type id symbols for all
// canonical types supported by the XLA runtime custom calls.
ExecutionEngine::SymbolsBinding ToSymbolsBinding(
    std::function<void(DirectCustomCallRegistry&)> custom_calls = {},
    std::function<void(TypeIDNameRegistry&)> types = {});

class Executable {
 public:
  // Forward declare types defined below.
  struct ArgumentsMemoryLayout;
  struct ResultsMemoryLayout;
  struct CallFrame;
  struct ExecuteOpts;

  // Initializes call frame by adding all arguments according to the exported
  // function ABI. Also allocates storage for the returned values according to
  // the results memory layout.
  //
  // If `verify_arguments` is true (in debug mode it's always on, independent of
  // the argument value) this function also verifies that arguments passed at
  // run time matches the exported function signature (e.g. all statically known
  // dimensions of the memrefs matches the arguments). Returns an error if finds
  // a mismatch.
  //
  // This function leaves the execution context argument (the first argument of
  // an exported function) uninitialized. It will be initialized in the
  // `Execute` function right before the actual execution.
  absl::Status InitializeCallFrame(unsigned ordinal, ArgumentsRef arguments,
                                   CallFrame* call_frame,
                                   bool verify_arguments = true) const;

  absl::Status InitializeCallFrame(ArgumentsRef arguments,
                                   CallFrame* call_frame,
                                   bool verify_arguments = true) const {
    return InitializeCallFrame(0, arguments, call_frame, verify_arguments);
  }

  // Converts returned values owned by the call frame using provided result
  // converter. If exported function execution finished with an error (error
  // flag is `true` in the call frame) returns error for all results (see
  // `ResultConverter::ReturnError` documentation).
  absl::Status ReturnResults(unsigned ordinal, const ResultConverter& results,
                             CallFrame* call_frame) const;

  absl::Status ReturnResults(const ResultConverter& results,
                             CallFrame* call_frame) const {
    return ReturnResults(0, results, call_frame);
  }

  // Executes exported function exported with given arguments.
  //
  // If `verify_arguments` is true (in debug mode it's always on, independent of
  // the argument value) this function also verifies that arguments passed at
  // run time matches the exported function signature. If some of the
  // arguments do not match the expected type, this function allocates error
  // async values for all results and returns an error.
  //
  // Returns exported function results via the user-provided results converter.
  // If execution completed in the error state, returns error for all results.
  absl::StatusOr<ExecutionReference> Execute(
      unsigned ordinal, ArgumentsRef arguments, const ResultConverter& results,
      const ExecuteOpts& opts, bool verify_arguments = true) const;

  absl::StatusOr<ExecutionReference> Execute(
      ArgumentsRef arguments, const ResultConverter& results,
      const ExecuteOpts& opts, bool verify_arguments = true) const {
    return Execute(0, arguments, results, opts, verify_arguments);
  }

  // Executes exported function using user provided call frame.
  //
  // It is the caller responsibility to handle the compiled function results
  // stored in the call frame.
  ExecutionReference Execute(unsigned ordinal, CallFrame& call_frame,
                             const ExecuteOpts& opts) const;

  void Execute(CallFrame& call_frame, const ExecuteOpts& opts) const {
    Execute(0, call_frame, opts);
  }

  std::string_view name() const { return name_; }

  std::optional<size_t> specialization() const { return specialization_; }

  // Returns the number of exported functions. Functions are indexed by their
  // ordinal number in the [0, num_functions) range.
  size_t num_functions() const { return functions_.size(); }

  // Returns a function reference to an exported function with given ordinal.
  FunctionRef function_ref(unsigned ordinal) const;

  // Returns true if exported function with given ordinal has async results.
  bool IsAsync(unsigned ordinal) const;
  bool IsAsync() const { return IsAsync(0); }

  // Returns the name of the exported function with the given ordinal.
  std::string_view function_name(unsigned ordinal) const;
  std::string_view function_name() const { return function_name(0); }

  // Returns the number of results of the exported function with given ordinal.
  unsigned num_results(unsigned ordinal) const;
  unsigned num_results() const { return num_results(0); }

  // Signature of the exported function with the given ordinal before lowering
  // to the runtime dialects. See JitExecutable::Function's `signature` for
  // more details.
  const FunctionType& signature(unsigned ordinal) const;
  const FunctionType& signature() const { return signature(0); }

  // Signature of the exported function with the given ordinal after lowering it
  // from high level dialects to the dialects supported by the XLA runtime. See
  // JitExecutable::Function's `signature` for more details.
  const FunctionType& runtime_signature(unsigned ordinal) const;
  const FunctionType& runtime_signature() const { return runtime_signature(0); }

  std::chrono::milliseconds time_to_compile() const;

  // Get the object file behind this executable (on linux for example, it will
  // be https://en.wikipedia.org/wiki/Executable_and_Linkable_Format
  // executable). Can be null.
  std::unique_ptr<llvm::MemoryBuffer> obj_file() const;

  // CallFrame provides a pointer-stable storage for packed function arguments
  // and storage for returned values.
  struct CallFrame {
    // Pointers to executable arguments.
    llvm::SmallVector<void*, 32> args;

    // We use single block of memory to store executable results. We need to be
    // able to store pointers to async values and tokens, and strided memrefs
    // which at runtime are represented as StridedMemrefType<T, rank>.
    //
    // Currently we only need to provide result storage for pointers and memref
    // sizes and strides (int64_t type). If we'll need to support more complex
    // return types we'll have to be more careful about alignment requirements.
    static_assert(sizeof(uintptr_t) == sizeof(int64_t),
                  "uintptr_t size must be the same as int64_t");

    // Memory where the executable will write its results.
    llvm::SmallVector<uint8_t, 128> results;

    // Tracks whether any of the outputs were set.
    bool has_set_outputs = false;

    // Indicates whether the execution finished with an error.
    bool is_error = false;

    // The error message which is available only if `is_error` is true. The
    // assumption is that the error message string is owned by the compiled
    // binary and the call frame can safely keep a non-owning pointer.
    std::string_view error;
  };

  // Requirements for passing arguments to the compiled function.
  struct ArgumentsMemoryLayout {
    size_t num_args_ptrs = 0;            // total number of required pointers
    llvm::SmallVector<size_t> num_ptrs;  // num_ptrs for each argument
    llvm::SmallVector<size_t> offsets;   // offsets into the args array
  };

  // Requirements for the contiguous block of memory to store compiled function
  // results. When we invoke a compiled fuction we allocate a block of memory,
  // and pass pointers to pre-computed offsets as output arguments to the
  // function.
  struct ResultsMemoryLayout {
    bool has_async_results = false;     // true iff returns async results
    size_t size = 0;                    // number of bytes required
    llvm::SmallVector<size_t> offsets;  // offsets in the block of memory
  };

  struct ExecuteOpts {
    // Async task runner for executing async runtime tasks. Typically it
    // schedules async tasks into the underlying thread pool. It's the caller's
    // responsibility to guarantee that it will outlive the execution of all
    // async tasks started by the executable.
    AsyncTaskRunner* async_task_runner = nullptr;

    // A container for passing arbitrary user-provided data to the custom call
    // handlers. Must outlive all async tasks launched by this executable.
    const CustomCall::UserData* custom_call_data = nullptr;

    // Dynamically registered custom calls library. These custom calls resolved
    // at run time by name. In contrast to custom calls defined by the
    // `DirectCustomCallRegistry` which are linked directly with the executable
    // at compile time.
    const DynamicCustomCallRegistry* custom_call_registry = nullptr;

    // Diagnostic engine is responsible for passing runtime diagnostics back
    // to the caller through the diagnostic handler.
    const DiagnosticEngine* diagnostic_engine = nullptr;
  };

  // Function specification for loading from the object file.
  struct LoadFunction {
    std::string name;
    FunctionType signature;
    FunctionType runtime_signature;
  };

  // Loads executable from an object file. It is the caller responsibility to
  // guarantee that signatures do match the compiled function in the object
  // file, otherwise it will surely lead to crash.
  static absl::StatusOr<Executable> LoadFromObjFile(
      std::string_view name, std::unique_ptr<llvm::MemoryBuffer> obj_file,
      std::vector<LoadFunction> load_functions,
      ExecutionEngine::SymbolsBinding symbols_binding = {},
      std::string_view memory_region_name = "");

  // Verifies that all arguments types in the exported function signature are
  // supported at run time. Returns a pre-computed layout for the function
  // arguments. If some arguments are not supported returns an error.
  static absl::StatusOr<ArgumentsMemoryLayout> GetArgumentsMemoryLayout(
      const FunctionType& signature);

  // Verifies that all results types in the exported function signature are
  // supported at run time . Returns a pre-computed layout for the function
  // results. If some results are not supported returns an error.
  static absl::StatusOr<ResultsMemoryLayout> GetResultsMemoryLayout(
      const FunctionType& signature);

  // TODO(ezhulenev): The following three functions should be decoupled from
  // the executable header file (maybe move them to runtime.h?) so that custom
  // call implementations do not have to depend on the `executable` target.

  // Returns the user data passed via the ExecuteOpts to the executable.
  static const CustomCall::UserData* GetUserData(ExecutionContext* ctx);

  // Returns the diagnostic engine passed via the ExecuteOpts to the executable.
  static const DiagnosticEngine* GetDiagnosticEngine(ExecutionContext* ctx);

  // Calls the custom call handler with the given runtime context, arguments,
  // attributes and results.
  static LogicalResult Call(ExecutionContext* ctx, CustomCall& call,
                            void** args, void** attrs, void** rets);

  bool RequiresBlas(int ordinal) const {
    return functions_[ordinal].requires_blas;
  }

 private:
  friend class JitCompiler;  // see `mlir/runtime/transforms/jit_compiler.h`

  // Executable exports multiple functions available for users to call into. At
  // run time they are referenced by their ordinal, so that we don't depend on
  // expensive by-name lookup on the hot path. We keep function name only for
  // debugging. Function ordinal is defined by its index in the `functions_`
  // vector.
  struct Function {
    Function(std::string_view name, ExecutionEngine::ExportedFunctionPtr fptr,
             FunctionType signature, FunctionType runtime_signature,
             ArgumentsMemoryLayout arguments_memory_layout,
             ResultsMemoryLayout results_memory_layout, bool requires_blas)
        : name(name),
          fptr(std::move(fptr)),
          signature(std::move(signature)),
          runtime_signature(std::move(runtime_signature)),
          arguments_memory_layout(std::move(arguments_memory_layout)),
          results_memory_layout(std::move(results_memory_layout)),
          requires_blas(requires_blas) {}
    Function(const Function&) = delete;
    Function(Function&&) = default;

    // Exported function name.
    std::string name;

    // Pointer to an exported function owned by the execution engine.
    ExecutionEngine::ExportedFunctionPtr fptr;

    // Signature of the exported function function before lowering to the
    // runtime dialects (see JitExecutable::Function's `signature`).
    FunctionType signature;

    // Signature of the exported function after lowering it from high level
    // dialects to the dialects supported by the XLA runtime.
    //
    // - Operands and results types converted to the types with well-defined ABI
    //   (e.g. tensors converted to memrefs).
    //
    // - First argument is always an execution context added to the function by
    //   the lowering pipeline.
    //
    // From this signatur, Executable infers how to pack runtime arguments
    // according to the expected memory layout, and how to convert results
    // returned from the JIT-compiled function into high level types (e.g. how
    // to convert StridedMemrefType into Tensorflow Tensor).
    //
    // To infer the type of the returned value, Executable looks at the type
    // defined by the `runtime_signature` to get the memory layout of the
    // returned value, and at the type defined by the `signature` to get the
    // type expected by the runtime.
    FunctionType runtime_signature;

    // Memory layout required for passing function arguments.
    ArgumentsMemoryLayout arguments_memory_layout;

    // Memory layout for returning function results.
    ResultsMemoryLayout results_memory_layout;

    // If this flag is true, then this function is outlined for cuda graph, and
    // cuBlas should be initiated when capturing the cuda graph.
    bool requires_blas;
  };

  Executable(std::string_view name,
             std::unique_ptr<XlaRuntimeMemoryMapper> memory_mapper,
             std::unique_ptr<ExecutionEngine> engine,
             std::vector<Function> functions,
             std::optional<size_t> specialization,
             std::chrono::milliseconds time_to_compile)
      : name_(name),
        memory_mapper_(std::move(memory_mapper)),
        engine_(std::move(engine)),
        functions_(std::move(functions)),
        specialization_(specialization),
        time_to_compile_(time_to_compile) {
    // All exported functions must have a non-null function pointer.
    assert(llvm::all_of(functions_, [](const Function& f) { return f.fptr; }));
  }

  std::string name_;  // name of the compiled executable

  // Called by `engine_`'s destructor; must appear before it.
  std::unique_ptr<XlaRuntimeMemoryMapper> memory_mapper_;  // optional

  // XLA runtime execution engine owns the LLVM ORC jit compilation stack.
  std::unique_ptr<ExecutionEngine> engine_;

  // Functions exported by this executable, indexed by function ordinal.
  std::vector<Function> functions_;

  // Specialization id if this executable is a specialization, or an empty
  // optional if this executable is a default one.
  std::optional<size_t> specialization_;

  // The time it took to compile this binary.
  std::chrono::milliseconds time_to_compile_;
};

// Function reference provides a function-like API for a function exported from
// the executabled with the given ordinal.
class FunctionRef {
 public:
  FunctionRef(const Executable* executable, unsigned ordinal);

  absl::StatusOr<ExecutionReference> operator()(
      ArgumentsRef arguments, const ResultConverter& results,
      const Executable::ExecuteOpts& opts, bool verify_arguments = true) const;

  bool RequiresBlas() const { return executable_->RequiresBlas(ordinal_); }

  unsigned ordinal() const { return ordinal_; }

 private:
  const Executable* executable_;
  unsigned ordinal_;
};

// Escape slashes, substituting them with double underscores to get a memory
// region name for the XlaRuntimeMemoryMapper.
//
// The profiler's UI might interpret slashes as callchain separators,
// whereas we want the region name to be shown in full.
inline std::string EscapeMemRegionName(std::string_view memory_region_name) {
  return llvm::join(llvm::split(memory_region_name, '/'), "__");
}

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_EXECUTABLE_H_
