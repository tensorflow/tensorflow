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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_EXECUTABLE_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/async_runtime.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/diagnostics.h"
#include "tensorflow/compiler/xla/runtime/execution_engine.h"
#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/compiler/xla/runtime/memory_mapper.h"
#include "tensorflow/compiler/xla/runtime/results.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"
#include "tensorflow/compiler/xla/runtime/types.h"

namespace xla {
namespace runtime {

class ExecutionContext;
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

  // Initializes call frame by adding all arguments according to the executable
  // ABI. Also allocates storage for the returned values according to the
  // results memory layout.
  //
  // If `verify_arguments` is true (in debug mode it's always on, independent of
  // the argument value) this function also verifies that operands passed at run
  // time matches the executable entrypoint signature (e.g. all statically known
  // dimensions of the memrefs matches the operands). Returns an error if finds
  // a mismatch.
  //
  // This function leaves the execution context argument (the first argument of
  // an entry function) uninitialized. It will be initialized in the `Execute`
  // function right before the actual execution.
  absl::Status InitializeCallFrame(ArgumentsRef arguments,
                                   CallFrame* call_frame,
                                   bool verify_arguments = true) const;

  // Converts returned values owned by the call frame using provided result
  // converter. If compiled function execution finished with an error (error
  // flag is `true` in the call frame) returns error for all results.
  absl::Status ReturnResults(const ResultConverter& results,
                             CallFrame* call_frame) const;

  // Executes compiled function with given arguments.
  //
  // If `verify_arguments` is true (in debug mode it's always on, independent of
  // the argument value) this function also verifies that arguments passed at
  // run time matches the executable entrypoint signature. If some of the
  // arguments do not match the expected type, this function allocates error
  // async values for all results and returns an error.
  //
  // Returns compiled function results via the user-provided results converter.
  // If execution completed in the error state, returns error for all results.
  absl::Status Execute(ArgumentsRef arguments, const ResultConverter& results,
                       const ExecuteOpts& opts,
                       bool verify_arguments = true) const;

  // Executes compiled function using user provided call frame.
  //
  // It is the caller responsibility to handle the compiled function results
  // stored in the call frame.
  void Execute(CallFrame& call_frame, const ExecuteOpts& opts) const;

  bool IsAsync() const { return results_memory_layout_.has_async_results; }

  std::string_view name() const { return name_; }

  std::optional<size_t> specialization() const { return specialization_; }

  // Returns the number of results in the runtime signature.
  unsigned num_results() const;

  // Signature of the compiled module entrypoint function before lowering to
  // the runtime dialects. See JitExecutable's `signature_` for more details.
  const FunctionType& signature() const;

  // Signature of the compiled module entrypoint function after lowering it from
  // high level dialects to the dialects supported by the XLA runtime.
  // See JitExecutable's `signature_` for more details.
  const FunctionType& runtime_signature() const;

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

  // Loads executable from an object file. It is the caller responsibility to
  // guarantee that signatures do match the compiled function in the object
  // file, otherwise it will surely lead to crash.
  static absl::StatusOr<Executable> LoadFromObjFile(
      std::string_view name, std::unique_ptr<llvm::MemoryBuffer> obj_file,
      std::string_view entrypoint, FunctionType signature,
      FunctionType runtime_signature,
      ExecutionEngine::SymbolsBinding symbols_binding = {},
      std::string_view memory_region_name = "");

  // Verifies that all operands types in the entrypoint function signature are
  // supported at run time . Returns a pre-computed layout for the function
  // arguments. If some arguments are not supported returns an error.
  static absl::StatusOr<ArgumentsMemoryLayout> GetArgumentsMemoryLayout(
      const FunctionType& signature);

  // Verifies that all results types in the entrypoint function signature are
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

 private:
  friend class JitCompiler;  // see `mlir/transforms/runtime/compiler.h`

  Executable(std::string_view name,
             std::unique_ptr<XlaRuntimeMemoryMapper> memory_mapper,
             std::unique_ptr<ExecutionEngine> engine, FunctionType signature,
             FunctionType runtime_signature,
             ArgumentsMemoryLayout arguments_memory_layout,
             ResultsMemoryLayout results_memory_layout,
             std::optional<size_t> specialization,
             std::chrono::milliseconds time_to_compile)
      : name_(name),
        memory_mapper_(std::move(memory_mapper)),
        engine_(std::move(engine)),
        fptr_(engine_->entrypoint()),
        signature_(std::move(signature)),
        runtime_signature_(std::move(runtime_signature)),
        arguments_memory_layout_(std::move(arguments_memory_layout)),
        results_memory_layout_(std::move(results_memory_layout)),
        specialization_(specialization),
        time_to_compile_(time_to_compile) {
    assert(fptr_ != nullptr && "executable function pointer must be not null");
  }

  std::string name_;  // name of the compiled executable

  // Called by `engine_`'s destructor; must appear before it.
  std::unique_ptr<XlaRuntimeMemoryMapper> memory_mapper_;  // optional

  // XLA runtime execution engine owns the LLVM ORC jit compilation stack.
  std::unique_ptr<ExecutionEngine> engine_;

  // Compiled function owned by the `engine_`.
  ExecutionEngine::EntrypointFunctionPtr fptr_;

  // Signature of the compiled module entrypoint function before lowering to
  // the runtime dialects (see JitExecutable `signature_` for more details).
  FunctionType signature_;

  // Signature of the compiled module entrypoint function after lowering it from
  // high level dialects to the dialects supported by the XLA runtime.
  //
  // - Operands and results types converted to the types with well-defined ABI
  //   (e.g. tensors converted to memrefs).
  //
  // - First argument is always a execution context added to the function by the
  //   lowering pipeline.
  //
  // From this signature executable infers how to pack runtime operands
  // according to the expected memory layout, and how to convert results
  // returned from the JIT-compiled function into high level types (e.g. how to
  // convert StridedMemrefType into Tensorflow Tensor).
  //
  // To infer the type of the returned value, executable looks at the type
  // defined by the `runtime_signature_` to get the memory layout of the
  // returned value, and at the type defined by the `signature_` to get the type
  // expected by the runtime.
  FunctionType runtime_signature_;

  ArgumentsMemoryLayout arguments_memory_layout_;
  ResultsMemoryLayout results_memory_layout_;

  // Specialization id if this executable is a specialization, or an empty
  // optional if this executable is a default one.
  std::optional<size_t> specialization_;

  // The time it took to compile this binary.
  std::chrono::milliseconds time_to_compile_;
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

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_EXECUTABLE_H_
