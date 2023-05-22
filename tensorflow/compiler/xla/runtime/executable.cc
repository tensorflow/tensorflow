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

#include "tensorflow/compiler/xla/runtime/executable.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/ErrorOr.h"
#include "tensorflow/compiler/xla/mlir/runtime/utils/async_runtime_api.h"
#include "tensorflow/compiler/xla/mlir/runtime/utils/c_runner_utils.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/errors.h"
#include "tensorflow/compiler/xla/runtime/runtime.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"

namespace xla {
namespace runtime {

using absl::Status;
using absl::StatusOr;

using llvm::dyn_cast;

// ExecutionContext encapsulates all the data that is required to implement XLA
// Runtime <-> XLA Executable integration API.
struct ExecutionContext {
  // Results memory layout is owned by the executable, and stays alive after the
  // function execution completes.
  const Executable::ResultsMemoryLayout* results_memory_layout = nullptr;

  // CallFrame life time bound to the compiled function execution and destroyed
  // immediately when the function returns. Only the compiled function itself
  // reads the arguments and writes to the function results storage.
  Executable::CallFrame* call_frame = nullptr;

  // User-defined data for custom call handlers.
  const CustomCall::UserData* custom_call_data = nullptr;

  // User-defined custom call registry.
  const DynamicCustomCallRegistry* custom_call_registry = nullptr;

  // User-defined diagnostic engine for reporting diagnostics.
  const DiagnosticEngine* diagnostic_engine = nullptr;
};

void DestroyExecutionContext::operator()(ExecutionContext* ctx) { delete ctx; }

//===----------------------------------------------------------------------===//
// Conversion from custom calls and type id registries to symbols binding.
//===----------------------------------------------------------------------===//

ExecutionEngine::SymbolsBinding ToSymbolsBinding(
    std::function<void(DirectCustomCallRegistry&)> custom_calls,
    std::function<void(TypeIDNameRegistry&)> types) {
  return [=](llvm::orc::MangleAndInterner mangle) {
    llvm::orc::SymbolMap symbol_map;

    DirectCustomCallRegistry custom_call_registry;
    if (custom_calls) custom_calls(custom_call_registry);

    TypeIDNameRegistry type_registry;
    if (types) types(type_registry);

    // Always register canonical custom call types.
    PopulateCustomCallTypeIdNames(type_registry);

    // Register direct custom calls.
    using DirectCustomCall = DirectCustomCallRegistry::DirectCustomCall;
    custom_call_registry.ForEach([&](std::string_view name,
                                     DirectCustomCall custom_call) {
      symbol_map[mangle(name)] = {llvm::orc::ExecutorAddr::fromPtr(custom_call),
                                  llvm::JITSymbolFlags()};
    });

    // Register type id symbols.
    type_registry.ForEach([&](std::string_view name, TypeID type_id) {
      auto type_id_ptr =
          reinterpret_cast<std::uintptr_t>(type_id.getAsOpaquePointer());
      symbol_map[mangle(name)] = {llvm::orc::ExecutorAddr(type_id_ptr),
                                  llvm::JITSymbolFlags()};
    });

    return symbol_map;
  };
}

//===----------------------------------------------------------------------===//
// Register XLA runtime symbols with XLA execution engine.
//===----------------------------------------------------------------------===//

static llvm::orc::SymbolMap RuntimeApiSymbolMap(llvm::orc::MangleAndInterner);

//===----------------------------------------------------------------------===//
// Construct a symbols binding for XLA executable.
//===----------------------------------------------------------------------===//

ExecutionEngine::SymbolsBinding RuntimeSymbolsBinding(
    ExecutionEngine::SymbolsBinding custom_binding) {
  return ExecutionEngine::BindAll(
      {// Register MLIR C Runner API intrinsics (defined in CRunnerUtils).
       CRunnerUtilsSymbolMap,
       // Register Async Runtime API intrinsics.
       AsyncRuntimeApiSymbolMap,
       // Register memory allocation functions (malloc, free, ...).
       AsyncRuntimeMemoryAllocationSymbolMap,
       // Register Runtime API intrinsics (returning results and errors).
       RuntimeApiSymbolMap,
       // Register any additional user-defined symbol bindings
       std::move(custom_binding)});
}

//===----------------------------------------------------------------------===//
// Get executable arguments and results memory layouts.
//===----------------------------------------------------------------------===//

/*static*/ StatusOr<Executable::ArgumentsMemoryLayout>
Executable::GetArgumentsMemoryLayout(const FunctionType& signature) {
  // Requirements for passing function arguments.
  ArgumentsMemoryLayout layout;

  for (unsigned i = 0; i < signature.num_operands(); ++i) {
    const Type* type = signature.operand(i);

    // Check if the type defines the ABI for passing it as an argument.
    if (StatusOr<Type::ArgumentAbi> abi = type->AsArgument(); abi.ok()) {
      layout.num_args_ptrs += abi->num_ptrs;
      layout.num_ptrs.emplace_back(abi->num_ptrs);
      layout.offsets.emplace_back(
          i == 0 ? 0 : (layout.offsets[i - 1] + layout.num_ptrs[i - 1]));
      continue;
    }

    return InternalError("unknown operand #%i argument ABI: %s", i,
                         type->ToString());
  }

  return layout;
}

/*static*/ StatusOr<Executable::ResultsMemoryLayout>
Executable::GetResultsMemoryLayout(const FunctionType& signature) {
  // Requirements for returning function results.
  ResultsMemoryLayout layout;
  layout.offsets.reserve(signature.num_results());

  // TODO(ezhulenev): We should support allocating storage for results with non
  // standard alignment requirements.

  for (unsigned i = 0; i < signature.num_results(); ++i) {
    const Type* type = signature.result(i);

    // Keep track if the function has asynchronous results.
    layout.has_async_results |= llvm::isa<AsyncTokenType, AsyncValueType>(type);

    // Check if the type defines the ABI for returning it as a result.
    if (StatusOr<Type::ResultAbi> abi = type->AsResult(); abi.ok()) {
      layout.offsets.emplace_back(layout.size);
      // TODO(ezhulenev): Support user-defined result alignment. As a
      // workaround, we require all results to be a multiple of 8 bytes. This
      // way, we automatically get 8-byte alignment for all results, which is
      // enough for all currently supported types.
      size_t size = std::max<size_t>(abi->size, 8);
      assert(size % 8 == 0 && "size must be a multiple of 8 bytes");
      layout.size += size;
      continue;
    }

    return InternalError("unknown result #%i argument ABI: %s", i,
                         type->ToString());
  }

  return layout;
}

//===----------------------------------------------------------------------===//
// Executable CallFrame initialization.
//===----------------------------------------------------------------------===//

// Always verify executable arguments in debug mode.
static bool VerifyArguments(bool verify_arguments) {
#if defined(NDEBUG)
  return verify_arguments;
#endif
  return true;
}

Status Executable::InitializeCallFrame(unsigned ordinal, ArgumentsRef arguments,
                                       CallFrame* call_frame,
                                       bool verify_arguments) const {
  assert(ordinal < functions_.size() && "function ordinal out of bounds");
  const Function& fn = functions_[ordinal];

  // TODO(ezhulenev): If executable is specialized for concrete shapes then
  // there is no need to verify them once more here. However currently we rely
  // on a hash code to look up specializations, and this can lead to collisions.
  if (VerifyArguments(verify_arguments)) {
    // We verify run time arguments against the run time signature.
    const FunctionType& signature = fn.runtime_signature;

    // Make sure that we call the executable with the correct number of
    // arguments. We subtract one argument from the signature because it
    // corresponds to the context that we prepend to the given arguments.
    if (LLVM_UNLIKELY(arguments.size() != signature.num_operands() - 1))
      return InvalidArgument(
          "number of arguments doesn't match the function signature: %i vs %i",
          arguments.size(), signature.num_operands() - 1);

    // Verify that all arguments passed at runtime are compatible with compiled
    // function signature.
    auto kctx = dyn_cast<ExecutionContextOperandType>(signature.operand(0));
    if (LLVM_UNLIKELY(!kctx)) {
      return InvalidArgument(
          "expected ExecutionContext in first argument of signature, got: %s",
          signature.operand(0)->ToString());
    }

    // We use 0-based index for arguments, because the execution context
    // argument is an internal implementation detail, and in case of an error
    // users should get back argument index corresponding to the user provided
    // signature.
    for (unsigned i = 0; i < arguments.size(); ++i) {
      unsigned idx = i + 1;  // use 1-based index to fetch signature operand
      if (auto st = arguments[i].Verify(*signature.operand(idx)); !st.ok())
        return InvalidArgument("argument #%i doesn't match the signature: %s",
                               i, st.message());
    }
  }

  size_t num_args_ptrs = fn.arguments_memory_layout.num_args_ptrs;
  call_frame->args.resize_for_overwrite(num_args_ptrs);

  // Add a placeholder for the execution context as the first argument.
  call_frame->args[0] = nullptr;

  // Mutable view into the call frame arguments.
  absl::Span<void*> args(call_frame->args);

  // Pack all arguments according to the ABI to the call frame arguments.
  //
  // We use layout information starting with offset 1 because the execution
  // context argument packed above, and is not passed as a regular argument.
  for (unsigned i = 0; i < arguments.size(); ++i) {
    size_t offset = fn.arguments_memory_layout.offsets[i + 1];
    size_t len = fn.arguments_memory_layout.num_ptrs[i + 1];
    assert(offset >= 1 && (offset + len) <= num_args_ptrs);
    arguments[i].Pack(args.subspan(offset, len));
  }

  // Allocate storage for results.
  call_frame->results.resize_for_overwrite(fn.results_memory_layout.size);

  // Mark results memory initialized to supress potential msan errors.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(call_frame->results.data(),
                                      call_frame->results.size());

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Execute the compiled XLA runtime executable.
//===----------------------------------------------------------------------===//

absl::StatusOr<ExecutionReference> Executable::Execute(
    unsigned ordinal, ArgumentsRef arguments, const ResultConverter& results,
    const ExecuteOpts& opts, bool verify_arguments) const {
  // CallFrame can be allocated on the stack because compiled function will
  // unpack all the arguments it needs, and async regions will not access
  // the data after the initial function will return the result.
  CallFrame call_frame;

  // Touch every byte of the memref arguments, to trigger memory sanitizer error
  // if some of the memrefs are already deallocated. Unfortunatelly sanitizers
  // do not work inside the JIT compiled code, and compiled executables still
  // can do out of bounds memory access, however this sanity check allows to
  // catch obvious errors earlier.
#if defined(MEMORY_SANITIZER)
  auto do_not_optimize = [&](const auto& value) -> void {
    asm volatile("" : : "r,m"(value) : "memory");
  };

  for (unsigned i = 0; i < arguments.size(); ++i) {
    auto* memref = dyn_cast<MemrefDesc>(&arguments[i]);
    if (!memref) continue;

    int64_t size_in_bytes = primitive_util::ByteWidth(memref->dtype());
    for (int64_t size : memref->sizes()) size_in_bytes *= size;

    uint8_t* data = static_cast<uint8_t*>(memref->data());
    for (int64_t i = 0; i < size_in_bytes; ++i) {
      uint8_t value = data[i];
      do_not_optimize(value);
    }
  }
#endif

  // Compiled function takes arguments and results as `void**` type erased
  // pointer. See mlir::ExecutionEngine `packFunctionArguments` for the details.
  if (auto st = InitializeCallFrame(ordinal, arguments, &call_frame,
                                    verify_arguments);
      !st.ok())
    return (results.ReturnError(st), st);

  auto exec_ref = Execute(ordinal, call_frame, opts);

  // Convert compiled function return values into results.
  if (auto st = ReturnResults(ordinal, results, &call_frame); !st.ok())
    return st;

  return {std::move(exec_ref)};
}

ExecutionReference Executable::Execute(unsigned ordinal, CallFrame& call_frame,
                                       const ExecuteOpts& opts) const {
  assert(ordinal < functions_.size() && "function ordinal out of bounds");
  const Function& fn = functions_[ordinal];

  // Set the AsyncRuntime to be used by all async tasks spawned by the
  // executable.
  AsyncRuntime::Set(AsyncRuntime(opts.async_task_runner));

  ExecutionReference exec_ref;
  ExecutionContext* execution_ctx_ptr = nullptr;
  // For sync executable, runtime execution context can be used only by the
  // compiled function and can be safely allocated on the stack.
  ExecutionContext execution_ctx = {
      &fn.results_memory_layout, &call_frame, opts.custom_call_data,
      opts.custom_call_registry, opts.diagnostic_engine};
  if (IsAsync()) {
    // With custom calls inside async functions the lifetime of the execution
    // context must be extended until all pending async tasks are completed.
    exec_ref = ExecutionReference(new ExecutionContext{
        &fn.results_memory_layout, &call_frame, opts.custom_call_data,
        opts.custom_call_registry, opts.diagnostic_engine});
    execution_ctx_ptr = exec_ref.get();
  } else {
    // Override the execution context argument.
    execution_ctx_ptr = &execution_ctx;
  }

  assert(call_frame.args.size() == fn.arguments_memory_layout.num_args_ptrs);
  assert(call_frame.args[0] == nullptr && "expected to see a placeholder");
  call_frame.args[0] = &execution_ctx_ptr;

  // Call the compiled function.
  (*fn.fptr)(call_frame.args.data());
  return exec_ref;
}

Status Executable::ReturnResults(unsigned ordinal,
                                 const ResultConverter& results,
                                 CallFrame* call_frame) const {
  // If execution failed, forward error to all results.
  if (call_frame->is_error) {
    auto err = InternalError("run time error: %s", call_frame->error);
    return (results.ReturnError(err), err);
  }

  assert(ordinal < functions_.size() && "function ordinal out of bounds");
  const Function& fn = functions_[ordinal];

  // Try to convert results using registered conversion functions.
  bool converted = true;

  for (unsigned i = 0; i < fn.runtime_signature.num_results(); ++i) {
    const Type* type = fn.signature.result(i);
    const Type* runtime_type = fn.runtime_signature.result(i);
    void* ret = &call_frame->results[fn.results_memory_layout.offsets[i]];
    bool res = succeeded(results.ReturnValue(i, type, runtime_type, ret));
    converted = converted && res;
  }

  if (LLVM_UNLIKELY(!converted))
    return InternalError("failed to convert all returned values");
  else
    return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Load AOT compiled executable from an object file.
//===----------------------------------------------------------------------===//

/*static*/ StatusOr<Executable> Executable::LoadFromObjFile(
    std::string_view name, std::unique_ptr<llvm::MemoryBuffer> obj_file,
    std::vector<LoadFunction> load_functions,
    ExecutionEngine::SymbolsBinding symbols_binding,
    std::string_view memory_region_name) {
  // Memory region name to mmap executable code.
  std::string mapper_name = llvm::formatv(
      "/xla_aot{0}{1}:@{2}", memory_region_name.empty() ? "" : ":",
      EscapeMemRegionName(memory_region_name), name);

  // Custom memory mapper to tag memory allocated for XLA executables.
  std::unique_ptr<XlaRuntimeMemoryMapper> memory_mapper =
      XlaRuntimeMemoryMapper::Create(std::move(mapper_name));

  // Construct options for the XLA execution engine.
  ExecutionEngine::AotOptions options;
  options.section_memory_mapper = memory_mapper.get();
  options.symbols_binding = RuntimeSymbolsBinding(std::move(symbols_binding));

  // Function that must be exported by the execution engine.
  std::vector<std::string_view> exported;
  llvm::transform(load_functions, std::back_inserter(exported),
                  [](LoadFunction& fn) -> std::string_view { return fn.name; });

  auto engine = ExecutionEngine::CreateFromObjFile(std::move(obj_file), options,
                                                   exported);

  // Prepare exported functions for the executable.
  std::vector<Executable::Function> functions;

  for (const auto& indexed : llvm::enumerate(load_functions)) {
    LoadFunction& fn = indexed.value();

    // Get the memory layout for passing function arguments.
    auto args_memory_layout = GetArgumentsMemoryLayout(fn.runtime_signature);
    if (!args_memory_layout.ok()) return args_memory_layout.status();

    // Get the memory layout for returning function results.
    auto results_memory_layout = GetResultsMemoryLayout(fn.runtime_signature);
    if (!results_memory_layout.ok()) return results_memory_layout.status();

    functions.push_back(Executable::Function(
        std::move(fn.name), (*engine)->exported(indexed.index()),
        std::move(fn.signature), std::move(fn.runtime_signature),
        std::move(*args_memory_layout), std::move(*results_memory_layout),
        true));
  }

  return Executable(name, std::move(memory_mapper), std::move(*engine),
                    std::move(functions),
                    /*specialization=*/std::nullopt,
                    /*time_to_compile*/ std::chrono::milliseconds(0));
}

//===----------------------------------------------------------------------===//

bool Executable::IsAsync(unsigned ordinal) const {
  assert(ordinal < functions_.size() && "function ordinal out of bounds");
  return functions_[ordinal].results_memory_layout.has_async_results;
}

unsigned Executable::num_results(unsigned ordinal) const {
  assert(ordinal < functions_.size() && "function ordinal out of bounds");
  return functions_[ordinal].runtime_signature.num_results();
}

const FunctionType& Executable::signature(unsigned ordinal) const {
  assert(ordinal < functions_.size() && "function ordinal out of bounds");
  return functions_[ordinal].signature;
}

const FunctionType& Executable::runtime_signature(unsigned ordinal) const {
  assert(ordinal < functions_.size() && "function ordinal out of bounds");
  return functions_[ordinal].runtime_signature;
}

std::chrono::milliseconds Executable::time_to_compile() const {
  return time_to_compile_;
}

std::unique_ptr<llvm::MemoryBuffer> Executable::obj_file() const {
  return engine_->obj_file();
}

const CustomCall::UserData* Executable::GetUserData(ExecutionContext* ctx) {
  return ctx->custom_call_data;
}

const DiagnosticEngine* Executable::GetDiagnosticEngine(ExecutionContext* ctx) {
  return ctx->diagnostic_engine;
}

LogicalResult Executable::Call(ExecutionContext* ctx, class CustomCall& call,
                               void** args, void** attrs, void** rets) {
  return call.call(args, attrs, rets, ctx->custom_call_data,
                   ctx->diagnostic_engine);
}

FunctionRef Executable::function_ref(unsigned ordinal) const {
  assert(ordinal < functions_.size() && "function ordinal out of bounds");
  return FunctionRef(this, ordinal);
}

//===----------------------------------------------------------------------===//
// Executable function reference.
//===----------------------------------------------------------------------===//

FunctionRef::FunctionRef(const Executable* executable, unsigned ordinal)
    : executable_(executable), ordinal_(ordinal) {
  assert(executable && "executable must be not null");
}

absl::StatusOr<ExecutionReference> FunctionRef::operator()(
    ArgumentsRef arguments, const ResultConverter& results,
    const Executable::ExecuteOpts& opts, bool verify_arguments) const {
  return executable_->Execute(ordinal_, arguments, results, opts,
                              verify_arguments);
}

//===----------------------------------------------------------------------===//
// Register XLA runtime symbols with XLA execution engine.
//===----------------------------------------------------------------------===//

llvm::orc::SymbolMap RuntimeApiSymbolMap(llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;

  auto bind = [&](std::string_view name, auto symbol_ptr) {
    symbol_map[mangle(name)] = {llvm::orc::ExecutorAddr::fromPtr(symbol_ptr),
                                llvm::JITSymbolFlags()};
  };

  bind("runtimeGetResultStorage", &GetResultStorage);
  bind("runtimeSetError", &SetError);
  bind("runtimeCustomCall", &CustomCall);

  return symbol_map;
}

//----------------------------------------------------------------------------//
// Implement XLA Runtime <-> XLA Executable integration API.
//----------------------------------------------------------------------------//

void* GetResultStorage(ExecutionContext* ctx, int64_t index) {
  assert(ctx && "execution context must be not null");
  assert(!ctx->call_frame->is_error && "error must not be set");
  size_t offset = ctx->results_memory_layout->offsets[index];
  assert(offset < ctx->call_frame->results.size() && "offset is out of bounds");
  ctx->call_frame->has_set_outputs = true;
  return &ctx->call_frame->results[offset];
}

void SetError(ExecutionContext* ctx, const char* error) {
  assert(ctx && "execution context must be not null");
  assert(error && "runtime error must be not null");
  assert(!ctx->call_frame->is_error && "error must be set only once");
  assert(!ctx->call_frame->has_set_outputs && "outputs must be undefined");
  ctx->call_frame->is_error = true;
  ctx->call_frame->error = {error};
}

bool CustomCall(ExecutionContext* ctx, const char* target, void** args,
                void** attrs, void** rets) {
  assert(ctx && target && args && attrs && rets && "must be not null");
  assert(ctx->custom_call_registry && "custom call registry must be not null");

  const DiagnosticEngine* diagnostic = ctx->diagnostic_engine;

  if (ctx->custom_call_registry == nullptr) {
    if (diagnostic)
      diagnostic->EmitError(
          absl::InternalError("custom call registry is not available"));
    return false;
  }

  auto* custom_call = ctx->custom_call_registry->Find(target);
  if (custom_call == nullptr) {
    if (diagnostic)
      diagnostic->EmitError(absl::InternalError(absl::StrFormat(
          "custom call is not registered with runtime: %s", target)));
    return false;
  }

  return succeeded(custom_call->call(args, attrs, rets, ctx->custom_call_data,
                                     ctx->diagnostic_engine));
}

}  // namespace runtime
}  // namespace xla
