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

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/ErrorOr.h"
#include "tensorflow/compiler/xla/mlir/utils/runtime/async_runtime_api.h"
#include "tensorflow/compiler/xla/mlir/utils/runtime/c_runner_utils.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/errors.h"
#include "tensorflow/compiler/xla/runtime/runtime.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"

namespace xla {
namespace runtime {

using absl::StatusOr;

using llvm::dyn_cast;

using llvm::Error;
using llvm::Expected;

using llvm::orc::MangleAndInterner;
using llvm::orc::SymbolMap;

// KernelContext encapsulates all the data that is required to implement XLA
// Runtime <-> XLA Executable integration API.
struct KernelContext {
  // Results memory layout is owned by the executable, and stays alive after
  // the entrypoint function execution completes.
  const Executable::ResultsMemoryLayout* results_memory_layout = nullptr;

  // CallFrame life time bound to the entrypoint function execution and
  // destroyed immediately when the function returns. Only the entrypoint
  // function itself reads the arguments and writes to the function results
  // storage.
  Executable::CallFrame* call_frame = nullptr;

  // User-defined data for custom call handlers.
  CustomCall::UserData* custom_call_data = nullptr;

  // User-defined diagnostic engine for reporting diagnostics.
  DiagnosticEngine* diagnostic_engine = nullptr;
};

//===----------------------------------------------------------------------===//
// Conversion from custom calls library and type id registry to symbols binding.
//===----------------------------------------------------------------------===//

ExecutionEngine::SymbolsBinding ToSymbolsBinding(
    DirectCustomCallLibrary lib, TypeIDNameRegistry::RegistrationFn types) {
  return [=](MangleAndInterner mangle) {
    SymbolMap symbol_map;

    // Always register canonical custom call types with the registry.
    TypeIDNameRegistry registry;
    PopulateCustomCallTypeIdNames(registry);
    if (types) types(registry);

    // Register direct custom calls.
    using DirectCustomCall = DirectCustomCallLibrary::DirectCustomCall;
    lib.ForEach([&](std::string_view name, DirectCustomCall custom_call) {
      symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
          llvm::pointerToJITTargetAddress(custom_call), llvm::JITSymbolFlags());
    });

    // Register type id symbols.
    registry.ForEach([&](std::string_view name, TypeID type_id) {
      auto type_id_ptr =
          reinterpret_cast<std::uintptr_t>(type_id.getAsOpaquePointer());
      symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
          static_cast<llvm::JITTargetAddress>(type_id_ptr),
          llvm::JITSymbolFlags());
    });

    return symbol_map;
  };
}

//===----------------------------------------------------------------------===//
// Register XLA runtime symbols with XLA execution engine.
//===----------------------------------------------------------------------===//

static SymbolMap RuntimeApiSymbolMap(MangleAndInterner);

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

/*static*/ Expected<Executable::ArgumentsMemoryLayout>
Executable::GetArgumentsMemoryLayout(const FunctionType& signature) {
  // Requirements for passing function arguments.
  ArgumentsMemoryLayout layout;

  for (unsigned i = 0; i < signature.num_operands(); ++i) {
    const Type* type = signature.operand(i);

    // Check if the type defines the ABI for passing it as an argument.
    if (StatusOr<Type::ArgumentAbi> abi = type->AsArgument(); abi.ok()) {
      layout.num_args_ptrs += abi->num_ptrs;
      continue;
    }

    return MakeStringError("unknown operand #", i, " argument ABI: ", *type);
  }

  return layout;
}

/*static*/ Expected<Executable::ResultsMemoryLayout>
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
      layout.size += abi->size;
      continue;
    }

    return MakeStringError("unknown result #", i, " type result ABI: ", *type);
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

Error Executable::InitializeCallFrame(ArgumentsRef arguments,
                                      CallFrame* call_frame,
                                      bool verify_arguments) const {
  // TODO(ezhulenev): If executable is specialized for concrete shapes then
  // there is no need to verify them once more here. However currently we rely
  // on a hash code to look up specializations, and this can lead to collisions.
  if (VerifyArguments(verify_arguments)) {
    // We verify run time arguments against the run time signature.
    const FunctionType& signature = runtime_signature_;

    // Make sure that we call the executable with the correct number of
    // arguments. We subtract one argument from the signature because it
    // corresponds to the context that we prepend to the given arguments.
    if (LLVM_UNLIKELY(arguments.size() != signature.num_operands() - 1))
      return MakeStringError(
          "number of arguments doesn't match the function signature: ",
          arguments.size(), " vs ", signature.num_operands() - 1);

    // Verify that all arguments passed at runtime are compatible with compiled
    // function signature.
    auto kctx = dyn_cast<KernelContextOperandType>(signature.operand(0));
    if (LLVM_UNLIKELY(!kctx)) {
      return MakeStringError(
          "expected KernelContext in first argument of signature, got: ",
          signature.operand(0));
    }

    // We use 0-based index for arguments, because the kernel context argument
    // is an internal implementation detail, and in case of an error users
    // should get back argument index corresponding to the user provided
    // signature.
    for (unsigned i = 0; i < arguments.size(); ++i) {
      unsigned idx = i + 1;  // use 1-based index to fetch signature operand
      if (auto st = arguments[i].Verify(*signature.operand(idx)); !st.ok())
        return MakeStringError("argument #", i,
                               " doesn't match the signature: ", st.message());
    }
  }

  size_t num_args_ptrs = arguments_memory_layout_.num_args_ptrs;
  call_frame->args.resize_for_overwrite(num_args_ptrs);

  // Add a placeholder for the kernel context as the first argument.
  call_frame->args[0] = nullptr;

  // Keep offset of the next argument in the `args` array, and update it every
  // time we pack a new argument.
  size_t offset = 1;

  // Mutable view into the call frame arguments.
  auto args = absl::Span<void*>(call_frame->args.data(), num_args_ptrs);

  // Pack all arguments according to the ABI to the call frame arguments.
  for (unsigned i = 0; i < arguments.size(); ++i)
    offset = arguments[i].Pack(args, offset);

  assert(offset == num_args_ptrs &&
         "reserved number of args must match the argument offset");

  // Allocate storage for results.
  call_frame->results.resize_for_overwrite(results_memory_layout_.size);

  // Mark results memory initialized to supress potential msan errors.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(call_frame->results.data(),
                                      call_frame->results.size());

  return Error::success();
}

//===----------------------------------------------------------------------===//
// Execute the compiled XLA runtime executable.
//===----------------------------------------------------------------------===//

Error Executable::Execute(ArgumentsRef arguments,
                          const ResultConverter& results,
                          const ExecuteOpts& opts,
                          bool verify_arguments) const {
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
  if (auto err = InitializeCallFrame(arguments, &call_frame, verify_arguments))
    return (results.ReturnError(err), std::move(err));

  Execute(call_frame, opts);

  // Convert compiled function return values into results.
  if (auto err = ReturnResults(results, &call_frame)) return err;

  return Error::success();
}

void Executable::Execute(CallFrame& call_frame, const ExecuteOpts& opts) const {
  // Set the AsyncRuntime to be used by all async tasks spawned by the
  // executable.
  AsyncRuntime::Set(AsyncRuntime(opts.async_task_runner));

  // Runtime kernel context can be used only by the entrypoint function and can
  // be safely allocated on the stack.
  KernelContext kernel_context = {&results_memory_layout_, &call_frame,
                                  opts.custom_call_data,
                                  opts.diagnostic_engine};

  // Override the kernel context argument.
  KernelContext* kernel_context_ptr = &kernel_context;
  assert(call_frame.args.size() == arguments_memory_layout_.num_args_ptrs);
  assert(call_frame.args[0] == nullptr && "expected to see a placeholder");
  call_frame.args[0] = &kernel_context_ptr;

  // Call the compiled function.
  (*fptr_)(call_frame.args.data());
}

Error Executable::ReturnResults(const ResultConverter& results,
                                CallFrame* call_frame) const {
  // If execution failed, forward error to all results.
  if (call_frame->is_error) {
    auto err = MakeStringError("run time error: ", call_frame->error);
    return (results.ReturnError(err), std::move(err));
  }

  // Try to convert results using registered conversion functions.
  bool converted = true;

  for (unsigned i = 0; i < runtime_signature_.num_results(); ++i) {
    const Type* type = signature_.result(i);
    const Type* runtime_type = runtime_signature_.result(i);
    void* ret = &call_frame->results[results_memory_layout_.offsets[i]];
    bool res = mlir::succeeded(results.ReturnValue(i, type, runtime_type, ret));
    converted = converted && res;
  }

  if (LLVM_UNLIKELY(!converted))
    return MakeStringError("failed to convert all returned values");
  else
    return Error::success();
}

//===----------------------------------------------------------------------===//
// Load AOT compiled executable from an object file.
//===----------------------------------------------------------------------===//

/*static*/ Expected<Executable> Executable::LoadFromObjFile(
    std::string_view name, std::unique_ptr<llvm::MemoryBuffer> obj_file,
    std::string_view entrypoint, FunctionType signature,
    FunctionType runtime_signature,
    ExecutionEngine::SymbolsBinding symbols_binding,
    std::string_view memory_region_name) {
  // Memory region name to mmap executable code.
  std::string mapper_name = llvm::formatv(
      "/xla_aot{0}{1}:@{2}::@{3}", memory_region_name.empty() ? "" : ":",
      EscapeMemRegionName(memory_region_name), name, entrypoint);

  // Custom memory mapper to tag memory allocated for XLA executables.
  std::unique_ptr<XlaRuntimeMemoryMapper> memory_mapper =
      XlaRuntimeMemoryMapper::Create(std::move(mapper_name));

  // Construct options for the XLA execution engine.
  ExecutionEngine::AotOptions options;
  options.section_memory_mapper = memory_mapper.get();
  options.symbols_binding = RuntimeSymbolsBinding(std::move(symbols_binding));

  auto engine = ExecutionEngine::CreateFromObjFile(std::move(obj_file),
                                                   entrypoint, options);

  // Get the memory layout for passing function arguments.
  auto arguments_memory_layout = GetArgumentsMemoryLayout(runtime_signature);
  if (auto err = arguments_memory_layout.takeError()) return std::move(err);

  // Get the memory layout for returning function results.
  auto results_memory_layout = GetResultsMemoryLayout(runtime_signature);
  if (auto err = results_memory_layout.takeError()) return std::move(err);

  return Executable(name, std::move(memory_mapper), std::move(*engine),
                    std::move(signature), std::move(runtime_signature),
                    std::move(*arguments_memory_layout),
                    std::move(*results_memory_layout),
                    /*specialization=*/std::nullopt,
                    /*time_to_compile*/ std::chrono::milliseconds(0));
}

//===----------------------------------------------------------------------===//

unsigned Executable::num_results() const {
  return runtime_signature_.num_results();
}

const FunctionType& Executable::signature() const { return signature_; }

const FunctionType& Executable::runtime_signature() const {
  return runtime_signature_;
}

std::chrono::milliseconds Executable::time_to_compile() const {
  return time_to_compile_;
}

std::unique_ptr<llvm::MemoryBuffer> Executable::obj_file() const {
  return engine_->obj_file();
}

CustomCall::UserData* Executable::GetUserData(KernelContext* ctx) {
  return ctx->custom_call_data;
}

DiagnosticEngine* Executable::GetDiagnosticEngine(KernelContext* ctx) {
  return ctx->diagnostic_engine;
}

mlir::LogicalResult Executable::Call(KernelContext* ctx, class CustomCall& call,
                                     void** args, void** attrs) {
  return call.call(args, attrs, ctx->custom_call_data, ctx->diagnostic_engine);
}

//===----------------------------------------------------------------------===//
// Register XLA runtime symbols with XLA execution engine.
//===----------------------------------------------------------------------===//

SymbolMap RuntimeApiSymbolMap(MangleAndInterner mangle) {
  SymbolMap symbol_map;

  auto bind = [&](std::string_view name, auto symbol_ptr) {
    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  bind("runtimeGetResultStorage", &GetResultStorage);
  bind("runtimeSetError", &SetError);
  bind("runtimeCustomCall", &CustomCall);

  return symbol_map;
}

//----------------------------------------------------------------------------//
// Implement XLA Runtime <-> XLA Executable integration API.
//----------------------------------------------------------------------------//

void* GetResultStorage(KernelContext* ctx, int64_t index) {
  assert(ctx && "kernel context must be not null");
  assert(!ctx->call_frame->is_error && "error must not be set");
  size_t offset = ctx->results_memory_layout->offsets[index];
  assert(offset < ctx->call_frame->results.size() && "offset is out of bounds");
  ctx->call_frame->has_set_outputs = true;
  return &ctx->call_frame->results[offset];
}

void SetError(KernelContext* ctx, const char* error) {
  assert(ctx && "kernel context must be not null");
  assert(error && "runtime error must be not null");
  assert(!ctx->call_frame->is_error && "error must be set only once");
  assert(!ctx->call_frame->has_set_outputs && "outputs must be undefined");
  ctx->call_frame->is_error = true;
  ctx->call_frame->error = {error};
}

bool CustomCall(KernelContext* ctx, const char* target, void** args,
                void** attrs) {
  assert(ctx && target && args && attrs && "all arguments must be not null");

  // Default custom calls registry for the XLA executables.
  static CustomCallRegistry* registry = []() {
    auto* registry = new CustomCallRegistry();
    RegisterStaticCustomCalls(registry);
    return registry;
  }();

  auto* custom_call = registry->Find(target);
  assert(custom_call && "custom call not found");
  if (custom_call == nullptr) return false;

  return succeeded(custom_call->call(args, attrs, ctx->custom_call_data,
                                     ctx->diagnostic_engine));
}

}  // namespace runtime
}  // namespace xla
