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

#include "xla/backends/cpu/codegen/jit_compiler.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla::cpu {

// We use static function to compile the function library, because we transfer
// compiler object into the function and make sure that it gets destroyed before
// returning the function library to the caller, as we test that we don't
// accidentally reference freed objects owned by the compiler.
static absl::StatusOr<std::unique_ptr<FunctionLibrary>> Compile(
    JitCompiler compiler, absl::Span<const FunctionLibrary::Symbol> symbols) {
  return std::move(compiler).Compile(symbols);
};

// Parses the LLVM IR into a ThreadSafeModule.
static absl::StatusOr<llvm::orc::ThreadSafeModule> ParseModule(
    llvm::orc::ThreadSafeContext& context, std::string_view ir,
    std::string_view name) {
  llvm::SMDiagnostic diagnostic;
  llvm::MemoryBufferRef ir_buffer(ir, name);

  auto m = llvm::parseAssembly(ir_buffer, diagnostic, *context.getContext());
  if (m == nullptr) {
    return Internal("Failed to parse LLVM IR: %s",
                    diagnostic.getMessage().str());
  }

  return llvm::orc::ThreadSafeModule(std::move(m), context);
}

TEST(JitCompilerTest, Compile) {
  auto context = std::make_unique<llvm::LLVMContext>();
  llvm::orc::ThreadSafeContext tsc(std::move(context));

  JitCompiler::Options options;
  options.num_dylibs = 2;

  // Use thread pool to run compilation tasks in parallel.
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 2);
  std::atomic<int32_t> num_tasks = 0;
  JitCompiler::TaskRunner task_runner = [&](JitCompiler::Task task) {
    num_tasks++;
    thread_pool.Schedule(std::move(task));
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto compiler,
      JitCompiler::Create(llvm::TargetOptions(), std::move(options),
                          std::move(task_runner)));

  constexpr std::string_view add_in_place_ir = R"(
    define void @AddInplace(ptr %arg) {
      %v0 = load float, ptr %arg
      %v1 = fadd float %v0, %v0
      store float %v1, ptr %arg
      ret void
    })";

  constexpr std::string_view mul_in_place_ir = R"(
    define void @MulInplace(ptr %arg) {
      %v0 = load float, ptr %arg
      %v1 = fmul float %v0, %v0
      store float %v1, ptr %arg
      ret void
    })";

  auto add_module = [&](std::string_view ir, std::string_view name,
                        size_t dylib_index) -> absl::Status {
    TF_ASSIGN_OR_RETURN(llvm::orc::ThreadSafeModule tsm,
                        ParseModule(tsc, ir, name));
    TF_RETURN_IF_ERROR(compiler.AddModule(std::move(tsm), dylib_index));
    return absl::OkStatus();
  };

  TF_ASSERT_OK(add_module(add_in_place_ir, "AddInplace", 0));
  TF_ASSERT_OK(add_module(mul_in_place_ir, "MulInplace", 1));

  using ScalarFn = void(float*);
  std::vector<FunctionLibrary::Symbol> symbols = {
      FunctionLibrary::Sym<ScalarFn>("AddInplace"),
      FunctionLibrary::Sym<ScalarFn>("MulInplace")};

  TF_ASSERT_OK_AND_ASSIGN(auto function_library,
                          Compile(std::move(compiler), symbols));

  EXPECT_GE(num_tasks, 2);

  TF_ASSERT_OK_AND_ASSIGN(
      ScalarFn * add_in_place,
      function_library->ResolveFunction<ScalarFn>("AddInplace"));

  TF_ASSERT_OK_AND_ASSIGN(
      ScalarFn * mul_in_place,
      function_library->ResolveFunction<ScalarFn>("MulInplace"));

  EXPECT_NE(add_in_place, nullptr);
  EXPECT_NE(mul_in_place, nullptr);

  float value = 1.0f;
  add_in_place(&value);
  EXPECT_EQ(value, 2.0f);

  mul_in_place(&value);
  EXPECT_EQ(value, 4.0f);
}

class ExternalDefinitionGenerator : public llvm::orc::DefinitionGenerator {
 public:
  static void AddInplace(float* value) { *value += *value; }

  llvm::Error tryToGenerate(llvm::orc::LookupState&, llvm::orc::LookupKind,
                            llvm::orc::JITDylib& jit_dylib,
                            llvm::orc::JITDylibLookupFlags,
                            const llvm::orc::SymbolLookupSet& names) final {
    llvm::orc::SymbolMap new_defs;
    for (auto& [name, flags] : names) {
      if (*name == "__external_fn") {
        new_defs[name] = llvm::orc::ExecutorSymbolDef{
            llvm::orc::ExecutorAddr(reinterpret_cast<uint64_t>(&AddInplace)),
            llvm::JITSymbolFlags::None};
      }
    }

    cantFail(jit_dylib.define(llvm::orc::absoluteSymbols(std::move(new_defs))));
    return llvm::Error::success();
  }
};

TEST(JitCompilerTest, ExternalDefinitionGenerator) {
  auto context = std::make_unique<llvm::LLVMContext>();
  llvm::orc::ThreadSafeContext tsc(std::move(context));

  JitCompiler::Options options;
  options.definition_generator = [](llvm::TargetMachine*) {
    return std::make_unique<ExternalDefinitionGenerator>();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto compiler,
      JitCompiler::Create(llvm::TargetOptions(), std::move(options),
                          /*task_runner=*/nullptr));

  constexpr std::string_view call_external_fn_ir = R"(
    declare void @__external_fn(ptr %arg)

    define void @CallExternalFn(ptr %arg) {
      call void @__external_fn(ptr %arg)
      ret void
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      llvm::orc::ThreadSafeModule tsm,
      ParseModule(tsc, call_external_fn_ir, "CallExternalFn"));

  TF_ASSERT_OK(compiler.AddModule(std::move(tsm)));

  using ScalarFn = void(float*);
  std::vector<FunctionLibrary::Symbol> symbols = {
      FunctionLibrary::Sym<ScalarFn>("CallExternalFn")};

  TF_ASSERT_OK_AND_ASSIGN(auto function_library,
                          Compile(std::move(compiler), symbols));

  TF_ASSERT_OK_AND_ASSIGN(
      ScalarFn * call_external_fn,
      function_library->ResolveFunction<ScalarFn>("CallExternalFn"));

  float value = 1.0f;
  call_external_fn(&value);
  EXPECT_EQ(value, 2.0f);
}

}  // namespace xla::cpu
