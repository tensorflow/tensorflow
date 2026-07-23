/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/compiled_function_library.h"

#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/debug_options_flags.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {
namespace {

using ScalarFn = void(float*);
using ComparatorFn = FunctionLibrary::Comparator;

static absl::StatusOr<llvm::orc::ThreadSafeModule> ParseModule(
    llvm::orc::ThreadSafeContext& context, absl::string_view ir,
    absl::string_view name) {
  llvm::SMDiagnostic diagnostic;
  auto m = context.withContextDo([&](llvm::LLVMContext* ctxt) {
    llvm::MemoryBufferRef ir_buffer(ir, name);
    return llvm::parseAssembly(ir_buffer, diagnostic, *ctxt);
  });
  if (m == nullptr) {
    return Internal("Failed to parse LLVM IR: %s",
                    diagnostic.getMessage().str());
  }

  SetModuleMemoryRegionName(*m, "compiled_function_library_test");

  return llvm::orc::ThreadSafeModule(std::move(m), context);
}

static absl::StatusOr<std::unique_ptr<FunctionLibrary>> BuildCompiledLibrary(
    absl::Span<const FunctionLibrary::Symbol> symbols) {
  auto context = std::make_unique<llvm::LLVMContext>();
  llvm::orc::ThreadSafeContext tsc(std::move(context));

  JitCompiler::Options options;
  options.num_dylibs = 1;

  std::unique_ptr<IrCompiler> ir_compiler = IrCompiler::Create(
      llvm::TargetOptions(),
      IrCompiler::Options{/*opt_level=*/llvm::CodeGenOptLevel::None,
                          /*optimize_for_size=*/false,
                          TargetMachineOptions(GetDebugOptionsFromFlags())},
      IrCompiler::CompilationHooks());

  ASSIGN_OR_RETURN(auto compiler, JitCompiler::Create(std::move(options),
                                                      std::move(ir_compiler)));

  constexpr absl::string_view ir = R"(
    define void @AddInplace(ptr %arg) {
      %v0 = load float, ptr %arg
      %v1 = fadd float %v0, %v0
      store float %v1, ptr %arg
      ret void
    }
  )";

  ASSIGN_OR_RETURN(llvm::orc::ThreadSafeModule tsm,
                   ParseModule(tsc, ir, "AddInplace"));
  RETURN_IF_ERROR(compiler.AddModule(std::move(tsm), 0));

  return std::move(compiler).Compile(symbols);
}

TEST(CompiledFunctionLibraryTest, ResolveFunctionSuccess) {
  std::vector<FunctionLibrary::Symbol> symbols = {
      FunctionLibrary::Sym<ScalarFn>("AddInplace")};

  TF_ASSERT_OK_AND_ASSIGN(auto lib, BuildCompiledLibrary(symbols));

  TF_ASSERT_OK_AND_ASSIGN(auto fn,
                          lib->ResolveFunction<ScalarFn>("AddInplace"));
  EXPECT_NE(fn, nullptr);

  float val = 2.0f;
  fn(&val);
  EXPECT_EQ(val, 4.0f);
}

TEST(CompiledFunctionLibraryTest, ResolveFunctionNotFound) {
  std::vector<FunctionLibrary::Symbol> symbols = {
      FunctionLibrary::Sym<ScalarFn>("AddInplace")};

  TF_ASSERT_OK_AND_ASSIGN(auto lib, BuildCompiledLibrary(symbols));

  auto status = lib->ResolveFunction<ScalarFn>("non_existent").status();
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(
      absl::StrContains(status.message(), "Function non_existent not found"));
}

TEST(CompiledFunctionLibraryTest, ResolveFunctionTypeIdMismatch) {
  std::vector<FunctionLibrary::Symbol> symbols = {
      FunctionLibrary::Sym<ScalarFn>("AddInplace")};

  TF_ASSERT_OK_AND_ASSIGN(auto lib, BuildCompiledLibrary(symbols));

  // Requesting ComparatorFn when symbol was registered as ScalarFn
  auto status = lib->ResolveFunction<ComparatorFn>("AddInplace").status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(status.message(), "has type id"));
}

TEST(CompiledFunctionLibraryTest, GetTypelessSymbolsMap) {
  std::vector<FunctionLibrary::Symbol> symbols = {
      FunctionLibrary::Sym<ScalarFn>("AddInplace")};

  TF_ASSERT_OK_AND_ASSIGN(auto lib, BuildCompiledLibrary(symbols));

  auto* compiled_lib = static_cast<CompiledFunctionLibrary*>(lib.get());
  auto typeless_map = compiled_lib->GetTypelessSymbolsMap();
  EXPECT_EQ(typeless_map.size(), 1);
  EXPECT_NE(typeless_map["AddInplace"], nullptr);
}

TEST(CompiledFunctionLibraryTest, ConcurrentMultiThreadedResolution) {
  std::vector<FunctionLibrary::Symbol> symbols = {
      FunctionLibrary::Sym<ScalarFn>("AddInplace")};

  TF_ASSERT_OK_AND_ASSIGN(auto lib_unique, BuildCompiledLibrary(symbols));
  std::shared_ptr<FunctionLibrary> lib = std::move(lib_unique);

  constexpr int kNumThreads = 10;
  constexpr int kIterations = 1000;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([lib]() {
      for (int i = 0; i < kIterations; ++i) {
        auto res = lib->ResolveFunction<ScalarFn>("AddInplace");
        EXPECT_TRUE(res.ok());
        if (res.ok()) {
          EXPECT_NE(*res, nullptr);
          float val = 1.0f;
          (*res)(&val);
          EXPECT_EQ(val, 2.0f);
        }
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
}

}  // namespace
}  // namespace xla::cpu
