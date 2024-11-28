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
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/function_library.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla::cpu {

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
      JitCompiler::Create(llvm::TargetOptions(), llvm::CodeGenOptLevel::None,
                          std::move(options), std::move(task_runner)));

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
    llvm::SMDiagnostic diagnostic;
    llvm::MemoryBufferRef ir_buffer(ir, name);

    auto m = llvm::parseAssembly(ir_buffer, diagnostic, *tsc.getContext());
    if (m == nullptr) {
      return Internal("Failed to parse LLVM IR: %s",
                      diagnostic.getMessage().str());
    }

    llvm::orc::ThreadSafeModule tsm(std::move(m), tsc);
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
                          std::move(compiler).Compile(symbols));

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

}  // namespace xla::cpu
