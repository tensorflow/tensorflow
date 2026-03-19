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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/debug_options_flags.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"

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

  SetModuleMemoryRegionName(*m, "jit_compiler_test");

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

  std::unique_ptr<IrCompiler> ir_compiler = IrCompiler::Create(
      llvm::TargetOptions(),
      IrCompiler::Options{/*opt_level=*/llvm::CodeGenOptLevel::None,
                          /*optimize_for_size=*/false,
                          TargetMachineOptions(GetDebugOptionsFromFlags())},
      IrCompiler::CompilationHooks());

  TF_ASSERT_OK_AND_ASSIGN(
      auto compiler,
      JitCompiler::Create(std::move(options), std::move(ir_compiler),
                          std::move(task_runner)));

  constexpr absl::string_view add_in_place_ir = R"(
    define void @AddInplace(ptr %arg) {
      %v0 = load float, ptr %arg
      %v1 = fadd float %v0, %v0
      store float %v1, ptr %arg
      ret void
    })";

  constexpr absl::string_view mul_in_place_ir = R"(
    define void @MulInplace(ptr %arg) {
      %v0 = load float, ptr %arg
      %v1 = fmul float %v0, %v0
      store float %v1, ptr %arg
      ret void
    })";

  auto add_module = [&](absl::string_view ir, absl::string_view name,
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
      if ((*name).contains("external_fn")) {
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
  options.definition_generator = [](const llvm::DataLayout& data_layout) {
    return std::make_unique<ExternalDefinitionGenerator>();
  };

  std::unique_ptr<IrCompiler> ir_compiler = IrCompiler::Create(
      llvm::TargetOptions(),
      IrCompiler::Options{/*opt_level=*/llvm::CodeGenOptLevel::None,
                          /*optimize_for_size=*/false,
                          TargetMachineOptions(GetDebugOptionsFromFlags())},
      IrCompiler::CompilationHooks());

  TF_ASSERT_OK_AND_ASSIGN(
      auto compiler,
      JitCompiler::Create(std::move(options), std::move(ir_compiler),
                          /*task_runner=*/nullptr));

  constexpr absl::string_view call_external_fn_ir = R"(
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

class JitCompilerTestFixture : public ::testing::Test {
 protected:
  // Access to private inner class types.
  using TaskDispatcher = JitCompiler::TaskDispatcher;
  using Task = JitCompiler::Task;
};

TEST_F(JitCompilerTestFixture, TaskMemoryReleasedWhenTaskRetainedByQueue) {
  // A minimal task that flags when its resource is freed.
  class TrackedTask : public llvm::orc::Task {
   public:
    explicit TrackedTask(bool* free_resource_flag, int* tasks_run)
        : free_resource_flag_(free_resource_flag), tasks_run_(tasks_run) {}
    void printDescription(llvm::raw_ostream&) override {}
    void run() override { (*tasks_run_)++; }
    ~TrackedTask() override { *free_resource_flag_ = true; }

   private:
    bool* free_resource_flag_ = nullptr;
    int* tasks_run_ = nullptr;
  };

  // Simulate the thread pool's internal storage.
  std::vector<JitCompiler::Task> thread_pool_storage;
  // The worker runs a queued task by copying it to its local stack, but leaving
  // the original in the queue as an optimization.
  JitCompiler::TaskRunner worker_thread =
      [&thread_pool_storage](JitCompiler::Task task) {
        thread_pool_storage.push_back(std::move(task));  // Enqueue

        // Copy to local stack (simulating thread-local execution)
        JitCompiler::Task local_task = thread_pool_storage.back();
        local_task();
      };

  auto dispatcher = std::make_unique<TaskDispatcher>(std::move(worker_thread));

  bool task_resource_freed = false;
  int tasks_run = 0;
  dispatcher->dispatch(
      std::make_unique<TrackedTask>(&task_resource_freed, &tasks_run));

  ASSERT_EQ(tasks_run, 1);
  // The task has run, but the thread pool still holds a reference to the task
  // wrapper.
  EXPECT_FALSE(thread_pool_storage.empty());
  // The task's resources must have been freed, otherwise a use-after free can
  // occur when the task wrapper is eventually destroyed from the thread pool.
  EXPECT_TRUE(task_resource_freed)
      << "The Queue is keeping the Task resource alive!";
}

}  // namespace xla::cpu
