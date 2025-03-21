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

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "xla/backends/cpu/codegen/execution_engine.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/object_loader.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::cpu {

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;

absl::StatusOr<JitCompiler> JitCompiler::Create(
    Options options, std::unique_ptr<IrCompiler> ir_compiler,
    TaskRunner task_runner) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::TargetMachine> target_machine,
                      ir_compiler->build_target_machine());

  // Dispatch compilation tasks using the provided task runner.
  auto task_dispatcher =
      std::make_unique<TaskDispatcher>(std::move(task_runner));
  TaskDispatcher* task_dispatcher_ptr = task_dispatcher.get();

  // LLVM execution session that holds jit-compiled functions.
  auto execution_session = std::make_unique<llvm::orc::ExecutionSession>(
      std::make_unique<llvm::orc::UnsupportedExecutorProcessControl>(
          /*SSP=*/nullptr, std::move(task_dispatcher)));

  execution_session->setErrorReporter([](llvm::Error err) {
    LOG(ERROR) << "LLVM compilation error: " << llvm::toString(std::move(err));
  });

  return JitCompiler(std::move(target_machine), task_dispatcher_ptr,
                     std::move(execution_session), std::move(ir_compiler),
                     options.num_dylibs,
                     std::move(options.definition_generator));
}

static std::unique_ptr<llvm::orc::IRCompileLayer> CreateCompileLayer(
    llvm::orc::ExecutionSession& execution_session,
    llvm::orc::RTDyldObjectLinkingLayer& object_layer,
    std::unique_ptr<IrCompiler> ir_compiler) {
  return std::make_unique<llvm::orc::IRCompileLayer>(
      execution_session, object_layer, std::move(ir_compiler));
}

JitCompiler::JitCompiler(
    std::unique_ptr<llvm::TargetMachine> target_machine,
    TaskDispatcher* task_dispatcher,
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
    std::unique_ptr<IrCompiler> ir_compiler, size_t num_dylibs,
    ExecutionEngine::DefinitionGenerator definition_generator)
    : target_machine_(std::move(target_machine)),
      task_dispatcher_(task_dispatcher),
      execution_engine_(std::make_unique<ExecutionEngine>(
          std::move(execution_session), target_machine_->createDataLayout(),
          definition_generator)),
      compile_layer_(CreateCompileLayer(*execution_engine_->execution_session(),
                                        *execution_engine_->object_layer(),
                                        std::move(ir_compiler))) {
  execution_engine_->AllocateDylibs(num_dylibs);
  execution_engine_->RegisterJITEventListeners();

  if (target_machine_->getTargetTriple().isOSBinFormatCOFF()) {
    execution_engine_->SetObjectLayerFlags();
  }
}

JitCompiler::~JitCompiler() = default;

static void AddDylibIndexModuleFlag(llvm::Module& llvm_module,
                                    size_t dylib_index) {
  auto i64ty = llvm::Type::getInt64Ty(llvm_module.getContext());
  llvm_module.addModuleFlag(llvm::Module::Error, "xla_dylib_index",
                            llvm::ConstantInt::get(i64ty, dylib_index));
}

absl::Status JitCompiler::AddModule(llvm::orc::ThreadSafeModule module,
                                    size_t dylib_index) {
  // Set up module for codegen for the target machine at hand.
  module.withModuleDo([&](llvm::Module& m) {
    m.setDataLayout(target_machine_->createDataLayout());
    m.setTargetTriple(target_machine_->getTargetTriple());
    AddDylibIndexModuleFlag(m, dylib_index);
  });

  // Add module to the selected dynamic library.
  TF_ASSIGN_OR_RETURN(llvm::orc::JITDylib * dylib,
                      execution_engine_->dylib(dylib_index));
  if (auto err = compile_layer_->add(*dylib, std::move(module))) {
    return Internal("Failed to add module to dylib %d: %s", dylib_index,
                    llvm::toString(std::move(err)));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> JitCompiler::Compile(
    absl::Span<const Symbol> symbols) && {
  TraceMe trace([&] {
    return TraceMeEncode("JitCompiler::Compile",
                         {{"num_symbols", symbols.size()}});
  });
  ObjectLoader object_loader(std::move(execution_engine_));
  llvm::DataLayout data_layout = target_machine_->createDataLayout();
  TF_ASSIGN_OR_RETURN(auto symbol_map, object_loader.LookupSymbols(symbols));
  // Wait for all compilation tasks to finish.
  task_dispatcher_->shutdown();
  return std::move(object_loader)
      .CreateFunctionLibrary(std::move(symbols), symbol_map);
}

JitCompiler::TaskDispatcher::TaskDispatcher(TaskRunner task_runner)
    : task_runner_(std::move(task_runner)) {}

JitCompiler::TaskDispatcher::~TaskDispatcher() { shutdown(); }

void JitCompiler::TaskDispatcher::dispatch(
    std::unique_ptr<llvm::orc::Task> task) {
  // Dispatch task in the current thread if no task runner is provided.
  if (task_runner_ == nullptr) {
    task->run();
    return;
  }

  // Dispatch task using user-provided task runner. We release the lock before
  // dispatching the task to avoid deadlock, because `task_runner_` may choose
  // to execute the task in the current thread.
  {
    absl::MutexLock lock(&mu_);
    ++num_dispatched_tasks_;
  }

  task_runner_([this, task = std::shared_ptr<llvm::orc::Task>(
                          std::move(task))]() mutable {
    TraceMe trace("TaskDispatcher::dispatch");

    // We run and explicitly destroy the task before decrementing the counter
    // and notifying the condition variable, to ensure that the task is fully
    // executed and cleaned up before task dispatcher shut down.
    task->run();
    task.reset();

    absl::MutexLock lock(&mu_);
    --num_dispatched_tasks_;
  });
}

void JitCompiler::TaskDispatcher::shutdown() {
  auto all_tasks_finished = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return num_dispatched_tasks_ == 0;
  };
  absl::MutexLock lock(&mu_, absl::Condition(&all_tasks_finished));
}

}  // namespace xla::cpu
