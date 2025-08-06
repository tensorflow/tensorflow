/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/mlrt/interpreter/execute.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/kernel.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/register_span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"
#include "tsl/profiler/lib/traceme.h"

namespace mlrt {
namespace {

struct CurrentExecutionInfo {
  // The ExecutionContext for the current thread.
  ExecutionContext* current_context = nullptr;
  // The next ExecutionContext that is ready for execution if `current_context`
  // is exiting.
  ExecutionContext* ready_context = nullptr;
};

// Get the ExecutionContext used in the current thread. This provides a
// side-channel for code which cannot access ExecutionContext explicitly.
CurrentExecutionInfo& GetCurrentExecutionInfo() {
  static thread_local CurrentExecutionInfo current_execution_info;
  return current_execution_info;
}

// Resume the execution in `ready_context`.
void Resume(ExecutionContext& ready_context) {
  // We need to resume ready contexts through the thread pool.
  auto* work_queue = ready_context.work_queue();
  DCHECK(work_queue);
  work_queue->AddTask([&ready_context]() { Execute(ready_context); });
}

}  // namespace

namespace execute_internal {

void UnwindOnError(ExecutionContext& context, int64_t pc);

}

// The single-threaded execution of the kernels.
void Execute(ExecutionContext& ctx) {
  auto& current_execution_info = GetCurrentExecutionInfo();
  current_execution_info.ready_context = &ctx;

  for (; current_execution_info.ready_context;) {
    // Set up the current execution info so that kernel that calls Resume() can
    // access it.
    current_execution_info.current_context =
        current_execution_info.ready_context;
    current_execution_info.ready_context = nullptr;
    auto& context = *current_execution_info.current_context;

    DCHECK(!context.function_stack_.empty());

    int function_stack_index = context.function_stack_.size() - 1;
    FunctionContext* current_function = &context.function_stack_.back();
    int64_t pc = current_function->pc_;

    auto kernels = context.loaded_executable().kernels();

    auto kernel_object_iter =
        current_function->function_object().kernels().begin();
    kernel_object_iter += pc;

    KernelFrame::State kstate(current_function);
    KernelFrame frame(&kstate);

    // The main loop for executing kernels in program order. The kernels may set
    // the execution state to break this loop for context-switching or error
    // handling.
    for (; context.state_ == ExecutionContext::State::kRunning; ++pc) {
      DCHECK(kernel_object_iter <
             current_function->function_object().kernels().end());
      bc::Kernel kernel_object = *kernel_object_iter;
      frame.set_kernel(kernel_object);
      kernels[kernel_object.code()](frame);
      ++kernel_object_iter;
    }

    // Update the program counter if we need to break the sequential execution
    // loop.
    current_function = &context.function_stack_[function_stack_index];
    current_function->pc_ = pc;

    // The current execution is pasued, and we need to reset the per-thread
    // context pointer.
    current_execution_info.current_context = nullptr;

    // The state transition if a kernel set the execution state other than
    // kRunning.
    switch (context.state_) {
      case ExecutionContext::State::kReady: {
        DCHECK(current_execution_info.ready_context == nullptr);
        context.state_ = ExecutionContext::State::kRunning;
        if (current_function->kernel_context().reenter) {
          // Rewind PC to comeback to the kernel that calls a function.
          current_function->pc_--;
        }
        current_execution_info.ready_context = &context;
        break;
      }
      case ExecutionContext::State::kRunning:
        LOG(FATAL) << "This cannot happen.";  // Crash Ok
        break;
      case ExecutionContext::State::kReturn: {
        tsl::profiler::TraceMe trace_me("Execute::Return");
        context.function_stack_.pop_back();
        if (context.function_stack_.empty()) {
          if (context.exit_handler_) {
            std::move(context.exit_handler_)();
          }
          break;
        }
        DCHECK(current_execution_info.ready_context == nullptr);
        context.state_ = ExecutionContext::State::kRunning;
        current_execution_info.ready_context = &context;
        break;
      }
      case ExecutionContext::State::kSuspended: {
        DCHECK(current_execution_info.ready_context == nullptr);
        tsl::profiler::TraceMe trace_me("Execute::Suspend");
        DCHECK(context.suspend_handler_);
        std::move(context.suspend_handler_)([&context]() { Resume(context); });
        return;
      }
      case ExecutionContext::State::kError: {
        DCHECK(current_execution_info.ready_context == nullptr);
        tsl::profiler::TraceMe trace_me("Execute::Error");
        // Upon an error, we unwind the function stack by calling HandleError()
        // on each register.
        execute_internal::UnwindOnError(context, -1);
        return;
      }
    }
  }
}

namespace execute_internal {

void UnwindOnError(ExecutionContext& context, int64_t pc) {
  std::string function_name;
  if (!context.function_stack_.empty()) {
    function_name = context.function_stack_.back().function_object().name();
  }
  context.LogError(context.status());
  context.LogError(absl::InternalError(absl::StrCat(
      "UnwindOnError: start from function ", function_name,
      " with stack size: ", context.function_stack_.size(), " at pc: ", pc,
      " for context ", absl::Hex(reinterpret_cast<std::uintptr_t>(&context)),
      " at state ", context.state_)));

  while (!context.function_stack_.empty()) {
    DCHECK(context.state_ == ExecutionContext::State::kError);

    FunctionContext* current_function = &context.function_stack_.back();

    Value context_value(&context);

    if (pc == -1) {
      // Unwind the input registers.
      DCHECK(context.state_ == ExecutionContext::State::kError);
      ++pc;
      RegisterSpan input_reg_span(
          current_function->function_object().input_regs(),
          current_function->regs());

      for (Value& reg : input_reg_span) {
        reg.HandleError(context_value);
        if (context.state_ != ExecutionContext::State::kError) {
          DCHECK(context.state_ == ExecutionContext::State::kSuspended);

          context.LogError(absl::InternalError(absl::StrCat(
              "UnwindOnError: entering state", context.state_, " for context ",
              absl::Hex(reinterpret_cast<std::uintptr_t>(&context)))));

          // Rewind current pc so that the execution context come back to where
          // is is suspended.
          --pc;
          break;
        }
      }
    }

    context.LogError(absl::InternalError(
        absl::StrCat("UnwindOnError: unwinding function from ", pc, " to ",
                     current_function->pc_, " for context ",
                     absl::Hex(reinterpret_cast<std::uintptr_t>(&context)),
                     " at state ", context.state_)));

    for (; context.state_ == ExecutionContext::State::kError &&
           pc <= current_function->pc_;
         ++pc) {
      bc::Kernel kernel = current_function->function_object().kernels()[pc];

      RegisterSpan reg_span(kernel.results(), current_function->regs());

      for (Value& reg : reg_span) {
        reg.HandleError(context_value);
        if (context.state_ != ExecutionContext::State::kError) {
          DCHECK(context.state_ == ExecutionContext::State::kSuspended);
          context.LogError(absl::InternalError(absl::StrCat(
              "UnwindOnError: entering state", context.state_, " for context ",
              absl::Hex(reinterpret_cast<std::uintptr_t>(&context)))));

          // Rewind current pc so that the execution context come back to where
          // is is suspended.
          --pc;
          break;
        }
      }
    }

    if (context.state_ == ExecutionContext::State::kSuspended) {
      DCHECK(context.suspend_handler_)
          << "suspend_handler_ must be populated when the state is set to "
             "kSuspended.";
      context.LogError(absl::InternalError(absl::StrCat(
          "UnwindOnError: suspended state ", context.state_, " for context ",
          absl::Hex(reinterpret_cast<std::uintptr_t>(&context)))));
      std::move(context.suspend_handler_)([&context, pc]() {
        auto* work_queue = context.work_queue();
        DCHECK(work_queue);
        work_queue->AddTask([&context, pc]() {
          context.state_ = ExecutionContext::State::kError;
          UnwindOnError(context, pc);
        });
      });
      return;
    }

    DCHECK(context.state_ != ExecutionContext::State::kSuspended);

    pc = -1;
    context.function_stack_.pop_back();
  }

  context.LogError(absl::InternalError(absl::StrCat(
      "UnwindOnError: done for function ", function_name,
      " for context: ", absl::Hex(reinterpret_cast<std::uintptr_t>(&context)),
      " at state ", context.state_)));

  // Context may no longer be valid after exit_handler_ is called.
  if (context.exit_handler_) {
    std::move(context.exit_handler_)();
  }
}

}  // namespace execute_internal
}  // namespace mlrt
