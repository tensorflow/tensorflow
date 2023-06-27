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

#include <utility>

#include "absl/log/log.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

namespace mlrt {
namespace execute_internal {

void UnwindOnError(ExecutionContext& context, int64_t pc);

}

// The single-threaded execution of the kernels.
void Execute(ExecutionContext& context) {
  for (;;) {
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

    // The state transition if a kernel set the execution state other than
    // kRunning.
    switch (context.state_) {
      case ExecutionContext::State::kReady: {
        context.state_ = ExecutionContext::State::kRunning;
        if (current_function->kernel_context().reenter) {
          // Rewind PC to comeback to the kernel that calls a function.
          current_function->pc_--;
        }
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
          return;
        }
        context.state_ = ExecutionContext::State::kRunning;
        break;
      }
      case ExecutionContext::State::kSuspended: {
        tsl::profiler::TraceMe trace_me("Execute::Suspend");
        DCHECK(context.suspend_handler_);
        std::move(context.suspend_handler_)([&context]() {
          auto* work_queue = context.work_queue();
          DCHECK(work_queue);
          work_queue->AddTask([&context]() { Execute(context); });
        });
        return;
      }
      case ExecutionContext::State::kError: {
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
          // Rewind current pc so that the execution context come back to where
          // is is suspended.
          --pc;
          break;
        }
      }
    }

    for (; context.state_ == ExecutionContext::State::kError &&
           pc <= current_function->pc_;
         ++pc) {
      bc::Kernel kernel = current_function->function_object().kernels()[pc];

      RegisterSpan reg_span(kernel.results(), current_function->regs());

      for (Value& reg : reg_span) {
        reg.HandleError(context_value);
        if (context.state_ != ExecutionContext::State::kError) {
          DCHECK(context.state_ == ExecutionContext::State::kSuspended);
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

  if (context.exit_handler_) {
    std::move(context.exit_handler_)();
  }
}

}  // namespace execute_internal
}  // namespace mlrt
