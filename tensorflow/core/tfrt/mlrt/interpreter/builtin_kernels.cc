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
#include "tensorflow/core/tfrt/mlrt/interpreter/builtin_kernels.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/function.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/async_handle.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/execute.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/future.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/register_span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"
#include "tsl/profiler/lib/traceme.h"

namespace mlrt {

void AsyncOp(KernelFrame frame) {
  tsl::profiler::TraceMe trace_me("mlrt.async");
  uint32_t func_idx = frame.attributes().GetAs<uint32_t>(0);

  auto& execution_context = frame.execution_context();

  auto function =
      execution_context.loaded_executable().executable().functions()[func_idx];

  trace_me.AppendMetadata([&]() {
    return tsl::profiler::TraceMeEncode({{"name", function.name().Get()}});
  });

  auto [promise, handle] = AsyncHandle::Allocate(execution_context);
  auto* work_queue = execution_context.work_queue();
  DCHECK(work_queue);

  handle.execution_context().set_exit_handler(
      [&execution_context = handle.execution_context(),
       promise = std::move(promise)]() mutable {
        std::move(promise).Finish(execution_context.status());
      });

  handle.execution_context().Call(function, frame.last_uses(),
                                  frame.arguments(),
                                  /*results=*/absl::Span<Value>());

  work_queue->AddTask(
      [&execution_context = handle.execution_context()]() mutable {
        Execute(execution_context);
      });

  frame.results()[0].Set<AsyncHandle>(std::move(handle));
}

void AwaitHandleOp(KernelFrame frame) {
  tsl::profiler::TraceMe trace_me("mlrt.await_handle");
  auto& handle = frame.arguments()[0].Get<AsyncHandle>();
  auto& execution_context = frame.execution_context();
  execution_context.Await(std::move(handle));
}

void AwaitAllHandleOp(KernelFrame frame) {
  tsl::profiler::TraceMe trace_me("mlrt.await_all_handle");
  RegisterValueSpan<AsyncHandle> handles(frame.arguments());
  auto& execution_context = frame.execution_context();
  execution_context.AwaitAll(handles);
}

void AllocateControlFutures(KernelFrame frame) {
  uint32_t num = frame.attributes().GetAs<uint32_t>(0);

  DCHECK_EQ(num * 2, frame.results().size());
  for (int i = 0; i < num; ++i) {
    auto promise = Promise::Allocate<Control>();
    frame.results()[num + i].Set<Future>(promise.GetFuture());
    frame.results()[i].Set<Promise>(std::move(promise));
  }
}

void AwaitControlOp(KernelFrame frame) {
  auto future = frame.arguments()[0].Get<Future>();
  auto& execution_context = frame.execution_context();
  execution_context.Await(std::move(future));
}

void AwaitAllControlOp(KernelFrame frame) {
  RegisterValueSpan<Future> futures(frame.arguments());
  auto& execution_context = frame.execution_context();
  execution_context.AwaitAll(futures);
}

void PromiseControlOp(KernelFrame frame) {
  auto& promise = frame.arguments()[0].Get<Promise>();
  std::move(promise).Set<Control>(Control{});
}

// The call op contains one uint32_t attribute that is the index into the
// functions list in the executable.
void CallOp(KernelFrame frame) {
  uint32_t func_idx = frame.attributes().GetAs<uint32_t>(0);

  auto& execution_context = frame.execution_context();

  auto function =
      execution_context.loaded_executable().executable().functions()[func_idx];

  execution_context.Call(function, frame.last_uses(), frame.arguments(),
                         frame.results());
}

struct CaseOp : KernelFrame {
  using KernelFrame::KernelFrame;

  static constexpr char kName[] = "mlrt.case";

  uint32_t branch_index() const { return arguments()[0].Get<uint32_t>(); }
  mlrt::bc::Vector<uint32_t> function_indices() const {
    return attributes().GetAs<mlrt::bc::Vector<uint32_t>>(0);
  }
  void Invoke();
};

void CaseOp::Invoke() {
  uint32_t argument_branch_idx = branch_index();
  mlrt::bc::Vector<uint32_t> attribute_function_indices = function_indices();

  if (argument_branch_idx >= attribute_function_indices.size()) {
    // Consistent with the behavior of the legacy TFRT case kernel behavior.
    argument_branch_idx = attribute_function_indices.size() - 1;
  }

  auto function =
      execution_context()
          .loaded_executable()
          .executable()
          .functions()[attribute_function_indices[argument_branch_idx]];
  execution_context().Call(function, last_uses().drop_front(),
                           arguments().drop_front(), results());
}

void ReturnOp(KernelFrame frame) {
  frame.execution_context().Return(frame.arguments());
}

void CondOp(KernelFrame frame) {
  bool cond = frame.arguments()[0].Get<bool>();

  uint32_t func_idx = frame.attributes().GetAs<uint32_t>(cond ? 0 : 1);

  auto& execution_context = frame.execution_context();

  auto function =
      execution_context.loaded_executable().executable().functions()[func_idx];

  execution_context.Call(function, frame.last_uses().drop_front(),
                         frame.arguments().drop_front(), frame.results());
}

struct WhileOp : KernelFrame {
  using KernelFrame::KernelFrame;

  static constexpr char kName[] = "mlrt.while";

  uint32_t body_function() const { return attributes().GetAs<uint32_t>(0); }
  void Invoke();
};

void WhileOp::Invoke() {
  tsl::profiler::TraceMe trace_me("mlrt.while");
  uint32_t func_idx = body_function();
  mlrt::bc::Function body_fn = execution_context()
                                   .loaded_executable()
                                   .executable()
                                   .functions()[func_idx];

  DCHECK_EQ(arguments().size(), results().size());
  DCHECK_EQ(body_fn.input_regs().size() + 1, arguments().size());
  DCHECK_EQ(body_fn.output_regs().size(), results().size());

  bool predicate;
  auto& kernel_context =
      execution_context().function_context().kernel_context();
  const int body_argument_size = arguments().size() - 1;
  if (!kernel_context.reenter) {
    // First time that enters this kernel.

    // Read the pass in initial boolean condition that decides whether the first
    // iteration will be executed.
    predicate = arguments()[0].Get<bool>();

    if (predicate) {
      // Executes the first iteration.
      DCHECK(kernel_context.registers.empty());
      kernel_context.registers.resize(body_argument_size);

      kernel_context.reenter++;
      execution_context().Call(body_fn, last_uses().drop_front(),
                               arguments().drop_front(), results());
    } else {
      // No execution at all; simply copy arguments (shift by 1) to results.
      for (int i = 0; i < body_argument_size; ++i) {
        results()[i] = arguments()[i + 1];
      }
    }
  } else {
    // Next iteration.

    // Read the last element of the previous iteration that decides whether we
    // continue or end iterations.
    predicate = results().back().Get<bool>();
    if (predicate) {
      // Continue to the next iteration.
      absl::Span<Value> body_args = absl::MakeSpan(kernel_context.registers);
      for (int i = 0; i < body_argument_size; ++i) {
        body_args[i] = std::move(results()[i]);
      }

      kernel_context.reenter++;
      execution_context().CallByMove(body_fn, body_args, results());
    } else {
      // Exit the loop. Frame results are already populated by the previous
      // iteration.
      kernel_context.reenter = 0;
      kernel_context.registers.clear();
    }
  }
}

void RegisterBuiltinKernels(KernelRegistry& registry) {
  // Keep kernels ordered by their names.
  registry.Register<CaseOp>();
  registry.Register<WhileOp>();
  registry.Register("mlrt.allocate_control_futures", &AllocateControlFutures);
  registry.Register("mlrt.async", &AsyncOp);
  registry.Register("mlrt.await_control", &AwaitControlOp);
  registry.Register("mlrt.await_all_control", &AwaitAllControlOp);
  registry.Register("mlrt.await_handle", &AwaitHandleOp);
  registry.Register("mlrt.await_all_handle", &AwaitAllHandleOp);
  registry.Register("mlrt.cond", &CondOp);
  registry.Register("mlrt.promise_control", &PromiseControlOp);
  // Built-in support for some non-MLRT specific OPs.
  registry.Register("call", &CallOp);
  registry.Register("return", &ReturnOp);
}

}  // namespace mlrt
