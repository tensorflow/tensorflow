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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_CONTEXT_H_
#define TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_CONTEXT_H_

#include <algorithm>
#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/function.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/kernel.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/attribute_span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/register_span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace mlrt {

class KernelFrame;
class ExecutionContext;

class Future;
template <typename T, typename FutureLikeContainer, typename ResultRefContainer>
Future AwaitAll(FutureLikeContainer futures, ResultRefContainer results);
template <typename FutureLikeContainer>
Future AwaitAll(FutureLikeContainer futures);

using KernelImplementation = void (*)(KernelFrame);

class KernelRegistry {
 public:
  void Register(absl::string_view name, KernelImplementation kernel);

  KernelImplementation Get(absl::string_view name) const;

  template <typename KernelClass>
  void Register(absl::string_view name);

  template <typename KernelClass>
  void Register() {
    Register<KernelClass>(KernelClass::kName);
  }

  void Merge(const KernelRegistry& other);

 private:
  absl::flat_hash_map<std::string, KernelImplementation> map_;
};

class LoadedExecutable {
 public:
  LoadedExecutable(bc::Executable executable,
                   const KernelRegistry& kernel_registry);

  absl::Span<const KernelImplementation> kernels() const { return kernels_; }

  bc::Function GetFunction(absl::string_view name) const {
    if (auto iter = functions_.find(name); iter != functions_.end()) {
      return iter->second;
    }

    return nullptr;
  }

  bc::Executable executable() const { return executable_; }

 private:
  bc::Executable executable_;

  absl::flat_hash_map<std::string, bc::Function> functions_;
  std::vector<KernelImplementation> kernels_;
};

// A helper structure that holds states for a kernel. Typical usuage is that a
// control kernel wants to call a function and then come back to the same
// kernel, e.g. WhileOp.
struct KernelContext {
  // Any non-zero value indicates the kernel just reentered.
  int reenter = 0;
  // Registers for callee.
  std::vector<Value> registers;
};

namespace execute_internal {

void UnwindOnError(ExecutionContext& context, int64_t pc);

}

class FunctionContext {
 public:
  FunctionContext(bc::Function function, ExecutionContext* execution_context)
      : pc_(0),
        registers_(function.num_regs()),
        function_object_(function),
        execution_context_(execution_context) {
    DCHECK(execution_context);
  }

  FunctionContext(const FunctionContext&) = delete;
  FunctionContext& operator=(const FunctionContext&) = delete;
  FunctionContext(FunctionContext&&) = default;
  FunctionContext& operator=(FunctionContext&&) = default;

  ExecutionContext& execution_context() { return *execution_context_; }

  const bc::Function& function_object() const { return function_object_; }

  absl::Span<Value> regs() { return absl::MakeSpan(registers_); }

  // Argument passing is via either copy or move.
  template <typename Args, typename Results>
  void Call(bc::Span<uint8_t> last_uses, Args args, Results results) {
    auto idx_iter = function_object_.input_regs().begin();

    DCHECK_EQ(function_object_.input_regs().size(), args.size());

    DCHECK_EQ(args.size(), last_uses.size());
    auto last_use_iter = last_uses.begin();
    for (auto& arg : args) {
      if (*last_use_iter) {
        registers_[*idx_iter] = std::move(arg);
      } else {
        registers_[*idx_iter] = arg;
      }
      ++idx_iter;
      ++last_use_iter;
    }

    results_.reserve(results.size());
    for (auto& result : results) {
      results_.push_back(&result);
    }
  }

  // Argument passing is via move.
  template <typename Args, typename Results>
  void CallByMove(Args args, Results results) {
    auto idx_iter = function_object_.input_regs().begin();

    DCHECK_EQ(function_object_.input_regs().size(), args.size());

    for (auto& arg : args) {
      registers_[*idx_iter] = std::move(arg);
      ++idx_iter;
    }

    results_.reserve(results.size());
    for (auto& result : results) {
      results_.push_back(&result);
    }
  }

  // The return operation copies or moves (if not a ref) the results.
  void Return(RegisterSpan results) {
    DCHECK_EQ(results.size(), function_object_.output_regs().size());
    auto result_iter = results.begin();
    auto output_last_uses = function_object_.output_last_uses();

    for (int i = 0; i < results_.size(); ++i) {
      auto* result = results_[i];

      if (!output_last_uses.empty() && output_last_uses[i]) {
        // We only move the result only if it is the last use.
        *result = std::move(*result_iter);
      } else {
        *result = *result_iter;
      }
      ++result_iter;
    }
  }

  const KernelContext& kernel_context() const { return kernel_context_; }
  KernelContext& kernel_context() { return kernel_context_; }

 private:
  int64_t pc_;
  std::vector<Value> registers_;
  std::vector<Value*> results_;
  bc::Function function_object_;
  KernelContext kernel_context_;

  ExecutionContext* execution_context_ = nullptr;

  friend class ExecutionContext;
  friend void Execute(ExecutionContext& context);
  friend void execute_internal::UnwindOnError(ExecutionContext& context,
                                              int64_t pc);
};

namespace context_internal {

inline std::atomic<int>& GetNextId() {
  static std::atomic<int> next_id = 0;
  return next_id;
}

class UserContextBase {
 public:
  virtual ~UserContextBase();

  virtual std::unique_ptr<UserContextBase> Copy() const = 0;
};

}  // namespace context_internal

// Every user context should inherit from this class. Internally it generates a
// unique id for each user context type for internal management.
template <typename Derived>
class UserContext : public context_internal::UserContextBase {
 public:
  using Base = context_internal::UserContextBase;

  static int id() { return id_; }

  std::unique_ptr<Base> Copy() const final {
    return std::make_unique<Derived>(*static_cast<const Derived*>(this));
  }

 private:
  inline static int id_ = context_internal::GetNextId()++;
};

class ExecutionContext {
 public:
  explicit ExecutionContext(const LoadedExecutable* loaded_executable)
      : user_contexts_(context_internal::GetNextId().load()),
        loaded_executable_(loaded_executable) {}

  ExecutionContext(
      const LoadedExecutable* loaded_executable,
      std::vector<std::unique_ptr<context_internal::UserContextBase>>
          user_contexts)
      : user_contexts_(std::move(user_contexts)),
        loaded_executable_(loaded_executable) {}

  void set_exit_handler(absl::AnyInvocable<void() &&> exit_handler) {
    exit_handler_ = std::move(exit_handler);
  }

  tfrt::ConcurrentWorkQueue* work_queue() const { return work_queue_; }

  void set_work_queue(tfrt::ConcurrentWorkQueue* work_queue) {
    work_queue_ = work_queue;
  }

  template <typename Args, typename Results>
  void Call(bc::Function function_object, bc::Span<uint8_t> last_uses,
            Args args, Results results) {
    auto& function_context =
        function_stack_.emplace_back(function_object, this);
    function_context.Call(last_uses, args, results);
    state_ = State::kReady;
  }

  template <typename Args, typename Results>
  void CallByMove(bc::Function function_object, Args args, Results results) {
    auto& function_context =
        function_stack_.emplace_back(function_object, this);
    function_context.CallByMove(args, results);
    state_ = State::kReady;
  }

  void Return(RegisterSpan results) {
    auto& function_context = function_stack_.back();
    function_context.Return(results);
    state_ = State::kReturn;
  }

  size_t function_stack_size() const { return function_stack_.size(); }
  FunctionContext& function_context() { return function_stack_.back(); }

  // Enqueues the current execution to the wait list of the `future`. Once the
  // `future` is ready, the execution will be resumed. And the value will be
  // populated in `result` if it is not an error.
  template <typename T, typename FutureLike>
  void Await(FutureLike future, Value* result) {
    if (future.IsReady()) {
      if (future.IsError()) {
        Fail(future.GetError());
      } else {
        std::move(future).Then(
            [result](T value) { result->Set(std::move(value)); });
      }
      return;
    }

    state_ = State::kSuspended;
    suspend_handler_ = [this, result, future = std::move(future)](
                           absl::AnyInvocable<void()&&> resume) mutable {
      std::move(future).Then([this, result, resume = std::move(resume)](
                                 absl::StatusOr<T> value) mutable {
        if (!value.ok()) {
          Fail(std::move(value).status());
        } else {
          result->Set(*std::move(value));
          state_ = State::kRunning;
        }

        std::move(resume)();
      });
    };
  }

  template <typename FutureLike>
  void Await(FutureLike future) {
    if (future.IsReady()) {
      if (future.IsError()) {
        Fail(future.GetError());
      }
      return;
    }

    state_ = State::kSuspended;
    suspend_handler_ = [this, future = std::move(future)](
                           absl::AnyInvocable<void()&&> resume) mutable {
      std::move(future).Then(
          [this, resume = std::move(resume)](absl::Status status) mutable {
            if (!status.ok()) {
              Fail(std::move(status));
            } else {
              state_ = State::kRunning;
            }

            std::move(resume)();
          });
    };
  }

  template <typename T, typename FutureLikeContainer,
            typename ResultRefContainer>
  ABSL_ATTRIBUTE_ALWAYS_INLINE void AwaitAll(FutureLikeContainer futures,
                                             ResultRefContainer results) {
    auto future = mlrt::AwaitAll<T>(futures, results);

    if (future.IsReady()) {
      if (future.IsError()) {
        Fail(future.GetError());
      }
      return;
    }

    state_ = State::kSuspended;
    suspend_handler_ = [this, future = std::move(future)](
                           absl::AnyInvocable<void()&&> resume) mutable {
      std::move(future).Then(
          [this, resume = std::move(resume)](absl::Status status) mutable {
            state_ = State::kRunning;

            if (!status.ok()) {
              Fail(status);
            }

            std::move(resume)();
          });
    };
  }

  template <typename FutureLikeContainer>
  ABSL_ATTRIBUTE_ALWAYS_INLINE void AwaitAll(FutureLikeContainer futures) {
    auto future = mlrt::AwaitAll(futures);

    if (future.IsReady()) {
      if (future.IsError()) {
        Fail(future.GetError());
      }
      return;
    }

    state_ = State::kSuspended;
    suspend_handler_ = [this, future = std::move(future)](
                           absl::AnyInvocable<void()&&> resume) mutable {
      std::move(future).Then(
          [this, resume = std::move(resume)](absl::Status status) mutable {
            state_ = State::kRunning;

            if (!status.ok()) {
              Fail(status);
            }

            std::move(resume)();
          });
    };
  }

  const LoadedExecutable& loaded_executable() const {
    return *loaded_executable_;
  }

  void Fail(absl::Status status) {
    state_ = State::kError;
    status_ = std::move(status);
  }

  void FailOnCancellation() { Fail(absl::CancelledError()); }

  const absl::Status& status() const { return status_; }

  // Add an instance of user context to the execution context.
  template <typename T>
  void AddUserContext(std::unique_ptr<T> user_context) {
    static_assert(std::is_base_of_v<UserContext<T>, T>);
    DCHECK_LT(T::id(), user_contexts_.size());
    user_contexts_[T::id()] = std::move(user_context);
  }

  // Return an reference to the user context.
  template <typename T>
  T& GetUserContext() const {
    static_assert(std::is_base_of_v<UserContext<T>, T>);
    DCHECK_LT(T::id(), user_contexts_.size());
    return *static_cast<T*>(user_contexts_[T::id()].get());
  }

  std::vector<std::unique_ptr<context_internal::UserContextBase>>
  CopyUserContexts() const {
    std::vector<std::unique_ptr<context_internal::UserContextBase>>
        user_contexts;
    user_contexts.reserve(user_contexts_.size());
    for (const auto& user_context : user_contexts_) {
      if (user_context) {
        user_contexts.push_back(user_context->Copy());
      } else {
        user_contexts.push_back(nullptr);
      }
    }
    return user_contexts;
  }

  enum class State {
    // The function is pushed to the stack, and ready for execution.
    kReady = 0,

    // The function is being executed and has not reached the return op yet.
    kRunning,

    // The function finished executing the return op, and ready for being popped
    // from the stack.
    kReturn,

    // The function is suspended from execution due to context switches.
    kSuspended,

    // The execution reports an error in the current thread, and the execution
    // will be aborted by cleaning the states.
    kError
  };
  State state() const { return state_; }

 private:
  absl::InlinedVector<FunctionContext, 2> function_stack_;

  State state_ = State::kReady;

  absl::Status status_;

  // The `suspend_handler_` is a callable whose argument is another callable
  // that resumes the execution (or error handling).
  absl::AnyInvocable<void(absl::AnyInvocable<void() &&> resume) &&>
      suspend_handler_;
  absl::AnyInvocable<void() &&> exit_handler_;

  tfrt::ConcurrentWorkQueue* work_queue_ = nullptr;

  std::vector<std::unique_ptr<context_internal::UserContextBase>>
      user_contexts_;

  const LoadedExecutable* loaded_executable_ = nullptr;

  friend class AsyncHandle;
  friend void Execute(ExecutionContext& context);
  friend void execute_internal::UnwindOnError(ExecutionContext& context,
                                              int64_t pc);
};

class KernelFrame {
 public:
  struct State {
    State(absl::Span<Value> regs, bc::Span<bc::String> attrs,
          ExecutionContext* execution_context)
        : regs(regs), attrs(attrs), execution_context(execution_context) {
      DCHECK(execution_context);
    }

    explicit State(FunctionContext* function_context)
        : State(function_context->regs(),
                function_context->execution_context()
                    .loaded_executable()
                    .executable()
                    .attributes(),
                &function_context->execution_context()) {}

    bc::Kernel kernel;
    absl::Span<Value> regs;
    bc::Span<bc::String> attrs;
    ExecutionContext* execution_context = nullptr;
  };

  explicit KernelFrame(State* state) : state_(state) { DCHECK(state_); }

  template <typename T>
  operator T() const {  // NOLINT
    return T(state_);
  }

  RegisterSpan arguments() const {
    return RegisterSpan(kernel().arguments(), regs());
  }

  RegisterSpan results() const {
    return RegisterSpan(kernel().results(), regs());
  }

  AttributeSpan attributes() const {
    return AttributeSpan(kernel().attributes(), attrs());
  }

  bc::Span<uint8_t> last_uses() const { return kernel().last_uses(); }

  ExecutionContext& execution_context() { return *state_->execution_context; }
  const ExecutionContext& execution_context() const {
    return *state_->execution_context;
  }

  void set_kernel(bc::Kernel kernel) { this->kernel() = kernel; }

 private:
  bc::Kernel& kernel() { return state_->kernel; }
  const bc::Kernel& kernel() const { return state_->kernel; }

  absl::Span<Value> regs() const { return state_->regs; }
  bc::Span<bc::String> attrs() const { return state_->attrs; }

  State* state_ = nullptr;

  friend void Execute(ExecutionContext& context);
};

template <typename KernelClass>
inline void KernelRegistry::Register(absl::string_view name) {
  Register(
      name, +[](KernelFrame frame) { KernelClass(frame).Invoke(); });
}

}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_CONTEXT_H_
