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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_ASYNC_RUNTIME_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_ASYNC_RUNTIME_H_

#define EIGEN_USE_THREADS

#include <cstddef>
#include <functional>
#include <utility>

#include "tensorflow/tsl/platform/threadpool.h"
#include "tfrt/concurrency/async_value.h"  // from @tf_runtime
#include "tfrt/concurrency/async_value_ref.h"  // from @tf_runtime
#include "tfrt/concurrency/chain.h"  // from @tf_runtime
#include "tfrt/concurrency/ref_count.h"  // from @tf_runtime

namespace mlir {
namespace runtime {

// Async runtime in the XLA implements the MLIR async runtime API that supports
// the lowering of the `async` dialect to the LLVM and LLVM coroutines.
struct AsyncToken;
struct AsyncValue;
struct AsyncGroup;

}  // namespace runtime
}  // namespace mlir

namespace xla {
namespace runtime {

// Forward declare a base class for async runtime objects.
class AsyncRuntimeObject;

// Async task runner abstracts over the underlying thread pool (or concurrent
// work queue) implementation.
class AsyncTaskRunner {
 public:
  using Task = std::function<void()>;
  virtual ~AsyncTaskRunner() = default;
  virtual void Schedule(Task task) = 0;
};

class AsyncRuntime {
 public:
  using Token = ::mlir::runtime::AsyncToken;
  using Value = ::mlir::runtime::AsyncValue;
  using Group = ::mlir::runtime::AsyncGroup;

  explicit AsyncRuntime(AsyncTaskRunner* runner) : runner_(runner) {
    assert(runner != nullptr && "async task runner must be not null");
  }

  // We need a default constructor to define a thread local variable for async
  // runtime passing between tasks (see implementation in async_runtime_api.cc).
  AsyncRuntime() : runner_(nullptr) {}

  // ------------------------------------------------------------------------ //
  // Implicit AsyncRuntime propagation.
  // ------------------------------------------------------------------------ //

  // Set the AsyncRuntime that will be implicitly propagated to all async tasks.
  //
  // On every launch of an async task (see `async_runtime_api.h`), current async
  // runtime will be captured, and restored when the task will start its
  // execution on a different thread.
  static void Set(AsyncRuntime runtime);

  // Returns the current async runtime.
  static AsyncRuntime& GetCurrentRuntime();

  // ------------------------------------------------------------------------ //
  // Async Token API.
  // ------------------------------------------------------------------------ //

  // Creates a new token in not-ready state.
  static Token* CreateToken();

  // Switches the token to the available state and runs all the awaiters.
  static void SetAvailable(Token* token);

  // Switches the token to the error state and runs all the awaiters.
  static void SetError(Token* token);

  // Returns `true` if the token is in the error state.
  static bool IsError(Token* token);

  // Blocks the caller thread until the token becomes ready.
  static void AwaitToken(Token* token);

  // ------------------------------------------------------------------------ //
  // Async Value API.
  // ------------------------------------------------------------------------ //

  // Creates a new value in not-ready state without allocating storage
  static Value* CreateValue();

  // Creates a new value in not-ready state with a storage of the given size.
  static Value* CreateValue(size_t size, size_t alignment);

  // Switches the value to the available state and runs all the awaiters.
  static void SetAvailable(Value* value);

  // Switches the value to the error state and runs all the awaiters.
  static void SetError(Value* value);

  // Returns `true` if the value is in the error state.
  static bool IsError(Value* value);

  // Blocks the caller thread until the value becomes ready.
  static void AwaitValue(Value* value);

  // ------------------------------------------------------------------------ //
  // Async Group API.
  // ------------------------------------------------------------------------ //

  // Creates a new empty group.
  static Group* CreateGroup(int64_t size);

  // Adds `token` to the `group`.
  static size_t AddTokenToGroup(Group* group, Token* token);

  // Returns `true` if the group is in the error state (any of the tokens or
  // values added to the group is in the error state).
  static bool IsError(Group* group);

  // Blocks the caller thread until the group becomes ready (all tokens that
  // were added to the group are emplaced).
  static void AwaitGroup(Group* group);

  // ------------------------------------------------------------------------ //
  // Execution and continuation based resumption API.
  // ------------------------------------------------------------------------ //

  // Execute the callable `f` on a thread managed by the runtime.
  template <typename F>
  void Execute(F&& f);

  // Await operation that do not block the caller thread, but instead execute
  // the callable `F` when the token/group become ready.
  template <typename F>
  static void AwaitToken(Token* token, F&& f);
  template <typename F>
  static void AwaitValue(Value* value, F&& f);
  template <typename F>
  static void AwaitGroup(Group* group, F&& f);

  // ------------------------------------------------------------------------ //

  // Returns a pointer to the async value storage.
  static std::byte* GetStorage(Value* value);

  // Allocate storage for the async value
  static void AllocateStorage(Value* value, size_t size, size_t alignment);

  // Extracts async value that holds a chain owned by the value.
  static tsl::AsyncValue* GetAsyncValue(Value* value);

  // Extracts async value that is owned by the token.
  static tsl::AsyncValue* GetAsyncValue(Token* token);

  // Extracts async value that signals group completion.
  static tsl::AsyncValue* GetAsyncValue(Group* group);

  // Reference counting operations for the runtime objects.
  static void AddRef(AsyncRuntimeObject* obj, unsigned count = 1);
  static void DropRef(AsyncRuntimeObject* obj, unsigned count = 1);

  // Convert Token/Value/Group to AsyncRuntimeObject*;
  static AsyncRuntimeObject* ToAsyncRuntimeObject(Token* token);
  static AsyncRuntimeObject* ToAsyncRuntimeObject(Value* value);
  static AsyncRuntimeObject* ToAsyncRuntimeObject(Group* group);

  // Convert async value/token to async runtime object.
  static Token* AsToken(tsl::AsyncValueRef<tsl::Chain> chain);

  template <typename T>
  static Value* AsValue(
      tsl::AsyncValueRef<T> value, size_t size, size_t alignment,
      absl::FunctionRef<void(const T*, std::byte* storage)> write) {
    Value* runtime_async_value = AsyncRuntime::CreateValue(size, alignment);
    value.AndThen([runtime_async_value, write](absl::StatusOr<T*> status_or) {
      if (!status_or.ok()) {
        AsyncRuntime::SetError(runtime_async_value);
      } else {
        auto* store = AsyncRuntime::GetStorage(runtime_async_value);
        write(*status_or, store);
        AsyncRuntime::SetAvailable(runtime_async_value);
      }
    });
    return runtime_async_value;
  }

  template <typename T>
  static Value* AsValue(
      tsl::AsyncValueRef<T> value,
      absl::FunctionRef<std::pair<size_t, size_t>(const T*)> size_and_alignment,
      absl::FunctionRef<void(const T*, std::byte* storage)> write) {
    Value* runtime_async_value = AsyncRuntime::CreateValue();
    value.AndThen([runtime_async_value, size_and_alignment,
                   write](absl::StatusOr<T*> status_or) {
      if (!status_or.ok()) {
        AsyncRuntime::SetError(runtime_async_value);
      } else {
        auto size_alignment = size_and_alignment(*status_or);
        AsyncRuntime::AllocateStorage(runtime_async_value, size_alignment.first,
                                      size_alignment.second);
        auto* store = AsyncRuntime::GetStorage(runtime_async_value);
        write(*status_or, store);
        AsyncRuntime::SetAvailable(runtime_async_value);
      }
    });
    return runtime_async_value;
  }

  AsyncTaskRunner* runner() const { return runner_; }

 private:
  // Blocks the caller thread until awaitable async value becomes available.
  static void Await(tsl::AsyncValue* awaitable);

  AsyncTaskRunner* runner_;  // must outlive *this
};

// A base class for all Async dialect types reference counted at runtime.
class AsyncRuntimeObject : public tsl::ReferenceCounted<AsyncRuntimeObject> {
 public:
  using ReferenceCounted::ReferenceCounted;  // inherit constructors
  virtual ~AsyncRuntimeObject() = default;
};

template <typename F>
void AsyncRuntime::Execute(F&& f) {
  runner_->Schedule(std::forward<F>(f));
}

template <typename F>
/*static*/ void AsyncRuntime::AwaitToken(Token* token, F&& f) {
  AsyncRuntime::GetAsyncValue(token)->AndThen(std::forward<F>(f));
}

template <typename F>
/*static*/ void AsyncRuntime::AwaitValue(Value* value, F&& f) {
  AsyncRuntime::GetAsyncValue(value)->AndThen(std::forward<F>(f));
}

template <typename F>
/*static*/ void AsyncRuntime::AwaitGroup(Group* group, F&& f) {
  AsyncRuntime::GetAsyncValue(group)->AndThen(std::forward<F>(f));
}

//===-----------------------------------------------------------------------===/
// AsyncTaskRunner implementation on top of the default ThreadPool.
//===-----------------------------------------------------------------------===/

class ThreadPoolAsyncTaskRunner : public AsyncTaskRunner {
 public:
  explicit ThreadPoolAsyncTaskRunner(tsl::thread::ThreadPool* thread_pool)
      : thread_pool_(thread_pool) {}
  void Schedule(Task task) final { thread_pool_->Schedule(std::move(task)); }

 private:
  tsl::thread::ThreadPool* thread_pool_;
};

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_ASYNC_RUNTIME_H_
