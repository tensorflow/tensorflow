/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/runtime/async_runtime.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/base/dynamic_annotations.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "tsl/platform/mem.h"

// -------------------------------------------------------------------------- //
// Define AsyncToken and AsyncGroup in the mlir::runtime namespace to implement
// opaque structs defined in the MLIR Async Runtime API header file.
// -------------------------------------------------------------------------- //

namespace mlir {
namespace runtime {

using tsl::AsyncValueOwningRef;
using tsl::Chain;
using tsl::MakeAvailableAsyncValueRef;
using tsl::MakeConstructedAsyncValueRef;
using tsl::internal::AsyncValueStorage;

using xla::runtime::AsyncRuntimeObject;

using tsl::port::AlignedFree;
using tsl::port::AlignedMalloc;

struct AsyncToken : public AsyncRuntimeObject {
  explicit AsyncToken(unsigned ref_count = 1)
      : AsyncRuntimeObject(ref_count),
        chain(MakeConstructedAsyncValueRef<Chain>(storage)) {}

  tsl::AsyncValue* GetAsyncValue() const { return chain.AsPtr().value(); }

  AsyncValueStorage<Chain> storage;
  AsyncValueOwningRef<Chain> chain;
};

struct AsyncValue : public AsyncRuntimeObject {
  explicit AsyncValue(unsigned ref_count = 1)
      : AsyncRuntimeObject(ref_count),
        chain(MakeConstructedAsyncValueRef<Chain>(storage)) {}

  explicit AsyncValue(size_t size, size_t alignment, unsigned ref_count = 1)
      : AsyncRuntimeObject(ref_count),
        data_storage(Storage(size, alignment)),
        chain(MakeConstructedAsyncValueRef<Chain>(storage)) {
    // Storage memory will be initialized by the compiled executable.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(GetStorage(), size);
  }

  std::byte* GetStorage() {
    assert(!GetAsyncValue()->IsError() && "unexpected error state");
    assert(data_storage.has_value() && "unallocated data storage");
    if (data_storage->is_inline) return &data_storage->inline_buffer[0];
    return data_storage->allocated_buffer;
  }

  void AllocateStorage(size_t size, size_t alignment) {
    data_storage = Storage(size, alignment);
    // Storage memory will be initialized by the compiled executable.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(GetStorage(), size);
  }

  tsl::AsyncValue* GetAsyncValue() const { return chain.AsPtr().value(); }

  // If the requested async value storage is small, use the inlined storage.
  // Fall back on dynamic allocation if the requested storage size is large.
  struct Storage {
    static const int kSize = 128;  // enough to fit memref descriptor of rank 5
    static const int kAlign = alignof(std::max_align_t);

    Storage(size_t size, size_t alignment)
        : is_inline(CanStoreInline(size, alignment)) {
      if (!is_inline)
        allocated_buffer =
            reinterpret_cast<std::byte*>(AlignedMalloc(size, alignment));
    }

    ~Storage() {
      if (!is_inline) AlignedFree(allocated_buffer);
    }

    static bool CanStoreInline(size_t size, size_t alignment) {
      assert(absl::has_single_bit(alignment));
      return size <= kSize && alignment <= kAlign;
    }

    bool is_inline;
    union {
      alignas(kAlign) std::array<std::byte, kSize> inline_buffer;
      std::byte* allocated_buffer;
    };
  };

  std::optional<Storage> data_storage;

  // Async value that tracks value readiness. It becomes available when result
  // is written to the data storage and ready for consumption.
  AsyncValueStorage<Chain> storage;
  AsyncValueOwningRef<Chain> chain;
};

struct AsyncGroup : public AsyncRuntimeObject {
  explicit AsyncGroup(int64_t size, unsigned ref_count = 1)
      : AsyncRuntimeObject(ref_count),
        size(size),
        rank(0),
        pending_tokens(size),
        num_errors(0),
        completed(size == 0 ? MakeAvailableAsyncValueRef<Chain>(storage)
                            : MakeConstructedAsyncValueRef<Chain>(storage)) {
    assert(size >= 0 && "size can't be negative");
  }

  size_t AddToken(AsyncToken* token) {
    size_t token_rank = rank.fetch_add(1, std::memory_order_relaxed);
    assert(token_rank < size && "can't add more tokens than the group size");

    // When token becomes available drop the number of pending tokens and maybe
    // make the group completion async value available.
    token->GetAsyncValue()->AndThen([group = this, token]() {
      // Increment the number of errors in the group.
      if (token->GetAsyncValue()->IsError()) group->num_errors.fetch_add(1);

      // Pending tokens can't drop below zero.
      assert(group->pending_tokens > 0 && "wrong group size");

      // We do track group error state with the number of errors, and never
      // set completion async value state to error.
      if (group->pending_tokens.fetch_sub(1) == 1)
        group->completed.AsPtr().SetStateConcrete();
    });

    return token_rank;
  }

  tsl::AsyncValue* GetCompletionAsyncValue() const {
    return completed.AsPtr().value();
  }

  bool IsError() const { return num_errors.load() != 0; }

  int64_t size;
  std::atomic<int64_t> rank;
  std::atomic<int64_t> pending_tokens;
  std::atomic<int64_t> num_errors;

  // Async value that keeps track the group completion, it will become available
  // when the number of pending tokens will drop to zero.
  AsyncValueStorage<Chain> storage;
  AsyncValueOwningRef<Chain> completed;
};

}  // namespace runtime
}  // namespace mlir

// -------------------------------------------------------------------------- //

namespace xla {
namespace runtime {

using tsl::AsyncValue;

namespace {
// Always keep the current active async runtime in a thread local variable.
static thread_local AsyncRuntime async_runtime;

static_assert(std::is_trivially_destructible<AsyncRuntime>::value,
              "AsyncRuntime must be trivially destructible");

static_assert(std::is_trivially_copy_assignable<AsyncRuntime>::value,
              "AsyncRuntime must be trivially copy assignable");

static_assert(std::is_trivially_copy_constructible<AsyncRuntime>::value,
              "AsyncRuntime must be trivially copy constructible");

// This is an arbitrary limitation, to make sure that AsyncRuntime would not
// become expensive to copy unnoticed.
static_assert(sizeof(AsyncRuntime) == 1 * sizeof(void*),
              "AsyncRuntime must only hold one pointer");

}  // namespace

/*static*/ void AsyncRuntime::Set(AsyncRuntime runtime) {
  assert(runtime.runner() != nullptr);
  async_runtime = runtime;
}

/*static*/ AsyncRuntime& AsyncRuntime::GetCurrentRuntime() {
  assert(async_runtime.runner() != nullptr);
  return async_runtime;
}

/*static*/ std::byte* AsyncRuntime::GetStorage(Value* value) {
  return value->GetStorage();
}

/*static*/ void AsyncRuntime::AllocateStorage(Value* value, size_t size,
                                              size_t alignment) {
  return value->AllocateStorage(size, alignment);
}

/*static*/ AsyncValue* AsyncRuntime::GetAsyncValue(AsyncRuntime::Value* value) {
  return value->GetAsyncValue();
}

/*static*/ AsyncValue* AsyncRuntime::GetAsyncValue(AsyncRuntime::Token* token) {
  return token->GetAsyncValue();
}

/*static*/ AsyncValue* AsyncRuntime::GetAsyncValue(AsyncRuntime::Group* group) {
  return group->GetCompletionAsyncValue();
}

/*static*/ void AsyncRuntime::Await(AsyncValue* awaitable) {
  // Short circuit the trivial case.
  if (awaitable->IsAvailable()) return;
  tsl::BlockUntilReady(awaitable);
}

/*static*/ void AsyncRuntime::AddRef(AsyncRuntimeObject* obj, unsigned count) {
  assert(count == 1 && "AsyncRuntimeObject can add just one ref");
  obj->AddRef();
}

/*static*/ void AsyncRuntime::DropRef(AsyncRuntimeObject* obj, unsigned count) {
  assert(count == 1 && "AsyncRuntimeObject can drop just one ref");
  obj->DropRef();
}

/*static*/ AsyncRuntimeObject* AsyncRuntime::ToAsyncRuntimeObject(
    AsyncRuntime::Token* token) {
  return static_cast<AsyncRuntimeObject*>(token);
}

/*static*/ AsyncRuntimeObject* AsyncRuntime::ToAsyncRuntimeObject(
    AsyncRuntime::Value* value) {
  return static_cast<AsyncRuntimeObject*>(value);
}

/*static*/ AsyncRuntimeObject* AsyncRuntime::ToAsyncRuntimeObject(
    AsyncRuntime::Group* group) {
  return static_cast<AsyncRuntimeObject*>(group);
}

/*static*/ AsyncRuntime::Token* AsyncRuntime::CreateToken() {
  // AsyncRuntime::Token created with a reference count of 2 because it will be
  // returned to the `async.execute` caller and also will be later on emplaced
  // by the asynchronously executed task. If the caller immediately will drop
  // its reference we must ensure that the token will be alive until the
  // asynchronous operation is completed.
  return new AsyncRuntime::Token(/*ref_count=*/2);
}

/*static*/ void AsyncRuntime::SetAvailable(AsyncRuntime::Token* token) {
  token->GetAsyncValue()->SetStateConcrete();
  // Async tokens created with a ref count `2` to keep token alive until the
  // async task completes. Drop extra reference explicitly when token emplaced.
  DropRef(token);
}

/*static*/ void AsyncRuntime::SetError(AsyncRuntime::Token* token) {
  // TODO(ezhulenev): Construct a better diagnostincs when async runtime API
  // will support passing custom error messages.
  token->GetAsyncValue()->SetError(
      absl::InternalError("<async runtime error>"));
  // Async tokens created with a ref count `2` to keep token alive until the
  // async task completes. Drop extra reference explicitly when token emplaced.
  DropRef(token);
}

/*static*/ bool AsyncRuntime::IsError(AsyncRuntime::Token* token) {
  return token->GetAsyncValue()->IsError();
}

/*static*/ void AsyncRuntime::AwaitToken(AsyncRuntime::Token* token) {
  Await(token->GetAsyncValue());
}

/*static*/ AsyncRuntime::Value* AsyncRuntime::CreateValue() {
  // AsyncRuntime::Value created with a reference count of 2 because it will be
  // returned to the `async.execute` caller and also will be later on emplaced
  // by the asynchronously executed task. If the caller immediately will drop
  // its reference we must ensure that the value will be alive until the
  // asynchronous operation is completed.
  return new AsyncRuntime::Value(/*ref_count=*/2);
}

/*static*/ AsyncRuntime::Value* AsyncRuntime::CreateValue(size_t size,
                                                          size_t alignment) {
  // AsyncRuntime::Value created with a reference count of 2 because it will be
  // returned to the `async.execute` caller and also will be later on emplaced
  // by the asynchronously executed task. If the caller immediately will drop
  // its reference we must ensure that the value will be alive until the
  // asynchronous operation is completed.
  return new AsyncRuntime::Value(size, alignment, /*ref_count=*/2);
}

/*static*/ void AsyncRuntime::SetAvailable(AsyncRuntime::Value* value) {
  value->GetAsyncValue()->SetStateConcrete();
  // Async values created with a ref count `2` to keep token alive until the
  // async task completes. Drop extra reference explicitly when token emplaced.
  DropRef(value);
}

/*static*/ void AsyncRuntime::SetError(AsyncRuntime::Value* value) {
  // TODO(ezhulenev): Construct a better diagnostincs when async runtime API
  // will support passing custom error messages.
  value->GetAsyncValue()->SetError(
      absl::InternalError("<async runtime error>"));
  // Async values created with a ref count `2` to keep token alive until the
  // async task completes. Drop extra reference explicitly when token emplaced.
  DropRef(value);
}

/*static*/ bool AsyncRuntime::IsError(AsyncRuntime::Value* value) {
  return value->GetAsyncValue()->IsError();
}

/*static*/ void AsyncRuntime::AwaitValue(AsyncRuntime::Value* value) {
  Await(value->GetAsyncValue());
}

/*static*/ AsyncRuntime::Group* AsyncRuntime::CreateGroup(int64_t size) {
  return new AsyncRuntime::Group(size);
}

/*static*/ size_t AsyncRuntime::AddTokenToGroup(AsyncRuntime::Group* group,
                                                AsyncRuntime::Token* token) {
  return group->AddToken(token);
}

/*static*/ bool AsyncRuntime::IsError(AsyncRuntime::Group* group) {
  return group->IsError();
}

/*static*/ void AsyncRuntime::AwaitGroup(AsyncRuntime::Group* group) {
  Await(group->GetCompletionAsyncValue());
}

/*static*/ AsyncRuntime::Token* AsyncRuntime::AsToken(
    tsl::AsyncValueRef<tsl::Chain> chain) {
  AsyncRuntime::Token* token = CreateToken();

  chain.AndThen([token](absl::StatusOr<tsl::Chain*> status_or) {
    if (!status_or.ok()) {
      SetError(token);
    } else {
      SetAvailable(token);
    }
  });

  return token;
}

}  // namespace runtime
}  // namespace xla
