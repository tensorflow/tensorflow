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

#include "tensorflow/compiler/xla/mlir/runtime/utils/async_runtime_api.h"

#include <stdlib.h>

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string_view>
#include <thread>  // NOLINT TODO(ezhulenev): Remove this header.
#include <utility>

#include "absl/base/dynamic_annotations.h"
#include "mlir/ExecutionEngine/AsyncRuntime.h"  // from @llvm-project
#include "tensorflow/compiler/xla/runtime/async_runtime.h"
#include "tensorflow/tsl/platform/mem.h"
#include "tfrt/concurrency/async_value.h"  // from @tf_runtime
#include "tfrt/concurrency/async_value_ref.h"  // from @tf_runtime
#include "tfrt/concurrency/chain.h"  // from @tf_runtime

namespace xla {
namespace runtime {

using tsl::AsyncValue;
using tsl::AsyncValueRef;
using tsl::Chain;
using tsl::port::AlignedFree;
using tsl::port::AlignedMalloc;

AsyncValueRef<Chain> ConvertAsyncTokenToChain(AsyncRuntime::Token *token) {
  auto *async_value = AsyncRuntime::GetAsyncValue(token);
  auto out_chain = AsyncValueRef<Chain>(FormRef(async_value));
  AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token));
  return out_chain;
}

void ExtractAsyncValue(
    AsyncRuntime::Value *value, AsyncValue *dst,
    absl::AnyInvocable<void(void *storage, AsyncValue *dst)> emplace_fn) {
  auto *async_value = AsyncRuntime::GetAsyncValue(value);

  // Fast path if async value is already available.
  if (async_value->IsAvailable()) {
    auto *storage = AsyncRuntime::GetStorage(value);
    std::move(emplace_fn)(storage, dst);
    AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
    return;
  }

  // Wait for the async value completion, and emplace the `dst`.
  async_value->AndThen([value, emplace_fn = std::move(emplace_fn),
                        dst = FormRef(dst)]() mutable {
    auto *storage = AsyncRuntime::GetStorage(value);
    emplace_fn(storage, dst.get());
    AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
  });
}

void ExtractAsyncValue(
    AsyncRuntime::Value *value, AsyncValue *dst, void *context,
    llvm::function_ref<void(void *storage, AsyncValue *dst, void *context)>
        emplace_fn) {
  auto *async_value = AsyncRuntime::GetAsyncValue(value);

  // Fast path if async value is already available.
  if (async_value->IsAvailable()) {
    auto *storage = AsyncRuntime::GetStorage(value);
    emplace_fn(storage, dst, context);
    AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
    return;
  }

  // Wait for the async value completion, and emplace the `dst`.
  async_value->AndThen([value, emplace_fn, context, dst = FormRef(dst)]() {
    auto *storage = AsyncRuntime::GetStorage(value);
    emplace_fn(storage, dst.get(), context);
    AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(value));
  });
}

llvm::orc::SymbolMap AsyncRuntimeApiSymbolMap(
    llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;

  auto bind = [&](std::string_view name, auto symbol_ptr) {
    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  bind("mlirAsyncRuntimeAddRef", &mlir::runtime::mlirAsyncRuntimeAddRef);
  bind("mlirAsyncRuntimeDropRef", &mlir::runtime::mlirAsyncRuntimeDropRef);
  bind("mlirAsyncRuntimeExecute", &mlir::runtime::mlirAsyncRuntimeExecute);
  bind("mlirAsyncRuntimeGetValueStorage",
       &mlir::runtime::mlirAsyncRuntimeGetValueStorage);
  bind("mlirAsyncRuntimeCreateToken",
       &mlir::runtime::mlirAsyncRuntimeCreateToken);
  bind("mlirAsyncRuntimeCreateValue",
       &mlir::runtime::mlirAsyncRuntimeCreateValue);
  bind("mlirAsyncRuntimeEmplaceToken",
       &mlir::runtime::mlirAsyncRuntimeEmplaceToken);
  bind("mlirAsyncRuntimeSetTokenError",
       &mlir::runtime::mlirAsyncRuntimeSetTokenError);
  bind("mlirAsyncRuntimeIsTokenError",
       &mlir::runtime::mlirAsyncRuntimeIsTokenError);
  bind("mlirAsyncRuntimeEmplaceValue",
       &mlir::runtime::mlirAsyncRuntimeEmplaceValue);
  bind("mlirAsyncRuntimeSetValueError",
       &mlir::runtime::mlirAsyncRuntimeSetValueError);
  bind("mlirAsyncRuntimeIsValueError",
       &mlir::runtime::mlirAsyncRuntimeIsValueError);
  bind("mlirAsyncRuntimeIsGroupError",
       &mlir::runtime::mlirAsyncRuntimeIsGroupError);
  bind("mlirAsyncRuntimeAwaitToken",
       &mlir::runtime::mlirAsyncRuntimeAwaitToken);
  bind("mlirAsyncRuntimeAwaitValue",
       &mlir::runtime::mlirAsyncRuntimeAwaitValue);
  bind("mlirAsyncRuntimeAwaitTokenAndExecute",
       &mlir::runtime::mlirAsyncRuntimeAwaitTokenAndExecute);
  bind("mlirAsyncRuntimeAwaitValueAndExecute",
       &mlir::runtime::mlirAsyncRuntimeAwaitValueAndExecute);
  bind("mlirAsyncRuntimeCreateGroup",
       &mlir::runtime::mlirAsyncRuntimeCreateGroup);
  bind("mlirAsyncRuntimeAddTokenToGroup",
       &mlir::runtime::mlirAsyncRuntimeAddTokenToGroup);
  bind("mlirAsyncRuntimeAwaitAllInGroup",
       &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroup);
  bind("mlirAsyncRuntimeAwaitAllInGroupAndExecute",
       &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroupAndExecute);
  bind("mlirAsyncRuntimePrintCurrentThreadId",
       &mlir::runtime::mlirAsyncRuntimePrintCurrentThreadId);

  return symbol_map;
}

namespace {

void *RuntimeAlignedAlloc(size_t alignment, size_t size) {
  return AlignedMalloc(size, alignment);
}

void *RuntimeMalloc(size_t size) {
  // AlignedAlloc() requires results to be deallocated with AlignedFree().
  // Make all allocations aligned because there is only one RuntimeFree().
  // Align to the size of a pointer by default.
  return RuntimeAlignedAlloc(sizeof(void *), size);
}

void RuntimeFree(void *ptr) { return AlignedFree(ptr); }

}  // namespace

llvm::orc::SymbolMap AsyncRuntimeMemoryAllocationSymbolMap(
    llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;

  auto bind = [&](std::string_view name, auto symbol_ptr) {
    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  bind("malloc", &RuntimeMalloc);
  bind("free", &RuntimeFree);
  bind("aligned_alloc", &RuntimeAlignedAlloc);

  return symbol_map;
}

}  // namespace runtime
}  // namespace xla

//===----------------------------------------------------------------------===//
// MLIR Async runtime API.
//===----------------------------------------------------------------------===//

// TODO(b/192775419): All pointers passed from the JIT compiled code to the
// runtime API must be marked initialized when running with msan enabled,
// because currently we do not have a way to enable sanitizer in the compiled
// code, and msan does not have any visibility into that code at runtime.

namespace mlir {
namespace runtime {

using xla::runtime::AsyncRuntime;
using xla::runtime::AsyncRuntimeObject;

// Adds references to reference counted runtime object.
void mlirAsyncRuntimeAddRef(RefCountedObjPtr ptr, int64_t count) {
  AsyncRuntimeObject *obj = static_cast<AsyncRuntimeObject *>(ptr);
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&ptr, sizeof(RefCountedObjPtr));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&count, sizeof(int64_t));
  AsyncRuntime::AddRef(obj, count);
}

// Drops references from reference counted runtime object.
void mlirAsyncRuntimeDropRef(RefCountedObjPtr ptr, int64_t count) {
  AsyncRuntimeObject *obj = static_cast<AsyncRuntimeObject *>(ptr);
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&ptr, sizeof(RefCountedObjPtr));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&count, sizeof(int64_t));
  AsyncRuntime::DropRef(obj, count);
}

// Create a new `async.token` in not-ready state.
AsyncToken *mlirAsyncRuntimeCreateToken() {
  return AsyncRuntime::CreateToken();
}

// Creates a new `async.value` in not-ready state.
AsyncValue *mlirAsyncRuntimeCreateValue(int64_t size) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&size, sizeof(int64_t));
  return AsyncRuntime::CreateValue(size,
                                   /*alignment=*/alignof(std::max_align_t));
}

// Create a new `async.group` in empty state.
AsyncGroup *mlirAsyncRuntimeCreateGroup(int64_t size) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&size, sizeof(int64_t));
  return AsyncRuntime::CreateGroup(size);
}

int64_t mlirAsyncRuntimeAddTokenToGroup(AsyncToken *token, AsyncGroup *group) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&group, sizeof(void *));
  return AsyncRuntime::AddTokenToGroup(group, token);
}

bool mlirAsyncRuntimeIsGroupError(AsyncGroup *group) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&group, sizeof(void *));
  return AsyncRuntime::IsError(group);
}

void mlirAsyncRuntimeEmplaceToken(AsyncToken *token) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  AsyncRuntime::SetAvailable(token);
}

void mlirAsyncRuntimeSetTokenError(AsyncToken *token) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  AsyncRuntime::SetError(token);
}

bool mlirAsyncRuntimeIsTokenError(AsyncToken *token) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  return AsyncRuntime::IsError(token);
}

void mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  AsyncRuntime &runtime = AsyncRuntime::GetCurrentRuntime();
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  runtime.AwaitToken(token);
}

void mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *group) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&group, sizeof(void *));
  AsyncRuntime::AwaitGroup(group);
}

ValueStorage mlirAsyncRuntimeGetValueStorage(AsyncValue *value) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  return AsyncRuntime::GetStorage(value);
}

void mlirAsyncRuntimeEmplaceValue(AsyncValue *value) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  AsyncRuntime::SetAvailable(value);
}

void mlirAsyncRuntimeSetValueError(AsyncValue *value) {
  AsyncRuntime &runtime = AsyncRuntime::GetCurrentRuntime();
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  runtime.SetError(value);
}

bool mlirAsyncRuntimeIsValueError(AsyncValue *value) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  return AsyncRuntime::IsError(value);
}

void mlirAsyncRuntimeAwaitValue(AsyncValue *value) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  AsyncRuntime::AwaitValue(value);
}

void mlirAsyncRuntimeExecute(CoroHandle handle, CoroResume resume) {
  AsyncRuntime &runtime = AsyncRuntime::GetCurrentRuntime();
  runtime.Execute([resume, handle, runtime]() {
    AsyncRuntime::Set(runtime);
    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token, CoroHandle handle,
                                          CoroResume resume) {
  // TODO(ezhulenev): Add fast path for available tokens.
  AsyncRuntime &runtime = AsyncRuntime::GetCurrentRuntime();
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&token, sizeof(void *));
  AsyncRuntime::AwaitToken(token, [handle, resume, runtime]() {
    AsyncRuntime::Set(runtime);
    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitValueAndExecute(AsyncValue *value, CoroHandle handle,
                                          CoroResume resume) {
  // TODO(ezhulenev): Add fast path for available values.
  AsyncRuntime &runtime = AsyncRuntime::GetCurrentRuntime();
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&value, sizeof(void *));
  runtime.AwaitValue(value, [handle, resume, runtime]() {
    AsyncRuntime::Set(runtime);

    (*resume)(handle);
  });
}

void mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *group,
                                               CoroHandle handle,
                                               CoroResume resume) {
  // TODO(ezhulenev): Add fast path for available groups.
  AsyncRuntime &runtime = AsyncRuntime::GetCurrentRuntime();
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&group, sizeof(void *));
  runtime.AwaitGroup(group, [handle, resume, runtime]() {
    AsyncRuntime::Set(runtime);
    (*resume)(handle);
  });
}

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

void mlirAsyncRuntimePrintCurrentThreadId() {
  static thread_local std::thread::id thisId = std::this_thread::get_id();
  std::cout << "Current thread id: " << thisId << std::endl;
}

}  // namespace runtime
}  // namespace mlir
