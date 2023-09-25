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

#ifndef XLA_MLIR_RUNTIME_UTILS_ASYNC_RUNTIME_API_H_
#define XLA_MLIR_RUNTIME_UTILS_ASYNC_RUNTIME_API_H_

#include "absl/functional/any_invocable.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "xla/runtime/async_runtime.h"
#include "tfrt/concurrency/async_value_ref.h"  // from @tf_runtime
#include "tfrt/concurrency/chain.h"  // from @tf_runtime

namespace xla {
namespace runtime {

// Converts MLIR Async Runtime token into the TFRT async chain, and drops the
// reference count on the token.
tsl::AsyncValueRef<tsl::Chain> ConvertAsyncTokenToChain(
    AsyncRuntime::Token* token);

// Extracts a payload from the MLIR Async Runtime `value` and emplaces it into
// the TFRT async value `dst` using a user provided emplace function. Drops the
// reference on the runtime value after it is no longer needed.
void ExtractAsyncValue(
    AsyncRuntime::Value* value, tsl::AsyncValue* dst,
    absl::AnyInvocable<void(void*, tsl::AsyncValue*)> emplace_fn);

// A version of the `ExtractAsyncValue` function defined above that takes an
// additional opaque pointer that will be passed to the emplace function when
// async value will become ready. It is the caller responsibility to ensure that
// the pointed object will stay alive.
void ExtractAsyncValue(
    AsyncRuntime::Value* value, tsl::AsyncValue* dst, void* context,
    llvm::function_ref<void(void*, tsl::AsyncValue*, void*)> emplace_fn);

// Builds a symbol map from the Async Runtime API functions.
llvm::orc::SymbolMap AsyncRuntimeApiSymbolMap(
    llvm::orc::MangleAndInterner mangle);

// TODO(ezhulenev): This should not be a part of async runtime api library.
llvm::orc::SymbolMap AsyncRuntimeMemoryAllocationSymbolMap(
    llvm::orc::MangleAndInterner mangle);

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_UTILS_ASYNC_RUNTIME_API_H_
