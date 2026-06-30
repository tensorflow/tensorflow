/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/tpu/tpu_executor_api.h"

#include <atomic>

#include "xla/tpu/tpu_executor_c_api.h"

namespace stream_executor {
namespace tpu {

namespace {
const TfTpu_ExecutorApiFn kEmptyExecutorApiFn{};
std::atomic<const TfTpu_ExecutorApiFn*> g_executor_api_fn_ptr{
    &kEmptyExecutorApiFn};
}  // namespace

TfTpu_ExecutorApiFn* ExecutorApiFn() {
  return const_cast<TfTpu_ExecutorApiFn*>(  // NOLINT
      g_executor_api_fn_ptr.load(std::memory_order_acquire));
}

void SetExecutorApiFn(const TfTpu_ExecutorApiFn* fn) {
  g_executor_api_fn_ptr.store(fn, std::memory_order_release);
}

bool IsStreamExecutorEnabled(const TfTpu_ExecutorApiFn* executor_api_fn) {
  return IsStreamExecutorEnabled();
}

bool IsInitialized(const TfTpu_ExecutorApiFn* executor_api_fn) {
  return IsInitialized();
}

bool IsStreamExecutorEnabled() { return false; }

bool IsInitialized() {
  auto* executor_api_fn = ExecutorApiFn();
  return executor_api_fn->TpuTopology_VersionFn != nullptr;
}

}  // namespace tpu
}  // namespace stream_executor
