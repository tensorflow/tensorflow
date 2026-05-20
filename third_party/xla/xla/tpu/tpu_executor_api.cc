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

const TfTpu_ExecutorApiFn* ExecutorApiFn() {
  return g_executor_api_fn_ptr.load(std::memory_order_acquire);
}

void SetExecutorApiFn(const TfTpu_ExecutorApiFn* fn) {
  g_executor_api_fn_ptr.store(fn, std::memory_order_release);
}

bool IsStreamExecutorEnabled() {
  auto* executor_api_fn = ExecutorApiFn();
  // Check if an arbitrary function pointer is initialized. We could check more
  // functions or add an explicit 'initialized' field to TfTpu_ExecutorApiFn,
  // but this works well enough.
  if (executor_api_fn->TpuPlatform_NewFn == nullptr) {
    return false;
  }
  bool is_se_enabled = false;
  auto* tpu_platform = executor_api_fn->TpuPlatform_NewFn();
  if (tpu_platform != nullptr) {
    is_se_enabled = true;
    executor_api_fn->TpuPlatform_FreeFn(tpu_platform);
  }
  return is_se_enabled;
}

bool IsInitialized() {
  // Check if an arbitrary function pointer is initialized. We could check more
  // functions or add an explicit 'initialized' field to TfTpu_ExecutorApiFn,
  // but this works well enough.
  auto* executor_api_fn = ExecutorApiFn();
  return executor_api_fn->TpuPlatform_NewFn != nullptr;
}

}  // namespace tpu
}  // namespace stream_executor
