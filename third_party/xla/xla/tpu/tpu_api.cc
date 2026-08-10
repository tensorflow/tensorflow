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

#include "xla/tpu/tpu_api.h"

#include <atomic>

#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "xla/tpu/tpu_ops_c_api.h"
#include "xla/tpu/tpu_profiler_c_api.h"

namespace stream_executor {
namespace tpu {

namespace {
absl::once_flag g_tpu_executor_init_once;
absl::once_flag g_tpu_profiler_init_once;
absl::once_flag g_tpu_ops_struct_fns_once;
absl::once_flag g_tpu_base_api_init_once;
}  // namespace

absl::once_flag* GetTpuExecutorInitOnceFlag() {
  return &g_tpu_executor_init_once;
}
absl::once_flag* GetTpuProfilerInitOnceFlag() {
  return &g_tpu_profiler_init_once;
}
absl::once_flag* GetTpuOpsStructFnsOnceFlag() {
  return &g_tpu_ops_struct_fns_once;
}
absl::once_flag* GetTpuBaseApiInitOnceFlag() {
  return &g_tpu_base_api_init_once;
}

// Returns a reference to a globally unique absl::Status for TPU Ops struct
// initialization.
absl::Status& GetTpuOpsStructFnsInitStatus() {
  static absl::NoDestructor<absl::Status> g_tpu_ops_struct_fns_init_status{
      absl::UnknownError("TPU Ops API not initialized")};
  return *g_tpu_ops_struct_fns_init_status;
}

// Returns a reference to a globally unique absl::Status for TPU Base API
// initialization.
absl::Status& GetTpuBaseApiInitStatus() {
  static absl::NoDestructor<absl::Status> g_tpu_base_api_init_status{
      absl::UnknownError("TPU Base API not initialized")};
  return *g_tpu_base_api_init_status;
}

// Returns a reference to a globally unique absl::Status for TPU executor
// initialization.
absl::Status& GetTpuExecutorInitStatus() {
  static absl::NoDestructor<absl::Status> g_tpu_executor_init_status{
      absl::FailedPreconditionError("TPU executor not yet initialized.")};
  return *g_tpu_executor_init_status;
}

// Returns a reference to a globally unique absl::Status for TPU profiler
// initialization.
absl::Status& GetTpuProfilerInitStatus() {
  static absl::NoDestructor<absl::Status> g_tpu_profiler_init_status{
      absl::FailedPreconditionError("TPU profiler not yet initialized.")};
  return *g_tpu_profiler_init_status;
}

namespace {
const TfTpu_BaseFn kEmptyBaseFn{};
const TfTpu_OpsApiFn kEmptyOpsApiFn{};
const TfTpu_ProfilerApiFn kEmptyProfilerApiFn{};
std::atomic<const TfTpu_BaseFn*> g_base_api_fn_ptr{&kEmptyBaseFn};
std::atomic<const TfTpu_OpsApiFn*> g_ops_api_fn_ptr{&kEmptyOpsApiFn};
std::atomic<const TfTpu_ProfilerApiFn*> g_profiler_api_fn_ptr{
    &kEmptyProfilerApiFn};
}  // namespace

const TfTpu_BaseFn* BaseApiFn() {
  return g_base_api_fn_ptr.load(std::memory_order_acquire);
}

void SetBaseApiFn(const TfTpu_BaseFn* fn) {
  g_base_api_fn_ptr.store(fn, std::memory_order_release);
}

const TfTpu_OpsApiFn* OpsApiFn() {
  return g_ops_api_fn_ptr.load(std::memory_order_acquire);
}

void SetOpsApiFn(const TfTpu_OpsApiFn* fn) {
  g_ops_api_fn_ptr.store(fn, std::memory_order_release);
}

const TfTpu_ProfilerApiFn* ProfilerApiFn() {
  return g_profiler_api_fn_ptr.load(std::memory_order_acquire);
}

void SetProfilerApiFn(const TfTpu_ProfilerApiFn* fn) {
  g_profiler_api_fn_ptr.store(fn, std::memory_order_release);
}

}  // namespace tpu
}  // namespace stream_executor
