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

#ifndef XLA_TPU_TPU_API_H_
#define XLA_TPU_TPU_API_H_

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "xla/tpu/libtftpu.h"
#include "xla/tpu/tpu_ops_c_api.h"
#include "xla/tpu/tpu_profiler_c_api.h"

namespace stream_executor {
namespace tpu {

// Exposing raw once-flags inside public headers is generally a design smell.
// However, because some initialization logic is defined within inlineable
// implementation header fragments (.inc files) that are included across
// multiple Translation Units (TUs), they must sync on the exact same
// synchronization state physically.
// Exposing getters returning pointers to static internal flags is used here
// instead of template-based callbacks (like absl::AnyInvocable) because it
// is ABI-stable across shared library boundaries (like dynamic loader builds)
// and avoids linker/mangling mismatches.
absl::once_flag* GetTpuExecutorInitOnceFlag();
absl::once_flag* GetTpuProfilerInitOnceFlag();
absl::once_flag* GetTpuOpsStructFnsOnceFlag();
absl::once_flag* GetTpuBaseApiInitOnceFlag();

absl::Status& GetTpuOpsStructFnsInitStatus();
absl::Status& GetTpuBaseApiInitStatus();
absl::Status& GetTpuExecutorInitStatus();

const TfTpu_BaseFn* BaseApiFn();
void SetBaseApiFn(const TfTpu_BaseFn* fn);

const TfTpu_OpsApiFn* OpsApiFn();
void SetOpsApiFn(const TfTpu_OpsApiFn* fn);

const TfTpu_ProfilerApiFn* ProfilerApiFn();
void SetProfilerApiFn(const TfTpu_ProfilerApiFn* fn);

}  // namespace tpu
}  // namespace stream_executor

#endif  // XLA_TPU_TPU_API_H_
