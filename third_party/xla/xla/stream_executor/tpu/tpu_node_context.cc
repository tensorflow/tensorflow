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

#include "xla/stream_executor/tpu/tpu_node_context.h"

#include <memory>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/backend.h"
#include "xla/service/stream_pool.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {
namespace tpu {

namespace {
ABSL_CONST_INIT absl::Mutex backend_mutex(absl::kConstInit);
static xla::Backend* tpu_backend ABSL_GUARDED_BY(backend_mutex) = nullptr;

void ResetBackend() {
  absl::MutexLock lock(backend_mutex);
  delete tpu_backend;
  tpu_backend = nullptr;
}
}  // namespace

/*static*/
absl::StatusOr<std::unique_ptr<TpuNodeContext>> TpuNodeContext::Create(
    int device_ordinal) {
  StatusHelper status;
  XLA_TpuNodeContext* node_context =
      stream_executor::tpu::OpsApiFn()->TpuNodeContext_CreateFn(
          device_ordinal, status.c_status);
  if (!status.status().ok()) {
    // TpuNodeContext_CreateFn allocates a new XLA_TpuNodeContext regardless of
    // status. It needs to be freed if it's not given to a TpuNodeContext below.
    stream_executor::tpu::OpsApiFn()->TpuNodeContext_FreeFn(node_context);
    return status.status();
  }
  return std::make_unique<TpuNodeContext>(device_ordinal, node_context);
}

TpuNodeContext::~TpuNodeContext() {
  stream_executor::tpu::OpsApiFn()->TpuNodeContext_FreeFn(node_context_);
}

/* static */
absl::Status TpuNodeContext::CloseTpuHost() {
  ResetBackend();
  StatusHelper status;
  stream_executor::tpu::OpsApiFn()->TpuNodeContext_CloseTpuHostFn(
      status.c_status);
  return status.status();
}

/* static */
absl::Status TpuNodeContext::Initialize(int device_ordinal) {
  StatusHelper status;
  stream_executor::tpu::OpsApiFn()->TpuNodeContext_InitializeFn(
      device_ordinal, status.c_status);
  return status.status();
}

/* static */
TpuPlatformInterface* TpuNodeContext::platform() {
  return TpuPlatformInterface::GetRegisteredPlatform();
}

int TpuNodeContext::device_ordinal() const { return device_ordinal_; }

xla::Backend* TpuNodeContext::backend() const {
  absl::MutexLock lock(backend_mutex);
  if (tpu_backend == nullptr) {
    auto backend_or = xla::Backend::CreateBackend(
        xla::BackendOptions().set_platform(platform()));
    CHECK_OK(backend_or.status());
    tpu_backend = backend_or.value().release();
  }
  return tpu_backend;
}

stream_executor::StreamExecutor* TpuNodeContext::stream_executor() const {
  return backend()->stream_executor(device_ordinal_).value();
}

bool TpuNodeContext::CompactionSupported(int device_ordinal) const {
  return stream_executor::tpu::OpsApiFn()->TpuNodeContext_CompactionSupportedFn(
      device_ordinal);
}

}  // namespace tpu
}  // namespace tensorflow
