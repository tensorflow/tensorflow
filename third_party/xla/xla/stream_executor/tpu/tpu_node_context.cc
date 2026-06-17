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
#include <utility>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/backend.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tpu/tpu_platform_interface.h"
#include "xla/tpu/status_helper.h"
#include "xla/tpu/tpu_api.h"
#include "xla/tpu/tpu_ops_c_api.h"

namespace tensorflow {
namespace tpu {

namespace {
absl::Mutex backend_mutex(absl::kConstInit);
absl::NoDestructor<std::unique_ptr<xla::Backend>> backend_ptr
    ABSL_GUARDED_BY(backend_mutex);
}  // namespace

/*static*/
absl::StatusOr<std::unique_ptr<TpuNodeContext>> TpuNodeContext::Create(
    int device_ordinal) {
  StatusHelper status;
  XLA_TpuNodeContext* node_context =
      stream_executor::tpu::OpsApiFn()->TpuNodeContext_CreateFn(
          device_ordinal, status.c_status);
  if (!status.ok()) {
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
  {
    absl::MutexLock lock(backend_mutex);
    backend_ptr->reset();
  }
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
  if (*backend_ptr == nullptr) {
    absl::StatusOr<std::unique_ptr<xla::Backend>> backend_or =
        xla::Backend::CreateBackend(
            xla::BackendOptions().set_platform(platform()));
    CHECK_OK(backend_or.status());
    *backend_ptr = std::move(backend_or.value());
  }
  return backend_ptr->get();
}

stream_executor::StreamExecutor* TpuNodeContext::stream_executor() const {
  absl::StatusOr<stream_executor::StreamExecutor*> se_or =
      backend()->stream_executor(device_ordinal_);
  CHECK_OK(se_or.status());
  return se_or.value();
}

bool TpuNodeContext::CompactionSupported() const {
  return stream_executor::tpu::OpsApiFn()->TpuNodeContext_CompactionSupportedFn(
      device_ordinal_);
}

}  // namespace tpu
}  // namespace tensorflow
