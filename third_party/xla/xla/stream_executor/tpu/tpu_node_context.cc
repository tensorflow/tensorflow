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

#include "absl/status/status.h"
#include "xla/service/backend.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {
namespace tpu {

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
  static xla::Backend* backend =
      xla::Backend::CreateBackend(
          xla::BackendOptions().set_platform(platform()))
          .value()
          .release();
  return backend;
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
