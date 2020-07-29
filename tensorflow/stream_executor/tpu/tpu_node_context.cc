/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/stream_executor/tpu/tpu_node_context.h"

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_node_context_c_api.h"

namespace tensorflow {
namespace tpu {

using stream_executor::port::Status;
using stream_executor::port::StatusOr;

/*static*/
StatusOr<std::unique_ptr<TpuNodeContext>> TpuNodeContext::Create(
    int device_ordinal) {
  StatusHelper status;
  XLA_TpuNodeContext* node_context =
      tpu::NodeContextApiFn()->TpuNodeContext_CreateFn(device_ordinal,
                                                       status.c_status);
  if (!status.status().ok()) {
    tpu::NodeContextApiFn()->TpuNodeContext_FreeFn(node_context);
    return status.status();
  }
  return std::make_unique<TpuNodeContext>(device_ordinal, node_context);
}

TpuNodeContext::~TpuNodeContext() {
  tpu::NodeContextApiFn()->TpuNodeContext_FreeFn(node_context_);
}

/* static */
Status TpuNodeContext::Initialize(int device_ordinal) {
  StatusHelper status;
  TpuNodeContext_Initialize(device_ordinal, status.c_status);
  return status.status();
}

/* static */
Status TpuNodeContext::StopChipHeartbeats() {
  StatusHelper status;
  tpu::NodeContextApiFn()->TpuNodeContext_StopChipHeartbeatsFn(status.c_status);
  return status.status();
}

/* static */
Status TpuNodeContext::CloseTpuHost() {
  StatusHelper status;
  tpu::NodeContextApiFn()->TpuNodeContext_CloseTpuHostFn(status.c_status);
  return status.status();
}

/* static */
tensorflow::tpu::TpuPlatformInterface* TpuNodeContext::platform() {
  return TpuPlatformInterface::GetRegisteredPlatform();
}

/* static */
stream_executor::DeviceMemoryAllocator* TpuNodeContext::memory_allocator() {
  static stream_executor::StreamExecutorMemoryAllocator* memory_allocator =
      new stream_executor::StreamExecutorMemoryAllocator(
          platform(),
          xla::PlatformUtil::GetStreamExecutors(platform()).ValueOrDie());
  return memory_allocator;
}

/* static */
xla::Backend* TpuNodeContext::backend() {
  static xla::Backend* backend =
      xla::Backend::CreateBackend(
          xla::BackendOptions().set_platform(platform()))
          .ValueOrDie()
          .release();
  return backend;
}

/* static */
StatusOr<xla::StreamPool::Ptr> TpuNodeContext::BorrowStream(
    int device_ordinal) {
  return backend()->BorrowStream(device_ordinal);
}

/* static */
StatusOr<xla::StreamPool::Ptr> TpuNodeContext::BorrowStream(
    stream_executor::StreamExecutor* executor) {
  return backend()->BorrowStream(executor);
}

/* static */
xla::TransferManager* TpuNodeContext::transfer_manager() {
  return xla::TransferManager::GetForPlatform(platform()).ValueOrDie();
}

}  // namespace tpu
}  // namespace tensorflow
