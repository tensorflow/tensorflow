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

#include "xla/stream_executor/tpu/tpu_transfer_manager.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/noncopyable_buffer.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_executor.h"
#include "xla/stream_executor/tpu/tpu_executor_api.h"
#include "xla/stream_executor/tpu/tpu_executor_c_api.h"
#include "xla/stream_executor/tpu/tpu_platform.h"
#include "xla/stream_executor/tpu/tpu_platform_id.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace tensorflow {
namespace tpu {

TpuTransferManager::TpuTransferManager() {
  manager_ = stream_executor::tpu::ExecutorApiFn()->TpuTransferManager_NewFn();
}

TpuTransferManager::~TpuTransferManager() {
  stream_executor::tpu::ExecutorApiFn()->TpuTransferManager_FreeFn(manager_);
}

stream_executor::Platform::Id TpuTransferManager::PlatformId() const {
  return GetTpuPlatformId();
}

xla::Shape TpuTransferManager::HostShapeToDeviceShape(
    const xla::Shape& host_shape) const {
  XLA_Shape c_host_shape;
  XLA_Shape c_device_shape;

  ApiConverter::ToC(host_shape, &c_host_shape);

  stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_HostShapeToDeviceShapeFn(manager_, &c_host_shape,
                                                    &c_device_shape);
  xla::Shape device_shape = ApiConverter::FromC(&c_device_shape);
  ApiConverter::Destroy(&c_host_shape);
  ApiConverter::Destroy(&c_device_shape);
  return device_shape;
}

absl::Status TpuTransferManager::TransferLiteralToDeviceAsync(
    stream_executor::Stream* stream, const xla::LiteralSlice& literal,
    const xla::ShapedBuffer& device_buffer,
    const TransferMetadata* transfer_metadata) {
  StatusHelper status;

  XLA_Literal c_literal;
  ApiConverter::ToC(literal, &c_literal);

  XLA_ShapedBuffer c_device_buffer;
  ApiConverter::ToC(device_buffer, &c_device_buffer);

  stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_TransferLiteralToDeviceAsyncFn(
          manager_, TpuPlatform::GetRegisteredPlatform()->LookupStream(stream),
          &c_literal, &c_device_buffer, status.c_status);
  ApiConverter::Destroy(&c_device_buffer);
  ApiConverter::Destroy(&c_literal);
  return status.status();
}

absl::Status TpuTransferManager::TransferLiteralToInfeed(
    stream_executor::StreamExecutor* executor,
    const xla::LiteralSlice& literal) {
  StatusHelper status;
  XLA_Literal c_literal;
  ApiConverter::ToC(literal, &c_literal);
  auto* tpu_executor =
      static_cast<stream_executor::tpu::TpuExecutor*>(executor);

  stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_TransferLiteralToInfeedFn(
          manager_, tpu_executor->se_executor(), &c_literal, status.c_status);

  ApiConverter::Destroy(&c_literal);

  return status.status();
}

absl::Status TpuTransferManager::TransferBuffersToInfeed(
    se::StreamExecutor* executor,
    const std::deque<tensorflow::tpu::NoncopyableBuffer>& buffers) {
  StatusHelper status;
  auto* tpu_executor =
      static_cast<stream_executor::tpu::TpuExecutor*>(executor);

  std::vector<int64_t> buffers_size;
  std::vector<uint32_t*> buffers_array;

  buffers_size.reserve(buffers.size());
  buffers_array.reserve(buffers.size());

  for (int64_t i = 0; i < buffers.size(); ++i) {
    absl::Span<const uint32_t> span = buffers[i].const_data<uint32_t>();
    buffers_array.push_back(const_cast<uint32_t*>(span.data()));
    buffers_size.push_back(span.size());
  }

  stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_TransferBuffersToInfeedFn(
          manager_, tpu_executor->se_executor(), buffers_array.data(),
          buffers_size.data(), buffers_size.size(), status.c_status);
  return status.status();
}

absl::Status TpuTransferManager::TransferLiteralFromOutfeed(
    stream_executor::StreamExecutor* executor,
    xla::MutableBorrowingLiteral literal) {
  StatusHelper status;
  XLA_Shape c_shape;
  XLA_Literal c_literal;
  auto* tpu_executor =
      static_cast<stream_executor::tpu::TpuExecutor*>(executor);

  ApiConverter::ToC(literal.shape(), &c_shape);
  ApiConverter::ToC(literal, &c_literal);

  stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_TransferLiteralFromOutfeedFn(
          manager_, tpu_executor->se_executor(), &c_shape, &c_literal,
          status.c_status);

  ApiConverter::Destroy(&c_shape);
  ApiConverter::Destroy(&c_literal);

  return status.status();
}

absl::Status TpuTransferManager::ResetDevices(
    absl::Span<stream_executor::StreamExecutor* const> executor) {
  StatusHelper status;
  std::vector<SE_StreamExecutor*> se;
  se.reserve(executor.size());
  for (int64_t i = 0; i < executor.size(); ++i) {
    se.push_back(static_cast<stream_executor::tpu::TpuExecutor*>(executor[i])
                     ->se_executor());
  }

  stream_executor::tpu::ExecutorApiFn()->TpuTransferManager_ResetDevicesFn(
      manager_, se.data(), executor.size(), status.c_status);
  return status.status();
}

struct TransferFromDeviceState {
  std::atomic<int64_t> remaining_transfers;
  TF_Status* overall_status = stream_executor::tpu::ExecutorApiFn()
                                  ->TpuStatus_NewFn();  // OK or the first error
  std::function<void(absl::Status)> done;

  void TransferFinished(TF_Status* status) {
    if (!stream_executor::tpu::ExecutorApiFn()->TpuStatus_OkFn(status) &&
        stream_executor::tpu::ExecutorApiFn()->TpuStatus_OkFn(overall_status)) {
      std::swap(overall_status, status);
    }
    stream_executor::tpu::ExecutorApiFn()->TpuStatus_FreeFn(status);

    if (--remaining_transfers == 0) {
      done(StatusHelper::FromC(overall_status));
      stream_executor::tpu::ExecutorApiFn()->TpuStatus_FreeFn(overall_status);
      delete this;
    }
  }
};

void TransferLiteralFromDeviceTrampoline(void* ctx, TF_Status* status) {
  reinterpret_cast<TransferFromDeviceState*>(ctx)->TransferFinished(status);
}

void TpuTransferManager::TransferLiteralFromDevice(
    stream_executor::Stream* stream, const xla::ShapedBuffer& device_buffer,
    xla::MutableBorrowingLiteral literal,
    std::function<void(absl::Status)> done,
    const TransferMetadata* transfer_metadata) {
  TransferFromDeviceState* state = new TransferFromDeviceState;
  state->remaining_transfers = 1;
  state->done = done;
  XLA_ShapedBuffer c_device_buffer;
  ApiConverter::ToC(device_buffer, &c_device_buffer);
  XLA_Literal c_literal;
  ApiConverter::ToC(literal, &c_literal);

  stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_TransferLiteralFromDeviceFn(
          manager_, TpuPlatform::GetRegisteredPlatform()->LookupStream(stream),
          &c_device_buffer, &c_literal, TransferLiteralFromDeviceTrampoline,
          state);
  ApiConverter::Destroy(&c_device_buffer);
  ApiConverter::Destroy(&c_literal);
}

int64_t TpuTransferManager::GetByteSizeRequirement(
    const xla::Shape& shape) const {
  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);

  int64_t size_in_bytes =
      stream_executor::tpu::ExecutorApiFn()
          ->TpuTransferManager_GetByteSizeRequirementFn(manager_, &c_shape);

  ApiConverter::Destroy(&c_shape);
  return size_in_bytes;
}

absl::StatusOr<xla::Shape> TpuTransferManager::ChooseCompactLayoutForShape(
    const xla::Shape& host_shape) const {
  XLA_Shape c_host_shape;
  ApiConverter::ToC(host_shape, &c_host_shape);
  XLA_Shape c_output;
  StatusHelper status;
  stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_ChooseCompactLayoutForShapeFn(
          manager_, &c_host_shape, &c_output, status.c_status);
  // TODO(skyewm): use a scoped version of XLA_Shape
  ApiConverter::Destroy(&c_host_shape);
  if (!status.status().ok()) {
    ApiConverter::Destroy(&c_output);
    return status.status();
  }
  xla::Shape output = ApiConverter::FromC(&c_output);
  ApiConverter::Destroy(&c_output);
  return output;
}

bool TpuTransferManager::CanShapedBufferBeAccessedNow(
    stream_executor::StreamExecutor* executor,
    const xla::ShapedBuffer& device_buffer) const {
  auto* tpu_executor = down_cast<stream_executor::tpu::TpuExecutor*>(executor);
  XLA_ShapedBuffer c_device_buffer;
  ApiConverter::ToC(device_buffer, &c_device_buffer);
  absl::Cleanup cleanup = [&c_device_buffer]() {
    ApiConverter::Destroy(&c_device_buffer);
  };
  return stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_CanShapedBufferBeAccessedNowFn(
          manager_, tpu_executor->se_executor(), &c_device_buffer);
}

bool TpuTransferManager::CanBufferBeAccessedNow(
    se::StreamExecutor* executor,
    const se::DeviceMemoryBase& device_buffer) const {
  auto* tpu_executor = down_cast<stream_executor::tpu::TpuExecutor*>(executor);
  SE_DeviceMemoryBase c_device_buffer{const_cast<void*>(device_buffer.opaque()),
                                      device_buffer.size(),
                                      device_buffer.payload()};
  return stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_CanBufferBeAccessedNowFn(
          manager_, tpu_executor->se_executor(), &c_device_buffer);
}

absl::Status TpuTransferManager::WriteSingleTupleIndexTable(
    stream_executor::Stream* stream,
    absl::Span<const stream_executor::DeviceMemoryBase> elements,
    const xla::Shape& shape, stream_executor::DeviceMemoryBase* region) {
  CHECK_GT(elements.size(), 0);
  SE_DeviceMemoryBase* elements_bases =
      new SE_DeviceMemoryBase[elements.size()];
  for (int i = 0; i < elements.size(); i++) {
    elements_bases[i] =
        SE_DeviceMemoryBase{const_cast<void*>(elements[i].opaque()),
                            elements[i].size(), elements[i].payload()};
  }
  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);
  SE_DeviceMemoryBase region_base{region->opaque(), region->size(),
                                  region->payload()};
  StatusHelper status;

  stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_WriteSingleTupleIndexTableFn(
          manager_, TpuPlatform::GetRegisteredPlatform()->LookupStream(stream),
          elements_bases, elements.size(), &c_shape, &region_base,
          status.c_status);

  delete[] elements_bases;
  ApiConverter::Destroy(&c_shape);
  return status.status();
}

absl::Status TpuTransferManager::LinearizeToBuffers(
    const xla::LiteralSlice& literal, const xla::Shape& device_shape,
    std::deque<tensorflow::tpu::NoncopyableBuffer>* buffers) {
  XLA_Literal c_literal;
  ApiConverter::ToC(literal, &c_literal);
  XLA_Shape c_device_shape;
  ApiConverter::ToC(device_shape, &c_device_shape);

  char** buffers_array;
  int64_t* buffers_size;
  int64_t buffers_array_size;
  StatusHelper status;

  stream_executor::tpu::ExecutorApiFn()
      ->TpuTransferManager_LinearizeToBuffersFn(
          manager_, &c_literal, &c_device_shape, &buffers_array, &buffers_size,
          &buffers_array_size, status.c_status);

  for (int64_t i = 0; i < buffers_array_size; ++i) {
    tpu::NoncopyableBuffer buf(buffers_size[i]);
    memcpy(buf.mutable_data<uint8_t>().data(), buffers_array[i],
           buffers_size[i]);
    buffers->push_back(std::move(buf));
  }

  stream_executor::tpu::ExecutorApiFn()->TpuTransferManager_FreeBuffersFn(
      buffers_array, buffers_size, buffers_array_size);

  ApiConverter::Destroy(&c_device_shape);
  ApiConverter::Destroy(&c_literal);
  return status.status();
}

absl::Status TpuTransferManager::ReadDynamicShapes(
    se::Stream* stream, const xla::ShapedBuffer* device_buffer,
    xla::Shape* device_shape) {
  XLA_ShapedBuffer c_device_buffer;
  XLA_Shape c_device_shape;
  ApiConverter::ToC(*device_buffer, &c_device_buffer);
  ApiConverter::ToC(*device_shape, &c_device_shape);
  XLA_Shape c_updated_shape;
  StatusHelper status;
  stream_executor::tpu::ExecutorApiFn()->TpuTransferManager_ReadDynamicShapesFn(
      TpuPlatform::GetRegisteredPlatform()->LookupStream(stream),
      &c_device_buffer, c_device_shape, &c_updated_shape, status.c_status);
  ApiConverter::Destroy(&c_device_buffer);
  ApiConverter::Destroy(&c_device_shape);
  if (!status.ok()) {
    return status.status();
  }
  *device_shape = ApiConverter::FromC(&c_updated_shape);
  ApiConverter::Destroy(&c_updated_shape);
  return absl::OkStatus();
}

}  // namespace tpu
}  // namespace tensorflow
