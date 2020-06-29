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

#include "tensorflow/stream_executor/tpu/tpu_transfer_manager.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"

namespace tensorflow {

using Status = stream_executor::port::Status;

TpuTransferManager::TpuTransferManager() {
  manager_ = tpu::ExecutorApiFn()->TpuTransferManager_NewFn();
}

TpuTransferManager::~TpuTransferManager() {
  tpu::ExecutorApiFn()->TpuTransferManager_FreeFn(manager_);
}

stream_executor::Platform::Id TpuTransferManager::PlatformId() const {
  return TpuPlatform::kId;
}

xla::Shape TpuTransferManager::HostShapeToDeviceShape(
    const xla::Shape& host_shape) const {
  XLA_Shape c_host_shape;
  XLA_Shape c_device_shape;

  TpuConversions::XlaShapeToCShape(host_shape, &c_host_shape);

  tpu::ExecutorApiFn()->TpuTransferManager_HostShapeToDeviceShapeFn(
      manager_, &c_host_shape, &c_device_shape);
  xla::Shape device_shape = TpuConversions::CShapeToXlaShape(&c_device_shape);
  TpuConversions::CShapeCleanup(&c_host_shape);
  TpuConversions::CShapeCleanup(&c_device_shape);
  return device_shape;
}

Status TpuTransferManager::TransferLiteralToDeviceAsync(
    stream_executor::Stream* stream, const xla::LiteralSlice& literal,
    const xla::ShapedBuffer& device_buffer,
    const TransferMetadata* transfer_metadata) {
  StatusHelper status;

  XLA_Literal c_literal;
  TpuConversions::XLALiteralToCLiteral(literal, &c_literal);

  XLA_ShapedBuffer c_device_buffer;
  TpuConversions::XLAShapedBufferToCShapedBuffer(device_buffer,
                                                 &c_device_buffer);

  tpu::ExecutorApiFn()->TpuTransferManager_TransferLiteralToDeviceAsyncFn(
      manager_,
      TpuPlatform::GetRegisteredPlatform()->stream_map()->at(
          stream->implementation()),
      &c_literal, &c_device_buffer, status.c_status);
  TpuConversions::CShapedBufferCleanup(&c_device_buffer);
  TpuConversions::CLiteralCleanup(&c_literal);
  return status.status();
}

struct TransferFromDeviceState {
  std::atomic<int64_t> remaining_transfers;
  StatusHelper status_helper;
  std::function<void(Status)> done;

  void TransferFinished(SE_Status* status) {
    if (!TpuStatus_Ok(status) && TpuStatus_Ok(status_helper.c_status)) {
      status_helper.c_status = status;
    } else {
      TpuStatus_Free(status);
    }

    if (--remaining_transfers == 0) {
      done(status_helper.status());
      delete this;
    }
  }
};

void TransferLiteralFromDeviceTrampoline(void* ctx, SE_Status* status) {
  reinterpret_cast<TransferFromDeviceState*>(ctx)->TransferFinished(status);
}

void TpuTransferManager::TransferLiteralFromDevice(
    stream_executor::Stream* stream, const xla::ShapedBuffer& device_buffer,
    xla::MutableBorrowingLiteral literal, std::function<void(Status)> done,
    const TransferMetadata* transfer_metadata) {
  TransferFromDeviceState* state = new TransferFromDeviceState;
  state->remaining_transfers = 1;
  state->done = done;
  XLA_ShapedBuffer c_device_buffer;
  TpuConversions::XLAShapedBufferToCShapedBuffer(device_buffer,
                                                 &c_device_buffer);
  XLA_Literal c_literal;
  TpuConversions::XLALiteralToCLiteral(literal, &c_literal);

  tpu::ExecutorApiFn()->TpuTransferManager_TransferLiteralFromDeviceFn(
      manager_,
      TpuPlatform::GetRegisteredPlatform()->stream_map()->at(
          stream->implementation()),
      &c_device_buffer, &c_literal, TransferLiteralFromDeviceTrampoline, state);
  TpuConversions::CShapedBufferCleanup(&c_device_buffer);
  TpuConversions::CLiteralCleanup(&c_literal);
}

int64 TpuTransferManager::GetByteSizeRequirement(
    const xla::Shape& shape) const {
  XLA_Shape c_shape;
  TpuConversions::XlaShapeToCShape(shape, &c_shape);

  int64 size_in_bytes =
      tpu::ExecutorApiFn()->TpuTransferManager_GetByteSizeRequirementFn(
          manager_, &c_shape);

  TpuConversions::CShapeCleanup(&c_shape);
  return size_in_bytes;
}

Status TpuTransferManager::WriteSingleTupleIndexTable(
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
  TpuConversions::XlaShapeToCShape(shape, &c_shape);
  SE_DeviceMemoryBase region_base{region->opaque(), region->size(),
                                  region->payload()};
  StatusHelper status;

  tpu::ExecutorApiFn()->TpuTransferManager_WriteSingleTupleIndexTableFn(
      manager_,
      TpuPlatform::GetRegisteredPlatform()->stream_map()->at(
          stream->implementation()),
      elements_bases, elements.size(), &c_shape, &region_base, status.c_status);

  delete[] elements_bases;
  TpuConversions::CShapeCleanup(&c_shape);
  return status.status();
}

}  // namespace tensorflow
