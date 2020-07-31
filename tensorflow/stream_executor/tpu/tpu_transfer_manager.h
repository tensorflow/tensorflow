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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_H_

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"

namespace tensorflow {

class TpuTransferManager : public xla::TransferManager {
 public:
  TpuTransferManager();
  ~TpuTransferManager() override;

  using Status = stream_executor::port::Status;
  template <typename T>
  using StatusOr = stream_executor::port::StatusOr<T>;

  stream_executor::Platform::Id PlatformId() const override;

  xla::Shape HostShapeToDeviceShape(
      const xla::Shape& host_shape) const override;

  Status TransferLiteralToDeviceAsync(
      stream_executor::Stream* stream, const xla::LiteralSlice& literal,
      const xla::ShapedBuffer& device_buffer,
      const TransferMetadata* transfer_metadata) override;

  void TransferLiteralFromDevice(
      stream_executor::Stream* stream, const xla::ShapedBuffer& device_buffer,
      xla::MutableBorrowingLiteral literal, std::function<void(Status)> done,
      const TransferMetadata* transfer_metadata) override;

  Status TransferLiteralToInfeed(stream_executor::StreamExecutor* executor,
                                 const xla::LiteralSlice& literal) override {
    LOG(FATAL) << "Not yet implemented";
  }

  Status TransferLiteralFromOutfeed(
      stream_executor::StreamExecutor* executor,
      const xla::Shape& literal_shape,
      xla::MutableBorrowingLiteral literal) override {
    LOG(FATAL) << "Not yet implemented";
  }

  Status ResetDevices(
      absl::Span<stream_executor::StreamExecutor* const> executor) override {
    LOG(FATAL) << "Not yet implemented";
  }

  int64 GetByteSizeRequirement(const xla::Shape& shape) const override;

  Status WriteSingleTupleIndexTable(
      stream_executor::Stream* stream,
      absl::Span<const stream_executor::DeviceMemoryBase> elements,
      const xla::Shape& shape,
      stream_executor::DeviceMemoryBase* region) override;

 private:
  XLA_TransferManager* manager_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_H_
