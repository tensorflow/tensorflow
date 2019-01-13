/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/transfer_manager.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"

#include <memory>

#include "tensorflow/compiler/xla/service/transfer_manager.h"

namespace xla {
namespace poplarplugin {

class PoplarInfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  explicit PoplarInfeedBuffer(int32 length)
      : length_(length),
        buffer_(new char[length]),
        device_memory_(buffer_, length_) {}
  ~PoplarInfeedBuffer() override { delete[] buffer_; }

  int32 length() override { return length_; }
  void* data() override { return buffer_; }
  void Done(StatusOr<Shape> /*shape*/) override { delete this; }

  se::DeviceMemoryBase* device_memory() { return &device_memory_; }

 private:
  int32 length_;
  char* buffer_;
  se::DeviceMemoryBase device_memory_;
};

PoplarTransferManager::PoplarTransferManager()
    : GenericTransferManager(kPoplarPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

Status PoplarTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  const Shape& shape = literal.shape();
  if (shape.IsTuple()) {
    return Unimplemented("Transferring tuples is not supported yet");
  }

  int64 size = GetByteSizeRequirement(shape);
  return TransferBufferToInfeed(executor, size, literal.untyped_data());
}

Status PoplarTransferManager::TransferBufferToInfeed(
    se::StreamExecutor* executor, int64 size, const void* source) {
  if (size > std::numeric_limits<int32>::max()) {
    return InvalidArgument("Infeed shape is too large: needs %d bytes", size);
  }

  if (size <= 0) {
    return InvalidArgument("Infeed shape must have positive size; got %d",
                           size);
  }

  int32 size_32 = static_cast<int32>(size);

  PoplarInfeedBuffer* queued_buffer = new PoplarInfeedBuffer(size_32);
  std::memcpy(queued_buffer->data(), source, size);

  cpu::runtime::XfeedManager* xfeed_manager =
      xla::poplarplugin::GetXfeedManager(executor->device_ordinal());
  xfeed_manager->infeed()->EnqueueBuffersAtomically({queued_buffer});

  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreatePoplarTransferManager() {
  return absl::make_unique<xla::poplarplugin::PoplarTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      xla::poplarplugin::kPoplarPlatformId, &CreatePoplarTransferManager);
  return true;
}

static bool module_initialized = InitModule();
