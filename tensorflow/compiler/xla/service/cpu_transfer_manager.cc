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

#include "tensorflow/compiler/xla/service/cpu_transfer_manager.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {

namespace {

class CpuInfeedBuffer : public cpu::runtime::InfeedBuffer {
 public:
  explicit CpuInfeedBuffer(int32 length)
      : length_(length),
        buffer_(new char[length]),
        device_memory_(buffer_, length_) {}
  ~CpuInfeedBuffer() override { delete[] buffer_; }

  int32 length() override { return length_; }
  void* data() override { return buffer_; }
  void Done() override { delete this; }

  se::DeviceMemoryBase* device_memory() { return &device_memory_; }

 private:
  int32 length_;
  char* buffer_;
  se::DeviceMemoryBase device_memory_;
};

}  // namespace

CpuTransferManager::CpuTransferManager()
    : GenericTransferManager(se::host::kHostPlatformId) {}

Status CpuTransferManager::TransferLiteralToInfeed(se::StreamExecutor* executor,
                                                   const Literal& literal) {
  const Shape& shape = literal.shape();
  VLOG(2) << "Transferring literal to infeed with shape: "
          << ShapeUtil::HumanString(shape);

  if (!ShapeUtil::IsTuple(shape)) {
    int64 size = GetByteSizeRequirement(shape);
    return TransferBufferToInfeed(executor, size, literal.InternalData());
  }

  if (ShapeUtil::IsNestedTuple(shape)) {
    return Unimplemented(
        "Infeed with a nested tuple shape is not supported: %s",
        ShapeUtil::HumanString(literal.shape()).c_str());
  }

  // For a tuple, we transfer each of its elements to the device and
  // enqueue the resulting destination device addresses with the
  // infeed manager.
  std::vector<cpu::runtime::InfeedBuffer*> buffers;
  buffers.reserve(literal.tuple_literals_size());
  auto cleanup = tensorflow::gtl::MakeCleanup([buffers]() {
    for (cpu::runtime::InfeedBuffer* b : buffers) {
      b->Done();
    }
  });

  for (const auto& tuple_element : literal.tuple_literals()) {
    const Shape& tuple_element_shape = tuple_element.shape();
    int64 tuple_element_size = GetByteSizeRequirement(tuple_element_shape);
    TF_ASSIGN_OR_RETURN(
        cpu::runtime::InfeedBuffer * buffer,
        TransferBufferToInfeedInternal(executor, tuple_element_size,
                                       tuple_element.InternalData()));
    buffers.push_back(buffer);
  }

  cpu::runtime::InfeedManager* infeed_manager =
      cpu::runtime::GetInfeedManager();
  infeed_manager->EnqueueBuffers(buffers);

  cleanup.release();
  return Status::OK();
}

Status CpuTransferManager::TransferBufferToInfeed(se::StreamExecutor* executor,
                                                  int64 size,
                                                  const void* source) {
  TF_ASSIGN_OR_RETURN(cpu::runtime::InfeedBuffer * buffer,
                      TransferBufferToInfeedInternal(executor, size, source));

  cpu::runtime::InfeedManager* infeed_manager =
      cpu::runtime::GetInfeedManager();
  infeed_manager->EnqueueBuffers({buffer});

  return Status::OK();
}

StatusOr<cpu::runtime::InfeedBuffer*>
CpuTransferManager::TransferBufferToInfeedInternal(se::StreamExecutor* executor,
                                                   int64 size,
                                                   const void* source) {
  if (size > std::numeric_limits<int32>::max()) {
    return InvalidArgument("Infeed shape is too large: needs %lld bytes", size);
  }

  if (size == 0) {
    return InvalidArgument("Infeed shape needs 0 bytes");
  }

  int32 size_32 = static_cast<int32>(size);
  CpuInfeedBuffer* queued_buffer = new CpuInfeedBuffer(size_32);
  Status s =
      TransferBufferToDevice(executor, /*size=*/size,
                             /*source=*/source, queued_buffer->device_memory());

  if (!s.ok()) {
    queued_buffer->Done();
    return s;
  }
  return queued_buffer;
}

}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreateCpuTransferManager() {
  return xla::MakeUnique<xla::CpuTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(se::host::kHostPlatformId,
                                                &CreateCpuTransferManager);
  return true;
}
static bool module_initialized = InitModule();
