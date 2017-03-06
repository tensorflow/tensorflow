/* Copyright 2017 Graphcore Ltd
 */

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

#include "tensorflow/stream_executor/poplar/poplar_platform_id.h"
#include "tensorflow/compiler/poplar/driver/transfer_manager.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/infeed_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

#include <string>
#include <utility>
#include <vector>

namespace xla {
namespace poplarplugin {

PoplarTransferManager::PoplarTransferManager() {
}

se::Platform::Id PoplarTransferManager::PlatformId() const {
  return se::poplarplugin::kPoplarPlatformId;
}

Status PoplarTransferManager::TransferLiteralFromDevice(
        se::StreamExecutor* executor, const se::DeviceMemoryBase& source,
        const Shape& device_shape, const Shape& literal_shape, Literal* literal) {
  VLOG(1) << "transferring literal shape from device: "
            << ShapeUtil::HumanString(literal_shape)
            << "; device location: " << source.opaque();
  TF_RET_CHECK(ShapeUtil::Compatible(device_shape, literal_shape));

  // Tuples are a special case and contain one or more shapes inside of them to
  // an arbitrary nesting depth.
  if (device_shape.element_type() == TUPLE) {
    *literal->mutable_shape() = literal_shape;
    TF_ASSIGN_OR_RETURN(
            std::vector<se::DeviceMemoryBase> element_buffers,
            ShallowCopyTupleFromDevice(executor, source, device_shape));
    TF_RET_CHECK(element_buffers.size() ==
                 ShapeUtil::TupleElementCount(device_shape));
    for (int64 i = 0; i < element_buffers.size(); ++i) {
      const Shape& element_device_shape = device_shape.tuple_shapes(i);
      const Shape& element_literal_shape = literal_shape.tuple_shapes(i);
      Literal* element_literal = literal->add_tuple_literals();
      // Recursively call TransferFromDevice to copy over the data in the
      // element array.
      TF_RETURN_IF_ERROR(TransferLiteralFromDevice(
              executor, element_buffers[i], element_device_shape,
              element_literal_shape, element_literal));
    }
    return Status::OK();
  }

  *literal->mutable_shape() = device_shape;
  LiteralUtil::Reserve(ShapeUtil::ElementsIn(device_shape), literal);
  TF_RETURN_IF_ERROR(TransferBufferFromDevice(
          executor, source, ShapeUtil::ByteSizeOf(device_shape),
          LiteralUtil::MutableInternalData(literal)));
  if (!ShapeUtil::Equal(literal_shape, device_shape)) {
    literal->Swap(
            LiteralUtil::Relayout(*literal, literal_shape.layout()).get());
  }
  TF_RET_CHECK(ShapeUtil::Equal(literal_shape, literal->shape()));
  return Status::OK();
}

StatusOr<std::vector<se::DeviceMemoryBase>>
PoplarTransferManager::ShallowCopyTupleFromDevice(
        se::StreamExecutor* executor, const se::DeviceMemoryBase& source,
        const Shape& shape) {
  TF_RET_CHECK(ShapeUtil::IsTuple(shape));

  std::vector<void*> element_pointers(ShapeUtil::TupleElementCount(shape),
                                      nullptr);
  int64 tuple_size =
          ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  auto copy_status = executor->SynchronousMemcpyD2H(source, tuple_size,
                                                    element_pointers.data());
  if (!copy_status.ok()) {
    return AddStatus(
            Status(static_cast<tensorflow::error::Code>(copy_status.code()),
                   copy_status.error_message()),
            "failed transfer of tuple buffer " + ShapeUtil::HumanString(shape));
  }

  // Create a DeviceMemoryBase from each void* pointer.
  std::vector<se::DeviceMemoryBase> destination;
  for (int i = 0; i < element_pointers.size(); ++i) {
    if (element_pointers[i] == nullptr &&
        !ShapeUtil::HasZeroElements(shape.tuple_shapes(i))) {
      return FailedPrecondition("tuple contains nullptr at element %d", i);
    }
    int64 buffer_size = ShapeUtil::ByteSizeOf(shape.tuple_shapes(i),
            sizeof(void*));
    destination.emplace_back(element_pointers[i], buffer_size);
  }
  return std::move(destination);
}

Status PoplarTransferManager::TransferLiteralToDevice(
        se::StreamExecutor* executor, const Literal& literal,
        se::DeviceMemoryBase* destination) {
  const Shape& shape = literal.shape();
  VLOG(1) << "transferring literal shape to device: "
          << ShapeUtil::HumanString(shape)
          << "; device location: " << destination->opaque();

  if (ShapeUtil::IsTuple(literal.shape())) {
    std::vector<void*> tuple_elements_on_device;
    for (const Literal& tuple_element : literal.tuple_literals()) {
      se::DeviceMemoryBase allocation = executor->AllocateArray<uint8>(
              GetByteSizeRequirement(tuple_element.shape()));
      TF_RETURN_IF_ERROR(
              TransferLiteralToDevice(executor, tuple_element, &allocation));
      tuple_elements_on_device.push_back(allocation.opaque());
    }
    return TransferBufferToDevice(
            executor, tuple_elements_on_device.size() * sizeof(void*),
            tuple_elements_on_device.data(), destination);
  }

  return TransferBufferToDevice(
          executor, GetByteSizeRequirement(shape),
          LiteralUtil::InternalData(literal), destination);
}

Status
PoplarTransferManager::TransferLiteralToInfeed(se::StreamExecutor *executor,
                                               const Literal &literal) {
  const Shape &shape = literal.shape();
  VLOG(1) << "transferring literal shape to infeed: "
            << ShapeUtil::HumanString(shape);

  return Status::OK();
}

Status
PoplarTransferManager::TransferLiteralFromOutfeed(
        perftools::gputools::StreamExecutor* executor,
        const Shape& literal_shape,
        Literal* literal) {
  const Shape &shape = literal->shape();
  VLOG(1) << "transferring literal shape from outfeed: "
          << ShapeUtil::HumanString(shape);

  return Status::OK();
}


Status PoplarTransferManager::ResetDevices(
        tensorflow::gtl::ArraySlice<perftools::gputools::StreamExecutor*>
        executors) {
  return Unimplemented(
          "Device reset is not yet supported on CPU and GPU (b/30481585)");
}

int64 PoplarTransferManager::GetByteSizeRequirement(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

}  // namespace poplarplugin
}  // namespace xla

static xla::TransferManager* CreatePoplarTransferManager() {
  return new xla::poplarplugin::PoplarTransferManager();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(sep::kPoplarPlatformId,
                                                &CreatePoplarTransferManager);
  return true;
}
static bool module_initialized = InitModule();
