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

#include "tensorflow/compiler/xla/service/transfer_manager.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace se = ::perftools::gputools;

namespace xla {

/* static */ tensorflow::mutex*
TransferManager::platform_transfer_manager_mutex() {
  static tensorflow::mutex* m = new tensorflow::mutex;
  return m;
}

/* static */ std::map<perftools::gputools::Platform::Id,
                      TransferManager::State>*
TransferManager::GetPlatformTransferManagers() {
  static auto* r =
      new std::map<perftools::gputools::Platform::Id, TransferManager::State>;
  return r;
}

/* static */ void TransferManager::RegisterTransferManager(
    se::Platform::Id platform_id,
    TransferManagerCreationFunction creation_function) {
  tensorflow::mutex_lock lock(
      *TransferManager::platform_transfer_manager_mutex());
  auto* managers = GetPlatformTransferManagers();
  CHECK(managers->find(platform_id) == managers->end());
  (*managers)[platform_id].creation_function = creation_function;
}

/* static */ StatusOr<TransferManager*> TransferManager::GetForPlatform(
    const se::Platform* platform) {
  tensorflow::mutex_lock lock(
      *TransferManager::platform_transfer_manager_mutex());
  auto* managers = GetPlatformTransferManagers();

  auto it = managers->find(platform->id());
  if (it == managers->end()) {
    return NotFound(
        "could not find registered transfer manager for platform %s -- check "
        "target linkage",
        platform->Name().c_str());
  }

  if (it->second.manager == nullptr) {
    // Lazily create the transfer manager the first time it is needed
    it->second.manager = (*it->second.creation_function)();
  }

  return it->second.manager.get();
}

Status TransferManager::TransferBufferFromDevice(
    se::StreamExecutor* executor, const se::DeviceMemoryBase& source,
    int64 size, void* destination) {
  if (source.size() < size) {
    return FailedPrecondition(
        "Source allocation on device not large enough for data tranfer: "
        "%lld < %lld",
        source.size(), size);
  }
  auto copy_status = executor->SynchronousMemcpyD2H(source, size, destination);
  if (!copy_status.ok()) {
    return AddStatus(
        Status(static_cast<tensorflow::error::Code>(copy_status.code()),
               copy_status.error_message()),
        "failed transfer from device to buffer");
  }
  return Status::OK();
}

Status TransferManager::TransferBufferToDevice(
    se::StreamExecutor* executor, int64 size, const void* source,
    se::DeviceMemoryBase* destination) {
  if (destination->size() < size) {
    return FailedPrecondition(
        "Destination allocation on device not large enough for data tranfer: "
        "%lld < %lld",
        destination->size(), size);
  }
  auto copy_status = executor->SynchronousMemcpyH2D(source, size, destination);
  if (!copy_status.ok()) {
    return AddStatus(
        Status(static_cast<tensorflow::error::Code>(copy_status.code()),
               copy_status.error_message()),
        "failed transfer of buffer to device");
  }
  return Status::OK();
}

StatusOr<std::set<se::DeviceMemoryBase>>
TransferManager::GatherBufferPointersFromTuple(
    se::StreamExecutor* executor, const se::DeviceMemoryBase& source,
    const Shape& shape) {
  TF_RET_CHECK(ShapeUtil::IsTuple(shape));

  std::set<se::DeviceMemoryBase> buffer_pointers;
  buffer_pointers.insert(source);

  TF_ASSIGN_OR_RETURN(std::vector<se::DeviceMemoryBase> tuple_elements,
                      ShallowCopyTupleFromDevice(executor, source, shape));
  for (auto i = 0; i < tuple_elements.size(); ++i) {
    const Shape& element_shape = shape.tuple_shapes(i);
    if (ShapeUtil::IsTuple(element_shape)) {
      TF_ASSIGN_OR_RETURN(
          std::set<se::DeviceMemoryBase> buffer_pointers_in_element,
          GatherBufferPointersFromTuple(executor, tuple_elements[i],
                                        element_shape));
      buffer_pointers.insert(buffer_pointers_in_element.begin(),
                             buffer_pointers_in_element.end());
    } else {
      buffer_pointers.insert(tuple_elements[i]);
    }
  }
  return std::move(buffer_pointers);
}

}  // namespace xla
