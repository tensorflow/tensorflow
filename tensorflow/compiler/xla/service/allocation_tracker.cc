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

#include "tensorflow/compiler/xla/service/allocation_tracker.h"

#include <utility>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {

AllocationTracker::AllocationTracker() : next_handle_(1) {}

GlobalDataHandle AllocationTracker::Register(Backend* backend,
                                             int device_ordinal,
                                             se::DeviceMemoryBase device_memory,
                                             const Shape& shape,
                                             const string& tag) {
  tensorflow::mutex_lock lock(allocation_mutex_);
  VLOG(2) << "Register";
  return RegisterInternal(backend, device_ordinal, device_memory, shape, tag,
                          /*initial_ref_count=*/1);
}

GlobalDataHandle AllocationTracker::RegisterInternal(
    Backend* backend, int device_ordinal, se::DeviceMemoryBase device_memory,
    const Shape& shape, const string& tag, int initial_ref_count) {
  VLOG(2) << "RegisterInternal("
          << "tag: \"" << tag << "\" "
          << "device_ordinal: " << device_ordinal << " "
          << "device_memory: " << device_memory.opaque() << " "
          << "shape: " << shape.ShortDebugString() << ")";
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape));

  int64 handle;
  HandleMap& handle_map = GetOrCreateOpaqueToHandleMap(device_ordinal);
  auto handle_it = handle_map.find(device_memory.opaque());
  if (handle_it != handle_map.end()) {
    handle = handle_it->second;
    auto& allocation = FindOrDie(handle_to_allocation_, handle);
    int ref_count = allocation->ref_count();
    CHECK_GT(ref_count, 0);
    VLOG(2) << "ref_count: " << ref_count << " -> " << ref_count + 1;
    allocation->increment_ref_count();
  } else {
    handle = next_handle_++;
    VLOG(2) << "ref_count: " << initial_ref_count;
    InsertOrDie(&handle_map, device_memory.opaque(), handle);
    auto inserted = handle_to_allocation_.emplace(
        handle, MakeUnique<Allocation>(backend, device_ordinal, device_memory,
                                       shape, tag, initial_ref_count));
    CHECK(inserted.second);
  }

  GlobalDataHandle result;
  result.set_handle(handle);
  VLOG(2) << "handle: " << handle;

  return result;
}

tensorflow::Status AllocationTracker::Unregister(const GlobalDataHandle& data) {
  tensorflow::mutex_lock lock(allocation_mutex_);
  TF_ASSIGN_OR_RETURN(Allocation * allocation, ResolveInternal(data));
  std::set<void*> deallocated_buffers;
  TF_RETURN_IF_ERROR(
      DeallocateShape(allocation->backend(), allocation->device_ordinal(),
                      allocation->mutable_device_memory(), allocation->shape(),
                      &deallocated_buffers));
  return tensorflow::Status::OK();
}

tensorflow::Status AllocationTracker::DeallocateShape(
    Backend* backend, int device_ordinal, se::DeviceMemoryBase* device_memory,
    const Shape& shape, std::set<void*>* deallocated_buffers) {
  VLOG(2) << "DeallocateShape("
          << "shape: \"" << shape.ShortDebugString() << "\" "
          << "device_memory: " << device_memory->opaque() << ")";
  if (ContainsKey(*deallocated_buffers, device_memory->opaque())) {
    // Buffer has already been deallocated. Nothing to do.
    VLOG(2) << "already deallocated";
    return tensorflow::Status::OK();
  }

  // Add buffer to deallocated set so we do not try to deallocate it again
  // if it is encountered again while traversing a tuple.
  deallocated_buffers->insert(device_memory->opaque());

  HandleMap& handle_map = GetOrCreateOpaqueToHandleMap(device_ordinal);
  auto handle_it = handle_map.find(device_memory->opaque());
  if (handle_it != handle_map.end()) {
    int64 handle = handle_it->second;
    auto& allocation = FindOrDie(handle_to_allocation_, handle);
    int ref_count = allocation->ref_count();
    VLOG(2) << "ref_count: " << ref_count << " -> " << ref_count - 1;
    allocation->decrement_ref_count();
    if (allocation->ref_count() > 0) {
      // Buffer is referred to by another allocation. Don't deallocate it.
      return tensorflow::Status::OK();
    }
    handle_map.erase(device_memory->opaque());
  }

  // TODO(b/36256956) Ideally tuple elements could always be distinct buffers.
  if (ShapeUtil::IsTuple(shape) &&
      backend->transfer_manager()->TupleElementsAreDistinctBuffers()) {
    // Traverse into tuple recursively deallocating buffers.
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                        backend->stream_executor(device_ordinal));
    TF_ASSIGN_OR_RETURN(std::vector<se::DeviceMemoryBase> elements,
                        backend->transfer_manager()->ShallowCopyTupleFromDevice(
                            executor, *device_memory, shape));

    TF_RET_CHECK(ShapeUtil::TupleElementCount(shape) == elements.size())
        << "tuple has unexpected number of elements: " << elements.size()
        << " != " << ShapeUtil::TupleElementCount(shape);
    for (std::vector<se::DeviceMemoryBase>::size_type i = 0;
         i < elements.size(); ++i) {
      VLOG(2) << "recursing onto the tuple elements";
      TF_RETURN_IF_ERROR(DeallocateShape(backend, device_ordinal, &elements[i],
                                         shape.tuple_shapes(i),
                                         deallocated_buffers));
    }
  }

  return backend->memory_allocator()->Deallocate(device_ordinal, device_memory);
}

StatusOr<std::vector<GlobalDataHandle>> AllocationTracker::DeconstructTuple(
    const GlobalDataHandle& data) {
  tensorflow::mutex_lock lock(allocation_mutex_);
  TF_ASSIGN_OR_RETURN(Allocation * allocation, ResolveInternal(data));

  if (!ShapeUtil::IsTuple(allocation->shape())) {
    return InvalidArgument("global data handle %lld is not a tuple",
                           data.handle());
  }

  if (ShapeUtil::IsNestedTuple(allocation->shape())) {
    return Unimplemented("deconstructing nested tuples not yet supported");
  }

  TF_ASSIGN_OR_RETURN(
      se::StreamExecutor * executor,
      allocation->backend()->stream_executor(allocation->device_ordinal()));
  TF_ASSIGN_OR_RETURN(
      std::vector<se::DeviceMemoryBase> element_bases,
      allocation->backend()->transfer_manager()->ShallowCopyTupleFromDevice(
          executor, allocation->device_memory(), allocation->shape()));

  std::vector<GlobalDataHandle> element_handles;
  for (int i = 0; i < element_bases.size(); ++i) {
    element_handles.push_back(RegisterInternal(
        allocation->backend(), allocation->device_ordinal(), element_bases[i],
        ShapeUtil::GetSubshape(allocation->shape(), {i}),
        tensorflow::strings::StrCat(allocation->tag(), ".element_", i),
        /*initial_ref_count=*/2));
  }
  return std::move(element_handles);
}

StatusOr<const Allocation*> AllocationTracker::Resolve(
    const GlobalDataHandle& data) {
  tensorflow::mutex_lock lock(allocation_mutex_);
  return AllocationTracker::ResolveInternal(data);
}

StatusOr<Allocation*> AllocationTracker::ResolveInternal(
    const GlobalDataHandle& data) {
  VLOG(2) << "resolve:" << data.handle();
  auto it = handle_to_allocation_.find(data.handle());
  if (it == handle_to_allocation_.end()) {
    return NotFound("no allocation record for global data handle: %lld",
                    data.handle());
  }
  Allocation* allocation = it->second.get();

  if (allocation->is_deallocated()) {
    return InvalidArgument("global data handle %lld was previously deallocated",
                           data.handle());
  }

  return allocation;
}

AllocationTracker::HandleMap& AllocationTracker::GetOrCreateOpaqueToHandleMap(
    int device_ordinal) {
  if (opaque_to_handle_.size() <= device_ordinal) {
    opaque_to_handle_.resize(device_ordinal + 1);
  }
  return opaque_to_handle_[device_ordinal];
}

}  // namespace xla
