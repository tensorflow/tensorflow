/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/allocation_tracker.h"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<GlobalDataHandle> AllocationTracker::Register(
    ScopedShapedBuffer shaped_buffer, const std::string& tag) {
  absl::MutexLock lock(&mutex_);
  VLOG(2) << "Register";
  std::vector<ScopedShapedBuffer> replicated_buffers;
  replicated_buffers.emplace_back(std::move(shaped_buffer));
  return RegisterInternal(std::move(replicated_buffers), tag);
}

absl::StatusOr<GlobalDataHandle> AllocationTracker::RegisterReplicatedBuffers(
    std::vector<ScopedShapedBuffer> replicated_buffers,
    const std::string& tag) {
  absl::MutexLock lock(&mutex_);
  VLOG(2) << "RegisterReplicatedBuffers";
  return RegisterInternal(std::move(replicated_buffers), tag);
}

// ReleaseIfScopedShapedBuffer lets RegisterInternal<ShapedBufferTy>(b) call
// b.release() if b is a ScopedShapedBuffer, or otherwise pass b through
// unmodified.
static ShapedBuffer ReleaseIfScopedShapedBuffer(ShapedBuffer b) { return b; }
static ShapedBuffer ReleaseIfScopedShapedBuffer(ScopedShapedBuffer b) {
  return b.release();
}

template <typename ShapedBufferTy>
absl::StatusOr<GlobalDataHandle> AllocationTracker::RegisterInternal(
    std::vector<ShapedBufferTy> replicated_buffers, const std::string& tag) {
  static_assert(std::is_same<ShapedBufferTy, ShapedBuffer>::value ||
                    std::is_same<ShapedBufferTy, ScopedShapedBuffer>::value,
                "ShapedBufferTy must be ShapedBuffer or ScopedShapedBuffer.");
  VLOG(2) << "RegisterInternal("
          << "tag: \"" << tag << "\" with " << replicated_buffers.size()
          << " shaped_buffers.";

  int64_t handle = next_handle_++;
  for (auto& shaped_buffer : replicated_buffers) {
    std::vector<ShapeIndex> shape_indices;
    ShapeUtil::ForEachSubshape(
        shaped_buffer.on_device_shape(),
        [&](const Shape& /*subshape*/, const ShapeIndex& index) {
          shape_indices.push_back(index);
        });
    // Add shaped_buffer's buffers to opaque_to_allocation_map_, which owns
    // them.
    for (const ShapeIndex& index : shape_indices) {
      AddAllocationOrIncrementRefCount(shaped_buffer.buffer(index),
                                       shaped_buffer.device_ordinal());
    }
    // If ShapedBufferTy is ScopedShapedBuffer, release the ScopedShapedBuffer
    // into a regular ShapedBuffer, which is stored in
    // handle_to_shaped_buffers_.
    handle_to_shaped_buffers_[handle].emplace_back(
        std::make_unique<ShapedBuffer>(
            ReleaseIfScopedShapedBuffer(std::move(shaped_buffer))));
  }

  GlobalDataHandle result;
  result.set_handle(handle);
  VLOG(2) << "handle: " << handle;
  return result;
}

absl::Status AllocationTracker::Unregister(const GlobalDataHandle& data) {
  absl::MutexLock lock(&mutex_);
  VLOG(2) << "Unregister("
          << "handle: " << data.handle() << ")";
  TF_ASSIGN_OR_RETURN(std::vector<const ShapedBuffer*> replicated_buffers,
                      ResolveInternal(data));
  for (const auto& shaped_buffer : replicated_buffers) {
    std::vector<ShapeIndex> shape_indices;
    ShapeUtil::ForEachSubshape(
        shaped_buffer->on_device_shape(),
        [&shape_indices](const Shape& /*subshape*/, const ShapeIndex& index) {
          shape_indices.push_back(index);
        });
    for (const ShapeIndex& index : shape_indices) {
      TF_RETURN_IF_ERROR(DecrementRefCount(shaped_buffer->buffer(index),
                                           shaped_buffer->device_ordinal()));
    }
  }
  // Keep a nullptr as a tombstone for unregistered handles. This enables
  // better error messages. That is, "handle has been deallocated" versus
  // "handle does not exist".
  auto it = handle_to_shaped_buffers_.find(data.handle());
  if (it == handle_to_shaped_buffers_.end()) {
    return NotFound("no allocation record for global data handle: %d",
                    data.handle());
  }
  for (auto& shaped_buffer : it->second) {
    shaped_buffer.reset();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<GlobalDataHandle>>
AllocationTracker::DeconstructTuple(const GlobalDataHandle& data) {
  absl::MutexLock lock(&mutex_);

  TF_ASSIGN_OR_RETURN(std::vector<const ShapedBuffer*> replicated_buffers,
                      ResolveInternal(data));
  // We only need to care about replica id 0 here, since the GlobalDataHandle is
  // the same for all buffers across replicas.
  const ShapedBuffer* shaped_buffer = replicated_buffers[0];
  if (!shaped_buffer->on_device_shape().IsTuple()) {
    return InvalidArgument("global data handle %d is not a tuple",
                           data.handle());
  }

  if (ShapeUtil::IsNestedTuple(shaped_buffer->on_device_shape())) {
    return Unimplemented("Deconstructing nested tuples is not implemented.");
  }

  std::vector<GlobalDataHandle> element_handles;
  const auto n = ShapeUtil::TupleElementCount(shaped_buffer->on_device_shape());
  element_handles.reserve(n);
  for (int i = 0; i < n; ++i) {
    auto element_buffer = ShapedBuffer(
        ShapeUtil::GetTupleElementShape(shaped_buffer->on_device_shape(), i),
        shaped_buffer->device_ordinal());
    element_buffer.set_buffer(shaped_buffer->buffer(/*index=*/{i}),
                              /*index=*/{});
    std::vector<ShapedBuffer> replicated_buffers;
    replicated_buffers.push_back(std::move(element_buffer));
    TF_ASSIGN_OR_RETURN(
        GlobalDataHandle element_handle,
        RegisterInternal(std::move(replicated_buffers), "deconstructed tuple"));

    element_handles.push_back(element_handle);
  }
  return std::move(element_handles);
}

absl::StatusOr<std::vector<const ShapedBuffer*>> AllocationTracker::Resolve(
    const GlobalDataHandle& data) const {
  absl::MutexLock lock(&mutex_);
  return AllocationTracker::ResolveInternal(data);
}

absl::StatusOr<const ShapedBuffer*> AllocationTracker::ResolveForReplica(
    const GlobalDataHandle& data, int replica_id) const {
  absl::MutexLock lock(&mutex_);
  TF_ASSIGN_OR_RETURN(std::vector<const ShapedBuffer*> replicated_buffers,
                      ResolveInternal(data));
  if (replica_id >= replicated_buffers.size()) {
    return InvalidArgument(
        "Requesting buffer for replica %d, but found buffers only for %lu "
        "replicas.",
        replica_id, replicated_buffers.size());
  }
  return replicated_buffers[replica_id];
}

absl::StatusOr<std::vector<const ShapedBuffer*>>
AllocationTracker::ResolveInternal(const GlobalDataHandle& data) const {
  VLOG(2) << "resolve:" << data.handle();
  auto it = handle_to_shaped_buffers_.find(data.handle());
  if (it == handle_to_shaped_buffers_.end()) {
    return NotFound("no allocation record for global data handle: %d",
                    data.handle());
  }
  std::vector<const ShapedBuffer*> replicated_buffers;
  for (const auto& shaped_buffer : it->second) {
    if (shaped_buffer == nullptr) {
      return InvalidArgument("global data handle %d was previously deallocated",
                             data.handle());
    }
    replicated_buffers.push_back(shaped_buffer.get());
  }

  return replicated_buffers;
}

void AllocationTracker::AddAllocationOrIncrementRefCount(
    se::DeviceMemoryBase device_memory, int device_ordinal) {
  AllocationMap& allocation_map = opaque_to_allocation_map_[device_ordinal];
  auto it = allocation_map.find(device_memory.opaque());
  if (it == allocation_map.end()) {
    allocation_map[device_memory.opaque()] = {
        se::OwningDeviceMemory(device_memory, device_ordinal,
                               backend_->memory_allocator()),
        /*ref_count=*/1};
  } else {
    it->second.ref_count++;
  }
}

absl::Status AllocationTracker::DecrementRefCount(
    se::DeviceMemoryBase device_memory, int device_ordinal) {
  AllocationMap& allocation_map = opaque_to_allocation_map_[device_ordinal];
  auto it = allocation_map.find(device_memory.opaque());
  TF_RET_CHECK(it != allocation_map.end());
  Allocation& allocation = it->second;
  TF_RET_CHECK(allocation.ref_count >= 1);
  if (allocation.ref_count == 1) {
    TF_RETURN_IF_ERROR(allocation.device_memory.Free());
    allocation_map.erase(it);
  } else {
    allocation.ref_count--;
  }
  return absl::OkStatus();
}

}  // namespace xla
