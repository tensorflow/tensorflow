/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime2/hal.h"

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "third_party/iree/runtime/src/iree/hal/api.h"  // IWYU pragma: keep
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Helper functions to work with IREE buffers
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Add support for all element types.
static PrimitiveType GetElementType(iree_hal_buffer_view_t* view) {
  switch (iree_hal_buffer_view_element_type(view)) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return PrimitiveType::F32;
    default:
      assert(false && "unsupported iree element type");
      return PrimitiveType::PRIMITIVE_TYPE_INVALID;
  }
}

static absl::InlinedVector<int64_t, 4> GetDims(iree_hal_buffer_view_t* view) {
  const iree_host_size_t* dims = iree_hal_buffer_view_shape_dims(view);
  iree_host_size_t rank = iree_hal_buffer_view_shape_rank(view);
  return absl::InlinedVector<int64_t, 4>(dims, dims + rank);
}

Shape GetBufferShape(iree_hal_buffer_view_t* view) {
  return ShapeUtil::MakeShape(GetElementType(view), GetDims(view));
}

StatusOr<se::DeviceMemoryBase> GetDeviceMemory(
    iree_hal_allocator_t* device_allocator, iree_hal_buffer_t* buffer) {
  // Get original allocation behind a buffer subspan.
  iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);

  // Export allocated buffer as an external device allocation.
  iree_hal_external_buffer_t external_buffer;
  iree::Status exported = iree_hal_allocator_export_buffer(
      device_allocator, allocated,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer);

  if (!exported.ok())
    return absl::InternalError(absl::StrFormat(
        "failed to export HAL buffer: %s", exported.ToString()));

  auto* data = reinterpret_cast<std::byte*>(
      external_buffer.handle.device_allocation.ptr);
  return se::DeviceMemoryBase(data + iree_hal_buffer_byte_offset(buffer),
                              iree_hal_buffer_byte_length(buffer));
}

StatusOr<se::DeviceMemoryBase> GetDeviceMemory(
    iree_hal_allocator_t* device_allocator, iree_hal_buffer_view_t* view) {
  return GetDeviceMemory(device_allocator, iree_hal_buffer_view_buffer(view));
}

}  // namespace xla::gpu
