/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"

namespace tensorflow {

/*static*/ XlaTensor* XlaTensor::FromTensor(Tensor* tensor) {
  if (tensor->NumElements() == 0) {
    return nullptr;
  }
  XlaTensor* xla_tensor =
      FromOpaquePointer(const_cast<char*>(tensor->tensor_data().data()));
  return xla_tensor;
}

/*static*/ const XlaTensor* XlaTensor::FromTensor(const Tensor* tensor) {
  return FromTensor(const_cast<Tensor*>(tensor));
}

/*static*/ se::DeviceMemoryBase XlaTensor::DeviceMemoryFromTensor(
    const Tensor& tensor) {
  const XlaTensor* xla_tensor = FromTensor(&tensor);
  if (xla_tensor) {
    CHECK(xla_tensor->has_shaped_buffer());
    return xla_tensor->shaped_buffer().root_buffer();
  } else {
    return se::DeviceMemoryBase(const_cast<char*>(tensor.tensor_data().data()),
                                tensor.tensor_data().size());
  }
}

Status XlaTensor::AllocateShapedBuffer(DataType dtype, const TensorShape& shape,
                                       xla::LocalClient* client,
                                       int device_ordinal) {
  xla::Shape on_host_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype, shape, &on_host_shape));
  xla::Shape on_device_shape =
      client->backend().transfer_manager()->HostShapeToDeviceShape(
          on_host_shape);

  xla::ShapedBuffer buffer(on_host_shape, on_device_shape, client->platform(),
                           device_ordinal);
  for (auto& index_to_buffer : buffer.buffers()) {
    xla::Shape subshape =
        xla::ShapeUtil::GetSubshape(on_device_shape, index_to_buffer.first);
    uint64 size =
        client->backend().transfer_manager()->GetByteSizeRequirement(subshape);
    TF_ASSIGN_OR_RETURN(index_to_buffer.second,
                        client->backend().memory_allocator()->Allocate(
                            device_ordinal, size, /*retry_on_failure=*/false));
  }

  set_shaped_buffer(xla::ScopedShapedBuffer(
      std::move(buffer), client->backend().memory_allocator()));
  return Status::OK();
}

// The pointer tag, OR-ed into the XlaTensor's address to distinguish it from
// device-side tensors, which are either CPU or GPU memory pointers. This works
// because we're guaranteed that CPU and GPU pointers are aligned to > 1 bits.
namespace {
constexpr uintptr_t kTag = 0x1ULL;
}

/*static*/ XlaTensor* XlaTensor::FromOpaquePointer(void* ptr) {
  uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
  if (value & kTag) {
    return reinterpret_cast<XlaTensor*>(value & ~kTag);
  } else {
    return nullptr;
  }
}

/*static*/ void* XlaTensor::ToOpaquePointer(XlaTensor* tensor) {
  uintptr_t value = reinterpret_cast<uintptr_t>(tensor);
  CHECK_EQ(value & kTag, 0);
  value |= kTag;
  return reinterpret_cast<XlaTensor*>(value);
}

}  // namespace tensorflow
