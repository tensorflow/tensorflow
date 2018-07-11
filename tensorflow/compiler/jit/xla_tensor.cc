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

/*static*/ XlaTensor* XlaTensor::FromTensor(const Tensor* tensor) {
  if (tensor->NumElements() == 0) {
    return nullptr;
  }
  XlaTensor* xla_tensor =
      FromOpaquePointer(const_cast<char*>(tensor->tensor_data().data()));
  return xla_tensor;
}

/*static*/ bool XlaTensor::RefCountIsOne(const Tensor& tensor) {
  return tensor.RefCountIsOne();
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

  xla::ScopedShapedBuffer shaped_buffer(on_host_shape, on_device_shape,
                                        client->backend().memory_allocator(),
                                        device_ordinal);
  for (auto& index_to_buffer : shaped_buffer.buffers()) {
    xla::Shape subshape =
        xla::ShapeUtil::GetSubshape(on_device_shape, index_to_buffer.first);
    uint64 size =
        client->backend().transfer_manager()->GetByteSizeRequirement(subshape);
    TF_ASSIGN_OR_RETURN(xla::OwningDeviceMemory buffer,
                        client->backend().memory_allocator()->Allocate(
                            device_ordinal, size, /*retry_on_failure=*/false));
    // Move our buffer into shaped_buffer, which takes ownership of it.
    index_to_buffer.second = buffer.Forget();
  }

  VLOG(4) << shaped_buffer.ToString();

  set_shaped_buffer(std::move(shaped_buffer));
  return Status::OK();
}

se::Event* XlaTensor::GetDefinitionEvent(se::Stream* stream) {
  if (!definition_event_.has_value()) {
    return nullptr;
  }

  // The set of defined streams is expected to be very small indeed (usually
  // 1-2), so a simple linear scan should be fast enough.
  if (std::find(streams_defined_on_.begin(), streams_defined_on_.end(),
                stream) != streams_defined_on_.end()) {
    // stream is in streams_defined_on_; it doesn't need to be waited on.
    return nullptr;
  }

  return &*definition_event_;
}

void XlaTensor::SetDefinedOn(se::Stream* stream, se::Event event) {
  CHECK(!definition_event_.has_value())
      << "SetDefinedOn must only be called once!";
  definition_event_ = std::move(event);
  streams_defined_on_.push_back(stream);
}

void XlaTensor::SetDefinedOn(se::Stream* stream) {
  streams_defined_on_.push_back(stream);
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
