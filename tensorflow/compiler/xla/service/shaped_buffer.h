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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SHAPED_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SHAPED_BUFFER_H_

#include <memory>
#include <ostream>
#include <string>

#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Class which encapsulates a buffer or set of buffers containing data of a
// particular XLA shape.
class ShapedBuffer {
 public:
  // Construct a ShapedBuffer with null DeviceMemoryBases at each index. The
  // shape of the data on the host and the device may differ because the device
  // may have a different representation for different data types. Therefore,
  // both the on-host and on-device shape are required. The on-device shape
  // determines the number of device allocations (DeviceMemoryBase) held by the
  // ShapedBuffer.
  ShapedBuffer(const Shape& on_host_shape, const Shape& on_device_shape,
               const perftools::gputools::Platform* platform,
               int device_ordinal);

  // Returns the shape of the on-host representation of the data held by this
  // ShapedBuffer.
  const Shape& on_host_shape() const { return on_host_shape_; }

  // Returns the shape of the on-device representation of the data held by this
  // ShapedBuffer.
  const Shape& on_device_shape() const { return on_device_shape_; }

  const perftools::gputools::Platform* platform() const { return platform_; }
  int device_ordinal() const { return device_ordinal_; }

  // Return the root buffer of the shape (shape index {}).
  const perftools::gputools::DeviceMemoryBase& root_buffer() const {
    return buffer(/*index=*/{});
  }

  // Returns the buffer at the given shape index where index is defined as in
  // ShapeUtil::GetSubshape.
  const perftools::gputools::DeviceMemoryBase& buffer(
      const ShapeIndex& index) const {
    return buffers_.element(index);
  }

  // Sets the device memory buffer at the given index.
  void set_buffer(const perftools::gputools::DeviceMemoryBase& buffer,
                  const ShapeIndex& index) {
    *buffers_.mutable_element(index) = buffer;
  }

  // Returns the underlying ShapeTree containing all the device addresses in the
  // ShapedBuffer.
  const ShapeTree<perftools::gputools::DeviceMemoryBase>& buffers() const {
    return buffers_;
  }
  ShapeTree<perftools::gputools::DeviceMemoryBase>& buffers() {
    return buffers_;
  }

  // Set all device memory pointers in the object to null.
  void clear();

  string ToString() const;

  ShapedBuffer(ShapedBuffer&& s);
  ShapedBuffer& operator=(ShapedBuffer&&);

 protected:
  ShapedBuffer(const ShapedBuffer&) = delete;
  ShapedBuffer& operator=(const ShapedBuffer&) = delete;

  // The shape of the data when represented on the host.
  Shape on_host_shape_;

  // The shape of the data on the device.
  Shape on_device_shape_;

  // The platform the memory is allocated on.
  const perftools::gputools::Platform* platform_;

  // The device the memory is allocated on.
  int device_ordinal_;

  // The tree of device buffers. Its shape is on_device_shape().
  ShapeTree<perftools::gputools::DeviceMemoryBase> buffers_;
};

std::ostream& operator<<(std::ostream& out, const ShapedBuffer& buffer);

// ShapedBuffer derived class which allocates all internal buffers on
// construction and deallocates the memory when the object is
// destructed.
class ScopedShapedBuffer : public ShapedBuffer {
 public:
  // Takes a ShapedBuffer and returns a ScopedShapedBuffer which manages the
  // deallocation of the device memory held in the shaped buffer. All device
  // memory pointers in the given ShapedBuffer are set to null.
  static StatusOr<std::unique_ptr<ScopedShapedBuffer>> MakeScoped(
      ShapedBuffer* shaped_buffer, DeviceMemoryAllocator* allocator);

  // Create a ScopedShapedBuffer with null DeviceMemoryBases at each index.
  ScopedShapedBuffer(const Shape& on_host_shape, const Shape& on_device_shape,
                     DeviceMemoryAllocator* allocator, int device_ordinal);

  // Create a ScopedShapedBuffer by taking over the memory from the incoming
  // ShapedBuffer.
  ScopedShapedBuffer(ShapedBuffer shaped_buffer,
                     DeviceMemoryAllocator* allocator);

  // Return the allocator used to allocate the device memory held in this
  // ScopedShapedBuffer.
  DeviceMemoryAllocator* memory_allocator() const { return allocator_; }

  // Release all device memory owned by this ScopedShapedBuffer and
  // return the device memory pointers in the form of a
  // ShapedBuffer. The returned ShapedBuffer takes over the memory
  // from the ScopedShapedBuffer. The resulting ScopedShapedBuffer can
  // only be destroyed.
  std::unique_ptr<ShapedBuffer> release();

  // All buffers in the shape are deallocated on destruction.
  virtual ~ScopedShapedBuffer();

 protected:
  ScopedShapedBuffer(const ScopedShapedBuffer&) = delete;
  void operator=(const ScopedShapedBuffer&) = delete;

  DeviceMemoryAllocator* allocator_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SHAPED_BUFFER_H_
