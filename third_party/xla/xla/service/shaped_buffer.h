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

#ifndef XLA_SERVICE_SHAPED_BUFFER_H_
#define XLA_SERVICE_SHAPED_BUFFER_H_

#include <memory>
#include <ostream>
#include <string>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {

class ScopedShapedBuffer;

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
  // Specify `physical_device_ordinal` if multiple devices share the same
  // physical device, e.g., virtual GPUs.
  ShapedBuffer(Shape on_device_shape, int device_ordinal,
               int physical_device_ordinal = -1);

  // TODO(b/170310047): remove this overload.
  ShapedBuffer(Shape on_host_shape, Shape on_device_shape, int device_ordinal,
               int physical_device_ordinal = -1);

  // Movable, but not copyable.
  ShapedBuffer(ShapedBuffer&& s);
  ShapedBuffer& operator=(ShapedBuffer&&);
  ShapedBuffer(const ShapedBuffer&) = delete;
  ShapedBuffer& operator=(const ShapedBuffer&) = delete;

  // Prevent (some forms of) accidental object slicing.
  ShapedBuffer(const ScopedShapedBuffer&) = delete;
  ShapedBuffer& operator=(const ScopedShapedBuffer&) = delete;

  virtual ~ShapedBuffer();

  // Returns the shape of the on-host representation of the data held by this
  // ShapedBuffer.
  const Shape& on_host_shape() const { return on_host_shape_; }

  // Returns the shape of the on-device representation of the data held by this
  // ShapedBuffer.
  const Shape& on_device_shape() const { return on_device_shape_; }

  int device_ordinal() const { return device_ordinal_; }
  int physical_device_ordinal() const { return physical_device_ordinal_; }

  // Return the root buffer of the shape (shape index {}).
  const se::DeviceMemoryBase& root_buffer() const {
    return buffer(/*index=*/{});
  }

  // Returns the buffer at the given shape index where index is defined as in
  // ShapeUtil::GetSubshape.
  const se::DeviceMemoryBase& buffer(const ShapeIndex& index) const {
    return buffers_.element(index);
  }

  // Sets the device memory buffer at the given index.
  void set_buffer(const se::DeviceMemoryBase& buffer, const ShapeIndex& index) {
    *buffers_.mutable_element(index) = buffer;
  }

  // Sets all buffers.
  //
  // Precondition: buffers.shape == on_device_shape_
  void set_buffers(ShapeTree<se::DeviceMemoryBase> buffers) {
    CHECK(ShapeUtil::Equal(buffers.shape(), on_device_shape_));
    buffers_ = std::move(buffers);
    buffers_.replace_shape_ptr(on_device_shape_);
  }

  // Reset the shape of this shaped buffer and underlying buffer structure.
  //
  // Precondition: EqualStructure(this->on_device_shape_, on_device_shape).
  void set_shapes(const Shape& on_device_shape) {
    CHECK(ShapeUtil::EqualStructure(on_device_shape, on_device_shape_))
        << "Structures are not the same. new: " << on_device_shape
        << ", old: " << on_device_shape_;
    on_host_shape_ = ShapeUtil::DeviceShapeToHostShape(on_device_shape);
    on_device_shape_ = on_device_shape;
    buffers_.replace_shape_ptr(on_device_shape_);
  }
  // TODO(b/170310047): remove this overload.
  void set_shapes(const Shape& on_host_shape, const Shape& on_device_shape) {
    set_shapes(on_device_shape);
  }

  // Returns the underlying ShapeTree containing all the device addresses in the
  // ShapedBuffer.
  const ShapeTree<se::DeviceMemoryBase>& buffers() const { return buffers_; }
  ShapeTree<se::DeviceMemoryBase>& buffers() { return buffers_; }

  absl::StatusOr<ShapedBuffer> SubShapedBuffer(const ShapeIndex& index) const;

  // Set all device memory pointers in the object to null.
  void clear();

  std::string ToString() const;

 protected:
  Shape on_host_shape_;

  // The shape of the data on the device.
  Shape on_device_shape_;

  // The device the memory is allocated on.
  int device_ordinal_;
  int physical_device_ordinal_;

  // The tree of device buffers. Its shape is on_device_shape().
  ShapeTree<se::DeviceMemoryBase> buffers_;
};

std::ostream& operator<<(std::ostream& out, const ShapedBuffer& buffer);

// ScopedShapedBuffer takes allocated buffers as inputs, and deallocates on
// destruction. This class represents an owning wrapper around `ShapedBuffer`.
//
// TODO(timshen): Remove inheritance between ScopedShapedBuffer and
// ShapedBuffer.  There should never be a need to consider a ScopedShapedBuffer
// as a ShapedBuffer, because in that case we should just be able to pass around
// our ShapeTree<DeviceMemoryBase>.  Inheritance only adds complexity.  See
// discussion in cl/192849370.
class ScopedShapedBuffer : public ShapedBuffer {
 public:
  // Creates a ScopedShapedBuffer with null DeviceMemoryBases at each index.
  explicit ScopedShapedBuffer(Shape on_device_shape,
                              se::DeviceMemoryAllocator* allocator,
                              int device_ordinal,
                              int physical_device_ordinal = -1);
  // TODO(b/170310047): remove this overload.
  explicit ScopedShapedBuffer(Shape on_host_shape, Shape on_device_shape,
                              se::DeviceMemoryAllocator* allocator,
                              int device_ordinal,
                              int physical_device_ordinal = -1);

  // Create a ScopedShapedBuffer by taking over the memory from the incoming
  // ShapedBuffer.
  explicit ScopedShapedBuffer(ShapedBuffer shaped_buffer,
                              se::DeviceMemoryAllocator* allocator);

  // Movable, but not copyable.
  ScopedShapedBuffer(ScopedShapedBuffer&& s);
  ScopedShapedBuffer& operator=(ScopedShapedBuffer&&);
  ScopedShapedBuffer(const ScopedShapedBuffer&) = delete;
  ScopedShapedBuffer& operator=(const ScopedShapedBuffer&) = delete;

  // All buffers in the shape are deallocated on destruction.
  ~ScopedShapedBuffer() override;

  // Return the allocator used to allocate the device memory held in this
  // ScopedShapedBuffer.
  se::DeviceMemoryAllocator* memory_allocator() const { return allocator_; }

  // Sets the device memory buffer at the given index.
  //
  // If the given buffer's device memory is non-null, its device_ordinal and
  // allocator must match those in `this`.
  void set_buffer(se::OwningDeviceMemory buffer, const ShapeIndex& index) {
    if (!buffer.is_null()) {
      CHECK_EQ(buffer.device_ordinal(), device_ordinal());
      CHECK_EQ(buffer.allocator(), allocator_);
      *buffers_.mutable_element(index) = buffer.Release();
    } else {
      *buffers_.mutable_element(index) = se::DeviceMemoryBase();
    }
  }

  // Like unique_ptr::release(), creates and returns a regular ShapedBuffer from
  // this ScopedShapedBuffer, without freeing any of the associated memory.
  //
  // It's the caller's job to ensure that the memory contained therein is freed.
  [[nodiscard]] ShapedBuffer release();

  // Extracts the sub-tree rooted at 'index' and returns a ScopedShapedBuffer
  // that holds ownership of the subtree. Sets the buffers corresponding to the
  // subtree to null in 'this'.
  ScopedShapedBuffer TakeSubTree(ShapeIndexView index);

 protected:
  void Deallocate();

  se::DeviceMemoryAllocator* allocator_;
};

}  // namespace xla

#endif  // XLA_SERVICE_SHAPED_BUFFER_H_
