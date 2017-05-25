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

#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Class which encapsulates a buffer or set of buffers containing data of a
// particular XLA shape. Used for zero-copy execution interface for a
// XLA client running in the same process as the service (LocalClient),
class ShapedBuffer {
 public:
  // Creates a ShapedBuffer of arbitrary shape. All buffer pointers
  // (DeviceMemoryBase) in the returned ShapedBuffer are initialized to null.
  static StatusOr<std::unique_ptr<ShapedBuffer>> MakeShapedBuffer(
      const Shape& shape, const perftools::gputools::Platform* platform,
      int device_ordinal);

  // Convenience method which creates a ShapedBuffer of array shape (not a
  // tuple). Its single buffer pointer is set to the given value "buffer". The
  // given buffer must be large enough to store the given shape as given by
  // ShapeUtil::ByteSizeOf.
  static StatusOr<std::unique_ptr<ShapedBuffer>> MakeArrayShapedBuffer(
      const Shape& shape, const perftools::gputools::Platform* platform,
      int device_ordinal, const perftools::gputools::DeviceMemoryBase& buffer);

  // Convenience method which creates a ShapedBuffer of a non-nested tuple. The
  // buffer pointers in the return ShapedBuffer are set to the given
  // "buffers". The size of buffers must match the number of elements in the
  // tuple shape and be large enough to store their respective shape as given by
  // ShapeUtil::ByteSizeOf.
  static StatusOr<std::unique_ptr<ShapedBuffer>> MakeUnnestedTupleShapedBuffer(
      const Shape& shape, const perftools::gputools::Platform* platform,
      int device_ordinal,
      const tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          buffers);

  const Shape& shape() const { return shape_; }
  const perftools::gputools::Platform* platform() const { return platform_; }
  int device_ordinal() const { return device_ordinal_; }

  // Returns the buffer at the given shape index where index is defined as in
  // ShapeUtil::GetSubshape.
  const perftools::gputools::DeviceMemoryBase& buffer(
      const ShapeIndex& index) const;
  perftools::gputools::DeviceMemoryBase* mutable_buffer(
      const ShapeIndex& index);

  // Returns the underlying structure which stores the buffer pointers.
  const std::vector<perftools::gputools::DeviceMemoryBase>& buffers() const {
    return buffers_;
  }
  std::vector<perftools::gputools::DeviceMemoryBase>* mutable_buffers() {
    return &buffers_;
  }

  // Returns the tree of indices which map to buffer pointers.
  const ShapeTree<size_t>& shape_index_to_buffer_entry() const {
    return shape_index_to_buffer_entry_;
  }
  ShapeTree<size_t>* mutable_shape_index_to_buffer_entry() {
    return &shape_index_to_buffer_entry_;
  }

 protected:
  ShapedBuffer(const Shape& shape,
               const perftools::gputools::Platform* platform,
               int device_ordinal);

  // The shape of the device buffer with layout.
  const Shape shape_;

  // The list of DeviceMemoryBase pointers representing this shape.
  // Note that there can be a many to one relationship between tuple elements
  // and buffers.  To account for this, shape_index_to_buffer_entry_ allows us
  // to make from a position in a shape to an index into this list.
  std::vector<perftools::gputools::DeviceMemoryBase> buffers_;

  // The tree of indices into buffers_.
  ShapeTree<size_t> shape_index_to_buffer_entry_;

  // The platform the memory is allocated on.
  const perftools::gputools::Platform* platform_;

  // The device the memory is allocated on.
  const int device_ordinal_;
};

// ShapedBuffer derived class which allocates all internal buffers on
// construction and deallocates the memory when the object is
// destructed.
class ScopedShapedBuffer : public ShapedBuffer {
 public:
  // Return a new ScopedShapedBuffer of an arbitrary shape. All buffers in the
  // ScopedShapedBuffers are automatically allocated to exactly the size of
  // their respective array shape.
  static StatusOr<std::unique_ptr<ScopedShapedBuffer>> MakeScopedShapedBuffer(
      const Shape& shape, DeviceMemoryAllocator* allocator, int device_ordinal);

  // All buffers in the shape are deallocated on destruction.
  ~ScopedShapedBuffer();

 protected:
  ScopedShapedBuffer(const Shape& shape, DeviceMemoryAllocator* allocator,
                     int device_ordinal);
  ScopedShapedBuffer(const ScopedShapedBuffer&) = delete;
  void operator=(const ScopedShapedBuffer&) = delete;

  DeviceMemoryAllocator* allocator_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SHAPED_BUFFER_H_
