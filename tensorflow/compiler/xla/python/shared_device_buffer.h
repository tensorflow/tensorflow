/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_SHARED_DEVICE_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_SHARED_DEVICE_BUFFER_H_

#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/owning_device_memory.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {

// Class that represents a node in a reference-counted DAG of device buffers.
// Unlike a ShapedBuffer, which owns none of its buffers, and
// ScopedShapedBuffer, which owns an entire buffer tree, the reference counting
// in a PySharedDeviceBuffer DAG is done at the level of individual device
// buffers. Reference counting buffer individually is more convenient when
// manipulating on-device tuples where a tuple and its elements may have
// different lifetimes.
class PySharedDeviceBuffer {
 public:
  // Converts a ScopedShapedBuffer into a Buffer tree. Takes ownership of the
  // contents of the shaped_buffer.
  static std::shared_ptr<PySharedDeviceBuffer> FromScopedShapedBuffer(
      ScopedShapedBuffer shaped_buffer);

  // Makes a tuple buffer. Does not initialize the tuple table.
  static StatusOr<std::shared_ptr<PySharedDeviceBuffer>> MakeTuple(
      std::vector<std::shared_ptr<PySharedDeviceBuffer>> children,
      TransferManager* transfer_manager, DeviceMemoryAllocator* allocator,
      int device_ordinal);

  // Makes an uninitialized array buffer.
  static StatusOr<std::shared_ptr<PySharedDeviceBuffer>> MakeArray(
      Shape on_device_shape, TransferManager* transfer_manager,
      DeviceMemoryAllocator* allocator, int device_ordinal);

  // Builds a ShapedBuffer view onto the buffers of 'tree'. Since
  // PySharedDeviceBuffer does not maintain the on-host shape, the caller must
  // provide it. We require but do not verify that
  // TransferManager::HostShapeToDeviceShape(on_host_shape) == on_device_shape()
  ShapedBuffer AsShapedBuffer(const Shape& on_host_shape) const;

  const Shape& on_device_shape() const { return on_device_shape_; }
  const std::vector<std::shared_ptr<PySharedDeviceBuffer>>& children() const {
    return children_;
  }
  const OwningDeviceMemory& device_memory() const { return device_memory_; }

  PySharedDeviceBuffer() = default;
  PySharedDeviceBuffer(
      Shape on_device_shape, OwningDeviceMemory device_memory,
      std::vector<std::shared_ptr<PySharedDeviceBuffer>> children);

 private:
  // We only represent the on-device shape. The on-host shape may not be
  // one-to-one with the tree of device buffers, so to avoid representational
  // awkwardness we maintain on-host shapes separately.
  Shape on_device_shape_;
  OwningDeviceMemory device_memory_;
  std::vector<std::shared_ptr<PySharedDeviceBuffer>> children_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_SHARED_DEVICE_BUFFER_H_
