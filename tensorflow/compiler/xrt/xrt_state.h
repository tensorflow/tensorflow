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

// Classes for keeping track of on-device state.

#ifndef TENSORFLOW_COMPILER_XRT_XRT_STATE_H_
#define TENSORFLOW_COMPILER_XRT_XRT_STATE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

// TODO(misard) make this a Tensor if and when that makes sense.
// A reference-counted wrapper around a buffer allocation. This maps an XLA
// tuple index or a non-tuple XLA shape to a region of device memory. The device
// memory buffer is freed when the reference count drops to zero.
class XRTBufferAllocation : public core::RefCounted {
 public:
  XRTBufferAllocation(const se::DeviceMemoryBase& allocation,
                      int device_ordinal,
                      xla::DeviceMemoryAllocator* allocator);
  ~XRTBufferAllocation() override;

  // The region of device memory being wrapped.
  const se::DeviceMemoryBase& allocation();

  // Sets the DeviceMemoryBase to be null. DiscardAllocation should be called
  // when ownership of the underlying buffer has been transferred, e.g., to an
  // output buffer when input and output buffers are aliased during
  // execution. The call to DiscardAllocation prevents any device buffer being
  // freed when the reference count drops to zero.
  void DiscardAllocation();

 private:
  se::DeviceMemoryBase allocation_;
  int device_ordinal_;
  xla::DeviceMemoryAllocator* allocator_;
};

// Entry in the resource manager corresponding to an allocation handle returned
// to a client. The handle identifies an immutable tuple of data in device
// memory. New handles can be created in three ways: by passing a literal in
// which case device memory is allocated and the literal is transferred to that
// memory; by aliasing a sub-shape of an existing tuple-shaped handle; or by
// aliasing a vector of existing handles to create a new tuple. The underlying
// storage is reference-counted. When a handle is released, the reference count
// of each storage buffer is decremented, and buffers with no outstanding
// references are freed.
class XRTTupleAllocation : public ResourceBase {
 public:
  ~XRTTupleAllocation() override;

  // Allocates new device memory buffers sufficient to store literal, transfers
  // literal to that memory, and returns a XRTTupleAllocation handle to the
  // allocated buffers.
  static Status CreateAndTransfer(const xla::Literal& literal,
                                  xla::Backend* backend, int device_ordinal,
                                  XRTTupleAllocation** allocation);

  // Wraps an existing ShapeBuffer in a new XRTTupleAllocation handle.
  static Status CreateFromBuffer(const xla::ShapedBuffer& shaped_buffer,
                                 xla::Backend* backend, int device_ordinal,
                                 XRTTupleAllocation** allocation);

  // Aliases a sub-shape of parent and returns a XRTTupleAllocation handle
  // to the sub-shape. If alias_base_allocation is true, the buffers in the
  // sub-shape will be shared between parent and the returned allocation,
  // otherwise the overlapping buffers in parent will be replaced by
  // nullptr.
  static Status MakeSubBuffer(XRTTupleAllocation* parent,
                              const xla::ShapeIndex& subshape,
                              XRTTupleAllocation** allocation,
                              bool alias_parent_allocation);

  // A structure describing a leaf of a tree of tuples to expand. Each leaf
  // contains an allocation and indicates whether or not the allocation's handle
  // should be freed after incorporating its buffers into the expanded tree.
  struct ExpandedTupleInput {
    XRTTupleAllocation* allocation;
    bool release_allocation_after_use;
  };

  // Returns a handle to a new tuple where the subtree of the new tuple at an
  // index corresponding to a leaf of 'elements' is constructed from the
  // allocation (i.e., a tuple or array) pointed to by that leaf. If
  // release_allocation_after_use is false at a leaf, the new tuple will alias
  // the input allocation at that leaf, otherwise the input allocation will be
  // released. Input allocations may be repeated (appear in more than one leaf)
  // in which case the corresponding buffers in the output tuple will alias. If
  // an input is repeated, release_input_handle must be false for every leaf
  // where that input appears. The latter property is not validated by MakeTuple
  // and must be enforced by the caller.
  static Status MakeTuple(xla::Backend* backend, int device_ordinal,
                          const xla::ShapeTree<ExpandedTupleInput>& elements,
                          XRTTupleAllocation** allocation);

  // Retrieves the allocation interned under key from rm. The caller owns a
  // reference to allocation after looking it up.
  static Status Lookup(ResourceMgr* rm, int64 key,
                       XRTTupleAllocation** allocation);

  // Deletes the reference in the rm to an allocation interned under key.
  static Status DeleteFromResourceManager(ResourceMgr* rm, int64 key);

  // Adds the allocation to a ResourceMgr and returns the key that will be used
  // to retrieve it. Transfers a reference on *this to rm.
  Status Intern(ResourceMgr* rm, int64* key);

  // Copies the allocation from device to host and returns it in literal.
  Status ToLiteral(xla::Backend* backend, int device_ordinal,
                   xla::Literal* literal);

  // Write a new literal value to the allocation.
  Status WriteLiteral(xla::Backend* backend, const xla::Literal& literal);

  // True if none of the buffers in the allocation are aliased by any other live
  // handle.
  bool IsExclusiveOwner();

  // The ordinal of the device holding this tuple.
  int device_ordinal();

  // Returns the shape of the tuple as seen by the host.
  const xla::Shape& on_host_shape();

  // Returns the shape of the tuple as stored on the device.
  const xla::Shape& on_device_shape();

  // Returns the buffer pointed to by the root of the tuple.
  const se::DeviceMemoryBase& root_allocation();

  // Stops managing the storage for the allocation at buffer_index, e.g.,
  // because it has been aliased to the output buffer of a computation.
  void DiscardAllocation(const xla::ShapeIndex& buffer_index);

  // Returns the tree of allocations as a ShapedBuffer. This tree may not have
  // the same shape as on_host_shape.
  xla::ShapedBuffer ToShapedBuffer();

  // Returns the device memory tree of this allocation. If 'release' is set, the
  // ownership of the device memory is transferred to the result.
  xla::ShapeTree<xla::MaybeOwningDeviceMemory> ToDeviceMemoryTree(bool release);

  string DebugString() override { return "XLA allocation handle"; }

 private:
  // Creates a new handle with (tuple) shape.
  XRTTupleAllocation(int device_ordinal, xla::DeviceMemoryAllocator* allocator,
                     const xla::Shape& on_host_shape,
                     const xla::Shape& on_device_shape);

  // Inherits the allocations represented in buffer, which must have the same
  // shape as buffers_.
  void InitializeFromShapedBuffer(const xla::ShapedBuffer& shaped_buffer,
                                  xla::DeviceMemoryAllocator* allocator,
                                  int device_ordinal);

  // Takes a tree 'elements' where each leaf is an allocation, validates that
  // they are all on device_ordinal managed by allocator, and returns in
  // host_shape and device_shape the host/device shapes of the expanded tree,
  // where at each leaf of elements the shape of the allocation at elements is
  // grafted on.
  static Status ExpandTreeOfTuples(
      const xla::ShapeTree<ExpandedTupleInput>& elements, int device_ordinal,
      xla::DeviceMemoryAllocator* allocator, xla::Shape* host_shape,
      xla::Shape* device_shape);

  // Location of the memory that is being managed.
  int device_ordinal_;
  xla::DeviceMemoryAllocator* allocator_;

  // The shape that the caller thinks the tuple has.
  const xla::Shape on_host_shape_;
  // The shape that the tuple has on device. Store this explicitly instead of
  // using a shape stored in ShapeTree because ShapeTree discards the layout.
  const xla::Shape on_device_shape_;
  // The tree of reference-counted buffers, which uses on_device_shape_ as its
  // shape.
  xla::ShapeTree<XRTBufferAllocation*> buffers_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_STATE_H_
