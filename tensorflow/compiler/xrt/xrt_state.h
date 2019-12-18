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

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt_refptr.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

// Cannot include xrt_memory_manager.h here, as it needs to include this file.
class XRTMemoryManager;

// TODO(misard) make this a Tensor if and when that makes sense.
// A reference-counted wrapper around a buffer allocation. This maps an XLA
// tuple index or a non-tuple XLA shape to a region of device memory. The device
// memory buffer is freed when the reference count drops to zero.
class XRTBufferAllocation : public core::RefCounted {
 public:
  XRTBufferAllocation(const se::DeviceMemoryBase& allocation,
                      int device_ordinal, se::DeviceMemoryAllocator* allocator);
  ~XRTBufferAllocation() override;

  // The region of device memory being wrapped.
  const se::DeviceMemoryBase& allocation();

  void DiscardAllocation() { allocation_ = se::DeviceMemoryBase(); }

 private:
  se::DeviceMemoryBase allocation_;
  int device_ordinal_;
  se::DeviceMemoryAllocator* allocator_;
};

// A XRTTupleAllocation represents an allocated memory area on the device.
// New tuples can be created in three ways: by passing a literal in which case
// device memory is allocated and the literal is transferred to that memory; by
// aliasing a sub-shape of an existing tuple-shaped handle; or by aliasing a
// vector of existing handles to create a new tuple. The underlying storage is
// reference-counted. When a handle is released, the reference count of each
// storage buffer is decremented, and buffers with no outstanding references are
// freed.
class XRTTupleAllocation : public core::RefCounted {
 public:
  ~XRTTupleAllocation() override;

  // Allocates new device memory buffers sufficient to store literal, transfers
  // literal to that memory, and returns a XRTTupleAllocation handle to the
  // allocated buffers.
  static Status CreateAndTransfer(const xla::LiteralBase& literal,
                                  XRTMemoryManager* memory_manager,
                                  xla::Backend* backend, int device_ordinal,
                                  XRTTupleAllocation** allocation);

  // Allocates new device memory buffers sufficient to store a tensor of
  // the specified shape, and returns a XRTTupleAllocation handle to the
  // allocated buffers.  The allocated buffers are not initialized.
  static Status CreateUninitialized(const xla::Shape& shape,
                                    XRTMemoryManager* memory_manager,
                                    xla::Backend* backend, int device_ordinal,
                                    XRTTupleAllocation** allocation);

  // Wraps an existing ShapeBuffer in a new XRTTupleAllocation handle.
  static Status CreateFromBuffer(const xla::ShapedBuffer& shaped_buffer,
                                 xla::Backend* backend, int device_ordinal,
                                 XRTTupleAllocation** allocation);

  // Same as the CreateFromBuffer() API above, but with the shapes being passed
  // as input. This API is used when creating tuple allocations with the output
  // of XLA computations which emit dynamic shaped output via the output shape
  // table.
  static Status CreateFromBuffer(const xla::ShapedBuffer& shaped_buffer,
                                 const xla::Shape& on_host_shape,
                                 const xla::Shape& on_device_shape,
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
    RefPtr<XRTTupleAllocation> allocation;
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
  static Status MakeTuple(XRTMemoryManager* memory_manager,
                          xla::Backend* backend, int device_ordinal,
                          const xla::ShapeTree<ExpandedTupleInput>& elements,
                          XRTTupleAllocation** allocation);

  // Copies the allocation from device to host and returns it in literal.
  Status ToLiteral(xla::Backend* backend, xla::MutableLiteralBase* literal);

  // Write a new literal value to the allocation.
  Status WriteLiteral(xla::Backend* backend, const xla::Literal& literal);

  // Stores the content of the tuple allocation into the internal literal, and
  // releases all the device buffers. The swap_pinned flag tells whether a
  // pinned allocation should be swapped out. It should be false on all cases,
  // but during the memory compaction operation from the XRTMemoryManager.
  // Returns a boolean telling whether the allocation was swapped out.
  xla::StatusOr<bool> SwapOut(xla::Backend* backend, bool swap_pinned);

  // Allocates the device memory required to store the tuple value held within
  // the internal literal, and transfer the literal value into the device
  // memory. Returns a boolean telling whether the allocation was swapped in.
  xla::StatusOr<bool> SwapIn(XRTMemoryManager* memory_manager,
                             xla::Backend* backend);

  // Pins the allocation first, then swap it in (if it is not already). After
  // this API returns, the allocation is pinned and its content on device
  // memory. The caller is responsible for releasing the pin-count using the
  // Unpin() API.
  xla::StatusOr<bool> PinAndSwapIn(XRTMemoryManager* memory_manager,
                                   xla::Backend* backend);

  // Checks whether the allocation is currently swapped out.
  bool IsSwapped() const;

  // Increases the pin-count of this allocation. If the pin-count is greater
  // than 0, the allocation cannot be swapped. Returned the pin-count value
  // before the increase.
  int64 Pin();

  // Decreases the pin-count of this allocation. Returned the pin-count value
  // before the decrease.
  int64 Unpin();

  // Checks whether the allocation is currently pinned.
  bool IsPinned() const;

  // True if none of the buffers in the allocation are aliased by any other live
  // handle.
  bool IsExclusiveOwner() const;

  // Retrieves the footprint in terms of device memory, of this allocation.
  size_t GetDeviceMemorySize() const;

  // The ordinal of the device holding this tuple.
  int device_ordinal() const;

  // Returns the shape of the tuple as seen by the host.
  const xla::Shape& on_host_shape() const;

  // Returns the shape of the tuple as stored on the device.
  const xla::Shape& on_device_shape() const;

  // Returns the buffer pointed to by the root of the tuple.
  const se::DeviceMemoryBase& root_allocation() const;

  // Stops managing the storage for the allocation at buffer_index, e.g.,
  // because it has been aliased to the output buffer of a computation.
  void DiscardAllocation(const xla::ShapeIndex& buffer_index);

  // Returns the tree of allocations as a ShapedBuffer. This tree may not have
  // the same shape as on_host_shape.
  xla::StatusOr<xla::ShapedBuffer> ToShapedBuffer();

  // Aliases the source buffer at source_index into the current tuple allocation
  // dest_index.
  Status AliasBufferFrom(const XRTTupleAllocation& source,
                         const xla::ShapeIndex& source_index,
                         const xla::ShapeIndex& dest_index);

  // Returns the device memory tree of this allocation. If the release_checker
  // function returns true for a given index, an owned device memory is returned
  // to the caller. But the tuple allocation cannot release the ownership in
  // full, as the execute operation might fail. So we rely on a call to
  // AliasBufferFrom() to re-alias back the buffers. This is not great (to say
  // the least), but the current aliasing logic relies on
  // MaybeOwningDeviceMemory being owned, to detect the fact that the user may
  // want to alias a buffer. Unfortunately to do that, it needs to release the
  // ownership, which is a problem if the execute will fail.
  // This calls for a refactoring of the whole owning/maybe-owning interface to
  // introduce a sharing concept (IOW shared_ptr model vs. unique_ptr).
  // We'd need something similar to XRTTupleAllocation instead of
  // ScopedShapedBuffer, which wants ownership and does not allow sharing.
  xla::StatusOr<xla::ShapeTree<xla::MaybeOwningDeviceMemory>>
  ToDeviceMemoryTree(
      const std::function<xla::StatusOr<bool>(const xla::ShapeIndex&)>&
          release_checker);

 private:
  // Creates a new handle with (tuple) shape.
  XRTTupleAllocation(int device_ordinal, se::DeviceMemoryAllocator* allocator,
                     const xla::Shape& on_host_shape,
                     const xla::Shape& on_device_shape);

  // Inherits the allocations represented in buffer, which must have the same
  // shape as buffers_.
  void InitializeFromShapedBuffer(const xla::ShapedBuffer& shaped_buffer,
                                  se::DeviceMemoryAllocator* allocator,
                                  int device_ordinal);

  // Releases all the XRTBufferAllocation buffer references and set the
  // corresponding shape tree entry to nullptr.
  void ReleaseBuffers();

  // Stores the content of the allocation from device memory to the target host
  // literal.
  Status StoreToLiteral(xla::Backend* backend,
                        xla::MutableLiteralBase* literal);

  // Sets the total size of the buffers held within this allocation buffers.
  // This API should be called once when an XRTTupleAllocation object is
  // created, as the XRTTupleAllocation shapes never change, and hence the
  // device memory size.
  void SetDeviceMemorySize();

  // Takes a tree 'elements' where each leaf is an allocation, validates that
  // they are all on device_ordinal managed by allocator, and returns in
  // host_shape and device_shape the host/device shapes of the expanded tree,
  // where at each leaf of elements the shape of the allocation at elements is
  // grafted on.
  static Status ExpandTreeOfTuples(
      const xla::ShapeTree<ExpandedTupleInput>& elements, int device_ordinal,
      se::DeviceMemoryAllocator* allocator, xla::Shape* host_shape,
      xla::Shape* device_shape);

  // The lock which protects the internal operations of the tuple allocation. Is
  // mutable to allow const-like operations to be declared as such.
  mutable mutex lock_;

  // Location of the memory that is being managed.
  const int device_ordinal_;
  se::DeviceMemoryAllocator* const allocator_;

  // The shape that the caller thinks the tuple has.
  const xla::Shape on_host_shape_;
  // The shape that the tuple has on device. Store this explicitly instead of
  // using a shape stored in ShapeTree because ShapeTree discards the layout.
  const xla::Shape on_device_shape_;
  // The tree of reference-counted buffers, which uses on_device_shape_ as its
  // shape.
  xla::ShapeTree<XRTBufferAllocation*> buffers_;
  // The footprint of the allocation, when residing on device memory.
  size_t device_memory_size_ = 0;
  // If the allocation is swapped out, this is the literal storing its content.
  std::unique_ptr<xla::Literal> literal_;
  // A pinned allocation is one which cannot be swapped out. If pin_count_ > 0
  // then the allocation is pinned.
  std::atomic<int64> pin_count_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_STATE_H_
