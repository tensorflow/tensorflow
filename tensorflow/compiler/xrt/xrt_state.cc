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

// Classes for allocating XLA literals in device memory and managing handles
// that refer to them.

#include "tensorflow/compiler/xrt/xrt_state.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xrt/xrt_memory_manager.h"

namespace tensorflow {
namespace {

// Helper typedef to make ShapeTree ForEach helper lambda signatures more
// readable. They need a type of const T& where in this case T is the
// following pointer.
typedef XRTBufferAllocation* XRTBufferAllocationPtr;

class BufferAllocStats {
 public:
  struct Stats {
    int64_t count = 0;
    int64_t size = 0;
  };

  Stats ReportAlloc(int64_t device, int64_t msize) {
    mutex_lock lock(lock_);
    Stats* device_stats = &stats_[device];
    device_stats->count += 1;
    device_stats->size += msize;
    return *device_stats;
  }

  Stats ReportFree(int64_t device, int64_t msize) {
    mutex_lock lock(lock_);
    Stats* device_stats = &stats_[device];
    device_stats->count -= 1;
    device_stats->size -= msize;
    return *device_stats;
  }

 private:
  mutable mutex lock_;
  std::map<int64_t, Stats> stats_;
};

BufferAllocStats* GetAllocStats() {
  static BufferAllocStats* stats = new BufferAllocStats();
  return stats;
}

Status AllocateScopedShapedBuffer(
    XRTMemoryManager* memory_manager, xla::Backend* backend, int device_ordinal,
    const xla::Shape& shape, std::unique_ptr<xla::ScopedShapedBuffer>* buffer,
    se::DeviceMemoryAllocator* allocator) {
  auto transfer_manager = backend->transfer_manager();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal));

  // XLA may use a different representation on device than the representation on
  // the host. XLA does not document any contract for the relationship between
  // these representations :/ Right now, the device shape is always a superset
  // of the host shape, meaning that for any valid ShapeIndex in the host shape
  // that ShapeIndex is also valid in the device shape, but not vice versa. In
  // particular, some host-side types are rewritten to be tuples. We rely on
  // this property when making sub-buffers, because we assume that if the client
  // requests the host-shape sub-buffer at index i, that will correspond to the
  // right device-shape sub-buffer at the same index.
  xla::Shape on_device_shape = transfer_manager->HostShapeToDeviceShape(shape);
  VLOG(3) << "Allocating literal buffer: host_shape="
          << xla::ShapeUtil::HumanStringWithLayout(shape) << " device_shape="
          << xla::ShapeUtil::HumanStringWithLayout(on_device_shape);

  // The ScopedShapedBuffer frees the buffers that have so far been allocated if
  // it goes out of scope. That's useful if we return early as the result of an
  // error allocating one of the later buffers.
  *buffer = absl::make_unique<xla::ScopedShapedBuffer>(
      shape, on_device_shape, allocator, device_ordinal);
  for (auto& index_to_buffer : (*buffer)->buffers()) {
    const xla::Shape& subshape =
        xla::ShapeUtil::GetSubshape(on_device_shape, index_to_buffer.first);
    uint64 size = transfer_manager->GetByteSizeRequirement(subshape);
    TF_ASSIGN_OR_RETURN(
        se::OwningDeviceMemory buffer,
        memory_manager->Allocate(backend, device_ordinal, size, allocator));
    // Move our buffer into shaped_buffer, which takes ownership of it.
    index_to_buffer.second = buffer.Release();
    VLOG(2) << "Allocated buffer at " << index_to_buffer.second.opaque()
            << " index " << index_to_buffer.first.ToString() << " (" << size
            << " bytes)";
  }

  TF_RETURN_IF_ERROR(
      transfer_manager->WriteTupleIndexTables(stream.get(), *(buffer->get())));

  return Status::OK();
}

}  // namespace

XRTBufferAllocation::XRTBufferAllocation(const se::DeviceMemoryBase& allocation,
                                         int device_ordinal,
                                         se::DeviceMemoryAllocator* allocator)
    : allocation_(allocation),
      device_ordinal_(device_ordinal),
      allocator_(allocator) {
  if (VLOG_IS_ON(2)) {
    auto stats =
        GetAllocStats()->ReportAlloc(device_ordinal_, allocation_.size());
    LOG(INFO) << "XRT Allocation Stats: device=" << device_ordinal_
              << " count=" << stats.count << " size=" << stats.size;
  }
}

XRTBufferAllocation::~XRTBufferAllocation() {
  if (VLOG_IS_ON(2)) {
    GetAllocStats()->ReportFree(device_ordinal_, allocation_.size());
  }
  // Deallocate explicitly allows allocation_ to be null.
  TF_CHECK_OK(allocator_->Deallocate(device_ordinal_, allocation_));
  VLOG(2) << "Freed buffer at " << allocation_.opaque() << " ("
          << allocation_.size() << " bytes)";
}

const se::DeviceMemoryBase& XRTBufferAllocation::allocation() {
  return allocation_;
}

XRTTupleAllocation::XRTTupleAllocation(int device_ordinal,
                                       se::DeviceMemoryAllocator* allocator,
                                       const xla::Shape& on_host_shape,
                                       const xla::Shape& on_device_shape)
    : device_ordinal_(device_ordinal),
      allocator_(allocator),
      on_host_shape_(on_host_shape),
      on_device_shape_(on_device_shape),
      buffers_(&on_device_shape_),
      pin_count_(0) {}

XRTTupleAllocation::~XRTTupleAllocation() { ReleaseBuffers(); }

void XRTTupleAllocation::ReleaseBuffers() {
  for (auto& index_buffer : buffers_) {
    if (index_buffer.second != nullptr) {
      index_buffer.second->Unref();
      index_buffer.second = nullptr;
    }
  }
}

/*static*/ Status XRTTupleAllocation::CreateAndTransfer(
    const xla::LiteralBase& literal, XRTMemoryManager* memory_manager,
    xla::Backend* backend, int device_ordinal, XRTTupleAllocation** allocation,
    se::DeviceMemoryAllocator* allocator) {
  auto transfer_manager = backend->transfer_manager();
  std::unique_ptr<xla::ScopedShapedBuffer> scoped_buffer;
  TF_RETURN_IF_ERROR(AllocateScopedShapedBuffer(memory_manager, backend,
                                                device_ordinal, literal.shape(),
                                                &scoped_buffer, allocator));
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal));
  TF_RETURN_IF_ERROR(transfer_manager->TransferLiteralToDevice(
      stream.get(), literal, *scoped_buffer));

  // By releasing the ScopedShapedBuffer we ensure that the underlying storage
  // won't be freed when the buffer goes out of scope at the end of this
  // call. To avoid a leak, there must be no error-case returns from here until
  // the end of the method.
  auto shaped_buffer = scoped_buffer->release();
  *allocation = new XRTTupleAllocation(device_ordinal, allocator,
                                       shaped_buffer.on_host_shape(),
                                       shaped_buffer.on_device_shape());
  (*allocation)
      ->InitializeFromShapedBuffer(shaped_buffer, allocator, device_ordinal);
  (*allocation)->SetDeviceMemorySize();
  return Status::OK();
}

/*static*/ Status XRTTupleAllocation::CreateUninitialized(
    const xla::Shape& shape, XRTMemoryManager* memory_manager,
    xla::Backend* backend, int device_ordinal, XRTTupleAllocation** allocation,
    se::DeviceMemoryAllocator* allocator) {
  std::unique_ptr<xla::ScopedShapedBuffer> scoped_buffer;
  TF_RETURN_IF_ERROR(AllocateScopedShapedBuffer(memory_manager, backend,
                                                device_ordinal, shape,
                                                &scoped_buffer, allocator));

  // By releasing the ScopedShapedBuffer we ensure that the underlying storage
  // won't be freed when the buffer goes out of scope at the end of this
  // call. To avoid a leak, there must be no error-case returns from here until
  // the end of the method.
  auto shaped_buffer = scoped_buffer->release();
  *allocation = new XRTTupleAllocation(device_ordinal, allocator,
                                       shaped_buffer.on_host_shape(),
                                       shaped_buffer.on_device_shape());
  (*allocation)
      ->InitializeFromShapedBuffer(shaped_buffer, allocator, device_ordinal);
  (*allocation)->SetDeviceMemorySize();
  return Status::OK();
}

/*static*/ Status XRTTupleAllocation::CreateFromBuffer(
    const xla::ShapedBuffer& shaped_buffer, const xla::Shape& on_host_shape,
    const xla::Shape& on_device_shape, xla::Backend* backend,
    int device_ordinal, XRTTupleAllocation** allocation,
    se::DeviceMemoryAllocator* allocator) {
  *allocation = new XRTTupleAllocation(device_ordinal, allocator, on_host_shape,
                                       on_device_shape);
  (*allocation)
      ->InitializeFromShapedBuffer(shaped_buffer, allocator, device_ordinal);
  (*allocation)->SetDeviceMemorySize();
  return Status::OK();
}

/*static*/ Status XRTTupleAllocation::CreateFromBuffer(
    const xla::ShapedBuffer& shaped_buffer, xla::Backend* backend,
    int device_ordinal, XRTTupleAllocation** allocation,
    se::DeviceMemoryAllocator* allocator) {
  return CreateFromBuffer(shaped_buffer, shaped_buffer.on_host_shape(),
                          shaped_buffer.on_device_shape(), backend,
                          device_ordinal, allocation, allocator);
}

Status XRTTupleAllocation::ToLiteral(xla::Backend* backend,
                                     xla::MutableLiteralBase* literal) {
  mutex_lock lock(lock_);
  return literal_ == nullptr ? StoreToLiteral(backend, literal)
                             : literal->CopyFrom(*literal_);
}

Status XRTTupleAllocation::StoreToLiteral(xla::Backend* backend,
                                          xla::MutableLiteralBase* literal) {
  auto transfer_manager = backend->transfer_manager();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal()));
  TF_ASSIGN_OR_RETURN(xla::ShapedBuffer shaped_buffer, ToShapedBuffer());
  return transfer_manager->TransferLiteralFromDevice(stream.get(),
                                                     shaped_buffer, literal);
}

Status XRTTupleAllocation::WriteLiteral(xla::Backend* backend,
                                        const xla::Literal& literal) {
  if (!xla::ShapeUtil::Equal(literal.shape(), on_host_shape())) {
    return errors::InvalidArgument(
        "New literal shape not matching the existing one: literal=",
        xla::ShapeUtil::HumanStringWithLayout(literal.shape()),
        " device=", xla::ShapeUtil::HumanStringWithLayout(on_host_shape()));
  }
  mutex_lock lock(lock_);
  if (literal_ != nullptr) {
    // The allocation is currently swapped out, and we have a host literal for
    // its content. Just update the host literal with the new value.
    return literal_->CopyFrom(literal);
  }
  TF_ASSIGN_OR_RETURN(xla::ShapedBuffer shaped_buffer, ToShapedBuffer());
  auto transfer_manager = backend->transfer_manager();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal()));
  return transfer_manager->TransferLiteralToDevice(stream.get(), literal,
                                                   shaped_buffer);
}

xla::StatusOr<bool> XRTTupleAllocation::SwapOut(xla::Backend* backend,
                                                bool swap_pinned) {
  mutex_lock lock(lock_);
  if (literal_ == nullptr && (!IsPinned() || swap_pinned)) {
    xla::Literal literal(on_host_shape());
    TF_RETURN_IF_ERROR(StoreToLiteral(backend, &literal));
    ReleaseBuffers();
    literal_ = absl::make_unique<xla::Literal>(std::move(literal));
    return true;
  }
  return false;
}

xla::StatusOr<bool> XRTTupleAllocation::SwapIn(
    XRTMemoryManager* memory_manager, xla::Backend* backend,
    se::DeviceMemoryAllocator* allocator) {
  // We need to call AllocateScopedShapedBuffer() outside the locks, since the
  // XRTMemoryManager might end up calling back into the SwapOut() API.
  // So we do a quick check before using the IsSwapped() API, and it can happen
  // that the allocation becomes swapped in after the check. This means which we
  // will end up doing an allocation, and then releasing it soon after (via its
  // scoped variables). This is an unlikely scenario (two threads calling
  // SwapIn() on the same allocation) though.
  if (!IsSwapped()) {
    return false;
  }

  auto transfer_manager = backend->transfer_manager();
  std::unique_ptr<xla::ScopedShapedBuffer> scoped_buffer;
  TF_RETURN_IF_ERROR(
      AllocateScopedShapedBuffer(memory_manager, backend, device_ordinal(),
                                 on_host_shape(), &scoped_buffer, allocator));
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal()));

  mutex_lock lock(lock_);
  if (literal_ != nullptr) {
    TF_RETURN_IF_ERROR(transfer_manager->TransferLiteralToDevice(
        stream.get(), *literal_, *scoped_buffer));

    auto shaped_buffer = scoped_buffer->release();
    InitializeFromShapedBuffer(shaped_buffer, allocator, device_ordinal());
    literal_ = nullptr;
    return true;
  }
  return false;
}

xla::StatusOr<bool> XRTTupleAllocation::PinAndSwapIn(
    XRTMemoryManager* memory_manager, xla::Backend* backend,
    se::DeviceMemoryAllocator* allocator) {
  Pin();
  return SwapIn(memory_manager, backend, allocator);
}

bool XRTTupleAllocation::IsSwapped() const {
  mutex_lock lock(lock_);
  return literal_ != nullptr;
}

int64_t XRTTupleAllocation::Pin() { return pin_count_.fetch_add(1); }

int64_t XRTTupleAllocation::Unpin() { return pin_count_.fetch_sub(1); }

bool XRTTupleAllocation::IsPinned() const { return pin_count_ != 0; }

void XRTTupleAllocation::DiscardAllocation(
    const xla::ShapeIndex& buffer_index) {
  buffers_.element(buffer_index)->DiscardAllocation();
}

const xla::Shape& XRTTupleAllocation::on_host_shape() const {
  return on_host_shape_;
}

const xla::Shape& XRTTupleAllocation::on_device_shape() const {
  return on_device_shape_;
}

int XRTTupleAllocation::device_ordinal() const { return device_ordinal_; }

const se::DeviceMemoryBase& XRTTupleAllocation::root_allocation() const {
  return buffers_.element({})->allocation();
}

/*static*/ Status XRTTupleAllocation::MakeSubBuffer(
    XRTTupleAllocation* parent, const xla::ShapeIndex& subshape,
    XRTTupleAllocation** allocation, bool alias_parent_allocation) {
  TF_ASSIGN_OR_RETURN(
      const xla::Shape* host_sub_shape,
      xla::ShapeUtil::TryGetSubshape(parent->on_host_shape(), subshape));
  TF_ASSIGN_OR_RETURN(
      const xla::Shape* device_sub_shape,
      xla::ShapeUtil::TryGetSubshape(parent->on_device_shape(), subshape));

  *allocation =
      new XRTTupleAllocation(parent->device_ordinal(), parent->allocator_,
                             *host_sub_shape, *device_sub_shape);
  if (alias_parent_allocation) {
    // Copy the subtree of allocations from the parent allocation.
    (*allocation)->buffers_.CopySubtreeFrom(parent->buffers_, subshape, {});
    // Increment the refcount on each aliased buffer.
    (*allocation)
        ->buffers_.ForEachElement(
            [](const xla::ShapeIndex& index,
               const XRTBufferAllocationPtr& buffer) { buffer->Ref(); });
  } else {
    // Find the buffers in the parent allocation that match the subtree, and
    // move the parent allocation's buffer over to the new allocation.
    (*allocation)
        ->buffers_.ForEachMutableElement(
            [&](const xla::ShapeIndex& index, XRTBufferAllocationPtr* buffer) {
              // Extend the allocation's index to the parent's frame by adding
              // subshape as a prefix.
              xla::ShapeIndex parent_index = subshape;
              for (int i = 0; i < index.size(); ++i) {
                parent_index.push_back(index[i]);
              }
              *buffer = parent->buffers_.element(parent_index);
              *parent->buffers_.mutable_element(parent_index) = nullptr;
            });
  }
  (*allocation)->SetDeviceMemorySize();
  return Status::OK();
}

void XRTTupleAllocation::SetDeviceMemorySize() {
  size_t size = 0;
  for (auto& index_buffer : buffers_) {
    if (index_buffer.second != nullptr) {
      size += index_buffer.second->allocation().size();
    }
  }
  device_memory_size_ = size;
}

/* static */ Status XRTTupleAllocation::ExpandTreeOfTuples(
    const xla::ShapeTree<ExpandedTupleInput>& elements, int device_ordinal,
    se::DeviceMemoryAllocator* allocator, xla::Shape* host_shape,
    xla::Shape* device_shape) {
  // Initialize both host and device shape to be the 'spine' of the new tuple
  // shape, given by the shape of the tree of tuples.
  *host_shape = elements.shape();
  *device_shape = elements.shape();
  // Now go over the leaves of the tree of tuples, and 'graft' the host/device
  // shapes of the allocation at that leaf onto the expanded host/device shapes
  // at the leaf position.
  TF_RETURN_IF_ERROR(elements.ForEachElementWithStatus(
      [&](const xla::ShapeIndex& index, const ExpandedTupleInput& element) {
        if (elements.IsLeaf(index)) {
          if (element.allocation == nullptr) {
            return errors::InvalidArgument(
                "MakeTuple elements has a null internal node at index ",
                index.ToString());
          }
          if (device_ordinal != element.allocation->device_ordinal() ||
              allocator != element.allocation->allocator_) {
            return errors::InvalidArgument(
                "MakeTuple elements must all be allocated on the same device "
                "as the destination.");
          }
          *xla::ShapeUtil::GetMutableSubshape(host_shape, index) =
              element.allocation->on_host_shape();
          *xla::ShapeUtil::GetMutableSubshape(device_shape, index) =
              element.allocation->on_device_shape();
        } else {
          if (element.allocation != nullptr) {
            return errors::InvalidArgument(
                "MakeTuple elements has a non-null internal node at index ",
                index.ToString());
          }
        }
        return Status::OK();
      }));
  return Status::OK();
}

/*static*/ Status XRTTupleAllocation::MakeTuple(
    XRTMemoryManager* memory_manager, xla::Backend* backend, int device_ordinal,
    const xla::ShapeTree<ExpandedTupleInput>& elements,
    XRTTupleAllocation** allocation, se::DeviceMemoryAllocator* allocator) {
  auto transfer_manager = backend->transfer_manager();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal));

  xla::Shape host_shape;
  xla::Shape device_shape;
  TF_RETURN_IF_ERROR(ExpandTreeOfTuples(elements, device_ordinal, allocator,
                                        &host_shape, &device_shape));

  // The aliasing is determined below based on whether or not all the inputs are
  // released while being transferred. allocation_tmp is a local pointer that is
  // copied to *allocation at the end only if the method succeeds.
  XRTTupleAllocation* allocation_tmp = new XRTTupleAllocation(
      device_ordinal, allocator, host_shape, device_shape);
  core::ScopedUnref allocation_unref(allocation_tmp);
  // First allocate device memory for the new tuple index tables, one at each
  // internal node of the elements tree. Do this in a separate pass into a
  // ScopedShapedBuffer so that it's easy to free the newly-allocated memory if
  // an allocation fails. Make sure the shape has layout so that the code that
  // writes index tables will be happy lower down.
  xla::Shape spine_shape = elements.shape();
  xla::LayoutUtil::SetToDefaultLayout(&spine_shape);
  auto new_tuple_buffers = absl::make_unique<xla::ScopedShapedBuffer>(
      spine_shape, spine_shape, allocator, device_ordinal);
  TF_RETURN_IF_ERROR(elements.ForEachElementWithStatus(
      [&](const xla::ShapeIndex& index, const ExpandedTupleInput& element) {
        if (!elements.IsLeaf(index)) {
          const xla::Shape& subshape =
              xla::ShapeUtil::GetSubshape(device_shape, index);
          uint64 size = transfer_manager->GetByteSizeRequirement(subshape);
          TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory buffer,
                              memory_manager->Allocate(backend, device_ordinal,
                                                       size, allocator));
          VLOG(2) << "Allocated buffer at " << buffer->opaque() << " index "
                  << index.ToString();
          // Move the new buffer into new_tuple_buffers, which takes ownership
          // of it.
          new_tuple_buffers->set_buffer(std::move(buffer), index);
        }
        return Status::OK();
      }));
  // Transfer from the ScopedShapedBuffer to a ShapedBuffer, which does not own
  // the newly-allocated index tables. Right now there's no owner for the new
  // index tables, so next we will transfer ownership to the new allocation,
  // taking care not to return early on any errors in the meantime.
  xla::ShapedBuffer tuple_buffers = new_tuple_buffers->release();
  // Now fill in the remaining datastructures. After this ForEachElement
  // completes:
  //   1) Every leaf element of tuple_buffers will be the root buffer of
  //      an existing allocation, and every internal element of tuple_buffers
  //      will be a newly-allocated index table. tuple_buffers does not own any
  //      of these.
  //   2) Every element of allocation_tmp->buffers_ will be a correctly
  //   constructed
  //      XRTBufferAllocation wrapping the necessary allocations. For buffers in
  //      existing allocations there will be a new reference owned by the new
  //      allocation, and for newly-allocated index tables there will be a
  //      single reference owned by the new allocation.
  elements.ForEachElement([&](const xla::ShapeIndex& index,
                              const ExpandedTupleInput& element) {
    if (elements.IsLeaf(index)) {
      allocation_tmp->buffers_.CopySubtreeFrom(element.allocation->buffers_, {},
                                               index);
      tuple_buffers.set_buffer(element.allocation->root_allocation(), index);
      if (element.release_allocation_after_use) {
        // Transfer the references from element's buffers to the new allocation
        // rather than incrementing the refcount. The caller should have
        // validated that release_allocation_after_use is false if
        // element.allocation appears in more than one leaf.
        element.allocation->buffers_.ForEachMutableElement(
            [&](const xla::ShapeIndex&, XRTBufferAllocationPtr* buffer) {
              *buffer = nullptr;
            });
      } else {
        // Increment the refcount on each newly-aliased buffer.
        element.allocation->buffers_.ForEachElement(
            [](const xla::ShapeIndex& index,
               const XRTBufferAllocationPtr& buffer) { buffer->Ref(); });
      }
    } else {
      // This is an internal node of the tuple tree so take ownership of the
      // newly-created index table.
      *allocation_tmp->buffers_.mutable_element(index) =
          new XRTBufferAllocation(tuple_buffers.buffer(index), device_ordinal,
                                  allocator);
    }
  });
  allocation_tmp->SetDeviceMemorySize();
  // Because the internal nodes of tuple_buffers are exactly the new index
  // tables, WriteTupleIndexTables will write only the new index tables and not
  // rewrite the index tables for the existing allocations.
  TF_RETURN_IF_ERROR(
      transfer_manager->WriteTupleIndexTables(stream.get(), tuple_buffers));

  *allocation = allocation_tmp;
  // Get another reference since allocation_tmp will be Unrefed automatically on
  // exit.
  (*allocation)->Ref();
  return Status::OK();
}

bool XRTTupleAllocation::IsExclusiveOwner() const {
  for (const auto& index_buffer : buffers_) {
    if (index_buffer.second != nullptr &&
        !index_buffer.second->RefCountIsOne()) {
      return false;
    }
  }
  return true;
}

size_t XRTTupleAllocation::GetDeviceMemorySize() const {
  return device_memory_size_;
}

void XRTTupleAllocation::InitializeFromShapedBuffer(
    const xla::ShapedBuffer& shaped_buffer,
    se::DeviceMemoryAllocator* allocator, int device_ordinal) {
  for (auto& index_buffer : buffers_) {
    if (index_buffer.second != nullptr) {
      index_buffer.second->Unref();
    }
    // Make a reference-counted version of the allocated buffer.
    index_buffer.second = new XRTBufferAllocation(
        shaped_buffer.buffer(index_buffer.first), device_ordinal, allocator);
  }
}

xla::StatusOr<xla::ShapedBuffer> XRTTupleAllocation::ToShapedBuffer() {
  xla::ShapedBuffer shaped_buffer(on_host_shape(), on_device_shape(),
                                  device_ordinal_);
  for (const auto& index_buffer : buffers_) {
    if (index_buffer.second == nullptr ||
        (index_buffer.second->allocation().is_null() &&
         index_buffer.second->allocation().size() > 0)) {
      return errors::InvalidArgument("Literal buffer at index ",
                                     index_buffer.first.ToString(),
                                     " has been released");
    }
    shaped_buffer.set_buffer(index_buffer.second->allocation(),
                             index_buffer.first);
  }
  return std::move(shaped_buffer);
}

Status XRTTupleAllocation::AliasBufferFrom(const XRTTupleAllocation& source,
                                           const xla::ShapeIndex& source_index,
                                           const xla::ShapeIndex& dest_index) {
  XRTBufferAllocation* source_buffer = source.buffers_.element(source_index);
  XRTBufferAllocation* dest_buffer = buffers_.element(dest_index);
  if (dest_buffer != nullptr) {
    // We allow the destination size being zero, because there are cases where
    // we are coming in later filling in null/uninitialized device buffers. In
    // all other cases, the size of the new buffer must match.
    if (source_buffer->allocation().size() !=
            dest_buffer->allocation().size() &&
        dest_buffer->allocation().size() != 0) {
      return errors::InvalidArgument(
          "Source buffer at index ", source_index.ToString(),
          " does not match the size of destination buffer at index ",
          dest_index.ToString(), ": ", source_buffer->allocation().size(),
          " vs ", dest_buffer->allocation().size());
    }
  } else {
    const xla::Shape& source_subshape =
        xla::ShapeUtil::GetSubshape(source.on_device_shape(), source_index);
    const xla::Shape& dest_subshape =
        xla::ShapeUtil::GetSubshape(on_device_shape(), dest_index);
    if (!xla::ShapeUtil::Equal(source_subshape, dest_subshape)) {
      return errors::InvalidArgument(
          "Source and destination subshapes do not match: source=",
          xla::ShapeUtil::HumanStringWithLayout(source_subshape),
          " dest=", xla::ShapeUtil::HumanStringWithLayout(dest_subshape));
    }
  }
  *buffers_.mutable_element(dest_index) = source_buffer;
  source_buffer->Ref();
  if (dest_buffer != nullptr) {
    // If we handed over the ownership of a buffer in ToExecutionInput(), we
    // will be called here on the way back from execution, to alias back the
    // buffer at that index. In that case the buffers will be the same. So we
    // need to discard the memory at the destination buffer, before releasing
    // the reference.
    if (dest_buffer->allocation().IsSameAs(source_buffer->allocation()) &&
        dest_buffer != source_buffer) {
      dest_buffer->DiscardAllocation();
    }
    dest_buffer->Unref();
  }
  return Status::OK();
}

xla::StatusOr<xla::ExecutionInput> XRTTupleAllocation::ToExecutionInput(
    const std::function<xla::StatusOr<bool>(const xla::ShapeIndex&)>&
        alias_checker) {
  xla::ExecutionInput result(on_device_shape(), on_host_shape());
  for (const auto& index_buffer : buffers_) {
    if (index_buffer.second == nullptr ||
        (index_buffer.second->allocation().is_null() &&
         index_buffer.second->allocation().size() > 0)) {
      return errors::InvalidArgument("Literal buffer at index ",
                                     index_buffer.first.ToString(),
                                     " has been released");
    }
    TF_ASSIGN_OR_RETURN(bool should_alias, alias_checker(index_buffer.first));
    if (!should_alias) {
      result.SetBuffer(
          index_buffer.first,
          xla::MaybeOwningDeviceMemory(index_buffer.second->allocation()));
    } else {
      // We keep the ownership of the device memory here.
      result.SetUnownedBuffer(
          index_buffer.first,
          xla::MaybeOwningDeviceMemory(se::OwningDeviceMemory(
              index_buffer.second->allocation(), device_ordinal_, allocator_)));
    }
  }
  return std::move(result);
}

}  // namespace tensorflow
