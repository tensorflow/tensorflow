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

#include <stdint.h>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

namespace {

class BufferAllocStats {
 public:
  struct Stats {
    int64 count = 0;
    int64 size = 0;
  };

  Stats ReportAlloc(int64 device, int64 msize) {
    mutex_lock lock(lock_);
    Stats* device_stats = &stats_[device];
    device_stats->count += 1;
    device_stats->size += msize;
    return *device_stats;
  }

  Stats ReportFree(int64 device, int64 msize) {
    mutex_lock lock(lock_);
    Stats* device_stats = &stats_[device];
    device_stats->count -= 1;
    device_stats->size -= msize;
    return *device_stats;
  }

 private:
  mutable mutex lock_;
  std::map<int64, Stats> stats_;
};

const char* kTupleContainer = "tuples";

int64 get_uid() {
  uint64 unsigned_rand = random::New64() & INT64_MAX;
  return static_cast<int64>(unsigned_rand);
}

BufferAllocStats* GetAllocStats() {
  static BufferAllocStats* stats = new BufferAllocStats();
  return stats;
}

Status AllocateScopedShapedBuffer(
    xla::Backend* backend, int device_ordinal, const xla::Shape& shape,
    std::unique_ptr<xla::ScopedShapedBuffer>* buffer) {
  auto transfer_manager = backend->transfer_manager();
  auto allocator = backend->memory_allocator();
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
    xla::Shape subshape =
        xla::ShapeUtil::GetSubshape(on_device_shape, index_to_buffer.first);
    uint64 size = transfer_manager->GetByteSizeRequirement(subshape);
    TF_ASSIGN_OR_RETURN(
        xla::OwningDeviceMemory buffer,
        allocator->Allocate(device_ordinal, size, /*retry_on_failure=*/false));
    // Move our buffer into shaped_buffer, which takes ownership of it.
    index_to_buffer.second = buffer.Forget();
    VLOG(2) << "Allocated buffer at " << index_to_buffer.second.opaque()
            << " index " << index_to_buffer.first.ToString();
  }

  TF_RETURN_IF_ERROR(
      transfer_manager->WriteTupleIndexTables(stream.get(), *(buffer->get())));

  return Status::OK();
}

}  // namespace

XRTBufferAllocation::XRTBufferAllocation(const se::DeviceMemoryBase& allocation,
                                         int device_ordinal,
                                         xla::DeviceMemoryAllocator* allocator)
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
  Status s = allocator_->Deallocate(device_ordinal_, allocation_);
  // Nothing to do but check fail here if memory datastructures are corrupted.
  CHECK(s.ok());
  VLOG(2) << "Freed buffer at " << allocation_.opaque();
}

const se::DeviceMemoryBase& XRTBufferAllocation::allocation() {
  return allocation_;
}

void XRTBufferAllocation::DiscardAllocation() {
  // Replace the allocation with a null.
  allocation_ = se::DeviceMemoryBase();
}

XRTTupleAllocation::XRTTupleAllocation(int device_ordinal,
                                       xla::DeviceMemoryAllocator* allocator,
                                       const xla::Shape& on_host_shape,
                                       const xla::Shape& on_device_shape)
    : device_ordinal_(device_ordinal),
      allocator_(allocator),
      on_host_shape_(on_host_shape),
      on_device_shape_(on_device_shape),
      buffers_(&on_device_shape_) {}

XRTTupleAllocation::~XRTTupleAllocation() {
  for (auto& buffer : buffers_) {
    buffer.second->Unref();
  }
}

/*static*/ Status XRTTupleAllocation::CreateAndTransfer(
    const xla::Literal& literal, xla::Backend* backend, int device_ordinal,
    XRTTupleAllocation** allocation) {
  auto transfer_manager = backend->transfer_manager();
  auto allocator = backend->memory_allocator();

  std::unique_ptr<xla::ScopedShapedBuffer> scoped_buffer;
  TF_RETURN_IF_ERROR(AllocateScopedShapedBuffer(
      backend, device_ordinal, literal.shape(), &scoped_buffer));
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
  return Status::OK();
}

/*static*/ Status XRTTupleAllocation::CreateFromBuffer(
    const xla::ShapedBuffer& shaped_buffer, xla::Backend* backend,
    int device_ordinal, XRTTupleAllocation** allocation) {
  auto allocator = backend->memory_allocator();

  *allocation = new XRTTupleAllocation(device_ordinal, allocator,
                                       shaped_buffer.on_host_shape(),
                                       shaped_buffer.on_device_shape());
  (*allocation)
      ->InitializeFromShapedBuffer(shaped_buffer, allocator, device_ordinal);
  return Status::OK();
}

Status XRTTupleAllocation::ToLiteral(xla::Backend* backend, int device_ordinal,
                                     xla::Literal* literal) {
  auto transfer_manager = backend->transfer_manager();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal));
  TF_ASSIGN_OR_RETURN(*literal, transfer_manager->TransferLiteralFromDevice(
                                    stream.get(), ToShapedBuffer()));
  return Status::OK();
}

Status XRTTupleAllocation::WriteLiteral(xla::Backend* backend,
                                        const xla::Literal& literal) {
  if (!xla::ShapeUtil::Equal(literal.shape(), on_host_shape())) {
    return errors::InvalidArgument(
        "New literal shape not matching the existing one: literal=",
        xla::ShapeUtil::HumanStringWithLayout(literal.shape()),
        " device=", xla::ShapeUtil::HumanStringWithLayout(on_host_shape()));
  }
  auto transfer_manager = backend->transfer_manager();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal()));
  return transfer_manager->TransferLiteralToDevice(stream.get(), literal,
                                                   ToShapedBuffer());
}

void XRTTupleAllocation::DiscardAllocation(
    const xla::ShapeIndex& buffer_index) {
  buffers_.element(buffer_index)->DiscardAllocation();
}

const xla::Shape& XRTTupleAllocation::on_host_shape() { return on_host_shape_; }

const xla::Shape& XRTTupleAllocation::on_device_shape() {
  return on_device_shape_;
}

int XRTTupleAllocation::device_ordinal() { return device_ordinal_; }

const se::DeviceMemoryBase& XRTTupleAllocation::root_allocation() {
  return buffers_.element({})->allocation();
}

/*static*/ Status XRTTupleAllocation::Lookup(ResourceMgr* rm, int64 key,
                                             XRTTupleAllocation** allocation) {
  string key_string = absl::StrCat(key);
  TF_RETURN_IF_ERROR(rm->Lookup(kTupleContainer, key_string, allocation));
  return Status::OK();
}

/*static*/ Status XRTTupleAllocation::DeleteFromResourceManager(ResourceMgr* rm,
                                                                int64 key) {
  string key_string = absl::StrCat(key);
  return rm->Delete<XRTTupleAllocation>(kTupleContainer, key_string);
}

/* static */ Status XRTTupleAllocation::ReleaseAllAllocations(ResourceMgr* rm) {
  VLOG(1) << "Releasing all XRT held device memory";
  return rm->Cleanup(kTupleContainer);
}

// Helper typedef to make ShapeTree ForEach helper lambda signatures more
// readable. They need a type of const T& where in this case T is the
// following pointer.
typedef XRTBufferAllocation* XRTBufferAllocationPtr;

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
              *parent->buffers_.mutable_element(parent_index) =
                  new XRTBufferAllocation(se::DeviceMemoryBase(),
                                          parent->device_ordinal(),
                                          parent->allocator_);
            });
  }

  return Status::OK();
}

/* static */ Status XRTTupleAllocation::ExpandTreeOfTuples(
    const xla::ShapeTree<ExpandedTupleInput>& elements, int device_ordinal,
    xla::DeviceMemoryAllocator* allocator, xla::Shape* host_shape,
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
    xla::Backend* backend, int device_ordinal,
    const xla::ShapeTree<ExpandedTupleInput>& elements,
    XRTTupleAllocation** allocation) {
  auto transfer_manager = backend->transfer_manager();
  auto allocator = backend->memory_allocator();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal));

  xla::Shape host_shape;
  xla::Shape device_shape;
  TF_RETURN_IF_ERROR(ExpandTreeOfTuples(elements, device_ordinal, allocator,
                                        &host_shape, &device_shape));

  // The aliasing is determined below based on whether or not all the inputs are
  // released while being transferred. allocation_tmp is a local pointer that is
  // copied to *allocation at the end only if the method succeeds.
  auto allocation_tmp = new XRTTupleAllocation(device_ordinal, allocator,
                                               host_shape, device_shape);
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
          xla::Shape subshape =
              xla::ShapeUtil::GetSubshape(device_shape, index);
          uint64 size = transfer_manager->GetByteSizeRequirement(subshape);
          TF_ASSIGN_OR_RETURN(xla::OwningDeviceMemory buffer,
                              allocator->Allocate(device_ordinal, size,
                                                  /*retry_on_failure=*/false));
          VLOG(2) << "Allocated buffer at " << buffer.opaque() << " index "
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
            [&](const xla::ShapeIndex& index, XRTBufferAllocationPtr* buffer) {
              *buffer = new XRTBufferAllocation(
                  se::DeviceMemoryBase(), element.allocation->device_ordinal(),
                  element.allocation->allocator_);
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

Status XRTTupleAllocation::Intern(ResourceMgr* rm, int64* key) {
  *key = get_uid();
  string key_string = absl::StrCat(*key);
  return rm->Create(kTupleContainer, key_string, this);
}

bool XRTTupleAllocation::IsExclusiveOwner() {
  for (const auto& buffer : buffers_) {
    if (!buffer.second->RefCountIsOne()) return false;
  }
  return true;
}

void XRTTupleAllocation::InitializeFromShapedBuffer(
    const xla::ShapedBuffer& shaped_buffer,
    xla::DeviceMemoryAllocator* allocator, int device_ordinal) {
  for (auto& buffer : buffers_) {
    // Make a reference-counted version of the allocated buffer.
    buffer.second = new XRTBufferAllocation(shaped_buffer.buffer(buffer.first),
                                            device_ordinal, allocator);
  }
}

xla::ShapedBuffer XRTTupleAllocation::ToShapedBuffer() {
  xla::ShapedBuffer shaped_buffer(on_host_shape(), on_device_shape(),
                                  allocator_->platform(), device_ordinal_);
  for (const auto& buffer : buffers_) {
    shaped_buffer.set_buffer(buffer.second->allocation(), buffer.first);
  }
  return shaped_buffer;
}

xla::ShapeTree<xla::MaybeOwningDeviceMemory>
XRTTupleAllocation::ToDeviceMemoryTree(bool release) {
  xla::ShapeTree<xla::MaybeOwningDeviceMemory> shaped_tree(on_device_shape());
  for (const auto& buffer : buffers_) {
    if (!release) {
      *shaped_tree.mutable_element(buffer.first) = buffer.second->allocation();
    } else {
      *shaped_tree.mutable_element(buffer.first) = xla::OwningDeviceMemory(
          buffer.second->allocation(), device_ordinal_, allocator_);
      DiscardAllocation(buffer.first);
    }
  }
  return shaped_tree;
}

}  // namespace tensorflow
