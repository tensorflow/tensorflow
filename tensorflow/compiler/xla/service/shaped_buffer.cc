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

#include "tensorflow/compiler/xla/service/shaped_buffer.h"

#include <set>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace se = ::perftools::gputools;

namespace xla {

/* static */ StatusOr<std::unique_ptr<ShapedBuffer>>
ShapedBuffer::MakeArrayShapedBuffer(const Shape& shape,
                                    const se::Platform* platform,
                                    int device_ordinal,
                                    const se::DeviceMemoryBase& buffer) {
  if (ShapeUtil::IsTuple(shape)) {
    return InvalidArgument("Shape must be an array: %s",
                           ShapeUtil::HumanStringWithLayout(shape).c_str());
  }
  auto shaped_buffer =
      MakeUnique<ShapedBuffer>(shape, platform, device_ordinal);
  *shaped_buffer->mutable_shape_index_to_buffer_entry()->mutable_element({}) =
      0;
  *shaped_buffer->mutable_buffers() = {buffer};
  return std::move(shaped_buffer);
}

ShapedBuffer::ShapedBuffer(const Shape& shape, const se::Platform* platform,
                           int device_ordinal)
    : shape_(shape),
      platform_(platform),
      device_ordinal_(device_ordinal),
      shape_index_to_buffer_entry_(shape) {}

void ShapedBuffer::clear() {
  for (se::DeviceMemoryBase& memory_base : buffers_) {
    // A default constructed DeviceMemoryBase is a null pointer.
    memory_base = se::DeviceMemoryBase();
  }
}

const se::DeviceMemoryBase& ShapedBuffer::buffer(
    const ShapeIndex& index) const {
  return buffers_[shape_index_to_buffer_entry_.element(index)];
}

se::DeviceMemoryBase* ShapedBuffer::mutable_buffer(const ShapeIndex& index) {
  return &buffers_[shape_index_to_buffer_entry_.element(index)];
}

/* static */ StatusOr<std::unique_ptr<ScopedShapedBuffer>>
ScopedShapedBuffer::Allocate(const Shape& shape,
                             DeviceMemoryAllocator* allocator,
                             int device_ordinal) {
  if (!LayoutUtil::HasLayout(shape)) {
    return InvalidArgument("Shape must have a layout: %s",
                           ShapeUtil::HumanStringWithLayout(shape).c_str());
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(shape));
  auto shaped_buffer =
      WrapUnique(new ScopedShapedBuffer(shape, allocator, device_ordinal));

  // Allocate an appropriate sized buffer for each element in the shape
  // including the tuple pointer arrays. Gather tuple element addresses in
  // 'element_addresses'. These will be written in the respective tuple's array
  // of pointers on the device.
  TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager,
                      TransferManager::GetForPlatform(allocator->platform()));
  ShapeTree<std::vector<se::DeviceMemoryBase>> element_addresses(shape);
  for (auto& pair : shaped_buffer->shape_index_to_buffer_entry_) {
    const ShapeIndex& index = pair.first;
    size_t& buffer_entry = pair.second;
    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase memory_base,
        shaped_buffer->allocator_->Allocate(
            shaped_buffer->device_ordinal(),
            transfer_manager->GetByteSizeRequirement(
                ShapeUtil::GetSubshape(shaped_buffer->shape(), index))));
    shaped_buffer->buffers_.push_back(memory_base);
    buffer_entry = shaped_buffer->buffers_.size() - 1;

    // If this is a tuple element, then push the address on to the
    // vector of tuple element addresses.
    if (!index.empty()) {
      ShapeIndex parent_index = index;
      parent_index.pop_back();
      element_addresses.mutable_element(parent_index)->push_back(memory_base);
    }
  }

  // Fill in the tuple pointer arrays with the addresses of their respective
  // elements.
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      allocator->platform()->ExecutorForDevice(
                          shaped_buffer->device_ordinal()));
  for (const auto& pair : element_addresses) {
    const ShapeIndex& index = pair.first;
    const std::vector<se::DeviceMemoryBase>& addresses = pair.second;
    const Shape& subshape = ShapeUtil::GetSubshape(shape, index);

    if (addresses.empty()) {
      TF_RET_CHECK(!ShapeUtil::IsTuple(subshape) ||
                   ShapeUtil::TupleElementCount(subshape) == 0);
      continue;
    }
    TF_RET_CHECK(ShapeUtil::IsTuple(subshape));
    TF_RETURN_IF_ERROR(transfer_manager->WriteTuplePointersToDevice(
        executor, addresses, subshape, shaped_buffer->mutable_buffer(index)));
  }

  return std::move(shaped_buffer);
}

/* static */
StatusOr<std::unique_ptr<ScopedShapedBuffer>> ScopedShapedBuffer::MakeScoped(
    ShapedBuffer* shaped_buffer, DeviceMemoryAllocator* allocator) {
  auto scoped_buffer = WrapUnique(new ScopedShapedBuffer(
      shaped_buffer->shape(), allocator, shaped_buffer->device_ordinal()));
  scoped_buffer->buffers_ = shaped_buffer->buffers();
  scoped_buffer->shape_index_to_buffer_entry_ =
      shaped_buffer->shape_index_to_buffer_entry();

  shaped_buffer->clear();

  return std::move(scoped_buffer);
}

ScopedShapedBuffer::ScopedShapedBuffer(const Shape& shape,
                                       DeviceMemoryAllocator* allocator,
                                       int device_ordinal)
    : ShapedBuffer(shape, allocator->platform(), device_ordinal),
      allocator_(allocator) {}

ScopedShapedBuffer::~ScopedShapedBuffer() {
  // Deallocate all non-null buffers. A buffer may appear in more than one spot
  // in the shape (eg, a tuple with a repeated element) so keep track of what
  // has been deallocated.
  std::set<void*> deallocated_opaques;
  for (se::DeviceMemoryBase& memory_base : buffers_) {
    if (!memory_base.is_null() &&
        deallocated_opaques.count(memory_base.opaque()) == 0) {
      deallocated_opaques.insert(memory_base.opaque());
      TF_CHECK_OK(
          this->allocator_->Deallocate(this->device_ordinal(), &memory_base));
    }
  }
}

std::unique_ptr<ShapedBuffer> ScopedShapedBuffer::release() {
  auto shaped_buffer =
      MakeUnique<ShapedBuffer>(shape(), platform(), device_ordinal());

  *shaped_buffer->mutable_buffers() = buffers();
  *shaped_buffer->mutable_shape_index_to_buffer_entry() =
      shape_index_to_buffer_entry();

  clear();

  return shaped_buffer;
}

}  // namespace xla
