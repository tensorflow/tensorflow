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
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/* static */ StatusOr<std::unique_ptr<ShapedBuffer>>
ShapedBuffer::MakeShapedBuffer(const Shape& shape,
                               const perftools::gputools::Platform* platform,
                               int device_ordinal) {
  if (!LayoutUtil::HasLayout(shape)) {
    return InvalidArgument("Shape must have a layout: %s",
                           ShapeUtil::HumanStringWithLayout(shape).c_str());
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(shape));
  return WrapUnique(new ShapedBuffer(shape, platform, device_ordinal));
}

/* static */ StatusOr<std::unique_ptr<ShapedBuffer>>
ShapedBuffer::MakeArrayShapedBuffer(
    const Shape& shape, const perftools::gputools::Platform* platform,
    int device_ordinal, const perftools::gputools::DeviceMemoryBase& buffer) {
  if (ShapeUtil::IsTuple(shape)) {
    return InvalidArgument("Shape must be an array: %s",
                           ShapeUtil::HumanStringWithLayout(shape).c_str());
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<ShapedBuffer> shaped_buffer,
                      MakeShapedBuffer(shape, platform, device_ordinal));
  *shaped_buffer->mutable_shape_index_to_buffer_entry()->mutable_element({}) =
      0;
  *shaped_buffer->mutable_buffers() = {buffer};
  return std::move(shaped_buffer);
}

/* static */ StatusOr<std::unique_ptr<ShapedBuffer>>
ShapedBuffer::MakeUnnestedTupleShapedBuffer(
    const Shape& shape, const perftools::gputools::Platform* platform,
    int device_ordinal,
    const tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
        buffers) {
  if (!ShapeUtil::IsTuple(shape) || ShapeUtil::IsNestedTuple(shape)) {
    return InvalidArgument("Shape must be an unnested tuple: %s",
                           ShapeUtil::HumanStringWithLayout(shape).c_str());
  }
  if (buffers.size() != ShapeUtil::TupleElementCount(shape)) {
    return InvalidArgument("Tuple has %lld elements, but %zu buffers given",
                           ShapeUtil::TupleElementCount(shape), buffers.size());
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<ShapedBuffer> shaped_buffer,
                      MakeShapedBuffer(shape, platform, device_ordinal));
  shaped_buffer->mutable_shape_index_to_buffer_entry()->ForEachMutableElement(
      [&shaped_buffer](const ShapeIndex& index, size_t* buffer_element) {
        if (ShapeUtil::IsLeafIndex(shaped_buffer->shape(), index)) {
          CHECK_EQ(index.size(), 1);
          *buffer_element = index[0];
        }
      });
  shaped_buffer->mutable_buffers()->reserve(buffers.size());
  for (const perftools::gputools::DeviceMemoryBase& memory_base : buffers) {
    shaped_buffer->mutable_buffers()->push_back(memory_base);
  }
  return std::move(shaped_buffer);
}

ShapedBuffer::ShapedBuffer(const Shape& shape,
                           const perftools::gputools::Platform* platform,
                           int device_ordinal)
    : shape_(shape),
      shape_index_to_buffer_entry_(shape),
      platform_(platform),
      device_ordinal_(device_ordinal) {}

const perftools::gputools::DeviceMemoryBase& ShapedBuffer::buffer(
    const ShapeIndex& index) const {
  // Buffer are only set at the leaves (array elements of the shape).
  CHECK(shape_index_to_buffer_entry_.IsLeaf(index));
  return buffers_[shape_index_to_buffer_entry_.element(index)];
}

perftools::gputools::DeviceMemoryBase* ShapedBuffer::mutable_buffer(
    const ShapeIndex& index) {
  // Buffer are only set at the leaves (array elements of the shape).
  CHECK(shape_index_to_buffer_entry_.IsLeaf(index));
  return &buffers_[shape_index_to_buffer_entry_.element(index)];
}

/* static */ StatusOr<std::unique_ptr<ScopedShapedBuffer>>
ScopedShapedBuffer::MakeScopedShapedBuffer(const Shape& shape,
                                           DeviceMemoryAllocator* allocator,
                                           int device_ordinal) {
  if (!LayoutUtil::HasLayout(shape)) {
    return InvalidArgument("Shape must have a layout: %s",
                           ShapeUtil::HumanStringWithLayout(shape).c_str());
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(shape));
  auto shaped_buffer =
      WrapUnique(new ScopedShapedBuffer(shape, allocator, device_ordinal));

  // Allocate an appropriate sized buffer for each array element in the shape.
  TF_RETURN_IF_ERROR(
      shaped_buffer->shape_index_to_buffer_entry_
          .ForEachMutableElementWithStatus([&shaped_buffer](
                                               const ShapeIndex& index,
                                               size_t* buffer_entry)
                                               -> tensorflow::Status {
            if (ShapeUtil::IsLeafIndex(shaped_buffer->shape(), index)) {
              TF_ASSIGN_OR_RETURN(
                  perftools::gputools::DeviceMemoryBase memory_base,
                  shaped_buffer->allocator_->Allocate(
                      shaped_buffer->device_ordinal(),
                      ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(
                          shaped_buffer->shape(), index))));
              shaped_buffer->buffers_.push_back(memory_base);
              *buffer_entry = shaped_buffer->buffers_.size() - 1;
            }
            return tensorflow::Status::OK();
          }));
  return std::move(shaped_buffer);
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
  for (perftools::gputools::DeviceMemoryBase& memory_base : buffers_) {
    if (!memory_base.is_null() &&
        deallocated_opaques.count(memory_base.opaque()) == 0) {
      deallocated_opaques.insert(memory_base.opaque());
      TF_CHECK_OK(
          this->allocator_->Deallocate(this->device_ordinal(), &memory_base));
    }
  }
}

}  // namespace xla
