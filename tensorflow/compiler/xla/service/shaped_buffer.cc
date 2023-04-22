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

#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

ShapedBuffer::ShapedBuffer(Shape on_device_shape, int device_ordinal)
    : on_device_shape_(std::move(on_device_shape)),
      device_ordinal_(device_ordinal),
      buffers_(&on_device_shape_) {
  on_host_shape_ = ShapeUtil::DeviceShapeToHostShape(on_device_shape_);
}

ShapedBuffer::ShapedBuffer(Shape on_host_shape, Shape on_device_shape,
                           int device_ordinal)
    : ShapedBuffer(on_device_shape, device_ordinal) {}

ShapedBuffer::ShapedBuffer(ShapedBuffer&& s)
    : on_host_shape_(std::move(s.on_host_shape_)),
      on_device_shape_(std::move(s.on_device_shape_)),
      device_ordinal_(s.device_ordinal_),
      buffers_(std::move(s.buffers_)) {
  // s.buffers_ has a pointer to s.on_device_shape_. When we move s.buffers_
  // into buffers_, we also need to update this pointer so that buffers_ doesn't
  // point into s.
  buffers_.replace_shape_ptr(&on_device_shape_);
}

ShapedBuffer& ShapedBuffer::operator=(ShapedBuffer&& s) {
  on_device_shape_ = std::move(s.on_device_shape_);
  on_host_shape_ = std::move(s.on_host_shape_);
  device_ordinal_ = s.device_ordinal_;
  buffers_ = std::move(s.buffers_);
  // buffers_ has a pointer to its on_device_shape_. When we move s.buffers_
  // into buffers_, we also need to update this pointer so that buffers_ doesn't
  // point into s.
  buffers_.replace_shape_ptr(&on_device_shape_);
  return *this;
}

ShapedBuffer::~ShapedBuffer() {}

StatusOr<ShapedBuffer> ShapedBuffer::SubShapedBuffer(
    const ShapeIndex& index) const {
  TF_ASSIGN_OR_RETURN(const Shape* device_sub_shape,
                      ShapeUtil::TryGetSubshape(on_device_shape(), index));
  ShapedBuffer sub_shaped_buffer(*device_sub_shape, device_ordinal_);
  TF_ASSIGN_OR_RETURN(ShapeTree<se::DeviceMemoryBase> sub_buffers,
                      buffers_.SubShapeTree(index));
  sub_shaped_buffer.set_buffers(std::move(sub_buffers));
  return std::move(sub_shaped_buffer);
}

void ShapedBuffer::clear() {
  for (auto& pair : buffers_) {
    // A default constructed DeviceMemoryBase is a null pointer.
    pair.second = se::DeviceMemoryBase();
  }
}

string ShapedBuffer::ToString() const {
  string s =
      absl::StrCat("ShapedBuffer(", device_ordinal(),
                   "), on-device shape=" +
                       ShapeUtil::HumanStringWithLayout(on_device_shape()),
                   ":\n");
  ShapeUtil::ForEachSubshape(
      on_device_shape(),
      [this, &s](const Shape& subshape, const ShapeIndex& index) {
        string shape_str;
        if (subshape.IsTuple()) {
          shape_str = "tuple";
        } else {
          shape_str = ShapeUtil::HumanStringWithLayout(subshape);
        }
        const se::DeviceMemoryBase& memory = buffer(index);
        absl::StrAppendFormat(&s, "  %s%p (%d bytes) : %s\n",
                              string(index.size() * 2, ' '), memory.opaque(),
                              memory.size(), shape_str);
      });
  return s;
}

std::ostream& operator<<(std::ostream& out, const ShapedBuffer& buffer) {
  out << buffer.ToString();
  return out;
}

ScopedShapedBuffer::ScopedShapedBuffer(Shape on_device_shape,
                                       se::DeviceMemoryAllocator* allocator,
                                       int device_ordinal)
    : ShapedBuffer(std::move(on_device_shape), device_ordinal),
      allocator_(allocator) {}

ScopedShapedBuffer::ScopedShapedBuffer(Shape on_host_shape,
                                       Shape on_device_shape,
                                       se::DeviceMemoryAllocator* allocator,
                                       int device_ordinal)
    : ScopedShapedBuffer(std::move(on_device_shape), allocator,
                         device_ordinal) {}

ScopedShapedBuffer::ScopedShapedBuffer(ShapedBuffer shaped_buffer,
                                       se::DeviceMemoryAllocator* allocator)
    : ShapedBuffer(std::move(shaped_buffer)), allocator_(allocator) {}

ScopedShapedBuffer::ScopedShapedBuffer(ScopedShapedBuffer&& s)
    : ShapedBuffer(static_cast<ShapedBuffer&&>(s)), allocator_(s.allocator_) {
  // Null out s.allocator_ so it doesn't try to free anything in its destructor.
  s.allocator_ = nullptr;
}

ScopedShapedBuffer& ScopedShapedBuffer::operator=(ScopedShapedBuffer&& s) {
  Deallocate();

  *static_cast<ShapedBuffer*>(this) = std::move(static_cast<ShapedBuffer&>(s));
  allocator_ = s.allocator_;
  // Null out s.allocator_ so it doesn't try to free anything in its destructor.
  s.allocator_ = nullptr;
  return *this;
}

ScopedShapedBuffer::~ScopedShapedBuffer() { Deallocate(); }

ShapedBuffer ScopedShapedBuffer::release() {
  ShapedBuffer shaped_buffer(static_cast<ShapedBuffer&&>(*this));
  buffers_ = ShapeTree<se::DeviceMemoryBase>();
  return shaped_buffer;
}

void ScopedShapedBuffer::Deallocate() {
  // allocator_ will be null if we were moved-from.
  if (allocator_ == nullptr) {
    return;
  }
  // Deallocate all non-null buffers. A buffer may appear in more than one spot
  // in the shape (eg, a tuple with a repeated element) so keep track of what
  // has been deallocated.
  absl::flat_hash_set<void*> deallocated_ptrs;
  for (auto& pair : buffers_) {
    se::DeviceMemoryBase& memory_base = pair.second;
    if (!memory_base.is_null() &&
        deallocated_ptrs.insert(memory_base.opaque()).second) {
      TF_CHECK_OK(allocator_->Deallocate(device_ordinal(), memory_base));
    }
  }
}

ScopedShapedBuffer ScopedShapedBuffer::TakeSubTree(ShapeIndexView index) {
  const xla::Shape& sub_on_device_shape =
      xla::ShapeUtil::GetSubshape(on_device_shape(), {index});

  ScopedShapedBuffer output(sub_on_device_shape, memory_allocator(),
                            device_ordinal());
  auto src_it = buffers().find(index);
  auto dst_it = output.buffers().begin();
  while (dst_it != output.buffers().end()) {
    dst_it->second = src_it->second;
    src_it->second = tensorflow::se::DeviceMemoryBase(nullptr, 0);
    ++src_it;
    ++dst_it;
  }
  return output;
}

}  // namespace xla
