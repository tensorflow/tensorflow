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
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using ::tensorflow::strings::Appendf;

ShapedBuffer::ShapedBuffer(const Shape& on_host_shape,
                           const Shape& on_device_shape,
                           const se::Platform* platform, int device_ordinal)
    : on_host_shape_(on_host_shape),
      on_device_shape_(on_device_shape),
      platform_(platform),
      device_ordinal_(device_ordinal),
      buffers_(&on_device_shape_) {}

ShapedBuffer::ShapedBuffer(ShapedBuffer&& s)
    : on_host_shape_(std::move(s.on_host_shape_)),
      on_device_shape_(std::move(s.on_device_shape_)),
      platform_(s.platform_),
      device_ordinal_(s.device_ordinal_),
      buffers_(std::move(s.buffers_)) {
  // s.buffers_ has a pointer to s.on_device_shape_. When we move s.buffers_
  // into buffers_, we also need to update this pointer so that buffers_ doesn't
  // point into s.
  buffers_.replace_shape_ptr(&on_device_shape_);
}

ShapedBuffer& ShapedBuffer::operator=(ShapedBuffer&& s) {
  on_host_shape_ = std::move(s.on_host_shape_);
  on_device_shape_ = std::move(s.on_device_shape_);
  platform_ = s.platform_;
  device_ordinal_ = s.device_ordinal_;
  buffers_ = std::move(s.buffers_);
  // buffers_ has a pointer to its on_device_shape_. When we move s.buffers_
  // into buffers_, we also need to update this pointer so that buffers_ doesn't
  // point into s.
  buffers_.replace_shape_ptr(&on_device_shape_);
  return *this;
}

ShapedBuffer::~ShapedBuffer() {}

void ShapedBuffer::clear() {
  for (auto& pair : buffers_) {
    // A default constructed DeviceMemoryBase is a null pointer.
    pair.second = se::DeviceMemoryBase();
  }
}

string ShapedBuffer::ToString() const {
  string s = tensorflow::strings::StrCat(
      "ShapedBuffer(", platform_->Name(), ":", device_ordinal(),
      "), on-host shape=" + ShapeUtil::HumanStringWithLayout(on_host_shape()),
      ", on-device shape=" +
          ShapeUtil::HumanStringWithLayout(on_device_shape()),
      ":\n");
  ShapeUtil::ForEachSubshape(
      on_device_shape(),
      [this, &s](const Shape& subshape, const ShapeIndex& index) {
        string shape_str;
        if (ShapeUtil::IsTuple(subshape)) {
          shape_str = "tuple";
        } else {
          shape_str = ShapeUtil::HumanStringWithLayout(subshape);
        }
        const se::DeviceMemoryBase& memory = buffer(index);
        Appendf(&s, "  %s%p (%lld bytes) : %s\n",
                string(index.size() * 2, ' ').c_str(), memory.opaque(),
                memory.size(), shape_str.c_str());
      });
  return s;
}

std::ostream& operator<<(std::ostream& out, const ShapedBuffer& buffer) {
  out << buffer.ToString();
  return out;
}

ScopedShapedBuffer::ScopedShapedBuffer(const Shape& on_host_shape,
                                       const Shape& on_device_shape,
                                       DeviceMemoryAllocator* allocator,
                                       int device_ordinal)
    : ShapedBuffer(on_host_shape, on_device_shape, allocator->platform(),
                   device_ordinal),
      allocator_(allocator) {}

ScopedShapedBuffer::ScopedShapedBuffer(ShapedBuffer shaped_buffer,
                                       DeviceMemoryAllocator* allocator)
    : ShapedBuffer(std::move(shaped_buffer)), allocator_(allocator) {}

ScopedShapedBuffer::ScopedShapedBuffer(ScopedShapedBuffer&& s)
    : ShapedBuffer(static_cast<ShapedBuffer&&>(s)), allocator_(s.allocator_) {
  // Null out s.allocator_ so it doesn't try to free anything in its destructor.
  s.allocator_ = nullptr;
}

ScopedShapedBuffer& ScopedShapedBuffer::operator=(ScopedShapedBuffer&& s) {
  *static_cast<ShapedBuffer*>(this) = std::move(static_cast<ShapedBuffer&>(s));
  allocator_ = s.allocator_;
  // Null out s.allocator_ so it doesn't try to free anything in its destructor.
  s.allocator_ = nullptr;
  return *this;
}

ScopedShapedBuffer::~ScopedShapedBuffer() {
  // allocator_ will be null if we were moved-from.
  if (allocator_ == nullptr) {
    return;
  }
  // Deallocate all non-null buffers. A buffer may appear in more than one spot
  // in the shape (eg, a tuple with a repeated element) so keep track of what
  // has been deallocated.
  std::set<void*> deallocated_opaques;
  for (auto& pair : buffers_) {
    se::DeviceMemoryBase& memory_base = pair.second;
    if (!memory_base.is_null() &&
        deallocated_opaques.count(memory_base.opaque()) == 0) {
      deallocated_opaques.insert(memory_base.opaque());
      TF_CHECK_OK(
          this->allocator_->Deallocate(this->device_ordinal(), &memory_base));
    }
  }
}

ShapedBuffer ScopedShapedBuffer::release() {
  ShapedBuffer shaped_buffer(static_cast<ShapedBuffer&&>(*this));
  buffers_ = ShapeTree<se::DeviceMemoryBase>();
  return shaped_buffer;
}

}  // namespace xla
