/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/ffi/call_frame.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_data.pb.h"

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// CallFrameBuilder
//===----------------------------------------------------------------------===//

struct CallFrameBuilder::Buffer {
  se::DeviceMemoryBase memory;
  PrimitiveType type;
  std::vector<int64_t> dims;
};

CallFrameBuilder::AttributesMap CallFrameBuilder::AttributesBuilder::Build() {
  return std::move(attrs_);
}

static CallFrameBuilder::Attribute FromFlatAttribute(
    CallFrameBuilder::FlatAttribute attr) {
  return std::visit(
      [](auto& attr) { return CallFrameBuilder::Attribute{attr}; }, attr);
}

CallFrameBuilder::AttributesBuilder::AttributesBuilder() = default;
CallFrameBuilder::AttributesBuilder::~AttributesBuilder() = default;

void CallFrameBuilder::AttributesBuilder::Insert(std::string name,
                                                 FlatAttribute attr) {
  attrs_.try_emplace(std::move(name), FromFlatAttribute(std::move(attr)));
}

void CallFrameBuilder::AttributesBuilder::Insert(std::string name,
                                                 FlatAttributesMap attrs) {
  AttributesBuilder builder;
  for (auto& [name, attr] : attrs) builder.Insert(name, std::move(attr));

  auto attrs_map = std::make_unique<AttributesMap>(builder.Build());
  attrs_.try_emplace(std::move(name), Dictionary{std::move(attrs_map)});
}

void CallFrameBuilder::AttributesBuilder::Append(FlatAttributesMap attrs) {
  for (auto& [name, attr] : attrs) Insert(name, std::move(attr));
}

CallFrameBuilder::CallFrameBuilder() = default;
CallFrameBuilder::~CallFrameBuilder() = default;

void CallFrameBuilder::AddBufferArg(se::DeviceMemoryBase memory,
                                    PrimitiveType type,
                                    absl::Span<const int64_t> dims) {
  args_.push_back(Buffer{memory, type, {dims.begin(), dims.end()}});
}

void CallFrameBuilder::AddAttributes(AttributesMap attrs) {
  for (auto& [name, attr] : attrs) {
    attrs_.try_emplace(std::move(name), std::move(attr));
  }
}

CallFrame CallFrameBuilder::Build() { return CallFrame(args_, attrs_); }

CallFrameBuilder::CallFrameBuilder(CallFrameBuilder&&) = default;
CallFrameBuilder& CallFrameBuilder::operator=(CallFrameBuilder&&) = default;

// ------------------------    !!! !!! !!!     ------------------------------ //

// WARNING: In many structs defined below we use a pattern where we declare
// a storage (e.g. an `std::string` member) and an XLA FFI reference type
// pointing into that storage in the same struct (XLA_FFI_ByteSpan). Extra care
// should be taken of keeping reference type up to date, e.g. if a parent
// struct put into an `std::vector` container, every time vector will reallocate
// storage all reference types will become invalid.

// We intentionally do not use smart pointers that would guarantee pointer
// stability for storage, as we are trying to minimize the number of heap
// allocations required for building a call frame.

// This is a low level internal implementation detail that should not leak via
// public header files, and can be changed at any time in the future.

//----------------------------------------------------------------------------//
// Arguments storage + reference types
//----------------------------------------------------------------------------//

struct CallFrame::Buffer {
  std::vector<int64_t> dims;  // XLA_FFI_Buffer::dims

  XLA_FFI_Buffer buffer = {XLA_FFI_Buffer_STRUCT_SIZE, nullptr};
};

struct CallFrame::Dictionary {
  std::unique_ptr<Attributes> attrs;
};

struct CallFrame::String {
  std::string value;  // XLA_FFI_ByteSpan::ptr

  XLA_FFI_ByteSpan span = {XLA_FFI_ByteSpan_STRUCT_SIZE, nullptr};
};

struct CallFrame::NamedAttribute {
  String name;
  Attribute value;
};

struct CallFrame::Arguments {
  explicit Arguments(size_t size) {
    arguments.reserve(size);
    types.reserve(size);
    args.reserve(size);
  }

  std::vector<Buffer> arguments;

  std::vector<XLA_FFI_ArgType> types;  // XLA_FFI_Args::types
  std::vector<void*> args;             // XLA_FFI_Args::args

  XLA_FFI_Args ffi_args = {XLA_FFI_Args_STRUCT_SIZE, nullptr};
};

struct CallFrame::Attributes {
  explicit Attributes(size_t size) {
    attributes.reserve(size);
    names.reserve(size);
    types.reserve(size);
    attrs.reserve(size);
  }

  std::vector<NamedAttribute> attributes;

  std::vector<XLA_FFI_ByteSpan*> names;  // XLA_FFI_Attributes::names
  std::vector<XLA_FFI_AttrType> types;   // XLA_FFI_Attributes::types
  std::vector<void*> attrs;              // XLA_FFI_Attributes::attrs

  XLA_FFI_Attrs ffi_attrs = {XLA_FFI_Attrs_STRUCT_SIZE, nullptr};
};

//===----------------------------------------------------------------------===//
// CallFrame
//===----------------------------------------------------------------------===//

CallFrame::CallFrame(absl::Span<const CallFrameBuilder::Buffer> args,
                     const CallFrameBuilder::AttributesMap& attrs)
    : arguments_(InitArgs(args)), attributes_(InitAttrs(attrs)) {}

XLA_FFI_CallFrame CallFrame::Build(XLA_FFI_Api* api,
                                   XLA_FFI_ExecutionContext* ctx) {
  XLA_FFI_CallFrame call_frame = {XLA_FFI_CallFrame_STRUCT_SIZE, nullptr};
  call_frame.api = api;
  call_frame.ctx = ctx;
  call_frame.args = arguments_->ffi_args;
  call_frame.attrs = attributes_->ffi_attrs;
  return call_frame;
}

CallFrame::~CallFrame() = default;

//===----------------------------------------------------------------------===//
// Call frame arguments
//===----------------------------------------------------------------------===//

/*static*/ std::unique_ptr<CallFrame::Arguments> CallFrame::InitArgs(
    absl::Span<const CallFrameBuilder::Buffer> bargs) {
  auto res = std::make_unique<Arguments>(bargs.size());

  // We rely on casting to and from underlying integral type to convert from
  // PrimitiveType to XLA FFI DataType, and for safety convert all unknown types
  // to invalid type, otherwise we can accidentally cause UB.
  auto to_data_type = [](PrimitiveType primitive_type) {
    switch (primitive_type) {
      case PrimitiveType::PRIMITIVE_TYPE_INVALID:
      case PrimitiveType::S8:
      case PrimitiveType::S16:
      case PrimitiveType::S32:
      case PrimitiveType::S64:
      case PrimitiveType::U8:
      case PrimitiveType::U16:
      case PrimitiveType::U32:
      case PrimitiveType::U64:
      case PrimitiveType::F16:
      case PrimitiveType::F32:
      case PrimitiveType::F64:
      case PrimitiveType::BF16:
        return static_cast<XLA_FFI_DataType>(primitive_type);
      default:
        DCHECK(false) << "Unsupported primitive type" << primitive_type;
        return XLA_FFI_DataType_INVALID;
    }
  };

  // Convert call frame builder arguments to call frame arguments.
  for (const CallFrameBuilder::Buffer& barg : bargs) {
    Buffer buffer;
    buffer.dims = barg.dims;
    buffer.buffer.data = const_cast<void*>(barg.memory.opaque());
    buffer.buffer.dtype = to_data_type(barg.type);
    buffer.buffer.rank = buffer.dims.size();
    res->arguments.push_back(std::move(buffer));
  }

  // Fix up pointers in XLA FFI structs.
  for (CallFrame::Buffer& arg : res->arguments) {
    arg.buffer.dims = arg.dims.data();
  }

  // Initialize vectors required for building XLA_FFI_Args.
  for (CallFrame::Buffer& arg : res->arguments) {
    res->types.push_back(XLA_FFI_ArgType_BUFFER);
    res->args.push_back(&arg.buffer);
  }

  // Finally initialize the XLA FFI struct. At this point all storage is
  // allocated and it's safe to grab a pointer to it.
  res->ffi_args.num_args = res->arguments.size();
  res->ffi_args.types = res->types.data();
  res->ffi_args.args = res->args.data();

  return res;
}

//===----------------------------------------------------------------------===//
// Call frame attributes
//===----------------------------------------------------------------------===//

// An std::visit overload set for converting CallFrameBuilder::Attribute to
// CallFrame::Attribute.
struct CallFrame::ConvertAttribute {
  template <typename T>
  CallFrame::Attribute operator()(const T& value) {
    return value;
  }

  CallFrame::Attribute operator()(const std::string& str) {
    return CallFrame::String{str};
  }

  CallFrame::Attribute operator()(const CallFrameBuilder::Dictionary& dict) {
    return CallFrame::Dictionary{InitAttrs(*dict.attrs)};
  }
};

// An std::visit overload set to fix up CallFrame::Attribute storage and
// initialize XLA FFI structs with valid pointers into storage objects.
struct CallFrame::FixupAttribute {
  template <typename T>
  void operator()(T& value) {}

  void operator()(CallFrame::String& str) {
    str.span.ptr = str.value.data();
    str.span.len = str.value.size();
  }
};

// An std::visit overload set to get CallFrame::Attribute XLA FFI type.
struct CallFrame::AttributeType {
  XLA_FFI_AttrType operator()(int32_t&) { return XLA_FFI_AttrType_I32; }

  XLA_FFI_AttrType operator()(int64_t&) { return XLA_FFI_AttrType_I64; }

  XLA_FFI_AttrType operator()(float&) { return XLA_FFI_AttrType_F32; }

  XLA_FFI_AttrType operator()(CallFrame::String&) {
    return XLA_FFI_AttrType_STRING;
  }

  XLA_FFI_AttrType operator()(CallFrame::Dictionary&) {
    return XLA_FFI_AttrType_DICTIONARY;
  }
};

// An std::visit overload set to get CallFrame::Attribute storage pointer.
struct CallFrame::AttributeStorage {
  template <typename T>
  void* operator()(T& value) {
    return &value;
  }

  void* operator()(CallFrame::String& str) { return &str.span; }

  void* operator()(CallFrame::Dictionary& dict) {
    return &dict.attrs->ffi_attrs;
  }
};

/*static*/ std::unique_ptr<CallFrame::Attributes> CallFrame::InitAttrs(
    const CallFrameBuilder::AttributesMap& battrs) {
  auto res = std::make_unique<Attributes>(battrs.size());

  // Convert call frame builder attributes to a collection of named attributes.
  for (auto& [name, battr] : battrs) {
    NamedAttribute attr = {String{name}, std::visit(ConvertAttribute(), battr)};
    res->attributes.push_back(std::move(attr));
  }

  // Sort attributes by name to enable binary search at run time.
  absl::c_sort(res->attributes,
               [](const NamedAttribute& a, const NamedAttribute& b) {
                 return a.name.value < b.name.value;
               });

  // Fix up XLA FFI structs to point to correct storage.
  for (NamedAttribute& attr : res->attributes) {
    std::invoke(FixupAttribute{}, attr.name);
    std::visit(FixupAttribute{}, attr.value);
  }

  // Initialize vectors required for building XLA_FFI_Attributes.
  for (NamedAttribute& attr : res->attributes) {
    res->names.push_back(&attr.name.span);
    res->types.push_back(std::visit(AttributeType(), attr.value));
    res->attrs.push_back(std::visit(AttributeStorage(), attr.value));
  }

  // Finally initialize XLA FFI struct. At this point all storage is allocated
  // and it's safe to grab a pointer to it.
  res->ffi_attrs.num_attrs = res->attributes.size();
  res->ffi_attrs.names = res->names.data();
  res->ffi_attrs.types = res->types.data();
  res->ffi_attrs.attrs = res->attrs.data();

  return res;
}

}  // namespace xla::ffi
