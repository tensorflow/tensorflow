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
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/stream_executor/device_memory.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// CallFrameBuilder
//===----------------------------------------------------------------------===//

struct CallFrameBuilder::Buffer {
  se::DeviceMemoryBase memory;
  PrimitiveType type;
  absl::InlinedVector<int64_t, 4> dims;
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
                                                 Attribute attr) {
  attrs_.try_emplace(std::move(name), std::move(attr));
}

void CallFrameBuilder::AttributesBuilder::Insert(std::string name,
                                                 AttributesMap attrs) {
  attrs_.try_emplace(std::move(name),
                     Dictionary{std::make_shared<AttributesMap>(attrs)});
}

void CallFrameBuilder::AttributesBuilder::Append(AttributesMap attrs) {
  for (auto& [name, attr] : attrs) Insert(name, std::move(attr));
}

CallFrameBuilder::CallFrameBuilder(size_t num_args, size_t num_rets) {
  args_.reserve(num_args);
  rets_.reserve(num_rets);
}

CallFrameBuilder::~CallFrameBuilder() = default;

void CallFrameBuilder::AddBufferArg(se::DeviceMemoryBase memory,
                                    PrimitiveType type,
                                    absl::Span<const int64_t> dims) {
  DCHECK(args_.capacity() > args_.size())
      << "CallFrame builder `num_args` argument was too small";
  args_.push_back(Buffer{memory, type, {dims.begin(), dims.end()}});
}

void CallFrameBuilder::AddTokenArg() {
  DCHECK(args_.capacity() > args_.size())
      << "CallFrame builder `num_args` argument was too small";
  args_.push_back(Buffer{se::DeviceMemoryBase(), PrimitiveType::TOKEN, {}});
}

void CallFrameBuilder::AddBufferRet(se::DeviceMemoryBase memory,
                                    PrimitiveType type,
                                    absl::Span<const int64_t> dims) {
  DCHECK(rets_.capacity() > rets_.size())
      << "CallFrame builder `num_rets` argument was too small";
  rets_.push_back(Buffer{memory, type, {dims.begin(), dims.end()}});
}

void CallFrameBuilder::AddTokenRet() {
  DCHECK(rets_.capacity() > rets_.size())
      << "CallFrame builder `num_rets` argument was too small";
  rets_.push_back(Buffer{se::DeviceMemoryBase(), PrimitiveType::TOKEN, {}});
}

void CallFrameBuilder::AddAttributes(AttributesMap attrs) {
  if (ABSL_PREDICT_TRUE(attrs_.empty())) {
    attrs_ = std::move(attrs);
    return;
  }

  for (auto& [name, attr] : attrs) {
    attrs_.try_emplace(std::move(name), std::move(attr));
  }
}

CallFrame CallFrameBuilder::Build() {
  return CallFrame(CallFrame::CreateArgs(args_), CallFrame::CreateRets(rets_),
                   CallFrame::CreateAttrs(attrs_));
}

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
  absl::InlinedVector<int64_t, 4> dims;  // XLA_FFI_Buffer::dims

  XLA_FFI_Buffer buffer = {XLA_FFI_Buffer_STRUCT_SIZE, nullptr};
};

struct CallFrame::Dictionary {
  std::unique_ptr<Attributes> attrs;
};

struct CallFrame::Array {
  CallFrameBuilder::Array value;  // XLA_FFI_Array::data

  XLA_FFI_Array array = {};
};

struct CallFrame::Scalar {
  CallFrameBuilder::Scalar value;  // XLA_FFI_Scalar::value

  XLA_FFI_Scalar scalar = {};
};

struct CallFrame::String {
  std::string value;  // XLA_FFI_ByteSpan::ptr

  XLA_FFI_ByteSpan span = {};
};

struct CallFrame::NamedAttribute {
  String name;
  Attribute value;
};

struct CallFrame::Arguments {
  std::vector<Buffer> arguments;

  std::vector<XLA_FFI_ArgType> types;  // XLA_FFI_Args::types
  std::vector<void*> args;             // XLA_FFI_Args::args

  XLA_FFI_Args ffi_args = {XLA_FFI_Args_STRUCT_SIZE, nullptr};
};

struct CallFrame::Results {
  std::vector<Buffer> results;

  std::vector<XLA_FFI_RetType> types;  // XLA_FFI_Rets::types
  std::vector<void*> rets;             // XLA_FFI_Rets::rets

  XLA_FFI_Rets ffi_rets = {XLA_FFI_Rets_STRUCT_SIZE, nullptr};
};

struct CallFrame::Attributes {
  std::vector<NamedAttribute> attributes;

  std::vector<XLA_FFI_ByteSpan*> names;  // XLA_FFI_Attributes::names
  std::vector<XLA_FFI_AttrType> types;   // XLA_FFI_Attributes::types
  std::vector<void*> attrs;              // XLA_FFI_Attributes::attrs

  XLA_FFI_Attrs ffi_attrs = {XLA_FFI_Attrs_STRUCT_SIZE, nullptr};
};

//===----------------------------------------------------------------------===//
// CallFrame
//===----------------------------------------------------------------------===//

CallFrame::CallFrame(CallFrame&&) = default;
CallFrame& CallFrame::operator=(CallFrame&&) = default;
CallFrame::~CallFrame() = default;

CallFrame::CallFrame(std::unique_ptr<Arguments> arguments,
                     std::unique_ptr<Results> results,
                     std::shared_ptr<Attributes> attributes)
    : arguments_(std::move(arguments)),
      results_(std::move(results)),
      attributes_(std::move(attributes)) {}

XLA_FFI_CallFrame CallFrame::Build(const XLA_FFI_Api* api,
                                   XLA_FFI_ExecutionContext* ctx,
                                   XLA_FFI_ExecutionStage stage) {
  XLA_FFI_CallFrame call_frame = {XLA_FFI_CallFrame_STRUCT_SIZE, nullptr};
  call_frame.api = api;
  call_frame.ctx = ctx;
  call_frame.stage = stage;
  call_frame.args = arguments_->ffi_args;
  call_frame.rets = results_->ffi_rets;
  call_frame.attrs = attributes_->ffi_attrs;
  return call_frame;
}

// We rely on casting to and from underlying integral type to convert from
// PrimitiveType to XLA FFI DataType, and for safety convert all unknown types
// to invalid type, otherwise we can accidentally cause UB.
static XLA_FFI_DataType ToDataType(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case PrimitiveType::PRIMITIVE_TYPE_INVALID:
    case PrimitiveType::PRED:
    case PrimitiveType::S1:
    case PrimitiveType::S2:
    case PrimitiveType::S4:
    case PrimitiveType::S8:
    case PrimitiveType::S16:
    case PrimitiveType::S32:
    case PrimitiveType::S64:
    case PrimitiveType::U1:
    case PrimitiveType::U2:
    case PrimitiveType::U4:
    case PrimitiveType::U8:
    case PrimitiveType::U16:
    case PrimitiveType::U32:
    case PrimitiveType::U64:
    case PrimitiveType::F16:
    case PrimitiveType::F32:
    case PrimitiveType::F64:
    case PrimitiveType::BF16:
    case PrimitiveType::C64:
    case PrimitiveType::C128:
    case PrimitiveType::TOKEN:
    case PrimitiveType::F4E2M1FN:
    case PrimitiveType::F8E5M2:
    case PrimitiveType::F8E4M3:
    case PrimitiveType::F8E4M3FN:
    case PrimitiveType::F8E4M3B11FNUZ:
    case PrimitiveType::F8E5M2FNUZ:
    case PrimitiveType::F8E4M3FNUZ:
    case PrimitiveType::F8E3M4:
    case PrimitiveType::F8E8M0FNU:
    case PrimitiveType::TUPLE:
    case PrimitiveType::OPAQUE_TYPE:
      return static_cast<XLA_FFI_DataType>(primitive_type);
    default:
      DCHECK(false) << "Unsupported primitive type "
                    << PrimitiveType_Name(primitive_type);
      return XLA_FFI_DataType_INVALID;
  }
}

CallFrame::Buffer CallFrame::ConvertBuffer(
    const CallFrameBuilder::Buffer& buffer) {
  Buffer result;
  result.dims = buffer.dims;
  result.buffer.data = const_cast<void*>(buffer.memory.opaque());
  result.buffer.dtype = ToDataType(buffer.type);
  result.buffer.rank = result.dims.size();
  return result;
}

//===----------------------------------------------------------------------===//
// Call frame arguments
//===----------------------------------------------------------------------===//

std::unique_ptr<CallFrame::Arguments> CallFrame::CreateArgs(
    absl::Span<const CallFrameBuilder::Buffer> bargs) {
  size_t num_args = bargs.size();

  auto args = std::make_unique<Arguments>();
  args->types.resize(num_args, XLA_FFI_ArgType_BUFFER);
  args->args.resize(num_args, nullptr);  // fixed up below

  // Convert call frame builder arguments to call frame arguments.
  args->arguments.reserve(num_args);
  for (const CallFrameBuilder::Buffer& barg : bargs) {
    args->arguments.push_back(ConvertBuffer(barg));
  }

  // Fix up XLA FFI structs with pointers to valid arguments storage.
  return FixUpArgs(std::move(args));
}

std::unique_ptr<CallFrame::Arguments> CallFrame::CopyArgs(
    const Arguments& args) {
  auto upd_args = std::make_unique<Arguments>();

  upd_args->arguments = args.arguments;
  upd_args->types = args.types;
  upd_args->args.resize(args.args.size(), nullptr);  // fixed up below

  // Fix up XLA FFI structs with pointers to valid arguments storage.
  return FixUpArgs(std::move(upd_args));
}

std::unique_ptr<CallFrame::Arguments> CallFrame::FixUpArgs(
    std::unique_ptr<Arguments> args) {
  size_t num_args = args->arguments.size();
  DCHECK_EQ(num_args, args->types.size());
  DCHECK_EQ(num_args, args->args.size());

  // Fix up pointers in XLA FFI structs and initialize vectors required for
  // building XLA_FFI_Args.
  for (size_t i = 0; i < num_args; ++i) {
    args->arguments[i].buffer.dims = args->arguments[i].dims.data();
    args->args[i] = &args->arguments[i].buffer;
  }

  // Finally initialize the XLA FFI struct. At this point all storage is
  // allocated and it's safe to grab a pointer to it.
  args->ffi_args.size = num_args;
  args->ffi_args.types = args->types.data();
  args->ffi_args.args = args->args.data();

  return args;
}

//===----------------------------------------------------------------------===//
// Call frame results
//===----------------------------------------------------------------------===//

std::unique_ptr<CallFrame::Results> CallFrame::CreateRets(
    absl::Span<const CallFrameBuilder::Buffer> brets) {
  auto rets = std::make_unique<Results>();

  size_t num_rets = brets.size();
  rets->types.resize(num_rets, XLA_FFI_RetType_BUFFER);
  rets->rets.resize(num_rets, nullptr);  // fixed up below

  // Convert call frame builder result to call frame results.
  rets->results.reserve(num_rets);
  for (const CallFrameBuilder::Buffer& bret : brets) {
    rets->results.push_back(ConvertBuffer(bret));
  }

  // Fix up XLA FFI structs with pointers to valid results storage.
  return FixUpRets(std::move(rets));
}

std::unique_ptr<CallFrame::Results> CallFrame::CopyRets(const Results& rets) {
  auto upd_rets = std::make_unique<Results>();

  upd_rets->results = rets.results;
  upd_rets->types = rets.types;
  upd_rets->rets.resize(rets.rets.size(), nullptr);  // fixed up below

  // Fix up XLA FFI structs with pointers to valid results storage.
  return FixUpRets(std::move(upd_rets));
}

std::unique_ptr<CallFrame::Results> CallFrame::FixUpRets(
    std::unique_ptr<Results> rets) {
  size_t num_rets = rets->results.size();
  DCHECK_EQ(num_rets, rets->types.size());
  DCHECK_EQ(num_rets, rets->rets.size());

  // Fix up pointers in XLA FFI structs and initialize vectors required for
  // building XLA_FFI_Args.
  for (size_t i = 0; i < num_rets; ++i) {
    rets->results[i].buffer.dims = rets->results[i].dims.data();
    rets->rets[i] = &rets->results[i].buffer;
  }

  // Finally initialize the XLA FFI struct. At this point all storage is
  // allocated and it's safe to grab a pointer to it.
  rets->ffi_rets.size = num_rets;
  rets->ffi_rets.types = rets->types.data();
  rets->ffi_rets.rets = rets->rets.data();

  return rets;
}

//===----------------------------------------------------------------------===//
// Call frame attributes
//===----------------------------------------------------------------------===//

// An std::visit overload set for converting CallFrameBuilder::Attribute to
// CallFrame::Attribute.
struct CallFrame::ConvertAttribute {
  CallFrame::Attribute operator()(const CallFrameBuilder::Array& array) {
    return CallFrame::Array{array};
  }

  CallFrame::Attribute operator()(const CallFrameBuilder::Scalar& scalar) {
    return CallFrame::Scalar{scalar};
  }

  CallFrame::Attribute operator()(const std::string& str) {
    return CallFrame::String{str};
  }

  CallFrame::Attribute operator()(const CallFrameBuilder::Dictionary& dict) {
    return CallFrame::Dictionary{CreateAttrs(*dict.attrs)};
  }
};

// An std::visit overload set to fix up CallFrame::Attribute storage and
// initialize XLA FFI structs with valid pointers into storage objects.
struct CallFrame::FixUpAttribute {
  void operator()(CallFrame::Array& array) {
    auto visitor = [&](auto& value) {
      using T = typename std::remove_reference_t<decltype(value)>::value_type;
      array.array.dtype = internal::NativeTypeToCApiDataType<T>();
      array.array.size = value.size();
      array.array.data = value.data();
    };
    std::visit(visitor, array.value);
  }

  void operator()(CallFrame::Scalar& scalar) {
    auto visitor = [&](auto& value) {
      using T = std::remove_reference_t<decltype(value)>;
      scalar.scalar.dtype = internal::NativeTypeToCApiDataType<T>();
      scalar.scalar.value = &value;
    };
    std::visit(visitor, scalar.value);
  }

  void operator()(CallFrame::String& str) {
    str.span.ptr = str.value.data();
    str.span.len = str.value.size();
  }

  void operator()(CallFrame::Dictionary&) {}
};

// An std::visit overload set to get CallFrame::Attribute XLA FFI type.
struct CallFrame::AttributeType {
  XLA_FFI_AttrType operator()(CallFrame::Array&) {
    return XLA_FFI_AttrType_ARRAY;
  }

  XLA_FFI_AttrType operator()(CallFrame::Scalar&) {
    return XLA_FFI_AttrType_SCALAR;
  }

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

  void* operator()(CallFrame::Array& array) { return &array.array; }

  void* operator()(CallFrame::Scalar& scalar) { return &scalar.scalar; }

  void* operator()(CallFrame::String& str) { return &str.span; }

  void* operator()(CallFrame::Dictionary& dict) {
    return &dict.attrs->ffi_attrs;
  }
};

std::unique_ptr<CallFrame::Attributes> CallFrame::CreateAttrs(
    const CallFrameBuilder::AttributesMap& battrs) {
  auto attrs = std::make_unique<Attributes>();

  // Convert call frame builder attributes to a collection of named attributes.
  attrs->attributes.reserve(battrs.size());
  for (auto& [name, battr] : battrs) {
    NamedAttribute attr = {String{name}, std::visit(ConvertAttribute(), battr)};
    attrs->attributes.push_back(std::move(attr));
  }

  // Sort attributes by name to enable binary search at run time.
  absl::c_sort(attrs->attributes,
               [](const NamedAttribute& a, const NamedAttribute& b) {
                 return a.name.value < b.name.value;
               });

  return FixUpAttrs(std::move(attrs));
}

std::unique_ptr<CallFrame::Attributes> CallFrame::FixUpAttrs(
    std::unique_ptr<CallFrame::Attributes> attrs) {
  size_t num_attrs = attrs->attributes.size();
  DCHECK(attrs->names.empty() && attrs->types.empty() && attrs->attrs.empty());

  attrs->names.reserve(num_attrs);
  attrs->types.reserve(num_attrs);
  attrs->attrs.reserve(num_attrs);

  // Fix up XLA FFI structs to point to correct storage.
  for (NamedAttribute& attr : attrs->attributes) {
    std::invoke(FixUpAttribute{}, attr.name);
    std::visit(FixUpAttribute{}, attr.value);
  }

  // Initialize vectors required for building XLA_FFI_Attributes.
  for (NamedAttribute& attr : attrs->attributes) {
    attrs->names.push_back(&attr.name.span);
    attrs->types.push_back(std::visit(AttributeType(), attr.value));
    attrs->attrs.push_back(std::visit(AttributeStorage(), attr.value));
  }

  // Finally initialize XLA FFI struct. At this point all storage is allocated
  // and it's safe to grab a pointer to it.
  attrs->ffi_attrs.size = attrs->attributes.size();
  attrs->ffi_attrs.names = attrs->names.data();
  attrs->ffi_attrs.types = attrs->types.data();
  attrs->ffi_attrs.attrs = attrs->attrs.data();

  return attrs;
}

//===----------------------------------------------------------------------===//
// Call frame update
//===----------------------------------------------------------------------===//

absl::Status CallFrame::UpdateWithBuffers(
    absl::Span<const se::DeviceMemoryBase> args,
    absl::Span<const se::DeviceMemoryBase> rets) {
  if (ABSL_PREDICT_FALSE(args.size() != arguments_->args.size())) {
    return InvalidArgument("Invalid number of updated arguments: %d vs %d",
                           args.size(), arguments_->args.size());
  }

  if (ABSL_PREDICT_FALSE(rets.size() != results_->rets.size())) {
    return InvalidArgument("Invalid number of updated results: %d vs %d",
                           rets.size(), results_->rets.size());
  }

  size_t num_args = args.size();
  for (size_t i = 0; i < num_args; ++i) {
    arguments_->arguments[i].buffer.data = const_cast<void*>(args[i].opaque());
  }

  size_t num_rets = rets.size();
  for (size_t i = 0; i < num_rets; ++i) {
    results_->results[i].buffer.data = const_cast<void*>(rets[i].opaque());
  }

  return absl::OkStatus();
}

absl::StatusOr<CallFrame> CallFrame::CopyWithBuffers(
    absl::Span<const se::DeviceMemoryBase> args,
    absl::Span<const se::DeviceMemoryBase> rets) {
  CallFrame clone(CopyArgs(*arguments_), CopyRets(*results_), attributes_);
  TF_RETURN_IF_ERROR(clone.UpdateWithBuffers(args, rets));
  return clone;
}

}  // namespace xla::ffi
