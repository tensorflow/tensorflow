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

#ifndef XLA_FFI_CALL_FRAME_H_
#define XLA_FFI_CALL_FRAME_H_

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/ffi/api/c_api.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::ffi {

// CallFrame library encodes C++ arguments using XLA FFI C API structs in a form
// compatible with the decoding defined in `ffi/api.h`.

//===----------------------------------------------------------------------===//
// CallFrameBuilder
//===----------------------------------------------------------------------===//

class CallFrame;  // forward declare

class CallFrameBuilder {
  // A little bit of template metaprogramming to append type to std::variant.
  template <typename V, class T>
  struct AppendType;

  template <typename... Ts, class T>
  struct AppendType<std::variant<Ts...>, T> {
    using Type = std::variant<Ts..., T>;
  };

 public:
  CallFrameBuilder();
  ~CallFrameBuilder();

  CallFrameBuilder(CallFrameBuilder&&);
  CallFrameBuilder& operator=(CallFrameBuilder&&);

  using Scalar = std::variant<int32_t, int64_t, float>;
  using Array = std::variant<std::vector<int32_t>, std::vector<int64_t>,
                             std::vector<float>>;

  // Declare implementation detail structs for call frame builder storage.
  struct Dictionary;

  // Attributes that do not support nested dictionaries.
  using FlatAttribute = std::variant<Scalar, Array, std::string>;
  using FlatAttributesMap = absl::flat_hash_map<std::string, FlatAttribute>;

  // Attributes that support arbitrary nesting.
  using Attribute = typename AppendType<FlatAttribute, Dictionary>::Type;
  using AttributesMap = absl::flat_hash_map<std::string, Attribute>;

  // Dictionary is just a wrapper around AttributesMap. We need an indirection
  // through `std::unique_ptr` to be able to define recursive `std::variant`.
  struct Dictionary {
    std::unique_ptr<AttributesMap> attrs;
  };

  // A helper class to build call frame attributes.
  class AttributesBuilder {
   public:
    AttributesBuilder();
    ~AttributesBuilder();

    void Insert(std::string name, FlatAttribute attr);
    void Insert(std::string name, FlatAttributesMap attrs);

    void Append(FlatAttributesMap attrs);

    AttributesMap Build();

   private:
    AttributesMap attrs_;
  };

  CallFrame Build();

  void AddBufferArg(se::DeviceMemoryBase memory, PrimitiveType type,
                    absl::Span<const int64_t> dims);

  void AddBufferRet(se::DeviceMemoryBase memory, PrimitiveType type,
                    absl::Span<const int64_t> dims);

  void AddAttributes(AttributesMap attrs);

 private:
  friend class CallFrame;

  struct Buffer;

  std::vector<Buffer> args_;
  std::vector<Buffer> rets_;
  AttributesMap attrs_;
};

//===----------------------------------------------------------------------===//
// CallFrame
//===----------------------------------------------------------------------===//

class CallFrame {
 public:
  ~CallFrame();

  // Builds an XLA_FFI_CallFrame from owned arguments and attributes.
  XLA_FFI_CallFrame Build(const XLA_FFI_Api* api,
                          XLA_FFI_ExecutionContext* ctx);

 private:
  friend class CallFrameBuilder;

  // Declare implementation detail structs for call frame storage.
  struct Arguments;
  struct Array;
  struct Attributes;
  struct Buffer;
  struct Dictionary;
  struct NamedAttribute;
  struct Results;
  struct Scalar;
  struct String;

  using Attribute = std::variant<Scalar, Array, String, Dictionary>;

  CallFrame(absl::Span<const CallFrameBuilder::Buffer> args,
            absl::Span<const CallFrameBuilder::Buffer> rets,
            const CallFrameBuilder::AttributesMap& attrs);

  static std::unique_ptr<Arguments> InitArgs(
      absl::Span<const CallFrameBuilder::Buffer> args);

  static std::unique_ptr<Results> InitRets(
      absl::Span<const CallFrameBuilder::Buffer> rets);

  static std::unique_ptr<Attributes> InitAttrs(
      const CallFrameBuilder::AttributesMap& attrs);

  static Buffer ConvertBuffer(const CallFrameBuilder::Buffer& buffer);

  std::unique_ptr<Arguments> arguments_;
  std::unique_ptr<Results> results_;
  std::unique_ptr<Attributes> attributes_;

  // Declare implementation detail structs to grant access to private members.
  struct ConvertAttribute;
  struct FixupAttribute;
  struct AttributeType;
  struct AttributeStorage;
};

}  // namespace xla::ffi

#endif  // XLA_FFI_CALL_FRAME_H_
