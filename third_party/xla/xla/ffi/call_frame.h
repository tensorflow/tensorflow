/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
 public:
  using Attribute = std::variant<int32_t, float, std::string>;
  using AttributesMap = absl::flat_hash_map<std::string, Attribute>;

  CallFrame Build();

  void AddBufferArg(se::DeviceMemoryBase memory, PrimitiveType type,
                    absl::Span<const int64_t> dims);

  void AddI32Attr(std::string name, int32_t value);
  void AddF32Attr(std::string name, float value);
  void AddStringAttr(std::string name, std::string value);

  void AddAttribute(std::string name, Attribute attr);
  void AddAttributes(const AttributesMap& attrs);

 private:
  friend class CallFrame;

  struct Buffer {
    se::DeviceMemoryBase memory;
    PrimitiveType type;
    std::vector<int64_t> dims;
  };

  std::vector<Buffer> args_;
  AttributesMap attrs_;
};

//===----------------------------------------------------------------------===//
// CallFrame
//===----------------------------------------------------------------------===//

class CallFrame {
 public:
  ~CallFrame();

  // Builds an XLA_FFI_CallFrame from owned arguments and attributes.
  XLA_FFI_CallFrame Build(XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx);

 private:
  friend class CallFrameBuilder;

  // Declare implementation detail structs for call frame storage.
  struct Arguments;
  struct Attributes;
  struct Buffer;
  struct NamedAttribute;
  struct String;

  using Attribute = std::variant<int32_t, float, String>;

  CallFrame(absl::Span<const CallFrameBuilder::Buffer> args,
            const CallFrameBuilder::AttributesMap& attrs);

  static std::unique_ptr<Arguments> InitArgs(
      absl::Span<const CallFrameBuilder::Buffer> args);

  static std::unique_ptr<Attributes> InitAttrs(
      const CallFrameBuilder::AttributesMap& attrs);

  std::unique_ptr<Arguments> arguments_;
  std::unique_ptr<Attributes> attributes_;

  // Declare implementation detail structs to grant access to private members.
  struct ConvertAttribute;
  struct FixupAttribute;
  struct AttributeType;
  struct AttributeStorage;
};

}  // namespace xla::ffi

#endif  // XLA_FFI_CALL_FRAME_H_
