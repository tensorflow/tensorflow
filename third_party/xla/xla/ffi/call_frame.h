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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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
  CallFrameBuilder(size_t num_args, size_t num_rets);
  ~CallFrameBuilder();

  CallFrameBuilder(CallFrameBuilder&&);
  CallFrameBuilder& operator=(CallFrameBuilder&&);

  using Scalar = std::variant<bool, int8_t, int16_t, int32_t, int64_t, uint8_t,
                              uint16_t, uint32_t, uint64_t, float, double>;
  using Array = std::variant<std::vector<int8_t>, std::vector<int16_t>,
                             std::vector<int32_t>, std::vector<int64_t>,
                             std::vector<uint8_t>, std::vector<uint16_t>,
                             std::vector<uint32_t>, std::vector<uint64_t>,
                             std::vector<float>, std::vector<double>>;

  // Declare implementation detail structs for call frame builder storage.
  struct Dictionary;

  // Attributes that do not support nested dictionaries.
  using FlatAttribute = std::variant<Scalar, Array, std::string>;
  using FlatAttributesMap = absl::flat_hash_map<std::string, FlatAttribute>;

  // Attributes that support arbitrary nesting.
  using Attribute = typename AppendType<FlatAttribute, Dictionary>::Type;
  using AttributesMap = absl::flat_hash_map<std::string, Attribute>;

  // Dictionary is just a wrapper around AttributesMap. We need an indirection
  // through `std::shared_ptr` to be able to define recursive `std::variant`. We
  // use shared pointer to keep `AttributesMap` copyable.
  struct Dictionary {
    std::shared_ptr<AttributesMap> attrs;
  };

  // A helper class to build call frame attributes.
  class AttributesBuilder {
   public:
    AttributesBuilder();
    ~AttributesBuilder();

    void Insert(std::string name, Attribute attr);
    void Insert(std::string name, AttributesMap attrs);
    void Append(AttributesMap attrs);

    // This overload is only necessary to support older GCC versions.
    void Insert(std::string name, const char* attr) {
      Insert(std::move(name), Attribute{std::string(attr)});
    }

    AttributesMap Build();

   private:
    AttributesMap attrs_;
  };

  CallFrame Build();

  void AddBufferArg(se::DeviceMemoryBase memory, PrimitiveType type,
                    absl::Span<const int64_t> dims);

  void AddTokenArg();

  void AddBufferRet(se::DeviceMemoryBase memory, PrimitiveType type,
                    absl::Span<const int64_t> dims);

  void AddTokenRet();

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
  CallFrame(CallFrame&&);
  CallFrame& operator=(CallFrame&&);

  ~CallFrame();

  // Updates *this call frame with new device memory pointers. It's up to the
  // caller to ensure that access to the call frame is synchronized.
  //
  // For any particular instance of a custom call in the XLA program, all
  // attributes are defined at compile time. Also types and dimensions of all
  // array (buffer) arguments and results are known at compile time. Instead of
  // rebuilding the call frame from scratch on every execution, we can just
  // update the arguments and results with new pointers to device memory.
  absl::Status UpdateWithBuffers(absl::Span<const se::DeviceMemoryBase> args,
                                 absl::Span<const se::DeviceMemoryBase> rets);

  // Creates a copy of the call frame with updated arguments and results.
  absl::StatusOr<CallFrame> CopyWithBuffers(
      absl::Span<const se::DeviceMemoryBase> args,
      absl::Span<const se::DeviceMemoryBase> rets);

  // Builds an XLA_FFI_CallFrame from owned arguments and attributes.
  XLA_FFI_CallFrame Build(
      const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx,
      XLA_FFI_ExecutionStage stage = XLA_FFI_ExecutionStage_EXECUTE);

 private:
  friend class CallFrameBuilder;

  // Declare implementation detail structs to grant access to private members.
  struct AttributeStorage;
  struct AttributeType;
  struct ConvertAttribute;
  struct FixUpAttribute;

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

  CallFrame(std::unique_ptr<Arguments> arguments,
            std::unique_ptr<Results> results,
            std::shared_ptr<Attributes> attributes);

  static Buffer ConvertBuffer(const CallFrameBuilder::Buffer& buffer);

  //===----- Call frame arguments -----------------------------------------===//

  // Creates call frame arguments from the call frame builder buffers.
  static std::unique_ptr<Arguments> CreateArgs(
      absl::Span<const CallFrameBuilder::Buffer> args);

  // Copies call frame arguments.
  static std::unique_ptr<Arguments> CopyArgs(const Arguments& args);

  // Fixes up call frame arguments by initializing XLA FFI structs with valid
  // pointers into storage objects.
  static std::unique_ptr<Arguments> FixUpArgs(std::unique_ptr<Arguments> args);

  //===----- Call frame results -------------------------------------------===//

  // Creates call frame results from the call frame builder buffers.
  static std::unique_ptr<Results> CreateRets(
      absl::Span<const CallFrameBuilder::Buffer> rets);

  // Copies call frame results.
  static std::unique_ptr<Results> CopyRets(const Results& rets);

  // Fixes up call frame results by initializing XLA FFI structs with valid
  // pointers into storage objects.
  static std::unique_ptr<Results> FixUpRets(std::unique_ptr<Results> rets);

  //===----- Call frame attributes ----------------------------------------===//

  // Creates call frame attributes from the call frame builder attributes.
  static std::unique_ptr<Attributes> CreateAttrs(
      const CallFrameBuilder::AttributesMap& attrs);

  // Fixes up call frame attributes by initializing XLA FFI structs with valid
  // pointers into storage objects.
  static std::unique_ptr<Attributes> FixUpAttrs(
      std::unique_ptr<Attributes> attrs);

  std::unique_ptr<Arguments> arguments_;
  std::unique_ptr<Results> results_;

  // Attributes are defined at compile time and can be shared between multiple
  // instances of a call frame (see `Update` above).
  std::shared_ptr<Attributes> attributes_;
};

}  // namespace xla::ffi

#endif  // XLA_FFI_CALL_FRAME_H_
