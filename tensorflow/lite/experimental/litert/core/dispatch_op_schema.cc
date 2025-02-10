// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/dispatch_op_schema.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"

namespace litert {
namespace internal {
namespace {

static constexpr const char kBytecodeSizeKey[] = "bytecode_size";
static constexpr const char kBytecodeOffsetKey[] = "bytecode_offset";
static constexpr const char kNameKey[] = "name";

}  // namespace

OwningBufferRef<uint8_t> MakeDispatchOpOptions(DispatchOpOptions options) {
  flexbuffers::Builder fbb;

  // Set maximum width for scalars to 64 bits. This prevents any upsizing of
  // the buffer when updating the bytecode size and offset in place.
  fbb.ForceMinimumBitWidth(flexbuffers::BIT_WIDTH_64);

  auto start = fbb.StartMap();

  fbb.UInt(kBytecodeSizeKey, options.bytecode_size);
  fbb.UInt(kBytecodeOffsetKey, options.bytecode_offset);
  fbb.String(kNameKey, options.name);

  fbb.EndMap(start);
  fbb.Finish();

  auto buf = fbb.GetBuffer();
  OwningBufferRef<uint8_t> res;
  res.Assign(buf.data(), buf.size());

  return res;
}

bool UpdateDispatchOpOptionsInPlace(DispatchOpOptions options,
                                    MutableBufferRef<uint8_t> buffer) {
  auto opts = flexbuffers::GetRoot(buffer.Data(), buffer.Size()).AsMap();

  // Update name if same len.
  const auto name_ok = opts[kNameKey].MutateString(options.name);

  // Update bytecode size and offset. Since min scalar bit width is set to max
  // possible value, it shouldn't fail in theory.
  const auto size_ok = opts[kBytecodeSizeKey].MutateUInt(options.bytecode_size);
  const auto offset_ok =
      opts[kBytecodeOffsetKey].MutateUInt(options.bytecode_offset);

  return name_ok && size_ok && offset_ok;
}

DispatchOpOptions GetDispatchOpOptions(BufferRef<uint8_t> buffer) {
  const auto opts = flexbuffers::GetRoot(buffer.Data(), buffer.Size()).AsMap();

  const size_t bytecode_size = opts[kBytecodeSizeKey].AsUInt64();
  const size_t bytecode_offset = opts[kBytecodeOffsetKey].AsUInt64();
  std::string name(opts[kNameKey].AsString().c_str());

  return DispatchOpOptions{
      bytecode_size,
      bytecode_offset,
      std::move(name),
  };
}

}  // namespace internal
}  // namespace litert
