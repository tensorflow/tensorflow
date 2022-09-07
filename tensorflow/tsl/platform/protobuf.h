/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_PROTOBUF_H_
#define TENSORFLOW_TSL_PLATFORM_PROTOBUF_H_

#include "tensorflow/tsl/platform/platform.h"
#include "tensorflow/tsl/platform/types.h"

// Import whatever namespace protobuf comes from into the
// ::tsl::protobuf namespace.
//
// TensorFlow code should use the ::tensorflow::protobuf namespace to
// refer to all protobuf APIs.

#include "google/protobuf/io/coded_stream.h"  // IWYU pragma: export
#include "google/protobuf/io/tokenizer.h"     // IWYU pragma: export
#include "google/protobuf/io/zero_copy_stream.h"  // IWYU pragma: export
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"  // IWYU pragma: export
#include "google/protobuf/descriptor.pb.h"     // IWYU pragma: export
#include "google/protobuf/arena.h"            // IWYU pragma: export
#include "google/protobuf/descriptor.h"       // IWYU pragma: export
#include "google/protobuf/dynamic_message.h"  // IWYU pragma: export
#include "google/protobuf/map.h"              // IWYU pragma: export
#include "google/protobuf/message.h"          // IWYU pragma: export
#include "google/protobuf/repeated_field.h"   // IWYU pragma: export
#include "google/protobuf/text_format.h"      // IWYU pragma: export
#include "google/protobuf/util/field_comparator.h"  // IWYU pragma: export
#include "google/protobuf/util/json_util.h"  // IWYU pragma: export
#include "google/protobuf/util/message_differencer.h"  // IWYU pragma: export
#include "google/protobuf/util/type_resolver_util.h"  // IWYU pragma: export

namespace tsl {

namespace protobuf = ::google::protobuf;
using protobuf_int64 = ::google::protobuf::int64;
using protobuf_uint64 = ::google::protobuf::uint64;
extern const char* kProtobufInt64Typename;
extern const char* kProtobufUint64Typename;

// Parses a protocol buffer contained in a string in the binary wire format.
// Returns true on success. Note: Unlike protobuf's builtin ParseFromString,
// this function has no size restrictions on the total size of the encoded
// protocol buffer.
bool ParseProtoUnlimited(protobuf::MessageLite* proto,
                         const std::string& serialized);
bool ParseProtoUnlimited(protobuf::MessageLite* proto, const void* serialized,
                         size_t size);
inline bool ParseProtoUnlimited(protobuf::MessageLite* proto,
                                const tstring& serialized) {
  return ParseProtoUnlimited(proto, serialized.data(), serialized.size());
}

// Returns the string value for the value of a string or bytes protobuf field.
inline const std::string& ProtobufStringToString(const std::string& s) {
  return s;
}

// Set <dest> to <src>. Swapping is allowed, as <src> does not need to be
// preserved.
inline void SetProtobufStringSwapAllowed(std::string* src, std::string* dest) {
  *dest = std::move(*src);
}

#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
// These versions of ProtobufStringToString and SetProtobufString get used by
// tools/proto_text's generated code.  They have the same name as the versions
// in core/platform/protobuf.h, so the generation code doesn't need to determine
// if the type is Cord or string at generation time.
inline std::string ProtobufStringToString(const absl::Cord& s) {
  return std::string(s);
}
inline void SetProtobufStringSwapAllowed(std::string* src, absl::Cord* dest) {
  dest->CopyFrom(*src);
}
#endif  // defined(TENSORFLOW_PROTOBUF_USES_CORD)

inline bool SerializeToTString(const protobuf::MessageLite& proto,
                               tstring* output) {
  size_t size = proto.ByteSizeLong();
  output->resize_uninitialized(size);
  return proto.SerializeWithCachedSizesToArray(
      reinterpret_cast<uint8*>(output->data()));
}

inline bool ParseFromTString(const tstring& input,
                             protobuf::MessageLite* proto) {
  return proto->ParseFromArray(input.data(), static_cast<int>(input.size()));
}

// Analogue to StringOutputStream for tstring.
class TStringOutputStream : public protobuf::io::ZeroCopyOutputStream {
 public:
  explicit TStringOutputStream(tstring* target);
  ~TStringOutputStream() override = default;

  TStringOutputStream(const TStringOutputStream&) = delete;
  void operator=(const TStringOutputStream&) = delete;

  bool Next(void** data, int* size) override;
  void BackUp(int count) override;
  int64_t ByteCount() const override;

 private:
  static constexpr int kMinimumSize = 16;

  tstring* target_;
};
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_PROTOBUF_H_
