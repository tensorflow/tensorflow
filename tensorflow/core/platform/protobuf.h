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

#ifndef TENSORFLOW_CORE_PLATFORM_PROTOBUF_H_
#define TENSORFLOW_CORE_PLATFORM_PROTOBUF_H_

#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

// Import whatever namespace protobuf comes from into the
// ::tensorflow::protobuf namespace.
//
// TensorFlow code should use the ::tensorflow::protobuf namespace to
// refer to all protobuf APIs.

#ifndef TENSORFLOW_LITE_PROTOS
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/type_resolver_util.h"
#endif

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/map.h"
#include "google/protobuf/repeated_field.h"

namespace tensorflow {

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
                         const string& serialized);
bool ParseProtoUnlimited(protobuf::MessageLite* proto, const void* serialized,
                         size_t size);
#ifdef USE_TSTRING
inline bool ParseProtoUnlimited(protobuf::MessageLite* proto,
                                const tstring& serialized) {
  return ParseProtoUnlimited(proto, serialized.data(), serialized.size());
}
#endif  // USE_TSTRING

// Returns the string value for the value of a string or bytes protobuf field.
inline const string& ProtobufStringToString(const string& s) { return s; }

// Set <dest> to <src>. Swapping is allowed, as <src> does not need to be
// preserved.
inline void SetProtobufStringSwapAllowed(string* src, string* dest) {
  *dest = std::move(*src);
}

#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
// These versions of ProtobufStringToString and SetProtobufString get used by
// tools/proto_text's generated code.  They have the same name as the versions
// in core/platform/protobuf.h, so the generation code doesn't need to determine
// if the type is Cord or string at generation time.
inline string ProtobufStringToString(const Cord& s) { return s.ToString(); }
inline void SetProtobufStringSwapAllowed(string* src, Cord* dest) {
  dest->CopyFrom(*src);
}
#endif  // defined(TENSORFLOW_PROTOBUF_USES_CORD)

inline bool SerializeToTString(const protobuf::MessageLite& proto,
                               tstring* output) {
#ifdef USE_TSTRING
  size_t size = proto.ByteSizeLong();
  output->resize_uninitialized(size);
  return proto.SerializeToArray(output->data(), static_cast<int>(size));
#else   // USE_TSTRING
  return proto.SerializeToString(output);
#endif  // USE_TSTRING
}

#ifdef USE_TSTRING
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
  static const int kMinimumSize = 16;

  tstring* target_;
};
#else   // USE_TSTRING
typedef protobuf::io::StringOutputStream TStringOutputStream;
#endif  // USE_TSTRING

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PROTOBUF_H_
