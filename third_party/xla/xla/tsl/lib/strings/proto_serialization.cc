/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/lib/strings/proto_serialization.h"

#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/hash.h"

namespace tsl {
namespace {

// Helper for deterministic serialization.
class DeterministicSerializer {
 public:
  explicit DeterministicSerializer(const google::protobuf::MessageLite& msg)
      : DeterministicSerializer(msg, msg.ByteSizeLong()) {}

  DeterministicSerializer(const google::protobuf::MessageLite& msg, size_t size)
      : size_(size) {
    char* ptr = space_;
    if (size_ > sizeof(space_)) {
      ptr = new char[size_];
      alloc_.reset(ptr);
    }
    bool ok = SerializeToBufferDeterministic(msg, ptr, size_);
    DCHECK(ok);
  }

  size_t size() const { return size_; }
  const char* data() const { return alloc_ == nullptr ? space_ : alloc_.get(); }

 private:
  // Avoid InlinedVector since it causes 2x slowdown in the compilation
  // of graphs containing large tensors in debug mode.
  static constexpr int kInlinedBufferSize = 256;
  const size_t size_;
  std::unique_ptr<char[]> alloc_;
  char space_[kInlinedBufferSize];
};
}  // namespace

bool SerializeToStringDeterministic(const google::protobuf::MessageLite& msg,
                                    std::string* result) {
  const size_t size = msg.ByteSizeLong();
  if (size > static_cast<size_t>(INT_MAX)) {
    return false;
  }
  *result = std::string(size, '\0');
  return SerializeToBufferDeterministic(msg, const_cast<char*>(result->data()),
                                        result->size());
}

bool SerializeToBufferDeterministic(const google::protobuf::MessageLite& msg,
                                    char* buffer, size_t size) {
  if (msg.ByteSizeLong() != size) {
    return false;
  }
  if (size > static_cast<size_t>(INT_MAX)) {
    return false;
  }
  google::protobuf::io::ArrayOutputStream array_stream(buffer, size);
  google::protobuf::io::CodedOutputStream output_stream(&array_stream);
  output_stream.SetSerializationDeterministic(true);
  msg.SerializeWithCachedSizes(&output_stream);
  return !output_stream.HadError() &&
         size == static_cast<size_t>(output_stream.ByteCount());
}

bool AreSerializedProtosEqual(const google::protobuf::MessageLite& x,
                              const google::protobuf::MessageLite& y) {
  const size_t size = x.ByteSizeLong();
  if (size != y.ByteSizeLong()) return false;
  if (size == 0) return true;
  DeterministicSerializer x_serialized(x, size);
  DeterministicSerializer y_serialized(y, size);
  return memcmp(x_serialized.data(), y_serialized.data(), size) == 0;
}

uint64_t DeterministicProtoHash64(const google::protobuf::MessageLite& proto,
                                  uint64_t seed) {
  DeterministicSerializer serialized(proto);
  return Hash64(serialized.data(), serialized.size(), seed);
}

uint64_t DeterministicProtoHash64(const google::protobuf::MessageLite& proto) {
  DeterministicSerializer serialized(proto);
  return Hash64(serialized.data(), serialized.size());
}

}  // namespace tsl
