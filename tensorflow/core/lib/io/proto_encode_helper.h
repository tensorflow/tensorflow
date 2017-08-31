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

#ifndef TENSORFLOW_LIB_IO_PROTO_ENCODE_HELPER_H_
#define TENSORFLOW_LIB_IO_PROTO_ENCODE_HELPER_H_

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/platform/protobuf.h"

// A helper class for appending various kinds of values in protocol
// buffer encoding format to a buffer.  The client gives a pointer to
// a buffer and a maximum size guarantee for the number of bytes they
// will add to this buffer.
namespace tensorflow {
class StringPiece;
namespace io {

class ProtoEncodeHelper {
 public:
  ProtoEncodeHelper(char* buf, int max_size)
      : base_(buf), p_(buf), limit_(base_ + max_size) {}

  ~ProtoEncodeHelper() {
    // Make sure callers didn't do operations that went over max_size promised
    DCHECK_LE(p_, limit_);
  }

  const char* data() const { return base_; }
  size_t size() const { return p_ - base_; }

  void WriteUint64(int tag, uint64 v) {
    Encode32(combine(tag, WIRETYPE_VARINT));
    Encode64(v);
  }
  void WriteBool(int tag, bool v) {
    Encode32(combine(tag, WIRETYPE_VARINT));
    EncodeBool(v);
  }
  void WriteString(int tag, StringPiece v) {
    Encode32(combine(tag, WIRETYPE_LENGTH_DELIMITED));
    Encode32(v.size());
    EncodeBytes(v.data(), v.size());
  }
  void WriteVarlengthBeginning(int tag, uint32 len) {
    Encode32(combine(tag, WIRETYPE_LENGTH_DELIMITED));
    Encode32(len);
  }
  void WriteRawBytes(StringPiece v) { EncodeBytes(v.data(), v.size()); }

 private:
  // Note: this module's behavior must match the protocol buffer wire encoding
  // format.
  enum {
    WIRETYPE_VARINT = 0,
    WIRETYPE_LENGTH_DELIMITED = 2,
  };
  static uint32 combine(uint32 tag, uint32 type) { return ((tag << 3) | type); }
  inline void Encode32(uint32 v) {
    if (v < 128) {
      // Fast path for single-byte values.  Many of the calls will use a
      // constant value for v, so the comparison will get optimized away
      // when Encode32 is inlined into the caller.
      *p_ = v;
      p_++;
    } else {
      p_ = core::EncodeVarint32(p_, v);
    }
  }
  void Encode64(uint64 v) { p_ = core::EncodeVarint64(p_, v); }
  void EncodeBool(bool v) {
    *p_ = (v ? 1 : 0);  // Equal to varint32 encoding of 0 or 1
    p_++;
  }
  void EncodeBytes(const char* bytes, int N) {
    memcpy(p_, bytes, N);
    p_ += N;
  }

  char* base_;
  char* p_;
  char* limit_;  // Just for CHECKs
};
}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_PROTO_ENCODE_HELPER_H_
