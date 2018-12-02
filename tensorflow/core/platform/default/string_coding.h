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
#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_STRING_CODING_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_STRING_CODING_H_

// IWYU pragma: private, include "third_party/tensorflow/core/platform/tensor_coding.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/tensor_coding.h

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace port {

// Encodes sequences of strings and serialized protocol buffers into a string.
// Normal usage consists of zero or more calls to Append() and a single call to
// Finalize().
class StringListEncoder {
 public:
  explicit StringListEncoder(string* out) : out_(out) {}

  // Encodes the given protocol buffer. This may not be called after Finalize().
  void Append(const protobuf::MessageLite& m) {
    core::PutVarint32(out_, m.ByteSize());
    m.AppendToString(&rest_);
  }

  // Encodes the given string. This may not be called after Finalize().
  void Append(const string& s) {
    core::PutVarint32(out_, s.length());
    strings::StrAppend(&rest_, s);
  }

  // Signals end of the encoding process. No other calls are allowed after this.
  void Finalize() { strings::StrAppend(out_, rest_); }

 private:
  string* out_;
  string rest_;
};

// Decodes a string into sequences of strings (which may represent serialized
// protocol buffers). Normal usage involves a single call to ReadSizes() in
// order to retrieve the length of all the strings in the sequence. For each
// size returned a call to Data() is expected and will return the actual
// string.
class StringListDecoder {
 public:
  explicit StringListDecoder(const string& in) : reader_(in) {}

  // Populates the given vector with the lengths of each string in the sequence
  // being decoded. Upon returning the vector is guaranteed to contain as many
  // elements as there are strings in the sequence.
  bool ReadSizes(std::vector<uint32>* sizes) {
    int64 total = 0;
    for (auto& size : *sizes) {
      if (!core::GetVarint32(&reader_, &size)) return false;
      total += size;
    }
    if (total != static_cast<int64>(reader_.size())) {
      return false;
    }
    return true;
  }

  // Returns a pointer to the next string in the sequence, then prepares for the
  // next call by advancing 'size' characters in the sequence.
  const char* Data(uint32 size) {
    const char* data = reader_.data();
    reader_.remove_prefix(size);
    return data;
  }

 private:
  StringPiece reader_;
};

std::unique_ptr<StringListEncoder> NewStringListEncoder(string* out);
std::unique_ptr<StringListDecoder> NewStringListDecoder(const string& in);

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_STRING_CODING_H_
