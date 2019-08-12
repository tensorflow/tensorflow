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

// Helper routines for encoding/decoding tensor contents.
#ifndef TENSORFLOW_PLATFORM_TENSOR_CODING_H_
#define TENSORFLOW_PLATFORM_TENSOR_CODING_H_

#include <string>
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace port {

// Store src contents in *out.  If backing memory for src is shared with *out,
// will ref obj during the call and will arrange to unref obj when no
// longer needed.
void AssignRefCounted(StringPiece src, core::RefCounted* obj, string* out);

// Copy contents of src to dst[0,src.size()-1].
inline void CopyToArray(const string& src, char* dst) {
  memcpy(dst, src.data(), src.size());
}

// Copy subrange [pos:(pos + n)) from src to dst. If pos >= src.size() the
// result is empty. If pos + n > src.size() the subrange [pos, size()) is
// copied.
inline void CopySubrangeToArray(const string& src, size_t pos, size_t n,
                                char* dst) {
  if (pos >= src.size()) return;
  memcpy(dst, src.data() + pos, std::min(n, src.size() - pos));
}

// Store encoding of strings[0..n-1] in *out.
void EncodeStringList(const tstring* strings, int64 n, string* out);

// Decode n strings from src and store in strings[0..n-1].
// Returns true if successful, false on parse error.
bool DecodeStringList(const string& src, tstring* strings, int64 n);

// Assigns base[0..bytes-1] to *s
void CopyFromArray(string* s, const char* base, size_t bytes);

// Encodes sequences of strings and serialized protocol buffers into a string.
// Normal usage consists of zero or more calls to Append() and a single call to
// Finalize().
class StringListEncoder {
 public:
  virtual ~StringListEncoder() = default;

  // Encodes the given protocol buffer. This may not be called after Finalize().
  virtual void Append(const protobuf::MessageLite& m) = 0;

  // Encodes the given string. This may not be called after Finalize().
  virtual void Append(const string& s) = 0;

  // Signals end of the encoding process. No other calls are allowed after this.
  virtual void Finalize() = 0;
};

// Decodes a string into sequences of strings (which may represent serialized
// protocol buffers). Normal usage involves a single call to ReadSizes() in
// order to retrieve the length of all the strings in the sequence. For each
// size returned a call to Data() is expected and will return the actual
// string.
class StringListDecoder {
 public:
  virtual ~StringListDecoder() = default;

  // Populates the given vector with the lengths of each string in the sequence
  // being decoded. Upon returning the vector is guaranteed to contain as many
  // elements as there are strings in the sequence.
  virtual bool ReadSizes(std::vector<uint32>* sizes) = 0;

  // Returns a pointer to the next string in the sequence, then prepares for the
  // next call by advancing 'size' characters in the sequence.
  virtual const char* Data(uint32 size) = 0;
};

std::unique_ptr<StringListEncoder> NewStringListEncoder(string* out);
std::unique_ptr<StringListDecoder> NewStringListDecoder(const string& in);

#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
// Store src contents in *out.  If backing memory for src is shared with *out,
// will ref obj during the call and will arrange to unref obj when no
// longer needed.
void AssignRefCounted(StringPiece src, core::RefCounted* obj, Cord* out);

// TODO(kmensah): Macro guard this with a check for Cord support.
inline void CopyToArray(const Cord& src, char* dst) { src.CopyToArray(dst); }

// Copy n bytes of src to dst. If pos >= src.size() the result is empty.
// If pos + n > src.size() the subrange [pos, size()) is copied.
inline void CopySubrangeToArray(const Cord& src, int64 pos, int64 n,
                                char* dst) {
  src.Subcord(pos, n).CopyToArray(dst);
}

// Store encoding of strings[0..n-1] in *out.
void EncodeStringList(const tstring* strings, int64 n, Cord* out);

// Decode n strings from src and store in strings[0..n-1].
// Returns true if successful, false on parse error.
bool DecodeStringList(const Cord& src, tstring* strings, int64 n);

// Assigns base[0..bytes-1] to *c
void CopyFromArray(Cord* c, const char* base, size_t bytes);

std::unique_ptr<StringListEncoder> NewStringListEncoder(Cord* out);
std::unique_ptr<StringListDecoder> NewStringListDecoder(const Cord& in);
#endif  // defined(TENSORFLOW_PROTOBUF_USES_CORD)

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_TENSOR_CODING_H_
