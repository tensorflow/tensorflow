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

// Endian-neutral encoding:
// * Fixed-length numbers are encoded with least-significant byte first
// * In addition we support variable length "varint" encoding
// * Strings are encoded prefixed by their length in varint format

#ifndef TENSORFLOW_CORE_LIB_CORE_CODING_H_
#define TENSORFLOW_CORE_LIB_CORE_CODING_H_

#include "tensorflow/core/lib/core/raw_coding.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace core {

// Maximum number of bytes occupied by a varint32.
static const int kMaxVarint32Bytes = 5;

// Maximum number of bytes occupied by a varint64.
static const int kMaxVarint64Bytes = 10;

// Lower-level versions of Put... that write directly into a character buffer
// REQUIRES: dst has enough space for the value being written
extern void EncodeFixed16(char* dst, uint16 value);
extern void EncodeFixed32(char* dst, uint32 value);
extern void EncodeFixed64(char* dst, uint64 value);
extern void PutFixed16(string* dst, uint16 value);
extern void PutFixed32(string* dst, uint32 value);
extern void PutFixed64(string* dst, uint64 value);

extern void PutVarint32(string* dst, uint32 value);
extern void PutVarint64(string* dst, uint64 value);

extern bool GetVarint32(StringPiece* input, uint32* value);
extern bool GetVarint64(StringPiece* input, uint64* value);

extern const char* GetVarint32Ptr(const char* p, const char* limit, uint32* v);
extern const char* GetVarint64Ptr(const char* p, const char* limit, uint64* v);

// Internal routine for use by fallback path of GetVarint32Ptr
extern const char* GetVarint32PtrFallback(const char* p, const char* limit,
                                          uint32* value);
inline const char* GetVarint32Ptr(const char* p, const char* limit,
                                  uint32* value) {
  if (p < limit) {
    uint32 result = *(reinterpret_cast<const unsigned char*>(p));
    if ((result & 128) == 0) {
      *value = result;
      return p + 1;
    }
  }
  return GetVarint32PtrFallback(p, limit, value);
}

extern char* EncodeVarint32(char* dst, uint32 v);
extern char* EncodeVarint64(char* dst, uint64 v);

// Returns the length of the varint32 or varint64 encoding of "v"
extern int VarintLength(uint64_t v);

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_CODING_H_
