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

// Simple hash functions used for internal data structures

#ifndef TENSORFLOW_LIB_HASH_HASH_H_
#define TENSORFLOW_LIB_HASH_HASH_H_

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

extern uint32 Hash32(const char* data, size_t n, uint32 seed);
extern uint64 Hash64(const char* data, size_t n, uint64 seed);

inline uint64 Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

inline uint64 Hash64(const string& str) {
  return Hash64(str.data(), str.size());
}

inline uint64 Hash64Combine(uint64 a, uint64 b) {
  return a ^ (b + 0x9e3779b97f4a7800ULL + (a << 10) + (a >> 4));
}

// Hash functor suitable for use with power-of-two sized hashtables.  Use
// instead of std::hash<T>.
//
// In particular, tensorflow::hash is not the identity function for pointers.
// This is important for power-of-two sized hashtables like FlatMap and FlatSet,
// because otherwise they waste the majority of their hash buckets.
template <typename T>
struct hash {
  size_t operator()(const T& t) const { return std::hash<T>()(t); }
};

template <typename T>
struct hash<T*> {
  size_t operator()(const T* t) const {
    // Hash pointers as integers, but bring more entropy to the lower bits.
    size_t k = static_cast<size_t>(reinterpret_cast<uintptr_t>(t));
    return k + (k >> 6);
  }
};

template <>
struct hash<string> {
  size_t operator()(const string& s) const {
    return static_cast<size_t>(Hash64(s));
  }
};

template <>
struct hash<StringPiece> {
  size_t operator()(StringPiece sp) const {
    return static_cast<size_t>(Hash64(sp.data(), sp.size()));
  }
};

template <typename T, typename U>
struct hash<std::pair<T, U>> {
  size_t operator()(const std::pair<T, U>& p) const {
    return Hash64Combine(hash<T>()(p.first), hash<U>()(p.second));
  }
};

// Wrapper around protocol buffer serialization that requests deterministic
// serialization, in particular for Map fields, which serialize in a random
// order by default. Returns true on success.
// Serialization is guaranteed to be deterministic for a given binary only.
// See the following for more details:
// https://github.com/google/protobuf/blob/a1bb147e96b6f74db6cdf3c3fcb00492472dbbfa/src/google/protobuf/io/coded_stream.h#L834
bool SerializeToStringDeterministic(const protobuf::MessageLite& msg,
                                    string* result);

}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_HASH_HASH_H_
