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

#ifndef TENSORFLOW_CORE_PLATFORM_HASH_H_
#define TENSORFLOW_CORE_PLATFORM_HASH_H_

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <string>

#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

extern uint32 Hash32(const char* data, size_t n, uint32 seed);
extern uint64 Hash64(const char* data, size_t n, uint64 seed);

inline uint64 Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

inline uint64 Hash64(const char* data) { return Hash64(data, ::strlen(data)); }

inline uint64 Hash64(const std::string& str) {
  return Hash64(str.data(), str.size());
}

inline uint64 Hash64(const tstring& str) {
  return Hash64(str.data(), str.size());
}

inline uint64 Hash64Combine(uint64 a, uint64 b) {
  return a ^ (b + 0x9e3779b97f4a7800ULL + (a << 10) + (a >> 4));
}

// Combine two hashes in an order-independent way. This operation should be
// associative and compute the same hash for a collection of elements
// independent of traversal order. Note that it is better to combine hashes
// symmetrically with addition rather than XOR, since (x^x) == 0 but (x+x) != 0.
inline uint64 Hash64CombineUnordered(uint64 a, uint64 b) { return a + b; }

// Hash functor suitable for use with power-of-two sized hashtables.  Use
// instead of std::hash<T>.
//
// In particular, tensorflow::hash is not the identity function for pointers.
// This is important for power-of-two sized hashtables like FlatMap and FlatSet,
// because otherwise they waste the majority of their hash buckets.
//
// The second type argument is only used for SFNIAE below.
template <typename T, typename = void>
struct hash {
  size_t operator()(const T& t) const { return std::hash<T>()(t); }
};

template <typename T>
struct hash<T, typename std::enable_if<std::is_enum<T>::value>::type> {
  size_t operator()(T value) const {
    // This works around a defect in the std::hash C++ spec that isn't fixed in
    // (at least) gcc 4.8.4:
    // http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2148
    //
    // We should be able to remove this and use the default
    // tensorflow::hash<EnumTy>() once we stop building with GCC versions old
    // enough to not have this defect fixed.
    return std::hash<uint64>()(static_cast<uint64>(value));
  }
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
struct hash<tstring> {
  size_t operator()(const tstring& s) const {
    return static_cast<size_t>(Hash64(s.data(), s.size()));
  }
};

template <>
struct hash<StringPiece> {
  size_t operator()(StringPiece sp) const {
    return static_cast<size_t>(Hash64(sp.data(), sp.size()));
  }
};
using StringPieceHasher = ::tensorflow::hash<StringPiece>;

template <typename T, typename U>
struct hash<std::pair<T, U>> {
  size_t operator()(const std::pair<T, U>& p) const {
    return Hash64Combine(hash<T>()(p.first), hash<U>()(p.second));
  }
};

}  // namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::tstring> {
  size_t operator()(const tensorflow::tstring& s) const {
    return static_cast<size_t>(tensorflow::Hash64(s.data(), s.size()));
  }
};
}  // namespace std

#endif  // TENSORFLOW_CORE_PLATFORM_HASH_H_
