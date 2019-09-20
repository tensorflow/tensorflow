/*
 * Copyright 2015 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLATBUFFERS_HASH_H_
#define FLATBUFFERS_HASH_H_

#include <cstdint>
#include <cstring>

#include "flatbuffers/flatbuffers.h"

namespace flatbuffers {

template<typename T> struct FnvTraits {
  static const T kFnvPrime;
  static const T kOffsetBasis;
};

template<> struct FnvTraits<uint32_t> {
  static const uint32_t kFnvPrime = 0x01000193;
  static const uint32_t kOffsetBasis = 0x811C9DC5;
};

template<> struct FnvTraits<uint64_t> {
  static const uint64_t kFnvPrime = 0x00000100000001b3ULL;
  static const uint64_t kOffsetBasis = 0xcbf29ce484222645ULL;
};

template<typename T> T HashFnv1(const char *input) {
  T hash = FnvTraits<T>::kOffsetBasis;
  for (const char *c = input; *c; ++c) {
    hash *= FnvTraits<T>::kFnvPrime;
    hash ^= static_cast<unsigned char>(*c);
  }
  return hash;
}

template<typename T> T HashFnv1a(const char *input) {
  T hash = FnvTraits<T>::kOffsetBasis;
  for (const char *c = input; *c; ++c) {
    hash ^= static_cast<unsigned char>(*c);
    hash *= FnvTraits<T>::kFnvPrime;
  }
  return hash;
}

template <> inline uint16_t HashFnv1<uint16_t>(const char *input) {
  uint32_t hash = HashFnv1<uint32_t>(input);
  return (hash >> 16) ^ (hash & 0xffff);
}

template <> inline uint16_t HashFnv1a<uint16_t>(const char *input) {
  uint32_t hash = HashFnv1a<uint32_t>(input);
  return (hash >> 16) ^ (hash & 0xffff);
}

template <typename T> struct NamedHashFunction {
  const char *name;

  typedef T (*HashFunction)(const char *);
  HashFunction function;
};

const NamedHashFunction<uint16_t> kHashFunctions16[] = {
  { "fnv1_16",  HashFnv1<uint16_t> },
  { "fnv1a_16", HashFnv1a<uint16_t> },
};

const NamedHashFunction<uint32_t> kHashFunctions32[] = {
  { "fnv1_32", HashFnv1<uint32_t> },
  { "fnv1a_32", HashFnv1a<uint32_t> },
};

const NamedHashFunction<uint64_t> kHashFunctions64[] = {
  { "fnv1_64", HashFnv1<uint64_t> },
  { "fnv1a_64", HashFnv1a<uint64_t> },
};

inline NamedHashFunction<uint16_t>::HashFunction FindHashFunction16(
    const char *name) {
  std::size_t size = sizeof(kHashFunctions16) / sizeof(kHashFunctions16[0]);
  for (std::size_t i = 0; i < size; ++i) {
    if (std::strcmp(name, kHashFunctions16[i].name) == 0) {
      return kHashFunctions16[i].function;
    }
  }
  return nullptr;
}

inline NamedHashFunction<uint32_t>::HashFunction FindHashFunction32(
    const char *name) {
  std::size_t size = sizeof(kHashFunctions32) / sizeof(kHashFunctions32[0]);
  for (std::size_t i = 0; i < size; ++i) {
    if (std::strcmp(name, kHashFunctions32[i].name) == 0) {
      return kHashFunctions32[i].function;
    }
  }
  return nullptr;
}

inline NamedHashFunction<uint64_t>::HashFunction FindHashFunction64(
    const char *name) {
  std::size_t size = sizeof(kHashFunctions64) / sizeof(kHashFunctions64[0]);
  for (std::size_t i = 0; i < size; ++i) {
    if (std::strcmp(name, kHashFunctions64[i].name) == 0) {
      return kHashFunctions64[i].function;
    }
  }
  return nullptr;
}

}  // namespace flatbuffers

#endif  // FLATBUFFERS_HASH_H_
