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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TYPE_INDEX_H_
#define TENSORFLOW_CORE_FRAMEWORK_TYPE_INDEX_H_

#include <string>

#if defined(__GXX_RTTI) || defined(_CPPRTTI)
#include <typeinfo>
#endif  // __GXX_RTTI

#include "tensorflow/core/platform/types.h"

#if defined(MACOS) || defined(TARGET_OS_MAC) || defined(PLATFORM_WINDOWS)
#include "tensorflow/core/platform/hash.h"
#endif  // defined(MACOS) || defined(TARGET_OS_MAC) || defined(PLATFORM_WINDOWS)

namespace tensorflow {

// On some platforms, we would like to avoid using RTTI in order to have smaller
// binary sizes. This file provides a thin TypeIndex class that mimics
// std::type_index but does not use RTTI (with a minimal set of functions needed
// by the TensorFlow framework, and more can be added if necessary). In the
// absence of RTTI, it does not provide the actual name of the type, and only
// returns a pre-baked string specifying that RTTI is disabled. The hash code
// provided in this class is unique for each class. However, it is generated at
// runtime so this hash code should not be serialized - the value for the same
// type can change from run to run.
class TypeIndex {
 public:
  TypeIndex(const TypeIndex& src) : hash_(src.hash_), name_(src.name_) {}
  TypeIndex& operator=(const TypeIndex& src) {
    hash_ = src.hash_;
    name_ = src.name_;
    return *this;
  }
  bool operator==(const TypeIndex& rhs) const { return (hash_ == rhs.hash_); }
  bool operator!=(const TypeIndex& rhs) const { return (hash_ != rhs.hash_); }
  ~TypeIndex() {}

  const char* name() const { return name_; }

  uint64 hash_code() const { return hash_; }

  // Returns a TypeIndex object that corresponds to a typename.
  template <typename T>
  static TypeIndex Make() {
    static bool hash_bit[1];

#if defined(__GXX_RTTI) || defined(_CPPRTTI)

#if defined(MACOS) || defined(TARGET_OS_MAC) || defined(PLATFORM_WINDOWS)
    // Use a hash based on the type name to avoid issues due to RTLD_LOCAL on
    // MacOS (b/156979412).
    return TypeIndex(Hash64(typeid(T).name()), typeid(T).name());
#else
    // Use the real type name if we have RTTI.
    return TypeIndex(static_cast<uint64>(reinterpret_cast<intptr_t>(hash_bit)),
                     typeid(T).name());
#endif  // defined(MACOS) || defined(TARGET_OS_MAC) || defined(PLATFORM_WINDOWS)

#else
#if TARGET_OS_OSX
    // Warn MacOS users that not using RTTI can cause problems (b/156979412).
#warning \
    "Compiling with RTTI disabled on MacOS can cause problems when comparing " \
    "types across shared libraries."
#endif  // TARGET_OS_OSX

    // No type names available.
    return TypeIndex(static_cast<uint64>(reinterpret_cast<intptr_t>(hash_bit)),
                     "[RTTI disabled]");
#endif  // __GXX_RTTI
  }

 private:
  // We hide the constructor of the TypeIndex class. Use the templated
  // Make<T>() function to create a TypeIndex object.
  explicit TypeIndex(const uint64 hash, const char* name)
      : hash_(hash), name_(name) {}
  uint64 hash_;
  const char* name_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TYPE_INDEX_H_
