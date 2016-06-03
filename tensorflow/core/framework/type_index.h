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

#ifndef TENSORFLOW_FRAMEWORK_TYPE_INDEX_H_
#define TENSORFLOW_FRAMEWORK_TYPE_INDEX_H_

#include <string>
#ifdef __GXX_RTTI
#include <typeindex>
#include <typeinfo>
#endif  // __GXX_RTTI

namespace tensorflow {

// On some platforms, we would like to avoid using RTTI in order to have smaller
// binary sizes. The following #ifdef section provides a non-RTTI
// replacement for std::type_index (with a minimal set of functions needed by
// the TensorFlow framework, and more can be added if necessary).
#ifndef __GXX_RTTI

// A thin TypeIndex class that mimics std::type_index but does not use RTTI. As
// a result, it does not provide the actual name of the type, and only returns a
// pre-baked string specifying that RTTI is disabled.
// The hash code provided in this class is unique for each class. However, it is
// generated at runtime so this hash code should not be serialized - the value
// for the same type can change from run to run.
class TypeIndex {
 public:
  TypeIndex(const TypeIndex& src) : hash_(src.hash_) {}
  TypeIndex& operator=(const TypeIndex& src) {
    hash_ = src.hash_;
    return *this;
  }
  bool operator==(const TypeIndex& rhs) const { return (hash_ == rhs.hash_); }
  bool operator!=(const TypeIndex& rhs) const { return (hash_ != rhs.hash_); }
  ~TypeIndex() {}

  const char* name() const { return "[RTTI disabled for Android]"; }
  uint64 hash_code() const { return hash_; }

  // Returns a TypeIndex object that corresponds to a typename.
  template <typename T>
  static TypeIndex Make() {
    static bool hash_bit[1];
    return TypeIndex(static_cast<uint64>(reinterpret_cast<intptr_t>(hash_bit)));
  }

 private:
  // We hide the constructor of the TypeIndex class. Use the templated
  // Make<T>() function to create a TypeIndex object.
  TypeIndex(const uint64 hash) : hash_(hash) {}
  uint64 hash_;
};

template <typename T>
inline TypeIndex MakeTypeIndex() {
  return TypeIndex::Make<T>();
}

#else  // __GXX_RTTI

// In the presence of RTTI, we will simply delegate to std::type_index for
// runtime type inference.
typedef std::type_index TypeIndex;
template <typename T>
inline TypeIndex MakeTypeIndex() {
  return TypeIndex(typeid(T));
}

#endif  // __GXX_RTTI
}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_TYPE_INDEX_H_
