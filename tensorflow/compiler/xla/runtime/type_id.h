/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_TYPE_ID_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_TYPE_ID_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "mlir/Support/TypeID.h"  // from @llvm-project

namespace mlir {
// Allow TypeID to be used as a key in ABSL map containers.
template <typename H>
H AbslHashValue(H h, const TypeID& type_id) {
  return H::combine(std::move(h), type_id.getAsOpaquePointer());
}
}  // namespace mlir

namespace xla {
namespace runtime {

using ::mlir::TypeID;  // NOLINT

//===----------------------------------------------------------------------===//
// Mapping from TypeID to unique symbol name.
//===----------------------------------------------------------------------===//

// A mapping from TypeID to unique type name, because we do not have any
// standard tools to get a type name in C++. We rely on this mapping to declare
// external symbols corresponding to type ids in compiled XLA executables.
class TypeIDNameRegistry {
 public:
  TypeIDNameRegistry() = default;
  ~TypeIDNameRegistry() = default;

  template <typename T>
  void Register(std::string_view type_name) {
    auto inserted = type_id_name_map_.try_emplace(TypeID::get<T>(), type_name);
    assert(inserted.second && "duplicate typeid name registration");
    (void)inserted;
  }

  std::string_view FindTypeIDSymbolName(TypeID type_id);

  void ForEach(std::function<void(std::string_view, TypeID)> f) const {
    for (auto& kv : type_id_name_map_) f(kv.second, kv.first);
  }

 private:
  absl::flat_hash_map<TypeID, std::string> type_id_name_map_;
};

//===----------------------------------------------------------------------===//
// DenseTypeID for generating sequential type ids.
//===----------------------------------------------------------------------===//

// Forward declare
template <typename IdSet>
class DenseTypeId;

namespace internal {
template <typename IdSet, typename T>
class DenseTypeIdResolver {
  friend DenseTypeId<IdSet>;
  static size_t get();
};
}  // namespace internal

// Use this as DenseTypeId<some_type_specific_to_your_use>, that way you are
// guaranteed to get contiguous IDs starting at 0 unique to your particular
// use case, as would be appropriate to use for indexes into a vector.
// 'some_type_specific_to_your_use' could (e.g.) be the class that contains
// that particular vector.
template <typename IdSet>
class DenseTypeId {
 public:
  template <typename T>
  static size_t get() {
    return internal::DenseTypeIdResolver<IdSet, T>::get();
  }

 private:
  // Partial template specialization can't be declared as a friend, so we
  // declare all `DenseTypeIdResolver` as a friend.
  template <typename OtherIdSet, typename T>
  friend class internal::DenseTypeIdResolver;

  static size_t next_id() {
    return next_id_.fetch_add(1, std::memory_order_relaxed);
  }

  static std::atomic<size_t> next_id_;
};

template <typename IdSet>
std::atomic<size_t> DenseTypeId<IdSet>::next_id_;

namespace internal {
template <typename IdSet, typename T>
size_t DenseTypeIdResolver<IdSet, T>::get() {
  static const size_t id = DenseTypeId<IdSet>::next_id();
  return id;
}
}  // namespace internal

}  // namespace runtime
}  // namespace xla

// Declare/define an explicit specialization for DenseTypeId.
//
// This forces the compiler to assign a dense type id for the given type and
// avoids checking the static initialization guard if the type id defined as a
// static variable (default implementation of the DenseTypeIdResolver).
//
//  Example:
//
//  // Foo.h
//  struct FooIdSet {};
//
//  XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(FooIdSet, int32_t);
//
//  // Foo.cpp
//  XLA_RUNTIME_DEFINE_EXPLICIT_DENSE_TYPE_ID(FooIdSet, int32_t);
//
#define XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(ID_SET, T) \
  namespace xla {                                             \
  namespace runtime {                                         \
  namespace internal {                                        \
                                                              \
  template <>                                                 \
  class DenseTypeIdResolver<ID_SET, T> {                      \
   public:                                                    \
    static size_t get() { return id; }                        \
                                                              \
   private:                                                   \
    static size_t id;                                         \
  };                                                          \
                                                              \
  } /* namespace internal */                                  \
  } /* namespace runtime */                                   \
  } /* namespace xla */

#define XLA_RUNTIME_DEFINE_EXPLICIT_DENSE_TYPE_ID(ID_SET, T)                  \
  namespace xla {                                                             \
  namespace runtime {                                                         \
  namespace internal {                                                        \
                                                                              \
  size_t DenseTypeIdResolver<ID_SET, T>::id = DenseTypeId<ID_SET>::next_id(); \
                                                                              \
  } /* namespace internal */                                                  \
  } /* namespace runtime */                                                   \
  } /* namespace xla */

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_TYPE_ID_H_
