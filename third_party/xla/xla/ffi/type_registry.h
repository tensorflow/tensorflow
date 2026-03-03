/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_FFI_TYPE_REGISTRY_H_
#define XLA_FFI_TYPE_REGISTRY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/util.h"

namespace xla::ffi {

namespace internal {
struct TypeRegistrationMap;  // Forward declare.
}  // namespace internal

// XLA FFI has a several APIs that can take ownership of an opaque user data,
// and then passes it back to the FFI handler via the execution context:
//
// 1. UserData put into the ExecutionContext
// 2. XLA FFI handler state created at instantiation stage
//
// To avoid passing the pointer of the wrong type and guard against undefined
// behavior, that likely will manifest as hard to debug crashes, we rely on
// type id to guarantee that we pass pointers of the correct type.
//
// There are two kinds of type ids:
//
// 1. External type id. When FFI handlers defined in a dynamically loaded
//    library, they must register types used in the execution context ahead
//    of time and explicitly get a unique type id for them.
//
// 2. Internal type id. When FFI handler defined in the same binary we rely
//    on the automatic type registration in the global static registry.
//
// TypeInfo defines a set of functions that allow XLA runtime to manipulate
// external types. For user data, that is forwarded to FFI handlers, they all
// can be `nullptr` as XLA runtime doesn't manage their lifetime. For stateful
// handlers, XLA runtime at least must know how to destroy the state when XLA
// executable is destroyed.
//
// WARNING: Some of the APIs accept external `TypeRegistrationMap` argument, and
// this is an internal implementation detail of how JAX and XLA work when they
// linked dynamically. For XLA:FFI to work correctly the XLA runtime and FFI
// handlers implementations should agree on the instance of type registry,
// otherwise XLA runtime can try to destroy FFI state using incorrect deleter,
// and it will lead to hard to debug run time crashes. `XLA_FFI_InternalApi`
// provides access to static instances of type and handler registries that are
// linked onto the object file that defines the XLA:FFI API itself, and `ffi.h`
// implementation carefully uses these instances when it works with types.
//
// All of these linking intricacies are gone when XLA and JAX linked statically
// into single binary as all global static objects are automatically deduped
// (this is how Google works internally). And we jumping through hoops here only
// to be able to "safely" use internal FFI headers in OSS projects. Safely put
// in quotes because it's still unsafe as the user should guarantee that they
// build from the same commit as XLA and JAX with exactly the same toolchain,
// but if they do that, everything should just work!
class TypeRegistry {
 public:
  // Unique (within a process) identifier for a type.
  TSL_LIB_GTL_DEFINE_INT_TYPE(TypeId, int64_t);

  static constexpr TypeId kUnknownTypeId = TypeId(0);

  // Pointers to functions that allow XLA runtime to manipulate external types.
  struct TypeInfo {
    using Deleter = void (*)(void*);
    using Serializer = absl::StatusOr<std::string> (*)(const void*);
    using Deserializer =
        absl::StatusOr<std::unique_ptr<void, Deleter>> (*)(absl::string_view);

    bool operator==(const TypeInfo& other) const {
      return deleter == other.deleter && serializer == other.serializer &&
             deserializer == other.deserializer;
    }
    bool operator!=(const TypeInfo& other) const { return !(*this == other); }

    Deleter deleter = nullptr;
    Serializer serializer = nullptr;
    Deserializer deserializer = nullptr;
  };

  // A type registration record.
  struct TypeRegistration {
    TypeId type_id;
    TypeInfo type_info;
  };

  // To declare a type `T` as serializable and deserializable, define a
  // specialization of `TypeSerDes<T>` with `Serialize` and `Deserialize` apis.
  //
  //   template <>
  //   struct TypeSerDes<T> : public std::true_type {
  //     static absl::StatusOr<std::string> Serialize(const T& value);
  //     static absl::StatusOr<std::unique_ptr<T>> Deserialize(
  //       absl::string_view data);
  // };
  //
  template <typename T>
  struct SerDes : public std::false_type {};

  // Returns type name for a given type id. Returns an error if type id is not
  // registered. Works for both external and internal type ids.
  static absl::StatusOr<absl::string_view> GetTypeName(TypeId type_id);

  // Returns type id for a given type name. Returns an error if type is
  // not registered. Works for both external and internal type ids.
  static absl::StatusOr<TypeId> GetTypeId(absl::string_view name);

  // Returns type info for a given type id. Returns an error if type id is not
  // registered. Works for both external and internal type ids.
  static absl::StatusOr<TypeInfo> GetTypeInfo(TypeId type_id);

  // Assigns a unique type id to a type with a given name. Returns an error if a
  // type with a given name is already registered in the process.
  static absl::StatusOr<TypeId> AssignTypeId(absl::string_view name,
                                             TypeInfo type_info);

  // Registers type with a given name and type id. Type id is provided by the
  // caller, and must be unique. Returns an error if a type with a given name is
  // already registered with a different type id.
  static absl::Status RegisterTypeId(absl::string_view name, TypeId type_id,
                                     TypeInfo type_info);

  // Returns a type name for a given type. For internal type ids only.
  template <typename T>
  static absl::string_view GetTypeName();

  // Returns a type id for a given type. For internal type ids only.
  template <typename T>
  static TypeId GetTypeId();

  // Returns type info for a given type id. For internal type ids only.
  template <typename T>
  static TypeInfo GetTypeInfo();

  // Serializes a value of a given type. For internal type ids only.
  template <typename T>
  static absl::StatusOr<std::string> Serialize(const T& value);

  // Deserializes a value of a given type. For internal type ids only.
  template <typename T>
  static absl::StatusOr<std::unique_ptr<T>> Deserialize(absl::string_view data);

  // WARNING: Internal APIs that accept custom type registration map.

  // Returns already assigned type id for a given type name or assigns a new
  // one. Returns an error if a type with a given name is already registered in
  // the process and it has a different type info.
  static absl::StatusOr<TypeId> GetOrAssignTypeId(
      internal::TypeRegistrationMap& registry, absl::string_view name,
      TypeInfo type_info);

  // A template that automatically infers type name and info for `T`.
  template <typename T>
  static absl::StatusOr<TypeId> GetOrAssignTypeId(
      internal::TypeRegistrationMap& registry);

 private:
  static TypeId GetNextTypeId();
};

namespace internal {

// `TypeRegistry` is implemented on top of the `TypeRegistrationMap` that
// holds registrations for all types in the process. It is critical that
// FFI handlers and XLA runtime share the same instance of this map.
struct TypeRegistrationMap {
  absl::Mutex mu;
  absl::flat_hash_map<std::string, TypeRegistry::TypeRegistration> map
      ABSL_GUARDED_BY(mu);
};

// Returns a reference to a static instance of a type registration map.
TypeRegistrationMap& StaticTypeRegistrationMap();

}  // namespace internal

//===---------------------------------------------------------------------===///
// TypeRegistry templates implementation.
//===---------------------------------------------------------------------===///

template <typename T>
absl::string_view TypeRegistry::GetTypeName() {
  return typeid(T).name();
}

template <typename T>
TypeRegistry::TypeId TypeRegistry::GetTypeId() {
  static const absl::NoDestructor<absl::StatusOr<TypeId>> id(
      AssignTypeId(GetTypeName<T>(), GetTypeInfo<T>()));
  return **id;
}

template <typename T>
TypeRegistry::TypeInfo TypeRegistry::GetTypeInfo() {
  // Define deleter as a static member, because it's always available for the
  // internal types.
  static TypeInfo::Deleter deleter =
      +[](void* state) { delete tsl::safe_reinterpret_cast<T*>(state); };

  // Serializer and deserializer are defined only if `T` opts in to the
  // serializable via the `SerDes` specialization.
  TypeInfo::Serializer serializer = nullptr;
  TypeInfo::Deserializer deserializer = nullptr;

  if constexpr (SerDes<T>::value) {
    serializer = +[](const void* value) {
      return SerDes<T>::Serialize(*tsl::safe_reinterpret_cast<const T*>(value));
    };

    deserializer = +[](absl::string_view data)
        -> absl::StatusOr<std::unique_ptr<void, TypeInfo::Deleter>> {
      TF_ASSIGN_OR_RETURN(auto value, SerDes<T>::Deserialize(data));
      return std::unique_ptr<void, TypeInfo::Deleter>(value.release(), deleter);
    };
  }

  return TypeInfo{deleter, serializer, deserializer};
}

template <typename T>
absl::StatusOr<std::string> TypeRegistry::Serialize(const T& value) {
  TypeInfo type_info = GetTypeInfo<T>();
  if (type_info.serializer == nullptr) {
    return FailedPrecondition(
        "Type is not serializable. Did you forget to specialize "
        "TypeRegistry::SerDes<T>?");
  }
  return type_info.serializer(&value);
}

template <typename T>
absl::StatusOr<std::unique_ptr<T>> TypeRegistry::Deserialize(
    absl::string_view data) {
  TypeInfo type_info = GetTypeInfo<T>();
  if (type_info.deserializer == nullptr) {
    return FailedPrecondition(
        "Type is not deserializable. Did you forget to specialize "
        "TypeRegistry::SerDes<T>?");
  }
  TF_ASSIGN_OR_RETURN(auto ptr, type_info.deserializer(data));
  return std::unique_ptr<T>(tsl::safe_reinterpret_cast<T*>(ptr.release()));
}

template <typename T>
absl::StatusOr<TypeRegistry::TypeId> TypeRegistry::GetOrAssignTypeId(
    internal::TypeRegistrationMap& registry) {
  return GetOrAssignTypeId(registry, GetTypeName<T>(), GetTypeInfo<T>());
}

}  // namespace xla::ffi

#endif  // XLA_FFI_TYPE_REGISTRY_H_
