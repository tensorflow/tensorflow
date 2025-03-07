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

#ifndef XLA_FFI_TYPE_ID_REGISTRY_H_
#define XLA_FFI_TYPE_ID_REGISTRY_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::ffi {

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
//    on a global static registry to automatically assign type ids.
class TypeIdRegistry {
 public:
  TSL_LIB_GTL_DEFINE_INT_TYPE(TypeId, int64_t);

  static constexpr TypeId kUnknownTypeId = TypeId(0);

  // Assigns a unique type id to an external type with a given name. Returns an
  // error if a type with a given name is already registered in the process.
  static absl::StatusOr<TypeId> AssignExternalTypeId(absl::string_view name);

  // Registers external type with a given name and type id. Type id is provided
  // by the caller, and must be unique. Returns an error if a type with a given
  // name is already registered with a different type id.
  static absl::Status RegisterExternalTypeId(absl::string_view name,
                                             TypeId type_id);

  // Returns a type id for a given type. For internal type ids only.
  template <typename T>
  static TypeId GetTypeId();

 private:
  // We never mix external and internal type ids, so we can use different type
  // id spaces to assign unique ids to each type.
  static TypeId GetNextInternalTypeId();
  static TypeId GetNextExternalTypeId();
};

template <typename T>
TypeIdRegistry::TypeId TypeIdRegistry::GetTypeId() {
  static const TypeId id = GetNextInternalTypeId();
  return id;
}

}  // namespace xla::ffi

#endif  // XLA_FFI_TYPE_ID_REGISTRY_H_
