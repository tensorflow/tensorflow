/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_SERDES_H_
#define XLA_PYTHON_IFRT_SERDES_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

// Base class for serialization options to be passed to `Serialize`.
struct SerializeOptions : llvm::RTTIExtends<SerializeOptions, llvm::RTTIRoot> {
  static char ID;  // NOLINT
};

// Base class for deserialization options to be passed to `Deserialize`.
struct DeserializeOptions
    : llvm::RTTIExtends<DeserializeOptions, llvm::RTTIRoot> {
  static char ID;  // NOLINT
};

// Base class for serializable IFRT types.
class Serializable : public llvm::RTTIExtends<Serializable, llvm::RTTIRoot> {
 public:
  static char ID;  // NOLINT

  // Expected `SerializeOptions` and `DeserializeOptions` types. A subclass of
  // `Serializable` can customize them.
  using SerializeOptions = ::xla::ifrt::SerializeOptions;
  using DeserializeOptions = ::xla::ifrt::DeserializeOptions;
};

// Serializer and deserializer implementations for one `Serializable` type.
// This, combined with the registration mechanism below, allows extending IFRT
// object serialization without having to extend the base IFRT itself.
class SerDes : public llvm::RTTIExtends<SerDes, llvm::RTTIRoot> {
 public:
  // Type name. Must be unique. The recommended convention is to use the fully
  // qualified type name of the class that implements `Serializable`.
  virtual absl::string_view type_name() const = 0;

  virtual absl::StatusOr<std::string> Serialize(
      Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) = 0;

  virtual absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) = 0;

  static char ID;  // NOLINT
};

// Registers a `SerDes` implementation to be used for the given `Serializable`
// type. `type_id` must be returned from `SerializableT::classID()`.
//
// Typically, this function should be called from a module initializer.
// Registering a serdes more than once for the same type crashes the process.
void RegisterSerDes(const void* type_id, std::unique_ptr<SerDes> serdes);

// Syntactic sugar of the above function that takes a `Serializable` class as a
// template argument.
template <typename T>
void RegisterSerDes(std::unique_ptr<SerDes> serdes) {
  static_assert(std::is_base_of_v<Serializable, T>,
                "Types must implement `xla::ifrt::Serializable` to have a "
                "serdes implementation");
  RegisterSerDes(T::classID(), std::move(serdes));
}

namespace serdes_internal {

// Internal implementation of Deserialize(). Performs deserialization with type
// erased.
absl::StatusOr<std::unique_ptr<Serializable>> DeserializeUnchecked(
    const Serialized& serialized, std::unique_ptr<DeserializeOptions> options);

}  // namespace serdes_internal

// Serializes the given `Serializable` object. The returned proto message can be
// deserialized by `Deserialize`.
//
// Returns an error if the `Serializable` type does not have a corresponding
// `SerDes` registered or the `SerDes` returns an error.
absl::StatusOr<Serialized> Serialize(Serializable& serializable,
                                     std::unique_ptr<SerializeOptions> options);

// Deserializes the given proto message produced by `Serialize()` back to an
// object of type `InterfaceType`, where `serialized.type_name()` is expected to
// be the same type or a subclass of `InterfaceType`.
//
// `options` is passed as-is to `SerDes::Deserialize()`, so it can be nullptr as
// long as the `SerDes` implementation can handle nullptr options.
//
// Returns an error if the type indicated by `serialized.type_name()` does not
// have a corresponding `SerDes` registered or the if the registered `SerDes`
// returns an error.
template <typename InterfaceType>
absl::StatusOr<std::unique_ptr<InterfaceType>> Deserialize(
    const Serialized& serialized,
    std::unique_ptr<typename InterfaceType::DeserializeOptions> options) {
  TF_ASSIGN_OR_RETURN(auto result, serdes_internal::DeserializeUnchecked(
                                       serialized, std::move(options)));
  if (!llvm::isa<InterfaceType>(result.get())) {
    return absl::InternalError(
        "Unexpected Serializable type after deserialization");
  }
  return std::unique_ptr<InterfaceType>(
      static_cast<InterfaceType*>(result.release()));
}

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SERDES_H_
