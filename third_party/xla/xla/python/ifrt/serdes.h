/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/serdes.pb.h"

namespace xla {
namespace ifrt {

// Base class for serializable IFRT types.
class Serializable : public llvm::RTTIExtends<Serializable, llvm::RTTIRoot> {
 public:
  static char ID;  // NOLINT
};

// Base class for deserialization options to be passed to `Deserialize`.
struct DeserializeOptions
    : llvm::RTTIExtends<DeserializeOptions, llvm::RTTIRoot> {
  static char ID;  // NOLINT
};

// Serializer and deserializer implementations for one `Serializable` type.
// This, combined with the registration mechanism below, allows extending IFRT
// object serialization without having to extend the base IFRT itself.
class SerDes : public llvm::RTTIExtends<SerDes, llvm::RTTIRoot> {
 public:
  // Type name. Must be unique. The recommended convention is to use the fully
  // qualified type name of the class that implements `Serializable`.
  virtual absl::string_view type_name() const = 0;

  virtual absl::StatusOr<std::string> Serialize(Serializable& serializable) = 0;

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

// Serializes the given `Serializable` object. The returned proto message can be
// deserialized by `Deserialize`.
//
// Returns an error if the `Serializable` type does not have a corresponding
// `SerDes` registered or the `SerDes` returns an error.
absl::StatusOr<Serialized> Serialize(Serializable& serializable);

// Deserializes the given proto message produced by `Serialize()` back to a
// `Serializable` object of the original type.
//
// `options` is passed as-is to `SerDes::Deserialize()`, so it can be nullptr as
// long as the `SerDes` implementation can handle nullptr options.
//
// Returns an error if the `Serializable` type from which `serialized` was
// produced does not have a corresponding `SerDes` registered or the `SerDes`
// returns an error.
absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
    const Serialized& serialized, std::unique_ptr<DeserializeOptions> options);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SERDES_H_
