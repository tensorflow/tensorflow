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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_SHARDING_SERDES_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_SHARDING_SERDES_H_

#include <memory>

#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/python/ifrt/serdes.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace ifrt {

class Client;

// Options for deserializing shardings.
struct DeserializeShardingOptions
    : llvm::RTTIExtends<DeserializeShardingOptions, DeserializeOptions> {
  explicit DeserializeShardingOptions(Client* client) : client(client) {}

  static char ID;  // NOLINT

  // The client whose devices will be used by deserialized shardings.
  Client* client;
};

// Casts `DeserializeOptions` into `DeserializeShardingOptions`.
StatusOr<std::unique_ptr<DeserializeShardingOptions>>
GetDeserializeShardingOptions(std::unique_ptr<DeserializeOptions> options);

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_SHARDING_SERDES_H_
