/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_SERDES_BASE_H_
#define XLA_BACKENDS_CPU_RUNTIME_SERDES_BASE_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"

namespace xla::cpu {

template <typename T>
class SerDesBase {
 public:
  virtual absl::StatusOr<std::string> Serialize(const T& serializable) = 0;
  virtual absl::StatusOr<std::unique_ptr<T>> Deserialize(
      const std::string& serialized) = 0;

  virtual ~SerDesBase() = default;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_SERDES_BASE_H_
