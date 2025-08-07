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

#ifndef XLA_PYTHON_IFRT_IR_UTILS_H_
#define XLA_PYTHON_IFRT_IR_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace ifrt {

// IMPORTANT: All these util functions are visible to IFRT IR users.

// Returns the total device memory in bytes for the given device kind.
absl::StatusOr<int64_t> GetDeviceMemoryInBytes(absl::string_view device_kind);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_UTILS_H_
