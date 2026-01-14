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

#ifndef XLA_STREAM_EXECUTOR_MEMORY_SPACE_H_
#define XLA_STREAM_EXECUTOR_MEMORY_SPACE_H_

#include <cstdint>

#include "absl/base/macros.h"

namespace stream_executor {

// Identifies the memory space where a physical allocation resides.
enum class MemorySpace : uint8_t {
  kDevice = 0,
  kUnified,
  kCollective,
  kP2P,
  kHost = 5,
};

using MemoryType ABSL_DEPRECATE_AND_INLINE() = MemorySpace;

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MEMORY_SPACE_H_
