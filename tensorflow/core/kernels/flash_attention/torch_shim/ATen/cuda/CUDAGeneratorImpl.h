/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Minimal stand-in for the PyTorch header FlashAttention-2 includes. Only
// at::PhiloxCudaState is needed: seed/offset either by value, or (captured_)
// as pointers to device memory.
#pragma once  // NOLINT(build/header_guard): stands in for a PyTorch header.
#include <cstdint>

namespace at {
struct PhiloxCudaState {
  PhiloxCudaState() = default;
  PhiloxCudaState(uint64_t seed, uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }
  PhiloxCudaState(const uint64_t *seed_dev, const uint64_t *offset_dev) {
    seed_.ptr = reinterpret_cast<int64_t *>(const_cast<uint64_t *>(seed_dev));
    offset_.ptr =
        reinterpret_cast<int64_t *>(const_cast<uint64_t *>(offset_dev));
    captured_ = true;
  }
  union Payload {
    uint64_t val;
    int64_t *ptr;
  };
  Payload seed_{};
  Payload offset_{};
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};
}  // namespace at
