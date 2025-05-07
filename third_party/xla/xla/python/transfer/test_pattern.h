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
#ifndef XLA_PYTHON_TRANSFER_TEST_PATTERN_H_
#define XLA_PYTHON_TRANSFER_TEST_PATTERN_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace aux::tests {

std::vector<int32_t> CreateTestPattern(size_t offset, size_t length);

absl::StatusOr<xla::ifrt::ArrayRef> CopyTestPatternToDevice(
    xla::ifrt::Client* client, xla::ifrt::Device* dest_device,
    const std::vector<int32_t>& pattern);

}  // namespace aux::tests

#endif  // XLA_PYTHON_TRANSFER_TEST_PATTERN_H_
