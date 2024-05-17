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
#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_SYNC_FENCE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_SYNC_FENCE_H_

#include <optional>
#include <variant>

#include "absl/types/span.h"

namespace tflite::delegates::utils {

// Blocks until all file descriptors have been signalled, or returns an error
// (signified by an instance with no value).
std::optional<std::monostate> WaitForAllFds(absl::Span<const int> fds);

// Returns (without blocking) `true` if all the provided file descriptors are
// signalled, `false` if at least one file descriptor is not yet signalled, or
// an error (indicated by an instance with no value).
std::optional<bool> AreAllFdsSignalled(absl::Span<const int> fds);

}  // namespace tflite::delegates::utils

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_SYNC_FENCE_H_
