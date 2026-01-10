/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"

#include <cstdarg>
#include <cstdio>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"

namespace mlir::TFL {

int AbslErrorReporter::Report(const char* format, va_list args) {
  // Use bounded formatting to avoid stack overflows. Cap output size to avoid
  // unbounded allocations on attacker-controlled inputs.
  constexpr size_t kInitialBufferSize = 1024;
  constexpr size_t kMaxBufferSize = 64 * 1024;

  std::vector<char> buffer(kInitialBufferSize);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
  va_list args_copy;
  va_copy(args_copy, args);
  int needed = vsnprintf(buffer.data(), buffer.size(), format, args_copy);
  va_end(args_copy);

  if (needed < 0) {
    LOG(ERROR) << "Failed to format error message.";
    return needed;
  }

  // If truncated, grow (up to a cap) and reformat with a fresh va_list copy.
  if (static_cast<size_t>(needed) >= buffer.size()) {
    const size_t required_size = static_cast<size_t>(needed) + 1;  // incl. NUL
    const bool will_truncate = required_size > kMaxBufferSize;
    buffer.resize(will_truncate ? kMaxBufferSize : required_size);

    va_copy(args_copy, args);
    (void)vsnprintf(buffer.data(), buffer.size(), format, args_copy);
    va_end(args_copy);

    if (will_truncate) {
      LOG(ERROR) << buffer.data() << " [truncated]";
      return 0;
    }
  }
#pragma clang diagnostic pop
  LOG(ERROR) << buffer.data();
  return 0;
}

tflite::ErrorReporter* GetAbslErrorReporter() {
  static AbslErrorReporter* error_reporter = new AbslErrorReporter;
  return error_reporter;
}

}  // namespace mlir::TFL
