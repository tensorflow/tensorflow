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
#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_STATUS_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_STATUS_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tsl {
#if ABSL_HAVE_BUILTIN(__builtin_LINE) && ABSL_HAVE_BUILTIN(__builtin_FILE)
#define TF_INTERNAL_HAVE_BUILTIN_LINE_FILE 1
#endif

class SourceLocationImpl {
 public:
  uint32_t line() const { return line_; }
  const char* file_name() const { return file_name_; }

#ifdef TF_INTERNAL_HAVE_BUILTIN_LINE_FILE
  static SourceLocationImpl current(uint32_t line = __builtin_LINE(),
                                    const char* file_name = __builtin_FILE()) {
    return SourceLocationImpl(line, file_name);
  }
#else
  static SourceLocationImpl current(uint32_t line = 0,
                                    const char* file_name = nullptr) {
    return SourceLocationImpl(line, file_name);
  }
#endif
 private:
  SourceLocationImpl(uint32_t line, const char* file_name)
      : line_(line), file_name_(file_name) {}
  uint32_t line_;
  const char* file_name_;
};

namespace internal {

inline absl::Status MakeAbslStatus(
    ::tensorflow::error::Code code, absl::string_view message,
    absl::Span<const SourceLocationImpl>,
    SourceLocationImpl loc = SourceLocationImpl::current()) {
  return absl::Status(static_cast<absl::StatusCode>(code), message);
}

inline absl::Span<const SourceLocationImpl> GetSourceLocations(
    const absl::Status& status) {
  return {};
}

}  // namespace internal

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_DEFAULT_STATUS_H_
