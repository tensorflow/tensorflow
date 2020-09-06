/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_HUMAN_READABLE_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_HUMAN_READABLE_H_

#include <assert.h>

#include <limits>

#include "absl/strings/str_format.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {
namespace port {

class HumanReadableNumBytes {
 public:
  static std::string ToString(int64 num_bytes) {
    if (num_bytes == std::numeric_limits<int64>::min()) {
      // Special case for number with not representable nagation.
      return "-8E";
    }

    const char* neg_str = GetNegStr(&num_bytes);

    // Special case for bytes.
    if (num_bytes < 1024LL) {
      // No fractions for bytes.
      return absl::StrFormat("%s%dB", neg_str, num_bytes);
    }

    static const char units[] = "KMGTPE";  // int64 only goes up to E.
    const char* unit = units;
    while (num_bytes >= (1024LL) * (1024LL)) {
      num_bytes /= (1024LL);
      ++unit;
      assert(unit < units + sizeof(units));
    }

    if (*unit == 'K') {
      return absl::StrFormat("%s%.1f%c", neg_str, num_bytes / 1024.0, *unit);
    }
    return absl::StrFormat("%s%.2f%c", neg_str, num_bytes / 1024.0, *unit);
  }

 private:
  template <typename T>
  static const char* GetNegStr(T* value) {
    if (*value < 0) {
      *value = -(*value);
      return "-";
    } else {
      return "";
    }
  }
};

}  // namespace port
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_HUMAN_READABLE_H_
