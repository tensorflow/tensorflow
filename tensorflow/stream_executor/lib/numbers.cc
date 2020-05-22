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

#include "tensorflow/stream_executor/lib/numbers.h"

#include <stdlib.h>

namespace stream_executor {
namespace port {

bool safe_strto32(const char* str, int32* value) {
  char* endptr;
  *value = strtol(str, &endptr, 10);  // NOLINT
  if (endptr != str) {
    while (isspace(*endptr)) ++endptr;
  }
  return *str != '\0' && *endptr == '\0';
}

// Convert strings to floating point values.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
bool safe_strto32(const std::string& str, int32* value) {
  return port::safe_strto32(str.c_str(), value);
}

}  // namespace port
}  // namespace stream_executor
