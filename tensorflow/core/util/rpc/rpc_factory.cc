/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/strings/numbers.h"

#include "tensorflow/core/util/rpc/rpc_factory.h"

namespace tensorflow {

template <>
bool GetEnvVar(const char* key, const string& default_value, string* value) {
  const char* env_value = std::getenv(key);
  if (!env_value || env_value[0] == '\0') {
    *value = default_value;
  } else {
    *value = env_value;
  }
  return true;
}

template <>
bool GetEnvVar(const char* key, const int64& default_value, int64* value) {
  const char* env_value = std::getenv(key);
  if (!env_value || env_value[0] == '\0') {
    *value = default_value;
    return true;
  }
  return strings::safe_strto64(env_value, value);
}

template <>
bool GetEnvVar(const char* key, const uint64& default_value, uint64* value) {
  const char* env_value = std::getenv(key);
  if (!env_value || env_value[0] == '\0') {
    *value = default_value;
    return true;
  }
  return strings::safe_strtou64(env_value, value);
}

}  // namespace tensorflow
