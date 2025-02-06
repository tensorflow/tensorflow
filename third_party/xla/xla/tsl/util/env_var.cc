/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/util/env_var.h"

#include <stdlib.h>

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/str_util.h"
#include "tsl/platform/strcat.h"

namespace tsl {

absl::Status ReadBoolFromEnvVar(absl::string_view env_var_name,
                                bool default_val, bool* value) {
  *value = default_val;
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return absl::OkStatus();
  }
  string str_value = absl::AsciiStrToLower(tf_env_var_val);
  if (str_value == "0" || str_value == "false") {
    *value = false;
    return absl::OkStatus();
  } else if (str_value == "1" || str_value == "true") {
    *value = true;
    return absl::OkStatus();
  }
  return errors::InvalidArgument(strings::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into bool: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

absl::Status ReadInt64FromEnvVar(absl::string_view env_var_name,
                                 int64_t default_val, int64_t* value) {
  *value = default_val;
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return absl::OkStatus();
  }
  if (absl::SimpleAtoi(tf_env_var_val, value)) {
    return absl::OkStatus();
  }
  return errors::InvalidArgument(strings::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into int64: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

absl::Status ReadFloatFromEnvVar(absl::string_view env_var_name,
                                 float default_val, float* value) {
  *value = default_val;
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return absl::OkStatus();
  }
  if (absl::SimpleAtof(tf_env_var_val, value)) {
    return absl::OkStatus();
  }
  return errors::InvalidArgument(strings::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into float: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

absl::Status ReadStringFromEnvVar(absl::string_view env_var_name,
                                  absl::string_view default_val,
                                  string* value) {
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val != nullptr) {
    *value = tf_env_var_val;
  } else {
    *value = string(default_val);
  }
  return absl::OkStatus();
}

absl::Status ReadStringsFromEnvVar(absl::string_view env_var_name,
                                   absl::string_view default_val,
                                   std::vector<string>* value) {
  string str_val;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar(env_var_name, default_val, &str_val));
  *value = str_util::Split(str_val, ',');
  return absl::OkStatus();
}

}  // namespace tsl
