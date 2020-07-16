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

#include "tensorflow/core/util/env_var.h"

#include <stdlib.h>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

Status ReadBoolFromEnvVar(StringPiece env_var_name, bool default_val,
                          bool* value) {
  *value = default_val;
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return Status::OK();
  }
  string str_value = absl::AsciiStrToLower(tf_env_var_val);
  if (str_value == "0" || str_value == "false") {
    *value = false;
    return Status::OK();
  } else if (str_value == "1" || str_value == "true") {
    *value = true;
    return Status::OK();
  }
  return errors::InvalidArgument(strings::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into bool: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

Status ReadInt64FromEnvVar(StringPiece env_var_name, int64 default_val,
                           int64* value) {
  *value = default_val;
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return Status::OK();
  }
  if (strings::safe_strto64(tf_env_var_val, value)) {
    return Status::OK();
  }
  return errors::InvalidArgument(strings::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into int64: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

Status ReadFloatFromEnvVar(StringPiece env_var_name, float default_val,
                           float* value) {
  *value = default_val;
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return Status::OK();
  }
  if (strings::safe_strtof(tf_env_var_val, value)) {
    return Status::OK();
  }
  return errors::InvalidArgument(strings::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into float: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

Status ReadStringFromEnvVar(StringPiece env_var_name, StringPiece default_val,
                            string* value) {
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val != nullptr) {
    *value = tf_env_var_val;
  } else {
    *value = string(default_val);
  }
  return Status::OK();
}

}  // namespace tensorflow
