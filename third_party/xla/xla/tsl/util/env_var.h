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

#ifndef XLA_TSL_UTIL_ENV_VAR_H_
#define XLA_TSL_UTIL_ENV_VAR_H_

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/stringpiece.h"

namespace tsl {

// Returns a boolean into "value" from the environmental variable
// "env_var_name". If it is unset, the default value is used. A string "0" or a
// case insensitive "false" is interpreted as false. A string "1" or a case
// insensitive "true" is interpreted as true. Otherwise, an error status is
// returned.
absl::Status ReadBoolFromEnvVar(absl::string_view env_var_name,
                                bool default_val, bool* value);

// Returns an int64 into "value" from the environmental variable "env_var_name".
// If it is unset, the default value is used.
// If the string cannot be parsed into int64, an error status is returned.
absl::Status ReadInt64FromEnvVar(absl::string_view env_var_name,
                                 int64_t default_val, int64_t* value);
// Returns a float into "value" from the environmental variable "env_var_name".
// If it is unset, the default value is used.
// If the string cannot be parsed into float, an error status is returned.
absl::Status ReadFloatFromEnvVar(absl::string_view env_var_name,
                                 float default_val, float* value);

// Returns a string into "value" from the environmental variable "env_var_name".
// If it is unset, the default value is used.
absl::Status ReadStringFromEnvVar(absl::string_view env_var_name,
                                  absl::string_view default_val,
                                  std::string* value);

// Returns a comma separated string into "value" from the environmental variable
// "env_var_name". If it is unset, the default value is comma split and used.
absl::Status ReadStringsFromEnvVar(absl::string_view env_var_name,
                                   absl::string_view default_val,
                                   std::vector<std::string>* value);

}  // namespace tsl

#endif  // XLA_TSL_UTIL_ENV_VAR_H_
