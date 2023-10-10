/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/config/flags.h"

#include "absl/strings/ascii.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace config {

Flag::Flag(StringPiece flag, bool default_value) {
  bool val = default_value;
  if (ReadBoolFromEnvVar(absl::AsciiStrToUpper(flag), default_value, &val)
          .ok()) {
    value_ = val;
    return;
  }
  value_ = default_value;
}

}  // namespace config
}  // namespace tensorflow
