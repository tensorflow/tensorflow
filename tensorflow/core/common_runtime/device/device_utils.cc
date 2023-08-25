/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/device/device_utils.h"

#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {
namespace device_utils {

Status ValidateDeviceType(StringPiece type) {
  static const LazyRE2 kTfDeviceTypeRegEx = {"[A-Z][A-Z_]*"};
  bool matches = RE2::FullMatch(type, *kTfDeviceTypeRegEx);
  if (!matches) {
    return Status(absl::StatusCode::kFailedPrecondition,
                  strings::StrCat("Device name/type '", type, "' must match ",
                                  kTfDeviceTypeRegEx->pattern(), "."));
  }
  return OkStatus();
}

}  // namespace device_utils
}  // namespace tensorflow
