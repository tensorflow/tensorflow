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

#ifndef TENSORFLOW_CORE_PLATFORM_BASE64_H_
#define TENSORFLOW_CORE_PLATFORM_BASE64_H_

#include <string>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tsl/platform/base64.h"

namespace tensorflow {

using tsl::Base64Decode;  // NOLINT
using tsl::Base64Encode;  // NOLINT

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_BASE64_H_
