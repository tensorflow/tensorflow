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

#ifndef TENSORFLOW_CORE_PLATFORM_RAW_CODING_H_
#define TENSORFLOW_CORE_PLATFORM_RAW_CODING_H_

#include <string.h>

#include "tensorflow/tsl/platform/raw_coding.h"

namespace tensorflow {
namespace core {
// NOLINTBEGIN(misc-unused-using-decls)
using ::tsl::core::DecodeFixed16;
using ::tsl::core::DecodeFixed32;
using ::tsl::core::DecodeFixed64;
// NOLINTEND(misc-unused-using-decls)
}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_RAW_CODING_H_
