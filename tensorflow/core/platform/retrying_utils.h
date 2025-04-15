/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_RETRYING_UTILS_H_
#define TENSORFLOW_CORE_PLATFORM_RETRYING_UTILS_H_

#include <functional>

#include "tensorflow/core/platform/status.h"
#include "tsl/platform/retrying_utils.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::RetryConfig;
using tsl::RetryingUtils;
// NOLINTEND(misc-unused-using-decls)
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_RETRYING_UTILS_H_
