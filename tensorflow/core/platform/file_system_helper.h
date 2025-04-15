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

#ifndef TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_HELPER_H_
#define TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_HELPER_H_

#include <string>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tsl/platform/file_system_helper.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::Env;
using tsl::FileSystem;

namespace internal {
using tsl::internal::FileExists;
using tsl::internal::GetMatchingPaths;
}  // namespace internal
// NOLINTEND(misc-unused-using-decls)
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_HELPER_H_
