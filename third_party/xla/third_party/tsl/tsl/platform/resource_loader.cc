/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/resource_loader.h"

#include <cstdlib>
#include <string>

#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/test.h"

namespace tsl {

std::string GetDataDependencyFilepath(const std::string& relative_path) {
  // TODO(ddunleavy): replace this with `TensorFlowSrcRoot()` from `test.h`.
  const char* srcdir = std::getenv("TEST_SRCDIR");
  if (!srcdir) {
    LOG(FATAL) << "Environment variable TEST_SRCDIR unset!";  // Crash OK
  }

  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (!workspace) {
    LOG(FATAL) << "Environment variable TEST_WORKSPACE unset!";  // Crash OK
  }

  return testing::kIsOpenSource
             ? io::JoinPath(srcdir, workspace, relative_path)
             : io::JoinPath(srcdir, workspace, "third_party", relative_path);
}

}  // namespace tsl
