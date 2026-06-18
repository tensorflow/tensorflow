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

#include "xla/tsl/platform/resource_loader.h"

#include <cstdlib>
#include <string>

#include "xla/tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace tsl {

std::string GetDataDependencyFilepath(const std::string& relative_path) {
  using bazel::tools::cpp::runfiles::Runfiles;
  std::string error;
  std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));
  if (runfiles == nullptr) {
    LOG(FATAL) << "Could not initialize runfiles: " << error.c_str();
  }

  std::string actual_relative_path;
  if (relative_path.find("external/", 0) == 0) {
    // This is a path from an external repo, remove "external/" for Rlocation
    actual_relative_path = relative_path.substr(strlen("external/"));
  } else {
    // This is a path from the main repo, preappend TEST_WORKSPACE for Rlocation
    const char* workspace = std::getenv("TEST_WORKSPACE");
    if (!workspace) {
      LOG(FATAL) << "Environment variable TEST_WORKSPACE unset!";  // Crash OK
    }
    actual_relative_path = io::JoinPath(workspace, relative_path);
  }

  std::string full_path = runfiles->Rlocation(actual_relative_path);
  if (full_path.empty()) {
    LOG(FATAL) << "Could not find runfile " << actual_relative_path;
  }

  return full_path;
}

}  // namespace tsl
