/* Copyright 2024 The OpenXLA Authors.

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

#include <linux/limits.h>
#include <unistd.h>

#include <string>

#include "absl/log/log.h"
#include "xla/pjrt/plugin/dynamic_registration.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

static constexpr char kMyPluginName[] = "myplugin";

[[maybe_unused]] auto setup_test_plugin = []() -> bool {
  std::string library_path = tsl::testing::XlaSrcRoot();
  library_path = tsl::io::JoinPath(
      library_path, "pjrt/plugin/example_plugin/pjrt_c_api_myplugin_plugin.so");

  if (tsl::setenv("MYPLUGIN_DYNAMIC_PATH", library_path.c_str(), 1) != 0) {
    LOG(ERROR) << "Failed to set MYPLUGIN_DYNAMIC_PATH environment variable.";
    return false;
  }
  REGISTER_DYNAMIC_PJRT_PLUGIN(kMyPluginName, "MYPLUGIN_DYNAMIC_PATH")
  return true;
}();
