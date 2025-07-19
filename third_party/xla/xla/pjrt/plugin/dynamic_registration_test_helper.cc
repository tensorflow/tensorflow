/* Copyright 2025 The OpenXLA Authors.

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

#include <string>

#include "xla/pjrt/plugin/dynamic_registration.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

static constexpr char kMyPluginName[] = "myplugin";
static constexpr char kMyPluginLibraryEnvName[] = "MYPLUGIN_DYNAMIC_PATH";

[[maybe_unused]] bool set_up_test_env = []() -> bool {
  std::string library_path = tsl::testing::XlaSrcRoot();
  library_path = tsl::io::JoinPath(
      library_path, "pjrt/plugin/example_plugin/pjrt_c_api_myplugin_plugin.so");

  if (tsl::setenv(kMyPluginLibraryEnvName, library_path.c_str(), 1) != 0) {
    return false;
  }
  // This registration is not how a normal dynamic registration would look,
  // this is only necessary for test code to set up the environment ahead.
  // Usually the environment variable is set by the OS, but we need to set it
  // in process here to test the dynamic registration path.
  REGISTER_DYNAMIC_PJRT_PLUGIN(kMyPluginName, kMyPluginLibraryEnvName);
  return true;
}();
