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

#include "xla/pjrt/plugin/dynamic_registration.h"

#include <cstdlib>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_api.h"

absl::Status RegisterDynamicPjrtPlugin(absl::string_view plugin_name,
                                       absl::string_view library_env_name) {
  std::string library_env_name_str(library_env_name);
  char* library_path = std::getenv(library_env_name_str.c_str());
  if (library_path == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("Environment variable ", library_env_name,
                     " is not set. Can't load PJRT plugin ", plugin_name, "."));
  }
  std::string library_path_str(library_path);
  auto status = pjrt::LoadPjrtPlugin(plugin_name, library_path);
  return status.status();
}
