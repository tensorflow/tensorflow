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

#include "xla/pjrt/plugin/static_registration.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_api.h"

bool RegisterStaticPjrtPlugin(absl::string_view plugin_name,
                              const PJRT_Api* plugin_api) {
  auto status = pjrt::SetPjrtApi(plugin_name, plugin_api);
  QCHECK(status.ok()) << "Failed to register PJRT plugin " << plugin_name
                      << ": " << status;
  return true;
}
