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
#include "xla/pjrt/plugin/test/plugin_test_fixture.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<std::string> GetRegisteredPluginName() {
  TF_ASSIGN_OR_RETURN(std::vector<std::string> pjrt_apis,
                      pjrt::GetRegisteredPjrtApis());
  if (pjrt_apis.size() != 1) {
    return absl::InvalidArgumentError(
        "Expected exactly one plugin to be registered.");
  }
  return pjrt_apis[0];
}

}  // namespace xla
