/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/global_device_id.h"

#include "absl/strings/str_join.h"

namespace xla {

std::string GlobalDeviceIdsToString(absl::Span<GlobalDeviceId const> ids) {
  std::vector<int64_t> values;
  values.reserve(ids.size());
  for (GlobalDeviceId id : ids) {
    values.push_back(id.value());
  }
  return absl::StrJoin(values, ",");
}

}  // namespace xla
