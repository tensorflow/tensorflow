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

#include "xla/python/ifrt/ir/utils.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace ifrt {

absl::StatusOr<int64_t> GetDeviceMemoryInBytes(absl::string_view device_kind) {
  constexpr int64_t kGB = 1024 * 1024 * 1024;
  if (device_kind == "TPU v2") {
    return 8LL * kGB;
  }
  if (device_kind == "TPU v3") {
    return 32LL * kGB;
  }
  if (device_kind == "TPU v4") {
    return 32LL * kGB;
  }
  if (device_kind == "TPU v4 lite") {
    return 8LL * kGB;
  }
  if (device_kind == "TPU v5" || device_kind == "TPU v5p") {
    return 95LL * kGB;
  }
  if (device_kind == "TPU v5 lite" || device_kind == "TPU v5e") {
    return 16LL * kGB;
  }
  if (device_kind == "TPU v6 lite") {
    return 32LL * kGB;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "`GetDeviceMemoryInBytes` is not supported for device kind: ",
      device_kind));
}

}  // namespace ifrt
}  // namespace xla
