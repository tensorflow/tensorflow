/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/pjrt_ifrt/xla_client.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/rtti.h"

namespace xla {
namespace ifrt {

char XlaLoadOptions::ID = 0;

absl::StatusOr<std::unique_ptr<XlaLoadOptions>> GetXlaLoadOptions(
    std::unique_ptr<LoadOptions> options) {
  if (auto xla_options =
          dyn_cast_if_present<XlaLoadOptions>(std::move(options));
      xla_options != nullptr) {
    return xla_options;
  }
  return absl::InvalidArgumentError("options must be XlaLoadOptions");
}

}  // namespace ifrt
}  // namespace xla
