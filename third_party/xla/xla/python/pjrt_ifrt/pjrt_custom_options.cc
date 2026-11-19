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

#include "xla/python/pjrt_ifrt/pjrt_custom_options.h"

#include "xla/python/ifrt/custom_options_util.h"

namespace xla {
namespace ifrt {
namespace {

// clang-format off
bool register_pjrt_custom_options = ([] {
  RegisterIfrtInternalCustomOption(kPjRtCustomOptionUseOutputArena);
  RegisterIfrtInternalCustomOption(kPjRtCustomOptionCallLocation);
}(), true);
// clang-format on

}  // namespace
}  // namespace ifrt
}  // namespace xla
