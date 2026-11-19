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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_CUSTOM_OPTIONS_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_CUSTOM_OPTIONS_H_

#include "absl/strings/string_view.h"

namespace xla {
namespace ifrt {

// Boolean flag requesting arena allocation for output buffers in PjRt-IFRT.
inline constexpr absl::string_view kPjRtCustomOptionUseOutputArena =
    "use_output_arena";

// String holding the call-site location / traceback for PjRt-IFRT.
inline constexpr absl::string_view kPjRtCustomOptionCallLocation =
    "call_location";

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_CUSTOM_OPTIONS_H_
