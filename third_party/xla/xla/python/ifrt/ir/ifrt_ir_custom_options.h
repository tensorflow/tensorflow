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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_IR_CUSTOM_OPTIONS_H_
#define XLA_PYTHON_IFRT_IR_IFRT_IR_CUSTOM_OPTIONS_H_

#include "absl/strings/string_view.h"

namespace xla {
namespace ifrt {

// Boolean flag controlling IFRT IR status filling. If set to true, all
// executables will have their status filled when `ExecuteOptions::fill_status`
// is set. Otherwise, only leaf executables will have their status filled.
inline constexpr absl::string_view kIfrtIrCustomOptionFillAllStatuses =
    "fill_all_statuses";

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_IFRT_IR_CUSTOM_OPTIONS_H_
