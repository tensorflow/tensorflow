/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/tsl/profiler/convert/xla_op_utils.h"

#include "absl/strings/string_view.h"

namespace tsl {
namespace profiler {

// LINT.IfChange
constexpr absl::string_view kHloSparseCoreV0Infeed = "sparsecorev0 infeed";
constexpr absl::string_view kHloSparseCoreV0Outfeed = "sparsecorev0 outfeed";
constexpr absl::string_view kHloSparseCoreV0InfeedWait =
    "sparsecorev0 infeed wait";
constexpr absl::string_view kHloSparseCoreV0InfeedTransform =
    "sparsecorev0 infeed transform";
// LINT.ThenChange(//tensorflow/compiler/xla/tsl/profiler/convert/google/xla_op_utils.cc)

}  // namespace profiler
}  // namespace tsl
