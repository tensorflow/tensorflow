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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_LIBRARY_FUSION_KINDS_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_LIBRARY_FUSION_KINDS_H_

#include "absl/strings/string_view.h"

inline constexpr absl::string_view kOneDnnFusionKind = "__onednn_fusion";
inline constexpr absl::string_view kYnnFusionKind = "__ynn_fusion";

// The maximum number of instructions allowed in a library fusion. This avoids
// crashing libraries with too large graphs. It could break good fusions, e.g.,
// Softmax, etc. We should deploy a smarter logic in the future.
inline constexpr int kMaxFusionSize = 100;

// TODO(intel-tf): Evaluate if there is performance benefit to increase the
// limit.
inline constexpr int kMaxOneDnnFusionSize = 10;

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_LIBRARY_FUSION_KINDS_H_
