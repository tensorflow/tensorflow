/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_TF2XLA_DEFS_H_
#define TENSORFLOW_COMPILER_TF2XLA_TF2XLA_DEFS_H_

#include <array>

#include "absl/strings/string_view.h"

namespace tensorflow {

// TODO(b/228344955) use inline constexpr with C++17

// Marks a node for XLA compilation. The attribute value indicates the
// compilation device type.
extern const absl::string_view kCompileDeviceTypeAttr;
// Marks a node for replication. The attribute value indicates the replication
// metadata op.
extern const absl::string_view kReplicationInfoAttr;
// Marks a node for XLA-TPU compilation. The attribute value indicates the
// associated compilation cluster and replication metadata op.
extern const absl::string_view kTpuReplicateAttr;
// Marks a node inside of an XLA compilation cluster to be placed outside of the
// cluster.
extern const absl::string_view kXlaOutsideCompilationAttr;
// Frontend attributes ID.
extern const absl::string_view kXlaFrontendAttributesAttrName;
// Device types.
extern const absl::string_view kCpuDevice;
extern const absl::string_view kGpuDevice;
extern const absl::string_view kTpuDevice;
extern const std::array<absl::string_view, 3> kValidDeviceTypes;
// Attributes that need to be propagated during rewrites (e.g., in
// functionalization).
extern const std::array<absl::string_view, 5> kAttrsToPropagate;

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_TF2XLA_DEFS_H_
