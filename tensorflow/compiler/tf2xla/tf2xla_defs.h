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

// Marks a node for XLA compilation. The attribute value indicates the
// compilation device type.
inline constexpr absl::string_view kCompileDeviceTypeAttr =
    "_xla_compile_device_type";
// Marks a node for replication. The attribute value indicates the replication
// metadata op.
inline constexpr absl::string_view kReplicationInfoAttr = "_replication_info";
// Marks a node for XLA-TPU compilation. The attribute value indicates the
// associated compilation cluster and replication metadata op.
inline constexpr absl::string_view kTpuReplicateAttr = "_tpu_replicate";
// Marks a node inside of an XLA compilation cluster to be placed outside of the
// cluster.
inline constexpr absl::string_view kXlaOutsideCompilationAttr =
    "_xla_outside_compilation";
// Frontend attributes ID.
inline constexpr absl::string_view kXlaFrontendAttributesAttrName =
    "_XlaFrontendAttributes";
// Device types.
inline constexpr absl::string_view kCpuDevice = "CPU";
inline constexpr absl::string_view kGpuDevice = "GPU";
inline constexpr absl::string_view kTpuDevice = "TPU";
inline constexpr std::array<absl::string_view, 3> kValidDeviceTypes = {
    kCpuDevice, kGpuDevice, kTpuDevice};
// Attributes that need to be propagated during rewrites (e.g., in
// functionalization).
inline constexpr std::array<absl::string_view, 5> kAttrsToPropagate = {
    kCompileDeviceTypeAttr,
    kReplicationInfoAttr,
    kXlaFrontendAttributesAttrName,
    kXlaOutsideCompilationAttr,
    kTpuReplicateAttr,
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_TF2XLA_DEFS_H_
