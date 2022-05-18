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

#include "tensorflow/compiler/tf2xla/tf2xla_defs.h"

namespace tensorflow {

// TODO(b/228344955) use inline constexpr with C++17
const absl::string_view kCompileDeviceTypeAttr = "_xla_compile_device_type";
const absl::string_view kReplicationInfoAttr = "_replication_info";
const absl::string_view kTpuReplicateAttr = "_tpu_replicate";
const absl::string_view kXlaOutsideCompilationAttr = "_xla_outside_compilation";
const absl::string_view kXlaFrontendAttributesAttrName =
    "_XlaFrontendAttributes";
const absl::string_view kCpuDevice = "CPU";
const absl::string_view kGpuDevice = "GPU";
const absl::string_view kTpuDevice = "TPU";
const std::array<absl::string_view, 3> kValidDeviceTypes = {
    kCpuDevice, kGpuDevice, kTpuDevice};
// TODO(b/160275126): if possible, avoid hard-coding these attributes here
const std::array<absl::string_view, 5> kAttrsToPropagate = {
    kCompileDeviceTypeAttr,
    kReplicationInfoAttr,
    kXlaFrontendAttributesAttrName,
    kXlaOutsideCompilationAttr,
    kTpuReplicateAttr,
};

}  // namespace tensorflow
