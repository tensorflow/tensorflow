/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_VALIDATE_FUNCTION_DEVICES_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_VALIDATE_FUNCTION_DEVICES_H_

#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {

// Validates that device constraints specified on nodes within a FunctionDef
// can be satisfied by the available devices. This is used to ensure that
// tf.device() constraints are validated even for XLA-compiled functions
// (jit_compile=True), where the Placer is bypassed.
//
// Returns OK if all device constraints are satisfiable, or an error status
// describing the first unsatisfiable device constraint.
absl::Status ValidateFunctionDeviceConstraints(
    const FunctionDef& fdef,
    const std::vector<DeviceAttributes>& available_devices);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_VALIDATE_FUNCTION_DEVICES_H_
