/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_PLACEMENT_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_PLACEMENT_UTILS_H_

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {
namespace eager {

bool IsColocationExempt(StringPiece op_name);

bool IsFunction(StringPiece op_name);

bool IsCustomDevice(StringPiece device_name, const EagerContext& ctx);

// TODO(b/154234908): Unify placement logic.
// TODO(b/159647422): Add C++ unit tests for placement logic.

// Pin the op to cpu if all op inputs are on the CPU, small (<64 elements) and
// integers (int32/int64). This can be disabled by setting the environment
// variable "TF_EAGER_ENABLE_SMALL_TENSOR_CPU_PINNING" to "0" or "false".
Status MaybePinSmallOpsToCpu(bool* result, StringPiece op_name,
                             absl::Span<ImmediateExecutionTensorHandle*> args,
                             const EagerContext& ctx);

// If a resource touching input is specified, all resource-touching ops run in
// the device the resource is, regardless of anything else that has been
// specified. This is identical to the graph mode behavior.
Status MaybePinToResourceDevice(VariantDevice* device,
                                const EagerOperation& op);

// If all the inputs are on the same custom device, use that custom
// device. Otherwise, it is an error to have a custom device as an input.
Status MaybePinToCustomDevice(VariantDevice* device, const EagerOperation& op);

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_PLACEMENT_UTILS_H_
