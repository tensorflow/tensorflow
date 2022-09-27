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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GLOBAL_DEVICE_ID_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GLOBAL_DEVICE_ID_H_

#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/lib/gtl/int_type.h"

namespace xla {

// Strongly-typed integer type for naming a device globally within a distributed
// system. XLA doesn't have a strong opinion about what global numbering scheme
// is applied to GPUs; the user must provide a local -> global mapping via
// GpuExecutableRunOptions for the local GPUs.
TSL_LIB_GTL_DEFINE_INT_TYPE(GlobalDeviceId, int64_t);

// Returns a comma-separated string of global device IDs.
std::string GlobalDeviceIdsToString(absl::Span<GlobalDeviceId const> ids);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GLOBAL_DEVICE_ID_H_
