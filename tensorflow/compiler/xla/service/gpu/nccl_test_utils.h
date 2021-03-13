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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_TEST_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_TEST_UTILS_H_

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"

namespace xla {
namespace gpu {

// Gets the set of devices that have a NCCL channel open.  This is primarily
// for testing.
//
// (Indeed, because the NCCL channels are a global variable, in the real
// world, the value returned here is stale as soon as you read it, so it's not
// clear how you *could* use it for anything other than tests.)
absl::flat_hash_set<GlobalDeviceId> DevicesWithOpenNcclChannels();

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_TEST_UTILS_H_
