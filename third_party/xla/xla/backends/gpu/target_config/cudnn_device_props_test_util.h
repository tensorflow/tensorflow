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

#ifndef XLA_BACKENDS_GPU_TARGET_CONFIG_CUDNN_DEVICE_PROPS_TEST_UTIL_H_
#define XLA_BACKENDS_GPU_TARGET_CONFIG_CUDNN_DEVICE_PROPS_TEST_UTIL_H_

#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Verifies that the cuDNN DeviceProperties synthesized from `desc` matches
// the live DeviceProperties queried from `device_id` (skipping the test if the
// cuDNN runtime is older than 9.8).
void ExpectDevicePropsMatchLiveDevice(
    const stream_executor::DeviceDescription& desc, int device_id = 0);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TARGET_CONFIG_CUDNN_DEVICE_PROPS_TEST_UTIL_H_
