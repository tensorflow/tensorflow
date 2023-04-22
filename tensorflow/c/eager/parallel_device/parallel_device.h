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

#ifndef TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_H_
#define TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_H_

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"

namespace tensorflow {
namespace parallel_device {

// Allocate a parallel device named `device_name` which forwards operations to
// `underlying_devices`, maintaining "parallel tensors" with components placed
// on each underlying device.
//
// For example if `device_name` is
//   "/job:localhost/replica:0/task:0/device:CUSTOM:0"
// and `underlying_devices` is
//   {"/job:localhost/replica:0/task:0/device:GPU:0",
//    "/job:localhost/replica:0/task:0/device:GPU:1"}
// Then executing an operation on CUSTOM:0 will execute it on GPU:0 and GPU:1.
//
// Implicit copies onto `device_name` are allowed, replicating the value once
// per device in `underlying_devices`. Implicit copies off of the device throw
// an error.
//
// All component tensors must have the same dtype. Currently they must also have
// the same shape, although this requirement may be relaxed in the future.
//
// `device_name` must not name an existing physical or custom device (see
// the documentation for TFE_RegisterCustomDevice for more information).
//
// Tensors may be copied on or off the device explicitly using
// TPUReplicatedInput and TPUReplicatedOutput respectively. For example, with
// two component devices, running `x = TPUReplicatedInput(inputs=[a, b])` on the
// parallel device creates a parallel tensor `x` with `a` on the first of
// `underlying_devices` and `b` on the second. Running `a_unpacked, b_unpacked =
// TPUReplicatedOutput(input=x, num_replicas=2)` un-packs the parallel tensor
// into its components.
//
// The filled `device` struct and the allocated `device_info` struct may be
// passed to TFE_RegisterCustomDevice. The `device_name` arguments must match.
void AllocateParallelDevice(const char* device_name,
                            const char* const* underlying_devices,
                            int num_underlying_devices,
                            TFE_CustomDevice* device, void** device_info);

}  // namespace parallel_device
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_H_
