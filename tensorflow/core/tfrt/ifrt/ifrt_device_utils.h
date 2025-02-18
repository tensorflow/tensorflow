/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_DEVICE_UTILS_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_DEVICE_UTILS_H_

#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"

namespace tensorflow {
namespace ifrt_serving {

// Returns the assigned IFRT devices based on the device assignment attribute.
//
// params:
//   ifrt_client: The ifrt client.
//   num_replicas: The number of replicas.
//   num_cores_per_replica: The number of cores per replica.
//
//   device_assignment: The device assignment array encoded as
//   [x0,y0,z0,core0,x1,y1,z1,core1, ...]. Optional. If not provided, the
//   devices will be assigned based on the default order returned by the IFRT
//   client.
//
// returns:
//   The assigned devices.
absl::StatusOr<std::vector<xla::ifrt::Device*>> GetAssignedIfrtDevices(
    const xla::ifrt::Client& ifrt_client, int num_replicas,
    int num_cores_per_replica,
    std::optional<std::vector<int>> device_assignment);

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_DEVICE_UTILS_H_
