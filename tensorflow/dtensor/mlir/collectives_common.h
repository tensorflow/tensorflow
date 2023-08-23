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

#ifndef TENSORFLOW_DTENSOR_MLIR_COLLECTIVES_COMMON_H_
#define TENSORFLOW_DTENSOR_MLIR_COLLECTIVES_COMMON_H_

#include <map>
#include <string>
#include <vector>

#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {

// Computes AllReduce partitions using reduced mesh dimension names.
StatusOr<std::map<DeviceLocation, std::vector<int32>>>
GetAllReducePartitionsFromReducedDims(
    const dtensor::Layout& output_layout,
    const absl::flat_hash_set<std::string>& reduced_dims);

// Use the first device in the mesh to extract the device name.
StatusOr<std::string> DeviceTypeFromMesh(const Mesh& mesh);

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_COLLECTIVES_COMMON_H_
