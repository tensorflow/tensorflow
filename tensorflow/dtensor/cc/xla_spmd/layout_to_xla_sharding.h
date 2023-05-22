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

#ifndef TENSORFLOW_DTENSOR_CC_XLA_SPMD_LAYOUT_TO_XLA_SHARDING_H_
#define TENSORFLOW_DTENSOR_CC_XLA_SPMD_LAYOUT_TO_XLA_SHARDING_H_

#include <cstdint>
#include <vector>

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {

// Mhlo sharding string attribute, used for setting hlo sharding on ops, inputs,
// and outputs of a function for XLA SPMD.
constexpr char kXlaShardingAttr[] = "mhlo.sharding";

// Represents a permutation of DTensor Layout mesh dimensions from major to
// minor.
//
// Sizes of `permutation` and `sizes` must be equal.
struct MeshMajorToMinor {
  // A permutation of range [0...n].
  std::vector<int64_t> permutation;
  // The size of mesh dimensions before the permutation.
  std::vector<int64_t> sizes;

  // Produces a flat list of device ids according to the permutation.
  std::vector<int64_t> ToDeviceList();
};

// Get the mesh dimensions from major to minor.
StatusOr<MeshMajorToMinor> ConvertMeshMajorToMinor(const Layout& layout,
                                                   const Mesh& mesh);

// Returns an ::xla::OpSharding protobuf from `layout`.
StatusOr<::xla::OpSharding> ConvertLayoutToXlaOpSharding(const Layout& layout);

}  // namespace dtensor
}  // namespace tensorflow
#endif  // TENSORFLOW_DTENSOR_CC_XLA_SPMD_LAYOUT_TO_XLA_SHARDING_H_
