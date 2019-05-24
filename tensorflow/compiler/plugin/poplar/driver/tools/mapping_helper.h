/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MAPPING_HELPER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MAPPING_HELPER_H_

#include "tensorflow/core/platform/default/integral_types.h"

#include "absl/container/flat_hash_map.h"

#include <poplar/Interval.hpp>

using tensorflow::uint32;
using tensorflow::uint64;

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace xla {
namespace poplarplugin {

using LinearMapperState = absl::flat_hash_map<poplar::Graph*, uint64>;
// A helper class for mapping tensors to the IPU which takes previous
// allocations into account.
class MappingHelper {
 public:
  // Maps the tensor linearly, however the starting tile is dependent on
  // previous allocations.
  static void MapTensorLinearly(LinearMapperState& state, poplar::Graph& graph,
                                poplar::Tensor& tensor);
  static void MapTensorLinearly(LinearMapperState& state, poplar::Graph& graph,
                                poplar::Tensor& tensor,
                                uint32 min_elements_per_tile,
                                uint32 grain_size);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MAPPING_HELPER_H_
