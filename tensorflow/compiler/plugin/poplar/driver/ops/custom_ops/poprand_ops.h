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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPRAND_OPS_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPRAND_OPS_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace xla {
class HloInstruction;

namespace poplarplugin {
const absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo>& GetPoprandOpInfoMap();
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPRAND_OPS_