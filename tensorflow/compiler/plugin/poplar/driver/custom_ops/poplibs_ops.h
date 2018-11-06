/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CUSTOM_OPS_POPLIBS_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CUSTOM_OPS_POPLIBS_OPS_H_

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"

#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#include "absl/container/flat_hash_map.h"

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace xla {
class HloInstruction;

namespace poplarplugin {

typedef StatusOr<poplar::Tensor> (*CustomPoplibOpAllocator)(
    poplar::Graph&, CompilerResources&, const std::string&,
    const HloInstruction*, const int64,
    const IPUCustomKernelsUtil::AttributeMap&);

typedef StatusOr<poplar::program::Program> (*CustomPoplibOpCreator)(
    poplar::Graph&, CompilerResources&, const HloInstruction*,
    const xla::Shape&, TensorMap&, const IPUCustomKernelsUtil::AttributeMap&);

using CustomPoplibOpInfo =
    std::pair<CustomPoplibOpAllocator, CustomPoplibOpCreator>;
// Call map functions
const absl::flat_hash_map<std::string, CustomPoplibOpInfo>& GetPopnnOpInfoMap();
const absl::flat_hash_map<std::string, CustomPoplibOpInfo>&
GetPoplinOpInfoMap();
const absl::flat_hash_map<std::string, CustomPoplibOpInfo>&
GetPoprandOpInfoMap();

// Popnn Ops
StatusOr<poplar::Tensor> AllocateLstmLayerFwdOp(
    poplar::Graph&, CompilerResources&, const std::string&,
    const HloInstruction*, const int64,
    const IPUCustomKernelsUtil::AttributeMap&);
StatusOr<poplar::Tensor> AllocateLstmLayerBwdOp(
    poplar::Graph&, CompilerResources&, const std::string&,
    const HloInstruction*, const int64,
    const IPUCustomKernelsUtil::AttributeMap&);
StatusOr<poplar::program::Program> CreateLstmLayerFwdOp(
    poplar::Graph&, CompilerResources&, const HloInstruction*,
    const xla::Shape&, TensorMap&, const IPUCustomKernelsUtil::AttributeMap&);
StatusOr<poplar::program::Program> CreateLstmLayerBwdOp(
    poplar::Graph&, CompilerResources&, const HloInstruction*,
    const xla::Shape&, TensorMap&, const IPUCustomKernelsUtil::AttributeMap&);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CUSTOM_OPS_POPLIBS_OPS_H_