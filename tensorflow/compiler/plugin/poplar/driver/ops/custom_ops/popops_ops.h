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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPOPS_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPOPS_OPS_H_

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/stream_executor/lib/statusor.h"

#include "absl/container/flat_hash_map.h"

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

#include <string>

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace xla {
class HloInstruction;

namespace poplarplugin {

const absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo>& GetPopopsOpInfoMap();

// Allocating functions.
StatusOr<poplar::Tensor> AllocateUnaryOp(
    poplar::Graph&, CompilerResources&, const std::string&, const TensorTarget&,
    const IPUCustomKernelsUtil::AttributeMap&, const TensorMap&);

// Creating functions.
StatusOr<poplar::program::Program> CreateUnaryOp(
    poplar::Graph&, CompilerResources&, const HloInstruction*,
    const xla::Shape&, TensorMap&, const IPUCustomKernelsUtil::AttributeMap&);
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPOPS_OPS_H_
