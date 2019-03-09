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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPNN_POOLING_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPNN_POOLING_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

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
// Allocating functions.
StatusOr<poplar::Tensor> AllocatePoolingOp(
    poplar::Graph&, CompilerResources&, const std::string&, const TensorTarget&,
    const IPUCustomKernelsUtil::AttributeMap&, const TensorMap&);

// Creating functions.
StatusOr<poplar::program::Program> CreateMaxPoolOp(
    poplar::Graph&, CompilerResources&, const HloInstruction*,
    const xla::Shape&, TensorMap&, const IPUCustomKernelsUtil::AttributeMap&);

StatusOr<poplar::program::Program> CreateAvgPoolOp(
    poplar::Graph&, CompilerResources&, const HloInstruction*,
    const xla::Shape&, TensorMap&, const IPUCustomKernelsUtil::AttributeMap&);

StatusOr<poplar::program::Program> CreateMaxPoolGradOp(
    poplar::Graph&, CompilerResources&, const HloInstruction*,
    const xla::Shape&, TensorMap&, const IPUCustomKernelsUtil::AttributeMap&);

StatusOr<poplar::program::Program> CreateAvgPoolGradOp(
    poplar::Graph&, CompilerResources&, const HloInstruction*,
    const xla::Shape&, TensorMap&, const IPUCustomKernelsUtil::AttributeMap&);
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPNN_POOLING_