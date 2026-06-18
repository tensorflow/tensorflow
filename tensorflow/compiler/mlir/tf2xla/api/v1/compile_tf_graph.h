/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_COMPILE_TF_GRAPH_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_COMPILE_TF_GRAPH_H_

#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/client/compile_only_client.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/shape.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tpu/kernels/tpu_compile.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {

// Compiles the given Tensorflow graph into xla::HLO. The result is in
// compilation_result. If the input computation is in MLIR, it will be
// converted to a Tensorflow graph. Otherwise, the graph compiler will be run.
absl::Status CompileTensorflowGraphToHlo(
    const std::variant<tpu::MlirToHloArgs, tpu::FunctionToHloArgs>& computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_funcs,
    const std::vector<tensorflow::TensorShape>& arg_shapes,
    tsl::DeviceType device_type,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    xla::CompileOnlyClient* client,
    XlaCompiler::CompilationResult* compilation_result);

}  // namespace v1
}  // namespace tf2xla
};  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_COMPILE_TF_GRAPH_H_
