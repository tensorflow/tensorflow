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

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

// Declare MlirXlaOpKernel for TF UniformQuantized ops.
// The lowering passes for these ops are located at:
// tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge

REGISTER_XLA_OP(Name("UniformQuantize")
                    .CompileTimeConstantInput("scales")
                    .CompileTimeConstantInput("zero_points"),
                MlirXlaOpKernel);
REGISTER_XLA_OP(Name("UniformDequantize")
                    .CompileTimeConstantInput("scales")
                    .CompileTimeConstantInput("zero_points"),
                MlirXlaOpKernel);
REGISTER_XLA_OP(Name("UniformRequantize")
                    .CompileTimeConstantInput("input_scales")
                    .CompileTimeConstantInput("input_zero_points")
                    .CompileTimeConstantInput("output_scales")
                    .CompileTimeConstantInput("output_zero_points"),
                MlirXlaOpKernel);
REGISTER_XLA_OP(Name("UniformQuantizedAdd")
                    .CompileTimeConstantInput("lhs_scales")
                    .CompileTimeConstantInput("lhs_zero_points")
                    .CompileTimeConstantInput("rhs_scales")
                    .CompileTimeConstantInput("rhs_zero_points")
                    .CompileTimeConstantInput("output_scales")
                    .CompileTimeConstantInput("output_zero_points"),
                MlirXlaOpKernel);
REGISTER_XLA_OP(Name("UniformQuantizedClipByValue")
                    .CompileTimeConstantInput("scales")
                    .CompileTimeConstantInput("zero_points"),
                MlirXlaOpKernel);
REGISTER_XLA_OP(Name("UniformQuantizedConvolution")
                    .CompileTimeConstantInput("lhs_scales")
                    .CompileTimeConstantInput("lhs_zero_points")
                    .CompileTimeConstantInput("rhs_scales")
                    .CompileTimeConstantInput("rhs_zero_points")
                    .CompileTimeConstantInput("output_scales")
                    .CompileTimeConstantInput("output_zero_points"),
                MlirXlaOpKernel);
REGISTER_XLA_OP(Name("UniformQuantizedDot")
                    .CompileTimeConstantInput("lhs_scales")
                    .CompileTimeConstantInput("lhs_zero_points")
                    .CompileTimeConstantInput("rhs_scales")
                    .CompileTimeConstantInput("rhs_zero_points")
                    .CompileTimeConstantInput("output_scales")
                    .CompileTimeConstantInput("output_zero_points"),
                MlirXlaOpKernel);

}  // namespace
}  // namespace tensorflow
