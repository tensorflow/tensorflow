/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

REGISTER_XLA_OP(Name("XlaConv")
                    .CompileTimeConstantInput("window_strides")
                    .CompileTimeConstantInput("lhs_dilation")
                    .CompileTimeConstantInput("rhs_dilation")
                    .CompileTimeConstantInput("feature_group_count")
                    .CompileTimeConstantInput("padding"),
                MlirXlaOpKernel);

REGISTER_XLA_OP(Name("XlaConvV2")
                    .CompileTimeConstantInput("window_strides")
                    .CompileTimeConstantInput("lhs_dilation")
                    .CompileTimeConstantInput("rhs_dilation")
                    .CompileTimeConstantInput("feature_group_count")
                    .CompileTimeConstantInput("padding"),
                MlirXlaOpKernel);

}  // namespace
}  // namespace tensorflow
