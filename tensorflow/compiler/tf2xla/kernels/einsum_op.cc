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

#include <array>

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

constexpr std::array<DataType, 9> kEinsumTypes = {
    {DT_INT32, DT_INT64, DT_UINT64, DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE,
     DT_COMPLEX64, DT_COMPLEX128}};

REGISTER_XLA_OP(Name("XlaEinsum").TypeConstraint("T", kEinsumTypes),
                MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Einsum").TypeConstraint("T", kEinsumTypes),
                MlirXlaOpKernel);

}  // namespace
}  // namespace tensorflow
