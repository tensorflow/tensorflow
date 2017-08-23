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

// Helper methods for XLA Gather Ops.

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_GATHER_OP_HELPERS_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_GATHER_OP_HELPERS_H_

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

// Adds to builder an XLA computation that performs a gather on input (of
// shape input_shape) keyed on indices (of shape indices_shape).
xla::ComputationDataHandle XlaComputeGatherDynamicSlice(
    XlaOpKernelContext* ctx, const xla::ComputationDataHandle& input,
    const TensorShape& input_shape, const xla::ComputationDataHandle& indices,
    const TensorShape& indices_shape, DataType dtype,
    xla::ComputationBuilder* builder);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_GATHER_OP_HELPERS_H_
