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
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

// Adds to builder an XLA computation that performs a gather on input (of
// shape input_shape) keyed on indices (of shape indices_shape).
//
// index_type must be must be DT_INT32 or DT_INT64.
// If `indices_are_nd` is true, the last dimension of `indices` are treated as
// a multidimensional index values. Otherwise, `indices` is treated as a tensor
// of scalar indices.
Status XlaGather(const xla::XlaOp& input, const TensorShape& input_shape,
                 const xla::XlaOp& indices, const TensorShape& indices_shape,
                 int64_t axis, bool indices_are_nd, DataType dtype,
                 DataType index_type, xla::XlaBuilder* builder,
                 xla::XlaOp* gather_output);

// The implementation of Gather and ResourceGather through XLA. Uses `input` as
// the input instead of context->input(0) in order to allow ResourceGather to
// handle obtaining the data from the ResourceVariable.
Status XlaGatherWithBatchDimsOpImpl(XlaOpKernelContext* context,
                                    xla::XlaOp input,
                                    const TensorShape& input_shape,
                                    int batch_dims, xla::XlaOp* gather_output);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_GATHER_OP_HELPERS_H_
