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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_TENSOR_LIST_UTILS_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_TENSOR_LIST_UTILS_H_

// TensorList utilities.
//
// Tensor lists are represented as tuple consisting of a pre-allocated buffer
// consisting of the tensors (and where dim 0 is the list index), along with a
// scalar telling us the next index to push a value at.

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

// Whether the input expression at `index` corresponds to a TensorList.
bool IsTensorListInput(XlaOpKernelContext* ctx, int index);

// Builds a TensorList from its constituents, `buffer` and `push_index`.
Status BuildTensorList(const xla::XlaOp& buffer, const xla::XlaOp& push_index,
                       xla::XlaOp* output_list);

// Returns the buffer for the TensorList.
Status GetTensorListBuffer(const xla::XlaOp& op, xla::XlaOp* buffer);

// Returns the push_index for the TensorList.
Status GetTensorListPushIndex(const xla::XlaOp& op, xla::XlaOp* push_index);

// Returns the shape of the TensorList buffer.
Status GetTensorListBufferShape(const xla::XlaOp& op,
                                TensorShape* buffer_shape);

// Inputs the TensorList shape and returns the buffer shape.
Status GetTensorListBufferShape(const xla::Shape& list_shape,
                                TensorShape* buffer_shape);

// Returns whether the TensorList has been initialized.
//
// A TensorList is considered initialized if its element_shape is completely
// known.
Status IsTensorListInitialized(const xla::XlaOp& op, bool* is_initialized);

// Inputs an uninitialized list and a buffer_shape and returns an initialized
// list. The initialized list uses the dtype and push index of the uninitialized
// list and is filled with zeros.
Status InitializeTensorList(const xla::XlaOp& uninitialized_list,
                            const TensorShape& buffer_shape,
                            xla::XlaOp* output_list);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_TENSOR_LIST_UTILS_H_
