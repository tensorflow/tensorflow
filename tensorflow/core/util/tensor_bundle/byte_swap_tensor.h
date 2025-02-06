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

#ifndef TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_BYTE_SWAP_TENSOR_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_BYTE_SWAP_TENSOR_H_

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap_array.h"

namespace tensorflow {

// Check if a data type is byte swappable.
bool IsByteSwappable(DataType dtype);

// Byte-swap a tensor's backing buffer in place.
//
// Args:
//  t: Tensor to be modified IN PLACE. Any tensors that share a backing
//     buffer with this one will also end up byte-swapped.
// Returns: OkStatus() on success, -1 otherwise
// TODO(frreiss): Should this be a member of the Tensor class?
absl::Status ByteSwapTensor(Tensor* t);

// Byte-swap a tensor proto's backing buffer in place.
//
// Args:
//  t: TensorProto to be modified IN PLACE.
// Returns: OkStatus() on success, -1 otherwise
absl::Status ByteSwapTensorProto(TensorProto* tp);

// Swap tensor_content field of Const Op Tensors in the named functions
// in NodeDef
absl::Status ByteSwapTensorContentInNode(NodeDef& node);

// Swap tensor_content field of Const Op Tensors in the named functions
// in MetaGraphDef
absl::Status ByteSwapTensorContentInMetaGraphDef(MetaGraphDef* meta_graph_def);

// Swap tensor_content field of Const Op Tensors in the named functions
// in GraphDef
absl::Status ByteSwapTensorContentInGraphDef(GraphDef* graph_def);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_BYTE_SWAP_TENSOR_H_
