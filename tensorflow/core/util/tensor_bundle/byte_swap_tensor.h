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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap_array.h"

namespace tensorflow {

// Byte-swap a tensor's backing buffer in place.
//
// Args:
//  t: Tensor to be modified IN PLACE. Any tensors that share a backing
//     buffer with this one will also end up byte-swapped.
// Returns: OkStatus() on success, -1 otherwise
// TODO(frreiss): Should this be a member of the Tensor class?
Status ByteSwapTensor(Tensor *t);

// Swap tensor_content field of Const Op Tensors in the named functions
Status ByteSwapTensorContent(MetaGraphDef *meta_graph_def);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_BYTE_SWAP_TENSOR_H_
