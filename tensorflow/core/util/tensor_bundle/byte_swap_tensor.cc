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

#include "tensorflow/core/util/tensor_bundle/byte_swap_tensor.h"

#include <cstddef>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap_array.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/mem.h"

namespace tensorflow {

namespace {

// Byte-swap a buffer in place.
//
// Args:
//  buff: pointer to the buffer to be modified IN PLACE.
//  size: size of bytes in this buffer.
//  dtype: type of data in this buffer.
//  num_of_elem: number of data in this buffer, set to -1 if it
//               could not be obtained directly from tensor data.
//               If num_of_elem is -1, this function will calculate
//               the number of data based on size and dtype.
// Returns: OkStatus() on success, -1 otherwise
absl::Status ByteSwapBuffer(char* buff, size_t size, DataType dtype,
                            int num_of_elem) {
  int array_len = num_of_elem;
  size_t bytes_per_elem = 0;

  switch (dtype) {
    // Types that don't need byte-swapping
    case DT_STRING:
    case DT_QINT8:
    case DT_QUINT8:
    case DT_BOOL:
    case DT_UINT8:
    case DT_INT8:
      return absl::OkStatus();

    // 16-bit types
    case DT_BFLOAT16:
    case DT_HALF:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_UINT16:
    case DT_INT16:
      bytes_per_elem = 2;
      array_len = (array_len == -1) ? size / bytes_per_elem : array_len;
      break;

    // 32-bit types
    case DT_FLOAT:
    case DT_INT32:
    case DT_QINT32:
    case DT_UINT32:
      bytes_per_elem = 4;
      array_len = (array_len == -1) ? size / bytes_per_elem : array_len;
      break;

    // 64-bit types
    case DT_INT64:
    case DT_DOUBLE:
    case DT_UINT64:
      bytes_per_elem = 8;
      array_len = (array_len == -1) ? size / bytes_per_elem : array_len;
      break;

    // Complex types need special handling
    case DT_COMPLEX64:
      bytes_per_elem = 4;
      array_len = (array_len == -1) ? size / bytes_per_elem : array_len;
      array_len *= 2;
      break;

    case DT_COMPLEX128:
      bytes_per_elem = 8;
      array_len = (array_len == -1) ? size / bytes_per_elem : array_len;
      array_len *= 2;
      break;

    // Types that ought to be supported in the future
    case DT_RESOURCE:
    case DT_VARIANT:
      return absl::UnimplementedError(absl::StrCat(
          "Byte-swapping not yet implemented for tensors with dtype ", dtype));

    // Byte-swapping shouldn't make sense for other dtypes.
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Byte-swapping not supported for tensors with dtype ", dtype));
  }

  TF_RETURN_IF_ERROR(ByteSwapArray(buff, bytes_per_elem, array_len));
  return absl::OkStatus();
}

}  // namespace

bool IsByteSwappable(DataType dtype) {
  switch (dtype) {
    // 16-bit types
    case DT_BFLOAT16:
    case DT_HALF:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_UINT16:
    case DT_INT16:

    // 32-bit types
    case DT_FLOAT:
    case DT_INT32:
    case DT_QINT32:
    case DT_UINT32:

    // 64-bit types
    case DT_INT64:
    case DT_DOUBLE:
    case DT_UINT64:

    // Complex types
    case DT_COMPLEX64:
    case DT_COMPLEX128:
      return true;

    default:
      return false;
  }
}

absl::Status ByteSwapTensor(Tensor* t) {
  char* buff = const_cast<char*>((t->tensor_data().data()));
  return ByteSwapBuffer(buff, t->tensor_data().size(), t->dtype(),
                        t->NumElements());
}

absl::Status ByteSwapTensorContentInNode(NodeDef& node) {
  if (node.op() == "Const") {
    auto node_iterator = node.mutable_attr()->find("value");
    if (node_iterator != node.mutable_attr()->end()) {
      AttrValue node_value = node_iterator->second;
      if (node_value.has_tensor()) {
        auto tsize = node_value.mutable_tensor()->tensor_content().size();
        auto p_type = node_value.mutable_tensor()->dtype();
        // Swap only when there is something in tensor_content field
        if (tsize != 0 && DataTypeCanUseMemcpy(p_type)) {
          Tensor parsed(p_type);
          DCHECK(parsed.FromProto(*node_value.mutable_tensor()));
          if (!parsed.tensor_data().empty()) {
            TF_RETURN_IF_ERROR(ByteSwapTensor(&parsed));
            (*node.mutable_attr())["value"]
                .mutable_tensor()
                ->set_tensor_content(std::string(
                    reinterpret_cast<const char*>(parsed.tensor_data().data()),
                    parsed.tensor_data().size()));
          } else {
            void* copy = tsl::port::Malloc(tsize);
            std::memcpy(
                copy,
                std::string(node_value.mutable_tensor()->tensor_content())
                    .data(),
                tsize);
            TF_RETURN_IF_ERROR(ByteSwapBuffer((char*)copy, tsize, p_type, -1));
            (*node.mutable_attr())["value"]
                .mutable_tensor()
                ->set_tensor_content(
                    std::string(reinterpret_cast<const char*>(copy), tsize));
            tsl::port::Free(copy);
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ByteSwapTensorContentInMetaGraphDef(MetaGraphDef* meta_graph_def) {
  for (auto& function : *meta_graph_def->mutable_graph_def()
                             ->mutable_library()
                             ->mutable_function())
    for (auto& node : (*function.mutable_node_def()))
      TF_RETURN_IF_ERROR(ByteSwapTensorContentInNode(node));
  return absl::OkStatus();
}

absl::Status ByteSwapTensorContentInGraphDef(GraphDef* graph_def) {
  for (auto& node : *graph_def->mutable_node())
    TF_RETURN_IF_ERROR(ByteSwapTensorContentInNode(node));
  return absl::OkStatus();
}

}  // namespace tensorflow
