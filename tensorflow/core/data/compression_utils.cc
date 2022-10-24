/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/compression_utils.h"

#include <limits>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {

Status CompressElement(const std::vector<Tensor>& element,
                       CompressedElement* out) {
  // Determine the total amount of non`memcpy`able tensor data.
  std::vector<TensorProto> nonmemcpyable_components;
  size_t total_nonmemcpyable_size = 0;
  for (const auto& component : element) {
    if (!DataTypeCanUseMemcpy(component.dtype())) {
      nonmemcpyable_components.emplace_back();
      component.AsProtoTensorContent(&nonmemcpyable_components.back());
      total_nonmemcpyable_size +=
          nonmemcpyable_components.back().ByteSizeLong();
    }
  }

  // Build an array of `iovec`s pointing to the tensor data and compress it.
  // - `memcpy`able data is pointed to directly using a `TensorBuffer` and does
  // not take a copy before compression.
  // - Non`memcpy`able data is serialized and written into a string (a `tstring`
  // for access to `resize_uninitialized`) and does take a copy before
  // compression.
  std::vector<struct iovec> iov(element.size());
  size_t total_size = 0;
  tstring nonmemcpyable;
  nonmemcpyable.resize_uninitialized(total_nonmemcpyable_size);
  char* nonmemcpyable_pos = nonmemcpyable.mdata();
  int nonmemcpyable_component_index = 0;
  for (int i = 0; i < element.size(); ++i) {
    const auto& component = element[i];
    CompressedComponentMetadata* metadata =
        out->mutable_component_metadata()->Add();
    metadata->set_dtype(component.dtype());
    component.shape().AsProto(metadata->mutable_tensor_shape());
    if (DataTypeCanUseMemcpy(component.dtype())) {
      const TensorBuffer* buffer = DMAHelper::buffer(&component);
      if (buffer) {
        iov[i].iov_base = buffer->data();
        iov[i].iov_len = buffer->size();
        metadata->set_tensor_size_bytes(buffer->size());
      }
    } else {
      TensorProto& proto =
          nonmemcpyable_components[nonmemcpyable_component_index++];
      proto.SerializeToArray(nonmemcpyable_pos, proto.ByteSizeLong());
      iov[i].iov_base = nonmemcpyable_pos;
      iov[i].iov_len = proto.ByteSizeLong();
      nonmemcpyable_pos += proto.ByteSizeLong();
      metadata->set_tensor_size_bytes(proto.ByteSizeLong());
    }
    total_size += iov[i].iov_len;
  }
  if (total_size > kuint32max) {
    return errors::OutOfRange("Encountered dataset element of size ",
                              total_size, ", exceeding the 4GB Snappy limit.");
  }
  DCHECK_EQ(nonmemcpyable_pos,
            nonmemcpyable.mdata() + total_nonmemcpyable_size);
  if (!port::Snappy_CompressFromIOVec(iov.data(), total_size,
                                      out->mutable_data())) {
    return errors::Internal("Failed to compress using snappy.");
  }
  VLOG(3) << "Compressed element from " << total_size << " bytes to "
          << out->data().size() << " bytes";
  return OkStatus();
}

Status UncompressElement(const CompressedElement& compressed,
                         std::vector<Tensor>* out) {
  int num_components = compressed.component_metadata_size();
  out->clear();
  out->reserve(num_components);

  // Step 1: Prepare the memory that we will uncompress into.
  std::vector<struct iovec> iov(num_components);
  // We use tstring for access to resize_uninitialized.
  std::vector<tstring> tensor_proto_strs;
  // num_components is a conservative estimate. It is important to reserve
  // vector space so that the vector doesn't resize itself, which could
  // invalidate pointers to its strings' data.
  tensor_proto_strs.reserve(num_components);
  int64_t total_size = 0;
  for (int i = 0; i < num_components; ++i) {
    const CompressedComponentMetadata& metadata =
        compressed.component_metadata(i);
    if (DataTypeCanUseMemcpy(metadata.dtype())) {
      out->emplace_back(metadata.dtype(), metadata.tensor_shape());
      TensorBuffer* buffer = DMAHelper::buffer(&out->back());
      if (buffer) {
        iov[i].iov_base = buffer->data();
        iov[i].iov_len = buffer->size();
      } else {
        iov[i].iov_base = nullptr;
        iov[i].iov_len = 0;
      }
    } else {
      // Allocate an empty Tensor. We will fill it out later after
      // uncompressing into the tensor_proto_str.
      out->emplace_back();
      tensor_proto_strs.emplace_back();
      tstring& tensor_proto_str = tensor_proto_strs.back();
      tensor_proto_str.resize_uninitialized(metadata.tensor_size_bytes());
      iov[i].iov_base = tensor_proto_str.mdata();
      iov[i].iov_len = tensor_proto_str.size();
    }
    total_size += iov[i].iov_len;
  }

  // Step 2: Uncompress into the iovec.
  const std::string& compressed_data = compressed.data();
  size_t uncompressed_size;
  if (!port::Snappy_GetUncompressedLength(
          compressed_data.data(), compressed_data.size(), &uncompressed_size)) {
    return errors::Internal(
        "Could not get snappy uncompressed length. Compressed data size: ",
        compressed_data.size());
  }
  if (uncompressed_size != static_cast<size_t>(total_size)) {
    return errors::Internal(
        "Uncompressed size mismatch. Snappy expects ", uncompressed_size,
        " whereas the tensor metadata suggests ", total_size);
  }
  if (!port::Snappy_UncompressToIOVec(compressed_data.data(),
                                      compressed_data.size(), iov.data(),
                                      num_components)) {
    return errors::Internal("Failed to perform snappy decompression.");
  }

  // Step 3: Deserialize tensor proto strings to tensors.
  int tensor_proto_strs_index = 0;
  for (int i = 0; i < num_components; ++i) {
    if (DataTypeCanUseMemcpy(compressed.component_metadata(i).dtype())) {
      continue;
    }
    TensorProto tp;
    if (!tp.ParseFromString(tensor_proto_strs[tensor_proto_strs_index++])) {
      return errors::Internal("Could not parse TensorProto");
    }
    if (!out->at(i).FromProto(tp)) {
      return errors::Internal("Could not parse Tensor");
    }
  }
  return OkStatus();
}

}  // namespace data
}  // namespace tensorflow
