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
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace {

// Increment this when making changes to the `CompressedElement` proto. The
// `UncompressElement` function will determine what to read according to the
// version.
constexpr int kCompressedElementVersion = 0;

}  // namespace

class Iov {
 public:
  explicit Iov(size_t size) : iov_(size), idx_(0), num_bytes_(0) {}

  void Add(void* base, size_t len) {
    iov_[idx_].iov_base = base;
    iov_[idx_].iov_len = len;
    num_bytes_ += len;
    ++idx_;
  }

  iovec* Data() { return iov_.data(); }

  size_t NumBytes() const { return num_bytes_; }

  size_t NumPieces() const { return iov_.size(); }

 private:
  std::vector<struct iovec> iov_;
  size_t idx_;
  size_t num_bytes_;
};

Status CompressElement(const std::vector<Tensor>& element,
                       CompressedElement* out) {
  // First pass: preprocess the non`memcpy`able tensors.
  size_t num_string_tensors = 0;
  size_t num_string_tensor_strings = 0;
  std::vector<TensorProto> nonmemcpyable_components;
  size_t total_nonmemcpyable_size = 0;
  for (const auto& component : element) {
    if (component.dtype() == DT_STRING) {
      ++num_string_tensors;
      num_string_tensor_strings += component.NumElements();
    } else if (!DataTypeCanUseMemcpy(component.dtype())) {
      nonmemcpyable_components.emplace_back();
      component.AsProtoTensorContent(&nonmemcpyable_components.back());
      total_nonmemcpyable_size +=
          nonmemcpyable_components.back().ByteSizeLong();
    }
  }

  // Second pass: build an iov array of the tensor data.
  // - `memcpy`able tensors are pointed to directly from a single iovec.
  // - String tensors are pointed to directly from multiple iovecs (one for each
  // string).
  // - All other tensors are serialized and copied into a string (a `tstring`
  // for access to `resize_unitialized`).
  Iov iov{element.size() + num_string_tensor_strings - num_string_tensors};
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
        iov.Add(buffer->data(), buffer->size());
        metadata->add_uncompressed_bytes(buffer->size());
      }
    } else if (component.dtype() == DT_STRING) {
      const auto& flats = component.unaligned_flat<tstring>();
      for (int i = 0; i < flats.size(); ++i) {
        iov.Add(const_cast<char*>(flats.data()[i].data()),
                flats.data()[i].size());
        metadata->add_uncompressed_bytes(flats.data()[i].size());
      }
    } else {
      TensorProto& proto =
          nonmemcpyable_components[nonmemcpyable_component_index++];
      proto.SerializeToArray(nonmemcpyable_pos, proto.ByteSizeLong());
      iov.Add(nonmemcpyable_pos, proto.ByteSizeLong());
      nonmemcpyable_pos += proto.ByteSizeLong();
      metadata->add_uncompressed_bytes(proto.ByteSizeLong());
    }
  }

  if (iov.NumBytes() > kuint32max) {
    return errors::OutOfRange("Encountered dataset element of size ",
                              iov.NumBytes(),
                              ", exceeding the 4GB Snappy limit.");
  }
  if (!port::Snappy_CompressFromIOVec(iov.Data(), iov.NumBytes(),
                                      out->mutable_data())) {
    return errors::Internal("Failed to compress using snappy.");
  }
  out->set_version(kCompressedElementVersion);
  VLOG(3) << "Compressed element from " << iov.NumBytes() << " bytes to "
          << out->data().size() << " bytes";
  return absl::OkStatus();
}

Status UncompressElement(const CompressedElement& compressed,
                         std::vector<Tensor>* out) {
  if (compressed.version() != kCompressedElementVersion) {
    return errors::Internal("Unsupported compressed element version: ",
                            compressed.version());
  }
  int num_components = compressed.component_metadata_size();
  out->clear();
  out->reserve(num_components);

  // First pass: preprocess the non`memcpy`able tensors.
  size_t num_string_tensors = 0;
  size_t num_string_tensor_strings = 0;
  size_t total_nonmemcpyable_size = 0;
  for (const auto& metadata : compressed.component_metadata()) {
    if (metadata.dtype() == DT_STRING) {
      ++num_string_tensors;
      num_string_tensor_strings += metadata.uncompressed_bytes_size();
    } else if (!DataTypeCanUseMemcpy(metadata.dtype())) {
      total_nonmemcpyable_size += metadata.uncompressed_bytes(0);
    }
  }

  // Second pass: prepare the memory to be uncompressed into.
  // - `memcpy`able tensors are directly uncompressed into via a single iovec.
  // - String tensors are directly uncompressed into via multiple iovecs (one
  // for each string).
  // - All other tensors are uncompressed into a string (a `tstring` for access
  // to `resize_unitialized`).
  Iov iov{num_components + num_string_tensor_strings - num_string_tensors};
  tstring nonmemcpyable;
  nonmemcpyable.resize_uninitialized(total_nonmemcpyable_size);
  char* nonmemcpyable_pos = nonmemcpyable.mdata();
  for (const auto& metadata : compressed.component_metadata()) {
    if (DataTypeCanUseMemcpy(metadata.dtype())) {
      out->emplace_back(metadata.dtype(), metadata.tensor_shape());
      TensorBuffer* buffer = DMAHelper::buffer(&out->back());
      if (buffer) {
        iov.Add(buffer->data(), metadata.uncompressed_bytes(0));
      }
    } else if (metadata.dtype() == DT_STRING) {
      out->emplace_back(metadata.dtype(), metadata.tensor_shape());
      const auto& flats = out->back().unaligned_flat<tstring>();
      for (int i = 0; i < metadata.uncompressed_bytes_size(); ++i) {
        flats.data()[i].resize(metadata.uncompressed_bytes(i));
        iov.Add(flats.data()[i].mdata(), metadata.uncompressed_bytes(i));
      }
    } else {
      out->emplace_back();
      iov.Add(nonmemcpyable_pos, metadata.uncompressed_bytes(0));
      nonmemcpyable_pos += metadata.uncompressed_bytes(0);
    }
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
  if (uncompressed_size != static_cast<size_t>(iov.NumBytes())) {
    return errors::Internal(
        "Uncompressed size mismatch. Snappy expects ", uncompressed_size,
        " whereas the tensor metadata suggests ", iov.NumBytes());
  }
  if (!port::Snappy_UncompressToIOVec(compressed_data.data(),
                                      compressed_data.size(), iov.Data(),
                                      iov.NumPieces())) {
    return errors::Internal("Failed to perform snappy decompression.");
  }

  // Third pass: deserialize nonstring, non`memcpy`able tensors.
  nonmemcpyable_pos = nonmemcpyable.mdata();
  for (int i = 0; i < num_components; ++i) {
    const CompressedComponentMetadata& metadata =
        compressed.component_metadata(i);
    if (!DataTypeCanUseMemcpy(metadata.dtype()) &&
        metadata.dtype() != DT_STRING) {
      TensorProto tp;
      if (!tp.ParseFromString(
              {nonmemcpyable_pos,
               static_cast<size_t>(metadata.uncompressed_bytes(0))})) {
        return errors::Internal("Could not parse TensorProto");
      }
      if (!out->at(i).FromProto(tp)) {
        return errors::Internal("Could not parse Tensor");
      }
      nonmemcpyable_pos += metadata.uncompressed_bytes(0);
    }
  }
  return absl::OkStatus();
}

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CompressedElement,
                                       "tensorflow.data.CompressedElement");

}  // namespace data
}  // namespace tensorflow
