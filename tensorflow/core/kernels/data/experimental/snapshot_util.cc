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

#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/snappy/snappy_inputbuffer.h"
#include "tensorflow/core/lib/io/snappy/snappy_outputbuffer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/data/experimental/snapshot.pb.h"

namespace tensorflow {
namespace data {
namespace experimental {

SnapshotWriter::SnapshotWriter(WritableFile* dest,
                               const string& compression_type, int version,
                               const DataTypeVector& dtypes)
    : dest_(dest), compression_type_(compression_type), version_(version) {
#if defined(IS_SLIM_BUILD)
  if (compression_type != io::compression::kNone) {
    LOG(ERROR) << "Compression is unsupported on mobile platforms. Turning "
               << "off compression.";
  }
#else   // IS_SLIM_BUILD
  if (compression_type == io::compression::kGzip) {
    io::ZlibCompressionOptions zlib_options;
    zlib_options = io::ZlibCompressionOptions::GZIP();

    io::ZlibOutputBuffer* zlib_output_buffer =
        new io::ZlibOutputBuffer(dest, zlib_options.input_buffer_size,
                                 zlib_options.output_buffer_size, zlib_options);
    TF_CHECK_OK(zlib_output_buffer->Init());
    dest_ = zlib_output_buffer;
    dest_is_owned_ = true;
  }
#endif  // IS_SLIM_BUILD
  simple_tensor_mask_.reserve(dtypes.size());
  for (const auto& dtype : dtypes) {
    if (DataTypeCanUseMemcpy(dtype)) {
      simple_tensor_mask_.push_back(true);
      num_simple_++;
    } else {
      simple_tensor_mask_.push_back(false);
      num_complex_++;
    }
  }
}

Status SnapshotWriter::WriteTensors(const std::vector<Tensor>& tensors) {
  if (compression_type_ != io::compression::kSnappy) {
    experimental::SnapshotRecord record;
    for (const auto& tensor : tensors) {
      TensorProto* t = record.add_tensor();
      tensor.AsProtoTensorContent(t);
    }
#if defined(PLATFORM_GOOGLE)
    return WriteRecord(record.SerializeAsCord());
#else   // PLATFORM_GOOGLE
    return WriteRecord(record.SerializeAsString());
#endif  // PLATFORM_GOOGLE
  }

  if (version_ != 1) {
    return errors::InvalidArgument("Version: ", version_, " is not supported.");
  }
  if (compression_type_ != io::compression::kSnappy) {
    return errors::InvalidArgument(
        "Version 1 is only compatible with snappy compression");
  }

  std::vector<const TensorBuffer*> tensor_buffers;
  tensor_buffers.reserve(num_simple_);
  std::vector<TensorProto> tensor_protos;
  tensor_protos.reserve(num_complex_);
  SnapshotTensorMetadata metadata;
  int64 total_size = 0;
  for (int i = 0; i < tensors.size(); ++i) {
    const Tensor& tensor = tensors[i];
    TensorMetadata* tensor_metadata = metadata.add_tensor_metadata();
    tensor.shape().AsProto(tensor_metadata->mutable_tensor_shape());
    int64 size = 0;
    if (simple_tensor_mask_[i]) {
      auto tensor_buffer = DMAHelper::buffer(&tensor);
      tensor_buffers.push_back(tensor_buffer);
      size = tensor_buffer->size();
    } else {
      TensorProto proto;
      tensor.AsProtoTensorContent(&proto);
      size = proto.ByteSizeLong();
      tensor_protos.push_back(std::move(proto));
    }
    tensor_metadata->set_tensor_size_bytes(size);
    total_size += size;
  }

  std::vector<char> uncompressed(total_size);
  char* position = uncompressed.data();
  int buffer_index = 0;
  int proto_index = 0;
  for (int i = 0; i < tensors.size(); ++i) {
    const auto& tensor_metadata = metadata.tensor_metadata(i);
    if (simple_tensor_mask_[i]) {
      memcpy(position, tensor_buffers[buffer_index]->data(),
             tensor_metadata.tensor_size_bytes());
      buffer_index++;
    } else {
      tensor_protos[proto_index].SerializeToArray(
          position, tensor_metadata.tensor_size_bytes());
      proto_index++;
    }
    position += tensor_metadata.tensor_size_bytes();
  }
  DCHECK_EQ(position, uncompressed.data() + total_size);

  string output;
  if (!port::Snappy_Compress(uncompressed.data(), total_size, &output)) {
    return errors::Internal("Failed to compress using snappy.");
  }
#if defined(PLATFORM_GOOGLE)
  absl::Cord metadata_serialized = metadata.SerializeAsCord();
#else   // PLATFORM_GOOGLE
  std::string metadata_serialized = metadata.SerializeAsString();
#endif  // PLATFORM_GOOGLE
  TF_RETURN_IF_ERROR(WriteRecord(metadata_serialized));
  TF_RETURN_IF_ERROR(WriteRecord(output));
  return Status::OK();
}

Status SnapshotWriter::Sync() { return dest_->Sync(); }

Status SnapshotWriter::Close() {
  if (dest_is_owned_) {
    Status s = dest_->Close();
    delete dest_;
    dest_ = nullptr;
    return s;
  }
  return Status::OK();
}

SnapshotWriter::~SnapshotWriter() {
  if (dest_ != nullptr) {
    Status s = Close();
    if (!s.ok()) {
      LOG(ERROR) << "Could not finish writing file: " << s;
    }
  }
}

Status SnapshotWriter::WriteRecord(const StringPiece& data) {
  char header[kHeaderSize];
  core::EncodeFixed64(header, data.size());
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  return dest_->Append(data);
}

#if defined(PLATFORM_GOOGLE)
Status SnapshotWriter::WriteRecord(const absl::Cord& data) {
  char header[kHeaderSize];
  core::EncodeFixed64(header, data.size());
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  return dest_->Append(data);
}
#endif  // PLATFORM_GOOGLE

SnapshotReader::SnapshotReader(RandomAccessFile* file,
                               const string& compression_type, int version,
                               const DataTypeVector& dtypes)
    : file_(file),
      input_stream_(new io::RandomAccessInputStream(file)),
      compression_type_(compression_type),
      version_(version),
      dtypes_(dtypes) {
#if defined(IS_SLIM_BUILD)
  if (compression_type_ != io::compression::kNone) {
    LOG(ERROR) << "Compression is unsupported on mobile platforms. Turning "
               << "off compression.";
  }
#else   // IS_SLIM_BUILD
  if (compression_type_ == io::compression::kGzip) {
    io::ZlibCompressionOptions zlib_options;
    zlib_options = io::ZlibCompressionOptions::GZIP();

    input_stream_ = absl::make_unique<io::ZlibInputStream>(
        input_stream_.release(), zlib_options.input_buffer_size,
        zlib_options.output_buffer_size, zlib_options, true);
  } else if (compression_type_ == io::compression::kSnappy) {
    if (version_ == 0) {
      input_stream_ = absl::make_unique<io::SnappyInputBuffer>(
          file_, /*input_buffer_bytes=*/kSnappyReaderInputBufferSizeBytes,
          /*output_buffer_bytes=*/kSnappyReaderOutputBufferSizeBytes);
    } else {
      input_stream_ =
          absl::make_unique<io::BufferedInputStream>(file_, 64 << 20);
    }
  }
#endif  // IS_SLIM_BUILD
  simple_tensor_mask_.reserve(dtypes.size());
  for (const auto& dtype : dtypes) {
    if (DataTypeCanUseMemcpy(dtype)) {
      simple_tensor_mask_.push_back(true);
      num_simple_++;
    } else {
      simple_tensor_mask_.push_back(false);
      num_complex_++;
    }
  }
}

Status SnapshotReader::ReadTensors(std::vector<Tensor>* read_tensors) {
  profiler::TraceMe activity(
      [&]() { return absl::StrCat(kClassName, kSeparator, "ReadTensors"); },
      profiler::TraceMeLevel::kInfo);
  if (version_ == 0 || compression_type_ != io::compression::kSnappy) {
    return ReadTensorsV0(read_tensors);
  }
  if (version_ != 1) {
    return errors::InvalidArgument("Version: ", version_, " is not supported.");
  }
  if (compression_type_ != io::compression::kSnappy) {
    return errors::InvalidArgument("Version 1 only supports snappy.");
  }

  SnapshotTensorMetadata metadata;
  tstring metadata_str;
  TF_RETURN_IF_ERROR(ReadRecord(&metadata_str));
  if (!metadata.ParseFromArray(metadata_str.data(), metadata_str.size())) {
    return errors::DataLoss("Could not parse SnapshotTensorMetadata");
  }
  read_tensors->reserve(metadata.tensor_metadata_size());

  std::vector<Tensor> simple_tensors;
  simple_tensors.reserve(num_simple_);
  std::vector<std::pair<std::unique_ptr<char[]>, size_t>> tensor_proto_strs;
  tensor_proto_strs.reserve(num_complex_);
  TF_RETURN_IF_ERROR(
      SnappyUncompress(&metadata, &simple_tensors, &tensor_proto_strs));

  int simple_index = 0;
  int complex_index = 0;
  for (int i = 0; i < simple_tensor_mask_.size(); ++i) {
    if (simple_tensor_mask_[i]) {
      read_tensors->push_back(std::move(simple_tensors[simple_index]));
      simple_index++;
    } else {
      auto tensor_proto_str = std::move(tensor_proto_strs[complex_index].first);
      size_t tensor_proto_size = tensor_proto_strs[complex_index].second;
      TensorProto tp;
#if defined(PLATFORM_GOOGLE)
      auto tensor_proto_ptr = tensor_proto_str.release();
      absl::Cord c;
      c.AppendExternalMemory(
          absl::string_view(tensor_proto_ptr, tensor_proto_size),
          tensor_proto_ptr,
          [](void* arg) { delete[] static_cast<char*>(arg); });
      if (!tp.ParseFromCord(c)) {
        return errors::Internal("Could not parse TensorProto");
      }
#else   // PLATFORM_GOOGLE
      if (!tp.ParseFromArray(tensor_proto_str.get(), tensor_proto_size)) {
        return errors::Internal("Could not parse TensorProto");
      }
#endif  // PLATFORM_GOOGLE
      Tensor t;
      if (!t.FromProto(tp)) {
        return errors::Internal("Could not parse Tensor");
      }
      read_tensors->push_back(std::move(t));
      complex_index++;
    }
  }
  return Status::OK();
}

Status SnapshotReader::ReadTensorsV0(std::vector<Tensor>* read_tensors) {
  experimental::SnapshotRecord record;
#if defined(PLATFORM_GOOGLE)
  absl::Cord c;
  TF_RETURN_IF_ERROR(ReadRecord(&c));
  record.ParseFromCord(c);
#else   // PLATFORM_GOOGLE
  tstring record_bytes;
  TF_RETURN_IF_ERROR(ReadRecord(&record_bytes));
  record.ParseFromArray(record_bytes.data(), record_bytes.size());
#endif  // PLATFORM_GOOGLE
  read_tensors->reserve(record.tensor_size());
  for (int i = 0; i < record.tensor_size(); ++i) {
    read_tensors->emplace_back();
    if (!read_tensors->back().FromProto(record.tensor(i))) {
      return errors::DataLoss("Unable to parse tensor from proto.");
    }
  }
  return Status::OK();
}

Status SnapshotReader::SnappyUncompress(
    const SnapshotTensorMetadata* metadata, std::vector<Tensor>* simple_tensors,
    std::vector<std::pair<std::unique_ptr<char[]>, size_t>>*
        tensor_proto_strs) {
  tstring compressed;
  TF_RETURN_IF_ERROR(ReadRecord(&compressed));
  size_t size;
  if (!port::Snappy_GetUncompressedLength(compressed.data(), compressed.size(),
                                          &size)) {
    return errors::Internal("Could not get snappy uncompressed length");
  }

  int num_tensors = metadata->tensor_metadata_size();
  std::vector<struct iovec> iov(num_tensors);
  int index = 0;
  int64 total_size = 0;
  for (int i = 0; i < simple_tensor_mask_.size(); ++i) {
    const auto& tensor_metadata = metadata->tensor_metadata(i);
    if (simple_tensor_mask_[i]) {
      TensorShape shape(tensor_metadata.tensor_shape());
      Tensor simple_tensor(dtypes_[i], shape);
      TensorBuffer* buffer = DMAHelper::buffer(&simple_tensor);
      iov[index].iov_base = buffer->data();
      iov[index].iov_len = buffer->size();
      simple_tensors->push_back(std::move(simple_tensor));
    } else {
      auto tensor_proto_str =
          absl::make_unique<char[]>(tensor_metadata.tensor_size_bytes());
      iov[index].iov_base = tensor_proto_str.get();
      iov[index].iov_len = tensor_metadata.tensor_size_bytes();
      tensor_proto_strs->push_back(std::make_pair(
          std::move(tensor_proto_str), tensor_metadata.tensor_size_bytes()));
    }
    total_size += iov[index].iov_len;
    index++;
  }
  if (size != total_size) {
    return errors::Internal("Uncompressed size mismatch. Snappy expects ", size,
                            " whereas the tensor metadata suggests ",
                            total_size);
  }
  if (!port::Snappy_UncompressToIOVec(compressed.data(), compressed.size(),
                                      iov.data(), num_tensors)) {
    return errors::Internal("Failed to perform snappy decompression.");
  }
  return Status::OK();
}

Status SnapshotReader::ReadRecord(tstring* record) {
  tstring header;
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
  uint64 length = core::DecodeFixed64(header.data());
  return input_stream_->ReadNBytes(length, record);
}

#if defined(PLATFORM_GOOGLE)
Status SnapshotReader::ReadRecord(absl::Cord* record) {
  tstring header;
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
  uint64 length = core::DecodeFixed64(header.data());
  if (compression_type_ == io::compression::kNone) {
    return input_stream_->ReadNBytes(length, record);
  } else {
    auto tmp_str = absl::make_unique<tstring>();
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(length, tmp_str.get()));
    tstring* tmp_str_raw = tmp_str.release();
    record->AppendExternalMemory(*tmp_str_raw, tmp_str_raw,
                                 [](absl::string_view unused_data, void* arg) {
                                   delete static_cast<tstring*>(arg);
                                 });
    return Status::OK();
  }
}
#endif

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
