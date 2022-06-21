/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/versions.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap.h"
#include "tensorflow/core/util/tensor_slice_util.h"

#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#endif

namespace tensorflow {

// Versioning of the tensor bundle format.
const int kTensorBundleMinProducer = 0;
const int kTensorBundleMinConsumer = 0;
const int kTensorBundleVersion = 1;

// Size of our input buffer for streaming reads
static const int kBufferSize = 1024 * 1024;

// Key to the special BundleHeaderProto entry.  Do not change this, as clients
// can make the assumption that the header is always the first entry in the
// bundle.
const char* const kHeaderEntryKey = "";

namespace {

// Reads "num_elements" string elements from file[offset, offset+size) into the
// length-N "destination".  Discards the original content of "destination".
//
// Checksums the string lengths (as restored uint32 or uint64, not varint64
// bytes) and string bytes, and stores it into "actual_crc32c".
Status ReadStringTensor(io::InputBuffer* buffered_file, size_t num_elements,
                        size_t offset, size_t size, tstring* destination,
                        uint32* actual_crc32c, bool need_to_swap_bytes) {
  if (size == 0) return OkStatus();
  CHECK_GT(size, 0);

  // Reads "num_elements" varint64's from "buffered_file".
  TF_RETURN_IF_ERROR(buffered_file->Seek(offset));
  TF_RETURN_IF_ERROR(buffered_file->Hint(size));
  std::vector<uint64> string_lengths(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    TF_RETURN_IF_ERROR(buffered_file->ReadVarint64(&string_lengths[i]));
    if (string_lengths[i] <= UINT32_MAX) {
      // We need to do this because older checkpoints only used uint32s and we
      // should still support them.
      uint32 elem_size_uint32 = static_cast<uint32>(string_lengths[i]);
      if (need_to_swap_bytes) {
        // Checksum would have been computed on the source machine's byte order
        elem_size_uint32 = BYTE_SWAP_32(elem_size_uint32);
      }
      *actual_crc32c = crc32c::Extend(
          *actual_crc32c, reinterpret_cast<const char*>(&elem_size_uint32),
          sizeof(uint32));
    } else {
      uint64 length = string_lengths[i];
      if (need_to_swap_bytes) {
        length = BYTE_SWAP_64(length);
      }
      *actual_crc32c =
          crc32c::Extend(*actual_crc32c, reinterpret_cast<const char*>(&length),
                         sizeof(uint64));
    }
  }
  if (offset + size < buffered_file->Tell()) {
    return errors::DataLoss("String lengths longer than expected offset ",
                            offset + size);
  }

  // Reads the length-checksum.
  uint32 raw_length_checksum = 0;  // Bytes in file
  uint32 length_checksum = 0;      // In-memory representation
  size_t unused_bytes_read = 0;
  TF_RETURN_IF_ERROR(buffered_file->ReadNBytes(
      sizeof(uint32), reinterpret_cast<char*>(&raw_length_checksum),
      &unused_bytes_read));
  length_checksum = need_to_swap_bytes ? BYTE_SWAP_32(raw_length_checksum)
                                       : raw_length_checksum;
  if (crc32c::Unmask(length_checksum) != *actual_crc32c) {
    return errors::DataLoss(
        "The length checksum does not match: expected ",
        strings::Printf("%08u", crc32c::Unmask(length_checksum)),
        " but actual is ", strings::Printf("%08u", *actual_crc32c));
  }
  *actual_crc32c = crc32c::Extend(*actual_crc32c,
                                  reinterpret_cast<char*>(&raw_length_checksum),
                                  sizeof(uint32));

  // Reads the actual string bytes.
  for (size_t i = 0; i < num_elements; ++i) {
    const uint64 string_length = string_lengths[i];
    tstring* buffer = &destination[i];

    buffer->resize(string_length);
    size_t bytes_read = 0;
    TF_RETURN_IF_ERROR(
        buffered_file->ReadNBytes(string_length, &(*buffer)[0], &bytes_read));
    *actual_crc32c = crc32c::Extend(*actual_crc32c, buffer->data(), bytes_read);
  }
  return OkStatus();
}

Status ReadVariantTensor(io::InputBuffer* buffered_file, Tensor* ret,
                         size_t offset, size_t size, uint32* actual_crc32c) {
  // On-disk format:
  //   [varint64 len1][bytes variant1][4 byte checksum]
  //   ..
  //   [varint64 lenN][bytes variantN][4 byte checksum]
  // Var "crc32c" checksums all the lens, variant bytes, individual variant
  // checksums (as uint32, not varint32 bytes).
  if (size == 0) return OkStatus();
  size_t num_elements = ret->NumElements();

  // Reads the actual string bytes.
  TF_RETURN_IF_ERROR(buffered_file->Seek(offset));
  TF_RETURN_IF_ERROR(buffered_file->Hint(size));
  for (size_t i = 0; i < num_elements; ++i) {
    // Read the serialized variant length.
    uint64 string_length = 0;
    TF_RETURN_IF_ERROR(buffered_file->ReadVarint64(&string_length));
    *actual_crc32c = crc32c::Extend(
        *actual_crc32c, reinterpret_cast<const char*>(&string_length),
        sizeof(uint64));
    // Read the actual serialized variant.
    string buffer;
    buffer.resize(string_length);
    size_t bytes_read = 0;
    TF_RETURN_IF_ERROR(
        buffered_file->ReadNBytes(string_length, &buffer[0], &bytes_read));
    *actual_crc32c = crc32c::Extend(*actual_crc32c, buffer.data(), bytes_read);
    VariantTensorDataProto proto;
    if (!proto.ParseFromString(buffer)) {
      return errors::DataLoss("Unable to parse VariantTensorDataProto from ",
                              "buffer of size ", string_length, ". ",
                              "Bundle entry offset: ", offset, " size: ", size);
    }
    Variant v = proto;
    if (!DecodeUnaryVariant(&v)) {
      return errors::Internal("Could not decode variant with type_name: \"",
                              v.TypeName(), "\".  Perhaps you forgot to ",
                              "register a decoder via ",
                              "REGISTER_UNARY_VARIANT_DECODE_FUNCTION?");
    }

    // Read the checksum.
    uint32 checksum = 0;
    size_t unused_bytes_read = 0;
    TF_RETURN_IF_ERROR(buffered_file->ReadNBytes(
        sizeof(uint32), reinterpret_cast<char*>(&checksum),
        &unused_bytes_read));
    if (crc32c::Unmask(checksum) != *actual_crc32c) {
      return errors::DataLoss(
          "The checksum after Variant ", i, " does not match.",
          " Expected: ", strings::Printf("%08u", crc32c::Unmask(checksum)),
          " Actual: ", strings::Printf("%08u", *actual_crc32c));
    }
    *actual_crc32c = crc32c::Extend(
        *actual_crc32c, reinterpret_cast<char*>(&checksum), sizeof(uint32));

    ret->flat<Variant>()(i) = std::move(v);
  }

  return OkStatus();
}

char* GetBackingBuffer(const Tensor& val) {
  CHECK(DataTypeCanUseMemcpy(val.dtype())) << val.dtype();
  return const_cast<char*>(val.tensor_data().data());
}

tstring* GetStringBackingBuffer(const Tensor& val) {
  CHECK_EQ(DT_STRING, val.dtype());
  return const_cast<tstring*>(val.flat<tstring>().data());
}

Status ParseEntryProto(StringPiece key, StringPiece value,
                       protobuf::MessageLite* out) {
  if (!out->ParseFromArray(value.data(), value.size())) {
    return errors::DataLoss("Entry for key ", key, " not parseable.");
  }
  return OkStatus();
}

// Serializes the data bytes of the non-string tensor "val".  Discards the
// original content of "bytes_written", and on OK updates it with number of
// bytes written.
// REQUIRES: val.dtype() != DT_STRING
Status WriteTensor(const Tensor& val, FileOutputBuffer* out,
                   size_t* bytes_written) {
  DCHECK_NE(val.dtype(), DT_STRING);
  DCHECK_NE(val.dtype(), DT_VARIANT);
  *bytes_written = val.TotalBytes();
  char* buf = GetBackingBuffer(val);
  VLOG(1) << "Appending " << *bytes_written << " bytes to file";
  return out->Append(StringPiece(buf, *bytes_written));
}

// Serializes string tensor "val".  "bytes_written" is treated in the same
// fashion as WriteTensor().
//
// Checksums all bytes written and stores it into "crc32c".
// REQUIRES: val.dtype() == DT_STRING
Status WriteStringTensor(const Tensor& val, FileOutputBuffer* out,
                         size_t* bytes_written, uint32* crc32c) {
  // On-disk format:
  //   [varint64 len0]..[varint64 lenL][4 byte cksum on lengths][string bytes]
  // Var "crc32c" checksums the string lengths (as uint64, not varint64 bytes),
  // the length-checksum, and all the string bytes.
  DCHECK_EQ(val.dtype(), DT_STRING);
  const tstring* strings = GetStringBackingBuffer(val);

  // Writes the varint lengths.
  string lengths;
  lengths.reserve(val.NumElements());  // At least 1 byte per element.
  *crc32c = 0;
  for (int64_t i = 0; i < val.NumElements(); ++i) {
    const tstring* elem = &strings[i];
    DCHECK_EQ(elem->size(), static_cast<uint64>(elem->size()));
    const uint64 elem_size = static_cast<uint64>(elem->size());

    core::PutVarint64(&lengths, elem_size);
    if (elem_size <= UINT32_MAX) {
      // We need to do this because older checkpoints only used uint32s and we
      // should still support them.
      const uint32 elem_size_uint32 = static_cast<uint32>(elem_size);
      *crc32c = crc32c::Extend(*crc32c,
                               reinterpret_cast<const char*>(&elem_size_uint32),
                               sizeof(uint32));
    } else {
      *crc32c = crc32c::Extend(
          *crc32c, reinterpret_cast<const char*>(&elem_size), sizeof(uint64));
    }
  }
  TF_RETURN_IF_ERROR(out->Append(lengths));
  *bytes_written = lengths.size();

  // Writes the length checksum.
  const uint32 length_checksum = crc32c::Mask(*crc32c);
  TF_RETURN_IF_ERROR(out->Append(StringPiece(
      reinterpret_cast<const char*>(&length_checksum), sizeof(uint32))));
  *crc32c = crc32c::Extend(
      *crc32c, reinterpret_cast<const char*>(&length_checksum), sizeof(uint32));
  *bytes_written += sizeof(uint32);

  // Writes all the string bytes out.
  for (int64_t i = 0; i < val.NumElements(); ++i) {
    const tstring* string = &strings[i];
    TF_RETURN_IF_ERROR(out->Append(*string));
    *bytes_written += string->size();
    *crc32c = crc32c::Extend(*crc32c, string->data(), string->size());
  }
  return OkStatus();
}

Status WriteVariantTensor(const Tensor& val, FileOutputBuffer* out,
                          size_t* bytes_written, uint32* crc32c) {
  // On-disk format:
  //   [varint64 len1][bytes variant1][4 byte checksum]
  //   ..
  //   [varint64 lenN][bytes variantN][4 byte checksum]
  // Var "crc32c" checksums all the lens, variant bytes, individual variant
  // checksums (as uint32, not varint32 bytes).
  DCHECK_EQ(val.dtype(), DT_VARIANT);

  *crc32c = 0;
  *bytes_written = 0;
  for (int64_t i = 0; i < val.NumElements(); ++i) {
    VariantTensorData data;
    val.flat<Variant>()(i).Encode(&data);
    VariantTensorDataProto proto;
    data.ToProto(&proto);
    string elem;
    if (!proto.SerializeToString(&elem)) {
      return errors::Unknown(
          "Failed to serialize tensor data of size ", proto.ByteSizeLong(),
          ". Tensor: ", val.flat<Variant>()(i).DebugString());
    }

    // Write the length of the serialized variant.
    DCHECK_EQ(elem.size(), static_cast<uint64>(elem.size()));
    const auto elem_size = static_cast<uint64>(elem.size());
    string len;
    core::PutVarint64(&len, elem_size);
    TF_RETURN_IF_ERROR(out->Append(len));
    *crc32c = crc32c::Extend(*crc32c, reinterpret_cast<const char*>(&elem_size),
                             sizeof(uint64));
    *bytes_written += len.size();

    // Write the serialized variant.
    TF_RETURN_IF_ERROR(out->Append(elem));
    *crc32c = crc32c::Extend(*crc32c, elem.data(), elem.size());
    *bytes_written += elem.size();

    // Write the checksum.
    const uint32 length_checksum = crc32c::Mask(*crc32c);
    TF_RETURN_IF_ERROR(out->Append(StringPiece(
        reinterpret_cast<const char*>(&length_checksum), sizeof(uint32))));
    *crc32c =
        crc32c::Extend(*crc32c, reinterpret_cast<const char*>(&length_checksum),
                       sizeof(uint32));
    *bytes_written += sizeof(uint32);
  }

  return OkStatus();
}

// Returns whether "slice_spec" is a full slice, with respect to the full shape.
//
// This can happen say, when "slice_spec" is
// "TensorSlice(full_tensor_shape.dims())", or when it is "TensorSlice({{0,
// dim(0)}, ..., {0, dim(N)}})" -- a degenerate case we need to guard against.
bool IsFullSlice(const TensorSlice& slice_spec,
                 const TensorShape& full_tensor_shape) {
  if (slice_spec.IsFull()) {
    return true;
  } else {
    TensorShape sliced_shape;
    slice_spec.SliceTensorShape(full_tensor_shape, &sliced_shape).IgnoreError();
    return sliced_shape == full_tensor_shape;
  }
}

Status CorruptFileError(const Status& in_status, const string& filename,
                        const string& detail) {
  if (in_status.ok()) {
    return errors::Internal("Unable to read file (", filename,
                            "). Perhaps the file is corrupt or was produced by "
                            "a newer version of TensorFlow with format changes "
                            "(",
                            detail, ")");
  }
  return Status(
      in_status.code(),
      strings::StrCat("Unable to read file (", filename,
                      "). Perhaps the file is corrupt or was produced by a "
                      "newer version of TensorFlow with format changes (",
                      detail, "): ", in_status.error_message()));
}

table::Options TableBuilderOptions() {
  table::Options o;
  // Compressed tables cannot be read by TensorFlow releases prior to 1.1.
  // To smoothen the transition, compressed writes are disabled for now
  // (version 1.2) with the intention that they will be enabled again at
  // some point (perhaps the 1.3 release?).
  o.compression = table::kNoCompression;
  return o;
}

// Writes zeros to output buffer to align the next write to the requested
// alignment. "size" is the current size of the buffer and is updated to the
// new size.
Status PadAlignment(FileOutputBuffer* out, int alignment, int64_t* size) {
  int bytes_over = *size % alignment;
  if (bytes_over == 0) {
    return OkStatus();
  }
  int bytes_to_write = alignment - bytes_over;
  Status status = out->Append(string(bytes_to_write, '\0'));
  if (status.ok()) {
    *size += bytes_to_write;
  }
  return status;
}

}  // namespace

BundleWriter::BundleWriter(Env* env, StringPiece prefix, const Options& options)
    : env_(env), options_(options), prefix_(prefix), out_(nullptr), size_(0) {
  status_ = env_->HasAtomicMove(prefix_, &use_temp_file_);
  if (!status_.ok()) return;

  data_path_ = DataFilename(prefix_, 0, 1);
  metadata_path_ = MetaFilename(prefix_);
  if (use_temp_file_) {
    data_path_ = strings::StrCat(data_path_, ".tempstate", random::New64());
    metadata_path_ =
        strings::StrCat(metadata_path_, ".tempstate", random::New64());
  }

  status_ = env_->CreateDir(string(io::Dirname(prefix_)));
  if (!status_.ok() && !errors::IsAlreadyExists(status_)) {
    return;
  }

  std::unique_ptr<WritableFile> wrapper;
  status_ = env_->NewWritableFile(data_path_, &wrapper);
  if (!status_.ok()) return;
  out_ = std::unique_ptr<FileOutputBuffer>(
      new FileOutputBuffer(wrapper.release(), 8 << 20 /* 8MB write buffer */));

  VLOG(1) << "Writing to file " << data_path_;
}

Status BundleWriter::Add(StringPiece key, const Tensor& val) {
  if (!status_.ok()) return status_;
  CHECK_NE(key, kHeaderEntryKey);
  const string key_string(key);
  if (entries_.find(key_string) != entries_.end()) {
    status_ = errors::InvalidArgument("Adding duplicate key: ", key);
    return status_;
  }

  BundleEntryProto* entry = &entries_[key_string];
  entry->set_dtype(val.dtype());
  val.shape().AsProto(entry->mutable_shape());
  entry->set_shard_id(0);
  entry->set_offset(size_);

  // Updates the data file.
  size_t data_bytes_written = 0;
  uint32 crc32c = 0;
  out_->clear_crc32c();
  if (val.dtype() == DT_STRING) {
    status_ = WriteStringTensor(val, out_.get(), &data_bytes_written, &crc32c);
  } else if (val.dtype() == DT_VARIANT) {
    status_ = WriteVariantTensor(val, out_.get(), &data_bytes_written, &crc32c);
  } else {
    status_ = WriteTensor(val, out_.get(), &data_bytes_written);
    crc32c = out_->crc32c();
  }

  if (status_.ok()) {
    entry->set_size(data_bytes_written);
    entry->set_crc32c(crc32c::Mask(crc32c));
    size_ += data_bytes_written;
    status_ = PadAlignment(out_.get(), options_.data_alignment, &size_);
  }
  return status_;
}

Status BundleWriter::AddSlice(StringPiece full_tensor_key,
                              const TensorShape& full_tensor_shape,
                              const TensorSlice& slice_spec,
                              const Tensor& slice_tensor) {
  if (!status_.ok()) return status_;
  CHECK_NE(full_tensor_key, kHeaderEntryKey);

  // If just a singleton full slice, use the regular Add() to be more efficient.
  if (IsFullSlice(slice_spec, full_tensor_shape)) {
    return Add(full_tensor_key, slice_tensor);
  }

  // Inserts/updates the full tensor's metadata entry.
  //
  // In the case of a sharded save, MergeBundles() is responsible for merging
  // the "slices" field of multiple metadata entries corresponding to the same
  // full tensor.
  const string full_tensor_key_string(full_tensor_key);
  BundleEntryProto* full_entry = &entries_[full_tensor_key_string];
  if (full_entry->dtype() != DT_INVALID) {
    CHECK_EQ(full_entry->dtype(), slice_tensor.dtype());
  }
  if (full_entry->has_shape()) {
    CHECK(TensorShape(full_entry->shape()) == full_tensor_shape);
  }

  // Populates dtype, shape, and slices.  Intentionally leaving out shard_id and
  // offset, which do not make sense for this full tensor entry.
  full_entry->set_dtype(slice_tensor.dtype());
  full_tensor_shape.AsProto(full_entry->mutable_shape());
  TensorSliceProto* slice_proto = full_entry->add_slices();
  slice_spec.AsProto(slice_proto);

  // The slice itself is handled by a regular Add(), which includes adding its
  // own metadata entry, and writing out the slice's values.
  const string slice_name =
      checkpoint::EncodeTensorNameSlice(full_tensor_key_string, slice_spec);
  status_ = Add(slice_name, slice_tensor);
  return status_;
}

// TODO(zongheng): on metadata write failure or !status_.ok(), consider removing
// the orphaned data file.
Status BundleWriter::Finish() {
  if (out_) {
    status_.Update(out_->Close());
    out_ = nullptr;
    if (status_.ok()) {
      if (use_temp_file_) {
        status_ =
            Env::Default()->RenameFile(data_path_, DataFilename(prefix_, 0, 1));
      }
    } else {
      Env::Default()->DeleteFile(data_path_).IgnoreError();
    }
  }
  if (!status_.ok()) return status_;
  // Build key -> BundleEntryProto table.
  std::unique_ptr<WritableFile> file;
  status_ = env_->NewWritableFile(metadata_path_, &file);
  if (!status_.ok()) return status_;
  {
    // N.B.: the default use of Snappy compression may not be supported on all
    // platforms (e.g. Android).  The metadata file is small, so this is fine.
    table::Options options;
    options.compression = table::kNoCompression;
    table::TableBuilder builder(options, file.get());
    // Header entry.
    BundleHeaderProto header;
    header.set_num_shards(1);
    header.set_endianness(BundleHeaderProto::LITTLE);
    if (!port::kLittleEndian) header.set_endianness(BundleHeaderProto::BIG);
    VersionDef* version = header.mutable_version();
    version->set_producer(kTensorBundleVersion);
    version->set_min_consumer(kTensorBundleMinConsumer);

    builder.Add(kHeaderEntryKey, header.SerializeAsString());

    // All others.
    for (const auto& p : entries_) {
      builder.Add(p.first, p.second.SerializeAsString());
    }
    status_ = builder.Finish();
  }
  status_.Update(file->Close());
  if (!status_.ok()) {
    Env::Default()->DeleteFile(metadata_path_).IgnoreError();
    return status_;
  } else if (use_temp_file_) {
    status_ = Env::Default()->RenameFile(metadata_path_, MetaFilename(prefix_));
    if (!status_.ok()) return status_;
  }
  status_ = errors::Internal("BundleWriter is closed");
  return OkStatus();
}

// Merging tensor bundles.

// Accumulator of metadata states during a merge.
struct MergeState {
  // Accumulated from the header entries.
  int num_shards = 0;

  // Derives "endianness" and "version" from the first bundle merged (hence the
  // "seen_first_bundle" guard).  The two fields must be the same for all
  // bundles in a merge.
  bool seen_first_bundle = false;
  BundleHeaderProto_Endianness endianness;
  VersionDef version;

  // Tensor key -> BundleEntryProto.
  std::map<string, BundleEntryProto> entries;
  // Data file path -> new shard id in the final merged bundle.
  std::unordered_map<string, int32> shard_ids;
};

// Merges entries of "prefix" into the accumulator state "merge".
// Returns OK iff the merge succeeds.
static Status MergeOneBundle(Env* env, StringPiece prefix,
                             MergeState* merge_state) {
  VLOG(1) << "Merging bundle:" << prefix;
  const string filename = MetaFilename(prefix);
  uint64 file_size;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  table::Table* table = nullptr;
  TF_RETURN_IF_ERROR(
      table::Table::Open(TableBuilderOptions(), file.get(), file_size, &table));
  std::unique_ptr<table::Table> table_deleter(table);
  std::unique_ptr<table::Iterator> iter(table->NewIterator());

  int num_shards;
  // Process header.
  {
    iter->Seek(kHeaderEntryKey);
    if (!iter->Valid()) {
      return CorruptFileError(iter->status(), filename,
                              "failed to seek to header entry");
    }
    BundleHeaderProto header;
    Status s = ParseEntryProto(iter->key(), iter->value(), &header);
    if (!s.ok()) return CorruptFileError(s, filename, "unable to parse header");

    merge_state->num_shards += header.num_shards();
    if (!merge_state->seen_first_bundle) {
      merge_state->seen_first_bundle = true;
      merge_state->endianness = header.endianness();
      merge_state->version = header.version();
    } else {
      // Validates "endianness".
      if (merge_state->endianness != header.endianness()) {
        return errors::InvalidArgument(
            "Merging bundles with conflicting endianness; inputs corrupted?");
      }
      // Validates "version".
      string curr_version, merge_version;
      header.version().SerializeToString(&curr_version);
      merge_state->version.SerializeToString(&merge_version);
      if (curr_version != merge_version) {
        return errors::InvalidArgument(
            "Merging bundles with different format versions: merged ",
            merge_version, " vs. curr ", curr_version);
      }
    }
    num_shards = header.num_shards();
    iter->Next();
  }

  // Loops through the non-header to-merge entries.
  BundleEntryProto to_merge_entry;
  for (; iter->Valid(); iter->Next()) {
    const string key(iter->key());
    const auto entry_iter = merge_state->entries.find(key);

    // Illegal: the duplicated entry is a non-slice tensor.
    if (entry_iter != merge_state->entries.end() &&
        entry_iter->second.slices().empty()) {
      return errors::InvalidArgument(
          "Duplicate tensor keyed by ", key,
          " encountered, when merging prefix: ", prefix);
    }

    TF_RETURN_IF_ERROR(
        ParseEntryProto(iter->key(), iter->value(), &to_merge_entry));

    // The duplicated entry holds metadata for a sliced full tensor.
    // Allows the duplication and merges "slices".
    if (entry_iter != merge_state->entries.end()) {
      BundleEntryProto& existing_entry = entry_iter->second;
      if (to_merge_entry.slices().empty()) {
        return errors::Internal(
            "Duplicate tensor keyed by ", key,
            "; attempting to merge in a non-slice bundle entry");
      }
      // Only needs merge the "slices" field (and validate dtype/shape).
      for (int i = 0; i < to_merge_entry.slices_size(); ++i) {
        TensorSliceProto* slot = existing_entry.add_slices();
        *slot = to_merge_entry.slices(i);
      }
      CHECK_EQ(existing_entry.dtype(), to_merge_entry.dtype());
      CHECK(TensorShape(existing_entry.shape()) ==
            TensorShape(to_merge_entry.shape()));
      continue;
    }

    // Key doesn't duplicate: a fresh tensor/slice entry.
    auto result = merge_state->shard_ids.insert(
        {DataFilename(prefix, to_merge_entry.shard_id(), num_shards),
         merge_state->shard_ids.size()});
    to_merge_entry.set_shard_id(result.first->second);
    merge_state->entries[key] = to_merge_entry;
  }
  return OkStatus();
}

Status MergeBundles(Env* env, gtl::ArraySlice<tstring> prefixes,
                    StringPiece merged_prefix) {
  // Merges all metadata tables.
  // TODO(zhifengc): KeyValue sorter if it becomes too big.
  MergeState merge;
  Status status = env->CreateDir(string(io::Dirname(merged_prefix)));
  if (!status.ok() && !errors::IsAlreadyExists(status)) return status;
  for (int i = 0; i < prefixes.size(); ++i) {
    TF_RETURN_IF_ERROR(MergeOneBundle(env, prefixes[i], &merge));
  }

  // Renames data files to contain the merged bundle prefix.
  for (const auto& p : merge.shard_ids) {
    VLOG(1) << "Renaming " << p.first << " to "
            << DataFilename(merged_prefix, p.second, merge.shard_ids.size());
    TF_RETURN_IF_ERROR(env->RenameFile(
        p.first,
        DataFilename(merged_prefix, p.second, merge.shard_ids.size())));
  }

  // Writes the final metadata table under the merged prefix.
  std::unique_ptr<WritableFile> merged_metadata;
  TF_RETURN_IF_ERROR(
      env->NewWritableFile(MetaFilename(merged_prefix), &merged_metadata));
  {
    table::TableBuilder builder(TableBuilderOptions(), merged_metadata.get());
    // Header entry.
    BundleHeaderProto header;
    header.set_num_shards(merge.num_shards);
    header.set_endianness(merge.endianness);
    *header.mutable_version() = merge.version;
    builder.Add(kHeaderEntryKey, header.SerializeAsString());
    // All others.
    for (const auto& p : merge.entries) {
      builder.Add(p.first, p.second.SerializeAsString());
    }
    status = builder.Finish();
  }
  status.Update(merged_metadata->Close());
  if (!status.ok()) return status;
  VLOG(1) << "Merged bundles to:" << merged_prefix;

  // Cleanup: best effort based and ignores errors.
  for (const tstring& prefix : prefixes) {
    env->DeleteFile(MetaFilename(prefix)).IgnoreError();
  }
  return status;
}

// Interface for reading a tensor bundle.

BundleReader::BundleReader(Env* env, StringPiece prefix)
    : env_(env),
      prefix_(prefix),
      metadata_(nullptr),
      table_(nullptr),
      index_cache_(nullptr),
      iter_(nullptr),
      need_to_swap_bytes_(false) {
  const string filename = MetaFilename(prefix_);
  uint64 file_size;
  status_ = env_->GetFileSize(filename, &file_size);
  if (!status_.ok()) return;

  // Opens the metadata table.
  std::unique_ptr<RandomAccessFile> wrapper;
  status_ = env_->NewRandomAccessFile(filename, &wrapper);
  if (!status_.ok()) return;
  metadata_ = wrapper.release();

  table::Options o;
  int64_t cache_size;
  Status s =
      ReadInt64FromEnvVar("TF_TABLE_INDEX_CACHE_SIZE_IN_MB", 0, &cache_size);
  if (s.ok() && cache_size > 0) {
    index_cache_ = table::NewLRUCache(cache_size << 20);
    o.block_cache = index_cache_;
  }

  status_ = table::Table::Open(o, metadata_, file_size, &table_);
  if (!status_.ok()) return;
  iter_ = table_->NewIterator();

  // Reads "num_shards_" from the first entry.
  iter_->Seek(kHeaderEntryKey);
  if (!iter_->Valid()) {
    status_ = CorruptFileError(iter_->status(), filename,
                               "failed to seek to header entry");
    return;
  }
  BundleHeaderProto header;
  status_ = ParseEntryProto(iter_->key(), iter_->value(), &header);
  if (!status_.ok()) {
    status_ = CorruptFileError(status_, filename, "unable to parse header");
    return;
  }
  num_shards_ = header.num_shards();
  if ((header.endianness() == BundleHeaderProto::BIG && port::kLittleEndian) ||
      (header.endianness() == BundleHeaderProto::LITTLE &&
       !port::kLittleEndian)) {
    need_to_swap_bytes_ = true;
  }
  status_ = CheckVersions(header.version(), kTensorBundleVersion,
                          kTensorBundleMinProducer, "Checkpoint", "checkpoint");
}

BundleReader::~BundleReader() {
  delete metadata_;
  delete iter_;
  delete table_;
  if (index_cache_) {
    delete index_cache_;
  }
  // InputBuffer does not own the underlying RandomAccessFile.
  for (auto pair : data_) {
    if (pair.second != nullptr && pair.second->file() != nullptr) {
      delete pair.second->file();
    }
  }
  for (auto& temp : data_) {
    delete temp.second;
  }
  for (auto& temp : tensor_slices_) {
    delete temp.second;
  }
  data_.clear();
  tensor_slices_.clear();
}

Status BundleReader::GetBundleEntryProto(StringPiece key,
                                         BundleEntryProto* entry) {
  entry->Clear();
  TF_CHECK_OK(status_);
  Seek(key);
  if (!iter_->Valid() || iter_->key() != key) {
    return errors::NotFound("Key ", key, " not found in checkpoint");
  }

  BundleEntryProto entry_copy;
  TF_RETURN_IF_ERROR(
      ParseEntryProto(iter_->key(), iter_->value(), &entry_copy));
  if (!TensorShape::IsValid(entry_copy.shape())) {
    return errors::DataLoss("Invalid tensor shape: ", key, " ",
                            entry_copy.shape().ShortDebugString());
  }

  entry->Swap(&entry_copy);
  return OkStatus();
}

Status BundleReader::GetValue(const BundleEntryProto& entry, Tensor* val) {
  Tensor* ret = val;
  const TensorShape stored_shape(TensorShape(entry.shape()));
  if (val->NumElements() == 0) {
    ret = new Tensor(entry.dtype(), stored_shape);
  }

  // Validates the "size" field.
  if (entry.dtype() != DT_STRING && entry.dtype() != DT_VARIANT) {
    if (entry.size() != ret->TotalBytes()) {
      return errors::DataLoss("Invalid size in bundle entry: key ", key(),
                              "; stored size ", entry.size(),
                              "; expected size ", ret->TotalBytes());
    }
  } else if (entry.dtype() == DT_STRING) {
    // Relaxes the check for string tensors as follows:
    //   entry.size() == bytes(varint lengths) + bytes(data)
    //                >= NumElems + bytes(data), since size bytes(varint) >= 1.
    //   TotalBytes() == sizeof(tstring) * NumElems + bytes(data)
    // Since we don't know bytes(varint lengths), we just check an inequality.
    const size_t lower_bound = ret->NumElements() + ret->TotalBytes() -
                               sizeof(tstring) * ret->NumElements();
    if (entry.size() < lower_bound) {
      return errors::DataLoss("Invalid size in bundle entry: key ", key(),
                              "; stored size ", entry.size(),
                              "; expected size is at least ", lower_bound);
    }
  }

  // Open the data file if it has not been opened.
  io::InputBuffer* buffered_file = data_[entry.shard_id()];
  if (buffered_file == nullptr) {
    std::unique_ptr<RandomAccessFile> file = nullptr;
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(
        DataFilename(prefix_, entry.shard_id(), num_shards_), &file));
    buffered_file = new io::InputBuffer(file.release(), kBufferSize);
    // The InputBuffer and RandomAccessFile objects are both released in dtor.
    data_[entry.shard_id()] = buffered_file;
  }
  CHECK(buffered_file != nullptr);

  TF_RETURN_IF_ERROR(buffered_file->Seek(entry.offset()));
  uint32 actual_crc32c = 0;

  if (DataTypeCanUseMemcpy(entry.dtype())) {
    char* backing_buffer = const_cast<char*>((ret->tensor_data().data()));
    size_t unused_bytes_read;
    if (entry.size() > kBufferSize) {
      StringPiece sp;
      TF_RETURN_IF_ERROR(buffered_file->file()->Read(
          entry.offset(), entry.size(), &sp, backing_buffer));
      if (sp.data() != backing_buffer) {
        memmove(backing_buffer, sp.data(), entry.size());
      }
    } else {
      TF_RETURN_IF_ERROR(buffered_file->ReadNBytes(entry.size(), backing_buffer,
                                                   &unused_bytes_read));
    }
    // Note that we compute the checksum *before* byte-swapping. The checksum
    // should be on the bytes in the order they appear in the file.
    actual_crc32c = crc32c::Value(backing_buffer, entry.size());
    if (need_to_swap_bytes_) {
      TF_RETURN_IF_ERROR(ByteSwapTensor(ret));
    }
  } else if (entry.dtype() == DT_VARIANT) {
    if (need_to_swap_bytes_) {
      return errors::Unimplemented(
          "TensorBundle at ", prefix_,
          "is of a different endianness than this machine's hardware, and "
          "the bundle contains a variant (arbitrary C++ type) tensor. "
          "Byte-swapping of variant tensors is not currently implemented.");
    }
    // Relies on io::InputBuffer's buffering, because we issue many neighboring
    // reads for a single string tensor.
    TF_RETURN_IF_ERROR(ReadVariantTensor(buffered_file, ret, entry.offset(),
                                         entry.size(), &actual_crc32c));
  } else {
    // Relies on io::InputBuffer's buffering, because we issue many neighboring
    // reads for a single string tensor.
    TF_RETURN_IF_ERROR(ReadStringTensor(
        buffered_file, ret->NumElements(), entry.offset(), entry.size(),
        GetStringBackingBuffer(*ret), &actual_crc32c, need_to_swap_bytes_));
  }
  if (crc32c::Unmask(entry.crc32c()) != actual_crc32c) {
    return errors::DataLoss(
        "TensorBundle at ", prefix_, " shard ", entry.shard_id(), " (",
        entry.size(), " bytes): Checksum does not match: stored ",
        strings::Printf("%08u", crc32c::Unmask(entry.crc32c())),
        " vs. calculated on the restored bytes ", actual_crc32c);
  }

  *val = *ret;
  if (ret != val) delete ret;
  return OkStatus();
}

Status BundleReader::Lookup(StringPiece key, Tensor* val) {
  CHECK(val != nullptr);
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(GetBundleEntryProto(key, &entry));

  if (entry.slices().empty()) {
    return GetValue(entry, val);
  } else {
    return GetSliceValue(
        key, entry,
        /* a full slice */ TensorSlice(TensorShape(entry.shape()).dims()), val);
  }
}

Status BundleReader::ReadCurrent(Tensor* val) {
  CHECK(val != nullptr);
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(ParseEntryProto(iter_->key(), iter_->value(), &entry));
  if (!TensorShape::IsValid(entry.shape())) {
    return errors::DataLoss("Invalid tensor shape: ", iter_->key(), " ",
                            entry.shape().ShortDebugString());
  }

  if (entry.slices().empty()) {
    return GetValue(entry, val);
  } else {
    return GetSliceValue(
        iter_->key(), entry,
        /* a full slice */ TensorSlice(TensorShape(entry.shape()).dims()), val);
  }
}

Status BundleReader::LookupTensorSlices(StringPiece key,
                                        std::vector<TensorSlice>* slices) {
  slices->clear();
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(GetBundleEntryProto(key, &entry));
  slices->reserve(entry.slices_size());
  for (const auto& slice : entry.slices()) {
    slices->emplace_back(slice);
  }
  return OkStatus();
}

Status BundleReader::LookupSlice(StringPiece full_tensor_key,
                                 const TensorSlice& slice_spec, Tensor* val) {
  CHECK(val != nullptr);
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(GetBundleEntryProto(full_tensor_key, &entry));
  return GetSliceValue(full_tensor_key, entry, slice_spec, val);
}

Status BundleReader::GetSliceValue(StringPiece full_tensor_key,
                                   const BundleEntryProto& full_tensor_entry,
                                   const TensorSlice& slice_spec, Tensor* val) {
  using checkpoint::RegisterTensorSlice;
  using checkpoint::TensorSliceSet;
  DCHECK_GE(full_tensor_entry.slices_size(), 0);

  const TensorShape full_shape(TensorShape(full_tensor_entry.shape()));
  std::vector<std::pair<TensorSlice, string>> details;
  const string full_tensor_key_string(full_tensor_key);
  const TensorSliceSet* tss =
      gtl::FindPtrOrNull(tensor_slices_, full_tensor_key_string);

  // Populates the "full tensor key -> TensorSliceSet" cache.
  if (tss == nullptr) {
    if (full_tensor_entry.slices().empty()) {
      // Special case: a writer has saved a tensor fully, but the reader wants
      // to read in slices.  We therefore register the full slice on-demand here
      // without further complicating the on-disk bundle format.
      TF_RETURN_IF_ERROR(RegisterTensorSlice(
          full_tensor_key_string, full_shape, full_tensor_entry.dtype(),
          /* tag */ "",
          /* full slice */ TensorSlice(full_shape.dims()), &tensor_slices_));
    }
    for (const TensorSliceProto& slice : full_tensor_entry.slices()) {
      TF_RETURN_IF_ERROR(RegisterTensorSlice(
          full_tensor_key_string, full_shape, full_tensor_entry.dtype(),
          /* tag */ "", TensorSlice(slice), &tensor_slices_));
    }
    tss = gtl::FindPtrOrNull(tensor_slices_, full_tensor_key_string);
    CHECK_NE(tss, nullptr);
  }
  if (!tss->QueryMeta(slice_spec, &details)) {
    return errors::InvalidArgument(
        "Does not have sufficient slices for partitioned tensor ",
        full_tensor_key,
        " to restore in slice_spec: ", slice_spec.DebugString());
  }

  // The union of the slices in "details" covers "slice_spec".  Performs the
  // copies from each.
  BundleEntryProto stored_slice_entry = full_tensor_entry;
  for (const auto& slice_tag_pair : details) {
    // Seeks for the stored slice.
    const TensorSlice& stored_slice = slice_tag_pair.first;

    // We already have the entry for the full tensor, so don't query again if
    // the slice is full.
    if (!stored_slice.IsFull()) {
      const string encoded_stored_slice_name =
          checkpoint::EncodeTensorNameSlice(full_tensor_key_string,
                                            stored_slice);
      status_ =
          GetBundleEntryProto(encoded_stored_slice_name, &stored_slice_entry);
      if (!status_.ok()) return status_;
    }

    // TODO(zongheng): should we take an OpKernelContext, so that we can call
    // allocate_temp()?  Note that without major refactorings to Saver, it's
    // hard for the caller of the tensor bundle module to allocate these
    // precisely-shaped scratch storage.

    // Optimization for the common case: the stored slice can be directly
    // copied to the destination without additional slicing. This is true when
    // either the slices are equal or when they are both full slices having the
    // same shape.
    TensorShape stored_slice_shape(stored_slice_entry.shape());
    if (stored_slice == slice_spec ||
        (stored_slice_shape == val->shape() &&
         IsFullSlice(stored_slice, stored_slice_shape) &&
         IsFullSlice(slice_spec, stored_slice_shape))) {
      VLOG(1) << "Optimized for common case: directly copying into "
                 "pre-allocated buffer; spec: "
              << slice_spec.DebugString();
      status_ = GetValue(stored_slice_entry, val);
      return status_;
    }

    Tensor stored_slice_tensor(stored_slice_entry.dtype(), stored_slice_shape);
    status_ = GetValue(stored_slice_entry, &stored_slice_tensor);
    if (!status_.ok()) return status_;

    // Copies the intersection over.
    const DataType common_dtype = full_tensor_entry.dtype();
    switch (common_dtype) {
#define HANDLE_COPY(T)                                                 \
  case DataTypeToEnum<T>::value:                                       \
    CHECK(CopyDataFromTensorSliceToTensorSlice(                        \
        full_shape, stored_slice, slice_spec,                          \
        stored_slice_tensor.flat<T>().data(), val->flat<T>().data())); \
    break;

      HANDLE_COPY(float)
      HANDLE_COPY(double)
      HANDLE_COPY(int32)
      HANDLE_COPY(uint8)
      HANDLE_COPY(int16)
      HANDLE_COPY(int8)
      HANDLE_COPY(complex64)
      HANDLE_COPY(complex128)
      HANDLE_COPY(int64_t)
      HANDLE_COPY(bool)
      HANDLE_COPY(qint32)
      HANDLE_COPY(quint8)
      HANDLE_COPY(qint8)
      HANDLE_COPY(bfloat16)
      default:
        return errors::InvalidArgument("Dtype ", DataTypeString(common_dtype),
                                       " not supported.");
    }
#undef HANDLE_COPY
  }
  return OkStatus();
}

bool BundleReader::Contains(StringPiece key) {
  Seek(key);
  return Valid() && (this->key() == key);
}

Status BundleReader::LookupDtypeAndShape(StringPiece key, DataType* dtype,
                                         TensorShape* shape) {
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(GetBundleEntryProto(key, &entry));
  *dtype = entry.dtype();
  *shape = TensorShape(entry.shape());
  return OkStatus();
}

Status BundleReader::LookupTensorShape(StringPiece key, TensorShape* shape) {
  DataType ignored;
  return LookupDtypeAndShape(key, &ignored, shape);
}

string BundleReader::DebugString() {
  // Format used below emulates that of TensorSliceReader::DebugString().
  string shape_str;
  BundleEntryProto entry;
  Seek(kHeaderEntryKey);
  for (Next(); Valid(); Next()) {
    CHECK(entry.ParseFromArray(value().data(), value().size()));
    if (entry.slices_size() > 0) continue;  // Slice of some partitioned var.

    strings::StrAppend(&shape_str, key(), " (", DataType_Name(entry.dtype()),
                       ") ", TensorShape(entry.shape()).DebugString());
    strings::StrAppend(&shape_str, "\n");
  }
  return shape_str;
}

namespace {
inline char* AlignedMalloc(size_t size) {
  char* buffer = static_cast<char*>(port::AlignedMalloc(size, 64));
  DCHECK(buffer);
  return buffer;
}
}  // namespace

FileOutputBuffer::FileOutputBuffer(WritableFile* file, size_t buffer_size)
    : file_(file), position_(0), buffer_size_(buffer_size) {
  DCHECK_GT(buffer_size, 0);
  buffer_ptr_ = AlignedMalloc(buffer_size);
}

FileOutputBuffer::~FileOutputBuffer() {
  if (buffer_ptr_) port::AlignedFree(buffer_ptr_);
  delete file_;
}

Status FileOutputBuffer::Append(StringPiece data) {
  // In the below, it is critical to calculate the checksum on the actually
  // copied bytes, not the source bytes.  This is because "data" typically
  // points to tensor buffers, which may be concurrently written.
  if (data.size() + position_ <= buffer_size_) {
    // Can fit into the current buffer.
    memcpy(buffer_ptr_ + position_, data.data(), data.size());
    crc32c_ = crc32c::Extend(crc32c_, buffer_ptr_ + position_, data.size());
  } else if (data.size() <= buffer_size_) {
    // Cannot fit, but can fit after flushing.
    TF_RETURN_IF_ERROR(FlushBuffer(false));
    memcpy(buffer_ptr_, data.data(), data.size());
    crc32c_ = crc32c::Extend(crc32c_, buffer_ptr_, data.size());
  } else {
    // Cannot fit even after flushing.  So we break down "data" by chunk, and
    // flush/checksum each chunk.
    TF_RETURN_IF_ERROR(FlushBuffer(false));
    for (size_t i = 0; i < data.size(); i += buffer_size_) {
      const size_t nbytes = std::min(data.size() - i, buffer_size_);
      memcpy(buffer_ptr_, data.data() + i, nbytes);
      crc32c_ = crc32c::Extend(crc32c_, buffer_ptr_, nbytes);
      position_ = nbytes;
      TF_RETURN_IF_ERROR(FlushBuffer(false));
    }
    return OkStatus();
  }
  position_ += data.size();
  return OkStatus();
}

Status FileOutputBuffer::Close() {
  TF_RETURN_IF_ERROR(FlushBuffer(true));
  return file_->Close();
}

Status FileOutputBuffer::FlushBuffer(bool closing) {
  if (position_ > 0) {
    // Use Cord to avoid extra data copy for some WritableFile implementations.
    absl::Cord buffer = absl::MakeCordFromExternal(
        StringPiece(buffer_ptr_, position_),
        [ptr = buffer_ptr_](StringPiece) { port::AlignedFree(ptr); });
    buffer_ptr_ = closing ? nullptr : AlignedMalloc(buffer_size_);
    TF_RETURN_IF_ERROR(file_->Append(buffer));
    position_ = 0;
  }
  return OkStatus();
}

}  // namespace tensorflow
