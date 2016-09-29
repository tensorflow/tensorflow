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
#include <memory>
#include <utility>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb_text.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb_text.h"
#include "tensorflow/core/framework/versions.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_slice_util.h"

namespace tensorflow {

// Versioning of the tensor bundle format.
const int kTensorBundleMinProducer = 0;
const int kTensorBundleMinConsumer = 0;
const int kTensorBundleVersion = 1;

// Key to the special BundleHeaderProto entry.  Do not change this, as clients
// can make the assumption that the header is always the first entry in the
// bundle.
const char* const kHeaderEntryKey = "";

namespace {

// Reads "num_elements" string elements from file[offset, offset+size) into the
// length-N "destination".  Discards the original content of "destination".
//
// Checksums the string lengths (as restored uint32, not varint32 bytes) and
// string bytes, and stores it into "actual_crc32c".
Status ReadStringTensor(io::InputBuffer* buffered_file, size_t num_elements,
                        size_t offset, size_t size, string* destination,
                        uint32* actual_crc32c) {
  if (size == 0) return Status::OK();
  CHECK_GT(size, 0);

  // Reads "num_elements" varint32's from "buffered_file".
  TF_RETURN_IF_ERROR(buffered_file->Seek(offset));
  std::vector<uint32> string_lengths(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    TF_RETURN_IF_ERROR(buffered_file->ReadVarint32(&string_lengths[i]));
  }
  if (offset + size < buffered_file->Tell()) {
    return errors::DataLoss("String lengths longer than expected offset ",
                            offset + size);
  }
  *actual_crc32c =
      crc32c::Value(reinterpret_cast<const char*>(string_lengths.data()),
                    sizeof(uint32) * num_elements);

  // Reads the length-checksum.
  uint32 length_checksum = 0;
  size_t unused_bytes_read = 0;
  TF_RETURN_IF_ERROR(buffered_file->ReadNBytes(
      sizeof(uint32), reinterpret_cast<char*>(&length_checksum),
      &unused_bytes_read));
  if (crc32c::Unmask(length_checksum) != *actual_crc32c) {
    return errors::DataLoss(
        "The length checksum does not match: expected ",
        strings::Printf("%08u", crc32c::Unmask(length_checksum)),
        " but actual is ", strings::Printf("%08u", *actual_crc32c));
  }
  *actual_crc32c =
      crc32c::Extend(*actual_crc32c, reinterpret_cast<char*>(&length_checksum),
                     sizeof(uint32));

  // Reads the actual string bytes.
  for (size_t i = 0; i < num_elements; ++i) {
    const uint32 string_length = string_lengths[i];
    string* buffer = &destination[i];

    buffer->resize(string_length);
    size_t bytes_read = 0;
    TF_RETURN_IF_ERROR(
        buffered_file->ReadNBytes(string_length, &(*buffer)[0], &bytes_read));
    *actual_crc32c = crc32c::Extend(*actual_crc32c, buffer->data(), bytes_read);
  }
  return Status::OK();
}

char* GetBackingBuffer(const Tensor& val) {
  CHECK(DataTypeCanUseMemcpy(val.dtype()));
  return const_cast<char*>(val.tensor_data().data());
}

string* GetStringBackingBuffer(const Tensor& val) {
  CHECK_EQ(DT_STRING, val.dtype());
  return const_cast<string*>(val.flat<string>().data());
}

Status ParseEntryProto(StringPiece key, StringPiece value,
                       protobuf::MessageLite* out) {
  if (!out->ParseFromArray(value.data(), value.size())) {
    return errors::DataLoss("Entry for key ", key, " not parseable.");
  }
  return Status::OK();
}

// Serializes the data bytes of the non-string tensor "val".  Discards the
// original content of "bytes_written", and on OK updates it with number of
// bytes written.
// REQUIRES: val.dtype() != DT_STRING
Status WriteTensor(const Tensor& val, FileOutputBuffer* out,
                   size_t* bytes_written) {
  DCHECK_NE(val.dtype(), DT_STRING);
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
  //   [varint32 len0]..[varint32 lenL][4 byte cksum on lengths][string bytes]
  // Var "crc32c" checksums the string lengths (as uint32, not varint32 bytes),
  // the length-checksum, and all the string bytes.
  DCHECK_EQ(val.dtype(), DT_STRING);
  const string* strings = GetStringBackingBuffer(val);

  // Writes the varint lengths.
  string lengths;
  lengths.reserve(val.NumElements());  // At least 1 byte per element.
  *crc32c = 0;
  for (int64 i = 0; i < val.NumElements(); ++i) {
    const string* elem = &strings[i];
    DCHECK_EQ(elem->size(), static_cast<uint32>(elem->size()));
    const uint32 elem_size = static_cast<uint32>(elem->size());

    core::PutVarint32(&lengths, elem_size);
    *crc32c = crc32c::Extend(*crc32c, reinterpret_cast<const char*>(&elem_size),
                             sizeof(uint32));
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
  for (int64 i = 0; i < val.NumElements(); ++i) {
    const string* string = &strings[i];
    TF_RETURN_IF_ERROR(out->Append(*string));
    *bytes_written += string->size();
    *crc32c = crc32c::Extend(*crc32c, string->data(), string->size());
  }
  return Status::OK();
}

// Reads file[offset:offset+size) into destination[0:size).  Each Read() copies
// at most "buffer_size" bytes.
//
// REQUIRES: "file" contains at least "offset + size" bytes.
// REQUIRES: "destination" contains at least "size" bytes.
// On error, "destination" may contain garbage.
Status ReadInputByChunk(const RandomAccessFile* file, size_t offset,
                        size_t size, size_t buffer_size, char* destination) {
  if (size == 0) return Status::OK();
  CHECK_GT(size, 0);
  CHECK_GT(buffer_size, 0);
  size_t bytes_read = 0;
  StringPiece result;

  while (bytes_read < size) {
    const size_t desired_bytes = std::min(buffer_size, size - bytes_read);
    Status status = file->Read(offset + bytes_read, desired_bytes, &result,
                               destination + bytes_read);

    if (!status.ok()) {
      return status;
    } else if (result.size() != desired_bytes) {
      return errors::DataLoss("Requested ", desired_bytes, " bytes but read ",
                              result.size(), " bytes.");
    } else if (result.data() == destination + bytes_read) {
      // Data is already in the correct location.
    } else {
      // memmove is guaranteed to handle overlaps safely (although the src and
      // dst buffers should not overlap for this function).
      memmove(destination + bytes_read, result.data(), result.size());
    }
    bytes_read += result.size();
  }
  CHECK_EQ(bytes_read, size);
  return Status::OK();
}

}  // namespace

string DataFilename(const string& prefix, int32 shard_id, int32 num_shards) {
  DCHECK_GT(num_shards, 0);
  DCHECK_LT(shard_id, num_shards);
  return strings::Printf("%s.data-%05d-of-%05d", prefix.c_str(), shard_id,
                         num_shards);
}

string MetaFilename(const string& prefix) {
  return strings::Printf("%s.index", prefix.c_str());
}

BundleWriter::BundleWriter(Env* env, const string& prefix)
    : env_(env), prefix_(prefix), out_(nullptr), size_(0) {
  status_ =
      env_->CreateDir(io::Dirname(prefix_).ToString());  // Ignores errors.
  const string filename = DataFilename(prefix_, 0, 1);
  std::unique_ptr<WritableFile> wrapper;
  status_ = env_->NewWritableFile(filename, &wrapper);
  if (!status_.ok()) return;
  out_ = std::unique_ptr<FileOutputBuffer>(
      new FileOutputBuffer(wrapper.release(), 8 << 20 /* 8MB write buffer */));

  VLOG(1) << "Writing to file " << filename;
}

BundleWriter::~BundleWriter() { CHECK(out_ == nullptr); }

Status BundleWriter::Add(const string& key, const Tensor& val) {
  CHECK_NE(key, kHeaderEntryKey);
  if (!status_.ok()) return status_;
  if (entries_.find(key) != entries_.end()) {
    status_ = errors::InvalidArgument("Adding duplicate key: ", key);
    return status_;
  }

  BundleEntryProto* entry = &entries_[key];
  entry->set_dtype(val.dtype());
  val.shape().AsProto(entry->mutable_shape());
  entry->set_shard_id(0);
  entry->set_offset(size_);

  // Updates the data file.
  size_t data_bytes_written = 0;
  uint32 crc32c = 0;
  out_->clear_crc32c();
  if (val.dtype() != DT_STRING) {
    status_ = WriteTensor(val, out_.get(), &data_bytes_written);
    crc32c = out_->crc32c();
  } else {
    status_ = WriteStringTensor(val, out_.get(), &data_bytes_written, &crc32c);
  }

  if (status_.ok()) {
    entry->set_size(data_bytes_written);
    entry->set_crc32c(crc32c::Mask(crc32c));
    size_ += data_bytes_written;
  }
  return status_;
}

Status BundleWriter::AddSlice(const string& full_tensor_key,
                              const TensorShape& full_tensor_shape,
                              const TensorSlice& slice_spec,
                              const Tensor& slice_tensor) {
  CHECK_NE(full_tensor_key, kHeaderEntryKey);
  if (!status_.ok()) return status_;

  // Inserts/updates the full tensor's metadata entry.
  //
  // In the case of a sharded save, MergeBundles() is responsible for merging
  // the "slices" field of multiple metadata entries corresponding to the same
  // full tensor.
  BundleEntryProto* full_entry = &entries_[full_tensor_key];
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
      checkpoint::EncodeTensorNameSlice(full_tensor_key, slice_spec);
  status_ = Add(slice_name, slice_tensor);
  return status_;
}

// TODO(zongheng): on metadata write failure or !status_.ok(), consider removing
// the orphaned data file.
Status BundleWriter::Finish() {
  if (out_) {
    status_.Update(out_->Close());
    out_ = nullptr;
  }
  if (!status_.ok()) return status_;
  // Build key -> BundleEntryProto table.
  std::unique_ptr<WritableFile> file;
  status_ = env_->NewWritableFile(MetaFilename(prefix_), &file);
  if (!status_.ok()) return status_;
  {
    table::TableBuilder builder(table::Options(), file.get());
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
  if (!status_.ok()) return status_;
  status_ = errors::Internal("BundleWriter is closed");
  return Status::OK();
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
static Status MergeOneBundle(Env* env, const string& prefix,
                             MergeState* merge_state) {
  VLOG(1) << "Merging bundle:" << prefix;
  const string& filename = MetaFilename(prefix);
  uint64 file_size;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  table::Table* table = nullptr;
  TF_RETURN_IF_ERROR(
      table::Table::Open(table::Options(), file.get(), file_size, &table));
  std::unique_ptr<table::Table> table_deleter(table);
  std::unique_ptr<table::Iterator> iter(table->NewIterator());

  int num_shards;
  // Process header.
  {
    iter->Seek(kHeaderEntryKey);
    CHECK(iter->Valid());
    BundleHeaderProto header;
    TF_CHECK_OK(ParseEntryProto(iter->key(), iter->value(), &header));
    CHECK_GE(header.num_shards(), 0);

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
    const string key = iter->key().ToString();
    const auto entry_iter = merge_state->entries.find(key);

    // Illegal: the duplicated entry is a non-slice tensor.
    if (entry_iter != merge_state->entries.end() &&
        entry_iter->second.slices().empty()) {
      return errors::InvalidArgument("Duplicate tensor keyed by ", key,
                                     " encountered, when merging prefix: ",
                                     prefix);
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
  return Status::OK();
}

Status MergeBundles(Env* env, gtl::ArraySlice<string> prefixes,
                    const string& merged_prefix) {
  // Merges all metadata tables.
  // TODO(zhifengc): KeyValue sorter if it becomes too big.
  MergeState merge;
  env->CreateDir(io::Dirname(merged_prefix).ToString());  // Ignores errors.
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
  Status status;
  {
    table::TableBuilder builder(table::Options(), merged_metadata.get());
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
  for (const string& prefix : prefixes) {
    env->DeleteFile(MetaFilename(prefix));
  }
  return status;
}

// Interface for reading a tensor bundle.

BundleReader::BundleReader(Env* env, const string& prefix)
    : env_(env),
      prefix_(prefix),
      metadata_(nullptr),
      table_(nullptr),
      iter_(nullptr) {
  const string& filename = MetaFilename(prefix_);
  uint64 file_size;
  status_ = env_->GetFileSize(filename, &file_size);
  if (!status_.ok()) return;

  // Opens the metadata table.
  std::unique_ptr<RandomAccessFile> wrapper;
  status_ = env_->NewRandomAccessFile(filename, &wrapper);
  if (!status_.ok()) return;
  metadata_ = wrapper.release();
  status_ = table::Table::Open(table::Options(), metadata_, file_size, &table_);
  if (!status_.ok()) return;
  iter_ = table_->NewIterator();

  // Reads "num_shards_" from the first entry.
  iter_->Seek(kHeaderEntryKey);
  CHECK(iter_->Valid());
  BundleHeaderProto header;
  TF_CHECK_OK(ParseEntryProto(iter_->key(), iter_->value(), &header));
  num_shards_ = header.num_shards();
  if ((header.endianness() == BundleHeaderProto::BIG && port::kLittleEndian) ||
      (header.endianness() == BundleHeaderProto::LITTLE &&
       !port::kLittleEndian)) {
    status_ = errors::Unimplemented(
        "Reading a bundle with different endianness from the reader");
    return;
  }
  status_ = CheckVersions(header.version(), kTensorBundleVersion,
                          kTensorBundleMinProducer, "Checkpoint", "checkpoint");
}

BundleReader::~BundleReader() {
  delete metadata_;
  delete iter_;
  delete table_;
  gtl::STLDeleteValues(&data_);
  gtl::STLDeleteValues(&tensor_slices_);
}

Status BundleReader::GetBundleEntryProto(const string& key,
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
    return errors::DataLoss("Invaid tensor shape: ", key, " ",
                            ProtoShortDebugString(entry_copy.shape()));
  }

  *entry = entry_copy;
  return Status::OK();
}

Status BundleReader::GetValue(const BundleEntryProto& entry, Tensor* val) {
  Tensor* ret = val;
  const TensorShape stored_shape(TensorShape(entry.shape()));
  if (val->NumElements() == 0) {
    ret = new Tensor(entry.dtype(), stored_shape);
  }

  // Validates the "size" field.
  if (entry.dtype() != DT_STRING) {
    if (entry.size() != ret->TotalBytes()) {
      return errors::DataLoss("Invalid size in bundle entry: key ", key(),
                              "; stored size ", entry.size(),
                              "; expected size ", ret->TotalBytes());
    }
  } else {
    // Relaxes the check for string tensors as follows:
    //   entry.size() == bytes(varint lengths) + bytes(data)
    //                >= NumElems + bytes(data), since size bytes(varint) >= 1.
    //   TotalBytes() == sizeof(string) * NumElems + bytes(data)
    // Since we don't know bytes(varint lengths), we just check an inequality.
    const size_t lower_bound = ret->NumElements() + ret->TotalBytes() -
                               sizeof(string) * ret->NumElements();
    if (entry.size() < lower_bound) {
      return errors::DataLoss("Invalid size in bundle entry: key ", key(),
                              "; stored size ", entry.size(),
                              "; expected size is at least ", lower_bound);
    }
  }

  // Open the data file if not opened it.
  std::unique_ptr<RandomAccessFile> file = nullptr;
  std::unique_ptr<io::InputBuffer> buffered_file(data_[entry.shard_id()]);
  if (buffered_file == nullptr) {
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(
        DataFilename(prefix_, entry.shard_id(), num_shards_), &file));
    buffered_file.reset(
        new io::InputBuffer(file.get(), 256 << 10 /* 256KB buffer */));
  }
  CHECK(buffered_file != nullptr);

  TF_RETURN_IF_ERROR(buffered_file->Seek(entry.offset()));
  uint32 actual_crc32c = 0;
  if (DataTypeCanUseMemcpy(entry.dtype())) {
    // Important: ReadInputByChunk() bounds the readahead as min(buffer, actual
    // bytes needed).  This is critical when reading small tensors, so we don't
    // rely on io::InputBuffer's blind buffering here.
    char* backing_buffer = const_cast<char*>((ret->tensor_data().data()));
    TF_RETURN_IF_ERROR(ReadInputByChunk(buffered_file->file(), entry.offset(),
                                        entry.size(), 8 << 20 /* 8MB buffer */,
                                        backing_buffer));
    actual_crc32c = crc32c::Value(backing_buffer, entry.size());
  } else {
    // Relies on io::InputBuffer's buffering, because we issue many neighboring
    // reads for a single string tensor.
    TF_RETURN_IF_ERROR(ReadStringTensor(
        buffered_file.get(), ret->NumElements(), entry.offset(), entry.size(),
        GetStringBackingBuffer(*ret), &actual_crc32c));
  }
  if (crc32c::Unmask(entry.crc32c()) != actual_crc32c) {
    return errors::DataLoss(
        "Checksum does not match: stored ",
        strings::Printf("%08u", crc32c::Unmask(entry.crc32c())),
        " vs. calculated on the restored bytes ", actual_crc32c);
  }

  *val = *ret;
  if (ret != val) delete ret;
  return Status::OK();
}

Status BundleReader::Lookup(const string& key, Tensor* val) {
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

Status BundleReader::LookupSlice(const string& full_tensor_key,
                                 const TensorSlice& slice_spec, Tensor* val) {
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(GetBundleEntryProto(full_tensor_key, &entry));
  return GetSliceValue(full_tensor_key, entry, slice_spec, val);
}

Status BundleReader::GetSliceValue(const string& full_tensor_key,
                                   const BundleEntryProto& full_tensor_entry,
                                   const TensorSlice& slice_spec, Tensor* val) {
  using checkpoint::TensorSliceSet;
  using checkpoint::RegisterTensorSlice;
  DCHECK_GE(full_tensor_entry.slices_size(), 0);

  const TensorShape full_shape(TensorShape(full_tensor_entry.shape()));
  std::vector<std::pair<TensorSlice, string>> details;
  const TensorSliceSet* tss =
      gtl::FindPtrOrNull(tensor_slices_, full_tensor_key);

  // Populates the "full tensor key -> TensorSliceSet" cache.
  if (tss == nullptr) {
    if (full_tensor_entry.slices().empty()) {
      // Special case: a writer has saved a tensor fully, but the reader wants
      // to read in slices.  We therefore register the full slice on-demand here
      // without further complicating the on-disk bundle format.
      RegisterTensorSlice(
          full_tensor_key, full_shape, full_tensor_entry.dtype(), /* tag */ "",
          /* full slice */ TensorSlice(full_shape.dims()), &tensor_slices_);
    }
    for (const TensorSliceProto& slice : full_tensor_entry.slices()) {
      RegisterTensorSlice(full_tensor_key, full_shape,
                          full_tensor_entry.dtype(),
                          /* tag */ "", TensorSlice(slice), &tensor_slices_);
    }
    tss = gtl::FindPtrOrNull(tensor_slices_, full_tensor_key);
    CHECK_NE(tss, nullptr);
  }
  if (!tss->QueryMeta(slice_spec, &details)) {
    return errors::InvalidArgument(
        "Does not have sufficient slices for partitioned tensor ",
        full_tensor_key, " to restore in slice_spec: ",
        slice_spec.DebugString());
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
      const string& encoded_stored_slice_name =
          checkpoint::EncodeTensorNameSlice(full_tensor_key, stored_slice);
      status_ =
          GetBundleEntryProto(encoded_stored_slice_name, &stored_slice_entry);
      if (!status_.ok()) return status_;
    }

    // TODO(zongheng): should we take an OpKernelContext, so that we can call
    // allocate_temp()?  Note that without major refactorings to Saver, it's
    // hard for the caller of the tensor bundle module to allocate these
    // precisely-shaped scratch storage.
    // TODO(zongheng): implement an important optimization: if the stored slice
    // is a subset of the to-restore slice, directly read the stored slice into
    // the latter's already-allocated backing buffer.

    // Optimization for the common case: stored slice == to-restore slice.
    if (stored_slice == slice_spec) {
      VLOG(1) << "Optimized for common case: directly copying into "
                 "pre-allocated buffer; spec: "
              << slice_spec.DebugString();
      status_ = GetValue(stored_slice_entry, val);
      return status_;
    }

    Tensor stored_slice_tensor(stored_slice_entry.dtype(),
                               TensorShape(stored_slice_entry.shape()));
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
      HANDLE_COPY(int64)
      HANDLE_COPY(bool)
      HANDLE_COPY(qint32)
      HANDLE_COPY(quint8)
      HANDLE_COPY(qint8)
      default:
        return errors::InvalidArgument("Dtype ", DataTypeString(common_dtype),
                                       " not supported.");
    }
#undef HANDLE_COPY
  }
  return Status::OK();
}

bool BundleReader::Contains(StringPiece key) {
  Seek(key.ToString());
  return Valid() && (this->key() == key);
}

Status BundleReader::LookupTensorShape(const string& key, TensorShape* shape) {
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(GetBundleEntryProto(key, &entry));

  *shape = TensorShape(entry.shape());
  return Status::OK();
}

string BundleReader::DebugString() {
  // Format used below emulates that of TensorSliceReader::DebugString().
  string shape_str;
  BundleEntryProto entry;
  Seek(kHeaderEntryKey);
  for (Next(); Valid(); Next()) {
    CHECK(entry.ParseFromArray(value().data(), value().size()));
    if (entry.slices_size() > 0) continue;  // Slice of some partitioned var.

    strings::StrAppend(&shape_str, key(), " (",
                       EnumName_DataType(entry.dtype()), ") ",
                       TensorShape(entry.shape()).DebugString());
    strings::StrAppend(&shape_str, "\n");
  }
  return shape_str;
}

FileOutputBuffer::~FileOutputBuffer() { delete file_; }

Status FileOutputBuffer::Append(StringPiece data) {
  // In the below, it is critical to calculate the checksum on the actually
  // copied bytes, not the source bytes.  This is because "data" typically
  // points to tensor buffers, which may be concurrently written.
  if (data.size() + position_ <= buffer_size_) {
    // Can fit into the current buffer.
    memcpy(&buffer_[position_], data.data(), data.size());
    crc32c_ = crc32c::Extend(crc32c_, &buffer_[position_], data.size());
  } else if (data.size() <= buffer_size_) {
    // Cannot fit, but can fit after flushing.
    TF_RETURN_IF_ERROR(Flush());
    memcpy(&buffer_[0], data.data(), data.size());
    crc32c_ = crc32c::Extend(crc32c_, &buffer_[0], data.size());
  } else {
    // Cannot fit even after flushing.  So we break down "data" by chunk, and
    // flush/checksum each chunk.
    TF_RETURN_IF_ERROR(Flush());
    for (size_t i = 0; i < data.size(); i += buffer_size_) {
      const size_t nbytes = std::min(data.size() - i, buffer_size_);
      memcpy(&buffer_[0], data.data() + i, nbytes);
      crc32c_ = crc32c::Extend(crc32c_, &buffer_[0], nbytes);
      position_ = nbytes;
      TF_RETURN_IF_ERROR(Flush());
    }
    return Status::OK();
  }
  position_ += data.size();
  return Status::OK();
}

Status FileOutputBuffer::Close() {
  TF_RETURN_IF_ERROR(Flush());
  return file_->Close();
}

Status FileOutputBuffer::Flush() {
  if (position_ > 0) {
    TF_RETURN_IF_ERROR(file_->Append(StringPiece(&buffer_[0], position_)));
    position_ = 0;
  }
  return file_->Flush();
}

}  // namespace tensorflow
