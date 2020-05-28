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

#include <queue>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/snappy/snappy_inputbuffer.h"
#include "tensorflow/core/lib/io/snappy/snappy_outputbuffer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/data/experimental/snapshot.pb.h"

namespace tensorflow {
namespace data {
namespace snapshot_util {

/* static */ constexpr const int64 Reader::kSnappyReaderInputBufferSizeBytes;
/* static */ constexpr const int64 Reader::kSnappyReaderOutputBufferSizeBytes;

std::string GetCurrentCheckpointFile(const std::string& shard_directory,
                                     const uint64 current_checkpoint_id) {
  return io::JoinPath(shard_directory,
                      absl::StrFormat("%08d.snapshot", current_checkpoint_id));
}

Writer::Writer(const std::string& filename, const std::string& compression_type,
               int version, const DataTypeVector& dtypes)
    : filename_(filename),
      compression_type_(compression_type),
      version_(version),
      dtypes_(dtypes) {}

Status Writer::Create(Env* env, const std::string& filename,
                      const std::string& compression_type, int version,
                      const DataTypeVector& dtypes,
                      std::unique_ptr<Writer>* out_writer) {
  *out_writer =
      absl::WrapUnique(new Writer(filename, compression_type, version, dtypes));

  return (*out_writer)->Initialize(env);
}

Status Writer::Initialize(tensorflow::Env* env) {
  TF_RETURN_IF_ERROR(env->NewAppendableFile(filename_, &dest_));
#if defined(IS_SLIM_BUILD)
  if (compression_type_ != io::compression::kNone) {
    LOG(ERROR) << "Compression is unsupported on mobile platforms. Turning "
               << "off compression.";
  }
#else   // IS_SLIM_BUILD
  if (compression_type_ == io::compression::kGzip) {
    zlib_underlying_dest_.swap(dest_);
    io::ZlibCompressionOptions zlib_options;
    zlib_options = io::ZlibCompressionOptions::GZIP();

    io::ZlibOutputBuffer* zlib_output_buffer = new io::ZlibOutputBuffer(
        zlib_underlying_dest_.get(), zlib_options.input_buffer_size,
        zlib_options.output_buffer_size, zlib_options);
    TF_CHECK_OK(zlib_output_buffer->Init());
    dest_.reset(zlib_output_buffer);
  }
#endif  // IS_SLIM_BUILD
  simple_tensor_mask_.reserve(dtypes_.size());
  for (const auto& dtype : dtypes_) {
    if (DataTypeCanUseMemcpy(dtype)) {
      simple_tensor_mask_.push_back(true);
      num_simple_++;
    } else {
      simple_tensor_mask_.push_back(false);
      num_complex_++;
    }
  }

  return Status::OK();
}

Status Writer::WriteTensors(const std::vector<Tensor>& tensors) {
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
  experimental::SnapshotTensorMetadata metadata;
  int64 total_size = 0;
  for (int i = 0; i < tensors.size(); ++i) {
    const Tensor& tensor = tensors[i];
    experimental::TensorMetadata* tensor_metadata =
        metadata.add_tensor_metadata();
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

Status Writer::Sync() { return dest_->Sync(); }

Status Writer::Close() {
  if (dest_ != nullptr) {
    TF_RETURN_IF_ERROR(dest_->Close());
    dest_ = nullptr;
  }
  if (zlib_underlying_dest_ != nullptr) {
    TF_RETURN_IF_ERROR(zlib_underlying_dest_->Close());
    zlib_underlying_dest_ = nullptr;
  }
  return Status::OK();
}

Writer::~Writer() {
  Status s = Close();
  if (!s.ok()) {
    LOG(ERROR) << "Could not finish writing file: " << s;
  }
}

Status Writer::WriteRecord(const StringPiece& data) {
  char header[kHeaderSize];
  core::EncodeFixed64(header, data.size());
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  return dest_->Append(data);
}

#if defined(PLATFORM_GOOGLE)
Status Writer::WriteRecord(const absl::Cord& data) {
  char header[kHeaderSize];
  core::EncodeFixed64(header, data.size());
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  return dest_->Append(data);
}
#endif  // PLATFORM_GOOGLE

Status Reader::Create(Env* env, const std::string& filename,
                      const string& compression_type, int version,
                      const DataTypeVector& dtypes,
                      std::unique_ptr<Reader>* out_reader) {
  *out_reader =
      absl::WrapUnique(new Reader(filename, compression_type, version, dtypes));

  return (*out_reader)->Initialize(env);
}

class Reader::Dataset : public DatasetBase {
 public:
  explicit Dataset(const std::string& shard_dir, const std::string& compression,
                   const int64 version, const DataTypeVector& dtypes,
                   const std::vector<PartialTensorShape>& shapes,
                   const int64 start_index, DatasetContext::Params params)
      : DatasetBase(DatasetContext(std::move(params))),
        shard_dir_(shard_dir),
        compression_(compression),
        version_(version),
        dtypes_(dtypes),
        shapes_(shapes),
        start_index_(start_index) {}

  const DataTypeVector& output_dtypes() const override { return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return shapes_;
  }

  std::string DebugString() const override {
    return "snapshot_util::Reader::Dataset";
  }

  Status CheckExternalState() const override { return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** node) const override {
    // Not necessary perform any serialization as this dataset is only
    // constructed at runtime in C++ and will be reconstructed every time.
    return Status::OK();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(node_name(), prefix)});
  }

 private:
  const std::string shard_dir_;
  const std::string compression_;
  const int64 version_;
  const DataTypeVector dtypes_;
  const std::vector<PartialTensorShape> shapes_;
  const int64 start_index_;

  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params), current_checkpoint_id_(0) {}

    Status Initialize(IteratorContext* ctx) override {
      TF_RETURN_IF_ERROR(Reader::Create(
          ctx->env(), GetCurrentFilename(), dataset()->compression_,
          dataset()->version_, dataset()->dtypes_, &reader_));
      bool end_of_sequence;
      for (int64 i = 0; i < dataset()->start_index_; ++i) {
        // TODO(frankchn): Optimize this to not parse every single element.
        std::vector<Tensor> unused;
        TF_RETURN_IF_ERROR(GetNextInternal(ctx, &unused, &end_of_sequence));
      }
      return Status::OK();
    }

   protected:
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      *end_of_sequence = false;
      Status s = reader_->ReadTensors(out_tensors);
      if (!errors::IsOutOfRange(s)) {
        return s;
      }
      Status status = AdvanceToNextFile(ctx->env());
      if (errors::IsNotFound(status)) {
        *end_of_sequence = true;
        return Status::OK();
      } else {
        return status;
      }
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      // Not necessary to save any state as this iterator will be reconstructed
      // from scratch when the parent snapshot dataset is restored from
      // checkpoint.
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      // Not necessary to restore any state as this iterator will be
      // reconstructed from scratch when the parent snapshot dataset is restored
      // from checkpoint.
      return Status::OK();
    }

   private:
    std::unique_ptr<Reader> reader_;

    // Stores the id current checkpoint file that we are in the process of
    // reading (e.g. if the file is currently 00000001.snapshot, then this will
    // be 1).
    uint64 current_checkpoint_id_;

    std::string GetCurrentFilename() {
      return GetCurrentCheckpointFile(dataset()->shard_dir_,
                                      current_checkpoint_id_);
    }

    Status AdvanceToNextFile(Env* env) {
      current_checkpoint_id_++;
      TF_RETURN_IF_ERROR(env->FileExists(GetCurrentFilename()));
      return Reader::Create(env, GetCurrentFilename(), dataset()->compression_,
                            dataset()->version_, dataset()->dtypes_, &reader_);
    }
  };
};

class Reader::NestedDataset : public DatasetBase {
 public:
  explicit NestedDataset(std::vector<DatasetBase*> datasets,
                         DatasetContext::Params params)
      : DatasetBase(DatasetContext(std::move(params))), datasets_(datasets) {
    dtypes_.push_back(DT_VARIANT);
    gtl::InlinedVector<int64, 1> element_dim_sizes;
    element_dim_sizes.push_back(1);
    partial_shapes_.emplace_back(element_dim_sizes);
  }

  const DataTypeVector& output_dtypes() const override { return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return partial_shapes_;
  }

  std::string DebugString() const override {
    return "snapshot_util::Reader::NestedDataset";
  }

  Status CheckExternalState() const override { return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** node) const override {
    // Not necessary perform any serialization as this dataset is only
    // constructed at runtime in C++ and will be reconstructed every time.
    return Status::OK();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(node_name(), prefix)});
  }

 private:
  std::vector<DatasetBase*> datasets_;
  DataTypeVector dtypes_;
  std::vector<PartialTensorShape> partial_shapes_;

  class Iterator : public DatasetIterator<NestedDataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<NestedDataset>(params), index_(0) {}

   protected:
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      *end_of_sequence = dataset()->datasets_.size() == index_;
      if (!*end_of_sequence) {
        Tensor tensor(DT_VARIANT, TensorShape({}));

        TF_RETURN_IF_ERROR(
            StoreDatasetInVariantTensor(dataset()->datasets_[index_], &tensor));
        out_tensors->clear();
        out_tensors->push_back(std::move(tensor));

        index_++;
      }
      return Status::OK();
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      // Not necessary to save any state as this iterator will be reconstructed
      // from scratch when the parent snapshot dataset is restored from
      // checkpoint.
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      // Not necessary to restore any state as this iterator will be
      // reconstructed from scratch when the parent snapshot dataset is restored
      // from checkpoint.
      return Status::OK();
    }

   private:
    int64 index_;
  };
};

Status Reader::MakeNestedDataset(Env* env,
                                 const std::vector<std::string>& shard_dirs,
                                 const string& compression_type, int version,
                                 const DataTypeVector& dtypes,
                                 const std::vector<PartialTensorShape>& shapes,
                                 const int64 start_index,
                                 DatasetBase** output) {
  std::vector<DatasetBase*> datasets;

  datasets.reserve(shard_dirs.size());
  for (const auto& shard_dir : shard_dirs) {
    // TODO(frankchn): The reading pattern could be controlled in a non-round
    // robin fashion, so we cannot assume a round-robin manner when restoring.
    int64 dataset_start_index = start_index / shard_dirs.size();
    if (start_index % shard_dirs.size() > datasets.size()) {
      dataset_start_index++;
    }

    datasets.push_back(
        new Dataset(shard_dir, compression_type, version, dtypes, shapes,
                    dataset_start_index,
                    DatasetContext::Params({"snapshot_util::Reader::Dataset",
                                            "snapshot_util_reader_Dataset"})));
  }

  // Rotate the vector such that the first dataset contains the next element
  // to be produced.
  std::rotate(datasets.begin(),
              datasets.begin() + (start_index % shard_dirs.size()),
              datasets.end());

  *output = new NestedDataset(
      datasets, DatasetContext::Params({"snapshot_util::Reader::NestedDataset",
                                        "snapshot_util_reader_NestedDataset"}));
  return Status::OK();
}

Reader::Reader(const std::string& filename, const string& compression_type,
               int version, const DataTypeVector& dtypes)
    : filename_(filename),
      compression_type_(compression_type),
      version_(version),
      dtypes_(dtypes) {}

Status Reader::Initialize(Env* env) {
  TF_RETURN_IF_ERROR(Env::Default()->NewRandomAccessFile(filename_, &file_));
  input_stream_ = std::make_unique<io::RandomAccessInputStream>(file_.get());

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
          file_.get(), /*input_buffer_bytes=*/kSnappyReaderInputBufferSizeBytes,
          /*output_buffer_bytes=*/kSnappyReaderOutputBufferSizeBytes);
    } else {
      input_stream_ =
          absl::make_unique<io::BufferedInputStream>(file_.get(), 64 << 20);
    }
  }
#endif  // IS_SLIM_BUILD
  simple_tensor_mask_.reserve(dtypes_.size());
  for (const auto& dtype : dtypes_) {
    if (DataTypeCanUseMemcpy(dtype)) {
      simple_tensor_mask_.push_back(true);
      num_simple_++;
    } else {
      simple_tensor_mask_.push_back(false);
      num_complex_++;
    }
  }

  return Status::OK();
}

Status Reader::SkipRecords(int64 num_records) {
  // TODO(frankchn): Optimize to not parse the entire Tensor and actually skip.
  for (int i = 0; i < num_records; ++i) {
    std::vector<Tensor> unused_tensors;
    TF_RETURN_IF_ERROR(ReadTensors(&unused_tensors));
  }
  return Status::OK();
}

Status Reader::ReadTensors(std::vector<Tensor>* read_tensors) {
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

  experimental::SnapshotTensorMetadata metadata;
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
      absl::string_view tensor_proto_view(tensor_proto_str.get(),
                                          tensor_proto_size);
      absl::Cord c = absl::MakeCordFromExternal(
          tensor_proto_view, [s = std::move(tensor_proto_str)] {});
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

Status Reader::ReadTensorsV0(std::vector<Tensor>* read_tensors) {
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

Status Reader::SnappyUncompress(
    const experimental::SnapshotTensorMetadata* metadata,
    std::vector<Tensor>* simple_tensors,
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

Status Reader::ReadRecord(tstring* record) {
  tstring header;
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
  uint64 length = core::DecodeFixed64(header.data());
  return input_stream_->ReadNBytes(length, record);
}

#if defined(PLATFORM_GOOGLE)
Status Reader::ReadRecord(absl::Cord* record) {
  tstring header;
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
  uint64 length = core::DecodeFixed64(header.data());
  if (compression_type_ == io::compression::kNone) {
    return input_stream_->ReadNBytes(length, record);
  } else {
    auto tmp_str = absl::make_unique<tstring>();
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(length, tmp_str.get()));
    absl::string_view tmp_str_view(*tmp_str);
    record->Append(
        absl::MakeCordFromExternal(tmp_str_view, [s = std::move(tmp_str)] {}));
    return Status::OK();
  }
}
#endif

Status WriteMetadataFile(const string& hash_dir,
                         const experimental::SnapshotMetadataRecord* metadata) {
  string metadata_filename = io::JoinPath(hash_dir, kMetadataFilename);
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(hash_dir));
  std::string tmp_filename =
      absl::StrCat(metadata_filename, "-tmp-", random::New64());
  TF_RETURN_IF_ERROR(WriteBinaryProto(Env::Default(), tmp_filename, *metadata));
  return Env::Default()->RenameFile(tmp_filename, metadata_filename);
}

Status ReadMetadataFile(const string& hash_dir,
                        experimental::SnapshotMetadataRecord* metadata,
                        bool* file_exists) {
  string metadata_filename = io::JoinPath(hash_dir, kMetadataFilename);
  Status s = Env::Default()->FileExists(metadata_filename);
  *file_exists = s.ok();

  if (*file_exists) {
    return ReadBinaryProto(Env::Default(), metadata_filename, metadata);
  } else {
    return Status::OK();
  }
}

Status DumpDatasetGraph(const std::string& path, uint64 hash,
                        const GraphDef* graph) {
  std::string hash_hex =
      strings::StrCat(strings::Hex(hash, strings::kZeroPad16));
  std::string graph_file =
      io::JoinPath(path, absl::StrCat(hash_hex, "-graph.pbtxt"));

  LOG(INFO) << "Graph hash is " << hash_hex << ", writing to " << graph_file;
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(path));
  return WriteTextProto(Env::Default(), graph_file, *graph);
}

Status DetermineOpState(const std::string& mode_string, bool file_exists,
                        const experimental::SnapshotMetadataRecord* metadata,
                        const uint64 pending_snapshot_expiry_seconds,
                        Mode* mode) {
  if (mode_string == kModeRead) {
    // In read mode, we should expect a metadata file is written.
    if (!file_exists) {
      return errors::NotFound("Metadata file does not exist.");
    }
    LOG(INFO) << "Overriding mode to reader.";
    *mode = READER;
    return Status::OK();
  }

  if (mode_string == kModeWrite) {
    LOG(INFO) << "Overriding mode to writer.";
    *mode = WRITER;
    return Status::OK();
  }

  if (mode_string == kModePassthrough) {
    LOG(INFO) << "Overriding mode to passthrough.";
    *mode = PASSTHROUGH;
    return Status::OK();
  }

  if (!file_exists) {
    *mode = WRITER;
    return Status::OK();
  }

  if (metadata->finalized()) {
    // File found, snapshot has been finalized.
    *mode = READER;
    return Status::OK();
  }

  if (metadata->creation_timestamp() >=
      (static_cast<int64>(EnvTime::NowMicros()) -
       pending_snapshot_expiry_seconds * 1000000)) {
    // Someone else is already writing and time has not expired.
    *mode = PASSTHROUGH;
    return Status::OK();
  } else {
    // Time has expired, we write regardless.
    *mode = WRITER;
    return Status::OK();
  }
}

}  // namespace snapshot_util
}  // namespace data
}  // namespace tensorflow
