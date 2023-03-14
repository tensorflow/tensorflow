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

#include "tensorflow/core/data/snapshot_utils.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/tsl/lib/io/snappy/snappy_inputbuffer.h"
#include "tensorflow/tsl/lib/io/snappy/snappy_outputbuffer.h"

namespace tensorflow {
namespace data {
namespace snapshot_util {
namespace {

constexpr const char* const kOutputTypes = "output_types";
constexpr const char* const kOutputShapes = "output_shapes";
constexpr const char* const kCompression = "compression";
constexpr const char* const kVersion = "version";
constexpr const char* const kCurrentCheckpointID = "current_checkpoint_id";
constexpr const char* const kIndex = "index";
constexpr const char* const kStartIndex = "start_index";

}  // namespace

/* static */ constexpr const int64_t
    CustomReader::kSnappyReaderInputBufferSizeBytes;
/* static */ constexpr const int64_t
    CustomReader::kSnappyReaderOutputBufferSizeBytes;

std::string HashDirectory(const std::string& path, uint64 hash) {
  return io::JoinPath(
      path, strings::Printf("%llu", static_cast<unsigned long long>(hash)));
}

std::string RunDirectory(const std::string& hash_directory, uint64 run_id) {
  return RunDirectory(
      hash_directory,
      strings::Printf("%llu", static_cast<unsigned long long>(run_id)));
}

std::string RunDirectory(const std::string& hash_directory,
                         const std::string& run_id) {
  return io::JoinPath(hash_directory, run_id);
}

std::string ShardDirectory(const std::string& run_directory, int64_t shard_id) {
  return io::JoinPath(
      run_directory,
      strings::Printf("%08llu%s", static_cast<unsigned long long>(shard_id),
                      kShardDirectorySuffix));
}
std::string GetCheckpointFileName(const std::string& shard_directory,
                                  uint64 checkpoint_id) {
  return io::JoinPath(
      shard_directory,
      strings::Printf("%08llu.snapshot",
                      static_cast<unsigned long long>(checkpoint_id)));
}

Status Writer::Create(Env* env, const std::string& filename,
                      const std::string& compression_type, int version,
                      const DataTypeVector& dtypes,
                      std::unique_ptr<Writer>* out_writer) {
  switch (version) {
    case 1:
      *out_writer =
          std::make_unique<CustomWriter>(filename, compression_type, dtypes);
      break;
    case 2:
      *out_writer =
          std::make_unique<TFRecordWriter>(filename, compression_type);
      break;
    default:
      return errors::InvalidArgument("Snapshot writer version: ", version,
                                     " is not supported.");
  }

  return (*out_writer)->Initialize(env);
}

TFRecordWriter::TFRecordWriter(const std::string& filename,
                               const std::string& compression_type)
    : filename_(filename), compression_type_(compression_type) {}

Status TFRecordWriter::Initialize(tensorflow::Env* env) {
  TF_RETURN_IF_ERROR(env->NewAppendableFile(filename_, &dest_));

  record_writer_ = std::make_unique<io::RecordWriter>(
      dest_.get(), io::RecordWriterOptions::CreateRecordWriterOptions(
                       /*compression_type=*/compression_type_));
  return OkStatus();
}

Status TFRecordWriter::WriteTensors(const std::vector<Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    TensorProto proto;
    tensor.AsProtoTensorContent(&proto);
#if defined(TF_CORD_SUPPORT)
    // Creating raw pointer here because std::move() in a releases in OSS TF
    // will result in a smart pointer being moved upon function creation, which
    // will result in proto_buffer == nullptr when WriteRecord happens.
    auto* proto_buffer = new std::string();
    proto.SerializeToString(proto_buffer);
    absl::Cord proto_serialized = absl::MakeCordFromExternal(
        *proto_buffer,
        [proto_buffer](absl::string_view) { delete proto_buffer; });
    TF_RETURN_IF_ERROR(record_writer_->WriteRecord(proto_serialized));
#else   // TF_CORD_SUPPORT
    TF_RETURN_IF_ERROR(record_writer_->WriteRecord(proto.SerializeAsString()));
#endif  // TF_CORD_SUPPORT
  }
  return OkStatus();
}

Status TFRecordWriter::Sync() {
  TF_RETURN_IF_ERROR(record_writer_->Flush());
  return dest_->Flush();
}

Status TFRecordWriter::Close() {
  if (record_writer_ != nullptr) {
    TF_RETURN_IF_ERROR(Sync());
    TF_RETURN_IF_ERROR(record_writer_->Close());
    TF_RETURN_IF_ERROR(dest_->Close());
    record_writer_ = nullptr;
    dest_ = nullptr;
  }
  return OkStatus();
}

TFRecordWriter::~TFRecordWriter() {
  Status s = Close();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to close snapshot file " << filename_ << ": " << s;
  }
}

CustomWriter::CustomWriter(const std::string& filename,
                           const std::string& compression_type,
                           const DataTypeVector& dtypes)
    : filename_(filename),
      compression_type_(compression_type),
      dtypes_(dtypes) {}

Status CustomWriter::Initialize(tensorflow::Env* env) {
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

  return OkStatus();
}

Status CustomWriter::WriteTensors(const std::vector<Tensor>& tensors) {
  if (compression_type_ != io::compression::kSnappy) {
    experimental::SnapshotRecord record;
    for (const auto& tensor : tensors) {
      TensorProto* t = record.add_tensor();
      tensor.AsProtoTensorContent(t);
    }
#if defined(TF_CORD_SUPPORT)
    auto record_buffer = new std::string();
    record.SerializeToString(record_buffer);
    absl::Cord record_serialized = absl::MakeCordFromExternal(
        *record_buffer,
        [record_buffer](absl::string_view) { delete record_buffer; });
    return WriteRecord(record_serialized);
#else   // TF_CORD_SUPPORT
    return WriteRecord(record.SerializeAsString());
#endif  // TF_CORD_SUPPORT
  }

  std::vector<const TensorBuffer*> tensor_buffers;
  tensor_buffers.reserve(num_simple_);
  std::vector<TensorProto> tensor_protos;
  tensor_protos.reserve(num_complex_);
  experimental::SnapshotTensorMetadata metadata;
  int64_t total_size = 0;
  for (int i = 0, end = tensors.size(); i < end; ++i) {
    const Tensor& tensor = tensors[i];
    experimental::TensorMetadata* tensor_metadata =
        metadata.add_tensor_metadata();
    tensor.shape().AsProto(tensor_metadata->mutable_tensor_shape());
    int64_t size = 0;
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
  for (int i = 0, end = tensors.size(); i < end; ++i) {
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
  if (!tsl::port::Snappy_Compress(uncompressed.data(), total_size, &output)) {
    return errors::Internal("Failed to compress using snappy.");
  }

#if defined(TF_CORD_SUPPORT)
  auto metadata_buffer = new std::string();
  metadata.SerializeToString(metadata_buffer);
  absl::Cord metadata_serialized = absl::MakeCordFromExternal(
      *metadata_buffer,
      [metadata_buffer](absl::string_view) { delete metadata_buffer; });
#else
  std::string metadata_serialized = metadata.SerializeAsString();
#endif  // TF_CORD_SUPPORT
  TF_RETURN_IF_ERROR(WriteRecord(metadata_serialized));
  TF_RETURN_IF_ERROR(WriteRecord(output));
  return OkStatus();
}

Status CustomWriter::Sync() { return dest_->Sync(); }

Status CustomWriter::Close() {
  if (dest_ != nullptr) {
    TF_RETURN_IF_ERROR(dest_->Close());
    dest_ = nullptr;
  }
  if (zlib_underlying_dest_ != nullptr) {
    TF_RETURN_IF_ERROR(zlib_underlying_dest_->Close());
    zlib_underlying_dest_ = nullptr;
  }
  return OkStatus();
}

CustomWriter::~CustomWriter() {
  Status s = Close();
  if (!s.ok()) {
    LOG(ERROR) << "Could not finish writing file: " << s;
  }
}

Status CustomWriter::WriteRecord(const StringPiece& data) {
  char header[kHeaderSize];
  core::EncodeFixed64(header, data.size());
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  return dest_->Append(data);
}

#if defined(TF_CORD_SUPPORT)
Status CustomWriter::WriteRecord(const absl::Cord& data) {
  char header[kHeaderSize];
  core::EncodeFixed64(header, data.size());
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  return dest_->Append(data);
}
#endif  // TF_CORD_SUPPORT

Status Reader::Create(Env* env, const std::string& filename,
                      const string& compression_type, int version,
                      const DataTypeVector& dtypes,
                      std::unique_ptr<Reader>* out_reader) {
  switch (version) {
    // CustomReader is able to read a legacy snapshot file format (v0) though
    // custom writer doesn't have the ability to write it any more since it is
    // strictly worse than V1.
    case 0:
    case 1:
      *out_reader = std::make_unique<CustomReader>(filename, compression_type,
                                                   version, dtypes);
      break;
    case 2:
      *out_reader =
          std::make_unique<TFRecordReader>(filename, compression_type, dtypes);
      break;
    default:
      return errors::InvalidArgument("Snapshot reader version: ", version,
                                     " is not supported.");
  }

  return (*out_reader)->Initialize(env);
}

Status Reader::SkipRecords(int64_t num_records) {
  // TODO(frankchn): Optimize to not parse the entire Tensor and actually skip.
  for (int i = 0; i < num_records; ++i) {
    std::vector<Tensor> unused_tensors;
    TF_RETURN_IF_ERROR(ReadTensors(&unused_tensors));
  }
  return OkStatus();
}

class Reader::Dataset : public DatasetBase {
 public:
  Dataset(DatasetContext&& ctx, const std::string& shard_dir,
          const std::string& compression, const int64_t version,
          const DataTypeVector& dtypes,
          const std::vector<PartialTensorShape>& shapes,
          const int64_t start_index)
      : DatasetBase(std::move(ctx)),
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

  std::string DebugString() const override { return "SnapshotDatasetReader"; }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return OkStatus();
  }

  Status CheckExternalState() const override { return OkStatus(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** node) const override {
    Node* shard_dir = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(shard_dir_, &shard_dir));

    Node* start_index = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(start_index_, &start_index));

    AttrValue compression;
    b->BuildAttrValue(compression_, &compression);

    AttrValue version;
    b->BuildAttrValue(version_, &version);

    return b->AddDataset(
        this,
        /*inputs=*/
        {std::make_pair(0, shard_dir), std::make_pair(1, start_index)},
        /*list_inputs=*/{},
        /*attrs=*/
        {{kCompression, compression}, {kVersion, version}},
        /*use_dataset_name=*/true, node);
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(node_name(), prefix)});
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          start_index_(dataset()->start_index_) {}

    Status Initialize(IteratorContext* ctx) override {
      // TODO(jsimsa): This only needs to happen when we are not restoring but
      // parallel_interleave op implementation caches IteratorContext (and thus
      // the is_restoring bit ends up being inaccurate).
      TF_RETURN_IF_ERROR(Reader::Create(
          ctx->env(), GetCurrentFilename(), dataset()->compression_,
          dataset()->version_, dataset()->dtypes_, &reader_));
      return AdvanceToStartIndex(ctx);
    }

   protected:
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      *end_of_sequence = false;
      Status s = reader_->ReadTensors(out_tensors);
      if (!errors::IsOutOfRange(s)) {
        start_index_++;
        return s;
      }
      Status status = AdvanceToNextFile(ctx->env());
      if (errors::IsNotFound(status)) {
        *end_of_sequence = true;
        return OkStatus();
      }
      return status;
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurrentCheckpointID),
                                             current_checkpoint_id_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kStartIndex), start_index_));
      return OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurrentCheckpointID),
                                            &current_checkpoint_id_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kStartIndex), &start_index_));
      TF_RETURN_IF_ERROR(ctx->env()->FileExists(GetCurrentFilename()));
      TF_RETURN_IF_ERROR(Reader::Create(
          ctx->env(), GetCurrentFilename(), dataset()->compression_,
          dataset()->version_, dataset()->dtypes_, &reader_));
      return AdvanceToStartIndex(ctx);
    }

   private:
    Status AdvanceToNextFile(Env* env) {
      start_index_ = 0;
      current_checkpoint_id_++;
      TF_RETURN_IF_ERROR(env->FileExists(GetCurrentFilename()));
      return Reader::Create(env, GetCurrentFilename(), dataset()->compression_,
                            dataset()->version_, dataset()->dtypes_, &reader_);
    }

    std::string GetCurrentFilename() {
      return GetCheckpointFileName(dataset()->shard_dir_,
                                   current_checkpoint_id_);
    }

    // TODO(frankchn): Optimize this to not parse every single element.
    Status AdvanceToStartIndex(IteratorContext* ctx) {
      for (int64_t i = 0; i < start_index_; ++i) {
        std::vector<Tensor> unused;
        TF_RETURN_IF_ERROR(reader_->ReadTensors(&unused));
      }
      return OkStatus();
    }

    std::unique_ptr<Reader> reader_;

    // Stores the id current checkpoint file that we are in the process of
    // reading (e.g. if the file is currently 00000001.snapshot, then this will
    // be 1).
    int64_t current_checkpoint_id_ = 0;
    int64_t start_index_;
  };

  const tstring shard_dir_;
  const std::string compression_;
  const int64_t version_;
  const DataTypeVector dtypes_;
  const std::vector<PartialTensorShape> shapes_;
  const int64_t start_index_;
};

Reader::DatasetOp::DatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCompression, &compression_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kVersion, &version_));
}

void Reader::DatasetOp::MakeDataset(OpKernelContext* ctx,
                                    DatasetBase** output) {
  tstring shard_dir;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "shard_dir", &shard_dir));

  int64_t start_index;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "start_index", &start_index));

  *output =
      new Reader::Dataset(DatasetContext(ctx), shard_dir, compression_,
                          version_, output_types_, output_shapes_, start_index);
}

class Reader::NestedDataset : public DatasetBase {
 public:
  explicit NestedDataset(DatasetContext&& ctx,
                         std::vector<DatasetBase*> datasets)
      : DatasetBase(std::move(ctx)), datasets_(datasets) {
    dtypes_.push_back(DT_VARIANT);
    gtl::InlinedVector<int64_t, 1> element_dim_sizes;
    element_dim_sizes.push_back(1);
    partial_shapes_.emplace_back(element_dim_sizes);
  }

  const DataTypeVector& output_dtypes() const override { return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return partial_shapes_;
  }

  std::string DebugString() const override {
    return "SnapshotNestedDatasetReader";
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->clear();
    return OkStatus();
  }

  Status CheckExternalState() const override { return OkStatus(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** node) const override {
    std::vector<Node*> input_graph_nodes;
    input_graph_nodes.reserve(datasets_.size());
    for (const auto& dataset : datasets_) {
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, dataset, &input_node));
      input_graph_nodes.emplace_back(input_node);
    }
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, /*inputs=*/{},
                      /*list_inputs=*/{std::make_pair(0, input_graph_nodes)},
                      /*attrs=*/{}, node));
    return OkStatus();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(node_name(), prefix)});
  }

 private:
  std::vector<DatasetBase*> datasets_;
  DataTypeVector dtypes_;
  std::vector<PartialTensorShape> partial_shapes_;

  class Iterator : public DatasetIterator<NestedDataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<NestedDataset>(params) {}

   protected:
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      const int64_t num_datasets = dataset()->datasets_.size();
      *end_of_sequence = num_datasets == index_;
      if (!*end_of_sequence) {
        Tensor tensor(DT_VARIANT, TensorShape({}));

        TF_RETURN_IF_ERROR(
            StoreDatasetInVariantTensor(dataset()->datasets_[index_], &tensor));
        out_tensors->clear();
        out_tensors->push_back(std::move(tensor));

        index_++;
      }
      return OkStatus();
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kIndex), index_));
      return OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kIndex), &index_));
      return OkStatus();
    }

   private:
    int64_t index_ = 0;
  };
};

Reader::NestedDatasetOp::NestedDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void Reader::NestedDatasetOp::MakeDataset(OpKernelContext* ctx,
                                          DatasetBase** output) {
  std::vector<DatasetBase*> inputs;
  for (size_t i = 0; i < ctx->num_inputs(); ++i) {
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(i), &input));
    inputs.push_back(input);
  }
  *output = new Reader::NestedDataset(DatasetContext(ctx), inputs);
  (*output)->Initialize(/*metadata=*/{});
}

Status Reader::MakeNestedDataset(Env* env,
                                 const std::vector<std::string>& shard_dirs,
                                 const string& compression_type, int version,
                                 const DataTypeVector& dtypes,
                                 const std::vector<PartialTensorShape>& shapes,
                                 const int64_t start_index,
                                 DatasetBase** output) {
  std::vector<DatasetBase*> datasets;

  datasets.reserve(shard_dirs.size());
  for (int64_t i = 0; i < shard_dirs.size(); ++i) {
    // TODO(frankchn): The reading pattern could be controlled in a non-round
    // robin fashion, so we cannot assume a round-robin manner when restoring.
    int64_t dataset_start_index = start_index / shard_dirs.size();
    if (start_index % shard_dirs.size() > datasets.size()) {
      dataset_start_index++;
    }

    datasets.push_back(
        new Dataset(DatasetContext(DatasetContext::Params(
                        {"SnapshotDatasetReader",
                         strings::StrCat("SnapshotDatasetReader/_", i)})),
                    shard_dirs.at(i), compression_type, version, dtypes, shapes,
                    dataset_start_index));
    datasets.back()->Initialize(/*metadata=*/{});
  }

  // Rotate the vector such that the first dataset contains the next element
  // to be produced, but not if there are no shards at all (then we just
  // construct an empty dataset).
  if (!shard_dirs.empty()) {
    std::rotate(datasets.begin(),
                datasets.begin() + (start_index % shard_dirs.size()),
                datasets.end());
  }

  *output = new NestedDataset(
      DatasetContext(DatasetContext::Params(
          {"SnapshotNestedDatasetReader", "SnapshotNestedDatasetReader"})),
      datasets);
  (*output)->Initialize(/*metadata=*/{});
  return OkStatus();
}

TFRecordReader::TFRecordReader(const std::string& filename,
                               const string& compression_type,
                               const DataTypeVector& dtypes,
                               std::optional<int64_t> output_buffer_size)
    : filename_(filename),
      offset_(0),
      compression_type_(compression_type),
      dtypes_(dtypes),
      output_buffer_size_(output_buffer_size) {}

Status TFRecordReader::Initialize(Env* env) {
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename_, &file_));
  auto options = io::RecordReaderOptions::CreateRecordReaderOptions(
      /*compression_type=*/compression_type_);
#if !defined(IS_SLIM_BUILD)
  if (output_buffer_size_.has_value()) {
    options.snappy_options.output_buffer_size = *output_buffer_size_;
    options.zlib_options.output_buffer_size = *output_buffer_size_;
  }
#endif  // IS_SLIM_BUILD
  record_reader_ = std::make_unique<io::RecordReader>(file_.get(), options);
  return OkStatus();
}

Status TFRecordReader::ReadTensors(std::vector<Tensor>* read_tensors) {
  read_tensors->reserve(dtypes_.size());
  for (int i = 0; i < dtypes_.size(); ++i) {
    tstring record;
    TF_RETURN_IF_ERROR(record_reader_->ReadRecord(&offset_, &record));

    TensorProto proto;
    if (!proto.ParseFromArray(record.data(), record.size())) {
      return errors::DataLoss(
          "Unable to parse tensor from stored proto in file: ", filename_,
          ", record ", offset_, ". Serialized proto: ", record);
    }

    Tensor tensor;
    if (!tensor.FromProto(proto)) {
      return errors::DataLoss(
          "Unable to parse tensor from stored proto in file: ", filename_,
          ", record ", offset_, ". TensorProto: ", proto.ShortDebugString());
    }

    read_tensors->push_back(std::move(tensor));
  }
  return OkStatus();
}

CustomReader::CustomReader(const std::string& filename,
                           const string& compression_type, const int version,
                           const DataTypeVector& dtypes)
    : filename_(filename),
      compression_type_(compression_type),
      version_(version),
      dtypes_(dtypes) {}

Status CustomReader::Initialize(Env* env) {
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename_, &file_));
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

    input_stream_ = std::make_unique<io::ZlibInputStream>(
        input_stream_.release(), zlib_options.input_buffer_size,
        zlib_options.output_buffer_size, zlib_options, true);
  } else if (compression_type_ == io::compression::kSnappy) {
    if (version_ == 0) {
      input_stream_ = std::make_unique<tsl::io::SnappyInputBuffer>(
          file_.get(), /*input_buffer_bytes=*/kSnappyReaderInputBufferSizeBytes,
          /*output_buffer_bytes=*/kSnappyReaderOutputBufferSizeBytes);
    } else {
      input_stream_ =
          std::make_unique<io::BufferedInputStream>(file_.get(), 64 << 20);
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

  return OkStatus();
}

Status CustomReader::ReadTensors(std::vector<Tensor>* read_tensors) {
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
    return errors::InvalidArgument("Compression ", compression_type_,
                                   " is not supported.");
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
  for (int i = 0, end = simple_tensor_mask_.size(); i < end; ++i) {
    if (simple_tensor_mask_[i]) {
      read_tensors->push_back(std::move(simple_tensors[simple_index]));
      simple_index++;
    } else {
      auto tensor_proto_str = std::move(tensor_proto_strs[complex_index].first);
      size_t tensor_proto_size = tensor_proto_strs[complex_index].second;
      TensorProto tp;
      if (!tp.ParseFromArray(tensor_proto_str.get(), tensor_proto_size)) {
        return errors::Internal("Could not parse TensorProto");
      }
      Tensor t;
      if (!t.FromProto(tp)) {
        return errors::Internal("Could not parse Tensor");
      }
      read_tensors->push_back(std::move(t));
      complex_index++;
    }
  }
  return OkStatus();
}

Status CustomReader::ReadTensorsV0(std::vector<Tensor>* read_tensors) {
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
  return OkStatus();
}

Status CustomReader::SnappyUncompress(
    const experimental::SnapshotTensorMetadata* metadata,
    std::vector<Tensor>* simple_tensors,
    std::vector<std::pair<std::unique_ptr<char[]>, size_t>>*
        tensor_proto_strs) {
  tstring compressed;
  TF_RETURN_IF_ERROR(ReadRecord(&compressed));
  size_t size;
  if (!tsl::port::Snappy_GetUncompressedLength(compressed.data(),
                                               compressed.size(), &size)) {
    return errors::Internal("Could not get snappy uncompressed length");
  }

  int num_tensors = metadata->tensor_metadata_size();
  std::vector<tsl::iovec> iov(num_tensors);
  int index = 0;
  int64_t total_size = 0;
  for (int i = 0, end = simple_tensor_mask_.size(); i < end; ++i) {
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
          std::make_unique<char[]>(tensor_metadata.tensor_size_bytes());
      iov[index].iov_base = tensor_proto_str.get();
      iov[index].iov_len = tensor_metadata.tensor_size_bytes();
      tensor_proto_strs->push_back(std::make_pair(
          std::move(tensor_proto_str), tensor_metadata.tensor_size_bytes()));
    }
    total_size += iov[index].iov_len;
    index++;
  }
  const int64_t size_int = size;
  if (size_int != total_size) {
    return errors::Internal("Uncompressed size mismatch. Snappy expects ", size,
                            " whereas the tensor metadata suggests ",
                            total_size);
  }
  if (!tsl::port::Snappy_UncompressToIOVec(compressed.data(), compressed.size(),
                                           iov.data(), num_tensors)) {
    return errors::Internal("Failed to perform snappy decompression.");
  }
  return OkStatus();
}

Status CustomReader::ReadRecord(tstring* record) {
  tstring header;
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
  uint64 length = core::DecodeFixed64(header.data());
  return input_stream_->ReadNBytes(length, record);
}

#if defined(TF_CORD_SUPPORT)
Status CustomReader::ReadRecord(absl::Cord* record) {
  tstring header;
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
  uint64 length = core::DecodeFixed64(header.data());
  if (compression_type_ == io::compression::kNone) {
    return input_stream_->ReadNBytes(length, record);
  } else {
    auto tmp_str = new tstring();
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(length, tmp_str));
    absl::string_view tmp_str_view(*tmp_str);
    record->Append(absl::MakeCordFromExternal(
        tmp_str_view, [tmp_str](absl::string_view) { delete tmp_str; }));
    return OkStatus();
  }
}
#endif  // TF_CORD_SUPPORT

Status WriteMetadataFile(Env* env, const string& dir,
                         const experimental::SnapshotMetadataRecord* metadata) {
  string metadata_filename = io::JoinPath(dir, kMetadataFilename);
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(dir));
  std::string tmp_filename =
      absl::StrCat(metadata_filename, "-tmp-", random::New64());
  TF_RETURN_IF_ERROR(WriteBinaryProto(env, tmp_filename, *metadata));
  return env->RenameFile(tmp_filename, metadata_filename);
}

Status WriteMetadataFile(
    Env* env, const string& dir,
    const experimental::DistributedSnapshotMetadata* metadata) {
  string metadata_filename = io::JoinPath(dir, kMetadataFilename);
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(dir));
  std::string tmp_filename =
      absl::StrCat(metadata_filename, "-tmp-", random::New64());
  TF_RETURN_IF_ERROR(WriteBinaryProto(env, tmp_filename, *metadata));
  return env->RenameFile(tmp_filename, metadata_filename);
}

Status ReadMetadataFile(Env* env, const string& dir,
                        experimental::SnapshotMetadataRecord* metadata,
                        bool* file_exists) {
  string metadata_filename = io::JoinPath(dir, kMetadataFilename);
  Status s = env->FileExists(metadata_filename);
  *file_exists = s.ok();

  if (*file_exists) {
    return ReadBinaryProto(env, metadata_filename, metadata);
  } else {
    return OkStatus();
  }
}

Status ReadMetadataFile(Env* env, const string& dir,
                        experimental::DistributedSnapshotMetadata* metadata,
                        bool* file_exists) {
  string metadata_filename = io::JoinPath(dir, kMetadataFilename);
  Status s = env->FileExists(metadata_filename);
  *file_exists = s.ok();

  if (*file_exists) {
    return ReadBinaryProto(env, metadata_filename, metadata);
  } else {
    return OkStatus();
  }
}

Status DumpDatasetGraph(Env* env, const std::string& path, uint64 hash,
                        const GraphDef* graph) {
  std::string hash_hex =
      strings::StrCat(strings::Hex(hash, strings::kZeroPad16));
  std::string graph_file =
      io::JoinPath(path, absl::StrCat(hash_hex, "-graph.pbtxt"));

  LOG(INFO) << "Graph hash is " << hash_hex << ", writing to " << graph_file;
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(path));
  return WriteTextProto(env, graph_file, *graph);
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
    return OkStatus();
  }

  if (mode_string == kModeWrite) {
    LOG(INFO) << "Overriding mode to writer.";
    *mode = WRITER;
    return OkStatus();
  }

  if (mode_string == kModePassthrough) {
    LOG(INFO) << "Overriding mode to passthrough.";
    *mode = PASSTHROUGH;
    return OkStatus();
  }

  if (!file_exists) {
    *mode = WRITER;
    return OkStatus();
  }

  if (metadata->finalized()) {
    // File found, snapshot has been finalized.
    *mode = READER;
    return OkStatus();
  }

  int64_t expiration_timer = static_cast<int64_t>(EnvTime::NowMicros()) -
                             pending_snapshot_expiry_seconds * 1000000;

  if (metadata->creation_timestamp() >= expiration_timer) {
    // Someone else is already writing and time has not expired.
    *mode = PASSTHROUGH;
    return OkStatus();
  } else {
    // Time has expired, we write regardless.
    *mode = WRITER;
    return OkStatus();
  }
}

AsyncWriter::AsyncWriter(Env* env, int64_t file_index,
                         const std::string& shard_directory,
                         uint64 checkpoint_id, const std::string& compression,
                         int64_t version, const DataTypeVector& output_types,
                         std::function<void(Status)> done) {
  thread_ = absl::WrapUnique(env->StartThread(
      ThreadOptions(), absl::StrCat("writer_thread_", file_index),
      [this, env, shard_directory, checkpoint_id, compression, version,
       &output_types, done = std::move(done)] {
        done(WriterThread(env, shard_directory, checkpoint_id, compression,
                          version, output_types));
      }));
}

void AsyncWriter::Write(const std::vector<Tensor>& tensors) {
  mutex_lock l(mu_);
  ElementOrEOF element;
  element.value = tensors;
  deque_.push_back(std::move(element));
}

void AsyncWriter::SignalEOF() {
  mutex_lock l(mu_);
  ElementOrEOF be;
  be.end_of_sequence = true;
  deque_.push_back(std::move(be));
}

void AsyncWriter::Consume(ElementOrEOF* be) {
  mutex_lock l(mu_);
  mu_.Await(tensorflow::Condition(this, &AsyncWriter::ElementAvailable));
  *be = deque_.front();
  deque_.pop_front();
}

bool AsyncWriter::ElementAvailable() { return !deque_.empty(); }

Status AsyncWriter::WriterThread(Env* env, const std::string& shard_directory,
                                 uint64 checkpoint_id,
                                 const std::string& compression,
                                 int64_t version, DataTypeVector output_types) {
  std::unique_ptr<snapshot_util::Writer> writer;
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(shard_directory));

  TF_RETURN_IF_ERROR(snapshot_util::Writer::Create(
      env, GetCheckpointFileName(shard_directory, checkpoint_id), compression,
      version, std::move(output_types), &writer));

  while (true) {
    ElementOrEOF be;
    Consume(&be);

    if (be.end_of_sequence) {
      TF_RETURN_IF_ERROR(writer->Close());
      break;
    }

    TF_RETURN_IF_ERROR(writer->WriteTensors(be.value));
  }
  return OkStatus();
}

namespace {

REGISTER_KERNEL_BUILDER(Name("SnapshotDatasetReader").Device(DEVICE_CPU),
                        Reader::DatasetOp);
REGISTER_KERNEL_BUILDER(Name("SnapshotNestedDatasetReader").Device(DEVICE_CPU),
                        Reader::NestedDatasetOp);

}  // namespace
}  // namespace snapshot_util
}  // namespace data
}  // namespace tensorflow
