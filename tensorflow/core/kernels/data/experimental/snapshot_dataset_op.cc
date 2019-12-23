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
#include <random>

#include "absl/time/clock.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/raw_coding.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/file_system.h"
#if !defined(IS_SLIM_BUILD)
#include "tensorflow/core/lib/io/snappy/snappy_inputbuffer.h"
#include "tensorflow/core/lib/io/snappy/snappy_outputbuffer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#endif  // IS_SLIM_BUILD
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/base64.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/data/experimental/snapshot.pb.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

enum SnapshotMode { READER = 0, WRITER = 1, PASSTHROUGH = 2 };

// Defaults to 10 GiB per shard.
const int64 kDefaultShardSizeBytes = 10LL * 1024 * 1024 * 1024;

const int64 kSnappyWriterInputBufferSizeBytes = 16 << 20;   // 16 MiB
const int64 kSnappyWriterOutputBufferSizeBytes = 16 << 20;  // 16 MiB

// The reader input buffer size is deliberately large because the input reader
// will throw an error if the compressed block length cannot fit in the input
// buffer.
const int64 kSnappyReaderInputBufferSizeBytes = 1 << 30;    // 1 GiB
const int64 kSnappyReaderOutputBufferSizeBytes = 16 << 20;  // 16 MiB

const size_t kHeaderSize = sizeof(uint64);

constexpr char kModeAuto[] = "auto";
constexpr char kModeWrite[] = "write";
constexpr char kModeRead[] = "read";
constexpr char kModePassthrough[] = "passthrough";

constexpr char kSnapshotFilename[] = "snapshot.metadata";
constexpr char kSnapshotReaderWorkerPool[] = "snapshot_reader_worker_pool";
constexpr char kSnapshotWriterWorkerPool[] = "snapshot_writer_worker_pool";
constexpr char kSeparator[] = "::";
constexpr char kBookkeeping[] = "Bookkeeping";
constexpr char kSnapshotReadElements[] = "snapshot_read_elements";
constexpr char kSnapshotReadThroughput[] = "snapshot_read_throughput";
constexpr char kSnapshotWrittenElements[] = "snapshot_written_elements";
constexpr char kSnapshotWriteThroughput[] = "snapshot_write_throughput";

constexpr char kSizeSuffix[] = "_size";
constexpr char kState[] = "state";
constexpr char kHashDir[] = "hash_dir";
constexpr char kRunId[] = "run_id";
constexpr char kRunDir[] = "run_dir";
constexpr char kFilenames[] = "filenames";
constexpr char kCurrentFilenames[] = "current_filenames";
constexpr char kElementsProduced[] = "elements_produced";
constexpr char kNextFileIndex[] = "next_file_index";
constexpr char kNumFilesDone[] = "num_files_done";
constexpr char kNumElementsRead[] = "num_elements_read";
constexpr char kStatus[] = "status";
constexpr char kCode[] = ".code";
constexpr char kErrorMessage[] = ".error_message";
constexpr char kEndOfSequence[] = "end_of_sequence";
constexpr char kBuffer[] = "buffer";
constexpr char kNumElementsWritten[] = "num_elements_written";
constexpr char kNextElem[] = "next_elem";

class SnapshotWriter {
 public:
  static constexpr const char* const kClassName = "SnapshotWriter";
  static constexpr const char* const kWriteStringPiece = "WriteStringPiece";
  static constexpr const char* const kWriteCord = "WriteCord";

  explicit SnapshotWriter(WritableFile* dest, const string& compression_type =
                                                  io::compression::kNone)
      : dest_(dest), compression_type_(compression_type) {
#if defined(IS_SLIM_BUILD)
    if (compression_type != io::compression::kNone) {
      LOG(ERROR) << "Compression is unsupported on mobile platforms. Turning "
                 << "off compression.";
    }
#else   // IS_SLIM_BUILD
    if (compression_type == io::compression::kGzip) {
      io::ZlibCompressionOptions zlib_options;
      zlib_options = io::ZlibCompressionOptions::GZIP();

      io::ZlibOutputBuffer* zlib_output_buffer = new io::ZlibOutputBuffer(
          dest, zlib_options.input_buffer_size, zlib_options.output_buffer_size,
          zlib_options);
      TF_CHECK_OK(zlib_output_buffer->Init());
      dest_ = zlib_output_buffer;
      dest_is_owned_ = true;
    } else if (compression_type == io::compression::kSnappy) {
      io::SnappyOutputBuffer* snappy_output_buffer = new io::SnappyOutputBuffer(
          dest, /*input_buffer_bytes=*/kSnappyWriterInputBufferSizeBytes,
          /*output_buffer_bytes=*/kSnappyWriterOutputBufferSizeBytes);
      dest_ = snappy_output_buffer;
      dest_is_owned_ = true;
    }
#endif  // IS_SLIM_BUILD
  }

  Status WriteRecord(const StringPiece& data) {
    profiler::TraceMe activity(
        absl::StrCat(kClassName, kSeparator, kWriteStringPiece),
        profiler::TraceMeLevel::kInfo);
    char header[kHeaderSize];
    core::EncodeFixed64(header, data.size());
    TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
    return dest_->Append(data);
  }

#if defined(PLATFORM_GOOGLE)
  Status WriteRecord(const absl::Cord& data) {
    profiler::TraceMe activity(absl::StrCat(kClassName, kSeparator, kWriteCord),
                               profiler::TraceMeLevel::kInfo);
    char header[kHeaderSize];
    core::EncodeFixed64(header, data.size());

    TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));

    return dest_->Append(data);
  }
#endif  // PLATFORM_GOOGLE

  Status Close() {
    if (dest_is_owned_) {
      Status s = dest_->Close();
      delete dest_;
      dest_ = nullptr;
      return s;
    }
    return Status::OK();
  }

  ~SnapshotWriter() {
    if (dest_ != nullptr) {
      Status s = Close();
      if (!s.ok()) {
        LOG(ERROR) << "Could not finish writing file: " << s;
      }
    }
  }

 private:
  WritableFile* dest_;
  bool dest_is_owned_ = false;
  const string compression_type_;
};

class SnapshotReader {
 public:
  static constexpr const char* const kClassName = "SnapshotReader";
  static constexpr const char* const kReadString = "ReadString";
  static constexpr const char* const kReadCord = "ReadCord";

  explicit SnapshotReader(
      RandomAccessFile* file,
      const string& compression_type = io::compression::kNone)
      : file_(file),
        input_stream_(new io::RandomAccessInputStream(file)),
        compression_type_(compression_type) {
#if defined(IS_SLIM_BUILD)
    if (compression_type_ != io::compression::kNone) {
      LOG(ERROR) << "Compression is unsupported on mobile platforms. Turning "
                 << "off compression.";
    }
#else   // IS_SLIM_BUILD
    if (compression_type_ == io::compression::kGzip) {
      io::ZlibCompressionOptions zlib_options;
      zlib_options = io::ZlibCompressionOptions::GZIP();

      input_stream_.reset(new io::ZlibInputStream(
          input_stream_.release(), zlib_options.input_buffer_size,
          zlib_options.output_buffer_size, zlib_options, true));
    } else if (compression_type_ == io::compression::kSnappy) {
      input_stream_ = absl::make_unique<io::SnappyInputBuffer>(
          file_, /*input_buffer_bytes=*/kSnappyReaderInputBufferSizeBytes,
          /*output_buffer_bytes=*/kSnappyReaderOutputBufferSizeBytes);
    }
#endif  // IS_SLIM_BUILD
  }

  Status ReadRecord(tstring* record) {
    profiler::TraceMe activity(
        absl::StrCat(kClassName, kSeparator, kReadString),
        profiler::TraceMeLevel::kInfo);
    tstring header;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
    uint64 length = core::DecodeFixed64(header.data());
    return input_stream_->ReadNBytes(length, record);
  }

#if defined(PLATFORM_GOOGLE)
  Status ReadRecord(absl::Cord* record) {
    profiler::TraceMe activity(absl::StrCat(kClassName, kSeparator, kReadCord),
                               profiler::TraceMeLevel::kInfo);
    tstring header;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
    uint64 length = core::DecodeFixed64(header.data());

    if (compression_type_ == io::compression::kNone) {
      return input_stream_->ReadNBytes(length, record);
    } else {
      tstring tmp_str;
      Status s = input_stream_->ReadNBytes(length, &tmp_str);
      record->Append(tmp_str);
      return s;
    }
  }
#endif

 private:
  RandomAccessFile* file_;
  std::unique_ptr<io::InputStreamInterface> input_stream_;
  const string compression_type_;
};

Status WriteMetadataFile(const string& hash_dir,
                         const experimental::SnapshotMetadataRecord& metadata) {
  string metadata_filename = io::JoinPath(hash_dir, kSnapshotFilename);
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(hash_dir));

  std::string tmp_filename =
      absl::StrCat(metadata_filename, "-tmp-", random::New64());

  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(tmp_filename, &file));

  auto writer = absl::make_unique<SnapshotWriter>(file.get());
  TF_RETURN_IF_ERROR(writer->WriteRecord(metadata.SerializeAsString()));
  TF_RETURN_IF_ERROR(writer->Close());
  TF_RETURN_IF_ERROR(file->Sync());
  TF_RETURN_IF_ERROR(file->Close());

  TF_RETURN_IF_ERROR(
      Env::Default()->RenameFile(tmp_filename, metadata_filename));

  return Status::OK();
}

Status ReadMetadataFile(const string& hash_dir,
                        experimental::SnapshotMetadataRecord* metadata) {
  string metadata_filename = io::JoinPath(hash_dir, kSnapshotFilename);
  TF_RETURN_IF_ERROR(Env::Default()->FileExists(metadata_filename));

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(
      Env::Default()->NewRandomAccessFile(metadata_filename, &file));

  tstring record_bytes;
  auto reader = absl::make_unique<SnapshotReader>(file.get());
  TF_RETURN_IF_ERROR(reader->ReadRecord(&record_bytes));

  metadata->ParseFromString(record_bytes);
  return Status::OK();
}

Status DumpDatasetGraph(const std::string& path, uint64 hash,
                        const GraphDef& graph) {
  std::string hash_hex =
      strings::StrCat(strings::Hex(hash, strings::kZeroPad16));
  std::string graph_file =
      io::JoinPath(path, absl::StrCat(hash_hex, "-graph.pbtxt"));

  LOG(INFO) << "Graph hash is " << hash_hex << ", writing to " << graph_file;
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(path));
  return WriteTextProto(Env::Default(), graph_file, graph);
}

Status DetermineOpState(const std::string& mode_string,
                        const Status& file_status,
                        const experimental::SnapshotMetadataRecord& metadata,
                        const uint64 pending_snapshot_expiry_seconds,
                        SnapshotMode* mode) {
  if (mode_string == kModeRead) {
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

  if (errors::IsNotFound(file_status)) {
    *mode = WRITER;
    return Status::OK();
  }

  if (!file_status.ok()) {
    return file_status;
  }

  if (metadata.finalized()) {
    // File found, snapshot has been finalized.
    *mode = READER;
    return Status::OK();
  }

  if (metadata.creation_timestamp() >=
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

class SnapshotDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit SnapshotDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));

    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("reader_path_prefix", &reader_path_prefix_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("writer_path_prefix", &writer_path_prefix_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("compression", &compression_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("shard_size_bytes", &shard_size_bytes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pending_snapshot_expiry_seconds",
                                     &pending_snapshot_expiry_seconds_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("num_reader_threads", &num_reader_threads_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("reader_buffer_size", &reader_buffer_size_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("num_writer_threads", &num_writer_threads_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("writer_buffer_size", &writer_buffer_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shuffle_on_read", &shuffle_on_read_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2_));

    mode_ = kModeAuto;
    if (ctx->HasAttr("mode")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_));
    }

    snapshot_name_ = "";
    if (ctx->HasAttr("snapshot_name")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("snapshot_name", &snapshot_name_));
    }

    if (shard_size_bytes_ == -1) shard_size_bytes_ = kDefaultShardSizeBytes;

    // Default to 1 day expiry for snapshots.
    if (pending_snapshot_expiry_seconds_ == -1) {
      pending_snapshot_expiry_seconds_ = 86400;
    }

    if (num_reader_threads_ == -1) num_reader_threads_ = 1;
    if (reader_buffer_size_ == -1) reader_buffer_size_ = 1;
    if (num_writer_threads_ == -1) num_writer_threads_ = 1;
    if (writer_buffer_size_ == -1) writer_buffer_size_ = 1;

    OP_REQUIRES(
        ctx,
        compression_ == io::compression::kNone ||
            compression_ == io::compression::kGzip ||
            compression_ == io::compression::kSnappy,
        errors::InvalidArgument("compression must be either '', 'GZIP' or "
                                "'SNAPPY'."));

    OP_REQUIRES(
        ctx, pending_snapshot_expiry_seconds_ >= 1,
        errors::InvalidArgument(
            "pending_snapshot_expiry_seconds must be at least 1 second."));

    OP_REQUIRES(ctx,
                mode_ == kModeAuto || mode_ == kModeRead ||
                    mode_ == kModeWrite || mode_ == kModePassthrough,
                errors::InvalidArgument("mode must be either '", kModeAuto,
                                        "', '", kModeRead, "', '", kModeWrite,
                                        "', or '", kModePassthrough, "'."));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    tstring path;

    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "path", &path));

    SerializationContext::Params params;
    std::vector<std::pair<string, Tensor>> input_list;
    params.input_list = &input_list;
    params.external_state_policy =
        SerializationContext::ExternalStatePolicy::kIgnore;

    GraphDef graph_def;
    OP_REQUIRES_OK(
        ctx, AsGraphDef(ctx, input, SerializationContext(params), &graph_def));

    uint64 hash;
    OP_REQUIRES_OK(ctx, HashGraph(graph_def, &hash));

    Status dump_status = DumpDatasetGraph(path, hash, graph_def);
    if (!dump_status.ok()) {
      LOG(WARNING) << "Unable to write graphdef to disk, error: "
                   << dump_status.ToString();
    }

    std::string graph_hash =
        strings::StrCat(strings::Hex(hash, strings::kZeroPad16));
    LOG(INFO) << "Graph def serialized to hash: " << graph_hash;

    *output = new Dataset(ctx, input, path, graph_hash, reader_path_prefix_,
                          writer_path_prefix_, compression_, shard_size_bytes_,
                          pending_snapshot_expiry_seconds_, num_reader_threads_,
                          reader_buffer_size_, num_writer_threads_,
                          writer_buffer_size_, shuffle_on_read_, seed_, seed2_,
                          mode_, snapshot_name_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input, const string& path,
            const string& graph_hash, const string& reader_path_prefix,
            const string& writer_path_prefix, const string& compression,
            const uint64 shard_size_bytes,
            const uint64 pending_snapshot_expiry_seconds,
            const uint64 num_reader_threads, const uint64 reader_buffer_size,
            const uint64 num_writer_threads, const uint64 writer_buffer_size,
            const bool shuffle_on_read, const uint64 seed, const uint64 seed2,
            const std::string& mode, const std::string& snapshot_name)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          dir_(path),
          graph_hash_(graph_hash),
          reader_path_prefix_(reader_path_prefix),
          writer_path_prefix_(writer_path_prefix),
          compression_(compression),
          shard_size_bytes_(shard_size_bytes),
          pending_snapshot_expiry_seconds_(pending_snapshot_expiry_seconds),
          num_reader_threads_(num_reader_threads),
          reader_buffer_size_(reader_buffer_size),
          num_writer_threads_(num_writer_threads),
          writer_buffer_size_(writer_buffer_size),
          shuffle_on_read_(shuffle_on_read),
          seed_(seed),
          seed2_(seed2),
          mode_(mode),
          snapshot_name_(snapshot_name) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, absl::StrCat(prefix, "::Snapshot")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "SnapshotDatasetOp::Dataset"; }

    int64 Cardinality() const override { return input_->Cardinality(); }

    Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      Node* path = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(dir_, &path));

      AttrValue compression_attr;
      b->BuildAttrValue(compression_, &compression_attr);

      AttrValue reader_path_prefix_attr;
      b->BuildAttrValue(reader_path_prefix_, &reader_path_prefix_attr);

      AttrValue writer_path_prefix_attr;
      b->BuildAttrValue(writer_path_prefix_, &writer_path_prefix_attr);

      AttrValue shard_size_bytes_attr;
      b->BuildAttrValue<int64>(shard_size_bytes_, &shard_size_bytes_attr);

      AttrValue pending_snapshot_expiry_seconds_attr;
      b->BuildAttrValue<int64>(pending_snapshot_expiry_seconds_,
                               &pending_snapshot_expiry_seconds_attr);

      AttrValue num_reader_threads_attr;
      b->BuildAttrValue<int64>(num_reader_threads_, &num_reader_threads_attr);

      AttrValue reader_buffer_size_attr;
      b->BuildAttrValue<int64>(reader_buffer_size_, &reader_buffer_size_attr);

      AttrValue num_writer_threads_attr;
      b->BuildAttrValue<int64>(num_writer_threads_, &num_writer_threads_attr);

      AttrValue writer_buffer_size_attr;
      b->BuildAttrValue<int64>(writer_buffer_size_, &writer_buffer_size_attr);

      AttrValue shuffle_on_read_attr;
      b->BuildAttrValue<bool>(shuffle_on_read_, &shuffle_on_read_attr);

      AttrValue seed_attr;
      b->BuildAttrValue<int64>(seed_, &seed_attr);

      AttrValue seed2_attr;
      b->BuildAttrValue<int64>(seed2_, &seed2_attr);

      AttrValue mode_attr;
      b->BuildAttrValue(mode_, &mode_attr);

      AttrValue snapshot_name_attr;
      b->BuildAttrValue(snapshot_name_, &snapshot_name_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          /*inputs=*/
          {std::make_pair(0, input_graph_node), std::make_pair(1, path)},
          /*list_inputs=*/
          {},
          /*attrs=*/
          {{"compression", compression_attr},
           {"reader_path_prefix", reader_path_prefix_attr},
           {"writer_path_prefix", writer_path_prefix_attr},
           {"shard_size_bytes", shard_size_bytes_attr},
           {"pending_snapshot_expiry_seconds",
            pending_snapshot_expiry_seconds_attr},
           {"num_reader_threads", num_reader_threads_attr},
           {"reader_buffer_size", reader_buffer_size_attr},
           {"num_writer_threads", num_writer_threads_attr},
           {"writer_buffer_size", writer_buffer_size_attr},
           {"shuffle_on_read", shuffle_on_read_attr},
           {"seed", seed_attr},
           {"seed2", seed2_attr},
           {"mode", mode_attr},
           {"snapshot_name", snapshot_name_attr}},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
        if (dataset()->snapshot_name_.empty()) {
          hash_dir_ = io::JoinPath(dataset()->dir_, dataset()->graph_hash_);
        } else {
          hash_dir_ = io::JoinPath(
              dataset()->dir_,
              strings::StrCat("custom-", dataset()->snapshot_name_));
        }
      }

      // We have a somewhat non traditional pattern for iterator initialization
      // for Snapshot. The protocol is that we initialize the Reader / Writer
      // iterator on the first GetNext call. We also invoke the same
      // initialization code when restoring as well. The reason why we don't do
      // this during the Initialize call is because during Restore we call
      // Initialize at first and at that point we don't know which iterator
      // (Reader / Writer / Passthrough) we need to restore as this info is part
      // of the checkpoint.
      Status Initialize(IteratorContext* ctx) override {
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (iterator_ == nullptr) {
          experimental::SnapshotMetadataRecord metadata;
          Status s = ReadMetadataFile(hash_dir_, &metadata);
          TF_RETURN_IF_ERROR(DetermineOpState(
              dataset()->mode_, s, metadata,
              dataset()->pending_snapshot_expiry_seconds_, &state_));
          TF_RETURN_IF_ERROR(InitializeIterator(ctx, metadata));
        }
        return iterator_->GetNext(ctx, out_tensors, end_of_sequence);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(writer, iterator_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kState), static_cast<int64>(state_)));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kHashDir), hash_dir_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        tstring hash_dir;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kHashDir), &hash_dir));
        if (hash_dir != hash_dir_) {
          LOG(ERROR) << "Dataset has changed while restoring from the "
                        "checkpoint. Old hash: "
                     << hash_dir << "; new hash: " << hash_dir_;
          return Status::OK();
        }
        {
          int64 temp;
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kState), &temp));
          state_ = SnapshotMode(temp);
        }
        experimental::SnapshotMetadataRecord metadata;
        TF_RETURN_IF_ERROR(ReadMetadataFile(hash_dir_, &metadata));
        TF_RETURN_IF_ERROR(InitializeIterator(ctx, metadata));
        return RestoreInput(ctx, reader, iterator_);
      }

      // This method expects that state_ is populated and it will create the
      // correct Reader / Writer / Passthrough iterator and initialize it.
      Status InitializeIterator(
          IteratorContext* ctx,
          const experimental::SnapshotMetadataRecord& metadata)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        std::string run_id = "";
        if (!dataset()->snapshot_name_.empty()) {
          // We have overridden the snapshot with a custom name, so we don't
          // generate random run ids, but just use the same one.
          run_id = "custom";
        }

        switch (state_) {
          case WRITER:
            iterator_ = absl::make_unique<SnapshotWriterIterator>(
                SnapshotWriterIterator::Params{
                    dataset(), absl::StrCat(prefix(), "WriterImpl")},
                hash_dir_, run_id);
            break;
          case READER:
            if (run_id.empty() && metadata.run_id().empty()) {
              return errors::NotFound(
                  "Could not find a valid snapshot to read.");
            }
            if (run_id.empty()) {
              run_id = metadata.run_id();
            }
            iterator_ = absl::make_unique<SnapshotReaderIterator>(
                SnapshotReaderIterator::Params{
                    dataset(), absl::StrCat(prefix(), "ReaderImpl")},
                hash_dir_, run_id);
            break;
          case PASSTHROUGH:
            iterator_ = absl::make_unique<SnapshotPassthroughIterator>(
                SnapshotPassthroughIterator::Params{
                    dataset(), absl::StrCat(prefix(), "PassthroughImpl")});
            break;
        }
        return iterator_->Initialize(ctx);
      }

     protected:
      class SnapshotReaderIterator : public DatasetIterator<Dataset> {
       public:
        static constexpr const char* const kParse = "Parse";

        explicit SnapshotReaderIterator(const Params& params,
                                        const string& hash_dir,
                                        const string& run_id)
            : DatasetIterator<Dataset>(params),
              hash_dir_(hash_dir),
              run_id_(run_id) {}

        ~SnapshotReaderIterator() override {
          mutex_lock l(mu_);
          cancelled_ = true;
          cond_var_.notify_all();
          while (num_active_threads_ > 0) {
            cond_var_.wait(l);
          }
        }

        Status Initialize(IteratorContext* ctx) override {
          mutex_lock l(mu_);
          thread_pool_ = ctx->CreateThreadPool(kSnapshotReaderWorkerPool,
                                               dataset()->num_reader_threads_);
          run_dir_ = io::JoinPath(hash_dir_, run_id_);
          // Get all the files in the run_dir.
          std::vector<std::string> filenames_str;
          TF_RETURN_IF_ERROR(ctx->env()->GetMatchingPaths(
              absl::StrCat(run_dir_, "/*"), &filenames_str));
          filenames_.resize(filenames_str.size());
          std::copy(filenames_str.begin(), filenames_str.end(),
                    filenames_.begin());
          if (filenames_.empty()) {
            return errors::NotFound("Could not find any files in dir: ",
                                    run_dir_);
          }

          if (dataset()->shuffle_on_read_) {
            uint64 seed = dataset()->seed_ + dataset()->seed2_;
            if (dataset()->seed_ == 0 && dataset()->seed2_ == 0) {
              seed = random::New64();
            }

            std::mt19937 rng(seed);
            std::shuffle(filenames_.begin(), filenames_.end(), rng);
          } else {
            std::sort(filenames_.begin(), filenames_.end());
          }

          for (auto i = 0; i < dataset()->num_reader_threads_; ++i) {
            curr_filenames_.push_back(GetNextFilename());
          }
          return Status::OK();
        }

        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
          absl::Time start = absl::Now();
          mutex_lock l(mu_);
          if (!background_threads_started_) {
            for (int i = 0; i < dataset()->num_reader_threads_; ++i) {
              ++num_active_threads_;
              thread_pool_->Schedule([this, i]() { ReadingFilesLoop(i); });
            }
            background_threads_started_ = true;
          }

          // Wait till the buffer has something in it.
          while (!cancelled_ && buffer_.empty() &&
                 !background_threads_finished_) {
            cond_var_.wait(l);
          }

          if (cancelled_) {
            return errors::Cancelled(
                "SnapshotDatasetOp::Dataset::SnapshotReaderIterator::GetNext");
          }

          const auto& stats_aggregator = ctx->stats_aggregator();
          if (stats_aggregator) {
            stats_aggregator->AddScalar(
                absl::StrCat(dataset()->node_name(), kSeparator,
                             kSnapshotReadElements),
                static_cast<float>(num_elements_read_), elements_produced_);
          }

          if (!buffer_.empty()) {
            Status s = buffer_.front().status;
            if (s.ok()) {
              *end_of_sequence = false;
              *out_tensors = std::move(buffer_.front().value);

              {
                profiler::TraceMe activity(
                    absl::StrCat(prefix(), kSeparator, kBookkeeping),
                    profiler::TraceMeLevel::kInfo);
                // Printing some statistics along the way.
                int64 num_bytes = 0;
                for (int i = 0; i < out_tensors->size(); ++i) {
                  num_bytes += (*out_tensors)[i].TotalBytes();
                }
                absl::Time end = absl::Now();
                absl::Duration d = end - start;
                time_spent_micros_ += absl::ToInt64Microseconds(d);
                kbytes_read_ += static_cast<double>(num_bytes) / 1024.0;
                float read_throughput =
                    (kbytes_read_ / 1024.0) / (time_spent_micros_ / 1000000.0);
                if (stats_aggregator) {
                  stats_aggregator->AddScalar(
                      absl::StrCat(dataset()->node_name(), kSeparator,
                                   kSnapshotReadThroughput),
                      read_throughput, elements_produced_);
                }
                elements_produced_++;
                if (elements_produced_ % 10000 == 0) {
                  LOG(INFO)
                      << "Current read throughput (MBPS): " << read_throughput;
                }
              }
            }
            buffer_.pop_front();
            cond_var_.notify_all();
            return s;
          }

          if (background_threads_finished_) {
            *end_of_sequence = true;
            return Status::OK();
          }

          return errors::Internal("Unreachable point in SnapshotReader");
        }

       protected:
        Status SaveInternal(IteratorStateWriter* writer) override {
          mutex_lock l(mu_);
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kHashDir), hash_dir_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kRunId), run_id_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kRunDir), run_dir_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(kFilenames, kSizeSuffix)),
              filenames_.size()));
          for (size_t i = 0; i < filenames_.size(); ++i) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(kFilenames, "[", i, "]")),
                filenames_[i]));
          }
          for (auto i = 0; i < dataset()->num_reader_threads_; ++i) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(kCurrentFilenames, "[", i, "]")),
                curr_filenames_[i]));
          }
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kElementsProduced),
                                                 elements_produced_));
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kNextFileIndex), next_file_index_));
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kNumFilesDone), num_files_done_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kNumElementsRead),
                                                 num_elements_read_));
          return Status::OK();
        }

        Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
          mutex_lock l(mu_);
          tstring hash_dir, run_id, run_dir;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name(kHashDir), &hash_dir));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kHashDir), &run_id));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kHashDir), &run_dir));
          if (run_dir != run_dir_) {
            LOG(ERROR) << "Restoring read iterator from ckpt with old "
                       << "run_dir: " << run_dir
                       << " but new run_dir is: " << run_dir_
                       << ". We'll now restart snapshot creation.";
            return Status::OK();
          }
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kRunId), &run_id_));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kRunDir), &run_dir_));
          curr_filenames_.clear();
          curr_filenames_.reserve(dataset()->num_reader_threads_);
          for (auto i = 0; i < dataset()->num_reader_threads_; ++i) {
            curr_filenames_.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kCurrentFilenames, "[", i, "]")),
                &curr_filenames_.back()));
          }
          size_t filenames_size;
          {
            int64 temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kFilenames, kSizeSuffix)), &temp));
            filenames_size = static_cast<size_t>(temp);
          }
          if (filenames_.size() != filenames_size) {
            LOG(ERROR) << "Old filenames size: " << filenames_size
                       << "; new filenames size: " << filenames_.size();
          }
          filenames_.clear();
          filenames_.reserve(filenames_size);
          for (size_t i = 0; i < filenames_size; ++i) {
            filenames_.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kFilenames, "[", i, "]")),
                &filenames_.back()));
          }
          {
            int64 temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name(kElementsProduced), &temp));
            elements_produced_ = static_cast<uint64>(temp);
          }
          {
            int64 temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name(kNextFileIndex), &temp));
            next_file_index_ = static_cast<uint64>(temp);
          }
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name(kNumFilesDone), &num_files_done_));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNumElementsRead),
                                                &num_elements_read_));
          return Status::OK();
        }

       private:
        // Reads one file end to end.
        Status ReadFile(const string& filename) {
          std::unique_ptr<RandomAccessFile> file;
          TF_CHECK_OK(Env::Default()->NewRandomAccessFile(filename, &file));
          std::unique_ptr<SnapshotReader> reader(
              new SnapshotReader(file.get(), dataset()->compression_));

          while (true) {
            // Wait for a slot in the buffer.
            {
              mutex_lock l(mu_);
              while (!cancelled_ &&
                     buffer_.size() >= dataset()->reader_buffer_size_) {
                cond_var_.wait(l);
              }

              if (cancelled_) {
                return errors::Cancelled(
                    "SnapshotDatasetOp::Dataset::SnapshotReaderIterator::"
                    "ReadFile");
              }
            }
#if !defined(PLATFORM_GOOGLE)
            string record_bytes;
            Status s = reader->ReadRecord(&record_bytes);
#else
            absl::Cord record_cord;
            Status s = reader->ReadRecord(&record_cord);
#endif
            if (s.ok()) {
              profiler::TraceMe activity(
                  absl::StrCat(prefix(), kSeparator, kParse),
                  profiler::TraceMeLevel::kInfo);
              experimental::SnapshotRecord record;
#if !defined(PLATFORM_GOOGLE)
              record.ParseFromString(record_bytes);
#else
              record.ParseFromCord(record_cord);
#endif
              std::vector<Tensor> out_tensors;
              for (int i = 0; i < record.tensor_size(); ++i) {
                Tensor t;
                if (!t.FromProto(record.tensor(i))) {
                  return errors::DataLoss("Unable to parse tensor from proto.");
                }
                out_tensors.push_back(t);
              }
              BufferElement elem;
              std::swap(elem.value, out_tensors);
              elem.status = Status::OK();
              mutex_lock l(mu_);
              buffer_.push_back(std::move(elem));
              num_elements_read_++;
              cond_var_.notify_all();
            } else if (errors::IsOutOfRange(s)) {
              return Status::OK();
            } else {
              return s;
            }
          }
          return Status::OK();
        }

        string GetNextFilename() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          if (next_file_index_ >= filenames_.size()) {
            return "";
          }
          string filename = io::JoinPath(dataset()->reader_path_prefix_,
                                         filenames_[next_file_index_]);
          next_file_index_++;
          return filename;
        }

        // Pulls one file off the filenames_ list and reads it through. When
        // all files are read, terminates.
        void ReadingFilesLoop(int i) {
          auto cleanup = gtl::MakeCleanup([this]() {
            mutex_lock l(mu_);
            --num_active_threads_;
            cond_var_.notify_all();
          });
          while (true) {
            string filename = "";
            {
              mutex_lock l(mu_);
              filename = curr_filenames_[i];
              if (filename.empty()) {
                return;
              }
              VLOG(2) << "Starting to read: " << filename;
            }
            Status s = ReadFile(filename);
            // If we get to the end of the file, it's a clean termination and
            // we are at the end of the file. If all files have been processed,
            // then we insert an end_of_sequence marker in the buffer and
            // terminate the loop.
            if (s.ok()) {
              VLOG(2) << "Finished reading: " << filename;
              mutex_lock l(mu_);
              num_files_done_++;
              if (num_files_done_ >= filenames_.size()) {
                background_threads_finished_ = true;
                cond_var_.notify_all();
                return;
              }
              curr_filenames_[i] = GetNextFilename();
            } else {
              LOG(ERROR) << "Encountered an error: " << s.ToString();
              BufferElement elem;
              elem.status = s;
              mutex_lock l(mu_);
              buffer_.push_back(std::move(elem));
              cond_var_.notify_all();
              return;
            }
          }
        }

        Status WriteStatus(IteratorStateWriter* writer, size_t index,
                           const Status& status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              CodeKey(index), static_cast<int64>(status.code())));
          if (!status.ok()) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(ErrorMessageKey(index),
                                                   status.error_message()));
          }
          return Status::OK();
        }

        Status ReadStatus(IteratorStateReader* reader, size_t index,
                          Status* status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          int64 code_int;
          TF_RETURN_IF_ERROR(reader->ReadScalar(CodeKey(index), &code_int));
          error::Code code = static_cast<error::Code>(code_int);

          if (code != error::Code::OK) {
            tstring error_message;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(ErrorMessageKey(index), &error_message));
            *status = Status(code, error_message);
          } else {
            *status = Status::OK();
          }
          return Status::OK();
        }

        string CodeKey(size_t index) {
          return full_name(strings::StrCat(kStatus, "[", index, "]", kCode));
        }

        string ErrorMessageKey(size_t index) {
          return full_name(
              strings::StrCat(kStatus, "[", index, "]", kErrorMessage));
        }

        struct BufferElement {
          Status status;
          std::vector<Tensor> value;
        };

        mutex mu_;
        condition_variable cond_var_;

        const string hash_dir_;
        const experimental::SnapshotMetadataRecord metadata_;
        tstring run_id_ GUARDED_BY(mu_);
        tstring run_dir_ GUARDED_BY(mu_);
        std::vector<tstring> filenames_;

        uint64 elements_produced_ GUARDED_BY(mu_) = 0;
        int64 time_spent_micros_ GUARDED_BY(mu_) = 0;
        double kbytes_read_ GUARDED_BY(mu_) = 0;
        size_t next_file_index_ GUARDED_BY(mu_) = 0;
        int64 num_files_done_ GUARDED_BY(mu_) = 0;

        std::unique_ptr<thread::ThreadPool> thread_pool_;
        int64 num_active_threads_ GUARDED_BY(mu_) = 0;
        std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
        bool cancelled_ GUARDED_BY(mu_) = false;
        bool background_threads_started_ GUARDED_BY(mu_) = false;
        bool background_threads_finished_ GUARDED_BY(mu_) = false;
        int64 num_elements_read_ GUARDED_BY(mu_) = 0;
        // curr_filenames_ tracks which file is being read by each thread.
        std::vector<tstring> curr_filenames_ GUARDED_BY(mu_);
      };

      class SnapshotWriterIterator : public DatasetIterator<Dataset> {
       public:
        static constexpr const char* const kProcessOneElement =
            "ProcessOneElement";

        explicit SnapshotWriterIterator(const Params& params,
                                        const string& hash_dir,
                                        const string& run_id)
            : DatasetIterator<Dataset>(params),
              hash_dir_(hash_dir),
              run_id_(run_id) {}

        ~SnapshotWriterIterator() override {
          mutex_lock l(mu_);
          cancelled_ = true;
          cond_var_.notify_all();
          while (num_active_threads_ > 0) {
            cond_var_.wait(l);
          }
        }

        Status Initialize(IteratorContext* ctx) override {
          mutex_lock l(mu_);
          thread_pool_ = ctx->CreateThreadPool(kSnapshotWriterWorkerPool,
                                               dataset()->num_writer_threads_);
          if (run_id_.empty()) {
            run_id_ = strings::StrCat(
                strings::Hex(random::New64(), strings::kZeroPad4));
          }
          run_dir_ =
              io::JoinPath(dataset()->writer_path_prefix_, hash_dir_, run_id_);
          return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
        }

        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
          absl::Time start = absl::Now();

          bool first_call;
          bool is_restored;
          {
            mutex_lock l(mu_);
            first_call = first_call_;
            is_restored = is_restored_;
            if (first_call_) {
              // If we're restoring then the directory already exists and we
              // don't want to overwrite the snapshot metadata file.
              if (!is_restored_) {
                TF_RETURN_IF_ERROR(
                    Env::Default()->RecursivelyCreateDir(run_dir_));
                experimental::SnapshotMetadataRecord metadata;
                metadata.set_creation_timestamp(EnvTime::NowMicros());
                metadata.set_graph_hash(dataset()->graph_hash_);
                metadata.set_run_id(run_id_.data(), run_id_.size());
                metadata.set_finalized(false);
                TF_RETURN_IF_ERROR(WriteMetadataFile(hash_dir_, metadata));
              }
              for (int i = 0; i < dataset()->num_writer_threads_; ++i) {
                ++num_active_threads_;
                thread_pool_->Schedule([this]() { WriterThread(); });
              }
              first_call_ = false;
            }
          }

          // When we reach the end of the data, we'd like to finalize the
          // snapshot and write the metadata file out. If we just check for
          // end_of_sequence on the GetNext call then we will need to make
          // N + 1 GetNext calls (if N is the total number of elements in the
          // dataset). So right now we solve this issue by prefetching the next
          // element in the data stream. Therefore the first call ends up
          // pulling two elements.
          if (first_call && !is_restored) {
            TF_RETURN_IF_ERROR(FillBuffer(ctx));
          }

          {
            mutex_lock l(mu_);
            // Populate out_tensors with the prefetched data.
            *out_tensors = std::move(next_elem_.value);
            *end_of_sequence = next_elem_.end_of_sequence;
          }

          // Update prefetched_elem with the next element.
          TF_RETURN_IF_ERROR(FillBuffer(ctx));

          {
            profiler::TraceMe activity(
                absl::StrCat(prefix(), kSeparator, kBookkeeping),
                profiler::TraceMeLevel::kInfo);

            // Book keeping to report some statistics.
            mutex_lock l(mu_);
            int64 num_bytes = 0;
            for (auto out_tensor : *out_tensors) {
              num_bytes += out_tensor.TotalBytes();
            }

            const auto& stats_aggregator = ctx->stats_aggregator();
            if (stats_aggregator) {
              stats_aggregator->AddScalar(
                  absl::StrCat(dataset()->node_name(), kSeparator,
                               kSnapshotWrittenElements),
                  static_cast<float>(num_elements_written_),
                  elements_produced_);
            }

            absl::Time end = absl::Now();
            absl::Duration d = end - start;
            time_spent_micros_ += absl::ToInt64Microseconds(d);
            bytes_produced_ += num_bytes;
            float write_throughput = (bytes_produced_ * 1000000.0) /
                                     (time_spent_micros_ * 1024.0 * 1024.0);
            if (stats_aggregator) {
              stats_aggregator->AddScalar(
                  absl::StrCat(dataset()->node_name(), kSeparator,
                               kSnapshotWriteThroughput),
                  write_throughput, elements_produced_);
            }

            elements_produced_++;
            if (elements_produced_ % 10000 == 0) {
              LOG(INFO) << "Current write throughput (MBPS): "
                        << write_throughput;
            }
          }
          return Status::OK();
        }

       protected:
        Status SaveInternal(IteratorStateWriter* writer) override {
          mutex_lock l(mu_);
          TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
          if (end_of_sequence_) {
            TF_RETURN_IF_ERROR(
                writer->WriteScalar(full_name(kEndOfSequence), ""));
          }
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kHashDir), hash_dir_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kRunId), run_id_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kRunDir), run_dir_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kElementsProduced),
                                                 elements_produced_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(kBuffer, kSizeSuffix)),
              buffer_.size()));
          for (size_t i = 0; i < buffer_.size(); ++i) {
            auto& buffer_element = buffer_[i];
            if (buffer_element.end_of_sequence) {
              TF_RETURN_IF_ERROR(writer->WriteScalar(
                  full_name(
                      strings::StrCat(kBuffer, "[", i, "].", kEndOfSequence)),
                  ""));
            }
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(kBuffer, "[", i, "]", kSizeSuffix)),
                buffer_element.value.size()));
            for (size_t j = 0; j < buffer_element.value.size(); j++) {
              TF_RETURN_IF_ERROR(writer->WriteTensor(
                  full_name(strings::StrCat(kBuffer, "[", i, "][", j, "]")),
                  buffer_element.value[j]));
            }
          }
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kNextFileIndex), next_file_index_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kNumElementsWritten),
                                                 num_elements_written_));
          if (next_elem_.end_of_sequence) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(kNextElem, ".", kEndOfSequence)),
                ""));
          }
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(kNextElem, kSizeSuffix)),
              next_elem_.value.size()));
          for (size_t i = 0; i < next_elem_.value.size(); i++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat(kNextElem, "[", i, "]")),
                next_elem_.value[i]));
          }
          return Status::OK();
        }

        Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
          mutex_lock l(mu_);
          buffer_.clear();
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
          tstring hash_dir;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name(kHashDir), &hash_dir));
          // If the hash dir has changed then we restart writing.
          if (hash_dir != hash_dir_) {
            LOG(INFO) << "Old hash dir from ckpt: " << hash_dir
                      << " is not the same as the new one: " << hash_dir_;
            return Status::OK();
          }
          is_restored_ = true;
          if (reader->Contains(full_name(kEndOfSequence))) {
            end_of_sequence_ = true;
          } else {
            end_of_sequence_ = false;
          }
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kRunId), &run_id_));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kRunDir), &run_dir_));
          {
            int64 temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name(kElementsProduced), &temp));
            elements_produced_ = static_cast<uint64>(temp);
          }
          size_t buffer_size;
          {
            int64 temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kBuffer, kSizeSuffix)), &temp));
            buffer_size = static_cast<size_t>(temp);
          }
          for (size_t i = 0; i < buffer_size; i++) {
            buffer_.emplace_back();
            auto& buffer_element = buffer_.back();
            size_t value_size;
            {
              int64 temp;
              TF_RETURN_IF_ERROR(reader->ReadScalar(
                  full_name(strings::StrCat(kBuffer, "[", i, "]", kSizeSuffix)),
                  &temp));
              value_size = static_cast<size_t>(temp);
            }
            if (reader->Contains(full_name(
                    strings::StrCat(kBuffer, "[", i, "].", kEndOfSequence)))) {
              buffer_element.end_of_sequence = true;
            } else {
              buffer_element.end_of_sequence = false;
            }
            buffer_element.value.reserve(value_size);
            for (size_t j = 0; j < value_size; j++) {
              buffer_element.value.emplace_back();
              TF_RETURN_IF_ERROR(reader->ReadTensor(
                  full_name(strings::StrCat(kBuffer, "[", i, "][", j, "]")),
                  &buffer_element.value.back()));
            }
          }
          {
            int64 temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name(kNextFileIndex), &temp));
            next_file_index_ = static_cast<uint64>(temp);
          }
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNumElementsWritten),
                                                &num_elements_written_));
          size_t next_elem_size;
          {
            int64 temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kNextElem, kSizeSuffix)), &temp));
            next_elem_size = static_cast<size_t>(temp);
          }
          if (reader->Contains(
                  full_name(strings::StrCat(kNextElem, ".", kEndOfSequence)))) {
            next_elem_.end_of_sequence = true;
          } else {
            next_elem_.end_of_sequence = false;
          }
          next_elem_.value.reserve(next_elem_size);
          for (size_t i = 0; i < next_elem_size; i++) {
            next_elem_.value.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat(kNextElem, "[", i, "]")),
                &next_elem_.value.back()));
          }
          return Status::OK();
        }

       private:
        struct BufferElement {
          std::vector<Tensor> value;
          bool end_of_sequence;
        };

        string GetSnapshotFilename() {
          mutex_lock l(mu_);
          string snapshot_data_filename = io::JoinPath(
              run_dir_,
              absl::StrCat(strings::Printf("%08llu", next_file_index_),
                           ".snapshot"));
          next_file_index_++;
          return snapshot_data_filename;
        }

        Status FillBuffer(IteratorContext* ctx) LOCKS_EXCLUDED(mu_) {
          BufferElement elem;
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &elem.value, &elem.end_of_sequence));

          mutex_lock l(mu_);
          next_elem_ = std::move(elem);

          if (next_elem_.end_of_sequence) {
            end_of_sequence_ = true;
            cond_var_.notify_all();
            // Now we wait till all background threads finish.
            while (num_active_threads_ > 0) {
              cond_var_.wait(l);
            }
            return Status::OK();
          }

          // Wait for a space in the buffer_.
          while (!cancelled_ &&
                 buffer_.size() >= dataset()->writer_buffer_size_) {
            cond_var_.wait(l);
          }

          if (cancelled_) {
            return errors::Cancelled(
                "SnapshotDatasetOp::SnapshotWriterIterator::GetNext");
          }

          if (buffer_.size() >= dataset()->writer_buffer_size_) {
            return errors::Internal(
                "Buffer size: ", buffer_.size(), " should be smaller than ",
                "maximum size: ", dataset()->writer_buffer_size_);
          }

          BufferElement elem_copy = next_elem_;
          buffer_.push_back(elem_copy);
          cond_var_.notify_all();
          return Status::OK();
        }

        Status ProcessOneElement(int64* bytes_written,
                                 string* snapshot_data_filename,
                                 std::unique_ptr<WritableFile>* file,
                                 std::unique_ptr<SnapshotWriter>* writer,
                                 bool* end_of_processing) {
          profiler::TraceMe activity(
              absl::StrCat(prefix(), kSeparator, kProcessOneElement),
              profiler::TraceMeLevel::kInfo);
          bool cancelled = false;
          *end_of_processing = false;
          bool produced_elem = false;
          bool snapshot_failed = false;
          BufferElement elem;
          {
            mutex_lock l(mu_);
            // Wait for buffer to not be empty.
            while (!cancelled_ && buffer_.empty() && !end_of_sequence_ &&
                   !snapshot_failed_) {
              cond_var_.wait(l);
            }
            cancelled = cancelled_;
            if (!buffer_.empty()) {
              produced_elem = true;
              std::swap(elem, buffer_.front());
              buffer_.pop_front();
              cond_var_.notify_all();
            } else {
              *end_of_processing = end_of_sequence_;
            }
            snapshot_failed = snapshot_failed_;
          }

          if (cancelled || snapshot_failed) {
            TF_RETURN_IF_ERROR((*writer)->Close());
            TF_RETURN_IF_ERROR((*file)->Sync());
            TF_RETURN_IF_ERROR((*file)->Close());
            if (snapshot_failed) {
              return errors::Internal(
                  "SnapshotDataset::SnapshotWriterIterator snapshot failed");
            }
            return errors::Cancelled(
                "SnapshotDataset::SnapshotWriterIterator cancelled");
          }

          if (produced_elem) {
            experimental::SnapshotRecord record;
            for (auto out_tensor : elem.value) {
              *bytes_written += out_tensor.TotalBytes();
              TensorProto* t = record.add_tensor();
              out_tensor.AsProtoTensorContent(t);
            }

            if (*bytes_written > dataset()->shard_size_bytes_) {
              // If we exceed the shard size, we get a new file and reset.
              TF_RETURN_IF_ERROR((*writer)->Close());
              TF_RETURN_IF_ERROR((*file)->Sync());
              TF_RETURN_IF_ERROR((*file)->Close());
              *snapshot_data_filename = GetSnapshotFilename();
              TF_RETURN_IF_ERROR(Env::Default()->NewAppendableFile(
                  *snapshot_data_filename, file));
              *writer = absl::make_unique<SnapshotWriter>(
                  file->get(), dataset()->compression_);
              *bytes_written = 0;
            }
#if defined(PLATFORM_GOOGLE)
            TF_RETURN_IF_ERROR(
                (*writer)->WriteRecord(record.SerializeAsCord()));
#else   // PLATFORM_GOOGLE
            TF_RETURN_IF_ERROR(
                (*writer)->WriteRecord(record.SerializeAsString()));
#endif  // PLATFORM_GOOGLE
            return Status::OK();
          }

          if (*end_of_processing) {
            TF_RETURN_IF_ERROR((*writer)->Close());
            TF_RETURN_IF_ERROR((*file)->Sync());
            TF_RETURN_IF_ERROR((*file)->Close());
            mutex_lock l(mu_);
            if (!written_final_metadata_file_) {
              experimental::SnapshotMetadataRecord metadata;
              TF_RETURN_IF_ERROR(ReadMetadataFile(hash_dir_, &metadata));

              if (metadata.run_id() == run_id_) {
                metadata.set_finalized(true);
                TF_RETURN_IF_ERROR(WriteMetadataFile(hash_dir_, metadata));
              } else {
                // TODO(frankchn): We lost the race, remove all snapshots.
              }
              written_final_metadata_file_ = true;
              cond_var_.notify_all();
            }
          }
          return Status::OK();
        }

        // Just pulls off elements from the buffer and writes them.
        void WriterThread() {
          auto cleanup = gtl::MakeCleanup([this]() {
            mutex_lock l(mu_);
            --num_active_threads_;
            cond_var_.notify_all();
          });

          int64 bytes_written = 0;
          string snapshot_data_filename = GetSnapshotFilename();
          std::unique_ptr<WritableFile> file;
          Status s =
              Env::Default()->NewAppendableFile(snapshot_data_filename, &file);
          if (!s.ok()) {
            LOG(ERROR) << "Creating " << snapshot_data_filename
                       << " failed: " << s.ToString();
            mutex_lock l(mu_);
            snapshot_failed_ = true;
            cond_var_.notify_all();
            return;
          }
          std::unique_ptr<SnapshotWriter> writer(
              new SnapshotWriter(file.get(), dataset()->compression_));

          bool end_of_processing = false;
          while (!end_of_processing) {
            Status s =
                ProcessOneElement(&bytes_written, &snapshot_data_filename,
                                  &file, &writer, &end_of_processing);
            if (!s.ok()) {
              LOG(INFO) << "Error while writing snapshot data to disk: "
                        << s.ToString();
              mutex_lock l(mu_);
              snapshot_failed_ = true;
              cond_var_.notify_all();
              return;
            }
            mutex_lock l(mu_);
            num_elements_written_++;
          }
        }

        mutex mu_;
        // This condition variable is notified
        // 1. By the background writer threads when an element from the buffer
        //    is consumed.
        // 2. By the main thread when it puts something into the buffer.
        // 3. By the main thread when the destructor is called to cancel.
        // 4. By the background writer threads when any error is encountered
        //    while writing.
        // 5. By the background threads when they finish.
        condition_variable cond_var_;

        BufferElement next_elem_ GUARDED_BY(mu_);
        std::unique_ptr<IteratorBase> input_impl_;

        const string hash_dir_;
        tstring run_id_ GUARDED_BY(mu_);
        tstring run_dir_ GUARDED_BY(mu_);
        bool is_restored_ GUARDED_BY(mu_) = false;

        uint64 elements_produced_ GUARDED_BY(mu_) = 0;
        int64 time_spent_micros_ GUARDED_BY(mu_) = 0;
        int64 bytes_produced_ GUARDED_BY(mu_) = 0;

        std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
        bool snapshot_failed_ GUARDED_BY(mu_) = false;
        bool cancelled_ GUARDED_BY(mu_) = false;
        bool first_call_ GUARDED_BY(mu_) = true;
        bool end_of_sequence_ GUARDED_BY(mu_) = false;
        bool written_final_metadata_file_ GUARDED_BY(mu_) = false;
        uint64 next_file_index_ GUARDED_BY(mu_) = 0;
        std::unique_ptr<thread::ThreadPool> thread_pool_;
        int64 num_active_threads_ GUARDED_BY(mu_) = 0;
        int64 num_elements_written_ = 0;
      };

      class SnapshotPassthroughIterator : public DatasetIterator<Dataset> {
       public:
        explicit SnapshotPassthroughIterator(const Params& params)
            : DatasetIterator<Dataset>(params) {}

        Status Initialize(IteratorContext* ctx) override {
          return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
        }

        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
          return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
        }

       protected:
        Status SaveInternal(IteratorStateWriter* writer) override {
          return SaveInput(writer, input_impl_);
        }

        Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
          return RestoreInput(ctx, reader, input_impl_);
        }

       private:
        std::unique_ptr<IteratorBase> input_impl_;
      };

      string hash_dir_ GUARDED_BY(mu_);
      SnapshotMode state_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> iterator_ GUARDED_BY(mu_);

      mutex mu_;
    };

    const DatasetBase* const input_;
    const tstring dir_;
    const string graph_hash_;

    const string reader_path_prefix_;
    const string writer_path_prefix_;
    const string compression_;

    const uint64 shard_size_bytes_;
    const uint64 pending_snapshot_expiry_seconds_;
    const uint64 num_reader_threads_;
    const uint64 reader_buffer_size_;
    const uint64 num_writer_threads_;
    const uint64 writer_buffer_size_;
    const bool shuffle_on_read_;

    const uint64 seed_;
    const uint64 seed2_;

    const std::string mode_;
    const std::string snapshot_name_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;

  string reader_path_prefix_;
  string writer_path_prefix_;
  string compression_;

  int64 shard_size_bytes_;
  int64 pending_snapshot_expiry_seconds_;
  int64 num_reader_threads_;
  int64 reader_buffer_size_;
  int64 num_writer_threads_;
  int64 writer_buffer_size_;
  bool shuffle_on_read_;

  int64 seed_;
  int64 seed2_;

  std::string mode_;
  std::string snapshot_name_;
};

REGISTER_KERNEL_BUILDER(Name("SnapshotDataset").Device(DEVICE_CPU),
                        SnapshotDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
