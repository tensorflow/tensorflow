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
#include "absl/time/clock.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/raw_coding.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#if !defined(IS_SLIM_BUILD)
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
#include "tensorflow/core/protobuf/data/experimental/snapshot.pb.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

enum SnapshotMode { READER = 0, WRITER = 1, PASSTHROUGH = 2 };

// Defaults to 10 GiB per shard.
const int64 kDefaultShardSizeBytes = 10LL * 1024 * 1024 * 1024;

const size_t kHeaderSize = sizeof(uint64);

const char kSnapshotFilename[] = "snapshot.metadata";
constexpr char kSnapshotReaderWorkerPool[] = "snapshot_reader_worker_pool";

class SnapshotWriter {
 public:
  explicit SnapshotWriter(WritableFile* dest, const string& compression_type =
                                                  io::compression::kNone)
      : dest_(dest), compression_type_(compression_type) {
    if (compression_type == io::compression::kGzip) {
#if defined(IS_SLIM_BUILD)
      LOG(ERROR) << "Compression is unsupported on mobile platforms. Turning "
                 << "off compression.";
#else   // IS_SLIM_BUILD
      io::ZlibCompressionOptions zlib_options;
      zlib_options = io::ZlibCompressionOptions::GZIP();

      io::ZlibOutputBuffer* zlib_output_buffer = new io::ZlibOutputBuffer(
          dest, zlib_options.input_buffer_size, zlib_options.output_buffer_size,
          zlib_options);
      TF_CHECK_OK(zlib_output_buffer->Init());
      dest_ = zlib_output_buffer;
      dest_is_owned_ = true;
#endif  // IS_SLIM_BUILD
    }
  }

  Status WriteRecord(const StringPiece& data) {
    char header[kHeaderSize];
    core::EncodeFixed64(header, data.size());
    TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
    return dest_->Append(data);
  }

#if defined(PLATFORM_GOOGLE)
  Status WriteRecord(const absl::Cord& data) {
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
  explicit SnapshotReader(
      RandomAccessFile* file,
      const string& compression_type = io::compression::kNone)
      : input_stream_(new io::RandomAccessInputStream(file)),
        compression_type_(compression_type) {
    if (compression_type_ == io::compression::kGzip) {
#if defined(IS_SLIM_BUILD)
      LOG(ERROR) << "Compression is unsupported on mobile platforms. Turning "
                 << "off compression.";
#else   // IS_SLIM_BUILD
      io::ZlibCompressionOptions zlib_options;
      zlib_options = io::ZlibCompressionOptions::GZIP();

      input_stream_.reset(new io::ZlibInputStream(
          input_stream_.release(), zlib_options.input_buffer_size,
          zlib_options.output_buffer_size, zlib_options, true));
#endif  // IS_SLIM_BUILD
    }
  }

  Status ReadRecord(string* record) {
    string header;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
    uint64 length = core::DecodeFixed64(header.data());
    return input_stream_->ReadNBytes(length, record);
  }

#if defined(PLATFORM_GOOGLE)
  Status ReadRecord(absl::Cord* record) {
    string header;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
    uint64 length = core::DecodeFixed64(header.data());

    if (compression_type_ == io::compression::kNone) {
      return input_stream_->ReadNBytes(length, record);
    } else {
      string tmp_str;
      Status s = input_stream_->ReadNBytes(length, &tmp_str);
      record->Append(tmp_str);
      return s;
    }
  }
#endif

 private:
  std::unique_ptr<io::InputStreamInterface> input_stream_;
  const string compression_type_;
};

Status WriteMetadataFile(const string& hash_dir,
                         const experimental::SnapshotMetadataRecord& metadata) {
  string metadata_filename = absl::StrCat(hash_dir, "/", kSnapshotFilename);
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(hash_dir));

  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(metadata_filename, &file));

  auto writer = absl::make_unique<SnapshotWriter>(file.get());
  TF_RETURN_IF_ERROR(writer->WriteRecord(metadata.SerializeAsString()));
  TF_RETURN_IF_ERROR(writer->Close());

  return Status::OK();
}

Status ReadMetadataFile(const string& hash_dir,
                        experimental::SnapshotMetadataRecord* metadata) {
  string metadata_filename = absl::StrCat(hash_dir, "/", kSnapshotFilename);
  TF_RETURN_IF_ERROR(Env::Default()->FileExists(metadata_filename));

  std::unique_ptr<RandomAccessFile> file;
  TF_CHECK_OK(Env::Default()->NewRandomAccessFile(metadata_filename, &file));

  string record_bytes;
  auto reader = absl::make_unique<SnapshotReader>(file.get());
  TF_CHECK_OK(reader->ReadRecord(&record_bytes));

  metadata->ParseFromString(record_bytes);
  return Status::OK();
}

SnapshotMode DetermineOpState(
    const Status& file_status,
    const experimental::SnapshotMetadataRecord& metadata,
    const uint64 pending_snapshot_expiry_seconds) {
  if (errors::IsNotFound(file_status)) {
    return WRITER;
  }

  if (metadata.finalized()) {
    // File found, snapshot has been finalized.
    return READER;
  }

  if (metadata.creation_timestamp() >=
      (static_cast<int64>(Env::Default()->NowMicros()) -
       pending_snapshot_expiry_seconds * 1000000)) {
    // Someone else is already writing and time has not expired.
    return PASSTHROUGH;
  } else {
    // Time has expired, we write regardless.
    return WRITER;
  }
}

Status GraphHash(const GraphDef& graph_def, std::string* hash) {
  grappler::GraphView gv(&graph_def);

  std::string sink_node_name;
  for (auto& node : graph_def.node()) {
    if (node.op() == "_Retval") {
      sink_node_name = node.name();
      break;
    }
  }

  if (sink_node_name.empty()) {
    return errors::Internal("Cannot find sink node for dataset graph.");
  }

  uint64 hash_int = HashSubgraph(graph_def, gv.GetNode(sink_node_name));
  *hash = strings::StrCat(strings::Hex(hash_int, strings::kZeroPad16));

  return Status::OK();
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
            compression_ == io::compression::kGzip,
        errors::InvalidArgument("compression must be either '' or 'GZIP'."));

    OP_REQUIRES(
        ctx, shard_size_bytes_ >= 1024 * 1024,
        errors::InvalidArgument("shard_size_bytes must be at least 1 MiB."));
    OP_REQUIRES(
        ctx, pending_snapshot_expiry_seconds_ >= 1,
        errors::InvalidArgument(
            "pending_snapshot_expiry_seconds must be at least 1 second."));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    string path;

    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "path", &path));

    SerializationContext::Params params;
    std::vector<std::pair<string, Tensor>> input_list;
    params.input_list = &input_list;
    params.optimization_only = true;

    GraphDef graph_def;
    OP_REQUIRES_OK(
        ctx, AsGraphDef(ctx, input, SerializationContext(params), &graph_def));

    string graph_hash;
    OP_REQUIRES_OK(ctx, GraphHash(graph_def, &graph_hash));

    *output = new Dataset(ctx, input, path, graph_hash, reader_path_prefix_,
                          writer_path_prefix_, compression_, shard_size_bytes_,
                          pending_snapshot_expiry_seconds_, num_reader_threads_,
                          reader_buffer_size_, num_writer_threads_,
                          writer_buffer_size_);
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
            const uint64 num_writer_threads, const uint64 writer_buffer_size)
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
          writer_buffer_size_(writer_buffer_size) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Snapshot")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "SnapshotDatasetOp::Dataset"; }

    int64 Cardinality() const override { return input_->Cardinality(); }

    bool IsStateful() const override { return input_->IsStateful(); }

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
           {"writer_buffer_size", writer_buffer_size_attr}},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        mutex_lock l(mu_);
        hash_dir_ = absl::StrCat(dataset()->dir_, "/", dataset()->graph_hash_);
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (iterator_ == nullptr) {
          experimental::SnapshotMetadataRecord metadata;
          Status s = ReadMetadataFile(hash_dir_, &metadata);
          state_ = DetermineOpState(
              s, metadata, dataset()->pending_snapshot_expiry_seconds_);

          switch (state_) {
            case WRITER:
              iterator_ = absl::make_unique<SnapshotWriterIterator>(
                  SnapshotWriterIterator::Params{
                      dataset(), strings::StrCat(prefix(), "Impl")},
                  hash_dir_);
              break;
            case READER:
              iterator_ = absl::make_unique<SnapshotReaderIterator>(
                  SnapshotReaderIterator::Params{
                      dataset(), strings::StrCat(prefix(), "Impl")},
                  hash_dir_, metadata);
              break;
            case PASSTHROUGH:
              iterator_ = absl::make_unique<SnapshotPassthroughIterator>(
                  SnapshotPassthroughIterator::Params{
                      dataset(), strings::StrCat(prefix(), "Impl")});
              break;
          }
          TF_RETURN_IF_ERROR(iterator_->Initialize(ctx));
        }

        return iterator_->GetNext(ctx, out_tensors, end_of_sequence);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        // TODO(frankchn): Make save iterators work
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        // TODO(frankchn): Make iterator restores work
        return Status::OK();
      }

     private:
      class SnapshotReaderIterator : public DatasetIterator<Dataset> {
       public:
        explicit SnapshotReaderIterator(
            const Params& params, const string& hash_dir,
            const experimental::SnapshotMetadataRecord& metadata)
            : DatasetIterator<Dataset>(params),
              hash_dir_(hash_dir),
              metadata_(metadata) {
          thread_pool_ = absl::make_unique<thread::ThreadPool>(
              Env::Default(), ThreadOptions(), kSnapshotReaderWorkerPool,
              params.dataset->num_reader_threads_, /*low_latency_hint=*/false);
        }

        ~SnapshotReaderIterator() override {
          mutex_lock l(mu_);
          cancelled_ = true;
          cond_var_.notify_all();
        }

        Status Initialize(IteratorContext* ctx) override {
          mutex_lock l(mu_);

          run_id_ = metadata_.run_id();
          run_dir_ = absl::StrCat(hash_dir_, "/", run_id_);
          // Get all the files in the run_dir.
          TF_RETURN_IF_ERROR(ctx->env()->GetMatchingPaths(
              absl::StrCat(run_dir_, "/*"), &filenames_));
          if (filenames_.empty()) {
            return errors::InvalidArgument("Could not find any files in dir: ",
                                           run_dir_);
          }
          std::sort(filenames_.begin(), filenames_.end());
          return Status::OK();
        }

        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
          absl::Time start = absl::Now();
          mutex_lock l(mu_);
          if (!background_threads_started_) {
            for (int i = 0; i < dataset()->num_reader_threads_; ++i) {
              thread_pool_->Schedule([this]() { ReadingFilesLoop(); });
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

          if (!buffer_.empty()) {
            Status s = buffer_.front().status;
            if (s.ok()) {
              *end_of_sequence = false;
              *out_tensors = std::move(buffer_.front().value);

              // Printing some statistics along the way.
              int64 num_bytes = 0;
              for (int i = 0; i < out_tensors->size(); ++i) {
                num_bytes += (*out_tensors)[i].TotalBytes();
              }
              absl::Time end = absl::Now();
              absl::Duration d = end - start;
              time_spent_micros_ += absl::ToInt64Microseconds(d);
              kbytes_read_ += static_cast<double>(num_bytes) / 1024.0;
              elements_produced_++;
              if (elements_produced_ % 10000 == 0) {
                LOG(INFO) << "Current read throughput (MBPS): "
                          << ((kbytes_read_ / 1024.0) /
                              (time_spent_micros_ / 1000000.0));
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
              cond_var_.notify_all();
            } else if (errors::IsOutOfRange(s)) {
              return Status::OK();
            } else {
              return s;
            }
          }
          return Status::OK();
        }

        // Pulls one file off the filenames_ list and reads it through. When
        // all files are read, terminates.
        void ReadingFilesLoop() {
          while (true) {
            string filename = "";
            {
              mutex_lock l(mu_);
              if (next_file_index_ >= filenames_.size()) {
                return;
              }
              filename = absl::StrCat(dataset()->reader_path_prefix_,
                                      filenames_[next_file_index_]);
              VLOG(2) << "Starting to read: " << filename;
              next_file_index_++;
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

        struct BufferElement {
          Status status;
          std::vector<Tensor> value;
        };

        const string hash_dir_;
        const experimental::SnapshotMetadataRecord metadata_;
        string run_id_ GUARDED_BY(mu_);
        string run_dir_ GUARDED_BY(mu_);
        std::vector<string> filenames_;

        uint64 elements_produced_ GUARDED_BY(mu_) = 0;
        int64 time_spent_micros_ GUARDED_BY(mu_) = 0;
        double kbytes_read_ GUARDED_BY(mu_) = 0;
        size_t next_file_index_ GUARDED_BY(mu_) = 0;
        int64 num_files_done_ GUARDED_BY(mu_) = 0;

        std::unique_ptr<thread::ThreadPool> thread_pool_;
        condition_variable cond_var_;
        std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
        bool cancelled_ GUARDED_BY(mu_) = false;
        bool background_threads_started_ GUARDED_BY(mu_) = false;
        bool background_threads_finished_ GUARDED_BY(mu_) = false;

        mutex mu_;
      };

      class SnapshotWriterIterator : public DatasetIterator<Dataset> {
       public:
        explicit SnapshotWriterIterator(const Params& params,
                                        const string& hash_dir)
            : DatasetIterator<Dataset>(params), hash_dir_(hash_dir) {
          thread_pool_ = absl::make_unique<thread::ThreadPool>(
              Env::Default(), ThreadOptions(), "snapshot_writer_pool",
              params.dataset->num_writer_threads_, /*low_latency_hint=*/false);
        }

        ~SnapshotWriterIterator() override {
          mutex_lock l(mu_);
          cancelled_ = true;
          cond_var_.notify_all();
        }

        Status Initialize(IteratorContext* ctx) override {
          mutex_lock l(mu_);

          run_id_ = strings::StrCat(
              strings::Hex(random::New64(), strings::kZeroPad4));
          run_dir_ = absl::StrCat(dataset()->writer_path_prefix_, hash_dir_,
                                  "/", run_id_);

          TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(run_dir_));

          experimental::SnapshotMetadataRecord metadata;
          metadata.set_creation_timestamp(Env::Default()->NowMicros());
          metadata.set_graph_hash(dataset()->graph_hash_);
          metadata.set_run_id(run_id_);
          metadata.set_finalized(false);

          TF_RETURN_IF_ERROR(WriteMetadataFile(hash_dir_, metadata));

          return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
        }

        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
          absl::Time start = absl::Now();

          bool first_call;
          {
            mutex_lock l(mu_);
            first_call = first_call_;
            if (first_call_) {
              for (int i = 0; i < dataset()->num_writer_threads_; ++i) {
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
          if (first_call) {
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

          // Book keeping to report some statistics.
          mutex_lock l(mu_);
          int64 num_bytes = 0;
          for (auto out_tensor : *out_tensors) {
            num_bytes += out_tensor.TotalBytes();
          }

          absl::Time end = absl::Now();
          absl::Duration d = end - start;
          time_spent_micros_ += absl::ToInt64Microseconds(d);
          bytes_produced_ += num_bytes;
          elements_produced_++;

          if (elements_produced_ % 10000 == 0) {
            LOG(INFO) << "Current write throughput (MBPS): "
                      << (bytes_produced_ * 1000000.0) /
                             (time_spent_micros_ * 1024.0 * 1024.0);
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
          string snapshot_data_filename = absl::StrCat(
              run_dir_, "/", strings::Printf("%08llu", next_file_index_),
              ".snapshot");
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
            while (num_threads_finished_ < dataset()->num_writer_threads_) {
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
              TF_RETURN_IF_ERROR((*file)->Close());
              *snapshot_data_filename = GetSnapshotFilename();
              TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(
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
          int64 bytes_written = 0;
          string snapshot_data_filename = GetSnapshotFilename();
          std::unique_ptr<WritableFile> file;
          Status s =
              Env::Default()->NewWritableFile(snapshot_data_filename, &file);
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
          }
          mutex_lock l(mu_);
          num_threads_finished_++;
          cond_var_.notify_all();
        }

        mutex mu_;
        BufferElement next_elem_ GUARDED_BY(mu_);
        std::unique_ptr<IteratorBase> input_impl_;

        const string hash_dir_;
        string run_id_ GUARDED_BY(mu_);
        string run_dir_ GUARDED_BY(mu_);

        uint64 elements_produced_ GUARDED_BY(mu_) = 0;
        int64 time_spent_micros_ GUARDED_BY(mu_) = 0;
        int64 bytes_produced_ GUARDED_BY(mu_) = 0;

        // This condition variable is notified
        // 1. By the background writer threads when an element from the buffer
        //    is consumed.
        // 2. By the main thread when it puts something into the buffer.
        // 3. By the main thread when the destructor is called to cancel.
        // 4. By the background writer threads when any error is encountered
        //    while writing.
        // 5. By the background threads when they finish.
        condition_variable cond_var_;
        std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
        bool snapshot_failed_ GUARDED_BY(mu_) = false;
        bool cancelled_ GUARDED_BY(mu_) = false;
        bool first_call_ GUARDED_BY(mu_) = true;
        bool end_of_sequence_ GUARDED_BY(mu_) = false;
        bool written_final_metadata_file_ GUARDED_BY(mu_) = false;
        uint64 next_file_index_ GUARDED_BY(mu_) = 0;
        int64 num_threads_finished_ GUARDED_BY(mu_) = 0;
        std::unique_ptr<thread::ThreadPool> thread_pool_;
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

       private:
        std::unique_ptr<IteratorBase> input_impl_;
      };

      string hash_dir_ GUARDED_BY(mu_);
      SnapshotMode state_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> iterator_ GUARDED_BY(mu_);

      mutex mu_;
    };

    const DatasetBase* const input_;
    const string dir_;
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
};

REGISTER_KERNEL_BUILDER(Name("SnapshotDataset").Device(DEVICE_CPU),
                        SnapshotDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
