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
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/data/experimental/snapshot.pb.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {
namespace {

enum SnapshotMode { READER = 0, WRITER = 1, PASSTHROUGH = 2 };

// Defaults to 10 GiB per shard.
const int64 kDefaultShardSizeBytes = 10L * 1024 * 1024 * 1024;

const size_t kHeaderSize = sizeof(uint64);

const char kSnapshotFilename[] = "snapshot.metadata";

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

    // TODO(frankchn): Remove after ZlibOutputBuffer Cord support is added.
    if (compression_type_.empty()) {
      return dest_->Append(data);
    } else {
      return dest_->Append(data.ToString());
    }
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

string GetCurrentSnapshotDataFilename(uint64 bytes_written,
                                      uint64 shard_size_bytes,
                                      const string& run_dir) {
  uint64_t shard_id = bytes_written / shard_size_bytes;
  return absl::StrCat(run_dir, "/", strings::Printf("%08lu", shard_id),
                      ".snapshot");
}

Status WriteMetadataFile(const string& fingerprint_dir,
                         const experimental::SnapshotMetadataRecord& metadata) {
  string metadata_filename =
      absl::StrCat(fingerprint_dir, "/", kSnapshotFilename);
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(fingerprint_dir));

  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(metadata_filename, &file));

  auto writer = absl::make_unique<SnapshotWriter>(file.get());
  TF_RETURN_IF_ERROR(writer->WriteRecord(metadata.SerializeAsString()));
  TF_RETURN_IF_ERROR(writer->Close());

  return Status::OK();
}

Status ReadMetadataFile(const string& fingerprint_dir,
                        experimental::SnapshotMetadataRecord* metadata) {
  string metadata_filename =
      absl::StrCat(fingerprint_dir, "/", kSnapshotFilename);
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

    if (shard_size_bytes_ == -1) shard_size_bytes_ = kDefaultShardSizeBytes;

    // Default to 1 day expiry for snapshots.
    if (pending_snapshot_expiry_seconds_ == -1) {
      pending_snapshot_expiry_seconds_ = 86400;
    }

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

    GraphDef graph_def;
    OP_REQUIRES_OK(
        ctx, AsGraphDef(ctx, input, SerializationContext({}), &graph_def));

    // TODO(frankchn): Find a better way than DeterministicProtoHash64()
    // This is not deterministic across different builds of binaries right now.
    string graph_fingerprint = strings::StrCat(
        strings::Hex(DeterministicProtoHash64(graph_def), strings::kZeroPad16));

    *output =
        new Dataset(ctx, input, path, graph_fingerprint, reader_path_prefix_,
                    writer_path_prefix_, compression_, shard_size_bytes_,
                    pending_snapshot_expiry_seconds_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input, const string& path,
            const string& graph_fingerprint, const string& reader_path_prefix,
            const string& writer_path_prefix, const string& compression,
            const uint64 shard_size_bytes,
            const uint64 pending_snapshot_expiry_seconds)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          dir_(path),
          graph_fingerprint_(graph_fingerprint),
          reader_path_prefix_(reader_path_prefix),
          writer_path_prefix_(writer_path_prefix),
          compression_(compression),
          shard_size_bytes_(shard_size_bytes),
          pending_snapshot_expiry_seconds_(pending_snapshot_expiry_seconds) {
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
            pending_snapshot_expiry_seconds_attr}},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        fingerprint_dir_ =
            absl::StrCat(dataset()->dir_, "/", dataset()->graph_fingerprint_);

        experimental::SnapshotMetadataRecord metadata;
        Status s = ReadMetadataFile(fingerprint_dir_, &metadata);
        state_ = DetermineOpState(s, metadata,
                                  dataset()->pending_snapshot_expiry_seconds_);

        switch (state_) {
          case WRITER:
            iterator_ = absl::make_unique<SnapshotWriterIterator>(
                SnapshotWriterIterator::Params{
                    dataset(), strings::StrCat(prefix(), "Impl")},
                fingerprint_dir_);
            break;
          case READER:
            iterator_ = absl::make_unique<SnapshotReaderIterator>(
                SnapshotReaderIterator::Params{
                    dataset(), strings::StrCat(prefix(), "Impl")},
                fingerprint_dir_, metadata);
            break;
          case PASSTHROUGH:
            iterator_ = absl::make_unique<SnapshotPassthroughIterator>(
                SnapshotPassthroughIterator::Params{
                    dataset(), strings::StrCat(prefix(), "Impl")});
            break;
        }

        return iterator_->Initialize(ctx);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
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
            const Params& params, const string& fingerprint_dir,
            const experimental::SnapshotMetadataRecord& metadata)
            : DatasetIterator<Dataset>(params),
              fingerprint_dir_(fingerprint_dir),
              metadata_(metadata) {}

        Status Initialize(IteratorContext* ctx) override {
          mutex_lock l(mu_);

          run_id_ = metadata_.run_id();
          run_dir_ = absl::StrCat(fingerprint_dir_, "/", run_id_);
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
          do {
            if (current_reader_) {
#if !defined(PLATFORM_GOOGLE)
              string record_bytes;
              Status s = current_reader_->ReadRecord(&record_bytes);
#else
              absl::Cord record_cord;
              Status s = current_reader_->ReadRecord(&record_cord);
#endif
              if (s.ok()) {
                *end_of_sequence = false;
                experimental::SnapshotRecord record;
#if !defined(PLATFORM_GOOGLE)
                record.ParseFromString(record_bytes);
#else
                record.ParseFromCord(record_cord);
#endif
                int64 num_bytes = 0;
                for (int i = 0; i < record.tensor_size(); ++i) {
                  Tensor t;
                  if (!t.FromProto(record.tensor(i))) {
                    return errors::DataLoss(
                        "Unable to parse Tensor from proto.");
                  }
                  out_tensors->push_back(t);
                  num_bytes += t.TotalBytes();
                }
                absl::Time end = absl::Now();
                absl::Duration d = end - start;
                time_spent_micros_ += absl::ToInt64Microseconds(d);
                kbytes_written_ += static_cast<double>(num_bytes) / 1024.0;
                next_index_++;

                if (next_index_ % 10000 == 0) {
                  LOG(INFO) << "Current read throughput (MBPS): "
                            << (kbytes_written_ * 1000000.0) /
                                   (time_spent_micros_ * 1024.0);
                }
                return Status::OK();
              } else if (!errors::IsOutOfRange(s)) {
                // Report non-EOF errors to the caller.
                return s;
              }
              // Now that we're reached the end of the current file, lets move
              // on to the next file.
              ResetReaderLocked();
              ++current_file_index_;
            }

            if (current_file_index_ == filenames_.size()) {
              *end_of_sequence = true;
              return Status::OK();
            }

            TF_RETURN_IF_ERROR(SetupReaderLocked(ctx->env()));
          } while (true);
        }

       private:
        Status SetupReaderLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          if (current_file_index_ >= filenames_.size()) {
            return errors::InvalidArgument("current_files_index_...");
          }
          const string filename = absl::StrCat(dataset()->reader_path_prefix_,
                                               filenames_[current_file_index_]);
          TF_CHECK_OK(Env::Default()->NewRandomAccessFile(filename,
                                                          &current_read_file_));

          current_reader_ = absl::make_unique<SnapshotReader>(
              current_read_file_.get(), dataset()->compression_);
          return Status::OK();
        }

        void ResetReaderLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          current_reader_.reset();
          current_read_file_.reset();
        }

        const string fingerprint_dir_;
        const experimental::SnapshotMetadataRecord metadata_;
        string run_id_ GUARDED_BY(mu_);
        string run_dir_ GUARDED_BY(mu_);
        std::vector<string> filenames_;

        std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);

        std::unique_ptr<RandomAccessFile> current_read_file_ GUARDED_BY(mu_);
        std::unique_ptr<SnapshotReader> current_reader_ GUARDED_BY(mu_);

        uint64 next_index_ GUARDED_BY(mu_) = 0;
        int64 time_spent_micros_ GUARDED_BY(mu_) = 0;
        double kbytes_written_ GUARDED_BY(mu_) = 0;
        size_t current_file_index_ GUARDED_BY(mu_) = 0;

        mutex mu_;
      };

      class SnapshotWriterIterator : public DatasetIterator<Dataset> {
       public:
        explicit SnapshotWriterIterator(const Params& params,
                                        const string& fingerprint_dir)
            : DatasetIterator<Dataset>(params),
              fingerprint_dir_(fingerprint_dir) {}

        Status Initialize(IteratorContext* ctx) override {
          mutex_lock l(mu_);

          run_id_ = strings::StrCat(
              strings::Hex(random::New64(), strings::kZeroPad4));
          run_dir_ = absl::StrCat(dataset()->writer_path_prefix_,
                                  fingerprint_dir_, "/", run_id_);

          TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(run_dir_));

          experimental::SnapshotMetadataRecord metadata;
          metadata.set_creation_timestamp(Env::Default()->NowMicros());
          metadata.set_graph_fingerprint(dataset()->graph_fingerprint_);
          metadata.set_run_id(run_id_);
          metadata.set_finalized(false);

          TF_RETURN_IF_ERROR(WriteMetadataFile(fingerprint_dir_, metadata));

          return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
        }

        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
          absl::Time start = absl::Now();
          mutex_lock l(mu_);

          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, out_tensors, end_of_sequence));

          if (*end_of_sequence) {
            experimental::SnapshotMetadataRecord metadata;
            TF_RETURN_IF_ERROR(ReadMetadataFile(fingerprint_dir_, &metadata));

            if (metadata.run_id() == run_id_) {
              if (current_writer_) TF_RETURN_IF_ERROR(current_writer_->Close());
              if (current_write_file_)
                TF_RETURN_IF_ERROR(current_write_file_->Close());
              current_writer_.reset();
              current_write_file_.reset();

              current_write_filename_ = "";

              metadata.set_finalized(true);
              TF_RETURN_IF_ERROR(WriteMetadataFile(fingerprint_dir_, metadata));
            } else {
              // TODO(frankchn): We lost the race, remove all snapshots.
            }

            return Status::OK();
          }

          string snapshot_data_filename = GetCurrentSnapshotDataFilename(
              bytes_written_, dataset()->shard_size_bytes_, run_dir_);

          if (current_write_filename_ != snapshot_data_filename) {
            if (current_writer_) TF_RETURN_IF_ERROR(current_writer_->Close());
            if (current_write_file_)
              TF_RETURN_IF_ERROR(current_write_file_->Close());

            current_writer_.reset();
            current_write_file_.reset();

            TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(
                snapshot_data_filename, &current_write_file_));
            current_writer_ = absl::make_unique<SnapshotWriter>(
                current_write_file_.get(), dataset()->compression_);
            current_write_filename_ = snapshot_data_filename;
          }

          experimental::SnapshotRecord record;

          int64 num_bytes = 0;
          for (auto out_tensor : *out_tensors) {
            num_bytes += out_tensor.TotalBytes();
            TensorProto* t = record.add_tensor();
            out_tensor.AsProtoTensorContent(t);
          }

#if defined(PLATFORM_GOOGLE)
          TF_RETURN_IF_ERROR(
              current_writer_->WriteRecord(record.SerializeAsCord()));
#else   // PLATFORM_GOOGLE
          TF_RETURN_IF_ERROR(
              current_writer_->WriteRecord(record.SerializeAsString()));
#endif  // PLATFORM_GOOGLE

          absl::Time end = absl::Now();
          absl::Duration d = end - start;
          time_spent_micros_ += absl::ToInt64Microseconds(d);
          bytes_written_ += num_bytes;

          next_index_++;

          if (next_index_ % 10000 == 0) {
            LOG(INFO) << "Current write throughput (MBPS): "
                      << (bytes_written_ * 1000000.0) /
                             (time_spent_micros_ * 1024.0 * 1024.0);
          }
          return Status::OK();
        }

       private:
        std::unique_ptr<IteratorBase> input_impl_;

        const string fingerprint_dir_;
        string run_id_ GUARDED_BY(mu_);
        string run_dir_ GUARDED_BY(mu_);

        string current_write_filename_ GUARDED_BY(mu_);
        std::unique_ptr<WritableFile> current_write_file_ GUARDED_BY(mu_);
        std::unique_ptr<SnapshotWriter> current_writer_ GUARDED_BY(mu_);

        uint64 next_index_ GUARDED_BY(mu_) = 0;
        int64 time_spent_micros_ GUARDED_BY(mu_) = 0;
        int64 bytes_written_ GUARDED_BY(mu_) = 0;

        mutex mu_;
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

      string fingerprint_dir_;
      SnapshotMode state_;

      std::unique_ptr<IteratorBase> iterator_;
    };

    const DatasetBase* const input_;
    const string dir_;
    const string graph_fingerprint_;

    const string reader_path_prefix_;
    const string writer_path_prefix_;
    const string compression_;

    const uint64 shard_size_bytes_;
    const uint64 pending_snapshot_expiry_seconds_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;

  string reader_path_prefix_;
  string writer_path_prefix_;
  string compression_;

  int64 shard_size_bytes_;
  int64 pending_snapshot_expiry_seconds_;
};

REGISTER_KERNEL_BUILDER(Name("SnapshotDataset").Device(DEVICE_CPU),
                        SnapshotDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
