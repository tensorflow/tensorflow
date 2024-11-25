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
#include "tensorflow/core/kernels/data/tf_record_dataset_op.h"

#include <cstdint>

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following ops.

/* static */ constexpr const char* const TFRecordDatasetOp::kDatasetType;
/* static */ constexpr const char* const TFRecordDatasetOp::kFileNames;
/* static */ constexpr const char* const TFRecordDatasetOp::kCompressionType;
/* static */ constexpr const char* const TFRecordDatasetOp::kBufferSize;
/* static */ constexpr const char* const TFRecordDatasetOp::kByteOffsets;

constexpr char kTFRecordDataset[] = "TFRecordDataset";
constexpr char kCurrentFileIndex[] = "current_file_index";
constexpr char kOffset[] = "offset";
constexpr char kGcsFsPrefix[] = "gs://";
constexpr char kS3FsPrefix[] = "s3://";
constexpr int64_t kUnspecifiedBufferSize = -1;
constexpr int64_t kDefaultBufferSize = 256LL << 10;  // 256KB
constexpr int64_t kCloudTpuBlockSize = 127LL << 20;  // 127MB.
constexpr int64_t kS3BlockSize = kCloudTpuBlockSize;

bool is_cloud_tpu_gcs_fs() {
#if (defined(PLATFORM_CLOUD_TPU) && defined(TPU_GCS_FS)) || \
    defined(LIBTPU_ON_GCE)
  return true;
#endif
  return false;
}

class TFRecordDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx, std::vector<string> filenames,
                   const string& compression_type, int64_t buffer_size,
                   std::vector<int64_t> byte_offsets, int op_version)
      : DatasetBase(DatasetContext(ctx)),
        filenames_(std::move(filenames)),
        compression_type_(compression_type),
        options_(io::RecordReaderOptions::CreateRecordReaderOptions(
            compression_type)),
        byte_offsets_(std::move(byte_offsets)),
        op_version_(op_version) {
    if (buffer_size > 0) {
      options_.buffer_size = buffer_size;
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
  }

  const DataTypeVector& output_dtypes() const override {
    static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
    return *dtypes;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    static std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({{}});
    return *shapes;
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.op_version = op_version_;
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override { return absl::OkStatus(); }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* filenames = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
    Node* compression_type = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(compression_type_, &compression_type));
    Node* buffer_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(options_.buffer_size, &buffer_size));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {filenames, compression_type, buffer_size}, output));
    Node* byte_offsets = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(byte_offsets_, &byte_offsets));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      out_tensors->reserve(1);
      mutex_lock l(mu_);
      do {
        // We are currently processing a file, so try to read the next record.
        if (reader_) {
          out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                    TensorShape({}));
          absl::Status s =
              reader_->ReadRecord(&out_tensors->back().scalar<tstring>()());
          if (s.ok()) {
            static monitoring::CounterCell* bytes_counter =
                metrics::GetTFDataBytesReadCounter(kDatasetType);
            bytes_counter->IncrementBy(
                out_tensors->back().scalar<tstring>()().size());
            *end_of_sequence = false;
            return absl::OkStatus();
          }
          out_tensors->pop_back();
          if (!errors::IsOutOfRange(s)) {
            // In case of other errors e.g., DataLoss, we still move forward
            // the file index so that it works with ignore_errors.
            // Otherwise the same file will repeat.
            ResetStreamsLocked();
            ++current_file_index_;
            return s;
          }

          // We have reached the end of the current file, so maybe move on to
          // next file.
          ResetStreamsLocked();
          ++current_file_index_;
        }

        // Iteration ends when there are no more files to process.
        if (current_file_index_ == dataset()->filenames_.size()) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }

        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      } while (true);
    }

    absl::Status SkipInternal(IteratorContext* ctx, int num_to_skip,
                              bool* end_of_sequence,
                              int* num_skipped) override {
      *num_skipped = 0;
      mutex_lock l(mu_);
      do {
        // We are currently processing a file, so try to skip reading
        // the next (num_to_skip - *num_skipped) record.
        if (reader_) {
          int last_num_skipped;
          absl::Status s = reader_->SkipRecords(num_to_skip - *num_skipped,
                                                &last_num_skipped);
          *num_skipped += last_num_skipped;
          if (s.ok()) {
            *end_of_sequence = false;
            return absl::OkStatus();
          }
          if (!errors::IsOutOfRange(s)) {
            // In case of other errors e.g., DataLoss, we still move forward
            // the file index so that it works with ignore_errors.
            // Otherwise the same file will repeat.
            ResetStreamsLocked();
            ++current_file_index_;
            return s;
          }

          // We have reached the end of the current file, so maybe move on to
          // next file.
          ResetStreamsLocked();
          ++current_file_index_;
        }

        // Iteration ends when there are no more files to process.
        if (current_file_index_ == dataset()->filenames_.size()) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }

        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      } while (true);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kCurrentFileIndex,
                                             current_file_index_));

      if (reader_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(prefix(), kOffset, reader_->TellOffset()));
      }
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      ResetStreamsLocked();
      int64_t current_file_index;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kCurrentFileIndex, &current_file_index));
      current_file_index_ = size_t(current_file_index);
      if (reader->Contains(prefix(), kOffset)) {
        int64_t offset;
        TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kOffset, &offset));
        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        TF_RETURN_IF_ERROR(reader_->SeekOffset(offset));
      }
      return absl::OkStatus();
    }

   private:
    // Sets up reader streams to read from the file at `current_file_index_`.
    absl::Status SetupStreamsLocked(Env* env) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (current_file_index_ >= dataset()->filenames_.size()) {
        return errors::InvalidArgument(
            "current_file_index_:", current_file_index_,
            " >= filenames_.size():", dataset()->filenames_.size());
      }

      // Actually move on to next file.
      tsl::profiler::TraceMe traceme(
          [&, current_file_index = current_file_index_] {
            return tsl::profiler::TraceMeEncode(
                "TFRecordDatasetOp::Iterator::SetupStreamsLocked",
                {{"filename", dataset()->filenames_[current_file_index]}});
          },
          tsl::profiler::kInfo);

      TF_RETURN_IF_ERROR(env->NewRandomAccessFile(
          TranslateFileName(dataset()->filenames_[current_file_index_]),
          &file_));
      reader_ = std::make_unique<io::SequentialRecordReader>(
          file_.get(), dataset()->options_);
      if (!dataset()->byte_offsets_.empty()) {
        TF_RETURN_IF_ERROR(
            reader_->SeekOffset(dataset()->byte_offsets_[current_file_index_]));
      }
      return absl::OkStatus();
    }

    // Resets all reader streams.
    void ResetStreamsLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      reader_.reset();
      file_.reset();
    }

    mutex mu_;
    size_t current_file_index_ TF_GUARDED_BY(mu_) = 0;

    // `reader_` will borrow the object that `file_` points to, so
    // we must destroy `reader_` before `file_`.
    std::unique_ptr<RandomAccessFile> file_ TF_GUARDED_BY(mu_);
    std::unique_ptr<io::SequentialRecordReader> reader_ TF_GUARDED_BY(mu_);
  };

  const std::vector<string> filenames_;
  const tstring compression_type_;
  io::RecordReaderOptions options_;
  const std::vector<int64_t> byte_offsets_;
  const int op_version_;
};

TFRecordDatasetOp::TFRecordDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx),
      op_version_(ctx->def().op() == kTFRecordDataset ? 1 : 2) {}

void TFRecordDatasetOp::MakeDataset(OpKernelContext* ctx,
                                    DatasetBase** output) {
  const Tensor* filenames_tensor;
  OP_REQUIRES_OK(ctx, ctx->input(kFileNames, &filenames_tensor));
  OP_REQUIRES(
      ctx, filenames_tensor->dims() <= 1,
      errors::InvalidArgument("`filenames` must be a scalar or a vector."));

  bool is_gcs_fs = true;
  bool is_s3_fs = true;
  std::vector<string> filenames;
  filenames.reserve(filenames_tensor->NumElements());
  for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
    VLOG(2) << "Reading file: " << filenames_tensor->flat<tstring>()(i);
    filenames.push_back(filenames_tensor->flat<tstring>()(i));
    is_gcs_fs &= absl::StartsWith(filenames[i], kGcsFsPrefix);
    is_s3_fs &= absl::StartsWith(filenames[i], kS3FsPrefix);
    metrics::RecordTFDataFilename(kDatasetType, filenames[i]);
  }
  LogFilenames(filenames);

  tstring compression_type;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, kCompressionType,
                                                   &compression_type));

  int64_t buffer_size = kUnspecifiedBufferSize;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(ctx,
              (buffer_size == kUnspecifiedBufferSize) || (buffer_size >= 0),
              errors::InvalidArgument(
                  "`buffer_size` must be >= 0 (0 == no buffering)"));

  std::vector<int64_t> byte_offsets;
  if (op_version_ > 1) {
    const Tensor* byte_offsets_tensor;
    OP_REQUIRES_OK(ctx, ctx->input(kByteOffsets, &byte_offsets_tensor));
    OP_REQUIRES(ctx, byte_offsets_tensor->dims() <= 1,
                absl::InvalidArgumentError(
                    "`byte_offsets` must be a scalar or a vector."));
    OP_REQUIRES(ctx, byte_offsets_tensor->dims() == filenames_tensor->dims(),
                absl::InvalidArgumentError(
                    "`byte_offsets` must be of same size as `filenames`"));
    byte_offsets.reserve(byte_offsets_tensor->NumElements());
    for (int i = 0; i < byte_offsets_tensor->NumElements(); ++i) {
      byte_offsets.push_back(byte_offsets_tensor->flat<int64_t>()(i));
    }
  }

  if (buffer_size == kUnspecifiedBufferSize) {
    if (is_gcs_fs && is_cloud_tpu_gcs_fs() &&
        buffer_size < kCloudTpuBlockSize) {
      LOG_FIRST_N(WARNING, 1)
          << "User buffer size is too small for reading Cloud TPU "
          << "TFRecords stored in GCS. Overriding " << buffer_size
          << " to the minimum recommended buffer_size = " << kCloudTpuBlockSize;
      buffer_size = kCloudTpuBlockSize;
    } else if (is_s3_fs && buffer_size < kS3BlockSize) {
      LOG_FIRST_N(WARNING, 1)
          << "User buffer size is too small for reading "
          << "TFRecords stored in S3. Overriding " << buffer_size
          << " to the minimum recommended buffer_size = " << kS3BlockSize;
      buffer_size = kS3BlockSize;
    } else {
      LOG_FIRST_N(INFO, 1)
          << "TFRecordDataset `buffer_size` is unspecified, default to "
          << kDefaultBufferSize;
      buffer_size = kDefaultBufferSize;
    }
  } else {
    LOG_FIRST_N(INFO, 1)
        << "The default buffer size is " << kDefaultBufferSize
        << ", which is overridden by the user specified `buffer_size` of "
        << buffer_size;
  }

  *output = new Dataset(ctx, std::move(filenames), compression_type,
                        buffer_size, std::move(byte_offsets), op_version_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("TFRecordDataset").Device(DEVICE_CPU),
                        TFRecordDatasetOp);

REGISTER_KERNEL_BUILDER(Name("TFRecordDatasetV2").Device(DEVICE_CPU),
                        TFRecordDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
