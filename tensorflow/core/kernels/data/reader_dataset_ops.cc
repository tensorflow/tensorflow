/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following ops.

class TextLineDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    string compression_type;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "compression_type",
                                                    &compression_type));

    int64 buffer_size = -1;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(
        ctx, buffer_size >= 0,
        errors::InvalidArgument("`buffer_size` must be >= 0 (0 == default)"));

    io::ZlibCompressionOptions zlib_compression_options =
        io::ZlibCompressionOptions::DEFAULT();
    if (compression_type == "ZLIB") {
      zlib_compression_options = io::ZlibCompressionOptions::DEFAULT();
    } else if (compression_type == "GZIP") {
      zlib_compression_options = io::ZlibCompressionOptions::GZIP();
    } else {
      OP_REQUIRES(ctx, compression_type.empty(),
                  errors::InvalidArgument("Unsupported compression_type."));
    }

    if (buffer_size != 0) {
      // Set the override size.
      zlib_compression_options.input_buffer_size = buffer_size;
    }

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    *output = new Dataset(ctx, std::move(filenames), compression_type,
                          zlib_compression_options);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<string> filenames,
            const string& compression_type,
            const io::ZlibCompressionOptions& options)
        : GraphDatasetBase(ctx),
          filenames_(std::move(filenames)),
          compression_type_(compression_type),
          use_compression_(!compression_type.empty()),
          options_(options) {}

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::TextLine")}));
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

    string DebugString() override { return "TextLineDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      Node* compression_type = nullptr;
      Node* buffer_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      TF_RETURN_IF_ERROR(b->AddScalar(compression_type_, &compression_type));
      TF_RETURN_IF_ERROR(
          b->AddScalar(options_.input_buffer_size, &buffer_size));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {filenames, compression_type, buffer_size}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a file, so try to read the next line.
          if (buffered_input_stream_) {
            string line_contents;
            Status s = buffered_input_stream_->ReadLine(&line_contents);

            if (s.ok()) {
              // Produce the line as output.
              Tensor line_tensor(cpu_allocator(), DT_STRING, {});
              line_tensor.scalar<string>()() = line_contents;
              out_tensors->emplace_back(std::move(line_tensor));
              *end_of_sequence = false;
              return Status::OK();
            } else if (!errors::IsOutOfRange(s)) {
              // Report non-EOF errors to the caller.
              return s;
            }
            // We have reached the end of the current file, so maybe
            // move on to next file.
            ResetStreamsLocked();
            ++current_file_index_;
          }

          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        } while (true);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_file_index"),
                                               current_file_index_));

        // `buffered_input_stream_` is empty if
        // 1. GetNext has not been called even once.
        // 2. All files have been read and iterator has been exhausted.
        if (buffered_input_stream_) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name("current_pos"), buffered_input_stream_->Tell()));
        }
        return Status::OK();
      }

      Status RestoreInternal(OpKernelContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        ResetStreamsLocked();
        int64 current_file_index;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_file_index"),
                                              &current_file_index));
        current_file_index_ = size_t(current_file_index);
        // The key "current_pos" is written only if the iterator was saved
        // with an open file.
        if (reader->Contains(full_name("current_pos"))) {
          int64 current_pos;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("current_pos"), &current_pos));

          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
          TF_RETURN_IF_ERROR(buffered_input_stream_->Seek(current_pos));
        }
        return Status::OK();
      }

     private:
      // Sets up reader streams to read from the file at `current_file_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }

        // Actually move on to next file.
        TF_RETURN_IF_ERROR(env->NewRandomAccessFile(
            dataset()->filenames_[current_file_index_], &file_));
        input_stream_.reset(
            new io::RandomAccessInputStream(file_.get(), false));

        if (dataset()->use_compression_) {
          zlib_input_stream_.reset(new io::ZlibInputStream(
              input_stream_.get(), dataset()->options_.input_buffer_size,
              dataset()->options_.input_buffer_size, dataset()->options_));
          buffered_input_stream_.reset(new io::BufferedInputStream(
              zlib_input_stream_.get(), dataset()->options_.input_buffer_size,
              false));
        } else {
          buffered_input_stream_.reset(new io::BufferedInputStream(
              input_stream_.get(), dataset()->options_.input_buffer_size,
              false));
        }
        return Status::OK();
      }

      // Resets all reader streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        input_stream_.reset();
        zlib_input_stream_.reset();
        buffered_input_stream_.reset();
        file_.reset();
      }

      mutex mu_;
      std::unique_ptr<io::RandomAccessInputStream> input_stream_
          GUARDED_BY(mu_);
      std::unique_ptr<io::ZlibInputStream> zlib_input_stream_ GUARDED_BY(mu_);
      std::unique_ptr<io::BufferedInputStream> buffered_input_stream_
          GUARDED_BY(mu_);
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<RandomAccessFile> file_
          GUARDED_BY(mu_);  // must outlive input_stream_
    };

    const std::vector<string> filenames_;
    const string compression_type_;
    const bool use_compression_;
    const io::ZlibCompressionOptions options_;
  };
};

REGISTER_KERNEL_BUILDER(Name("TextLineDataset").Device(DEVICE_CPU),
                        TextLineDatasetOp);

class FixedLengthRecordDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    int64 header_bytes = -1;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "header_bytes", &header_bytes));
    OP_REQUIRES(ctx, header_bytes >= 0,
                errors::InvalidArgument("`header_bytes` must be >= 0"));

    int64 record_bytes = -1;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "record_bytes", &record_bytes));
    OP_REQUIRES(ctx, record_bytes > 0,
                errors::InvalidArgument("`record_bytes` must be > 0"));

    int64 footer_bytes = -1;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "footer_bytes", &footer_bytes));
    OP_REQUIRES(ctx, footer_bytes >= 0,
                errors::InvalidArgument("`footer_bytes` must be >= 0"));

    int64 buffer_size = -1;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(ctx, buffer_size >= 0,
                errors::InvalidArgument("`buffer_size` must be >= 0"));
    if (buffer_size == 0) {
      buffer_size = 256 << 10;  // 256 kB as default.
    }

    *output = new Dataset(ctx, std::move(filenames), header_bytes, record_bytes,
                          footer_bytes, buffer_size);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, std::vector<string> filenames,
                     int64 header_bytes, int64 record_bytes, int64 footer_bytes,
                     int64 buffer_size)
        : GraphDatasetBase(ctx),
          filenames_(std::move(filenames)),
          header_bytes_(header_bytes),
          record_bytes_(record_bytes),
          footer_bytes_(footer_bytes),
          buffer_size_(buffer_size) {}

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::FixedLengthRecord")}));
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

    string DebugString() override {
      return "FixedLengthRecordDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      Node* header_bytes = nullptr;
      Node* record_bytes = nullptr;
      Node* footer_bytes = nullptr;
      Node* buffer_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      TF_RETURN_IF_ERROR(b->AddScalar(header_bytes_, &header_bytes));
      TF_RETURN_IF_ERROR(b->AddScalar(record_bytes_, &record_bytes));
      TF_RETURN_IF_ERROR(b->AddScalar(footer_bytes_, &footer_bytes));
      TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          {filenames, header_bytes, record_bytes, footer_bytes, buffer_size},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a file, so try to read the next record.
          if (input_buffer_) {
            const int64 current_pos = input_buffer_->Tell();
            DCHECK_GE(file_pos_limit_, 0);
            if (current_pos < file_pos_limit_) {
              string record;
              TF_RETURN_IF_ERROR(
                  input_buffer_->ReadNBytes(dataset()->record_bytes_, &record));
              // Produce the record as output.
              Tensor record_tensor(cpu_allocator(), DT_STRING, {});
              record_tensor.scalar<string>()() = record;
              out_tensors->emplace_back(std::move(record_tensor));
              *end_of_sequence = false;
              return Status::OK();
            }

            // We have reached the end of the current file, so maybe
            // move on to next file.
            input_buffer_.reset();
            file_.reset();
            ++current_file_index_;
          }

          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          // Actually move on to next file.
          uint64 file_size;
          TF_RETURN_IF_ERROR(ctx->env()->GetFileSize(
              dataset()->filenames_[current_file_index_], &file_size));
          file_pos_limit_ = file_size - dataset()->footer_bytes_;

          uint64 body_size =
              file_size - (dataset()->header_bytes_ + dataset()->footer_bytes_);

          if (body_size % dataset()->record_bytes_ != 0) {
            return errors::InvalidArgument(
                "Excluding the header (", dataset()->header_bytes_,
                " bytes) and footer (", dataset()->footer_bytes_,
                " bytes), input file \"",
                dataset()->filenames_[current_file_index_],
                "\" has body length ", body_size,
                " bytes, which is not an exact multiple of the record length (",
                dataset()->record_bytes_, " bytes).");
          }
          TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(
              dataset()->filenames_[current_file_index_], &file_));
          input_buffer_.reset(
              new io::InputBuffer(file_.get(), dataset()->buffer_size_));
          TF_RETURN_IF_ERROR(
              input_buffer_->SkipNBytes(dataset()->header_bytes_));
        } while (true);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_file_index"),
                                               current_file_index_));

        // `input_buffer_` is empty if
        // 1. GetNext has not been called even once.
        // 2. All files have been read and iterator has been exhausted.
        int64 current_pos = input_buffer_ ? input_buffer_->Tell() : -1;
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("current_pos"), current_pos));
        return Status::OK();
      }

      Status RestoreInternal(OpKernelContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        int64 current_file_index;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_file_index"),
                                              &current_file_index));
        current_file_index_ = size_t(current_file_index);
        int64 current_pos;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("current_pos"), &current_pos));

        // Seek to current_pos.
        input_buffer_.reset();
        file_.reset();
        if (current_pos >= 0) {  // There was an active input_buffer_.
          uint64 file_size;
          TF_RETURN_IF_ERROR(ctx->env()->GetFileSize(
              dataset()->filenames_[current_file_index_], &file_size));
          file_pos_limit_ = file_size - dataset()->footer_bytes_;
          TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(
              dataset()->filenames_[current_file_index_], &file_));
          input_buffer_.reset(
              new io::InputBuffer(file_.get(), dataset()->buffer_size_));
          TF_RETURN_IF_ERROR(input_buffer_->Seek(current_pos));
        }

        return Status::OK();
      }

     private:
      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<RandomAccessFile> file_
          GUARDED_BY(mu_);  // must outlive input_buffer_
      std::unique_ptr<io::InputBuffer> input_buffer_ GUARDED_BY(mu_);
      int64 file_pos_limit_ GUARDED_BY(mu_) = -1;
    };

    const std::vector<string> filenames_;
    const int64 header_bytes_;
    const int64 record_bytes_;
    const int64 footer_bytes_;
    const int64 buffer_size_;
  };
};

REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordDataset").Device(DEVICE_CPU),
                        FixedLengthRecordDatasetOp);

class TFRecordDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    string compression_type;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "compression_type",
                                                    &compression_type));

    int64 buffer_size = -1;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(ctx, buffer_size >= 0,
                errors::InvalidArgument(
                    "`buffer_size` must be >= 0 (0 == no buffering)"));

    *output =
        new Dataset(ctx, std::move(filenames), compression_type, buffer_size);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, std::vector<string> filenames,
                     const string& compression_type, int64 buffer_size)
        : GraphDatasetBase(ctx),
          filenames_(std::move(filenames)),
          compression_type_(compression_type),
          options_(io::RecordReaderOptions::CreateRecordReaderOptions(
              compression_type)) {
      if (buffer_size > 0) {
        options_.buffer_size = buffer_size;
      }
    }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::TFRecord")}));
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

    string DebugString() override { return "TFRecordDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      Node* compression_type = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(compression_type_, &compression_type));
      Node* buffer_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(options_.buffer_size, &buffer_size));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {filenames, compression_type, buffer_size}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a file, so try to read the next record.
          if (reader_) {
            Tensor result_tensor(cpu_allocator(), DT_STRING, {});
            Status s = reader_->ReadRecord(&result_tensor.scalar<string>()());
            if (s.ok()) {
              out_tensors->emplace_back(std::move(result_tensor));
              *end_of_sequence = false;
              return Status::OK();
            } else if (!errors::IsOutOfRange(s)) {
              return s;
            }

            // We have reached the end of the current file, so maybe
            // move on to next file.
            ResetStreamsLocked();
            ++current_file_index_;
          }

          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        } while (true);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_file_index"),
                                               current_file_index_));

        if (reader_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("offset"), reader_->TellOffset()));
        }
        return Status::OK();
      }

      Status RestoreInternal(OpKernelContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        ResetStreamsLocked();
        int64 current_file_index;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_file_index"),
                                              &current_file_index));
        current_file_index_ = size_t(current_file_index);
        if (reader->Contains(full_name("offset"))) {
          int64 offset;
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("offset"), &offset));
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
          TF_RETURN_IF_ERROR(reader_->SeekOffset(offset));
        }
        return Status::OK();
      }

     private:
      // Sets up reader streams to read from the file at `current_file_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }

        // Actually move on to next file.
        const string& next_filename =
            dataset()->filenames_[current_file_index_];
        TF_RETURN_IF_ERROR(env->NewRandomAccessFile(next_filename, &file_));
        reader_.reset(
            new io::SequentialRecordReader(file_.get(), dataset()->options_));
        return Status::OK();
      }

      // Resets all reader streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        reader_.reset();
        file_.reset();
      }

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;

      // `reader_` will borrow the object that `file_` points to, so
      // we must destroy `reader_` before `file_`.
      std::unique_ptr<RandomAccessFile> file_ GUARDED_BY(mu_);
      std::unique_ptr<io::SequentialRecordReader> reader_ GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
    const string compression_type_;
    io::RecordReaderOptions options_;
  };
};

REGISTER_KERNEL_BUILDER(Name("TFRecordDataset").Device(DEVICE_CPU),
                        TFRecordDatasetOp);

}  // namespace

}  // namespace tensorflow
