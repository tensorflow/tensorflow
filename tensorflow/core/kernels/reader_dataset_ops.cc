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
#include "tensorflow/core/kernels/dataset.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/record_reader.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following ops.

class TextLineDatasetOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
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

    DatasetBase* dataset = new Dataset(std::move(filenames));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    ResourceHandle handle = MakeResourceHandle<DatasetBase>(
        ctx, ctx->step_container()->name(), name());
    OP_REQUIRES_OK(ctx, CreateResource(ctx, handle, dataset));
    output->scalar<ResourceHandle>()() = handle;
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(std::vector<string> filenames)
        : filenames_(std::move(filenames)) {}

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      return std::unique_ptr<IteratorBase>(new Iterator(this));
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

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a file, so try to read the next line.
          if (input_buffer_) {
            string line_contents;
            Status s = input_buffer_->ReadLine(&line_contents);
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
          TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(
              dataset()->filenames_[current_file_index_], &file_));
          input_buffer_.reset(new io::InputBuffer(file_.get(), kBufferSize));
        } while (true);
      }

     private:
      // TODO(mrry): Make this configurable via an attr on the dataset op?
      // Or maybe via a data input?
      enum { kBufferSize = 256 << 10 /* 256 kB */ };

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<RandomAccessFile> file_
          GUARDED_BY(mu_);  // must outlive input_buffer_
      std::unique_ptr<io::InputBuffer> input_buffer_ GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
  };
};

REGISTER_KERNEL_BUILDER(Name("TextLineDataset").Device(DEVICE_CPU),
                        TextLineDatasetOp);

class FixedLengthRecordDatasetOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
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

    const Tensor* header_bytes_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("header_bytes", &header_bytes_tensor));
    OP_REQUIRES(ctx, header_bytes_tensor->dims() == 0,
                errors::InvalidArgument("`header_bytes` must be a scalar."));
    const int64 header_bytes = header_bytes_tensor->scalar<int64>()();

    const Tensor* record_bytes_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("record_bytes", &record_bytes_tensor));
    OP_REQUIRES(ctx, record_bytes_tensor->dims() == 0,
                errors::InvalidArgument("`record_bytes` must be a scalar."));
    const int64 record_bytes = record_bytes_tensor->scalar<int64>()();

    const Tensor* footer_bytes_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("footer_bytes", &footer_bytes_tensor));
    OP_REQUIRES(ctx, footer_bytes_tensor->dims() == 0,
                errors::InvalidArgument("`footer_bytes` must be a scalar."));
    const int64 footer_bytes = footer_bytes_tensor->scalar<int64>()();

    DatasetBase* dataset = new Dataset(std::move(filenames), header_bytes,
                                       record_bytes, footer_bytes);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    ResourceHandle handle = MakeResourceHandle<DatasetBase>(
        ctx, ctx->step_container()->name(), name());
    OP_REQUIRES_OK(ctx, CreateResource(ctx, handle, dataset));
    output->scalar<ResourceHandle>()() = handle;
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(std::vector<string> filenames, int64 header_bytes,
                     int64 record_bytes, int64 footer_bytes)
        : filenames_(std::move(filenames)),
          header_bytes_(header_bytes),
          record_bytes_(record_bytes),
          footer_bytes_(footer_bytes) {}

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      return std::unique_ptr<IteratorBase>(new Iterator(this));
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

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
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
          TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(
              dataset()->filenames_[current_file_index_], &file_));
          input_buffer_.reset(new io::InputBuffer(file_.get(), kBufferSize));
          TF_RETURN_IF_ERROR(
              input_buffer_->SkipNBytes(dataset()->header_bytes_));
        } while (true);
      }

     private:
      // TODO(mrry): Make this configurable via an attr on the dataset op?
      // Or maybe via a data input?
      enum { kBufferSize = 256 << 10 /* 256 kB */ };

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
  };
};

REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordDataset").Device(DEVICE_CPU),
                        FixedLengthRecordDatasetOp);

class TFRecordDatasetOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
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

    const Tensor* compression_type_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->input("compression_type", &compression_type_tensor));
    OP_REQUIRES(
        ctx, compression_type_tensor->dims() == 0,
        errors::InvalidArgument("`compression_type` must be a scalar."));
    const string& compression_type =
        compression_type_tensor->scalar<string>()();

    DatasetBase* dataset = new Dataset(std::move(filenames), compression_type);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    ResourceHandle handle = MakeResourceHandle<DatasetBase>(
        ctx, ctx->step_container()->name(), name());
    OP_REQUIRES_OK(ctx, CreateResource(ctx, handle, dataset));
    output->scalar<ResourceHandle>()() = handle;
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(std::vector<string> filenames,
                     const string& compression_type)
        : filenames_(std::move(filenames)),
          options_(io::RecordReaderOptions::CreateRecordReaderOptions(
              compression_type)) {}

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      return std::unique_ptr<IteratorBase>(new Iterator(this));
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

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a file, so try to read the next record.
          if (reader_) {
            Tensor result_tensor(cpu_allocator(), DT_STRING, {});
            Status s = reader_->ReadRecord(&offset_,
                                           &result_tensor.scalar<string>()());
            if (s.ok()) {
              out_tensors->emplace_back(std::move(result_tensor));
              *end_of_sequence = false;
              return Status::OK();
            } else if (!errors::IsOutOfRange(s)) {
              return s;
            }

            // We have reached the end of the current file, so maybe
            // move on to next file.
            reader_.reset();
            file_.reset();
            ++current_file_index_;
          }

          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          // Actually move on to next file.
          const string& next_filename =
              dataset()->filenames_[current_file_index_];
          TF_RETURN_IF_ERROR(
              ctx->env()->NewRandomAccessFile(next_filename, &file_));
          reader_.reset(new io::RecordReader(file_.get(), dataset()->options_));
          offset_ = 0;
        } while (true);
      }

     private:
      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      uint64 offset_ GUARDED_BY(mu_) = 0;

      // `reader_` will borrow the object that `file_` points to, so
      // we must destroy `reader_` before `file_`.
      std::unique_ptr<RandomAccessFile> file_ GUARDED_BY(mu_);
      std::unique_ptr<io::RecordReader> reader_ GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
    io::RecordReaderOptions options_;
  };
};

REGISTER_KERNEL_BUILDER(Name("TFRecordDataset").Device(DEVICE_CPU),
                        TFRecordDatasetOp);

}  // namespace

}  // namespace tensorflow
