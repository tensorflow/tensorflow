/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// TODO posix specific
#include <arpa/inet.h>
#include <sys/socket.h>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"
#include "arrow/util/io-util.h"
#include "tensorflow/core/framework/dataset.h"

#define CHECK_ARROW(arrow_status)             \
  do {                                        \
    arrow::Status _s = (arrow_status);        \
    if (!_s.ok()) {                           \
      return errors::Internal(_s.ToString()); \
    }                                         \
  } while (false)

namespace tensorflow {

// Class to wrap a socket as a readable Arrow InputStream
class SocketStream : public arrow::io::InputStream {
 public:
  SocketStream(int sock) : sock_(sock), pos_(0) {}
  ~SocketStream() override {}

  arrow::Status Close() override {
    close(sock_);
    return arrow::Status::OK();
  }

  arrow::Status Tell(int64_t* position) const override {
    *position = pos_;
    return arrow::Status::OK();
  }

  arrow::Status Read(int64_t nbytes, int64_t* bytes_read, void* out) override {
    int status = recv(sock_, out, nbytes, MSG_WAITALL);
    // if (status == 0) socket closed
    if (status == -1) {
      return arrow::Status::IOError("error reading from socket");
    }
    *bytes_read = nbytes;
    pos_ += *bytes_read;
    return arrow::Status::OK();
  }

  arrow::Status Read(int64_t nbytes,
                     std::shared_ptr<arrow::Buffer>* out) override {
    std::shared_ptr<arrow::ResizableBuffer> buffer;
    ARROW_RETURN_NOT_OK(arrow::AllocateResizableBuffer(nbytes, &buffer));
    int64_t bytes_read;
    ARROW_RETURN_NOT_OK(Read(nbytes, &bytes_read, buffer->mutable_data()));
    ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read, false));
    buffer->ZeroPadding();
    *out = buffer;
    return arrow::Status::OK();
  }

 private:
  int sock_;
  int64_t pos_;
};

// Convert an element of an Arrow Array to a Tensor
class ArrowConvertTensor : public arrow::ArrayVisitor {
 public:
  ArrowConvertTensor(int64_t row_idx, IteratorContext* ctx)
      : curr_row_idx_(row_idx), curr_ctx_(ctx), curr_array_values_(-1) {}

  // Convert to a Tensor and append to the output vector
  Status AppendTensor(std::shared_ptr<arrow::Array> array, DataType output_type,
                      std::vector<Tensor>* out_tensors) {
    curr_type_ = output_type;
    out_tensors_ = out_tensors;
    // TODO: make sure null count is 0
    CHECK_ARROW(array->Accept(this));
    return Status::OK();
  }

 protected:
  template <typename ArrayType>
  arrow::Status VisitFixedWidth(const ArrayType& array) {
    // TODO check type is correct
    // Create a Tensor of required size, curr_array_values_ < 0 indicates scalar
    Tensor tensor(curr_ctx_->allocator({}), curr_type_,
                  curr_array_values_ < 0 ? TensorShape({})
                                         : TensorShape({curr_array_values_}));

    const auto& fw_type =
        static_cast<const arrow::FixedWidthType&>(*array.type());
    const int64_t type_width = fw_type.bit_width() / 8;

    auto values = array.data()->buffers[1];
    if (values != NULLPTR) {
      const void* src = (values->data() + array.data()->offset * type_width) +
                        curr_row_idx_ * type_width;
      void* dst = const_cast<char*>(tensor.tensor_data().data());
      int32_t num_values = curr_array_values_ < 0 ? 1 : curr_array_values_;
      std::memcpy(dst, src, num_values * type_width);
    }

    out_tensors_->emplace_back(std::move(tensor));
    return arrow::Status::OK();
  }

#define VISIT_FIXED_WIDTH(TYPE)                             \
  virtual arrow::Status Visit(const TYPE& array) override { \
    return VisitFixedWidth(array);                          \
  }

  VISIT_FIXED_WIDTH(arrow::Int8Array)
  VISIT_FIXED_WIDTH(arrow::Int16Array)
  VISIT_FIXED_WIDTH(arrow::Int32Array)
  VISIT_FIXED_WIDTH(arrow::Int64Array)
  VISIT_FIXED_WIDTH(arrow::UInt8Array)
  VISIT_FIXED_WIDTH(arrow::UInt16Array)
  VISIT_FIXED_WIDTH(arrow::UInt32Array)
  VISIT_FIXED_WIDTH(arrow::UInt64Array)
  VISIT_FIXED_WIDTH(arrow::HalfFloatArray)
  VISIT_FIXED_WIDTH(arrow::FloatArray)
  VISIT_FIXED_WIDTH(arrow::DoubleArray)
#undef VISIT_FIXED_WITH

  virtual arrow::Status Visit(const arrow::ListArray& array) override {
    // TODO check if another array
    int32 values_offset = array.value_offset(curr_row_idx_);
    curr_array_values_ = array.value_length(curr_row_idx_);
    int32 tmp_row_idx = curr_row_idx_;
    curr_row_idx_ = 0;

    std::shared_ptr<arrow::Array> values = array.values();
    std::shared_ptr<arrow::Array> element_values =
        values->Slice(values_offset, curr_array_values_);
    auto result = element_values->Accept(this);
    curr_row_idx_ = tmp_row_idx;
    curr_array_values_ = -1;
    return result;
  }

 private:
  int64_t curr_row_idx_;
  DataType curr_type_;
  IteratorContext* curr_ctx_;
  int32_t curr_array_values_;
  std::vector<Tensor>* out_tensors_;
};

// Base class for defining a Dataset over Arrow record batches with an
// iterator that iterates over rows of the batch to get Tensors
class ArrowDatasetBase : public DatasetBase {
 public:
  ArrowDatasetBase(OpKernelContext* ctx, const std::vector<int32>& columns,
                   const DataTypeVector& output_types,
                   const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        columns_(columns),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    return Status::OK();
  }

  // Abstract base class for iterating over rows of Arrow record
  // batches. Implementations will define how record batches are
  // initialized and consumed.
  template <typename DatasetType>
  class ArrowBaseIterator : public DatasetIterator<DatasetType> {
   public:
    ArrowBaseIterator(
        const typename DatasetIterator<DatasetType>::Params& params)
        : DatasetIterator<DatasetType>(params) {}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);

      // Intialize and read first batch
      if (current_batch_ == nullptr) {
        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      }

      // Try to go to next batch if consumed all rows
      if (current_row_idx_ >= current_batch_->num_rows()) {
        TF_RETURN_IF_ERROR(NextStreamLocked());
      }

      // Check if reached end of stream
      if (current_batch_ == nullptr) {
        ResetStreamsLocked();
        *end_of_sequence = true;
      }

      // Assign Tensors for each column in the current row
      ArrowConvertTensor arrow_converter(current_row_idx_, ctx);
      for (size_t i = 0; i < this->dataset()->columns_.size(); ++i) {
        int32 col = this->dataset()->columns_[i];
        DataType dt = this->dataset()->output_types_[i];
        std::shared_ptr<arrow::Array> arr = current_batch_->column(col);
        TF_RETURN_IF_ERROR(arrow_converter.AppendTensor(arr, dt, out_tensors));
      }

      // Increment to next row
      ++current_row_idx_;
      *end_of_sequence = false;

      return Status::OK();
    }

   protected:
    Status SaveInternal(IteratorStateWriter* writer) override {
      return errors::Unimplemented("SaveInternal is currently not supported");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return errors::Unimplemented(
          "RestoreInternal is currently not supported");
    }

    // Setup Arrow record batch consumer and initialze current_batch_
    virtual Status SetupStreamsLocked(Env* env)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

    // Get the next Arrow record batch, if available. If not then
    // current_batch_ will be set to nullptr to indicate no further batches.
    virtual Status NextStreamLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      current_batch_ = nullptr;
      current_row_idx_ = 0;
      return Status::OK();
    }

    // Reset the Arrow record batch consumer when done with batches.
    virtual void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      current_batch_ = nullptr;
      current_row_idx_ = 0;
    }

    mutex mu_;
    std::shared_ptr<arrow::RecordBatch> current_batch_ GUARDED_BY(mu_) =
        nullptr;
    int64_t current_row_idx_ GUARDED_BY(mu_) = 0;
  };

  const std::vector<int32> columns_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

// Abstract base class to define an Arrow OpKernel with output_types and
// output_shapes attributes, and list of column indices. Implementations
// will define how to create the Arrow Dataset.
class ArrowOpKernelBase : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  ArrowOpKernelBase(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    // TODO: check types and shapes
  }

 private:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* columns_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("columns", &columns_tensor));
    OP_REQUIRES(
        ctx, columns_tensor->dims() <= 1,
        errors::InvalidArgument("`columns` must be a scalar or a vector."));

    std::vector<int32> columns;
    columns.reserve(columns_tensor->NumElements());
    for (int32 i = 0; i < static_cast<int32>(columns_tensor->NumElements());
         ++i) {
      columns.push_back(columns_tensor->flat<int32>()(i));
    }

    ArrowDatasetBase* arrow_output;
    MakeArrowDataset(ctx, columns, output_types_, output_shapes_,
                     &arrow_output);
    *output = arrow_output;
  }

 protected:
  // Define to construct an implementation of ArrowDatasetBase
  virtual void MakeArrowDataset(
      OpKernelContext* ctx, const std::vector<int32>& columns,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) = 0;

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

// Op to create an ArrowDataset that consumes Arrow record batches from
// memory in a Python process, or a Pandas DataFrame.
class ArrowDatasetOp : public ArrowOpKernelBase {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  explicit ArrowDatasetOp(OpKernelConstruction* ctx) : ArrowOpKernelBase(ctx) {}

  virtual void MakeArrowDataset(
      OpKernelContext* ctx, const std::vector<int32>& columns,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) override {
    const Tensor* batches_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("serialized_batches", &batches_tensor));
    OP_REQUIRES(
        ctx, batches_tensor->dims() <= 0,
        errors::InvalidArgument("`serialized_batches` must be a scalar."));
    string batches = batches_tensor->flat<string>()(0);

    *output = new Dataset(ctx, batches, columns, output_types_, output_shapes_);
  }

 private:
  class Dataset : public ArrowDatasetBase {
   public:
    // Construct a Dataset that consumed Arrow batches from serialized bytes
    // in a string. Record batches should be serialized in Arrow File format.
    Dataset(OpKernelContext* ctx, const string& serialized_batches,
            const std::vector<int32>& columns,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : ArrowDatasetBase(ctx, columns, output_types, output_shapes),
          batches_(serialized_batches) {}

   protected:
    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Arrow")}));
    }

    string DebugString() const override { return "ArrowDatasetOp::Dataset"; }

   private:
    class Iterator : public ArrowBaseIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : ArrowBaseIterator<Dataset>(params) {}

     private:
      Status SetupStreamsLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        std::shared_ptr<arrow::Buffer> buffer;
        CHECK_ARROW(arrow::Buffer::FromString(dataset()->batches_, &buffer));
        auto buffer_reader = std::make_shared<arrow::io::BufferReader>(buffer);
        CHECK_ARROW(
            arrow::ipc::RecordBatchFileReader::Open(buffer_reader, &reader_));
        num_batches_ = reader_->num_record_batches();
        if (num_batches_ > 0) {
          CHECK_ARROW(
              reader_->ReadRecordBatch(current_batch_idx_, &current_batch_));
        }
        return Status::OK();
      }

      Status NextStreamLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::NextStreamLocked();
        if (++current_batch_idx_ < num_batches_) {
          CHECK_ARROW(
              reader_->ReadRecordBatch(current_batch_idx_, &current_batch_));
        }
        return Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::ResetStreamsLocked();
        reader_.reset();
        current_batch_idx_ = 0;
        num_batches_ = 0;
      }

      std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader_
          GUARDED_BY(mu_);
      int current_batch_idx_ GUARDED_BY(mu_) = 0;
      int num_batches_ GUARDED_BY(mu_) = 0;
    };

    const string batches_;
  };
};

// Op to create an Arrow Dataset that consumes record batches from a list of
// files in Arrow Feather format. Feather is a light-weight columnar format
// ideal for simple writing of Pandas DataFrames.
class ArrowFeatherDatasetOp : public ArrowOpKernelBase {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  explicit ArrowFeatherDatasetOp(OpKernelConstruction* ctx)
      : ArrowOpKernelBase(ctx) {}

  virtual void MakeArrowDataset(
      OpKernelContext* ctx, const std::vector<int32>& columns,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filename` must be a scalar or vector."));
    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    *output =
        new Dataset(ctx, filenames, columns, output_types_, output_shapes_);
  }

 private:
  class Dataset : public ArrowDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const std::vector<string>& filenames,
            const std::vector<int32>& columns,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : ArrowDatasetBase(ctx, columns, output_types, output_shapes),
          filenames_(filenames) {}

   protected:
    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::ArrowFeather")}));
    }

    string DebugString() const override {
      return "ArrowFeatherDatasetOp::Dataset";
    }

   private:
    class Iterator : public ArrowBaseIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : ArrowBaseIterator<Dataset>(params) {}

     private:
      Status SetupStreamsLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        return SetupStreamsLocked();
      }

      Status SetupStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        const string& filename = dataset()->filenames_[current_file_idx_];
        std::shared_ptr<arrow::io::ReadableFile> in_file;
        CHECK_ARROW(arrow::io::ReadableFile::Open(filename, &in_file));
        std::unique_ptr<arrow::ipc::feather::TableReader> reader;
        CHECK_ARROW(arrow::ipc::feather::TableReader::Open(in_file, &reader));

        // Read file columns and build a table
        int64_t num_columns = reader->num_columns();
        std::vector<std::shared_ptr<arrow::Field>> fields(num_columns);
        std::vector<std::shared_ptr<arrow::Column>> columns(num_columns);
        for (int64_t i = 0; i < num_columns; ++i) {
          CHECK_ARROW(reader->GetColumn(i, &columns[i]));
          fields[i] = columns[i]->field();
        }
        auto schema = std::make_shared<arrow::Schema>(fields);
        auto table = arrow::Table::Make(schema, columns);

        // Convert the table to a sequence of batches
        arrow::TableBatchReader tr(*table.get());
        std::shared_ptr<arrow::RecordBatch> batch;
        CHECK_ARROW(tr.ReadNext(&batch));
        current_batch_ = batch;
        while (batch != nullptr) {
          record_batches_.push_back(batch);
          CHECK_ARROW(tr.ReadNext(&batch));
        }
        return Status::OK();
      }

      Status NextStreamLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::NextStreamLocked();
        if (++current_batch_idx_ < record_batches_.size()) {
          current_batch_ = record_batches_[current_batch_idx_];
        } else if (++current_file_idx_ < dataset()->filenames_.size()) {
          size_t temp_file_idx = current_file_idx_;
          ResetStreamsLocked();
          current_file_idx_ = temp_file_idx;
          SetupStreamsLocked();
        }
        return Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::ResetStreamsLocked();
        current_file_idx_ = 0;
        current_batch_idx_ = 0;
        record_batches_.clear();
      }

      size_t current_file_idx_ GUARDED_BY(mu_) = 0;
      size_t current_batch_idx_ GUARDED_BY(mu_) = 0;
      std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_
          GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
  };
};

// Op to create an Arrow Dataset that consumes record batches from an input
// stream. Currently supported input streams are a POSIX socket client, with
// host given as "<IP>:<PORT>", or from STDIN if host is "STDIN".
class ArrowStreamDatasetOp : public ArrowOpKernelBase {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  explicit ArrowStreamDatasetOp(OpKernelConstruction* ctx)
      : ArrowOpKernelBase(ctx) {}

  virtual void MakeArrowDataset(
      OpKernelContext* ctx, const std::vector<int32>& columns,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) override {
    const Tensor* host_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("host", &host_tensor));
    OP_REQUIRES(ctx, host_tensor->dims() == 0,
                errors::InvalidArgument("`host` must be a scalar."));
    string host = host_tensor->flat<string>()(0);

    *output = new Dataset(ctx, host, columns, output_types_, output_shapes_);
  }

 private:
  class Dataset : public ArrowDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const string& host,
            const std::vector<int32>& columns,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : ArrowDatasetBase(ctx, columns, output_types, output_shapes),
          host_(host) {}

   protected:
    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::ArrowStream")}));
    }

    string DebugString() const override {
      return "ArrowStreamDatasetOp::Dataset";
    }

   private:
    class Iterator : public ArrowBaseIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : ArrowBaseIterator<Dataset>(params) {}

     private:
      Status SetupStreamsLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        if (dataset()->host_ == "STDIN") {
          in_stream_ = std::make_shared<arrow::io::StdinStream>();
        } else {
          int sock = 0;
          struct sockaddr_in serv_addr;
          if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            return errors::InvalidArgument("Socket creation error");
          }
          bzero((char*)&serv_addr, sizeof(serv_addr));
          serv_addr.sin_addr.s_addr = inet_addr(dataset()->host_.c_str());
          serv_addr.sin_family = AF_INET;
          // TODO parse port with hostname
          serv_addr.sin_port = htons(8080);
          // if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)
          // printf("\nInvalid address/ Address not supported \n");
          if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) <
              0) {
            return errors::InvalidArgument("Connection failed to host: " +
                                           dataset()->host_);
          }
          in_stream_ = std::make_shared<SocketStream>(sock);
        }

        CHECK_ARROW(arrow::ipc::RecordBatchStreamReader::Open(in_stream_.get(),
                                                              &reader_));
        CHECK_ARROW(reader_->ReadNext(&current_batch_));
        return Status::OK();
      }

      Status NextStreamLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::NextStreamLocked();
        CHECK_ARROW(reader_->ReadNext(&current_batch_));
        return Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::ResetStreamsLocked();
        reader_.reset();
        in_stream_.reset();
      }

      std::shared_ptr<arrow::io::InputStream> in_stream_ GUARDED_BY(mu_);
      std::shared_ptr<arrow::ipc::RecordBatchReader> reader_ GUARDED_BY(mu_);
    };

    const string host_;
  };
};

REGISTER_KERNEL_BUILDER(Name("ArrowDataset").Device(DEVICE_CPU),
                        ArrowDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ArrowFeatherDataset").Device(DEVICE_CPU),
                        ArrowFeatherDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ArrowStreamDataset").Device(DEVICE_CPU),
                        ArrowStreamDatasetOp);

}  // namespace tensorflow
