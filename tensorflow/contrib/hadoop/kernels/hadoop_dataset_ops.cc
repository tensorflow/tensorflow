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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {
namespace {

static const size_t kSyncMarkerSize = 16;
static const size_t kSequenceFileBufferSize = 1024 * 1024;

class SequenceFileReader {
 public:
  explicit SequenceFileReader(RandomAccessFile* file)
      : input_stream_(
            new io::BufferedInputStream(file, kSequenceFileBufferSize)) {}

  Status ReadHeader() {
    string version;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(4, &version));
    if (version.substr(0, 3) != "SEQ" || version[3] != 6) {
      return errors::InvalidArgument(
          "sequence file header must starts with `SEQ6`, received \"",
          version.substr(0, 3), static_cast<int>(version[3]), "\"");
    }
    TF_RETURN_IF_ERROR(ReadString(&key_class_name_));
    TF_RETURN_IF_ERROR(ReadString(&value_class_name_));

    // At the moment we only support `org.apache.hadoop.io.Text` for key/value.
    // TODO (yongtang): Add more class name support.
    if (key_class_name_ != "org.apache.hadoop.io.Text" ||
        value_class_name_ != "org.apache.hadoop.io.Text") {
      return errors::Unimplemented("key/value of '", key_class_name_, "/",
                                   value_class_name_,
                                   "' is currently not supported");
    }

    string buffer;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(2, &buffer));
    compression_ = buffer[0];
    block_compression_ = buffer[1];
    if (compression_ || block_compression_) {
      TF_RETURN_IF_ERROR(ReadString(&compression_codec_class_name_));
    }

    // At the moment no compression is supported.
    // TODO (yongtang): Add compression support.
    if (compression_ || block_compression_) {
      return errors::Unimplemented("compression is currently not supported");
    }

    // Not interested in metadata for now.
    uint32 num_metadata_pairs = 0;
    TF_RETURN_IF_ERROR(ReadUInt32(&num_metadata_pairs));
    if (num_metadata_pairs > 1024) {
      return errors::InvalidArgument(
          "sequence file metadata should have key value pairs < 1024,  "
          "received ",
          num_metadata_pairs);
    }
    for (int i = 0; i < num_metadata_pairs; i++) {
      TF_RETURN_IF_ERROR(ReadString(nullptr));
      TF_RETURN_IF_ERROR(ReadString(nullptr));
    }

    TF_RETURN_IF_ERROR(
        input_stream_->ReadNBytes(kSyncMarkerSize, &sync_marker_));

    return Status::OK();
  }

  Status ReadRecord(string* key, string* value) {
    uint32 length = 0;
    TF_RETURN_IF_ERROR(ReadUInt32(&length));
    if (length == static_cast<uint32>(-1)) {
      // Sync marker.
      string sync_marker;
      TF_RETURN_IF_ERROR(
          input_stream_->ReadNBytes(kSyncMarkerSize, &sync_marker));
      if (sync_marker != sync_marker_) {
        return errors::InvalidArgument(
            "sequence file should have sync marker \"", sync_marker_,
            "\" at pos ", input_stream_->Tell() - kSyncMarkerSize,
            ", received \"", sync_marker, "\"");
      }
      return ReadRecord(key, value);
    }
    uint32 key_length = 0;
    TF_RETURN_IF_ERROR(ReadUInt32(&key_length));
    if (key_length > length) {
      return errors::InvalidArgument("key length (", key_length,
                                     ") should be < record length (", length,
                                     ")");
    }
    // At the moment we only support `org.apache.hadoop.io.Text` for key/value.
    // TODO (yongtang): Expand supported format.
    TF_RETURN_IF_ERROR(ReadString(key));
    TF_RETURN_IF_ERROR(ReadString(value));
    return Status::OK();
  }

  Status ReadString(string* value) {
    int64 length = 0;
    TF_RETURN_IF_ERROR(ReadVInt(&length));
    if (value == nullptr) {
      return input_stream_->SkipNBytes(length);
    }
    return input_stream_->ReadNBytes(length, value);
  }

  Status ReadUInt32(uint32* value) {
    string buffer;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(4, &buffer));
    *value = ((static_cast<uint32>(buffer[0]) << 24) |
              static_cast<uint32>(buffer[1]) << 16) |
             (static_cast<uint32>(buffer[2]) << 8) |
             static_cast<uint32>(buffer[3]);
    return Status::OK();
  }

  Status ReadVInt(int64* value) {
    string buffer;
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(1, &buffer));
    if (buffer[0] >= -112) {
      *value = static_cast<int64>(buffer[0]);
      return Status::OK();
    }

    int64 remaining = 0;
    bool negative = false;
    if (buffer[0] >= -120) {
      remaining = static_cast<int64>(-112) - static_cast<int64>(buffer[0]);
    } else {
      remaining = static_cast<int64>(-120) - static_cast<int64>(buffer[0]);
      negative = true;
    }
    buffer.clear();
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(remaining, &buffer));

    uint64 v = 0;
    for (int i = 0; i < buffer.size(); i++) {
      v = (v << 8) | static_cast<uint64>(buffer[i]);
    }
    if (negative) {
      v = ~v;
    }
    *value = static_cast<int64>(v);
    return Status::OK();
  }

  virtual ~SequenceFileReader() = default;

 private:
  std::unique_ptr<io::InputStreamInterface> input_stream_;
  string key_class_name_;
  string value_class_name_;
  string sync_marker_;
  bool compression_;
  bool block_compression_;
  string compression_codec_class_name_;
  TF_DISALLOW_COPY_AND_ASSIGN(SequenceFileReader);
};
class SequenceFileDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit SequenceFileDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    for (const DataType& dt : output_types_) {
      OP_REQUIRES(ctx, dt == DT_STRING,
                  errors::InvalidArgument(
                      "Each element of `output_types_` must be one of: "
                      "DT_STRING"));
    }
  }
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

    *output = new Dataset(ctx, filenames, output_types_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const std::vector<string>& filenames,
            const DataTypeVector& output_types)
        : DatasetBase(DatasetContext(ctx)),
          filenames_(filenames),
          output_types_(output_types) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::SequenceFile")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}, {}});
      return *shapes;
    }

    string DebugString() const override {
      return "SequenceFileDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {filenames}, output));
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
            string key, value;
            Status status = reader_->ReadRecord(&key, &value);
            if (!errors::IsOutOfRange(status)) {
              TF_RETURN_IF_ERROR(status);

              Tensor key_tensor(ctx->allocator({}), DT_STRING, {});
              key_tensor.scalar<string>()() = key;
              out_tensors->emplace_back(std::move(key_tensor));

              Tensor value_tensor(ctx->allocator({}), DT_STRING, {});
              value_tensor.scalar<string>()() = value;
              out_tensors->emplace_back(std::move(value_tensor));

              *end_of_sequence = false;
              return Status::OK();
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
        return errors::Unimplemented("SaveInternal is currently not supported");
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented(
            "RestoreInternal is currently not supported");
      }

     private:
      // Sets up SequenceFile streams to read from the topic at
      // `current_file_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }

        // Actually move on to next file.
        const string& filename = dataset()->filenames_[current_file_index_];
        TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file_));
        reader_.reset(new SequenceFileReader(file_.get()));
        return reader_->ReadHeader();
      }

      // Resets all Hadoop SequenceFile streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        reader_.reset();
        file_.reset();
      }

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<RandomAccessFile> file_ GUARDED_BY(mu_);
      std::unique_ptr<SequenceFileReader> reader_ GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
    const DataTypeVector output_types_;
  };
  DataTypeVector output_types_;
};
}  // namespace

REGISTER_KERNEL_BUILDER(Name("SequenceFileDataset").Device(DEVICE_CPU),
                        SequenceFileDatasetOp);

}  // namespace tensorflow
