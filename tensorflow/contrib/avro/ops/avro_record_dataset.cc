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

#include <avro.h>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"

// As boiler plate I used
// https://github.com/tensorflow/tensorflow/core/kernels/reader_dataset_ops.cc
// https://github.com/tensorflow/tensorflow/blob/v1.4.1/tensorflow/core/ops/dataset_ops.cc
// (register op)

namespace tensorflow {

// Register the avro record dataset operator
REGISTER_OP("AvroRecordDataset")
    .Input("filenames: string")
    .Input("reader_schema: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that emits the avro records from one or more files.
filenames: A scalar or vector containing the name(s) of the file(s) to be
  read.
reader_schema: Reader schema used for schema resolution.
buffer_size: The size of the buffer used when reading avro files into memory
)doc");

void AvroFileReaderDestructor(avro_file_reader_t reader) {
  // I don't think we need the CHECK_NOTNULL
  CHECK_GE(avro_file_reader_close(reader), 0);
}

void AvroSchemaDestructor(avro_schema_t schema) {
  // Confusingly, it appears that the avro_file_reader_t creates its
  // own reference to this schema, so the schema is not really
  // "uniquely" owned...
  CHECK_GE(avro_schema_decref(schema), 0);
};

void AvroValueInterfaceDestructor(avro_value_iface_t * iface)  {
  avro_value_iface_decref(iface);
}

void AvroValueDestructor(avro_value_t * value) {
    // This is unnecessary clunky because avro's free assumes that
    // the iface ptr is initialized which is only the case once used
    if (value->iface != nullptr) {
      avro_value_decref(value);
    } else {
      // free the container
      free(value);
    }
}

// This reader is not thread safe
class SequentialAvroRecordReader {
 public:
  // Construct a sequential avro record reader
  //
  // 'file' is the random access file
  // 'file_size' is the size of the file
  // 'filename' is the name of the file
  // 'reader_schema' are avro reader options
  // 'buffer_size' size of the buffer reading from files
  //
  SequentialAvroRecordReader(RandomAccessFile* p_file, const uint64 file_size,
                             const string& filename,
                             const string& reader_schema,
                             const int64 buffer_size)
    : p_file_(p_file),
      filename_(filename),
      file_size_(file_size),
      reader_schema_str_(reader_schema),
      file_reader_(nullptr, AvroFileReaderDestructor),
      reader_schema_(nullptr, AvroSchemaDestructor),
      writer_schema_(nullptr, AvroSchemaDestructor),
      p_reader_iface_(nullptr, AvroValueInterfaceDestructor),
      p_writer_iface_(nullptr, AvroValueInterfaceDestructor),
      p_reader_value_(new avro_value_t, AvroValueDestructor),
      p_writer_value_(new avro_value_t, AvroValueDestructor) {
      // Initialize the iface to null for the destructor
      p_reader_value_.get()->iface = nullptr;
      p_writer_value_.get()->iface = nullptr;
  }
  // Call for startup of work after construction. Loads data into memory and
  // sets up the avro file reader
  //
  Status OnWorkStartup() {
    // Clear the error message, so we won't get a wrong message
    avro_set_error("");

    // Read the file into memory via the gfile API so we can accept
    // files on S3, HDFS, etc.
    char* file_data = new (std::nothrow) char[file_size_];
    if (file_data == nullptr) {
        return Status(errors::InvalidArgument("Unable to allocate ", file_size_,
                                              " B on memory in avro reader."));
    }
    StringPiece result;
    TF_RETURN_IF_ERROR(p_file_->Read(0, file_size_, &result, file_data));
    FILE* fp = fmemopen(static_cast<void*>(file_data), file_size_, "r");
    if (fp == nullptr) {
      return Status(errors::InvalidArgument("Unable to open file ", filename_,
                                            " on memory in avro reader."));
    }

    // Get an avro file reader for that file handle, the 1 indicates to close
    // the file handle when done
    avro_file_reader_t file_reader_tmp;
    if (avro_file_reader_fp(fp, filename_.c_str(), 1, &file_reader_tmp) != 0) {
      return Status(errors::InvalidArgument("Unable to open file ", filename_,
                                            " in avro reader. ", avro_strerror()));
    }
    file_reader_.reset(file_reader_tmp);

    writer_schema_.reset(avro_file_reader_get_writer_schema(file_reader_.get()));

    // The user provided a schema for the reader, check if we need to do schema
    // resolution
    bool do_resolution = false;
    if (reader_schema_str_.length() > 0) {
      avro_schema_t reader_schema_tmp;

      // Create value to read into using the provided schema
      if (avro_schema_from_json_length(reader_schema_str_.data(),
                                       reader_schema_str_.length(),
                                       &reader_schema_tmp) != 0) {
        return Status(errors::InvalidArgument(
            "The provided json schema is invalid. ", avro_strerror()));
      }
      reader_schema_.reset(reader_schema_tmp);
      do_resolution = !avro_schema_equal(writer_schema_.get(), reader_schema_.get());
      // We need to do a schema resolution, if the schemas are not the same
    }

    if (do_resolution) {
      // Create reader class
      p_reader_iface_.reset(avro_generic_class_from_schema(reader_schema_.get()));
      // Create instance for reader class
      if (avro_generic_value_new(p_reader_iface_.get(), p_reader_value_.get()) != 0) {
        return Status(errors::InvalidArgument(
            "Unable to value for user-supplied schema. ", avro_strerror()));
      }
      // Create resolved writer class
      p_writer_iface_.reset(avro_resolved_writer_new(writer_schema_.get(), reader_schema_.get()));
      if (p_writer_iface_.get() == nullptr) {
        // Cleanup
        return Status(errors::InvalidArgument("Schemas are incompatible. ",
                                              avro_strerror()));
      }
      // Create instance for resolved writer class
      if (avro_resolved_writer_new_value(p_writer_iface_.get(), p_writer_value_.get()) !=
          0) {
        // Cleanup
        return Status(
            errors::InvalidArgument("Unable to create resolved writer."));
      }
      avro_resolved_writer_set_dest(p_writer_value_.get(), p_reader_value_.get());
    } else {
      p_writer_iface_.reset(avro_generic_class_from_schema(writer_schema_.get()));
      if (avro_generic_value_new(p_writer_iface_.get(), p_writer_value_.get()) != 0) {
        return Status(errors::InvalidArgument(
            "Unable to create instance for generic class."));
      }
      // The p_reader_value_ is the same as the p_writer_value_ in the case we do
      // not need to resolve the schema
      avro_value_copy_ref(p_reader_value_.get(), p_writer_value_.get());
    }

    return Status::OK();
  }
  // Reads the next record into the string record. Note, `OnWorkStartup` must
  // be called before calling this method
  //
  // 'record' a string holding the serialized version of the record
  //
  Status ReadRecord(string* record) {
    bool at_end =
      avro_file_reader_read_value(file_reader_.get(), p_writer_value_.get()) != 0;
    size_t len;
    if (avro_value_sizeof(p_reader_value_.get(), &len)) {
      return Status(errors::InvalidArgument("Could not find size of value, ",
                                            avro_strerror()));
    }
    record->resize(len);
    avro_writer_t mem_writer = avro_writer_memory(record->data(), len);
    if (avro_value_write(mem_writer, p_reader_value_.get())) {
      avro_writer_free(mem_writer);
      return Status(errors::InvalidArgument("Unable to write value to memory."));
    }
    avro_writer_free(mem_writer);
    return at_end ? errors::OutOfRange("eof") : Status::OK();
  }

 private:
  const RandomAccessFile* p_file_;  // Pointer to file
  const string filename_;           // Name of the file
  const uint64 file_size_;          // Size of files in B
  const string reader_schema_str_;  // User supplied string to read this avro
                                    // file

  using AvroFileReaderUPtr = std::unique_ptr<struct avro_file_reader_t_,
                                             void(*)(avro_file_reader_t)>;
  AvroFileReaderUPtr file_reader_;  // Avro file reader

  using AvroSchemaUPtr = std::unique_ptr<struct avro_obj_t,
                                         void(*)(avro_schema_t)>;
  AvroSchemaUPtr reader_schema_;  // Schema to read, set only when doing schema
                                  // resolution
  AvroSchemaUPtr writer_schema_; // Schema that the file was written with

  using AvroValueInterfacePtr = std::unique_ptr<avro_value_iface_t,
                                                void(*)(avro_value_iface_t*)>;
  AvroValueInterfacePtr p_reader_iface_;  // Reader class info to create instances
  AvroValueInterfacePtr p_writer_iface_;  // Writer class info to create instances

  using AvroValueUPtr = std::unique_ptr<avro_value_t, void(*)(avro_value_t*)>;
  AvroValueUPtr p_reader_value_; // Reader value, unequal from writer value for
                                 // schema resolution
  AvroValueUPtr p_writer_value_; // Writer value
};

class AvroRecordDatasetOp : public DatasetOpKernel {
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

    string reader_schema_str;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "reader_schema",
      &reader_schema_str));

    int64 buffer_size = -1;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(ctx, buffer_size >= 256,
                errors::InvalidArgument("`buffer_size` must be >= 256 B"));

    *output = new Dataset(ctx, std::move(filenames), reader_schema_str,
      buffer_size);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, std::vector<string> filenames,
                     const string& reader_schema_str,
                     const int64 buffer_size)
        : DatasetBase(DatasetContext(ctx)),
          filenames_(std::move(filenames)),
          reader_schema_str_(reader_schema_str),
          buffer_size_(buffer_size) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const
        override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::AvroRecord")}));
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

    string DebugString() const override { return "AvroRecordDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {

      // TODO(fraudies): Implement me, below is a copy of the code from the
      // protobuf example reader
      /*
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      Node* num_parallle_calls_node;
      std::vector<Node*> dense_defaults_nodes;
      dense_defaults_nodes.reserve(dense_defaults_.size());

      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallle_calls_node));

      for (const Tensor& dense_default : dense_defaults_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(dense_default, &node));
        dense_defaults_nodes.emplace_back(node);
      }

      AttrValue sparse_keys_attr;
      AttrValue dense_keys_attr;
      AttrValue sparse_types_attr;
      AttrValue dense_attr;
      AttrValue dense_shapes_attr;

      b->BuildAttrValue(sparse_keys_, &sparse_keys_attr);
      b->BuildAttrValue(dense_keys_, &dense_keys_attr);
      b->BuildAttrValue(sparse_types_, &sparse_types_attr);
      b->BuildAttrValue(dense_types_, &dense_attr);
      b->BuildAttrValue(dense_shapes_, &dense_shapes_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(this,
                                       {
                                           {0, input_graph_node},
                                           {1, num_parallle_calls_node},
                                       },
                                       {{2, dense_defaults_nodes}},
                                       {{"sparse_keys", sparse_keys_attr},
                                        {"dense_keys", dense_keys_attr},
                                        {"sparse_types", sparse_types_attr},
                                        {"Tdense", dense_attr},
                                        {"dense_shapes", dense_shapes_attr}},
                                       output));
                                       */
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
        do { // What is the point of this loop???
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
            } else {
              CHECK(errors::IsOutOfRange(s));
              // We have reached the end of the current file, so maybe
              // move on to next file.
              reader_.reset();
              file_.reset();
              ++current_file_index_;
            }
          }

          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          // Actually move on to next file.
          // Looks like this cannot request multiple files in parallel. Hmm.
          const string& next_filename =
              dataset()->filenames_[current_file_index_];

          TF_RETURN_IF_ERROR(
            ctx->env()->NewRandomAccessFile(next_filename, &file_));

          uint64 file_size;
          TF_RETURN_IF_ERROR(
            ctx->env()->GetFileSize(next_filename, &file_size));

          reader_.reset(new SequentialAvroRecordReader(
              file_.get(), file_size, next_filename,
              dataset()->reader_schema_str_, dataset()->buffer_size_));
          TF_RETURN_IF_ERROR(reader_->OnWorkStartup());
        } while (true);
      }

     private:
      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;

      // `reader_` will borrow the object that `file_` points to, so
      // we must destroy `reader_` before `file_`.
      std::unique_ptr<RandomAccessFile> file_ GUARDED_BY(mu_);
      std::unique_ptr<SequentialAvroRecordReader> reader_ GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
    const string reader_schema_str_;
    const int64 buffer_size_;
  };
};

REGISTER_KERNEL_BUILDER(Name("AvroRecordDataset").Device(DEVICE_CPU),
                        AvroRecordDatasetOp);

}  // namespace tensorflow
