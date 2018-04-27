#include <avro.h>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"

// As boiler plate I used
// https://github.com/tensorflow/tensorflow/core/kernels/reader_dataset_ops.cc
// https://github.com/tensorflow/tensorflow/blob/v1.4.1/tensorflow/core/ops/dataset_ops.cc (register op)

using namespace tensorflow;

// Register the avro record dataset operator
REGISTER_OP("AvroRecordDataset")
    .Input("filenames: string")
    .Input("schema: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that emits the avro records from one or more files.
filenames: A scalar or vector containing the name(s) of the file(s) to be
  read.
schema: A string used that is used for schema resolution.
)doc");

// This class represents the avro reader options
class AvroReaderOptions {
public:
  // Creates avro reader options with the given schema and buffer size.
  //
  static AvroReaderOptions CreateAvroReaderOptions(const string& schema, int64 buffer_size) {
    AvroReaderOptions options;
    options.schema = schema;
    options.buffer_size = buffer_size;
    return options;
  }
  string schema;
  int64 buffer_size = 256 * 1024; // 256 kB as default but this can be overwritten by the user
};

// This reader is not thread safe
class SequentialAvroRecordReader {
public:
  // Construct a sequential avro record reader
  //
  // 'file' is the random access file
  //
  // 'file_size' is the size of the file
  //
  // 'filename' is the name of the file
  //
  // 'options' are avro reader options
  //
  SequentialAvroRecordReader(RandomAccessFile* file, const uint64 file_size, const string& filename,
    const AvroReaderOptions& options = AvroReaderOptions())
    :
    initialized_(false),
    file_size_(file_size),
    filename_(filename),
    reader_schema_str_(options.schema),
    do_resolution_(false),
    options_(options) {
    input_buffer_.reset(new io::InputBuffer(file, options.buffer_size));
  }
  virtual ~SequentialAvroRecordReader() {
    // Guard against clean-up of non-initialized instances
    if (initialized_) {
      // If we used schema resolution we need to clean-up reader structures
      if (do_resolution_) {
        avro_schema_decref(reader_schema_);
        avro_value_iface_decref(p_reader_iface_);
      }
      avro_value_decref(&reader_value_); // we created another reference, so decrease regardless of do_resolution_
      avro_schema_decref(writer_schema_);
      avro_value_iface_decref(p_writer_iface_);
      avro_value_decref(&writer_value_);
      avro_file_reader_close(file_reader_);
      free(data_buffer_);
    }
  }
  // Reads the next record into the string record
  //
  // 'record' pointer to the string where to load the record in
  //
  // returns Status about this operation
  //
  Status ReadRecord(string* record) {
    // For any error in this method I'm not cleaning up memory because the user may try again with success
    avro_writer_t mem_writer;
    size_t len;
    char* serialized;
    bool at_end = avro_file_reader_read_value(file_reader_, &writer_value_) != 0;
    if (avro_value_sizeof(&reader_value_, &len)) {
      return Status(errors::InvalidArgument("Could not find size of value, ", avro_strerror()));
    }
    serialized = (char*) malloc(len * sizeof(char));
    if (serialized == nullptr) {
      return Status(errors::ResourceExhausted("Unable to allocate ", len/1024/1024, " MB of memory."));
    }
    mem_writer = avro_writer_memory(serialized, len);
    if (mem_writer == nullptr) {
      free(serialized);
      return Status(errors::ResourceExhausted("Unable to allocate ", len/1024/1024, " MB of memory."));
    }
    if (avro_value_write(mem_writer, &reader_value_)) {
      free(serialized);
      return Status(errors::InvalidArgument("Unable to write value to memory."));
    }
    *record = string(serialized, len);
    free(serialized);
    avro_writer_free(mem_writer);
    return at_end ? errors::OutOfRange("eof") : Status::OK();
  }
  // Call for startup of work after construction. Loads data into memory and sets up the avro file reader
  //
  // returns Status about this operation
  //
  Status OnWorkStartup() {

    // Clear the error message, so we won't get a wrong message because not all errors set a message in avro
    avro_set_error("");
    Status status;

    // Create the buffer and load the file contents into the buffer
    TF_RETURN_IF_ERROR(CreateAndLoadFileIntoBuffer(options_.buffer_size));

    // We read the data from one file into memory and get a file handle to that memory
    FILE* fp = fmemopen((void*)data_buffer_, file_size_, "r"); // -1 to remove the trailing space from string
    if (fp == nullptr) {
      // Cleanup
      free(data_buffer_);
      return Status(errors::InvalidArgument("Unable to open file ", filename_, " on memory in avro reader."));
    }

    // Get an avro file reader for that file handle, the 1 indicates to close the file handle when done
    if (avro_file_reader_fp(fp, filename_.c_str(), 1, &file_reader_) != 0) {
      // Cleanup
      free(data_buffer_);
      return Status(errors::InvalidArgument("Unable to open file ", filename_, " in avro reader. ", avro_strerror()));
    }

    // Get the schema this file was written in
    writer_schema_ = avro_file_reader_get_writer_schema(file_reader_);

    // The user provided a schema for the reader, check if we need to do schema resolution
    if (reader_schema_str_.length() > 0) {

      // Create value to read into using the provided schema
      if (avro_schema_from_json_length(reader_schema_str_.c_str(),
                                       reader_schema_str_.length(), &reader_schema_) != 0) {
        // Cleanup
        avro_schema_decref(writer_schema_);
        avro_file_reader_close(file_reader_);
        free(data_buffer_);
        return Status(errors::InvalidArgument("The provided json schema is invalid. ", avro_strerror()));
      }
      // We need to do a schema resolution, if the schemas are not the same
      do_resolution_ = !avro_schema_equal(writer_schema_, reader_schema_);
      // We don't need the reader schema anymore if we don't need to do resolution
      if (!do_resolution_) {
        avro_schema_decref(reader_schema_);
      }
    }

    // If we need to do resolution
    if (do_resolution_) {
      #ifdef DEBUG_LOG_ENABLED
        LOG(INFO) << "Do schema resolution.";
      #endif
      // Create reader class
      p_reader_iface_ = avro_generic_class_from_schema(reader_schema_);
      if (p_reader_iface_ == nullptr) {
        // Cleanup
        avro_schema_decref(writer_schema_);
        avro_schema_decref(reader_schema_);
        avro_file_reader_close(file_reader_);
        free(data_buffer_);
        return Status(errors::InvalidArgument("Unable to create class for user-supplied schema. ", avro_strerror()));
      }
      // Create instance for reader class
      if (avro_generic_value_new(p_reader_iface_, &reader_value_) != 0) {
        // Cleanup
        avro_value_iface_decref(p_reader_iface_);
        avro_schema_decref(writer_schema_);
        avro_schema_decref(reader_schema_);
        avro_file_reader_close(file_reader_);
        free(data_buffer_);
        return Status(errors::InvalidArgument("Unable to value for user-supplied schema. ", avro_strerror()));
      }
      // Create resolved writer class
      p_writer_iface_ = avro_resolved_writer_new(writer_schema_, reader_schema_);
      if (p_writer_iface_ == nullptr) {
        // Cleanup
        avro_value_decref(&reader_value_);
        avro_value_iface_decref(p_reader_iface_);
        avro_schema_decref(writer_schema_);
        avro_schema_decref(reader_schema_);
        avro_file_reader_close(file_reader_);
        free(data_buffer_);
        return Status(errors::InvalidArgument("Schemas are incompatible. ", avro_strerror()));
      }
      // Create instance for resolved writer class
      if (avro_resolved_writer_new_value(p_writer_iface_, &writer_value_) != 0) {
        // Cleanup
        avro_value_iface_decref(p_writer_iface_);
        avro_value_decref(&reader_value_);
        avro_value_iface_decref(p_reader_iface_);
        avro_schema_decref(writer_schema_);
        avro_schema_decref(reader_schema_);
        avro_file_reader_close(file_reader_);
        free(data_buffer_);
        return Status(errors::InvalidArgument("Unable to create resolved writer."));
      }
      avro_resolved_writer_set_dest(&writer_value_, &reader_value_);

    // If we don't do resolution
    } else {
      // Create class for writer
      p_writer_iface_ = avro_generic_class_from_schema(writer_schema_);
      if (p_writer_iface_ == nullptr) {
        // Cleanup
        avro_schema_decref(writer_schema_);
        avro_file_reader_close(file_reader_);
        free(data_buffer_);
        return Status(errors::InvalidArgument("Unable to create generic class."));
      }
      // Create instance for writer class
      if (avro_generic_value_new(p_writer_iface_, &writer_value_) != 0) {
        // Cleanup
        avro_value_iface_decref(p_writer_iface_);
        avro_schema_decref(writer_schema_);
        avro_file_reader_close(file_reader_);
        free(data_buffer_);
        return Status(errors::InvalidArgument("Unable to create instance for generic class."));
      }
      // The reader_value_ is the same as the writer_value_ in the case we do not need to resolve the schema
      avro_value_copy_ref(&reader_value_, &writer_value_);
    }

    // We initialized this avro record reader
    initialized_ = true;

    // All is ok
    return Status::OK();
  }
private:
  // Creates buffer and loads file contents into it the buffer
  //
  // 'read_buffer_size' buffer size when reading file contents
  //
  Status CreateAndLoadFileIntoBuffer(int64 read_buffer_size) {
    // Some variables
    uint64 data_buffer_size = 0;
    size_t bytes_read;
    Status status;

    // Allocate memory and check we got it
    data_buffer_ = (char*) malloc(file_size_ * sizeof(char));
    if (data_buffer_ == nullptr) {
      return Status(errors::ResourceExhausted("Unable to allocate ", file_size_/1024/1024,
                    " MB of memory for file buffer."));
    }

    // While we still need to read data
    char* i_data_buffer = data_buffer_;
    while (data_buffer_size < file_size_) {
      // Read data in junks of byte_to_read
      status = input_buffer_->ReadNBytes(read_buffer_size, i_data_buffer, &bytes_read);
      // Increment the size and the write pointer
      data_buffer_size += bytes_read;
      i_data_buffer += bytes_read;
      // If we are at the end of the file
      if (errors::IsOutOfRange(status)) {
        break;
      }
      // If we encountered any error
      if (!status.ok()) {
        free(data_buffer_);
        return status;
      }
    }

    // After reading check that we read the expected amount of data
    if (data_buffer_size != file_size_) {
      free(data_buffer_);
      return Status(errors::InvalidArgument("File ", filename_, " size is ", file_size_, " B but finished reading ",
                    data_buffer_size," B."));
    }
    return Status::OK();
  }

  char* data_buffer_; // The data buffer
  bool initialized_; // Has been initialized
  string filename_; // Name of the file
  uint64 file_size_; // Size of the file in B
  std::unique_ptr<io::InputBuffer> input_buffer_; // input buffer used to load from random access file
  bool do_resolution_; // True to do schema resolution
  const string reader_schema_str_; // User supplied string to read this avro file
  int64 line_number_; // Line number == record number
  avro_file_reader_t file_reader_; // Avro file reader
  avro_schema_t reader_schema_; // Schema to read, set only when doing schema resolution
  avro_schema_t writer_schema_; // Schema that the file was written with
  avro_value_iface_t* p_reader_iface_; // Reader class info to create instances
  avro_value_iface_t* p_writer_iface_; // Writer class info to create instances
  avro_value_t reader_value_; // Reader value, unequal from writer value when doing schema resolution
  avro_value_t writer_value_; // Writer value
  AvroReaderOptions options_; // Options for the avro reader
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

    string schema;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "schema", &schema));

    int64 buffer_size = -1;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(ctx, buffer_size >= 256, errors::InvalidArgument("`buffer_size` must be >= 256 B"));

    *output = new Dataset(std::move(filenames), schema, buffer_size);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(std::vector<string> filenames, const string& schema, int64 buffer_size)
        : filenames_(std::move(filenames)),
          options_(AvroReaderOptions::CreateAvroReaderOptions(schema, buffer_size)) {
    }

    std::unique_ptr<IteratorBase> MakeIterator(const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::AvroRecord")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() override { return "AvroRecordDatasetOp::Dataset"; }

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
	    // It may make more sense to simply allocate this on the
	    // GPU to start with. Or at least make that an option.
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
          const string& next_filename = dataset()->filenames_[current_file_index_];

          TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(next_filename, &file_));

          uint64 file_size;
          TF_RETURN_IF_ERROR(ctx->env()->GetFileSize(next_filename, &file_size));

          reader_.reset(new SequentialAvroRecordReader(file_.get(), file_size, next_filename, dataset()->options_));
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
    AvroReaderOptions options_;
  };
};


REGISTER_KERNEL_BUILDER(Name("AvroRecordDataset").Device(DEVICE_CPU), AvroRecordDatasetOp);
