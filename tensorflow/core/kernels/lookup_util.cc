/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/lookup_util.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"

namespace tensorflow {
namespace lookup {
namespace {

static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */
static const int kLineNumber = -1;
static const int kWholeLine = -2;

Status GetNumLinesInTextFile(Env* env, const string& vocab_file,
                             int64* num_lines) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(vocab_file, &file));

  io::InputBuffer input_buffer(file.get(), kInputBufferSize);
  string line;
  Status s = input_buffer.ReadLine(&line);
  int64 next_id = 0;
  while (s.ok()) {
    next_id++;
    s = input_buffer.ReadLine(&line);
  }
  if (!errors::IsOutOfRange(s)) {
    return s;
  }
  *num_lines = next_id;
  return Status::OK();
}

// Iterator that reads a text file. Each iteration process one line, it parses
// the line and populates the keys and values tensors used for initialization
// with a single key and corresponding value.
//
// What information of the line to populate the key or values is specified by
// providing key_index and value_index.
class TextFileLineIterator
    : public InitializableLookupTable::InitTableIterator {
 public:
  TextFileLineIterator()
      : valid_(false),
        vocab_size_(-1),
        status_(errors::FailedPrecondition("Not initialized")) {}

  // Initialize iterator.
  //
  // Prepares the file 'filename' and sets the data types to return the keys and
  // values tensors. It requires the indices of the tokens in the line given a
  // delimiter to specify where to pick the data from.
  //
  // - Index -2 means the entire line as string.
  // - Index -1 means the line number stored in int64.
  // - Index >= 0 represent index (starting at zero) of the split line based on
  //   delimiter.
  Status Init(const string& filename, int64 vocab_size, char delimiter,
              DataType key_dtype, int64 key_index, DataType value_dtype,
              int64 value_index, Env* env) {
    filename_ = filename;
    vocab_size_ = vocab_size;
    delimiter_ = delimiter;
    key_ = Tensor(key_dtype, TensorShape({}));
    value_ = Tensor(value_dtype, TensorShape({}));
    key_index_ = key_index;
    value_index_ = value_index;
    env_ = env;

    status_ = env->NewRandomAccessFile(filename_, &file_);
    if (!status_.ok()) return status_;

    input_buffer_.reset(new io::InputBuffer(file_.get(), kInputBufferSize));
    valid_ = true;
    next_id_ = 0;
    ignore_split_ = std::max(key_index_, value_index_) < 0;
    Next();
    return status_;
  }

  void Next() override {
    if (!valid_) return;

    string line;
    status_ = input_buffer_->ReadLine(&line);
    if (!status_.ok()) {
      if (errors::IsOutOfRange(status_) && vocab_size_ != -1 &&
          next_id_ != vocab_size_) {
        status_ = errors::InvalidArgument("Invalid vocab_size in ", filename_,
                                          ": expected ", vocab_size_,
                                          " but got ", next_id_);
      }
      valid_ = false;
      return;
    }
    if (vocab_size_ != -1 && next_id_ >= vocab_size_) {
      LOG(WARNING) << "Truncated " << filename_ << " before its end at "
                   << vocab_size_ << " records.";
      LOG(WARNING) << "next_id_  : " << next_id_;
      status_ = errors::OutOfRange("Finished reading ", vocab_size_,
                                   " of lines from ", filename_);
      valid_ = false;
      return;
    }
    if (line.empty()) {
      status_ = errors::InvalidArgument("Invalid content in ", filename_,
                                        ": empty line found at position ",
                                        input_buffer_->Tell(), ".");
      valid_ = false;
      return;
    }

    std::vector<string> tokens;
    if (!ignore_split_) {
      tokens = str_util::Split(line, delimiter_);
      if (std::max(key_index_, value_index_) >= tokens.size()) {
        status_ = errors::InvalidArgument(
            "Invalid number of columns in ", filename_, " line ", next_id_,
            " (", line, ") : expected ", std::max(key_index_, value_index_),
            " got ", tokens.size());
        valid_ = false;
        return;
      }
    }
    status_ = SetValue(line, tokens, key_index_, &key_);
    if (!status_.ok()) {
      valid_ = false;
      return;
    }
    status_ = SetValue(line, tokens, value_index_, &value_);
    if (!status_.ok()) {
      valid_ = false;
      return;
    }

    next_id_++;
  }

  bool Valid() const override { return valid_; }

  const Tensor& keys() const override { return key_; }

  const Tensor& values() const override { return value_; }

  Status status() const override { return status_; }

  int64 total_size() const override {
    if (vocab_size_ == -1) {
      int64 new_size;
      Status status = GetNumLinesInTextFile(env_, filename_, &new_size);
      if (!status.ok()) {
        LOG(WARNING) << "Unable to get line count: " << status;
        new_size = -1;
      }
      *const_cast<int64*>(&vocab_size_) = new_size;
    }
    return vocab_size_;
  }

 private:
  Tensor key_;
  Tensor value_;
  bool valid_;  // true if the iterator points to an existing range.
  int64 key_index_;
  int64 value_index_;
  Env* env_;
  int64 next_id_;
  int64 vocab_size_;
  string filename_;
  char delimiter_;
  Status status_;
  bool ignore_split_;
  std::unique_ptr<RandomAccessFile> file_;  // must outlive input_buffer_
  std::unique_ptr<io::InputBuffer> input_buffer_;

  // Set the corresponding value from line or tokens based on 'index' into the
  // tensor 't'. The value is transformed to the given data type 'dtype'.
  Status SetValue(const string& line, const std::vector<string>& tokens,
                  int64 index, Tensor* tensor) {
    if (index == kLineNumber) {
      tensor->flat<int64>()(0) = next_id_;
      return Status::OK();
    }
    const string& token = (index == kWholeLine) ? line : tokens[index];
    const DataType& dtype = tensor->dtype();
    switch (dtype) {
      case DT_INT32: {
        int32 value;
        if (!strings::safe_strto32(token.c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", token, " in line ", next_id_,
                                         " is not a valid int32.");
        }
        tensor->flat<int32>()(0) = value;
      } break;
      case DT_INT64: {
        int64 value;
        if (!strings::safe_strto64(token.c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", token, " in line ", next_id_,
                                         " is not a valid int64.");
        }
        tensor->flat<int64>()(0) = value;
      } break;
      case DT_FLOAT: {
        float value;
        if (!strings::safe_strtof(token.c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", token, " in line ", next_id_,
                                         " is not a valid float.");
        }
        tensor->flat<float>()(0) = value;
      } break;
      case DT_DOUBLE: {
        double value;
        if (!strings::safe_strtod(token.c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", token, " in line ", next_id_,
                                         " is not a valid double.");
        }
        tensor->flat<double>()(0) = value;
      } break;
      case DT_STRING:
        tensor->flat<string>()(0) = token;
        break;
      default:
        valid_ = false;
        return errors::InvalidArgument("Data type ", DataTypeString(dtype),
                                       " not supported.");
    }
    return Status::OK();
  }

  TF_DISALLOW_COPY_AND_ASSIGN(TextFileLineIterator);
};

Status GetTableHandle(const string& input_name, OpKernelContext* ctx,
                      string* container, string* table_handle) {
  {
    mutex* mu;
    TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
    mutex_lock l(*mu);
    Tensor tensor;
    TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Lookup table handle must be scalar, but had shape: ",
          tensor.shape().DebugString());
    }
    auto h = tensor.flat<string>();
    *container = h(0);
    *table_handle = h(1);
  }
  return Status::OK();
}

}  // namespace

Status GetLookupTable(const string& input_name, OpKernelContext* ctx,
                      LookupInterface** table) {
  string container;
  string table_handle;
  DataType handle_dtype;
  TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &handle_dtype));
  if (handle_dtype == DT_RESOURCE) {
    ResourceHandle handle;
    TF_RETURN_IF_ERROR(HandleFromInput(ctx, input_name, &handle));
    return LookupResource(ctx, handle, table);
  } else {
    TF_RETURN_IF_ERROR(
        GetTableHandle(input_name, ctx, &container, &table_handle));
    return ctx->resource_manager()->Lookup(container, table_handle, table);
  }
}

Status GetInitializableLookupTable(const string& input_name,
                                   OpKernelContext* ctx,
                                   InitializableLookupTable** table) {
  LookupInterface* lookup_table;
  DataType handle_dtype;
  TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &handle_dtype));
  if (handle_dtype == DT_RESOURCE) {
    ResourceHandle handle;
    TF_RETURN_IF_ERROR(HandleFromInput(ctx, input_name, &handle));
    TF_RETURN_IF_ERROR(LookupResource(ctx, handle, &lookup_table));
    *table = lookup_table->GetInitializableLookupTable();
    if (*table == nullptr) {
      lookup_table->Unref();
      return errors::InvalidArgument("Table ", handle.container(), " ",
                                     handle.name(), " is not initializable");
    }
  } else {
    string container;
    string table_handle;
    TF_RETURN_IF_ERROR(
        GetTableHandle(input_name, ctx, &container, &table_handle));
    TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(container, table_handle,
                                                       &lookup_table));
    *table = lookup_table->GetInitializableLookupTable();
    if (*table == nullptr) {
      lookup_table->Unref();
      return errors::InvalidArgument("Table ", container, " ", table_handle,
                                     " is not initializable");
    }
  }
  return Status::OK();
}

Status CheckTableDataTypes(const LookupInterface& table, DataType key_dtype,
                           DataType value_dtype, const string& table_name) {
  if (table.key_dtype() != key_dtype || table.value_dtype() != value_dtype) {
    return errors::InvalidArgument(
        "Conflicting key/value dtypes ", DataTypeString(key_dtype), "->",
        DataTypeString(value_dtype), " with ",
        DataTypeString(table.key_dtype()), "-",
        DataTypeString(table.value_dtype()), " for table ", table_name);
  }
  return Status::OK();
}

// Helper function to initialize an InitializableLookupTable from a text file.
Status InitializeTableFromTextFile(const string& filename, int64 vocab_size,
                                   char delimiter, int32 key_index,
                                   int32 value_index, Env* env,
                                   InitializableLookupTable* table) {
  if (key_index == kLineNumber && table->key_dtype() != DT_INT64) {
    return errors::InvalidArgument(
        "Key index for line number requires table key dtype of int64, got ",
        DataTypeString(table->key_dtype()));
  }
  const DataType& key_dtype = table->key_dtype();
  const DataType& value_dtype = table->value_dtype();
  if (key_index == kWholeLine && !DataTypeIsInteger(key_dtype) &&
      key_dtype != DT_STRING) {
    return errors::InvalidArgument(
        "Key index for whole line requires string or integer table key, got ",
        DataTypeString(table->key_dtype()));
  }
  if (value_index == kLineNumber && value_dtype != DT_INT64) {
    return errors::InvalidArgument(
        "Value index for line number requires table value dtype of int64, got ",
        DataTypeString(table->value_dtype()));
  }
  if (value_index == kWholeLine && value_dtype != DT_STRING) {
    return errors::InvalidArgument(
        "Value index for whole line requires table value dtype of string, got ",
        DataTypeString(table->value_dtype()));
  }

  TextFileLineIterator iter;
  TF_RETURN_IF_ERROR(iter.Init(filename, vocab_size, delimiter, key_dtype,
                               key_index, value_dtype, value_index, env));
  // For initialization from files, ignore if the table is already
  // initialized. The table shared name should contain the filename to
  // avoid trying to initialize the same table from the same file at the same
  // time.
  Status s = table->Initialize(iter);
  if (errors::IsFailedPrecondition(s) && table->is_initialized()) {
    LOG(INFO) << "Table trying to initialize from file " << filename
              << " is already initialized.";
    return Status::OK();
  }
  return s;
}

}  // namespace lookup
}  // namespace tensorflow
