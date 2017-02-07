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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/initializable_lookup_table.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace lookup {
namespace {

static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */
static const int kLineNumber = -1;
static const int kWholeLine = -2;

// Iterator to initialize tables given 'keys' and 'values' tensors.
//
// The two tensors are returned in the first iteration. It doesn't loop
// over each element of the tensor since insertions in the lookup table can
// process batches.
class KeyValueTensorIterator
    : public InitializableLookupTable::InitTableIterator {
 public:
  // keys and values are not owned by the iterator.
  explicit KeyValueTensorIterator(const Tensor* keys, const Tensor* values)
      : keys_(keys), values_(values), valid_(true), status_(Status::OK()) {
    TensorShape key_shape = keys_->shape();
    if (!key_shape.IsSameSize(values_->shape())) {
      valid_ = false;
      status_ = errors::InvalidArgument(
          "keys and values should have the same dimension.",
          key_shape.DebugString(), " vs ", values_->shape().DebugString());
    }
    if (key_shape.num_elements() == 0) {
      valid_ = false;
      status_ =
          errors::InvalidArgument("keys and values cannot be empty tensors.");
    }
  }

  bool Valid() const override { return valid_; }

  void Next() override {
    valid_ = false;
    status_ = errors::OutOfRange("No more data.");
  }

  const Tensor& keys() const override { return *keys_; }

  const Tensor& values() const override { return *values_; }

  Status status() const override { return status_; }

  int64 total_size() const override {
    return keys_ == nullptr ? -1 : keys_->NumElements();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(KeyValueTensorIterator);

  const Tensor* keys_;    // Doesn't own it.
  const Tensor* values_;  // Doesn't own it.
  bool valid_;            // true if the iterator points to an existing range.
  Status status_;
};

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
    if (vocab_size == -1) {
      TF_RETURN_IF_ERROR(GetNumLinesInTextFile(env, filename, &vocab_size));
    }
    filename_ = filename;
    vocab_size_ = vocab_size;
    delimiter_ = delimiter;
    key_ = Tensor(key_dtype, TensorShape({}));
    value_ = Tensor(value_dtype, TensorShape({}));
    key_index_ = key_index;
    value_index_ = value_index;

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
      if (errors::IsOutOfRange(status_) && next_id_ != vocab_size_) {
        status_ = errors::InvalidArgument("Invalid vocab_size in ", filename_,
                                          ": expected ", vocab_size_,
                                          " but got ", next_id_);
      }
      valid_ = false;
      return;
    }
    if (next_id_ >= vocab_size_) {
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
    status_ = SetValue(line, tokens, key_index_, key_.dtype(), &key_);
    if (!status_.ok()) {
      valid_ = false;
      return;
    }
    status_ = SetValue(line, tokens, value_index_, value_.dtype(), &value_);
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

  int64 total_size() const override { return vocab_size_; }

 private:
  Tensor key_;
  Tensor value_;
  bool valid_;  // true if the iterator points to an existing range.
  int64 key_index_;
  int64 value_index_;
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
                  int64 index, DataType dtype, Tensor* tensor) {
    if (index == kLineNumber) {
      tensor->flat<int64>()(0) = next_id_;
      return Status::OK();
    }
    if (index == kWholeLine) {
      tensor->flat<string>()(0) = line;
      return Status::OK();
    }
    const string& token = tokens[index];
    switch (tensor->dtype()) {
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
        return errors::InvalidArgument("Data type ", dtype, " not supported.");
    }
    return Status::OK();
  }

  TF_DISALLOW_COPY_AND_ASSIGN(TextFileLineIterator);
};

// Helper function to initialize an InitializableLookupTable from a text file.
Status InitializeTableFromTextFile(const string& filename, int64 vocab_size,
                                   char delimiter, int32 key_index,
                                   int32 value_index, Env* env,
                                   InitializableLookupTable* table) {
  if (key_index == kLineNumber && table->key_dtype() != DT_INT64) {
    return errors::InvalidArgument(
        "Key index for line number requires table key dtype of int64, got ",
        table->key_dtype());
  }
  if (key_index == kWholeLine && table->key_dtype() != DT_STRING) {
    return errors::InvalidArgument(
        "Key index for whole line requires table key dtype of string, got ",
        table->key_dtype());
  }
  if (value_index == kLineNumber && table->value_dtype() != DT_INT64) {
    return errors::InvalidArgument(
        "Value index for line number requires table value dtype of int64, got ",
        table->value_dtype());
  }
  if (value_index == kWholeLine && table->value_dtype() != DT_STRING) {
    return errors::InvalidArgument(
        "Value index for whole line requires table value dtype of string, got ",
        table->value_dtype());
  }

  TextFileLineIterator iter;
  TF_RETURN_IF_ERROR(iter.Init(filename, vocab_size, delimiter,
                               table->key_dtype(), key_index,
                               table->value_dtype(), value_index, env));
  // For initialization from files, ignore if the table is already
  // initialized. The table shared name should contain the filename to
  // avoid trying to initialize the same table from the same file at the same
  // time.
  Status s = table->Initialize(iter);
  if (errors::IsFailedPrecondition(s) && table->is_initialized()) {
    LOG(WARNING) << "Table trying to initialize from file " << filename
                 << " is already initialized.";
    return Status::OK();
  }
  return s;
}

}  // namespace
}  // namespace lookup

// Kernel to initialize a look table given a key and value tensors.
// After this operation, the table becomes read-only.
class InitializeTableOp : public OpKernel {
 public:
  explicit InitializeTableOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    lookup::InitializableLookupTable* table;
    OP_REQUIRES_OK(ctx,
                   GetInitializableLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {DT_STRING_REF, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& keys = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(keys.shape()),
                errors::InvalidArgument("Keys must be a vector, but received ",
                                        keys.shape().DebugString()));

    const Tensor& values = ctx->input(2);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(values.shape()),
        errors::InvalidArgument("Values must be a vector, but received ",
                                values.shape().DebugString()));

    OP_REQUIRES(ctx, keys.NumElements() == values.NumElements(),
                errors::InvalidArgument(
                    "Keys and values must have the same size ",
                    keys.NumElements(), " vs ", values.NumElements()));

    lookup::KeyValueTensorIterator iter(&keys, &values);
    OP_REQUIRES_OK(ctx, table->Initialize(iter));
  }

 private:
  mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("InitializeTable").Device(DEVICE_CPU),
                        InitializeTableOp);

// Kernel to initialize a lookup table from a text file.
//
// After this operation, the table becomes read-only.
class InitializeTableFromTextFileOp : public OpKernel {
 public:
  explicit InitializeTableFromTextFileOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_size", &vocab_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_index", &key_index_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value_index", &value_index_));
    string delimiter;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("delimiter", &delimiter));
    OP_REQUIRES(ctx, delimiter.size() == 1,
                errors::InvalidArgument("delimiter should be only 1 char"));
    delimiter_ = delimiter[0];
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    lookup::InitializableLookupTable* table;
    OP_REQUIRES_OK(ctx,
                   GetInitializableLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {DT_STRING_REF, DT_STRING};
    DataTypeVector expected_outputs = {};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& vocab_filename_tensor = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(vocab_filename_tensor.shape()),
        errors::InvalidArgument("filename should be a single string, but got",
                                vocab_filename_tensor.shape().DebugString()));

    string vocab_filename = vocab_filename_tensor.scalar<string>()();
    OP_REQUIRES(ctx, !vocab_filename.empty(),
                errors::InvalidArgument("filename cannot be empty."));

    OP_REQUIRES_OK(ctx, lookup::InitializeTableFromTextFile(
                            vocab_filename, vocab_size_, delimiter_, key_index_,
                            value_index_, ctx->env(), table));
  }

 private:
  mutex mu_;
  int64 vocab_size_;
  char delimiter_;
  int64 key_index_;
  int64 value_index_;

  TF_DISALLOW_COPY_AND_ASSIGN(InitializeTableFromTextFileOp);
};

REGISTER_KERNEL_BUILDER(Name("InitializeTableFromTextFile").Device(DEVICE_CPU),
                        InitializeTableFromTextFileOp);

}  // namespace tensorflow
