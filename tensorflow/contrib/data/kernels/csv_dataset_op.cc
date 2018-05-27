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

// See docs in ../ops/parsing_ops.cc.
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"

namespace tensorflow {
namespace {

class CSVDatasetOp : public DatasetOpKernel {
 public:
  explicit CSVDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    OpInputList record_defaults_list;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("record_defaults", &record_defaults_list));
    for (int i = 0; i < record_defaults_list.size(); ++i) {
      OP_REQUIRES(ctx, record_defaults_list[i].NumElements() < 2,
                  errors::InvalidArgument(
                      "There should only be 1 default per field but field ", i,
                      " has ", record_defaults_list[i].NumElements()));
    }

    const Tensor* select_cols_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("select_cols", &select_cols_tensor));
    OP_REQUIRES(ctx, select_cols_tensor->dims() == 1,
                errors::InvalidArgument("`select_cols` must be a vector."));

    int64 buffer_size;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(ctx, buffer_size > 0,
                errors::InvalidArgument("buffer_size should be positive"));

    string delim;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<string>(ctx, "field_delim", &delim));
    OP_REQUIRES(ctx, delim.size() == 1,
                errors::InvalidArgument("field_delim should be only 1 char"));

    bool header;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "header", &header));

    bool use_quote_delim;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "use_quote_delim",
                                                  &use_quote_delim));
    string na_value;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<string>(ctx, "na_value", &na_value));

    std::vector<Tensor> record_defaults;
    record_defaults.reserve(record_defaults_list.size());
    for (const Tensor& t : record_defaults_list) {
      record_defaults.push_back(t);
    }

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    std::vector<int64> select_cols;
    select_cols.reserve(select_cols_tensor->NumElements());
    for (int i = 0; i < select_cols_tensor->NumElements(); ++i) {
      select_cols.push_back(select_cols_tensor->flat<int64>()(i));
    }
    OP_REQUIRES(
        ctx, output_types_.size() == select_cols.size() || select_cols.empty(),
        errors::InvalidArgument("select_cols should match output size"));
    for (int i = 1; i < select_cols.size(); i++) {
      OP_REQUIRES(ctx, select_cols[i - 1] < select_cols[i],
                  errors::InvalidArgument(
                      "select_cols should be strictly increasing indices"));
    }
    OP_REQUIRES(
        ctx, select_cols.empty() || select_cols.front() >= 0,
        errors::InvalidArgument("select_cols should be non-negative indices"));
    bool select_all_cols = select_cols.empty();

    *output = new Dataset(
        ctx, std::move(filenames), header, buffer_size, output_types_,
        output_shapes_, std::move(record_defaults), std::move(select_cols),
        select_all_cols, use_quote_delim, delim[0], std::move(na_value));
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<string> filenames, bool header,
            int64 buffer_size, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            std::vector<Tensor> record_defaults, std::vector<int64> select_cols,
            bool select_all_cols, bool use_quote_delim, char delim,
            string na_value)
        : GraphDatasetBase(ctx),
          filenames_(std::move(filenames)),
          header_(header),
          buffer_size_(buffer_size),
          out_type_(output_types),
          output_shapes_(output_shapes),
          record_defaults_(std::move(record_defaults)),
          select_cols_(std::move(select_cols)),
          select_all_cols_(select_all_cols),
          use_quote_delim_(use_quote_delim),
          delim_(delim),
          na_value_(std::move(na_value)) {}

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::CSV")}));
    }

    const DataTypeVector& output_dtypes() const override { return out_type_; }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "CSVDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                              Node** output) const override {
      // TODO(rachelim): Implement this
      std::vector<Node*> input_tensors;
      TF_RETURN_IF_ERROR(b->AddDataset(this, input_tensors, output));
      return errors::Unimplemented("CSVDataset: AsGraphDefInternal");
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
          // We are currently processing a file, so try to read the next record
          if (buffered_input_stream_) {
            Status s = ReadRecord(ctx, out_tensors);
            if (s.ok() || !errors::IsOutOfRange(s)) {
              // Not at the end of file, return OK or non-EOF errors to caller.
              *end_of_sequence = false;
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
        // TODO(rachelim): Implement save
        return errors::Unimplemented("CSVDataset: SaveInternal");
      }
      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        // TODO(rachelim): Implement restore
        return errors::Unimplemented("CSVDataset: RestoreInternal");
      }

     private:
      // Reads a record by parsing the input buffer, and converting extracted
      // fields to output tensors as we go.
      Status ReadRecord(IteratorContext* ctx, std::vector<Tensor>* out_tensors)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        // Extracts fields from line(s) from the buffered input stream.
        out_tensors->reserve(dataset()->record_defaults_.size());

        string input;
        TF_RETURN_IF_ERROR(buffered_input_stream_->ReadLine(&input));

        size_t current_idx = 0;
        size_t num_fields_parsed = 0;
        size_t selector_idx = 0;  // Keep track of index into select_cols

        while (current_idx < input.size()) {
          // In each iteration, parse one field
          if (input[current_idx] == '\n' || input[current_idx] == '\r') {
            // This should never happen, because buffered input reader splits
            // input on newlines.
            return errors::InvalidArgument("Parsing error.");
          }

          bool quoted = false;
          bool include =
              (dataset()->select_all_cols_ ||
               dataset()->select_cols_[selector_idx] == num_fields_parsed);

          if (dataset()->use_quote_delim_ && input[current_idx] == '"') {
            quoted = true;
            current_idx++;
          }

          // Parse the body of the field
          string field;
          if (!quoted) {
            while (current_idx < input.size() &&
                   input[current_idx] != dataset()->delim_) {
              if ((dataset()->use_quote_delim_ && input[current_idx] == '"') ||
                  input[current_idx] == '\n' || input[current_idx] == '\r') {
                return errors::InvalidArgument(
                    "Unquoted fields cannot have quotes/CRLFs inside");
              }
              if (include) field += input[current_idx];
              current_idx++;
            }  // Exit condition: end of input, or current index at delim

            // Go to next field or the end
            current_idx++;
          } else {
            // Quoted field needs to be ended with '"' and delim or end
            while (true) {
              if (current_idx >= input.size() - 1 || input.empty()) {
                if (current_idx == input.size() - 1 &&
                    input[current_idx] == '"') {
                  // We're at the end of the input, and the quote terminates the
                  // record. Go to end.
                  current_idx++;
                  break;
                }
                // If there's no terminating quote, it means our buffered record
                // line reader split a record up. This can happen if there is a
                // newline encased in quotes. The next line is also part of the
                // record, so we read it and reset the index.
                if (include && current_idx == input.size() - 1) {
                  // TODO(rachelim): Instead of building up a string, keep track
                  //  of terminal indices (or starting char* and length)
                  // Also look into using /lib/strings/Scanner
                  field += input[current_idx];
                }
                if (include) {
                  field += '\n';
                }
                current_idx = 0;
                Status s = buffered_input_stream_->ReadLine(&input);
                if (!s.ok()) {
                  return errors::InvalidArgument(
                      "Quoted field has to end with quote followed by delim, "
                      "CRLF, or EOF");
                }
              } else if (input[current_idx] == '"' &&
                         input[current_idx + 1] == dataset()->delim_) {
                // End of field, go to next field or end
                current_idx += 2;
                break;
              } else if (input[current_idx] == '"') {
                // Current char is a quote. Since we're not at end of field,
                // the next character must also be a quote.
                if (input[current_idx + 1] != '"') {
                  return errors::InvalidArgument(
                      "Quote inside a string has to be escaped by another "
                      "quote");
                }
                if (include) field += '"';
                current_idx += 2;
              } else {
                if (include) field += input[current_idx];
                current_idx++;
              }
            }
          }

          num_fields_parsed++;

          if (include) {
            // Add the tensor to the result
            TF_RETURN_IF_ERROR(FieldToOutput(ctx, std::move(field),
                                             selector_idx, out_tensors));
            selector_idx++;
            // Terminate early if we have all the fields we want
            if (selector_idx == dataset()->select_cols_.size())
              return Status::OK();
          }
        }  // Exit condition: current_idx has reached the end of record

        // Check if the last field is empty, and include it if necessary
        bool include =
            (dataset()->select_all_cols_ ||
             dataset()->select_cols_[selector_idx] == num_fields_parsed);
        if (include && !input.empty() &&
            input[input.size() - 1] == dataset()->delim_) {
          TF_RETURN_IF_ERROR(
              FieldToOutput(ctx, string(), selector_idx, out_tensors));
        }

        // Check that number of fields matches
        if (out_tensors->size() != dataset()->out_type_.size()) {
          return errors::InvalidArgument("Expect ", dataset()->out_type_.size(),
                                         " fields but have ",
                                         out_tensors->size(), " in record");
        }
        return Status::OK();
      }

      // Given a string field, and its index in the output,
      // converts it to a Tensor of the right type and adds it to the
      // out_tensors vector.
      Status FieldToOutput(IteratorContext* ctx, string field,
                           size_t output_idx,
                           std::vector<Tensor>* out_tensors) {
        if (output_idx >= dataset()->out_type_.size()) {
          // We can get here if we're selecting all columns, but the number of
          // fields exceeds the number of defaults provided
          return errors::InvalidArgument("Expect ", dataset()->out_type_.size(),
                                         " fields but have more in record");
        }
        const DataType& dtype = dataset()->out_type_[output_idx];
        Tensor component(ctx->allocator({}), dtype, {});
        if ((field.empty() || field == dataset()->na_value_) &&
            dataset()->record_defaults_[output_idx].NumElements() != 1) {
          // If the field is empty or NA value, and default is not given,
          // report error.
          return errors::InvalidArgument("Field ", output_idx,
                                         " is required but missing in record!");
        }

        switch (dtype) {
          // For each case, if the field is empty, we use the default.
          // Otherwise, we convert it to the right type.
          case DT_INT32: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<int32>()() =
                  dataset()->record_defaults_[output_idx].flat<int32>()(0);
            } else {
              int32 value;
              if (!strings::safe_strto32(field, &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid int32: ", field);
              }
              component.scalar<int32>()() = value;
            }
            break;
          }
          case DT_INT64: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<int64>()() =
                  dataset()->record_defaults_[output_idx].flat<int64>()(0);
            } else {
              int64 value;
              if (!strings::safe_strto64(field, &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid int64: ", field);
              }
              component.scalar<int64>()() = value;
            }
            break;
          }
          case DT_FLOAT: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<float>()() =
                  dataset()->record_defaults_[output_idx].flat<float>()(0);
            } else {
              float value;
              if (!strings::safe_strtof(field.c_str(), &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid float: ", field);
              }
              component.scalar<float>()() = value;
            }
            break;
          }
          case DT_DOUBLE: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<double>()() =
                  dataset()->record_defaults_[output_idx].flat<double>()(0);
            } else {
              double value;
              if (!strings::safe_strtod(field.c_str(), &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid double: ", field);
              }
              component.scalar<double>()() = value;
            }
            break;
          }
          case DT_STRING: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<string>()() =
                  dataset()->record_defaults_[output_idx].flat<string>()(0);
            } else {
              component.scalar<string>()() = std::move(field);
            }
            break;
          }
          default:
            return errors::InvalidArgument("csv: data type ", dtype,
                                           " not supported in field ",
                                           output_idx);
        }
        out_tensors->push_back(std::move(component));
        return Status::OK();
      }

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
        // TODO(rachelim): Maintain our own buffer so we don't read every record
        //   twice
        buffered_input_stream_.reset(new io::BufferedInputStream(
            input_stream_.get(), dataset()->buffer_size_, false));
        if (dataset()->header_) {
          // Ignore header line
          string str;
          Status s = buffered_input_stream_->ReadLine(&str);
          if (errors::IsOutOfRange(s)) {
            return errors::InvalidArgument("Can't read header of empty file");
          }
        }
        return Status::OK();
      }

      // Resets all reader streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        input_stream_.reset();
        buffered_input_stream_.reset();
        file_.reset();
      }

      mutex mu_;
      std::unique_ptr<io::RandomAccessInputStream> input_stream_
          GUARDED_BY(mu_);
      std::unique_ptr<io::BufferedInputStream> buffered_input_stream_
          GUARDED_BY(mu_);
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<RandomAccessFile> file_
          GUARDED_BY(mu_);  // must outlive input_stream_
    };                      // class Iterator

    const std::vector<string> filenames_;
    const bool header_;
    const int64 buffer_size_;
    const DataTypeVector out_type_;
    const std::vector<PartialTensorShape> output_shapes_;
    const std::vector<Tensor> record_defaults_;
    const std::vector<int64> select_cols_;
    const bool select_all_cols_;
    const bool use_quote_delim_;
    const char delim_;
    const string na_value_;
  };  // class Dataset

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};  // class CSVDatasetOp

// Register the kernel implementation for CSVDataset.
REGISTER_KERNEL_BUILDER(Name("CSVDataset").Device(DEVICE_CPU), CSVDatasetOp);

}  // namespace
}  // namespace tensorflow
