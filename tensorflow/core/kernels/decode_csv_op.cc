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

// See docs in ../ops/parsing_ops.cc.
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

class DecodeCSVOp : public OpKernel {
 public:
  explicit DecodeCSVOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string delim;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("OUT_TYPE", &out_type_));
    OP_REQUIRES(ctx, out_type_.size() < std::numeric_limits<int>::max(),
                errors::InvalidArgument("Out type too large"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_delim", &delim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_quote_delim", &use_quote_delim_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("select_cols", &select_cols_));
    OP_REQUIRES(
        ctx, out_type_.size() == select_cols_.size() || select_cols_.empty(),
        errors::InvalidArgument("select_cols should match output size"));
    select_all_cols_ = select_cols_.empty();
    for (int i = 1; i < select_cols_.size(); i++) {
      OP_REQUIRES(ctx, select_cols_[i - 1] < select_cols_[i],
                  errors::InvalidArgument(
                      "select_cols should be strictly increasing indices"));
    }
    OP_REQUIRES(
        ctx, select_cols_.empty() || select_cols_.front() >= 0,
        errors::InvalidArgument("select_cols should be non-negative indices"));
    OP_REQUIRES(ctx, delim.size() == 1,
                errors::InvalidArgument("field_delim should be only 1 char"));
    delim_ = delim[0];
    OP_REQUIRES_OK(ctx, ctx->GetAttr("na_value", &na_value_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* records;
    OpInputList record_defaults;

    OP_REQUIRES_OK(ctx, ctx->input("records", &records));
    OP_REQUIRES_OK(ctx, ctx->input_list("record_defaults", &record_defaults));

    for (int i = 0; i < record_defaults.size(); ++i) {
      OP_REQUIRES(ctx, record_defaults[i].dims() <= 1,
                  errors::InvalidArgument(
                      "Each record default should be at most rank 1"));
      OP_REQUIRES(ctx, record_defaults[i].NumElements() < 2,
                  errors::InvalidArgument(
                      "There should only be 1 default per field but field ", i,
                      " has ", record_defaults[i].NumElements()));
    }

    auto records_t = records->flat<tstring>();
    int64_t records_size = records_t.size();

    OpOutputList output;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));

    for (int i = 0; i < static_cast<int>(out_type_.size()); ++i) {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, output.allocate(i, records->shape(), &out));
    }

    for (int64_t i = 0; i < records_size; ++i) {
      const absl::string_view record(records_t(i));
      std::vector<string> fields;
      ExtractFields(ctx, record, &fields);
      OP_REQUIRES(ctx, fields.size() == out_type_.size(),
                  errors::InvalidArgument("Expect ", out_type_.size(),
                                          " fields but have ", fields.size(),
                                          " in record ", i));

      // Check each field in the record
      for (int f = 0; f < static_cast<int>(out_type_.size()); ++f) {
        const DataType& dtype = out_type_[f];
        switch (dtype) {
          case DT_INT32: {
            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));

              output[f]->flat<int32>()(i) = record_defaults[f].flat<int32>()(0);
            } else {
              int32_t value;
              OP_REQUIRES(ctx, absl::SimpleAtoi(fields[f], &value),
                          errors::InvalidArgument(
                              "Field ", f, " in record ", i,
                              " is not a valid int32: ", fields[f]));
              output[f]->flat<int32>()(i) = value;
            }
            break;
          }
          case DT_INT64: {
            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));

              output[f]->flat<int64_t>()(i) =
                  record_defaults[f].flat<int64_t>()(0);
            } else {
              int64_t value;
              OP_REQUIRES(ctx, absl::SimpleAtoi(fields[f], &value),
                          errors::InvalidArgument(
                              "Field ", f, " in record ", i,
                              " is not a valid int64: ", fields[f]));
              output[f]->flat<int64_t>()(i) = value;
            }
            break;
          }
          case DT_FLOAT: {
            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));
              output[f]->flat<float>()(i) = record_defaults[f].flat<float>()(0);
            } else {
              float value;
              OP_REQUIRES(ctx, absl::SimpleAtof(fields[f], &value),
                          errors::InvalidArgument(
                              "Field ", f, " in record ", i,
                              " is not a valid float: ", fields[f]));
              output[f]->flat<float>()(i) = value;
            }
            break;
          }
          case DT_DOUBLE: {
            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));
              output[f]->flat<double>()(i) =
                  record_defaults[f].flat<double>()(0);
            } else {
              double value;
              OP_REQUIRES(ctx, absl::SimpleAtod(fields[f], &value),
                          errors::InvalidArgument(
                              "Field ", f, " in record ", i,
                              " is not a valid double: ", fields[f]));
              output[f]->flat<double>()(i) = value;
            }
            break;
          }
          case DT_STRING: {
            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));
              output[f]->flat<tstring>()(i) =
                  record_defaults[f].flat<tstring>()(0);
            } else {
              output[f]->flat<tstring>()(i) = std::move(fields[f]);
            }
            break;
          }
          default:
            OP_REQUIRES(ctx, false,
                        errors::InvalidArgument("csv: data type ", dtype,
                                                " not supported in field ", f));
        }
      }
    }
  }

 private:
  std::vector<DataType> out_type_;
  std::vector<int64_t> select_cols_;
  char delim_;
  bool use_quote_delim_;
  bool select_all_cols_;
  string na_value_;

  void ExtractFields(OpKernelContext* ctx, absl::string_view input,
                     std::vector<string>* result) {
    int64_t current_idx = 0;
    int64_t num_fields_parsed = 0;
    int64_t selector_idx = 0;  // Keep track of index into select_cols

    if (!input.empty()) {
      while (static_cast<size_t>(current_idx) < input.size()) {
        if (input[current_idx] == '\n' || input[current_idx] == '\r') {
          current_idx++;
          continue;
        }

        bool quoted = false;
        bool include =
            (select_all_cols_ || select_cols_[selector_idx] ==
                                     static_cast<size_t>(num_fields_parsed));

        if (use_quote_delim_ && input[current_idx] == '"') {
          quoted = true;
          current_idx++;
        }

        // This is the body of the field;
        string field;
        if (!quoted) {
          while (static_cast<size_t>(current_idx) < input.size() &&
                 input[current_idx] != delim_) {
            OP_REQUIRES(ctx,
                        (!use_quote_delim_ || input[current_idx] != '"') &&
                            input[current_idx] != '\n' &&
                            input[current_idx] != '\r',
                        errors::InvalidArgument(
                            "Unquoted fields cannot have quotes/CRLFs inside"));
            if (include) field += input[current_idx];
            current_idx++;
          }

          // Go to next field or the end
          current_idx++;
        } else if (use_quote_delim_) {
          // Quoted field needs to be ended with '"' and delim or end
          while (
              (static_cast<size_t>(current_idx) < input.size() - 1) &&
              (input[current_idx] != '"' || input[current_idx + 1] != delim_)) {
            if (input[current_idx] != '"') {
              if (include) field += input[current_idx];
              current_idx++;
            } else {
              OP_REQUIRES(
                  ctx, input[current_idx + 1] == '"',
                  errors::InvalidArgument("Quote inside a string has to be "
                                          "escaped by another quote"));
              if (include) field += '"';
              current_idx += 2;
            }
          }

          OP_REQUIRES(
              ctx,
              (static_cast<size_t>(current_idx) < input.size() &&
               input[current_idx] == '"' &&
               (static_cast<size_t>(current_idx) == input.size() - 1 ||
                input[current_idx + 1] == delim_)),
              errors::InvalidArgument("Quoted field has to end with quote "
                                      "followed by delim or end"));

          current_idx += 2;
        }

        num_fields_parsed++;
        if (include) {
          result->push_back(field);
          selector_idx++;
          if (selector_idx == select_cols_.size()) return;
        }
      }

      bool include =
          (select_all_cols_ || select_cols_[selector_idx] ==
                                   static_cast<size_t>(num_fields_parsed));
      // Check if the last field is missing
      if (include && input[input.size() - 1] == delim_)
        result->push_back(string());
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("DecodeCSV").Device(DEVICE_CPU), DecodeCSVOp);

}  // namespace tensorflow
