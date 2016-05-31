/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/dynamicsplit_ops.cc.
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

class DynamicSplitOp : public OpKernel {
 public:
  explicit DynamicSplitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string delim;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("OUT_TYPE", &out_type_));
    //OP_REQUIRES_OK(ctx, ctx->GetAttr("INPUT_TYPE", &input_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_delim", &delim));

    OP_REQUIRES(ctx, delim.size() == 1,
                errors::InvalidArgument("field_delim should be only 1 char"));

    delim_ = delim[0];
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* records;
    const Tensor* record_defaults;
    printf("in compute\n");
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));
    OP_REQUIRES_OK(ctx, ctx->input("record_defaults", &record_defaults));
    printf("after\n");

    auto records_t = records->flat<string>();
    int64 records_size = records_t.size();
    OP_REQUIRES(ctx, records_size == 1, 
                errors::InvalidArgument("There should be only 1 record but has "                , records_size));

    OpOutputList output;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));

    for (int64 i = 0; i < records_size; ++i) {
      const StringPiece record(records_t(i));
      std::vector<string> fields;
      ExtractFields(ctx, record, &fields);


      Tensor *out = nullptr;
      output.allocate(0, records->shape(), &out);
   
      std::cout<< "aa" <<std::endl;
      // Check each field in the record
      for (int f = 0; f < fields.size(); ++f) {
        const DataType& dtype = out_type_;
        switch (dtype) {
          case DT_INT32: {
            int32 value;
            OP_REQUIRES(ctx, strings::safe_strto32(fields[f], &value),
                          errors::InvalidArgument("Field ", f, " in record ", i,
                                                  " is not a valid int32: ",
                                                  fields[f]));
            output[0]->flat<int32>()(f) = value;
            break;
          }
          case DT_INT64: {
            int64 value;
            OP_REQUIRES(ctx, strings::safe_strto64(fields[f], &value),
                          errors::InvalidArgument("Field ", f, " in record ", i,
                                                  " is not a valid int64: ",
                                                  fields[f]));
            output[0]->flat<int64>()(f) = value;
            break;
          }
          case DT_FLOAT: {
            float value;
            OP_REQUIRES(ctx, strings::safe_strtof(fields[f].c_str(), &value),
                          errors::InvalidArgument("Field ", f, " in record ", i,
                                                  " is not a valid float: ",
                                                  fields[f]));
            output[0]->flat<float>()(f) = value;
            break;
          }
          case DT_STRING: {
            std::cout << fields[f] << std::endl;
            output[0]->flat<string>()(f) = fields[f];
            std::cout <<"aaaa"<<std::endl;
            break;
          }
          default:
            OP_REQUIRES(ctx, false,
                        errors::InvalidArgument("csv: data type ", dtype,
                                                " not supported in field ", f));
        }
      }
    }
    printf("end\n");
  }

 private:
  DataType out_type_;
  char delim_;

  void ExtractFields(OpKernelContext* ctx, StringPiece input,
                     std::vector<string>* result) {
    int64 current_idx = 0;
    if (!input.empty()) {
      while (static_cast<size_t>(current_idx) < input.size()) {
        if (input[current_idx] == '\n' || input[current_idx] == '\r') {
          current_idx++;
          continue;
        }

        bool quoted = false;
        if (input[current_idx] == '"') {
          quoted = true;
          current_idx++;
        }

        // This is the body of the field;
        string field;
        if (!quoted) {
          while (static_cast<size_t>(current_idx) < input.size() &&
                 input[current_idx] != delim_) {
            OP_REQUIRES(ctx, input[current_idx] != '"' &&
                                 input[current_idx] != '\n' &&
                                 input[current_idx] != '\r',
                        errors::InvalidArgument(
                            "Unquoted fields cannot have quotes/CRLFs inside"));
            field += input[current_idx];
            current_idx++;
          }

          // Go to next field or the end
          current_idx++;
        } else {
          // Quoted field needs to be ended with '"' and delim or end
          while (
              (static_cast<size_t>(current_idx) < input.size() - 1) &&
              (input[current_idx] != '"' || input[current_idx + 1] != delim_)) {
            if (input[current_idx] != '"') {
              field += input[current_idx];
              current_idx++;
            } else {
              OP_REQUIRES(
                  ctx, input[current_idx + 1] == '"',
                  errors::InvalidArgument("Quote inside a string has to be "
                                          "escaped by another quote"));
              field += '"';
              current_idx += 2;
            }
          }

          OP_REQUIRES(
              ctx, (static_cast<size_t>(current_idx) < input.size() &&
                    input[current_idx] == '"' &&
                    (static_cast<size_t>(current_idx) == input.size() - 1 ||
                     input[current_idx + 1] == delim_)),
              errors::InvalidArgument("Quoted field has to end with quote "
                                      "followed by delim or end"));

          current_idx += 2;
        }

        result->push_back(field);
      }

      // Check if the last field is missing
      if (input[input.size() - 1] == delim_) result->push_back(string());
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("DynamicSplit").Device(DEVICE_CPU), DynamicSplitOp);

}  // namespace tensorflow
