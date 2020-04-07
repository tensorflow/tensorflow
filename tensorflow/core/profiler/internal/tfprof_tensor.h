/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

// TFProf representation of a Tensor's value.
// 1. Multi-dimension tensor is flattened in row major, and stored in proto.
// 2. integer are up-casted to int64. floats are up-casted to double. string
//    is not supported by TensorFlow CheckPointReader library, though it is
//    supported in current code.

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TENSOR_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TENSOR_H_

#include <typeinfo>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {

class TFProfTensor {
 public:
  explicit TFProfTensor(std::unique_ptr<Tensor> tensor)
      : tensor_(std::move(tensor)) {
    Build();
  }

  // If pointers are provided, they are filled by the method.
  void Display(string* formatted_str, TFProfTensorProto* tfprof_tensor_pb);

 private:
  // Max length of tensor value displayed to CLI.
  const int64 kTFProfTenosrMaxDisplayLen = 10000;
  // Max length after which a latency warning will be printed.
  const int64 kTFProfTensorMaxWarnLen = 100000;

  void Build();

  template <typename T>
  bool AddValue(const T& value, TFProfTensorProto* dim) {
    std::ostringstream sstream;
    sstream << value;
    if (typeid(value) == typeid(double)) {
      double double_val = 0.0;
      CHECK(absl::SimpleAtod(sstream.str(), &double_val));  // Crash OK
      dim->add_value_double(double_val);
      absl::StrAppendFormat(&formatted_str_, "%.2f ",
                            dim->value_double(dim->value_double_size() - 1));
    } else if (typeid(value) == typeid(int64)) {
      int64 int64_val = 0;
      CHECK(absl::SimpleAtoi(sstream.str(), &int64_val));  // Crash OK
      dim->add_value_int64(int64_val);
      absl::StrAppendFormat(&formatted_str_, "%d ",
                            dim->value_int64(dim->value_int64_size() - 1));
    } else if (typeid(value) == typeid(string)) {
      dim->add_value_str(sstream.str());
      absl::StrAppend(&formatted_str_, "'",
                      dim->value_str(dim->value_str_size() - 1), "' ");
    } else {
      CHECK(false) << "Unsupported type: " << typeid(value).name();
    }
  }

  // It assumes the flatten values are stored in row-major, which is mentioned
  // indirectly at various places:
  // TODO(xpan): Further verifying it.
  template <typename T>
  int64 BuildOutput(int64 start, int depth, const std::vector<T>& values,
                    TFProfTensorProto* dim) {
    formatted_str_ += "[";
    int64 nstart = start;
    if (tensor_->dims() == 0 && values.size() == 1) {
      std::ostringstream sstream;
      sstream << values[nstart];

      if (typeid(values[nstart]) == typeid(double)) {
        double double_val = 0.0;
        CHECK(absl::SimpleAtod(sstream.str(), &double_val));  // Crash OK
        dim->add_value_double(double_val);
        absl::StrAppendFormat(&formatted_str_, "%.2f ",
                              dim->value_double(dim->value_double_size() - 1));
      } else if (typeid(values[nstart]) == typeid(int64)) {
        int64 int64_val = 0;
        CHECK(absl::SimpleAtoi(sstream.str(), &int64_val));  // Crash OK
        dim->add_value_int64(int64_val);
        absl::StrAppendFormat(&formatted_str_, "%d ",
                              dim->value_int64(dim->value_int64_size() - 1));
      } else if (typeid(values[nstart]) == typeid(string)) {
        dim->add_value_str(sstream.str());
        absl::StrAppend(&formatted_str_, "'",
                        dim->value_str(dim->value_str_size() - 1), "' ");
      } else {
        CHECK(false) << "Unsupported type: " << typeid(values[nstart]).name();
      }
    } else {
      for (int i = 0; i < tensor_->dim_size(depth); i++) {
        // Last dimension, pull the values.
        if (depth == tensor_->dims() - 1) {
          std::ostringstream sstream;
          sstream << values[nstart];

          if (typeid(values[nstart]) == typeid(double)) {
            double double_val = 0.0;
            CHECK(absl::SimpleAtod(sstream.str(), &double_val));  // Crash OK
            dim->add_value_double(double_val);
            absl::StrAppendFormat(
                &formatted_str_, "%.2f ",
                dim->value_double(dim->value_double_size() - 1));
          } else if (typeid(values[nstart]) == typeid(int64)) {
            int64 int64_val = 0;
            CHECK(absl::SimpleAtoi(sstream.str(), &int64_val));  // Crash OK
            dim->add_value_int64(int64_val);
            absl::StrAppendFormat(
                &formatted_str_, "%d ",
                dim->value_int64(dim->value_int64_size() - 1));
          } else if (typeid(values[nstart]) == typeid(string)) {
            dim->add_value_str(sstream.str());
            absl::StrAppend(&formatted_str_, "'",
                            dim->value_str(dim->value_str_size() - 1), "' ");
          } else {
            CHECK(false) << "Unsupported type: "
                         << typeid(values[nstart]).name();
          }
          ++nstart;
        } else {
          // Not-last dimension. Drill deeper.
          nstart = BuildOutput<T>(nstart, depth + 1, values, dim);
        }
      }
    }
    if (formatted_str_.length() > kTFProfTenosrMaxDisplayLen) {
      formatted_str_ = formatted_str_.substr(0, kTFProfTenosrMaxDisplayLen);
    }
    formatted_str_ += "],\n";
    return nstart;
  }

  template <typename T, typename U>
  void GetValueVec(std::vector<U>* value_vec) {
    // TODO(xpan): Address the huge tensor problem.
    if (tensor_->NumElements() > kTFProfTensorMaxWarnLen) {
      absl::FPrintF(stderr, "Showing huge tensor, the tool might halt...\n");
    }
    auto values = tensor_->flat<T>();
    for (int64 i = 0; i < tensor_->NumElements(); i++) {
      value_vec->push_back(static_cast<U>(values(i)));
    }
  }

  TFProfTensorProto tfprof_tensor_pb_;
  std::unique_ptr<Tensor> tensor_;
  string formatted_str_;
};
}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TENSOR_H_
