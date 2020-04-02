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

#ifndef TENSORFLOW_LITE_TOOLS_ACCURACY_CSV_WRITER_H_
#define TENSORFLOW_LITE_TOOLS_ACCURACY_CSV_WRITER_H_

#include <fstream>
#include <memory>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"

namespace tensorflow {
namespace metrics {
// A simple CSV writer that writes values of same type for fixed number of
// columns. This supports a very limited set of CSV spec and doesn't do any
// escaping.
// Usage:
// std::unqiue_str<std::ofstream> output_stream = ...
// CSVWriter writer({"column1", "column2"}, std::move(output_stream));
// writer.WriteRow({4, 5});
// writer.Flush(); // flush results immediately.
class CSVWriter {
 public:
  CSVWriter(const std::vector<string>& columns,
            std::unique_ptr<std::ofstream> output_stream)
      : num_columns_(columns.size()), output_stream_(std::move(output_stream)) {
    if (WriteRow(columns, output_stream_.get()) != kTfLiteOk) {
      LOG(ERROR) << "Could not write column names to file";
    }
  }

  template <typename T>
  TfLiteStatus WriteRow(const std::vector<T>& values) {
    if (values.size() != num_columns_) {
      LOG(ERROR) << "Invalid size for row:" << values.size()
                 << " expected: " << num_columns_;
      return kTfLiteError;
    }
    return WriteRow(values, output_stream_.get());
  }

  void Flush() { output_stream_->flush(); }

  ~CSVWriter() { output_stream_->flush(); }

 private:
  template <typename T>
  static TfLiteStatus WriteRow(const std::vector<T>& values,
                               std::ofstream* output_stream) {
    bool first = true;
    for (const auto& v : values) {
      if (!first) {
        (*output_stream) << ", ";
      } else {
        first = false;
      }
      (*output_stream) << v;
    }
    (*output_stream) << "\n";
    if (!output_stream->good()) {
      LOG(ERROR) << "Writing to stream failed.";
      return kTfLiteError;
    }
    return kTfLiteOk;
  }
  const size_t num_columns_;
  std::unique_ptr<std::ofstream> output_stream_;
};
}  // namespace metrics
}  // namespace tensorflow
#endif  // TENSORFLOW_LITE_TOOLS_ACCURACY_CSV_WRITER_H_
