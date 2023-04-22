/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/spectrogram_test_utils.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace wav {

// This takes a CSV file representing an array of complex numbers, and saves out
// a version using a binary format to save space in the repository.
Status ConvertCsvToRaw(const string& input_filename) {
  std::vector<std::vector<std::complex<double>>> input_data;
  ReadCSVFileToComplexVectorOrDie(input_filename, &input_data);
  const string output_filename = input_filename + ".bin";
  if (!WriteComplexVectorToRawFloatFile(output_filename, input_data)) {
    return errors::InvalidArgument("Failed to write raw float file ",
                                   input_filename);
  }
  LOG(INFO) << "Wrote raw file to " << output_filename;
  return Status::OK();
}

}  // namespace wav
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc < 2) {
    LOG(ERROR) << "You must supply a CSV file as the first argument";
    return 1;
  }
  tensorflow::string filename(argv[1]);
  tensorflow::Status status = tensorflow::wav::ConvertCsvToRaw(filename);
  if (!status.ok()) {
    LOG(ERROR) << "Error processing '" << filename << "':" << status;
    return 1;
  }
  return 0;
}
