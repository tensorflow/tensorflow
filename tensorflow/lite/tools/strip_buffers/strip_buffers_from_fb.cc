/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// Binary to test strip_buffers/reconstitution.h.
#include <fstream>  // NOLINT
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/strip_buffers/stripping_lib.h"

#define TFLITE_SCHEMA_VERSION 3

namespace tflite {

using ::flatbuffers::FlatBufferBuilder;

constexpr char kInputFlatbufferFlag[] = "input_flatbuffer";
constexpr char kOutputFlatbufferFlag[] = "output_flatbuffer";

int Main(int argc, char* argv[]) {
  // Command Line Flags.
  std::string input_flatbuffer_path;
  std::string output_flatbuffer_path;

  std::vector<Flag> flag_list = {
      tflite::Flag::CreateFlag(kInputFlatbufferFlag, &input_flatbuffer_path,
                               "Path to input TFLite flatbuffer."),
      tflite::Flag::CreateFlag(kOutputFlatbufferFlag, &output_flatbuffer_path,
                               "Path to output TFLite flatbuffer."),
  };
  Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);

  // Read in input flatbuffer.
  auto input_model =
      FlatBufferModel::BuildFromFile(input_flatbuffer_path.c_str());

  // Strip applicable constants from the flatbuffer.
  FlatBufferBuilder builder(/*initial_size=*/10240);
  if (StripWeightsFromFlatbuffer(input_model->GetModel(), &builder) !=
      kTfLiteOk) {
    return 0;
  }

  LOG(INFO) << "Flatbuffer size (KB) BEFORE: "
            << input_model->allocation()->bytes() / 1000.0;
  LOG(INFO) << "Flatbuffer size (KB) AFTER: " << builder.GetSize() / 1000.0;

  // Write the output model to file.
  std::string output_model_content(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize());
  std::ofstream output_file_stream(output_flatbuffer_path);
  output_file_stream << output_model_content;
  output_file_stream.close();

  return 0;
}
}  // namespace tflite

int main(int argc, char* argv[]) { return tflite::Main(argc, argv); }
