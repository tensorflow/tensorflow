/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
// Command line tool for embedding validation data in tflite models.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/idl.h"  // from @flatbuffers
#include "flatbuffers/reflection.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "flatbuffers/util.h"  // from @flatbuffers
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/call_register.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_register.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/embedder.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace tflite {
namespace acceleration {
struct EmbedderOptions {
  std::string schema, main_model, metrics_model, output, jpegs_arg;
  float scale = 0.;
  int64_t zero_point = -1;
  bool use_ondevice_cpu_for_golden = false;
};

int RunEmbedder(const EmbedderOptions& options) {
  // Load schema.
  std::string fbs_contents;
  if (!flatbuffers::LoadFile(options.schema.c_str(), false, &fbs_contents)) {
    std::cerr << "Unable to load schema file " << options.schema << std::endl;
    return 1;
  }
  const char* include_directories[] = {nullptr};
  flatbuffers::Parser schema_parser;
  if (!schema_parser.Parse(fbs_contents.c_str(), include_directories)) {
    std::cerr << "Unable to parse schema " << schema_parser.error_ << std::endl;
    return 2;
  }
  schema_parser.Serialize();
  const reflection::Schema* schema =
      reflection::GetSchema(schema_parser.builder_.GetBufferPointer());

  // Load main model.
  std::string main_model_contents;
  if (!flatbuffers::LoadFile(options.main_model.c_str(), false,
                             &main_model_contents)) {
    std::cerr << "Unable to load main model file " << options.main_model
              << std::endl;
    return 3;
  }
  const Model* main_model =
      flatbuffers::GetRoot<Model>(main_model_contents.data());

  // Load metrics model.
  std::string metrics_model_contents;
  if (!flatbuffers::LoadFile(options.metrics_model.c_str(), false,
                             &metrics_model_contents)) {
    std::cerr << "Unable to load metrics model file " << options.metrics_model
              << std::endl;
    return 4;
  }
  const Model* metrics_model =
      flatbuffers::GetRoot<Model>(metrics_model_contents.data());

  // Load sample images.
  std::vector<std::string> jpeg_paths = absl::StrSplit(options.jpegs_arg, ',');
  std::vector<std::string> jpeg_data;
  for (const std::string& jpeg_path : jpeg_paths) {
    std::string data;
    if (!flatbuffers::LoadFile(jpeg_path.c_str(), false, &data)) {
      std::cerr << "Unable to load jpeg file '" << jpeg_path << "'"
                << std::endl;
      return 5;
    }
    jpeg_data.push_back(data);
  }

  // Create model with embedded validation.
  tflite::acceleration::Embedder embedder(
      main_model, jpeg_data, options.scale, options.zero_point, metrics_model,
      schema, options.use_ondevice_cpu_for_golden);
  flatbuffers::FlatBufferBuilder fbb;
  ::tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom("validation/call",
                     ::tflite::acceleration::ops::Register_CALL(), 1);
  resolver.AddCustom(
      "validation/decode_jpeg",
      ::tflite::acceleration::decode_jpeg_kernel::Register_DECODE_JPEG(), 1);
  auto status = embedder.CreateModelWithEmbeddedValidation(&fbb, &resolver);
  if (!status.ok()) {
    std::cerr << "Creating model with embedded validation failed: "
              << status.ToString() << std::endl;
    return 6;
  }

  // Write created model to output path.
  std::string binary(reinterpret_cast<const char*>(fbb.GetBufferPointer()),
                     fbb.GetSize());
  std::ofstream f;
  f.open(options.output);
  if (!f.good()) {
    std::cerr << "Opening " << options.output
              << " for writing failed: " << strerror(errno) << std::endl;
    return 7;
  }
  f << binary;
  f.close();
  if (!f.good()) {
    std::cerr << "Writing to " << options.output
              << " failed: " << strerror(errno) << std::endl;
    return 8;
  }

  const Model* model = flatbuffers::GetRoot<Model>(fbb.GetBufferPointer());
  std::unique_ptr<Interpreter> interpreter;
  if (InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Loading the created model failed" << std::endl;
    return 9;
  }

  return 0;
}

}  // namespace acceleration
}  // namespace tflite

int main(int argc, char* argv[]) {
  tflite::acceleration::EmbedderOptions options;
  std::vector<tflite::Flag> flags = {
      tflite::Flag::CreateFlag("schema", &options.schema,
                               "Path to tflite schema.fbs"),
      tflite::Flag::CreateFlag("main_model", &options.main_model,
                               "Path to main inference tflite model"),
      tflite::Flag::CreateFlag("metrics_model", &options.metrics_model,
                               "Path to metrics tflite model"),
      tflite::Flag::CreateFlag("output", &options.output,
                               "Path to tflite output file"),
      tflite::Flag::CreateFlag(
          "jpegs", &options.jpegs_arg,
          "Comma-separated list of jpeg files to use as input"),
      tflite::Flag::CreateFlag("scale", &options.scale,
                               "Scale to use when dequantizing input images"),
      tflite::Flag::CreateFlag(
          "zero_point", &options.zero_point,
          "Zero-point to use when dequantizing input images"),
      tflite::Flag::CreateFlag(
          "use_ondevice_cpu_for_golden", &options.use_ondevice_cpu_for_golden,
          "Use on-device CPU as golden data (rather than embedding golden "
          "data)"),
  };
  if (!tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flags) ||
      options.schema.empty() || options.main_model.empty() ||
      options.output.empty() || options.jpegs_arg.empty()) {
    std::cerr << tflite::Flags::Usage("embedder_cmdline", flags);
    return 1;
  }
  return tflite::acceleration::RunEmbedder(options);

  return 0;
}
