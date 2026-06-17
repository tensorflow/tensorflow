/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
// Simple program to convert from JSON to binary flatbuffers for given schema.
//
// Used for creating the binary version of a compatibility list.
//
// The flatc command line is not available in all build environments.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/idl.h"  // from @flatbuffers
#include "flatbuffers/reflection.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "flatbuffers/util.h"  // from @flatbuffers
#include "tensorflow/lite/tools/command_line_flags.h"

int main(int argc, char** argv) {
  std::string json_path, fbs_path, fb_path;
  std::vector<tflite::Flag> flags = {
      tflite::Flag::CreateFlag("json_input", &json_path,
                               "Path to input json file."),
      tflite::Flag::CreateFlag("fbs", &fbs_path,
                               "Path to flatbuffer schema to use."),
      tflite::Flag::CreateFlag("fb_output", &fb_path,
                               "Path to a output binary flatbuffer."),
  };
  const bool parse_result =
      tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flags);
  if (!parse_result || json_path.empty() || fbs_path.empty() ||
      fb_path.empty()) {
    std::cerr << tflite::Flags::Usage(argv[0], flags);
    return 1;
  }
  std::string json_contents;
  if (!flatbuffers::LoadFile(json_path.c_str(), false, &json_contents)) {
    std::cerr << "Unable to load file " << json_path << std::endl;
    return 2;
  }
  std::string fbs_contents;
  if (!flatbuffers::LoadFile(fbs_path.c_str(), false, &fbs_contents)) {
    std::cerr << "Unable to load file " << fbs_path << std::endl;
    return 3;
  }
  const char* include_directories[] = {nullptr};
  flatbuffers::Parser schema_parser;
  if (!schema_parser.Parse(fbs_contents.c_str(), include_directories)) {
    std::cerr << "Unable to parse schema " << schema_parser.error_ << std::endl;
    return 4;
  }
  schema_parser.Serialize();
  auto schema =
      reflection::GetSchema(schema_parser.builder_.GetBufferPointer());
  auto root_table = schema->root_table();
  flatbuffers::Parser parser;
  parser.Deserialize(schema_parser.builder_.GetBufferPointer(),
                     schema_parser.builder_.GetSize());

  if (!parser.Parse(json_contents.c_str(), include_directories,
                    json_path.c_str())) {
    std::cerr << "Unable to parse json " << parser.error_ << std::endl;
    return 5;
  }

  // Use CopyTable() to deduplicate the strings.
  const uint8_t* buffer = parser.builder_.GetBufferPointer();
  flatbuffers::FlatBufferBuilder fbb;
  auto root_offset = flatbuffers::CopyTable(
      fbb, *schema, *root_table, *flatbuffers::GetAnyRoot(buffer), true);
  fbb.Finish(root_offset);
  std::string binary(reinterpret_cast<const char*>(fbb.GetBufferPointer()),
                     fbb.GetSize());
  std::ofstream output;
  output.open(fb_path);
  output << binary;
  output.close();
  return 0;
}
