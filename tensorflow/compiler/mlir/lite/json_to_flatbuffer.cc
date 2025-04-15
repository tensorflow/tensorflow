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
#include <stdint.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/idl.h"  // from @flatbuffers
#include "flatbuffers/util.h"  // from @flatbuffers

int main(int argc, char** argv) {
  // load FlatBuffer schema (.fbs) and JSON from disk
  if (argc < 2) {
    std::cerr << "Missing input argument. Usage:\n"
              << argv[0] << " <filename or - for stdin>\n\n";
    return 1;
  }
  const char* schema_path = argv[1];
  const char* json_path = argv[2];
  std::string schema;
  std::string json;

  const bool status =
      flatbuffers::LoadFile(schema_path, /*binary=*/false, &schema) &&
      flatbuffers::LoadFile(json_path, /*binary=*/false, &json);
  if (!status) {
    std::cerr << "couldn't load files!\n";
    return 1;
  }

  // parse schema first, so we can use it to parse the data after
  flatbuffers::Parser parser;
  const bool schema_parse_result =
      parser.Parse(schema.c_str()) && parser.Parse(json.c_str());
  if (!schema_parse_result) {
    std::cerr << "Parse error.\n";
    return 1;
  }
  const size_t length = parser.builder_.GetSize();
  const size_t n =
      std::fwrite(parser.builder_.GetBufferPointer(), 1, length, stdout);
  if (n != length) {
    std::cerr << "print to stdout filed.\n";
    return 1;
  }
  return 0;
}
