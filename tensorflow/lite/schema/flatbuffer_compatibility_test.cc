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

#include <fstream>
#include <gtest/gtest.h>
#include "flatbuffers/flatc.h"  // from @flatbuffers
#include "tensorflow/core/platform/platform.h"

#ifdef PLATFORM_GOOGLE
#define TFLITE_TF_PREFIX "third_party/tensorflow/"
#else
#define TFLITE_TF_PREFIX "tensorflow/"
#endif
/// Load filename `name`
bool LoadFileRaw(const char *name, std::string *buf) {
  std::ifstream fp(name, std::ios::binary);
  if (!fp) {
    fprintf(stderr, "Failed to read '%s'\n", name);
    return false;
  }
  std::string s((std::istreambuf_iterator<char>(fp)),
                std::istreambuf_iterator<char>());
  if (s.empty()) {
    fprintf(stderr, "Read '%s' resulted in empty\n", name);
    return false;
  }
  *buf = s;
  return true;
}

bool ParseFile(flatbuffers::Parser *parser, const std::string &filename,
               const std::string &contents) {
  std::vector<const char *> include_directories;
  auto local_include_directory = flatbuffers::StripFileName(filename);
  include_directories.push_back(local_include_directory.c_str());
  include_directories.push_back(nullptr);
  if (!parser->Parse(contents.c_str(), include_directories.data(),
                     filename.c_str())) {
    fprintf(stderr, "Failed to parse flatbuffer schema '%s'\n",
            contents.c_str());
    return false;
  }
  return true;
}

// Checks to make sure current schema in current code does not cause an
// incompatibility.
TEST(SchemaTest, TestCompatibility) {
  // Read file contents of schemas into strings
  // TODO(aselle): Need a reliable way to load files.
  std::string base_contents, current_contents;
  const char *base_filename = TFLITE_TF_PREFIX "lite/schema/schema_v3b.fbs";
  const char *current_filename =
      TFLITE_TF_PREFIX "lite/schema/schema.fbs";

  ASSERT_TRUE(LoadFileRaw(base_filename, &base_contents));
  ASSERT_TRUE(LoadFileRaw(current_filename, &current_contents));
  // Parse the schemas
  flatbuffers::Parser base_parser, current_parser;
  std::vector<const char *> include_directories;
  ASSERT_TRUE(ParseFile(&base_parser, base_filename, base_contents));
  ASSERT_TRUE(ParseFile(&current_parser, current_filename, current_contents));
  // Check that the schemas conform and fail if they don't
  auto err = current_parser.ConformTo(base_parser);
  if (!err.empty()) {
    fprintf(stderr,
            "Schemas don't conform:\n%s\n"
            "In other words some change you made means that new parsers can't"
            "parse old files.\n",
            err.c_str());
    FAIL();
  }
}
