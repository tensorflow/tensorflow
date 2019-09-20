/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include "monster_test_generated.h"
#include "monster_generated.h" // Already includes "flatbuffers/flatbuffers.h".

using namespace MyGame::Sample;

// This is an example of parsing text straight into a buffer and then
// generating flatbuffer (JSON) text from the buffer.
int main(int /*argc*/, const char * /*argv*/[]) {
  // load FlatBuffer schema (.fbs) and JSON from disk
  std::string schema_file;
  std::string json_file;
  std::string bfbs_file;
  bool ok =
      flatbuffers::LoadFile("tests/monster_test.fbs", false, &schema_file) &&
      flatbuffers::LoadFile("tests/monsterdata_test.golden", false, &json_file) &&
      flatbuffers::LoadFile("tests/monster_test.bfbs", true, &bfbs_file);
  if (!ok) {
    printf("couldn't load files!\n");
    return 1;
  }

  const char *include_directories[] = { "samples", "tests",
                                        "tests/include_test", nullptr };
  // parse fbs schema
  flatbuffers::Parser parser1;
  ok = parser1.Parse(schema_file.c_str(), include_directories);
  assert(ok);

  // inizialize parser by deserializing bfbs schema
  flatbuffers::Parser parser2;
  ok = parser2.Deserialize((uint8_t *)bfbs_file.c_str(), bfbs_file.length());
  assert(ok);

  // parse json in parser from fbs and bfbs
  ok = parser1.Parse(json_file.c_str(), include_directories);
  assert(ok);
  ok = parser2.Parse(json_file.c_str(), include_directories);
  assert(ok);

  // to ensure it is correct, we now generate text back from the binary,
  // and compare the two:
  std::string jsongen1;
  if (!GenerateText(parser1, parser1.builder_.GetBufferPointer(), &jsongen1)) {
    printf("Couldn't serialize parsed data to JSON!\n");
    return 1;
  }

  std::string jsongen2;
  if (!GenerateText(parser2, parser2.builder_.GetBufferPointer(), &jsongen2)) {
    printf("Couldn't serialize parsed data to JSON!\n");
    return 1;
  }

  if (jsongen1 != jsongen2) {
    printf("%s----------------\n%s", jsongen1.c_str(), jsongen2.c_str());
  }

  printf("The FlatBuffer has been parsed from JSON successfully.\n");
}
