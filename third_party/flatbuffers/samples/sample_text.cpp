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

#include "monster_generated.h"  // Already includes "flatbuffers/flatbuffers.h".

using namespace MyGame::Sample;

// This is an example of parsing text straight into a buffer and then
// generating flatbuffer (JSON) text from the buffer.
int main(int /*argc*/, const char * /*argv*/ []) {
  // load FlatBuffer schema (.fbs) and JSON from disk
  std::string schemafile;
  std::string jsonfile;
  bool ok = flatbuffers::LoadFile("samples/monster.fbs", false, &schemafile) &&
            flatbuffers::LoadFile("samples/monsterdata.json", false, &jsonfile);
  if (!ok) {
    printf("couldn't load files!\n");
    return 1;
  }

  // parse schema first, so we can use it to parse the data after
  flatbuffers::Parser parser;
  const char *include_directories[] = { "samples", nullptr };
  ok = parser.Parse(schemafile.c_str(), include_directories) &&
       parser.Parse(jsonfile.c_str(), include_directories);
  assert(ok);

  // here, parser.builder_ contains a binary buffer that is the parsed data.

  // to ensure it is correct, we now generate text back from the binary,
  // and compare the two:
  std::string jsongen;
  if (!GenerateText(parser, parser.builder_.GetBufferPointer(), &jsongen)) {
    printf("Couldn't serialize parsed data to JSON!\n");
    return 1;
  }

  if (jsongen != jsonfile) {
    printf("%s----------------\n%s", jsongen.c_str(), jsonfile.c_str());
  }

  printf("The FlatBuffer has been parsed from JSON successfully.\n");
}
