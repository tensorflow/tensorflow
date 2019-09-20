/*
 * Copyright 2017 Google Inc. All rights reserved.
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

#ifndef FLATBUFFERS_REGISTRY_H_
#define FLATBUFFERS_REGISTRY_H_

#include "flatbuffers/idl.h"

namespace flatbuffers {

// Convenience class to easily parse or generate text for arbitrary FlatBuffers.
// Simply pre-populate it with all schema filenames that may be in use, and
// This class will look them up using the file_identifier declared in the
// schema.
class Registry {
 public:
  // Call this for all schemas that may be in use. The identifier has
  // a function in the generated code, e.g. MonsterIdentifier().
  void Register(const char *file_identifier, const char *schema_path) {
    Schema schema;
    schema.path_ = schema_path;
    schemas_[file_identifier] = schema;
  }

  // Generate text from an arbitrary FlatBuffer by looking up its
  // file_identifier in the registry.
  bool FlatBufferToText(const uint8_t *flatbuf, size_t len, std::string *dest) {
    // Get the identifier out of the buffer.
    // If the buffer is truncated, exit.
    if (len < sizeof(uoffset_t) + FlatBufferBuilder::kFileIdentifierLength) {
      lasterror_ = "buffer truncated";
      return false;
    }
    std::string ident(
        reinterpret_cast<const char *>(flatbuf) + sizeof(uoffset_t),
        FlatBufferBuilder::kFileIdentifierLength);
    // Load and parse the schema.
    Parser parser;
    if (!LoadSchema(ident, &parser)) return false;
    // Now we're ready to generate text.
    if (!GenerateText(parser, flatbuf, dest)) {
      lasterror_ = "unable to generate text for FlatBuffer binary";
      return false;
    }
    return true;
  }

  // Converts a binary buffer to text using one of the schemas in the registry,
  // use the file_identifier to indicate which.
  // If DetachedBuffer::data() is null then parsing failed.
  DetachedBuffer TextToFlatBuffer(const char *text,
                                  const char *file_identifier) {
    // Load and parse the schema.
    Parser parser;
    if (!LoadSchema(file_identifier, &parser)) return DetachedBuffer();
    // Parse the text.
    if (!parser.Parse(text)) {
      lasterror_ = parser.error_;
      return DetachedBuffer();
    }
    // We have a valid FlatBuffer. Detach it from the builder and return.
    return parser.builder_.Release();
  }

  // Modify any parsing / output options used by the other functions.
  void SetOptions(const IDLOptions &opts) { opts_ = opts; }

  // If schemas used contain include statements, call this function for every
  // directory the parser should search them for.
  void AddIncludeDirectory(const char *path) { include_paths_.push_back(path); }

  // Returns a human readable error if any of the above functions fail.
  const std::string &GetLastError() { return lasterror_; }

 private:
  bool LoadSchema(const std::string &ident, Parser *parser) {
    // Find the schema, if not, exit.
    auto it = schemas_.find(ident);
    if (it == schemas_.end()) {
      // Don't attach the identifier, since it may not be human readable.
      lasterror_ = "identifier for this buffer not in the registry";
      return false;
    }
    auto &schema = it->second;
    // Load the schema from disk. If not, exit.
    std::string schematext;
    if (!LoadFile(schema.path_.c_str(), false, &schematext)) {
      lasterror_ = "could not load schema: " + schema.path_;
      return false;
    }
    // Parse schema.
    parser->opts = opts_;
    if (!parser->Parse(schematext.c_str(), vector_data(include_paths_),
                       schema.path_.c_str())) {
      lasterror_ = parser->error_;
      return false;
    }
    return true;
  }

  struct Schema {
    std::string path_;
    // TODO(wvo) optionally cache schema file or parsed schema here.
  };

  std::string lasterror_;
  IDLOptions opts_;
  std::vector<const char *> include_paths_;
  std::map<std::string, Schema> schemas_;
};

}  // namespace flatbuffers

#endif  // FLATBUFFERS_REGISTRY_H_
