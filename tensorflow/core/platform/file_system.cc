/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

FileSystem::~FileSystem() {}

string FileSystem::TranslateName(const string& name) const { return name; }

RandomAccessFile::~RandomAccessFile() {}

WritableFile::~WritableFile() {}

FileSystemRegistry::~FileSystemRegistry() {}

string GetSchemeFromURI(const string& name) {
  auto colon_loc = name.find(":");
  // Make sure scheme matches [a-zA-Z][0-9a-zA-Z.]*
  // TODO(keveman): Allow "+" and "-" in the scheme.
  if (colon_loc != string::npos &&
      strings::Scanner(StringPiece(name.data(), colon_loc))
          .One(strings::Scanner::LETTER)
          .Many(strings::Scanner::LETTER_DIGIT_DOT)
          .GetResult()) {
    return name.substr(0, colon_loc);
  }
  return "";
}

string GetNameFromURI(const string& name) {
  string scheme = GetSchemeFromURI(name);
  if (scheme == "") {
    return name;
  }
  // Skip the 'scheme:' portion.
  StringPiece filename{name.data() + scheme.length() + 1,
                       name.length() - scheme.length() - 1};
  // If the URI confirmed to scheme://filename, skip the two '/'s and return
  // filename. Otherwise return the original 'name', and leave it up to the
  // implementations to handle the full URI.
  if (filename[0] == '/' && filename[1] == '/') {
    return filename.substr(2).ToString();
  }
  return name;
}

}  // namespace tensorflow
