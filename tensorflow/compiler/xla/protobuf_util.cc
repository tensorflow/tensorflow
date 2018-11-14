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

#include "tensorflow/compiler/xla/protobuf_util.h"

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace protobuf_util {

bool ProtobufEquals(const tensorflow::protobuf::Message& m1,
                    const tensorflow::protobuf::Message& m2) {
  // This is a bit fast and loose, but avoids introducing a dependency on
  // the much more complex protobuf::util::MessageDifferencer class.  For
  // our purposes we just say that two protobufs are equal if their serialized
  // representations are equal.
  string serialized1, serialized2;
  m1.AppendToString(&serialized1);
  m2.AppendToString(&serialized2);
  return (serialized1 == serialized2);
}

namespace {

string SanitizeFilename(const string& file_name) {
  string safe_file_name = file_name;
  for (char& c : safe_file_name) {
    if (c == '/' || c == '\\') {
      c = '_';
    }
  }
  return safe_file_name;
}

std::pair<tensorflow::mutex*, std::vector<std::function<string(string)>>*>
GetDirectoryExpanders() {
  static auto* mutex = new tensorflow::mutex;
  static auto* singleton = new std::vector<std::function<string(string)>>;
  return {mutex, singleton};
}

// Runs all the directory expanders over x and returns the result.
string Expand(string x) {
  auto pair = GetDirectoryExpanders();
  tensorflow::mutex_lock lock(*pair.first);
  for (const auto& f : *pair.second) {
    x = f(x);
  }
  return x;
}

}  // namespace

Status DumpProtoToDirectory(const tensorflow::protobuf::Message& message,
                            const string& directory, const string& file_name) {
  tensorflow::Env* env = tensorflow::Env::Default();
  string expanded_dir = Expand(directory);
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(expanded_dir));
  string safe_file_name = SanitizeFileName(file_name) + ".pb";
  const string path = tensorflow::io::JoinPath(expanded_dir, safe_file_name);
  return tensorflow::WriteBinaryProto(env, path, message);
}

void RegisterDirectoryExpander(const std::function<string(string)>& expander) {
  auto pair = GetDirectoryExpanders();
  tensorflow::mutex_lock lock(*pair.first);
  pair.second->push_back(expander);
}

}  // namespace protobuf_util
}  // namespace xla
