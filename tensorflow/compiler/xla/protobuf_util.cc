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

#include "absl/hash/hash.h"
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
  std::string serialized1, serialized2;
  m1.AppendToString(&serialized1);
  m2.AppendToString(&serialized2);
  return (serialized1 == serialized2);
}

size_t ProtobufHash(const tensorflow::protobuf::Message& m) {
  // This is a bit fast and loose, but avoids introducing a dependency on
  // the much more complex protobuf::util::MessageDifferencer class.
  // We perform the hash on their serialized representation.
  std::string serialized;
  m.AppendToString(&serialized);
  return absl::Hash<std::string>()(serialized);
}

Status DumpProtoToDirectory(const tensorflow::protobuf::Message& message,
                            const std::string& directory,
                            const std::string& file_name,
                            std::string* full_path) {
  tensorflow::Env* env = tensorflow::Env::Default();
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(directory));
  std::string safe_file_name = SanitizeFileName(file_name) + ".pb";
  std::string full_path_impl;
  if (!full_path) {
    full_path = &full_path_impl;
  }
  *full_path = tensorflow::io::JoinPath(directory, safe_file_name);
  return tensorflow::WriteBinaryProto(env, *full_path, message);
}

}  // namespace protobuf_util
}  // namespace xla
