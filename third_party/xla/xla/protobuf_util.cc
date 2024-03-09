/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/protobuf_util.h"

#include <string>

#include "absl/hash/hash.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace protobuf_util {

bool ProtobufEquals(const tsl::protobuf::Message& m1,
                    const tsl::protobuf::Message& m2) {
  // This is a bit fast and loose, but avoids introducing a dependency on
  // the much more complex protobuf::util::MessageDifferencer class.  For
  // our purposes we just say that two protobufs are equal if their serialized
  // representations are equal.
  std::string serialized1, serialized2;
  m1.AppendToString(&serialized1);
  m2.AppendToString(&serialized2);
  return (serialized1 == serialized2);
}

size_t ProtobufHash(const tsl::protobuf::Message& m) {
  // This is a bit fast and loose, but avoids introducing a dependency on
  // the much more complex protobuf::util::MessageDifferencer class.
  // We perform the hash on their serialized representation.
  std::string serialized;
  m.AppendToString(&serialized);
  return absl::HashOf(serialized);
}

Status DumpProtoToDirectory(const tsl::protobuf::Message& message,
                            const std::string& directory,
                            const std::string& file_name,
                            std::string* full_path) {
  tsl::Env* env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(directory));
  std::string safe_file_name = SanitizeFileName(file_name) + ".pb";
  std::string full_path_impl;
  if (!full_path) {
    full_path = &full_path_impl;
  }
  *full_path = tsl::io::JoinPath(directory, safe_file_name);
  return tsl::WriteBinaryProto(env, *full_path, message);
}

}  // namespace protobuf_util
}  // namespace xla
