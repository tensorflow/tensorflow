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

#ifndef TENSORFLOW_COMPILER_XLA_PROTOBUF_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_PROTOBUF_UTIL_H_

#include <functional>
#include <string>

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace protobuf_util {

// Returns true if m1 is equal to m2.
//
// WARNING: We use protocol buffer serialization and then check for
// equality of the serialized representation, which may miss some
// cases of equality.  However, for the purposes of the XLA code
// base, this form of equality checking is sufficient.
extern bool ProtobufEquals(const tensorflow::protobuf::Message& m1,
                           const tensorflow::protobuf::Message& m2);

// Return the hash of the message "m".
//
// WARNING: This uses the same serialization approach used by ProtobufEquals,
// so the WARNING for that function applies here.
size_t ProtobufHash(const tensorflow::protobuf::Message& m);

// Wrappers for above methods so that they can be used in containers.
class ProtobufEqualsWrapper {
 public:
  bool operator()(const tensorflow::protobuf::Message& m1,
                  const tensorflow::protobuf::Message& m2) const {
    return ProtobufEquals(m1, m2);
  }
};

class ProtobufHashWrapper {
 public:
  size_t operator()(const tensorflow::protobuf::Message& m) const {
    return ProtobufHash(m);
  }
};
// Writes the given message in binary proto to the path formed by joining
// 'directory/file_name.pb'. The 'directory' is recursively created if it
// doesn't already exist, and the 'file_name' is sanitized by replacing
// illegal characters with underscore '_'.
//
// If 'full_name' is not null then it is set to the name of the file the
// protobuf was written to.
Status DumpProtoToDirectory(const tensorflow::protobuf::Message& message,
                            const std::string& directory,
                            const std::string& file_name,
                            std::string* full_path = nullptr);

// Registers a function that may either expand a dirpath or forward the original
// dirpath along as-is.
void RegisterDirectoryExpander(
    const std::function<std::string(std::string)>& expander);

}  // namespace protobuf_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PROTOBUF_UTIL_H_
