/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_SPLIT_H_
#define TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_SPLIT_H_

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tensorflow/tools/proto_splitter/versions.pb.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace tools::proto_splitter {

using ::tensorflow::proto_splitter::ChunkedMessage;
using ::tensorflow::proto_splitter::VersionDef;

// Interface for proto message splitters.
class Splitter {
 public:
  virtual ~Splitter() = default;

  // Split message into chunks.
  virtual absl::StatusOr<ChunkedProto> Split() = 0;

  // Write message to disk.
  virtual absl::Status Write(std::string file_prefix) = 0;

  // Version info about the Splitter and required Merger versions.
  virtual VersionDef Version() = 0;
};

}  // namespace tools::proto_splitter
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_SPLIT_H_
