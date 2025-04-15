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
#ifndef TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_SIZE_SPLITTER_H_
#define TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_SIZE_SPLITTER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow/tools/proto_splitter/cc/composable_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"

namespace tensorflow {
namespace tools::proto_splitter {

// Abstract composable Splitter that returns the size of chunks created.
// Modifies the user-provided message when building chunks.
class SizeSplitter : public ComposableSplitter {
 public:
  // Initializer.
  using ComposableSplitter::ComposableSplitter;

  absl::Status BuildChunks() override {
    return BuildChunksReturnSize().status();
  }

  // Chunks the message and returns the int size diff.
  virtual absl::StatusOr<int> BuildChunksReturnSize() = 0;
};

class SizeSplitterFactory {
 public:
  explicit SizeSplitterFactory() = default;

  // Creates a new SizeSplitter object based on the input parameters. May
  // return nullptr if no split is necessary.
  virtual absl::StatusOr<std::unique_ptr<SizeSplitter>> CreateSplitter(
      tsl::protobuf::Message* message, ComposableSplitterBase* parent_splitter,
      std::vector<FieldType>* fields_in_parent, int size) = 0;

  virtual ~SizeSplitterFactory() = default;
};

}  // namespace tools::proto_splitter
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_SIZE_SPLITTER_H_
