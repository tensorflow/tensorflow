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
#ifndef TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_LARGE_NODE_SPLITTER_H_
#define TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_LARGE_NODE_SPLITTER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow/tools/proto_splitter/cc/composable_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/size_splitter.h"

namespace tensorflow {
namespace tools::proto_splitter {

// Extracts messages that are large but not over the limit into a separate
// message chunk.
template <typename MessageType>
class LargeNodeSplitter : public SizeSplitter {
 public:
  using SizeSplitter::SizeSplitter;

  void SetChunkIndex(int* index) { index_ = index; }

  absl::StatusOr<int> BuildChunksReturnSize() override;

 private:
  int* index_ = nullptr;
};

template <typename MessageType>
class LargeNodeSplitterFactory : public SizeSplitterFactory {
 public:
  using SizeSplitterFactory::SizeSplitterFactory;

  absl::StatusOr<std::unique_ptr<SizeSplitter>> CreateSplitter(
      tsl::protobuf::Message* message, ComposableSplitterBase* parent_splitter,
      std::vector<FieldType>* fields_in_parent, int size) override;
};

}  // namespace tools::proto_splitter
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_LARGE_NODE_SPLITTER_H_
