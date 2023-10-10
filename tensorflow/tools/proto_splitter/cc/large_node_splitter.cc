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
#include "tensorflow/tools/proto_splitter/cc/large_node_splitter.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/tools/proto_splitter/cc/max_size.h"
#include "tensorflow/tools/proto_splitter/cc/size_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace tools::proto_splitter {

template <typename MessageType>
absl::StatusOr<int> LargeNodeSplitter<MessageType>::BuildChunksReturnSize() {
  MessageType* msg =
      tsl::protobuf::DynamicCastToGenerated<MessageType>(message());
  int initial_size = GetInitialSize();
  std::shared_ptr<MessageType> new_msg = std::make_shared<MessageType>();
  msg->Swap(new_msg.get());
  std::vector<FieldType> fields = {};
  auto x = std::make_unique<MessageBytes>(new_msg);
  TF_RETURN_IF_ERROR(AddChunk(std::move(x), &fields, index_));
  return initial_size;
}

template <typename MessageType>
absl::StatusOr<std::unique_ptr<SizeSplitter>>
LargeNodeSplitterFactory<MessageType>::CreateSplitter(
    tsl::protobuf::Message* message, ComposableSplitterBase* parent_splitter,
    std::vector<FieldType>* fields_in_parent, int size) {
  if (!(LARGE_SIZE_CHECK(size, GetMaxSize()))) return nullptr;
  LargeNodeSplitter<MessageType>* splitter = new LargeNodeSplitter<MessageType>(
      message, parent_splitter, fields_in_parent);
  return absl::WrapUnique(splitter);
}

template class LargeNodeSplitter<GraphDef>;
template class LargeNodeSplitter<FunctionDefLibrary>;
template class LargeNodeSplitter<FunctionDef>;
template class LargeNodeSplitter<NodeDef>;
template class LargeNodeSplitterFactory<GraphDef>;
template class LargeNodeSplitterFactory<FunctionDefLibrary>;
template class LargeNodeSplitterFactory<FunctionDef>;
template class LargeNodeSplitterFactory<NodeDef>;

}  // namespace tools::proto_splitter
}  // namespace tensorflow
