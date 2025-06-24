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
#ifndef TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_REPEATED_FIELD_SPLITTER_H_
#define TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_REPEATED_FIELD_SPLITTER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/tools/proto_splitter/cc/composable_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/max_size.h"
#include "tensorflow/tools/proto_splitter/cc/size_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow::tools::proto_splitter {

// Splitter that works on repeated message fields.
template <typename ParentMessage, typename RepeatedMessage>
class RepeatedFieldSplitter : public SizeSplitter {
 public:
  static absl::StatusOr<RepeatedFieldSplitter> Create(
      tsl::protobuf::Message* message, ComposableSplitter* parent_splitter,
      std::vector<FieldType>* fields_in_parent, const FieldType& repeated_field,
      std::vector<SizeSplitterFactory*>* splitter_factories);

  absl::StatusOr<int> BuildChunksReturnSize() override;

 private:
  explicit RepeatedFieldSplitter(
      tsl::protobuf::Message* message, ComposableSplitter* parent_splitter,
      std::vector<FieldType>* fields_in_parent, const FieldType& repeated_field,
      std::vector<SizeSplitterFactory*>* splitter_factories)
      : SizeSplitter(message, parent_splitter, fields_in_parent),
        repeated_field_(repeated_field),
        splitter_factories_(splitter_factories) {}

  FieldType repeated_field_;
  std::vector<SizeSplitterFactory*>* splitter_factories_;
};

// Additional bytes added to each node to account for the extra info needed to
// encode the field key (realistically 3 but making it 5 for some wiggle room).
constexpr int kExtraBytes = 5;

template <typename ParentMessage, typename RepeatedMessage>
absl::StatusOr<RepeatedFieldSplitter<ParentMessage, RepeatedMessage>>
RepeatedFieldSplitter<ParentMessage, RepeatedMessage>::Create(
    tsl::protobuf::Message* message, ComposableSplitter* parent_splitter,
    std::vector<FieldType>* fields_in_parent, const FieldType& repeated_field,
    std::vector<SizeSplitterFactory*>* splitter_factories) {
  TF_ASSIGN_OR_RETURN(auto field_ret, GetField(*message, {repeated_field}));
  if (!field_ret.field->is_repeated()) {
    return absl::FailedPreconditionError("Unable to split non-repeated field.");
  }

  auto ret = RepeatedFieldSplitter<ParentMessage, RepeatedMessage>(
      message, parent_splitter, fields_in_parent, repeated_field,
      splitter_factories);
  return ret;
}

template <typename ParentMessage, typename RepeatedMessage>
absl::StatusOr<int>
RepeatedFieldSplitter<ParentMessage, RepeatedMessage>::BuildChunksReturnSize() {
  TF_ASSIGN_OR_RETURN(MutableFieldResult mfr,
                      GetMutableField(message(), {repeated_field_}));
  tsl::protobuf::Message* parent = mfr.parent;
  const tsl::protobuf::FieldDescriptor* repeated_field = mfr.field;

  uint64_t max_size = GetMaxSize();
  size_t initial_size = GetInitialSize();

  // List of indices at which to split the repeated field. For example, [3, 5]
  // means that the field list is split into: [:3], [3:5], [5:]
  std::vector<int> repeated_msg_split;
  // Track the total byte size of the current node split.
  uint64_t total_size = 0;

  // Linearly iterate through all nodes. It may be possible to optimize this
  // further by making best guesses as to where to split the nodes, since
  // most nodes (aside from constants) are relatively small.
  int repeated_field_length =
      parent->GetReflection()->FieldSize(*parent, repeated_field);
  for (int i = 0; i < repeated_field_length; ++i) {
    tsl::protobuf::Message* node =
        parent->GetReflection()->MutableRepeatedMessage(parent, repeated_field,
                                                        i);
    auto node_size = node->ByteSizeLong();

    std::vector<FieldType> new_fields = {repeated_field_, i};

    for (auto factory : *splitter_factories_) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<SizeSplitter> new_splitter,
          factory->CreateSplitter(node, this, &new_fields, node_size));
      if (new_splitter != nullptr) {
        TF_ASSIGN_OR_RETURN(auto size_diff,
                            new_splitter->BuildChunksReturnSize());
        node_size -= size_diff;
      }
    }
    if (total_size + node_size > max_size) {
      repeated_msg_split.push_back(i);
      total_size = 0;
    }
    total_size += node_size + kExtraBytes;
  }

  if (!repeated_msg_split.empty()) {
    auto repeated_nodes_ptrs =
        parent->GetReflection()
            ->template MutableRepeatedPtrField<RepeatedMessage>(parent,
                                                                repeated_field);

    std::vector<RepeatedMessage*> extracted_nodes(repeated_field_length);
    repeated_nodes_ptrs->ExtractSubrange(0, repeated_field_length,
                                         &extracted_nodes.at(0));
    // Last range end is the size of the repeated field.
    repeated_msg_split.push_back(repeated_field_length);

    int range_start = 0;
    for (int range_end : repeated_msg_split) {
      auto new_msg = std::make_shared<ParentMessage>();
      std::vector<FieldType> empty_fields;
      auto x = std::make_unique<MessageBytes>(new_msg);
      TF_RETURN_IF_ERROR(AddChunk(std::move(x), &empty_fields));

      // Move nodes into new_msg.
      TF_ASSIGN_OR_RETURN(auto new_ret,
                          GetMutableField(new_msg.get(), repeated_field_));

      for (int j = range_start; j < range_end; ++j) {
        new_msg->GetReflection()->AddAllocatedMessage(
            new_msg.get(), new_ret.field, extracted_nodes[j]);
      }

      range_start = range_end;
    }
  }

  // Estimate the size diff by subtracting the first computed chunk size from
  // the initial size of the repeated field.
  return initial_size - message()->ByteSizeLong();
}

}  // namespace tensorflow::tools::proto_splitter

#endif  // TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_REPEATED_FIELD_SPLITTER_H_
