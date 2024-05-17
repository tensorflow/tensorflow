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

#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow/tools/proto_splitter/cc/composable_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/size_splitter.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace tools::proto_splitter {

// Splitter that works on repeated message fields.
template <typename ParentMessage, typename RepeatedMessage>
class RepeatedFieldSplitters : public SizeSplitter {
 public:
  static absl::StatusOr<RepeatedFieldSplitters> Create(
      tsl::protobuf::Message* message, ComposableSplitter* parent_splitter,
      std::vector<FieldType>* fields_in_parent, const FieldType& repeated_field,
      std::vector<SizeSplitterFactory*>* splitter_factories);

  absl::StatusOr<int> BuildChunksReturnSize() override;
  FieldType repeated_field_;

 private:
  explicit RepeatedFieldSplitters(
      tsl::protobuf::Message* message, ComposableSplitter* parent_splitter,
      std::vector<FieldType>* fields_in_parent, const FieldType& repeated_field,
      std::vector<SizeSplitterFactory*>* splitter_factories)
      : SizeSplitter(message, parent_splitter, fields_in_parent),
        repeated_field_(repeated_field),
        splitter_factories_(splitter_factories) {}

  std::vector<SizeSplitterFactory*>* splitter_factories_;
};

}  // namespace tools::proto_splitter
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_REPEATED_FIELD_SPLITTER_H_
