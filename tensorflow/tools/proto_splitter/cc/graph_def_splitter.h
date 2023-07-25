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
#ifndef TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_GRAPH_DEF_SPLITTER_H_
#define TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_GRAPH_DEF_SPLITTER_H_

#include "tensorflow/tools/proto_splitter/cc/composable_splitter.h"

namespace tensorflow {
namespace tools::proto_splitter {

// GraphDef Splitter.
// Modifies the user-provided message when building chunks.
class GraphDefSplitter : public ComposableSplitter {
 public:
  // Initializer.
  using ComposableSplitter::ComposableSplitter;

  absl::Status BuildChunks() override;
};
}  // namespace tools::proto_splitter
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_GRAPH_DEF_SPLITTER_H_
