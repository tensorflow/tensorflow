/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
vcyou may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_GRAPH_LOADER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_GRAPH_LOADER_H_

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// GraphTransferer transfers graph definitions into SoC memory.
// This functionality is effective if SoC is capable to run
// the graph on that chip.
// TODO(satok): support transferring subgraphs to be able to split graphs
// to avoid unsupported ops in SoC.
class GraphTransferer {
 public:
  GraphTransferer() = default;
  void LoadGraphFromProto(::tensorflow::protobuf::MessageLite* proto);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GraphTransferer);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_GRAPH_LOADER_H_
