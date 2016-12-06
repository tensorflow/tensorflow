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

#include "tensorflow/core/kernels/hexagon/hexagon_control_wrapper.h"

namespace tensorflow {

int HexagonControlWrapper::GetVersion() const {
  // TODO: Implement
  return 1;
}

bool HexagonControlWrapper::Init() {
  // TODO: Implement
  return false;
}

bool HexagonControlWrapper::Finalize() {
  // TODO: Implement
  return false;
}
bool HexagonControlWrapper::SetupGraph(
    const GraphTransferer &graph_transferer) {
  // TODO: Implement
  return false;
}

bool HexagonControlWrapper::ExecuteGraph() {
  // TODO: Implement
  return false;
}

bool HexagonControlWrapper::TeardownGraph() {
  // TODO: Implement
  return false;
}

bool HexagonControlWrapper::FillInputNode(const string node_name,
                                          const ByteArray bytes) {
  // TODO: Implement
  return false;
}

bool HexagonControlWrapper::ReadOutputNode(
    const string node_name, std::vector<ByteArray> *const outputs) const {
  CHECK(outputs != nullptr);
  // TODO: Implement
  return false;
}

}  // namespace tensorflow
