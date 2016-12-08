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

#ifdef USE_HEXAGON_LIBS
#include "tensorflow/core/platform/hexagon/soc_interface.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/types.h"
#endif

namespace tensorflow {

#ifdef USE_HEXAGON_LIBS
int HexagonControlWrapper::GetVersion() {
  return soc_interface_GetSocControllerVersion();
}

bool HexagonControlWrapper::Init() { return soc_interface_Init(); }

bool HexagonControlWrapper::Finalize() { return soc_interface_Finalize(); }
bool HexagonControlWrapper::SetupGraph(
    const GraphTransferer &graph_transferer) {
  return soc_interface_SetupGraphDummy(3 /* inception version */);
}

bool HexagonControlWrapper::ExecuteGraph() {
  return soc_interface_ExecuteGraph();
}

bool HexagonControlWrapper::TeardownGraph() {
  return soc_interface_TeardownGraph();
}

bool HexagonControlWrapper::FillInputNode(const string node_name,
                                          const ByteArray bytes) {
  // TODO(satok): Use arguments instead of dummy input
  const int x = 1;
  const int y = 299;
  const int z = 299;
  const int d = 3;
  const int array_length = x * y * z * d;
  const int byte_size = array_length * sizeof(float);
  dummy_input_float_.resize(array_length);
  return soc_interface_FillInputNodeFloat(
      1, 299, 299, 3, reinterpret_cast<uint8 *>(dummy_input_float_.data()),
      byte_size);
}

bool HexagonControlWrapper::ReadOutputNode(
    const string node_name, std::vector<ByteArray> *const outputs) {
  CHECK(outputs != nullptr);
  ByteArray output;
  soc_interface_ReadOutputNodeFloat(node_name.c_str(), &std::get<0>(output),
                                    &std::get<1>(output));
  std::get<2>(output) = DT_FLOAT;
  outputs->emplace_back(output);
  return true;
}

#else
int HexagonControlWrapper::GetVersion() { return -1; }
bool HexagonControlWrapper::Init() { return false; }
bool HexagonControlWrapper::Finalize() { return false; }
bool HexagonControlWrapper::SetupGraph(const GraphTransferer &) {
  return false;
}
bool HexagonControlWrapper::ExecuteGraph() { return false; }
bool HexagonControlWrapper::TeardownGraph() { return false; }
bool HexagonControlWrapper::FillInputNode(const string, const ByteArray) {
  return false;
}
bool HexagonControlWrapper::ReadOutputNode(const string,
                                           std::vector<ByteArray> *const) {
  return false;
}
#endif

}  // namespace tensorflow
