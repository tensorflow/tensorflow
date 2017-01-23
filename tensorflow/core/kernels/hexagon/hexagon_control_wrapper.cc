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
#endif

namespace tensorflow {

const bool SHOW_DBG_IN_SOC = false;
const bool DBG_USE_DUMMY_INPUT = false;
const bool DBG_USE_SAMPLE_INPUT = false;
const int64 FLAG_ENABLE_PANDA_BINARY_INPUT = 0x01;

#ifdef USE_HEXAGON_LIBS
int HexagonControlWrapper::GetVersion() {
  return soc_interface_GetSocControllerVersion();
}

bool HexagonControlWrapper::Init() {
  soc_interface_SetLogLevel(SHOW_DBG_IN_SOC ? -1 /* debug */ : 0 /* info */);
  if (DBG_USE_SAMPLE_INPUT) {
    soc_interface_SetDebugFlag(FLAG_ENABLE_PANDA_BINARY_INPUT);
  }
  return soc_interface_Init();
}

bool HexagonControlWrapper::Finalize() { return soc_interface_Finalize(); }
bool HexagonControlWrapper::SetupGraph(
    const GraphTransferer &graph_transferer) {
  int inputs_count = 0;
  int outputs_count = 0;
  for (const GraphTransferer::NodeInputParams& input_params :
       graph_transferer.GetNodeInputParams()) {
    inputs_count += input_params.input_node_id_and_output_port_list.size();
  }
  for (const GraphTransferer::NodeOutputParams& output_params :
       graph_transferer.GetNodeOutputParams()) {
    outputs_count += output_params.max_sizes.size();
  }
  // Allocate memory for node inputs and node outputs
  soc_interface_AllocateNodeInputAndNodeOutputArray(inputs_count,
                                                    outputs_count);

  // Construct node input parameters
  std::unordered_map<int, std::tuple<void*, int>> inputs_map;
  for (const GraphTransferer::NodeInputParams& input_params :
       graph_transferer.GetNodeInputParams()) {
    const int count = input_params.input_node_id_and_output_port_list.size();
    int node_ids[count];
    int ports[count];
    for (int i = 0; i < count; ++i) {
      const std::tuple<int, int> id_and_port =
          input_params.input_node_id_and_output_port_list.at(i);
      node_ids[i] = std::get<0>(id_and_port) + NODE_ID_OFFSET;
      ports[i] = std::get<1>(id_and_port);
    }
    void* inputs_ptr = soc_interface_SetOneNodeInputs(count, node_ids, ports);
    const int node_id = input_params.node_id;
    CHECK(inputs_map.count(node_id) == 0);
    inputs_map.emplace(node_id, std::make_tuple(inputs_ptr, count));
  }

  // Construct node output parameters
  std::unordered_map<int, std::tuple<void*, int>> outputs_map;
  for (const GraphTransferer::NodeOutputParams& output_params :
       graph_transferer.GetNodeOutputParams()) {
    const int count = output_params.max_sizes.size();
    int sizes[count];
    for (int i = 0; i < count; ++i) {
      const int size = output_params.max_sizes.at(i);
      sizes[i] = size;
    }
    void* outputs_ptr = soc_interface_SetOneNodeOutputs(count, sizes);
    const int node_id = output_params.node_id;
    CHECK(outputs_map.count(node_id) == 0);
    outputs_map.emplace(node_id, std::make_tuple(outputs_ptr, count));
  }

  // Instantiate graph
  soc_interface_InstantiateGraph();

  // Initialize graph
  // 1. Setup const nodes
  for (const GraphTransferer::ConstNodeTransferParams& params :
       graph_transferer.GetConstNodeParams()) {
    const int node_id = params.node_id;
    const int64 shape_0 = params.shape[0];
    const int64 shape_1 = params.shape[1];
    const int64 shape_2 = params.shape[2];
    const int64 shape_3 = params.shape[3];
    const int data_size = params.data_size;
    CHECK(dummy_const_data_.count(node_id) == 0);
    auto data = dummy_const_data_.emplace(
        std::piecewise_construct, std::make_tuple(node_id), std::make_tuple());
    CHECK(data.second);
    const int additional_bytes_for_alignment = 16;
    data.first->second.resize(data_size + additional_bytes_for_alignment - 1);
    const uintptr_t data_ptr_int =
        reinterpret_cast<uintptr_t>(data.first->second.data());
    const int shift_count = (16 - data_ptr_int % 16) % 16;
    uint8* data_ptr = data.first->second.data() + shift_count;
    std::memcpy(data_ptr, params.data.data(), data_size);
    soc_interface_AppendConstNode(params.name.c_str(), node_id + NODE_ID_OFFSET,
                                  shape_0, shape_1, shape_2, shape_3, data_ptr,
                                  data_size);
  }

  // 2. Setup op nodes
  for (const GraphTransferer::NodeTransferParams& params :
       graph_transferer.GetOpNodeParams()) {
    const int node_id = params.node_id;
    const int op_id = params.soc_op_id;
    CHECK(inputs_map.count(node_id) == 1);
    CHECK(outputs_map.count(node_id) <= 1);
    // Only output node doesn't have output
    const bool has_output = outputs_map.count(node_id) == 1;
    const auto& input_ptr_and_count = inputs_map.at(node_id);
    const void* input_ptr = std::get<0>(input_ptr_and_count);
    const int input_count = std::get<1>(input_ptr_and_count);
    void* output_ptr = nullptr;
    int output_count = 0;
    if (has_output) {
      const auto& output_ptr_and_count = outputs_map.at(node_id);
      output_ptr = std::get<0>(output_ptr_and_count);
      output_count = std::get<1>(output_ptr_and_count);
      CHECK(output_count > 0);
    }
    int padding_id = -1;
    if (params.padding == 0) {
      padding_id = 0;
    } else if (params.padding == Padding::SAME) {
      padding_id = 1;
    } else if (params.padding == Padding::VALID) {
      padding_id = 2;
    } else {
      CHECK(false);
    }
    soc_interface_AppendNode(params.name.c_str(), node_id + NODE_ID_OFFSET,
                             op_id, padding_id, input_ptr, input_count,
                             output_ptr, output_count);
  }

  LOG(INFO) << "Setup graph completed";

  // 3. construct graph
  return soc_interface_ConstructGraph();

  // Keep following comment to use dummy graph construction
  // return soc_interface_setupDummyGraph(3 /* inception version */);
}

bool HexagonControlWrapper::ExecuteGraph() {
  return soc_interface_ExecuteGraph();
}

bool HexagonControlWrapper::TeardownGraph() {
  soc_interface_ReleaseNodeInputAndNodeOutputArray();
  return soc_interface_TeardownGraph();
}

bool HexagonControlWrapper::FillInputNode(const string node_name,
                                          const ByteArray bytes) {
  uint64 byte_size;
  const int x = 1;
  const int y = 299;
  const int z = 299;
  const int d = 3;
  if (DBG_USE_DUMMY_INPUT) {
    const int array_length = x * y * z * d;
    byte_size = array_length * sizeof(float);
    dummy_input_float_.resize(array_length);
    std::memset(dummy_input_float_.data(), 0, byte_size);
  } else {
    CHECK(std::get<2>(bytes) == DT_FLOAT);
    byte_size = std::get<1>(bytes);
    dummy_input_float_.resize(byte_size / sizeof(float));
    std::memcpy(dummy_input_float_.data(), std::get<0>(bytes), byte_size);
  }
  return soc_interface_FillInputNodeFloat(
      x, y, z, d, reinterpret_cast<uint8*>(dummy_input_float_.data()),
      byte_size);
}

bool HexagonControlWrapper::ReadOutputNode(
    const string node_name, std::vector<ByteArray> *const outputs) {
  CHECK(outputs != nullptr);
  ByteArray output;
  soc_interface_ReadOutputNodeFloat(node_name.c_str(), &std::get<0>(output),
                                    &std::get<1>(output));
  // TODO: Accept all results
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
