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

#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"

#ifdef USE_HEXAGON_LIBS
#include "tensorflow/core/platform/hexagon/soc_interface.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#endif

namespace tensorflow {

constexpr const char* const INPUT_OP_NAME = "INPUT";
constexpr const char* const OUTPUT_OP_NAME = "OUTPUT";

const bool DBG_DUMP_VERIFICATION_STRING = false;
const bool SHOW_DBG_IN_SOC = false;
const bool DBG_USE_DUMMY_INPUT = false;
const bool DBG_USE_SAMPLE_INPUT = false;
const int64 FLAG_ENABLE_PANDA_BINARY_INPUT = 0x01;
const bool DBG_DUMP_INPUT_TENSOR_AS_FLOAT_DATA = false;

/* static */ GraphTransferInfo::NodeInfo* HexagonControlWrapper::FindNodeInfo(
    const string& name, GraphTransferInfo* graph_transfer_info) {
  for (GraphTransferInfo::NodeInfo& node_info :
       *graph_transfer_info->mutable_node_info()) {
    if (node_info.name() == name) {
      return &node_info;
    }
  }
  return nullptr;
}

#ifdef USE_HEXAGON_LIBS
int HexagonControlWrapper::GetVersion() {
  return soc_interface_GetSocControllerVersion();
}

bool HexagonControlWrapper::Init(const RemoteFusedGraphExecuteInfo& info) {
  soc_interface_SetLogLevel(SHOW_DBG_IN_SOC ? -1 /* debug */ : 0 /* info */);
  if (DBG_USE_SAMPLE_INPUT) {
    soc_interface_SetDebugFlag(FLAG_ENABLE_PANDA_BINARY_INPUT);
  }
  graph_transferer_.SetSerializedGraphTransferInfo(
      info.serialized_executor_parameters());
  execute_info_ = &info;
  return soc_interface_Init();
}

bool HexagonControlWrapper::Finalize() { return soc_interface_Finalize(); }
bool HexagonControlWrapper::SetupGraph() {
  // Copy graph transfer info to modify to adapt hexnn library
  GraphTransferInfo& graph_transfer_info =
      graph_transferer_.GetMutableGraphTransferInfo();

  // Overwrite op type of input nodes for hexagon
  for (const GraphTransferInfo::GraphInputNodeInfo& graph_input :
       graph_transfer_info.graph_input_node_info()) {
    GraphTransferInfo::NodeInfo* node_info =
        FindNodeInfo(graph_input.name(), &graph_transfer_info);
    CHECK_NE(node_info, nullptr);
    node_info->set_type_name(INPUT_OP_NAME);
    node_info->set_soc_op_id(
        HexagonOpsDefinitions::getInstance().GetOpIdFor(INPUT_OP_NAME));
  }

  // Generate a new output node which is connected to graph output node
  // TODO(satok): Support multiple output nodes
  CHECK_EQ(graph_transfer_info.graph_output_node_info_size(), 1);
  for (const GraphTransferInfo::GraphOutputNodeInfo& graph_output :
       graph_transfer_info.graph_output_node_info()) {
    const int new_output_node_id = graph_transfer_info.node_info_size() +
                                   graph_transfer_info.const_node_info_size() +
                                   2 /* offset for ids */;
    // Register a new output node
    GraphTransferInfo::NodeInfo& new_output_node_info =
        *graph_transfer_info.add_node_info();
    new_output_node_info.set_name(OUTPUT_OP_NAME);
    new_output_node_info.set_node_id(new_output_node_id);
    new_output_node_info.set_type_name(OUTPUT_OP_NAME);
    new_output_node_info.set_soc_op_id(
        HexagonOpsDefinitions::getInstance().GetOpIdFor(OUTPUT_OP_NAME));
    new_output_node_info.set_padding_id(0 /* PADDING_NA_ID */);
    new_output_node_info.set_input_count(1);
    new_output_node_info.set_output_count(0);

    // Register node input for the new output node
    const GraphTransferInfo::NodeInfo* node_info =
        FindNodeInfo(graph_output.name(), &graph_transfer_info);
    CHECK_NE(node_info, nullptr);
    GraphTransferInfo::NodeInputInfo& node_input_info =
        *graph_transfer_info.add_node_input_info();
    node_input_info.set_node_id(new_output_node_id);
    GraphTransferInfo::NodeInput& node_input =
        *node_input_info.add_node_input();
    node_input.set_node_id(node_info->node_id());
    node_input.set_output_port(0);
  }

  if (DBG_DUMP_VERIFICATION_STRING) {
    GraphTransferer gt;
    gt.SetSerializedGraphTransferInfo(graph_transfer_info.SerializeAsString());
    gt.DumpVerificationStringOfNodeTransferParams();
  }

  int inputs_count = 0;
  int outputs_count = 0;
  for (const GraphTransferInfo::NodeInputInfo& input_params :
       graph_transfer_info.node_input_info()) {
    inputs_count += input_params.node_input_size();
  }

  for (const GraphTransferInfo::NodeOutputInfo& output_params :
       graph_transfer_info.node_output_info()) {
    outputs_count += output_params.max_byte_size_size();
  }
  // Allocate memory for node inputs and node outputs
  soc_interface_AllocateNodeInputAndNodeOutputArray(inputs_count,
                                                    outputs_count);

  // Construct node input parameters
  std::unordered_map<int, std::tuple<void*, int>> inputs_map;
  for (const GraphTransferInfo::NodeInputInfo& input_params :
       graph_transfer_info.node_input_info()) {
    const int count = input_params.node_input_size();
    int node_ids[count];
    int ports[count];
    for (int i = 0; i < count; ++i) {
      const GraphTransferInfo::NodeInput& node_input =
          input_params.node_input(i);
      node_ids[i] = node_input.node_id() + NODE_ID_OFFSET;
      ports[i] = node_input.output_port();
    }
    void* inputs_ptr = soc_interface_SetOneNodeInputs(count, node_ids, ports);
    const int node_id = input_params.node_id();
    CHECK(inputs_map.count(node_id) == 0);
    inputs_map.emplace(node_id, std::make_tuple(inputs_ptr, count));
  }

  // Construct node output parameters
  std::unordered_map<int, std::tuple<void*, int>> outputs_map;
  for (const GraphTransferInfo::NodeOutputInfo& output_params :
       graph_transfer_info.node_output_info()) {
    const int count = output_params.max_byte_size_size();
    int sizes[count];
    for (int i = 0; i < count; ++i) {
      const int size = output_params.max_byte_size(i);
      sizes[i] = size;
    }
    void* outputs_ptr = soc_interface_SetOneNodeOutputs(count, sizes);
    const int node_id = output_params.node_id();
    CHECK(outputs_map.count(node_id) == 0);
    outputs_map.emplace(node_id, std::make_tuple(outputs_ptr, count));
  }

  // Instantiate graph
  soc_interface_InstantiateGraph();

  // Initialize graph
  // 1. Setup const nodes
  for (const GraphTransferInfo::ConstNodeInfo& params :
       graph_transfer_info.const_node_info()) {
    const int node_id = params.node_id();
    // TODO(satok): Stop assuming shape size is 4.
    CHECK(params.shape_size() == 4);
    const int64 shape_0 = params.shape(0);
    const int64 shape_1 = params.shape(1);
    const int64 shape_2 = params.shape(2);
    const int64 shape_3 = params.shape(3);
    const int data_size = params.data().length();
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
    std::memcpy(data_ptr, params.data().data(), data_size);
    soc_interface_AppendConstNode(params.name().c_str(),
                                  node_id + NODE_ID_OFFSET, shape_0, shape_1,
                                  shape_2, shape_3, data_ptr, data_size);
  }

  // 2. Setup op nodes
  for (const GraphTransferInfo::NodeInfo& params :
       graph_transfer_info.node_info()) {
    const int node_id = params.node_id();
    const int op_id = params.soc_op_id();
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
      // CHECK(output_count > 0);
    }
    int padding_id = -1;
    if (params.padding_id() == 0) {
      padding_id = 0;
    } else if (params.padding_id() == Padding::SAME) {
      padding_id = 1;
    } else if (params.padding_id() == Padding::VALID) {
      padding_id = 2;
    } else {
      CHECK(false);
    }
    soc_interface_AppendNode(params.name().c_str(), node_id + NODE_ID_OFFSET,
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

bool HexagonControlWrapper::FillInputNode(const string& node_name,
                                          const ConstByteArray bytes) {
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
    const string& node_name, TensorAllocatorFunc tensor_allocator) {
  CHECK_NE(execute_info_, nullptr);
  TensorShape output_shape;
  // TODO(satok): Switch shape corresponding to input shape
  for (int i = 0; i < execute_info_->graph_output_node_name_size(); ++i) {
    if (execute_info_->graph_output_node_name(i) == node_name) {
      for (const TensorShapeProto::Dim& dim :
           execute_info_->default_graph_output_tensor_shape(i).shape().dim()) {
        output_shape.AddDim(dim.size());
      }
      break;
    }
  }
  std::vector<IRemoteFusedGraphExecutor::ByteArray> outputs;
  ReadOutputNode(node_name, &outputs);
  Tensor* output = tensor_allocator(output_shape);
  CHECK(output->TotalBytes() >= std::get<1>(outputs[0]));
  // TODO(satok): Avoid specifying float
  std::memcpy(output->flat<float>().data(), std::get<0>(outputs[0]),
              std::get<1>(outputs[0]));
}

bool HexagonControlWrapper::ReadOutputNode(
    const string& node_name, std::vector<ByteArray>* const outputs) {
  CHECK(outputs != nullptr);
  ByteArray output;
  soc_interface_ReadOutputNodeFloat(node_name.c_str(), &std::get<0>(output),
                                    &std::get<1>(output));
  // TODO: Accept all results
  std::get<2>(output) = DT_FLOAT;
  outputs->emplace_back(output);
  return true;
}

bool HexagonControlWrapper::FillInputNode(const string& node_name,
                                          const Tensor& tensor) {
  StringPiece tensor_data = tensor.tensor_data();
  const ConstByteArray ba =
      ConstByteArray(reinterpret_cast<const uint8*>(tensor_data.data()),
                     tensor_data.size(), tensor.dtype());
  if (DBG_DUMP_INPUT_TENSOR_AS_FLOAT_DATA) {
    LOG(INFO) << "Input tensor data: element size = " << tensor.NumElements()
              << ", byte syze = " << tensor.TotalBytes();
    std::stringstream line;
    for (int i = 0; i < tensor.NumElements(); ++i) {
      line << tensor.flat<float>().data()[i] << ", ";
      if ((i - 2) % 3 == 0 || i == tensor.NumElements() - 1) {
        LOG(INFO) << "(" << ((i - 2) / 3) << ") " << line.str();
        line.str("");
        line.clear();
      }
    }
  }
  FillInputNode(node_name, ba);
  return true;
}

#else
int HexagonControlWrapper::GetVersion() { return -1; }
bool HexagonControlWrapper::Init(const RemoteFusedGraphExecuteInfo&) {
  return false;
}
bool HexagonControlWrapper::Finalize() { return false; }
bool HexagonControlWrapper::SetupGraph() { return false; }
bool HexagonControlWrapper::ExecuteGraph() { return false; }
bool HexagonControlWrapper::TeardownGraph() { return false; }
bool HexagonControlWrapper::FillInputNode(const string&, const ConstByteArray) {
  return false;
}
bool HexagonControlWrapper::FillInputNode(const string&, const Tensor&) {
  return false;
}
bool HexagonControlWrapper::ReadOutputNode(
    const string& node_name, TensorAllocatorFunc tensor_allocator) {
  return false;
}
bool HexagonControlWrapper::ReadOutputNode(const string&,
                                           std::vector<ByteArray>* const) {
  return false;
}
#endif

}  // namespace tensorflow
