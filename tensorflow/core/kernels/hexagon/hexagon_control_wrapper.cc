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

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"
#include "tensorflow/core/kernels/hexagon/soc_interface.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"

namespace tensorflow {

constexpr const char* const OUTPUT_OP_NAME = "OUTPUT";
constexpr const char* const REMOTE_FUSED_GRAPH_NODE_NAME_PREFIX =
    "hexagon_remote_fused_graph";
/* static */ constexpr const char* const
    HexagonControlWrapper::REMOTE_FUSED_GRAPH_EXECUTOR_NAME;

constexpr int ALIGNMENT_BYTES = 16;
constexpr int MAX_IN_OUT_COUNT = 128;

const bool DBG_DUMP_VERIFICATION_STRING = false;
const int DBG_LEVEL = 0;  // -2: verbose, -1: debug, 0: info
const bool DBG_USE_DUMMY_INPUT = false;
const bool DBG_USE_SAMPLE_INPUT = false;
const int64 FLAG_ENABLE_PANDA_BINARY_INPUT = 0x01;
const bool DBG_DUMP_INPUT_TENSOR_AS_FLOAT_DATA = false;

static string AddPort(const string& node_name) {
  if (node_name.find(':') != string::npos) {
    return node_name;
  } else {
    return strings::StrCat(node_name, ":", 0);
  }
}

static uint8* FindAlignedPointer(uint8* ptr) {
  const uintptr_t data_ptr_int = reinterpret_cast<uintptr_t>(ptr);
  const int shift_count =
      (ALIGNMENT_BYTES - data_ptr_int % ALIGNMENT_BYTES) % ALIGNMENT_BYTES;
  uint8* data_ptr = ptr + shift_count;
  return data_ptr;
}

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

int HexagonControlWrapper::GetVersion() {
  return soc_interface_GetSocControllerVersion();
}

bool HexagonControlWrapper::Init(const RemoteFusedGraphExecuteInfo& info) {
  soc_interface_SetLogLevel(DBG_LEVEL);
  if (DBG_USE_SAMPLE_INPUT) {
    soc_interface_SetDebugFlag(FLAG_ENABLE_PANDA_BINARY_INPUT);
  }
  if (info.serialized_executor_parameters().empty()) {
    std::vector<std::pair<string, Tensor>> inputs;
    std::vector<string> outputs;
    RemoteFusedGraphExecuteUtils::BuildRemoteGraphInputsAndOutputsFromProto(
        info, &inputs, &outputs);
    Status status = graph_transferer_.LoadGraphFromProto(
        HexagonOpsDefinitions::getInstance(), info.remote_graph(), inputs,
        outputs,
        false  // shape_inference_for_unknown_shape
    );
    TF_CHECK_OK(status) << status;
  } else {
    // If graph transfer info is attached, just import it.
    graph_transferer_.SetSerializedGraphTransferInfo(
        info.serialized_executor_parameters());
  }
  execute_info_ = &info;
  bool success = soc_interface_Init();
  if (!success) {
    LOG(ERROR) << "Hexagon initialization was failed.  See log output.";
    return false;
  }
  std::vector<int> input_sizes;
  std::vector<int> output_sizes;
  CHECK_NOTNULL(execute_info_);
  for (int i = 0; i < execute_info_->graph_input_node_name_size(); ++i) {
    const string& input = execute_info_->graph_input_node_name(i);
    LOG(INFO) << "Add input: " << input << ", " << i;
    CHECK(input_port_map_.emplace(AddPort(input), i).second);
    const RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& shape_type =
        execute_info_->default_graph_input_tensor_shape(i);
    int64 buf_size = DataTypeSize(shape_type.dtype());
    for (const TensorShapeProto::Dim& dim : shape_type.shape().dim()) {
      buf_size *= dim.size();
    }
    input_sizes.emplace_back(static_cast<int>(buf_size));
  }
  for (int i = 0; i < execute_info_->graph_output_node_name_size(); ++i) {
    const string& output = execute_info_->graph_output_node_name(i);
    CHECK(output_port_map_.emplace(AddPort(output), i).second);
    const RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& shape_type =
        execute_info_->default_graph_output_tensor_shape(i);

    int64 buf_size = DataTypeSize(shape_type.dtype());
    for (const TensorShapeProto::Dim& dim : shape_type.shape().dim()) {
      buf_size *= dim.size();
    }
    output_sizes.emplace_back(static_cast<int>(buf_size));
  }

  LOG(INFO) << "Allocate inout buffer";
  success &= soc_interface_AllocateInOutNodeBuffers(
      input_sizes.size(), input_sizes.data(), output_sizes.size(),
      output_sizes.data());
  return success;
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
        HexagonOpsDefinitions::getInstance().GetOpIdFor(OUTPUT_OP_NAME, {}));
    new_output_node_info.set_padding_id(0 /* PADDING_NA_ID */);
    new_output_node_info.set_input_count(1);
    new_output_node_info.set_output_count(0);

    const TensorId tid = ParseTensorName(graph_output.name());
    const string node_name = tid.first.ToString();
    const int port = tid.second;
    // Register node input for the new output node
    const GraphTransferInfo::NodeInfo* node_info =
        FindNodeInfo(node_name, &graph_transfer_info);
    CHECK_NE(node_info, nullptr);
    GraphTransferInfo::NodeInputInfo& node_input_info =
        *graph_transfer_info.add_node_input_info();
    node_input_info.set_node_id(new_output_node_id);
    GraphTransferInfo::NodeInput& node_input =
        *node_input_info.add_node_input();
    node_input.set_node_id(node_info->node_id());
    node_input.set_output_port(port);
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
    CHECK(count <= MAX_IN_OUT_COUNT);
    int node_ids[MAX_IN_OUT_COUNT];
    int ports[MAX_IN_OUT_COUNT];
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
    CHECK(count <= MAX_IN_OUT_COUNT);
    int sizes[MAX_IN_OUT_COUNT];
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
    data.first->second.resize(data_size + ALIGNMENT_BYTES - 1);
    uint8* data_ptr = FindAlignedPointer(data.first->second.data());
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

bool HexagonControlWrapper::FillInputNode(
    const string& node_name,
    const std::array<int64, GraphTransferer::SHAPE_ARRAY_SIZE>& shape,
    const ConstByteArray bytes) {
  const string tensor_name = AddPort(node_name);
  CHECK(input_port_map_.count(tensor_name) > 0);
  const int port = input_port_map_.at(tensor_name);
  if (input_tensor_data_.count(port) <= 0) {
    input_tensor_data_.emplace(port, std::vector<uint8>{});
  }
  std::vector<uint8>& input_tensor_data = input_tensor_data_.at(port);

  // hexagon only supports 32bit dimension
  const int x = static_cast<int>(shape[0]);
  const int y = static_cast<int>(shape[1]);
  const int z = static_cast<int>(shape[2]);
  const int d = static_cast<int>(shape[3]);

  const uint64 byte_size = x * y * z * d * DataTypeSize(std::get<2>(bytes));
  CHECK_EQ(byte_size, std::get<1>(bytes));
  input_tensor_data.resize(byte_size + ALIGNMENT_BYTES);
  uint8* data_ptr = FindAlignedPointer(input_tensor_data.data());

  if (DBG_USE_DUMMY_INPUT) {
    std::memset(data_ptr, 0, byte_size);
  } else {
    std::memcpy(data_ptr, std::get<0>(bytes), byte_size);
  }

  return soc_interface_FillInputNodeWithPort(port, x, y, z, d, data_ptr,
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
  std::vector<ByteArray> outputs;
  ReadOutputNode(node_name, &outputs);
  CHECK_EQ(1, outputs.size());
  ByteArray& output = outputs[0];
  Tensor* output_tensor = tensor_allocator(output_shape);
  CHECK(output_tensor->TotalBytes() >= std::get<1>(output))
      << output_tensor->TotalBytes() << ", " << std::get<1>(output);
  TF_CHECK_OK(RemoteFusedGraphExecuteUtils::CopyByteArrayToTensor(
      std::get<0>(output), std::get<1>(output), output_tensor));
  return true;
}

bool HexagonControlWrapper::ReadOutputNode(
    const string& node_name, std::vector<ByteArray>* const outputs) {
  CHECK(outputs != nullptr);
  ByteArray output;
  const string tensor_name = AddPort(node_name);
  CHECK(output_port_map_.count(tensor_name) > 0);
  const int port = output_port_map_.at(tensor_name);
  soc_interface_ReadOutputNodeWithPort(
      port, &std::get<0>(output),
      reinterpret_cast<uint64_t*>(&std::get<1>(output)));
  // TODO: Accept all results
  // std::get<2>(output) = DT_FLOAT;
  outputs->emplace_back(output);
  return true;
}

Status HexagonControlWrapper::FuseRemoteGraph(
    const GraphDef& original_graph_def, const std::vector<string>& inputs,
    const std::vector<string>& outputs, GraphDef* fused_graph_def) {
  const std::unordered_set<string> fused_node_names =
      RemoteFusedGraphExecuteUtils::BuildNodeMapFromOpsDefinitions(
          original_graph_def, HexagonOpsDefinitions::getInstance());
  // TODO(satok): We may want to place shape and type inside this function
  // if they are not placed in the given graph.
  TF_RETURN_IF_ERROR(RemoteFusedGraphExecuteUtils::FuseRemoteGraphByNodeNames(
      original_graph_def, inputs, outputs, REMOTE_FUSED_GRAPH_NODE_NAME_PREFIX,
      fused_node_names, REMOTE_FUSED_GRAPH_EXECUTOR_NAME,
      /*require_shape_type=*/true, fused_graph_def));
  return Status::OK();
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
  const std::array<int64, GraphTransferer::SHAPE_ARRAY_SIZE> shape =
      GraphTransferer::ToTensorShapeArray(tensor.shape());
  FillInputNode(node_name, shape, ba);
  return true;
}

bool HexagonControlWrapper::IsEnabled() const { return true; };
}  // namespace tensorflow
