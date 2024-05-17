/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/special/fc_fc_add.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {
bool UseBufferForWeights(const GpuInfo& gpu_info) {
  return gpu_info.IsAdreno() || gpu_info.IsAMD() || gpu_info.IsMali();
}

void RearrangeFCWeightsToOIO4I4(
    const tflite::gpu::Tensor<OHWI, DataType::INT8>& weights, uint8_t* dst) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);

  int counter = 0;
  for (int d = 0; d < dst_depth; ++d) {
    for (int s = 0; s < src_depth; ++s) {
      for (int i = 0; i < 4; ++i) {
        const int src_ch = s * 4 + i;
        for (int j = 0; j < 4; ++j) {
          const int dst_ch = d * 4 + j;
          if (src_ch < weights.shape.i && dst_ch < weights.shape.o) {
            int t =
                127 +
                weights.data[weights.shape.LinearIndex({dst_ch, 0, 0, src_ch})];
            if (t < 0) {
              t = 0;
            }
            dst[counter++] = t;
          } else {
            dst[counter++] = 127;
          }
        }
      }
    }
  }
}
}  // namespace

FCFCAdd::FCFCAdd(const OperationDef& definition, const GpuInfo& gpu_info)
    : GPUOperation(definition) {
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno3xx()) {
      work_group_size_ = int3(16, 4, 1);
    } else if (gpu_info.adreno_info.IsAdreno4xx()) {
      work_group_size_ = int3(32, 4, 1);
    } else {
      work_group_size_ = int3(32, 4, 1);
    }
  } else if (gpu_info.IsIntel()) {
    work_group_size_ = int3(8, 4, 1);
  } else if (gpu_info.IsNvidia()) {
    work_group_size_ = int3(8, 4, 1);
  } else if (gpu_info.IsPowerVR()) {
    work_group_size_ = int3(8, 4, 1);
  } else {
    work_group_size_ = int3(16, 4, 1);
  }
}

FCFCAdd::FCFCAdd(FCFCAdd&& kernel) : GPUOperation(std::move(kernel)) {}

FCFCAdd& FCFCAdd::operator=(FCFCAdd&& kernel) {
  if (this != &kernel) {
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

// We split vec vec dot (every thread do vec vec dot product in basic
// vec mat mult) on 4 parts to create more threads
// tid.y thread process every 4-th element in vec vec dot
// Good results for ~1024 x 1024 sizes, for other can be written more
// optimized shaders

std::string FCFCAdd::GetFCFCAddKernelCode(const OperationDef& op_def,
                                          const GpuInfo& gpu_info,
                                          bool weights_are_buffer,
                                          bool quantized_0, bool quantized_1) {
  AddSrcTensor("src_tensor_0", op_def.src_tensors[0]);
  AddSrcTensor("src_tensor_1", op_def.src_tensors[1]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  std::string c;

  c += "#define WG_X " + std::to_string(work_group_size_.x) + "\n";
  c += "#define WG_Y " + std::to_string(work_group_size_.y) + "\n";

  c += R"(MAIN_FUNCTION($0) {
  int gid = get_global_id(0);
  int2 tid;
  tid.x = LOCAL_ID_0;
  tid.y = LOCAL_ID_1;
  ACCUM_FLT4 s = INIT_ACCUM_FLT4(0.0f);
  if (gid < args.dst_tensor.Slices()) {
    for (int c = tid.y; c < args.src_tensor_0.Slices(); c += WG_Y) {
      FLT4 v = args.src_tensor_0.Read(0, 0, c);
)";
  if (weights_are_buffer) {
    c += R"(int weights_index = (c * args.dst_tensor.Slices() + gid) * 4;
      FLT4 partial = v.x * args.weights0.Read(weights_index + 0);
      partial += v.y * args.weights0.Read(weights_index + 1);
      partial += v.z * args.weights0.Read(weights_index + 2);
      partial += v.w * args.weights0.Read(weights_index + 3);
      s += TO_ACCUM_TYPE(partial);
)";
  } else {
    const std::string read_as_type =
        op_def.precision == CalculationsPrecision::F32 ? "float" : "half";
    c += "      FLT4 w0 = args.weights0.Read<" + read_as_type +
         ">(c * 4 + 0, gid);\n";
    c += "      FLT4 w1 = args.weights0.Read<" + read_as_type +
         ">(c * 4 + 1, gid);\n";
    c += "      FLT4 w2 = args.weights0.Read<" + read_as_type +
         ">(c * 4 + 2, gid);\n";
    c += "      FLT4 w3 = args.weights0.Read<" + read_as_type +
         ">(c * 4 + 3, gid);\n";
    if (quantized_0) {
      c += R"(w0 = w0 * args.q0_m + args.q0_a;
      w1 = w1 * args.q0_m + args.q0_a;
      w2 = w2 * args.q0_m + args.q0_a;
      w3 = w3 * args.q0_m + args.q0_a;
)";
    }
    c += R"(FLT4 partial = v.x * w0;
      partial += v.y * w1;
      partial += v.z * w2;
      partial += v.w * w3;
      s += TO_ACCUM_TYPE(partial);
)";
  }
  c += R"(    }
    for (int c = tid.y; c < args.src_tensor_1.Slices(); c += WG_Y) {
      FLT4 v = args.src_tensor_1.Read(0, 0, c);
      )";
  if (weights_are_buffer) {
    c += R"(int weights_index = (c * args.dst_tensor.Slices() + gid) * 4;
      FLT4 partial = v.x * args.weights1.Read(weights_index + 0);
      partial += v.y * args.weights1.Read(weights_index + 1);
      partial += v.z * args.weights1.Read(weights_index + 2);
      partial += v.w * args.weights1.Read(weights_index + 3);
      s += TO_ACCUM_TYPE(partial);
)";
  } else {
    const std::string read_as_type =
        op_def.precision == CalculationsPrecision::F32 ? "float" : "half";
    c += "      FLT4 w0 = args.weights1.Read<" + read_as_type +
         ">(c * 4 + 0, gid);\n";
    c += "      FLT4 w1 = args.weights1.Read<" + read_as_type +
         ">(c * 4 + 1, gid);\n";
    c += "      FLT4 w2 = args.weights1.Read<" + read_as_type +
         ">(c * 4 + 2, gid);\n";
    c += "      FLT4 w3 = args.weights1.Read<" + read_as_type +
         ">(c * 4 + 3, gid);\n";
    if (quantized_1) {
      c += R"(w0 = w0 * args.q1_m + args.q1_a;
      w1 = w1 * args.q1_m + args.q1_a;
      w2 = w2 * args.q1_m + args.q1_a;
      w3 = w3 * args.q1_m + args.q1_a;
)";
    }
    c += R"(FLT4 partial = v.x * w0;
      partial += v.y * w1;
      partial += v.z * w2;
      partial += v.w * w3;
      s += TO_ACCUM_TYPE(partial);
)";
  }
  c += R"(    }
  }
  __local ACCUM_FLT4 temp[WG_X][WG_Y];
  temp[tid.x][tid.y] = s;
  LOCAL_MEM_BARRIER;
  if (gid >= args.dst_tensor.Slices()) {
    return;
  }
  if (tid.y == 0) {
)";
  for (int i = 1; i < work_group_size_.y; ++i) {
    c += "    s += temp[tid.x][" + std::to_string(i) + "];\n";
  }
  c +=
      R"(    FLT4 r0 = TO_FLT4(s) + args.biases0.Read(gid) + args.biases1.Read(gid);
    args.dst_tensor.Write(r0, 0, 0, gid);
  }
})";

  return c;
}

int3 FCFCAdd::GetGridSize() const { return int3(dst_[0]->Slices(), 1, 1); }

void FCFCAdd::UploadQuantizedWeights(
    const tflite::gpu::Tensor<OHWI, DataType::INT8>& weights, float scale,
    float zero_point, int index) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);

  std::vector<uint8_t> data(src_depth * 4 * dst_depth * 4);
  RearrangeFCWeightsToOIO4I4(weights, data.data());
  TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
      DataType::UINT8, TensorStorageType::TEXTURE_2D, src_depth * 4, dst_depth,
      data.data());

  std::string q_name = "q" + std::to_string(index) + "_";
  if (definition_.precision == CalculationsPrecision::F32) {
    args_.AddFloat(q_name + "m", scale);
    args_.AddFloat(q_name + "a", -scale * (127.0 + zero_point));
  } else {
    args_.AddHalf(q_name + "m", half(scale));
    args_.AddHalf(q_name + "a", half(-scale * (127.0 + zero_point)));
  }
  args_.AddObject("weights" + std::to_string(index),
                  std::make_unique<TensorDescriptor>(std::move(desc)));
}

FCFCAdd CreateFCFCAdd(const GpuInfo& gpu_info, const OperationDef& definition,
                      const FullyConnectedAttributes& attr0,
                      const FullyConnectedAttributes& attr1) {
  FCFCAdd result(definition, gpu_info);
  bool weights_are_buffer = UseBufferForWeights(gpu_info);
  result.UploadWeights(attr0.weights, "weights0", weights_are_buffer);
  result.UploadWeights(attr1.weights, "weights1", weights_are_buffer);
  result.code_ = result.GetFCFCAddKernelCode(definition, gpu_info,
                                             weights_are_buffer, false, false);

  TensorDescriptor bias0_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), attr0.bias);
  result.args_.AddObject("biases0", std::make_unique<TensorDescriptor>(
                                        std::move(bias0_tensor_desc)));

  TensorDescriptor bias1_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), attr1.bias);
  result.args_.AddObject("biases1", std::make_unique<TensorDescriptor>(
                                        std::move(bias1_tensor_desc)));

  return result;
}

FCFCAdd CreateFCFCAdd(const GpuInfo& gpu_info, const OperationDef& definition,
                      const FullyConnectedInt8Attributes& attr0,
                      const FullyConnectedInt8Attributes& attr1) {
  FCFCAdd result(definition, gpu_info);
  result.UploadQuantizedWeights(attr0.weights, attr0.scale, attr0.zero_point,
                                0);
  result.UploadQuantizedWeights(attr1.weights, attr1.scale, attr1.zero_point,
                                1);
  result.code_ =
      result.GetFCFCAddKernelCode(definition, gpu_info, false, true, true);

  TensorDescriptor bias0_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), attr0.bias);
  result.args_.AddObject("biases0", std::make_unique<TensorDescriptor>(
                                        std::move(bias0_tensor_desc)));

  TensorDescriptor bias1_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), attr1.bias);
  result.args_.AddObject("biases1", std::make_unique<TensorDescriptor>(
                                        std::move(bias1_tensor_desc)));

  return result;
}

// fully connected + fully connected + add
absl::Status TryFCFCAdd(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const GraphFloat32& graph, NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
  if (!(gpu_info.IsIntel() || gpu_info.IsNvidia() || gpu_info.IsAMD())) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto* fc0_node = graph.GetNode(first_node_id);
  if (fc0_node == nullptr) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto first_op_type = OperationTypeFromString(fc0_node->operation.type);
  if (first_op_type != OperationType::FULLY_CONNECTED &&
      first_op_type != OperationType::FULLY_CONNECTED_INT8) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  const bool first_quantized =
      first_op_type == OperationType::FULLY_CONNECTED_INT8;
  auto fc0_inputs = graph.FindInputs(fc0_node->id);
  if (fc0_inputs.size() != 1) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto fc0_output_id = graph.FindOutputs(fc0_node->id)[0]->id;
  auto consumers = graph.FindConsumers(fc0_output_id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto* add_node = consumers[0];
  if (add_node == nullptr) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  if (consumed_nodes->find(add_node->id) != consumed_nodes->end()) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  if (OperationTypeFromString(add_node->operation.type) != OperationType::ADD) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto add_inputs = graph.FindInputs(add_node->id);
  if (add_inputs.size() != 2) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto fc1_output_id = add_inputs[0]->id + add_inputs[1]->id - fc0_output_id;
  auto* fc1_node = graph.FindProducer(fc1_output_id);
  if (fc1_node == nullptr) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto second_op_type = OperationTypeFromString(fc1_node->operation.type);
  if (second_op_type != OperationType::FULLY_CONNECTED &&
      second_op_type != OperationType::FULLY_CONNECTED_INT8) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  const bool second_quantized =
      second_op_type == OperationType::FULLY_CONNECTED_INT8;
  const bool both_quantized = first_quantized && second_quantized;
  const bool both_not_quantized = !first_quantized && !second_quantized;
  if (!(both_quantized || both_not_quantized)) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  if (consumed_nodes->find(fc1_node->id) != consumed_nodes->end()) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto fc1_inputs = graph.FindInputs(fc1_node->id);
  if (fc1_inputs.size() != 1) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto add_outputs = graph.FindOutputs(add_node->id);

  OperationDef op_def;
  op_def.precision = precision;
  auto it = tensor_descriptors.find(fc0_inputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.src_tensors.push_back(it->second);
  }
  it = tensor_descriptors.find(fc1_inputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.src_tensors.push_back(it->second);
  }
  it = tensor_descriptors.find(add_outputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.dst_tensors.push_back(it->second);
  }

  for (int i = 0; i < fc1_inputs.size(); ++i) {
    fc0_inputs.push_back(fc1_inputs[i]);
  }
  std::unique_ptr<GPUOperation>* gpu_op =
      InitSingleOpSubgraph(fc0_inputs, add_outputs, gpu_subgraph);
  FCFCAdd fc;
  if (both_not_quantized) {
    auto fc0_attr = absl::any_cast<FullyConnectedAttributes>(
        fc0_node->operation.attributes);
    auto fc1_attr = absl::any_cast<FullyConnectedAttributes>(
        fc1_node->operation.attributes);
    if (fc0_attr.weights.shape.o != fc1_attr.weights.shape.o) {
      return absl::NotFoundError("FCFCAdd not suitable.");
    }
    fc = CreateFCFCAdd(gpu_info, op_def, fc0_attr, fc1_attr);
  } else {
    // both_quantized
    auto fc0_attr = absl::any_cast<FullyConnectedInt8Attributes>(
        fc0_node->operation.attributes);
    auto fc1_attr = absl::any_cast<FullyConnectedInt8Attributes>(
        fc1_node->operation.attributes);
    if (fc0_attr.weights.shape.o != fc1_attr.weights.shape.o) {
      return absl::NotFoundError("FCFCAdd not suitable.");
    }
    fc = CreateFCFCAdd(gpu_info, op_def, fc0_attr, fc1_attr);
  }
  *gpu_op = std::make_unique<FCFCAdd>(std::move(fc));
  const std::string fused_nodes = std::to_string(fc0_node->id) + " " +
                                  std::to_string(fc1_node->id) + " " +
                                  std::to_string(add_node->id);
  gpu_subgraph->operations[0].name =
      "fully_connected_x2_and_add " + fused_nodes;
  consumed_nodes->insert(fc0_node->id);
  consumed_nodes->insert(fc1_node->id);
  consumed_nodes->insert(add_node->id);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
