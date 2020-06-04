/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/op_builder.h"

#include "hexagon/hexagon_nn_ops.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/delegates/hexagon/builders/op_factory.h"

namespace tflite {
namespace delegates {
namespace hexagon {

OpBuilder* GraphBuilder::CreateOpBuilderFromTfLiteOp(int op_type,
                                                     TfLiteNode* node) {
  switch (op_type) {
    case kTfLiteBuiltinAdd:
      return CreateArithmeticBuilder(this, OP_QuantizedAdd_8p8to8);
    case kTfLiteBuiltinArgMax:
      return CreateArgMinMaxOpBuilder(this, OP_ArgMax_8toInt32);
    case kTfLiteBuiltinArgMin:
      return CreateArgMinMaxOpBuilder(this, OP_ArgMin_8);
    case kTfLiteBuiltinMul:
      // The 32-bit version of Mul is more accurate, and robust to disparities
      // in input/output ranges.
      return CreateArithmeticBuilder(this, OP_QuantizedMul_8x8to32);
    case kTfLiteBuiltinSub:
      return CreateArithmeticBuilder(this, OP_QuantizedSub_8p8to8);
    case kTfLiteBuiltinMean:
      return CreateReduceBuilder(this, OP_QuantizedMean_8);
    case kTfLiteBuiltinSum:
      return CreateReduceBuilder(this, OP_QuantizedSum_8to32);
    case kTfLiteBuiltinPad:
      return CreatePadBuilder(this, OP_QuantizedPad_8);
    case kTfLiteBuiltinMirrorPad:
      return CreateMirrorPadBuilder(this, OP_MirrorPad_8);
    case kTfLiteBuiltinFullyConnected: {
      const auto& weights_tensor = context_->tensors[node->inputs->data[1]];
      if (weights_tensor.allocation_type == kTfLiteMmapRo)
        return CreateMatMulWithConstWeightsOpBuilder(
            this, OP_QuantizedMatMul_8x8to32);
      else
        return CreateMatMulOpBuilder(this, OP_Transpose_8);
    }
    case kTfLiteBuiltinAveragePool2d:
      return CreatePool2DBuilder(this, OP_QuantizedAvgPool_8);
    case kTfLiteBuiltinMaxPool2d:
      return CreatePool2DBuilder(this, OP_QuantizedMaxPool_8);
    case kTfLiteBuiltinConcatenation:
      return CreateConcatBuilder(this, OP_QuantizedConcat_8);
    case kTfLiteBuiltinConv2d:
      return CreateConv2DBuilder(this, OP_Supernode_8x8p32to8);
    case kTfLiteBuiltinTransposeConv:
      return CreateTransposeConv2DBuilder(
          this, OP_QuantizedTransposeConv2d_8x8p32to8);
    case kTfLiteBuiltinDepthwiseConv2d:
      return CreateConv2DBuilder(this, OP_DepthwiseSupernode_8x8p32to8);
    case kTfLiteBuiltinReshape:
      return CreateReshapeBuilder(this, OP_Reshape);
    case kTfLiteBuiltinSoftmax:
      return CreateSoftmaxBuilder(this, OP_QuantizedSoftmax_8);
    case kTfLiteBuiltinResizeNearestNeighbor:
      return CreateResizeNearestNeighborBuilder(this,
                                                OP_ResizeNearestNeighbor_8);
    case kTfLiteBuiltinL2Normalization:
      return CreateL2NormalizationBuilder(this, OP_L2Normalize_8);
    case kTfLiteBuiltinRelu:
      return CreateActivationBuilder(this, OP_QuantizedRelu_8);
    case kTfLiteBuiltinRelu6:
      return CreateActivationBuilder(this, OP_QuantizedReluX_8);
    case kTfLiteBuiltinTanh:
      return CreateActivationBuilder(this, OP_QuantizedTanh_8);
    case kTfLiteBuiltinLogistic:
      return CreateActivationBuilder(this, OP_QuantizedSigmoid_8);
    case kTfLiteBuiltinSplit:
      return CreateSplitBuilder(this, OP_QuantizedSplit_8);
    case kTfLiteBuiltinResizeBilinear:
      return CreateResizeBilinearOpBuilder(this, OP_QuantizedResizeBilinear_8);
    case kTfLiteBuiltinNeg:
      return CreateNegOpBuilder(this, OP_QuantizedNeg_8);
    case kTfLiteBuiltinTranspose:
      return CreateTransposeBuilder(this, OP_Transpose_8);
    case kTfLiteBuiltinSpaceToDepth:
      return CreateSpaceToDepthBuilder(this, OP_SpaceToDepth_8);
    case kTfLiteBuiltinDepthToSpace:
      return CreateSpaceToDepthBuilder(this, OP_DepthToSpace_8);
    case kTfLiteBuiltinQuantize:
      return CreateQuantizeBuilder(this, OP_Requantize_8to8);
    case kTfLiteBuiltinHardSwish:
      return CreateHardSwishBuilder(this, OP_QuantizedHardSwish_8);
    case kTfLiteBuiltinMinimum:
      return CreateMinMaxBuilder(this, OP_QuantizedMinimum_8);
    case kTfLiteBuiltinMaximum:
      return CreateMinMaxBuilder(this, OP_QuantizedMaximum_8);
    case kTfLiteBuiltinSlice:
      return CreateSliceOpBuilder(this, OP_QuantizedSlice_8);
    case kTfLiteBuiltinPack:
      return CreatePackBuilder(this, OP_QuantizedPack_8);
    default:
      context_->ReportError(context_, "Op not supported: %d", op_type);
      return nullptr;
  }
}

OpBuilder* GraphBuilder::AddConstNodeWithData(const int shape[], char* data,
                                              int data_size) {
  builders_.emplace_back(new OpBuilder(this, OP_Const));
  builders_.back()->SetConstNode();
  builders_.back()->SetNodeId(builders_.size());
  int error = hexagon_nn_->hexagon_nn_append_const_node(
      graph_id_, builders_.size(), shape[0], shape[1], shape[2], shape[3],
      reinterpret_cast<const uint8_t*>(data), data_size);
  if (error != 0) {
    context_->ReportError(context_, "Error adding const node with shape id: %d",
                          (int)builders_.size());
    return nullptr;
  }
  return builders_.back().get();
}

OpBuilder* GraphBuilder::AddConstNodeWithData(int tensor_id,
                                              const TfLiteTensor& tensor,
                                              bool int8_to_uint8) {
  builders_.emplace_back(new OpBuilder(this, OP_Const));
  const int node_id = builders_.size();
  builders_.back()->SetConstNode();
  builders_.back()->SetNodeId(node_id);
  int batch_size, height_size, width_size, depth_size;
  GetDims(&batch_size, &height_size, &width_size, &depth_size, tensor.dims);
  int error = hexagon_nn_->hexagon_nn_append_const_node(
      graph_id_, node_id, batch_size, height_size, width_size, depth_size,
      reinterpret_cast<const uint8_t*>(tensor.data.raw), tensor.bytes);
  if (error > 0) {
    context_->ReportError(
        context_, "Failed to add const node for tensor with id: %d", tensor_id);
    return nullptr;
  }
  AddTensorWithID(tensor_id, node_id, 0);
  // Cast int8 to uint8 if requested.
  // This will add cast op to uint8 and update tensor map to point
  // to the casted tensor.
  if (int8_to_uint8 && tensor.type == kTfLiteInt8) {
    AddCastOp(context_, OP_Quantized_CastInt8ToUInt8, tensor_id);
  }
  return builders_.back().get();
}

// TODO(b/154604279): Support these casting ops in Hexagon op profiling (which
// seems to key tensors on a single op, which may not be the case now).
TfLiteStatus GraphBuilder::AddCastOp(TfLiteContext* context, int op_type,
                                     int tensor_id) {
  // Create a new OpBuilder for casting the tensor.
  OpBuilder* cast_builder = CreateCastBuilder(this, op_type);
  builders_.emplace_back(cast_builder);
  cast_builder->SetNodeId(builders_.size());
  // We cast the tensor in-place, so there is only 1 input & output which is the
  // same.
  auto* tensor_data = TfLiteIntArrayCreate(1);
  tensor_data->data[0] = tensor_id;

  TF_LITE_ENSURE_STATUS(
      cast_builder->PopulateSubGraph(tensor_data, tensor_data, context));
  TF_LITE_ENSURE_STATUS(cast_builder->RegisterOutputs(tensor_data, context));

  TfLiteIntArrayFree(tensor_data);
  return kTfLiteOk;
}

TfLiteStatus GraphBuilder::AddInputTensors(const TfLiteIntArray* input_tensors,
                                           TfLiteContext* context) {
  auto* input_op = AddNode();
  input_op->SetOpType(OP_INPUT);

  // We need to track num_inputs since not all input_tensors are actual input
  // data. Some are constants.
  int num_inputs = 0;
  for (int i = 0; i < input_tensors->size; ++i) {
    const int tensor_id = input_tensors->data[i];
    const auto& tensor = context->tensors[tensor_id];
    if (tensor.allocation_type == kTfLiteMmapRo) continue;
    input_op->AddOutput(tensor.dims);
    AddTensorWithID(tensor_id, input_op->GetID(), num_inputs);
    // If tensor is of type int8, add an op to cast it to uint8.
    if (tensor.type == kTfLiteInt8) {
      TF_LITE_ENSURE_STATUS(
          AddCastOp(context, OP_Quantized_CastInt8ToUInt8, tensor_id));
    }
    ++num_inputs;
  }

  return kTfLiteOk;
}

TfLiteStatus GraphBuilder::AddOutputTensors(
    const TfLiteIntArray* output_tensors, TfLiteContext* context) {
  std::vector<OpBuilder::TensorID> hexagon_output_ids;
  hexagon_output_ids.reserve(output_tensors->size);

  for (int i = 0; i < output_tensors->size; ++i) {
    const int tensor_id = output_tensors->data[i];
    const auto& tensor = context->tensors[tensor_id];
    // If tensor is of type int8, add an op to cast it to uint8.
    if (tensor.type == kTfLiteInt8) {
      TF_LITE_ENSURE_STATUS(
          AddCastOp(context, OP_Quantized_CastUInt8ToInt8, tensor_id));
    }
    hexagon_output_ids.push_back(GetHexagonTensorId(tensor_id));
  }

  // Add Hexagon OUTPUT op.
  auto* output_op = AddNode();
  output_op->SetOpType(OP_OUTPUT);
  for (auto hexagon_output : hexagon_output_ids) {
    output_op->AddInput(hexagon_output);
  }

  return kTfLiteOk;
}

OpBuilder::TensorID OpBuilder::AddOutput(const TfLiteIntArray* dims) {
  op_node_.outputs.push_back(hexagon_nn_output());
  op_node_.outputs.back().elementsize = sizeof(uint8_t);
  op_node_.outputs.back().rank = 4;
  // TODO(karimnosseir): What is a good to estimate the max size ?
  int batch_size, height_size, width_size, depth_size;
  GetDims(&batch_size, &height_size, &width_size, &depth_size, dims);
  auto& max_sizes = op_node_.outputs.back().max_sizes;
  if (graph_builder_->GraphHasDynamicBatch()) {
    max_sizes[0] = graph_builder_->GetMaxBatchSize();
  } else {
    max_sizes[0] = batch_size;
  }
  max_sizes[1] = height_size;
  max_sizes[2] = width_size;
  max_sizes[3] = depth_size;
  return TensorID(GetID(), op_node_.outputs.size() - 1);
}

OpBuilder::TensorID OpBuilder::AddOutput(int elementsize, int rank,
                                         const int* max_sizes_vect) {
  op_node_.outputs.push_back(hexagon_nn_output());
  op_node_.outputs.back().elementsize = elementsize;
  op_node_.outputs.back().rank = rank;
  auto& max_sizes = op_node_.outputs.back().max_sizes;
  for (int i = 0; i < rank; ++i) {
    max_sizes[i] = max_sizes_vect[i];
  }
  if (graph_builder_->GraphHasDynamicBatch()) {
    max_sizes[0] = graph_builder_->GetMaxBatchSize();
  }
  return TensorID(GetID(), op_node_.outputs.size() - 1);
}

OpBuilder::TensorID OpBuilder::AddOutput(
    int elementsize, int rank, const std::vector<int>& max_sizes_vect) {
  return AddOutput(elementsize, rank, max_sizes_vect.data());
}

const OpNode* OpBuilder::Build() {
  for (const auto& id : input_ids_) {
    op_node_.inputs.push_back(hexagon_nn_input());
    op_node_.inputs.back().src_id = id.first;
    op_node_.inputs.back().output_idx = id.second;
  }
  return &op_node_;
}

OpBuilder* GraphBuilder::AddNode(int tflite_node_index) {
  OpBuilder* op = new OpBuilder(this, OP_Nop);
  builders_.emplace_back(op);
  op->SetNodeId(builders_.size());
  op->SetTFLiteNodeId(tflite_node_index);
  return op;
}

OpBuilder* GraphBuilder::AddNodeFromTfLiteOp(int op_type, TfLiteNode* node,
                                             int tflite_node_index) {
  OpBuilder* op = CreateOpBuilderFromTfLiteOp(op_type, node);
  builders_.emplace_back(op);
  op->SetNodeId(builders_.size());
  op->SetTFLiteNodeId(tflite_node_index);
  op->SetBuiltinData(node->builtin_data);
  op->SetTfLiteNode(node);
  return op;
}

void GraphBuilder::AddBatchSeqConfig(int max_size_for_batch,
                                     TfLiteIntArray* input_batch_dimensions,
                                     TfLiteIntArray* output_batch_dimensions) {
  OpBuilder* batch_seq_node =
      CreateBatchSeqBuilder(this, OP_BatchSeqConfig, max_size_for_batch,
                            input_batch_dimensions, output_batch_dimensions);
  builders_.emplace_back(batch_seq_node);
  batch_seq_node->SetNodeId(builders_.size());
  batch_seq_node->PopulateSubGraph(nullptr, nullptr, nullptr);
  max_size_for_batch_ = max_size_for_batch;
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
