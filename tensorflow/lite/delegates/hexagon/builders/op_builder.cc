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
#include "tensorflow/lite/delegates/hexagon/builders/op_builder.h"

#include "hexagon/hexagon_nn_ops.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/builders/op_factory.h"
#include <farmhash.h>

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {
// Farmhash Fingerprint
inline uint64_t CombineFingerprints(uint64_t l, uint64_t h) {
  // Murmur-inspired hashing.
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (l ^ h) * kMul;
  a ^= (a >> 47);
  uint64_t b = (h ^ a) * kMul;
  b ^= (b >> 44);
  b *= kMul;
  b ^= (b >> 41);
  b *= kMul;
  return b;
}

inline uint64_t ComputeHash(const int shape[], const char* data,
                            const int data_len) {
  return CombineFingerprints(
      ::util::Fingerprint64(data, data_len),
      ::util::Fingerprint64(reinterpret_cast<const char*>(shape),
                              sizeof(shape[0]) * 4));
}

inline uint64_t ComputeHash(const TfLiteTensor& tensor, const int shape[],
                            int int8_to_uint8) {
  auto data_hash = ComputeHash(shape, tensor.data.raw_const, tensor.bytes);
  auto int8_to_uint8_hash = ::util::Fingerprint64(
      reinterpret_cast<char*>(&int8_to_uint8), sizeof(int8_to_uint8));
  return CombineFingerprints(data_hash, int8_to_uint8_hash);
}

int GetElementSize(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return sizeof(float);
    case kTfLiteBool:
      return sizeof(bool);
    case kTfLiteInt32:
      return sizeof(int32_t);
    case kTfLiteInt8:
      return sizeof(int8_t);
    case kTfLiteUInt8:
      return sizeof(uint8_t);
    default:
      return sizeof(int8_t);
  }
}
}  // namespace

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
    case kTfLiteBuiltinStridedSlice:
      return CreateStridedSliceBuilder(this, OP_QuantizedStridedSlice_8);
    case kTfLiteBuiltinSquaredDifference:
      return CreateSquaredDifferenceOpBuilder(this, OP_QuantizedSub_8p8to8);
    case kTfLiteBuiltinRsqrt:
      return CreateRSqrtOpBuilder(this, OP_QuantizedSqrt_8);
    default:
      context_->ReportError(context_, "Op not supported: %d", op_type);
      return nullptr;
  }
}

OpBuilder* GraphBuilder::LookupConstData(uint64_t cache_key) {
  auto lookup_result = cache_.find(cache_key);
  if (lookup_result != cache_.end()) return lookup_result->second;
  return nullptr;
}

void GraphBuilder::AddToCache(uint64_t cache_key, OpBuilder* value) {
  cache_[cache_key] = value;
}

OpBuilder* GraphBuilder::AddConstNodeWithData(const int shape[], char* data,
                                              int data_size) {
  auto cache_key = ComputeHash(shape, data, data_size);
  if (auto lookup_result = LookupConstData(cache_key)) return lookup_result;
  builders_.emplace_back(new OpBuilder(this, OP_Const));
  builders_.back()->SetConstNode();
  builders_.back()->SetNodeId(builders_.size());
  int error = hexagon_nn_->hexagon_nn_append_const_node(
      graph_id_, builders_.size(), shape[0], shape[1], shape[2], shape[3],
      reinterpret_cast<const uint8_t*>(data), data_size);
  if (error != 0) {
    TF_LITE_KERNEL_LOG(context_, "Error adding const node with shape id: %d",
                       static_cast<int>(builders_.size()));
    return nullptr;
  }
  AddToCache(cache_key, builders_.back().get());
  return builders_.back().get();
}

OpBuilder* GraphBuilder::AddConstNodeWithData(int tensor_id,
                                              const TfLiteTensor& tensor,
                                              bool int8_to_uint8) {
  // Fetch shape of tensor and pad 1's so it is always 4D.
  int batch_size, height_size, width_size, depth_size;
  GetDims(&batch_size, &height_size, &width_size, &depth_size, tensor.dims);
  const int shape[] = {batch_size, height_size, width_size, depth_size};

  auto cache_key = ComputeHash(tensor, shape, int8_to_uint8 ? 1 : 0);
  if (auto lookup_result = LookupConstData(cache_key)) {
    // If tensor is cached but with no id, that can happen when the same
    // data is added from a constant value (not tensor). We can cache the data
    // and reuse it.
    // We assign the tensor to this cached const node before returning.
    if (!HasTensor(tensor_id))
      AddTensorWithID(tensor_id, lookup_result->GetID(), 0);
    return lookup_result;
  }
  builders_.emplace_back(new OpBuilder(this, OP_Const));
  const int node_id = builders_.size();
  builders_.back()->SetConstNode();
  builders_.back()->SetNodeId(node_id);
  int error = hexagon_nn_->hexagon_nn_append_const_node(
      graph_id_, node_id, batch_size, height_size, width_size, depth_size,
      reinterpret_cast<const uint8_t*>(tensor.data.raw), tensor.bytes);
  if (error > 0) {
    context_->ReportError(
        context_, "Failed to add const node for tensor with id: %d", tensor_id);
    return nullptr;
  }
  AddTensorWithID(tensor_id, node_id, 0);
  // We need to return the builder with result, so we can't rely
  // on builders_.back() as it can change while casting, so we hold pointer
  // and update with value from casting if needed.
  OpBuilder* result_builder = builders_.back().get();
  // Cast int8 to uint8 if requested.
  // This will add cast op to uint8 and update tensor map to point
  // to the casted tensor.
  if (int8_to_uint8 && tensor.type == kTfLiteInt8) {
    AddCastOp(context_, OP_Quantized_CastInt8ToUInt8, tensor_id,
              &result_builder);
  }
  AddToCache(cache_key, result_builder);
  return result_builder;
}

// TODO(b/154604279): Support these casting ops in Hexagon op profiling (which
// seems to key tensors on a single op, which may not be the case now).
TfLiteStatus GraphBuilder::AddCastOp(TfLiteContext* context, int op_type,
                                     int tensor_id,
                                     OpBuilder** cast_op_builder) {
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
  if (cast_op_builder != nullptr) *cast_op_builder = cast_builder;
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
    input_op->AddOutput(tensor.dims, GetElementSize(tensor.type));
    AddTensorWithID(tensor_id, input_op->GetID(), num_inputs);
    // If tensor is of type int8, add an op to cast it to uint8.
    if (tensor.type == kTfLiteInt8) {
      TF_LITE_ENSURE_STATUS(AddCastOp(context, OP_Quantized_CastInt8ToUInt8,
                                      tensor_id, /*cast_op_builder=*/nullptr));
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
      TF_LITE_ENSURE_STATUS(AddCastOp(context, OP_Quantized_CastUInt8ToInt8,
                                      tensor_id, /*cast_op_builder=*/nullptr));
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

OpBuilder::TensorID OpBuilder::AddOutput(const TfLiteIntArray* dims,
                                         int element_size) {
  op_node_.outputs.push_back(hexagon_nn_output());
  op_node_.outputs.back().elementsize = element_size;
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

TfLiteStatus OpBuilder::ComputeAndAddMinAndMax(TfLiteContext* context,
                                               const TfLiteTensor& tensor) {
  float tensor_min, tensor_max;
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(tensor, &tensor_min, &tensor_max));
  auto* min_const_node = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&tensor_min), sizeof(tensor_min));
  auto* max_const_node = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&tensor_max), sizeof(tensor_max));
  AddInput(TensorID(min_const_node->GetID(), 0));
  AddInput(TensorID(max_const_node->GetID(), 0));

  return kTfLiteOk;
}

// Static
constexpr int OpBuilder::kScalarShape[];

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
