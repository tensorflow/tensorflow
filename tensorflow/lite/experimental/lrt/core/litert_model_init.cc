// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NDEBUG
// Make flatbuffers verifier `assert` in debug mode.
#define FLATBUFFERS_DEBUG_VERIFICATION_FAILURE

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers  // IWYU pragma: keep
#include "tensorflow/lite/experimental/lrt/core/util/buffer_ref.h"
#endif

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "tensorflow/compiler/mlir/lite/core/model_builder_base.h"
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/c/litert_op_code.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/litert_model_init.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/experimental/lrt/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"

using ::litert::OwningBufferRef;
using ::litert::internal::VerifyFlatbuffer;

LiteRtStatus IsOpSupported(const tflite::OperatorT& op) {
  // TODO: b/365299994 - Check for supported options.

  if (!op.custom_options.empty()) {
    // TODO: b/365299994 - Support custom options.
    _LITERT_D_MSG("Custom options not supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (!op.intermediates.empty()) {
    // TODO: b/365299994 - Support intermediates.
    _LITERT_D_MSG("Intermediate tensors not supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (op.large_custom_options_size != 0) {
    // TODO: b/365299994 - Support large custom options.
    _LITERT_D_MSG("Large custom options not supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  for (auto m_input : op.mutating_variable_inputs) {
    if (m_input) {
      // TODO: b/365299994 - Support mutating variable inputs.
      _LITERT_D_MSG("Mutating variable inputs not supported.");
      return kLiteRtStatusErrorUnsupported;
    }
  }

  return kLiteRtStatusOk;
}

LiteRtStatus IsBufferSupported(const tflite::BufferT& buffer) {
  if (buffer.offset != 0) {
    // TODO: b/365299994 - Support buffer with offset.
    _LITERT_D_MSG("Buffers with offset not supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus IsTensorSupported(const tflite::TensorT& tensor) {
  if (!tensor.has_rank) {
    // TODO: b/365299994 - Support unranked tensors.
    _LITERT_D_MSG("Unranked tensors not supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tensor.is_variable) {
    // TODO: b/365299994 - Support variable tensors.
    _LITERT_D_MSG("Variable tensors not supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (!tensor.variant_tensors.empty()) {
    // TODO: b/365299994 - Support variant tensors.
    _LITERT_D_MSG("Variant tensors not supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (!tensor.shape_signature.empty()) {
    // TODO: b/365299994 - Support shape signature.
    _LITERT_D_MSG("Shape signature not supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tensor.sparsity) {
    // TODO: b/365299994 - Support sparsity tensors.
    _LITERT_D_MSG("Sparsity tensors not supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tensor.type != tflite::TensorType_FLOAT32 &&
      tensor.type != tflite::TensorType_INT32 &&
      tensor.type != tflite::TensorType_BOOL) {
    // TODO: b/365299994 - Support all element types.
    _LITERT_D_MSG("Only f32 supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus SetDefaultOptions(tflite::BuiltinOptionsUnion& opts,
                               LiteRtOpCode code) {
  switch (code) {
    case kLiteRtOpCodeTflMul:
      opts.Set(tflite::MulOptionsT());
      break;
    case kLiteRtOpCodeTflAdd:
      opts.Set(tflite::AddOptionsT());
      break;
    case kLiteRtOpCodeTflCustom:
      return kLiteRtStatusOk;
    default:
      return kLiteRtStatusErrorUnsupported;
  }
  return kLiteRtStatusOk;
}

//===----------------------------------------------------------------------===//
//                               Load                                         //
//===----------------------------------------------------------------------===//

class ModelUnpacker {
 public:
  static LiteRtStatus Unpack(LiteRtModel model);

 private:
  explicit ModelUnpacker(LiteRtModel model) : model_(model) {}

  LiteRtStatus ConvertTensor(const tflite::TensorT& tensor,
                             LiteRtTensor target);

  LiteRtStatus ConvertOp(const tflite::OperatorT& op,
                         std::vector<LiteRtTensor>& tensors, LiteRtOp target);

  LiteRtStatus UnpackSubgraph(LiteRtSubgraph target);

  LiteRtOpCode GetOpCode(uint32_t ind) {
    return static_cast<LiteRtOpCode>(Fb().operator_codes[ind]->builtin_code);
  }

  std::unique_ptr<tflite::BufferT> GetBuffer(uint32_t ind) {
    return std::move(Fb().buffers[ind]);
  }

  tflite::ModelT& Fb() { return *model_->flatbuffer_model; }

  LiteRtModel model_;
};

LiteRtStatus ModelUnpacker::ConvertTensor(const tflite::TensorT& tensor,
                                          LiteRtTensor target) {
  LITERT_RETURN_STATUS_IF_NOT_OK(IsTensorSupported(tensor));

  const auto buffer_ind = tensor.buffer;

  if (buffer_ind != 0) {
    target->weights.fb_buffer = GetBuffer(buffer_ind);
    LITERT_RETURN_STATUS_IF_NOT_OK(
        IsBufferSupported(*target->weights.fb_buffer));
  }

  target->type_id = kLiteRtRankedTensorType;

  auto& ranked_tensor = target->type_detail.ranked_tensor_type;

  ranked_tensor.element_type = kLiteRtElementTypeFloat32;
  ranked_tensor.layout.rank = tensor.shape.size();
  ranked_tensor.layout.dimensions = tensor.shape.data();
  ranked_tensor.layout.strides =
      nullptr;  // TFL tensors don't support strides yet.

  return kLiteRtStatusOk;
}

LiteRtStatus ModelUnpacker::ConvertOp(const tflite::OperatorT& op,
                                      std::vector<LiteRtTensor>& tensors,
                                      LiteRtOp target) {
  target->op_code = GetOpCode(op.opcode_index);

  for (auto input : op.inputs) {
    // Skipping optional input tensor.
    if (input == -1) {
      continue;
    }
    auto& input_tensor = tensors[input];

    input_tensor->users.push_back(target);
    input_tensor->user_arg_inds.push_back(target->inputs.size());

    target->inputs.push_back(input_tensor);
  }

  for (auto output : op.outputs) {
    auto& output_tensor = tensors[output];

    output_tensor->defining_op_out_ind = target->outputs.size();
    output_tensor->defining_op = target;

    target->outputs.push_back(output_tensor);
  }
  target->option = op.builtin_options;

  return kLiteRtStatusOk;
}

LiteRtStatus ModelUnpacker::UnpackSubgraph(LiteRtSubgraph target) {
  auto& subgraph = target->flatbuffer_subgraph;

  for (int i = 0; i < subgraph->tensors.size(); ++i) {
    auto& flatbuffer_tensor = *subgraph->tensors[i];
    LITERT_RETURN_STATUS_IF_NOT_OK(IsTensorSupported(flatbuffer_tensor));

    auto& tensor = target->tensors_storage.emplace_back();
    target->tensors.push_back(&tensor);

    LITERT_RETURN_STATUS_IF_NOT_OK(ConvertTensor(flatbuffer_tensor, &tensor));
  }

  for (int i = 0; i < subgraph->operators.size(); ++i) {
    auto& flatbuffer_op = *subgraph->operators[i];

    auto& op = target->ops_storage.emplace_back();
    target->ops.push_back(&op);

    LITERT_RETURN_STATUS_IF_NOT_OK(
        ConvertOp(flatbuffer_op, target->tensors, &op));
  }

  for (auto input : subgraph->inputs) {
    target->inputs.push_back(target->tensors[input]);
  }

  for (auto output : subgraph->outputs) {
    target->outputs.push_back(target->tensors[output]);
  }

  return kLiteRtStatusOk;
}

LiteRtStatus ModelUnpacker::Unpack(LiteRtModel model) {
  ModelUnpacker unpacker(model);

  if (unpacker.Fb().subgraphs.size() != 1) {
    // TODO: b/365299994 - Support multi subgraph.
    _LITERT_D_MSG("Only single subgraph models suported.");
    return kLiteRtStatusErrorUnsupported;
  }

  auto& subgraph = model->subgraphs.emplace_back();
  subgraph.flatbuffer_subgraph = std::move(unpacker.Fb().subgraphs[0]);
  LITERT_RETURN_STATUS_IF_NOT_OK(unpacker.UnpackSubgraph(&subgraph));

  return kLiteRtStatusOk;
}

LiteRtStatus RegisterCustomOpCode(LiteRtModel model, const char* new_op_code) {
  model->custom_op_code.assign(new_op_code);
  return kLiteRtStatusOk;
}

LiteRtStatus LoadModel(std::unique_ptr<tflite::ModelT> flatbuffer,
                       LiteRtModel* model) {
  auto litert_model = std::make_unique<LiteRtModelT>();
  litert_model->flatbuffer_model = std::move(flatbuffer);
  litert_model->subgraphs.reserve(100);

  LITERT_RETURN_STATUS_IF_NOT_OK(ModelUnpacker::Unpack(litert_model.get()));

  litert_model->flatbuffer_model->subgraphs.clear();

  // Set as empty string in case its not set explictly.
  LITERT_RETURN_STATUS_IF_NOT_OK(RegisterCustomOpCode(litert_model.get(), ""));

  *model = litert_model.release();

  return kLiteRtStatusOk;
}

LiteRtStatus LoadModel(const uint8_t* buf, size_t buf_size,
                       LiteRtModel* model) {
  LITERT_ENSURE(VerifyFlatbuffer(buf, buf_size),
                kLiteRtStatusErrorInvalidFlatbuffer,
                "Failed to verify flatbuffer");
  return LoadModel(tflite::UnPackModel(buf), model);
}

LiteRtStatus LoadModelFromFile(const char* path, LiteRtModel* model) {
  std::unique_ptr<tflite::Allocation> alloc =
      tflite::GetAllocationFromFile(path, tflite::DefaultErrorReporter());
  if (!alloc->valid()) {
    return kLiteRtStatusErrorFileIO;
  }

  return LoadModel(reinterpret_cast<const uint8_t*>(alloc->base()),
                   alloc->bytes(), model);
}

void ModelDestroy(LiteRtModel model) { delete model; }

//===----------------------------------------------------------------------===//
//                                 Serialize                                  //
//===----------------------------------------------------------------------===//

class ModelRepacker {
 public:
  static LiteRtStatus Repack(LiteRtModel model);

 private:
  static void BuildOpCodeMap(LiteRtModel model,
                             std::unordered_map<LiteRtOpCode, uint32_t>& map);

  explicit ModelRepacker(LiteRtModel model) : model_(model) {
    BuildOpCodeMap(model_, op_code_map_);
  }

  LiteRtStatus SerializeTensor(LiteRtTensor tensor, tflite::TensorT& target);

  LiteRtStatus SerializeOp(
      LiteRtOp op, tflite::OperatorT& target,
      const std::unordered_map<LiteRtTensor, int32_t>& tensor_map);

  LiteRtStatus SerializeSubgraph(LiteRtSubgraph subgraph,
                                 tflite::SubGraphT& target);

  uint32_t SubmitBuffer(std::unique_ptr<tflite::BufferT> buffer) {
    OldFb().buffers.push_back(std::move(buffer));
    return OldFb().buffers.size() - 1;
  }

  tflite::ModelT& OldFb() { return *model_->flatbuffer_model; }

  LiteRtModel model_;
  std::unordered_map<LiteRtOpCode, uint32_t> op_code_map_;
};

void ModelRepacker::BuildOpCodeMap(
    LiteRtModel model, std::unordered_map<LiteRtOpCode, uint32_t>& map) {
  // Add the user set custom code to the flatbuffers known codes.
  auto& custom_code = model->flatbuffer_model->operator_codes.emplace_back(
      std::make_unique<tflite::OperatorCodeT>());
  custom_code->builtin_code = tflite::BuiltinOperator_CUSTOM;
  custom_code->custom_code = model->custom_op_code;
  custom_code->version = 1;

  auto& codes = model->flatbuffer_model->operator_codes;

  for (int i = 0; i < codes.size(); ++i) {
    const auto tfl_code = codes[i]->builtin_code;
    map.insert({static_cast<LiteRtOpCode>(tfl_code), i});
  }
}

LiteRtStatus ModelRepacker::SerializeTensor(LiteRtTensor tensor,
                                            tflite::TensorT& target) {
  target.has_rank = true;
  const auto& type = tensor->type_detail.ranked_tensor_type;
  // TODO: b/365299994 - Map litert element types to flatbuffer elements types.
  target.type = tflite::TensorType_FLOAT32;

  for (int i = 0; i < type.layout.rank; ++i) {
    target.shape.push_back(type.layout.dimensions[i]);
  }

  // TFL tensors don't support strides yet.
  DCHECK(type.layout.strides == nullptr);

  DCHECK(tensor->weights.fb_buffer != nullptr) << "Submitting a null buffer";
  target.buffer = SubmitBuffer(std::move(tensor->weights.fb_buffer));

  return kLiteRtStatusOk;
}

LiteRtStatus ModelRepacker::SerializeOp(
    LiteRtOp op, tflite::OperatorT& target,
    const std::unordered_map<LiteRtTensor, int32_t>& tensor_map) {
  target.opcode_index = op_code_map_.at(op->op_code);

  for (auto in : op->inputs) {
    target.inputs.push_back(tensor_map.at(in));
  }

  for (auto out : op->outputs) {
    target.outputs.push_back(tensor_map.at(out));
  }

  // TODO: b/365299994 - Support options in serialize.
  LITERT_RETURN_STATUS_IF_NOT_OK_MSG(
      SetDefaultOptions(target.builtin_options, op->op_code),
      "Failed serializing options");

  if (op->custom_options.Size() != 0) {
    target.custom_options = op->custom_options.ToVec();
    target.custom_options_format = tflite::CustomOptionsFormat_FLEXBUFFERS;
  }
  // TODO: b/365299994 - Support exotic op fields in serialize.

  return kLiteRtStatusOk;
}

LiteRtStatus ModelRepacker::SerializeSubgraph(LiteRtSubgraph subgraph,
                                              tflite::SubGraphT& target) {
  std::unordered_map<LiteRtTensor, int32_t> tensor_map;

  for (auto tensor : subgraph->tensors) {
    tensor_map.insert({tensor, tensor_map.size()});
    target.tensors.push_back(std::make_unique<tflite::TensorT>());
    LITERT_RETURN_STATUS_IF_NOT_OK(
        SerializeTensor(tensor, *target.tensors.back()));
  }

  for (auto op : subgraph->ops) {
    target.operators.push_back(std::make_unique<tflite::OperatorT>());
    LITERT_RETURN_STATUS_IF_NOT_OK(
        SerializeOp(op, *target.operators.back(), tensor_map));
  }

  for (auto in : subgraph->inputs) {
    target.inputs.push_back(tensor_map.at(in));
  }
  for (auto out : subgraph->outputs) {
    target.outputs.push_back(tensor_map.at(out));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus ModelRepacker::Repack(LiteRtModel model) {
  ModelRepacker repacker(model);

  auto& target = repacker.OldFb();

  std::vector<std::pair<std::string, std::unique_ptr<tflite::BufferT>>>
      metadata;
  for (auto& flatbuffer_metadata : target.metadata) {
    const auto metadata_buffer_ind = flatbuffer_metadata->buffer;
    metadata.push_back({flatbuffer_metadata->name,
                        std::move(target.buffers[metadata_buffer_ind])});
  }

  target.subgraphs.clear();
  target.buffers.clear();
  target.metadata.clear();
  target.metadata_buffer.clear();

  target.buffers.push_back(std::make_unique<tflite::BufferT>());

  for (auto& subgraph : model->subgraphs) {
    target.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
    LITERT_RETURN_STATUS_IF_NOT_OK(
        repacker.SerializeSubgraph(&subgraph, *target.subgraphs.back()));
  }

  for (auto& [name, buf] : metadata) {
    const auto new_ind = target.buffers.size();
    auto new_metadata = std::make_unique<tflite::MetadataT>();
    new_metadata->name = name;
    new_metadata->buffer = new_ind;
    target.metadata.emplace_back(std::move(new_metadata));
    target.metadata_buffer.push_back(new_ind);
    target.buffers.emplace_back(std::move(buf));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus AppendMetadata(LiteRtModel model, const void* metadata,
                            size_t metadata_size, const char* metadata_name) {
  const auto metadata_buffer_ind = model->flatbuffer_model->buffers.size();

  auto& metadata_buffer = model->flatbuffer_model->buffers.emplace_back(
      std::make_unique<tflite::BufferT>());
  auto raw_metadata = reinterpret_cast<const uint8_t*>(metadata);
  metadata_buffer->data.assign(raw_metadata, raw_metadata + metadata_size);
  model->flatbuffer_model->metadata_buffer.push_back(metadata_buffer_ind);

  auto& fb_metadata = model->flatbuffer_model->metadata.emplace_back(
      std::make_unique<tflite::MetadataT>());
  fb_metadata->name.assign(metadata_name);
  fb_metadata->buffer = metadata_buffer_ind;

  return kLiteRtStatusOk;
}

LiteRtStatus SerializeModel(LiteRtModel model, uint8_t** buf, size_t* size,
                            size_t* offset) {
  // Destroy model before return.
  UniqueLiteRtModel u_model(model);

  LITERT_RETURN_STATUS_IF_NOT_OK_MSG(ModelRepacker::Repack(model),
                                     "Failed to repack model.");

  flatbuffers::FlatBufferBuilder b;
  auto model_offset = tflite::Model::Pack(b, model->flatbuffer_model.get());
  tflite::FinishModelBuffer(b, model_offset);

  OwningBufferRef<uint8_t> buffer;
  auto [new_buf, new_size, new_offset] = buffer.GetWeak();
  new_buf = b.ReleaseRaw(new_size, new_offset);

  LITERT_ENSURE(VerifyFlatbuffer(buffer.Span()),
                kLiteRtStatusErrorInvalidFlatbuffer,
                "Failed to verify flatbuffer");

  std::tie(*buf, *size, *offset) = buffer.Release();

  return kLiteRtStatusOk;
}

// NOLINTEND
