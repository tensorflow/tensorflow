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
#endif

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/core/model_builder_base.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/lite_rt_model_init.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"

// NOLINTBEGIN
void SetFbVerifyOptions(flatbuffers::Verifier::Options& opts) {
#ifndef NDEBUG
  opts.assert = true;
#endif
}

LrtStatus VerifyFlatbuffer(const uint8_t* buf, size_t buf_size) {
  // TODO: b/365299994 - If buffer verification is slow, run only in debug.
  // Also check file size.
  flatbuffers::Verifier::Options options;
  SetFbVerifyOptions(options);
  flatbuffers::Verifier verifier(buf, buf_size, options);
  if (!tflite::VerifyModelBuffer(verifier)) {
    _LRT_D_MSG("Failed to verify fb");
    return kLrtStatusFlatbufferFailedVerify;
  }
  return kLrtStatusOk;
}

LrtStatus IsOpSupported(const tflite::OperatorT& op) {
  // TODO: b/365299994 - Check for supported options.

  if (!op.custom_options.empty()) {
    // TODO: b/365299994 - Support custom options.
    _LRT_D_MSG("Custom options not supported.");
    return kLrtStatusErrorUnsupported;
  }

  if (!op.intermediates.empty()) {
    // TODO: b/365299994 - Support intermediates.
    _LRT_D_MSG("Intermediate tensors not supported.");
    return kLrtStatusErrorUnsupported;
  }

  if (op.large_custom_options_size != 0) {
    // TODO: b/365299994 - Support large custom options.
    _LRT_D_MSG("Large custom options not supported.");
    return kLrtStatusErrorUnsupported;
  }

  for (auto m_input : op.mutating_variable_inputs) {
    if (m_input) {
      // TODO: b/365299994 - Support mutating variable inputs.
      _LRT_D_MSG("Mutating variable inputs not supported.");
      return kLrtStatusErrorUnsupported;
    }
  }

  return kLrtStatusOk;
}

LrtStatus IsBufferSupported(const tflite::BufferT& buffer) {
  if (buffer.offset != 0) {
    // TODO: b/365299994 - Support buffer with offset.
    _LRT_D_MSG("Buffers with offset not supported.");
    return kLrtStatusErrorUnsupported;
  }

  return kLrtStatusOk;
}

LrtStatus IsTensorSupported(const tflite::TensorT& tensor) {
  if (!tensor.has_rank) {
    // TODO: b/365299994 - Support unranked tensors.
    _LRT_D_MSG("Unranked tensors not supported.");
    return kLrtStatusErrorUnsupported;
  }

  if (tensor.is_variable) {
    // TODO: b/365299994 - Support variable tensors.
    _LRT_D_MSG("Variable tensors not supported.");
    return kLrtStatusErrorUnsupported;
  }

  if (!tensor.variant_tensors.empty()) {
    // TODO: b/365299994 - Support variant tensors.
    _LRT_D_MSG("Variant tensors not supported.");
    return kLrtStatusErrorUnsupported;
  }

  if (!tensor.shape_signature.empty()) {
    // TODO: b/365299994 - Support shape signature.
    _LRT_D_MSG("Shape signature not supported.");
    return kLrtStatusErrorUnsupported;
  }

  if (tensor.sparsity) {
    // TODO: b/365299994 - Support sparsity tensors.
    _LRT_D_MSG("Sparsity tensors not supported.");
    return kLrtStatusErrorUnsupported;
  }

  if (tensor.type != tflite::TensorType_FLOAT32) {
    // TODO: b/365299994 - Support all element types.
    _LRT_D_MSG("Only f32 supported.");
    return kLrtStatusErrorUnsupported;
  }

  return kLrtStatusOk;
}

LrtStatus SetDefaultOptions(tflite::BuiltinOptionsUnion& opts, LrtOpCode code) {
  switch (code) {
    case kLrtOpCodeTflMul:
      opts.Set(tflite::MulOptionsT());
      break;
    case kLrtOpCodeTflAdd:
      opts.Set(tflite::AddOptionsT());
      break;
    case kLrtOpCodeTflCustom:
      return kLrtStatusOk;
    default:
      return kLrtStatusErrorUnsupported;
  }
  return kLrtStatusOk;
}

void SetCustomOptions(tflite::OperatorT& op, std::string_view options_data) {
  const uint8_t* data = reinterpret_cast<const uint8_t*>(options_data.data());
  op.custom_options.assign(data, data + options_data.size());
  op.custom_options_format = tflite::CustomOptionsFormat_FLEXBUFFERS;
}

//===----------------------------------------------------------------------===//
//                               Load                                         //
//===----------------------------------------------------------------------===//

class ModelUnpacker {
 public:
  static LrtStatus Unpack(LrtModel model);

 private:
  explicit ModelUnpacker(LrtModel model) : model_(model) {}

  LrtStatus ConvertTensor(const tflite::TensorT& tensor, LrtTensor target);

  LrtStatus ConvertOp(const tflite::OperatorT& op,
                      std::vector<LrtTensor>& tensors, LrtOp target);

  LrtStatus UnpackSubgraph(LrtSubgraph target);

  LrtOpCode GetOpCode(uint32_t ind) {
    return static_cast<LrtOpCode>(Fb().operator_codes[ind]->builtin_code);
  }

  std::unique_ptr<tflite::BufferT> GetBuffer(uint32_t ind) {
    return std::move(Fb().buffers[ind]);
  }

  tflite::ModelT& Fb() { return *model_->flatbuffer_model; }

  LrtModel model_;
};

LrtStatus ModelUnpacker::ConvertTensor(const tflite::TensorT& tensor,
                                       LrtTensor target) {
  LRT_RETURN_STATUS_IF_NOT_OK(IsTensorSupported(tensor));

  const auto buffer_ind = tensor.buffer;

  if (buffer_ind != 0) {
    target->weights.fb_buffer = GetBuffer(buffer_ind);
    LRT_RETURN_STATUS_IF_NOT_OK(IsBufferSupported(*target->weights.fb_buffer));
  }

  target->type_id = kLrtRankedTensorType;

  auto& ranked_tensor = target->type_detail.ranked_tensor_type;

  ranked_tensor.element_type = kLrtElementTypeFloat32;
  ranked_tensor.layout.rank = tensor.shape.size();
  ranked_tensor.layout.dimensions = tensor.shape.data();
  ranked_tensor.layout.strides =
      nullptr;  // TFL tensors don't support strides yet.

  return kLrtStatusOk;
}

LrtStatus ModelUnpacker::ConvertOp(const tflite::OperatorT& op,
                                   std::vector<LrtTensor>& tensors,
                                   LrtOp target) {
  target->op_code = GetOpCode(op.opcode_index);

  for (auto input : op.inputs) {
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

  return kLrtStatusOk;
}

LrtStatus ModelUnpacker::UnpackSubgraph(LrtSubgraph target) {
  auto& subgraph = target->flatbuffer_subgraph;

  for (int i = 0; i < subgraph->tensors.size(); ++i) {
    auto& flatbuffer_tensor = *subgraph->tensors[i];
    LRT_RETURN_STATUS_IF_NOT_OK(IsTensorSupported(flatbuffer_tensor));

    auto& tensor = target->tensors_storage.emplace_back();
    target->tensors.push_back(&tensor);

    LRT_RETURN_STATUS_IF_NOT_OK(ConvertTensor(flatbuffer_tensor, &tensor));
  }

  for (int i = 0; i < subgraph->operators.size(); ++i) {
    auto& flatbuffer_op = *subgraph->operators[i];

    auto& op = target->ops_storage.emplace_back();
    target->ops.push_back(&op);

    LRT_RETURN_STATUS_IF_NOT_OK(ConvertOp(flatbuffer_op, target->tensors, &op));
  }

  for (auto input : subgraph->inputs) {
    target->inputs.push_back(target->tensors[input]);
  }

  for (auto output : subgraph->outputs) {
    target->outputs.push_back(target->tensors[output]);
  }

  return kLrtStatusOk;
}

LrtStatus ModelUnpacker::Unpack(LrtModel model) {
  ModelUnpacker unpacker(model);

  if (unpacker.Fb().subgraphs.size() != 1) {
    // TODO: b/365299994 - Support multi subgraph.
    _LRT_D_MSG("Only single subgraph models suported.");
    return kLrtStatusErrorUnsupported;
  }

  auto& subgraph = model->subgraphs.emplace_back();
  subgraph.flatbuffer_subgraph = std::move(unpacker.Fb().subgraphs[0]);
  LRT_RETURN_STATUS_IF_NOT_OK(unpacker.UnpackSubgraph(&subgraph));

  return kLrtStatusOk;
}

LrtStatus RegisterCustomOpCode(LrtModel model, const char* new_op_code) {
  model->custom_op_code.assign(new_op_code);
  return kLrtStatusOk;
}

LrtStatus LoadModel(std::unique_ptr<tflite::ModelT> flatbuffer,
                    LrtModel* model) {
  auto lrt_model = std::make_unique<LrtModelT>();
  lrt_model->flatbuffer_model = std::move(flatbuffer);
  lrt_model->subgraphs.reserve(100);

  LRT_RETURN_STATUS_IF_NOT_OK(ModelUnpacker::Unpack(lrt_model.get()));

  lrt_model->flatbuffer_model->subgraphs.clear();

  // Set as empty string in case its not set explictly.
  LRT_RETURN_STATUS_IF_NOT_OK(RegisterCustomOpCode(lrt_model.get(), ""));

  *model = lrt_model.release();

  return kLrtStatusOk;
}

LrtStatus LoadModel(const uint8_t* buf, size_t buf_size, LrtModel* model) {
  LRT_RETURN_STATUS_IF_NOT_OK(VerifyFlatbuffer(buf, buf_size));
  return LoadModel(tflite::UnPackModel(buf), model);
}

LrtStatus LoadModelFromFile(const char* path, LrtModel* model) {
  std::unique_ptr<tflite::Allocation> alloc =
      tflite::GetAllocationFromFile(path, tflite::DefaultErrorReporter());
  if (!alloc->valid()) {
    return kLrtStatusBadFileOp;
  }

  return LoadModel(reinterpret_cast<const uint8_t*>(alloc->base()),
                   alloc->bytes(), model);
}

void ModelDestroy(LrtModel model) { delete model; }

//===----------------------------------------------------------------------===//
//                                 Serialize                                  //
//===----------------------------------------------------------------------===//

class ModelRepacker {
 public:
  static LrtStatus Repack(LrtModel model);

 private:
  static void BuildOpCodeMap(LrtModel model,
                             std::unordered_map<LrtOpCode, uint32_t>& map);

  explicit ModelRepacker(LrtModel model) : model_(model) {
    BuildOpCodeMap(model_, op_code_map_);
  }

  LrtStatus SerializeTensor(LrtTensor tensor, tflite::TensorT& target);

  LrtStatus SerializeOp(
      LrtOp op, tflite::OperatorT& target,
      const std::unordered_map<LrtTensor, int32_t>& tensor_map);

  LrtStatus SerializeSubgraph(LrtSubgraph subgraph, tflite::SubGraphT& target);

  uint32_t SubmitBuffer(std::unique_ptr<tflite::BufferT> buffer) {
    OldFb().buffers.push_back(std::move(buffer));
    return OldFb().buffers.size() - 1;
  }

  tflite::ModelT& OldFb() { return *model_->flatbuffer_model; }

  LrtModel model_;
  std::unordered_map<LrtOpCode, uint32_t> op_code_map_;
};

void ModelRepacker::BuildOpCodeMap(
    LrtModel model, std::unordered_map<LrtOpCode, uint32_t>& map) {
  // Add the user set custom code to the flatbuffers known codes.
  auto& custom_code = model->flatbuffer_model->operator_codes.emplace_back(
      std::make_unique<tflite::OperatorCodeT>());
  custom_code->builtin_code = tflite::BuiltinOperator_CUSTOM;
  custom_code->custom_code = model->custom_op_code;
  custom_code->version = 1;

  auto& codes = model->flatbuffer_model->operator_codes;

  for (int i = 0; i < codes.size(); ++i) {
    const auto tfl_code = codes[i]->builtin_code;
    map.insert({static_cast<LrtOpCode>(tfl_code), i});
  }
}

LrtStatus ModelRepacker::SerializeTensor(LrtTensor tensor,
                                         tflite::TensorT& target) {
  target.has_rank = true;
  const auto& type = tensor->type_detail.ranked_tensor_type;
  // TODO: b/365299994 - Map lrt element types to flatbuffer elements types.
  target.type = tflite::TensorType_FLOAT32;

  for (int i = 0; i < type.layout.rank; ++i) {
    target.shape.push_back(type.layout.dimensions[i]);
  }

  // TFL tensors don't support strides yet.
  DCHECK(type.layout.strides == nullptr);

  DCHECK(tensor->weights.fb_buffer != nullptr) << "Submitting a null buffer";
  target.buffer = SubmitBuffer(std::move(tensor->weights.fb_buffer));

  return kLrtStatusOk;
}

LrtStatus ModelRepacker::SerializeOp(
    LrtOp op, tflite::OperatorT& target,
    const std::unordered_map<LrtTensor, int32_t>& tensor_map) {
  target.opcode_index = op_code_map_.at(op->op_code);

  for (auto in : op->inputs) {
    target.inputs.push_back(tensor_map.at(in));
  }

  for (auto out : op->outputs) {
    target.outputs.push_back(tensor_map.at(out));
  }

  // TODO: b/365299994 - Support options in serialize.
  LRT_RETURN_STATUS_IF_NOT_OK_MSG(
      SetDefaultOptions(target.builtin_options, op->op_code),
      "Failed serializing options");

  if (!op->custom_options.empty()) {
    SetCustomOptions(target, op->custom_options);
  }
  // TODO: b/365299994 - Support exotic op fields in serialize.

  return kLrtStatusOk;
}

LrtStatus ModelRepacker::SerializeSubgraph(LrtSubgraph subgraph,
                                           tflite::SubGraphT& target) {
  std::unordered_map<LrtTensor, int32_t> tensor_map;

  for (auto tensor : subgraph->tensors) {
    tensor_map.insert({tensor, tensor_map.size()});
    target.tensors.push_back(std::make_unique<tflite::TensorT>());
    LRT_RETURN_STATUS_IF_NOT_OK(
        SerializeTensor(tensor, *target.tensors.back()));
  }

  for (auto op : subgraph->ops) {
    target.operators.push_back(std::make_unique<tflite::OperatorT>());
    LRT_RETURN_STATUS_IF_NOT_OK(
        SerializeOp(op, *target.operators.back(), tensor_map));
  }

  for (auto in : subgraph->inputs) {
    target.inputs.push_back(tensor_map.at(in));
  }
  for (auto out : subgraph->outputs) {
    target.outputs.push_back(tensor_map.at(out));
  }

  return kLrtStatusOk;
}

LrtStatus ModelRepacker::Repack(LrtModel model) {
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
    LRT_RETURN_STATUS_IF_NOT_OK(
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

  return kLrtStatusOk;
}

LrtStatus AppendMetadata(LrtModel model, const void* metadata,
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

  return kLrtStatusOk;
}

LrtStatus SerializeModel(LrtModel model, uint8_t** buf, size_t* size,
                         size_t* offset) {
  // Destroy model before return.
  UniqueLrtModel u_model(model);

  LRT_RETURN_STATUS_IF_NOT_OK_MSG(ModelRepacker::Repack(model),
                                  "Failed to repack model.");

  flatbuffers::FlatBufferBuilder b;
  auto model_offset = tflite::Model::Pack(b, model->flatbuffer_model.get());
  tflite::FinishModelBuffer(b, model_offset);

  size_t new_buf_size;
  size_t new_buf_offset;

  uint8_t* new_buf = b.ReleaseRaw(new_buf_size, new_buf_offset);

  LRT_RETURN_STATUS_IF_NOT_OK_MSG(
      VerifyFlatbuffer(new_buf + new_buf_offset, new_buf_size - new_buf_offset),
      "Failed to verify flatbuffer");

  *buf = new_buf;
  *size = new_buf_size;
  *offset = new_buf_offset;

  return kLrtStatusOk;
}

// NOLINTEND
