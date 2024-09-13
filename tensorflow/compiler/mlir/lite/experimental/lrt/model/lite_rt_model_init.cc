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
#include <utility>
#include <vector>

#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/compiler/mlir/lite/core/model_builder_base.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/api/lite_rt_model_api.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_common.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_support.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/model/lite_rt_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/model/lite_rt_model_init.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"

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
    return StatusCreate(kLrtStatusFlatbufferFailedVerify);
  }
  return StatusOk();
}

LrtStatus IsOpSupported(const tflite::OperatorT& op) {
  // TODO: b/365299994 - Check for supported options.

  if (!op.custom_options.empty()) {
    // TODO: b/365299994 - Support custom options.
    _LRT_D_MSG("Custom options not supported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  if (!op.intermediates.empty()) {
    // TODO: b/365299994 - Support intermediates.
    _LRT_D_MSG("Intermediate tensors not supported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  if (op.large_custom_options_size != 0) {
    // TODO: b/365299994 - Support large custom options.
    _LRT_D_MSG("Large custom options not supported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  for (auto m_input : op.mutating_variable_inputs) {
    if (m_input) {
      // TODO: b/365299994 - Support mutating variable inputs.
      _LRT_D_MSG("Mutating variable inputs not supported.");
      return StatusCreate(kLrtStatusErrorUnsupported);
    }
  }

  return StatusOk();
}

LrtStatus IsBufferSupported(const tflite::BufferT& buffer) {
  if (buffer.offset != 0) {
    // TODO: b/365299994 - Support buffer with offset.
    _LRT_D_MSG("Buffers with offset not supported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  return StatusOk();
}

LrtStatus IsTensorSupported(const tflite::TensorT& tensor) {
  if (!tensor.has_rank) {
    // TODO: b/365299994 - Support unranked tensors.
    _LRT_D_MSG("Unranked tensors not supported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  if (tensor.is_variable) {
    // TODO: b/365299994 - Support variable tensors.
    _LRT_D_MSG("Variable tensors not supported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  if (!tensor.variant_tensors.empty()) {
    // TODO: b/365299994 - Support variant tensors.
    _LRT_D_MSG("Variant tensors not supported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  if (!tensor.shape_signature.empty()) {
    // TODO: b/365299994 - Support shape signature.
    _LRT_D_MSG("Shape signature not supported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  if (tensor.sparsity) {
    // TODO: b/365299994 - Support sparsity tensors.
    _LRT_D_MSG("Sparsity tensors not supported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  if (tensor.type != tflite::TensorType_FLOAT32) {
    // TODO: b/365299994 - Support all element types.
    _LRT_D_MSG("Only f32 supported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  return StatusOk();
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
    target->buffer.fb_buffer = GetBuffer(buffer_ind);
    LRT_RETURN_STATUS_IF_NOT_OK(IsBufferSupported(*target->buffer.fb_buffer));
  }

  target->type_id = kLrtRankedTensorType;

  auto& ranked_tensor = target->type_detail.ranked_tensor_type;

  ranked_tensor.layout.dimensions = tensor.shape.data();
  ranked_tensor.layout.rank = tensor.shape.size();

  ranked_tensor.element_type = kLrtElementTypeFloat32;

  return StatusOk();
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

  return StatusOk();
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

  return StatusOk();
}

LrtStatus ModelUnpacker::Unpack(LrtModel model) {
  ModelUnpacker unpacker(model);

  if (unpacker.Fb().subgraphs.size() != 1) {
    // TODO: b/365299994 - Support multi subgraph.
    _LRT_D_MSG("Only single subgraph models suported.");
    return StatusCreate(kLrtStatusErrorUnsupported);
  }

  auto& subgraph = model->subgraphs.emplace_back();
  subgraph.flatbuffer_subgraph = std::move(unpacker.Fb().subgraphs[0]);
  LRT_RETURN_STATUS_IF_NOT_OK(unpacker.UnpackSubgraph(&subgraph));

  return StatusOk();
}

LrtStatus LoadModel(std::unique_ptr<tflite::ModelT> flatbuffer,
                    LrtModel* model) {
  auto lrt_model = std::make_unique<LrtModelT>();
  lrt_model->flatbuffer_model = std::move(flatbuffer);

  LRT_RETURN_STATUS_IF_NOT_OK(ModelUnpacker::Unpack(lrt_model.get()));

  lrt_model->flatbuffer_model->buffers.clear();
  lrt_model->flatbuffer_model->subgraphs.clear();

  *model = lrt_model.release();

  return StatusOk();
}

LrtStatus LoadModel(const uint8_t* buf, size_t buf_size, LrtModel* model) {
  LRT_RETURN_STATUS_IF_NOT_OK(VerifyFlatbuffer(buf, buf_size));
  return LoadModel(tflite::UnPackModel(buf), model);
}

LrtStatus LoadModelFromFile(const char* path, LrtModel* model) {
  std::unique_ptr<tflite::Allocation> alloc =
      tflite::GetAllocationFromFile(path, tflite::DefaultErrorReporter());
  if (!alloc->valid()) {
    return StatusCreate(kLrtStatusBadFileOp);
  }

  return LoadModel(reinterpret_cast<const uint8_t*>(alloc->base()),
                   alloc->bytes(), model);
}

void ModelDestroy(LrtModel model) { delete model; }
