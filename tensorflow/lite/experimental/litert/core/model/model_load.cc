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

#include "tensorflow/lite/experimental/litert/core/model/model_load.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/compiler/mlir/lite/core/model_builder_base.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_util.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"

using ::litert::internal::VerifyFlatbuffer;

namespace litert::internal {
namespace {

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

  LiteRtOpCode LiteRtGetOpCode(uint32_t ind) {
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

  ranked_tensor.element_type = MapElementType(tensor.type);
  ranked_tensor.layout.rank = tensor.shape.size();
  ranked_tensor.layout.dimensions = tensor.shape.data();
  ranked_tensor.layout.strides =
      nullptr;  // TFL tensors don't support strides yet.

  target->name = tensor.name;

  return kLiteRtStatusOk;
}

LiteRtStatus ModelUnpacker::ConvertOp(const tflite::OperatorT& op,
                                      std::vector<LiteRtTensor>& tensors,
                                      LiteRtOp target) {
  target->op_code = LiteRtGetOpCode(op.opcode_index);

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

  target->custom_options = OwningBufferRef<uint8_t>(op.custom_options.data(),
                                                    op.custom_options.size());

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
    LITERT_LOG(LITERT_ERROR, "%s",
               "Only models with 1 subgraph current supported\n");
    return kLiteRtStatusErrorUnsupported;
  }

  auto& subgraph = model->subgraphs.emplace_back();
  subgraph.flatbuffer_subgraph = std::move(unpacker.Fb().subgraphs[0]);
  LITERT_RETURN_STATUS_IF_NOT_OK(unpacker.UnpackSubgraph(&subgraph));

  return kLiteRtStatusOk;
}

LiteRtStatus LoadModelFromFlatbuffer(std::unique_ptr<tflite::ModelT> flatbuffer,
                                     LiteRtModel* model) {
  auto litert_model = std::make_unique<LiteRtModelT>();
  litert_model->flatbuffer_model = std::move(flatbuffer);
  litert_model->subgraphs.reserve(100);

  LITERT_RETURN_STATUS_IF_NOT_OK(ModelUnpacker::Unpack(litert_model.get()));

  litert_model->flatbuffer_model->subgraphs.clear();

  *model = litert_model.release();

  return kLiteRtStatusOk;
}

}  // namespace

Expected<Model> LoadModelFromMemory(BufferRef<uint8_t> serialized) {
  LiteRtModel model;
  LITERT_EXPECT_OK(
      LiteRtLoadModelFromMemory(serialized.Data(), serialized.Size(), &model));
  return Model::CreateFromOwnedHandle(model);
}

Expected<Model> LoadModelFromFile(absl::string_view path) {
  LiteRtModel model;
  LITERT_EXPECT_OK(LiteRtLoadModelFromFile(path.data(), &model));
  return Model::CreateFromOwnedHandle(model);
}

}  // namespace litert::internal

LiteRtStatus LiteRtLoadModelFromMemory(const uint8_t* buf, size_t buf_size,
                                       LiteRtModel* model) {
  LITERT_ENSURE(VerifyFlatbuffer(buf, buf_size),
                kLiteRtStatusErrorInvalidFlatbuffer,
                "Failed to verify flatbuffer");
  return litert::internal::LoadModelFromFlatbuffer(tflite::UnPackModel(buf),
                                                   model);
}

LiteRtStatus LiteRtLoadModelFromFile(const char* path, LiteRtModel* model) {
  std::unique_ptr<tflite::Allocation> alloc =
      tflite::GetAllocationFromFile(path, tflite::DefaultErrorReporter());
  if (!alloc->valid()) {
    return kLiteRtStatusErrorFileIO;
  }
  return LiteRtLoadModelFromMemory(
      reinterpret_cast<const uint8_t*>(alloc->base()), alloc->bytes(), model);
}
