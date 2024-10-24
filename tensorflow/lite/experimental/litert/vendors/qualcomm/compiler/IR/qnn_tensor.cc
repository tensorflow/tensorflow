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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"

#include <memory>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_support.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor.h"

namespace litert::qnn {

using ::litert::LiteRtTensorManager;

namespace {

LiteRtStatus LegalizeElementType(LiteRtElementType src, Qnn_DataType_t& dest) {
  switch (src) {
    case kLiteRtElementTypeFloat32:
      dest = QNN_DATATYPE_FLOAT_32;
      return kLiteRtStatusOk;
    default:
      return kLiteRtStatusErrorUnsupported;
      // TODO: Finish legalizing datatypes.
  }
}

LiteRtStatus LegalizeShapeInfo(const LiteRtTensorManager& src,
                               Qnn_Tensor_t& dest) {
  LITERT_ENSURE_SUPPORTED(!src.HasStrides(), "Strides not yet supported");

  dest.v2.rank = src.Rank();
  dest.v2.dimensions = new uint32_t[dest.v2.rank];
  for (int i = 0; i < dest.v2.rank; ++i) {
    const auto src_dim = src.Dims()[i];
    LITERT_ENSURE(src_dim >= 1, kLiteRtStatusErrorInvalidArgument,
                  "Cannot pass dim < 1 to QNN Tensor.");

    dest.v2.dimensions[i] = src.Dims()[i];
  }
  return kLiteRtStatusOk;
}

void FreeTensorDims(Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2 &&
      tensor.v2.dimensions != nullptr) {
    delete[] tensor.v2.dimensions;
    tensor.v2.dimensions = nullptr;
    tensor.v2.rank = 0;
  }
}

}  // namespace

void SetInputTensorAttrs(Qnn_Tensor_t& tensor) {
  ABSL_DCHECK(tensor.version == QNN_TENSOR_VERSION_2);
  tensor.v2.type = QNN_TENSOR_TYPE_APP_WRITE;
  tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
  tensor.v2.clientBuf = QNN_CLIENT_BUFFER_INIT;
}

void SetOutputTensorAttrs(Qnn_Tensor_t& tensor) {
  ABSL_DCHECK(tensor.version == QNN_TENSOR_VERSION_2);
  tensor.v2.type = QNN_TENSOR_TYPE_APP_READ;
}

void ResetTensor(Qnn_Tensor_t& tensor) {
  FreeTensorDims(tensor);
  tensor = QNN_TENSOR_INIT;
  tensor.version = QNN_TENSOR_VERSION_2;
  tensor.v2 = QNN_TENSOR_V2_INIT;
  tensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
}

Qnn_Tensor_t BuildDefaultTensor(uint32_t id) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  ResetTensor(tensor);
  tensor.v2.id = id;
  return tensor;
}

Qnn_Tensor_t BuildDefaultTensor() { return BuildDefaultTensor(0); }

Qnn_Tensor_t BuildInputTensor() {
  auto tensor = BuildDefaultTensor();
  SetInputTensorAttrs(tensor);
  return tensor;
}

Qnn_Tensor_t BuildOutputTensor() {
  Qnn_Tensor_t tensor = BuildDefaultTensor();
  SetOutputTensorAttrs(tensor);
  return tensor;
}

uint32_t MoveToId(Qnn_Tensor_t& tensor) {
  const auto id = tensor.v2.id;
  ResetTensor(tensor);
  tensor.v2.id = id;
  return id;
}

LiteRtStatus LegalizeTensor(LiteRtTensor src, Qnn_Tensor_t& dest) {
  ResetTensor(dest);

  LiteRtTensorManager::Unique src_tensor;
  LITERT_RETURN_STATUS_IF_NOT_OK(
      LiteRtTensorManager::MakeFromTensor(src, src_tensor));

  LITERT_RETURN_STATUS_IF_NOT_OK(
      LegalizeElementType(src_tensor->ElementType(), dest.v2.dataType));

  LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeShapeInfo(*src_tensor, dest));

  const bool is_subgraph_in = src_tensor->IsSubgraphInput();
  const bool is_subgraph_out = src_tensor->IsSubgraphOutput();

  LITERT_ENSURE(!(is_subgraph_in && is_subgraph_out),
                kLiteRtStatusErrorInvalidArgument,
                "Malformed tensor, cannot be both subgraph in and out.");

  if (is_subgraph_in) {
    SetInputTensorAttrs(dest);
  }
  if (is_subgraph_out) {
    SetOutputTensorAttrs(dest);
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
