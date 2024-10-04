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

#include "tensorflow/lite/experimental/lrt/qnn/IR/qnn_tensor.h"

#include <iostream>
#include <memory>

#include "absl/log/absl_check.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_tensor.h"

// NOTE: QNN Tensors must be created with a unique name. This will ensure
// uniqueness but will want to have more meaningful names in the future.
#define QNN_TENSOR_NAME absl::StrFormat("Tensor_%lu", __COUNTER__).c_str();

namespace qnn {

using ::lrt::LrtTensorManager;

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
  tensor = QNN_TENSOR_INIT;
  tensor.version = QNN_TENSOR_VERSION_2;
  tensor.v2 = QNN_TENSOR_V2_INIT;
  tensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
  if (tensor.v2.dimensions != nullptr) {
    delete[] tensor.v2.dimensions;
    tensor.v2.dimensions = nullptr;
  }
}

Qnn_Tensor_t BuildDefaultTensor() {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  ResetTensor(tensor);
  tensor.v2.name = QNN_TENSOR_NAME;
  return tensor;
}

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

LrtStatus LegalizeElementType(LrtElementType src, Qnn_DataType_t& dest) {
  switch (src) {
    case kLrtElementTypeFloat32:
      dest = QNN_DATATYPE_FLOAT_32;
      return kLrtStatusOk;
    default:
      return kLrtStatusErrorUnsupported;
      // TODO: Finish legalizing datatypes.
  }
}

LrtStatus LegalizeShapeInfo(const LrtTensorManager& src, Qnn_Tensor_t& dest) {
  dest.v2.rank = src.Rank();
  dest.v2.dimensions = new uint32_t[dest.v2.rank];
  for (int i = 0; i < dest.v2.rank; ++i) {
    const auto src_dim = src.Dims()[i];
    if (src_dim < 1) {
      std::cerr << "Cannot pass dim < 1 to QNN tensor.\n";
      return kLrtStatusErrorInvalidArgument;
    }
    dest.v2.dimensions[i] = src.Dims()[i];
  }
  return kLrtStatusOk;
}

LrtStatus LegalizeTensor(LrtTensor src, Qnn_Tensor_t& dest) {
  ResetTensor(dest);

  LrtTensorManager::Unique src_tensor;
  LRT_RETURN_STATUS_IF_NOT_OK(
      LrtTensorManager::MakeFromTensor(src, src_tensor));

  LRT_RETURN_STATUS_IF_NOT_OK(
      LegalizeElementType(src_tensor->ElementType(), dest.v2.dataType));

  LRT_RETURN_STATUS_IF_NOT_OK(LegalizeShapeInfo(*src_tensor, dest));

  return kLrtStatusOk;
}

}  // namespace qnn
