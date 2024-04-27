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

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/ragged_tensor_variant.h"

namespace tensorflow {

string RaggedTensorVariant::TypeName() const { return "RaggedTensorVariant"; }

string RaggedTensorVariant::DebugString() const {
  return absl::StrCat(
      "RaggedTensorVariant(dtype=", DataTypeString(values_.dtype()),
      ", ragged_rank=", nested_splits_.size(), ", splits_dtype=",
      DataTypeString(nested_splits_.empty() ? DT_INVALID
                                            : nested_splits_.back().dtype()));
}

void RaggedTensorVariant::Encode(VariantTensorData* data) const {
  data->set_type_name(TypeName());
  for (const auto& splits : nested_splits_) {
    *data->add_tensors() = splits;
  }
  *data->add_tensors() = values_;
}

bool RaggedTensorVariant::Decode(const VariantTensorData& data) {
  if (data.tensors_size() < 1) {
    return false;
  }
  nested_splits_.assign(data.tensors().begin(),
                        std::prev(data.tensors().end()));
  values_ = data.tensors().back();
  return true;
}

namespace {

Status RaggedTensorVariantDeviceCopy(
    const RaggedTensorVariant& from, RaggedTensorVariant* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  TF_RETURN_IF_ERROR(copy(from.values(), to->mutable_values()));
  // TODO(b/170415165) Should we use `copy` to move splits from device<->host?
  *to->mutable_nested_splits() = from.nested_splits();
  return absl::OkStatus();
}

}  // namespace

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(
    ZEROS_LIKE_VARIANT_UNARY_OP, DEVICE_CPU, RaggedTensorVariant,
    RaggedTensorVariantZerosLike<CPUDevice>);

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(
    ADD_VARIANT_BINARY_OP, DEVICE_CPU, RaggedTensorVariant,
    RaggedTensorVariantBinaryAdd<CPUDevice>);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(RaggedTensorVariant,
                                       "RaggedTensorVariant");

#define REGISTER_RAGGED_TENSOR_VARIANT_COPY(DIRECTION)  \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      RaggedTensorVariant, DIRECTION, RaggedTensorVariantDeviceCopy)

REGISTER_RAGGED_TENSOR_VARIANT_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_RAGGED_TENSOR_VARIANT_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_RAGGED_TENSOR_VARIANT_COPY(
    VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

}  // namespace tensorflow
