#include "tensorflow/core/framework/tensor_key.h"
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

#ifndef TENSORFLOW_CORE_KERNELS_RAGGED_TENSOR_VARIANT_H_
#define TENSORFLOW_CORE_KERNELS_RAGGED_TENSOR_VARIANT_H_

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/util/tensor_ops_util.h"

namespace tensorflow {

// Class used to store a RaggedTensor as a Variant scalar.
class RaggedTensorVariant {
 public:
  RaggedTensorVariant() {}
  RaggedTensorVariant(Tensor values, const std::vector<Tensor>& nested_splits)
      : values_(std::move(values)), nested_splits_(nested_splits) {}

  // Variant support methods.
  string TypeName() const;
  string DebugString() const;
  void Encode(VariantTensorData* data) const;
  bool Decode(const VariantTensorData& data);

  // The flat_values of the RaggedTensor.
  const Tensor& values() const { return values_; }
  Tensor* mutable_values() { return &values_; }
  void set_values(const Tensor& new_values) { values_ = new_values; }

  // The nested row_splits of the RaggedTensor.
  int ragged_rank() const { return nested_splits_.size(); }
  const std::vector<Tensor>& nested_splits() const { return nested_splits_; }
  std::vector<Tensor>* mutable_nested_splits() { return &nested_splits_; }
  const Tensor& splits(int i) const { return nested_splits_[i]; }
  Tensor* mutable_splits(int i) { return &nested_splits_[i]; }
  void set_nested_splits(const std::vector<Tensor>& nested_splits) {
    nested_splits_ = nested_splits;
  }
  void append_splits(const Tensor& splits) { nested_splits_.push_back(splits); }

 private:
  Tensor values_;
  std::vector<Tensor> nested_splits_;
};

template <typename Device>
absl::Status RaggedTensorVariantZerosLike(OpKernelContext* c,
                                          const RaggedTensorVariant& x,
                                          RaggedTensorVariant* y) {
  y->set_nested_splits(x.nested_splits());
  TF_RETURN_IF_ERROR(
      ZerosLikeTensor<Device>(c, x.values(), y->mutable_values()));
  return absl::OkStatus();
}

template <typename Device>
absl::Status RaggedTensorVariantBinaryAdd(OpKernelContext* c,
                                          const RaggedTensorVariant& x,
                                          const RaggedTensorVariant& y,
                                          RaggedTensorVariant* out) {
  if (x.values().dtype() != y.values().dtype()) {
    return errors::InvalidArgument(
        "Can't add RaggedTensorVariants of different dtypes. One is ",
        DataTypeString(x.values().dtype()), " and the other is ",
        DataTypeString(y.values().dtype()));
  }
  if (x.ragged_rank() != y.ragged_rank()) {
    return errors::InvalidArgument(
        "Can't add RaggedTensorVariants of different ragged rank. ", "One is ",
        x.ragged_rank(), " and the other is ", y.ragged_rank());
  }
  for (int i = 0; i < x.ragged_rank(); ++i) {
    if (TensorKey(x.splits(i)) != TensorKey(y.splits(i))) {
      return errors::InvalidArgument(
          "Can't add RaggedTensorVariants with different row_splits.");
    }
  }
  out->set_nested_splits(x.nested_splits());
  TF_RETURN_IF_ERROR(BinaryAddTensors<Device>(c, x.values(), y.values(),
                                              out->mutable_values()));
  return absl::OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RAGGED_TENSOR_VARIANT_H_
