/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_TFLITE_TFL_TENSOR_REF_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_TFLITE_TFL_TENSOR_REF_H_

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

namespace ml_adj::data {

// Immutable wrapper of `TfLiteTensor`.
class TflTensorRef : public DataRef {
 public:
  explicit TflTensorRef(const TfLiteTensor* tfl_tensor);

  // TODO(b/290283768) Implement copy and move semantics.
  TflTensorRef(const TflTensorRef&) = delete;
  TflTensorRef(TflTensorRef&&) = delete;
  TflTensorRef& operator=(const TflTensorRef&) = delete;
  TflTensorRef& operator=(TflTensorRef&&) = delete;

  const void* Data() const override;

  ind_t NumElements() const override;

  size_t Bytes() const override;

 private:
  const TfLiteTensor* const tfl_tensor_;
};

// Mutable wrapper of `TfLiteTensor`.
class MutableTflTensorRef : public MutableDataRef {
 public:
  MutableTflTensorRef(TfLiteTensor* tfl_tensor, TfLiteContext* tfl_ctx);

  // TODO(b/290283768) Implement copy and move semantics.
  MutableTflTensorRef(const MutableTflTensorRef&) = delete;
  MutableTflTensorRef(MutableTflTensorRef&&) = delete;
  MutableTflTensorRef& operator=(const MutableTflTensorRef&) = delete;
  MutableTflTensorRef& operator=(MutableTflTensorRef&&) = delete;

  const void* Data() const override;

  ind_t NumElements() const override;

  size_t Bytes() const override;

  void Resize(dims_t&& dims) override;

  void* Data() override;

 private:
  TfLiteTensor* const tfl_tensor_;
  TfLiteContext* tfl_ctx_;
};

}  // namespace ml_adj::data

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_TFLITE_TFL_TENSOR_REF_H_
