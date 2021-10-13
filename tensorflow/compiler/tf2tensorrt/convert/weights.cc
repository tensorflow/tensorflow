/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/tf2tensorrt/convert/weights.h"

#include <functional>
#include <numeric>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

namespace convert {

TRT_ShapedWeights::TRT_ShapedWeights(nvinfer1::DataType type) : type_(type) {
  shape_.nbDims = 0;
  shape_.d[0] = 0;
}

TRT_ShapedWeights::TRT_ShapedWeights(nvinfer1::DataType type,
                                     nvinfer1::Dims dims, Tensor tensor)
    : shape_(dims), type_(type), tensor_(tensor) {
  if (dims.nbDims == 0) {
    DCHECK(dims.d[0] == 0 || dims.d[0] == 1);
  }
}

TRT_ShapedWeights::TRT_ShapedWeights(const TRT_ShapedWeights& rhs)
    : shape_(rhs.shape_), type_(rhs.type_), tensor_(rhs.tensor_) {}

int64_t TRT_ShapedWeights::count(nvinfer1::Dims dims) {
  if (dims.nbDims == 0) {
    assert(dims.d[0] == 0 || dims.d[0] == 1);
    return dims.d[0];
  }
  return static_cast<int64_t>(
      std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>()));
}

nvinfer1::Weights TRT_ShapedWeights::GetTrtWeights() const {
  return nvinfer1::Weights{type_, GetPointer<int8>(), count()};
}

Status TRT_ShapedWeights::SetShape(nvinfer1::Dims dims) {
  if (this->count() != TRT_ShapedWeights::count(dims)) {
    VLOG(2) << "Changing shape from "
            << tensorflow::tensorrt::DebugString(shape_) << ", to "
            << tensorflow::tensorrt::DebugString(dims);
    return errors::Internal("SetShape would change number of elements");
  }
  shape_ = dims;
  return Status::OK();
}

size_t TRT_ShapedWeights::size_bytes() const {
  size_t data_type_size = -1;
  switch (type_) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      data_type_size = 4;
      break;
    case nvinfer1::DataType::kHALF:
      data_type_size = 2;
      break;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
      data_type_size = 1;
      break;
  }
  return this->count() * data_type_size;
}

string TRT_ShapedWeights::DebugString() const {
  return absl::StrCat(
      "TRT_ShapedWeights(shape=", tensorflow::tensorrt::DebugString(shape_),
      ", type=", tensorflow::tensorrt::DebugString(type_),
      ", values=", reinterpret_cast<uintptr_t>(GetPointer<int8>()), ")");
}

TRT_TensorOrWeights::TRT_TensorOrWeights(ITensorProxyPtr tensor)
    : tensor_proxy_ptr_(tensor), initialized_(true), is_tensor_(true) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(ITensorProxyPtr tensor, int batch_size)
    : tensor_proxy_ptr_(tensor),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(nvinfer1::ITensor* tensor,
                                         int batch_size)
    : tensor_proxy_ptr_(tensor),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(nvinfer1::DataType trt_dtype,
                                         const nvinfer1::Dims& trt_dims,
                                         int batch_size)
    : tensor_proxy_ptr_(new SimpleITensor(trt_dtype, trt_dims)),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(const TRT_ShapedWeights& weights)
    : weights_(weights), initialized_(true), is_tensor_(false) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(const TRT_TensorOrWeights& rhs)
    : tensor_proxy_ptr_(rhs.tensor_proxy_ptr_),
      batch_size_(rhs.batch_size_),
      weights_(rhs.weights_),
      initialized_(rhs.initialized_),
      is_tensor_(rhs.is_tensor_) {}

void TRT_TensorOrWeights::operator=(const TRT_TensorOrWeights& rhs) {
  tensor_proxy_ptr_ = rhs.tensor_proxy_ptr_;
  batch_size_ = rhs.batch_size_;
  weights_ = rhs.weights_;
  initialized_ = rhs.initialized_;
  is_tensor_ = rhs.is_tensor_;
}

ITensorProxyPtr TRT_TensorOrWeights::tensor() const {
  DCHECK(is_tensor());
  return tensor_proxy_ptr_;
}

nvinfer1::Dims TRT_TensorOrWeights::GetTrtDims() const {
  if (is_tensor()) {
    return tensor()->getDimensions();
  } else {
    return weights().shape_;
  }
}

Status TRT_TensorOrWeights::GetTfType(DataType* tf_type) const {
  if (is_tensor()) {
    nvinfer1::DataType trt_type = tensor()->getType();
    return TrtTypeToTfType(trt_type, tf_type);
  }
  if (is_weights()) {
    *tf_type = weights().GetTensor().dtype();
    return Status::OK();
  }
  return errors::Internal("The object is probably not initialized");
}

string TRT_TensorOrWeights::DebugString() const {
  string output = "TRT_TensorOrWeights(type=";
  if (is_tensor()) {
    absl::StrAppend(&output,
                    "tensor=", tensorflow::tensorrt::DebugString(tensor()),
                    ", batch_size=", batch_size_);
  } else {
    absl::StrAppend(&output, "weights=", weights_.DebugString());
  }
  absl::StrAppend(&output, ")");
  return output;
}

TRT_ShapedWeights TrtWeightStore::GetTempWeights(nvinfer1::DataType trt_dtype,
                                                 const nvinfer1::Dims& dims) {
  TensorShape shape;
  DataType tf_dtype;
  // TODO(laigd): make it return a status.
  TF_CHECK_OK(TensorShapeUtils::MakeShape(dims.d, dims.nbDims, &shape));
  TF_CHECK_OK(TrtTypeToTfType(trt_dtype, &tf_dtype));
  // TODO(jie): check weights size_bytes. 0 means type error
  Tensor tensor(tf_dtype, shape);
  TRT_ShapedWeights weights(trt_dtype, dims, tensor);
  store_.emplace_back(std::move(tensor));
  return weights;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
