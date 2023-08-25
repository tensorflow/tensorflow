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

TRT_ShapedWeights::TRT_ShapedWeights(nvinfer1::DataType type)
    : shape_(0, DimsAdapter::StorageType{}), type_(type), volume_(0) {}

StatusOr<TRT_ShapedWeights> TRT_ShapedWeights::CreateWithTensor(
    nvinfer1::DataType type, DimsAdapter dims, Tensor tensor) {
  TRT_ShapedWeights weights(type);
  weights.shape_ = dims;
  weights.tensor_ = std::forward<Tensor>(tensor);
  weights.volume_ = weights.shape_.Volume();
  if (weights.shape_.NumDims() == 0) {
    DCHECK(weights.shape_.IsEmpty() || weights.shape_.IsScalar());
  }
  return weights;
}

nvinfer1::Weights TRT_ShapedWeights::GetTrtWeights() const {
  return nvinfer1::Weights{type_, GetPointer<int8>(), volume_};
}

Status TRT_ShapedWeights::SetShape(DimsAdapter dims) {
  if (volume_ != dims.Volume()) {
    VLOG(2) << "Changing shape from " << shape_.DebugString() << ", to "
            << dims.DebugString();
    return errors::Internal("SetShape would change number of elements");
  }
  shape_ = std::move(dims);
  return OkStatus();
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
#if IS_TRT_VERSION_GE(8, 5, 0, 0)
    case nvinfer1::DataType::kUINT8:
#endif
#if IS_TRT_VERSION_GE(8, 6, 0, 0)
    case nvinfer1::DataType::kFP8:
#endif
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
      data_type_size = 1;
      break;
  }
  return volume_ * data_type_size;
}

string TRT_ShapedWeights::DebugString() const {
  return absl::StrCat(
      "TRT_ShapedWeights(shape=", shape_.DebugString(),
      ", type=", tensorflow::tensorrt::DebugString(type_),
      ", values=", reinterpret_cast<uintptr_t>(GetPointer<int8>()), ")");
}

TRT_TensorOrWeights::TRT_TensorOrWeights(ITensorProxyPtr tensor)
    : tensor_proxy_ptr_(tensor),
      initialized_(true),
      arg_type_(TRT_ArgumentType::TENSOR) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(ITensorProxyPtr tensor, int batch_size)
    : tensor_proxy_ptr_(tensor),
      batch_size_(batch_size),
      initialized_(true),
      arg_type_(TRT_ArgumentType::TENSOR) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(nvinfer1::ITensor* tensor,
                                         int batch_size)
    : tensor_proxy_ptr_(tensor),
      batch_size_(batch_size),
      initialized_(true),
      arg_type_(TRT_ArgumentType::TENSOR) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(nvinfer1::DataType trt_dtype,
                                         const nvinfer1::Dims& trt_dims,
                                         int batch_size)
    : tensor_proxy_ptr_(new SimpleITensor(trt_dtype, trt_dims)),
      batch_size_(batch_size),
      initialized_(true),
      arg_type_(TRT_ArgumentType::TENSOR) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(const TRT_ShapedWeights& weights)
    : weights_(weights),
      initialized_(true),
      arg_type_(TRT_ArgumentType::WEIGHTS) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(const ResourceHandle& resource)
    : resource_(resource),
      initialized_(true),
      arg_type_(TRT_ArgumentType::RESOURCE) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(const TRT_TensorOrWeights& rhs)
    : tensor_proxy_ptr_(rhs.tensor_proxy_ptr_),
      batch_size_(rhs.batch_size_),
      resource_(rhs.resource_),
      weights_(rhs.weights_),
      initialized_(rhs.initialized_),
      arg_type_(rhs.arg_type_) {}

void TRT_TensorOrWeights::operator=(const TRT_TensorOrWeights& rhs) {
  tensor_proxy_ptr_ = rhs.tensor_proxy_ptr_;
  batch_size_ = rhs.batch_size_;
  weights_ = rhs.weights_;
  resource_ = rhs.resource_;
  initialized_ = rhs.initialized_;
  arg_type_ = rhs.arg_type_;
}

ITensorProxyPtr TRT_TensorOrWeights::tensor() const {
  DCHECK(is_tensor());
  return tensor_proxy_ptr_;
}

ResourceHandle TRT_TensorOrWeights::resource() const {
  DCHECK(is_resource());
  return resource_;
}

nvinfer1::Dims TRT_TensorOrWeights::GetTrtDims() const {
  switch (arg_type_) {
    case TRT_ArgumentType::TENSOR:
      return tensor()->getDimensions();
    case TRT_ArgumentType::WEIGHTS:
      return weights().Shape().AsTrtDims();
    case TRT_ArgumentType::RESOURCE:
      return {0, {}};  // Scalar.
  }
}

Status TRT_TensorOrWeights::GetTfType(DataType* tf_type) const {
  if (!initialized_) {
    return errors::Internal("The object is not initialized");
  }
  switch (arg_type_) {
    case TRT_ArgumentType::TENSOR: {
      nvinfer1::DataType trt_type = tensor()->getType();
      return TrtTypeToTfType(trt_type, tf_type);
    }
    case TRT_ArgumentType::WEIGHTS:
      *tf_type = weights().GetTensor().dtype();
      return OkStatus();
    case TRT_ArgumentType::RESOURCE:
      *tf_type = DataType::DT_RESOURCE;
      return OkStatus();
  }
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

StatusOr<TRT_ShapedWeights> TrtWeightStore::GetTempWeights(
    nvinfer1::DataType trt_dtype, const DimsAdapter& dims) {
  DataType tf_dtype;
  TF_RETURN_IF_ERROR(TrtTypeToTfType(trt_dtype, &tf_dtype));
  TensorShape shape;
  TF_RETURN_IF_ERROR(dims.TensorShape(&shape));
  // TODO(jie): check weights size_bytes. 0 means type error
  Tensor tensor(tf_dtype, shape);
  StatusOr<TRT_ShapedWeights> weights =
      TRT_ShapedWeights::CreateWithTensor(trt_dtype, dims, tensor);
  TRT_ENSURE_OK(weights);
  store_.emplace_back(std::move(tensor));
  return weights;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
