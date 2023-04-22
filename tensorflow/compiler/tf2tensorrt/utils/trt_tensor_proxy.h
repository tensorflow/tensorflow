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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_TENSOR_PROXY_H
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_TENSOR_PROXY_H

#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {

namespace tensorrt {

// SimpleITensor implements part of the ITensor interfaces to support the TF-TRT
// validator, as well as some TF-TRT tests. The former use case only utilizes
// the interfaces related to shape and type information.
class SimpleITensor {
 public:
  SimpleITensor(nvinfer1::DataType trt_dtype, const nvinfer1::Dims& trt_dims)
      : trt_dtype_(trt_dtype), trt_dims_(trt_dims) {}

  SimpleITensor() : dynamic_range_min_(0.0f), dynamic_range_max_(0.0f) {}
  SimpleITensor(const nvinfer1::Dims& dims)
      : trt_dims_(dims), dynamic_range_min_(0.0f), dynamic_range_max_(0.0f) {}

  SimpleITensor(const std::vector<int>& dims) {
    trt_dims_.nbDims = dims.size();
    for (int i = 0; i < dims.size(); ++i) {
      trt_dims_.d[i] = dims[i];
    }
    dynamic_range_min_ = 0.0f;
    dynamic_range_max_ = 0.0f;
  }

  void setName(const char* name) {}

  const char* getName() const { return ""; }

  void setDimensions(nvinfer1::Dims dimensions) { trt_dims_ = dimensions; }

  nvinfer1::Dims getDimensions() const { return trt_dims_; }

  void setType(nvinfer1::DataType trt_dtype) { trt_dtype_ = trt_dtype; }

  nvinfer1::DataType getType() const { return trt_dtype_; }

  bool isNetworkInput() const { return false; }

  bool isNetworkOutput() const { return false; }

  void setBroadcastAcrossBatch(bool broadcastAcrossBatch) {}

  bool getBroadcastAcrossBatch() const { return false; }

  nvinfer1::TensorLocation getLocation() const { return location_; }

  void setLocation(nvinfer1::TensorLocation location) { location_ = location; }
  bool setDynamicRange(float min, float max) {
    dynamic_range_max_ = max;
    dynamic_range_min_ = min;
    return true;
  }

  float getDynamicRange() const {
    return (std::abs(dynamic_range_min_) + dynamic_range_max_) / 2.f;
  }
  bool dynamicRangeIsSet() const { return true; }

  void resetDynamicRange() {
    dynamic_range_min_ = 0.f;
    dynamic_range_max_ = 0.f;
  }
  float getDynamicRangeMin() const { return dynamic_range_min_; }

  float getDynamicRangeMax() const { return dynamic_range_max_; }

  void setAllowedFormats(nvinfer1::TensorFormats formats) {}

  nvinfer1::TensorFormats getAllowedFormats() const { return 1; }

  bool isShapeTensor() const { return false; }
  bool isExecutionTensor() const { return true; }

 private:
  nvinfer1::DataType trt_dtype_;
  nvinfer1::Dims trt_dims_;
  std::string name_;
  nvinfer1::TensorLocation location_;
  float dynamic_range_min_;
  float dynamic_range_max_;
};

enum class TensorType : int { kTRT, kSIMPLE };

class ITensorProxy {
 public:
  //! ITensor not owned
  ITensorProxy(nvinfer1::ITensor* trt_tensor)
      : trt_tensor_(trt_tensor), ttype_(TensorType::kTRT) {}

  //! SimpleITensor owned
  ITensorProxy(SimpleITensor* simple_itensor)
      : simple_tensor_(simple_itensor), ttype_(TensorType::kSIMPLE) {}

  //! SimpleITensor owned
  explicit ITensorProxy(nvinfer1::DataType trt_dtype,
                        const nvinfer1::Dims& trt_dims)
      : simple_tensor_(std::unique_ptr<SimpleITensor>(
            new SimpleITensor(trt_dtype, trt_dims))),
        ttype_(TensorType::kSIMPLE) {}

  //! Variants for testing purposes
  ITensorProxy()
      : simple_tensor_(std::unique_ptr<SimpleITensor>(new SimpleITensor())),
        ttype_(TensorType::kSIMPLE) {}

  explicit ITensorProxy(const nvinfer1::Dims& dims)
      : simple_tensor_(std::unique_ptr<SimpleITensor>(new SimpleITensor(dims))),
        ttype_(TensorType::kSIMPLE) {}

  explicit ITensorProxy(const std::vector<int>& dims)
      : simple_tensor_(std::unique_ptr<SimpleITensor>(new SimpleITensor(dims))),
        ttype_(TensorType::kSIMPLE) {}

  bool is_trt_tensor() const {
    assert(validate());
    assert(ttype_ == TensorType::kTRT);
    return trt_tensor_ != nullptr;
  }

  bool is_simple_tensor() const {
    assert(validate());
    assert(ttype_ == TensorType::kSIMPLE);
    return simple_tensor_ != nullptr;
  }

  TensorType ttype() const { return ttype_; }

  nvinfer1::ITensor* trt_tensor() const {
    assert(trt_tensor_ != nullptr);
    assert(ttype_ == TensorType::kTRT);
    return trt_tensor_;
  }

  SimpleITensor* simple_tensor() const {
    assert(simple_tensor_ != nullptr);
    assert(ttype_ == TensorType::kSIMPLE);
    return simple_tensor_.get();
  }

  void setName(const char* name) {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setName(name);
      case TensorType::kSIMPLE:
        return simple_tensor_->setName(name);
    }
    assert(0 && "Unsupported itensor_ type");
  }

  const char* getName() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getName();
      case TensorType::kSIMPLE:
        return simple_tensor_->getName();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  void setDimensions(nvinfer1::Dims dimensions) {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setDimensions(dimensions);
      case TensorType::kSIMPLE:
        return simple_tensor_->setDimensions(dimensions);
    }
    assert(0 && "Unsupported itensor_ type");
  }

  nvinfer1::Dims getDimensions() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getDimensions();
      case TensorType::kSIMPLE:
        return simple_tensor_->getDimensions();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  void setType(nvinfer1::DataType type) {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setType(type);
      case TensorType::kSIMPLE:
        return simple_tensor_->setType(type);
    }
    assert(0 && "Unsupported itensor_ type");
  }

  nvinfer1::DataType getType() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getType();
      case TensorType::kSIMPLE:
        return simple_tensor_->getType();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  bool isNetworkInput() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->isNetworkInput();
      case TensorType::kSIMPLE:
        return simple_tensor_->isNetworkInput();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  bool isNetworkOutput() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->isNetworkOutput();
      case TensorType::kSIMPLE:
        return simple_tensor_->isNetworkOutput();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  void setBroadcastAcrossBatch(bool broadcastAcrossBatch) {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setBroadcastAcrossBatch(broadcastAcrossBatch);
      case TensorType::kSIMPLE:
        return simple_tensor_->setBroadcastAcrossBatch(broadcastAcrossBatch);
    }
    assert(0 && "Unsupported itensor_ type");
  }

  bool getBroadcastAcrossBatch() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getBroadcastAcrossBatch();
      case TensorType::kSIMPLE:
        return simple_tensor_->getBroadcastAcrossBatch();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  nvinfer1::TensorLocation getLocation() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getLocation();
      case TensorType::kSIMPLE:
        return simple_tensor_->getLocation();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  void setLocation(nvinfer1::TensorLocation location) {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setLocation(location);
      case TensorType::kSIMPLE:
        return simple_tensor_->setLocation(location);
    }
    assert(0 && "Unsupported itensor_ type");
  }

  bool setDynamicRange(float min, float max) {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setDynamicRange(min, max);
      case TensorType::kSIMPLE:
        return simple_tensor_->setDynamicRange(min, max);
    }
    assert(0 && "Unsupported itensor_ type");
  }

  bool dynamicRangeIsSet() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->dynamicRangeIsSet();
      case TensorType::kSIMPLE:
        return simple_tensor_->dynamicRangeIsSet();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  void resetDynamicRange() {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->resetDynamicRange();
      case TensorType::kSIMPLE:
        return simple_tensor_->resetDynamicRange();
    }
    assert(0 && "Unsupported itensor_ type");
  }
  float getDynamicRangeMin() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getDynamicRangeMin();
      case TensorType::kSIMPLE:
        return simple_tensor_->getDynamicRangeMin();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  float getDynamicRangeMax() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getDynamicRangeMax();
      case TensorType::kSIMPLE:
        return simple_tensor_->getDynamicRangeMax();
    }
    assert(0 && "Unsupported itensor_ type");
  }
#if IS_TRT_VERSION_GE(5, 0, 0, 0) && !IS_TRT_VERSION_GE(8, 0, 0, 0)
  float getDynamicRange() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getDynamicRange();
      case TensorType::kSIMPLE:
        return simple_tensor_->getDynamicRange();
    }
    assert(0 && "Unsupported itensor_ type");
  }
#endif
  void setAllowedFormats(nvinfer1::TensorFormats formats) {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setAllowedFormats(formats);
      case TensorType::kSIMPLE:
        return simple_tensor_->setAllowedFormats(formats);
    }
    assert(0 && "Unsupported itensor_ type");
  }

  nvinfer1::TensorFormats getAllowedFormats() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getAllowedFormats();
      case TensorType::kSIMPLE:
        return simple_tensor_->getAllowedFormats();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  bool isShapeTensor() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->isShapeTensor();
      case TensorType::kSIMPLE:
        return simple_tensor_->isShapeTensor();
    }
    assert(0 && "Unsupported itensor_ type");
  }

  bool isExecutionTensor() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->isExecutionTensor();
      case TensorType::kSIMPLE:
        return simple_tensor_->isExecutionTensor();
    }
    assert(0 && "Unsupported itensor_ type");
  }

 private:
  bool validate() const {
    return (trt_tensor_ && !simple_tensor_) || (!trt_tensor_ && simple_tensor_);
  }

  // When ITensorProxy represents an ITensor, the ITensor can be either passed
  // by the caller via the constructor that takes an ITensor* as parameter, or
  // be created as a SimpleITensor.
  //
  // In the first case, the ITensor pointer is stored in 'tensor_' below, and
  // the ITensor itself is not owned by this class. This method is used by
  // Converter (e.g. AddInputTensor) and op converters during TRT network
  // construction, where the TRT network owns the ITensor.
  //
  nvinfer1::ITensor* trt_tensor_ = nullptr;  // Not owned.
  // In the second case, the created SimpleITensor is stored in
  // 'simple_itensor_' below and is owned by this class. SimpleITensor is a fake
  // implementation of ITensor and is used for testing and by TrtNodeValidator
  //  to validate the graph nodes.
  std::shared_ptr<SimpleITensor> simple_tensor_ = nullptr;

  TensorType ttype_;
};

class ITensorProxyPtr {
 public:
  ITensorProxyPtr(std::nullptr_t) : p_(nullptr) {}
  ITensorProxyPtr(ITensorProxy* p) : p_(p) {}
  ITensorProxyPtr(nvinfer1::ITensor* p) : p_(new ITensorProxy(p)) {}
  ITensorProxyPtr(SimpleITensor* p) : p_(new ITensorProxy(p)) {}

  ITensorProxyPtr() : p_(new ITensorProxy()) {}
  ITensorProxyPtr(const nvinfer1::Dims& dims) : p_(new ITensorProxy(dims)) {}
  ITensorProxyPtr(const std::vector<int>& dims) : p_(new ITensorProxy(dims)) {}

  std::shared_ptr<ITensorProxy> p_{nullptr};
  ITensorProxy* operator->() { return p_.get(); }
  ITensorProxy* operator->() const { return p_.get(); }
  ITensorProxy* operator*() { return p_.get(); }
  ITensorProxy* operator*() const { return p_.get(); }
};

inline bool operator==(const ITensorProxyPtr& p1, const ITensorProxyPtr& p2) {
  if (p1.p_ == nullptr) {
    return p2.p_ == nullptr;
  }
  if (p2.p_ == nullptr) {
    return p1.p_ == nullptr;
  }
  return (p1->ttype() == p2->ttype()) &&
         ((p1->ttype() == TensorType::kTRT &&
           p1->trt_tensor() == p2->trt_tensor()) ||
          (p1->ttype() == TensorType::kSIMPLE &&
           p1->simple_tensor() == p2->simple_tensor()));
}

struct ITensorProxyHash {
  size_t operator()(const ITensorProxyPtr& tensor) const {
    return reinterpret_cast<std::uintptr_t>(tensor.p_.get());
  }
};

}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_TENSOR_PROXY_H
