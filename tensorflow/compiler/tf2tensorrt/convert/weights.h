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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_WEIGHTS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_WEIGHTS_H_

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <vector>

#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// Class to convert TF compile-time constants (e.g. Const nodes) to TRT weight.
class TRT_ShapedWeights {
 public:
  explicit TRT_ShapedWeights(
      nvinfer1::DataType type = nvinfer1::DataType::kFLOAT);

  // Constructs a weights from another weights.
  //
  // NOTE: this does not copy the underlying buffer but only increase its
  // reference count.
  TRT_ShapedWeights(const TRT_ShapedWeights& rhs) = default;

  nvinfer1::Weights GetTrtWeights() const;

  const Tensor& GetTensor() const { return tensor_; }

  // Returns a pointer of type const T to the underlying buffer of the tensor.
  template <typename T>
  const T* GetPointer() const {
    int64 num_elem =
        (tensor_.NumElements() * DataTypeSize(tensor_.dtype())) / sizeof(T);
    return tensor_.bit_casted_shaped<T, 1>({num_elem}).data();
  }

  // Returns a pointer of type T to the underlying buffer of the tensor.
  template <typename T>
  T* GetPointer() {
    int64 num_elem =
        (tensor_.NumElements() * DataTypeSize(tensor_.dtype())) / sizeof(T);
    return tensor_.bit_casted_shaped<T, 1>({num_elem}).data();
  }

  // Fills all the weight values with value.
  template <typename T>
  Status SetValues(T value) {
    switch (type_) {
      case nvinfer1::DataType::kFLOAT: {
        float* ptr = tensor_.flat<float>().data();
        std::fill(ptr, ptr + volume_, value);
        break;
      }
      case nvinfer1::DataType::kHALF: {
        Eigen::half* ptr = tensor_.flat<Eigen::half>().data();
        std::fill(ptr, ptr + volume_, Eigen::half(value));
        break;
      }
      case nvinfer1::DataType::kINT32: {
        int32* ptr = tensor_.flat<int32>().data();
        std::fill(ptr, ptr + volume_, value);
        break;
      }
      default:
        return errors::InvalidArgument(
            "Unsupported data type ", tensorflow::tensorrt::DebugString(type_));
    }
    return Status::OK();
  }

  Status SetShape(DimsAdapter dims);
  void SetShapeUnsafe(DimsAdapter dims) { shape_ = std::move(dims); }

  // Returns total number of elements. Returning 0 means either some dim is 0
  // or the number of dims is 0. Note that a TF scalar constant is marked as
  // Dims{0, {1}}, and has a count() == 1.
  int64_t count() const { return volume_; }

  size_t size_bytes() const;

  string DebugString() const;

  template <typename T>
  absl::Span<const T> GetSpan() const {
    return absl::Span<const T>(tensor_.flat<T>().data(), volume_);
  }

  template <typename T>
  std::vector<T> ToVector() const {
    auto span = GetSpan<T>();
    return std::vector<T>(span.data(), span.data() + span.size());
  }

  nvinfer1::DataType TrtDType() const { return type_; }

  const DimsAdapter& Shape() const { return shape_; }
  DimsAdapter& Shape() { return shape_; }

 private:
  // The shape of the weights. Defaults to the empty shape.
  DimsAdapter shape_;

  // This creation method is only used by TrtWeightStore, which creates the
  // underlying buffer.
  static StatusOr<TRT_ShapedWeights> CreateWithTensor(nvinfer1::DataType type,
                                                      DimsAdapter dims,
                                                      Tensor tensor);

  nvinfer1::DataType type_;

  // All weights should be stored inside TrtWeightStore to make sure lifetime of
  // all the underlying tensors are available until the engine is built. For
  // this reason, tensor_ should never be reassigned to a different value that
  // is not already present in the TrtWeightStore.
  Tensor tensor_;
  // Contains the volume of the weight's shape.
  int64_t volume_;

  friend class TrtWeightStore;
};

// Container for TRT_ShapedWeights. We need this container because TRT does not
// manage the lifetime of the weights buffer, it only keeps a pointer to it and
// requires that the data referenced by the pointer be available until the
// building of engine is complete. For more information see
// https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_weights.html
//
// TODO(laigd): consider adding garbage collection to the unused weights.
class TrtWeightStore {
 public:
  // Gets a TRT_ShapedWeights with 'type' and 'dims'.
  StatusOr<TRT_ShapedWeights> GetTempWeights(nvinfer1::DataType trt_type,
                                             const DimsAdapter& dims);

  // Gets a TRT_ShapedWeights with the same data type and dimensions as
  // 'weights'.
  StatusOr<TRT_ShapedWeights> GetTempWeights(const TRT_ShapedWeights& weights) {
    return GetTempWeights(weights.TrtDType(), weights.Shape());
  }

 private:
  // The backend storage of the TRT_ShapedWeights.
  std::vector<Tensor> store_;
};

// Represents a TRT-style input to a TF node, it can be either a
// ITensorProxyPtr (representing nvinfer1::ITensor* or SimpleITensor),
// or TRT_ShapedWeights which is compile-time constant.
//
// TODO(laigd): maybe rename it to TrtArgument, or mimic XlaCompiler::Argument.
class TRT_TensorOrWeights {
 public:
  TRT_TensorOrWeights() {}
  TRT_TensorOrWeights(ITensorProxyPtr);
  TRT_TensorOrWeights(ITensorProxyPtr tensor, int batch_size);

  // Constructs a wrapper for the given ITensor.
  // This is used by Converter when building the TRT network, where the ITensor
  // is owned by the TRT network being built. See comment for 'trt_tensor_'
  // in trt_proxy_tensor.h.
  explicit TRT_TensorOrWeights(nvinfer1::ITensor* tensor, int batch_size = -1);

  // Creates a SimpleITensor for trt_dtype and trt_dims and takes ownership of
  // the object. Constructs a wrapper for the SimpleITensor. This is used by
  // TrtNodeValidator to encapsulate the type and shape information for
  // validation of graph nodes, and the created ITensor is fake and temporary,
  // and should not be used to build any TRT network. See comment for
  // 'simple_tensor_' in trt_proxy_tensor.h.
  explicit TRT_TensorOrWeights(nvinfer1::DataType trt_dtype,
                               const nvinfer1::Dims& trt_dims, int batch_size);

  // Constructs a wrapper for the given weights.
  explicit TRT_TensorOrWeights(const TRT_ShapedWeights& weights);

  TRT_TensorOrWeights(const TRT_TensorOrWeights& rhs);

  void operator=(const TRT_TensorOrWeights& rhs);

  bool is_tensor() const { return initialized_ && is_tensor_; }
  bool is_weights() const { return initialized_ && !is_tensor_; }

  ITensorProxyPtr tensor() const;

  TRT_ShapedWeights& weights() {
    DCHECK(is_weights());
    return weights_;
  }

  const TRT_ShapedWeights& weights() const {
    DCHECK(is_weights());
    return weights_;
  }

  nvinfer1::Dims GetTrtDims() const;

  Status GetTfType(DataType* tf_type) const;

  int batch_size() const { return batch_size_; }

  string DebugString() const;

  nvinfer1::DataType TrtDType() const {
    return is_tensor_ ? tensor_proxy_ptr_->getType() : weights_.TrtDType();
  }

 private:
  void set_batch_size(int batch_size) { batch_size_ = batch_size; }

  // First dimension of the TF tensor (NOT tensor_) that is represented by
  // tensor_ is treated as the "batch dimension" by TRT, and tensor_'s
  // dimensions (obtained via tensor_->getDimensions()) do not contain the batch
  // dimension. For example, when a TF tensor with shape (A,B,C) is represented
  // in TRT, tensor_->getDimensions() will be (B,C) and batch_size_ will be A.
  //
  // This requires that all tensors in the subgraph that is converted to a TRT
  // engine have the same batch size are represented by the first dimension of
  // their shape, and Converter will verify this during conversion. The drawback
  // is that currently it cannot convert a graph that doesn't have the batch
  // size represented in the shapes or the batch sizes are different. See
  // b/118387490 for more details.
  //
  // If use_implicit_batch is false, batch_size_ is unused and
  // tensor_->getDimensions() will contain the entire shape (A,B,C).
  ITensorProxyPtr tensor_proxy_ptr_ = nullptr;
  int batch_size_ = -1;

  TRT_ShapedWeights weights_;
  bool initialized_ = false;
  bool is_tensor_ = false;

  friend class Converter;
};
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_WEIGHTS_H_
