/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TENSOR_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TENSOR_H_

#include <vector>
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {

template <typename T>
inline T* GetTensorData(TfLiteTensor* tensor);

template <>
inline float* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.f : nullptr;
}

template <>
inline uint8_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.uint8 : nullptr;
}

template <>
inline int32_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.i32 : nullptr;
}

template <>
inline int64_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.i64 : nullptr;
}

inline int RemapDim(int max_dimensions, int d) {
  return max_dimensions - d - 1;
}

// TODO(ahentz): the implementations in kernels/internal/ take a Dims<4> object
// even if the original tensors were not 4D. We should consider rewriting them
// to take a more generic 'shape' object.
inline Dims<4> GetTensorDims(const int data[], const int size) {
  Dims<4> d;
  for (int i = 0; i < 4; ++i) {
    int src = size - i - 1;
    if (src >= 0) {
      d.sizes[i] = data[src];
    } else {
      d.sizes[i] = 1;
    }
  }
  d.strides[0] = 1;
  for (int i = 1; i < 4; i++) {
    d.strides[i] = d.strides[i - 1] * d.sizes[i - 1];
  }
  return d;
}

inline Dims<4> GetTensorDims(std::vector<int32_t> data) {
  return GetTensorDims(data.data(), data.size());
}

inline Dims<4> GetTensorDims(const TfLiteTensor* tensor) {
  if (tensor == nullptr) {
    return Dims<4>();
  }

  auto* dims = tensor->dims;
  return GetTensorDims(dims->data, dims->size);
}

// A list of tensors in a format that can be used by kernels like split and
// concatenation.
template <typename T>
class VectorOfTensors {
 public:
  // Build with the tensors in 'tensor_list'.
  VectorOfTensors(const TfLiteContext& context,
                  const TfLiteIntArray& tensor_list) {
    int num_tensors = tensor_list.size;

    all_data_.reserve(num_tensors);
    all_dims_.reserve(num_tensors);
    all_dims_ptr_.reserve(num_tensors);

    for (int i = 0; i < num_tensors; ++i) {
      TfLiteTensor* t = &context.tensors[tensor_list.data[i]];
      all_data_.push_back(GetTensorData<T>(t));
      all_dims_.push_back(GetTensorDims(t));
    }

    // Taking the pointer from inside a std::vector is only OK if the vector is
    // never modified, so we populate all_dims in the previous loop and then we
    // are free to grab iterators here.
    for (int i = 0; i < num_tensors; ++i) {
      all_dims_ptr_.push_back(&all_dims_[i]);
    }
  }
  // Return a pointer to the data pointers of all tensors in the list. For
  // example:
  //   float* const* f = v.data();
  //   f[0][1] is the second element of the first tensor.
  T* const* data() const { return all_data_.data(); }

  // Return a pointer the dim pointers of all tensors in the list. For
  // example:
  //   const Dims<4>* const* d = v.dims();
  //   dims[1] are the dimensions of the second tensor in the list.
  const Dims<4>* const* dims() const { return all_dims_ptr_.data(); }

 private:
  std::vector<T*> all_data_;
  std::vector<Dims<4>> all_dims_;
  std::vector<Dims<4>*> all_dims_ptr_;
};

// A list of quantized tensors in a format that can be used by kernels like
// split and concatenation.
class VectorOfQuantizedTensors : public VectorOfTensors<uint8> {
 public:
  // Build with the tensors in 'tensor_list'.
  VectorOfQuantizedTensors(const TfLiteContext& context,
                           const TfLiteIntArray& tensor_list)
      : VectorOfTensors<uint8>(context, tensor_list) {
    for (int i = 0; i < tensor_list.size; ++i) {
      TfLiteTensor* t = &context.tensors[tensor_list.data[i]];
      zero_point_.push_back(t->params.zero_point);
      scale_.push_back(t->params.scale);
    }
  }

  const float* scale() const { return scale_.data(); }
  const int32* zero_point() const { return zero_point_.data(); }

 private:
  std::vector<int32> zero_point_;
  std::vector<float> scale_;
};

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TENSOR_H_
