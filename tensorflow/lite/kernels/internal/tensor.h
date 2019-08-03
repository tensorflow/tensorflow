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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_H_

#include <complex>
#include <vector>
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {

template <>
inline std::complex<float>* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr
             ? reinterpret_cast<std::complex<float>*>(tensor->data.c64)
             : nullptr;
}

template <>
inline const std::complex<float>* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr
             ? reinterpret_cast<const std::complex<float>*>(tensor->data.c64)
             : nullptr;
}

inline RuntimeShape GetTensorShape(std::vector<int32_t> data) {
  return RuntimeShape(data.size(), data.data());
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
    all_shape_.reserve(num_tensors);
    all_shape_ptr_.reserve(num_tensors);

    for (int i = 0; i < num_tensors; ++i) {
      TfLiteTensor* t = &context.tensors[tensor_list.data[i]];
      all_data_.push_back(GetTensorData<T>(t));
      all_shape_.push_back(GetTensorShape(t));
    }

    // Taking the pointer from inside a std::vector is only OK if the vector is
    // never modified, so we populate all_shape in the previous loop and then we
    // are free to grab iterators here.
    for (int i = 0; i < num_tensors; ++i) {
      all_shape_ptr_.push_back(&all_shape_[i]);
    }
  }
  // Return a pointer to the data pointers of all tensors in the list. For
  // example:
  //   float* const* f = v.data();
  //   f[0][1] is the second element of the first tensor.
  T* const* data() const { return all_data_.data(); }

  // Return a pointer the shape pointers of all tensors in the list. For
  // example:
  //   const RuntimeShape* const* d = v.dims();
  //   dims[1] are the dimensions of the second tensor in the list.
  const RuntimeShape* const* shapes() const { return all_shape_ptr_.data(); }

 private:
  std::vector<T*> all_data_;
  std::vector<RuntimeShape> all_shape_;
  std::vector<RuntimeShape*> all_shape_ptr_;
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

// Writes randomly accessed values from `input` sequentially into `output`.
template <typename T>
class SequentialTensorWriter {
 public:
  SequentialTensorWriter(const TfLiteTensor* input, TfLiteTensor* output) {
    input_data_ = GetTensorData<T>(input);
    output_ptr_ = GetTensorData<T>(output);
  }
  SequentialTensorWriter(const T* input_data, T* output_data)
      : input_data_(input_data), output_ptr_(output_data) {}

  void Write(int position) { *output_ptr_++ = input_data_[position]; }
  void WriteN(int position, int len) {
    memcpy(output_ptr_, &input_data_[position], sizeof(T) * len);
    output_ptr_ += len;
  }

 private:
  const T* input_data_;
  T* output_ptr_;
};

template <>
class SequentialTensorWriter<string> {
 public:
  SequentialTensorWriter(const TfLiteTensor* input, TfLiteTensor* output)
      : input_(input), output_(output) {}
  ~SequentialTensorWriter() { buffer_.WriteToTensor(output_, nullptr); }

  void Write(int position) { this->WriteN(position, 1); }
  void WriteN(int position, int len) {
    for (int i = 0; i < len; i++) {
      buffer_.AddString(GetString(input_, position + i));
    }
  }

 private:
  const TfLiteTensor* input_;
  TfLiteTensor* output_;
  DynamicBuffer buffer_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_H_
