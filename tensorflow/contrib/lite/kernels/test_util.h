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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_TEST_UTIL_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_TEST_UTIL_H_

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/testing/util.h"
#include "tensorflow/core/platform/logging.h"

namespace tflite {

// A gmock matcher that check that elements of a float vector match to a given
// tolerance.
std::vector<::testing::Matcher<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error = 1e-5);

template <typename T>
inline std::vector<T> Quantize(const std::vector<float>& data, float scale,
                               int32_t zero_point) {
  std::vector<T> q;
  for (float f : data) {
    q.push_back(std::max(
        std::numeric_limits<T>::min(),
        std::min(std::numeric_limits<T>::max(),
                 static_cast<T>(std::round(zero_point + (f / scale))))));
  }
  return q;
}

template <typename T>
inline std::vector<float> Dequantize(const std::vector<T>& data, float scale,
                                     int32_t zero_point) {
  std::vector<float> f;
  for (T q : data) {
    f.push_back(scale * (q - zero_point));
  }
  return f;
}

// A test model that contains a single operator. All operator inputs and
// output are external to the model, so the tests can directly access them.
// Typical usage:
//    SingleOpModel m;
//    int a = m.AddInput({TensorType_FLOAT32, a_shape});
//    int b = m.AddInput({TensorType_FLOAT32, b_shape});
//    int c = m.AddOutput({TensorType_FLOAT32, {}});
//    m.SetBuiltinOp(...);
//    m.BuildInterpreter({GetShape(a), GetShape(b)});
//    m.PopulateTensor(a, {...});
//    m.PopulateTensor(b, {...});
//    m.Invoke();
//    EXPECT_THAT(m.ExtractVector<float>(c), ArrayFloatNear({...}));
//

// A helper struct to construct test tensors. This is particularly useful for
// quantized tensor which must have their scale and zero_point defined before
// the actual data is known. This mimics what happens in practice: quantization
// parameters are calculate during training.
struct TensorData {
  TensorType type;
  std::vector<int> shape;
  float min;
  float max;
  float scale;
  int32_t zero_point;
};

class SingleOpModel {
 public:
  SingleOpModel() {}
  ~SingleOpModel() {}

  // Copying or assignment is disallowed to simplify ownership semantics.
  SingleOpModel(const SingleOpModel&) = delete;
  SingleOpModel& operator=(const SingleOpModel&) = delete;

  // Add a TensorType input tensor and return its index.
  int AddInput(TensorType type) { return AddInput(TensorData{type}); }
  int AddInput(const TensorData& t);

  // Add a null input tensor (optional input) and return kOptionalTensor.
  int AddNullInput();

  // Add a TensorType output tensor and return its index.
  int AddOutput(TensorType type) { return AddOutput(TensorData{type}); }
  int AddOutput(const TensorData& t);

  template <typename T>
  void QuantizeAndPopulate(int index, std::initializer_list<float> data) {
    TfLiteTensor* t = interpreter_->tensor(index);
    auto q = Quantize<T>(data, t->params.scale, t->params.zero_point);
    PopulateTensor(index, 0, q.data(), q.data() + q.size());
  }

  const std::vector<int>& GetShape(int id) { return tensor_data_.at(id).shape; }

  float GetScale(int id) { return tensor_data_.at(id).scale; }
  int32_t GetZeroPoint(int id) { return tensor_data_.at(id).zero_point; }

  // Define the operator in this model.
  void SetBuiltinOp(BuiltinOperator type, BuiltinOptions builtin_options_type,
                    flatbuffers::Offset<void> builtin_options);
  void SetCustomOp(const string& name,
                   const std::vector<uint8_t>& custom_option,
                   const std::function<TfLiteRegistration*()>& registeration);

  // Build the interpreter for this model. Also, resize and allocate all
  // tensors given the shapes of the inputs.
  void BuildInterpreter(std::vector<std::vector<int>> input_shapes);

  void Invoke();

  void PopulateStringTensor(int index, const std::vector<string>& content) {
    auto tensor = interpreter_->tensor(index);
    DynamicBuffer buf;
    for (const string& s : content) {
      buf.AddString(s.data(), s.length());
    }
    buf.WriteToTensor(tensor);
  }

  // Populate the tensor given its index.
  template <typename T>
  void PopulateTensor(int index, std::initializer_list<T> data) {
    T* v = interpreter_->typed_tensor<T>(index);
    CHECK(v) << "No tensor with index '" << index << "'.";
    for (T f : data) {
      *v = f;
      ++v;
    }
  }

  // Partially populate the tensor, starting at the given offset.
  template <typename T>
  void PopulateTensor(int index, int offset, T* begin, T* end) {
    T* v = interpreter_->typed_tensor<T>(index);
    memcpy(v + offset, begin, (end - begin) * sizeof(T));
  }

  // Return a vector with the flattened contents of a tensor.
  template <typename T>
  std::vector<T> ExtractVector(int index) {
    T* v = interpreter_->typed_tensor<T>(index);
    CHECK(v);
    return std::vector<T>(v, v + GetTensorSize(index));
  }

  std::vector<int> GetTensorShape(int index) {
    std::vector<int> result;
    TfLiteTensor* t = interpreter_->tensor(index);
    for (int i = 0; i < t->dims->size; ++i) {
      result.push_back(t->dims->data[i]);
    }
    return result;
  }

 protected:
  int32_t GetTensorSize(int index) const;

  flatbuffers::FlatBufferBuilder builder_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

 private:
  int AddTensor(TensorData t);

  std::map<int, TensorData> tensor_data_;
  std::vector<int32_t> inputs_;
  std::vector<int32_t> outputs_;
  std::vector<flatbuffers::Offset<Tensor>> tensors_;
  std::vector<flatbuffers::Offset<OperatorCode>> opcodes_;
  std::vector<flatbuffers::Offset<Operator>> operators_;
  std::map<string, std::function<TfLiteRegistration*()>> custom_registrations_;
};

}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_TEST_UTIL_H_
