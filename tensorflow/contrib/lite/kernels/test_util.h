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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_TEST_UTIL_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_TEST_UTIL_H_

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
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
    q.push_back(static_cast<T>(std::max<float>(
        std::numeric_limits<T>::min(),
        std::min<float>(std::numeric_limits<T>::max(),
                        std::round(zero_point + (f / scale))))));
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

class SingleOpResolver : public OpResolver {
 public:
  SingleOpResolver(const BuiltinOperator op, TfLiteRegistration* registration)
      : op_(op), registration_(*registration) {
    registration_.builtin_code = static_cast<int32_t>(op);
    registration_.version = 1;
  }
  const TfLiteRegistration* FindOp(BuiltinOperator op,
                                   int version) const override {
    if (op == op_) {
      return &registration_;
    }
    return nullptr;
  }
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
    return nullptr;
  }

 private:
  const BuiltinOperator op_;
  TfLiteRegistration registration_;
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

  // Templated version of AddConstInput().
  template <typename T>
  int AddConstInput(TensorType type, std::initializer_list<T> data,
                    std::initializer_list<int> shape) {
    int id = AddTensor(TensorData{type, shape}, data);
    inputs_.push_back(id);
    return id;
  }

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

  void SymmetricQuantizeAndPopulate(int index,
                                    std::initializer_list<float> data) {
    TfLiteTensor* t = interpreter_->tensor(index);
    std::vector<float> values(data);
    const int length = values.size();
    std::vector<int8_t> q(length);
    float min, max, scaling_factor;
    tensor_utils::SymmetricQuantizeFloats(values.data(), length, q.data(), &min,
                                          &max, &scaling_factor);
    // Update quantization params.
    t->params.scale = scaling_factor;
    t->params.zero_point = 0;
    PopulateTensor(index, /*offset=*/0, reinterpret_cast<uint8_t*>(q.data()),
                   reinterpret_cast<uint8_t*>(q.data() + q.size()));
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

  void SetResolver(std::unique_ptr<OpResolver> resolver) {
    resolver_ = std::move(resolver);
  }

 protected:
  int32_t GetTensorSize(int index) const;

  flatbuffers::FlatBufferBuilder builder_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<OpResolver> resolver_;

 private:
  // TODO(gavinbelson): sync this method with
  // //tensorflow/contrib/lite/kernels/internal/quantization_util.h?l=31
  template <typename T>
  std::pair<float, int32_t> QuantizationParams(float f_min, float f_max) {
    // These are required by many quantized operations.
    CHECK_LE(f_min, 0);
    CHECK_GE(f_max, 0);
    T q_min = std::numeric_limits<T>::min();
    T q_max = std::numeric_limits<T>::max();
    float range = q_max - q_min;
    float scale = (f_max - f_min) / range;
    int32_t zero_point = std::min(
        q_max,
        std::max(q_min, static_cast<T>(std::round(q_min - f_min / scale))));
    return {scale, zero_point};
  }

  template <typename T>
  int AddTensor(TensorData t, std::initializer_list<T> data) {
    int id = tensors_.size();

    // This is slightly different depending on whether we are adding a
    // quantized or a regular tensor.
    bool is_quantized = (t.min != 0 || t.max != 0 || t.scale != 0);

    flatbuffers::Offset<QuantizationParameters> q_params = 0;

    if (is_quantized) {
      if (t.min != 0 || t.max != 0) {
        if (t.type == TensorType_UINT8) {
          std::tie(t.scale, t.zero_point) =
              QuantizationParams<uint8_t>(t.min, t.max);
        } else if (t.type == TensorType_INT32) {
          std::tie(t.scale, t.zero_point) =
              QuantizationParams<int32_t>(t.min, t.max);
        } else {
          LOG(FATAL) << "No support for the requested quantized type";
        }
        t.min = 0;
        t.max = 0;
      }

      q_params = CreateQuantizationParameters(
          builder_, /*min=*/0, /*max=*/0,
          builder_.CreateVector<float>({t.scale}),
          builder_.CreateVector<int64_t>({t.zero_point}));
    }

    int buffer_id = 0;
    if (data.size()) {
      // Initialize buffers list with empty buffer to allow for non-const
      // tensors.
      if (buffers_.empty()) {
        buffers_.push_back(CreateBuffer(builder_, builder_.CreateVector({})));
      }

      // Add data as a Buffer to buffers list.
      buffer_id = buffers_.size();
      auto data_buffer =
          builder_.CreateVector(reinterpret_cast<const uint8_t*>(data.begin()),
                                sizeof(T) * data.size());
      buffers_.push_back(CreateBuffer(builder_, data_buffer));
    }

    tensors_.push_back(CreateTensor(builder_,
                                    builder_.CreateVector<int>(t.shape), t.type,
                                    /*buffer=*/buffer_id,
                                    /*name=*/0, q_params));

    tensor_data_[id] = t;

    return id;
  }

  std::map<int, TensorData> tensor_data_;
  std::vector<int32_t> inputs_;
  std::vector<int32_t> outputs_;
  std::vector<flatbuffers::Offset<Tensor>> tensors_;
  std::vector<flatbuffers::Offset<OperatorCode>> opcodes_;
  std::vector<flatbuffers::Offset<Operator>> operators_;
  std::vector<flatbuffers::Offset<Buffer>> buffers_;
  std::map<string, std::function<TfLiteRegistration*()>> custom_registrations_;
};

// Base class for single op unit tests.
// The tests are parameterized to test multiple kernels for a single op.
// The parameters are strings like "optimized" and "reference" to have better
// readability in test reports.
//
// To use this class:
// * Define a constant map from strings to TfLiteRegistration.
// * Implement a test class that inherits SingleOpTest.
// * Instantiate the test cases with SingleOpTest::GetKernelTags helper
//   function.
// * Call GetRegistration to get the TfLiteRegistration to be used before
//   building the interpreter.
class SingleOpTest : public ::testing::TestWithParam<string> {
 public:
  static std::vector<string> GetKernelTags(
      const std::map<string, TfLiteRegistration*>& kernel_map) {
    std::vector<string> tags;
    for (auto it : kernel_map) {
      tags.push_back(it.first);
    }
    return tags;
  }

 protected:
  virtual const std::map<string, TfLiteRegistration*>& GetKernelMap() = 0;
  TfLiteRegistration* GetRegistration() {
    return GetKernelMap().at(GetParam());
  }
};

// Strings have a special implementation that is in test_util.cc
template <>
std::vector<string> SingleOpModel::ExtractVector(int index);
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_TEST_UTIL_H_
