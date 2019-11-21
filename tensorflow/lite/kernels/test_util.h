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
#ifndef TENSORFLOW_LITE_KERNELS_TEST_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_TEST_UTIL_H_

#include <cmath>
#include <complex>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/optimize/quantization_utils.h"

namespace tflite {

// A gmock matcher that check that elements of a float vector match to a given
// tolerance.
std::vector<::testing::Matcher<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error = 1e-5);

// A gmock matcher that check that elements of a complex vector match to a given
// tolerance.
std::vector<::testing::Matcher<std::complex<float>>> ArrayComplex64Near(
    const std::vector<std::complex<float>>& values, float max_abs_error = 1e-5);

template <typename T>
inline std::vector<T> Quantize(const std::vector<float>& data, float scale,
                               int32_t zero_point) {
  std::vector<T> q;
  for (const auto& f : data) {
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
  f.reserve(data.size());
  for (const T& q : data) {
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
// parameters are calculated during training or post training..
struct TensorData {
  TensorData(TensorType type = TensorType_FLOAT32, std::vector<int> shape = {},
             float min = 0.0f, float max = 0.0f, float scale = 0.0f,
             int32_t zero_point = 0, bool per_channel_quantization = false,
             std::vector<float> per_channel_quantization_scales = {},
             std::vector<int64_t> per_channel_quantization_offsets = {},
             int32_t channel_index = 0)
      : type(type),
        shape(shape),
        min(min),
        max(max),
        scale(scale),
        zero_point(zero_point),
        per_channel_quantization(per_channel_quantization),
        per_channel_quantization_scales(
            std::move(per_channel_quantization_scales)),
        per_channel_quantization_offsets(
            std::move(per_channel_quantization_offsets)),
        channel_index(channel_index) {}
  TensorType type;
  std::vector<int> shape;
  float min;
  float max;
  float scale;
  int32_t zero_point;
  bool per_channel_quantization;
  std::vector<float> per_channel_quantization_scales;
  std::vector<int64_t> per_channel_quantization_offsets;
  int32_t channel_index;
};

class SingleOpResolver : public OpResolver {
 public:
  SingleOpResolver(const BuiltinOperator op, TfLiteRegistration* registration,
                   int version = 1)
      : op_(op), registration_(*registration) {
    registration_.builtin_code = static_cast<int32_t>(op);
    registration_.version = version;
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
  ~SingleOpModel();

  // Set a function callback that is run right after graph is prepared
  // that allows applying external delegates. This is useful for testing
  // other runtimes like NN API or GPU.
  void SetApplyDelegate(std::function<void(Interpreter*)> apply_delegate_fn) {
    apply_delegate_fn_ = apply_delegate_fn;
  }

  void ApplyDelegate();

  // Copying or assignment is disallowed to simplify ownership semantics.
  SingleOpModel(const SingleOpModel&) = delete;
  SingleOpModel& operator=(const SingleOpModel&) = delete;

  // Add a TensorType input tensor and return its index.
  int AddInput(TensorType type, bool is_variable = false) {
    return AddInput(TensorData{type}, is_variable);
  }
  int AddInput(const TensorData& t, bool is_variable = false);

  // Templated version of AddConstInput().
  template <typename T>
  int AddConstInput(const TensorData& t, std::initializer_list<T> data) {
    int id = 0;
    if (t.per_channel_quantization) {
      id = AddTensorPerChannelQuant(t);
    } else {
      id = AddTensor(t, data);
    }
    inputs_.push_back(id);
    return id;
  }
  template <typename T>
  int AddConstInput(TensorType type, std::initializer_list<T> data,
                    std::initializer_list<int> shape) {
    return AddConstInput(TensorData{type, shape}, data);
  }

  // Add a null input tensor (optional input) and return kOptionalTensor.
  int AddNullInput();

  // Add a TensorType output tensor and return its index.
  int AddOutput(TensorType type) { return AddOutput(TensorData{type}); }
  int AddOutput(const TensorData& t);

  template <typename T>
  void QuantizeAndPopulate(int index, const std::vector<float>& data) {
    TfLiteTensor* t = interpreter_->tensor(index);
    auto q = Quantize<T>(data, t->params.scale, t->params.zero_point);
    PopulateTensor(index, 0, q.data(), q.data() + q.size());
  }

  void SymmetricQuantizeAndPopulate(int index, const std::vector<float>& data) {
    std::vector<int8_t> q = QuantizeTensor(index, data);
    PopulateTensor(index, /*offset=*/0, reinterpret_cast<uint8_t*>(q.data()),
                   reinterpret_cast<uint8_t*>(q.data() + q.size()));
  }

  void SignedSymmetricQuantizeAndPopulate(int index,
                                          const std::vector<float>& data) {
    std::vector<int8_t> q = QuantizeTensor(index, data);
    PopulateTensor(index, /*offset=*/0, q.data(), q.data() + q.size());
  }

  // Quantize and populate data for filter with per channel quantization.
  void PerChannelSymmetricQuantizeAndPopulate(
      int index, const std::vector<float>& input_data) {
    TfLiteTensor* t = interpreter_->tensor(index);
    auto* params =
        reinterpret_cast<TfLiteAffineQuantization*>(t->quantization.params);
    const int channel_index = params->quantized_dimension;

    std::vector<int32_t> shape(t->dims->size);
    for (size_t i = 0; i < shape.size(); ++i) {
      shape[i] = t->dims->data[i];
    }
    const int32_t num_inputs = input_data.size();
    const int32_t num_channel = shape[channel_index];
    std::vector<int8_t> quantized_output(num_inputs);
    std::vector<float> scales_inv(num_channel);
    for (int i = 0; i < num_channel; ++i) {
      scales_inv[i] = 1.0f / params->scale->data[i];
    }
    optimize::utils::SymmetricPerChannelQuantizeValues(
        input_data.data(), scales_inv, shape, channel_index, &quantized_output);

    PopulateTensor(index, /*offset=*/0, quantized_output.data(),
                   quantized_output.data() + quantized_output.size());
  }

  // Quantize and populate data for bias with per channel quantization.
  void PerChannelQuantizeBias(int index, const std::vector<float>& input_data) {
    const int32_t num_inputs = input_data.size();
    std::vector<int32_t> quantized_output(num_inputs);
    TfLiteTensor* t = interpreter_->tensor(index);
    auto* params =
        reinterpret_cast<TfLiteAffineQuantization*>(t->quantization.params);
    for (int i = 0; i < num_inputs; ++i) {
      quantized_output[i] = input_data[i] / params->scale->data[i];
    }
    PopulateTensor(index, /*offset=*/0, quantized_output.data(),
                   quantized_output.data() + quantized_output.size());
  }

  const std::vector<int>& GetShape(int id) { return tensor_data_.at(id).shape; }

  float GetScale(int id) { return tensor_data_.at(id).scale; }
  int32_t GetZeroPoint(int id) { return tensor_data_.at(id).zero_point; }

  // Define the operator in this model.
  void SetBuiltinOp(BuiltinOperator type, BuiltinOptions builtin_options_type,
                    flatbuffers::Offset<void> builtin_options);
  void SetCustomOp(const string& name,
                   const std::vector<uint8_t>& custom_option,
                   const std::function<TfLiteRegistration*()>& registration);

  // Build the interpreter for this model. Also, resize and allocate all
  // tensors given the shapes of the inputs.
  void BuildInterpreter(std::vector<std::vector<int>> input_shapes,
                        int num_threads, bool allow_fp32_relax_to_fp16,
                        bool apply_delegate = true);

  void BuildInterpreter(std::vector<std::vector<int>> input_shapes,
                        int num_threads);

  void BuildInterpreter(std::vector<std::vector<int>> input_shapes,
                        bool allow_fp32_relax_to_fp16, bool apply_delegate);

  void BuildInterpreter(std::vector<std::vector<int>> input_shapes);

  // Executes inference, asserting success.
  void Invoke();

  // Executes inference *without* asserting success.
  TfLiteStatus InvokeUnchecked();

  void PopulateStringTensor(int index, const std::vector<string>& content) {
    auto tensor = interpreter_->tensor(index);
    DynamicBuffer buf;
    for (const string& s : content) {
      buf.AddString(s.data(), s.length());
    }
    buf.WriteToTensor(tensor, /*new_shape=*/nullptr);
  }

  // Populate the tensor given its index.
  // TODO(b/110696148) clean up and merge with vector-taking variant below.
  template <typename T>
  void PopulateTensor(int index, const std::initializer_list<T>& data) {
    T* v = interpreter_->typed_tensor<T>(index);
    if (!v) {
      auto* t = interpreter_->tensor(index);
      CHECK(t) << "No tensor with index " << index << ".";
      CHECK(t->data.raw) << "Empty data for tensor with index " << index << ".";
      CHECK_EQ(t->type, typeToTfLiteType<T>())
          << "Type mismatch for tensor with index " << index << ". Requested "
          << TfLiteTypeGetName(typeToTfLiteType<T>()) << ", got "
          << TfLiteTypeGetName(t->type) << ".";
      LOG(FATAL) << "Unknown tensor error.";
    }
    for (const T& f : data) {
      *v = f;
      ++v;
    }
  }

  // Populate the tensor given its index.
  // TODO(b/110696148) clean up and merge with initializer_list-taking variant
  // above.
  template <typename T>
  void PopulateTensor(int index, const std::vector<T>& data) {
    T* v = interpreter_->typed_tensor<T>(index);
    if (!v) {
      auto* t = interpreter_->tensor(index);
      CHECK(t) << "No tensor with index " << index << ".";
      CHECK(t->data.raw) << "Empty data for tensor with index " << index << ".";
      CHECK_EQ(t->type, typeToTfLiteType<T>())
          << "Type mismatch for tensor with index " << index << ". Requested "
          << TfLiteTypeGetName(typeToTfLiteType<T>()) << ", got "
          << TfLiteTypeGetName(t->type) << ".";
      LOG(FATAL) << "Unknown tensor error.";
    }
    for (const T& f : data) {
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
  std::vector<T> ExtractVector(int index) const {
    const T* v = interpreter_->typed_tensor<T>(index);
    CHECK(v);
    return std::vector<T>(v, v + GetTensorSize(index));
  }

  std::vector<int> GetTensorShape(int index) {
    std::vector<int> result;
    TfLiteTensor* t = interpreter_->tensor(index);
    result.reserve(t->dims->size);
    for (int i = 0; i < t->dims->size; ++i) {
      result.push_back(t->dims->data[i]);
    }
    return result;
  }

  void SetNumThreads(int num_threads) {
    CHECK(interpreter_ != nullptr);
    interpreter_->SetNumThreads(num_threads);
  }

  void SetResolver(std::unique_ptr<OpResolver> resolver) {
    resolver_ = std::move(resolver);
  }

  // Enables NNAPI delegate application during interpreter creation.
  static void SetForceUseNnapi(bool use_nnapi);
  static bool GetForceUseNnapi();

 protected:
  int32_t GetTensorSize(int index) const;

  flatbuffers::FlatBufferBuilder builder_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<OpResolver> resolver_;

 private:
  template <typename T>
  std::pair<float, int32_t> QuantizationParams(float f_min, float f_max) {
    int32_t zero_point = 0;
    float scale = 0;
    const T qmin = std::numeric_limits<T>::min();
    const T qmax = std::numeric_limits<T>::max();
    const float qmin_double = qmin;
    const float qmax_double = qmax;
    // 0 should always be a representable value. Let's assume that the initial
    // min,max range contains 0.
    CHECK_LE(f_min, 0);
    CHECK_GE(f_max, 0);
    if (f_min == f_max) {
      // Special case where the min,max range is a point. Should be {0}.
      CHECK_EQ(f_min, 0);
      CHECK_EQ(f_max, 0);
      return {scale, zero_point};
    }

    // General case.
    //
    // First determine the scale.
    scale = (f_max - f_min) / (qmax_double - qmin_double);

    // Zero-point computation.
    // First the initial floating-point computation. The zero-point can be
    // determined from solving an affine equation for any known pair
    // (real value, corresponding quantized value).
    // We know two such pairs: (rmin, qmin) and (rmax, qmax).
    // The arithmetic error on the zero point computed from either pair
    // will be roughly machine_epsilon * (sum of absolute values of terms)
    // so we want to use the variant that adds the smaller terms.
    const float zero_point_from_min = qmin_double - f_min / scale;
    const float zero_point_from_max = qmax_double - f_max / scale;

    const float zero_point_from_min_error =
        std::abs(qmin_double) + std::abs(f_min / scale);

    const float zero_point_from_max_error =
        std::abs(qmax_double) + std::abs(f_max / scale);

    const float zero_point_double =
        zero_point_from_min_error < zero_point_from_max_error
            ? zero_point_from_min
            : zero_point_from_max;

    // Now we need to nudge the zero point to be an integer
    // (our zero points are integer, and this is motivated by the requirement
    // to be able to represent the real value "0" exactly as a quantized value,
    // which is required in multiple places, for example in Im2col with SAME
    //  padding).

    T nudged_zero_point = 0;
    if (zero_point_double < qmin_double) {
      nudged_zero_point = qmin;
    } else if (zero_point_double > qmax_double) {
      nudged_zero_point = qmax;
    } else {
      nudged_zero_point = static_cast<T>(std::round(zero_point_double));
    }

    // The zero point should always be in the range of quantized value,
    // // [qmin, qmax].
    CHECK_GE(nudged_zero_point, qmin);
    CHECK_LE(nudged_zero_point, qmax);

    zero_point = nudged_zero_point;
    // finally, return the values
    return {scale, zero_point};
  }

  int AddTensorPerChannelQuant(TensorData t) {
    const int id = tensors_.size();
    flatbuffers::Offset<QuantizationParameters> q_params = 0;
    q_params = CreateQuantizationParameters(
        builder_, /*min=*/0, /*max=*/0,
        /*scale=*/
        builder_.CreateVector<float>(t.per_channel_quantization_scales),
        /*zero point=*/
        builder_.CreateVector<int64_t>(t.per_channel_quantization_offsets),
        QuantizationDetails_NONE, 0, t.channel_index);
    tensors_.push_back(
        CreateTensor(builder_, builder_.CreateVector<int>(t.shape), t.type,
                     /*buffer=*/0,
                     /*name=*/0, q_params, /*is_variable=*/false));
    tensor_data_[id] = t;
    return id;
  }

  template <typename T>
  int AddTensor(TensorData t, std::initializer_list<T> data,
                bool is_variable = false) {
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
        } else if (t.type == TensorType_INT8) {
          std::tie(t.scale, t.zero_point) =
              QuantizationParams<int8_t>(t.min, t.max);
        } else if (t.type == TensorType_INT32) {
          std::tie(t.scale, t.zero_point) =
              QuantizationParams<int32_t>(t.min, t.max);
        } else if (t.type == TensorType_INT16) {
          std::tie(t.scale, t.zero_point) =
              QuantizationParams<int16_t>(t.min, t.max);
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
                                    /*name=*/0, q_params, is_variable));

    tensor_data_[id] = t;

    return id;
  }

  std::vector<int8_t> QuantizeTensor(int index,
                                     const std::vector<float>& data) {
    TfLiteTensor* t = interpreter_->tensor(index);
    const int length = data.size();
    std::vector<int8_t> q(length);
    float min, max, scaling_factor;
    tensor_utils::SymmetricQuantizeFloats(data.data(), length, q.data(), &min,
                                          &max, &scaling_factor);
    // Update quantization params.
    t->params.scale = scaling_factor;
    t->params.zero_point = 0;
    // Populate the new quantization params.
    TfLiteQuantizationFree(&t->quantization);
    t->quantization.type = kTfLiteAffineQuantization;
    auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
        malloc(sizeof(TfLiteAffineQuantization)));
    affine_quantization->quantized_dimension = 0;
    affine_quantization->scale = TfLiteFloatArrayCreate(1);
    affine_quantization->zero_point = TfLiteIntArrayCreate(1);
    affine_quantization->scale->data[0] = scaling_factor;
    affine_quantization->zero_point->data[0] = 0;
    t->quantization.params = affine_quantization;
    return q;
  }

  // Checks if acceleration has been done as expected.
  // Currently supports only NNAPI.
  // It verifies if the test was configured to run with NNAPI acceleration
  // or not (SetForceUseNnapi(true)).
  // In affirmative case it checks if:
  // - the test case has been listed in the list of nnapi-accelerated cases
  // - the test is running on a device (NNAPI has been loaded)
  //
  // The list of nnapi-accelerated test cases is a file containing regex to
  // include or exclude specific test cases plus the minimum android SDK version
  // the acceleration should be enabled for. For example:
  // To enable the test BorderFloat in TopKV2OpTest only from
  // android_sdk_version 29:
  //
  // TopKV2OpTest/BorderFloat,29
  //
  // And to have it always excluded while enabling all other Float tests
  // (the order of the rules is important, the first one matching is used):
  //
  // -TopKV2OpTest/BorderFloat
  // TopKV2OpTest/.+Float

  void ValidateAcceleration();

  // If the test was configured to use NNAPI and NNAPI was actually loaded,
  // checks if the single operation in the model has been accelerated.
  void ExpectOpAcceleratedWithNnapi(const std::string& test_id);

  std::map<int, TensorData> tensor_data_;
  std::vector<int32_t> inputs_;
  std::vector<int32_t> outputs_;
  std::vector<flatbuffers::Offset<Tensor>> tensors_;
  std::vector<flatbuffers::Offset<OperatorCode>> opcodes_;
  std::vector<flatbuffers::Offset<Operator>> operators_;
  std::vector<flatbuffers::Offset<Buffer>> buffers_;
  std::map<string, std::function<TfLiteRegistration*()>> custom_registrations_;
  // A function pointer that gets called after the interpreter is created but
  // before evaluation happens. This is useful for applying a delegate.
  std::function<void(Interpreter*)> apply_delegate_fn_;
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
    tags.reserve(kernel_map.size());
    for (const auto& it : kernel_map) {
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

// Returns the corresponding TensorType given the type T.
template <typename T>
TensorType GetTensorType() {
  if (std::is_same<T, float>::value) return TensorType_FLOAT32;
  if (std::is_same<T, TfLiteFloat16>::value) return TensorType_FLOAT16;
  if (std::is_same<T, int32_t>::value) return TensorType_INT32;
  if (std::is_same<T, int64_t>::value) return TensorType_INT64;
  if (std::is_same<T, uint8_t>::value) return TensorType_UINT8;
  if (std::is_same<T, int8_t>::value) return TensorType_INT8;
  if (std::is_same<T, string>::value) return TensorType_STRING;
  return TensorType_MIN;  // default value
}

// Strings have a special implementation that is in test_util.cc
template <>
std::vector<string> SingleOpModel::ExtractVector(int index) const;

// The TypeUnion struct specializations hold a collection of related types.
// Each struct holds: 1. a primitive type (e.g. float), 2. a TensorType (e.g.
// TensorType_FLOAT32, and 3. a TfLiteType (e.g. kTfLiteFloat32). The latter
// two are actually enum values and not raw types, but these specializations
// make it easy to use gUnit Typed Test Suite:
// https://github.com/google/googletest/blob/master/googletest/docs/advanced.md#typed-tests
template <typename T>
struct TypeUnion;

template <>
struct TypeUnion<float> {
 public:
  static const TensorType tensor_type = TensorType::TensorType_FLOAT32;
  static const TfLiteType tflite_type = TfLiteType::kTfLiteFloat32;
  typedef float ScalarType;
};

template <>
struct TypeUnion<int32_t> {
 public:
  static const TensorType tensor_type = TensorType::TensorType_INT32;
  static const TfLiteType tflite_type = TfLiteType::kTfLiteInt32;
  typedef int32_t ScalarType;
};

template <>
struct TypeUnion<int16_t> {
 public:
  static const TensorType tensor_type = TensorType::TensorType_INT16;
  static const TfLiteType tflite_type = TfLiteType::kTfLiteInt16;
  typedef int16_t ScalarType;
};

template <>
struct TypeUnion<int8_t> {
 public:
  static const TensorType tensor_type = TensorType::TensorType_INT8;
  static const TfLiteType tflite_type = TfLiteType::kTfLiteInt8;
  typedef int8_t ScalarType;
};

template <>
struct TypeUnion<uint8_t> {
 public:
  static const TensorType tensor_type = TensorType::TensorType_UINT8;
  static const TfLiteType tflite_type = TfLiteType::kTfLiteUInt8;
  typedef uint8_t ScalarType;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_TEST_UTIL_H_
