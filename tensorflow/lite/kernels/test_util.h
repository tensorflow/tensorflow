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

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"  // IWYU pragma: keep
#include "tensorflow/lite/tools/optimize/quantization_utils.h"
#include "tensorflow/lite/type_to_tflitetype.h"

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

template <>
constexpr TfLiteType typeToTfLiteType<Eigen::half>() {
  return kTfLiteFloat16;
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
  // NOLINTNEXTLINE
  TensorData(TensorType type = TensorType_FLOAT32, std::vector<int> shape = {},
             float min = 0.0f, float max = 0.0f, float scale = 0.0f,
             int32_t zero_point = 0, bool per_channel_quantization = false,
             std::vector<float> per_channel_quantization_scales = {},
             std::vector<int64_t> per_channel_quantization_offsets = {},
             int32_t channel_index = 0, std::vector<int> traversal_order = {},
             std::vector<TfLiteDimensionType> format = {},
             std::vector<int> block_size = {}, std::vector<int> block_map = {},
             std::vector<int> shape_signature = {})
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
        channel_index(channel_index),
        traversal_order(traversal_order),
        format(format),
        block_size(block_size),
        block_map(block_map),
        shape_signature(shape_signature) {}
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
  std::vector<int> traversal_order;
  std::vector<TfLiteDimensionType> format;
  std::vector<int> block_size;
  std::vector<int> block_map;
  std::vector<int> shape_signature;
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

class SingleOpModel;
class AccelerationValidator {
 public:
  using Callback = std::function<void(const SingleOpModel& model)>;

  // Returns a global AccelerationValidator instance.
  static AccelerationValidator* Get();

  // Adds a callback function that will be invoked at the end of a kernel test
  // to validate acceleration.
  void AddCallback(Callback callback);

  // Performs acceleration validation with all registered callbacks.
  void Validate(const SingleOpModel& model) const;

 private:
  std::vector<Callback> callbacks_;
};

class SingleOpModel {
 public:
  SingleOpModel() {}
  ~SingleOpModel();

  // Set a delegate that is applied right after graph is prepared. This is
  // useful for testing other runtimes like NN API or GPU.
  // Note: the caller still owns the memory of the passed-in `delegate`.
  void SetDelegate(TfLiteDelegate* delegate) {
    delegate_ = delegate;
    // As this is a manually-set TF Lite delegate, we assume the intention of
    // the test is to test against the particular delegate, hence bypassing
    // applying TfLite default delegates (i.e. the XNNPACK delegate).
    if (delegate_ != nullptr) {
      SetBypassDefaultDelegates();
    }
  }

  TfLiteStatus ApplyDelegate();

  // Copying or assignment is disallowed to simplify ownership semantics.
  SingleOpModel(const SingleOpModel&) = delete;
  SingleOpModel& operator=(const SingleOpModel&) = delete;

  // Add a TensorType input tensor and return its index.
  int AddInput(const TensorData& t);
  int AddVariableInput(const TensorData& t);

  int AddIntermediate(TensorType type, const std::vector<float>& scale,
                      const std::vector<int64_t>& zero_point);

  // Templated version of AddConstInput() taking pointer and size.
  template <typename T>
  int AddConstInput(const TensorData& t, const T* data, size_t size) {
    int id = 0;
    if (t.per_channel_quantization) {
      id = AddTensorPerChannelQuant(t, data, size);
    } else {
      id = AddTensor(t, data, size);
    }
    inputs_.push_back(id);
    return id;
  }

  // Templated version of AddConstInput() taking vector and shape.
  template <typename T>
  int AddConstInput(TensorType type, const std::vector<T>& data,
                    std::initializer_list<int> shape) {
    return AddConstInput(TensorData{type, shape}, data.data(), data.size());
  }

  // Templated version of AddConstInput() taking TensorType, initializer_list
  // and shape.
  template <typename T>
  int AddConstInput(TensorType type, std::initializer_list<T> data,
                    std::initializer_list<int> shape) {
    return AddConstInput<T>(TensorData{type, shape}, data.begin(), data.size());
  }

  // Templated version of AddConstInput() taking TensorData, initializer_list
  // and shape.
  template <typename T>
  int AddConstInput(const TensorData& t, std::initializer_list<T> data) {
    return AddConstInput(t, data.begin(), data.size());
  }

  // Templated version of AddConstInput() taking TensorData and vector.
  template <typename T>
  int AddConstInput(const TensorData& t, const std::vector<T>& data) {
    return AddConstInput(t, data.data(), data.size());
  }

  // TODO(b/166202747): Use a better way to do type specialization. Reduce
  // duplicate code in the two functions below.
  int AddConstSparseInput(const TensorData& t,
                          const std::vector<int8_t>& data) {
    int id = tensors_.size();
    const int dims_count = t.traversal_order.size();
    std::vector<int8_t> dense_data(data);

    tflite::internal::sparsity::FormatConverter<int8_t> converter(
        t.shape, t.traversal_order, t.format, t.block_size, t.block_map);
    converter.DenseToSparse(dense_data.data());

    const auto& dim_metadata = converter.GetDimMetadata();
    const auto& sparse_data = converter.GetData();

    // Build sparsity parameter.
    std::vector<flatbuffers::Offset<DimensionMetadata>> fb_dim_metadata(
        dims_count);
    for (int i = 0; i < dims_count; i++) {
      const int metadata_idx = 2 * i;
      if (i < t.shape.size() &&
          t.format[t.traversal_order[i]] == kTfLiteDimSparseCSR) {
        auto array_segments =
            CreateInt32Vector(builder_, builder_.CreateVector<int>(
                                            dim_metadata[metadata_idx]))
                .Union();
        auto array_indices =
            CreateInt32Vector(builder_, builder_.CreateVector<int>(
                                            dim_metadata[metadata_idx + 1]))
                .Union();
        fb_dim_metadata[i] = CreateDimensionMetadata(
            builder_, DimensionType_SPARSE_CSR, 0,
            SparseIndexVector_Int32Vector, array_segments,
            SparseIndexVector_Int32Vector, array_indices);
      } else {
        fb_dim_metadata[i] = CreateDimensionMetadata(
            builder_, DimensionType_DENSE, dim_metadata[metadata_idx][0]);
      }
    }

    flatbuffers::Offset<SparsityParameters> s_param = CreateSparsityParameters(
        builder_, builder_.CreateVector<int>(t.traversal_order),
        builder_.CreateVector<int>(t.block_map),
        builder_.CreateVector(fb_dim_metadata));

    int buffer_id = 0;
    if (!data.empty()) {
      // Initialize buffers list with empty buffer to allow for non-const
      // tensors.
      if (buffers_.empty()) {
        buffers_.push_back(CreateBuffer(builder_, builder_.CreateVector({})));
      }

      // Add compressed data as a Buffer to buffers list.
      buffer_id = buffers_.size();
      auto data_buffer = builder_.CreateVector(
          reinterpret_cast<const uint8_t*>(sparse_data.data()),
          sparse_data.size());
      buffers_.push_back(CreateBuffer(builder_, data_buffer));
    }

    tensors_.push_back(CreateTensor(
        builder_, builder_.CreateVector<int>(t.shape), t.type,
        /*buffer=*/buffer_id,
        /*name=*/0, /*quantization=*/0, /*is_variable=*/false, s_param));

    inputs_.push_back(id);
    tensor_data_[id] = t;

    return id;
  }

  // Add a constant sparse tensor as input.
  template <typename T>
  int AddConstSparseInput(const TensorData& t, const std::vector<T>& data,
                          bool symmetric_quantize = false) {
    int id = tensors_.size();
    const int dims_count = t.traversal_order.size();
    std::vector<T> dense_data(data);

    tflite::internal::sparsity::FormatConverter<T> converter(
        t.shape, t.traversal_order, t.format, t.block_size, t.block_map);
    converter.DenseToSparse(dense_data.data());

    const auto dim_metadata = converter.GetDimMetadata();
    const auto sparse_data = converter.GetData();

    // Build sparsity parameter.
    std::vector<flatbuffers::Offset<DimensionMetadata>> fb_dim_metadata(
        dims_count);
    for (int i = 0; i < dims_count; i++) {
      const int metadata_idx = 2 * i;
      if (i < t.shape.size() &&
          t.format[t.traversal_order[i]] == kTfLiteDimSparseCSR) {
        auto array_segments =
            CreateInt32Vector(builder_,
                              builder_.CreateVector(dim_metadata[metadata_idx]))
                .Union();
        auto array_indices =
            CreateInt32Vector(
                builder_, builder_.CreateVector(dim_metadata[metadata_idx + 1]))
                .Union();
        fb_dim_metadata[i] = CreateDimensionMetadata(
            builder_, DimensionType_SPARSE_CSR, 0,
            SparseIndexVector_Int32Vector, array_segments,
            SparseIndexVector_Int32Vector, array_indices);
      } else {
        fb_dim_metadata[i] = CreateDimensionMetadata(
            builder_, DimensionType_DENSE, dim_metadata[metadata_idx][0]);
      }
    }

    flatbuffers::Offset<SparsityParameters> s_param = CreateSparsityParameters(
        builder_, builder_.CreateVector(t.traversal_order),
        builder_.CreateVector(t.block_map),
        builder_.CreateVector(fb_dim_metadata));

    flatbuffers::Offset<QuantizationParameters> q_params = 0;
    int buffer_id = 0;
    if (!data.empty()) {
      // Initialize buffers list with empty buffer to allow for non-const
      // tensors.
      if (buffers_.empty()) {
        buffers_.push_back(CreateBuffer(builder_, builder_.CreateVector({})));
      }

      // Add compressed data as a Buffer to buffers list.
      buffer_id = buffers_.size();
      // When the quantization parameter is set for the added tensor, we
      // quantize the given data.
      bool is_quantized = (t.min != 0 || t.max != 0 || t.scale != 0);
      if (symmetric_quantize) {
        const int length = sparse_data.size();
        std::vector<int8_t> q(length);
        float min, max, scaling_factor;
        tensor_utils::SymmetricQuantizeFloats(
            sparse_data.data(), length, q.data(), &min, &max, &scaling_factor);
        std::vector<float> scales{scaling_factor};
        std::vector<int64_t> zero_points{0};
        q_params = CreateQuantizationParameters(
            builder_, 0, 0, builder_.CreateVector<float>(scales),
            builder_.CreateVector<int64_t>(zero_points));
        auto data_buffer = builder_.CreateVector(
            reinterpret_cast<const uint8_t*>(q.data()), q.size());
        buffers_.push_back(CreateBuffer(builder_, data_buffer));
      } else if (is_quantized) {
        CHECK_EQ(t.type, TensorType_INT8)
            << "The INT8 quantization is only supported for sparsified tensor";
        auto q = Quantize<int8_t>(sparse_data, t.scale, t.zero_point);
        std::vector<float> scales{t.scale};
        std::vector<int64_t> zero_points{0};
        q_params = CreateQuantizationParameters(
            builder_, t.min, t.max, builder_.CreateVector<float>(scales),
            builder_.CreateVector<int64_t>(zero_points));
        auto data_buffer = builder_.CreateVector(
            reinterpret_cast<const uint8_t*>(q.data()), q.size());
        buffers_.push_back(CreateBuffer(builder_, data_buffer));
      } else {
        auto data_buffer = builder_.CreateVector(
            reinterpret_cast<const uint8_t*>(sparse_data.data()),
            sizeof(T) * sparse_data.size());
        buffers_.push_back(CreateBuffer(builder_, data_buffer));
      }
    }

    tensors_.push_back(
        CreateTensor(builder_, builder_.CreateVector<int>(t.shape),
                     symmetric_quantize ? TensorType_INT8 : t.type,
                     /*buffer=*/buffer_id,
                     /*name=*/0, q_params, /*is_variable=*/false, s_param));

    inputs_.push_back(id);
    tensor_data_[id] = t;

    return id;
  }

  // Add a null input tensor (optional input) and return kTfLiteOptionalTensor.
  int AddNullInput();

  // Add a TensorType output tensor and return its index.
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
      const float scale = params->scale->size == 1 ? params->scale->data[0]
                                                   : params->scale->data[i];
      scales_inv[i] = 1.0f / scale;
    }
    optimize::utils::SymmetricPerChannelQuantizeValues(
        input_data.data(), scales_inv, shape, channel_index, &quantized_output);

    PopulateTensor(index, /*offset=*/0, quantized_output.data(),
                   quantized_output.data() + quantized_output.size());
  }

  template <typename T>
  void PerChannelQuantizeBiasPopulateTensor(
      const std::vector<float>& input_data, int index,
      TfLiteAffineQuantization* params) {
    const int32_t num_inputs = input_data.size();
    std::vector<T> quantized_output(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      const float scale = params->scale->size == 1 ? params->scale->data[0]
                                                   : params->scale->data[i];
      quantized_output[i] = input_data[i] / scale;
    }
  }

  template <typename T>
  void PerChannelQuantizeBiasPopulateTensor(
      int index, const std::vector<float>& input_data,
      const TfLiteAffineQuantization* params) {
    const int32_t num_inputs = input_data.size();
    std::vector<T> quantized_output(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      const float scale = params->scale->size == 1 ? params->scale->data[0]
                                                   : params->scale->data[i];
      quantized_output[i] = input_data[i] / scale;
    }
    PopulateTensor(index, /*offset=*/0, quantized_output.data(),
                   quantized_output.data() + quantized_output.size());
  }

  // Quantize and populate data for bias with per channel quantization.
  void PerChannelQuantizeBias(int index, const std::vector<float>& input_data) {
    TfLiteTensor* t = interpreter_->tensor(index);
    auto* params =
        reinterpret_cast<TfLiteAffineQuantization*>(t->quantization.params);
    CHECK(t->type == kTfLiteInt32 || t->type == kTfLiteInt64);
    if (t->type == kTfLiteInt32) {
      PerChannelQuantizeBiasPopulateTensor<int32_t>(index, input_data, params);
    } else {
      PerChannelQuantizeBiasPopulateTensor<int64_t>(index, input_data, params);
    }
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

  // Allocate tensors and apply delegate.
  // Note that this is called by default in BuiltInterpreter().
  void AllocateAndDelegate(bool apply_delegate);

  // Build the interpreter for this model. Also, resize and allocate all
  // tensors given the shapes of the inputs.
  // Note, if `allocate_and_delegate` is `false`, then the value of
  // `apply_delegate` is ignored.
  void BuildInterpreter(std::vector<std::vector<int>> input_shapes,
                        int num_threads, bool allow_fp32_relax_to_fp16,
                        bool apply_delegate, bool allocate_and_delegate = true,
                        bool use_simple_allocator = false);

  void BuildInterpreter(std::vector<std::vector<int>> input_shapes,
                        bool use_simple_allocator = false);

  // Executes inference and return status code.
  TfLiteStatus Invoke();

  void PopulateStringTensor(int index, const std::vector<string>& content) {
    auto tensor = interpreter_->tensor(index);
    DynamicBuffer buf;
    for (const string& s : content) {
      buf.AddString(s.data(), s.length());
    }
    buf.WriteToTensor(tensor, /*new_shape=*/nullptr);
  }

  // Populates the tensor given its index.
  template <typename T>
  void PopulateTensor(int index, const std::initializer_list<T>& data) {
    PopulateTensorImpl<T>(index, /*offset=*/0, data);
  }

  // Populates the tensor given its index.
  template <typename T>
  void PopulateTensor(int index, const std::vector<T>& data) {
    PopulateTensorImpl<T>(index, /*offset=*/0, data);
  }

  // Partially populates the tensor, starting at the given offset.
  template <typename T>
  void PopulateTensor(int index, int offset, T* begin, T* end) {
    PopulateTensorImpl<T>(index, offset, absl::Span<T>(begin, end - begin));
  }

  // Return a vector with the flattened contents of a tensor.
  template <typename T>
  std::vector<T> ExtractVector(int index) const {
    const T* v = interpreter_->typed_tensor<T>(index);
    const auto* tensor = interpreter_->tensor(index);
    CHECK(v) << "Could not extract vector at index: " << index;
    int tensor_size;
    if (tensor->sparsity) {
      // Getting the size of the sparse buffer this way is based on the
      // assumption that the last dimension of the tensor is a compressed
      // dimension.
      tensor_size = tensor->sparsity
                        ->dim_metadata[tensor->sparsity->dim_metadata_size - 1]
                        .array_indices->size;
    } else {
      tensor_size = GetTensorSize(index);
    }

    return std::vector<T>(v, v + tensor_size);
  }

  // Return the TFLite model buffer, only available after BuildInterpreter.
  const uint8_t* GetModelBuffer() { return builder_.GetBufferPointer(); }

  std::vector<int> GetTensorShape(int index) {
    std::vector<int> result;
    TfLiteTensor* t = interpreter_->tensor(index);
    result.reserve(t->dims->size);
    for (int i = 0; i < t->dims->size; ++i) {
      result.push_back(t->dims->data[i]);
    }
    return result;
  }

  // Sets the number of threads available to the interpreter.
  // Reconstruct the interpreter if reset_interpreter is true.
  void SetNumThreads(int num_threads, bool reset_interpreter = false) {
    CHECK(interpreter_ != nullptr);
    if (reset_interpreter) {
      // Reconstruct interpreter as number of threads may affect internal state,
      // e.g. stratch buffer allocation.
      BuildInterpreter(input_shapes_, num_threads, allocate_and_delegate_,
                       apply_delegate_, allocate_and_delegate_);
    }
    interpreter_->SetNumThreads(num_threads);
  }

  void SetResolver(std::unique_ptr<OpResolver> resolver) {
    resolver_ = std::move(resolver);
  }

  // Indicate whether the test has the NNAPI delegate applied.
  static bool GetForceUseNnapi();
  int CountOpsExecutedByCpuKernel();
  int CountNumberOfDelegatedPartitions() const;
  int GetNumberOfAppliedDelegates() const { return num_applied_delegates_; }

 protected:
  int32_t GetTensorSize(int index) const;

  // Tell TF Lite runtime to skip applying default delegates (i.e. XNNPACK
  // delegate) when handling this op-level model.
  void SetBypassDefaultDelegates() { bypass_default_delegates_ = true; }

  flatbuffers::FlatBufferBuilder builder_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<OpResolver> resolver_;

  std::vector<flatbuffers::Offset<OperatorCode>> opcodes_;
  std::vector<flatbuffers::Offset<Operator>> operators_;
  std::map<string, std::function<TfLiteRegistration*()>> custom_registrations_;

  template <typename T>
  int AddTensor(TensorData t, const T* data, size_t size,
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

      std::vector<float> scales{t.scale};
      std::vector<int64_t> zero_points{t.zero_point};
      q_params = CreateQuantizationParameters(
          builder_, /*min=*/0, /*max=*/0, builder_.CreateVector<float>(scales),
          builder_.CreateVector<int64_t>(zero_points));
    }

    int buffer_id = 0;
    if (size) {
      // Initialize buffers list with empty buffer to allow for non-const
      // tensors.
      if (buffers_.empty()) {
        buffers_.push_back(CreateBuffer(builder_, builder_.CreateVector({})));
      }

      builder_.ForceVectorAlignment(size, sizeof(T), 16);
      // Add data as a Buffer to buffers list.
      buffer_id = buffers_.size();
      auto data_buffer = builder_.CreateVector(
          reinterpret_cast<const uint8_t*>(data), sizeof(T) * size);
      buffers_.push_back(CreateBuffer(builder_, data_buffer));
    }

    tensors_.push_back(CreateTensor(
        builder_, builder_.CreateVector<int>(t.shape), t.type,
        /*buffer=*/buffer_id,
        /*name=*/0, q_params, is_variable,
        /*sparsity=*/0, builder_.CreateVector<int>(t.shape_signature)));

    tensor_data_[id] = t;

    return id;
  }

 private:
  // Populates the tensor starting at offset using given data.
  template <typename T, typename Container>
  void PopulateTensorImpl(int index, int offset, const Container& data) {
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
    absl::c_copy(data, v + offset);
  }

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

  int AddTensorPerChannelQuant(const TensorData& t) {
    // type does not matter when adding empty data.
    return AddTensorPerChannelQuant<uint8_t>(t, nullptr, 0);
  }

  template <typename T>
  int AddTensorPerChannelQuant(const TensorData& t, const T* data,
                               size_t size) {
    const int id = tensors_.size();
    flatbuffers::Offset<QuantizationParameters> q_params = 0;
    q_params = CreateQuantizationParameters(
        builder_, /*min=*/0, /*max=*/0,
        /*scale=*/
        builder_.CreateVector<float>(t.per_channel_quantization_scales),
        /*zero point=*/
        builder_.CreateVector<int64_t>(t.per_channel_quantization_offsets),
        QuantizationDetails_NONE, 0, t.channel_index);

    int buffer_id = 0;
    if (size) {
      // Initialize buffers list with empty buffer to allow for non-const
      // tensors.
      if (buffers_.empty()) {
        buffers_.push_back(CreateBuffer(builder_, builder_.CreateVector({})));
      }

      // Add data as a Buffer to buffers list.
      buffer_id = buffers_.size();
      auto data_buffer = builder_.CreateVector(
          reinterpret_cast<const uint8_t*>(data), sizeof(T) * size);
      buffers_.push_back(CreateBuffer(builder_, data_buffer));
    }

    tensors_.push_back(
        CreateTensor(builder_, builder_.CreateVector<int>(t.shape), t.type,
                     /*buffer=*/buffer_id,
                     /*name=*/0, q_params, /*is_variable=*/false));
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
  std::vector<int32_t> intermediates_;
  std::vector<int32_t> outputs_;
  std::vector<flatbuffers::Offset<Tensor>> tensors_;
  std::vector<flatbuffers::Offset<Buffer>> buffers_;
  TfLiteDelegate* delegate_ = nullptr;  // not own the memory.
  std::vector<std::vector<int>> input_shapes_;
  int num_applied_delegates_ = 0;
  bool allow_fp32_relax_to_fp16_ = false;
  bool apply_delegate_ = true;
  bool allocate_and_delegate_ = true;

  // Whether to bypass the application of TF Lite default delegates (i.e.
  // XNNPACK delegate) at rutnime.
  bool bypass_default_delegates_ = false;
};

// Populate string tensors.
template <>
inline void SingleOpModel::PopulateTensor<string>(
    int index, const std::initializer_list<string>& data) {
  PopulateStringTensor(index, data);
}

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
  if (std::is_same<T, Eigen::half>::value) return TensorType_FLOAT16;
  if (std::is_same<T, double>::value) return TensorType_FLOAT64;
  if (std::is_same<T, int8_t>::value) return TensorType_INT8;
  if (std::is_same<T, int16_t>::value) return TensorType_INT16;
  if (std::is_same<T, uint16_t>::value) return TensorType_UINT16;
  if (std::is_same<T, int32_t>::value) return TensorType_INT32;
  if (std::is_same<T, uint32_t>::value) return TensorType_UINT32;
  if (std::is_same<T, int64_t>::value) return TensorType_INT64;
  if (std::is_same<T, uint8_t>::value) return TensorType_UINT8;
  if (std::is_same<T, string>::value) return TensorType_STRING;
  if (std::is_same<T, bool>::value) return TensorType_BOOL;
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
  // NOLINTNEXTLINE
  static constexpr TensorType tensor_type = TensorType::TensorType_FLOAT32;
  // NOLINTNEXTLINE
  static constexpr TfLiteType tflite_type = TfLiteType::kTfLiteFloat32;
  typedef float ScalarType;
};

template <>
struct TypeUnion<int32_t> {
 public:
  // NOLINTNEXTLINE
  static constexpr TensorType tensor_type = TensorType::TensorType_INT32;
  // NOLINTNEXTLINE
  static constexpr TfLiteType tflite_type = TfLiteType::kTfLiteInt32;
  typedef int32_t ScalarType;
};

template <>
struct TypeUnion<uint32_t> {
 public:
  // NOLINTNEXTLINE
  static constexpr TensorType tensor_type = TensorType::TensorType_UINT32;
  // NOLINTNEXTLINE
  static constexpr TfLiteType tflite_type = TfLiteType::kTfLiteUInt32;
  typedef uint32_t ScalarType;
};

template <>
struct TypeUnion<int16_t> {
 public:
  // NOLINTNEXTLINE
  static constexpr TensorType tensor_type = TensorType::TensorType_INT16;
  // NOLINTNEXTLINE
  static constexpr TfLiteType tflite_type = TfLiteType::kTfLiteInt16;
  typedef int16_t ScalarType;
};

template <>
struct TypeUnion<uint16_t> {
 public:
  // NOLINTNEXTLINE
  static constexpr TensorType tensor_type = TensorType::TensorType_UINT16;
  // NOLINTNEXTLINE
  static constexpr TfLiteType tflite_type = TfLiteType::kTfLiteUInt16;
  typedef uint16_t ScalarType;
};

template <>
struct TypeUnion<int8_t> {
 public:
  // NOLINTNEXTLINE
  static constexpr TensorType tensor_type = TensorType::TensorType_INT8;
  // NOLINTNEXTLINE
  static constexpr TfLiteType tflite_type = TfLiteType::kTfLiteInt8;
  typedef int8_t ScalarType;
};

template <>
struct TypeUnion<uint8_t> {
 public:
  // NOLINTNEXTLINE
  static constexpr TensorType tensor_type = TensorType::TensorType_UINT8;
  // NOLINTNEXTLINE
  static constexpr TfLiteType tflite_type = TfLiteType::kTfLiteUInt8;
  typedef uint8_t ScalarType;
};

class MultiOpModel : public SingleOpModel {
 public:
  MultiOpModel() : SingleOpModel() {}
  ~MultiOpModel() {}

  void AddBuiltinOp(BuiltinOperator type, BuiltinOptions builtin_options_type,
                    const flatbuffers::Offset<void>& builtin_options,
                    const std::vector<int32_t>& inputs,
                    const std::vector<int32_t>& outputs);

  void AddCustomOp(const string& name,
                   const std::vector<uint8_t>& custom_option,
                   const std::function<TfLiteRegistration*()>& registration,
                   const std::vector<int32_t>& inputs,
                   const std::vector<int32_t>& outputs);

  template <typename T>
  int AddInnerTensor(TensorData t) {
    return AddTensor<T>(t, {}, false);
  }
};
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_TEST_UTIL_H_
