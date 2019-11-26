/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/micro/test_helpers.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/experimental/micro/micro_utils.h"

namespace tflite {
namespace testing {
namespace {

class StackAllocator : public flatbuffers::Allocator {
 public:
  StackAllocator() : data_(data_backing_), data_size_(0) {}

  uint8_t* allocate(size_t size) override {
    if ((data_size_ + size) > kStackAllocatorSize) {
      // TODO(petewarden): Add error reporting beyond returning null!
      return nullptr;
    }
    uint8_t* result = data_;
    data_ += size;
    data_size_ += size;
    return result;
  }

  void deallocate(uint8_t* p, size_t) override {}

  static StackAllocator& instance() {
    // Avoid using true dynamic memory allocation to be portable to bare metal.
    static char inst_memory[sizeof(StackAllocator)];
    static StackAllocator* inst = new (inst_memory) StackAllocator;
    return *inst;
  }

  static constexpr int kStackAllocatorSize = 4096;

 private:
  uint8_t data_backing_[kStackAllocatorSize];
  uint8_t* data_;
  int data_size_;
};

flatbuffers::FlatBufferBuilder* BuilderInstance() {
  static char inst_memory[sizeof(flatbuffers::FlatBufferBuilder)];
  static flatbuffers::FlatBufferBuilder* inst =
      new (inst_memory) flatbuffers::FlatBufferBuilder(
          StackAllocator::kStackAllocatorSize, &StackAllocator::instance());
  return inst;
}

const Model* BuildMockModel() {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();

  constexpr size_t buffer_data_size = 1;
  const uint8_t buffer_data[buffer_data_size] = {21};
  constexpr size_t buffers_size = 2;
  const Offset<Buffer> buffers[buffers_size] = {
      CreateBuffer(*builder),
      CreateBuffer(*builder,
                   builder->CreateVector(buffer_data, buffer_data_size))};
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {1};
  constexpr size_t tensors_size = 3;
  const Offset<Tensor> tensors[tensors_size] = {
      CreateTensor(*builder,
                   builder->CreateVector(tensor_shape, tensor_shape_size),
                   TensorType_INT32, 0,
                   builder->CreateString("test_input_tensor"), 0, false),
      CreateTensor(*builder,
                   builder->CreateVector(tensor_shape, tensor_shape_size),
                   TensorType_UINT8, 1,
                   builder->CreateString("test_weight_tensor"), 0, false),
      CreateTensor(*builder,
                   builder->CreateVector(tensor_shape, tensor_shape_size),
                   TensorType_INT32, 0,
                   builder->CreateString("test_output_tensor"), 0, false),
  };
  constexpr size_t inputs_size = 1;
  const int32_t inputs[inputs_size] = {0};
  constexpr size_t outputs_size = 1;
  const int32_t outputs[outputs_size] = {2};
  constexpr size_t operator_inputs_size = 2;
  const int32_t operator_inputs[operator_inputs_size] = {0, 1};
  constexpr size_t operator_outputs_size = 1;
  const int32_t operator_outputs[operator_outputs_size] = {2};
  constexpr size_t operators_size = 1;
  const Offset<Operator> operators[operators_size] = {CreateOperator(
      *builder, 0, builder->CreateVector(operator_inputs, operator_inputs_size),
      builder->CreateVector(operator_outputs, operator_outputs_size),
      BuiltinOptions_NONE)};
  constexpr size_t subgraphs_size = 1;
  const Offset<SubGraph> subgraphs[subgraphs_size] = {
      CreateSubGraph(*builder, builder->CreateVector(tensors, tensors_size),
                     builder->CreateVector(inputs, inputs_size),
                     builder->CreateVector(outputs, outputs_size),
                     builder->CreateVector(operators, operators_size),
                     builder->CreateString("test_subgraph"))};
  constexpr size_t operator_codes_size = 1;
  const Offset<OperatorCode> operator_codes[operator_codes_size] = {
      CreateOperatorCodeDirect(*builder, BuiltinOperator_CUSTOM, "mock_custom",
                               0)};
  const Offset<Model> model_offset = CreateModel(
      *builder, 0, builder->CreateVector(operator_codes, operator_codes_size),
      builder->CreateVector(subgraphs, subgraphs_size),
      builder->CreateString("test_model"),
      builder->CreateVector(buffers, buffers_size));
  FinishModelBuffer(*builder, model_offset);
  void* model_pointer = builder->GetBufferPointer();
  const Model* model = flatbuffers::GetRoot<Model>(model_pointer);
  return model;
}

}  // namespace

const Model* GetMockModel() {
  static Model* model = nullptr;
  if (!model) {
    model = const_cast<Model*>(BuildMockModel());
  }
  return model;
}

const Tensor* Create1dFlatbufferTensor(int size) {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {size};
  const Offset<Tensor> tensor_offset = CreateTensor(
      *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
      TensorType_INT32, 0, builder->CreateString("test_tensor"), 0, false);
  builder->Finish(tensor_offset);
  void* tensor_pointer = builder->GetBufferPointer();
  const Tensor* tensor = flatbuffers::GetRoot<Tensor>(tensor_pointer);
  return tensor;
}

const Tensor* CreateMissingQuantizationFlatbufferTensor(int size) {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();
  const Offset<QuantizationParameters> quant_params =
      CreateQuantizationParameters(*builder, 0, 0, 0, 0,
                                   QuantizationDetails_NONE, 0, 0);
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {size};
  const Offset<Tensor> tensor_offset = CreateTensor(
      *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
      TensorType_INT32, 0, builder->CreateString("test_tensor"), quant_params,
      false);
  builder->Finish(tensor_offset);
  void* tensor_pointer = builder->GetBufferPointer();
  const Tensor* tensor = flatbuffers::GetRoot<Tensor>(tensor_pointer);
  return tensor;
}

const flatbuffers::Vector<flatbuffers::Offset<Buffer>>*
CreateFlatbufferBuffers() {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();
  constexpr size_t buffers_size = 1;
  const Offset<Buffer> buffers[buffers_size] = {
      CreateBuffer(*builder),
  };
  const flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
      buffers_offset = builder->CreateVector(buffers, buffers_size);
  builder->Finish(buffers_offset);
  void* buffers_pointer = builder->GetBufferPointer();
  const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* result =
      flatbuffers::GetRoot<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>(
          buffers_pointer);
  return result;
}

int TestStrcmp(const char* a, const char* b) {
  if ((a == nullptr) || (b == nullptr)) {
    return -1;
  }
  while ((*a != 0) && (*a == *b)) {
    a++;
    b++;
  }
  return *reinterpret_cast<const unsigned char*>(a) -
         *reinterpret_cast<const unsigned char*>(b);
}

// Wrapper to forward kernel errors to the interpreter's error reporter.
void ReportOpError(struct TfLiteContext* context, const char* format, ...) {
  ErrorReporter* error_reporter = static_cast<ErrorReporter*>(context->impl_);
  va_list args;
  va_start(args, format);
  error_reporter->Report(format, args);
  va_end(args);
}

// Create a TfLiteIntArray from an array of ints.  The first element in the
// supplied array must be the size of the array expressed as an int.
TfLiteIntArray* IntArrayFromInts(const int* int_array) {
  return const_cast<TfLiteIntArray*>(
      reinterpret_cast<const TfLiteIntArray*>(int_array));
}

// Create a TfLiteFloatArray from an array of floats.  The first element in the
// supplied array must be the size of the array expressed as a float.
TfLiteFloatArray* FloatArrayFromFloats(const float* floats) {
  static_assert(sizeof(float) == sizeof(int),
                "assumes sizeof(float) == sizeof(int) to perform casting");
  int size = static_cast<int>(floats[0]);
  *reinterpret_cast<int32_t*>(const_cast<float*>(floats)) = size;
  return reinterpret_cast<TfLiteFloatArray*>(const_cast<float*>(floats));
}

TfLiteTensor CreateTensor(TfLiteIntArray* dims, const char* name,
                          bool is_variable) {
  TfLiteTensor result;
  result.dims = dims;
  result.name = name;
  result.params = {};
  result.quantization = {kTfLiteNoQuantization, nullptr};
  result.is_variable = is_variable;
  result.allocation_type = kTfLiteMemNone;
  result.allocation = nullptr;
  return result;
}

TfLiteTensor CreateFloatTensor(const float* data, TfLiteIntArray* dims,
                               const char* name, bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, name, is_variable);
  result.type = kTfLiteFloat32;
  result.data.f = const_cast<float*>(data);
  result.bytes = ElementCount(*dims) * sizeof(float);
  return result;
}

void PopulateFloatTensor(TfLiteTensor* tensor, float* begin, float* end) {
  float* p = begin;
  float* v = tensor->data.f;
  while (p != end) {
    *v++ = *p++;
  }
}

TfLiteTensor CreateBoolTensor(const bool* data, TfLiteIntArray* dims,
                              const char* name, bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, name, is_variable);
  result.type = kTfLiteBool;
  result.data.b = const_cast<bool*>(data);
  result.bytes = ElementCount(*dims) * sizeof(bool);
  return result;
}

TfLiteTensor CreateInt32Tensor(const int32_t* data, TfLiteIntArray* dims,
                               const char* name, bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, name, is_variable);
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(data);
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  return result;
}

TfLiteTensor CreateQuantizedTensor(const uint8_t* data, TfLiteIntArray* dims,
                                   float scale, int zero_point,
                                   const char* name, bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, name, is_variable);
  result.type = kTfLiteUInt8;
  result.data.uint8 = const_cast<uint8_t*>(data);
  result.params = {scale, zero_point};
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  result.bytes = ElementCount(*dims) * sizeof(uint8_t);
  return result;
}

// Create Quantized tensor which contains a quantized version of the supplied
// buffer.
TfLiteTensor CreateQuantizedTensor(const float* input, uint8_t* quantized,
                                   TfLiteIntArray* dims, float scale,
                                   int zero_point, const char* name,
                                   bool is_variable) {
  int input_size = ElementCount(*dims);
  tflite::AsymmetricQuantize(input, quantized, input_size, scale, zero_point);
  return CreateQuantizedTensor(quantized, dims, scale, zero_point, name);
}

TfLiteTensor CreateQuantizedTensor(const int8_t* data, TfLiteIntArray* dims,
                                   float scale, int zero_point,
                                   const char* name, bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, name, is_variable);
  result.type = kTfLiteInt8;
  result.data.int8 = const_cast<int8_t*>(data);
  result.params = {scale, zero_point};
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  return result;
}

TfLiteTensor CreateQuantizedTensor(const float* input, int8_t* quantized,
                                   TfLiteIntArray* dims, float scale,
                                   int zero_point, const char* name,
                                   bool is_variable) {
  int input_size = ElementCount(*dims);
  tflite::AsymmetricQuantize(input, quantized, input_size, scale, zero_point);
  return CreateQuantizedTensor(quantized, dims, scale, zero_point, name);
}

TfLiteTensor CreateQuantized32Tensor(const int32_t* data, TfLiteIntArray* dims,
                                     float scale, const char* name,
                                     bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, name, is_variable);
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(data);
  // Quantized int32 tensors always have a zero point of 0, since the range of
  // int32 values is large, and because zero point costs extra cycles during
  // processing.
  result.params = {scale, 0};
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  return result;
}

TfLiteTensor CreateQuantizedBiasTensor(const float* data, int32_t* quantized,
                                       TfLiteIntArray* dims, float input_scale,
                                       float weights_scale, const char* name,
                                       bool is_variable) {
  float bias_scale = input_scale * weights_scale;
  tflite::SymmetricQuantize(data, quantized, ElementCount(*dims), bias_scale);
  return CreateQuantized32Tensor(quantized, dims, bias_scale, name,
                                 is_variable);
}

// Quantizes int32 bias tensor with per-channel weights determined by input
// scale multiplied by weight scale for each channel.
TfLiteTensor CreatePerChannelQuantizedBiasTensor(
    const float* input, int32_t* quantized, TfLiteIntArray* dims,
    float input_scale, float* weight_scales, float* scales, int* zero_points,
    TfLiteAffineQuantization* affine_quant, int quantized_dimension,
    const char* name, bool is_variable) {
  int input_size = ElementCount(*dims);
  int num_channels = dims->data[quantized_dimension];
  // First element is reserved for array length
  zero_points[0] = num_channels;
  scales[0] = static_cast<float>(num_channels);
  float* scales_array = &scales[1];
  for (int i = 0; i < num_channels; i++) {
    scales_array[i] = input_scale * weight_scales[i];
    zero_points[i + 1] = 0;
  }

  SymmetricPerChannelQuantize(input, quantized, input_size, num_channels,
                              scales_array);

  affine_quant->scale = FloatArrayFromFloats(scales);
  affine_quant->zero_point = IntArrayFromInts(zero_points);
  affine_quant->quantized_dimension = quantized_dimension;

  TfLiteTensor result = CreateTensor(dims, name, is_variable);
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(quantized);
  result.quantization = {kTfLiteAffineQuantization, affine_quant};
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  return result;
}

TfLiteTensor CreateSymmetricPerChannelQuantizedTensor(
    const float* input, int8_t* quantized, TfLiteIntArray* dims, float* scales,
    int* zero_points, TfLiteAffineQuantization* affine_quant,
    int quantized_dimension, const char* name, bool is_variable) {
  int channel_count = dims->data[quantized_dimension];
  scales[0] = static_cast<float>(channel_count);
  zero_points[0] = channel_count;

  SignedSymmetricPerChannelQuantize(input, dims, quantized_dimension, quantized,
                                    &scales[1]);

  affine_quant->scale = FloatArrayFromFloats(scales);
  affine_quant->zero_point = IntArrayFromInts(zero_points);
  affine_quant->quantized_dimension = quantized_dimension;

  TfLiteTensor result = CreateTensor(dims, name, is_variable);
  result.type = kTfLiteInt8;
  result.data.int8 = const_cast<int8_t*>(quantized);
  result.quantization = {kTfLiteAffineQuantization, affine_quant};
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  return result;
}

}  // namespace testing
}  // namespace tflite
