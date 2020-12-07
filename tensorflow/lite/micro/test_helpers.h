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

#ifndef TENSORFLOW_LITE_MICRO_TEST_HELPERS_H_
#define TENSORFLOW_LITE_MICRO_TEST_HELPERS_H_

// Useful functions for writing tests.

#include <cstdint>
#include <limits>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite//kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace testing {

constexpr int kOfflinePlannerHeaderSize = 3;

struct NodeConnection_ {
  std::initializer_list<int32_t> input;
  std::initializer_list<int32_t> output;
};
typedef struct NodeConnection_ NodeConnection;

// A simple operator that returns the median of the input with the number of
// times the kernel was invoked. The implementation below is deliberately
// complicated, just to demonstrate how kernel memory planning works.
class SimpleStatefulOp {
  static constexpr int kBufferNotAllocated = 0;
  // Inputs:
  static constexpr int kInputTensor = 0;
  // Outputs:
  static constexpr int kMedianTensor = 0;
  static constexpr int kInvokeCount = 1;
  struct OpData {
    int* invoke_count = nullptr;
    int sorting_buffer = kBufferNotAllocated;
  };

 public:
  static const TfLiteRegistration* getRegistration();
  static TfLiteRegistration* GetMutableRegistration();
  static void* Init(TfLiteContext* context, const char* buffer, size_t length);
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);
};

class MockCustom {
 public:
  static const TfLiteRegistration* getRegistration();
  static TfLiteRegistration* GetMutableRegistration();
  static void* Init(TfLiteContext* context, const char* buffer, size_t length);
  static void Free(TfLiteContext* context, void* buffer);
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);

  static bool freed_;
};

// A simple operator with the purpose of testing multiple inputs. It returns
// the sum of the inputs.
class MultipleInputs {
 public:
  static const TfLiteRegistration* getRegistration();
  static TfLiteRegistration* GetMutableRegistration();
  static void* Init(TfLiteContext* context, const char* buffer, size_t length);
  static void Free(TfLiteContext* context, void* buffer);
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);

  static bool freed_;
};

// Returns an Op Resolver that can be used in the testing code.
AllOpsResolver GetOpResolver();

// Returns a simple example flatbuffer TensorFlow Lite model. Contains 1 input,
// 1 layer of weights, 1 output Tensor, and 1 operator.
const Model* GetSimpleMockModel();

// Returns a flatbuffer TensorFlow Lite model with more inputs, variable
// tensors, and operators.
const Model* GetComplexMockModel();

// Returns a simple flatbuffer model with two branches.
const Model* GetSimpleModelWithBranch();

// Returns a simple example flatbuffer TensorFlow Lite model. Contains 3 inputs,
// 1 output Tensor, and 1 operator.
const Model* GetSimpleMultipleInputsModel();

// Returns a simple flatbuffer model with offline planned tensors
// @param[in]       num_tensors           Number of tensors in the model.
// @param[in]       metadata_buffer       Metadata for offline planner.
// @param[in]       node_con              List of connections, i.e. operators
//                                        in the model.
// @param[in]       num_conns             Number of connections.
// @param[in]       num_subgraph_inputs   How many of the input tensors are in
//                                        the subgraph inputs. The default value
//                                        of 0 means all of the input tensors
//                                        are in the subgraph input list. There
//                                        must be at least 1 input tensor in the
//                                        subgraph input list.
const Model* GetModelWithOfflinePlanning(int num_tensors,
                                         const int32_t* metadata_buffer,
                                         NodeConnection* node_conn,
                                         int num_conns,
                                         int num_subgraph_inputs = 0);

// Returns a flatbuffer model with `simple_stateful_op`
const Model* GetSimpleStatefulModel();

// Builds a one-dimensional flatbuffer tensor of the given size.
const Tensor* Create1dFlatbufferTensor(int size, bool is_variable = false);

// Builds a one-dimensional flatbuffer tensor of the given size with
// quantization metadata.
const Tensor* CreateQuantizedFlatbufferTensor(int size);

// Creates a one-dimensional tensor with no quantization metadata.
const Tensor* CreateMissingQuantizationFlatbufferTensor(int size);

// Creates a vector of flatbuffer buffers.
const flatbuffers::Vector<flatbuffers::Offset<Buffer>>*
CreateFlatbufferBuffers();

// Performs a simple string comparison without requiring standard C library.
int TestStrcmp(const char* a, const char* b);

// Wrapper to forward kernel errors to the interpreter's error reporter.
void ReportOpError(struct TfLiteContext* context, const char* format, ...);

void PopulateContext(TfLiteTensor* tensors, int tensors_size,
                     TfLiteContext* context);

// Create a TfLiteIntArray from an array of ints.  The first element in the
// supplied array must be the size of the array expressed as an int.
TfLiteIntArray* IntArrayFromInts(const int* int_array);

// Create a TfLiteFloatArray from an array of floats.  The first element in the
// supplied array must be the size of the array expressed as a float.
TfLiteFloatArray* FloatArrayFromFloats(const float* floats);

template <typename T>
TfLiteTensor CreateTensor(const T* data, TfLiteIntArray* dims,
                          const bool is_variable = false) {
  TfLiteTensor result;
  result.dims = dims;
  result.params = {};
  result.quantization = {kTfLiteNoQuantization, nullptr};
  result.is_variable = is_variable;
  result.allocation_type = kTfLiteMemNone;
  result.type = typeToTfLiteType<T>();
  // Const cast is used to allow passing in const and non-const arrays within a
  // single CreateTensor method. A Const array should be used for immutable
  // input tensors and non-const array should be used for mutable and output
  // tensors.
  result.data.data = const_cast<T*>(data);
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  result.bytes = ElementCount(*dims) * sizeof(T);
  return result;
}

template <typename T>
TfLiteTensor CreateQuantizedTensor(const T* data, TfLiteIntArray* dims,
                                   const float scale, const int zero_point = 0,
                                   const bool is_variable = false) {
  TfLiteTensor result = CreateTensor(data, dims, is_variable);
  result.params = {scale, zero_point};
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  return result;
}

template <typename T>
TfLiteTensor CreateQuantizedTensor(const float* input, T* quantized,
                                   TfLiteIntArray* dims, float scale,
                                   int zero_point, bool is_variable = false) {
  int input_size = ElementCount(*dims);
  tflite::Quantize(input, quantized, input_size, scale, zero_point);
  return CreateQuantizedTensor(quantized, dims, scale, zero_point, is_variable);
}

TfLiteTensor CreateQuantizedBiasTensor(const float* data, int32_t* quantized,
                                       TfLiteIntArray* dims, float input_scale,
                                       float weights_scale,
                                       bool is_variable = false);

// Quantizes int32_t bias tensor with per-channel weights determined by input
// scale multiplied by weight scale for each channel.
TfLiteTensor CreatePerChannelQuantizedBiasTensor(
    const float* input, int32_t* quantized, TfLiteIntArray* dims,
    float input_scale, float* weight_scales, float* scales, int* zero_points,
    TfLiteAffineQuantization* affine_quant, int quantized_dimension,
    bool is_variable = false);

TfLiteTensor CreateSymmetricPerChannelQuantizedTensor(
    const float* input, int8_t* quantized, TfLiteIntArray* dims, float* scales,
    int* zero_points, TfLiteAffineQuantization* affine_quant,
    int quantized_dimension, bool is_variable = false);

// Returns the number of tensors in the default subgraph for a tflite::Model.
size_t GetModelTensorCount(const Model* model);

// Derives the quantization scaling factor from a min and max range.
template <typename T>
inline float ScaleFromMinMax(const float min, const float max) {
  return (max - min) /
         static_cast<float>((std::numeric_limits<T>::max() * 1.0) -
                            std::numeric_limits<T>::min());
}

// Derives the quantization zero point from a min and max range.
template <typename T>
inline int ZeroPointFromMinMax(const float min, const float max) {
  return static_cast<int>(std::numeric_limits<T>::min()) +
         static_cast<int>(-min / ScaleFromMinMax<T>(min, max) + 0.5f);
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TEST_HELPERS_H_
