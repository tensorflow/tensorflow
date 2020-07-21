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

#include <cstdint>
#include <chrono>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

static const int number_of_invocations = 1000;

MockAllocator *mock_allocator;
TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* ctx, size_t bytes, void** ptr)
{
	return mock_allocator->AllocatePersistentBuffer(ctx, bytes, ptr);
}

void InitTestDataQuantized(const int accum_depth, const int output_depth,const int batches,
    int8_t* input_data, const float input_min, const float input_max,
	int8_t* weights_data, const float weights_min, const float weights_max,
    int32_t* bias_data, const float bias_scale,
	int8_t* expected_output_data, const float output_min, const float output_max) {

	  using tflite::testing::F2Q32;
	  using tflite::testing::F2QS;
	  for(int b = 0; b < batches; b++) {
		  for(int out_c = 0; out_c < output_depth; out_c++) {
			  float acc = 0;
			  for(int d = 0; d < accum_depth; d++) {
				  float input_value = d;
				  float weight_value = d;
				  acc += input_value * weight_value;
				  input_data[b * accum_depth + d] = F2QS(input_value, input_min, input_max);
				  weights_data[out_c * accum_depth + d] = F2QS(weight_value, weights_min, weights_max);
			  }
			  float bias_value = out_c;
			  bias_data[out_c] = F2Q32(bias_value, bias_scale);
			  float output_value = acc + bias_value;
			  expected_output_data[out_c + output_depth * b] = F2QS(output_value, output_min, output_max);
		  }
	  }
}

void InitTestDataQuantized(const int accum_depth, const int output_depth,const int batches,
	uint8_t* input_data, const float input_min, const float input_max,
	uint8_t* weights_data, const float weights_min, const float weights_max,
    int32_t* bias_data, const float bias_scale,
	uint8_t* expected_output_data, const float output_min, const float output_max) {

	  using tflite::testing::F2Q32;
	  using tflite::testing::F2Q;
	  for(int b = 0; b < batches; b++) {
		  for(int out_c = 0; out_c < output_depth; out_c++) {
			  float acc = 0;
			  for(int d = 0; d < accum_depth; d++) {
				  float input_value = d;
				  float weight_value = d;
				  acc += input_value * weight_value;
				  input_data[b * accum_depth + d] = F2Q(input_value, input_min, input_max);
				  weights_data[out_c * accum_depth + d] = F2Q(weight_value, weights_min, weights_max);
			  }
			  float bias_value = out_c;
			  bias_data[out_c] = F2Q32(bias_value, bias_scale);
			  float output_value = acc + bias_value;
			  expected_output_data[out_c + output_depth * b] = F2Q(output_value, output_min, output_max);
		  }
	  }
}

template <typename T>
void TestFullyConnectedQuantized(
    const int* input_dims_data, const T* input_data, const float input_min,
    const float input_max, const int* weights_dims_data, const T* weights_data,
    const float weights_min, const float weights_max, const int* bias_dims_data,
    const int32_t* bias_data, const float bias_scale,
    const T* expected_output_data, const int* output_dims_data,
    const float output_min, const float output_max,
    TfLiteFusedActivation activation, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, input_min,
                            input_max),
      CreateQuantizedTensor(weights_data, weights_dims, 
                            weights_min, weights_max),
      CreateQuantized32Tensor(bias_data, bias_dims, bias_scale),
      CreateQuantizedTensor(output_data, output_dims, 
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  const size_t buffer_arena_size = 1024;
  const size_t bytes_alignment = 4;
  uint8_t buffer_arena[buffer_arena_size];
  mock_allocator = new MockAllocator(buffer_arena, buffer_arena_size, bytes_alignment);
  context.AllocatePersistentBuffer = AllocatePersistentBuffer;

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_FULLY_CONNECTED);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteFullyConnectedParams builtin_data = {
      activation,
      kTfLiteFullyConnectedWeightsFormatDefault,
  };
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  auto start = std::chrono::high_resolution_clock::now();

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  //Perform inference several times after preparation is done
  for (int n = 0; n < number_of_invocations; n++) {
	  registration->invoke(&context, &node);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  micro_test::reporter->Report("Avg Invoke run time %d invocations =  %d us", number_of_invocations, duration);

  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(PerformanceTestQuantizedInt8) {

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int accum_depth = 128;
  const int output_depth = 64;
  const int batches = 32;

  //Init input data
  const int input_dims_data[] = {2, batches, accum_depth};
  int8_t input_data[batches * accum_depth];

  const int weights_dims_data[] = {2, output_depth, accum_depth};
  int8_t weights_data[output_depth * accum_depth];

  const int bias_dims_data[] = {1, output_depth};
  int32_t bias_data[output_depth];

  const int output_dims_data[] = {2, batches, output_depth};
  const int output_dims_count = batches * output_depth;
  int8_t expected_output_data[output_dims_count];

  tflite::testing::InitTestDataQuantized(
		  accum_depth, output_depth, batches,
		  input_data, input_min, input_max,
		  weights_data, weights_min, weights_max,
		  bias_data, bias_scale,
		  expected_output_data, output_min, output_max);

  int8_t output_data[output_dims_count];
  const int number_of_invocations = 1000;
  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}


TF_LITE_MICRO_TEST(PerformanceTestQuantizedUint8) {

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int accum_depth = 128;
  const int output_depth = 64;
  const int batches = 32;

  //Init input data
  const int input_dims_data[] = {2, batches, accum_depth};
  uint8_t input_data[batches * accum_depth];

  const int weights_dims_data[] = {2, output_depth, accum_depth};
  uint8_t weights_data[output_depth * accum_depth];

  const int bias_dims_data[] = {1, output_depth};
  int32_t bias_data[output_depth];

  const int output_dims_data[] = {2, batches, output_depth};
  const int output_dims_count = batches * output_depth;
  uint8_t expected_output_data[output_dims_count];

  tflite::testing::InitTestDataQuantized(
		  accum_depth, output_depth, batches,
		  input_data, input_min, input_max,
		  weights_data, weights_min, weights_max,
		  bias_data, bias_scale,
		  expected_output_data, output_min, output_max);

  uint8_t output_data[output_dims_count];
  const int number_of_invocations = 100;
  tflite::testing::TestFullyConnectedQuantized<uint8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}


TF_LITE_MICRO_TESTS_END
