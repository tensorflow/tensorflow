/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/recording_micro_allocator.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_conv_model.h"

#define TF_LITE_TENSOR_STRUCT_SIZE sizeof(TfLiteTensor)
#define TF_LITE_AFFINE_QUANTIZATION_SIZE sizeof(TfLiteAffineQuantization)
#define NODE_AND_REGISTRATION_STRUCT_SIZE sizeof(tflite::NodeAndRegistration)

// TODO(b/158303868): Move tests into anonymous namespace.
namespace {

constexpr int kTestConvArenaSize = 1024 * 12;

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestRecordsTfLiteTensorArrayData) {
  TfLiteContext context;
  tflite::AllOpsResolver all_ops_resolver;
  tflite::NodeAndRegistration* node_and_registration;
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize,
                                              micro_test::reporter);
  TF_LITE_MICRO_EXPECT_NE(nullptr, micro_allocator);
  TF_LITE_MICRO_EXPECT_GE(kTfLiteOk, micro_allocator->StartModelAllocation(
                                         model, &context, all_ops_resolver,
                                         &node_and_registration));
  TF_LITE_MICRO_EXPECT_GE(
      kTfLiteOk, micro_allocator->FinishModelAllocation(model, &context));

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kTfLiteTensorArray);
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.count, context.tensors_size);
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.requested_bytes,
                          context.tensors_size * TF_LITE_TENSOR_STRUCT_SIZE);
  TF_LITE_MICRO_EXPECT_GE(recorded_allocation.used_bytes,
                          context.tensors_size * TF_LITE_TENSOR_STRUCT_SIZE);
}

TF_LITE_MICRO_TEST(TestRecordsTensorArrayQuantizationData) {
  TfLiteContext context;
  tflite::AllOpsResolver all_ops_resolver;
  tflite::NodeAndRegistration* node_and_registration;
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize,
                                              micro_test::reporter);
  TF_LITE_MICRO_EXPECT_NE(nullptr, micro_allocator);
  TF_LITE_MICRO_EXPECT_GE(kTfLiteOk, micro_allocator->StartModelAllocation(
                                         model, &context, all_ops_resolver,
                                         &node_and_registration));
  TF_LITE_MICRO_EXPECT_GE(
      kTfLiteOk, micro_allocator->FinishModelAllocation(model, &context));

  // Walk the model subgraph to find all tensors with quantization params and
  // keep a tally.
  size_t quantized_tensor_count = 0;
  size_t quantized_channel_bytes = 0;
  for (size_t i = 0; i < context.tensors_size; ++i) {
    const tflite::Tensor* cur_tensor =
        model->subgraphs()->Get(0)->tensors()->Get(i);
    const tflite::QuantizationParameters* quantization_params =
        cur_tensor->quantization();
    if (quantization_params && quantization_params->scale() &&
        quantization_params->scale()->size() > 0 &&
        quantization_params->zero_point() &&
        quantization_params->zero_point()->size() > 0) {
      quantized_tensor_count++;
      size_t num_channels = quantization_params->scale()->size();
      quantized_channel_bytes += TfLiteIntArrayGetSizeInBytes(num_channels);
    }
  }

  // Calculate the expected allocation bytes with subgraph quantization data:
  size_t expected_requested_bytes =
      quantized_tensor_count * TF_LITE_AFFINE_QUANTIZATION_SIZE +
      quantized_channel_bytes;

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kTfLiteTensorArrayQuantizationData);

  // Each quantized tensors has 2 mallocs (quant struct, zero point dimensions):
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.count,
                          quantized_tensor_count * 2);
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.requested_bytes,
                          expected_requested_bytes);
  TF_LITE_MICRO_EXPECT_GE(recorded_allocation.used_bytes,
                          expected_requested_bytes);
}

TF_LITE_MICRO_TEST(TestRecordsNodeAndRegistrationArrayData) {
  TfLiteContext context;
  tflite::AllOpsResolver all_ops_resolver;
  tflite::NodeAndRegistration* node_and_registration;
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize,
                                              micro_test::reporter);
  TF_LITE_MICRO_EXPECT_NE(nullptr, micro_allocator);
  TF_LITE_MICRO_EXPECT_GE(kTfLiteOk, micro_allocator->StartModelAllocation(
                                         model, &context, all_ops_resolver,
                                         &node_and_registration));
  TF_LITE_MICRO_EXPECT_GE(
      kTfLiteOk, micro_allocator->FinishModelAllocation(model, &context));

  size_t num_ops = model->subgraphs()->Get(0)->operators()->size();
  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kNodeAndRegistrationArray);
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.count, num_ops);
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.requested_bytes,
                          num_ops * NODE_AND_REGISTRATION_STRUCT_SIZE);
  TF_LITE_MICRO_EXPECT_GE(recorded_allocation.used_bytes,
                          num_ops * NODE_AND_REGISTRATION_STRUCT_SIZE);
}

// TODO(b/158124094): Find a way to audit OpData allocations on
// cross-architectures.

TF_LITE_MICRO_TESTS_END
