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
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_conv_model.h"

#define TF_LITE_TENSOR_STRUCT_SIZE sizeof(TfLiteTensor)
#define TF_LITE_EVAL_TENSOR_STRUCT_SIZE sizeof(TfLiteEvalTensor)
#define TF_LITE_AFFINE_QUANTIZATION_SIZE sizeof(TfLiteAffineQuantization)
#define NODE_AND_REGISTRATION_STRUCT_SIZE sizeof(tflite::NodeAndRegistration)

// TODO(b/158303868): Move tests into anonymous namespace.
namespace {

constexpr int kTestConvArenaSize = 1024 * 12;

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestRecordsTfLiteEvalTensorArrayData) {
  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::AllOpsResolver all_ops_resolver;
  tflite::NodeAndRegistration* node_and_registration;
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize,
                                              tflite::GetMicroErrorReporter());
  // TODO(b/158102673): ugly workaround for not having fatal assertions. Same
  // throughout this file.
  TF_LITE_MICRO_EXPECT_NE(micro_allocator, nullptr);
  if (micro_allocator == nullptr) return 1;

  TfLiteStatus status;
  status = micro_allocator->StartModelAllocation(
      model, all_ops_resolver, &node_and_registration, &eval_tensors);
  TF_LITE_MICRO_EXPECT_EQ(status, kTfLiteOk);
  if (status != kTfLiteOk) return 1;

  status = micro_allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles);
  TF_LITE_MICRO_EXPECT_EQ(status, kTfLiteOk);
  if (status != kTfLiteOk) return 1;

  micro_allocator->PrintAllocations();

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kTfLiteEvalTensorData);

  micro_allocator->PrintAllocations();

  size_t tensors_count = tflite::testing::GetModelTensorCount(model);

  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.count, tensors_count);
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.requested_bytes,
                          tensors_count * TF_LITE_EVAL_TENSOR_STRUCT_SIZE);
  TF_LITE_MICRO_EXPECT_GE(recorded_allocation.used_bytes,
                          tensors_count * TF_LITE_EVAL_TENSOR_STRUCT_SIZE);
}

TF_LITE_MICRO_TEST(TestRecordsNodeAndRegistrationArrayData) {
  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::AllOpsResolver all_ops_resolver;
  tflite::NodeAndRegistration* node_and_registration;
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize,
                                              tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(micro_allocator, nullptr);
  if (micro_allocator == nullptr) return 1;

  TfLiteStatus status;
  status = micro_allocator->StartModelAllocation(
      model, all_ops_resolver, &node_and_registration, &eval_tensors);
  TF_LITE_MICRO_EXPECT_EQ(status, kTfLiteOk);
  if (status != kTfLiteOk) return 1;

  status = micro_allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles);
  TF_LITE_MICRO_EXPECT_EQ(status, kTfLiteOk);
  if (status != kTfLiteOk) return 1;

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

TF_LITE_MICRO_TEST(TestRecordsMultiTenantAllocations) {
  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::AllOpsResolver all_ops_resolver;
  tflite::NodeAndRegistration* node_and_registration;
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);

  // Double the arena size to allocate two models inside of it:
  uint8_t arena[kTestConvArenaSize * 2];

  TfLiteStatus status;

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize * 2,
                                              tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(micro_allocator, nullptr);
  if (micro_allocator == nullptr) return 1;

  // First allocation with the model in the arena:
  status = micro_allocator->StartModelAllocation(
      model, all_ops_resolver, &node_and_registration, &eval_tensors);
  TF_LITE_MICRO_EXPECT_EQ(status, kTfLiteOk);
  if (status != kTfLiteOk) return 1;

  status = micro_allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles);
  TF_LITE_MICRO_EXPECT_EQ(status, kTfLiteOk);
  if (status != kTfLiteOk) return 1;

  // Second allocation with the same model in the arena:
  status = micro_allocator->StartModelAllocation(
      model, all_ops_resolver, &node_and_registration, &eval_tensors);
  TF_LITE_MICRO_EXPECT_EQ(status, kTfLiteOk);
  if (status != kTfLiteOk) return 1;

  status = kTfLiteOk, micro_allocator->FinishModelAllocation(
                          model, eval_tensors, &scratch_buffer_handles);
  TF_LITE_MICRO_EXPECT_EQ(status, kTfLiteOk);
  if (status != kTfLiteOk) return 1;

  size_t tensors_count = tflite::testing::GetModelTensorCount(model);

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kTfLiteEvalTensorData);
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.count, tensors_count * 2);
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.requested_bytes,
                          tensors_count * TF_LITE_EVAL_TENSOR_STRUCT_SIZE * 2);
  TF_LITE_MICRO_EXPECT_GE(recorded_allocation.used_bytes,
                          tensors_count * TF_LITE_EVAL_TENSOR_STRUCT_SIZE * 2);
}

TF_LITE_MICRO_TEST(TestRecordsPersistentTfLiteTensorData) {
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize,
                                              tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(micro_allocator, nullptr);
  if (micro_allocator == nullptr) return 1;

  TfLiteTensor* tensor = micro_allocator->AllocatePersistentTfLiteTensor(
      model, /*eval_tensors=*/nullptr, 0);
  TF_LITE_MICRO_EXPECT_NE(tensor, nullptr);
  if (tensor == nullptr) return 1;

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kPersistentTfLiteTensorData);

  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.count, static_cast<size_t>(1));
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.requested_bytes,
                          TF_LITE_TENSOR_STRUCT_SIZE);
  TF_LITE_MICRO_EXPECT_GE(recorded_allocation.used_bytes,
                          TF_LITE_TENSOR_STRUCT_SIZE);
}

TF_LITE_MICRO_TEST(TestRecordsPersistentTfLiteTensorQuantizationData) {
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize,
                                              tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(micro_allocator, nullptr);
  if (micro_allocator == nullptr) return 1;

  TfLiteTensor* tensor = micro_allocator->AllocatePersistentTfLiteTensor(
      model, /*eval_tensors=*/nullptr, 0);
  TF_LITE_MICRO_EXPECT_NE(tensor, nullptr);
  if (tensor == nullptr) return 1;

  // Walk the model subgraph to find all tensors with quantization params and
  // keep a tally.
  size_t quantized_channel_bytes = 0;
  const tflite::Tensor* cur_tensor =
      model->subgraphs()->Get(0)->tensors()->Get(0);
  const tflite::QuantizationParameters* quantization_params =
      cur_tensor->quantization();
  if (quantization_params && quantization_params->scale() &&
      quantization_params->scale()->size() > 0 &&
      quantization_params->zero_point() &&
      quantization_params->zero_point()->size() > 0) {
    size_t num_channels = quantization_params->scale()->size();
    quantized_channel_bytes += TfLiteIntArrayGetSizeInBytes(num_channels);
  }

  // Calculate the expected allocation bytes with subgraph quantization data:
  size_t expected_requested_bytes =
      TF_LITE_AFFINE_QUANTIZATION_SIZE + quantized_channel_bytes;

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::
              kPersistentTfLiteTensorQuantizationData);

  // Each quantized tensors has 2 mallocs (quant struct, zero point dimensions):
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.count, static_cast<size_t>(2));
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.requested_bytes,
                          expected_requested_bytes);
  TF_LITE_MICRO_EXPECT_GE(recorded_allocation.used_bytes,
                          expected_requested_bytes);
}

TF_LITE_MICRO_TEST(TestRecordsPersistentBufferData) {
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize,
                                              tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(micro_allocator, nullptr);
  if (micro_allocator == nullptr) return 1;

  void* buffer = micro_allocator->AllocatePersistentBuffer(/*bytes=*/100);
  TF_LITE_MICRO_EXPECT_NE(buffer, nullptr);
  if (buffer == nullptr) return 1;

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kPersistentBufferData);

  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.count, static_cast<size_t>(1));
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.requested_bytes,
                          static_cast<size_t>(100));
  TF_LITE_MICRO_EXPECT_GE(recorded_allocation.used_bytes,
                          static_cast<size_t>(100));

  buffer = micro_allocator->AllocatePersistentBuffer(/*bytes=*/50);
  TF_LITE_MICRO_EXPECT_NE(buffer, nullptr);
  if (buffer == nullptr) return 1;

  recorded_allocation = micro_allocator->GetRecordedAllocation(
      tflite::RecordedAllocationType::kPersistentBufferData);

  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.count, static_cast<size_t>(2));
  TF_LITE_MICRO_EXPECT_EQ(recorded_allocation.requested_bytes,
                          static_cast<size_t>(150));
  TF_LITE_MICRO_EXPECT_GE(recorded_allocation.used_bytes,
                          static_cast<size_t>(150));
}

// TODO(b/158124094): Find a way to audit OpData allocations on
// cross-architectures.

TF_LITE_MICRO_TESTS_END
