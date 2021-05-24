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

#include <stdint.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_conv_model.h"

/**
 * Tests to ensure arena memory allocation does not regress by more than 3%.
 */

namespace {

// Ensure memory doesn't expand more that 3%:
constexpr float kAllocationThreshold = 0.03;

// TODO(b/160617245): Record persistent allocations to provide a more accurate
// number here.
constexpr float kAllocationTailMiscCeiling = 2 * 1024;

const bool kIs64BitSystem = (sizeof(void*) == 8);

constexpr int kKeywordModelTensorArenaSize = 22 * 1024;
uint8_t keyword_model_tensor_arena[kKeywordModelTensorArenaSize];

constexpr int kKeywordModelTensorCount = 54;
constexpr int kKeywordModelNodeAndRegistrationCount = 15;

// NOTE: These values are measured on x86-64:
// TODO(b/158651472): Consider auditing these values on non-64 bit systems.
//
// Run this test with '--copt=-DTF_LITE_STATIC_MEMORY' to get optimized memory
// runtime values:
#ifdef TF_LITE_STATIC_MEMORY
constexpr int kKeywordModelTotalSize = 14416;
constexpr int kKeywordModelTailSize = 13744;
constexpr int kKeywordModelPersistentTfLiteTensorDataSize = 128;
constexpr int kKeywordModelPersistentBufferDataSize = 564;
#else
constexpr int kKeywordModelTotalSize = 14992;
constexpr int kKeywordModelTailSize = 14320;
constexpr int kKeywordModelPersistentTfLiteTensorDataSize = 224;
constexpr int kKeywordModelPersistentBufferDataSize = 564;
#endif
constexpr int kKeywordModelHeadSize = 672;
constexpr int kKeywordModelTfLiteTensorVariableBufferDataSize = 10240;
constexpr int kKeywordModelPersistentTfLiteTensorQuantizationData = 64;
constexpr int kKeywordModelOpRuntimeDataSize = 148;

constexpr int kTestConvModelArenaSize = 12 * 1024;
uint8_t test_conv_tensor_arena[kTestConvModelArenaSize];

constexpr int kTestConvModelTensorCount = 15;
constexpr int kTestConvModelNodeAndRegistrationCount = 7;

// NOTE: These values are measured on x86-64:
// TODO(b/158651472): Consider auditing these values on non-64 bit systems.
#ifdef TF_LITE_STATIC_MEMORY
constexpr int kTestConvModelTotalSize = 9792;
constexpr int kTestConvModelTailSize = 2048;
constexpr int kTestConvModelPersistentTfLiteTensorDataSize = 128;
constexpr int kTestConvModelPersistentBufferDataSize = 680;
#else
constexpr int kTestConvModelTotalSize = 10112;
constexpr int kTestConvModelTailSize = 2368;
constexpr int kTestConvModelPersistentTfLiteTensorDataSize = 224;
constexpr int kTestConvModelPersistentBufferDataSize = 680;
#endif
constexpr int kTestConvModelHeadSize = 7744;
constexpr int kTestConvModelOpRuntimeDataSize = 136;
constexpr int kTestConvModelPersistentTfLiteTensorQuantizationData = 0;

struct ModelAllocationThresholds {
  size_t tensor_count = 0;
  size_t node_and_registration_count = 0;
  size_t total_alloc_size = 0;
  size_t head_alloc_size = 0;
  size_t tail_alloc_size = 0;
  size_t tensor_variable_buffer_data_size = 0;
  size_t persistent_tflite_tensor_data_size = 0;
  size_t persistent_tflite_tensor_quantization_data_size = 0;
  size_t op_runtime_data_size = 0;
  size_t persistent_buffer_data = 0;
};

void EnsureAllocatedSizeThreshold(const char* allocation_type, size_t actual,
                                  size_t expected) {
  // TODO(b/158651472): Better auditing of non-64 bit systems:
  if (kIs64BitSystem) {
    // 64-bit systems should check floor and ceiling to catch memory savings:
    TF_LITE_MICRO_EXPECT_NEAR(actual, expected,
                              expected * kAllocationThreshold);
    if (actual != expected) {
      TF_LITE_REPORT_ERROR(tflite::GetMicroErrorReporter(),
                           "%s threshold failed: %d != %d", allocation_type,
                           actual, expected);
    }
  } else {
    // Non-64 bit systems should just expect allocation does not exceed the
    // ceiling:
    TF_LITE_MICRO_EXPECT_LE(actual, expected + expected * kAllocationThreshold);
  }
}

void ValidateModelAllocationThresholds(
    const tflite::RecordingMicroAllocator& allocator,
    const ModelAllocationThresholds& thresholds) {
  allocator.PrintAllocations();

  EnsureAllocatedSizeThreshold(
      "Total", allocator.GetSimpleMemoryAllocator()->GetUsedBytes(),
      thresholds.total_alloc_size);
  EnsureAllocatedSizeThreshold(
      "Head", allocator.GetSimpleMemoryAllocator()->GetHeadUsedBytes(),
      thresholds.head_alloc_size);
  EnsureAllocatedSizeThreshold(
      "Tail", allocator.GetSimpleMemoryAllocator()->GetTailUsedBytes(),
      thresholds.tail_alloc_size);
  EnsureAllocatedSizeThreshold(
      "TfLiteEvalTensor",
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kTfLiteEvalTensorData)
          .used_bytes,
      sizeof(TfLiteEvalTensor) * thresholds.tensor_count);
  EnsureAllocatedSizeThreshold(
      "VariableBufferData",
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kTfLiteTensorVariableBufferData)
          .used_bytes,
      thresholds.tensor_variable_buffer_data_size);
  EnsureAllocatedSizeThreshold(
      "PersistentTfLiteTensor",
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kPersistentTfLiteTensorData)
          .used_bytes,
      thresholds.persistent_tflite_tensor_data_size);
  EnsureAllocatedSizeThreshold(
      "PersistentTfliteTensorQuantizationData",
      allocator
          .GetRecordedAllocation(tflite::RecordedAllocationType::
                                     kPersistentTfLiteTensorQuantizationData)
          .used_bytes,
      thresholds.persistent_tflite_tensor_quantization_data_size);
  EnsureAllocatedSizeThreshold(
      "PersistentBufferData",
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kPersistentBufferData)
          .used_bytes,
      thresholds.persistent_buffer_data);
  EnsureAllocatedSizeThreshold(
      "NodeAndRegistration",
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kNodeAndRegistrationArray)
          .used_bytes,
      sizeof(tflite::NodeAndRegistration) *
          thresholds.node_and_registration_count);

  // Ensure tail allocation recording is not missing any large chunks:
  size_t tail_est_length = sizeof(TfLiteEvalTensor) * thresholds.tensor_count +
                           thresholds.tensor_variable_buffer_data_size +
                           sizeof(tflite::NodeAndRegistration) *
                               thresholds.node_and_registration_count +
                           thresholds.op_runtime_data_size;
  TF_LITE_MICRO_EXPECT_LE(thresholds.tail_alloc_size - tail_est_length,
                          kAllocationTailMiscCeiling);
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestKeywordModelMemoryThreshold) {
  tflite::AllOpsResolver all_ops_resolver;
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(g_keyword_scrambled_model_data), all_ops_resolver,
      keyword_model_tensor_arena, kKeywordModelTensorArenaSize,
      tflite::GetMicroErrorReporter());

  interpreter.AllocateTensors();

  ModelAllocationThresholds thresholds;
  thresholds.tensor_count = kKeywordModelTensorCount;
  thresholds.node_and_registration_count =
      kKeywordModelNodeAndRegistrationCount;
  thresholds.total_alloc_size = kKeywordModelTotalSize;
  thresholds.head_alloc_size = kKeywordModelHeadSize;
  thresholds.tail_alloc_size = kKeywordModelTailSize;
  thresholds.tensor_variable_buffer_data_size =
      kKeywordModelTfLiteTensorVariableBufferDataSize;
  thresholds.op_runtime_data_size = kKeywordModelOpRuntimeDataSize;
  thresholds.persistent_buffer_data = kKeywordModelPersistentBufferDataSize;
  thresholds.persistent_tflite_tensor_data_size =
      kKeywordModelPersistentTfLiteTensorDataSize;
  thresholds.persistent_tflite_tensor_quantization_data_size =
      kKeywordModelPersistentTfLiteTensorQuantizationData;

  ValidateModelAllocationThresholds(interpreter.GetMicroAllocator(),
                                    thresholds);
}

TF_LITE_MICRO_TEST(TestConvModelMemoryThreshold) {
  tflite::AllOpsResolver all_ops_resolver;
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(kTestConvModelData), all_ops_resolver,
      test_conv_tensor_arena, kTestConvModelArenaSize,
      tflite::GetMicroErrorReporter());

  interpreter.AllocateTensors();

  ModelAllocationThresholds thresholds;
  thresholds.tensor_count = kTestConvModelTensorCount;
  thresholds.node_and_registration_count =
      kTestConvModelNodeAndRegistrationCount;
  thresholds.total_alloc_size = kTestConvModelTotalSize;
  thresholds.head_alloc_size = kTestConvModelHeadSize;
  thresholds.tail_alloc_size = kTestConvModelTailSize;
  thresholds.op_runtime_data_size = kTestConvModelOpRuntimeDataSize;
  thresholds.persistent_buffer_data = kTestConvModelPersistentBufferDataSize;
  thresholds.persistent_tflite_tensor_data_size =
      kTestConvModelPersistentTfLiteTensorDataSize;
  thresholds.persistent_tflite_tensor_quantization_data_size =
      kTestConvModelPersistentTfLiteTensorQuantizationData;

  ValidateModelAllocationThresholds(interpreter.GetMicroAllocator(),
                                    thresholds);
}

TF_LITE_MICRO_TESTS_END
