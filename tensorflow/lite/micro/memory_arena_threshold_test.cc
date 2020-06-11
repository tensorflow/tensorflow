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
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

/**
 * Tests to ensure arena memory allocation does not regress by more than 3%.
 */

namespace {

// Ensure memory doesn't expand more that 3%:
constexpr float kAllocationThreshold = 0.03;
const bool kIs64BitSystem = sizeof(void*) == 8;

constexpr int kKeywordTensorArenaSize = 22 * 1024;
uint8_t tensor_arena[kKeywordTensorArenaSize];

constexpr int kKeywordModelTensorCount = 54;
constexpr int kKeywordModelNodeAndRegistrationCount = 15;

// NOTE: These values are measured on x86-64:
// TODO(b/158651472): Consider auditing these values on non-64 bit systems.
constexpr int kKeywordModelTotalSize = 21440;
constexpr int kKeywordModelHeadSize = 672;
constexpr int kKeywordModelTailSize = 20768;
constexpr int kKeywordModelTfLiteTensorQuantizationDataSize = 2160;
constexpr int kKeywordModelOpRuntimeDataSize = 148;

void EnsureAllocatedSizeThreshold(size_t actual, size_t expected) {
  // TODO(b/158651472): Better auditing of non-64 bit systems:
  if (kIs64BitSystem) {
    // 64-bit systems should check floor and ceiling to catch memory savings:
    TF_LITE_MICRO_EXPECT_NEAR(actual, expected, kAllocationThreshold);
  } else {
    // Non-64 bit systems should just expect allocation does not exceed the
    // ceiling:
    TF_LITE_MICRO_EXPECT_LE(actual, expected + expected * kAllocationThreshold);
  }
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestKeywordModelMemoryThreshold) {
  tflite::AllOpsResolver all_ops_resolver;
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(g_keyword_scrambled_model_data), &all_ops_resolver,
      tensor_arena, kKeywordTensorArenaSize, micro_test::reporter);

  interpreter.AllocateTensors();

  const tflite::RecordingMicroAllocator& allocator =
      interpreter.GetMicroAllocator();
  allocator.PrintAllocations();

  EnsureAllocatedSizeThreshold(
      allocator.GetSimpleMemoryAllocator()->GetUsedBytes(),
      kKeywordModelTotalSize);
  EnsureAllocatedSizeThreshold(
      allocator.GetSimpleMemoryAllocator()->GetHeadUsedBytes(),
      kKeywordModelHeadSize);
  EnsureAllocatedSizeThreshold(
      allocator.GetSimpleMemoryAllocator()->GetTailUsedBytes(),
      kKeywordModelTailSize);
  EnsureAllocatedSizeThreshold(
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kTfLiteTensorArray)
          .used_bytes,
      sizeof(TfLiteTensor) * kKeywordModelTensorCount);
  EnsureAllocatedSizeThreshold(
      allocator
          .GetRecordedAllocation(tflite::RecordedAllocationType::
                                     kTfLiteTensorArrayQuantizationData)
          .used_bytes,
      kKeywordModelTfLiteTensorQuantizationDataSize);
  EnsureAllocatedSizeThreshold(
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kNodeAndRegistrationArray)
          .used_bytes,
      sizeof(tflite::NodeAndRegistration) *
          kKeywordModelNodeAndRegistrationCount);
  EnsureAllocatedSizeThreshold(
      allocator.GetRecordedAllocation(tflite::RecordedAllocationType::kOpData)
          .used_bytes,
      kKeywordModelOpRuntimeDataSize);
}

TF_LITE_MICRO_TESTS_END
