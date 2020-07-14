/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_interpreter.h"

#include <cstdint>

#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_optional_debug_tools.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {

class MockProfiler : public tflite::Profiler {
 public:
  MockProfiler() : event_starts_(0), event_ends_(0) {}
  ~MockProfiler() override = default;

  // AddEvent is unused for Tf Micro.
  void AddEvent(const char* tag, EventType event_type, uint64_t start,
                uint64_t end, int64_t event_metadata1,
                int64_t event_metadata2) override{};

  // BeginEvent followed by code followed by EndEvent will profile the code
  // enclosed. Multiple concurrent events are unsupported, so the return value
  // is always 0. Event_metadata1 and event_metadata2 are unused. The tag
  // pointer must be valid until EndEvent is called.
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override {
    event_starts_++;
    return 0;
  }

  // Event_handle is ignored since TF Micro does not support concurrent events.
  void EndEvent(uint32_t event_handle) override { event_ends_++; }

  int event_starts() { return event_starts_; }
  int event_ends() { return event_ends_; }

 private:
  int event_starts_;
  int event_ends_;
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInterpreter) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);

  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();

  constexpr size_t allocator_buffer_size = 1000;
  uint8_t allocator_buffer[allocator_buffer_size];

  // Create a new scope so that we can test the destructor.
  {
    tflite::MicroInterpreter interpreter(model, op_resolver, allocator_buffer,
                                         allocator_buffer_size,
                                         micro_test::reporter);
    TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
    TF_LITE_MICRO_EXPECT_LE(interpreter.arena_used_bytes(), 928 + 100);
    TF_LITE_MICRO_EXPECT_EQ(1, interpreter.inputs_size());
    TF_LITE_MICRO_EXPECT_EQ(2, interpreter.outputs_size());

    TfLiteTensor* input = interpreter.input(0);
    TF_LITE_MICRO_EXPECT_NE(nullptr, input);
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, input->type);
    TF_LITE_MICRO_EXPECT_EQ(1, input->dims->size);
    TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
    TF_LITE_MICRO_EXPECT_EQ(4, input->bytes);
    TF_LITE_MICRO_EXPECT_NE(nullptr, input->data.i32);
    input->data.i32[0] = 21;

    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());

    TfLiteTensor* output = interpreter.output(0);
    TF_LITE_MICRO_EXPECT_NE(nullptr, output);
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, output->type);
    TF_LITE_MICRO_EXPECT_EQ(1, output->dims->size);
    TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
    TF_LITE_MICRO_EXPECT_EQ(4, output->bytes);
    TF_LITE_MICRO_EXPECT_NE(nullptr, output->data.i32);
    TF_LITE_MICRO_EXPECT_EQ(42, output->data.i32[0]);

    output = interpreter.output(1);
    TF_LITE_MICRO_EXPECT_NE(nullptr, output);
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, output->type);
    TF_LITE_MICRO_EXPECT_EQ(1, output->dims->size);
    TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
    TF_LITE_MICRO_EXPECT_EQ(4, output->bytes);
    TF_LITE_MICRO_EXPECT_NE(nullptr, output->data.i32);
    TF_LITE_MICRO_EXPECT_EQ(42, output->data.i32[0]);

    // Just to make sure that this method works.
    tflite::PrintInterpreterState(&interpreter);
  }

  TF_LITE_MICRO_EXPECT_EQ(tflite::testing::MockCustom::freed_, true);
}

TF_LITE_MICRO_TEST(TestKernelMemoryPlanning) {
  const tflite::Model* model = tflite::testing::GetSimpleStatefulModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);

  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();

  constexpr size_t allocator_buffer_size = 1024;
  uint8_t allocator_buffer[allocator_buffer_size];
  tflite::MicroInterpreter interpreter(model, op_resolver, allocator_buffer,
                                       allocator_buffer_size,
                                       micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.inputs_size());
  TF_LITE_MICRO_EXPECT_EQ(2, interpreter.outputs_size());

  TfLiteTensor* input = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(3, input->dims->data[0]);
  input->data.uint8[0] = 2;
  input->data.uint8[1] = 3;
  input->data.uint8[2] = 1;

  uint8_t expected_median = 2;

  {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());
    TfLiteTensor* median = interpreter.output(0);
    TF_LITE_MICRO_EXPECT_EQ(expected_median, median->data.uint8[0]);
    TfLiteTensor* invoke_count = interpreter.output(1);
    TF_LITE_MICRO_EXPECT_EQ(1, invoke_count->data.i32[0]);
  }

  {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());
    TfLiteTensor* median = interpreter.output(0);
    TF_LITE_MICRO_EXPECT_EQ(expected_median, median->data.uint8[0]);
    TfLiteTensor* invoke_count = interpreter.output(1);
    TF_LITE_MICRO_EXPECT_EQ(2, invoke_count->data.i32[0]);
  }
}

TF_LITE_MICRO_TEST(TestVariableTensorReset) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);

  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();

  constexpr size_t allocator_buffer_size =
      2096 /* optimal arena size at the time of writting. */ +
      16 /* alignment */ + 100 /* some headroom */;
  uint8_t allocator_buffer[allocator_buffer_size];
  tflite::MicroInterpreter interpreter(model, op_resolver, allocator_buffer,
                                       allocator_buffer_size,
                                       micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_LE(interpreter.arena_used_bytes(), 2096 + 100);
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.inputs_size());
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.outputs_size());

  // Assign hard-code values:
  for (size_t i = 0; i < interpreter.tensors_size(); ++i) {
    TfLiteTensor* cur_tensor = interpreter.tensor(i);
    int buffer_length = tflite::ElementCount(*cur_tensor->dims);
    // Assign all buffers to non-zero values. Variable tensors will be assigned
    // 2 here and will be verified that they have been reset after the API call.
    int buffer_value = cur_tensor->is_variable ? 2 : 1;
    switch (cur_tensor->type) {
      case kTfLiteInt32: {
        int32_t* buffer = tflite::GetTensorData<int32_t>(cur_tensor);
        for (int j = 0; j < buffer_length; ++j) {
          buffer[j] = static_cast<int32_t>(buffer_value);
        }
        break;
      }
      case kTfLiteUInt8: {
        uint8_t* buffer = tflite::GetTensorData<uint8_t>(cur_tensor);
        for (int j = 0; j < buffer_length; ++j) {
          buffer[j] = static_cast<uint8_t>(buffer_value);
        }
        break;
      }
      default:
        TF_LITE_MICRO_FAIL("Unsupported dtype");
    }
  }

  interpreter.ResetVariableTensors();

  // Ensure only variable tensors have been reset to zero:
  for (size_t i = 0; i < interpreter.tensors_size(); ++i) {
    TfLiteTensor* cur_tensor = interpreter.tensor(i);
    int buffer_length = tflite::ElementCount(*cur_tensor->dims);
    // Variable tensors should be zero (not the value assigned in the for loop
    // above).
    int buffer_value = cur_tensor->is_variable ? 0 : 1;
    switch (cur_tensor->type) {
      case kTfLiteInt32: {
        int32_t* buffer = tflite::GetTensorData<int32_t>(cur_tensor);
        for (int j = 0; j < buffer_length; ++j) {
          TF_LITE_MICRO_EXPECT_EQ(buffer_value, buffer[j]);
        }
        break;
      }
      case kTfLiteUInt8: {
        uint8_t* buffer = tflite::GetTensorData<uint8_t>(cur_tensor);
        for (int j = 0; j < buffer_length; ++j) {
          TF_LITE_MICRO_EXPECT_EQ(buffer_value, buffer[j]);
        }
        break;
      }
      default:
        TF_LITE_MICRO_FAIL("Unsupported dtype");
    }
  }
}

// The interpreter initialization requires multiple steps and this test case
// ensures that simply creating and destructing an interpreter object is ok.
// b/147830765 has one example of a change that caused trouble for this simple
// case.
TF_LITE_MICRO_TEST(TestIncompleteInitialization) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);

  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();

  constexpr size_t allocator_buffer_size = 2048;
  uint8_t allocator_buffer[allocator_buffer_size];

  tflite::MicroInterpreter interpreter(model, op_resolver, allocator_buffer,
                                       allocator_buffer_size,
                                       micro_test::reporter);
}

// Test that an interpreter with a supplied profiler correctly calls the
// profiler each time an operator is invoked.
TF_LITE_MICRO_TEST(InterpreterWithProfilerShouldProfileOps) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);

  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();

  constexpr size_t allocator_buffer_size = 2048;
  uint8_t allocator_buffer[allocator_buffer_size];
  tflite::MockProfiler profiler;
  tflite::MicroInterpreter interpreter(model, op_resolver, allocator_buffer,
                                       allocator_buffer_size,
                                       micro_test::reporter, &profiler);

  TF_LITE_MICRO_EXPECT_EQ(profiler.event_starts(), 0);
  TF_LITE_MICRO_EXPECT_EQ(profiler.event_ends(), 0);
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(interpreter.Invoke(), kTfLiteOk);
#ifndef NDEBUG
  TF_LITE_MICRO_EXPECT_EQ(profiler.event_starts(), 3);
  TF_LITE_MICRO_EXPECT_EQ(profiler.event_ends(), 3);
#else  // Profile events will not occur on release builds.
  TF_LITE_MICRO_EXPECT_EQ(profiler.event_starts(), 0);
  TF_LITE_MICRO_EXPECT_EQ(profiler.event_ends(), 0);
#endif
}

TF_LITE_MICRO_TEST(TestIncompleteInitializationAllocationsWithSmallArena) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);

  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();

  constexpr size_t allocator_buffer_size = 500;
  uint8_t allocator_buffer[allocator_buffer_size];

  tflite::RecordingMicroAllocator* allocator =
      tflite::RecordingMicroAllocator::Create(
          allocator_buffer, allocator_buffer_size, micro_test::reporter);
  TF_LITE_MICRO_EXPECT_NE(nullptr, allocator);

  tflite::MicroInterpreter interpreter(model, op_resolver, allocator,
                                       micro_test::reporter);

  // Interpreter fails because arena is too small:
  TF_LITE_MICRO_EXPECT_EQ(interpreter.Invoke(), kTfLiteError);

  // Ensure allocations are zero (ignore tail since some internal structs are
  // initialized with this space):
  TF_LITE_MICRO_EXPECT_EQ(
      0, allocator->GetSimpleMemoryAllocator()->GetHeadUsedBytes());
  TF_LITE_MICRO_EXPECT_EQ(
      0, allocator
             ->GetRecordedAllocation(
                 tflite::RecordedAllocationType::kTfLiteTensorArray)
             .used_bytes);
  TF_LITE_MICRO_EXPECT_EQ(
      0, allocator
             ->GetRecordedAllocation(tflite::RecordedAllocationType::
                                         kTfLiteTensorArrayQuantizationData)
             .used_bytes);
  TF_LITE_MICRO_EXPECT_EQ(
      0,
      allocator
          ->GetRecordedAllocation(
              tflite::RecordedAllocationType::kTfLiteTensorVariableBufferData)
          .used_bytes);
  TF_LITE_MICRO_EXPECT_EQ(
      0,
      allocator->GetRecordedAllocation(tflite::RecordedAllocationType::kOpData)
          .used_bytes);
}

TF_LITE_MICRO_TEST(TestInterpreterDoesNotAllocateUntilInvoke) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);

  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();

  constexpr size_t allocator_buffer_size = 1024 * 10;
  uint8_t allocator_buffer[allocator_buffer_size];

  tflite::RecordingMicroAllocator* allocator =
      tflite::RecordingMicroAllocator::Create(
          allocator_buffer, allocator_buffer_size, micro_test::reporter);
  TF_LITE_MICRO_EXPECT_NE(nullptr, allocator);

  tflite::MicroInterpreter interpreter(model, op_resolver, allocator,
                                       micro_test::reporter);

  // Ensure allocations are zero (ignore tail since some internal structs are
  // initialized with this space):
  TF_LITE_MICRO_EXPECT_EQ(
      0, allocator->GetSimpleMemoryAllocator()->GetHeadUsedBytes());
  TF_LITE_MICRO_EXPECT_EQ(
      0, allocator
             ->GetRecordedAllocation(
                 tflite::RecordedAllocationType::kTfLiteTensorArray)
             .used_bytes);
  TF_LITE_MICRO_EXPECT_EQ(
      0,
      allocator
          ->GetRecordedAllocation(
              tflite::RecordedAllocationType::kTfLiteTensorVariableBufferData)
          .used_bytes);
  TF_LITE_MICRO_EXPECT_EQ(
      0,
      allocator->GetRecordedAllocation(tflite::RecordedAllocationType::kOpData)
          .used_bytes);

  TF_LITE_MICRO_EXPECT_EQ(interpreter.Invoke(), kTfLiteOk);
  allocator->PrintAllocations();

  // Allocation sizes vary based on platform - check that allocations are now
  // non-zero:
  TF_LITE_MICRO_EXPECT_GT(
      allocator->GetSimpleMemoryAllocator()->GetHeadUsedBytes(), 0);
  TF_LITE_MICRO_EXPECT_GT(
      allocator
          ->GetRecordedAllocation(
              tflite::RecordedAllocationType::kTfLiteTensorArray)
          .used_bytes,
      0);

  TF_LITE_MICRO_EXPECT_GT(
      allocator
          ->GetRecordedAllocation(
              tflite::RecordedAllocationType::kTfLiteTensorVariableBufferData)
          .used_bytes,
      0);

  // TODO(b/160160549): This check is mostly meaningless right now because the
  // operator creation in our mock models is inconsistent. Revisit what this
  // check should be once the mock models are properly created.
  TF_LITE_MICRO_EXPECT_EQ(
      allocator->GetRecordedAllocation(tflite::RecordedAllocationType::kOpData)
          .used_bytes,
      0);
}

TF_LITE_MICRO_TESTS_END
