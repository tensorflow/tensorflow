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

#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestAllocateTensors) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];

  TfLiteContext context;
  tflite::MicroAllocator allocator =
          tflite::MicroAllocator(&context, tflite::GetLargerMockModel(), arena, arena_size, micro_test::reporter);

  allocator.AllocateTensors(); // Allocates 0, 1 and 3

  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[1].data.raw);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, context.tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, context.tensors[3].data.raw);

  // Should allocate memory for tensor #2, which is the output of op0
  allocator.AllocateForOperator(0);

  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[1].data.raw);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, context.tensors[3].data.raw);

    allocator.DeallocateAfterOperator(0);

  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[1].data.raw);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, context.tensors[3].data.raw);

  allocator.AllocateForOperator(1);
  // Should allocate memory for tensor #3, which is the output of op1
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[1].data.raw);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[3].data.raw);
}


TF_LITE_MICRO_TEST(TestInitialiseTensor) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];

  TfLiteContext context;
  tflite::MicroAllocator allocator =
          tflite::MicroAllocator(&context, tflite::GetMockModel(), arena, arena_size, micro_test::reporter);
  allocator.AllocateTensors();

  TfLiteTensor tensor0 = context.tensors[0];
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, tensor0.type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor0.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor0.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, tensor0.bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor0.data.i32);

  TfLiteTensor tensor1 = context.tensors[1];
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, tensor1.type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor1.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor1.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor1.bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor1.data.uint8);

  TfLiteTensor tensor2 = context.tensors[2];
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, tensor2.type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor2.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor2.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, tensor2.bytes);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, tensor2.data.i32); // not allocated until AllocateForOperator(0)
}

TF_LITE_MICRO_TESTS_END
