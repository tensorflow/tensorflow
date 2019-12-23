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

#include "tensorflow/lite/micro/micro_allocator.h"

#include <cstdint>

#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInitializeRuntimeTensor) {
  const tflite::Model* model = tflite::testing::GetMockModel();
  TfLiteContext context;
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator allocator(&context, model, arena, arena_size,
                                   micro_test::reporter);

  const tflite::Tensor* tensor = tflite::testing::Create1dFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.InitializeRuntimeTensor(
                                         *tensor, buffers, micro_test::reporter,
                                         &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(400, allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, allocated_tensor.data.i32);
}

TF_LITE_MICRO_TEST(TestMissingQuantization) {
  const tflite::Model* model = tflite::testing::GetMockModel();
  TfLiteContext context;
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator allocator(&context, model, arena, arena_size,
                                   micro_test::reporter);

  const tflite::Tensor* tensor =
      tflite::testing::CreateMissingQuantizationFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.InitializeRuntimeTensor(
                                         *tensor, buffers, micro_test::reporter,
                                         &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(400, allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, allocated_tensor.data.i32);
}

TF_LITE_MICRO_TEST(TestFinishTensorAllocation) {
  const tflite::Model* model = tflite::testing::GetMockModel();
  TfLiteContext context;
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator allocator(&context, model, arena, arena_size,
                                   micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(3, context.tensors_size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.FinishTensorAllocation());
  // No allocation to be done afterwards.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError, allocator.FinishTensorAllocation());

  constexpr int kExpectedAlignment = 4;

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, context.tensors[0].type);
  TF_LITE_MICRO_EXPECT_EQ(1, context.tensors[0].dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, context.tensors[0].dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, context.tensors[0].bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_EQ(
      0, (reinterpret_cast<std::uintptr_t>(context.tensors[0].data.raw) %
          kExpectedAlignment));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, context.tensors[1].type);
  TF_LITE_MICRO_EXPECT_EQ(1, context.tensors[1].dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, context.tensors[1].dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, context.tensors[1].bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[1].data.raw);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, context.tensors[2].type);
  TF_LITE_MICRO_EXPECT_EQ(1, context.tensors[2].dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, context.tensors[2].dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, context.tensors[2].bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, context.tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_EQ(
      0, (reinterpret_cast<std::uintptr_t>(context.tensors[2].data.raw) %
          kExpectedAlignment));

  TF_LITE_MICRO_EXPECT_NE(context.tensors[1].data.raw,
                          context.tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(context.tensors[2].data.raw,
                          context.tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(context.tensors[1].data.raw,
                          context.tensors[2].data.raw);
}

TF_LITE_MICRO_TESTS_END
