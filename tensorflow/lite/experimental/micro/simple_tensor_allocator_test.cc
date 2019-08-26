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

#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

namespace tflite {
namespace {

// Not currently used, but kept for future use
flatbuffers::FlatBufferBuilder* BuilderInstance() {
  static char inst_memory[sizeof(flatbuffers::FlatBufferBuilder)];
  static StackAllocator stack_allocator;
  static flatbuffers::FlatBufferBuilder* inst =
      new (inst_memory) flatbuffers::FlatBufferBuilder(
          StackAllocator::kStackAllocatorSize, &stack_allocator);
  return inst;
}

const Tensor* Create1dTensor(int size) {
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

const Tensor* CreateMissingQuantizationTensor(int size) {
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

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestSmallStatic) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(nullptr, arena, arena_size);

  uint8_t *result;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          allocator.AllocateStaticMemory(512, 1, micro_test::reporter, &result));
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);
}

TF_LITE_MICRO_TEST(TestJustFitsStatic) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(nullptr, arena, arena_size);

  uint8_t *result;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          allocator.AllocateStaticMemory(arena_size, 1, micro_test::reporter, &result));
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);
}

TF_LITE_MICRO_TEST(TestTooLargeStatic) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(nullptr, arena, arena_size);

  uint8_t *result;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          allocator.AllocateStaticMemory(arena_size + 1, 1, micro_test::reporter, &result));
  TF_LITE_MICRO_EXPECT_EQ(nullptr, result);
}

TF_LITE_MICRO_TEST(TestAlignedStatic) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(nullptr, arena, arena_size);

  uint8_t *result;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          allocator.AllocateStaticMemory(1, 1, micro_test::reporter, &result));
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          allocator.AllocateStaticMemory(16, 4, micro_test::reporter, &result));
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);
  TF_LITE_MICRO_EXPECT_EQ(0, reinterpret_cast<size_t>(result) & 3);
}

TF_LITE_MICRO_TEST(TestMultipleTooLargeStatic) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(nullptr, arena, arena_size);

  uint8_t *result;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          allocator.AllocateStaticMemory(768, 1, micro_test::reporter, &result));
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          allocator.AllocateStaticMemory(768, 1, micro_test::reporter, &result));
  TF_LITE_MICRO_EXPECT_EQ(nullptr, result);
}

TF_LITE_MICRO_TEST(TestAllocateTensor) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(nullptr, arena, arena_size);

  TfLiteTensor tensor;
  int dims[] = {1, 100};
  tensor.data.uint8 = nullptr;
  tensor.type = kTfLiteInt32;
  tensor.bytes = 400;
  tensor.dims = reinterpret_cast<TfLiteIntArray*>(dims);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.AllocateBuffer(&tensor, micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(arena, tensor.data.uint8);
}

TF_LITE_MICRO_TESTS_END
