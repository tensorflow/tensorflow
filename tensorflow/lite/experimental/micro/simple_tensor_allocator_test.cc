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

namespace tflite {
namespace {
class StackAllocator : public flatbuffers::Allocator {
 public:
  StackAllocator() : data_(data_backing_), data_size_(0) {}

  uint8_t* allocate(size_t size) override {
    if ((data_size_ + size) > kStackAllocatorSize) {
      // TODO(petewarden): Add error reporting beyond returning null!
      return nullptr;
    }
    uint8_t* result = data_;
    data_ += size;
    data_size_ += size;
    return result;
  }

  void deallocate(uint8_t* p, size_t) override {}

  static StackAllocator& instance() {
    // Avoid using true dynamic memory allocation to be portable to bare metal.
    static char inst_memory[sizeof(StackAllocator)];
    static StackAllocator* inst = new (inst_memory) StackAllocator;
    return *inst;
  }

  static constexpr int kStackAllocatorSize = 4096;

 private:
  uint8_t data_backing_[kStackAllocatorSize];
  uint8_t* data_;
  int data_size_;
};

flatbuffers::FlatBufferBuilder* BuilderInstance() {
  static char inst_memory[sizeof(flatbuffers::FlatBufferBuilder)];
  static flatbuffers::FlatBufferBuilder* inst =
      new (inst_memory) flatbuffers::FlatBufferBuilder(
          StackAllocator::kStackAllocatorSize, &StackAllocator::instance());
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

const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* CreateBuffers() {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();
  constexpr size_t buffers_size = 1;
  const Offset<Buffer> buffers[buffers_size] = {
      CreateBuffer(*builder),
  };
  const flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
      buffers_offset = builder->CreateVector(buffers, buffers_size);
  builder->Finish(buffers_offset);
  void* buffers_pointer = builder->GetBufferPointer();
  const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* result =
      flatbuffers::GetRoot<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>(
          buffers_pointer);
  return result;
}

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestAllocateTensor) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(arena, arena_size);

  const tflite::Tensor* tensor = tflite::Create1dTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::CreateBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.AllocateTensor(*tensor, 0, 1, buffers, micro_test::reporter,
                               &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(400, allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, allocated_tensor.data.i32);
}

TF_LITE_MICRO_TEST(TestTooLarge) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(arena, arena_size);

  const tflite::Tensor* tensor = tflite::Create1dTensor(2000);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::CreateBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_NE(
      kTfLiteOk,
      allocator.AllocateTensor(*tensor, 0, 1, buffers, micro_test::reporter,
                               &allocated_tensor));
}

TF_LITE_MICRO_TEST(TestJustFits) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(arena, arena_size);

  uint8_t* result = allocator.AllocateMemory(arena_size, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);
}

TF_LITE_MICRO_TEST(TestAligned) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(arena, arena_size);

  uint8_t* result = allocator.AllocateMemory(1, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);

  result = allocator.AllocateMemory(16, 4);
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);
  TF_LITE_MICRO_EXPECT_EQ(0, reinterpret_cast<size_t>(result) & 3);
}

TF_LITE_MICRO_TEST(TestMultipleTooLarge) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleTensorAllocator allocator(arena, arena_size);

  uint8_t* result = allocator.AllocateMemory(768, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);

  result = allocator.AllocateMemory(768, 1);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, result);
}

TF_LITE_MICRO_TESTS_END
