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

#include "tensorflow/lite/micro/micro_allocator.h"

#include <cstdint>

#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_conv_model.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kExpectedAlignment = 4;
constexpr int t0 = 0;
constexpr int t1 = 1;
constexpr int t2 = 2;
constexpr int t3 = 3;
constexpr int t4 = 4;
constexpr int t5 = 5;

void VerifyMockTfLiteTensor(TfLiteTensor* tensor, bool is_variable = false) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(is_variable, tensor->is_variable);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), tensor->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor->data.raw);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          (reinterpret_cast<std::uintptr_t>(tensor->data.raw) %
                           kExpectedAlignment));
}

void VerifyMockWeightTfLiteTensor(TfLiteTensor* tensor) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(1), tensor->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor->data.raw);
}

void VerifyMockTfLiteEvalTensor(TfLiteEvalTensor* tensor) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  size_t buffer_size;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::TfLiteEvalTensorByteLength(tensor, &buffer_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), buffer_size);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor->data.raw);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          (reinterpret_cast<std::uintptr_t>(tensor->data.raw) %
                           kExpectedAlignment));
}

void VerifyMockWeightTfLiteEvalTensor(TfLiteEvalTensor* tensor) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  size_t buffer_size;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::TfLiteEvalTensorByteLength(tensor, &buffer_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(1), buffer_size);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor->data.raw);
}

void VerifyMockTensor(const Model* model, MicroAllocator* allocator,
                      TfLiteEvalTensor* eval_tensors, int tensor_idx,
                      bool is_variable = false) {
  VerifyMockTfLiteTensor(allocator->AllocatePersistentTfLiteTensor(
                             model, eval_tensors, tensor_idx),
                         is_variable);
  VerifyMockTfLiteEvalTensor(&eval_tensors[tensor_idx]);
}

void VerifyMockWeightTensor(const Model* model, MicroAllocator* allocator,
                            TfLiteEvalTensor* eval_tensors, int tensor_idx) {
  VerifyMockWeightTfLiteTensor(allocator->AllocatePersistentTfLiteTensor(
      model, eval_tensors, tensor_idx));
  VerifyMockWeightTfLiteEvalTensor(&eval_tensors[tensor_idx]);
}

void EnsureUniqueVariableTensorBuffer(const Model* model,
                                      TfLiteEvalTensor* eval_tensors,
                                      const int variable_tensor_idx) {
  for (size_t i = 0; i < GetModelTensorCount(model); i++) {
    if (i != static_cast<size_t>(variable_tensor_idx)) {
      TF_LITE_MICRO_EXPECT_NE(eval_tensors[variable_tensor_idx].data.raw,
                              eval_tensors[i].data.raw);
    }
  }
}

void VerifyRegistrationAndNodeAllocation(
    NodeAndRegistration* node_and_registration, size_t count) {
  for (size_t i = 0; i < count; i++) {
    TF_LITE_MICRO_EXPECT_NE(nullptr, node_and_registration[i].registration);
    TF_LITE_MICRO_EXPECT_NE(nullptr, node_and_registration[i].node.inputs);
    TF_LITE_MICRO_EXPECT_NE(nullptr, node_and_registration[i].node.outputs);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInitializeRuntimeTensor) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator* simple_allocator =
      tflite::SimpleMemoryAllocator::Create(tflite::GetMicroErrorReporter(),
                                            arena, arena_size);

  const tflite::Tensor* tensor = tflite::testing::Create1dFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
          simple_allocator, /*allocate_temp=*/false, *tensor, buffers,
          tflite::GetMicroErrorReporter(), &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(400), allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT(nullptr == allocated_tensor.data.i32);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteArenaRw, allocated_tensor.allocation_type);

  simple_allocator->~SimpleMemoryAllocator();
}

// TODO(b/162311891): Drop this test when InitializeTfLiteTensorFromFlatbuffer()
// always allocates from temp (interpreter returns buffers from
// TfLiteEvalTensor):
TF_LITE_MICRO_TEST(TestInitializeTempRuntimeTensor) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator* simple_allocator =
      tflite::SimpleMemoryAllocator::Create(tflite::GetMicroErrorReporter(),
                                            arena, arena_size);

  const tflite::Tensor* tensor = tflite::testing::Create1dFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_temp_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
                     simple_allocator, /*allocate_temp=*/true, *tensor, buffers,
                     tflite::GetMicroErrorReporter(), &allocated_temp_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_temp_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_temp_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_temp_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(400),
                          allocated_temp_tensor.bytes);
  TF_LITE_MICRO_EXPECT(nullptr == allocated_temp_tensor.data.i32);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteArenaRw,
                          allocated_temp_tensor.allocation_type);

  simple_allocator->~SimpleMemoryAllocator();
}

TF_LITE_MICRO_TEST(TestInitializeQuantizedTensor) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator* simple_allocator =
      tflite::SimpleMemoryAllocator::Create(tflite::GetMicroErrorReporter(),
                                            arena, arena_size);

  const tflite::Tensor* tensor =
      tflite::testing::CreateQuantizedFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
          simple_allocator, /*allocate_temp=*/false, *tensor, buffers,
          tflite::GetMicroErrorReporter(), &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(400), allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT(nullptr == allocated_tensor.data.i32);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteArenaRw, allocated_tensor.allocation_type);

  simple_allocator->~SimpleMemoryAllocator();
}

TF_LITE_MICRO_TEST(TestMissingQuantization) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator* simple_allocator =
      tflite::SimpleMemoryAllocator::Create(tflite::GetMicroErrorReporter(),
                                            arena, arena_size);

  const tflite::Tensor* tensor =
      tflite::testing::CreateMissingQuantizationFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
          simple_allocator, /*allocate_temp=*/false, *tensor, buffers,
          tflite::GetMicroErrorReporter(), &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(400), allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT(nullptr == allocated_tensor.data.i32);
}

TF_LITE_MICRO_TEST(TestFailsWhenModelStartsTwice) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
}

TF_LITE_MICRO_TEST(TestFailsWithWrongSequence) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(nullptr, allocator);

  // We can't finish allocation before it ever got started.
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, allocator->FinishModelAllocation(model, eval_tensors,
                                                     &scratch_buffer_handles));

  // Start twice is not allowed.
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
}

TF_LITE_MICRO_TEST(TestMockModelAllocation) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles));

  size_t model_tensor_size = tflite::testing::GetModelTensorCount(model);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), model_tensor_size);

  tflite::testing::VerifyMockTensor(model, allocator, eval_tensors, 0);
  tflite::testing::VerifyMockWeightTensor(model, allocator, eval_tensors, 1);
  tflite::testing::VerifyMockTensor(model, allocator, eval_tensors, 2);
  tflite::testing::VerifyMockTensor(model, allocator, eval_tensors, 3);

  TF_LITE_MICRO_EXPECT_NE(eval_tensors[1].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[2].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[1].data.raw, eval_tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[1].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_LE(allocator->used_bytes(), 856 + 100);

  // SimpleMockModel has 2 operators:
  tflite::testing::VerifyRegistrationAndNodeAllocation(node_and_registration,
                                                       /*count=*/2);
}

TF_LITE_MICRO_TEST(TestMultiTenantAllocation) {
  // The `OpResolver` is shared among different models in this test for
  // simplicity but in practice you could have different `OpResolver`.
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();

  // Create a shared allocator.
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(nullptr, allocator);
  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;

  // Allocate for model 1. We use ComplexMockModel here to cover the code path
  // allocatig variables.
  const tflite::Model* model1 = tflite::testing::GetComplexMockModel();
  tflite::NodeAndRegistration* node_and_registration1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model1, op_resolver,
                                      &node_and_registration1, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model1, eval_tensors,
                                                  &scratch_buffer_handles));
  const size_t single_model_used_bytes = allocator->used_bytes();

  // Allocate for model 2.
  const tflite::Model* model2 = tflite::testing::GetComplexMockModel();
  tflite::NodeAndRegistration* node_and_registration2;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model2, op_resolver,
                                      &node_and_registration2, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model2, eval_tensors,
                                                  &scratch_buffer_handles));

  // Allocation for two instances of the same model takes less memory as `head`
  // of the arena is reused.
  TF_LITE_MICRO_EXPECT_LE(allocator->used_bytes(), 2 * single_model_used_bytes);
}

TF_LITE_MICRO_TEST(TestAllocationForModelsWithBranches) {
  const tflite::Model* model = tflite::testing::GetSimpleModelWithBranch();
  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(nullptr, allocator);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles));

  uint8_t* start = eval_tensors[0].data.uint8;
  // Check test_helpers.cc BuildSimpleModelWithBranch for model structure.
  // t0 is the first tensor, so place it in offset 0.
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[0].data.uint8 - start);
  // bytes = 2 * 2 * 3 * sizeof(float32) = 48, same for other tensors.
  size_t buffer_size;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, tflite::TfLiteEvalTensorByteLength(
                                         &eval_tensors[0], &buffer_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(48), buffer_size);
  // t1 can't reuse any memory, as n0 requires both t0 and t1.
  TF_LITE_MICRO_EXPECT_EQ(96, eval_tensors[1].data.uint8 - start);
  // t2 can't reuse any memory, as n1 requires both t0 and t2. Also n2 requires
  // both t1 and t2.
  TF_LITE_MICRO_EXPECT_EQ(48, eval_tensors[2].data.uint8 - start);
  // t3 reuses the same memory from t0 as t0 is not an input to any node.
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[3].data.uint8 - start);

  // SimpleModelWithBranch has 3 operators:
  tflite::testing::VerifyRegistrationAndNodeAllocation(node_and_registration,
                                                       /*count=*/3);
}

TF_LITE_MICRO_TEST(TestAllocationForComplexModelAllocation) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  constexpr size_t arena_size = 2048;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles));

  size_t model_tensor_size = tflite::testing::GetModelTensorCount(model);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(10), model_tensor_size);

  // NOTE: Tensor indexes match the values in GetComplexMockModel().
  tflite::testing::VerifyMockTensor(model, allocator, eval_tensors, 0);
  tflite::testing::VerifyMockTensor(model, allocator, eval_tensors, 1,
                                    /*is_variable=*/true);
  tflite::testing::VerifyMockWeightTensor(model, allocator, eval_tensors, 2);
  tflite::testing::VerifyMockTensor(model, allocator, eval_tensors, 3);
  tflite::testing::VerifyMockTensor(model, allocator, eval_tensors, 4,
                                    /*is_variable=*/true);
  tflite::testing::VerifyMockWeightTensor(model, allocator, eval_tensors, 5);
  tflite::testing::VerifyMockTensor(model, allocator, eval_tensors, 6);
  tflite::testing::VerifyMockTensor(model, allocator, eval_tensors, 7,
                                    /*is_variable=*/true);
  tflite::testing::VerifyMockWeightTensor(model, allocator, eval_tensors, 8);
  tflite::testing::VerifyMockTensor(model, allocator, eval_tensors, 9);

  // // Ensure that variable tensors have unique address
  tflite::testing::EnsureUniqueVariableTensorBuffer(model, eval_tensors, 1);
  tflite::testing::EnsureUniqueVariableTensorBuffer(model, eval_tensors, 4);
  tflite::testing::EnsureUniqueVariableTensorBuffer(model, eval_tensors, 7);

  // ComplexMockModel has 3 operators:
  tflite::testing::VerifyRegistrationAndNodeAllocation(node_and_registration,
                                                       /*count=*/3);
}

TF_LITE_MICRO_TEST(OfflinePlannerBranchesAllOnline) {
  int version = 1;
  int subgraph = 0;
  constexpr int number_tensors = 4;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {version, subgraph,
                                                   number_tensors,  // header
                                                   // memory offsets:
                                                   -1, -1, -1, -1};

  // The structure is identical to the one in
  // TestAllocationForModelsWithBranches
  int number_connections = 3;
  tflite::testing::NodeConnection node_list[3] = {{
                                                      {0},  // input
                                                      {1}   // output
                                                  },
                                                  {
                                                      {0},  // input
                                                      {2}   // output
                                                  },
                                                  {
                                                      {1, 2},  // input1, input2
                                                      {3}      // output
                                                  }};

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections);

  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;

  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles));

  // Since all of the tensors are online planned and the model structure is
  // identical to that in TestAllocationForModelsWithBranches,
  // the offsets be should identical to that test.
  uint8_t* start = eval_tensors[0].data.uint8;
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[0].data.uint8 - start);

  size_t buffer_size;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, tflite::TfLiteEvalTensorByteLength(
                                         &eval_tensors[0], &buffer_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(48), buffer_size);
  TF_LITE_MICRO_EXPECT_EQ(96, eval_tensors[1].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, eval_tensors[2].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[3].data.uint8 - start);
}

TF_LITE_MICRO_TEST(OfflinePlannerBasic) {
  constexpr int number_tensors = 4;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {1,         0, number_tensors,
                                                   /*t0=*/0,
                                                   /*t1=*/48,
                                                   /*t2=*/0,
                                                   /*t3=*/48};
  constexpr int number_connections = 3;
  tflite::testing::NodeConnection node_list[number_connections] = {
      {/*input=*/{tflite::testing::t0},
       /*output=*/{tflite::testing::t1}},
      {/*input=*/{tflite::testing::t1},
       /*output=*/{tflite::testing::t2}},
      {/*input=*/{tflite::testing::t2},
       /*output=*/{tflite::testing::t3}}};

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections);

  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles));

  uint8_t* start = eval_tensors[0].data.uint8;
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[0].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, eval_tensors[1].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[2].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, eval_tensors[3].data.uint8 - start);
}

TF_LITE_MICRO_TEST(OfflinePlannerOverlappingAllocation) {
  constexpr int number_tensors = 4;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {/*version=*/1,
                                                   /*subgraph=*/0,
                                                   number_tensors,
                                                   /*t0=*/0,
                                                   /*t1=*/0,
                                                   /*t2=*/48,
                                                   /*t3=*/-1};

  int number_connections = 2;
  tflite::testing::NodeConnection node_list[2] = {
      {/*input, scratch=*/{tflite::testing::t0, tflite::testing::t1},
       /*output=*/{tflite::testing::t2}},
      {/*input=*/{tflite::testing::t2},
       /*output=*/{tflite::testing::t3}},
  };

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections);

  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles));

  uint8_t* start = eval_tensors[0].data.uint8;
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[0].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[1].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, eval_tensors[2].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[3].data.uint8 - start);
  // TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(48), context.tensors[0].bytes);
}

TF_LITE_MICRO_TEST(OfflinePlannerOfflineOnline) {
  constexpr int number_tensors = 5;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {/*version=*/1,
                                                   /*subgraph=*/0,
                                                   number_tensors,
                                                   /*t0=*/0,
                                                   /*t1=*/48,
                                                   /*t2=*/-1,
                                                   /*t3=*/0,
                                                   /*t4=*/-1};

  constexpr int number_connections = 2;
  tflite::testing::NodeConnection node_list[number_connections] = {
      {
          /*input, scratch=*/{tflite::testing::t0, tflite::testing::t1},
          /*output=*/{tflite::testing::t2},
      },
      {
          /*input=*/{tflite::testing::t2},
          /*output1, output2=*/{tflite::testing::t3, tflite::testing::t4},
      },
  };

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections);

  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles));

  uint8_t* start = eval_tensors[0].data.uint8;
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[0].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, eval_tensors[1].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(96, eval_tensors[2].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, eval_tensors[4].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[3].data.uint8 - start);
}

TF_LITE_MICRO_TEST(TestAllocatePersistentTfLiteTensor) {
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  constexpr size_t arena_size = 1024 * 12;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(allocator, nullptr);

  TfLiteTensor* tensor1 = allocator->AllocatePersistentTfLiteTensor(
      model, /*eval_tensors=*/nullptr, /*tensor_index=*/1);
  TF_LITE_MICRO_EXPECT_NE(tensor1, nullptr);
  TF_LITE_MICRO_EXPECT_NE(tensor1->quantization.params, nullptr);
  TF_LITE_MICRO_EXPECT_FALSE(tensor1->is_variable);

  TfLiteTensor* tensor2 = allocator->AllocatePersistentTfLiteTensor(
      model, /*eval_tensors=*/nullptr, /*tensor_index=*/2);
  TF_LITE_MICRO_EXPECT_NE(tensor2, nullptr);
  TF_LITE_MICRO_EXPECT_NE(tensor2->quantization.params, nullptr);
  TF_LITE_MICRO_EXPECT_FALSE(tensor2->is_variable);

  // The address of tensor1 should be higher than the address of tensor2 since
  // persistent allocations take place in the tail which grows downward.
  TF_LITE_MICRO_EXPECT_GT(tensor1, tensor2);
}

TF_LITE_MICRO_TEST(TestAllocateSingleTempTfLiteTensor) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(allocator, nullptr);

  TfLiteTensor* tensor1 = allocator->AllocateTempTfLiteTensor(
      model, /*eval_tensors=*/nullptr, /*tensor_index=*/1);
  TF_LITE_MICRO_EXPECT_NE(tensor1, nullptr);
}

TF_LITE_MICRO_TEST(TestAllocateChainOfTfLiteTensor) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT_NE(allocator, nullptr);

  TfLiteTensor* tensor1 = allocator->AllocateTempTfLiteTensor(
      model, /*eval_tensors=*/nullptr, /*tensor_index=*/1);
  TF_LITE_MICRO_EXPECT_NE(tensor1, nullptr);

  TfLiteTensor* tensor2 = allocator->AllocateTempTfLiteTensor(
      model, /*eval_tensors=*/nullptr, /*tensor_index=*/2);
  TF_LITE_MICRO_EXPECT_NE(tensor2, nullptr);

  // The address of tensor2 should be higher than the address of tensor1
  // (chained allocations):
  TF_LITE_MICRO_EXPECT_GT(tensor2, tensor1);
}

TF_LITE_MICRO_TEST(TestAllocateTfLiteTensorWithReset) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT(allocator != nullptr);

  TfLiteTensor* tensor1 = allocator->AllocateTempTfLiteTensor(
      model, /*eval_tensors=*/nullptr, /*tensor_index=*/1);
  TF_LITE_MICRO_EXPECT(tensor1 != nullptr);

  allocator->ResetTempAllocations();

  TfLiteTensor* tensor2 = allocator->AllocateTempTfLiteTensor(
      model, /*eval_tensors=*/nullptr, /*tensor_index=*/2);
  TF_LITE_MICRO_EXPECT(tensor2 != nullptr);

  // The address of tensor2 should be equal than the address of tensor1 since
  // allocations were not chained:
  TF_LITE_MICRO_EXPECT(tensor2 == tensor1);
}

TF_LITE_MICRO_TEST(TestOperatorInputsNotInSubgraphInputs) {
  constexpr int number_tensors = 5;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {/*version=*/1,
                                                   /*subgraph=*/0,
                                                   number_tensors,
                                                   /*t0=*/0,
                                                   /*t1=*/0,
                                                   /*t2=*/0,
                                                   /*t3=*/48,
                                                   /*t4=*/-1};

  constexpr int number_connections = 2;
  tflite::testing::NodeConnection node_list[number_connections] = {
      {// t0: input (actual input part of subgraph inputs as
       // well as operator inputs)
       // t1: scratch1 (only in operator inputs)
       // t2: scratch2 (only in operator inputs)
       {tflite::testing::t0, tflite::testing::t1, tflite::testing::t2},
       /*t3: output=*/{tflite::testing::t3}},
      {/*t3: input=*/{tflite::testing::t3},
       /*t4: output=*/{tflite::testing::t4}},
  };

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections,
      /*Only first tensor (t0) is in subgraph input list=*/1);

  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles));

  uint8_t* start = eval_tensors[0].data.uint8;
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[0].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[1].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[2].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, eval_tensors[3].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[4].data.uint8 - start);
}

TF_LITE_MICRO_TEST(TestTypicalFirstOpAndSecondOpWithScratchTensors) {
  constexpr int number_tensors = 6;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  tflite::NodeAndRegistration* node_and_registration;
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {/*version=*/1,
                                                   /*subgraph=*/0,
                                                   number_tensors,
                                                   /*t0=*/0,
                                                   /*t1=*/0,
                                                   /*t2=*/0,
                                                   /*t3=*/0,
                                                   /*t4=*/48,
                                                   /*t5=*/-1};

  constexpr int number_connections = 3;
  tflite::testing::NodeConnection node_list[number_connections] = {
      {/*t0: input (subgraph and operator input)=*/{tflite::testing::t0},
       /*t1: output=*/{tflite::testing::t1}},
      {// t1: input
       // t2: scratch1 (only in operator inputs)
       // t3: scratch2 (only in operator inputs)
       {tflite::testing::t1, tflite::testing::t2, tflite::testing::t3},

       /*t4: output=*/{tflite::testing::t4}},
      {/*t4: input=*/{tflite::testing::t4},
       /*t5: output=*/{tflite::testing::t5}},
  };

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections,
      /*Only first tensor (t0) is in subgraph input list=*/1);

  TfLiteEvalTensor* eval_tensors = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator->StartModelAllocation(model, op_resolver,
                                      &node_and_registration, &eval_tensors));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, eval_tensors,
                                                  &scratch_buffer_handles));

  uint8_t* start = eval_tensors[0].data.uint8;
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[0].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[1].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[2].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[3].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, eval_tensors[4].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, eval_tensors[5].data.uint8 - start);
}

TF_LITE_MICRO_TESTS_END
