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

#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kExpectedAlignment = 4;

void VerifyMockTensor(TfLiteTensor* tensor, bool is_variable = false) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(is_variable, tensor->is_variable);
  TF_LITE_MICRO_EXPECT_EQ(4, tensor->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor->data.raw);
  TF_LITE_MICRO_EXPECT_EQ(0,
                          (reinterpret_cast<std::uintptr_t>(tensor->data.raw) %
                           kExpectedAlignment));
}

void VerifyMockWeightTensor(TfLiteTensor* tensor) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor->data.raw);
}

void EnsureUniqueVariableTensorBuffer(TfLiteContext* context,
                                      const int variable_tensor_idx) {
  for (int i = 0; i < context->tensors_size; i++) {
    if (i != variable_tensor_idx) {
      TF_LITE_MICRO_EXPECT_NE(context->tensors[variable_tensor_idx].data.raw,
                              context->tensors[i].data.raw);
    }
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInitializeRuntimeTensor) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  TfLiteContext context;
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator simple_allocator(micro_test::reporter, arena,
                                                 arena_size);

  const tflite::Tensor* tensor = tflite::testing::Create1dFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
                     &simple_allocator, *tensor, buffers, micro_test::reporter,
                     &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(400, allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, allocated_tensor.data.i32);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteArenaRw, allocated_tensor.allocation_type);
}

TF_LITE_MICRO_TEST(TestInitializeQuantizedTensor) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  TfLiteContext context;
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator simple_allocator(micro_test::reporter, arena,
                                                 arena_size);

  const tflite::Tensor* tensor =
      tflite::testing::CreateQuantizedFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
                     &simple_allocator, *tensor, buffers, micro_test::reporter,
                     &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(400, allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, allocated_tensor.data.i32);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteArenaRw, allocated_tensor.allocation_type);
}

TF_LITE_MICRO_TEST(TestMissingQuantization) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  TfLiteContext context;
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator simple_allocator(micro_test::reporter, arena,
                                                 arena_size);

  const tflite::Tensor* tensor =
      tflite::testing::CreateMissingQuantizationFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
                     &simple_allocator, *tensor, buffers, micro_test::reporter,
                     &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(400, allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, allocated_tensor.data.i32);
}

TF_LITE_MICRO_TEST(TestFinishTensorAllocation) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  TfLiteContext context;
  constexpr size_t arena_size =
      760 /* minimal arena size at the time of writting */ +
      16 /* alignment */ + 100 /* leave some headroom for future proof */;
  uint8_t arena[arena_size];
  tflite::MicroAllocator allocator(&context, model, arena, arena_size,
                                   micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(4, context.tensors_size);
  // Memory planning hasn't been finalized, so the used bytes is unknown.
  TF_LITE_MICRO_EXPECT_EQ(0, allocator.used_bytes());

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.FinishTensorAllocation());
  // No allocation to be done afterwards.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError, allocator.FinishTensorAllocation());

  // NOTE: Tensor indexes match the values in GetSimpleMockModel().
  tflite::testing::VerifyMockTensor(&context.tensors[0]);
  tflite::testing::VerifyMockWeightTensor(&context.tensors[1]);
  tflite::testing::VerifyMockTensor(&context.tensors[2]);
  tflite::testing::VerifyMockTensor(&context.tensors[3]);

  TF_LITE_MICRO_EXPECT_NE(context.tensors[1].data.raw,
                          context.tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(context.tensors[2].data.raw,
                          context.tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(context.tensors[1].data.raw,
                          context.tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_NE(context.tensors[3].data.raw,
                          context.tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(context.tensors[3].data.raw,
                          context.tensors[1].data.raw);
  TF_LITE_MICRO_EXPECT_NE(context.tensors[3].data.raw,
                          context.tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_LE(allocator.used_bytes(), 760 + 100);
}

TF_LITE_MICRO_TEST(TestAllocationForModelsWithBranches) {
  const tflite::Model* model = tflite::testing::GetSimpleModelWithBranch();
  TfLiteContext context;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator allocator(&context, model, arena, arena_size,
                                   micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.FinishTensorAllocation());

  uint8_t* start = context.tensors[0].data.uint8;
  // Check test_helpers.cc BuildSimpleModelWithBranch for model structure.
  // t0 is the first tensor, so place it in offset 0.
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[0].data.uint8 - start);
  // bytes = 2 * 2 * 3 * sizeof(float32) = 48, same for other tensors.
  TF_LITE_MICRO_EXPECT_EQ(48, context.tensors[0].bytes);
  // t1 can't reuse any memory, as n0 requires both t0 and t1.
  TF_LITE_MICRO_EXPECT_EQ(96, context.tensors[1].data.uint8 - start);
  // t2 can't reuse any memory, as n1 requires both t0 and t2. Also n2 requires
  // both t1 and t2.
  TF_LITE_MICRO_EXPECT_EQ(48, context.tensors[2].data.uint8 - start);
  // t3 reuses the same memory from t0 as t0 is not an input to any node.
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[3].data.uint8 - start);
}

TF_LITE_MICRO_TEST(TestFinishComplexTensorAllocation) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  TfLiteContext context;
  constexpr size_t arena_size = 2048;
  uint8_t arena[arena_size];
  tflite::MicroAllocator allocator(&context, model, arena, arena_size,
                                   micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(10, context.tensors_size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.FinishTensorAllocation());
  // No allocation to be done afterwards.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError, allocator.FinishTensorAllocation());

  // NOTE: Tensor indexes match the values in GetComplexMockModel().
  tflite::testing::VerifyMockTensor(&context.tensors[0]);
  tflite::testing::VerifyMockTensor(&context.tensors[1],
                                    true /* is_variable */);
  tflite::testing::VerifyMockWeightTensor(&context.tensors[2]);
  tflite::testing::VerifyMockTensor(&context.tensors[3]);
  tflite::testing::VerifyMockTensor(&context.tensors[4],
                                    true /* is_variable */);
  tflite::testing::VerifyMockWeightTensor(&context.tensors[5]);
  tflite::testing::VerifyMockTensor(&context.tensors[6]);
  tflite::testing::VerifyMockTensor(&context.tensors[7],
                                    true /* is_variable */);
  tflite::testing::VerifyMockWeightTensor(&context.tensors[8]);
  tflite::testing::VerifyMockTensor(&context.tensors[9]);

  // Ensure that variable tensors have unique address
  tflite::testing::EnsureUniqueVariableTensorBuffer(&context, 1);
  tflite::testing::EnsureUniqueVariableTensorBuffer(&context, 4);
  tflite::testing::EnsureUniqueVariableTensorBuffer(&context, 7);
}

TF_LITE_MICRO_TEST(OfflinePlannerBranchesAllOnline) {
  int version = 1;
  int subgraph = 0;
  int nbr_tensors = 4;
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                nbr_tensors] = {version, subgraph,
                                                nbr_tensors,  // header
                                                // memory offsets:
                                                -1, -1, -1, -1};

  // The structure is identical to the one in
  // TestAllocationForModelsWithBranches
  int num_conns = 3;
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
      nbr_tensors, metadata_buffer, node_list, num_conns);

  TfLiteContext context;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator allocator(&context, model, arena, arena_size,
                                   micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.FinishTensorAllocation());

  // Since all of the tensors are online planned and the model structure is
  // identical to that in TestAllocationForModelsWithBranches,
  // the offsets be should identical to that test.
  uint8_t* start = context.tensors[0].data.uint8;
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[0].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, context.tensors[0].bytes);
  TF_LITE_MICRO_EXPECT_EQ(96, context.tensors[1].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, context.tensors[2].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[3].data.uint8 - start);
}

TF_LITE_MICRO_TEST(OfflinePlannerBasic) {
  int nbr_tensors = 4;
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                nbr_tensors] = {1,  0, nbr_tensors,
                                                0,    // t0
                                                48,   // t1
                                                0,    // t2
                                                48};  // t3

  int t0 = 0;
  int t1 = 1;
  int t2 = 2;
  int t3 = 3;

  int num_conns = 3;
  tflite::testing::NodeConnection node_list[3] = {{
                                                      {t0},  // input
                                                      {t1}   // output
                                                  },
                                                  {
                                                      {t1},  // input
                                                      {t2}   // output
                                                  },
                                                  {
                                                      {t2},  // input
                                                      {t3}   // output
                                                  }};

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      nbr_tensors, metadata_buffer, node_list, num_conns);

  TfLiteContext context;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator allocator(&context, model, arena, arena_size,
                                   micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.FinishTensorAllocation());

  uint8_t* start = context.tensors[0].data.uint8;
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[0].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, context.tensors[1].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[2].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, context.tensors[3].data.uint8 - start);
}

TF_LITE_MICRO_TEST(OfflinePlannerOverlappingAllocation) {
  int nbr_tensors = 4;
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                nbr_tensors] = {
      1, 0, nbr_tensors,  // header: version, subgraph, nbr tensors
      // memory offsets:
      0,    // t0
      0,    // t1
      48,   // t2
      -1};  // t3

  int t0 = 0;
  int t1 = 1;
  int t2 = 2;
  int t3 = 3;

  int num_conns = 2;
  tflite::testing::NodeConnection node_list[2] = {
      {
          {t0, t1},  // input, scratch
          {t2}       // output
      },
      {
          {t2},  // input
          {t3}   // output
      },
  };

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      nbr_tensors, metadata_buffer, node_list, num_conns);

  TfLiteContext context;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator allocator(&context, model, arena, arena_size,
                                   micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.FinishTensorAllocation());

  uint8_t* start = context.tensors[0].data.uint8;
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[0].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[1].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, context.tensors[2].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[3].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, context.tensors[0].bytes);
}

TF_LITE_MICRO_TEST(OfflinePlannerOfflineOnline) {
  int nbr_tensors = 5;
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                nbr_tensors] = {
      1, 0, nbr_tensors,  // header: version, subgraph, nbr tensors
      // memory offsets:
      0,    // t0
      48,   // t1
      -1,   // t2
      0,    // t3
      -1};  // t4

  int t0 = 0;
  int t1 = 1;
  int t2 = 2;
  int t3 = 3;
  int t4 = 4;

  int num_conns = 2;
  tflite::testing::NodeConnection node_list[2] = {
      {
          {t0, t1},  // input, scratch
          {t2},      // output
      },
      {
          {t2},      // input
          {t3, t4},  // output1, output2
      },
  };

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      nbr_tensors, metadata_buffer, node_list, num_conns);

  TfLiteContext context;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator allocator(&context, model, arena, arena_size,
                                   micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.FinishTensorAllocation());

  uint8_t* start = context.tensors[0].data.uint8;
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[0].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, context.tensors[1].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(96, context.tensors[2].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(48, context.tensors[4].data.uint8 - start);
  TF_LITE_MICRO_EXPECT_EQ(0, context.tensors[3].data.uint8 - start);
}

TF_LITE_MICRO_TESTS_END
