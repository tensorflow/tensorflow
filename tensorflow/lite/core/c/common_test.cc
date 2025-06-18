/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/c/common.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/test_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
using ::testing::AnyOf;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Not;

// NOTE: this tests only the TfLiteIntArray part of context.
// most of common.h is provided in the context of using it with
// interpreter.h and interpreter.cc, so interpreter_test.cc tests context
// structures more thoroughly.

TEST(IntArray, TestIntArrayCreate) {
  TfLiteIntArray* a = TfLiteIntArrayCreate(0);
  TfLiteIntArray* b = TfLiteIntArrayCreate(3);
  TfLiteIntArrayFree(a);
  TfLiteIntArrayFree(b);
}

TEST(IntArray, TestIntArrayCopy) {
  TfLiteIntArray* a = TfLiteIntArrayCreate(2);
  a->data[0] = 22;
  a->data[1] = 24;
  TfLiteIntArray* b = TfLiteIntArrayCopy(a);
  ASSERT_NE(a, b);
  ASSERT_EQ(a->size, b->size);
  ASSERT_EQ(a->data[0], b->data[0]);
  ASSERT_EQ(a->data[1], b->data[1]);
  TfLiteIntArrayFree(a);
  TfLiteIntArrayFree(b);
}

TEST(IntArray, TestIntArrayEqual) {
  TfLiteIntArray* a = TfLiteIntArrayCreate(1);
  a->data[0] = 1;
  TfLiteIntArray* b = TfLiteIntArrayCreate(2);
  b->data[0] = 5;
  b->data[1] = 6;
  TfLiteIntArray* c = TfLiteIntArrayCreate(2);
  c->data[0] = 5;
  c->data[1] = 6;
  TfLiteIntArray* d = TfLiteIntArrayCreate(2);
  d->data[0] = 6;
  d->data[1] = 6;
  EXPECT_FALSE(TfLiteIntArrayEqual(a, b));
  EXPECT_TRUE(TfLiteIntArrayEqual(b, c));
  EXPECT_TRUE(TfLiteIntArrayEqual(b, b));
  EXPECT_FALSE(TfLiteIntArrayEqual(c, d));
  EXPECT_FALSE(TfLiteIntArrayEqual(nullptr, a));
  EXPECT_FALSE(TfLiteIntArrayEqual(a, nullptr));
  EXPECT_TRUE(TfLiteIntArrayEqual(nullptr, nullptr));
  TfLiteIntArrayFree(a);
  TfLiteIntArrayFree(b);
  TfLiteIntArrayFree(c);
  TfLiteIntArrayFree(d);
}

TEST(FloatArray, TestFloatArrayCreate) {
  TfLiteFloatArray* a = TfLiteFloatArrayCreate(0);
  TfLiteFloatArray* b = TfLiteFloatArrayCreate(3);
  TfLiteFloatArrayFree(a);
  TfLiteFloatArrayFree(b);
}

TEST(FloatArray, TestFloatArrayCopy) {
  TfLiteFloatArray* a = TfLiteFloatArrayCreate(2);
  a->data[0] = 22.0;
  a->data[1] = 24.0;
  TfLiteFloatArray* b = TfLiteFloatArrayCopy(a);
  ASSERT_NE(a, b);
  ASSERT_EQ(a->size, b->size);
  ASSERT_EQ(a->data[0], b->data[0]);
  ASSERT_EQ(a->data[1], b->data[1]);
  TfLiteFloatArrayFree(a);
  TfLiteFloatArrayFree(b);
}

TEST(Types, TestTypeNames) {
  auto type_name = [](TfLiteType t) {
    return std::string(TfLiteTypeGetName(t));
  };
  EXPECT_EQ(type_name(kTfLiteNoType), "NOTYPE");
  EXPECT_EQ(type_name(kTfLiteFloat64), "FLOAT64");
  EXPECT_EQ(type_name(kTfLiteFloat32), "FLOAT32");
  EXPECT_EQ(type_name(kTfLiteFloat16), "FLOAT16");
  EXPECT_EQ(type_name(kTfLiteBFloat16), "BFLOAT16");
  EXPECT_EQ(type_name(kTfLiteInt16), "INT16");
  EXPECT_EQ(type_name(kTfLiteUInt16), "UINT16");
  EXPECT_EQ(type_name(kTfLiteInt32), "INT32");
  EXPECT_EQ(type_name(kTfLiteUInt32), "UINT32");
  EXPECT_EQ(type_name(kTfLiteUInt8), "UINT8");
  EXPECT_EQ(type_name(kTfLiteUInt64), "UINT64");
  EXPECT_EQ(type_name(kTfLiteInt8), "INT8");
  EXPECT_EQ(type_name(kTfLiteInt64), "INT64");
  EXPECT_EQ(type_name(kTfLiteBool), "BOOL");
  EXPECT_EQ(type_name(kTfLiteComplex64), "COMPLEX64");
  EXPECT_EQ(type_name(kTfLiteComplex128), "COMPLEX128");
  EXPECT_EQ(type_name(kTfLiteString), "STRING");
  EXPECT_EQ(type_name(kTfLiteResource), "RESOURCE");
  EXPECT_EQ(type_name(kTfLiteVariant), "VARIANT");
  EXPECT_EQ(type_name(kTfLiteInt4), "INT4");
}

TEST(Quantization, TestQuantizationFree) {
  TfLiteTensor t;
  // Set these values, otherwise TfLiteTensorFree has uninitialized values.
  t.allocation_type = kTfLiteArenaRw;
  t.dims = nullptr;
  t.dims_signature = nullptr;
  t.quantization.type = kTfLiteAffineQuantization;
  t.sparsity = nullptr;
  auto* params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  params->scale = TfLiteFloatArrayCreate(3);
  params->zero_point = TfLiteIntArrayCreate(3);
  t.quantization.params = reinterpret_cast<void*>(params);
  TfLiteTensorFree(&t);
}

TEST(Sparsity, TestSparsityFree) {
  TfLiteTensor t = {};
  // Set these values, otherwise TfLiteTensorFree has uninitialized values.
  t.allocation_type = kTfLiteArenaRw;
  t.dims = nullptr;
  t.dims_signature = nullptr;

  // A dummy CSR sparse matrix.
  t.sparsity = static_cast<TfLiteSparsity*>(malloc(sizeof(TfLiteSparsity)));
  t.sparsity->traversal_order = TfLiteIntArrayCreate(2);
  t.sparsity->block_map = nullptr;

  t.sparsity->dim_metadata = static_cast<TfLiteDimensionMetadata*>(
      malloc(sizeof(TfLiteDimensionMetadata) * 2));
  t.sparsity->dim_metadata_size = 2;

  t.sparsity->dim_metadata[0].format = kTfLiteDimDense;
  t.sparsity->dim_metadata[0].dense_size = 4;

  t.sparsity->dim_metadata[1].format = kTfLiteDimSparseCSR;
  t.sparsity->dim_metadata[1].array_segments = TfLiteIntArrayCreate(2);
  t.sparsity->dim_metadata[1].array_indices = TfLiteIntArrayCreate(3);

  TfLiteTensorFree(&t);
}

TEST(TensorCopy, TensorCopy_VALID) {
  const int kNumElements = 32;
  const int kBytes = sizeof(float) * kNumElements;
  TfLiteTensor src;
  TfLiteTensor dst;
  TfLiteDelegate delegate;
  memset(&delegate, 0, sizeof(delegate));
  memset(&src, 0, sizeof(TfLiteTensor));
  memset(&dst, 0, sizeof(TfLiteTensor));
  src.data.raw = static_cast<char*>(malloc(kBytes));
  for (int i = 0; i < kNumElements; ++i) {
    src.data.f[i] = i;
  }
  dst.data.raw = static_cast<char*>(malloc(kBytes));

  src.bytes = dst.bytes = kBytes;
  src.delegate = &delegate;
  src.data_is_stale = true;
  src.allocation_type = kTfLiteDynamic;
  src.type = kTfLiteFloat32;
  src.dims = TfLiteIntArrayCreate(1);
  src.dims->data[0] = 1;
  src.dims_signature = TfLiteIntArrayCopy(src.dims);
  src.buffer_handle = 5;

  EXPECT_EQ(kTfLiteOk, TfLiteTensorCopy(&src, &dst));

  EXPECT_EQ(dst.bytes, src.bytes);
  EXPECT_EQ(dst.delegate, src.delegate);
  EXPECT_EQ(dst.data_is_stale, src.data_is_stale);
  EXPECT_EQ(dst.type, src.type);
  EXPECT_EQ(1, TfLiteIntArrayEqual(dst.dims, src.dims));
  EXPECT_EQ(dst.buffer_handle, src.buffer_handle);
  for (int i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(dst.data.f[i], src.data.f[i]);
  }

  TfLiteTensorFree(&src);
  // We don't change allocation type, and since the test keeps the dst
  // allocation as non dynamic, then we have to delete it manually.
  free(dst.data.raw);
  TfLiteTensorFree(&dst);
}

TEST(TensorCopy, TensorCopy_INVALID) {
  TfLiteTensor src;
  TfLiteTensor dst;

  // Nullptr passed, should just return.
  EXPECT_EQ(kTfLiteOk, TfLiteTensorCopy(&src, nullptr));
  EXPECT_EQ(kTfLiteOk, TfLiteTensorCopy(nullptr, &dst));

  // Incompatible sizes passed.
  src.bytes = 10;
  dst.bytes = 12;
  EXPECT_EQ(kTfLiteError, TfLiteTensorCopy(&src, &dst));
}

TEST(TestTensorRealloc, TensorReallocMoreBytesSucceeds) {
  const TfLiteType t = kTfLiteFloat32;
  const int num_elements = 4;
  const int new_num_elements = 6;
  const size_t bytes = sizeof(float) * num_elements;
  const size_t new_bytes = sizeof(float) * new_num_elements;
  float* data = (float*)malloc(bytes);
  memset(data, 0, bytes);

  TfLiteIntArray* dims = ConvertVectorToTfLiteIntArray({num_elements});
  TfLiteTensor* tensor = (TfLiteTensor*)malloc(sizeof(TfLiteTensor));
  tensor->sparsity = nullptr;
  tensor->quantization.type = kTfLiteNoQuantization;
  tensor->bytes = bytes;
  tensor->type = t;
  tensor->data.data = data;
  tensor->allocation_type = kTfLiteDynamic;
  tensor->dims = dims;
  tensor->dims_signature = TfLiteIntArrayCopy(dims);

  ASSERT_EQ(TfLiteTensorRealloc(new_bytes, tensor), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, new_bytes);

  ASSERT_THAT(std::vector<int>(tensor->data.f, tensor->data.f + num_elements),
              ElementsAreArray({0, 0, 0, 0}));

  TfLiteTensorFree(tensor);
  free(tensor);
}

TEST(TestTensorRealloc, TensorReallocLessBytesSucceeds) {
  const TfLiteType t = kTfLiteFloat32;
  const int num_elements = 4;
  const int new_num_elements = 2;
  const size_t bytes = sizeof(float) * num_elements;
  const size_t new_bytes = sizeof(float) * new_num_elements;
  float* data = (float*)malloc(bytes);
  memset(data, 0, bytes);

  TfLiteIntArray* dims = ConvertVectorToTfLiteIntArray({num_elements});
  TfLiteTensor* tensor = (TfLiteTensor*)malloc(sizeof(TfLiteTensor));
  tensor->sparsity = nullptr;
  tensor->bytes = bytes;
  tensor->type = t;
  tensor->data.data = data;
  tensor->allocation_type = kTfLiteDynamic;
  tensor->dims = dims;
  tensor->dims_signature = TfLiteIntArrayCopy(dims);
  tensor->quantization.type = kTfLiteNoQuantization;

  ASSERT_EQ(TfLiteTensorRealloc(new_bytes, tensor), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, new_bytes);

  ASSERT_THAT(std::vector<int>(tensor->data.f, tensor->data.f + 2),
              ElementsAreArray({0, 0}));

  TfLiteTensorFree(tensor);
  free(tensor);
}

TEST(TestTensorRealloc, TensorReallocNonDynamicNoChange) {
  const TfLiteType t = kTfLiteFloat32;
  const int num_elements = 4;
  const int new_num_elements = 6;
  const size_t bytes = sizeof(float) * num_elements;
  const size_t new_bytes = sizeof(float) * new_num_elements;
  float* data = (float*)malloc(bytes);
  memset(data, 0, bytes);

  TfLiteIntArray* dims = ConvertVectorToTfLiteIntArray({num_elements});
  TfLiteTensor* tensor = (TfLiteTensor*)malloc(sizeof(TfLiteTensor));
  tensor->sparsity = nullptr;
  tensor->bytes = bytes;
  tensor->type = t;
  tensor->data.data = data;
  tensor->allocation_type = kTfLiteArenaRw;
  tensor->quantization.type = kTfLiteNoQuantization;
  tensor->dims = dims;
  tensor->dims_signature = TfLiteIntArrayCopy(dims);

  EXPECT_EQ(TfLiteTensorRealloc(new_bytes, tensor), kTfLiteOk);
  // Tensor should still be intact.
  EXPECT_EQ(tensor->bytes, bytes);

  EXPECT_THAT(std::vector<int>(tensor->data.i32, tensor->data.i32 + 4),
              ElementsAreArray({0, 0, 0, 0}));

  free(tensor->data.data);
  TfLiteTensorFree(tensor);
  free(tensor);
}

TEST(TestTensorRealloc, TensorReallocNumByte0) {
  const TfLiteType t = kTfLiteFloat32;
  const int num_elements = 4;
  const int new_num_elements = 0;
  const size_t bytes = sizeof(float) * num_elements;
  const size_t new_bytes = sizeof(float) * new_num_elements;
  float* data = (float*)malloc(bytes);
  memset(data, 0, bytes);

  TfLiteIntArray* dims = ConvertVectorToTfLiteIntArray({num_elements});
  TfLiteTensor* tensor = (TfLiteTensor*)malloc(sizeof(TfLiteTensor));
  tensor->sparsity = nullptr;
  tensor->bytes = bytes;
  tensor->type = t;
  tensor->data.data = data;
  tensor->allocation_type = kTfLiteDynamic;
  tensor->quantization.type = kTfLiteNoQuantization;
  tensor->dims = dims;
  tensor->dims_signature = TfLiteIntArrayCopy(dims);

  EXPECT_EQ(TfLiteTensorRealloc(new_bytes, tensor), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 0);

  TfLiteTensorFree(tensor);
  free(tensor);
}

TEST(TestTensorRealloc, TensorReallocLargeBytesFails) {
  const TfLiteType t = kTfLiteFloat32;
  const int num_elements = 4;
  const size_t bytes = sizeof(float) * num_elements;

  float* data = (float*)malloc(bytes);
  memset(data, 0, bytes);

  TfLiteIntArray* dims = ConvertVectorToTfLiteIntArray({num_elements});
  TfLiteTensor* tensor = (TfLiteTensor*)malloc(sizeof(TfLiteTensor));
  tensor->sparsity = nullptr;
  tensor->bytes = bytes;
  tensor->type = t;
  tensor->data.data = data;
  tensor->allocation_type = kTfLiteDynamic;
  tensor->dims = dims;
  tensor->dims_signature = TfLiteIntArrayCopy(dims);
  tensor->quantization.type = kTfLiteNoQuantization;

  const size_t large_bytes = std::numeric_limits<size_t>::max() - 16;
  // Subtract 16 to account for adding 16 for XNN_EXTRA_BYTES
  EXPECT_EQ(TfLiteTensorRealloc(large_bytes, tensor), kTfLiteError);

  TfLiteTensorFree(tensor);
  free(data);
  free(tensor);
}

TEST(TestTfLiteTensorGetDimsSignature, NullDimsSignatureReturnsDims) {
  TfLiteTensor t{
      .dims = ConvertVectorToTfLiteIntArray({1, 2, 3}),
      .dims_signature = nullptr,
  };

  EXPECT_THAT(TfLiteTensorGetDimsSignature(&t), TfLiteArrayIs({1, 2, 3}));

  TfLiteTensorFree(&t);
}

TEST(TestTfLiteTensorGetDimsSignature, EmptyDimsSignatureReturnsDims) {
  TfLiteTensor t{
      .dims = ConvertVectorToTfLiteIntArray({1, 2, 3}),
      .dims_signature = ConvertVectorToTfLiteIntArray({}),
  };

  EXPECT_THAT(TfLiteTensorGetDimsSignature(&t), TfLiteArrayIs({1, 2, 3}));

  TfLiteTensorFree(&t);
}

TEST(TestTfLiteTensorGetDimsSignature,
     NonEmptyDimsSignatureReturnsDimsSignature) {
  TfLiteTensor t{
      .dims = ConvertVectorToTfLiteIntArray({1, 2, 3}),
      .dims_signature = ConvertVectorToTfLiteIntArray({4, -1, 5}),
  };

  EXPECT_THAT(TfLiteTensorGetDimsSignature(&t), TfLiteArrayIs({4, -1, 5}));

  TfLiteTensorFree(&t);
}

TEST(TestTfLiteTensorGetAllocationStrategy, MemNoneIsAllocatedWithNone) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteTensorGetAllocationStrategy(&t),
            kTfLiteAllocationStrategyNone);
}

TEST(TestTfLiteTensorGetAllocationStrategy, MmapRoIsAllocatedWithMMap) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteTensorGetAllocationStrategy(&t),
            kTfLiteAllocationStrategyMMap);
}

TEST(TestTfLiteTensorGetAllocationStrategy, ArenaRwIsAllocatedWithArena) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteTensorGetAllocationStrategy(&t),
            kTfLiteAllocationStrategyArena);
}

TEST(TestTfLiteTensorGetAllocationStrategy,
     ArenaRwPersistentIsAllocatedWithArena) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteTensorGetAllocationStrategy(&t),
            kTfLiteAllocationStrategyArena);
}

TEST(TestTfLiteTensorGetAllocationStrategy, DynamicIsAllocatedWithMalloc) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteTensorGetAllocationStrategy(&t),
            kTfLiteAllocationStrategyMalloc);
}

TEST(TestTfLiteTensorGetAllocationStrategy, PersistentRoIsAllocatedWithMalloc) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteTensorGetAllocationStrategy(&t),
            kTfLiteAllocationStrategyMalloc);
}

TEST(TestTfLiteTensorGetAllocationStrategy, CustomIsAllocatedWithUnknown) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteTensorGetAllocationStrategy(&t),
            kTfLiteAllocationStrategyUnknown);
}

TEST(TestTfLiteTensorGetAllocationStrategy, VariantObjectIsAllocatedWithNew) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteTensorGetAllocationStrategy(&t),
            kTfLiteAllocationStrategyNew);
}

TEST(TestTfLiteTensorGetBufferAddressStability,
     MemNoneBufferIsStableAcrossRuns) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteTensorGetBufferAddressStability(&t),
            kTfLiteRunStabilityAcrossRuns);
}

TEST(TestTfLiteTensorGetBufferAddressStability,
     MmapRoBufferIsStableAcrossRuns) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteTensorGetBufferAddressStability(&t),
            kTfLiteRunStabilityAcrossRuns);
}

TEST(TestTfLiteTensorGetBufferAddressStability, ArenaRwBufferIsStableUnstable) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteTensorGetBufferAddressStability(&t),
            kTfLiteRunStabilityUnstable);
}

TEST(TestTfLiteTensorGetBufferAddressStability,
     ArenaRwPersistentBufferIsStableUnstable) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteTensorGetBufferAddressStability(&t),
            kTfLiteRunStabilityUnstable);
}

TEST(TestTfLiteTensorGetBufferAddressStability,
     DynamicBufferIsStableSingleRun) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteTensorGetBufferAddressStability(&t),
            kTfLiteRunStabilitySingleRun);
}

TEST(TestTfLiteTensorGetBufferAddressStability,
     PersistentRoBufferIsStableSingleRun) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteTensorGetBufferAddressStability(&t),
            kTfLiteRunStabilitySingleRun);
}

TEST(TestTfLiteTensorGetBufferAddressStability, CustomBufferIsStableUnknown) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteTensorGetBufferAddressStability(&t),
            kTfLiteRunStabilityUnknown);
}

TEST(TestTfLiteTensorGetBufferAddressStability,
     VariantObjectBufferIsStableAcrossRuns) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteTensorGetBufferAddressStability(&t),
            kTfLiteRunStabilityAcrossRuns);
}

TEST(TestTfLiteTensorGetDataStability, MemNoneDataIsStableAcrossRuns) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteTensorGetDataStability(&t), kTfLiteRunStabilityAcrossRuns);
}

TEST(TestTfLiteTensorGetDataStability, MmapRoDataIsStableAcrossRuns) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteTensorGetDataStability(&t), kTfLiteRunStabilityAcrossRuns);
}

TEST(TestTfLiteTensorGetDataStability, ArenaRwDataIsStableSingleRun) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteTensorGetDataStability(&t), kTfLiteRunStabilitySingleRun);
}

TEST(TestTfLiteTensorGetDataStability,
     ArenaRwPersistentDataIsStableAcrossRuns) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteTensorGetDataStability(&t), kTfLiteRunStabilityAcrossRuns);
}

TEST(TestTfLiteTensorGetDataStability, DynamicDataIsStableSingleRun) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteTensorGetDataStability(&t), kTfLiteRunStabilitySingleRun);
}

TEST(TestTfLiteTensorGetDataStability, PersistentRoDataIsStableSingleRun) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteTensorGetDataStability(&t), kTfLiteRunStabilitySingleRun);
}

TEST(TestTfLiteTensorGetDataStability, CustomDataIsStableUnknown) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteTensorGetDataStability(&t), kTfLiteRunStabilityUnknown);
}

TEST(TestTfLiteTensorGetDataStability, VariantObjectDataIsStableSingleRun) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteTensorGetDataStability(&t), kTfLiteRunStabilitySingleRun);
}

TEST(TestTfLiteTensorGetDataKnownStep, MemNoneDataIsKnownAtInit) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteTensorGetDataKnownStep(&t), kTfLiteRunStepInit);
}

TEST(TestTfLiteTensorGetDataKnownStep, MmapRoDataIsKnownAtInit) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteTensorGetDataKnownStep(&t), kTfLiteRunStepInit);
}

TEST(TestTfLiteTensorGetDataKnownStep, ArenaRwDataIsKnownAtEval) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteTensorGetDataKnownStep(&t), kTfLiteRunStepEval);
}

TEST(TestTfLiteTensorGetDataKnownStep, ArenaRwPersistentDataIsKnownAtEval) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteTensorGetDataKnownStep(&t), kTfLiteRunStepEval);
}

TEST(TestTfLiteTensorGetDataKnownStep, DynamicDataIsKnownAtEval) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteTensorGetDataKnownStep(&t), kTfLiteRunStepEval);
}

TEST(TestTfLiteTensorGetDataKnownStep, PersistentRoDataIsKnownAtPrepare) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteTensorGetDataKnownStep(&t), kTfLiteRunStepPrepare);
}

TEST(TestTfLiteTensorGetDataKnownStep, CustomDataIsKnownAtUnknown) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteTensorGetDataKnownStep(&t), kTfLiteRunStepUnknown);
}

TEST(TestTfLiteTensorGetDataKnownStep, VariantObjectDataIsKnownAtEval) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteTensorGetDataKnownStep(&t), kTfLiteRunStepEval);
}

TEST(TestTfLiteTensorGetShapeKnownStep, MemNoneShapeIsKnownAtInit) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteTensorGetShapeKnownStep(&t), kTfLiteRunStepInit);
}

TEST(TestTfLiteTensorGetShapeKnownStep, MmapRoShapeIsKnownAtInit) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteTensorGetShapeKnownStep(&t), kTfLiteRunStepInit);
}

TEST(TestTfLiteTensorGetShapeKnownStep, ArenaRwShapeIsKnownAtPrepare) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteTensorGetShapeKnownStep(&t), kTfLiteRunStepPrepare);
}

TEST(TestTfLiteTensorGetShapeKnownStep,
     ArenaRwPersistentShapeIsKnownAtPrepare) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteTensorGetShapeKnownStep(&t), kTfLiteRunStepPrepare);
}

TEST(TestTfLiteTensorGetShapeKnownStep, DynamicShapeIsKnownAtEval) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteTensorGetShapeKnownStep(&t), kTfLiteRunStepEval);
}

TEST(TestTfLiteTensorGetShapeKnownStep, PersistentRoShapeIsKnownAtPrepare) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteTensorGetShapeKnownStep(&t), kTfLiteRunStepPrepare);
}

TEST(TestTfLiteTensorGetShapeKnownStep, CustomShapeIsKnownAtUnknown) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteTensorGetShapeKnownStep(&t), kTfLiteRunStepUnknown);
}

TEST(TestTfLiteTensorGetShapeKnownStep, VariantObjectShapeIsKnownAtEval) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteTensorGetShapeKnownStep(&t), kTfLiteRunStepEval);
}

struct Foo {
  int data;
  bool copied;
};

class VariantFoo : public AbstractVariantData<VariantFoo> {
 public:
  explicit VariantFoo(int number) : foo_data_(Foo{number, false}) {}
  VariantFoo(const VariantFoo& other) {
    foo_data_ = other.foo_data_;
    foo_data_.copied = true;
  }
  int GetFooInt() { return foo_data_.data; }
  bool GetFooCopied() { return foo_data_.copied; }

 private:
  Foo foo_data_;
};

// Want to validate the TfLiteTensorVariantRealloc works as intended
// with > 1 constructor arguments.
class VariantFoo2 : public AbstractVariantData<VariantFoo2> {
 public:
  explicit VariantFoo2(int number, float float_number)
      : foo_data_(Foo{number, false}), float_data_(float_number) {}
  VariantFoo2(const VariantFoo2& other) {
    foo_data_ = other.foo_data_;
    foo_data_.copied = true;
    float_data_ = other.float_data_;
  }
  int GetFooInt() { return foo_data_.data; }
  bool GetFooCopied() { return foo_data_.copied; }
  float GetFloatData() { return float_data_; }

 private:
  Foo foo_data_;
  float float_data_;
};

TEST(TestTfLiteReallocWithObject, ConstructSingleParamVariant) {
  TensorUniquePtr t = BuildTfLiteTensor();
  t->type = kTfLiteVariant;
  ASSERT_EQ((TfLiteTensorVariantRealloc<VariantFoo>(t.get(), 3)), kTfLiteOk);
  ASSERT_EQ(reinterpret_cast<VariantFoo*>(t->data.data)->GetFooInt(), 3);
  ASSERT_EQ(t->type, kTfLiteVariant);
  ASSERT_EQ(t->allocation_type, kTfLiteVariantObject);
}

TEST(TestTfLiteReallocWithObject, ConstructMultiParamVariant) {
  TensorUniquePtr t = BuildTfLiteTensor();
  t->type = kTfLiteVariant;
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo2, int, float>(t.get(), 3, 1.0)),
      kTfLiteOk);
  VariantFoo2* data = reinterpret_cast<VariantFoo2*>(t->data.data);
  ASSERT_EQ(data->GetFooInt(), 3);
  ASSERT_EQ(data->GetFloatData(), 1.0);
  ASSERT_EQ(t->type, kTfLiteVariant);
  ASSERT_EQ(t->allocation_type, kTfLiteVariantObject);
}

TEST(TestTfLiteReallocWithObject,
     ConstructSingleParamVariantWithAlreadyAllocated) {
  TensorUniquePtr t = BuildTfLiteTensor();
  t->type = kTfLiteVariant;
  ASSERT_EQ((TfLiteTensorVariantRealloc<VariantFoo>(t.get(), 3)), kTfLiteOk);
  void* before_address = t->data.data;
  ASSERT_EQ((TfLiteTensorVariantRealloc<VariantFoo>(t.get(), 5)), kTfLiteOk);
  EXPECT_EQ(t->data.data, before_address);
  EXPECT_EQ(reinterpret_cast<VariantFoo*>(t->data.data)->GetFooInt(), 5);
  EXPECT_EQ(t->type, kTfLiteVariant);
  EXPECT_EQ(t->allocation_type, kTfLiteVariantObject);
}

TEST(TestTfLiteReallocWithObject,
     ConstructMutliParamVariantWithAlreadyAllocated) {
  TensorUniquePtr t = BuildTfLiteTensor();
  t->type = kTfLiteVariant;
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo2, int, float>(t.get(), 3, 1.0)),
      kTfLiteOk);
  void* before_address = t->data.data;
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo2, int, float>(t.get(), 5, 2.0)),
      kTfLiteOk);
  EXPECT_EQ(t->data.data, before_address);
  VariantFoo2* data = reinterpret_cast<VariantFoo2*>(t->data.data);
  EXPECT_EQ(data->GetFooInt(), 5);
  EXPECT_EQ(data->GetFloatData(), 2.0);
  EXPECT_EQ(t->type, kTfLiteVariant);
  EXPECT_EQ(t->allocation_type, kTfLiteVariantObject);
}

TEST(TestTfLiteReallocWithObject, NonVariantTypeError) {
  TensorUniquePtr t = BuildTfLiteTensor();
  t->type = kTfLiteInt32;
  ASSERT_EQ((TfLiteTensorVariantRealloc<VariantFoo>(t.get(), 3)), kTfLiteError);
}

TEST(TestVariantData, CopyVariantTensorCallsDerivedCopyCstor) {
  // Basic setup for tensors.
  TensorUniquePtr src_variant_tensor = BuildTfLiteTensor();
  TensorUniquePtr dst_variant_tensor = BuildTfLiteTensor();
  for (TfLiteTensor* tensor :
       {src_variant_tensor.get(), dst_variant_tensor.get()}) {
    tensor->dims = ConvertVectorToTfLiteIntArray({0});
    tensor->allocation_type = kTfLiteVariantObject;
    tensor->type = kTfLiteVariant;
  }
  // Initialize variant data object. `src_variant_tensor` takes ownership
  // of any arguments passed to `TfLiteTensorRealloc` should it succeed.
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo>(src_variant_tensor.get(), 1)),
      kTfLiteOk);
  auto* src_variant_data =
      reinterpret_cast<VariantFoo*>(src_variant_tensor->data.data);
  EXPECT_EQ(src_variant_data->GetFooInt(), 1);
  EXPECT_EQ(src_variant_data->GetFooCopied(), false);

  // Copy one variant tensor to another as usual.
  ASSERT_EQ(
      TfLiteTensorCopy(src_variant_tensor.get(), dst_variant_tensor.get()),
      kTfLiteOk);

  // The destination tensor's data.data member will point to the result
  // of calling the copy constructor of the source tensors underlying object.
  auto* dst_variant_data =
      reinterpret_cast<VariantFoo*>(dst_variant_tensor->data.data);
  EXPECT_EQ(dst_variant_data->GetFooInt(), 1);
  EXPECT_EQ(dst_variant_data->GetFooCopied(), true);
}

TEST(TestVariantData, CopyVariantTensorCallsDerivedCopyCstorWithAllocation) {
  // Basic setup for tensors.
  TensorUniquePtr src_variant_tensor = BuildTfLiteTensor();
  TensorUniquePtr dst_variant_tensor = BuildTfLiteTensor();
  for (TfLiteTensor* tensor :
       {src_variant_tensor.get(), dst_variant_tensor.get()}) {
    tensor->dims = ConvertVectorToTfLiteIntArray({0});
    tensor->allocation_type = kTfLiteVariantObject;
    tensor->type = kTfLiteVariant;
  }
  // Initialize variant data objects.
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo>(src_variant_tensor.get(), 1)),
      kTfLiteOk);
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo>(dst_variant_tensor.get(), 2)),
      kTfLiteOk);

  void* before_address = dst_variant_tensor->data.data;

  // Copy one variant tensor to another as usual.
  ASSERT_EQ(
      TfLiteTensorCopy(src_variant_tensor.get(), dst_variant_tensor.get()),
      kTfLiteOk);

  auto* dst_variant_data =
      reinterpret_cast<VariantFoo*>(dst_variant_tensor->data.data);
  EXPECT_EQ(dst_variant_data->GetFooInt(), 1);

  // If the destination tensor is already populated, the dstor will be called
  // and the buffer reused.
  EXPECT_EQ(dst_variant_tensor->data.data, before_address);
}

TEST(TestVariantData, CopyTensorToNonVariantObjectSetsAllocationType) {
  // Basic setup for tensors.
  TensorUniquePtr src_variant_tensor = BuildTfLiteTensor();
  TensorUniquePtr dst_variant_tensor = BuildTfLiteTensor();

  for (TfLiteTensor* tensor :
       {src_variant_tensor.get(), dst_variant_tensor.get()}) {
    tensor->dims = ConvertVectorToTfLiteIntArray({0});
    tensor->type = kTfLiteVariant;
  }
  src_variant_tensor->allocation_type = kTfLiteVariantObject;

  // Initialize variant data objects.
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo>(src_variant_tensor.get(), 1)),
      kTfLiteOk);
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo>(dst_variant_tensor.get(), 2)),
      kTfLiteOk);

  void* before_address = dst_variant_tensor->data.data;

  // Copy one variant tensor to another as usual.
  ASSERT_EQ(
      TfLiteTensorCopy(src_variant_tensor.get(), dst_variant_tensor.get()),
      kTfLiteOk);

  ASSERT_EQ(dst_variant_tensor->allocation_type, kTfLiteVariantObject);

  auto* dst_variant_data =
      reinterpret_cast<VariantFoo*>(dst_variant_tensor->data.data);
  EXPECT_EQ(dst_variant_data->GetFooInt(), 1);

  // If the destination tensor is already populated, the dstor will be called
  // and the buffer reused.
  EXPECT_EQ(dst_variant_tensor->data.data, before_address);
}

TfLiteTensor CreateDefaultTestTensor() {
  return {/*.type=*/kTfLiteInt32,
          /*.data.data=*/{nullptr},
          /*.dims=*/nullptr,
          /*.params=*/TfLiteQuantizationParams{/*scale=*/2, /*zero_point=*/3},
          /*.allocation_type=*/kTfLiteMemNone,
          /*.bytes=*/0,
          /*.allocation=*/nullptr,
          /*.name=*/"fake name",
          /*.delegate=*/nullptr,
          /*.buffer_handle=*/kTfLiteNullBufferHandle,
          /*.data_is_stale=*/false,
          /*.is_variable=*/true,
          /*.quantization=*/TfLiteQuantization{},
          /*.sparsity=*/nullptr,
          /*.dims_signature=*/nullptr};
}

TEST(TensorCloneTest, CloneATensorAttributes) {
  TfLiteTensor model = [&] {
    auto dims_data = BuildTfLiteArray<int>({1, 2, 3});
    auto dims_signature_data = BuildTfLiteArray<int>({11, 12, 13});
    TfLiteAffineQuantization* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            malloc(sizeof(TfLiteAffineQuantization)));
    affine_quantization->scale = BuildTfLiteArray<float>({7, 8, 9}).release();
    affine_quantization->zero_point = BuildTfLiteArray({4, 5, 6}).release();
    affine_quantization->quantized_dimension = 34;
    TfLiteQuantization quantization = {/*type=*/kTfLiteAffineQuantization,
                                       /*params=*/affine_quantization};

    const int kDimMetadataCount = 2;
    TfLiteDimensionMetadata* dim_metadata =
        reinterpret_cast<TfLiteDimensionMetadata*>(
            malloc(kDimMetadataCount * sizeof(TfLiteDimensionMetadata)));
    for (int i = 0; i < kDimMetadataCount; ++i) {
      (dim_metadata + i)->format = kTfLiteDimSparseCSR;
      (dim_metadata + i)->dense_size = 3;
      (dim_metadata + i)->array_segments =
          BuildTfLiteArray<int>({21 + i, 22 + i, 23 + i}).release();
      (dim_metadata + i)->array_indices =
          BuildTfLiteArray<int>({24 + i, 25 + i, 26 + i}).release();
    }
    TfLiteSparsity* sparsity =
        reinterpret_cast<TfLiteSparsity*>(malloc(sizeof(TfLiteSparsity)));
    sparsity->traversal_order = BuildTfLiteArray<int>({31, 32, 33}).release();
    sparsity->block_map = BuildTfLiteArray<int>({34, 35, 36}).release();
    sparsity->dim_metadata = dim_metadata;
    sparsity->dim_metadata_size = kDimMetadataCount;

    TfLiteTensor model = CreateDefaultTestTensor();
    model.dims = dims_data.release();
    model.dims_signature = dims_signature_data.release();
    model.quantization = quantization;
    model.sparsity = sparsity;
    return model;
  }();

  TfLiteTensor clone = TfLiteTensorClone(model);

  EXPECT_THAT(clone.type, Eq(model.type));
  // Note: `data` is not checked, it may be different depending on the tensor
  // allocation.
  EXPECT_THAT(clone.dims, TfLiteArrayIs(model.dims));
  EXPECT_THAT(clone.params.scale, Eq(model.params.scale));
  EXPECT_THAT(clone.params.zero_point, Eq(model.params.zero_point));
  EXPECT_THAT(clone.allocation_type, Eq(model.allocation_type));
  EXPECT_THAT(clone.bytes, Eq(model.bytes));
  // Note: `allocation` is not checked, it may be different depending on the
  // tensor allocation.
  EXPECT_THAT(clone.name, Eq(model.name));
  EXPECT_THAT(clone.delegate, Eq(model.delegate));
  EXPECT_THAT(clone.buffer_handle, Eq(model.buffer_handle));
  EXPECT_THAT(clone.data_is_stale, Eq(model.data_is_stale));
  EXPECT_THAT(clone.is_variable, Eq(model.is_variable));

  auto GetAffineQuantization = [](const TfLiteTensor& tensor) {
    return reinterpret_cast<TfLiteAffineQuantization*>(
        tensor.quantization.params);
  };
  EXPECT_THAT(clone.quantization.type, Eq(model.quantization.type));
  // Ensure that this is a deep clone and not just a pointer copy.
  ASSERT_THAT(clone.quantization.params,
              Not(AnyOf(nullptr, model.quantization.params)));
  EXPECT_THAT(GetAffineQuantization(clone)->scale,
              TfLiteArrayIs(GetAffineQuantization(model)->scale));
  EXPECT_THAT(GetAffineQuantization(clone)->zero_point,
              TfLiteArrayIs(GetAffineQuantization(model)->zero_point));
  EXPECT_THAT(GetAffineQuantization(clone)->quantized_dimension,
              Eq(GetAffineQuantization(model)->quantized_dimension));

  // Ensure that this is a deep clone and not just a pointer copy.
  ASSERT_THAT(clone.sparsity, Not(AnyOf(nullptr, model.sparsity)));
  EXPECT_THAT(clone.sparsity->traversal_order,
              TfLiteArrayIs(model.sparsity->traversal_order));
  EXPECT_THAT(clone.sparsity->block_map,
              TfLiteArrayIs(model.sparsity->block_map));
  ASSERT_THAT(clone.sparsity->dim_metadata,
              Not(AnyOf(nullptr, model.sparsity->dim_metadata)));
  ASSERT_THAT(clone.sparsity->dim_metadata_size,
              Eq(model.sparsity->dim_metadata_size));
  auto GetDimMetadata = [](TfLiteTensor& tensor, int idx) {
    return tensor.sparsity->dim_metadata[idx];
  };
  for (int i = 0; i < clone.sparsity->dim_metadata_size; ++i) {
    EXPECT_THAT(GetDimMetadata(clone, i).format,
                Eq(GetDimMetadata(model, i).format))
        << i;
    EXPECT_THAT(GetDimMetadata(clone, i).dense_size,
                Eq(GetDimMetadata(model, i).dense_size))
        << i;
    EXPECT_THAT(GetDimMetadata(clone, i).array_segments,
                TfLiteArrayIs(GetDimMetadata(model, i).array_segments))
        << i;
    EXPECT_THAT(GetDimMetadata(clone, i).array_indices,
                TfLiteArrayIs(GetDimMetadata(model, i).array_indices))
        << i;
  }

  EXPECT_THAT(clone.dims_signature, TfLiteArrayIs(model.dims_signature));

  TfLiteTensorFree(&clone);
  TfLiteTensorFree(&model);

  // model.sparsity;
}

}  // namespace tflite
