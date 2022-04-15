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

#include "tensorflow/lite/c/common.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"

namespace tflite {

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
  ASSERT_FALSE(TfLiteIntArrayEqual(a, b));
  ASSERT_TRUE(TfLiteIntArrayEqual(b, c));
  ASSERT_TRUE(TfLiteIntArrayEqual(b, b));
  ASSERT_FALSE(TfLiteIntArrayEqual(c, d));
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

TEST(Types, TestTypeNames) {
  auto type_name = [](TfLiteType t) {
    return std::string(TfLiteTypeGetName(t));
  };
  EXPECT_EQ(type_name(kTfLiteNoType), "NOTYPE");
  EXPECT_EQ(type_name(kTfLiteFloat64), "FLOAT64");
  EXPECT_EQ(type_name(kTfLiteFloat32), "FLOAT32");
  EXPECT_EQ(type_name(kTfLiteFloat16), "FLOAT16");
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

}  // namespace tflite
