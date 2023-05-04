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
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/util.h"

namespace tflite {
using ::testing::ElementsAreArray;

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

TEST(TestTfLiteOpaqueDelegate, CreateAndDelete) {
  std::unique_ptr<TfLiteOpaqueDelegateBuilder> opaque_delegate_builder(
      new TfLiteOpaqueDelegateBuilder{});

  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(opaque_delegate_builder.get());

  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST(TestTfLiteOpaqueDelegate, CallTfLiteOpaqueDelegateCreateWithNull) {
  EXPECT_EQ(nullptr, TfLiteOpaqueDelegateCreate(nullptr));
}

TEST(TestTfLiteOpaqueDelegate, CallTfLiteOpaqueDelegateDeleteWithNull) {
  TfLiteOpaqueDelegateDelete(nullptr);
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

TEST(TestTfLiteOpaqueDelegate, GetData_WellFormedOpaqueDelegate) {
  int delegate_data = 42;
  TfLiteOpaqueDelegateBuilder builder{};
  builder.data = &delegate_data;

  TfLiteOpaqueDelegate* opaque_delegate = TfLiteOpaqueDelegateCreate(&builder);

  EXPECT_EQ(&delegate_data, TfLiteOpaqueDelegateGetData(opaque_delegate));

  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST(TestTfLiteOpaqueDelegate,
     GetData_NotConstructedWithTfLiteOpaqueDelegateCreate) {
  // Given a non-opaque delegate, that was created with 'TfLiteDelegateCreate'
  // and has its 'data_' field set manually.
  int delegate_data = 42;
  TfLiteDelegate non_opaque_delegate = TfLiteDelegateCreate();
  non_opaque_delegate.data_ = &delegate_data;
  // The following cast is safe only because this code is part of the
  // TF Lite test suite.  Apps using TF Lite should not rely on
  // 'TfLiteOpaqueDelegate' and 'TfLiteDelegate' being equivalent.
  auto* opaque_delegate =
      reinterpret_cast<TfLiteOpaqueDelegate*>(&non_opaque_delegate);

  // The accessor returns '&delegate_data', because the
  // 'opaque_delegate_builder' field inside the delegate was not set so it falls
  // back to returning the data_ field of TfLiteDelegate.
  EXPECT_EQ(&delegate_data, TfLiteOpaqueDelegateGetData(opaque_delegate));
}

TEST(TestTfLiteOpaqueDelegate, GetData_NoDataSetViaOpaqueDelegateBuilder) {
  TfLiteOpaqueDelegateBuilder builder{};
  TfLiteOpaqueDelegate* opaque_delegate = TfLiteOpaqueDelegateCreate(&builder);
  // The accessor returns 'nullptr', because the 'data' field inside the opaque
  // delegate builder was not set.
  EXPECT_EQ(nullptr, TfLiteOpaqueDelegateGetData(opaque_delegate));
  TfLiteOpaqueDelegateDelete(opaque_delegate);
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

TEST(TestTfLiteReallocWithObject, SimpleConstruction) {
  TfLiteTensor t = {};
  t.type = kTfLiteVariant;
  ASSERT_EQ((TfLiteTensorVariantRealloc<VariantFoo, int>(&t, 3)), kTfLiteOk);
  ASSERT_EQ(reinterpret_cast<VariantFoo*>(t.data.data)->GetFooInt(), 3);
  ASSERT_EQ(t.type, kTfLiteVariant);
  ASSERT_EQ(t.allocation_type, kTfLiteVariantObject);
  TfLiteTensorFree(&t);
}

TEST(TestTfLiteReallocWithObject, ConstructWithAlreadyAllocated) {
  TfLiteTensor t = {};
  t.type = kTfLiteVariant;
  ASSERT_EQ((TfLiteTensorVariantRealloc<VariantFoo, int>(&t, 3)), kTfLiteOk);
  void* before_address = t.data.data;
  ASSERT_EQ((TfLiteTensorVariantRealloc<VariantFoo, int>(&t, 5)), kTfLiteOk);
  EXPECT_EQ(t.data.data, before_address);
  EXPECT_EQ(reinterpret_cast<VariantFoo*>(t.data.data)->GetFooInt(), 5);
  EXPECT_EQ(t.type, kTfLiteVariant);
  EXPECT_EQ(t.allocation_type, kTfLiteVariantObject);
  TfLiteTensorFree(&t);
}

TEST(TestTfLiteReallocWithObject, NonVariantTypeError) {
  TfLiteTensor t = {};
  t.type = kTfLiteInt32;
  ASSERT_EQ((TfLiteTensorVariantRealloc<VariantFoo, int>(&t, 3)), kTfLiteError);
}

TEST(TestVariantData, CopyVariantTensorCallsDerivedCopyCstor) {
  // Basic setup for tensors.
  TfLiteTensor src_variant_tensor = {};
  TfLiteTensor dst_variant_tensor = {};
  for (auto* tensor : {&src_variant_tensor, &dst_variant_tensor}) {
    tensor->dims = ConvertVectorToTfLiteIntArray({0});
    tensor->allocation_type = kTfLiteVariantObject;
    tensor->type = kTfLiteVariant;
  }
  // Initialize variant data object. `src_variant_tensor` takes ownership
  // of any arguments passed to `TfLiteTensorRealloc` should it succeed.
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo, int>(&src_variant_tensor, 1)),
      kTfLiteOk);
  auto* src_variant_data =
      reinterpret_cast<VariantFoo*>(src_variant_tensor.data.data);
  EXPECT_EQ(src_variant_data->GetFooInt(), 1);
  EXPECT_EQ(src_variant_data->GetFooCopied(), false);

  // Copy one variant tensor to another as usual.
  ASSERT_EQ(TfLiteTensorCopy(&src_variant_tensor, &dst_variant_tensor),
            kTfLiteOk);

  // The destination tensor's data.data member will point to the result
  // of calling the copy constructor of the source tensors underlying object.
  auto* dst_variant_data =
      reinterpret_cast<VariantFoo*>(dst_variant_tensor.data.data);
  EXPECT_EQ(dst_variant_data->GetFooInt(), 1);
  EXPECT_EQ(dst_variant_data->GetFooCopied(), true);

  TfLiteTensorFree(&src_variant_tensor);
  TfLiteTensorFree(&dst_variant_tensor);
}

TEST(TestVariantData, CopyVariantTensorCallsDerivedCopyCstorWithAllocation) {
  // Basic setup for tensors.
  TfLiteTensor src_variant_tensor = {};
  TfLiteTensor dst_variant_tensor = {};
  for (auto* tensor : {&src_variant_tensor, &dst_variant_tensor}) {
    tensor->dims = ConvertVectorToTfLiteIntArray({0});
    tensor->allocation_type = kTfLiteVariantObject;
    tensor->type = kTfLiteVariant;
  }
  // Initialize variant data objects.
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo, int>(&src_variant_tensor, 1)),
      kTfLiteOk);
  ASSERT_EQ(
      (TfLiteTensorVariantRealloc<VariantFoo, int>(&dst_variant_tensor, 2)),
      kTfLiteOk);

  void* before_address = dst_variant_tensor.data.data;

  // Copy one variant tensor to another as usual.
  ASSERT_EQ(TfLiteTensorCopy(&src_variant_tensor, &dst_variant_tensor),
            kTfLiteOk);

  auto* dst_variant_data =
      reinterpret_cast<VariantFoo*>(dst_variant_tensor.data.data);
  EXPECT_EQ(dst_variant_data->GetFooInt(), 1);

  // If the destination tensor is already populated, the dstor will be called
  // and the buffer reused.
  EXPECT_EQ(dst_variant_tensor.data.data, before_address);

  TfLiteTensorFree(&src_variant_tensor);
  TfLiteTensorFree(&dst_variant_tensor);
}

}  // namespace tflite
