/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/tensor.h"

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/float8.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class TensorTestHelper {
 public:
  // This is an operation that can be done by VariableOp.
  static void set_shape(Tensor* t, const TensorShape& s) { t->set_shape(s); }
};

// To make TestCopies do the right thing.
bool operator==(const ResourceHandle& a, const ResourceHandle& b) {
  return a.device() == b.device() && a.container() == b.container() &&
         a.name() == b.name() && a.hash_code() == b.hash_code() &&
         a.maybe_type_name() == b.maybe_type_name();
}

bool operator==(const Variant& a, const Variant& b) {
  if (a.is_empty()) {
    return b.is_empty();
  }

  if (a.TypeId() != b.TypeId()) return false;
  if (a.TypeName() != b.TypeName()) return false;

  VariantTensorData a_data, b_data;
  a.Encode(&a_data);
  b.Encode(&b_data);

  string a_metadata;
  string b_metadata;
  a_data.get_metadata(&a_metadata);
  b_data.get_metadata(&b_metadata);
  if (a_metadata != b_metadata) return false;

  if (a_data.tensors_size() != b_data.tensors_size()) return false;

  for (int i = 0; i < a_data.tensors_size(); ++i) {
    TensorProto a_proto, b_proto;
    a_data.tensors(i).AsProtoTensorContent(&a_proto);
    b_data.tensors(i).AsProtoTensorContent(&b_proto);
    string a_str, b_str;
    a_proto.SerializeToString(&a_str);
    b_proto.SerializeToString(&b_str);
    if (a_str != b_str) return false;
  }

  return true;
}

namespace {

TEST(TensorTest, Default) {
  Tensor t;
  EXPECT_EQ(t.dtype(), DT_FLOAT);
  EXPECT_EQ(t.dims(), 1);
  EXPECT_EQ(t.NumElements(), 0);
}

TEST(TensorTest, DataType_Traits) {
  EXPECT_TRUE(std::is_trivial<float>::value);
  EXPECT_TRUE(std::is_trivial<double>::value);
  EXPECT_TRUE(std::is_trivial<int32>::value);
  EXPECT_TRUE(std::is_trivial<uint8>::value);
  EXPECT_TRUE(std::is_trivial<uint16>::value);
  EXPECT_TRUE(std::is_trivial<int16>::value);
  EXPECT_TRUE(std::is_trivial<int8>::value);
  EXPECT_TRUE(std::is_trivial<int64_t>::value);
  EXPECT_TRUE(std::is_trivial<bool>::value);
  EXPECT_FALSE(std::is_trivial<tstring>::value);
  EXPECT_FALSE(std::is_trivial<string>::value);

  EXPECT_EQ(sizeof(bool), 1);

  // Unfortunately. std::complex::complex() initializes (0, 0).
  EXPECT_FALSE(std::is_trivial<complex64>::value);
  EXPECT_FALSE(std::is_trivial<complex128>::value);
  EXPECT_TRUE(std::is_trivial<float[2]>::value);
  EXPECT_TRUE(std::is_trivial<double[2]>::value);
  struct MyComplex64 {
    float re, im;
  };
  EXPECT_TRUE(std::is_trivial<MyComplex64>::value);
  struct MyComplex128 {
    double re, im;
  };
  EXPECT_TRUE(std::is_trivial<MyComplex128>::value);
}

template <typename T>
void ExpectEqual(const Tensor& x, const Tensor& y) {
  test::ExpectEqual(x, y);
}
// test::ExpectEqual does not support ResourceHandle or Variant.
template <>
void ExpectEqual<ResourceHandle>(const Tensor& x, const Tensor& y) {
  EXPECT_EQ(x, y);
}
template <>
void ExpectEqual<Variant>(const Tensor& x, const Tensor& y) {
  EXPECT_EQ(x, y);
}

template <typename T>
void TestCopies(const Tensor& t) {
  {
    LOG(INFO) << "CopyFrom()";
    Tensor t2(t.dtype());
    EXPECT_TRUE(t2.CopyFrom(t, t.shape()));
    ExpectEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "operator=()";
    Tensor t2(t.dtype());
    t2 = t;
    ExpectEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "deep copy";
    Tensor t2(t.dtype(), t.shape());
    t2.flat<T>() = t.flat<T>();
    ExpectEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "AsProtoField()";
    TensorProto proto;
    t.AsProtoField(&proto);
    Tensor t2(t.dtype());
    EXPECT_TRUE(t2.FromProto(proto));
    ExpectEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "AsProtoTensorContent()";
    TensorProto proto;
    t.AsProtoTensorContent(&proto);
    Tensor t2(t.dtype());
    EXPECT_TRUE(t2.FromProto(proto));
    ExpectEqual<T>(t, t2);
    // Make another copy via tensor_content field.
    *proto.mutable_tensor_content() = proto.tensor_content();
    Tensor t3(t.dtype());
    EXPECT_TRUE(t3.FromProto(proto));
    ExpectEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "AsTensor";
    gtl::ArraySlice<T> values(t.flat<T>().data(), t.NumElements());
    Tensor t2 = test::AsTensor(values, t.shape());
    ExpectEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "Move constructor";
    Tensor t2 = t;
    Tensor t3 = std::move(t2);
    ExpectEqual<T>(t, t3);
    EXPECT_TRUE(t3.IsInitialized());
    EXPECT_FALSE(t2.IsInitialized());  // NOLINT(bugprone-use-after-move)
  }
  {
    LOG(INFO) << "Move assignment";
    Tensor t2 = t;
    Tensor t3;
    t3 = std::move(t2);
    ExpectEqual<T>(t, t3);
    EXPECT_TRUE(t3.IsInitialized());
    EXPECT_FALSE(t2.IsInitialized());  // NOLINT(bugprone-use-after-move)
  }
  {
    LOG(INFO) << "Move self-assignment";
    Tensor t2 = t;
    Tensor* t3 = &t2;
    *t3 = std::move(t2);
    ExpectEqual<Variant>(t, *t3);
    EXPECT_TRUE(t3->IsInitialized());
  }
}

TEST(Tensor_Half, Simple) {
  Tensor t(DT_HALF, TensorShape({5, 7}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({5, 7})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<Eigen::half>()(a, b) = static_cast<Eigen::half>(a * b);
    }
  }
  TestCopies<Eigen::half>(t);
}

TEST(Tensor_Bfloat16, Simple) {
  Tensor t(DT_BFLOAT16, TensorShape({5, 7}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({5, 7})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<bfloat16>()(a, b) = static_cast<bfloat16>(a * b);
    }
  }
  TestCopies<bfloat16>(t);
}

TEST(Tensor_Float8_E5m2, Simple) {
  Tensor t(DT_FLOAT8_E5M2, TensorShape({5, 7}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({5, 7})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<float8_e5m2>()(a, b) = static_cast<float8_e5m2>(a * b);
    }
  }
  TestCopies<float8_e5m2>(t);
}

TEST(Tensor_Float8_E4m3fn, Simple) {
  Tensor t(DT_FLOAT8_E4M3FN, TensorShape({5, 7}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({5, 7})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<float8_e4m3fn>()(a, b) = static_cast<float8_e4m3fn>(a * b);
    }
  }
  TestCopies<float8_e4m3fn>(t);
}

TEST(Tensor_Float, Simple) {
  Tensor t(DT_FLOAT, TensorShape({10, 20}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 20})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<float>()(a, b) = static_cast<float>(a * b);
    }
  }
  TestCopies<float>(t);
}

TEST(Tensor_Double, Simple) {
  Tensor t(DT_DOUBLE, TensorShape({10, 20}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 20})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<double>()(a, b) = static_cast<double>(a * b);
    }
  }
  TestCopies<double>(t);
}

TEST(Tensor_int8, Simple) {
  Tensor t(DT_INT8, TensorShape({10, 20}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 20})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<int8>()(a, b) = static_cast<int8>(a * b);
    }
  }
  TestCopies<int8>(t);
}

TEST(Tensor_int16, Simple) {
  Tensor t(DT_INT16, TensorShape({10, 20}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 20})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<int16>()(a, b) = static_cast<int16>(a * b);
    }
  }
  TestCopies<int16>(t);
}

TEST(Tensor_int32, Simple) {
  Tensor t(DT_INT32, TensorShape({10, 20}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 20})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<int>()(a, b) = static_cast<int>(a * b);
    }
  }
  TestCopies<int>(t);
}

TEST(Tensor_int64, Simple) {
  Tensor t(DT_INT64, TensorShape({10, 20}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 20})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<int64_t>()(a, b) = static_cast<int64_t>(a * b);
    }
  }
  TestCopies<int64_t>(t);
}

TEST(Tensor_uint64, Simple) {
  Tensor t(DT_UINT64, TensorShape({10, 20}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 20})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<uint64_t>()(a, b) = static_cast<uint64_t>(a * b);
    }
  }
  TestCopies<uint64_t>(t);
}

TEST(Tensor_ResourceHandle, Simple) {
  Tensor t(DT_RESOURCE, TensorShape({}));
  ResourceHandle tmp;
  tmp.set_name("a");
  t.flat<ResourceHandle>()(0) = tmp;
  TestCopies<ResourceHandle>(t);
}

TEST(Tensor_Variant, Simple) {
  Tensor t(DT_VARIANT, TensorShape({}));
  Tensor value(DT_FLOAT, TensorShape({}));
  value.flat<float>()(0) = 42.0f;
  t.flat<Variant>()(0) = value;
  // All the tests in TestCopies except the ones that serialize and deserialize
  // the tensor. The consumer of a serialized Variant Tensor should know what
  // type is stored in the Tensor, so not testing the generic
  // serialize/deserialize case here.
  {
    LOG(INFO) << "CopyFrom()";
    Tensor t2(t.dtype());
    EXPECT_TRUE(t2.CopyFrom(t, t.shape()));
    ExpectEqual<Variant>(t, t2);
  }
  {
    LOG(INFO) << "operator=()";
    Tensor t2(t.dtype());
    t2 = t;
    ExpectEqual<Variant>(t, t2);
  }
  {
    LOG(INFO) << "deep copy";
    Tensor t2(t.dtype(), t.shape());
    t2.flat<Variant>() = t.flat<Variant>();
    ExpectEqual<Variant>(t, t2);
  }
  {
    LOG(INFO) << "AsTensor";
    gtl::ArraySlice<Variant> values(t.flat<Variant>().data(), t.NumElements());
    Tensor t2 = test::AsTensor(values, t.shape());
    ExpectEqual<Variant>(t, t2);
  }
  {
    LOG(INFO) << "Move constructor";
    Tensor t2 = t;
    Tensor t3 = std::move(t2);
    ExpectEqual<Variant>(t, t3);
    EXPECT_TRUE(t3.IsInitialized());
    EXPECT_FALSE(t2.IsInitialized());  // NOLINT(bugprone-use-after-move)
  }
  {
    LOG(INFO) << "Move assignment";
    Tensor t2 = t;
    Tensor t3;
    t3 = std::move(t2);
    ExpectEqual<Variant>(t, t3);
    EXPECT_TRUE(t3.IsInitialized());
    EXPECT_FALSE(t2.IsInitialized());  // NOLINT(bugprone-use-after-move)
  }
  {
    LOG(INFO) << "Move self-assignment";
    Tensor t2 = t;
    Tensor* t3 = &t2;
    *t3 = std::move(t2);
    ExpectEqual<Variant>(t, *t3);
    EXPECT_TRUE(t3->IsInitialized());
  }
  {
    LOG(INFO) << "AsProtoTensorContent()";
    TensorProto proto;
    t.AsProtoTensorContent(&proto);
    Tensor t2(t.dtype());
    EXPECT_TRUE(t2.FromProto(proto));
    ExpectEqual<Variant>(t, t2);
  }
}

TEST(Tensor_Variant, Marshal) {
  Tensor t(DT_VARIANT, TensorShape({}));

  Tensor internal(DT_FLOAT, TensorShape({}));
  internal.flat<float>()(0) = 42.0f;
  t.flat<Variant>()(0) = internal;

  LOG(INFO) << "AsProtoField()";
  TensorProto proto;
  t.AsProtoField(&proto);

  // This performs a decode operation.
  Tensor t2(t.dtype());
  EXPECT_TRUE(t2.FromProto(proto));

  Tensor* out = t2.flat<Variant>()(0).get<Tensor>();
  EXPECT_NE(out, nullptr);
  EXPECT_FLOAT_EQ(out->scalar<float>()(), 42.0f);
}

TEST(Tensor_Bool, Simple) {
  Tensor t(DT_BOOL, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<bool>()(a, b) = true;
    }
  }
  TestCopies<bool>(t);
}

TEST(Tensor_UInt8, Simple) {
  Tensor t(DT_UINT8, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<uint8>()(a, b) = static_cast<uint8>(a * b);
    }
  }
  TestCopies<uint8>(t);
}

TEST(Tensor_UInt16, Simple) {
  Tensor t(DT_UINT16, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<uint16>()(a, b) = static_cast<uint16>(a * b);
    }
  }
  TestCopies<uint16>(t);
}

TEST(Tensor_UInt32, Simple) {
  Tensor t(DT_UINT32, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<uint32>()(a, b) = static_cast<uint32>(a * b);
    }
  }
  TestCopies<uint32>(t);
}

TEST(Tensor_QInt8, Simple) {
  Tensor t(DT_QINT8, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<qint8>()(a, b) = qint8(a * b);
    }
  }
  TestCopies<qint8>(t);
}

TEST(Tensor_QUInt8, Simple) {
  Tensor t(DT_QUINT8, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<Eigen::QUInt8>()(a, b) = Eigen::QUInt8(a * b);
    }
  }
  TestCopies<Eigen::QUInt8>(t);
}

TEST(Tensor_QInt16, Simple) {
  Tensor t(DT_QINT16, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<qint16>()(a, b) = qint16(a * b);
    }
  }
  TestCopies<qint16>(t);
}

TEST(Tensor_QUInt16, Simple) {
  Tensor t(DT_QUINT16, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<Eigen::QUInt16>()(a, b) = Eigen::QUInt16(a * b);
    }
  }
  TestCopies<Eigen::QUInt16>(t);
}

TEST(Tensor_QInt32, Simple) {
  Tensor t(DT_QINT32, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<qint32>()(a, b) = qint32(static_cast<int32>(a * b));
    }
  }
  TestCopies<qint32>(t);
}

class TensorReshapeTest : public ::testing::Test {
 protected:
  Tensor t;
  Tensor zero_t;

  TensorReshapeTest()
      : t(DT_FLOAT, TensorShape({2, 3, 4, 5})),
        zero_t(DT_FLOAT, TensorShape({3, 0, 2, 0, 5})) {}

  void SetUp() override {
    EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 3, 4, 5})));
    EXPECT_TRUE(zero_t.shape().IsSameSize(TensorShape({3, 0, 2, 0, 5})));

    auto tensor = t.tensor<float, 4>();
    EXPECT_EQ(2, tensor.dimension(0));
    EXPECT_EQ(3, tensor.dimension(1));
    EXPECT_EQ(4, tensor.dimension(2));
    EXPECT_EQ(5, tensor.dimension(3));

    // Set first and last elements.
    tensor(0, 0, 0, 0) = 0.01f;
    tensor(1, 2, 3, 4) = 0.02f;
  }

  template <typename T>
  using ReshapeFunc = T (Tensor::*)(gtl::ArraySlice<int64_t>);
  template <typename T>
  using ConstReshapeFunc = T (Tensor::*)(gtl::ArraySlice<int64_t>) const;

  template <typename T, ReshapeFunc<T> Func>
  void TestReshape(std::initializer_list<int64_t> sizes) {
    T shaped = (t.*Func)(sizes);
    TestReshapeImpl(shaped, sizes);
  }

  template <typename T, ConstReshapeFunc<T> Func>
  void TestReshape(std::initializer_list<int64_t> sizes) {
    T shaped = (static_cast<const Tensor&>(t).*Func)(sizes);
    TestReshapeImpl(shaped, sizes);
  }

  template <typename T>
  void TestReshapeImpl(T shaped, std::initializer_list<int64_t> sizes) {
    auto iter = sizes.begin();
    for (int i = 0; i < shaped.rank(); ++i, ++iter) {
      EXPECT_EQ(*iter, shaped.dimension(i));
    }

    using Index = typename T::Index;
    using Scalar = typename T::Scalar;
    constexpr int N = T::NumIndices;

    // To handle the cast when `shaped` is bit casted into a different type.
    const float expected_first = 0.01f;
    Eigen::DSizes<Index, N> coord;
    EXPECT_EQ(shaped(coord), *reinterpret_cast<const Scalar*>(&expected_first));

    for (int i = 0; i < N; ++i) {
      coord[i] = shaped.dimension(i) - 1;
    }
    const float expected_last = 0.02f;
    constexpr int kNumScalarPerFloat =
        sizeof(float) / sizeof(Scalar);  // Assuming even divide.
    EXPECT_EQ(shaped(coord), reinterpret_cast<const Scalar*>(
                                 &expected_last)[kNumScalarPerFloat - 1]);
  }
};

TEST_F(TensorReshapeTest, Reshape) {
  LOG(INFO) << "shaped";

#define TEST_RESHAPE(...)                                                  \
  {                                                                        \
    int _tmp[] = {__VA_ARGS__};                                            \
    constexpr int N = (sizeof(_tmp) / sizeof(int));                        \
    TestReshape<TTypes<float, N>::Tensor, &Tensor::shaped<float, N>>(      \
        {__VA_ARGS__});                                                    \
    TestReshape<TTypes<float, N>::ConstTensor, &Tensor::shaped<float, N>>( \
        {__VA_ARGS__});                                                    \
    TestReshape<TTypes<float, N>::UnalignedTensor,                         \
                &Tensor::unaligned_shaped<float, N>>({__VA_ARGS__});       \
    TestReshape<TTypes<float, N>::UnalignedConstTensor,                    \
                &Tensor::unaligned_shaped<float, N>>({__VA_ARGS__});       \
    TestReshape<TTypes<float, N>::Tensor,                                  \
                &Tensor::bit_casted_shaped<float, N>>({__VA_ARGS__});      \
    TestReshape<TTypes<float, N>::ConstTensor,                             \
                &Tensor::bit_casted_shaped<float, N>>({__VA_ARGS__});      \
    TestReshape<TTypes<int32, N>::Tensor,                                  \
                &Tensor::bit_casted_shaped<int32, N>>({__VA_ARGS__});      \
    TestReshape<TTypes<int32, N>::ConstTensor,                             \
                &Tensor::bit_casted_shaped<int32, N>>({__VA_ARGS__});      \
  }

  TEST_RESHAPE(120);
  TEST_RESHAPE(6, 20);
  TEST_RESHAPE(6, 4, 5);
  TEST_RESHAPE(2, 3, 4, 5);
#undef TEST_RESHAPE
}

TEST_F(TensorReshapeTest, BitcastReshapeDifferentSize) {
#define TEST_BITCAST8_RESHAPE(...)                                    \
  {                                                                   \
    int _tmp[] = {__VA_ARGS__};                                       \
    constexpr int N = (sizeof(_tmp) / sizeof(int));                   \
    TestReshape<TTypes<uint8, N>::Tensor,                             \
                &Tensor::bit_casted_shaped<uint8, N>>({__VA_ARGS__}); \
  }

  TEST_BITCAST8_RESHAPE(480);
  TEST_BITCAST8_RESHAPE(24, 20);
  TEST_BITCAST8_RESHAPE(6, 16, 5);
  TEST_BITCAST8_RESHAPE(2, 3, 4, 20);
#undef TEST_BITCAST8_RESHAPE
#define TEST_BITCAST16_RESHAPE(...)                                   \
  {                                                                   \
    int _tmp[] = {__VA_ARGS__};                                       \
    constexpr int N = (sizeof(_tmp) / sizeof(int));                   \
    TestReshape<TTypes<int16, N>::Tensor,                             \
                &Tensor::bit_casted_shaped<int16, N>>({__VA_ARGS__}); \
  }

  TEST_BITCAST16_RESHAPE(240);
  TEST_BITCAST16_RESHAPE(6, 40);
  TEST_BITCAST16_RESHAPE(12, 4, 5);
  TEST_BITCAST16_RESHAPE(2, 3, 8, 5);
  TEST_BITCAST16_RESHAPE(2, 3, 4, 1, 10);
#undef TEST_BITCAST16_RESHAPE
}

TEST_F(TensorReshapeTest, ReshapeError) {
  EXPECT_DEATH((t.shaped<float, 0>({})), "1 vs. 120");
  EXPECT_DEATH((t.shaped<float, 1>({119})), "119 vs. 120");
  EXPECT_DEATH((t.shaped<float, 4>({2, 3, 4, 6})), "144 vs. 120");

  EXPECT_DEATH((t.unaligned_shaped<float, 0>({})), "1 vs. 120");
  EXPECT_DEATH((t.unaligned_shaped<float, 1>({119})), "119 vs. 120");
  EXPECT_DEATH((t.unaligned_shaped<float, 4>({2, 3, 4, 6})), "144 vs. 120");

  EXPECT_DEATH((t.bit_casted_shaped<float, 0>({})), "4 vs. 480");
  EXPECT_DEATH((t.bit_casted_shaped<float, 1>({119})), "476 vs. 480");
  EXPECT_DEATH((t.bit_casted_shaped<float, 4>({2, 3, 4, 6})), "576 vs. 480");

  Tensor string_tensor{DT_STRING, {10}};
  // Note that the error message compare # of elements, not # of bytes.
  EXPECT_DEATH((string_tensor.bit_casted_shaped<tstring, 1>({9})), "9 vs. 10");
}

TEST_F(TensorReshapeTest, Flat) {
  LOG(INFO) << "flat";
  {
    auto flat = t.flat<float>();
    EXPECT_EQ(flat(0), 0.01f);
    EXPECT_EQ(120, flat.dimension(0));
    EXPECT_EQ(flat(0), 0.01f);
    EXPECT_EQ(flat(119), 0.02f);
  }
}

TEST_F(TensorReshapeTest, FlatInnerDims) {
  LOG(INFO) << "flat_inner_dims";
  {
    auto flat_inner_dims = t.flat_inner_dims<float>();
    EXPECT_EQ(24, flat_inner_dims.dimension(0));
    EXPECT_EQ(5, flat_inner_dims.dimension(1));
    EXPECT_EQ(flat_inner_dims(0, 0), 0.01f);
    EXPECT_EQ(flat_inner_dims(23, 4), 0.02f);
  }
  {
    auto flat_inner_dims = t.flat_inner_dims<float, 3>();
    EXPECT_EQ(6, flat_inner_dims.dimension(0));
    EXPECT_EQ(4, flat_inner_dims.dimension(1));
    EXPECT_EQ(5, flat_inner_dims.dimension(2));
    EXPECT_EQ(flat_inner_dims(0, 0, 0), 0.01f);
    EXPECT_EQ(flat_inner_dims(5, 3, 4), 0.02f);
  }
  {
    auto flat_inner_dims = t.flat_inner_dims<float, 5>();
    EXPECT_EQ(1, flat_inner_dims.dimension(0));
    EXPECT_EQ(2, flat_inner_dims.dimension(1));
    EXPECT_EQ(3, flat_inner_dims.dimension(2));
    EXPECT_EQ(4, flat_inner_dims.dimension(3));
    EXPECT_EQ(5, flat_inner_dims.dimension(4));
    EXPECT_EQ(flat_inner_dims(0, 0, 0, 0, 0), 0.01f);
    EXPECT_EQ(flat_inner_dims(0, 1, 2, 3, 4), 0.02f);
  }
  {
    auto flat_inner_dims = zero_t.flat_inner_dims<float>();
    EXPECT_EQ(0, flat_inner_dims.dimension(0));
    EXPECT_EQ(5, flat_inner_dims.dimension(1));
  }
  {
    auto flat_inner_dims = zero_t.flat_inner_dims<float, 3>();
    EXPECT_EQ(0, flat_inner_dims.dimension(0));
    EXPECT_EQ(0, flat_inner_dims.dimension(1));
    EXPECT_EQ(5, flat_inner_dims.dimension(2));
  }
  {
    auto flat_inner_dims = zero_t.flat_inner_dims<float, 5>();
    EXPECT_EQ(3, flat_inner_dims.dimension(0));
    EXPECT_EQ(0, flat_inner_dims.dimension(1));
    EXPECT_EQ(2, flat_inner_dims.dimension(2));
    EXPECT_EQ(0, flat_inner_dims.dimension(3));
    EXPECT_EQ(5, flat_inner_dims.dimension(4));
  }
}

TEST_F(TensorReshapeTest, FlatOuterDims) {
  LOG(INFO) << "flat_outer_dims";
  {
    auto flat_outer_dims = t.flat_outer_dims<float>();
    EXPECT_EQ(2, flat_outer_dims.dimension(0));
    EXPECT_EQ(60, flat_outer_dims.dimension(1));
    EXPECT_EQ(flat_outer_dims(0, 0), 0.01f);
    EXPECT_EQ(flat_outer_dims(1, 59), 0.02f);
  }
  {
    auto flat_outer_dims = t.flat_outer_dims<float, 3>();
    EXPECT_EQ(2, flat_outer_dims.dimension(0));
    EXPECT_EQ(3, flat_outer_dims.dimension(1));
    EXPECT_EQ(20, flat_outer_dims.dimension(2));
    EXPECT_EQ(flat_outer_dims(0, 0, 0), 0.01f);
    EXPECT_EQ(flat_outer_dims(1, 2, 19), 0.02f);
  }
  {
    auto flat_outer_dims = t.flat_outer_dims<float, 5>();
    EXPECT_EQ(2, flat_outer_dims.dimension(0));
    EXPECT_EQ(3, flat_outer_dims.dimension(1));
    EXPECT_EQ(4, flat_outer_dims.dimension(2));
    EXPECT_EQ(5, flat_outer_dims.dimension(3));
    EXPECT_EQ(1, flat_outer_dims.dimension(4));
    EXPECT_EQ(flat_outer_dims(0, 0, 0, 0, 0), 0.01f);
    EXPECT_EQ(flat_outer_dims(1, 2, 3, 4, 0), 0.02f);
  }
  {
    auto flat_outer_dims = zero_t.flat_outer_dims<float>();
    EXPECT_EQ(3, flat_outer_dims.dimension(0));
    EXPECT_EQ(0, flat_outer_dims.dimension(1));
  }
  {
    auto flat_outer_dims = zero_t.flat_outer_dims<float, 3>();
    EXPECT_EQ(3, flat_outer_dims.dimension(0));
    EXPECT_EQ(0, flat_outer_dims.dimension(1));
    EXPECT_EQ(0, flat_outer_dims.dimension(2));
  }
  {
    auto flat_outer_dims = zero_t.flat_outer_dims<float, 5>();
    EXPECT_EQ(3, flat_outer_dims.dimension(0));
    EXPECT_EQ(0, flat_outer_dims.dimension(1));
    EXPECT_EQ(2, flat_outer_dims.dimension(2));
    EXPECT_EQ(0, flat_outer_dims.dimension(3));
    EXPECT_EQ(5, flat_outer_dims.dimension(4));
  }
}

TEST_F(TensorReshapeTest, FlatInnerOuterDims) {
  LOG(INFO) << "flat_inner_outer_dims";
  {
    auto flat_inner_outer_dims = t.flat_inner_outer_dims<float, 4>(0);
    EXPECT_EQ(2, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(3, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(4, flat_inner_outer_dims.dimension(2));
    EXPECT_EQ(5, flat_inner_outer_dims.dimension(3));
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 0, 0), 0.01f);
    EXPECT_EQ(flat_inner_outer_dims(1, 2, 3, 4), 0.02f);
  }
  {
    auto flat_inner_outer_dims = t.flat_inner_outer_dims<float, 6>(-2);
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(2, flat_inner_outer_dims.dimension(2));
    EXPECT_EQ(3, flat_inner_outer_dims.dimension(3));
    EXPECT_EQ(4, flat_inner_outer_dims.dimension(4));
    EXPECT_EQ(5, flat_inner_outer_dims.dimension(5));
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 0, 0, 0, 0), 0.01f);
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 1, 2, 3, 4), 0.02f);
  }
  {
    auto flat_inner_outer_dims = t.flat_inner_outer_dims<float, 6>(0);
    EXPECT_EQ(2, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(3, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(4, flat_inner_outer_dims.dimension(2));
    EXPECT_EQ(5, flat_inner_outer_dims.dimension(3));
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(4));
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(5));
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 0, 0, 0, 0), 0.01f);
    EXPECT_EQ(flat_inner_outer_dims(1, 2, 3, 4, 0, 0), 0.02f);
  }
  {
    auto flat_inner_outer_dims = t.flat_inner_outer_dims<float, 8>(-2);
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(2, flat_inner_outer_dims.dimension(2));
    EXPECT_EQ(3, flat_inner_outer_dims.dimension(3));
    EXPECT_EQ(4, flat_inner_outer_dims.dimension(4));
    EXPECT_EQ(5, flat_inner_outer_dims.dimension(5));
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(6));
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(7));
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 0, 0, 0, 0, 0, 0), 0.01f);
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 1, 2, 3, 4, 0, 0), 0.02f);
  }
  {
    auto flat_inner_outer_dims = t.flat_inner_outer_dims<float, 3>(1);
    EXPECT_EQ(6, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(4, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(5, flat_inner_outer_dims.dimension(2));
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 0), 0.01f);
    EXPECT_EQ(flat_inner_outer_dims(5, 3, 4), 0.02f);
  }
  {
    auto flat_inner_outer_dims = t.flat_inner_outer_dims<float, 5>(1);
    EXPECT_EQ(6, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(4, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(5, flat_inner_outer_dims.dimension(2));
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(3));
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(4));
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 0, 0, 0), 0.01f);
    EXPECT_EQ(flat_inner_outer_dims(5, 3, 4, 0, 0), 0.02f);
  }
  {
    auto flat_inner_outer_dims = t.flat_inner_outer_dims<float, 3>(0);
    EXPECT_EQ(2, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(3, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(20, flat_inner_outer_dims.dimension(2));
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 0), 0.01f);
    EXPECT_EQ(flat_inner_outer_dims(1, 2, 19), 0.02f);
  }
  {
    auto flat_inner_outer_dims = t.flat_inner_outer_dims<float, 5>(-2);
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(1, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(2, flat_inner_outer_dims.dimension(2));
    EXPECT_EQ(3, flat_inner_outer_dims.dimension(3));
    EXPECT_EQ(20, flat_inner_outer_dims.dimension(4));
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 0, 0, 0), 0.01f);
    EXPECT_EQ(flat_inner_outer_dims(0, 0, 1, 2, 19), 0.02f);
  }
  {
    auto flat_inner_outer_dims = t.flat_inner_outer_dims<float, 2>(1);
    EXPECT_EQ(6, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(20, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(flat_inner_outer_dims(0, 0), 0.01f);
    EXPECT_EQ(flat_inner_outer_dims(5, 19), 0.02f);
  }
  {
    auto flat_inner_outer_dims = zero_t.flat_inner_outer_dims<float, 2>(0);
    EXPECT_EQ(3, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(0, flat_inner_outer_dims.dimension(1));
  }
  {
    auto flat_inner_outer_dims = zero_t.flat_inner_outer_dims<float, 3>(0);
    EXPECT_EQ(3, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(0, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(0, flat_inner_outer_dims.dimension(2));
  }
  {
    auto flat_inner_outer_dims = zero_t.flat_inner_outer_dims<float, 5>(0);
    EXPECT_EQ(3, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(0, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(2, flat_inner_outer_dims.dimension(2));
    EXPECT_EQ(0, flat_inner_outer_dims.dimension(3));
    EXPECT_EQ(5, flat_inner_outer_dims.dimension(4));
  }
  {
    auto flat_inner_outer_dims = zero_t.flat_inner_outer_dims<float, 2>(3);
    EXPECT_EQ(0, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(5, flat_inner_outer_dims.dimension(1));
  }
  {
    auto flat_inner_outer_dims = zero_t.flat_inner_outer_dims<float, 3>(2);
    EXPECT_EQ(0, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(0, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(5, flat_inner_outer_dims.dimension(2));
  }
  {
    auto flat_inner_outer_dims = zero_t.flat_inner_outer_dims<float, 3>(1);
    EXPECT_EQ(0, flat_inner_outer_dims.dimension(0));
    EXPECT_EQ(2, flat_inner_outer_dims.dimension(1));
    EXPECT_EQ(0, flat_inner_outer_dims.dimension(2));
  }
}

TEST(ReinterpretLastDimension, Reinterpret_NCHW_VECT_C_as_NCHW) {
  LOG(INFO) << "reinterpret_last_dimension";
  {
    Tensor t_nchw_vect_c(DT_QINT8, TensorShape({2, 3, 5, 7, 4}));
    auto nchw_vect_c = t_nchw_vect_c.tensor<qint8, 5>();
    Tensor t_expected_nchw(DT_INT32, TensorShape({2, 3, 5, 7}));
    auto expected_nchw = t_expected_nchw.tensor<int32, 4>();
    int8_t val = 0;
    for (int n = 0; n < t_nchw_vect_c.shape().dim_size(0); ++n) {
      for (int c = 0; c < t_nchw_vect_c.shape().dim_size(1); ++c) {
        for (int h = 0; h < t_nchw_vect_c.shape().dim_size(2); ++h, ++val) {
          int8 packet[4];
          for (int w = 0; w < t_nchw_vect_c.shape().dim_size(3); ++w) {
            packet[0] = nchw_vect_c(n, c, h, w, 0) = ++val;
            packet[1] = nchw_vect_c(n, c, h, w, 1) = ++val;
            packet[2] = nchw_vect_c(n, c, h, w, 2) = ++val;
            packet[3] = nchw_vect_c(n, c, h, w, 3) = ++val;
            expected_nchw(n, c, h, w) = *reinterpret_cast<int32*>(&packet[0]);
          }
        }
      }
    }
    auto actual_nchw = t_nchw_vect_c.reinterpret_last_dimension<int32, 4>();
    const auto& const_t_nchw_vect_c = t_nchw_vect_c;
    auto const_actual_nchw =
        const_t_nchw_vect_c.reinterpret_last_dimension<int32, 4>();
    for (int n = 0; n < t_nchw_vect_c.shape().dim_size(0); ++n) {
      for (int c = 0; c < t_nchw_vect_c.shape().dim_size(1); ++c) {
        for (int h = 0; h < t_nchw_vect_c.shape().dim_size(2); ++h) {
          for (int w = 0; w < t_nchw_vect_c.shape().dim_size(3); ++w) {
            EXPECT_EQ(expected_nchw(n, c, h, w), actual_nchw(n, c, h, w));
            EXPECT_EQ(expected_nchw(n, c, h, w), const_actual_nchw(n, c, h, w));
          }
        }
      }
    }
  }
}

TEST(Tensor_Scalar, Basics) {
  {
    Tensor t(DT_BOOL, TensorShape({}));
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.scalar<bool>();
    EXPECT_EQ(1, Tt.size());
    EXPECT_EQ(0, Tt.rank());
    t.scalar<bool>()() = true;
    EXPECT_TRUE(Tt());
  }
  {
    Tensor t(DT_FLOAT, TensorShape({}));
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.scalar<float>();
    EXPECT_EQ(1, Tt.size());
    EXPECT_EQ(0, Tt.rank());
    t.scalar<float>()() = 123.45f;
    EXPECT_FLOAT_EQ(123.45f, Tt());
  }
  {
    Tensor t(DT_FLOAT, TensorShape({1}));
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.vec<float>();
    EXPECT_EQ(1, Tt.size());
    t.vec<float>()(0) = 123.45f;
    EXPECT_FLOAT_EQ(123.45f, Tt(0));
  }
  {
    Tensor t(DT_FLOAT, TensorShape({1, 1, 1}));
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.scalar<float>();
    EXPECT_EQ(1, Tt.size());
    EXPECT_EQ(0, Tt.rank());
    t.flat<float>()(0) = 123.45f;
    EXPECT_FLOAT_EQ(123.45f, Tt());
  }
  {
    Tensor t(DT_STRING, TensorShape({}));
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.scalar<tstring>();
    EXPECT_EQ(1, Tt.size());
    EXPECT_EQ(0, Tt.rank());
    t.scalar<tstring>()() = "foo";
    EXPECT_EQ("foo", Tt());
  }
  {
    Tensor t(DT_STRING, TensorShape({1}));
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.vec<tstring>();
    EXPECT_EQ(1, Tt.size());
    t.flat<tstring>()(0) = "foo";
    EXPECT_EQ("foo", Tt(0));
  }
  {
    Tensor t(DT_STRING, TensorShape({1, 1, 1}));
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.scalar<tstring>();
    EXPECT_EQ(1, Tt.size());
    EXPECT_EQ(0, Tt.rank());
    t.flat<tstring>()(0) = "bar";
    EXPECT_EQ("bar", Tt());
  }
  {
    Tensor t(DT_FLOAT, TensorShape({0, 1}));
    EXPECT_EQ(0, t.NumElements());
    auto Tt = t.flat<float>();
    EXPECT_EQ(0, Tt.size());
    auto Tm = t.matrix<float>();
    EXPECT_EQ(0, Tm.size());
    EXPECT_EQ(0, Tm.dimensions()[0]);
    EXPECT_EQ(1, Tm.dimensions()[1]);
  }
}

TEST(Tensor_HostScalar, Basics) {
  {
    Tensor t(true);
    EXPECT_EQ(DT_BOOL, t.dtype());
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.scalar<bool>();
    EXPECT_EQ(1, Tt.size());
    EXPECT_EQ(0, Tt.rank());
    EXPECT_TRUE(Tt());
    Tt() = false;
    EXPECT_FALSE(Tt());
  }
  {
    Tensor t(123.45f);
    EXPECT_EQ(DT_FLOAT, t.dtype());
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.scalar<float>();
    EXPECT_EQ(1, Tt.size());
    EXPECT_EQ(0, Tt.rank());
    EXPECT_FLOAT_EQ(123.45f, Tt());
    Tt() = 42.0f;
    EXPECT_FLOAT_EQ(42.0f, Tt());
  }
  {
    // NOTE(mrry): Use long enough strings so that the contents are dynamically
    // allocated, and the absence of a call to the string destructor would
    // cause a memory leak.
    Tensor t("fooooooooooooooooooooooooooooooooooooo");
    EXPECT_EQ(DT_STRING, t.dtype());
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.scalar<tstring>();
    EXPECT_EQ(1, Tt.size());
    EXPECT_EQ(0, Tt.rank());
    EXPECT_EQ("fooooooooooooooooooooooooooooooooooooo", Tt());
    Tt() = "baaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaar";
    EXPECT_EQ("baaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaar", Tt());
  }
}

TEST(Tensor_Float, Reshape_And_Slice_Assignment) {
  // A test to experiment with a way to assign to a subset of a tensor
  Tensor t(DT_FLOAT, TensorShape({10, 4, 3, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 4, 3, 2})));

  // Get the N dimensional tensor (N==4 here)
  auto e_t = t.tensor<float, 4>();
  // Reshape to view it as a two-dimensional tensor
  auto e_2d = t.shaped<float, 2>({10, 4 * 3 * 2});
  for (int i = 0; i < 10; i++) {
    // Assign a 1 x 4*3*2 matrix (really vector) to a slice of size
    // 1 x 4*3*2 in e_t.
    Eigen::Tensor<float, 2, Eigen::RowMajor> m(1, 4 * 3 * 2);
    m.setConstant(i * 2.0);

    Eigen::DSizes<Eigen::DenseIndex, 2> indices(i, 0);
    Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, 4 * 3 * 2);
    e_2d.slice(indices, sizes) = m;
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 2; l++) {
          EXPECT_EQ(e_t(i, j, k, l), i * 2.0f);
          LOG(INFO) << i << "," << j << "," << k << "," << l
                    << " &e_t(i, j, k, l): " << &e_t(i, j, k, l) << " = "
                    << e_t(i, j, k, l);
        }
      }
    }
  }
}

TEST(Tensor_Float, Bitcast_From) {
  Tensor t(DT_FLOAT, TensorShape({2, 4}));
  Tensor inp(DT_DOUBLE, TensorShape({2, 2}));
  auto s = t.BitcastFrom(inp, inp.dtype(), inp.shape());
  ASSERT_TRUE(s.ok());
}

TEST(Tensor_String, Simple) {
  Tensor t = test::AsTensor<tstring>(
      {"hello", "world", "machine", "learning", "new", "york"},
      TensorShape({3, 2}));
  auto s = t.shape();
  ASSERT_EQ(s.dims(), 2);
  ASSERT_EQ(s.dim_size(0), 3);
  ASSERT_EQ(s.dim_size(1), 2);
  auto m = t.matrix<tstring>();
  EXPECT_EQ(t.TotalBytes(), 3 * 2 * sizeof(tstring) + 5 + 5 + 7 + 8 + 3 + 4);

  EXPECT_EQ(m(0, 0), "hello");
  EXPECT_EQ(m(0, 1), "world");
  EXPECT_EQ(m(1, 0), "machine");
  EXPECT_EQ(m(1, 1), "learning");
  EXPECT_EQ(m(2, 0), "new");
  EXPECT_EQ(m(2, 1), "york");

  TestCopies<tstring>(t);
}

TEST(Tensor_Float, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<float>({0, 1, 2, 3, 4, 5}, {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<float>() = t1.flat<float>() * 2.0f;
  Tensor t3 = test::AsTensor<float>({0, 2, 4, 6, 8, 10}, t1.shape());
  ExpectEqual<float>(t2, t3);
}

TEST(Tensor_Float, SimpleBuildTensor) {
  Tensor t;
  auto s = Tensor::BuildTensor(DT_FLOAT, {10, 20}, &t);
  ASSERT_TRUE(s.ok());
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 20})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<float>()(a, b) = static_cast<float>(a * b);
    }
  }
  TestCopies<float>(t);
  ASSERT_EQ(t.AllocatedBytes(), 800);  // 10 * 20 * 4
}

TEST(Tensor_Float, SimpleWithAllocator) {
  EnableCPUAllocatorFullStats();
  auto* a = cpu_allocator();
  Tensor t(a, DT_FLOAT, TensorShape({10, 20}), {});
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 20})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<float>()(a, b) = static_cast<float>(a * b);
    }
  }
  TestCopies<float>(t);

  TensorDescription p;
  t.FillDescription(&p);
  ASSERT_EQ(p.dtype(), DT_FLOAT);
}

TEST(Tensor_Int32, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<int32>({0, 1, 2, 3, 4, 5}, {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<int32>() = t1.flat<int32>() * 2;
  Tensor t3 = test::AsTensor<int32>({0, 2, 4, 6, 8, 10}, t1.shape());
  ExpectEqual<int32>(t2, t3);
}

TEST(Tensor_UInt16, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<uint16>({0, 1, 2, 3, 4, 5}, {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<uint16>() = t1.flat<uint16>() * uint16(2);
  Tensor t3 = test::AsTensor<uint16>({0, 2, 4, 6, 8, 10}, t1.shape());
  ExpectEqual<uint16>(t2, t3);
}

TEST(Tensor_QInt8, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<qint8>({0, 1, 2, 3, 4, 5}, {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<qint8>() = t1.flat<qint8>() + qint8(-2);
  Tensor t3 = test::AsTensor<qint8>({-2, -1, 0, 1, 2, 3}, {2, 3});
  ExpectEqual<qint8>(t2, t3);
}

TEST(Tensor_QUInt8, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<quint8>({0, 1, 2, 3, 4, 5}, {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<quint8>() = t1.flat<quint8>() + quint8(2);
  Tensor t3 = test::AsTensor<quint8>({2, 3, 4, 5, 6, 7}, {2, 3});
  ExpectEqual<quint8>(t2, t3);
}

TEST(Tensor_Int64, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<int64_t>(
      {0LL << 48, 1LL << 48, 2LL << 48, 3LL << 48, 4LL << 48, 5LL << 48},
      {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<int64_t>() = t1.flat<int64_t>() * static_cast<int64_t>(2);
  Tensor t3 = test::AsTensor<int64_t>(
      {0LL << 48, 2LL << 48, 4LL << 48, 6LL << 48, 8LL << 48, 10LL << 48},
      {2, 3});
  ExpectEqual<int64_t>(t2, t3);
}

TEST(Tensor_String, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<tstring>({"0", "1", "2", "3", "4", "5"}, {2, 3});
  Tensor t2(DT_STRING, {2, 3});
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      t2.matrix<tstring>()(i, j) = strings::StrCat(i * 3 + j);
    }
  }

  // Test with helper.
  ExpectEqual<tstring>(t1, t2);
}

TEST(Tensor_Bool, SimpleWithHelper) {
  Tensor t1 =
      test::AsTensor<bool>({false, true, false, true, false, true}, {2, 3});

  Tensor t2(DT_BOOL, {2, 3});
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      t2.matrix<bool>()(i, j) = (((i + j) % 2) != 0);
    }
  }

  // Test with helper.
  ExpectEqual<bool>(t1, t2);
}

TEST(Tensor_Bool, DecodeValidate) {
  TensorProto proto;
  proto.set_dtype(DT_BOOL);
  proto.mutable_tensor_shape()->add_dim()->set_size(5);
  proto.set_tensor_content(std::string{0, 1, 2, 3, 4});

  Tensor t;
  EXPECT_FALSE(t.FromProto(proto));
}

TEST(Tensor_Complex, Simple64) {
  Tensor t(DT_COMPLEX64, {4, 5, 3, 7});
  t.flat<complex64>().setRandom();
  TestCopies<complex64>(t);
}

TEST(Tensor_Complex, Simple128) {
  Tensor t(DT_COMPLEX128, {4, 5, 3, 7});
  t.flat<complex128>().setRandom();
  TestCopies<complex128>(t);
}

TEST(Tensor_Complex, SimpleWithHelper64) {
  {
    Tensor t1 = test::AsTensor<complex64>({0,
                                           {1, 1},
                                           complex64(2),
                                           complex64(3, 3),
                                           complex64(0, 4),
                                           complex64(2, 5)},
                                          {2, 3});
    Tensor t2(t1.dtype(), t1.shape());
    t2.flat<complex64>() = t1.flat<complex64>() * complex64(0, 2);
    Tensor t3 = test::AsTensor<complex64>(
        {0, {-2, 2}, {0, 4}, {-6, 6}, {-8, 0}, {-10, 4}},
        // shape
        {2, 3});
    ExpectEqual<complex64>(t2, t3);
  }

  // Does some numeric operations for complex64 numbers.
  {
    const float PI = std::acos(-1);
    const complex64 rotate_45 = std::polar(1.0f, PI / 4);

    // x contains all the 8-th root of unity.
    Tensor x(DT_COMPLEX64, TensorShape({8}));
    for (int i = 0; i < 8; ++i) {
      x.vec<complex64>()(i) = MathUtil::IPow(rotate_45, i);
    }

    // Shift the roots by 45 degree.
    Tensor y(DT_COMPLEX64, TensorShape({8}));
    y.vec<complex64>() = x.vec<complex64>() * rotate_45;
    Tensor y_expected(DT_COMPLEX64, TensorShape({8}));
    for (int i = 0; i < 8; ++i) {
      y_expected.vec<complex64>()(i) = MathUtil::IPow(rotate_45, i + 1);
    }
    test::ExpectTensorNear<complex64>(y, y_expected, 1e-5);

    // Raise roots to the power of 8.
    Tensor z(DT_COMPLEX64, TensorShape({8}));
    z.vec<complex64>() = x.vec<complex64>().pow(8);
    Tensor z_expected(DT_COMPLEX64, TensorShape({8}));
    for (int i = 0; i < 8; ++i) {
      z_expected.vec<complex64>()(i) = 1;
    }
    test::ExpectTensorNear<complex64>(z, z_expected, 1e-5);
  }
}

TEST(Tensor_Complex, SimpleWithHelper128) {
  {
    Tensor t1 = test::AsTensor<complex128>({0,
                                            {1, 1},
                                            complex128(2),
                                            complex128(3, 3),
                                            complex128(0, 4),
                                            complex128(2, 5)},
                                           {2, 3});
    Tensor t2(t1.dtype(), t1.shape());
    t2.flat<complex128>() = t1.flat<complex128>() * complex128(0, 2);
    Tensor t3 = test::AsTensor<complex128>(
        {0, {-2, 2}, {0, 4}, {-6, 6}, {-8, 0}, {-10, 4}},
        // shape
        {2, 3});
    ExpectEqual<complex128>(t2, t3);
  }

  // Does some numeric operations for complex128 numbers.
  {
    const double PI = std::acos(-1);
    const complex128 rotate_45 = std::polar(1.0, PI / 4);

    // x contains all the 8-th root of unity.
    Tensor x(DT_COMPLEX128, TensorShape({8}));
    for (int i = 0; i < 8; ++i) {
      x.vec<complex128>()(i) = MathUtil::IPow(rotate_45, i);
    }

    // Shift the roots by 45 degree.
    Tensor y(DT_COMPLEX128, TensorShape({8}));
    y.vec<complex128>() = x.vec<complex128>() * rotate_45;
    Tensor y_expected(DT_COMPLEX128, TensorShape({8}));
    for (int i = 0; i < 8; ++i) {
      y_expected.vec<complex128>()(i) = MathUtil::IPow(rotate_45, i + 1);
    }
    test::ExpectTensorNear<complex128>(y, y_expected, 1e-5);

    // Raise roots to the power of 8.
    Tensor z(DT_COMPLEX128, TensorShape({8}));
    z.vec<complex128>() = x.vec<complex128>().pow(8);
    Tensor z_expected(DT_COMPLEX128, TensorShape({8}));
    for (int i = 0; i < 8; ++i) {
      z_expected.vec<complex128>()(i) = 1;
    }
    test::ExpectTensorNear<complex128>(z, z_expected, 1e-5);
  }
}

// An allocator that always returns nullptr, for testing
// failures to allocate.
class DummyCPUAllocator : public Allocator {
 public:
  DummyCPUAllocator() = default;
  string Name() override { return "cpu"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return nullptr;
  }
  void DeallocateRaw(void* ptr) override {}
};

TEST(Tensor, SharesBufferWith) {
  Tensor a_empty;
  Tensor b_empty;
  Tensor a(DT_FLOAT, TensorShape({1}));
  Tensor b(DT_FLOAT, TensorShape({1}));
  Tensor copy(a);
  EXPECT_FALSE(a_empty.SharesBufferWith(a_empty));
  EXPECT_FALSE(a_empty.SharesBufferWith(b_empty));
  EXPECT_FALSE(a_empty.SharesBufferWith(a));
  EXPECT_FALSE(a_empty.SharesBufferWith(copy));
  EXPECT_TRUE(a.SharesBufferWith(a));
  EXPECT_FALSE(a.SharesBufferWith(b));
  EXPECT_TRUE(a.SharesBufferWith(copy));
}

TEST(TensorFromProto, CompressedTensorProto) {
  int size = 100;
  TensorShape shape({size});
  Allocator* allocator = cpu_allocator();
  Tensor a(allocator, DT_FLOAT, shape);
  std::fill_n(a.flat<float>().data(), size, 42.0f);
  TensorProto p;
  a.AsProtoField(&p);
  tensor::CompressTensorProtoInPlace(&p);
  Tensor b;
  ASSERT_TRUE(b.FromProto(p));
}

TEST(Tensor, FailureToAllocate) {
  TensorShape shape({1});
  DummyCPUAllocator allocator;
  {
    Tensor a(&allocator, DT_FLOAT, shape);
    ASSERT_FALSE(a.IsInitialized());
  }

  // Float
  {
    Tensor t(DT_FLOAT, TensorShape({1}));
    t.vec<float>()(0) = 1.0;
    TensorProto proto;
    t.AsProtoField(&proto);

    // FromProto should fail nicely.
    Tensor a(&allocator, DT_FLOAT, TensorShape({1}));
    ASSERT_FALSE(a.FromProto(&allocator, proto));
  }

  // String
  {
    Tensor t(DT_STRING, TensorShape({1}));
    t.vec<tstring>()(0) = "foo";
    TensorProto proto;
    t.AsProtoField(&proto);

    // FromProto should fail nicely.
    Tensor a(&allocator, DT_STRING, TensorShape({1}));
    ASSERT_FALSE(a.FromProto(&allocator, proto));
  }

  // Half
  {
    Tensor t(DT_HALF, TensorShape({1}));
    t.vec<Eigen::half>()(0) = Eigen::half(1.0);
    TensorProto proto;
    t.AsProtoField(&proto);

    // FromProto should fail nicely.
    Tensor a(&allocator, DT_HALF, TensorShape({1}));
    ASSERT_FALSE(a.FromProto(&allocator, proto));
  }
}

// On the alignment.
//
// As of 2018/5, tensorflow::Tensor allocates its buffer with 64-byte
// alignment. Tensor::tensor/flat/vec/matrix methods requires the
// buffer satisfies Eigen::Aligned (e.g., 16-bytes aligned usually,
// 32-bytes for AVX, and 64-bytes for AVX512). Tensor::Slice requires
// the caller to ensure its result is aligned if the caller intends
// to use those methods. In this test case, we simply make sure each
// slice is 64-byte aligned: sizeof(float) * 4 * 36 = 576.  576 % 64 = 0.
TEST(Tensor, Slice_Basic) {
  Tensor saved;
  {  // General
    Tensor x(DT_FLOAT, TensorShape({10, 4, 36}));
    // Fills in known values.
    for (int i = 0; i < 10; ++i) {
      x.Slice(i, i + 1).flat<float>().setConstant(i * 1.f);
    }
    // A simple slice along dim0.
    Tensor y = x.Slice(4, 8);
    EXPECT_TRUE(y.shape().IsSameSize(TensorShape({4, 4, 36})));
    auto tx = x.tensor<float, 3>();
    auto ty = y.tensor<float, 3>();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 36; ++k) {
          EXPECT_EQ(ty(i, j, k), 4.0 + i);
          EXPECT_EQ(&tx(4 + i, j, k), &ty(i, j, k));
        }
      }
    }

    ASSERT_EQ(y.AllocatedBytes(), 2304);  // 4 * 4 * 36 * 4

    TensorDescription p;
    y.FillDescription(&p);
    ASSERT_EQ(p.dtype(), DT_FLOAT);

    // A simple slice equivalent to identity.
    TestCopies<float>(y);
    y = x.Slice(0, 10);
    ExpectEqual<float>(x, y);
    EXPECT_EQ(x.flat<float>().data(), y.flat<float>().data());

    // A slice of a slice.
    auto z = x.Slice(4, 8).Slice(2, 3);
    auto tz = z.tensor<float, 3>();
    EXPECT_EQ(1, z.dim_size(0));
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 36; ++k) {
        EXPECT_EQ(tz(0, j, k), 6.0);
      }
    }

    // x and y will be out of scope. But 'saved' should be alive.
    saved = z;
  }
  {
    EXPECT_EQ(1, saved.dim_size(0));
    auto tsaved = saved.tensor<float, 3>();
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 36; ++k) {
        EXPECT_EQ(tsaved(0, j, k), 6.0);
      }
    }
  }
  {  // Empty
    Tensor x(DT_FLOAT, TensorShape({10, 0, 36}));
    x.flat<float>().setRandom();
    Tensor y = x.Slice(4, 8);
    EXPECT_TRUE(y.shape().IsSameSize(TensorShape({4, 0, 36})));
  }

  {
    // Test unaligned access via a Slice.
    Tensor x(DT_FLOAT, TensorShape({30}));
    x.flat<float>().setConstant(0.0);

    // Take an unaligned slice.
    Tensor y = x.Slice(1, 13);
#if EIGEN_MAX_ALIGN_BYTES > 0
    EXPECT_FALSE(y.IsAligned());
#endif
    y.unaligned_flat<float>().setConstant(1.0);
    for (int64_t i = 0; i < y.NumElements(); ++i) {
      EXPECT_EQ(1.0, y.unaligned_flat<float>()(i));
    }
  }
}

TEST(Tensor, SubSlice_Basic) {
  {  // General
    Tensor x(DT_FLOAT, TensorShape({10, 4, 36}));
    // Fills in known values.
    for (int i = 0; i < 10; ++i) {
      x.SubSlice(i).flat<float>().setConstant(i * 1.f);
    }
    // A simple sub-slice along dim0.
    Tensor y = x.SubSlice(5);
    EXPECT_TRUE(y.shape().IsSameSize(TensorShape({4, 36})));
    auto tx = x.tensor<float, 3>();
    auto ty = y.tensor<float, 2>();
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 36; ++k) {
        EXPECT_EQ(ty(j, k), 5.0);
        EXPECT_EQ(&tx(5, j, k), &ty(j, k));
      }
    }
    Tensor z = y.SubSlice(3).SubSlice(31);
    auto tz = z.unaligned_flat<float>();
    EXPECT_EQ(*tz.data(), 5.0);
  }
  {
    // Test unaligned access via a SubSlice.
    Tensor x(DT_FLOAT, TensorShape({30, 5}));
    x.flat<float>().setConstant(0.0);

    // Take an unaligned subslice.
    Tensor y = x.SubSlice(1);
#if EIGEN_MAX_ALIGN_BYTES > 0
    EXPECT_FALSE(y.IsAligned());
#endif
    y.unaligned_flat<float>().setConstant(1.0);
    for (int64_t i = 0; i < y.NumElements(); ++i) {
      EXPECT_EQ(1.0, y.unaligned_flat<float>()(i));
    }
  }
}

template <typename T>
Tensor MkTensor(DataType dt, const TensorShape& shape,
                std::vector<T> init_values) {
  Tensor x(dt, shape);
  const int limit = x.NumElements();
  int vi = 0;
  for (int i = 0; i < limit; ++i) {
    x.flat<T>()(i) = init_values[vi++];
    if (vi >= init_values.size()) vi = 0;
  }
  return x;
}

template <typename T>
Tensor MkTensorNoInitList(DataType dt, const TensorShape& shape,
                          const std::vector<int>& init_values) {
  Tensor x(dt, shape);
  const int limit = x.NumElements();
  int vi = 0;
  for (int i = 0; i < limit; ++i) {
    x.flat<T>()(i) = static_cast<T>(init_values[vi++]);
    if (vi >= init_values.size()) vi = 0;
  }
  return x;
}

TEST(SummarizeValue, Uninitialized) {
  Tensor x(DT_INT32);
  TensorTestHelper::set_shape(&x, TensorShape({4, 4}));
  EXPECT_EQ(
      strings::StrCat("uninitialized Tensor of 16 elements of type ", DT_INT32),
      x.SummarizeValue(16));
}

struct SummarizeValueTestCase {
  std::string test_name;
  Tensor x_1d;
  Tensor x_2d;
  Tensor x_4d;
  Tensor x_empty;
};

using SimpleNum = ::testing::TestWithParam<SummarizeValueTestCase>;

TEST_P(SimpleNum, MultiLength) {
  const SummarizeValueTestCase& test_case = GetParam();

  EXPECT_EQ("1 2 3 4 0", test_case.x_1d.SummarizeValue(16));
  EXPECT_EQ("[1 2][3 4]", test_case.x_2d.SummarizeValue(16));
  EXPECT_EQ("[[[1]][[2]]][[[3]][[4]]]", test_case.x_4d.SummarizeValue(16));
  EXPECT_EQ("[[[1]][[2]]][[[3]]]...", test_case.x_4d.SummarizeValue(3));
  EXPECT_EQ("", test_case.x_empty.SummarizeValue(16));
}

INSTANTIATE_TEST_SUITE_P(
    SummarizeValue, SimpleNum,
    ::testing::ValuesIn<SummarizeValueTestCase>({
        {"DT_QUINT8",
         MkTensor<Eigen::QUInt8>(DT_QUINT8, TensorShape({5}), {1, 2, 3, 4, 0}),
         MkTensor<Eigen::QUInt8>(DT_QUINT8, TensorShape({2, 2}),
                                 {1, 2, 3, 4, 0}),
         MkTensor<Eigen::QUInt8>(DT_QUINT8, TensorShape({2, 2, 1, 1}),
                                 {1, 2, 3, 4, 0}),
         MkTensor<Eigen::QUInt8>(DT_QUINT8, TensorShape({0}), {})},
        {"DT_QUINT16",
         MkTensor<Eigen::QUInt16>(DT_QUINT16, TensorShape({5}),
                                  {1, 2, 3, 4, 0}),
         MkTensor<Eigen::QUInt16>(DT_QUINT16, TensorShape({2, 2}),
                                  {1, 2, 3, 4, 0}),
         MkTensor<Eigen::QUInt16>(DT_QUINT16, TensorShape({2, 2, 1, 1}),
                                  {1, 2, 3, 4, 0}),
         MkTensor<Eigen::QUInt16>(DT_QUINT16, TensorShape({0}), {})},
        {"DT_UINT8",
         MkTensor<uint8>(DT_UINT8, TensorShape({5}), {1, 2, 3, 4, 0}),
         MkTensor<uint8>(DT_UINT8, TensorShape({2, 2}), {1, 2, 3, 4, 0}),
         MkTensor<uint8>(DT_UINT8, TensorShape({2, 2, 1, 1}), {1, 2, 3, 4, 0}),
         MkTensor<uint8>(DT_UINT8, TensorShape({0}), {})},
        {"DT_UINT16",
         MkTensor<uint16>(DT_UINT16, TensorShape({5}), {1, 2, 3, 4, 0}),
         MkTensor<uint16>(DT_UINT16, TensorShape({2, 2}), {1, 2, 3, 4, 0}),
         MkTensor<uint16>(DT_UINT16, TensorShape({2, 2, 1, 1}),
                          {1, 2, 3, 4, 0}),
         MkTensor<uint16>(DT_UINT16, TensorShape({0}), {})},
        {"DT_UINT32",
         MkTensor<uint32>(DT_UINT32, TensorShape({5}), {1, 2, 3, 4, 0}),
         MkTensor<uint32>(DT_UINT32, TensorShape({2, 2}), {1, 2, 3, 4, 0}),
         MkTensor<uint32>(DT_UINT32, TensorShape({2, 2, 1, 1}),
                          {1, 2, 3, 4, 0}),
         MkTensor<uint32>(DT_UINT32, TensorShape({0}), {})},
        {"DT_UINT64",
         MkTensor<uint64>(DT_UINT64, TensorShape({5}), {1, 2, 3, 4, 0}),
         MkTensor<uint64>(DT_UINT64, TensorShape({2, 2}), {1, 2, 3, 4, 0}),
         MkTensor<uint64>(DT_UINT64, TensorShape({2, 2, 1, 1}),
                          {1, 2, 3, 4, 0}),
         MkTensor<uint64>(DT_UINT64, TensorShape({0}), {})},
        {"DT_INT8", MkTensor<int8>(DT_INT8, TensorShape({5}), {1, 2, 3, 4, 0}),
         MkTensor<int8>(DT_INT8, TensorShape({2, 2}), {1, 2, 3, 4, 0}),
         MkTensor<int8>(DT_INT8, TensorShape({2, 2, 1, 1}), {1, 2, 3, 4, 0}),
         MkTensor<int8>(DT_INT8, TensorShape({0}), {})},
        {"DT_INT16",
         MkTensor<int16>(DT_INT16, TensorShape({5}), {1, 2, 3, 4, 0}),
         MkTensor<int16>(DT_INT16, TensorShape({2, 2}), {1, 2, 3, 4, 0}),
         MkTensor<int16>(DT_INT16, TensorShape({2, 2, 1, 1}), {1, 2, 3, 4, 0}),
         MkTensor<int16>(DT_INT16, TensorShape({0}), {})},
        {"DT_INT32", MkTensor<int>(DT_INT32, TensorShape({5}), {1, 2, 3, 4, 0}),
         MkTensor<int>(DT_INT32, TensorShape({2, 2}), {1, 2, 3, 4, 0}),
         MkTensor<int>(DT_INT32, TensorShape({2, 2, 1, 1}), {1, 2, 3, 4, 0}),
         MkTensor<int>(DT_INT32, TensorShape({0}), {})},
        {"DT_INT64",
         MkTensor<int64_t>(DT_INT64, TensorShape({5}), {1, 2, 3, 4, 0}),
         MkTensor<int64_t>(DT_INT64, TensorShape({2, 2}), {1, 2, 3, 4, 0}),
         MkTensor<int64_t>(DT_INT64, TensorShape({2, 2, 1, 1}),
                           {1, 2, 3, 4, 0}),
         MkTensor<int64_t>(DT_INT64, TensorShape({0}), {})},
        {"DT_FLOAT8_E4M3FN",
         MkTensorNoInitList<float8_e4m3fn>(DT_FLOAT8_E4M3FN, TensorShape({5}),
                                           {1, 2, 3, 4, 0}),
         MkTensorNoInitList<float8_e4m3fn>(
             DT_FLOAT8_E4M3FN, TensorShape({2, 2}), {1, 2, 3, 4, 0}),
         MkTensorNoInitList<float8_e4m3fn>(
             DT_FLOAT8_E4M3FN, TensorShape({2, 2, 1, 1}), {1, 2, 3, 4, 0}),
         MkTensorNoInitList<float8_e4m3fn>(DT_FLOAT8_E4M3FN, TensorShape({0}),
                                           {})},
        {"DT_FLOAT8_E5M2",
         MkTensorNoInitList<float8_e5m2>(DT_FLOAT8_E5M2, TensorShape({5}),
                                         {1, 2, 3, 4, 0}),
         MkTensorNoInitList<float8_e5m2>(DT_FLOAT8_E5M2, TensorShape({2, 2}),
                                         {1, 2, 3, 4, 0}),
         MkTensorNoInitList<float8_e5m2>(
             DT_FLOAT8_E5M2, TensorShape({2, 2, 1, 1}), {1, 2, 3, 4, 0}),
         MkTensorNoInitList<float8_e5m2>(DT_FLOAT8_E5M2, TensorShape({0}), {})},
        {"DT_HALF",
         MkTensorNoInitList<Eigen::half>(DT_HALF, TensorShape({5}),
                                         {1, 2, 3, 4, 0}),
         MkTensorNoInitList<Eigen::half>(DT_HALF, TensorShape({2, 2}),
                                         {1, 2, 3, 4, 0}),
         MkTensorNoInitList<Eigen::half>(DT_HALF, TensorShape({2, 2, 1, 1}),
                                         {1, 2, 3, 4, 0}),
         MkTensorNoInitList<Eigen::half>(DT_HALF, TensorShape({0}), {})},
        {"DT_BFLOAT16",
         MkTensorNoInitList<bfloat16>(DT_BFLOAT16, TensorShape({5}),
                                      {1, 2, 3, 4, 0}),
         MkTensorNoInitList<bfloat16>(DT_BFLOAT16, TensorShape({2, 2}),
                                      {1, 2, 3, 4, 0}),
         MkTensorNoInitList<bfloat16>(DT_BFLOAT16, TensorShape({2, 2, 1, 1}),
                                      {1, 2, 3, 4, 0}),
         MkTensorNoInitList<bfloat16>(DT_BFLOAT16, TensorShape({0}), {})},
        {"DT_FLOAT",
         MkTensorNoInitList<float>(DT_FLOAT, TensorShape({5}), {1, 2, 3, 4, 0}),
         MkTensorNoInitList<float>(DT_FLOAT, TensorShape({2, 2}),
                                   {1, 2, 3, 4, 0}),
         MkTensorNoInitList<float>(DT_FLOAT, TensorShape({2, 2, 1, 1}),
                                   {1, 2, 3, 4, 0}),
         MkTensorNoInitList<float>(DT_FLOAT, TensorShape({0}), {})},
        {"DT_DOUBLE",
         MkTensorNoInitList<double>(DT_DOUBLE, TensorShape({5}),
                                    {1, 2, 3, 4, 0}),
         MkTensorNoInitList<double>(DT_DOUBLE, TensorShape({2, 2}),
                                    {1, 2, 3, 4, 0}),
         MkTensorNoInitList<double>(DT_DOUBLE, TensorShape({2, 2, 1, 1}),
                                    {1, 2, 3, 4, 0}),
         MkTensorNoInitList<double>(DT_DOUBLE, TensorShape({0}), {})},
    }),
    [](const ::testing::TestParamInfo<SimpleNum::ParamType>& info) {
      return info.param.test_name;
    });

TEST(SummarizeValue, INT32Dims) {
  Tensor x = MkTensor<int>(DT_INT32, TensorShape({3, 4}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  EXPECT_EQ("[1 2 3...]...", x.SummarizeValue(3));
  EXPECT_EQ("[1 2 3 4][5 6 7 8][9 10...]...", x.SummarizeValue(10));
}

TEST(SummarizeValue, BOOL) {
  Tensor x = MkTensor<bool>(DT_BOOL, TensorShape({5}), {false, true, true});
  EXPECT_EQ("0 1 1 0 1", x.SummarizeValue(16));
  EXPECT_EQ("0 1 1...", x.SummarizeValue(3));
}

TEST(SummarizeValue, STRING) {
  Tensor x = MkTensor<tstring>(DT_STRING, TensorShape({5}),
                               {"one", "two", "three", "four", "five"});
  EXPECT_EQ("one two three four five", x.SummarizeValue(16));
  x = MkTensor<tstring>(DT_STRING, TensorShape({5, 1, 5}),
                        {"one", "two", "three", "four", "five"});
  EXPECT_EQ("[[one two three four five]][[one...]]...", x.SummarizeValue(6));
}

TEST(SummarizeValue, INT32_PRINT_V2) {
  Tensor x = MkTensor<int>(DT_INT32, TensorShape({5}), {1, 2, 3, 4, 0});
  EXPECT_EQ("[1 2 3 4 0]", x.SummarizeValue(16, true));
  EXPECT_EQ("[1 2 3 4 0]", x.SummarizeValue(-1, true));
  EXPECT_EQ("[1 2 ... 4 0]", x.SummarizeValue(2, true));
  EXPECT_EQ("[1 ... 0]", x.SummarizeValue(1, true));
  x = MkTensor<int>(DT_INT32, TensorShape({2, 2}), {1, 2, 3, 4, 0});
  EXPECT_EQ("[[1 2]\n [3 4]]", x.SummarizeValue(16, true));
  x = MkTensor<int>(DT_INT32, TensorShape({2, 2, 1, 1}), {1, 2, 3, 4, 0});
  EXPECT_EQ("[[[[1]]\n\n  [[2]]]\n\n\n [[[3]]\n\n  [[4]]]]",
            x.SummarizeValue(16, true));
  x = MkTensor<int>(DT_INT32, TensorShape({0}), {});
  EXPECT_EQ("[]", x.SummarizeValue(16, true));
}

TEST(SummarizeValue, INT32Dims_PRINT_V2) {
  Tensor x = MkTensor<int>(DT_INT32, TensorShape({3, 4}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  EXPECT_EQ("[[1 ... 4]\n ...\n [9 ... 12]]", x.SummarizeValue(1, true));
  EXPECT_EQ("[[1 2 3 4]\n [5 6 7 8]\n [9 10 11 12]]",
            x.SummarizeValue(10, true));
  EXPECT_EQ("[[1 2 3 4]\n [5 6 7 8]\n [9 10 11 12]]",
            x.SummarizeValue(-1, true));
}

TEST(SummarizeValue, FLOAT_PRINT_V2) {
  Tensor x = MkTensor<float>(DT_FLOAT, TensorShape({5}), {1, 2, 3, 4, 0});
  EXPECT_EQ("[1 2 3 4 0]", x.SummarizeValue(16, true));
  EXPECT_EQ("[1 2 3 4 0]", x.SummarizeValue(-1, true));
  EXPECT_EQ("[1 2 ... 4 0]", x.SummarizeValue(2, true));
  EXPECT_EQ("[1 ... 0]", x.SummarizeValue(1, true));
  x = MkTensor<float>(DT_FLOAT, TensorShape({2, 2}), {1, 2, 3, 4, 0});
  EXPECT_EQ("[[1 2]\n [3 4]]", x.SummarizeValue(16, true));
  x = MkTensor<float>(DT_FLOAT, TensorShape({2, 2, 1, 1}), {1, 2, 3, 4, 0});
  EXPECT_EQ("[[[[1]]\n\n  [[2]]]\n\n\n [[[3]]\n\n  [[4]]]]",
            x.SummarizeValue(16, true));
  x = MkTensor<float>(DT_FLOAT, TensorShape({0}), {});
  EXPECT_EQ("[]", x.SummarizeValue(16, true));
}

TEST(SummarizeValue, BOOL_PRINT_V2) {
  Tensor x = MkTensor<bool>(DT_BOOL, TensorShape({5}), {false, true, true});
  EXPECT_EQ("[0 1 1 0 1]", x.SummarizeValue(16, true));
  EXPECT_EQ("[0 1 1 0 1]", x.SummarizeValue(-1, true));
  EXPECT_EQ("[0 1 ... 0 1]", x.SummarizeValue(2, true));
}

TEST(SummarizeValue, STRING_PRINT_V2) {
  Tensor x = MkTensor<tstring>(DT_STRING, TensorShape({5}),
                               {"one", "two", "three", "four", "five"});
  EXPECT_EQ("[\"one\" \"two\" \"three\" \"four\" \"five\"]",
            x.SummarizeValue(16, true));
  EXPECT_EQ("[\"one\" \"two\" \"three\" \"four\" \"five\"]",
            x.SummarizeValue(-1, true));
  EXPECT_EQ("[\"one\" \"two\" ... \"four\" \"five\"]",
            x.SummarizeValue(2, true));
  x = MkTensor<tstring>(DT_STRING, TensorShape({2, 2}),
                        {"one", "two", "three", "four", "five"});
  EXPECT_EQ("[[\"one\" \"two\"]\n [\"three\" \"four\"]]",
            x.SummarizeValue(16, true));
}

TEST(SummarizeValue, DT_RESOURCE) {
  Tensor t(DT_RESOURCE, TensorShape({}));
  ResourceHandle tmp;
  tmp.set_name("a");
  t.flat<ResourceHandle>()(0) = tmp;
  EXPECT_EQ(
      "<ResourceHandle(name=\"a\", device=\"\", container=\"\", "
      "type=\"\", dtype and shapes : \"[  ]\")>",
      t.SummarizeValue(-1, true));
}

TEST(SummarizeValue, DT_VARIANT) {
  Tensor t(DT_VARIANT, TensorShape({}));

  Tensor internal(DT_FLOAT, TensorShape({}));
  internal.flat<float>()(0) = 42.0f;
  t.flat<Variant>()(0) = internal;

  EXPECT_EQ("<Tensor<type: float shape: [] values: 42>>",
            t.SummarizeValue(-1, true));
}

void BM_CreateAndDestroy(::testing::benchmark::State& state) {
  TensorShape shape({10, 20});
  for (auto s : state) {
    Tensor t(DT_FLOAT, shape);
  }
}
BENCHMARK(BM_CreateAndDestroy);

void BM_Assign(::testing::benchmark::State& state) {
  Tensor a(DT_FLOAT, TensorShape({10, 20}));
  Tensor b(DT_FLOAT, TensorShape({10, 20}));
  bool a_to_b = true;
  for (auto s : state) {
    if (a_to_b) {
      b = a;
    } else {
      a = b;
    }
    a_to_b = !a_to_b;
  }
}
BENCHMARK(BM_Assign);

// Ensure tensor_data() works on empty tensors
TEST(Tensor, EmptyTensorData) {
  Tensor empty;
  EXPECT_EQ(empty.tensor_data().size(), 0);
}

// Benchmark create and destroy a tensor, with an allocated buffer.
void BM_CreateAndDestroyWithBuf(::testing::benchmark::State& state) {
  TensorShape shape({10, 20});
  Allocator* allocator = cpu_allocator();
  for (auto s : state) {
    Tensor a(allocator, DT_FLOAT, shape);
  }
}
BENCHMARK(BM_CreateAndDestroyWithBuf);

// Benchmark create+copy a tensor, with an allocated buffer.
void BM_CreateAndCopyCtrWithBuf(::testing::benchmark::State& state) {
  TensorShape shape({10, 20});
  Allocator* allocator = cpu_allocator();
  for (auto s : state) {
    Tensor a(allocator, DT_FLOAT, shape);
    Tensor b(a);
  }
}
BENCHMARK(BM_CreateAndCopyCtrWithBuf);

// Benchmark create+move a tensor, with an allocated buffer.
void BM_CreateAndMoveCtrWithBuf(::testing::benchmark::State& state) {
  TensorShape shape({10, 20});
  Allocator* allocator = cpu_allocator();
  for (auto s : state) {
    Tensor a(allocator, DT_FLOAT, shape);
    Tensor b(std::move(a));
  }
}
BENCHMARK(BM_CreateAndMoveCtrWithBuf);

// Benchmark creating and destroy a host-scalar tensor, using the allocator
// interface.
void BM_CreateAndDestroyHostScalarNonOptimized(
    ::testing::benchmark::State& state) {
  TensorShape shape({});
  Allocator* allocator = cpu_allocator();
  for (auto s : state) {
    Tensor a(allocator, DT_FLOAT, shape);
    a.scalar<float>()() = 37.0;
  }
}
BENCHMARK(BM_CreateAndDestroyHostScalarNonOptimized);

// Benchmark creating and destroy a host-scalar tensor, using the specialized
// constructor.
void BM_CreateAndDestroyHostScalarOptimized(
    ::testing::benchmark::State& state) {
  for (auto s : state) {
    Tensor a(37.0);
  }
}
BENCHMARK(BM_CreateAndDestroyHostScalarOptimized);

void BM_FromProto(::testing::benchmark::State& state) {
  const int size = state.range(0);

  TensorShape shape({size});
  Allocator* allocator = cpu_allocator();
  Tensor a(allocator, DT_FLOAT, shape);
  std::fill_n(a.flat<float>().data(), size, 42.0);
  TensorProto p;
  a.AsProtoField(&p);
  for (auto s : state) {
    Tensor b;
    ASSERT_TRUE(b.FromProto(p));
  }
}
BENCHMARK(BM_FromProto)->Range(1, 1 << 20);

void BM_FromProtoCompressed(::testing::benchmark::State& state) {
  const int size = state.range(0);

  TensorShape shape({size});
  Allocator* allocator = cpu_allocator();
  Tensor a(allocator, DT_FLOAT, shape);
  std::fill_n(a.flat<float>().data(), size, 42.0f);
  TensorProto p;
  a.AsProtoField(&p);
  tensor::CompressTensorProtoInPlace(&p);
  for (auto s : state) {
    Tensor b;
    ASSERT_TRUE(b.FromProto(p));
  }
}
BENCHMARK(BM_FromProtoCompressed)->Range(1, 1 << 20);

void BM_FromProtoCompressedZero(::testing::benchmark::State& state) {
  const int size = state.range(0);

  TensorShape shape({size});
  Allocator* allocator = cpu_allocator();
  Tensor a(allocator, DT_FLOAT, shape);
  std::fill_n(a.flat<float>().data(), size, 0);
  a.flat<float>()(0) = 1;
  TensorProto p;
  a.AsProtoField(&p);
  tensor::CompressTensorProtoInPlace(&p);
  for (auto s : state) {
    Tensor b;
    ASSERT_TRUE(b.FromProto(p));
  }
}
BENCHMARK(BM_FromProtoCompressedZero)->Range(1, 1 << 20);

}  // namespace
}  // namespace tensorflow
