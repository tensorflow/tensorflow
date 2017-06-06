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

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/strcat.h"
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
inline bool operator==(const ResourceHandle& a, const ResourceHandle& b) {
  return a.device() == b.device() && a.container() == b.container() &&
         a.name() == b.name() && a.hash_code() == b.hash_code() &&
         a.maybe_type_name() == b.maybe_type_name();
}

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
  EXPECT_TRUE(std::is_trivial<int64>::value);
  EXPECT_TRUE(std::is_trivial<bool>::value);
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
void TestCopies(const Tensor& t) {
  {
    LOG(INFO) << "CopyFrom()";
    Tensor t2(t.dtype());
    EXPECT_TRUE(t2.CopyFrom(t, t.shape()));
    test::ExpectTensorEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "operator=()";
    Tensor t2(t.dtype());
    t2 = t;
    test::ExpectTensorEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "deep copy";
    Tensor t2(t.dtype(), t.shape());
    t2.flat<T>() = t.flat<T>();
    test::ExpectTensorEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "AsProtoField()";
    TensorProto proto;
    t.AsProtoField(&proto);
    Tensor t2(t.dtype());
    EXPECT_TRUE(t2.FromProto(proto));
    test::ExpectTensorEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "AsProtoTensorContent()";
    TensorProto proto;
    t.AsProtoTensorContent(&proto);
    Tensor t2(t.dtype());
    EXPECT_TRUE(t2.FromProto(proto));
    test::ExpectTensorEqual<T>(t, t2);
    // Make another copy via tensor_content field.
    *proto.mutable_tensor_content() = proto.tensor_content();
    Tensor t3(t.dtype());
    EXPECT_TRUE(t3.FromProto(proto));
    test::ExpectTensorEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "AsTensor";
    gtl::ArraySlice<T> values(t.flat<T>().data(), t.NumElements());
    Tensor t2 = test::AsTensor(values, t.shape());
    test::ExpectTensorEqual<T>(t, t2);
  }
  {
    LOG(INFO) << "Move constructor";
    Tensor t2 = t;
    Tensor t3(std::move(t2));
    test::ExpectTensorEqual<T>(t, t3);
    EXPECT_TRUE(t3.IsInitialized());
    EXPECT_FALSE(t2.IsInitialized());
  }
  {
    LOG(INFO) << "Move assignment";
    Tensor t2 = t;
    Tensor t3 = std::move(t2);
    Tensor* t4 = &t3;
    *t4 = std::move(t3);
    test::ExpectTensorEqual<T>(t, t3);
    EXPECT_TRUE(t3.IsInitialized());
    EXPECT_FALSE(t2.IsInitialized());
  }
}

TEST(Tensor_Float, Simple) {
  Tensor t(DT_FLOAT, TensorShape({10, 20}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({10, 20})));
  for (int64 a = 0; a < t.shape().dim_size(0); a++) {
    for (int64 b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<float>()(a, b) = static_cast<float>(a * b);
    }
  }
  TestCopies<float>(t);
}

TEST(Tensor_ResourceHandle, Simple) {
  Tensor t(DT_RESOURCE, TensorShape({}));
  ResourceHandle tmp;
  tmp.set_name("a");
  t.flat<ResourceHandle>()(0) = tmp;
  TestCopies<ResourceHandle>(t);
}

TEST(Tensor_UInt16, Simple) {
  Tensor t(DT_UINT16, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64 a = 0; a < t.shape().dim_size(0); a++) {
    for (int64 b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<uint16>()(a, b) = uint16(a * b);
    }
  }
  TestCopies<uint16>(t);
}

TEST(Tensor_QInt8, Simple) {
  Tensor t(DT_QINT8, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64 a = 0; a < t.shape().dim_size(0); a++) {
    for (int64 b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<qint8>()(a, b) = qint8(a * b);
    }
  }
  TestCopies<qint8>(t);
}

TEST(Tensor_QUInt8, Simple) {
  Tensor t(DT_QUINT8, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64 a = 0; a < t.shape().dim_size(0); a++) {
    for (int64 b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<Eigen::QUInt8>()(a, b) = Eigen::QUInt8(a * b);
    }
  }
  TestCopies<Eigen::QUInt8>(t);
}

TEST(Tensor_QInt32, Simple) {
  Tensor t(DT_QINT32, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64 a = 0; a < t.shape().dim_size(0); a++) {
    for (int64 b = 0; b < t.shape().dim_size(1); b++) {
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
};

TEST_F(TensorReshapeTest, Reshape) {
  LOG(INFO) << "shaped";
  {
    auto shaped = t.shaped<float, 1>({120});
    EXPECT_EQ(120, shaped.dimension(0));
    EXPECT_EQ(shaped(0), 0.01f);
    EXPECT_EQ(shaped(119), 0.02f);
  }
  {
    auto shaped = t.shaped<float, 2>({6, 20});
    EXPECT_EQ(6, shaped.dimension(0));
    EXPECT_EQ(20, shaped.dimension(1));
    EXPECT_EQ(shaped(0, 0), 0.01f);
    EXPECT_EQ(shaped(5, 19), 0.02f);
  }
  {
    auto shaped = t.shaped<float, 3>({6, 4, 5});
    EXPECT_EQ(6, shaped.dimension(0));
    EXPECT_EQ(4, shaped.dimension(1));
    EXPECT_EQ(5, shaped.dimension(2));
    EXPECT_EQ(shaped(0, 0, 0), 0.01f);
    EXPECT_EQ(shaped(5, 3, 4), 0.02f);
  }
  {
    auto shaped = t.shaped<float, 4>({2, 3, 4, 5});
    EXPECT_EQ(2, shaped.dimension(0));
    EXPECT_EQ(3, shaped.dimension(1));
    EXPECT_EQ(4, shaped.dimension(2));
    EXPECT_EQ(5, shaped.dimension(3));

    EXPECT_EQ(shaped(0, 0, 0, 0), 0.01f);
    EXPECT_EQ(shaped(1, 2, 3, 4), 0.02f);
  }
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
    auto Tt = t.scalar<string>();
    EXPECT_EQ(1, Tt.size());
    EXPECT_EQ(0, Tt.rank());
    t.scalar<string>()() = "foo";
    EXPECT_EQ("foo", Tt());
  }
  {
    Tensor t(DT_STRING, TensorShape({1}));
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.vec<string>();
    EXPECT_EQ(1, Tt.size());
    t.flat<string>()(0) = "foo";
    EXPECT_EQ("foo", Tt(0));
  }
  {
    Tensor t(DT_STRING, TensorShape({1, 1, 1}));
    EXPECT_EQ(1, t.NumElements());
    auto Tt = t.scalar<string>();
    EXPECT_EQ(1, Tt.size());
    EXPECT_EQ(0, Tt.rank());
    t.flat<string>()(0) = "bar";
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

TEST(Tensor_String, Simple) {
  Tensor t = test::AsTensor<string>(
      {"hello", "world", "machine", "learning", "new", "york"},
      TensorShape({3, 2}));
  auto s = t.shape();
  ASSERT_EQ(s.dims(), 2);
  ASSERT_EQ(s.dim_size(0), 3);
  ASSERT_EQ(s.dim_size(1), 2);
  auto m = t.matrix<string>();
  EXPECT_EQ(t.TotalBytes(), 3 * 2 * sizeof(string) + 5 + 5 + 7 + 8 + 3 + 4);

  EXPECT_EQ(m(0, 0), "hello");
  EXPECT_EQ(m(0, 1), "world");
  EXPECT_EQ(m(1, 0), "machine");
  EXPECT_EQ(m(1, 1), "learning");
  EXPECT_EQ(m(2, 0), "new");
  EXPECT_EQ(m(2, 1), "york");

  TestCopies<string>(t);
}

TEST(Tensor_Float, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<float>({0, 1, 2, 3, 4, 5}, {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<float>() = t1.flat<float>() * 2.0f;
  Tensor t3 = test::AsTensor<float>({0, 2, 4, 6, 8, 10}, t1.shape());
  test::ExpectTensorEqual<float>(t2, t3);
}

TEST(Tensor_Int32, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<int32>({0, 1, 2, 3, 4, 5}, {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<int32>() = t1.flat<int32>() * 2;
  Tensor t3 = test::AsTensor<int32>({0, 2, 4, 6, 8, 10}, t1.shape());
  test::ExpectTensorEqual<int32>(t2, t3);
}

TEST(Tensor_UInt16, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<uint16>({0, 1, 2, 3, 4, 5}, {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<uint16>() = t1.flat<uint16>() * uint16(2);
  Tensor t3 = test::AsTensor<uint16>({0, 2, 4, 6, 8, 10}, t1.shape());
  test::ExpectTensorEqual<uint16>(t2, t3);
}

TEST(Tensor_QInt8, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<qint8>({0, 1, 2, 3, 4, 5}, {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<qint8>() = t1.flat<qint8>() + qint8(-2);
  Tensor t3 = test::AsTensor<qint8>({-2, -1, 0, 1, 2, 3}, {2, 3});
  test::ExpectTensorEqual<qint8>(t2, t3);
}

TEST(Tensor_QUInt8, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<quint8>({0, 1, 2, 3, 4, 5}, {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<quint8>() = t1.flat<quint8>() + quint8(2);
  Tensor t3 = test::AsTensor<quint8>({2, 3, 4, 5, 6, 7}, {2, 3});
  test::ExpectTensorEqual<quint8>(t2, t3);
}

TEST(Tensor_Int64, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<int64>(
      {0LL << 48, 1LL << 48, 2LL << 48, 3LL << 48, 4LL << 48, 5LL << 48},
      {2, 3});
  Tensor t2(t1.dtype(), t1.shape());
  t2.flat<int64>() = t1.flat<int64>() * static_cast<int64>(2);
  Tensor t3 = test::AsTensor<int64>(
      {0LL << 48, 2LL << 48, 4LL << 48, 6LL << 48, 8LL << 48, 10LL << 48},
      {2, 3});
  test::ExpectTensorEqual<int64>(t2, t3);
}

TEST(Tensor_String, SimpleWithHelper) {
  Tensor t1 = test::AsTensor<string>({"0", "1", "2", "3", "4", "5"}, {2, 3});
  Tensor t2(DT_STRING, {2, 3});
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      t2.matrix<string>()(i, j) = strings::StrCat(i * 3 + j);
    }
  }

  // Test with helper.
  test::ExpectTensorEqual<string>(t1, t2);
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
  test::ExpectTensorEqual<bool>(t1, t2);
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
    test::ExpectTensorEqual<complex64>(t2, t3);
  }

  // Does some numeric operations for complex64 numbers.
  {
    const float PI = std::acos(-1);
    const complex64 rotate_45 = std::polar(1.0f, PI / 4);

    // x contains all the 8-th root of unity.
    Tensor x(DT_COMPLEX64, TensorShape({8}));
    for (int i = 0; i < 8; ++i) {
      x.vec<complex64>()(i) = std::pow(rotate_45, i);
    }

    // Shift the roots by 45 degree.
    Tensor y(DT_COMPLEX64, TensorShape({8}));
    y.vec<complex64>() = x.vec<complex64>() * rotate_45;
    Tensor y_expected(DT_COMPLEX64, TensorShape({8}));
    for (int i = 0; i < 8; ++i) {
      y_expected.vec<complex64>()(i) = std::pow(rotate_45, i + 1);
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
    test::ExpectTensorEqual<complex128>(t2, t3);
  }

  // Does some numeric operations for complex128 numbers.
  {
    const double PI = std::acos(-1);
    const complex128 rotate_45 = std::polar(1.0, PI / 4);

    // x contains all the 8-th root of unity.
    Tensor x(DT_COMPLEX128, TensorShape({8}));
    for (int i = 0; i < 8; ++i) {
      x.vec<complex128>()(i) = std::pow(rotate_45, i);
    }

    // Shift the roots by 45 degree.
    Tensor y(DT_COMPLEX128, TensorShape({8}));
    y.vec<complex128>() = x.vec<complex128>() * rotate_45;
    Tensor y_expected(DT_COMPLEX128, TensorShape({8}));
    for (int i = 0; i < 8; ++i) {
      y_expected.vec<complex128>()(i) = std::pow(rotate_45, i + 1);
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

namespace {

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
    t.vec<string>()(0) = "foo";
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
// As of 2015/8, tensorflow::Tensor allocates its buffer with 32-byte
// alignment. Tensor::tensor/flat/vec/matrix methods requires the
// buffer satisfies Eigen::Aligned (e.g., 16-bytes aligned usually,
// and 32-bytes for AVX). Tensor::Slice requires the caller to ensure
// its result is aligned if the caller intends to use those methods.
// In this test case, we simply make sure each slice is 32-byte
// aligned: sizeof(float) * 4 * 2 = 32.
TEST(Tensor, Slice_Basic) {
  Tensor saved;
  {  // General
    Tensor x(DT_FLOAT, TensorShape({10, 4, 34}));
    // Fills in known values.
    for (int i = 0; i < 10; ++i) {
      x.Slice(i, i + 1).flat<float>().setConstant(i * 1.f);
    }
    // A simple slice along dim0.
    Tensor y = x.Slice(4, 8);
    EXPECT_TRUE(y.shape().IsSameSize(TensorShape({4, 4, 34})));
    auto tx = x.tensor<float, 3>();
    auto ty = y.tensor<float, 3>();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 34; ++k) {
          EXPECT_EQ(ty(i, j, k), 4.0 + i);
          EXPECT_EQ(&tx(4 + i, j, k), &ty(i, j, k));
        }
      }
    }
    // A simple slice equivalent to identity.
    TestCopies<float>(y);
    y = x.Slice(0, 10);
    test::ExpectTensorEqual<float>(x, y);
    EXPECT_EQ(x.flat<float>().data(), y.flat<float>().data());

    // A slice of a slice.
    auto z = x.Slice(4, 8).Slice(2, 3);
    auto tz = z.tensor<float, 3>();
    EXPECT_EQ(1, z.dim_size(0));
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 34; ++k) {
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
      for (int k = 0; k < 34; ++k) {
        EXPECT_EQ(tsaved(0, j, k), 6.0);
      }
    }
  }
  {  // Empty
    Tensor x(DT_FLOAT, TensorShape({10, 0, 34}));
    x.flat<float>().setRandom();
    Tensor y = x.Slice(4, 8);
    EXPECT_TRUE(y.shape().IsSameSize(TensorShape({4, 0, 34})));
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
    for (int64 i = 0; i < y.NumElements(); ++i) {
      EXPECT_EQ(1.0, y.unaligned_flat<float>()(i));
    }
  }
}

namespace {
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
}  // namespace

TEST(SummarizeValue, Uninitialized) {
  Tensor x(DT_INT32);
  TensorTestHelper::set_shape(&x, TensorShape({4, 4}));
  EXPECT_EQ(
      strings::StrCat("uninitialized Tensor of 16 elements of type ", DT_INT32),
      x.SummarizeValue(16));
}

TEST(SummarizeValue, INT32) {
  Tensor x = MkTensor<int>(DT_INT32, TensorShape({5}), {1, 2, 3, 4, 0});
  EXPECT_EQ("1 2 3 4 0", x.SummarizeValue(16));
  x = MkTensor<int>(DT_INT32, TensorShape({2, 2}), {1, 2, 3, 4, 0});
  EXPECT_EQ("[1 2][3 4]", x.SummarizeValue(16));
  x = MkTensor<int>(DT_INT32, TensorShape({2, 2, 1, 1}), {1, 2, 3, 4, 0});
  EXPECT_EQ("[[[1]][[2]]][[[3]][[4]]]", x.SummarizeValue(16));
  EXPECT_EQ("[[[1]][[2]]][[[3]]]...", x.SummarizeValue(3));
  x = MkTensor<int>(DT_INT32, TensorShape({0}), {});
  EXPECT_EQ("", x.SummarizeValue(16));
}

TEST(SummarizeValue, FLOAT) {
  Tensor x = MkTensor<float>(DT_FLOAT, TensorShape({5}), {1, 2, 3, 4, 0});
  EXPECT_EQ("1 2 3 4 0", x.SummarizeValue(16));
  x = MkTensor<float>(DT_FLOAT, TensorShape({2, 2}), {1, 2, 3, 4, 0});
  EXPECT_EQ("[1 2][3 4]", x.SummarizeValue(16));
  x = MkTensor<float>(DT_FLOAT, TensorShape({2, 2, 1, 1}), {1, 2, 3, 4, 0});
  EXPECT_EQ("[[[1]][[2]]][[[3]][[4]]]", x.SummarizeValue(16));
  EXPECT_EQ("[[[1]][[2]]][[[3]]]...", x.SummarizeValue(3));
  x = MkTensor<float>(DT_FLOAT, TensorShape({0}), {});
  EXPECT_EQ("", x.SummarizeValue(16));
}

TEST(SummarizeValue, BOOL) {
  Tensor x = MkTensor<bool>(DT_BOOL, TensorShape({5}), {false, true, true});
  EXPECT_EQ("0 1 1 0 1", x.SummarizeValue(16));
  EXPECT_EQ("0 1 1...", x.SummarizeValue(3));
}

TEST(SummarizeValue, STRING) {
  Tensor x = MkTensor<string>(DT_STRING, TensorShape({5}),
                              {"one", "two", "three", "four", "five"});
  EXPECT_EQ("one two three four five", x.SummarizeValue(16));
  x = MkTensor<string>(DT_STRING, TensorShape({5, 1, 5}),
                       {"one", "two", "three", "four", "five"});
  EXPECT_EQ("one two three four five one...", x.SummarizeValue(6));
}

static void BM_CreateAndDestroy(int iters) {
  TensorShape shape({10, 20});
  while (--iters) {
    Tensor t(DT_FLOAT, shape);
  }
}
BENCHMARK(BM_CreateAndDestroy);

static void BM_Assign(int iters) {
  Tensor a(DT_FLOAT, TensorShape({10, 20}));
  Tensor b(DT_FLOAT, TensorShape({10, 20}));
  bool a_to_b = true;
  while (--iters) {
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
static void BM_CreateAndDestroyWithBuf(int iters) {
  TensorShape shape({10, 20});
  Allocator* allocator = cpu_allocator();
  while (--iters) {
    Tensor a(allocator, DT_FLOAT, shape);
  }
}
BENCHMARK(BM_CreateAndDestroyWithBuf);

// Benchmark create+copy a tensor, with an allocated buffer.
static void BM_CreateAndCopyCtrWithBuf(int iters) {
  TensorShape shape({10, 20});
  Allocator* allocator = cpu_allocator();
  while (--iters) {
    Tensor a(allocator, DT_FLOAT, shape);
    Tensor b(a);
  }
}
BENCHMARK(BM_CreateAndCopyCtrWithBuf);

// Benchmark create+move a tensor, with an allocated buffer.
static void BM_CreateAndMoveCtrWithBuf(int iters) {
  TensorShape shape({10, 20});
  Allocator* allocator = cpu_allocator();
  while (--iters) {
    Tensor a(allocator, DT_FLOAT, shape);
    Tensor b(std::move(a));
  }
}
BENCHMARK(BM_CreateAndMoveCtrWithBuf);

}  // namespace
}  // namespace tensorflow
