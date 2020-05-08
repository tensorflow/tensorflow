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

#include "tensorflow/cc/experimental/base/public/tensor.h"

#include <stddef.h>

#include <cstdint>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Each of the following struct types have two members: a kDType that
// corresponds to a TF_Datatype enum value, and a typedef "type"
// of its corresponding C++ type. These types allow us to write Dtype-agnostic
// tests via GoogleTest's TypedTests:
// https://github.com/google/googletest/blob/e589a337170554c48bc658cc857cf15080c9eacc/googletest/docs/advanced.md#typed-tests
struct FloatType {
  using type = float;
  static constexpr TF_DataType kDType = TF_FLOAT;
};

struct DoubleType {
  using type = double;
  static constexpr TF_DataType kDType = TF_DOUBLE;
};

struct Int32Type {
  using type = int32_t;
  static constexpr TF_DataType kDType = TF_INT32;
};

struct UINT8Type {
  using type = uint8_t;
  static constexpr TF_DataType kDType = TF_UINT8;
};

struct INT8Type {
  using type = int8_t;
  static constexpr TF_DataType kDType = TF_INT8;
};

struct INT64Type {
  using type = int64_t;
  static constexpr TF_DataType kDType = TF_INT64;
};

struct UINT16Type {
  using type = uint16_t;
  static constexpr TF_DataType kDType = TF_UINT16;
};

struct UINT32Type {
  using type = uint32_t;
  static constexpr TF_DataType kDType = TF_UINT32;
};

struct UINT64Type {
  using type = uint64_t;
  static constexpr TF_DataType kDType = TF_UINT64;
};

using SimpleTypes =
    ::testing::Types<FloatType, DoubleType, Int32Type, UINT8Type, INT8Type,
                     INT64Type, UINT16Type, UINT32Type, UINT64Type>;

template <typename T>
class ConstructScalarTensorTest : public ::testing::Test {};
TYPED_TEST_SUITE(ConstructScalarTensorTest, SimpleTypes);

// This test constructs a scalar tensor for each of the types in "SimpleTypes",
// and verifies the expected dimensions, dtype, value, number of bytes, and
// number of elements.
TYPED_TEST(ConstructScalarTensorTest, ValidTensorAttributesAfterConstruction) {
  cc::Status status;
  TF_DataType dtype = TypeParam::kDType;
  typename TypeParam::type value = 42;
  cc::Tensor tensor =
      cc::Tensor::FromBuffer(/*dtype=*/dtype, /*shape=*/{},
                             /*data=*/&value,
                             /*len=*/sizeof(value),
                             /*deleter=*/[](void*, size_t) {}, &status);
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(tensor.dims(), 0);
  EXPECT_EQ(tensor.dtype(), dtype);
  EXPECT_EQ(*reinterpret_cast<typename TypeParam::type*>(tensor.data()), 42);
  EXPECT_EQ(tensor.num_bytes(), sizeof(typename TypeParam::type));
  EXPECT_EQ(tensor.num_elements(), 1);
}

template <typename T>
class Construct1DTensorTest : public ::testing::Test {};
TYPED_TEST_SUITE(Construct1DTensorTest, SimpleTypes);

// This test constructs a 1D tensor for each of the types in "SimpleTypes",
// and verifies the expected dimensions, dtype, value, number of bytes, and
// number of elements.
TYPED_TEST(Construct1DTensorTest, ValidTensorAttributesAfterConstruction) {
  cc::Status status;
  TF_DataType dtype = TypeParam::kDType;
  // This is our 1D tensor of varying dtype.
  std::vector<typename TypeParam::type> value = {42, 100, 0, 1, 4, 29};
  // Shape is Rank 1 vector.
  std::vector<int64_t> shape;
  shape.push_back(value.size());

  cc::Tensor tensor = cc::Tensor::FromBuffer(
      /*dtype=*/dtype, /*shape=*/shape,
      /*data=*/value.data(),
      /*len=*/value.size() * sizeof(typename TypeParam::type),
      /*deleter=*/[](void*, size_t) {}, &status);
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(tensor.dims(), 1);
  EXPECT_EQ(tensor.dtype(), dtype);
  gtl::ArraySlice<typename TypeParam::type> tensor_view(
      reinterpret_cast<typename TypeParam::type*>(tensor.data()), value.size());
  EXPECT_EQ(tensor_view[0], 42);
  EXPECT_EQ(tensor_view[1], 100);
  EXPECT_EQ(tensor_view[2], 0);
  EXPECT_EQ(tensor_view[3], 1);
  EXPECT_EQ(tensor_view[4], 4);
  EXPECT_EQ(tensor_view[5], 29);

  EXPECT_EQ(tensor.num_bytes(),
            value.size() * sizeof(typename TypeParam::type));
  EXPECT_EQ(tensor.num_elements(), value.size());
}

template <typename T>
class Construct2DTensorTest : public ::testing::Test {};
TYPED_TEST_SUITE(Construct2DTensorTest, SimpleTypes);

// This test constructs a 2D tensor for each of the types in "SimpleTypes",
// and verifies the expected dimensions, dtype, value, number of bytes, and
// number of elements.
TYPED_TEST(Construct2DTensorTest, ValidTensorAttributesAfterConstruction) {
  cc::Status status;
  TF_DataType dtype = TypeParam::kDType;
  // This is our 1D tensor of varying dtype.
  std::vector<typename TypeParam::type> value = {42, 100, 0, 1, 4, 29};
  // Shape is Rank 2 vector with shape 2 x 3.
  std::vector<int64_t> shape({2, 3});

  cc::Tensor tensor = cc::Tensor::FromBuffer(
      /*dtype=*/dtype, /*shape=*/shape,
      /*data=*/value.data(),
      /*len=*/value.size() * sizeof(typename TypeParam::type),
      /*deleter=*/[](void*, size_t) {}, &status);

  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(tensor.dims(), 2);
  EXPECT_EQ(tensor.dtype(), dtype);
  gtl::ArraySlice<typename TypeParam::type> tensor_view(
      reinterpret_cast<typename TypeParam::type*>(tensor.data()), value.size());
  EXPECT_EQ(tensor_view[0], 42);
  EXPECT_EQ(tensor_view[1], 100);
  EXPECT_EQ(tensor_view[2], 0);
  EXPECT_EQ(tensor_view[3], 1);
  EXPECT_EQ(tensor_view[4], 4);
  EXPECT_EQ(tensor_view[5], 29);

  EXPECT_EQ(tensor.num_bytes(),
            value.size() * sizeof(typename TypeParam::type));
  EXPECT_EQ(tensor.num_elements(), value.size());
}

TEST(CPPTensorAPI, ConstructTensorFromBuffer) {
  bool done = false;
  cc::Status status;
  std::vector<int32_t> data_vector({12, 14, 20, 18, 39, 42, 100});
  {
    // data_vector is a rank 1 tensor.
    std::vector<int64_t> shape;
    shape.push_back(data_vector.size());

    cc::Tensor::DeleterCallback callback = [&done](void* data, size_t len) {
      done = true;
    };

    cc::Tensor tensor =
        cc::Tensor::FromBuffer(/*dtype=*/TF_INT32, /*shape=*/shape,
                               /*data=*/data_vector.data(),
                               /*len=*/data_vector.size() * sizeof(int32_t),
                               /*deleter=*/callback, &status);
    ASSERT_TRUE(status.ok()) << status.message();
  }
  // At this point, tensor has been destroyed, and the deleter callback should
  // have run.
  EXPECT_TRUE(done);
}

}  // namespace
}  // namespace tensorflow
