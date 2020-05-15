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
#include <stdint.h>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/cc/experimental/base/tests/tensor_types_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"

namespace {

using tensorflow::experimental::cc::Status;
using tensorflow::experimental::cc::Tensor;

using SimpleTypes = ::testing::Types<
    tensorflow::FloatType, tensorflow::DoubleType, tensorflow::Int32Type,
    tensorflow::UINT8Type, tensorflow::INT8Type, tensorflow::INT64Type,
    tensorflow::UINT16Type, tensorflow::UINT32Type, tensorflow::UINT64Type>;

template <typename T>
class ConstructScalarTensorTest : public ::testing::Test {};
TYPED_TEST_SUITE(ConstructScalarTensorTest, SimpleTypes);

// This test constructs a scalar tensor for each of the types in "SimpleTypes",
// and verifies the expected dimensions, dtype, value, number of bytes, and
// number of elements.
TYPED_TEST(ConstructScalarTensorTest, ValidTensorAttributesAfterConstruction) {
  Status status;
  TF_DataType dtype = TypeParam::kDType;
  typename TypeParam::type value = 42;
  Tensor tensor = Tensor::FromBuffer(/*dtype=*/dtype, /*shape=*/{},
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
  Status status;
  TF_DataType dtype = TypeParam::kDType;
  // This is our 1D tensor of varying dtype.
  std::vector<typename TypeParam::type> value = {42, 100, 0, 1, 4, 29};
  // Shape is Rank 1 vector.
  std::vector<int64_t> shape;
  shape.push_back(value.size());

  Tensor tensor = Tensor::FromBuffer(
      /*dtype=*/dtype, /*shape=*/shape,
      /*data=*/value.data(),
      /*len=*/value.size() * sizeof(typename TypeParam::type),
      /*deleter=*/[](void*, size_t) {}, &status);
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(tensor.dims(), 1);
  EXPECT_EQ(tensor.dtype(), dtype);
  tensorflow::gtl::ArraySlice<typename TypeParam::type> tensor_view(
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
  Status status;
  TF_DataType dtype = TypeParam::kDType;
  // This is our 1D tensor of varying dtype.
  std::vector<typename TypeParam::type> value = {42, 100, 0, 1, 4, 29};
  // Shape is Rank 2 vector with shape 2 x 3.
  std::vector<int64_t> shape({2, 3});

  Tensor tensor = Tensor::FromBuffer(
      /*dtype=*/dtype, /*shape=*/shape,
      /*data=*/value.data(),
      /*len=*/value.size() * sizeof(typename TypeParam::type),
      /*deleter=*/[](void*, size_t) {}, &status);

  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(tensor.dims(), 2);
  EXPECT_EQ(tensor.dtype(), dtype);
  tensorflow::gtl::ArraySlice<typename TypeParam::type> tensor_view(
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
  Status status;
  std::vector<int32_t> data_vector({12, 14, 20, 18, 39, 42, 100});
  {
    // data_vector is a rank 1 tensor.
    std::vector<int64_t> shape;
    shape.push_back(data_vector.size());

    Tensor::DeleterCallback callback = [&done](void* data, size_t len) {
      done = true;
    };

    Tensor tensor =
        Tensor::FromBuffer(/*dtype=*/TF_INT32, /*shape=*/shape,
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
