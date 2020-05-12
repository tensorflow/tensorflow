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

#include "tensorflow/cc/experimental/base/public/tensorhandle.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/cc/experimental/base/public/runtime.h"
#include "tensorflow/cc/experimental/base/public/runtime_builder.h"
#include "tensorflow/cc/experimental/base/public/tensor.h"
#include "tensorflow/cc/experimental/base/tests/tensor_types_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using tensorflow::experimental::cc::Runtime;
using tensorflow::experimental::cc::RuntimeBuilder;
using tensorflow::experimental::cc::Status;
using tensorflow::experimental::cc::Tensor;
using tensorflow::experimental::cc::TensorHandle;

using SimpleTypes = ::testing::Types<
    tensorflow::FloatType, tensorflow::DoubleType, tensorflow::Int32Type,
    tensorflow::UINT8Type, tensorflow::INT8Type, tensorflow::INT64Type,
    tensorflow::UINT16Type, tensorflow::UINT32Type, tensorflow::UINT64Type>;

template <typename T>
class ConstructScalarTensorHandleTest : public ::testing::Test {};
TYPED_TEST_SUITE(ConstructScalarTensorHandleTest, SimpleTypes);

// This test constructs a scalar tensor for each of the types in "SimpleTypes",
// then wraps it in a TensorHandle. We then unwrap it back into a Tensor, and
// verify the expected dims, dtype, value, num bytes, and num elements.
TYPED_TEST(ConstructScalarTensorHandleTest,
           ValidTensorAttributesAfterConstruction) {
  Status status;
  RuntimeBuilder runtime_builder;
  std::unique_ptr<Runtime> runtime = runtime_builder.Build(&status);
  ASSERT_TRUE(status.ok()) << status.message();

  TF_DataType dtype = TypeParam::kDType;
  typename TypeParam::type value = 42;
  Tensor original_tensor =
      Tensor::FromBuffer(/*dtype=*/dtype, /*shape=*/{},
                         /*data=*/&value,
                         /*len=*/sizeof(value),
                         /*deleter=*/[](void*, size_t) {}, &status);
  ASSERT_TRUE(status.ok()) << status.message();

  TensorHandle handle =
      TensorHandle::FromTensor(original_tensor, *runtime, &status);
  ASSERT_TRUE(status.ok()) << status.message();

  Tensor tensor = handle.Resolve(&status);
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(tensor.dims(), 0);
  EXPECT_EQ(tensor.dtype(), dtype);
  EXPECT_EQ(*reinterpret_cast<typename TypeParam::type*>(tensor.data()), 42);
  EXPECT_EQ(tensor.num_bytes(), sizeof(typename TypeParam::type));
  EXPECT_EQ(tensor.num_elements(), 1);
}

template <typename T>
class Construct1DTensorHandleTest : public ::testing::Test {};
TYPED_TEST_SUITE(Construct1DTensorHandleTest, SimpleTypes);

// This test constructs a 1D tensor for each of the types in "SimpleTypes",
// and verifies the expected dimensions, dtype, value, number of bytes, and
// number of elements.
TYPED_TEST(Construct1DTensorHandleTest,
           ValidTensorAttributesAfterConstruction) {
  Status status;
  RuntimeBuilder runtime_builder;
  std::unique_ptr<Runtime> runtime = runtime_builder.Build(&status);
  ASSERT_TRUE(status.ok()) << status.message();

  TF_DataType dtype = TypeParam::kDType;
  // This is our 1D tensor of varying dtype.
  std::vector<typename TypeParam::type> value = {42, 100, 0, 1, 4, 29};
  // Shape is Rank 1 vector.
  std::vector<int64_t> shape;
  shape.push_back(value.size());

  Tensor original_tensor = Tensor::FromBuffer(
      /*dtype=*/dtype, /*shape=*/shape,
      /*data=*/value.data(),
      /*len=*/value.size() * sizeof(typename TypeParam::type),
      /*deleter=*/[](void*, size_t) {}, &status);
  ASSERT_TRUE(status.ok()) << status.message();

  TensorHandle handle =
      TensorHandle::FromTensor(original_tensor, *runtime, &status);
  ASSERT_TRUE(status.ok()) << status.message();

  Tensor tensor = handle.Resolve(&status);
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
class Construct2DTensorHandleTest : public ::testing::Test {};
TYPED_TEST_SUITE(Construct2DTensorHandleTest, SimpleTypes);

// This test constructs a 2D tensor for each of the types in "SimpleTypes",
// and verifies the expected dimensions, dtype, value, number of bytes, and
// number of elements.
TYPED_TEST(Construct2DTensorHandleTest,
           ValidTensorAttributesAfterConstruction) {
  Status status;
  RuntimeBuilder runtime_builder;
  std::unique_ptr<Runtime> runtime = runtime_builder.Build(&status);
  ASSERT_TRUE(status.ok()) << status.message();

  TF_DataType dtype = TypeParam::kDType;
  // This is our 1D tensor of varying dtype.
  std::vector<typename TypeParam::type> value = {42, 100, 0, 1, 4, 29};
  // Shape is Rank 2 vector with shape 2 x 3.
  std::vector<int64_t> shape({2, 3});

  Tensor original_tensor = Tensor::FromBuffer(
      /*dtype=*/dtype, /*shape=*/shape,
      /*data=*/value.data(),
      /*len=*/value.size() * sizeof(typename TypeParam::type),
      /*deleter=*/[](void*, size_t) {}, &status);
  ASSERT_TRUE(status.ok()) << status.message();

  TensorHandle handle =
      TensorHandle::FromTensor(original_tensor, *runtime, &status);
  ASSERT_TRUE(status.ok()) << status.message();

  Tensor tensor = handle.Resolve(&status);
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

}  // namespace
}  // namespace tensorflow
